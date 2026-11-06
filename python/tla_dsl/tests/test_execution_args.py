"""Smoke tests for ``ExecutionArgs`` (layout packing details live in hivm tests)."""

from __future__ import annotations

import inspect
import struct
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ForwardRef

import pytest

pytest.importorskip("catlass", exc_type=ImportError)

import catlass.tla as tla
from catlass import compiler_bridge, execution
from catlass._mlir import ir as mlir_ir
from catlass.base_dsl.jit_executor import ExecutionArgs
from catlass.base_dsl import BaseDSL
from catlass.base_dsl.runtime.argument_tree import (
    build_runtime_argument_tree,
    flatten_runtime_leaves,
)
from catlass.tla.runtime import make_fake_tensor


@pytest.mark.parametrize(
    "container", ["direct", "tuple", "list", "dataclass"]
)
@pytest.mark.parametrize("binding,pointer", [(False, 0x1000), (True, 0)])
def test_launch_rejects_unbound_tensor_in_each_container(
    monkeypatch, container, binding, pointer
):
    from catlass.base_dsl.jit_executor import JitExecutor
    from catlass.base_dsl.runtime import ascend_stream_adapter
    from catlass.base_dsl.runtime.argument_tree import (
        ArgumentTreeError,
        build_runtime_argument_tree,
    )
    from catlass.tla.runtime import make_fake_tensor

    @dataclass
    class Aux:
        tensor: tla.Tensor

    tensor = make_fake_tensor(
        tla.Float32, (8,), (1,), origin_shape=(8,), layout_tag=tla.arch.RowMajor
    )
    tensor._external_binding = binding
    tensor.data_ptr = pointer
    value = {
        "direct": lambda: tensor,
        "tuple": lambda: (tensor,),
        "list": lambda: [tensor],
        "dataclass": lambda: Aux(tensor),
    }[container]()
    layout = compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="kernel",
        total_size=8,
        arguments=(
            compiler_bridge.KernelAbiArgument(
                index=0,
                kind=compiler_bridge.KernelAbiArgumentKind.POINTER,
                scalar=None,
                mlir_type="!llvm.ptr",
                offset=0,
                storage_size=8,
                alignment=4,
            ),
        ),
    )
    executor = object.__new__(JitExecutor)
    executor.device = executor.function_handle = executor.binary_handle = 0
    executor.jit_module = SimpleNamespace(
        execution_args=ExecutionArgs(
            kernel_abi=layout, argument_trees=(build_runtime_argument_tree(value),)
        ),
        uses_scalar_print=False,
        simt_print_formats=None,
        uses_tensor_print=False,
        print_argument_layout=(),
        print_metadata=None,
        print_tensor_position=None,
        is_mixed=False,
        ub_dynamic_base=-1,
    )
    launched = []
    checks = []
    require_bound = type(tensor)._require_bound

    def check_bound(value):
        checks.append(value)
        require_bound(value)

    monkeypatch.setattr(type(tensor), "_require_bound", check_bound)
    monkeypatch.setattr(ascend_stream_adapter, "current_device", lambda: 0)
    monkeypatch.setattr(ascend_stream_adapter, "as_stream", lambda *a, **kw: 0)
    monkeypatch.setattr(
        execution, "launch_kernel", lambda **kw: launched.append(kw["payload"])
    )
    with pytest.raises(
        (RuntimeError, ArgumentTreeError), match="Tensor buffer is not bound"
    ):
        executor(value, stream=0)
    assert not launched
    # CPU payload check only: no allocation or device launch is performed.
    tensor._external_binding = True
    tensor.data_ptr = 0x1000
    executor(value, stream=0)
    tensor.data_ptr = 0x2000
    executor(value, stream=0)
    assert launched == [struct.pack("Q", 0x1000), struct.pack("Q", 0x2000)]
    assert checks == [tensor, tensor, tensor]


@pytest.mark.parametrize(
    "name,runtime_names",
    [
        ("value", ("value",)),
        # ``is_arg_annotation_constexpr`` reserves a leading ``self`` by position
        # and name, so it leaves the runtime signature like a ``Constexpr`` does.
        ("self", ()),
        # ``cls`` is reserved only when the owner is a classmethod, and this
        # synthetic signature has no owner, so it stays a runtime parameter.
        ("cls", ("cls",)),
    ],
)
def test_tree_launch_signature_follows_ported_constexpr_rule(name, runtime_names):
    sig = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(
                "tile",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=tla.Constexpr,
            ),
        ]
    )
    runtime_args = (1,) if runtime_names else ()
    binder = ExecutionArgs(
        original_signature=sig,
        argument_trees=tuple(
            build_runtime_argument_tree(1) for _ in runtime_names
        ),
    )
    assert tuple(binder.signature.parameters) == runtime_names
    assert binder.get_rectified_args_from_original_args((1, 128)) == runtime_args
    assert binder.get_rectified_args_from_original_args((1,)) == runtime_args


def test_lowered_self_parameter_and_launch_signature_agree():
    from catlass.base_dsl import BaseDSL

    def kernel(self, value, tile: tla.Constexpr):
        _ = self + value + tile

    lowered = BaseDSL()._lower(
        kernel, kind="kernel", options={}, type_args=(1, 3, 128)
    )
    # ``self`` and ``tile`` are compile-time; only ``value`` reaches the ABI.
    assert len(lowered.argument_trees) == 1
    binder = ExecutionArgs(
        original_signature=inspect.signature(kernel),
        argument_trees=lowered.argument_trees,
        kernel_abi=compiler_bridge.lower_tlair_module_to_mlir(
            lowered.module
        ).kernel_abi,
    )
    assert tuple(binder.signature.parameters) == ("value",)
    assert binder.generate_launch_payload((3,))[:4] == struct.pack("i", 3)


@pytest.mark.parametrize(
    "annotation,is_static",
    [
        (tla.Constexpr[int], True),
        ("tla.Constexpr[int]", True),
        (ForwardRef("tla.Constexpr[int]"), False),
    ],
)
def test_lowering_and_launch_share_constexpr_annotation_rules(annotation, is_static):
    def kernel(value, tile):
        _ = value + tile

    kernel.__annotations__["tile"] = annotation
    lowered = BaseDSL()._lower(kernel, kind="kernel", options={}, type_args=(1, 128))
    layout = compiler_bridge.lower_tlair_module_to_mlir(lowered.module).kernel_abi
    binder = ExecutionArgs(
        original_signature=inspect.signature(kernel),
        argument_trees=lowered.argument_trees,
        kernel_abi=layout,
    )
    runtime_args = (3,) if is_static else (3, 128)
    runtime_names = ("value",) if is_static else ("value", "tile")
    assert tuple(binder.signature.parameters) == runtime_names
    assert len(lowered.argument_trees) == len(runtime_args)
    assert binder.get_rectified_args_from_original_args((3, 128)) == runtime_args
    assert binder.generate_launch_payload(runtime_args) == execution._pack_launch_args(
        runtime_args, layout
    )


def test_execution_args_preserves_positional_signature_argument():
    def kernel(value):
        pass

    signature = inspect.signature(kernel)
    binder = ExecutionArgs(None, None, None, signature)
    assert binder.signature is signature
    assert binder.argument_trees is None


def test_bound_method_signature_keeps_existing_receiver_binding():
    from catlass.base_dsl.runtime.argument_tree import build_runtime_argument_tree

    class Owner:
        def method(self, value, tile: tla.Constexpr):
            pass

    binder = ExecutionArgs(
        original_signature=inspect.signature(Owner().method),
        argument_trees=(build_runtime_argument_tree(1),),
    )
    assert tuple(binder.signature.parameters) == ("value",)
    assert binder.get_rectified_args_from_original_args((1, 128)) == (1,)


class _Ptr:
    def __c_pointers__(self):
        return [0x123456789ABCDEF0]


def _i32_ptr_layout() -> compiler_bridge.KernelAbiLayout:
    return compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="kernel",
        total_size=16,
        arguments=(
            compiler_bridge.KernelAbiArgument(
                index=0,
                kind=compiler_bridge.KernelAbiArgumentKind.SCALAR,
                scalar=compiler_bridge.KernelAbiScalarDescriptor(
                    compiler_bridge.KernelAbiScalarCategory.INTEGER,
                    32,
                    compiler_bridge.KernelAbiIntegerSignedness.SIGNLESS,
                    None,
                ),
                mlir_type="i32",
                offset=0,
                storage_size=4,
                alignment=4,
            ),
            compiler_bridge.KernelAbiArgument(
                index=1,
                kind=compiler_bridge.KernelAbiArgumentKind.POINTER,
                scalar=None,
                mlir_type="!llvm.ptr",
                offset=8,
                storage_size=8,
                alignment=4,
            ),
        ),
    )


def test_execution_args_requires_kernel_abi() -> None:
    with pytest.raises(execution.TlaUnsupportedAbiError, match="kernel ABI layout"):
        ExecutionArgs().generate_launch_payload([tla.Int32(1)])


def test_execution_args_delegates_pack_to_layout_path() -> None:
    args = [tla.Int32(5), _Ptr()]
    layout = _i32_ptr_layout()
    assert ExecutionArgs(kernel_abi=layout).generate_launch_payload(
        args
    ) == execution._pack_launch_args(args, layout)


def test_from_callable_without_tree_plan_keeps_legacy_rectification() -> None:
    def kernel(value) -> None:
        del value

    value = _Ptr()
    binder = ExecutionArgs.from_callable(kernel)
    assert binder.argument_trees is None
    assert binder.get_rectified_args((value,)) == (value,)


def _structured_tensor(shape=(8,), stride=(1,), dtype=tla.Float32, *, layout_tag=tla.arch.RowMajor):
    return make_fake_tensor(dtype, shape, stride, layout_tag=layout_tag)

@dataclass
class _StaticConfig:
    tile: tla.Constexpr[int] = 128
    width: tla.Constexpr[int] = 256

def test_constexpr_only_tree_retains_logical_launch_position() -> None:
    static = _StaticConfig()
    tree = build_runtime_argument_tree(static)
    assert tree.runtime_leaf_count == 0

    def kernel(config: _StaticConfig) -> None:
        del config

    binder = ExecutionArgs(
        original_signature=BaseDSL()._get_signature(kernel),
        argument_trees=(tree,),
    )
    assert binder.get_rectified_args((static,)) == ()
    with pytest.raises(ValueError, match="compiled kernel signature"):
        binder.get_rectified_args(())

def test_constexpr_only_tree_does_not_shift_direct_runtime_arguments() -> None:
    tensor = _structured_tensor()
    static = _StaticConfig()
    static_tree = build_runtime_argument_tree(static)

    def kernel(value: tla.Tensor, config: _StaticConfig) -> None:
        del value, config

    binder = ExecutionArgs(
        original_signature=BaseDSL()._get_signature(kernel),
        argument_trees=(None, static_tree),
    )

    # The zero-leaf tree remains a logical argument and must not shift the
    # preceding direct tensor into its position.
    assert binder.get_rectified_args((tensor, static)) == (tensor,)
    with pytest.raises(ValueError, match="compiled kernel signature"):
        binder.get_rectified_args((tensor,))

@dataclass
class _EmptyConfig:
    pass


@pytest.mark.parametrize("empty", ((), [], _EmptyConfig()), ids=("tuple", "list", "dataclass"))
def test_empty_aggregate_retains_logical_launch_position(empty) -> None:
    tree = build_runtime_argument_tree(empty)
    assert tree.runtime_leaf_count == 0

    def kernel(aux) -> None:
        del aux

    binder = ExecutionArgs(
        original_signature=BaseDSL()._get_signature(kernel),
        argument_trees=(tree,),
    )
    assert binder.get_rectified_args((empty,)) == ()
    with pytest.raises(ValueError, match="compiled kernel signature"):
        binder.get_rectified_args(())

def test_top_level_none_is_a_unit_argument() -> None:
    tree = build_runtime_argument_tree(None)
    assert tree.runtime_leaf_count == 0

    def kernel(aux) -> None:
        del aux

    lowered = BaseDSL()._lower(
        kernel,
        kind="kernel",
        options={},
        type_args=(None,),
    )
    assert len(lowered.argument_trees) == 1
    assert lowered.argument_trees[0] is not None
    assert lowered.argument_trees[0].runtime_leaf_count == 0

    binder = ExecutionArgs(
        original_signature=BaseDSL()._get_signature(kernel),
        argument_trees=(tree,),
    )
    assert binder.get_rectified_args((None,)) == ()
    with pytest.raises(ValueError, match="compiled kernel signature"):
        binder.get_rectified_args(())

def test_compiled_zero_runtime_parameters_reject_extra_launch_args() -> None:
    def kernel(limit: tla.Constexpr[int]) -> None:
        del limit

    binder = ExecutionArgs(
        original_signature=BaseDSL()._get_signature(kernel),
        argument_trees=(),
    )
    assert binder.get_rectified_args(()) == ()
    with pytest.raises(ValueError, match="compiled kernel signature"):
        binder.get_rectified_args((1,))

@pytest.mark.parametrize("container", ["plain", "tuple", "dataclass"])
@pytest.mark.parametrize(
    "sample,valid,invalid,expected",
    [
        (1, 3, [3.0, 2**31, tla.Int64(3)], struct.pack("<i", 3)),
        (1.0, 3.0, [3, 1e100, tla.Float16(3)], struct.pack("<f", 3.0)),
        (tla.UInt8(1), tla.UInt8(255), [tla.Int8(1), tla.UInt16(1)], b"\xff"),
    ],
)
def test_scalar_launch_uses_compiler_abi_in_each_container(
    container, sample, valid, invalid, expected
):
    @dataclass
    class Config:
        value: object

    def wrap(value):
        if container == "tuple":
            return ([value, None],)
        if container == "dataclass":
            return Config(value)
        return value

    def kernel(aux):
        pass

    lowered = BaseDSL()._lower(
        kernel, kind="kernel", options={}, type_args=(wrap(sample),)
    )
    layout = compiler_bridge.lower_tlair_module_to_mlir(lowered.module).kernel_abi
    assert layout is not None
    binder = ExecutionArgs(
        argument_trees=lowered.argument_trees,
        kernel_abi=layout,
        abi_packer=execution._prepare_abi_packer(layout),
    )
    assert binder.generate_launch_payload((wrap(valid),)) == expected + bytes(
        layout.total_size - len(expected)
    )
    for value in invalid:
        with pytest.raises(execution.TlaUnsupportedAbiError):
            binder.generate_launch_payload((wrap(value),))

@pytest.mark.parametrize("container", ["plain", "unit", "tuple", "dataclass"])
@pytest.mark.parametrize("prepared", [False, True])
def test_signless_scalar_packing_is_independent_of_structure(container, prepared):
    @dataclass
    class Config:
        value: object

    def arguments(value):
        if container == "unit":
            return value, None
        if container == "tuple":
            return (([value, None],),)
        if container == "dataclass":
            return (Config(value),)
        return (value,)

    descriptor = compiler_bridge.KernelAbiScalarDescriptor(
        compiler_bridge.KernelAbiScalarCategory.INTEGER,
        16,
        compiler_bridge.KernelAbiIntegerSignedness.SIGNLESS,
        None,
    )
    layout = compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="kernel",
        total_size=8,
        arguments=(
            compiler_bridge.KernelAbiArgument(
                index=0,
                kind=compiler_bridge.KernelAbiArgumentKind.SCALAR,
                scalar=descriptor,
                mlir_type="i16",
                offset=0,
                storage_size=2,
                alignment=4,
            ),
        ),
    )
    with mlir_ir.Context() as context:
        binder = ExecutionArgs(
            argument_trees=tuple(
                build_runtime_argument_tree(arg, context)
                for arg in arguments(tla.Int16(0))
            ),
            kernel_abi=layout,
            abi_packer=execution._prepare_abi_packer(layout) if prepared else None,
        )
    for value, bits in (
        (tla.UInt16(0xBEEF), b"\xef\xbe"),
        (tla.Int16(-1), b"\xff\xff"),
    ):
        assert binder.generate_launch_payload(arguments(value)) == bits + bytes(6)
    for invalid in (tla.UInt32(1), tla.Float16(1), 1, True):
        with pytest.raises(execution.TlaUnsupportedAbiError):
            binder.generate_launch_payload(arguments(invalid))

@pytest.mark.parametrize("container", ["plain", "unit", "tuple", "dataclass"])
@pytest.mark.parametrize("prepared", [False, True])
def test_pointer_packing_is_independent_of_structure(container, prepared):
    @dataclass
    class Config:
        value: object

    class Pointer:
        def __init__(self, *values):
            self.values = values

        def __c_pointers__(self):
            return self.values

    def arguments(value):
        if container == "unit":
            return value, None
        if container == "tuple":
            return (([value, None],),)
        if container == "dataclass":
            return (Config(value),)
        return (value,)

    layout = compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="kernel",
        total_size=8,
        arguments=(
            compiler_bridge.KernelAbiArgument(
                index=0,
                kind=compiler_bridge.KernelAbiArgumentKind.POINTER,
                scalar=None,
                mlir_type="!llvm.ptr",
                offset=0,
                storage_size=8,
                alignment=4,
            ),
        ),
    )
    binder = ExecutionArgs(
        argument_trees=tuple(
            build_runtime_argument_tree(arg) for arg in arguments(_structured_tensor())
        ),
        kernel_abi=layout,
        abi_packer=execution._prepare_abi_packer(layout) if prepared else None,
    )
    for address in (4096, 8192):
        assert binder.generate_launch_payload(arguments(Pointer(address))) == (
            address.to_bytes(8, "little")
        )
    for invalid in (
        3, True, tla.Int64(1), Pointer(), Pointer(1, 2), Pointer(-1), Pointer(2**64)
    ):
        with pytest.raises(execution.TlaUnsupportedAbiError):
            binder.generate_launch_payload(arguments(invalid))

def test_plain_launch_uses_packer_checks_without_tree_replay(monkeypatch):
    tensor = _structured_tensor()
    tensor.data_ptr = 0x1234
    tensor._external_binding = True
    layout = compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="kernel",
        total_size=16,
        arguments=(
            compiler_bridge.KernelAbiArgument(
                index=0, kind=compiler_bridge.KernelAbiArgumentKind.POINTER,
                scalar=None, mlir_type="!llvm.ptr", offset=0,
                storage_size=8, alignment=4,
            ),
            compiler_bridge.KernelAbiArgument(
                index=1, kind=compiler_bridge.KernelAbiArgumentKind.SCALAR,
                scalar=compiler_bridge.KernelAbiScalarDescriptor(
                    compiler_bridge.KernelAbiScalarCategory.INTEGER, 32,
                    compiler_bridge.KernelAbiIntegerSignedness.SIGNLESS, None,
                ),
                mlir_type="i32", offset=8, storage_size=4, alignment=4,
            ),
        ),
    )
    binder = ExecutionArgs(
        kernel_abi=layout,
        abi_packer=execution._prepare_abi_packer(layout),
        argument_trees=tuple(build_runtime_argument_tree(v) for v in (tensor, 3)),
    )

    def unexpected(*args, **kwargs):
        pytest.fail("plain launch should use ABI packing directly")

    monkeypatch.setattr(ExecutionArgs, "get_rectified_args", unexpected)
    assert binder.generate_launch_payload((tensor, 3))[:12] == struct.pack("<Qi", 0x1234, 3)
    tensor.data_ptr = 0x5678
    assert binder.generate_launch_payload((tensor, 4))[:12] == struct.pack("<Qi", 0x5678, 4)
    for invalid in (
        (tensor,), (tensor, 3, 4), (3, 3), (tensor, 3.0),
        (tensor, tla.Int64(3)), (tensor, 1 << 40),
    ):
        with pytest.raises(execution.TlaUnsupportedAbiError):
            binder.generate_launch_payload(invalid)
    tensor.data_ptr = -1
    with pytest.raises(execution.TlaUnsupportedAbiError):
        binder.generate_launch_payload((tensor, 3))

def test_flat_launch_preserves_values_without_container_replay(monkeypatch) -> None:
    from catlass.base_dsl.runtime import argument_tree

    tensor = _structured_tensor()
    values = (tensor, tla.UInt8(7), 0.5, True)
    with mlir_ir.Context() as context:
        binder = ExecutionArgs(
            argument_trees=tuple(
                build_runtime_argument_tree(value, context) for value in values
            )
        )

    def unexpected_replay(*args, **kwargs):
        pytest.fail("flat arguments must not replay a container tree")

    monkeypatch.setattr(argument_tree, "_walk_runtime_tree", unexpected_replay)
    assert binder.get_rectified_args(values) == values
    assert flatten_runtime_leaves(tensor, binder.argument_trees[0]) == (tensor,)
    replacement = _structured_tensor()
    assert binder.get_rectified_args((replacement, *values[1:]))[0] is replacement
    wider_scalar = tla.UInt16(7)
    assert binder.get_rectified_args((tensor, wider_scalar, *values[2:]))[1] is wider_scalar
    with pytest.raises(ValueError, match="compiled kernel signature"):
        binder.get_rectified_args(values[:-1])
    with pytest.raises(ValueError, match="compiled kernel signature"):
        binder.get_rectified_args((*values, 1))

def test_invalid_tree_plan_is_rejected_when_binder_is_created() -> None:
    def kernel(value):
        pass

    with pytest.raises(ValueError, match="tree plan does not match"):
        ExecutionArgs(
            original_signature=BaseDSL()._get_signature(kernel), argument_trees=()
        )

def test_recursive_tree_launches_pointer_and_scalar_leaves_in_order() -> None:
    bias = _structured_tensor()
    bias.data_ptr = 0x1234
    bias._external_binding = True
    value = (bias, 3)
    tree = build_runtime_argument_tree(value)
    layout = compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="kernel",
        total_size=16,
        arguments=(
            compiler_bridge.KernelAbiArgument(
                index=0,
                kind=compiler_bridge.KernelAbiArgumentKind.POINTER,
                scalar=None,
                mlir_type="!llvm.ptr",
                offset=0,
                storage_size=8,
                alignment=4,
            ),
            compiler_bridge.KernelAbiArgument(
                index=1,
                kind=compiler_bridge.KernelAbiArgumentKind.SCALAR,
                scalar=compiler_bridge.KernelAbiScalarDescriptor(
                    compiler_bridge.KernelAbiScalarCategory.INTEGER,
                    32,
                    compiler_bridge.KernelAbiIntegerSignedness.SIGNLESS,
                    None,
                ),
                mlir_type="i32",
                offset=8,
                storage_size=4,
                alignment=4,
            ),
        ),
    )
    payload = ExecutionArgs(
        kernel_abi=layout, argument_trees=(tree,)
    ).generate_launch_payload([value])
    assert int.from_bytes(payload[0:8], "little") == 0x1234
    assert int.from_bytes(payload[8:12], "little", signed=True) == 3
    assert len(payload) == 16
