"""``tla.Constexpr[...]`` kernel parameters must deliver their host value.

Regression guard for a family of bugs where a parameter stopped being treated
as a constexpr and its value was silently dropped: the kernel body saw ``None``,
traced the wrong branch, and — because the compile cache key hashes the emitted
MLIR — every variant collapsed onto one cached kernel.

The later sections exercise the public ``tla.compile(...)`` entry point,
ending with the launch payload it hands to the device. No test here needs an
NPU: the payload is packed with the real ABI
packer against fake operands.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import Mock

import pytest

import catlass.tla as tla
from catlass import execution
from catlass.base_dsl.typing import is_constexpr_annotation
from catlass.execution_lowering import TlaLoweringError, UnsupportedExecutionLowering
from catlass.tla.runtime import make_fake_tensor


def _alloc(n: int) -> None:
    # 32 is UB's local-memory alignment, and each request is sized to it.
    # A coarser alignment would round the small ones up to the same block and
    # erase the differences these tests use to tell branches apart.
    tla.allocate(n, tla.Float32, tla.AddressSpace.ub, 32)


# --------------------------------------------------------------------------
# The value reaches the kernel body
# --------------------------------------------------------------------------


@tla.kernel
def _str_selector(sel: tla.Constexpr[str]) -> None:
    _alloc(64 if sel == "wide" else 16)


@tla.kernel
def _int_selector(sel: tla.Constexpr[int]) -> None:
    _alloc(64 if sel == 1 else 16)


@tla.kernel
def _none_probe(sel: tla.Constexpr[str]) -> None:
    _alloc(64 if sel is None else 16)


def test_str_constexpr_selects_distinct_branches() -> None:
    wide = _str_selector.dump_mlir(type_args=("wide",))
    narrow = _str_selector.dump_mlir(type_args=("narrow",))
    assert wide != narrow, "Constexpr[str] did not reach the kernel body"


def test_int_constexpr_control() -> None:
    assert _int_selector.dump_mlir(type_args=(1,)) != _int_selector.dump_mlir(
        type_args=(0,)
    )


@pytest.mark.parametrize("value", ["wide", "narrow"])
def test_str_constexpr_is_distinguishable_from_none(value: str) -> None:
    assert _none_probe.dump_mlir(type_args=(value,)) != _none_probe.dump_mlir(
        type_args=(None,)
    )


@pytest.mark.parametrize("value", ["wide", "narrow", (4, 8)])
def test_compile_preserves_constexpr_value(monkeypatch, value) -> None:
    @tla.kernel
    def fn(out, sel: tla.Constexpr) -> None: ...

    compile_spy = Mock()
    monkeypatch.setattr(fn, "compile", compile_spy)
    assert tla.compile(fn, None, value) is compile_spy.return_value
    compile_spy.assert_called_once_with(type_args=(None, value))
    assert compile_spy.call_args.kwargs["type_args"][1] is value


@pytest.mark.parametrize("aux", [(), [], None, (3, [None, 2.0])])
def test_compile_preserves_structured_sample(monkeypatch, aux) -> None:
    @tla.kernel
    def fn(aux) -> None: ...

    compile_spy = Mock()
    monkeypatch.setattr(fn, "compile", compile_spy)
    tla.compile(fn, aux)
    compile_spy.assert_called_once_with(type_args=(aux,))
    assert compile_spy.call_args.kwargs["type_args"][0] is aux


@tla.kernel
def _runtime_sample(aux) -> None: ...


@tla.kernel
def _runtime_pair(value, aux) -> None: ...


@pytest.mark.parametrize(
    "samples",
    [
        (object(),),
        ("bad",),
        (None, object()),
        (1, object()),
        (None, "bad"),
        (1, "bad"),
        ((object(),),),
        (([None, "bad"],),),
    ],
)
def test_compile_rejects_invalid_samples_without_erasing_them(samples) -> None:
    kernel = _runtime_sample if len(samples) == 1 else _runtime_pair
    with pytest.raises(
        TlaLoweringError, match="has no runtime type|unsupported runtime"
    ) as error:
        tla.compile(kernel, *samples)
    assert "requires type_args" not in str(error.value)


def test_compile_keeps_all_none_runtime_positions(monkeypatch) -> None:
    compile_spy = Mock()
    monkeypatch.setattr(_runtime_pair, "compile", compile_spy)
    tla.compile(_runtime_pair, None, None)
    compile_spy.assert_called_once_with(type_args=(None, None))


def test_compile_without_parameters_passes_no_samples(monkeypatch) -> None:
    @tla.kernel
    def fn() -> None: ...

    compile_spy = Mock()
    monkeypatch.setattr(fn, "compile", compile_spy)
    tla.compile(fn)
    compile_spy.assert_called_once_with(type_args=None)


@tla.kernel
def _tuple_constexpr(shape: tla.Constexpr[tuple]) -> None:
    _alloc(shape[0] * shape[1])


def test_non_scalar_constexpr_payload() -> None:
    assert _tuple_constexpr.dump_mlir(type_args=((4, 8),)) != _tuple_constexpr.dump_mlir(
        type_args=((4, 16),)
    )


@tla.kernel
def _dtype_constexpr(dt: tla.Constexpr[type]) -> None:
    tla.allocate(64, dt, tla.AddressSpace.ub, 256)


def test_dtype_as_constexpr() -> None:
    # A dtype is a class whose unbound ``__get_mlir_types__`` is callable, so
    # type-resolving it as if it were a runtime argument crashes inside MLIR.
    assert _dtype_constexpr.dump_mlir(
        type_args=(tla.Float32,)
    ) != _dtype_constexpr.dump_mlir(type_args=(tla.Float16,))


@tla.kernel
def _unused_constexpr(sel: tla.Constexpr[str]) -> None:
    _alloc(32)


def test_constexpr_that_changes_nothing_shares_one_kernel() -> None:
    # The flip side of the bug: identical MLIR *should* reuse one kernel.
    assert _unused_constexpr.dump_mlir(
        type_args=("wide",)
    ) == _unused_constexpr.dump_mlir(type_args=("narrow",))


@dataclasses.dataclass
class _StrParams:
    mode: tla.Constexpr[str] = "wide"


@tla.kernel
def _dataclass_str_field(p: _StrParams) -> None:
    _alloc(64 if p.mode == "wide" else 16)


def test_str_constexpr_dataclass_field() -> None:
    assert _dataclass_str_field.dump_mlir(
        type_args=(_StrParams("wide"),)
    ) != _dataclass_str_field.dump_mlir(type_args=(_StrParams("narrow"),))


# --------------------------------------------------------------------------
# Annotation recognition
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "form", ["Constexpr", "Constexpr[str]", "tla.Constexpr[str]", "x.y.Constexpr[int]"]
)
def test_qualified_constexpr_annotations_are_recognized(form: str) -> None:
    assert is_constexpr_annotation(form)


@pytest.mark.parametrize("form", ["MyConstexpr[str]", "int", "tla.Tensor"])
def test_unrelated_annotations_are_not_constexpr(form: str) -> None:
    assert not is_constexpr_annotation(form)


# --------------------------------------------------------------------------
# Call-site handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("samples", [(), (None,)])
def test_omitted_constexpr_param_uses_its_default(monkeypatch, samples) -> None:
    @tla.kernel
    def fn(a=None, mode: tla.Constexpr[str] = "wide") -> None: ...

    compile_spy = Mock()
    monkeypatch.setattr(fn, "compile", compile_spy)
    tla.compile(fn, *samples)
    compile_spy.assert_called_once_with(type_args=(None, "wide"))


def test_constexpr_passed_by_keyword_is_bound_positionally(monkeypatch) -> None:
    @tla.kernel
    def fn(a, mode: tla.Constexpr[str]) -> None: ...

    compile_spy = Mock()
    monkeypatch.setattr(fn, "compile", compile_spy)
    tla.compile(fn, mode="wide", a=None)
    compile_spy.assert_called_once_with(type_args=(None, "wide"))


def test_runtime_options_are_not_bound_as_kernel_args(monkeypatch) -> None:
    @tla.kernel
    def fn(a, mode: tla.Constexpr[str]) -> None: ...

    compile_spy = Mock()
    monkeypatch.setattr(fn, "compile", compile_spy)
    tla.compile(fn, None, mode="wide", options="--npu-arch 3510")
    compile_spy.assert_called_once_with(
        type_args=(None, "wide"), options="--npu-arch 3510"
    )


def test_keyword_constexpr_behind_a_defaulted_param(monkeypatch) -> None:
    # The defaulted `b` must not end the scan: `sel` would then stay in the
    # runtime options and be silently replaced by its own default.
    @tla.kernel
    def fn(a, b=1, sel: tla.Constexpr[str] = "narrow") -> None: ...

    compile_spy = Mock()
    monkeypatch.setattr(fn, "compile", compile_spy)
    tla.compile(fn, None, sel="wide")
    compile_spy.assert_called_once_with(type_args=(None, 1, "wide"))


def test_sparse_keyword_constexprs(monkeypatch) -> None:
    @tla.kernel
    def fn(a, x: tla.Constexpr[int] = 1, y: tla.Constexpr[str] = "narrow") -> None: ...

    compile_spy = Mock()
    monkeypatch.setattr(fn, "compile", compile_spy)
    tla.compile(fn, None, y="wide")
    compile_spy.assert_called_once_with(type_args=(None, 1, "wide"))


def test_unfillable_gap_stops_binding_rather_than_shifting() -> None:
    @tla.kernel
    def fn(a, b, sel: tla.Constexpr[str]) -> None: ...

    with pytest.raises(TlaLoweringError, match="expected 3, got 1"):
        tla.compile(fn, None, sel="wide")


def test_variadic_signature_is_left_untouched(monkeypatch) -> None:
    # No fixed parameter list to bind positional host values against.
    @tla.kernel
    def fn(a, *rest, sel: tla.Constexpr[str] = "narrow") -> None: ...

    compile_spy = Mock()
    monkeypatch.setattr(fn, "compile", compile_spy)
    tla.compile(fn, None, sel="wide")
    compile_spy.assert_called_once_with(type_args=(None,), sel="wide")


def test_param_colliding_with_a_runtime_option_is_reported() -> None:
    @tla.kernel
    def fn(options: tla.Constexpr[str]) -> None: ...

    with pytest.raises(TypeError, match="collides with the runtime option"):
        tla.compile(fn, options="wide")


@pytest.mark.parametrize(
    "samples, error_type, message",
    [
        ((), UnsupportedExecutionLowering, "requires type_args"),
        ((None,), TlaLoweringError, "expected 2, got 1"),
        ((None, None, None), TlaLoweringError, "expected 2, got 3"),
    ],
)
def test_compile_reports_missing_or_extra_samples(samples, error_type, message) -> None:
    with pytest.raises(error_type, match=message):
        tla.compile(_runtime_pair, *samples)


@tla.kernel
def _keyword_only_constexpr(*, sel: tla.Constexpr[str]) -> None:
    _alloc(64 if sel == "wide" else 16)


def test_keyword_only_constexpr_param(monkeypatch) -> None:
    # Keyword-only params are ordinary entries in `arg_names`; the kernel body
    # must not be called with them positionally.
    assert _keyword_only_constexpr.dump_mlir(
        type_args=("wide",)
    ) != _keyword_only_constexpr.dump_mlir(type_args=("narrow",))
    compile_spy = Mock()
    monkeypatch.setattr(_keyword_only_constexpr, "compile", compile_spy)
    tla.compile(_keyword_only_constexpr, sel="wide")
    compile_spy.assert_called_once_with(type_args=("wide",))


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


@tla.kernel
def _unmarked_str_param(sel: str) -> None:
    _alloc(64 if sel == "wide" else 16)


def test_unmarked_compile_time_param_names_itself() -> None:
    with pytest.raises(TlaLoweringError) as excinfo:
        tla.compile(_unmarked_str_param, "wide")
    message = str(excinfo.value)
    assert "'sel'" in message
    assert "Constexpr" in message


# --------------------------------------------------------------------------
# Public entry points
#
# ``tla.compile(...)`` takes kernel arguments and runtime options in one
# ``**kwargs`` and must keep constexpr values out of the launch ABI. The tests
# above stop at the compile boundary or lowering; these also verify artifacts.
# --------------------------------------------------------------------------

VECTOR_ELE = 64
VL_ELE = 64
_OPTIONS = "--npu-arch 3510"


@tla.kernel
def _selectable_binary(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    mode: tla.Constexpr[str],
) -> None:
    """``gm_c = gm_a + gm_b`` or ``gm_a - gm_b``, chosen at compile time."""
    n_ele = gm_a.origin_shape[0]
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    ub_ptr_a = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_ptr_b = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_ptr_c = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)

    ub_a = tla.make_tensor_like(ub_ptr_a, gm_a, tla.arch.RowMajor)
    ub_b = tla.make_tensor_like(ub_ptr_b, gm_b, tla.arch.RowMajor)
    ub_c = tla.make_tensor_like(ub_ptr_c, gm_c, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(ub_a, gm_a)
        tla.copy(ub_b, gm_b)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range((n_ele + VL_ELE - 1) // VL_ELE):
                shape = tla.make_shape(VL_ELE)
                coord = tla.make_coord(i)
                reg_a = tla.tile_view(ub_a, shape, coord).load()
                reg_b = tla.tile_view(ub_b, shape, coord).load()
                if tla.const_expr(mode == "add"):
                    reg_c = tla.add(reg_a, reg_b)
                else:
                    reg_c = tla.sub(reg_a, reg_b)
                tla.tile_view(ub_c, shape, coord).store(reg_c)
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_c, ub_c)
        tla.pipe_barrier(tla.pipes.ALL)


def _fake_operands():
    return tuple(
        make_fake_tensor(
            tla.Float32,
            (VECTOR_ELE,),
            (1,),
            origin_shape=(VECTOR_ELE,),
            coord=(0,),
            layout_tag=tla.arch.RowMajor,
        )
        for _ in range(3)
    )


def _bound_operands():
    """Fake operands stamped with a data pointer, so the ABI packer can run."""
    tensors = _fake_operands()
    for index, tensor in enumerate(tensors, start=1):
        tensor.data_ptr = index * 0x1000
        tensor._external_binding = True
    return tensors


def test_compile_separates_named_constexpr_from_runtime_options(
    require_bc,
) -> None:
    a, b, c = _fake_operands()
    add = tla.compile(_selectable_binary, a, b, c, mode="add", options=_OPTIONS)
    sub = tla.compile(_selectable_binary, a, b, c, mode="sub", options=_OPTIONS)

    # A distinct kernel per mode proves `mode=` was bound as the kernel argument
    # and not swallowed as an unknown runtime option, while `options=` still
    # reached the runtime (a bogus option string would have raised).
    assert add.cache_key != sub.cache_key


def test_compiled_artifact_abi_excludes_the_constexpr(require_bc) -> None:
    a, b, c = _fake_operands()
    compiled = tla.compile(
        _selectable_binary, a, b, c, mode="add", options=_OPTIONS
    )

    # Three tensors in, one compile-time constant that must own no ABI slot.
    assert execution._logical_launch_arg_count(
        compiled.execution_args.kernel_abi
    ) == 3


def _pack_payload(compiled, launch_args):
    """Pack ``launch_args`` through the compiled function's ABI binder."""
    return compiled.execution_args.generate_launch_payload(launch_args)


def test_compiled_artifact_payload_matches_the_tensors_alone(require_bc) -> None:
    a, b, c = _bound_operands()
    compiled = tla.compile(
        _selectable_binary, a, b, c, mode="add", options=_OPTIONS
    )
    kernel_abi = compiled.execution_args.kernel_abi
    assert kernel_abi is not None

    # tla.compile takes the constexpr; the launch that follows must not.
    assert len(_pack_payload(compiled, (a, b, c))) == kernel_abi.total_size
