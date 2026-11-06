from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

compiler_bridge = pytest.importorskip(
    "catlass.compiler_bridge", exc_type=ImportError
)
execution = pytest.importorskip("catlass.execution", exc_type=ImportError)
base_dsl_mod = pytest.importorskip("catlass.base_dsl", exc_type=ImportError)
tla = pytest.importorskip("catlass.tla", exc_type=ImportError)
runtime_mod = pytest.importorskip("catlass.runtime", exc_type=ImportError)
mlir_ir = pytest.importorskip("catlass._mlir.ir", exc_type=ImportError)


_DYNAMIC_GM_FIELDS = (
    "allocated",
    "aligned",
    "offset",
    "size0",
    "size1",
    "size2",
    "size3",
    "stride0",
    "stride1",
    "stride2",
    "stride3",
    "originShape0",
    "originShape1",
)


@tla.kernel
def _native_kernel_abi_types(
    output: tla.Tensor,
    predicate: tla.Bool,
    signed: tla.Int16,
    unsigned: tla.UInt32,
    position: int,
    wide: tla.Int64,
    half: tla.Float16,
    brain: tla.BFloat16,
    single: tla.Float32,
) -> None:
    pass


@tla.kernel
def _native_mixed_kernel_abi(value: tla.Int16) -> None:
    with tla.cube():
        pass
    with tla.vector():
        pass


@tla.kernel
def _native_debug_kernel_abi(value: tla.Int32) -> None:
    with tla.vector():
        tla.print(value)


def _gm_i32_tensor() -> tla.Tensor:
    return make_fake_tensor(
               tla.Int32,
               (1,),
               (1,),
               origin_shape=(1,),
               layout_tag=tla.arch.RowMajor,
           )


def _gm_f16_rank2_tensor() -> tla.Tensor:
    return make_fake_tensor(
        tla.Float16,
        (4, 8),
        (8, 1),
        origin_shape=(4, 8),
        layout_tag=tla.arch.RowMajor,
    )


def _native_lower(kernel, *, type_args=None):
    lowered = base_dsl_mod.BaseDSL()._lower(
        kernel.fn,
        kind=kernel.kind,
        options=dict(kernel.options),
        type_args=type_args,
        location=kernel.decorator_location,
    )
    return compiler_bridge.lower_tlair_module_to_mlir(lowered.module)


def test_native_bridge_returns_anonymous_mlir_provenance_on_pipeline_failure() -> None:
    fixture = (
        Path(__file__).parent
        / "lit"
        / "sync"
        / "mixed-split-symbol-conflict.mlir"
    )
    with mlir_ir.Context() as context:
        context.allow_unregistered_dialects = True
        _tla_type_bridge = pytest.importorskip("catlass._tla_type_bridge")
        _tla_type_bridge._load_bridge_extension().load_tla_dialect(context)
        module = mlir_ir.Module.parse(fixture.read_text())
        with pytest.raises(compiler_bridge.BridgeLoweringError) as exc_info:
            compiler_bridge.lower_tlair_module_to_mlir(module)

    diagnostic = next(
        diagnostic
        for diagnostic in exc_info.value.diagnostics
        if diagnostic.severity == "error"
        and "cannot split mixed function" in diagnostic.message
    )
    assert diagnostic.severity == "error"
    assert "cannot split mixed function" in diagnostic.message
    assert diagnostic.locations == (
        compiler_bridge.BridgeSourceLocation(filename="-", line=4, column=3),
        compiler_bridge.BridgeSourceLocation(
            filename="kernels/example.py", line=17, column=9
        ),
    )
    assert diagnostic.preferred_location == diagnostic.locations[1]
    assert diagnostic.rendered.startswith("loc(fused[")
    assert ": error: " in diagnostic.rendered
    assert "cannot split mixed function" in diagnostic.rendered
    diagnostic_index = exc_info.value.diagnostics.index(diagnostic)
    note = exc_info.value.diagnostics[diagnostic_index + 1]
    assert note.severity == "note"
    assert note.message.startswith("see current operation:")
    assert note.locations == diagnostic.locations
    assert ": note: see current operation:" in note.rendered


def test_native_bridge_omits_unknown_location_from_rendered_diagnostic() -> None:
    fixture = (
        Path(__file__).parent
        / "lit"
        / "sync"
        / "mixed-split-symbol-conflict.mlir"
    )
    with mlir_ir.Context() as context:
        context.allow_unregistered_dialects = True
        _tla_type_bridge = pytest.importorskip("catlass._tla_type_bridge")
        _tla_type_bridge._load_bridge_extension().load_tla_dialect(context)
        module = mlir_ir.Module.parse(
            fixture.read_text().replace(
                'loc(fused["-":4:3, "kernels/example.py":17:9])',
                "loc(unknown)",
            )
        )
        with pytest.raises(compiler_bridge.BridgeLoweringError) as exc_info:
            compiler_bridge.lower_tlair_module_to_mlir(module)

    diagnostic = next(
        diagnostic
        for diagnostic in exc_info.value.diagnostics
        if diagnostic.severity == "error"
        and "cannot split mixed function" in diagnostic.message
    )
    assert diagnostic.locations == ()
    assert diagnostic.rendered.startswith("error: ")
    assert not diagnostic.rendered.startswith("loc(unknown):")


def test_native_bridge_collects_post_lowering_supported_kernel_abi() -> None:
    result = _native_lower(
        _native_kernel_abi_types,
        type_args=(
            _gm_i32_tensor(),
            tla.Bool(True),
            tla.Int16(-2),
            tla.UInt32(3),
            4,
            tla.Int64(-5),
            tla.Float16(1.5),
            tla.BFloat16(2.5),
            tla.Float32(3.5),
        ),
    )

    assert result.kernel_abi is not None
    assert result.kernel_abi.entrypoint == "_native_kernel_abi_types"
    assert [
        (argument.kind, argument.storage_size, argument.offset)
        for argument in result.kernel_abi.arguments
    ] == [
        ("pointer", 8, 0),
        ("scalar", 1, 8),
        ("scalar", 2, 12),
        ("scalar", 4, 16),
        ("scalar", 4, 20),
        ("scalar", 8, 24),
        ("scalar", 2, 32),
        ("scalar", 2, 36),
        ("scalar", 4, 40),
    ]
    assert result.kernel_abi.total_size == 48
    assert tuple(
        argument.mlir_type for argument in result.kernel_abi.arguments[1:]
    ) == ("i1", "i16", "ui32", "i32", "i64", "f16", "bf16", "f32")
    assert result.kernel_abi.arguments[0].scalar is None
    assert tuple(
        (argument.scalar.category, argument.scalar.bit_width)
        for argument in result.kernel_abi.arguments[1:]
        if argument.scalar is not None
    ) == (
        (compiler_bridge.KernelAbiScalarCategory.INTEGER, 1),
        (compiler_bridge.KernelAbiScalarCategory.INTEGER, 16),
        (compiler_bridge.KernelAbiScalarCategory.INTEGER, 32),
        (compiler_bridge.KernelAbiScalarCategory.INTEGER, 32),
        (compiler_bridge.KernelAbiScalarCategory.INTEGER, 64),
        (compiler_bridge.KernelAbiScalarCategory.FLOAT, 16),
        (compiler_bridge.KernelAbiScalarCategory.FLOAT, 16),
        (compiler_bridge.KernelAbiScalarCategory.FLOAT, 32),
    )


def test_native_bridge_mixed_splits_have_one_agreed_logical_layout() -> None:
    result = _native_lower(
        _native_mixed_kernel_abi, type_args=(tla.Int16(7),)
    )

    assert result.kernel_abi is not None
    assert result.kernel_abi.entrypoint == "_native_mixed_kernel_abi"
    assert result.kernel_abi.total_size == 8
    assert [
        (argument.kind, argument.mlir_type, argument.offset, argument.storage_size)
        for argument in result.kernel_abi.arguments
    ] == [("scalar", "i16", 0, 2)]


def test_native_bridge_excludes_hidden_debug_workspace_from_public_abi() -> None:
    result = _native_lower(
        _native_debug_kernel_abi,
        type_args=(tla.Int32(7),),
    )

    assert result.kernel_abi is not None
    assert result.kernel_abi.entrypoint == "_native_debug_kernel_abi"
    assert result.kernel_abi.total_size == 8
    assert [
        (argument.index, argument.mlir_type, argument.offset, argument.storage_size)
        for argument in result.kernel_abi.arguments
    ] == [(0, "i32", 0, 4)]
    assert "tla.debug_print.workspace" in result.lowered_mlir


def test_native_bridge_excludes_hidden_print_tensor_workspace_from_public_abi() -> (
    None
):
    @tla.kernel
    def print_workspace(value: tla.Tensor):
        with tla.vector():
            tla.print(value, 1)

    result = _native_lower(print_workspace, type_args=(_gm_i32_tensor(),))

    assert result.kernel_abi is not None
    assert result.kernel_abi.entrypoint == "print_workspace"
    assert result.kernel_abi.total_size == 8
    assert [
        (argument.index, argument.kind, argument.offset, argument.storage_size)
        for argument in result.kernel_abi.arguments
    ] == [(0, "pointer", 0, 8)]


def test_native_bridge_returns_no_layout_for_unsupported_entry_signature() -> None:
    with mlir_ir.Context() as context:
        context.allow_unregistered_dialects = True
        module = mlir_ir.Module.parse(
            """
            module {
              func.func @unsupported(%arg0: f64) attributes {hacc.entry} {
                return
              }
            }
            """
        )
        result = compiler_bridge.lower_tlair_module_to_mlir(module)

    assert result.kernel_abi is None


def _argument(
    index: int,
    kind: str,
    mlir_type: str,
    offset: int,
    storage_size: int,
) -> dict[str, object]:
    scalar = None
    if kind == "scalar":
        if mlir_type == "index":
            scalar = {
                "category": "index",
                "bit_width": 64,
                "integer_signedness": None,
                "float_format": None,
            }
        elif mlir_type in {"f16", "bf16", "f32"}:
            scalar = {
                "category": "float",
                "bit_width": int(mlir_type[-2:]),
                "integer_signedness": None,
                "float_format": mlir_type,
            }
        else:
            prefix = (
                "signed"
                if mlir_type.startswith("si")
                else "unsigned"
                if mlir_type.startswith("ui")
                else "signless"
            )
            scalar = {
                "category": "integer",
                "bit_width": int(mlir_type.lstrip("sui")),
                "integer_signedness": prefix,
                "float_format": None,
            }
    return {
        "index": index,
        "kind": kind,
        "mlir_type": mlir_type,
        "scalar": scalar,
        "offset": offset,
        "storage_size": storage_size,
        "alignment": 4,
    }


@pytest.mark.parametrize(
    ("entrypoint", "arguments", "total_size", "expected_offsets"),
    [
        (
            "ptr_ptr_i32s",
            [
                _argument(0, "pointer", "memref<16xi32>", 0, 8),
                _argument(1, "pointer", "!llvm.ptr", 8, 8),
                _argument(2, "scalar", "i32", 16, 4),
                _argument(3, "scalar", "ui32", 20, 4),
                _argument(4, "scalar", "f32", 24, 4),
            ],
            32,
            (0, 8, 16, 20, 24),
        ),
        (
            "ptr_i16_ptr",
            [
                _argument(0, "pointer", "!llvm.ptr", 0, 8),
                _argument(1, "scalar", "i16", 8, 2),
                _argument(2, "pointer", "!llvm.ptr", 12, 8),
            ],
            24,
            (0, 8, 12),
        ),
        (
            "narrow_and_wide",
            [
                _argument(0, "scalar", "i1", 0, 1),
                _argument(1, "scalar", "i8", 4, 1),
                _argument(2, "scalar", "bf16", 8, 2),
                _argument(3, "scalar", "f16", 12, 2),
                _argument(4, "scalar", "index", 16, 8),
                _argument(5, "scalar", "i64", 24, 8),
            ],
            32,
            (0, 4, 8, 12, 16, 24),
        ),
    ],
)
def test_typed_bridge_returns_versioned_compiler_kernel_abi_layout(
    monkeypatch,
    entrypoint: str,
    arguments: list[dict[str, object]],
    total_size: int,
    expected_offsets: tuple[int, ...],
) -> None:
    class _FakeExtension:
        def lower_to_mlir(self, *_args):
            return {
                "lowered_mlir": f"module {{ func.func @{entrypoint}() }}",
                "pass_ir_dump": "",
                "runtime_metadata": compiler_bridge.KernelRuntimeMetadata(
                    entrypoint
                ).to_dict(),
                "kernel_abi": {
                    "schema_version": 3,
                    "entrypoint": entrypoint,
                    "total_size": total_size,
                    "arguments": arguments,
                },
            }

    monkeypatch.setattr(
        compiler_bridge, "_load_bridge_extension", lambda: _FakeExtension()
    )

    result = compiler_bridge.lower_tlair_module_to_mlir(object())

    assert isinstance(result.kernel_abi, compiler_bridge.KernelAbiLayout)
    assert result.kernel_abi.schema_version == 3
    assert result.kernel_abi.entrypoint == entrypoint
    assert result.kernel_abi.total_size == total_size
    assert tuple(arg.offset for arg in result.kernel_abi.arguments) == expected_offsets
    assert all(
        isinstance(arg, compiler_bridge.KernelAbiArgument)
        for arg in result.kernel_abi.arguments
    )


def test_typed_bridge_preserves_signed_unsigned_and_float_mlir_types(
    monkeypatch,
) -> None:
    mlir_types = ("i1", "i8", "si16", "ui32", "index", "f16", "bf16", "f32", "i64")

    class _FakeExtension:
        def lower_to_mlir(self, *_args):
            arguments = []
            offset = 0
            for index, mlir_type in enumerate(mlir_types):
                storage_size = (
                    8
                    if mlir_type in {"index", "i64"}
                    else 2
                    if mlir_type in {"si16", "f16", "bf16"}
                    else 1
                    if mlir_type in {"i1", "i8"}
                    else 4
                )
                arguments.append(
                    _argument(
                        index,
                        "scalar",
                        mlir_type,
                        offset,
                        storage_size,
                    )
                )
                offset += max(storage_size, 4)
            return {
                "lowered_mlir": "module { func.func @typed() }",
                "runtime_metadata": compiler_bridge.KernelRuntimeMetadata(
                    "typed"
                ).to_dict(),
                "kernel_abi": {
                    "schema_version": 3,
                    "entrypoint": "typed",
                    "total_size": 48,
                    "arguments": arguments,
                },
            }

    monkeypatch.setattr(
        compiler_bridge, "_load_bridge_extension", lambda: _FakeExtension()
    )

    layout = compiler_bridge.lower_tlair_module_to_mlir(object()).kernel_abi

    assert layout is not None
    assert tuple(argument.mlir_type for argument in layout.arguments) == mlir_types
    assert tuple(argument.scalar for argument in layout.arguments) == (
        compiler_bridge.KernelAbiScalarDescriptor(
            compiler_bridge.KernelAbiScalarCategory.INTEGER,
            1,
            compiler_bridge.KernelAbiIntegerSignedness.SIGNLESS,
            None,
        ),
        compiler_bridge.KernelAbiScalarDescriptor(
            compiler_bridge.KernelAbiScalarCategory.INTEGER,
            8,
            compiler_bridge.KernelAbiIntegerSignedness.SIGNLESS,
            None,
        ),
        compiler_bridge.KernelAbiScalarDescriptor(
            compiler_bridge.KernelAbiScalarCategory.INTEGER,
            16,
            compiler_bridge.KernelAbiIntegerSignedness.SIGNED,
            None,
        ),
        compiler_bridge.KernelAbiScalarDescriptor(
            compiler_bridge.KernelAbiScalarCategory.INTEGER,
            32,
            compiler_bridge.KernelAbiIntegerSignedness.UNSIGNED,
            None,
        ),
        compiler_bridge.KernelAbiScalarDescriptor(
            compiler_bridge.KernelAbiScalarCategory.INDEX, 64, None, None
        ),
        compiler_bridge.KernelAbiScalarDescriptor(
            compiler_bridge.KernelAbiScalarCategory.FLOAT,
            16,
            None,
            compiler_bridge.KernelAbiFloatFormat.F16,
        ),
        compiler_bridge.KernelAbiScalarDescriptor(
            compiler_bridge.KernelAbiScalarCategory.FLOAT,
            16,
            None,
            compiler_bridge.KernelAbiFloatFormat.BF16,
        ),
        compiler_bridge.KernelAbiScalarDescriptor(
            compiler_bridge.KernelAbiScalarCategory.FLOAT,
            32,
            None,
            compiler_bridge.KernelAbiFloatFormat.F32,
        ),
        compiler_bridge.KernelAbiScalarDescriptor(
            compiler_bridge.KernelAbiScalarCategory.INTEGER,
            64,
            compiler_bridge.KernelAbiIntegerSignedness.SIGNLESS,
            None,
        ),
    )


@pytest.mark.parametrize(
    "argument",
    [
        {
            **_argument(0, "pointer", "!llvm.ptr", 0, 8),
            "scalar": {
                "category": "integer",
                "bit_width": 64,
                "integer_signedness": "signless",
                "float_format": None,
            },
        },
        {**_argument(0, "scalar", "i32", 0, 4), "scalar": None},
        {
            **_argument(0, "scalar", "i32", 0, 4),
            "scalar": {
                "category": "integer",
                "bit_width": 32,
                "integer_signedness": None,
                "float_format": None,
            },
        },
        {
            **_argument(0, "scalar", "f16", 0, 2),
            "scalar": {
                "category": "float",
                "bit_width": 16,
                "integer_signedness": "signless",
                "float_format": "f16",
            },
        },
        {
            **_argument(0, "scalar", "f16", 0, 2),
            "scalar": {
                "category": "float",
                "bit_width": 32,
                "integer_signedness": None,
                "float_format": "f16",
            },
        },
        {
            **_argument(0, "scalar", "i32", 0, 8),
            "scalar": {
                "category": "integer",
                "bit_width": 32,
                "integer_signedness": "signless",
                "float_format": None,
            },
        },
    ],
)
def test_kernel_abi_from_dict_rejects_incoherent_structured_scalar(
    argument: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="scalar|pointer|descriptor|width"):
        compiler_bridge.kernel_abi_from_dict(
            {
                "schema_version": 3,
                "entrypoint": "kernel",
                "total_size": 8,
                "arguments": [argument],
            }
        )


class _FakeLowered:
    entrypoint = "artifact_kernel"
    module = object()
    extern_compile_specs = ()
    argument_trees = ()

    def asm(self, *, generic: bool = False) -> str:
        del generic
        return "module { tla.func @artifact_kernel() { tla.return } }"


def test_compile_artifact_propagates_and_persists_kernel_abi(
    monkeypatch, tmp_path
) -> None:
    layout = compiler_bridge.KernelAbiLayout(
        schema_version=3,
        entrypoint="artifact_kernel",
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
    hivmc = tmp_path / "hivmc-a5"
    hivmc.write_text("")
    template_bc = tmp_path / "bc" / "meta_op.aiv.c310.bc"
    template_bc.parent.mkdir(parents=True)
    template_bc.write_bytes(b"bc")
    monkeypatch.setattr(
        execution,
        "_get_bc_paths",
        lambda kernel_mode, **_kwargs: str(template_bc.resolve()),
    )
    monkeypatch.setattr(
        base_dsl_mod.BaseDSL, "_lower", lambda *_a, **_k: _FakeLowered()
    )
    monkeypatch.setattr(execution, "resolve_bridge_extension_path", lambda: None)
    monkeypatch.setattr(execution, "_find_hivmc_a5", lambda: hivmc)
    monkeypatch.setattr(execution, "_tool_version", lambda _path: "test-version")
    monkeypatch.setattr(
        execution,
        "lower_tlair_module_to_mlir",
        lambda *_a, **_k: compiler_bridge.TlaLoweringResult(
            "module { func.func @artifact_kernel(!llvm.ptr, i32) }",
            kernel_abi=layout,
            runtime_metadata=compiler_bridge.KernelRuntimeMetadata(layout.entrypoint),
        ),
    )

    def fake_run_checked(
        _cmd, *, label, cwd, stdin_text=None, diagnostic_input_mlir=""
    ):
        assert label == "hivmc-a5"
        assert stdin_text is None
        assert diagnostic_input_mlir
        Path(cwd, "kernel.o").write_bytes(b"object")

    monkeypatch.setattr(execution, "_run_checked", fake_run_checked)

    compiled = execution._compile_kernel(
        lambda: None,
        kind="kernel",
        options={},
        compile_option=execution.TlaCompileOption(
            cache_enabled=False, cache_dir=tmp_path / "cache"
        ),
    )
    manifest = json.loads((compiled.artifacts.cache_dir / "manifest.json").read_text())

    assert compiled.execution_args.kernel_abi == layout
    assert manifest["kernel_abi"] == {
        "schema_version": 3,
        "entrypoint": "artifact_kernel",
        "total_size": 16,
        "arguments": [
            {
                "index": 0,
                "logical_index": 0,
                "kind": "pointer",
                "field": None,
                "scalar": None,
                "mlir_type": "!llvm.ptr",
                "offset": 0,
                "storage_size": 8,
                "alignment": 4,
            },
            {
                "index": 1,
                "logical_index": 1,
                "kind": "scalar",
                "field": None,
                "scalar": {
                    "category": "integer",
                    "bit_width": 32,
                    "integer_signedness": "signless",
                    "float_format": None,
                },
                "mlir_type": "i32",
                "offset": 8,
                "storage_size": 4,
                "alignment": 4,
            },
        ],
    }


def test_cached_manifest_without_kernel_abi_is_stale() -> None:
    with pytest.raises(
        execution.TlaKernelCompileError,
        match="kernel ABI|kernel_abi|cache",
    ):
        execution._kernel_abi_from_manifest(
            {
                "cache_key": "old-implicit-uint64-contract",
                "entrypoint": "kernel",
                "kernel_binary": "kernel.o",
                "lowered_mlir": "lowered.mlir",
            }
        )


@pytest.mark.parametrize("schema_version", [1, 2])
def test_cached_older_kernel_abi_schema_is_stale(schema_version: int) -> None:
    with pytest.raises(
        execution.TlaKernelCompileError,
        match="schema version",
    ):
        execution._kernel_abi_from_manifest(
            {
                "entrypoint": "kernel",
                "kernel_abi": {
                    "schema_version": schema_version,
                    "entrypoint": "kernel",
                    "total_size": 0,
                    "arguments": [],
                },
            }
        )


def _dynamic_gm_rank1_tensor() -> tla.Tensor:
    tensor = make_fake_tensor(
                 tla.Int32,
                 (8,),
                 (1,),
                 origin_shape=(8,),
                 layout_tag=tla.arch.RowMajor,
             )
    return tensor.mark_compact_shape_dynamic(0)


def _dynamic_gm_rank2_tensor() -> tla.Tensor:
    tensor = make_fake_tensor(
                 tla.Float16,
                 (16, 32),
                 (32, 1),
                 origin_shape=(16, 32),
                 layout_tag=tla.arch.RowMajor,
             )
    return tensor.mark_layout_dynamic()


def _dynamic_gm_zn_tensor() -> tla.Tensor:
    tensor = make_fake_tensor(tla.Float16, ((16, 2), (16, 4)), ((16, 256), (1, 512)), layout_tag=tla.arch.zN, origin_shape=(32, 64), coord=(0, 0))
    return tensor.mark_layout_dynamic(leading_dim=2)


@tla.kernel
def _native_dynamic_rank1_gm_abi(buf: tla.Tensor) -> None:
    pass


@tla.kernel
def _native_mixed_dynamic_static_gm_abi(
    dynamic: tla.Tensor, static: tla.Tensor, output: tla.Tensor
) -> None:
    pass


@tla.kernel
def _native_dynamic_rank1_gm_reads_origin(buf: tla.Tensor) -> None:
    _ = buf.origin_shape[0]


@tla.kernel
def _native_dynamic_rank2_gm_abi(buf: tla.Tensor) -> None:
    pass


@tla.kernel
def _native_dynamic_zn_gm_reads_origin(buf: tla.Tensor) -> None:
    _ = buf.origin_shape[0]


def test_native_bridge_dynamic_rank1_gm_emits_memref_fields() -> None:
    type_args = (_dynamic_gm_rank1_tensor(),)
    tlair = _native_dynamic_rank1_gm_abi.dump_mlir(type_args=type_args)
    assert "tla.tensor_extent" not in tlair
    assert "memref.dim" in tlair
    assert "tla.tensor_desc" in tlair
    assert "memref<?x?x?x?" in tlair

    result = _native_lower(_native_dynamic_rank1_gm_abi, type_args=type_args)

    assert result.kernel_abi is not None
    assert result.kernel_abi.schema_version == 4
    assert [
        (argument.kind, argument.field, argument.logical_index, argument.storage_size)
        for argument in result.kernel_abi.arguments
    ] == [
        ("memref_field", "allocated", 0, 8),
        ("memref_field", "aligned", 0, 8),
        ("memref_field", "offset", 0, 8),
        ("memref_field", "size0", 0, 8),
        ("memref_field", "size1", 0, 8),
        ("memref_field", "size2", 0, 8),
        ("memref_field", "size3", 0, 8),
        ("memref_field", "stride0", 0, 8),
        ("memref_field", "stride1", 0, 8),
        ("memref_field", "stride2", 0, 8),
        ("memref_field", "stride3", 0, 8),
        ("memref_field", "originShape0", 0, 8),
        ("memref_field", "originShape1", 0, 8),
    ]
    assert result.kernel_abi.total_size == 104


@pytest.mark.parametrize(
    ("static_tensor", "static_fields", "total_size"),
    [
        pytest.param(
            _gm_i32_tensor,
            ("allocated", "aligned", "offset", "size0", "stride0"),
            184,
            id="rank1",
        ),
        pytest.param(
            _gm_f16_rank2_tensor,
            ("allocated", "aligned", "offset", "size0", "size1", "stride0", "stride1"),
            216,
            id="rank2",
        ),
    ],
)
def test_native_bridge_mixed_gm_memrefs_use_descriptor_fields_for_all_gm_args(
    static_tensor, static_fields, total_size
) -> None:
    result = _native_lower(
        _native_mixed_dynamic_static_gm_abi,
        type_args=(
            _dynamic_gm_rank1_tensor(),
            static_tensor(),
            static_tensor(),
        ),
    )

    assert result.kernel_abi is not None
    assert result.kernel_abi.schema_version == 4
    fields = [
        (argument.logical_index, argument.field)
        for argument in result.kernel_abi.arguments
    ]
    assert fields == [
        *((0, field) for field in _DYNAMIC_GM_FIELDS),
        *((1, field) for field in static_fields),
        *((2, field) for field in static_fields),
    ]
    assert result.kernel_abi.total_size == total_size


def _lower_memref_entry(memref, *, dynamic_gm=False):
    from catlass import _tla_type_bridge

    marker = " {tla.dynamic_gm}" if dynamic_gm else ""
    with mlir_ir.Context() as context:
        context.allow_unregistered_dialects = True
        _tla_type_bridge.load_tla_dialect(context)
        module = mlir_ir.Module.parse(
            "module { func.func @memref_entry("
            f"%buffer: {memref}{marker}, %first: index, %second: index, "
            "%output: memref<1xf32, 1>) attributes {hacc.entry} { return } }"
        )
        return compiler_bridge.lower_tlair_module_to_mlir(module)


@pytest.mark.parametrize(
    "layout,descriptor",
    [
        ("strided<[3, 1], offset: 0>", False),
        ("strided<[6, 1], offset: 0>", False),
        ("strided<[?, 1], offset: 0>", True),
        ("strided<[6, 1], offset: ?>", True),
        ("strided<[?, 1], offset: ?>", True),
    ],
)
def test_static_shape_memref_abi_accounts_for_strides_and_offset(layout, descriptor):
    abi = _lower_memref_entry(f"memref<2x3xf32, {layout}, 1>").kernel_abi
    assert abi is not None
    fields = [(arg.logical_index, arg.field) for arg in abi.arguments]
    if descriptor:
        assert fields == [
            *(
                (0, field)
                for field in (
                    "allocated",
                    "aligned",
                    "offset",
                    "size0",
                    "size1",
                    "stride0",
                    "stride1",
                )
            ),
            (1, None),
            (2, None),
            *(
                (3, field)
                for field in ("allocated", "aligned", "offset", "size0", "stride0")
            ),
        ]
        assert abi.total_size == 112
    else:
        assert fields == [(0, None), (1, None), (2, None), (3, None)]
        assert abi.total_size == 32


@pytest.mark.parametrize("dynamic_gm", [False, True])
def test_origin_arguments_are_folded_only_for_marked_dynamic_gm(dynamic_gm):
    abi = _lower_memref_entry(
        "memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, 1>",
        dynamic_gm=dynamic_gm,
    ).kernel_abi
    assert abi is not None
    fields = [(arg.logical_index, arg.field) for arg in abi.arguments]
    output_fields = ("allocated", "aligned", "offset", "size0", "stride0")
    if dynamic_gm:
        assert fields == [(0, field) for field in _DYNAMIC_GM_FIELDS] + [
            (1, field) for field in output_fields
        ]
    else:
        assert fields == [(0, field) for field in _DYNAMIC_GM_FIELDS[:-2]] + [
            (1, None),
            (2, None),
            *((3, field) for field in output_fields),
        ]


@pytest.mark.parametrize(
    "memref,dynamic_gm",
    [
        ("memref<2x3xf32, strided<[?, 1], offset: ?>, 1>", True),
        ("memref<?x?x?x?x?xf32, 1>", False),
        ("memref<2x3xf32, affine_map<(d0, d1) -> (d0 floordiv 2, d1)>, 1>", False),
    ],
)
def test_unrepresentable_memref_abi_is_not_emitted(memref, dynamic_gm):
    assert _lower_memref_entry(memref, dynamic_gm=dynamic_gm).kernel_abi is None


def test_native_dynamic_gm_origin_shape_uses_prologue_metadata() -> None:
    type_args = (_dynamic_gm_rank1_tensor(),)
    tlair = _native_dynamic_rank1_gm_reads_origin.dump_mlir(type_args=type_args)
    assert "tla.tensor_extent" not in tlair
    assert "memref.dim" in tlair
    assert "tla.tensor_desc" in tlair
    # origin_shape read is a side-table Numeric; no second extent op.
    assert tlair.count("memref.dim") >= 1


def test_native_bridge_dynamic_rank2_gm_emits_memref_fields() -> None:
    type_args = (_dynamic_gm_rank2_tensor(),)
    tlair = _native_dynamic_rank2_gm_abi.dump_mlir(type_args=type_args)
    assert "tla.tensor_extent" not in tlair
    assert "memref.dim" in tlair
    assert "tla.tensor_desc" in tlair

    result = _native_lower(_native_dynamic_rank2_gm_abi, type_args=type_args)

    assert result.kernel_abi is not None
    assert result.kernel_abi.schema_version == 4
    assert [
        (argument.kind, argument.field, argument.logical_index)
        for argument in result.kernel_abi.arguments
    ] == [
        ("memref_field", "allocated", 0),
        ("memref_field", "aligned", 0),
        ("memref_field", "offset", 0),
        ("memref_field", "size0", 0),
        ("memref_field", "size1", 0),
        ("memref_field", "size2", 0),
        ("memref_field", "size3", 0),
        ("memref_field", "stride0", 0),
        ("memref_field", "stride1", 0),
        ("memref_field", "stride2", 0),
        ("memref_field", "stride3", 0),
        ("memref_field", "originShape0", 0),
        ("memref_field", "originShape1", 0),
    ]
    assert result.kernel_abi.total_size == 104
    assert all(argument.storage_size == 8 for argument in result.kernel_abi.arguments)


def test_native_bridge_dynamic_zn_gm_materializes_four_slot_descriptor() -> None:
    type_args = (_dynamic_gm_zn_tensor(),)
    tlair = _native_dynamic_zn_gm_reads_origin.dump_mlir(type_args=type_args)

    assert "memref<?x?x?x?xf16" in tlair
    assert "tla.tensor_desc" in tlair
    desc_line = next(line for line in tlair.splitlines() if "tla.tensor_desc" in line)
    shape = desc_line.split(" shape[", 1)[1].split("] stride[", 1)[0]
    stride = desc_line.split(" stride[", 1)[1].split("] origin_shape[", 1)[0]
    origin = desc_line.split(" origin_shape[", 1)[1].split("] coord[", 1)[0]
    coord = desc_line.split(" coord[", 1)[1].split("] :", 1)[0]
    assert len(shape.split(", ")) == 4
    assert len(stride.split(", ")) == 4
    assert len(origin.split(", ")) == 2
    assert len(coord.split(", ")) == 2
    assert "!tla.shape<(?,?),(?,?)>" in desc_line
    assert "!tla.stride<(?,?),(1,?)>" in desc_line
    assert "!tla.shape<?,?>" in desc_line
    result = _native_lower(
        _native_dynamic_zn_gm_reads_origin, type_args=type_args
    )
    assert result.kernel_abi is not None
    assert result.kernel_abi.schema_version == 4
    assert result.kernel_abi.total_size == 104


@dataclass
class _ScalarArgs:
    predicate: tla.Bool
    a: tla.Int16
    b: tla.Int16


@tla.kernel
def _native_scalar_then_tensor_abi(
    scalars: _ScalarArgs,
    tensor: tla.Tensor,
    c: tla.Int32,
) -> None:
    pass


def test_dataclass_kernel_arg_layout_abi() -> None:
    """ABI packs parameters with uniform 4-byte alignment, not struct layout.

    The dataclass scalars (Bool / Int16 / Int16) unpack to the leading 4-byte
    aligned args; even the 8-byte tensor pointer sits at a 4-byte-aligned offset
    (12), which is sufficient. The total payload is rounded up to a multiple of 8.
    """
    tensor = make_fake_tensor(
                 tla.Float32,
                 (8,),
                 (1,),
                 origin_shape=(8,),
                 layout_tag=tla.arch.RowMajor,
             )
    result = _native_lower(
        _native_scalar_then_tensor_abi,
        type_args=(
            _ScalarArgs(tla.Bool(True), tla.Int16(1), tla.Int16(2)),
            tensor,
            tla.Int32(3),
        ),
    )
    abi = result.kernel_abi
    assert abi is not None
    arguments = [
        (a.kind.value, a.storage_size, a.offset, a.alignment)
        for a in abi.arguments
    ]
    assert arguments == [
        ("scalar", 1, 0, 4),  # dataclass Bool
        ("scalar", 2, 4, 4),  # dataclass Int16
        ("scalar", 2, 8, 4),  # dataclass Int16
        ("pointer", 8, 12, 4),  # tensor — 8-byte param at a 4-byte-aligned offset
        ("scalar", 4, 20, 4),  # Int32
    ]
    # Uniform 4-byte alignment: every arg offset is aligned to its declared 4.
    for argument in abi.arguments:
        assert argument.alignment == 4
        assert argument.offset % 4 == 0
    assert abi.total_size == 24
    assert abi.total_size % 8 == 0
