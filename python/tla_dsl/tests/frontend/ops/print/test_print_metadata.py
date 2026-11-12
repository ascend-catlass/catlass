import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


METADATA_LIKE_MESSAGE = (
    "tla.simt_print.workspace tla.print_tensor.workspace tla.debug_tunnel.state "
    "@_mlir_ciface_tla_simt_printf call @tla_print_tensor_l1_f32 "
    "func.func @literal_mix_aic() func.func @literal_mix_aiv() "
    "hivm.aic_bitcode hivm.aiv_bitcode hivm.module_core_type<AIC>"
)


@pytest.mark.parametrize(
    "kind,core_ratio,subblocks",
    [
        ("outer", None, 1),
        ("mixed", None, 2),
        ("mixed", tla.arch.C1V1, 1),
        ("mixed", tla.arch.C1V2, 2),
        ("simt_tensor", None, 1),
    ],
)
def test_print_content_does_not_change_compiled_context(kind, core_ratio, subblocks):
    options = {} if core_ratio is None else {"core_ratio": core_ratio}

    @tla.kernel(**options)
    def kernel(source: tla.Tensor):
        if tla.const_expr(kind == "mixed"):
            with tla.cube():
                tla.print(source, 2)
        with tla.vector():
            tla.print(METADATA_LIKE_MESSAGE)
            if tla.const_expr(kind == "mixed"):
                tla.print(source, 2)
            elif tla.const_expr(kind == "simt_tensor"):
                with tla.vec.func(mode="simt", thread_block_dim=32):
                    thread, _, _ = tla.arch.thread_idx()
                    if thread == 0:
                        tla.print(source, 2)

    source = make_fake_tensor(
        tla.Float32, (8,), (1,), origin_shape=(8,), layout_tag=tla.arch.RowMajor
    )
    compiled = kernel.compile(type_args=(source,), options="--npu-arch 3510")
    module = compiled.jit_module
    assert module.print_config.uses_scalar_print
    assert module.print_config.uses_tensor_print == (kind == "mixed")
    assert module.print_config.mixed_core == (kind == "mixed")
    assert module.print_config.aiv_subblock_count == subblocks
    assert module.kernel_mode == ("mix" if kind == "mixed" else "aiv")
    assert module.print_config.print_tensor_position is None
    assert ("simt" in module.print_config.print_argument_layout) == (
        kind == "simt_tensor"
    )
    assert module.print_config.simt_print_formats == (
        b"" if kind == "simt_tensor" else None
    )


@pytest.mark.parametrize("storage", ["gm", "ub"])
@pytest.mark.parametrize("length_kind", ["constant", "runtime", "conditional"])
def test_native_metadata_keeps_tensor_capacity_facts(storage, length_kind):
    from catlass.base_dsl import BaseDSL
    from catlass.compiler_bridge import lower_tlair_module_to_mlir
    from catlass.compiler_metadata import PrintTensorMetadata

    @tla.kernel
    def kernel(source: tla.Tensor, length: tla.Int32):
        with tla.vector():
            if tla.const_expr(storage == "ub"):
                source_ptr = tla.allocate(8, tla.Float32, tla.AddressSpace.ub, 32)
                source = tla.make_tensor_like(source_ptr, source, tla.arch.RowMajor)
            if tla.const_expr(length_kind == "conditional"):
                if length > 0:
                    tla.print(source, 4)
            elif tla.const_expr(length_kind == "runtime"):
                tla.print(source, length)
            else:
                tla.print(source, 4)

    source = make_fake_tensor(tla.Float32, (2, 4), (4, 1), origin_shape=(2, 4))
    lowered = BaseDSL()._lower(
        kernel.fn,
        kind=kernel.kind,
        options=kernel.options,
        type_args=(source, tla.Int32(4)),
    )
    metadata = lower_tlair_module_to_mlir(lowered.module).runtime_metadata
    assert metadata.tensor_records == (
        PrintTensorMetadata(
            (2, 4),
            None if length_kind == "runtime" else 4,
            "f32",
            storage.upper(),
            dynamic_execution=length_kind != "constant",
        ),
    )
    assert metadata.print_helper_core == "aiv"
    assert metadata.print_argument_layout == ("fifo", "user")


@pytest.mark.parametrize("core_ratio", [None, tla.arch.C1V1, tla.arch.C1V2])
def test_disk_cache_restores_metadata_without_parsing_ir(
    monkeypatch, isolated_compile_cache, core_ratio
):
    import json
    import catlass.execution as execution

    options = {} if core_ratio is None else {"core_ratio": core_ratio}

    @tla.kernel(**options)
    def kernel():
        if tla.const_expr(core_ratio is not None):
            with tla.cube():
                pass
        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=32):
                thread, _, _ = tla.arch.thread_idx()
                tla.print("thread={}", thread)

    monkeypatch.setenv("CATLASS_DSL_FORCE_RECOMPILE", "0")
    monkeypatch.setattr(execution, "_MEMORY_COMPILE_CACHE", {})
    compiled = kernel.compile()
    manifest = json.loads((compiled.artifacts.cache_dir / "manifest.json").read_text())
    assert manifest["runtime_metadata"]["print_argument_layout"] == ["user", "simt"]
    execution._MEMORY_COMPILE_CACHE.clear()
    # Cached assembly is a diagnostic artifact, not the source of launch facts.
    (compiled.artifacts.cache_dir / "lowered.mlir").write_text(
        "opaque diagnostic artifact"
    )
    monkeypatch.setattr(
        execution,
        "_run_tla_lowering_to_mlir",
        lambda **kw: pytest.fail("cache hit must not lower"),
    )
    monkeypatch.setattr(
        execution,
        "_run_checked",
        lambda *a, **kw: pytest.fail("cache hit must not compile"),
    )
    restored = kernel.compile()
    assert (
        restored.jit_module.print_config.print_argument_layout
        == compiled.jit_module.print_config.print_argument_layout
    )
    assert (
        restored.jit_module.print_config.simt_print_formats
        == compiled.jit_module.print_config.simt_print_formats
    )
    assert restored.jit_module.kernel_mode == compiled.jit_module.kernel_mode

    assert restored.jit_module.print_config.aiv_subblock_count == (
        2 if core_ratio is tla.arch.C1V2 else 1
    )


@pytest.mark.parametrize(
    "corruption", ["missing", "old", "entrypoint", "layout", "subblocks"]
)
def test_invalid_cached_metadata_recompiles(
    monkeypatch, isolated_compile_cache, corruption
):
    import json
    import catlass.execution as execution

    @tla.kernel
    def kernel():
        with tla.vector():
            tla.print("value={}", 7)

    monkeypatch.setenv("CATLASS_DSL_FORCE_RECOMPILE", "0")
    monkeypatch.setattr(execution, "_MEMORY_COMPILE_CACHE", {})
    # BC lookup uses the independent prebuilt test cache in this suite.
    compiled = kernel.compile()
    path = compiled.artifacts.cache_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    if corruption == "missing":
        del manifest["runtime_metadata"]
    elif corruption == "old":
        manifest["runtime_metadata"]["schema_version"] = 1
    elif corruption == "entrypoint":
        manifest["runtime_metadata"]["entrypoint"] = "other"
    elif corruption == "subblocks":
        manifest["runtime_metadata"]["aiv_subblock_count"] = 3
    else:
        manifest["runtime_metadata"]["print_argument_layout"] = ["user", "simt"]
    path.write_text(json.dumps(manifest))
    execution._MEMORY_COMPILE_CACHE.clear()
    original = execution._run_tla_lowering_to_mlir
    calls = []

    def lower(**kwargs):
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(execution, "_run_tla_lowering_to_mlir", lower)
    restored = kernel.compile()
    assert calls == [1]
    assert restored.jit_module.print_config.print_argument_layout == ("user", "fifo")
    assert json.loads(path.read_text())["runtime_metadata"]["schema_version"] == 2


def test_stale_native_requires_matching_build(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import catlass.compiler_bridge as bridge
    import catlass.execution as execution

    monkeypatch.setattr(
        bridge,
        "_load_bridge_extension",
        lambda: SimpleNamespace(
            lower_to_mlir=lambda *args: {"success": True, "lowered_mlir": "module {}"}
        ),
    )
    monkeypatch.setattr(
        execution,
        "_resolve_tla_compile",
        lambda: pytest.fail("must not mask stale native with CLI fallback"),
    )
    with pytest.raises(
        execution.TlaKernelCompileError, match="rebuild the matching DSL native"
    ):
        execution._run_tla_lowering_to_mlir(
            lowered_module=object(), tlair_mlir="", mlir_path=tmp_path / "lowered.mlir"
        )


def test_native_metadata_preserves_dynamic_shape():
    from catlass.base_dsl import BaseDSL
    from catlass.compiler_bridge import lower_tlair_module_to_mlir

    @tla.kernel
    def kernel(source: tla.Tensor):
        with tla.vector():
            tla.print(source, 4)

    source = make_fake_tensor(tla.Float32, (8,), (1,)).mark_compact_shape_dynamic(0)
    lowered = BaseDSL()._lower(
        kernel.fn, kind=kernel.kind, options=kernel.options, type_args=(source,)
    )
    record = lower_tlair_module_to_mlir(lowered.module).runtime_metadata.tensor_records[
        0
    ]
    assert record.shape == (-1,)
    assert record.count == 4
    assert record.dynamic_execution


@pytest.mark.parametrize(
    "change", [{"count": 9}, {"shape": [0]}, {"call": 1}, {"count": None}]
)
def test_cached_tensor_metadata_validates_capacity_contract(change):
    from catlass.compiler_metadata import KernelRuntimeMetadata, PrintTensorMetadata

    data = KernelRuntimeMetadata(
        "kernel",
        uses_tensor_print=True,
        print_helper_core="aiv",
        print_argument_layout=("fifo", "user"),
        tensor_records=(PrintTensorMetadata((8,), 4, "f32", "GM"),),
    ).to_dict()
    data["tensor_records"][0].update(change)
    with pytest.raises(ValueError):
        KernelRuntimeMetadata.from_dict(data)


@pytest.mark.parametrize("count", [0, 3, True, "2"])
def test_runtime_metadata_rejects_invalid_subblock_count(count):
    from catlass.compiler_metadata import KernelRuntimeMetadata

    data = KernelRuntimeMetadata("kernel", kernel_mode="mix").to_dict()
    data["aiv_subblock_count"] = count
    with pytest.raises(ValueError, match="subblock count"):
        KernelRuntimeMetadata.from_dict(data)


def test_nonmixed_metadata_rejects_multiple_subblocks():
    from catlass.compiler_metadata import KernelRuntimeMetadata

    with pytest.raises(ValueError, match="subblock count"):
        KernelRuntimeMetadata("kernel", aiv_subblock_count=2).validate()
