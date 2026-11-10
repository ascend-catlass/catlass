from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path

import pytest

import catlass.tla as tla
from catlass.base_dsl import BaseDSL
from catlass.runtime import TlaCoreAPIError
from catlass import execution
from catlass.base_dsl.arch import get_kernel_target
from catlass.execution_lowering import ExternCompileSpec


EXAMPLES_DIR = Path(__file__).parents[3] / "examples" / "python" / "tla_dsl"
EXAMPLE_PATH = (
    EXAMPLES_DIR / "dsl_tutorials" / "extern_op" / "extern_vecadd.py"
)
DUAL_CORE_EXAMPLE_PATH = EXAMPLE_PATH.with_name("extern_dual_core.py")
MULTI_OPS_EXAMPLE_PATH = EXAMPLE_PATH.with_name("extern_multi_ops.py")

EXTERN_SOURCE_CODE = r"""
#include <cstdint>

extern "C" {
[aicore] void tla_user_gm_to_ub_f32(
    uint64_t gm_ptr, uint64_t ub_ptr, int32_t ele_num) {
  (void)gm_ptr;
  (void)ub_ptr;
  (void)ele_num;
}
}
"""

DUAL_CORE_SOURCE_CODE = r"""
#include <cstdint>

extern "C" {
[aicore] __attribute__((noinline)) void tla_user_dual_core(int32_t value) {
  (void)value;
}
}
"""

SHARED_EXTERN_SOURCE_CODE = r"""
#include <cstdint>

extern "C" {
[aicore] void tla_user_shared_a(int32_t value) { (void)value; }
[aicore] void tla_user_shared_b(int32_t value) { (void)value; }
}
"""


@tla.extern(source=EXTERN_SOURCE_CODE, name="tla_user_gm_to_ub_f32")
def gm_to_ub(
    gm_ptr: tla.Pointer[tla.Float32, tla.AddressSpace.gm],
    ub_ptr: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    ele_num: tla.Int32,
) -> None: ...


@tla.extern(source=DUAL_CORE_SOURCE_CODE)
def tla_user_dual_core(value: tla.Int32) -> None: ...


@tla.extern(source=SHARED_EXTERN_SOURCE_CODE)
def tla_user_shared_a(value: tla.Int32) -> None: ...


@tla.extern(source=SHARED_EXTERN_SOURCE_CODE)
def tla_user_shared_b(value: tla.Int32) -> None: ...


@tla.extern(
    source=SHARED_EXTERN_SOURCE_CODE,
    name="tla_user_shared_a",
    include_dirs=[Path(__file__).resolve().parent],
)
def alternate_tla_user_shared_a(value: tla.Int32) -> None: ...


@tla.kernel
def extern_load_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    with tla.vector():
        gm_to_ub(gm.ptr, ub, 256)


@tla.kernel
def two_extern_ops_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    with tla.vector():
        tla_user_dual_core(256)
        gm_to_ub(gm.ptr, ub, 256)


@tla.kernel
def shared_source_mixed_symbols_kernel(value: tla.Int32) -> None:
    with tla.cube():
        tla_user_shared_a(value)
    with tla.vector():
        tla_user_shared_b(value)


@tla.kernel
def shared_source_conflicting_includes_kernel(value: tla.Int32) -> None:
    with tla.cube():
        tla_user_shared_b(value)
        alternate_tla_user_shared_a(value)


@tla.kernel
def wrong_address_space_kernel(gm: tla.Tensor) -> None:
    with tla.vector():
        gm_to_ub(gm.ptr, gm.ptr, 256)


@tla.kernel
def tensor_argument_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    with tla.vector():
        gm_to_ub(gm, ub, 256)


@tla.kernel
def outside_vector_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    gm_to_ub(gm.ptr, ub, 256)


@tla.kernel
def extern_cube_kernel(value: tla.Int32) -> None:
    with tla.cube():
        tla_user_dual_core(value)


@tla.kernel
def alternate_shared_source_kernel(value: tla.Int32) -> None:
    with tla.cube():
        alternate_tla_user_shared_a(value)


@tla.kernel
def duplicate_symbol_in_one_kernel(value: tla.Int32) -> None:
    with tla.cube():
        tla_user_shared_a(value)
        alternate_tla_user_shared_a(value)


@tla.kernel
def extern_mix_kernel(value: tla.Int32) -> None:
    with tla.cube():
        tla_user_dual_core(value)
    with tla.vector():
        tla_user_dual_core(value)


@tla.kernel
def inside_vec_func_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            gm_to_ub(gm.ptr, ub, 256)


@tla.kernel
def inside_simt_func_kernel(value: tla.Int32) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=64):
            tla_user_dual_core(value)


@tla.kernel
def wrong_scalar_type_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    with tla.vector():
        gm_to_ub(gm.ptr, ub, tla.Int64(256))


@tla.kernel
def wrong_argument_count_kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    with tla.vector():
        gm_to_ub(gm.ptr, ub)


@tla.kernel
def unused_extern_kernel(gm: tla.Tensor) -> None:
    pass


def _fake_gm_tensor():
    return tla.make_fake_tensor(
        tla.Float32,
        (256,),
        (1,),
        origin_shape=(256,),
        coord=(0,),
        layout_tag=tla.arch.RowMajor,
    )


def _lower_kernel(kernel, *type_args):
    return BaseDSL()._lower(
        kernel.fn,
        kind="kernel",
        options={},
        type_args=type_args,
        location=kernel.decorator_location,
    )


def _compile_spec(source, core_types, include_dirs=()):
    return ExternCompileSpec(
        source=source,
        core_types=frozenset(core_types),
        include_dirs=tuple(include_dirs),
    )


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extern_declaration_describes_inline_source() -> None:
    assert tla_user_dual_core.source == DUAL_CORE_SOURCE_CODE
    assert tla_user_dual_core.symbol == "tla_user_dual_core"
    assert tla_user_dual_core.arg_types == (tla.Int32,)
    assert tla_user_dual_core.include_dirs == ()
    assert callable(tla_user_dual_core)


def test_pointer_subscription_returns_public_typed_pointer() -> None:
    pointer_type = tla.Pointer[tla.Float32, tla.AddressSpace.gm]

    assert isinstance(pointer_type, tla.TypedPointer)
    assert pointer_type.dtype is tla.Float32
    assert pointer_type.space is tla.AddressSpace.gm


def test_pointer_subscription_requires_dtype_and_memory_space() -> None:
    with pytest.raises(TypeError, match="expects \\(dtype, memory_space\\)"):
        tla.Pointer[tla.Float32]


def test_extern_rejects_non_string_or_empty_source(tmp_path) -> None:
    with pytest.raises(TypeError, match="source must be str"):
        tla.extern(source=tmp_path / "op.cpp")
    with pytest.raises(ValueError, match="source must not be empty"):
        tla.extern(source="  \n")


def test_extern_rejects_invalid_name() -> None:
    with pytest.raises(ValueError, match="name must be a C identifier"):
        tla.extern(source=EXTERN_SOURCE_CODE, name="op-name")


def test_extern_rejects_invalid_function_signature() -> None:
    def missing_return(value: tla.Int32): ...

    with pytest.raises(TypeError, match="return annotation.*is missing"):
        tla.extern(source=EXTERN_SOURCE_CODE)(missing_return)

    def variadic(*values: tla.Int32) -> None: ...

    with pytest.raises(TypeError, match="positional and fixed"):
        tla.extern(source=EXTERN_SOURCE_CODE)(variadic)

    def missing_parameter_annotation(value) -> None: ...

    with pytest.raises(TypeError, match="missing an annotation"):
        tla.extern(source=EXTERN_SOURCE_CODE)(missing_parameter_annotation)


def test_extern_frontend_emits_call_and_tracks_dependency() -> None:
    lowered = _lower_kernel(extern_load_kernel, _fake_gm_tensor())
    mlir = lowered.asm()
    assert "tla.call_extern" in mlir
    assert "@tla_user_gm_to_ub_f32" in mlir
    assert "!tla.ptr<f32, gm" in mlir
    assert "!tla.ptr<f32, ub" in mlir
    assert lowered.extern_compile_specs == (_compile_spec(EXTERN_SOURCE_CODE, {"aiv"}),)


def test_extern_call_location_points_to_user_call() -> None:
    source_lines, first_line = inspect.getsourcelines(extern_load_kernel.fn)
    call_line = first_line + next(
        index for index, line in enumerate(source_lines) if "gm_to_ub(" in line
    )

    lowered = _lower_kernel(extern_load_kernel, _fake_gm_tensor())
    with lowered.context:
        mlir = lowered.module.operation.get_asm(
            enable_debug_info=True,
            assume_verified=False,
        )

    assert f'test_extern_op.py":{call_line}:' in mlir


def test_extern_supports_multiple_ops_and_preserves_source_call_order() -> None:
    lowered = _lower_kernel(two_extern_ops_kernel, _fake_gm_tensor())

    assert lowered.asm().count("tla.call_extern") == 2
    assert lowered.extern_compile_specs == (
        _compile_spec(DUAL_CORE_SOURCE_CODE, {"aiv"}),
        _compile_spec(EXTERN_SOURCE_CODE, {"aiv"}),
    )


def test_extern_declarations_are_isolated_across_kernels() -> None:
    default = _lower_kernel(shared_source_mixed_symbols_kernel, tla.Int32(0))
    alternate = _lower_kernel(alternate_shared_source_kernel, tla.Int32(0))

    assert tla_user_shared_a.symbol == alternate_tla_user_shared_a.symbol
    (default_spec,) = default.extern_compile_specs
    (alternate_spec,) = alternate.extern_compile_specs
    assert default_spec.source == alternate_spec.source == SHARED_EXTERN_SOURCE_CODE
    assert default_spec.include_dirs == ()
    assert alternate_spec.include_dirs == (Path(__file__).resolve().parent,)


def test_extern_rejects_duplicate_symbol_declarations_in_one_kernel() -> None:
    with pytest.raises(
        TlaCoreAPIError, match="symbol 'tla_user_shared_a' has multiple declarations"
    ):
        _lower_kernel(duplicate_symbol_in_one_kernel, tla.Int32(0))


def test_extern_rejects_different_include_dirs_for_one_source_in_one_kernel() -> None:
    with pytest.raises(TlaCoreAPIError) as exc_info:
        _lower_kernel(shared_source_conflicting_includes_kernel, tla.Int32(0))

    message = str(exc_info.value)
    assert "source compile configuration conflict" in message
    assert "symbol 'tla_user_shared_b' uses include_dirs=[]" in message
    assert (
        "symbol 'tla_user_shared_a' uses include_dirs="
        f"{[str(Path(__file__).resolve().parent)]!r}" in message
    )


def test_extern_checks_pointer_address_space() -> None:
    with pytest.raises(TlaCoreAPIError, match="expects pointer.*ub"):
        wrong_address_space_kernel.dump_mlir(type_args=(_fake_gm_tensor(),))


def test_extern_rejects_tensor_and_requests_explicit_pointer() -> None:
    with pytest.raises(TlaCoreAPIError, match=r"Tensor; pass tensor\.ptr explicitly"):
        tensor_argument_kernel.dump_mlir(type_args=(_fake_gm_tensor(),))


def test_extern_checks_call_region() -> None:
    with pytest.raises(TlaCoreAPIError, match="exactly one.*vector.*cube"):
        outside_vector_kernel.dump_mlir(type_args=(_fake_gm_tensor(),))
    with pytest.raises(TlaCoreAPIError, match="SIMT tla.vec.func"):
        inside_simt_func_kernel.dump_mlir(type_args=(tla.Int32(1),))
    lowered = _lower_kernel(inside_vec_func_kernel, _fake_gm_tensor())
    assert lowered.extern_compile_specs == (_compile_spec(EXTERN_SOURCE_CODE, {"aiv"}),)


def test_extern_infers_aic_and_mixed_call_targets() -> None:
    cube = _lower_kernel(extern_cube_kernel, tla.Int32(0))
    mixed = _lower_kernel(extern_mix_kernel, tla.Int32(0))

    assert cube.extern_compile_specs == (_compile_spec(DUAL_CORE_SOURCE_CODE, {"aic"}),)
    assert mixed.extern_compile_specs == (
        _compile_spec(DUAL_CORE_SOURCE_CODE, {"aic", "aiv"}),
    )
    assert mixed.asm().count("@tla_user_dual_core") == 2


def test_extern_groups_symbols_from_same_source_across_targets() -> None:
    lowered = _lower_kernel(shared_source_mixed_symbols_kernel, tla.Int32(0))

    assert lowered.extern_compile_specs == (
        _compile_spec(SHARED_EXTERN_SOURCE_CODE, {"aic", "aiv"}),
    )


def test_extern_checks_argument_count_and_scalar_type() -> None:
    with pytest.raises(TlaCoreAPIError, match="expects 3 arguments, got 2"):
        wrong_argument_count_kernel.dump_mlir(type_args=(_fake_gm_tensor(),))
    with pytest.raises(TlaCoreAPIError, match="expects Int32, got Int64"):
        wrong_scalar_type_kernel.dump_mlir(type_args=(_fake_gm_tensor(),))


def test_unused_extern_declaration_is_not_a_kernel_dependency() -> None:
    lowered = _lower_kernel(unused_extern_kernel, _fake_gm_tensor())

    assert lowered.extern_compile_specs == ()


def test_ascendc_compile_command_reuses_bitcode_backend(tmp_path, monkeypatch) -> None:
    compiler = tmp_path / "ccec"
    compiler.write_text("")
    commands = []
    user_a = tmp_path / "user-a"
    user_b = tmp_path / "user-b"
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    builtin_header = builtin / "kernel_operator.h"
    builtin_header.write_bytes(b"builtin")

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        source = tmp_path / "extern.0.cpp"
        output.write_bytes(b"BC")
        Path(command[command.index("-MF") + 1]).write_text(
            f"{output}: {source} {builtin_header}\n"
        )

    monkeypatch.setattr(execution, "_resolve_ccec", lambda: compiler)
    monkeypatch.setattr(execution, "_run_checked", fake_run)
    monkeypatch.setattr(
        execution, "_ascendc_builtin_include_dirs", lambda _root: [builtin]
    )
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))

    bitcodes, dependencies_by_target = execution._compile_ascendc_extern_source(
        gm_to_ub.source,
        source_index=0,
        artifact_dir=tmp_path,
        targets=(get_kernel_target(target_arch="c310", core_type="aiv"),),
        user_include_dirs=(user_a, user_a, user_b),
    )

    assert [path.name for path in bitcodes] == ["extern.0.aiv.c310.bc"]
    assert dependencies_by_target == {"aiv.c310": frozenset({builtin_header.resolve()})}
    command = commands[0]
    assert "-emit-llvm" in command
    assert "--cce-aicore-arch=dav-c310-vec" in command
    source = tmp_path / "extern.0.cpp"
    assert str(source) in command
    assert source.read_text() == EXTERN_SOURCE_CODE
    assert [command[index + 1] for index, arg in enumerate(command) if arg == "-I"] == [
        str(user_a),
        str(user_a),
        str(user_b),
        str(builtin),
    ]


def test_ascendc_multi_target_compile_uses_one_ccec_invocation_per_core(
    tmp_path, monkeypatch
) -> None:
    compiler = tmp_path / "ccec"
    compiler.write_text("")
    aic_header = tmp_path / "aic.h"
    aiv_header = tmp_path / "aiv.h"
    aic_header.write_bytes(b"aic")
    aiv_header.write_bytes(b"aiv")
    commands = []

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        source = tmp_path / "extern.1.cpp"
        dependency = (
            aic_header if "--cce-aicore-arch=dav-c310-cube" in command else aiv_header
        )
        output.write_bytes(b"BC")
        Path(command[command.index("-MF") + 1]).write_text(
            f"{output}: {source} {dependency}\n"
        )

    monkeypatch.setattr(execution, "_resolve_ccec", lambda: compiler)
    monkeypatch.setattr(execution, "_run_checked", fake_run)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    targets = execution._resolve_extern_targets(
        core_types={"aic", "aiv"},
        target_arch="c310",
    )
    user_include = tmp_path / "user-include"

    bitcodes, dependencies_by_target = execution._compile_ascendc_extern_source(
        tla_user_dual_core.source,
        source_index=1,
        artifact_dir=tmp_path,
        targets=targets,
        user_include_dirs=(user_include,),
    )

    assert [path.name for path in bitcodes] == [
        "extern.1.aic.c310.bc",
        "extern.1.aiv.c310.bc",
    ]
    assert len(commands) == 2
    assert "--cce-aicore-arch=dav-c310-cube" in commands[0]
    assert "--cce-aicore-arch=dav-c310-vec" in commands[1]
    assert commands[0][commands[0].index("-o") + 1].endswith("extern.1.aic.c310.bc")
    assert commands[1][commands[1].index("-o") + 1].endswith("extern.1.aiv.c310.bc")
    assert dependencies_by_target == {
        "aic.c310": frozenset({aic_header.resolve()}),
        "aiv.c310": frozenset({aiv_header.resolve()}),
    }
    for command in commands:
        assert command[command.index("-I") + 1] == str(user_include)


def test_extern_dependency_rescan_detects_new_earlier_header(
    tmp_path, monkeypatch
) -> None:
    early = tmp_path / "early"
    late = tmp_path / "late"
    late.mkdir()
    late_header = late / "cstdint"
    late_header.write_bytes(b"late")
    scan_paths: list[Path] = []

    def fake_run(command, **kwargs):
        del kwargs
        assert "-M" in command
        assert "-emit-llvm" not in command
        include_dirs = [
            Path(command[index + 1]) for index, arg in enumerate(command) if arg == "-I"
        ]
        selected_header = next(
            include_dir / "cstdint"
            for include_dir in include_dirs
            if (include_dir / "cstdint").exists()
        )
        source = next(Path(arg) for arg in command if arg.endswith(".cpp"))
        depfile = Path(command[command.index("-MF") + 1])
        scan_paths.extend((source, depfile))
        depfile.write_text(f"dependencies: {source} {selected_header}\n")

    monkeypatch.setattr(execution, "_resolve_ccec", lambda: tmp_path / "ccec")
    monkeypatch.setattr(execution, "_run_checked", fake_run)
    monkeypatch.setattr(execution, "_ascendc_builtin_include_dirs", lambda _root: [])
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    dependency_group = (
        execution._extern_source_sha256(EXTERN_SOURCE_CODE),
        "aiv.c310",
    )
    manifest = {
        "extern_dependency_groups": execution._snapshot_extern_dependency_groups(
            {dependency_group: frozenset({late_header})}
        )
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    extern_compile_specs = (
        _compile_spec(
            EXTERN_SOURCE_CODE,
            {"aiv"},
            include_dirs=(early, late),
        ),
    )
    assert (
        execution._load_current_extern_cache_manifest(
            manifest_path,
            cache_key="cache-key",
            target_arch="c310",
            extern_compile_specs=extern_compile_specs,
        )
        == manifest
    )
    assert all(not path.exists() for path in scan_paths)

    early.mkdir()
    early_header = early / "cstdint"
    early_header.write_bytes(b"early")

    assert (
        execution._load_current_extern_cache_manifest(
            manifest_path,
            cache_key="cache-key",
            target_arch="c310",
            extern_compile_specs=extern_compile_specs,
        )
        is None
    )


def test_extern_dependency_manifest_uses_hash_and_builtin_stat(
    tmp_path,
) -> None:
    header = tmp_path / "custom.h"
    header.write_bytes(b"first")
    builtin_dir = tmp_path / "builtin"
    builtin_dir.mkdir()
    builtin_header = builtin_dir / "kernel_operator.h"
    builtin_header.write_bytes(b"builtin")
    source_sha256 = execution._extern_source_sha256(EXTERN_SOURCE_CODE)
    dependency_groups = {
        (source_sha256, "aiv.c310"): frozenset({header, builtin_header})
    }
    manifest = {
        "extern_dependency_groups": execution._snapshot_extern_dependency_groups(
            dependency_groups,
            builtin_include_roots=[builtin_dir],
        )
    }
    snapshots = {
        snapshot["path"]: snapshot
        for snapshot in manifest["extern_dependency_groups"][0]["dependency_snapshots"]
    }
    assert set(snapshots[str(header)]) == {"path", "sha256"}
    assert set(snapshots[str(builtin_header)]) == {"path", "size", "mtime_ns"}
    original_stat = header.stat()
    assert execution._manifest_extern_dependency_groups_match_current_files(
        manifest,
        current_dependency_groups=dependency_groups,
    )

    header.write_bytes(b"other")
    os.utime(
        header,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    changed_stat = header.stat()
    assert changed_stat.st_size == original_stat.st_size
    assert changed_stat.st_mtime_ns == original_stat.st_mtime_ns
    assert not execution._manifest_extern_dependency_groups_match_current_files(
        manifest,
        current_dependency_groups=dependency_groups,
    )

    manifest = {
        "extern_dependency_groups": execution._snapshot_extern_dependency_groups(
            dependency_groups,
            builtin_include_roots=[builtin_dir],
        )
    }
    builtin_stat = builtin_header.stat()
    os.utime(
        builtin_header,
        ns=(builtin_stat.st_atime_ns, builtin_stat.st_mtime_ns + 1_000_000),
    )
    assert not execution._manifest_extern_dependency_groups_match_current_files(
        manifest,
        current_dependency_groups=dependency_groups,
    )


def test_extern_dependency_groups_preserve_target_association(tmp_path) -> None:
    aic_header = tmp_path / "aic.h"
    aiv_header = tmp_path / "aiv.h"
    aic_header.write_bytes(b"aic")
    aiv_header.write_bytes(b"aiv")
    source_sha256 = execution._extern_source_sha256(EXTERN_SOURCE_CODE)
    dependency_groups = {
        (source_sha256, "aic.c310"): frozenset({aic_header}),
        (source_sha256, "aiv.c310"): frozenset({aiv_header}),
    }
    manifest = {
        "extern_dependency_groups": execution._snapshot_extern_dependency_groups(
            dependency_groups
        )
    }

    assert execution._manifest_extern_dependency_groups_match_current_files(
        manifest,
        current_dependency_groups=dependency_groups,
    )
    assert not execution._manifest_extern_dependency_groups_match_current_files(
        manifest,
        current_dependency_groups={
            (source_sha256, "aic.c310"): frozenset({aiv_header}),
            (source_sha256, "aiv.c310"): frozenset({aic_header}),
        },
    )


def _cache_key_kwargs(tmp_path, target):
    return {
        "tlair_mlir": "module { func.func @kernel() }",
        "entrypoint": "kernel",
        "compile_option": execution.TlaCompileOption(arch_scope="aiv.c310"),
        "compiler_bridge_path": None,
        "hivmc": tmp_path / "hivmc-a5",
        "target": target,
    }


def test_extern_compile_environment_participates_in_kernel_cache_key(
    tmp_path, monkeypatch
) -> None:
    compiler = (tmp_path / "toolchain" / "bin" / "ccec").resolve()
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"ccec")
    compiler_version = ["first"]
    target = get_kernel_target(target_arch="c310", core_type="aiv")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.setattr(execution, "_resolve_ccec", lambda: compiler)
    monkeypatch.setattr(
        execution,
        "_tool_version",
        lambda path: compiler_version[0] if path == compiler else "hivmc-version",
    )
    monkeypatch.setattr(execution, "_tool_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(
        execution,
        "_ascendc_builtin_include_dirs",
        lambda _ascend_home: [tmp_path / "include"],
    )
    kwargs = _cache_key_kwargs(tmp_path, target)
    kwargs.update(extern_compile_specs=(_compile_spec(gm_to_ub.source, {"aiv"}),))
    first = execution._cache_key(**kwargs)

    compiler_version[0] = "second"

    assert execution._cache_key(**kwargs) != first


def test_extern_source_participates_in_kernel_cache_key(tmp_path, monkeypatch) -> None:
    target = get_kernel_target(target_arch="c310", core_type="aiv")
    monkeypatch.setattr(execution, "_tool_version", lambda _path: "version")
    monkeypatch.setattr(execution, "_tool_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(
        execution,
        "_ascendc_compile_environment_identity",
        lambda: {"revision": "same"},
    )
    kwargs = _cache_key_kwargs(tmp_path, target)

    first = execution._cache_key(
        **kwargs,
        extern_compile_specs=(_compile_spec('extern "C" void op() {}\n', {"aiv"}),),
    )
    second = execution._cache_key(
        **kwargs,
        extern_compile_specs=(
            _compile_spec('extern "C" void op() { /* changed */ }\n', {"aiv"}),
        ),
    )

    assert second != first


def test_extern_include_dirs_participate_in_kernel_cache_key(
    tmp_path, monkeypatch
) -> None:
    target = get_kernel_target(target_arch="c310", core_type="aiv")
    monkeypatch.setattr(execution, "_tool_version", lambda _path: "version")
    monkeypatch.setattr(execution, "_tool_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(
        execution,
        "_ascendc_compile_environment_identity",
        lambda: {"revision": "same"},
    )
    kwargs = _cache_key_kwargs(tmp_path, target)
    include_a = tmp_path / "a"
    include_b = tmp_path / "b"

    first = execution._cache_key(
        **kwargs,
        extern_compile_specs=(
            _compile_spec(EXTERN_SOURCE_CODE, {"aiv"}, (include_a, include_b)),
        ),
    )
    reordered = execution._cache_key(
        **kwargs,
        extern_compile_specs=(
            _compile_spec(EXTERN_SOURCE_CODE, {"aiv"}, (include_b, include_a)),
        ),
    )
    duplicated = execution._cache_key(
        **kwargs,
        extern_compile_specs=(
            _compile_spec(
                EXTERN_SOURCE_CODE,
                {"aiv"},
                (include_a, include_a, include_b),
            ),
        ),
    )

    assert reordered != first
    assert duplicated != first


def test_extern_cache_key_is_independent_of_source_iteration_order(
    tmp_path, monkeypatch
) -> None:
    target = get_kernel_target(target_arch="c310", core_type="aiv")
    monkeypatch.setattr(execution, "_tool_version", lambda _path: "version")
    monkeypatch.setattr(execution, "_tool_fingerprint", lambda _path: "fingerprint")
    monkeypatch.setattr(
        execution,
        "_ascendc_compile_environment_identity",
        lambda: {"revision": "same"},
    )
    kwargs = _cache_key_kwargs(tmp_path, target)
    specs = (
        _compile_spec(EXTERN_SOURCE_CODE, {"aiv"}),
        _compile_spec(DUAL_CORE_SOURCE_CODE, {"aic", "aiv"}),
    )

    first = execution._cache_key(**kwargs, extern_compile_specs=specs)
    second = execution._cache_key(**kwargs, extern_compile_specs=tuple(reversed(specs)))

    assert first == second


def _load_vecadd_example():
    return _load_module("extern_vecadd_example", EXAMPLE_PATH)


def _load_dual_core_example():
    return _load_module("extern_dual_core_example", DUAL_CORE_EXAMPLE_PATH)


def _load_multi_ops_example():
    return _load_module("extern_multi_ops_example", MULTI_OPS_EXAMPLE_PATH)


def test_extern_vecadd_frontend_uses_custom_load_and_tla_compute() -> None:
    example = _load_vecadd_example()
    tensor = _fake_gm_tensor()
    lowered = _lower_kernel(example.extern_vecadd, tensor, tensor, tensor)
    mlir = lowered.asm()
    assert mlir.count("tla.call_extern") == 2
    assert "tla.add" in mlir
    assert "tla.copy" in mlir
    assert lowered.extern_compile_specs == (
        _compile_spec(example.OP_SOURCE_CODES, {"aiv"}),
    )


def test_extern_dual_core_example_calls_one_op_from_aic_and_aiv() -> None:
    example = _load_dual_core_example()
    tensor = tla.make_fake_tensor(
        tla.Int32,
        (example.RESULT_SIZE,),
        (1,),
        origin_shape=(example.RESULT_SIZE,),
        coord=(0,),
        layout_tag=tla.arch.RowMajor,
    )
    lowered = _lower_kernel(example.extern_dual_core, tensor)
    mlir = lowered.asm()

    assert mlir.count("tla.call_extern") == 2
    assert mlir.count("@tla_user_store_i32") == 2
    assert lowered.extern_compile_specs == (
        _compile_spec(example.OP_SOURCE_CODES, {"aic", "aiv"}),
    )


def test_extern_multi_ops_example_tracks_two_aiv_sources_in_call_order() -> None:
    example = _load_multi_ops_example()
    tensor = _fake_gm_tensor()
    lowered = _lower_kernel(example.extern_multi_ops, tensor, tensor)
    mlir = lowered.asm()

    assert mlir.count("tla.call_extern") == 2
    assert "@tla_multi_gm_to_ub_f32" in mlir
    assert "@tla_multi_ub_to_gm_f32" in mlir
    assert lowered.extern_compile_specs == (
        _compile_spec(example.GM_TO_UB_SOURCE, {"aiv"}),
        _compile_spec(example.UB_TO_GM_SOURCE, {"aiv"}),
    )
