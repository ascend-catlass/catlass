"""Compilation and execution support for Tla DSL kernels."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import shlex
import struct
import subprocess
import sys
import tempfile
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from .base_dsl.arch import (
    DEFAULT_NPU_ARCH,
    TlaKernelTarget,
    arch_scope_for_target as _arch_scope_for_target_impl,
    get_kernel_target as _get_kernel_target,
    parse_arch_scope as _parse_arch_scope_impl,
    resolve_npu_arch as _resolve_npu_arch,
)
from .base_dsl import BaseDSL, DSLLocation
from .bc_compile import (
    get_bc_paths as _get_bc_paths,
)
from .base_dsl.typing import Numeric
from .compiler_bridge import (
    BridgeDiagnostic,
    BridgeLoweringError,
    BridgeSourceLocation,
    BridgeUnavailableError,
    KernelAbiArgumentKind,
    KernelAbiIntegerSignedness,
    KernelAbiLayout,
    KernelAbiScalarCategory,
    KernelAbiScalarDescriptor,
    kernel_abi_from_dict,
    kernel_abi_to_dict,
    lower_tlair_module_to_mlir,
    render_bridge_diagnostic,
    resolve_bridge_extension_path,
)
from .frontend_diagnostics import verbose_errors_enabled
from .types import dtype_size_bytes

if TYPE_CHECKING:
    from .base_dsl.jit_executor import JitCompiledFunction
    from .execution_lowering import ExternCompileSpec

# CATLASS_DSL_KEEP tokens: ir / ir-debug / kernel.
_KEEP_ALL_TOKENS: frozenset[str] = frozenset({"ir", "ir-debug", "kernel"})
_POINTER_ABI_SIZE = 8
_DYNAMIC_GM_MEMREF_FIELDS = (
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
_DYNAMIC_GM_MEMREF_PACKER = struct.Struct("<13Q")
_DEBUG_PRINT_WORKSPACE_SENTINEL_TEXT = b"TLA_PRNT"
_DEBUG_PRINT_WORKSPACE_SENTINEL = int.from_bytes(
    _DEBUG_PRINT_WORKSPACE_SENTINEL_TEXT, byteorder="big"
)
_PRINT_TENSOR_WORKSPACE_SENTINEL_TEXT = b"TLA_TPRN"
_PRINT_TENSOR_WORKSPACE_SENTINEL = int.from_bytes(
    _PRINT_TENSOR_WORKSPACE_SENTINEL_TEXT, byteorder="big"
)
_DEBUG_TUNNEL_STATE_SENTINEL = int.from_bytes(b"TLA_DTUN", byteorder="big")
# Cache compatibility tokens; change only when the corresponding ABI changes.
_DEBUG_PRINT_WORKSPACE_ABI_REVISION = "debug-print-workspace-i64-v1"
_PRINT_TENSOR_WORKSPACE_ABI_REVISION = "print-tensor-workspace-i64-dynamic-shape-typed-multirecord-mixed-subblock-l1-l0c-local-memref-abi"
_DEBUG_TUNNEL_STATE_ABI_REVISION = "debug-tunnel-state-i64-v1"
_PRINT_TENSOR_HELPER_ABI_MARKER = b"tla_print_tensor_abi"
_PRINT_TENSOR_MAX_BLOCKS = 1 << 16  # uint16_t block_idx
_PRINT_TENSOR_CORE_RECORDS = 108  # RuntimeWrapper::kDebugCoreRecords
_PRINT_TENSOR_FIFO_BYTES = 1 << 20  # kRingBufferBytes
_PRINT_TENSOR_SHAPE_HEADER_BYTES = 48  # sizeof(PrintShapeTlv)
_PRINT_TENSOR_HEADER_BYTES = 72  # sizeof(PrintTensorTlv)
_PRINT_TENSOR_PAYLOAD_ALIGNMENT = 32  # tensor payload alignment
_PRINT_TENSOR_MAX_F32_ELEMENTS = 262_112
_NATIVE_PRINT_TENSOR_RECORD_RE = re.compile(
    r"^DumpTensor: call=(?P<call>\d+), block=(?P<block>\d+), "
    r"(?:subblock=(?P<subblock>[01]), )?"
    r"data_type=(?P<dtype>[A-Za-z0-9_]+), "
    r"position=(?P<position>[A-Za-z0-9_]+), "
    r"shape=\[(?P<shape>[\d,\s]+)\] dump_size=(?P<count>\d+) "
    r"\[(?P<values>[^\]]*)\]$"
)
_FLOAT_LITERAL_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)$",
    re.IGNORECASE,
)
_INTEGER_LITERAL_RE = re.compile(r"^[+-]?\d+$")
_HIVM_TEMPLATE_BITCODE_ATTRS = {
    "meta_op.aic.c310.bc": ("hivm.aic_bitcode", "hivm.aic_bitcode"),
    "meta_op.aiv.c310.bc": ("hivm.aiv_bitcode", "hivm.aiv_bitcode"),
}
_MAX_KERNEL_ABI_PAYLOAD_SIZE = 1 << 20
_ONLINE_CACHE_ABI_VERSION = 8


class TlaExecutionError(RuntimeError):
    """Base class for execution and toolchain failures."""


class TlaCompilerBridgeUnavailableError(TlaExecutionError):
    """Raised when the typed Tla compiler bridge cannot be resolved."""


class TlaBackendCompilerNotFoundError(TlaExecutionError):
    """Raised when the backend compiler cannot be resolved."""


class TlaKernelCompileError(TlaExecutionError):
    """Raised when kernel lowering or backend compilation fails."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Sequence[BridgeDiagnostic] = (),
        pass_ir_dump: str = "",
        compiler_output: Sequence[str] = (),
        verbose_detail: str = "",
        diagnostic_input_mlir: str = "",
    ) -> None:
        super().__init__(message)
        self.diagnostics = tuple(diagnostics)
        self.pass_ir_dump = pass_ir_dump
        self.compiler_output = tuple(compiler_output)
        self.verbose_detail = verbose_detail
        self.diagnostic_input_mlir = diagnostic_input_mlir

    def format(self, *, verbose: bool = False) -> str:
        """Render compiler output, retaining bulky implementation detail on demand."""

        rendered = "\n".join(
            render_bridge_diagnostic(diagnostic, input_mlir=self.diagnostic_input_mlir)
            for diagnostic in self.diagnostics
        )
        if not rendered:
            rendered = "\n".join(
                output.rstrip() for output in self.compiler_output if output.rstrip()
            )
        if not rendered:
            rendered = str(self.args[0])
        if verbose:
            details = [
                detail.rstrip()
                for detail in (self.verbose_detail, self.pass_ir_dump)
                if detail
            ]
            if details:
                rendered += "\n\n" + "\n\n".join(details)
        return rendered

    def __str__(self) -> str:
        return self.format(verbose=verbose_errors_enabled())


class TlaRuntimeUnavailableError(TlaExecutionError):
    """Raised when Ascend runtime dependencies are unavailable."""


class TlaUnsupportedAbiError(TlaExecutionError):
    """Raised when attempting to execute an unsupported ABI shape."""


@dataclass(frozen=True)
class _PrintTensorDTypeSpec:
    mlir_token: str
    native_name: str
    numeric_kind: str
    minimum: int | None = None
    maximum: int | None = None


@dataclass(frozen=True)
class _PrintTensorMetadata:
    shape: tuple[int, ...]
    count: int | None
    dtype: str
    position: str
    call: int = 0
    dynamic_execution: bool = False


@dataclass(frozen=True)
class _NativePrintTensorRecord:
    native_dtype: str
    position: str
    declared_count: int
    values_text: str
    shape: tuple[int, ...]
    call: int
    block: int
    subblock: int | None


_PRINT_TENSOR_DTYPES = {
    "f16": _PrintTensorDTypeSpec("f16", "float16", "float"),
    "f32": _PrintTensorDTypeSpec("f32", "float32", "float"),
    "i8": _PrintTensorDTypeSpec("i8", "int8", "integer", -128, 127),
    "i16": _PrintTensorDTypeSpec("i16", "int16", "integer", -32768, 32767),
    "i32": _PrintTensorDTypeSpec("i32", "int32", "integer", -2147483648, 2147483647),
    "u8": _PrintTensorDTypeSpec("ui8", "uint8", "integer", 0, 255),
    "u16": _PrintTensorDTypeSpec("ui16", "uint16", "integer", 0, 65535),
    "u32": _PrintTensorDTypeSpec("ui32", "uint32", "integer", 0, 4294967295),
}
_PRINT_TENSOR_DTYPES_BY_MLIR = {
    spec.mlir_token: token for token, spec in _PRINT_TENSOR_DTYPES.items()
}
_PRINT_TENSOR_DTYPES_BY_NATIVE = {
    spec.native_name: token for token, spec in _PRINT_TENSOR_DTYPES.items()
}


@dataclass(frozen=True)
class TlaExecutionResult:
    binary_handle: int
    function_handle: int
    device: int


@dataclass(frozen=True)
class TlaCompileOption:
    cache_enabled: bool = True
    cache_dir: Path | None = None
    force_recompile: bool = False
    kernel_mode: str = "aiv"
    # Placeholder until ``_resolve_compile_option_from_lowered_mlir``; derived from
    # public ``--npu-arch`` default (not a separate Host knob).
    arch_scope: str = _arch_scope_for_target_impl(
        target_arch=_resolve_npu_arch(DEFAULT_NPU_ARCH),
        core_type="aiv",
    )
    #: When True, dump IR across the lowering pipeline (env ``CATLASS_DSL_PRINT_IR``).
    print_ir: bool = False
    # Unified Buffer the compiler holds back from the kernel, and which these
    # release. Both are spelled exactly as bisheng spells them, since that is
    # where they are ultimately understood.
    #
    # The reserve is 8 KB in total: 2 KB for Ascend C and 6 KB of VF stack that
    # the compiler spills vector registers into. Releasing either hands those
    # bytes to the kernel, and releasing the VF stack means the kernel must not
    # provoke a spill -- nothing checks, and the consequence is a silent write
    # into whatever sits above.
    cce_disable_asc_reserved_ubuf: bool = False
    cce_disable_vf_stack_reserved_ubuf: bool = False


@dataclass(frozen=True)
class _PreparedAbiSlot:
    argument: Any
    logical_index: int
    start: int
    end: int


@dataclass(frozen=True)
class _PreparedMemrefRun:
    logical_index: int
    start: int
    packer: struct.Struct
    value_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _PreparedAbiPacker:
    logical_count: int
    host_size: int
    slots: tuple[_PreparedAbiSlot, ...]
    memref_runs: tuple[_PreparedMemrefRun, ...]


@dataclass(frozen=True)
class _LogicalMixedHandoff:
    entrypoint: str


@dataclass(frozen=True)
class _ArtifactStaticMetadata:
    uses_scalar_print: bool
    uses_tensor_print: bool
    logical_mixed_handoff: _LogicalMixedHandoff | None


_MEMORY_COMPILE_CACHE_LOCK = threading.RLock()
_MEMORY_COMPILE_CACHE: dict[str, JitCompiledFunction] = {}
_NATIVE_PRINT_TENSOR_STDOUT_LOCK = threading.RLock()


#: Compile options that carry no value: naming one turns it on.
_VALUELESS_COMPILE_OPTIONS = frozenset(
    {
        "cce_disable_asc_reserved_ubuf",
        "cce_disable_vf_stack_reserved_ubuf",
    }
)


def _parse_compile_options_from_str(options: Any) -> dict[str, str]:
    """Parse an ``options="--key value"`` token string.

    Most options are a key and a value. The few in
    ``_VALUELESS_COMPILE_OPTIONS`` are switches instead: writing one turns it
    on, and it takes no value.
    """
    if options is None:
        return {}
    if not isinstance(options, str):
        raise TypeError(
            "compile options must be a string "
            '(e.g. options="--npu-arch 3510"), '
            f"got {type(options).__name__}"
        )
    text = options.strip()
    if not text:
        return {}
    tokens = shlex.split(text)
    parsed: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--") and "=" in token:
            key, value = token[2:].split("=", 1)
            parsed[key.replace("-", "_")] = value
            index += 1
            continue
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if key in _VALUELESS_COMPILE_OPTIONS:
                parsed[key] = ""
                index += 1
                continue
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("-"):
                raise ValueError(f"Missing value for compile option {token!r}")
            parsed[key] = tokens[index + 1]
            index += 2
            continue
        raise ValueError(f"Unexpected token in compile options: {token!r}")
    return parsed


def _compile_kernel(
    fn: Any,
    *,
    kind: str,
    options: Mapping[str, Any],
    compile_option: TlaCompileOption,
    type_args: Sequence[Any] | None = None,
    decorator_location: DSLLocation | None = None,
) -> "JitCompiledFunction":
    lowered = BaseDSL()._lower(
        fn,
        kind=kind,
        options=dict(options),
        type_args=type_args,
        location=decorator_location,
    )
    extern_compile_specs = lowered.extern_compile_specs
    tlair_mlir = lowered.asm(generic=True)
    entrypoint = _extract_entrypoint(tlair_mlir)
    compiler_bridge_path = resolve_bridge_extension_path()
    hivmc = _resolve_hivmc_a5()
    target = _resolve_kernel_target(compile_option)
    cache_dir = compile_option.cache_dir or _default_cache_dir()
    lowered_argument_trees = lowered.argument_trees
    original_signature = BaseDSL()._get_signature(fn)
    cache_key = _cache_key(
        tlair_mlir=tlair_mlir,
        entrypoint=entrypoint,
        compile_option=compile_option,
        compiler_bridge_path=compiler_bridge_path,
        hivmc=hivmc,
        target=target,
        extern_compile_specs=extern_compile_specs,
    )
    artifact_dir = cache_dir / cache_key
    manifest = artifact_dir / "manifest.json"

    can_reuse_cache = (
        compile_option.cache_enabled and not compile_option.force_recompile
    )
    current_extern_manifest = None
    # For kernels with tla.extern, check if current cache is valid, load it or recompile if not.
    if can_reuse_cache and extern_compile_specs:
        current_extern_manifest = _load_current_extern_cache_manifest(
            manifest,
            cache_key=cache_key,
            target_arch=target.target_arch,
            extern_compile_specs=extern_compile_specs,
        )
        can_reuse_cache = current_extern_manifest is not None

    # For all kernels, check the memory cache, and reuse it if valid.
    if can_reuse_cache:
        cached_memory = _get_memory_cached_function(cache_key)
        if cached_memory is not None:
            compiled = _new_jit_compiled_function_from_cached(
                cached_memory,
                argument_trees=lowered_argument_trees,
                original_signature=original_signature,
            )
            _set_memory_cached_function(compiled)
            _copy_kept_artifacts(compiled)
            return compiled

    # For kernels with tla.extern, reuse the above cache (or None);
    # For other kernels, load the manifest if it exists.
    if can_reuse_cache and manifest.exists():
        cached = (
            current_extern_manifest
            if current_extern_manifest is not None
            else _load_manifest(manifest)
        )
        kernel_binary_path = artifact_dir / str(cached["kernel_binary"])
        mlir_path = artifact_dir / str(cached["lowered_mlir"])
        cached_pass_dump = cached.get("pass_ir_dump")
        pass_dump_path = (
            artifact_dir / str(cached_pass_dump) if cached_pass_dump else None
        )
        if (
            _cache_manifest_has_current_workspace_abis(cached)
            and kernel_binary_path.exists()
            and mlir_path.exists()
        ):
            lowered_llvm = mlir_path.read_text()
            static_metadata = _analyze_compiled_artifact_static_metadata(
                tlair_mlir, lowered_llvm
            )
            expected_entrypoint = (
                static_metadata.logical_mixed_handoff.entrypoint
                if static_metadata.logical_mixed_handoff is not None
                else entrypoint
            )
            cached_kernel_abi, cached_abi_packer = _prepared_kernel_abi_from_manifest(
                cached, expected_entrypoint=expected_entrypoint
            )
            compiled = _new_jit_compiled_function(
                cache_key=cache_key,
                cache_dir=artifact_dir,
                tlair_mlir=tlair_mlir,
                lowered_llvm=lowered_llvm,
                entrypoint=entrypoint,
                compiler_bridge_path=compiler_bridge_path,
                hivmc_path=hivmc,
                kernel_binary_path=kernel_binary_path,
                compile_option=_resolve_compile_option_from_lowered_mlir(
                    compile_option,
                    lowered_llvm,
                    has_logical_mixed_handoff=(
                        static_metadata.logical_mixed_handoff is not None
                    ),
                ),
                pass_ir_dump=pass_dump_path.read_text()
                if pass_dump_path and pass_dump_path.exists()
                else "",
                kernel_abi=cached_kernel_abi,
                abi_packer=cached_abi_packer,
                uses_scalar_print=static_metadata.uses_scalar_print,
                uses_tensor_print=static_metadata.uses_tensor_print,
                logical_mixed_handoff=static_metadata.logical_mixed_handoff,
                ub_static_bytes=int(cached.get("ub_static_bytes", 0)),
                ub_programmable_bytes=int(cached.get("ub_programmable_bytes", 0)),
                ub_dynamic_base=int(cached.get("ub_dynamic_base", -1)),
                argument_trees=lowered_argument_trees,
                original_signature=original_signature,
            )
            _set_memory_cached_function(compiled)
            _copy_kept_artifacts(compiled)
            return compiled

    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = artifact_dir / "lowered.mlir"
    pass_dump_path = artifact_dir / "pass-ir-dump.mlir"
    kernel_binary_path = artifact_dir / "kernel.o"

    try:
        lowering_result = _run_tla_lowering_to_mlir(
            lowered_module=lowered.module,
            tlair_mlir=tlair_mlir,
            mlir_path=mlir_path,
            compile_option=compile_option,
        )
    except TlaKernelCompileError as exc:
        if exc.pass_ir_dump:
            _write_pass_ir_dump(pass_dump_path, exc.pass_ir_dump)
            raise TlaKernelCompileError(
                str(exc),
                diagnostics=exc.diagnostics,
                pass_ir_dump=exc.pass_ir_dump,
                compiler_output=exc.compiler_output,
                verbose_detail="\n".join(
                    detail
                    for detail in (
                        exc.verbose_detail,
                        f"pass IR dump: {pass_dump_path}",
                    )
                    if detail
                ),
                diagnostic_input_mlir=exc.diagnostic_input_mlir,
            ) from None
        raise
    if lowering_result.pass_ir_dump:
        _write_pass_ir_dump(pass_dump_path, lowering_result.pass_ir_dump)
    lowered_llvm = mlir_path.read_text()
    static_metadata = _analyze_compiled_artifact_static_metadata(
        tlair_mlir, lowered_llvm
    )
    resolved_option = _resolve_compile_option_from_lowered_mlir(
        compile_option,
        lowered_llvm,
        has_logical_mixed_handoff=(static_metadata.logical_mixed_handoff is not None),
    )
    kernel_abi = getattr(lowering_result, "kernel_abi", None)
    # Already accounts for a released reserve: the module is stamped before
    # lowering, so the pass and the host agree on one number.
    ub_programmable_bytes = lowering_result.ub_programmable_bytes
    expected_entrypoint = (
        static_metadata.logical_mixed_handoff.entrypoint
        if static_metadata.logical_mixed_handoff is not None
        else entrypoint
    )
    abi_packer = _prepare_compiled_abi_packer(
        kernel_abi, expected_entrypoint=expected_entrypoint
    )
    hivmc_mlir_path, template_bitcode = _create_stamped_hivmc_input(
        mlir_path, resolved_option
    )
    user_bitcodes: list[Path] = []
    # Group dependencies by source content and compile target, because even if
    # the same source is compiled for multiple targets, the dependencies may differ.
    # For example:
    # {
    #     ("<source-sha256>", "aic.c310"): frozenset({Path("/path/to/aic.h")}),
    #     ("<source-sha256>", "aiv.c310"): frozenset({Path("/path/to/aiv.h")}),
    # }
    extern_dependency_groups: dict[tuple[str, str], frozenset[Path]] = {}
    for source_index, spec in enumerate(extern_compile_specs):
        bitcodes, dependencies_by_target = _compile_ascendc_extern_source(
            spec.source,
            source_index=source_index,
            artifact_dir=artifact_dir,
            targets=_resolve_extern_targets(
                core_types=spec.core_types,
                target_arch=target.target_arch,
            ),
            user_include_dirs=spec.include_dirs,
        )
        user_bitcodes.extend(bitcodes)
        source_sha256 = _extern_source_sha256(spec.source)
        for target_name, dependencies in dependencies_by_target.items():
            extern_dependency_groups[(source_sha256, target_name)] = dependencies
    if user_bitcodes:
        if template_bitcode is None:
            template_bitcode = _resolve_hivm_template_bitcode(resolved_option)
        template_bitcode = ",".join(
            (template_bitcode, *(str(path) for path in user_bitcodes))
        )
    hivmc_input_mlir = hivmc_mlir_path.read_text()
    try:
        _run_checked(
            _build_hivmc_a5_command(
                compiler=hivmc,
                mlir_path=hivmc_mlir_path,
                kernel_binary_path=kernel_binary_path,
                compile_option=resolved_option,
                template_bitcode=template_bitcode,
            ),
            label="hivmc-a5",
            cwd=artifact_dir,
            diagnostic_input_mlir=hivmc_input_mlir,
        )
    finally:
        if hivmc_mlir_path != mlir_path:
            hivmc_mlir_path.unlink(missing_ok=True)
    if not kernel_binary_path.exists():
        raise TlaKernelCompileError(
            "hivmc-a5 completed but output kernel artifact was not "
            f"created at {kernel_binary_path}"
        )

    manifest_data = {
        "cache_key": cache_key,
        "debug_print_workspace_abi_revision": (_DEBUG_PRINT_WORKSPACE_ABI_REVISION),
        "print_tensor_workspace_abi_revision": (_PRINT_TENSOR_WORKSPACE_ABI_REVISION),
        "debug_tunnel_state_abi_revision": _DEBUG_TUNNEL_STATE_ABI_REVISION,
        "entrypoint": entrypoint,
        "kernel_binary": kernel_binary_path.name,
        "lowered_mlir": mlir_path.name,
        "pass_ir_dump": pass_dump_path.name if lowering_result.pass_ir_dump else None,
        "compiler_bridge": (
            str(compiler_bridge_path) if compiler_bridge_path else None
        ),
        "hivmc": str(hivmc),
        "arch_scope": resolved_option.arch_scope,
        "kernel_abi": kernel_abi_to_dict(kernel_abi),
        # Read back on a cache hit, where lowering never runs to produce them.
        "ub_static_bytes": lowering_result.ub_static_bytes,
        "ub_programmable_bytes": ub_programmable_bytes,
        "ub_dynamic_base": lowering_result.ub_dynamic_base,
    }
    if extern_compile_specs:
        _, builtin_include_roots = _ascendc_compiler_inputs()
        manifest_data["extern_dependency_groups"] = _snapshot_extern_dependency_groups(
            extern_dependency_groups,
            builtin_include_roots=builtin_include_roots,
        )
    manifest.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n")
    compiled = _new_jit_compiled_function(
        cache_key=cache_key,
        cache_dir=artifact_dir,
        tlair_mlir=tlair_mlir,
        lowered_llvm=lowered_llvm,
        entrypoint=entrypoint,
        compiler_bridge_path=compiler_bridge_path,
        hivmc_path=hivmc,
        kernel_binary_path=kernel_binary_path,
        pass_ir_dump=lowering_result.pass_ir_dump,
        compile_option=resolved_option,
        kernel_abi=kernel_abi,
        abi_packer=abi_packer,
        uses_scalar_print=static_metadata.uses_scalar_print,
        uses_tensor_print=static_metadata.uses_tensor_print,
        logical_mixed_handoff=static_metadata.logical_mixed_handoff,
        ub_static_bytes=lowering_result.ub_static_bytes,
        ub_programmable_bytes=ub_programmable_bytes,
        ub_dynamic_base=lowering_result.ub_dynamic_base,
        argument_trees=lowered_argument_trees,
        original_signature=original_signature,
    )
    if compile_option.cache_enabled:
        _set_memory_cached_function(compiled)
    _copy_kept_artifacts(compiled)
    return compiled


def compile_and_cache(
    fn: Any,
    *,
    kind: str,
    options: Mapping[str, Any],
    compile_option: TlaCompileOption,
    type_args: Sequence[Any] | None = None,
    decorator_location: DSLLocation | None = None,
) -> "JitCompiledFunction":
    """Compile or load one kernel and return an independent JIT owner."""
    return _compile_kernel(
        fn,
        kind=kind,
        options=options,
        compile_option=compile_option,
        type_args=type_args,
        decorator_location=decorator_location,
    )


def _new_jit_compiled_function(
    *,
    cache_key: str,
    cache_dir: Path,
    tlair_mlir: str,
    lowered_llvm: str,
    entrypoint: str,
    compiler_bridge_path: Path | None,
    hivmc_path: Path,
    kernel_binary_path: Path,
    compile_option: TlaCompileOption,
    pass_ir_dump: str,
    kernel_abi: KernelAbiLayout | None,
    abi_packer: _PreparedAbiPacker | None,
    uses_scalar_print: bool,
    uses_tensor_print: bool,
    logical_mixed_handoff: _LogicalMixedHandoff | None,
    ub_static_bytes: int = 0,
    ub_programmable_bytes: int = 0,
    ub_dynamic_base: int = -1,
    argument_trees: Sequence[Any | None] | None = None,
    original_signature: inspect.Signature | None = None,
) -> "JitCompiledFunction":
    """Build a compiled function from validated, device-independent state."""
    from .base_dsl.jit_executor import (
        ExecutionArgs,
        JitCompiledFunction,
        JitFunctionArtifacts,
        JitModule,
    )

    if abi_packer is None:
        raise TlaUnsupportedAbiError(
            "kernel ABI was not validated while constructing the compiled function"
        )

    is_mixed = logical_mixed_handoff is not None and compile_option.kernel_mode == "mix"
    print_metadata = (
        _print_tensor_static_metadata_records(tlair_mlir, entrypoint=entrypoint)
        if uses_tensor_print
        else None
    )
    print_helper_core = (
        _mixed_print_tensor_helper_core(lowered_llvm)
        if uses_tensor_print and compile_option.kernel_mode == "mix"
        else (
            _core_type_from_arch_scope(compile_option.arch_scope)
            if uses_tensor_print
            else None
        )
    )
    local_print_positions = {
        metadata.position
        for metadata in (print_metadata or ())
        if metadata.position in {"L1", "L0C"}
    }
    if len(local_print_positions) > 1:
        raise TlaExecutionError("one kernel cannot mix L1 and L0C tensor prints")
    print_tensor_position = (
        next(iter(local_print_positions)) if len(local_print_positions) == 1 else None
    )

    return JitCompiledFunction(
        jit_module=JitModule(
            kernel_binary_path=kernel_binary_path,
            entrypoint=entrypoint,
            kernel_mode=compile_option.kernel_mode,
            execution_args=ExecutionArgs(
                original_signature=original_signature,
                kernel_abi=kernel_abi,
                abi_packer=abi_packer,
                argument_trees=(
                    None if argument_trees is None else tuple(argument_trees)
                ),
            ),
            uses_scalar_print=uses_scalar_print,
            uses_tensor_print=uses_tensor_print,
            is_mixed=is_mixed,
            print_metadata=print_metadata,
            print_helper_core=print_helper_core,
            print_tensor_position=print_tensor_position,
            ub_static_bytes=ub_static_bytes,
            ub_programmable_bytes=ub_programmable_bytes,
            ub_dynamic_base=ub_dynamic_base,
        ),
        artifacts=JitFunctionArtifacts(
            MLIR=tlair_mlir,
            LLVM=lowered_llvm,
            BINARY=kernel_binary_path,
            PASS_IR_DUMP=pass_ir_dump or None,
            cache_key=cache_key,
            cache_dir=cache_dir,
            compiler_bridge_path=compiler_bridge_path,
            hivmc_path=hivmc_path,
        ),
    )


def _new_jit_compiled_function_from_cached(
    cached: "JitCompiledFunction",
    *,
    argument_trees: Sequence[Any | None],
    original_signature: inspect.Signature,
) -> "JitCompiledFunction":
    """Reuse compiled artifacts while binding the current Python arguments."""
    from .base_dsl.jit_executor import JitCompiledFunction

    execution_args = replace(
        cached.execution_args,
        original_signature=original_signature,
        signature=None,
        argument_trees=tuple(argument_trees),
    )
    return JitCompiledFunction(
        jit_module=replace(cached.jit_module, execution_args=execution_args),
        artifacts=cached.artifacts,
    )


def _get_memory_cached_function(cache_key: str) -> "JitCompiledFunction | None":
    with _MEMORY_COMPILE_CACHE_LOCK:
        compiled = _MEMORY_COMPILE_CACHE.get(cache_key)
    if compiled is None:
        return None
    if not compiled.kernel_binary_path.exists():
        _drop_memory_cached_function(cache_key)
        return None
    return compiled


def _set_memory_cached_function(compiled: "JitCompiledFunction") -> None:
    with _MEMORY_COMPILE_CACHE_LOCK:
        _MEMORY_COMPILE_CACHE[compiled.cache_key] = compiled


def _drop_memory_cached_function(cache_key: str) -> None:
    with _MEMORY_COMPILE_CACHE_LOCK:
        _MEMORY_COMPILE_CACHE.pop(cache_key, None)


def _checked_print_tensor_block_count(block_num: int) -> int:
    block_count = int(block_num)
    if block_count <= 0 or block_count > _PRINT_TENSOR_MAX_BLOCKS:
        raise TlaExecutionError(
            "tla.print_tensor 16-bit block identity requires a positive "
            f"block_num with at most {_PRINT_TENSOR_MAX_BLOCKS} blocks; "
            f"got {block_num}"
        )
    return block_count


def _print_tensor_native_wire_bytes(metadata: _PrintTensorMetadata) -> int:
    count = _PRINT_TENSOR_MAX_F32_ELEMENTS if metadata.count is None else metadata.count
    element_bytes = dtype_size_bytes(metadata.dtype)
    if element_bytes <= 0:
        raise TlaExecutionError(
            "tla.print_tensor FIFO capacity calculation encountered "
            f"unsupported dtype {metadata.dtype!r}"
        )
    payload_bytes = count * element_bytes
    aligned_payload_bytes = (
        payload_bytes + _PRINT_TENSOR_PAYLOAD_ALIGNMENT - 1
    ) // _PRINT_TENSOR_PAYLOAD_ALIGNMENT
    return (
        _PRINT_TENSOR_SHAPE_HEADER_BYTES
        + _PRINT_TENSOR_HEADER_BYTES
        + aligned_payload_bytes * _PRINT_TENSOR_PAYLOAD_ALIGNMENT
    )


def _validate_print_tensor_fifo_capacity(
    metadata: Sequence[_PrintTensorMetadata],
    block_num: int,
    *,
    helper_core: str,
    mixed: bool,
) -> None:
    block_count = _checked_print_tensor_block_count(block_num)
    records_per_block = 2 if helper_core == "aiv" and mixed else 1
    core_records = block_count * records_per_block
    if core_records > _PRINT_TENSOR_CORE_RECORDS:
        raise TlaExecutionError(
            "tla.print_tensor fixed 1 MiB FIFO workspace supports at most "
            f"{_PRINT_TENSOR_CORE_RECORDS} core records; got {core_records} "
            f"from {block_count} blocks"
        )
    static_bytes = 0
    dynamic_sites = 0
    for item in metadata:
        record_bytes = _print_tensor_native_wire_bytes(item)
        if item.dynamic_execution or item.count is None:
            dynamic_sites += 1
        else:
            static_bytes += record_bytes
        if item.count is not None and record_bytes > _PRINT_TENSOR_FIFO_BYTES:
            raise TlaExecutionError(
                "tla.print_tensor record exceeds the per-print FIFO capacity: "
                f"{record_bytes} bytes, capacity is {_PRINT_TENSOR_FIFO_BYTES} bytes"
            )
    if static_bytes > _PRINT_TENSOR_FIFO_BYTES:
        raise TlaExecutionError(
            "statically executed tla.print_tensor aggregate output exceeds the per-core "
            f"FIFO capacity: {static_bytes} bytes, capacity is "
            f"{_PRINT_TENSOR_FIFO_BYTES} bytes"
        )
    if dynamic_sites:
        warnings.warn(
            "tla.print_tensor uses runtime length, dynamic shape, or runtime "
            "control flow; aggregate use of the fixed 1 MiB per-core FIFO "
            "cannot be validated and excessive output may be truncated or omitted",
            RuntimeWarning,
            stacklevel=3,
        )


def _runtime_arg_values(arg: Any) -> list[int]:
    try:
        if hasattr(arg, "__c_pointers__"):
            return [int(ptr) for ptr in arg.__c_pointers__()]
        data_ptr = getattr(arg, "data_ptr", None)
        if callable(data_ptr):
            return [int(data_ptr())]
        if data_ptr is not None:
            return [int(data_ptr)]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TlaUnsupportedAbiError(
            "Launch argument cannot provide exactly one concrete host value."
        ) from exc
    raise TlaUnsupportedAbiError(
        "Launch arguments must provide exactly one host value."
    )


def _align_payload(payload: bytearray, alignment: int) -> None:
    payload.extend(b"\0" * (-len(payload) % alignment))


def _scalar_storage_size(descriptor: KernelAbiScalarDescriptor) -> int:
    if descriptor.category is KernelAbiScalarCategory.INDEX:
        return 8
    return max(1, descriptor.bit_width // 8)


def _pack_scalar_argument(
    value: Any,
    descriptor: KernelAbiScalarDescriptor,
    mlir_type: str,
    storage_size: int,
) -> bytes:
    expected_size = _scalar_storage_size(descriptor)
    if storage_size != expected_size:
        raise TlaUnsupportedAbiError(
            f"kernel ABI scalar {mlir_type} declares unsupported storage size "
            f"{storage_size}; expected {expected_size}"
        )
    if not isinstance(value, Numeric):
        value_type = type(value)
        if (
            value_type is bool
            and descriptor.category is KernelAbiScalarCategory.INTEGER
            and descriptor.bit_width == 1
        ):
            return bytes((int(value),))
        if (
            value_type is int
            and descriptor.category is KernelAbiScalarCategory.INTEGER
            and descriptor.bit_width == 32
            and descriptor.integer_signedness is KernelAbiIntegerSignedness.SIGNLESS
        ):
            lower, upper = -(1 << 31), (1 << 31) - 1
            if value < lower or value > upper:
                raise TlaUnsupportedAbiError(
                    f"scalar value for {mlir_type} does not fit in its declared type"
                )
            return value.to_bytes(storage_size, byteorder="little", signed=True)
        if (
            value_type is float
            and descriptor.category is KernelAbiScalarCategory.FLOAT
            and descriptor.float_format is not None
            and descriptor.float_format.value == "f32"
        ):
            try:
                return struct.pack("<f", value)
            except (OverflowError, struct.error) as exc:
                raise TlaUnsupportedAbiError(
                    f"scalar value for {mlir_type} does not fit in its declared type"
                ) from exc
        # Numeric subclasses use their base value. Keep primitive values allocation-free.
        if value_type is not int and value_type is not bool and isinstance(value, int):
            return _pack_scalar_argument(
                int(value), descriptor, mlir_type, storage_size
            )
        if value_type is not float and isinstance(value, float):
            return _pack_scalar_argument(
                float(value), descriptor, mlir_type, storage_size
            )
        raise TlaUnsupportedAbiError(
            f"plain Python {value_type.__name__} does not match kernel ABI type "
            f"{mlir_type}"
        )
    host_type = type(value)
    host_dtype = str(getattr(host_type, "dtype", "")).lower()
    host_is_float = bool(getattr(host_type, "is_float", False))
    type_matches = host_is_float == (
        descriptor.category is KernelAbiScalarCategory.FLOAT
    )
    if descriptor.category is KernelAbiScalarCategory.FLOAT:
        if descriptor.float_format is None:
            raise TlaUnsupportedAbiError(
                f"kernel ABI float scalar {mlir_type} has no format"
            )
        type_matches = type_matches and host_dtype == descriptor.float_format.value
    elif descriptor.category is KernelAbiScalarCategory.INDEX:
        type_matches = type_matches and host_dtype == "index"
    else:
        type_matches = (
            type_matches and int(getattr(host_type, "width", 0)) == descriptor.bit_width
        )
        if descriptor.integer_signedness is KernelAbiIntegerSignedness.SIGNED:
            type_matches = type_matches and bool(getattr(host_type, "signed", False))
        elif descriptor.integer_signedness is KernelAbiIntegerSignedness.UNSIGNED:
            type_matches = type_matches and not bool(getattr(host_type, "signed", True))
    if not type_matches:
        raise TlaUnsupportedAbiError(
            f"typed host scalar {host_dtype or type(value).__name__} does not match "
            f"kernel ABI type {mlir_type}"
        )
    if descriptor.category is not KernelAbiScalarCategory.FLOAT:
        host_value = getattr(value, "value", None)
        if not isinstance(host_value, (bool, int)):
            raise TlaUnsupportedAbiError(
                f"kernel ABI scalar {mlir_type} requires a concrete integer host value"
            )
        width = int(getattr(host_type, "width", storage_size * 8))
        if width == 1:
            lower, upper = 0, 1
        elif bool(getattr(host_type, "signed", False)):
            lower, upper = -(1 << (width - 1)), (1 << (width - 1)) - 1
        else:
            lower, upper = 0, (1 << width) - 1
        if int(host_value) < lower or int(host_value) > upper:
            raise TlaUnsupportedAbiError(
                f"scalar value for {mlir_type} does not fit in its declared type"
            )
    values = _runtime_arg_values(value)
    if len(values) != 1:
        raise TlaUnsupportedAbiError(
            "each logical launch argument must provide exactly one host value"
        )
    bits = int(values[0])
    if bits < 0 or bits >= (1 << (storage_size * 8)):
        raise TlaUnsupportedAbiError(
            f"scalar value for {mlir_type} does not fit in {storage_size} bytes"
        )
    return bits.to_bytes(storage_size, byteorder="little", signed=False)


def _logical_launch_arg_count(layout: KernelAbiLayout) -> int:
    if not layout.arguments:
        return 0
    return 1 + max(
        (argument.index if argument.logical_index is None else argument.logical_index)
        for argument in layout.arguments
    )


def _prepare_abi_packer(
    layout: KernelAbiLayout | None, *, expected_entrypoint: str | None = None
) -> _PreparedAbiPacker:
    if expected_entrypoint is None:
        _validate_kernel_abi_layout(layout)
    else:
        _validate_kernel_abi_layout(layout, expected_entrypoint=expected_entrypoint)
    if layout is None:
        raise TlaUnsupportedAbiError("kernel ABI layout is missing")
    logical_count = _logical_launch_arg_count(layout)
    host_offsets: list[int] = []
    host_size = 0
    for argument in layout.arguments:
        host_alignment = 8 if argument.storage_size == 8 else 4
        host_size = (
            (host_size + host_alignment - 1) // host_alignment
        ) * host_alignment
        host_offsets.append(host_size)
        host_size += argument.storage_size
    host_size = ((host_size + 7) // 8) * 8
    if host_size > _MAX_KERNEL_ABI_PAYLOAD_SIZE:
        raise TlaUnsupportedAbiError(
            "kernel ABI host payload exceeds the supported maximum size"
        )
    all_slots = []
    for index, argument in enumerate(layout.arguments):
        logical_index = (
            argument.index if argument.logical_index is None else argument.logical_index
        )
        start = host_offsets[index]
        end = start + argument.storage_size
        all_slots.append(_PreparedAbiSlot(argument, logical_index, start, end))

    slots: list[_PreparedAbiSlot] = []
    memref_runs: list[_PreparedMemrefRun] = []
    memref_logical_indices: set[int] = set()
    index = 0
    while index < len(all_slots):
        slot = all_slots[index]
        if slot.argument.kind is not KernelAbiArgumentKind.MEMREF_FIELD:
            slots.append(slot)
            index += 1
            continue
        fields = []
        run_start = slot.start
        logical_index = slot.logical_index
        if logical_index in memref_logical_indices:
            raise TlaUnsupportedAbiError(
                "memref fields must form one contiguous group per logical argument"
            )
        memref_logical_indices.add(logical_index)
        previous_end = slot.start
        while index < len(all_slots):
            candidate = all_slots[index]
            argument = candidate.argument
            if (
                argument.kind is not KernelAbiArgumentKind.MEMREF_FIELD
                or candidate.logical_index != logical_index
                or candidate.start != previous_end
            ):
                break
            if argument.field is None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI memref_field argument {argument.index} "
                    "requires a field name"
                )
            if argument.mlir_type != slot.argument.mlir_type:
                raise TlaUnsupportedAbiError(
                    "memref fields in a logical group must have the same MLIR type"
                )
            fields.append(argument.field)
            previous_end = candidate.end
            index += 1
        field_order = tuple(fields)
        value_indices = None
        run_packer = _DYNAMIC_GM_MEMREF_PACKER
        if field_order != _DYNAMIC_GM_MEMREF_FIELDS:
            # General strided memrefs use 3 + 2R fields, not the
            # rank-4 Dynamic-GM descriptor plus its two origin parameters.
            # Type validity is checked by the compiler; ABI fields define packing.
            rank = (len(field_order) - 3) // 2
            expected = (
                ("allocated", "aligned", "offset")
                + tuple(f"size{dim}" for dim in range(rank))
                + tuple(f"stride{dim}" for dim in range(rank))
            )
            if not 1 <= rank <= 4 or field_order != expected:
                raise TlaUnsupportedAbiError(
                    "memref fields must be one contiguous canonical 13-field run "
                    "or a rank-1..4 descriptor group, "
                    f"got {field_order!r} for {slot.argument.mlir_type}"
                )
            value_indices = tuple(
                _DYNAMIC_GM_MEMREF_FIELDS.index(field) for field in field_order
            )
            run_packer = struct.Struct(f"<{len(value_indices)}Q")
        memref_runs.append(
            _PreparedMemrefRun(
                logical_index=logical_index,
                start=run_start,
                packer=run_packer,
                value_indices=value_indices,
            )
        )
    return _PreparedAbiPacker(
        logical_count, host_size, tuple(slots), tuple(memref_runs)
    )


def _pack_launch_args_prepared(
    args: Sequence[Any], packer: _PreparedAbiPacker
) -> bytes:
    if len(args) != packer.logical_count:
        raise TlaUnsupportedAbiError(
            "kernel launch argument count does not match ABI layout: "
            f"got {len(args)}, expected {packer.logical_count}"
        )
    payload = bytearray(packer.host_size)
    pointer_values: dict[int, list[int]] = {}
    for slot in packer.slots:
        argument = slot.argument
        logical_index = slot.logical_index
        value = args[logical_index]
        if argument.kind is KernelAbiArgumentKind.POINTER:
            if isinstance(value, Numeric) or type(value) in (bool, int, float):
                raise TlaUnsupportedAbiError(
                    f"kernel ABI argument {argument.index} requires a pointer"
                )
            values = pointer_values.get(logical_index)
            if values is None:
                values = _runtime_arg_values(value)
                pointer_values[logical_index] = values
            if len(values) != 1:
                raise TlaUnsupportedAbiError(
                    "each logical launch argument must provide exactly one host value"
                )
            if argument.storage_size != _POINTER_ABI_SIZE:
                raise TlaUnsupportedAbiError(
                    f"unsupported pointer storage size {argument.storage_size}"
                )
            pointer = int(values[0])
            if pointer < 0 or pointer >= (1 << (argument.storage_size * 8)):
                raise TlaUnsupportedAbiError(
                    f"pointer value does not fit in {argument.storage_size} bytes"
                )
            encoded = pointer.to_bytes(
                argument.storage_size, byteorder="little", signed=False
            )
        elif argument.kind is KernelAbiArgumentKind.SCALAR:
            if argument.scalar is None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI scalar argument {argument.index} has no scalar descriptor"
                )
            encoded = _pack_scalar_argument(
                value,
                argument.scalar,
                argument.mlir_type,
                argument.storage_size,
            )
        else:
            raise TlaUnsupportedAbiError(
                f"unsupported kernel ABI argument kind {argument.kind!r}"
            )
        payload[slot.start : slot.end] = encoded

    for run in packer.memref_runs:
        value = args[run.logical_index]
        values = value.build_memref_launch_fields()
        if run.value_indices is not None:
            # Projection would hide missing or extra canonical fields.
            field_count = len(values)
            if field_count != len(_DYNAMIC_GM_MEMREF_FIELDS):
                raise TlaUnsupportedAbiError(
                    f"kernel ABI argument {run.logical_index} descriptor provider "
                    f"returned {field_count} fields, expected "
                    f"{len(_DYNAMIC_GM_MEMREF_FIELDS)} canonical fields"
                )
            values = tuple(values[index] for index in run.value_indices)
        run.packer.pack_into(payload, run.start, *values)
    return bytes(payload)


def _pack_launch_args(
    args: Sequence[Any], layout: KernelAbiLayout | None = None
) -> bytes:
    return _pack_launch_args_prepared(args, _prepare_abi_packer(layout))


def _append_debug_print_workspace_payload(
    payload: bytes,
    *,
    enabled: bool,
) -> bytes:
    if not enabled:
        return payload
    extension = bytearray(payload)
    _align_payload(extension, _POINTER_ABI_SIZE)
    extension.extend(
        _DEBUG_PRINT_WORKSPACE_SENTINEL.to_bytes(
            _POINTER_ABI_SIZE, byteorder="little", signed=False
        )
    )
    return bytes(extension)


def _validate_kernel_abi_layout(
    layout: KernelAbiLayout | None, *, expected_entrypoint: str | None = None
) -> None:
    if layout is None or layout.schema_version not in (3, 4):
        raise TlaUnsupportedAbiError(
            "A supported compiler-produced kernel ABI layout is required before launch."
        )
    if not layout.entrypoint.strip():
        raise TlaUnsupportedAbiError("kernel ABI layout entrypoint must be nonempty")
    if expected_entrypoint is not None and layout.entrypoint != expected_entrypoint:
        raise TlaUnsupportedAbiError(
            "kernel ABI layout entrypoint does not match launch artifact: "
            f"{layout.entrypoint!r} != {expected_entrypoint!r}"
        )
    if layout.total_size < 0:
        raise TlaUnsupportedAbiError("kernel ABI layout has an invalid total size")
    if layout.total_size > _MAX_KERNEL_ABI_PAYLOAD_SIZE:
        raise TlaUnsupportedAbiError(
            "kernel ABI payload exceeds the supported maximum size"
        )
    if layout.total_size % 8 != 0:
        raise TlaUnsupportedAbiError("kernel ABI total size must be rounded to 8 bytes")
    logical_count = _logical_launch_arg_count(layout)
    previous_end = 0
    for index, argument in enumerate(layout.arguments):
        if argument.index != index:
            raise TlaUnsupportedAbiError(
                "kernel ABI arguments must be ordered by contiguous index"
            )
        if argument.alignment != 4:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} must declare 4-byte alignment"
            )
        if argument.offset < previous_end or argument.offset % 4 != 0:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} offset must be ordered, "
                "non-overlapping, and 4-byte aligned"
            )
        if argument.storage_size <= 0:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} storage size must be positive"
            )
        logical_index = (
            argument.index if argument.logical_index is None else argument.logical_index
        )
        if logical_index < 0 or logical_index >= logical_count:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} logical_index {logical_index} is out of range"
            )
        if argument.kind is KernelAbiArgumentKind.POINTER:
            if argument.scalar is not None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI pointer argument {index} cannot have a scalar descriptor"
                )
            if argument.storage_size != _POINTER_ABI_SIZE:
                raise TlaUnsupportedAbiError(
                    f"unsupported pointer storage size {argument.storage_size}"
                )
        elif argument.kind is KernelAbiArgumentKind.MEMREF_FIELD:
            if argument.scalar is not None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI memref_field argument {index} cannot have a scalar descriptor"
                )
            if argument.field is None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI memref_field argument {index} requires a field name"
                )
            if argument.storage_size != _POINTER_ABI_SIZE:
                raise TlaUnsupportedAbiError(
                    f"unsupported memref_field storage size {argument.storage_size}"
                )
        elif argument.kind is KernelAbiArgumentKind.SCALAR:
            if argument.scalar is None:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI scalar argument {index} requires a scalar descriptor"
                )
            expected_size = _scalar_storage_size(argument.scalar)
            if argument.storage_size != expected_size:
                raise TlaUnsupportedAbiError(
                    f"kernel ABI scalar {argument.mlir_type} declares unsupported "
                    f"storage size {argument.storage_size}; expected {expected_size}"
                )
        else:
            raise TlaUnsupportedAbiError(
                f"unsupported kernel ABI argument kind {argument.kind!r}"
            )
        previous_end = argument.offset + argument.storage_size
        if previous_end > layout.total_size:
            raise TlaUnsupportedAbiError(
                f"kernel ABI argument {index} storage does not fit in payload"
            )
    required_size = ((previous_end + 7) // 8) * 8
    if layout.total_size != required_size:
        raise TlaUnsupportedAbiError(
            "kernel ABI total size is not exactly sufficient for its arguments"
        )


def _prepare_compiled_abi_packer(
    layout: KernelAbiLayout | None, *, expected_entrypoint: str
) -> _PreparedAbiPacker:
    try:
        return _prepare_abi_packer(layout, expected_entrypoint=expected_entrypoint)
    except TlaUnsupportedAbiError as exc:
        raise TlaKernelCompileError(
            f"Invalid compiler-produced kernel ABI descriptor: {exc}"
        ) from exc


def _extract_mixed_handoff_entrypoints(mlir_text: str) -> tuple[str, str] | None:
    names = re.findall(
        r"(?:func\.func|\"func\.func\")\s+@([A-Za-z_][A-Za-z0-9_]*)",
        mlir_text,
    )
    aic_names = [name for name in names if name.endswith("_mix_aic")]
    aiv_names = [name for name in names if name.endswith("_mix_aiv")]
    if len(aic_names) != 1 or len(aiv_names) != 1:
        return None
    return aic_names[0], aiv_names[0]


def _extract_logical_mixed_handoff(mlir_text: str) -> _LogicalMixedHandoff | None:
    entrypoints = _extract_mixed_handoff_entrypoints(mlir_text)
    if entrypoints is None:
        return None
    aic_entrypoint, aiv_entrypoint = entrypoints
    logical_entrypoint = aic_entrypoint.removesuffix("_mix_aic")
    if aiv_entrypoint.removesuffix("_mix_aiv") != logical_entrypoint:
        return None
    return _LogicalMixedHandoff(logical_entrypoint)


def _analyze_artifact_static_metadata(
    tlair_mlir: str, lowered_llvm: str
) -> _ArtifactStaticMetadata:
    mlir_text = lowered_llvm or tlair_mlir
    return _ArtifactStaticMetadata(
        uses_scalar_print="tla.debug_print.workspace" in mlir_text,
        uses_tensor_print="tla.print_tensor.workspace" in mlir_text,
        logical_mixed_handoff=_extract_logical_mixed_handoff(mlir_text),
    )


def _analyze_compiled_artifact_static_metadata(
    tlair_mlir: str, lowered_llvm: str
) -> _ArtifactStaticMetadata:
    try:
        return _analyze_artifact_static_metadata(tlair_mlir, lowered_llvm)
    except TlaUnsupportedAbiError as exc:
        raise TlaKernelCompileError(
            f"Invalid compiler-produced launch metadata: {exc}"
        ) from exc


def _capture_c_stdout(launch: Callable[[], None]) -> str:
    libc = ctypes.CDLL(None)
    with _NATIVE_PRINT_TENSOR_STDOUT_LOCK:
        sys.stdout.flush()
        libc.fflush(None)
        saved_stdout = os.dup(1)
        try:
            with tempfile.TemporaryFile(mode="w+b") as captured:
                os.dup2(captured.fileno(), 1)
                try:
                    launch()
                finally:
                    libc.fflush(None)
                    sys.stdout.flush()
                    os.dup2(saved_stdout, 1)
                captured.seek(0)
                return captured.read().decode("utf-8", errors="replace")
        finally:
            os.close(saved_stdout)


def _print_tensor_static_metadata_records(
    tlair_mlir: str, *, entrypoint: str | None = None
) -> tuple[_PrintTensorMetadata, ...]:
    function_text = (
        _mlir_entrypoint_text(tlair_mlir, entrypoint)
        if entrypoint is not None
        else tlair_mlir
    )
    op_matches = list(re.finditer(r'"tla\.print_tensor"', function_text))
    if not op_matches:
        raise TlaExecutionError(
            "tla.print_tensor requires static shape metadata; found 0 records"
        )
    constants = {
        match.group("result"): int(match.group("value"))
        for match in re.finditer(
            r'(?P<result>%[A-Za-z0-9_.$-]+)\s*=\s*"arith\.constant"\(\)'
            r"\s*<\{\s*value\s*=\s*(?P<value>\d+)\s*:\s*i64\s*\}>",
            function_text,
        )
    }
    op_ends = [match.start() for match in op_matches[1:]]
    op_ends.append(len(function_text))
    metadata = []
    for call, (match, end) in enumerate(zip(op_matches, op_ends)):
        op_text = function_text[match.start() : end]
        operand_match = re.match(
            r'"tla\.print_tensor"\(\s*[^,]+,\s*'
            r"(?P<length>%[A-Za-z0-9_.$-]+)\s*\)",
            op_text,
        )
        metadata.append(
            _parse_print_tensor_static_metadata(
                op_text,
                call=call,
                length=(
                    constants.get(operand_match.group("length"))
                    if operand_match is not None
                    else None
                ),
            )
        )
    return tuple(metadata)


def _mlir_entrypoint_text(mlir_text: str, entrypoint: str) -> str:
    function_matches = list(
        re.finditer(
            r'"(?:tla|func)\.func"\s*\('
            r"|(?:tla|func)\.func\s+@(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b",
            mlir_text,
        )
    )
    for index, match in enumerate(function_matches):
        end = (
            function_matches[index + 1].start()
            if index + 1 < len(function_matches)
            else len(mlir_text)
        )
        function_text = mlir_text[match.start() : end]
        name = match.group("name")
        if name is None:
            name_match = re.search(r'\bsym_name\s*=\s*"([^"]+)"', function_text)
            name = name_match.group(1) if name_match is not None else None
        if name == entrypoint:
            return function_text
    raise TlaExecutionError(
        f"tla.print_tensor entrypoint {entrypoint!r} is missing from the compiled artifact"
    )


def _parse_print_tensor_static_metadata(
    op_text: str, *, call: int, length: int | None = None
) -> _PrintTensorMetadata:
    shape_match = re.search(r"\bshape\s*=\s*array<i64:\s*([-\d,\s]+)>", op_text)
    if shape_match is None:
        raise TlaExecutionError(
            "tla.print_tensor static shape metadata is missing from the compiled artifact"
        )
    operand_match = re.search(
        r"!tla\.ptr<\s*([^,\s>]+)\s*,\s*(gm|ub|l1|l0c)\s*,", op_text
    )
    if operand_match is None:
        raise TlaExecutionError(
            "tla.print_tensor operand type metadata is missing or malformed"
        )
    mlir_dtype = operand_match.group(1)
    position = operand_match.group(2).upper()
    shape = tuple(int(item.strip()) for item in shape_match.group(1).split(","))
    if len(shape) not in (1, 2) or any(extent == 0 or extent < -1 for extent in shape):
        raise TlaExecutionError(
            "tla.print_tensor static shape metadata contains an invalid extent"
        )
    length_match = re.search(r"\blength\s*=\s*(\d+)", op_text)
    if length is None and length_match is not None:
        length = int(length_match.group(1))
    max_length = (
        8
        if position == "L1"
        else 256
        if position == "L0C"
        else _PRINT_TENSOR_MAX_F32_ELEMENTS
    )
    if length is not None and (
        not 1 <= length <= max_length
        or (all(extent > 0 for extent in shape) and length > math.prod(shape))
    ):
        raise TlaExecutionError(
            "tla.print_tensor static length metadata is invalid for the tensor shape"
        )
    dtype = _PRINT_TENSOR_DTYPES_BY_MLIR.get(mlir_dtype)
    if dtype is None:
        raise TlaExecutionError(
            f"tla.print_tensor runtime metadata has unsupported dtype {mlir_dtype!r}; "
            "supported: f16, f32, i8, i16, i32, ui8, ui16, ui32"
        )
    return _PrintTensorMetadata(
        shape=shape,
        count=length,
        dtype=dtype,
        position=position,
        call=call,
        dynamic_execution=(
            length is None
            or any(extent < 0 for extent in shape)
            or "dynamic_execution" in op_text
        ),
    )


def _print_tensor_decode_error(detail: str) -> TlaExecutionError:
    return TlaExecutionError(
        f"tla.print_tensor native initialization or decoding failed: {detail}"
    )


def _parse_native_print_tensor_records(output: str) -> list[_NativePrintTensorRecord]:
    records: list[_NativePrintTensorRecord] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line.startswith("DumpTensor:"):
            continue
        match = _NATIVE_PRINT_TENSOR_RECORD_RE.fullmatch(line)
        if match is None:
            raise _print_tensor_decode_error("malformed native record")
        shape = tuple(int(item.strip()) for item in match.group("shape").split(","))
        records.append(
            _NativePrintTensorRecord(
                native_dtype=match.group("dtype"),
                position=match.group("position"),
                declared_count=int(match.group("count")),
                values_text=match.group("values"),
                shape=shape,
                call=int(match.group("call")),
                block=int(match.group("block")),
                subblock=(
                    int(match.group("subblock"))
                    if match.group("subblock") is not None
                    else None
                ),
            )
        )
    return records


def _print_tensor_integer_bounds(
    dtype: str, spec: _PrintTensorDTypeSpec
) -> tuple[int, int]:
    if spec.minimum is None or spec.maximum is None:
        raise TlaExecutionError(
            f"tla.print tensor dtype metadata is missing integer bounds (dtype={dtype})"
        )
    return spec.minimum, spec.maximum


def _decode_print_tensor_record_values(
    record: _NativePrintTensorRecord, *, dtype: str
) -> list[float | int]:
    spec = _PRINT_TENSOR_DTYPES[dtype]
    items = (
        [item.strip() for item in record.values_text.split(",")]
        if record.values_text.strip()
        else []
    )
    if len(items) != record.declared_count:
        raise _print_tensor_decode_error(
            f"payload count is {len(items)}, record declares {record.declared_count}",
        )
    if spec.numeric_kind == "float":
        if any(_FLOAT_LITERAL_RE.fullmatch(item) is None for item in items):
            raise _print_tensor_decode_error(f"invalid {dtype} numeric syntax")
        return [float(item) for item in items]
    if any(_INTEGER_LITERAL_RE.fullmatch(item) is None for item in items):
        raise _print_tensor_decode_error(f"invalid {dtype} numeric syntax")
    values = [int(item, 10) for item in items]
    minimum, maximum = _print_tensor_integer_bounds(dtype, spec)
    if any(value < minimum or value > maximum for value in values):
        raise _print_tensor_decode_error(
            f"{dtype} integer value outside [{minimum}, {maximum}]"
        )
    return values


def _decode_native_print_tensor_records(
    output: str,
    *,
    metadata: Sequence[_PrintTensorMetadata],
    block_count: int = 1,
    expected_subblocks: tuple[int | None, ...] = (None,),
) -> list[tuple[_PrintTensorMetadata, _NativePrintTensorRecord, list[float | int]]]:
    records = _parse_native_print_tensor_records(output)
    expected_metadata = {item.call: item for item in metadata}
    valid_blocks = set(range(block_count))
    valid_subblocks = set(expected_subblocks)
    record_values: dict[int, list[float | int]] = {}
    malformed_identities: set[tuple[int, int, int | None]] = set()
    for record in records:
        identity = (record.call, record.block, record.subblock)
        metadata_item = expected_metadata.get(record.call)
        dtype = (
            metadata_item.dtype
            if metadata_item is not None
            else _PRINT_TENSOR_DTYPES_BY_NATIVE.get(record.native_dtype)
        )
        if dtype is None:
            malformed_identities.add(identity)
            continue
        try:
            record_values[id(record)] = _decode_print_tensor_record_values(
                record, dtype=dtype
            )
        except TlaExecutionError:
            malformed_identities.add(identity)
    if malformed_identities:
        raise _print_tensor_decode_error(
            f"malformed records {_sort_print_tensor_identities(malformed_identities)}"
        )

    # A dynamic branch can produce no record and a loop can repeat one. Native
    # output is therefore best-effort: validate every observed record without
    # requiring a one-to-one correspondence with statically lowered calls.
    identities = [(record.call, record.block, record.subblock) for record in records]
    unexpected_identities = _sort_print_tensor_identities(
        {
            identity
            for identity in identities
            if (
                identity[0] not in expected_metadata
                or identity[1] not in valid_blocks
                or identity[2] not in valid_subblocks
            )
        }
    )
    if unexpected_identities:
        raise _print_tensor_decode_error(
            f"unexpected call/block/subblock identities {unexpected_identities}",
        )

    mismatches: dict[str, set[tuple[int, int, int | None]]] = {
        "native dtype": set(),
        "position": set(),
        "declared count": set(),
        "shape": set(),
    }
    for record in records:
        identity = (record.call, record.block, record.subblock)
        item = expected_metadata[record.call]
        if record.native_dtype != _PRINT_TENSOR_DTYPES[item.dtype].native_name:
            mismatches["native dtype"].add(identity)
        if record.position != item.position:
            mismatches["position"].add(identity)
        if item.count is not None and record.declared_count != item.count:
            mismatches["declared count"].add(identity)
        if (
            len(record.shape) != len(item.shape)
            or any(extent < 1 for extent in record.shape)
            or math.prod(record.shape) < record.declared_count
            or any(
                expected != -1 and expected != actual
                for expected, actual in zip(item.shape, record.shape, strict=True)
            )
        ):
            mismatches["shape"].add(identity)
    for label, mismatch in mismatches.items():
        if mismatch:
            raise _print_tensor_decode_error(
                f"unexpected {label} records {_sort_print_tensor_identities(mismatch)}"
            )

    return [
        (expected_metadata[record.call], record, record_values[id(record)])
        for record in records
    ]


def _sort_print_tensor_identities(
    identities: Iterable[tuple[int, int, int | None]],
) -> list[tuple[int, int, int | None]]:
    return sorted(
        identities,
        key=lambda identity: (
            identity[0],
            identity[1],
            -1 if identity[2] is None else identity[2],
        ),
    )


def _format_print_tensor_record(
    values: Sequence[float | int],
    *,
    shape: Sequence[int],
    dtype: str = "f32",
    call: int | None = None,
    block: int | None = None,
    position: str | None = None,
    subblock: int | None = None,
) -> str:
    spec = _PRINT_TENSOR_DTYPES.get(dtype)
    if spec is None:
        raise TlaExecutionError(
            f"tla.print tensor formatting failed (dtype={dtype}): unsupported dtype"
        )
    count = len(values)
    rendered_shape = ",".join(str(int(extent)) for extent in shape)
    if spec.numeric_kind == "float":
        rendered_values = ", ".join(str(float(value)) for value in values)
    else:
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        ):
            raise TlaExecutionError(
                f"tla.print tensor formatting failed (dtype={dtype}): "
                "integer dtype requires integer values"
            )
        minimum, maximum = _print_tensor_integer_bounds(dtype, spec)
        if any(value < minimum or value > maximum for value in values):
            raise TlaExecutionError(
                f"tla.print tensor formatting failed (dtype={dtype}): "
                f"integer value outside [{minimum}, {maximum}]"
            )
        rendered_values = ", ".join(str(value) for value in values)
    public_dtype = "float32" if dtype == "f32" else dtype
    identity = "" if call is None else f"call={call} block={block} "
    rendered_position = f"position={position} " if position is not None else ""
    rendered_subblock = f"subblock={subblock} " if subblock is not None else ""
    return (
        f"tla.print {identity}dtype={public_dtype} "
        f"{rendered_position}{rendered_subblock}"
        f"shape=[{rendered_shape}] count={count} "
        f"values=[{rendered_values}]"
    )


def compile_option_from_kwargs(kwargs: Mapping[str, Any]) -> TlaCompileOption:
    """Build compile options from Host compile/launch kwargs.

    Public arch selection::

        tla.compile(fn, *args, options="--npu-arch 3510")

    AIC/AIV is not a Host compile knob; it is inferred from lowered MLIR.
    Cache / cache dir / force-recompile / IR dump use env vars
    (``CATLASS_DSL_CACHE``, ``CATLASS_DSL_CACHE_DIR``, ``CATLASS_DSL_FORCE_RECOMPILE``,
    ``CATLASS_DSL_PRINT_IR``, ``CATLASS_DSL_KEEP``, ``CATLASS_DSL_DUMP_DIR``).
    """
    parsed_options = _parse_compile_options_from_str(kwargs.get("options"))
    npu_arch = str(parsed_options.get("npu_arch", DEFAULT_NPU_ARCH))
    target_arch = _resolve_npu_arch(npu_arch)
    # Placeholder until ``_resolve_compile_option_from_lowered_mlir`` reads core type.
    arch_scope = _arch_scope_for_target(target_arch=target_arch, core_type="aiv")
    _, core_type = _parse_arch_scope(arch_scope)
    cache_dir_env = os.getenv("CATLASS_DSL_CACHE_DIR")
    return TlaCompileOption(
        cache_enabled=_env_truthy("CATLASS_DSL_CACHE", default="1"),
        cache_dir=(
            Path(cache_dir_env).expanduser().resolve() if cache_dir_env else None
        ),
        force_recompile=_env_truthy("CATLASS_DSL_FORCE_RECOMPILE", default="0"),
        kernel_mode=core_type,
        arch_scope=arch_scope,
        print_ir=_env_truthy("CATLASS_DSL_PRINT_IR", default="0"),
        cce_disable_asc_reserved_ubuf="cce_disable_asc_reserved_ubuf" in parsed_options,
        cce_disable_vf_stack_reserved_ubuf="cce_disable_vf_stack_reserved_ubuf"
        in parsed_options,
    )


def compile_option_for_launch(
    compile_option: TlaCompileOption,
) -> TlaCompileOption:
    if compile_option.cache_dir is not None:
        return compile_option
    if compile_option.cache_enabled:
        return replace(compile_option, cache_dir=_default_cache_dir())
    temp_dir = Path(tempfile.mkdtemp(prefix="tla-dsl-kernel-")).resolve()
    return replace(compile_option, cache_enabled=False, cache_dir=temp_dir)


def _resolve_compile_option_from_lowered_mlir(
    compile_option: TlaCompileOption,
    mlir_text: str,
    *,
    has_logical_mixed_handoff: bool,
) -> TlaCompileOption:
    target_arch, core_type = _parse_arch_scope(compile_option.arch_scope)
    kernel_mode = compile_option.kernel_mode

    if has_logical_mixed_handoff:
        core_type = "aic"
        kernel_mode = "mix"
    elif (
        "hivm.module_core_type<AIC>" in mlir_text
        or "hivm.func_core_type = #hivm.func_core_type<AIC>" in mlir_text
    ):
        core_type = "aic"
        kernel_mode = "aic"
    elif (
        "hivm.module_core_type<AIV>" in mlir_text
        or "hivm.func_core_type = #hivm.func_core_type<AIV>" in mlir_text
    ):
        core_type = "aiv"
        kernel_mode = "aiv"

    if "dav-c310" in mlir_text or 'hacc.target<"Ascend950PR_9589">' in mlir_text:
        target_arch = "c310"

    arch_scope = _arch_scope_for_target(target_arch=target_arch, core_type=core_type)
    if (
        compile_option.kernel_mode == kernel_mode
        and compile_option.arch_scope == arch_scope
    ):
        return compile_option
    return replace(
        compile_option,
        kernel_mode=kernel_mode,
        arch_scope=arch_scope,
    )


def _extract_entrypoint(mlir_text: str) -> str:
    match = re.search(r"@([A-Za-z_][A-Za-z0-9_]*)\s*\(", mlir_text)
    if match:
        return match.group(1)
    sym_match = re.search(r'sym_name\s*=\s*"([A-Za-z_][A-Za-z0-9_]*)"', mlir_text)
    if sym_match:
        return sym_match.group(1)
    raise TlaExecutionError("Could not infer kernel entrypoint from lowered MLIR.")


def _cache_key(
    *,
    tlair_mlir: str,
    entrypoint: str,
    compile_option: TlaCompileOption,
    compiler_bridge_path: Path | None,
    hivmc: Path,
    target: TlaKernelTarget,
    extern_compile_specs: Sequence[ExternCompileSpec] = (),
) -> str:
    extern_compile_environment = (
        None if not extern_compile_specs else _ascendc_compile_environment_identity()
    )
    extern_compile_identities = sorted(
        (
            _extern_source_sha256(spec.source),
            tuple(sorted(spec.core_types)),
            tuple(str(path) for path in spec.include_dirs),
        )
        for spec in extern_compile_specs
    )
    key_payload = {
        "debug_print_workspace_abi_revision": _DEBUG_PRINT_WORKSPACE_ABI_REVISION,
        "print_tensor_workspace_abi_revision": _PRINT_TENSOR_WORKSPACE_ABI_REVISION,
        "cache_abi_version": _ONLINE_CACHE_ABI_VERSION,
        "entrypoint": entrypoint,
        "kernel_mode": compile_option.kernel_mode,
        "arch_scope": compile_option.arch_scope,
        "cce_arch": target.cce_arch,
        "compiler_bridge": str(compiler_bridge_path) if compiler_bridge_path else None,
        "hivmc": str(hivmc),
        "compiler_bridge_fingerprint": _tool_fingerprint(compiler_bridge_path),
        "hivmc_fingerprint": _tool_metadata_fingerprint(hivmc),
        "mlir": tlair_mlir,
        "print_ir": compile_option.print_ir,
        "cce_disable_asc_reserved_ubuf": compile_option.cce_disable_asc_reserved_ubuf,
        "cce_disable_vf_stack_reserved_ubuf": compile_option.cce_disable_vf_stack_reserved_ubuf,
        "extern_compile_specs": extern_compile_identities,
        "extern_compile_environment": extern_compile_environment,
    }
    return hashlib.sha256(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _extern_source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _tool_version(binary: Path) -> str:
    try:
        proc = subprocess.run(
            [str(binary), "--version"], check=False, capture_output=True, text=True
        )
    except OSError:
        return "unknown"
    text = (proc.stdout or "") + (proc.stderr or "")
    text = text.strip()
    if not text:
        return f"exit:{proc.returncode}"
    return text.splitlines()[0][:200]


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_fingerprint(binary: Path | None) -> str:
    if binary is None:
        return "unresolved"
    try:
        stat = binary.stat()
    except OSError:
        return "missing"
    try:
        digest = _file_sha256(binary)
    except OSError:
        return f"stat:{stat.st_size}:{stat.st_mtime_ns}"
    return f"{stat.st_size}:{stat.st_mtime_ns}:{digest}"


def _tool_metadata_fingerprint(binary: Path) -> str:
    """Identify a compiler by its resolved path and filesystem change metadata."""
    try:
        resolved = binary.resolve()
        stat = resolved.stat()
    except (OSError, RuntimeError):  # Missing/inaccessible files or symlink loops.
        return "missing"
    return (
        f"stat:{resolved}:{stat.st_dev}:{stat.st_ino}:{stat.st_size}:"
        f"{stat.st_mtime_ns}:{stat.st_ctime_ns}"
    )


def _default_cache_dir() -> Path:
    cache = os.getenv("CATLASS_DSL_CACHE_DIR")
    if cache:
        return Path(cache).expanduser().resolve()
    xdg = os.getenv("XDG_CACHE_HOME")
    if xdg:
        return (Path(xdg) / "catlass").expanduser().resolve()
    return (Path.home() / ".cache" / "catlass").resolve()


def _write_pass_ir_dump(pass_dump_path: Path, text: str) -> None:
    """Write pass IR dump to the cache path; mirror to DUMP_DIR when set."""
    pass_dump_path.write_text(text)
    dump_dir_env = os.getenv("CATLASS_DSL_DUMP_DIR")
    if not dump_dir_env:
        return
    dump_dir = Path(dump_dir_env).expanduser().resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / pass_dump_path.name).write_text(text)


def _parse_keep_tokens(raw: str) -> frozenset[str]:
    """Parse ``CATLASS_DSL_KEEP`` into canonical artifact tokens.

    Tokens (comma-separated, case-insensitive):
      ir        — lowered MLIR after the Ascend pipeline (``*.mlir``)
      ir-debug  — TLA IR before HIVM lowering (``*.tlair.mlir``); also
                  ``*.pass-ir-dump.mlir`` when pass dump text is available
      kernel    — device object ``*.o``
      all       — expand to every token above

    Unknown tokens are ignored.
    """
    tokens = frozenset(t.strip().lower() for t in raw.split(",") if t.strip())
    if "all" in tokens:
        return _KEEP_ALL_TOKENS
    return tokens & _KEEP_ALL_TOKENS


def _copy_kept_artifacts(compiled: "JitCompiledFunction") -> None:
    """Copy selected compile artifacts to DUMP_DIR (or CWD) when KEEP is set."""
    tokens = _parse_keep_tokens(os.getenv("CATLASS_DSL_KEEP", ""))
    if not tokens:
        return
    dump_dir_env = os.getenv("CATLASS_DSL_DUMP_DIR")
    dump_dir = (
        Path(dump_dir_env).expanduser().resolve()
        if dump_dir_env
        else Path.cwd().resolve()
    )
    dump_dir.mkdir(parents=True, exist_ok=True)
    artifacts = compiled.artifacts
    stem = compiled.entrypoint or compiled.cache_key
    if "ir" in tokens:
        (dump_dir / f"{stem}.mlir").write_text(artifacts.LLVM)
    if "ir-debug" in tokens:
        (dump_dir / f"{stem}.tlair.mlir").write_text(artifacts.MLIR)
        if artifacts.PASS_IR_DUMP:
            (dump_dir / f"{stem}.pass-ir-dump.mlir").write_text(artifacts.PASS_IR_DUMP)
    if "kernel" in tokens and artifacts.BINARY.exists():
        shutil.copy2(artifacts.BINARY, dump_dir / f"{stem}.o")


def _parse_arch_scope(arch_scope: str) -> tuple[str, str]:
    try:
        return _parse_arch_scope_impl(arch_scope)
    except ValueError as exc:
        raise TlaExecutionError(str(exc)) from exc


def _core_type_from_arch_scope(arch_scope: str) -> str:
    return _parse_arch_scope(arch_scope)[1]


def _arch_scope_for_target(*, target_arch: str, core_type: str) -> str:
    try:
        return _arch_scope_for_target_impl(target_arch=target_arch, core_type=core_type)
    except ValueError as exc:
        raise TlaExecutionError(str(exc)) from exc


def _resolve_kernel_target(compile_option: TlaCompileOption) -> TlaKernelTarget:
    target_arch, core_type = _parse_arch_scope(compile_option.arch_scope)
    try:
        return _get_kernel_target(
            target_arch=target_arch,
            core_type=core_type,
            arch_scope=compile_option.arch_scope,
        )
    except ValueError as exc:
        raise TlaExecutionError(str(exc)) from exc


def _resolve_hivmc_a5() -> Path:
    """Resolve ``hivmc-a5`` from PATH / ``ASCEND_HOME_PATH`` after ``set_env.sh``."""

    which = shutil.which("hivmc-a5")
    if which:
        return Path(which).resolve()
    ascend_home = os.getenv("ASCEND_HOME_PATH")
    if ascend_home:
        candidate = Path(ascend_home).expanduser().resolve() / "bin" / "hivmc-a5"
        if candidate.exists():
            return candidate
    raise TlaBackendCompilerNotFoundError(
        "hivmc-a5 not found on PATH. Source the CANN toolkit "
        "(`source .../ascend-toolkit/set_env.sh`) so `hivmc-a5` is available."
    )


def _resolve_ccec() -> Path:
    """Resolve ``ccec`` from PATH / ``ASCEND_HOME_PATH`` after ``set_env.sh``."""
    which = shutil.which("ccec")
    if which:
        return Path(which).resolve()
    ascend_home = os.getenv("ASCEND_HOME_PATH")
    if ascend_home:
        candidate = Path(ascend_home).expanduser().resolve() / "bin" / "ccec"
        if candidate.exists():
            return candidate.resolve()
    raise TlaBackendCompilerNotFoundError(
        "ccec not found on PATH. Source the CANN toolkit set_env.sh so user "
        "Ascend C external functions can be compiled."
    )


def _ascendc_builtin_include_dirs(ascend_home: Path) -> list[Path]:
    asc_root = (ascend_home / "asc").resolve()
    highlevel_api = asc_root.parent / "ascendc" / "include" / "highlevel_api"
    return [
        path.resolve()
        for path in (
            asc_root,
            asc_root / "impl" / "adv_api",
            asc_root / "impl" / "basic_api",
            asc_root / "impl" / "basic_api" / "reg_compute",
            asc_root / "impl" / "c_api",
            asc_root / "impl" / "micro_api",
            asc_root / "impl" / "simt_api",
            asc_root / "impl" / "utils",
            asc_root / "include",
            asc_root / "include" / "adv_api",
            asc_root / "include" / "aicpu_api",
            asc_root / "include" / "basic_api",
            asc_root / "include" / "basic_api" / "reg_compute",
            asc_root / "include" / "c_api",
            asc_root / "include" / "interface",
            asc_root / "include" / "micro_api",
            asc_root / "include" / "simt_api",
            asc_root / "include" / "tiling",
            asc_root / "include" / "utils",
            highlevel_api,
            Path(__file__).resolve().parents[3] / "include",
        )
        if path.is_dir()
    ]


def _ascendc_compiler_inputs() -> tuple[Path, list[Path]]:
    compiler = _resolve_ccec()
    ascend_home_env = os.getenv("ASCEND_HOME_PATH")
    if not ascend_home_env:
        raise TlaBackendCompilerNotFoundError(
            "ASCEND_HOME_PATH is not set. Source the CANN toolkit set_env.sh so "
            "Ascend C headers can be found."
        )
    ascend_home = Path(ascend_home_env).expanduser().resolve()
    return compiler, _ascendc_builtin_include_dirs(ascend_home)


def _resolve_extern_targets(
    *,
    core_types: Iterable[str],
    target_arch: str,
) -> tuple[TlaKernelTarget, ...]:
    core_types = frozenset(core_types)
    if not core_types:
        raise TlaKernelCompileError("external source has no call target")
    unsupported = core_types.difference(("aic", "aiv"))
    if unsupported:
        raise TlaKernelCompileError(
            f"unsupported external function core types: {sorted(unsupported)}"
        )

    targets = tuple(
        _get_kernel_target(target_arch=target_arch, core_type=core_type)
        for core_type in ("aic", "aiv")
        if core_type in core_types
    )
    return targets


def _ascendc_compile_environment_identity() -> dict[str, object]:
    """Return the resolved Ascend C compile environment tracked by the cache."""

    compiler, builtin_include_dirs = _ascendc_compiler_inputs()
    return {
        "ccec": str(compiler),
        "ccec_version": _tool_version(compiler),
        "ccec_fingerprint": _tool_fingerprint(compiler),
        "builtin_include_dirs": [str(path) for path in builtin_include_dirs],
    }


def _parse_ascendc_depfile(depfile: Path, *, cwd: Path) -> set[Path]:
    try:
        dependency_text = re.sub(
            r"\\\r?\n", " ", depfile.read_text(encoding="utf-8")
        ).split(":", 1)[1]
        raw_paths = shlex.split(dependency_text, comments=False, posix=True)
    except (OSError, IndexError, ValueError) as exc:
        raise TlaKernelCompileError(
            f"invalid CCEC dependency file {depfile}: {exc}"
        ) from exc

    dependencies: set[Path] = set()
    for raw_path in raw_paths:
        path = Path(raw_path.replace("$$", "$"))
        dependencies.add((path if path.is_absolute() else cwd / path).resolve())
    return dependencies


def _snapshot_extern_dependency_groups(
    dependency_groups: Mapping[tuple[str, str], frozenset[Path]],
    *,
    builtin_include_roots: Sequence[Path] = (),
) -> list[dict[str, object]]:
    unique_dependencies = {
        path for dependencies in dependency_groups.values() for path in dependencies
    }
    dependency_snapshots: dict[Path, dict[str, object]] = {}
    for path in unique_dependencies:
        if any(path.is_relative_to(root) for root in builtin_include_roots):
            stat = path.stat()
            dependency_snapshots[path] = {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        else:
            dependency_snapshots[path] = {
                "path": str(path),
                "sha256": _file_sha256(path),
            }
    return [
        {
            "source_sha256": source_sha256,
            "target": target,
            "dependency_snapshots": [
                dependency_snapshots[path]
                for path in sorted(dependency_groups[(source_sha256, target)], key=str)
            ],
        }
        for source_sha256, target in sorted(dependency_groups)
    ]


def _ascendc_extern_base_command(
    source_path: Path,
    *,
    user_include_dirs: Sequence[Path],
) -> list[str]:
    compiler, builtin_include_dirs = _ascendc_compiler_inputs()
    command = [
        str(compiler),
        "-O2",
        "-x",
        "cce",
        "--cce-auto-sync=off",
        "--cce-aicore-only",
        "--cce-generic-addrspace=off",
        str(source_path),
        "-DCATLASS_ARCH=3510",
        "-DTILING_KEY_VAR",
        "-Wno-ignored-attributes",
        "-std=c++17",
    ]
    for include_dir in (*user_include_dirs, *builtin_include_dirs):
        command.extend(["-I", str(include_dir)])
    return command


def _parse_ascendc_dependencies(
    depfile: Path,
    *,
    cwd: Path,
    source_path: Path,
) -> set[Path]:
    source_path = source_path.resolve()
    return {
        dependency
        for dependency in _parse_ascendc_depfile(depfile, cwd=cwd)
        if dependency != source_path
    }


def _scan_ascendc_extern_source_dependencies(
    source: str,
    *,
    source_index: int,
    targets: Sequence[TlaKernelTarget],
    user_include_dirs: Sequence[Path],
) -> dict[str, frozenset[Path]]:
    with tempfile.TemporaryDirectory(prefix="catlass-extern-deps-") as temp_dir:
        scan_dir = Path(temp_dir)
        source_path = scan_dir / f"extern.{source_index}.cpp"
        source_path.write_text(source, encoding="utf-8")
        command = _ascendc_extern_base_command(
            source_path,
            user_include_dirs=user_include_dirs,
        )

        dependencies_by_target: dict[str, frozenset[Path]] = {}
        for target in targets:
            depfile = scan_dir / f"extern.{source_index}.{target.arch_scope}.scan.d"
            scan_command = [
                *command,
                f"--cce-aicore-arch={target.cce_arch}",
                "-M",
                "-MF",
                str(depfile),
            ]
            _run_checked(
                scan_command,
                label=f"ccec external source dependency scan {source_index}",
                cwd=scan_dir,
            )
            dependencies_by_target[target.arch_scope] = frozenset(
                _parse_ascendc_dependencies(
                    depfile,
                    cwd=scan_dir,
                    source_path=source_path,
                )
            )
        return dependencies_by_target


def _compile_ascendc_extern_source(
    source: str,
    *,
    source_index: int,
    artifact_dir: Path,
    targets: Sequence[TlaKernelTarget],
    user_include_dirs: Sequence[Path],
) -> tuple[tuple[Path, ...], dict[str, frozenset[Path]]]:
    # ``targets`` is produced by ``_resolve_extern_targets`` in the compile path.
    targets = tuple(targets)
    assert 1 <= len(targets) <= 2
    source_path = artifact_dir / f"extern.{source_index}.cpp"
    source_path.write_text(source, encoding="utf-8")
    command = _ascendc_extern_base_command(
        source_path,
        user_include_dirs=user_include_dirs,
    )
    command.extend(
        [
            "-emit-llvm",
            "-c",
            "-mllvm",
            "-disable-llvm-optzns",
        ]
    )

    outputs_and_depfiles = tuple(
        (
            artifact_dir / f"extern.{source_index}.{target.arch_scope}.bc",
            artifact_dir / f"extern.{source_index}.{target.arch_scope}.d",
        )
        for target in targets
    )
    dependencies_by_target: dict[str, frozenset[Path]] = {}
    for target, (output, depfile) in zip(targets, outputs_and_depfiles, strict=True):
        target_command = [
            *command,
            f"--cce-aicore-arch={target.cce_arch}",
            "-MD",
            "-MF",
            str(depfile),
            "-MT",
            str(output),
            "-o",
            str(output),
        ]
        _run_checked(
            target_command,
            label=f"ccec external source {source_index}",
            cwd=artifact_dir,
        )
        if not output.exists():
            raise TlaKernelCompileError(
                "ccec completed but external source bitcode was not created at "
                f"{output}"
            )
        dependencies_by_target[target.arch_scope] = frozenset(
            _parse_ascendc_dependencies(
                depfile,
                cwd=artifact_dir,
                source_path=source_path,
            )
        )
        depfile.unlink()

    return (
        tuple(output.resolve() for output, _ in outputs_and_depfiles),
        dependencies_by_target,
    )


def _build_hivmc_a5_command(
    *,
    compiler: Path,
    mlir_path: Path,
    kernel_binary_path: Path,
    compile_option: TlaCompileOption,
    template_bitcode: str | None = None,
) -> list[str]:
    if template_bitcode is None:
        template_bitcode = _resolve_hivm_template_bitcode(compile_option)
    command = [
        str(compiler),
        str(mlir_path),
        "--target=Ascend950PR_9589",
        "--mlir-print-op-on-diagnostic",
    ]
    reserve_flags = []
    if compile_option.cce_disable_asc_reserved_ubuf:
        reserve_flags.append("--cce-disable-asc-reserved-ubuf")
    if compile_option.cce_disable_vf_stack_reserved_ubuf:
        reserve_flags.append("--cce-disable-vf-stack-reserved-ubuf")
    if reserve_flags:
        # hivmc has no option of its own for these; it forwards whatever this
        # carries on to bisheng, which is where they are understood.
        command.append("--append-bisheng-options=" + " ".join(reserve_flags))
    if compile_option.kernel_mode == "mix":
        command.extend(
            [
                "--disable-ffts",
                f"--link-aicore-bitcode={template_bitcode}",
                "-o",
                str(kernel_binary_path),
            ]
        )
    else:
        command.extend(
            [
                "--disable-ffts",
                "--enable-hivm-compile=False",
                f"--link-aicore-bitcode={template_bitcode}",
                "-o",
                str(kernel_binary_path),
            ]
        )
    return command


def _create_stamped_hivmc_input(
    mlir_path: Path, compile_option: TlaCompileOption
) -> tuple[Path, str | None]:
    """Stamp a private HIVMC input only when debug-print helpers are present.

    Ordinary and external-function kernels rely on ``--link-aicore-bitcode``.
    Debug / ``print_tensor`` helpers also need module attrs
    ``hivm.aiv_bitcode`` / ``hivm.aic_bitcode``, so copy and stamp a private
    ``*.hivmc-input.mlir`` only for those kernels.
    """
    compiler_text = mlir_path.read_text()
    if (
        "tla.debug_print.workspace" not in compiler_text
        and "tla.print_tensor.workspace" not in compiler_text
    ):
        return mlir_path, None
    compiler_input = mlir_path.with_name(f"{mlir_path.stem}.hivmc-input.mlir")
    compiler_input.write_text(compiler_text)
    template_bitcode = _resolve_hivm_template_bitcode(compile_option)
    if "tla.print_tensor.workspace" in compiler_text:
        helper_core_type = (
            _mixed_print_tensor_helper_core(compiler_text)
            if compile_option.kernel_mode == "mix"
            else _core_type_from_arch_scope(compile_option.arch_scope)
        )
        if helper_core_type not in {"aic", "aiv"}:
            raise TlaRuntimeUnavailableError(
                "tla.print_tensor helper split could not be determined"
            )
        aggregate_candidates = [
            Path(path)
            for path in template_bitcode.split(",")
            if Path(path).name == f"meta_op.{helper_core_type}.c310.bc"
        ]
        if len(aggregate_candidates) != 1:
            raise TlaRuntimeUnavailableError(
                f"C310 {helper_core_type} aggregate print tensor bitcode was not selected"
            )
        if _PRINT_TENSOR_HELPER_ABI_MARKER not in aggregate_candidates[0].read_bytes():
            raise TlaRuntimeUnavailableError(
                "C310 aggregate print tensor bitcode does not provide the required ABI marker"
            )
    _stamp_hivm_template_bitcode_attrs(compiler_input, template_bitcode)
    return compiler_input, template_bitcode


def _mixed_print_tensor_helper_core(compiler_text: str) -> str:
    helper_call = re.search(r"\bcall\s+@_mlir_ciface_tla_print_tensor_", compiler_text)
    if helper_call is None:
        raise TlaRuntimeUnavailableError(
            "mixed tla.print_tensor helper call is missing from lowered MLIR"
        )
    functions = list(
        re.finditer(
            r"func\.func(?:\s+private)?\s+@(?P<name>[^\s(]+)\s*\(",
            compiler_text,
        )
    )
    containing_function = [
        match for match in functions if match.start() < helper_call.start()
    ]
    if not containing_function:
        raise TlaRuntimeUnavailableError(
            "mixed tla.print_tensor helper split could not be determined"
        )
    split = re.search(r"_mix_(aic|aiv)$", containing_function[-1].group("name"))
    if split is None:
        raise TlaRuntimeUnavailableError(
            "mixed tla.print_tensor helper call is not inside a split function"
        )
    return split.group(1)


def _stamp_hivm_template_bitcode_attrs(mlir_path: Path, template_bitcode: str) -> None:
    text = mlir_path.read_text()
    additions: list[str] = []
    added_attr_names: set[str] = set()
    for raw_path in template_bitcode.split(","):
        path = Path(raw_path)
        attr = _HIVM_TEMPLATE_BITCODE_ATTRS.get(path.name)
        if attr is None or attr[0] in text or attr[0] in added_attr_names:
            continue
        added_attr_names.add(attr[0])
        additions.append(
            f"{attr[0]} = #hivm.{attr[1].removeprefix('hivm.')}<{json.dumps(str(path))}>"
        )
    if not additions:
        return
    add_text = ", ".join(additions)
    attributes_re = re.compile(
        r"(?P<prefix>\bmodule(?:\s+@[^\s{]+)?\s+attributes\s*\{)"
        r"(?P<body>.*?)"
        r"(?P<suffix>\}\s*\{)",
        re.DOTALL,
    )
    match = attributes_re.search(text)
    if match:
        body = match.group("body").strip()
        new_body = f"{add_text}, {body}" if body else add_text
        mlir_path.write_text(
            text[: match.start()]
            + match.group("prefix")
            + new_body
            + match.group("suffix")
            + text[match.end() :]
        )
        return

    match = re.search(r"(?P<prefix>\bmodule(?:\s+@[^\s{]+)?)(?P<brace>\s*\{)", text)
    if match:
        mlir_path.write_text(
            text[: match.start()]
            + match.group("prefix")
            + f" attributes {{{add_text}}}"
            + match.group("brace")
            + text[match.end() :]
        )


def _mlir_build_dirs() -> list[Path]:
    # dev: CMake build tree; release wheels fall back to the packaged lib dir.
    dsl_root = Path(__file__).resolve().parents[1]
    nested = dsl_root / "csrc" / "mlir" / "build"
    legacy = Path(__file__).resolve().parents[2] / "mlir" / "build"
    packaged = Path(__file__).resolve().parent / "lib"
    return [nested, legacy, packaged]


def _resolve_hivm_template_bitcode(compile_option: TlaCompileOption) -> str:
    return _get_bc_paths(compile_option.kernel_mode)


_HIVMC_DIAGNOSTIC_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<severity>error|warning|remark|note):\s*(?P<message>.+)$",
    re.MULTILINE,
)
_HIVMC_LOCATION_RE = re.compile(
    r'(?:loc\()?(?:"(?P<quoted_filename>[^"\n]+)"|(?P<anonymous>-)):'
    r"(?P<line>[1-9]\d*):(?P<column>\d+)"
)


_MLIR_DEBUG_FILE_LOCATION_RE = re.compile(
    r'loc\("(?P<filename>[^"\n]+)":(?P<line>[1-9]\d*):(?P<column>\d+)\)'
)
_MLIR_DEBUG_LOCATION_ALIAS_RE = re.compile(
    r'^(?P<alias>#loc\d+) = loc\("(?P<filename>[^"\n]+)":'
    r"(?P<line>[1-9]\d*):(?P<column>\d+)\)$",
    re.MULTILINE,
)
_MLIR_DEBUG_LOCATION_ALIAS_REFERENCE_RE = re.compile(
    r'^\s*(?P<alias>#loc\d+) = loc\("[^"\n]+"\((?P<target>#loc\d+)\)\)$',
    re.MULTILINE,
)
_MLIR_DEBUG_LOCATION_REFERENCE_RE = re.compile(r"loc\((?P<alias>#loc\d+)\)")


def _hivmc_input_python_provenance(
    input_mlir: str, *, line: int
) -> BridgeSourceLocation | None:
    """Resolve a Hivmc input-file coordinate to its inline Python location.

    The bridge prints Hivmc input with MLIR debug information, which places
    the operation's ``loc(\"...py\":line:column)`` on the same physical line as
    the operation.  A parse error points at that physical line, so it can be
    mapped without guessing from compiler text or a separate artifact.
    """

    lines = input_mlir.splitlines()
    if line <= 0 or line > len(lines):
        return None
    locations = _MLIR_DEBUG_FILE_LOCATION_RE.finditer(lines[line - 1])
    for match in locations:
        filename = match.group("filename")
        if not filename.endswith(".py"):
            continue
        return BridgeSourceLocation(
            filename=filename,
            line=int(match.group("line")),
            column=int(match.group("column")),
        )
    alias_match = _MLIR_DEBUG_LOCATION_REFERENCE_RE.search(lines[line - 1])
    if alias_match is None:
        return None
    aliases = {
        match.group("alias"): BridgeSourceLocation(
            filename=match.group("filename"),
            line=int(match.group("line")),
            column=int(match.group("column")),
        )
        for match in _MLIR_DEBUG_LOCATION_ALIAS_RE.finditer(input_mlir)
    }
    references = {
        match.group("alias"): match.group("target")
        for match in _MLIR_DEBUG_LOCATION_ALIAS_REFERENCE_RE.finditer(input_mlir)
    }
    alias = alias_match.group("alias")
    seen: set[str] = set()
    while alias not in seen:
        seen.add(alias)
        location = aliases.get(alias)
        if location is not None and location.filename.endswith(".py"):
            return location
        alias = references.get(alias, "")
        if not alias:
            break
    return None


def _same_path(left: str, right: Path) -> bool:
    """Compare diagnostic and input paths without requiring either to exist."""

    try:
        return Path(left).resolve(strict=False) == right.resolve(strict=False)
    except (OSError, ValueError):
        return left == str(right)


def _hivmc_diagnostics(
    compiler_output: Sequence[str],
    *,
    diagnostic_input_mlir: str = "",
    diagnostic_input_path: Path | None = None,
) -> tuple[BridgeDiagnostic, ...]:
    """Decode Hivmc's MLIR diagnostics and their attached operation locations."""

    output = "\n".join(compiler_output)
    matches = tuple(_HIVMC_DIAGNOSTIC_RE.finditer(output))
    records: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(output)
        locations: list[Any] = []
        for location_match in _HIVMC_LOCATION_RE.finditer(output[match.start() : end]):
            filename = location_match.group("quoted_filename") or location_match.group(
                "anonymous"
            )
            location = (
                filename,
                int(location_match.group("line")),
                int(location_match.group("column")),
            )
            if location not in locations:
                locations.append(location)
        records.append(
            {
                "severity": match.group("severity"),
                "message": match.group("message").strip(),
                "locations": locations,
                "rendered": match.group(0).strip(),
            }
        )

    # Hivmc emits the failing operation as the following note when the primary
    # diagnostic itself has no location.  Associate that exact operation
    # provenance with the error so the default output points at user code.
    for index, record in enumerate(records[:-1]):
        next_record = records[index + 1]
        if (
            record["severity"] == "error"
            and not record["locations"]
            and next_record["severity"] == "note"
            and "see current operation" in next_record["message"]
        ):
            record["locations"] = next_record["locations"]

    # Hivmc parse diagnostics identify only the generated input file.  When
    # that file is the exact debug-location-preserving MLIR emitted by the
    # bridge, promote its inline Python location while retaining the generated
    # coordinate as secondary context.
    if diagnostic_input_mlir and diagnostic_input_path is not None:
        for record in records:
            provenance: list[tuple[str, int, int]] = []
            for filename, line, _column in record["locations"]:
                if not _same_path(filename, diagnostic_input_path):
                    continue
                source = _hivmc_input_python_provenance(
                    diagnostic_input_mlir,
                    line=line,
                )
                if source is not None:
                    source_record = (source.filename, source.line, source.column)
                    if source_record not in provenance:
                        provenance.append(source_record)
            if provenance:
                record["locations"] = [
                    *(location for location in provenance),
                    *(
                        location
                        for location in record["locations"]
                        if location not in provenance
                    ),
                ]

    return tuple(
        BridgeDiagnostic(
            severity=record["severity"],
            message=record["message"],
            locations=tuple(
                BridgeSourceLocation(*location) for location in record["locations"]
            ),
            rendered=record["rendered"],
        )
        for record in records
    )


def _run_checked(
    command: list[str],
    *,
    label: str,
    cwd: Path,
    stdin_text: str | None = None,
    diagnostic_input_mlir: str = "",
) -> None:
    try:
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            input=stdin_text,
        )
    except subprocess.CalledProcessError as exc:
        compiler_output = tuple(
            output for output in (exc.stderr or "", exc.stdout or "") if output.strip()
        )
        diagnostics = (
            _hivmc_diagnostics(
                compiler_output,
                diagnostic_input_mlir=diagnostic_input_mlir,
                diagnostic_input_path=Path(command[1]),
            )
            if label == "hivmc-a5" and len(command) > 1
            else ()
        )
        raise TlaKernelCompileError(
            f"{label} failed with exit code {exc.returncode}",
            diagnostics=diagnostics,
            compiler_output=compiler_output,
            verbose_detail=(
                f"{label} failed with exit code {exc.returncode}\n"
                f"cmd: {' '.join(command)}\n"
                f"stdout:\n{exc.stdout or ''}\n"
                f"stderr:\n{exc.stderr or ''}"
            ),
            diagnostic_input_mlir=diagnostic_input_mlir,
        ) from None


def _run_tla_lowering_to_mlir(
    *,
    lowered_module: Any | None,
    tlair_mlir: str,
    mlir_path: Path,
    compile_option: TlaCompileOption | None = None,
) -> Any:
    try:
        return _run_typed_bridge_to_mlir(
            lowered_module=lowered_module,
            mlir_path=mlir_path,
            compile_option=compile_option,
        )
    except TlaCompilerBridgeUnavailableError:
        tla_compile = _resolve_tla_compile()
        if tla_compile is None:
            raise
        return _run_tla_compile_cli_to_mlir(
            tla_compile=tla_compile,
            tlair_mlir=tlair_mlir,
            mlir_path=mlir_path,
            compile_option=compile_option,
        )


def _mark_ub_reserve_released(
    lowered_module: Any, compile_option: "TlaCompileOption | None"
) -> None:
    """Tell the pass the caller took the compiler's 8 KB UB reserve.

    The UB ceiling is enforced in ``tla-lower-ptr``, which sees only the module
    -- bisheng options never reach it. Without this a kernel that legitimately
    fills the whole buffer, having asked for the reserve, is refused before the
    host can say otherwise.

    Only both switches together release it; one alone leaves the other half of
    the reserve in place, and assuming otherwise would let a kernel through that
    faults at launch instead of failing to compile.
    """
    if compile_option is None:
        return
    if not (
        compile_option.cce_disable_asc_reserved_ubuf
        and compile_option.cce_disable_vf_stack_reserved_ubuf
    ):
        return
    from catlass._mlir import ir as mlir_ir

    operation = getattr(lowered_module, "operation", None)
    if operation is None:
        return
    with lowered_module.context:
        operation.attributes["tla.ub_reserve_released"] = mlir_ir.UnitAttr.get()


def _run_typed_bridge_to_mlir(
    *,
    lowered_module: Any | None,
    mlir_path: Path,
    compile_option: TlaCompileOption | None = None,
) -> Any:
    if lowered_module is None:
        raise TlaCompilerBridgeUnavailableError(
            "Python runtime compilation requires a live MLIR module. "
            "String TLA MLIR lowering is not supported."
        )
    _mark_ub_reserve_released(lowered_module, compile_option)
    try:
        # Host surface is a single PRINT_IR switch; bridge still takes
        # pass-print flags — enable full pipeline dump when requested.
        print_ir = bool(compile_option.print_ir) if compile_option else False
        result = lower_tlair_module_to_mlir(
            lowered_module,
            mlir_print_ir_before=(),
            mlir_print_ir_after=(),
            mlir_print_ir_before_all=print_ir,
            mlir_print_ir_after_all=print_ir,
        )
    except BridgeUnavailableError as exc:
        raise TlaCompilerBridgeUnavailableError(str(exc)) from exc
    except BridgeLoweringError as exc:
        raise TlaKernelCompileError(
            "In-process Tla compiler bridge failed.",
            diagnostics=exc.diagnostics,
            pass_ir_dump=exc.pass_ir_dump,
            diagnostic_input_mlir=exc.input_mlir,
        ) from None
    except Exception as exc:
        raise TlaKernelCompileError(
            f"In-process Tla compiler bridge failed.\nerror:\n{exc}"
        ) from exc
    mlir_path.write_text(result.lowered_mlir)
    return result


def _resolve_tla_compile() -> Path | None:
    candidates: list[Path] = []
    for build_dir in _mlir_build_dirs():
        candidates.extend(
            [
                build_dir / "TlaCompile",
                build_dir / "tools" / "tla-compile" / "TlaCompile",
            ]
        )
    which = shutil.which("TlaCompile")
    if which:
        candidates.append(Path(which).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _tla_compile_env() -> dict[str, str]:
    return os.environ.copy()


def _run_tla_compile_cli_to_mlir(
    *,
    tla_compile: Path,
    tlair_mlir: str,
    mlir_path: Path,
    compile_option: TlaCompileOption | None = None,
) -> Any:
    input_path = mlir_path.with_suffix(".tlair.mlir")
    input_path.write_text(tlair_mlir)
    cmd = [str(tla_compile), str(input_path), "-o", str(mlir_path)]
    print_requested = bool(compile_option is not None and compile_option.print_ir)
    if print_requested:
        cmd.append("--mlir-print-ir-before-all")
        cmd.append("--mlir-print-ir-after-all")
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=_tla_compile_env(),
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        compiler_output = tuple(
            output
            for output in (
                "<captured in pass IR dump>" if print_requested else stderr,
                exc.stdout or "",
            )
            if output.strip()
        )
        raise TlaKernelCompileError(
            f"TlaCompile CLI fallback failed with exit code {exc.returncode}",
            pass_ir_dump=stderr if print_requested else "",
            compiler_output=compiler_output,
            verbose_detail=(
                f"TlaCompile CLI fallback failed with exit code {exc.returncode}\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{exc.stdout or ''}\n"
                f"stderr:\n{stderr}"
            ),
        ) from None
    if not mlir_path.exists():
        raise TlaKernelCompileError(
            "TlaCompile CLI fallback completed but did not produce lowered MLIR at "
            f"{mlir_path}"
        )
    return type(
        "TlaLoweringResult",
        (),
        {
            "lowered_mlir": mlir_path.read_text(),
            "pass_ir_dump": completed.stderr if print_requested else "",
        },
    )()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise TlaExecutionError(f"Invalid cache manifest at {path}: {exc}") from exc


def _manifest_extern_dependency_groups_match_current_files(
    manifest: Mapping[str, Any],
    *,
    current_dependency_groups: Mapping[tuple[str, str], frozenset[Path]],
) -> bool:
    groups = manifest.get("extern_dependency_groups")
    if not isinstance(groups, list):
        return False
    try:
        manifest_groups = {
            (group["source_sha256"], group["target"]): {
                Path(snapshot["path"]): snapshot
                for snapshot in group["dependency_snapshots"]
            }
            for group in groups
        }
        # Check the (source, target)s match or not.
        if set(manifest_groups) != set(current_dependency_groups):
            return False
        # Check the dependencies path for each (source, target) match or not.
        for key, current_dependencies in current_dependency_groups.items():
            if set(manifest_groups[key].keys()) != set(current_dependencies):
                return False
        # Check the dependency snapshots for each path match or not.
        all_dependency_snapshots = {
            path: snapshot
            for dependencies in manifest_groups.values()
            for path, snapshot in dependencies.items()
        }
        for path, snapshot in all_dependency_snapshots.items():
            if "sha256" in snapshot:
                if _file_sha256(path) != snapshot["sha256"]:
                    return False
            else:
                stat = path.stat()
                if (
                    stat.st_size != snapshot["size"]
                    or stat.st_mtime_ns != snapshot["mtime_ns"]
                ):
                    return False
        return True
    except (KeyError, TypeError, OSError):
        return False


def _load_current_extern_cache_manifest(
    path: Path,
    *,
    cache_key: str,
    target_arch: str,
    extern_compile_specs: Sequence[ExternCompileSpec],
) -> dict[str, Any] | None:
    manifest = _load_manifest(path) if path.exists() else None
    if manifest is not None and isinstance(
        manifest.get("extern_dependency_groups"), list
    ):
        # Re-resolve every compiled source/target so a newly shadowing header changes
        # its dependency group even when the previously selected file is intact.
        current_dependency_groups: dict[tuple[str, str], frozenset[Path]] = {}
        for source_index, spec in enumerate(extern_compile_specs):
            dependencies_by_target = _scan_ascendc_extern_source_dependencies(
                spec.source,
                source_index=source_index,
                targets=_resolve_extern_targets(
                    core_types=spec.core_types,
                    target_arch=target_arch,
                ),
                user_include_dirs=spec.include_dirs,
            )
            source_sha256 = _extern_source_sha256(spec.source)
            for target_name, dependencies in dependencies_by_target.items():
                current_dependency_groups[(source_sha256, target_name)] = dependencies
        if _manifest_extern_dependency_groups_match_current_files(
            manifest,
            current_dependency_groups=current_dependency_groups,
        ):
            return manifest
    _drop_memory_cached_function(cache_key)
    return None


def _cache_manifest_has_current_debug_print_workspace_abi(
    manifest: Mapping[str, Any],
) -> bool:
    return (
        manifest.get("debug_print_workspace_abi_revision")
        == _DEBUG_PRINT_WORKSPACE_ABI_REVISION
    )


def _cache_manifest_has_current_print_tensor_workspace_abi(
    manifest: Mapping[str, Any],
) -> bool:
    return (
        manifest.get("print_tensor_workspace_abi_revision")
        == _PRINT_TENSOR_WORKSPACE_ABI_REVISION
    )


def _cache_manifest_has_current_workspace_abis(
    manifest: Mapping[str, Any],
) -> bool:
    return (
        _cache_manifest_has_current_debug_print_workspace_abi(manifest)
        and _cache_manifest_has_current_print_tensor_workspace_abi(manifest)
        and manifest.get("debug_tunnel_state_abi_revision")
        == _DEBUG_TUNNEL_STATE_ABI_REVISION
    )


def _prepared_kernel_abi_from_manifest(
    manifest: Mapping[str, Any], *, expected_entrypoint: str | None = None
) -> tuple[KernelAbiLayout, _PreparedAbiPacker]:
    if "kernel_abi" not in manifest or manifest["kernel_abi"] is None:
        raise TlaKernelCompileError(
            "Cached artifact has no compiler-produced kernel_abi descriptor; "
            "the cache predates the current kernel ABI contract."
        )
    try:
        layout = kernel_abi_from_dict(manifest["kernel_abi"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TlaKernelCompileError(
            f"Invalid cached kernel ABI descriptor: {exc}"
        ) from exc
    if layout is None:
        raise TlaKernelCompileError(
            "Invalid cached kernel ABI descriptor: decoded to no layout"
        )
    try:
        packer = _prepare_abi_packer(
            layout,
            expected_entrypoint=(
                expected_entrypoint
                if expected_entrypoint is not None
                else str(manifest.get("entrypoint", ""))
            ),
        )
    except TlaUnsupportedAbiError as exc:
        raise TlaKernelCompileError(
            f"Invalid cached kernel ABI descriptor: {exc}"
        ) from exc
    return layout, packer


def _kernel_abi_from_manifest(
    manifest: Mapping[str, Any], *, expected_entrypoint: str | None = None
) -> KernelAbiLayout:
    layout, _ = _prepared_kernel_abi_from_manifest(
        manifest, expected_entrypoint=expected_entrypoint
    )
    return layout


def _env_truthy(name: str, *, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on", "y"}


from .base_dsl.runtime.ascend import (
    launch_kernel,
    load_acl,
    load_binary,
)
