from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    from ..execution import TlaExecutionResult


@lru_cache(maxsize=1)
def _argument_tree_runtime():
    # Resolve after runtime initialization to avoid the execution import cycle.
    from .runtime import argument_tree

    return argument_tree


def _rectify_legacy_launch_args(launch_args: Sequence[Any]) -> tuple[Any, ...]:
    """Normalize a binder created without a compiled argument-tree plan.

    ``ExecutionArgs`` is also used directly by existing master-level ABI tests
    and callers that provide a hand-built ``KernelAbiLayout``.  Keep that
    direct ABI behavior isolated from the compiled recursive-tree path.
    """

    from .runtime.jit_arg_adapters import (
        JitArgAdapterRegistry,
        _adapt_from_data_ptr,
    )
    from .typing import Numeric, as_numeric, is_constexpr_annotation

    # Tensor/pointer providers are already in the representation consumed by
    # the prepared ABI packer. This is the common kernel-launch fast path.
    if all(hasattr(arg, "__c_pointers__") for arg in launch_args):
        return tuple(launch_args)

    rectified: list[Any] = []
    for arg in launch_args:
        if is_dataclass(arg) and not isinstance(arg, type):
            # This path is used only when no compiled argument tree was
            # supplied.  A compiled kernel always uses the shared tree plan.
            for field in fields(arg):
                if is_constexpr_annotation(field.type):
                    continue
                value = getattr(arg, field.name)
                if isinstance(value, Numeric) or hasattr(value, "__c_pointers__"):
                    rectified.append(value)
                else:
                    rectified.append(as_numeric(value))
            continue
        if isinstance(arg, Numeric) or hasattr(arg, "__c_pointers__"):
            rectified.append(arg)
            continue
        adapter = JitArgAdapterRegistry.get_registered_adapter(arg)
        rectified.append(
            adapter(arg) if adapter is not None else _adapt_from_data_ptr(arg)
        )
    return tuple(rectified)


@dataclass(frozen=True)
class JitFunctionArtifacts:
    """Side artifacts retained for inspection and diagnostics.

    These values are not inputs to ``JitModule`` and are never used to own
    runtime binary or function handles.
    """

    MLIR: str
    LLVM: str
    BINARY: Path
    cache_key: str
    cache_dir: Path
    # None where no CANN installation was found: the frontend and the whole TLA
    # pass pipeline run without one, so a compile can get this far having never
    # resolved hivmc-a5.
    hivmc_path: Path | None
    PASS_IR_DUMP: str | None = None
    compiler_bridge_path: Path | None = None


@dataclass(frozen=True)
class ExecutionArgs:
    """Runtime ABI binder for TLA kernel launch arguments.

    ``original_signature`` (when set) is the pre-Constexpr-strip kernel
    signature; ``signature`` is the filtered runtime copy. Packing always
    requires a compiler-produced ``kernel_abi`` layout so Dynamic-GM / memref
    ABI stays signature-driven.
    """

    original_signature: inspect.Signature | None = None
    kernel_abi: Any | None = None
    abi_packer: Any | None = None
    signature: inspect.Signature | None = None
    # One entry per logical (non-Constexpr) top-level argument when a compiled
    # tree plan is available. ``None`` means the legacy ABI-only binder path.
    argument_trees: tuple[Any | None, ...] | None = None
    _runtime_argument_count: int | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _flat_runtime_arguments: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _direct_abi_arguments: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _adapt_launch_args: Callable[[Sequence[Any]], Sequence[Any]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Resolve after runtime initialization, not on each launch. The adapter
        # reads the live registry so later registrations still take effect.
        from .runtime.jit_arg_adapters import _adapt_registered_launch_args

        object.__setattr__(self, "_adapt_launch_args", _adapt_registered_launch_args)
        if self.original_signature is not None and self.signature is None:
            object.__setattr__(
                self,
                "signature",
                self.filter_runtime_signature(self.original_signature),
            )
        if self.signature is not None:
            runtime_count = sum(
                1
                for param in self.signature.parameters.values()
                if param.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            )
        elif self.argument_trees is not None:
            runtime_count = len(self.argument_trees)
        else:
            runtime_count = None
        object.__setattr__(self, "_runtime_argument_count", runtime_count)
        if (
            self.argument_trees is not None
            and len(self.argument_trees) != runtime_count
        ):
            raise ValueError(
                "compiled runtime argument tree plan does not match kernel "
                f"signature: got {len(self.argument_trees)}, expected {runtime_count}"
            )
        if self.argument_trees is not None:
            from .utils.tree_utils import Leaf

            if all(
                tree is not None
                and isinstance(tree.treedef, Leaf)
                and tree.treedef.is_runtime
                for tree in self.argument_trees
            ):
                object.__setattr__(self, "_flat_runtime_arguments", True)
                object.__setattr__(
                    self,
                    "_direct_abi_arguments",
                    all(
                        tree.bindings[tree.treedef.binding_index].kind
                        in ("pointer", "scalar")
                        for tree in self.argument_trees
                    ),
                )

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., Any],
        *,
        kernel_abi: Any | None = None,
        abi_packer: Any | None = None,
    ) -> "ExecutionArgs":
        """Build a binder from a kernel callable's original signature."""
        from .core import BaseDSL

        return cls(
            original_signature=BaseDSL()._get_signature(fn),
            kernel_abi=kernel_abi,
            abi_packer=abi_packer,
        )

    def filter_runtime_signature(self, sig: inspect.Signature) -> inspect.Signature:
        """Drop static parameters (``Constexpr`` and method receivers)."""
        from .runtime.jit_arg_adapters import is_arg_annotation_constexpr

        filtered_params = []
        for index, (name, param) in enumerate(sig.parameters.items()):
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                filtered_params.append(param)
                continue
            # One rule classifies static parameters, so the launch signature
            # agrees with execution lowering on which slots the compiled plan
            # has: a leading ``self`` / ``cls`` receiver is static too.
            if is_arg_annotation_constexpr(param.annotation, name, index, None):
                continue
            filtered_params.append(param)
        return sig.replace(parameters=filtered_params)

    def get_rectified_args_from_original_args(
        self,
        full_args: Sequence[Any],
        full_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        """Strip Constexpr parameters from a full kernel call argument list."""
        sig = self.original_signature
        runtime_sig = self.signature
        assert sig is not None and runtime_sig is not None

        runtime_arity = self._runtime_argument_count
        assert runtime_arity is not None
        # Already-stripped launch args: do not re-bind against the original
        # signature (Constexpr slots between runtime params would shift).
        if len(full_args) == runtime_arity and not full_kwargs:
            return tuple(full_args)

        bound = sig.bind_partial(*full_args, **dict(full_kwargs or {}))
        bound.apply_defaults()
        for name in sig.parameters:
            if name not in runtime_sig.parameters:
                bound.arguments.pop(name, None)

        runtime_bound = inspect.BoundArguments(runtime_sig, bound.arguments)
        return tuple(runtime_bound.args) + tuple(runtime_bound.kwargs.values())

    def get_rectified_args(
        self, launch_args: Sequence[Any], **_kwargs: Any
    ) -> tuple[Any, ...]:
        """Normalize call args before packing (adapters + passthrough)."""
        if self.argument_trees is not None:
            # Only top-level Constexpr parameters are omitted; zero-leaf trees
            # retain their logical positions.
            expected_count = self._runtime_argument_count
            assert expected_count is not None
            if len(launch_args) != expected_count:
                raise ValueError(
                    "runtime argument count does not match compiled kernel signature: "
                    f"got {len(launch_args)}, expected {expected_count}"
                )
            if self._flat_runtime_arguments:
                return tuple(self._adapt_launch_args(launch_args))

            tree_runtime = _argument_tree_runtime()
            rectified: list[Any] = []
            for arg, tree in zip(launch_args, self.argument_trees, strict=True):
                if tree is not None:
                    rectified.extend(tree_runtime.flatten_runtime_leaves(arg, tree))
                    continue
                rectified.append(arg)
            return tuple(self._adapt_launch_args(rectified))
        return _rectify_legacy_launch_args(launch_args)

    def generate_launch_payload(self, launch_args: Sequence[Any]) -> bytes:
        """Pack ``launch_args`` into the Ascend host launch byte buffer."""
        from .. import execution as execution_mod

        if self.kernel_abi is None:
            raise execution_mod.TlaUnsupportedAbiError(
                "A compiler-produced kernel ABI layout is required before packing "
                "launch arguments."
            )
        # Plain leaves already occupy their ABI positions. The packer checks
        # count, pointer/scalar kinds and scalar ranges without tree replay.
        rectified = (
            self._adapt_launch_args(launch_args)
            if self._direct_abi_arguments
            else self.get_rectified_args(launch_args)
        )
        if self.abi_packer is not None:
            return execution_mod._pack_launch_args_prepared(rectified, self.abi_packer)
        return execution_mod._pack_launch_args(rectified, self.kernel_abi)


@dataclass(frozen=True)
class JitModule:
    """Prepared TLA launch state shared by executors of one compiled function."""

    kernel_binary_path: Path
    entrypoint: str
    kernel_mode: str
    execution_args: ExecutionArgs
    uses_scalar_print: bool
    uses_tensor_print: bool
    is_mixed: bool
    print_metadata: tuple[Any, ...] | None
    print_helper_core: str | None
    print_tensor_position: str | None = None
    simt_print_formats: bytes | None = None
    print_argument_layout: tuple[str, ...] = ()
    aiv_subblock_count: int = 1
    # UB accounting. ub_bytes is the kernel's requirement; the window is what a
    # SIMT kernel declares at launch, and 0 for a kernel that declares nothing.
    ub_static_bytes: int = 0
    ub_programmable_bytes: int = 0
    ub_dynamic_base: int = -1


def _has_dynamic_region(module: "JitModule") -> bool:
    """Whether the kernel declared a launch-sized UB region.

    Carried as the base itself rather than a separate flag: zero is a real base,
    since a kernel can declare a region and allocate nothing statically, so the
    absent case needs a value outside the range.
    """
    return module.ub_dynamic_base >= 0


def _resolved_dyn_ubuf_bytes(
    module: "JitModule",
    requested_ub_bytes: int,
) -> int:
    """UB to declare at launch, and where the cumulative bound is checked.

    The dynamic region is measured from its own base, not from the static byte
    count: the base is the high-water mark rounded up to the region's alignment,
    and a kernel may have a region with no static allocations at all. Exceeding
    the ceiling does not reliably fault -- it corrupts the reserve or the cache
    -- so it is rejected here.
    """
    from .. import execution as execution_mod

    if requested_ub_bytes > 0:
        if not _has_dynamic_region(module):
            raise execution_mod.TlaExecutionError(
                "ub= was given but this kernel declares no dynamic UB region. "
                "Call tla.arch.get_dyn_ub in the kernel, or drop the argument."
            )
        limit = module.ub_programmable_bytes
        available = max(0, limit - module.ub_dynamic_base)
        if requested_ub_bytes > available:
            raise execution_mod.TlaExecutionError(
                f"ub={requested_ub_bytes} is more than this kernel "
                f"can be given: at most {available} bytes are available for the "
                f"dynamic region. It starts at {module.ub_dynamic_base} bytes, "
                f"above the kernel's static allocations, and the kernel's UB "
                f"limit is {limit} bytes."
            )
    if not _has_dynamic_region(module):
        # Nothing to declare: the compiler placed every byte this kernel uses.
        return 0
    # The attribute is measured from the static footprint the binary records,
    # not from the region's base, and the two are not the same address: the base
    # is rounded up to the region pointer's alignment. Declaring the extent alone
    # leaves that rounding gap unbacked, and the kernel writes off the end of what
    # the launch actually handed it -- silently, since nothing faults.
    #
    # Declaring the end offset instead would count the static allocations twice
    # and cost the caller its whole static footprint off the top of the ceiling.
    alignment_gap = max(0, module.ub_dynamic_base - module.ub_static_bytes)
    return alignment_gap + requested_ub_bytes


class JitExecutor:
    """Callable TLA executor owning one loaded binary/function pair."""

    def __init__(self, jit_module: JitModule) -> None:
        from .. import execution as execution_mod
        from .runtime.ascend_stream_adapter import current_device

        self.jit_module = jit_module
        self.device = int(current_device())
        execution_mod.load_acl()
        binary_handle, function_handle = execution_mod.load_binary(
            entrypoint=jit_module.entrypoint,
            kernel_mode=jit_module.kernel_mode,
            kernel_binary_path=jit_module.kernel_binary_path,
        )
        self.binary_handle = int(binary_handle)
        self.function_handle = int(function_handle)

    @property
    def kernel_binary_path(self) -> Path:
        return self.jit_module.kernel_binary_path

    def __call__(
        self,
        *launch_args: Any,
        block_num: int | None = None,
        args: Sequence[Any] | None = None,
        **launch_kwargs: Any,
    ) -> TlaExecutionResult:
        """Launch using this executor's prepared binary state."""
        from .. import execution as execution_mod
        from .runtime.ascend_stream_adapter import current_device

        launch_device = int(current_device())
        if launch_device != self.device:
            raise execution_mod.TlaUnsupportedAbiError(
                f"compiled kernel was loaded on device {self.device}, but the "
                f"current device is {launch_device}; create a new compiled "
                "function after switching devices"
            )

        if launch_args and args is not None:
            raise execution_mod.TlaUnsupportedAbiError(
                "Launch arguments specified multiple times."
            )
        if args is None:
            args = launch_args
        if block_num is None:
            block_num = 1
        if not isinstance(block_num, int):
            raise execution_mod.TlaUnsupportedAbiError("`block_num` must be an int.")

        module = self.jit_module
        dyn_ubuf_bytes = _resolved_dyn_ubuf_bytes(
            module, int(launch_kwargs.get("ub", 0) or 0)
        )
        uses_tensor_print = module.uses_tensor_print
        print_metadata = module.print_metadata
        if uses_tensor_print:
            assert print_metadata is not None
            assert module.print_helper_core is not None
            execution_mod._validate_print_tensor_fifo_capacity(
                print_metadata,
                block_num,
                helper_core=module.print_helper_core,
                mixed=module.is_mixed,
                aiv_subblock_count=module.aiv_subblock_count,
                # Tensor-only launches retain their existing static preflight.
                # Mixed/VF logs have no legacy aggregate metadata and rely on
                # the device's bounded writer instead.
                **(
                    {"bounded": not print_metadata}
                    if module.print_argument_layout
                    else {}
                ),
            )

        payload = module.execution_args.generate_launch_payload(args)
        if not module.print_argument_layout:
            payload = execution_mod._append_debug_print_workspace_payload(
                payload, enabled=module.uses_scalar_print
            )
            if uses_tensor_print:
                extension = bytearray(payload)
                execution_mod._align_payload(extension, execution_mod._POINTER_ABI_SIZE)
                extension.extend(
                    execution_mod._PRINT_TENSOR_WORKSPACE_SENTINEL.to_bytes(
                        execution_mod._POINTER_ABI_SIZE,
                        byteorder="little",
                        signed=False,
                    )
                )
                payload = bytes(extension)
            if module.print_tensor_position in {"L1", "L0C"}:
                extension = bytearray(payload)
                execution_mod._align_payload(extension, execution_mod._POINTER_ABI_SIZE)
                extension.extend(
                    execution_mod._DEBUG_TUNNEL_STATE_SENTINEL.to_bytes(
                        execution_mod._POINTER_ABI_SIZE,
                        byteorder="little",
                        signed=False,
                    )
                )
                payload = bytes(extension)
        raw_stream = launch_kwargs.get("stream")
        if raw_stream is None:
            from .runtime.ascend_stream_adapter import current_stream

            stream = int(current_stream(self.device))
        else:
            from .runtime.ascend_stream_adapter import as_stream

            stream = as_stream(raw_stream, device=self.device)

        def launch() -> None:
            try:
                execution_mod.launch_kernel(
                    function_handle=self.function_handle,
                    stream=int(stream),
                    block_num=block_num,
                    payload=payload,
                    dyn_ubuf_bytes=dyn_ubuf_bytes,
                    uses_scalar_print=module.uses_scalar_print,
                    uses_tensor_print=uses_tensor_print,
                    is_mixed=module.is_mixed,
                    **(
                        {"aiv_subblock_count": module.aiv_subblock_count}
                        if module.is_mixed
                        and (
                            module.uses_scalar_print
                            or uses_tensor_print
                            or module.simt_print_formats is not None
                        )
                        else {}
                    ),
                    print_tensor_position=module.print_tensor_position,
                    device_id=self.device,
                    **(
                        {"print_argument_layout": module.print_argument_layout}
                        if module.print_argument_layout
                        else {}
                    ),
                    **(
                        {"simt_print_formats": module.simt_print_formats}
                        if module.simt_print_formats is not None
                        else {}
                    ),
                )
            except execution_mod.TlaRuntimeUnavailableError as exc:
                from .runtime import ascend as ascend_rt

                # The runtime's own UB accounting is not the one computed here:
                # it reads the binary, whose recorded footprint can differ from
                # what the compiler placed, and it does not expose that figure
                # for us to use. Our bound is therefore an upper bound, and a
                # refusal here is a size problem, not a broken runtime -- say
                # so, rather than leaving a bare ACL code.
                if dyn_ubuf_bytes <= 0 or getattr(exc, "ret", None) != (
                    ascend_rt.ACL_ERROR_INVALID_PARAM
                ):
                    raise
                raise execution_mod.TlaExecutionError(
                    f"the runtime refused a launch declaring {dyn_ubuf_bytes} bytes of "
                    f"dynamic UB, which is within the {module.ub_programmable_bytes}-byte "
                    f"bound computed for this kernel. That bound is an upper "
                    f"bound: the runtime accounts for the binary's own footprint, "
                    f"which the compiler's figure need not match. Retry with a "
                    f"smaller ub=."
                ) from exc

        # The legacy Tensor-only renderer captures stdout and emits only Tensor
        # lines. Shared launches already validate each record in the transport;
        # capturing them here would silently discard scalar and SIMT output.
        if print_metadata is None or (
            module.print_argument_layout
            and (
                module.uses_scalar_print
                or module.simt_print_formats is not None
                or module.print_helper_core == "both"
            )
        ):
            launch()
        else:
            native_output = execution_mod._capture_c_stdout(launch)
            print_block_count = execution_mod._checked_print_tensor_block_count(
                block_num
            )
            helper_core = module.print_helper_core
            assert helper_core is not None
            expected_subblocks: tuple[int | None, ...] = (
                tuple(range(module.aiv_subblock_count))
                if helper_core == "aiv" and module.is_mixed
                else (0,)
                if helper_core == "aiv"
                else (None,)
            )
            decoded = execution_mod._decode_native_print_tensor_records(
                native_output,
                metadata=print_metadata,
                block_count=print_block_count,
                expected_subblocks=expected_subblocks,
            )
            preserve_legacy_format = len(print_metadata) * print_block_count == 1
            for metadata, record, values in decoded:
                print(
                    execution_mod._format_print_tensor_record(
                        values,
                        shape=record.shape,
                        dtype=metadata.dtype,
                        call=None if preserve_legacy_format else metadata.call,
                        block=record.block,
                        position=metadata.position if module.is_mixed else None,
                        subblock=record.subblock,
                    )
                )
            for line in native_output.splitlines():
                if line.startswith("TLA print: output truncated"):
                    print(line)

        return execution_mod.TlaExecutionResult(
            binary_handle=self.binary_handle,
            function_handle=self.function_handle,
            device=self.device,
        )


class JitCompiledFunction:
    """Own compiled launch state and one lazily materialized executor."""

    def __init__(
        self,
        *,
        jit_module: JitModule,
        artifacts: JitFunctionArtifacts,
    ) -> None:
        self.jit_module = jit_module
        self.artifacts = artifacts
        self._executor_lock = threading.Lock()
        self._executor: JitExecutor | None = None

    @property
    def __mlir__(self) -> str:
        return self.artifacts.MLIR

    @property
    def cache_key(self) -> str:
        return self.artifacts.cache_key

    @property
    def kernel_binary_path(self) -> Path:
        return self.jit_module.kernel_binary_path

    @property
    def entrypoint(self) -> str:
        return self.jit_module.entrypoint

    @property
    def kernel_mode(self) -> str:
        return self.jit_module.kernel_mode

    def get_kernel_ub_size(self) -> int:
        """Total Unified Buffer this kernel allocates statically, in bytes.

        The launch-sized region is not counted: its extent is chosen per launch,
        not compiled in.
        """
        return self.jit_module.ub_static_bytes

    @property
    def max_ub_bytes(self) -> int:
        """Largest ``ub=`` this kernel can be launched with.

        An upper bound, not a recommendation: the launch hands over exactly what
        it is asked for, and on a SIMT kernel whatever is left over becomes Data
        Cache. Passing this number claims the buffer and starves the cache.
        """
        module = self.jit_module
        if not _has_dynamic_region(module):
            return 0
        return max(0, module.ub_programmable_bytes - module.ub_dynamic_base)

    @property
    def execution_args(self) -> ExecutionArgs:
        return self.jit_module.execution_args

    def _get_executor(self) -> JitExecutor:
        executor = self._executor
        if executor is not None:
            return executor

        with self._executor_lock:
            if self._executor is None:
                self._executor = JitExecutor(self.jit_module)
            return self._executor

    def __call__(
        self,
        *launch_args: Any,
        block_num: int | None = None,
        args: Sequence[Any] | None = None,
        **launch_kwargs: Any,
    ) -> TlaExecutionResult:
        """Directory: Compile and Launch / Launch
        Description:
            Launch a compiled kernel on the NPU, passing runtime kernel arguments
            and launch options such as `block_num` and `stream`. The executor and
            binary are loaded lazily on the first call and reused thereafter.

        Parameters:
            - *`launch_args`* (`Any`): Positional runtime kernel arguments matching
              the `@tla.kernel` signature (bound Host tensors, scalars, or
              `@dataclass` instances). Mutually exclusive with `args=`.
            - *`block_num`* (`int | None`): Number of blocks to launch. Optional;
              default `1`. Must be an `int` when provided.
            - *`args`* (`Sequence[Any] | None`): Explicit runtime argument sequence.
              Optional; default `None`. Cannot be combined with non-empty
              `*launch_args`.
            - *`stream`* (`Any`, via `**launch_kwargs`): Optional ACL stream handle.
              When omitted, uses the current stream for the executor's device.
            - *`ub`* (`int`, via `**launch_kwargs`): Bytes of UB to back the
              kernel's `tla.arch.get_dyn_ub` region with, i.e. the size of the
              dynamic shared memory the launch provides. Optional, default
              `0`. Rejected unless the kernel declares a region, and rejected
              above `artifact.max_ub_bytes`. Ask for what the kernel uses: the
              launch hands over exactly this much, and on SIMT the UB left
              unclaimed becomes Data Cache.

        Constraints:
            - `*launch_args` and `args=` must not both be non-empty
              (`TlaUnsupportedAbiError`).
            - Launch args must be bound NPU buffers for tensors (`from_dlpack`);
              `make_fake_tensor` samples are for compile only.
            - `block_num` must be an `int` (default `1`).

        Example:
        ```python
        compiled = tla.compile(vadd, tx, ty, options="--npu-arch 3510")
        compiled(tx, ty, block_num=1)
        compiled(args=(tx, ty), block_num=1)
        ```

        """
        return self._get_executor()(
            *launch_args,
            block_num=block_num,
            args=args,
            **launch_kwargs,
        )


__all__ = [
    "ExecutionArgs",
    "JitFunctionArtifacts",
    "JitModule",
    "JitCompiledFunction",
]
