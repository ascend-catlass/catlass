"""TLA Python runtime entry points and dynamic op helper registry."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from .base_dsl.runtime.dlpack_types import (
    ASCEND_DEVICE_TYPES,
    DLDeviceType,
)

from .execution import (
    TlaBackendCompilerNotFoundError,
    TlaKernelCompileError,
    TlaCompilerBridgeUnavailableError,
    TlaExecutionError,
    TlaExecutionResult,
    TlaRuntimeUnavailableError,
    TlaUnsupportedAbiError,
    compile_and_cache,
    compile_option_for_launch,
    compile_option_from_kwargs,
)
from .base_dsl.op import (
    _FRONTEND_EMIT_STATE,
    _FrontendEmitState,
    _bind_frontend_category,
    _bind_frontend_value,
    _current_frontend_state,
    _frame_source_info,
    _frontend_emission,
    _has_enclosing_region,
    _in_simt_vec_func,
    _resolve_frontend_bound_category,
    _resolve_frontend_bound_value,
    _resolve_identity_binding,
)
from .frontend_diagnostics import FrontendDiagnosticError
from .types import RuntimeTensorError


class TlaIRNotExecutableError(RuntimeError):
    """Raised when attempting to execute APIs that only exist on the lowered TLA MLIR path."""


class TlaCoreAPIError(FrontendDiagnosticError):
    """Raised when a user-facing Tla API call violates preconditions."""


@contextmanager
def _eager_capture() -> Iterator[Any]:
    """Internal minimal frontend session (not a user-facing API).

    Used by :func:`~catlass.tla.runtime.make_fake_tensor` so Host fake construction
    can open a disposable emission context without exposing capture to callers.
    """
    from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

    with mlir_ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with mlir_ir.Location.unknown(ctx):
            module = mlir_ir.Module.create()
            with mlir_ir.InsertionPoint(module.body):
                with _frontend_emission(module=module) as state:
                    yield state


def _require_enclosing_region(op_name: str, kind: str) -> None:
    """Require an enclosing ``tla.<kind>()`` region (cube / vector / vec.func)."""
    if not _has_enclosing_region(kind):
        raise TlaCoreAPIError(f"tla.{op_name} must be nested inside tla.{kind}()")


def _require_enclosing_cube_or_vector(op_name: str) -> None:
    """Require an enclosing tla.cube() or tla.vector() region (either core kind).

    Used by synchronization/mutex/barrier ops, which must sit inside a core
    region but not the bare tla.func scope. Mirrors the MLIR op verifier.
    """
    if not (_has_enclosing_region("cube") or _has_enclosing_region("vector")):
        raise TlaCoreAPIError(
            f"tla.{op_name} must be nested inside tla.cube() or tla.vector()"
        )


def _coerce_bool_value(value: Any) -> Any:
    """Lower a bool-like frontend value to MLIR ``i1`` SSA."""
    from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

    from .base_dsl.typing import Bool, Numeric, as_numeric

    if isinstance(value, Numeric) and type(value) is Bool:
        if isinstance(value.value, mlir_ir.Value):
            return value.value
        return _const_i1(int(bool(value.value)))
    if isinstance(value, bool):
        return _const_i1(int(value))

    def require_scalar_i1(candidate: Any) -> Any:
        if not (
            isinstance(candidate.type, mlir_ir.IntegerType)
            and candidate.type.width == 1
        ):
            raise TlaCoreAPIError(
                f"Expected scalar Bool predicate, got {candidate.type}"
            )
        return candidate

    if isinstance(value, mlir_ir.Value):
        return require_scalar_i1(value)
    resolved = _resolve_frontend_bound_value(value)
    if isinstance(resolved, mlir_ir.Value):
        return require_scalar_i1(resolved)
    # Other Numerics / host scalars: as_numeric then require Bool.
    try:
        num = as_numeric(value) if not isinstance(value, Numeric) else value
    except (TypeError, ValueError) as exc:
        raise TlaCoreAPIError(
            f"Expected bool-like value, got {type(value).__name__}"
        ) from exc
    if type(num) is not Bool:
        raise TlaCoreAPIError(
            f"Expected Bool, got {type(num).__name__}; cast explicitly if needed"
        )
    return _coerce_bool_value(num)


def _const_i1(value: int) -> Any:
    from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

    i1 = mlir_ir.IntegerType.get_signless(1)
    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[i1],
        attributes={"value": mlir_ir.IntegerAttr.get(i1, int(value))},
    )
    return op.results[0]


def _coerce_index_value(value: Any) -> Any:
    from catlass._mlir import ir as mlir_ir

    from .base_dsl.typing import Numeric

    def _to_index_ssa(ssa: Any) -> Any:
        """Cast a signless integer SSA value to ``index`` when needed."""
        if isinstance(ssa.type, mlir_ir.IndexType):
            return ssa
        if (
            mlir_ir.IntegerType.isinstance(ssa.type)
            and mlir_ir.IntegerType(ssa.type).is_signless
        ):
            return mlir_ir.Operation.create(
                "arith.index_cast",
                operands=[ssa],
                results=[mlir_ir.IndexType.get()],
            ).results[0]
        raise TlaCoreAPIError(f"Expected index-like SSA value, got type {ssa.type}")

    # Signed Integer Numeric → index, with ``index_cast`` for element SSA.
    # Reject UInt* and Bool the same way as ``core_api._as_index_value``.
    if isinstance(value, Numeric):
        if not (type(value).is_integer and type(value).signed):
            raise TlaCoreAPIError(
                f"Expected signed integer Numeric index, got {type(value).__name__}; "
                f"cast explicitly with .to(Int32) (or another Int*) before comparing"
            )
        if isinstance(value.value, (int, bool)):
            return _const_index(int(value.value))
        return _to_index_ssa(value.ir_value())

    resolved = _resolve_frontend_bound_value(value)
    if isinstance(resolved, mlir_ir.Value):
        return _to_index_ssa(resolved)
    if isinstance(value, mlir_ir.Value):
        return _to_index_ssa(value)
    if isinstance(value, bool):
        return _const_index(int(value))
    if isinstance(value, int):
        return _const_index(int(value))
    raise TlaCoreAPIError(f"Expected index-like value, got {type(value).__name__}")


def _const_index(value: int) -> Any:
    from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[mlir_ir.IndexType.get()],
        attributes={
            "value": mlir_ir.IntegerAttr.get(mlir_ir.IndexType.get(), int(value))
        },
    )
    return op.results[0]


class _Sentinel:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


class _Pipes:
    SCALAR = _Sentinel("SCALAR")
    VECTOR = _Sentinel("VECTOR")
    CUBE = _Sentinel("CUBE")
    MTE1 = _Sentinel("MTE1")
    MTE2 = _Sentinel("MTE2")
    MTE3 = _Sentinel("MTE3")
    ALL = _Sentinel("ALL")
    FIX = _Sentinel("FIX")


class _RegionStub:
    def __init__(self, display_name: str) -> None:
        self._display_name = display_name

    def __enter__(self) -> None:
        raise TlaIRNotExecutableError(
            f"{self._display_name} is only available in lowered TLA MLIR"
        )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc_type, exc, tb
        return False


def _capture_caller_location() -> Any:
    from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

    frame = inspect.currentframe()
    if frame is None:
        return mlir_ir.Location.unknown()
    try:
        frame = frame.f_back
        while frame is not None:
            filename = frame.f_code.co_filename
            if filename != __file__:
                filename, function, lineno, col_offset = _frame_source_info(frame)
                if lineno <= 0:
                    return mlir_ir.Location.unknown()
                file_loc = mlir_ir.Location.file(filename, lineno, col_offset)
                return mlir_ir.Location.name(function, childLoc=file_loc)
            frame = frame.f_back
    finally:
        del frame
    return mlir_ir.Location.unknown()


def _internal_frontend_for(
    range_value: Any,
    body_fn: Callable[..., Any],
    *carried_values: Any,
    carried_names: tuple[str, ...] | list[str] | None = None,
) -> Any:
    from . import tla_ast_decorators as _ast_decorators

    return _ast_decorators._internal_frontend_for(
        range_value, body_fn, *carried_values, carried_names=carried_names
    )


_VEC_FUNC_MODES = {"simd", "SIMD", "simt", "SIMT"}


def _validate_vec_func_mode(mode: Any) -> None:
    if not isinstance(mode, str):
        raise TlaCoreAPIError(
            f"tla.vec.func: mode must be a string; got {type(mode).__name__}"
        )
    if mode not in _VEC_FUNC_MODES:
        accepted = ", ".join(sorted(repr(value) for value in _VEC_FUNC_MODES))
        raise TlaCoreAPIError(
            f"tla.vec.func: mode must be one of {accepted}; got {mode!r}"
        )


# Mirrors kMaxSimtThreadsPerBlock in TlaOps.cpp; the op verifier enforces the
# same bound for IR that does not come through this frontend.
_MAX_SIMT_THREADS_PER_BLOCK = 2048


def _normalize_vec_func_thread_block_dim(
    thread_block_dim: Any, mode: Any
) -> tuple[int, int, int]:
    """Validate ``thread_block_dim`` and normalize it to an (x, y, z) triple.

    Only meaningful for SIMT mode; an int ``n`` means ``(n, 1, 1)``.
    """
    if str(mode).lower() != "simt":
        raise TlaCoreAPIError(
            f"tla.vec.func: thread_block_dim is only allowed with mode='simt'; got mode={mode!r}"
        )
    if isinstance(thread_block_dim, bool):
        raise TlaCoreAPIError(
            "tla.vec.func: thread_block_dim must be an int or a triple of ints"
        )
    if isinstance(thread_block_dim, int):
        values = (thread_block_dim, 1, 1)
    elif isinstance(thread_block_dim, (tuple, list)):
        if len(thread_block_dim) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in thread_block_dim
        ):
            raise TlaCoreAPIError(
                "tla.vec.func: thread_block_dim must be an int or a triple of ints; "
                f"got {thread_block_dim!r}"
            )
        values = (
            int(thread_block_dim[0]),
            int(thread_block_dim[1]),
            int(thread_block_dim[2]),
        )
    else:
        raise TlaCoreAPIError(
            "tla.vec.func: thread_block_dim must be an int or a triple of ints; "
            f"got {type(thread_block_dim).__name__}"
        )
    if any(value <= 0 for value in values):
        raise TlaCoreAPIError(
            f"tla.vec.func: thread_block_dim must be positive; got {thread_block_dim!r}"
        )
    total = values[0] * values[1] * values[2]
    if total > _MAX_SIMT_THREADS_PER_BLOCK:
        raise TlaCoreAPIError(
            f"tla.vec.func: thread_block_dim describes {total} threads per block, "
            f"more than the supported maximum of {_MAX_SIMT_THREADS_PER_BLOCK}"
        )
    return values


def _internal_frontend_region(
    kind: str,
    body_fn: Callable[[], Any],
    *,
    mode: Any = None,
    thread_block_dim: Any = None,
) -> None:
    from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

    if kind not in {"cube", "vector", "vec.func"}:
        raise TlaIRNotExecutableError(f"Unsupported TLA region wrapper: {kind}")
    if mode is not None:
        if kind != "vec.func":
            raise TlaCoreAPIError(f"tla.{kind}: unexpected mode argument")
        _validate_vec_func_mode(mode)
    thread_dims: tuple[int, int, int] | None = None
    if thread_block_dim is not None:
        if kind != "vec.func":
            raise TlaCoreAPIError(f"tla.{kind}: unexpected thread_block_dim argument")
        thread_dims = _normalize_vec_func_thread_block_dim(thread_block_dim, mode)
    if kind == "vec.func":
        _require_enclosing_region("vec.func", "vector")
    mlir_loc = _capture_caller_location()
    op = mlir_ir.Operation.create(f"tla.{kind}", regions=1, loc=mlir_loc)
    if kind == "vec.func":
        op.attributes["mode"] = mlir_ir.StringAttr.get("simd" if mode is None else mode)
        if thread_dims is not None:
            op.attributes["thread_block_dim"] = mlir_ir.DenseI64ArrayAttr.get(
                list(thread_dims)
            )
    block = op.regions[0].blocks.append()
    state = _FRONTEND_EMIT_STATE.get()
    with mlir_ir.InsertionPoint(block):
        if state is not None:
            state.active_regions.append(kind)
            if kind == "vec.func":
                state.vec_func_modes.append("simd" if mode is None else str(mode))
        try:
            from . import tla_ast_decorators as _ast_decorators

            _ast_decorators._call_with_control_flow_source(body_fn)
        finally:
            if state is not None:
                state.active_regions.pop()
                if kind == "vec.func":
                    state.vec_func_modes.pop()


pipes = _Pipes()
_CORE_API_EXPORTS = (
    "dsl_user_op",
    "arch",
    "vec",
    "mask",
    "create_mask",
    "update_mask",
    "tile_view",
    "get_tile",
    "make_tensor",
    "make_tensor_like",
    "copy",
    "print",
    "flag",
    "cross_flag",
    "cross_core_set_flag",
    "cross_core_wait_flag",
    "set_flag",
    "wait_flag",
    "pipe_barrier",
    "prof_start",
    "prof_end",
    "prof_scope",
    "local_mem_bar",
    "mutex",
    "mutex_guard",
    "range",
    "cube",
    "vector",
    "mmad",
    "mmad_mx",
    "full",
    "arange",
    "add",
    "sub",
    "mul",
    "max",
    "min",
    "div",
    "shift_left",
    "shift_right",
    "where",
    "squeeze",
    "HistogramsType",
    "HistogramsBinType",
    "histograms",
    "bitwise_not",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "exp",
    "log",
    "sqrt",
    "abs",
    "neg",
    "interleave",
    "deinterleave",
    "pack",
    "unpack",
    "RegPart",
    "gather",
    "cmp",
    "make_ptr",
    "allocate",
    "recast_ptr",
    "make_shape",
    "make_coord",
    "make_stride",
    "make_layout",
    "make_layout_with_tag",
    "IndexTree",
    "range_constexpr",
    "VectorSSA",
    "MaskSSA",
)


def const_expr(value: Any) -> bool:
    """Return a Python bool for frontend-time control flow."""
    from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

    resolved = _resolve_frontend_bound_value(value)
    if isinstance(resolved, mlir_ir.Value) or isinstance(value, mlir_ir.Value):
        raise TlaCoreAPIError("tla.const_expr requires a compile-time Python value")
    return bool(value)


def jit(fn: Callable[..., Any]) -> Any:
    """Compat wrapper for the TLA Python jit decorator."""

    from .dsl import jit as _jit

    return _jit(fn)


def kernel(fn: Callable[..., Any]) -> Any:
    """Compat wrapper for the TLA Python kernel decorator."""

    from .dsl import kernel as _kernel

    return _kernel(fn)


def __getattr__(name: str) -> Any:
    if name in _CORE_API_EXPORTS:
        from . import core_api as _core_api

        return getattr(_core_api, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_CORE_API_EXPORTS))


from .tla.runtime import (  # noqa: E402
    DlpackBridgeError,
    export_dlpack_capsule,
    from_dlpack,
    make_fake_tensor,
)


__all__ = [
    "TlaCoreAPIError",
    "TlaIRNotExecutableError",
    "TlaExecutionError",
    "TlaCompilerBridgeUnavailableError",
    "TlaBackendCompilerNotFoundError",
    "TlaKernelCompileError",
    "TlaRuntimeUnavailableError",
    "TlaUnsupportedAbiError",
    "TlaExecutionResult",
    "RuntimeTensorError",
    "DlpackBridgeError",
    "ASCEND_DEVICE_TYPES",
    "DLDeviceType",
    "export_dlpack_capsule",
    "from_dlpack",
    "make_fake_tensor",
    "arch",
    "const_expr",
    "jit",
    "kernel",
    "pipes",
    *_CORE_API_EXPORTS,
]
