"""Shared helpers for user-facing DSL operation wrappers.

Also owns the frontend emission context. Kept free of ``runtime`` /
``execution`` / ``types`` imports so ``typing`` → ``op`` stays acyclic
(``runtime`` → ``execution`` → ``typing``).
"""

from __future__ import annotations

import contextvars
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, Iterator, get_type_hints

from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

if TYPE_CHECKING:
    from ..tla.ffi import ExternFunction

_IdentityBinding = tuple[Any, Any]


@dataclass
class _ExternUsage:
    """Declaration and call sites for one original external symbol."""

    function: ExternFunction
    calls: list[tuple[mlir_ir.Operation, str]] = field(default_factory=list)


@dataclass
class _FrontendEmitState:
    """Active frontend emission state."""

    arg_bindings: dict[int, _IdentityBinding]
    category_bindings: dict[int, _IdentityBinding]
    module: Any | None = None
    #: ``mlir.Value`` -> host :class:`~catlass.tla.runtime._Tensor` for execution lowering.
    tensor_host_by_value: dict[Any, Any] = field(default_factory=dict)
    #: ``mlir.Value`` -> structured Tla tensor type descriptor.
    tensor_type_by_value: dict[Any, Any] = field(default_factory=dict)
    #: ``mlir.Value`` -> resolved tensor metadata fields (shape/stride/coord/origin_shape/...).
    tensor_metadata_by_value: dict[Any, dict[str, Any]] = field(default_factory=dict)
    #: ``mlir.Value`` of every L0 tile written by ``tla.copy(..., scale=...)``.
    #: MX-ness is not visible in a tile's type -- an MX fp8 L0 tile looks exactly
    #: like a plain one -- so it is only knowable from how the tile was written.
    #: Recording it here lets tla.mmad and tla.mmad_mx each reject the operands
    #: that belong to the other, which is otherwise a silent wrong-results bug.
    mx_scaled_l0_values: set[Any] = field(default_factory=set)
    mutex_guard_depth: int = 0
    prof_started: bool = False
    prof_ended: bool = False
    #: Stack of enclosing region wrappers, each one of "cube" / "vector" /
    #: "vec.func" (the wrapper's own name).
    active_regions: list[str] = field(default_factory=list)
    #: Stack of ``mode`` values for the enclosing ``tla.vec.func`` regions.
    vec_func_modes: list[str] = field(default_factory=list)
    #: Original symbol -> declaration and all call sites in this kernel.
    extern_usages: dict[str, _ExternUsage] = field(default_factory=dict)
    #: Explicit ``@tla.kernel(core_ratio=...)`` when provided; else ``None``.
    core_ratio: tuple[int, int] | None = None
    #: On-chip bytes already claimed per address-space token (``"ub"``,
    #: ``"l1"``, ...), each value being the next free offset after the
    #: allocations made so far. ``tla.allocate`` adds to it so that a kernel
    #: cannot sum its way past an on-chip capacity while every individual
    #: request looks fine; the bounds themselves live in
    #: :mod:`catlass.base_dsl.arch`.
    allocated_bytes_by_scope: dict[str, int] = field(default_factory=dict)


_FRONTEND_EMIT_STATE: contextvars.ContextVar[_FrontendEmitState | None] = (
    contextvars.ContextVar("tla_frontend_emit_state", default=None)
)


@contextmanager
def _frontend_emission(
    *,
    arg_bindings: dict[int, _IdentityBinding] | None = None,
    category_bindings: dict[int, _IdentityBinding] | None = None,
    tensor_host_by_value: dict[Any, Any] | None = None,
    module: Any | None = None,
    core_ratio: tuple[int, int] | None = None,
) -> Iterator[_FrontendEmitState]:
    """Activate frontend direct-op emission context."""
    state = _FrontendEmitState(
        arg_bindings=dict(arg_bindings or {}),
        category_bindings=dict(category_bindings or {}),
        tensor_host_by_value=dict(tensor_host_by_value or {}),
        module=module,
        core_ratio=core_ratio,
    )
    token = _FRONTEND_EMIT_STATE.set(state)
    try:
        yield state
        if state.prof_started and not state.prof_ended:
            from ..runtime import TlaCoreAPIError

            raise TlaCoreAPIError("tla.prof_start: missing prof_end")
    finally:
        _FRONTEND_EMIT_STATE.reset(token)


def _current_frontend_state() -> _FrontendEmitState | None:
    return _FRONTEND_EMIT_STATE.get()


def _resolve_identity_binding(
    bindings: dict[int, _IdentityBinding], value: Any
) -> Any | None:
    binding = bindings.get(id(value))
    if binding is None or binding[0] is not value:
        return None
    return binding[1]


def _resolve_frontend_bound_value(value: Any) -> Any | None:
    state = _FRONTEND_EMIT_STATE.get()
    if state is None:
        return None
    return _resolve_identity_binding(state.arg_bindings, value)


def _bind_frontend_value(proxy: Any, value: Any) -> None:
    state = _FRONTEND_EMIT_STATE.get()
    if state is None:
        return
    state.arg_bindings[id(proxy)] = (proxy, value)


def _bind_frontend_category(value: Any, category: str) -> None:
    state = _FRONTEND_EMIT_STATE.get()
    if state is None:
        return
    state.category_bindings[id(value)] = (value, category)


def _resolve_frontend_bound_category(value: Any) -> str | None:
    state = _FRONTEND_EMIT_STATE.get()
    if state is None:
        return None
    return _resolve_identity_binding(state.category_bindings, value)


def _has_enclosing_region(kind: str) -> bool:
    """True if some enclosing region is ``kind`` (``cube`` / ``vector`` / ``vec.func``).

    Walks all active region wrappers, so an op nested several levels deep (e.g.
    inside an ``scf.for`` inside a ``tla.vec.func``) still matches. A ``tla.vec.func``
    is always nested inside a ``tla.vector`` (enforced when it is entered), so a
    ``"vector"`` requirement stays satisfied from inside a ``vec.func`` via the
    enclosing region on the stack.
    """
    state = _FRONTEND_EMIT_STATE.get()
    if state is None:
        return True  # No frontend state to inspect; defer to the MLIR verifier.
    return kind in state.active_regions


def _in_simt_vec_func() -> bool:
    """True if the innermost enclosing ``tla.vec.func`` uses SIMT mode."""
    state = _FRONTEND_EMIT_STATE.get()
    if state is None or not state.vec_func_modes:
        return False
    return state.vec_func_modes[-1].lower() == "simt"


# Filled by ``catlass.types`` after marker annotations exist. Lives here so
# ``op`` need not import ``types`` (``types`` → ``typing`` → ``op``).
_ANNOTATION_CATEGORY: dict[Any, str] = {}


def register_annotation_category(annotation: Any, category: str) -> None:
    """Register a return-annotation → frontend category mapping."""
    _ANNOTATION_CATEGORY[annotation] = category


def annotation_to_category(annotation: Any) -> str | None:
    return _ANNOTATION_CATEGORY.get(annotation)


def _capture_user_loc() -> mlir_ir.Location | None:
    frame = inspect.currentframe()
    caller = (
        frame.f_back.f_back
        if frame is not None
        and frame.f_back is not None
        and frame.f_back.f_back is not None
        else None
    )
    try:
        if caller is None:
            return None
        filename, function, lineno, col_offset = _frame_source_info(caller)
        if lineno <= 0:
            return mlir_ir.Location.unknown()
        file_loc = mlir_ir.Location.file(filename, lineno, col_offset)
        return mlir_ir.Location.name(function, childLoc=file_loc)
    finally:
        del caller, frame


def _record_category(value: Any, category: str) -> None:
    _bind_frontend_category(value, category)
    try:
        setattr(value, "__tla_category__", category)
    except (AttributeError, TypeError):
        pass


def dsl_user_op(op_func: Callable[..., Any]) -> Callable[..., Any]:
    """Attach caller source location to user-facing DSL op calls."""
    return_category: str | None = None
    try:
        return_annotation = get_type_hints(op_func, globalns=op_func.__globals__).get(
            "return"
        )
        return_category = annotation_to_category(return_annotation)
    except Exception:
        return_category = None

    @wraps(op_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        loc = kwargs.pop("loc", None)
        if loc is None and _current_frontend_state() is not None:
            loc = _capture_user_loc()
        elif loc is not None and not isinstance(loc, mlir_ir.Location):
            raise TypeError(
                f"loc must be mlir.ir.Location or None, got {type(loc).__name__}"
            )
        result = op_func(*args, loc=loc, **kwargs)
        if return_category is not None and result is not None:
            _record_category(result, return_category)
        return result

    return wrapper


def _frame_source_info(frame: FrameType) -> tuple[str, str, int, int]:
    """Read source positions without fetching unused source lines."""
    info = inspect.getframeinfo(frame, context=0)
    positions = getattr(info, "positions", None)
    return (
        info.filename,
        info.function,
        int(getattr(positions, "lineno", info.lineno) or info.lineno),
        int(getattr(positions, "col_offset", 0) or 0),
    )
