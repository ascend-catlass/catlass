"""Runtime helper objects consumed by the Tla AST preprocessor."""

from __future__ import annotations

import builtins
import warnings
from dataclasses import dataclass
from typing import Any, Callable


class DSLOptimizationWarning(Warning):
    """Warning for TLA Python patterns that may be expensive to compile."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__()

    def __str__(self) -> str:
        return self.message


_STATIC_LOOP_WARNING_THRESHOLD = 64


@dataclass(frozen=True)
class FrontendRange:
    """Frontend-only descriptor for a Tla dynamic range."""

    start: Any
    end: Any
    step: Any

    def __iter__(self) -> Any:
        raise RuntimeError("tla.range is only iterable during lowering")


def range(
    start: Any,
    end: Any | None = None,
    step: Any | None = None,
) -> FrontendRange:
    """Create a frontend-only dynamic range descriptor."""

    if end is None and step is None:
        return FrontendRange(0, start, 1)
    if step is None:
        return FrontendRange(start, end, 1)
    return FrontendRange(start, end, step)


def range_constexpr(
    start: int, end: int | None = None, step: int | None = None
) -> builtins.range:
    """Create a frontend-time static Python range."""

    if end is None and step is None:
        return _checked_range_constexpr(start)
    if step is None:
        return _checked_range_constexpr(start, end)
    return _checked_range_constexpr(start, end, step)


def _checked_range_constexpr(*args: int) -> builtins.range:
    result = builtins.range(*args)
    range_length = len(result)
    if range_length >= _STATIC_LOOP_WARNING_THRESHOLD:
        _warn_static_loop(range_length, stacklevel=4)
    return result


def _warn_static_loop(
    iterations: int, *, once_at_threshold: bool = False, stacklevel: int = 2
) -> None:
    if iterations < _STATIC_LOOP_WARNING_THRESHOLD or (
        once_at_threshold and iterations != _STATIC_LOOP_WARNING_THRESHOLD
    ):
        return
    warnings.warn(
        f"This static loop has {iterations} iterations, which may be very "
        "slow to compile, consider using `tla.range(...)` instead.",
        category=DSLOptimizationWarning,
        stacklevel=stacklevel,
    )


def is_frontend_range(value: Any) -> bool:
    return isinstance(value, FrontendRange)


def while_selector(*, write_args: list[Any] | tuple[Any, ...] = ()) -> Callable:
    """Decorate an AST-generated dynamic while wrapper."""

    def ir_while_loop(func: Callable) -> Any:
        return func(*write_args)

    return ir_while_loop


def while_executor(
    while_before_block: Callable,
    while_after_block: Callable,
    write_args: list[Any] | tuple[Any, ...] = (),
    full_write_args_count: int = 0,
    write_args_names: list[str] | tuple[str, ...] = (),
    captured_names: tuple[str, ...] | None = None,
    captured_values: tuple[Any, ...] | None = None,
) -> Any:
    """Execute an AST-generated dynamic while loop."""

    from .. import tla_ast_decorators

    return tla_ast_decorators._while_execute_dynamic(
        while_before_block,
        while_after_block,
        *write_args,
        carried_names=write_args_names,
        captured_names=captured_names,
        captured_values=captured_values,
        full_write_args_count=full_write_args_count,
    )


__all__ = [
    "DSLOptimizationWarning",
    "FrontendRange",
    "is_frontend_range",
    "range",
    "range_constexpr",
    "while_executor",
    "while_selector",
]
