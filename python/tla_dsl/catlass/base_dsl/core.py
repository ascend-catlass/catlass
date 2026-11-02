"""BaseDSL scaffold for the frontend rewrite series."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, Sequence

from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

if TYPE_CHECKING:
    from ..execution_lowering import LoweredTlaIR


@dataclass(frozen=True)
class DSLLocation:
    """Source location for IR ops emitted from a Python decorator site."""

    filename: str
    lineno: int
    col_offset: int
    function_name: str


@dataclass
class _LoweringContext:
    """Transient Host JIT state while one Python callable is being lowered.

    Phase-1 only needs the active Python callable, its decorator call site, and
    the last TLA IR produced for that session.
    """

    call_site: DSLLocation | None = None
    python_fn: Callable[..., Any] | None = None
    last_tlair: LoweredTlaIR | None = None


_MANGLE_PUNCT_TABLE = str.maketrans("", "", r"'-![]#,.<>()\":{}=%?@;")
_MANGLE_HEX_ADDR = re.compile(r"0x[a-f0-9]{8,16}")
_MANGLE_WHITESPACE = re.compile(r"\s+")
_MANGLE_MAX_LEN = 180


class BaseDSL:
    """Orchestrates decorator wiring and Host→TLA lowering for one session."""

    @staticmethod
    def get_location_from_frame(frame: FrameType | None) -> DSLLocation | None:
        if frame is None:
            return None
        return DSLLocation(
            filename=frame.f_code.co_filename,
            lineno=int(frame.f_lineno),
            col_offset=0,
            function_name=frame.f_code.co_name,
        )

    @staticmethod
    def jit_runner(
        target_cls: type[BaseDSL],
        executor_name: str,
        frame: FrameType | None,
        *dargs: Any,
        **dkwargs: Any,
    ) -> Any:
        """Wire ``@tla.jit`` / ``@tla.kernel`` onto a user function.

        Public argument checking lives on ``CatlassBaseDSL.kernel`` / ``jit``
        (explicit signatures + ``auto_sync`` values). This helper only applies the
        wrapper and rejects a non-callable target (covers ``dec(fn)`` and
        ``dec()(fn)``).
        """

        call_site = BaseDSL.get_location_from_frame(frame)

        def _wrap(fn: Callable[..., Any]) -> Any:
            from ..dsl import (
                _make_jit_wrapper,
                _make_kernel_wrapper,
                _reject_async_dsl_function,
            )

            if not callable(fn):
                kind = "kernel" if executor_name == "_kernel_helper" else "jit"
                raise TypeError(f"tla.{kind} expects a callable")

            fn._tla_dsl_cls = target_cls  # type: ignore[attr-defined]
            fn._tla_decorator_location = call_site  # type: ignore[attr-defined]
            fn._tla_executor_name = executor_name  # type: ignore[attr-defined]

            if executor_name == "_kernel_helper":
                _reject_async_dsl_function(fn, kind="kernel")
                auto_sync = dkwargs.get("auto_sync")
                core_ratio = dkwargs.get("core_ratio")
                options: dict[str, Any] = {}
                if auto_sync is not None:
                    options["auto_sync"] = auto_sync
                if core_ratio is not None:
                    from ..dsl import parse_core_ratio

                    options["core_ratio"] = parse_core_ratio(core_ratio)
                return _make_kernel_wrapper(fn, location=call_site, options=options)

            if executor_name != "_jit_helper":
                raise ValueError(f"unknown jit_runner executor {executor_name!r}")

            _reject_async_dsl_function(fn, kind="jit")
            return _make_jit_wrapper(fn, location=call_site)

        # Bare ``@dec`` / ``dec(fn)`` vs parameterized ``@dec(...)``.
        match dargs:
            case []:
                return _wrap
            case [candidate]:
                return _wrap(candidate)
            case _:
                kind = "kernel" if executor_name == "_kernel_helper" else "jit"
                raise TypeError(
                    f"tla.{kind}() takes at most one positional argument, "
                    f"got {len(dargs)}"
                )

    def __init__(self) -> None:
        self._ctx = _LoweringContext()

    # Compatibility aliases used by existing BaseDSL call sites in this module.
    @property
    def decorator_location(self) -> DSLLocation | None:
        return self._ctx.call_site

    @decorator_location.setter
    def decorator_location(self, value: DSLLocation | None) -> None:
        self._ctx.call_site = value

    @property
    def funcBody(self) -> Callable[..., Any] | None:
        return self._ctx.python_fn

    @funcBody.setter
    def funcBody(self, value: Callable[..., Any] | None) -> None:
        self._ctx.python_fn = value

    @property
    def _lowered(self) -> LoweredTlaIR | None:
        return self._ctx.last_tlair

    @_lowered.setter
    def _lowered(self, value: LoweredTlaIR | None) -> None:
        self._ctx.last_tlair = value

    def _get_signature(self, python_fn: Callable[..., Any]) -> inspect.Signature:
        """Return the callable signature, evaluating postponed annotations when possible.

        Prefer ``eval_str=True`` so ``Constexpr[...]`` / other postponed forms resolve.
        Fall back without evaluation when the annotation string is not resolvable in
        the function globals (e.g. frontend tokens like ``\"index\"``, or a class
        defined only in an enclosing local scope).
        """
        try:
            return inspect.signature(python_fn, eval_str=True)
        except Exception:
            return inspect.signature(python_fn, eval_str=False)

    def get_ir_location(self, location: DSLLocation | None = None) -> mlir_ir.Location:
        site = location if location is not None else self._ctx.call_site
        if site is None:
            return mlir_ir.Location.unknown()
        file_loc = mlir_ir.Location.file(site.filename, site.lineno, site.col_offset)
        return mlir_ir.Location.name(site.function_name, childLoc=file_loc)

    def _get_function_bound_args(
        self,
        sig: inspect.Signature,
        func_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> inspect.BoundArguments:
        try:
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
        except Exception as exc:  # pragma: no cover - pass-through for callers
            raise RuntimeError(
                f"Failed to bind arguments to function `{func_name}` with signature `{sig}`"
            ) from exc
        return bound_args

    def _require_python_fn(self) -> Callable[..., Any]:
        fn = self._ctx.python_fn
        if fn is None:
            raise RuntimeError("Function body is not set.")
        return fn

    def _canonicalize_args(
        self, sig: inspect.Signature, *args: Any, **kwargs: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        fn = self._require_python_fn()
        bound_args = self._get_function_bound_args(sig, fn.__name__, *args, **kwargs)
        return bound_args.args, bound_args.kwargs

    def _check_arg_count(self, *args: Any, **kwargs: Any) -> inspect.Signature:
        fn = self._require_python_fn()
        sig = self._get_signature(fn)
        bound_args = self._get_function_bound_args(sig, fn.__name__, *args, **kwargs)
        for param in sig.parameters.values():
            if (
                param.default is inspect.Parameter.empty
                and param.name not in bound_args.arguments
            ):
                raise RuntimeError(
                    f"Missing required argument in `{fn.__name__}`: '{param.name}'"
                )
        return sig

    def mangle_name(
        self, function_name: str, args: tuple[Any, ...], args_spec: inspect.FullArgSpec
    ) -> str:
        """Derive a filesystem-/symbol-safe key from ``function_name`` + args."""
        parts: list[str] = [function_name]
        annotations = args_spec.annotations
        for pname, value in zip(args_spec.args, args):
            ann = annotations.get(pname)
            if inspect.isclass(ann):
                parts.append(str(value).replace("class", "").replace(" ", ""))
            elif isinstance(value, (list, tuple)):
                parts.extend(map(str, value))
            else:
                parts.append(str(value))

        symbol = "_".join(parts)
        symbol = symbol.translate(_MANGLE_PUNCT_TABLE)
        symbol = _MANGLE_HEX_ADDR.sub("", symbol)
        symbol = _MANGLE_WHITESPACE.sub(" ", symbol)
        symbol = symbol.replace(" ", "_").replace("\n", "_").replace("/", "_")
        return symbol[:_MANGLE_MAX_LEN]

    def generate_original_ir(self, *args: Any, **kwargs: Any) -> Any:
        return self.generate_mlir(*args, **kwargs)

    def lower_module(
        self,
        python_fn: Callable[..., Any],
        *,
        kind: str,
        options: dict[str, Any] | None = None,
        generic: bool = False,
        type_args: Sequence[Any] | None = None,
        location: DSLLocation | None = None,
    ) -> LoweredTlaIR:
        from ..execution_lowering import lower_jit_to_tlair_module_by_execution

        lowered = lower_jit_to_tlair_module_by_execution(
            python_fn,
            kind=kind,
            options=options or {},
            generic=generic,
            type_args=type_args,
            location=location,
        )
        self._ctx.last_tlair = lowered
        return lowered

    def generate_mlir(
        self,
        python_fn: Callable[..., Any],
        *,
        kind: str,
        options: dict[str, Any] | None = None,
        generic: bool = False,
        type_args: Sequence[Any] | None = None,
        location: DSLLocation | None = None,
    ) -> str:
        return self.lower_module(
            python_fn,
            kind=kind,
            options=options or {},
            generic=generic,
            type_args=type_args,
            location=location,
        ).asm(generic=generic)

    def _bind_active_callable(
        self,
        python_fn: Callable[..., Any],
        location: DSLLocation | None,
    ) -> None:
        self._ctx.python_fn = python_fn
        self._ctx.call_site = location

    def _func(
        self,
        python_fn: Callable[..., Any],
        *,
        kind: str,
        options: dict[str, Any] | None = None,
        generic: bool = False,
        type_args: Sequence[Any] | None = None,
        location: DSLLocation | None = None,
    ) -> str:
        self._bind_active_callable(python_fn, location)
        return self.generate_mlir(
            python_fn,
            kind=kind,
            options=options,
            generic=generic,
            type_args=type_args,
            location=location,
        )

    def _lower(
        self,
        python_fn: Callable[..., Any],
        *,
        kind: str,
        options: dict[str, Any] | None = None,
        generic: bool = False,
        type_args: Sequence[Any] | None = None,
        location: DSLLocation | None = None,
    ) -> LoweredTlaIR:
        self._bind_active_callable(python_fn, location)
        return self.lower_module(
            python_fn,
            kind=kind,
            options=options,
            generic=generic,
            type_args=type_args,
            location=location,
        )


__all__ = ["DSLLocation", "BaseDSL"]
