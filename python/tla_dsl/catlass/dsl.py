"""TLA Python decorators and lowering entry points."""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, replace
import functools
import inspect
from typing import Any, Callable, Mapping, Sequence

from . import runtime as _runtime
from .base_dsl import BaseDSL, DSLLocation
from .base_dsl.compiler import CompileCallable, compile
from .base_dsl.jit_executor import ExecutionArgs, JitCompiledFunction
from .catlass_dsl.catlass import TlaDSL


class _CoreRatioToken:
    """Singleton token for ``@tla.kernel(core_ratio=tla.arch.C1V*)``."""

    __slots__ = ("_name", "_ratio")

    def __init__(self, name: str, ratio: tuple[int, int]) -> None:
        self._name = name
        self._ratio = ratio

    def __repr__(self) -> str:
        return f"tla.arch.{self._name}"

    @property
    def ratio(self) -> tuple[int, int]:
        return self._ratio


#: Cube:Vector = 1:1. Only legal via identity (``tla.arch.C1V1``), not ``(1, 1)``.
C1V1 = _CoreRatioToken("C1V1", (1, 1))
#: Cube:Vector = 1:2. Only legal via identity (``tla.arch.C1V2``), not ``(1, 2)``.
C1V2 = _CoreRatioToken("C1V2", (1, 2))

_CORE_RATIO_BY_ID: dict[int, _CoreRatioToken] = {
    id(C1V1): C1V1,
    id(C1V2): C1V2,
}


_NORMALIZED_CORE_RATIOS: frozenset[tuple[int, int]] = frozenset({(1, 1), (1, 2)})


def parse_core_ratio(
    value: Any, *, allow_normalized_tuple: bool = False
) -> tuple[int, int]:
    """Normalize a user ``core_ratio`` to ``(aic, aiv)``.

    Public surface accepts only ``tla.arch.C1V1`` / ``tla.arch.C1V2``
    (identity). Tuples, strings, and any other value are rejected.

    When ``allow_normalized_tuple`` is true (internal options re-entry after
    decorate-time normalization), ``(1, 1)`` / ``(1, 2)`` are also accepted.
    """
    token = _CORE_RATIO_BY_ID.get(id(value))
    if token is not None and value is token:
        return token.ratio
    if (
        allow_normalized_tuple
        and isinstance(value, tuple)
        and len(value) == 2
        and value in _NORMALIZED_CORE_RATIOS
        and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    ):
        return (value[0], value[1])
    raise ValueError(
        f"tla.kernel core_ratio must be tla.arch.C1V1 or tla.arch.C1V2, got {value!r}"
    )


def _kernel_signature(fn: Callable[..., Any] | None) -> inspect.Signature | None:
    if fn is None:
        return None
    try:
        return BaseDSL()._get_signature(fn)
    except (TypeError, ValueError, NameError):
        return None


#: Compile/launch kwargs consumed by the runtime rather than by the kernel
#: signature. A kernel parameter sharing one of these names is ambiguous, so
#: ``_bind_kernel_call_args`` reports the clash instead of guessing.
_RESERVED_CALL_KWARGS = frozenset(
    {"options", "block_num", "device", "stream", "cache", "cache_dir", "type_args"}
)


def _bind_kernel_call_args(
    fn: Callable[..., Any] | None,
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Resolve a call into one positional host value per kernel parameter.

    Lowering takes host values positionally, but callers write ordinary Python
    calls — by name, out of order, relying on defaults — and kernel arguments
    share one ``**kwargs`` with the runtime options. Binding is delegated to
    ``Signature.bind_partial`` so keyword, keyword-only and defaulted parameters
    follow the same rules they would in a plain Python call.

    Returns the bound values and the kwargs left over for the runtime.
    """
    sig = _kernel_signature(fn)
    if sig is None:
        return tuple(args), dict(kwargs or {})
    params = list(sig.parameters.values())
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    if any(p.kind in variadic for p in params):
        # No fixed parameter list to bind against; leave the call untouched.
        return tuple(args), dict(kwargs or {})

    remaining = dict(kwargs or {})
    by_name = {}
    for param in params[len(args) :]:
        if param.name not in remaining:
            continue
        if param.name in _RESERVED_CALL_KWARGS:
            raise TypeError(
                f"kernel parameter {param.name!r} collides with the runtime "
                f"option of the same name; pass it positionally or rename the "
                f"parameter"
            )
        by_name[param.name] = remaining.pop(param.name)

    try:
        bound = sig.bind_partial(*args, **by_name)
    except TypeError:
        # Not a call this signature accepts; let the arity check report it.
        return tuple(args), dict(kwargs or {})
    bound.apply_defaults()

    # Values are positional from here on, so stop at the first parameter the
    # call left unbound rather than shifting every later value left.
    values: list[Any] = []
    for param in params:
        if param.name not in bound.arguments:
            break
        values.append(bound.arguments[param.name])
    # Hand back anything the truncation dropped, so it is not silently lost.
    for name, value in by_name.items():
        if name not in {p.name for p in params[: len(values)]}:
            remaining[name] = value
    return tuple(values), remaining


def _strip_constexpr_launch_args(
    args: Sequence[Any], fn: Callable[..., Any] | None
) -> tuple[Any, ...]:
    """Drop ``Constexpr`` params from a launch arg list."""
    if fn is None:
        return tuple(args)
    return ExecutionArgs(
        original_signature=BaseDSL()._get_signature(fn)
    ).get_rectified_args_from_original_args(args)


_TLA_JIT_MARKER = "_tla_jit"

_JIT_HELPER_INLINE: contextvars.ContextVar[
    Callable[[Callable[..., Any]], Callable[..., Any]] | None
] = contextvars.ContextVar("tla_jit_helper_inline", default=None)


@contextlib.contextmanager
def _jit_helper_inline(
    transform: Callable[[Callable[..., Any]], Callable[..., Any]],
):
    """While lowering a kernel, inline ``@tla.jit`` helpers via *transform*."""

    token = _JIT_HELPER_INLINE.set(transform)
    try:
        yield
    finally:
        _JIT_HELPER_INLINE.reset(token)


def is_jit_callable(value: Any) -> bool:
    """Return whether *value* is a ``@tla.jit`` helper wrapper."""

    return getattr(value, _TLA_JIT_MARKER, False) is True


def unwrap_jit_callable(value: Any) -> Any:
    """Follow ``__wrapped__`` to the underlying Python function."""

    if not is_jit_callable(value):
        return value
    return inspect.unwrap(value)


def _make_jit_wrapper(
    fn: Callable[..., Any], *, location: DSLLocation | None
) -> Callable[..., Any]:
    """Build a ``@tla.jit`` helper wrapper (Phase-1 inline staging path)."""

    @functools.wraps(fn)
    def jit_wrapper(*args: Any, **kwargs: Any) -> Any:
        inline = _JIT_HELPER_INLINE.get()
        if inline is None:
            return fn(*args, **kwargs)
        return inline(jit_wrapper)(*args, **kwargs)

    setattr(jit_wrapper, _TLA_JIT_MARKER, True)
    setattr(jit_wrapper, "_tla_decorator_location", location)
    return jit_wrapper


@dataclass
class TlaJitFunction:
    """Wrapper for ``@tla.kernel`` functions that emit and execute Tla IR."""

    fn: Callable[..., Any]
    kind: str
    options: Mapping[str, Any]
    decorator_location: DSLLocation | None = None
    #: The instance this kernel was read off, if any.
    receiver: Any | None = None
    _mlir: str | None = None
    _base_dsl: BaseDSL | None = None
    _lowered: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "__wrapped__", self.fn)
        object.__setattr__(self, "__signature__", inspect.signature(self.fn))
        object.__setattr__(
            self,
            "__name__",
            getattr(self.fn, "__name__", type(self).__name__),
        )
        object.__setattr__(
            self,
            "__qualname__",
            getattr(self.fn, "__qualname__", self.__name__),
        )
        object.__setattr__(
            self,
            "__module__",
            getattr(self.fn, "__module__", type(self).__module__),
        )
        annotations = getattr(self.fn, "__annotations__", {})
        object.__setattr__(
            self, "__annotations__", dict(annotations) if annotations else {}
        )

    def __get__(self, obj: Any, objtype: type | None = None) -> TlaJitFunction:
        """Bind the kernel to ``obj`` the way a plain method would."""

        if obj is None:
            return self
        # The memo fields belong to the unbound kernel.
        return replace(self, receiver=obj, _mlir=None, _base_dsl=None, _lowered=None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            "Direct @tla.kernel invocation is disabled; compile explicitly "
            "with `compiled = tla.compile(kernel, *sample_args)`, then launch "
            "with `compiled(*runtime_args, block_num=...)`."
        )

    def compile(
        self, *, type_args: Sequence[Any] | None = None, **kwargs: Any
    ) -> JitCompiledFunction:
        """Directory: Compile and Launch / Compile
        Description:
            Compile this `@tla.kernel` function and return a
            `JitCompiledFunction`.
            Use `tla.compile(fn, *args, options=...)` as the usual Host entry; call
            `.compile()` only when you already hold a `TlaJitFunction` and need the
            compiled-function owner directly.

            Parameters:
            - *`type_args`* (`Sequence[Any] | None`): Host tensors / scalars used as
              compile type samples. Optional; default `None` (no tensor
              specialization).
            - *`kwargs`*: Host compile kwargs. Pass `options="--npu-arch 3510"` to
              select the public chip name; see the option table above for every
              accepted option. Cache / IR-dump knobs use `CATLASS_DSL_*`
              environment variables.

            Constraints:
            - `type_args` are compile-time type samples; they need not be bound NPU
              buffers (`make_fake_tensor` is valid).
            - Pass the public chip name with `options="--npu-arch 3510"`;
              unsupported tokens raise at compile time. Switch options such as
              `--cce-disable-asc-reserved-ubuf` take no value.

            Example:
            ```python
            compiled = my_kernel.compile(
                type_args=[tx, ty],
                options="--npu-arch 3510",
            )
            ```

        """
        compile_option = _runtime.compile_option_from_kwargs(kwargs)
        return _runtime.compile_and_cache(
            self.fn,
            kind=self.kind,
            options=self.options,
            compile_option=compile_option,
            type_args=type_args,
            decorator_location=self.decorator_location,
        )

    @property
    def mlir(self) -> str:
        if self._mlir is None:
            base_dsl = self._base_dsl or BaseDSL()
            lowered = base_dsl._lower(
                self.fn,
                kind=self.kind,
                options=dict(self.options),
                location=self.decorator_location,
            )
            self._lowered = lowered
            self._mlir = lowered.asm()
            self._base_dsl = base_dsl
        return self._mlir

    def dump_mlir(self, *, type_args: Sequence[Any] | None = None) -> str:
        """Directory: Compile and Launch / Inspect
        Description:
            Return the TLA IR (`tlair`) MLIR text for this kernel. Does not compile
            to a device binary and does not launch.

            Parameters:
            - *`type_args`* (`Sequence[Any] | None`): Host tensors / scalars used as
              type samples, same as `.compile()`. Optional; default `None`.

            Constraints:
            - `type_args` follow the same rules as `.compile()`.
            - The returned string is frontend TLA IR (`tlair`), not the HIVM/LLVM
              form stored on `JitCompiledFunction.artifacts.LLVM`.

            Example:
            ```python
            text = my_kernel.dump_mlir(type_args=[fa, fb])
            print(text[:500])
            ```

        """
        base_dsl = self._base_dsl or BaseDSL()
        if type_args is None and self._mlir is not None:
            return self._mlir
        lowered = base_dsl._lower(
            self.fn,
            kind=self.kind,
            options=dict(self.options),
            type_args=type_args,
            location=self.decorator_location,
        )
        mlir = lowered.asm()
        if type_args is None:
            self._mlir = mlir
            self._lowered = lowered
        self._base_dsl = base_dsl
        return mlir


def _make_kernel_wrapper(
    fn: Callable[..., Any],
    *,
    location: DSLLocation | None,
    options: Mapping[str, Any],
) -> TlaJitFunction:
    """Build a ``TlaJitFunction`` kernel wrapper from ``jit_runner``."""

    return TlaJitFunction(
        fn,
        kind="kernel",
        options=dict(options),
        decorator_location=location,
    )


# User-facing decorators: ``@tla.jit`` / ``@tla.kernel`` (docs live on TlaDSL).
jit = TlaDSL.jit
kernel = TlaDSL.kernel


def _reject_async_dsl_function(fn: Callable[..., Any], *, kind: str) -> None:
    unwrapped = inspect.unwrap(fn)
    if not (
        inspect.iscoroutinefunction(unwrapped) or inspect.isasyncgenfunction(unwrapped)
    ):
        return
    filename = inspect.getsourcefile(unwrapped) or "<unknown>"
    try:
        source_lines, lineno = inspect.getsourcelines(unwrapped)
    except (OSError, IOError, TypeError):
        source_lines, lineno = [], 1
    text = source_lines[0] if source_lines else None
    offset = len(text) - len(text.lstrip()) + 1 if text is not None else None
    error = SyntaxError(f"async Tla {kind} functions are not supported")
    error.filename = filename
    error.lineno = lineno
    error.offset = offset
    error.text = text
    raise error


__all__ = [
    "C1V1",
    "C1V2",
    "parse_core_ratio",
    "DSLLocation",
    "BaseDSL",
    "TlaDSL",
    "TlaJitFunction",
    "is_jit_callable",
    "unwrap_jit_callable",
    "_jit_helper_inline",
    "CompileCallable",
    "compile",
    "jit",
    "kernel",
]
