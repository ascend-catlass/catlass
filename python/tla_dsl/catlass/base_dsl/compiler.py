from __future__ import annotations

import ast
import functools
import inspect
import textwrap
from typing import Any, Callable, NoReturn

from .jit_executor import JitCompiledFunction, JitModule


def _check_jit_call_is_pure_forwarding(
    fn: Callable[..., Any], *, class_name: str, error_type: type[Exception]
) -> tuple[str, tuple[str, ...]]:
    """Read a one-line ``__call__`` forward as ``(method name, forwarded names)``."""

    from .ast_preprocessor import _find_function_def

    def reject(reason: str) -> NoReturn:
        raise error_type(
            f"{class_name}.__call__ {reason} A @tla.jit __call__ must be one line "
            "calling self.<kernel>(<its own parameters, in order>)."
        )

    try:
        source, _ = inspect.getsourcelines(fn)
        target = _find_function_def(
            ast.parse(textwrap.dedent("".join(source))), fn.__name__
        )
    except (OSError, TypeError, SyntaxError, ValueError):
        target = None
    if target is None:
        reject("has no readable source.")
    params = [arg.arg for arg in (*target.args.posonlyargs, *target.args.args)]
    if not params:
        reject("takes no parameters.")
    if params[0] not in ("self", "cls"):
        reject(f"does not take a receiver first (found {params[0]!r}).")
    body = target.body[1:] if ast.get_docstring(target) else target.body
    if len(body) != 1:
        reject(f"has {len(body)} statements instead of one.")
    statement = body[0]
    call = statement.value if isinstance(statement, (ast.Return, ast.Expr)) else None
    if not isinstance(call, ast.Call):
        reject("has a single statement, but it is not a call.")
    if call.keywords:
        reject("passes keyword arguments.")
    callee = call.func
    if (
        not isinstance(callee, ast.Attribute)
        or not isinstance(callee.value, ast.Name)
        or callee.value.id != params[0]
    ):
        reject(f"calls {ast.unparse(callee)}, not {params[0]}.<kernel>(...).")
    computed = [ast.unparse(arg) for arg in call.args if not isinstance(arg, ast.Name)]
    if computed:
        reject(f"passes {', '.join(computed)} instead of its own parameter names.")
    forwarded = [arg.id for arg in call.args]
    if forwarded != params[1:]:
        reject(
            f"forwards ({', '.join(forwarded)}) instead of its own parameters "
            f"({', '.join(params[1:])}) in order."
        )
    return callee.attr, tuple(forwarded)


def _resolve_kernel_target(
    func: Any, *, error_type: type[Exception]
) -> tuple[Any, Any]:
    """Return ``(kernel, receiver)`` for one ``tla.compile`` entry."""

    from ..dsl import TlaJitFunction, is_jit_callable, unwrap_jit_callable
    from .runtime.jit_arg_adapters import receiver_argument_name

    if isinstance(func, TlaJitFunction):
        return func, func.receiver
    if isinstance(func, (type, functools.partial)) or inspect.isroutine(func):
        raise error_type(
            "tla.compile expects a @tla.kernel function, or an instance of a class "
            "that declares one."
        )
    class_name = type(func).__name__
    # The class's own ``__call__``, not the metaclass's.
    call = next(
        (vars(k)["__call__"] for k in type(func).__mro__ if "__call__" in vars(k)), None
    )
    if call is None or not is_jit_callable(call):
        raise error_type(
            f"tla.compile() received an instance of {class_name!r}, but that class "
            "declares no @tla.jit __call__. Add one that names the kernel to compile:\n"
            "    @tla.jit\n"
            "    def __call__(self, ...):\n"
            "        self.<kernel>(...)"
        )
    target, forwarded = _check_jit_call_is_pure_forwarding(
        unwrap_jit_callable(call), class_name=class_name, error_type=error_type
    )
    kernel = getattr(type(func), target, None)
    if not isinstance(kernel, TlaJitFunction) or kernel.kind != "kernel":
        raise error_type(
            f"{class_name}.__call__ forwards to {target!r}, which is not one of that "
            "class's @tla.kernel methods."
        )
    params = list(inspect.signature(kernel.fn).parameters.values())
    if receiver_argument_name(params) is not None:
        params = params[1:]
    if len(forwarded) != len(params):
        raise error_type(
            f"{class_name}.__call__ forwards {len(forwarded)} parameters to "
            f"{target!r}, which takes {len(params)} (self excluded)."
        )
    return kernel, func


class CompileCallable:
    """Compile a Tla kernel and return a callable compiled executor."""

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> JitCompiledFunction:
        """Directory: Compile and Launch / Compile
        Description:
            Compile a `@tla.kernel` function and return a callable
            `JitCompiledFunction`. This is the public `tla.compile` entry. Call the
            returned object to launch (`compiled(*tensors, block_num=...)`). Use this
            path when you compile once and launch the same binary repeatedly.

            Parameters:
            - *`func`* (`TlaJitFunction | Any`): Decorated `@tla.kernel` function,
              or a class instance whose `@tla.jit __call__` names its kernel. Required.
            - *`args`* (`Any`): Host tensors / scalars / `@dataclass` instances used
              as compile type samples (e.g. `from_dlpack` or `make_fake_tensor`
              results); they need not be bound NPU buffers.
            - *`kwargs`*: Host compile kwargs. Pass `options="--npu-arch 3510"` to
              select the public chip name; see the option table above for every
              accepted option. Cache / IR dump / force-recompile use
              `CATLASS_DSL_*` environment variables. Launch kwargs such as
              `block_num` / `stream` belong on the returned compiled function, not
              here.

            Constraints:
            - Concurrent calls from multiple processes are not supported.
              Invoke this interface from a single process. Concurrent compilation
              of the same kernel in a shared cache directory can corrupt
              compilation artifacts and cause compilation failures.
            - `func` must be a `@tla.kernel` `TlaJitFunction`, or an instance of a
              class that declares one.
            - A kernel read off an instance (`instance.kernel`) carries that instance
              as its receiver.
            - Passing a class instance compiles the kernel its `@tla.jit __call__`
              forwards to, so the class must declare that `__call__`. The forward must
              be a plain `self.<kernel>(...)` call passing the receiver's own parameters
              in order; any other `@tla.jit __call__` body is rejected.
            - A kernel method's first `self` parameter is compile-time configuration.
              It occupies no runtime ABI slot, and `self.attr` reads fold into the
              kernel as constants while it is staged.

            Example:
            ```python
            compiled = tla.compile(vadd, tx, ty, options="--npu-arch 3510")
            compiled(tx, ty, block_num=1)  # launch; call again to reuse the binary

            class VectorAddWithStaticLength:
                def __init__(self, n_ele: int, vl_ele: int = 64):
                    self.n_ele, self.vl_ele = n_ele, vl_ele

                @tla.jit
                def __call__(self, gm_a, gm_b, gm_c):
                    self.kernel(gm_a, gm_b, gm_c)

                @tla.kernel
                def kernel(self, gm_a, gm_b, gm_c):
                    ...  # gm_c = gm_a + gm_b; the trip count comes from self.n_ele

            compiled = tla.compile(
                VectorAddWithStaticLength(384), gm_a, gm_b, gm_c,
                options="--npu-arch 3510",
            )
            compiled(gm_a, gm_b, gm_c, block_num=1)
            ```

        """
        from ..execution import TlaUnsupportedAbiError
        from ..dsl import _bind_kernel_call_args

        if func is None:
            raise TlaUnsupportedAbiError("Function is not set or invalid.")

        kernel, receiver = _resolve_kernel_target(
            func, error_type=TlaUnsupportedAbiError
        )
        if "type_args" in kwargs:
            raise TypeError(
                "tla.compile() does not accept `type_args`; "
                "pass compile-time sample arguments positionally."
            )
        if receiver is not None:
            if args and args[0] is receiver:
                raise TlaUnsupportedAbiError(
                    "tla.compile() got the receiver instance as its first argument "
                    "too; the instance already rides along as the receiver, so drop "
                    "the extra one."
                )
            args = (receiver, *args)
        # Kernel args passed by name arrive mixed in with the compile options.
        args, kwargs = _bind_kernel_call_args(kernel.fn, args, kwargs)
        return kernel.compile(type_args=args if args else None, **kwargs)


compile = CompileCallable()


__all__ = [
    "JitCompiledFunction",
    "JitModule",
    "CompileCallable",
    "compile",
]
