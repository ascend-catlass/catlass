"""CATLASS Python class entry points for ``@tla.jit`` / ``@tla.kernel``."""

from __future__ import annotations

import inspect
from types import FrameType
from typing import Any, Callable

from ..base_dsl import BaseDSL


class CatlassBaseDSL(BaseDSL):
    """TLA decorator surface; delegates wrapping to ``BaseDSL.jit_runner``."""

    @staticmethod
    def _capture_call_site() -> FrameType:
        """Frame where ``@tla.jit`` / ``@tla.kernel`` was written."""
        here = inspect.currentframe()
        # here -> this helper; here.f_back -> jit/kernel; here.f_back.f_back -> call site
        site = None if here is None else here.f_back
        site = None if site is None else site.f_back
        if site is None:
            raise RuntimeError("unable to capture decorator call site")
        return site

    @classmethod
    def kernel(
        cls,
        fn: Callable[..., Any] | None = None,
        *,
        auto_sync: str | None = None,
    ) -> Any:
        """Directory: Decorators
        Description:
            Mark a Python function as a TLA kernel entry. The function body is not
            executed on the Host. Returns a `TlaJitFunction`. Direct invocation of a
            kernel is disabled; compile it explicitly with `tla.compile`, then call the
            returned `JitCompiledFunction` to launch.

            Parameters:
            - *`fn`* (`Callable[..., Any] | None`): The function being decorated.
              Use `@tla.kernel` or `@tla.kernel(auto_sync=...)`; call `tla.kernel(fn)`
              only when decorator syntax is unavailable.
            - *`auto_sync`* (`str | None`): Optional. `"v0"` enables experimental
              automatic in-core synchronization for supported `tla.copy`, `tla.mmad`,
              and `tla.vec.func` accesses. Default `None` (synchronization stays
              explicit).

            Constraints:
            - The decorated function must not be defined with Python `async def`.
            - Keyword options are whitelisted: currently only `auto_sync`.
            - `auto_sync` must be `"v0"` or `None`.
            - With `auto_sync="v0"`:
              - Only pipeline synchronization within one AIC or AIV is generated.
                Cross-core synchronization and thread synchronization inside
                `tla.vec.func` remain explicit.
              - Local `tla.flag` / `tla.set_flag` / `tla.wait_flag` and `tla.mutex` /
                `tla.mutex_guard` cannot be mixed with automatic synchronization.
                `tla.call_extern` is also unsupported.
              - Protected on-chip tensors must originate from `tla.allocate`; tensors
                built by `tla.make_ptr` from raw on-chip addresses are unsupported.
              - UB `tla.scalar_load` / `tla.scalar_store` directly under `tla.vector`
                are unsupported by AutoSync. Place them inside `tla.vec.func` when
                AutoSync is enabled. This placement is not required when AutoSync is
                disabled.
              - Runtime selection among buffers created by `tla.allocate` is supported,
                but switching a carried pointer to another allocation across loop
                iterations and inconsistent multi-buffer allocation order are not.
              - `tla.mmad` `unit_flag` must be provably always zero or always enabled
                with value 2/3. L0C copy `unit_flag` supports only 0 or 3.
              - `tla.print_tensor` and `tla.debug_print` do not receive automatic
                synchronization.
            - Compile with `tla.compile(kernel, *sample_args)` before launching; calling
              the decorated kernel directly raises `TypeError`.

            **Kernel parameter types**

            | Kind | Types | Passed at launch |
            | --- | --- | --- |
            | Tensor | `tla.Tensor` | Yes |
            | Python scalars | `bool` / `int` / `float` | Yes |
            | `tla` scalars | `Bool`, `Int8/16/32/64`, `UInt8/16/32/64`, `Float16/32`, `BFloat16` | Yes |
            | Compile-time constant | `tla.Constexpr[...]` | No |
            | Compile-time function | `tla.Constexpr[Callable[...]]` or `tla.Constexpr` | No; see [Constexpr Callable arguments](#constexpr-callable-arguments) |
            | Struct | `@dataclass` whose fields are among the above | Unpacked by field; Constexpr fields are not passed at launch |

            Example:
            ```python
            @tla.kernel
            def copy_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
                with tla.vector():
                    tla.copy(dst, src)

            @tla.kernel(auto_sync="v0")
            def copy_kernel_auto(src: tla.Tensor, dst: tla.Tensor) -> None:
                with tla.vector():
                    tla.copy(dst, src)

            compiled = tla.compile(copy_kernel, tx, ty, options="--npu-arch 3510")
            compiled(tx, ty, block_num=1)
            ```

        """
        if auto_sync not in (None, "v0"):
            raise ValueError(
                f"tla.kernel auto_sync must be 'v0' or None, got {auto_sync!r}"
            )
        site = cls._capture_call_site()
        if fn is None:
            return BaseDSL.jit_runner(cls, "_kernel_helper", site, auto_sync=auto_sync)
        return BaseDSL.jit_runner(cls, "_kernel_helper", site, fn, auto_sync=auto_sync)

    @classmethod
    def jit(cls, fn: Callable[..., Any] | None = None) -> Any:
        """Directory: Decorators
        Description:
            Mark a Python function as a device-side DSL helper.

            - When called during `@tla.kernel` lowering, the body is inlined into that
              kernel's device IR.
            - May be passed as a [Constexpr Callable](#constexpr-callable-arguments) kernel
              argument, or called by name from a kernel.
            - A top-level Host call runs as ordinary Python.

            Parameters:
            - *`fn`* (`Callable[..., Any] | None`): The function being decorated.
              Use `@tla.jit`; call `tla.jit(fn)` only when decorator syntax is
              unavailable.

            Constraints:
            - May be a plain `def`; must not use `async def`.
            - Accepts no keyword options.
            - No standalone `dump_mlir` / `compile`.
            - Helpers must not call each other recursively.
            - Helpers may contain control flow that the framework rewrites: dynamic
              `if` / `while`, `tla.range`, and similar; this is handled when the helper
              is inlined during `@tla.kernel` lowering.

            Example:
            ```python
            @tla.jit
            def apply_abs(value):
                return tla.abs(value)

            @tla.kernel
            def k(src: tla.Tensor, dst: tla.Tensor) -> None:
                ...
                y = apply_abs(x)  # named call; inlined at compile time

            compiled = tla.compile(k, tx, ty, options="--npu-arch 3510")
            compiled(tx, ty, block_num=1)
            ```

        """
        site = cls._capture_call_site()
        if fn is None:
            return BaseDSL.jit_runner(cls, "_jit_helper", site)
        return BaseDSL.jit_runner(cls, "_jit_helper", site, fn)


class TlaDSL(CatlassBaseDSL):
    """Concrete TLA Python class used by ``catlass.tla``."""


__all__ = ["CatlassBaseDSL", "TlaDSL"]
