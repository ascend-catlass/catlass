"""Public TLA Python namespace.

Preferred import::

    import catlass.tla as tla
    from catlass.tla.runtime import from_dlpack

Host bind uses the same ``tla.make_ptr`` / ``tla.make_layout`` / ``tla.make_tensor``
as Kernel (outside ``@tla.kernel`` they are dual-mode Host entries).
``catlass.tla.runtime`` keeps Host-only helpers such as ``from_dlpack`` /
``make_fake_tensor`` (not dual-mode builders like ``make_ptr`` / ``make_tensor``).
"""

from __future__ import annotations

from typing import Any

from .core import lower_copy
from .runtime import (
    DlpackBridgeError,
    export_dlpack_capsule,
    from_dlpack,
    make_fake_tensor,
)
from .tensor import (
    normalize_tile_view_coord,
    scale_tile_coord_by_shape,
)
from .typing import Tensor, TypedTensor

_EXPLICIT_EXPORTS = (
    "experimental",
    "Tensor",
    "TypedTensor",
    "DlpackBridgeError",
    "export_dlpack_capsule",
    "from_dlpack",
    "make_fake_tensor",
    "lower_copy",
    "normalize_tile_view_coord",
    "scale_tile_coord_by_shape",
)

# Parent-package symbols previously exposed via ``import catlass.tla as tla``.
_PARENT_EXPORTS = (
    "kernel",
    "jit",
    "compile",
    "TlaJitFunction",
    "TlaDSL",
    "is_jit_callable",
    "unwrap_jit_callable",
    "JitCompiledFunction",
    "JitModule",
    "TlaIRNotExecutableError",
    "TlaCoreAPIError",
    "TlaExecutionError",
    "TlaCompilerBridgeUnavailableError",
    "TlaBackendCompilerNotFoundError",
    "TlaKernelCompileError",
    "TlaRuntimeUnavailableError",
    "TlaUnsupportedAbiError",
    "TlaExecutionResult",
    "DSLLocation",
    "BaseDSL",
    "const_expr",
    "Constexpr",
    "Pointer",
    "TypedPointer",
    "JitArgument",
    "AddressSpace",
    "extern",
    "Numeric",
    "Integer",
    "Float",
    "as_numeric",
    "cast",
    "DslType",
    "NumericMeta",
    "IntegerMeta",
    "FloatMeta",
    "Bool",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float16",
    "BFloat16",
    "Float8E4M3FN",
    "Float8E5M2",
    "Float4E2M1",
    "Float4E1M2",
    "Float8E8M0",
    "HiFloat8",
    "Float4E2M1x2",
    "Float4E1M2x2",
    "utils",
    "pipes",
    "arch",
    "fp16",
    "bf16",
    "fp32",
    "params",
    "types",
    "core",
    "ReductionOp",
    "ReduceMode",
    "PASSES",
    "__version__",
)

__all__ = list(_EXPLICIT_EXPORTS) + list(_PARENT_EXPORTS)


def __getattr__(name: str) -> Any:
    value: Any
    if name == "experimental":
        from importlib import import_module

        value = import_module(".experimental", __name__)
    elif name in (
        "kernel",
        "jit",
        "compile",
        "TlaJitFunction",
        "TlaDSL",
        "is_jit_callable",
        "unwrap_jit_callable",
    ):
        from .. import dsl as _dsl

        value = getattr(_dsl, name)
    elif name in ("JitCompiledFunction", "JitModule"):
        from ..base_dsl import jit_executor as _jit_executor

        value = getattr(_jit_executor, name)
    elif name in ("DSLLocation", "BaseDSL"):
        from .. import base_dsl as _base_dsl

        value = getattr(_base_dsl, name)
    elif name == "Constexpr" or name in (
        "Pointer",
        "TypedPointer",
        "JitArgument",
        "as_numeric",
        "cast",
        "DslType",
        "NumericMeta",
        "IntegerMeta",
        "FloatMeta",
    ):
        from ..base_dsl import typing as _typing

        value = getattr(_typing, name)
    elif name == "AddressSpace":
        from ..address_space import AddressSpace

        value = AddressSpace
    elif name == "extern":
        from .ffi import extern

        value = extern
    elif name in (
        "Numeric",
        "Integer",
        "Float",
        "Bool",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float16",
        "BFloat16",
        "Float8E4M3FN",
        "Float8E5M2",
        "Float4E2M1",
        "Float4E1M2",
        "Float8E8M0",
        "HiFloat8",
        "Float4E2M1x2",
        "Float4E1M2x2",
    ):
        from .. import types as _types

        value = getattr(_types, name)
    elif name == "types":
        from .. import types as _types

        value = _types
    elif name == "params":
        from .. import params as _params

        value = _params
    elif name == "core":
        from .. import core_api as _core

        value = _core
    elif name == "ReductionOp":
        from ..core_api import ReductionOp

        value = ReductionOp
    elif name == "ReduceMode":
        from ..params import ReduceMode

        value = ReduceMode
    elif name == "PASSES":
        value = ()
    elif name == "__version__":
        try:
            from .._version import __version__ as _version
        except ImportError:  # pragma: no cover
            _version = "0.0.0"
        value = _version
    else:
        # Errors, const_expr, arch/utils/pipes, and core_api helpers live on
        # catlass.runtime (includes Host hot path make_ptr/shape/stride/layout).
        from .. import runtime as _runtime

        try:
            value = getattr(_runtime, name)
        except AttributeError as exc:
            raise AttributeError(
                f"module 'catlass.tla' has no attribute {name!r}"
            ) from exc

    # Cache so subsequent ``tla.make_*`` / dtype / arch lookups are dict hits.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    from .. import runtime as _runtime

    return sorted(
        set(__all__)
        | set(getattr(_runtime, "_CORE_API_EXPORTS", ()))
        | set(dir(_runtime))
    )
