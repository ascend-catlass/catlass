"""Declaration API for user-provided TLA device functions."""

from __future__ import annotations

import inspect
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeAlias, get_type_hints

from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

from ..base_dsl.op import dsl_user_op
from ..base_dsl.typing import Numeric, TypedPointer
from ..types import TlaVectorSSATypeDescriptor, dtype_size_bytes


_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


ExternArgType: TypeAlias = TypedPointer | type[Numeric] | TlaVectorSSATypeDescriptor
_IncludeDir: TypeAlias = str | os.PathLike[str]


def _extern_register_type(element_type: str) -> TlaVectorSSATypeDescriptor:
    if element_type not in (
        "f16",
        "bf16",
        "f32",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
    ):
        raise TypeError(
            "extern VectorSSA annotations support Float16, BFloat16, Float32 "
            "and Int/UInt8/16/32/64"
        )
    return TlaVectorSSATypeDescriptor(
        256 // dtype_size_bytes(element_type), element_type
    )


@dataclass(frozen=True)
class ExternFunction:
    """A C ABI entry point supplied by user source code."""

    source: str
    symbol: str
    arg_types: tuple[ExternArgType, ...]
    include_dirs: tuple[Path, ...] = ()
    return_type: TlaVectorSSATypeDescriptor | None = None

    @dsl_user_op
    def __call__(
        self,
        *args: object,
        loc: mlir_ir.Location | None = None,
    ) -> object:
        """Emit a call to this external function during frontend lowering."""

        # Importing here keeps declaration independent from the frontend's
        # comparatively heavy core_api module.
        from ..core_api import _emit_extern_call

        return _emit_extern_call(self, args, loc=loc)


def _validate_symbol(symbol: object) -> str:
    if not isinstance(symbol, str) or _SYMBOL_RE.fullmatch(symbol) is None:
        raise ValueError(f"tla.extern name must be a C identifier, got {symbol!r}")
    return symbol


def _validate_arg_type(annotation: object, *, parameter: str) -> ExternArgType:
    if isinstance(annotation, TlaVectorSSATypeDescriptor):
        if annotation != _extern_register_type(annotation.element_type):
            raise TypeError("extern registers must have all lanes valid (256 bytes)")
        return annotation
    if isinstance(annotation, TypedPointer):
        return annotation
    if (
        isinstance(annotation, type)
        and issubclass(annotation, Numeric)
        and annotation.dtype
    ):
        return annotation
    raise TypeError(
        "tla.extern parameter annotations must be Pointer[dtype, memory_space], "
        "VectorSSA[dtype] or a concrete Numeric type; "
        f"parameter {parameter!r} has {annotation!r}"
    )


def extern(
    *,
    source: str,
    name: str | None = None,
    include_dirs: _IncludeDir | Sequence[_IncludeDir] = (),
) -> Callable[[Callable[..., object]], ExternFunction]:
    """Directory: Decorators
    Description:
        Declare a Ascend C ABI entry point implemented by inline Ascend C source. The
        decorated Python function body is not executed; its name and annotations
        describe the symbol and argument ABI emitted by calls inside a TLA kernel.

    Parameters:
        - *`source`* (`str`): Non-empty Ascend C translation-unit source that
          defines the declared C ABI symbol.
        - *`name`* (`str | None`): Optional C symbol name. Defaults to the
          decorated Python function name.
        - *`include_dirs`* (`str | os.PathLike[str] | Sequence[str | os.PathLike[str]]`):
          Optional include search directory or ordered directories. Relative paths
          are resolved against the file containing the extern declaration.

    Constraints:
        - `name` must be a valid C identifier, and `source` must be non-empty.
        - Parameters must be positional, have no defaults, and be annotated with
          `tla.Pointer[dtype, address_space]`, a concrete TLA numeric type,
          or `tla.VectorSSA[dtype]`.
        - The return annotation must be `None` or one `tla.VectorSSA[dtype]`.
          Supported register dtypes are `Float16`, `BFloat16`, `Float32`, and
          `Int8/16/32/64` / `UInt8/16/32/64`. Every register must have all lanes
          valid: 256 for 8-bit, 128 for 16-bit, 64 for 32-bit, and 32 for 64-bit
          elements (one 256-byte register). The external implementation must fill
          every lane of the result.
        - Register arguments and results must be used within the same SIMD VF.
          The result can feed subsequent TLA operations.
        - Calls must be inside exactly one `tla.cube()` or `tla.vector()` region.
          A SIMD VF containing an extern call must access tensor memory or pass a
          `tla.Pointer` to at least one extern call. An ordinary `Int64` argument counts
          as a scalar. The compiler outlines the VF or reports an error. The extern
          implementation must inline into that VF without starting another VF. SIMT
          VFs do not support extern calls.
        - Within one kernel, a symbol may use only one extern declaration object;
          declarations sharing the same source must use identical ordered
          `include_dirs`.

    Example:
    ```python
    SOURCE = r'''
    #include "kernel_operator.h"
    extern "C" {
    [aicore] __attribute__((always_inline)) void store_value(
        uint64_t dst_addr, int32_t value) {
      auto dst = reinterpret_cast<__gm__ int32_t *>(dst_addr);
      dst[0] = value;
    }
    }
    '''

    @tla.extern(source=SOURCE, include_dirs="include")
    def store_value(
        dst: tla.Pointer[tla.Int32, tla.AddressSpace.gm],
        value: tla.Int32,
    ) -> None: ...

    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        with tla.cube():
            store_value(dst.ptr, 42)
    ```

    """

    # Validity check
    if not isinstance(source, str):
        raise TypeError(f"tla.extern source must be str, got {type(source).__name__}")
    if not source.strip():
        raise ValueError("tla.extern source must not be empty")
    explicit_symbol = None if name is None else _validate_symbol(name)

    def decorator(fn: Callable[..., object]) -> ExternFunction:
        signature = inspect.signature(fn)
        hints = get_type_hints(fn)
        symbol = (
            explicit_symbol
            if explicit_symbol is not None
            else _validate_symbol(fn.__name__)
        )
        source_file = inspect.getsourcefile(fn)
        declaration_dir = (
            Path(source_file).expanduser().resolve().parent
            if source_file
            else Path.cwd()
        )
        include_dir_values = (
            (include_dirs,)
            if isinstance(include_dirs, (str, os.PathLike))
            else include_dirs
        )
        resolved_include_dir_list: list[Path] = []
        for include_dir in include_dir_values:
            path = Path(include_dir).expanduser()
            if not path.is_absolute():
                path = declaration_dir / path
            resolved_include_dir_list.append(path.resolve())
        resolved_include_dirs = tuple(resolved_include_dir_list)

        # Validity check
        return_annotation = hints.get("return", signature.return_annotation)
        return_type = None
        if isinstance(return_annotation, TlaVectorSSATypeDescriptor):
            return_type = _validate_arg_type(return_annotation, parameter="return")
        elif return_annotation not in (None, type(None)):
            if return_annotation is inspect.Signature.empty:
                detail = "is missing"
            else:
                detail = f"must be None or VectorSSA[dtype], got {return_annotation!r}"
            raise TypeError(f"tla.extern return annotation for {fn.__name__} {detail}")

        arg_types: list[ExternArgType] = []
        for parameter in signature.parameters.values():
            # Validity check
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                raise TypeError(
                    "tla.extern parameters must be positional and fixed; "
                    f"parameter {parameter.name!r} has kind {parameter.kind.description}"
                )
            if parameter.default is not inspect.Parameter.empty:
                raise TypeError(
                    "tla.extern parameters must not have default values; "
                    f"parameter {parameter.name!r} has default {parameter.default!r}"
                )
            annotation = hints.get(parameter.name, parameter.annotation)
            if annotation is inspect.Parameter.empty:
                raise TypeError(
                    f"tla.extern parameter {parameter.name!r} is missing an annotation"
                )
            # Add to arg_types after validation
            arg_types.append(_validate_arg_type(annotation, parameter=parameter.name))

        return ExternFunction(
            source=source,
            symbol=symbol,
            arg_types=tuple(arg_types),
            include_dirs=resolved_include_dirs,
            return_type=return_type,
        )

    return decorator


__all__ = [
    "extern",
]
