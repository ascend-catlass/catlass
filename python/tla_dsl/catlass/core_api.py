"""Explicit user-facing Tla DSL op helpers with inline preconditions."""

from __future__ import annotations

import builtins as _builtins
import inspect
import math
import sys
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Iterable, NoReturn, Sequence, TypeAlias

from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]
from catlass._mlir._mlir_libs._mlir import (  # type: ignore[import-not-found]
    register_value_caster as _register_value_caster,
)
from catlass._mlir.dialects import arith as _mlir_arith  # type: ignore[import-not-found]

from . import _tla_type_bridge
from .params import CastParams

# Element-type tokens the tla.cast lowering supports: signed ints and the AVE
# float set. Unsigned ints, Bool (i1) and Float64 (f64) are rejected by VectorSSA.to.
#
# f8e8m0 is absent on purpose: it is a block-scale exponent, an opaque byte
# converted by shifting rather than by a cast.
_CAST_FP8_DTYPES = frozenset({"f8e4m3fn", "f8e5m2"})

# Element width per cast token, used to tell a 4x width step from a 2x one.
_CAST_TOKEN_BITS = {
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "f8e4m3fn": 8,
    "f8e5m2": 8,
    "f16": 16,
    "bf16": 16,
    "f32": 32,
}
_CAST_SUPPORTED_DTYPES = (
    frozenset({"i8", "i16", "i32", "i64", "f16", "bf16", "f32"}) | _CAST_FP8_DTYPES
)
from catlass._mlir.dialects import tla as _tla_ops_gen  # type: ignore[import-not-found]
from .base_dsl import ast_helpers as _ast_helpers
from .base_dsl.op import _ExternUsage, dsl_user_op, _capture_user_loc
from .base_dsl.typing import Bool, Float32, Int8, Int16, Int32, Numeric, as_numeric
from .base_dsl.typing import Pointer, TypedPointer
from .tla.tensor import normalize_tile_view_coord
from .tla.typing import Tensor
from . import runtime as _runtime
from .tla.tensor import _Tensor
from .runtime import (
    TlaCoreAPIError,
    TlaIRNotExecutableError,
    _RegionStub,
    _Sentinel,
)
from .execution_lowering import TlaLoweringError
from .types import (
    AddressSpace,
    TlaIndexTreeType,
    TlaLayoutDescriptor,
    TlaMaskSSATypeDescriptor,
    TlaVectorSSATypeDescriptor,
    TlaTensorTypeDescriptor,
    PtrType,
    LayoutType,
    TlaCoord,
    TlaCrossFlag,
    TlaFlag,
    TlaLayout,
    TlaMutex,
    TlaRegion,
    TlaShape,
    TlaStride,
    TlaTensor,
    dtype_size_bytes,
    dtype_size_bits,
    _replace_flat_leaves_in_tree,
)
from .params import (
    AtomicMode,
    RegSlot,
    HF32Mode,
    ComputeOrder,
    CopyL0C2DstParams,
    CopyParams,
    L0C2UBMode,
    MemType,
    QuantMode,
)


_PIPE_VALUES = {
    "scalar",
    "vector",
    "cube",
    "mte1",
    "mte2",
    "mte3",
    "all",
    "mte4",
    "mte5",
    "v2",
    "fix",
    "virtual_mte2_l1a",
    "virtual_mte2_l1b",
    "num",
}
_MISSING = object()
_SUPPORTED_COMPARE_ELEMENT_TYPES = frozenset({"f16", "f32", "i32", "u32"})
_MASK_CMP_MODES = ("lt", "le", "gt", "ge", "eq", "ne")
# The two packed 4-bit cube float formats. Same width, different exponent /
# mantissa split, and the two matmul operands may use different ones.
_FP4_FORMATS = frozenset({"f4e2m1", "f4e1m2"})

_MAKE_TENSOR_SUPPORTED_ELEMENT_TYPES = frozenset(
    # fp8 tiles are staged and computed on like any other element type: the
    # vector routes carry them (Vector/dma.cpp registers the copies) and
    # tla.cast converts them. The packed fp4 formats are narrower still --
    # their tiles are buffered as i8 storage with half the elements; nothing
    # addresses them element-wise. f8e8m0 is the microscaling scale element,
    # only ever moved, never computed on.
    {
        "f4e2m1",
        "f4e1m2",
        "f8e8m0",
        "f16",
        "bf16",
        "f32",
        "i32",
        "u32",
        "i16",
        "u16",
        "i1",
        "i8",
        "u8",
        "f8e4m3fn",
        "f8e5m2",
    }
)


# An MX shared-exponent scale block nests exactly like the zZ layout, but its C0 is fixed by
# the e8m0 scale format (2 elements per C0, 32 bytes per fractal) instead of being
# derived from the element width. Everything else about the layout is identical.
_MX_SCALE_ELE_NUM_PER_C0 = 2
_MX_SCALE_ELE_NUM_PER_FRACTAL = 32


def _check_compare_element_type_supported(op_name: str, element_type: str) -> None:
    if element_type in _SUPPORTED_COMPARE_ELEMENT_TYPES:
        return
    supported = ", ".join(sorted(_SUPPORTED_COMPARE_ELEMENT_TYPES))
    _op_error(
        op_name,
        f"unsupported compare element type {element_type}; "
        f"supported element types are {supported}",
    )


class ReductionOp(Enum):
    ADD = "add"
    MAX = "max"
    MIN = "min"


_SUPPORTED_REDUCTION_ELEMENT_TYPES = frozenset(
    {
        "f16",
        "f32",
        "i16",
        "i32",
        "u16",
        "u32",
    }
)


def _check_reduction_element_type_supported(op_name: str, element_type: str) -> None:
    if element_type in _SUPPORTED_REDUCTION_ELEMENT_TYPES:
        return
    supported = ", ".join(sorted(_SUPPORTED_REDUCTION_ELEMENT_TYPES))
    _op_error(
        op_name,
        f"unsupported reduction element type {element_type}; "
        f"supported element types are {supported}",
    )


class _Shape:
    """SSA wrapper for the **result** of ``make_shape`` (``!tla.shape<...>``).

    Nested structure is expressed only via **tuple trees** in ``make_shape(*components)``;
    do not pass ``_Shape`` / ``_Coord`` / ``_Stride`` as inner components.
    """

    __slots__ = ("_shape_value", "_components")

    def __init__(
        self, *, shape_value: mlir_ir.Value, components: tuple[Any, ...]
    ) -> None:
        self._shape_value = shape_value
        self._components = components


class _Coord:
    """SSA wrapper for the **result** of ``make_coord`` (``!tla.coord<...>``). See ``_Shape`` re: tuple-only nesting."""

    __slots__ = ("_coord_value", "_components")

    def __init__(
        self, *, coord_value: mlir_ir.Value, components: tuple[Any, ...]
    ) -> None:
        self._coord_value = coord_value
        self._components = components


class _Stride:
    """SSA wrapper for ``make_stride`` (``!tla.stride<...>``).

    Like :class:`_Shape` / :class:`_Coord`, stores ``_components`` (the tuple tree passed to
    ``make_stride``) so Python code can recover the same tree as ``make_shape`` without
    reparsing the MLIR type string. :class:`_Layout` does **not** mirror this: layout is a
    single fused SSA value built from shape+stride (+ optional origin) operands.
    """

    __slots__ = ("_stride_value", "_components")

    def __init__(
        self, *, stride_value: mlir_ir.Value, components: tuple[Any, ...]
    ) -> None:
        self._stride_value = stride_value
        self._components = components


class _Layout:
    """SSA wrapper for ``make_layout`` (``!tla.layout<...>``).

    The fused layout SSA is the primary payload. The source ``_Shape`` / ``_Stride`` /
    origin ``_Shape`` wrappers and the resolved layout-tag token are also retained so
    ``make_tensor`` can rebuild the Python layout/tensor descriptor trees without
    re-parsing the MLIR type string (same component trees as MLIR: layout type is derived
    from shape+stride types).
    """

    __slots__ = ("_layout_value", "_shape", "_stride", "_origin_shape", "_layout_tag")

    def __init__(
        self,
        *,
        layout_value: mlir_ir.Value,
        shape: _Shape | None = None,
        stride: _Stride | None = None,
        origin_shape: _Shape | None = None,
        layout_tag: str | None = None,
    ) -> None:
        self._layout_value = layout_value
        self._shape = shape
        self._stride = stride
        self._origin_shape = origin_shape
        self._layout_tag = layout_tag


IndexLike: TypeAlias = int | mlir_ir.Value | Numeric
IndexTree: TypeAlias = IndexLike | tuple["IndexTree", ...]
ShapeLike: TypeAlias = IndexTree
CoordLike: TypeAlias = IndexTree
StrideLike: TypeAlias = IndexTree
MemrefLike: TypeAlias = Tensor | mlir_ir.Value
FlagLike: TypeAlias = mlir_ir.Value
CrossFlagLike: TypeAlias = mlir_ir.Value
MutexLike: TypeAlias = mlir_ir.Value
PipeLike: TypeAlias = str | _Sentinel
AddressSpaceLike: TypeAlias = AddressSpace | _Sentinel
DTypeLike: TypeAlias = mlir_ir.Type | type[Numeric]
LiteralLike: TypeAlias = bool | int | float | str | mlir_ir.Type


class _LayoutTag(_Sentinel):
    """Marks ``tla.arch.*`` values that are valid ``Tensor.layout_tag`` / ``make_tensor_like`` tags."""


@_register_value_caster(PtrType.get_static_typeid(), replace=True)
class _Pointer(Pointer):
    """Concrete JIT pointer for ``!tla.ptr<...>``."""

    __slots__ = ("value", "_ptr_ty", "_alloc_size_bytes")
    __tla_category__ = "pointer"

    def __new__(cls, value: Any, alloc_size_bytes: int | None = None) -> Any:
        # MLIR may assign one TypeID to all unregistered ``!tla.*`` types; the value
        # caster still dispatches on that id, so only wrap true ``!tla.ptr`` SSAs.
        if cls is _Pointer and isinstance(value, _Pointer):
            return value
        if cls is _Pointer and isinstance(value, mlir_ir.Value):
            if not PtrType.isinstance(value.type):
                return value
        return super().__new__(cls)

    def __init__(self, value: Any, alloc_size_bytes: int | None = None) -> None:
        if isinstance(value, _Pointer):
            if alloc_size_bytes is None:
                alloc_size_bytes = value._alloc_size_bytes
            value = value.value
        if not isinstance(value, mlir_ir.Value):
            raise TypeError(
                f"Pointer expects mlir.ir.Value, got {type(value).__name__}"
            )
        if not PtrType.isinstance(value.type):
            raise TypeError(f"Pointer expects !tla.ptr<...>, got {value.type}")
        object.__setattr__(self, "value", value)
        pt = value.type
        if not PtrType.isinstance(pt):
            raise TypeError(f"Pointer expects !tla.ptr typeid, got {pt!r}")
        if not isinstance(pt, PtrType):
            pt = PtrType(pt)
        object.__setattr__(self, "_ptr_ty", pt)
        object.__setattr__(self, "_alloc_size_bytes", alloc_size_bytes)

    def __tla_type__(self) -> str:
        return str(self.value.type)

    def __str__(self) -> str:
        return str(self.value.type)

    def __repr__(self) -> str:
        return f"Pointer({self.value})"

    def __get_mlir_types__(self, context: mlir_ir.Context | None = None) -> list[Any]:
        del context
        return [self.value.type]

    def __extract_mlir_values__(self) -> list[Any]:
        return [self.value]

    def __new_from_mlir_values__(self, values: list[Any]) -> "_Pointer":
        """Rebuild from MLIR SSA values.

        Accepts a single ``mlir.ir.Value`` (``!tla.ptr``) or an existing :class:`_Pointer`;
        ``_Pointer.__init__`` enforces ``!tla.ptr``.
        """
        if len(values) != 1:
            raise ValueError(f"Pointer expects 1 MLIR value, got {len(values)}")
        v0 = values[0]
        if isinstance(v0, _Pointer):
            inner = v0.value
        elif isinstance(v0, mlir_ir.Value):
            inner = v0
        else:
            raise TypeError(
                f"Expected _Pointer or mlir.ir.Value, but got {type(v0).__name__}"
            )
        # Keep rebuilt pointers self-contained; frontend binding is only needed for
        # proxies whose exposed value differs from their underlying SSA value.
        return _Pointer(inner)

    @property
    def dtype(self) -> type[Numeric]:
        return Numeric.from_mlir_type(self.pointee)

    @property
    def alignment(self) -> int:
        return self._ptr_ty.alignment

    @property
    def max_alignment(self) -> int:
        return self.alignment

    @property
    def memspace(self) -> AddressSpace:
        return AddressSpace.from_mlir_token(self._ptr_ty.addrspace)

    @property
    def type(self) -> Any:
        return self.value.type

    @property
    def pointee(self) -> mlir_ir.Type:
        return self._ptr_ty.pointee

    @property
    def addrspace(self) -> AddressSpace:
        """Address space of ``!tla.ptr`` (same as :attr:`memspace`)."""
        return AddressSpace.from_mlir_token(self._ptr_ty.addrspace)

    def __add__(self, other: Any) -> "_Pointer":
        """Offset this pointer by a scalar **element count** (``ptr + n`` / ``n + ptr``).

        Advances the pointer by ``other`` elements of its pointee type (not bytes),
        preserving the pointee type and address space. ``other`` may be a Python ``int``
        or an integer/index SSA value (e.g. a ``tla.range`` loop index).
        """
        _require_frontend_state("ptr_add")
        _require_category("ptr_add", "ptr", self, "pointer", 0)
        loc = _capture_user_loc()
        ctx = loc.context if loc is not None else mlir_ir.Context()
        alloc_size_bytes = getattr(self, "_alloc_size_bytes", None)
        p = _coerce_pointer_arg(self)
        src_ty = p._ptr_ty

        offset_value = _as_index_value(other)
        offset_ty = offset_value.type
        if not (
            isinstance(offset_ty, mlir_ir.IndexType)
            or isinstance(offset_ty, mlir_ir.IntegerType)
        ):
            _op_error(
                "ptr_add",
                f"offset must be an integer or index SSA value, got {offset_ty}",
            )

        out_ptr_ty = PtrType.get(
            src_ty.pointee, src_ty.addrspace, src_ty.alignment, context=ctx
        )
        op = mlir_ir.Operation.create(
            "tla.ptr_add",
            operands=[p.value, offset_value],
            results=[out_ptr_ty],
            loc=loc,
        )
        return _Pointer(op.results[0], alloc_size_bytes=alloc_size_bytes)

    def __radd__(self, other: Any) -> "_Pointer":
        return self.__add__(other)


class _RegisterSSA:
    """Shared one-value protocol for register-resident frontend SSA wrappers."""

    _category = ""
    _expected_type = ""

    @classmethod
    def _matches_register_type(cls, value_type: mlir_ir.Type) -> bool:
        raise NotImplementedError

    def __init__(self, value: mlir_ir.Value) -> None:
        if not isinstance(value, mlir_ir.Value):
            raise TypeError(
                f"{type(self).__name__} expects mlir.ir.Value, "
                f"got {type(value).__name__}"
            )
        if not self._matches_register_type(value.type):
            raise TypeError(
                f"{type(self).__name__} expects {self._expected_type}, got {value.type}"
            )
        self.value = value
        self.__tla_category__ = self._category
        _runtime._bind_frontend_value(self, value)
        _runtime._bind_frontend_category(self, self._category)
        _runtime._bind_frontend_category(value, self._category)

    def __tla_type__(self) -> str:
        return str(self.value.type)

    def __get_mlir_types__(self, context: mlir_ir.Context | None = None) -> list[Any]:
        del context
        return [self.value.type]

    def __extract_mlir_values__(self) -> list[Any]:
        return [self.value]

    def __new_from_mlir_values__(self, values: list[Any]) -> "_RegisterSSA":
        if len(values) != 1 or not isinstance(values[0], mlir_ir.Value):
            raise TlaCoreAPIError(
                f"{type(self).__name__} control-flow reconstruction expects "
                "exactly one MLIR SSA value"
            )
        return type(self)(values[0])


class VectorSSA(_RegisterSSA):
    """Frontend proxy for a register-resident data vector SSA value.

    Arithmetic overloads (inside ``tla.vec.func``) map to the matching Core APIs:

    - ``a + b`` / ``b + a`` → :func:`add`
    - ``a - b`` → :func:`sub`
    - ``a * b`` / ``b * a`` → :func:`mul`
    - ``a / b`` → :func:`div`

    Prefer the operators for unmasked vector–vector / vector–scalar math; use
    ``tla.add`` / ``tla.sub`` / … when you need an explicit ``mask=``.
    """

    _category = "vector_ssa"
    _expected_type = "!tla.vector<NxT>"

    @classmethod
    def _matches_register_type(cls, value_type: mlir_ir.Type) -> bool:
        return _tla_type_bridge.type_is_vector_ssa(value_type)

    def __add__(self, other: Any) -> "VectorSSA":
        return add(self, other)

    def __radd__(self, other: Any) -> "VectorSSA":
        return add(other, self)

    def __sub__(self, other: Any) -> "VectorSSA":
        return sub(self, other)

    def __mul__(self, other: Any) -> "VectorSSA":
        return mul(self, other)

    def __rmul__(self, other: Any) -> "VectorSSA":
        return mul(other, self)

    def __truediv__(self, other: Any) -> "VectorSSA":
        return div(self, other)

    @dsl_user_op
    def reduce(
        self,
        kind: ReductionOp,
        *,
        mask: Any,
        init_value: Any | None = None,
        reduction_profile: Any | None = None,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        """Directory: Vector Compute / Discrete and Aggregate

        Description:
            Reduce this register-resident vector under a mask to a one-lane
            fragment (`tla.ReductionOp.ADD` / `MAX` / `MIN`). Broadcast the
            result back to full width with `tla.full(fragment, dtype=...)`.

            Parameters:
            - `kind` (`ReductionOp`): Reduction kind (`ADD`, `MAX`, or `MIN`). Required.
            - `mask` (`MaskSSA`): Predicates which lanes participate. Required.
            - `init_value`: Reserved; only `None` is supported today. Optional, default `None`.
            - `reduction_profile`: Reserved; only `None` is supported today. Optional, default `None`.

            Constraints:
            - Must be called inside a `@tla.kernel`-decorated kernel function.
            - Must be called inside `tla.vec.func()`.
            - Element type must be one of f16/f32/i16/i32/u16/u32.
            - `init_value` and `reduction_profile` must be omitted or `None`.

            Example:
            ```python
            with tla.vec.func(mode="simd"):
                m = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                total = src_reg.reduce(tla.ReductionOp.ADD, mask=m)
                broadcast = tla.full(total, dtype=tla.Float32)
            ```
        """
        return _emit_vector_reduce(
            self,
            kind,
            mask=mask,
            init_value=init_value,
            reduction_profile=reduction_profile,
            loc=loc,
        )

    @dsl_user_op
    def to(
        self,
        dst_type: Any,
        params: CastParams,
        mask: Any | None = None,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        """Directory: Vector Compute / Type Conversion

        Description:
            Convert a register-resident vector to another element type.

            The lowering supports the signed integers `tla.Int8` .. `tla.Int64`,
            the floats `tla.Float16` / `tla.BFloat16` / `tla.Float32`, and the
            two OCP fp8 formats `tla.Float8E4M3FN` / `tla.Float8E5M2`. Unsigned
            integers, `tla.Bool` (i1), `tla.Float64` and `tla.Float8E8M0` are
            rejected.

            An fp8 operand converts to and from `tla.Float32`, `tla.Float16` and
            `tla.BFloat16`. Only the f32 pair is a single instruction: the f16
            and bf16 pairs are expanded by the lowering into two casts through
            f32, exact on the widening leg and rounding once on the narrowing
            one. `f8e5m2 -> f16` instead uses an exact integer widen and a
            shift, because the two formats share a sign and exponent layout; it
            selects the same elements as the composed path, in two instructions
            rather than five. There is no
            fp8-to-fp8 re-encode -- the exponent ranges differ, which
            makes it a requantisation rather than a cast -- and no fp8-to-integer
            path.

            Parameters:
            - `dst_type` (`type[Numeric]`): Element type to convert to. Required.
            - `params` (`CastParams`): Rounding, saturation and register slot. Required.
            - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`. The
              mask is indexed in the **source** element domain: lane `i` predicates
              source element `i`, whatever the width change. A masked-off element
              contributes zero to the result.

            Constraints:
            - Must be called inside a `@tla.kernel`-decorated kernel function.
            - Must be called inside `tla.vec.func()`.
            - `reg_slot` `TWO` and `THREE` name a pack quarter. They are valid
              on a 4x integer cast and the direct fp8<->f32 instruction.

            Example:
            ```python
            trait = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.NOSAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            with tla.vec.func(mode="simd"):
                wide = narrow.to(tla.Float32, trait)
            ```
        """
        _require_category("cast", "operand", self, "vector_ssa", 0)
        if not (
            isinstance(dst_type, type)
            and issubclass(dst_type, Numeric)
            and dst_type.dtype
        ):
            _op_error(
                "cast",
                f"invalid argument 'dst_type' (position 0): expected a concrete "
                f"Numeric element type, got {_type_name(dst_type)}",
            )
        # The lowering only emits signed-int and {f16,bf16,f32} AVE cast paths, so
        # reject unsigned ints, Bool (i1) and Float64 up front (rather than
        # emitting AVE IR the backend cannot legalize / would treat as signed).
        if dst_type.dtype not in _CAST_SUPPORTED_DTYPES:
            _op_error(
                "cast",
                f"unsupported cast target dtype '{dst_type.dtype}': tla.cast "
                f"supports signed integers (i8/i16/i32/i64) and floats "
                f"(f16/bf16/f32/f8e4m3fn/f8e5m2); unsigned, bool, f64 and "
                f"f8e8m0 are not supported",
            )
        if not isinstance(params, CastParams):
            _op_error(
                "cast",
                f"invalid argument 'params' (position 1): expected CastParams, "
                f"got {_type_name(params)}",
            )
        if mask is not None:
            _require_category("cast", "mask", mask, "mask_ssa", 2)
        _require_frontend_state("cast")
        _runtime._require_enclosing_region("cast", "vec.func")
        operand_value = _as_value(self)
        context = operand_value.type.context
        src_desc = _vector_ssa_type_for_mlir_value(operand_value)
        dst_token = _dtype_to_str(dst_type)
        src_token = src_desc.element_type
        _require_reg_slot_reachable(src_token, dst_token, params.reg_slot)
        result_desc = _cast_result_descriptor(src_desc, dst_token, params)
        with context:
            trait_attr = mlir_ir.DenseI32ArrayAttr.get(params.codes())
        mask_value = _as_value(mask) if mask is not None else None
        if mask_value is not None:
            _require_mask_matches_vector("cast", mask_value, operand_value)
        result = _tla_ops_gen.cast(
            result_desc.to_mlir_type(context),
            operand_value,
            trait_attr,
            mask=mask_value,
            loc=loc,
        )
        return VectorSSA(result)


class MaskSSA(_RegisterSSA):
    """Frontend proxy for a register-resident predicate mask SSA value."""

    _category = "mask_ssa"
    _expected_type = "!tla.mask<N>"

    @classmethod
    def _matches_register_type(cls, value_type: mlir_ir.Type) -> bool:
        return _tla_type_bridge.type_is_mask_ssa(value_type)


class _MutexValue:
    """Frontend proxy for an SSA ``!tla.mutex`` value."""

    def __init__(self, value: mlir_ir.Value, resource: str, mutex_id: int) -> None:
        if not isinstance(value, mlir_ir.Value):
            raise TypeError(
                f"Mutex value expects mlir.ir.Value, got {type(value).__name__}"
            )
        if not _tla_type_bridge.type_is_mutex(value.type):
            raise TypeError(f"Mutex value expects !tla.mutex, got {value.type}")
        self.value = value
        self.resource = resource
        self.id = mutex_id
        self.__tla_category__ = "mutex"
        _runtime._bind_frontend_value(self, value)
        _runtime._bind_frontend_category(self, "mutex")
        _runtime._bind_frontend_category(value, "mutex")

    def __tla_type__(self) -> str:
        return str(self.value.type)

    def __get_mlir_types__(self, context: mlir_ir.Context | None = None) -> list[Any]:
        del context
        return [self.value.type]

    def __extract_mlir_values__(self) -> list[Any]:
        return [self.value]

    def __new_from_mlir_values__(self, values: list[Any]) -> "_MutexValue":
        if len(values) != 1:
            raise ValueError(f"Mutex expects 1 MLIR value, got {len(values)}")
        v0 = values[0]
        if isinstance(v0, _MutexValue):
            inner = v0.value
        elif isinstance(v0, mlir_ir.Value):
            inner = v0
        else:
            raise TypeError(
                f"Expected _MutexValue or mlir.ir.Value, but got {type(v0).__name__}"
            )
        return _MutexValue(inner, self.resource, self.id)

    def lock(self, *, pipe: PipeLike, loc: mlir_ir.Location | None = None) -> None:
        return mutex_lock(self, pipe=pipe, loc=loc)

    def unlock(self, *, pipe: PipeLike, loc: mlir_ir.Location | None = None) -> None:
        return mutex_unlock(self, pipe=pipe, loc=loc)


class _MutexGuard:
    """Context manager that wraps a TLA op block with inferred mutex access."""

    def __init__(
        self, mutexes: tuple[Any, ...], loc: mlir_ir.Location | None = None
    ) -> None:
        self._mutexes = mutexes
        self._loc = loc
        self._state: Any | None = None
        self._block: Any | None = None
        self._start_op_count = 0
        self._entered = False

    def __enter__(self) -> "_MutexGuard":
        state = _runtime._current_frontend_state()
        if state is None:
            raise TlaIRNotExecutableError(
                "tla.mutex_guard is only available in lowered Tla IR"
            )
        for index, mutex_value in enumerate(self._mutexes):
            _require_category(
                "mutex_guard", f"mutex[{index}]", mutex_value, "mutex", index
            )
        try:
            block = mlir_ir.InsertionPoint.current.block
        except Exception as exc:
            raise TlaLoweringError(
                "tla.mutex_guard requires an active MLIR insertion point"
            ) from exc
        self._state = state
        self._block = block
        self._start_op_count = len(list(block.operations))
        state.mutex_guard_depth += 1
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc, tb
        if self._state is not None and self._entered:
            self._state.mutex_guard_depth = _builtins.max(
                0, self._state.mutex_guard_depth - 1
            )
        if exc_type is not None:
            return False
        if self._block is None:
            raise TlaLoweringError("tla.mutex_guard lost its MLIR insertion block")
        block_ops = list(self._block.operations)
        body_ops = block_ops[self._start_op_count :]
        if not body_ops:
            raise TlaLoweringError(
                "tla.mutex_guard body must emit at least one tla.copy or tla.mmad"
            )
        pipe = _infer_mutex_guard_pipe(body_ops)
        first_body_op = _raw_operation(body_ops[0])
        with mlir_ir.InsertionPoint(first_body_op):
            for mutex_value in self._mutexes:
                _emit_mutex_lock_op(mutex_value, pipe=pipe, loc=self._loc)
        for mutex_value in reversed(self._mutexes):
            _emit_mutex_unlock_op(mutex_value, pipe=pipe, loc=self._loc)
        return False


class _Namespace:
    def __init__(self) -> None:
        self._members: dict[str, Callable[..., Any]] = {}

    def _set(self, name: str, value: Callable[..., Any]) -> None:
        self._members[name] = value

    def __getattr__(self, name: str) -> Any:
        try:
            return self._members[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _require_generated(symbol_name: str) -> None:
    if not hasattr(_tla_ops_gen, symbol_name):
        raise RuntimeError(
            f"Generated Tla binding is missing `{symbol_name}`; regenerate "
            "catlass/_mlir/dialects/_tla_ops_gen.py"
        )


def _require_frontend_state(op_name: str) -> None:
    state = _runtime._current_frontend_state()
    if state is None:
        raise TlaIRNotExecutableError(
            f"tla.{op_name} is only available in lowered Tla IR"
        )


def _region_stub(op_name: str) -> _RegionStub:
    return _RegionStub(f"tla.{op_name}")


def _resolve_bound_value(value: Any) -> Any:
    """Resolve frontend proxy bindings.

    Keep ``Numeric`` / ``VectorSSA`` / ``MaskSSA`` wrappers intact (user-facing
    object stays typed; use ``_as_value`` / ``ir_value`` to obtain SSA).
    """
    if isinstance(value, mlir_ir.Value):
        return value
    if isinstance(value, (Numeric, VectorSSA, MaskSSA)):
        return value
    bound = _runtime._resolve_frontend_bound_value(value)
    if bound is not None:
        return bound
    return value


def _coerce_pointer_arg(x: Any) -> _Pointer:
    """Resolve frontend bindings, then same path as :meth:`_Pointer.__new_from_mlir_values__`."""
    return _Pointer.__new_from_mlir_values__(None, [_resolve_bound_value(x)])  # type: ignore[arg-type]


def _const_bool(value: bool) -> mlir_ir.Value:
    i1_type = mlir_ir.IntegerType.get_signless(1)
    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[i1_type],
        attributes={"value": mlir_ir.IntegerAttr.get(i1_type, bool(value))},
    )
    return op.results[0]


def _as_i1_value(value: Any) -> mlir_ir.Value:
    # Bare bool and Bool Numeric both lower to i1.
    if isinstance(value, bool):
        return _const_bool(value)
    if isinstance(value, Numeric) and type(value) is Bool:
        return value.ir_value()
    raise TlaLoweringError(f"value expected to be a bool, got {type(value).__name__}")


def _const_index(value: int, *, loc: mlir_ir.Location | None = None) -> mlir_ir.Value:
    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[mlir_ir.IndexType.get()],
        attributes={
            "value": mlir_ir.IntegerAttr.get(mlir_ir.IndexType.get(), int(value))
        },
        loc=loc,
    )
    return op.results[0]


def _const_i64(value: int, *, loc: mlir_ir.Location | None = None) -> mlir_ir.Value:
    i64_type = mlir_ir.IntegerType.get_signless(64)
    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[i64_type],
        attributes={"value": mlir_ir.IntegerAttr.get(i64_type, int(value))},
        loc=loc,
    )
    return op.results[0]


def _const_i32(value: int, *, loc: mlir_ir.Location | None = None) -> mlir_ir.Value:
    i32_type = mlir_ir.IntegerType.get_signless(32)
    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[i32_type],
        attributes={"value": mlir_ir.IntegerAttr.get(i32_type, value)},
        loc=loc,
    )
    return op.results[0]


def _const_f32(value: float, *, loc: mlir_ir.Location | None = None) -> mlir_ir.Value:
    op = mlir_ir.Operation.create(
        "arith.constant",
        results=[mlir_ir.F32Type.get()],
        attributes={"value": mlir_ir.FloatAttr.get_f32(float(value))},
        loc=loc,
    )
    return op.results[0]


_FULL_SUPPORTED_DTYPES = frozenset(("i8", "i16", "i32", "i64", "bf16", "f16", "f32"))

_ARANGE_SUPPORTED_DTYPES = frozenset(("i8", "i16", "i32", "i64"))
_ARANGE_ORDERS = frozenset(("increase", "decrease"))

# Width of one Ascend vector register tile in bytes. Must stay in sync with
# TlaVectorRegionPass::kVectorBytes in csrc/mlir/lib/Passes/TlaVectorRegionPass.cpp.
_VECTOR_REGISTER_BYTES = 256


def _vector_lane_count(element_bytes: int) -> int:
    """Return lane count for one vector register tile at the given element width."""
    if element_bytes <= 0:
        raise TlaCoreAPIError(
            f"element size must be positive for vector lane count, got {element_bytes}"
        )
    return _VECTOR_REGISTER_BYTES // element_bytes


def _as_index_value(value: Any) -> mlir_ir.Value:
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, Numeric):
        # Index path: signed Int* only. Reject UInt* and Bool (Bool is 0/1 and
        # ``signed=False``) — use .to(Int32) (or another Int*) before indexing.
        if not (type(resolved).is_integer and type(resolved).signed):
            raise TlaLoweringError(
                f"Expected signed integer Numeric index, got {type(resolved).__name__}; "
                f"cast explicitly with .to(Int32) (or another Int*) before indexing"
            )
        resolved = resolved.ir_value()
    if isinstance(resolved, mlir_ir.Value):
        if isinstance(resolved.type, mlir_ir.IndexType):
            return resolved
        if (
            mlir_ir.IntegerType.isinstance(resolved.type)
            and mlir_ir.IntegerType(resolved.type).is_signless
        ):
            return mlir_ir.Operation.create(
                "arith.index_cast",
                operands=[resolved],
                results=[mlir_ir.IndexType.get()],
            ).results[0]
        raise TlaLoweringError(
            f"Expected index-like operand, got SSA type {resolved.type}"
        )
    if isinstance(resolved, bool):
        return _const_index(int(resolved))
    if isinstance(resolved, int):
        return _const_index(resolved)
    raise TlaLoweringError(f"Expected index-like operand, got {type(value).__name__}")


def _as_i64_value(value: Any, *, loc: mlir_ir.Location | None = None) -> mlir_ir.Value:
    resolved = _resolve_bound_value(value)
    i64_type = mlir_ir.IntegerType.get_signless(64)
    if isinstance(resolved, Numeric):
        if isinstance(resolved.value, (bool, int)):
            dtype = type(resolved).dtype.lower()
            if dtype.startswith("i") and dtype[1:].isdigit():
                return _const_i64(int(resolved.value), loc=loc)
            raise TlaLoweringError(
                f"Expected i64-like Numeric, got {type(resolved).__name__}"
            )
        resolved = resolved.ir_value(loc=loc)
    if isinstance(resolved, mlir_ir.Value):
        if isinstance(resolved.type, mlir_ir.IndexType):
            return mlir_ir.Operation.create(
                "arith.index_cast",
                operands=[resolved],
                results=[i64_type],
                loc=loc,
            ).results[0]
        if mlir_ir.IntegerType.isinstance(resolved.type):
            int_type = mlir_ir.IntegerType(resolved.type)
            if int_type.width == 64:
                return resolved
            cast_name = "arith.extsi" if int_type.width < 64 else "arith.trunci"
            return mlir_ir.Operation.create(
                cast_name,
                operands=[resolved],
                results=[i64_type],
                loc=loc,
            ).results[0]
    if isinstance(resolved, bool):
        return _const_i64(int(resolved), loc=loc)
    if isinstance(resolved, int):
        return _const_i64(resolved, loc=loc)
    raise TlaLoweringError(f"Expected i64-like operand, got {type(value).__name__}")


def _coerce_inttoptr_address(
    addr_token: str,
    value: int | mlir_ir.Value | Numeric,
    loc: mlir_ir.Location | None,
) -> mlir_ir.Value:
    """Integer SSA for ``tla.inttoptr`` (``gm`` / ``generic`` → i64, else i32)."""
    t = addr_token.strip().lower()
    target_ty = (
        mlir_ir.IntegerType.get_signless(64)
        if t in ("gm", "generic")
        else mlir_ir.IntegerType.get_signless(32)
    )
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, Numeric):
        if not type(resolved).is_integer:
            _op_error(
                "make_ptr",
                f"address must be int or integer Numeric/SSA, got {_type_name(value)}",
            )
        resolved = resolved.ir_value(loc=loc)
    if isinstance(resolved, mlir_ir.Value):
        if PtrType.isinstance(resolved.type):
            _op_error(
                "make_ptr",
                "address must be integer or index SSA, not !tla.ptr",
            )
        vt = resolved.type
        if isinstance(vt, mlir_ir.IndexType):
            return mlir_ir.Operation.create(
                "arith.index_cast",
                operands=[resolved],
                results=[target_ty],
                loc=loc,
            ).results[0]
        if mlir_ir.IntegerType.isinstance(vt):
            int_type = mlir_ir.IntegerType(vt)
            if int_type.width == target_ty.width:
                return resolved
            cast_name = (
                "arith.extsi" if int_type.width < target_ty.width else "arith.trunci"
            )
            return mlir_ir.Operation.create(
                cast_name,
                operands=[resolved],
                results=[target_ty],
                loc=loc,
            ).results[0]
        _op_error(
            "make_ptr",
            f"address SSA must be integer or index, got {vt}",
        )
    if isinstance(resolved, bool) or not isinstance(resolved, int):
        _op_error(
            "make_ptr",
            f"address must be int or integer SSA, got {_type_name(value)}",
        )
    return mlir_ir.Operation.create(
        "arith.constant",
        results=[target_ty],
        attributes={"value": mlir_ir.IntegerAttr.get(target_ty, int(resolved))},
        loc=loc,
    ).results[0]


def _as_value(value: Any) -> mlir_ir.Value:
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, _Pointer):
        resolved = _resolve_bound_value(resolved.value)
    if isinstance(resolved, _Tensor):
        resolved = _resolve_bound_value(resolved.value)
    if isinstance(resolved, VectorSSA):
        resolved = _resolve_bound_value(resolved.value)
    if isinstance(resolved, MaskSSA):
        resolved = _resolve_bound_value(resolved.value)
    if isinstance(resolved, Numeric):
        resolved = resolved.ir_value()
    if isinstance(resolved, _MutexValue):
        resolved = _resolve_bound_value(resolved.value)
    if isinstance(resolved, mlir_ir.Value):
        st = _runtime._current_frontend_state()
        if st is not None:
            host = st.tensor_host_by_value.get(resolved)
            if host is not None:
                st.tensor_host_by_value[resolved] = host
        return resolved
    if isinstance(resolved, bool):
        return _const_i32(int(resolved))
    if isinstance(resolved, int):
        return _const_i32(resolved)
    if isinstance(resolved, float):
        return _const_f32(resolved)
    raise TlaLoweringError(f"Expected SSA operand, got {type(value).__name__}")


def _as_branch_value(value: Any) -> mlir_ir.Value:
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        i1 = mlir_ir.IntegerType.get_signless(1)
        return mlir_ir.Operation.create(
            "arith.constant",
            results=[i1],
            attributes={"value": mlir_ir.IntegerAttr.get(i1, int(resolved))},
        ).results[0]
    return _as_value(value)


def _wrap_frontend_value(value: mlir_ir.Value) -> Any:
    if PtrType.isinstance(value.type):
        return _Pointer(value)
    if _tla_type_bridge.type_is_vector_ssa(value.type):
        return VectorSSA(value)
    if _tla_type_bridge.type_is_mask_ssa(value.type):
        return MaskSSA(value)
    if _tla_type_bridge.type_is_tensor(value.type):
        return _Tensor(value)
    if _tla_type_bridge.type_is_mutex(value.type):
        return _MutexValue(value, "", -1)
    if isinstance(value.type, mlir_ir.IndexType):
        # User model: Int32; Ascend IR may still use ``index``.
        return as_numeric(value)
    if mlir_ir.IntegerType.isinstance(value.type):
        int_type = mlir_ir.IntegerType(value.type)
        if int_type.width == 1:
            # i1 surfaces as Bool Numeric.
            return Bool(value)
        return Numeric.from_mlir_type(value.type)(value)
    try:
        # f16 / bf16 / f32 (and other scalar Numerics) wrap into Numerics.
        return Numeric.from_mlir_type(value.type)(value)
    except TypeError:
        return value


def unpack_to_irvalue(
    mixed_values: list[Any] | tuple[Any, ...],
    body_name: str,
    full_write_args_count: int = 0,
    mixed_value_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[mlir_ir.Value], tuple[list[Any], list[str]]]:
    """Flatten frontend values into MLIR values for dynamic SCF regions."""
    del full_write_args_count
    from .base_dsl.utils import tree_utils

    names = tuple(
        mixed_value_names
        if mixed_value_names is not None
        else (str(index) for index in _builtins.range(len(mixed_values)))
    )
    specs: list[Any] = []
    ir_values: list[mlir_ir.Value] = []
    leaf_names: list[str] = []
    for index, value in enumerate(mixed_values):
        name = names[index] if index < len(names) else str(index)
        leaves, spec, names_for_value = tree_utils.flatten_frontend_if_tree(value, name)
        ir_values.extend(leaves)
        specs.append(spec)
        leaf_names.extend(names_for_value)
    if not all(isinstance(value, mlir_ir.Value) for value in ir_values):
        raise TlaCoreAPIError(
            f"Dynamic {body_name} values must flatten to MLIR SSA values"
        )
    return ir_values, (specs, leaf_names)


def _collect_tla_tensor_type_metadata(
    ir_values: list[mlir_ir.Value] | tuple[mlir_ir.Value, ...],
) -> list[TlaTensorTypeDescriptor | None]:
    return [
        _tla_tensor_type_for_mlir_value(value)
        if _tla_type_bridge.type_is_tensor(value.type)
        else None
        for value in ir_values
    ]


def pack_from_irvalue(
    ir_values: list[mlir_ir.Value] | tuple[mlir_ir.Value, ...],
    pytree_def: tuple[list[Any], list[str]],
    mixed_values: list[Any] | tuple[Any, ...],
    full_write_args_count: int = 0,
    tensor_type_metadata: list[TlaTensorTypeDescriptor | None] | None = None,
) -> list[Any]:
    """Rebuild frontend values from MLIR values produced by dynamic SCF ops."""
    del full_write_args_count
    from .base_dsl.utils import tree_utils

    specs, _ = pytree_def
    if tensor_type_metadata is None:
        source_values, _ = unpack_to_irvalue(
            mixed_values, "SCF tensor metadata propagation"
        )
        tensor_type_metadata = _collect_tla_tensor_type_metadata(source_values)
    if len(tensor_type_metadata) != len(ir_values):
        raise TlaCoreAPIError(
            "Dynamic SCF result count does not match its carried value metadata"
        )
    for result, tensor_type in zip(ir_values, tensor_type_metadata, strict=True):
        if not _tla_type_bridge.type_is_tensor(result.type):
            continue
        if tensor_type is None:
            raise TlaCoreAPIError(
                "Dynamic SCF tensor result is missing structured type metadata"
            )
        # SCF already requires every carried edge to have the same MLIR tensor
        # type. Preserve that structured type model on block arguments and
        # results so they remain valid inputs to tile_view/make_tensor_like.
        # The _Tensor tree reconstruction below restores runtime descriptor
        # leaves from the accompanying SCF-carried index SSA values.
        _register_tla_tensor_type(result, tensor_type)
    wrapped = [_wrap_frontend_value(value) for value in ir_values]
    return list(tree_utils.rebuild_frontend_if_carried_values(wrapped, specs))


def _coerce_type(type_like: Any) -> mlir_ir.Type:
    if isinstance(type_like, mlir_ir.Type):
        return type_like
    to_mlir_type = getattr(type_like, "to_mlir_type", None)
    if callable(to_mlir_type):
        return to_mlir_type()
    raise TypeError(
        "expected mlir.ir.Type or object with to_mlir_type(); "
        f"got {type(type_like).__name__}"
    )


def _const_int_value(value: Any) -> int | None:
    # Treat plain Python ints as static dimensions first so they are not confused with
    # execution-lowering arg_bindings: e.g. type_args=(4,) binds id(4) to a block arg, and
    # resolving before this check could make literals in make_shape(4, 8, 16) look dynamic.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        return int(resolved)
    if isinstance(resolved, int):
        return resolved
    if isinstance(resolved, Numeric) and isinstance(resolved.value, (bool, int)):
        dtype = type(resolved).dtype.lower()
        if dtype.startswith("i"):
            return int(resolved.value)
    if isinstance(resolved, mlir_ir.Value):
        owner = getattr(resolved, "owner", None)
        if owner is not None and getattr(owner, "name", "") == "arith.constant":
            attrs = owner.attributes
            if "value" in attrs:
                attr = attrs["value"]
                if isinstance(attr, mlir_ir.IntegerAttr):
                    return int(attr.value)
    return None


def _vector_ssa_type_for_mlir_value(
    value: mlir_ir.Value,
) -> TlaVectorSSATypeDescriptor:
    if not _tla_type_bridge.type_is_vector_ssa(value.type):
        raise TlaLoweringError(f"expected !tla.vector<NxT> SSA value, got {value.type}")
    element_type = _tla_type_bridge.vector_ssa_element_type_get(value.type)
    valid_lanes = _tla_type_bridge.vector_ssa_valid_lanes_get(value.type)
    return TlaVectorSSATypeDescriptor(
        valid_lanes=valid_lanes,
        element_type=_dtype_to_str(element_type),
    )


def _cast_result_descriptor(
    src_desc: TlaVectorSSATypeDescriptor,
    dst_token: str,
    params: CastParams,
) -> TlaVectorSSATypeDescriptor:
    """Compute the result VectorSSA descriptor for a ``tla.cast``.

    ``valid_lanes`` follows the AVE cast lane mapping:

    - dynamic source (``None``) -> dynamic
    - same-width cast (``dst_bytes == src_bytes``) -> source lanes unchanged
    - narrowing (``dst_bytes < src_bytes``) -> dynamic (``None``): the result
      lane placement depends on the AVE even/odd or pack part and is not
      tracked statically
    - widening (``dst_bytes > src_bytes``) -> ``ceil((src - slot) / ratio)``
      where ``ratio = dst_bytes // src_bytes`` and ``slot = params.codes()[0]``
      (the ``reg_slot`` code); ``src <= slot`` (no source lane at this slot)
      raises :class:`TlaCoreAPIError` -- the result would contain invalid data
    """
    src_lanes = src_desc.valid_lanes
    if src_lanes is None:
        return TlaVectorSSATypeDescriptor(None, dst_token)
    src_bytes = dtype_size_bytes(src_desc.element_type)
    dst_bytes = dtype_size_bytes(dst_token)
    if dst_bytes == src_bytes:
        return TlaVectorSSATypeDescriptor(src_lanes, dst_token)
    if dst_bytes < src_bytes:
        return TlaVectorSSATypeDescriptor(None, dst_token)
    ratio = dst_bytes // src_bytes
    slot = params.codes()[0]
    if src_lanes <= slot:
        _op_error(
            "cast",
            f"widening {src_desc.element_type}->{dst_token} source has only "
            f"{src_lanes} valid lane(s) but reg_slot {params.reg_slot} (slot "
            f"{slot}) requires a source lane at index {slot}; the result "
            f"would contain invalid data",
        )
    lanes = (src_lanes - slot + ratio - 1) // ratio
    return TlaVectorSSATypeDescriptor(lanes, dst_token)


def _mask_ssa_type_for_mlir_value(
    value: mlir_ir.Value,
) -> TlaMaskSSATypeDescriptor:
    if not _tla_type_bridge.type_is_mask_ssa(value.type):
        raise TlaLoweringError(f"expected !tla.mask<N> SSA value, got {value.type}")
    return TlaMaskSSATypeDescriptor(
        physical_lanes=_tla_type_bridge.mask_ssa_physical_lanes_get(value.type)
    )


def _mask_ssa_type_for_element_type(
    element_type: str,
) -> TlaMaskSSATypeDescriptor:
    token = str(element_type).strip().lower()
    element_bytes = dtype_size_bytes(token)
    if (
        token == "i1"
        or element_bytes <= 0
        or _VECTOR_REGISTER_BYTES % element_bytes != 0
    ):
        _op_error(
            "mask",
            f"unsupported predicate element type {token!r}; expected a "
            "byte-aligned type that fits a 256-byte vector register",
        )
    return TlaMaskSSATypeDescriptor(physical_lanes=_vector_lane_count(element_bytes))


def _require_mask_matches_element_type(
    op_name: str, mask_value: mlir_ir.Value, element_type: str
) -> None:
    """The lane check, against an element type rather than a vector value.

    tla.full has to validate its mask against the *result* it is about to
    build, which does not exist yet, so the check cannot start from a Value.
    """
    mask_desc = _mask_ssa_type_for_mlir_value(mask_value)
    expected_physical_lanes = _mask_ssa_type_for_element_type(
        element_type
    ).physical_lanes
    if mask_desc.physical_lanes != expected_physical_lanes:
        _op_error(
            op_name,
            f"mask has {mask_desc.physical_lanes} predicate lanes, expected "
            f"{expected_physical_lanes} for {element_type} VectorSSA",
        )


def _require_mask_matches_vector(
    op_name: str, mask_value: mlir_ir.Value, vector_value: mlir_ir.Value
) -> None:
    vector_desc = _vector_ssa_type_for_mlir_value(vector_value)
    _require_mask_matches_element_type(op_name, mask_value, vector_desc.element_type)


def _vector_ssa_type_from_tensor_descriptor(
    tensor_type: TlaTensorTypeDescriptor,
) -> TlaVectorSSATypeDescriptor:
    valid_lanes = 1
    for dim in _flatten_tla_tuple(tensor_type.origin_shape):
        if dim is None:
            valid_lanes = None
            break
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TlaLoweringError(
                "VectorSSA load requires static integer or dynamic origin_shape leaves"
            )
        valid_lanes *= dim
    return TlaVectorSSATypeDescriptor(valid_lanes, tensor_type.element_type)


def _register_tla_tensor_type(
    value: mlir_ir.Value,
    tensor_type: TlaTensorTypeDescriptor,
) -> None:
    st = _runtime._current_frontend_state()
    if st is None:
        return
    st.tensor_type_by_value[value] = tensor_type


def _tla_tensor_type_for_mlir_value(v: mlir_ir.Value) -> TlaTensorTypeDescriptor:
    """Resolve structured ``!tla.tensor`` metadata for an SSA value."""
    st = _runtime._current_frontend_state()
    if st is not None:
        cached = st.tensor_type_by_value.get(v)
        if cached is not None:
            return cached
        host = st.tensor_host_by_value.get(v)
        if host is not None:
            desc = host.tla_tensor_type_descriptor()
            st.tensor_type_by_value[v] = desc
            return desc
    raise TlaLoweringError(
        "missing structured Tla tensor metadata for SSA value; tensor values used by "
        "Python lowering must come from a host tla.Tensor argument or a Tla Python op "
        f"that registered TlaTensorTypeDescriptor, got {str(v.type)!r}"
    )


def _tla_tensor_descriptor_from_type_or_value(
    source: mlir_ir.Value | TlaTensorTypeDescriptor,
) -> TlaTensorTypeDescriptor:
    if isinstance(source, mlir_ir.Value):
        return _tla_tensor_type_for_mlir_value(source)
    if isinstance(source, TlaTensorTypeDescriptor):
        return source
    raise TlaLoweringError(
        "expected a vector SSA value with registered TlaTensorTypeDescriptor "
        f"or a TlaTensorTypeDescriptor, got {type(source).__name__}"
    )


def _register_tla_tensor_metadata(
    value: mlir_ir.Value, metadata: dict[str, Any]
) -> None:
    st = _runtime._current_frontend_state()
    if st is None:
        return
    st.tensor_metadata_by_value[value] = metadata


def _tensor_metadata_field(value: mlir_ir.Value, field: str) -> Any:
    st = _runtime._current_frontend_state()
    if st is not None:
        cached = st.tensor_metadata_by_value.get(value)
        if cached is not None and field in cached:
            materialized = _require_resolved_metadata_leaves(
                value, cached[field], field
            )
            if materialized is not cached[field]:
                cached = dict(cached)
                cached[field] = materialized
                st.tensor_metadata_by_value[value] = cached
            return materialized
    metadata = _tla_tensor_type_for_mlir_value(value).metadata()
    if field not in metadata:
        raise TlaLoweringError(f"unknown tensor metadata field: {field}")
    materialized = _require_resolved_metadata_leaves(value, metadata[field], field)
    if st is not None:
        stored = dict(metadata)
        stored[field] = materialized
        st.tensor_metadata_by_value[value] = stored
    return materialized


def is_dynamic_gm_tensor_arg(host: Any) -> bool:
    """True when a host ``Tensor`` enters the kernel as a dynamic GM ``memref`` arg.

    Matches the Approach A split: GM tensor + any dynamic shape/stride/origin leaf
    uses the bridged memref entry ABI; static GM stays ``!tla.tensor``.
    """
    from .tla.runtime import _Tensor as HostTensor

    if not isinstance(host, HostTensor):
        return False
    addr = str(getattr(host, "addrspace", None) or "gm").strip().lower()
    if addr != "gm":
        return False
    desc = host.tla_tensor_type_descriptor()
    return (
        _tree_contains_dynamic(desc.shape)
        or _tree_contains_dynamic(desc.stride)
        or _tree_contains_dynamic(desc.origin_shape)
    )


def _materialize_dynamic_gm_root_tensor_descriptor(
    memref_arg: mlir_ir.Value,
    origin0_arg: mlir_ir.Value,
    origin1_arg: mlir_ir.Value,
    tensor_ty: Any,
    *,
    loc: mlir_ir.Location | None = None,
) -> tuple[mlir_ir.Value, dict[str, Any]]:
    """Emit ``memref.dim`` / strided metadata + ``tla.tensor_desc`` for a dynamic GM tensor.

    ``memref_arg`` is the schema-v4 unified GM memref. ``origin0_arg`` / ``origin1_arg``
    are the companion index ABI args (no shape→origin derivation).
    """
    from catlass._mlir.dialects import arith, memref

    from .base_dsl.typing import as_numeric

    if not isinstance(tensor_ty, TlaTensorTypeDescriptor):
        raise TlaLoweringError("dynamic GM prologue expects a TlaTensorTypeDescriptor")

    shape_leaves = list(_flatten_tla_tuple(tensor_ty.shape))
    stride_leaves = list(_flatten_tla_tuple(tensor_ty.stride))
    origin_leaves = list(_flatten_tla_tuple(tensor_ty.origin_shape))
    coord_leaves = list(_flatten_tla_tuple(tensor_ty.coord))
    rank = len(shape_leaves)
    layout_tag = str(tensor_ty.layout_tag)
    is_nz_family = layout_tag in _NZ_FAMILY_LAYOUT_TOKENS
    valid_linear_metadata = (
        layout_tag in _LINEAR_LAYOUT_TOKENS
        and rank in (1, 2)
        and len(stride_leaves) == rank
        and len(origin_leaves) == rank
        and len(coord_leaves) == rank
    )
    valid_nz_family_metadata = (
        is_nz_family
        and rank == 4
        and len(stride_leaves) == 4
        and len(origin_leaves) == 2
        and len(coord_leaves) == 2
    )
    if not valid_linear_metadata and not valid_nz_family_metadata:
        raise TlaLoweringError(
            "dynamic GM prologue supports rank-1/rank-2 linear metadata or "
            "four-leaf NZFamily shape/stride with two-leaf origin/coord; "
            f"got layout={layout_tag}, ranks shape={rank}, "
            f"stride={len(stride_leaves)}, origin={len(origin_leaves)}, "
            f"coord={len(coord_leaves)}"
        )
    # Same root-coord rule as TlaLowerFuncPass::validateKernelTensorArg: prologue
    # and launch ABI hard-code coord/offset to 0.
    if any(leaf is None for leaf in coord_leaves) or not all(
        int(leaf) == 0 for leaf in coord_leaves
    ):
        raise TlaLoweringError(
            "dynamic GM root must be a root tensor with zero coordinates; "
            f"got {tensor_ty.coord!r}"
        )

    # Unified ABI memref exposes 4 size/stride slots; logical leaves map to the leading axes.
    abi_rank = 4
    shape_vals: list[mlir_ir.Value] = []
    for axis, extent in enumerate(shape_leaves):
        if extent is None:
            axis_v = _const_index(axis, loc=loc)
            shape_vals.append(memref.DimOp(memref_arg, axis_v, loc=loc).result)
        else:
            shape_vals.append(_const_index(int(extent), loc=loc))

    abi_strides: list[mlir_ir.Value] | None = None
    is_row_major = layout_tag in _ROW_MAJOR_LAYOUT_TOKENS
    is_column_major = layout_tag in _COLUMN_MAJOR_LAYOUT_TOKENS
    if (is_row_major or is_column_major or is_nz_family) and any(
        leaf is None for leaf in stride_leaves
    ):
        meta = memref.ExtractStridedMetadataOp(memref_arg, loc=loc)
        results = list(meta.results)
        stride_start = 2 + abi_rank
        if len(results) < stride_start + abi_rank:
            raise TlaLoweringError(
                "extract_strided_metadata returned unexpected result count "
                f"{len(results)} for ABI rank {abi_rank}"
            )
        abi_strides = list(results[stride_start : stride_start + abi_rank])

    stride_vals: list[mlir_ir.Value] = []
    # Both orientations take their dynamic stride leaves from the memref itself.
    # Deriving the column-major pitch as "shape[0]" instead was only right while
    # the buffer's first dimension happened to be the logical major axis; a
    # tensor handed over with its logical shape and column-major strides -- the
    # natural way to express one -- breaks that assumption silently.
    if is_row_major or is_column_major or is_nz_family:
        for axis, leaf in enumerate(stride_leaves):
            if leaf is None:
                if abi_strides is None:
                    raise TlaLoweringError(
                        f"{layout_tag} dynamic stride requires extract_strided_metadata"
                    )
                stride_vals.append(abi_strides[axis])
            else:
                stride_vals.append(_const_index(int(leaf), loc=loc))
    elif rank == 1:
        stride_vals.append(_const_index(1, loc=loc))
    else:
        # Never fall back to a column-major stride pair for an unclassified tag:
        # it lowers silently and only shows up as wrong numbers on device.
        raise TlaLoweringError(
            f"dynamic GM prologue has no stride rule for layout {layout_tag!r}; "
            "classify it in _ROW_MAJOR_LAYOUT_TOKENS or _COLUMN_MAJOR_LAYOUT_TOKENS"
        )

    def origin_abi_value(axis: int) -> mlir_ir.Value:
        """Use companion ABI origin args when the type leaf is dynamic."""
        leaf = origin_leaves[axis]
        if leaf is None:
            return origin0_arg if axis == 0 else origin1_arg
        return _const_index(int(leaf), loc=loc)

    zero = _const_index(0, loc=loc)
    one = _const_index(1, loc=loc)
    if rank == 1:
        shape0 = _const_index(1, loc=loc)
        shape1 = shape_vals[0]
        shape2 = one
        shape3 = one
        stride1 = stride_vals[0]
        stride0 = arith.MulIOp(shape1, stride1, loc=loc).result
        stride2 = one
        stride3 = one
        # Internal desc keeps origin0=1; user-facing origin is ABI originShape0.
        origin0 = _const_index(1, loc=loc)
        origin1 = origin_abi_value(0)
        meta_shape = shape_vals[0]
        meta_stride = stride_vals[0]
        meta_origin = origin1
        metadata = {
            "shape": _replace_flat_leaves_in_tree(
                tensor_ty.shape, (as_numeric(meta_shape),)
            ),
            "stride": _replace_flat_leaves_in_tree(
                tensor_ty.stride, (as_numeric(meta_stride),)
            ),
            "coord": tensor_ty.coord,
            "origin_shape": _replace_flat_leaves_in_tree(
                tensor_ty.origin_shape, (as_numeric(meta_origin),)
            ),
            "dtype": tensor_ty.element_type,
            "addrspace": tensor_ty.addrspace,
            "layout_tag": layout_tag,
        }
    elif is_nz_family:
        shape0, shape1, shape2, shape3 = shape_vals
        stride0, stride1, stride2, stride3 = stride_vals
        origin0 = origin_abi_value(0)
        origin1 = origin_abi_value(1)
        metadata = {
            "shape": _replace_flat_leaves_in_tree(
                tensor_ty.shape, tuple(as_numeric(value) for value in shape_vals)
            ),
            "stride": _replace_flat_leaves_in_tree(
                tensor_ty.stride,
                tuple(as_numeric(value) for value in stride_vals),
            ),
            "coord": tensor_ty.coord,
            "origin_shape": _replace_flat_leaves_in_tree(
                tensor_ty.origin_shape,
                (as_numeric(origin0), as_numeric(origin1)),
            ),
            "dtype": tensor_ty.element_type,
            "addrspace": tensor_ty.addrspace,
            "layout_tag": layout_tag,
        }
    else:
        shape0 = shape_vals[0]
        shape1 = shape_vals[1]
        shape2 = one
        shape3 = one
        stride0 = stride_vals[0]
        stride1 = stride_vals[1]
        stride2 = one
        stride3 = one
        origin0 = origin_abi_value(0)
        origin1 = origin_abi_value(1)
        metadata = {
            "shape": _replace_flat_leaves_in_tree(
                tensor_ty.shape, (as_numeric(shape0), as_numeric(shape1))
            ),
            "stride": _replace_flat_leaves_in_tree(
                tensor_ty.stride, (as_numeric(stride0), as_numeric(stride1))
            ),
            "coord": tensor_ty.coord,
            "origin_shape": _replace_flat_leaves_in_tree(
                tensor_ty.origin_shape, (as_numeric(origin0), as_numeric(origin1))
            ),
            "dtype": tensor_ty.element_type,
            "addrspace": tensor_ty.addrspace,
            "layout_tag": layout_tag,
        }

    result_ty = tensor_ty.to_mlir_type(memref_arg.type.context)
    desc = _tla_ops_gen.tensor_desc(
        result_ty,
        memref_arg,
        shape0,
        shape1,
        shape2,
        shape3,
        stride0,
        stride1,
        stride2,
        stride3,
        origin0,
        origin1,
        zero,
        zero,
        loc=loc,
    )
    _register_tla_tensor_type(desc, tensor_ty)
    _register_tla_tensor_metadata(desc, metadata)
    return desc, metadata


def _require_resolved_metadata_leaves(
    tensor: mlir_ir.Value, tree: Any, field: str
) -> Any:
    """Assert metadata leaves are resolved; dynamic ``None`` is a lowering bug."""

    def walk(node: Any) -> Any:
        if isinstance(node, tuple):
            return tuple(walk(child) for child in node)
        if node is not None:
            return node
        raise TlaLoweringError(
            f"dynamic tensor metadata field {field!r} was not resolved at "
            "kernel entry; dynamic GM arguments must materialize a root "
            f"descriptor first (value={tensor})"
        )

    return walk(tree)


# Layout constants aligned with ``catlass/catlass.hpp`` and ``tla/layout.hpp``.
_CATLASS_BYTE_PER_C0 = 32
_CATLASS_C0_NUM_PER_FRACTAL = 16
_MX_SCALE_GM_LAYOUT_TOKENS = frozenset(
    {"RowMajorMxScaleA", "ColMajorMxScaleA", "RowMajorMxScaleB", "ColMajorMxScaleB"}
)
_LINEAR_LAYOUT_TOKENS = frozenset({"RowMajor", "ColumnMajor"})
# Orientation of a linear layout, named once. Everything downstream -- stride
# derivation, the dynamic GM prologue, leading-dim inference -- must ask these
# rather than compare against "RowMajor"/"ColumnMajor" literally: a tag that is
# row-major but not spelled RowMajor otherwise falls into a column-major
# fallback that no validator can see, because the tag is a perfectly valid
# linear tag.
_ROW_MAJOR_LAYOUT_TOKENS = frozenset({"RowMajor"})
_COLUMN_MAJOR_LAYOUT_TOKENS = _LINEAR_LAYOUT_TOKENS - _ROW_MAJOR_LAYOUT_TOKENS
# Which flat stride leaf of each GM MxScale tag is the runtime pitch (everything
# else is a structural constant: the 0 broadcast, unit strides, and the e8m0
# scale C0 = 2). mark_layout_dynamic keeps the constants static and marks only
# this leaf dynamic.
_MX_SCALE_GM_DYNAMIC_STRIDE_LEAF = {
    "RowMajorMxScaleA": 1,
    "ColMajorMxScaleA": 3,
    "RowMajorMxScaleB": 1,
    "ColMajorMxScaleB": 3,
}

_NZ_FAMILY_LAYOUT_TOKENS = frozenset(
    {
        "zN",
        "nZ",
        "L0Clayout",
        "zNUnAlign",
        "zZMxScale",
        "nNMxScale",
    }
    | _MX_SCALE_GM_LAYOUT_TOKENS
)
_MAKE_TENSOR_LIKE_ON_CHIP_ADDRSPACES = frozenset({"l1", "l0a", "l0b", "l0c", "ub"})
_BITS_PER_BYTE = 8


def _c0_geometry(element_bits: int) -> tuple[int, int]:
    """(elements per C0, elements per fractal) for an element width in bits.

    The hardware C0 is 32 *bytes* whatever the element type, so the count is
    ``32 * 8 / element_bits``. Doing it in bits is what makes sub-byte elements
    work: in bytes the fp4 divisor would be 0.5 and floor division would silently
    return the count for an 8-bit element (32 instead of 64). This mirrors the
    bc layer, which has always computed it this way via
    ``BytesToBits(BYTE_PER_C0) / SizeOfBits<T>``.
    """
    c0 = _builtins.max(1, (_CATLASS_BYTE_PER_C0 * _BITS_PER_BYTE) // element_bits)
    return c0, c0 * _CATLASS_C0_NUM_PER_FRACTAL


@dataclass(frozen=True, slots=True)
class _RemappedTensorMetadata:
    """Layout-specific trees and their canonical flat metadata views."""

    # Compiler-facing shape tree; NZ-family layouts use a nested 2x2 tuple.
    shape: tuple[Any, ...]
    # Compiler-facing stride tree matching the structure of ``shape``.
    stride: tuple[Any, ...]
    # Flat logical coordinate tuple used by compiler tensor metadata.
    coord: tuple[Any, ...]
    # Flat logical origin shape shared by compiler metadata and launch packing.
    origin_shape: tuple[Any, ...]
    # Canonical preorder leaves of ``shape`` for launch packing.
    flat_shape: tuple[Any, ...]
    # Canonical preorder leaves of ``stride`` for launch packing.
    flat_stride: tuple[Any, ...]


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("ceil_div divisor must be positive")
    return (a + b - 1) // b


def _round_up(a: int, m: int) -> int:
    return _ceil_div(a, m) * m


def _linear_stride_alignment_elements(
    element_bits: int, alignment_bytes: int | None
) -> int:
    """How many elements a linear (RowMajor/ColumnMajor) stride must round up to.

    In bits, so a sub-byte element gives its true count (a 32-byte alignment is
    512 fp4 elements, not an error). Only the linear layouts use this; the
    NZ-family ones derive their strides from the fractal geometry instead, but
    it is computed before the layout is known, so it must tolerate any width.
    """
    if alignment_bytes is None:
        return 1
    alignment_bits = alignment_bytes * _BITS_PER_BYTE
    if alignment_bytes <= 0 or element_bits <= 0 or alignment_bits % element_bits != 0:
        raise ValueError(
            "linear stride byte alignment must be a positive multiple of element size"
        )
    return alignment_bits // element_bits


def _mul_int_optional(a: int | None, b: int) -> int | None:
    if a is None:
        return None
    return a * b


def _as_index_expr_or_int(value: Any) -> Any:
    const = _const_int_value(value)
    if const is not None:
        return int(const)
    resolved = _resolve_bound_value(value)
    if (
        isinstance(resolved, Numeric)
        and type(resolved).is_integer
        and type(resolved).signed
    ):
        return resolved
    if isinstance(resolved, mlir_ir.Value):
        return as_numeric(resolved)
    return value


def _components_to_index_tree(components: Any) -> Any:
    if isinstance(components, tuple):
        return tuple(_components_to_index_tree(x) for x in components)
    return _as_index_expr_or_int(components)


def _components_to_type_tree(components: Any) -> Any:
    if isinstance(components, tuple):
        return tuple(_components_to_type_tree(x) for x in components)
    const = _const_int_value(components)
    return int(const) if const is not None else None


def _is_flat_pair(tree: Any) -> bool:
    return (
        isinstance(tree, tuple)
        and len(tree) == 2
        and all(not isinstance(component, tuple) for component in tree)
    )


def _is_nz_family_2x2_tree(tree: Any) -> bool:
    return (
        isinstance(tree, tuple)
        and len(tree) == 2
        and all(
            isinstance(group, tuple)
            and len(group) == 2
            and all(not isinstance(leaf, tuple) for leaf in group)
            for group in tree
        )
    )


def _infer_padded_origin_tree_from_nz_family_shape(
    shape_tree: Any, *, for_op: str
) -> tuple[Any, Any]:
    if not _is_nz_family_2x2_tree(shape_tree):
        raise TlaLoweringError(
            f"tla.{for_op} NZFamily layout expects shape as two 2-leaf groups "
            f"((m0, m1), (n0, n1)); got {shape_tree!r}"
        )
    return (
        shape_tree[0][0] * shape_tree[0][1],
        shape_tree[1][0] * shape_tree[1][1],
    )


def _validate_static_make_tensor_layout(
    shape_tree: Any,
    stride_tree: Any,
    *,
    dtype: str,
    layout_tag: str,
) -> None:
    """Validate static shape/stride leaves using ``tla/layout.hpp`` traits."""

    shape_const_tree = _components_to_type_tree(shape_tree)
    stride_const_tree = _components_to_type_tree(stride_tree)

    if layout_tag in _LINEAR_LAYOUT_TOKENS:
        if len(shape_const_tree) == 1:
            checks = ((stride_const_tree[0], 1),)
        elif layout_tag in _ROW_MAJOR_LAYOUT_TOKENS:
            checks = ((stride_const_tree[1], 1),)
        else:
            checks = ((stride_const_tree[0], 1),)
    else:
        ele_num_per_c0, ele_num_per_fractal = _c0_geometry(dtype_size_bits(dtype))
        shape00, shape01 = shape_const_tree[0]
        shape10, _ = shape_const_tree[1]
        stride00, stride01 = stride_const_tree[0]
        stride10, stride11 = stride_const_tree[1]

        if layout_tag == "zN":
            checks = (
                (shape00, _CATLASS_C0_NUM_PER_FRACTAL),
                (shape10, ele_num_per_c0),
                (stride10, 1),
                (stride01, ele_num_per_fractal),
            )
        elif layout_tag == "nZ":
            checks = (
                (shape00, ele_num_per_c0),
                (shape10, _CATLASS_C0_NUM_PER_FRACTAL),
                (stride00, 1),
                (stride11, ele_num_per_fractal),
            )
        elif layout_tag == "nNMxScale":
            checks = (
                (shape00, _MX_SCALE_ELE_NUM_PER_C0),
                (shape10, _CATLASS_C0_NUM_PER_FRACTAL),
                (stride00, 1),
                (stride01, _MX_SCALE_ELE_NUM_PER_FRACTAL),
            )
        elif layout_tag == "RowMajorMxScaleA":
            checks = (
                (shape00, 1),
                (shape10, _MX_SCALE_ELE_NUM_PER_C0),
                (stride00, 0),
                (stride10, 1),
                (stride11, _MX_SCALE_ELE_NUM_PER_C0),
            )
        elif layout_tag == "ColMajorMxScaleA":
            checks = (
                (shape00, 1),
                (shape10, _MX_SCALE_ELE_NUM_PER_C0),
                (stride00, 0),
                (stride01, _MX_SCALE_ELE_NUM_PER_C0),
                (stride10, 1),
            )
        elif layout_tag == "RowMajorMxScaleB":
            checks = (
                (shape00, _MX_SCALE_ELE_NUM_PER_C0),
                (shape10, 1),
                (stride00, 1),
                (stride10, 0),
                (stride11, _MX_SCALE_ELE_NUM_PER_C0),
            )
        elif layout_tag == "ColMajorMxScaleB":
            checks = (
                (shape00, _MX_SCALE_ELE_NUM_PER_C0),
                (shape10, 1),
                (stride00, 1),
                (stride01, _MX_SCALE_ELE_NUM_PER_C0),
                (stride10, 0),
            )
        elif layout_tag == "zZMxScale":
            checks = (
                (shape00, _CATLASS_C0_NUM_PER_FRACTAL),
                (shape10, _MX_SCALE_ELE_NUM_PER_C0),
                (stride10, 1),
                (stride11, _MX_SCALE_ELE_NUM_PER_FRACTAL),
            )
        elif layout_tag == "L0Clayout":
            checks = (
                (shape00, _CATLASS_C0_NUM_PER_FRACTAL),
                (shape10, _CATLASS_C0_NUM_PER_FRACTAL),
                (stride10, 1),
                (stride01, 256),
            )
        else:  # zNUnAlign
            checks = (
                (shape01, 1),
                (shape10, ele_num_per_c0),
                (stride00, ele_num_per_c0),
                (stride10, 1),
            )

    if any(actual is not None and actual != expected for actual, expected in checks):
        raise TlaLoweringError(
            f"tla.make_tensor shape {shape_const_tree!r} and stride "
            f"{stride_const_tree!r} do not match layout {layout_tag!r}"
        )


def _tree_add(a: Any, b: Any) -> Any:
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            raise TlaLoweringError(
                "tensor metadata tree rank mismatch in coord addition"
            )
        return tuple(_tree_add(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, str) or isinstance(b, str):
        return "?"
    return a + b


def _tree_crop_origin(parent_origin: Any, tile_shape: Any, tile_coord: Any) -> Any:
    if isinstance(parent_origin, tuple):
        if not (isinstance(tile_shape, tuple) and isinstance(tile_coord, tuple)):
            raise TlaLoweringError("tensor metadata tree rank mismatch in origin crop")
        if len(parent_origin) != len(tile_shape) or len(parent_origin) != len(
            tile_coord
        ):
            raise TlaLoweringError("tensor metadata tree rank mismatch in origin crop")
        return tuple(
            _tree_crop_origin(po, ts, tc)
            for po, ts, tc in zip(parent_origin, tile_shape, tile_coord, strict=True)
        )
    if (
        isinstance(parent_origin, str)
        or isinstance(tile_shape, str)
        or isinstance(tile_coord, str)
    ):
        return "?"
    if (
        isinstance(parent_origin, int)
        and isinstance(tile_shape, int)
        and isinstance(tile_coord, int)
    ):
        rest = parent_origin - tile_coord
        return rest if rest < tile_shape else tile_shape
    rest = parent_origin - tile_coord
    rest_v = _runtime._coerce_index_value(rest)
    tile_v = _runtime._coerce_index_value(tile_shape)
    op = mlir_ir.Operation.create(
        "arith.minsi",
        operands=[rest_v, tile_v],
        results=[mlir_ir.IndexType.get()],
    )
    return as_numeric(op.results[0])


def _ceil_div_expr(a: Any, b: int) -> Any:
    if isinstance(a, str):
        return "?"
    return (a + (b - 1)) // b


def _round_up_expr(a: Any, m: int) -> Any:
    if isinstance(a, str):
        return "?"
    return _ceil_div_expr(a, m) * m


def _materialize_layout_trees_from_origin(
    origin_shape: Any, dtype: str, layout: str
) -> tuple[Any, Any, Any, Any] | None:
    if not isinstance(origin_shape, tuple) or len(origin_shape) != 2:
        return None
    rows, cols = origin_shape
    if dtype_size_bits(dtype) <= 0:
        return None
    ele_num_per_c0, ele_num_per_fractal = _c0_geometry(dtype_size_bits(dtype))
    c0_num_per_fractal = _CATLASS_C0_NUM_PER_FRACTAL
    coord = (0, 0)
    linear_alignment_elements = _linear_stride_alignment_elements(
        dtype_size_bits(dtype), _CATLASS_BYTE_PER_C0
    )
    if layout == "RowMajor":
        return (
            (rows, cols),
            (_round_up_expr(cols, linear_alignment_elements), 1),
            coord,
            origin_shape,
        )
    if layout == "ColumnMajor":
        return (
            (rows, cols),
            (1, _round_up_expr(rows, linear_alignment_elements)),
            coord,
            origin_shape,
        )
    if layout in _MX_SCALE_GM_LAYOUT_TOKENS:
        # The GM MxScale tags are NZFamily: a 2x2 nested tree following
        # tla::MakeMxScaleLayout, except the rank-1 extent (the plain matrix
        # dimension) is spelled as a (1, dim) pair whose leading leaf carries
        # stride 0 -- addressing is unchanged, but the tree now has the four
        # leaves every NZFamily consumer expects. The scale C0 is 2, fixed by
        # the e8m0 format, and no 32-byte C0 rounding applies. The pitch has to
        # land on the leaf the bc layer reads (stride<0,1> for A, stride<1,1>
        # for B), or the copy strides by one element.
        rows_ru = _round_up_expr(rows, _MX_SCALE_ELE_NUM_PER_C0)
        cols_ru = _round_up_expr(cols, _MX_SCALE_ELE_NUM_PER_C0)
        if layout == "RowMajorMxScaleA":
            return (
                (
                    (1, rows),
                    (
                        _MX_SCALE_ELE_NUM_PER_C0,
                        _ceil_div_expr(cols, _MX_SCALE_ELE_NUM_PER_C0),
                    ),
                ),
                ((0, cols_ru), (1, _MX_SCALE_ELE_NUM_PER_C0)),
                coord,
                origin_shape,
            )
        if layout == "ColMajorMxScaleA":
            return (
                (
                    (1, rows),
                    (
                        _MX_SCALE_ELE_NUM_PER_C0,
                        _ceil_div_expr(cols, _MX_SCALE_ELE_NUM_PER_C0),
                    ),
                ),
                ((0, _MX_SCALE_ELE_NUM_PER_C0), (1, rows * _MX_SCALE_ELE_NUM_PER_C0)),
                coord,
                origin_shape,
            )
        if layout == "RowMajorMxScaleB":
            return (
                (
                    (
                        _MX_SCALE_ELE_NUM_PER_C0,
                        _ceil_div_expr(rows, _MX_SCALE_ELE_NUM_PER_C0),
                    ),
                    (1, cols),
                ),
                (
                    (1, cols * _MX_SCALE_ELE_NUM_PER_C0),
                    (0, _MX_SCALE_ELE_NUM_PER_C0),
                ),
                coord,
                origin_shape,
            )
        # ColMajorMxScaleB
        return (
            (
                (
                    _MX_SCALE_ELE_NUM_PER_C0,
                    _ceil_div_expr(rows, _MX_SCALE_ELE_NUM_PER_C0),
                ),
                (1, cols),
            ),
            (
                (1, _MX_SCALE_ELE_NUM_PER_C0),
                (0, rows_ru),
            ),
            coord,
            origin_shape,
        )
    if layout == "zN":
        rows_ru = _round_up_expr(rows, c0_num_per_fractal)
        return (
            (
                (c0_num_per_fractal, _ceil_div_expr(rows, c0_num_per_fractal)),
                (ele_num_per_c0, _ceil_div_expr(cols, ele_num_per_c0)),
            ),
            ((ele_num_per_c0, ele_num_per_fractal), (1, rows_ru * ele_num_per_c0)),
            coord,
            origin_shape,
        )
    if layout == "nZ":
        cols_ru = _round_up_expr(cols, c0_num_per_fractal)
        return (
            (
                (ele_num_per_c0, _ceil_div_expr(rows, ele_num_per_c0)),
                (c0_num_per_fractal, _ceil_div_expr(cols, c0_num_per_fractal)),
            ),
            ((1, cols_ru * ele_num_per_c0), (ele_num_per_c0, ele_num_per_fractal)),
            coord,
            origin_shape,
        )
    if layout == "nNMxScale":
        rows_ru = _round_up_expr(rows, _MX_SCALE_ELE_NUM_PER_C0)
        return (
            (
                (
                    _MX_SCALE_ELE_NUM_PER_C0,
                    _ceil_div_expr(rows, _MX_SCALE_ELE_NUM_PER_C0),
                ),
                (c0_num_per_fractal, _ceil_div_expr(cols, c0_num_per_fractal)),
            ),
            (
                (1, _MX_SCALE_ELE_NUM_PER_FRACTAL),
                (_MX_SCALE_ELE_NUM_PER_C0, rows_ru * c0_num_per_fractal),
            ),
            coord,
            origin_shape,
        )
    if layout == "zZMxScale":
        ele_num_per_c0 = _MX_SCALE_ELE_NUM_PER_C0
        ele_num_per_fractal = _MX_SCALE_ELE_NUM_PER_FRACTAL
        cols_ru = _round_up_expr(cols, ele_num_per_c0)
        return (
            (
                (c0_num_per_fractal, _ceil_div_expr(rows, c0_num_per_fractal)),
                (ele_num_per_c0, _ceil_div_expr(cols, ele_num_per_c0)),
            ),
            ((ele_num_per_c0, cols_ru * c0_num_per_fractal), (1, ele_num_per_fractal)),
            coord,
            origin_shape,
        )
    if layout == "L0Clayout":
        rows_ru = _round_up_expr(rows, c0_num_per_fractal)
        return (
            (
                (c0_num_per_fractal, _ceil_div_expr(rows, c0_num_per_fractal)),
                (c0_num_per_fractal, _ceil_div_expr(cols, c0_num_per_fractal)),
            ),
            ((c0_num_per_fractal, 256), (1, rows_ru * c0_num_per_fractal)),
            coord,
            origin_shape,
        )
    if layout == "zNUnAlign":
        ceil_div_cols = _ceil_div_expr(cols, ele_num_per_c0)
        stride_scale = rows * ele_num_per_c0
        return (
            ((rows, 1), (ele_num_per_c0, ceil_div_cols)),
            ((ele_num_per_c0, stride_scale), (1, stride_scale)),
            coord,
            origin_shape,
        )
    return None


def _remap_tensor_like_prefix_fields_for_layout_trees(
    origin_shape: Any,
    dtype: str,
    layout: str,
    *,
    linear_stride_alignment_bytes: int | None = None,
) -> _RemappedTensorMetadata | None:
    """Derive shape/stride/coord/origin as nested tuple trees for ``layout`` (TLA-style when fractal).

    ``origin_shape`` must be a flat ``(N,)`` or ``(M, N)`` Tla index tree with ``int`` or
    ``None`` leaves. ``None`` represents an unknown dimension, spelled ``?`` in MLIR.
    Rank-1 ``RowMajor`` uses ``(N):(1)``, ``coord=(0,)``. Rank-2 **coord** is always
    ``(0, 0)``. Naming follows ``tla::GetTileLayout`` / fractal ``MakeLayout``
    (``rows`` / ``cols`` / ``ELE_NUM_PER_C0`` / ``C0_NUM_PER_FRACTAL``).
    """
    if isinstance(origin_shape, tuple) and len(origin_shape) == 1:
        length = origin_shape[0]
        if isinstance(length, tuple):
            return None
        layout_tag = layout.strip()
        if layout_tag == "RowMajor":
            shape = (length,)
            stride = (1,)
            origin = (length,)
            return _RemappedTensorMetadata(shape, stride, (0,), origin, shape, stride)
        return None

    origin_pair = _flat_dim_pair_from_tree(origin_shape)
    if origin_pair == (None, None) and origin_shape != (None, None):
        return None
    rows, cols = origin_pair
    layout_tag = layout.strip()
    if dtype_size_bits(dtype) <= 0:
        return None
    ele_num_per_c0, ele_num_per_fractal = _c0_geometry(dtype_size_bits(dtype))
    c0_num_per_fractal = _CATLASS_C0_NUM_PER_FRACTAL
    origin_shape_tree: tuple[Any, ...] = (rows, cols)
    coord_tree: tuple[Any, ...] = (0, 0)

    linear_alignment_elements = _linear_stride_alignment_elements(
        dtype_size_bits(dtype), linear_stride_alignment_bytes
    )

    if layout_tag == "RowMajor":
        leading_stride = (
            None if cols is None else _round_up(cols, linear_alignment_elements)
        )
        shape = (rows, cols)
        stride = (leading_stride, 1)
        return _RemappedTensorMetadata(
            shape,
            stride,
            coord_tree,
            origin_shape_tree,
            shape,
            stride,
        )
    if layout_tag == "ColumnMajor":
        leading_stride = (
            None if rows is None else _round_up(rows, linear_alignment_elements)
        )
        shape = (rows, cols)
        stride = (1, leading_stride)
        return _RemappedTensorMetadata(
            shape,
            stride,
            coord_tree,
            origin_shape_tree,
            shape,
            stride,
        )
    if layout_tag in _MX_SCALE_GM_LAYOUT_TOKENS:
        # NZFamily 2x2 trees per tla::MakeMxScaleLayout, with the rank-1 extent
        # spelled as a (1, dim) pair whose leading leaf carries stride 0 -- the
        # scale C0 is 2, fixed by the e8m0 format. The pitch lands on the leaf
        # the bc layer reads (stride<0,1> for A, stride<1,1> for B). Static
        # leaves (the 0/1/2 structural constants) stay int; runtime-varying
        # extents/strides become None, as in the other NZFamily branches.
        c0 = _MX_SCALE_ELE_NUM_PER_C0
        rows_ru = None if rows is None else _round_up(rows, c0)
        cols_ru = None if cols is None else _round_up(cols, c0)
        ceil_rows = None if rows is None else _ceil_div(rows, c0)
        ceil_cols = None if cols is None else _ceil_div(cols, c0)
        if layout_tag == "RowMajorMxScaleA":
            flat_shape = (1, rows, c0, ceil_cols)
            flat_stride = (0, cols_ru, 1, c0)
        elif layout_tag == "ColMajorMxScaleA":
            flat_shape = (1, rows, c0, ceil_cols)
            flat_stride = (0, c0, 1, _mul_int_optional(rows, c0))
        elif layout_tag == "RowMajorMxScaleB":
            flat_shape = (c0, ceil_rows, 1, cols)
            flat_stride = (1, _mul_int_optional(cols, c0), 0, c0)
        else:  # ColMajorMxScaleB
            flat_shape = (c0, ceil_rows, 1, cols)
            flat_stride = (1, c0, 0, rows_ru)
        shape = ((flat_shape[0], flat_shape[1]), (flat_shape[2], flat_shape[3]))
        stride = ((flat_stride[0], flat_stride[1]), (flat_stride[2], flat_stride[3]))
        return _RemappedTensorMetadata(
            shape,
            stride,
            coord_tree,
            origin_shape_tree,
            flat_shape,
            flat_stride,
        )
    if layout_tag == "zN":
        rows_round_up = None if rows is None else _round_up(rows, c0_num_per_fractal)
        ceil_div_rows = None if rows is None else _ceil_div(rows, c0_num_per_fractal)
        ceil_div_cols = None if cols is None else _ceil_div(cols, ele_num_per_c0)
        stride_scale = _mul_int_optional(rows_round_up, ele_num_per_c0)
        flat_shape = (c0_num_per_fractal, ceil_div_rows, ele_num_per_c0, ceil_div_cols)
        flat_stride = (ele_num_per_c0, ele_num_per_fractal, 1, stride_scale)
        shape = ((flat_shape[0], flat_shape[1]), (flat_shape[2], flat_shape[3]))
        stride = ((flat_stride[0], flat_stride[1]), (flat_stride[2], flat_stride[3]))
        return _RemappedTensorMetadata(
            shape,
            stride,
            coord_tree,
            origin_shape_tree,
            flat_shape,
            flat_stride,
        )
    if layout_tag == "nZ":
        cols_round_up = None if cols is None else _round_up(cols, c0_num_per_fractal)
        ceil_div_rows = None if rows is None else _ceil_div(rows, ele_num_per_c0)
        ceil_div_cols = None if cols is None else _ceil_div(cols, c0_num_per_fractal)
        stride_scale = _mul_int_optional(cols_round_up, ele_num_per_c0)
        flat_shape = (ele_num_per_c0, ceil_div_rows, c0_num_per_fractal, ceil_div_cols)
        flat_stride = (1, stride_scale, ele_num_per_c0, ele_num_per_fractal)
        shape = ((flat_shape[0], flat_shape[1]), (flat_shape[2], flat_shape[3]))
        stride = ((flat_stride[0], flat_stride[1]), (flat_stride[2], flat_stride[3]))
        return _RemappedTensorMetadata(
            shape,
            stride,
            coord_tree,
            origin_shape_tree,
            flat_shape,
            flat_stride,
        )
    if layout_tag == "nNMxScale":
        rows_round_up = (
            None if rows is None else _round_up(rows, _MX_SCALE_ELE_NUM_PER_C0)
        )
        ceil_div_rows = (
            None if rows is None else _ceil_div(rows, _MX_SCALE_ELE_NUM_PER_C0)
        )
        ceil_div_cols = None if cols is None else _ceil_div(cols, c0_num_per_fractal)
        stride_scale = _mul_int_optional(rows_round_up, c0_num_per_fractal)
        flat_shape = (
            _MX_SCALE_ELE_NUM_PER_C0,
            ceil_div_rows,
            c0_num_per_fractal,
            ceil_div_cols,
        )
        flat_stride = (
            1,
            _MX_SCALE_ELE_NUM_PER_FRACTAL,
            _MX_SCALE_ELE_NUM_PER_C0,
            stride_scale,
        )
        shape = ((flat_shape[0], flat_shape[1]), (flat_shape[2], flat_shape[3]))
        stride = ((flat_stride[0], flat_stride[1]), (flat_stride[2], flat_stride[3]))
        return _RemappedTensorMetadata(
            shape,
            stride,
            coord_tree,
            origin_shape_tree,
            flat_shape,
            flat_stride,
        )
    if layout_tag == "zZMxScale":
        ele_num_per_c0 = _MX_SCALE_ELE_NUM_PER_C0
        ele_num_per_fractal = _MX_SCALE_ELE_NUM_PER_FRACTAL
        cols_round_up = None if cols is None else _round_up(cols, ele_num_per_c0)
        ceil_div_rows = None if rows is None else _ceil_div(rows, c0_num_per_fractal)
        ceil_div_cols = None if cols is None else _ceil_div(cols, ele_num_per_c0)
        stride_scale = _mul_int_optional(cols_round_up, c0_num_per_fractal)
        flat_shape = (c0_num_per_fractal, ceil_div_rows, ele_num_per_c0, ceil_div_cols)
        flat_stride = (ele_num_per_c0, stride_scale, 1, ele_num_per_fractal)
        shape = ((flat_shape[0], flat_shape[1]), (flat_shape[2], flat_shape[3]))
        stride = ((flat_stride[0], flat_stride[1]), (flat_stride[2], flat_stride[3]))
        return _RemappedTensorMetadata(
            shape,
            stride,
            coord_tree,
            origin_shape_tree,
            flat_shape,
            flat_stride,
        )
    if layout_tag == "L0Clayout":
        # Keep L0C consistent with tla::MakeLayout<..., L0C>, which uses
        # a fixed fractal element count (256) regardless of dtype.
        l0c_ele_num_per_fractal = 256
        rows_round_up = None if rows is None else _round_up(rows, c0_num_per_fractal)
        ceil_div_rows = None if rows is None else _ceil_div(rows, c0_num_per_fractal)
        ceil_div_cols = None if cols is None else _ceil_div(cols, c0_num_per_fractal)
        stride_scale = _mul_int_optional(rows_round_up, c0_num_per_fractal)
        flat_shape = (
            c0_num_per_fractal,
            ceil_div_rows,
            c0_num_per_fractal,
            ceil_div_cols,
        )
        flat_stride = (c0_num_per_fractal, l0c_ele_num_per_fractal, 1, stride_scale)
        shape = ((flat_shape[0], flat_shape[1]), (flat_shape[2], flat_shape[3]))
        stride = ((flat_stride[0], flat_stride[1]), (flat_stride[2], flat_stride[3]))
        return _RemappedTensorMetadata(
            shape,
            stride,
            coord_tree,
            origin_shape_tree,
            flat_shape,
            flat_stride,
        )
    if layout_tag == "zNUnAlign":
        # zNUnAlign is zN without M-axis fractal blocking: leaf[0] = rows (runtime,
        # not the compile-time C0_NUM_PER_FRACTAL), leaf[1] = 1. stride[1] = stride[3]
        # = rows * ele_num_per_c0 (runtime, not the compile-time ele_num_per_fractal).
        # N axis keeps the ele_num_per_c0 sub-blocking of zN.
        ceil_div_cols = None if cols is None else _ceil_div(cols, ele_num_per_c0)
        stride_scale = _mul_int_optional(rows, ele_num_per_c0)
        flat_shape = (rows, 1, ele_num_per_c0, ceil_div_cols)
        flat_stride = (ele_num_per_c0, stride_scale, 1, stride_scale)
        shape = ((flat_shape[0], flat_shape[1]), (flat_shape[2], flat_shape[3]))
        stride = ((flat_stride[0], flat_stride[1]), (flat_stride[2], flat_stride[3]))
        return _RemappedTensorMetadata(
            shape,
            stride,
            coord_tree,
            origin_shape_tree,
            flat_shape,
            flat_stride,
        )
    return None


def _flat_dim_pair_from_tree(tree: Any) -> tuple[int | None, int | None]:
    if not isinstance(tree, tuple) or len(tree) != 2:
        return (None, None)
    out: list[int | None] = []
    for item in tree:
        if item is None:
            out.append(None)
        elif isinstance(item, int):
            out.append(item)
        else:
            return (None, None)
    return (out[0], out[1])


def _remap_tensor_like_trees_for_layout(
    origin_shape: Any,
    dtype: str,
    layout: str,
) -> _RemappedTensorMetadata | None:
    return _remap_tensor_like_prefix_fields_for_layout_trees(
        origin_shape, dtype, layout
    )


def _flat_pair_sum_type_tree(a: Any, b: Any) -> tuple[int | None, int | None] | None:
    a_pair = _flat_dim_pair_from_tree(a)
    b_pair = _flat_dim_pair_from_tree(b)
    if a_pair == (None, None) and a != (None, None):
        return None
    if b_pair == (None, None) and b != (None, None):
        return None
    out: list[int | None] = []
    for lhs, rhs in zip(a_pair, b_pair, strict=True):
        out.append(None if lhs is None or rhs is None else lhs + rhs)
    return (out[0], out[1])


def _tree_contains_dynamic(tree: Any) -> bool:
    if isinstance(tree, tuple):
        return any(_tree_contains_dynamic(x) for x in tree)
    return tree is None


def _crop_origin_shape_type_tree(
    parent_origin: Any, tile_shape: Any, tile_coord: Any
) -> Any | None:
    origin_pair = _flat_dim_pair_from_tree(parent_origin)
    shape_pair = _flat_dim_pair_from_tree(tile_shape)
    coord_pair = _flat_dim_pair_from_tree(tile_coord)
    if (
        (origin_pair == (None, None) and parent_origin != (None, None))
        or (shape_pair == (None, None) and tile_shape != (None, None))
        or (coord_pair == (None, None) and tile_coord != (None, None))
    ):
        return None
    out: list[int | None] = []
    for dim, (origin, shape, coord) in enumerate(
        zip(origin_pair, shape_pair, coord_pair, strict=True), start=1
    ):
        if origin is None or shape is None or coord is None:
            out.append(None)
        else:
            if coord < 0:
                raise TlaLoweringError(
                    f"tile_view: element offset along dimension {dim} is negative ({coord}); "
                    "parent origin_shape requires non-negative offsets"
                )
            if coord >= origin:
                raise TlaLoweringError(
                    f"tile_view: element offset along dimension {dim} ({coord}) is out of range "
                    "for parent origin_shape; each offset must be strictly less than the "
                    "corresponding logical extent"
                )
            out.append(shape if origin - coord >= shape else origin - coord)
    return (out[0], out[1])


def _metadata_from_type_tree(tree: Any, dynamic_values: Iterable[Any]) -> Any:
    dyn_iter = iter(dynamic_values)

    def walk(node: Any) -> Any:
        if isinstance(node, tuple):
            return tuple(walk(x) for x in node)
        if node is None:
            return next(dyn_iter)
        return node

    return walk(tree)


def _format_tensor_type_descriptor(
    source: mlir_ir.Value | TlaTensorTypeDescriptor,
    shape_tree: Any,
    coord_tree: Any,
) -> TlaTensorTypeDescriptor:
    """Build the ``tile_view`` result tensor descriptor."""
    parent = _tla_tensor_descriptor_from_type_or_value(source)
    tile_shape = _components_to_type_tree(shape_tree)
    tile_coord = _components_to_type_tree(coord_tree)
    layout_remap = _remap_tensor_like_trees_for_layout(
        tile_shape, parent.element_type, parent.layout_tag
    )
    shape = layout_remap.shape if layout_remap is not None else tile_shape
    # ``tile_view`` follows the parent storage stride (TLA ``GetTileLayout``): only the
    # logical view shape/coord/origin update here; fractal stride is not re-derived from the tile.
    stride = parent.stride
    coord = _flat_pair_sum_type_tree(parent.coord, tile_coord)
    origin_shape = _crop_origin_shape_type_tree(
        parent.origin_shape, tile_shape, tile_coord
    )
    if origin_shape is None:
        if coord is not None and _tree_contains_dynamic(coord):
            origin_shape = (None, None)
        else:
            origin_shape = shape
    if coord is None:
        coord = tile_coord
    return TlaTensorTypeDescriptor(
        layout=TlaLayoutDescriptor(
            shape=TlaIndexTreeType("shape", shape),
            stride=TlaIndexTreeType("stride", stride),
            origin_shape=TlaIndexTreeType("shape", origin_shape),
            layout_tag=parent.layout_tag,
        ),
        coord=coord,
        element_type=parent.element_type,
        addrspace=parent.addrspace,
        ptr_alignment=parent.ptr_alignment,
    )


def _format_tensor_type(
    source: mlir_ir.Value | TlaTensorTypeDescriptor,
    shape_value: mlir_ir.Value,
    coord_value: mlir_ir.Value,
) -> str:
    """Build ``tile_view`` result ``!tla.tensor<…>`` from source tensor + shape/coord SSA.

    **Stride**, **dtype**, **addr**, **layout** follow the parent tensor (including NZFamily
    layouts: stride is never replaced by a ``tile_view``-local fractal remap).

    **Shape** (memory-layout field): for flat ``M,N`` tile sizes, when
    :func:`_remap_tensor_like_prefix_fields_for_layout_trees` applies to the parent's layout tag,
    the nested fractal spellings are taken from that remap—using the tile's flat logical
    ``M,N`` as the logical pair (aligned with ``tla::GetTileLayout`` for non-MxScale paths).
    Otherwise the ``!tla.shape<…>`` spelling is used unchanged.

    **Coord** and **origin** follow ``tla::TileViewImpl`` / ``GetTileLayout`` / ``CropOriginShape``:
    element offset is the Hadamard product already encoded in ``coord_value``'s ``!tla.coord<…>``;
    ``coordNew = Add(parent.coord, offset)`` and ``origin`` is ``CropOriginShape(parent.origin,
    tileShape, offset)`` for flat trees: :func:`_crop_origin_shape_type_tree` applies the same ``min``
    semantics per dimension, emitting ``?`` where a leaf is dynamic. When **coord**
    carries ``?`` but origin cannot be flattened (nested fractal), **origin** uses ``?,?`` instead
    of mirroring the memory **shape** tree.

    Out-of-range tile starts (``offset_i >= parent_origin_i`` or negative offset) are rejected
    inside :func:`_crop_origin_shape_type_tree` with :class:`TlaLoweringError` where the offset is
    statically known.

    ``source`` must be an SSA value with registered structured tensor metadata or an explicit
    :class:`~catlass.types.TlaTensorTypeDescriptor`.
    """
    return _format_tensor_type_descriptor(source, shape_value, coord_value).to_asm()


_format_tile_type = _format_tensor_type


def _is_integer(value: Any) -> bool:
    """Return whether ``value`` is a static ``int``, index/integer SSA, or Numeric Int*."""
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        return False
    if isinstance(resolved, int):
        return True
    if (
        isinstance(resolved, Numeric)
        and type(resolved).is_integer
        and type(resolved).signed
    ):
        return True
    if isinstance(resolved, mlir_ir.Value) and isinstance(
        resolved.type, (mlir_ir.IndexType, mlir_ir.IntegerType)
    ):
        return True
    if _category(resolved) == "index":
        return True
    return False


def _flatten_tla_tuple(a: IndexTree) -> tuple[Any, ...]:
    """Flatten a nested tuple tree to leaves (depth-first leaf order)."""
    if not isinstance(a, tuple):
        return (a,)
    return tuple(chain.from_iterable(_flatten_tla_tuple(x) for x in a))


def _check_index_tree_group_depth(
    op_name: str, tree: Any, *, _tuple_depth: int = 0
) -> None:
    if not isinstance(tree, tuple):
        return
    if len(tree) == 0:
        _op_error(op_name, "expected non-empty tuple in index tree")
    if _tuple_depth >= 2:
        _op_error(
            op_name,
            "Tla index trees support only top-level leaves or one-level leaf groups",
        )
    for child in tree:
        _check_index_tree_group_depth(op_name, child, _tuple_depth=_tuple_depth + 1)


def _check_shape(shape: IndexTree) -> None:
    """Validate a shape tree (positive static sizes, nested tuple of shapes, or dynamic index)."""
    _check_index_tree_group_depth("make_shape", shape)
    if _is_integer(shape):
        resolved = _resolve_bound_value(shape)
        if isinstance(resolved, int) and resolved <= 0:
            _op_error(
                "make_shape",
                f"Expected size in shape to be strictly positive, but got {resolved}",
            )
        return
    if isinstance(shape, tuple):
        if len(shape) == 0:
            _op_error("make_shape", "expected non-empty tuple in shape tree")
        for s in shape:
            _check_shape(s)
        return
    _op_error(
        "make_shape",
        f"Expected Shape, which is a positive integer or tuple of Shapes, but got {_type_name(shape)}",
    )


def _check_coord(coord: IndexTree) -> None:
    """Validate a Coord tree: leaves are index-like; static leaves must be >= 0.

    Tla does not support ``None`` coord leaves in packing yet; leaves must satisfy :func:`_is_integer`.
    Compile-time-known negative leaves are rejected; dynamic SSA leaves are not checked here.
    """
    _check_index_tree_group_depth("make_coord", coord)
    if isinstance(coord, tuple):
        if len(coord) == 0:
            _op_error("make_coord", "expected non-empty tuple in coord tree")
        flat = _flatten_tla_tuple(coord)
        if len(flat) == 0:
            _op_error(
                "make_coord",
                f"Expected Coord with at least one leaf, but got {coord!r}",
            )
        if not all(_is_integer(c) for c in flat):
            _op_error(
                "make_coord",
                f"Expected Coord, whose leaves are integers, but got {coord!r}",
            )
    elif _is_integer(coord):
        flat = (coord,)
    else:
        _op_error(
            "make_coord",
            f"Expected Coord, which is an integer or tuple of Coords, but got {_type_name(coord)}",
        )
        return

    for c in flat:
        static = _const_int_value(c)
        if static is not None and static < 0:
            _op_error(
                "make_coord",
                f"Expected coord leaf >= 0, but got {static} (in {coord!r})",
            )


def _check_stride(stride: IndexTree) -> None:
    """Validate a Stride tree: leaves are index-like; static leaves must be > 0.

    Tla strides are index trees only (no scaled-basis leaves in this frontend).
    Compile-time-known non-positive leaves are rejected; dynamic SSA leaves are not checked here.
    """
    _check_index_tree_group_depth("make_stride", stride)
    if isinstance(stride, tuple):
        if len(stride) == 0:
            _op_error("make_stride", "expected non-empty tuple in stride tree")
        flat = _flatten_tla_tuple(stride)
        if len(flat) == 0:
            _op_error(
                "make_stride",
                f"Expected Stride with at least one leaf, but got {stride!r}",
            )
        if not all(_is_integer(s) for s in flat):
            _op_error(
                "make_stride",
                f"Expected Stride, whose leaves are integers, but got {stride!r}",
            )
    elif _is_integer(stride):
        flat = (stride,)
    else:
        _op_error(
            "make_stride",
            f"Expected Stride, which is an integer or tuple of Strides, but got {_type_name(stride)}",
        )
        return

    for s in flat:
        static = _const_int_value(s)
        if static is not None and static <= 0:
            _op_error(
                "make_stride",
                f"Expected stride leaf strictly positive, but got {static} (in {stride!r})",
            )


def _transform_leaf(f: Callable[..., Any], *args: Any) -> Any:
    if all(isinstance(t, tuple) for t in args):
        return tuple(_transform_leaf(f, *_a) for _a in zip(*args))
    if all(not isinstance(t, tuple) for t in args):
        return f(*args)
    raise TypeError(f"profile of input tuples doesn't match: {args}")


def _is_static(x: Any) -> bool:
    if isinstance(x, mlir_ir.Value):
        owner = getattr(x, "owner", None)
        if (
            owner is not None
            and str(getattr(owner, "name", "") or "") == "arith.constant"
        ):
            return True
        return False
    if isinstance(x, tuple):
        return all(_is_static(a) for a in x)
    if isinstance(x, (bool, int)):
        return True
    if isinstance(x, Numeric) and isinstance(getattr(x, "value", None), (bool, int)):
        return _const_int_value(x) is not None
    return False


def _pack_x(
    x: tuple[Any, ...],
    packer: Callable[[tuple[Any, ...]], tuple[str, list[mlir_ir.Value]]],
    op_name: str,
    *,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    x = _transform_leaf(_resolve_bound_value, x)
    res_ty, dyn_elems = packer(x)
    dyn_elems = [t for t in dyn_elems if not _is_static(t)]
    return mlir_ir.Operation.create(
        f"tla.{op_name}",
        results=[_coerce_type(res_ty)],
        operands=dyn_elems,
        loc=loc,
    ).results[0]


def _pack_tree(
    op_name: str,
    kind: str,
    components: tuple[Any, ...],
) -> tuple[TlaIndexTreeType, list[mlir_ir.Value]]:
    type_tree: list[Any] = []

    def pack_one(c: Any) -> tuple[str, list[mlir_ir.Value], Any]:
        if isinstance(c, list):
            _op_error(
                op_name,
                "expected nested tuple tree for make_* components, got list (use parentheses, not brackets)",
            )
        if isinstance(c, tuple):
            if len(c) == 0:
                _op_error(op_name, "expected non-empty nested tuple in tree")
            child_packs = [pack_one(x) for x in c]
            dyns: list[mlir_ir.Value] = []
            for _, d, _ in child_packs:
                dyns.extend(d)
            return (
                f"({','.join(ty for ty, _, _ in child_packs)})",
                dyns,
                tuple(tree for _, _, tree in child_packs),
            )
        _require_index(op_name, "leaf", c, 0)
        const = _const_int_value(c)
        if const is not None:
            return (str(const), [], int(const))
        return ("?", [_as_index_value(c)], None)

    parts: list[str] = []
    dyn: list[mlir_ir.Value] = []
    for c in components:
        frag, d, tree = pack_one(c)
        parts.append(frag)
        dyn.extend(d)
        type_tree.append(tree)
    return TlaIndexTreeType(kind, tuple(type_tree)), dyn


def _pack_shape(
    components: tuple[Any, ...], *, loc: mlir_ir.Location | None = None
) -> mlir_ir.Value:
    _check_shape(tuple(components))
    return _pack_x(
        tuple(components),
        lambda t: _pack_tree("make_shape", "shape", t),
        "make_shape",
        loc=loc,
    )


def _pack_coord(
    components: tuple[Any, ...], *, loc: mlir_ir.Location | None = None
) -> mlir_ir.Value:
    _check_coord(tuple(components))
    return _pack_x(
        tuple(components),
        lambda t: _pack_tree("make_coord", "coord", t),
        "make_coord",
        loc=loc,
    )


def _pack_stride(
    components: tuple[Any, ...], *, loc: mlir_ir.Location | None = None
) -> mlir_ir.Value:
    _check_stride(tuple(components))
    return _pack_x(
        tuple(components),
        lambda t: _pack_tree("make_stride", "stride", t),
        "make_stride",
        loc=loc,
    )


def _dtype_to_str(value: Any) -> str:
    if isinstance(value, mlir_ir.Type):
        try:
            return Numeric.from_mlir_type(value).dtype
        except TypeError:
            return str(value)
    if isinstance(value, type) and issubclass(value, Numeric):
        if not value.dtype:
            raise TypeError(
                f"expected concrete Numeric element type, got abstract {value!r}"
            )
        return value.dtype
    return str(value)


def _op_error(op_name: str, message: str) -> None:
    raise TlaCoreAPIError(f"tla.{op_name}: {message}")


def _type_name(value: Any) -> str:
    resolved = _resolve_bound_value(value)
    if resolved is not value:
        return _type_name(resolved)
    return type(value).__name__


def _category(value: Any) -> str | None:
    if isinstance(value, _Shape):
        return "shape"
    if isinstance(value, _Coord):
        return "coord"
    if isinstance(value, _Stride):
        return "stride"
    if isinstance(value, _Layout):
        return "layout"
    if isinstance(value, _Pointer):
        return "pointer"
    if isinstance(value, Tensor):
        return "tensor"
    category = getattr(value, "__tla_category__", None)
    if isinstance(category, str):
        return category
    category = _runtime._resolve_frontend_bound_category(value)
    if isinstance(category, str):
        return category
    resolved = _resolve_bound_value(value)
    if resolved is not value:
        category = getattr(resolved, "__tla_category__", None)
        if isinstance(category, str):
            return category
        category = _runtime._resolve_frontend_bound_category(resolved)
        if isinstance(category, str):
            return category
        value = resolved
    if isinstance(value, _MutexValue):
        return "mutex"
    if isinstance(value, mlir_ir.Value):
        if isinstance(value.type, mlir_ir.IndexType):
            return "index"
        category = _tla_type_bridge.tla_type_category(value.type)
        if category is not None:
            return category
    value_type = getattr(value, "type", None)
    if value_type is not None:
        category = _tla_type_bridge.tla_type_category(value_type)
        if category is not None:
            return category
    return None


def _token(value: Any) -> str | None:
    if isinstance(value, str):
        token = value.strip().lower()
    else:
        name = getattr(value, "name", None)
        if not isinstance(name, str):
            return None
        token = name.strip().lower()
    return token


def _name_token(value: Any) -> str | None:
    if isinstance(value, str):
        token = value.strip()
        return token or None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        token = name.strip()
        return token or None
    return None


def _require_index(op_name: str, name: str, value: Any, position: int) -> None:
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): expected index, got bool",
        )
    if isinstance(resolved, int):
        return
    if _category(resolved) == "index":
        return
    if (
        isinstance(resolved, Numeric)
        and type(resolved).is_integer
        and type(resolved).signed
    ):
        # User Int32 (etc.): lowering index_casts at the use site.
        return
    _op_error(
        op_name,
        f"invalid argument '{name}' (position {position}): expected index, got {_type_name(value)}",
    )


def _require_numeric(
    op_name: str,
    name: str,
    value: Any,
    position: int,
    *,
    integer: bool = False,
    signed: bool | None = None,
) -> None:
    resolved = _resolve_bound_value(value)
    if not isinstance(resolved, Numeric):
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): "
            f"expected Numeric, got {_type_name(value)}",
        )
    cls = type(resolved)
    if integer and not cls.is_integer:
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): "
            f"expected integer Numeric, got {cls.__name__}",
        )
    if signed is True and not cls.signed:
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): "
            f"expected signed integer Numeric, got {cls.__name__}",
        )
    if signed is False and cls.signed:
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): "
            f"expected unsigned integer Numeric, got {cls.__name__}",
        )


def _require_index_or_numeric(
    op_name: str, name: str, value: Any, position: int
) -> None:
    """Accept Python ``int``, index SSA, or signed integer ``Numeric``."""
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): "
            f"expected index or signed integer Numeric, got bool",
        )
    if isinstance(resolved, int):
        return
    if _category(resolved) == "index":
        return
    if isinstance(resolved, Numeric):
        _require_numeric(op_name, name, value, position, integer=True, signed=True)
        return
    _op_error(
        op_name,
        f"invalid argument '{name}' (position {position}): "
        f"expected index or signed integer Numeric, got {_type_name(value)}",
    )


def _require_category(
    op_name: str, name: str, value: Any, expected: str, position: int
) -> None:
    if _category(value) != expected:
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): expected {expected}, got {_type_name(value)}",
        )


def _require_categories(
    op_name: str,
    name: str,
    value: Any,
    expected: tuple[str, ...],
    position: int,
) -> None:
    if _category(value) not in expected:
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): "
            f"expected one of {expected}, got {_type_name(value)}",
        )


def _require_pipe(op_name: str, name: str, value: Any, position: int) -> None:
    token = _token(value)
    if token is None or token not in _PIPE_VALUES:
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): expected pipe, got {_type_name(value)}",
        )


def _pipe_attr_from_token(
    pipe: PipeLike | str, *, loc: mlir_ir.Location | None = None
) -> mlir_ir.Attribute:
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    pipe_value = str(_token(pipe)).lower()
    return mlir_ir.Attribute.parse(f"#tla.pipe<{pipe_value}>", context=ctx)


def _ensure_no_explicit_mutex_access_in_guard() -> None:
    state = _runtime._current_frontend_state()
    if state is None or state.mutex_guard_depth <= 0:
        return
    raise TlaCoreAPIError(
        "tla.mutex_guard body cannot contain explicit mutex lock/unlock calls"
    )


def _raw_operation(op_or_view: Any) -> mlir_ir.Operation:
    operation = getattr(op_or_view, "operation", op_or_view)
    if not isinstance(operation, mlir_ir.Operation):
        raise TlaLoweringError(
            "expected MLIR operation while scanning tla.mutex_guard body, got "
            f"{type(op_or_view).__name__}"
        )
    return operation


def _walk_mutex_guard_ops(ops: Sequence[Any]) -> list[mlir_ir.Operation]:
    walked: list[mlir_ir.Operation] = []

    def visit(op_or_view: Any) -> None:
        op = _raw_operation(op_or_view)
        walked.append(op)
        if op.name == "tla.vec.func":
            return
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child)

    for op in ops:
        visit(op)
    return walked


def _infer_copy_mutex_pipe(copy_op: mlir_ir.Operation) -> str:
    operands = list(copy_op.operands)
    if len(operands) != 2 and len(operands) != 3:
        raise TlaLoweringError("malformed tla.copy op in tla.mutex_guard body")
    src_addrspace = _tla_tensor_type_for_mlir_value(operands[1]).addrspace.lower()
    if src_addrspace == "l0c":
        if len(operands) != 3:
            raise TlaLoweringError("malformed tla.copy op in tla.mutex_guard body")
    elif len(operands) != 2:
        raise TlaLoweringError("malformed tla.copy op in tla.mutex_guard body")
    if src_addrspace == "gm":
        return "mte2"
    if src_addrspace == "l1":
        return "mte1"
    if src_addrspace == "ub":
        return "mte3"
    if src_addrspace == "l0c":
        return "fix"
    raise TlaLoweringError(
        "tla.mutex_guard cannot infer pipe for tla.copy with source addrspace "
        f"{src_addrspace!r}"
    )


def _infer_mutex_guard_pipe(body_ops: Sequence[Any]) -> str:
    inferred: list[str] = []
    for op in _walk_mutex_guard_ops(body_ops):
        name = op.name
        if name in {"tla.mutex_lock", "tla.mutex_unlock"}:
            raise TlaCoreAPIError(
                "tla.mutex_guard body cannot contain explicit mutex lock/unlock calls"
            )
        if name == "tla.copy":
            inferred.append(_infer_copy_mutex_pipe(op))
        elif name == "tla.mmad":
            inferred.append("cube")
        elif name == "tla.vec.func":
            inferred.append("vector")
    if not inferred:
        raise TlaLoweringError(
            "tla.mutex_guard body must emit at least one tla.copy, tla.mmad, "
            "or tla.vec.func"
        )
    unique = set(inferred)
    if len(unique) != 1:
        pipes = ", ".join(sorted(unique))
        raise TlaLoweringError(f"tla.mutex_guard body inferred multiple pipes: {pipes}")
    return inferred[0]


def _require_pointer_addrspace(op_name: str, value: Any, position: int) -> str:
    """Require :class:`AddressSpace` for pointer construction and allocation.

    Returns the MLIR addrspace keyword (``str(enum)`` == ``enum.name``). Callers that only validate may ignore it.
    """
    if not isinstance(value, AddressSpace):
        _op_error(
            op_name,
            f"invalid argument 'mem_space' (position {position}): expected AddressSpace, got {_type_name(value)}",
        )
    return str(value)


def _require_bool(op_name: str, name: str, value: Any, position: int) -> None:
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        return
    if isinstance(resolved, Numeric) and type(resolved) is Bool:
        return
    _op_error(
        op_name,
        f"invalid argument '{name}' (position {position}): expected bool|Bool, got {_type_name(value)}",
    )


_DEBUG_PRINT_I32_MIN = -(2**31)
_DEBUG_PRINT_I32_MAX = 2**31 - 1
_PRINT_TENSOR_SUPPORTED_DTYPES = (
    "f16",
    "f32",
    "i8",
    "i16",
    "i32",
    "u8",
    "u16",
    "u32",
)
_PRINT_TENSOR_SUPPORTED_DTYPES_TEXT = ", ".join(_PRINT_TENSOR_SUPPORTED_DTYPES)
_DEBUG_PRINT_SUPPORTED_DTYPES = _PRINT_TENSOR_SUPPORTED_DTYPES
_DEBUG_PRINT_SUPPORTED_DTYPES_TEXT = _PRINT_TENSOR_SUPPORTED_DTYPES_TEXT


def _is_supported_debug_print_scalar_type(value_type: mlir_ir.Type) -> bool:
    if isinstance(value_type, (mlir_ir.F16Type, mlir_ir.F32Type)):
        return True
    if not mlir_ir.IntegerType.isinstance(value_type):
        return False
    int_type = mlir_ir.IntegerType(value_type)
    return int_type.width in (8, 16, 32) and (
        int_type.is_signless or int_type.is_unsigned
    )


def _print_scalar_type_error(value_type: Any) -> NoReturn:
    _op_error(
        "print",
        f"unsupported value type {value_type}; expected one of "
        f"{_DEBUG_PRINT_SUPPORTED_DTYPES_TEXT} scalar",
    )


_DEBUG_PRINT_MAX_FORMAT_FIELDS = 8
_DEBUG_PRINT_FIFO_BYTES = 1024 * 1024
_DEBUG_PRINT_FORMAT_TLV_BYTES = 24
_DEBUG_PRINT_FORMAT_SLOT_BYTES = 8


class _TlaPrintSignature(inspect.Signature):
    def __str__(self) -> str:
        return "(value, *args, /)"


def _validate_debug_print_format_text(format_value: str) -> None:
    if "\x00" in format_value:
        _op_error("print", "format string must not contain embedded NUL")
    if not format_value.isascii():
        _op_error("print", "format string must contain ASCII only")


def _scan_debug_print_format(format_value: str) -> tuple[int, int]:
    fields = 0
    generated_length = 0
    i = 0
    while i < len(format_value):
        char = format_value[i]
        if char == "{":
            if i + 1 >= len(format_value):
                _op_error("print", "malformed format string")
            next_char = format_value[i + 1]
            if next_char == "{":
                generated_length += 1
                i += 2
                continue
            if next_char == "}":
                fields += 1
                if fields > _DEBUG_PRINT_MAX_FORMAT_FIELDS:
                    _op_error(
                        "print",
                        f"formatted print supports at most {_DEBUG_PRINT_MAX_FORMAT_FIELDS} fields",
                    )
                generated_length += 2
                i += 2
                continue
            close = format_value.find("}", i + 1)
            if close < 0:
                _op_error("print", "malformed format string")
            _op_error("print", "unsupported format field")
        if char == "}":
            if i + 1 < len(format_value) and format_value[i + 1] == "}":
                generated_length += 1
                i += 2
                continue
            _op_error("print", "malformed format string")
        elif char == "%":
            generated_length += 2
        else:
            generated_length += 1
        i += 1
    return fields, generated_length


def _check_debug_print_record_size(fields: int, generated_length: int) -> None:
    record_bytes = (
        _DEBUG_PRINT_FORMAT_TLV_BYTES
        + fields * _DEBUG_PRINT_FORMAT_SLOT_BYTES
        + generated_length
        + 1
    )
    record_bytes = (
        (record_bytes + _DEBUG_PRINT_FORMAT_SLOT_BYTES - 1)
        // _DEBUG_PRINT_FORMAT_SLOT_BYTES
        * _DEBUG_PRINT_FORMAT_SLOT_BYTES
    )
    if record_bytes > _DEBUG_PRINT_FIFO_BYTES:
        _op_error(
            "print",
            f"formatted print record exceeds {_DEBUG_PRINT_FIFO_BYTES} byte debug FIFO limit",
        )


def _parse_debug_print_format(format_value: str) -> tuple[int, int]:
    _validate_debug_print_format_text(format_value)
    fields, generated_length = _scan_debug_print_format(format_value)
    _check_debug_print_record_size(fields, generated_length)
    return fields, generated_length


def _check_formatted_print_arg(value: Any) -> None:
    resolved = _resolve_bound_value(value)
    if _category(value) == "tensor" or (
        isinstance(resolved, mlir_ir.Value)
        and _tla_type_bridge.type_is_tensor(resolved.type)
    ):
        _op_error("print", "tensor arguments are unsupported in formatted print calls")
    if isinstance(resolved, bool):
        _print_scalar_type_error("bool")


def _materialize_debug_print_numeric(
    value: Numeric, *, loc: mlir_ir.Location | None
) -> mlir_ir.Value:
    value_type = type(value).mlir_type()
    if isinstance(value.value, mlir_ir.Value):
        return value.value
    if mlir_ir.IntegerType.isinstance(value_type):
        int_type = mlir_ir.IntegerType(value_type)
        if int_type.is_unsigned:
            signless_type = mlir_ir.IntegerType.get_signless(int_type.width)
            constant = mlir_ir.Operation.create(
                "arith.constant",
                results=[signless_type],
                attributes={
                    "value": mlir_ir.IntegerAttr.get(signless_type, int(value.value))
                },
                loc=loc,
            ).results[0]
            return mlir_ir.Operation.create(
                "builtin.unrealized_conversion_cast",
                operands=[constant],
                results=[value_type],
                loc=loc,
            ).results[0]
    return value.ir_value(loc=loc)


def _print_scalar_operand(value: Any, *, loc: mlir_ir.Location | None) -> mlir_ir.Value:
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        _print_scalar_type_error("bool")
    if isinstance(resolved, int):
        if not _DEBUG_PRINT_I32_MIN <= resolved <= _DEBUG_PRINT_I32_MAX:
            _op_error("print", f"Python int {resolved} is outside signless i32 range")
        return _const_i32(resolved, loc=loc)
    if isinstance(resolved, float):
        return _const_f32(resolved, loc=loc)
    if isinstance(resolved, Numeric):
        dtype = type(resolved).dtype.lower()
        if dtype not in _DEBUG_PRINT_SUPPORTED_DTYPES:
            _print_scalar_type_error(dtype)
        return _materialize_debug_print_numeric(resolved, loc=loc)
    if isinstance(resolved, VectorSSA):
        resolved = _resolve_bound_value(resolved.value)
    if not isinstance(resolved, mlir_ir.Value):
        _print_scalar_type_error(_type_name(value))

    value_type = resolved.type
    if _is_supported_debug_print_scalar_type(value_type):
        return resolved
    _print_scalar_type_error(value_type)


def _emit_scalar_print(value: Any, *, loc: mlir_ir.Location | None) -> None:
    if isinstance(value, bool):
        _print_scalar_type_error("bool")
    if (
        isinstance(value, int)
        and not _DEBUG_PRINT_I32_MIN <= value <= _DEBUG_PRINT_I32_MAX
    ):
        _op_error("print", f"Python int {value} is outside signless i32 range")
    _require_frontend_state("print")
    _runtime._require_enclosing_cube_or_vector("print")
    _tla_ops_gen.debug_print([_print_scalar_operand(value, loc=loc)], loc=loc)


def _emit_formatted_print(
    format_value: str, args: Sequence[Any], *, loc: mlir_ir.Location | None
) -> None:
    field_count, _ = _parse_debug_print_format(format_value)
    if field_count != len(args):
        _op_error(
            "print",
            f"format argument count mismatch: format has {field_count} fields but got {len(args)} arguments",
        )
    for arg in args:
        _check_formatted_print_arg(arg)
    _require_frontend_state("print")
    _runtime._require_enclosing_cube_or_vector("print")
    operands = []
    for arg in args:
        operand = _print_scalar_operand(arg, loc=loc)
        operand_type = operand.type
        if not _is_supported_debug_print_scalar_type(operand_type):
            _print_scalar_type_error(operand_type)
        operands.append(operand)
    _tla_ops_gen.debug_print(operands, format=format_value, loc=loc)


# CANN's 1 MiB debug FIFO reserves 48 bytes for the shape TLV and 72 bytes
# for the tensor TLV. Its 32-byte payload alignment leaves 262_112 f32 values:
# floor((1 MiB - 48 - 72) / 32) * (32 / sizeof(f32)).
_PRINT_TENSOR_MAX_F32_ELEMENTS = 262_112
_LOCAL_PRINT_MAX_BYTE_OFFSET = (1 << 32) - 1


def _print_tensor_shape_pattern_leaves(tree: Any) -> tuple[int, ...]:
    if isinstance(tree, (tuple, list)):
        leaves: list[int] = []
        for item in tree:
            leaves.extend(_print_tensor_shape_pattern_leaves(item))
        return tuple(leaves)
    if isinstance(tree, int):
        return (tree,)
    if tree is None:
        return (-1,)
    _op_error("print", "requires static or dynamic integer shape metadata")


def _static_nz_physical_element_offset(
    descriptor: TlaTensorTypeDescriptor, coord: tuple[int, int]
) -> int:
    """Map a root-relative logical coordinate through an NZ-family packed layout."""

    packed_shape = _print_tensor_shape_pattern_leaves(descriptor.shape)
    packed_stride = _print_tensor_shape_pattern_leaves(descriptor.stride)
    if (
        len(packed_shape) != 4
        or len(packed_stride) != 4
        or any(value <= 0 for value in packed_shape)
        or any(value < 0 for value in packed_stride)
    ):
        _op_error(
            "print",
            "L1 tensor printing requires static packed shape and stride metadata",
        )
    row, col = coord
    packed_coord = (
        row % packed_shape[0],
        row // packed_shape[0],
        col % packed_shape[2],
        col // packed_shape[2],
    )
    return sum(
        index * stride
        for index, stride in zip(packed_coord, packed_stride, strict=True)
    )


def _emit_tensor_print(
    value: Any, length: Any, *, loc: mlir_ir.Location | None
) -> None:
    """Dump a physical prefix of one rank-1/rank-2 supported GM, UB, L1, or L0C tensor.

    The logical tensor shape is display metadata; values are read contiguously
    from the effective physical address without gathering through strides.
    Static or runtime shape leaves and static or SSA lengths are supported in
    AIC-only or AIV-only C310 kernels across one or more launch blocks.
    """
    if _runtime._current_frontend_state() is None:
        _op_error("print", "tensor printing is only available in lowered Tla IR")
    in_cube = _runtime._has_enclosing_region("cube")
    in_vector = _runtime._has_enclosing_region("vector")
    if not in_cube and not in_vector:
        _op_error("print", "must be nested inside tla.cube() or tla.vector()")
    if in_cube and in_vector:
        _op_error("print", "mixed cube/vector placement is unsupported")

    if hasattr(value, "value") and _tla_type_bridge.type_is_tensor(value.value.type):
        value = value.value
    if not isinstance(value, mlir_ir.Value) or not _tla_type_bridge.type_is_tensor(
        value.type
    ):
        _op_error("print", "expected a TLA tensor")
    descriptor = _tla_tensor_type_for_mlir_value(value)
    addrspace = descriptor.addrspace.lower()
    if addrspace not in ("gm", "ub", "l1", "l0c"):
        _op_error("print", "requires a GM-, UB-, L1-, or L0C-resident tensor")
    if descriptor.element_type not in _PRINT_TENSOR_SUPPORTED_DTYPES:
        _op_error(
            "print",
            f"unsupported tensor dtype {descriptor.element_type}; supported dtypes: "
            f"{_PRINT_TENSOR_SUPPORTED_DTYPES_TEXT}",
        )
    if addrspace == "ub" and not in_vector:
        _op_error("print", "UB tensor printing requires AIV placement")
    if addrspace == "l1":
        if not in_cube:
            _op_error("print", "L1 tensor printing requires AIC placement")
        if descriptor.layout_tag not in ("zN", "nZ"):
            _op_error("print", "L1 tensor printing requires a zN or nZ layout")
    if addrspace == "l0c":
        if not in_cube:
            _op_error("print", "L0C tensor printing requires AIC placement")
        if descriptor.layout_tag != "L0Clayout":
            _op_error("print", "L0C tensor printing requires L0Clayout")
    logical_shape = (
        descriptor.origin_shape
        if addrspace in ("l1", "l0c")
        or descriptor.layout_tag
        in (
            "zN",
            "nZ",
            "L0Clayout",
            "zNUnAlign",
            "zZMxScale",
            "nNMxScale",
        )
        else descriptor.shape
    )
    shape = _print_tensor_shape_pattern_leaves(logical_shape)
    if len(shape) not in (1, 2):
        _op_error("print", "requires a rank-1 or rank-2 tensor")
    if any(extent == 0 or extent < -1 for extent in shape):
        _op_error("print", "tensor element count must be positive")

    dynamic_shape = any(extent < 0 for extent in shape)
    if addrspace == "l1":
        coord = _print_tensor_shape_pattern_leaves(descriptor.coord)
        if (
            len(shape) != 2
            or len(coord) != 2
            or any(value < 0 for value in (*shape, *coord))
        ):
            _op_error(
                "print",
                "L1 tensor printing requires static rank-2 shape and coordinates",
            )
        rows, cols = shape
        row, col = coord
        c0_elements, _ = _c0_geometry(dtype_size_bits(descriptor.element_type))
        if descriptor.layout_tag == "zN":
            if cols != c0_elements:
                _op_error(
                    "print",
                    "zN L1 tensor printing requires logical shape "
                    f"[M, {c0_elements}] for {descriptor.element_type}",
                )
        else:
            if rows != c0_elements:
                _op_error(
                    "print",
                    "nZ L1 tensor printing requires logical shape "
                    f"[{c0_elements}, N] for {descriptor.element_type}",
                )
        element_offset = _static_nz_physical_element_offset(descriptor, (row, col))
        element_bytes = dtype_size_bytes(descriptor.element_type)
        byte_offset = element_offset * element_bytes
        if byte_offset > _LOCAL_PRINT_MAX_BYTE_OFFSET:
            _op_error("print", "requires an L1 byte offset representable as uint32")
        if math.gcd(descriptor.ptr_alignment, _builtins.abs(byte_offset)) < 32:
            _op_error(
                "print", "requires a statically proven 32-byte aligned L1 address"
            )
    elif addrspace == "l0c":
        coord = _print_tensor_shape_pattern_leaves(descriptor.coord)
        if (
            len(shape) != 2
            or len(coord) != 2
            or any(value < 0 for value in (*shape, *coord))
        ):
            _op_error(
                "print",
                "L0C tensor printing requires static rank-2 shape and coordinates",
            )
        rows, cols = shape
        row, col = coord
        if rows < 16 or cols < 16:
            _op_error(
                "print",
                "L0C print requires a complete 16x16 tile inside the logical shape",
            )
        if row % 16 or col % 16:
            _op_error(
                "print", "L0C tensor coordinates must be aligned to a 16x16 fractal"
            )
        element_offset = _static_nz_physical_element_offset(descriptor, (row, col))
        element_bytes = dtype_size_bytes(descriptor.element_type)
        byte_offset = element_offset * element_bytes
        if byte_offset > _LOCAL_PRINT_MAX_BYTE_OFFSET:
            _op_error("print", "requires an L0C byte offset representable as uint32")
        if math.gcd(descriptor.ptr_alignment, _builtins.abs(byte_offset)) < 32:
            _op_error(
                "print", "requires a statically proven 32-byte aligned L0C address"
            )
        shape = (16, 16)
    element_count = None if dynamic_shape else math.prod(shape)
    max_length = (
        8
        if addrspace == "l1"
        else 256
        if addrspace == "l0c"
        else _PRINT_TENSOR_MAX_F32_ELEMENTS
    )
    if length is None:
        if dynamic_shape:
            _op_error("print", "dynamic-shaped tensors require an explicit length")
        assert element_count is not None
        if element_count > max_length:
            _op_error(
                "print",
                "tensors exceeding the print capacity require an explicit length",
            )
        static_length = element_count
    else:
        resolved_length = _resolve_bound_value(length)
        if isinstance(resolved_length, bool):
            _op_error("print", "length must be an integer or integer SSA value")
        if (
            isinstance(resolved_length, Numeric)
            and not type(resolved_length).is_integer
        ):
            _op_error("print", "length must be an integer or integer SSA value")
        if isinstance(resolved_length, mlir_ir.Value) and (
            not mlir_ir.IntegerType.isinstance(resolved_length.type)
            and not isinstance(resolved_length.type, mlir_ir.IndexType)
        ):
            _op_error("print", "length must be an integer or integer SSA value")
        static_length = resolved_length if isinstance(resolved_length, int) else None
    if static_length is not None:
        if not 1 <= static_length <= max_length:
            _op_error(
                "print",
                f"length must be between 1 and {max_length} elements",
            )
        if element_count is not None and static_length > element_count:
            _op_error("print", "length must not exceed the tensor element count")
        if addrspace == "l0c" and static_length != 256:
            _op_error("print", "L0C tensor printing requires exactly 256 elements")
    length_value = _as_i64_value(
        static_length if static_length is not None else length, loc=loc
    )

    print_op = _tla_ops_gen.print_tensor(value, length_value, shape, loc=loc)
    operation = print_op.operation
    parent = operation.parent
    while parent is not None:
        if parent.name in {"scf.if", "scf.for", "scf.while"}:
            operation.attributes["dynamic_execution"] = mlir_ir.UnitAttr.get()
            break
        parent = parent.parent


def print(*args: object, **kwargs: object) -> None:
    """Directory: Debug APIs
    Description:
        Print a scalar, a formatted scalar string, or a supported physical tensor prefix from GM, UB, L1, or L0C inside a `cube` / `vector` region.

        Parameters:
        - *`args`* (`object`): Values to print (variadic positional arguments). Required.
        - **`kwargs`** (`object`): Keyword arguments are not accepted; passing any raises an error. Optional.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()` or `tla.vector()`.
        - GM/UB support the documented tensor dtypes and a rank-1/rank-2 physical prefix.
        - L1 requires AIC placement, a supported dtype, a 32-byte-aligned zN/nZ dense prefix whose C0 dimension spans 32 bytes, and at most 8 elements.
        - L0C requires AIC placement, a supported dtype, L0Clayout, and exactly one 32-byte-aligned 16x16 typed view (256 elements).
        - Aggregate FIFO use is checked for statically executed tensor prints. Runtime lengths, dynamic shapes, and prints under dynamic control flow emit a warning because their aggregate output cannot be bounded before launch; excessive output may be truncated or omitted.

        Example:
        ```python
        with tla.vector():
            tla.print(x_scalar)
            tla.print(x_ub, 64)  # tensor + prefix length

        with tla.cube():
            tla.print(x_l0c, 256)  # one aligned 16x16 CO1 tile
        ```
    """
    if kwargs:
        _op_error("print", "does not accept keyword arguments")
    if len(args) < 1:
        _op_error("print", f"expects at least one positional argument; got {len(args)}")

    original_value = args[0]
    value = _resolve_bound_value(original_value)
    if _category(original_value) == "tensor":
        if len(args) > 2:
            _op_error(
                "print",
                f"tensor printing expects one tensor and optional length; got {len(args)} arguments",
            )
        if _runtime._current_frontend_state() is None:
            return _emit_tensor_print(
                value, args[1] if len(args) == 2 else None, loc=None
            )
        return _emit_tensor_print(
            value,
            args[1] if len(args) == 2 else None,
            loc=_capture_user_loc(),
        )
    if isinstance(value, str):
        loc = (
            _capture_user_loc()
            if _runtime._current_frontend_state() is not None
            else None
        )
        return _emit_formatted_print(value, args[1:], loc=loc)
    if len(args) == 2:
        _op_error("print", "length is only valid when printing a tensor")
    if len(args) > 2:
        _op_error("print", "format string must be a host Python str")
    loc = (
        _capture_user_loc() if _runtime._current_frontend_state() is not None else None
    )
    return _emit_scalar_print(value, loc=loc)


print.__signature__ = _TlaPrintSignature(
    [
        inspect.Parameter("value", inspect.Parameter.POSITIONAL_ONLY),
        inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
    ]
)


def _require_dtype(op_name: str, name: str, value: Any, position: int) -> None:
    if isinstance(value, mlir_ir.Type):
        return
    if (
        isinstance(value, type)
        and issubclass(value, Numeric)
        and getattr(value, "dtype", "")
    ):
        return
    if isinstance(value, str):
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): "
            "use a concrete Numeric type (e.g. tla.Float32) or mlir_ir.Type, not a str.",
        )
    _op_error(
        op_name,
        f"invalid argument '{name}' (position {position}): "
        f"expected mlir_ir.Type or concrete Numeric (e.g. tla.Float32), got {_type_name(value)}",
    )


def _require_byte_alignment(op_name: str, value: Any, position: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _op_error(
            op_name,
            f"invalid argument 'byte_alignment' (position {position}): expected positive_int, "
            f"got {_type_name(value)}",
        )
    return int(value)


def _require_allocation_dtype(op_name: str, dtype: Any) -> tuple[type[Numeric], int]:
    if (
        not isinstance(dtype, type)
        or not issubclass(dtype, Numeric)
        or not getattr(dtype, "dtype", "")
    ):
        _op_error(
            op_name,
            f"invalid argument 'dtype' (position 1): expected concrete Numeric "
            f"(e.g. tla.Float32), got {_type_name(dtype)}",
        )
    width = int(getattr(dtype, "width", 0) or 0)
    # Sub-byte widths are allowed only when a whole number of them tiles a byte
    # (fp4's 4 bits), so an allocation is still a whole number of bytes.
    if width <= 0 or (width % 8 != 0 and 8 % width != 0):
        _op_error(
            op_name,
            f"unsupported allocation dtype {dtype.dtype}; expected byte-addressable "
            "fixed-width scalar Numeric",
        )
    element_bits = dtype_size_bits(str(dtype.dtype))
    if element_bits <= 0:
        _op_error(
            op_name,
            f"unsupported allocation dtype {dtype.dtype}; expected a fixed-width "
            "scalar Numeric whose width tiles a byte",
        )
    return dtype, element_bits


def _static_allocation_size_bytes(
    op_name: str,
    shape: ShapeLike,
    dtype: type[Numeric],
    element_bits: int,
) -> int:
    _check_shape(shape)
    num_elements = 1
    for dim in _flatten_tla_tuple(shape):
        dim_const = _const_int_value(dim)
        if dim_const is None:
            raise TlaLoweringError(
                f"tla.{op_name} requires a static shape (compile-time constants); "
                "dynamic shapes are not supported."
            )
        if dim_const <= 0:
            _op_error(
                op_name,
                f"Expected size in shape to be strictly positive, but got {dim_const}",
            )
        num_elements *= int(dim_const)

    # In bits, so a sub-byte element type is not rounded up per element; the
    # width guard above ensures the total lands on a byte boundary.
    size_bytes = _ceil_div(num_elements * element_bits, _BITS_PER_BYTE)
    if size_bytes <= 0 or size_bytes > 9_223_372_036_854_775_807:
        raise TlaLoweringError(
            f"tla.{op_name} allocation size_bytes must be in [1, 2**63-1] "
            f"for tla.alloc_ptr {{size_bytes : i64}}; got {size_bytes} "
            f"for dtype {dtype.dtype}"
        )
    return size_bytes


@dsl_user_op
def make_shape(
    *components: IndexTree,
    loc: mlir_ir.Location | None = None,
) -> TlaShape:
    """Directory: Basic Data Types and Operations
    Description:
        Build a packed `!tla.shape`; components may be nested tuples.

        Parameters:
        - *`components`* (`IndexTree`): Shape components per dimension. Nested
          tuples are used for `zN` / `nZ` / `L0Clayout` / `zNUnAlign`
          physical layouts. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Provide at least one shape component.
        - RowMajor / ColumnMajor: use a 2D shape `(M, N)`.
        - `zN` / `nZ` / `L0Clayout` / `zNUnAlign`: for `make_layout` /
          `make_tensor`, use a nested physical shape `((m0, m1), (n0, n1))`.
          A plain 2D `(M, N)` is **not** valid for those tags; either nest it, or
          prefer `make_tensor_like(..., layoutTag=zN)` which remaps from the
          logical 2D `origin_shape`.

        Example:
        ```python
        # RowMajor / ColumnMajor (logical 2D):
        shape = tla.make_shape(256, 128)

        # zN physical shape (f16, logical 128x64):
        # m0=16, m1=8 (=128/16); n0=16, n1=4 (=64/16)
        zn_shape = tla.make_shape((16, 8), (16, 4))
        ```
    """
    if len(components) == 0:
        _op_error("make_shape", "expected at least 1 component")
    _require_frontend_state("make_shape")
    v = _pack_shape(components, loc=loc)
    return _Shape(shape_value=v, components=components)


@dsl_user_op
def make_coord(
    *components: IndexTree,
    loc: mlir_ir.Location | None = None,
) -> TlaCoord:
    """Directory: Basic Data Types and Operations

    Description:
        Build a packed `!tla.coord`.

        Parameters:
        - *`components`* (`IndexTree`): Coordinate components per dimension. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Provide at least one coordinate component.

        Example:
        ```python
        coord = tla.make_coord(block_row, 0)
        ```
    """
    if len(components) == 0:
        _op_error("make_coord", "expected at least 1 component")
    _require_frontend_state("make_coord")
    v = _pack_coord(components, loc=loc)
    return _Coord(coord_value=v, components=components)


@dsl_user_op
def make_stride(
    *components: IndexTree,
    loc: mlir_ir.Location | None = None,
) -> TlaStride:
    """Directory: Basic Data Types and Operations

    Description:
        Build a packed `!tla.stride` (same nesting rules as `make_shape`).

        Parameters:
        - *`components`* (`IndexTree`): Stride components per dimension. Required.

          Common patterns:

          | Layout | Typical `make_stride(...)` | Meaning |
          |---|---|---|
          | RowMajor 2D `(M, N)` | `(N, 1)` | Row step is `N` elements; column step is 1 |
          | ColumnMajor 2D `(M, N)` | `(1, M)` | Row step is 1; column step is `M` |
          | `zN` / `nZ` / … | nested `((s00, s01), (s10, s11))` | Same nesting as physical `shape`; values must match the layout tag |

          For `zN` / `nZ` / `L0Clayout` / `zNUnAlign`, C0 is 32 bytes and
          the M-side block size is 16. Let
          `elems_per_c0 = 32 // sizeof(dtype)` (f16→16, f32→8) and
          `elems_per_block = elems_per_c0 * 16`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Provide at least one stride component.
        - For `zN` / `nZ` / `L0Clayout` / `zNUnAlign`, stride values must
          match the layout tag (see examples); `make_tensor` checks them against
          `shape` + `layoutTag`.

        Example:
        ```python
        # RowMajor 2D (logical 256x128, tightly packed):
        stride = tla.make_stride(128, 1)

        # f16 zN for logical origin (128, 64):
        # shape = ((16, 8), (16, 4))
        # stride = ((elems_per_c0, elems_per_block),
        #           (1, round_up(M, 16) * elems_per_c0))
        #        = ((16, 256), (1, 2048))
        zn_stride = tla.make_stride((16, 256), (1, 2048))

        # f16 nZ for the same logical (128, 64):
        # shape = ((16, 8), (16, 4))
        # stride = ((1, round_up(N, 16) * elems_per_c0),
        #           (elems_per_c0, elems_per_block))
        #        = ((1, 1024), (16, 256))
        nz_stride = tla.make_stride((1, 1024), (16, 256))
        ```
    """
    if len(components) == 0:
        _op_error("make_stride", "expected at least 1 component")
    _require_frontend_state("make_stride")
    v = _pack_stride(components, loc=loc)
    return _Stride(stride_value=v, components=tuple(components))


@dsl_user_op
def make_layout(
    shape: _Shape,
    stride: _Stride,
    *,
    origin_shape: _Shape | None = None,
    layoutTag: _LayoutTag | None = None,
    loc: mlir_ir.Location | None = None,
) -> TlaLayout:
    """Directory: Basic Data Types and Operations

    Description:
        Compose a `!tla.layout` from shape / stride (maps to `tla.make_layout`).

        Parameters:
        - `shape` (`_Shape`): Layout shape from `tla.make_shape`. Required.
          RowMajor / ColumnMajor: 2D `(M, N)`.
          `zN` / `nZ` / `L0Clayout` / `zNUnAlign`: nested
          `((m0, m1), (n0, n1))`.
        - `stride` (`_Stride`): Layout stride from `tla.make_stride`. Required.
          Use the same nesting as `shape`.
        - `origin_shape` (`_Shape | None`): Logical working shape (true data size
          before alignment fill). Optional, default `None`. Copy / tiling use this
          logical size; physical packing lives in `shape` / `stride`.
        - `layoutTag` (`_LayoutTag | None`): Layout tag (e.g. `tla.arch.RowMajor`,
          `tla.arch.zN`). Optional, default `None` (RowMajor).

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - `shape` / `stride` must be values returned by `make_shape` / `make_stride`.
        - For RowMajor / ColumnMajor, an omitted `origin_shape` is inferred as
          `shape`. For `zN` / `nZ` / `L0Clayout` / `zNUnAlign`, it is
          inferred as `(m0*m1, n0*n1)` from `shape=((m0,m1),(n0,n1))`.
        - Do **not** pass a plain 2D `shape` with a `zN` / `nZ` /
          `L0Clayout` / `zNUnAlign` tag; that fails checks. Either build the nested
          physical shape, or use `make_tensor_like(ptr, like, layoutTag=...)` so
          the front end remaps from `like.origin_shape`.

        Example:
        ```python
        # RowMajor 2D:
        layout = tla.make_layout(
            tla.make_shape(256, 128),
            tla.make_stride(128, 1),
            layoutTag=tla.arch.RowMajor,
        )

        # Explicit f16 zN (logical 128x64 → nested physical + 2D origin):
        # Before: logical ND tile is (128, 64).
        zn = tla.make_layout(
            tla.make_shape((16, 8), (16, 4)),
            tla.make_stride((16, 256), (1, 2048)),
            origin_shape=tla.make_shape(128, 64),  # logical size for copy/tiling
            layoutTag=tla.arch.zN,
        )
        # After: layout.shape is zN-packed; layout.origin_shape stays (128, 64).
        ```
    """
    if not isinstance(shape, _Shape) or not isinstance(stride, _Stride):
        _op_error(
            "make_layout",
            "expected shape from tla.make_shape (TlaShape) and stride from tla.make_stride (TlaStride); "
            f"got shape={_type_name(shape)}, stride={_type_name(stride)}",
        )
    if origin_shape is not None and not isinstance(origin_shape, _Shape):
        _op_error(
            "make_layout",
            "expected origin_shape from tla.make_shape (TlaShape) or None; "
            f"got {_type_name(origin_shape)}",
        )
    _require_frontend_state("make_layout")
    layout_token = _resolve_arch_layout_tag(layoutTag, for_op="make_layout")
    if origin_shape is None and layout_token in _LINEAR_LAYOUT_TOKENS:
        origin_shape = shape
    elif origin_shape is None and layout_token in _NZ_FAMILY_LAYOUT_TOKENS:
        inferred_origin = _infer_padded_origin_tree_from_nz_family_shape(
            _components_to_index_tree(shape._components), for_op="make_layout"
        )
        origin_shape = make_shape(*inferred_origin, loc=loc)
    # Rank/layout consistency is enforced by C++ LayoutType::verify via LayoutType.get.
    shape_val = shape._shape_value
    stride_val = stride._stride_value
    origin_for_type: mlir_ir.Value | None = (
        origin_shape._shape_value if origin_shape is not None else None
    )
    layout_ty = LayoutType.get(
        shape_val, stride_val, origin_for_type, layout_tag=layout_token
    )
    origin_ssa: mlir_ir.Value | None = None
    if origin_shape is not None and origin_shape._shape_value is not shape_val:
        origin_ssa = origin_shape._shape_value
    attrs: dict[str, mlir_ir.Attribute] = {}
    if layout_token != "RowMajor":
        ctx = loc.context if loc is not None else mlir_ir.Context.current
        attrs["layoutTag"] = mlir_ir.Attribute.parse(
            f"#tla.layout_tag<{layout_token}>", ctx
        )
    operands: list[mlir_ir.Value] = [shape_val, stride_val]
    if origin_ssa is not None:
        operands.append(origin_ssa)
    op = mlir_ir.Operation.create(
        "tla.make_layout",
        operands=operands,
        results=[layout_ty],
        attributes=attrs,
        loc=loc,
    )
    return _Layout(
        layout_value=op.results[0],
        shape=shape,
        stride=stride,
        origin_shape=origin_shape,
        layout_tag=layout_token,
    )


_MAKE_LAYOUT_WITH_TAG_TOKENS = frozenset(
    {
        "RowMajor",
        "ColumnMajor",
        "zN",
        "nZ",
        "zZMxScale",
        "nNMxScale",
        "L0Clayout",
        "zNUnAlign",
    }
)

# The on-chip MxScale tags pin the C0 geometry at 2 elements / 32 bytes, which
# is only true for e8m0. Any other element type under them compiles into a
# silently wrong layout.
_MAKE_LAYOUT_WITH_TAG_MX_TOKENS = frozenset({"zZMxScale", "nNMxScale"})


@dsl_user_op
def make_layout_with_tag(
    shape: tuple,
    dtype: Any,
    layoutTag: _LayoutTag | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaLayout:
    """Directory: Basic Data Types and Operations

    Description:
        Compose a `!tla.layout` directly from a flat 2D logical shape, an element
        type and a layout tag (maps to
        `tla.make_layout` after the front end remaps the physical trees).

        `make_layout_with_tag` takes the extent straight from the
        caller, so an on-chip tile can be larger than the GM tile it is fed from
        (e.g. a K extent rounded up to 64 for MX mmad).

        Parameters:
        - `shape` (`tuple`): Flat 2D logical extent `(rows, cols)`. Each leaf is
          an `int` or a runtime `Numeric` (dynamic leaf). Required.
        - `dtype` (`Numeric` subclass): Element type; drives the C0/fractal
          geometry of the remapped physical trees. Required.
        - `layoutTag` (`_LayoutTag | None`): Layout tag (e.g. `tla.arch.zN`).
          Optional, default `None` (RowMajor).

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Only for on-chip tensor layouts: the composed layout is meant to be
          landed on an on-chip pointer via `tla.make_tensor(ptr, layout)`, not
          bound to GM host memory.
        - `shape` must be a flat 2-tuple; nested trees and `None` leaves are
          rejected — the whole extent is caller-owned.
        - The origin shape of the resulting layout equals `shape` (the aligned
          extent); pair with `tla.make_tensor(ptr, layout)` to land it on an
          on-chip pointer.
        - On-chip tags only: `zZMxScale` / `nNMxScale` require element type
          `tla.Float8E8M0` (their C0 is fixed at 2 elements / 32 bytes by the
          e8m0 format). The GM-side MxScale tags (`RowMajorMxScaleA`,
          `ColMajorMxScaleA`, `RowMajorMxScaleB`, `ColMajorMxScaleB`) are
          rejected.

        Example:
        ```python
        # fp8 zN tile with M cropped at runtime and K rounded up to 64:
        lay = tla.make_layout_with_tag((m_valid, k_l0), dtype_a, tla.arch.zN)
        t_l1a = tla.make_tensor(l1a, lay)
        ```
    """
    _require_frontend_state("make_layout_with_tag")
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(isinstance(leaf, (tuple, list)) for leaf in shape)
    ):
        _op_error(
            "make_layout_with_tag",
            "expected a flat 2D logical shape tuple (rows, cols) with int or "
            f"Numeric leaves; got {shape!r}",
        )
    try:
        dtype_str = _dtype_to_str(dtype)
    except TypeError as exc:
        raise TlaLoweringError(
            "tla.make_layout_with_tag cannot derive element type from dtype "
            f"{dtype!r}: {exc}"
        ) from exc
    if dtype_size_bits(dtype_str) <= 0:
        _op_error(
            "make_layout_with_tag",
            f"unsupported element type for layout construction: {dtype_str}",
        )
    layout_token = _resolve_arch_layout_tag(layoutTag, for_op="make_layout_with_tag")
    if layout_token not in _MAKE_LAYOUT_WITH_TAG_TOKENS:
        _op_error(
            "make_layout_with_tag",
            f"unsupported layout tag '{layout_token}'; expected one of "
            + ", ".join(sorted(_MAKE_LAYOUT_WITH_TAG_TOKENS)),
        )
    if layout_token in _MAKE_LAYOUT_WITH_TAG_MX_TOKENS:
        dtype_token = str(getattr(dtype, "dtype", "")).strip().lower()
        if dtype_token != "f8e8m0":
            _op_error(
                "make_layout_with_tag",
                f"layout tag '{layout_token}' requires element type "
                f"tla.Float8E8M0, got {dtype_token!r}",
            )
    trees = _materialize_layout_trees_from_origin(shape, dtype_str, layout_token)
    if trees is None:
        _op_error(
            "make_layout_with_tag",
            f"cannot materialize layout trees for shape={shape!r}, "
            f"dtype={dtype_str}, layoutTag='{layout_token}'",
        )
    shape_tree, stride_tree, _coord_tree, origin_tree = trees
    sh = make_shape(*shape_tree, loc=loc)
    st = make_stride(*stride_tree, loc=loc)
    og = make_shape(*origin_tree, loc=loc)
    return make_layout(sh, st, origin_shape=og, layoutTag=layoutTag, loc=loc)


def _emit_tile_view(
    source: Any,
    shape: _Shape,
    coord: _Coord,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaTensor:
    _require_category("tile_view", "source", source, "tensor", 0)
    if not isinstance(shape, _Shape) or not isinstance(coord, _Coord):
        _op_error(
            "tile_view",
            f"expected shape from tla.make_shape (TlaShape) and coord from tla.make_coord (TlaCoord); "
            f"got shape={_type_name(shape)}, coord={_type_name(coord)}",
        )
    _require_frontend_state("tile_view")
    source_value = _as_value(source)
    view_ty = _format_tensor_type_descriptor(
        source_value, shape._components, coord._components
    )
    result = _tla_ops_gen.tile_view(
        _coerce_type(view_ty),
        source_value,
        shape._shape_value,
        coord._coord_value,
        loc=loc,
    )
    _register_tla_tensor_type(result, view_ty)
    try:
        source_meta = {
            "stride": _tensor_metadata_field(source_value, "stride"),
            "coord": _tensor_metadata_field(source_value, "coord"),
            "origin_shape": _tensor_metadata_field(source_value, "origin_shape"),
        }
        tile_shape_tree = _components_to_index_tree(shape._components)
        tile_coord_tree = _components_to_index_tree(coord._components)
        shape_meta = _metadata_from_type_tree(
            view_ty.shape, list(_flatten_tla_tuple(tile_shape_tree))
        )
        metadata = {
            "shape": shape_meta,
            "stride": source_meta["stride"],
            "coord": _tree_add(source_meta["coord"], tile_coord_tree),
            "origin_shape": _tree_crop_origin(
                source_meta["origin_shape"], tile_shape_tree, tile_coord_tree
            ),
            "dtype": view_ty.element_type,
            "addrspace": view_ty.addrspace,
            "layout_tag": view_ty.layout_tag,
        }
        _register_tla_tensor_metadata(result, metadata)
    except Exception:
        # Keep lowering permissive: metadata property access falls back to type parsing.
        pass
    return _Tensor(result)


@dsl_user_op
def tile_view(
    source: Tensor,
    shape: _Shape,
    coord: _Coord,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaTensor:
    """Directory: Basic Data Types and Operations

    Description:
        Create a tile view on a `!tla.tensor` source at tile-coordinate granularity.

        Parameters:
        - `source` (`Tensor`): Source `!tla.tensor`. Required.
        - `shape` (`_Shape`): Tile shape from `tla.make_shape`. Required.
        - `coord` (`_Coord`): Tile coordinate from `tla.make_coord` (tile granularity). Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - `coord` is tile-granularity; the front end converts it to an element offset using `shape`.

        Example:
        ```python
        tile = tla.tile_view(
            source, tla.make_shape(256, 128), tla.make_coord(block_row, 0)
        )
        ```
    """
    if not isinstance(shape, _Shape) or not isinstance(coord, _Coord):
        _op_error(
            "tile_view",
            f"expected shape from tla.make_shape (TlaShape) and coord from tla.make_coord (TlaCoord); "
            f"got shape={_type_name(shape)}, coord={_type_name(coord)}",
        )
    normalized_coord = normalize_tile_view_coord(
        shape_components=shape._components,
        coord_components=coord._components,
    )
    return _emit_tile_view(
        source,
        shape,
        make_coord(*normalized_coord, loc=loc),
        loc=loc,
    )


@dsl_user_op
def get_tile(
    tensor: Tensor,
    coord: TlaCoord,
    shape: TlaShape,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaTensor:
    """Directory: Basic Data Types and Operations

    Description:
        Extract a sub-tile of a tensor as a new `!tla.tensor` built with
        `tla.make_tensor` over the source's backing pointer.

        The returned tensor shares the source's buffer geometry -- the
        `stride` tree passes through unchanged -- and carries:

        - `coord`: `coord + source.coord` (element offsets, per dim);
        - `origin_shape`: `shape` cropped against the source's `origin_shape`
          (per dim `min(shape[d], origin[d] - coord[d])`), so the tile never
          runs past the source's logical extent;
        - `shape`: the physical packing regenerated from the cropped extent
          (for zN/nZ the same fractal decomposition `make_layout_with_tag`
          would produce; for RowMajor/ColumnMajor the extent itself).

        Downstream ops see the tile's absolute coord and its own logical
        extent; for example `Tensor.fill` on the result covers exactly this
        tile.

        Parameters:
        - `tensor` (`Tensor`): Source `!tla.tensor`. Required.
        - `coord` (`TlaCoord`): Tile start from `tla.make_coord`, flat 2-D
          element offsets relative to the source. Required.
        - `shape` (`TlaShape`): Requested tile extent from `tla.make_shape`,
          flat 2-D. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - `coord` offsets must stay within the source's `origin_shape` -- the
          crop only trims the extent, it does not clamp the offsets.
        - Supported layout tags follow `tla.make_tensor` (RowMajor, ColumnMajor,
          zN, nZ, L0Clayout, zNUnAlign).

        Example:
        ```python
        # Zero the K-pad tail of an L1 A tile (see Tensor.fill):
        pad = tla.get_tile(t_l1a, tla.make_coord(0, k_valid),
                           tla.make_shape(m, k_l0 - k_valid))
        pad.fill(0)
        ```
    """
    op_name = "get_tile"
    _require_category(op_name, "tensor", tensor, "tensor", 0)
    if not isinstance(coord, _Coord) or not isinstance(shape, _Shape):
        _op_error(
            op_name,
            "expected coord from tla.make_coord (TlaCoord) and shape from "
            f"tla.make_shape (TlaShape); got coord={_type_name(coord)}, "
            f"shape={_type_name(shape)}",
        )
    _require_frontend_state(op_name)

    coord_components = coord._components
    shape_components = shape._components
    if (
        len(coord_components) != 2
        or any(isinstance(leaf, (tuple, str)) for leaf in coord_components)
        or len(shape_components) != 2
        or any(isinstance(leaf, (tuple, str)) for leaf in shape_components)
    ):
        _op_error(
            op_name,
            "get_tile expects a flat 2-D coord and shape; got coord="
            f"{coord_components}, shape={shape_components}",
        )

    source_value = _as_value(tensor)
    parent_shape = _tensor_metadata_field(source_value, "shape")
    parent_stride = _tensor_metadata_field(source_value, "stride")
    parent_coord = _tensor_metadata_field(source_value, "coord")
    parent_origin = _tensor_metadata_field(source_value, "origin_shape")
    parent_tag = _tensor_metadata_field(source_value, "layout_tag")
    for name, tree in (
        ("coord", parent_coord),
        ("origin_shape", parent_origin),
        ("shape", parent_shape),
        ("stride", parent_stride),
    ):
        if not isinstance(tree, tuple) or any(isinstance(leaf, str) for leaf in tree):
            _op_error(
                op_name,
                f"get_tile source {name} must be a resolved tree of int leaves, "
                f"got {tree!r}",
            )
    if (
        not isinstance(parent_coord, tuple)
        or len(parent_coord) != 2
        or not isinstance(parent_origin, tuple)
        or len(parent_origin) != 2
    ):
        _op_error(
            op_name,
            "get_tile source coord and origin_shape must be flat 2-D trees; got "
            f"coord={parent_coord}, origin_shape={parent_origin}",
        )

    # New coord = passed coord + the source's own coord, per dim; extent =
    # passed shape cropped against the source's origin (per dim
    # min(shape[d], origin[d] - coord[d])). Static leaves fold in Python;
    # dynamic leaves emit arith.minsi.
    new_coord_components = tuple(
        leaf + base for leaf, base in zip(coord_components, parent_coord, strict=True)
    )
    cropped = tuple(
        _tree_crop_origin(origin_leaf, shape_leaf, coord_leaf)
        for origin_leaf, shape_leaf, coord_leaf in zip(
            parent_origin, shape_components, coord_components, strict=True
        )
    )

    # Regenerate the physical shape from the cropped logical extent instead of
    # passing the parent's through: the tile carries its own extents. The
    # stride tree stays the source's -- the tile addresses the parent buffer,
    # whose pitch (e.g. zN's rows-ru stride leaf, RowMajor's column alignment)
    # follows the parent extent, not the tile's.
    if parent_tag in _LINEAR_LAYOUT_TOKENS:
        new_shape_tree = tuple(cropped)
    else:
        dtype_str = str(_tensor_metadata_field(source_value, "dtype"))
        materialized = _materialize_layout_trees_from_origin(
            tuple(cropped), dtype_str, str(parent_tag)
        )
        if materialized is None:
            _op_error(
                op_name,
                f"cannot materialize a physical shape for the cropped extent "
                f"{cropped!r} with dtype={dtype_str}, layoutTag={parent_tag!r}; "
                "falling back to the source's physical shape is not supported",
            )
        new_shape_tree = materialized[0]

    layout = make_layout(
        make_shape(*new_shape_tree, loc=loc),
        make_stride(*parent_stride, loc=loc),
        origin_shape=make_shape(*cropped, loc=loc),
        layoutTag=_LayoutTag(str(parent_tag)),
        loc=loc,
    )
    ptr = _emit_tensor_ptr(source_value, loc=loc)
    return make_tensor(
        ptr,
        layout,
        coord=make_coord(*new_coord_components, loc=loc),
        loc=loc,
    )


@dsl_user_op
def make_tensor(
    ptr: Pointer,
    layout: TlaLayout,
    coord: CoordLike | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaTensor:
    """Directory: Basic Data Types and Operations

    Description:
        Build a `!tla.tensor` from an explicit pointer, layout, and optional coord.

        Parameters:
        - `ptr` (`Pointer`): Underlying data pointer (`!tla.ptr`). Required.
        - `layout` (`TlaLayout`): Tensor layout from `tla.make_layout`. Required.
        - `coord` (`CoordLike | None`): Optional start coordinate; treated as zeros when omitted. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Pointer, layout, and coord must match the target address space and dtype.
        - `coord` defaults to a zero coord matching the layout's rank (rank-2 ->
          `make_coord(0, 0)`, rank-1 -> `make_coord(0)`). Element type and address
          space come from `ptr`'s `!tla.ptr`; layout tag, shape, stride, and origin
          come from the `!tla.layout` operand (origin defaults to `shape`).
        - Lowering supports RowMajor, ColumnMajor, zN, nZ, L0Clayout, and
          zNUnAlign. For `zN` / `nZ` / `L0Clayout` / `zNUnAlign`, physical
          `shape` / `stride` are nested 2x2, while logical coord / `origin_shape`
          stay 2D `(M, N)`. If `make_layout` omitted `origin_shape`, the logical
          size is inferred from the physical shape (for example `(m0*m1, n0*n1)`).
        - A full compile requires `ptr` to carry backing storage; an on-chip pointer
          from `allocate` (optionally via `recast_ptr`) is the supported form for
          runnable kernels.

        Example:
        ```python
        tensor = tla.make_tensor(ptr, layout, coord=tla.make_coord(0, 0))
        ```
    """
    _require_category("make_tensor", "ptr", ptr, "pointer", 0)
    if not isinstance(layout, _Layout):
        _op_error(
            "make_tensor",
            "invalid argument 'layout': expected tla.make_layout (TlaLayout); "
            f"got {_type_name(layout)}",
        )
    _require_frontend_state("make_tensor")
    ptr_value = _as_value(ptr)
    if not PtrType.isinstance(ptr_value.type):
        _op_error(
            "make_tensor",
            f"invalid argument 'ptr': expected !tla.ptr, got {ptr_value.type}",
        )
    ptr_ty = PtrType(ptr_value.type)
    addr = ptr_ty.addrspace
    try:
        dtype = _dtype_to_str(ptr_ty.pointee)
    except TypeError as exc:
        raise TlaLoweringError(
            f"tla.make_tensor cannot derive element type from ptr pointee {ptr_ty.pointee}"
        ) from exc
    if dtype not in _MAKE_TENSOR_SUPPORTED_ELEMENT_TYPES:
        raise TlaLoweringError(
            f"tla.make_tensor expects a supported element type, got [{dtype}]"
        )

    shape_tree = _components_to_index_tree(layout._shape._components)
    stride_tree = _components_to_index_tree(layout._stride._components)
    if layout._origin_shape is not None:
        origin_tree = _components_to_index_tree(layout._origin_shape._components)
    elif layout._layout_tag in _NZ_FAMILY_LAYOUT_TOKENS:
        origin_tree = _infer_padded_origin_tree_from_nz_family_shape(
            shape_tree, for_op="make_tensor"
        )
    else:
        origin_tree = shape_tree

    shape_leaf_count = len(_flatten_tla_tuple(shape_tree))
    stride_leaf_count = len(_flatten_tla_tuple(stride_tree))
    if layout._layout_tag in _LINEAR_LAYOUT_TOKENS:
        logical_rank = shape_leaf_count
        if logical_rank not in (1, 2) or stride_leaf_count not in (1, 2):
            raise TlaLoweringError(
                f"tla.make_tensor supports at most 2-D linear layouts (got shape rank "
                f"{shape_leaf_count}, stride rank {stride_leaf_count})"
            )
        if len(_flatten_tla_tuple(origin_tree)) != logical_rank:
            raise TlaLoweringError(
                "tla.make_tensor linear origin_shape rank must match layout rank"
            )
    elif layout._layout_tag in _NZ_FAMILY_LAYOUT_TOKENS:
        if not _is_nz_family_2x2_tree(shape_tree) or not _is_nz_family_2x2_tree(
            stride_tree
        ):
            raise TlaLoweringError(
                f"tla.make_tensor layout {layout._layout_tag!r} expects shape and "
                "stride as two 2-leaf groups ((m0, m1), (n0, n1))"
            )
        if not _is_flat_pair(origin_tree):
            raise TlaLoweringError(
                f"tla.make_tensor layout {layout._layout_tag!r} expects a flat "
                "logical 2-D origin_shape"
            )
        logical_rank = 2
    else:
        raise TlaLoweringError(
            f"tla.make_tensor does not support layout {layout._layout_tag!r}"
        )

    _validate_static_make_tensor_layout(
        shape_tree,
        stride_tree,
        dtype=dtype,
        layout_tag=layout._layout_tag,
    )

    if coord is None:
        coord = make_coord(*([0] * logical_rank), loc=loc)
    elif not isinstance(coord, _Coord):
        _op_error(
            "make_tensor",
            "invalid argument 'coord': expected tla.make_coord (TlaCoord) or None; "
            f"got {_type_name(coord)}",
        )
    coord_tree = _components_to_index_tree(coord._components)
    coord_rank = len(_flatten_tla_tuple(coord_tree))
    if coord_rank != logical_rank:
        raise TlaLoweringError(
            f"tla.make_tensor coord rank must match layout rank (got coord rank "
            f"{coord_rank}, expected {logical_rank})"
        )
    if layout._layout_tag in _NZ_FAMILY_LAYOUT_TOKENS and not _is_flat_pair(coord_tree):
        raise TlaLoweringError(
            f"tla.make_tensor layout {layout._layout_tag!r} expects a flat "
            "logical 2-D coord"
        )

    # Type trees spell dynamic leaves as ``None`` (``?`` in the ``!tla.tensor`` type);
    # the dynamic SSA values themselves travel in the make_shape / make_stride /
    # make_coord operands bundled into ``layout._layout_value`` / ``coord._coord_value``,
    # so the type only needs to mark which leaves are dynamic - same approach as
    # ``_format_tensor_type_descriptor`` (``tile_view``). The index trees above (which
    # carry the concrete ``int`` / Numeric leaf values) back the metadata below.
    shape_type_tree = _components_to_type_tree(layout._shape._components)
    stride_type_tree = _components_to_type_tree(layout._stride._components)
    origin_type_tree = (
        _components_to_type_tree(layout._origin_shape._components)
        if layout._origin_shape is not None
        else shape_type_tree
    )
    coord_type_tree = _components_to_type_tree(coord._components)

    result_desc = TlaTensorTypeDescriptor(
        layout=TlaLayoutDescriptor(
            shape=TlaIndexTreeType("shape", shape_type_tree),
            stride=TlaIndexTreeType("stride", stride_type_tree),
            origin_shape=TlaIndexTreeType("shape", origin_type_tree),
            layout_tag=layout._layout_tag,
        ),
        coord=coord_type_tree,
        element_type=dtype,
        addrspace=addr,
        ptr_alignment=ptr_ty.alignment,
    )
    op = mlir_ir.Operation.create(
        "tla.make_tensor",
        operands=[ptr_value, layout._layout_value, coord._coord_value],
        results=[_coerce_type(result_desc)],
        loc=loc,
    )
    out = op.results[0]
    _register_tla_tensor_type(out, result_desc)
    try:
        # Metadata carries the concrete leaf values (``int`` / Numeric) so that
        # downstream ops can do coord arithmetic on dynamic leaves; equivalent to
        # ``result_desc.metadata()`` for the static fields but preserving Numeric
        # for dynamic shape/stride/coord/origin (like ``tile_view`` does).
        _register_tla_tensor_metadata(
            out,
            {
                "shape": shape_tree,
                "stride": stride_tree,
                "coord": coord_tree,
                "origin_shape": origin_tree,
                "dtype": dtype,
                "addrspace": addr,
                "layout_tag": layout._layout_tag,
            },
        )
    except Exception:
        # Metadata property access falls back to type parsing when unavailable.
        pass
    return _Tensor(out)


@dsl_user_op
def make_tensor_like(
    ptr: Pointer,
    like: Tensor,
    layoutTag: _LayoutTag | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaTensor:
    """Directory: Basic Data Types and Operations

    Description:
        Build a same-shaped tensor on the given pointer using structured metadata from a reference tile.

        Parameters:
        - `ptr` (`Pointer`): Destination data pointer. Required.
        - `like` (`Tensor`): Reference tile providing structured tensor metadata. Required.
        - `layoutTag` (`_LayoutTag | None`): Layout tag overriding the reference tile. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - The reference tile must provide usable structured tensor metadata.
        - Element type comes from `ptr`'s `!tla.ptr` pointee; only on-chip destination pointers are accepted.

        Example:
        ```python
        dst = tla.make_tensor_like(ptr, like=src_tile, layoutTag=tla.arch.RowMajor)
        ```
    """
    _require_category("make_tensor_like", "like", like, "tensor", 1)
    _require_frontend_state("make_tensor_like")
    ptr_value = _as_value(ptr)
    like_value = _as_value(like)
    try:
        like_type = _tla_tensor_type_for_mlir_value(like_value)
    except TlaLoweringError as exc:
        raise TlaLoweringError(
            "tla.make_tensor_like expects ``like`` to carry structured Tla tensor metadata; "
            f"got {str(like_value.type)!r}"
        ) from exc
    if not PtrType.isinstance(ptr_value.type):
        _op_error(
            "make_tensor_like",
            f"invalid argument 'ptr': expected !tla.ptr, got {ptr_value.type}",
        )
    # Keep frontend MLIR pointer spelling aligned with the pointer operand.
    ptr_ty = PtrType(ptr_value.type)
    try:
        dtype = _dtype_to_str(ptr_ty.pointee)
    except TypeError as exc:
        raise TlaLoweringError(
            "tla.make_tensor_like cannot derive element type from ptr pointee "
            f"{ptr_ty.pointee}"
        ) from exc
    if dtype not in _MAKE_TENSOR_SUPPORTED_ELEMENT_TYPES:
        raise TlaLoweringError(
            f"tla.make_tensor_like expects a supported element type, got [{dtype}]"
        )
    addr = ptr_ty.addrspace
    if addr not in _MAKE_TENSOR_LIKE_ON_CHIP_ADDRSPACES:
        _op_error(
            "make_tensor_like",
            "invalid argument 'ptr': expected an on-chip address space "
            "(l1, l0a, l0b, l0c, ub), "
            f"got '{addr}'",
        )

    # Infer layout if not provided
    if layoutTag is None:
        if addr == "l0a":
            layout = "zN"
        elif addr == "l0b":
            layout = "nZ"
        elif addr == "l0c":
            layout = "L0Clayout"
        elif addr == "l1":
            if like_type.layout_tag in ("RowMajor", "zN"):
                layout = "zN"
            elif like_type.layout_tag in ("ColumnMajor", "nZ"):
                layout = "nZ"
            else:
                raise TlaLoweringError(
                    f"tla.make_tensor_like cannot infer layout for addrspace l1 "
                    f"with likeTensor layoutTag '{like_type.layout_tag}'; please specify layoutTag explicitly"
                )
        else:
            raise TlaLoweringError(
                f"tla.make_tensor_like cannot infer layout for addrspace '{addr}'; "
                f"please specify layoutTag explicitly"
            )
    else:
        # Validate user-provided layoutTag
        if not isinstance(layoutTag, _LayoutTag):
            _op_error(
                "make_tensor_like",
                "invalid argument 'layoutTag': "
                "expected a tla.arch layout sentinel (e.g. tla.arch.zN) or None; "
                f"got {_type_name(layoutTag)}",
            )
        layout = _name_token(layoutTag)
        if layout is None:
            _op_error(
                "make_tensor_like",
                "invalid argument 'layoutTag': "
                f"expected a tla.arch layout sentinel with a token name; got {_type_name(layoutTag)}",
            )

    ptr_alignment = ptr_ty.alignment
    remapped = _remap_tensor_like_prefix_fields_for_layout_trees(
        like_type.origin_shape,
        dtype,
        layout,
        linear_stride_alignment_bytes=_CATLASS_BYTE_PER_C0,
    )
    if remapped is not None:
        shape = remapped.shape
        stride = remapped.stride
        coord = remapped.coord
        origin = remapped.origin_shape
    else:
        shape = like_type.shape
        stride = like_type.stride
        coord = like_type.coord
        origin = like_type.origin_shape
    result_desc = TlaTensorTypeDescriptor(
        layout=TlaLayoutDescriptor(
            shape=TlaIndexTreeType("shape", shape),
            stride=TlaIndexTreeType("stride", stride),
            origin_shape=TlaIndexTreeType("shape", origin),
            layout_tag=layout,
        ),
        coord=coord,
        element_type=dtype,
        addrspace=addr,
        ptr_alignment=ptr_alignment,
    )
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    layout_tag_attr = mlir_ir.Attribute.parse(f"#tla.layout_tag<{layout}>", ctx)
    op = mlir_ir.Operation.create(
        "tla.make_tensor_like",
        operands=[ptr_value, like_value],
        results=[_coerce_type(result_desc)],
        attributes={"layoutTag": layout_tag_attr},
        loc=loc,
    )
    out = op.results[0]
    _register_tla_tensor_type(out, result_desc)
    try:
        like_meta = {
            "shape": _tensor_metadata_field(like_value, "shape"),
            "stride": _tensor_metadata_field(like_value, "stride"),
            "coord": _tensor_metadata_field(like_value, "coord"),
            "origin_shape": _tensor_metadata_field(like_value, "origin_shape"),
        }
        metadata = {
            "shape": like_meta["shape"],
            "stride": like_meta["stride"],
            "coord": like_meta["coord"],
            "origin_shape": like_meta["origin_shape"],
            "dtype": dtype,
            "addrspace": addr,
            "layout_tag": layout,
        }
        if remapped is not None:
            remapped_trees = _materialize_layout_trees_from_origin(
                like_meta["origin_shape"], dtype, layout
            )
            if remapped_trees is not None:
                metadata["shape"] = remapped_trees[0]
                metadata["stride"] = remapped_trees[1]
                metadata["coord"] = remapped_trees[2]
                metadata["origin_shape"] = remapped_trees[3]
            else:
                metadata = result_desc.metadata()
        _register_tla_tensor_metadata(out, metadata)
    except Exception:
        pass
    return _Tensor(out)


# (src addrspace, dst addrspace) routes and the region each must be nested in.
_COPY_CUBE_ROUTES = {
    ("gm", "l1"),
    ("l1", "l0a"),
    ("l1", "l0b"),
    ("l0c", "gm"),
    ("l0c", "l1"),
    ("l0c", "ub"),
    ("l1", "ub"),
}
_COPY_VECTOR_ROUTES = {("gm", "ub"), ("ub", "gm"), ("ub", "l1")}

# fmt: off
# Cube copy
_SUPPORTED_CUBE_COPY_ROUTES = {
    # (src_addr, src_layout, dst_addr, dst_layout) : dtypes
    # GM->L1
    ("gm", "RowMajor", "l1", "zN"): ("f32", "f16", "bf16", "i8", "f8e4m3fn", "f8e5m2", "f4e2m1", "f4e1m2"),
    ("gm", "ColumnMajor", "l1", "nZ"): ("f32", "f16", "bf16", "i8", "f8e4m3fn", "f8e5m2", "f4e2m1", "f4e1m2"),
    ("gm", "RowMajorMxScaleA", "l1", "zZMxScale"): ("f8e8m0",),
    ("gm", "ColMajorMxScaleA", "l1", "zZMxScale"): ("f8e8m0",),
    ("gm", "RowMajorMxScaleB", "l1", "nNMxScale"): ("f8e8m0",),
    ("gm", "ColMajorMxScaleB", "l1", "nNMxScale"): ("f8e8m0",),
    # L1->L0
    ("l1", "zN", "l0a", "zN"): ("f32", "f16", "bf16", "i8", "f8e4m3fn", "f8e5m2"),
    ("l1", "nZ", "l0a", "zN"): ("f32", "f16", "bf16", "i8", "f8e4m3fn", "f8e5m2"),
    ("l1", "zN", "l0b", "nZ"): ("f32", "f16", "bf16", "i8", "f8e4m3fn", "f8e5m2"),
    ("l1", "nZ", "l0b", "nZ"): ("f32", "f16", "bf16", "i8", "f8e4m3fn", "f8e5m2"),
}

# Cube copy that carries a scale
_SUPPORTED_CUBE_MXFP_WITH_SCALE_COPY_ROUTES = {
    # (src_addr, src_layout, dst_addr, dst_layout, scale_addr, scale_layout) : dtypes
    ("l1", "zN", "l0a", "zN", "l1", "zZMxScale"): ("f8e4m3fn", "f8e5m2", "f4e2m1", "f4e1m2"),
    ("l1", "nZ", "l0a", "zN", "l1", "zZMxScale"): ("f8e4m3fn", "f8e5m2", "f4e2m1", "f4e1m2"),
    ("l1", "zN", "l0b", "nZ", "l1", "nNMxScale"): ("f8e4m3fn", "f8e5m2", "f4e2m1", "f4e1m2"),
    ("l1", "nZ", "l0b", "nZ", "l1", "nNMxScale"): ("f8e4m3fn", "f8e5m2", "f4e2m1", "f4e1m2"),
}

# Vector copy
_SUPPORTED_VECTOR_COPY_ROUTES = {
    # (src_addr, src_layout, dst_addr, dst_layout) : dtypes
    ("gm", "RowMajor", "ub", "RowMajor"): ("f32", "f16", "bf16", "i32", "i16", "i8", "f8e4m3fn", "f8e5m2"),
    ("ub", "RowMajor", "gm", "RowMajor"): ("f32", "f16", "bf16", "i32", "i16", "i8", "f8e4m3fn", "f8e5m2"),
    ("ub", "RowMajor", "l1", "zN"): ("f32", "f16", "bf16", "i8", "f8e4m3fn", "f8e5m2"),
    ("ub", "zN", "l1", "zN"): ("f32", "f16", "bf16", "i8", "f8e4m3fn", "f8e5m2"),
    ("ub", "zNUnAlign", "l1", "zN"): ("f32", "f16", "bf16", "i8", "f8e4m3fn", "f8e5m2"),
}

# Cube Copy Fixpipe
_SUPPORTED_CUBE_FIXPIPE_ROUTES = {
    # (dst_addr, dst_layout, split_mode, quant_mode, relu) : {src_dtype: dst_dtypes}
    ("gm", "RowMajor", "-", "NO_QUANT", False): {"f32": ("f32", "f16", "bf16"), "i32": ("i32",)},
    ("gm", "ColumnMajor", "-", "NO_QUANT", False): {"f32": ("f32", "f16", "bf16"), "i32": ("i32",)},
    ("l1", "zN",  "-", "NO_QUANT", False): {"f32": ("f32", "f16", "bf16"), "i32": ("i32",)},

    ("ub", "RowMajor", "nosplit", "NO_QUANT", False): {"f32": ("f32", "f16", "bf16"), "i32": ("i32",)},
    ("ub", "RowMajor", "split", "NO_QUANT", False): {"f32": ("f32",), "i32": ("i32",)},
    ("ub", "ColumnMajor", "nosplit", "NO_QUANT", False): {"f32": ("f32", "f16", "bf16"), "i32": ("i32",)},

    # ReLU rides on the fixpipe transfer itself, so it exists only where a
    # no-split variant is registered: split_M / split_N have no ReLU symbol.
    ("gm", "RowMajor", "-", "NO_QUANT", True): {"f32": ("f32", "f16", "bf16"), "i32": ("i32",)},
    ("gm", "ColumnMajor", "-", "NO_QUANT", True): {"f32": ("f32", "f16", "bf16"), "i32": ("i32",)},
    ("l1", "zN",  "-", "NO_QUANT", True): {"f32": ("f32", "f16", "bf16"), "i32": ("i32",)},

    ("ub", "RowMajor", "nosplit", "NO_QUANT", True): {"f32": ("f32", "f16", "bf16"), "i32": ("i32",)},
    ("ub", "ColumnMajor", "nosplit", "NO_QUANT", True): {"f32": ("f32", "f16", "bf16"), "i32": ("i32",)},
}
# fmt: on


def _validate_supported_copy(
    src_addr,
    src_layout,
    dst_addr,
    dst_layout,
    dtype,
    scale_addr=None,
    scale_layout=None,
) -> bool:
    if scale_addr is not None or scale_layout is not None:
        key = (src_addr, src_layout, dst_addr, dst_layout, scale_addr, scale_layout)
        table = _SUPPORTED_CUBE_MXFP_WITH_SCALE_COPY_ROUTES
        if key in table:
            return dtype in table[key]
    else:
        key = (src_addr, src_layout, dst_addr, dst_layout)
        if key in _SUPPORTED_CUBE_COPY_ROUTES:
            return dtype in _SUPPORTED_CUBE_COPY_ROUTES[key]
        elif key in _SUPPORTED_VECTOR_COPY_ROUTES:
            return dtype in _SUPPORTED_VECTOR_COPY_ROUTES[key]
    return False


def _validate_supported_fixpipe(
    src_addr,
    src_layout,
    dst_addr,
    dst_layout,
    src_dtype,
    dst_dtype,
    params: CopyL0C2DstParams,
) -> bool:
    if src_addr != "l0c" or src_layout != "L0Clayout":
        _op_error("copy", f"L0C layout_tag must be L0Clayout, got {src_layout}")
    if src_dtype not in ("f32", "i32"):
        _op_error("copy", f"l0c dtype only support [f32, i32], got {src_dtype}")
    if not isinstance(params, CopyL0C2DstParams):
        _op_error(
            "copy",
            f"operand `params` expects to be a CopyL0C2DstParams when route is [{src_addr, dst_addr}]",
        )
    if dst_addr != "ub" and params.l0c2ub_mode != L0C2UBMode.NO_SPLIT_VEC_0:
        _op_error("copy", "fixpipe to gm/l1 split_mode should be `NO_SPLIT_VEC_0`")

    if dst_addr == "ub":
        is_split = params.l0c2ub_mode in (L0C2UBMode.SPLIT_M, L0C2UBMode.SPLIT_N)
        split_mode = "split" if is_split else "nosplit"
    else:
        split_mode = "-"

    key = (
        dst_addr,
        dst_layout,
        split_mode,
        str(params.quant_mode),
        params.relu_enable,
    )
    if key in _SUPPORTED_CUBE_FIXPIPE_ROUTES:
        if src_dtype in _SUPPORTED_CUBE_FIXPIPE_ROUTES[key]:
            return dst_dtype in _SUPPORTED_CUBE_FIXPIPE_ROUTES[key][src_dtype]
    return False


def _modify_doc_copy(fn: Callable[..., Any]) -> None:
    """Append the supported copy route table to copy doc."""

    def _stringify_types(types):
        return ",".join([f"`{t}`" for t in types])

    cube_list = [
        "\n",
        "Supported cube copy routes without scale, `tla.copy(dst, src, scale=None)`\n",
        "|`src_addr`|`src_layout`|`dst_addr`|`dst_layout`|`dtypes`|",
        "|----------|------------|----------|------------|--------|",
    ]
    for (
        src_addr,
        src_layout,
        dst_addr,
        dst_layout,
    ), dtypes in _SUPPORTED_CUBE_COPY_ROUTES.items():
        cube_list.append(
            f"|`{src_addr}`|`{src_layout}`|`{dst_addr}`|`{dst_layout}`|{_stringify_types(dtypes)}|"
        )

    cube_scale_list = [
        "\n",
        "Supported cube copy routes carrying a scale (tla.mmad_mx),",
        "i.e. `tla.copy(dst=l0a|l0b, src=l1, scale=scale)`, the scale only support `f8e8m0`.",
        "and the scale `l1->l0` don't support transpose.\n",
        "|`src_addr`|`src_layout`|`dst_addr`|`dst_layout`|`scale_addr`|`scale_layout`|`dtypes`|",
        "|----------|------------|----------|------------|------------|--------------|--------|",
    ]
    for (
        src_addr,
        src_layout,
        dst_addr,
        dst_layout,
        scale_addr,
        scale_layout,
    ), dtypes in _SUPPORTED_CUBE_MXFP_WITH_SCALE_COPY_ROUTES.items():
        cube_scale_list.append(
            f"|`{src_addr}`|`{src_layout}`|`{dst_addr}`|`{dst_layout}`|`{scale_addr}`|`{scale_layout}`|{_stringify_types(dtypes)}|"
        )

    def _stringify_type_map(type_map):
        items = [
            f"`{src_dtype}`: {_stringify_types(dst_dtypes)}"
            for src_dtype, dst_dtypes in type_map.items()
        ]
        return "<br>".join(items)

    cube_fixpipe_list = [
        "\n",
        "Supported cube fixpipe routes, copy from l0c->gm/l1/ub.",
        "The fixpipe support on-the-fly quant and relu, quant mode support `NO_QUANT`, `PER_TENSOR`, `PER_CHANNEL`.",
        "The fixpipe has these rules:",
        "- the fixpipe chain is (l0c acc) -> quant(optional) -> relu(optional) -> copy out to dst.",
        "- `NO_QUANT` mode includes pure data cast, `f32`->`f16`, `f32`->`bf16`",
        "- If the dst is ub, there's an extra `split_mode`, support nosplit(`NO_SPLIT_VEC_0`, `NO_SPLIT_VEC_1`), and split(`SPLIT_M`, `SPLIT_N`)",
        "- If dst is (ub, ColumnMajor), only support `nosplit` mode",
        "- In to ub is `split` mode, any quant_mode/relu is disabled, neither pure cast, so only `f32`->`f32` and `i32`->`i32` is enabled\n",
        "|`dst`|`dst_layout`|`split_mode`|`quant_mode`|`relu`|`src_dtype`:`dst_dtype`|",
        "|-----|------------|------------|------------|------|-----------------------|",
    ]
    for (
        dst_addr,
        dst_layout,
        split_mode,
        quant_mode,
        relu_enable,
    ), dtype_map in _SUPPORTED_CUBE_FIXPIPE_ROUTES.items():
        cube_fixpipe_list.append(
            f"|`{dst_addr}`|`{dst_layout}`|`{split_mode}`|`{quant_mode}`|`{relu_enable}`"
            f"|{_stringify_type_map(dtype_map)}|"
        )

    vec_list = [
        "\n",
        "Supported vector copy routes\n",
        "|`src_addr`|`src_layout`|`dst_addr`|`dst_layout`|`dtypes`|",
        "|----------|------------|----------|------------|--------|",
    ]
    for (
        src_addr,
        src_layout,
        dst_addr,
        dst_layout,
    ), dtypes in _SUPPORTED_VECTOR_COPY_ROUTES.items():
        vec_list.append(
            f"|`{src_addr}`|`{src_layout}`|`{dst_addr}`|`{dst_layout}`|{_stringify_types(dtypes)}|"
        )
    fn.__doc__ += "\n".join(
        [*cube_list, *cube_scale_list, *cube_fixpipe_list, *vec_list]
    )


# tla.fill (Tensor.fill) element rules: only the packed fp4/fp8 cube operand
# formats are supported, and only to zero the tile (the MX pad use case) --
# wider types and non-zero patterns are refused up front.
_FILL_SUPPORTED_ELEMENT_TYPES = frozenset({"f4e2m1", "f4e1m2", "f8e4m3fn", "f8e5m2"})


def _fill_value_bits(op_name: str, element_type: str, value: Any) -> int:
    """Convert a trace-time scalar to the i32 bit pattern consumed by tla.fill."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _op_error(
            op_name,
            f"fill value must be a trace-time int or float, got {type(value).__name__}",
        )
    if value != 0:
        _op_error(
            op_name,
            f"fill of {element_type} supports only value 0, got {value!r}",
        )
    return 0


def _tensor_fill_impl(
    tensor: Tensor,
    value: int | float,
    *,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Implementation behind ``Tensor.fill``; not exported as a free function.

    The fill region is the tile itself: the emitted coord is the tensor's own
    coord and the extent is its ``origin_shape`` -- no caller-side region, no
    cropping. Sub-regions are expressed by tiling first (``tla.get_tile``).
    """
    op_name = "fill"
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region("fill", "cube")

    dest = _as_value(tensor)
    dest_desc = _tla_tensor_type_for_mlir_value(dest)
    if dest_desc.addrspace.lower() != "l1":
        _op_error(
            op_name,
            "invalid argument 'tensor' (position 0): expected addrspace l1, "
            f"got {dest_desc.addrspace}",
        )
    layout_tag = dest_desc.layout_tag
    if layout_tag not in ("zN", "nZ"):
        _op_error(
            op_name,
            "invalid argument 'tensor' (position 0): expected layout tag zN or nZ, "
            f"got {layout_tag!r}",
        )
    element_type = str(dest_desc.element_type).strip().lower()
    if element_type not in _FILL_SUPPORTED_ELEMENT_TYPES:
        _op_error(
            op_name,
            f"unsupported element type {element_type!r}: fill supports only the "
            "packed fp4/fp8 operand formats (f4e2m1/f4e1m2/f8e4m3fn/f8e5m2), "
            "zero only",
        )

    # The region is the tile itself. bc places the fill relative to the buffer
    # base without adding the descriptor coord, so the tensor's own coord must
    # travel in the emitted coord.
    coord_tree = tensor.coord
    origin_tree = tensor.origin_shape

    value_bits = _fill_value_bits(op_name, element_type, value)
    value_val = _const_i32(value_bits, loc=loc)
    coord_val = _pack_coord(tuple(coord_tree), loc=loc)
    shape_val = _pack_shape(tuple(origin_tree), loc=loc)
    _tla_ops_gen.fill(dest, value_val, coord_val, shape_val, loc=loc)


@dsl_user_op
def copy(
    dst: Tensor,
    src: Tensor,
    params: CopyParams | None = None,
    *,
    scale: Tensor | None = None,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Directory: Data Movement
    Description:
        Copy data between tiles. The hardware path follows `src`/`dst` address spaces
        (vector: GM↔UB, UB→L1; cube: GM→L1, L1→L0A/L0B, L0C→GM|UB|L1).
        Layout tags on the tiles select format conversion (for example ND→zN).

        Copy / tiling sizes follow each tile's logical `origin_shape`
        (not the nested physical `shape`). Physical `shape` / `stride` describe
        how those logical elements are stored (alignment fill, zN packing, …).

        Parameters:
        - `dst` (`Tensor`): Destination tile. Required.
        - `src` (`Tensor`): Source tile. Required.
        - `params` (`CopyParams | None`): Optional path-specific params
          (`CopyL0C2DstParams`, `CopyUbToGmParams` / atomic, …). Default `None`.
        - **MX scale tiles** (`zZMxScale` / `nNMxScale` destinations) reach L1 as a
          flat run of bytes, so the source tile must span full rows of its buffer.
          Stack per-chunk scale blocks along rows; do not slice a wide scale buffer
          along columns. This is checked only when the row pitch is statically
          known -- a layout-dynamic tensor cannot be checked at compile time.
        - `scale` (`Tensor | None`): L1 tile of OCP `e8m0` shared exponents, one per
          32 elements along K, which turns an L1→L0A/L0B load into a microscaling
          (MX) load and lowers to `tla.copy_mx`. Carried as `i8`/`u8` storage because
          MLIR has no e8m0 type, and tagged `tla.arch.zZMxScale` for an L0A
          destination or `tla.arch.nNMxScale` for an L0B destination. Default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - `scale` is only valid on the L1→L0A/L0B route, and the tile it loads must
          then be consumed by `tla.mmad`, which becomes a microscaling matmul
          because of this load. The scale is consumed by the load, not by the matmul:
          the hardware's `mad_mx` takes no scale operand, so the MX association is
          established when the operand lands in L0. This mirrors AscendC, where the
          MX load is the five-argument overload of `LoadData` rather than a separate
          entry point.
        - Must be called inside `tla.cube()` or `tla.vector()` (cube routes above
          in `cube()`, vector routes in `vector()`).
        - Whole-tile DMA uses `tla.copy`. Register-level UB unaligned access uses
          `tensor.load` / `tensor.store` with `UnalignLoadParams` /
          `UnalignStoreParams` instead.
        - **Shapes not aligned to 32 bytes:** C0 is 32 bytes. For ND GM↔UB, prefer
          a leading dimension whose byte size is a multiple of 32 (e.g. for
          RowMajor f16, choose `N` so `N % 16 == 0`). If the true data size cannot
          meet that for DMA, keep `origin_shape` as the real logical size and
          either enlarge the physical layout to the aligned size, or use the
          unaligned register load/store path above. For zN when `M` is not a
          multiple of 16, use `tla.arch.zNUnAlign` instead of `zN`.

        Example:
        ```python
        # --- 1) Aligned GM ↔ UB (RowMajor, 32B-friendly) ---
        # Before: x_gm[i, j] holds ND data; x_ub is empty. origin_shape==(M, N).
        with tla.vector():
            tla.copy(dst=x_ub, src=x_gm)
            # After: x_ub[i, j] == x_gm[i, j] for all logical (i, j) in origin_shape.
            tla.copy(dst=y_gm, src=y_ub)

        # --- 2) ND → zN: GM RowMajor → L1 zN (layout change) ---
        # Before (logical): gm_a.origin_shape==(128, 64), RowMajor; element (r, c)
        #   sits at ND offset r*64+c.
        # After (physical on L1): l1_a uses nested zN shape/stride; the same
        #   logical (r, c) is zN-packed. l1_a.origin_shape stays (128, 64).
        l1_a = tla.make_tensor_like(l1_ptr, gm_a, layoutTag=tla.arch.zN)
        with tla.cube():
            tla.copy(dst=l1_a, src=gm_a)

        # Explicit zN (same logical 128x64 f16) if not using make_tensor_like:
        l1_a = tla.make_tensor(
            l1_ptr,
            tla.make_layout(
                tla.make_shape((16, 8), (16, 4)),
                tla.make_stride((16, 256), (1, 2048)),
                origin_shape=tla.make_shape(128, 64),
                layoutTag=tla.arch.zN,
            ),
        )

        # --- 3) M not a multiple of 16: zNUnAlign ---
        # Before: rows may be runtime and not a multiple of 16.
        l1_unalign = tla.make_tensor_like(l1_ptr, gm_tile, layoutTag=tla.arch.zNUnAlign)
        with tla.cube():
            tla.copy(dst=l1_unalign, src=gm_tile)

        # Related (register path, not tla.copy): unaligned UB ↔ vector register
        #   with tla.vec.func(mode="simd"):
        #       x_reg = x_ub.load(tla.params.UnalignLoadParams())
        #       y_ub.store(y_reg, tla.params.UnalignStoreParams())
        ```
    """
    _require_category("copy", "dst", dst, "tensor", 0)
    _require_category("copy", "src", src, "tensor", 1)
    _require_frontend_state("copy")
    dst_value = _as_value(dst)
    src_value = _as_value(src)
    src_desc = _tla_tensor_type_for_mlir_value(src_value)
    dst_desc = _tla_tensor_type_for_mlir_value(dst_value)

    # Cube data-path copies (GM->L1, L1->L0A/L0B, L0C->GM, L0C->UB) must
    # live in a tla.cube region; vector staging copies (GM<->UB, UB->L1) must
    # live in a tla.vector region. Mirrors tla.copy's MLIR verifier. Reading .addrspace
    # needs registered tensor metadata, which is unavailable for values carried
    # through scf.if/scf.for; when it can't be resolved, skip the frontend check
    # and let the MLIR verifier enforce placement.
    _route = (src_desc.addrspace.lower(), dst_desc.addrspace.lower())

    if _route in _COPY_CUBE_ROUTES:
        _runtime._require_enclosing_region("copy", "cube")
    elif _route in _COPY_VECTOR_ROUTES:
        _runtime._require_enclosing_region("copy", "vector")
    else:
        _op_error("copy", f"Hardware unsupported copy route {_route}")

    # Read dtype/addrspace from MLIR descriptors; runtime argument proxies do
    # not expose Python .dtype / .addrspace attributes.
    src_dtype = src_desc.element_type.lower()
    dst_dtype = dst_desc.element_type.lower()
    # GM scale buffers enter through from_dlpack as tla.Float8E8M0, so both
    # sides of an MX scale route name their own format and the dtype check is
    # plain equality, like every other copy route.
    same_dtype = src_dtype == dst_dtype
    src_layout = src_desc.layout_tag
    dst_layout = dst_desc.layout_tag

    if _route[0] == "l0c":
        if params is None:
            params = CopyL0C2DstParams()  # use default
        params._validate()
        fixpipe_route = (
            _route[0],
            src_layout,
            _route[1],
            dst_layout,
            src_dtype,
            dst_dtype,
            params,
        )
        if not _validate_supported_fixpipe(*fixpipe_route):
            _op_error(
                "copy",
                "unsupported fixpipe route "
                f"{_route[0]},{src_layout} -> {_route[1]},{dst_layout} "
                f"{src_dtype}, params={params}",
            )

        ctx = loc.context if loc is not None else mlir_ir.Context.current
        quant_mode_attr = mlir_ir.Attribute.parse(
            f"#tla.quant_mode<{params.quant_mode}>", context=ctx
        )
        l0c2ub_mode_attr = mlir_ir.Attribute.parse(
            f"#tla.l0c2ub_mode<{params.l0c2ub_mode}>", context=ctx
        )
        quant_scale_or_tensor = None
        if params.quant_mode == QuantMode.PER_TENSOR:
            quant_scale_or_tensor = _const_f32(params.quant_scale)
        elif params.quant_mode == QuantMode.PER_CHANNEL:
            quant_scale_or_tensor = _as_value(params.quant_tensor)
        params_value = _tla_ops_gen.CopyL0C2DstParams(
            _tla_type_bridge.copy_l0c2dst_params_type_get(ctx),
            params.unit_flag,
            params.relu_enable,
            quant_mode_attr,
            l0c2ub_mode_attr,
            quant_scale_or_tensor=quant_scale_or_tensor,
        )
    else:
        if not same_dtype:
            _op_error(
                "copy",
                "Except the fixpipe pipe, other copy src and dst dtype "
                f"must be same, got {src_dtype} and {dst_dtype}",
            )
        if scale is None:
            copy_route = (_route[0], src_layout, _route[1], dst_layout, src_dtype)
            # fp4 l1->l0a/l0b will be checked in the `if scale is not None` bellow
            if _route[1] in ("l0a", "l0b") and src_dtype in _FP4_FORMATS:
                _op_error(
                    "copy",
                    "packed fp4 to l0a/l0b must carry scale, tla.copy(l0, l1, scale=...)",
                )
            elif not _validate_supported_copy(*copy_route):
                _op_error(
                    "copy",
                    "unsupported copy route "
                    f"{_route[0]},{src_layout} -> {_route[1]},{dst_layout} {src_dtype}",
                )
        params_value = None

    # Check if atomic mode enabled and acquire lowered atomic_mode_attr
    atomic_mode_attr = None
    atomic_mode = params.atomic_mode if params is not None else None
    if atomic_mode is not None and atomic_mode != AtomicMode.NONE:
        if atomic_mode != AtomicMode.ADD:
            raise NotImplementedError(
                f"currently unsupported atomic mode {str(atomic_mode)}"
            )

        if _route[1] != "gm":
            raise TlaLoweringError(
                f"When atomic operation is enabled, the dst location should only be GM but got {_route[1]}"
            )

        dst_dtype = (
            str(_tla_tensor_type_for_mlir_value(dst_value).element_type).strip().lower()
        )
        if dst_dtype not in ("f32", "f16", "i16", "i32", "i8", "bf16"):
            raise TlaLoweringError(
                "The supported atomic operation's data type includes f32, f16, i16, i32, i8, bf16, "
                f"the data type {dst_dtype} is not supported"
            )

        ctx = loc.context if loc is not None else mlir_ir.Context.current
        atomic_mode_attr = mlir_ir.Attribute.parse(
            f"#tla.atomic_mode<{atomic_mode.value}>", context=ctx
        )

    copy_attrs: dict[str, Any] = {}

    # A scale operand turns this into the MX L1 -> L0A/L0B load. Only that route
    # can carry one, and the scale's storage and tag must match the destination
    # side: A-side scale is zZMxScale, B-side nNMxScale.
    if scale is not None:
        _require_category("copy", "scale", scale, "tensor", 2)
        scale_value = _as_value(scale)
        scale_desc = _tla_tensor_type_for_mlir_value(scale_value)

        # The MX table pins the address spaces, the layouts and the data dtype
        # (packed fp4 only appears there), and `same_dtype` above already holds
        # both sides to one element type. What is left is the scale: an L1 tile
        # whose tag matches the destination side -- A-side scales are zZ,
        # B-side nN.
        if not _validate_supported_copy(
            src_desc.addrspace,
            src_desc.layout_tag,
            dst_desc.addrspace,
            dst_desc.layout_tag,
            src_desc.element_type,
            scale_desc.addrspace,
            scale_desc.layout_tag,
        ):
            _op_error(
                "copy",
                "unsupported MX load "
                f"{src_desc.addrspace}({src_desc.layout_tag}, "
                f"{src_desc.element_type}) -> "
                f"{dst_desc.addrspace}({dst_desc.layout_tag}) with scale "
                f"{scale_desc.addrspace}({scale_desc.layout_tag}); expected an "
                "l1 source of f8e4m3fn, f8e5m2, f4e2m1 or f4e1m2 laid out zN "
                "or nZ, loaded into l0a(zN) or l0b(nZ), with the l1 scale "
                "tile tagged zZMxScale for l0a or nNMxScale for l0b",
            )
        if src_desc.element_type != dst_desc.element_type:
            _op_error("copy", "src and dst element types must match")
        if scale_desc.element_type != "f8e8m0":
            _op_error("copy", "scale must be an f8e8m0 tile")
        # Remember that this L0 tile carries a scale block. Nothing in the tile's
        # type records it, so this is the only thing that can tell tla.mmad and
        # tla.mmad_mx apart later.
        _runtime._current_frontend_state().mx_scaled_l0_values.add(dst_value)
        # An MX load is its own op in the dialect.
        return _tla_ops_gen.copy_mx(dst_value, src_value, scale_value, loc=loc)

    return _tla_ops_gen.copy(
        dst_value,
        src_value,
        params=params_value,
        loc=loc,
        atomic_mode=atomic_mode_attr,
        **copy_attrs,
    )


copy.__modify_doc__ = _modify_doc_copy


@dsl_user_op
def flag(
    name: str,
    src_pipe: PipeLike,
    dst_pipe: PipeLike,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaFlag:
    """Directory: Sync Control
    Description:
        Create an in-pipe synchronization flag between two pipes.

        Parameters:
        - `name` (`str`): In-pipe flag name. Required.
        - `src_pipe` (`PipeLike`): Source pipe id (e.g. `tla.arch.MTE2`). Required.
        - `dst_pipe` (`PipeLike`): Destination pipe id (e.g. `tla.arch.VECTOR`). Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Creates a flag handle; `set_flag`/`wait_flag` must be paired inside cube/vector regions.

        Example:
        ```python
        # MTE2 finishes a GM→UB copy, then VECTOR may consume the UB tile.
        ub_loaded = tla.flag("ub_loaded", src_pipe=tla.arch.MTE2, dst_pipe=tla.arch.VECTOR)
        with tla.vector():
            tla.copy(dst=x_ub, src=x_gm)
            tla.set_flag(ub_loaded)   # after copy: mark UB data ready
            tla.wait_flag(ub_loaded)  # before compute: wait until ready
        ```
    """
    if not isinstance(name, str):
        _op_error(
            "flag",
            f"invalid argument 'name' (position 0): expected str, got {_type_name(name)}",
        )
    _require_pipe("flag", "src_pipe", src_pipe, 1)
    _require_pipe("flag", "dst_pipe", dst_pipe, 2)
    _require_frontend_state("flag")
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    src_value = str(_token(src_pipe)).lower()
    dst_value = str(_token(dst_pipe)).lower()
    src_attr = mlir_ir.Attribute.parse(f"#tla.pipe<{src_value}>", context=ctx)
    dst_attr = mlir_ir.Attribute.parse(f"#tla.pipe<{dst_value}>", context=ctx)
    return _tla_ops_gen.flag(
        _tla_type_bridge.flag_type_get(ctx),
        name,
        src_attr,
        dst_attr,
        loc=loc,
    )


@dsl_user_op
def cross_flag(
    name: str,
    *,
    mode: int = 2,
    loc: mlir_ir.Location | None = None,
) -> TlaCrossFlag:
    """Directory: Sync Control

    Description:
        Create a cross-core synchronization flag.

        Parameters:
        - `name` (`str`): Cross-core flag name. Required.
        - `mode` (`int`): Cross-core sync mode. Optional, default `2`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - `mode` supports only 0/1/2/4; source/destination pipes are specified by the matching set/wait.

        Example:
        ```python
        cf = tla.cross_flag("aic_aiv", mode=2)
        ```
    """
    if not isinstance(name, str):
        _op_error(
            "cross_flag",
            f"invalid argument 'name' (position 0): expected str, got {_type_name(name)}",
        )
    if not isinstance(mode, int) or isinstance(mode, bool) or mode not in (0, 1, 2, 4):
        _op_error(
            "cross_flag",
            f"invalid argument 'mode': expected one of 0, 1, 2, or 4, got {mode!r}",
        )
    _require_frontend_state("cross_flag")
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    return _tla_ops_gen.cross_flag(
        _tla_type_bridge.cross_flag_type_get(ctx, mode),
        name,
        loc=loc,
    )


def _cross_flag_aiv_id_attr(
    op_name: str,
    cross_flag_value: CrossFlagLike,
    aiv_id: int | None,
    *,
    loc: mlir_ir.Location | None,
) -> mlir_ir.IntegerAttr | None:
    mode = _tla_type_bridge.cross_flag_mode(_as_value(cross_flag_value).type)
    if mode == 4:
        if (
            not isinstance(aiv_id, int)
            or isinstance(aiv_id, bool)
            or aiv_id not in (0, 1)
        ):
            _op_error(
                op_name,
                "invalid argument 'aiv_id': mode 4 requires compile-time 0 or 1, "
                f"got {aiv_id!r}",
            )
    elif aiv_id is not None:
        _op_error(
            op_name,
            f"invalid argument 'aiv_id': mode {mode} requires None, got {aiv_id!r}",
        )
    if aiv_id is None:
        return None
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    return mlir_ir.IntegerAttr.get(
        mlir_ir.IntegerType.get_signless(64, context=ctx), aiv_id
    )


@dsl_user_op
def cross_core_set_flag(
    cross_flag_value: CrossFlagLike,
    pipe: PipeLike,
    aiv_id: int | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Directory: Sync Control

    Description:
        Set a cross-core flag on the given pipe.

        Parameters:
        - `cross_flag_value` (`CrossFlagLike`): Cross-core flag from `tla.cross_flag`. Required.
        - `pipe` (`PipeLike`): Pipe that issues the set. Required.
        - `aiv_id` (`int | None`): Target AIV id; omit for broadcast/default routing. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()` or `tla.vector()`; when `mode=4`, `aiv_id` must be 0 or 1.

        Example:
        ```python
        with tla.cube():
            tla.cross_core_set_flag(cf, tla.arch.CUBE)
            # mode=4: tla.cross_core_set_flag(cf, tla.arch.CUBE, aiv_id=0)
        ```
    """
    _require_category("cross_core_set_flag", "flag", cross_flag_value, "cross_flag", 0)
    _require_pipe("cross_core_set_flag", "pipe", pipe, 1)
    _require_frontend_state("cross_core_set_flag")
    _runtime._require_enclosing_cube_or_vector("cross_core_set_flag")
    aiv_id_attr = _cross_flag_aiv_id_attr(
        "cross_core_set_flag", cross_flag_value, aiv_id, loc=loc
    )
    return _tla_ops_gen.cross_core_set_flag(
        _as_value(cross_flag_value),
        _pipe_attr_from_token(pipe, loc=loc),
        aiv_id=aiv_id_attr,
        loc=loc,
    )


@dsl_user_op
def cross_core_wait_flag(
    cross_flag_value: CrossFlagLike,
    pipe: PipeLike,
    aiv_id: int | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Directory: Sync Control

    Description:
        Wait on a cross-core flag on the given pipe.

        Parameters:
        - `cross_flag_value` (`CrossFlagLike`): Cross-core flag from `tla.cross_flag`. Required.
        - `pipe` (`PipeLike`): Pipe that performs the wait. Required.
        - `aiv_id` (`int | None`): Target AIV id; omit for broadcast/default routing. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()` or `tla.vector()`; when `mode=4`, `aiv_id` must be 0 or 1.

        Example:
        ```python
        with tla.vector():
            tla.cross_core_wait_flag(cf, tla.arch.VECTOR)
        ```
    """
    _require_category("cross_core_wait_flag", "flag", cross_flag_value, "cross_flag", 0)
    _require_pipe("cross_core_wait_flag", "pipe", pipe, 1)
    _require_frontend_state("cross_core_wait_flag")
    _runtime._require_enclosing_cube_or_vector("cross_core_wait_flag")
    aiv_id_attr = _cross_flag_aiv_id_attr(
        "cross_core_wait_flag", cross_flag_value, aiv_id, loc=loc
    )
    return _tla_ops_gen.cross_core_wait_flag(
        _as_value(cross_flag_value),
        _pipe_attr_from_token(pipe, loc=loc),
        aiv_id=aiv_id_attr,
        loc=loc,
    )


@dsl_user_op
def set_flag(flag_value: FlagLike, *, loc: mlir_ir.Location | None = None) -> None:
    """Directory: Sync Control

    Description:
        Set an in-pipe flag.

        Parameters:
        - `flag_value` (`FlagLike`): In-pipe flag from `tla.flag`. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()` or `tla.vector()`.

        Example:
        ```python
        with tla.vector():
            tla.set_flag(ub_loaded)
        ```
    """
    _require_category("set_flag", "flag", flag_value, "flag", 0)
    _require_frontend_state("set_flag")
    _runtime._require_enclosing_cube_or_vector("set_flag")
    return _tla_ops_gen.set_flag(_as_value(flag_value), loc=loc)


@dsl_user_op
def wait_flag(flag_value: FlagLike, *, loc: mlir_ir.Location | None = None) -> None:
    """Directory: Sync Control

    Description:
        Wait on an in-pipe flag.

        Parameters:
        - `flag_value` (`FlagLike`): In-pipe flag from `tla.flag`. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()` or `tla.vector()`.

        Example:
        ```python
        with tla.vector():
            tla.wait_flag(ub_loaded)
        ```
    """
    _require_category("wait_flag", "flag", flag_value, "flag", 0)
    _require_frontend_state("wait_flag")
    _runtime._require_enclosing_cube_or_vector("wait_flag")
    return _tla_ops_gen.wait_flag(_as_value(flag_value), loc=loc)


@dsl_user_op
def pipe_barrier(pipe: PipeLike, *, loc: mlir_ir.Location | None = None) -> None:
    """Directory: Sync Control

    Description:
        Insert a pipe barrier.

        Parameters:
        - `pipe` (`PipeLike`): Pipe on which to insert the barrier. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()` or `tla.vector()`.

        Example:
        ```python
        with tla.vector():
            tla.pipe_barrier(tla.arch.MTE2)
        ```
    """
    _require_pipe("pipe_barrier", "pipe", pipe, 0)
    _require_frontend_state("pipe_barrier")
    _runtime._require_enclosing_cube_or_vector("pipe_barrier")
    ctx = loc.context if loc is not None else mlir_ir.Context()
    pipe_value = str(_token(pipe)).lower()
    cube_pipes = ["cube", "mte1", "mte2", "fix", "all"]
    vector_pipes = [
        "mte2",
        "mte3",
        "all",
    ]  # NOTE: arch 3510 do not support pipe_barrier<pipe_v> specific
    if _runtime._has_enclosing_region("cube") and pipe_value not in cube_pipes:
        raise TlaLoweringError(
            f"in cube pipe_barrier only support {cube_pipes}, got {pipe_value}"
        )
    elif _runtime._has_enclosing_region("vector") and pipe_value not in vector_pipes:
        raise TlaLoweringError(
            f"in vector pipe_barrier only support {vector_pipes}, got {pipe_value}"
        )
    pipe_attr = mlir_ir.Attribute.parse(f"#tla.pipe<{pipe_value}>", context=ctx)
    return _tla_ops_gen.pipe_barrier(pipe_attr, loc=loc)


@dsl_user_op
def mutex(
    resource: str,
    id: int = -1,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaMutex:
    """Directory: Sync Control

    Description:
        Create a mutex for a named resource.

        Parameters:
        - `resource` (`str`): Mutex resource name. Required.
        - `id` (`int`): Mutex instance id; `-1` means default. Optional, default `-1`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - `resource` must be non-empty; `id` must be -1 or 0..31.

        Example:
        ```python
        mtx = tla.mutex("l1_buf", id=0)
        ```
    """
    if not isinstance(resource, str):
        _op_error(
            "mutex",
            f"invalid argument 'resource' (position 0): expected non-empty str, got {_type_name(resource)}",
        )
    resource_value = resource.strip()
    if not resource_value:
        _op_error("mutex", "resource must be non-empty")
    if isinstance(id, bool) or not isinstance(id, int):
        _op_error(
            "mutex",
            f"invalid argument 'id' (position 1): expected int in {{-1, 0..31}}, got {_type_name(id)}",
        )
    if id != -1 and not 0 <= id <= 31:
        _op_error("mutex", "id must be -1 or in range 0..31")
    _require_frontend_state("mutex")
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    value = _tla_ops_gen.mutex(
        _tla_type_bridge.mutex_type_get(ctx), resource_value, int(id), loc=loc
    )
    return _MutexValue(value, resource_value, int(id))


@dsl_user_op
def mutex_guard(
    *mutexes: MutexLike, loc: mlir_ir.Location | None = None
) -> _MutexGuard:
    """Directory: Sync Control

    Description:
        Context manager that locks/unlocks one or more mutexes.

        Parameters:
        - *`mutexes`* (`MutexLike`): One or more mutex objects. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - The block must emit `copy` or `mmad`; do not explicitly lock/unlock inside the guard.

        Example:
        ```python
        with tla.mutex_guard(mtx):
            tla.copy(dst, src)
        ```
    """
    if not mutexes:
        _op_error("mutex_guard", "expected at least one mutex")
    return _MutexGuard(tuple(mutexes), loc=loc)


def _emit_mutex_lock_op(
    mutex_value: MutexLike, *, pipe: PipeLike | str, loc: mlir_ir.Location | None = None
) -> None:
    pipe_attr = _pipe_attr_from_token(pipe, loc=loc)
    return _tla_ops_gen.mutex_lock(_as_value(mutex_value), pipe_attr, loc=loc)


def _emit_mutex_unlock_op(
    mutex_value: MutexLike, *, pipe: PipeLike | str, loc: mlir_ir.Location | None = None
) -> None:
    pipe_attr = _pipe_attr_from_token(pipe, loc=loc)
    return _tla_ops_gen.mutex_unlock(_as_value(mutex_value), pipe_attr, loc=loc)


@dsl_user_op
def mutex_lock(
    mutex_value: MutexLike, *, pipe: PipeLike, loc: mlir_ir.Location | None = None
) -> None:
    """Directory: Sync Control

    Description:
        Lock a mutex on the given pipe.

        Parameters:
        - `mutex_value` (`MutexLike`): Mutex to lock. Required.
        - `pipe` (`PipeLike`): Pipe used for the lock. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()` or `tla.vector()`; `pipe` is required.

        Example:
        ```python
        tla.mutex_lock(mtx, pipe=tla.arch.MTE2)
        ```
    """
    _ensure_no_explicit_mutex_access_in_guard()
    _require_category("mutex_lock", "mutex", mutex_value, "mutex", 0)
    _require_pipe("mutex_lock", "pipe", pipe, 1)
    _require_frontend_state("mutex_lock")
    _runtime._require_enclosing_cube_or_vector("mutex_lock")
    return _emit_mutex_lock_op(mutex_value, pipe=pipe, loc=loc)


@dsl_user_op
def mutex_unlock(
    mutex_value: MutexLike, *, pipe: PipeLike, loc: mlir_ir.Location | None = None
) -> None:
    """Directory: Sync Control

    Description:
        Unlock a mutex on the given pipe.

        Parameters:
        - `mutex_value` (`MutexLike`): Mutex to unlock. Required.
        - `pipe` (`PipeLike`): Pipe used for the unlock. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()` or `tla.vector()`; `pipe` is required.

        Example:
        ```python
        tla.mutex_unlock(mtx, pipe=tla.arch.MTE2)
        ```
    """
    _ensure_no_explicit_mutex_access_in_guard()
    _require_category("mutex_unlock", "mutex", mutex_value, "mutex", 0)
    _require_pipe("mutex_unlock", "pipe", pipe, 1)
    _require_frontend_state("mutex_unlock")
    _runtime._require_enclosing_cube_or_vector("mutex_unlock")
    return _emit_mutex_unlock_op(mutex_value, pipe=pipe, loc=loc)


@dsl_user_op
def local_mem_bar(
    src: MemType,
    dst: MemType,
    *,
    loc: mlir_ir.Location | None = None,
):
    # MemType pair → encoded I32 imm (matching hivmave.membar encoding)
    """Directory: Sync Control

    Description:
        Insert a local-memory barrier inside `vec.func` (encoded from a `MemType` pair).

        Parameters:
        - `src` (`MemType`): Source local memory type. Required.
        - `dst` (`MemType`): Destination local memory type. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; `(src, dst)` must be a supported `MemType` pair.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        ```
    """
    _local_mem_bar_barrier_kind = {
        (MemType.VEC_STORE, MemType.VEC_LOAD): 1,
        (MemType.VEC_LOAD, MemType.VEC_STORE): 2,
        (MemType.VEC_STORE, MemType.VEC_STORE): 3,
        (MemType.VEC_STORE, MemType.SCALAR_LOAD): 5,
        (MemType.VEC_STORE, MemType.SCALAR_STORE): 7,
        (MemType.VEC_LOAD, MemType.SCALAR_STORE): 6,
        (MemType.SCALAR_STORE, MemType.VEC_LOAD): 9,
        (MemType.SCALAR_STORE, MemType.VEC_STORE): 11,
        (MemType.SCALAR_LOAD, MemType.VEC_STORE): 10,
        (MemType.VEC_ALL, MemType.VEC_ALL): 0,
        (MemType.VEC_ALL, MemType.SCALAR_ALL): 4,
        (MemType.SCALAR_ALL, MemType.VEC_ALL): 8,
    }
    # check support
    if (src, dst) not in _local_mem_bar_barrier_kind:
        _op_error(
            "local_mem_bar",
            f"unsupported src and dst: {src.name} and {dst.name}",
        )
    _require_frontend_state("local_mem_bar")
    _runtime._require_enclosing_region("local_mem_bar", "vec.func")
    _tla_ops_gen.local_mem_bar(_local_mem_bar_barrier_kind[(src, dst)], loc=loc)


@dsl_user_op
def range(
    start: IndexLike,
    end: IndexLike | None = None,
    step: IndexLike | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> _ast_helpers.FrontendRange:
    """Directory: Scopes and Control Flow

    Description:
        Create a dynamic loop range for kernel-side iteration.

        Parameters:
        - `start` (`IndexLike`): Loop start (or exclusive end when `end` is omitted, with start 0). Required.
        - `end` (`IndexLike | None`): Exclusive loop end; when omitted, `start` is treated as the end. Optional, default `None`.
        - `step` (`IndexLike | None`): Step; defaults to 1. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Dynamic range; the loop body must satisfy front-end dynamic-for constraints.

        Example:
        ```python
        for i in tla.range(0, n, 1):
            ...
        ```
    """
    del loc
    if end is None and step is None:
        _require_index_or_numeric("range", "end", start, 0)
        return _ast_helpers.range(start)
    if step is None:
        _require_index_or_numeric("range", "start", start, 0)
        _require_index_or_numeric("range", "end", end, 1)
        return _ast_helpers.range(start, end)
    if end is None:
        _op_error("range", "expected 1, 2, or 3 arguments")
    _require_index_or_numeric("range", "start", start, 0)
    _require_index_or_numeric("range", "end", end, 1)
    _require_index_or_numeric("range", "step", step, 2)
    return _ast_helpers.range(start, end, step)


def _require_constexpr_range_bound(op_name: str, name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _op_error(op_name, f"{name} must be a compile-time Python int")
    return value


@dsl_user_op
def range_constexpr(
    start: int,
    end: int | None = None,
    step: int | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> range:
    """Directory: Scopes and Control Flow
    Description:
        Create a front-end static range for unrollable Python loops.

        Parameters:
        - `start` (`int`): Compile-time loop start (or exclusive end when `end` is omitted). Required.
        - `end` (`int | None`): Compile-time exclusive loop end. Optional, default `None`.
        - `step` (`int | None`): Compile-time step; defaults to 1. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Bounds and step must be compile-time constants for unrollable loops.
        - Bounds may come from compile-time Numeric values (for example via
          `tla.as_numeric(...)`).
        - Emits `DSLOptimizationWarning` when the loop has 64 or more
          iterations; expansion continues. Prefer `tla.range(...)` for large
          counted loops.

        Example:
        ```python
        for k in tla.range_constexpr(0, 4):
            ...
        ```
    """
    del loc
    if end is None and step is None:
        return _ast_helpers.range_constexpr(
            _require_constexpr_range_bound("range_constexpr", "end", start),
        )
    if step is None:
        return _ast_helpers.range_constexpr(
            _require_constexpr_range_bound("range_constexpr", "start", start),
            _require_constexpr_range_bound("range_constexpr", "end", end),
        )
    if end is None:
        _op_error("range_constexpr", "expected 1, 2, or 3 arguments")
    return _ast_helpers.range_constexpr(
        _require_constexpr_range_bound("range_constexpr", "start", start),
        _require_constexpr_range_bound("range_constexpr", "end", end),
        _require_constexpr_range_bound("range_constexpr", "step", step),
    )


@dsl_user_op
def cube(*, loc: mlir_ir.Location | None = None) -> TlaRegion:
    """Directory: Scopes and Control Flow
    Description:
        Enter a cube-core region.

        Parameters:
        None.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Wraps cube-side matmul and related copies.

        Example:
        ```python
        with tla.cube():
            tla.mmad(acc=l0c, lhs=l0a, rhs=l0b, init_c=True)
        ```
    """
    return _region_stub("cube")


@dsl_user_op
def vector(*, loc: mlir_ir.Location | None = None) -> TlaRegion:
    """Directory: Scopes and Control Flow

    Description:
        Enter a vector-core region.

        Parameters:
        None.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Wraps vector-side copies and sync; `tla.vec.func` must nest inside it.

        Example:
        ```python
        with tla.vector():
            tla.copy(dst=x_ub, src=x_gm)
        ```
    """
    return _region_stub("vector")


_VEC_FUNC_MODES = {"simd", "SIMD", "simt", "SIMT"}


def _validate_vec_func_mode(mode: str) -> None:
    if not isinstance(mode, str):
        _op_error("vec.func", f"mode must be a string; got {_type_name(mode)}")
    if mode not in _VEC_FUNC_MODES:
        accepted = ", ".join(sorted(repr(value) for value in _VEC_FUNC_MODES))
        _op_error("vec.func", f"mode must be one of {accepted}; got {mode!r}")


@dsl_user_op
def _vec_func(
    *,
    mode: str = "simd",
    thread_block_dim: int | tuple[int, int, int] | list[int] | None = None,
    loc: mlir_ir.Location | None = None,
) -> TlaRegion:
    """Directory: Scopes and Control Flow
    Description:
        Enter a vector-function region for register-vector / mask compute
        (`tla.vec.func`).

        Parameters:
        - `mode` (`str`): Execution mode; `simd` (default) or `simt`. Optional,
          default `"simd"`.
        - `thread_block_dim` (`int | tuple[int, int, int] | list[int] | None`): SIMT thread-block shape; only valid with
          `mode="simt"`. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must nest inside `tla.vector()`.
        - Register-vector / mask APIs and `local_mem_bar` must run in this region.

        Example:
        ```python
        with tla.vector():
            with tla.vec.func(mode="simd"):
                z = tla.add(x_reg, y_reg)
        ```
    """
    del loc
    _validate_vec_func_mode(mode)
    if thread_block_dim is not None:
        from . import runtime as _rt

        _rt._normalize_vec_func_thread_block_dim(thread_block_dim, mode)
    return _region_stub("vec.func")


# Operand handling shared by every matmul: the checks here are the same whether
# the op ends up as tla.mmad or tla.mmad_mx, which tla-cube-region decides.
def _prepare_mmad_operands(
    op_name, acc, lhs, rhs, init_c, unit_flag, compute_order, loc, extra_kwargs
):
    if extra_kwargs:
        _op_error(
            op_name,
            f"unknown keyword argument(s): {', '.join(sorted(extra_kwargs))}",
        )
    _require_category(op_name, "acc", acc, "tensor", 0)
    _require_category(op_name, "lhs", lhs, "tensor", 1)
    _require_category(op_name, "rhs", rhs, "tensor", 2)
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "cube")

    if init_c is None:
        init_c = False
    _require_bool(op_name, "init_c", init_c, 3)
    init_c_value = _as_i1_value(init_c)

    if unit_flag is None:
        unit_flag = 0
    _require_index(op_name, "unit_flag", unit_flag, 4)
    if isinstance(unit_flag, int):
        if unit_flag not in [0b00, 0b10, 0b11]:
            raise TlaLoweringError(
                f"tla.{op_name} operand 'unit_flag' expects values [0b00, 0b10, 0b11], "
                f"got [{unit_flag}]"
            )
        unit_flag_value = _const_i64(unit_flag)
    elif (
        isinstance(unit_flag, Numeric)
        and type(unit_flag).is_integer
        and type(unit_flag).signed
    ):
        unit_flag_value = _as_i64_value(unit_flag)
    elif _category(unit_flag) == "index":
        unit_flag_value = _as_i64_value(unit_flag)
    else:
        raise TlaLoweringError(f"tla.{op_name} unit_flag must be a int")

    if not isinstance(compute_order, ComputeOrder):
        raise TlaLoweringError(
            f"tla.{op_name} attribute 'compute_order' must be a "
            f"{ComputeOrder}, got {type(compute_order).__name__}"
        )
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    compute_order_attr = mlir_ir.Attribute.parse(
        f"#tla.compute_order<{str(compute_order)}>", context=ctx
    )

    return (
        _as_value(acc),
        _as_value(lhs),
        _as_value(rhs),
        init_c_value,
        unit_flag_value,
        compute_order_attr,
        ctx,
    )


# A @ B -> C
_SUPPORTED_MMAD = frozenset(
    {
        ("f32", "f32", "f32"),
        ("f16", "f16", "f32"),
        ("bf16", "bf16", "f32"),
        ("f8e4m3fn", "f8e4m3fn", "f32"),
        ("f8e5m2", "f8e5m2", "f32"),
        ("f8e4m3fn", "f8e5m2", "f32"),
        ("f8e5m2", "f8e4m3fn", "f32"),
        ("i8", "i8", "i32"),
    }
)

# (MxA, MxScaleA) @ (MxB, MxScaleB) -> C
_SUPPORTED_MMAD_MX = frozenset(
    {
        ("f8e4m3fn", "f8e4m3fn", "f32"),
        ("f8e5m2", "f8e5m2", "f32"),
        ("f8e4m3fn", "f8e5m2", "f32"),
        ("f8e5m2", "f8e4m3fn", "f32"),
        ("f4e2m1", "f4e2m1", "f32"),
        ("f4e1m2", "f4e1m2", "f32"),
        ("f4e2m1", "f4e1m2", "f32"),
        ("f4e1m2", "f4e2m1", "f32"),
    }
)


def _validate_mmad_dtypes(lhs: str, rhs: str, acc: str, is_mx: bool) -> bool:
    key = (lhs, rhs, acc)
    if is_mx:
        return key in _SUPPORTED_MMAD_MX
    else:
        return key in _SUPPORTED_MMAD


def _modify_doc_mmad(fn: Callable[..., Any]) -> None:
    table = _SUPPORTED_MMAD if fn.__name__ == "mmad" else _SUPPORTED_MMAD_MX
    rows = "\n".join(
        [f"| `{lhs}` | `{rhs}` | `{acc}` |" for lhs, rhs, acc in sorted(table)]
    )
    fn.__doc__ += (
        f"Supported {fn.__name__} dtypes\n\n"
        "| lhs | rhs | acc |\n"
        "| --- | --- | --- |\n"
        f"{rows}\n"
    )


_modify_doc_mmad_mx = _modify_doc_mmad


# Shared by every matmul the frontend emits. There is one contract because there
# is one entry point: tla.mmad. Which cube instruction it becomes is settled in
# tla-cube-region from operand provenance, not here.
def _validate_mmad_contract(
    acc: mlir_ir.Value,
    lhs: mlir_ir.Value,
    rhs: mlir_ir.Value,
    is_mx: bool,
) -> None:
    acc_desc = _tla_tensor_type_for_mlir_value(acc)
    lhs_desc = _tla_tensor_type_for_mlir_value(lhs)
    rhs_desc = _tla_tensor_type_for_mlir_value(rhs)

    addrspaces = (
        acc_desc.addrspace,
        lhs_desc.addrspace,
        rhs_desc.addrspace,
    )
    if addrspaces != ("l0c", "l0a", "l0b"):
        raise TlaLoweringError(
            "unsupported tla.mmad tile addrspaces; expected acc/lhs/rhs in l0c/l0a/l0b"
        )

    element_types = (
        lhs_desc.element_type,
        rhs_desc.element_type,
        acc_desc.element_type,
    )
    if not _validate_mmad_dtypes(*element_types, is_mx=is_mx):
        _op_name = is_mx and "mmad_mx" or "mmad"
        raise TlaLoweringError(
            f"unsupported tla.{_op_name} element types, got {element_types}"
        )

    lhs_m, lhs_k = _flat_dim_pair_from_tree(lhs_desc.origin_shape)
    rhs_k, rhs_n = _flat_dim_pair_from_tree(rhs_desc.origin_shape)
    acc_m, acc_n = _flat_dim_pair_from_tree(acc_desc.origin_shape)
    if None not in (lhs_m, lhs_k, rhs_k, rhs_n, acc_m, acc_n) and (
        lhs_k != rhs_k or lhs_m != acc_m or rhs_n != acc_n
    ):
        raise TlaLoweringError(
            "unsupported tla.mmad tile shape contract; expected lhs(MxK), rhs(KxN), acc(MxN)"
        )

    expected_layouts = (
        (acc, "L0Clayout"),
        (lhs, "zN"),
        (rhs, "nZ"),
    )
    for operand, expected in expected_layouts:
        layout = _tla_tensor_type_for_mlir_value(operand).layout_tag
        if layout != expected:
            raise TlaLoweringError(
                "unsupported tla.mmad operand layout; expected acc L0Clayout, lhs zN, rhs nZ"
            )


@dsl_user_op
def mmad(
    acc: Tensor,
    lhs: Tensor,
    rhs: Tensor,
    init_c: bool | Bool | None = None,
    unit_flag: IndexLike | None = None,
    compute_order: ComputeOrder = ComputeOrder.M_FIRST,
    hf32_mode: HF32Mode = HF32Mode.HF32_DISABLE,
    loc: mlir_ir.Location | None = None,
    **extra_kwargs: object,
) -> None:
    """Directory: Matrix Compute
    Description:
        Emit matrix-multiply-accumulate on TLA tiles.

        Parameters:
        - `acc` (`Tensor`): Accumulator / output tile (usually on L0C). Required.
        - `lhs` (`Tensor`): Left-hand matrix tile (usually on L0A). Required.
        - `rhs` (`Tensor`): Right-hand matrix tile (usually on L0B). Required.
        - `init_c` (`bool | Bool | None`): Whether to clear the accumulator first;
          defaults to `False` when omitted. Optional, default `None`.
        - `unit_flag` (`IndexLike | None`): Unit-flag control bits; defaults to `0`
          when omitted. Optional, default `None`.
        - `compute_order` (`ComputeOrder`): M/N compute-direction priority; default `M_FIRST`.
        - `hf32_mode` (`HF32Mode`): HF32 rounding mode for FP32 operands in L0A/L0B
          before the matrix multiply. Optional, default `HF32_DISABLE`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()`; `acc`/`lhs`/`rhs` must be matching L0 tiles.
        - Supported accumulating dtypes are fp32/i32, supported (`lhs`, `rhs`, `acc`) dtype
          conbinations are as fllows.
        - The `f8e4m3fn` and `f8e5m2` matmul both support non-MxScale matmul and MxScale matmul,
          in the tla.mmad op scale is unsupported.
        - `init_c` accepts only a Python `bool` or an `i1` SSA value.
        - Unknown keyword arguments are not accepted; passing any raises an error.

        Example:
        ```python
        # Before: l0a / l0b hold the current K-slice; l0c is the accumulator on L0C.
        with tla.cube():
            tla.mmad(l0c, l0a, l0b, init_c=True, unit_flag=0b11)
            # After: l0c accumulates lhs@rhs (cleared first when init_c=True).
        ```
    """
    (
        acc_value,
        lhs_value,
        rhs_value,
        init_c_value,
        unit_flag_value,
        compute_order_attr,
        ctx,
    ) = _prepare_mmad_operands(
        "mmad", acc, lhs, rhs, init_c, unit_flag, compute_order, loc, extra_kwargs
    )
    _check_mx_provenance("mmad", lhs_value, rhs_value)
    if not isinstance(hf32_mode, HF32Mode):
        raise TlaLoweringError(
            "tla.mmad attribute 'hf32_mode' must be a "
            f"{HF32Mode}, got {type(hf32_mode).__name__}"
        )
    hf32_mode_attr = mlir_ir.Attribute.parse(
        f"#tla.hf32_mode<{str(hf32_mode)}>", context=ctx
    )
    _validate_mmad_contract(acc_value, lhs_value, rhs_value, is_mx=False)
    _tla_ops_gen.mmad(
        acc_value,
        lhs_value,
        rhs_value,
        init_c_value,
        unit_flag_value,
        loc=loc,
        compute_order=compute_order_attr,
        hf32_mode=hf32_mode_attr,
    )


def _check_mx_provenance(op_name: str, lhs_value: Any, rhs_value: Any) -> None:
    """Hold tla.mmad and tla.mmad_mx to their own operand contracts.

    Whether a matmul is microscaling is a property of how its operands were
    *written* -- by a tla.copy(..., scale=...) that attaches the e8m0 block --
    and not of their type or layout: an MX fp8 L0 tile is indistinguishable from
    a plain one. So neither op can be inferred from the other, and each has to
    reject the operands belonging to the other. Getting it wrong is silent:
    tla.mmad lowers to mad and simply drops the exponents, while mad_mx reads a
    scale block nothing populated.
    """
    state = _runtime._current_frontend_state()
    scaled = state.mx_scaled_l0_values if state is not None else set()
    wants_scaled = op_name == "mmad_mx"
    for value, name in ((lhs_value, "lhs"), (rhs_value, "rhs")):
        if (value in scaled) == wants_scaled:
            continue
        if wants_scaled:
            _op_error(
                op_name,
                f"{name} operand was not loaded by tla.copy(..., scale=...); "
                "tla.mmad_mx needs the e8m0 scale block that load attaches, and "
                "would otherwise read one that was never populated",
            )
        _op_error(
            op_name,
            f"{name} operand was loaded by tla.copy(..., scale=...); a "
            "microscaling matmul must be written as tla.mmad_mx, since tla.mmad "
            "lowers to mad and would drop the scale",
        )


@dsl_user_op
def mmad_mx(
    acc: Tensor,
    lhs: Tensor,
    rhs: Tensor,
    init_c: bool | Bool | None = None,
    unit_flag: IndexLike | None = None,
    compute_order: ComputeOrder = ComputeOrder.M_FIRST,
    loc: mlir_ir.Location | None = None,
    **extra_kwargs: object,
) -> None:
    """Directory: Matrix Compute
    Description:
        Emit a microscaling (MX) matrix-multiply-accumulate on TLA tiles.

        A separate op from `tla.mmad` because the cube has a dedicated `mad_mx`
        instruction, which AscendC exposes as `asc_mmad_mx`.

        Like that instruction, this takes **no scale argument**. `lhs`/`rhs` must
        have been loaded by `tla.copy` with its `scale` operand, which attaches the
        e8m0 block to the L0 tile; the hardware then reads the scale from a side
        buffer addressed off the L0 tile. Passing operands that were *not* loaded
        that way is an error, as is passing microscaling operands to `tla.mmad` --
        the two ops are not interchangeable, and nothing about the operands
        themselves reveals which is which.

        Parameters:
        - `acc` (`Tensor`): Accumulator / output tile on L0C, always fp32. Required.
        - `lhs` (`Tensor`): Left-hand matrix tile on L0A. Required.
        - `rhs` (`Tensor`): Right-hand matrix tile on L0B. Required.
        - `init_c` (`bool | Bool | None`): Whether to clear the accumulator first;
          defaults to `False` when omitted. Optional, default `None`.
        - `unit_flag` (`IndexLike | None`): Unit-flag control bits; defaults to `0`
          when omitted. Optional, default `None`.
        - `compute_order` (`ComputeOrder`): M/N compute-direction priority; default `M_FIRST`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()`; `acc`/`lhs`/`rhs` must be matching L0 tiles.
        - Operands must be an `f8e4m3fn`/`f8e5m2` pair or an `f4e2m1`/`f4e1m2`
          pair; either pair may mix its two formats, and the accumulator is fp32.
        - The `f8e4m3fn` and `f8e5m2` matmul both support non-MxScale matmul and MxScale matmul,
          in the tla.mmad_mx op scale is needed.
        - Both operands must have been loaded by `tla.copy(..., scale=...)`.

        Example:
        ```python
        # Before: l1a/l1b hold fp8 operands, l1sa/l1sb their e8m0 scale blocks.
        with tla.cube():
            tla.copy(l0a, l1a, scale=l1sa)   # scale rides the load, not the matmul
            tla.copy(l0b, l1b, scale=l1sb)
            tla.mmad_mx(l0c, l0a, l0b, init_c=True)
            # After: l0c accumulates the microscaled lhs@rhs.
        ```
    """
    (
        acc_value,
        lhs_value,
        rhs_value,
        init_c_value,
        unit_flag_value,
        compute_order_attr,
        ctx,
    ) = _prepare_mmad_operands(
        "mmad_mx", acc, lhs, rhs, init_c, unit_flag, compute_order, loc, extra_kwargs
    )
    _check_mx_provenance("mmad_mx", lhs_value, rhs_value)
    _validate_mmad_contract(acc_value, lhs_value, rhs_value, is_mx=True)
    # hf32_mode is deliberately absent: it is an fp32 rounding mode, and MX
    # operands are fp8/fp4.
    # An fp4 operand states its encoding in its own element type, so there is
    # nothing to name here and nothing that can contradict the load.
    mmad_kwargs: dict[str, Any] = {"compute_order": compute_order_attr}
    _tla_ops_gen.mmad_mx(
        acc_value,
        lhs_value,
        rhs_value,
        init_c_value,
        unit_flag_value,
        loc=loc,
        **mmad_kwargs,
    )


mmad.__modify_doc__ = _modify_doc_mmad
mmad_mx.__modify_doc__ = _modify_doc_mmad_mx


@dsl_user_op
def full(
    value: bool | int | float | Numeric | VectorSSA,
    dtype: type[Numeric],
    *,
    mask: Any | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: Vector Compute / Data Fill
    Description:
        Fill a 1-D vector SSA with a Python scalar literal, or broadcast a
        one-lane vector fragment (e.g. a `VectorSSA.reduce` result) across every lane.

        Parameters:
        - `value` (`bool | int | float | Numeric | VectorSSA`): Fill constant, or a
          one-lane vector fragment to broadcast. Required.
        - `dtype` (`type[Numeric]`): Vector element type. Required.
        - `mask` (`MaskSSA | None`): Predicates which lanes are written. Optional;
          defaults to writing every lane.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; `value` must be a Python scalar
          literal, a host `Numeric`, or a one-lane vector fragment.
        - A vector `value` must have exactly one valid lane and its element type
          must match `dtype`.
        - `mask`, when given, must match the result element type.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            zeros = tla.full(0.0, dtype=tla.Float32)
            all_lanes, _ = tla.update_mask(64, tla.Float32)
            total = src.reduce(tla.ReductionOp.ADD, mask=all_lanes)
            spread = tla.full(total, dtype=tla.Float32)
        ```
    """
    state = _runtime._current_frontend_state()
    if state is None or not state.active_regions:
        raise TlaCoreAPIError("tla.full is only allowed inside tla.vec.func")
    _runtime._require_enclosing_region("full", "vec.func")
    _require_dtype("full", "dtype", dtype, 1)
    mask_value = None
    if mask is not None:
        _require_category("full", "mask", mask, "mask_ssa", 2)
        mask_value = _as_value(mask)
    if not (
        isinstance(dtype, type)
        and issubclass(dtype, Numeric)
        and getattr(dtype, "dtype", "")
    ):
        _op_error(
            "full",
            f"invalid argument 'dtype' (position 1): expected concrete Numeric "
            f"(e.g. tla.Float32), got {_type_name(dtype)}",
        )
    dtype_token = str(dtype.dtype).strip().lower()
    if dtype_token not in _FULL_SUPPORTED_DTYPES:
        _op_error(
            "full",
            f"unsupported vector element dtype {dtype.dtype}; supported dtypes are "
            f"{', '.join(sorted(_FULL_SUPPORTED_DTYPES))}",
        )
    desc = _full_vector_ssa_descriptor(dtype_token)
    if mask_value is not None:
        # Against the result: a 64-lane f32 predicate is not valid for a
        # 128-lane f16 result, and the lowering would forward it unchanged.
        _require_mask_matches_element_type("full", mask_value, desc.element_type)

    # A one-lane vector fragment (e.g. a VectorSSA.reduce result) can be broadcast
    # directly across the register -- no scalar materialization needed, which
    # matters because the SIMD backend cannot read a vector lane into a scalar
    # register.
    if _category(value) == "vector_ssa":
        src_value = _as_value(value)
        src_desc = _vector_ssa_type_for_mlir_value(src_value)
        if src_desc.element_type != dtype_token:
            _op_error(
                "full",
                f"invalid argument 'value' (position 0): fragment element type "
                f"{src_desc.element_type} does not match dtype {dtype_token}",
            )
        # Only a one-lane fragment. A full-width vector would lower to the same
        # AVE vector_broadcast, which keeps lane 0 and silently drops the rest.
        if src_desc.valid_lanes != 1:
            _op_error(
                "full",
                f"invalid argument 'value' (position 0): expected a one-lane "
                f"vector fragment (e.g. a VectorSSA.reduce result), got "
                f"{src_desc.valid_lanes} valid lanes; broadcasting a full-width "
                f"vector would keep lane 0 only",
            )
        return VectorSSA(
            _tla_ops_gen.full(_coerce_type(desc), src_value, mask=mask_value, loc=loc)
        )

    resolved = _resolve_bound_value(value)
    if isinstance(resolved, Numeric):
        if isinstance(resolved.value, mlir_ir.Value):
            _op_error(
                "full",
                "value must be a Python scalar literal, host Numeric, or "
                "one-lane vector fragment",
            )
        resolved = resolved.value
    if not isinstance(resolved, (bool, int, float)):
        _op_error(
            "full",
            "value must be a Python scalar literal, host Numeric, or one-lane "
            "vector fragment",
        )
    scalar_value = int(resolved) if isinstance(resolved, bool) else resolved
    context = loc.context if loc is not None else mlir_ir.Context.current
    scalar = _scalar_constant_for_element_type(
        "full",
        scalar_value,
        desc.element_mlir_type(context),
        loc=loc,
    )
    result = _tla_ops_gen.full(_coerce_type(desc), scalar, mask=mask_value, loc=loc)
    return VectorSSA(result)


def _full_vector_ssa_descriptor(dtype_token: str) -> TlaVectorSSATypeDescriptor:
    element_bytes = dtype_size_bytes(dtype_token)
    lanes = _vector_lane_count(element_bytes)
    return TlaVectorSSATypeDescriptor(lanes, dtype_token)


@dsl_user_op
def arange(
    base: bool | int | float | Numeric = 0,
    *,
    order: str = "increase",
    dtype: type[Numeric],
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: Vector Compute / Data Fill

    Description:
        Create a monotonically increasing or decreasing 1-D vector SSA (`base` + `order`).

        Parameters:
        - `base` (`bool | int | float | Numeric`): Base offset. Optional, default `0`.
        - `order` (`str`): `'increase'` ascending or `'decrease'` descending. Optional, default `'increase'`.
        - `dtype` (`type[Numeric]`): Vector element type. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; `order` supports only `increase` / `decrease`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            lane_idx = tla.arange(base=0, order="increase", dtype=tla.Int32)
        ```
    """
    op_name = "arange"
    order = str(order).lower()
    if order not in _ARANGE_ORDERS:
        _op_error(
            op_name,
            f"order must be one of {sorted(_ARANGE_ORDERS)}; got {order!r}",
        )
    _require_dtype(op_name, "dtype", dtype, 2)
    if not (
        isinstance(dtype, type)
        and issubclass(dtype, Numeric)
        and getattr(dtype, "dtype", "")
    ):
        _op_error(
            op_name,
            f"invalid argument 'dtype' (position 2): expected concrete Numeric "
            f"(e.g. tla.Float32), got {_type_name(dtype)}",
        )
    dtype_token = str(dtype.dtype).strip().lower()
    if dtype_token not in _ARANGE_SUPPORTED_DTYPES:
        _op_error(
            op_name,
            f"unsupported vector element dtype {dtype.dtype}; supported dtypes are "
            f"{', '.join(sorted(_ARANGE_SUPPORTED_DTYPES))}",
        )
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    desc = _full_vector_ssa_descriptor(dtype_token)
    context = loc.context if loc is not None else mlir_ir.Context.current
    element_type = desc.element_mlir_type(context)
    const = _const_int_value(base)
    if const is not None:
        start_value = _scalar_constant_for_element_type(
            op_name, const, element_type, loc=loc
        )
    else:
        resolved = _resolve_bound_value(base)
        if isinstance(resolved, mlir_ir.Value):
            if resolved.type == element_type:
                start_value = resolved
            elif isinstance(resolved.type, mlir_ir.IndexType):
                start_value = mlir_ir.Operation.create(
                    "arith.index_cast",
                    operands=[resolved],
                    results=[element_type],
                    loc=loc,
                ).results[0]
            else:
                _op_error(
                    op_name,
                    "base must be an integer literal or index SSA value",
                )
        elif (
            isinstance(resolved, Numeric)
            and type(resolved).is_integer
            and type(resolved).signed
        ):
            index_value = _runtime._coerce_index_value(resolved)
            start_value = mlir_ir.Operation.create(
                "arith.index_cast",
                operands=[index_value],
                results=[element_type],
                loc=loc,
            ).results[0]
        else:
            _op_error(
                op_name,
                "base must be an integer literal or index SSA value",
            )
    result = _tla_ops_gen.arange(_coerce_type(desc), start_value, order=order, loc=loc)
    return VectorSSA(result)


def _emit_vector_binary(
    op_name: str,
    emitter: Any,
    lhs: VectorSSA,
    rhs: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Shared lowering for element-wise vector-vector binary ops.

    Optional ``mask`` (a ``MaskSSA`` from ``tla.create_mask`` or
    ``tla.update_mask``) controls which lanes are computed; masked-out lanes
    are undefined/zeroed.
    """
    _require_category(op_name, "lhs", lhs, "vector_ssa", 0)
    _require_category(op_name, "rhs", rhs, "vector_ssa", 1)
    if mask is not None:
        _require_category(op_name, "mask", mask, "mask_ssa", 2)
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    lhs_value = _as_value(lhs)
    rhs_value = _as_value(rhs)
    mask_value = _as_value(mask) if mask is not None else None
    lhs_desc = _vector_ssa_type_for_mlir_value(lhs_value)
    rhs_desc = _vector_ssa_type_for_mlir_value(rhs_value)
    if lhs_desc.element_type != rhs_desc.element_type:
        _op_error(
            op_name,
            f"rhs has element type {rhs_desc.element_type}, expected "
            f"{lhs_desc.element_type}",
        )
    if mask_value is not None:
        _require_mask_matches_vector(op_name, mask_value, lhs_value)
    result = emitter(
        lhs_value.type,
        lhs_value,
        rhs_value,
        mask=mask_value,
        loc=loc,
    )
    return VectorSSA(result)


def _as_vector_scalar_numeric(value: Any) -> Numeric | None:
    """Canonicalize vector–scalar rhs to ``Numeric`` via ``as_numeric``.

    Returns ``None`` if ``value`` is not a scalar operand.
    """
    if isinstance(value, VectorSSA) or _category(value) == "vector_ssa":
        return None
    if isinstance(value, Numeric):
        return value
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, Numeric):
        return resolved
    if isinstance(resolved, (bool, int, float)):
        return as_numeric(resolved)
    if isinstance(resolved, mlir_ir.Value):
        # Also accept bare ``ir.Value`` via ``as_numeric``.
        try:
            return as_numeric(resolved)
        except (TypeError, ValueError, KeyError):
            return None
    return None


def _numeric_ir_value_for_element_type(
    op_name: str,
    scalar: Numeric,
    element_type: mlir_ir.Type,
    *,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    """Materialize scalar for vector–scalar ops via ``to`` + ``ir_value``.

    Host Python numbers keep literal ``arith.constant`` emission (range / fraction
    checks). Dynamic SSA uses ``Numeric.to`` + ``ir_value``.
    """
    if isinstance(scalar.value, (bool, int, float)):
        return _scalar_constant_for_element_type(
            op_name, scalar.value, element_type, loc=loc
        )
    dest_cls = Numeric.from_mlir_type(element_type)
    return scalar.to(dest_cls, loc=loc).ir_value(loc=loc)


def _scalar_constant_for_element_type(
    op_name: str,
    scalar: Any,
    element_type: mlir_ir.Type,
    *,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    """Emit ``arith.constant`` from a host scalar literal only."""
    resolved = _resolve_bound_value(scalar)
    if isinstance(resolved, Numeric):
        if isinstance(resolved.value, mlir_ir.Value):
            _op_error(
                op_name,
                "invalid argument 'rhs' (position 1): expected scalar literal, got Numeric SSA",
            )
        resolved = resolved.value
    if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
        _op_error(
            op_name,
            f"invalid argument 'rhs' (position 1): expected scalar, got {_type_name(scalar)}",
        )
    if isinstance(element_type, mlir_ir.IndexType) or mlir_ir.IntegerType.isinstance(
        element_type
    ):
        if isinstance(resolved, float) and not resolved.is_integer():
            _op_error(
                op_name,
                f"invalid argument 'rhs' (position 1): expected integer scalar for "
                f"{element_type}, got {resolved!r}",
            )
        int_value = int(resolved)
        # Range check before IntegerAttr.get (signed bounds for signless integer types).
        if isinstance(element_type, mlir_ir.IndexType):
            lo, hi = -(2**63), 2**63 - 1
        else:
            int_ty = mlir_ir.IntegerType(element_type)
            width = int(int_ty.width)
            if int_ty.is_unsigned:
                lo, hi = 0, 2**width - 1
            else:
                lo, hi = -(2 ** (width - 1)), 2 ** (width - 1) - 1
        if not (lo <= int_value <= hi):
            _op_error(
                op_name,
                f"integer scalar {int_value} out of range for {element_type} "
                f"(valid range [{lo}, {hi}])",
            )
        return mlir_ir.Operation.create(
            "arith.constant",
            results=[element_type],
            attributes={"value": mlir_ir.IntegerAttr.get(element_type, int_value)},
            loc=loc,
        ).results[0]
    if (
        mlir_ir.F16Type.isinstance(element_type)
        or mlir_ir.F32Type.isinstance(element_type)
        or mlir_ir.F64Type.isinstance(element_type)
        or mlir_ir.BF16Type.isinstance(element_type)
    ):
        return mlir_ir.Operation.create(
            "arith.constant",
            results=[element_type],
            attributes={"value": mlir_ir.FloatAttr.get(element_type, float(resolved))},
            loc=loc,
        ).results[0]
    _op_error(
        op_name, f"unsupported vector element type for scalar literal: {element_type}"
    )


def _emit_vector_scalar_binary(
    op_name: str,
    emitter: Any,
    lhs: VectorSSA,
    rhs: Any,
    *,
    mask: Any | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    _require_category(op_name, "lhs", lhs, "vector_ssa", 0)
    rhs_num = _as_vector_scalar_numeric(rhs)
    if rhs_num is None:
        _op_error(
            op_name,
            f"invalid argument 'rhs' (position 1): expected scalar, got {_type_name(rhs)}",
        )
    if mask is not None:
        _require_category(op_name, "mask", mask, "mask_ssa", 2)
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    lhs_value = _as_value(lhs)
    lhs_desc = _vector_ssa_type_for_mlir_value(lhs_value)
    rhs_value = _numeric_ir_value_for_element_type(
        op_name, rhs_num, lhs_desc.element_mlir_type(lhs_value.type.context), loc=loc
    )
    mask_value = _as_value(mask) if mask is not None else None
    if mask_value is not None:
        _require_mask_matches_vector(op_name, mask_value, lhs_value)
    result = emitter(
        lhs_value.type,
        lhs_value,
        rhs_value,
        mask=mask_value,
        loc=loc,
    )
    return VectorSSA(result)


def _emit_vector_binary_or_scalar(
    op_name: str,
    vector_emitter: Any,
    scalar_op_name: str,
    lhs: Any,
    rhs: Any,
    *,
    mask: Any | None = None,
    loc: mlir_ir.Location | None = None,
    commutative: bool = False,
) -> VectorSSA:
    lhs_category = _category(lhs)
    rhs_category = _category(rhs)
    lhs_num = _as_vector_scalar_numeric(lhs)
    rhs_num = _as_vector_scalar_numeric(rhs)
    if lhs_category == "vector_ssa" and rhs_category == "vector_ssa":
        return _emit_vector_binary(
            op_name, vector_emitter, lhs, rhs, mask=mask, loc=loc
        )
    if lhs_category == "vector_ssa" and rhs_num is not None:
        return _emit_vector_scalar_binary(
            op_name, getattr(_tla_ops_gen, scalar_op_name), lhs, rhs, mask=mask, loc=loc
        )
    if commutative and lhs_num is not None and rhs_category == "vector_ssa":
        return _emit_vector_scalar_binary(
            op_name, getattr(_tla_ops_gen, scalar_op_name), rhs, lhs, mask=mask, loc=loc
        )
    if lhs_category != "vector_ssa":
        _require_category(op_name, "lhs", lhs, "vector_ssa", 0)
    _op_error(op_name, "expected vector-vector or vector-scalar operands")


_SHIFT_SUPPORTED_DTYPES = frozenset({"i8", "i16", "i32"})
_SIGNED_SHIFT_ELEMENT_TYPE = {
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
}


def _materialize_full_shift_mask(
    source_desc: TlaVectorSSATypeDescriptor,
    *,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    """Build an all-lanes-active mask matching ``source_desc`` element type."""
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    mask_ty = _mask_ssa_type_for_element_type(source_desc.element_type).to_mlir_type(
        ctx
    )
    return _tla_ops_gen.create_mask(
        mask_ty,
        pattern="ALL",
        dtype=mlir_ir.TypeAttr.get(source_desc.element_mlir_type(ctx)),
        loc=loc,
    )


def _emit_vector_shift(
    op_name: str,
    source: VectorSSA,
    shift: Any,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Emit a masked register shift, dispatching vector and scalar amounts."""
    _require_category(op_name, "source", source, "vector_ssa", 0)
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")

    source_value = _as_value(source)
    source_desc = _vector_ssa_type_for_mlir_value(source_value)
    source_element = str(source_desc.element_type).lower()
    if source_element not in _SHIFT_SUPPORTED_DTYPES:
        _op_error(
            op_name,
            f"unsupported source element type {source_desc.element_type}; "
            f"supported types are {', '.join(sorted(_SHIFT_SUPPORTED_DTYPES))}",
        )

    if mask is None:
        mask_value = _materialize_full_shift_mask(source_desc, loc=loc)
    else:
        _require_category(op_name, "mask", mask, "mask_ssa", 2)
        mask_value = _as_value(mask)
        _require_mask_matches_vector(op_name, mask_value, source_value)

    if _category(shift) == "vector_ssa":
        shift_value = _as_value(shift)
        shift_desc = _vector_ssa_type_for_mlir_value(shift_value)
        expected = _SIGNED_SHIFT_ELEMENT_TYPE[source_element]
        if str(shift_desc.element_type).lower() != expected:
            _op_error(
                op_name,
                f"shift vector has element type {shift_desc.element_type}, "
                f"expected {expected} for {source_element} source",
            )
        if shift_desc.valid_lanes != source_desc.valid_lanes:
            _op_error(
                op_name,
                f"shift vector has {shift_desc.valid_lanes} valid lanes, expected "
                f"{source_desc.valid_lanes}",
            )
        emitter = getattr(_tla_ops_gen, op_name)
    else:
        shift_num = _as_vector_scalar_numeric(shift)
        if (
            shift_num is None
            or not type(shift_num).is_integer
            or type(shift_num) is Bool
        ):
            _op_error(
                op_name,
                f"invalid argument 'shift' (position 1): expected integer scalar "
                f"or signed VectorSSA, got {_type_name(shift)}",
            )
        resolved_shift = _resolve_bound_value(shift_num)
        literal = (
            resolved_shift.value
            if isinstance(resolved_shift, Numeric)
            else resolved_shift
        )
        if isinstance(literal, int) and literal < 0:
            _op_error(op_name, "shift amount must be non-negative")
        shift_value = _numeric_ir_value_for_element_type(
            op_name, shift_num, Int16.mlir_type(), loc=loc
        )
        emitter = getattr(_tla_ops_gen, f"{op_name}s")

    result = emitter(
        source_value.type,
        source_value,
        shift_value,
        mask=mask_value,
        loc=loc,
    )
    return VectorSSA(result)


_FLOAT_UNARY_ELEMENT_TYPES = frozenset({"f16", "f32"})
_INTEGER_ABS_ELEMENT_TYPES = frozenset({"i8", "i16", "i32"})
_ABS_ELEMENT_TYPES = _FLOAT_UNARY_ELEMENT_TYPES | _INTEGER_ABS_ELEMENT_TYPES
_BITWISE_UNARY_TYPES = _ABS_ELEMENT_TYPES


def _emit_vector_unary(
    op_name: str,
    emitter: Any,
    operand: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    _require_category(op_name, "operand", operand, "vector_ssa", 0)
    if mask is not None:
        _require_category(op_name, "mask", mask, "mask_ssa", 1)
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    operand_value = _as_value(operand)
    element_type = _vector_ssa_type_for_mlir_value(operand_value).element_type
    if op_name in {"exp", "log", "sqrt"}:
        if element_type not in _FLOAT_UNARY_ELEMENT_TYPES:
            _op_error(
                op_name,
                f"tla.{op_name} requires f16 or f32 element type, got {element_type}",
            )
    elif op_name in {"abs", "neg"} and element_type not in _ABS_ELEMENT_TYPES:
        _op_error(
            op_name,
            f"tla.{op_name} requires f16/f32 or i8/i16/i32 element type, "
            f"got {element_type}",
        )
    mask_value = _as_value(mask) if mask is not None else None
    if mask_value is not None:
        _require_mask_matches_vector(op_name, mask_value, operand_value)
    result = emitter(
        operand_value.type,
        operand_value,
        mask=mask_value,
        loc=loc,
    )
    return VectorSSA(result)


def _make_unary_op(mnemonic: str, *, doc: str) -> Callable[..., VectorSSA]:
    @dsl_user_op
    def _unary(
        operand: VectorSSA,
        *,
        mask: MaskSSA | None = None,
        loc: mlir_ir.Location | None = None,
    ) -> VectorSSA:
        return _emit_vector_unary(
            mnemonic,
            getattr(_tla_ops_gen, mnemonic),
            operand,
            mask=mask,
            loc=loc,
        )

    _unary.__name__ = mnemonic
    _unary.__doc__ = doc
    return _unary


# Scalar counterparts of the vector unaries, for use inside a SIMT region. The
# public names below dispatch on the operand: a VectorSSA keeps the whole-vector
# AVE op, a per-thread Numeric gets the tla.simt_* op.
_SIMT_UNARY_OPS = {
    "exp": "tla.simt_exp",
    "sqrt": "tla.simt_sqrt",
    "abs": "tla.simt_abs",
    "log": "tla.simt_log",
}


# Scalar unaries that also have a form outside a SIMT region, on the core's own
# scalar unit. These reach the backend as plain ``math`` ops, the same route
# ``Numeric.__abs__`` takes. exp and log are absent on purpose: probing hivmc
# shows math.exp/log on a non-SIMT scalar lower to a libm call and leave an
# undefined expf/logf symbol, so neither is usable on this path.
_SCALAR_UNARY_OPS = {
    "sqrt": "math.sqrt",
}

# ...and f32 is the only element type the scalar path accepts. math.sqrt on a
# scalar f16 or bf16 reaches the backend as "Cannot select: f16 = fsqrt", an
# error from deep inside the compiler that says nothing about the kernel, so the
# frontend rejects those here instead. The SIMT form is unaffected: tla.simt_sqrt
# selects for f16, and tla-vector-region promotes bf16 through f32.
_SCALAR_UNARY_ELEMENT_TYPE = Float32


@dsl_user_op
def _emit_simt_unary(
    mnemonic: str, operand: Any, *, loc: mlir_ir.Location | None = None
) -> Any:
    _require_frontend_state(mnemonic)
    in_simt = _runtime._in_simt_vec_func()
    if not in_simt and mnemonic not in _SCALAR_UNARY_OPS:
        _op_error(
            mnemonic,
            "on a scalar is only available inside a tla.vec.func with mode='simt'; "
            "pass a VectorSSA for the whole-vector form",
        )
    num = operand if isinstance(operand, Numeric) else as_numeric(operand)
    if not type(num).is_float:
        _op_error(mnemonic, f"is float-only; got {type(num).__name__}")
    if not in_simt and type(num) is not _SCALAR_UNARY_ELEMENT_TYPE:
        _op_error(
            mnemonic,
            f"on a scalar outside a SIMT region requires "
            f"{_SCALAR_UNARY_ELEMENT_TYPE.__name__}; got {type(num).__name__}. "
            "Convert first with .to(tla.Float32), or use a "
            "tla.vec.func(mode='simt') region",
        )
    value = num.ir_value(loc=loc)
    name = _SIMT_UNARY_OPS[mnemonic] if in_simt else _SCALAR_UNARY_OPS[mnemonic]
    result = mlir_ir.Operation.create(
        name, operands=[value], results=[value.type], loc=loc
    ).results[0]
    return type(num)(result)


def _make_scalar_aware_unary_op(
    mnemonic: str, doc: str | None = None
) -> Callable[..., Any]:
    vector_form = _make_unary_op(mnemonic, doc=doc)

    @dsl_user_op
    def _unary(
        operand: Any, *, mask: Any = None, loc: mlir_ir.Location | None = None
    ) -> Any:
        if not isinstance(operand, VectorSSA):
            if mask is not None:
                _op_error(
                    mnemonic, "mask is not supported for the per-thread scalar form"
                )
            return _emit_simt_unary(mnemonic, operand, loc=loc)
        return vector_form(operand, mask=mask, loc=loc)

    _unary.__name__ = mnemonic
    _unary.__doc__ = doc
    return _unary


_SIMT_BINARY_OPS = {"max": "tla.simt_max", "min": "tla.simt_min"}


@dsl_user_op
def _emit_simt_binary(
    mnemonic: str, lhs: Any, rhs: Any, *, loc: mlir_ir.Location | None = None
) -> Any:
    _require_frontend_state(mnemonic)
    if not _runtime._in_simt_vec_func():
        _op_error(
            mnemonic,
            "on scalars is only available inside a tla.vec.func with mode='simt'; "
            "pass VectorSSA values for the whole-vector form",
        )
    left = lhs if isinstance(lhs, Numeric) else as_numeric(lhs)
    right = rhs if isinstance(rhs, Numeric) else as_numeric(rhs)
    if type(left) is not type(right):
        _op_error(
            mnemonic,
            f"needs both operands of the same type; got {type(left).__name__} "
            f"and {type(right).__name__}",
        )
    if type(left) is Bool or (type(left).is_integer and not type(left).signed):
        _op_error(
            mnemonic,
            "does not support unsigned integers or Bool operands; use signed "
            "integers or floats",
        )
    value = left.ir_value(loc=loc)
    result = mlir_ir.Operation.create(
        _SIMT_BINARY_OPS[mnemonic],
        operands=[value, right.ir_value(loc=loc)],
        results=[value.type],
        loc=loc,
    ).results[0]
    return type(left)(result)


def _make_scalar_aware_binary_op(mnemonic: str, vector_form: Callable[..., Any]):
    # dsl_user_op captures the caller's frame, so it has to sit on the public
    # name: without it the location recorded for tla.max/tla.min would point at
    # this wrapper instead of the user's kernel line.
    @dsl_user_op
    def _binary(
        lhs: Any, rhs: Any, *, mask: Any = None, loc: mlir_ir.Location | None = None
    ) -> Any:
        scalars = not isinstance(lhs, VectorSSA) and not isinstance(rhs, VectorSSA)
        # Only a SIMT region takes the per-thread op; elsewhere scalars keep
        # reaching the existing scalar form of the vector op.
        if scalars and _runtime._in_simt_vec_func():
            if mask is not None:
                _op_error(
                    mnemonic, "mask is not supported for the per-thread scalar form"
                )
            return _emit_simt_binary(mnemonic, lhs, rhs, loc=loc)
        return vector_form(lhs, rhs, mask=mask, loc=loc)

    _binary.__name__ = mnemonic
    # Keep the vector form's documentation on the public name: the wrapper only
    # adds per-thread dispatch, so the docs still describe what callers get.
    _binary.__doc__ = vector_form.__doc__
    return _binary


exp = _make_scalar_aware_unary_op(
    "exp",
    doc="""Directory: Vector Compute / Basic Arithmetic

Description:
    Element-wise exponential on a vector (requires f16/f32).

    Parameters:
    - `operand` (`VectorSSA`): Source vector register. Required.
    - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

    Constraints:
    - Must be called inside a `@tla.kernel`-decorated kernel function.
    - Must be called inside `tla.vec.func()`; element type must be f16/f32.

    Example:
    ```python
    with tla.vec.func(mode="simd"):
        y = tla.exp(x_reg)
    ```
    """,
)
log = _make_scalar_aware_unary_op(
    "log",
    doc="""Directory: Vector Compute / Basic Arithmetic

Description:
    Element-wise logarithm on a vector (requires f16/f32).

    Parameters:
    - `operand` (`VectorSSA`): Source vector register. Required.
    - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

    Constraints:
    - Must be called inside a `@tla.kernel`-decorated kernel function.
    - Must be called inside `tla.vec.func()`; element type must be f16/f32.

    Example:
    ```python
    with tla.vec.func(mode="simd"):
        y = tla.log(x_reg)
    ```
    """,
)
sqrt = _make_scalar_aware_unary_op(
    "sqrt",
    doc="""Directory: Vector Compute / Basic Arithmetic

Description:
    Square root of a vector or of a scalar.

    Given a `VectorSSA` this is the element-wise AVE op (requires f16/f32).
    Given a float `Numeric` it is a scalar square root on the core's own scalar
    unit, available anywhere a scalar is: a `tla.cube` region, a SIMD or SIMT
    `tla.vec.func`, or the kernel body. Inside a SIMT region it becomes the
    per-thread `tla.simt_sqrt`; elsewhere it is `math.sqrt`. `x.sqrt()` on a
    float `Numeric` is the same operation.

    Parameters:
    - `operand` (`VectorSSA | Numeric`): Source vector register, or a float scalar. Required.
    - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Vector form only. Optional, default `None`.

    Constraints:
    - Must be called inside a `@tla.kernel`-decorated kernel function.
    - The vector form must be called inside `tla.vec.func()`; element type must be f16/f32.
    - The scalar form is float-only and takes no `mask`.

    Example:
    ```python
    with tla.vec.func(mode="simd"):
        y = tla.sqrt(x_reg)

    with tla.cube():
        s = tla.sqrt(scale)
    ```
    """,
)
abs = _make_scalar_aware_unary_op(
    "abs",
    doc="""Directory: Vector Compute / Basic Arithmetic

Description:
    Element-wise absolute value on a vector.

    Parameters:
    - `operand` (`VectorSSA`): Source vector register. Required.
    - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

    Constraints:
    - Must be called inside a `@tla.kernel`-decorated kernel function.
    - Must be called inside `tla.vec.func()`.

    Example:
    ```python
    with tla.vec.func(mode="simd"):
        y = tla.abs(x_reg)
    ```
    """,
)
neg = _make_unary_op(
    "neg",
    doc="""Directory: Vector Compute / Basic Arithmetic

Description:
    Element-wise negation on a vector.

    Parameters:
    - `operand` (`VectorSSA`): Source vector register. Required.
    - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

    Constraints:
    - Must be called inside a `@tla.kernel`-decorated kernel function.
    - Must be called inside `tla.vec.func()`.

    Example:
    ```python
    with tla.vec.func(mode="simd"):
        y = tla.neg(x_reg)
    ```
    """,
)


_INTERLEAVE_ELEMENT_TYPES = frozenset(
    {
        "i8",
        "i16",
        "i32",
        "i64",
        "f16",
        "bf16",
        "f32",
    }
)


@dsl_user_op
def interleave(
    src0: VectorSSA | MaskSSA,
    src1: VectorSSA | MaskSSA,
    *,
    loc: mlir_ir.Location | None = None,
) -> tuple[VectorSSA, VectorSSA] | tuple[MaskSSA, MaskSSA]:
    """Directory: Vector Compute / Data Rearrange
    Description:
        Interleave two vector registers or predicate masks lane-wise.

        Parameters:
        - `src0` (`VectorSSA` or `MaskSSA`): Even-lane input. Required.
        - `src1` (`VectorSSA` or `MaskSSA`): Odd-lane input of the same category and type. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; inputs must match type. MaskSSA supports only mask<64>, mask<128>, and mask<256>.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            lo, hi = tla.interleave(a, b)
            mask_lo, mask_hi = tla.interleave(mask0, mask1)
        ```
    """
    src0_category = _category(src0)
    src1_category = _category(src1)
    expected = ("mask_ssa", "vector_ssa")
    if src0_category not in expected:
        _require_categories("interleave", "src0", src0, expected, 0)
    if src1_category not in expected:
        _require_categories("interleave", "src1", src1, expected, 1)
    if src0_category != src1_category:
        _op_error(
            "interleave",
            "src0 and src1 must both be MaskSSA values or both be VectorSSA values",
        )
    _require_frontend_state("interleave")
    _runtime._require_enclosing_region("interleave", "vec.func")

    src0_value = _as_value(src0)
    src1_value = _as_value(src1)

    if src0_category == "mask_ssa":
        if src0_value.type != src1_value.type:
            _op_error("interleave", "MaskSSA operands must have the same type")
        lanes = _mask_ssa_type_for_mlir_value(src0_value).physical_lanes
        if lanes not in (64, 128, 256):
            _op_error(
                "interleave",
                "MaskSSA interleave supports mask<64>, mask<128>, and mask<256>; mask<32>/b64 is unsupported",
            )
        dst0, dst1 = _tla_ops_gen.interleave(
            src0_value.type, src0_value.type, src0_value, src1_value, loc=loc
        )
        return MaskSSA(dst0), MaskSSA(dst1)

    src_desc = _vector_ssa_type_for_mlir_value(src0_value)
    element_type = str(src_desc.element_type).lower()
    if element_type not in _INTERLEAVE_ELEMENT_TYPES:
        _op_error(
            "interleave",
            f"unsupported element type {src_desc.element_type}; supported types are "
            f"{', '.join(sorted(_INTERLEAVE_ELEMENT_TYPES))}",
        )

    dst0_value, dst1_value = _tla_ops_gen.interleave(
        src0_value.type,
        src0_value.type,
        src0_value,
        src1_value,
        loc=loc,
    )

    return VectorSSA(dst0_value), VectorSSA(dst1_value)


@dsl_user_op
def deinterleave(
    src0: VectorSSA | MaskSSA,
    src1: VectorSSA | MaskSSA,
    *,
    loc: mlir_ir.Location | None = None,
) -> tuple[VectorSSA, VectorSSA] | tuple[MaskSSA, MaskSSA]:
    """Directory: Vector Compute / Data Rearrange

    Description:
        Concatenate two vector registers or predicate masks, then return
        the even-position and odd-position lanes as two registers.

        Parameters:
        - `src0` (`VectorSSA` or `MaskSSA`): First half of the concatenated input. Required.
        - `src1` (`VectorSSA` or `MaskSSA`): Second half, of the same category and type. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; both vectors must match element type and lane count.
        - MaskSSA operands and results have the same type; only mask<64>, mask<128>, and mask<256> are supported. mask<32>/b64 is unsupported.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            even, odd = tla.deinterleave(a, b)
            mask_even, mask_odd = tla.deinterleave(mask0, mask1)
        ```
    """
    src0_category = _category(src0)
    src1_category = _category(src1)
    expected = ("mask_ssa", "vector_ssa")
    if src0_category not in expected:
        _require_categories("deinterleave", "src0", src0, expected, 0)
    if src1_category not in expected:
        _require_categories("deinterleave", "src1", src1, expected, 1)
    if src0_category != src1_category:
        _op_error(
            "deinterleave",
            "src0 and src1 must both be MaskSSA values or both be VectorSSA values",
        )
    _require_frontend_state("deinterleave")
    _runtime._require_enclosing_region("deinterleave", "vec.func")

    src0_value = _as_value(src0)
    src1_value = _as_value(src1)

    if src0_category == "mask_ssa":
        if src0_value.type != src1_value.type:
            _op_error("deinterleave", "MaskSSA operands must have the same type")
        lanes = _mask_ssa_type_for_mlir_value(src0_value).physical_lanes
        if lanes not in (64, 128, 256):
            _op_error(
                "deinterleave",
                "MaskSSA deinterleave supports mask<64>, mask<128>, and mask<256>; "
                "mask<32>/b64 is unsupported",
            )
        dst0, dst1 = _tla_ops_gen.deinterleave(
            src0_value.type, src0_value.type, src0_value, src1_value, loc=loc
        )
        return MaskSSA(dst0), MaskSSA(dst1)

    src_desc = _vector_ssa_type_for_mlir_value(src0_value)
    element_type = str(src_desc.element_type).lower()
    if element_type not in _INTERLEAVE_ELEMENT_TYPES:
        _op_error(
            "deinterleave",
            f"unsupported element type {src_desc.element_type}; supported types are "
            f"{', '.join(sorted(_INTERLEAVE_ELEMENT_TYPES))}",
        )

    dst0_value, dst1_value = _tla_ops_gen.deinterleave(
        src0_value.type,
        src0_value.type,
        src0_value,
        src1_value,
        loc=loc,
    )

    return VectorSSA(dst0_value), VectorSSA(dst1_value)


@dsl_user_op
def shift_left(
    source: VectorSSA,
    shift: Any,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: Vector Compute / Logical Compute
    Description:
        Element-wise left shift of a vector by per-lane or scalar shift amounts.

        Parameters:
        - `source` (`VectorSSA`): Source vector to shift. Required.
        - `shift` (`VectorSSA | Numeric | int`): Shift amount per lane. A signed
          `VectorSSA` with the same element width as `source`, or an integer
          scalar convertible to i16. Required.
        - `mask` (`MaskSSA | None`): Execution mask. Optional, default `None`
          (all lanes enabled); masked-out lanes are zeroed.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.
        - Source elements must be signed 8/16/32-bit integers (i8/i16/i32);
          64-bit and unsigned shifts are not supported.
        - Shift amounts must be non-negative.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            _tile = tla.shift_left(tile, bit_shift_reg, mask=mask)
        ```
    """
    return _emit_vector_shift("shift_left", source, shift, mask=mask, loc=loc)


@dsl_user_op
def shift_right(
    source: VectorSSA,
    shift: Any,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: Vector Compute / Logical Compute
    Description:
        Element-wise right shift of a vector by per-lane or scalar shift amounts.

        Signed source vectors use arithmetic right shift.

        Parameters:
        - `source` (`VectorSSA`): Source vector to shift. Required.
        - `shift` (`VectorSSA | Numeric | int`): Shift amount per lane. A signed
          `VectorSSA` with the same element width as `source`, or an integer
          scalar convertible to i16. Required.
        - `mask` (`MaskSSA | None`): Execution mask. Optional, default `None`
          (all lanes enabled); masked-out lanes are zeroed.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.
        - Source elements must be signed 8/16/32-bit integers (i8/i16/i32);
          64-bit and unsigned shifts are not supported.
        - Shift amounts must be non-negative.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            _tile = tla.shift_right(tile, bit_shift_reg, mask=mask)
        ```
    """
    return _emit_vector_shift("shift_right", source, shift, mask=mask, loc=loc)


@dsl_user_op
def bitwise_not(
    operand: VectorSSA | MaskSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA | VectorSSA:
    """Directory: Vector Compute / Logical Compute
    Description:
        Element-wise bitwise/logical not (Mask or Vector).

        Parameters:
        - `operand` (`VectorSSA | MaskSSA`): Source operand for bitwise/logical not. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            m2 = tla.bitwise_not(m)
        ```
    """
    return _emit_bitwise_unary(
        "bitwise_not",
        _tla_ops_gen.bitwise_not,
        operand,
        mask=mask,
        loc=loc,
    )


@dsl_user_op
def add(
    lhs: VectorSSA | Numeric | bool | int | float,
    rhs: VectorSSA | Numeric | bool | int | float,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: Vector Compute / Basic Arithmetic
    Description:
        Element-wise vector addition (supports vector–vector and vector–scalar).
        `VectorSSA` also overloads `+` / `__radd__` to this op when `mask` is not needed.

        Parameters:
        - `lhs` (`VectorSSA | Numeric | bool | int | float`): Left-hand operand. Required.
        - `rhs` (`VectorSSA | Numeric | bool | int | float`): Right-hand operand. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; supports vector–vector and vector–scalar.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            # Before: x_reg, y_reg are register vectors loaded from UB.
            z = x_reg + y_reg            # same as tla.add(x_reg, y_reg)
            z = x_reg + 1.0              # vector–scalar via __add__/__radd__
            # Masked path: build MaskSSA, then pass mask= (no operator overload).
            m = tla.create_mask(pattern=tla.mask.VL16, dtype=tla.Float16)
            z = tla.add(x_reg, y_reg, mask=m)
            # After: valid elements hold the sum; other elements stay inactive.
        ```
    """
    return _emit_vector_binary_or_scalar(
        "add",
        _tla_ops_gen.add,
        "adds",
        lhs,
        rhs,
        mask=mask,
        loc=loc,
        commutative=True,
    )


@dsl_user_op
def sub(
    lhs: VectorSSA | Numeric | bool | int | float,
    rhs: VectorSSA | Numeric | bool | int | float,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: Vector Compute / Basic Arithmetic

    Description:
        Element-wise vector subtraction.
        `VectorSSA` also overloads `-` (`__sub__`) to this op when `mask` is not needed.

        Parameters:
        - `lhs` (`VectorSSA | Numeric | bool | int | float`): Left-hand operand (minuend). Required.
        - `rhs` (`VectorSSA | Numeric | bool | int | float`): Right-hand operand (subtrahend). Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            # Before: x_reg / y_reg are source vectors.
            z = x_reg - y_reg                 # same as tla.sub(x_reg, y_reg)
            z = tla.sub(x_reg, y_reg, mask=m) # use the function when masking
            # After: z holds the difference.
        ```
    """
    return _emit_vector_binary_or_scalar(
        "sub", _tla_ops_gen.sub, "subs", lhs, rhs, mask=mask, loc=loc
    )


@dsl_user_op
def mul(
    lhs: VectorSSA | Numeric | bool | int | float,
    rhs: VectorSSA | Numeric | bool | int | float,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: Vector Compute / Basic Arithmetic

    Description:
        Element-wise vector multiplication.
        `VectorSSA` also overloads `*` / `__rmul__` to this op when `mask` is not needed.

        Parameters:
        - `lhs` (`VectorSSA | Numeric | bool | int | float`): Left-hand operand. Required.
        - `rhs` (`VectorSSA | Numeric | bool | int | float`): Right-hand operand. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            # Before: x_reg holds activations; scale may be a vector or a scalar.
            z = x_reg * y_reg      # same as tla.mul(x_reg, y_reg)
            z = x_reg * 2.0        # vector–scalar
            z = tla.mul(x_reg, y_reg, mask=m)
            # After: z holds the products.
        ```
    """
    return _emit_vector_binary_or_scalar(
        "mul",
        _tla_ops_gen.mul,
        "muls",
        lhs,
        rhs,
        mask=mask,
        loc=loc,
        commutative=True,
    )


@dsl_user_op
def _vector_max(
    lhs: VectorSSA | Numeric | bool | int | float,
    rhs: VectorSSA | Numeric | bool | int | float,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: Vector Compute / Basic Arithmetic

    Description:
        Element-wise vector maximum.

        Parameters:
        - `lhs` (`VectorSSA | Numeric | bool | int | float`): Left-hand operand. Required.
        - `rhs` (`VectorSSA | Numeric | bool | int | float`): Right-hand operand. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            z = tla.max(x_reg, y_reg)
        ```
    """
    return _emit_vector_binary_or_scalar(
        "max",
        _tla_ops_gen.max,
        "maxs",
        lhs,
        rhs,
        mask=mask,
        loc=loc,
        commutative=True,
    )


@dsl_user_op
def _vector_min(
    lhs: VectorSSA | Numeric | bool | int | float,
    rhs: VectorSSA | Numeric | bool | int | float,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: Vector Compute / Basic Arithmetic

    Description:
        Element-wise vector minimum.

        Parameters:
        - `lhs` (`VectorSSA | Numeric | bool | int | float`): Left-hand operand. Required.
        - `rhs` (`VectorSSA | Numeric | bool | int | float`): Right-hand operand. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            z = tla.min(x_reg, y_reg)
        ```
    """
    return _emit_vector_binary_or_scalar(
        "min",
        _tla_ops_gen.min,
        "mins",
        lhs,
        rhs,
        mask=mask,
        loc=loc,
        commutative=True,
    )


max = _make_scalar_aware_binary_op("max", _vector_max)
min = _make_scalar_aware_binary_op("min", _vector_min)


@dsl_user_op
def div(
    lhs: VectorSSA | Numeric | bool | int | float,
    rhs: VectorSSA | Numeric | bool | int | float,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: Vector Compute / Basic Arithmetic

    Description:
        Element-wise vector division.
        `VectorSSA` also overloads `/` (`__truediv__`) to this op when `mask` is not needed.

        Parameters:
        - `lhs` (`VectorSSA | Numeric | bool | int | float`): Left-hand operand (dividend). Required.
        - `rhs` (`VectorSSA | Numeric | bool | int | float`): Right-hand operand (divisor). Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            # Before: x_reg is the dividend; y_reg / scalar is the divisor.
            z = x_reg / y_reg                 # same as tla.div(x_reg, y_reg)
            z = tla.div(x_reg, y_reg, mask=m)
            # After: z holds the quotients.
        ```
    """
    return _emit_vector_binary_or_scalar(
        "div", _tla_ops_gen.div, "divs", lhs, rhs, mask=mask, loc=loc
    )


def _reduction_result_descriptor(
    operand_value: mlir_ir.Value,
) -> TlaVectorSSATypeDescriptor:
    operand_desc = _vector_ssa_type_for_mlir_value(operand_value)
    return TlaVectorSSATypeDescriptor(1, operand_desc.element_type)


def _emit_vector_reduce(
    operand: VectorSSA,
    kind: ReductionOp,
    *,
    mask: MaskSSA,
    init_value: Any | None = None,
    reduction_profile: Any | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    op_name = "VectorSSA.reduce"
    _require_category(op_name, "operand", operand, "vector_ssa", 0)
    _require_category(op_name, "mask", mask, "mask_ssa", 1)
    if init_value is not None:
        raise NotImplementedError(f"{op_name} only supports init_value=None")
    if reduction_profile is not None:
        raise NotImplementedError(f"{op_name} only supports reduction_profile=None")
    if not isinstance(kind, ReductionOp):
        _op_error(
            op_name,
            "invalid argument 'kind' (position 1): "
            f"expected ReductionOp, got {_type_name(kind)}",
        )
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    operand_value = _as_value(operand)
    operand_desc = _vector_ssa_type_for_mlir_value(operand_value)
    _check_reduction_element_type_supported(op_name, operand_desc.element_type)
    mask_value = _as_value(mask)
    _require_mask_matches_vector(op_name, mask_value, operand_value)
    result_desc = _reduction_result_descriptor(operand_value)
    result = _tla_ops_gen.reduce(
        result_desc.to_mlir_type(operand_value.type.context),
        operand_value,
        kind.value,
        mask=mask_value,
        loc=loc,
    )
    return VectorSSA(result)


def _vector_where(
    mask: MaskSSA,
    x: VectorSSA,
    y: VectorSSA,
    *,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: Vector Compute / Compare and Select

    Description:
        Select between two vectors under a mask.

        Parameters:
        - `mask` (`MaskSSA`): Select mask; true selects `x`, false selects `y`. Required.
        - `x` (`VectorSSA`): Value when the mask is true. Required.
        - `y` (`VectorSSA`): Value when the mask is false. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; `mask`/`x`/`y` lane layouts must match.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            z = tla.where(m, x_reg, y_reg)
        ```
    """
    _require_category("where", "mask", mask, "mask_ssa", 0)
    _require_category("where", "x", x, "vector_ssa", 1)
    _require_category("where", "y", y, "vector_ssa", 2)
    _require_frontend_state("where")
    _runtime._require_enclosing_region("where", "vec.func")
    x_value = _as_value(x)
    y_value = _as_value(y)
    mask_value = _as_value(mask)
    x_desc = _vector_ssa_type_for_mlir_value(x_value)
    y_desc = _vector_ssa_type_for_mlir_value(y_value)
    if x_desc.element_type != y_desc.element_type:
        _op_error(
            "where",
            f"y has element type {y_desc.element_type}, expected {x_desc.element_type}",
        )
    _require_mask_matches_vector("where", mask_value, x_value)
    result = _tla_ops_gen.where(
        x_value.type,
        mask_value,
        x_value,
        y_value,
        loc=loc,
    )
    return VectorSSA(result)


def _emit_simt_where(
    condition: Any, x: Any, y: Any, *, loc: mlir_ir.Location | None = None
) -> Any:
    """Per-thread select: `x` where the condition holds, `y` otherwise."""
    _require_frontend_state("where")
    if not _runtime._in_simt_vec_func():
        _op_error(
            "where",
            "on scalars is only available inside a tla.vec.func with mode='simt'; "
            "pass a MaskSSA and VectorSSA values for the whole-vector form",
        )
    cond = condition if isinstance(condition, Numeric) else as_numeric(condition)
    if not isinstance(cond, Bool):
        _op_error(
            "where",
            f"condition must be a Bool (from a comparison); got {type(cond).__name__}",
        )
    left = x if isinstance(x, Numeric) else as_numeric(x)
    right = y if isinstance(y, Numeric) else as_numeric(y)
    if type(left) is not type(right):
        _op_error(
            "where",
            f"needs both arms of the same type; got {type(left).__name__} "
            f"and {type(right).__name__}",
        )
    x_value = left.ir_value(loc=loc)
    result = mlir_ir.Operation.create(
        "tla.simt_where",
        operands=[cond.ir_value(loc=loc), x_value, right.ir_value(loc=loc)],
        results=[x_value.type],
        loc=loc,
    ).results[0]
    return type(left)(result)


@dsl_user_op
def where(mask: Any, x: Any, y: Any, *, loc: mlir_ir.Location | None = None) -> Any:
    """Directory: Vector Compute / Compare and Select

    Description:
        Select between two vectors under a mask, or between two per-thread scalars
        inside a SIMT region.

        Parameters:
        - `mask` (`MaskSSA | Bool`): Select mask; true selects `x`, false selects `y`. A MaskSSA whole-vector, or the Bool from a comparison inside a SIMT region. Required.
        - `x` (`VectorSSA | Numeric`): Value when the mask is true. Required.
        - `y` (`VectorSSA | Numeric`): Value when the mask is false. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; `mask`/`x`/`y` lane layouts must match.
        - The per-thread form requires `mode="simt"` and emits `tla.simt_where`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            z = tla.where(m, x_reg, y_reg)
        ```
    """
    if not isinstance(mask, MaskSSA) and _runtime._in_simt_vec_func():
        return _emit_simt_where(mask, x, y, loc=loc)
    return _vector_where(mask, x, y, loc=loc)


@dsl_user_op
def squeeze(
    src: VectorSSA,
    mask: MaskSSA,
    *,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: Vector Compute / Data Compress
    Description:
        Compress vector lanes kept by a mask.

        Parameters:
        - `src` (`VectorSSA`): Source vector to compress. Required.
        - `mask` (`MaskSSA`): Mask of elements to keep. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            packed = tla.squeeze(src, m)
        ```
    """
    _require_category("squeeze", "src", src, "vector_ssa", 0)
    _require_category("squeeze", "mask", mask, "mask_ssa", 1)
    _require_frontend_state("squeeze")
    _runtime._require_enclosing_region("squeeze", "vec.func")
    src_value = _as_value(src)
    mask_value = _as_value(mask)
    _require_mask_matches_vector("squeeze", mask_value, src_value)
    result = _tla_ops_gen.squeeze(
        src_value.type,
        src_value,
        mask_value,
        loc=loc,
    )
    return VectorSSA(result)


def _emit_bitwise_unary(
    op_name: str,
    emitter: Any,
    operand: Any,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA | VectorSSA:
    operand_category = _category(operand)
    expected = ("mask_ssa", "vector_ssa")
    if operand_category not in expected:
        _require_categories(op_name, "operand", operand, expected, 0)
    if mask is not None:
        _require_category(op_name, "mask", mask, "mask_ssa", 1)
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    operand_value = _as_value(operand)
    mask_value = _as_value(mask) if mask is not None else None
    if mask_value is not None:
        if operand_category == "mask_ssa":
            if mask_value.type != operand_value.type:
                _op_error(
                    op_name, "optional mask must have the same MaskSSA type as operand"
                )
        else:
            _require_mask_matches_vector(op_name, mask_value, operand_value)
    if operand_category == "mask_ssa":
        return MaskSSA(
            emitter(operand_value.type, operand_value, mask=mask_value, loc=loc)
        )

    result_desc = _vector_ssa_type_for_mlir_value(operand_value)
    element_type = str(result_desc.element_type)
    if element_type not in _BITWISE_UNARY_TYPES:
        _op_error(
            op_name,
            f"tla.{op_name} requires f16/f32 or i8/i16/i32 element type, "
            f"got {element_type}",
        )
    result = emitter(
        operand_value.type,
        operand_value,
        mask=mask_value,
        loc=loc,
    )
    return VectorSSA(result)


def _emit_bitwise_binary(
    op_name: str,
    emitter: Any,
    src0_reg: Any,
    src1_reg: Any,
    *,
    mask: Any | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA | VectorSSA:
    src0_category = _category(src0_reg)
    src1_category = _category(src1_reg)
    expected = ("mask_ssa", "vector_ssa")
    if src0_category not in expected:
        _require_categories(op_name, "src0_reg", src0_reg, expected, 0)
    if src1_category not in expected:
        _require_categories(op_name, "src1_reg", src1_reg, expected, 1)
    if src0_category != src1_category:
        _op_error(
            op_name,
            "src0_reg and src1_reg must both be MaskSSA values or both be VectorSSA values",
        )
    if mask is not None:
        _require_category(op_name, "mask", mask, "mask_ssa", 2)
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    src0_value = _as_value(src0_reg)
    src1_value = _as_value(src1_reg)
    if src0_category == "mask_ssa":
        if src1_value.type != src0_value.type:
            _op_error(
                op_name,
                f"src1_reg has type {src1_value.type}, expected {src0_value.type}",
            )
    else:
        src0_desc = _vector_ssa_type_for_mlir_value(src0_value)
        src1_desc = _vector_ssa_type_for_mlir_value(src1_value)
        if src1_desc.element_type != src0_desc.element_type:
            _op_error(
                op_name,
                f"src1_reg has element type {src1_desc.element_type}, expected "
                f"{src0_desc.element_type}",
            )
    mask_value = _as_value(mask) if mask is not None else None
    if mask_value is not None:
        if src0_category == "mask_ssa":
            if mask_value.type != src0_value.type:
                _op_error(
                    op_name, "optional mask must have the same MaskSSA type as operands"
                )
        else:
            _require_mask_matches_vector(op_name, mask_value, src0_value)
    result = emitter(
        src0_value.type,
        src0_value,
        src1_value,
        mask=mask_value,
        loc=loc,
    )
    if src0_category == "mask_ssa":
        return MaskSSA(result)

    return VectorSSA(result)


@dsl_user_op
def cmp(
    lhs: VectorSSA,
    rhs: VectorSSA | Numeric | bool | int | float,
    mode: str,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA:
    """Directory: Vector Compute / Compare and Select
    Description:
        Element-wise compare; returns a MaskSSA.

        Parameters:
        - `lhs` (`VectorSSA`): Left-hand compare operand. Required.
        - `rhs` (`VectorSSA | Numeric | bool | int | float`): Right-hand compare operand (scalar or vector). Required.
        - `mode` (`str`): Compare mode, e.g. `'eq'` / `'lt'` / `'gt'`. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; `mode` must be a supported compare mnemonic.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            m = tla.cmp(x_reg, y_reg, mode="lt")
        ```
    """
    mode = str(mode).lower()
    if mode not in _MASK_CMP_MODES:
        _op_error(
            "cmp",
            f"mode must be one of {_MASK_CMP_MODES}; got {mode!r}",
        )
    _require_category("cmp", "lhs", lhs, "vector_ssa", 0)
    rhs_category = _category(rhs)
    rhs_num = _as_vector_scalar_numeric(rhs)
    if rhs_category != "vector_ssa" and rhs_num is None:
        _op_error(
            "cmp",
            f"invalid argument 'rhs' (position 1): expected vector or scalar, got {_type_name(rhs)}",
        )
    if mask is not None:
        _require_category("cmp", "mask", mask, "mask_ssa", 2)
    _require_frontend_state("cmp")
    _runtime._require_enclosing_region("cmp", "vec.func")
    lhs_value = _as_value(lhs)
    lhs_desc = _vector_ssa_type_for_mlir_value(lhs_value)
    if rhs_category == "vector_ssa":
        rhs_value = _as_value(rhs)
    else:
        assert rhs_num is not None
        rhs_value = _numeric_ir_value_for_element_type(
            "cmp", rhs_num, lhs_desc.element_mlir_type(lhs_value.type.context), loc=loc
        )
        _check_compare_element_type_supported("cmp", lhs_desc.element_type)
    mask_ty = _mask_ssa_type_for_element_type(lhs_desc.element_type).to_mlir_type(
        lhs_value.type.context
    )
    mask_value = _as_value(mask) if mask is not None else None
    if rhs_category == "vector_ssa":
        rhs_desc = _vector_ssa_type_for_mlir_value(rhs_value)
        if rhs_desc.element_type != lhs_desc.element_type:
            _op_error(
                "cmp",
                f"rhs has element type {rhs_desc.element_type}, expected "
                f"{lhs_desc.element_type}",
            )
    if mask_value is not None:
        _require_mask_matches_vector("cmp", mask_value, lhs_value)
    return MaskSSA(
        _tla_ops_gen.cmp(mask_ty, lhs_value, rhs_value, mode, mask=mask_value, loc=loc)
    )


@dsl_user_op
def bitwise_and(
    src0_reg: VectorSSA | MaskSSA,
    src1_reg: VectorSSA | MaskSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA | VectorSSA:
    """Directory: Vector Compute / Logical Compute

    Description:
        Element-wise bitwise and (Mask or Vector).

        Parameters:
        - `src0_reg` (`VectorSSA | MaskSSA`): Left-hand bitwise-and operand. Required.
        - `src1_reg` (`VectorSSA | MaskSSA`): Right-hand bitwise-and operand. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            m3 = tla.bitwise_and(m0, m1)
        ```
    """
    return _emit_bitwise_binary(
        "bitwise_and",
        _tla_ops_gen.bitwise_and,
        src0_reg,
        src1_reg,
        mask=mask,
        loc=loc,
    )


@dsl_user_op
def bitwise_or(
    src0_reg: VectorSSA | MaskSSA,
    src1_reg: VectorSSA | MaskSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA | VectorSSA:
    """Directory: Vector Compute / Logical Compute

    Description:
        Element-wise bitwise or (Mask or Vector).

        Parameters:
        - `src0_reg` (`VectorSSA | MaskSSA`): Left-hand bitwise-or operand. Required.
        - `src1_reg` (`VectorSSA | MaskSSA`): Right-hand bitwise-or operand. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            m3 = tla.bitwise_or(m0, m1)
        ```
    """
    return _emit_bitwise_binary(
        "bitwise_or",
        _tla_ops_gen.bitwise_or,
        src0_reg,
        src1_reg,
        mask=mask,
        loc=loc,
    )


@dsl_user_op
def bitwise_xor(
    src0_reg: VectorSSA | MaskSSA,
    src1_reg: VectorSSA | MaskSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA | VectorSSA:
    """Directory: Vector Compute / Logical Compute

    Description:
        Element-wise bitwise xor (Mask or Vector).

        Parameters:
        - `src0_reg` (`VectorSSA | MaskSSA`): Left-hand bitwise-xor operand. Required.
        - `src1_reg` (`VectorSSA | MaskSSA`): Right-hand bitwise-xor operand. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            m3 = tla.bitwise_xor(m0, m1)
        ```
    """
    return _emit_bitwise_binary(
        "bitwise_xor",
        _tla_ops_gen.bitwise_xor,
        src0_reg,
        src1_reg,
        mask=mask,
        loc=loc,
    )


def _require_reg_slot_reachable(src_token: str, dst_token: str, slot: RegSlot) -> None:
    """Refuse a reg_slot the lowering would quietly ignore.

    Slots TWO and THREE name a pack quarter, and only an integer cast of 4x
    width or more, or a direct fp8<->f32 cast, reads one (pp0..pp3). Every
    other cast carries an even/odd part, where mapCastPart folds TWO and THREE
    onto PART_EVEN -- so accepting them would hand back slot ZERO's lanes with
    no diagnostic.

    The fp8<->f32 instruction is the one float exception: it is a 4x-width
    conversion and uses the same pp0..pp3 selector family as i8<->i32.
    """
    if slot not in (RegSlot.TWO, RegSlot.THREE):
        return
    src_bits = _CAST_TOKEN_BITS.get(src_token, 0)
    dst_bits = _CAST_TOKEN_BITS.get(dst_token, 0)
    int_pair = src_token.startswith("i") and dst_token.startswith("i")
    ratio = (
        _builtins.max(src_bits, dst_bits) // _builtins.min(src_bits, dst_bits)
        if src_bits and dst_bits
        else 0
    )
    fp8_f32_pair = (src_token in _CAST_FP8_DTYPES and dst_token == "f32") or (
        src_token == "f32" and dst_token in _CAST_FP8_DTYPES
    )
    if (int_pair and ratio >= 4) or fp8_f32_pair:
        return
    _op_error(
        "cast",
        f"invalid reg_slot {slot} for cast '{src_token}'->'{dst_token}': slots "
        f"TWO and THREE select a pack quarter, which requires a 4x integer cast "
        f"or a direct fp8<->f32 cast; this conversion carries an even/odd part, "
        f"where they would silently behave as ZERO",
    )


@dsl_user_op
def gather(
    x: Tensor,
    y: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: Vector Compute / Discrete and Aggregate
    Description:
        Gather vector values from a tile/table by indices.

        Parameters:
        - `x` (`Tensor`): Source tile / table to gather from. Required.
        - `y` (`VectorSSA`): Index vector register. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; the source tensor must reside in UB.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            vals = tla.gather(ub_tile, idx_reg)
        ```
    """
    _require_category("gather", "x", x, "tensor", 0)
    _require_category("gather", "y", y, "vector_ssa", 1)
    _require_frontend_state("gather")
    _runtime._require_enclosing_region("gather", "vec.func")
    x_value = _as_value(x)
    y_value = _as_value(y)
    x_desc = _tla_tensor_descriptor_from_type_or_value(x_value)
    y_desc = _vector_ssa_type_for_mlir_value(y_value)

    # validate x addrspace is ub
    if x_desc.addrspace.lower() != "ub":
        _op_error(
            "gather",
            f"invalid argument 'x' (position 0): expected addrspace ub, got {x_desc.addrspace}",
        )

    # validate x element type
    _GATHER_SUPPORTED_X_ELEM_TYPES = frozenset(
        {
            "i1",
            "i8",
            "u8",
            "i16",
            "u16",
            "i32",
            "u32",
            "i64",
            "u64",
            "f16",
            "bf16",
            "f32",
        }
    )
    if x_desc.element_type.lower() not in _GATHER_SUPPORTED_X_ELEM_TYPES:
        _op_error(
            "gather",
            f"invalid argument 'x' (position 0): unsupported element type "
            f"{x_desc.element_type}; supported types are "
            f"{', '.join(sorted(_GATHER_SUPPORTED_X_ELEM_TYPES))}",
        )

    # Gather hardware consumes one i32 index per lane.
    if y_desc.element_type.lower() != "i32":
        _op_error(
            "gather",
            f"invalid argument 'y' (position 1): gather indices must be i32, got "
            f"{y_desc.element_type}",
        )
    if mask is not None:
        _require_category("gather", "mask", mask, "mask_ssa", 2)
    mask_value = _as_value(mask) if mask is not None else None
    result_desc = _vector_ssa_type_from_tensor_descriptor(x_desc)
    if mask_value is not None:
        expected_mask = _mask_ssa_type_for_element_type(result_desc.element_type)
        actual_mask = _mask_ssa_type_for_mlir_value(mask_value)
        if actual_mask.physical_lanes != expected_mask.physical_lanes:
            _op_error(
                "gather",
                f"mask has {actual_mask.physical_lanes} predicate lanes, expected "
                f"{expected_mask.physical_lanes} for {result_desc.element_type} VectorSSA",
            )
    result = _tla_ops_gen.gather(
        result_desc.to_mlir_type(x_value.type.context),
        x_value,
        y_value,
        mask=mask_value,
        loc=loc,
    )
    return VectorSSA(result)


@dsl_user_op
def arch_block_idx(*, loc: mlir_ir.Location | None = None) -> Int32:
    """Return block index in Tla execution model (``Int32``)."""
    _require_frontend_state("arch.block_idx")
    i32 = mlir_ir.IntegerType.get_signless(32)
    value = _tla_ops_gen.arch_block_idx(i32, loc=loc)
    return Int32(value)


@dsl_user_op
def arch_sub_block_idx(*, loc: mlir_ir.Location | None = None) -> Int32:
    """Return sub-block index in Tla execution model (``Int32``)."""
    _require_frontend_state("arch.sub_block_idx")
    i32 = mlir_ir.IntegerType.get_signless(32)
    value = _tla_ops_gen.arch_sub_block_idx(i32, loc=loc)
    return Int32(value)


@dsl_user_op
def arch_thread_idx(
    *, loc: mlir_ir.Location | None = None
) -> tuple[Int32, Int32, Int32]:
    """Return the SIMT thread index ``(x, y, z)`` inside the thread block."""
    _require_frontend_state("arch.thread_idx")
    if not _runtime._in_simt_vec_func():
        _op_error(
            "arch.thread_idx",
            "is only available inside a tla.vec.func with mode='simt'",
        )
    i32 = mlir_ir.IntegerType.get_signless(32)
    values = _tla_ops_gen.arch_thread_idx(i32, i32, i32, loc=loc)
    return (Int32(values[0]), Int32(values[1]), Int32(values[2]))


@dsl_user_op
def arch_sync_threads(*, loc: mlir_ir.Location | None = None) -> None:
    """Barrier across the threads of the enclosing SIMT ``tla.vec.func``."""
    _require_frontend_state("arch.sync_threads")
    if not _runtime._in_simt_vec_func():
        _op_error(
            "arch.sync_threads",
            "is only available inside a tla.vec.func with mode='simt'",
        )
    _tla_ops_gen.arch_sync_threads(loc=loc)


@dsl_user_op
def arch_block_num(*, loc: mlir_ir.Location | None = None) -> Int32:
    """Return the number of blocks (AI cores) in the launch (``Int32``).

    The grid extent for the current kernel launch. For the per-block thread
    extents inside a SIMT region see :func:`arch_thread_block_dim`.
    """
    _require_frontend_state("arch.block_num")
    i32 = mlir_ir.IntegerType.get_signless(32)
    value = _tla_ops_gen.arch_block_num(i32, loc=loc)
    return Int32(value)


@dsl_user_op
def arch_thread_block_dim(
    *, loc: mlir_ir.Location | None = None
) -> tuple[Int32, Int32, Int32]:
    """Return the enclosing thread block's ``(x, y, z)`` extents (SIMT only)."""
    _require_frontend_state("arch.thread_block_dim")
    if not _runtime._in_simt_vec_func():
        _op_error(
            "arch.thread_block_dim",
            "is the SIMT thread-block geometry and is only available inside a "
            "tla.vec.func with mode='simt'; use tla.arch.block_num() for the "
            "number of blocks in the launch",
        )
    i32 = mlir_ir.IntegerType.get_signless(32)
    values = _tla_ops_gen.arch_thread_block_dim(i32, i32, i32, loc=loc)
    return (Int32(values[0]), Int32(values[1]), Int32(values[2]))


def arch_get_capacity_in_bytes(mem_scope: AddressSpace) -> int:
    """Directory: System Variable Access
    Description:
        Return the byte size of an on-chip memory space (l1 / l0a / l0b / l0c / ub )
        for the compile target architecture.

    Parameters:
    - `mem_scope`: the memory-scope token to query. Only ``tla.AddressSpace`` tokens are
      accepted: `tla.AddressSpace.l1`, `tla.AddressSpace.l0a`, `tla.AddressSpace.l0b`, `tla.AddressSpace.l0c`,
      `tla.AddressSpace.ub`.

    Returns:
    - `int`: capacity in bytes (e.g. `512 * 1024` for `tla.AddressSpace.l1` on
      3510/Ascend950).

    Constraints:
    - Only on-chip `AddressSpace` members are accepted; `AddressSpace.generic` /
      `AddressSpace.gm` (global memory has no fixed on-chip capacity).
    - This is a host-time constant query: it is legal both outside and inside a
      `@tla.kernel`-decorated kernel function. Inside a kernel it is evaluated at
      trace time and folds into a compile-time constant (no runtime op is emitted).

    Example:
    ```python
    l1_bytes = tla.arch.get_capacity_in_bytes(tla.AddressSpace.l1)    # 512 * 1024 (host side)
    ub_bytes = tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub)    # 256 * 1024

    @tla.kernel
    def k(...):
        # kernel side: folds to the same constant
        tiles = tla.arch.get_capacity_in_bytes(tla.AddressSpace.l1) // block_bytes
    ```
    """
    from .base_dsl.arch import get_localmem_capacity_bytes

    _ON_CHIP_SCOPES: tuple[AddressSpace, ...] = (
        AddressSpace.l1,
        AddressSpace.l0a,
        AddressSpace.l0b,
        AddressSpace.l0c,
        AddressSpace.ub,
    )

    if not isinstance(mem_scope, AddressSpace) or mem_scope not in _ON_CHIP_SCOPES:
        supported = ", ".join(
            f"tla.AddressSpace.{scope.name.upper()}" for scope in _ON_CHIP_SCOPES
        )
        _op_error(
            "arch.get_capacity_in_bytes",
            f"expected a tla.arch memory-scope token ({supported}); got "
            f"{_type_name(mem_scope)}",
        )
    return get_localmem_capacity_bytes(mem_scope.name.upper())


@dsl_user_op
def allocate(
    shape: ShapeLike,
    dtype: type[Numeric],
    mem_scope: AddressSpace,
    byte_alignment: int,
    *,
    loc: mlir_ir.Location | None = None,
) -> Pointer:
    """Directory: Resource Management
    Description:
        Allocate local memory and return a typed pointer.

        Parameters:
        - `shape` (`ShapeLike`): Shape of the allocation. Required.
        - `dtype` (`type[Numeric]`): Element numeric type. Required.
        - `mem_scope` (`AddressSpace`): Address space (e.g. L1 / UB). Required.
        - `byte_alignment` (`int`): Byte alignment requirement. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - `mem_scope` must be an on-chip address space (l1/l0a/l0b/l0c/ub), not gm/generic; `shape` must be fully static.

        Example:
        ```python
        ptr = tla.allocate(
            shape=(256, 128),
            dtype=tla.Float16,
            mem_scope=tla.AddressSpace.ub,
            byte_alignment=32,
        )
        ```
    """
    _require_frontend_state("allocate")
    dtype, element_bits = _require_allocation_dtype("allocate", dtype)
    align = _require_byte_alignment("allocate", byte_alignment, 3)
    addr_token = _require_pointer_addrspace("allocate", mem_scope, 2)
    if mem_scope in (AddressSpace.generic, AddressSpace.gm):
        _op_error(
            "allocate",
            "invalid argument 'mem_scope' (position 2): expected on-chip AddressSpace "
            "(l1, l0a, l0b, l0c, ub)",
        )
    size_bytes = _static_allocation_size_bytes("allocate", shape, dtype, element_bits)

    ctx = loc.context if loc is not None else mlir_ir.Context.current
    ptr_ty = PtrType.get(dtype.mlir_type(ctx), addr_token, align, context=ctx)
    i64_ty = mlir_ir.IntegerType.get_signless(64, context=ctx)
    op = mlir_ir.Operation.create(
        "tla.alloc_ptr",
        operands=[],
        results=[ptr_ty],
        attributes={
            "size_bytes": mlir_ir.IntegerAttr.get(i64_ty, size_bytes),
        },
        loc=loc,
    )
    return _Pointer(op.results[0], alloc_size_bytes=size_bytes)


@dsl_user_op
def arch_get_dyn_ub(
    dtype: type[Numeric],
    byte_alignment: int | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> Pointer:
    """Directory: System Variable Access
    Description:
        Retrieve a pointer to the dynamic UB allocation — UB whose size is set
        at launch instead of compiled in.

        Static `tla.allocate` buffers sit at the bottom of UB; this region
        begins just above them and the result pointer is offset appropriately.
        How much of it is backed is set per launch with `ub=`.
        The base is a compile-time constant, so only the extent is dynamic.

        Ask for what the kernel uses, not for everything available. The launch
        hands over exactly the requested extent, and on a SIMT kernel the UB
        left unclaimed becomes Data Cache: measured on Ascend950PR, a
        global-memory-bound SIMT kernel that claims the whole buffer runs 1.62x
        slower than the same kernel claiming only the 2 KB it uses.

        Parameters:
        - `dtype` (`type[Numeric]`): Pointee element type. Required.
        - `byte_alignment` (`int | None`): Pointer alignment in bytes, spelled
          as `tla.allocate` spells it. Optional; defaults to the element type's
          width.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - UB only; no other memory has a launch-time size.
        - At most one region per kernel: it has one base, so a second call would
          alias the first. Carve it up with pointer arithmetic instead.
        - The extent is not bounds-checked in the kernel. Going past what the
          launch declared corrupts UB rather than reliably faulting.

        Example:
        ```python
        base = tla.arch.get_dyn_ub(tla.Float32, byte_alignment=256)
        lo = tla.make_tensor(base, layout)
        hi = tla.make_tensor(base + HALF, layout)
        ...
        artifact(out, block_num=1, ub=64 * 1024)
        ```
    """
    _require_frontend_state("arch.get_dyn_ub")
    dtype, element_bits = _require_allocation_dtype("arch.get_dyn_ub", dtype)
    if byte_alignment is None:
        # The helper answers in bits. A sub-byte type packs several elements into
        # a byte, so the smallest alignment it can be given is one byte.
        # Not `max(...)`: this module rebinds `max` to the tla vector op above,
        # so the builtin is not in scope here.
        byte_alignment = element_bits // 8 or 1
    align = _require_byte_alignment("arch.get_dyn_ub", byte_alignment, 1)

    ctx = loc.context if loc is not None else mlir_ir.Context.current
    ptr_ty = PtrType.get(dtype.mlir_type(ctx), str(AddressSpace.ub), align, context=ctx)
    op = mlir_ir.Operation.create(
        "tla.dynamic_ub_base", operands=[], results=[ptr_ty], loc=loc
    )
    # No alloc_size_bytes: the extent is not known here, and claiming one would
    # let the scratch planner believe it owns bytes it has not reserved.
    return _Pointer(op.results[0])


@dsl_user_op
def make_ptr(
    dtype: type[Numeric] | None,
    value: int | mlir_ir.Value | Numeric,
    mem_space: AddressSpace = AddressSpace.gm,
    *,
    assumed_align: int | None = None,
    loc: mlir_ir.Location | None = None,
) -> Pointer:
    """Directory: Basic Data Types and Operations

    Description:
        Build a `!tla.ptr` from an address value.

        Parameters:
        - `dtype` (`type[Numeric] | None`): Pointee element type; `None` means
          `Int8`. Optional, default `None`.
        - `value` (`int | mlir_ir.Value | Numeric`): Address value (int, MLIR Value, or Numeric). Required.
        - `mem_space` (`AddressSpace`): Address space of the pointer. Optional, default `AddressSpace.gm`.
        - `assumed_align` (`int | None`): Assumed alignment in bytes. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Integer address bit-width must match the target `mem_space`.

        Example:
        ```python
        ptr = tla.make_ptr(tla.Float16, addr, mem_space=tla.AddressSpace.gm)
        ```
    """
    _require_frontend_state("make_ptr")

    if dtype is not None and (
        not isinstance(dtype, type) or not issubclass(dtype, Numeric) or not dtype.dtype
    ):
        _op_error("make_ptr", f"expects dtype to be a type of Numeric, but got {dtype}")
    dt = Int8 if dtype is None else dtype
    ctx = loc.context if loc is not None else mlir_ir.Context()
    pointee = dt.mlir_type(ctx)
    addr_token = _require_pointer_addrspace("make_ptr", mem_space, 2)
    bytes_per_elt = _builtins.max(1, int(dt.width) // 8)
    align = assumed_align if assumed_align is not None else bytes_per_elt
    if bytes_per_elt % align != 0 and align % bytes_per_elt != 0:
        _op_error(
            "make_ptr",
            f"element size {bytes_per_elt} is incompatible with assumed_align={align}",
        )
    if align <= 0 or (align & (align - 1)) != 0:
        _op_error("make_ptr", "assumed_align must be a positive power of 2")
    out_ptr_ty = PtrType.get(pointee, addr_token, align, context=ctx)
    addr_ssa = _coerce_inttoptr_address(addr_token, value, loc)
    return _Pointer(_tla_ops_gen.inttoptr(out_ptr_ty, addr_ssa, loc=loc, ip=None))


@dsl_user_op
def recast_ptr(
    ptr: Pointer,
    *,
    dtype: type[Numeric],
    loc: mlir_ir.Location | None = None,
) -> Pointer:
    """Directory: Basic Data Types and Operations

    Description:
        Reinterpret a `!tla.ptr` element type only (no swizzle).

        Parameters:
        - `ptr` (`Pointer`): Pointer to reinterpret. Required.
        - `dtype` (`type[Numeric]`): New element type. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Changes the logical element type only; address and swizzle are unchanged.

        Example:
        ```python
        ptr_f32 = tla.recast_ptr(ptr_f16, dtype=tla.Float32)
        ```
    """
    _require_frontend_state("recast_ptr")
    _require_category("recast_ptr", "ptr", ptr, "pointer", 0)
    if not isinstance(dtype, type) or not issubclass(dtype, Numeric) or not dtype.dtype:
        _op_error(
            "recast_ptr", f"expects dtype to be a type of Numeric, but got {dtype}"
        )
    ctx = loc.context if loc is not None else mlir_ir.Context()
    alloc_size_bytes = getattr(ptr, "_alloc_size_bytes", None)
    p = _coerce_pointer_arg(ptr)
    src_ty = p._ptr_ty
    new_pointee = dtype.mlir_type(ctx)
    out_ptr_ty = PtrType.get(
        new_pointee, src_ty.addrspace, src_ty.alignment, context=ctx
    )
    return _Pointer(
        _tla_ops_gen.recast_ptr(out_ptr_ty, p.value, loc=loc, ip=None),
        alloc_size_bytes=alloc_size_bytes,
    )


def _emit_tensor_ptr(
    source: mlir_ir.Value, loc: mlir_ir.Location | None = None
) -> _Pointer:
    """Emit ``tla.tensor_ptr`` extracting the backing ``!tla.ptr`` of a tensor value.

    Shared by :meth:`catlass.tla.tensor._Tensor.ptr` and the execution-mode
    runtime argument proxy so kernel arguments support ``arg.ptr`` the same way
    as frontend ``_Tensor`` values.
    """
    _require_frontend_state("tensor_ptr")
    ptr_ty = _tla_type_bridge.tensor_ptr_type_get(source.type)
    op = mlir_ir.Operation.create(
        "tla.tensor_ptr",
        operands=[source],
        results=[ptr_ty],
        loc=loc,
    )
    return _Pointer(op.results[0])


_require_generated("tile_view")
_require_generated("copy")
_require_generated("load")
_require_generated("flag")
_require_generated("cross_flag")
_require_generated("cross_core_set_flag")
_require_generated("cross_core_wait_flag")
_require_generated("set_flag")
_require_generated("store")
_require_generated("wait_flag")
_require_generated("pipe_barrier")
_require_generated("mutex")
_require_generated("mutex_lock")
_require_generated("mutex_unlock")
_require_generated("cube")
_require_generated("vector")
_require_generated("mmad")
_require_generated("mmad_mx")
_require_generated("add")
_require_generated("adds")
_require_generated("sub")
_require_generated("subs")
_require_generated("mul")
_require_generated("muls")
_require_generated("maxs")
_require_generated("mins")
_require_generated("div")
_require_generated("shift_left")
_require_generated("shift_right")
_require_generated("shift_lefts")
_require_generated("shift_rights")
_require_generated("where")
_require_generated("squeeze")
_require_generated("bitwise_not")
_require_generated("bitwise_and")
_require_generated("bitwise_or")
_require_generated("bitwise_xor")
_require_generated("divs")
_require_generated("reduce")
_require_generated("interleave")
_require_generated("deinterleave")
for _unary_op_name in ("exp", "log", "sqrt", "abs", "neg"):
    _require_generated(_unary_op_name)
_require_generated("cmp")
_require_generated("arch_block_idx")
_require_generated("arch_sub_block_idx")
_require_generated("arch_block_num")
_require_generated("arch_thread_block_dim")
_require_generated("arch_thread_idx")
_require_generated("simt_add")
_require_generated("simt_sub")
_require_generated("simt_mul")
_require_generated("simt_div")
_require_generated("simt_abs")
_require_generated("simt_exp")
_require_generated("simt_sqrt")
_require_generated("simt_pow")
_require_generated("simt_load")
_require_generated("simt_store")
_require_generated("arch_sync_threads")
_require_generated("inttoptr")
_require_generated("recast_ptr")

arch = _Namespace()
arch.__doc__ = """Directory: System Variable Access
Description:
Architecture attribute group under `tla.arch`: layout tags, pipe identifiers,
on-chip memory-scope tokens, and block / SIMT helpers.

Parameters:
- Layout tags (`_LayoutTag`, used by `make_layout` / `make_tensor` /
  `make_tensor_like`): `RowMajor`, `ColumnMajor`, `zN`, `nZ`,
  `L0Clayout`, `zNUnAlign`.
- Pipe identifiers (used by `flag` / `pipe_barrier` / `mutex_*` /
  cross-core sync): `SCALAR`, `VECTOR`, `CUBE`, `MTE1`, `MTE2`, `MTE3`, `FIX`.
- Memory-scope tokens (used by `local_mem_bar` and related APIs): `L1`,
  `L0A`, `L0B`, `L0C`, `UB`.
- Callables (return `Int32` or `tuple[Int32, Int32, Int32]` as noted):
  - `block_idx()`: Block index for the current AI core in the launch.
  - `block_num()`: Number of blocks (AI cores) in the launch.
  - `sub_block_idx()`: Sub-block index within the current block.
  - `thread_idx()`: SIMT thread index `(x, y, z)` inside the thread block
    (only inside `tla.vec.func(mode="simt")`).
  - `thread_block_dim()`: SIMT thread-block extents `(x, y, z)`
    (only inside `tla.vec.func(mode="simt")`).
  - `sync_threads()`: Barrier across threads of the enclosing SIMT
    `tla.vec.func` (only inside `mode="simt"`).
  - `get_capacity_in_bytes(mem_scope)`: Byte capacity of an on-chip memory
    space for the compile target. Takes a `tla.AddressSpace` token
    (`tla.AddressSpace.l1` / `l0a` / `l0b` / `l0c` / `ub`). Returns a plain
    `int`; valid on host and inside a kernel (folds to a constant).
  - `get_dyn_ub(dtype, byte_alignment=None)`: Pointer to the dynamic UB
    allocation -- UB whose size the launch chooses, rather than the compiler.
    The region starts above every `tla.allocate`, and `ub=` at launch says
    how much of it is backed. At most one per kernel: carve it up
    with pointer arithmetic. Ask for what the kernel uses -- on SIMT the UB
    left unclaimed becomes Data Cache.

Constraints:
- Layout tags / pipe identifiers / memory-scope tokens are ordinary attributes
  on the `tla.arch` object (Python has no C++-style namespace); they do not
  emit compute ops by themselves.
- `block_idx` / `block_num` / `sub_block_idx` / `thread_idx` /
  `thread_block_dim` / `sync_threads` are callables and must be used inside a
  `@tla.kernel`-decorated kernel function.
- `thread_idx` / `thread_block_dim` / `sync_threads` additionally require
  nesting inside `tla.vec.func(mode="simt")`.

Example:
```python
# Layout tag for make_layout / make_tensor:
tag = tla.arch.RowMajor
# Pipe id for flag / barrier / mutex:
pipe = tla.arch.MTE2
# Runtime block helpers:
bid = tla.arch.block_idx()
nblocks = tla.arch.block_num()
# On-chip memory capacity (host or kernel side):
l1_bytes = tla.arch.get_capacity_in_bytes(tla.AddressSpace.l1)
ub_bytes = tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub)
```
"""
arch._set("block_idx", arch_block_idx)
arch._set("sub_block_idx", arch_sub_block_idx)
arch._set("block_num", arch_block_num)
arch._set("thread_block_dim", arch_thread_block_dim)
arch._set("thread_idx", arch_thread_idx)
arch._set("sync_threads", arch_sync_threads)
arch._set("get_capacity_in_bytes", arch_get_capacity_in_bytes)
arch._set("get_dyn_ub", arch_get_dyn_ub)
arch._set("SCALAR", _runtime.pipes.SCALAR)
arch._set("VECTOR", _runtime.pipes.VECTOR)
arch._set("CUBE", _runtime.pipes.CUBE)
arch._set("MTE1", _runtime.pipes.MTE1)
arch._set("MTE2", _runtime.pipes.MTE2)
arch._set("MTE3", _runtime.pipes.MTE3)
arch._set("FIX", _runtime.pipes.FIX)
arch._set("zN", _LayoutTag("zN"))
arch._set("nZ", _LayoutTag("nZ"))
arch._set("zZMxScale", _LayoutTag("zZMxScale"))
arch._set("nNMxScale", _LayoutTag("nNMxScale"))
arch._set("RowMajorMxScaleA", _LayoutTag("RowMajorMxScaleA"))
arch._set("ColMajorMxScaleA", _LayoutTag("ColMajorMxScaleA"))
arch._set("RowMajorMxScaleB", _LayoutTag("RowMajorMxScaleB"))
arch._set("ColMajorMxScaleB", _LayoutTag("ColMajorMxScaleB"))
arch._set("RowMajor", _LayoutTag("RowMajor"))
arch._set("ColumnMajor", _LayoutTag("ColumnMajor"))
arch._set("L0Clayout", _LayoutTag("L0Clayout"))
arch._set("zNUnAlign", _LayoutTag("zNUnAlign"))

vec = _Namespace()
vec._set("func", _vec_func)


class _MaskPattern:
    """A fixed ``tla.mask`` pattern token (e.g. ``tla.mask.ALL``, ``tla.mask.VL8``).

    Pass it to ``tla.create_mask(pattern=...)`` to materialize a ``MaskSSA``;
    ops that take ``mask=`` accept only a ``MaskSSA`` (from ``tla.create_mask``
    or ``tla.update_mask``), not a pattern token directly.
    """

    __slots__ = ("_token",)

    def __init__(self, token: str) -> None:
        self._token = token

    def __repr__(self) -> str:
        return f"tla.mask.{self._token}"


# AVE pge patterns exposed under tla.mask.* attributes.
_MASK_PATTERN_TOKENS = (
    "ALL",
    "ALLF",
    "VL1",
    "VL2",
    "VL3",
    "VL4",
    "VL8",
    "VL16",
    "VL32",
    "VL64",
    "VL128",
    "M3",
    "M4",
    "H",
    "Q",
)


def _mask_elem_type(
    op_name: str, dtype: Any, loc: mlir_ir.Location | None
) -> mlir_ir.Type:
    """Resolve a mask ``dtype`` (Numeric type or mlir.ir.Type) to an element type.

    The element width fixes the mask lane count (256 bytes / dtype size).
    """
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    if isinstance(dtype, mlir_ir.Type):
        return dtype
    if isinstance(dtype, type) and issubclass(dtype, Numeric) and dtype.dtype:
        return dtype.mlir_type(ctx)
    _op_error(
        op_name,
        f"expects dtype to be a Numeric type (e.g. tla.Float32) or mlir.ir.Type, got {dtype}",
    )


@dsl_user_op
def create_mask(
    *,
    pattern: _MaskPattern | str | None = None,
    dtype: DTypeLike = Float32,
    loc: mlir_ir.Location | None = None,
) -> MaskSSA:
    """Directory: Vector Compute / Mask Compute
    Description:
        Create a vector mask from a fixed pattern (`tla.mask.*` tokens).

        Parameters:
        - `pattern` (`_MaskPattern | str | None`): Mask pattern token or its name
          string. Required at runtime (`None` raises). Tokens live under `tla.mask`
          (for example `tla.mask.ALL`, `tla.mask.VL8`):

          | Pattern | Meaning |
          |---|---|
          | `ALL` | All elements are valid |
          | `ALLF` | All elements are invalid |
          | `VL1` / `VL2` / `VL3` / `VL4` | Lowest 1 / 2 / 3 / 4 elements are valid |
          | `VL8` / `VL16` / `VL32` / `VL64` / `VL128` | Lowest 8 / 16 / 32 / 64 / 128 elements are valid |
          | `M3` | Elements whose index is a multiple of 3 are valid |
          | `M4` | Elements whose index is a multiple of 4 are valid |
          | `H` | Lowest half of the elements are valid |
          | `Q` | Lowest quarter of the elements are valid |

        - `dtype` (`DTypeLike`): Element type associated with the mask (also decides
          how many elements fit in one vector: 256 bytes / element size). Optional,
          default `Float32`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; `pattern` is required.
        - Ops that take `mask=` need a `MaskSSA` from `create_mask` /
          `update_mask`, not a raw `tla.mask.*` token.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            # Build masks from pattern tokens:
            m_all = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            m_tail = tla.create_mask(pattern=tla.mask.VL8, dtype=tla.Float16)
            # Before: x_reg / y_reg are full vectors; only the lowest 8 elements add.
            z = tla.add(x_reg, y_reg, mask=m_tail)
            # After: valid elements hold x+y; masked-out elements stay inactive.
        ```
    """
    if pattern is None:
        _op_error("create_mask", "pattern is required")
    _require_frontend_state("create_mask")
    _runtime._require_enclosing_region("create_mask", "vec.func")
    elem_type = _mask_elem_type("create_mask", dtype, loc)
    mask_ty = _mask_ssa_type_for_element_type(_dtype_to_str(elem_type)).to_mlir_type(
        elem_type.context
    )
    token = pattern._token if isinstance(pattern, _MaskPattern) else str(pattern)
    return MaskSSA(
        _tla_ops_gen.create_mask(
            mask_ty, pattern=token, dtype=mlir_ir.TypeAttr.get(elem_type), loc=loc
        )
    )


@dsl_user_op
def update_mask(
    true_shape: IndexLike,
    dtype: DTypeLike = Float32,
    *,
    loc: mlir_ir.Location | None = None,
) -> tuple[MaskSSA, Numeric]:
    """Directory: Vector Compute / Mask Compute

    Description:
        Create a tail mask and return the remaining element count.

        Parameters:
        - `true_shape` (`IndexLike`): Shape of the currently valid (true) region. Required.
        - `dtype` (`DTypeLike`): Element type associated with the mask. Optional, default `Float32`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            tail_mask, remain = tla.update_mask(true_shape, dtype=tla.Float32)
        ```
    """
    _require_frontend_state("update_mask")
    _runtime._require_enclosing_region("update_mask", "vec.func")
    elem_type = _mask_elem_type("update_mask", dtype, loc)
    true_shape_value = _as_index_value(true_shape)
    mask_ty = _mask_ssa_type_for_element_type(_dtype_to_str(elem_type)).to_mlir_type(
        elem_type.context
    )
    index_ty = mlir_ir.IndexType.get()
    mask_value, new_true_shape = _tla_ops_gen.update_mask(
        mask_ty, index_ty, true_shape_value, mlir_ir.TypeAttr.get(elem_type), loc=loc
    )
    return MaskSSA(mask_value), as_numeric(new_true_shape)


# Keep this type-only: tla.ffi lazily calls back into core_api, so a
# runtime import here would add an unnecessary reverse dependency.
if TYPE_CHECKING:
    from .tla.ffi import ExternFunction


def _emit_extern_call(
    extern_function: ExternFunction,
    args: tuple[object, ...],
    *,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Validate and emit a call to a declared external function."""

    _require_frontend_state("extern")
    in_vector = _runtime._has_enclosing_region("vector")
    in_cube = _runtime._has_enclosing_region("cube")
    # Validate: scope, args count.
    if in_vector == in_cube:
        _op_error(
            "extern",
            "call must be nested inside exactly one of tla.vector() or tla.cube()",
        )
    if in_vector and _runtime._has_enclosing_region("vec.func"):
        _op_error("extern", "call must be outside tla.vec.func()")
    core_type = "aiv" if in_vector else "aic"
    if len(args) != len(extern_function.arg_types):
        _op_error(
            "extern",
            f"{extern_function.symbol} expects {len(extern_function.arg_types)} arguments, got {len(args)}",
        )

    operands: list[mlir_ir.Value] = []
    for position, (arg, expected) in enumerate(
        zip(args, extern_function.arg_types, strict=True)
    ):
        if isinstance(expected, TypedPointer):
            # Pointer case: get and validate ptr type, pointee type and address space.
            resolved = _resolve_bound_value(arg)
            if isinstance(arg, Tensor) or (
                isinstance(resolved, mlir_ir.Value)
                and _tla_type_bridge.type_is_tensor(resolved.type)
            ):
                _op_error(
                    "extern",
                    f"argument {position} of {extern_function.symbol} expects a Pointer, "
                    "but received a Tensor; pass tensor.ptr explicitly",
                )
            value = _as_value(arg)
            ptr_type = PtrType.try_cast(value.type)
            if ptr_type is None:
                _op_error(
                    "extern",
                    f"argument {position} of {extern_function.symbol} must be "
                    f"!tla.ptr<{expected.dtype.dtype}, {expected.space.name}, ...>; "
                    f"got {value.type}",
                )
            actual_dtype = Numeric.from_mlir_type(ptr_type.pointee)
            actual_space = AddressSpace.from_mlir_token(ptr_type.addrspace)
            if actual_dtype is not expected.dtype or actual_space is not expected.space:
                _op_error(
                    "extern",
                    f"argument {position} of {extern_function.symbol} expects pointer "
                    f"to {expected.dtype.__name__} in {expected.space.name}, "
                    f"got {actual_dtype.__name__} in {actual_space.name}",
                )
        else:
            # Numeric case: get and validate numeric type.
            resolved = _resolve_bound_value(arg)
            if isinstance(resolved, Numeric):
                value = resolved.ir_value(loc=loc)
            elif isinstance(resolved, mlir_ir.Value):
                value = resolved
            elif isinstance(resolved, (bool, int, float)):
                value = expected(resolved).ir_value(loc=loc)
            else:
                _op_error(
                    "extern",
                    f"argument {position} of {extern_function.symbol} must be "
                    f"{expected.__name__}, got {_type_name(arg)}",
                )
            if isinstance(value.type, mlir_ir.IndexType):
                _op_error(
                    "extern",
                    f"argument {position} of {extern_function.symbol} must not be index; "
                    f"declare and pass Int32 or Int64 explicitly",
                )
            try:
                actual_dtype = Numeric.from_mlir_type(value.type)
            except TypeError:
                _op_error(
                    "extern",
                    f"argument {position} of {extern_function.symbol} must be {expected.__name__}, got {value.type}",
                )
            if actual_dtype is not expected:
                _op_error(
                    "extern",
                    f"argument {position} of {extern_function.symbol} expects "
                    f"{expected.__name__}, got {actual_dtype.__name__}",
                )
        operands.append(value)

    state = _runtime._current_frontend_state()
    assert state is not None
    usage = state.extern_usages.get(extern_function.symbol)
    if usage is not None and usage.function is not extern_function:
        _op_error(
            "extern",
            f"symbol {extern_function.symbol!r} has multiple declarations",
        )

    # Create the call op and record it in usage.
    call_op = _tla_ops_gen.call_extern(extern_function.symbol, operands, loc=loc)
    if usage is None:
        usage = _ExternUsage(function=extern_function)
        state.extern_usages[extern_function.symbol] = usage
    usage.calls.append((call_op, core_type))


_mask_namespace = _Namespace()
for _mask_pattern_token in _MASK_PATTERN_TOKENS:
    _mask_namespace._set(_mask_pattern_token, _MaskPattern(_mask_pattern_token))


def __getattr__(name: str) -> Any:
    if name == "mask":
        return _mask_namespace
    raise AttributeError(name)


def _resolve_arch_layout_tag(value: Any | None, *, for_op: str) -> str:
    """Normalize ``Tensor(..., layout_tag=...)`` to the MLIR layout token string."""
    if value is None:
        token = _name_token(arch.RowMajor)
        assert token is not None
        return token
    if not isinstance(value, _LayoutTag):
        raise TypeError(
            f"{for_op}: layout_tag must be a tla.arch layout sentinel "
            f"(e.g. tla.arch.RowMajor); got {_type_name(value)}"
        )
    token = _name_token(value)
    if token is None:
        raise TypeError(f"{for_op}: layout sentinel produced no token: {value!r}")
    return token


__all__ = [
    "TlaCoreAPIError",
    "dsl_user_op",
    "arch",
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
    "local_mem_bar",
    "mutex",
    "mutex_guard",
    "mutex_lock",
    "mutex_unlock",
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
    "gather",
    "ReductionOp",
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
    "_Pointer",
    "VectorSSA",
    "MaskSSA",
]


# Capture the genuine frontend identities before user staging code can mutate
# public module namespaces.  The consumers retain these immutable snapshots.
from .base_dsl import ast_preprocessor as _ast_preprocessor  # noqa: E402
from .base_dsl import typing as _dsl_typing  # noqa: E402
from . import tla_ast_decorators as _tla_ast_decorators  # noqa: E402

_TRUSTED_DSL_TYPE_MODULES = frozenset(
    {
        "catlass.address_space",
        "catlass.base_dsl.typing",
        "catlass.core_api",
        "catlass.execution_lowering",
        "catlass.params",
        "catlass.runtime",
        "catlass.tla.tensor",
        "catlass.tla.typing",
        "catlass.types",
    }
)
# Trusted module identity for ``import catlass.tla as tla`` alias recognition.
# Must be ``catlass.tla`` (not ``catlass``): the preprocessor matches
# ``value is identities.module`` against user globals.
_dsl_module = sys.modules["catlass.tla"]
_ast_preprocessor._register_trusted_lazy_callables(
    tuple(
        value
        for value in globals().values()
        if callable(value)
        and getattr(value, "__module__", None) == __name__
        and not inspect.isclass(value)
    )
    + (_runtime.const_expr,),
    _dsl_module,
    range_callable=range,
    range_constexpr_callable=range_constexpr,
    as_numeric_callable=as_numeric,
    const_expr_callable=_runtime.const_expr,
    cube_callable=cube,
    vector_callable=vector,
    vec_namespace=vec,
    vec_func_callable=_vec_func,
)
_tla_ast_decorators._register_trusted_dsl_types(
    frozenset(
        candidate
        for namespace in (globals(), vars(_dsl_typing))
        for candidate in namespace.values()
        if isinstance(candidate, type)
        and getattr(candidate, "__module__", None) in _TRUSTED_DSL_TYPE_MODULES
    )
)
