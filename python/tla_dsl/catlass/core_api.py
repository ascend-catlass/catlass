"""Explicit user-facing TLA Python op helpers with inline preconditions."""

from __future__ import annotations

import builtins as _builtins
import inspect
import math
import os
import struct
import sys

import numpy as np
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
from .params import CastParams, RoundMode, SatMode

# Element-type tokens the tla.cast lowering supports: signed ints, unsigned
# ints and the AVE float set. Bool (i1) and Float64 (f64) are rejected by
# VectorSSA.to.
_CAST_FP8_DTYPES = frozenset({"f8e4m3fn", "f8e5m2"})
_FP4_PAIR_DTYPES = frozenset({"f4e2m1x2", "f4e1m2x2"})
# Int4x2 pairs two signed int4 values per byte in a cast to or from 16-bit lanes.
_INT4_CAST_SOURCES = frozenset({"i16", "f16"})
_INT4_CAST_TARGETS = frozenset({"i16", "f16", "bf16"})
_HIF8_CAST_FLOAT_DTYPES = frozenset({"f16", "f32"})
_CAST_UNSIGNED_DTYPES = frozenset({"u8", "u16", "u32", "u64"})
_CAST_SUPPORTED_DTYPES = (
    frozenset(
        {"i4x2", "i8", "i16", "i32", "i64", "f16", "bf16", "f32", "hif8", "f8e8m0"}
    )
    | _CAST_UNSIGNED_DTYPES
    | _CAST_FP8_DTYPES
    | _FP4_PAIR_DTYPES
)

# Reachable unsigned cast routes. Not derivable from widths: AVE advertises
# more than HIVMAVEToAVEIntrin implements, and a missing route hits
# llvm_unreachable rather than a diagnostic.
_CAST_UNSIGNED_ROUTES = frozenset(
    {
        ("u8", "u16"),
        ("u16", "u32"),
        ("u8", "u32"),
        ("u16", "u8"),
        ("u32", "u16"),
        ("u32", "u8"),
        ("i16", "u8"),
        ("i16", "u32"),
        ("i32", "u8"),
        ("i32", "u16"),
        ("u32", "i16"),
        ("u8", "f16"),
        ("f16", "u8"),
        ("u8", "f32"),
    }
)

# Same-type float casts lower to the AVE vtrc op ("just rounding"): the value
# keeps its element type and only the round mode reaches the instruction. No
# other same-type cast exists -- integers and fp8 formats have no such
# instruction, so the frontend refuses them rather than let the lowering crash.
_CAST_SAME_TYPE_FLOAT_DTYPES = frozenset({"f16", "bf16", "f32"})

# Element types tla.bitcast may reinterpret. Register views retain physical
# and valid bit counts; element widths may differ. HiF8 and FP4x2 values retain their
# formats in TLA IR and use byte registers in native lowering.
#
# E8M0 exposes its encoded bytes through UInt8. Logical FP4 types are
# excluded: their 4-bit values differ from byte registers holding two values per lane.
#
# i64/u64 registers can come from supported UB loads, extern, or bitcast.
# Tensor load/store and copy support depends on the distribution, mask, and
# route; bitcast does not extend that support or enable 64-bit arithmetic.
_BITCAST_ELEMENT_TYPES = frozenset(
    {
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "f16",
        "bf16",
        "f32",
        "f8e4m3fn",
        "f8e5m2",
        "f8e8m0",
        "hif8",
        "f4e2m1x2",
        "f4e1m2x2",
    }
)
from catlass._mlir.dialects import tla as _tla_ops_gen  # type: ignore[import-not-found]
from .base_dsl import ast_helpers as _ast_helpers
from .base_dsl.op import _ExternUsage, dsl_user_op, _capture_user_loc
from .base_dsl.typing import (
    BFloat16,
    Bool,
    Float16,
    Float32,
    Int8,
    Int16,
    Int32,
    Int64,
    Numeric,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    as_numeric,
    _float_literal_attr,
    _emit_integer_literal,
)
from .base_dsl.typing import Integer as _Integer
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
    is_bool_element_type,
    _replace_flat_leaves_in_tree,
)
from .params import (
    AtomicMode,
    CachePolicy,
    DcciTarget,
    RegPart,
    RegSlot,
    HF32Mode,
    ComputeOrder,
    CopyGmToUbParams,
    CopyL0C2DstParams,
    CopyPadParams,
    CopyParams,
    CopyUbToGmParams,
    HistogramsBinType,
    HistogramsParams,
    HistogramsType,
    L0C2UBMode,
    DsbTarget,
    LogAlgo,
    MemType,
    QuantMode,
    ReduceMode,
)


_PIPE_VALUES = {
    "scalar",
    "vector",
    "cube",
    "mte1",
    "mte2",
    "mte3",
    "all",
    "fix",
}
_MISSING = object()
_SUPPORTED_COMPARE_ELEMENT_TYPES = frozenset(
    {
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
    }
)
# AscendC Select: bool plus the 8/16/32/64-bit integers and half/float/bf16.
# AVE vectors are signless, so unsigned arms select whole bytes like their
# signed twins; Bool rides the b8 carrier. The low-precision formats below
# select whole byte/pair carriers too: f8e4m3fn/f8e5m2 use the native f8 vsel,
# hif8 and the packed FP4 pair carriers ride the i8 path.
_SUPPORTED_SELECT_ELEMENT_TYPES = frozenset(
    {
        "i1",
        "i8",
        "u8",
        "i16",
        "u16",
        "i32",
        "u32",
        "f16",
        "bf16",
        "f32",
        "i64",
        "u64",
        "f8e4m3fn",
        "f8e5m2",
        "hif8",
        "f4e2m1x2",
        "f4e1m2x2",
    }
)
_MASK_CMP_MODES = ("lt", "le", "gt", "ge", "eq", "ne")
# The two packed 4-bit cube float formats. Same width, different exponent /
# mantissa split, and the two matmul operands may use different ones.
_FP4_FORMATS = frozenset({"f4e2m1", "f4e1m2"})


#: Packed fp4 elements per byte. The descriptor halving is exact only on a
#: whole number of bytes, so this is the granularity the routes below require.
_FP4_PER_BYTE = 2

#: The copy routes that stage packed fp4 through byte descriptors, and so
#: carry the halving constraint below. These are exactly the routes whose
#: registration in ``Vector/dma.cpp`` converts the descriptor before handing it
#: to the copy templates: GM <-> UB through ``fp4RowMajorDescToBytes``, and
#: UB -> L1 through ``copyUBToL1Fp4``, which converts each side with whichever
#: of ``fp4RowMajorDescToBytes`` / ``fp4zNDescToBytes`` matches its layout.
#: Keeping the set explicit is what stops the rule from leaking onto GM -> L1,
#: a cube route whose tiles stay in fp4 denomination and legitimately carry a
#: runtime extent.
_FP4_BYTE_STAGED_ROUTES = frozenset({("gm", "ub"), ("ub", "gm"), ("ub", "l1")})

#: The stride each layout's byte conversion halves, as the name to report and
#: the leaf's position in the tile's flattened stride tree counted from the end.
#: Both converters halve the innermost origin extent and the innermost origin
#: coordinate and differ only here: ``fp4RowMajorDescToBytes`` halves
#: ``stride0``, the first of a RowMajor tile's two leaves, and
#: ``fp4zNDescToBytes`` halves ``stride3``, the last of a zN tile's four -- the
#: pitch between C0 columns. The two sit at different depths, so the position is
#: per layout rather than shared; indexing one tree with the other's position
#: reads a nested tuple where an integer belongs.
#:
#: These are the only two layouts that reach a byte-staged route: GM <-> UB is
#: registered RowMajor <-> RowMajor, and UB -> L1 as ``(RowMajor, zN)`` and
#: ``(zN, zN)``. Counting from the end leaves a rank-1 tile, whose tree is
#: shorter than the position it would need, out of the stride check, which is
#: right because such a tile has no stride between rows and its single leaf is
#: covered by the extent rule.
_FP4_HALVED_STRIDE_LEAF = {
    "RowMajor": ("row stride", -2),
    "zN": ("C0-column pitch", -1),
}


def _partial_byte_fp4_extent(
    desc, elements_per_byte: int = _FP4_PER_BYTE
) -> int | str | None:
    """The packed-axis origin extent of a byte-staged packed tile, unless it is
    a whole number of bytes. ``elements_per_byte`` is 2 for fp4 and int4.

    The fp4 staging routes do not run on fp4 descriptors: the AIV backend
    refuses sub-byte floats, so ``Vector/dma.cpp`` converts the descriptor to
    byte denomination (``fp4RowMajorDescToBytes``) and copies as ``int8_t``.
    Halving the extent is exact only on whole bytes. A row of an odd number of
    elements ends mid-byte, and the other half of that byte is an element
    outside the tile -- so the copy either drops the row's last element or
    overwrites the first element past it. An even extent has neither problem:
    34 elements are exactly 17 bytes, and copy byte for byte.

    A dynamic extent cannot be shown to be even, so it is refused. That is the
    case a static-only check misses and the one that stays wrong if the check
    is all there is: the halving happens on device, on whatever value arrives.

    Carrying an odd tail was implemented and withdrawn. Rounding the byte
    length up moves the problem to the neighbouring element, and merging the
    two halves needs a read-modify-write on the scalar unit, which a copy
    meta-op cannot order against the DMA around it: ``PipeBarrier`` does not
    order one pipe against another, and the event ids ``SetFlag`` would need
    belong to the frontend's flag allocation and to the copy templates. On
    device that read back a stale nibble, differing between two runs of one
    kernel.

    Returns the offending extent, ``"dynamic"`` when it is not a static
    integer, or ``None`` when it is a whole number of bytes.

    ``mx_to`` is deliberately more permissive and stays so: it accepts an odd
    extent, where ``preserveFp4Tail`` does the read-modify-write a DMA
    descriptor cannot express. Such a tile just cannot be staged to or from
    global memory.
    """
    origin = getattr(desc, "origin_shape", None)
    if not origin:
        return None
    last = origin[-1]
    if not isinstance(last, int):
        return "dynamic"
    if last % elements_per_byte:
        return last
    return None


def _fp4_odd_byte_mapping(
    desc,
    *,
    stride_field: tuple[str, int] | None,
    elements_per_byte: int = _FP4_PER_BYTE,
) -> tuple[str, int | str] | None:
    """The packed stride or origin coordinate of a byte-staged packed tile,
    unless both are a whole number of bytes. ``elements_per_byte`` is 2 for fp4
    and int4.

    A whole-byte extent fixes the length of each row, but not where the row
    starts. An odd halved stride puts every row, or every C0 column, after the
    first half a byte off, and an odd innermost origin coordinate does the same
    to the tile itself; the halved descriptor can express neither.

    Which stride that is depends on the layout, and a value only an explicit
    layout can reach is still reachable: every stride the frontend materializes
    for fp4 is even, so the odd ones arrive through ``make_layout`` and
    ``make_fake_tensor``. Being unreachable along one path is not an invariant
    of the field.

    A value that is only known at run time is refused for the same reason
    ``_partial_byte_fp4_extent`` refuses a dynamic extent: the halving runs on
    device on whatever value arrives, so a coordinate of 1 silently truncates
    to 0 and the copy reads from the element before the one asked for. The
    descriptor carries a runtime leaf as ``None``, which a static ``% 2`` test
    passes straight through -- that is the hole this spells out.

    ``stride_field`` names the stride this tile's byte conversion halves and
    gives its position in the flattened stride tree, or is ``None`` when the
    conversion halves no stride; see ``_FP4_HALVED_STRIDE_LEAF``.

    Returns the offending field and its value, ``"dynamic"`` in place of a
    value that is not a static integer, or ``None`` when both are whole bytes.
    """
    fields = [("origin coordinate", getattr(desc, "coord", None), -1)]
    if stride_field is not None:
        what, index = stride_field
        fields.insert(0, (what, getattr(desc, "stride", None), index))
    for what, tree, index in fields:
        if not tree:
            continue
        # A tile's trees are nested as deeply as its layout needs -- a zN
        # stride is a 2x2 tree where a RowMajor one is flat -- so the leaves
        # are flattened before the position is applied. Indexing the tree
        # itself reads a whole subtree on the nested layouts, which is not an
        # integer and so reports every such tile as a runtime value.
        leaves = _flatten_tla_tuple(tree)
        if len(leaves) < -index:
            continue
        value = leaves[index]
        if not isinstance(value, int):
            return what, "dynamic"
        if value % elements_per_byte:
            return what, value
    return None


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
        "i4x2",
        "i8",
        "u8",
        "f8e4m3fn",
        "f8e5m2",
        "hif8",
        "f4e2m1x2",
        "f4e1m2x2",
    }
)

# 64-bit tiles admitted outside the global make_tensor set:
# - i64/u64 on UB for Scatter;
# - i64/u64/f64 on RowMajor or VectorLayout GM/UB staging;
# - flat VectorLayout GM -> L1 (and make_tensor_like on L1) for fixpipe
#   dequant words (u64) and general b64 runs (i64 / f64).
# The copy route table limits each staging combination to registered wrappers.
_B64_STAGING_DTYPES = frozenset({"u64", "i64", "f64"})
_B64_STAGING_ADDRSPACES = frozenset({"gm", "l1"})
_B64_STAGING_LAYOUT_TAG = "VectorLayout"


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


def _check_select_element_type_supported(op_name: str, element_type: str) -> None:
    if element_type in _SUPPORTED_SELECT_ELEMENT_TYPES:
        return
    supported = ", ".join(sorted(_SUPPORTED_SELECT_ELEMENT_TYPES))
    _op_error(
        op_name,
        f"unsupported select element type {element_type}; "
        f"supported element types are {supported}",
    )


_SUPPORTED_MUL_INPLACE_ADD_ELEMENT_TYPES = frozenset(
    {"bf16", "f16", "f32", "i16", "i32", "u16", "u32"}
)


def _check_mul_inplace_add_element_type_supported(
    op_name: str, element_type: str
) -> None:
    if element_type in _SUPPORTED_MUL_INPLACE_ADD_ELEMENT_TYPES:
        return
    supported = ", ".join(sorted(_SUPPORTED_MUL_INPLACE_ADD_ELEMENT_TYPES))
    _op_error(
        op_name,
        f"unsupported mul_inplace_add element type {element_type}; "
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


# Vcg group-reduction instructions (ReduceIntraBlock) cover f16/f32 and
# 16/32-bit integers only; i8/u8/i64/bf16/f8 are not supported.
_SUPPORTED_REDUCE_INTRA_BLOCK_ELEMENT_TYPES = frozenset(
    {
        "f16",
        "f32",
        "i16",
        "i32",
        "u16",
        "u32",
    }
)


def _check_reduce_intra_block_element_type_supported(
    op_name: str, element_type: str
) -> None:
    if element_type in _SUPPORTED_REDUCE_INTRA_BLOCK_ELEMENT_TYPES:
        return
    supported = ", ".join(sorted(_SUPPORTED_REDUCE_INTRA_BLOCK_ELEMENT_TYPES))
    _op_error(
        op_name,
        f"unsupported reduce_intra_block element type {element_type}; "
        f"supported element types are {supported}",
    )


# PairWise (``ReduceMode.PAIR_WISE``) maps to the Vcpadd regbase
# instruction, which only covers the half and float element types.
_SUPPORTED_PAIR_WISE_ELEMENT_TYPES = frozenset({"f16", "f32"})


def _check_pair_wise_element_type_supported(op_name: str, element_type: str) -> None:
    if element_type in _SUPPORTED_PAIR_WISE_ELEMENT_TYPES:
        return
    supported = ", ".join(sorted(_SUPPORTED_PAIR_WISE_ELEMENT_TYPES))
    _op_error(
        op_name,
        f"unsupported pair_wise element type {element_type}; "
        f"supported element types are {supported}",
    )


class _Shape:
    """Result of ``make_shape``: Kernel ``!tla.shape`` SSA, or Host int-tree components.

    Nested structure is expressed only via **tuple trees** in ``make_shape(*components)``;
    do not pass ``_Shape`` / ``_Coord`` / ``_Stride`` as inner components.
    """

    __slots__ = ("_shape_value", "_components")

    def __init__(
        self, *, shape_value: mlir_ir.Value | None, components: tuple[Any, ...]
    ) -> None:
        self._shape_value = shape_value
        self._components = components

    @classmethod
    def _from_host(cls, components: tuple[Any, ...]) -> "_Shape":
        return cls(shape_value=None, components=components)


class _Coord:
    """Result of ``make_coord``: Kernel ``!tla.coord`` SSA, or Host int-tree components."""

    __slots__ = ("_coord_value", "_components")

    def __init__(
        self, *, coord_value: mlir_ir.Value | None, components: tuple[Any, ...]
    ) -> None:
        self._coord_value = coord_value
        self._components = components

    @classmethod
    def _from_host(cls, components: tuple[Any, ...]) -> "_Coord":
        return cls(coord_value=None, components=components)


class _Stride:
    """Result of ``make_stride``: Kernel ``!tla.stride`` SSA, or Host int-tree components.

    Like :class:`_Shape` / :class:`_Coord`, stores ``_components`` (the tuple tree passed to
    ``make_stride``) so Python code can recover the same tree as ``make_shape`` without
    reparsing the MLIR type string. :class:`_Layout` does **not** mirror this: layout is a
    single fused SSA value built from shape+stride (+ optional origin) operands.
    """

    __slots__ = ("_stride_value", "_components")

    def __init__(
        self, *, stride_value: mlir_ir.Value | None, components: tuple[Any, ...]
    ) -> None:
        self._stride_value = stride_value
        self._components = components

    @classmethod
    def _from_host(cls, components: tuple[Any, ...]) -> "_Stride":
        return cls(stride_value=None, components=components)


class _Layout:
    """Layout from ``make_layout``: Kernel ``!tla.layout`` SSA, or Host metadata.

    Kernel: fused layout SSA is the primary payload; source ``_Shape`` / ``_Stride`` /
    origin wrappers and the layout-tag token are retained for ``make_tensor`` IR rebuild.

    Host (outside ``@tla.kernel``): ``_layout_value`` is ``None``; ``_shape`` /
    ``_stride`` / ``_origin_shape`` are Host ``_Shape`` / ``_Stride`` (no MLIR SSA).
    """

    __slots__ = (
        "_layout_value",
        "_shape",
        "_stride",
        "_origin_shape",
        "_layout_tag",
    )

    def __init__(
        self,
        *,
        layout_value: mlir_ir.Value | None,
        shape: _Shape | None = None,
        stride: _Stride | None = None,
        origin_shape: _Shape | None = None,
        layout_tag: Any = None,
    ) -> None:
        self._layout_value = layout_value
        self._shape = shape
        self._stride = stride
        self._origin_shape = origin_shape
        self._layout_tag = layout_tag

    @classmethod
    def _from_host(
        cls,
        shape: _Shape,
        stride: _Stride,
        *,
        layout_tag: Any,
        origin_shape: _Shape | None,
    ) -> "_Layout":
        return cls(
            layout_value=None,
            shape=shape,
            stride=stride,
            origin_shape=origin_shape,
            layout_tag=layout_tag,
        )

    @property
    def shape(self) -> _Shape | None:
        return self._shape

    @property
    def stride(self) -> _Stride | None:
        return self._stride

    @property
    def layout_tag(self) -> Any:
        return self._layout_tag

    @property
    def origin_shape(self) -> _Shape | None:
        return self._origin_shape


def _require_host_index_components(value: Any, *, arg: str) -> Any:
    """Validate a Host-only int / nested-int-tuple tree (no SSA / Numeric)."""
    if isinstance(value, int):
        return value
    if isinstance(value, tuple):
        if len(value) == 0:
            raise TypeError(f"{arg} must be a non-empty int tree")
        return tuple(_require_host_index_components(x, arg=arg) for x in value)
    raise TypeError(
        f"{arg} must be an int or nested int tuple, got {type(value).__name__}"
    )


# Host make_layout accepts the typed object or a bare int tree; map each kind
# to its SSA field (must be None on Host) and the public factory name.
_HOST_TREE_SPEC: dict[type, tuple[str, str]] = {
    _Shape: ("_shape_value", "make_shape"),
    _Stride: ("_stride_value", "make_stride"),
}


def _coerce_host_tree(value: Any, *, cls: type, arg: str) -> Any:
    """Normalize Host shape/stride input to ``cls`` (``_Shape`` / ``_Stride``)."""
    ssa_attr, factory = _HOST_TREE_SPEC[cls]
    if isinstance(value, cls):
        if getattr(value, ssa_attr) is not None:
            _op_error(
                "make_layout",
                f"outside a kernel, {arg} must be from Host tla.{factory} "
                "(not Kernel SSA)",
            )
        # Already built by Host make_shape/make_stride; trust components.
        return value
    if isinstance(value, int):
        return cls._from_host((value,))
    if isinstance(value, tuple):
        return cls._from_host(_require_host_index_components(value, arg=arg))
    _op_error(
        "make_layout",
        f"outside a kernel, {arg} must be tla.{factory}(...) or an int tree; "
        f"got {type(value).__name__}",
    )


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
    """Pointer from ``make_ptr``: Kernel ``!tla.ptr`` SSA, or Host address metadata.

    Host (outside ``@tla.kernel``): ``value`` is ``None``; dtype / data_ptr /
    addrspace are held for Host ``tla.make_tensor``.
    """

    __slots__ = (
        "value",
        "_ptr_ty",
        "_alloc_size_bytes",
        "_host_dtype",
        "_host_data_ptr",
        "_host_addrspace",
        "_host_assumed_align",
    )
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
            if value.value is None:
                raise TypeError(
                    "cannot wrap a Host make_ptr result as a Kernel pointer"
                )
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
        object.__setattr__(self, "_host_dtype", None)
        object.__setattr__(self, "_host_data_ptr", None)
        object.__setattr__(self, "_host_addrspace", None)
        object.__setattr__(self, "_host_assumed_align", None)

    @classmethod
    def _from_host(
        cls,
        dtype: type[Numeric],
        data_ptr: int,
        addrspace: AddressSpace,
        *,
        assumed_align: int | None = None,
    ) -> "_Pointer":
        self = object.__new__(cls)
        object.__setattr__(self, "value", None)
        object.__setattr__(self, "_ptr_ty", None)
        object.__setattr__(self, "_alloc_size_bytes", None)
        object.__setattr__(self, "_host_dtype", dtype)
        object.__setattr__(self, "_host_data_ptr", int(data_ptr))
        object.__setattr__(self, "_host_addrspace", addrspace)
        object.__setattr__(
            self,
            "_host_assumed_align",
            None if assumed_align is None else int(assumed_align),
        )
        return self

    def __tla_type__(self) -> str:
        if self.value is None:
            return f"!tla.ptr<{self._host_dtype.dtype}, {self._host_addrspace}, host>"
        return str(self.value.type)

    def __str__(self) -> str:
        if self.value is None:
            return self.__tla_type__()
        return str(self.value.type)

    def __repr__(self) -> str:
        if self.value is None:
            return (
                f"Pointer(host, dtype={self._host_dtype!r}, "
                f"data_ptr={self._host_data_ptr!r}, addrspace={self._host_addrspace!r})"
            )
        return f"Pointer({self.value})"

    def __get_mlir_types__(self, context: mlir_ir.Context | None = None) -> list[Any]:
        del context
        if self.value is None:
            raise TypeError(
                "Host make_ptr result has no MLIR type; use inside @tla.kernel"
            )
        return [self.value.type]

    def __extract_mlir_values__(self) -> list[Any]:
        if self.value is None:
            raise TypeError(
                "Host make_ptr result has no MLIR value; use inside @tla.kernel"
            )
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
            if v0.value is None:
                raise TypeError("cannot rebuild Host make_ptr result from MLIR values")
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
        if self.value is None:
            return self._host_dtype
        return Numeric.from_mlir_type(self.pointee)

    @property
    def data_ptr(self) -> int:
        if self.value is None:
            return int(self._host_data_ptr)
        raise TypeError(
            "Kernel pointer has no Host data_ptr; use make_ptr outside a kernel"
        )

    @property
    def assumed_align(self) -> int | None:
        if self.value is None:
            return self._host_assumed_align
        return None

    @property
    def alignment(self) -> int:
        if self.value is None:
            if self._host_assumed_align is not None:
                return int(self._host_assumed_align)
            return _builtins.max(1, int(self._host_dtype.width) // 8)
        return self._ptr_ty.alignment

    @property
    def max_alignment(self) -> int:
        return self.alignment

    @property
    def memspace(self) -> AddressSpace:
        if self.value is None:
            return self._host_addrspace
        return AddressSpace.from_mlir_token(self._ptr_ty.addrspace)

    @property
    def type(self) -> Any:
        if self.value is None:
            raise TypeError("Host make_ptr result has no MLIR type")
        return self.value.type

    @property
    def pointee(self) -> mlir_ir.Type:
        if self.value is None:
            raise TypeError("Host make_ptr result has no MLIR pointee type")
        return self._ptr_ty.pointee

    @property
    def addrspace(self) -> AddressSpace:
        """Address space of ``!tla.ptr`` (same as :attr:`memspace`)."""
        if self.value is None:
            return self._host_addrspace
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
    def __class_getitem__(cls, dtype: type[Numeric]) -> TlaVectorSSATypeDescriptor:
        """Describe a full 256-byte register of the given dtype.

        Used for typing (including Bool move-only) and as the basis for
        ``tla.extern`` register annotations. Extern validation separately
        rejects Bool via :func:`catlass.tla.ffi._extern_register_type`.
        """
        from .tla.ffi import _full_register_type

        if not isinstance(dtype, type) or not issubclass(dtype, Numeric):
            raise TypeError("VectorSSA[dtype] requires a concrete TLA numeric type")
        return _full_register_type(dtype.dtype)

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
        mode: ReduceMode = ReduceMode.INTER_BLOCK,
        mask: Any,
        init_value: Any | None = None,
        reduction_profile: Any | None = None,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        """Directory: SIMD Vector Compute / Discrete and Aggregate

        Description:
            Reduce this register-resident vector under a mask. `mode` picks
            the reduction scope:
            - `ReduceMode.INTER_BLOCK` (default): reduce across all lanes
              (`tla.ReductionOp.ADD` / `MAX` / `MIN`) to a one-lane fragment.
              Broadcast the result back to full width with
              `tla.full(fragment, dtype=...)`.
            - `ReduceMode.INTRA_BLOCK`: reduce each 32-byte data block
              independently (`tla.ReductionOp.ADD` / `MAX` / `MIN`); each
              block's result is written contiguously to the least significant
              lanes of the returned vector (packed); lanes not occupied by a
              block result are 0, and a fully masked-off block's result lane
              carries the identity value (see Constraints). The result
              mirrors the operand shape and element type, except for `ADD`
              on i16/u16 sources, which returns the widened 32-bit result
              (see Constraints).
            - `ReduceMode.PAIR_WISE`: add every two adjacent lanes (only
              `tla.ReductionOp.ADD`); the pairwise sums are written to the
              least significant lanes of the returned vector and the upper
              half lanes are 0. Masked-out source lanes are treated as 0.

            Parameters:
            - `kind` (`ReductionOp`): Reduction kind (`ADD`, `MAX`, or `MIN`).
              `PAIR_WISE` accepts only `ADD`. Required.
            - `mode` (`ReduceMode`): Reduction scope. Optional, default `ReduceMode.INTER_BLOCK`.
            - `mask` (`MaskSSA`): Predicates which lanes participate. Required.
            - `init_value`: Reserved; only `None` is supported today. Optional, default `None`.
            - `reduction_profile`: Reserved; only `None` is supported today. Optional, default `None`.

            Constraints:
            - Must be called inside a `@tla.kernel`-decorated kernel function.
            - Must be called inside `tla.vec.func()`; `INTRA_BLOCK` and
              `PAIR_WISE` require `tla.vec.func(mode="simd")`.
            - Element type must be one of f16/f32/i16/i32/u16/u32.
              `PAIR_WISE` supports only f16/f32.
            - `init_value` and `reduction_profile` must be omitted or `None`.
            - `INTER_BLOCK`: a fully masked-off block yields the hardware
              identity: 0 for `ADD`; for `MAX` the element type's minimum
              (negative infinity for floating point) and for `MIN` the element
              type's maximum (positive infinity for floating point).
            - `INTRA_BLOCK` `ADD` on i16/u16 sources returns the widened
              32-bit result of the underlying instruction (same as vcadd): the
              result has half the lanes rounded up with i32/u32 element type
              (so a tail's last valid element keeps its widened lane), and no
              cast back to 16 bits is performed. Block sums that exceed the
              source type's range stay exact in the i32/u32 result.
            - `PAIR_WISE`: The reduction mode supports
              only `ADD`, and the mask merge mode is ZEROING, i.e., the high
              lanes are 0. With a fully active f16 vector, lanes 0..63 hold
              the pairwise sums and lanes 64..127 are 0.

            Example:
            ```python
            with tla.vec.func(mode="simd"):
                m = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                total = src_reg.reduce(tla.ReductionOp.ADD, mask=m)
                broadcast = tla.full(total, dtype=tla.Float32)
                packed = src_reg.reduce(
                    tla.ReductionOp.ADD, mode=tla.ReduceMode.INTRA_BLOCK, mask=m
                )
                sums = src_reg.reduce(
                    tla.ReductionOp.ADD, mode=tla.ReduceMode.PAIR_WISE, mask=m
                )
            ```
        """
        return _emit_vector_reduce(
            self,
            kind,
            mode=mode,
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
        """Directory: SIMD Vector Compute / Type Conversion

        Description:
            Convert the elements of a `VectorSSA` to `dst_type` and return a
            `VectorSSA` of that type.

            | Source element type | Destination element types |
            | --- | --- |
            | `Float16` | `Float16`, `BFloat16`, `Float32`, `Float8E4M3FN`, `Float8E5M2`, `HiFloat8`, `Int4x2`, `Int8`, `Int16`, `Int32`, `UInt8` |
            | `BFloat16` | `Float16`, `BFloat16`, `Float32`, `Float8E4M3FN`, `Float8E5M2`, `Float8E8M0`, `Float4E2M1x2`, `Float4E1M2x2`, `Int32` |
            | `Float32` | `Float16`, `BFloat16`, `Float32`, `Float8E4M3FN`, `Float8E5M2`, `HiFloat8`, `Int16`, `Int32`, `Int64` |
            | `Float8E4M3FN`, `Float8E5M2` | `Float16`, `BFloat16`, `Float32` |
            | `HiFloat8` | `Float16`, `Float32` |
            | `Float8E8M0` | `BFloat16` |
            | `Float4E2M1x2`, `Float4E1M2x2` | `BFloat16` |
            | `Int4x2` | `Int16`, `Float16`, `BFloat16` |
            | `Int8` | `Int16`, `Int32`, `Float16` |
            | `Int16` | `Int4x2`, `Int32`, `UInt8`, `UInt32`, `Float16`, `Float32` |
            | `Int32` | `Int8`, `Int16`, `Int64`, `UInt8`, `UInt16`, `Float32` |
            | `Int64` | `Int32`, `Float32` |
            | `UInt8` | `UInt16`, `UInt32`, `Float16`, `Float32` |
            | `UInt16` | `UInt8`, `UInt32` |
            | `UInt32` | `UInt8`, `UInt16`, `Int16` |

            Converting `Float16` or `Float32` to `HiFloat8` requires
            `CAST_ROUND` (nearest, ties away from zero) or `CAST_HYBRID`.
            Hybrid rounding can return a value other than the nearest
            representable value.

            `BFloat16` to `Float8E8M0` extracts the exponent bits. Zeros and
            subnormal values of either sign encode as `0x00`. Infinities always
            saturate to `0xfe`, the largest finite scale, and NaNs encode as
            `0xff`. In the reverse direction, `0x00` decodes to
            `2**-127` (BF16 bits `0x0040`), and `0xff` to a quiet NaN.

            Packed 4-bit types store two elements per byte, with the first in
            the low nibble. Packing writes pair `j` to byte `4*j + reg_slot`,
            and unpacking reads the same layout. Use `CAST_NONE` when widening
            `Int4x2`. Converting `Float32` to packed FP4 through `BFloat16`
            can round twice.

        Parameters:
            - `dst_type` (`type[Numeric]`): Destination element type from the table.
            - `params` (`CastParams`): Required. Sets `round_mode`, `sat_mode`, and `reg_slot`.
            - `mask` (`MaskSSA | None`): Selects source lanes. The default `None`
              enables all mask positions. Unselected destination lanes are zero.
              When packing 4-bit values,
              use the even lanes of a 128-lane source mask or a 64-lane pair mask
              created with `dtype=tla.UInt32`.

        Constraints:
            - Call in `tla.vec.func(mode="simd")` inside a `@tla.kernel` function.
              For scalar conversions in SIMT mode, use `Numeric.to`.
            - `reg_slot` defaults to `RegSlot.ZERO`. For twofold width changes
              with a slot selector, `ZERO` and `ONE` select even and odd lanes
              on the narrower side. `ZERO` through `THREE` select one of four
              positions for packed 4-bit conversions, fourfold integer
              conversions, `UInt8` to `Float32`, and conversions between
              `Float32` and `Float8E4M3FN`, `Float8E5M2`, or `HiFloat8`.
            - With the default `CAST_NONE`, conversions that round and support
              `CAST_RINT` use ties-to-even rounding. `BFloat16` to `Float16`,
              `Float16` to `Int32` or `Int4x2`, `Float32` and `Int64` in either
              direction, and `BFloat16` to packed FP4 reject `CAST_ODD`.
            - For `Float16` to `BFloat16`, set `sat_mode` to `SatMode.UNKNOWN`.
              `BFloat16` to packed FP4 always saturates and requires
              `SatMode.UNKNOWN` or `SatMode.SAT`. `Int16` to `UInt32` uses
              fixed saturation.
            - Same-type casts for `Float16`, `BFloat16`, and `Float32` round
              selected elements in their original lanes. Use `ZERO` or `ONE`
              and leave saturation disabled.

        Example:
            ```python
            with tla.vec.func(mode="simd"):
                as_i32 = x_f32.to(tla.Int32, tla.params.CastParams())
                to_hif8 = tla.params.CastParams(round_mode=tla.params.RoundMode.CAST_ROUND)
                as_hif8 = x_f32.to(tla.HiFloat8, to_hif8)
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
        if dst_type.dtype not in _CAST_SUPPORTED_DTYPES:
            _op_error(
                "cast",
                f"unsupported cast target dtype '{dst_type.dtype}': tla.cast "
                f"supports signed integers (i4x2/i8/i16/i32/i64), unsigned integers "
                f"(u8/u16/u32/u64) and floats "
                f"(f16/bf16/f32/f8e4m3fn/f8e5m2/hif8/f4e2m1x2/f4e1m2x2/f8e8m0)",
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
        if is_bool_element_type(src_token):
            _op_error(
                "cast",
                "Bool VectorSSA is move-only (load/store); cast is not supported",
            )
        type_pair = (src_token, dst_token)
        if type_pair == ("i16", "i8"):
            _op_error("cast", "unsupported cast from Int16 to Int8")
        e8m0_cast = type_pair in (("bf16", "f8e8m0"), ("f8e8m0", "bf16"))
        if "f8e8m0" in type_pair and not e8m0_cast:
            _op_error("cast", "E8M0 pairs with BFloat16")
        if params.round_mode is RoundMode.CAST_HYBRID and not (
            e8m0_cast or (src_token in _HIF8_CAST_FLOAT_DTYPES and dst_token == "hif8")
        ):
            _op_error("cast", "CAST_HYBRID only supports Float16/Float32->HiFloat8")
        if params.round_mode is RoundMode.CAST_ODD and type_pair in (
            ("bf16", "f16"),
            ("f16", "i32"),
            ("f32", "i64"),
            ("i64", "f32"),
        ):
            _op_error("cast", f"CAST_ODD is unsupported for {src_token}->{dst_token}")
        if src_token == "hif8" or dst_token == "hif8":
            other_token = dst_token if src_token == "hif8" else src_token
            if other_token not in _HIF8_CAST_FLOAT_DTYPES:
                _op_error("cast", "HiFloat8 pairs with Float16 or Float32")
            if dst_token == "hif8" and params.round_mode not in (
                RoundMode.CAST_ROUND,
                RoundMode.CAST_HYBRID,
            ):
                _op_error(
                    "cast",
                    "conversion to HiFloat8 requires CAST_ROUND or CAST_HYBRID",
                )
        fp4_narrow = src_token == "bf16" and dst_token in _FP4_PAIR_DTYPES
        if src_token in _FP4_PAIR_DTYPES or dst_token in _FP4_PAIR_DTYPES:
            if not (
                fp4_narrow or (src_token in _FP4_PAIR_DTYPES and dst_token == "bf16")
            ):
                _op_error(
                    "cast",
                    "FP4x2 pairs with BFloat16; compose other routes explicitly",
                )
            if fp4_narrow and params.round_mode is RoundMode.CAST_ODD:
                _op_error(
                    "cast",
                    "BF16->FP4x2 supports CAST_ROUND/CAST_FLOOR/CAST_CEIL/"
                    "CAST_TRUNC/CAST_RINT/CAST_NONE only",
                )
            if fp4_narrow and params.sat_mode not in (SatMode.SAT, SatMode.UNKNOWN):
                _op_error(
                    "cast",
                    "BF16->FP4x2 requires SAT or UNKNOWN; the native instruction always saturates",
                )
        if dst_token == "i4x2":
            if src_token not in _INT4_CAST_SOURCES:
                _op_error("cast", "Int4x2 narrows from Int16 or Float16")
            if src_token == "f16" and params.round_mode is RoundMode.CAST_ODD:
                _op_error("cast", "CAST_ODD is unsupported for Float16->Int4x2")
        if src_token == "i4x2":
            if dst_token not in _INT4_CAST_TARGETS:
                _op_error(
                    "cast",
                    f"unsupported cast 'i4x2'->'{dst_token}': Int4x2 widens to "
                    f"i16/f16/bf16 only; compose another route explicitly",
                )
            if params.round_mode is not RoundMode.CAST_NONE:
                _op_error(
                    "cast",
                    "Int4x2 source casts require CAST_NONE; the widening "
                    "instruction has no rounding operand",
                )
        if (
            src_token == "f16"
            and dst_token == "bf16"
            and params.sat_mode is not SatMode.UNKNOWN
        ):
            _op_error("cast", "f16->bf16 cast only supports UNKNOWN sat_mode")
        _require_unsigned_cast_route(src_token, dst_token)
        if src_token == dst_token and src_token not in _CAST_SAME_TYPE_FLOAT_DTYPES:
            _op_error(
                "cast",
                f"unsupported cast '{src_token}'->'{dst_token}': a same-type "
                f"cast is only supported for floats (f16/bf16/f32), where it "
                f"lowers to the AVE vtrc rounding op; integers and fp8 formats "
                f"have no same-type cast",
            )
        if src_token == dst_token and params.sat_mode is SatMode.SAT:
            _op_error(
                "cast",
                f"invalid params for cast '{src_token}'->'{dst_token}': a "
                f"same-type cast lowers to the AVE vtrc op, which never "
                f"saturates; use UNKNOWN or NOSAT sat_mode",
            )
        _require_reg_slot_reachable(src_token, dst_token, params.reg_slot)
        result_desc = _cast_result_descriptor(src_desc, dst_token, params)
        with context:
            trait_attr = mlir_ir.DenseI32ArrayAttr.get(params.codes())
        mask_value = _as_value(mask) if mask is not None else None
        if mask_value is not None:
            if fp4_narrow or dst_token == "i4x2":
                if _mask_ssa_type_for_mlir_value(mask_value).physical_lanes not in (
                    64,
                    128,
                ):
                    _op_error(
                        "cast",
                        "packed 4-bit narrowing requires a 128-lane source mask or a 64-lane pair mask (dtype=UInt32)",
                    )
            else:
                _require_mask_matches_vector("cast", mask_value, operand_value)
        result = _tla_ops_gen.cast(
            result_desc.to_mlir_type(context),
            operand_value,
            trait_attr,
            mask=mask_value,
            loc=loc,
        )
        return VectorSSA(result)

    @dsl_user_op
    def bitcast(
        self,
        dst_type: Any,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        """Directory: SIMD Vector Compute / Type Conversion

        Description:
            Reinterpret the bits of a register-resident vector as another
            element type without numeric conversion. The physical register
            retains all 2048 bits. Element widths may differ; the physical
            and valid lane counts change to preserve their respective bit
            counts. For example, 64 Float32 lanes become 128 BFloat16 lanes,
            exposing both halfwords of each Float32 bit pattern.

            Supported types are `f32`/`i32`/`u32`, `f16`/`bf16`/`i16`/`u16`,
            `i8`/`u8`/`f8e4m3fn`/`f8e5m2`/`hif8`/`f4e2m1x2`/`f4e1m2x2`, and
            `i64`/`u64`. Float8E8M0 supports a UInt8 byte view in both directions.
            Bool and logical four-bit Tensor types are rejected. This operation
            changes neither byte order nor register slot and does not pack sparse
            values. Use `.to()` for numeric conversion and explicit rearrangement
            for layout changes.

            `i64`/`u64` registers can come from UB loads, `tla.extern`, or
            bitcast. Register staging supports aligned, full-width
            `DIST_NORM` loads and aligned `DIST_NORM` stores without an
            explicit mask. Tensor copy support depends on the route; see
            the `tla.copy` route table. Bitcast does not extend these
            movement constraints or enable 64-bit arithmetic.

            Parameters:
            - `dst_type` (`type[Numeric]`): Element type to reinterpret as. Required.

            Constraints:
            - Must be called inside a `@tla.kernel`-decorated kernel function.
            - Must be called inside `tla.vec.func()`.
            - A static valid bit count must contain whole destination elements.
            - Dynamic valid ranges remain dynamic; consumers need explicit masks.
            - A bitcast to the source's own element type returns `self`.

            Example:
            ```python
            with tla.vec.func(mode="simd"):
                bits = x.bitcast(tla.Int32)
                exponent = tla.shift_right(bits, 23, mask=m)
            ```
        """
        _require_category("bitcast", "operand", self, "vector_ssa", 0)
        if not (
            isinstance(dst_type, type)
            and issubclass(dst_type, Numeric)
            and dst_type.dtype
        ):
            _op_error(
                "bitcast",
                f"invalid argument 'dst_type' (position 0): expected a concrete "
                f"Numeric element type, got {_type_name(dst_type)}",
            )
        _require_frontend_state("bitcast")
        _runtime._require_enclosing_region("bitcast", "vec.func")
        operand_value = _as_value(self)
        src_desc = _vector_ssa_type_for_mlir_value(operand_value)
        src_token = src_desc.element_type
        dst_token = _dtype_to_str(dst_type)
        if is_bool_element_type(src_token) or is_bool_element_type(dst_token):
            _op_error(
                "bitcast",
                "Bool VectorSSA is move-only (load/store); bitcast is not supported",
            )
        for role, token in (("source", src_token), ("destination", dst_token)):
            if token not in _BITCAST_ELEMENT_TYPES:
                _op_error(
                    "bitcast",
                    f"unsupported {role} element type '{token}': tla.bitcast "
                    f"supports {', '.join(sorted(_BITCAST_ELEMENT_TYPES))}",
                )
        if dst_token == src_token:
            return self
        if "f8e8m0" in (src_token, dst_token) and "u8" not in (src_token, dst_token):
            _op_error("bitcast", "E8M0 register bitcast requires UInt8")
        src_bytes = dtype_size_bytes(src_token)
        dst_bytes = dtype_size_bytes(dst_token)
        valid_lanes = src_desc.valid_lanes
        if valid_lanes is not None and (valid_lanes * src_bytes) % dst_bytes:
            _op_error(
                "bitcast",
                f"valid source byte count {valid_lanes * src_bytes} is not divisible by the {dst_bytes}-byte destination element width",
            )
        if valid_lanes is not None:
            valid_lanes = valid_lanes * src_bytes // dst_bytes
        result_desc = TlaVectorSSATypeDescriptor(valid_lanes, dst_token)
        context = operand_value.type.context
        result = _tla_ops_gen.bitcast(
            result_desc.to_mlir_type(context),
            operand_value,
            loc=loc,
        )
        return VectorSSA(result)

    @dsl_user_op
    def generate_mask(
        self,
        offset: int,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> "MaskSSA":
        """Directory: SIMD Vector Compute / Mask Compute

        Description:
            Build a predicate mask from one data block of this register.

            The register is treated as a sequence of data blocks ordered by
            element and then by in-element bit (LSB first). `offset` selects one
            block, and each bit of that block becomes one predicate lane. The
            source element width decides the block width and the block count,
            and together all blocks cover one 2048-bit register:

            - 32-bit elements: 32 blocks of 8Byte (64 bits) each, so `offset`
              is `0..31`. The result is a `mask<64>` whose lane `L` is register
              bit `64 * offset + L`.
            - 16-bit elements: 16 blocks of 16Byte (128 bits) each, so `offset`
              is `0..15`. The result is a `mask<128>` whose lane `L` is register
              bit `128 * offset + L`.

            Physically the 32Byte predicate register gives each element 2 bits
            (16-bit) or 4 bits (32-bit); the selected bit is broadcast to those
            bits, and only the lowest one is effective.

            Parameters:
            - `offset` (`int`): Compile-time block index. Required. `0..15` for
              16-bit elements and `0..31` for 32-bit elements.

            Constraints:
            - Supports Ascend 950 (`--npu-arch 3510`) only.
            - Must be called inside a `@tla.kernel`-decorated kernel function.
            - Must be called inside `tla.vec.func(mode="simd")`.
            - Element width must be 16 or 32 bits; other widths are unsupported.
            - `offset` must be a compile-time Python int within the block range.

            Example:
            ```python
            with tla.vec.func(mode="simd"):
                bits = src_tile.load()      # !tla.vector<64xf32>
                m = bits.generate_mask(0)   # !tla.mask<64>
            ```
        """
        _require_category("generate_mask", "operand", self, "vector_ssa", 0)
        _require_frontend_state("generate_mask")
        _runtime._require_enclosing_region("generate_mask", "vec.func")
        if _runtime._in_simt_vec_func():
            _op_error("generate_mask", "is only supported in tla.vec.func(mode='simd')")

        src_value = _as_value(self)
        src_desc = _vector_ssa_type_for_mlir_value(src_value)
        limit = _generate_mask_offset_limit("generate_mask", src_desc.element_type)
        block_index = _require_constexpr_range_bound("generate_mask", "offset", offset)
        if block_index < 0 or block_index > limit:
            _op_error(
                "generate_mask",
                f"offset must be in 0..{limit} for a {src_desc.element_type} source, "
                f"got {block_index}",
            )

        mask_ty = _mask_ssa_type_for_element_type(src_desc.element_type).to_mlir_type(
            src_value.type.context
        )
        result = _tla_ops_gen.generate_mask(
            mask_ty,
            src_value,
            mlir_ir.IntegerAttr.get(mlir_ir.IntegerType.get_signless(32), block_index),
            loc=loc,
        )
        return MaskSSA(result)


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


# Rejection hint per tensor-construction API. The op name is the API the user
# wrote, so a `get_tile` call is never reported as its internal `make_tensor`.
_SIMT_TENSOR_CONSTRUCTION_HINTS = {
    "tile_view": "请在区域外创建视图，再在区域内读取 view.ptr_with_coord。",
    "get_tile": "请在区域外创建视图，再在区域内读取 view.ptr_with_coord。",
    "make_tensor": "请在区域外创建张量，再在区域内读取 tensor 元素或 tensor.ptr_with_coord。",
    "make_tensor_like": "请在区域外创建张量，再在区域内读取 tensor 元素或 tensor.ptr_with_coord。",
}


def _reject_simt_tensor_construction(op_name: str) -> None:
    """Refuse to build a tensor/view inside a ``tla.vec.func(mode="simt")``.

    A SIMT region is outlined into a per-thread function whose buffers cross the
    launch ABI as one bare pointer each, with no descriptor or layout. A tensor
    built inside the region materializes a ``tla.tensor_desc`` whose base is the
    enclosing raw memref, so the SIMT capture check used to reject the region
    for "indexing 1 raw memref(s) directly" -- a message about the launch ABI,
    not about the call the user wrote. Reject the construction at its call site
    and name the supported pattern: build the view/tensor outside the region and
    capture it.
    """
    if not _runtime._in_simt_vec_func():
        return
    hint = _SIMT_TENSOR_CONSTRUCTION_HINTS[op_name]
    _op_error(
        op_name,
        f"SIMT vec.func 内不支持创建 {op_name}。{hint}",
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


# AscendC Duplicate element types, plus the TLA byte-carried formats. Complex
# elements have no TLA vector type and stay unsupported. i64 goes through the
# linked bitcode helper: the AVE 64-bit broadcast ABI cannot legalize.
_FULL_SUPPORTED_DTYPES = frozenset(
    (
        "i1",
        "i8",
        "u8",
        "i16",
        "u16",
        "i32",
        "u32",
        "i64",
        "u64",
        "bf16",
        "f16",
        "f32",
        "f8e4m3fn",
        "f8e5m2",
        "f8e8m0",
        "hif8",
        "f4e2m1x2",
        "f4e1m2x2",
    )
)
# Bool/hif8/e8m0/fp4x2 broadcast an i8 byte; unsigned broadcasts the same-width
# signless integer (arith.constant and the AVE scalar ABI reject ui*).
_FULL_BYTE_CARRIER_DTYPES = frozenset(("i1", "hif8", "f8e8m0", "f4e2m1x2", "f4e1m2x2"))

_ARANGE_SUPPORTED_DTYPES = frozenset(("i8", "i16", "i32", "i64", "f16", "f32"))
_ARANGE_ORDERS = frozenset(("increase", "decrease"))

# Width of one Ascend vector register tile in bytes. Must stay in sync with
# kVectorBytes in csrc/mlir/lib/Passes/TlaTensorToMemref.cpp.
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


def _is_bool_like_pad_operand(value: Any) -> bool:
    """True for Python bool, ``tla.Bool``, or ``i1`` SSA (not integer 0/1)."""
    if isinstance(value, bool):
        return True
    if isinstance(value, Bool) or (
        isinstance(value, Numeric) and type(value).dtype.lower() in ("i1", "bool")
    ):
        return True
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        return True
    if isinstance(resolved, Bool) or (
        isinstance(resolved, Numeric) and type(resolved).dtype.lower() in ("i1", "bool")
    ):
        return True
    if isinstance(resolved, mlir_ir.Value) and mlir_ir.IntegerType.isinstance(
        resolved.type
    ):
        return mlir_ir.IntegerType(resolved.type).width == 1
    return False


def _as_signless_int_ssa(
    value: mlir_ir.Value,
    width: int,
    *,
    what: str,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    """Bridge an integer SSA value to the signless ``i<width>``.

    ``arith.*`` only accepts signless integers, so an unsigned ``uiN`` first goes
    through ``builtin.unrealized_conversion_cast`` to its signless carrier and is
    then widened/narrowed. ``index`` is cast with ``arith.index_cast``. Rejects
    ``i1``; callers that allow only one width should check ``IntegerType.width``
    first.
    """
    signless = mlir_ir.IntegerType.get_signless(width)
    if isinstance(value.type, mlir_ir.IndexType):
        return mlir_ir.Operation.create(
            "arith.index_cast", operands=[value], results=[signless], loc=loc
        ).results[0]
    if mlir_ir.IntegerType.isinstance(value.type):
        int_type = mlir_ir.IntegerType(value.type)
        if int_type.width == 1:
            raise TypeError(
                f"{what} must not be bool/tla.Bool/i1; "
                "use a non-boolean integer (e.g. Int32/Int64)"
            )
        if int_type.is_unsigned:
            # arith.* only accepts signless ints; uiN -> iN via
            # unrealized_conversion_cast (same pattern as debug print).
            carrier = mlir_ir.Operation.create(
                "builtin.unrealized_conversion_cast",
                operands=[value],
                results=[mlir_ir.IntegerType.get_signless(int_type.width)],
                loc=loc,
            ).results[0]
            if int_type.width == width:
                return carrier
            if int_type.width > width:
                raise TypeError(
                    f"{what} unsigned width {int_type.width} cannot narrow to {width}"
                )
            return mlir_ir.Operation.create(
                "arith.extui", operands=[carrier], results=[signless], loc=loc
            ).results[0]
        if int_type.width == width:
            return value
        cast = "arith.extsi" if int_type.width < width else "arith.trunci"
        return mlir_ir.Operation.create(
            cast, operands=[value], results=[signless], loc=loc
        ).results[0]
    raise TypeError(f"{what} expected an integer SSA value, got {value.type}")


def _as_signless_i64_ssa(
    value: mlir_ir.Value, *, what: str, loc: mlir_ir.Location | None = None
) -> mlir_ir.Value:
    """Bridge an integer SSA value to signless ``i64`` (width-64 specialization)."""
    return _as_signless_int_ssa(value, 64, what=what, loc=loc)


def _as_i64_copy_pad_operand(
    value: Any, *, what: str, loc: mlir_ir.Location | None = None
) -> mlir_ir.Value:
    """Widen CopyPadParams left/right/value SSA to signless ``i64``.

    Rejects ``bool`` / ``tla.Bool`` / ``i1``. Unsigned integers go uiN→iN via
    ``builtin.unrealized_conversion_cast`` then ``arith.extui`` (or identity at
    width 64), not the signed ``arith.extsi`` path used by ``_as_i64_value``.
    """
    if _is_bool_like_pad_operand(value):
        raise TypeError(
            f"CopyPadParams.{what} must not be bool/tla.Bool/i1; "
            "use a non-boolean integer (e.g. Int32/Int64)"
        )
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, Numeric):
        dtype = type(resolved).dtype.lower()
        host = resolved.value
        # Whole-number float hosts on Int*/UInt* (e.g. Int32(9.0), UInt32(1.0))
        # must lower as signless i64 constants — same as int hosts — so static
        # checks and IR emission stay aligned and uiN constants are not emitted.
        if isinstance(host, float) and not isinstance(host, bool):
            if (dtype.startswith("i") or dtype.startswith("u")) and dtype[1:].isdigit():
                if not host.is_integer():
                    raise TypeError(
                        f"CopyPadParams.{what} integer Numeric host must be a "
                        f"whole number (got {host!r} on {type(resolved).__name__})"
                    )
                host = int(host)
        if isinstance(host, (bool, int)):
            if isinstance(host, bool) or dtype in ("i1", "bool"):
                raise TypeError(
                    f"CopyPadParams.{what} must not be bool/tla.Bool/i1; "
                    "use a non-boolean integer (e.g. Int32/Int64)"
                )
            if dtype.startswith("u") and dtype[1:].isdigit():
                width = int(dtype[1:])
                return _const_i64(int(host) & ((1 << width) - 1), loc=loc)
            if dtype.startswith("i") and dtype[1:].isdigit():
                return _const_i64(int(host), loc=loc)
            raise TypeError(
                f"CopyPadParams.{what} expected integer Numeric, "
                f"got {type(resolved).__name__}"
            )
        resolved = resolved.ir_value(loc=loc)
    if isinstance(resolved, mlir_ir.Value):
        if isinstance(
            resolved.type, mlir_ir.IndexType
        ) or mlir_ir.IntegerType.isinstance(resolved.type):
            return _as_signless_i64_ssa(resolved, what=f"CopyPadParams.{what}", loc=loc)
    if isinstance(resolved, int) and not isinstance(resolved, bool):
        return _const_i64(resolved, loc=loc)
    raise TypeError(
        f"CopyPadParams.{what} expected i64-like integer, got {type(value).__name__}"
    )


def _u64_two_complement(value: int) -> int:
    """Wrap an unsigned 64-bit value into the signless i64 constant domain."""
    bits = value & ((1 << 64) - 1)
    return bits - (1 << 64) if bits >= (1 << 63) else bits


def _as_u64_bit_operand(name: str, value: Any, *, loc) -> mlir_ir.Value:
    """Resolve a uint64 bit-query operand to a signless i64 SSA value.

    Accepts a ``UInt64`` Numeric (host literal or SSA), a non-negative Python
    ``int`` that fits 64 bits, or a runtime scalar argument proxy bound to a
    64-bit unsigned SSA. ``bool`` / ``tla.Bool`` / ``i1`` and signed or narrower
    integers are rejected: the query is only defined on a uint64 bit pattern.
    """
    if isinstance(value, bool):
        _op_error(name, "value must be a UInt64 scalar, not bool")
    if isinstance(value, int):
        if not 0 <= value < (1 << 64):
            _op_error(
                name, "value must be a non-negative Python int that fits in 64 bits"
            )
        return _const_i64(_u64_two_complement(value), loc=loc)

    # Only a non-numeric category is rejected here.
    category = _category(value)
    if category not in (None, "numeric"):
        _op_error(name, f"value must be a UInt64 scalar, got {_type_name(value)}")
    resolved = _resolve_bound_value(value)

    if isinstance(resolved, Numeric):
        if not isinstance(resolved, UInt64):
            _op_error(
                name,
                "value must be a UInt64 scalar (64-bit unsigned integer), "
                f"got {type(resolved).__name__}",
            )
        host = resolved.value
        if isinstance(host, mlir_ir.Value):
            ssa = resolved.ir_value(loc=loc)
        elif isinstance(host, int) and not isinstance(host, bool):
            if not 0 <= host < (1 << 64):
                _op_error(name, "value must fit in 64 bits")
            return _const_i64(_u64_two_complement(host), loc=loc)
        else:
            _op_error(name, "value must be a UInt64 scalar")
    elif isinstance(resolved, mlir_ir.Value):
        ssa = resolved
    else:
        _op_error(name, f"value must be a UInt64 scalar, got {_type_name(value)}")

    # The bitcode helper takes uint64; bridge the unsigned SSA to signless i64
    if not mlir_ir.IntegerType.isinstance(ssa.type):
        _op_error(name, f"value must be a 64-bit unsigned integer, got {ssa.type}")
    int_type = mlir_ir.IntegerType(ssa.type)
    if int_type.width != 64 or not int_type.is_unsigned:
        _op_error(name, f"value must be a 64-bit unsigned integer, got {ssa.type}")
    return _as_signless_i64_ssa(ssa, what=name, loc=loc)


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
        # Shared helper: signed Int* only. CopyPad unsigned / whole-float hosts
        # use ``_const_pad_count_value`` so arange and other callers keep their
        # pre-existing acceptance rules (do not widen this helper for Pad).
        if dtype.startswith("i") and dtype[1:].isdigit():
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


def _pad_count_host_int(value: Any) -> int | None:
    """Host integer for CopyPad left/right counts, or None if not static.

    Accepts Python ``int``, signed/unsigned integer ``Numeric`` with an ``int``
    host, and integer ``Numeric`` whose host is a whole-number ``float``
    (``Int32(9.0)`` → ``9``). Fractional floats raise ``TypeError``. Bool /
    ``tla.Bool`` return ``None`` (caller rejects separately).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        return None
    if isinstance(resolved, int):
        return resolved
    if isinstance(resolved, Numeric):
        dtype = type(resolved).dtype.lower()
        if not (
            (dtype.startswith("i") or dtype.startswith("u")) and dtype[1:].isdigit()
        ):
            return None
        host = resolved.value
        if isinstance(host, bool):
            return None
        if isinstance(host, float):
            if not host.is_integer():
                raise TypeError(
                    f"CopyPadParams pad count Numeric host must be a whole number "
                    f"(got {host!r} on {type(resolved).__name__})"
                )
            host = int(host)
        if not isinstance(host, int):
            return None
        if dtype.startswith("u"):
            width = int(dtype[1:])
            return int(host) & ((1 << width) - 1)
        return int(host)
    return None


def _const_pad_count_value(value: Any) -> int | None:
    """Static pad-count extraction for CopyPad capacity / 32B checks only."""
    return _pad_count_host_int(value)


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
    packed_widen = (
        src_desc.element_type == "i4x2" and dst_token in _INT4_CAST_TARGETS
    ) or (src_desc.element_type in _FP4_PAIR_DTYPES and dst_token == "bf16")
    if packed_widen:
        # One packed byte lane holds two 4-bit values and widens to two 16-bit
        # lanes. A cast reads one of four register slots, so the result is
        # 2 * ceil((src_lanes - slot) / 4).
        slot = params.codes()[0]
        if src_lanes <= slot:
            _op_error(
                "cast", "packed 4-bit source has no valid byte in the selected slot"
            )
        return TlaVectorSSATypeDescriptor(2 * ((src_lanes - slot + 3) // 4), dst_token)
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
    # dtype_size_bytes("i1")==1 matches AscendC CreateMask<bool> (sizeof==1 → pset_b8)
    # → 256 predicate lanes for a 256B VL.
    element_bytes = dtype_size_bytes(token)
    if element_bytes <= 0 or _VECTOR_REGISTER_BYTES % element_bytes != 0:
        _op_error(
            "mask",
            f"unsupported predicate element type {token!r}; expected a "
            "byte-aligned type that fits a 256-byte vector register "
            "(Bool is accepted as b8-width)",
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


def _static_lane_count_or_none(tree: Any) -> int | None:
    lanes = 1
    for dim in _flatten_tla_tuple(tree):
        if dim is None:
            return None
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TlaLoweringError(
                "VectorSSA load requires static integer or dynamic shape leaves"
            )
        lanes *= dim
    return lanes


def _vector_ssa_type_from_tensor_descriptor(
    tensor_type: TlaTensorTypeDescriptor,
) -> TlaVectorSSATypeDescriptor:
    # The register width comes from the statically folded origin window. A tile_view
    # whose origin is dynamic keeps the width dynamic: a plain dynamic view does not
    # know how many lanes are valid, so the mask on ``load`` carries that information
    # and inferring the width from the physical shape would read past the view. Fixed
    # width slices into a static parent get a static origin from
    # :func:`_crop_origin_shape_type_tree` instead.
    return TlaVectorSSATypeDescriptor(
        _static_lane_count_or_none(tensor_type.origin_shape),
        tensor_type.element_type,
    )


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


def _resolve_copy_l2_cache_policies(
    route: tuple[str, str],
    cache_policy: CachePolicy | None,
) -> tuple[CachePolicy, CachePolicy]:
    """Map optional ``cache_policy`` onto GM copy endpoints."""
    if cache_policy is None:
        return CachePolicy.NORMAL, CachePolicy.NORMAL
    if not isinstance(cache_policy, CachePolicy):
        raise TypeError(
            "copy cache_policy expects tla.params.CachePolicy, "
            f"got {type(cache_policy).__name__}"
        )
    src_is_gm = route[0] == "gm"
    dst_is_gm = route[1] == "gm"
    if not src_is_gm and not dst_is_gm:
        raise TlaLoweringError(
            "copy cache_policy requires a GM src and/or dst endpoint, "
            f"got route {route[0]}->{route[1]}"
        )
    return (
        cache_policy if src_is_gm else CachePolicy.NORMAL,
        cache_policy if dst_is_gm else CachePolicy.NORMAL,
    )


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
            "four-leaf NZFamily shape/stride with two-leaf origin/coord "
            "(schema v4 packs at most 4 physical leaves + 2 origin into "
            "13 launch slots); "
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
        batch_info=[],
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
_LINEAR_LAYOUT_TOKENS = frozenset({"RowMajor", "ColumnMajor", "VectorLayout"})
# Orientation of a linear layout, named once. Everything downstream -- stride
# derivation, the dynamic GM prologue, leading-dim inference -- must ask these
# rather than compare against "RowMajor"/"ColumnMajor" literally: a tag that is
# row-major but not spelled RowMajor otherwise falls into a column-major
# fallback that no validator can see, because the tag is a perfectly valid
# linear tag.
_ROW_MAJOR_LAYOUT_TOKENS = frozenset({"RowMajor"})
_COLUMN_MAJOR_LAYOUT_TOKENS = frozenset({"ColumnMajor"})
# The linear tags whose rank-1 form is the flat contiguous run ``(N):(1)``. The
# bc layer builds both through ``makeVectorTlaLayout``, whose stride is a literal 1,
# so neither can carry a real stride: anything that holds a buffer with its own
# strides must reject a strided one-dimensional view rather than describe it
# dense. Rank-1 ColumnMajor is deliberately absent -- it has no flat form at all.
_FLAT_1D_LAYOUT_TOKENS = frozenset({"RowMajor", "VectorLayout"})
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
_MAKE_TENSOR_LIKE_ON_CHIP_ADDRSPACES = frozenset(
    {"l1", "l0a", "l0b", "l0c", "ub", "biasbuf"}
)
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


def _validate_ceil_div_operand(value: int | Int32, name: str) -> None:
    if type(value) is int:
        literal = value
    elif isinstance(value, Int32):
        literal = value.value
        if not isinstance(literal, (int, float)):
            return
    else:
        raise TypeError(f"ceil_div {name} must be a Python int or Int32")

    if isinstance(literal, float):
        raise TypeError(f"ceil_div {name} must be an integer")
    if not -(1 << 31) <= literal < (1 << 31):
        raise ValueError(f"ceil_div {name} is outside the Int32 range")


def ceil_div(num1: int | Int32, num2: int | Int32) -> int | Int32:
    """Directory: Scalar Compute

    Description:
        Divide two integers and round the quotient up. A zero divisor returns zero.
        The result is a Python `int` when both arguments are Python `int` values,
        or a `tla.Int32` otherwise.

    Parameters:
        - `num1`: Dividend, a Python `int` or `tla.Int32`.
        - `num2`: Divisor, a Python `int` or `tla.Int32`.

    Constraints:
        - Both operands must be in the inclusive range 0 to 2,147,483,647. The caller must ensure this also holds for runtime values. Behavior outside this range is unspecified.
        - Constant operands can be used on the host or folded during kernel tracing. Runtime operands are supported in kernel scalar code.

    Example:
        ```python
        import catlass.tla as tla

        tile_count = tla.ceil_div(129, 64)
        zero = tla.ceil_div(129, 0)
        ```
    """
    _validate_ceil_div_operand(num1, "num1")
    _validate_ceil_div_operand(num2, "num2")
    lhs = num1.value if isinstance(num1, Int32) else num1
    rhs = num2.value if isinstance(num2, Int32) else num2
    if not isinstance(lhs, mlir_ir.Value) and not isinstance(rhs, mlir_ir.Value):
        quotient, remainder = divmod(lhs, rhs) if rhs else (0, 0)
        result = quotient + int(remainder != 0)
        return result if type(num1) is int and type(num2) is int else Int32(result)

    numerator = Int32(lhs)
    divisor = Int32(rhs)
    zero = Int32(0)
    nonzero = (divisor != zero).ir_value()
    safe_divisor = Int32(
        _mlir_arith.SelectOp(nonzero, divisor.ir_value(), Int32(1).ir_value()).result
    )
    quotient = numerator // safe_divisor
    remainder = numerator % safe_divisor
    rounded = quotient + (remainder != zero).to(Int32)
    return Int32(
        _mlir_arith.SelectOp(nonzero, rounded.ir_value(), zero.ir_value()).result
    )


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
    if (
        isinstance(shape_tree, tuple)
        and len(shape_tree) == 3
        and not isinstance(shape_tree[0], tuple)
    ):
        return (
            shape_tree[0],
            *_infer_padded_origin_tree_from_nz_family_shape(
                shape_tree[1:], for_op=for_op
            ),
        )
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
    for_op: str = "make_tensor",
) -> None:
    """Validate static shape/stride leaves using ``tla/layout.hpp`` traits."""

    if isinstance(shape_tree, tuple) and len(shape_tree) == 3:
        if (
            not isinstance(stride_tree, tuple)
            or len(stride_tree) != 3
            or isinstance(shape_tree[0], tuple)
            or isinstance(stride_tree[0], tuple)
        ):
            raise TlaLoweringError(
                f"tla.{for_op}: a batch mode must have one scalar shape and stride"
            )
        _validate_static_make_tensor_layout(
            shape_tree[1:],
            stride_tree[1:],
            dtype=dtype,
            layout_tag=layout_tag,
            for_op=for_op,
        )
        return
    shape_const_tree = _components_to_type_tree(shape_tree)
    stride_const_tree = _components_to_type_tree(stride_tree)

    if layout_tag in _LINEAR_LAYOUT_TOKENS:
        if len(shape_const_tree) == 1:
            # VectorLayout is unit-stride by definition: the bc layer builds its
            # rank-1 layout with a literal stride of 1, so an unknown (dynamic)
            # stride would be silently treated as 1 rather than honoured. The
            # one-dimensional linear spelling that does admit a runtime stride
            # is RowMajor, whose descriptor keeps the real value.
            if layout_tag == "VectorLayout" and stride_const_tree[0] is None:
                raise TlaLoweringError(
                    f"tla.{for_op} layout 'VectorLayout' requires a static stride "
                    "of 1; a dynamic stride has no representation in this layout "
                    "(use RowMajor for a strided rank-1 view)"
                )
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
            # The first physical mode has extent one, so its stride never
            # contributes to an address. Accept positive singleton strides as
            # well as the zero used by MakeMxScaleLayout.
            checks = (
                (shape00, 1),
                (shape10, _MX_SCALE_ELE_NUM_PER_C0),
                (stride10, 1),
                (stride11, _MX_SCALE_ELE_NUM_PER_C0),
            )
        elif layout_tag == "ColMajorMxScaleA":
            checks = (
                (shape00, 1),
                (shape10, _MX_SCALE_ELE_NUM_PER_C0),
                (stride01, _MX_SCALE_ELE_NUM_PER_C0),
                (stride10, 1),
            )
        elif layout_tag == "RowMajorMxScaleB":
            checks = (
                (shape00, _MX_SCALE_ELE_NUM_PER_C0),
                (shape10, 1),
                (stride00, 1),
                (stride11, _MX_SCALE_ELE_NUM_PER_C0),
            )
        elif layout_tag == "ColMajorMxScaleB":
            checks = (
                (shape00, _MX_SCALE_ELE_NUM_PER_C0),
                (shape10, 1),
                (stride00, 1),
                (stride01, _MX_SCALE_ELE_NUM_PER_C0),
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
            f"tla.{for_op} shape {shape_const_tree!r} and stride "
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
    if isinstance(origin_shape, tuple) and len(origin_shape) == 3:
        inner = _materialize_layout_trees_from_origin(origin_shape[1:], dtype, layout)
        if inner is None:
            return None
        shape, stride, coord, origin = inner
        # The default batch pitch covers the padded matrix storage. Explicit
        # layouts may choose a different pitch, including disjoint NZ interleaving.
        if layout == "RowMajor":
            pitch = shape[0] * stride[0]
        elif layout == "ColumnMajor":
            pitch = shape[1] * stride[1]
        else:
            pitch = 1
            for leaf in _flatten_tla_tuple(shape):
                pitch = pitch * leaf
        return ((origin_shape[0], *shape), (pitch, *stride), (0, *coord), origin_shape)
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
    Rank-1 ``VectorLayout`` uses ``(N):(1)``, ``coord=(0,)``. Rank-2 **coord** is always
    ``(0, 0)``. Naming follows ``tla::GetTileLayout`` / fractal ``MakeLayout``
    (``rows`` / ``cols`` / ``ELE_NUM_PER_C0`` / ``C0_NUM_PER_FRACTAL``).
    """
    if isinstance(origin_shape, tuple) and len(origin_shape) == 1:
        length = origin_shape[0]
        if isinstance(length, tuple):
            return None
        layout_tag = layout.strip()
        # A rank-1 linear tile is the flat contiguous run ``(N):(1)``: the bc
        # layer's ``makeVectorTlaLayout`` carries a literal stride of 1, so this
        # derivation cannot express a real one and must not be asked to. Callers
        # that hold a buffer with its own strides (``from_dlpack``) reject a
        # strided one-dimensional view themselves; this is why they have to.
        if layout_tag in _FLAT_1D_LAYOUT_TOKENS:
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


_RANK1_MISS = object()


def _flat_rank1_leaf_from_tree(tree: Any) -> Any:
    """Return a rank-1 tree's single ``int``/``None`` leaf, else ``_RANK1_MISS``.

    ``tla.tile_view`` on a rank-1 ``VectorLayout`` tensor keeps a one-leaf tree
    (``shape=(N,)``, ``coord=(offset,)``), which :func:`_flat_dim_pair_from_tree`
    correctly reports as unusable. Callers that only do leaf-wise arithmetic on
    rank-1 trees use this helper instead; the miss stays distinct from a dynamic
    leaf (``None``) so a dynamic one-d leaf is never mistaken for a rank mismatch.
    """
    if not isinstance(tree, tuple) or len(tree) != 1:
        return _RANK1_MISS
    leaf = tree[0]
    if leaf is None or isinstance(leaf, int):
        return leaf
    return _RANK1_MISS


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


def _flat_pair_sum_type_tree(a: Any, b: Any) -> Any | None:
    a_leaf = _flat_rank1_leaf_from_tree(a)
    b_leaf = _flat_rank1_leaf_from_tree(b)
    if a_leaf is not _RANK1_MISS and b_leaf is not _RANK1_MISS:
        return (None if a_leaf is None or b_leaf is None else a_leaf + b_leaf,)
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


def _crop_origin_shape_leaf(
    dim: int, origin: int | None, shape: int | None, coord: int | None
) -> int | None:
    if origin is None or shape is None or coord is None:
        return None
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
    return shape if origin - coord >= shape else origin - coord


def _crop_origin_shape_type_tree(
    parent_origin: Any, tile_shape: Any, tile_coord: Any
) -> Any | None:
    origin_leaf = _flat_rank1_leaf_from_tree(parent_origin)
    shape_leaf = _flat_rank1_leaf_from_tree(tile_shape)
    coord_leaf = _flat_rank1_leaf_from_tree(tile_coord)
    if (
        origin_leaf is not _RANK1_MISS
        and shape_leaf is not _RANK1_MISS
        and coord_leaf is not _RANK1_MISS
    ):
        if coord_leaf is None and origin_leaf is not None and shape_leaf is not None:
            # A slice whose offset is a loop induction variable: ``min(shape, origin -
            # coord)`` cannot be folded without the offset. Only keep the tile width
            # static when every legal offset still yields a full tile -- the parent
            # extent is a whole multiple of the tile width, so a tail block never
            # exists. Otherwise the valid length is unknown and must stay dynamic;
            # pinning it to ``shape`` over-reports a short tail block (a 160-element
            # run sliced by 128 would claim 128 valid lanes where only 32 remain).
            if (
                shape_leaf > 0
                and origin_leaf >= shape_leaf
                and origin_leaf % shape_leaf == 0
            ):
                return (shape_leaf,)
            return (None,)
        return (_crop_origin_shape_leaf(1, origin_leaf, shape_leaf, coord_leaf),)
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
        out.append(_crop_origin_shape_leaf(dim, origin, shape, coord))
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

    def pack_one(c: Any) -> tuple[list[mlir_ir.Value], Any]:
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
            for d, _ in child_packs:
                dyns.extend(d)
            return (
                dyns,
                tuple(tree for _, tree in child_packs),
            )
        _require_index(op_name, "leaf", c, 0)
        const = _const_int_value(c)
        if const is not None:
            return ([], int(const))
        return ([_as_index_value(c)], None)

    dyn: list[mlir_ir.Value] = []
    for c in components:
        d, tree = pack_one(c)
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
        if (
            type(value).is_integer
            and not type(value).signed
            and value.value.type != value_type
        ):
            # Printing uses the operand type to select unsigned formatting and extension.
            return mlir_ir.Operation.create(
                "builtin.unrealized_conversion_cast",
                operands=[value.value],
                results=[value_type],
                loc=loc,
            ).results[0]
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

    Read contiguously from the view's starting address without following strides.
    The view shape describes the log and also bounds the printed element count.
    Placement, shape and length checks apply to outer, SIMD and SIMT calls;
    SIMT requires static shapes.
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
    in_vf = _runtime._has_enclosing_region("vec.func")
    if in_vf and not _runtime._in_simt_vec_func() and addrspace != "ub":
        _op_error("print", "tensor printing inside a SIMD vec.func requires UB storage")
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
    if in_vf and not _runtime._in_simt_vec_func():
        max_length = 1952 // dtype_size_bytes(descriptor.element_type)
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
        if isinstance(resolved_length, (bool, float)):
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


def _emit_vector_print(
    value: Any, length: Any, *, loc: mlir_ir.Location | None
) -> None:
    _require_frontend_state("print")
    _runtime._require_enclosing_region("print", "vec.func")
    register = _as_value(value)
    descriptor = _vector_ssa_type_for_mlir_value(register)
    if (
        descriptor is None
        or descriptor.element_type not in _PRINT_TENSOR_SUPPORTED_DTYPES
    ):
        _op_error(
            "print",
            f"register dtype must be one of {_PRINT_TENSOR_SUPPORTED_DTYPES_TEXT}",
        )
    capacity = 256 // dtype_size_bytes(descriptor.element_type)
    if length is None:
        length = descriptor.valid_lanes or capacity
    resolved = _resolve_bound_value(length)
    if isinstance(resolved, (bool, float)) or (
        isinstance(resolved, Numeric) and not type(resolved).is_integer
    ):
        _op_error("print", "register prefix length must be an integer")
    if isinstance(resolved, mlir_ir.Value) and not (
        mlir_ir.IntegerType.isinstance(resolved.type)
        or isinstance(resolved.type, mlir_ir.IndexType)
    ):
        _op_error("print", "register prefix length must be an integer")
    static_length = _const_int_value(length)
    if static_length is not None and not 1 <= static_length <= capacity:
        _op_error("print", f"register prefix length must be between 1 and {capacity}")
    _tla_ops_gen.print_vector(register, _as_i64_value(length, loc=loc), loc=loc)


def print(*args: object, **kwargs: object) -> None:
    """Directory: Debug APIs
    Description:
        Print scalars, formatted strings, Tensor prefixes, or SIMD register values from a kernel.

        Parameters:
        - *`args`* (`object`): Values to print (variadic positional arguments). Required.
        - **`kwargs`** (`object`): Keyword arguments are not accepted; passing any raises an error. Optional.

        Constraints:
        - Call inside `tla.cube()` / `tla.vector()` or their SIMD/SIMT `tla.vec.func` regions; output order across threads, cores and channels is unspecified.
        - Supported dtypes are `f16`, `f32`, `i8`, `i16`, `i32`, `u8`, `u16` and `u32`. Cube prints GM/L1/L0C Tensors, outer Vector prints GM/UB Tensors, SIMD prints scalars, registers and UB Tensors, and SIMT prints per-thread scalars and statically shaped GM/UB Tensors.
        - Strings must be host-side `str` values with ASCII characters only and no NUL bytes. A format string accepts at most eight empty `{}` fields, whose count must equal the number of arguments; format specifiers, indexes and names such as `{:d}`, `{0}` and `{name}` are rejected. `{{`/`}}` print literal braces and `%` prints literally.
        - GM/UB Tensors are rank 1 or 2; printing reads a contiguous prefix from the view start and ignores strides. The caller ensures the data is ready and readable.
        - `length` is an integer element count, omitted only for a static Tensor whose elements all fit one call. A register defaults to its valid element count, or its physical capacity when that count is dynamic.
        - L1 requires Cube placement, a 32-byte-aligned zN/nZ dense prefix with a 32-byte C0 dimension, and at most 8 elements; L0C requires Cube placement, L0Clayout and a 32-byte-aligned 16x16 view (256 elements). L1 and L0C cannot be combined, and L0A/L0B are unsupported.
        - SIMD orders print reads against preceding/subsequent Vector stores. SIMT printing does not synchronize threads; shared UB requires a block-wide barrier. Other dependencies remain the caller's responsibility.
        - Printing adds buffer, transfer and stream-synchronization costs; remove calls before performance measurements.
        - Per-call limits, alignment, runtime length handling, synchronization examples, log capacity, truncation and error recovery are described in the Printing Guide.

        Example:
        ```python
        # source_gm and source_ub contain 64 ready FP32 elements.
        with tla.vector():
            tla.print("length={}", 64)
            tla.print(source_gm, 8)
            with tla.vec.func(mode="simd"):
                values = source_ub.load()
                tla.print(values, 4)
                tla.print(source_ub, 4)

            with tla.vec.func(mode="simt", thread_block_dim=32):
                thread, _, _ = tla.arch.thread_idx()
                if thread < 4:
                    tla.print("thread={} value={}", thread, source_gm[thread])
                if thread == 0:
                    tla.print(source_gm, 8)
        ```
    """
    if kwargs:
        _op_error("print", "does not accept keyword arguments")
    if len(args) < 1:
        _op_error("print", f"expects at least one positional argument; got {len(args)}")

    original_value = args[0]
    value = _resolve_bound_value(original_value)
    if _category(original_value) == "vector_ssa":
        if len(args) > 2:
            _op_error(
                "print", "register printing expects a value and optional prefix length"
            )
        loc = (
            _capture_user_loc()
            if _runtime._current_frontend_state() is not None
            else None
        )
        return _emit_vector_print(value, args[1] if len(args) == 2 else None, loc=loc)
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


@dsl_user_op
def trap(*, loc: mlir_ir.Location | None = None) -> None:
    """Directory: Debug APIs
    Description:
        Terminate the AI Core with a trap exception: on device the core stops
        and the kernel launch fails, and every output of the running kernel is
        undefined.

        Parameters:
        - None.

        Constraints:
        - Call inside a `@tla.kernel` function: the kernel body itself (outside
          any `tla.cube()` / `tla.vector()` region), or the scalar code inside
          those regions (the AIC / AIV scalar units). Placement inside a SIMD or
          SIMT `tla.vec.func` is rejected at compile time; that lowering is not
          implemented yet, so hoist the check out of the region.
        - Call it only when the kernel must fail. `tla.trap` is unconditional and
          does not return; code after it is unreachable at run time.
        - After a trap the stream and context are not recoverable. Exit the
          process instead of reusing the stream or launching again.

        Example:
        ```python
        with tla.vector():
            divisor = src_gm[0]
            if divisor == 0:
                tla.trap()
        ```
    """
    api = "trap"
    _require_frontend_state(api)
    if _runtime._has_enclosing_region("vec.func"):
        _op_error(
            api,
            "requires scalar code outside tla.vec.func; hoist the check out of the region",
        )
    _tla_ops_gen.trap(loc=loc)


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


def _resolve_allocation_alignment(
    op_name: str, mem_scope: AddressSpace, byte_alignment: int
) -> int:
    """Validate the byte alignment of an on-chip allocation and return it.

    The space's local-memory alignment is the only admissible alignment: an
    address either starts on an alignment boundary or it does not, so the
    request must be a multiple of it. A multiple is kept as asked; anything
    else -- including a request below it -- is refused rather than silently
    raised, because rounding it up would mean placing the buffer at an
    alignment the caller did not ask for.

    Both the start address and the allocation size follow this same
    ``byte_alignment``: the size is rounded up to it, so a short buffer still
    occupies a whole alignment period.
    """
    from .base_dsl.arch import _get_localmem_align_bytes

    scope = mem_scope.name.upper()
    localmem_alignment = _get_localmem_align_bytes(scope)
    if byte_alignment % localmem_alignment:
        _op_error(
            op_name,
            f"invalid argument 'byte_alignment': expected a multiple of the "
            f"{localmem_alignment}-byte {scope} local-memory alignment, got {byte_alignment}. "
            f"An address aligned to anything else would start mid-block.",
        )
    return byte_alignment


def _require_allocation_fits(
    op_name: str,
    mem_scope: AddressSpace,
    byte_alignment: int,
    size_bytes: int,
) -> None:
    """Refuse a request that no longer fits above the ones already granted.

    Each space is handed out the way ``tla-lower-ptr`` hands it out: a request
    starts at the previous high-water mark rounded up to its own alignment, and
    the mark advances by that start plus the size rounded up. Keeping the two
    layers on the same rule means a kernel the frontend accepts cannot be
    refused later for a total the frontend already counted.

    UB is capped only at the whole buffer here. Its sharper ceilings -- 248 KiB
    for SIMD, less the Data Cache for SIMT, the full buffer with both
    ``cce-disable-*-reserved-ubuf`` switches -- depend on the core type and the
    compile options, neither of which exists while tracing, so ``tla-lower-ptr``
    owns them. A dynamic UB region costs nothing here: it starts above the static
    allocations rather than being one.
    """
    from .base_dsl.arch import get_localmem_capacity_bytes

    state = _runtime._current_frontend_state()
    if state is None:
        # The only caller (``allocate``) already guards with
        # ``_require_frontend_state``; reaching here without one is a caller bug,
        # not a supported stateless mode to account for.
        raise TlaIRNotExecutableError(
            f"tla.{op_name} requires the kernel frontend state"
        )
    scope = mem_scope.name
    capacity = get_localmem_capacity_bytes(scope.upper())
    already = state.allocated_bytes_by_scope.get(scope, 0)
    aligned_start = _round_up(already, byte_alignment)
    end = aligned_start + size_bytes
    if end > capacity:
        _op_error(
            op_name,
            f"allocation exceeds {scope.upper()} capacity: {already} bytes "
            f"already allocated, + {aligned_start - already} bytes of alignment "
            f"padding to offset {aligned_start}, + {size_bytes} bytes requested "
            f"= {end} bytes, past the {capacity}-byte capacity. Split the work "
            f"across kernels or shrink the buffers.",
        )
    state.allocated_bytes_by_scope[scope] = aligned_start + size_bytes


@dsl_user_op
def make_shape(
    *components: IndexTree,
    loc: mlir_ir.Location | None = None,
) -> TlaShape:
    """Directory: Basic Data Types and Operations
    Description:
        Build a shape; components may be nested tuples.

        Inside ``@tla.kernel``: builds packed `!tla.shape`. Outside a kernel:
        same ``_Shape`` type with Host int-tree components for
        ``tla.make_layout`` / Host ``tla.make_tensor``.

        Parameters:
        - *`components`* (`IndexTree`): Shape components per dimension. Nested
          tuples are used for `zN` / `nZ` / `L0Clayout` / `zNUnAlign`
          physical layouts. Required. Host path: Python ints only.

        Constraints:
        - Kernel path: must be inside `@tla.kernel`.
        - Host path: no frontend; int trees only.
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
    if _runtime._current_frontend_state() is None:
        host_components = tuple(
            _require_host_index_components(c, arg="make_shape") for c in components
        )
        return _Shape._from_host(host_components)
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
        Build a coordinate tree.

        Inside ``@tla.kernel``: builds packed `!tla.coord`. Outside a kernel:
        same ``_Coord`` type with Host int-tree components.

        Parameters:
        - *`components`* (`IndexTree`): Coordinate components per dimension. Required.
          Host path: Python ints only.

        Constraints:
        - Kernel path: must be inside `@tla.kernel`.
        - Host path: no frontend; int trees only.
        - Provide at least one coordinate component.

        Example:
        ```python
        coord = tla.make_coord(block_row, 0)
        ```
    """
    if len(components) == 0:
        _op_error("make_coord", "expected at least 1 component")
    if _runtime._current_frontend_state() is None:
        host_components = tuple(
            _require_host_index_components(c, arg="make_coord") for c in components
        )
        return _Coord._from_host(host_components)
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
        Build a stride (same nesting rules as `make_shape`).

        Inside ``@tla.kernel``: builds packed `!tla.stride`. Outside a kernel:
        same ``_Stride`` type with Host int-tree components.

        Parameters:
        - *`components`* (`IndexTree`): Stride components per dimension. Required.
          Host path: Python ints only.

          Common patterns:

          | Layout | Typical `make_stride(...)` | Meaning |
          | --- | --- | --- |
          | RowMajor 2D `(M, N)` | `(N, 1)` | Row step is `N` elements; column step is 1 |
          | ColumnMajor 2D `(M, N)` | `(1, M)` | Row step is 1; column step is `M` |
          | `zN` / `nZ` / … | nested `((s00, s01), (s10, s11))` | Same nesting as physical `shape`; values must match the layout tag |

          For `zN` / `nZ` / `L0Clayout` / `zNUnAlign`, C0 is 32 bytes and
          the M-side block size is 16. Let
          `elems_per_c0 = 32 // sizeof(dtype)` (f16→16, f32→8) and
          `elems_per_block = elems_per_c0 * 16`.

        Constraints:
        - Kernel path: must be inside `@tla.kernel`.
        - Host path: no frontend; int trees only.
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
    if _runtime._current_frontend_state() is None:
        host_components = tuple(
            _require_host_index_components(c, arg="make_stride") for c in components
        )
        return _Stride._from_host(host_components)
    _require_frontend_state("make_stride")
    v = _pack_stride(components, loc=loc)
    return _Stride(stride_value=v, components=tuple(components))


@dsl_user_op
def make_layout(
    shape: Any,
    stride: Any,
    *,
    origin_shape: Any | None = None,
    layoutTag: _LayoutTag | None = None,
    loc: mlir_ir.Location | None = None,
) -> TlaLayout:
    """Directory: Basic Data Types and Operations

    Description:
        Compose a layout from shape / stride.

        Inside ``@tla.kernel``: builds `!tla.layout` (maps to `tla.make_layout`);
        ``shape`` / ``stride`` must come from ``make_shape`` / ``make_stride``.

        Outside a kernel: same ``_Layout`` type; ``shape`` / ``stride`` from Host
        ``make_shape`` / ``make_stride`` (or equivalent int trees) for
        Host ``tla.make_tensor``.

        Parameters:
        - `shape` (`_Shape` | int tree): From `tla.make_shape`. Host also accepts a
          Python int tree. RowMajor / ColumnMajor: `(M, N)` or `(B, M, N)`.
          NZ-family layouts use `((m0, m1), (n0, n1))`, optionally preceded by
          a scalar batch mode `(B, (m0, m1), (n0, n1))`.
        - `stride` (`_Stride` | int tree): From `tla.make_stride`. Host also accepts
          a Python int tree with the same nesting as `shape`.
        - `origin_shape` (`_Shape` | int tree | None): Logical working shape.
          Optional, default `None` (inferred / equals `shape` on Host).
        - `layoutTag` (`_LayoutTag | None`): Layout tag; see
          **System Variable Access / Layout Tags**. Optional, default `None`
          (`RowMajor`).

        Constraints:
        - Kernel path: must be inside `@tla.kernel`; `shape` / `stride` from
          `make_shape` / `make_stride` (SSA).
        - Host path: no frontend; Host `make_shape` / `make_stride` or int trees;
          result is for Host `tla.make_tensor`, not Kernel IR.
        - For RowMajor / ColumnMajor, an omitted `origin_shape` is inferred as
          `shape`. For `zN` / `nZ` / `L0Clayout` / `zNUnAlign`, Kernel infers
          `(m0*m1, n0*n1)`; Host defaults omitted origin to `shape`.
        - Do **not** pass a plain 2D `shape` with a `zN` / `nZ` /
          `L0Clayout` / `zNUnAlign` tag; that fails checks. Either build the nested
          physical shape, or use `make_tensor_like(ptr, like, layoutTag=...)` so
          the front end remaps from `like.origin_shape`.

        Example:
        ```python
        layout = tla.make_layout(
            tla.make_shape(256, 128),
            tla.make_stride(128, 1),
            layoutTag=tla.arch.RowMajor,
        )
        ```
    """
    in_kernel = _runtime._current_frontend_state() is not None
    if not in_kernel:
        # Validate tag early; keep the arch sentinel for ``_Tensor`` (not the
        # resolved MLIR token string).
        _resolve_arch_layout_tag(layoutTag, for_op="make_layout")
        host_shape = _coerce_host_tree(shape, cls=_Shape, arg="shape")
        host_stride = _coerce_host_tree(stride, cls=_Stride, arg="stride")
        # Omit origin → None (same as Kernel); Host make_tensor defaults it.
        host_origin = (
            None
            if origin_shape is None
            else _coerce_host_tree(origin_shape, cls=_Shape, arg="origin_shape")
        )
        return _Layout._from_host(
            host_shape,
            host_stride,
            layout_tag=arch.RowMajor if layoutTag is None else layoutTag,
            origin_shape=host_origin,
        )

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
        Build an on-chip `!tla.layout` from a flat matrix or batched-matrix shape, element type, and layout tag.

        Parameters:
        - `shape` (`tuple`): Flat logical extent `(rows, cols)` or `(batch, rows, cols)`. Each leaf is
          an `int` or a runtime `Numeric` (dynamic leaf). Required.
        - `dtype` (`Numeric` subclass): Element type; drives the C0/fractal
          geometry of the remapped physical trees. Required.
        - `layoutTag` (`_LayoutTag | None`): Layout tag; see
          **System Variable Access / Layout Tags**. Optional, default `None`
          (`RowMajor`).

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - The composed layout is meant to be landed on an on-chip pointer via
          `tla.make_tensor(ptr, layout)`. It also serves a GM pointer when that
          buffer is already packed the way the tag says -- a prepacked `zN` or
          `nZ` weight, say -- which is how `tla.copy`'s GM → L1 route spells such
          a source; it is not for binding ordinary host memory, whose layout
          comes from the host tensor instead.
        - `shape` must be a flat 2-tuple or 3-tuple; nested trees and `None`
          leaves are rejected. The batch pitch covers the complete padded
          matrix storage; explicit layouts can choose independent pitches.
        - The origin shape of the resulting layout equals `shape` (the aligned
          extent); pair with `tla.make_tensor(ptr, layout)` to land it on an
          on-chip pointer.
        - This API accepts only: `RowMajor`, `ColumnMajor`, `zN`, `nZ`,
          `zZMxScale`, `nNMxScale`, `L0Clayout`, `zNUnAlign`.
        - `zZMxScale` / `nNMxScale` require element type `tla.Float8E8M0`.
          GM-side MxScale tags are rejected.

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
        or len(shape) not in (2, 3)
        or any(isinstance(leaf, (tuple, list)) for leaf in shape)
    ):
        _op_error(
            "make_layout_with_tag",
            "expected a flat 2D or 3D logical shape tuple with int or "
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
    _reject_simt_tensor_construction("tile_view")
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
        Extract a flat 2-D or 3-D sub-tile of a tensor (shares the source buffer).

        Parameters:
        - `tensor` (`Tensor`): Source `!tla.tensor`. Required.
        - `coord` (`TlaCoord`): Tile start from `tla.make_coord`, flat 2-D or 3-D
          element offsets relative to the source. Required.
        - `shape` (`TlaShape`): Requested tile extent from `tla.make_shape`,
          flat 2-D or 3-D with the same logical rank as the source. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Result `coord` is `coord + source.coord`; `origin_shape` is `shape`
          cropped to the source extent; physical `shape` is regenerated from
          that crop; `stride` is unchanged.
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
    _reject_simt_tensor_construction(op_name)

    coord_components = coord._components
    shape_components = shape._components
    if (
        len(coord_components) not in (2, 3)
        or any(isinstance(leaf, (tuple, str)) for leaf in coord_components)
        or len(shape_components) != len(coord_components)
        or any(isinstance(leaf, (tuple, str)) for leaf in shape_components)
    ):
        _op_error(
            op_name,
            "get_tile expects a flat 2-D or 3-D coord and shape; got coord="
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
        or len(parent_coord) != len(coord_components)
        or not isinstance(parent_origin, tuple)
        or len(parent_origin) != len(coord_components)
    ):
        _op_error(
            op_name,
            "get_tile source coord and origin_shape must match the requested logical rank; got "
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
        Build a tensor from an explicit pointer, layout, and optional coord.

        Inside ``@tla.kernel``: builds packed `!tla.tensor` IR.
        Outside a kernel: same entry binds a Host tensor from Host
        ``tla.make_ptr`` / ``tla.make_layout`` (no DLPack).

        Parameters:
        - `ptr` (`Pointer`): Underlying data pointer (`!tla.ptr` in Kernel, or
          Host ``make_ptr``). Required.
        - `layout` (`TlaLayout`): Tensor layout from `tla.make_layout`. Required.
        - `coord` (`CoordLike | None`): Optional start coordinate; treated as zeros when omitted. Optional, default `None`.

        Constraints:
        - Kernel path: must be inside `@tla.kernel`; pointer, layout, and coord
          must match the target address space and dtype.
        - Host path: no frontend; Host ``make_ptr`` / ``make_layout`` only.
          Nested NZFamily 2x2 trees must satisfy layout traits and have a flat
          logical origin `(M, N)`. Trait-valid partial views are allowed. For a
          full-tile remap and buffer validation, use ``from_dlpack``. ``loc`` is ignored.
        - `coord` defaults to a zero coord matching the layout's rank (rank-2 ->
          `make_coord(0, 0)`, rank-1 -> `make_coord(0)`). Element type and address
          space come from `ptr`; layout tag, shape, stride, and origin
          come from the layout (origin defaults to `shape`).
        - Lowering supports RowMajor, ColumnMajor, zN, nZ, L0Clayout, and
          zNUnAlign. For `zN` / `nZ` / `L0Clayout` / `zNUnAlign`, physical
          `shape` / `stride` are nested 2x2, while logical coord / `origin_shape`
          stay 2D `(M, N)`. If `make_layout` omitted `origin_shape`, the logical
          size is inferred from the physical shape (for example `(m0*m1, n0*n1)`).
        - Kernel-local batched views prepend a scalar batch mode to the physical
          shape and stride, with flat `(B, M, N)` origin and coordinates.
          The batch stride is in elements and may describe disjoint interleaved
          NZ matrices. In the kernel, batched views support GM-to-L1 `copy` and
          `get_tile`. For Host `from_dlpack` bindings of NZ layouts, the logical
          origin is two-dimensional `(M, N)`.
        - A full Kernel compile requires `ptr` to carry backing storage; an on-chip
          pointer from `allocate` (optionally via `recast_ptr`) is the supported
          form for runnable kernels.

        Example:
        ```python
        # Kernel IR:
        tensor = tla.make_tensor(ptr, layout, coord=tla.make_coord(0, 0))

        # Host bind (no DLPack):
        host = tla.make_tensor(
            tla.make_ptr(tla.Float16, int(buf.data_ptr()), tla.AddressSpace.gm),
            tla.make_layout(
                tla.make_shape(128, 64),
                tla.make_stride(64, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ```
    """
    if _runtime._current_frontend_state() is None:
        del loc
        from .tla.runtime import _host_make_tensor

        return _host_make_tensor(ptr, layout, coord=coord)

    _require_category("make_tensor", "ptr", ptr, "pointer", 0)
    if not isinstance(layout, _Layout):
        _op_error(
            "make_tensor",
            "invalid argument 'layout': expected tla.make_layout (TlaLayout); "
            f"got {_type_name(layout)}",
        )
    _require_frontend_state("make_tensor")
    _reject_simt_tensor_construction("make_tensor")
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
    if dtype not in _MAKE_TENSOR_SUPPORTED_ELEMENT_TYPES and not (
        (dtype in _B64_STAGING_DTYPES and addr == "ub")
        or (
            dtype in _B64_STAGING_DTYPES
            and addr == "gm"
            and layout._layout_tag == "RowMajor"
        )
        or (
            dtype in _B64_STAGING_DTYPES
            and addr in _B64_STAGING_ADDRSPACES
            and layout._layout_tag == _B64_STAGING_LAYOUT_TAG
        )
    ):
        raise TlaLoweringError(
            f"tla.make_tensor expects a supported element type, got [{dtype}]"
            + (
                "; i64/u64/f64 are accepted in UB and RowMajor GM tiles, and "
                "i64/u64/f64 as GM/L1 VectorLayout tiles"
                if dtype in _B64_STAGING_DTYPES
                else ""
            )
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
        if logical_rank not in (1, 2, 3) or stride_leaf_count != logical_rank:
            raise TlaLoweringError(
                f"tla.make_tensor supports 1-D, 2-D or 3-D linear layouts with matching strides (got shape rank "
                f"{shape_leaf_count}, stride rank {stride_leaf_count})"
            )
        if len(_flatten_tla_tuple(origin_tree)) != logical_rank:
            raise TlaLoweringError(
                "tla.make_tensor linear origin_shape rank must match layout rank"
            )
    elif layout._layout_tag in _NZ_FAMILY_LAYOUT_TOKENS:
        batched = (
            len(origin_tree) == 3
            and len(shape_tree) == 3
            and len(stride_tree) == 3
            and not isinstance(shape_tree[0], tuple)
            and not isinstance(stride_tree[0], tuple)
        )
        matrix_shape = shape_tree[1:] if batched else shape_tree
        matrix_stride = stride_tree[1:] if batched else stride_tree
        if not _is_nz_family_2x2_tree(matrix_shape) or not _is_nz_family_2x2_tree(
            matrix_stride
        ):
            raise TlaLoweringError(
                f"tla.make_tensor layout {layout._layout_tag!r} expects shape and "
                "stride as two 2-leaf groups ((m0, m1), (n0, n1))"
            )
        logical_rank = 3 if batched else 2
        if len(origin_tree) != logical_rank or any(
            isinstance(x, tuple) for x in origin_tree
        ):
            raise TlaLoweringError(
                "tla.make_tensor expects flat matrix or batch-matrix origin_shape"
            )
    else:
        raise TlaLoweringError(
            f"tla.make_tensor does not support layout {layout._layout_tag!r}"
        )

    _validate_static_make_tensor_layout(
        shape_tree,
        stride_tree,
        dtype=dtype,
        layout_tag=layout._layout_tag,
        for_op="make_tensor",
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
    if layout._layout_tag in _NZ_FAMILY_LAYOUT_TOKENS and any(
        isinstance(x, tuple) for x in coord_tree
    ):
        raise TlaLoweringError(
            f"tla.make_tensor layout {layout._layout_tag!r} expects a flat "
            f"logical {logical_rank}-D coord"
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
    _reject_simt_tensor_construction("make_tensor_like")
    ptr_value = _as_value(ptr)
    like_value = _as_value(like)
    try:
        like_type = _tla_tensor_type_for_mlir_value(like_value)
    except TlaLoweringError as exc:
        raise TlaLoweringError(
            "tla.make_tensor_like expects ``like`` to carry structured Tla tensor metadata; "
            f"got {str(like_value.type)!r}"
        ) from exc
    if len(like_type.origin_shape) == 3:
        _op_error(
            "make_tensor_like",
            "batched views currently require make_tensor with an explicit layout",
        )
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
    addr = ptr_ty.addrspace
    # Resolve the effective layout tag early enough that the b64 VectorLayout
    # staging gate can run before the generic element-type refusal. When the
    # caller omits layoutTag, L1 inference keeps VectorLayout as VectorLayout
    # (see below), so using like_type.layout_tag here matches that path.
    _like_layout_for_gate = (
        _name_token(layoutTag) if layoutTag is not None else like_type.layout_tag
    )
    if dtype not in _MAKE_TENSOR_SUPPORTED_ELEMENT_TYPES and not (
        (dtype in _B64_STAGING_DTYPES and addr == "ub")
        or (
            dtype in _B64_STAGING_DTYPES
            and addr in _B64_STAGING_ADDRSPACES
            and _like_layout_for_gate == _B64_STAGING_LAYOUT_TAG
        )
    ):
        raise TlaLoweringError(
            f"tla.make_tensor_like expects a supported element type, got [{dtype}]"
            + (
                "; i64/u64/f64 are accepted in UB or as L1 VectorLayout tiles"
                if dtype in _B64_STAGING_DTYPES
                else ""
            )
        )
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
            # A flat VectorLayout source (the gemv A row) keeps that form on
            # L0A, matching the l1 inference above; every other source is the
            # fractal zN operand mmad reads.
            layout = "VectorLayout" if like_type.layout_tag == "VectorLayout" else "zN"
        elif addr == "l0b":
            layout = "nZ"
        elif addr == "l0c":
            layout = "L0Clayout"
        elif addr == "l1":
            if like_type.layout_tag in ("RowMajor", "zN"):
                layout = "zN"
            elif like_type.layout_tag in ("ColumnMajor", "nZ"):
                layout = "nZ"
            elif like_type.layout_tag == "VectorLayout":
                # VectorLayout stays VectorLayout. Folding it into zN would drop
                # the flat (rank-1) form the VectorLayout copy specializations
                # match, and a 1-D tile has no zN fractal structure to infer
                # from. A 1-D RowMajor source keeps its existing zN inference
                # above.
                layout = "VectorLayout"
            else:
                raise TlaLoweringError(
                    f"tla.make_tensor_like cannot infer layout for addrspace l1 "
                    f"with likeTensor layoutTag '{like_type.layout_tag}'; please specify layoutTag explicitly"
                )
        elif addr == "biasbuf":
            # The bias table is one contiguous run of elements, so it mirrors the
            # VectorLayout source it is loaded from. There is no other layout it
            # can take, so any other source tag asks the caller to be explicit.
            if like_type.layout_tag == "VectorLayout":
                layout = "VectorLayout"
            else:
                raise TlaLoweringError(
                    f"tla.make_tensor_like cannot infer layout for addrspace biasbuf "
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

    if (
        addr == "ub"
        and dtype in _B64_STAGING_DTYPES
        and layout not in ("RowMajor", "VectorLayout")
    ):
        raise TlaLoweringError("64-bit UB data requires RowMajor or VectorLayout")

    ptr_alignment = ptr_ty.alignment
    remapped = _remap_tensor_like_prefix_fields_for_layout_trees(
        like_type.origin_shape,
        dtype,
        layout,
        linear_stride_alignment_bytes=_CATLASS_BYTE_PER_C0,
    )
    # ``make_tensor_like(tile.ptr, tile)`` re-shapes the tile's own pointer, so it
    # names the same storage. That makes it an alias rather than a freshly
    # allocated buffer: it keeps the reference's view start and strides, which
    # may be runtime values, instead of re-deriving them from the child shape.
    # Only a same-layout alias may borrow that geometry: a new layout has its own
    # rank and strides, so keeping the reference's trees would pair the new tag
    # with the old shape.
    ptr_owner = ptr_value.owner
    aliases_reference = (
        layout == like_type.layout_tag
        and getattr(ptr_owner, "name", None) == "tla.tensor_ptr"
        and len(ptr_owner.operands) == 1
        and str(ptr_owner.operands[0]) == str(like_value)
    )
    if remapped is not None and not aliases_reference:
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
    attributes = {"layoutTag": layout_tag_attr}
    if aliases_reference:
        attributes["aliasOfReference"] = mlir_ir.UnitAttr.get(ctx)
    op = mlir_ir.Operation.create(
        "tla.make_tensor_like",
        operands=[ptr_value, like_value],
        results=[_coerce_type(result_desc)],
        attributes=attributes,
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
        if remapped is not None and not aliases_reference:
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
    ("l1", "biasbuf"),
}

_COPY_VECTOR_ROUTES = {("gm", "ub"), ("ub", "gm"), ("ub", "l1"), ("ub", "ub")}

# Data types of the Ascend C LoadDataWithStride operands and their L1 staging copy.
_IMG2COL_L0B_DTYPES = ("f16", "bf16", "f32", "i32", "u32")
_IMG2COL_L0A_DTYPES = (*_IMG2COL_L0B_DTYPES, "i8", "u8", "hif8", "f8e4m3fn", "f8e5m2")

# fmt: off
# Cube copy
_SUPPORTED_CUBE_COPY_ROUTES = {
    # (src_addr, src_layout, dst_addr, dst_layout) : dtypes
    # GM->L1
    ("gm", "RowMajor", "l1", "zN"): (
        "f32",
        "f16",
        "bf16",
        "i8",
        # AscendC bool is b8; TileCopyTla accepts bool (SizeOfBits==8).
        "i1",
        "hif8",
        "i16",
        "i32",
        "u8",
        "u16",
        "u32",
        "f8e4m3fn",
        "f8e5m2",
        "f4e2m1",
        "f4e1m2",
    ),
    ("gm", "ColumnMajor", "l1", "nZ"): (
        "f32",
        "f16",
        "bf16",
        "i8",
        "i1",
        "hif8",
        "i16",
        "i32",
        "u8",
        "u16",
        "u32",
        "f8e4m3fn",
        "f8e5m2",
        "f4e2m1",
        "f4e1m2",
    ),
    # A GM tile already laid out in fractal order. The source tag mirrors the
    # destination's zN, and the copy is a plain block move -- no ND -> zN
    # reorder. Only the aligned zN is a GM source here; the tight M-axis packing
    # belongs to the UB -> L1 route.
    ("gm", "zN", "l1", "zN"): (
        "f32",
        "f16",
        "bf16",
        "i8",
        "i16",
        "i32",
        "u8",
        "u16",
        "u32",
        "i1",
        "hif8",
        "f8e4m3fn",
        "f8e5m2",
        "f4e2m1",
        "f4e1m2",
    ),
    # The transposed mirror: a GM tile already packed as nZ (C0-row by 16-column
    # fractals) copied to an L1 nZ tile. nZ has no unaligned variant at all.
    ("gm", "nZ", "l1", "nZ"): (
        "f32",
        "f16",
        "bf16",
        "i8",
        "i16",
        "i32",
        "u8",
        "u16",
        "u32",
        "i1",
        "hif8",
        "f8e4m3fn",
        "f8e5m2",
        "f4e2m1",
        "f4e1m2",
    ),
    # NC1HWC0 feature map: the same tree on both sides, so this is a run of
    # contiguous C1 slices rather than a re-layout. It is what feeds tla.experimental.img2col.
    ("gm", "zNUnAlign", "l1", "zNUnAlign"): _IMG2COL_L0A_DTYPES,
    ("gm", "RowMajorMxScaleA", "l1", "zZMxScale"): ("f8e8m0",),
    ("gm", "ColMajorMxScaleA", "l1", "zZMxScale"): ("f8e8m0",),
    ("gm", "RowMajorMxScaleB", "l1", "nNMxScale"): ("f8e8m0",),
    ("gm", "ColMajorMxScaleB", "l1", "nNMxScale"): ("f8e8m0",),

    ("gm", "VectorLayout", "l1", "VectorLayout"): (
        "f32",
        "f16",
        "bf16",
        "i8",
        # AscendC bool is b8; TileCopyTla VectorLayout path accepts bool.
        "i1",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "f64",
        "f8e4m3fn",
        "f8e5m2",
        "hif8",
    ),
    ("l1", "VectorLayout", "biasbuf", "VectorLayout"): ("f32", "i32"),

    # L1->L0 — AscendC LoadData 2dv2 accepts u32 but not u16 on dav-3510.
    ("l1", "zN", "l0a", "zN"): ("f32", "f16", "bf16", "i8", "u32", "f8e4m3fn", "f8e5m2", "hif8"),
    ("l1", "nZ", "l0a", "zN"): ("f32", "f16", "bf16", "i8", "u32", "f8e4m3fn", "f8e5m2", "hif8"),
    # gemv A operand: a flat L1 row moved into a flat L0A row.
    ("l1", "VectorLayout", "l0a", "VectorLayout"): ("f32", "f16", "bf16", "i8", "f8e4m3fn", "f8e5m2", "hif8"),
    ("l1", "zN", "l0b", "nZ"): ("f32", "f16", "bf16", "i8", "u32", "f8e4m3fn", "f8e5m2", "hif8"),
    ("l1", "nZ", "l0b", "nZ"): ("f32", "f16", "bf16", "i8", "u32", "f8e4m3fn", "f8e5m2", "hif8"),
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
    ("gm", "RowMajor", "ub", "RowMajor"): (
        "i64",
        "u64",
        "f64",
        "f32",
        "f16",
        "bf16",
        "i32",
        "i16",
        "i8",
        # AscendC bool is b8 storage (1 byte/lane); GM↔UB stages like i8.
        "i1",
        "u32",
        "u16",
        "u8",
        "f8e4m3fn",
        "f8e5m2",
        "f8e8m0",
        "f4e2m1",
        "f4e1m2",
        "i4x2",
        "hif8",
        "f4e2m1x2",
        "f4e1m2x2",
    ),
    # Flat VectorLayout GM→UB: b64 + bool + ExtParams-safe scalars + hif8/fp8/fp4/i4x2.
    ("gm", "VectorLayout", "ub", "VectorLayout"): (
        "i64",
        "u64",
        "f64",
        "i1",
        "f32",
        "f16",
        "bf16",
        "i32",
        "i16",
        "i8",
        "u32",
        "u16",
        "u8",
        "f8e4m3fn",
        "f8e5m2",
        "f4e2m1",
        "f4e1m2",
        "hif8",
        "f4e2m1x2",
        "f4e1m2x2",
        "i4x2",
    ),
    ("ub", "RowMajor", "gm", "RowMajor"): (
        "i64",
        "u64",
        "f64",
        "f32",
        "f16",
        "bf16",
        "i32",
        "i16",
        "i8",
        "i1",
        "u32",
        "u16",
        "u8",
        "f8e4m3fn",
        "f8e5m2",
        "f8e8m0",
        "f4e2m1",
        "f4e1m2",
        "i4x2",
        "hif8",
        "f4e2m1x2",
        "f4e1m2x2",
    ),
    # Flat VectorLayout UB→GM: same dtype allow-list as GM→UB VectorLayout.
    ("ub", "VectorLayout", "gm", "VectorLayout"): (
        "i64",
        "u64",
        "f64",
        "i1",
        "f32",
        "f16",
        "bf16",
        "i32",
        "i16",
        "i8",
        "u32",
        "u16",
        "u8",
        "f8e4m3fn",
        "f8e5m2",
        "f4e2m1",
        "f4e1m2",
        "hif8",
        "f4e2m1x2",
        "f4e1m2x2",
        "i4x2",
    ),
    # UB -> UB is DataCopy (MOV.UB.TO.UB) throughout, with a ragged row rounded
    # up to whole 32-byte blocks, so every element width is carried, 1-byte
    # included. The MX types stay in their own denomination: the block scale is
    # one byte, and packed fp4 keeps 4-bit elements because this route is not
    # one of _FP4_BYTE_STAGED_ROUTES.
    ("ub", "RowMajor", "ub", "RowMajor"): (
        "f32",
        "f16",
        "bf16",
        "i32",
        "i16",
        "i8",
        "i1",
        "u32",
        "u16",
        "u8",
        "i64",
        "u64",
        "f64",
        "f8e4m3fn",
        "f8e5m2",
        "f8e8m0",
        "f4e2m1",
        "f4e1m2",
        "hif8",
        "i4x2",
    ),
    # UB -> UB transposing: the destination tile is tagged ColumnMajor, so the
    # same logical (M, N) shape lands as an (N, M) array. VNCHWCONV carries every
    # UB element width, unsigned ones included.
    ("ub", "RowMajor", "ub", "ColumnMajor"): (
        "f32",
        "f16",
        "bf16",
        "i32",
        "i16",
        "i8",
        "i1",
        "u32",
        "u16",
        "u8",
        "f8e4m3fn",
        "f8e5m2",
        "f8e8m0",
        "hif8",
        "i4x2",
    ),
    ("ub", "RowMajor", "l1", "zN"): (
        "f32",
        "f16",
        "bf16",
        "i8",
        "i1",
        "i16",
        "i32",
        "u8",
        "u16",
        "u32",
        "f8e4m3fn",
        "f8e5m2",
        "hif8",
        "f4e2m1",
        "f4e1m2",
    ),
    ("ub", "zN", "l1", "zN"): (
        "f32",
        "f16",
        "bf16",
        "i8",
        "i1",
        "i16",
        "i32",
        "u8",
        "u16",
        "u32",
        "f8e4m3fn",
        "f8e5m2",
        "hif8",
        "f4e2m1",
        "f4e1m2",
    ),
    # The block scale keeps its own fractal layouts on both sides.
    ("ub", "zZMxScale", "l1", "zZMxScale"): ("f8e8m0",),
    ("ub", "nNMxScale", "l1", "nNMxScale"): ("f8e8m0",),
    ("ub", "zNUnAlign", "l1", "zN"): (
        "f32",
        "f16",
        "bf16",
        "i8",
        "f8e4m3fn",
        "f8e5m2",
        "hif8",
    ),
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

    ("gm", "RowMajor", "-", "PER_TENSOR", False): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("gm", "RowMajor", "-", "PER_TENSOR", True): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("gm", "ColumnMajor", "-", "PER_TENSOR", False): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("gm", "ColumnMajor", "-", "PER_TENSOR", True): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("ub", "RowMajor", "nosplit", "PER_TENSOR", False): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("ub", "RowMajor", "nosplit", "PER_TENSOR", True): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("ub", "ColumnMajor", "nosplit", "PER_TENSOR", False): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("ub", "ColumnMajor", "nosplit", "PER_TENSOR", True): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },

    ("gm", "RowMajor", "-", "PER_CHANNEL", False): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("gm", "RowMajor", "-", "PER_CHANNEL", True): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("gm", "ColumnMajor", "-", "PER_CHANNEL", False): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("gm", "ColumnMajor", "-", "PER_CHANNEL", True): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("ub", "RowMajor", "nosplit", "PER_CHANNEL", False): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("ub", "RowMajor", "nosplit", "PER_CHANNEL", True): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("ub", "ColumnMajor", "nosplit", "PER_CHANNEL", False): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },
    ("ub", "ColumnMajor", "nosplit", "PER_CHANNEL", True): {
        "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        "i32": ("f16", "bf16", "i8", "u8"),
    },

    ("l1", "zN", "-", "PER_TENSOR", False): {"f32": ("f16", "bf16", "i8"), "i32": ("f16", "bf16", "i8")},
    ("l1", "zN", "-", "PER_TENSOR", True): {"f32": ("f16", "bf16", "i8"), "i32": ("f16", "bf16", "i8")},
    ("l1", "zN", "-", "PER_CHANNEL", False): {"f32": ("f16", "bf16", "i8"), "i32": ("f16", "bf16", "i8")},
    ("l1", "zN", "-", "PER_CHANNEL", True): {"f32": ("f16", "bf16", "i8"), "i32": ("f16", "bf16", "i8")},
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


# L0C→UB Fixpipe 1-byte destinations: Fixpipe RoundUp's the UB write extent
# (RowMajor N / ColumnMajor M) to a 32-element block regardless of unitFlag.
# That write needs a matching UB pitch, or pad spills into the next row / OOB.
# With unitFlag!=0, mmad still counts fractals by 16, so an unaligned extent
# can also hang. Static extents/pitches are checked here; dynamic values stay
# a caller contract.
_FIXPIPE_UB_1BYTE_DTYPES = frozenset({"i8", "u8", "f8e4m3fn", "hif8"})
_FIXPIPE_UB_1BYTE_ALIGN = 32


def _round_up_pos_int(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _check_fixpipe_ub_1byte_align(
    dst_addr: str,
    dst_layout: str,
    dst_dtype: str,
    params: CopyL0C2DstParams,
    dst_desc,
) -> None:
    """Reject unsafe L0C→UB 1-byte static extents / pitches for Fixpipe RoundUp."""
    if dst_addr != "ub" or dst_dtype not in _FIXPIPE_UB_1BYTE_DTYPES:
        return
    dst_m, dst_n = _flat_dim_pair_from_tree(dst_desc.origin_shape)
    dst_s0, dst_s1 = _flat_dim_pair_from_tree(dst_desc.stride)

    # Pitch must cover RoundUp(write-axis, 32) for every unit_flag: Fixpipe
    # always writes that many elements along the axis.
    if dst_layout == "RowMajor":
        if dst_n is not None and dst_s0 is not None:
            need = _round_up_pos_int(dst_n, _FIXPIPE_UB_1BYTE_ALIGN)
            if dst_s0 < need:
                _op_error(
                    "copy",
                    "L0C→UB 1-byte RowMajor destination pitch (stride0) must be "
                    f">= RoundUp(N, {_FIXPIPE_UB_1BYTE_ALIGN})={need} "
                    f"(got N={dst_n}, pitch={dst_s0}, dtype={dst_dtype}); "
                    "pad the UB row stride or align N to 32",
                )
    elif dst_layout == "ColumnMajor":
        if dst_m is not None and dst_s1 is not None:
            need = _round_up_pos_int(dst_m, _FIXPIPE_UB_1BYTE_ALIGN)
            if dst_s1 < need:
                _op_error(
                    "copy",
                    "L0C→UB 1-byte ColumnMajor destination pitch (stride1) must be "
                    f">= RoundUp(M, {_FIXPIPE_UB_1BYTE_ALIGN})={need} "
                    f"(got M={dst_m}, pitch={dst_s1}, dtype={dst_dtype}); "
                    "pad the UB column stride or align M to 32",
                )

    # unit_flag!=0 also needs the write-axis extent itself 32-aligned, else
    # Fixpipe waits for mmad fractals that were never produced.
    if params.unit_flag == 0:
        return
    if dst_layout == "RowMajor":
        if dst_n is not None and dst_n % _FIXPIPE_UB_1BYTE_ALIGN != 0:
            _op_error(
                "copy",
                "L0C→UB 1-byte destination with unit_flag!=0 requires "
                f"RowMajor N % {_FIXPIPE_UB_1BYTE_ALIGN} == 0 "
                f"(got N={dst_n}, dtype={dst_dtype}); align N to 32, or use "
                "unit_flag=0 only when UB pitch >= RoundUp(N, 32) and mmad "
                "uses flag sync",
            )
    elif dst_layout == "ColumnMajor":
        if dst_m is not None and dst_m % _FIXPIPE_UB_1BYTE_ALIGN != 0:
            _op_error(
                "copy",
                "L0C→UB 1-byte destination with unit_flag!=0 requires "
                f"ColumnMajor M % {_FIXPIPE_UB_1BYTE_ALIGN} == 0 "
                f"(got M={dst_m}, dtype={dst_dtype}); align M to 32, or use "
                "unit_flag=0 only when UB pitch >= RoundUp(M, 32) and mmad "
                "uses flag sync",
            )


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


# CopyPad coverage for ``tla.copy(..., params=CopyPadParams(...))`` docs.
# One table covers Vector GM→UB and Cube GM→L1 VL (no cube/vector sub-tables).
# Dtype order matches Constraints wording; keep aligned with
# ``_COPY_PAD_SAFE_DTYPES`` / ``_COPY_PAD_GM_L1_SAFE_DTYPES``.
_COPY_PAD_SAFE_DTYPES_DOC = (
    "f32",
    "f16",
    "bf16",
    "i32",
    "u32",
    "i16",
    "u16",
    "i8",
    "u8",
    "i1",
    "i64",
    "u64",
    "f64",
)
# GM→UB Pad also opens hif8/fp8/fp4x2/i4x2 (8-bit pattern fill); GM→L1 keeps hif8 only.
# Scalar f4e2m1/f4e1m2 stay closed: pad counts are elements on 4-bit tiles.
_COPY_PAD_GM_UB_SAFE_DTYPES_DOC = (
    *_COPY_PAD_SAFE_DTYPES_DOC,
    "hif8",
    "f8e4m3fn",
    "f8e5m2",
    "f4e2m1x2",
    "f4e1m2x2",
    "i4x2",
)
_COPY_PAD_GM_L1_SAFE_DTYPES_DOC = (*_COPY_PAD_SAFE_DTYPES_DOC, "hif8")
_SUPPORTED_COPY_PAD_ROUTES = {
    ("gm", "RowMajor", "ub", "RowMajor"): _COPY_PAD_GM_UB_SAFE_DTYPES_DOC,
    ("gm", "VectorLayout", "ub", "VectorLayout"): _COPY_PAD_GM_UB_SAFE_DTYPES_DOC,
    ("gm", "VectorLayout", "l1", "VectorLayout"): _COPY_PAD_GM_L1_SAFE_DTYPES_DOC,
}


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
        "- In to ub is `split` mode, any quant_mode/relu is disabled, neither pure cast, so only `f32`->`f32` and `i32`->`i32` is enabled",
        "- A quantized copy to l1 supports only the `zN` layout",
        "- L0C→UB 1-byte destinations (`i8`/`u8`/`f8e4m3fn`/`hif8`): Fixpipe "
        "always RoundUp's the write axis (RowMajor `N` / ColumnMajor `M`) to "
        "a 32-element block, independent of `unit_flag`. Static UB pitch must "
        "be `>= RoundUp(extent, 32)` (RowMajor `stride0`, ColumnMajor "
        "`stride1`), or pad overwrites the next row / goes OOB. Prefer "
        "aligning the write-axis extent to 32. With `unit_flag != 0`, mmad "
        "counts fractals by 16, so a static unaligned extent also hangs; the "
        "frontend then requires `N % 32 == 0` (RowMajor) or `M % 32 == 0` "
        "(ColumnMajor). `unit_flag=0` only removes the hang — pitch must "
        "still cover the RoundUp write (and mmad must use flag sync). "
        "Dynamic extents/pitches skip these checks (caller contract). GM "
        "1-byte destinations are not under this rule.\n",
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

    pad_list = [
        "\n",
        "Supported CopyPad routes, `tla.copy(dst, src, params=CopyPadParams(...))`.",
        "Enables AscendC DataCopyPad (`isPad=true`) on GM→UB (vector) or",
        "GM→L1 VectorLayout→VectorLayout (cube). GM→UB opens hif8/fp8/fp4x2/i4x2",
        "(scalar `f4e2m1`/`f4e1m2` Pad stay closed); GM→L1 adds hif8 only",
        "(fp8/fp4 Pad stay closed on GM→L1).",
        "Side counts must not be bool/tla.Bool/i1; a static side count must",
        "satisfy pad bytes ≤ 32B; with static extents, dst innermost ≥",
        "src + left + right. See Constraints for `padding_value` dtype rules",
        "(including 8-bit integer bit patterns for hif8/fp8/fp4x2/i4x2).\n",
        "|`src_addr`|`src_layout`|`dst_addr`|`dst_layout`|`dtypes`|",
        "|----------|------------|----------|------------|--------|",
    ]
    for (
        src_addr,
        src_layout,
        dst_addr,
        dst_layout,
    ), dtypes in _SUPPORTED_COPY_PAD_ROUTES.items():
        pad_list.append(
            f"|`{src_addr}`|`{src_layout}`|`{dst_addr}`|`{dst_layout}`|{_stringify_types(dtypes)}|"
        )

    fn.__doc__ += "\n".join(
        [*cube_list, *cube_scale_list, *cube_fixpipe_list, *vec_list, *pad_list]
    )


# Width, mantissa bits, exponent bias and largest finite positive encoding.
# Packed FP4/FP8 constants are converted at trace time into a repeated
# uint16 pattern, including four identical nibbles for packed fp4.
_FILL_FLOAT_FORMATS = {
    "f4e2m1": (4, 1, 1, 0x7),
    "f4e1m2": (4, 2, 1, 0x7),
    "f8e4m3fn": (8, 3, 7, 0x7E),
    "f8e5m2": (8, 2, 15, 0x7B),
}
_FILL_INTEGER_TYPES = {
    "i16": (16, True),
    "u16": (16, False),
    "i32": (32, True),
    "u32": (32, False),
}
_FILL_FLOAT_TYPES = {
    "f16": (Float16, 0x7C00),
    "bf16": (BFloat16, 0x7F80),
    "f32": (Float32, 0x7F800000),
}
_FILL_SUPPORTED_ELEMENT_TYPES = (
    frozenset(_FILL_FLOAT_FORMATS)
    | {"f8e8m0"}
    | _FILL_INTEGER_TYPES.keys()
    | _FILL_FLOAT_TYPES.keys()
)


def _fill_value_bits(op_name: str, element_type: str, value: Any) -> int:
    """Encode a trace-time scalar in the destination type, carried by signed i32."""
    if type(value) not in (int, float):
        _op_error(
            op_name,
            f"fill value must be a trace-time int or float, got {type(value).__name__}",
        )
    if element_type in _FILL_INTEGER_TYPES:
        width, signed = _FILL_INTEGER_TYPES[element_type]
        if not isinstance(value, int):
            _op_error(op_name, f"fill of {element_type} requires an int")
        lower = -(1 << (width - 1)) if signed else 0
        upper = (1 << (width - int(signed))) - 1
        if not lower <= value <= upper:
            _op_error(
                op_name, f"fill value must be in [{lower}, {upper}] for {element_type}"
            )
        bits = value & ((1 << width) - 1)
    elif element_type in _FILL_FLOAT_TYPES:
        dtype, exponent_mask = _FILL_FLOAT_TYPES[element_type]
        try:
            scalar = float(value)
            with np.errstate(over="ignore", invalid="ignore"):
                bits = dtype._host_bits(scalar)
        except (OverflowError, ValueError):
            _op_error(op_name, f"fill value cannot be encoded as {element_type}")
        if math.isfinite(scalar) and (bits & exponent_mask) == exponent_mask:
            _op_error(op_name, f"fill value overflows {element_type}")
    elif element_type == "f8e8m0":
        # E8M0 codes 0..254 are 2**(code-127). Check integers before any
        # float conversion can round an inexact scale to a power of two.
        if isinstance(value, int):
            exponent = value.bit_length()
            is_power_of_two = value > 0 and (value & (value - 1)) == 0
        else:
            if not math.isfinite(value):
                _op_error(op_name, "fill value must be finite")
            fraction, exponent = math.frexp(value)
            is_power_of_two = fraction == 0.5
        if not is_power_of_two or not -126 <= exponent <= 128:
            _op_error(
                op_name,
                "E8M0 fill requires a positive power of two in [2**-127, 2**127]",
            )
        return (exponent + 126) * 0x0101
    elif element_type in _FILL_FLOAT_FORMATS:
        try:
            number = _builtins.float(value)
        except OverflowError:
            _op_error(
                op_name, f"fill value is outside the finite range of {element_type}"
            )
        if not math.isfinite(number):
            _op_error(op_name, "fill value must be finite")

        width, mantissa_bits, bias, largest_code = _FILL_FLOAT_FORMATS[element_type]
        mantissa_unit = 1 << mantissa_bits
        largest_exp, largest_mantissa = divmod(largest_code, mantissa_unit)
        largest_value = math.ldexp(
            1 + largest_mantissa / mantissa_unit, largest_exp - bias
        )
        magnitude = _builtins.abs(number)
        if magnitude > largest_value:
            _op_error(
                op_name,
                f"fill value {value!r} is outside the finite range of {element_type}",
            )
        fraction, exponent = math.frexp(magnitude)
        encoded_exp = exponent - 1 + bias
        if magnitude == 0:
            code = 0
        else:
            if encoded_exp <= 0:
                significand = math.ldexp(magnitude, bias - 1 + mantissa_bits)
                code = 0
            else:
                significand = math.ldexp(fraction, mantissa_bits + 1)
                code = (encoded_exp - 1) << mantissa_bits
            # Binary scaling is exact; fractional bits would require rounding.
            if not significand.is_integer():
                _op_error(
                    op_name,
                    f"fill value {value!r} is not exactly representable as {element_type}",
                )
            code += _builtins.int(significand)
        if math.copysign(1.0, number) < 0:
            code |= 1 << (width - 1)
        return _builtins.sum(code << shift for shift in _builtins.range(0, 16, width))
    else:
        _op_error(op_name, f"unsupported fill element type {element_type!r}")
    return bits if bits < (1 << 31) else bits - (1 << 32)


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
    if len(dest_desc.origin_shape) != 2:
        _op_error(op_name, "fill currently requires a logical rank-2 tile")
    if dest_desc.addrspace.lower() != "l1":
        _op_error(
            op_name,
            "invalid argument 'tensor' (position 0): expected addrspace l1, "
            f"got {dest_desc.addrspace}",
        )
    layout_tag = dest_desc.layout_tag
    if layout_tag not in ("zN", "nZ", "zZMxScale", "nNMxScale"):
        _op_error(
            op_name,
            "invalid argument 'tensor' (position 0): expected layout tag zN or nZ "
            "for operands, zZMxScale or nNMxScale for E8M0, "
            f"got {layout_tag!r}",
        )
    element_type = str(dest_desc.element_type).strip().lower()
    if element_type not in _FILL_SUPPORTED_ELEMENT_TYPES:
        _op_error(
            op_name,
            f"unsupported element type {element_type!r}: fill supports "
            "i16/u16/f16/bf16/i32/u32/f32, packed fp4/fp8 operand formats "
            "and f8e8m0 scale tiles",
        )
    scale_layout = layout_tag in ("zZMxScale", "nNMxScale")
    if scale_layout != (element_type == "f8e8m0"):
        _op_error(
            op_name,
            "E8M0 fill requires zZMxScale/nNMxScale; operand fill requires zN/nZ",
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


def _ubub_static_geometry(src_desc, dst_desc, src_dtype):
    """Static block geometry of a UB->UB copy, or ``None`` when it is not
    statically known (dynamic or non-2D shapes, or a src/dst shape mismatch).

    Lengths are in bits rather than bytes: a packed fp4 element is half a byte,
    and the block move counts whole 32-byte blocks of elements, so rounding to
    bytes first would hide exactly the misalignment this geometry exists to
    catch.
    """
    try:
        # The extent is the origin shape: it is the tile's valid region, while
        # ``shape`` is the packing it is addressed through, which a tail tile
        # pads out past that region. ``copyUBToUB`` sizes the move the same way.
        s_shape, d_shape = src_desc.origin_shape, dst_desc.origin_shape
        s_stride, d_stride = src_desc.stride, dst_desc.stride
        s_coord, d_coord = src_desc.coord, dst_desc.coord
        if len(s_shape) != 2 or len(d_shape) != 2:
            return None
        rows, cols = int(s_shape[-2]), int(s_shape[-1])
        s_pitch, d_pitch = int(s_stride[-2]), int(d_stride[-2])
        if (rows, cols) != (int(d_shape[-2]), int(d_shape[-1])):
            return None
        # The coordinate is the tile's origin inside its parent, in elements.
        s_base = int(s_coord[-2]) * s_pitch + int(s_coord[-1])
        d_base = int(d_coord[-2]) * d_pitch + int(d_coord[-1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    # The move's block is ``BYTE_PER_BLK`` (32) bytes of elements. A width it
    # cannot split into whole blocks has no copyable geometry at all, so this
    # fails closed instead of reporting zero-byte lengths that pass every check.
    bits = dtype_size_bits(src_dtype)
    block_bits = 256
    if bits <= 0 or block_bits % bits != 0:
        _op_error(
            "copy",
            f"unsupported UB->UB copy: {src_dtype!r} has no element width the "
            "MOV.UB.TO.UB block move can split into 32-byte blocks",
        )
    row_bits = cols * bits
    return {
        "rows": rows,
        "cols": cols,
        "bits": bits,
        "block_bits": block_bits,
        "row_bits": row_bits,
        "row_rounded_bits": -(-row_bits // block_bits) * block_bits,
        "src_base_bits": s_base * bits,
        "dst_base_bits": d_base * bits,
        "src_pitch_bits": s_pitch * bits,
        "dst_pitch_bits": d_pitch * bits,
    }


def _ubub_require_metaop_representable(geom) -> None:
    """Reject the UB->UB geometries the ``MOV.UB.TO.UB`` block move cannot copy.

    The meta-op rounds the row length *up* to a whole number of 32-byte blocks
    (see ``copyUBToUB`` in ``csrc/mlir/bc/Vector/dma.cpp``), so it moves the
    row's alignment fill along with the row. Two things have to hold for that to
    be a copy rather than a corruption, and neither is checkable on the device:

    - Both origins must sit on a 32-byte boundary, because the block move
      addresses whole blocks. A sub-block view can break this -- a 12-element
      f32 tile at tile index 1 starts at byte 48.
    - When there is more than one row, every row pitch must leave room for the
      rounded row, and the slack must itself be a whole number of blocks so the
      inter-row gap is exact. Otherwise the rounded row runs over the head of
      the next one.

    A single-row copy needs no pitch at all: the rounded tail lands in the
    buffer's own 32-byte alignment fill.

    Widths are compared in bits so packed fp4 is held to the same rule: 96 fp4
    elements are 48 bytes, and the move would step rows by the rounded 64.
    """
    if geom is None:
        return
    block_bits = geom["block_bits"]
    row_bits, row_ru_bits = geom["row_bits"], geom["row_rounded_bits"]
    for base_bits, side in (
        (geom["src_base_bits"], "source"),
        (geom["dst_base_bits"], "destination"),
    ):
        if base_bits % block_bits != 0:
            # Keep whole-byte offsets spelled in bytes, half-byte ones in bits.
            offset = (
                f"byte {base_bits // 8}" if base_bits % 8 == 0 else f"bit {base_bits}"
            )
            _op_error(
                "copy",
                f"unsupported UB->UB copy: the {side} tile starts at {offset} of "
                "its parent, which is not a multiple of 32. The MOV.UB.TO.UB "
                "block move addresses whole 32-byte blocks, so it cannot start "
                "there. Align the tile's offset (or its width) to 32 bytes.",
            )
    if geom["rows"] <= 1:
        return
    for pitch_bits, side in (
        (geom["src_pitch_bits"], "source"),
        (geom["dst_pitch_bits"], "destination"),
    ):
        # Same bytes-for-whole-bytes spelling as the offset above.
        pitch = (
            f"{pitch_bits // 8} bytes" if pitch_bits % 8 == 0 else f"{pitch_bits} bits"
        )
        if pitch_bits < row_ru_bits:
            _op_error(
                "copy",
                f"unsupported UB->UB copy: rows of {row_bits} bits are rounded "
                f"up to {row_ru_bits} for the MOV.UB.TO.UB block move, which the "
                f"{side} row pitch of {pitch} cannot carry -- the "
                "rounded row would run into the next one. Pad the row pitch to a "
                f"multiple of 32 bytes that is at least {row_ru_bits // 8} bytes.",
            )
        if pitch_bits % block_bits != 0:
            _op_error(
                "copy",
                f"unsupported UB->UB copy: the {side} row pitch is {pitch}, "
                "which is not a whole number of 32-byte blocks. The block "
                "move steps rows in whole blocks, so it would put the next row "
                "at the rounded-up pitch instead of the layout's. Pad the row "
                "pitch to a multiple of 32 bytes.",
            )


def _validate_ubub_transpose_shape(src_desc, dst_desc, src_dtype) -> None:
    """Reject a tile VNCHWCONV cannot transpose.

    The meta-op transposes whole blocks, so the tile has to be a whole number of
    them: 16 source rows (32 for b8, which needs four calls to cover its tile) by
    C0 columns, C0 being the elements in a 32-byte DataBlock. A tile that is not
    a multiple of that would have the meta-op read and write past its own edges,
    so it is rejected here rather than silently corrupting neighbouring data.

    The extent checked is `origin_shape`, the tile's valid extent, because that
    is what the meta-op transposes -- `shape` is the packing it is addressed
    through, which a tail tile pads out further.
    """
    esz = dtype_size_bytes(src_dtype)
    if esz not in (1, 2, 4):
        return
    c0 = 32 // esz
    tile_rows = 32 if esz == 1 else 16
    try:
        shape = src_desc.origin_shape
        if len(shape) != 2:
            return
        rows, cols = int(shape[-2]), int(shape[-1])
    except (TypeError, ValueError, IndexError, KeyError):
        return  # dynamic shape: nothing to check at compile time
    if rows % tile_rows or cols % c0:
        _op_error(
            "copy",
            f"transposing UB->UB copy of a {rows}x{cols} {src_dtype} tile: the "
            f"shape must be a multiple of {tile_rows}x{c0} for a {esz}-byte "
            f"element",
        )


# ExtParams-safe scalar GM→UB pad dtypes (must stay aligned with Vector/dma.cpp).
# 8B: static zero padding_value enforced in CopyPadParams._validate; runtime
# padding_value is not value-gated (BC always zero-fills 8B pads).
_COPY_PAD_SAFE_DTYPES = frozenset(
    {
        "f32",
        "f16",
        "bf16",
        "i32",
        "u32",
        "i16",
        "u16",
        "i8",
        "u8",
        "i1",
        "i64",
        "u64",
        "f64",
    }
)
# GM→UB Pad: base scalars ∪ hif8/fp8/fp4x2/i4x2 (8-bit fill pattern).
# Scalar f4e2m1/f4e1m2 excluded (pad counts are elements on 4-bit tiles).
# i4x2 stages as int8_t (like fp4x2); pattern fill matches 1B carrier width.
_COPY_PAD_LOWP_DTYPES = frozenset(
    {
        "hif8",
        "f8e4m3fn",
        "f8e5m2",
        "f4e2m1x2",
        "f4e1m2x2",
        "i4x2",
    }
)
_COPY_PAD_GM_UB_SAFE_DTYPES = _COPY_PAD_SAFE_DTYPES | _COPY_PAD_LOWP_DTYPES
# GM→L1 VL Pad reuses the scalar base and additionally opens hif8 only
# (Cube BC + Tile uint8 ExtParams). Do not merge fp8/fp4 into GM→L1.
_COPY_PAD_GM_L1_SAFE_DTYPES = _COPY_PAD_SAFE_DTYPES | {"hif8"}
_COPY_PAD_FLOAT_DTYPES = frozenset({"f32", "f16", "bf16", "f64"})
_COPY_PAD_INT_DTYPES = frozenset({"i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"})


_COPY_PAD_FLOAT_HOST_BITS = {
    "f16": (Float16, 0x7C00),
    "bf16": (BFloat16, 0x7F80),
    "f32": (Float32, 0x7F800000),
}


def _normalize_copy_pad_dtype_token(dtype: str) -> str:
    """Map ``bool`` → ``i1``; otherwise lower-case dtype token."""
    token = dtype.lower()
    return "i1" if token == "bool" else token


def _padding_value_float_scalar(value: Any, dtype: str) -> float:
    """Host float for encoding ``padding_value`` on a float tile dtype."""
    if isinstance(value, Numeric):
        host = value.value
        if isinstance(host, bool):
            raise TypeError(
                f"padding_value for dtype {dtype} must match element type "
                f"(got bool host on {type(value).__name__})"
            )
        if not isinstance(host, (int, float)):
            raise TypeError(
                f"padding_value Numeric host must be int/float "
                f"(got {type(host).__name__})"
            )
        return float(host)
    if isinstance(value, bool):
        raise TypeError(
            f"padding_value for dtype {dtype} must be float or matching "
            f"Numeric (got bool)"
        )
    if isinstance(value, float):
        return value
    raise TypeError(
        f"padding_value for dtype {dtype} must be float or matching Numeric "
        f"(got {type(value).__name__})"
    )


def _encode_float_padding_bits(scalar: float, dtype: str, mask: int) -> int:
    """Pack a Python float into ``paddingValueBits`` for ``dtype``."""
    host = _COPY_PAD_FLOAT_HOST_BITS.get(dtype)
    try:
        if host is not None:
            dtype_cls, exponent_mask = host
            with np.errstate(over="ignore", invalid="ignore"):
                bits = int(dtype_cls._host_bits(scalar)) & mask
            if math.isfinite(scalar) and (bits & exponent_mask) == exponent_mask:
                raise ValueError(
                    f"padding_value {scalar!r} overflows finite range of {dtype}"
                )
            return bits
        if dtype == "f64":
            return struct.unpack("<Q", struct.pack("<d", scalar))[0]
    except (OverflowError, struct.error, ValueError) as exc:
        raise ValueError(
            f"padding_value {scalar!r} cannot be encoded for dtype {dtype}: {exc}"
        ) from exc
    raise TypeError(f"cannot encode padding_value for dtype {dtype}")


def _padding_value_to_i64_bits(value: Any, dtype: str, elem_nbytes: int) -> int:
    """Pack ``CopyPadParams.padding_value`` into ABI ``paddingValueBits``.

    Static values must **match** the tile element dtype:
    - ``i1``: Python ``bool`` / ``tla.Bool`` only (encoded as 0/1 bits).
    - integer tiles: bare ``int`` or Numeric whose ``dtype`` equals the tile.
    - float tiles: bare ``float`` or Numeric whose ``dtype`` equals the tile.
    Cross-width / cross-family Numerics and bare ``int`` on float tiles are
    rejected.
    """
    dtype = _normalize_copy_pad_dtype_token(dtype)
    if elem_nbytes <= 0 or elem_nbytes > 8:
        raise ValueError(f"unsupported pad element width {elem_nbytes}B for {dtype}")
    bit_width = elem_nbytes * 8
    mask = (1 << bit_width) - 1

    # --- i1 / bool tile: fill must be bool-typed (bare int 0 = default zero) ---
    if dtype == "i1":
        if isinstance(value, bool):
            return 1 if value else 0
        # Dataclass default ``padding_value=0``: accept as False / zero-fill only.
        if isinstance(value, int) and not isinstance(value, bool) and value == 0:
            return 0
        if isinstance(value, Numeric):
            nd = _normalize_copy_pad_dtype_token(type(value).dtype)
            if nd != "i1":
                raise TypeError(
                    f"padding_value dtype {nd} does not match element dtype i1"
                )
            host = value.value
            if isinstance(host, bool):
                return 1 if host else 0
            if isinstance(host, int) and not isinstance(host, bool):
                return 1 if host != 0 else 0
            raise TypeError(
                f"padding_value for i1 must be bool/tla.Bool "
                f"(got host {type(host).__name__})"
            )
        raise TypeError(
            f"padding_value for i1 must be bool/tla.Bool "
            f"(got {type(value).__name__}; non-zero bare int is rejected)"
        )

    # --- Typed Numeric: dtype token must equal the tile ---
    if isinstance(value, Numeric):
        nd = _normalize_copy_pad_dtype_token(type(value).dtype)
        if nd != dtype:
            raise TypeError(
                f"padding_value dtype {nd} does not match element dtype {dtype}"
            )
        if dtype in _COPY_PAD_INT_DTYPES:
            host = value.value
            if isinstance(host, bool):
                raise TypeError(
                    f"padding_value for integer dtype {dtype} must be an int (got bool)"
                )
            if isinstance(host, float):
                if not host.is_integer():
                    raise TypeError(
                        f"padding_value for integer dtype {dtype} must be an int "
                        f"(got float {host!r})"
                    )
                host = int(host)
            if not isinstance(host, int):
                raise TypeError(
                    f"padding_value for integer dtype {dtype} must be an int "
                    f"(got {type(host).__name__})"
                )
            return int(host) & mask
        if dtype in _COPY_PAD_FLOAT_DTYPES:
            try:
                scalar = _padding_value_float_scalar(value, dtype)
            except TypeError as exc:
                raise TypeError(str(exc)) from exc
            return _encode_float_padding_bits(scalar, dtype, mask)
        if dtype in _COPY_PAD_LOWP_DTYPES:
            # No host float encode; pad fill is a raw 8-bit pattern (hif8/fp8/fp4).
            host = value.value
            if isinstance(host, bool):
                raise TypeError(f"padding_value for {dtype} must not be bool")
            if isinstance(host, int):
                return int(host) & mask
            raise TypeError(
                f"padding_value for {dtype} must be an 8-bit integer bit pattern "
                f"(bare int or matching Numeric with int host; got host "
                f"{type(host).__name__})"
            )
        raise TypeError(f"cannot encode padding_value for dtype {dtype}")

    # --- Untyped Python literals ---
    if isinstance(value, bool):
        raise TypeError(
            f"padding_value for dtype {dtype} must not be bool "
            f"(bool fill is only valid for i1 tiles)"
        )
    if isinstance(value, int):
        # Dataclass default is ``0``. Accept bare int 0 as universal zero-fill
        # (0 bits / False / +0.0). Non-zero bare int: integer tiles or lowp bits.
        if value == 0:
            return 0
        if dtype in _COPY_PAD_LOWP_DTYPES:
            return int(value) & mask
        if dtype not in _COPY_PAD_INT_DTYPES:
            raise TypeError(
                f"padding_value for dtype {dtype} must be float or a Numeric "
                f"whose dtype matches {dtype} (got bare int {value!r})"
            )
        return int(value) & mask
    if isinstance(value, float):
        if dtype not in _COPY_PAD_FLOAT_DTYPES:
            raise TypeError(
                f"padding_value for integer dtype {dtype} must be an int "
                f"(got float {value!r})"
            )
        return _encode_float_padding_bits(value, dtype, mask)
    raise TypeError(
        f"padding_value must be bool/int/float/Numeric (got {type(value).__name__})"
    )


def _padding_value_as_i64_ssa(
    value: Any,
    dtype: str,
    elem_nbytes: int,
    *,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    """Encode ``padding_value`` as the BC ``paddingValueBits`` ``i64`` SSA.

    Compile-time Python / Numeric constants use ``_padding_value_to_i64_bits``
    with strict dtype matching. For ``i1`` tiles, static ``bool`` / ``tla.Bool``
    encode as 0/1. Runtime non-i1 values must already be an integer/index bit
    pattern (``_as_i64_copy_pad_operand``); runtime i1 SSA is rejected (use a
    static bool fill).
    """
    dtype = _normalize_copy_pad_dtype_token(dtype)
    # i1: bool/tla.Bool, or bare int 0 (dataclass default zero-fill).
    if dtype == "i1":
        if (
            _is_bool_like_pad_operand(value)
            or (isinstance(value, int) and not isinstance(value, bool) and value == 0)
            or (
                isinstance(value, Numeric)
                and _normalize_copy_pad_dtype_token(type(value).dtype) == "i1"
                and isinstance(value.value, (bool, int))
            )
        ):
            # Runtime i1 SSA cannot be widened via the integer pad helper.
            resolved = _resolve_bound_value(value)
            if isinstance(resolved, mlir_ir.Value):
                raise TypeError(
                    "CopyPadParams.padding_value for i1 tiles must be a static "
                    "bool/tla.Bool (runtime i1 SSA is not supported)"
                )
            bits = _padding_value_to_i64_bits(value, dtype, elem_nbytes)
            return _const_i64(bits, loc=loc)
        raise TypeError(
            "CopyPadParams.padding_value for i1 tiles must be bool/tla.Bool "
            f"(got {type(value).__name__}; non-zero bare int is rejected)"
        )
    if _is_bool_like_pad_operand(value):
        raise TypeError(
            "CopyPadParams.padding_value must not be bool/tla.Bool/i1 unless "
            "the element dtype is i1"
        )
    # Prefer static encoding so dtype checks see Float32 vs Int32 wrappers.
    if isinstance(value, (int, float)) or (
        isinstance(value, Numeric) and isinstance(value.value, (bool, int, float))
    ):
        bits = _padding_value_to_i64_bits(value, dtype, elem_nbytes)
        return _const_i64(bits, loc=loc)
    # Runtime SSA / dynamic Numeric: treat as bit pattern (non-i1 only).
    return _as_i64_copy_pad_operand(value, what="padding_value", loc=loc)


def _copy_pad_innermost_extent(desc) -> int | None:
    """Return static innermost origin extent, or None if dynamic/unavailable."""
    try:
        origin = desc.origin_shape
        if not origin:
            return None
        return _const_int_value(origin[-1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _copy_pad_static_rows(desc) -> int | None:
    """Return static row count (origin_shape[0]) for Rank-2 tiles."""
    try:
        origin = desc.origin_shape
        if not origin or len(origin) < 2:
            return None
        return _const_int_value(origin[0])
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _copy_pad_static_row_stride(desc) -> int | None:
    """Return static RowMajor row stride (stride[0]), or None if dynamic."""
    try:
        stride = desc.stride
        if not stride:
            return None
        return _const_int_value(stride[0])
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _copy_pad_align_up(value: int, align: int) -> int:
    if align <= 0:
        return value
    return ((value + align - 1) // align) * align


@dsl_user_op
def copy(
    dst: Tensor,
    src: Tensor,
    params: CopyParams | None = None,
    *,
    scale: Tensor | None = None,
    cache_policy: CachePolicy | None = None,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Directory: Data Movement
    Description:
        Copy a tile from `src` to `dst`. Supported routes and dtypes are listed
        after Constraints.

        Parameters:
        - `dst` (`Tensor`): Destination tile. Required.
        - `src` (`Tensor`): Source tile. Required.
        - `params` (`CopyParams | None`): Path-specific params:
          `CopyL0C2DstParams`, `CopyGmToUbParams`, `CopyUbToGmParams`, or `CopyPadParams`.
          Default `None` (exact-byte copy / no pad / default fixpipe).
          - `CopyL0C2DstParams`: L0C → GM/L1/UB fixpipe options
            (`unit_flag`, `relu_enable`, `quant_mode`, `quant_scale`,
            `quant_offset`, `quant_tensor`, `l0c2ub_mode`, `atomic_mode`).
            See the Cube fixpipe table for supported combinations.
          - `CopyGmToUbParams`: GM → UB load options. Set `compact=True`
            to pack the destination rows.
          - `CopyUbToGmParams`: UB → GM writeback options. Fields:
            `compact` (`bool`, default `False`) and
            `atomic_mode` (`AtomicMode`, default `NONE`).
          - `CopyPadParams`: enables DataCopyPad (`isPad=true`) on GM→UB or
            GM→L1 VectorLayout→VectorLayout (cube). Supported routes/dtypes
            are listed in the CopyPad table below. Optional fields
            (may be set together; each defaults to `0`):
            - `left_padding`: elements padded on the left of the innermost
              axis (compile-time int or runtime integer/index SSA). Must
              not be bool/tla.Bool/i1.
            - `right_padding`: elements padded on the right of the innermost
              axis (same type rules as `left_padding`).
            - `padding_value`: fill for padded elements. Default `0` means
              zero-fill. Must match the source element dtype (see
              Constraints); for `hif8`/`fp8`/`fp4x2`/`i4x2` use an 8-bit
              integer bit pattern (bare `int` or matching Numeric with int
              host), not a float.
        - `scale` (`Tensor | None`): L1 OCP `e8m0` shared-exponent tile for
          L1→L0A/L0B MX load. Stored as `i8`/`u8`; destination tag is
          `tla.arch.zZMxScale` (L0A) or `tla.arch.nNMxScale` (L0B).
          Default `None`.
        - `cache_policy` (`CachePolicy | None`): Optional L2 cache hint for the
          GM endpoint of this copy (`DISABLE` / `NORMAL`). On `GM→*` routes it
          tags the GM src; on `*→GM` routes it tags the GM dst. There is no
          GM→GM copy route. Default `None` (omit the hint; BC uses NORMAL).
          On non-GM routes any explicit value (including `NORMAL`) is rejected;
          omit the argument instead.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()` or `tla.vector()`.
        - Copy size follows each tile's `origin_shape`.
        - UB-to-GM and L0C-to-GM copies accept
          `params.atomic_mode`. Use `AtomicMode.ADD`, `MAX`, or `MIN` to combine
          this write with the old GM value. The copy disables atomic writes
          after an explicitly requested operation. The default, `NONE`, uses
          the current CTRL atomic state. Kernel entry disables atomic writes.
          When enabling atomic writes with `tla.arch.set_ctrl`, set
          `atomic_format=AtomicFormat.DISABLED` before a copy that must overwrite GM.
          The GM element type must be i8, i16, i32, f16, bf16, or f32 for atomic
          writes. For L0C, conversion, quantization, and ReLU take effect before
          the atomic operation.
          For UB-to-GM, supply `CopyUbToGmParams` or omit `params`. Within
          that parameter object, `atomic_mode` must be an `AtomicMode` member
          and defaults to `AtomicMode.NONE`.
        - When `params` is `CopyPadParams`: see the CopyPad coverage table
          below. Side counts must not be bool/tla.Bool/i1; a static side
          count must satisfy pad bytes ≤ 32B; with static extents, dst
          innermost ≥ src + left + right. GM→UB opens hif8/fp8/fp4x2/i4x2
          (scalar `f4e2m1`/`f4e1m2` Pad stay closed); GM→L1 adds hif8 only
          (fp8/fp4 Pad stay closed on GM→L1).
        - `CopyPadParams.padding_value` must match the source element dtype
          (i1: bool/tla.Bool; integer: bare int or same-dtype Numeric;
          float f16/bf16/f32/f64: bare float or matching float Numeric;
          hif8/fp8/fp4x2/i4x2: 8-bit integer bit pattern via bare int or
          matching Numeric with int host — no float encode; bare int 0 is
          the dataclass default zero-fill). Runtime non-i1 is an i64 bit
          pattern; runtime i1 SSA is unsupported. For 8B (`i64`/`u64`/
          `f64`), only a static zero fill is guaranteed; runtime is not
          value-checked and the BC always zero-fills.
        - On Ascend 950, two logical rank-3 views `(B, M, N)` select batched
          GM-to-L1 layout conversion. FP8 supports RowMajor→zN and
          ColumnMajor→nZ; E8M0 supports the four GM MX-scale routes below.
          Batch count is `origin_shape[0]`; source and destination matrix
          strides are each layout's leading stride, in elements. Inner strides
          retain their 2-D meaning. Both logical extents must match, including
          runtime values. Zero batches or zero matrix extents perform no DMA.
          Other rank-3 layout/dtype combinations fail at compilation instead
          of falling back to a 2-D copy.
        - Batched copies require `params=None` and `scale=None`, a batch count
          in 0..65535, and positive destination batch strides aligned to 32
          bytes and at most 65535 32-byte blocks. E8M0 source matrix strides
          must preserve exponent pairs. Runtime geometry must satisfy the same
          constraints. The caller supplies enough backing storage and disjoint
          destination matrices; interleaved NZ layouts are permitted.
        - On Ascend 950, `CopyGmToUbParams(compact=True)` loads packed UB
          rows and `CopyUbToGmParams(compact=True)` stores packed UB rows.
          Both require matching RowMajor tile extents and element types.
          Use the element types listed for the corresponding RowMajor route.
          Both column strides must be 1. The UB row stride must equal the
          valid row width. The GM row stride can be larger. For runtime geometry,
          the caller ensures the GM row stride covers the valid width and the
          extents fit one DMA request. For a compact load, use a 32-byte-aligned
          UB destination address.
        - Packed fp4 is staged as bytes on the routes that convert the
          descriptor before copying -- GM->UB, UB->GM and UB->L1 -- so each
          tile's innermost `origin_shape`, its innermost origin coordinate and
          the stride the conversion halves must all be even: two fp4 elements
          share a byte, and an odd one of the three lands half a byte off. The
          halved stride is the row stride of a `RowMajor` tile and the pitch
          between C0 columns of a `zN` tile, so a `zN` destination on UB->L1
          carries the requirement as well. A value known only at run time
          cannot be shown to be even and is refused as well. The cube's GM->L1
          fp4 route keeps its own denomination and is unconstrained, and
          `tla.experimental.mx_to` still accepts an odd extent -- such a tile
          simply cannot be staged through a byte descriptor.
        - `scale` is only valid on L1→L0A/L0B; the loaded tile must then be
          consumed by `tla.mmad_mx`.
        - For MX scale destinations (`zZMxScale` / `nNMxScale`), the source
          tile must span full L1 rows; stack scale blocks along rows.
          Checked only when row pitch is statically known.
        - UB→UB counts in whole 32-byte blocks; rows round up. Rejected if the
          origin is not 32B-aligned, or multi-row pitch cannot hold the rounded
          row. Destination tagged `ColumnMajor` is a transpose: tile shape must
          be whole 16×C0 blocks, or 32×32 for 1-byte types.
        - `GM`→`L1` / `L1`→`biasbuf` contiguous runs use rank-1 `VectorLayout`;
          multi-row bias tiles are not representable — slice one row.
        - PER_CHANNEL dequant parameters (`CopyL0C2DstParams.quant_tensor`)
          are a rank-1 `VectorLayout` tile in the `L1` address space, element
          type `u64`, unit stride. One element carries the dequant parameter
          shared by every element of one output channel, so the run must hold
          exactly one element per channel of the destination tile, ``originShape[1]``
          after `tile_view` cropping. When both lengths are static, a mismatch
          fails compilation.
        - L0C→UB 1-byte destinations (`i8`/`u8`/`f8e4m3fn`/`hif8`): Fixpipe
          always RoundUp's the write axis (RowMajor `N` / ColumnMajor `M`)
          to a 32-element block, independent of `unit_flag`. Static UB pitch
          must be `>= RoundUp(extent, 32)` (RowMajor `stride0`, ColumnMajor
          `stride1`), or pad overwrites the next row / goes OOB. Prefer
          aligning the write-axis extent to 32. With `unit_flag != 0`, mmad
          counts fractals by 16, so a static unaligned extent also hangs;
          the frontend then requires `N % 32 == 0` (RowMajor) or
          `M % 32 == 0` (ColumnMajor). `unit_flag=0` only removes the hang —
          pitch must still cover the RoundUp write (and mmad must use flag
          sync). Dynamic extents/pitches skip these checks (caller
          contract). GM 1-byte destinations are not under this rule.
        - **Prepacked `GM` tiles:** a buffer already packed in fractal order is
          tagged `zN` (or `nZ` for the transposed packing) and copied to the
          matching `L1` tile with no re-layout. Only the aligned `zN` / `nZ` are
          `GM` source tags — the tight M-axis is `zNUnAlign` on the `UB`→`L1`
          route, and the `GM` `zNUnAlign` route is the NC1HWC0 feature map that
          feeds `tla.experimental.img2col`. Size both ends for the padded tile:
          `ceil(M/16) * 16` rows for `zN`, and `ceil(M/C0) * C0` rows by
          `ceil(N/16) * 16` columns for `nZ`, since the block move walks whole
          fractals and the last row group travels in full.
        - Whole-tile DMA uses `tla.copy`. For register-level UB unaligned
          access, use `tensor.load` / `tensor.store` with `UnalignLoadParams` /
          `UnalignStoreParams`.
        - **Shapes not aligned to 32 bytes:** C0 is 32 bytes. For ND GM↔UB,
          prefer a leading dimension whose byte size is a multiple of 32.
          Otherwise keep `origin_shape` as the real size and enlarge the
          physical layout, or use unaligned load/store.

        Example:
        ```python
        # --- 1) Aligned GM ↔ UB (RowMajor, 32B-friendly) ---
        with tla.vector():
            tla.copy(dst=x_ub, src=x_gm)
            tla.copy(
                dst=x_ub,
                src=x_gm,
                cache_policy=tla.params.CachePolicy.DISABLE,
            )  # optional L2 hint on GM src
            tla.copy(dst=y_gm, src=y_ub)

        # --- 2) ND → zN: GM RowMajor → L1 zN ---
        l1_a = tla.make_tensor_like(l1_ptr, gm_a, layoutTag=tla.arch.zN)
        with tla.cube():
            tla.copy(dst=l1_a, src=gm_a)

        # --- 3) M not a multiple of 16: zNUnAlign ---
        # Before: rows may be runtime and not a multiple of 16.
        l1_unalign = tla.make_tensor_like(l1_ptr, gm_tile, layoutTag=tla.arch.zNUnAlign)
        with tla.cube():
            tla.copy(dst=l1_unalign, src=gm_tile)

        # Batched FP8: both tensor layouts carry the batch geometry.
        # src_step and dst_step are independent element strides.
        src3 = tla.make_tensor(src_ptr, tla.make_layout(
            tla.make_shape(batch_count, valid_rows, 128),
            tla.make_stride(src_step, 128, 1)))
        dst3 = tla.make_tensor(dst_ptr, tla.make_layout(
            tla.make_shape(batch_count, (16, 8), (32, 4)),
            tla.make_stride(dst_step, (32, 512), (1, 4096)),
            origin_shape=tla.make_shape(batch_count, valid_rows, 128),
            layoutTag=tla.arch.zN))
        with tla.cube():
            tla.copy(dst3, src3)

        # --- 4) Prepacked GM zN: a plain block move, no re-layout ---
        # The host buffer is already zN-packed and padded to ceil(M/16) * 16 rows.
        # Both ends carry the same tree, so the copy reorders nothing.
        gm_zn = tla.make_tensor(
            gm_ptr, tla.make_layout_with_tag((m_pad16, k), dtype_a, tla.arch.zN)
        )
        l1_zn = tla.make_tensor_like(l1_ptr, gm_zn, tla.arch.zN)
        with tla.cube():
            tla.copy(dst=l1_zn, src=gm_zn)

        # --- 5) Prepacked GM nZ: the transposed packing, same idea ---
        # nZ pads both axes -- ceil(M/C0) * C0 rows by ceil(N/16) * 16 columns --
        # and the copy steps whole C0-row groups, so both buffers carry it.
        gm_nz = tla.make_tensor(
            gm_ptr, tla.make_layout_with_tag((m_pad_c0, n_pad16), dtype_a, tla.arch.nZ)
        )
        l1_nz = tla.make_tensor_like(l1_ptr, gm_nz, tla.arch.nZ)
        with tla.cube():
            tla.copy(dst=l1_nz, src=gm_nz)

        # --- 6) Flat contiguous run: VectorLayout ---
        # A 1-N bias row is one run of elements, not a (1, N) tile. Both
        # endpoints of the staging pipeline carry the same tag: GM -> L1 first,
        # then L1 -> the bias table buffer.
        gm_bias = tla.tile_view(gm_bias_full, tla.make_shape(n), tla.make_coord(0))
        l1_bias = tla.make_tensor_like(l1_bias_ptr, gm_bias, tla.arch.VectorLayout)
        l0_bias = tla.make_tensor_like(l0_bias_ptr, l1_bias, tla.arch.VectorLayout)
        with tla.cube():
            tla.copy(dst=l1_bias, src=gm_bias)
            tla.copy(dst=l0_bias, src=l1_bias)
            tla.mmad(l0_c, l0_a, l0_b, init_c=True, unit_flag=0, bias=l0_bias)

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

    src_rank = len(src_desc.origin_shape)
    dst_rank = len(dst_desc.origin_shape)
    if src_rank == 3 or dst_rank == 3:
        if src_rank != 3 or dst_rank != 3:
            _op_error("copy", "batch-matrix copy requires two logical rank-3 tensors")
        if _route != ("gm", "l1"):
            _op_error(
                "copy",
                "batch-matrix copy currently supports GM to L1 layout conversion",
            )
        if scale is not None or params is not None:
            _op_error("copy", "GM to L1 batch copy takes its geometry from the layouts")
        if src_dtype not in ("f8e4m3fn", "f8e5m2", "f8e8m0"):
            _op_error("copy", "batch-matrix copy currently supports FP8 and E8M0")
        batch_layouts = (
            {("RowMajor", "zN"), ("ColumnMajor", "nZ")}
            if src_dtype in ("f8e4m3fn", "f8e5m2")
            else {
                ("RowMajorMxScaleA", "zZMxScale"),
                ("ColMajorMxScaleA", "zZMxScale"),
                ("RowMajorMxScaleB", "nNMxScale"),
                ("ColMajorMxScaleB", "nNMxScale"),
            }
        )
        if (src_layout, dst_layout) not in batch_layouts:
            _op_error(
                "copy",
                f"unsupported batch-matrix copy layout conversion "
                f"{src_layout} -> {dst_layout} for {src_dtype}",
            )
        for a, b in zip(src_desc.origin_shape, dst_desc.origin_shape, strict=True):
            if a is not None and b is not None and a != b:
                _op_error(
                    "copy", "batch-matrix copies require matching logical extents"
                )
        count = src_desc.origin_shape[0]
        if count is not None and not 0 <= count <= 65535:
            _op_error("copy", "batch count must fit the unsigned 16-bit hardware field")
        dst_step = dst_desc.stride[0]
        if dst_step is not None and dst_step % 32:
            _op_error(
                "copy", "FP8/E8M0 destination batch stride must be 32-byte aligned"
            )
        if dst_step is not None and not 0 < dst_step // 32 <= 65535:
            _op_error(
                "copy",
                "destination batch stride must fit the unsigned 16-bit 32-byte-block field",
            )
        src_step = src_desc.stride[0]
        if src_dtype == "f8e8m0" and src_step is not None and src_step % 2:
            _op_error("copy", "E8M0 source batch stride must preserve exponent pairs")

    compact = False
    if isinstance(params, (CopyGmToUbParams, CopyUbToGmParams)):
        try:
            params._validate()
        except (TypeError, ValueError) as exc:
            _op_error("copy", str(exc))
        expected_route = (
            ("gm", "ub") if isinstance(params, CopyGmToUbParams) else ("ub", "gm")
        )
        if _route != expected_route:
            direction = " -> ".join(expected_route).upper()
            _op_error("copy", f"{type(params).__name__} requires {direction}")
        compact = params.compact
    if compact:
        if src_layout != "RowMajor" or dst_layout != "RowMajor":
            _op_error("copy", "compact requires RowMajor src and dst")

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
        _check_fixpipe_ub_1byte_align(
            _route[1], dst_layout, dst_dtype, params, dst_desc
        )

        ctx = loc.context if loc is not None else mlir_ir.Context.current
        quant_mode_attr = mlir_ir.Attribute.parse(
            f"#tla.quant_mode<{params.quant_mode}>", context=ctx
        )
        l0c2ub_mode_attr = mlir_ir.Attribute.parse(
            f"#tla.l0c2ub_mode<{params.l0c2ub_mode}>", context=ctx
        )
        quant_scale = None
        quant_offset = None
        quant_tensor = None
        if params.quant_mode == QuantMode.PER_TENSOR:
            if not isinstance(params.quant_scale, (float, Float32)):
                _op_error(
                    "CopyL0C2DstParams",
                    "invalid argument 'quant_scale' (position 3): "
                    f"expected float or Float32, got {type(params.quant_scale)}",
                )
            quant_scale = _as_value(params.quant_scale)
            if params.quant_offset is not None:
                if dst_dtype not in ("i8", "u8"):
                    _op_error(
                        "copy",
                        "CopyL0C2DstParams.quant_offset only support i8/u8 dstDtype",
                    )
                # The op's operand is a fixed I32, so only int / Int32 can be
                # passed through. A wider or narrower integer would be accepted
                # here and then rejected by the verifier ("operand #1 must be
                # 32-bit signless integer"), long after the user's mistake.
                if isinstance(params.quant_offset, bool) or not isinstance(
                    params.quant_offset, (int, Int32)
                ):
                    _op_error(
                        "CopyL0C2DstParams",
                        "invalid argument 'quant_offset' (position 4): "
                        f"expected int or Int32, got {type(params.quant_offset).__name__}",
                    )
                quant_offset = _as_value(params.quant_offset)
        elif params.quant_mode == QuantMode.PER_CHANNEL:
            _require_category(
                "CopyL0C2DstParams", "quant_tensor", params.quant_tensor, "tensor", 5
            )
            quant_tensor = _as_value(params.quant_tensor)
            # The fixpipe reads one packed dequant element per output channel, and
            # that element is a u64: an fp32 scale in [31:0], the int8 sign selector
            # in [46], and the 9-bit offset in [45:37]. Anything else would reach
            # bc as a dma copy of the wrong element width -- the flat GM -> L1
            # VectorLayout route also stages i64/f64, but those are not packed
            # dequant words -- so refuse it here, where the message can name the
            # argument.
            quant_desc = _tla_tensor_type_for_mlir_value(quant_tensor)
            quant_dtype = quant_desc.element_type
            if str(quant_dtype).strip().lower() != "u64":
                _op_error(
                    "CopyL0C2DstParams",
                    "invalid argument 'quant_tensor' (position 5): expected a u64 "
                    f"tile of packed dequant elements, got {quant_dtype}",
                )
            # Width is not the whole contract. The bc wrapper materializes this
            # operand as an L1 VectorLayout run (`makeL1Tensor<..., uint64_t>`),
            # and `makeVectorTlaLayout` on the device side hardcodes the element
            # stride to a literal 1 -- a strided run would be read contiguously
            # anyway. `make_tensor` already refuses a VectorLayout tile whose
            # stride is anything but a static 1, but a tensor bound from the host
            # never goes through it.
            quant_addrspace = str(quant_desc.addrspace).strip().lower()
            if quant_addrspace != "l1":
                _op_error(
                    "CopyL0C2DstParams",
                    "invalid argument 'quant_tensor' (position 5): the per-channel "
                    "dequant elements must be staged in l1, "
                    f"got {quant_addrspace}",
                )
            if quant_desc.layout_tag != _B64_STAGING_LAYOUT_TAG:
                _op_error(
                    "CopyL0C2DstParams",
                    "invalid argument 'quant_tensor' (position 5): the per-channel "
                    f"dequant elements must be a flat {_B64_STAGING_LAYOUT_TAG} run, "
                    f"got {quant_desc.layout_tag}",
                )
            # A rank-1 VectorLayout descriptor is a single stride leaf.
            quant_stride_leaf = _components_to_type_tree(quant_desc.stride)
            quant_stride = (
                quant_stride_leaf[0]
                if isinstance(quant_stride_leaf, tuple)
                else quant_stride_leaf
            )
            if quant_stride != 1:
                _op_error(
                    "CopyL0C2DstParams",
                    "invalid argument 'quant_tensor' (position 5): the per-channel "
                    "dequant elements must be a unit-stride run, got stride "
                    f"{quant_stride if quant_stride is not None else 'dynamic'}",
                )
            # Static length check: the run must line up with the destination
            # tile's columns exactly -- quant and the C tile share the same
            # tiling, so a static mismatch means one of the two was cropped
            # against the wrong tail. A dynamic leaf skips the check, and
            # matching the channels is then the caller's contract.
            (quant_n,) = quant_desc.origin_shape
            (_, dst_n) = _flat_dim_pair_from_tree(dst_desc.origin_shape)
            if None not in (quant_n, dst_n) and quant_n != dst_n:
                _op_error(
                    "CopyL0C2DstParams",
                    "invalid argument 'quant_tensor' (position 5): the per-channel "
                    f"dequant run holds {quant_n} elements but the destination "
                    f"tile spans {dst_n} output channels; the run must match "
                    "the destination's channel count exactly",
                )
        params_value = _tla_ops_gen.CopyL0C2DstParams(
            _tla_type_bridge.copy_l0c2dst_params_type_get(ctx),
            params.unit_flag,
            params.relu_enable,
            quant_mode_attr,
            l0c2ub_mode_attr,
            quant_scale=quant_scale,
            quant_offset=quant_offset,
            quant_tensor=quant_tensor,
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
            elif src_dtype in _FP4_FORMATS and _route in _FP4_BYTE_STAGED_ROUTES:
                # Only the byte-staged routes pack the descriptor, so only
                # they constrain the shape. Other packed routes keep their own
                # denomination and are left alone. fp4 stores two logical
                # elements per byte, so an odd extent, stride or coordinate
                # starts or ends a tile mid-byte, and the other half belongs to
                # a neighbouring element the copy cannot carry or disturb.
                _elements_per_byte = _FP4_PER_BYTE
                _sides = (
                    ("src", src_desc, src_layout),
                    ("dst", dst_desc, dst_layout),
                )
                for _side, _desc, _layout in _sides:
                    _bad = _fp4_odd_byte_mapping(
                        _desc,
                        stride_field=_FP4_HALVED_STRIDE_LEAF.get(str(_layout)),
                        elements_per_byte=_elements_per_byte,
                    )
                    if _bad is not None:
                        _what, _value = _bad
                        _detail = (
                            "is not known until run time, so it cannot be "
                            "shown to be even"
                            if _value == "dynamic"
                            else f"is {_value}"
                        )
                        _op_error(
                            "copy",
                            f"invalid argument '{_side}': packed {src_dtype} is "
                            f"staged as bytes on this route, so its {_what} "
                            f"must be even -- it {_detail}. An odd one puts "
                            "the tile's rows, its C0 columns or its first "
                            "element half a byte off, which a byte-granular "
                            "copy cannot express",
                        )
                for _side, _desc, _layout in _sides:
                    _bad_extent = _partial_byte_fp4_extent(
                        _desc, elements_per_byte=_elements_per_byte
                    )
                    if _bad_extent is not None:
                        _detail = (
                            "is not known until run time, so it cannot be "
                            "shown to be even"
                            if _bad_extent == "dynamic"
                            else f"is {_bad_extent}"
                        )
                        _op_error(
                            "copy",
                            f"invalid argument '{_side}': packed {src_dtype} is "
                            f"staged as bytes on this route, so its innermost "
                            f"origin extent must be even -- it {_detail}. An "
                            "odd row ends mid-byte, and the other half of that "
                            "byte is an element outside the tile, which the "
                            "copy can neither carry nor disturb",
                        )
        params_value = None

    if isinstance(params, CopyPadParams):
        if _route not in {("gm", "ub"), ("gm", "l1")}:
            _op_error(
                "copy",
                "CopyPadParams is only supported on GM→UB or GM→L1 routes, "
                f"got {_route[0]}→{_route[1]}",
            )
        if _route == ("gm", "ub"):
            _pad_layouts_ok = (src_layout, dst_layout) in {
                ("RowMajor", "RowMajor"),
                ("VectorLayout", "VectorLayout"),
            }
            if not _pad_layouts_ok:
                _op_error(
                    "copy",
                    "CopyPadParams currently requires RowMajor→RowMajor or "
                    "VectorLayout→VectorLayout GM→UB "
                    f"(got {src_layout}→{dst_layout})",
                )
            _pad_dtype_ok = src_dtype in _COPY_PAD_GM_UB_SAFE_DTYPES
            _pad_dtype_supported = _COPY_PAD_GM_UB_SAFE_DTYPES
        else:
            # GM→L1: VectorLayout only (ND2NZ / fractal internal pads ≠ user Pad).
            if src_layout != "VectorLayout" or dst_layout != "VectorLayout":
                _op_error(
                    "copy",
                    "CopyPadParams on GM→L1 currently requires "
                    "VectorLayout→VectorLayout "
                    f"(got {src_layout}→{dst_layout})",
                )
            _pad_dtype_ok = src_dtype in _COPY_PAD_GM_L1_SAFE_DTYPES
            _pad_dtype_supported = _COPY_PAD_GM_L1_SAFE_DTYPES
        if not _pad_dtype_ok:
            _op_error(
                "copy",
                "CopyPadParams is not supported for dtype "
                f"{src_dtype}; supported: {sorted(_pad_dtype_supported)}",
            )
        elem_nbytes = dtype_size_bytes(src_dtype)
        if elem_nbytes <= 0:
            _op_error(
                "copy",
                f"CopyPadParams cannot size-check unknown dtype {src_dtype}",
            )
        try:
            params._validate(elem_nbytes=elem_nbytes)
        except (TypeError, ValueError) as exc:
            _op_error("copy", str(exc))

        # Pad-only static extract (unsigned + whole-float Int*/UInt* hosts).
        # Do not use shared ``_const_int_value`` here — that would change arange.
        try:
            left_static = _const_pad_count_value(params.left_padding)
            right_static = _const_pad_count_value(params.right_padding)
        except TypeError as exc:
            _op_error("copy", str(exc))
        # Reject Python bool and tla.Bool/i1 for pad counts (bool→0/1 is wrong).
        if _is_bool_like_pad_operand(params.left_padding) or _is_bool_like_pad_operand(
            params.right_padding
        ):
            _op_error(
                "copy",
                "CopyPadParams left/right_padding must not be bool/tla.Bool/i1",
            )
        if left_static is not None and left_static < 0:
            _op_error("copy", "CopyPadParams.left_padding must be >= 0")
        if right_static is not None and right_static < 0:
            _op_error("copy", "CopyPadParams.right_padding must be >= 0")
        if left_static is not None and left_static * elem_nbytes > 32:
            _op_error(
                "copy",
                f"CopyPadParams.left_padding*{elem_nbytes}B exceeds AscendC "
                f"per-side pad limit of 32B (got {left_static} elements)",
            )
        if right_static is not None and right_static * elem_nbytes > 32:
            _op_error(
                "copy",
                f"CopyPadParams.right_padding*{elem_nbytes}B exceeds AscendC "
                f"per-side pad limit of 32B (got {right_static} elements)",
            )

        src_inner = _copy_pad_innermost_extent(src_desc)
        dst_inner = _copy_pad_innermost_extent(dst_desc)
        ele_per_blk = 32 // elem_nbytes if elem_nbytes > 0 else 0
        if (
            src_inner is not None
            and dst_inner is not None
            and left_static is not None
            and right_static is not None
        ):
            need = src_inner + left_static + right_static
            if dst_inner < need:
                _op_error(
                    "copy",
                    "CopyPadParams requires dst innermost origin extent >= "
                    f"src + left_padding + right_padding "
                    f"(need {need}, got dst={dst_inner}, src={src_inner})",
                )
            # Multi-row: AscendC Normal mode AlignUp(blockLen+pads, 32B) per row,
            # then adds (dstGapElems / ELE_NUM_PER_BLK) * 32B (integer truncate).
            # Stride must cover AlignUp(need) and be a multiple of ELE_NUM_PER_BLK
            # so BC gap truncation cannot skew rows vs the declared layout.
            rows = _copy_pad_static_rows(dst_desc) or _copy_pad_static_rows(src_desc)
            dst_row_stride = _copy_pad_static_row_stride(dst_desc)
            if (
                rows is not None
                and rows > 1
                and dst_row_stride is not None
                and ele_per_blk > 0
            ):
                aligned_need = _copy_pad_align_up(need, ele_per_blk)
                if dst_row_stride < aligned_need or dst_row_stride % ele_per_blk != 0:
                    _op_error(
                        "copy",
                        "CopyPadParams multi-row dst row stride must be >= "
                        f"AlignUp(src+left+right, {ele_per_blk}) and a multiple "
                        f"of {ele_per_blk} "
                        f"(need {aligned_need}, got stride0={dst_row_stride}, "
                        f"src={src_inner}, left={left_static}, right={right_static})",
                    )

        try:
            pad_left_value = _as_i64_copy_pad_operand(
                params.left_padding, what="left_padding", loc=loc
            )
            pad_right_value = _as_i64_copy_pad_operand(
                params.right_padding, what="right_padding", loc=loc
            )
            pad_value_bits_value = _padding_value_as_i64_ssa(
                params.padding_value, src_dtype, elem_nbytes, loc=loc
            )
        except (TypeError, ValueError, TlaLoweringError) as exc:
            _op_error("copy", str(exc))
        pad_ctx = loc.context if loc is not None else mlir_ir.Context.current
        # Same pattern as CopyL0C2DstParams: typed params SSA, then copy(params=).
        params_value = _tla_ops_gen.CopyPadParams(
            _tla_type_bridge.copy_pad_params_type_get(pad_ctx),
            pad_left_value,
            pad_right_value,
            pad_value_bits_value,
            loc=loc,
        )
    elif _route == ("ub", "gm") and params is not None:
        if not isinstance(params, CopyUbToGmParams):
            _op_error(
                "copy",
                f"UB-to-GM copy expects CopyUbToGmParams, got {type(params).__name__}",
            )
    elif params is not None and not isinstance(
        params, (CopyGmToUbParams, CopyL0C2DstParams)
    ):
        # Reject unknown CopyParams subclasses early when they lack atomic_mode.
        if not hasattr(params, "atomic_mode") and _route[0] != "l0c":
            _op_error(
                "copy",
                f"unsupported params type {type(params).__name__} for route "
                f"{_route[0]}→{_route[1]}",
            )

    # Check if atomic mode enabled and acquire lowered atomic_mode_attr
    atomic_mode_attr = None
    atomic_mode = getattr(params, "atomic_mode", None) if params is not None else None
    if atomic_mode is not None and atomic_mode != AtomicMode.NONE:
        if not isinstance(atomic_mode, AtomicMode):
            raise NotImplementedError(
                f"currently unsupported atomic mode {str(atomic_mode)}"
            )

        if _route[1] != "gm":
            raise TlaLoweringError(
                f"When atomic operation is enabled, the dst location should only be GM but got {_route[1]}"
            )

        if _route[0] not in ("ub", "l0c"):
            raise TlaLoweringError("atomic writeback requires a UB or L0C source")

        if dst_dtype not in ("f32", "f16", "i16", "i32", "i8", "bf16"):
            raise TlaLoweringError(
                "The supported atomic operation's data type includes f32, f16, i16, i32, i8, bf16, "
                f"the data type {dst_dtype} is not supported"
            )

        ctx = loc.context if loc is not None else mlir_ir.Context.current
        atomic_mode_attr = mlir_ir.Attribute.parse(
            f"#tla.atomic_mode<{atomic_mode.value}>", context=ctx
        )

    # L2 cache hints via copy(cache_policy=...). Non-NORMAL values become
    # optional copy attrs so lowering can append the BC gmCachePolicy operand.
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    src_cache_policy, dst_cache_policy = _resolve_copy_l2_cache_policies(
        _route, cache_policy
    )
    copy_attrs: dict[str, Any] = {}
    if compact:
        copy_attrs["compact"] = True
    # Non-NORMAL policies are only returned for GM endpoints by the helper above.
    if src_cache_policy != CachePolicy.NORMAL:
        copy_attrs["src_cache_policy"] = mlir_ir.Attribute.parse(
            f"#tla.cache_policy<{src_cache_policy.value}>", context=ctx
        )
    if dst_cache_policy != CachePolicy.NORMAL:
        copy_attrs["dst_cache_policy"] = mlir_ir.Attribute.parse(
            f"#tla.cache_policy<{dst_cache_policy.value}>", context=ctx
        )

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

    # UB->UB is the meta-op's MOV.UB.TO.UB block move for every row length; it
    # rounds a ragged row up to whole blocks, which only copies for the
    # geometries this guard admits.
    if _route == ("ub", "ub") and src_layout == "RowMajor" and dst_layout == "RowMajor":
        _ubub_require_metaop_representable(
            _ubub_static_geometry(src_desc, dst_desc, src_dtype)
        )

    # Transposing UB->UB is the meta-op's VNCHWCONV, which transposes whole
    # 16-row-by-C0-column blocks; this guard admits only the tiles that are a
    # whole number of them.
    if (
        _route == ("ub", "ub")
        and src_layout == "RowMajor"
        and dst_layout == "ColumnMajor"
    ):
        _validate_ubub_transpose_shape(src_desc, dst_desc, src_dtype)

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
    id: int = -1,
    *,
    loc: mlir_ir.Location | None = None,
) -> TlaFlag:
    """Directory: Sync Control
    Description:
        Create an in-pipe synchronization flag between two pipes.

        Parameters:
        - `name` (`str`): In-pipe flag name. Required.
        - `src_pipe` (`PipeLike`): Source pipe id; see
          **System Variable Access / Pipes**. Required.
        - `dst_pipe` (`PipeLike`): Destination pipe id; see
          **System Variable Access / Pipes**. Required.
        - `id` (`int`): Hardware event id. Optional, default `-1` for automatic allocation.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Creates a flag handle; `set_flag`/`wait_flag` must be paired inside cube/vector regions.
        - `id` must be `-1` or `0..7`. Automatic allocation skips explicit ids for the same pipe pair.
        - Explicit ids may be reused; the caller must ensure their synchronization lifetimes do not overlap.

        Example:
        ```python
        # MTE2 finishes a GM→UB copy, then VECTOR may consume the UB tile.
        ub_loaded = tla.flag("ub_loaded", src_pipe=tla.pipes.MTE2, dst_pipe=tla.pipes.VECTOR)
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
    if isinstance(id, bool) or not isinstance(id, int):
        _op_error(
            "flag",
            f"invalid argument 'id' (position 3): expected int in {{-1, 0..7}}, got {_type_name(id)}",
        )
    if id != -1 and not 0 <= id <= 7:
        _op_error("flag", "id must be -1 or in range 0..7")
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
        id=id,
        loc=loc,
    )


#: Cross-core flag names owned by :func:`sync_all` / :func:`sync_all_aiv`.
#: ``tla.cross_flag`` rejects them so a user flag cannot alias the internal
#: handshake emitted by the two helpers.
_SYNC_ALL_RESERVED_FLAG_NAMES = frozenset(
    {
        "__tla_sync_all_inter_core_aiv_ready__",  # sync all aiv
        "__tla_sync_all_intra_block_aiv_ready__",  # sync inner aiCore, aic wait aiv
        "__tla_sync_all_inter_core_aic_ready__",  # all aiCores, sync all aic
        "__tla_sync_all_intra_block_aic_ready__",  # sync inner aiCore, aiv wait aic
    }
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
    if name in _SYNC_ALL_RESERVED_FLAG_NAMES:
        _op_error(
            "cross_flag",
            f"invalid argument 'name': {name!r} is reserved for the "
            "tla.sync_all / tla.sync_all_aiv handshake",
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
        allowed = (0, 1)
        state = _runtime._current_frontend_state()
        if state is not None and state.core_ratio == (1, 1):
            allowed = (0,)
        if (
            not isinstance(aiv_id, int)
            or isinstance(aiv_id, bool)
            or aiv_id not in allowed
        ):
            if allowed == (0,):
                _op_error(
                    op_name,
                    "invalid argument 'aiv_id': core_ratio=tla.arch.C1V1 requires "
                    f"compile-time 0, got {aiv_id!r}",
                )
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
            tla.cross_core_set_flag(cf, tla.pipes.CUBE)
            # mode=4: tla.cross_core_set_flag(cf, tla.pipes.CUBE, aiv_id=0)
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
          Use `tla.pipes.SCALAR` for `PIPE_S`, `tla.pipes.VECTOR` for `PIPE_V`, etc.
        - `aiv_id` (`int | None`): Target AIV id; omit for broadcast/default routing. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()` or `tla.vector()`; when `mode=4`, `aiv_id` must be 0 or 1.
        - Under `@tla.kernel(core_ratio=tla.arch.C1V1)`, Ascend 950 CV 1:1
          handshakes expect `pipe=tla.pipes.SCALAR` (`PIPE_S`), matching AscendC
          SyncAll. The compiler does not rewrite a wrong wait pipe; the caller
          must pass `SCALAR` explicitly (see
          `python/tla_dsl/tests/end_to_end/basic_mixed_core_ratio_11.py`).

        Example:
        ```python
        with tla.vector():
            # Default MIX 1:2 often waits on VECTOR:
            tla.cross_core_wait_flag(cf, tla.pipes.VECTOR)
            # CV 1:1 (core_ratio=C1V1): wait on SCALAR / PIPE_S, aiv_id=0 only:
            # tla.cross_core_wait_flag(cf, tla.pipes.SCALAR, aiv_id=0)
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
        - Allowed pipes depend on the region. In `tla.cube()`: `CUBE`, `MTE1`,
          `MTE2`, `FIX`, `ALL`. In `tla.vector()`: `MTE2`, `MTE3`, `ALL`
          (Ascend 950 does not support `pipe_barrier<PIPE_V>`).
        - Use `tla.pipes.ALL` to drain every pipe at kernel end.

        Example:
        ```python
        with tla.vector():
            tla.pipe_barrier(tla.pipes.MTE2)
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


#: Pipes a ``sync_all_aiv`` helper accepts for the trigger and the wait: the
#: rendezvous signal is carried by a copy pipe or by every pipe at once.
_SYNC_ALL_AIV_PIPES = ("mte2", "mte3", "all")


def _require_sync_all_aiv_pipe(
    op_name: str, name: str, value: Any, position: int
) -> str:
    token = _token(value)
    if token is None or token not in _SYNC_ALL_AIV_PIPES:
        _op_error(
            op_name,
            f"invalid argument '{name}' (position {position}): expected one of "
            f"{list(_SYNC_ALL_AIV_PIPES)}, got {_type_name(value)}",
        )
    return token


def _sync_all_flag(name: str, mode: int, *, loc: mlir_ir.Location | None) -> Any:
    """Return the internal cross-core flag ``name`` of the enclosing kernel.

    The flag op is inserted at the beginning of the ``tla.func`` entry block and
    cached in the emission state, so repeated calls in one kernel share a single
    definition and every use is dominated by it.
    """
    state = _runtime._current_frontend_state()
    if state is None:
        raise TlaCoreAPIError(
            "tla.sync_all: handshake flags must be created during kernel emission"
        )
    cached = state.internal_cross_flags.get((name, mode))
    if cached is not None:
        return cached
    try:
        func_op = mlir_ir.InsertionPoint.current.block.owner
    except ValueError:
        func_op = None
    while func_op is not None and func_op.name != "tla.func":
        func_op = func_op.parent
    if func_op is None:
        raise TlaCoreAPIError(
            "tla.sync_all: must be called inside a @tla.kernel-decorated function"
        )
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    entry_block = func_op.regions[0].blocks[0]
    with mlir_ir.InsertionPoint.at_block_begin(entry_block):
        _flag = _tla_ops_gen.cross_flag(
            _tla_type_bridge.cross_flag_type_get(ctx, mode), name, loc=loc
        )
    state.internal_cross_flags[(name, mode)] = _flag
    return _flag


@dsl_user_op
def sync_all_aiv(
    trigger_pipe: PipeLike = _runtime.pipes.ALL,
    wait_pipe: PipeLike = _runtime.pipes.ALL,
    *,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Directory: Sync Control

    Description:
        Synchronize all AIV cores that participate in the kernel.

        Every AIV signals its arrival through a cross-core flag and then waits
        for all peers, so the call is an all-core synchronization for the AIV
        cores.

        Parameters:
        - `trigger_pipe` (`PipeLike`): Pipe that issues the arrival signal.
          Optional, default `tla.pipes.ALL`; only `tla.pipes.MTE2`,
          `tla.pipes.MTE3` and `tla.pipes.ALL` are supported.
          `tla.pipes.ALL` first drains every pipe and signals from `MTE3`.
        - `wait_pipe` (`PipeLike`): Pipe that waits for the peers. Optional,
          default `tla.pipes.ALL`; only `tla.pipes.MTE2`, `tla.pipes.MTE3` and
          `tla.pipes.ALL` are supported. `tla.pipes.ALL` waits on the scalar
          pipe.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vector()`: valid in an AIV-only kernel and
          in the vector part of a MIX kernel.
        - Every AIV must call it the same number of times, in the same order,
          and with the same pipe selection; otherwise the rendezvous never
          completes.
        - A MIX kernel that needs a cube/vector barrier uses `tla.sync_all()`
          on both sides instead.

        Example:
        ```python
        with tla.vector():
            tla.sync_all_aiv()
            # or pick the copy pipe that carries the data
            tla.sync_all_aiv(tla.pipes.MTE3, tla.pipes.ALL)
        ```
    """
    if not _runtime._has_enclosing_region("vector"):
        raise TlaLoweringError("tla.sync_all_aiv must be called inside tla.vector()")

    _require_frontend_state("sync_all_aiv")
    trigger = _require_sync_all_aiv_pipe(
        "sync_all_aiv", "trigger_pipe", trigger_pipe, 0
    )
    wait = _require_sync_all_aiv_pipe("sync_all_aiv", "wait_pipe", wait_pipe, 1)
    _flag = _sync_all_flag("__tla_sync_all_inter_core_aiv_ready__", 0, loc=loc)
    pipe_barrier(trigger, loc=loc)
    cross_core_set_flag(_flag, "mte3" if trigger == "all" else trigger, loc=loc)
    cross_core_wait_flag(_flag, "scalar" if wait == "all" else wait, loc=loc)


@dsl_user_op
def sync_all(*, loc: mlir_ir.Location | None = None) -> None:
    """Directory: Sync Control

    Description:
        Synchronize the cube and vector cores of a MIX kernel.

        Write one call on the cube side and one on the vector side. Both calls
        are needed: the cube waits for every AIV to arrive and then releases
        them, the vector core signals its arrival and waits for the release.

        Parameters:
        - None.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Only valid in a MIX kernel declared with `core_ratio`; an AIV-only
          kernel uses `tla.sync_all_aiv()` instead.
        - Must be called inside `tla.cube()` or `tla.vector()`; the cube call
          pairs with the vector call of the same kernel.

        Example:
        ```python
        @tla.kernel(core_ratio=tla.arch.C1V2)
        def mixed(...):
            with tla.cube():
                ...
                tla.sync_all()
            with tla.vector():
                tla.sync_all()
                ...
        ```
    """
    _require_frontend_state("sync_all")
    state = _runtime._current_frontend_state()
    if state is None or state.core_ratio is None:
        raise TlaLoweringError(
            "tla.sync_all synchronizes the cube and vector cores of a MIX "
            "kernel; declare @tla.kernel(core_ratio=...) or use "
            "tla.sync_all_aiv() in an AIV-only kernel"
        )
    in_cube = _runtime._has_enclosing_region("cube")
    in_vector = _runtime._has_enclosing_region("vector")

    intra_block_aiv_ready = _sync_all_flag(
        "__tla_sync_all_intra_block_aiv_ready__", 2, loc=loc
    )
    inter_core_aic_ready = _sync_all_flag(
        "__tla_sync_all_inter_core_aic_ready__", 0, loc=loc
    )
    intra_block_aic_ready = _sync_all_flag(
        "__tla_sync_all_intra_block_aic_ready__", 2, loc=loc
    )
    if in_cube:
        pipe_barrier("all", loc=loc)
        cross_core_wait_flag(intra_block_aiv_ready, "scalar", loc=loc)
        cross_core_set_flag(inter_core_aic_ready, "fix", loc=loc)
        cross_core_wait_flag(inter_core_aic_ready, "scalar", loc=loc)
        cross_core_set_flag(
            intra_block_aic_ready,
            "mte3" if state.core_ratio == (1, 1) else "scalar",
            loc=loc,
        )
    elif in_vector:
        pipe_barrier("all", loc=loc)
        cross_core_set_flag(intra_block_aiv_ready, "mte3", loc=loc)
        cross_core_wait_flag(intra_block_aic_ready, "scalar", loc=loc)
    else:
        raise TlaLoweringError(
            "tla.sync_all must be called inside tla.cube() or tla.vector()"
        )


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
        - `id` (`int`): Mutex instance id. Optional, default `-1`, but the current
          bitcode call lowering does not support auto allocation: always pass an
          explicit id in `0..31`. No id in this range is reserved.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - `resource` must be non-empty; `id` must be in `0..31`. `id=-1` is accepted
          by the frontend but fails later with `mutex id auto allocation is not
          implemented for bitcode call lowering`.
        - A mutex works only within one AIC or AIV; use `tla.cross_flag` across cores.

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
        - Which pipes are valid depends on the enclosing core: `SCALAR`, `CUBE`,
          `MTE1`, `MTE2`, `MTE3` and `FIX` inside `tla.cube()`; `SCALAR`,
          `VECTOR`, `MTE2` and `MTE3` inside `tla.vector()`. `ALL` has no mutex
          form. A pipe the core does not support is reported when the kernel is
          compiled, not at this call.

        Example:
        ```python
        tla.mutex_lock(mtx, pipe=tla.pipes.MTE2)
        # Scalar pipe, to guard a buffer the scalar unit reads or writes.
        # `tla.mutex` returns a value with this method form:
        mtx.lock(pipe=tla.pipes.SCALAR)
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
        - Which pipes are valid depends on the enclosing core: `SCALAR`, `CUBE`,
          `MTE1`, `MTE2`, `MTE3` and `FIX` inside `tla.cube()`; `SCALAR`,
          `VECTOR`, `MTE2` and `MTE3` inside `tla.vector()`. `ALL` has no mutex
          form. A pipe the core does not support is reported when the kernel is
          compiled, not at this call.

        Example:
        ```python
        tla.mutex_unlock(mtx, pipe=tla.pipes.MTE2)
        # Scalar pipe, to guard a buffer the scalar unit reads or writes.
        # `tla.mutex` returns a value with this method form:
        mtx.unlock(pipe=tla.pipes.SCALAR)
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
def data_sync_barrier(
    domain: DsbTarget, *, loc: mlir_ir.Location | None = None
) -> None:
    """Directory: Sync Control

    Description:
        Insert a data synchronization barrier over a storage domain. Blocks
        subsequent instructions until the memory access instructions of
        `domain` that were issued earlier have completed.

        Parameters:
        - `domain` (`DsbTarget`): Storage domain to wait on. Required. `GM_UB`
          waits for previously issued GM and UB accesses; `GM` for GM only; `UB`
          for UB only; `SEQ` for sequencer accesses (including the parameter
          buffer).

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function; the
          kernel body, `tla.cube()` and `tla.vector()` are all valid.
        - Must not be called inside `tla.vec.func()`; `dsb` is a scalar-pipeline
          instruction over whole storage domains, not a per-thread view.
        - `UB` is rejected inside `tla.cube()`: AIC has no UB and would not
          issue the barrier, so use `GM`, `GM_UB` or `SEQ` there, or move the
          call into `tla.vector()`. In a mixed kernel's scalar body the `UB`
          domain is a no-op on the AIC copy.
        - `GM_UB` covers GM and UB only: L1 and the L0A/L0B/L0C buffers are not
          waited on; order those with `tla.pipe_barrier` (`MTE1`/`FIX`/`CUBE`)
          or `tla.flag` / `set_flag` / `wait_flag`.
        - The cross-core visibility guarantee applies to GM accesses that
          bypass the D-cache. Ordinary
          scalar-pipe GM writes stay invisible to other engines and other
          cores until kernel end, and V-pipe UB stores are not waited by `UB` /
          `GM_UB` — order a `vec.func` UB store with a flag pair before draining
          it to GM.
        - Does not participate in `auto_sync`; place it outside hot inner loops,
          each barrier costs a wait on every iteration.

        Example:
        ```python
        with tla.vector():
            # src holds two Int32 words. The consumer can poll the flag at
            # src.ptr with tla.arch.load_bypass_dcache, then read the data
            # at src.ptr + 1 with the same API. See tests/end_to_end/data_sync_barrier.py.
            tla.arch.store_bypass_dcache(src.ptr + 1, data)
            tla.data_sync_barrier(tla.params.DsbTarget.GM)
            tla.arch.store_bypass_dcache(src.ptr, flag)
        ```
    """
    _require_frontend_state("data_sync_barrier")
    if not isinstance(domain, DsbTarget):
        _op_error(
            "data_sync_barrier",
            f"invalid argument 'domain': expected tla.params.DsbTarget, got {domain!r}",
        )
    if _runtime._has_enclosing_region("vec.func"):
        _op_error(
            "data_sync_barrier",
            "must not be called inside tla.vec.func(); move the barrier outside the region",
        )
    if _runtime._has_enclosing_region("cube") and domain == DsbTarget.UB:
        _op_error(
            "data_sync_barrier",
            "DsbTarget.UB is not issued on AIC; use DsbTarget.GM, "
            "DsbTarget.GM_UB or DsbTarget.SEQ inside tla.cube(), or move the "
            "call into tla.vector()",
        )
    ctx = loc.context if loc is not None else mlir_ir.Context.current
    domain_attr = mlir_ir.Attribute.parse(
        f"#tla.dsb_target<{domain.name.lower()}>", context=ctx
    )
    return _tla_ops_gen.data_sync_barrier(domain_attr, loc=loc)


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
        Enter a vector-function region (`tla.vec.func`) in `simd` or `simt` mode.

        Parameters:
        - `mode` (`str`): `simd` (default, register-vector / mask) or `simt`
          (per-thread). Optional, default `"simd"`.
        - `thread_block_dim` (`int | tuple[int, int, int] | list[int] | None`):
          SIMT thread-block shape; only valid with `mode="simt"`. An `int n`
          means `(n, 1, 1)`. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must nest inside `tla.vector()`.
        - Register-vector / mask APIs and `local_mem_bar` must run in this region.
        - `mode` must be `"simd"` or `"simt"`.
        - `thread_block_dim` is required with `mode="simt"` and rejected with
          `mode="simd"`.
        - `thread_block_dim` entries must be positive, and their product must
          not exceed 2048 threads per block.
        - A SIMT region may capture scalar runtime values from the kernel
          body, plus tensor pointers read through `t.ptr` /
          `t.ptr_with_coord`; captures of `index`-typed values are rejected.
        - SIMT: allocate on-chip scratch with `tla.allocate` outside the region;
          index work with `tla.arch.thread_idx` / `thread_block_dim`, sync with
          `tla.arch.sync_threads`.

        Example:
        ```python
        with tla.vector():
            with tla.vec.func(mode="simd"):
                z = tla.add(x_reg, y_reg)

        with tla.vector():
            with tla.vec.func(mode="simt", thread_block_dim=400):
                tid, _, _ = tla.arch.thread_idx()
                nthreads, _, _ = tla.arch.thread_block_dim()
                for i in tla.range(tid, 400, nthreads):
                    gm_c[i] = gm_a[i] + gm_b[i]
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
        ("hif8", "hif8", "f32"),
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
    bias: mlir_ir.Value | None = None,
) -> None:
    """Check one matmul's operands; the lhs layout carries the gemv mode."""
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
    _op_name = is_mx and "mmad_mx" or "mmad"
    if not _validate_mmad_dtypes(*element_types, is_mx=is_mx):
        raise TlaLoweringError(
            f"unsupported tla.{_op_name} element types, got {element_types}"
        )

    lhs_m, lhs_k = _flat_dim_pair_from_tree(lhs_desc.origin_shape)
    rhs_k, rhs_n = _flat_dim_pair_from_tree(rhs_desc.origin_shape)
    acc_m, acc_n = _flat_dim_pair_from_tree(acc_desc.origin_shape)

    # GEMV is not a separate op: it is what a 1-D VectorLayout lhs means. The
    # tile then carries K alone, with the row implied to be one, while rhs and
    # acc keep their matrix layouts. Read as a rank-2 tree the lhs yields
    # (None, None), so the matrix-form check below would skip every comparison
    # and let a mismatched K or N through; the gemv branch spells them out.
    is_gemv = not is_mx and lhs_desc.layout_tag == "VectorLayout"
    if is_gemv:
        lhs_origin = lhs_desc.origin_shape
        if not isinstance(lhs_origin, tuple) or len(lhs_origin) != 1:
            raise TlaLoweringError(
                "unsupported tla.mmad gemv lhs shape; a VectorLayout tile is "
                f"rank-1, got origin shape {lhs_origin!r}"
            )
        (lhs_k,) = lhs_origin
        if acc_m is not None and acc_m != 1:
            raise TlaLoweringError(
                "unsupported tla.mmad gemv shape contract; a VectorLayout lhs "
                f"puts the row on one M, got acc M = {acc_m}"
            )
        if isinstance(lhs_k, int) and isinstance(rhs_k, int) and lhs_k != rhs_k:
            raise TlaLoweringError(
                "unsupported tla.mmad gemv shape contract; expected lhs(K), rhs(KxN), "
                f"acc(1xN), got lhs K = {lhs_k}, rhs K = {rhs_k}"
            )
        if isinstance(rhs_n, int) and isinstance(acc_n, int) and rhs_n != acc_n:
            raise TlaLoweringError(
                "unsupported tla.mmad gemv shape contract; expected rhs N matching acc "
                f"N, got rhs N = {rhs_n}, acc N = {acc_n}"
            )
    elif None not in (lhs_m, lhs_k, rhs_k, rhs_n, acc_m, acc_n) and (
        lhs_k != rhs_k or lhs_m != acc_m or rhs_n != acc_n
    ):
        raise TlaLoweringError(
            "unsupported tla.mmad tile shape contract; expected lhs(MxK), rhs(KxN), acc(MxN)"
        )

    # The bias table feeds the accumulator directly, so it has to carry the same
    # element type the accumulator does: fp32 alongside fp32 accumulation, i32
    # alongside the integer MMAD route. The device reads it element for element,
    # and a mismatch would be reinterpreted rather than converted.
    if bias is not None:
        bias_desc = _tla_tensor_type_for_mlir_value(bias)
        if bias_desc.element_type != acc_desc.element_type:
            raise TlaLoweringError(
                f"unsupported tla.{_op_name} bias element type; expected the "
                f"accumulator type {acc_desc.element_type}, got "
                f"{bias_desc.element_type}"
            )
        if bias_desc.addrspace != "biasbuf":
            raise TlaLoweringError(
                f"unsupported tla.{_op_name} bias addrspace; expected biasbuf, "
                f"got {bias_desc.addrspace}"
            )
        # VectorLayout is rank-1 by definition -- its verifier admits exactly
        # one shape leaf -- so a bias tile is a flat N-element run and there is
        # no leading extent to check. All that is left is N against the
        # accumulator's.
        if bias_desc.layout_tag != "VectorLayout":
            raise TlaLoweringError(
                f"unsupported tla.{_op_name} bias layout; expected a 1-D "
                f"VectorLayout tile, got {bias_desc.layout_tag}"
            )
        (bias_n,) = bias_desc.origin_shape
        if None not in (bias_n, acc_n) and bias_n != acc_n:
            raise TlaLoweringError(
                f"unsupported tla.{_op_name} bias shape; expected N matching acc "
                f"({acc_n}), got {bias_n}"
            )

    expected_layouts = (
        (acc, "L0Clayout"),
        (lhs, "VectorLayout" if is_gemv else "zN"),
        (rhs, "nZ"),
    )
    for operand, expected in expected_layouts:
        layout = _tla_tensor_type_for_mlir_value(operand).layout_tag
        if layout != expected:
            raise TlaLoweringError(
                "unsupported tla.mmad operand layout; expected acc L0Clayout, rhs nZ,"
                "lhs zN (matrix form) or VectorLayout (gemv form)"
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
    bias: Tensor | None = None,
    loc: mlir_ir.Location | None = None,
    **extra_kwargs: object,
) -> None:
    """Directory: Matrix Compute
    Description:
        Emit matrix-multiply-accumulate on TLA tiles.

        Parameters:
        - `acc` (`Tensor`): Accumulator / output tile. Required.
        - `lhs` (`Tensor`): Left-hand matrix tile. Its layout selects the mode:
          `zN` is the matrix form, a rank-1 `VectorLayout` row the GEMV form. Required.
        - `rhs` (`Tensor`): Right-hand matrix tile. Required.
        - `init_c` (`bool | Bool | None`): Whether to clear the accumulator first;
          defaults to `False` when omitted. Ignored when `bias` is provided, since
          the bias then decides what the accumulator starts from. Optional,
          default `None`.
        - `unit_flag` (`IndexLike | None`): Unit-flag control bits; defaults to `0`
          when omitted. Optional, default `None`.
        - `compute_order` (`ComputeOrder`): M/N compute-direction priority; default `M_FIRST`.
        - `hf32_mode` (`HF32Mode`): HF32 rounding mode for FP32 operands in L0A/L0B
          before the matrix multiply. Optional, default `HF32_DISABLE`.
        - `bias` (`Tensor | None`): 1-D `VectorLayout` tile of `N` elements on the bias
          table buffer (C2) that the first matmul of each block accumulates from;
          its element type follows the accumulator. When it is not `None` the
          accumulator starts from the bias instead of from the values already in
          L0C, so `init_c` has no effect on the instruction that consumes it.
          Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()`; `acc`/`lhs`/`rhs` must be matching L0 tiles.
        - Supported accumulating dtypes are fp32/i32, supported (`lhs`, `rhs`, `acc`) dtype
          conbinations are as fllows.
        - The `f8e4m3fn` and `f8e5m2` matmul both support non-MxScale matmul and MxScale matmul,
          in the tla.mmad op scale is unsupported.
        - `init_c` accepts only a Python `bool` or an `i1` SSA value.
        - GEMV form: `lhs` is a 1-D `VectorLayout` tile carrying the K elements of
          a single row, `rhs` stays `nZ`, and `acc` stays `L0Clayout` with M = 1;
        - When `bias` is set: tile must live in `biasbuf` with rank-1 `VectorLayout`,
          element type matching `acc`, and length `N` matching `acc`'s N; stage it
          with `GM`→`L1`→`biasbuf` `tla.copy` (see `copy`). `init_c` is ignored.
        - Unknown keyword arguments are not accepted; passing any raises an error.

        Example:
        ```python
        # Before: l0a / l0b hold the current K-slice; l0c is the accumulator on L0C.
        with tla.cube():
            tla.mmad(l0c, l0a, l0b, init_c=True, unit_flag=0b11)
            # With bias (l0_bias already on biasbuf via copy):
            # tla.mmad(l0c, l0a, l0b, bias=l0_bias)
            # GEMV form: l0a_row is a 1-D VectorLayout tile of K elements.
            # tla.mmad(l0c, l0a_row, l0b, init_c=True, unit_flag=0b11)
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

    if bias is not None:
        _require_category("mmad", "bias", bias, "tensor", 7)
        bias_value = _as_value(bias)
    else:
        bias_value = None

    # The layout of lhs is the mode switch: a rank-1 VectorLayout row means the
    # gemv form, a zN tile the matrix form. The op does not spell the mode out;
    # the pass derives disableGemv from the layout at the bc boundary.
    _validate_mmad_contract(
        acc_value, lhs_value, rhs_value, is_mx=False, bias=bias_value
    )
    _tla_ops_gen.mmad(
        acc_value,
        lhs_value,
        rhs_value,
        init_c_value,
        unit_flag_value,
        bias=bias_value,
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
    bias: Tensor | None = None,
    loc: mlir_ir.Location | None = None,
    **extra_kwargs: object,
) -> None:
    """Directory: Matrix Compute
    Description:
        Microscaling (MX) matrix-multiply-accumulate on L0 tiles.

        Parameters:
        - `acc` (`Tensor`): Accumulator / output tile on L0C, always fp32. Required.
        - `lhs` (`Tensor`): Left-hand matrix tile on L0A. Required.
        - `rhs` (`Tensor`): Right-hand matrix tile on L0B. Required.
        - `init_c` (`bool | Bool | None`): Whether to clear the accumulator first;
          defaults to `False` when omitted. Ignored when `bias` is provided, since
          the bias then decides what the accumulator starts from. Optional,
          default `None`.
        - `unit_flag` (`IndexLike | None`): Unit-flag control bits; defaults to `0`
          when omitted. Optional, default `None`.
        - `compute_order` (`ComputeOrder`): M/N compute-direction priority; default `M_FIRST`.
        - `bias` (`Tensor | None`): 1-D `VectorLayout` tile of `N` elements on the bias
          table buffer (C2) that the first matmul of each block accumulates from;
          always f32, matching the accumulator. When it is not `None` the
          accumulator starts from the bias instead of from the values already in
          L0C, so `init_c` has no effect on the instruction that consumes it.
          Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()`; `acc`/`lhs`/`rhs` must be matching L0 tiles.
        - Operands must be an `f8e4m3fn`/`f8e5m2` pair or an `f4e2m1`/`f4e1m2`
          pair; either pair may mix its two formats, and the accumulator is fp32.
        - The `f8e4m3fn` and `f8e5m2` matmul both support non-MxScale matmul and MxScale matmul,
          in the tla.mmad_mx op scale is needed.
        - Both operands must have been loaded by `tla.copy(..., scale=...)`.
        - When `bias` is set: tile must live in `biasbuf` with rank-1 `VectorLayout`,
          element type f32 matching `acc`, and length `N` matching `acc`'s N; stage
          it with `GM`→`L1`→`biasbuf` `tla.copy` (see `copy`). `init_c` is ignored.

        Example:
        ```python
        # Before: l1a/l1b hold fp8 operands, l1sa/l1sb their e8m0 scale blocks.
        with tla.cube():
            tla.copy(l0a, l1a, scale=l1sa)   # scale rides the load, not the matmul
            tla.copy(l0b, l1b, scale=l1sb)
            tla.mmad_mx(l0c, l0a, l0b, init_c=True)
            # With bias: tla.mmad_mx(l0c, l0a, l0b, bias=l0_bias)
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

    if bias is not None:
        _require_category("mmad_mx", "bias", bias, "tensor", 6)
        bias_value = _as_value(bias)
    else:
        bias_value = None

    _validate_mmad_contract(
        acc_value, lhs_value, rhs_value, is_mx=True, bias=bias_value
    )
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
        bias=bias_value,
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
    """Directory: SIMD Vector Compute / Data Fill
    Description:
        Fill a vector SSA with a scalar, or broadcast a one-lane fragment.

        Parameters:
        - `value` (`bool | int | float | Numeric | VectorSSA`): Fill constant, a
          runtime scalar SSA, or a one-lane vector fragment to broadcast. Required.
        - `dtype` (`type[Numeric]`): Vector element type. Required. Supported:
          Bool, i8/u8, i16/u16, i32/u32, i64/u64, f16, bf16, f32, f8e4m3fn,
          f8e5m2, f8e8m0, hif8, f4e2m1x2, f4e1m2x2. Complex has no TLA vector
          type and is rejected.
        - `mask` (`MaskSSA | None`): Predicates which lanes are written. Optional;
          defaults to writing every lane.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func(mode="simd")`; `value` must be a
          Python scalar literal, a host `Numeric`, a runtime scalar SSA (e.g. a
          kernel scalar argument or a tensor element read), or a one-lane vector
          fragment.
        - A runtime scalar must be the element type or its broadcast carrier:
          Bool/hif8/e8m0/fp4x2 broadcast an i8 byte, unsigned broadcasts the
          same-width signless integer, and every other type broadcasts itself.
        - `f4e2m1x2` / `f4e1m2x2` take one packed pair byte (low nibble first),
          not two independent FP4 values. `hif8` takes an explicit encoded byte
          (there is no host encoder). `f8e8m0` takes a power of two in
          `[2**-127, 2**127]` (for example 0.5, 1.0 or 2.0).
        - A trace-time `f8e4m3fn` / `f8e5m2` scalar must be exactly
          representable in the target format; the frontend rejects inexact
          values rather than rounding them. Pass an already-rounded value
          (for example 0.125), or a runtime scalar that was converted
          explicitly.
        - 64-bit broadcasts lower through linked bitcode helpers. A 64-bit
          fragment source needs a 64-bit reduction, which `tla.reduce` does not
          support yet.
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
    if _runtime._in_simt_vec_func():
        _op_error("full", "is only supported in tla.vec.func(mode='simd')")
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
        if dtype_token in ("i64", "u64"):
            _op_error(
                "full",
                "64-bit fragment source is not supported: it needs a 64-bit "
                "tla.reduce, which is not supported yet; pass a scalar instead",
            )
        return VectorSSA(
            _tla_ops_gen.full(_coerce_type(desc), src_value, mask=mask_value, loc=loc)
        )

    resolved = _resolve_bound_value(value)
    # Computed once for both the runtime-scalar and the literal paths below.
    context = loc.context if loc is not None else mlir_ir.Context.current
    if isinstance(resolved, Numeric):
        if isinstance(resolved.value, mlir_ir.Value):
            # A runtime scalar SSA (kernel scalar argument, tensor element
            # read). The SIMD backend lowers a plain scalar source via AVE
            # broadcast (lowerFull -> VFBroadcastScalarMaskOp, i.e. vdups.z);
            # the value must be the element type or its broadcast carrier, and
            # lowerFull narrows a TLA-typed value onto the carrier. Catch a
            # mismatch here so it surfaces at trace time.
            expected = desc.element_mlir_type(context)
            carrier = _full_scalar_carrier_type(dtype_token, context)
            actual = resolved.value.type
            if actual != expected and actual != carrier:
                _op_error(
                    "full",
                    f"invalid argument 'value' (position 0): runtime scalar "
                    f"type {actual} does not match result element type "
                    f"{expected} (broadcast carrier {carrier}); cast the scalar first",
                )
            return VectorSSA(
                _tla_ops_gen.full(
                    _coerce_type(desc), resolved.value, mask=mask_value, loc=loc
                )
            )
        resolved = resolved.value
    if not isinstance(resolved, (bool, int, float)):
        _op_error(
            "full",
            "value must be a Python scalar literal, host Numeric, a runtime "
            "scalar SSA, or one-lane vector fragment",
        )
    scalar_value = int(resolved) if isinstance(resolved, bool) else resolved
    scalar = _full_scalar_literal("full", dtype_token, scalar_value, context, loc=loc)
    result = _tla_ops_gen.full(_coerce_type(desc), scalar, mask=mask_value, loc=loc)
    return VectorSSA(result)


def _full_vector_ssa_descriptor(dtype_token: str) -> TlaVectorSSATypeDescriptor:
    token = str(dtype_token).strip().lower()
    # dtype_size_bytes("i1")==1 matches AscendC TypeGet<bool> → vector_s8 (b8 / 256 lanes).
    element_bytes = dtype_size_bytes(token)
    lanes = _vector_lane_count(element_bytes)
    return TlaVectorSSATypeDescriptor(lanes, token)


def _arange_integer_start_value(
    op_name: str,
    base: Any,
    element_type: mlir_ir.Type,
    *,
    require_constant: bool = False,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    """Materialize the integral ``arange`` start (base) as ``element_type`` SSA."""
    const = _const_int_value(base)
    if const is not None:
        return _scalar_constant_for_element_type(op_name, const, element_type, loc=loc)
    resolved = _resolve_bound_value(base)
    if isinstance(resolved, float) or (
        isinstance(resolved, Numeric)
        and type(resolved).is_float
        and isinstance(resolved.value, (int, float))
    ):
        value = resolved if isinstance(resolved, float) else resolved.value
        if isinstance(value, float) and not value.is_integer():
            _op_error(
                op_name,
                f"base must be an integer value; got {value!r}",
            )
        return _scalar_constant_for_element_type(
            op_name, int(value), element_type, loc=loc
        )
    if require_constant:
        # Floating-point results are built from a constant base in the float
        # domain; a runtime base would be narrowed to the helper integer type
        # without a range check.
        _op_error(
            op_name,
            "base must be a compile-time integer constant for floating-point dtype",
        )
    if isinstance(resolved, mlir_ir.Value):
        if resolved.type == element_type:
            return resolved
        if isinstance(resolved.type, mlir_ir.IndexType):
            return mlir_ir.Operation.create(
                "arith.index_cast",
                operands=[resolved],
                results=[element_type],
                loc=loc,
            ).results[0]
        _op_error(op_name, "base must be an integer literal or index SSA value")
    if (
        isinstance(resolved, Numeric)
        and type(resolved).is_integer
        and type(resolved).signed
    ):
        index_value = _runtime._coerce_index_value(resolved)
        return mlir_ir.Operation.create(
            "arith.index_cast",
            operands=[index_value],
            results=[element_type],
            loc=loc,
        ).results[0]
    _op_error(op_name, "base must be an integer literal or index SSA value")


@dsl_user_op
def arange(
    base: bool | int | float | Numeric = 0,
    *,
    order: str = "increase",
    dtype: type[Numeric],
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Data Fill

    Description:
        Create a monotonically increasing or decreasing 1-D vector SSA (`base` + `order`).

        Parameters:
        - `base` (`bool | int | float | Numeric`): Base offset. Optional, default `0`.
        - `order` (`str`): `'increase'` ascending or `'decrease'` descending. Optional, default `'increase'`.
        - `dtype` (`type[Numeric]`): Vector element type. Required. Supported types:
          `Int8`, `Int16`, `Int32`, `Int64`, `Float16`, `Float32`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; `order` supports only `increase` / `decrease`.
        - `base` must be an integer value. With integer dtypes a lane index that
          overflows wraps around (for `Int8`, lane 128 holds `-128`). With
          floating-point dtypes `base` must be a compile-time integer constant; it
          is rounded into the result type and added in the float domain, so the
          sequence does not wrap. Floating-point indices are only exact while they
          are representable.

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
    # A floating-point result is built from an integral lane sequence, so its
    # start operand keeps the matching integer width (f16 -> i16, f32 -> i32)
    # and must be a compile-time constant: a runtime base would be narrowed to
    # that integer type without a range check.
    if mlir_ir.F16Type.isinstance(element_type):
        start_element_type = mlir_ir.IntegerType.get_signless(16)
        require_constant = True
    elif mlir_ir.F32Type.isinstance(element_type):
        start_element_type = mlir_ir.IntegerType.get_signless(32)
        require_constant = True
    else:
        start_element_type = element_type
        require_constant = False
    start_value = _arange_integer_start_value(
        op_name,
        base,
        start_element_type,
        require_constant=require_constant,
        loc=loc,
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
    if is_bool_element_type(lhs_desc.element_type):
        _op_error(
            op_name,
            "Bool VectorSSA is move-only (load/store); arithmetic is not supported",
        )
    if lhs_desc.element_type == "hif8":
        _op_error(
            op_name,
            "HiFloat8 arithmetic requires an explicit numeric conversion or bitcast",
        )
    if lhs_desc.element_type in _FP4_PAIR_DTYPES:
        _op_error(
            op_name,
            "FP4x2 arithmetic requires an explicit numeric conversion or bitcast",
        )
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


def _runtime_scalar_as_signless_integer(
    op_name: str,
    scalar: Numeric,
    element_type: mlir_ir.Type,
    *,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    """Convert a runtime integer scalar to the signless carrier of ``element_type``.

    Unsigned vector element types keep their unsigned ``!tla.vector`` type, but
    the scalar ABI is signless. A signed scalar widening to the element width
    sign-extends (an ``Int32`` threshold compared with a ``UInt64`` vector
    becomes ``i64``); an unsigned source zero-extends.
    """
    int_ty = mlir_ir.IntegerType(element_type)
    width = int(int_ty.width)
    signless_ty = mlir_ir.IntegerType.get_signless(width)
    value = scalar.ir_value(loc=loc)
    if value.type == signless_ty:
        return value
    if isinstance(value.type, mlir_ir.IndexType):
        return mlir_ir.Operation.create(
            "arith.index_cast", operands=[value], results=[signless_ty], loc=loc
        ).results[0]
    if not mlir_ir.IntegerType.isinstance(value.type):
        _op_error(
            op_name,
            f"invalid scalar type {value.type} for {element_type}; "
            "expected an integer scalar",
        )
    src_int = mlir_ir.IntegerType(value.type)
    src_width = int(src_int.width)
    src_unsigned = bool(src_int.is_unsigned)
    if not src_int.is_signless:
        # arith integer casts are signless-only.
        signless_src = mlir_ir.IntegerType.get_signless(src_width)
        value = mlir_ir.Operation.create(
            "builtin.unrealized_conversion_cast",
            operands=[value],
            results=[signless_src],
            loc=loc,
        ).results[0]
    if src_width == width:
        return value
    op = "arith.trunci"
    if src_width < width:
        op = "arith.extui" if src_unsigned else "arith.extsi"
    return mlir_ir.Operation.create(
        op, operands=[value], results=[signless_ty], loc=loc
    ).results[0]


def _numeric_ir_value_for_element_type(
    op_name: str,
    scalar: Numeric,
    element_type: mlir_ir.Type,
    *,
    normalize_unsigned: bool = False,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    """Materialize scalar for vector–scalar ops via ``to`` + ``ir_value``.

    Host Python numbers keep literal ``arith.constant`` emission (range / fraction
    checks). Dynamic SSA uses ``Numeric.to`` + ``ir_value``.

    ``normalize_unsigned`` emits an unsigned literal as its same-width signless
    carrier, keeping the unsigned range check. ``arith.constant`` and the AVE
    scalar ABI only accept signless integers, while the vector result keeps its
    unsigned ``!tla.vector`` type.
    """
    if isinstance(scalar.value, (bool, int, float)):
        return _scalar_constant_for_element_type(
            op_name,
            scalar.value,
            element_type,
            normalize_unsigned=normalize_unsigned,
            loc=loc,
        )
    if (
        normalize_unsigned
        and mlir_ir.IntegerType.isinstance(element_type)
        and mlir_ir.IntegerType(element_type).is_unsigned
    ):
        return _runtime_scalar_as_signless_integer(
            op_name, scalar, element_type, loc=loc
        )
    dest_cls = Numeric.from_mlir_type(element_type)
    return scalar.to(dest_cls, loc=loc).ir_value(loc=loc)


def _scalar_constant_for_element_type(
    op_name: str,
    scalar: Any,
    element_type: mlir_ir.Type,
    *,
    normalize_unsigned: bool = False,
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
        # Check the logical dtype range before emitting the literal.
        if isinstance(element_type, mlir_ir.IndexType):
            lo, hi = -(2**63), 2**63 - 1
        else:
            int_ty = mlir_ir.IntegerType(element_type)
            width = int(int_ty.width)
            if int_ty.is_unsigned or width == 1:
                lo, hi = 0, 2**width - 1
            else:
                lo, hi = -(2 ** (width - 1)), 2 ** (width - 1) - 1
        if not (lo <= int_value <= hi):
            _op_error(
                op_name,
                f"integer scalar {int_value} out of range for {element_type} "
                f"(valid range [{lo}, {hi}])",
            )
        return _emit_integer_literal(
            element_type, int_value, normalize_unsigned=normalize_unsigned, loc=loc
        )
    if (
        mlir_ir.F16Type.isinstance(element_type)
        or mlir_ir.F32Type.isinstance(element_type)
        or mlir_ir.F64Type.isinstance(element_type)
        or mlir_ir.BF16Type.isinstance(element_type)
        or isinstance(element_type, mlir_ir.Float8E4M3FNType)
        or isinstance(element_type, mlir_ir.Float8E5M2Type)
    ):
        return mlir_ir.Operation.create(
            "arith.constant",
            results=[element_type],
            attributes={"value": _float_literal_attr(element_type, float(resolved))},
            loc=loc,
        ).results[0]
    _op_error(
        op_name, f"unsupported vector element type for scalar literal: {element_type}"
    )


def _full_scalar_carrier_type(
    dtype_token: str, context: mlir_ir.Context
) -> mlir_ir.Type:
    """MLIR type of the scalar operand ``tla.full`` broadcasts for ``dtype_token``.

    Bool/hif8/e8m0/fp4x2 broadcast an i8 byte and unsigned elements broadcast
    their same-width signless twin; every other type broadcasts itself.
    """
    if dtype_token in _FULL_BYTE_CARRIER_DTYPES:
        return mlir_ir.IntegerType.get_signless(8)
    if dtype_token.startswith("u") and dtype_token[1:].isdigit():
        return mlir_ir.IntegerType.get_signless(int(dtype_token[1:]))
    return Numeric.from_dtype_token(dtype_token).mlir_type(context)


def _full_byte_carrier_value(code: int) -> int:
    """Spell an 8-bit code as the signless i8 carrier constant."""
    code &= 0xFF
    return code - 256 if code >= 128 else code


def _full_byte_literal(op_name: str, dtype_token: str, value: Any) -> int:
    """Encode a trace-time scalar into the 8-bit carrier of ``dtype_token``."""
    if dtype_token == "i1":
        if isinstance(value, (bool, int)) and int(value) in (0, 1):
            return int(value)
        _op_error(op_name, f"Bool full requires 0 or 1, got {value!r}")
    if dtype_token in ("f4e2m1x2", "f4e1m2x2"):
        # The packed pair is one pre-encoded byte: two nibbles, low first.
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not -128 <= value <= 255
        ):
            _op_error(
                op_name,
                f"{dtype_token} full requires one packed pair byte in [-128, 255], "
                f"not two independent FP4 values; got {value!r}",
            )
        return _full_byte_carrier_value(value)
    if dtype_token == "hif8":
        # No host hif8 encoder exists: the caller supplies the encoded byte.
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not -128 <= value <= 255
        ):
            _op_error(
                op_name,
                f"hif8 full requires an explicit encoded byte in [-128, 255]; "
                f"encode the value first, got {value!r}",
            )
        return _full_byte_carrier_value(value)
    if dtype_token == "f8e8m0":
        # Byte broadcast only: a shared exponent, never arithmetic.
        return _full_byte_carrier_value(_fill_value_bits(op_name, dtype_token, value))
    _op_error(op_name, f"unsupported byte carrier for {dtype_token}")


def _full_scalar_literal(
    op_name: str,
    dtype_token: str,
    value: Any,
    context: mlir_ir.Context,
    *,
    loc: mlir_ir.Location | None = None,
) -> mlir_ir.Value:
    """Emit the scalar carrier constant for a trace-time ``tla.full`` value."""
    carrier = _full_scalar_carrier_type(dtype_token, context)
    if dtype_token in _FULL_BYTE_CARRIER_DTYPES:
        return _scalar_constant_for_element_type(
            op_name, _full_byte_literal(op_name, dtype_token, value), carrier, loc=loc
        )
    if dtype_token in ("f8e4m3fn", "f8e5m2"):
        # _fill_value_bits rejects out-of-range or inexact encodings (it raises
        # through _op_error) and returns the packed 8-bit code; FloatAttr would
        # round silently, so check the encoding before emitting the attribute.
        encoded = _fill_value_bits(op_name, dtype_token, value)
        if encoded < 0:
            _op_error(op_name, f"cannot encode {value!r} as {dtype_token}")
        return _scalar_constant_for_element_type(op_name, value, carrier, loc=loc)
    if dtype_token.startswith("u") and dtype_token[1:].isdigit():
        # Check the unsigned range, then emit the signless carrier bit pattern.
        width = int(dtype_token[1:])
        if isinstance(value, float):
            if not value.is_integer():
                _op_error(
                    op_name,
                    f"invalid argument 'rhs' (position 1): expected integer scalar "
                    f"for {dtype_token}, got {value!r}",
                )
            value = int(value)
        if not isinstance(value, int) or isinstance(value, bool):
            _op_error(
                op_name,
                f"invalid argument 'rhs' (position 1): expected integer scalar "
                f"for {dtype_token}, got {value!r}",
            )
        if not 0 <= value <= 2**width - 1:
            _op_error(
                op_name,
                f"integer scalar {value} out of range for {dtype_token} "
                f"(valid range [0, {2**width - 1}])",
            )
        if value >= 2 ** (width - 1):
            value -= 2**width
        return _scalar_constant_for_element_type(op_name, value, carrier, loc=loc)
    return _scalar_constant_for_element_type(op_name, value, carrier, loc=loc)


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
    if is_bool_element_type(lhs_desc.element_type):
        _op_error(
            op_name,
            "Bool VectorSSA is move-only (load/store); arithmetic is not supported",
        )
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


_SIGNED_SHIFT_ELEMENT_TYPE = {
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "u8": "i8",
    "u16": "i16",
    "u32": "i32",
    "u64": "i64",
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
    if source_element not in _SIGNED_SHIFT_ELEMENT_TYPE:
        _op_error(
            op_name,
            f"unsupported source element type {source_desc.element_type}; "
            f"supported types are {', '.join(sorted(_SIGNED_SHIFT_ELEMENT_TYPE))}",
        )

    if source_element in ("i64", "u64") and source_desc.valid_lanes != 32:
        _op_error(op_name, "64-bit shifts require a full 256-byte register (32 lanes)")

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
# Packed FP4 pair carriers are byte-sized, so bitwise ops handle them exactly
# like the integer carriers: both nibbles of a lane travel together.
_BITWISE_UNARY_TYPES = _ABS_ELEMENT_TYPES | _FP4_PAIR_DTYPES


def _emit_vector_unary(
    op_name: str,
    emitter: Any,
    operand: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    extra_attrs: dict[str, Any] | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    _require_category(op_name, "operand", operand, "vector_ssa", 0)
    if mask is not None:
        _require_category(op_name, "mask", mask, "mask_ssa", 1)
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    operand_value = _as_value(operand)
    operand_desc = _vector_ssa_type_for_mlir_value(operand_value)
    element_type = operand_desc.element_type
    if is_bool_element_type(element_type):
        _op_error(
            op_name,
            "Bool VectorSSA is move-only (load/store); arithmetic is not supported",
        )
    if op_name in {"exp", "log", "sqrt"}:
        if element_type not in _FLOAT_UNARY_ELEMENT_TYPES:
            _op_error(
                op_name,
                f"tla.{op_name} requires f16 or f32 element type, got {element_type}",
            )
    elif op_name == "relu":
        if element_type not in ("f16", "f32", "i32", "i64"):
            _op_error(
                op_name, f"requires f16, f32, i32, or i64 elements, got {element_type}"
            )
        if element_type == "i64" and operand_desc.valid_lanes != 32:
            _op_error(op_name, "Int64 requires a full 32-lane vector register")
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
        **(extra_attrs or {}),
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
# selects for f16, and tla-vector-region-simt promotes bf16 through f32.
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
    doc="""Directory: SIMD Vector Compute / Basic Arithmetic

Description:
    Exponential of a vector (requires f16/f32), or of a per-thread scalar.

    Parameters:
    - `operand` (`VectorSSA | Numeric`): Source vector register, or a float scalar. Required.
    - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Vector form only. Optional, default `None`.

    Constraints:
    - Must be called inside a `@tla.kernel`-decorated kernel function.
    - Must be called inside `tla.vec.func()`; element type must be f16/f32.
    - Inside a `tla.vec.func(mode="simt")` region a scalar operand emits the
      per-thread `tla.simt_exp`; outside one the scalar form is rejected,
      because the core's own scalar unit has no exponential. Pass a
      `VectorSSA` for the whole-vector form.
    - The scalar form is float-only and takes no `mask`.

    Example:
    ```python
    with tla.vec.func(mode="simd"):
        y = tla.exp(x_reg)
    ```
    """,
)


@dsl_user_op
def arch_sub_exp(
    lhs: VectorSSA,
    rhs: VectorSSA,
    *,
    reg_slot: RegSlot = RegSlot.ZERO,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
        Use the native ExpSub instruction to subtract and then take the natural
        exponential. For each output index `i`, `reg_slot` selects an input
        index `j`. When `j` is valid in both inputs and enabled by the mask,
        the result is

        ```text
        out[i] = exp(fp32(lhs[j]) - fp32(rhs[j]))
        ```

        `fp32(x)` converts `x` to FP32 before subtraction. All other output
        elements are FP32 positive zero (`+0.0`, ZEROING).

        Parameters:
        - `lhs` (`VectorSSA`): FP16 or FP32 input. Required.
        - `rhs` (`VectorSSA`): Input register with the same element type and
          physical width as `lhs`. Required.
        - `reg_slot` (`RegSlot`): Selects the input index `j` for each output
          index `i`. Indices start at zero. For FP16 inputs, `ZERO` uses
          even positions (`j = 2*i`) and `ONE` uses odd positions
          (`j = 2*i + 1`). With `ONE`, the results for input positions 1, 3,
          and 5 go to output positions 0, 1, and 2, respectively. For FP32
          inputs, both values use `j = i`. Defaults to `RegSlot.ZERO`.
          Keyword-only.
        - `mask` (`MaskSSA | None`): Optional mask over input element positions.
          Defaults to `None`, which enables all positions within the valid input
          range. Keyword-only.

        Constraints:
        - Must be called inside a `@tla.kernel` and `tla.vec.func(mode="simd")`.
        - Both inputs must have the same FP16 or FP32 element type. A known
          valid element count limits an input to that many leading positions.
          When both counts are known, use the smaller count.
        - The output contains 64 FP32 elements.
        - ZEROING applies even to nonfinite input values.
          With only the first FP16 input element valid, `RegSlot.ONE`
          returns all zeros.
        - Scalar and SIMT forms, other input types, other register slots,
          and merging masked results are unsupported.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            even = tla.arch.sub_exp(lhs_half, rhs_half)
            odd = tla.arch.sub_exp(
                lhs_half, rhs_half, reg_slot=tla.params.RegSlot.ONE
            )
            result = tla.arch.sub_exp(lhs_float, rhs_float, mask=None)
        ```
    """
    op_name = "arch.sub_exp"
    _require_category(op_name, "lhs", lhs, "vector_ssa", 0)
    _require_category(op_name, "rhs", rhs, "vector_ssa", 1)
    if mask is not None:
        _require_category(op_name, "mask", mask, "mask_ssa", 2)
    if reg_slot not in (RegSlot.ZERO, RegSlot.ONE):
        _op_error(op_name, "reg_slot must be RegSlot.ZERO or RegSlot.ONE")
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    if _runtime._in_simt_vec_func():
        _op_error(op_name, "requires tla.vec.func(mode='simd')")
    lhs_value = _as_value(lhs)
    rhs_value = _as_value(rhs)
    lhs_desc = _vector_ssa_type_for_mlir_value(lhs_value)
    rhs_desc = _vector_ssa_type_for_mlir_value(rhs_value)
    if (
        lhs_desc.element_type not in ("f16", "f32")
        or rhs_desc.element_type != lhs_desc.element_type
    ):
        _op_error(op_name, "lhs and rhs must have matching f16 or f32 element types")
    lanes = 128 if lhs_desc.element_type == "f16" else 64
    if mask is None:
        input_dtype = mlir_ir.Type.parse(
            lhs_desc.element_type, context=lhs_value.type.context
        )
        mask = create_mask(pattern="ALL", dtype=input_dtype, loc=loc)
    mask_value = _as_value(mask)
    mask_desc = _mask_ssa_type_for_mlir_value(mask_value)
    if mask_desc.physical_lanes != lanes:
        _op_error(op_name, f"mask must be a {lanes}-lane input-domain MaskSSA")
    context = lhs_value.type.context
    result_type = _coerce_type(TlaVectorSSATypeDescriptor(64, "f32"))
    part_attr = mlir_ir.IntegerAttr.get(
        mlir_ir.IntegerType.get_signless(32, context=context),
        0 if reg_slot is RegSlot.ZERO else 1,
    )
    result = _tla_ops_gen.exp_sub(
        result_type, lhs_value, rhs_value, mask_value, part_attr, loc=loc
    )
    return VectorSSA(result)


@dsl_user_op
def log(
    operand: VectorSSA | Numeric,
    *,
    mask: MaskSSA | None = None,
    mode: LogAlgo = LogAlgo.FLUSH_TO_ZERO,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA | Numeric:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
        Logarithm of a vector (requires f16/f32), or of a per-thread scalar.

        Parameters:
        - `operand` (`VectorSSA | Numeric`): Source vector register, or a float scalar. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Vector form only. Optional, default `None`.
        - `mode` (`tla.params.LogAlgo`): Subnormal handling of the vector
          form. `tla.params.LogAlgo.FLUSH_TO_ZERO` (default) reuses the AVE
          `vln` path: on Ascend 950 it flushes positive f32 subnormal inputs
          (`0 < x < 2^-126`) to zero and returns `-inf`, while the f16 `vln`
          path already computes positive subnormal inputs. Negative inputs
          return NaN and `+/-0.0` returns `-inf` in both modes.
          `tla.params.LogAlgo.SUBNORMAL` always runs the software sequence, so
          positive subnormal inputs are computed for both dtypes.
          Optional, default `tla.params.LogAlgo.FLUSH_TO_ZERO`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; element type must be f16/f32.
        - Inside a `tla.vec.func(mode="simt")` region a scalar operand emits the
          per-thread `tla.simt_log`; outside one the scalar form is rejected,
          because the core's own scalar unit has no logarithm. Pass a
          `VectorSSA` for the whole-vector form.
        - The scalar form is float-only and takes no `mask`.
        - `mode=SUBNORMAL` is available only for the SIMD vector form: the
          per-thread scalar form and `tla.vec.func(mode="simt")` reject it.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            y = tla.log(x_reg)
            z = tla.log(x_reg, mode=tla.params.LogAlgo.SUBNORMAL)
        ```
    """
    if not isinstance(mode, LogAlgo):
        _op_error(
            "log",
            f"invalid argument 'mode': expected LogAlgo, got {_type_name(mode)}",
        )
    if not isinstance(operand, VectorSSA):
        if mask is not None:
            _op_error("log", "mask is not supported for the per-thread scalar form")
        if mode is not LogAlgo.FLUSH_TO_ZERO:
            _op_error(
                "log",
                "mode=LogAlgo.SUBNORMAL is only supported for the SIMD vector form",
            )
        return _emit_simt_unary("log", operand, loc=loc)
    if mode is LogAlgo.SUBNORMAL and _runtime._in_simt_vec_func():
        _op_error(
            "log",
            "mode=LogAlgo.SUBNORMAL is only supported in tla.vec.func(mode='simd')",
        )
    extra_attrs = None
    if mode is LogAlgo.SUBNORMAL:
        ctx = loc.context if loc is not None else mlir_ir.Context.current
        extra_attrs = {
            "log_algo": mlir_ir.Attribute.parse("#tla.log_algo<subnormal>", context=ctx)
        }
    return _emit_vector_unary(
        "log",
        _tla_ops_gen.log,
        operand,
        mask=mask,
        extra_attrs=extra_attrs,
        loc=loc,
    )


sqrt = _make_scalar_aware_unary_op(
    "sqrt",
    doc="""Directory: SIMD Vector Compute / Basic Arithmetic

Description:
    Square root of a vector or of a scalar.

    Given a `VectorSSA` this is the element-wise AVE op (requires f16/f32).
    Given a float `Numeric` it is a scalar square root on the core's own scalar
    unit, available anywhere a scalar is: a `tla.cube` region, a SIMD or SIMT
    `tla.vec.func`, or the kernel body. `x.sqrt()` on a float `Numeric` is the
    same operation.

    Parameters:
    - `operand` (`VectorSSA | Numeric`): Source vector register, or a float scalar. Required.
    - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Vector form only. Optional, default `None`.

    Constraints:
    - Must be called inside a `@tla.kernel`-decorated kernel function.
    - The vector form must be called inside `tla.vec.func()`; element type must be f16/f32.
    - Inside a `tla.vec.func(mode="simt")` region a scalar operand emits the
      per-thread `tla.simt_sqrt`, which also accepts f16; outside one it emits
      `math.sqrt` and requires `tla.Float32` (convert first with
      `.to(tla.Float32)`).
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
    doc="""Directory: SIMD Vector Compute / Basic Arithmetic

Description:
    Absolute value of a vector, or of a per-thread scalar.

    Parameters:
    - `operand` (`VectorSSA | Numeric`): Source vector register, or a float scalar. Required.
    - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Vector form only. Optional, default `None`.

    Constraints:
    - Must be called inside a `@tla.kernel`-decorated kernel function.
    - Must be called inside `tla.vec.func()`.
    - Inside a `tla.vec.func(mode="simt")` region a scalar operand emits the
      per-thread `tla.simt_abs`; outside one `tla.abs` on a scalar is
      rejected. The Python builtin `abs(x)` on a `Numeric` works in either
      place: `tla.simt_abs` inside a SIMT region, `math.absf` elsewhere.
    - The scalar form is float-only and takes no `mask`.

    Example:
    ```python
    with tla.vec.func(mode="simd"):
        y = tla.abs(x_reg)
    ```
    """,
)
neg = _make_unary_op(
    "neg",
    doc="""Directory: SIMD Vector Compute / Basic Arithmetic

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
        "u8",
        "u16",
        "u32",
        "f16",
        "bf16",
        "f32",
    }
)


# 64-bit sources are deliberately unsupported: unpack of an i64/u64 source
# would need a 128-bit destination. 64-bit *destinations* (i32->i64, u32->u64)
# are supported: Ascend 950 has no native vunpack for that width, so the AIV
# stubs reuse AscendC's composition (sign/zero fill + Interleave).
# Pack produces unsigned elements with half the source bit width.
# Signed and unsigned source types share the same destination type.
_VECTOR_PACK_DST_FOR_SRC = {"i16": "u8", "u16": "u8", "i32": "u16", "u32": "u16"}
_VECTOR_UNPACK_DST_FOR_SRC = {
    "i8": "i16",
    "u8": "u16",
    "i16": "i32",
    "u16": "u32",
    "i32": "i64",
    "u32": "u64",
}


@dsl_user_op
def pack(
    src: VectorSSA | MaskSSA,
    *,
    part: RegPart,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA | MaskSSA:
    """Directory: SIMD Vector Compute / Data Rearrange

    Description:
        Pack a predicate value, or truncate integer vector elements to their
        low 8 or 16 bits and place them in the selected destination half.
        The other destination half of a vector is unspecified. The
        destination element type is derived from the source: the result is
        always unsigned, with half the source bit width.

        Parameters:
        - `src` (`VectorSSA | MaskSSA`): Source register value. Required.
        - `part` (`tla.RegPart`): Destination register half: `LOW` (low) or `HIGH` (high). Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func(mode="simd")`.
        - Mask inputs support `mask<64> -> mask<128>` and `mask<128> -> mask<256>`.
        - Vector pairs (destination <- source): `UInt8 <- Int16/UInt16`, `UInt16 <- Int32/UInt32`. 64-bit sources (`Int64`/`UInt64`) are not supported.
        - Vector inputs must fill one 256Byte register (RegTraitNumOne); the result has twice as many lanes. RegTraitNumTwo is not representable by `VectorSSA`; CANN permits only LOW for that trait.
        - Vector form supports Ascend 950 (`--npu-arch 3510`) only. `LOW/HIGH` map to CANN `HighLowPart::LOWEST/HIGHEST`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            packed = tla.pack(src, part=tla.RegPart.LOW)
            narrow = tla.pack(src_reg, part=tla.RegPart.HIGH)
        ```
    """
    _require_categories("pack", "src", src, ("vector_ssa", "mask_ssa"), 0)
    if not isinstance(part, RegPart):
        _op_error(
            "pack",
            "invalid argument 'part': expected tla.RegPart.LOW or "
            "tla.RegPart.HIGH, "
            f"got {_type_name(part)}",
        )
    _require_frontend_state("pack")
    _runtime._require_enclosing_region("pack", "vec.func")
    if _runtime._in_simt_vec_func():
        _op_error("pack", "is only supported in tla.vec.func(mode='simd')")

    src_value = _as_value(src)
    context = src_value.type.context
    part_attr = mlir_ir.Attribute.parse(f"#tla.reg_part<{part.value}>", context=context)
    if _category(src) == "vector_ssa":
        src_desc = _vector_ssa_type_for_mlir_value(src_value)
        dst_token = _VECTOR_PACK_DST_FOR_SRC.get(src_desc.element_type)
        if dst_token is None:
            _op_error(
                "pack",
                f"unsupported vector element type {src_desc.element_type}; "
                "expected a 16/32-bit integer source",
            )
        if (
            src_desc.valid_lanes
            != _full_vector_ssa_descriptor(src_desc.element_type).valid_lanes
        ):
            _op_error(
                "pack", "requires full 256-byte vector registers (RegTraitNumOne)"
            )
        result_desc = _full_vector_ssa_descriptor(dst_token)
        result = _tla_ops_gen.pack(
            result_desc.to_mlir_type(context), src_value, part_attr, loc=loc
        )
        return VectorSSA(result)

    src_desc = _mask_ssa_type_for_mlir_value(src_value)
    if src_desc.physical_lanes not in (64, 128):
        _op_error(
            "pack",
            f"unsupported mask width {src_desc.physical_lanes}; "
            "supported source widths are 64 and 128",
        )
    result_desc = TlaMaskSSATypeDescriptor(src_desc.physical_lanes * 2)
    result = _tla_ops_gen.pack(
        result_desc.to_mlir_type(context), src_value, part_attr, loc=loc
    )
    return MaskSSA(result)


@dsl_user_op
def unpack(
    src: VectorSSA,
    *,
    part: RegPart,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Data Rearrange

    Description:
        Extend integer elements from the selected source register half.
        Signed types sign-extend; unsigned types zero-extend. The
        destination element type is derived from the source: the result
        keeps the source signedness, with twice the source bit width.

        Parameters:
        - `src` (`VectorSSA`): Narrow integer source register. Required.
        - `part` (`tla.RegPart`): Source half to read: `LOW` or `HIGH`. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function and `tla.vec.func(mode="simd")`.
        - Pairs (destination <- source): `Int16 <- Int8`, `UInt16 <- UInt8`, `Int32 <- Int16`, `UInt32 <- UInt16`, `Int64 <- Int32`, `UInt64 <- UInt32`. 64-bit sources (`Int64`/`UInt64`) are not supported; 64-bit destinations reuse AscendC's composition (no native vunpack for that width).
        - Source and destination signedness must match. The result has half as many lanes.
        - Inputs must fill one 256Byte register (RegTraitNumOne). RegTraitNumTwo is not representable by `VectorSSA`; CANN permits only LOW for that trait.
        - Supports Ascend 950 (`--npu-arch 3510`) only. `LOW/HIGH` map to CANN `HighLowPart::LOWEST/HIGHEST`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            wide = tla.unpack(src_reg, part=tla.RegPart.HIGH)
            wide64 = tla.unpack(src32_reg, part=tla.RegPart.LOW)  # Int32 -> Int64, sign extension
        ```
    """
    _require_category("unpack", "src", src, "vector_ssa", 0)
    if not isinstance(part, RegPart):
        _op_error("unpack", "expected tla.RegPart.LOW or tla.RegPart.HIGH")
    _require_frontend_state("unpack")
    _runtime._require_enclosing_region("unpack", "vec.func")
    if _runtime._in_simt_vec_func():
        _op_error("unpack", "is only supported in tla.vec.func(mode='simd')")

    src_value = _as_value(src)
    src_desc = _vector_ssa_type_for_mlir_value(src_value)
    dst_token = _VECTOR_UNPACK_DST_FOR_SRC.get(src_desc.element_type)
    if dst_token is None:
        _op_error(
            "unpack",
            f"unsupported vector element type {src_desc.element_type}; "
            "expected an 8/16/32-bit integer source",
        )
    if (
        src_desc.valid_lanes
        != _full_vector_ssa_descriptor(src_desc.element_type).valid_lanes
    ):
        _op_error("unpack", "requires full 256-byte vector registers (RegTraitNumOne)")
    result_desc = _full_vector_ssa_descriptor(dst_token)
    context = src_value.type.context
    part_attr = mlir_ir.Attribute.parse(f"#tla.reg_part<{part.value}>", context=context)
    result = _tla_ops_gen.unpack(
        result_desc.to_mlir_type(context), src_value, part_attr, loc=loc
    )
    return VectorSSA(result)


def _prepare_b32_pair(op_name, lhs, rhs, mask, loc):
    _runtime._require_kernel_simd_region(op_name)
    _require_category(op_name, "lhs", lhs, "vector_ssa", 0)
    _require_category(op_name, "rhs", rhs, "vector_ssa", 1)
    lhs_value, rhs_value = _as_value(lhs), _as_value(rhs)
    desc = _vector_ssa_type_for_mlir_value(lhs_value)
    if str(desc.element_type).lower() not in ("i32", "u32") or desc.valid_lanes != 64:
        _op_error(op_name, "requires full 64-lane i32/u32 registers")
    if lhs_value.type != rhs_value.type:
        _op_error(op_name, "lhs and rhs must have the same VectorSSA type")
    if mask is None:
        mask = create_mask(pattern="ALL", dtype=UInt32, loc=loc)
    _require_category(op_name, "mask", mask, "mask_ssa", 2)
    mask_value = _as_value(mask)
    _require_mask_matches_vector(op_name, mask_value, lhs_value)
    return lhs_value, rhs_value, mask_value


def _prepare_b32_carry_pair(op_name, lhs, rhs, carry_in, mask, loc):
    lhs_value, rhs_value, mask_value = _prepare_b32_pair(op_name, lhs, rhs, mask, loc)
    _require_category(op_name, "carry_in", carry_in, "mask_ssa", 2)
    carry_value = _as_value(carry_in)
    _require_mask_matches_vector(op_name, carry_value, lhs_value)
    return lhs_value, rhs_value, carry_value, mask_value


@dsl_user_op
def addc(
    lhs: VectorSSA,
    rhs: VectorSSA,
    carry_in: MaskSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> tuple[VectorSSA, MaskSSA]:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
    Add the operand bit patterns and the carry-in per active lane. Return `(sum, carry_out)`
    with the low 32 bits of the sum and the carry bit.

    Parameters:
    - `lhs` (`VectorSSA`): Left-hand operand. Required.
    - `rhs` (`VectorSSA`): Right-hand operand. Required.
    - `carry_in` (`MaskSSA`): Per-lane input carry predicate. Required.
    - `mask` (`MaskSSA | None`): Execution mask. Defaults to `None`, which enables all lanes.

    Constraints:
    - Call inside `tla.vec.func(mode="simd")` on Ascend 950.
    - The operands must be 64-lane vectors of the same `i32` or `u32` type. An explicit `mask` must have 64 lanes.
    - `carry_in` and `carry_out` are 64-lane `MaskSSA` predicates (`!tla.mask<64>`) with one bit per lane.
    - A `carry_in` bit of `1` means an incoming carry and adds one to the sum. A bit of `0` means no incoming carry.
    - A `carry_out` bit of `1` means the sum carries beyond the low 32 bits. A bit of `0` means no outgoing carry.
    - `carry_out` is undefined for lanes disabled by `mask`.

    Example:
    ```python
    with tla.vec.func(mode="simd"):
        # ALLF sets every input bit to 0 for no incoming carry.
        carry_in = tla.create_mask(pattern=tla.mask.ALLF, dtype=tla.UInt32)
        total, carry_out = tla.addc(lhs, rhs, carry_in)
    ```
    """
    lhs_value, rhs_value, carry_value, mask_value = _prepare_b32_carry_pair(
        "addc", lhs, rhs, carry_in, mask, loc
    )

    sum_value, carry_out = _tla_ops_gen.addc(
        lhs_value.type,
        carry_value.type,
        lhs_value,
        rhs_value,
        carry_value,
        mask_value,
        loc=loc,
    )
    return VectorSSA(sum_value), MaskSSA(carry_out)


@dsl_user_op
def subc(
    lhs: VectorSSA,
    rhs: VectorSSA,
    carry_in: MaskSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> tuple[VectorSSA, MaskSSA]:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
    Compute `lhs - rhs - (1 - carry_in)` per active lane. Return `(diff, carry_out)`
    with the low 32 bits of the difference and the carry bit.

    Parameters:
    - `lhs` (`VectorSSA`): Left-hand operand (minuend). Required.
    - `rhs` (`VectorSSA`): Right-hand operand (subtrahend). Required.
    - `carry_in` (`MaskSSA`): Per-lane input carry predicate. Required.
    - `mask` (`MaskSSA | None`): Execution mask. Defaults to `None`, which enables all lanes.

    Constraints:
    - Call inside `tla.vec.func(mode="simd")` on Ascend 950.
    - The operands must be 64-lane vectors of the same `i32` or `u32` type. An explicit `mask` must have 64 lanes.
    - `carry_in` and `carry_out` are 64-lane `MaskSSA` predicates (`!tla.mask<64>`) with one bit per lane.
    - A `carry_in` bit of `1` means no incoming borrow. A bit of `0` means an incoming borrow and subtracts one more.
    - A `carry_out` bit of `1` means no outgoing borrow. A bit of `0` means an outgoing borrow.
    - `carry_out` is undefined for lanes disabled by `mask`.

    Example:
    ```python
    with tla.vec.func(mode="simd"):
        # ALL sets every input bit to 1 for no incoming borrow.
        carry_in = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt32)
        diff, carry_out = tla.subc(lhs, rhs, carry_in)
    ```
    """
    lhs_value, rhs_value, carry_value, mask_value = _prepare_b32_carry_pair(
        "subc", lhs, rhs, carry_in, mask, loc
    )
    diff, carry_out = _tla_ops_gen.subc(
        lhs_value.type,
        carry_value.type,
        lhs_value,
        rhs_value,
        carry_value,
        mask_value,
        loc=loc,
    )
    return VectorSSA(diff), MaskSSA(carry_out)


@dsl_user_op
def mull(
    lhs: VectorSSA,
    rhs: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> tuple[VectorSSA, VectorSSA]:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
    Return the low and high 32-bit halves `(low, high)` of each active lane's product.
    The multiplication is signed for `i32` and unsigned for `u32`.

    Parameters:
    - `lhs` (`VectorSSA`): Left-hand operand. Required.
    - `rhs` (`VectorSSA`): Right-hand operand. Required.
    - `mask` (`MaskSSA | None`): Execution mask. Defaults to `None`, which enables all lanes.

    Constraints:
    - Call inside `tla.vec.func(mode="simd")` on Ascend 950.
    - The operands must be 64-lane vectors of the same `i32` or `u32` type. An explicit `mask` must have 64 lanes.

    Example:
    ```python
    with tla.vec.func(mode="simd"):
        low, high = tla.mull(lhs, rhs)
    ```
    """
    lhs_value, rhs_value, mask_value = _prepare_b32_pair("mull", lhs, rhs, mask, loc)
    lo, hi = _tla_ops_gen.mull(
        lhs_value.type,
        lhs_value.type,
        lhs_value,
        rhs_value,
        mask_value,
        loc=loc,
    )
    return VectorSSA(lo), VectorSSA(hi)


@dsl_user_op
def interleave(
    src0: VectorSSA | MaskSSA,
    src1: VectorSSA | MaskSSA,
    *,
    loc: mlir_ir.Location | None = None,
) -> tuple[VectorSSA, VectorSSA] | tuple[MaskSSA, MaskSSA]:
    """Directory: SIMD Vector Compute / Data Rearrange
    Description:
        Interleave two vector registers or predicate masks lane-wise.

        Parameters:
        - `src0` (`VectorSSA` or `MaskSSA`): Even-lane input. Required.
        - `src1` (`VectorSSA` or `MaskSSA`): Odd-lane input of the same category and type. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; inputs must match type. MaskSSA supports only mask<64>, mask<128>, and mask<256>.
        - VectorSSA supports i8/i16/i32, u8/u16/u32, and f16/bf16/f32.
        - Use MaskSSA for predicate rearrangement. The b64 RegTraitNumTwo form is unsupported.

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
    """Directory: SIMD Vector Compute / Data Rearrange

    Description:
        Concatenate two vector registers or predicate masks, then return
        the even-position and odd-position lanes as two registers.

        Parameters:
        - `src0` (`VectorSSA` or `MaskSSA`): First half of the concatenated input. Required.
        - `src1` (`VectorSSA` or `MaskSSA`): Second half, of the same category and type. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; both vectors must match element type and lane count.
        - VectorSSA supports i8/i16/i32, u8/u16/u32, and f16/bf16/f32.
        - Use MaskSSA for predicate rearrangement. The b64 RegTraitNumTwo form is unsupported.
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
    """Directory: SIMD Vector Compute / Logical Compute
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
        - Source elements may be signed or unsigned 8/16/32/64-bit integers.
          Each 64-bit vector operand must fill one 256-byte register (32 lanes).
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
    """Directory: SIMD Vector Compute / Logical Compute
    Description:
        Element-wise right shift of a vector by per-lane or scalar shift amounts.

        Signed source vectors use arithmetic right shift. Unsigned source vectors
        use logical right shift.

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
        - Source elements may be signed or unsigned 8/16/32/64-bit integers.
          Each 64-bit vector operand must fill one 256-byte register (32 lanes).
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
    """Directory: SIMD Vector Compute / Logical Compute
    Description:
        Element-wise bitwise/logical not (Mask or Vector).

        Parameters:
        - `operand` (`VectorSSA | MaskSSA`): Source operand for bitwise/logical not. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.
        - `f4e2m1x2` / `f4e1m2x2` vectors carry one FP4 pair per lane; a `mask` selects whole pairs.

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
    """Directory: SIMD Vector Compute / Basic Arithmetic
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
    """Directory: SIMD Vector Compute / Basic Arithmetic

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
def arch_inplace_mul_add(
    dst: VectorSSA,
    multiplier: VectorSSA,
    addend: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
        Return a new register containing `dst * multiplier + addend` in
        each active lane, with a single rounding to the input element type.
        The first input supplies the old destination value and remains available.

        Parameters:
        - `dst` (`VectorSSA`): First multiplicand. Required.
        - `multiplier` (`VectorSSA`): Second multiplicand. Required.
        - `addend` (`VectorSSA`): Value added to the product. Required.
        - `mask` (`MaskSSA | None`): Optional keyword-only execution mask.
          Defaults to all lanes. Inactive result lanes are zero (ZEROING).

        Constraints:
        - Call inside `@tla.kernel` and `tla.vec.func(mode="simd")`.
        - All three input registers must have the same element type and valid
          lane count. Supported types are `Float16`, `BFloat16` and `Float32`.
        - When the result is used as the next `dst`, a previously masked-off
          lane supplies zero as the first multiplicand. To retain its old value,
          use `tla.where(mask, result, dst)` as the next `dst`.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            updated = tla.arch.inplace_mul_add(previous, scale, current, mask=active)
            previous = tla.where(active, updated, previous)
            out_tile.store(previous)
        ```
    """
    name = "arch.inplace_mul_add"
    _runtime._require_kernel_simd_region(name)
    for position, (arg, value) in enumerate(
        (("dst", dst), ("multiplier", multiplier), ("addend", addend))
    ):
        _require_category(name, arg, value, "vector_ssa", position)
    dst_value, multiplier_value, addend_value = map(
        _as_value, (dst, multiplier, addend)
    )
    element_type = _vector_ssa_type_for_mlir_value(dst_value).element_type
    if element_type not in ("f16", "bf16", "f32"):
        _op_error(name, "expected Float16, BFloat16 or Float32 registers")
    if multiplier_value.type != dst_value.type or addend_value.type != dst_value.type:
        _op_error(name, "all inputs must have the same type and valid lane count")
    mask_value = None
    if mask is not None:
        _require_category(name, "mask", mask, "mask_ssa", 3)
        mask_value = _as_value(mask)
        _require_mask_matches_element_type(name, mask_value, element_type)
    return VectorSSA(
        _tla_ops_gen.inplace_mul_add(
            dst_value.type,
            dst_value,
            multiplier_value,
            addend_value,
            mask=mask_value,
            loc=loc,
        )
    )


@dsl_user_op
def mul(
    lhs: VectorSSA | Numeric | bool | int | float,
    rhs: VectorSSA | Numeric | bool | int | float,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Basic Arithmetic

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
def arch_axpy(
    x: VectorSSA,
    y: VectorSSA,
    alpha: Numeric | float | int,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
        Compute `alpha * x + y` using Ascend C Reg Axpy and return a full
        register of the input element type. Lanes disabled by the mask or
        outside a known valid input prefix are zeroed.

        Parameters:
        - `x` (`VectorSSA`): Source register multiplied by `alpha`. Required.
        - `y` (`VectorSSA`): Addend register with the same element type as `x`. Required.
        - `alpha` (`Numeric | float | int`): Shared scalar multiplier. A typed
          scalar must match the vector element type. Python `float` and `int`
          literals are accepted for floating-point and integer registers,
          respectively. Required.
        - `mask` (`MaskSSA | None`): Input-domain mask. Defaults to `None`,
          which enables all lanes within the known valid input range.

        Constraints:
        - Call inside a `@tla.kernel` and `tla.vec.func(mode="simd")`.
        - Supports matching FP16, FP32, Int64, and UInt64 inputs, with 128, 64,
          32, and 32 mask lanes, respectively.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            result = tla.arch.axpy(x, y, 0.5, mask=mask)
        ```
    """
    op_name = "arch.axpy"
    _runtime._require_kernel_simd_region(op_name)
    _require_category(op_name, "x", x, "vector_ssa", 0)
    _require_category(op_name, "y", y, "vector_ssa", 1)
    x_value, y_value = _as_value(x), _as_value(y)
    x_desc = _vector_ssa_type_for_mlir_value(x_value)
    dtype = x_desc.element_type
    if (
        dtype not in ("f16", "f32", "i64", "u64")
        or _vector_ssa_type_for_mlir_value(y_value).element_type != dtype
    ):
        _op_error(op_name, "x and y must have matching f16, f32, i64, or u64 elements")

    resolved_alpha = _resolve_bound_value(alpha)
    numeric = _as_vector_scalar_numeric(resolved_alpha)
    literal_type = float if dtype in ("f16", "f32") else int
    if (
        numeric is None
        or isinstance(resolved_alpha, bool)
        or (not isinstance(resolved_alpha, literal_type) and numeric.dtype != dtype)
    ):
        _op_error(op_name, f"scalar must match the input element type {dtype}")
    element_type = x_desc.element_mlir_type(x_value.type.context)
    alpha_value = _numeric_ir_value_for_element_type(
        op_name, numeric, element_type, normalize_unsigned=True, loc=loc
    )

    if mask is None:
        mask = create_mask(pattern="ALL", dtype=element_type, loc=loc)
    _require_category(op_name, "mask", mask, "mask_ssa", 3)
    mask_value = _as_value(mask)
    _require_mask_matches_element_type(op_name, mask_value, dtype)
    result_type = _coerce_type(
        TlaVectorSSATypeDescriptor(256 // dtype_size_bytes(dtype), dtype)
    )
    return VectorSSA(
        _tla_ops_gen.axpy(
            result_type, x_value, y_value, alpha_value, mask_value, loc=loc
        )
    )


@dsl_user_op
def arch_sub_abs(
    lhs: VectorSSA,
    rhs: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
        Compute `abs(lhs - rhs)` and return a full register of the input
        element type. Lanes disabled by the mask or outside a known valid
        input prefix are zeroed.

        Parameters:
        - `lhs` (`VectorSSA`): Left input register. Required.
        - `rhs` (`VectorSSA`): Right input register with the same element type. Required.
        - `mask` (`MaskSSA | None`): Input-domain mask. Defaults to `None`,
          which enables all lanes within the known valid input range.

        Constraints:
        - Call inside a `@tla.kernel` and `tla.vec.func(mode="simd")`.
        - Supports matching FP16, FP32, and Int64 inputs, with 128, 64, and 32
          mask lanes, respectively.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            result = tla.arch.sub_abs(lhs, rhs, mask=mask)
        ```
    """
    op_name = "arch.sub_abs"
    _runtime._require_kernel_simd_region(op_name)
    _require_category(op_name, "lhs", lhs, "vector_ssa", 0)
    _require_category(op_name, "rhs", rhs, "vector_ssa", 1)
    lhs_value, rhs_value = _as_value(lhs), _as_value(rhs)
    lhs_desc = _vector_ssa_type_for_mlir_value(lhs_value)
    dtype = lhs_desc.element_type
    if (
        dtype not in ("f16", "f32", "i64")
        or _vector_ssa_type_for_mlir_value(rhs_value).element_type != dtype
    ):
        _op_error(
            op_name,
            "vector operands must have matching elements of type f16, f32, or i64",
        )

    if mask is None:
        element_type = lhs_desc.element_mlir_type(lhs_value.type.context)
        mask = create_mask(pattern="ALL", dtype=element_type, loc=loc)
    _require_category(op_name, "mask", mask, "mask_ssa", 2)
    mask_value = _as_value(mask)
    _require_mask_matches_element_type(op_name, mask_value, dtype)
    result_type = _coerce_type(
        TlaVectorSSATypeDescriptor(256 // dtype_size_bytes(dtype), dtype)
    )
    return VectorSSA(
        _tla_ops_gen.sub_abs(result_type, lhs_value, rhs_value, mask_value, loc=loc)
    )


@dsl_user_op
def arch_muls_cast(
    src: VectorSSA,
    scalar: Float32 | float,
    *,
    reg_slot: RegSlot = RegSlot.ZERO,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
        Multiply an FP32 register by an FP32 scalar using Ascend C Reg
        MulsCast and narrow each selected product to FP16 with `CAST_ROUND`.
        The result is a 128-lane FP16 register.

        Parameters:
        - `src` (`VectorSSA`): FP32 input register. Required.
        - `scalar` (`Float32 | float`): Shared multiplier as an FP32 scalar
          or Python `float` literal. Required.
        - `reg_slot` (`RegSlot`): `ZERO` writes input lane `i` to output lane
          `2*i`. `ONE` writes it to lane `2*i + 1`. Defaults to `ZERO`.
        - `mask` (`MaskSSA | None`): 64-lane input-domain mask. Defaults to
          `None`, which enables all lanes within the known valid input range.

        Constraints:
        - Call inside a `@tla.kernel` and `tla.vec.func(mode="simd")`.
        - Consume the chosen output slot for active input lanes.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            result = tla.arch.muls_cast(src, 0.5, reg_slot=tla.params.RegSlot.ONE)
        ```
    """
    op_name = "arch.muls_cast"
    _runtime._require_kernel_simd_region(op_name)
    _require_category(op_name, "src", src, "vector_ssa", 0)
    src_value = _as_value(src)
    if _vector_ssa_type_for_mlir_value(src_value).element_type != "f32":
        _op_error(op_name, "src must have Float32 elements")
    if reg_slot not in (RegSlot.ZERO, RegSlot.ONE):
        _op_error(op_name, "reg_slot must be RegSlot.ZERO or RegSlot.ONE")
    context = src_value.type.context
    numeric = _as_vector_scalar_numeric(scalar)
    if not isinstance(numeric, Float32):
        _op_error(op_name, "scalar must match the input element type f32")
    element_type = Float32.mlir_type(context)
    scalar_value = _numeric_ir_value_for_element_type(
        op_name, numeric, element_type, loc=loc
    )
    if mask is None:
        mask = create_mask(pattern="ALL", dtype=Float32, loc=loc)
    _require_category(op_name, "mask", mask, "mask_ssa", 3)
    mask_value = _as_value(mask)
    _require_mask_matches_element_type(op_name, mask_value, "f32")
    part = mlir_ir.IntegerAttr.get(
        mlir_ir.IntegerType.get_signless(32, context=context),
        0 if reg_slot is RegSlot.ZERO else 1,
    )
    result_type = _coerce_type(TlaVectorSSATypeDescriptor(128, "f16"))
    return VectorSSA(
        _tla_ops_gen.muls_cast(
            result_type, src_value, scalar_value, mask_value, part, loc=loc
        )
    )


@dsl_user_op
def arch_mul_inplace_add(
    accumulator: VectorSSA,
    lhs: VectorSSA,
    rhs: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
        Return a new register containing `accumulator + lhs * rhs` in each
        active lane. Floating-point results are rounded once after the fused
        operation. Integer results wrap to the input element width. The original
        `accumulator` remains available after the call.

        Parameters:
        - `accumulator` (`VectorSSA`): Register holding the addend. Required.
        - `lhs` (`VectorSSA`): First multiplicand. Required.
        - `rhs` (`VectorSSA`): Second multiplicand. Required.
        - `mask` (`MaskSSA | None`): Execution mask. Inactive result lanes are zero
          (ZEROING). `None` enables all lanes. Optional, default `None`.

        Constraints:
        - Must be called inside `@tla.kernel` and `tla.vec.func(mode="simd")`.
        - All three inputs must have the same element type and valid lane count.
        - Supports `Float16`, `BFloat16`, `Float32`, `Int16`, `UInt16`, `Int32`
          and `UInt32`.
        - When chaining calls, a lane masked off in one call starts from zero
          if a later call selects it again. To keep its previous sum, use
          `tla.where(mask, result, accumulator)` before the next call.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            accumulator = accumulator_tile.load()
            left = left_tile.load()
            right = right_tile.load()
            accumulator = tla.arch.mul_inplace_add(accumulator, left, right, mask=active)
            output_tile.store(accumulator, mask=active)
        ```
    """
    op_name = "arch.mul_inplace_add"
    _runtime._require_kernel_simd_region(op_name)
    for position, (name, value) in enumerate(
        (("accumulator", accumulator), ("lhs", lhs), ("rhs", rhs))
    ):
        _require_category(op_name, name, value, "vector_ssa", position)
    if mask is not None:
        _require_category(op_name, "mask", mask, "mask_ssa", 3)
    accumulator_value = _as_value(accumulator)
    lhs_value = _as_value(lhs)
    rhs_value = _as_value(rhs)
    element_type = _vector_ssa_type_for_mlir_value(accumulator_value).element_type
    _check_mul_inplace_add_element_type_supported(op_name, element_type)
    for name, value in (("lhs", lhs_value), ("rhs", rhs_value)):
        if value.type != accumulator_value.type:
            _op_error(
                op_name,
                f"{name} must have the same type and valid lane count as accumulator",
            )
    mask_value = _as_value(mask) if mask is not None else None
    if mask_value is not None:
        _require_mask_matches_element_type(op_name, mask_value, element_type)
    return VectorSSA(
        _tla_ops_gen.mul_inplace_add(
            accumulator_value.type,
            accumulator_value,
            lhs_value,
            rhs_value,
            mask=mask_value,
            loc=loc,
        )
    )


@dsl_user_op
def relu(
    src: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
        Compute ReLU per lane and return a vector of the same type as `src`.
        FP16, FP32, and Int32 use the native VRELU instruction.
        Int64 uses a signed 64-bit maximum with zero.

        Parameters:
        - `src` (`VectorSSA`): FP16, FP32, Int32, or Int64 input register. Required.
        - `mask` (`MaskSSA | None`): Execution mask in the source lane
          domain. Defaults to `None`, which enables all physical lanes.
          Masked-off lanes are zeroed.

        Constraints:
        - Call inside a `@tla.kernel` and `tla.vec.func(mode="simd")`.
        - Use a 128-lane mask for FP16, a 64-lane mask for FP32 or Int32,
          or a 32-lane mask for Int64. Int64 requires a full 32-lane
          source register. Use a mask to limit execution on partial inputs.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            y = tla.relu(x, mask=mask)
        ```
    """
    _runtime._require_kernel_simd_region("relu")
    return _emit_vector_unary("relu", _tla_ops_gen.relu, src, mask=mask, loc=loc)


@dsl_user_op
def prelu(
    src: VectorSSA,
    slope: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
        Apply PReLU per lane using the native VPRELU instruction. Positive
        values and `+0.0` pass through. Negative values and `-0.0` are
        multiplied by the corresponding slope. A slope of `+inf` produces
        `NaN` for `-0.0`.

        Parameters:
        - `src` (`VectorSSA`): FP16 or FP32 input register. Required.
        - `slope` (`VectorSSA`): Per-lane slopes with the same element type
          as `src`. Required.
        - `mask` (`MaskSSA | None`): Execution mask in the source lane
          domain. Defaults to `None`, which enables all physical lanes.
          Masked-off lanes are zeroed.

        Constraints:
        - Call inside a `@tla.kernel` and `tla.vec.func(mode="simd")`.
        - The result has the same type as `src`. Use a 128-lane mask for FP16
          or a 64-lane mask for FP32. For partial inputs, use a mask that
          enables only valid lanes in both input registers.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            y = tla.prelu(x, slopes, mask=mask)
        ```
    """
    _runtime._require_kernel_simd_region("prelu")
    _require_category("prelu", "src", src, "vector_ssa", 0)
    element_type = _vector_ssa_type_for_mlir_value(_as_value(src)).element_type
    if element_type not in ("f16", "f32"):
        _op_error("prelu", "src must have Float16 or Float32 elements")
    return _emit_vector_binary(
        "prelu", _tla_ops_gen.prelu, src, slope, mask=mask, loc=loc
    )


@dsl_user_op
def leaky_relu(
    src: VectorSSA,
    slope: Float16 | Float32 | float,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
        Apply Leaky ReLU per lane using the native VLRELU instruction.
        Positive values and `+0.0` pass through. Negative values and `-0.0`
        are multiplied by the shared scalar slope. A slope of `+inf`
        produces `NaN` for `-0.0`.

        Parameters:
        - `src` (`VectorSSA`): FP16 or FP32 input register. Required.
        - `slope` (`Float16 | Float32 | float`): Shared scalar slope. A typed
          slope must match the element type of `src`. A Python `float` literal
          is materialized in the element type of `src`. Required.
        - `mask` (`MaskSSA | None`): Execution mask in the source lane
          domain. Defaults to `None`, which enables all physical lanes.
          Masked-off lanes are zeroed.

        Constraints:
        - Call inside a `@tla.kernel` and `tla.vec.func(mode="simd")`.
        - The result has the same type as `src`. Use a 128-lane mask for FP16
          or a 64-lane mask for FP32. For partial inputs, use a mask that
          enables only valid source lanes.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            y = tla.leaky_relu(x, 0.25, mask=mask)
        ```
    """
    _runtime._require_kernel_simd_region("leaky_relu")
    _require_category("leaky_relu", "src", src, "vector_ssa", 0)
    element_type = _vector_ssa_type_for_mlir_value(_as_value(src)).element_type
    if element_type not in ("f16", "f32"):
        _op_error("leaky_relu", "src must have Float16 or Float32 elements")
    resolved_slope = _resolve_bound_value(slope)
    slope_num = _as_vector_scalar_numeric(resolved_slope)
    if not isinstance(slope_num, (Float16, Float32)):
        _op_error(
            "leaky_relu", "slope must be a Float16 or Float32 scalar or a Python float"
        )
    if not isinstance(resolved_slope, float) and slope_num.dtype != element_type:
        _op_error(
            "leaky_relu",
            f"slope must match the src element type {element_type}, got {slope_num.dtype}",
        )
    return _emit_vector_scalar_binary(
        "leaky_relu", _tla_ops_gen.leaky_relu, src, slope_num, mask=mask, loc=loc
    )


@dsl_user_op
def _vector_max(
    lhs: VectorSSA | Numeric | bool | int | float,
    rhs: VectorSSA | Numeric | bool | int | float,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
        Element-wise vector maximum, or the maximum of two per-thread scalars.

        Parameters:
        - `lhs` (`VectorSSA | Numeric | bool | int | float`): Left-hand operand. Required.
        - `rhs` (`VectorSSA | Numeric | bool | int | float`): Right-hand operand. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Vector form only. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.
        - Inside a `tla.vec.func(mode="simt")` region two scalar operands emit
          the per-thread `tla.simt_max`; outside one, scalars keep reaching the
          scalar form of the vector op.
        - The per-thread form needs both operands of the same type, and rejects
          unsigned integers and `Bool`; use signed integers or floats.

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
    """Directory: SIMD Vector Compute / Basic Arithmetic

    Description:
        Element-wise vector minimum, or the minimum of two per-thread scalars.

        Parameters:
        - `lhs` (`VectorSSA | Numeric | bool | int | float`): Left-hand operand. Required.
        - `rhs` (`VectorSSA | Numeric | bool | int | float`): Right-hand operand. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Vector form only. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.
        - Inside a `tla.vec.func(mode="simt")` region two scalar operands emit
          the per-thread `tla.simt_min`; outside one, scalars keep reaching the
          scalar form of the vector op.
        - The per-thread form needs both operands of the same type, and rejects
          unsigned integers and `Bool`; use signed integers or floats.

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
    """Directory: SIMD Vector Compute / Basic Arithmetic

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
    mode: ReduceMode = ReduceMode.INTER_BLOCK,
    mask: MaskSSA,
    init_value: Any | None = None,
    reduction_profile: Any | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    op_name = "VectorSSA.reduce"
    _require_category(op_name, "operand", operand, "vector_ssa", 0)
    _require_category(op_name, "mask", mask, "mask_ssa", 1)
    if not isinstance(mode, ReduceMode):
        _op_error(
            op_name,
            f"invalid argument 'mode': expected ReduceMode, got {_type_name(mode)}",
        )
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
    mask_value = _as_value(mask)
    _require_mask_matches_vector(op_name, mask_value, operand_value)
    # The scopes emit different IR ops (the lowering stays separate); only the
    # frontend dispatch is merged here.
    if mode is ReduceMode.PAIR_WISE:
        if _runtime._in_simt_vec_func():
            _op_error(
                op_name, "PAIR_WISE is only supported in tla.vec.func(mode='simd')"
            )
        if kind is not ReductionOp.ADD:
            _op_error(op_name, "PAIR_WISE only supports ReductionOp.ADD")
        _check_pair_wise_element_type_supported(op_name, operand_desc.element_type)
        result = _tla_ops_gen.pair_wise(
            operand_value.type,
            operand_value,
            kind.value,
            mask=mask_value,
            loc=loc,
        )
        return VectorSSA(result)
    if mode is ReduceMode.INTRA_BLOCK:
        if _runtime._in_simt_vec_func():
            _op_error(
                op_name, "INTRA_BLOCK is only supported in tla.vec.func(mode='simd')"
            )
        _check_reduce_intra_block_element_type_supported(
            op_name, operand_desc.element_type
        )
        # ADD on 16-bit sources returns the widened 32-bit dst (same as vcadd):
        # the lane count halves per 32-bit result lane, rounding up so a tail's
        # last valid element keeps its widened lane (one valid i16 lane still
        # yields one valid i32 result lane). A dynamic lane count stays dynamic
        # in the widened result, matching the dynamic pass-through convention
        # of _cast_result_descriptor.
        if kind is ReductionOp.ADD and operand_desc.element_type in ("i16", "u16"):
            widened_lanes = (
                None
                if operand_desc.valid_lanes is None
                else (operand_desc.valid_lanes + 1) // 2
            )
            result_desc = TlaVectorSSATypeDescriptor(
                widened_lanes,
                "i32" if operand_desc.element_type == "i16" else "u32",
            )
        else:
            result_desc = TlaVectorSSATypeDescriptor(
                operand_desc.valid_lanes, operand_desc.element_type
            )
        result = _tla_ops_gen.reduce_intra_block(
            result_desc.to_mlir_type(operand_value.type.context),
            operand_value,
            mask_value,
            kind.value,
            loc=loc,
        )
        return VectorSSA(result)
    _check_reduction_element_type_supported(op_name, operand_desc.element_type)
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
    x: VectorSSA | MaskSSA,
    y: VectorSSA | MaskSSA,
    *,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA | MaskSSA:
    """Directory: SIMD Vector Compute / Compare and Select

    Description:
        Select between two vectors or two predicate masks under a mask.

        Parameters:
        - `mask` (`MaskSSA`): Select mask; true selects `x`, false selects `y`. Required.
        - `x` (`VectorSSA | MaskSSA`): Value when the mask is true. Required.
        - `y` (`VectorSSA | MaskSSA`): Value when the mask is false. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; `x` and `y` must belong to the same category.
        - For MaskSSA arms, `mask`, `x`, and `y` must have exactly the same type.
        - For VectorSSA arms, `mask` must match the vectors' lane layout.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            z = tla.where(m, x_reg, y_reg)
            selected_mask = tla.where(m, x_mask, y_mask)
        ```
    """
    op_name = "where"
    _require_category(op_name, "mask", mask, "mask_ssa", 0)
    supported_arms = ("vector_ssa", "mask_ssa")
    x_category = _category(x)
    y_category = _category(y)
    if x_category not in supported_arms:
        _require_categories(op_name, "x", x, supported_arms, 1)
    if y_category not in supported_arms:
        _require_categories(op_name, "y", y, supported_arms, 2)
    if x_category != y_category:
        _op_error(
            op_name,
            "x and y must both be VectorSSA values or both be MaskSSA values",
        )

    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    x_value = _as_value(x)
    y_value = _as_value(y)
    mask_value = _as_value(mask)

    if x_category == "mask_ssa":
        if y_value.type != x_value.type:
            _op_error(
                op_name,
                f"y has type {y_value.type}, expected {x_value.type}",
            )
        if mask_value.type != x_value.type:
            _op_error(
                op_name,
                f"mask has type {mask_value.type}, expected {x_value.type}",
            )
        result = _tla_ops_gen.where(
            x_value.type,
            mask_value,
            x_value,
            y_value,
            loc=loc,
        )
        return MaskSSA(result)

    x_desc = _vector_ssa_type_for_mlir_value(x_value)
    y_desc = _vector_ssa_type_for_mlir_value(y_value)
    if x_desc.element_type != y_desc.element_type:
        _op_error(
            "where",
            f"y has element type {y_desc.element_type}, expected {x_desc.element_type}",
        )
    _check_select_element_type_supported("where", x_desc.element_type)
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
    """Directory: SIMD Vector Compute / Compare and Select

    Description:
        Select between two vectors or predicate masks under a mask, or between
        two per-thread scalars inside a SIMT region.

        Parameters:
        - `mask` (`MaskSSA | Bool`): Select mask; true selects `x`, false selects `y`. A MaskSSA whole-vector, or the Bool from a comparison inside a SIMT region. Required.
        - `x` (`VectorSSA | MaskSSA | Numeric`): Value when the mask is true. Required.
        - `y` (`VectorSSA | MaskSSA | Numeric`): Value when the mask is false. Required.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; SIMD arms must belong to the same category.
        - MaskSSA arms and their selector must have exactly the same type.
        - VectorSSA arms must use one of Bool, i8/u8, i16/u16, i32/u32, f16/bf16/f32,
          i64/u64, f8e4m3fn/f8e5m2, hif8, or f4e2m1x2/f4e1m2x2. Bool values
          select whole b8 storage bytes; the low-precision formats select their
          byte/pair carriers.
        - Inside a `tla.vec.func(mode="simt")` region a `Bool` mask emits the
          per-thread `tla.simt_where`; outside one, only the `MaskSSA`
          whole-vector form is available.

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
    record_nbytes: bool = False,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Data Compress
    Description:
        Compress vector lanes kept by a mask.

        Parameters:
        - `src` (`VectorSSA`): Source vector to compress. Required.
        - `mask` (`MaskSSA`): Mask of elements to keep. Required.
        - `record_nbytes` (`bool`): When `True`, hardware calculates the byte
          count of the data selected by `mask` and records it in a dedicated
          internal register. `store` uses this count in
          `POST_MODE_UPDATE_NBYTES` mode. Defaults to `False`. Keyword-only.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func(mode="simd")`.
        - Supported element types: `f32`, `f16`, `i8`, `i16`, `i32`, `u8`, `u16`, `u32`.

        With `record_nbytes=True`, the byte count is the number of kept
        elements times the size of each element in bytes. Hardware records
        this count in a dedicated internal register. Before the count is
        overwritten, the caller must call `store` with
        `UnalignStoreParams(post_mode=POST_MODE_UPDATE_NBYTES)` to write the
        intended data to the destination. For an example, see `Tensor.store`.

        Example:
        ```python
        # src_ub is a contiguous Int16 UB window with 128 elements.
        with tla.vec.func(mode="simd"):
            keep, _ = tla.update_mask(3, dtype=tla.Int16)
            packed = tla.squeeze(src_ub.load(), keep)
        ```
    """
    _require_category("squeeze", "src", src, "vector_ssa", 0)
    _require_category("squeeze", "mask", mask, "mask_ssa", 1)
    if not isinstance(record_nbytes, bool):
        _op_error("squeeze", "record_nbytes must be a bool")
    _require_frontend_state("squeeze")
    _runtime._require_enclosing_region("squeeze", "vec.func")
    if _runtime._in_simt_vec_func():
        _op_error("squeeze", "requires tla.vec.func(mode='simd')")
    src_value = _as_value(src)
    mask_value = _as_value(mask)
    _require_mask_matches_vector("squeeze", mask_value, src_value)
    result = _tla_ops_gen.squeeze(
        src_value.type,
        src_value,
        mask_value,
        store_reg=record_nbytes,
        loc=loc,
    )
    return VectorSSA(result)


_UNSQUEEZE_SUPPORTED_DTYPES = frozenset(("i8", "u8", "i16", "u16", "i32", "u32"))


@dsl_user_op
def unsqueeze(
    mask: MaskSSA,
    dtype: type[Numeric],
    *,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Data Compress
    Description:
        Expand a mask into per-lane compressed prefix indices.

        Lane `i` of the result holds the number of active mask bits in lanes
        `< i`. This is the inverse mapping of `tla.squeeze`: feeding the indices
        to `tla.gather` reads a compacted register back into the lanes selected
        by `mask`.

        Parameters:
        - `mask` (`MaskSSA`): Mask to expand. Required.
        - `dtype` (`type[Numeric]`): Result element type. Required. The width
          fixes the mask lane count: `i8`/`u8` -> `mask<256>`, `i16`/`u16` ->
          `mask<128>`, `i32`/`u32` -> `mask<64>`. The result always covers the
          full vector register (`256`/`128`/`64` valid lanes for 8/16/32-bit
          elements) and does not inherit the mask's valid-lane count.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func(mode="simd")`.
        - Supported result element types: `i8`, `u8`, `i16`, `u16`, `i32`, `u32`.
        - `dtype` width must match the mask lane count.
        - A partially active mask (for example a 32-lane tail in a 64-lane
          register) still yields a full-register index; pair it with a masked
          `store` so only the restored lanes are written.
        - Counts are written in `dtype`'s bit width. An `i8` result wraps counts
          above 127 (e.g. 128 -> -128); use `u8`/`u16`/`u32` when the caller
          needs the full non-negative count. The wrapped bit pattern is still a
          valid gather index.

        Example:
        ```python
        # Restore a 32-lane tail through squeeze -> unsqueeze -> gather.
        with tla.vec.func(mode="simd"):
            keep, _ = tla.update_mask(32, dtype=tla.Int32)
            packed = tla.squeeze(src_tile.load(), keep)
            idx = tla.unsqueeze(keep, tla.UInt32)
            dst_tile.store(tla.gather(packed, idx), mask=keep)
        ```
    """
    op_name = "unsqueeze"
    _require_category(op_name, "mask", mask, "mask_ssa", 0)
    if not (
        isinstance(dtype, type)
        and issubclass(dtype, Numeric)
        and getattr(dtype, "dtype", "")
    ):
        _op_error(
            op_name,
            f"invalid argument 'dtype' (position 1): expected an integer Numeric "
            f"type (tla.Int8/UInt8/Int16/UInt16/Int32/UInt32), got {_type_name(dtype)}",
        )
    dtype_token = str(dtype.dtype).strip().lower()
    if dtype_token not in _UNSQUEEZE_SUPPORTED_DTYPES:
        _op_error(
            op_name,
            f"unsupported vector element dtype {dtype.dtype}; supported dtypes are "
            f"{', '.join(sorted(_UNSQUEEZE_SUPPORTED_DTYPES))}",
        )
    _require_frontend_state(op_name)
    _runtime._require_enclosing_region(op_name, "vec.func")
    if _runtime._in_simt_vec_func():
        _op_error(op_name, "requires tla.vec.func(mode='simd')")
    mask_value = _as_value(mask)
    _require_mask_matches_element_type(op_name, mask_value, dtype_token)
    desc = _full_vector_ssa_descriptor(dtype_token)
    context = loc.context if loc is not None else mlir_ir.Context.current
    return VectorSSA(
        _tla_ops_gen.unsqueeze(desc.to_mlir_type(context), mask_value, loc=loc)
    )


def _normalize_histograms_params(params: HistogramsParams) -> HistogramsParams:
    """Accept enum or str for HistogramsParams fields; return enum-typed params."""
    mode = params.mode
    hist_type = params.type
    if isinstance(mode, str):
        try:
            mode = HistogramsBinType(mode)
        except ValueError:
            _op_error(
                "histograms",
                f"invalid HistogramsParams.mode: expected HistogramsBinType or "
                f"'bin0'|'bin1', got {mode!r}",
            )
    elif not isinstance(mode, HistogramsBinType):
        _op_error(
            "histograms",
            f"invalid HistogramsParams.mode: expected HistogramsBinType, "
            f"got {_type_name(mode)}",
        )
    if isinstance(hist_type, str):
        try:
            hist_type = HistogramsType(hist_type)
        except ValueError:
            _op_error(
                "histograms",
                f"invalid HistogramsParams.type: expected HistogramsType or "
                f"'frequency'|'accumulate', got {hist_type!r}",
            )
    elif not isinstance(hist_type, HistogramsType):
        _op_error(
            "histograms",
            f"invalid HistogramsParams.type: expected HistogramsType, "
            f"got {_type_name(hist_type)}",
        )
    return HistogramsParams(mode=mode, type=hist_type)


@dsl_user_op
def histograms(
    src: VectorSSA,
    dst: VectorSSA,
    params: HistogramsParams,
    mask: MaskSSA | None = None,
    *,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Histogram Compute

    Description:
        Register-resident FREQUENCY or ACCUMULATE histogram.

        Parameters:
        - `src` (`VectorSSA`): ``i8``/``ui8`` source register (256 lanes). Required.
        - `dst` (`VectorSSA`): ``i16``/``ui16`` accumulator (128 lanes). Required.
          Accumulates onto the prior ``dst`` value. For a fresh histogram, pass
          ``tla.full(0, dtype=...)`` first.
        - `params` (`HistogramsParams`): ``mode`` (BIN0/BIN1) and ``type``
          (FREQUENCY/ACCUMULATE). Required.
        - `mask` (`MaskSSA`): Optional lane mask; default all-true.

        Constraints:
        - Must be called inside `@tla.kernel` and `tla.vec.func()`.
        - FREQUENCY: half-range per-value counts. ACCUMULATE: CDF over u8 then
          split; under ACCUMULATE, ``BIN1`` is not ``[128,255]``-only.

        Example:
        ```python
        hist_params = tla.params.HistogramsParams(
            mode=tla.params.HistogramsBinType.BIN0,
            type=tla.params.HistogramsType.ACCUMULATE,
        )
        with tla.vec.func(mode="simd"):
            hist_lo = tla.full(0, dtype=tla.Int16)
            hist_lo = tla.histograms(keys_u8, hist_lo, hist_params, mask=m)
        ```
    """
    _require_category("histograms", "src", src, "vector_ssa", 0)
    _require_category("histograms", "dst", dst, "vector_ssa", 1)
    if not isinstance(params, HistogramsParams):
        _op_error(
            "histograms",
            f"invalid argument 'params' (position 2): expected HistogramsParams, "
            f"got {_type_name(params)}",
        )
    if mask is not None:
        _require_category("histograms", "mask", mask, "mask_ssa", 3)
    _require_frontend_state("histograms")
    _runtime._require_enclosing_region("histograms", "vec.func")

    params = _normalize_histograms_params(params)
    mode_token = params.mode.value
    type_token = params.type.value

    src_value = _as_value(src)
    src_desc = _vector_ssa_type_for_mlir_value(src_value)
    if src_desc.element_type not in ("i8", "ui8", "u8"):
        _op_error(
            "histograms",
            f"src element type must be i8/ui8, got {src_desc.element_type}",
        )
    if src_desc.valid_lanes is not None and src_desc.valid_lanes != 256:
        _op_error(
            "histograms",
            f"src must have 256 lanes (full VL for i8), got {src_desc.valid_lanes}",
        )

    dst_value = _as_value(dst)
    dst_desc = _vector_ssa_type_for_mlir_value(dst_value)
    if dst_desc.element_type not in ("i16", "ui16", "u16"):
        _op_error(
            "histograms",
            f"dst element type must be i16/ui16, got {dst_desc.element_type}",
        )
    if dst_desc.valid_lanes is not None and dst_desc.valid_lanes != 128:
        _op_error(
            "histograms",
            f"dst must have 128 lanes (full VL for i16), got {dst_desc.valid_lanes}",
        )

    mask_value = _as_value(mask) if mask is not None else None
    if mask_value is not None:
        _require_mask_matches_vector("histograms", mask_value, src_value)

    context = src_value.type.context
    # mode → histograms_bin; type → histograms_type.
    mode_attr = mlir_ir.Attribute.parse(
        f"#tla.histograms_bin<{mode_token}>", context=context
    )
    hist_type_attr = mlir_ir.Attribute.parse(
        f"#tla.histograms_type<{type_token}>", context=context
    )

    result = _tla_ops_gen.histograms(
        dst_desc.to_mlir_type(context),
        src_value,
        dst_value,
        mask=mask_value,
        mode=mode_attr,
        hist_type=hist_type_attr,
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
    if is_bool_element_type(element_type):
        _op_error(
            op_name,
            "Bool VectorSSA is move-only (load/store); arithmetic is not supported",
        )
    if element_type not in _BITWISE_UNARY_TYPES:
        _op_error(
            op_name,
            f"tla.{op_name} requires f16/f32, i8/i16/i32 or an FP4 pair "
            f"(f4e2m1x2/f4e1m2x2) element type, got {element_type}",
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
        if is_bool_element_type(src0_desc.element_type):
            _op_error(
                op_name,
                "Bool VectorSSA is move-only (load/store); arithmetic is not supported",
            )
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
    """Directory: SIMD Vector Compute / Compare and Select
    Description:
        Element-wise compare; returns a MaskSSA.

        Parameters:
        - `lhs` (`VectorSSA`): Left-hand compare operand. Required.
        - `rhs` (`VectorSSA | Numeric | bool | int | float`): Right-hand compare operand (scalar or vector). Required.
        - `mode` (`str`): Compare mode. Required.

          | Value | Meaning |
          |---|---|
          | `lt` | Less than |
          | `le` | Less than or equal |
          | `gt` | Greater than |
          | `ge` | Greater than or equal |
          | `eq` | Equal |
          | `ne` | Not equal |

        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; `mode` must be a supported compare mnemonic.
        - Element type must be one of f16/bf16/f32, i8/i16/i32/i64, or u8/u16/u32/u64.
          Vector-vector and vector-scalar compares use the same element-type set.

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
    _check_compare_element_type_supported("cmp", lhs_desc.element_type)
    if rhs_category == "vector_ssa":
        rhs_value = _as_value(rhs)
    else:
        assert rhs_num is not None
        # Unsigned vectors keep their unsigned TLA type, but the scalar carrier
        # is a signless integer: arith.constant only accepts signless, and the
        # AVE scalar ABI is signless too. The compare itself still derives
        # unsigned semantics from the vector element type.
        rhs_value = _numeric_ir_value_for_element_type(
            "cmp",
            rhs_num,
            lhs_desc.element_mlir_type(lhs_value.type.context),
            normalize_unsigned=True,
            loc=loc,
        )
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
    """Directory: SIMD Vector Compute / Logical Compute

    Description:
        Element-wise bitwise and (Mask or Vector).

        Parameters:
        - `src0_reg` (`VectorSSA | MaskSSA`): Left-hand bitwise-and operand. Required.
        - `src1_reg` (`VectorSSA | MaskSSA`): Right-hand bitwise-and operand. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.
        - Vector source elements must be f16, bf16, f32, i32, i16, i8, u8, u16, u32, f4e2m1x2, or f4e1m2x2.
        - `f4e2m1x2` / `f4e1m2x2` vectors carry one FP4 pair per lane; a `mask` selects whole pairs.

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
    """Directory: SIMD Vector Compute / Logical Compute

    Description:
        Element-wise bitwise or (Mask or Vector).

        Parameters:
        - `src0_reg` (`VectorSSA | MaskSSA`): Left-hand bitwise-or operand. Required.
        - `src1_reg` (`VectorSSA | MaskSSA`): Right-hand bitwise-or operand. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.
        - Vector source elements must be f16, bf16, f32, i32, i16, i8, u8, u16, u32, f4e2m1x2, or f4e1m2x2.
        - `f4e2m1x2` / `f4e1m2x2` vectors carry one FP4 pair per lane; a `mask` selects whole pairs.

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
    """Directory: SIMD Vector Compute / Logical Compute

    Description:
        Element-wise bitwise xor (Mask or Vector).

        Parameters:
        - `src0_reg` (`VectorSSA | MaskSSA`): Left-hand bitwise-xor operand. Required.
        - `src1_reg` (`VectorSSA | MaskSSA`): Right-hand bitwise-xor operand. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask; `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.
        - Vector source elements must be f16, bf16, f32, i32, i16, i8, u8, u16, u32, f4e2m1x2, or f4e1m2x2.
        - `f4e2m1x2` / `f4e1m2x2` vectors carry one FP4 pair per lane; a `mask` selects whole pairs.

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


def _require_unsigned_cast_route(src_token: str, dst_token: str) -> None:
    """Refuse a cast the unsigned AVE instruction set cannot express."""
    src_unsigned = src_token in _CAST_UNSIGNED_DTYPES
    dst_unsigned = dst_token in _CAST_UNSIGNED_DTYPES
    if not src_unsigned and not dst_unsigned:
        return
    if (src_token, dst_token) not in _CAST_UNSIGNED_ROUTES:
        routes = ", ".join(f"{a}->{b}" for a, b in sorted(_CAST_UNSIGNED_ROUTES))
        _op_error(
            "cast",
            f"unsupported cast '{src_token}'->'{dst_token}': the unsigned "
            f"integers do not form a complete width lattice. Supported unsigned "
            f"routes are {routes}",
        )


def _require_reg_slot_reachable(src_token: str, dst_token: str, slot: RegSlot) -> None:
    """Refuse a reg_slot the lowering would quietly ignore.

    Slots TWO and THREE name a pack quarter, and only an integer cast of 4x
    width or more, or a direct fp8<->f32 cast, reads one (pp0..pp3). Every
    other width-changing cast carries an even/odd part, where mapCastPart folds
    TWO and THREE onto PART_EVEN -- so accepting them would hand back slot
    ZERO's lanes with no diagnostic. The equal-width f16->bf16 instruction
    preserves all lanes without a selector and also rejects pack quarters.

    The fp8<->f32 instruction is the one float exception: it is a 4x-width
    conversion and uses the same pp0..pp3 selector family as i8<->i32.
    """
    if slot not in (RegSlot.TWO, RegSlot.THREE):
        return
    if (src_token == "f16" and dst_token == "bf16") or (
        src_token == dst_token and src_token in _CAST_SAME_TYPE_FLOAT_DTYPES
    ):
        _op_error(
            "cast",
            f"invalid reg_slot {slot} for cast '{src_token}'->'{dst_token}': "
            f"this equal-width conversion preserves all lanes and does not "
            f"select a pack quarter; use ZERO or ONE",
        )
    src_bytes = dtype_size_bytes(src_token)
    dst_bytes = dtype_size_bytes(dst_token)
    int_pair = (src_token.startswith("i") and dst_token.startswith("i")) or (
        src_token in _CAST_UNSIGNED_DTYPES and dst_token in _CAST_UNSIGNED_DTYPES
    )
    ratio = (
        _builtins.max(src_bytes, dst_bytes) // _builtins.min(src_bytes, dst_bytes)
        if src_bytes and dst_bytes
        else 0
    )
    fp8_f32_pair = (src_token in _CAST_FP8_DTYPES and dst_token == "f32") or (
        src_token == "f32" and dst_token in _CAST_FP8_DTYPES
    )
    # u8 -> f32 is composed as a 4x widen then convert; reg_slot still names
    # the pack quarter of the widen leg.
    u8_f32_pair = src_token == "u8" and dst_token == "f32"
    hif8_f32_pair = (src_token, dst_token) in (("f32", "hif8"), ("hif8", "f32"))
    fp4_pair = (src_token == "bf16" and dst_token in _FP4_PAIR_DTYPES) or (
        src_token in _FP4_PAIR_DTYPES and dst_token == "bf16"
    )
    if (
        (int_pair and ratio >= 4)
        or (src_token, dst_token) == ("i32", "u8")
        or (src_token == "i4x2" and dst_token in _INT4_CAST_TARGETS)
        or (dst_token == "i4x2" and src_token in _INT4_CAST_SOURCES)
        or fp8_f32_pair
        or u8_f32_pair
        or hif8_f32_pair
        or fp4_pair
    ):
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
    x: Tensor | VectorSSA,
    y: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA:
    """Directory: SIMD Vector Compute / Discrete and Aggregate
    Description:
        Gather vector values by indices from a UB tile or a register.

        Two forms:

        - **UB→Reg**: ``x`` is a UB ``Tensor``; ``y`` is ``i32`` indices.
          Optional ``mask``.
        - **Reg→Reg**: ``x`` is ``VectorSSA``; ``y`` is an integer index
          vector with the same element bit-width as ``x`` (``i8``/``u8``,
          ``i16``/``u16``, or ``i32``/``u32``); ``result[i] = x[y[i] % VL]``.
          ``y`` may have more valid lanes than ``x``; extra index lanes are
          ignored and the result keeps ``x``'s valid lane count. No ``mask``.

        Parameters:
        - `x` (`Tensor | VectorSSA`): Source tile (UB) or source register. Required.
        - `y` (`VectorSSA`): Index vector register. Required.
        - `mask` (`MaskSSA | None`): Optional execution mask for UB→Reg only;
          `None` means all lanes enabled. Optional, default `None`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`.
        - Tensor source must reside in UB.
        - For Reg→Reg, the index vector must have at least as many valid lanes
          as ``x``; a wider index is allowed and its extra lanes are ignored.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            vals = tla.gather(ub_tile, idx_i32)
            w_brc = tla.gather(w, idx_u16)  # Reg→Reg
        ```
    """
    _require_categories("gather", "x", x, ("tensor", "vector_ssa"), 0)
    _require_category("gather", "y", y, "vector_ssa", 1)
    _require_frontend_state("gather")
    _runtime._require_enclosing_region("gather", "vec.func")
    x_value = _as_value(x)
    y_value = _as_value(y)
    y_desc = _vector_ssa_type_for_mlir_value(y_value)
    if y_desc is None:
        _op_error("gather", "invalid argument 'y' (position 1)")

    # Reg→Reg: x is VectorSSA → bc vselr.
    if _category(x) == "vector_ssa":
        if mask is not None:
            _op_error("gather", "Reg→Reg gather does not support mask")
        x_desc = _vector_ssa_type_for_mlir_value(x_value)
        if x_desc is None:
            _op_error("gather", "invalid argument 'x' (position 0)")
        _GATHER_REG_SRC = frozenset(
            {
                "i8",
                "u8",
                "i16",
                "u16",
                "i32",
                "u32",
                "f16",
                "bf16",
                "f32",
            }
        )
        src_tok = x_desc.element_type.lower()
        if src_tok not in _GATHER_REG_SRC:
            _op_error(
                "gather",
                f"unsupported Reg→Reg src element type {x_desc.element_type}; "
                f"expected one of {sorted(_GATHER_REG_SRC)}",
            )

        src_bytes = dtype_size_bytes(src_tok)
        index_tok = y_desc.element_type.lower()
        _INDEX_OK = {
            1: frozenset({"u8", "i8"}),
            2: frozenset({"u16", "i16"}),
            4: frozenset({"u32", "i32"}),
        }
        if index_tok not in _INDEX_OK[src_bytes]:
            _op_error(
                "gather",
                f"Reg→Reg gather index element type must match src bit-width "
                f"({sorted(_INDEX_OK[src_bytes])} for {src_bytes}-byte src), "
                f"got {y_desc.element_type}",
            )

        if (
            x_desc.valid_lanes is not None
            and y_desc.valid_lanes is not None
            and y_desc.valid_lanes < x_desc.valid_lanes
        ):
            _op_error(
                "gather",
                f"index has fewer lanes than src: got {y_desc.valid_lanes} vs "
                f"{x_desc.valid_lanes}; the index may be wider but not narrower",
            )

        result = _tla_ops_gen.gather(
            x_value.type,
            x_value,
            y_value,
            mask=None,
            loc=loc,
        )
        return VectorSSA(result)

    # UB→Reg: x is a UB tensor → HIVMAVE vgather.
    x_desc = _tla_tensor_descriptor_from_type_or_value(x_value)
    if x_desc is None:
        _op_error("gather", "invalid argument 'x' (position 0)")

    # validate x addrspace is ub
    if x_desc.addrspace.lower() != "ub":
        _op_error(
            "gather",
            f"invalid argument 'x' (position 0): expected addrspace ub, got {x_desc.addrspace}",
        )

    # validate x element type. Bool (i1) is move-only load/store; UB→Reg
    # gather has no i8-carrier path and fails device compile.
    if is_bool_element_type(x_desc.element_type):
        _op_error(
            "gather",
            "Bool VectorSSA is move-only (load/store); gather is not supported",
        )
    _GATHER_SUPPORTED_X_ELEM_TYPES = frozenset(
        {
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
def _arch_atomic_add_simt(
    ptr: Pointer,
    value: Any,
    *,
    loc: mlir_ir.Location | None = None,
) -> Numeric | None:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Atomically add a scalar to the element `ptr` addresses
        (`tla.arch.atomic_add(ptr, value)`).

        The pointer carries the element type and address space, so the caller
        computes the element address. `t.ptr` is the storage base, so
        `t.ptr + i` addresses element `i` from the base; for a view, use
        `t.ptr_with_coord + i` (the view's first element plus `i`) or
        `t.ptr + element_offset`. Any pointer built with `tla.make_ptr` /
        `tla.arch.get_dyn_ub` / `tla.allocate` is accepted too.

        The read-modify-write is atomic, so any number of threads may target
        the same element without losing an update. Typical uses are counters,
        histogram bins and reduction accumulators.

        Parameters:
        - `ptr` (`Pointer`): Pointer to the element that is updated. Required.
        - `value` (`Numeric | int | float | mlir.ir.Value`): Addend. A Python
          literal is converted to the pointer's pointee type; a `Numeric` value
          must match it exactly. Required.

        Returns:
        - The element's value before the update, as a `Numeric`, for every
          supported type except `f16`/`bf16`: for those the hardware return
          value is not the original element, so nothing is returned.
        - When the return value is unused the compiler can emit the better
          instruction for the 32-bit integer and floating-point types; the
          64-bit forms have no such choice. Ignore it when only the accumulate
          matters.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Only available inside `tla.vec.func(mode="simt")`.
        - The pointer must be in `gm` or `ub`, and must be built inside the
          region: from `tensor.ptr` / `tensor.ptr_with_coord` or
          `tla.make_ptr`. A pointer value cannot be captured from outside the
          region.
        - Pointee type must be `i32`, `u32`, `i64`, `u64`, `f32`, `f16` or
          `bf16`; the 64-bit types are only available in `gm`.

        Example:
        ```python
        with tla.vec.func(mode="simt", thread_block_dim=256):
            tid, _, _ = tla.arch.thread_idx()
            tla.arch.atomic_add(gm_histogram.ptr + tid, 1.0)
        ```
    """
    from .base_dsl.typing import as_numeric

    _require_category("atomic_add", "ptr", ptr, "pointer", 0)
    _require_frontend_state("atomic_add")
    if not _runtime._in_simt_vec_func():
        _op_error(
            "atomic_add",
            "is only available inside a tla.vec.func with mode='simt'",
        )

    p = _coerce_pointer_arg(ptr)
    ptr_ty = p._ptr_ty
    addrspace = ptr_ty.addrspace.lower()
    if addrspace not in ("gm", "ub"):
        _op_error(
            "atomic_add",
            f"invalid argument 'ptr' (position 0): expected a gm or ub pointer, got {addrspace}",
        )

    elem_type = ptr_ty.pointee
    if mlir_ir.IntegerType.isinstance(elem_type):
        integer_type = mlir_ir.IntegerType(elem_type)
        element_token = ("u" if integer_type.is_unsigned else "i") + str(
            integer_type.width
        )
    else:
        element_token = str(elem_type)
    supported = ("i32", "u32", "i64", "u64", "f32", "f16", "bf16")
    if element_token not in supported:
        _op_error(
            "atomic_add",
            f"unsupported pointee type {element_token}; supported types are "
            f"{', '.join(supported)}",
        )
    # The atom unit's 64-bit forms exist only for GM.
    if element_token in ("i64", "u64") and addrspace != "gm":
        _op_error(
            "atomic_add",
            f"invalid argument 'ptr' (position 0): 64-bit atomic add is only "
            f"supported on gm pointers, got {addrspace}",
        )

    def _atomic_addend(literal):
        """``arith.constant`` of the pointee type for a host scalar literal.

        arith only carries signless integers, so an unsigned pointee gets a
        signless constant of the same width and the cast below lifts it back.
        The literal is range-checked against the unsigned bounds here.
        """
        if mlir_ir.IntegerType.isinstance(elem_type):
            integer_type = mlir_ir.IntegerType(elem_type)
            if integer_type.is_unsigned:
                if isinstance(literal, float) and not literal.is_integer():
                    _op_error(
                        "atomic_add",
                        f"invalid argument 'value' (position 1): expected integer "
                        f"scalar for {elem_type}, got {literal!r}",
                    )
                width = int(integer_type.width)
                number = int(literal)
                if not 0 <= number <= 2**width - 1:
                    _op_error(
                        "atomic_add",
                        f"invalid argument 'value' (position 1): integer scalar "
                        f"{number} out of range for {elem_type} (valid range "
                        f"[0, {2**width - 1}])",
                    )
                signless = mlir_ir.IntegerType.get_signless(width)
                # arith constants are signless: the same bits read as two's
                # complement. A uint64 literal past 2**63 therefore arrives as
                # its negative bit pattern, not as an out-of-range attribute.
                bits = number - (1 << width) if number >= 1 << (width - 1) else number
                return mlir_ir.Operation.create(
                    "arith.constant",
                    results=[signless],
                    attributes={"value": mlir_ir.IntegerAttr.get(signless, bits)},
                    loc=loc,
                ).results[0]
        return _scalar_constant_for_element_type(
            "atomic_add", literal, elem_type, loc=loc
        )

    resolved = _resolve_bound_value(value)
    if isinstance(resolved, (int, float)) and not isinstance(resolved, bool):
        add_value = _atomic_addend(resolved)
    else:
        try:
            if isinstance(resolved, Numeric):
                num = resolved
            elif isinstance(resolved, mlir_ir.Value):
                num = as_numeric(resolved)
            elif isinstance(value, Numeric):
                num = value
            elif isinstance(value, mlir_ir.Value):
                num = as_numeric(value)
            else:
                raise TypeError(type(value).__name__)
        except (TypeError, ValueError, KeyError):
            _op_error(
                "atomic_add",
                f"invalid argument 'value' (position 1): expected Numeric or "
                f"scalar literal, got {_type_name(value)}",
            )
        dest_cls = Numeric.from_mlir_type(elem_type)
        if not isinstance(num, dest_cls):
            _op_error(
                "atomic_add",
                f"type mismatch, atomic_add {type(num).dtype} to a pointer whose "
                f"pointee is {dest_cls.dtype}; cast explicitly with "
                f".to({dest_cls.__name__}) before atomic_add",
            )
        if isinstance(num.value, (bool, int, float)):
            add_value = _atomic_addend(num.value)
        else:
            add_value = num.ir_value(loc=loc)
    # A ui32/ui64 pointer's op operand is the unsigned type; the conversion
    # helpers produce the signless one.
    if (
        mlir_ir.IntegerType.isinstance(elem_type)
        and mlir_ir.IntegerType(elem_type).is_unsigned
        and add_value.type
        == mlir_ir.IntegerType.get_signless(mlir_ir.IntegerType(elem_type).width)
    ):
        add_value = mlir_ir.Operation.create(
            "builtin.unrealized_conversion_cast",
            operands=[add_value],
            results=[elem_type],
            loc=loc,
        ).results[0]
    if str(add_value.type) != str(elem_type):
        raise TlaLoweringError(
            f"tla.atomic_add value type mismatch: expected {elem_type}, got {add_value.type}"
        )
    old_value = _tla_ops_gen.simt_atomic_add(elem_type, p.value, add_value, loc=loc)
    if str(old_value.type) != str(elem_type):
        raise TlaLoweringError(
            f"tla.atomic_add result type mismatch: expected {elem_type}, got {old_value.type}"
        )
    # f16/bf16: the hardware return value is not the original element, so the
    # API returns nothing rather than hand out an unusable value.
    if element_token in ("f16", "bf16"):
        return None
    dtype = Numeric.from_mlir_type(elem_type)
    if dtype.is_integer and not dtype.signed:
        old_value = mlir_ir.Operation.create(
            "builtin.unrealized_conversion_cast",
            operands=[old_value],
            results=[mlir_ir.IntegerType.get_signless(dtype.width)],
            loc=loc,
        ).results[0]
    return dtype(old_value)


@dsl_user_op
def scatter(
    dst: Tensor,
    src: VectorSSA,
    indices: VectorSSA,
    *,
    mask: MaskSSA | None = None,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Directory: SIMD Vector Compute / Discrete and Aggregate

    Description:
        Scatter register values into UB at element offsets relative to `dst`.
        Enabled lane `j` writes `dst[indices[j]] = src[j]`; for 8-bit data,
        it writes `src[2*j]` instead. Offsets are not scaled by the layout.

        Parameters:
        - `dst` (`Tensor`): UB destination with the same dtype as `src`.
        - `src` (`VectorSSA`): One register of i8/u8/i16/u16/i32/u32/
          i64/u64/f16/bf16/f32 values.
        - `indices` (`VectorSSA`): Unsigned element offsets: u16 for 8/16-bit
          data, u32 for 32-bit data, and u32 or u64 for 64-bit data.
        - `mask` (`MaskSSA | None`): Predicate with 128 lanes for 8/16-bit
          data, 64 for 32-bit, or 32 for 64-bit. `None` enables all lanes;
          masked-off destination elements retain their values.

        Constraints:
        - Call inside `@tla.kernel` and `tla.vec.func(mode="simd")`.
        - The caller must ensure a 32-byte-aligned destination base and unique
          enabled indices within the destination allocation.
        - Source and indices each occupy one 256-byte register. Indices must
          cover all valid source lanes, or ceil(valid_lanes / 2) for 8-bit data.
        - For 64-bit data, only the first 32 u32 indices are used; u64 indices
          use only their low 32 bits.
        - `i64`/`u64` data can be staged between UB and registers with aligned,
          full-width `DIST_NORM` loads and aligned `DIST_NORM` stores without
          an explicit mask. GM/UB `RowMajor` copies are supported in both
          directions, as are GM-to-UB `VectorLayout` copies. See the
          `tla.copy` route table for copy constraints.

        Example:
        ```python
        with tla.vector():
            with tla.vec.func(mode="simd"):
                tla.scatter(dst_ub, values, indices_u32)
        ```
    """
    _runtime._require_kernel_simd_region("scatter")
    _require_category("scatter", "dst", dst, "tensor", 0)
    _require_category("scatter", "src", src, "vector_ssa", 1)
    _require_category("scatter", "indices", indices, "vector_ssa", 2)
    dst_value = _as_value(dst)
    src_value = _as_value(src)
    indices_value = _as_value(indices)
    dst_desc = _tla_tensor_descriptor_from_type_or_value(dst_value)
    src_desc = _vector_ssa_type_for_mlir_value(src_value)
    indices_desc = _vector_ssa_type_for_mlir_value(indices_value)
    if dst_desc is None:
        _op_error("scatter", "invalid argument 'dst' (position 0)")
    if dst_desc.addrspace.lower() != "ub":
        _op_error(
            "scatter",
            f"invalid argument 'dst' (position 0): expected addrspace ub, "
            f"got {dst_desc.addrspace}",
        )
    src_token = src_desc.element_type.lower()
    if is_bool_element_type(src_token) or is_bool_element_type(dst_desc.element_type):
        _op_error(
            "scatter",
            "Bool VectorSSA is move-only (load/store); scatter is not supported",
        )
    supported_types = frozenset(
        {"i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "f16", "bf16", "f32"}
    )
    if src_token not in supported_types:
        _op_error(
            "scatter",
            f"unsupported src element type {src_desc.element_type}; expected "
            f"one of {', '.join(sorted(supported_types))}",
        )
    if dst_desc.element_type.lower() != src_token:
        _op_error(
            "scatter",
            f"src and dst element types must match, got {src_desc.element_type} "
            f"and {dst_desc.element_type}",
        )
    element_bytes = dtype_size_bytes(src_token)
    index_types = {1: ("u16",), 2: ("u16",), 4: ("u32",), 8: ("u32", "u64")}
    if indices_desc.element_type.lower() not in index_types[element_bytes]:
        _op_error(
            "scatter",
            f"indices must have native unsigned element type "
            f"{' or '.join(index_types[element_bytes])} for {src_token} data, "
            f"got {indices_desc.element_type}",
        )
    if src_desc.valid_lanes is not None and indices_desc.valid_lanes is not None:
        scatter_lanes = (
            (src_desc.valid_lanes + 1) // 2
            if element_bytes == 1
            else src_desc.valid_lanes
        )
        if indices_desc.valid_lanes < scatter_lanes:
            _op_error(
                "scatter",
                f"indices has {indices_desc.valid_lanes} valid lanes, but src "
                f"requires at least {scatter_lanes} scatter indices",
            )
    mask_value = None
    if mask is not None:
        _require_category("scatter", "mask", mask, "mask_ssa", 3)
        mask_value = _as_value(mask)
        mask_lanes = _mask_ssa_type_for_mlir_value(mask_value).physical_lanes
        expected_mask_lanes = 128 if element_bytes == 1 else 256 // element_bytes
        if mask_lanes != expected_mask_lanes:
            _op_error(
                "scatter",
                f"mask has {mask_lanes} predicate lanes, expected "
                f"{expected_mask_lanes} for {src_token} scatter",
            )
    _tla_ops_gen.scatter(dst_value, src_value, indices_value, mask=mask_value, loc=loc)


@dsl_user_op
def arch_block_idx(*, loc: mlir_ir.Location | None = None) -> Int32:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return the block index of the current AI core in the launch (`Int32`).

        Parameters:
        - None.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.

        Example:
        ```python
        bid = tla.arch.block_idx()
        ```
    """
    _require_frontend_state("arch.block_idx")
    i32 = mlir_ir.IntegerType.get_signless(32)
    value = _tla_ops_gen.arch_block_idx(i32, loc=loc)
    return Int32(value)


@dsl_user_op
def arch_sub_block_idx(*, loc: mlir_ir.Location | None = None) -> Int32:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return the sub-block index within the current block (`Int32`).

        Parameters:
        - None.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.

        Example:
        ```python
        sid = tla.arch.sub_block_idx()
        ```
    """
    _require_frontend_state("arch.sub_block_idx")
    i32 = mlir_ir.IntegerType.get_signless(32)
    value = _tla_ops_gen.arch_sub_block_idx(i32, loc=loc)
    return Int32(value)


def _require_scalar_icache_region(op_name: str) -> None:
    _require_frontend_state(op_name)
    aiv = _runtime._has_enclosing_region("vector")
    aic = _runtime._has_enclosing_region("cube")
    if aiv == aic or _runtime._has_enclosing_region("vec.func"):
        _op_error(
            op_name, "requires one scalar vector or cube region outside tla.vec.func"
        )


@dsl_user_op
def arch_icache_prefetch(prefetch_units: int, *, loc=None) -> None:
    """Directory: System Variable Access / Instruction-Cache Prefetch
    Description:
        Request instruction prefetch from the current call-site PC on Ascend 950.
    Parameters:
        - *`prefetch_units`*: Compile-time Python integer in 2 KiB units.
    Constraints:
        - Call from a scalar `tla.vector()` or `tla.cube()` region inside a kernel.
        - Vector accepts 1 to 7 units. Cube accepts 1 to 15 units.
    Example:
    ```python
    with tla.vector():
        tla.arch.icache_prefetch(2)
    ```
    """
    _require_scalar_icache_region("arch.icache_prefetch")
    limit = 7 if _runtime._has_enclosing_region("vector") else 15
    if type(prefetch_units) is not int or not 1 <= prefetch_units <= limit:
        _op_error(
            "arch.icache_prefetch",
            f"prefetch_units must be a Python int in [1, {limit}]",
        )
    _tla_ops_gen.arch_icache_prefetch(prefetch_units, loc=loc)


@dsl_user_op
def arch_get_icache_prefetch_status(*, loc=None) -> Int64:
    """Directory: System Variable Access / Instruction-Cache Prefetch
    Description:
        Observe the issuing core's instruction-prefetch status on Ascend 950.
    Parameters:
        - None.

    Returns:

        - A staged Int64 comparable to `tla.params.ICachePrefetchStatus.COMPLETE`
          (0) or `INCOMPLETE` (1). An unknown code matches neither member.
    Constraints:
        - Call from a scalar `tla.vector()` or `tla.cube()` region inside a kernel.
        - The observation returns immediately.
    Example:
    ```python
    from catlass.tla.params import ICachePrefetchStatus

    with tla.vector():
        tla.arch.icache_prefetch(2)
        status = tla.arch.get_icache_prefetch_status()
        if status == ICachePrefetchStatus.COMPLETE:
            ...
    ```
    """
    _require_scalar_icache_region("arch.get_icache_prefetch_status")
    i64 = mlir_ir.IntegerType.get_signless(64)
    return Int64(_tla_ops_gen.arch_get_icache_prefetch_status(i64, loc=loc))


def _require_arch_warp_region(op_name: str) -> None:
    _require_frontend_state(op_name)
    if not _runtime._in_simt_vec_func():
        _op_error(op_name, "is only available inside a tla.vec.func with mode='simt'")


@dsl_user_op
def arch_warp_reduce_sync(
    value: Numeric | int | float,
    kind: str = "add",
    *,
    loc: mlir_ir.Location | None = None,
) -> Numeric:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Reduce a scalar over active warp lanes, including partial warps.
        Every participating lane receives the result in the input scalar type.

        Parameters:
        - `value`: Int8/16/32, UInt8/16/32, Float16, or Float32 scalar.
        - `kind`: compile-time string `"add"`, `"max"`, or `"min"`.

        Constraints:
        - Only available in a kernel's `tla.vec.func(mode="simt")`; no cross-warp synchronization.
        - Signed/unsigned narrow integers are sign-/zero-extended to i32, then truncated back.
        - Float16 uses native f16 reduction; floating-point addition order is hardware-defined.

        Example:
        ```python
        with tla.vec.func(mode="simt", thread_block_dim=32):
            tid, _, _ = tla.arch.thread_idx()
            gm_out[tid] = tla.arch.warp_reduce_sync(gm_in[tid])
        ```
    """
    _require_arch_warp_region("arch.warp_reduce_sync")
    if not isinstance(kind, str) or kind not in ("add", "max", "min"):
        _op_error(
            "arch.warp_reduce_sync",
            "kind must be a compile-time string: add, max, or min",
        )
    num = value if isinstance(value, Numeric) else as_numeric(value)
    if type(num) not in (Int8, Int16, Int32, UInt8, UInt16, UInt32, Float16, Float32):
        _op_error(
            "arch.warp_reduce_sync", f"unsupported value type {type(num).__name__}"
        )
    operand = num.ir_value(loc=loc)
    result = _tla_ops_gen.arch_warp_reduce(
        operand.type,
        operand,
        kind,
        is_unsigned=type(num) in (UInt8, UInt16, UInt32),
        loc=loc,
    )
    return type(num)(result)


@dsl_user_op
def arch_dcache_prefetch(
    address: Pointer, byte_offset: int | Int16 | Int64 = 0, *, loc=None
) -> None:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Request the data-cache line containing the effective GM address plus
        `byte_offset` bytes on the issuing Ascend 950 scalar core.
    Parameters:
        - `address`: a typed GM pointer.
        - `byte_offset`: a signed Python integer, Int16 or Int64. Defaults to zero.
    Constraints:
        Use inside a scalar vector or cube region. The effective address must be valid.
        The hint returns no value and provides no synchronization.
        A tile view exposes its backing pointer. Include the tile displacement in bytes.
    Example:
        ```python
        with tla.vector():
            view = tla.tile_view(src, tla.make_shape(16), tla.make_coord(2))  # contiguous Int32
            tla.arch.dcache_prefetch(view.ptr, byte_offset=128)
        ```
    """
    name = "arch.dcache_prefetch"
    _require_frontend_state(name)
    if _runtime._has_enclosing_region("vec.func") or not (
        _runtime._has_enclosing_region("vector")
        or _runtime._has_enclosing_region("cube")
    ):
        _op_error(name, "requires a scalar tla.vector() or tla.cube() region")
    if (
        not isinstance(address, _Pointer)
        or address.value is None
        or address.memspace != AddressSpace.gm
    ):
        _op_error(name, "address must be a typed GM pointer")
    if isinstance(byte_offset, int) and not isinstance(byte_offset, bool):
        if not -(1 << 63) <= byte_offset < (1 << 63):
            _op_error(name, "byte_offset must fit signed i64")
        byte_offset = Int64(byte_offset)
    elif isinstance(byte_offset, Int16):
        byte_offset = byte_offset.to(Int64)
    elif not isinstance(byte_offset, Int64):
        _op_error(name, "byte_offset must be a Python integer, Int16 or Int64")
    _tla_ops_gen.arch_dcache_prefetch(
        address.value, byte_offset.ir_value(loc=loc), loc=loc
    )


# --- scalar dcache maintenance --------------------------------------------
#
# These three run on the issuing core's own scalar unit, so each is available on
# AIC and AIV alike. What they share is a placement rule: the core that issues
# the maintenance is the core whose cache it affects, so the op has to sit in a
# region that names one. A kernel body outside both regions does not: in a mixed
# kernel it is duplicated into the AIC and the AIV function, which would mean
# two cache operations where the source asked for one.
def _require_core_region(op_name: str) -> None:
    if _runtime._has_enclosing_region("vec.func"):
        _op_error(
            op_name,
            "is a scalar operation and must be placed outside tla.vec.func, "
            "directly in the tla.cube() or tla.vector() region",
        )
    in_cube = _runtime._has_enclosing_region("cube")
    in_vector = _runtime._has_enclosing_region("vector")
    if not in_cube and not in_vector:
        _op_error(op_name, "must be nested inside tla.cube() or tla.vector()")
    if in_cube and in_vector:
        _op_error(op_name, "mixed cube/vector placement is unsupported")


@dsl_user_op
def arch_dcci_cacheline(
    address: Pointer, *, loc: mlir_ir.Location | None = None
) -> None:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Clean and invalidate the one dcache line that holds `address`.

        The line is written back to global memory and then marked invalid, so a
        later scalar read of it reloads from memory. Exactly one line is
        affected -- the line that holds `address` -- and `address` need not be
        line-aligned. How much of a range a single operation covers is not
        specified here: to maintain more than one line, issue one call per
        address or use `tla.arch.dcci_cachelines`.

        This publishes one line. It is not a barrier, and it does not
        synchronize other cores: whatever flag / wait protocol the consumer
        needs is still the caller's to issue. To publish a whole domain in one
        operation use `tla.arch.dcci_cachelines`.

        Parameters:
        - `address` (`Pointer`): GM pointer to the data whose line is
          maintained. Required. The line maintained is the one holding this
          address, whatever produced the pointer.

          A tensor's `ptr` is its backing pointer, so `t.ptr` addresses element
          0 and `t.ptr + i` addresses element `i`. A view exposes the same
          backing pointer as the tensor it views rather than its own first
          element, so include the view's displacement: a `tla.tile_view` at tile
          index `c` with tile shape `n` starts at `view.ptr + c * n`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be inside `tla.cube()` or `tla.vector()`, and outside
          `tla.vec.func`.
        - `address` must be a `Pointer` in the `gm` address space. A UB-addressed
          line is a different native operation and is not available here.
        - Ascend 950 (arch 3510).

        Example:
        ```python
        with tla.vector():
            out[0] = tla.Int32(101)
            tla.arch.dcci_cacheline(out.ptr)
            # A view's first element, addressed explicitly.
            tile = tla.tile_view(out, tla.make_shape(8), tla.make_coord(3))
            tla.arch.dcci_cacheline(tile.ptr + 3 * 8)
        ```
    """
    name = "arch.dcci_cacheline"
    _require_frontend_state(name)
    _require_core_region(name)
    _require_category(name, "address", address, "pointer", 0)
    ptr = _coerce_pointer_arg(address)
    if ptr.addrspace is not AddressSpace.gm:
        _op_error(name, f"expected a gm pointer, got {ptr.addrspace}")
    byte_address = _tla_ops_gen.ptrtoint(
        mlir_ir.IntegerType.get_signless(64), ptr.value, loc=loc
    )
    _tla_ops_gen.arch_dcci_cacheline(byte_address, loc=loc)


@dsl_user_op
def arch_dcci_cachelines(
    target: Any = None, *, loc: mlir_ir.Location | None = None
) -> None:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Clean and invalidate every valid dcache line of one domain on this core.

        Each line of the selected domain is written back and then marked
        invalid. There is no address and no line count: "cachelines" is the whole
        domain, not a range.

        `tla.params.DcciTarget` selects it. `ALL` is this core's whole dcache,
        `GM_L2` the lines holding GM addresses (native OUT), and `ST_ATOMIC` the
        lines used by scalar atomic stores, which is the domain the documented
        atomic saturation transition asks for.

        Like `tla.arch.dcci_cacheline` this is cache maintenance only: it is not
        a pipe barrier and not cross-core synchronization, and `GM_L2` does not
        promise a flush of the shared L2.

        Parameters:
        - `target` (`tla.params.DcciTarget`): Domain to process, positionally or
          by keyword. Optional, default `DcciTarget.ALL`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be inside `tla.cube()` or `tla.vector()`, and outside
          `tla.vec.func`.
        - `target` is compile-time; no address or line count is accepted.
        - `DcciTarget.UB` is rejected: the native whole-UB form is AIV-only and
          depends on UB dcache being enabled through CTRL[49].
        - Ascend 950 (arch 3510).

        Example:
        ```python
        with tla.vector():
            out[0] = tla.Int32(7)
            tla.arch.dcci_cachelines(tla.params.DcciTarget.GM_L2)
        ```
    """
    _require_frontend_state("arch.dcci_cachelines")
    _require_core_region("arch.dcci_cachelines")
    selected = DcciTarget.ALL if target is None else target
    if not isinstance(selected, DcciTarget):
        _op_error(
            "arch.dcci_cachelines",
            f"invalid argument 'target' (position 0): expected tla.params.DcciTarget, "
            f"got {_type_name(target)}",
        )
    if selected is DcciTarget.UB:
        _op_error(
            "arch.dcci_cachelines",
            "target DcciTarget.UB is not available: the native whole-UB form is "
            "AIV-only and requires UB dcache enabled through CTRL[49]",
        )
    ctx = loc.context if loc is not None else mlir_ir.Context()
    attr = mlir_ir.Attribute.parse(
        f"#tla.dcci_target<{selected.mlir_token}>", context=ctx
    )
    _tla_ops_gen.arch_dcci_cachelines(attr, loc=loc)


@dsl_user_op
def arch_dci(*, loc: mlir_ir.Location | None = None) -> None:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Invalidate this core's whole dcache **without** writing it back.

        Every line is marked invalid and its dirty data is discarded, so later
        accesses reload from global memory.

        This is not a faster publication. Dirty data is lost, including scalar
        writes the kernel never named, such as compiler spills, so use it only
        where discarding those values is the intent. To publish instead, use
        `tla.arch.dcci_cachelines`, which cleans before it invalidates. Nothing
        here synchronizes peer cores, the icache or the shared L2.

        Parameters:
        - None.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be inside `tla.cube()` or `tla.vector()`, and outside
          `tla.vec.func`.
        - Ascend 950 (arch 3510) only; A2 and A3 have no such instruction.

        - Issue a `DSB_ALL` data barrier first to complete preceding memory
          accesses. TLA has no public API for this barrier.

        Example:
        ```python
        # Required sequence in a scalar region:
        # 1. Issue a DSB_ALL data barrier.
        # 2. Call tla.arch.dci().
        ```
    """
    _require_frontend_state("arch.dci")
    _require_core_region("arch.dci")
    _tla_ops_gen.arch_dci(loc=loc)


def _bypass_dcache_alignment(pointer: _Pointer) -> int:
    """Conservatively check the alignment of a pointer used by a DCache bypass access."""
    alignment = pointer.alignment
    constant_byte_offset = 0
    value = pointer.value
    while not isinstance(value.owner, mlir_ir.Block):
        producer = value.owner
        if producer.name == "tla.recast_ptr":
            value = _as_value(producer.operands[0])
            continue
        if producer.name != "tla.ptr_add":
            break
        value = _as_value(producer.operands[0])
        offset = producer.operands[1]
        pointee = PtrType(value.type).pointee
        element_bytes = _builtins.max(1, Numeric.from_mlir_type(pointee).width // 8)
        if (
            not isinstance(offset.owner, mlir_ir.Block)
            and offset.owner.name == "arith.index_cast"
        ):
            offset = offset.owner.operands[0]
        offset_producer = offset.owner
        if (
            not isinstance(offset_producer, mlir_ir.Block)
            and offset_producer.name == "arith.constant"
        ):
            constant_byte_offset += (
                element_bytes
                * mlir_ir.IntegerAttr(offset_producer.attributes["value"]).value
            )
        else:
            alignment = math.gcd(alignment, element_bytes)
    return math.gcd(alignment, constant_byte_offset)


@dsl_user_op
def arch_load_bypass_dcache(
    ptr: Pointer, *, loc: mlir_ir.Location | None = None
) -> Numeric:
    """Directory: System Variable Access / Architecture Helpers

    Description:
        Read one GM integer scalar while bypassing DCache. Each call performs
        a separate read.

    Parameters:
        - `ptr` (`Pointer`): GM pointer to a signed or unsigned 8-, 16-,
          32-, or 64-bit integer. Required.

    Returns:

    - `Numeric`: Scalar with the same type as the pointee.

    Constraints:
        - On Ascend 950, call inside `tla.cube()` or `tla.vector()` and outside
          `tla.vec.func()`.
        - The pointer must refer to a valid GM element and be naturally aligned
          for its pointee type.

    Example:
        ```python
        @tla.kernel
        def read_first(src: tla.Tensor, dst: tla.Tensor):
            with tla.cube():
                dst[0] = tla.arch.load_bypass_dcache(src.ptr)
        ```
    """
    name = "arch.load_bypass_dcache"
    _require_frontend_state(name)
    if not (
        _runtime._has_enclosing_region("cube")
        or _runtime._has_enclosing_region("vector")
    ):
        _op_error(name, "must be nested inside tla.cube() or tla.vector()")
    _require_core_region(name)
    _require_category(name, "ptr", ptr, "pointer", 0)
    pointer = _coerce_pointer_arg(ptr)
    if pointer.addrspace != AddressSpace.gm:
        _op_error(name, "ptr must point to GM")
    pointee = pointer.pointee
    if not isinstance(pointee, mlir_ir.IntegerType) or pointee.width not in (
        8,
        16,
        32,
        64,
    ):
        _op_error(name, "ptr must point to an 8-, 16-, 32-, or 64-bit integer")
    if _bypass_dcache_alignment(pointer) < pointee.width // 8:
        _op_error(name, "ptr must be naturally aligned for its pointee type")
    byte_address = _tla_ops_gen.ptrtoint(
        mlir_ir.IntegerType.get_signless(64), pointer.value, loc=loc
    )
    result = _tla_ops_gen.arch_load_bypass_dcache(pointee, byte_address, loc=loc)
    return Numeric.from_mlir_type(pointee)(result)


@dsl_user_op
def arch_store_bypass_dcache(
    ptr: Pointer, value: IndexLike, *, loc: mlir_ir.Location | None = None
) -> None:
    """Directory: System Variable Access / Architecture Helpers

    Description:
        Write one Int32 GM scalar while bypassing DCache.

    Parameters:
        - `ptr` (`Pointer`): GM pointer to the element to write.
        - `value` (`IndexLike`): Integer scalar converted to Int32.

    Constraints:
        - On Ascend 950, call inside `tla.cube()` or `tla.vector()` and outside
          `tla.vec.func()`.
        - The pointer must name a valid, naturally aligned Int32 GM element.
        - Before publishing a ready word, wait for the producing pipe when
          required, such as MTE3→SCALAR on AIV or FIX→SCALAR on AIC.

    Example:
        ```python
        @tla.kernel
        def write_first(dst: tla.Tensor):
            with tla.vector():
                tla.arch.store_bypass_dcache(dst.ptr_with_coord, 1)
        ```
    """
    name = "arch.store_bypass_dcache"
    _require_frontend_state(name)
    _require_core_region(name)
    _require_category(name, "ptr", ptr, "pointer", 0)
    pointer = _coerce_pointer_arg(ptr)
    if pointer.addrspace != AddressSpace.gm:
        _op_error(name, "ptr must point to GM")
    if pointer.dtype != Int32:
        _op_error(name, "ptr must point to an Int32 element")
    if _bypass_dcache_alignment(pointer) < 4:
        _op_error(name, "ptr must be naturally aligned for its pointee type")

    try:
        scalar = as_numeric(_resolve_bound_value(value))
    except (TypeError, ValueError):
        _op_error(name, "value must be an integer scalar")
    if not type(scalar).is_integer:
        _op_error(name, "value must be an integer scalar")

    byte_address = _tla_ops_gen.ptrtoint(
        mlir_ir.IntegerType.get_signless(64), pointer.value, loc=loc
    )
    _tla_ops_gen.arch_store_bypass_dcache(
        byte_address, scalar.to(Int32, loc=loc).ir_value(loc=loc), loc=loc
    )


# --- SIMD scalar atomics ---------------------------------------------------
#
# `tla.arch.atomic_add(ptr, value)` / `tla.arch.atomic_exch(ptr, value)` are the
# two read-modify-write primitives available to a scalar region of `tla.vector()`
# or `tla.cube()`. They are pure pointer APIs: the caller names the element
# address, the pointer's pointee type carries the element type, and the DSL
# guarantees the RMW is atomic with respect to every other core touching that
# address. Only global memory has the atom unit, so both are gm-only.
#
# The old value comes back in the pointer's own type. For the unsigned forms
# (u32/u64) that means the caller can write it straight back into a u32/u64
# tensor element.
_ATOMIC_ADD_ELEMENT_TOKENS = ("i32", "u32", "f32", "i64", "u64")
_ATOMIC_EXCH_ELEMENT_TOKENS = ("u32", "u64")


def _scalar_element_token(element_type: mlir_ir.Type) -> str:
    if mlir_ir.IntegerType.isinstance(element_type):
        integer_type = mlir_ir.IntegerType(element_type)
        return ("u" if integer_type.is_unsigned else "i") + str(integer_type.width)
    return str(element_type)


def _atomic_scalar_operand(
    op_name: str,
    element_type: mlir_ir.Type,
    value: Any,
    *,
    loc: mlir_ir.Location | None,
) -> mlir_ir.Value:
    """Coerce the addend/exchange value to exactly the pointer's pointee type."""
    resolved = _resolve_bound_value(value)
    if isinstance(resolved, bool):
        _op_error(
            op_name,
            f"invalid argument 'value' (position 1): expected scalar, got {_type_name(value)}",
        )
    if isinstance(resolved, (int, float)):
        # A host literal is range-checked and emitted as the pointee type; for an
        # unsigned pointee the literal is a signless constant bridged by the cast
        # the op's operand type requires.
        return _scalar_constant_for_element_type(
            op_name, resolved, element_type, loc=loc
        )
    if isinstance(resolved, Numeric):
        num = resolved
    elif isinstance(resolved, mlir_ir.Value):
        num = as_numeric(resolved)
    else:
        _op_error(
            op_name,
            f"invalid argument 'value' (position 1): expected Numeric or scalar "
            f"literal, got {_type_name(value)}",
        )
    try:
        dest_cls = Numeric.from_mlir_type(element_type)
    except TypeError:
        _op_error(op_name, f"unsupported pointee type {element_type}")
    if not isinstance(num, dest_cls):
        _op_error(
            op_name,
            f"type mismatch, the pointer's pointee is {dest_cls.dtype} but the value "
            f"is {type(num).dtype}; cast explicitly with .to({dest_cls.__name__})",
        )
    ir_value = num.ir_value(loc=loc)
    if str(ir_value.type) != str(element_type):
        raise TlaLoweringError(
            f"tla.{op_name} value type mismatch: expected {element_type}, got {ir_value.type}"
        )
    return ir_value


def _emit_simd_atomic(
    op_name: str,
    emitter: Any,
    element_tokens: tuple[str, ...],
    ptr: Pointer,
    value: Any,
    *,
    loc: mlir_ir.Location | None,
) -> Numeric:
    _require_frontend_state(op_name)
    _require_core_region(op_name)
    _require_category(op_name, "ptr", ptr, "pointer", 0)
    p = _coerce_pointer_arg(ptr)
    if p.addrspace is not AddressSpace.gm:
        _op_error(
            op_name,
            f"invalid argument 'ptr' (position 0): expected a gm pointer, got {p.addrspace}",
        )
    element_type = p.pointee
    token = _scalar_element_token(element_type)
    if token not in element_tokens:
        _op_error(
            op_name,
            f"unsupported pointee type {token}; supported types are "
            f"{', '.join(element_tokens)}",
        )
    operand = _atomic_scalar_operand(op_name, element_type, value, loc=loc)
    old_value = emitter(element_type, p.value, operand, loc=loc)
    if str(old_value.type) != str(element_type):
        raise TlaLoweringError(
            f"tla.{op_name} result type mismatch: expected {element_type}, got {old_value.type}"
        )
    return Numeric.from_mlir_type(element_type)(old_value)


@dsl_user_op
def arch_atomic_add(
    ptr: Pointer, value: Any, *, loc: mlir_ir.Location | None = None
) -> Numeric | None:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Atomically add `value` to the element `ptr` addresses
        (`tla.arch.atomic_add(ptr, value)`), returning the old element.

        The pointer carries both the element type and the address space. The
        caller passes the element address -- `t.ptr` is the storage base, so
        `t.ptr + i` addresses element `i`; for a view, `t.ptr_with_coord` names
        its first element -- and the read-modify-write is atomic, so any number
        of cores or threads may target the same element without losing an
        update. Typical uses are counters, histogram bins and reduction
        accumulators.

        The entry serves both scalar and SIMT code:

        - Scalar form, inside `tla.vector()` or `tla.cube()` outside any
          `tla.vec.func`: the atom unit is shared, so a mixed AIV/AIC kernel
          may target the same element from either core. The returned value is
          the element's contents before the update, in the pointer's own type:
          a `u32` pointer returns `UInt32`, a `u64` pointer returns `UInt64`.
        - SIMT form, inside `tla.vec.func(mode="simt")`: every thread issues
          its own atomic add and receives the value it replaced. The pointer
          must be built inside the region and may name GM or UB; `f16`/`bf16`
          have no usable hardware return value and return `None`.

        Parameters:
        - `ptr` (`Pointer`): pointer to the element that is updated. Required.
        - `value` (`Numeric | int | float`): Addend. A host literal is converted
          to the pointer's pointee type; a `Numeric` must already be that type.
          Required.

        Returns:
        - `Numeric | None`: the element's value before the update, typed like
          the pointee. `None` for the SIMT `f16`/`bf16` forms. Ignore it when
          only the accumulation matters.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Scalar form: inside `tla.vector()` or `tla.cube()` and not inside any
          `tla.vec.func`; `ptr` must be a `gm` pointer; pointee type must be
          `i32`, `u32`, `i64`, `u64` or `f32`.
        - SIMT form: inside `tla.vec.func(mode="simt")`; `ptr` must be in `gm`
          or `ub` and built inside the region; pointee type must be `i32`,
          `u32`, `i64`, `u64`, `f32`, `f16` or `bf16`, with the 64-bit types
          only in `gm`.
        - Ascend 950 (arch 3510).

        Example:
        ```python
        with tla.vector():
            old = tla.arch.atomic_add(out.ptr_with_coord, 1)
            out[0] = old  # same type: u32 -> u32, i32 -> i32

        with tla.vec.func(mode="simt", thread_block_dim=32):
            tid, _, _ = tla.arch.thread_idx()
            tla.arch.atomic_add(hist.ptr + tid, 1)
        ```
    """
    if _runtime._in_simt_vec_func():
        return _arch_atomic_add_simt(ptr, value, loc=loc)
    return _emit_simd_atomic(
        "atomic_add",
        _tla_ops_gen.simd_atomic_add,
        _ATOMIC_ADD_ELEMENT_TOKENS,
        ptr,
        value,
        loc=loc,
    )


@dsl_user_op
def arch_atomic_exch(
    ptr: Pointer, value: Any, *, loc: mlir_ir.Location | None = None
) -> Numeric:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Atomically replace the element `ptr` addresses with `value`
        (`tla.arch.atomic_exch(ptr, value)`), returning the old element.

        Exchange is a single atomic swap, not a compare-and-swap: the element
        becomes `value` whatever it held before, and the previous contents are
        returned. Use it to publish a slot or to claim a location without a
        retry loop. Like `tla.arch.atomic_add`, it is available from
        `tla.vector()` and `tla.cube()`.

        The returned value is the element's contents before the swap, in the
        pointer's own type: `u32` returns `UInt32`, `u64` returns `UInt64`. As
        with `tla.arch.atomic_add`, the old value can be written straight back
        into a same-type tensor.

        Parameters:
        - `ptr` (`Pointer`): GM pointer to the element that is exchanged.
          Required.
        - `value` (`Numeric | int`): Replacement value. A host literal is
          converted to the pointer's pointee type; a `Numeric` must already be
          that type. Required.

        Returns:
        - `Numeric`: the element's value before the swap, typed like the pointee.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be inside `tla.vector()` or `tla.cube()`, and must not be inside
          any `tla.vec.func`: the atom unit is a scalar instruction, so it
          cannot run as part of a vector function. Call it directly in the
          scalar part of `tla.vector()`, or anywhere in `tla.cube()`.
        - `ptr` must be a `gm` pointer; the atom unit has no UB form.
        - Pointee type must be `u32` or `u64`; the exchange instruction has no
          signed or floating-point form.
        - Ascend 950 (arch 3510).

        Example:
        ```python
        with tla.vector():
            old = tla.arch.atomic_exch(flag.ptr_with_coord, tla.UInt32(1))
        ```
    """
    return _emit_simd_atomic(
        "atomic_exch",
        _tla_ops_gen.simd_atomic_exch,
        _ATOMIC_EXCH_ELEMENT_TOKENS,
        ptr,
        value,
        loc=loc,
    )


@dsl_user_op
def arch_get_ctrl(*, start_bit=0, end_bit=63, loc=None):
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Read the local Ascend 950 CTRL register as UInt64. The default returns
        the full snapshot. A selected documented field is right-justified.
    Parameters:
        - `start_bit`, `end_bit`: first and last bits of the field, defaulting to 0 and 63.

        Returns:

        - `UInt64`: Full snapshot or right-justified field.
    Constraints:
        On Ascend 950 (arch 3510), call inside `tla.cube()` or `tla.vector()` and outside `tla.vec.func()`.
        Bit endpoints are inclusive compile-time integers.
    Example:
        ```python
        with tla.vector():
            snapshot = tla.arch.get_ctrl()
        ```
    """
    from .tla import _ctrl

    return _ctrl.get_ctrl(start_bit=start_bit, end_bit=end_bit, loc=loc)


@dsl_user_op
def arch_set_ctrl(value, *, start_bit=None, end_bit=None, loc=None):
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Update one documented field or apply a multi-field `tla.params.CTRLValue`
        patch in a single read-modify-write. Omitted fields retain their current values.
    Parameters:
        - `value`: raw Int64/UInt64 or Python integer, or a CTRLValue patch.
        - `start_bit`, `end_bit`: inclusive endpoints for the raw form.

        Returns:

        - `None`.
    Constraints:
        On Ascend 950 (arch 3510), call inside `tla.cube()` or `tla.vector()` and outside `tla.vec.func()`.
        Casts preserve the current saturation strategy [60]. Select `USE_API` explicitly to use instruction saturation.
        E4M3 matrix inputs can select E4M3 interpretation [45].
        GM copies with the default `atomic_mode=NONE` use the current CTRL atomic state.
        Set `atomic_format=AtomicFormat.DISABLED` to disable atomics enabled by this API.
        MMAD parameters also control compute order [51] and HF32 settings [46:47], which this setter does not expose.
        Raw writes require both endpoints and a fitting integer. Use a semantic patch
        without endpoints. Writes selecting bit 48 clean and invalidate the atomic cache domain
        and issue an all-pipe barrier before updating CTRL.
        The supported fields are atomic format [8:6], atomic operation [10:9],
        FP8 input interpretation [45], floating saturation [48], UB dcache [49],
        FP8 NaN saturation [50], integer saturation [53],
        conversion mode [59], and saturation strategy [60]. Raw atomic format and operation values must be compile-time
        integers. One-bit fields accept runtime values of 0 or 1. For floating and FP8 NaN saturation, semantic True encodes zero.
    Example:
        ```python
        with tla.vector():
            tla.arch.set_ctrl(tla.params.CTRLValue(integer_saturation=True))
        ```
    """
    from .tla import _ctrl

    _ctrl.set_ctrl(value, start_bit=start_bit, end_bit=end_bit, loc=loc)


@dsl_user_op
def arch_warpsize(*, loc: mlir_ir.Location | None = None) -> Int32:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return the SIMT warp size as the `Int32` constant 32.

        A warp groups 32 consecutive threads in x-fastest linear order.
        A final partial warp does not change this constant.

        Parameters:
        - None.

        Returns:

        - `Int32`: the constant 32.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Only available inside `tla.vec.func(mode="simt")`.

        Example:
        ```python
        with tla.vec.func(mode="simt", thread_block_dim=48):
            tid, _, _ = tla.arch.thread_idx()
            gm_out[tid] = tla.arch.warpsize()
        ```
    """
    _require_arch_warp_region("arch.warpsize")
    return Int32(32)


@dsl_user_op
def arch_lane_idx(*, loc: mlir_ir.Location | None = None) -> Int32:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return the calling thread's lane index within its warp as `Int32`.

        The index is in `[0, 31]` and equals
        `(x + y * BX + z * BX * BY) % 32`, where `(x, y, z)` is the
        thread index and `(BX, BY, BZ)` is `thread_block_dim`.
        Branches do not renumber lanes. This query does not synchronize threads.

        Parameters:
        - None.

        Returns:

        - `Int32`: the calling thread's lane index.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Only available inside `tla.vec.func(mode="simt")`.

        Example:
        ```python
        with tla.vec.func(mode="simt", thread_block_dim=48):
            tid, _, _ = tla.arch.thread_idx()
            gm_out[tid] = tla.arch.lane_idx()
        ```
    """
    _require_arch_warp_region("arch.lane_idx")
    i32 = mlir_ir.IntegerType.get_signless(32)
    return Int32(_tla_ops_gen.arch_lane_idx(i32, loc=loc))


@dsl_user_op
def arch_activemask(*, loc: mlir_ir.Location | None = None) -> UInt32:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return the currently active lanes in this SIMT warp as `UInt32`.
        Bit `j` is set exactly when lane `j` executes this instruction.
        Unlaunched lanes in a partial warp contribute zero. Divergent branches
        observe their own active set. Unlike `lanemask_*`, the result depends
        on activity at the call site.
        Parameters:
        - None.

        Returns:

        - `UInt32`: The current active lane mask.
    Constraints:
        - Only inside `tla.vec.func(mode="simt")` in a `@tla.kernel`.
        - Supported on Ascend 950 only; no explicit member mask.
        - Does not provide cross-warp synchronization. The compiler omits unused queries.
        - Reports actual instruction activity, not source-predicate membership:
          the backend may reconverge identical branch bodies. Do not use this
          snapshot as a synchronization guarantee or a persistent member mask.
    Example:
        with tla.vec.func(mode="simt", thread_block_dim=(48, 1, 1)):
            tid, _, _ = tla.arch.thread_idx()
            gm_out[tid] = tla.arch.activemask()
    """
    _require_arch_warp_region("arch.activemask")
    i32 = mlir_ir.IntegerType.get_signless(32)
    return UInt32(_tla_ops_gen.arch_activemask(i32, loc=loc))


@dsl_user_op
def arch_vote_ballot_sync(
    predicate: Any, *, loc: mlir_ir.Location | None = None
) -> UInt32:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return a `UInt32` mask with bit `j` set when executing lane `j` has a
        nonzero predicate. Only currently executing lanes participate; inactive
        or unlaunched lanes contribute zero. Integer predicates are compared
        against zero in their original width before conversion to `Bool`.
        Parameters:
        - `predicate` (`Bool | Integer | bool | int`): Per-thread predicate.
    Constraints:
        - Only inside `tla.vec.func(mode="simt")` in a `@tla.kernel`.
        - Ascend 950 only. No explicit member mask or cross-warp synchronization.
        - The `_sync` name does not provide a memory barrier or force source
          branches to remain divergent. Equivalent branch bodies may reconverge.
        - Floating-point values must be explicitly compared to produce `Bool`.
    Example:
        with tla.vec.func(mode="simt", thread_block_dim=(48, 1, 1)):
            tid, _, _ = tla.arch.thread_idx()
            gm_out[tid] = tla.arch.vote_ballot_sync(tid % 2 == 0)
    """
    pred = _as_arch_vote_predicate("arch.vote_ballot_sync", predicate)
    i32 = mlir_ir.IntegerType.get_signless(32)
    return UInt32(_tla_ops_gen.arch_vote_ballot(i32, pred.ir_value(loc=loc), loc=loc))


def _as_arch_vote_predicate(op_name: str, predicate: Any) -> Bool:
    _require_arch_warp_region(op_name)
    if isinstance(predicate, (bool, int)):
        return Bool(predicate != 0)
    if isinstance(predicate, Bool):
        return predicate
    if isinstance(predicate, _Integer):
        return predicate != type(predicate)(0)
    _op_error(
        op_name,
        f"predicate must be a Bool or integer scalar; got {_type_name(predicate)}",
    )


@dsl_user_op
def arch_vote_all_sync(predicate: Any, *, loc: mlir_ir.Location | None = None) -> Bool:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return `Bool` indicating whether all currently executing lanes have
        a nonzero predicate. Integer predicates are compared with zero at their
        original width. Inactive or unlaunched lanes do not participate.
        Parameters:
        - `predicate` (`Bool | Integer | bool | int`): Per-thread predicate.
    Constraints:
        - Only inside `tla.vec.func(mode="simt")` in a `@tla.kernel`.
        - Ascend 950 only. No explicit member mask or cross-warp synchronization.
        - The `_sync` name does not provide a memory barrier or force source
          branches to remain divergent. Equivalent branch bodies may reconverge.
        - Floating-point values must be explicitly compared to produce `Bool`.
    Example:
        with tla.vec.func(mode="simt", thread_block_dim=48):
            tid, _, _ = tla.arch.thread_idx()
            gm_out[tid] = tla.arch.vote_all_sync(tid % 2 == 0).to(tla.Int32)
    """
    pred = _as_arch_vote_predicate("arch.vote_all_sync", predicate)
    i1 = mlir_ir.IntegerType.get_signless(1)
    return Bool(_tla_ops_gen.arch_vote_all(i1, pred.ir_value(loc=loc), loc=loc))


@dsl_user_op
def arch_vote_any_sync(predicate: Any, *, loc: mlir_ir.Location | None = None) -> Bool:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return `Bool` indicating whether any currently executing lanes have
        a nonzero predicate. Integer predicates are compared with zero at their
        original width. Inactive or unlaunched lanes do not participate.
        Parameters:
        - `predicate` (`Bool | Integer | bool | int`): Per-thread predicate.
    Constraints:
        - Only inside `tla.vec.func(mode="simt")` in a `@tla.kernel`.
        - Ascend 950 only. No explicit member mask or cross-warp synchronization.
        - The `_sync` name does not provide a memory barrier or force source
          branches to remain divergent. Equivalent branch bodies may reconverge.
        - Floating-point values must be explicitly compared to produce `Bool`.
    Example:
        with tla.vec.func(mode="simt", thread_block_dim=48):
            tid, _, _ = tla.arch.thread_idx()
            gm_out[tid] = tla.arch.vote_any_sync(tid % 2 == 0).to(tla.Int32)
    """
    pred = _as_arch_vote_predicate("arch.vote_any_sync", predicate)
    i1 = mlir_ir.IntegerType.get_signless(1)
    return Bool(_tla_ops_gen.arch_vote_any(i1, pred.ir_value(loc=loc), loc=loc))


def _emit_arch_lanemask(op_name: str, mode: str, *, loc=None) -> UInt32:
    _require_arch_warp_region(op_name)
    i32 = mlir_ir.IntegerType.get_signless(32)
    return UInt32(_tla_ops_gen.arch_lanemask(i32, mode, loc=loc))


@dsl_user_op
def arch_lanemask_eq(*, loc: mlir_ir.Location | None = None) -> UInt32:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return a `UInt32` mask with bit `j` set when `j == lane_idx()`.
        Positions 0–31 include inactive/unlaunched lanes; no synchronization.
    Parameters:
        - None.
    Constraints:
        - Only inside `tla.vec.func(mode="simt")` in a `@tla.kernel`.
    Example:
        with tla.vec.func(mode="simt", thread_block_dim=32):
            mask = tla.arch.lanemask_eq()
    """
    return _emit_arch_lanemask("arch.lanemask_eq", "eq", loc=loc)


@dsl_user_op
def arch_lanemask_lt(*, loc: mlir_ir.Location | None = None) -> UInt32:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return a `UInt32` mask with bit `j` set when `j < lane_idx()`.
        Positions 0–31 include inactive/unlaunched lanes; no synchronization.
    Parameters:
        - None.
    Constraints:
        - Only inside `tla.vec.func(mode="simt")` in a `@tla.kernel`.
    Example:
        with tla.vec.func(mode="simt", thread_block_dim=32):
            mask = tla.arch.lanemask_lt()
    """
    return _emit_arch_lanemask("arch.lanemask_lt", "lt", loc=loc)


@dsl_user_op
def arch_lanemask_le(*, loc: mlir_ir.Location | None = None) -> UInt32:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return a `UInt32` mask with bit `j` set when `j <= lane_idx()`.
        Positions 0–31 include inactive/unlaunched lanes; no synchronization.
    Parameters:
        - None.
    Constraints:
        - Only inside `tla.vec.func(mode="simt")` in a `@tla.kernel`.
    Example:
        with tla.vec.func(mode="simt", thread_block_dim=32):
            mask = tla.arch.lanemask_le()
    """
    return _emit_arch_lanemask("arch.lanemask_le", "le", loc=loc)


@dsl_user_op
def arch_lanemask_gt(*, loc: mlir_ir.Location | None = None) -> UInt32:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return a `UInt32` mask with bit `j` set when `j > lane_idx()`.
        Positions 0–31 include inactive/unlaunched lanes; no synchronization.
    Parameters:
        - None.
    Constraints:
        - Only inside `tla.vec.func(mode="simt")` in a `@tla.kernel`.
    Example:
        with tla.vec.func(mode="simt", thread_block_dim=32):
            mask = tla.arch.lanemask_gt()
    """
    return _emit_arch_lanemask("arch.lanemask_gt", "gt", loc=loc)


@dsl_user_op
def arch_lanemask_ge(*, loc: mlir_ir.Location | None = None) -> UInt32:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return a `UInt32` mask with bit `j` set when `j >= lane_idx()`.
        Positions 0–31 include inactive/unlaunched lanes; no synchronization.
    Parameters:
        - None.
    Constraints:
        - Only inside `tla.vec.func(mode="simt")` in a `@tla.kernel`.
    Example:
        with tla.vec.func(mode="simt", thread_block_dim=32):
            mask = tla.arch.lanemask_ge()
    """
    return _emit_arch_lanemask("arch.lanemask_ge", "ge", loc=loc)


@dsl_user_op
def arch_thread_idx(
    *, loc: mlir_ir.Location | None = None
) -> tuple[Int32, Int32, Int32]:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return the calling thread's index `(x, y, z)` inside its thread block.

        Each component is an `Int32` in `[0, thread_block_dim[i])`. Only the
        components a kernel actually uses are materialized, so unpacking the
        unused ones into `_` costs nothing. This index is what makes one
        thread's work different from another's: a thread normally derives the
        elements it owns from it, for example by walking
        `tla.range(tid, n, nthreads)`.

        Parameters:
        - None.

        Returns:

        - `tuple[Int32, Int32, Int32]`: the thread's `(x, y, z)` index.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Only available inside `tla.vec.func(mode="simt")`. For the block
          (AI Core) index of the launch use `tla.arch.block_idx()`.

        Example:
        ```python
        with tla.vec.func(mode="simt", thread_block_dim=256):
            tid, _, _ = tla.arch.thread_idx()
            gm_out[tid] = gm_in[tid]
        ```
    """
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
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Barrier across the threads of the enclosing SIMT `tla.vec.func`.

        Every thread of the block must reach this point before any of them
        proceeds, and writes issued before it are visible to the other threads
        of the block afterwards. It orders threads within one block only; it is
        not a cross-core barrier and not a pipe barrier -- for those see the
        flag and `tla.pipe_barrier` APIs under Sync Control.

        The typical use is a two-phase reduction: each thread writes its partial
        result to a shared UB tensor, all threads meet at the barrier, then one
        thread combines the partials.

        Parameters:
        - None.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Only available inside `tla.vec.func(mode="simt")`.
        - Must be reached by every thread of the block; placing it under a
          condition that only some threads take will hang the kernel.

        Example:
        ```python
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            ub_partial[tid] = my_partial
            tla.arch.sync_threads()
            if tid == 0:
                ...  # thread 0 combines ub_partial[0:THREADS]
        ```
    """
    _require_frontend_state("arch.sync_threads")
    if not _runtime._in_simt_vec_func():
        _op_error(
            "arch.sync_threads",
            "is only available inside a tla.vec.func with mode='simt'",
        )
    _tla_ops_gen.arch_sync_threads(loc=loc)


# --- the AR special register ------------------------------------------------
#
# AR counts the bytes the compressing vector stores have written. It is a
# vector-unit register, so both operations are AIV-only: a `tla.cube()`
# placement has no AR to name, and the bitcode helpers behind them are linked
# into the vector meta-op bitcode alone.
#
# The two have different placement rules because the two instructions do. The
# read runs on the scalar unit and is written outside `tla.vec.func`, after the
# vector sequence that produced the count. The clear writes a vector-unit
# register and belongs inside the SIMD `tla.vec.func` that holds the stores,
# which is where the native API documents it; the scalar placement is supported
# as well, and the helper behind it opens a vector function of its own.
def _require_aiv_region(op_name: str, *, inside_vec_func: bool) -> None:
    if _runtime._has_enclosing_region("cube"):
        _op_error(
            op_name,
            "uses the vector unit's AR register and is unavailable on the AIC; "
            "place it inside tla.vector()",
        )
    if not _runtime._has_enclosing_region("vector"):
        _op_error(op_name, "must be nested inside tla.vector()")
    in_vec_func = _runtime._has_enclosing_region("vec.func")
    if in_vec_func and not inside_vec_func:
        _op_error(
            op_name,
            "runs on the scalar unit and must be placed outside tla.vec.func, "
            "directly in the tla.vector() region",
        )
    if in_vec_func and _runtime._in_simt_vec_func():
        _op_error(op_name, "is unavailable inside tla.vec.func(mode='simt')")


@dsl_user_op
def arch_get_ar(*, loc: mlir_ir.Location | None = None) -> Int64:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Read this AIV's AR special register (`Int64`).

        AR holds the byte count from a compressed store sequence that uses
        `tla.squeeze(..., record_nbytes=True)` and
        `POST_MODE_UPDATE_NBYTES`.

        The read runs on the scalar unit and observes the register as it stands.
        It does not wait for the vector pipe, so the handoff that completes the
        producing sequence is still the caller's to issue, exactly as for a UB
        value the same sequence wrote.

        Parameters:
        - None.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be inside `tla.vector()`, and outside `tla.vec.func`.
        - AIV only; `tla.cube()` has no AR.
        - Ascend 950 (arch 3510).

        Example:
        ```python
        count_ready = tla.flag("count_ready", tla.pipes.VECTOR, tla.pipes.SCALAR)
        with tla.vector():
            with tla.vec.func(mode="simd"):
                tla.arch.clear_ar()
                ...  # the compression sequence that records bytes in AR
            tla.set_flag(count_ready)
            tla.wait_flag(count_ready)
            count_bytes = tla.arch.get_ar()
        ```
    """
    _require_frontend_state("arch.get_ar")
    _require_aiv_region("arch.get_ar", inside_vec_func=False)
    i64 = mlir_ir.IntegerType.get_signless(64)
    return Int64(_tla_ops_gen.arch_get_ar(i64, loc=loc))


@dsl_user_op
def arch_clear_ar(*, loc: mlir_ir.Location | None = None) -> None:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Clear this AIV's AR special register to zero.

        Clear the register before a compression sequence to start the count at
        zero.

        Write it inside the SIMD `tla.vec.func` that holds those stores: the
        clear then sits in the same vector function as the instructions that
        populate AR, which is the placement the native API documents. A scalar
        placement directly in `tla.vector()` is accepted too and clears the
        same register; the vector function it needs is supplied for it.

        Clearing AR is not synchronization and does not wait for an outstanding
        sequence. A clear issued while one is still accumulating loses its
        count.

        Parameters:
        - None.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be inside `tla.vector()`, either directly or inside a
          `tla.vec.func(mode="simd")` nested in it.
        - AIV only; `tla.cube()` has no AR. `tla.vec.func(mode="simt")` is
          rejected.
        - Ascend 950 (arch 3510).

        Example:
        ```python
        with tla.vector():
            with tla.vec.func(mode="simd"):
                tla.arch.clear_ar()
                ...  # the compression sequence that records bytes in AR
        ```
    """
    _require_frontend_state("arch.clear_ar")
    _require_aiv_region("arch.clear_ar", inside_vec_func=True)
    _tla_ops_gen.arch_clear_ar(loc=loc)


# A shuffle moves a bit pattern between registers without interpreting it, so
# every integer and float width the intrinsics cover is supported, signed and
# unsigned alike: inside a SIMT region a Numeric presents its value as a
# signless integer of its own width, and the Python type is carried through to
# the result unchanged. `Bool` is the one integer type left out -- it is i1, and
# there is no 1-bit shuffle to widen it to.
_WARP_SHUFFLE_TYPES = (
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    BFloat16,
)

_WARP_SHUFFLE_REJECTED_TYPES = (Bool,)

# A warp is this many lanes. It is also the domain of every shuffle index: the
# hardware keeps the low 5 bits of the control field's index and discards the
# rest, so 32 is where a value stops meaning what the caller wrote.
_WARP_SIZE = 32

_SHUFFLE_INDEX_NAME = {
    "idx": "src_lane",
    "up": "delta",
    "down": "delta",
    "bfly": "lane_mask",
}


@dsl_user_op
def _emit_arch_shuffle(
    api_name: str,
    mode: str,
    value: Any,
    index: Any,
    width: Any,
    *,
    loc: mlir_ir.Location | None = None,
) -> Any:
    """Emit tla.arch.shuffle; the four public wrappers differ only in `mode`."""
    _require_arch_warp_region(api_name)

    operand = value if isinstance(value, Numeric) else as_numeric(value)
    if isinstance(operand, _WARP_SHUFFLE_REJECTED_TYPES):
        _op_error(
            api_name,
            "does not support Bool operands; there is no 1-bit shuffle, so "
            "shuffle an Int8 or wider value instead",
        )
    if not isinstance(operand, _WARP_SHUFFLE_TYPES):
        _op_error(
            api_name,
            f"does not support {type(operand).__name__} values; supported types are "
            + ", ".join(t.__name__ for t in _WARP_SHUFFLE_TYPES),
        )

    if not isinstance(width, int) or isinstance(width, bool):
        _op_error(api_name, f"width must be a Python int constant; got {width!r}")
    if width < 2 or width > 32 or (width & (width - 1)) != 0:
        _op_error(api_name, f"width must be a power of two in [2, 32]; got {width}")

    # The index is a lane number, not data: it is always i32, whatever the value
    # type is.
    lane = index if isinstance(index, Numeric) else as_numeric(index)
    if not isinstance(lane, Int32):
        lane = lane.to(Int32) if isinstance(index, Numeric) else Int32(lane)

    if isinstance(index, int) and not isinstance(index, bool):
        if index < 0:
            _op_error(
                api_name,
                f"{_SHUFFLE_INDEX_NAME[mode]} must be non-negative; got {index}",
            )
        # The hardware carries the index in a 5-bit field, so 32 is where the
        # value starts being lost: 33 becomes 1 and reads a neighbour instead.
        # Below 32 nothing is dropped, and a distance or mask that reaches
        # outside the lane's group is answered by the hardware keeping the
        # lane's own value, which is a legal request rather than an error.
        # "idx" is exempt because there the index is a lane number and the
        # masking is exactly the documented src_lane % width.
        if mode != "idx" and index >= _WARP_SIZE:
            _op_error(
                api_name,
                f"{_SHUFFLE_INDEX_NAME[mode]} must be less than {_WARP_SIZE}; "
                f"got {index}",
            )

    ctx = loc.context if loc is not None else mlir_ir.Context.current
    value_ir = operand.ir_value(loc=loc)
    result = mlir_ir.Operation.create(
        "tla.arch.shuffle",
        operands=[value_ir, lane.ir_value(loc=loc)],
        results=[value_ir.type],
        attributes={
            "mode": mlir_ir.Attribute.parse(f"#tla.shuffle_mode<{mode}>", context=ctx),
            "width": mlir_ir.IntegerAttr.get(
                mlir_ir.IntegerType.get_signless(32, ctx), width
            ),
        },
        loc=loc,
    ).results[0]
    return type(operand)(result)


@dsl_user_op
def arch_shuffle_sync(
    value: Any, src_lane: Any, width: int = 32, *, loc: mlir_ir.Location | None = None
) -> Any:
    """Directory: System Variable Access / Warp Functions
    Description:
        Return the `value` held by lane `src_lane` of the calling thread's warp.

        The source lane is `src_lane % width` within the group, so pointing
        every lane at the same source broadcasts one lane's value to the group.

        Parameters:
        - `value`: the value to publish to the other lanes. Supported types are
          `Int8`/`Int16`/`Int32`/`Int64`, their unsigned forms, `Float16`,
          `Float32` and `BFloat16`; the result keeps the type of `value`.
          `Bool` is not supported -- it is `i1`, and there is no 1-bit shuffle
          to widen it to.
        - `src_lane`: `Int32`, the lane to read from, in `[0, 32)`. A negative
          constant is rejected at compile time. A larger value is a lane number
          taken modulo `width`, as above, whether it is a constant or a runtime
          value; a negative runtime value is the caller's responsibility and is
          masked to its low 5 bits rather than diagnosed.
        - `width`: a Python `int`, a power of two in `[2, 32]`. Optional,
          default `32` (the whole warp).

        Returns:

        - the shuffled value, of the same type as `value`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Only available inside `tla.vec.func(mode="simt")`.
        - Must be reached by every thread of the warp that participates. A
          thread held back by a branch does not supply its value, and reading
          from an inactive lane gives an undefined result.

        Example:
        ```python
        with tla.vec.func(mode="simt", thread_block_dim=256):
            # every lane of the warp ends up with lane 0's value
            leader = tla.arch.shuffle_sync(my_value, 0)
        ```
    """
    _require_frontend_state("arch.shuffle_sync")
    return _emit_arch_shuffle(
        "arch.shuffle_sync", "idx", value, src_lane, width, loc=loc
    )


@dsl_user_op
def arch_shuffle_sync_up(
    value: Any, delta: Any, width: int = 32, *, loc: mlir_ir.Location | None = None
) -> Any:
    """Directory: System Variable Access / Warp Functions
    Description:
        Return the `value` held by the lane `delta` positions below this one.

        The source lane is `laneid - delta`. A lane whose source falls below the
        start of its group keeps its own value, which is what makes this the
        right primitive for an inclusive prefix sum.

        Parameters:
        - `value`: the value to publish to the other lanes. Supported types are
          `Int8`/`Int16`/`Int32`/`Int64`, their unsigned forms, `Float16`,
          `Float32` and `BFloat16`; the result keeps the type of `value`.
          `Bool` is not supported -- it is `i1`, and there is no 1-bit shuffle
          to widen it to.
        - `delta`: `Int32`, how many lanes down to reach, in `[0, 32)`. A
          `delta` that reaches below the start of the lane's group is allowed:
          that lane keeps its own value. May be a runtime value; a constant outside
          `[0, 32)` is rejected at compile time, while a runtime value is the
          caller's responsibility and is masked to its low 5 bits rather than
          diagnosed, so a runtime `delta` of 33 reads the neighbour at distance
          1.
        - `width`: a Python `int`, a power of two in `[2, 32]`. Optional,
          default `32` (the whole warp).

        Returns:

        - the shuffled value, of the same type as `value`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Only available inside `tla.vec.func(mode="simt")`.
        - Must be reached by every thread of the warp that participates. A
          thread held back by a branch does not supply its value, and reading
          from an inactive lane gives an undefined result.
        - `delta` must be in `[0, 32)`. A constant outside that range is
          rejected at compile time; a runtime value is the caller's
          responsibility.

        Example:
        ```python
        with tla.vec.func(mode="simt", thread_block_dim=256):
            # Kogge-Stone inclusive scan across the warp
            lane, _, _ = tla.arch.thread_idx()
            total = my_value
            for offset in (1, 2, 4, 8, 16):
                other = tla.arch.shuffle_sync_up(total, offset)
                total = tla.where(lane % 32 >= offset, total + other, total)
        ```
    """
    _require_frontend_state("arch.shuffle_sync_up")
    return _emit_arch_shuffle(
        "arch.shuffle_sync_up", "up", value, delta, width, loc=loc
    )


@dsl_user_op
def arch_shuffle_sync_down(
    value: Any, delta: Any, width: int = 32, *, loc: mlir_ir.Location | None = None
) -> Any:
    """Directory: System Variable Access / Warp Functions
    Description:
        Return the `value` held by the lane `delta` positions above this one.

        The source lane is `laneid + delta`. A lane whose source falls past the
        end of its group keeps its own value. Halving `delta` from 16 down to 1
        leaves the whole-warp reduction in lane 0.

        Parameters:
        - `value`: the value to publish to the other lanes. Supported types are
          `Int8`/`Int16`/`Int32`/`Int64`, their unsigned forms, `Float16`,
          `Float32` and `BFloat16`; the result keeps the type of `value`.
          `Bool` is not supported -- it is `i1`, and there is no 1-bit shuffle
          to widen it to.
        - `delta`: `Int32`, how many lanes up to reach, in `[0, 32)`. A
          `delta` that reaches past the end of the lane's group is allowed:
          that lane keeps its own value. May be a runtime value; a constant outside
          `[0, 32)` is rejected at compile time, while a runtime value is the
          caller's responsibility and is masked to its low 5 bits rather than
          diagnosed, so a runtime `delta` of 33 reads the neighbour at distance
          1.
        - `width`: a Python `int`, a power of two in `[2, 32]`. Optional,
          default `32` (the whole warp).

        Returns:

        - the shuffled value, of the same type as `value`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Only available inside `tla.vec.func(mode="simt")`.
        - Must be reached by every thread of the warp that participates. A
          thread held back by a branch does not supply its value, and reading
          from an inactive lane gives an undefined result.
        - `delta` must be in `[0, 32)`. A constant outside that range is
          rejected at compile time; a runtime value is the caller's
          responsibility.

        Example:
        ```python
        with tla.vec.func(mode="simt", thread_block_dim=256):
            total = my_value
            for offset in (16, 8, 4, 2, 1):
                total = total + tla.arch.shuffle_sync_down(total, offset)
            # lane 0 of each warp now holds the warp's sum
        ```
    """
    _require_frontend_state("arch.shuffle_sync_down")
    return _emit_arch_shuffle(
        "arch.shuffle_sync_down", "down", value, delta, width, loc=loc
    )


@dsl_user_op
def arch_shuffle_sync_bfly(
    value: Any, lane_mask: Any, width: int = 32, *, loc: mlir_ir.Location | None = None
) -> Any:
    """Directory: System Variable Access / Warp Functions
    Description:
        Return the `value` held by the lane whose id is this one's XOR
        `lane_mask`.

        The source lane is `laneid ^ lane_mask`, a butterfly exchange. Unlike
        the `down` form it leaves the result in every lane, so a reduction built
        from it needs no follow-up broadcast.

        Parameters:
        - `value`: the value to publish to the other lanes. Supported types are
          `Int8`/`Int16`/`Int32`/`Int64`, their unsigned forms, `Float16`,
          `Float32` and `BFloat16`; the result keeps the type of `value`.
          `Bool` is not supported -- it is `i1`, and there is no 1-bit shuffle
          to widen it to.
        - `lane_mask`: `Int32`, the bits to flip in the lane id, in `[0, 32)`.
          The group boundary is enforced in one direction only, so a mask larger
          than `width` is allowed and the two directions differ: a lane whose
          source falls past the end of its group keeps its own value, while a
          lane whose source falls in a lower-numbered group reads that lane. A
          butterfly confined to its group therefore needs `lane_mask < width`,
          which is the caller's choice rather than a rejected one. May be a
          runtime value; a constant outside `[0, 32)` is rejected at compile
          time, while a runtime value is the caller's responsibility and is
          masked to its low 5 bits rather than diagnosed.
        - `width`: a Python `int`, a power of two in `[2, 32]`. Optional,
          default `32` (the whole warp).

        Returns:

        - the shuffled value, of the same type as `value`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Only available inside `tla.vec.func(mode="simt")`.
        - Must be reached by every thread of the warp that participates. A
          thread held back by a branch does not supply its value, and reading
          from an inactive lane gives an undefined result.
        - `lane_mask` must be in `[0, 32)`. A constant outside that range is
          rejected at compile time; a runtime value is the caller's
          responsibility.

        Example:
        ```python
        with tla.vec.func(mode="simt", thread_block_dim=256):
            total = my_value
            for mask in (16, 8, 4, 2, 1):
                total = total + tla.arch.shuffle_sync_bfly(total, mask)
            # every lane of the warp now holds the warp's sum
        ```
    """
    _require_frontend_state("arch.shuffle_sync_bfly")
    return _emit_arch_shuffle(
        "arch.shuffle_sync_bfly", "bfly", value, lane_mask, width, loc=loc
    )


@dsl_user_op
def arch_sub_block_num(*, loc: mlir_ir.Location | None = None) -> Int64:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return the configured count of this core type per logical AI Core.
        In separated mode each side of a MIX kernel evaluates its own i64 result.

        Parameters:
        - None.

        Constraints:
        - Call in kernel scalar code, including cube and vector scopes. Compute
          outside SIMD and SIMT `tla.vec.func` regions and pass the scalar in.

        Example:
        ```python
        value = tla.arch.sub_block_num()  # tla.Int64
        ```
    """
    _require_frontend_state("arch.sub_block_num")
    if _runtime._has_enclosing_region("vec.func"):
        raise TlaIRNotExecutableError(
            "tla.arch.sub_block_num: compute outside tla.vec.func and pass the scalar into the region"
        )
    i64 = mlir_ir.IntegerType.get_signless(64)
    return Int64(_tla_ops_gen.arch_sub_block_num(i64, loc=loc))


@dsl_user_op
def arch_block_num(*, loc: mlir_ir.Location | None = None) -> Int32:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return the number of blocks (AI cores) in the launch (`Int32`).

        For the per-block thread extents inside a SIMT region, see
        `tla.arch.thread_block_dim`.

        Parameters:
        - None.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.

        Example:
        ```python
        nblocks = tla.arch.block_num()
        ```
    """
    _require_frontend_state("arch.block_num")
    i32 = mlir_ir.IntegerType.get_signless(32)
    value = _tla_ops_gen.arch_block_num(i32, loc=loc)
    return Int32(value)


@dsl_user_op
def arch_get_system_cycle(*, loc: mlir_ir.Location | None = None) -> Int64:
    """Directory: System Variable Access / Counters

    Description:
        Read the issuing core's 64-bit system counter as an Int64 bit pattern.
        This volatile Scalar-pipeline read inserts no synchronization.

    Parameters:
        loc: Optional MLIR source location.

    Constraints:
        Legal inside Cube or Vector scope, outside vec.func. Compare reads
        only on the same physical issuer, with subtraction modulo 2**64.
        Int64 subtraction preserves the bits without unsigned casts. Raw
        ticks are not active-pipe cycles; frequency is target-specific.

    Example:
        ```python
        with tla.vector():
            before = tla.arch.get_system_cycle()
            after = tla.arch.get_system_cycle()
            ticks = after - before
        ```
    """
    api = "arch.get_system_cycle"
    _require_frontend_state(api)
    _runtime._require_enclosing_cube_or_vector(api)
    if _runtime._has_enclosing_region("vec.func"):
        _op_error(api, "must execute on Scalar outside tla.vec.func")
    i64 = mlir_ir.IntegerType.get_signless(64)
    return Int64(_tla_ops_gen.arch_get_system_cycle(i64, loc=loc))


@dsl_user_op
def arch_thread_block_dim(
    *, loc: mlir_ir.Location | None = None
) -> tuple[Int32, Int32, Int32]:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return the enclosing thread block's `(x, y, z)` extents.

        These are the values the region was declared with in
        `tla.vec.func(thread_block_dim=...)`, read back as `Int32` so a thread
        can use them in index arithmetic -- most often as the stride of the loop
        over the elements it owns.

        This is the *thread* geometry of one block, not the grid: the number of
        blocks (AI Cores) in the launch is `tla.arch.block_num()`.

        Parameters:
        - None.

        Returns:

        - `tuple[Int32, Int32, Int32]`: the block's `(x, y, z)` thread extents.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Only available inside `tla.vec.func(mode="simt")`.

        Example:
        ```python
        with tla.vec.func(mode="simt", thread_block_dim=256):
            tid, _, _ = tla.arch.thread_idx()
            nthreads, _, _ = tla.arch.thread_block_dim()
            for i in tla.range(tid, N, nthreads):
                gm_out[i] = gm_in[i]
        ```
    """
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


def arch_get_vec_len_in_bytes() -> int:
    """Directory: System Variable Access / Architecture Helpers

    Description:
        Return the vector register length in bytes as a Python `int`.

    Parameters:
        None.

    Constraints:
        - Returns 256 bytes for the Ascend 950 (3510) compile target.
        - Can be called on the host or evaluated as a constant during kernel tracing.

    Example:
        ```python
        import catlass.tla as tla

        vector_bytes = tla.arch.get_vec_len_in_bytes()
        ```
    """
    return _VECTOR_REGISTER_BYTES


def arch_get_datablock_size_in_bytes() -> int:
    """Directory: System Variable Access / Architecture Helpers

    Description:
        Return the size of one DataBlock in bytes as a Python `int`.

    Parameters:
        None.

    Constraints:
        - Returns 32 bytes for the Ascend 950 (3510) compile target.
        - Can be called on the host or evaluated as a constant during kernel tracing.

    Example:
        ```python
        import catlass.tla as tla

        datablock_bytes = tla.arch.get_datablock_size_in_bytes()
        ```
    """
    return 32


def arch_get_capacity_in_bytes(mem_scope: AddressSpace) -> int:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return the byte capacity of an on-chip memory space for the compile
        target.

        Parameters:
        - `mem_scope` (`AddressSpace`): On-chip space to query; see
          **System Variable Access / On-chip Address Spaces**. Required.

        Constraints:
        - Only on-chip `AddressSpace` members; `gm` / `generic` are rejected.
        - Host-time constant: legal on host and inside a `@tla.kernel`. Inside a
          kernel it folds at trace time (no runtime op).

        Example:
        ```python
        l1_bytes = tla.arch.get_capacity_in_bytes(tla.AddressSpace.l1)
        ub_bytes = tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub)
        ```
    """
    from .base_dsl.arch import get_localmem_capacity_bytes

    _ON_CHIP_SCOPES: tuple[AddressSpace, ...] = (
        AddressSpace.l1,
        AddressSpace.l0a,
        AddressSpace.l0b,
        AddressSpace.l0c,
        AddressSpace.ub,
        AddressSpace.biasbuf,
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
        - `mem_scope` (`AddressSpace`): On-chip address space; see
          **System Variable Access / On-chip Address Spaces**. Required.
        - `byte_alignment` (`int`): Byte alignment requirement. Required. Must be
          a whole multiple of the space's local-memory alignment; a smaller or
          non-multiple value is refused.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - `mem_scope` must be on-chip (`l1` / `l0a` / `l0b` / `l0c` / `ub` /
          `biasbuf`), not `gm` / `generic`; `shape` must be fully static.
        - `byte_alignment` must be a whole multiple of the space's local-memory
          alignment, since a buffer cannot start mid-block: 32 bytes for `l1` / `ub`, 512 for
          `l0a` / `l0b`, 1024 for `l0c`, 64 for `biasbuf`. L0C is stricter than
          the L0 transfer block because `tla.mmad` requires a 1024-byte aligned
          accumulator. Multiples are kept as asked, power of two or not (1024 is
          fine, and so is 1536 for L0A); anything else -- including a value
          below that alignment -- is refused rather than rounded up, because
          raising it would place the buffer at an alignment the caller did not
          ask for.
        - The allocation size is rounded up to `byte_alignment` as well, so the
          start address and the `size_bytes` follow the same alignment:
          allocating 16 bytes in UB at `byte_alignment=32` takes 32, at 256
          takes 256.
        - The allocations made in one kernel must fit the chosen space once
          summed, alignment padding included. The limit is the buffer capacity:
          512 KiB for `l1`, 64 KiB each for `l0a` / `l0b`, 256 KiB for `l0c`.
          `ub` is checked against its whole 256 KiB here; the tighter ceiling
          (248 KiB, or less for a SIMT kernel) is applied during lowering, which
          is where the core type and the compiler-reserve switches are known.
        - Scratch for a `tla.vec.func(mode="simt")` region must be allocated
          *outside* that region and viewed with `tla.make_tensor`: a SIMT region
          establishes no UB base of its own, and outlining passes the buffer to
          the per-thread function.

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
            "(l1, l0a, l0b, l0c, ub, biasbuf)",
        )
    align = _resolve_allocation_alignment("allocate", mem_scope, align)
    size_bytes = _round_up(
        _static_allocation_size_bytes("allocate", shape, dtype, element_bits),
        align,
    )
    _require_allocation_fits("allocate", mem_scope, align, size_bytes)

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
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Return a pointer to the dynamic UB region (size set at launch via `ub=`).

        Parameters:
        - `dtype` (`type[Numeric]`): Pointee element type. Required.
        - `byte_alignment` (`int | None`): Pointer alignment in bytes, spelled
          as `tla.allocate` spells it. Optional; defaults to the element type's
          width.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - UB only; region sits above static `tla.allocate` buffers.
        - At most one region per kernel; carve sub-buffers with pointer arithmetic.
        - Request only what the kernel uses (unused UB can become Data Cache).
        - The extent is not bounds-checked in the kernel.

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


def _emit_arch_bit_op(name: str, mode: str, value: Any, *, loc) -> Int64:
    _require_frontend_state(name)
    _require_core_region(name)
    operand = _as_u64_bit_operand(name, value, loc=loc)
    result = _tla_ops_gen.scalar_bit_op(
        mlir_ir.IntegerType.get_signless(64), operand, mode, loc=loc
    )
    return Int64(result)


@dsl_user_op
def arch_bcnt0(value: Any, *, loc: mlir_ir.Location | None = None) -> Int64:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Count the `0` bits in the binary representation of a `UInt64` value.

        This is the scalar complement of `bcnt1`; together they count the bits
        of a 64-bit mask. It runs on the core's scalar unit and must be placed
        directly in a `tla.cube()` or `tla.vector()` region, outside
        `tla.vec.func`.

        Parameters:
        - `value`: a `UInt64` scalar, a non-negative Python `int` that fits
          64 bits, or a runtime `UInt64` kernel scalar argument.

        Returns:

        - `Int64`: the number of `0` bits, in `[0, 64]`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be placed in a `tla.cube()` or `tla.vector()` region, outside
          `tla.vec.func`.
        - Ascend 950 (3510) only.

        Example:
        ```python
        with tla.vector():
            zeros = tla.arch.bcnt0(mask)
        ```
    """
    return _emit_arch_bit_op("arch.bcnt0", "bcnt0", value, loc=loc)


@dsl_user_op
def arch_bcnt1(value: Any, *, loc: mlir_ir.Location | None = None) -> Int64:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Count the `1` bits in the binary representation of a `UInt64` value.

        The result is the population count of the value. It runs on the core's
        scalar unit and must be placed directly in a `tla.cube()` or
        `tla.vector()` region, outside `tla.vec.func`.

        Parameters:
        - `value`: a `UInt64` scalar, a non-negative Python `int` can fit
          64 bits, or a runtime `UInt64` kernel scalar argument.

        Returns:

        - `Int64`: the number of `1` bits, in `[0, 64]`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be placed in a `tla.cube()` or `tla.vector()` region, outside
          `tla.vec.func`.
        - Ascend 950 (3510) only.

        Example:
        ```python
        with tla.vector():
            ones = tla.arch.bcnt1(mask)
        ```
    """
    return _emit_arch_bit_op("arch.bcnt1", "bcnt1", value, loc=loc)


@dsl_user_op
def arch_clz(value: Any, *, loc: mlir_ir.Location | None = None) -> Int64:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Count the leading `0` bits of a `UInt64` value, from the most
        significant bit down to the first `1`.

        A value of `0` has no `1` bit, so the query returns `64`. It runs on the
        core's scalar unit and must be placed directly in a `tla.cube()` or
        `tla.vector()` region, outside `tla.vec.func`.

        Parameters:
        - `value`: a `UInt64` scalar, a non-negative Python `int` that fits
          64 bits, or a runtime `UInt64` kernel scalar argument.

        Returns:

        - `Int64`: the leading zero count, in `[0, 64]`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be placed in a `tla.cube()` or `tla.vector()` region, outside
          `tla.vec.func`.
        - Ascend 950 (3510) only.

        Example:
        ```python
        with tla.cube():
            zeros = tla.arch.clz(value)

        with tla.vector():
            zeros = tla.arch.clz(value)
        ```
    """
    return _emit_arch_bit_op("arch.clz", "clz", value, loc=loc)


@dsl_user_op
def arch_sff0(value: Any, *, loc: mlir_ir.Location | None = None) -> Int64:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Find the position of the first `0` bit of a `UInt64` value, starting
        from the least significant bit.

        A value whose bits are all `1` has no such bit, so the query returns
        `-1`. It runs on the core's scalar unit and must be placed directly in a
        `tla.cube()` or `tla.vector()` region, outside `tla.vec.func`.

        Parameters:
        - `value`: a `UInt64` scalar, a non-negative Python `int` that fits
          64 bits, or a runtime `UInt64` kernel scalar argument.

        Returns:

        - `Int64`: the bit position, in `[0, 63]`, or `-1` if not found.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be placed in a `tla.cube()` or `tla.vector()` region, outside
          `tla.vec.func`.
        - Ascend 950 (3510) only.

        Example:
        ```python
        with tla.vector():
            pos = tla.arch.sff0(value)
        ```
    """
    return _emit_arch_bit_op("arch.sff0", "sff0", value, loc=loc)


@dsl_user_op
def arch_sff1(value: Any, *, loc: mlir_ir.Location | None = None) -> Int64:
    """Directory: System Variable Access / Architecture Helpers
    Description:
        Find the position of the first `1` bit of a `UInt64` value, starting
        from the least significant bit.

        A value of `0` has no `1` bit, so the query returns `-1`. It runs on the
        core's scalar unit and must be placed directly in a `tla.cube()` or
        `tla.vector()` region, outside `tla.vec.func`.

        Parameters:
        - `value`: a `UInt64` scalar, a non-negative Python `int` that fits
          64 bits, or a runtime `UInt64` kernel scalar argument.

        Returns:

        - `Int64`: the bit position, in `[0, 63]`, or `-1` if not found.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be placed in a `tla.cube()` or `tla.vector()` region, outside
          `tla.vec.func`.
        - Ascend 950 (3510) only.

        Example:
        ```python
        with tla.vector():
            pos = tla.arch.sff1(value)
        ```
    """
    return _emit_arch_bit_op("arch.sff1", "sff1", value, loc=loc)


# Largest divisor the fast-division closed form / long division is defined on.
_FAST_DIV_INT32_MAX = (1 << 31) - 1
_FAST_DIV_INT64_MAX = (1 << 63) - 1
# Diagnostics op-name for the reflected fast-div operators.
_FAST_DIVMOD_OP = "fast_div_get_divisor"


def _bare_int_ssa_width(value: mlir_ir.Value, *, op_name: str, what: str) -> int:
    """Validate a bare integer SSA operand and return its bit width.

    Mirrors ``_as_u64_bit_operand``'s handling of a raw ``mlir_ir.Value``: only
    an integer type is meaningful here, and ``i1`` is never a scalar operand.
    """
    if isinstance(value.type, mlir_ir.IndexType):
        _op_error(op_name, f"{what} must be an integer SSA, not index")
    if not mlir_ir.IntegerType.isinstance(value.type):
        _op_error(op_name, f"{what} must be an integer SSA, got {value.type}")
    int_type = mlir_ir.IntegerType(value.type)
    if int_type.width == 1:
        _op_error(op_name, f"{what} must not be i1/bool")
    return int_type.width


def _wrap_bare_uint_ssa(
    value: mlir_ir.Value, width: int, *, what: str, loc: mlir_ir.Location | None
) -> Numeric:
    """Wrap a bare integer SSA as ``UInt32`` / ``UInt64`` of ``width``.

    A bare value carries no signedness, so it is read as an unsigned bit
    pattern: first take a same-width signless carrier (bits unchanged), then
    zero-extend with ``arith.extui`` to the target width.
    """
    if not mlir_ir.IntegerType.isinstance(value.type):
        raise TypeError(f"{what} expected an integer SSA, got {value.type}")
    src_width = mlir_ir.IntegerType(value.type).width
    if src_width > width:
        raise TypeError(f"{what} is {src_width}-bit, wider than the {width}-bit target")

    signless_src_type = mlir_ir.IntegerType.get_signless(src_width)
    if value.type == signless_src_type:
        signless_src = value
    else:
        signless_src = mlir_ir.Operation.create(
            "builtin.unrealized_conversion_cast",
            operands=[value],
            results=[signless_src_type],
            loc=loc,
        ).results[0]

    if src_width != width:
        signless_src = mlir_ir.Operation.create(
            "arith.extui",
            operands=[signless_src],
            results=[mlir_ir.IntegerType.get_signless(width)],
            loc=loc,
        ).results[0]

    target = UInt32 if width == 32 else UInt64
    return target(_as_unsigned_view(signless_src, width, loc=loc))


def _fast_div_divisor_numeric(divisor: Any, *, loc: mlir_ir.Location | None) -> Numeric:
    """Coerce a fast-division divisor to ``UInt32`` or ``UInt64``.

    A ``Numeric`` (host literal or runtime proxy) is converted by its bit width;
    a Python ``int`` is converted by its value range. The 32-bit variant only
    accepts divisors up to ``INT32_MAX`` and the 64-bit one up to ``INT64_MAX``,
    which is the domain its closed form / long division is defined on. The
    divisor must be positive.
    """
    if isinstance(divisor, bool) or isinstance(divisor, Bool):
        _op_error("fast_div_get_divisor", "divisor must be an integer, not bool")
    resolved = _resolve_bound_value(divisor)

    if isinstance(resolved, Numeric):
        if not type(resolved).is_integer:
            _op_error(
                "fast_div_get_divisor",
                f"divisor must be an integer, got {type(resolved).__name__}",
            )
        if isinstance(resolved, Bool):
            _op_error("fast_div_get_divisor", "divisor must be an integer, not Bool")
        width = type(resolved).width
        if width <= 32:
            target, limit = UInt32, _FAST_DIV_INT32_MAX
        elif width <= 64:
            target, limit = UInt64, _FAST_DIV_INT64_MAX
        else:
            _op_error(
                "fast_div_get_divisor",
                f"divisor width {width} is wider than 64 bits",
            )
        host = resolved.value
        if isinstance(host, int) and not isinstance(host, bool):
            if not 1 <= host <= limit:
                _op_error(
                    "fast_div_get_divisor",
                    f"divisor must be in [1, {limit}] for {target.__name__}, got {host}",
                )
            return target(host)
        if isinstance(resolved, target):
            # Already the requested unsigned width (runtime value / proxy).
            return resolved
        # Another integer width or signedness: keep the width, view it as
        # unsigned. ``arith.bitcast`` rejects an unsigned result, so use the
        # placeholder cast that the backend resolves to a no-op.
        signless = _as_signless_int_ssa(
            resolved.ir_value(loc=loc),
            target.width,
            what="fast_div_get_divisor divisor",
            loc=loc,
        )
        return target(_as_unsigned_view(signless, target.width, loc=loc))

    if isinstance(resolved, int) and not isinstance(resolved, bool):
        if not 1 <= resolved <= _FAST_DIV_INT64_MAX:
            _op_error(
                "fast_div_get_divisor",
                f"divisor must be in [1, {_FAST_DIV_INT64_MAX}], got {resolved}",
            )
        if resolved <= _FAST_DIV_INT32_MAX:
            return UInt32(resolved)
        return UInt64(resolved)

    if isinstance(resolved, mlir_ir.Value):
        value_width = _bare_int_ssa_width(
            resolved, op_name="fast_div_get_divisor", what="divisor"
        )
        if value_width > 64:
            _op_error(
                "fast_div_get_divisor",
                f"divisor width {value_width} is wider than 64 bits",
            )
        target_width = 32 if value_width <= 32 else 64
        return _wrap_bare_uint_ssa(
            resolved, target_width, what="fast_div_get_divisor divisor", loc=loc
        )

    _op_error(
        "fast_div_get_divisor",
        "divisor must be a UInt32/UInt64 scalar or a Python int, "
        f"got {_type_name(divisor)}",
    )


def _as_unsigned_view(
    value: mlir_ir.Value, width: int, *, loc: mlir_ir.Location | None = None
) -> mlir_ir.Value:
    """View a signless ``i<width>`` SSA value as its unsigned ``ui<width>``.

    ``arith.bitcast`` does not accept an unsigned result type, so the same-bit
    reinterpretation goes through the placeholder cast the backend elides.
    """
    unsigned = mlir_ir.IntegerType.get_unsigned(width)
    if value.type == unsigned:
        return value
    return mlir_ir.Operation.create(
        "builtin.unrealized_conversion_cast",
        operands=[value],
        results=[unsigned],
        loc=loc,
    ).results[0]


def _numeric_to_signless_int(
    value: Numeric, width: int, *, what: str, loc: mlir_ir.Location | None = None
) -> mlir_ir.Value:
    """Signless ``i<width>`` SSA for a Numeric (host literal or runtime value)."""
    host = value.value
    if isinstance(host, int) and not isinstance(host, bool):
        return _const_i32(host, loc=loc) if width == 32 else _const_i64(host, loc=loc)
    return _as_signless_int_ssa(value.ir_value(loc=loc), width, what=what, loc=loc)


def _emit_fast_div_divisor(
    divisor: Numeric, *, loc: mlir_ir.Location | None
) -> tuple[mlir_ir.Value, mlir_ir.Value]:
    """Emit ``tla.scalar_fast_div_divisor``; return the signless results.

    The raw ``i<width>`` results are returned so the SIMD fast div/mod ops can
    consume them without an unsigned round-trip; the public ``magic`` / ``shift``
    properties wrap them as `UInt32` / `UInt64`.
    """
    width = type(divisor).width
    int_type = mlir_ir.IntegerType.get_signless(width)
    operand = _numeric_to_signless_int(
        divisor, width, what="fast_div_get_divisor divisor", loc=loc
    )
    values = _tla_ops_gen.scalar_fast_div_divisor(int_type, int_type, operand, loc=loc)
    return values[0], values[1]


def _fast_div_dividend_numeric(
    value: Any, width: int, *, loc: mlir_ir.Location | None
) -> Numeric:
    """Coerce a fast-division dividend to ``UInt32`` / ``UInt64``.

    The dividend is forced to the divisor's width: a narrower integer widens, a
    wider one is rejected. The asc-devkit algorithm is only exact while the
    dividend is at most ``INT32_MAX`` (32-bit) / ``INT64_MAX`` (64-bit), which
    also keeps ``dividend + mulhi`` from wrapping.
    """
    target = UInt32 if width == 32 else UInt64
    limit = _FAST_DIV_INT32_MAX if width == 32 else _FAST_DIV_INT64_MAX
    if isinstance(value, bool) or isinstance(value, Bool):
        _op_error(_FAST_DIVMOD_OP, "dividend must be an integer, not bool")
    resolved = _resolve_bound_value(value)

    if isinstance(resolved, Numeric):
        if not type(resolved).is_integer:
            _op_error(
                _FAST_DIVMOD_OP,
                f"dividend must be an integer, got {type(resolved).__name__}",
            )
        if isinstance(resolved, Bool):
            _op_error(_FAST_DIVMOD_OP, "dividend must be an integer, not Bool")
        if type(resolved).width > width:
            _op_error(
                _FAST_DIVMOD_OP,
                f"dividend is {type(resolved).width}-bit, wider than the "
                f"{width}-bit divisor",
            )
        host = resolved.value
        if isinstance(host, int) and not isinstance(host, bool):
            if not 0 <= host <= limit:
                _op_error(
                    _FAST_DIVMOD_OP, f"dividend must be in [0, {limit}], got {host}"
                )
            return target(host)
        if isinstance(resolved, target):
            return resolved
        signless = _as_signless_int_ssa(
            resolved.ir_value(loc=loc),
            width,
            what="fast_divmod dividend",
            loc=loc,
        )
        return target(_as_unsigned_view(signless, width, loc=loc))

    if isinstance(resolved, int) and not isinstance(resolved, bool):
        if not 0 <= resolved <= limit:
            _op_error(
                _FAST_DIVMOD_OP, f"dividend must be in [0, {limit}], got {resolved}"
            )
        return target(resolved)

    if isinstance(resolved, mlir_ir.Value):
        value_width = _bare_int_ssa_width(
            resolved, op_name=_FAST_DIVMOD_OP, what="dividend"
        )
        if value_width > width:
            _op_error(
                _FAST_DIVMOD_OP,
                f"dividend is {value_width}-bit, wider than the {width}-bit divisor",
            )
        return _wrap_bare_uint_ssa(
            resolved, width, what="fast_divmod dividend", loc=loc
        )

    _op_error(
        _FAST_DIVMOD_OP,
        "dividend must be a UInt32/UInt64 scalar or a Python int, "
        f"got {_type_name(value)}",
    )


class FastDivDivisor:
    """A ``(magic, shift)`` pair for fast unsigned division by a fixed divisor.

    This is the asc-devkit/``asc_uintdiv`` pair, so the consumer folds in the
    dividend before the shift: ``(x + mulhi_w(x, magic)) >> shift``. Dropping
    the add would need a different magic.

    Construct it inside a ``tla.cube()`` or ``tla.vector()`` region (it runs on
    the scalar unit). ``divisor_mlir`` keeps the coerced `UInt32` / `UInt64`
    divisor; ``magic`` and ``shift`` are same-width `UInt32` / `UInt64` values.
    """

    def __init__(self, divisor: Any, *, loc: mlir_ir.Location | None = None) -> None:
        _require_frontend_state("fast_div_get_divisor")
        _require_core_region("fast_div_get_divisor")
        self.divisor_mlir: Numeric = _fast_div_divisor_numeric(divisor, loc=loc)
        width = type(self.divisor_mlir).width
        target = UInt32 if width == 32 else UInt64
        self._magic_ssa, self._shift_ssa = _emit_fast_div_divisor(
            self.divisor_mlir, loc=loc
        )
        self._magic = target(_as_unsigned_view(self._magic_ssa, width, loc=loc))
        self._shift = target(_as_unsigned_view(self._shift_ssa, width, loc=loc))

    @property
    def magic(self) -> Numeric:
        """The multiplication factor, same width as `divisor_mlir`."""
        return self._magic

    @property
    def shift(self) -> Numeric:
        """The right-shift amount after `mulhi`, same width as `divisor_mlir`."""
        return self._shift

    @property
    def divisor(self) -> Numeric:
        """The coerced `UInt32` / `UInt64` divisor."""
        return self.divisor_mlir

    @dsl_user_op
    def __rfloordiv__(
        self, dividend: Any, *, loc: mlir_ir.Location | None = None
    ) -> Numeric:
        """Fast ``dividend // divisor`` using this pair (SIMD: ``x // fd``).

        The dividend must fit the divisor's width and be at most ``INT32_MAX``
        (32-bit) / ``INT64_MAX`` (64-bit), the domain the asc-devkit magic is
        exact on. Returns a `UInt32` / `UInt64` quotient.
        """
        _require_frontend_state(_FAST_DIVMOD_OP)
        _require_core_region(_FAST_DIVMOD_OP)
        width = type(self.divisor_mlir).width
        target = UInt32 if width == 32 else UInt64
        value = _fast_div_dividend_numeric(dividend, width, loc=loc)
        int_type = mlir_ir.IntegerType.get_signless(width)
        quotient = _tla_ops_gen.scalar_simd_fast_div(
            int_type,
            _numeric_to_signless_int(
                value, width, what="fast_divmod dividend", loc=loc
            ),
            self._magic_ssa,
            self._shift_ssa,
            loc=loc,
        )
        return target(_as_unsigned_view(quotient, width, loc=loc))

    @dsl_user_op
    def __rmod__(
        self, dividend: Any, *, loc: mlir_ir.Location | None = None
    ) -> Numeric:
        """Fast ``dividend % divisor`` using this pair (SIMD: ``x % fd``).

        Same dividend/divisor domain as ``__rfloordiv__``. Returns a `UInt32` /
        `UInt64` remainder, computed as ``dividend - quotient * divisor``.
        """
        _require_frontend_state(_FAST_DIVMOD_OP)
        _require_core_region(_FAST_DIVMOD_OP)
        width = type(self.divisor_mlir).width
        target = UInt32 if width == 32 else UInt64
        value = _fast_div_dividend_numeric(dividend, width, loc=loc)
        int_type = mlir_ir.IntegerType.get_signless(width)
        remainder = _tla_ops_gen.scalar_simd_fast_mod(
            int_type,
            _numeric_to_signless_int(
                value, width, what="fast_divmod dividend", loc=loc
            ),
            _numeric_to_signless_int(
                self.divisor_mlir, width, what="fast_divmod divisor", loc=loc
            ),
            self._magic_ssa,
            self._shift_ssa,
            loc=loc,
        )
        return target(_as_unsigned_view(remainder, width, loc=loc))


@dsl_user_op
def fast_div_get_divisor(
    divisor: Any, *, loc: mlir_ir.Location | None = None
) -> FastDivDivisor:
    """Directory: Scalar Compute

    Description:
        Precompute a `FastDivDivisor` for unsigned fast division by a fixed
        `divisor`. The returned object exposes `magic`, `shift` and the coerced
        `divisor_mlir`, and is consumed through the reflected operators:
        `x // fd` computes `(x + mulhi_w(x, magic)) >> shift`, and `x % fd`
        returns `x - quotient * divisor`. Both results are `UInt32` / `UInt64`
        values of the divisor's width.

        The precompute is exact for a divisor in `[1, INT32_MAX]` (`i32`) or
        `[1, INT64_MAX]` (`i64`) and dividends up to `INT32_MAX` / `INT64_MAX`.
        The divisor must be non-zero and non-negative (i.e. positive), and
        `shift` is below the operand width. No dynamic check is emitted for
        runtime values, so the caller must guarantee these. `magic` is the
        unsigned bit pattern of the same width and may exceed `INT_MAX`;
        `shift` is the precomputed shift amount.

    Parameters:
        - `divisor`: a `UInt32` / `UInt64` scalar, a bare integer SSA, a runtime
          kernel scalar argument, or a Python `int`. A `Numeric` or SSA is
          converted by its bit width; a Python `int` is converted by its value
          range.

    Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be placed in a `tla.cube()` or `tla.vector()` region, outside
          `tla.vec.func`.
        - The divisor must be non-zero, non-negative, and no larger than
          `INT32_MAX` (32-bit variant) or `INT64_MAX` (64-bit variant); a
          Python `int` above `INT32_MAX` selects the 64-bit variant. Runtime
          values are not checked.
        - `shift` is below the operand width (32 or 64 bits).
        - The dividend of a later `x // fd` / `x % fd` must also be at most
          `INT32_MAX` / `INT64_MAX` (runtime values are the caller's contract).
        - Ascend 950 (3510) only.

    Example:
        ```python
        with tla.vector():
            fd = tla.fast_div_get_divisor(divisor)
            quotient = x // fd   # reflected __rfloordiv__
            remainder = x % fd   # reflected __rmod__
            magic, shift = fd.magic, fd.shift
        ```
    """
    return FastDivDivisor(divisor, loc=loc)


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
        Build a pointer from an address value.

        Inside ``@tla.kernel``: builds `!tla.ptr`. Outside a kernel: same
        ``Pointer`` type holding a Host address for Host ``tla.make_tensor``
        (no MLIR); ``value`` must be a Python ``int`` address.

        Parameters:
        - `dtype` (`type[Numeric] | None`): Pointee element type; `None` means
          `Int8`. Optional, default `None`.
        - `value` (`int | Numeric | address SSA`): Address value. Required.
          Host path requires ``int``.
        - `mem_space` (`AddressSpace`): Address space of the pointer. Optional, default `AddressSpace.gm`.
        - `assumed_align` (`int | None`): Assumed alignment in bytes. Optional, default `None`.

        Constraints:
        - Kernel path: must be inside `@tla.kernel`; integer address bit-width
          must match the target `mem_space`.
        - Host path: no frontend; result is for Host `tla.make_tensor` only.

        Example:
        ```python
        ptr = tla.make_ptr(tla.Float16, addr, mem_space=tla.AddressSpace.gm)
        ```
    """
    if dtype is not None and (
        not isinstance(dtype, type) or not issubclass(dtype, Numeric) or not dtype.dtype
    ):
        _op_error("make_ptr", f"expects dtype to be a type of Numeric, but got {dtype}")
    dt = Int8 if dtype is None else dtype

    if _runtime._current_frontend_state() is None:
        if not isinstance(value, int):
            _op_error(
                "make_ptr",
                "outside a kernel, value must be a Python int device address; "
                f"got {type(value).__name__}",
            )
        if not isinstance(mem_space, AddressSpace):
            _op_error(
                "make_ptr",
                f"expects mem_space to be AddressSpace, got {type(mem_space).__name__}",
            )
        return _Pointer._from_host(
            dt, int(value), mem_space, assumed_align=assumed_align
        )

    _require_frontend_state("make_ptr")
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


def _emit_tensor_ptr_with_coord(
    source: mlir_ir.Value, loc: mlir_ir.Location | None = None
) -> _Pointer:
    """Emit ``tla.tensor_ptr_with_coord``: the view-start address of a tensor.

    The result is the address element ``t[0]`` of the view accesses --
    ``coord[0]*stride[0] + coord[1]*stride[1]`` elements past the storage base
    that :meth:`_emit_tensor_ptr` returns. Requires a linear-layout tensor of
    rank one or two. The coordinate and stride are read from the view's
    producer chain, so a runtime coordinate (a block index, a loop variable)
    lowers to runtime arithmetic; a view selected by runtime control flow has
    no single producer, so take this on each branch's view first and select the
    resulting pointers. Inside a SIMT ``tla.vec.func`` the address comes from
    the outlined buffer parameter, whose launch pointer already carries the
    view start, so no extra launch argument is needed.
    """
    _require_frontend_state("tensor_ptr_with_coord")
    ptr_ty = _tla_type_bridge.tensor_ptr_type_get(source.type)
    op = mlir_ir.Operation.create(
        "tla.tensor_ptr_with_coord",
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
_require_generated("prof_start")
_require_generated("prof_end")
_require_generated("trap")
_require_generated("mutex")
_require_generated("mutex_lock")
_require_generated("mutex_unlock")
_require_generated("cube")
_require_generated("vector")
_require_generated("generate_mask")
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
_require_generated("unsqueeze")
_require_generated("bitwise_not")
_require_generated("bitwise_and")
_require_generated("bitwise_or")
_require_generated("bitwise_xor")
_require_generated("divs")
_require_generated("reduce")
_require_generated("reduce_intra_block")
_require_generated("pair_wise")
_require_generated("bitcast")
_require_generated("interleave")
_require_generated("addc")
_require_generated("subc")
_require_generated("mull")
_require_generated("deinterleave")
_require_generated("pack")
_require_generated("unpack")
for _unary_op_name in ("exp", "log", "sqrt", "abs", "neg"):
    _require_generated(_unary_op_name)
_require_generated("cmp")
_require_generated("arch_block_idx")
_require_generated("arch_sub_block_idx")
_require_generated("arch_block_num")
_require_generated("arch_sub_block_num")
_require_generated("arch_get_system_cycle")
_require_generated("arch_thread_block_dim")
_require_generated("arch_thread_idx")
_require_generated("arch_lane_idx")
_require_generated("arch_activemask")
_require_generated("arch_vote_ballot")
_require_generated("arch_vote_all")
_require_generated("arch_vote_any")
_require_generated("arch_lanemask")
_require_generated("arch_warp_reduce")
_require_generated("scalar_bit_op")
_require_generated("scalar_fast_div_divisor")
_require_generated("scalar_simd_fast_div")
_require_generated("scalar_simd_fast_mod")
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
_require_generated("simt_atomic_add")
_require_generated("arch_sync_threads")
_require_generated("arch_dcci_cacheline")
_require_generated("arch_dcci_cachelines")
_require_generated("arch_dci")
_require_generated("simd_atomic_add")
_require_generated("simd_atomic_exch")
_require_generated("arch_get_ar")
_require_generated("arch_clear_ar")
_require_generated("arch_load_bypass_dcache")
_require_generated("arch_store_bypass_dcache")
_require_generated("inttoptr")
_require_generated("recast_ptr")

arch = _Namespace()
arch._set("sub_exp", arch_sub_exp)
arch._set("dcache_prefetch", arch_dcache_prefetch)
arch._set("dcci_cacheline", arch_dcci_cacheline)
arch._set("dcci_cachelines", arch_dcci_cachelines)
arch._set("dci", arch_dci)
arch._set("load_bypass_dcache", arch_load_bypass_dcache)
arch._set("store_bypass_dcache", arch_store_bypass_dcache)
arch._set("atomic_exch", arch_atomic_exch)
arch._set("bcnt0", arch_bcnt0)
arch._set("bcnt1", arch_bcnt1)
arch._set("clz", arch_clz)
arch._set("sff0", arch_sff0)
arch._set("sff1", arch_sff1)
arch._set("axpy", arch_axpy)
arch._set("sub_abs", arch_sub_abs)
arch._set("muls_cast", arch_muls_cast)
arch._set("get_ctrl", arch_get_ctrl)
arch._set("set_ctrl", arch_set_ctrl)
arch._set("block_idx", arch_block_idx)
arch._set("sub_block_idx", arch_sub_block_idx)
arch._set("block_num", arch_block_num)
arch._set("sub_block_num", arch_sub_block_num)
arch._set("get_system_cycle", arch_get_system_cycle)
arch._set("thread_block_dim", arch_thread_block_dim)
arch._set("thread_idx", arch_thread_idx)
arch._set("icache_prefetch", arch_icache_prefetch)
arch._set("get_icache_prefetch_status", arch_get_icache_prefetch_status)
arch._set("warpsize", arch_warpsize)
arch._set("lane_idx", arch_lane_idx)
arch._set("warp_reduce_sync", arch_warp_reduce_sync)
arch._set("activemask", arch_activemask)
arch._set("vote_ballot_sync", arch_vote_ballot_sync)
arch._set("vote_all_sync", arch_vote_all_sync)
arch._set("vote_any_sync", arch_vote_any_sync)
arch._set("lanemask_eq", arch_lanemask_eq)
arch._set("lanemask_lt", arch_lanemask_lt)
arch._set("lanemask_le", arch_lanemask_le)
arch._set("lanemask_gt", arch_lanemask_gt)
arch._set("lanemask_ge", arch_lanemask_ge)
arch._set("sync_threads", arch_sync_threads)
arch._set("get_ar", arch_get_ar)
arch._set("clear_ar", arch_clear_ar)
arch._set("shuffle_sync", arch_shuffle_sync)
arch._set("shuffle_sync_up", arch_shuffle_sync_up)
arch._set("shuffle_sync_down", arch_shuffle_sync_down)
arch._set("shuffle_sync_bfly", arch_shuffle_sync_bfly)
arch._set("inplace_mul_add", arch_inplace_mul_add)
arch._set("atomic_add", arch_atomic_add)
arch._set("mul_inplace_add", arch_mul_inplace_add)
arch._set("get_capacity_in_bytes", arch_get_capacity_in_bytes)
arch._set("get_vec_len_in_bytes", arch_get_vec_len_in_bytes)
arch._set("get_datablock_size_in_bytes", arch_get_datablock_size_in_bytes)
arch._set("get_dyn_ub", arch_get_dyn_ub)
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
arch._set("VectorLayout", _LayoutTag("VectorLayout"))
from .dsl import C1V1 as _C1V1
from .dsl import C1V2 as _C1V2

arch._set("C1V1", _C1V1)
arch._set("C1V2", _C1V2)

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
    """Directory: SIMD Vector Compute / Mask Compute
    Description:
        Create a vector mask from a fixed pattern (`tla.mask.*` tokens).

        Parameters:
        - `pattern` (`_MaskPattern | str | None`): Mask pattern token or its name
          string. Required at runtime (`None` raises). Tokens live under
          `tla.mask`:

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
          how many elements fit in one vector: 256 bytes / element size). `Bool`
          is b8-width → `!tla.mask<256>`. Optional, default `Float32`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.vec.func()`; `pattern` is required.
        - Ops that take `mask=` need a `MaskSSA` from `create_mask` /
          `update_mask`, not a raw `tla.mask.*` token.
        - `VectorSSA[tla.Bool]` is single-VL; it supports load/store and
          `tla.where` select on the b8 carrier. Arithmetic and cast remain
          unsupported.

        Example:
        ```python
        with tla.vec.func(mode="simd"):
            # Build masks from pattern tokens:
            m_all = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            m_tail = tla.create_mask(pattern=tla.mask.VL8, dtype=tla.Float16)
            m_bool = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Bool)  # mask<256>
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
    """Directory: SIMD Vector Compute / Mask Compute

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


# generate_mask covers only 16- and 32-bit source elements; the element width
# fixes both the mask lane count and the legal block offset range.
_GENERATE_MASK_OFFSET_LIMITS = {16: 15, 32: 31}


def _generate_mask_offset_limit(op_name: str, element_type: str) -> int:
    element_bytes = dtype_size_bytes(element_type)
    if element_bytes not in (2, 4):
        _op_error(
            op_name,
            f"unsupported source element type {element_type}; "
            "generate_mask only supports 16- and 32-bit elements",
        )
    return _GENERATE_MASK_OFFSET_LIMITS[element_bytes * 8]


# Keep this type-only: tla.ffi lazily calls back into core_api, so a
# runtime import here would add an unnecessary reverse dependency.
if TYPE_CHECKING:
    from .tla.ffi import ExternFunction


def _emit_extern_call(
    extern_function: ExternFunction,
    args: tuple[object, ...],
    *,
    loc: mlir_ir.Location | None = None,
) -> VectorSSA | None:
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
    if _runtime._in_simt_vec_func():
        _op_error("extern", "call is not supported inside a SIMT tla.vec.func()")
    has_register = extern_function.return_type is not None or any(
        isinstance(t, TlaVectorSSATypeDescriptor) for t in extern_function.arg_types
    )
    if has_register:
        _runtime._require_enclosing_region("extern register call", "vec.func")
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
        if isinstance(expected, TlaVectorSSATypeDescriptor):
            _require_category(
                "extern", f"argument {position}", arg, "vector_ssa", position
            )
            value = _as_value(arg)
            actual = _vector_ssa_type_for_mlir_value(value)
            if actual != expected:
                _op_error(
                    "extern",
                    f"argument {position} of {extern_function.symbol} requires a full "
                    f"{expected.element_type} register ({expected.valid_lanes} valid lanes), "
                    f"got {value.type}",
                )
        elif isinstance(expected, TypedPointer):
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
    result_types = (
        [extern_function.return_type.to_mlir_type()]
        if extern_function.return_type is not None
        else []
    )
    call_op = _tla_ops_gen.CallExternOp(
        result_types, extern_function.symbol, operands, loc=loc
    ).operation
    if usage is None:
        usage = _ExternUsage(function=extern_function)
        state.extern_usages[extern_function.symbol] = usage
    usage.calls.append((call_op, core_type))
    if result_types:
        return VectorSSA(call_op.results[0])
    return None


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


@dsl_user_op
def prof_start(*, loc: mlir_ir.Location | None = None) -> None:
    """Directory: Debug APIs
    Description:
        Start best-effort device profiling at this position. On A5, collect
        with msprof op's KernelScale metric. Emits the native ALL barrier,
        CTRL bit 0 set, ALL barrier sequence.

        Parameters:
        - None.

        Constraints:
        - Use one start/end pair per kernel specialization.
        - Placement, execution count and synchronization remain the caller's
          responsibility. No timing guarantee is provided.

        Example:
        ```python
        with tla.cube():
            tla.prof_start()
            tla.mmad(acc, lhs, rhs, init_c=True)
            tla.prof_end()
        ```
    """
    _require_frontend_state("prof_start")
    state = _runtime._current_frontend_state()
    if state.prof_started:
        raise TlaCoreAPIError("tla.prof_start: only one profiling pair is supported")
    _tla_ops_gen.prof_start(loc=loc)
    state.prof_started = True


@dsl_user_op
def prof_end(*, loc: mlir_ir.Location | None = None) -> None:
    """Directory: Debug APIs
    Description:
        End the interval opened by `tla.prof_start()`. Emits the native ALL
        barrier, CTRL bit 0 clear, ALL barrier sequence.

        Parameters:
        - None.

        Constraints:
        - Requires an unmatched start while staging; dynamic pairing is not proved.
        - Invalid placement can fail compilation or hang the kernel.

        Example:
        ```python
        with tla.cube():
            tla.prof_start()
            tla.mmad(acc, lhs, rhs, init_c=True)
            tla.prof_end()
        ```
    """
    _require_frontend_state("prof_end")
    state = _runtime._current_frontend_state()
    if not state.prof_started or state.prof_ended:
        raise TlaCoreAPIError("tla.prof_end: expected an unmatched prof_start")
    _tla_ops_gen.prof_end(loc=loc)
    state.prof_ended = True


class _ProfScope:
    def __init__(self, loc: mlir_ir.Location | None) -> None:
        self.loc = loc

    def __enter__(self) -> None:
        prof_start(loc=self.loc)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc_type is None:
            prof_end(loc=self.loc)
        return False


@dsl_user_op
def prof_scope(*, loc: mlir_ir.Location | None = None) -> _ProfScope:
    """Directory: Debug APIs
    Description:
        Emit the same profiling start/end pair around the body without moving
        or widening it. Adds only the native profiling fences, which perturb
        overlap; no guarantee of isolated timing is provided.

        Parameters:
        - None.

        Constraints:
        - Use one region per kernel specialization. Obvious nested or duplicate
          pairs are rejected; execution count and synchronization are not proved.
        - Validated A5 Vector placement surrounds a complete `tla.vec.func`
          call in its scalar caller. Unsupported placement may fail compilation.
        - SIMT, multiple intervals and nested intervals are not promised.

        Example:
        ```python
        with tla.vector():
            with tla.prof_scope():
                with tla.vec.func(mode="simd"):
                    dst.store(tla.add(lhs.load(), rhs.load()))
        ```
    """
    return _ProfScope(loc)


__all__ = [
    "TlaCoreAPIError",
    "ceil_div",
    "fast_div_get_divisor",
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
    "trap",
    "flag",
    "cross_flag",
    "cross_core_set_flag",
    "cross_core_wait_flag",
    "set_flag",
    "wait_flag",
    "pipe_barrier",
    "sync_all",
    "sync_all_aiv",
    "prof_start",
    "prof_end",
    "prof_scope",
    "local_mem_bar",
    "data_sync_barrier",
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
    "relu",
    "prelu",
    "leaky_relu",
    "shift_left",
    "shift_right",
    "where",
    "squeeze",
    "unsqueeze",
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
    "addc",
    "subc",
    "mull",
    "deinterleave",
    "pack",
    "unpack",
    "RegPart",
    "gather",
    "scatter",
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
