"""Lightweight TLA Python type helpers, MLIR ``!tla.ptr`` (:class:`PtrType`), and addrspace helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Literal, TypeAlias

from .base_dsl.typing import (
    Numeric,
    Pointer as PointerABC,
)

from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

from . import _tla_type_bridge
from .address_space import AddressSpace


def _coerce_host_tensor_dtype(value: Any) -> str:
    if isinstance(value, type) and issubclass(value, Numeric):
        token = getattr(value, "dtype", "") or ""
        if not token:
            raise TypeError(
                "Tensor dtype=... expected a concrete Tla element type (e.g. tla.Float16)"
            )
        return str(token).strip().lower()
    raise TypeError(
        "Tensor dtype=... expected a Tla element type (e.g. tla.Float16), not str; "
        f"got {type(value).__name__}"
    )


def _coerce_host_tensor_addrspace(value: Any) -> str:
    if isinstance(value, AddressSpace):
        return value.name
    raise TypeError(
        "Tensor addrspace=... expected tla.AddressSpace, not str; "
        f"got {type(value).__name__}"
    )


_cached_ptr_typeid: mlir_ir.TypeID | None = None


class PtrType(mlir_ir.Type):
    """``!tla.ptr<...>`` as a proper Python subclass of :class:`mlir.ir.Type` (Tla-style)."""

    def __init__(self, cast_from_type: mlir_ir.Type) -> None:
        super().__init__(cast_from_type)

    def __repr__(self) -> str:
        return f"PtrType({str(self)})"

    @classmethod
    def get_static_typeid(cls) -> mlir_ir.TypeID:
        global _cached_ptr_typeid
        if _cached_ptr_typeid is None:
            ctx = mlir_ir.Context()
            with ctx:
                probe = cls.get(mlir_ir.IndexType.get(), "gm", 1, context=ctx)
            _cached_ptr_typeid = probe.typeid
        return _cached_ptr_typeid

    @classmethod
    def isinstance(cls, ty: mlir_ir.Type) -> bool:
        return _tla_type_bridge.type_is_ptr(ty)

    @classmethod
    def try_cast(cls, ty: mlir_ir.Type) -> PtrType | None:
        if not cls.isinstance(ty):
            return None
        return ty if isinstance(ty, cls) else cls(ty)

    @staticmethod
    def get(
        pointee: mlir_ir.Type,
        addrspace: str,
        alignment: int,
        *,
        context: mlir_ir.Context | None = None,
    ) -> PtrType:
        """Build ``!tla.ptr<pointee, token, align>`` (frontend uses arch tokens, e.g. ``l0c``)."""
        ctx = context if context is not None else pointee.context
        return PtrType(
            _tla_type_bridge.ptr_type_get(ctx, pointee, str(addrspace), int(alignment))
        )

    @property
    def pointee(self) -> mlir_ir.Type:
        return _tla_type_bridge.ptr_pointee_type_get(self.context, self)

    @property
    def addrspace(self) -> str:
        return _tla_type_bridge.ptr_addrspace(self)

    @property
    def alignment(self) -> int:
        return _tla_type_bridge.ptr_alignment(self)

    @property
    def mlir_type(self) -> mlir_ir.Type:
        """Former façade field: the underlying MLIR type is ``self``."""
        return self


class LayoutType(mlir_ir.Type):
    """``!tla.layout<!tla.shape<...>, !tla.stride<...>[, !tla.shape<...>]>`` type wrapper."""

    def __init__(self, cast_from_type: mlir_ir.Type) -> None:
        super().__init__(cast_from_type)

    def __repr__(self) -> str:
        return f"LayoutType({str(self)})"

    @classmethod
    def isinstance(cls, ty: mlir_ir.Type) -> bool:
        return _tla_type_bridge.type_is_layout(ty)

    @classmethod
    def try_cast(cls, ty: mlir_ir.Type) -> "LayoutType | None":
        if not cls.isinstance(ty):
            return None
        return ty if isinstance(ty, cls) else cls(ty)

    @staticmethod
    def get(
        shape_val: mlir_ir.Value,
        stride_val: mlir_ir.Value,
        origin_shape_val: mlir_ir.Value | None = None,
        *,
        layout_tag: str = "RowMajor",
        context: mlir_ir.Context | None = None,
    ) -> "LayoutType":
        """Construct a structured ``!tla.layout`` from shape/stride SSA values."""
        if not isinstance(shape_val, mlir_ir.Value):
            raise TypeError(
                f"LayoutType.get shape_val expects mlir.ir.Value, got {type(shape_val).__name__}"
            )
        if not isinstance(stride_val, mlir_ir.Value):
            raise TypeError(
                f"LayoutType.get stride_val expects mlir.ir.Value, got {type(stride_val).__name__}"
            )
        if origin_shape_val is not None and not isinstance(
            origin_shape_val, mlir_ir.Value
        ):
            raise TypeError(
                "LayoutType.get origin_shape_val expects mlir.ir.Value or None, "
                f"got {type(origin_shape_val).__name__}"
            )
        ctx = context if context is not None else shape_val.type.context
        tag = str(layout_tag).strip()
        if not tag:
            raise TypeError("LayoutType.get layout_tag must be non-empty")
        if not _tla_type_bridge.type_is_shape(shape_val.type):
            raise TypeError(
                f"LayoutType.get expected !tla.shape operand, got {shape_val.type}"
            )
        if not _tla_type_bridge.type_is_stride(stride_val.type):
            raise TypeError(
                f"LayoutType.get expected !tla.stride operand, got {stride_val.type}"
            )
        if origin_shape_val is not None and not _tla_type_bridge.type_is_shape(
            origin_shape_val.type
        ):
            raise TypeError(
                f"LayoutType.get expected !tla.shape origin operand, got {origin_shape_val.type}"
            )
        return LayoutType(
            _tla_type_bridge.layout_type_from_components_get(
                ctx,
                shape_val.type,
                stride_val.type,
                None if origin_shape_val is None else origin_shape_val.type,
                tag,
            )
        )

    @property
    def mlir_type(self) -> mlir_ir.Type:
        return self


TlaIndexTreeKind = Literal["shape", "coord", "stride"]
TlaIndexTreeLeaf: TypeAlias = int | None
TlaIndexTree: TypeAlias = TlaIndexTreeLeaf | tuple["TlaIndexTree", ...]


def _tla_type_context(context: mlir_ir.Context | None = None) -> mlir_ir.Context:
    if context is not None:
        return context
    try:
        current = mlir_ir.Context.current
    except ValueError:
        return mlir_ir.Context()
    return current if current is not None else mlir_ir.Context()


def _normalize_index_tree_leaf(value: Any) -> int | None:
    if isinstance(value, bool):
        raise TypeError("boolean not allowed in Tla index tree")
    if value is None:
        return None
    if isinstance(value, int):
        if value < 0:
            raise ValueError(
                "Tla index tree leaves must be non-negative integers or None"
            )
        return int(value)
    raise TypeError(
        "Tla index type metadata expects static int leaves or None for dynamic leaves; "
        f"got {type(value).__name__}"
    )


def _normalize_index_tree(tree: Any, *, _tuple_depth: int = 0) -> Any:
    if isinstance(tree, list):
        raise ValueError(
            "expected nested tuple tree for index components (use parentheses, not brackets)"
        )
    if isinstance(tree, tuple):
        if len(tree) == 0:
            raise ValueError("empty tuple in index tree")
        if _tuple_depth >= 2:
            raise ValueError(
                "Tla index tree supports only top-level leaves or one-level leaf groups"
            )
        return tuple(
            _normalize_index_tree(x, _tuple_depth=_tuple_depth + 1) for x in tree
        )
    return _normalize_index_tree_leaf(tree)


def _index_tree_to_asm(tree: Any) -> str:
    if isinstance(tree, tuple):
        return "(" + ",".join(_index_tree_to_asm(x) for x in tree) + ")"
    leaf = _normalize_index_tree_leaf(tree)
    return "?" if leaf is None else str(leaf)


def _index_tree_top_level_to_asm(tree: Any) -> str:
    if isinstance(tree, tuple):
        return ",".join(_index_tree_to_asm(x) for x in tree)
    return _index_tree_to_asm(tree)


def _index_tree_to_metadata(tree: Any, dynamic_values: Iterator[Any]) -> Any:
    if isinstance(tree, tuple):
        return tuple(_index_tree_to_metadata(x, dynamic_values) for x in tree)
    leaf = _normalize_index_tree_leaf(tree)
    if leaf is not None:
        return leaf
    return next(dynamic_values)


@dataclass(frozen=True)
class TlaIndexTreeType:
    """Structured Python descriptor for ``!tla.shape`` / ``!tla.coord`` / ``!tla.stride``."""

    kind: TlaIndexTreeKind
    tree: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree", _normalize_index_tree(self.tree))

    @classmethod
    def from_components(
        cls, kind: TlaIndexTreeKind, components: tuple[Any, ...]
    ) -> "TlaIndexTreeType":
        return cls(kind, tuple(_normalize_index_tree(c) for c in components))

    def body_asm(self) -> str:
        return _index_tree_top_level_to_asm(self.tree)

    def to_asm(self) -> str:
        return str(self.to_mlir_type())

    def to_mlir_type(self, context: mlir_ir.Context | None = None) -> mlir_ir.Type:
        ctx = _tla_type_context(context)
        if self.kind == "shape":
            return _tla_type_bridge.shape_type_get(ctx, self.tree)
        if self.kind == "coord":
            return _tla_type_bridge.coord_type_get(ctx, self.tree)
        if self.kind == "stride":
            return _tla_type_bridge.stride_type_get(ctx, self.tree)
        raise ValueError(f"unsupported Tla index tree type kind: {self.kind!r}")

    def metadata(self, dynamic_values: Iterable[Any] = ()) -> Any:
        return _index_tree_to_metadata(self.tree, iter(dynamic_values))


@dataclass(frozen=True)
class TlaLayoutDescriptor:
    """Structured Python descriptor for ``!tla.layout``."""

    shape: TlaIndexTreeType
    stride: TlaIndexTreeType
    origin_shape: TlaIndexTreeType | None = None
    layout_tag: str = "RowMajor"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout_tag", str(self.layout_tag).strip())
        if not self.layout_tag:
            raise ValueError("TlaLayoutDescriptor layout_tag must be non-empty")

    def to_asm(self) -> str:
        return str(self.to_mlir_type())

    def to_mlir_type(self, context: mlir_ir.Context | None = None) -> mlir_ir.Type:
        ctx = _tla_type_context(context)
        origin_tree = self.origin_shape.tree if self.origin_shape is not None else None
        return _tla_type_bridge.layout_type_get(
            ctx, self.shape.tree, self.stride.tree, origin_tree, self.layout_tag
        )


@dataclass(frozen=True)
class TlaVectorSSATypeDescriptor:
    """Internal structured descriptor for the TLA vector SSA type."""

    valid_lanes: int | None
    element_type: str

    def __post_init__(self) -> None:
        lanes = self.valid_lanes
        if lanes is not None:
            if isinstance(lanes, bool) or not isinstance(lanes, int):
                raise TypeError(
                    "TlaVectorSSATypeDescriptor valid_lanes must be int or None"
                )
            if lanes <= 0:
                raise ValueError(
                    "TlaVectorSSATypeDescriptor valid_lanes must be positive"
                )
        element_type = str(self.element_type).strip().lower()
        if not element_type:
            raise ValueError(
                "TlaVectorSSATypeDescriptor element_type must be non-empty"
            )
        element_bytes = dtype_size_bytes(element_type)
        if element_type == "i1" or element_bytes <= 0 or 256 % element_bytes != 0:
            raise ValueError(f"unsupported VectorSSA element type {element_type!r}")
        capacity = 256 // element_bytes
        if lanes is not None and lanes > capacity:
            raise ValueError(
                f"VectorSSA valid_lanes must be <= {capacity} for {element_type}, got {lanes}"
            )
        object.__setattr__(self, "element_type", element_type)

    def element_mlir_type(self, context: mlir_ir.Context | None = None) -> mlir_ir.Type:
        ctx = _tla_type_context(context)
        with ctx:
            return Numeric.from_dtype_token(self.element_type).mlir_type(ctx)

    def to_mlir_type(self, context: mlir_ir.Context | None = None) -> mlir_ir.Type:
        ctx = _tla_type_context(context)
        return _tla_type_bridge.vector_ssa_type_get(
            ctx,
            self.valid_lanes,
            self.element_mlir_type(ctx),
        )


@dataclass(frozen=True)
class TlaMaskSSATypeDescriptor:
    """Structured descriptor for a register-resident predicate mask."""

    physical_lanes: int

    def __post_init__(self) -> None:
        if isinstance(self.physical_lanes, bool) or not isinstance(
            self.physical_lanes, int
        ):
            raise TypeError("TlaMaskSSATypeDescriptor physical_lanes must be int")
        if self.physical_lanes not in (32, 64, 128, 256):
            raise ValueError(
                "TlaMaskSSATypeDescriptor physical_lanes must be one of "
                f"32, 64, 128, or 256, got {self.physical_lanes}"
            )

    def to_mlir_type(self, context: mlir_ir.Context | None = None) -> mlir_ir.Type:
        return _tla_type_bridge.mask_ssa_type_get(
            _tla_type_context(context), self.physical_lanes
        )


@dataclass(frozen=True)
class TlaTensorTypeDescriptor:
    """Structured Python descriptor for ``!tla.tensor`` metadata.

    The Python frontend uses this as its data model. Text is produced only at the MLIR API
    boundary because this repo currently exposes Tla op bindings, not Tla TypeDef builders.
    """

    layout: TlaLayoutDescriptor
    coord: TlaIndexTree
    element_type: str
    addrspace: str
    ptr_alignment: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "coord", _normalize_index_tree(self.coord))
        object.__setattr__(self, "element_type", str(self.element_type).strip())
        object.__setattr__(self, "addrspace", str(self.addrspace).strip())
        object.__setattr__(self, "ptr_alignment", int(self.ptr_alignment))
        if not self.element_type:
            raise ValueError("TlaTensorTypeDescriptor element_type must be non-empty")
        if not self.addrspace:
            raise ValueError("TlaTensorTypeDescriptor addrspace must be non-empty")
        if self.ptr_alignment < 1:
            raise ValueError("TlaTensorTypeDescriptor ptr_alignment must be positive")

    @property
    def shape(self) -> TlaIndexTree:
        return self.layout.shape.tree

    @property
    def stride(self) -> TlaIndexTree:
        return self.layout.stride.tree

    @property
    def origin_shape(self) -> TlaIndexTree:
        if self.layout.origin_shape is None:
            raise ValueError("Tla tensor layout must carry an origin shape")
        return self.layout.origin_shape.tree

    @property
    def layout_tag(self) -> str:
        return self.layout.layout_tag

    @property
    def ptr_asm(self) -> str:
        ctx = _tla_type_context()
        return str(
            PtrType.get(self.element_mlir_type(ctx), self.addrspace, self.ptr_alignment)
        )

    def element_mlir_type(self, context: mlir_ir.Context | None = None) -> mlir_ir.Type:
        ctx = _tla_type_context(context)
        with ctx:
            return Numeric.from_dtype_token(self.element_type).mlir_type(ctx)

    def to_asm(self) -> str:
        return str(self.to_mlir_type())

    def to_mlir_type(self, context: mlir_ir.Context | None = None) -> mlir_ir.Type:
        ctx = _tla_type_context(context)
        return _tla_type_bridge.tensor_type_get(
            ctx,
            shape=self.shape,
            stride=self.stride,
            coord=self.coord,
            origin_shape=self.origin_shape,
            element_type=self.element_mlir_type(ctx),
            addrspace=self.addrspace,
            layout=self.layout_tag,
            ptr_alignment=self.ptr_alignment,
        )

    def with_updates(self, **updates: Any) -> "TlaTensorTypeDescriptor":
        layout_updates: dict[str, Any] = {}
        for source_key, target_key, kind in (
            ("shape", "shape", "shape"),
            ("stride", "stride", "stride"),
            ("origin_shape", "origin_shape", "shape"),
        ):
            if source_key in updates:
                layout_updates[target_key] = TlaIndexTreeType(
                    kind, updates.pop(source_key)
                )
        if "layout_tag" in updates:
            layout_updates["layout_tag"] = updates.pop("layout_tag")
        layout = self.layout
        if layout_updates:
            layout_values = {
                "shape": layout.shape,
                "stride": layout.stride,
                "origin_shape": layout.origin_shape,
                "layout_tag": layout.layout_tag,
            }
            layout_values.update(layout_updates)
            layout = TlaLayoutDescriptor(**layout_values)
        values = {
            "layout": layout,
            "coord": self.coord,
            "element_type": self.element_type,
            "addrspace": self.addrspace,
            "ptr_alignment": self.ptr_alignment,
        }
        values.update(updates)
        return TlaTensorTypeDescriptor(**values)

    def metadata(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "stride": self.stride,
            "coord": self.coord,
            "origin_shape": self.origin_shape,
            "dtype": self.element_type,
            "addrspace": self.addrspace,
            "layout_tag": self.layout_tag,
        }


class TlaTensor:
    """Marker annotation for Tla tensor/view values."""


TlaTile = TlaTensor


class TlaAllocPtr:
    """Marker annotation for raw allocator-backed byte buffers."""


class TlaFlag:
    """Marker annotation for Tla flag values."""


class TlaCrossFlag:
    """Marker annotation for Tla cross-flag values."""


class TlaMutex:
    """Marker annotation for Tla mutex values."""


class TlaIndex:
    """Marker annotation for index values."""


class TlaShape:
    """Marker for ``make_shape`` / ``!tla.shape<...>``; ``tla.tile_view`` takes this value."""


class TlaCoord:
    """Marker for ``make_coord`` / ``!tla.coord<...>``; ``tla.tile_view`` takes this value."""


class TlaStride:
    """Marker annotation for ``make_stride`` / ``!tla.stride<...>``."""


class TlaLayout:
    """Marker for ``make_layout(shape, stride, *, origin_shape=...)`` / ``!tla.layout<!tla.shape<…>, !tla.stride<…>, !tla.shape<…>, RowMajor>``."""


class TlaRegion:
    """Marker annotation for Tla region stubs."""


_ANNOTATION_CATEGORY = {
    TlaTensor: "tensor",
    TlaAllocPtr: "alloc_ptr",
    TlaFlag: "flag",
    TlaCrossFlag: "cross_flag",
    TlaMutex: "mutex",
    TlaIndex: "index",
    TlaShape: "shape",
    TlaCoord: "coord",
    TlaStride: "stride",
    TlaLayout: "layout",
    TlaRegion: "region",
    PointerABC: "pointer",
}

from .base_dsl.op import annotation_to_category, register_annotation_category

for _annotation, _category in _ANNOTATION_CATEGORY.items():
    register_annotation_category(_annotation, _category)
del _annotation, _category

_TENSOR_DTYPE_SIZES: dict[str, int] = {
    "i1": 1,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "u8": 1,
    "u16": 2,
    "u32": 4,
    "u64": 8,
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "f8e4m3fn": 1,
    "f8e5m2": 1,
    # An e8m0 shared exponent is one opaque byte.
    "f8e8m0": 1,
}


# Element widths in bits. Bits are the primitive: the bytes table cannot express
# a sub-byte type, and the packed fp4 formats are deliberately absent from it so
# byte-denominated callers (vector registers, masks, host storage) reject fp4
# outright rather than silently rounding it up to a byte.
_TENSOR_DTYPE_BITS: dict[str, int] = {
    **{name: size * 8 for name, size in _TENSOR_DTYPE_SIZES.items()},
    "f4e2m1": 4,
    "f4e1m2": 4,
}


def dtype_size_bits(dtype: str) -> int:
    """Storage size in bits for a Tla element type token.

    Unlike :func:`dtype_size_bytes` this can express the sub-byte types (``i4``),
    which is what lets a packed fp4 tile derive its C0 geometry from the element
    type. Unknown names return ``0``.
    """
    return int(_TENSOR_DTYPE_BITS.get(dtype.strip().lower(), 0))


def dtype_size_bytes(dtype: str) -> int:
    """Storage size in bytes for a Tla element type token (e.g. ``f16``, ``i32``).

    Spelling is normalized with :func:`str.strip` / :func:`str.lower`. Unknown names
    return ``0`` (same table as :class:`Tensor` host storage sizing).
    """
    return int(_TENSOR_DTYPE_SIZES.get(dtype.strip().lower(), 0))


def _flatten_int_leaves_tree(tree: Any) -> list[int]:
    """Preorder flatten of positive-int leaves (same leaf order as :func:`catlass.core_api._flatten_tla_tuple`)."""
    if isinstance(tree, (tuple, list)):
        out: list[int] = []
        for x in tree:
            out.extend(_flatten_int_leaves_tree(x))
        return out
    if isinstance(tree, int):
        return [tree]
    raise TypeError(f"tensor shape tree expects int leaves, got {type(tree).__name__}")


def _tree_structure_mask(tree: Any) -> Any:
    """Shape-only mask: nested tuples match; leaves are ``None`` placeholders."""
    if isinstance(tree, (tuple, list)):
        return tuple(_tree_structure_mask(x) for x in tree)
    return None


def _try_remap_stride_coord_trees(
    comp: tuple[Any, ...],
    origin_components: tuple[Any, ...],
    dtype: str,
    layout_token: str,
) -> tuple[tuple[Any, ...] | None, tuple[Any, ...] | None]:
    """Use :func:`catlass.core_api._remap_tensor_like_prefix_fields_for_layout_trees` for defaults.

    Returns ``(stride_tree, coord_tree)`` with ``None`` entries when remap is unavailable or
    when the remap stride tree does not match ``comp``'s nesting. The caller must raise if a
    ``None`` entry was needed as a default.
    """
    from .core_api import _remap_tensor_like_prefix_fields_for_layout_trees

    if (
        layout_token.strip() == "RowMajor"
        and len(comp) == 1
        and len(origin_components) == 1
        and not isinstance(comp[0], (tuple, list))
        and not isinstance(origin_components[0], (tuple, list))
    ):
        return ((1,), (0,))

    remapped = _remap_tensor_like_prefix_fields_for_layout_trees(
        origin_components, dtype.strip().lower(), layout_token.strip()
    )
    if remapped is None:
        return (None, None)
    stride_tree = remapped.stride
    coord_tree = remapped.coord
    stride_ok = _tree_structure_mask(stride_tree) == _tree_structure_mask(comp)
    return (stride_tree if stride_ok else None, coord_tree)


class RuntimeTensorError(RuntimeError):
    """Raised when tensor buffer binding or layout validation fails."""


def _flat_layout_leaves(
    tree: Any,
    *,
    allow_dynamic: bool = False,
    expected_rank: int | None = None,
) -> list[Any]:
    """Preorder flatten of layout shape/stride/origin leaves.

    By default leaves must be ``int`` (concrete layout). Pass
    ``allow_dynamic=True`` to preserve ``None`` dynamic markers after
    ``mark_*_dynamic``. Optionally require ``len(leaves) == expected_rank``.
    """

    def _visit(node: Any) -> list[Any]:
        if isinstance(node, (tuple, list)):
            out: list[Any] = []
            for child in node:
                out.extend(_visit(child))
            return out
        if node is None:
            if not allow_dynamic:
                raise TypeError(
                    "layout tree expects int leaves; pass allow_dynamic=True "
                    "to accept None dynamic markers"
                )
            return [None]
        if isinstance(node, int):
            return [node]
        raise TypeError(
            "layout tree expects int"
            + (" or None" if allow_dynamic else "")
            + f" leaves, got {type(node).__name__}"
        )

    leaves = _visit(tree)
    if expected_rank is not None and len(leaves) != expected_rank:
        raise RuntimeTensorError(
            f"layout tree rank mismatch: expected {expected_rank} leaves, "
            f"got {len(leaves)}"
        )
    return leaves


def _deduce_compact_stride_order(
    shape: tuple[int, ...] | list[int],
    strides: tuple[int, ...] | list[int],
    *,
    strict_unit_stride: bool = False,
) -> tuple[int, ...]:
    """Deduce outer→inner mode order for a compact layout.

    When ``strict_unit_stride`` is False (default, ``mark_compact_shape_dynamic``):
    prefer a unique ``stride==1`` mode as innermost; otherwise sort by ascending
    stride and reverse to outer→inner.

    When ``strict_unit_stride`` is True (``mark_layout_dynamic`` leading-dim
    deduction): require a resolvable unit-stride mode (unique, or unique among
    modes with ``shape>1``); otherwise raise. The unit-stride mode is innermost.
    """
    rank = len(shape)
    if rank == 0:
        return ()
    if len(strides) != rank:
        raise RuntimeTensorError(
            f"stride rank {len(strides)} does not match shape rank {rank}"
        )

    unit_dims = [index for index, stride in enumerate(strides) if int(stride) == 1]
    innermost: int | None = None
    if len(unit_dims) == 1:
        innermost = unit_dims[0]
    elif strict_unit_stride:
        if not unit_dims:
            raise RuntimeTensorError(
                "cannot deduce leading_dim: no dimension has stride 1"
            )
        sized = [index for index in unit_dims if int(shape[index]) > 1]
        if len(sized) != 1:
            raise RuntimeTensorError(
                "cannot deduce leading_dim: multiple dimensions have stride 1"
            )
        innermost = sized[0]
    else:
        order = sorted(range(rank), key=lambda index: (int(strides[index]), index))
        return tuple(reversed(order))

    remaining = [index for index in range(rank) if index != innermost]
    remaining.sort(key=lambda index: int(strides[index]), reverse=True)
    return tuple(remaining + [innermost])


def _replace_flat_leaves_in_tree(tree: Any, new_leaves: Iterable[Any]) -> Any:
    iterator = iter(new_leaves)

    def _visit(node: Any) -> Any:
        if isinstance(node, (tuple, list)):
            return tuple(_visit(child) for child in node)
        return next(iterator)

    return _visit(tree)


from .base_dsl.typing import (
    DslType,
    NumericMeta,
    IntegerMeta,
    FloatMeta,
    Integer,
    Float,
    cast,
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    BFloat16,
    Float32,
    Float8E4M3FN,
    Float8E5M2,
    Float4E2M1,
    Float4E1M2,
    Float8E8M0,
)

__all__ = [
    "AddressSpace",
    "PtrType",
    "TlaIndexTreeKind",
    "TlaIndexTreeLeaf",
    "TlaIndexTree",
    "TlaIndexTreeType",
    "TlaLayoutDescriptor",
    "TlaMaskSSATypeDescriptor",
    "TlaVectorSSATypeDescriptor",
    "TlaTensorTypeDescriptor",
    "TlaTensor",
    "TlaTile",
    "TlaFlag",
    "TlaCrossFlag",
    "TlaIndex",
    "TlaShape",
    "TlaCoord",
    "TlaStride",
    "TlaLayout",
    "TlaRegion",
    "RuntimeTensorError",
    "DslType",
    "NumericMeta",
    "IntegerMeta",
    "FloatMeta",
    "Numeric",
    "Integer",
    "Float",
    "cast",
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
    "annotation_to_category",
    "dtype_size_bytes",
    "dtype_size_bits",
]
