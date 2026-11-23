"""Tla tensor IR values and view helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]
from catlass._mlir.dialects import tla as _tla_ops_gen  # type: ignore[import-not-found]

from .. import _tla_type_bridge
from .. import runtime as _runtime
from ..base_dsl.op import dsl_user_op
from ..base_dsl.typing import Bool, Numeric
from ..params import LoadParams, StoreParams
from .typing import Tensor as TensorABC

if TYPE_CHECKING:
    from ..core_api import MaskSSA, VectorSSA


def _dynamic_metadata_values(type_tree: Any, metadata_tree: Any) -> list[Any]:
    """Return runtime values corresponding to dynamic leaves in a type tree."""
    if isinstance(type_tree, tuple):
        values: list[Any] = []
        for type_child, metadata_child in zip(type_tree, metadata_tree, strict=True):
            values.extend(_dynamic_metadata_values(type_child, metadata_child))
        return values
    return [metadata_tree] if type_tree is None else []


# AscendC ``LoadAlign/StoreAlign<__ubuf__ T*, MaskReg>``:
# ``SupportBytes<T, 1, 2, 4, 8>`` (b8/b16/b32/b64). ``T`` is address typing
# only; hardware path is still plds/psts.b8.
_MASK_UB_ELEM_BYTES = frozenset((1, 2, 4, 8))
# AscendC MaskReg width after plds (VL bits). Used when tile shape cannot hint N.
_MASK_DIST_US_DS_DEFAULT_LANES = 256


def _rank1_leaf_or_meta(
    type_leaves: list[Any],
    meta_leaves: list[Any],
    *,
    rank_msg: str,
    missing_msg: str,
    unresolved_msg: str,
) -> Any:
    """Resolve one rank-1 type leaf, falling back to metadata when the type is ``?``."""
    from ..execution_lowering import TlaLoweringError

    if len(type_leaves) != 1:
        raise TlaLoweringError(rank_msg.format(rank=len(type_leaves)))
    value: Any = type_leaves[0]
    if value is None:
        if len(meta_leaves) != 1:
            raise TlaLoweringError(missing_msg.format(meta=meta_leaves))
        value = meta_leaves[0]
    if value is None:
        raise TlaLoweringError(unresolved_msg)
    return value


def _tile_extent_and_elem_coord(tile: _Tensor) -> tuple[Any, Any]:
    """Return ``(shape_extent, element_coord)`` for a rank-1 UPDATE window tile.

    ``shape_extent`` is the layout/addressing window size. UPDATE HIVM
    count/advance uses :func:`_tile_origin_extent` (``origin_shape``) instead.
    """
    from ..core_api import (
        _as_value,
        _flatten_tla_tuple,
        _tla_tensor_type_for_mlir_value,
    )

    desc = _tla_tensor_type_for_mlir_value(_as_value(tile))
    extent = _rank1_leaf_or_meta(
        list(_flatten_tla_tuple(desc.shape)),
        list(_flatten_tla_tuple(tile.shape)),
        rank_msg="post_mode=update currently supports rank-1 tiles only, got shape rank {rank}",
        missing_msg="post_mode=update dynamic tile extent metadata is missing; got {meta!r}",
        unresolved_msg=(
            "post_mode=update could not resolve the tile extent; use "
            "tile_view(parent, make_shape(...), ...) with a static or dynamic rank-1 shape"
        ),
    )
    elem_coord = _rank1_leaf_or_meta(
        list(_flatten_tla_tuple(desc.coord)),
        list(_flatten_tla_tuple(tile.coord)),
        rank_msg="post_mode=update requires a 1D tile coord, got rank {rank}",
        missing_msg="post_mode=update dynamic coord metadata is missing; got {meta!r}",
        unresolved_msg="post_mode=update could not resolve the tile coord",
    )
    return extent, elem_coord


def _tile_origin_extent(tile: _Tensor) -> Any:
    """Rank-1 ``origin_shape`` leaf: UPDATE tile_extent (HIVM count/advance)."""
    from ..core_api import (
        _as_value,
        _flatten_tla_tuple,
        _tla_tensor_type_for_mlir_value,
    )

    desc = _tla_tensor_type_for_mlir_value(_as_value(tile))
    return _rank1_leaf_or_meta(
        list(_flatten_tla_tuple(desc.origin_shape)),
        list(_flatten_tla_tuple(tile.origin_shape)),
        rank_msg=(
            "post_mode=update currently supports rank-1 origin_shape only, "
            "got origin rank {rank}"
        ),
        missing_msg="post_mode=update dynamic origin_shape metadata is missing; got {meta!r}",
        unresolved_msg=(
            "post_mode=update could not resolve origin_shape; pass "
            "make_layout(..., origin_shape=...) on the first UPDATE window"
        ),
    )


def _unalign_update_stride_value(
    params: Any, window: _Tensor, *, element_type: Any
) -> Any:
    """Default ``0`` → tile_extent; effective stride must be in ``(0, VL]``."""
    from ..base_dsl.typing import as_numeric
    from ..core_api import _as_value, _const_int_value, _vector_lane_count
    from ..execution_lowering import TlaLoweringError
    from ..types import dtype_size_bytes

    vl = _vector_lane_count(dtype_size_bytes(str(element_type).strip().lower()))
    static_stride = _const_int_value(params.post_update_stride)

    if static_stride == 0:
        extent = _tile_origin_extent(window)
        effective = _const_int_value(extent)
        result = _as_value(as_numeric(extent))
    elif static_stride is not None:
        effective = static_stride
        result = _as_value(params.post_update_stride)
    else:
        # Dynamic explicit stride: no static bound check.
        return _as_value(params.post_update_stride)

    if effective is not None and not 0 < effective <= vl:
        raise TlaLoweringError(
            "UnAlign post_mode=update post_update_stride must be in "
            f"(0, VL] for the dtype (VL={vl} elements), got "
            f"{effective!r} (post_update_stride={params.post_update_stride!r})"
        )
    return result


def _require_update_origin_window(
    tile: _Tensor,
    *,
    loc: mlir_ir.Location | None = None,
) -> _Tensor:
    """Reject UPDATE load/store when the tile is not an origin window (``coord=0``).

    Cut the needed window with ``tile_view`` (or equivalent) so the tile itself
    is an origin window. UPDATE does not automatically fold non-zero coordinates.
    """
    del loc
    from ..core_api import (
        _as_value,
        _const_int_value,
        _flatten_tla_tuple,
        _tla_tensor_type_for_mlir_value,
    )
    from ..execution_lowering import TlaLoweringError

    if not isinstance(tile, _Tensor):
        return tile
    try:
        desc = _tla_tensor_type_for_mlir_value(_as_value(tile))
        type_leaves = list(_flatten_tla_tuple(desc.coord))
        _, elem_coord = _tile_extent_and_elem_coord(tile)
    except TlaLoweringError:
        return tile
    type_leaf = type_leaves[0] if len(type_leaves) == 1 else None
    if (
        isinstance(type_leaf, int)
        and not isinstance(type_leaf, bool)
        and type_leaf == 0
    ):
        return tile
    if _const_int_value(elem_coord) == 0:
        return tile
    raise TlaLoweringError(
        "post_mode=update requires an origin window (coord=0); cut the "
        "needed tile with tile_view (or equivalent) before load/store"
    )


def _require_update_unit_stride(tile: _Tensor) -> None:
    """Reject UPDATE when any layout stride leaf is not statically 1.

    ``_make_update_tile_from_ptr`` rebuilds the next window with
    ``make_stride(1)``; non-unit (or dynamic) strides would be silently dropped.
    """
    from ..core_api import (
        _as_value,
        _const_int_value,
        _flatten_tla_tuple,
        _tla_tensor_type_for_mlir_value,
    )
    from ..execution_lowering import TlaLoweringError

    if not isinstance(tile, _Tensor):
        return
    desc = _tla_tensor_type_for_mlir_value(_as_value(tile))
    type_leaves = list(_flatten_tla_tuple(desc.stride))
    meta_leaves = list(_flatten_tla_tuple(tile.stride))
    if not type_leaves:
        raise TlaLoweringError(
            "post_mode=update requires a contiguous layout (all strides == 1); "
            "got an empty stride tree"
        )
    for index, type_leaf in enumerate(type_leaves):
        leaf: Any = type_leaf
        if leaf is None:
            leaf = meta_leaves[index] if index < len(meta_leaves) else None
        static = None if isinstance(leaf, bool) else _const_int_value(leaf)
        if static != 1:
            raise TlaLoweringError(
                "post_mode=update requires a contiguous layout "
                f"(all strides == 1), got stride={tile.stride!r}"
            )


def _make_update_tile_from_ptr(
    ptr: Any,
    template: _Tensor,
    *,
    loc: mlir_ir.Location | None = None,
) -> _Tensor:
    """Build next UPDATE window from a UB ptr at coord 0 (keeps template origin)."""
    from ..base_dsl.typing import as_numeric
    from ..core_api import (
        _Pointer,
        make_coord,
        make_layout,
        make_shape,
        make_stride,
        make_tensor,
    )
    from ..execution_lowering import TlaLoweringError

    if not isinstance(ptr, _Pointer):
        ptr = _Pointer(ptr)

    _require_update_unit_stride(template)
    extent, _ = _tile_extent_and_elem_coord(template)
    shape_extent = as_numeric(extent)
    origin_extent = as_numeric(_tile_origin_extent(template))
    tag_name = str(template.layout_tag)
    import catlass.tla as tla  # noqa: PLC0415

    layout_tag = getattr(tla.arch, tag_name, None)
    if layout_tag is None and tag_name != "RowMajor":
        raise TlaLoweringError(
            "post_mode=update requires a tla.arch layout tag on "
            f"the template tile, got layout_tag={tag_name!r}"
        )
    layout = make_layout(
        make_shape(shape_extent, loc=loc),
        make_stride(1, loc=loc),
        origin_shape=make_shape(origin_extent, loc=loc),
        layoutTag=layout_tag,
        loc=loc,
    )
    return make_tensor(ptr, layout, make_coord(as_numeric(0), loc=loc), loc=loc)


# Align / UnAlign PostMode UPDATE: b8/b16/b32/b64 widths (signed/unsigned
# integers, Bool/i1, fp8/hif8/e8m0/fp4x2 on b8, f16/bf16/f32, and f64). Align
# byte encodings UPDATE as s8 bits (no ``vldsx1.post.v256f8*``); UnAlign f8e4/e5
# keep native ``…post.f8e*``, while Bool/e8m0/fp4x2/hif8 also alias s8 (AscendC →
# int8_t/uint8_t). Bool/f64 are move-only (no arithmetic on the register).
# AscendC ``TypeGet<double>`` → ``vector_u64`` (same b64 u32+stride*2 path as i64).
_UPDATE_SUPPORTED_DTYPES = frozenset(
    {
        "i1",
        "i8",
        "u8",
        "f8e4m3fn",
        "f8e5m2",
        "hif8",
        "f8e8m0",
        "f4e2m1x2",
        "f4e1m2x2",
        "i16",
        "u16",
        "f16",
        "bf16",
        "i32",
        "u32",
        "f32",
        "i64",
        "u64",
        "f64",
    }
)


def _require_update_supported_dtype(element_type: Any) -> None:
    """Reject UPDATE tiles whose element type is not wired for `.post` emit."""
    from ..execution_lowering import TlaLoweringError
    from ..types import normalize_element_type_token

    dtype = normalize_element_type_token(element_type)
    if dtype not in _UPDATE_SUPPORTED_DTYPES:
        raise TlaLoweringError(
            "post_mode=update currently supports element types "
            "b8/b16/b32/b64 (Bool/i1, i/u integers, f8e4m3fn/f8e5m2/hif8/"
            "f8e8m0/f4e2m1x2/f4e1m2x2, f16/bf16/f32, and f64) only, "
            f"got {element_type}"
        )


def _require_update_store_matching_dtype(
    value_val: Any, dest_element_type: Any
) -> None:
    """Check the register type against the UB element type before emitting UPDATE."""
    from ..core_api import _vector_ssa_type_for_mlir_value
    from ..execution_lowering import TlaLoweringError
    from ..types import normalize_element_type_token

    src_elem = normalize_element_type_token(
        _vector_ssa_type_for_mlir_value(value_val).element_type
    )
    dst_elem = normalize_element_type_token(dest_element_type)
    if src_elem != dst_elem:
        raise TlaLoweringError(
            "post_mode=update store source and destination element types must match "
            f"(got source {src_elem}, dest {dst_elem})"
        )


def _validate_post_update_params(params: Any) -> bool:
    """Return True if UPDATE is requested; raise on illegal combinations."""
    from ..core_api import _const_int_value
    from ..execution_lowering import TlaLoweringError
    from ..params import (
        BlockLoadParams,
        BlockStoreParams,
        LoadDist,
        PostMode,
        StoreDist,
        UnalignStoreParams,
    )

    # Params without PostMode fields are never UPDATE.
    if not hasattr(params, "post_mode"):
        return False
    if params.post_mode == PostMode.POST_MODE_UPDATE_NBYTES:
        if not isinstance(params, UnalignStoreParams):
            raise TlaLoweringError(
                "POST_MODE_UPDATE_NBYTES requires UnalignStoreParams"
            )
        if _const_int_value(params.post_update_stride) != 0:
            raise TlaLoweringError(
                "POST_MODE_UPDATE_NBYTES uses the recorded byte count. "
                "Leave post_update_stride at 0"
            )
        return True
    # BlockLoad NORMAL: nonzero repeat_stride is the vsldb pre-offset.
    # BlockStore NORMAL: no pre-offset path — nonzero must not be ignored.
    # Block UPDATE: repeat_stride is the vsldb/vsstb POST repeat stride.
    if isinstance(params, (BlockLoadParams, BlockStoreParams)):
        stride = params.repeat_stride
        static_stride = None if isinstance(stride, bool) else _const_int_value(stride)
        if params.post_mode != PostMode.POST_MODE_UPDATE:
            # static_stride is None for runtime values; NORMAL must stay 0.
            if isinstance(params, BlockStoreParams) and static_stride != 0:
                raise TlaLoweringError(
                    "BlockStoreParams.repeat_stride requires "
                    "post_mode=update; NORMAL BlockStore has no pre-offset "
                    f"(got repeat_stride={params.repeat_stride!r})"
                )
            return False
        if static_stride is not None and not 0 < static_stride <= 0xFFFF:
            raise TlaLoweringError(
                "DATA_BLOCK_COPY post_mode=update requires "
                "repeat_stride in [1, 65535] (32B DataBlocks), got "
                f"{params.repeat_stride!r}"
            )
        block_stride = getattr(params, "block_stride", 0)
        if (
            isinstance(block_stride, int)
            and not isinstance(block_stride, bool)
            and not 0 <= block_stride <= 0xFFFF
        ):
            raise TlaLoweringError(
                "DATA_BLOCK_COPY block_stride must be in [0, 65535], got "
                f"{block_stride}"
            )
        return True

    is_update = params.post_mode == PostMode.POST_MODE_UPDATE
    stride = params.post_update_stride
    static_stride = None if isinstance(stride, bool) else _const_int_value(stride)
    if is_update:
        load_dist = getattr(params, "load_dist", None)
        if load_dist is not None and load_dist != LoadDist.DIST_NORM:
            raise NotImplementedError(
                f"post_mode=update with load_dist={load_dist!r} is not supported yet"
            )
        store_dist = getattr(params, "store_dist", None)
        if store_dist is not None and store_dist != StoreDist.DIST_NORM:
            raise NotImplementedError(
                f"post_mode=update with store_dist={store_dist!r} is not supported yet"
            )
        if static_stride is not None and static_stride < 0:
            raise TlaLoweringError(
                "post_mode=update post_update_stride must be >= 0 "
                "(0 = default to tile_extent / origin_shape)"
            )
    elif static_stride is None or static_stride != 0:
        raise TlaLoweringError(
            "post_update_stride requires post_mode=update, "
            f"got post_mode={params.post_mode!r} with "
            f"post_update_stride={params.post_update_stride!r}"
        )
    return is_update


class _Tensor(TensorABC):
    """Frontend proxy for an SSA ``!tla.tensor`` value."""

    def __init__(self, value: mlir_ir.Value) -> None:
        if not isinstance(value, mlir_ir.Value):
            raise TypeError(
                f"Tensor value expects mlir.ir.Value, got {type(value).__name__}"
            )
        if not _tla_type_bridge.type_is_tensor(value.type):
            raise TypeError(f"Tensor value expects !tla.tensor<...>, got {value.type}")
        self.value = value
        self.__tla_category__ = "tensor"
        _runtime._bind_frontend_value(self, value)
        _runtime._bind_frontend_category(self, "tensor")
        _runtime._bind_frontend_category(value, "tensor")

    def __tla_type__(self) -> str:
        return str(self.value.type)

    def __get_mlir_types__(self, context: mlir_ir.Context | None = None) -> list[Any]:
        del context
        return [self.value.type]

    def __extract_mlir_values__(self) -> list[Any]:
        """Flatten this tensor and the SSA leaves of its dynamic metadata.

        UPDATE windows are ordinary tiles: ``tile.load`` / ``tile.store`` with
        ``post_mode=update`` build the next window with ``make_tensor`` and
        the loop yields that tensor, the same packing as any other carried
        tensor. UnAlign ureg is **not** packed on the tensor: Python emits
        UPDATE without ureg; ``tla-unalign-ureg`` tracks load/store ureg
        separately, fills each UPDATE's ureg input, inserts
        ``load_unalign_pre`` / ``create_ureg`` when needed (before a *clean*
        ``scf.for`` when the tile is already that for's iter_arg; ``for``⊃``if``
        or a same-window load after store keeps per-trip dual-instr), and
        places ``store_unalign_post`` after the store ureg that is not taken
        by a later store UPDATE.
        """
        from ..core_api import _tla_tensor_type_for_mlir_value

        tensor_type = _tla_tensor_type_for_mlir_value(self.value)
        values: list[Any] = [self.value]
        for field, type_tree in (
            ("shape", tensor_type.shape),
            ("stride", tensor_type.stride),
            ("coord", tensor_type.coord),
            ("origin_shape", tensor_type.origin_shape),
        ):
            values.extend(
                _dynamic_metadata_values(
                    type_tree, _tensor_metadata_field(self.value, field)
                )
            )
        return values

    def __new_from_mlir_values__(self, values: list[Any]) -> _Tensor:
        """Rebuild an SCF-carried tensor and register its selected metadata."""
        from ..core_api import (
            _metadata_from_type_tree,
            _register_tla_tensor_metadata,
            _register_tla_tensor_type,
            _tla_tensor_type_for_mlir_value,
        )

        expected_count = len(self.__extract_mlir_values__())
        if len(values) != expected_count:
            raise ValueError(
                f"Tensor expects {expected_count} MLIR values, got {len(values)}"
            )
        result = _Tensor(values[0])
        tensor_type = _tla_tensor_type_for_mlir_value(self.value)
        _register_tla_tensor_type(result.value, tensor_type)

        from ..base_dsl.typing import as_numeric

        dynamic_values = iter(as_numeric(value) for value in values[1:])
        metadata = {
            "shape": _metadata_from_type_tree(tensor_type.shape, dynamic_values),
            "stride": _metadata_from_type_tree(tensor_type.stride, dynamic_values),
            "coord": _metadata_from_type_tree(tensor_type.coord, dynamic_values),
            "origin_shape": _metadata_from_type_tree(
                tensor_type.origin_shape, dynamic_values
            ),
            "dtype": tensor_type.element_type,
            "addrspace": tensor_type.addrspace,
            "layout_tag": tensor_type.layout_tag,
        }
        _register_tla_tensor_metadata(result.value, metadata)
        return result

    @property
    def shape(self) -> Any:
        return _tensor_metadata_field(self.value, "shape")

    @property
    def stride(self) -> Any:
        return _tensor_metadata_field(self.value, "stride")

    @property
    def coord(self) -> Any:
        return _tensor_metadata_field(self.value, "coord")

    @property
    def origin_shape(self) -> Any:
        return _tensor_metadata_field(self.value, "origin_shape")

    @property
    def dtype(self) -> str:
        return str(_tensor_metadata_field(self.value, "dtype"))

    @property
    def element_type(self) -> type[Numeric]:
        from ..core_api import _tla_tensor_descriptor_from_type_or_value

        parent = _tla_tensor_descriptor_from_type_or_value(self.value)
        return Numeric.from_mlir_type(parent.element_mlir_type())

    @property
    def addrspace(self) -> str:
        return str(_tensor_metadata_field(self.value, "addrspace"))

    @property
    def layout_tag(self) -> str:
        return str(_tensor_metadata_field(self.value, "layout_tag"))

    @property
    def ptr(self) -> Any:
        """Return the backing ``!tla.ptr`` of this tensor.

        The result is a :class:`~catlass.core_api._Pointer` and supports element-count
        offset arithmetic via ``+`` (e.g. ``a.ptr + 16`` advances by 16 elements), which
        can be fed to :func:`tla.make_tensor` to construct a tensor at an offset address.
        """
        from ..base_dsl.op import _capture_user_loc
        from ..core_api import _as_value, _emit_tensor_ptr

        loc = (
            _capture_user_loc()
            if _runtime._current_frontend_state() is not None
            else None
        )
        return _emit_tensor_ptr(_as_value(self), loc)

    @staticmethod
    def _prepare_scalar_indices(
        crd: Any,
        desc: Any,
        *,
        addrspace: str,
        action: str,
    ) -> list[mlir_ir.Value]:
        """Validate and materialize indices used by GM or UB scalar access."""
        from ..core_api import _as_index_value, _flatten_tla_tuple
        from ..execution_lowering import TlaLoweringError

        indices = crd if isinstance(crd, tuple) else (crd,)
        shape = _flatten_tla_tuple(desc.shape)
        if addrspace == "ub":
            _runtime._require_enclosing_region(f"scalar_{action}", "vector")
        if len(shape) not in (1, 2) or len(indices) != len(shape):
            raise TlaLoweringError(
                f"tla.scalar_{action} index rank must match tensor logical rank "
                f"(shape rank {len(shape)}, index rank {len(indices)})"
            )

        if addrspace == "ub":
            if any(isinstance(index, bool) for index in indices):
                raise TlaLoweringError(
                    "scalar tensor index must be an integer or index SSA"
                )
            origin_shape = _flatten_tla_tuple(desc.origin_shape)
            if len(origin_shape) != len(shape):
                raise TlaLoweringError(
                    f"scalar tensor {action} origin rank must match tensor rank"
                )
            for index, dim in zip(indices, origin_shape):
                if isinstance(index, int) and isinstance(dim, int):
                    if dim <= 0 or index < 0 or index >= dim:
                        raise TlaLoweringError(
                            f"scalar tensor index {index} is out of bounds for length {dim}"
                        )

        try:
            return [_as_index_value(index) for index in indices]
        except TlaLoweringError as exc:
            if addrspace == "ub":
                raise TlaLoweringError(
                    "scalar tensor index must be an integer or index SSA"
                ) from exc
            raise

    @dsl_user_op
    def load(
        self,
        params: LoadParams | None = None,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> MaskSSA | VectorSSA | tuple[VectorSSA, VectorSSA] | tuple[VectorSSA, _Tensor]:
        """Directory: Data Movement
        Description:
            Load this UB tensor tile into vector or mask SSA (`tile.load`).

            Parameters:
            - `params` (`LoadParams | None`): Load mode. Optional, default `None`.
              - `None` / `NormalLoadParams` / `UnalignLoadParams` /
                `BlockLoadParams` → `VectorSSA` (or a pair for
                `DIST_DINTLV_B8` / `DIST_DINTLV_B16` / `DIST_DINTLV_B32`).
              - `MaskLoadParams` → `MaskSSA`.
              - `NormalLoadParams`:
                - `load_dist` (`str`): `tla.params.LoadDist` token. Default
                  `DIST_NORM`.

                  | Value | Meaning |
                  |---|---|
                  | `DIST_NORM` | Default contiguous load |
                  | `DIST_BRC_B8` | Broadcast one b8 element to all 256 lanes |
                  | `DIST_BRC_B32` | Broadcast one b32 element to all lanes |
                  | `DIST_DINTLV_B8` | Load 512 bytes; even/odd b8 elements fill two 256-lane vectors |
                  | `DIST_DINTLV_B16` | Load 512 bytes; return two 128-lane b16 registers, even positions first and odd positions second |
                  | `DIST_DINTLV_B32` | b32 deinterleave (f32/i32/u32); dual results |
                  | `DIST_US_B8` / `DIST_US_B16` | 2x up-sample |
                  | `DIST_DS_B8` / `DIST_DS_B16` | Load 512 bytes and retain every other b8/b16 element into one full vector |
                  | `DIST_BRC_B16` | Broadcast one b16 element |
                  | `DIST_UNPACK_B8` | Load 128 b8 elements; insert one zero byte after each into a 256-lane vector |
                  | `DIST_UNPACK_B16` | Zero-extend b16 to double width |
                  | `DIST_UNPACK_B32` | Load 32 b32 elements; insert one zero b32 element after each into a 64-lane vector |
                  | `DIST_UNPACK4_B8` | Load exactly 64 bytes into register byte slots 0,4,...,252; zero the other register bytes |
                  | `DIST_E2B_B16` / `DIST_E2B_B32` | Element-to-DataBlock broadcast |
                  | `DIST_BLK` | One 32-byte DataBlock broadcast to all 8 |

                - `post_mode` (`str`): `POST_MODE_NORMAL` (default) or
                  `POST_MODE_UPDATE`.
                - `post_update_stride` (`Any`): Element stride when
                  `post_mode=POST_MODE_UPDATE`; `0` advances by `tile_extent`
                  (`origin_shape`). Default `0`.
              - `UnalignLoadParams`:
                - `post_mode` (`str`): `POST_MODE_NORMAL` (default) or
                  `POST_MODE_UPDATE`.
                - `post_update_stride` (`Any`): `0` (default) advances by
                  `origin_shape`; effective stride must be in `(0, VL]`. Default `0`.
              - `BlockLoadParams`:
                - `block_stride` (`int`): Heads apart in 32B DataBlocks; `0`
                  repeats the first block into all 8 slots. Default `0`.
                - `post_mode` (`str`): `POST_MODE_NORMAL` (default) or
                  `POST_MODE_UPDATE`.
                - `repeat_stride` (`Any`): Compile-time pre-offset in 32B
                  blocks; with `POST_MODE_UPDATE`, the post-update stride.
                  Default `0`.
              - `MaskLoadParams`:
                - `load_dist` (`str`): `tla.params.MaskLoadDist` token.
                  Default `DIST_NORM`.

                  | Value | Meaning |
                  |---|---|
                  | `DIST_NORM` | Bit-packed mask load |
                  | `DIST_US` | Bit upsample |
                  | `DIST_DS` | Bit downsample |

            Constraints:
            - Must be called inside a `@tla.kernel`-decorated kernel function.
            - Must be called inside `tla.vec.func(mode="simd")`; source tile must be UB.
            - The actual read range must stay within the usable UB address space.
            - Mask load ``DIST_NORM`` requires a b8/b16/b32/b64 scalar UB element type.
            - Each width-suffixed distribution requires the corresponding
              element storage width. b8 includes Bool and byte encodings.
              `DIST_BLK` supports b8/b16/b32. For b64, use `DIST_NORM`
              or a full 32-lane `UnalignLoadParams` load.
            - `DIST_DINTLV_B8` supports i8/u8/f8E4M3FN/f8E5M2/HiFloat8/E8M0 and packed
              Float4E2M1x2/Float4E1M2x2. Each register
              holds 256 one-byte elements; one load reads `2 * 256 = 512` bytes
              and returns two full vectors of the source type, even positions
              first and odd positions second. For input `[a0, a1, a2, a3, ...]`,
              the results are `[a0, a2, ...]` and `[a1, a3, ...]`. The source
              address must be 32-byte aligned and all 512 bytes must be valid,
              independently of the tile shape. For FP4x2, each element is one byte
              holding two FP4 values. The load deinterleaves these packed bytes
              without changing their two nibbles; each result holds 256 packed
              elements (512 FP4 values). E8M0 is transported as raw bytes, including
              the 0xFF NaN encoding, and can be written back with a normal aligned
              store; arithmetic and register casts are unsupported. Logical
              (unpacked) FP4 is not supported. Bool is supported as b8
              move-only transport (same as i8).
            - `DIST_DINTLV_B16` requires a 32-byte-aligned source address.
            - `DIST_UNPACK4_B8` requires a 1-byte element type and a
              32-byte-aligned source address.
            - `DIST_BRC_B8` / `DIST_DS_B8` / `DIST_UNPACK_B8` support
              Bool/i8/u8/f8E4M3FN/f8E5M2/hif8/f8e8m0/fp4x2; `DIST_DS_B16` supports
              f16/bf16/i16/u16; `DIST_UNPACK_B32` supports f32/i32/u32.
              These modes preserve the source element type and produce full
              256-byte vector registers.
              `DIST_BRC_B8` requires 1-byte source alignment; the other four
              require 32-byte alignment. `DIST_UNPACK_B8` / `DIST_UNPACK_B32` read 128 bytes;
              DS reads 512 bytes. The caller must provide the
              complete valid source range, independently of the tile shape.
            - `BlockLoadParams` follows AscendC ``DATA_BLOCK_COPY``
              (b8/b16/b32; b8 includes i8/u8, Bool as b8);
              `block_stride` and `repeat_stride` must fit in [0, 65535].
            - ``DIST_US`` / ``DIST_DS``: UB element type may be any
              b8/b16/b32/b64 scalar. The element type only types the UB
              address; the mask bit layout does not depend on element width.
              When the static tile-shape product is in ``{32,64,128,256}``,
              it is used as ``N`` on ``!tla.mask<N>``; otherwise ``N=256``.
            - Mask load byte-offset alignment (UB byte address =
              base + offset×sizeof(T)): ``DIST_NORM`` → ``VL/8``;
              ``DIST_US`` → ``VL/16``; ``DIST_DS`` → ``min(32, VL/4)``.
              On VL=256 this is 32B / 16B / 32B.
            - `POST_MODE_UPDATE` requires a rank-1 contiguous tile,
              `DIST_NORM`, and element type of width b8/b16/b32/b64
              (Bool, i/u integers, FP8, HiFloat8, E8M0, FP4x2, f16, bf16,
              f32 and f64). Returns
              `(vector, next_tile)`. Inside `tla.vec.func`, assign
              `next_tile` back with `tla.range` (`while` is not supported
              nested in `vec.func`). The window must already be an origin
              tile (`coord=0`); cut it with `tile_view` (or equivalent).
              UPDATE never automatically folds non-zero coordinates.
              `NormalLoadParams`: `post_update_stride=0` (default) advances
              by `tile_extent` (`origin_shape`); an explicit stride keeps
              fill / stride-advance decoupling. `UnalignLoadParams`: default
              `0`→`tile_extent`; effective stride must be in `(0, VL]`. On a clean
              `tla.range`, `tla-unalign-ureg` hoists Pre/Post outside the for;
              `for`⊃dynamic `if` keeps per-trip dual-instr.

            Example:
            ```python
            with tla.vec.func(mode="simd"):
                x_reg = x_ub.load()
                x_unalign = x_ub.load(tla.params.UnalignLoadParams())
                x_blk = x_ub.load(tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_BLK))
                x_gather = x_ub.load(tla.params.BlockLoadParams(block_stride=4))
                preg = mask_ub.load(
                    tla.params.MaskLoadParams(load_dist=tla.params.MaskLoadDist.DIST_DS)
                )
                y = tla.where(preg, x_reg, min_reg)
                # fp8_ub: 64 Float8E4M3FN elements at a 32-byte-aligned address.
                fp8_reg = fp8_ub.load(tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_UNPACK4_B8))
                fp32_reg = fp8_reg.to(tla.Float32)
            ```
        """
        from ..core_api import (
            MaskSSA,
            VectorSSA,
            _as_value,
            _coerce_type,
            _full_vector_ssa_descriptor,
            _mask_ssa_type_for_element_type,
            _op_error,
            _require_frontend_state,
            _tla_tensor_type_for_mlir_value,
            _vector_ssa_type_from_tensor_descriptor,
        )
        from ..execution_lowering import TlaLoweringError
        from ..params import (
            BlockLoadParams,
            LoadDist,
            MaskLoadDist,
            MaskLoadParams,
            NormalLoadParams,
            UnalignLoadParams,
        )
        from ..types import TlaMaskSSATypeDescriptor, dtype_size_bytes

        loc = _normalize_user_loc(loc)
        _require_frontend_state("load")
        _runtime._require_enclosing_region("load", "vec.func")

        source = _as_value(self)
        source_desc = _tla_tensor_type_for_mlir_value(source)
        if source_desc.addrspace.lower() != "ub":
            _op_error(
                "load",
                "invalid argument 'source' (position 0): expected addrspace ub, "
                f"got {source_desc.addrspace}",
            )

        # MaskSSA path: selected by MaskLoadParams.
        if isinstance(params, MaskLoadParams):
            allowed = (
                MaskLoadDist.DIST_NORM,
                MaskLoadDist.DIST_US,
                MaskLoadDist.DIST_DS,
            )
            if params.load_dist not in allowed:
                raise NotImplementedError(
                    f"unsupported MaskLoadDist {params.load_dist!r}; "
                    f"expected one of {allowed}"
                )
            elem = str(source_desc.element_type).strip().lower()
            # AscendC LoadAlign<T, MaskDist> uses T only for address typing;
            # SupportBytes<T,1,2,4,8> admits bool via sizeof(bool)==1 (b8).
            elem_bytes = dtype_size_bytes(elem)
            is_us_ds = params.load_dist in (
                MaskLoadDist.DIST_US,
                MaskLoadDist.DIST_DS,
            )
            # AscendC MaskDist: T ∈ b8/b16/b32/b64 for NORM/US/DS.
            if elem_bytes not in _MASK_UB_ELEM_BYTES:
                _op_error(
                    "load",
                    f"invalid argument 'source': expected b8/b16/b32/b64 "
                    f"scalar UB (AscendC LoadAlign<T, MaskDist>), got "
                    f"{source_desc.element_type}",
                )
            # DIST_US / DIST_DS: ``T`` does not set predicate width (AscendC
            # plds bit window is independent of sizeof(T)). Prefer a static
            # tile-shape product in {32,64,128,256} as the companion VL hint
            # (Softmax: i8/u32 tile of 64 → mask<64> with f32). Otherwise use
            # the full AscendC MaskReg width (256).
            from ..core_api import _flatten_tla_tuple

            lanes = None
            if is_us_ds:
                shape_leaves = list(_flatten_tla_tuple(source_desc.shape))
                prod = 1
                static_ok = True
                for dim in shape_leaves:
                    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                        static_ok = False
                        break
                    prod *= int(dim)
                if static_ok and prod in (32, 64, 128, 256):
                    lanes = prod
                else:
                    lanes = _MASK_DIST_US_DS_DEFAULT_LANES
            if lanes is None:
                lanes = _mask_ssa_type_for_element_type(elem).physical_lanes
            mask_desc = TlaMaskSSATypeDescriptor(physical_lanes=lanes)
            mask_ty = mask_desc.to_mlir_type(
                loc.context if loc is not None else mlir_ir.Context.current
            )
            load_kwargs: dict[str, Any] = {"loc": loc}
            if params.load_dist != MaskLoadDist.DIST_NORM:
                ctx = loc.context if loc is not None else mlir_ir.Context.current
                load_kwargs["load_dist"] = mlir_ir.Attribute.parse(
                    f"#tla.load_dist<{params.load_dist}>",
                    context=ctx,
                )
            return MaskSSA(_tla_ops_gen.load(mask_ty, None, source, **load_kwargs))

        if params is None:
            params = NormalLoadParams()
        elif not isinstance(
            params, (NormalLoadParams, UnalignLoadParams, BlockLoadParams)
        ):
            raise TlaLoweringError(
                "load params must be NormalLoadParams, UnalignLoadParams, "
                f"BlockLoadParams, or MaskLoadParams, got {type(params).__name__}"
            )

        if _validate_post_update_params(params):
            if isinstance(params, UnalignLoadParams):
                # UPDATE only; ``tla-unalign-ureg`` inserts Pre and fills ureg.
                window = _require_update_origin_window(self, loc=loc)
                _require_update_unit_stride(window)
                source = _as_value(window)
                source_desc = _tla_tensor_type_for_mlir_value(source)
                _require_update_supported_dtype(source_desc.element_type)
                result_type = _coerce_type(
                    _vector_ssa_type_from_tensor_descriptor(source_desc)
                )
                ptr_type = _tla_type_bridge.tensor_ptr_type_get(source.type)
                stride_val = _unalign_update_stride_value(
                    params, window, element_type=source_desc.element_type
                )
                ctx = loc.context if loc is not None else mlir_ir.Context.current
                ureg_type = mlir_ir.Type.parse("!tla.ureg", context=ctx)
                vec_val, next_ptr, _ureg_out = _tla_ops_gen.load_unalign_update(
                    result_type,
                    ptr_type,
                    ureg_type,
                    source,
                    stride_val,
                    loc=loc,
                )
                next_tile = _make_update_tile_from_ptr(next_ptr, window, loc=loc)
                return VectorSSA(vec_val), next_tile
            is_block = isinstance(params, BlockLoadParams)
            if not is_block and not isinstance(params, NormalLoadParams):
                raise TlaLoweringError(
                    "post_mode=update on tile.load only accepts NormalLoadParams, "
                    f"UnalignLoadParams, or BlockLoadParams, got {type(params).__name__}"
                )
            from ..base_dsl.typing import as_numeric
            from ..core_api import _const_int_value

            window = _require_update_origin_window(self, loc=loc)
            if not is_block:
                _require_update_unit_stride(window)
            source = _as_value(window)
            source_desc = _tla_tensor_type_for_mlir_value(source)
            if not is_block:
                _require_update_supported_dtype(source_desc.element_type)
            result_type = _coerce_type(
                _vector_ssa_type_from_tensor_descriptor(source_desc)
            )
            ptr_type = _tla_type_bridge.tensor_ptr_type_get(source.type)
            load_kwargs: dict[str, Any] = {"loc": loc}
            if is_block:
                load_kwargs["block_stride"] = _as_value(params.block_stride)
                stride_val = _as_value(params.repeat_stride)
            else:
                # Default 0 → tile_extent (origin_shape); explicit keeps decoupling.
                static_stride = _const_int_value(params.post_update_stride)
                if static_stride == 0:
                    stride_val = _as_value(as_numeric(_tile_origin_extent(window)))
                else:
                    stride_val = _as_value(params.post_update_stride)
            vec_val, next_ptr = _tla_ops_gen.load_update(
                result_type,
                ptr_type,
                source,
                stride_val,
                **load_kwargs,
            )
            return VectorSSA(vec_val), _make_update_tile_from_ptr(
                next_ptr, window, loc=loc
            )

        is_dintlv = isinstance(params, NormalLoadParams) and params.load_dist in (
            LoadDist.DIST_DINTLV_B8,
            LoadDist.DIST_DINTLV_B16,
            LoadDist.DIST_DINTLV_B32,
        )
        dist_bytes = (
            {
                LoadDist.DIST_BRC_B8: (1,),
                LoadDist.DIST_BRC_B16: (2,),
                LoadDist.DIST_BRC_B32: (4,),
                LoadDist.DIST_US_B8: (1,),
                LoadDist.DIST_US_B16: (2,),
                LoadDist.DIST_DS_B8: (1,),
                LoadDist.DIST_DS_B16: (2,),
                LoadDist.DIST_UNPACK_B8: (1,),
                LoadDist.DIST_UNPACK_B16: (2,),
                LoadDist.DIST_UNPACK_B32: (4,),
                LoadDist.DIST_UNPACK4_B8: (1,),
                LoadDist.DIST_E2B_B16: (2,),
                LoadDist.DIST_E2B_B32: (4,),
                LoadDist.DIST_DINTLV_B8: (1,),
                LoadDist.DIST_DINTLV_B16: (2,),
                LoadDist.DIST_DINTLV_B32: (4,),
                LoadDist.DIST_BLK: (1, 2, 4),
            }.get(params.load_dist)
            if isinstance(params, NormalLoadParams)
            else None
        )
        is_block_load = isinstance(params, BlockLoadParams)
        if is_block_load:
            # vsldb packs each stride into a 16-bit immediate field: reject
            # out-of-range values at compile time (Issue 519 acceptance).
            for stride_name, stride_value in (
                ("block_stride", params.block_stride),
                ("repeat_stride", params.repeat_stride),
            ):
                if not 0 <= stride_value <= 0xFFFF:
                    raise TlaLoweringError(
                        f"BlockLoadParams.{stride_name} must be in [0, 65535] "
                        "(16-bit immediate field), got "
                        f"{stride_value}"
                    )
        load_kwargs: dict[str, Any] = {"loc": loc}
        if isinstance(params, UnalignLoadParams):
            load_kwargs["unaligned_ub_access"] = True
        elif isinstance(params, BlockLoadParams):
            # block_stride / vsldb repeat_stride are passed as explicit kwargs
            # below (same as master) — do not also put them in load_kwargs.
            if not isinstance(params.block_stride, int) or isinstance(
                params.block_stride, bool
            ):
                raise TlaLoweringError(
                    "BlockLoadParams.block_stride must be a compile-time int, "
                    f"got {type(params.block_stride).__name__}"
                )
        elif (
            isinstance(params, NormalLoadParams)
            and params.load_dist != LoadDist.DIST_NORM
        ):
            ctx = loc.context if loc is not None else mlir_ir.Context.current
            load_kwargs["load_dist"] = mlir_ir.Attribute.parse(
                f"#tla.load_dist<{params.load_dist}>",
                context=ctx,
            )

        if dist_bytes is not None:
            elem = source_desc.element_type
            if dtype_size_bytes(elem) not in dist_bytes:
                widths = "/".join(map(str, dist_bytes))
                raise TlaLoweringError(
                    f"DIST_{params.load_dist.upper()} requires a {widths}-byte "
                    f"element type (got {elem})"
                )
            result_desc = _full_vector_ssa_descriptor(elem)
        elif is_block_load:
            result_desc = _full_vector_ssa_descriptor(source_desc.element_type)
        else:
            result_desc = _vector_ssa_type_from_tensor_descriptor(source_desc)

        result_type = _coerce_type(result_desc)
        if is_dintlv:
            results = _tla_ops_gen.load(result_type, result_type, source, **load_kwargs)
            return tuple(VectorSSA(result) for result in results)

        block_stride_attr = params.block_stride if is_block_load else None
        # BlockLoadParams.repeat_stride is vsldb repeat_stride (NORMAL
        # compile-time pre-offset in 32B DataBlocks).
        repeat_stride_attr = (
            params.repeat_stride
            if is_block_load and params.repeat_stride != 0
            else None
        )
        result = _tla_ops_gen.load(
            result_type,
            None,
            source,
            block_stride=block_stride_attr,
            repeat_stride=repeat_stride_attr,
            **load_kwargs,
        )
        return VectorSSA(result)

    @dsl_user_op
    def store(
        self,
        value: VectorSSA | MaskSSA | tuple[VectorSSA, VectorSSA],
        params: StoreParams | None = None,
        *,
        mask: MaskSSA | None = None,
        loc: mlir_ir.Location | None = None,
    ) -> _Tensor | None:
        """Directory: Data Movement
        Description:
            Store vector or mask SSA into this UB tensor tile (`tile.store`).

            Parameters:
            - `value` (`VectorSSA | MaskSSA | tuple[VectorSSA, VectorSSA]`):
              `VectorSSA` or `MaskSSA` to store. For `DIST_INTLV_B8` /
              `DIST_INTLV_B16` / `DIST_INTLV_B32`, pass a two-vector tuple
              `(even, odd)` (same types / full VL). Required.
            - `params` (`StoreParams | None`): Store mode. Optional, default
              `None`.
              - `None` / `NormalStoreParams` / `UnalignStoreParams` /
                `BlockStoreParams` → vector store.
              - `MaskStoreParams` → mask store.
              - `NormalStoreParams`:
                - `store_dist` (`str`): `tla.params.StoreDist` token. Default
                  `DIST_NORM`.

                  | Value | Meaning |
                  |---|---|
                  | `DIST_NORM` | Default contiguous store |
                  | `DIST_PACK_B16` | AscendC same-type PK_B16 (b16→b16, or b8→b8 after prior cast) |
                  | `DIST_PACK_B32` | AscendC same-type PK_B32 (b32→b32, or b16→b16 after prior cast) |
                  | `DIST_PACK_B64` | Pack the low 32 bits of i64, u64 or f64 lanes |
                  | `DIST_PACK4_B32` | Store register byte slots 0,4,...,252 to UB byte slots 0,1,...,63, subject to the mask |
                  | `DIST_INTLV_B8` | Interleave two b8 registers into consecutive dst elements |
                  | `DIST_INTLV_B16` | Interleave two b16 registers |
                  | `DIST_INTLV_B32` | Interleave two b32 registers |
                  | `DIST_FIRST_ELEMENT_B8` | Write only lane 0 (1-byte) |
                  | `DIST_FIRST_ELEMENT_B16` | Write only lane 0 (2-byte) |
                  | `DIST_FIRST_ELEMENT_B32` | Write only lane 0 (4-byte) |

                - `post_mode` (`str`): `POST_MODE_NORMAL` (default) or
                  `POST_MODE_UPDATE`.
                - `post_update_stride` (`Any`): Element stride when
                  `post_mode=POST_MODE_UPDATE`; `0` advances by `tile_extent`
                  (`origin_shape`). Default `0`.
              - `UnalignStoreParams`:
                - `post_mode` (`str`): `POST_MODE_NORMAL` (default),
                  `POST_MODE_UPDATE`, or `POST_MODE_UPDATE_NBYTES`.
                - `post_update_stride` (`Any`): With `POST_MODE_UPDATE`,
                  `0` advances by `origin_shape` and the effective stride
                  must be in `(0, VL]`. With `POST_MODE_UPDATE_NBYTES`,
                  use `0`. Default `0`.
              - `BlockStoreParams`:
                - `block_stride` (`int`): Heads apart in 32B DataBlocks.
                  Default `0`.
                - `post_mode` (`str`): `POST_MODE_NORMAL` (default) or
                  `POST_MODE_UPDATE`.
                - `repeat_stride` (`Any`): Used with `POST_MODE_UPDATE` as the
                  post-update stride in 32B blocks. Default `0`.
              - `MaskStoreParams`:
                - `store_dist` (`str`): `tla.params.MaskStoreDist` token.
                  Default `DIST_NORM`.
            - `mask` (`MaskSSA | None`): Optional predicate for vector stores;
              rejected by `UnalignStoreParams`, `MaskStoreParams`, first-element
              and interleaved modes. Default `None`. For `DIST_PACK4_B32`, masks use source lanes:
              4/2/1 lanes per destination byte for 1/2/4-byte sources.
              An omitted mask follows the destination's valid length;
              an explicit mask is not clipped to that length. Masked-out
              destination bytes remain unchanged.

            Constraints:
            - Must be called inside a `@tla.kernel`-decorated kernel function.
            - Must be called inside `tla.vec.func(mode="simd")`; destination tile must be UB.

            **`POST_MODE_NORMAL` mode**

            - `DIST_PACK4_B32` supports matching 4-byte element types.
              It also supports 1/2/4-byte source elements with a 1-byte
              destination. It writes at most 64 bytes to a 32-byte-aligned
              address. For FP32→FP8, use `RegSlot.ZERO` to place the
              converted values in byte slots 0,4,...,252 before storing.
            - `DIST_PACK_B16` / `DIST_PACK_B32` / `DIST_PACK_B64` require
              matching source and destination element types. B16 accepts
              1- or 2-byte elements, B32 accepts 2- or 4-byte elements, and
              B64 accepts 8-byte elements. Each destination element has room
              for two active source lanes.
            - `DIST_INTLV_B8` / `DIST_INTLV_B16` / `DIST_INTLV_B32` require a
              two-vector `value` tuple of identical full-VL registers matching
              the dest element type (b8: Bool/i8/u8/f8*/hif8/e8m0/fp4x2, b16:
              f16/bf16/i16/u16; b32: f32/i32/u32). Predicate masks are rejected.
            - First-element modes require the stored value and the dest tile to
              use the suffix-matching element width (b8 → 1-byte, b16 → 2-byte,
              b32 → 4-byte); passing a predicate mask raises an error.
            - `UnalignStoreParams` in NORMAL mode writes up to one vector of
              elements using the destination's valid length. In UPDATE mode,
              the effective stride sets both the write length and pointer advance.

            ```python
            with tla.vec.func(mode="simd"):
                y_ub.store(y_reg)
                y_ub.store(y_reg, tla.params.UnalignStoreParams())
                y_ub.store(y_reg, tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B16))
                # fp32_reg: 64 FP32 lanes; fp8_ub: 64 Float8E4M3FN elements.
                # fp8_ub starts at a 32-byte-aligned address.
                fp8_reg = fp32_reg.to(tla.Float8E4M3FN, tla.params.CastParams(
                    round_mode=tla.params.RoundMode.CAST_RINT,
                    sat_mode=tla.params.SatMode.SAT,
                    reg_slot=tla.params.RegSlot.ZERO))
                fp8_ub.store(fp8_reg, tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_PACK4_B32))
            ```

            - `POST_MODE_UPDATE` requires a rank-1 contiguous tile, `DIST_NORM`
              and matching source and destination element types of width
              b8/b16/b32/b64. It returns the next tile. Inside `tla.vec.func`, assign it back with
              `tla.range`. Same origin window (`coord=0`) contract as UPDATE
              `load`. `NormalStoreParams`: default `post_update_stride=0`
              advances by `tile_extent` (`origin_shape`); an explicit stride is
              decoupled. `UnalignStoreParams`: default `0`→`tile_extent`;
              effective stride must be in `(0, VL]`; clean for hoists Pre/Post.

            **`POST_MODE_UPDATE_NBYTES` mode**

            This mode appends data using the byte count recorded by `squeeze`.
            Before each append, call `squeeze(..., record_nbytes=True)` in the
            same SIMD region so hardware calculates and records the byte count.
            Before that count is overwritten, call `store` with
            `UnalignStoreParams(post_mode=POST_MODE_UPDATE_NBYTES)`.
            `store` takes that many bytes from the start of the source vector
            and writes them at the supplied window's base address plus the
            current write offset. It advances the offset and returns the
            supplied window.

            **Usage restrictions**

            - Use a contiguous one-dimensional UB window at `coord=0`. The
              source and destination element types must match. Supported types
              are `Float16`, `Float32`, `Int8`, `UInt8`, `Int16`, `UInt16`,
              `Int32`, and `UInt32`. Keep `post_update_stride` at the
              compile-time constant `0` and omit `mask` from `store`.
            - Appends in the same SIMD region share the write offset and
              pending tail. Tail writeback uses the destination base address
              of the last `store` executed with `POST_MODE_UPDATE_NBYTES`.
            - Reserve space for appends and tail writeback. From the first
              append through SIMD exit and synchronization, keep other reads
              and writes outside the compressed output byte range.
            - External calls must preserve the write offset and pending tail
              from initialization through tail writeback, and the byte count
              from when `squeeze` records it until the corresponding `store`
              uses it.

            **Example using `POST_MODE_UPDATE_NBYTES`**

            The loop appends `[7, 9, 12]` from `src_ub[0:3]` and `[20, 24]`
            from `src_ub[128:130]` to the same destination, `dst_ub`. In
            this example, the start address of `dst_ub` is 32-byte aligned.

            1. **Initialize.** On entry to the SIMD region, the compiler sets
               up a zero write offset and an empty buffer for pending bytes.
            2. **Compress and append.** In the first iteration,
               `squeeze(..., record_nbytes=True)` packs three Int16 elements
               and hardware records 6 bytes. The `store` leaves those bytes
               in the hardware buffer and advances the write offset to 6 bytes.
               In the second iteration, hardware records 4 bytes for two kept
               elements, and `store` adds them after the first 6. The 10
               bytes fall short of a 32-byte block, so they stay in the buffer
               as the pending tail. See the
               [`tla.squeeze` reference](language/simd/compress.md#tlasqueeze).
            3. **Write back the tail.** After the loop, the compiler inserts
               a writeback of the 10 buffered bytes to `dst_ub[0:5]`. After
               leaving the SIMD region and synchronizing, the first five
               elements of `dst_ub` are `[7, 9, 12, 20, 24]`.

            Example:
            ```python
            # src_ub and dst_ub are separate contiguous Int16 UB windows at coord=0.
            # src_ub contains 256 elements, and dst_ub contains 128.
            with tla.vec.func(mode="simd"):
                # Use the hardware-recorded byte count for each append.
                update = tla.params.UnalignStoreParams(
                    post_mode=tla.params.PostMode.POST_MODE_UPDATE_NBYTES,
                )
                block = tla.make_shape(128)
                for i in tla.range(2):
                    # Select source indices 0 through 127, then 128 through 255.
                    src_tile = tla.tile_view(src_ub, block, tla.make_coord(i))
                    # Keep the first 3 elements of the first tile and the first 2 of the second.
                    keep, _ = tla.update_mask(3 - i, dtype=tla.Int16)
                    # Pack the kept elements. At 2 bytes per Int16, hardware records 6, then 4 bytes.
                    packed = tla.squeeze(src_tile.load(), keep, record_nbytes=True)
                    # Pass dst_ub each time. Hardware advances the write offset.
                    dst_ub.store(packed, update)
            ```
        """
        from ..core_api import (
            _as_value,
            _mask_ssa_type_for_element_type,
            _mask_ssa_type_for_mlir_value,
            _op_error,
            _require_category,
            _require_frontend_state,
            _require_mask_matches_vector,
            _tla_tensor_type_for_mlir_value,
            _vector_ssa_type_for_mlir_value,
        )
        from ..execution_lowering import TlaLoweringError
        from ..params import (
            BlockStoreParams,
            MaskStoreDist,
            MaskStoreParams,
            NormalStoreParams,
            PostMode,
            StoreDist,
            UnalignStoreParams,
        )
        from ..types import (
            dtype_size_bytes,
            is_bool_element_type,
            normalize_element_type_token,
        )

        loc = _normalize_user_loc(loc)
        _require_frontend_state("store")
        _runtime._require_enclosing_region("store", "vec.func")

        if isinstance(params, UnalignStoreParams) and mask is not None:
            raise TlaLoweringError("UnalignStoreParams does not accept mask=")

        dest = _as_value(self)
        dest_desc = _tla_tensor_type_for_mlir_value(dest)
        if dest_desc.addrspace.lower() != "ub":
            _op_error(
                "store",
                "invalid argument 'dest' (position 0): expected addrspace ub, "
                f"got {dest_desc.addrspace}",
            )

        # Dual-source INTLV: ``tile.store((v0, v1), NormalStoreParams(...))``.
        value2_val = None
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise TlaLoweringError(
                    "intlv store expects a 2-tuple of VectorSSA values "
                    f"(got length {len(value)})"
                )
            _require_category("store", "value[0]", value[0], "vector_ssa", 1)
            _require_category("store", "value[1]", value[1], "vector_ssa", 1)
            value_val = _as_value(value[0])
            value2_val = _as_value(value[1])
        else:
            value_val = _as_value(value)

        # MaskSSA path: selected by MaskStoreParams.
        if isinstance(params, MaskStoreParams):
            if value2_val is not None:
                raise TlaLoweringError(
                    "store(..., MaskStoreParams) does not accept a dual-source tuple"
                )
            if mask is not None:
                raise TlaLoweringError(
                    "store(..., MaskStoreParams) does not accept predicate mask="
                )
            if params.store_dist != MaskStoreDist.DIST_NORM:
                # TODO: Once all `StoreDist` mode supported, remove this guard.
                raise NotImplementedError(
                    f"currently unsupported store_dist {params.store_dist!r}"
                )
            _require_category("store", "value", value, "mask_ssa", 1)
            elem = str(dest_desc.element_type).strip().lower()
            # AscendC StoreAlign<__ubuf__ T*, MaskReg>: T is address typing;
            # sizeof(bool)==1 admits Bool as the b8 MaskReg spill path.
            if dtype_size_bytes(elem) not in _MASK_UB_ELEM_BYTES:
                _op_error(
                    "store",
                    f"invalid argument 'dest': expected b8/b16/b32/b64 "
                    f"scalar element type for MaskSSA store, got "
                    f"{dest_desc.element_type}",
                )
            mask_desc = _mask_ssa_type_for_mlir_value(value_val)
            ub_lanes = _mask_ssa_type_for_element_type(elem).physical_lanes
            if mask_desc.physical_lanes != ub_lanes:
                _op_error(
                    "store",
                    f"MaskSSA has {mask_desc.physical_lanes} lanes, but dest "
                    f"element type {dest_desc.element_type} implies "
                    f"!tla.mask<{ub_lanes}> (256/sizeof); use matching "
                    "UB width (e.g. f32 UB with f32 / mask<64>)",
                )
            _tla_ops_gen.store(dest, value_val, loc=loc)
            return None

        if value2_val is None and _tla_type_bridge.type_is_mask_ssa(value_val.type):
            raise TlaLoweringError(
                "storing MaskSSA requires MaskStoreParams "
                "(got "
                f"{type(params).__name__ if params is not None else 'None'})"
            )

        if value2_val is None:
            _require_category("store", "value", value, "vector_ssa", 1)
        if mask is not None:
            _require_category("store", "mask", mask, "mask_ssa", 2)
        if params is None:
            params = NormalStoreParams()
        elif not isinstance(
            params, (NormalStoreParams, UnalignStoreParams, BlockStoreParams)
        ):
            raise TlaLoweringError(
                "store params must be NormalStoreParams, UnalignStoreParams, or "
                f"MaskStoreParams, got {type(params).__name__}"
            )
        is_intlv = isinstance(params, NormalStoreParams) and params.store_dist in (
            StoreDist.DIST_INTLV_B8,
            StoreDist.DIST_INTLV_B16,
            StoreDist.DIST_INTLV_B32,
        )
        if is_intlv and value2_val is None:
            raise TlaLoweringError(
                f"{params.store_dist} requires a 2-tuple of VectorSSA values "
                "(got a single value)"
            )
        if value2_val is not None and not is_intlv:
            raise TlaLoweringError(
                "dual-source store tuple is only valid with "
                "DIST_INTLV_B8/DIST_INTLV_B16/DIST_INTLV_B32"
            )
        if _validate_post_update_params(params):
            if isinstance(params, UnalignStoreParams):
                # UPDATE only; ``tla-unalign-ureg`` inserts create_ureg/Post.
                window = _require_update_origin_window(self, loc=loc)
                _require_update_unit_stride(window)
                dest = _as_value(window)
                elem_ty = _tla_tensor_type_for_mlir_value(dest).element_type
                _require_update_store_matching_dtype(value_val, elem_ty)
                ptr_type = _tla_type_bridge.tensor_ptr_type_get(dest.type)
                compressed = params.post_mode == PostMode.POST_MODE_UPDATE_NBYTES
                if compressed:
                    stride_val = _as_value(0)
                else:
                    _require_update_supported_dtype(elem_ty)
                    stride_val = _unalign_update_stride_value(
                        params, window, element_type=elem_ty
                    )
                ctx = loc.context if loc is not None else mlir_ir.Context.current
                ureg_type = mlir_ir.Type.parse("!tla.ureg", context=ctx)
                next_ptr, _ureg_out = _tla_ops_gen.store_unalign_update(
                    ptr_type,
                    ureg_type,
                    dest,
                    value_val,
                    stride_val,
                    compressed=True if compressed else None,
                    loc=loc,
                )
                if compressed:
                    return window
                return _make_update_tile_from_ptr(next_ptr, window, loc=loc)
            is_block = isinstance(params, BlockStoreParams)
            if not is_block and not isinstance(params, NormalStoreParams):
                raise TlaLoweringError(
                    "post_mode=update on tile.store only accepts NormalStoreParams, "
                    f"UnalignStoreParams, or BlockStoreParams, got {type(params).__name__}"
                )
            from ..base_dsl.typing import as_numeric
            from ..core_api import _const_int_value

            mask_val = _as_value(mask) if mask is not None else None
            if mask_val is not None:
                _require_category("store", "mask", mask, "mask_ssa", 2)
                _require_mask_matches_vector("store", mask_val, value_val)
            window = _require_update_origin_window(self, loc=loc)
            if not is_block:
                _require_update_unit_stride(window)
            dest = _as_value(window)
            dest_desc = _tla_tensor_type_for_mlir_value(dest)
            src_elem = normalize_element_type_token(
                _vector_ssa_type_for_mlir_value(value_val).element_type
            )
            dst_elem = normalize_element_type_token(dest_desc.element_type)
            if src_elem != dst_elem:
                _op_error(
                    "store",
                    "source and destination element types must match "
                    f"(got source {src_elem}, dest {dst_elem})",
                )
            if not is_block:
                _require_update_supported_dtype(dest_desc.element_type)
            ptr_type = _tla_type_bridge.tensor_ptr_type_get(dest.type)
            store_kwargs: dict[str, Any] = {"mask": mask_val, "loc": loc}
            if is_block:
                store_kwargs["block_stride"] = _as_value(params.block_stride)
                stride_val = _as_value(params.repeat_stride)
            else:
                static_stride = _const_int_value(params.post_update_stride)
                if static_stride == 0:
                    stride_val = _as_value(as_numeric(_tile_origin_extent(window)))
                else:
                    stride_val = _as_value(params.post_update_stride)
            next_ptr = _tla_ops_gen.store_update(
                ptr_type,
                dest,
                value_val,
                stride_val,
                **store_kwargs,
            )
            return _make_update_tile_from_ptr(next_ptr, window, loc=loc)
        mask_val = _as_value(mask) if mask is not None else None
        if mask_val is not None:
            _require_mask_matches_vector("store", mask_val, value_val)
        if (
            isinstance(params, NormalStoreParams)
            and params.store_dist == StoreDist.DIST_PACK4_B32
        ):
            src_elem = normalize_element_type_token(
                _vector_ssa_type_for_mlir_value(value_val).element_type
            )
            dst_elem = normalize_element_type_token(dest_desc.element_type)
            source_bytes = dtype_size_bytes(src_elem)
            if not (
                (source_bytes in (1, 2, 4) and dtype_size_bytes(dst_elem) == 1)
                or (source_bytes == 4 and src_elem == dst_elem)
            ):
                raise TlaLoweringError(
                    "DIST_PACK4_B32 requires a 1/2/4-byte source and a "
                    "1-byte destination, or matching 4-byte element types"
                )
        if isinstance(params, NormalStoreParams) and params.store_dist in (
            StoreDist.DIST_PACK_B16,
            StoreDist.DIST_PACK_B32,
            StoreDist.DIST_PACK_B64,
        ):
            src_elem = normalize_element_type_token(
                _vector_ssa_type_for_mlir_value(value_val).element_type
            )
            dst_elem = normalize_element_type_token(dest_desc.element_type)
            sizes = {
                StoreDist.DIST_PACK_B16: (1, 2),
                StoreDist.DIST_PACK_B32: (2, 4),
                StoreDist.DIST_PACK_B64: (8,),
            }[params.store_dist]
            if src_elem != dst_elem or dtype_size_bytes(src_elem) not in sizes:
                widths = "/".join(map(str, sizes))
                raise TlaLoweringError(
                    f"DIST_{params.store_dist.upper()} requires matching "
                    f"{widths}-byte source and destination element types "
                    f"(got source {src_elem}, dest {dst_elem})"
                )
        if is_intlv:
            if mask_val is not None:
                raise TlaLoweringError(
                    f"{params.store_dist} does not accept a predicate mask"
                )
            src0 = _vector_ssa_type_for_mlir_value(value_val)
            src1 = _vector_ssa_type_for_mlir_value(value2_val)
            src_elem = normalize_element_type_token(src0.element_type)
            dst_elem = normalize_element_type_token(dest_desc.element_type)
            if (
                src0.element_type != src1.element_type
                or src0.valid_lanes != src1.valid_lanes
            ):
                raise TlaLoweringError(
                    f"{params.store_dist} requires two identical VectorSSA sources"
                )
            if src_elem != dst_elem:
                raise TlaLoweringError(
                    f"{params.store_dist} requires matching source/dest element "
                    f"types (got source {src_elem}, dest {dst_elem})"
                )
            width = {
                StoreDist.DIST_INTLV_B8: 1,
                StoreDist.DIST_INTLV_B16: 2,
                StoreDist.DIST_INTLV_B32: 4,
            }[params.store_dist]
            if dtype_size_bytes(src_elem) != width:
                raise TlaLoweringError(
                    f"DIST_{params.store_dist.upper()} requires a {width}-byte "
                    f"element type (got {src_elem})"
                )
            expected_lanes = 256 // width
            if src0.valid_lanes != expected_lanes:
                raise TlaLoweringError(
                    f"{params.store_dist} requires two full {expected_lanes}-lane "
                    f"vectors (got {src0.valid_lanes})"
                )
        if isinstance(params, NormalStoreParams) and params.store_dist in (
            StoreDist.DIST_FIRST_ELEMENT_B8,
            StoreDist.DIST_FIRST_ELEMENT_B16,
            StoreDist.DIST_FIRST_ELEMENT_B32,
        ):
            if mask_val is not None:
                raise TlaLoweringError(
                    f"{params.store_dist} does not accept a predicate mask; it "
                    "writes only lane 0 and would silently ignore the mask, so "
                    "drop mask= or use a normal store_dist"
                )
            # AscendC DIST_FIRST_ELEMENT_* (HIVMAVE ONEPT_B*): mask-ignoring
            # element store that writes only lane 0 of the source register to
            # the dst base (asc_storealign_1st). The _b8/_b16/_b32 suffix fixes
            # the stored element width in bytes; a mismatch would silently
            # truncate/widen the stored bytes, so require it on both sides.
            suffix = str(params.store_dist).rsplit("_", 1)[1]
            suffix_bytes = int(suffix[1:]) // 8
            mode = "DIST_FIRST_ELEMENT_" + suffix[1:].upper()
            src_elem = (
                str(_vector_ssa_type_for_mlir_value(value_val).element_type)
                .strip()
                .lower()
            )
            dst_elem = str(dest_desc.element_type).strip().lower()
            if (
                dtype_size_bytes(src_elem) != suffix_bytes
                or dtype_size_bytes(dst_elem) != suffix_bytes
            ):
                raise TlaLoweringError(
                    f"{mode} requires {suffix_bytes}-byte element types on both "
                    f"the stored value (got {src_elem}, "
                    f"{dtype_size_bytes(src_elem)} bytes) and the dest tile "
                    f"(got {dst_elem}, {dtype_size_bytes(dst_elem)} bytes)"
                )
        store_kwargs: dict[str, Any] = {"loc": loc}
        if (
            isinstance(params, NormalStoreParams)
            and params.store_dist != StoreDist.DIST_NORM
        ):
            ctx = loc.context if loc is not None else mlir_ir.Context.current
            store_kwargs["store_dist"] = mlir_ir.Attribute.parse(
                f"#tla.store_dist<{params.store_dist}>",
                context=ctx,
            )
        if isinstance(params, UnalignStoreParams):
            store_kwargs["unaligned_ub_access"] = True
        if isinstance(params, BlockStoreParams):
            # AscendC DATA_BLOCK_COPY accepts bool via b8 (vsstb int8 stub).
            block_stride_value = _as_value(params.block_stride)
        else:
            block_stride_value = None
        # Keep Bool storage canonical across byte-distribution stores.
        if not is_intlv:
            from ..types import is_bool_element_type

            src_elem = normalize_element_type_token(
                _vector_ssa_type_for_mlir_value(value_val).element_type
            )
            dst_elem = normalize_element_type_token(dest_desc.element_type)
            if src_elem != dst_elem and (
                is_bool_element_type(src_elem) or is_bool_element_type(dst_elem)
            ):
                _op_error(
                    "store",
                    "source and destination element types must match "
                    f"(got source {src_elem}, dest {dst_elem})",
                )
        _tla_ops_gen.store(
            dest,
            value_val,
            source2=value2_val,
            mask=mask_val,
            block_stride=block_stride_value,
            **store_kwargs,
        )
        return None

    @dsl_user_op
    def fill(
        self,
        value: int | float,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> None:
        """Directory: Data Movement
        Description:
            Fill this L1 tile with a numeric constant.

            Parameters:
            - `value` (`int | float`): Trace-time scalar, encoded in the
              destination type. Integer destinations require an in-range
              `int`; f16/bf16/f32 destinations accept `int` or `float` and
              reject finite overflow. FP4/FP8 require finite values exactly
              representable in the destination format; out-of-range and inexact
              constants are rejected without rounding. E8M0 requires an exact
              positive power of two in ``[2**-127, 2**127]``; zero, negative
              values and non-powers of two are rejected.

            Constraints:
            - Must be called inside a `@tla.kernel`-decorated kernel function and
              nested inside `tla.cube()`.
            - Destination must be an L1 tensor. Operand types use ``zN`` or
              ``nZ``; E8M0 scales use ``zZMxScale`` or ``nNMxScale``.
            - Fill covers ``[coord, coord + origin_shape)``. Operand C0-axis
              boundaries are aligned up to 32-byte units. MX scales align
              the M/N axis to 16 and the scale-group axis to 2; each 16-by-2
              fractal occupies 32 bytes. A partial leading unit is left
              untouched; the final unit may include physical Padding beyond
              the logical extent, which must fit the allocation. Empty
              regions perform no writes.
            - Supported element types: ``i16/u16/f16/bf16/i32/u32/f32``,
              packed fp4/fp8 operands
              (``f4e2m1``/``f4e1m2``/``f8e4m3fn``/``f8e5m2``), and ``f8e8m0``.
              8-bit integer destinations are not supported.

            Example:
            ```python
            # Zero the K-pad tail of an L1 A tile: tile first, then fill.
            pad = tla.get_tile(t_l1a, tla.make_coord(0, k_valid),
                               tla.make_shape(m, k_l0 - k_valid))
            pad.fill(0)
            # Numeric one is encoded by the destination dtype.
            t_l1a.fill(1.0)
            t_l1sa.fill(1.0)  # E8M0 scale one, encoded as 0x7f.
            ```
        """
        from ..core_api import _tensor_fill_impl

        loc = _normalize_user_loc(loc)
        _tensor_fill_impl(self, value, loc=loc)

    def _check_can_scalar_load_store(self) -> None:
        """Phase-1 GM/UB scalar access preconditions."""
        if self.addrspace not in ("gm", "ub"):
            raise ValueError(f"{self!r} doesn't support scalar_load/store")
        if self.layout_tag not in ("RowMajor", "ColumnMajor", "VectorLayout"):
            raise ValueError(
                f"{self!r} doesn't support scalar_load/store (layout={self.layout_tag!r})"
            )

    def _check_can_dereference(self) -> None:
        sub_byte_types = (Bool,)
        if self.element_type.width % 8 != 0 and self.element_type not in sub_byte_types:
            raise ValueError(
                f"Sub-byte scalar dereference not supported for type {self.element_type.__name__}"
            )

    def _cvt_to_dest(
        self,
        data: Numeric,
        dest_element_type: mlir_ir.Type,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> mlir_ir.Value:
        """Require exact element type match, then ``ir_value`` / host constant."""
        from ..core_api import _op_error, _scalar_constant_for_element_type

        if not isinstance(data, Numeric):
            raise TypeError(f"expected Numeric, got {type(data).__name__}")

        dest_cls = Numeric.from_mlir_type(dest_element_type)
        src_cls = type(data)

        # No silent upcast/promote: store value dtype must match the tensor element
        # type. Callers should convert explicitly with ``.to(...)`` / ``cast``.
        if src_cls is not dest_cls:
            _op_error(
                "scalar_store",
                f"type mismatch, store {src_cls.dtype} "
                f"to Tensor with element type {dest_cls.dtype}; "
                f"cast explicitly with .to({dest_cls.__name__}) before store",
            )

        # Host literals: emit dest-typed constant (range / fraction checks).
        if isinstance(data.value, (bool, int, float)):
            return _scalar_constant_for_element_type(
                "scalar_store", data.value, dest_element_type, loc=loc
            )
        return data.ir_value(loc=loc)

    @dsl_user_op
    def __getitem__(
        self,
        crd: Any,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        """Directory: Data Movement
        Description:
            Load one tensor element as a scalar (`tensor[i]` / `tensor[i, j]`).

            Parameters:
            - `crd` (`int | Int32 | tuple`): Element coordinate. One index per
              logical dimension of the tensor. Required.

            Returns:

            - `Numeric`: the element, typed by the tensor's element type
              (`Bool` for an i1 tensor).

            Constraints:
            - Must be called inside a `@tla.kernel`-decorated kernel function.
            - The tensor must live in `gm` or `ub`; a `ub` tensor must be
              accessed inside `tla.vector()`.
            - Layout must be `RowMajor` or `ColumnMajor`.
            - Logical rank must be 1 or 2, and the number of indices must match
              it exactly.
            - Sub-byte element types other than `Bool` are not supported.
            - `None` / `_` coordinates are rejected; use `tile_view` to take a
              sub-tensor instead.

            Example:
            ```python
            with tla.vec.func(mode="simt", thread_block_dim=256):
                tid, _, _ = tla.arch.thread_idx()
                value = gm_in[tid]
            ```
        """
        from ..core_api import (
            _as_value,
            _op_error,
            _require_category,
            _require_frontend_state,
            _tla_tensor_descriptor_from_type_or_value,
        )
        from ..execution_lowering import TlaLoweringError

        loc = _normalize_user_loc(loc)
        if crd is None or (type(crd) is tuple and any(part is None for part in crd)):
            raise TlaLoweringError(
                "tensor indexing does not support None/underscore coordinates; "
                "use scalar indices only"
            )

        _require_category("scalar_load", "source", self, "tensor", 0)
        _require_frontend_state("scalar_load")
        if type(crd) is tuple and not crd:
            _op_error("scalar_load", "expected at least one index")

        source_value = _as_value(self)
        parent = _tla_tensor_descriptor_from_type_or_value(source_value)
        addrspace = parent.addrspace.lower()
        if isinstance(self, _Tensor):
            self._check_can_scalar_load_store()
            self._check_can_dereference()
        else:
            if addrspace not in ("gm", "ub"):
                raise ValueError("tensor doesn't support scalar_load")
            elem_numeric = Numeric.from_mlir_type(parent.element_mlir_type())
            sub_byte_types = (Bool,)
            if elem_numeric.width % 8 != 0 and elem_numeric not in sub_byte_types:
                raise ValueError(
                    "Sub-byte scalar dereference not supported for type "
                    f"{elem_numeric.__name__}"
                )
        if parent.layout_tag not in ("RowMajor", "ColumnMajor", "VectorLayout"):
            raise TlaLoweringError(
                "tla.scalar_load currently supports "
                "RowMajor/ColumnMajor/VectorLayout only"
            )

        index_values = _Tensor._prepare_scalar_indices(
            crd, parent, addrspace=addrspace, action="load"
        )

        elem_type = parent.element_mlir_type()
        # Per-thread element access inside a SIMT region is its own op: it is
        # lowered against the outlined vector function's memref parameter rather
        # than through the general descriptor view.
        emit_load = (
            _tla_ops_gen.simt_load
            if _runtime._in_simt_vec_func()
            else _tla_ops_gen.scalar_load
        )
        result = emit_load(
            elem_type,
            source_value,
            index_values,
            loc=loc,
        )
        if str(result.type) != str(elem_type):
            raise TlaLoweringError(
                f"tla scalar element load result type mismatch: expected {elem_type}, "
                f"got {result.type}"
            )
        # Bool / i1 loads: Bool Numeric (``if tensor[i]`` via coerce).
        if mlir_ir.IntegerType.isinstance(elem_type):
            int_ty = mlir_ir.IntegerType(elem_type)
            if int_ty.width == 1:
                return Bool(result)
        dtype = Numeric.from_mlir_type(elem_type)
        if _runtime._in_simt_vec_func() and dtype.is_integer and not dtype.signed:
            result = mlir_ir.Operation.create(
                "builtin.unrealized_conversion_cast",
                operands=[result],
                results=[mlir_ir.IntegerType.get_signless(dtype.width)],
                loc=loc,
            ).results[0]
        return dtype(result)

    @dsl_user_op
    def __setitem__(
        self,
        crd: Any,
        data: Any,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> None:
        """Directory: Data Movement
        Description:
            Store a scalar into one tensor element (`tensor[i] = value`).

            Parameters:
            - `crd` (`int | Int32 | tuple`): Element coordinate. One index per
              logical dimension of the tensor. Required.
            - `data` (`Numeric | int | float | mlir.ir.Value`): Value to store.
              A Python literal is converted to the tensor's element type.
              Required.

            Constraints:
            - Must be called inside a `@tla.kernel`-decorated kernel function.
            - The tensor must live in `gm` or `ub`; a `ub` tensor must be
              accessed inside `tla.vector()`.
            - Layout must be `RowMajor` or `ColumnMajor`.
            - Logical rank must be 1 or 2, and the number of indices must match
              it exactly.
            - Sub-byte element types other than `Bool` are not supported.
            - A `Numeric` value's type must match the tensor's element type
              exactly; there is no implicit promotion. Convert explicitly with
              `.to(...)` first.

            Example:
            ```python
            with tla.vec.func(mode="simt", thread_block_dim=256):
                tid, _, _ = tla.arch.thread_idx()
                gm_out[tid] = gm_a[tid] + gm_b[tid]
            ```
        """
        from ..base_dsl.typing import as_numeric
        from ..core_api import (
            _as_value,
            _op_error,
            _require_category,
            _require_frontend_state,
            _resolve_bound_value,
            _scalar_constant_for_element_type,
            _tla_tensor_descriptor_from_type_or_value,
            _type_name,
        )
        from ..execution_lowering import TlaLoweringError

        loc = _normalize_user_loc(loc)
        if crd is None or (type(crd) is tuple and any(part is None for part in crd)):
            raise TlaLoweringError(
                "tensor indexing does not support None/underscore coordinates; "
                "use scalar indices only"
            )

        _require_category("scalar_store", "dest", self, "tensor", 0)
        _require_frontend_state("scalar_store")
        if type(crd) is tuple and not crd:
            _op_error("scalar_store", "expected at least one index")

        dest_value = _as_value(self)
        parent = _tla_tensor_descriptor_from_type_or_value(dest_value)
        addrspace = parent.addrspace.lower()
        if isinstance(self, _Tensor):
            self._check_can_scalar_load_store()
            self._check_can_dereference()
        else:
            if addrspace not in ("gm", "ub"):
                raise ValueError("tensor doesn't support scalar_store")
            elem_numeric = Numeric.from_mlir_type(parent.element_mlir_type())
            sub_byte_types = (Bool,)
            if elem_numeric.width % 8 != 0 and elem_numeric not in sub_byte_types:
                raise ValueError(
                    "Sub-byte scalar dereference not supported for type "
                    f"{elem_numeric.__name__}"
                )
        if parent.layout_tag not in ("RowMajor", "ColumnMajor", "VectorLayout"):
            raise TlaLoweringError(
                "tla.scalar_store currently supports "
                "RowMajor/ColumnMajor/VectorLayout only"
            )

        index_values = _Tensor._prepare_scalar_indices(
            crd, parent, addrspace=addrspace, action="store"
        )

        elem_type = parent.element_mlir_type()
        # Canonicalize to Numeric, then _cvt_to_dest → ir_value.
        # Bare Python numbers stay dest-typed constants (range / fraction checks).
        resolved = _resolve_bound_value(data)
        if isinstance(resolved, (int, float)) and not isinstance(resolved, bool):
            store_value = _scalar_constant_for_element_type(
                "scalar_store", resolved, elem_type, loc=loc
            )
        else:
            try:
                if isinstance(resolved, Numeric):
                    num = resolved
                elif isinstance(resolved, mlir_ir.Value):
                    num = as_numeric(resolved)
                elif isinstance(data, Numeric):
                    num = data
                elif isinstance(data, mlir_ir.Value):
                    num = as_numeric(data)
                else:
                    raise TypeError(type(data).__name__)
            except (TypeError, ValueError, KeyError):
                _op_error(
                    "scalar_store",
                    f"invalid argument 'value' (position 1): expected Numeric or "
                    f"scalar literal, got {_type_name(data)}",
                )
            store_value = _Tensor._cvt_to_dest(self, num, elem_type, loc=loc)
        if (
            _runtime._in_simt_vec_func()
            and mlir_ir.IntegerType.isinstance(elem_type)
            and mlir_ir.IntegerType(elem_type).is_unsigned
            and store_value.type
            == mlir_ir.IntegerType.get_signless(mlir_ir.IntegerType(elem_type).width)
        ):
            store_value = mlir_ir.Operation.create(
                "builtin.unrealized_conversion_cast",
                operands=[store_value],
                results=[elem_type],
                loc=loc,
            ).results[0]
        if str(store_value.type) != str(elem_type):
            raise TlaLoweringError(
                f"tla.scalar_store value type mismatch: expected {elem_type}, got {store_value.type}"
            )
        emit_store = (
            _tla_ops_gen.simt_store
            if _runtime._in_simt_vec_func()
            else _tla_ops_gen.scalar_store
        )
        emit_store(dest_value, index_values, store_value, loc=loc)


def _normalize_user_loc(loc: mlir_ir.Location | None) -> mlir_ir.Location | None:
    if loc is None and _runtime._current_frontend_state() is not None:
        from ..core_api import _capture_user_loc

        return _capture_user_loc()
    if loc is not None and not isinstance(loc, mlir_ir.Location):
        raise TypeError(
            f"loc must be mlir.ir.Location or None, got {type(loc).__name__}"
        )
    return loc


def _tensor_metadata_field(value: mlir_ir.Value, field: str) -> Any:
    from ..core_api import _tensor_metadata_field as _lookup

    return _lookup(value, field)


def _scale_coord_leaf(coord: Any, shape: Any) -> Any:
    # Tile-granularity coord 0 must stay a static origin after scaling by a
    # (possibly dynamic) extent. ``0 * vl`` would otherwise become a dynamic
    # mul and spell ``!tla.coord<?>``, breaking UPDATE's origin-window check.
    from ..core_api import _const_int_value

    coord_const = _const_int_value(coord)
    if coord_const == 0:
        return 0
    shape_const = _const_int_value(shape)
    if coord_const is not None and shape_const is not None:
        return int(coord_const) * int(shape_const)

    if isinstance(coord, int):
        return coord * shape
    from ..base_dsl.typing import as_numeric

    resolved = _runtime._resolve_frontend_bound_value(coord)
    if (
        resolved is not coord
        or _runtime._resolve_frontend_bound_category(coord) == "index"
    ):
        return as_numeric(coord) * shape
    return coord * shape


def scale_tile_coord_by_shape(coord_tree: Any, shape_tree: Any) -> Any:
    """Convert tile coordinates into element offsets using the tile shape."""

    if isinstance(coord_tree, tuple) and isinstance(shape_tree, tuple):
        if len(coord_tree) != len(shape_tree):
            raise ValueError(
                "tile-view coord/shape trees must have matching tuple profiles"
            )
        return tuple(
            scale_tile_coord_by_shape(coord_part, shape_part)
            for coord_part, shape_part in zip(coord_tree, shape_tree)
        )
    if isinstance(coord_tree, tuple) or isinstance(shape_tree, tuple):
        raise ValueError(
            "tile-view coord/shape trees must have matching tuple profiles"
        )
    return _scale_coord_leaf(coord_tree, shape_tree)


def normalize_tile_view_coord(
    *,
    shape_components: tuple[Any, ...],
    coord_components: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Convert ``tla.tile_view`` tile coordinates into element offsets."""

    if len(coord_components) != len(shape_components):
        raise ValueError("tile-view coord/shape ranks must match")
    return tuple(
        scale_tile_coord_by_shape(coord_part, shape_part)
        for coord_part, shape_part in zip(coord_components, shape_components)
    )


__all__ = [
    "_Tensor",
    "normalize_tile_view_coord",
    "scale_tile_coord_by_shape",
]
