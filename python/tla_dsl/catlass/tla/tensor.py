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


# MaskSSA load/store UB: any 1/2/4-byte scalar (int or float).
# ``N = 256 / sizeof(UB elem)``; hardware path is still plds/psts.b8.
_MASK_UB_ELEM_BYTES = frozenset((1, 2, 4))
# AscendC ``LoadAlign<T, MaskDist>``: ``SupportBytes<T, 1, 2, 4, 8>`` (b8/b16/b32/b64).
# ``T`` is address typing only; ``plds`` always takes ``uint32_t*``.
_MASK_DIST_US_DS_ELEM_BYTES = frozenset((1, 2, 4, 8))
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


# Align PostMode UPDATE emit path today only wires HIVM
# ``vldsx1.post`` / ``vstsx1.post`` for these element types.
_UPDATE_SUPPORTED_DTYPES = frozenset({"f16", "bf16", "f32"})


def _require_update_supported_dtype(element_type: Any) -> None:
    """Reject UPDATE tiles whose element type is not yet wired in emit."""
    from ..execution_lowering import TlaLoweringError

    dtype = str(element_type).strip().lower()
    if dtype not in _UPDATE_SUPPORTED_DTYPES:
        raise TlaLoweringError(
            "post_mode=update currently supports element types "
            f"f16/bf16/f32 only, got {element_type}"
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
        UnalignLoadParams,
        UnalignStoreParams,
    )

    # Params without PostMode fields are never UPDATE.
    if not hasattr(params, "post_mode"):
        return False
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
    is_unalign = isinstance(params, (UnalignLoadParams, UnalignStoreParams))
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
        if is_unalign:
            # Dense-only: HIVM count/advance = tile_extent (origin_shape).
            # Keep the field for API symmetry with Align, but forbid an
            # explicit value — emit always uses origin_shape.
            if static_stride is None or static_stride != 0:
                raise TlaLoweringError(
                    "UnAlign post_mode=update derives post_update_stride from "
                    "tile_extent (origin_shape); do not pass post_update_stride "
                    "(leave default 0)"
                )
        elif static_stride is not None and static_stride < 0:
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
        the loop yields that tile, the same packing as any other carried tensor.
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
                `DIST_DINTLV_B32`).
              - `MaskLoadParams` → `MaskSSA`.
              - `NormalLoadParams`:
                - `load_dist` (`str`): `tla.params.LoadDist` token. Default
                  `DIST_NORM`.

                  | Value | Meaning |
                  |---|---|
                  | `DIST_NORM` | Default contiguous load |
                  | `DIST_BRC_B32` | Broadcast one b32 element to all lanes |
                  | `DIST_DINTLV_B32` | f32 deinterleave; dual results |
                  | `DIST_US_B8` / `DIST_US_B16` | 2x up-sample |
                  | `DIST_BRC_B16` | Broadcast one b16 element |
                  | `DIST_UNPACK_B16` | Zero-extend b16 to double width |
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
                - `post_update_stride` (`Any`): Must stay `0` (window advances
                  by `origin_shape` only). Default `0`.
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
            - Mask load ``DIST_NORM`` requires a 1/2/4-byte scalar UB element type.
            - Distribution modes fix the element width: `DIST_US_B8` needs
              i8/u8; `DIST_BRC_B16` / `DIST_US_B16` / `DIST_UNPACK_B16` /
              `DIST_E2B_B16` need 2-byte (f16/bf16/i16/u16); `DIST_E2B_B32`
              needs 4-byte (f32/i32/u32); `DIST_DINTLV_B32` needs f32;
              `DIST_BLK` works for any element size.
            - `DIST_UNPACK4_B8` requires a 1-byte element type and a
              32-byte-aligned source address.
            - `BlockLoadParams` requires a 2/4-byte element type
              (f32/f16/bf16/i32/u32/i16/u16); `block_stride` and
              `repeat_stride` must fit in [0, 65535].
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
              `DIST_NORM`, and element type in `{f16, bf16, f32}`. Returns
              `(vector, next_tile)`. Inside `tla.vec.func`, assign
              `next_tile` back with `tla.range` (`while` is not supported
              nested in `vec.func`). The window must already be an origin
              tile (`coord=0`); cut it with `tile_view` (or equivalent).
              UPDATE never automatically folds non-zero coordinates.
              `NormalLoadParams`: `post_update_stride=0` (default) advances
              by `tile_extent` (`origin_shape`); an explicit stride keeps
              fill / stride-advance decoupling. `UnalignLoadParams` must
              leave `post_update_stride` at `0` (dense `origin_shape` only).

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
            elem_bytes = dtype_size_bytes(elem)
            is_us_ds = params.load_dist in (
                MaskLoadDist.DIST_US,
                MaskLoadDist.DIST_DS,
            )
            # AscendC MaskDist: T ∈ b8/b16/b32/b64. DIST_NORM keeps the
            # existing 1/2/4-byte MaskSSA carrier rule.
            allowed_bytes = (
                _MASK_DIST_US_DS_ELEM_BYTES if is_us_ds else _MASK_UB_ELEM_BYTES
            )
            if elem_bytes not in allowed_bytes:
                if is_us_ds:
                    raise NotImplementedError(
                        f"MaskLoadDist {params.load_dist!r} requires a "
                        f"b8/b16/b32/b64 scalar UB (AscendC "
                        f"LoadAlign<T, MaskDist>), got "
                        f"{source_desc.element_type}"
                    )
                _op_error(
                    "load",
                    f"invalid argument 'source': expected 1/2/4-byte scalar "
                    f"element type for MaskSSA load, got "
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
                # Dual instr at VectorRegion: local vldas + vldus.post.
                from ..base_dsl.typing import as_numeric

                window = _require_update_origin_window(self, loc=loc)
                _require_update_unit_stride(window)
                source = _as_value(window)
                source_desc = _tla_tensor_type_for_mlir_value(source)
                _require_update_supported_dtype(source_desc.element_type)
                result_type = _coerce_type(
                    _vector_ssa_type_from_tensor_descriptor(source_desc)
                )
                ptr_type = _tla_type_bridge.tensor_ptr_type_get(source.type)
                extent = _tile_origin_extent(window)
                vec_val, next_ptr = _tla_ops_gen.load_unalign_update(
                    result_type,
                    ptr_type,
                    source,
                    _as_value(as_numeric(extent)),
                    loc=loc,
                )
                return VectorSSA(vec_val), _make_update_tile_from_ptr(
                    next_ptr, window, loc=loc
                )
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

        is_dintlv = (
            isinstance(params, NormalLoadParams)
            and params.load_dist == LoadDist.DIST_DINTLV_B32
        )
        is_us_b8 = (
            isinstance(params, NormalLoadParams)
            and params.load_dist == LoadDist.DIST_US_B8
        )
        is_brc_b16 = (
            isinstance(params, NormalLoadParams)
            and params.load_dist == LoadDist.DIST_BRC_B16
        )
        is_us_b16 = (
            isinstance(params, NormalLoadParams)
            and params.load_dist == LoadDist.DIST_US_B16
        )
        is_unpack_b16 = (
            isinstance(params, NormalLoadParams)
            and params.load_dist == LoadDist.DIST_UNPACK_B16
        )
        is_unpack4_b8 = (
            isinstance(params, NormalLoadParams)
            and params.load_dist == LoadDist.DIST_UNPACK4_B8
        )
        is_e2b_b16 = (
            isinstance(params, NormalLoadParams)
            and params.load_dist == LoadDist.DIST_E2B_B16
        )
        is_e2b_b32 = (
            isinstance(params, NormalLoadParams)
            and params.load_dist == LoadDist.DIST_E2B_B32
        )
        is_blk = (
            isinstance(params, NormalLoadParams)
            and params.load_dist == LoadDist.DIST_BLK
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
            # vsldb stubs cover 2/4-byte lanes (f32/f16/bf16/i32/u32/i16/u16);
            # b8 has no VectorReg shim yet.
            elem = str(source_desc.element_type).strip().lower()
            if dtype_size_bytes(elem) not in (2, 4):
                raise TlaLoweringError(
                    "BlockLoadParams requires a 2/4-byte element type "
                    "(f32/f16/bf16/i32/u32/i16/u16), got "
                    f"{source_desc.element_type}"
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

        if is_dintlv:
            # AscendNPU-IR lowers DINTLV_B32 only to vldsx2.v64f32; reject
            # i32/u32 (and other dtypes) at the frontend until IR dispatches.
            elem = str(source_desc.element_type).strip().lower()
            if elem != "f32":
                raise TlaLoweringError(
                    "DIST_DINTLV_B32 currently requires f32 element type "
                    f"(got {source_desc.element_type})"
                )
            # Dual-destination load writes two full VL registers (even/odd).
            # Source tile is typically 2*VL elements; do not derive result
            # VectorSSA lanes from the source origin_shape.
            result_desc = _full_vector_ssa_descriptor(source_desc.element_type)
        elif (
            isinstance(params, NormalLoadParams)
            and params.load_dist == LoadDist.DIST_BRC_B32
        ):
            result_desc = _full_vector_ssa_descriptor(source_desc.element_type)
        elif is_us_b8:
            # AscendC LoadDist::DIST_US_B8: 2x up-sample of b8 elements. The
            # transfer reads VL/2 b8 elements and repeats each twice to fill a
            # VL-wide b8 register, so the result is a full VL vector (i8/u8 ->
            # 256 lanes) regardless of the source origin_shape.
            elem = str(source_desc.element_type).strip().lower()
            if dtype_size_bytes(elem) != 1:
                raise TlaLoweringError(
                    "DIST_US_B8 requires a 1-byte (b8: i8/u8) element type "
                    f"(got {source_desc.element_type})"
                )
            result_desc = _full_vector_ssa_descriptor(source_desc.element_type)
        elif is_brc_b16 or is_us_b16:
            # AscendC LoadDist::DIST_BRC_B16 / DIST_US_B16: b16-only modes.
            # BRC_B16 reads one b16 element and replicates it across all lanes;
            # US_B16 reads VL/2 b16 elements and repeats each twice. Either way
            # the result is a full VL register (f16/bf16/i16/u16 -> 128 lanes)
            # regardless of the source origin_shape.
            elem = str(source_desc.element_type).strip().lower()
            if dtype_size_bytes(elem) != 2:
                mode = "DIST_BRC_B16" if is_brc_b16 else "DIST_US_B16"
                raise TlaLoweringError(
                    f"{mode} requires a 2-byte (b16: f16/bf16/i16/u16) "
                    f"element type (got {source_desc.element_type})"
                )
            result_desc = _full_vector_ssa_descriptor(source_desc.element_type)
        elif is_unpack_b16:
            # AscendC LoadDist::DIST_UNPACK_B16 (HIVMAVE UNPK_B16): b16-only
            # unpack. The transfer reads VL/2 b16 elements and inserts one
            # zero b16 element after each to fill a VL-wide b16 register, so
            # the result is a full VL vector regardless of the source
            # origin_shape.
            elem = str(source_desc.element_type).strip().lower()
            if dtype_size_bytes(elem) != 2:
                raise TlaLoweringError(
                    "DIST_UNPACK_B16 requires a 2-byte (b16: f16/bf16/i16/u16) "
                    f"element type (got {source_desc.element_type})"
                )
            result_desc = _full_vector_ssa_descriptor(source_desc.element_type)
        elif is_unpack4_b8:
            if dtype_size_bytes(source_desc.element_type) != 1:
                raise TlaLoweringError(
                    "DIST_UNPACK4_B8 requires a 1-byte element type "
                    f"(got {source_desc.element_type})"
                )
            result_desc = _full_vector_ssa_descriptor(source_desc.element_type)
        elif is_e2b_b16 or is_e2b_b32:
            # AscendC LoadDist::DIST_E2B_B16 / DIST_E2B_B32 (HIVMAVE E2B_B16 /
            # E2B_B32): element-to-DataBlock broadcast. The transfer reads
            # VL/DB = 8 source elements (DB = 32-byte DataBlock) and
            # replicates each element across all lanes of its destination
            # DataBlock, so the result is a full VL register (b16 -> 128
            # lanes / b32 -> 64 lanes) regardless of the source origin_shape.
            elem = str(source_desc.element_type).strip().lower()
            mode = "DIST_E2B_B16" if is_e2b_b16 else "DIST_E2B_B32"
            want_bytes = 2 if is_e2b_b16 else 4
            if dtype_size_bytes(elem) != want_bytes:
                raise TlaLoweringError(
                    f"{mode} requires a {want_bytes}-byte element type "
                    "(b16: f16/bf16/i16/u16; b32: f32/i32/u32; "
                    f"got {source_desc.element_type})"
                )
            result_desc = _full_vector_ssa_descriptor(source_desc.element_type)
        elif is_blk:
            # AscendC LoadDist::DIST_BLK (HIVMAVE BLK): DataBlock broadcast.
            # The transfer reads one 32-byte DataBlock worth of elements
            # (b8: 32, b16: 16, b32: 8) from the tile view base and
            # replicates it across all VL/DB (= 8) DataBlocks to fill a
            # VL-wide register. Unlike the suffixed modes, ``blk`` works for
            # any element size, so there is no dtype check here; the result
            # is a full VL vector regardless of the source origin_shape.
            result_desc = _full_vector_ssa_descriptor(source_desc.element_type)
        elif is_block_load:
            # AscendC vsldb strided block load: one instruction gathers 8
            # DataBlocks (256B) into a full VL register regardless of the
            # source origin_shape; the tile view only supplies the address.
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
        value: VectorSSA | MaskSSA,
        params: StoreParams | None = None,
        *,
        mask: MaskSSA | None = None,
        loc: mlir_ir.Location | None = None,
    ) -> _Tensor | None:
        """Directory: Data Movement
        Description:
            Store vector or mask SSA into this UB tensor tile (`tile.store`).

            Parameters:
            - `value` (`VectorSSA | MaskSSA`): `VectorSSA` or `MaskSSA` to store.
              Required.
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
                  | `DIST_PACK4_B32` | Store register byte slots 0,4,...,252 to UB byte slots 0,1,...,63, subject to the mask |
                  | `DIST_FIRST_ELEMENT_B8` | Write only lane 0 (1-byte) |
                  | `DIST_FIRST_ELEMENT_B16` | Write only lane 0 (2-byte) |
                  | `DIST_FIRST_ELEMENT_B32` | Write only lane 0 (4-byte) |

                - `post_mode` (`str`): `POST_MODE_NORMAL` (default) or
                  `POST_MODE_UPDATE`.
                - `post_update_stride` (`Any`): Element stride when
                  `post_mode=POST_MODE_UPDATE`; `0` advances by `tile_extent`
                  (`origin_shape`). Default `0`.
              - `UnalignStoreParams`:
                - `post_mode` (`str`): `POST_MODE_NORMAL` (default) or
                  `POST_MODE_UPDATE`.
                - `post_update_stride` (`Any`): Must stay `0`. Default `0`.
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
              rejected by `MaskStoreParams` and first-element modes. Default
              `None`. For `DIST_PACK4_B32`, masks use source lanes:
              4/2/1 lanes per destination byte for 1/2/4-byte sources.
              An omitted mask follows the destination's valid length;
              an explicit mask is not clipped to that length. Masked-out
              destination bytes remain unchanged.

            Constraints:
            - Must be called inside a `@tla.kernel`-decorated kernel function.
            - Must be called inside `tla.vec.func(mode="simd")`; destination tile must be UB.
            - `DIST_PACK4_B32` requires 1/2/4-byte source elements,
              1-byte destination elements and a 32-byte-aligned destination address.
              It stores at most 64 bytes, subject to the mask, without numeric
              conversion or FP4 nibble packing. For FP32→FP8 results, use
              `RegSlot.ZERO` to place the values in byte slots 0,4,...,252.
            - First-element modes require the stored value and the dest tile to
              use the suffix-matching element width (b8 → 1-byte, b16 → 2-byte,
              b32 → 4-byte); passing a predicate mask raises an error.
            - `POST_MODE_UPDATE` requires a rank-1 contiguous tile,
              `DIST_NORM`, and element type in `{f16, bf16, f32}`; returns the
              next tile. Inside `tla.vec.func`, assign it back with
              `tla.range`. Same origin window (`coord=0`) contract as UPDATE
              `load`. `NormalStoreParams`: default `post_update_stride=0`
              advances by `tile_extent` (`origin_shape`); an explicit stride is
              decoupled. `UnalignStoreParams` must leave stride at `0`
              (dense `origin_shape` only).

            Example:
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
            StoreDist,
            UnalignStoreParams,
        )
        from ..types import dtype_size_bytes

        loc = _normalize_user_loc(loc)
        _require_frontend_state("store")
        _runtime._require_enclosing_region("store", "vec.func")

        dest = _as_value(self)
        dest_desc = _tla_tensor_type_for_mlir_value(dest)
        if dest_desc.addrspace.lower() != "ub":
            _op_error(
                "store",
                "invalid argument 'dest' (position 0): expected addrspace ub, "
                f"got {dest_desc.addrspace}",
            )
        value_val = _as_value(value)

        # MaskSSA path: selected by MaskStoreParams.
        if isinstance(params, MaskStoreParams):
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
            if dtype_size_bytes(elem) not in _MASK_UB_ELEM_BYTES:
                _op_error(
                    "store",
                    f"invalid argument 'dest': expected 1/2/4-byte scalar "
                    f"element type for MaskSSA store, got "
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

        if _tla_type_bridge.type_is_mask_ssa(value_val.type):
            raise TlaLoweringError(
                "storing MaskSSA requires MaskStoreParams "
                "(got "
                f"{type(params).__name__ if params is not None else 'None'})"
            )

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
        if _validate_post_update_params(params):
            if isinstance(params, UnalignStoreParams):
                # Dual instr at VectorRegion: empty ureg + vstus.post + vstas.
                from ..base_dsl.typing import as_numeric

                if mask is not None:
                    raise TlaLoweringError(
                        "UnalignStoreParams(post_mode=update) does not accept mask="
                    )
                window = _require_update_origin_window(self, loc=loc)
                _require_update_unit_stride(window)
                dest = _as_value(window)
                _require_update_supported_dtype(
                    _tla_tensor_type_for_mlir_value(dest).element_type
                )
                ptr_type = _tla_type_bridge.tensor_ptr_type_get(dest.type)
                extent = _tile_origin_extent(window)
                next_ptr = _tla_ops_gen.store_unalign_update(
                    ptr_type,
                    dest,
                    value_val,
                    _as_value(as_numeric(extent)),
                    loc=loc,
                )
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
            if not is_block:
                _require_update_supported_dtype(
                    _tla_tensor_type_for_mlir_value(dest).element_type
                )
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
            src_elem = _vector_ssa_type_for_mlir_value(value_val).element_type
            if (
                dtype_size_bytes(src_elem) not in (1, 2, 4)
                or dtype_size_bytes(dest_desc.element_type) != 1
            ):
                raise TlaLoweringError(
                    "DIST_PACK4_B32 requires a 1/2/4-byte source and a "
                    "1-byte destination element type"
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
            block_stride_value = _as_value(params.block_stride)
        else:
            block_stride_value = None
        _tla_ops_gen.store(
            dest,
            value_val,
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
            Fill this L1 zN/nZ tile with a constant (currently only zero).

            Parameters:
            - `value` (`int | float`): Trace-time scalar; only ``0`` is
              supported (the device writes the zero bit pattern).

            Constraints:
            - Must be called inside a `@tla.kernel`-decorated kernel function and
              nested inside `tla.cube()`.
            - Destination must be an L1 tensor tagged ``zN`` or ``nZ``.
            - Fill covers ``[coord, coord + origin_shape)``; the C0 axis start
              is aligned up to a 32-byte boundary, so a partial leading C0 unit
              is left untouched.
            - Supported element types: the packed fp4/fp8 operand formats
              (``f4e2m1``/``f4e1m2``/``f8e4m3fn``/``f8e5m2``), zero only.
              Wider types, 8-bit integers and e8m0 scale tiles are not
              supported.

            Example:
            ```python
            # Zero the K-pad tail of an L1 A tile: tile first, then fill.
            pad = tla.get_tile(t_l1a, tla.make_coord(0, k_valid),
                               tla.make_shape(m, k_l0 - k_valid))
            pad.fill(0)
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
        return Numeric.from_mlir_type(elem_type)(result)

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
