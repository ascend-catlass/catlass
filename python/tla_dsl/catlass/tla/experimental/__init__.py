"""Experimental kernel APIs; signatures and behavior may change without compatibility guarantees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from catlass._mlir import ir as mlir_ir

from ... import runtime as _runtime
from ...base_dsl.op import dsl_user_op
from ...core_api import (
    IndexLike,
    _as_index_value,
    _as_value,
    _flatten_tla_tuple,
    _MaskPattern,
    _op_error,
    _require_category,
    _tla_ops_gen,
    _tla_tensor_type_for_mlir_value,
)
from ...params import CastParams, RoundMode
from ..typing import Tensor

__all__ = ["Img2colParams", "img2col", "mx_to"]


@dsl_user_op
def mx_to(
    src: Tensor | tuple[Tensor, Tensor],
    dst: Tensor | tuple[Tensor, Tensor],
    *,
    round_mode: RoundMode = RoundMode.CAST_ROUND,
    mask: Any | None = None,
    loc: mlir_ir.Location | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Directory: Experimental APIs
    Description:
        Convert tensor tiles to, from, or between microscaling formats.

        The source and destination are UB tensor tiles. Use an ordinary `Tensor`
        for dense `f16`, `bf16` or `f32` values and a `(payload, scale)` tuple
        for MX values. The operation writes the conversion into the destination
        and returns that same destination object.

        Call this in `tla.vector()`. The bitcode path loops over the complete tile.
        One- and two-dimensional tensors support strided rows or columns. Standard
        two-dimensional blocked layouts, including `zN`, `nZ`, `zZMxScale`, and
        `nNMxScale`, are supported by a general bitcode path. MX-to-MX conversion
        can change the scale grouping axis. MX tile coordinates
        must start on a 32-element group boundary. Contiguous packed FP4 tiles
        must start on a byte boundary.
        In conversions between dense and MX tensors, the dense tensor may have
        different strides, including a different contiguous axis. Scale tensors
        can use different strides from the payload tensors.
        The general layout path stages individual groups and can be slower than
        the contiguous and supported strided fast paths.
        For runtime shapes, strides, or coordinates, the caller must ensure the
        alignment requirements, matching payload shapes, sufficient scale extents,
        and an even row stride for two-dimensional packed FP4 tiles. When a
        blocked layout's grouping axis is inferred from runtime extents, that
        axis must contain at least 32 elements.
        For two-dimensional layouts the scale grouping axis must be identifiable
        unambiguously from compile-time payload and scale extents. Source and
        destination payload storage must not overlap each other or any scale
        storage. Input and output scales may share storage if their views begin
        at the same address. Destination layouts must map distinct logical
        elements or scale groups to distinct storage locations. `VLn` masks select the first
        n values in grouping order (destination groups for MX outputs, otherwise
        source groups),
        and `ALLF` selects none. Inactive payload elements are written as zero;
        output scales are still written, and widening copies the input scales.
        Other masks and
        unsupported layouts raise an error. GM↔UB copies remain explicit.

        Parameters:
        - `src` (`Tensor | tuple[Tensor, Tensor]`): dense UB tile or MX
          `(payload, scale)` pair. Required.
        - `dst` (`Tensor | tuple[Tensor, Tensor]`): destination tile or MX
          `(payload, scale)` pair. Required and returned unchanged.
        - `round_mode` (`RoundMode`): payload rounding mode. Optional;
          defaults to `RoundMode.CAST_ROUND`.
        - `mask` (`pattern | None`): `ALL`, `ALLF`, or a `VLn` prefix pattern.

        Constraints:
        - Must be called inside a `@tla.kernel` decorated function and directly
          inside `tla.vector()`, outside `tla.vec.func()`.
        - At least one operand must be a `(payload, scale)` pair; dense-to-dense conversion is
          not an MX conversion.
        - A dense source must target a `(payload, scale)` pair; an MX source may target a dense
          tensor or another `(payload, scale)` pair.
        - MXFP4 destinations support `RoundMode.CAST_ROUND` only.

        Example:
        ```python
        with tla.vector():
            tla.experimental.mx_to(dense_ub, mx_fp8_ub)
            tla.experimental.mx_to(mx_fp8_ub, dense_out_ub)
            tla.experimental.mx_to(mx_fp8_ub, mx_fp4_ub)
        ```
    """
    from .._mx_bitcode import (
        _is_mx_tensor_pair,
        try_mx_tensor_bitcode,
        validate_mx_tensor_arguments,
    )

    src_is_mx = _is_mx_tensor_pair(src)
    dst_is_mx = _is_mx_tensor_pair(dst)
    if not isinstance(src, Tensor) and not src_is_mx:
        _op_error(
            "mx_to",
            f"src must be a Tensor or (payload, scale) tuple, got {type(src).__name__}",
        )
    if not isinstance(dst, Tensor) and not dst_is_mx:
        _op_error(
            "mx_to",
            f"dst must be a Tensor or (payload, scale) tuple, got {type(dst).__name__}",
        )
    if not src_is_mx and not dst_is_mx:
        _op_error(
            "mx_to", "at least one of src or dst must be a `(payload, scale)` pair"
        )
    if not isinstance(round_mode, RoundMode):
        _op_error(
            "mx_to", f"round_mode must be RoundMode, got {type(round_mode).__name__}"
        )
    if (
        dst_is_mx
        and dst[0].dtype in {"f4e2m1", "f4e1m2"}
        and round_mode != RoundMode.CAST_ROUND
    ):
        _op_error("mx_to", "MXFP4 destinations support only RoundMode.CAST_ROUND")
    params = CastParams(round_mode=round_mode)
    if mask is not None and not isinstance(mask, (Tensor, _MaskPattern, str)):
        _op_error(
            "mx_to",
            f"mask must be a Tensor or fixed tla.mask pattern, got {type(mask).__name__}",
        )
    _runtime._require_enclosing_region("mx_to", "vector")
    if _runtime._has_enclosing_region("vec.func"):
        _op_error(
            "mx_to",
            "must be called directly inside tla.vector(), outside tla.vec.func()",
        )
    token = mask._token if isinstance(mask, _MaskPattern) else mask
    if mask is not None and (
        not isinstance(token, str)
        or token
        not in {
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
        }
    ):
        _op_error("mx_to", "unsupported mask; use ALL, ALLF, or a VLn prefix pattern")
    validate_mx_tensor_arguments(src, dst)
    if try_mx_tensor_bitcode(src, dst, params, mask, loc=loc):
        return dst
    _op_error("mx_to", "unsupported tensor layout for whole-tile bitcode conversion")


@dataclass(frozen=True)
class Img2colParams:
    """Convolution geometry for `tla.experimental.img2col`.

    `filter_hw` is required; everything else defaults to the no-op convolution
    (unit stride, unit dilation, no padding), so a 1x1 conv needs only the
    filter shape.

    Feature-map extent, stride, dilation and padding accept runtime values.
    Compile-time values are checked against the documented ranges. For
    runtime values, callers must satisfy those ranges; violating them is
    undefined behavior. The filter shape is static so K windows can be
    validated against whole C1 slices before the load.
    """

    fmap_hw: tuple[IndexLike, IndexLike]
    filter_hw: tuple[int, int]
    stride_hw: tuple[IndexLike, IndexLike] = (1, 1)
    dilation_hw: tuple[IndexLike, IndexLike] = (1, 1)
    # left, right, top, bottom
    pad: tuple[IndexLike, IndexLike, IndexLike, IndexLike] = (0, 0, 0, 0)
    pad_value: int = 0


@dsl_user_op
def img2col(
    dst: Tensor,
    src: Tensor,
    params: Img2colParams,
    *,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Directory: Experimental APIs
    Description:
        Load an L1 feature-map tile into L0A with img2col expansion for convolution.

        Parameters:
        - `dst` (`Tensor`): L0A tile, tagged `zN`, shaped `(M, K)`. Its shape gives
          the load's extents and its coord gives the start points, so M and K of the
          implicit GEMM are tiled by aiming this at a different window of the same
          L1 tile -- at no extra L1 traffic. Required.
        - `src` (`Tensor`): L1 feature-map tile, tagged `zNUnAlign`, which is the
          NC1HWC0 nesting `((h*w, 1), (C0, c1))`. Required.
        - `params` (`Img2colParams`): Convolution geometry. Required.
          - `fmap_hw` (`tuple`): Feature-map `(H, W)` extents. Required.
          - `filter_hw` (`tuple[int, int]`): Filter `(kH, kW)`; compile-time
            integers. Required.
          - `stride_hw` (`tuple`): Stride `(sH, sW)`. Default `(1, 1)`.
          - `dilation_hw` (`tuple`): Dilation `(dH, dW)`. Default `(1, 1)`.
          - `pad` (`tuple`): Padding `(left, right, top, bottom)`. Default
            `(0, 0, 0, 0)`.
          - `pad_value` (`int`): Padding fill value. Default `0`.

        Constraints:
        - Must be called inside a `@tla.kernel`-decorated kernel function.
        - Must be called inside `tla.cube()`.
        - Element type must be `f32`, `f16` or `bf16` (2- and 4-byte elements
          only; no int8/fp8 route).
        - `src` and `dst` must share an element type.
        - Source channels come from the NC1HWC0 source shape, independently of
          the destination K window. Source C0/C1 and destination K extent and
          K coordinate must be static. `filter_hw` must be compile-time integers.
        - The destination K extent and starting coordinate must cover whole
          C1 slices (`kh * kw * C0` elements), within the source's full K axis.
          Source channels and destination K extent must fit in 1..65535;
          the destination K coordinate must fit in 0..65535.
        - `fmap_hw` entries must be in 1..65535, stride and dilation entries
          in 1..255, and each padding entry in 0..255. Compile-time constants
          outside these ranges are rejected. For runtime values, satisfying
          these ranges is a caller precondition; violating them is undefined
          behavior.
        - Filter extents are compile-time integers in 1..511.

        Example:
        ```python
        with tla.cube():
            # l1_fmap holds NC1HWC0; l1_filter holds the (K, N) weight matrix.
            tla.experimental.img2col(l0_a, l1_fmap, tla.experimental.Img2colParams(
                fmap_hw=(hi, wi), filter_hw=(3, 3), pad=(1, 1, 1, 1)))
            tla.copy(l0_b, l1_filter)
            tla.mmad(l0_c, l0_a, l0_b, init_c=True)
        ```
    """
    if not isinstance(params, Img2colParams):
        _op_error(
            "img2col", f"params must be an Img2colParams, got {type(params).__name__}"
        )
    _require_category("img2col", "dst", dst, "tensor", 0)
    _require_category("img2col", "src", src, "tensor", 1)
    dst_value = _as_value(dst)
    src_value = _as_value(src)
    dst_desc = _tla_tensor_type_for_mlir_value(dst_value)
    src_desc = _tla_tensor_type_for_mlir_value(src_value)

    if (src_desc.addrspace, src_desc.layout_tag) != ("l1", "zNUnAlign"):
        _op_error(
            "img2col",
            "src must be an l1 feature-map tile tagged zNUnAlign (the NC1HWC0 nesting), "
            f"got {src_desc.addrspace}/{src_desc.layout_tag}",
        )
    if (dst_desc.addrspace, dst_desc.layout_tag) != ("l0a", "zN"):
        _op_error(
            "img2col",
            f"dst must be an l0a tile tagged zN, got {dst_desc.addrspace}/{dst_desc.layout_tag}",
        )
    if src_desc.element_type != dst_desc.element_type:
        _op_error(
            "img2col",
            f"src and dst element types must match, got {src_desc.element_type} and {dst_desc.element_type}",
        )
    # The A5 img2col path is instantiated for 2- and 4-byte elements only.
    if src_desc.element_type not in ("f32", "f16", "bf16"):
        _op_error(
            "img2col",
            f"element type must be one of f32/f16/bf16, got {src_desc.element_type}",
        )

    filter_hw = params.filter_hw
    if (
        not isinstance(filter_hw, tuple)
        or len(filter_hw) != 2
        or any(type(v) is not int or not 1 <= v <= 511 for v in filter_hw)
    ):
        _op_error(
            "img2col", "filter_hw must contain two compile-time integers in 1..511"
        )
    src_shape = list(_flatten_tla_tuple(src_desc.shape))
    dst_shape = list(_flatten_tla_tuple(dst_desc.shape))
    dst_coord = list(_flatten_tla_tuple(dst_desc.coord))
    c0 = 8 if src_desc.element_type == "f32" else 16
    if len(src_shape) != 4 or len(dst_shape) != 4 or len(dst_coord) != 2:
        _op_error(
            "img2col",
            "requires four-leaf tensor shapes and a two-axis destination coordinate",
        )
    if (
        src_shape[2] != c0
        or not isinstance(src_shape[3], int)
        or not 1 <= src_shape[3] <= 65535 // c0
    ):
        _op_error(
            "img2col",
            "source channels must be static whole C1 slices within 1..65535 channels",
        )
    if (
        dst_shape[2] != c0
        or not isinstance(dst_shape[3], int)
        or not 1 <= dst_shape[3] <= 65535 // c0
    ):
        _op_error(
            "img2col",
            "destination K extent must be static, C0-aligned and within 1..65535",
        )
    k_extent = c0 * dst_shape[3]
    slice_k = filter_hw[0] * filter_hw[1] * c0
    if k_extent % slice_k:
        _op_error(
            "img2col",
            "destination K extent must span whole C1 slices (filter_h * filter_w * C0)",
        )
    k_start = dst_coord[1]
    if not isinstance(k_start, int) or not 0 <= k_start <= 65535 or k_start % slice_k:
        _op_error(
            "img2col",
            "destination K coordinate must be static, within 0..65535 and on a whole C1 slice",
        )
    if k_start + k_extent > src_shape[2] * src_shape[3] * filter_hw[0] * filter_hw[1]:
        _op_error(
            "img2col", "destination K window exceeds the source channel/filter extent"
        )

    def _pair(name: str, value: Any, count: int) -> tuple[mlir_ir.Value, ...]:
        if not isinstance(value, tuple) or len(value) != count:
            _op_error("img2col", f"{name} must be a {count}-tuple, got {value!r}")
        return tuple(_as_index_value(v) for v in value)

    fmap_h, fmap_w = _pair("fmap_hw", params.fmap_hw, 2)
    filter_h, filter_w = _pair("filter_hw", params.filter_hw, 2)
    stride_h, stride_w = _pair("stride_hw", params.stride_hw, 2)
    dilation_h, dilation_w = _pair("dilation_hw", params.dilation_hw, 2)
    pad_left, pad_right, pad_top, pad_bottom = _pair("pad", params.pad, 4)

    if isinstance(params.pad_value, bool) or not isinstance(params.pad_value, int):
        _op_error(
            "img2col",
            "pad_value must be a trace-time int holding the fill value's bit pattern, "
            f"got {params.pad_value!r}",
        )
    return _tla_ops_gen.img2col(
        dst_value,
        src_value,
        fmap_h,
        fmap_w,
        filter_h,
        filter_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        pad_left,
        pad_right,
        pad_top,
        pad_bottom,
        pad_value=params.pad_value,
        loc=loc,
    )
