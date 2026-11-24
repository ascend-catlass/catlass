"""Tests for tla.experimental.img2col, the img2col L1->L0A/L0B load.

These cover the frontend only: that the op reaches TLA IR carrying its
convolution geometry, and that the operand descriptors it refuses are refused
with a diagnostic rather than passed down to the compiler. The lit tests under
tests/lit/copy/ cover the lowering, and the end-to-end examples cover numerics.

Each kernel spells its body out in full rather than sharing a helper: a
@tla.kernel body cannot call a plain Python function that emits ops -- the tracer
does not descend into it, the kernel falls back to execution mode, and every test
here fails with "tla.cube is only available in lowered TLA MLIR" instead of the
diagnostic it was checking for.
"""

from __future__ import annotations

from typing import Any

import pytest

import catlass.tla as tla
from catlass.tla.experimental import Img2colParams, img2col
from catlass.tla.runtime import make_fake_tensor

C0 = 16
CIN1 = 2
HI = WI = 8
KH = KW = 3

FMAP_ROWS = HI * WI
CIN = CIN1 * C0
M = HI * WI  # pad 1, stride 1, 3x3 -> ho*wo == hi*wi
K = CIN1 * KH * KW * C0

_FMAP_SHAPE = ((FMAP_ROWS, 1), (C0, CIN1))
_FMAP_STRIDE = ((C0, FMAP_ROWS * C0), (1, FMAP_ROWS * C0))


def _gm_tensor(
    dtype: type[tla.Numeric] = tla.Float16, *, cols: int = K
) -> tla.Tensor:
    """A kernel argument: flat and RowMajor, which is all the argument ABI takes."""
    return make_fake_tensor(
        dtype,
        (M, cols),
        (cols, 1),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(M, cols),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def img2col_3x3_pad1(fmap: tla.Tensor) -> None:
    l1a_ptr = tla.allocate(CIN1 * FMAP_ROWS * C0, tla.Float16, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l0a, 512)
    with tla.cube():
        gm_a = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape(*_FMAP_SHAPE),
                tla.make_stride(*_FMAP_STRIDE),
                origin_shape=tla.make_shape(FMAP_ROWS, CIN),
                layoutTag=tla.arch.zNUnAlign,
            ),
        )
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zNUnAlign)
        tla.copy(l1_a, gm_a)
        ref = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape(M, K),
                tla.make_stride(K, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        l0_a = tla.make_tensor_like(l0a_ptr, ref, tla.arch.zN)
        img2col(
            l0_a,
            l1_a,
            Img2colParams(
                fmap_hw=(HI, WI),
                filter_hw=(KH, KW),
                pad=(1, 1, 1, 1),
            ),
        )


@tla.kernel
def img2col_stride2_dilation2(fmap: tla.Tensor) -> None:
    l1a_ptr = tla.allocate(CIN1 * FMAP_ROWS * C0, tla.Float16, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l0a, 512)
    with tla.cube():
        gm_a = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape(*_FMAP_SHAPE),
                tla.make_stride(*_FMAP_STRIDE),
                origin_shape=tla.make_shape(FMAP_ROWS, CIN),
                layoutTag=tla.arch.zNUnAlign,
            ),
        )
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zNUnAlign)
        tla.copy(l1_a, gm_a)
        ref = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape(M, K),
                tla.make_stride(K, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        l0_a = tla.make_tensor_like(l0a_ptr, ref, tla.arch.zN)
        tla.experimental.img2col(
            l0_a,
            l1_a,
            tla.experimental.Img2colParams(
                fmap_hw=(HI, WI),
                filter_hw=(KH, KW),
                stride_hw=(2, 2),
                dilation_hw=(2, 2),
                # Deliberately all different: padList order is the easiest thing
                # to get wrong, and a symmetric case cannot catch a swap.
                pad=(4, 5, 6, 7),
                repeat_load_stride=0,
                repeat_load_time=1,
                repeat_load_axis=tla.experimental.height,
            ),
        )


@tla.kernel
def img2col_src_not_zNUnAlign(fmap: tla.Tensor) -> None:
    l1a_ptr = tla.allocate(CIN1 * FMAP_ROWS * C0, tla.Float16, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l0a, 512)
    with tla.cube():
        ref = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape(M, K),
                tla.make_stride(K, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        l1_a = tla.make_tensor_like(l1a_ptr, ref, tla.arch.zN)
        l0_a = tla.make_tensor_like(l0a_ptr, ref, tla.arch.zN)
        tla.experimental.img2col(
            l0_a,
            l1_a,
            Img2colParams(
                fmap_hw=(HI, WI),
                filter_hw=(KH, KW),
                repeat_load_stride=0,
                repeat_load_time=1,
                repeat_load_axis=tla.experimental.height,
            ),
        )


@tla.kernel
def img2col_dst_is_l0b(fmap: tla.Tensor, layout: tla.Constexpr) -> None:
    l1a_ptr = tla.allocate(CIN1 * FMAP_ROWS * C0, tla.Float16, tla.AddressSpace.l1, 512)
    l0b_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l0b, 512)
    with tla.cube():
        gm_a = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape(*_FMAP_SHAPE),
                tla.make_stride(*_FMAP_STRIDE),
                origin_shape=tla.make_shape(FMAP_ROWS, CIN),
                layoutTag=tla.arch.zNUnAlign,
            ),
        )
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zNUnAlign)
        ref = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape(M, K),
                tla.make_stride(K, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        l0_b = tla.make_tensor_like(l0b_ptr, ref, layout)
        tla.experimental.img2col(
            l0_b,
            l1_a,
            Img2colParams(
                fmap_hw=(HI, WI),
                filter_hw=(KH, KW),
                repeat_load_stride=0,
                repeat_load_time=1,
                repeat_load_axis=tla.experimental.height,
            ),
        )


@tla.kernel
def img2col_dtype(
    fmap: tla.Tensor,
    dtype: tla.Constexpr,
    c0: tla.Constexpr,
    dst_space: tla.Constexpr,
    dst_layout: tla.Constexpr,
) -> None:
    k = CIN1 * KH * KW * c0
    l1_ptr = tla.allocate(CIN1 * FMAP_ROWS * c0, dtype, tla.AddressSpace.l1, 512)
    l0_ptr = tla.allocate(M * k, dtype, dst_space, 512)
    with tla.cube():
        gm_a = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape((FMAP_ROWS, 1), (c0, CIN1)),
                tla.make_stride((c0, FMAP_ROWS * c0), (1, FMAP_ROWS * c0)),
                origin_shape=tla.make_shape(FMAP_ROWS, CIN1 * c0),
                layoutTag=tla.arch.zNUnAlign,
            ),
        )
        l1_a = tla.make_tensor_like(l1_ptr, gm_a, tla.arch.zNUnAlign)
        ref = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape(M, k),
                tla.make_stride(k, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        dst = tla.make_tensor_like(l0_ptr, ref, dst_layout)
        tla.experimental.img2col(
            dst,
            l1_a,
            Img2colParams(
                fmap_hw=(HI, WI),
                filter_hw=(KH, KW),
                repeat_load_stride=0,
                repeat_load_time=1,
                repeat_load_axis=tla.experimental.height,
            ),
        )


@tla.kernel
def img2col_params_wrong_type(fmap: tla.Tensor) -> None:
    l1a_ptr = tla.allocate(CIN1 * FMAP_ROWS * C0, tla.Float16, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l0a, 512)
    with tla.cube():
        gm_a = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape(*_FMAP_SHAPE),
                tla.make_stride(*_FMAP_STRIDE),
                origin_shape=tla.make_shape(FMAP_ROWS, CIN),
                layoutTag=tla.arch.zNUnAlign,
            ),
        )
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zNUnAlign)
        ref = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape(M, K),
                tla.make_stride(K, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        l0_a = tla.make_tensor_like(l0a_ptr, ref, tla.arch.zN)
        tla.experimental.img2col(l0_a, l1_a, "not-a-params-object")


@tla.kernel
def img2col_pad_value_and_repeat(fmap: tla.Tensor) -> None:
    l1a_ptr = tla.allocate(CIN1 * FMAP_ROWS * C0, tla.Float16, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l0a, 512)
    with tla.cube():
        gm_a = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape(*_FMAP_SHAPE),
                tla.make_stride(*_FMAP_STRIDE),
                origin_shape=tla.make_shape(FMAP_ROWS, CIN),
                layoutTag=tla.arch.zNUnAlign,
            ),
        )
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zNUnAlign)
        tla.copy(l1_a, gm_a)
        ref = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape(M, K),
                tla.make_stride(K, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        l0_a = tla.make_tensor_like(l0a_ptr, ref, tla.arch.zN)
        window = tla.get_tile(l0_a, tla.make_coord(0, 0), tla.make_shape(M, 2 * C0))
        img2col(
            window,
            l1_a,
            Img2colParams(
                fmap_hw=(HI, WI),
                filter_hw=(KH, KW),
                pad=(1, 1, 1, 1),
                pad_value=7,
                repeat_load_stride=2,
                repeat_load_time=3,
                repeat_load_axis=tla.experimental.width,
            ),
        )


def test_img2col_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(img2col_3x3_pad1, type_args=(_gm_tensor(),))

    assert "tla.img2col" in mlir
    # The feature map is read through the zNUnAlign (NC1HWC0) view, and the
    # destination is a zN L0A tile.
    assert "zNUnAlign" in mlir
    assert "l0a" in mlir
    assert "pad_value = 0 : i32" in mlir
    assert "repeat_stride = 0 : i32" in mlir
    assert "repeat_time = 1 : i32" in mlir
    assert "repeat_mode = 0 : i32" in mlir


def test_img2col_emits_pad_value_and_repeat_load(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(img2col_pad_value_and_repeat, type_args=(_gm_tensor(),))

    assert "tla.img2col" in mlir
    assert "pad_value = 7 : i32" in mlir
    assert "repeat_stride = 2 : i32" in mlir
    assert "repeat_time = 3 : i32" in mlir
    assert "repeat_mode = 1 : i32" in mlir
    assert "dst_stride" not in mlir


@pytest.mark.parametrize(
    "field,value",
    [
        ("repeat_load_stride", -1),
        ("repeat_load_time", -1),
        ("repeat_load_axis", None),
        ("repeat_load_stride", 65536),
        ("repeat_load_time", 256),
        ("repeat_load_axis", 1),
    ],
)
def test_img2col_rejects_invalid_repeat_load(field: str, value: Any) -> None:
    params = Img2colParams(fmap_hw=(HI, WI), filter_hw=(KH, KW), **{field: value})

    with pytest.raises(tla.TlaCoreAPIError, match=rf"{field} must be"):

        @tla.kernel
        def bad(fmap: tla.Tensor) -> None:
            l1a_ptr = tla.allocate(CIN1 * FMAP_ROWS * C0, tla.Float16, tla.AddressSpace.l1, 512)
            l0a_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l0a, 512)
            with tla.cube():
                gm_a = tla.make_tensor(
                    fmap.ptr,
                    tla.make_layout(
                        tla.make_shape(*_FMAP_SHAPE),
                        tla.make_stride(*_FMAP_STRIDE),
                        origin_shape=tla.make_shape(FMAP_ROWS, CIN),
                        layoutTag=tla.arch.zNUnAlign,
                    ),
                )
                l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zNUnAlign)
                ref = tla.make_tensor(
                    fmap.ptr,
                    tla.make_layout(
                        tla.make_shape(M, K),
                        tla.make_stride(K, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                l0_a = tla.make_tensor_like(l0a_ptr, ref, tla.arch.zN)
                img2col(l0_a, l1_a, params)

        bad.dump_mlir(type_args=(_gm_tensor(),))


def test_img2col_carries_stride_dilation_and_asymmetric_pad(
    compiler_tlair: Any,
) -> None:
    mlir = compiler_tlair(img2col_stride2_dilation2, type_args=(_gm_tensor(),))

    assert "tla.img2col" in mlir
    # The geometry travels as index operands, not attributes, which is what lets
    # one compiled kernel serve any pad/stride/dilation.
    # compiler_tlair returns the GENERIC assembly form, in which a constant
    # prints as "arith.constant"() <{value = 4 : index}> -- not the pretty form.
    for value in (4, 5, 6, 7):
        assert f"value = {value} : index" in mlir


def test_img2col_source_must_be_l1_zNUnAlign() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match=r"src must be an l1 feature-map tile tagged zNUnAlign",
    ):
        img2col_src_not_zNUnAlign.dump_mlir(type_args=(_gm_tensor(),))


def test_img2col_accepts_nz_l0b_destination(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(img2col_dst_is_l0b, type_args=(_gm_tensor(), tla.arch.nZ))
    img2col_line = next(line for line in mlir.splitlines() if "tla.img2col" in line)
    assert f"!tla.shape<{M},{K}>, nZ>" in img2col_line
    assert "!tla.ptr<f16, l0b, 512>" in img2col_line


def test_img2col_rejects_zn_l0b_destination() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=r"l0b tile tagged nZ"):
        img2col_dst_is_l0b.dump_mlir(type_args=(_gm_tensor(), tla.arch.zN))


@pytest.mark.parametrize(
    "dtype,c0,space,layout",
    [
        (tla.Int16, 16, tla.AddressSpace.l0a, tla.arch.zN),
        (tla.UInt16, 16, tla.AddressSpace.l0b, tla.arch.nZ),
        (tla.Int8, 32, tla.AddressSpace.l0b, tla.arch.nZ),
        (tla.Float8E4M3FN, 32, tla.AddressSpace.l0b, tla.arch.nZ),
    ],
)
def test_img2col_rejects_unsupported_element_type(dtype, c0, space, layout) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=r"element type must be one of"):
        img2col_dtype.dump_mlir(
            type_args=(
                _gm_tensor(dtype, cols=CIN1 * KH * KW * c0),
                dtype,
                c0,
                space,
                layout,
            )
        )


def test_img2col_rejects_non_params_object() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=r"params must be an Img2colParams"):
        img2col_params_wrong_type.dump_mlir(type_args=(_gm_tensor(),))


def _img2col_k_window(
    k_extent: int, k_start: int = 0, filter_h: int = 3, repeat_time: int = 1
):
    @tla.kernel
    def kernel(fmap: tla.Tensor) -> None:
        l1_ptr = tla.allocate(
            CIN1 * FMAP_ROWS * C0, tla.Float16, tla.AddressSpace.l1, 512
        )
        l0_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l0a, 512)
        with tla.cube():
            src = tla.make_tensor(
                l1_ptr,
                tla.make_layout(
                    tla.make_shape(*_FMAP_SHAPE),
                    tla.make_stride(*_FMAP_STRIDE),
                    origin_shape=tla.make_shape(FMAP_ROWS, CIN),
                    layoutTag=tla.arch.zNUnAlign,
                ),
            )
            ref = tla.make_tensor(
                fmap.ptr,
                tla.make_layout(
                    tla.make_shape(M, K * 2),
                    tla.make_stride(K * 2, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            dst = tla.make_tensor_like(l0_ptr, ref, tla.arch.zN)
            dst = tla.get_tile(
                dst, tla.make_coord(0, k_start), tla.make_shape(M, k_extent)
            )
            tla.experimental.img2col(
                dst,
                src,
                Img2colParams(
                    fmap_hw=(HI, WI),
                    filter_hw=(filter_h, KW),
                    repeat_load_stride=1,
                    repeat_load_time=repeat_time,
                    repeat_load_axis=tla.experimental.width,
                ),
            )

    return kernel


@pytest.mark.parametrize(
    "k_extent,k_start,filter_h,message",
    [
        (
            144,
            8,
            3,
            "K coordinate must be static, within 0..65535 and aligned to 16 elements",
        ),
        (432, 0, 3, "K window exceeds"),
        (288, 0, 0, "filter_hw must contain"),
        (288, 0, 512, "filter_hw must contain"),
    ],
)
def test_img2col_rejects_invalid_k_window(k_extent, k_start, filter_h, message):
    with pytest.raises(tla.TlaCoreAPIError, match=message):
        _img2col_k_window(k_extent, k_start, filter_h).dump_mlir(
            type_args=(_gm_tensor(),)
        )


def test_img2col_rejects_last_repeat_outside_k_window():
    with pytest.raises(tla.TlaCoreAPIError, match="K window exceeds"):
        _img2col_k_window(C0, K - C0, repeat_time=2).dump_mlir(
            type_args=(_gm_tensor(),)
        )


@pytest.mark.parametrize(
    "k_extent,k_start",
    [
        (144, 0),  # one whole C1 slice -- the only window the old rule allowed
        (144, 144),
        (48, 48),  # inside one C1 slice
        (96, 96),  # straddles the boundary between two C1 slices
        (288, 0),  # two whole slices
        (16, 16),  # the C0 minimum
    ],
)
def test_img2col_accepts_c0_aligned_k_window(compiler_tlair, k_extent, k_start):
    # The K window only has to be C0-aligned: Load3Dv2 takes kStartPt/kExtension
    # from the destination tile and channelSize from the source, so a window
    # inside or across a C1 slice still describes the same feature map.
    mlir = compiler_tlair(
        _img2col_k_window(k_extent, k_start), type_args=(_gm_tensor(),)
    )
    assert "tla.img2col" in mlir
