from __future__ import annotations

import pytest

from catlass import tla
from catlass.tla.runtime import make_fake_tensor


def _gm(shape, stride=None):
    stride = stride or (shape[1], 1)
    return make_fake_tensor(
        tla.Int4x2,
        shape,
        stride,
        addrspace=tla.AddressSpace.gm,
        origin_shape=shape,
        layout_tag=tla.arch.RowMajor,
    )


def _gm_to_ub_kernel(shape, ub_stride=None):
    ub_stride = ub_stride or (shape[1], 1)

    @tla.kernel
    def kernel(gm: tla.Tensor) -> None:
        ub_ptr = tla.allocate(shape, tla.Int4x2, tla.AddressSpace.ub, 32)
        ub = tla.make_tensor(
            ub_ptr,
            tla.make_layout(tla.make_shape(*shape), tla.make_stride(*ub_stride)),
        )
        with tla.vector():
            tla.copy(ub, gm)

    return kernel


def _ub_to_gm_kernel(shape):
    @tla.kernel
    def kernel(gm: tla.Tensor) -> None:
        ub_ptr = tla.allocate(shape, tla.Int4x2, tla.AddressSpace.ub, 32)
        ub = tla.make_tensor(
            ub_ptr,
            tla.make_layout(tla.make_shape(*shape), tla.make_stride(shape[1], 1)),
        )
        with tla.vector():
            tla.copy(gm, ub)

    return kernel


@pytest.mark.parametrize(
    ("make_kernel", "src_first"),
    [(_gm_to_ub_kernel, False), (_ub_to_gm_kernel, True)],
)
def test_int4_row_major_gm_ub_routes_are_emitted(
    compiler_tlair, make_kernel, src_first
):
    ir = compiler_tlair(make_kernel((8, 16)), type_args=(_gm((8, 16)),))
    assert '"tla.copy"' in ir
    assert "!tla.i4x2" in ir


def test_int4x2_allows_odd_row_stride(compiler_tlair):
    ir = compiler_tlair(
        _gm_to_ub_kernel((8, 16), ub_stride=(17, 1)),
        type_args=(_gm((8, 16)),),
    )
    assert '"tla.copy"' in ir
    assert "!tla.i4x2" in ir


def test_int4x2_allows_odd_origin_extent(compiler_tlair):
    ir = compiler_tlair(
        _gm_to_ub_kernel((8, 15), ub_stride=(16, 1)),
        type_args=(_gm((8, 15), stride=(16, 1)),),
    )
    assert '"tla.copy"' in ir
    assert "!tla.i4x2" in ir


def _gm_vl(extent: int = 16):
    return make_fake_tensor(
        tla.Int4x2,
        (extent,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(extent,),
        layout_tag=tla.arch.VectorLayout,
    )


def test_int4x2_vector_layout_gm_ub_roundtrip_emits(compiler_tlair):
    @tla.kernel
    def kernel(gm_src: tla.Tensor, gm_dst: tla.Tensor) -> None:
        ub_ptr = tla.allocate(16, tla.Int4x2, tla.AddressSpace.ub, 32)
        with tla.vector():
            ub = tla.make_tensor(
                ub_ptr,
                tla.make_layout(
                    tla.make_shape(16),
                    tla.make_stride(1),
                    origin_shape=tla.make_shape(16),
                    layoutTag=tla.arch.VectorLayout,
                ),
            )
            tla.copy(ub, gm_src)
            tla.copy(gm_dst, ub)

    ir = compiler_tlair(kernel, type_args=(_gm_vl(16), _gm_vl(16)))
    assert '"tla.copy"' in ir
    assert "!tla.i4x2" in ir


def test_int4x2_gm_ub_pad_accepted(compiler_tlair):
    @tla.kernel
    def kernel(gm: tla.Tensor, ub: tla.Tensor) -> None:
        with tla.vector():
            tla.copy(
                ub,
                gm,
                params=tla.params.CopyPadParams(right_padding=1, padding_value=0xA5),
            )

    gm = make_fake_tensor(
        tla.Int4x2,
        (2, 31),
        (31, 1),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(2, 31),
        layout_tag=tla.arch.RowMajor,
    )
    ub = make_fake_tensor(
        tla.Int4x2,
        (2, 32),
        (32, 1),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(2, 32),
        layout_tag=tla.arch.RowMajor,
    )
    ir = compiler_tlair(kernel, type_args=(gm, ub))
    assert "tla.CopyPadParams" in ir
