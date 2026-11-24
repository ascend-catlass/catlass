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
        with tla.vector():  # noqa: SIM117 -- separate DSL regions
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
        with tla.vector():  # noqa: SIM117 -- separate DSL regions
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
