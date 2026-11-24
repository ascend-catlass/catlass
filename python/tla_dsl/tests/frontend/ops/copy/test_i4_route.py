"""Frontend and verifier coverage for int4 GM<->UB RowMajor copies."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


def _gm(shape=(8, 16), stride=None):
    stride = stride or (shape[1], 1)
    return make_fake_tensor(
        tla.Int4x2,
        shape,
        stride,
        addrspace=tla.AddressSpace.gm,
        origin_shape=shape,
        layout_tag=tla.arch.RowMajor,
    )


def _gm_to_ub_kernel(gm_stride=None):
    shape = (8, 16)
    gm_stride = gm_stride or (16, 1)

    @tla.kernel
    def kernel(gm: tla.Tensor) -> None:
        ub_ptr = tla.allocate(shape, tla.Int4x2, tla.AddressSpace.ub, 32)
        ub = tla.make_tensor(
            ub_ptr,
            tla.make_layout(tla.make_shape(*shape), tla.make_stride(16, 1)),
        )
        with tla.vector():  # noqa: SIM117 -- separate DSL regions
            tla.copy(ub, gm)

    return kernel, _gm(shape, gm_stride)


def _ub_to_gm_kernel():
    shape = (8, 16)

    @tla.kernel
    def kernel(gm: tla.Tensor) -> None:
        ub_ptr = tla.allocate(shape, tla.Int4x2, tla.AddressSpace.ub, 32)
        ub = tla.make_tensor(
            ub_ptr,
            tla.make_layout(tla.make_shape(*shape), tla.make_stride(16, 1)),
        )
        with tla.vector():  # noqa: SIM117 -- separate DSL regions
            tla.copy(gm, ub)

    return kernel, _gm(shape)


@pytest.mark.parametrize("make_kernel", [_gm_to_ub_kernel, _ub_to_gm_kernel])
def test_i4_gm_ub_row_major_copy_is_accepted(make_kernel):
    kernel, gm = make_kernel()
    mlir = kernel.dump_mlir(type_args=(gm,))
    assert 'tla.copy' in mlir
    assert "!tla.i4x2" in mlir