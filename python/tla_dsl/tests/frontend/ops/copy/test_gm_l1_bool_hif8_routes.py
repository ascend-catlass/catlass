# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend acceptance for GM→L1 bool (i1) and hif8 DataCopy routes (T5)."""

from __future__ import annotations

import pytest

from catlass import tla
from catlass.core_api import _remap_tensor_like_prefix_fields_for_layout_trees
from catlass.runtime import TlaCoreAPIError
from catlass.tla.runtime import make_fake_tensor

# Tensor i1 / hif8 are 1-byte elements → C0 = 32.
_C0 = 32
_ROWS = 32
_COLS = 32


def _gm_row(dtype):
    return make_fake_tensor(
        dtype,
        (_ROWS, _COLS),
        (_COLS, 1),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(_ROWS, _COLS),
        layout_tag=tla.arch.RowMajor,
    )


def _gm_col(dtype):
    return make_fake_tensor(
        dtype,
        (_ROWS, _COLS),
        (1, _ROWS),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(_ROWS, _COLS),
        layout_tag=tla.arch.ColumnMajor,
    )


def _zn_tensor(dtype, addrspace):
    origin = (_ROWS, _COLS)
    geometry = _remap_tensor_like_prefix_fields_for_layout_trees(
        origin, dtype.dtype, str(tla.arch.zN)
    )
    assert geometry is not None
    return make_fake_tensor(
        dtype,
        geometry.shape,
        geometry.stride,
        addrspace=addrspace,
        origin_shape=origin,
        layout_tag=tla.arch.zN,
    )


def _nz_tensor(dtype, addrspace):
    c0 = 32
    shape = ((c0, _ROWS // c0), (16, _COLS // 16))
    stride = ((1, _COLS * c0), (c0, 16 * c0))
    return make_fake_tensor(
        dtype,
        shape,
        stride,
        addrspace=addrspace,
        origin_shape=(_ROWS, _COLS),
        layout_tag=tla.arch.nZ,
    )


def _vec(dtype, addrspace, n=32):
    return make_fake_tensor(
        dtype,
        (n,),
        (1,),
        addrspace=addrspace,
        origin_shape=(n,),
        layout_tag=tla.arch.VectorLayout,
    )


@tla.kernel
def gm_to_l1_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(dst, src)


@pytest.mark.parametrize("dtype", [tla.Bool, tla.HiFloat8])
def test_gm_rowmajor_to_l1_zn_accepts_bool_hif8(dtype) -> None:
    mlir = gm_to_l1_kernel.dump_mlir(
        type_args=(_zn_tensor(dtype, tla.AddressSpace.l1), _gm_row(dtype))
    )
    assert "tla.copy" in mlir


@pytest.mark.parametrize("dtype", [tla.Bool, tla.HiFloat8])
def test_gm_colmajor_to_l1_nz_accepts_bool_hif8(dtype) -> None:
    mlir = gm_to_l1_kernel.dump_mlir(
        type_args=(_nz_tensor(dtype, tla.AddressSpace.l1), _gm_col(dtype))
    )
    assert "tla.copy" in mlir


@pytest.mark.parametrize("dtype", [tla.Bool, tla.HiFloat8])
def test_gm_nz_to_l1_nz_accepts_bool_hif8(dtype) -> None:
    mlir = gm_to_l1_kernel.dump_mlir(
        type_args=(
            _nz_tensor(dtype, tla.AddressSpace.l1),
            _nz_tensor(dtype, tla.AddressSpace.gm),
        )
    )
    assert "tla.copy" in mlir


@pytest.mark.parametrize("dtype", [tla.Bool, tla.HiFloat8])
def test_gm_vector_to_l1_vector_accepts_bool_hif8(dtype) -> None:
    mlir = gm_to_l1_kernel.dump_mlir(
        type_args=(
            _vec(dtype, tla.AddressSpace.l1),
            _vec(dtype, tla.AddressSpace.gm),
        )
    )
    assert "tla.copy" in mlir


def test_gm_rowmajor_to_l1_zn_rejects_int64() -> None:
    """Bool/hif8 GM→L1 expansion must not silently accept unregistered dtypes."""
    with pytest.raises(TlaCoreAPIError, match="unsupported copy route"):
        gm_to_l1_kernel.dump_mlir(
            type_args=(
                _zn_tensor(tla.Int64, tla.AddressSpace.l1),
                _gm_row(tla.Int64),
            )
        )


def _ub_row(dtype):
    return make_fake_tensor(
        dtype,
        (_ROWS, _COLS),
        (_COLS, 1),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(_ROWS, _COLS),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def ub_to_l1_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(dst, src)


def test_ub_rowmajor_to_l1_zn_accepts_hif8() -> None:
    mlir = ub_to_l1_kernel.dump_mlir(
        type_args=(_zn_tensor(tla.HiFloat8, tla.AddressSpace.l1), _ub_row(tla.HiFloat8))
    )
    assert "tla.copy" in mlir


def test_ub_zn_to_l1_zn_accepts_hif8() -> None:
    mlir = ub_to_l1_kernel.dump_mlir(
        type_args=(
            _zn_tensor(tla.HiFloat8, tla.AddressSpace.l1),
            _zn_tensor(tla.HiFloat8, tla.AddressSpace.ub),
        )
    )
    assert "tla.copy" in mlir
