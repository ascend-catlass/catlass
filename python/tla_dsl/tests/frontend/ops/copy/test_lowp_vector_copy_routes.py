# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend smoke for low-precision / VectorLayout / UB→UB two-arg copy routes."""

from __future__ import annotations

from typing import Any

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


def _gm_vl(dtype: type[tla.Numeric], n: int = 16) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (n,),
        (1,),
        origin_shape=(n,),
        layout_tag=tla.arch.VectorLayout,
    )


def _ub_vl(dtype: type[tla.Numeric], n: int = 16) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (n,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(n,),
        layout_tag=tla.arch.VectorLayout,
    )


def _ub_rm(dtype: type[tla.Numeric], shape: tuple[int, int] = (16, 16)) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        shape,
        (shape[1], 1),
        addrspace=tla.AddressSpace.ub,
        origin_shape=shape,
        layout_tag=tla.arch.RowMajor,
    )


def _ub_cm(dtype: type[tla.Numeric], shape: tuple[int, int] = (32, 32)) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        shape,
        (1, shape[0]),
        addrspace=tla.AddressSpace.ub,
        origin_shape=shape,
        layout_tag=tla.arch.ColumnMajor,
    )


@tla.kernel
def copy_two_arg(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(dst, src)


def test_gm2ub_vl_fp8_two_arg(compiler_tlair: Any) -> None:
    mlir = copy_two_arg.dump_mlir(
        type_args=(_ub_vl(tla.Float8E4M3FN), _gm_vl(tla.Float8E4M3FN)),
    )
    assert "tla.copy" in mlir


def test_ub2gm_vl_f32_two_arg(compiler_tlair: Any) -> None:
    mlir = copy_two_arg.dump_mlir(
        type_args=(_gm_vl(tla.Float32), _ub_vl(tla.Float32)),
    )
    assert "tla.copy" in mlir


def test_ub2ub_rm_hif8_two_arg(compiler_tlair: Any) -> None:
    # 1B elems need row pitch multiple of 32B for MOV.UB.TO.UB.
    mlir = copy_two_arg.dump_mlir(
        type_args=(_ub_rm(tla.HiFloat8, (16, 32)), _ub_rm(tla.HiFloat8, (16, 32))),
    )
    assert "tla.copy" in mlir


def test_ub2ub_rm_b64_two_arg(compiler_tlair: Any) -> None:
    mlir = copy_two_arg.dump_mlir(
        type_args=(_ub_rm(tla.Int64, (8, 4)), _ub_rm(tla.Int64, (8, 4))),
    )
    assert "tla.copy" in mlir


def test_ub2ub_rm_cm_hif8_two_arg(compiler_tlair: Any) -> None:
    mlir = copy_two_arg.dump_mlir(
        type_args=(_ub_cm(tla.HiFloat8), _ub_rm(tla.HiFloat8, (32, 32))),
    )
    assert "tla.copy" in mlir


def test_ub2ub_rm_cm_e8m0_two_arg(compiler_tlair: Any) -> None:
    mlir = copy_two_arg.dump_mlir(
        type_args=(_ub_cm(tla.Float8E8M0), _ub_rm(tla.Float8E8M0, (32, 32))),
    )
    assert "tla.copy" in mlir
