# -------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the
# License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------

"""Rounding mode validation and encoding for VectorSSA.to."""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


def _ub(dtype: type[tla.Numeric]) -> tla.Tensor:
    lanes = 2048 // dtype.width
    return make_fake_tensor(
        dtype,
        (lanes,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(lanes,),
        layout_tag=tla.arch.RowMajor,
    )


def _cast_kernel(dst_dtype: type[tla.Numeric], params: tla.params.CastParams):
    @tla.kernel
    def k(src: tla.Tensor, dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst.store(src.load().to(dst_dtype, params))

    return k


@pytest.mark.parametrize(
    "mode, code",
    [
        (tla.params.RoundMode.CAST_ROUND, 0),
        (tla.params.RoundMode.CAST_FLOOR, 1),
        (tla.params.RoundMode.CAST_CEIL, 2),
        (tla.params.RoundMode.CAST_TRUNC, 3),
        (tla.params.RoundMode.CAST_NONE, 4),
        (tla.params.RoundMode.CAST_RINT, 5),
        (tla.params.RoundMode.CAST_ODD, 6),
        (tla.params.RoundMode.CAST_HYBRID, 7),
    ],
)
def test_round_mode_trait_codes(mode, code) -> None:
    assert tla.params.CastParams(round_mode=mode).codes()[2] == code


def test_same_type_f32_cast_odd_with_unknown_saturation() -> None:
    params = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.UNKNOWN,
        round_mode=tla.params.RoundMode.CAST_ODD,
    )
    mlir = _cast_kernel(tla.Float32, params).dump_mlir(
        type_args=(_ub(tla.Float32), _ub(tla.Float32))
    )
    assert "tla.cast" in mlir


@pytest.mark.parametrize(
    "src,dst",
    [
        (tla.Float32, tla.Float32),
        (tla.Float32, tla.Float16),
        (tla.Float16, tla.Int8),
        (tla.Float16, tla.UInt8),
        (tla.Int16, tla.Float16),
        (tla.Int32, tla.Float32),
        (tla.Float32, tla.Int32),
        (tla.Float32, tla.BFloat16),
        (tla.Float16, tla.Int32),
        (tla.BFloat16, tla.Float16),
        (tla.Float32, tla.Int64),
        (tla.Int64, tla.Float32),
        (tla.UInt8, tla.Float32),
        (tla.Float16, tla.Int4x2),
        (tla.BFloat16, tla.Float4E2M1x2),
        (tla.Float8E4M3FN, tla.Float16),
        (tla.Float8E5M2, tla.BFloat16),
        # CAST_NONE also works on instructions without a rounding operand.
        (tla.Float16, tla.Float32),
        (tla.Float8E4M3FN, tla.Float32),
        (tla.Float8E5M2, tla.Float16),
        (tla.Int16, tla.Float32),
        (tla.UInt8, tla.Float16),
        (tla.Int16, tla.Int4x2),
        (tla.Int64, tla.Int32),
    ],
)
def test_cast_preserves_default_round_mode(src, dst) -> None:
    params = tla.params.CastParams()
    expected_trait = "[0, 2, 4]"
    if dst is tla.Float4E2M1x2:
        params = tla.params.CastParams(sat_mode=tla.params.SatMode.UNKNOWN)
        expected_trait = "[0, 0, 4]"
    ir = _cast_kernel(dst, params).dump_mlir(type_args=(_ub(src), _ub(dst)))
    (cast_line,) = [line for line in ir.splitlines() if "tla.cast" in line]
    assert expected_trait in cast_line


@pytest.mark.parametrize(
    "src,dst",
    [
        (tla.BFloat16, tla.Float16),
        (tla.Float16, tla.Int32),
        (tla.Float32, tla.Int64),
        (tla.Int64, tla.Float32),
    ],
)
def test_cast_odd_rejected_by_register_conversion(src, dst) -> None:
    params = tla.params.CastParams(round_mode=tla.params.RoundMode.CAST_ODD)
    with pytest.raises(
        tla.TlaCoreAPIError,
        match=f"CAST_ODD is unsupported for {src.dtype}->{dst.dtype}",
    ):
        _cast_kernel(dst, params).dump_mlir(type_args=(_ub(src), _ub(dst)))
