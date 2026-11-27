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

"""Rounding and saturation mode validation and encoding for VectorSSA.to."""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

_EXACT_WIDEN_PAIRS = (
    (tla.Float16, tla.Float32),
    (tla.BFloat16, tla.Float32),
    (tla.Float8E4M3FN, tla.Float16),
    (tla.Float8E4M3FN, tla.BFloat16),
    (tla.Float8E4M3FN, tla.Float32),
    (tla.Float8E5M2, tla.Float16),
    (tla.Float8E5M2, tla.BFloat16),
    (tla.Float8E5M2, tla.Float32),
    (tla.HiFloat8, tla.Float16),
    (tla.HiFloat8, tla.Float32),
    (tla.Float8E8M0, tla.BFloat16),
    (tla.Float4E2M1x2, tla.BFloat16),
    (tla.Float4E1M2x2, tla.BFloat16),
    (tla.Int4x2, tla.Int16),
    (tla.Int4x2, tla.Float16),
    (tla.Int4x2, tla.BFloat16),
    (tla.Int8, tla.Int16),
    (tla.Int8, tla.Int32),
    (tla.Int8, tla.Float16),
    (tla.Int16, tla.Int32),
    (tla.Int16, tla.Float32),
    (tla.Int32, tla.Int64),
    (tla.UInt8, tla.UInt16),
    (tla.UInt8, tla.UInt32),
    (tla.UInt8, tla.Float16),
    (tla.UInt8, tla.Float32),
    (tla.UInt16, tla.UInt32),
)

_CASTS_WITHOUT_SATURATION = (
    (tla.Float16, tla.BFloat16),
    (tla.Float16, tla.Int32),
    (tla.Int16, tla.Float16),
    (tla.Int16, tla.UInt32),
    (tla.Int32, tla.Float32),
    (tla.Int64, tla.Float32),
)


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


@pytest.mark.parametrize(
    "src,dst",
    [
        (tla.Float32, tla.Float32),
        (tla.Float32, tla.Float16),
        (tla.Float16, tla.Int8),
        (tla.Float16, tla.UInt8),
        (tla.Float32, tla.Int32),
        (tla.Float32, tla.BFloat16),
        (tla.BFloat16, tla.Float16),
        (tla.Float32, tla.Int64),
        (tla.Float16, tla.Int4x2),
        (tla.BFloat16, tla.Float4E2M1x2),
        # CAST_NONE also works on instructions without a rounding operand.
        (tla.Int16, tla.Int8),
        (tla.Int32, tla.Int8),
        (tla.Int16, tla.Int4x2),
        (tla.Int64, tla.Int32),
        *_EXACT_WIDEN_PAIRS,
        *_CASTS_WITHOUT_SATURATION,
    ],
)
def test_cast_preserves_default_round_mode(src, dst) -> None:
    if (
        src is dst
        or dst is tla.Float4E2M1x2
        or (src, dst) in (*_EXACT_WIDEN_PAIRS, *_CASTS_WITHOUT_SATURATION)
    ):
        params = tla.params.CastParams()
        expected_trait = "[0, 0, 4]"
    else:
        params = tla.params.CastParams(sat_mode=tla.params.SatMode.NOSAT)
        expected_trait = "[0, 2, 4]"
    ir = _cast_kernel(dst, params).dump_mlir(type_args=(_ub(src), _ub(dst)))
    (cast_line,) = [line for line in ir.splitlines() if "tla.cast" in line]
    assert expected_trait in cast_line


@pytest.mark.parametrize(
    "src,dst",
    (
        (tla.Float32, tla.Float16),
        (tla.Float32, tla.Int32),
        (tla.Float32, tla.Float8E4M3FN),
        (tla.BFloat16, tla.Float8E5M2),
        (tla.Float16, tla.HiFloat8),
        (tla.Int32, tla.UInt8),
        (tla.Int16, tla.Int4x2),
    ),
)
def test_cast_with_saturation_operand_requires_explicit_mode(src, dst) -> None:
    with pytest.raises(
        tla.TlaCoreAPIError, match=r"requires explicit SatMode\.SAT or SatMode\.NOSAT"
    ):
        _cast_kernel(dst, tla.params.CastParams()).dump_mlir(
            type_args=(_ub(src), _ub(dst))
        )


@pytest.mark.parametrize(
    "src,dst",
    [
        (tla.BFloat16, tla.Float16),
        (tla.BFloat16, tla.Int32),
        (tla.Float16, tla.BFloat16),
        (tla.Float16, tla.Int8),
        (tla.Float16, tla.Int16),
        (tla.Float16, tla.Int32),
        (tla.Float16, tla.UInt8),
        (tla.Float32, tla.BFloat16),
        (tla.Float32, tla.Int16),
        (tla.Float32, tla.Int32),
        (tla.Float32, tla.Int64),
        (tla.Int16, tla.Float16),
        (tla.Int32, tla.Float32),
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


@pytest.mark.parametrize("src,dst", _CASTS_WITHOUT_SATURATION)
@pytest.mark.parametrize("sat", (tla.params.SatMode.SAT, tla.params.SatMode.NOSAT))
def test_cast_without_saturation_operand_requires_unknown(src, dst, sat) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="UNKNOWN"):
        _cast_kernel(dst, tla.params.CastParams(sat_mode=sat)).dump_mlir(
            type_args=(_ub(src), _ub(dst))
        )


@pytest.mark.parametrize(
    "src,dst",
    (
        (tla.Int16, tla.Int4x2),
        (tla.Int16, tla.Int8),
        (tla.Int16, tla.UInt8),
        (tla.Int16, tla.UInt32),
        (tla.Int32, tla.Int8),
        (tla.Int32, tla.Int16),
        (tla.Int32, tla.UInt8),
        (tla.Int32, tla.UInt16),
        (tla.Int64, tla.Int32),
        (tla.UInt16, tla.UInt8),
        (tla.UInt32, tla.Int16),
        (tla.UInt32, tla.UInt8),
        (tla.UInt32, tla.UInt16),
    ),
)
def test_integer_cast_without_rounding_operand_requires_none(src, dst) -> None:
    sat = tla.params.SatMode.UNKNOWN if dst is tla.UInt32 else tla.params.SatMode.NOSAT
    params = tla.params.CastParams(
        sat_mode=sat, round_mode=tla.params.RoundMode.CAST_RINT
    )
    with pytest.raises(tla.TlaCoreAPIError, match="requires CAST_NONE"):
        _cast_kernel(dst, params).dump_mlir(type_args=(_ub(src), _ub(dst)))


@pytest.mark.parametrize("src,dst", _EXACT_WIDEN_PAIRS)
@pytest.mark.parametrize(
    "sat,round_mode",
    (
        *(
            (tla.params.SatMode.UNKNOWN, mode)
            for mode in tla.params.RoundMode
            if mode is not tla.params.RoundMode.CAST_NONE
        ),
        (tla.params.SatMode.SAT, tla.params.RoundMode.CAST_NONE),
        (tla.params.SatMode.NOSAT, tla.params.RoundMode.CAST_NONE),
    ),
)
def test_exact_widen_requires_none_and_unknown(src, dst, sat, round_mode) -> None:
    params = tla.params.CastParams(sat_mode=sat, round_mode=round_mode)
    with pytest.raises(
        tla.TlaCoreAPIError,
        match=f"{src.dtype}->{dst.dtype} requires CAST_NONE and SatMode.UNKNOWN",
    ):
        _cast_kernel(dst, params).dump_mlir(type_args=(_ub(src), _ub(dst)))


@pytest.mark.parametrize("src", (tla.Int16, tla.Int32))
@pytest.mark.parametrize("sat", (tla.params.SatMode.SAT, tla.params.SatMode.UNKNOWN))
def test_signed_narrow_to_int8_requires_nosat(src, sat) -> None:
    with pytest.raises(
        tla.TlaCoreAPIError, match=f"{src.__name__} to Int8 requires SatMode.NOSAT"
    ):
        _cast_kernel(tla.Int8, tla.params.CastParams(sat_mode=sat)).dump_mlir(
            type_args=(_ub(src), _ub(tla.Int8))
        )


@pytest.mark.parametrize("slot", (tla.params.RegSlot.TWO, tla.params.RegSlot.THREE))
def test_int16_to_int8_requires_even_or_odd_slot(slot) -> None:
    params = tla.params.CastParams(reg_slot=slot, sat_mode=tla.params.SatMode.NOSAT)
    with pytest.raises(tla.TlaCoreAPIError, match="pack quarter"):
        _cast_kernel(tla.Int8, params).dump_mlir(type_args=(_ub(tla.Int16), _ub(tla.Int8)))


@pytest.mark.parametrize("slot", (tla.params.RegSlot.TWO, tla.params.RegSlot.THREE))
def test_int32_to_int8_accepts_pack_quarter_slots(slot) -> None:
    params = tla.params.CastParams(reg_slot=slot, sat_mode=tla.params.SatMode.NOSAT)
    _cast_kernel(tla.Int8, params).dump_mlir(type_args=(_ub(tla.Int32), _ub(tla.Int8)))
