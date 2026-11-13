# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Test-only FP32/FP4 register compositions built from public DSL operations."""

import catlass.tla as tla
from catlass.params import CastParams, RegSlot


@tla.jit
def round_odd_bf16_view(value):
    """Keep a sticky bit so a later FP4 cast cannot double-round FP32 input.

    The meaningful BF16 values occupy even halfwords. Arithmetic right shift
    is sufficient: only its low 16 bits are used. A deinterleave below discards
    the other halfwords, including sign extension for negative inputs.
    """
    bits = value.bitcast(tla.Int32)
    discarded = tla.bitwise_and(bits, tla.full(0xFFFF, dtype=tla.Int32))
    sticky = tla.where(
        tla.cmp(discarded, 0, "ne"),
        tla.full(1, dtype=tla.Int32),
        tla.full(0, dtype=tla.Int32),
    )
    upper = tla.shift_right(bits, 16)
    return tla.bitwise_or(upper, sticky).bitcast(tla.BFloat16)


@tla.jit
def fp32_to_fp4(first, second, dtype: tla.Constexpr, params: tla.Constexpr):
    """Encode two consecutive 64-value FP32 registers into 64 FP4 pairs.

    Output bytes occupy slot zero; params must select RegSlot.ZERO. Scale
    arithmetic and memory layout belong to the caller. The native BF16->FP4
    instruction performs the final rounding and saturation.
    """
    dense, _unused = tla.deinterleave(
        round_odd_bf16_view(first), round_odd_bf16_view(second)
    )
    return dense.to(dtype, params)


@tla.jit
def fp4_to_fp32(pairs):
    """Decode 64 FP4 pair bytes at slot zero into two dense FP32 registers."""
    dense = pairs.to(tla.BFloat16, CastParams())
    even = dense.to(tla.Float32, CastParams(reg_slot=RegSlot.ZERO))
    odd = dense.to(tla.Float32, CastParams(reg_slot=RegSlot.ONE))
    return tla.interleave(even, odd)


@tla.jit
def merge_owned_nibbles(encoded, previous, first_index, count: tla.Constexpr):
    """Preserve every output nibble whose source index is outside [0,count).

    encoded and previous are byte registers expanded to slot zero. first_index
    is a 64-lane Int32 vector describing the low nibble of each packed byte.
    The caller owns the whole byte while performing this read/modify/write;
    concurrent writers to adjacent nibbles require external coordination.
    """
    low = tla.bitwise_and(
        tla.cmp(first_index, 0, "ge"), tla.cmp(first_index, count, "lt")
    )
    high_index = first_index + 1
    high = tla.bitwise_and(
        tla.cmp(high_index, 0, "ge"), tla.cmp(high_index, count, "lt")
    )
    zero = tla.full(0, dtype=tla.Int32)
    owned = tla.bitwise_or(
        tla.where(low, tla.full(0x0F, dtype=tla.Int32), zero),
        tla.where(high, tla.full(0xF0, dtype=tla.Int32), zero),
    )
    return tla.bitwise_or(
        tla.bitwise_and(encoded.bitcast(tla.Int32), owned),
        tla.bitwise_and(previous.bitcast(tla.Int32), tla.bitwise_not(owned)),
    ).bitcast(tla.UInt8)
