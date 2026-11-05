# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any

import catlass.tla as tla

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

# tla.cast exercise: a single-block, multi-chunk kernel (256 elements = 4 exact
# 64-lane chunks) that runs four cast paths over different element types and
# combines them into one f32 output. Every tla.params.CastParams attribute *value* is used at
# least once across the casts below:
#
#   tla.params.RegSlot        : ZERO, ONE (2x part_even/odd); ZERO..THREE (4x pp0..pp3)
#   tla.params.SatMode        : NOSAT, SAT, UNKNOWN
#   tla.params.RoundMode      : CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC
#
# Coverage notes:
#   - On a float->float narrowing (f32->f16/bf16) the round_mode rounds to the
#     nearest representable target float (sub-ULP); for the exactly-representable
#     inputs here it is a no-op, so those round modes do not change the result.
#   - Integer rounding (trunc/floor) is observable on the float->int casts.
#   - This lowering treats sat_mode==UNKNOWN as non-saturating, so the UNKNOWN sat
#     is placed on a float->float cast where it cannot change numerics.
#
# The paths:
#   f16 add   : sum = (a as f16) + (b as f16)                     -> f32
#   bf16 max  : mx  = max(c as bf16, d as bf16)  (part_odd)       -> f32
#   bf16 min  : mn  = min(c as bf16, d as bf16)  (part_even)      -> f32
#   i32 mul   : pr  = (trunc(a) as i32) * (floor(e) as i32)       -> f32
#   int chain : ic  = float(trunc(a)) via i32->i16->i8->i16->i32 (2x steps),
#                     i32->i8->i32 (4x direct), and i16->f16 (int->float)
#   i16 mul   : p16 = trunc(a) * trunc(c), computed at i16 width -> f32
#   i8  add   : s8  = trunc(a) + trunc(b), computed at i8 width  -> f32
#   fp8       : every leg the vector unit accepts. e4m3 and e5m2 against f32
#               (one instruction each), and against f16 and bf16, which have no
#               fp8 instruction and are expanded by the lowering into two ops
#               through f32. The composed routes that describe the same value as
#               their f32 round trip are subtracted, so they contribute zero
#               unless a composition is wrong.
#   same-type : st  = trunc(a) + trunc(c) via f32->f32, with zero-difference
#               legs through f16->f16 and bf16->bf16. A same-type float cast
#               keeps the element type and only the round mode lands, so the
#               rounding is directly observable (1.5 -> 1.0): CAST_TRUNC on
#               f32/bf16, CAST_FLOOR on f16.
#
# The integer chain exercises: i32<->i16 and i16<->i8 (2x, even/odd part),
# i32<->i8 (4x, pack pattern), and i16->f16 (int->float). int<->int casts do not
# round (truncation); the small positive values round-trip losslessly.
VECTOR_ELE = 256
VL_ELE = 64
NUM_CHUNKS = 4
LOOPS = NUM_CHUNKS
ALL_DTYPES = ("f32",)

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)


# --- fp8 path traits ---------------------------------------------------------
# fp8 pairs with f32 in hardware and uses all four pp0..pp3 byte positions.
# The f16/bf16 routes compose through f32 with the compact pp0/pp1 layout.
def _fp8_pp_trait(layout: tla.params.RegSlot) -> tla.params.CastParams:
    return tla.params.CastParams(
        reg_slot=layout,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )


_TRAIT_FP8_PP0 = _fp8_pp_trait(tla.params.RegSlot.ZERO)
_TRAIT_FP8_PP1 = _fp8_pp_trait(tla.params.RegSlot.ONE)
_TRAIT_FP8_PP2 = _fp8_pp_trait(tla.params.RegSlot.TWO)
_TRAIT_FP8_PP3 = _fp8_pp_trait(tla.params.RegSlot.THREE)
# Composed fp8<->f16/bf16 uses the compact pp0/pp1 layout.
_TRAIT_FP8 = _TRAIT_FP8_PP0

# --- f16 path traits ---------------------------------------------------------
# a -> f16: even lane, no saturation, zeroing, round-to-nearest.
_TRAIT_F16_A = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_FLOOR,  # float->float: sub-ULP, no-op here
)
# b -> f16: carries the two UNKNOWN values (benign on a float->float cast).
_TRAIT_F16_B = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.UNKNOWN,
    round_mode=tla.params.RoundMode.CAST_TRUNC,  # float->float: sub-ULP, no-op here
)
# (a+b in f16) -> f32: even widen.
_TRAIT_TO_F32_EVEN = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)

# --- bf16 max path (ODD packing) ---------------------------------------------
# c, d -> bf16: odd lane, merging, ceil.
_TRAIT_BF16_ODD = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ONE,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_CEIL,
)
# max(c,d) -> f32: odd widen (must match the narrow's part_odd).
_TRAIT_TO_F32_ODD = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ONE,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)

# --- bf16 min path (EVEN packing) --------------------------------------------
# c, d -> bf16: even lane, zeroing, ceil.
_TRAIT_BF16_EVEN = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_CEIL,
)
# min(c,d) -> f32: even widen.
_TRAIT_TO_F32_EVEN2 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)

# --- i32 path traits ---------------------------------------------------------
# a -> i32: saturating, truncate toward zero.
_TRAIT_I32_TRUNC = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.SAT,
    round_mode=tla.params.RoundMode.CAST_TRUNC,
)
# e -> i32: saturating, floor.
_TRAIT_I32_FLOOR = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.SAT,
    round_mode=tla.params.RoundMode.CAST_FLOOR,
)
# (a*e in i32) -> f32.
_TRAIT_TO_F32_INT = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)

# --- integer-to-integer / integer-to-float chain traits ----------------------
# int<->int casts do not round; round_mode is carried but ignored. The 2x width
# steps (i32<->i16, i16<->i8) use the even/odd part; the 4x step (i32<->i8) uses
# the pack pattern internally. Keep all these even/zeroing so a narrow followed
# by the matching widen round-trips.
_TRAIT_INT = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_TRUNC,
)
# i16 -> f16 (int -> float; exact for the small integers here).
_TRAIT_I16_F16 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)


# The four pack quarters for the 4x i32<->i8 cast: reg_slot selects pp0..pp3.
# A narrow (i32->i8) and its widen (i8->i32) must use the SAME quarter to
# round-trip (the widen reads the quarter the narrow wrote).
def _i8_pp_trait(layout: tla.params.RegSlot) -> tla.params.CastParams:
    return tla.params.CastParams(
        reg_slot=layout,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_TRUNC,
    )


_TRAIT_I8_PP0 = _i8_pp_trait(tla.params.RegSlot.ZERO)
_TRAIT_I8_PP1 = _i8_pp_trait(tla.params.RegSlot.ONE)
_TRAIT_I8_PP2 = _i8_pp_trait(tla.params.RegSlot.TWO)
_TRAIT_I8_PP3 = _i8_pp_trait(tla.params.RegSlot.THREE)

# Unsigned twin of the i8 chain: f16→u8→u32/f32 with values a+200 (>=128).
_TRAIT_U = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_TRUNC,
)

# --- same-type float traits ---------------------------------------------------
# f16->f16 / bf16->bf16 / f32->f32 keep the element type and re-round every
# lane, so a round mode is directly observable (1.5 -> 1.0): CAST_TRUNC on
# f32/bf16 and CAST_FLOOR on f16. reg_slot stays on
# ZERO/ONE: an equal-width conversion reads the whole register, and SAT is
# refused because a same-type cast never saturates.
_TRAIT_SAME_F32 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.UNKNOWN,
    round_mode=tla.params.RoundMode.CAST_TRUNC,
)
_TRAIT_SAME_F16 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_FLOOR,
)
_TRAIT_SAME_BF16 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ONE,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_TRUNC,
)


@tla.kernel
def cast_multi(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_c: tla.Tensor,
    mem_d: tla.Tensor,
    mem_e: tla.Tensor,
    mem_out: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    a_gm = tla.tile_view(mem_a, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    b_gm = tla.tile_view(mem_b, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    c_gm = tla.tile_view(mem_c, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    d_gm = tla.tile_view(mem_d, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    e_gm = tla.tile_view(mem_e, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    out_gm = tla.tile_view(mem_out, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    a_ub = _make_ub_tensor(a_gm)
    b_ub = _make_ub_tensor(b_gm)
    c_ub = _make_ub_tensor(c_gm)
    d_ub = _make_ub_tensor(d_gm)
    e_ub = _make_ub_tensor(e_gm)
    out_ub = _make_ub_tensor(out_gm)

    with tla.vector():
        tla.copy(a_ub, a_gm)
        tla.copy(b_ub, b_gm)
        tla.copy(c_ub, c_gm)
        tla.copy(d_ub, d_gm)
        tla.copy(e_ub, e_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range(NUM_CHUNKS):
                a_v = _chunk(a_ub, i).load()
                b_v = _chunk(b_ub, i).load()
                c_v = _chunk(c_ub, i).load()
                d_v = _chunk(d_ub, i).load()
                e_v = _chunk(e_ub, i).load()
                out_chunk = _chunk(out_ub, i)

                # A full (all-lanes) predicate mask, threaded through the primary
                # f32-source casts so they exercise the masked cast path (an
                # all-true predicate leaves the result unchanged).
                full_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)

                # f16 add (a uses FLOOR trait, b carries the UNKNOWN sat).
                a_h = a_v.to(tla.Float16, _TRAIT_F16_A, full_mask)
                b_h = b_v.to(tla.Float16, _TRAIT_F16_B, full_mask)
                sum_f32 = (a_h + b_h).to(tla.Float32, _TRAIT_TO_F32_EVEN)

                # bf16 max, part_odd (merging).
                c_bf_o = c_v.to(tla.BFloat16, _TRAIT_BF16_ODD, full_mask)
                d_bf_o = d_v.to(tla.BFloat16, _TRAIT_BF16_ODD, full_mask)
                max_f32 = tla.max(c_bf_o, d_bf_o).to(tla.Float32, _TRAIT_TO_F32_ODD)

                # bf16 min, part_even (zeroing).
                c_bf_e = c_v.to(tla.BFloat16, _TRAIT_BF16_EVEN, full_mask)
                d_bf_e = d_v.to(tla.BFloat16, _TRAIT_BF16_EVEN, full_mask)
                min_f32 = tla.min(c_bf_e, d_bf_e).to(tla.Float32, _TRAIT_TO_F32_EVEN2)

                # i32 mul (a trunc, e floor, saturating).
                a_i = a_v.to(tla.Int32, _TRAIT_I32_TRUNC, full_mask)
                e_i = e_v.to(tla.Int32, _TRAIT_I32_FLOOR, full_mask)
                prod_f32 = (a_i * e_i).to(tla.Float32, _TRAIT_TO_F32_INT)

                # integer narrow/widen chain, all routes computing float(trunc(a)):
                #   2x chain : i32 -> i16 -> i8 -> i16 -> i32 -> f32
                #   4x direct: i32 -> i8 -> i32 -> f32, through every pack quarter
                #   int->flt : i16 -> f16 -> f32
                i16a = a_i.to(tla.Int16, _TRAIT_INT)  # i32 -> i16 (2x)
                i8a = i16a.to(tla.Int8, _TRAIT_INT)  # i16 -> i8  (2x)
                i16b = i8a.to(tla.Int16, _TRAIT_INT)  # i8  -> i16 (2x)
                chain_i32 = i16b.to(tla.Int32, _TRAIT_INT)  # i16 -> i32 (2x)
                chain_f32 = chain_i32.to(tla.Float32, _TRAIT_TO_F32_INT)

                # 4x i32 -> i8 -> i32 through each of the four pack quarters
                # (pp0..pp3 via reg slot ZERO/ONE/TWO/THREE). Narrow and widen use
                # the same quarter, so each round-trips to trunc(a).
                pp0_f32 = (
                    a_i.to(tla.Int8, _TRAIT_I8_PP0)
                    .to(tla.Int32, _TRAIT_I8_PP0)
                    .to(tla.Float32, _TRAIT_TO_F32_INT)
                )
                pp1_f32 = (
                    a_i.to(tla.Int8, _TRAIT_I8_PP1)
                    .to(tla.Int32, _TRAIT_I8_PP1)
                    .to(tla.Float32, _TRAIT_TO_F32_INT)
                )
                pp2_f32 = (
                    a_i.to(tla.Int8, _TRAIT_I8_PP2)
                    .to(tla.Int32, _TRAIT_I8_PP2)
                    .to(tla.Float32, _TRAIT_TO_F32_INT)
                )
                pp3_f32 = (
                    a_i.to(tla.Int8, _TRAIT_I8_PP3)
                    .to(tla.Int32, _TRAIT_I8_PP3)
                    .to(tla.Float32, _TRAIT_TO_F32_INT)
                )

                i2f_f16 = i16a.to(tla.Float16, _TRAIT_I16_F16)  # i16 -> f16
                i2f_f32 = i2f_f16.to(tla.Float32, _TRAIT_TO_F32_EVEN)

                # chain_f32 == i2f_f32 == pp{0..3}_f32 == float(trunc(a)). The net
                # is a single float(trunc(a)); the other routes appear as
                # zero-differences that become nonzero if any int cast is wrong
                # (e.g. a pp quarter reading back the wrong lanes).
                int_contrib = (
                    chain_f32
                    + (i2f_f32 - chain_f32)
                    + (pp0_f32 - chain_f32)
                    + (pp1_f32 - chain_f32)
                    + (pp2_f32 - chain_f32)
                    + (pp3_f32 - chain_f32)
                )

                # Compute *at* i16 and i8 width so the cast results are exercised by
                # real arithmetic, not just round-trips.
                #   i16 multiply : prod16 = trunc(a) * trunc(c), computed in i16
                #   i8  add      : sum8   = trunc(a) + trunc(b), computed in i8
                c_i = c_v.to(tla.Int32, _TRAIT_I32_TRUNC)
                b_i = b_v.to(tla.Int32, _TRAIT_I32_TRUNC)

                a_i16 = a_i.to(tla.Int16, _TRAIT_INT)  # i32 -> i16
                c_i16 = c_i.to(tla.Int16, _TRAIT_INT)
                prod16 = a_i16 * c_i16  # i16 multiply
                prod16_f32 = prod16.to(tla.Float32, _TRAIT_TO_F32_INT)  # i16->f32

                a_i8 = a_i.to(tla.Int8, _TRAIT_INT)  # i32 -> i8 (4x)
                b_i8 = b_i.to(tla.Int8, _TRAIT_INT)
                sum8 = a_i8 + b_i8  # i8 add
                sum8_i32 = sum8.to(tla.Int32, _TRAIT_INT)  # i8 -> i32 (4x)
                sum8_f32 = sum8_i32.to(tla.Float32, _TRAIT_TO_F32_INT)

                # fp8. Every native f32<->fp8 pp0..pp3 round trip must describe
                # the same values. The second route reaches f16 from fp8, which has no
                # instruction and is expanded by the lowering into fp8 -> f32
                # -> f16. Both describe e4m3(a), so their difference is zero
                # and becomes nonzero if the composition is wrong.
                a_f8 = a_v.to(tla.Float8E4M3FN, _TRAIT_FP8_PP0, full_mask)
                f8_rt_f32 = a_f8.to(tla.Float32, _TRAIT_FP8_PP0)
                f8_pp1_f32 = a_v.to(tla.Float8E4M3FN, _TRAIT_FP8_PP1, full_mask).to(
                    tla.Float32, _TRAIT_FP8_PP1
                )
                f8_pp2_f32 = a_v.to(tla.Float8E4M3FN, _TRAIT_FP8_PP2, full_mask).to(
                    tla.Float32, _TRAIT_FP8_PP2
                )
                f8_pp3_f32 = a_v.to(tla.Float8E4M3FN, _TRAIT_FP8_PP3, full_mask).to(
                    tla.Float32, _TRAIT_FP8_PP3
                )
                f8_via16_f32 = a_f8.to(tla.Float16, _TRAIT_FP8).to(
                    tla.Float32, _TRAIT_TO_F32_EVEN
                )
                # The other composed direction: f16 -> e4m3 becomes
                # f16 -> f32 -> e4m3. e4m3 is used rather than e5m2 so that the
                # inputs here (1.0 .. 6.0 in steps of 0.5) are all exactly
                # representable: e5m2 carries two mantissa bits, so its spacing
                # over [4, 8) is 1.0 and 4.5 lands exactly on a tie, where the
                # AVE round mode and torch's round-half-to-even disagree. This
                # test is about the composition, not about tie-breaking.
                f16_to_f8_f32 = a_h.to(tla.Float8E4M3FN, _TRAIT_FP8).to(
                    tla.Float32, _TRAIT_FP8
                )

                # e5m2, on c rather than a: c is 1.0, 1.5, 2.0, 2.5, 3.0, and
                # e5m2's spacing is 0.25 over [1, 2) and 0.5 over [2, 4), so all
                # of them are exact. a would not be -- 4.5 sits exactly on a tie
                # there, where the AVE round mode and torch's round-half-to-even
                # disagree.
                c_e5 = c_v.to(tla.Float8E5M2, _TRAIT_FP8_PP0, full_mask)
                e5_rt_f32 = c_e5.to(tla.Float32, _TRAIT_FP8_PP0)
                e5_pp1_f32 = c_v.to(tla.Float8E5M2, _TRAIT_FP8_PP1, full_mask).to(
                    tla.Float32, _TRAIT_FP8_PP1
                )
                e5_pp2_f32 = c_v.to(tla.Float8E5M2, _TRAIT_FP8_PP2, full_mask).to(
                    tla.Float32, _TRAIT_FP8_PP2
                )
                e5_pp3_f32 = c_v.to(tla.Float8E5M2, _TRAIT_FP8_PP3, full_mask).to(
                    tla.Float32, _TRAIT_FP8_PP3
                )
                e5_via16_f32 = c_e5.to(tla.Float16, _TRAIT_FP8).to(
                    tla.Float32, _TRAIT_TO_F32_EVEN
                )

                # bf16, both directions and both encodings. bf16 carries eight
                # mantissa bits, so it holds every value here exactly and the
                # round trips stay identities.
                f8_viabf_f32 = a_f8.to(tla.BFloat16, _TRAIT_FP8).to(
                    tla.Float32, _TRAIT_TO_F32_EVEN
                )
                c_bf = c_v.to(tla.BFloat16, _TRAIT_FP8)
                bf_to_e4m3_f32 = c_bf.to(tla.Float8E4M3FN, _TRAIT_FP8).to(
                    tla.Float32, _TRAIT_FP8
                )
                bf_to_e5m2_f32 = c_bf.to(tla.Float8E5M2, _TRAIT_FP8).to(
                    tla.Float32, _TRAIT_FP8
                )

                # The two "via" routes describe the same values as their round
                # trips, so they enter as zero-differences that become nonzero
                # the moment a composition is wrong.
                fp8_contrib = (
                    f8_rt_f32
                    + (f8_pp1_f32 - f8_rt_f32)
                    + (f8_pp2_f32 - f8_rt_f32)
                    + (f8_pp3_f32 - f8_rt_f32)
                    + (f8_via16_f32 - f8_rt_f32)
                    + (f8_viabf_f32 - f8_rt_f32)
                    + f16_to_f8_f32
                    + e5_rt_f32
                    + (e5_pp1_f32 - e5_rt_f32)
                    + (e5_pp2_f32 - e5_rt_f32)
                    + (e5_pp3_f32 - e5_rt_f32)
                    + (e5_via16_f32 - e5_rt_f32)
                    + bf_to_e4m3_f32
                    + bf_to_e5m2_f32
                )

                # Unsigned: (a+200) via f16↔u8 (values >=128 bite signed twin).
                u_contrib = (
                    (a_v + 200.0)
                    .to(tla.Float16, _TRAIT_F16_A)
                    .to(tla.UInt8, _TRAIT_U)
                    .to(tla.Float16, _TRAIT_U)
                    .to(tla.Float32, _TRAIT_TO_F32_EVEN)
                )

                # Same-type float casts: the element type survives and only the
                # rounding lands, so the rounding is directly observable
                # (1.5 -> 1.0): CAST_TRUNC on f32/bf16, CAST_FLOOR on f16.
                # a and c are exact in f16/bf16 too, so the
                # f16->f16 and bf16->bf16 legs describe the same trunc() values
                # as their f32 routes and enter as zero-differences that become
                # nonzero if the same-type cast is dropped, widened, or
                # mislaned.
                st_a_f32 = a_v.to(tla.Float32, _TRAIT_SAME_F32)
                st_c_f32 = c_v.to(tla.Float32, _TRAIT_SAME_F32)
                f16_leg = a_h.to(tla.Float16, _TRAIT_SAME_F16).to(
                    tla.Float32, _TRAIT_TO_F32_EVEN
                )
                bf16_leg = c_bf_e.to(tla.BFloat16, _TRAIT_SAME_BF16).to(
                    tla.Float32, _TRAIT_TO_F32_EVEN2
                )
                same_type_contrib = (
                    st_a_f32 + st_c_f32 + (f16_leg - st_a_f32) + (bf16_leg - st_c_f32)
                )

                out_chunk.store(
                    sum_f32
                    + (max_f32 - min_f32)
                    + prod_f32
                    + int_contrib
                    + prod16_f32
                    + sum8_f32
                    + fp8_contrib
                    + u_contrib
                    + same_type_contrib
                )

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(out_gm, out_ub)

        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any) -> Any:
    alignment = 512 if _KERNEL_ELEMENT_BYTES == 8 else 256
    ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, alignment)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(
        tensor,
        tla.make_shape(VL_ELE),
        tla.make_coord(chunk_idx),
    )


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "cast_multi": {
            "default_atol": 1e-2,
        },
    }


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    del op_name, dtype_name
    return False


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: unsupported case")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE
    if op_name not in _operator_specs():
        raise SystemExit("unknown cast-multi operator")
    del shape
    config = vector_kernel_config(dtype_name, (VECTOR_ELE,), ALL_DTYPES)
    _KERNEL_SHAPE = (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = NUM_CHUNKS
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, _, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    dtype = vector_kernel_config(dtype_name, args.shape, ALL_DTYPES).torch_dtype
    device = "npu"
    idx = torch.arange(VECTOR_ELE, dtype=torch.float32, device=device)
    # Small magnitudes in multiples of 0.25, so every value is exactly
    # representable in f16 AND bf16 (the reduced-precision casts are no-ops and
    # the reference is exact).
    return (
        ((idx % 11.0) * 0.5 + 1.0).to(dtype),  # a
        ((idx % 7.0) * 0.25 + 1.0).to(dtype),  # b
        ((idx % 5.0) * 0.5 + 1.0).to(dtype),  # c
        ((idx % 9.0) * 0.25 + 1.0).to(dtype),  # d
        ((idx % 3.0) + 1.0).to(dtype),  # e
    )


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    del op_name
    import torch

    a, b, c, d, e = inputs
    f32 = torch.float32
    # Each op computed in its cast dtype then combined in f32. The reduced-
    # precision (f16/bf16) casts are no-ops for these exact values; the float->int
    # casts apply trunc/floor.
    sum_f32 = (a.to(torch.float16) + b.to(torch.float16)).to(f32)
    c_bf = c.to(torch.bfloat16)
    d_bf = d.to(torch.bfloat16)
    max_f32 = torch.maximum(c_bf, d_bf).to(f32)
    min_f32 = torch.minimum(c_bf, d_bf).to(f32)
    prod_f32 = (a.to(torch.int32) * torch.floor(e).to(torch.int32)).to(f32)
    # integer narrow/widen chain + i16->f16, all equal float(trunc(a)); the
    # kernel combines them as (chain + direct) - i2f == one float(trunc(a)).
    int_contrib = a.to(torch.int32).to(f32)
    # compute at i16 / i8 width (small values, so int32 mirrors them exactly).
    # fp8: the e4m3 round trip, plus e4m3 applied to the f16 value. The
    # fp8 -> f16 route is the same value as the round trip, so it contributes
    # nothing here -- which is the point: any error in the composition shows up
    # as a nonzero difference.
    f8_rt = a.to(torch.float8_e4m3fn).to(f32)
    f16_to_f8 = a.to(torch.float16).to(torch.float8_e4m3fn).to(f32)
    # e5m2 rides on c, whose values are all exact there; bf16 holds everything
    # exactly. The fp8 -> f16 and fp8 -> bf16 routes equal their round trips and
    # so contribute nothing -- which is the point of subtracting them.
    e5_rt = c.to(torch.float8_e5m2).to(f32)
    c_bf = c.to(torch.bfloat16)
    bf_to_e4m3 = c_bf.to(torch.float8_e4m3fn).to(f32)
    bf_to_e5m2 = c_bf.to(torch.float8_e5m2).to(f32)
    fp8_contrib = f8_rt + f16_to_f8 + e5_rt + bf_to_e4m3 + bf_to_e5m2
    prod16_f32 = (a.to(torch.int32) * c.to(torch.int32)).to(f32)
    sum8_f32 = (a.to(torch.int32) + b.to(torch.int32)).to(f32)
    u_contrib = (a + 200.0).to(torch.float16).to(torch.uint8).to(torch.float16).to(f32)
    # same-type floats: f32->f32 trunc on a and c. The f16->f16 and bf16->bf16
    # legs re-round values that are already exact in their dtype, so each lands
    # on the same trunc() value as its f32 route -- zero-differences in the
    # kernel, nothing to add here. If a leg is dropped or mislaned the kernel
    # sum moves while this reference does not.
    same_type_contrib = torch.trunc(a) + torch.trunc(c)
    return (
        sum_f32
        + (max_f32 - min_f32)
        + prod_f32
        + int_contrib
        + prod16_f32
        + sum8_f32
        + fp8_contrib
        + u_contrib
        + same_type_contrib,
    )


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="tla.cast graph exercising every tla.params.CastParams value.",
        kernel=cast_multi,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: _KERNEL_SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=_is_unsupported_case,
        print_skip=_print_skip,
        script_path=Path(__file__).resolve(),
        float_dtypes=frozenset({"f32"}),
        output_count=1,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
