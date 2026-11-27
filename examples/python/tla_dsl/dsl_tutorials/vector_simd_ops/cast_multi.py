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
# combines them into one f32 output. The casts use these parameter values:
#
#   tla.params.RegSlot        : ZERO, ONE (2x part_even/odd); ZERO..THREE (4x pp0..pp3)
#   tla.params.SatMode        : NOSAT, SAT, UNKNOWN
#   tla.params.RoundMode      : CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC,
#                               CAST_NONE
#
# Coverage notes:
#   - On a float->float narrowing (f32->f16/bf16) the round_mode rounds to the
#     nearest representable target float (sub-ULP); for the exactly-representable
#     inputs here it is a no-op, so those round modes do not change the result.
#   - Integer rounding (trunc/floor) is observable on the float->int casts.
#   - Narrowing to FP8 uses NOSAT. Exact widening from FP8 uses UNKNOWN.
#
# The paths:
#   f16 add   : sum = (a as f16) + (b as f16)                     -> f32
#   bf16 max  : mx  = max(c as bf16, d as bf16)  (part_odd)       -> f32
#   bf16 min  : mn  = min(c as bf16, d as bf16)  (part_even)      -> f32
#   i32 mul   : pr  = (trunc(a) as i32) * (floor(e) as i32)       -> f32
#   int chain : ic  = 2 * float(trunc(a)) via i32->i16->i32 and i32->i16->f16
#   i16 mul   : p16 = trunc(a) * trunc(c), computed at i16 width -> f32
#   i8  add   : s8  = trunc(a) + trunc(b), computed at i8 width  -> f32
#   fp8       : Float32 round trips in all four byte slots, plus Float16
#               and BFloat16 conversions with both FP8 formats.
#   unsigned  : Float16 -> UInt8 -> Float16/Float32 for values above 127.
#   same-type : st  = trunc(a) + trunc(c) via f32->f32, with zero-difference
#               legs through f16->f16 and bf16->bf16. A same-type float cast
#               keeps the element type and only the round mode lands, so the
#               rounding is directly observable (1.5 -> 1.0): CAST_TRUNC on
#               f32/bf16, CAST_FLOOR on f16.
#
VECTOR_ELE = 256
VL_ELE = 64
NUM_CHUNKS = 4
ALL_DTYPES = ("f32",)


# --- fp8 path traits ---------------------------------------------------------
# fp8 pairs with f32 in hardware and uses all four pp0..pp3 byte positions.
def _fp8_pp_trait(
    layout: tla.params.RegSlot,
    sat_mode: tla.params.SatMode = tla.params.SatMode.UNKNOWN,
) -> tla.params.CastParams:
    return tla.params.CastParams(
        reg_slot=layout,
        sat_mode=sat_mode,
    )


_TRAIT_FP8_PP0 = _fp8_pp_trait(tla.params.RegSlot.ZERO)
_TRAIT_FP8_PP1 = _fp8_pp_trait(tla.params.RegSlot.ONE)
_TRAIT_FP8_PP2 = _fp8_pp_trait(tla.params.RegSlot.TWO)
_TRAIT_FP8_PP3 = _fp8_pp_trait(tla.params.RegSlot.THREE)

_TRAIT_TO_FP8_PP0 = _fp8_pp_trait(tla.params.RegSlot.ZERO, tla.params.SatMode.NOSAT)
_TRAIT_TO_FP8_PP1 = _fp8_pp_trait(tla.params.RegSlot.ONE, tla.params.SatMode.NOSAT)
_TRAIT_TO_FP8_PP2 = _fp8_pp_trait(tla.params.RegSlot.TWO, tla.params.SatMode.NOSAT)
_TRAIT_TO_FP8_PP3 = _fp8_pp_trait(tla.params.RegSlot.THREE, tla.params.SatMode.NOSAT)

# --- f16 path traits ---------------------------------------------------------
# a -> f16: even lane, no saturation, zeroing, round-to-nearest.
_TRAIT_F16_A = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_FLOOR,  # float->float: sub-ULP, no-op here
)
# b -> f16: even lane, no saturation, round toward zero.
_TRAIT_F16_B = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_TRUNC,  # float->float: sub-ULP, no-op here
)

# --- bf16 max path (ODD packing) ---------------------------------------------
# c, d -> bf16: odd lane, merging, ceil.
_TRAIT_BF16_ODD = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ONE,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_CEIL,
)
# max(c,d) -> f32: odd widen (must match the narrow's part_odd).
_TRAIT_TO_F32_ODD = tla.params.CastParams(reg_slot=tla.params.RegSlot.ONE)

# --- bf16 min path (EVEN packing) --------------------------------------------
# c, d -> bf16: even lane, zeroing, ceil.
_TRAIT_BF16_EVEN = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_CEIL,
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
    round_mode=tla.params.RoundMode.CAST_ROUND,
)

# --- integer-to-integer / integer-to-float chain traits ----------------------
# Slot ZERO selects the even lanes on the narrower side.
_TRAIT_INT = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
)
# i16 -> f16 (int -> float; exact for the small integers here).
_TRAIT_I16_F16 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)


# The unsigned round trip uses values above the signed 8-bit range.
_TRAIT_U = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_TRUNC,
)
_TRAIT_EXACT_WIDEN = tla.params.CastParams()

# --- same-type float traits ---------------------------------------------------
# f16->f16 / bf16->bf16 / f32->f32 keep the element type and re-round every
# lane, so a round mode is directly observable (1.5 -> 1.0): CAST_TRUNC on
# f32/bf16 and CAST_FLOOR on f16. Use ZERO or ONE with SatMode.UNKNOWN.
_TRAIT_SAME_F32 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.UNKNOWN,
    round_mode=tla.params.RoundMode.CAST_TRUNC,
)
_TRAIT_SAME_F16 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    round_mode=tla.params.RoundMode.CAST_FLOOR,
)
_TRAIT_SAME_BF16 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ONE,
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
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

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

                # f16 add (a rounds toward negative infinity, b toward zero).
                a_h = a_v.to(tla.Float16, _TRAIT_F16_A, full_mask)
                b_h = b_v.to(tla.Float16, _TRAIT_F16_B, full_mask)
                sum_f32 = (a_h + b_h).to(tla.Float32, _TRAIT_EXACT_WIDEN)

                # bf16 max, part_odd (merging).
                c_bf_o = c_v.to(tla.BFloat16, _TRAIT_BF16_ODD, full_mask)
                d_bf_o = d_v.to(tla.BFloat16, _TRAIT_BF16_ODD, full_mask)
                max_f32 = tla.max(c_bf_o, d_bf_o).to(tla.Float32, _TRAIT_TO_F32_ODD)

                # bf16 min, part_even (zeroing).
                c_bf_e = c_v.to(tla.BFloat16, _TRAIT_BF16_EVEN, full_mask)
                d_bf_e = d_v.to(tla.BFloat16, _TRAIT_BF16_EVEN, full_mask)
                min_f32 = tla.min(c_bf_e, d_bf_e).to(tla.Float32, _TRAIT_EXACT_WIDEN)

                # i32 mul (a trunc, e floor, saturating).
                a_i = a_v.to(tla.Int32, _TRAIT_I32_TRUNC, full_mask)
                e_i = e_v.to(tla.Int32, _TRAIT_I32_FLOOR, full_mask)
                prod_f32 = (a_i * e_i).to(tla.Float32, _TRAIT_TO_F32_INT)

                # With matching slots, both paths recover the small integers.
                a_i16 = a_i.to(tla.Int16, _TRAIT_INT)
                chain_f32 = a_i16.to(tla.Int32, _TRAIT_EXACT_WIDEN).to(
                    tla.Float32, _TRAIT_TO_F32_INT
                )
                i2f_f32 = a_i16.to(tla.Float16, _TRAIT_I16_F16).to(
                    tla.Float32, _TRAIT_EXACT_WIDEN
                )
                int_contrib = chain_f32 + i2f_f32

                # Compute *at* i16 and i8 width so the cast results are exercised by
                # real arithmetic, not just round-trips.
                #   i16 multiply : prod16 = trunc(a) * trunc(c), computed in i16
                #   i8  add      : sum8   = trunc(a) + trunc(b), computed in i8
                c_i = c_v.to(tla.Int32, _TRAIT_I32_TRUNC)
                b_i = b_v.to(tla.Int32, _TRAIT_I32_TRUNC)

                c_i16 = c_i.to(tla.Int16, _TRAIT_INT)
                prod16 = a_i16 * c_i16  # i16 multiply
                prod16_f32 = prod16.to(tla.Float32, _TRAIT_EXACT_WIDEN)

                # Two even-slot narrows put each original lane in byte 0 of
                # its 4-byte group. The widen reads that same byte.
                a_i8 = a_i16.to(tla.Int8, _TRAIT_INT)
                b_i8 = b_i.to(tla.Int16, _TRAIT_INT).to(tla.Int8, _TRAIT_INT)
                sum8 = a_i8 + b_i8  # i8 add
                sum8_i32 = sum8.to(tla.Int32, _TRAIT_EXACT_WIDEN)  # i8 -> i32 (4x)
                sum8_f32 = sum8_i32.to(tla.Float32, _TRAIT_TO_F32_INT)

                # Each Float32 round trip reads the byte slot it wrote.
                a_f8 = a_v.to(tla.Float8E4M3FN, _TRAIT_TO_FP8_PP0, full_mask)
                f8_rt_f32 = a_f8.to(tla.Float32, _TRAIT_FP8_PP0)
                f8_pp1_f32 = a_v.to(tla.Float8E4M3FN, _TRAIT_TO_FP8_PP1, full_mask).to(
                    tla.Float32, _TRAIT_FP8_PP1
                )
                f8_pp2_f32 = a_v.to(tla.Float8E4M3FN, _TRAIT_TO_FP8_PP2, full_mask).to(
                    tla.Float32, _TRAIT_FP8_PP2
                )
                f8_pp3_f32 = a_v.to(tla.Float8E4M3FN, _TRAIT_TO_FP8_PP3, full_mask).to(
                    tla.Float32, _TRAIT_FP8_PP3
                )

                # The c values are exactly representable in E5M2, so these
                # round trips exercise lane placement with exact inputs.
                c_e5 = c_v.to(tla.Float8E5M2, _TRAIT_TO_FP8_PP0, full_mask)
                e5_rt_f32 = c_e5.to(tla.Float32, _TRAIT_FP8_PP0)
                e5_pp1_f32 = c_v.to(tla.Float8E5M2, _TRAIT_TO_FP8_PP1, full_mask).to(
                    tla.Float32, _TRAIT_FP8_PP1
                )
                e5_pp2_f32 = c_v.to(tla.Float8E5M2, _TRAIT_TO_FP8_PP2, full_mask).to(
                    tla.Float32, _TRAIT_FP8_PP2
                )
                e5_pp3_f32 = c_v.to(tla.Float8E5M2, _TRAIT_TO_FP8_PP3, full_mask).to(
                    tla.Float32, _TRAIT_FP8_PP3
                )
                c_h = c_e5.to(tla.Float16, _TRAIT_EXACT_WIDEN)
                e5_via16_f32 = c_h.to(tla.Float32, _TRAIT_EXACT_WIDEN)
                f8_via16_f32 = a_f8.to(tla.Float16, _TRAIT_FP8_PP0).to(
                    tla.Float32, _TRAIT_EXACT_WIDEN
                )
                f8_viabf_f32 = a_f8.to(tla.BFloat16, _TRAIT_FP8_PP0).to(
                    tla.Float32, _TRAIT_EXACT_WIDEN
                )
                e5_viabf_f32 = c_e5.to(tla.BFloat16, _TRAIT_FP8_PP0).to(
                    tla.Float32, _TRAIT_EXACT_WIDEN
                )

                # These 16-bit inputs use even (Float16) and odd (BFloat16)
                # lanes. M3 selects every third source lane before narrowing.
                mask16 = tla.create_mask(pattern=tla.mask.M3, dtype=tla.Float16)
                half_e4 = a_h.to(tla.Float8E4M3FN, _TRAIT_TO_FP8_PP0, mask16).to(
                    tla.Float32, _TRAIT_FP8_PP0
                )
                half_e5 = c_h.to(tla.Float8E5M2, _TRAIT_TO_FP8_PP0, mask16).to(
                    tla.Float32, _TRAIT_FP8_PP0
                )
                bf16_e4 = c_bf_o.to(tla.Float8E4M3FN, _TRAIT_TO_FP8_PP1, mask16).to(
                    tla.Float32, _TRAIT_FP8_PP3
                )
                bf16_e5 = c_bf_o.to(tla.Float8E5M2, _TRAIT_TO_FP8_PP1, mask16).to(
                    tla.Float32, _TRAIT_FP8_PP3
                )

                # Every slot yields the same values. Widening through
                # Float16 or BFloat16 also preserves these FP8 values.
                fp8_contrib = (
                    f8_rt_f32
                    + (f8_pp1_f32 - f8_rt_f32)
                    + (f8_pp2_f32 - f8_rt_f32)
                    + (f8_pp3_f32 - f8_rt_f32)
                    + e5_rt_f32
                    + (e5_pp1_f32 - e5_rt_f32)
                    + (e5_pp2_f32 - e5_rt_f32)
                    + (e5_pp3_f32 - e5_rt_f32)
                    + (e5_via16_f32 - e5_rt_f32)
                    + f8_via16_f32
                    + f8_viabf_f32
                    + e5_viabf_f32
                    + half_e4
                    + half_e5
                    + bf16_e4
                    + bf16_e5
                )

                # Use values above 127 to check that both conversions zero-extend.
                a_shifted_h = (a_v + 200.0).to(tla.Float16, _TRAIT_F16_A)
                a_u8 = a_shifted_h.to(tla.UInt8, _TRAIT_U)
                u_via16 = a_u8.to(tla.Float16, _TRAIT_EXACT_WIDEN).to(
                    tla.Float32, _TRAIT_EXACT_WIDEN
                )
                u_contrib = a_u8.to(tla.Float32, _TRAIT_EXACT_WIDEN) + u_via16

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
                    tla.Float32, _TRAIT_EXACT_WIDEN
                )
                bf16_leg = c_bf_e.to(tla.BFloat16, _TRAIT_SAME_BF16).to(
                    tla.Float32, _TRAIT_EXACT_WIDEN
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
    ptr = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
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
    if op_name not in _operator_specs():
        raise SystemExit("unknown cast-multi operator")
    del shape
    config = vector_kernel_config(dtype_name, (VECTOR_ELE,), ALL_DTYPES)
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
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
    int_contrib = 2 * a.to(torch.int32).to(f32)
    f8_rt = a.to(torch.float8_e4m3fn).to(f32)
    e5_rt = c.to(torch.float8_e5m2).to(f32)
    lane = torch.arange(VECTOR_ELE, device=a.device) % VL_ELE
    half_fp8 = torch.where(2 * lane % 3 == 0, f8_rt + e5_rt, 0)
    c_e4 = c.to(torch.float8_e4m3fn).to(f32)
    bf16_fp8 = torch.where((2 * lane + 1) % 3 == 0, c_e4 + e5_rt, 0)
    fp8_contrib = 3 * f8_rt + 2 * e5_rt + half_fp8 + bf16_fp8
    prod16_f32 = (a.to(torch.int32) * c.to(torch.int32)).to(f32)
    sum8_f32 = (a.to(torch.int32) + b.to(torch.int32)).to(f32)
    u_contrib = 2 * (a + 200.0).to(torch.float16).to(torch.uint8).to(f32)
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
        description="Type conversion across multiple register widths and slots.",
        kernel=cast_multi,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: (VECTOR_ELE,),
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
