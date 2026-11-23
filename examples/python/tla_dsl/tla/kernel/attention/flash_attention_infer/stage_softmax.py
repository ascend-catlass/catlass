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
import catlass.tla as tla
from fa_common import VL_F16, VL_F32
from catlass.params import MaskStoreParams
from catlass.params import MaskLoadParams

MIN_VALUE = -3.0e38


@tla.jit
def _mask_sf_first(
    ctx, n, fineMaskBit, tailN, m, ub_s, lastMaxAddr, ub_mask, lastSumAddr, ub_p_zN
):
    """softmax 首(mask)。"""
    c0 = 0
    c1 = 1
    if n > 64:
        if fineMaskBit:
            with tla.vec.func(mode="simd"):
                pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
                cast_trait_zero = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    sat_mode=tla.params.SatMode.SAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                )
                cast_trait_one = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ONE,
                    sat_mode=tla.params.SatMode.SAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                )
                minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
                for i in tla.range(m):
                    ub_s_i0 = tla.tile_view(
                        ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                    )
                    ub_s_i1 = tla.tile_view(
                        ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c1)
                    )
                    ub_last_max_i = tla.tile_view(
                        lastMaxAddr, tla.make_shape(1), tla.make_coord(i)
                    )
                    ub_mask_i0 = tla.tile_view(
                        ub_mask, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                    )
                    ub_mask_i1 = tla.tile_view(
                        ub_mask, tla.make_shape(1, VL_F32), tla.make_coord(i, c1)
                    )
                    ub_s_reg0 = ub_s_i0.load()
                    ub_s_reg1 = ub_s_i1.load()
                    mask_reg00 = ub_mask_i0.load(MaskLoadParams())
                    mask_reg11 = ub_mask_i1.load(MaskLoadParams())
                    ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                    ub_s_reg1 = tla.mul(ub_s_reg1, ctx.scaleValue_, mask=pregFull)
                    ub_s_reg0 = tla.where(mask_reg00, ub_s_reg0, minVreg)
                    ub_s_reg1 = tla.where(mask_reg11, ub_s_reg1, minVreg)
                    ub_s_i0.store(ub_s_reg0, mask=pregFull)
                    ub_s_i1.store(ub_s_reg1, mask=pregFull)
                    max_tmp_reg = tla.max(ub_s_reg0, ub_s_reg1, mask=pregFull)
                    max_reg = max_tmp_reg.reduce(tla.ReductionOp.MAX, mask=pregFull)
                    ub_last_max_i.store(max_reg, tla.params.UnalignStoreParams())
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
                for j in tla.range(m):
                    ub_last_max_iDe = tla.tile_view(
                        lastMaxAddr, tla.make_shape(1), tla.make_coord(j)
                    )
                    ub_last_sum_iDe = tla.tile_view(
                        lastSumAddr, tla.make_shape(1), tla.make_coord(j)
                    )
                    ub_s_i0De = tla.tile_view(
                        ub_s, tla.make_shape(1, VL_F32), tla.make_coord(j, c0)
                    )
                    ub_s_i1De = tla.tile_view(
                        ub_s, tla.make_shape(1, VL_F32), tla.make_coord(j, c1)
                    )
                    ub_p_zN_f16_i = tla.tile_view(
                        ub_p_zN, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
                    )
                    max_regDe = ub_last_max_iDe.load(
                        params=tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                        )
                    )
                    ub_s_reg0De = ub_s_i0De.load()
                    ub_s_reg1De = ub_s_i1De.load()
                    ub_s_odd_reg = tla.exp(
                        tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull
                    )
                    ub_s_even_reg = tla.exp(
                        tla.sub(ub_s_reg1De, max_regDe, mask=pregFull), mask=pregFull
                    )
                    exp_odd_reg0, exp_even_reg1 = tla.deinterleave(
                        ub_s_odd_reg, ub_s_even_reg
                    )
                    exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                    exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                    ub_last_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams())
                    exp_dst_reg0 = exp_even_reg1.to(
                        ctx.DTYPE_P, cast_trait_one, mask=pregFull
                    )
                    exp_dst_reg1 = exp_odd_reg0.to(
                        ctx.DTYPE_P, cast_trait_zero, mask=pregFull
                    )
                    exp_dst_reg = tla.bitwise_or(
                        exp_dst_reg0, exp_dst_reg1, mask=preg_all_b16
                    )
                    ub_p_zN_f16_i.store(
                        exp_dst_reg,
                        params=tla.params.BlockStoreParams(block_stride=65),
                        mask=preg_all_b16,
                    )
        else:
            with tla.vec.func(mode="simd"):
                pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
                cast_trait_zero = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    sat_mode=tla.params.SatMode.SAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                )
                cast_trait_one = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ONE,
                    sat_mode=tla.params.SatMode.SAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                )
                minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
                for i in tla.range(m):
                    ub_s_i0 = tla.tile_view(
                        ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                    )
                    ub_s_i1 = tla.tile_view(
                        ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c1)
                    )
                    ub_last_max_i = tla.tile_view(
                        lastMaxAddr, tla.make_shape(1), tla.make_coord(i)
                    )
                    ub_s_reg0 = ub_s_i0.load()
                    ub_s_reg1 = ub_s_i1.load()
                    ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                    ub_s_reg1 = tla.mul(ub_s_reg1, ctx.scaleValue_, mask=pregFull)
                    ub_s_reg1 = tla.where(pregTailN, ub_s_reg1, minVreg)
                    ub_s_i0.store(ub_s_reg0, mask=pregFull)
                    ub_s_i1.store(ub_s_reg1, mask=pregFull)
                    max_tmp_reg = tla.max(ub_s_reg0, ub_s_reg1, mask=pregFull)
                    max_reg = max_tmp_reg.reduce(tla.ReductionOp.MAX, mask=pregFull)
                    ub_last_max_i.store(max_reg, tla.params.UnalignStoreParams())
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
                for j in tla.range(m):
                    ub_last_max_iDe = tla.tile_view(
                        lastMaxAddr, tla.make_shape(1), tla.make_coord(j)
                    )
                    ub_last_sum_iDe = tla.tile_view(
                        lastSumAddr, tla.make_shape(1), tla.make_coord(j)
                    )
                    ub_s_i0De = tla.tile_view(
                        ub_s, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
                    )
                    ub_p_zN_f16_i = tla.tile_view(
                        ub_p_zN, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
                    )
                    max_regDe = ub_last_max_iDe.load(
                        params=tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                        )
                    )
                    ub_s_odd_reg, ub_s_even_reg = ub_s_i0De.load(
                        params=tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                        )
                    )
                    exp_odd_reg0 = tla.exp(
                        tla.sub(ub_s_odd_reg, max_regDe, mask=pregFull), mask=pregFull
                    )
                    exp_even_reg1 = tla.exp(
                        tla.sub(ub_s_even_reg, max_regDe, mask=pregFull), mask=pregFull
                    )
                    exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                    exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                    ub_last_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams())
                    exp_dst_reg0 = exp_even_reg1.to(
                        ctx.DTYPE_P, cast_trait_one, mask=pregFull
                    )
                    exp_dst_reg1 = exp_odd_reg0.to(
                        ctx.DTYPE_P, cast_trait_zero, mask=pregFull
                    )
                    exp_dst_reg = tla.bitwise_or(
                        exp_dst_reg0, exp_dst_reg1, mask=preg_all_b16
                    )
                    ub_p_zN_f16_i.store(
                        exp_dst_reg,
                        params=tla.params.BlockStoreParams(block_stride=65),
                        mask=preg_all_b16,
                    )
    else:
        if fineMaskBit:
            with tla.vec.func(mode="simd"):
                pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
                cast_trait_zero = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    sat_mode=tla.params.SatMode.SAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                )
                cast_trait_one = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ONE,
                    sat_mode=tla.params.SatMode.SAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                )
                minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
                for i in tla.range(m):
                    ub_s_i0 = tla.tile_view(
                        ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                    )
                    ub_last_max_i = tla.tile_view(
                        lastMaxAddr, tla.make_shape(1), tla.make_coord(i)
                    )
                    ub_mask_i0 = tla.tile_view(
                        ub_mask, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                    )
                    ub_s_reg0 = ub_s_i0.load()
                    mask_reg00 = ub_mask_i0.load(MaskLoadParams())
                    ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                    ub_s_reg0 = tla.where(mask_reg00, ub_s_reg0, minVreg)
                    ub_s_i0.store(ub_s_reg0, mask=pregFull)
                    max_reg = ub_s_reg0.reduce(tla.ReductionOp.MAX, mask=pregFull)
                    ub_last_max_i.store(max_reg, tla.params.UnalignStoreParams())
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
                for j in tla.range(m):
                    ub_last_max_iDe = tla.tile_view(
                        lastMaxAddr, tla.make_shape(1), tla.make_coord(j)
                    )
                    ub_last_sum_iDe = tla.tile_view(
                        lastSumAddr, tla.make_shape(1), tla.make_coord(j)
                    )
                    ub_s_i0De = tla.tile_view(
                        ub_s, tla.make_shape(1, VL_F32), tla.make_coord(j, c0)
                    )
                    ub_p_zN_f16_i = tla.tile_view(
                        ub_p_zN, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
                    )
                    max_regDe = ub_last_max_iDe.load(
                        params=tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                        )
                    )
                    ub_s_reg0De = ub_s_i0De.load()
                    exp_reg0 = tla.exp(
                        tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull
                    )
                    exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                    ub_last_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams())
                    exp_dst_reg0 = exp_reg0.to(
                        ctx.DTYPE_P, cast_trait_zero, mask=pregFull
                    )
                    exp_dst_reg, zero_reg = tla.deinterleave(exp_dst_reg0, exp_dst_reg0)
                    ub_p_zN_f16_i.store(
                        exp_dst_reg,
                        params=tla.params.BlockStoreParams(block_stride=65),
                        mask=preg_all_b16,
                    )
        else:
            with tla.vec.func(mode="simd"):
                pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
                cast_trait_zero = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    sat_mode=tla.params.SatMode.SAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                )
                cast_trait_one = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ONE,
                    sat_mode=tla.params.SatMode.SAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                )
                minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
                for i in tla.range(m):
                    ub_s_i0 = tla.tile_view(
                        ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                    )
                    ub_last_max_i = tla.tile_view(
                        lastMaxAddr, tla.make_shape(1), tla.make_coord(i)
                    )
                    ub_s_reg0 = ub_s_i0.load()
                    ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                    ub_s_reg0 = tla.where(pregTailN, ub_s_reg0, minVreg)
                    ub_s_i0.store(ub_s_reg0, mask=pregFull)
                    max_reg = ub_s_reg0.reduce(tla.ReductionOp.MAX, mask=pregFull)
                    ub_last_max_i.store(max_reg, tla.params.UnalignStoreParams())
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
                for j in tla.range(m):
                    ub_last_max_iDe = tla.tile_view(
                        lastMaxAddr, tla.make_shape(1), tla.make_coord(j)
                    )
                    ub_last_sum_iDe = tla.tile_view(
                        lastSumAddr, tla.make_shape(1), tla.make_coord(j)
                    )
                    ub_s_i0De = tla.tile_view(
                        ub_s, tla.make_shape(1, VL_F32), tla.make_coord(j, c0)
                    )
                    ub_p_zN_f16_i = tla.tile_view(
                        ub_p_zN, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
                    )
                    max_regDe = ub_last_max_iDe.load(
                        params=tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                        )
                    )
                    ub_s_reg0De = ub_s_i0De.load()
                    exp_reg0 = tla.exp(
                        tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull
                    )
                    exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                    ub_last_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams())
                    exp_dst_reg0 = exp_reg0.to(
                        ctx.DTYPE_P, cast_trait_zero, mask=pregFull
                    )
                    exp_dst_reg, zero_reg = tla.deinterleave(exp_dst_reg0, exp_dst_reg0)
                    ub_p_zN_f16_i.store(
                        exp_dst_reg,
                        params=tla.params.BlockStoreParams(block_stride=65),
                        mask=preg_all_b16,
                    )


@tla.jit
def _mask_sf_mid_n64(
    ctx,
    fineMaskBit,
    tailN,
    m,
    lastMaxAddr,
    nowMaxAddr,
    lastSumAddr,
    expMaxUbAddr,
    nowSumAddr,
    ub_s,
    ub_mask,
    ub_p_zN,
):
    """softmax 中/末 n>64(mask)。"""
    c0 = 0
    c1 = 1
    if fineMaskBit:
        with tla.vec.func(mode="simd"):
            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
            cast_trait_zero = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            cast_trait_one = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ONE,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
            for i in tla.range(m):
                ub_s_i0 = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                )
                ub_s_i1 = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c1)
                )
                ub_now_max_i = tla.tile_view(
                    nowMaxAddr, tla.make_shape(1), tla.make_coord(i)
                )
                ub_mask_i0 = tla.tile_view(
                    ub_mask, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                )
                ub_mask_i1 = tla.tile_view(
                    ub_mask, tla.make_shape(1, VL_F32), tla.make_coord(i, c1)
                )
                ub_s_reg0 = ub_s_i0.load()
                ub_s_reg1 = ub_s_i1.load()
                mask_reg00 = ub_mask_i0.load(MaskLoadParams())
                mask_reg11 = ub_mask_i1.load(MaskLoadParams())
                ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                ub_s_reg1 = tla.mul(ub_s_reg1, ctx.scaleValue_, mask=pregFull)
                ub_s_reg0 = tla.where(mask_reg00, ub_s_reg0, minVreg)
                ub_s_reg1 = tla.where(mask_reg11, ub_s_reg1, minVreg)
                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                max_tmp_reg = tla.max(ub_s_reg0, ub_s_reg1, mask=pregFull)
                max_reg = max_tmp_reg.reduce(tla.ReductionOp.MAX, mask=pregFull)
                ub_now_max_i.store(max_reg, tla.params.UnalignStoreParams())
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            ub_last_max_i_de = tla.tile_view(
                lastMaxAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            ub_now_max_i_de = tla.tile_view(
                nowMaxAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            ub_last_sum_i_de = tla.tile_view(
                lastSumAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            ub_dm_i_de = tla.tile_view(
                expMaxUbAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            now_max_reg_de = ub_now_max_i_de.load()
            last_max_reg_de = ub_last_max_i_de.load()
            last_sum_reg = ub_last_sum_i_de.load()
            max_reg_de = tla.max(now_max_reg_de, last_max_reg_de, mask=pregFull)
            exp_sub_max_reg = tla.exp(
                tla.sub(last_max_reg_de, max_reg_de, mask=pregFull), mask=pregFull
            )
            update_exp_sub_reg = tla.mul(exp_sub_max_reg, last_sum_reg, mask=pregFull)
            ub_now_max_i_de.store(max_reg_de, mask=pregFull)
            ub_last_max_i_de.store(max_reg_de, mask=pregFull)
            ub_dm_i_de.store(exp_sub_max_reg, mask=pregFull)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            for j in tla.range(m):
                ub_now_max_iDe = tla.tile_view(
                    nowMaxAddr, tla.make_shape(1), tla.make_coord(j)
                )
                ub_now_sum_iDe = tla.tile_view(
                    nowSumAddr, tla.make_shape(1), tla.make_coord(j)
                )
                ub_s_i0De = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
                )
                ub_s_i1De = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(j, c1)
                )
                ub_p_zN_f16_i = tla.tile_view(
                    ub_p_zN, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
                )
                max_regDe = ub_now_max_iDe.load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                ub_s_reg0De = ub_s_i0De.load()
                ub_s_reg1De = ub_s_i1De.load()
                ub_s_odd_reg = tla.exp(
                    tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull
                )
                ub_s_even_reg = tla.exp(
                    tla.sub(ub_s_reg1De, max_regDe, mask=pregFull), mask=pregFull
                )
                exp_odd_reg0, exp_even_reg1 = tla.deinterleave(
                    ub_s_odd_reg, ub_s_even_reg
                )
                exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                ub_now_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams())
                exp_dst_reg0 = exp_even_reg1.to(
                    ctx.DTYPE_P, cast_trait_one, mask=pregFull
                )
                exp_dst_reg1 = exp_odd_reg0.to(
                    ctx.DTYPE_P, cast_trait_zero, mask=pregFull
                )
                exp_dst_reg = tla.bitwise_or(
                    exp_dst_reg0, exp_dst_reg1, mask=preg_all_b16
                )
                ub_p_zN_f16_i.store(
                    exp_dst_reg,
                    params=tla.params.BlockStoreParams(block_stride=65),
                    mask=preg_all_b16,
                )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            ub_now_sum_i_de = tla.tile_view(
                nowSumAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            now_sum_reg = ub_now_sum_i_de.load()
            update_exp_sub_reg = tla.add(update_exp_sub_reg, now_sum_reg, mask=pregFull)
            ub_last_sum_i_de.store(update_exp_sub_reg, mask=pregFull)
    else:
        with tla.vec.func(mode="simd"):
            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
            cast_trait_zero = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            cast_trait_one = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ONE,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
            for i in tla.range(m):
                ub_s_i0 = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                )
                ub_s_i1 = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c1)
                )
                ub_now_max_i = tla.tile_view(
                    nowMaxAddr, tla.make_shape(1), tla.make_coord(i)
                )
                ub_s_reg0 = ub_s_i0.load()
                ub_s_reg1 = ub_s_i1.load()
                ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                ub_s_reg1 = tla.mul(ub_s_reg1, ctx.scaleValue_, mask=pregTailN)
                ub_s_reg1 = tla.where(pregTailN, ub_s_reg1, minVreg)
                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                max_tmp_reg = tla.max(ub_s_reg0, ub_s_reg1, mask=pregFull)
                max_reg = max_tmp_reg.reduce(tla.ReductionOp.MAX, mask=pregFull)
                ub_now_max_i.store(max_reg, tla.params.UnalignStoreParams())
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            ub_last_max_i_de = tla.tile_view(
                lastMaxAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            ub_now_max_i_de = tla.tile_view(
                nowMaxAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            ub_last_sum_i_de = tla.tile_view(
                lastSumAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            ub_dm_i_de = tla.tile_view(
                expMaxUbAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            now_max_reg_de = ub_now_max_i_de.load()
            last_max_reg_de = ub_last_max_i_de.load()
            last_sum_reg = ub_last_sum_i_de.load()
            max_reg_de = tla.max(now_max_reg_de, last_max_reg_de, mask=pregFull)
            exp_sub_max_reg = tla.exp(
                tla.sub(last_max_reg_de, max_reg_de, mask=pregFull), mask=pregFull
            )
            update_exp_sub_reg = tla.mul(exp_sub_max_reg, last_sum_reg, mask=pregFull)
            ub_now_max_i_de.store(max_reg_de, mask=pregFull)
            ub_last_max_i_de.store(max_reg_de, mask=pregFull)
            ub_dm_i_de.store(exp_sub_max_reg, mask=pregFull)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            for j in tla.range(m):
                ub_now_max_iDe = tla.tile_view(
                    nowMaxAddr, tla.make_shape(1), tla.make_coord(j)
                )
                ub_now_sum_iDe = tla.tile_view(
                    nowSumAddr, tla.make_shape(1), tla.make_coord(j)
                )
                ub_s_i0De = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(j, c0)
                )
                ub_p_zN_f16_i = tla.tile_view(
                    ub_p_zN, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
                )
                max_regDe = ub_now_max_iDe.load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                ub_s_odd_reg, ub_s_even_reg = ub_s_i0De.load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                    )
                )
                exp_odd_reg0 = tla.exp(
                    tla.sub(ub_s_odd_reg, max_regDe, mask=pregFull), mask=pregFull
                )
                exp_even_reg1 = tla.exp(
                    tla.sub(ub_s_even_reg, max_regDe, mask=pregFull), mask=pregFull
                )
                exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                ub_now_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams())
                exp_dst_reg0 = exp_even_reg1.to(
                    ctx.DTYPE_P, cast_trait_one, mask=pregFull
                )
                exp_dst_reg1 = exp_odd_reg0.to(
                    ctx.DTYPE_P, cast_trait_zero, mask=pregFull
                )
                exp_dst_reg = tla.bitwise_or(
                    exp_dst_reg0, exp_dst_reg1, mask=preg_all_b16
                )
                ub_p_zN_f16_i.store(
                    exp_dst_reg,
                    params=tla.params.BlockStoreParams(block_stride=65),
                    mask=preg_all_b16,
                )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            ub_now_sum_i_de = tla.tile_view(
                nowSumAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            now_sum_reg = ub_now_sum_i_de.load()
            update_exp_sub_reg = tla.add(update_exp_sub_reg, now_sum_reg, mask=pregFull)
            ub_last_sum_i_de.store(update_exp_sub_reg, mask=pregFull)


@tla.jit
def _mask_sf_mid_nlt64(
    ctx,
    fineMaskBit,
    tailN,
    kvSStartIdx,
    m,
    lastMaxAddr,
    nowMaxAddr,
    lastSumAddr,
    expMaxUbAddr,
    nowSumAddr,
    ub_s,
    ub_mask,
    ub_p_zN,
):
    """softmax 中/末 n<=64(mask)。"""
    c0 = 0
    if fineMaskBit:
        with tla.vec.func(mode="simd"):
            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
            cast_trait_zero = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
            for i in tla.range(m):
                ub_s_i0 = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                )
                ub_now_max_i = tla.tile_view(
                    nowMaxAddr, tla.make_shape(1), tla.make_coord(i)
                )
                ub_mask_i0 = tla.tile_view(
                    ub_mask, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                )
                ub_s_reg0 = ub_s_i0.load()
                mask_reg00 = ub_mask_i0.load(MaskLoadParams())
                ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                ub_s_reg0 = tla.where(mask_reg00, ub_s_reg0, minVreg)
                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                max_reg = ub_s_reg0.reduce(tla.ReductionOp.MAX, mask=pregFull)
                ub_now_max_i.store(max_reg, tla.params.UnalignStoreParams())
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            ub_last_max_i_de = tla.tile_view(
                lastMaxAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            ub_now_max_i_de = tla.tile_view(
                nowMaxAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            ub_last_sum_i_de = tla.tile_view(
                lastSumAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            ub_dm_i_de = tla.tile_view(
                expMaxUbAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            now_max_reg_de = ub_now_max_i_de.load()
            last_max_reg_de = ub_last_max_i_de.load()
            last_sum_reg = ub_last_sum_i_de.load()
            max_reg_de = tla.max(now_max_reg_de, last_max_reg_de, mask=pregFull)
            exp_sub_max_reg = tla.exp(
                tla.sub(last_max_reg_de, max_reg_de, mask=pregFull), mask=pregFull
            )
            update_exp_sub_reg = tla.mul(exp_sub_max_reg, last_sum_reg, mask=pregFull)
            ub_now_max_i_de.store(max_reg_de, mask=pregFull)
            ub_last_max_i_de.store(max_reg_de, mask=pregFull)
            ub_dm_i_de.store(exp_sub_max_reg, mask=pregFull)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            for j in tla.range(m):
                ub_now_max_iDe = tla.tile_view(
                    nowMaxAddr, tla.make_shape(1), tla.make_coord(j)
                )
                ub_now_sum_iDe = tla.tile_view(
                    nowSumAddr, tla.make_shape(1), tla.make_coord(j)
                )
                ub_s_i0De = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(j, c0)
                )
                ub_p_zN_f16_i = tla.tile_view(
                    ub_p_zN, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
                )
                max_regDe = ub_now_max_iDe.load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                ub_s_reg0De = ub_s_i0De.load()
                exp_reg0 = tla.exp(
                    tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull
                )
                exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                ub_now_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams())
                exp_dst_reg0 = exp_reg0.to(ctx.DTYPE_P, cast_trait_zero, mask=pregFull)
                exp_dst_reg, zero_reg = tla.deinterleave(exp_dst_reg0, exp_dst_reg0)
                ub_p_zN_f16_i.store(
                    exp_dst_reg,
                    params=tla.params.BlockStoreParams(block_stride=65),
                    mask=preg_all_b16,
                )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            ub_now_sum_i_de = tla.tile_view(
                nowSumAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            now_sum_reg = ub_now_sum_i_de.load()
            update_exp_sub_reg = tla.add(update_exp_sub_reg, now_sum_reg, mask=pregFull)
            ub_last_sum_i_de.store(update_exp_sub_reg, mask=pregFull)
    else:
        with tla.vec.func(mode="simd"):
            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
            cast_trait_zero = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
            for i in tla.range(m):
                ub_s_i0 = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                )
                ub_now_max_i = tla.tile_view(
                    nowMaxAddr, tla.make_shape(1), tla.make_coord(i)
                )
                ub_s_reg0 = ub_s_i0.load()
                ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                ub_s_reg0 = tla.where(pregTailN, ub_s_reg0, minVreg)
                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                max_reg = ub_s_reg0.reduce(tla.ReductionOp.MAX, mask=pregFull)
                ub_now_max_i.store(max_reg, tla.params.UnalignStoreParams())
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            ub_last_max_i_de = tla.tile_view(
                lastMaxAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            ub_now_max_i_de = tla.tile_view(
                nowMaxAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            ub_last_sum_i_de = tla.tile_view(
                lastSumAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            ub_dm_i_de = tla.tile_view(
                expMaxUbAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            now_max_reg_de = ub_now_max_i_de.load()
            last_max_reg_de = ub_last_max_i_de.load()
            last_sum_reg = ub_last_sum_i_de.load()
            max_reg_de = tla.max(now_max_reg_de, last_max_reg_de, mask=pregFull)
            exp_sub_max_reg = tla.exp(
                tla.sub(last_max_reg_de, max_reg_de, mask=pregFull), mask=pregFull
            )
            update_exp_sub_reg = tla.mul(exp_sub_max_reg, last_sum_reg, mask=pregFull)
            ub_now_max_i_de.store(max_reg_de, mask=pregFull)
            ub_last_max_i_de.store(max_reg_de, mask=pregFull)
            ub_dm_i_de.store(exp_sub_max_reg, mask=pregFull)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            for j in tla.range(m):
                ub_now_max_iDe = tla.tile_view(
                    nowMaxAddr, tla.make_shape(1), tla.make_coord(j)
                )
                ub_now_sum_iDe = tla.tile_view(
                    nowSumAddr, tla.make_shape(1), tla.make_coord(j)
                )
                ub_s_i0De = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(j, c0)
                )
                ub_p_zN_f16_i = tla.tile_view(
                    ub_p_zN, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
                )
                max_regDe = ub_now_max_iDe.load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                ub_s_reg0De = ub_s_i0De.load()
                exp_reg0 = tla.exp(
                    tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull
                )
                exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                ub_now_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams())
                exp_dst_reg0 = exp_reg0.to(ctx.DTYPE_P, cast_trait_zero, mask=pregFull)
                exp_dst_reg, zero_reg = tla.deinterleave(exp_dst_reg0, exp_dst_reg0)
                ub_p_zN_f16_i.store(
                    exp_dst_reg,
                    params=tla.params.BlockStoreParams(block_stride=65),
                    mask=preg_all_b16,
                )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            ub_now_sum_i_de = tla.tile_view(
                nowSumAddr, tla.make_shape(VL_F32), tla.make_coord(0)
            )
            now_sum_reg = ub_now_sum_i_de.load()
            update_exp_sub_reg = tla.add(update_exp_sub_reg, now_sum_reg, mask=pregFull)
            ub_last_sum_i_de.store(update_exp_sub_reg, mask=pregFull)


@tla.jit
def _sf_first(ctx, n, tailN, m, ub_s, lastMaxAddr, lastSumAddr, ub_p_zN):
    """softmax 首(nomask)。"""
    c0 = 0
    c1 = 1
    if n > 64:
        with tla.vec.func(mode="simd"):
            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
            cast_trait_zero = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            cast_trait_one = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ONE,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
            for i in tla.range(m):
                ub_s_i0 = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                )
                ub_s_i1 = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c1)
                )
                ub_last_max_i = tla.tile_view(
                    lastMaxAddr, tla.make_shape(1), tla.make_coord(i)
                )
                ub_s_reg0 = ub_s_i0.load()
                ub_s_reg1 = ub_s_i1.load()
                ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                ub_s_reg1 = tla.mul(ub_s_reg1, ctx.scaleValue_, mask=pregFull)
                ub_s_reg1 = tla.where(pregTailN, ub_s_reg1, minVreg)
                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                max_tmp_reg = tla.max(ub_s_reg0, ub_s_reg1, mask=pregFull)
                max_reg = max_tmp_reg.reduce(tla.ReductionOp.MAX, mask=pregFull)
                ub_last_max_i.store(max_reg, tla.params.UnalignStoreParams())
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            for j in tla.range(m):
                ub_last_max_iDe = tla.tile_view(
                    lastMaxAddr, tla.make_shape(1), tla.make_coord(j)
                )
                ub_last_sum_iDe = tla.tile_view(
                    lastSumAddr, tla.make_shape(1), tla.make_coord(j)
                )
                ub_s_i0De = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
                )
                ub_p_zN_f16_i = tla.tile_view(
                    ub_p_zN, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
                )
                max_regDe = ub_last_max_iDe.load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                ub_s_odd_reg, ub_s_even_reg = ub_s_i0De.load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                    )
                )
                exp_odd_reg0 = tla.exp(
                    tla.sub(ub_s_odd_reg, max_regDe, mask=pregFull), mask=pregFull
                )
                exp_even_reg1 = tla.exp(
                    tla.sub(ub_s_even_reg, max_regDe, mask=pregFull), mask=pregFull
                )
                exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                ub_last_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams())
                exp_dst_reg0 = exp_even_reg1.to(
                    ctx.DTYPE_P, cast_trait_one, mask=pregFull
                )
                exp_dst_reg1 = exp_odd_reg0.to(
                    ctx.DTYPE_P, cast_trait_zero, mask=pregFull
                )
                exp_dst_reg = tla.bitwise_or(
                    exp_dst_reg0, exp_dst_reg1, mask=preg_all_b16
                )
                ub_p_zN_f16_i.store(
                    exp_dst_reg,
                    params=tla.params.BlockStoreParams(block_stride=65),
                    mask=preg_all_b16,
                )
    else:
        with tla.vec.func(mode="simd"):
            pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
            cast_trait_zero = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            cast_trait_one = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ONE,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
            for i in tla.range(m):
                ub_s_i0 = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
                )
                ub_last_max_i = tla.tile_view(
                    lastMaxAddr, tla.make_shape(1), tla.make_coord(i)
                )
                ub_s_reg0 = ub_s_i0.load()
                ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                ub_s_reg0 = tla.where(pregTailN, ub_s_reg0, minVreg)
                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                max_reg = ub_s_reg0.reduce(tla.ReductionOp.MAX, mask=pregFull)
                ub_last_max_i.store(max_reg, tla.params.UnalignStoreParams())
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            for j in tla.range(m):
                ub_last_max_iDe = tla.tile_view(
                    lastMaxAddr, tla.make_shape(1), tla.make_coord(j)
                )
                ub_last_sum_iDe = tla.tile_view(
                    lastSumAddr, tla.make_shape(1), tla.make_coord(j)
                )
                ub_s_i0De = tla.tile_view(
                    ub_s, tla.make_shape(1, VL_F32), tla.make_coord(j, c0)
                )
                ub_p_zN_f16_i = tla.tile_view(
                    ub_p_zN, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
                )
                max_regDe = ub_last_max_iDe.load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                ub_s_reg0De = ub_s_i0De.load()
                exp_reg0 = tla.exp(
                    tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull
                )
                exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                ub_last_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams())
                exp_dst_reg0 = exp_reg0.to(ctx.DTYPE_P, cast_trait_zero, mask=pregFull)
                exp_dst_reg, zero_reg = tla.deinterleave(exp_dst_reg0, exp_dst_reg0)
                ub_p_zN_f16_i.store(
                    exp_dst_reg,
                    params=tla.params.BlockStoreParams(block_stride=65),
                    mask=preg_all_b16,
                )


@tla.jit
def _sf_mid_n64(
    ctx,
    tailN,
    m,
    lastMaxAddr,
    nowMaxAddr,
    lastSumAddr,
    expMaxUbAddr,
    nowSumAddr,
    ub_s,
    ub_p_zN,
):
    """softmax 中/末 n>64(nomask)。"""
    c0 = 0
    c1 = 1
    with tla.vec.func(mode="simd"):
        pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
        pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
        cast_trait_zero = tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO,
            sat_mode=tla.params.SatMode.SAT,
            round_mode=tla.params.RoundMode.CAST_ROUND,
        )
        cast_trait_one = tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ONE,
            sat_mode=tla.params.SatMode.SAT,
            round_mode=tla.params.RoundMode.CAST_ROUND,
        )
        minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
        for i in tla.range(m):
            ub_s_i0 = tla.tile_view(
                ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
            )
            ub_s_i1 = tla.tile_view(
                ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c1)
            )
            ub_now_max_i = tla.tile_view(
                nowMaxAddr, tla.make_shape(1), tla.make_coord(i)
            )
            ub_s_reg0 = ub_s_i0.load()
            ub_s_reg1 = ub_s_i1.load()
            ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
            ub_s_reg1 = tla.mul(ub_s_reg1, ctx.scaleValue_, mask=pregTailN)
            ub_s_reg1 = tla.where(pregTailN, ub_s_reg1, minVreg)
            ub_s_i0.store(ub_s_reg0, mask=pregFull)
            ub_s_i1.store(ub_s_reg1, mask=pregFull)
            max_tmp_reg = tla.max(ub_s_reg0, ub_s_reg1, mask=pregFull)
            max_reg = max_tmp_reg.reduce(tla.ReductionOp.MAX, mask=pregFull)
            ub_now_max_i.store(max_reg, tla.params.UnalignStoreParams())
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        ub_last_max_i_de = tla.tile_view(
            lastMaxAddr, tla.make_shape(VL_F32), tla.make_coord(0)
        )
        ub_now_max_i_de = tla.tile_view(
            nowMaxAddr, tla.make_shape(VL_F32), tla.make_coord(0)
        )
        ub_last_sum_i_de = tla.tile_view(
            lastSumAddr, tla.make_shape(VL_F32), tla.make_coord(0)
        )
        ub_dm_i_de = tla.tile_view(
            expMaxUbAddr, tla.make_shape(VL_F32), tla.make_coord(0)
        )
        now_max_reg_de = ub_now_max_i_de.load()
        last_max_reg_de = ub_last_max_i_de.load()
        last_sum_reg = ub_last_sum_i_de.load()
        max_reg_de = tla.max(now_max_reg_de, last_max_reg_de, mask=pregFull)
        exp_sub_max_reg = tla.exp(
            tla.sub(last_max_reg_de, max_reg_de, mask=pregFull), mask=pregFull
        )
        update_exp_sub_reg = tla.mul(exp_sub_max_reg, last_sum_reg, mask=pregFull)
        ub_now_max_i_de.store(max_reg_de, mask=pregFull)
        ub_last_max_i_de.store(max_reg_de, mask=pregFull)
        ub_dm_i_de.store(exp_sub_max_reg, mask=pregFull)
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        for j in tla.range(m):
            ub_now_max_iDe = tla.tile_view(
                nowMaxAddr, tla.make_shape(1), tla.make_coord(j)
            )
            ub_now_sum_iDe = tla.tile_view(
                nowSumAddr, tla.make_shape(1), tla.make_coord(j)
            )
            ub_s_i0De = tla.tile_view(
                ub_s, tla.make_shape(1, VL_F32), tla.make_coord(j, c0)
            )
            ub_p_zN_f16_i = tla.tile_view(
                ub_p_zN, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
            )
            max_regDe = ub_now_max_iDe.load(
                params=tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                )
            )
            ub_s_odd_reg, ub_s_even_reg = ub_s_i0De.load(
                params=tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                )
            )
            exp_odd_reg0 = tla.exp(
                tla.sub(ub_s_odd_reg, max_regDe, mask=pregFull), mask=pregFull
            )
            exp_even_reg1 = tla.exp(
                tla.sub(ub_s_even_reg, max_regDe, mask=pregFull), mask=pregFull
            )
            exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
            exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
            ub_now_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams())
            exp_dst_reg0 = exp_even_reg1.to(ctx.DTYPE_P, cast_trait_one, mask=pregFull)
            exp_dst_reg1 = exp_odd_reg0.to(ctx.DTYPE_P, cast_trait_zero, mask=pregFull)
            exp_dst_reg = tla.bitwise_or(exp_dst_reg0, exp_dst_reg1, mask=preg_all_b16)
            ub_p_zN_f16_i.store(
                exp_dst_reg,
                params=tla.params.BlockStoreParams(block_stride=65),
                mask=preg_all_b16,
            )
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        ub_now_sum_i_de = tla.tile_view(
            nowSumAddr, tla.make_shape(VL_F32), tla.make_coord(0)
        )
        now_sum_reg = ub_now_sum_i_de.load()
        update_exp_sub_reg = tla.add(update_exp_sub_reg, now_sum_reg, mask=pregFull)
        ub_last_sum_i_de.store(update_exp_sub_reg, mask=pregFull)


@tla.jit
def _sf_mid_nlt64(
    ctx,
    tailN,
    m,
    lastMaxAddr,
    nowMaxAddr,
    lastSumAddr,
    expMaxUbAddr,
    nowSumAddr,
    ub_s,
    ub_p_zN,
):
    """softmax 中/末 n<=64(nomask)。"""
    c0 = 0
    with tla.vec.func(mode="simd"):
        pregFull = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        preg_all_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
        pregTailN, _ = tla.update_mask(tailN, dtype=tla.Float32)
        cast_trait_zero = tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO,
            sat_mode=tla.params.SatMode.SAT,
            round_mode=tla.params.RoundMode.CAST_ROUND,
        )
        minVreg = tla.full(MIN_VALUE, dtype=tla.Float32)
        for i in tla.range(m):
            ub_s_i0 = tla.tile_view(
                ub_s, tla.make_shape(1, VL_F32), tla.make_coord(i, c0)
            )
            ub_now_max_i = tla.tile_view(
                nowMaxAddr, tla.make_shape(1), tla.make_coord(i)
            )
            ub_s_reg0 = ub_s_i0.load()
            ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
            ub_s_reg0 = tla.where(pregTailN, ub_s_reg0, minVreg)
            ub_s_i0.store(ub_s_reg0, mask=pregFull)
            max_reg = ub_s_reg0.reduce(tla.ReductionOp.MAX, mask=pregFull)
            ub_now_max_i.store(max_reg, tla.params.UnalignStoreParams())
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        ub_last_max_i_de = tla.tile_view(
            lastMaxAddr, tla.make_shape(VL_F32), tla.make_coord(0)
        )
        ub_now_max_i_de = tla.tile_view(
            nowMaxAddr, tla.make_shape(VL_F32), tla.make_coord(0)
        )
        ub_last_sum_i_de = tla.tile_view(
            lastSumAddr, tla.make_shape(VL_F32), tla.make_coord(0)
        )
        ub_dm_i_de = tla.tile_view(
            expMaxUbAddr, tla.make_shape(VL_F32), tla.make_coord(0)
        )
        now_max_reg_de = ub_now_max_i_de.load()
        last_max_reg_de = ub_last_max_i_de.load()
        last_sum_reg = ub_last_sum_i_de.load()
        max_reg_de = tla.max(now_max_reg_de, last_max_reg_de, mask=pregFull)
        exp_sub_max_reg = tla.exp(
            tla.sub(last_max_reg_de, max_reg_de, mask=pregFull), mask=pregFull
        )
        update_exp_sub_reg = tla.mul(exp_sub_max_reg, last_sum_reg, mask=pregFull)
        ub_now_max_i_de.store(max_reg_de, mask=pregFull)
        ub_last_max_i_de.store(max_reg_de, mask=pregFull)
        ub_dm_i_de.store(exp_sub_max_reg, mask=pregFull)
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        for j in tla.range(m):
            ub_now_max_iDe = tla.tile_view(
                nowMaxAddr, tla.make_shape(1), tla.make_coord(j)
            )
            ub_now_sum_iDe = tla.tile_view(
                nowSumAddr, tla.make_shape(1), tla.make_coord(j)
            )
            ub_s_i0De = tla.tile_view(
                ub_s, tla.make_shape(1, VL_F32), tla.make_coord(j, c0)
            )
            ub_p_zN_f16_i = tla.tile_view(
                ub_p_zN, tla.make_shape(1, VL_F16), tla.make_coord(j, c0)
            )
            max_regDe = ub_now_max_iDe.load(
                params=tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                )
            )
            ub_s_reg0De = ub_s_i0De.load()
            exp_reg0 = tla.exp(
                tla.sub(ub_s_reg0De, max_regDe, mask=pregFull), mask=pregFull
            )
            exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
            ub_now_sum_iDe.store(exp_sum_reg, tla.params.UnalignStoreParams())
            exp_dst_reg0 = exp_reg0.to(ctx.DTYPE_P, cast_trait_zero, mask=pregFull)
            exp_dst_reg, zero_reg = tla.deinterleave(exp_dst_reg0, exp_dst_reg0)
            ub_p_zN_f16_i.store(
                exp_dst_reg,
                params=tla.params.BlockStoreParams(block_stride=65),
                mask=preg_all_b16,
            )
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        ub_now_sum_i_de = tla.tile_view(
            nowSumAddr, tla.make_shape(VL_F32), tla.make_coord(0)
        )
        now_sum_reg = ub_now_sum_i_de.load()
        update_exp_sub_reg = tla.add(update_exp_sub_reg, now_sum_reg, mask=pregFull)
        ub_last_sum_i_de.store(update_exp_sub_reg, mask=pregFull)
