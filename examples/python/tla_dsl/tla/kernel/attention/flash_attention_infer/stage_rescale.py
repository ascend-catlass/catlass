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
from fa_common import P_L1_BUF, UB_S_OTMP_BUF_STAGES, VL_F16, VL_F32


@tla.jit
def _rescale_stage(
    task,
    ctx,
    bufs,
    sync,
    validIdxDe,
    isLastKvSTileDe,
    ubOTmp_ptr,
    isFirstKvSTileDe,
    attentionOut,
):
    """rescale+写回。"""
    c0 = 0
    c1 = 1
    with tla.vector():
        oShapeColDe = tla.as_numeric(ctx.embed_)
        gm_o = tla.make_tensor(
            attentionOut.ptr + task.qBOffset + task.qHeadIdx * ctx.qNOffset,
            tla.make_layout(
                tla.make_shape(task.curQSeqlen, oShapeColDe),
                tla.make_stride(ctx.oSOffset, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        rowNumOri = task.rowNum
        colNumOri = tla.as_numeric(ctx.embed_)
        subBlockIdxDe = tla.arch.sub_block_idx()
        subIdxEffDe = subBlockIdxDe  # 原始索引：m=1时sub1跳过计算与GM写回
        subBlockNum = tla.as_numeric(2)
        rowNumSplit = (rowNumOri + 1) // subBlockNum
        rowNumSplit = rowNumOri if rowNumOri < rowNumSplit else rowNumSplit
        rowNumCurSubCore = (
            rowNumSplit if subIdxEffDe == c0 else (rowNumOri - rowNumSplit)
        )
        colNumCurSubCore = colNumOri
        gmO_tile = tla.tile_view(
            gm_o,
            tla.make_shape(ctx.qBaseTile_, colNumCurSubCore),
            tla.make_coord(task.qSTileIdx, c0),
        )
        ubOTmpBufId = validIdxDe % UB_S_OTMP_BUF_STAGES
        curTileMod = validIdxDe % P_L1_BUF
        expMax_ptrDe = (
            bufs.expMax_ptrs[0]
            if curTileMod == c0
            else (bufs.expMax_ptrs[1] if curTileMod == c1 else bufs.expMax_ptrs[2])
        )
        # SubCoreCompute
        if rowNumCurSubCore == c0:
            if ubOTmpBufId == c0:
                tla.cross_core_wait_flag(sync.ub_otmp_cross_ready_0, tla.pipes.VECTOR)
                tla.cross_core_set_flag(sync.ub_otmp_cross_ready_0, tla.pipes.VECTOR)
            else:
                tla.cross_core_wait_flag(sync.ub_otmp_cross_ready_1, tla.pipes.VECTOR)
                tla.cross_core_set_flag(sync.ub_otmp_cross_ready_1, tla.pipes.VECTOR)
        else:
            # SubCoreCompute 标量参数
            mDe = rowNumCurSubCore
            nDe = colNumCurSubCore
            vlSizeDe = VL_F32
            colTail = (nDe - 1) % vlSizeDe + 1
            # UB 地址视图
            loUb = tla.make_tensor(
                ubOTmp_ptr,
                tla.make_layout(tla.make_shape(mDe, 128), tla.make_stride(128, 1)),
            )
            goUb = tla.make_tensor(
                bufs.ubO_ptr,
                tla.make_layout(tla.make_shape(mDe, 128), tla.make_stride(128, 1)),
            )
            goUb16 = tla.make_tensor(
                bufs.ubO16_ptr,
                tla.make_layout(tla.make_shape(mDe, 128), tla.make_stride(128, 1)),
            )
            dmUb = tla.make_tensor(
                expMax_ptrDe, tla.make_layout(tla.make_shape(mDe), tla.make_stride(1))
            )
            glUb = tla.make_tensor(
                bufs.lastSum_ptr,
                tla.make_layout(tla.make_shape(mDe), tla.make_stride(1)),
            )
            # 等 PV fixpipe 完成
            if ubOTmpBufId == c0:
                tla.cross_core_wait_flag(sync.ub_otmp_cross_ready_0, tla.pipes.VECTOR)
            else:
                tla.cross_core_wait_flag(sync.ub_otmp_cross_ready_1, tla.pipes.VECTOR)
            tla.wait_flag(sync.mte3_ready_rescale)
            # 四分支 rescale
            if isFirstKvSTileDe and isLastKvSTileDe:
                # 首 & 末：O = OTmp / lastSum
                if nDe > 64:
                    with tla.vec.func(mode="simd"):
                        pregFullDee = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        pregTailDee, colTail2 = tla.update_mask(
                            colTail, dtype=tla.Float32
                        )
                        for i0 in tla.range(rowNumCurSubCore):
                            ub_lo_0 = tla.tile_view(
                                loUb, tla.make_shape(1, VL_F32), tla.make_coord(i0, c0)
                            )
                            ub_go_0 = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i0, c0)
                            )
                            ub_go_1 = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i0, c1)
                            )
                            ub_gl = tla.tile_view(
                                glUb, tla.make_shape(1), tla.make_coord(i0)
                            )
                            lo_reg_0, lo_reg_1 = ub_lo_0.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                )
                            )
                            gl_reg = ub_gl.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            div_reg_0 = tla.div(lo_reg_0, gl_reg, mask=pregFullDee)
                            div_reg_1 = tla.div(lo_reg_1, gl_reg, mask=pregFullDee)
                            ub_go_0.store(div_reg_0, mask=pregFullDee)
                            ub_go_1.store(div_reg_1, mask=pregFullDee)
                else:
                    with tla.vec.func(mode="simd"):
                        pregFullDee = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        pregTailDee, colTail2 = tla.update_mask(
                            colTail, dtype=tla.Float32
                        )
                        for i0 in tla.range(rowNumCurSubCore):
                            ub_lo_0 = tla.tile_view(
                                loUb, tla.make_shape(1, VL_F32), tla.make_coord(i0, c0)
                            )
                            ub_go_0 = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i0, c0)
                            )
                            ub_gl = tla.tile_view(
                                glUb, tla.make_shape(1), tla.make_coord(i0)
                            )
                            lo_reg_0 = ub_lo_0.load()
                            gl_reg = ub_gl.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            div_reg_0 = tla.div(lo_reg_0, gl_reg, mask=pregFullDee)
                            ub_go_0.store(div_reg_0, mask=pregFullDee)
            elif isFirstKvSTileDe and (not isLastKvSTileDe):
                if nDe > 64:
                    with tla.vec.func(mode="simd"):
                        pregFullDee1 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        for i1 in tla.range(rowNumCurSubCore):
                            ub_lo_0 = tla.tile_view(
                                loUb, tla.make_shape(1, VL_F32), tla.make_coord(i1, c0)
                            )
                            ub_go_0 = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i1, c0)
                            )
                            ub_go_1 = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i1, c1)
                            )
                            lo_reg_0, lo_reg_1 = ub_lo_0.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                )
                            )
                            ub_go_0.store(lo_reg_0, mask=pregFullDee1)
                            ub_go_1.store(lo_reg_1, mask=pregFullDee1)
                else:
                    with tla.vec.func(mode="simd"):
                        pregFullDee1 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        for i1 in tla.range(rowNumCurSubCore):
                            ub_lo_0 = tla.tile_view(
                                loUb, tla.make_shape(1, VL_F32), tla.make_coord(i1, c0)
                            )
                            ub_go_0 = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i1, c0)
                            )
                            lo_reg_0 = ub_lo_0.load()
                            ub_go_0.store(lo_reg_0, mask=pregFullDee1)
            elif (not isFirstKvSTileDe) and isLastKvSTileDe:
                # 非首 & 末：O = (O*expMax + OTmp) / lastSum
                if nDe > 64:
                    with tla.vec.func(mode="simd"):
                        pregFullDee2 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        for i2 in tla.range(rowNumCurSubCore):
                            ub_lo_0 = tla.tile_view(
                                loUb, tla.make_shape(1, VL_F32), tla.make_coord(i2, c0)
                            )
                            ub_go_0 = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i2, c0)
                            )
                            ub_go_1 = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i2, c1)
                            )
                            ub_gl = tla.tile_view(
                                glUb, tla.make_shape(1), tla.make_coord(i2)
                            )
                            ub_dm = tla.tile_view(
                                dmUb, tla.make_shape(1), tla.make_coord(i2)
                            )
                            lo_reg_0, lo_reg_1 = ub_lo_0.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                )
                            )
                            go_reg_0 = ub_go_0.load()
                            go_reg_1 = ub_go_1.load()
                            gl_reg = ub_gl.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            dm_reg = ub_dm.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            mul_reg_0 = tla.mul(go_reg_0, dm_reg, mask=pregFullDee2)
                            mul_reg_1 = tla.mul(go_reg_1, dm_reg, mask=pregFullDee2)
                            add_reg_0 = tla.add(mul_reg_0, lo_reg_0, mask=pregFullDee2)
                            add_reg_1 = tla.add(mul_reg_1, lo_reg_1, mask=pregFullDee2)
                            div_reg_0 = tla.div(add_reg_0, gl_reg, mask=pregFullDee2)
                            div_reg_1 = tla.div(add_reg_1, gl_reg, mask=pregFullDee2)
                            ub_go_0.store(div_reg_0, mask=pregFullDee2)
                            ub_go_1.store(div_reg_1, mask=pregFullDee2)
                else:
                    with tla.vec.func(mode="simd"):
                        pregFullDee2 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        for i2 in tla.range(rowNumCurSubCore):
                            ub_lo_0 = tla.tile_view(
                                loUb, tla.make_shape(1, VL_F32), tla.make_coord(i2, c0)
                            )
                            ub_go_0 = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i2, c0)
                            )
                            ub_gl = tla.tile_view(
                                glUb, tla.make_shape(1), tla.make_coord(i2)
                            )
                            ub_dm = tla.tile_view(
                                dmUb, tla.make_shape(1), tla.make_coord(i2)
                            )
                            lo_reg_0 = ub_lo_0.load()
                            go_reg_0 = ub_go_0.load()
                            gl_reg = ub_gl.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            dm_reg = ub_dm.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            mul_reg_0 = tla.mul(go_reg_0, dm_reg, mask=pregFullDee2)
                            add_reg_0 = tla.add(mul_reg_0, lo_reg_0, mask=pregFullDee2)
                            div_reg_0 = tla.div(add_reg_0, gl_reg, mask=pregFullDee2)
                            ub_go_0.store(div_reg_0, mask=pregFullDee2)
            else:
                # 非首 & 非末：O = O*expMax + OTmp
                if nDe > 64:
                    with tla.vec.func(mode="simd"):
                        pregFullDee3 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        for i3 in tla.range(rowNumCurSubCore):
                            ub_lo_0 = tla.tile_view(
                                loUb, tla.make_shape(1, VL_F32), tla.make_coord(i3, c0)
                            )
                            ub_go_0 = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i3, c0)
                            )
                            ub_go_1 = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i3, c1)
                            )
                            ub_dm = tla.tile_view(
                                dmUb, tla.make_shape(1), tla.make_coord(i3)
                            )
                            lo_reg_0, lo_reg_1 = ub_lo_0.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                )
                            )
                            go_reg_0 = ub_go_0.load()
                            go_reg_1 = ub_go_1.load()
                            dm_reg = ub_dm.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            mul_reg_0 = tla.mul(go_reg_0, dm_reg, mask=pregFullDee3)
                            mul_reg_1 = tla.mul(go_reg_1, dm_reg, mask=pregFullDee3)
                            add_reg_0 = tla.add(mul_reg_0, lo_reg_0, mask=pregFullDee3)
                            add_reg_1 = tla.add(mul_reg_1, lo_reg_1, mask=pregFullDee3)
                            ub_go_0.store(add_reg_0, mask=pregFullDee3)
                            ub_go_1.store(add_reg_1, mask=pregFullDee3)
                else:
                    with tla.vec.func(mode="simd"):
                        pregFullDee3 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        for i3 in tla.range(rowNumCurSubCore):
                            ub_lo_0 = tla.tile_view(
                                loUb, tla.make_shape(1, VL_F32), tla.make_coord(i3, c0)
                            )
                            ub_go_0 = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i3, c0)
                            )
                            ub_dm = tla.tile_view(
                                dmUb, tla.make_shape(1), tla.make_coord(i3)
                            )
                            lo_reg_0 = ub_lo_0.load()
                            go_reg_0 = ub_go_0.load()
                            dm_reg = ub_dm.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            mul_reg_0 = tla.mul(go_reg_0, dm_reg, mask=pregFullDee3)
                            add_reg_0 = tla.add(mul_reg_0, lo_reg_0, mask=pregFullDee3)
                            ub_go_0.store(add_reg_0, mask=pregFullDee3)
            if ubOTmpBufId == c0:
                tla.cross_core_set_flag(sync.ub_otmp_cross_ready_0, tla.pipes.VECTOR)
            else:
                tla.cross_core_set_flag(sync.ub_otmp_cross_ready_1, tla.pipes.VECTOR)
            if isLastKvSTileDe:
                if nDe > 64:
                    with tla.vec.func(mode="simd"):
                        cast_trait_zero_de = tla.params.CastParams(
                            reg_slot=tla.params.RegSlot.ZERO,
                            sat_mode=tla.params.SatMode.SAT,
                            round_mode=tla.params.RoundMode.CAST_ROUND,
                        )
                        cast_trait_one_de = tla.params.CastParams(
                            reg_slot=tla.params.RegSlot.ONE,
                            sat_mode=tla.params.SatMode.SAT,
                            round_mode=tla.params.RoundMode.CAST_ROUND,
                        )
                        pregFullDee4 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        pregAll_b16 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float16
                        )
                        for i4 in tla.range(rowNumCurSubCore):
                            ub_go_0_de = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i4, c0)
                            )
                            ub_go_1_de = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i4, c1)
                            )
                            ub_go_b16 = tla.tile_view(
                                goUb16,
                                tla.make_shape(1, VL_F16),
                                tla.make_coord(i4, c0),
                            )
                            go_reg_0_de = ub_go_0_de.load()
                            go_reg_1_de = ub_go_1_de.load()
                            out_reg0 = go_reg_0_de.to(
                                ctx.DTYPE_Q, cast_trait_zero_de, mask=pregFullDee4
                            )
                            out_reg1 = go_reg_1_de.to(
                                ctx.DTYPE_Q, cast_trait_one_de, mask=pregFullDee4
                            )
                            lo_reg_0_de = tla.bitwise_or(
                                out_reg0, out_reg1, mask=pregAll_b16
                            )
                            ub_go_b16.store(lo_reg_0_de, mask=pregAll_b16)
                else:
                    with tla.vec.func(mode="simd"):
                        cast_trait_zero_de = tla.params.CastParams(
                            reg_slot=tla.params.RegSlot.ZERO,
                            sat_mode=tla.params.SatMode.SAT,
                            round_mode=tla.params.RoundMode.CAST_ROUND,
                        )
                        cast_trait_one_de = tla.params.CastParams(
                            reg_slot=tla.params.RegSlot.ONE,
                            sat_mode=tla.params.SatMode.SAT,
                            round_mode=tla.params.RoundMode.CAST_ROUND,
                        )
                        pregFullDee4 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        pregAll_b16 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float16
                        )
                        for i4 in tla.range(rowNumCurSubCore):
                            ub_go_0_de = tla.tile_view(
                                goUb, tla.make_shape(1, VL_F32), tla.make_coord(i4, c0)
                            )
                            ub_go_b16 = tla.tile_view(
                                goUb16,
                                tla.make_shape(1, VL_F16),
                                tla.make_coord(i4, c0),
                            )
                            go_reg_0_de = ub_go_0_de.load()
                            go_reg_0_de = go_reg_0_de.to(
                                ctx.DTYPE_Q, cast_trait_zero_de, mask=pregFullDee4
                            )
                            lo_reg_0_de, zero_reg = tla.deinterleave(
                                go_reg_0_de, go_reg_0_de
                            )
                            ub_go_b16.store(lo_reg_0_de, mask=pregAll_b16)
                tla.set_flag(sync.ub_p_data_ready_0)
                tla.wait_flag(sync.ub_p_data_ready_0)
                gm_out_tile = tla.tile_view(
                    gmO_tile,
                    tla.make_shape(rowNumSplit, nDe),
                    tla.make_coord(subBlockIdxDe, c0),
                )
                tla.copy(gm_out_tile, goUb16)
            tla.set_flag(sync.mte3_ready_rescale)


@tla.jit
def _drain(sync):
    # release
    with tla.cube():
        tla.wait_flag(sync.q_l0a_data_ready)
        tla.wait_flag(sync.k_l0b_data_ready_0)
        tla.wait_flag(sync.k_l0b_data_ready_1)
        tla.wait_flag(sync.v_l0b_data_ready_0)
        tla.wait_flag(sync.v_l0b_data_ready_1)
        tla.wait_flag(sync.l0a_available_0)
        tla.wait_flag(sync.l0a_available_1)
        tla.wait_flag(sync.l0b_available_0)
        tla.wait_flag(sync.l0b_available_1)
        tla.wait_flag(sync.l0c_available_0)
        tla.wait_flag(sync.l0c_available_1)
        tla.wait_flag(sync.l0c_available_2)
        tla.wait_flag(sync.l0c_available_3)
        tla.cross_core_wait_flag(sync.ub_s_cross_ready_0, tla.pipes.FIX)
        tla.cross_core_wait_flag(sync.ub_s_cross_ready_1, tla.pipes.FIX)
        tla.cross_core_wait_flag(sync.ub_otmp_cross_ready_0, tla.pipes.FIX)
        tla.cross_core_wait_flag(sync.ub_otmp_cross_ready_1, tla.pipes.FIX)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla.wait_flag(sync.vec_ready_mte2_0)
        tla.wait_flag(sync.vec_ready_mte2_1)
        tla.wait_flag(sync.mte3_ready_mask_0)
        tla.wait_flag(sync.mte3_ready_mask_1)
        tla.wait_flag(sync.ub_p_available_0)
        tla.wait_flag(sync.ub_p_available_1)
        tla.wait_flag(sync.mte3_ready_rescale)
        tla.cross_core_wait_flag(sync.ub_p_cross_ready_0, tla.pipes.MTE3)
        tla.cross_core_wait_flag(sync.ub_p_cross_ready_1, tla.pipes.MTE3)
        tla.cross_core_wait_flag(sync.ub_p_cross_ready_2, tla.pipes.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)
