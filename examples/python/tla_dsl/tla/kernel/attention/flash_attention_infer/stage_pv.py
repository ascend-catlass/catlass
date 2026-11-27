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
from fa_common import L0_STAGES, PRE_LAUNCH

V_L1_BUF = 2


@tla.jit
def _pv_cube_stage(
    task,
    ctx,
    bufs,
    sync,
    validIdxDe,
    kvSTileSizeActDe,
    l1PBufIdDe,
    ubOTmpTensorTla,
    l1PTensorTlaDe,
    ubOTmpBufId,
    blockTable,
    gatheredKvSTileIdx,
    gatheredKvSTileIdxDe,
    value,
    kvSLoopNum,
    validNum,
):
    """PV 相位。"""
    c0 = 0
    c1 = 1
    with tla.cube():
        kvShapeRowDe = task.curKvSeqlen
        kvShapeColDe = tla.as_numeric(ctx.embed_)
        if tla.const_expr(ctx.paged):
            phys_v = blockTable[
                task.curBatch * ctx.max_blocks_per_batch + gatheredKvSTileIdxDe
            ]
            v_tile_base = (
                value.ptr
                + (
                    phys_v * ctx.kvHeads_ * ctx.kvBaseTile_
                    + task.kvHeadIdx * ctx.kvBaseTile_
                )
                * ctx.embed_
            )
        else:
            gm_v = tla.make_tensor(
                value.ptr + task.gmOffsetV,
                tla.make_layout(
                    tla.make_shape(kvShapeRowDe, kvShapeColDe),
                    tla.make_stride(ctx.vSOffset, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
        # 跨相 L0A/L0B 前缀和
        prefixSumL0AStagesDe = (
            (
                (validIdxDe + c1 + PRE_LAUNCH) * ctx.mm1L0ATotalStages_
                + validIdxDe * ctx.mm2L0ATotalStages_
            )
            if gatheredKvSTileIdx < kvSLoopNum - PRE_LAUNCH
            else (
                (validNum) * ctx.mm1L0ATotalStages_
                + validIdxDe * ctx.mm2L0ATotalStages_
            )
        )
        prefixSumL0BStagesDe = (
            (
                (validIdxDe + c1 + PRE_LAUNCH) * ctx.mm1L0BTotalStages_
                + validIdxDe * ctx.mm2L0BTotalStages_
            )
            if gatheredKvSTileIdx < kvSLoopNum - PRE_LAUNCH
            else (
                (validNum) * ctx.mm1L0BTotalStages_
                + validIdxDe * ctx.mm2L0BTotalStages_
            )
        )
        # L1 buffer 选择
        l1BvBufId = validIdxDe % V_L1_BUF  # l1BBufId = idxDe % vL1BufNum
        l1V_ptr = bufs.l1V_ptrs[0] if l1BvBufId == c0 else bufs.l1V_ptrs[1]
        if tla.const_expr(ctx.paged):
            gm_v_tile = tla.make_tensor(
                v_tile_base,
                tla.make_layout(
                    tla.make_shape(ctx.kvBaseTile_, ctx.embed_),
                    tla.make_stride(ctx.embed_, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
        else:
            gm_v_tile = tla.tile_view(
                gm_v,
                tla.make_shape(ctx.kvBaseTile_, ctx.embed_),
                tla.make_coord(gatheredKvSTileIdxDe, c0),
            )
        l1_v_tile = tla.make_tensor_like(l1V_ptr, gm_v_tile, layoutTag=tla.arch.zN)
        if l1BvBufId == c0:
            tla.wait_flag(sync.v_l0b_data_ready_0)  # MTE1_MTE2
        else:
            tla.wait_flag(sync.v_l0b_data_ready_1)
        tla.copy(l1_v_tile, gm_v_tile)
        if l1BvBufId == c0:
            tla.set_flag(sync.v_l1_data_ready_0)  # MTE2_MTE1
            tla.wait_flag(sync.v_l1_data_ready_0)  # MTE2_MTE1
        else:
            tla.set_flag(sync.v_l1_data_ready_1)
            tla.wait_flag(sync.v_l1_data_ready_1)
        if l1PBufIdDe == c0:
            tla.cross_core_wait_flag(sync.ub_p_cross_ready_0, tla.pipes.MTE1)
        elif l1PBufIdDe == c1:
            tla.cross_core_wait_flag(sync.ub_p_cross_ready_1, tla.pipes.MTE1)
        else:
            tla.cross_core_wait_flag(sync.ub_p_cross_ready_2, tla.pipes.MTE1)
        # copy L1 to l0
        nLoopCounter = validIdxDe
        l0CBufIdDe = nLoopCounter % L0_STAGES  # l0C 只按 n 分 buffer
        l0c_ptrDe = bufs.l0c_ptrs[2] if l0CBufIdDe == c0 else bufs.l0c_ptrs[3]
        ub_o_tile = tla.tile_view(
            ubOTmpTensorTla,
            tla.make_shape(ctx.qBaseTile_, ctx.embed_),
            tla.make_coord(c0, c0),
        )
        l0c_o = tla.make_tensor_like(l0c_ptrDe, ub_o_tile, layoutTag=tla.arch.L0Clayout)
        # L0A/L0B buffer id = (跨相前缀和 + pv mmad 内 stage 号) % L0_STAGES
        l0ALoopCounterDe = prefixSumL0AStagesDe
        l0BLoopCounterDe = prefixSumL0BStagesDe
        l0ABufIdDe = l0ALoopCounterDe % L0_STAGES
        l0BBufIdDe = l0BLoopCounterDe % L0_STAGES
        # V: L1 -> L0B
        l0b_ptrDe = bufs.l0b_ptrs[0] if l0BBufIdDe == c0 else bufs.l0b_ptrs[1]
        l0_b2 = tla.make_tensor_like(l0b_ptrDe, l1_v_tile, tla.arch.nZ)
        if l0BBufIdDe == c0:
            tla.wait_flag(sync.l0b_available_0)  # CUBE_MTE1
        else:
            tla.wait_flag(sync.l0b_available_1)
        tla.copy(l0_b2, l1_v_tile)  # copyL1ToL0B
        if l0BBufIdDe == c0:
            tla.set_flag(sync.l0b_data_ready_0)  # MTE1_CUBE
        else:
            tla.set_flag(sync.l0b_data_ready_1)
        if l1BvBufId == c0:
            tla.set_flag(sync.v_l0b_data_ready_0)  # MTE1_MTE2
        else:
            tla.set_flag(sync.v_l0b_data_ready_1)
        l1_p_l0 = tla.tile_view(
            l1PTensorTlaDe, tla.make_shape(128, 128), tla.make_coord(c0, c0)
        )
        l0a_ptrDe = bufs.l0a_ptrs[0] if l0ABufIdDe == c0 else bufs.l0a_ptrs[1]
        l0_a2 = tla.make_tensor_like(l0a_ptrDe, l1PTensorTlaDe, tla.arch.zN)
        if l0ABufIdDe == c0:
            tla.wait_flag(sync.l0a_available_0)  # CUBE_MTE1
        else:
            tla.wait_flag(sync.l0a_available_1)
        tla.copy(l0_a2, l1_p_l0)  # copyL1ToL0A
        if l0ABufIdDe == c0:
            tla.set_flag(sync.l0a_data_ready_0)  # MTE1_CUBE
        else:
            tla.set_flag(sync.l0a_data_ready_1)
        if l1PBufIdDe == c0:
            tla.cross_core_set_flag(sync.ub_p_cross_ready_0, tla.pipes.MTE1)
        elif l1PBufIdDe == c1:
            tla.cross_core_set_flag(sync.ub_p_cross_ready_1, tla.pipes.MTE1)
        else:
            tla.cross_core_set_flag(sync.ub_p_cross_ready_2, tla.pipes.MTE1)
        if l0ABufIdDe == c0:
            tla.wait_flag(sync.l0a_data_ready_0)  # MTE1_CUBE
        else:
            tla.wait_flag(sync.l0a_data_ready_1)
        if l0BBufIdDe == c0:
            tla.wait_flag(sync.l0b_data_ready_0)  # MTE1_CUBE
        else:
            tla.wait_flag(sync.l0b_data_ready_1)
        if l0CBufIdDe == c0:
            tla.wait_flag(sync.l0c_available_2)  # FIX_CUBE
        else:
            tla.wait_flag(sync.l0c_available_3)
        tla.mmad(l0c_o, l0_a2, l0_b2, init_c=True)
        if l0ABufIdDe == c0:
            tla.set_flag(sync.l0a_available_0)  # CUBE_MTE1
        else:
            tla.set_flag(sync.l0a_available_1)
        if l0BBufIdDe == c0:
            tla.set_flag(sync.l0b_available_0)  # CUBE_MTE1
        else:
            tla.set_flag(sync.l0b_available_1)
        if ubOTmpBufId == c0:
            tla.cross_core_wait_flag(sync.ub_otmp_cross_ready_0, tla.pipes.FIX)
        else:
            tla.cross_core_wait_flag(sync.ub_otmp_cross_ready_1, tla.pipes.FIX)
        if l0CBufIdDe == c0:
            tla.set_flag(sync.l0c_data_ready_2)  # CUBE_FIX
            tla.wait_flag(sync.l0c_data_ready_2)  # CUBE_FIX
        else:
            tla.set_flag(sync.l0c_data_ready_3)
            tla.wait_flag(sync.l0c_data_ready_3)
        tla.copy(
            ubOTmpTensorTla,
            l0c_o,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        if l0CBufIdDe == c0:
            tla.set_flag(sync.l0c_available_2)  # FIX_CUBE
        else:
            tla.set_flag(sync.l0c_available_3)
        if ubOTmpBufId == c0:
            tla.cross_core_set_flag(sync.ub_otmp_cross_ready_0, tla.pipes.FIX)
        else:
            tla.cross_core_set_flag(sync.ub_otmp_cross_ready_1, tla.pipes.FIX)
