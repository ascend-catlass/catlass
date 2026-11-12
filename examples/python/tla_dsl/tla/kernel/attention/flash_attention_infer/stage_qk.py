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
from fa_common import L0_STAGES, PRE_LAUNCH, FaCtx, FaBufs, FaSync

DTYPE_S = tla.Float32
DTYPE_OTMP = tla.Float32
DTYPE_ACC = tla.Float32
L0_TILE_M = 128
L0_TILE_N = 128
L0_TILE_K = 128
K_L1_BUF = 2


@tla.jit
def _read_tiling(tilingInt, tilingScale, head_dim):
    """tiling 读取 + 派生偏移，返回 FaCtx。"""
    _TI_BATCH = 0
    _TI_NUMHEADS = 1
    _TI_KVHEADS = 2
    _TI_MAXQ = 3
    _TI_MAXKV = 4
    _TI_FIRSTTASK = 5
    _TI_TOTALTASK = 6
    _TI_QFORMAT = 21
    _TI_KVFORMAT = 22
    qBaseTile_ = 128
    kvBaseTile_ = 128
    embed_ = head_dim
    embedV_ = head_dim
    qHeads_ = tilingInt[_TI_NUMHEADS]
    kvHeads_ = tilingInt[_TI_KVHEADS]
    maxQSeqlen_ = tilingInt[_TI_MAXQ]
    maxKvSeqlen_ = tilingInt[_TI_MAXKV]
    kvSeqlen_ = tilingInt[_TI_MAXKV]
    qFormat_ = tilingInt[_TI_QFORMAT]  # (0=TND, 1=BSND)
    kvFormat_ = tilingInt[_TI_KVFORMAT]
    scaleValue_ = tilingScale[0]
    mm1L0ATotalStages_ = (qBaseTile_ // L0_TILE_M) * (128 // L0_TILE_K)
    mm1L0BTotalStages_ = (kvBaseTile_ // L0_TILE_N) * (128 // L0_TILE_K)
    mm2L0ATotalStages_ = (qBaseTile_ // L0_TILE_M) * (kvBaseTile_ // L0_TILE_K)
    mm2L0BTotalStages_ = (kvBaseTile_ // L0_TILE_K) * (128 // L0_TILE_N)
    firstBatchTaskNum_ = tilingInt[_TI_FIRSTTASK]  # qSTileNum_ * qHeads_
    totalTaskNum_ = tilingInt[_TI_TOTALTASK]  # batch_ * firstBatchTaskNum_
    groupSize = qHeads_ // kvHeads_
    qNOffset = embed_
    kNOffset = embed_
    vNOffset = embedV_
    oNOffset = embedV_
    qSOffset = qHeads_ * qNOffset
    kSOffset = kvHeads_ * kNOffset
    vSOffset = kvHeads_ * vNOffset
    oSOffset = qHeads_ * oNOffset
    return FaCtx(
        qHeads_=qHeads_,
        kvHeads_=kvHeads_,
        maxQSeqlen_=maxQSeqlen_,
        maxKvSeqlen_=maxKvSeqlen_,
        kvSeqlen_=kvSeqlen_,
        qFormat_=qFormat_,
        kvFormat_=kvFormat_,
        scaleValue_=scaleValue_,
        mm1L0ATotalStages_=mm1L0ATotalStages_,
        mm1L0BTotalStages_=mm1L0BTotalStages_,
        mm2L0ATotalStages_=mm2L0ATotalStages_,
        mm2L0BTotalStages_=mm2L0BTotalStages_,
        firstBatchTaskNum_=firstBatchTaskNum_,
        totalTaskNum_=totalTaskNum_,
        embed_=embed_,
        embedV_=embedV_,
        qBaseTile_=qBaseTile_,
        kvBaseTile_=kvBaseTile_,
        groupSize=groupSize,
        qNOffset=qNOffset,
        kNOffset=kNOffset,
        vNOffset=vNOffset,
        oNOffset=oNOffset,
        qSOffset=qSOffset,
        kSOffset=kSOffset,
        vSOffset=vSOffset,
        oSOffset=oSOffset,
    )


@tla.jit
def _declare_flags():
    """flag 声明。"""
    # --- L1->L0 完成 -> GM->L1 可开始 ---
    q_l0a_data_ready = tla.flag("q_l0a_data_ready", tla.arch.MTE1, tla.arch.MTE2)
    k_l0b_data_ready_0 = tla.flag("k_l0b_data_ready_0", tla.arch.MTE1, tla.arch.MTE2)
    k_l0b_data_ready_1 = tla.flag("k_l0b_data_ready_1", tla.arch.MTE1, tla.arch.MTE2)
    v_l0b_data_ready_0 = tla.flag("v_l0b_data_ready_0", tla.arch.MTE1, tla.arch.MTE2)
    v_l0b_data_ready_1 = tla.flag("v_l0b_data_ready_1", tla.arch.MTE1, tla.arch.MTE2)
    # --- CUBE MMAD 完成 -> L1->L0 可开始 ---
    l0a_available_0 = tla.flag("l0a_available_0", tla.arch.CUBE, tla.arch.MTE1)
    l0a_available_1 = tla.flag("l0a_available_1", tla.arch.CUBE, tla.arch.MTE1)
    l0b_available_0 = tla.flag("l0b_available_0", tla.arch.CUBE, tla.arch.MTE1)
    l0b_available_1 = tla.flag("l0b_available_1", tla.arch.CUBE, tla.arch.MTE1)
    # --- FIX (L0C->UB) 完成 -> CUBE 可开始 ---
    l0c_available_0 = tla.flag("l0c_available_0", tla.arch.FIX, tla.arch.CUBE)
    l0c_available_1 = tla.flag("l0c_available_1", tla.arch.FIX, tla.arch.CUBE)
    l0c_available_2 = tla.flag("l0c_available_2", tla.arch.FIX, tla.arch.CUBE)
    l0c_available_3 = tla.flag("l0c_available_3", tla.arch.FIX, tla.arch.CUBE)
    # --- GM->L1 加载完成 -> L1->L0 可开始 ---
    q_l1_data_ready = tla.flag("q_l1_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    k_l1_data_ready_0 = tla.flag("k_l1_data_ready_0", tla.arch.MTE2, tla.arch.MTE1)
    k_l1_data_ready_1 = tla.flag("k_l1_data_ready_1", tla.arch.MTE2, tla.arch.MTE1)
    v_l1_data_ready_0 = tla.flag("v_l1_data_ready_0", tla.arch.MTE2, tla.arch.MTE1)
    v_l1_data_ready_1 = tla.flag("v_l1_data_ready_1", tla.arch.MTE2, tla.arch.MTE1)
    # --- L1->L0 完成 -> CUBE 可开始 ---
    l0a_data_ready_0 = tla.flag("l0a_data_ready_0", tla.arch.MTE1, tla.arch.CUBE)
    l0a_data_ready_1 = tla.flag("l0a_data_ready_1", tla.arch.MTE1, tla.arch.CUBE)
    l0b_data_ready_0 = tla.flag("l0b_data_ready_0", tla.arch.MTE1, tla.arch.CUBE)
    l0b_data_ready_1 = tla.flag("l0b_data_ready_1", tla.arch.MTE1, tla.arch.CUBE)
    # --- CUBE MMAD 完成 -> FIX (L0C -> UB) 可开始 ---
    l0c_data_ready_0 = tla.flag("l0c_data_ready_0", tla.arch.CUBE, tla.arch.FIX)
    l0c_data_ready_1 = tla.flag("l0c_data_ready_1", tla.arch.CUBE, tla.arch.FIX)
    l0c_data_ready_2 = tla.flag("l0c_data_ready_2", tla.arch.CUBE, tla.arch.FIX)
    l0c_data_ready_3 = tla.flag("l0c_data_ready_3", tla.arch.CUBE, tla.arch.FIX)
    # --- VECTOR 完成 -> gm -> UB ---
    vec_ready_mte2_0 = tla.flag("vec_ready_mte2_0", tla.arch.VECTOR, tla.arch.MTE2)
    vec_ready_mte2_1 = tla.flag("vec_ready_mte2_1", tla.arch.VECTOR, tla.arch.MTE2)
    # --- mask2index
    mte3_ready_mask_0 = tla.flag("mte3_ready_mask_0", tla.arch.MTE3, tla.arch.VECTOR)
    mte3_ready_mask_1 = tla.flag("mte3_ready_mask_1", tla.arch.MTE3, tla.arch.VECTOR)
    # --- softmax
    ub_p_available_0 = tla.flag("ub_p_available_0", tla.arch.MTE3, tla.arch.VECTOR)
    ub_p_available_1 = tla.flag("ub_p_available_1", tla.arch.MTE3, tla.arch.VECTOR)
    # --- rescale
    mte3_ready_rescale = tla.flag("mte3_ready_rescale", tla.arch.MTE3, tla.arch.VECTOR)
    ub_p_data_ready_0 = tla.flag("ub_p_data_ready_0", tla.arch.VECTOR, tla.arch.MTE3)
    ub_p_data_ready_1 = tla.flag("ub_p_data_ready_1", tla.arch.VECTOR, tla.arch.MTE3)
    ub_s_cross_ready_0 = tla.cross_flag("ub_s_cross_ready_0")
    ub_s_cross_ready_1 = tla.cross_flag("ub_s_cross_ready_1")
    ub_otmp_cross_ready_0 = tla.cross_flag("ub_otmp_cross_ready_0")
    ub_otmp_cross_ready_1 = tla.cross_flag("ub_otmp_cross_ready_1")
    ub_p_cross_ready_0 = tla.cross_flag("ub_p_cross_ready_0")
    ub_p_cross_ready_1 = tla.cross_flag("ub_p_cross_ready_1")
    ub_p_cross_ready_2 = tla.cross_flag("ub_p_cross_ready_2")
    return FaSync(
        q_l0a_data_ready,
        k_l0b_data_ready_0,
        k_l0b_data_ready_1,
        v_l0b_data_ready_0,
        v_l0b_data_ready_1,
        l0a_available_0,
        l0a_available_1,
        l0b_available_0,
        l0b_available_1,
        l0c_available_0,
        l0c_available_1,
        l0c_available_2,
        l0c_available_3,
        q_l1_data_ready,
        k_l1_data_ready_0,
        k_l1_data_ready_1,
        v_l1_data_ready_0,
        v_l1_data_ready_1,
        l0a_data_ready_0,
        l0a_data_ready_1,
        l0b_data_ready_0,
        l0b_data_ready_1,
        l0c_data_ready_0,
        l0c_data_ready_1,
        l0c_data_ready_2,
        l0c_data_ready_3,
        vec_ready_mte2_0,
        vec_ready_mte2_1,
        mte3_ready_mask_0,
        mte3_ready_mask_1,
        ub_p_available_0,
        ub_p_available_1,
        mte3_ready_rescale,
        ub_p_data_ready_0,
        ub_p_data_ready_1,
        ub_s_cross_ready_0,
        ub_s_cross_ready_1,
        ub_otmp_cross_ready_0,
        ub_otmp_cross_ready_1,
        ub_p_cross_ready_0,
        ub_p_cross_ready_1,
        ub_p_cross_ready_2,
    )


@tla.jit
def _init_flags(sync):
    """flag 初始置位。"""
    with tla.cube():
        tla.set_flag(sync.q_l0a_data_ready)
        tla.set_flag(sync.k_l0b_data_ready_0)
        tla.set_flag(sync.k_l0b_data_ready_1)
        tla.set_flag(sync.v_l0b_data_ready_0)
        tla.set_flag(sync.v_l0b_data_ready_1)
        tla.set_flag(sync.l0a_available_0)
        tla.set_flag(sync.l0a_available_1)
        tla.set_flag(sync.l0b_available_0)
        tla.set_flag(sync.l0b_available_1)
        tla.set_flag(sync.l0c_available_0)
        tla.set_flag(sync.l0c_available_1)
        tla.set_flag(sync.l0c_available_2)
        tla.set_flag(sync.l0c_available_3)
        tla.cross_core_set_flag(sync.ub_p_cross_ready_0, tla.arch.MTE1)
        tla.cross_core_set_flag(sync.ub_p_cross_ready_1, tla.arch.MTE1)
        tla.cross_core_set_flag(sync.ub_p_cross_ready_2, tla.arch.MTE1)
    with tla.vector():
        tla.set_flag(sync.vec_ready_mte2_0)
        tla.set_flag(sync.vec_ready_mte2_1)
        tla.set_flag(sync.mte3_ready_mask_0)
        tla.set_flag(sync.mte3_ready_mask_1)
        tla.set_flag(sync.ub_p_available_0)
        tla.set_flag(sync.ub_p_available_1)
        tla.set_flag(sync.mte3_ready_rescale)
        tla.cross_core_set_flag(sync.ub_s_cross_ready_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(sync.ub_s_cross_ready_1, tla.arch.VECTOR)
        tla.cross_core_set_flag(sync.ub_otmp_cross_ready_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(sync.ub_otmp_cross_ready_1, tla.arch.VECTOR)


@tla.jit
def _alloc_buffers(ctx):
    """缓冲分配。"""
    # 片上内存分配
    l1Q_ptrs = [
        tla.allocate(
            ctx.qBaseTile_ * ctx.embed_, ctx.DTYPE_Q, tla.AddressSpace.l1, 512
        ),
    ]
    l1K_ptrs = [
        tla.allocate(
            ctx.embed_ * ctx.kvBaseTile_, ctx.DTYPE_K, tla.AddressSpace.l1, 512
        ),
        tla.allocate(
            ctx.embed_ * ctx.kvBaseTile_, ctx.DTYPE_K, tla.AddressSpace.l1, 512
        ),
    ]
    l1P_ptrs = [
        tla.allocate(
            ctx.qBaseTile_ * ctx.kvBaseTile_, ctx.DTYPE_P, tla.AddressSpace.l1, 512
        ),
        tla.allocate(
            ctx.qBaseTile_ * ctx.kvBaseTile_, ctx.DTYPE_P, tla.AddressSpace.l1, 512
        ),
        tla.allocate(
            ctx.qBaseTile_ * ctx.kvBaseTile_, ctx.DTYPE_P, tla.AddressSpace.l1, 512
        ),
    ]
    l1V_ptrs = [
        tla.allocate(
            ctx.kvBaseTile_ * ctx.embedV_, ctx.DTYPE_V, tla.AddressSpace.l1, 512
        ),
        tla.allocate(
            ctx.kvBaseTile_ * ctx.embedV_, ctx.DTYPE_V, tla.AddressSpace.l1, 512
        ),
    ]
    l0a_ptrs = [
        tla.allocate(L0_TILE_M * L0_TILE_K, ctx.DTYPE_Q, tla.AddressSpace.l0a, 512),
        tla.allocate(L0_TILE_M * L0_TILE_K, ctx.DTYPE_Q, tla.AddressSpace.l0a, 512),
    ]
    l0b_ptrs = [
        tla.allocate(L0_TILE_K * L0_TILE_N, ctx.DTYPE_K, tla.AddressSpace.l0b, 512),
        tla.allocate(L0_TILE_K * L0_TILE_N, ctx.DTYPE_K, tla.AddressSpace.l0b, 512),
    ]
    l0c_ptrs = [
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 1024),
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 1024),
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 1024),
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 1024),
    ]
    ubS_ptrs = [
        tla.allocate(
            ctx.qBaseTile_ // 2 * ctx.kvBaseTile_, DTYPE_S, tla.AddressSpace.ub, 256
        ),
        tla.allocate(
            ctx.qBaseTile_ // 2 * ctx.kvBaseTile_, DTYPE_S, tla.AddressSpace.ub, 256
        ),
    ]
    ubP_ptrs = [
        tla.allocate(
            (ctx.qBaseTile_ // 2 + 1) * ctx.kvBaseTile_,
            ctx.DTYPE_P,
            tla.AddressSpace.ub,
            256,
        ),
        tla.allocate(
            (ctx.qBaseTile_ // 2 + 1) * ctx.kvBaseTile_,
            ctx.DTYPE_P,
            tla.AddressSpace.ub,
            256,
        ),
    ]
    ubOTmp_ptrs = [
        tla.allocate(
            ctx.qBaseTile_ // 2 * ctx.embed_, DTYPE_OTMP, tla.AddressSpace.ub, 256
        ),
        tla.allocate(
            ctx.qBaseTile_ // 2 * ctx.embed_, DTYPE_OTMP, tla.AddressSpace.ub, 256
        ),
    ]
    # O 累加器 + 行统计标量
    ubO_ptr = tla.allocate(
        ctx.qBaseTile_ // 2 * ctx.embed_, DTYPE_OTMP, tla.AddressSpace.ub, 256
    )
    ubO16_ptr = tla.recast_ptr(ubO_ptr, dtype=ctx.DTYPE_Q)
    nowMax_ptr = tla.allocate(
        ctx.qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256
    )
    expMax_ptrs = [
        tla.allocate(ctx.qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256),
        tla.allocate(ctx.qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256),
        tla.allocate(ctx.qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256),
    ]
    nowSum_ptr = tla.allocate(
        ctx.qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256
    )
    lastMax_ptr = tla.allocate(
        ctx.qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256
    )
    lastSum_ptr = tla.allocate(
        ctx.qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256
    )
    return FaBufs(
        l1Q_ptrs,
        l1K_ptrs,
        l1P_ptrs,
        l1V_ptrs,
        l0a_ptrs,
        l0b_ptrs,
        l0c_ptrs,
        ubS_ptrs,
        ubP_ptrs,
        ubOTmp_ptrs,
        ubO_ptr,
        ubO16_ptr,
        nowMax_ptr,
        expMax_ptrs,
        nowSum_ptr,
        lastMax_ptr,
        lastSum_ptr,
    )


@tla.jit
def _load_q_tile(task, ctx, sync, query, l1Q_ptr):
    """Q 常驻 L1。"""
    c0 = 0
    with tla.cube():
        gm_q = tla.make_tensor(
            query.ptr + task.qBOffset + task.qHeadIdx * ctx.qNOffset,
            tla.make_layout(
                tla.make_shape(task.curQSeqlen, ctx.embed_),
                tla.make_stride(ctx.qSOffset, 1),
            ),
        )
        gmQTensorTla = tla.tile_view(
            gm_q,
            tla.make_shape(ctx.qBaseTile_, ctx.embed_),
            tla.make_coord(task.qSTileIdx, c0),
        )
        l1_q = tla.make_tensor_like(l1Q_ptr, gmQTensorTla, tla.arch.zN)
        tla.wait_flag(sync.q_l0a_data_ready)  # MTE1_MTE2
        tla.copy(l1_q, gmQTensorTla)
        tla.set_flag(sync.q_l1_data_ready)  # MTE2_MTE1
        tla.wait_flag(sync.q_l1_data_ready)  # MTE2_MTE1


@tla.jit
def _qk_cube_stage(
    ctx,
    task,
    bufs,
    sync,
    kvSTileSizeAct,
    validIdx,
    ubSTensorTla,
    ubSBufId,
    gatheredKvSTileIdx,
    blockTable,
    kvSLoopNum,
    key,
    query,
):
    """QK 相位。"""
    c0 = 0
    with tla.cube():
        gm_q = tla.make_tensor(
            query.ptr + task.qBOffset + task.qHeadIdx * ctx.qNOffset,
            tla.make_layout(
                tla.make_shape(task.curQSeqlen, ctx.embed_),
                tla.make_stride(ctx.qSOffset, 1),
            ),
        )
        gmQTensorTla = tla.tile_view(
            gm_q,
            tla.make_shape(ctx.qBaseTile_, ctx.embed_),
            tla.make_coord(task.qSTileIdx, c0),
        )
        l1_q = tla.make_tensor_like(bufs.l1Q_ptrs[0], gmQTensorTla, tla.arch.zN)
        if tla.const_expr(ctx.paged):
            # PA: 每 tile 读块表页号, 直接以物理页为基址
            phys_k = blockTable[
                task.curBatch * ctx.max_blocks_per_batch + gatheredKvSTileIdx
            ]
            k_tile_base = (
                key.ptr
                + (
                    phys_k * ctx.kvHeads_ * ctx.kvBaseTile_
                    + task.kvHeadIdx * ctx.kvBaseTile_
                )
                * ctx.embed_
            )
        else:
            gm_k = tla.make_tensor(
                key.ptr + (task.kBOffset + task.kvHeadIdx * ctx.kNOffset),
                tla.make_layout(
                    tla.make_shape(ctx.embed_, task.curKvSeqlen),
                    tla.make_stride(1, ctx.kSOffset),
                    layoutTag=tla.arch.ColumnMajor,
                ),
            )
        prefixSumL0AStages = (
            (validIdx * ctx.mm1L0ATotalStages_)
            if validIdx <= PRE_LAUNCH
            else (
                validIdx * ctx.mm1L0ATotalStages_
                + (validIdx - PRE_LAUNCH) * ctx.mm2L0ATotalStages_
            )
        )
        prefixSumL0BStages = (
            (validIdx * ctx.mm1L0BTotalStages_)
            if validIdx <= PRE_LAUNCH
            else (
                validIdx * ctx.mm1L0BTotalStages_
                + (validIdx - PRE_LAUNCH) * ctx.mm2L0BTotalStages_
            )
        )
        # -----------------QK-----------------
        nLoopCounterL1 = validIdx
        # copy gm_k to L1
        l1BBufId = nLoopCounterL1 % K_L1_BUF
        l1K_ptr = bufs.l1K_ptrs[0] if l1BBufId == c0 else bufs.l1K_ptrs[1]
        if tla.const_expr(ctx.paged):
            gm_k_tile = tla.make_tensor(
                k_tile_base,
                tla.make_layout(
                    tla.make_shape(ctx.embed_, ctx.kvBaseTile_),
                    tla.make_stride(1, ctx.embed_),
                    layoutTag=tla.arch.ColumnMajor,
                ),
            )
        else:
            gm_k_tile = tla.tile_view(
                gm_k,
                tla.make_shape(ctx.embed_, ctx.kvBaseTile_),
                tla.make_coord(c0, gatheredKvSTileIdx),
            )
        l1_k_tile = tla.make_tensor_like(l1K_ptr, gm_k_tile, layoutTag=tla.arch.nZ)
        if l1BBufId == c0:
            tla.wait_flag(sync.k_l0b_data_ready_0)  # MTE1_MTE2
        else:
            tla.wait_flag(sync.k_l0b_data_ready_1)
        tla.copy(l1_k_tile, gm_k_tile)
        if l1BBufId == c0:
            tla.set_flag(sync.k_l1_data_ready_0)  # MTE2_MTE1
        else:
            tla.set_flag(sync.k_l1_data_ready_1)
        # copy L1 to l0
        l0CBufId = (nLoopCounterL1) % L0_STAGES
        l0c_ptr = bufs.l0c_ptrs[0] if l0CBufId == c0 else bufs.l0c_ptrs[1]
        ub_s_tile = tla.tile_view(
            ubSTensorTla,
            tla.make_shape(ctx.qBaseTile_, ctx.kvBaseTile_),
            tla.make_coord(c0, c0),
        )
        l0c_s = tla.make_tensor_like(l0c_ptr, ub_s_tile, layoutTag=tla.arch.L0Clayout)
        l0ALoopCounter = prefixSumL0AStages
        l0BLoopCounter = prefixSumL0BStages
        l0ABufId = l0ALoopCounter % L0_STAGES
        l0BBufId = l0BLoopCounter % L0_STAGES
        l0a_ptr = bufs.l0a_ptrs[0] if l0ABufId == c0 else bufs.l0a_ptrs[1]
        l0a_q_tensor = tla.make_tensor_like(l0a_ptr, l1_q, tla.arch.zN)
        if l0ABufId == 0:
            tla.wait_flag(sync.l0a_available_0)  # CUBE_MTE1
        else:
            tla.wait_flag(sync.l0a_available_1)
        tla.copy(l0a_q_tensor, l1_q)
        if l0ABufId == 0:
            tla.set_flag(sync.l0a_data_ready_0)  # MTE1_CUBE
        else:
            tla.set_flag(sync.l0a_data_ready_1)
        l0b_ptr = bufs.l0b_ptrs[0] if l0BBufId == c0 else bufs.l0b_ptrs[1]
        l0b_k_tensor = tla.make_tensor_like(l0b_ptr, l1_k_tile)
        if l0BBufId == 0:
            tla.wait_flag(sync.l0b_available_0)  # CUBE_MTE1
        else:
            tla.wait_flag(sync.l0b_available_1)
        if l1BBufId == c0:
            tla.wait_flag(sync.k_l1_data_ready_0)  # MTE2_MTE1
        else:
            tla.wait_flag(sync.k_l1_data_ready_1)
        tla.copy(l0b_k_tensor, l1_k_tile)
        if l0BBufId == 0:
            tla.set_flag(sync.l0b_data_ready_0)  # MTE1_CUBE
        else:
            tla.set_flag(sync.l0b_data_ready_1)
        if l1BBufId == 0:
            tla.set_flag(sync.k_l0b_data_ready_0)  # MTE1_MTE2
        else:
            tla.set_flag(sync.k_l0b_data_ready_1)
        if l0ABufId == 0:
            tla.wait_flag(sync.l0a_data_ready_0)  # MTE1_CUBE
        else:
            tla.wait_flag(sync.l0a_data_ready_1)
        if l0BBufId == 0:
            tla.wait_flag(sync.l0b_data_ready_0)  # MTE1_CUBE
        else:
            tla.wait_flag(sync.l0b_data_ready_1)
        if l0CBufId == 0:
            tla.wait_flag(sync.l0c_available_0)  # FIX_CUBE
        else:
            tla.wait_flag(sync.l0c_available_1)
        tla.mmad(l0c_s, l0a_q_tensor, l0b_k_tensor, init_c=True)
        if l0ABufId == 0:
            tla.set_flag(sync.l0a_available_0)  # CUBE_MTE1
        else:
            tla.set_flag(sync.l0a_available_1)
        if l0BBufId == 0:
            tla.set_flag(sync.l0b_available_0)  # CUBE_MTE1
        else:
            tla.set_flag(sync.l0b_available_1)
        # ---- fixPipe：L0C(fp32) -> UB(fp16 S) ----
        if ubSBufId == 0:
            tla.cross_core_wait_flag(sync.ub_s_cross_ready_0, tla.arch.FIX)
        else:
            tla.cross_core_wait_flag(sync.ub_s_cross_ready_1, tla.arch.FIX)
        if l0CBufId == 0:
            tla.set_flag(sync.l0c_data_ready_0)  # CUBE_FIX
            tla.wait_flag(sync.l0c_data_ready_0)  # CUBE_FIX
        else:
            tla.set_flag(sync.l0c_data_ready_1)
            tla.wait_flag(sync.l0c_data_ready_1)
        tla.copy(
            ubSTensorTla,
            l0c_s,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        if l0CBufId == 0:
            tla.set_flag(sync.l0c_available_0)  # FIX_CUBE
        else:
            tla.set_flag(sync.l0c_available_1)
        if ubSBufId == 0:
            tla.cross_core_set_flag(sync.ub_s_cross_ready_0, tla.arch.FIX)
        else:
            tla.cross_core_set_flag(sync.ub_s_cross_ready_1, tla.arch.FIX)
        if gatheredKvSTileIdx == kvSLoopNum - 1:
            tla.set_flag(sync.q_l0a_data_ready)  # MTE1_MTE2
