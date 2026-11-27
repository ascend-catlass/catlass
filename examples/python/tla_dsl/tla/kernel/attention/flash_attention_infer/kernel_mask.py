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
from catlass.params import MaskStoreParams
from catlass.params import MaskLoadParams
from fa_common import (
    PRE_LAUNCH,
    P_L1_BUF,
    UB_S_OTMP_BUF_STAGES,
    VL_F32,
    FaTask,
    ceil_div,
)
from stage_qk import (
    _read_tiling,
    _declare_flags,
    _init_flags,
    _alloc_buffers,
    _load_q_tile,
    _qk_cube_stage,
)
from stage_softmax import _mask_sf_first, _mask_sf_mid_n64, _mask_sf_mid_nlt64
from stage_pv import _pv_cube_stage
from stage_rescale import _rescale_stage, _drain


@tla.kernel
def flash_attention_mask_kernel(
    query: tla.Tensor,  # gQ  [S1, D]  fp16（BSND/TND：packed [*,S,N,D] 的单 head 切片）
    key: tla.Tensor,  # gK  [D, S2]  fp16（ColumnMajor：直接给出 K^T）
    value: tla.Tensor,  # gV  [S2, D]  fp16
    attentionOut: tla.Tensor,  # gO  [S1, D]  fp16
    tilingInt: tla.Tensor,  # Tiling 整型包(Int32 1D): FA 21字段 + qFormat/kvFormat
    tilingScale: tla.Tensor,  # Tiling 浮点包（fp32 1D，[scaleValue]）
    actualQseqlen: tla.Tensor,  # gActualQseqlen：TND 为累加 [B+1]，BSND 占位 [B]
    actualKvseqlen: tla.Tensor,  # gActualKvseqlen：TND 非 paged 累加 [B+1]，BSND 逐 batch [B]
    tileRange: tla.Tensor,  # TileMask tile_range：[batch, Tq] int32 — 每 Q tile 的 KV 循环上界
    tileCompute: tla.Tensor,  # TileMask tile_compute_bp：[batch, Tq, Wk] int32 — bit=1 表示该 KV tile 有可见元素需计算
    fineMask: tla.Tensor,  # TileMask fine_mask_bp：[batch, Tq, Wk] int32 — bit=1 需行级右边界 mask
    maskr: tla.Tensor,  # TileMask maskr：[batch, Sq] int32 — 行右边界（k >= maskr 的列被遮蔽）
    blockTable: tla.Tensor,  # PA: [B*maxBlocksPerBatch] int32 块表(物理页号)
    is_fp16: tla.Constexpr[
        bool
    ],  # True=fp16 输入，False=bf16 输入（编译期决定 DTYPE_Q/K/V/P/O）
    uniform_q_seqlen: tla.Constexpr[
        int
    ],  # 定长 Q 每 batch 序列长度（0=非 uniform，走运行时路径）
    uniform_kv_seqlen: tla.Constexpr[int],  # 定长 KV 每 batch 序列长度（0=非 uniform）
    uniform_tasks_per_batch: tla.Constexpr[int],  # 每 batch 的 Q 任务数（0=非 uniform）
    paged: tla.Constexpr[bool],  # PA: kernel 侧块表寻址(cache [numBlocks,Hkv,bs,D])
    max_blocks_per_batch: tla.Constexpr[int],  # PA: 每 batch 块表长度(= ceil(kv/bs))
    head_dim: tla.Constexpr[int],  # head 维度(编译期, 替代 tiling embed 字段)
) -> None:
    """causal mask kernel: chunk-prefill 底对齐因果掩码(TileMask) + 可选分页"""

    # 编译期根据 is_fp16 选择 Q/K/V/P/O 的数据类型；S/OTMP/ACC 恒为 Float32
    ctx = _read_tiling(tilingInt, tilingScale, head_dim)
    ctx.paged = paged
    ctx.max_blocks_per_batch = max_blocks_per_batch
    c0 = 0
    c1 = 1
    TND = 0
    BSND = 1
    if tla.const_expr(is_fp16):
        ctx.DTYPE_Q = tla.Float16
        ctx.DTYPE_K = tla.Float16
        ctx.DTYPE_V = tla.Float16
        ctx.DTYPE_P = tla.Float16
    else:
        ctx.DTYPE_Q = tla.BFloat16
        ctx.DTYPE_K = tla.BFloat16
        ctx.DTYPE_V = tla.BFloat16
        ctx.DTYPE_P = tla.BFloat16
    sync = _declare_flags()
    _init_flags(sync)
    bufs = _alloc_buffers(ctx)
    mask_ub_ptr = tla.allocate(
        ctx.qBaseTile_ // 2 * head_dim, tla.Int32, tla.AddressSpace.ub, 256
    )
    maskr_ub_ptr = tla.allocate(64, tla.Int32, tla.AddressSpace.ub, 256)
    mutex_maskr = tla.mutex(resource="maskr_ub", id=16)
    qBOffset = tla.as_numeric(0)
    kBOffset = tla.as_numeric(0)
    vBOffset = tla.as_numeric(0)
    oBOffset = tla.as_numeric(0)
    preTotalTaskNum = tla.as_numeric(0)
    curBatch = tla.as_numeric(0)
    curQSeqlen = tla.as_numeric(ctx.maxQSeqlen_)
    curKvSeqlen = tla.as_numeric(ctx.maxKvSeqlen_)
    curTotalTaskNum = ctx.firstBatchTaskNum_
    if tla.const_expr(uniform_tasks_per_batch != 0):
        # 定长
        curQSeqlen = tla.as_numeric(uniform_q_seqlen)
        curKvSeqlen = tla.as_numeric(uniform_kv_seqlen)
    else:
        curKvSeqlen = actualKvseqlen[curBatch]
        if ctx.qFormat_ == TND:
            curQSeqlen = actualQseqlen[curBatch + 1] - actualQseqlen[curBatch]
        if ctx.kvFormat_ == TND:
            curKvSeqlen = actualKvseqlen[curBatch + 1] - actualKvseqlen[curBatch]
    maxQsBlockNum = (ctx.maxQSeqlen_ + ctx.qBaseTile_ - 1) // ctx.qBaseTile_
    task_range = tla.range(
        tla.arch.block_idx(), ctx.totalTaskNum_, tla.arch.block_num()
    )
    for taskSlotIdx in task_range:
        if tla.const_expr(uniform_tasks_per_batch != 0):
            curBatch = taskSlotIdx // uniform_tasks_per_batch
            taskIdxCurBatch = taskSlotIdx - curBatch * uniform_tasks_per_batch
            qBOffset = curBatch * uniform_q_seqlen * ctx.qSOffset
            kBOffset = curBatch * uniform_kv_seqlen * ctx.kSOffset
            vBOffset = curBatch * uniform_kv_seqlen * ctx.vSOffset
            oBOffset = curBatch * uniform_q_seqlen * ctx.oSOffset
        else:
            while taskSlotIdx >= curTotalTaskNum:
                curBatch = curBatch + 1
                preTotalTaskNum = curTotalTaskNum
                qBOffset = qBOffset + curQSeqlen * ctx.qSOffset
                kBOffset = kBOffset + curKvSeqlen * ctx.kSOffset
                vBOffset = vBOffset + curKvSeqlen * ctx.vSOffset
                oBOffset = oBOffset + curQSeqlen * ctx.oSOffset
                curQSeqlen = tla.as_numeric(ctx.maxQSeqlen_)
                curKvSeqlen = actualKvseqlen[curBatch]
                if ctx.qFormat_ == TND:
                    curQSeqlen = actualQseqlen[curBatch + 1] - actualQseqlen[curBatch]
                if ctx.kvFormat_ == TND:
                    curKvSeqlen = (
                        actualKvseqlen[curBatch + 1] - actualKvseqlen[curBatch]
                    )
                curTotalTaskNum = (
                    curTotalTaskNum + ceil_div(curQSeqlen, ctx.qBaseTile_) * ctx.qHeads_
                )
            taskIdxCurBatch = taskSlotIdx - preTotalTaskNum
        logicalQSTileIdx = taskIdxCurBatch // ctx.qHeads_
        qHeadIdx = taskIdxCurBatch - logicalQSTileIdx * ctx.qHeads_
        qsBlockNum = ceil_div(curQSeqlen, ctx.qBaseTile_)
        evenTileCount = (qsBlockNum + 1) // 2
        qSTileIdx = (
            (logicalQSTileIdx * 2)
            if logicalQSTileIdx < evenTileCount
            else ((qsBlockNum - 1 - logicalQSTileIdx) * 2 + 1)
        )
        kvHeadIdx = qHeadIdx // ctx.groupSize
        gmOffsetV = vBOffset + kvHeadIdx * ctx.vNOffset
        rowNum = (
            (curQSeqlen - (qsBlockNum - 1) * ctx.qBaseTile_)
            if qSTileIdx == qsBlockNum - 1
            else ctx.qBaseTile_
        )
        kvSTileSizeAct = tla.as_numeric(ctx.kvBaseTile_)
        noSkipKvS = curKvSeqlen
        trIdx = curBatch * maxQsBlockNum + qSTileIdx
        tileRangeVal = tileRange[trIdx]
        tileRangeCount = ctx.kvBaseTile_ * tileRangeVal
        noSkipKvS = min(tileRangeCount, noSkipKvS)
        kvSLoopNum = (noSkipKvS + ctx.kvBaseTile_ - 1) // ctx.kvBaseTile_
        if kvSLoopNum > 0:
            gatheredKvSeqlen = noSkipKvS
            kvSTileSizeActDe = tla.as_numeric(ctx.kvBaseTile_)
            _load_q_tile(
                FaTask(
                    qBOffset=qBOffset,
                    qHeadIdx=qHeadIdx,
                    curQSeqlen=curQSeqlen,
                    qSTileIdx=qSTileIdx,
                ),
                ctx,
                sync,
                query,
                bufs.l1Q_ptrs[0],
            )

            # 循环两阶段：idx < kvSLoopNum 做 QK Mmad(cube) + online Softmax(vector)；
            #            idx >= PRE_LAUNCH 做 PV Mmad(cube) + rescale O(vector)
            KvS_range = tla.range(c0, kvSLoopNum + PRE_LAUNCH, c1)
            launch_idx_0 = tla.as_numeric(-1)
            launch_idx_1 = tla.as_numeric(-2)
            launch_idx_2 = tla.as_numeric(-3)
            validIdx = tla.as_numeric(-1)
            maskMoved = tla.as_numeric(0)  # maskr 是否已搬入 UB（每 Q 块只搬一次）
            validNum = tla.as_numeric(0)
            cmp = tla.as_numeric(True)
            Tk = (ctx.kvSeqlen_ + ctx.kvBaseTile_ - 1) // ctx.kvBaseTile_
            Wk = ceil_div(Tk, 32)
            fineMaskWordBase = (curBatch * maxQsBlockNum + qSTileIdx) * Wk
            maskHasFine = fineMask[fineMaskWordBase] != c0
            for fineMaskWordIdx in tla.range(c1, Wk, c1):
                if fineMask[fineMaskWordBase + fineMaskWordIdx] != c0:
                    maskHasFine = tla.as_numeric(True)
            is_move_mask = maskHasFine
            for gatheredKvSTileIdx in KvS_range:
                if gatheredKvSTileIdx < kvSLoopNum:
                    computeWordIdx = (
                        curBatch * maxQsBlockNum + qSTileIdx
                    ) * Wk + gatheredKvSTileIdx // 32
                    computeBitPos = gatheredKvSTileIdx % 32
                    cmp = ((tileCompute[computeWordIdx] >> computeBitPos) & 1) == 1
                if cmp:
                    launch_idx_2 = launch_idx_1
                    launch_idx_1 = launch_idx_0
                    launch_idx_0 = gatheredKvSTileIdx
                    validIdx = validIdx + 1
                # ==================== 前半 idx<kvSLoopNum：QK(cube) + Softmax(vector) ====================
                if gatheredKvSTileIdx < kvSLoopNum and cmp:
                    validNum = validNum + 1
                    if gatheredKvSTileIdx == kvSLoopNum - c1:
                        kvSTileSizeAct = (
                            gatheredKvSeqlen - gatheredKvSTileIdx * ctx.kvBaseTile_
                        )
                    else:
                        kvSTileSizeAct = tla.as_numeric(ctx.kvBaseTile_)
                    isFirstKvSTile = validIdx == c0
                    ubSBufId = validIdx % UB_S_OTMP_BUF_STAGES
                    ubS_ptr = bufs.ubS_ptrs[0] if ubSBufId == c0 else bufs.ubS_ptrs[1]
                    kvSStartIdx = gatheredKvSTileIdx * ctx.kvBaseTile_
                    wordIdx = (
                        curBatch * maxQsBlockNum + qSTileIdx
                    ) * Wk + gatheredKvSTileIdx // 32
                    bitPos = gatheredKvSTileIdx % 32
                    fineMaskBit = (fineMask[wordIdx] >> bitPos) & 1 == 1
                    ubSTensorTla = tla.make_tensor(
                        ubS_ptr,
                        tla.make_layout(
                            tla.make_shape(rowNum, kvSTileSizeAct),
                            tla.make_stride(128, 1),
                        ),
                    )
                    # QK Mmad
                    _qk_cube_stage(
                        ctx,
                        FaTask(
                            qSTileIdx=qSTileIdx,
                            qBOffset=qBOffset,
                            qHeadIdx=qHeadIdx,
                            curQSeqlen=curQSeqlen,
                            curBatch=curBatch,
                            kBOffset=kBOffset,
                            curKvSeqlen=curKvSeqlen,
                            kvHeadIdx=kvHeadIdx,
                        ),
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
                    )
                    # ------QK end-------
                    l1PBufId = validIdx % P_L1_BUF
                    l1p_ptr = bufs.l1P_ptrs[0]
                    if l1PBufId == c0:
                        l1p_ptr = bufs.l1P_ptrs[0]
                    elif l1PBufId == c1:
                        l1p_ptr = bufs.l1P_ptrs[1]
                    else:
                        l1p_ptr = bufs.l1P_ptrs[2]
                    ubS_tile = tla.tile_view(
                        ubSTensorTla,
                        tla.make_shape(ctx.qBaseTile_, ctx.kvBaseTile_),
                        tla.make_coord(c0, c0),
                    )
                    l1PTensorTla = tla.make_tensor_like(
                        l1p_ptr,
                        ubS_tile,
                        tla.arch.zN,
                    )
                    # online Softmax（maskr 只在首个 KV 迭代搬一次，状态由调用方维护）
                    moveMaskNow = is_move_mask
                    if maskMoved != c0:
                        moveMaskNow = tla.as_numeric(False)
                    _softmax_wrap_mask(
                        ctx,
                        bufs,
                        sync,
                        rowNum,
                        kvSTileSizeAct,
                        isFirstKvSTile,
                        ubSBufId,
                        l1PBufId,
                        ubS_ptr,
                        l1PTensorTla,
                        fineMaskBit,
                        mask_ub_ptr,
                        maskr_ub_ptr,
                        moveMaskNow,
                        mutex_maskr,
                        kvSStartIdx,
                        curBatch,
                        qSTileIdx,
                        maskr,
                    )
                    if moveMaskNow:
                        maskMoved = c1
                # ==================== 后半 idx>=PRE_LAUNCH：PV(cube) + rescale O(vector) ====================
                if gatheredKvSTileIdx >= PRE_LAUNCH and cmp and launch_idx_2 >= 0:
                    gatheredKvSTileIdxDe = launch_idx_2
                    validIdxDe = validIdx - PRE_LAUNCH
                    if gatheredKvSTileIdxDe == kvSLoopNum - c1:
                        kvSTileSizeActDe = (
                            gatheredKvSeqlen - gatheredKvSTileIdxDe * ctx.kvBaseTile_
                        )
                    else:
                        kvSTileSizeActDe = tla.as_numeric(ctx.kvBaseTile_)
                    isFirstKvSTileDe = validIdxDe == c0
                    isLastKvSTileDe = gatheredKvSTileIdxDe == kvSLoopNum - c1
                    ubSBufIdDe = validIdxDe % UB_S_OTMP_BUF_STAGES
                    ubS_ptrDe = (
                        bufs.ubS_ptrs[0] if ubSBufIdDe == c0 else bufs.ubS_ptrs[1]
                    )
                    ubSTensorTlaDe = tla.make_tensor(
                        ubS_ptrDe,
                        tla.make_layout(
                            tla.make_shape(rowNum, kvSTileSizeActDe),
                            tla.make_stride(128, 1),
                        ),
                    )
                    ubOTmpBufId = validIdxDe % UB_S_OTMP_BUF_STAGES
                    ubOTmp_ptr = (
                        bufs.ubOTmp_ptrs[0]
                        if ubOTmpBufId == c0
                        else bufs.ubOTmp_ptrs[1]
                    )
                    ubOTmpTensorTla = tla.make_tensor(
                        ubOTmp_ptr,
                        tla.make_layout(
                            tla.make_shape(rowNum, ctx.embed_), tla.make_stride(128, 1)
                        ),
                    )
                    l1PBufIdDe = validIdxDe % P_L1_BUF
                    l1P_ptrDe = (
                        bufs.l1P_ptrs[0]
                        if l1PBufIdDe == c0
                        else (
                            bufs.l1P_ptrs[1] if l1PBufIdDe == c1 else bufs.l1P_ptrs[2]
                        )
                    )
                    l1PTensorTlaDe = tla.make_tensor_like(
                        l1P_ptrDe,
                        ubSTensorTlaDe,
                        tla.arch.zN,
                    )
                    # PV Mmad
                    _pv_cube_stage(
                        FaTask(
                            curKvSeqlen=curKvSeqlen,
                            rowNum=rowNum,
                            gmOffsetV=gmOffsetV,
                            curBatch=curBatch,
                            kvHeadIdx=kvHeadIdx,
                        ),
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
                    )
                    _rescale_stage(
                        FaTask(
                            rowNum=rowNum,
                            qSTileIdx=qSTileIdx,
                            qBOffset=qBOffset,
                            qHeadIdx=qHeadIdx,
                            curQSeqlen=curQSeqlen,
                        ),
                        ctx,
                        bufs,
                        sync,
                        validIdxDe,
                        isLastKvSTileDe,
                        ubOTmp_ptr,
                        isFirstKvSTileDe,
                        attentionOut,
                    )
                cmp = tla.as_numeric(True)
    _drain(sync)


@tla.jit
def _softmax_wrap_mask(
    ctx,
    bufs,
    sync,
    rowNum,
    kvSTileSizeAct,
    isFirstKvSTile,
    ubSBufId,
    l1PBufId,
    ubS_ptr,
    l1PTensorTla,
    fineMaskBit,
    mask_ub_ptr,
    maskr_ub_ptr,
    is_move_mask,
    mutex_maskr,
    kvSStartIdx,
    curBatch,
    qSTileIdx,
    maskr,
):
    """softmax 向量段。"""
    c0 = 0
    c1 = 1
    with tla.vector():
        subBlockIdx = tla.arch.sub_block_idx()
        subIdxEff = subBlockIdx  # 原始索引：m=1 时 sub1 走 m==c0 守卫分支，不做重复计算
        mCopyOffset = (rowNum + 1) // 2
        mHalf = rowNum if rowNum < mCopyOffset else mCopyOffset
        m = mHalf if subIdxEff == c0 else (rowNum - mHalf)
        ubP_ptr = bufs.ubP_ptrs[0] if ubSBufId == c0 else bufs.ubP_ptrs[1]
        expMax_ptr = (
            bufs.expMax_ptrs[0]
            if l1PBufId == c0
            else (bufs.expMax_ptrs[1] if l1PBufId == c1 else bufs.expMax_ptrs[2])
        )
        if m == c0:
            if ubSBufId == 0:
                tla.cross_core_wait_flag(sync.ub_s_cross_ready_0, tla.pipes.VECTOR)
                tla.cross_core_set_flag(sync.ub_s_cross_ready_0, tla.pipes.VECTOR)
            else:
                tla.cross_core_wait_flag(sync.ub_s_cross_ready_1, tla.pipes.VECTOR)
                tla.cross_core_set_flag(sync.ub_s_cross_ready_1, tla.pipes.VECTOR)
            if l1PBufId == c0:
                tla.cross_core_wait_flag(sync.ub_p_cross_ready_0, tla.pipes.MTE3)
                tla.cross_core_set_flag(sync.ub_p_cross_ready_0, tla.pipes.MTE3)
            elif l1PBufId == c1:
                tla.cross_core_wait_flag(sync.ub_p_cross_ready_1, tla.pipes.MTE3)
                tla.cross_core_set_flag(sync.ub_p_cross_ready_1, tla.pipes.MTE3)
            else:
                tla.cross_core_wait_flag(sync.ub_p_cross_ready_2, tla.pipes.MTE3)
                tla.cross_core_set_flag(sync.ub_p_cross_ready_2, tla.pipes.MTE3)
        else:
            n = kvSTileSizeAct
            vlSize = VL_F32
            tailN = (n - 1) % vlSize + 1
            # UB 地址视图
            ub_s = tla.make_tensor(
                ubS_ptr,
                tla.make_layout(tla.make_shape(m, 128), tla.make_stride(128, 1)),
            )
            # P 的 UB 形状 (65, 128)：每行 64 个 f32 exp 结果转 f16 后占 128;
            # 行 stride 取 64+1=65，错开避免128*2B行间映射到同一UB bank导致读写性能劣化
            ub_p = tla.make_tensor(
                ubP_ptr,
                tla.make_layout(tla.make_shape(65, 128), tla.make_stride(128, 1)),
            )
            nowMaxAddr = tla.make_tensor(
                bufs.nowMax_ptr, tla.make_layout(tla.make_shape(m), tla.make_stride(1))
            )
            nowSumAddr = tla.make_tensor(
                bufs.nowSum_ptr, tla.make_layout(tla.make_shape(m), tla.make_stride(1))
            )
            lastMaxAddr = tla.make_tensor(
                bufs.lastMax_ptr, tla.make_layout(tla.make_shape(m), tla.make_stride(1))
            )
            lastSumAddr = tla.make_tensor(
                bufs.lastSum_ptr, tla.make_layout(tla.make_shape(m), tla.make_stride(1))
            )
            expMaxUbAddr = tla.make_tensor(
                expMax_ptr, tla.make_layout(tla.make_shape(m), tla.make_stride(1))
            )
            ub_mask = tla.make_tensor(
                mask_ub_ptr,
                tla.make_layout(tla.make_shape(m, 128), tla.make_stride(128, 1)),
            )
            maskr_ub = tla.make_tensor(
                maskr_ub_ptr, tla.make_layout(tla.make_shape(m), tla.make_stride(1))
            )
            if is_move_mask and m != c0:
                maskr_offset = (
                    curBatch * ctx.maxQSeqlen_
                    + qSTileIdx * ctx.qBaseTile_
                    + subIdxEff * mHalf
                )
                gm_maskr = tla.make_tensor(
                    maskr.ptr + maskr_offset,
                    tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
                )
                mutex_maskr.lock(pipe=tla.pipes.MTE2)
                tla.copy(maskr_ub, gm_maskr)
                mutex_maskr.unlock(pipe=tla.pipes.MTE2)
            if fineMaskBit:
                mutex_maskr.lock(pipe=tla.pipes.VECTOR)
                if n > 64:
                    with tla.vec.func(mode="simd"):
                        pregFull0 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        pregTailN0, _ = tla.update_mask(tailN, dtype=tla.Float32)
                        pos0 = tla.arange(kvSStartIdx, dtype=tla.Int32)
                        pos1 = tla.arange(kvSStartIdx + 64, dtype=tla.Int32)
                        for im in tla.range(m):
                            ub_mask_i0m = tla.tile_view(
                                ub_mask,
                                tla.make_shape(1, VL_F32),
                                tla.make_coord(im, c0),
                            )
                            ub_mask_i1m = tla.tile_view(
                                ub_mask,
                                tla.make_shape(1, VL_F32),
                                tla.make_coord(im, c1),
                            )
                            maskr_vec = tla.tile_view(
                                maskr_ub, tla.make_shape(1), tla.make_coord(im)
                            ).load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            preg0 = tla.cmp(
                                pos0, maskr_vec, "lt", mask=pregFull0
                            )  # pos >= maskr -> 屏蔽
                            preg1 = tla.cmp(pos1, maskr_vec, "lt", mask=pregFull0)
                            preg1 = tla.bitwise_and(preg1, pregTailN0, mask=pregFull0)
                            ub_mask_i0m.store(preg0, MaskStoreParams())
                            ub_mask_i1m.store(preg1, MaskStoreParams())
                else:
                    with tla.vec.func(mode="simd"):
                        pregFull0 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        pregTailN0, _ = tla.update_mask(tailN, dtype=tla.Float32)
                        pos0 = tla.arange(kvSStartIdx, dtype=tla.Int32)
                        for im in tla.range(m):
                            ub_mask_i0m = tla.tile_view(
                                ub_mask,
                                tla.make_shape(1, VL_F32),
                                tla.make_coord(im, c0),
                            )
                            maskr_vec = tla.tile_view(
                                maskr_ub, tla.make_shape(1), tla.make_coord(im)
                            ).load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            preg0 = tla.cmp(pos0, maskr_vec, "lt", mask=pregFull0)
                            preg0 = tla.bitwise_and(preg0, pregTailN0, mask=pregFull0)
                            ub_mask_i0m.store(preg0, MaskStoreParams())
            # 等 QK Fixpipe 完成
            if ubSBufId == 0:
                tla.cross_core_wait_flag(sync.ub_s_cross_ready_0, tla.pipes.VECTOR)
            else:
                tla.cross_core_wait_flag(sync.ub_s_cross_ready_1, tla.pipes.VECTOR)
            if ubSBufId == c0:
                tla.wait_flag(sync.ub_p_available_0)
            else:
                tla.wait_flag(sync.ub_p_available_1)
            ub_p_zN_full = tla.make_tensor_like(ubP_ptr, ub_p, tla.arch.zNUnAlign)
            ub_p_zN = tla.tile_view(
                ub_p_zN_full, tla.make_shape(m, n), tla.make_coord(c0, c0)
            )
            if isFirstKvSTile:
                _mask_sf_first(
                    ctx,
                    n,
                    fineMaskBit,
                    tailN,
                    m,
                    ub_s,
                    lastMaxAddr,
                    ub_mask,
                    lastSumAddr,
                    ub_p_zN,
                )
            else:
                if n > 64:
                    _mask_sf_mid_n64(
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
                    )
                else:
                    _mask_sf_mid_nlt64(
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
                    )
            # ------------- vf end ----------------------------
            if ubSBufId == c0:
                tla.set_flag(sync.ub_p_data_ready_0)
                tla.wait_flag(sync.ub_p_data_ready_0)
                tla.cross_core_set_flag(sync.ub_s_cross_ready_0, tla.pipes.VECTOR)
            else:
                tla.set_flag(sync.ub_p_data_ready_1)
                tla.wait_flag(sync.ub_p_data_ready_1)
                tla.cross_core_set_flag(sync.ub_s_cross_ready_1, tla.pipes.VECTOR)
            if fineMaskBit:
                mutex_maskr.unlock(pipe=tla.pipes.VECTOR)
            # P(UB) -> L1
            l1P_tile = tla.tile_view(
                l1PTensorTla, tla.make_shape(mHalf, n), tla.make_coord(subIdxEff, c0)
            )
            if l1PBufId == c0:
                tla.cross_core_wait_flag(sync.ub_p_cross_ready_0, tla.pipes.MTE3)
            elif l1PBufId == c1:
                tla.cross_core_wait_flag(sync.ub_p_cross_ready_1, tla.pipes.MTE3)
            else:
                tla.cross_core_wait_flag(sync.ub_p_cross_ready_2, tla.pipes.MTE3)
            tla.copy(l1P_tile, ub_p_zN)
            # 通知 P buffer 已写完可复用
            if ubSBufId == c0:
                tla.set_flag(sync.ub_p_available_0)
            else:
                tla.set_flag(sync.ub_p_available_1)
            # 通知 PV
            if l1PBufId == c0:
                tla.cross_core_set_flag(sync.ub_p_cross_ready_0, tla.pipes.MTE3)
            elif l1PBufId == c1:
                tla.cross_core_set_flag(sync.ub_p_cross_ready_1, tla.pipes.MTE3)
            else:
                tla.cross_core_set_flag(sync.ub_p_cross_ready_2, tla.pipes.MTE3)
