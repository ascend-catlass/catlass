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

import math
from typing import Any
import catlass.tla as tla
from dataclasses import dataclass

from catlass.params import MaskStoreParams
from catlass.params import MaskLoadParams

DTYPE_S = tla.Float32
DTYPE_OTMP = tla.Float32
DTYPE_ACC = tla.Float32

L0_TILE_M = 128
L0_TILE_N = 128
L0_TILE_K = 128

PRE_LAUNCH = 2
UB_S_OTMP_BUF_STAGES = 2
L0_STAGES = 2
Q_L1_BUF = 1
K_L1_BUF = 2
V_L1_BUF = 2
P_L1_BUF = 3

MIN_VALUE = -65504.0


def _ceil_div(curQSeqlen: int, qBaseTile: int) -> int:
    curQSTileNum = (curQSeqlen + qBaseTile - 1) // qBaseTile
    return curQSTileNum


_VL_F32 = 64
_VL_F16 = 128
FRACTAL_ALIGN = 16


@tla.kernel
def flash_attention_infer_kernel(
    query: tla.Tensor,  # gQ  [S1, D]  fp16（BSND/TND：packed [*,S,N,D] 的单 head 切片）
    key: tla.Tensor,  # gK  [D, S2]  fp16（ColumnMajor：直接给出 K^T）
    value: tla.Tensor,  # gV  [S2, D]  fp16
    attentionOut: tla.Tensor,  # gO  [S1, D]  fp16
    tilingInt: tla.Tensor,  # Tiling 整型包(Int32 1D): FA 21字段 + qFormat/kvFormat
    tilingScale: tla.Tensor,  # Tiling 浮点包（fp32 1D，[scaleValue]）
    actualQseqlen: tla.Tensor,  # gActualQseqlen：TND 为累加 [B+1]，BSND 占位 [B]
    actualKvseqlen: tla.Tensor,  # gActualKvseqlen：TND 非 paged 累加 [B+1]，BSND 逐 batch [B]
    tileRange: tla.Tensor,  # TileMask：[batch, Tq] int32（收缩 KV 循环）
    tileCompute: tla.Tensor,  # TileMask：[batch, Tq, Wk] int32（bit=1 表示该 tile 有可见元素需计算）
    fineMask: tla.Tensor,  # TileMask：[batch, Tq, Wk] int32（bit=1 需精细 mask）
    maskr: tla.Tensor,  # TileMask：[batch, Sq] int32
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
    # 编译期根据 is_fp16 选择 Q/K/V/P/O 的数据类型；S/OTMP/ACC 恒为 Float32
    if tla.const_expr(is_fp16):
        DTYPE_Q = tla.Float16
        DTYPE_K = tla.Float16
        DTYPE_V = tla.Float16
        DTYPE_P = tla.Float16
        DTYPE_O = tla.Float16
    else:
        DTYPE_Q = tla.BFloat16
        DTYPE_K = tla.BFloat16
        DTYPE_V = tla.BFloat16
        DTYPE_P = tla.BFloat16
        DTYPE_O = tla.BFloat16
    c0 = 0
    c1 = 1
    TND = 0
    BSND = 1

    # TilingData: FA 21字段前缀 + qFormat/kvFormat(2字段)
    _TI_BATCH = 0
    _TI_NUMHEADS = 1
    _TI_KVHEADS = 2
    _TI_MAXQ = 3
    _TI_MAXKV = 4
    _TI_FIRSTTASK = 5
    _TI_TOTALTASK = 6
    _TI_QBASE = 7
    _TI_KVBASE = 8
    _TI_QFORMAT = 21
    _TI_KVFORMAT = 22

    embed_ = head_dim  # 编译期常量(替代 tiling 字段)
    embedV_ = head_dim
    qBaseTile_ = 128
    kvBaseTile_ = 128
    batch_ = tilingInt[_TI_BATCH]
    qHeads_ = tilingInt[_TI_NUMHEADS]
    kvHeads_ = tilingInt[_TI_KVHEADS]
    maxQSeqlen_ = tilingInt[_TI_MAXQ]
    maxKvSeqlen_ = tilingInt[_TI_MAXKV]
    qSeqlen_ = tilingInt[_TI_MAXQ]
    kvSeqlen_ = tilingInt[_TI_MAXKV]
    qFormat_ = tilingInt[_TI_QFORMAT]  # (0=TND, 1=BSND)
    kvFormat_ = tilingInt[_TI_KVFORMAT]
    scaleValue_ = tilingScale[0]

    mm1L1TileN_ = 128
    mm2L1TileN_ = 128

    mm1L0ATotalStages_ = (qBaseTile_ // L0_TILE_M) * (128 // L0_TILE_K)
    mm1L0BTotalStages_ = (kvBaseTile_ // L0_TILE_N) * (128 // L0_TILE_K)
    mm2L0ATotalStages_ = (qBaseTile_ // L0_TILE_M) * (kvBaseTile_ // L0_TILE_K)
    mm2L0BTotalStages_ = (kvBaseTile_ // L0_TILE_K) * (128 // L0_TILE_N)

    qSTileNum_ = _ceil_div(maxQSeqlen_, qBaseTile_)  # 首 batch qStile 数
    firstBatchTaskNum_ = tilingInt[_TI_FIRSTTASK]  # qSTileNum_ * qHeads_
    totalTaskNum_ = tilingInt[_TI_TOTALTASK]  # batch_ * firstBatchTaskNum_

    # --- L1->L0 完成 -> GM->L1 可开始 ---
    q_l0a_ready_l1 = tla.flag("l0a_ready_l1", tla.arch.MTE1, tla.arch.MTE2)
    k_l0b_ready_l1_0 = tla.flag("k_l0b_ready_l1_0", tla.arch.MTE1, tla.arch.MTE2)
    k_l0b_ready_l1_1 = tla.flag("k_l0b_ready_l1_1", tla.arch.MTE1, tla.arch.MTE2)
    v_l0b_ready_l1_0 = tla.flag("v_l0b_ready_l1_0", tla.arch.MTE1, tla.arch.MTE2)
    v_l0b_ready_l1_1 = tla.flag("v_l0b_ready_l1_1", tla.arch.MTE1, tla.arch.MTE2)

    # --- CUBE MMAD 完成 -> L1->L0 可开始 ---
    mmad_ready_l0a_0 = tla.flag("mmad_ready_l0a_0", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0a_1 = tla.flag("mmad_ready_l0a_1", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0b_0 = tla.flag("mmad_ready_l0b_0", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0b_1 = tla.flag("mmad_ready_l0b_1", tla.arch.CUBE, tla.arch.MTE1)

    # --- FIX (L0C->UB) 完成 -> CUBE 可开始 ---
    fix_ready_mmad_0 = tla.flag("fix_ready_mmad_0", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_1 = tla.flag("fix_ready_mmad_1", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_2 = tla.flag("fix_ready_mmad_2", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_3 = tla.flag("fix_ready_mmad_3", tla.arch.FIX, tla.arch.CUBE)
    # --- GM->L1 加载完成 -> L1->L0 可开始 ---
    q_l1_ready_l0 = tla.flag("q_l1_ready_l0", tla.arch.MTE2, tla.arch.MTE1)
    k_l1_ready_l0_0 = tla.flag("k_l1_ready_l0_0", tla.arch.MTE2, tla.arch.MTE1)
    k_l1_ready_l0_1 = tla.flag("k_l1_ready_l0_1", tla.arch.MTE2, tla.arch.MTE1)
    v_l1_ready_l0_0 = tla.flag("v_l1_ready_l0_0", tla.arch.MTE2, tla.arch.MTE1)
    v_l1_ready_l0_1 = tla.flag("v_l1_ready_l0_1", tla.arch.MTE2, tla.arch.MTE1)

    # --- L1->L0 完成 -> CUBE 可开始 ---
    l0a_ready_mmad_0 = tla.flag("l0a_ready_mmad_0", tla.arch.MTE1, tla.arch.CUBE)
    l0a_ready_mmad_1 = tla.flag("l0a_ready_mmad_1", tla.arch.MTE1, tla.arch.CUBE)
    l0b_ready_mmad_0 = tla.flag("l0b_ready_mmad_0", tla.arch.MTE1, tla.arch.CUBE)
    l0b_ready_mmad_1 = tla.flag("l0b_ready_mmad_1", tla.arch.MTE1, tla.arch.CUBE)

    # --- CUBE MMAD 完成 -> FIX (L0C -> UB) 可开始 ---
    mmad_ready_fix_0 = tla.flag("mmad_ready_fix_0", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_1 = tla.flag("mmad_ready_fix_1", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_2 = tla.flag("mmad_ready_fix_2", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_3 = tla.flag("mmad_ready_fix_3", tla.arch.CUBE, tla.arch.FIX)

    # --- VECTOR 完成 -> gm -> UB ---
    vec_ready_mte2_0 = tla.flag("vec_ready_mte2_0", tla.arch.VECTOR, tla.arch.MTE2)
    vec_ready_mte2_1 = tla.flag("vec_ready_mte2_1", tla.arch.VECTOR, tla.arch.MTE2)
    # --- mask2index
    mte3_ready_mask_0 = tla.flag("mte3_ready_mask_0", tla.arch.MTE3, tla.arch.VECTOR)
    mte3_ready_mask_1 = tla.flag("mte3_ready_mask_1", tla.arch.MTE3, tla.arch.VECTOR)
    # --- softmax
    mte3_ready_softmax_0 = tla.flag(
        "mte3_ready_softmax_0", tla.arch.MTE3, tla.arch.VECTOR
    )
    mte3_ready_softmax_1 = tla.flag(
        "mte3_ready_softmax_1", tla.arch.MTE3, tla.arch.VECTOR
    )
    # --- rescale
    mte3_ready_rescale = tla.flag("mte3_ready_rescale", tla.arch.MTE3, tla.arch.VECTOR)

    # --- tilemask
    mutex_maskr = tla.mutex(resource="maskr_ub", id=16)

    p_ub_ready_l1_0 = tla.flag("p_ub_ready_l1_0", tla.arch.VECTOR, tla.arch.MTE3)
    p_ub_ready_l1_1 = tla.flag("p_ub_ready_l1_1", tla.arch.VECTOR, tla.arch.MTE3)

    mm1_ready_sm_0 = tla.cross_flag("mm1_ready_sm_0")
    mm1_ready_sm_1 = tla.cross_flag("mm1_ready_sm_1")

    mm2_ready_re_0 = tla.cross_flag("mm2_ready_re_0")
    mm2_ready_re_1 = tla.cross_flag("mm2_ready_re_1")

    sm_ready_mm2_0 = tla.cross_flag("sm_ready_mm2_0")
    sm_ready_mm2_1 = tla.cross_flag("sm_ready_mm2_1")
    sm_ready_mm2_2 = tla.cross_flag("sm_ready_mm2_2")

    with tla.cube():
        tla.set_flag(q_l0a_ready_l1)
        tla.set_flag(k_l0b_ready_l1_0)
        tla.set_flag(k_l0b_ready_l1_1)
        tla.set_flag(v_l0b_ready_l1_0)
        tla.set_flag(v_l0b_ready_l1_1)
        tla.set_flag(mmad_ready_l0a_0)
        tla.set_flag(mmad_ready_l0a_1)
        tla.set_flag(mmad_ready_l0b_0)
        tla.set_flag(mmad_ready_l0b_1)
        tla.set_flag(fix_ready_mmad_0)
        tla.set_flag(fix_ready_mmad_1)
        tla.set_flag(fix_ready_mmad_2)
        tla.set_flag(fix_ready_mmad_3)

        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1)
        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1)
        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1)
    with tla.vector():
        tla.set_flag(vec_ready_mte2_0)
        tla.set_flag(vec_ready_mte2_1)
        tla.set_flag(mte3_ready_mask_0)
        tla.set_flag(mte3_ready_mask_1)
        tla.set_flag(mte3_ready_softmax_0)
        tla.set_flag(mte3_ready_softmax_1)
        tla.set_flag(mte3_ready_rescale)

        tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.VECTOR)
        tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.VECTOR)

    # 片上内存分配
    l1Q_ptrs = [
        tla.allocate(qBaseTile_ * 128, DTYPE_Q, tla.AddressSpace.l1, 512),
    ]
    l1K_ptrs = [
        tla.allocate(128 * kvBaseTile_, DTYPE_K, tla.AddressSpace.l1, 512),
        tla.allocate(128 * kvBaseTile_, DTYPE_K, tla.AddressSpace.l1, 512),
    ]
    l1P_ptrs = [
        tla.allocate(qBaseTile_ * kvBaseTile_, DTYPE_P, tla.AddressSpace.l1, 512),
        tla.allocate(qBaseTile_ * kvBaseTile_, DTYPE_P, tla.AddressSpace.l1, 512),
        tla.allocate(qBaseTile_ * kvBaseTile_, DTYPE_P, tla.AddressSpace.l1, 512),
    ]
    l1V_ptrs = [
        tla.allocate(kvBaseTile_ * 128, DTYPE_V, tla.AddressSpace.l1, 512),
        tla.allocate(kvBaseTile_ * 128, DTYPE_V, tla.AddressSpace.l1, 512),
    ]

    l0a_ptrs = [
        tla.allocate(L0_TILE_M * L0_TILE_K, DTYPE_Q, tla.AddressSpace.l0a, 512),
        tla.allocate(L0_TILE_M * L0_TILE_K, DTYPE_Q, tla.AddressSpace.l0a, 512),
    ]
    l0b_ptrs = [
        tla.allocate(L0_TILE_K * L0_TILE_N, DTYPE_K, tla.AddressSpace.l0b, 512),
        tla.allocate(L0_TILE_K * L0_TILE_N, DTYPE_K, tla.AddressSpace.l0b, 512),
    ]
    l0c_ptrs = [
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 512),
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 512),
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 512),
        tla.allocate(L0_TILE_M * L0_TILE_N, DTYPE_ACC, tla.AddressSpace.l0c, 512),
    ]

    ubS_ptrs = [
        tla.allocate(qBaseTile_ // 2 * kvBaseTile_, DTYPE_S, tla.AddressSpace.ub, 256),
        tla.allocate(qBaseTile_ // 2 * kvBaseTile_, DTYPE_S, tla.AddressSpace.ub, 256),
    ]
    ubP_ptrs = [
        tla.allocate(
            (qBaseTile_ // 2 + 1) * kvBaseTile_, DTYPE_P, tla.AddressSpace.ub, 256
        ),
        tla.allocate(
            (qBaseTile_ // 2 + 1) * kvBaseTile_, DTYPE_P, tla.AddressSpace.ub, 256
        ),
    ]
    ubOTmp_ptrs = [
        tla.allocate(qBaseTile_ // 2 * 128, DTYPE_OTMP, tla.AddressSpace.ub, 256),
        tla.allocate(qBaseTile_ // 2 * 128, DTYPE_OTMP, tla.AddressSpace.ub, 256),
    ]

    # O 累加器 + 行统计标量
    ubO_ptr = tla.allocate(qBaseTile_ // 2 * 128, DTYPE_OTMP, tla.AddressSpace.ub, 256)
    ubO16_ptr = tla.recast_ptr(ubO_ptr, dtype=DTYPE_Q)
    nowMax_ptr = tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256)
    expMax_ptrs = [
        tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256),
        tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256),
        tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256),
    ]
    nowSum_ptr = tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256)
    lastMax_ptr = tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256)
    lastSum_ptr = tla.allocate(qBaseTile_ // 2, tla.Float32, tla.AddressSpace.ub, 256)

    mask_ub_ptr = tla.allocate(
        qBaseTile_ // 2 * 128, tla.Int32, tla.AddressSpace.ub, 256
    )
    maskr_ub_ptr = tla.allocate(64, tla.Int32, tla.AddressSpace.ub, 256)

    coreIdx = tla.arch.block_idx()
    coreNum = tla.arch.block_num()
    with tla.cube():
        coreIdx = tla.arch.block_idx()
    with tla.vector():
        coreIdx = tla.arch.block_idx() // 2

    qNOffset = embed_
    kNOffset = embed_
    vNOffset = embedV_
    oNOffset = embedV_
    qSOffset = qHeads_ * qNOffset
    kSOffset = kvHeads_ * kNOffset
    vSOffset = kvHeads_ * vNOffset
    oSOffset = qHeads_ * oNOffset

    embedVRound = (embedV_ + FRACTAL_ALIGN - 1) // FRACTAL_ALIGN * FRACTAL_ALIGN
    groupSize = qHeads_ // kvHeads_

    qBOffset = tla.as_numeric(0)
    kBOffset = tla.as_numeric(0)
    vBOffset = tla.as_numeric(0)
    oBOffset = tla.as_numeric(0)
    preTotalTaskNum = tla.as_numeric(0)
    curBatch = tla.as_numeric(0)

    curQSeqlen = tla.as_numeric(maxQSeqlen_)
    curKvSeqlen = tla.as_numeric(maxKvSeqlen_)
    curTotalTaskNum = firstBatchTaskNum_
    if tla.const_expr(uniform_tasks_per_batch != 0):
        # 定长
        curQSeqlen = tla.as_numeric(uniform_q_seqlen)
        curKvSeqlen = tla.as_numeric(uniform_kv_seqlen)
    else:
        curKvSeqlen = actualKvseqlen[curBatch]
        if qFormat_ == TND:
            curQSeqlen = actualQseqlen[curBatch + 1] - actualQseqlen[curBatch]
        if kvFormat_ == TND:
            curKvSeqlen = actualKvseqlen[curBatch + 1] - actualKvseqlen[curBatch]
    maxQsBlockNum = (maxQSeqlen_ + qBaseTile_ - 1) // qBaseTile_

    task_range = tla.range(tla.arch.block_idx(), totalTaskNum_, tla.arch.block_num())
    for taskSlotIdx in task_range:
        if tla.const_expr(uniform_tasks_per_batch != 0):
            curBatch = taskSlotIdx // uniform_tasks_per_batch
            taskIdxCurBatch = taskSlotIdx - curBatch * uniform_tasks_per_batch
            qBOffset = curBatch * uniform_q_seqlen * qSOffset
            kBOffset = curBatch * uniform_kv_seqlen * kSOffset
            vBOffset = curBatch * uniform_kv_seqlen * vSOffset
            oBOffset = curBatch * uniform_q_seqlen * oSOffset
        else:
            while taskSlotIdx >= curTotalTaskNum:
                curBatch = curBatch + 1
                preTotalTaskNum = curTotalTaskNum
                qBOffset = qBOffset + curQSeqlen * qSOffset
                kBOffset = kBOffset + curKvSeqlen * kSOffset
                vBOffset = vBOffset + curKvSeqlen * vSOffset
                oBOffset = oBOffset + curQSeqlen * oSOffset
                curQSeqlen = tla.as_numeric(maxQSeqlen_)
                curKvSeqlen = actualKvseqlen[curBatch]
                if qFormat_ == TND:
                    curQSeqlen = curQSeqlen = (
                        actualQseqlen[curBatch + 1] - actualQseqlen[curBatch]
                    )
                if kvFormat_ == TND:
                    curKvSeqlen = (
                        actualKvseqlen[curBatch + 1] - actualKvseqlen[curBatch]
                    )
                curTotalTaskNum = (
                    curTotalTaskNum + _ceil_div(curQSeqlen, qBaseTile_) * qHeads_
                )
            taskIdxCurBatch = taskSlotIdx - preTotalTaskNum
        logicalQSTileIdx = taskIdxCurBatch // qHeads_
        qHeadIdx = taskIdxCurBatch - logicalQSTileIdx * qHeads_
        qsBlockNum = _ceil_div(curQSeqlen, qBaseTile_)
        evenTileCount = (qsBlockNum + 1) // 2
        qSTileIdx = (
            (logicalQSTileIdx * 2)
            if logicalQSTileIdx < evenTileCount
            else ((qsBlockNum - 1 - logicalQSTileIdx) * 2 + 1)
        )
        kvHeadIdx = qHeadIdx // groupSize
        qSIdex = qSTileIdx * qBaseTile_
        gmOffsetQ = qBOffset + qHeadIdx * qNOffset
        gmOffsetO = oBOffset + qHeadIdx * oNOffset
        gmOffsetK = kBOffset + kvHeadIdx * kNOffset
        gmOffsetV = vBOffset + kvHeadIdx * vNOffset
        rowNum = (
            (curQSeqlen - (qsBlockNum - 1) * qBaseTile_)
            if qSTileIdx == qsBlockNum - 1
            else qBaseTile_
        )
        rowNumRound = (rowNum + FRACTAL_ALIGN - 1) // FRACTAL_ALIGN * FRACTAL_ALIGN
        kvSTileSizeAct = tla.as_numeric(kvBaseTile_)
        noSkipKvS = curKvSeqlen

        trIdx = curBatch * maxQsBlockNum + qSTileIdx
        tileRangeVal = tileRange[trIdx]
        tileRangeCount = kvBaseTile_ * tileRangeVal
        noSkipKvS = min(tileRangeCount, noSkipKvS)
        kvSLoopNum = (noSkipKvS + kvBaseTile_ - 1) // kvBaseTile_
        if kvSLoopNum > 0:
            gatheredKvSeqlen = noSkipKvS
            kvSTileSizeActDe = tla.as_numeric(kvBaseTile_)

            with tla.cube():
                # loadQGM：整块 Q 常驻 L1，task 内只加载一次。
                gm_q = tla.make_tensor(
                    query.ptr + qBOffset + qHeadIdx * qNOffset,
                    tla.make_layout(
                        tla.make_shape(curQSeqlen, embed_), tla.make_stride(qSOffset, 1)
                    ),
                )
                gmQTensorTla = tla.tile_view(
                    gm_q,
                    tla.make_shape(qBaseTile_, embed_),
                    tla.make_coord(qSTileIdx, c0),
                )
                l1_q = tla.make_tensor_like(l1Q_ptrs[0], gmQTensorTla, tla.arch.zN)
                tla.wait_flag(q_l0a_ready_l1)  # MTE1_MTE2
                tla.copy(l1_q, gmQTensorTla)
                tla.set_flag(q_l1_ready_l0)  # MTE2_MTE1
                tla.wait_flag(q_l1_ready_l0)  # MTE2_MTE1

            # 循环两阶段：idx < kvSLoopNum 做 QK Mmad(cube) + online Softmax(vector)；
            #            idx >= PRE_LAUNCH 做 PV Mmad(cube) + rescale O(vector)
            KvS_range = tla.range(c0, kvSLoopNum + PRE_LAUNCH, c1)
            launch_idx_0 = tla.as_numeric(-1)
            launch_idx_1 = tla.as_numeric(-2)
            launch_idx_2 = tla.as_numeric(-3)
            validIdx = tla.as_numeric(-1)
            validNum = tla.as_numeric(0)
            cmp = tla.as_numeric(True)
            Tk = (kvSeqlen_ + kvBaseTile_ - 1) // kvBaseTile_
            Wk = _ceil_div(Tk, 32)
            fineMaskWordBase = (curBatch * maxQsBlockNum + qSTileIdx) * Wk
            maskHasFine = fineMask[fineMaskWordBase] != c0
            for fineMaskWordIdx in tla.range(c1, Wk, c1):
                if fineMask[fineMaskWordBase + fineMaskWordIdx] != c0:
                    maskHasFine = tla.as_numeric(True)
            is_move_mask = maskHasFine
            is_first = tla.as_numeric(True)
            Tq = (maxQSeqlen_ + qBaseTile_ - 1) // qBaseTile_
            for gatheredKvSTileIdx in KvS_range:
                elemIdx = (
                    curBatch * maxQsBlockNum + qSTileIdx
                ) * Tk + gatheredKvSTileIdx
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
                            gatheredKvSeqlen - gatheredKvSTileIdx * kvBaseTile_
                        )
                    else:
                        kvSTileSizeAct = tla.as_numeric(kvBaseTile_)
                    isFirstKvSTile = validIdx == c0
                    isLastKvS = gatheredKvSTileIdx == kvSLoopNum - c1
                    ubSBufId = validIdx % UB_S_OTMP_BUF_STAGES
                    ubS_ptr = ubS_ptrs[0] if ubSBufId == c0 else ubS_ptrs[1]

                    kvSStartIdx = gatheredKvSTileIdx * kvBaseTile_
                    # TileMask 位图三分支判定
                    wordIdx = (
                        curBatch * maxQsBlockNum + qSTileIdx
                    ) * Wk + gatheredKvSTileIdx // 32
                    bitPos = gatheredKvSTileIdx % 32
                    fineMaskBit = (fineMask[wordIdx] >> bitPos) & 1 == 1

                    colNumRound = (
                        (kvSTileSizeAct + FRACTAL_ALIGN - 1)
                        // FRACTAL_ALIGN
                        * FRACTAL_ALIGN
                    )
                    ubSTensorTla = tla.make_tensor(
                        ubS_ptr,
                        tla.make_layout(
                            tla.make_shape(rowNum, kvSTileSizeAct),
                            tla.make_stride(128, 1),
                        ),
                    )
                    # QK Mmad
                    with tla.cube():
                        gm_q = tla.make_tensor(
                            query.ptr + qBOffset + qHeadIdx * qNOffset,
                            tla.make_layout(
                                tla.make_shape(curQSeqlen, embed_),
                                tla.make_stride(qSOffset, 1),
                            ),
                        )

                        gmQTensorTla = tla.tile_view(
                            gm_q,
                            tla.make_shape(qBaseTile_, embed_),
                            tla.make_coord(qSTileIdx, c0),
                        )
                        l1_q = tla.make_tensor_like(
                            l1Q_ptrs[0], gmQTensorTla, tla.arch.zN
                        )

                        if tla.const_expr(paged):
                            # PA: 每 tile 读块表页号, 直接以物理页为基址
                            phys_k = blockTable[
                                curBatch * max_blocks_per_batch + gatheredKvSTileIdx
                            ]
                            k_tile_base = (
                                key.ptr
                                + (
                                    phys_k * kvHeads_ * kvBaseTile_
                                    + kvHeadIdx * kvBaseTile_
                                )
                                * embed_
                            )
                        else:
                            gm_k = tla.make_tensor(
                                key.ptr + (kBOffset + kvHeadIdx * kNOffset),
                                tla.make_layout(
                                    tla.make_shape(embed_, curKvSeqlen),
                                    tla.make_stride(1, kSOffset),
                                    layoutTag=tla.arch.ColumnMajor,
                                ),
                            )

                        prefixSumL0AStages = (
                            (validIdx * mm1L0ATotalStages_)
                            if validIdx <= PRE_LAUNCH
                            else (
                                validIdx * mm1L0ATotalStages_
                                + (validIdx - PRE_LAUNCH) * mm2L0ATotalStages_
                            )
                        )
                        prefixSumL0BStages = (
                            (validIdx * mm1L0BTotalStages_)
                            if validIdx <= PRE_LAUNCH
                            else (
                                validIdx * mm1L0BTotalStages_
                                + (validIdx - PRE_LAUNCH) * mm2L0BTotalStages_
                            )
                        )
                        # -----------------QK-----------------
                        l1TileNAct = kvSTileSizeAct
                        nLoopCounterL1 = validIdx

                        # copy gm_k to L1
                        l1BBufId = nLoopCounterL1 % K_L1_BUF
                        l1K_ptr = l1K_ptrs[0] if l1BBufId == c0 else l1K_ptrs[1]
                        if tla.const_expr(paged):
                            gm_k_tile = tla.make_tensor(
                                k_tile_base,
                                tla.make_layout(
                                    tla.make_shape(embed_, kvBaseTile_),
                                    tla.make_stride(1, embed_),
                                    layoutTag=tla.arch.ColumnMajor,
                                ),
                            )
                        else:
                            gm_k_tile = tla.tile_view(
                                gm_k,
                                tla.make_shape(embed_, kvBaseTile_),
                                tla.make_coord(c0, gatheredKvSTileIdx),
                            )
                        l1_k_tile = tla.make_tensor_like(
                            l1K_ptr, gm_k_tile, layoutTag=tla.arch.nZ
                        )
                        if l1BBufId == c0:
                            tla.wait_flag(k_l0b_ready_l1_0)  # MTE1_MTE2
                        else:
                            tla.wait_flag(k_l0b_ready_l1_1)
                        tla.copy(l1_k_tile, gm_k_tile)
                        if l1BBufId == c0:
                            tla.set_flag(k_l1_ready_l0_0)  # MTE2_MTE1
                        else:
                            tla.set_flag(k_l1_ready_l0_1)

                        # copy L1 to l0
                        l0TileNAct = l1TileNAct
                        l0CBufId = (nLoopCounterL1) % L0_STAGES
                        l0c_ptr = l0c_ptrs[0] if l0CBufId == c0 else l0c_ptrs[1]
                        ub_s_tile = tla.tile_view(
                            ubSTensorTla,
                            tla.make_shape(qBaseTile_, kvBaseTile_),
                            tla.make_coord(c0, c0),
                        )
                        l0c_s = tla.make_tensor_like(
                            l0c_ptr, ub_s_tile, layoutTag=tla.arch.L0Clayout
                        )

                        l0ALoopCounter = prefixSumL0AStages
                        l0BLoopCounter = prefixSumL0BStages
                        l0ABufId = l0ALoopCounter % L0_STAGES
                        l0BBufId = l0BLoopCounter % L0_STAGES

                        l0a_ptr = l0a_ptrs[0] if l0ABufId == c0 else l0a_ptrs[1]
                        l0a_q_tensor = tla.make_tensor_like(l0a_ptr, l1_q, tla.arch.zN)

                        if l0ABufId == 0:
                            tla.wait_flag(mmad_ready_l0a_0)  # CUBE_MTE1
                        else:
                            tla.wait_flag(mmad_ready_l0a_1)
                        tla.copy(l0a_q_tensor, l1_q)
                        if l0ABufId == 0:
                            tla.set_flag(l0a_ready_mmad_0)  # MTE1_CUBE
                        else:
                            tla.set_flag(l0a_ready_mmad_1)

                        l0b_ptr = l0b_ptrs[0] if l0BBufId == c0 else l0b_ptrs[1]
                        l0b_k_tensor = tla.make_tensor_like(l0b_ptr, l1_k_tile)
                        if l0BBufId == 0:
                            tla.wait_flag(mmad_ready_l0b_0)  # CUBE_MTE1
                        else:
                            tla.wait_flag(mmad_ready_l0b_1)
                        if l1BBufId == c0:
                            tla.wait_flag(k_l1_ready_l0_0)  # MTE2_MTE1
                        else:
                            tla.wait_flag(k_l1_ready_l0_1)
                        tla.copy(l0b_k_tensor, l1_k_tile)

                        if l0BBufId == 0:
                            tla.set_flag(l0b_ready_mmad_0)  # MTE1_CUBE
                        else:
                            tla.set_flag(l0b_ready_mmad_1)
                        if l1BBufId == 0:
                            tla.set_flag(k_l0b_ready_l1_0)  # MTE1_MTE2
                        else:
                            tla.set_flag(k_l0b_ready_l1_1)

                        if l0ABufId == 0:
                            tla.wait_flag(l0a_ready_mmad_0)  # MTE1_CUBE
                        else:
                            tla.wait_flag(l0a_ready_mmad_1)
                        if l0BBufId == 0:
                            tla.wait_flag(l0b_ready_mmad_0)  # MTE1_CUBE
                        else:
                            tla.wait_flag(l0b_ready_mmad_1)
                        if l0CBufId == 0:
                            tla.wait_flag(fix_ready_mmad_0)  # FIX_CUBE
                        else:
                            tla.wait_flag(fix_ready_mmad_1)

                        tla.mmad(l0c_s, l0a_q_tensor, l0b_k_tensor, init_c=True)

                        if l0ABufId == 0:
                            tla.set_flag(mmad_ready_l0a_0)  # CUBE_MTE1
                        else:
                            tla.set_flag(mmad_ready_l0a_1)
                        if l0BBufId == 0:
                            tla.set_flag(mmad_ready_l0b_0)  # CUBE_MTE1
                        else:
                            tla.set_flag(mmad_ready_l0b_1)

                        # ---- fixPipe：L0C(fp32) -> UB(fp16 S) ----
                        if ubSBufId == 0:
                            tla.cross_core_wait_flag(mm1_ready_sm_0, tla.arch.FIX)
                        else:
                            tla.cross_core_wait_flag(mm1_ready_sm_1, tla.arch.FIX)
                        if l0CBufId == 0:
                            tla.set_flag(mmad_ready_fix_0)  # CUBE-FIX
                            tla.wait_flag(mmad_ready_fix_0)  # CUBE-FIX
                        else:
                            tla.set_flag(mmad_ready_fix_1)
                            tla.wait_flag(mmad_ready_fix_1)

                        tla.copy(
                            ubSTensorTla,
                            l0c_s,
                            tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ),
                        )

                        if l0CBufId == 0:
                            tla.set_flag(fix_ready_mmad_0)  # FIX_CUBE
                        else:
                            tla.set_flag(fix_ready_mmad_1)
                        if ubSBufId == 0:
                            tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.FIX)
                        else:
                            tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.FIX)

                        if gatheredKvSTileIdx == kvSLoopNum - 1:
                            tla.set_flag(q_l0a_ready_l1)  # MTE1_MTE2

                    # ------QK end-------
                    l1PBufId = validIdx % P_L1_BUF
                    l1p_ptr = l1P_ptrs[0]
                    if l1PBufId == c0:
                        l1p_ptr = l1P_ptrs[0]
                    elif l1PBufId == c1:
                        l1p_ptr = l1P_ptrs[1]
                    else:
                        l1p_ptr = l1P_ptrs[2]
                    ubS_tile = tla.tile_view(
                        ubSTensorTla,
                        tla.make_shape(qBaseTile_, kvBaseTile_),
                        tla.make_coord(c0, c0),
                    )
                    l1PTensorTla = tla.make_tensor_like(
                        l1p_ptr,
                        ubS_tile,
                        tla.arch.zN,
                    )
                    # online Softmax
                    with tla.vector():
                        subBlockIdx = tla.arch.sub_block_idx()
                        subIdxEff = subBlockIdx if rowNum > 1 else c0
                        mCopyOffset = (rowNum + 1) // 2
                        mHalf = rowNum if rowNum < mCopyOffset else mCopyOffset
                        m = mHalf if subIdxEff == c0 else (rowNum - mHalf)

                        ubP_ptr = ubP_ptrs[0] if ubSBufId == c0 else ubP_ptrs[1]
                        expMax_ptr = (
                            expMax_ptrs[0]
                            if l1PBufId == c0
                            else (expMax_ptrs[1] if l1PBufId == c1 else expMax_ptrs[2])
                        )

                        if m == c0:
                            if ubSBufId == 0:
                                tla.cross_core_wait_flag(
                                    mm1_ready_sm_0, tla.arch.VECTOR
                                )
                                tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.VECTOR)
                            else:
                                tla.cross_core_wait_flag(
                                    mm1_ready_sm_1, tla.arch.VECTOR
                                )
                                tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.VECTOR)
                            if l1PBufId == c0:
                                tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
                                tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE3)
                            elif l1PBufId == c1:
                                tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
                                tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE3)
                            else:
                                tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)
                                tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE3)
                        else:
                            # 标量参数
                            n = kvSTileSizeAct
                            mRound = (
                                (m + FRACTAL_ALIGN - 1) // FRACTAL_ALIGN * FRACTAL_ALIGN
                            )
                            nRound = (
                                (n + FRACTAL_ALIGN - 1) // FRACTAL_ALIGN * FRACTAL_ALIGN
                            )
                            blockStride = mRound
                            vlSize = _VL_F32  # GetVecLen()/sizeof(fp32) = 64
                            nLoops = (n + vlSize - 1) // vlSize - 1
                            tailN = (n - 1) % vlSize + 1
                            mLoops = (m + vlSize - 1) // vlSize - 1
                            tailM = (m - 1) % vlSize + 1
                            nPadding = (
                                (tailN + 31) // 32 * 32
                            )  # RoundUp(tailN, BLOCK_SIZE_IN_BYTE=32)

                            # UB 地址视图
                            ub_s = tla.make_tensor(
                                ubS_ptr,
                                tla.make_layout(
                                    tla.make_shape(m, 128), tla.make_stride(128, 1)
                                ),
                            )
                            ub_p = tla.make_tensor(
                                ubP_ptr,
                                tla.make_layout(
                                    tla.make_shape(65, 128), tla.make_stride(128, 1)
                                ),
                            )
                            nowMaxAddr = tla.make_tensor(
                                nowMax_ptr,
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
                            )
                            nowSumAddr = tla.make_tensor(
                                nowSum_ptr,
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
                            )
                            lastMaxAddr = tla.make_tensor(
                                lastMax_ptr,
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
                            )
                            lastSumAddr = tla.make_tensor(
                                lastSum_ptr,
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
                            )
                            expMaxUbAddr = tla.make_tensor(
                                expMax_ptr,
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
                            )
                            ub_mask = tla.make_tensor(
                                mask_ub_ptr,
                                tla.make_layout(
                                    tla.make_shape(m, 128), tla.make_stride(128, 1)
                                ),
                            )

                            maskr_ub = tla.make_tensor(
                                maskr_ub_ptr,
                                tla.make_layout(tla.make_shape(m), tla.make_stride(1)),
                            )

                            if is_move_mask and m != c0:
                                maskr_offset = (
                                    curBatch * maxQSeqlen_
                                    + qSTileIdx * qBaseTile_
                                    + subIdxEff * mHalf
                                )
                                gm_maskr = tla.make_tensor(
                                    maskr.ptr + maskr_offset,
                                    tla.make_layout(
                                        tla.make_shape(m), tla.make_stride(1)
                                    ),
                                )
                                mutex_maskr.lock(pipe=tla.arch.MTE2)
                                tla.copy(maskr_ub, gm_maskr)
                                mutex_maskr.unlock(pipe=tla.arch.MTE2)

                            if fineMaskBit:
                                mutex_maskr.lock(pipe=tla.arch.VECTOR)
                                if n > 64:
                                    with tla.vec.func(mode="simd"):
                                        pregFull0 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        pregTailN0, _ = tla.update_mask(
                                            tailN, dtype=tla.Float32
                                        )
                                        pos0 = tla.arange(kvSStartIdx, dtype=tla.Int32)
                                        pos1 = tla.arange(
                                            kvSStartIdx + 64, dtype=tla.Int32
                                        )
                                        for im in tla.range(m):
                                            ub_mask_i0m = tla.tile_view(
                                                ub_mask,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(im, c0),
                                            )
                                            ub_mask_i1m = tla.tile_view(
                                                ub_mask,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(im, c1),
                                            )
                                            maskr_vec = tla.tile_view(
                                                maskr_ub,
                                                tla.make_shape(1),
                                                tla.make_coord(im),
                                            ).load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            preg0 = tla.cmp(
                                                pos0, maskr_vec, "lt", mask=pregFull0
                                            )  # pos >= maskr -> 屏蔽
                                            preg1 = tla.cmp(
                                                pos1, maskr_vec, "lt", mask=pregFull0
                                            )
                                            preg1 = tla.bitwise_and(
                                                preg1, pregTailN0, mask=pregFull0
                                            )
                                            ub_mask_i0m.store(preg0, MaskStoreParams())
                                            ub_mask_i1m.store(preg1, MaskStoreParams())
                                else:
                                    with tla.vec.func(mode="simd"):
                                        pregFull0 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        pregTailN0, _ = tla.update_mask(
                                            tailN, dtype=tla.Float32
                                        )
                                        pos0 = tla.arange(kvSStartIdx, dtype=tla.Int32)
                                        for im in tla.range(m):
                                            ub_mask_i0m = tla.tile_view(
                                                ub_mask,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(im, c0),
                                            )
                                            maskr_vec = tla.tile_view(
                                                maskr_ub,
                                                tla.make_shape(1),
                                                tla.make_coord(im),
                                            ).load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            preg0 = tla.cmp(
                                                pos0, maskr_vec, "lt", mask=pregFull0
                                            )
                                            preg0 = tla.bitwise_and(
                                                preg0, pregTailN0, mask=pregFull0
                                            )
                                            ub_mask_i0m.store(preg0, MaskStoreParams())

                            # 等 QK Fixpipe 完成
                            if ubSBufId == 0:
                                tla.cross_core_wait_flag(
                                    mm1_ready_sm_0, tla.arch.VECTOR
                                )
                            else:
                                tla.cross_core_wait_flag(
                                    mm1_ready_sm_1, tla.arch.VECTOR
                                )
                            if ubSBufId == c0:
                                tla.wait_flag(mte3_ready_softmax_0)
                            else:
                                tla.wait_flag(mte3_ready_softmax_1)

                            ub_p_zN_full = tla.make_tensor_like(
                                ubP_ptr, ub_p, tla.arch.zNUnAlign
                            )
                            ub_p_zN = tla.tile_view(
                                ub_p_zN_full,
                                tla.make_shape(m, n),
                                tla.make_coord(c0, c0),
                            )
                            if isFirstKvSTile:
                                if n > 64:
                                    if fineMaskBit:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_s_i1 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c1),
                                                )
                                                ub_last_max_i = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_mask_i0 = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_mask_i1 = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c1),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg1 = ub_s_i1.load()
                                                mask_reg00 = ub_mask_i0.load(
                                                    MaskLoadParams()
                                                )
                                                mask_reg11 = ub_mask_i1.load(
                                                    MaskLoadParams()
                                                )
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg1 = tla.mul(
                                                    ub_s_reg1,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg0 = tla.where(
                                                    mask_reg00, ub_s_reg0, minVreg
                                                )
                                                ub_s_reg1 = tla.where(
                                                    mask_reg11, ub_s_reg1, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                                                max_tmp_reg = tla.max(
                                                    ub_s_reg0, ub_s_reg1, mask=pregFull
                                                )
                                                max_reg = max_tmp_reg.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_last_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_last_max_iDe = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_last_sum_iDe = tla.tile_view(
                                                    lastSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_s_i1De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c1),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_mask_i0De = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_mask_i1De = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c1),
                                                )
                                                max_regDe = ub_last_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_reg0De = ub_s_i0De.load()
                                                ub_s_reg1De = ub_s_i1De.load()
                                                mask_reg0 = ub_mask_i0De.load(
                                                    MaskLoadParams()
                                                )
                                                mask_reg1 = ub_mask_i1De.load(
                                                    MaskLoadParams()
                                                )
                                                ub_s_odd_reg = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg0De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                ub_s_even_reg = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg1De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_odd_reg0, exp_even_reg1 = (
                                                    tla.deinterleave(
                                                        ub_s_odd_reg, ub_s_even_reg
                                                    )
                                                )
                                                exp_sum_reg = tla.add(
                                                    exp_odd_reg0,
                                                    exp_even_reg1,
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_sum_reg.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_last_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_even_reg1.to(
                                                    DTYPE_P,
                                                    cast_trait_one,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg1 = exp_odd_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg = tla.bitwise_or(
                                                    exp_dst_reg0,
                                                    exp_dst_reg1,
                                                    mask=preg_all_b16,
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )
                                    else:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_s_i1 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c1),
                                                )
                                                ub_last_max_i = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg1 = ub_s_i1.load()
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg1 = tla.mul(
                                                    ub_s_reg1,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg1 = tla.where(
                                                    pregTailN, ub_s_reg1, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                                                max_tmp_reg = tla.max(
                                                    ub_s_reg0, ub_s_reg1, mask=pregFull
                                                )
                                                max_reg = max_tmp_reg.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_last_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_last_max_iDe = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_last_sum_iDe = tla.tile_view(
                                                    lastSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )

                                                max_regDe = ub_last_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_odd_reg, ub_s_even_reg = (
                                                    ub_s_i0De.load(
                                                        params=tla.params.NormalLoadParams(
                                                            load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                                        )
                                                    )
                                                )
                                                exp_odd_reg0 = tla.exp(
                                                    tla.sub(
                                                        ub_s_odd_reg,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_even_reg1 = tla.exp(
                                                    tla.sub(
                                                        ub_s_even_reg,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = tla.add(
                                                    exp_odd_reg0,
                                                    exp_even_reg1,
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_sum_reg.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_last_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )

                                                exp_dst_reg0 = exp_even_reg1.to(
                                                    DTYPE_P,
                                                    cast_trait_one,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg1 = exp_odd_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg = tla.bitwise_or(
                                                    exp_dst_reg0,
                                                    exp_dst_reg1,
                                                    mask=preg_all_b16,
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )
                                else:
                                    if fineMaskBit:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_last_max_i = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_mask_i0 = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                mask_reg00 = ub_mask_i0.load(
                                                    MaskLoadParams()
                                                )
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg0 = tla.where(
                                                    mask_reg00, ub_s_reg0, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                max_reg = ub_s_reg0.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_last_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_last_max_iDe = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_last_sum_iDe = tla.tile_view(
                                                    lastSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_mask_i0De = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                max_regDe = ub_last_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_reg0De = ub_s_i0De.load()
                                                mask_reg0 = ub_mask_i0De.load(
                                                    MaskLoadParams()
                                                )
                                                exp_reg0 = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg0De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_reg0.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_last_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg, zero_reg = (
                                                    tla.deinterleave(
                                                        exp_dst_reg0, exp_dst_reg0
                                                    )
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )
                                    else:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_last_max_i = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg0 = tla.where(
                                                    pregTailN, ub_s_reg0, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                max_reg = ub_s_reg0.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_last_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_last_max_iDe = tla.tile_view(
                                                    lastMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_last_sum_iDe = tla.tile_view(
                                                    lastSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                max_regDe = ub_last_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_reg0De = ub_s_i0De.load()
                                                exp_reg0 = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg0De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_reg0.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_last_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg, zero_reg = (
                                                    tla.deinterleave(
                                                        exp_dst_reg0, exp_dst_reg0
                                                    )
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )
                            else:
                                if n > 64:
                                    if fineMaskBit:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_s_i1 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c1),
                                                )
                                                ub_now_max_i = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_mask_i0 = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_mask_i1 = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c1),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg1 = ub_s_i1.load()
                                                mask_reg00 = ub_mask_i0.load(
                                                    MaskLoadParams()
                                                )
                                                mask_reg11 = ub_mask_i1.load(
                                                    MaskLoadParams()
                                                )
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg1 = tla.mul(
                                                    ub_s_reg1,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg0 = tla.where(
                                                    mask_reg00, ub_s_reg0, minVreg
                                                )
                                                ub_s_reg1 = tla.where(
                                                    mask_reg11, ub_s_reg1, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                                                max_tmp_reg = tla.max(
                                                    ub_s_reg0, ub_s_reg1, mask=pregFull
                                                )
                                                max_reg = max_tmp_reg.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_now_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            ub_last_max_i_de = tla.tile_view(
                                                lastMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_now_max_i_de = tla.tile_view(
                                                nowMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_last_sum_i_de = tla.tile_view(
                                                lastSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_dm_i_de = tla.tile_view(
                                                expMaxUbAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_max_reg_de = ub_now_max_i_de.load()
                                            last_max_reg_de = ub_last_max_i_de.load()
                                            last_sum_reg = ub_last_sum_i_de.load()
                                            max_reg_de = tla.max(
                                                now_max_reg_de,
                                                last_max_reg_de,
                                                mask=pregFull,
                                            )
                                            exp_sub_max_reg = tla.exp(
                                                tla.sub(
                                                    last_max_reg_de,
                                                    max_reg_de,
                                                    mask=pregFull,
                                                ),
                                                mask=pregFull,
                                            )
                                            update_exp_sub_reg = tla.mul(
                                                exp_sub_max_reg,
                                                last_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_now_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_last_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_dm_i_de.store(
                                                exp_sub_max_reg, mask=pregFull
                                            )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_now_max_iDe = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_now_sum_iDe = tla.tile_view(
                                                    nowSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_s_i1De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c1),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_mask_i0De = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_mask_i1De = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c1),
                                                )
                                                max_regDe = ub_now_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_reg0De = ub_s_i0De.load()
                                                ub_s_reg1De = ub_s_i1De.load()
                                                mask_reg0 = ub_mask_i0De.load(
                                                    MaskLoadParams()
                                                )
                                                mask_reg1 = ub_mask_i1De.load(
                                                    MaskLoadParams()
                                                )
                                                # masked 位置在 max pass 已置 MIN_VALUE，exp 用全 lane mask 依赖下溢置 0
                                                ub_s_odd_reg = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg0De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                ub_s_even_reg = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg1De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_odd_reg0, exp_even_reg1 = (
                                                    tla.deinterleave(
                                                        ub_s_odd_reg, ub_s_even_reg
                                                    )
                                                )
                                                exp_sum_reg = tla.add(
                                                    exp_odd_reg0,
                                                    exp_even_reg1,
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_sum_reg.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_now_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_even_reg1.to(
                                                    DTYPE_P,
                                                    cast_trait_one,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg1 = exp_odd_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg = tla.bitwise_or(
                                                    exp_dst_reg0,
                                                    exp_dst_reg1,
                                                    mask=preg_all_b16,
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )
                                            ub_now_sum_i_de = tla.tile_view(
                                                nowSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_sum_reg = ub_now_sum_i_de.load()
                                            update_exp_sub_reg = tla.add(
                                                update_exp_sub_reg,
                                                now_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_last_sum_i_de.store(
                                                update_exp_sub_reg, mask=pregFull
                                            )
                                    else:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_s_i1 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c1),
                                                )
                                                ub_now_max_i = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg1 = ub_s_i1.load()
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg1 = tla.mul(
                                                    ub_s_reg1,
                                                    scaleValue_,
                                                    mask=pregTailN,
                                                )
                                                ub_s_reg1 = tla.where(
                                                    pregTailN, ub_s_reg1, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                                                max_tmp_reg = tla.max(
                                                    ub_s_reg0, ub_s_reg1, mask=pregFull
                                                )
                                                max_reg = max_tmp_reg.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_now_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            ub_last_max_i_de = tla.tile_view(
                                                lastMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_now_max_i_de = tla.tile_view(
                                                nowMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_last_sum_i_de = tla.tile_view(
                                                lastSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_dm_i_de = tla.tile_view(
                                                expMaxUbAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_max_reg_de = ub_now_max_i_de.load()
                                            last_max_reg_de = ub_last_max_i_de.load()
                                            last_sum_reg = ub_last_sum_i_de.load()
                                            max_reg_de = tla.max(
                                                now_max_reg_de,
                                                last_max_reg_de,
                                                mask=pregFull,
                                            )
                                            exp_sub_max_reg = tla.exp(
                                                tla.sub(
                                                    last_max_reg_de,
                                                    max_reg_de,
                                                    mask=pregFull,
                                                ),
                                                mask=pregFull,
                                            )
                                            update_exp_sub_reg = tla.mul(
                                                exp_sub_max_reg,
                                                last_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_now_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_last_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_dm_i_de.store(
                                                exp_sub_max_reg, mask=pregFull
                                            )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_now_max_iDe = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_now_sum_iDe = tla.tile_view(
                                                    nowSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                max_regDe = ub_now_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_odd_reg, ub_s_even_reg = (
                                                    ub_s_i0De.load(
                                                        params=tla.params.NormalLoadParams(
                                                            load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                                        )
                                                    )
                                                )
                                                exp_odd_reg0 = tla.exp(
                                                    tla.sub(
                                                        ub_s_odd_reg,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_even_reg1 = tla.exp(
                                                    tla.sub(
                                                        ub_s_even_reg,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = tla.add(
                                                    exp_odd_reg0,
                                                    exp_even_reg1,
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_sum_reg.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_now_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_even_reg1.to(
                                                    DTYPE_P,
                                                    cast_trait_one,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg1 = exp_odd_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg = tla.bitwise_or(
                                                    exp_dst_reg0,
                                                    exp_dst_reg1,
                                                    mask=preg_all_b16,
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )
                                            ub_now_sum_i_de = tla.tile_view(
                                                nowSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_sum_reg = ub_now_sum_i_de.load()
                                            update_exp_sub_reg = tla.add(
                                                update_exp_sub_reg,
                                                now_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_last_sum_i_de.store(
                                                update_exp_sub_reg, mask=pregFull
                                            )
                                else:
                                    if fineMaskBit:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            pos0 = tla.arange(
                                                kvSStartIdx, dtype=tla.Int32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_now_max_i = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_mask_i0 = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                mask_reg00 = ub_mask_i0.load(
                                                    MaskLoadParams()
                                                )

                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )

                                                ub_s_reg0 = tla.where(
                                                    mask_reg00, ub_s_reg0, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                max_reg = ub_s_reg0.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_now_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            ub_last_max_i_de = tla.tile_view(
                                                lastMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_now_max_i_de = tla.tile_view(
                                                nowMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_last_sum_i_de = tla.tile_view(
                                                lastSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_dm_i_de = tla.tile_view(
                                                expMaxUbAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_max_reg_de = ub_now_max_i_de.load()
                                            last_max_reg_de = ub_last_max_i_de.load()
                                            last_sum_reg = ub_last_sum_i_de.load()
                                            max_reg_de = tla.max(
                                                now_max_reg_de,
                                                last_max_reg_de,
                                                mask=pregFull,
                                            )
                                            exp_sub_max_reg = tla.exp(
                                                tla.sub(
                                                    last_max_reg_de,
                                                    max_reg_de,
                                                    mask=pregFull,
                                                ),
                                                mask=pregFull,
                                            )
                                            update_exp_sub_reg = tla.mul(
                                                exp_sub_max_reg,
                                                last_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_now_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_last_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_dm_i_de.store(
                                                exp_sub_max_reg, mask=pregFull
                                            )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_now_max_iDe = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_now_sum_iDe = tla.tile_view(
                                                    nowSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_mask_i0De = tla.tile_view(
                                                    ub_mask,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                max_regDe = ub_now_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_reg0De = ub_s_i0De.load()
                                                mask_reg0 = ub_mask_i0De.load(
                                                    MaskLoadParams()
                                                )
                                                exp_reg0 = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg0De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_reg0.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_now_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg, zero_reg = (
                                                    tla.deinterleave(
                                                        exp_dst_reg0, exp_dst_reg0
                                                    )
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )

                                            ub_now_sum_i_de = tla.tile_view(
                                                nowSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_sum_reg = ub_now_sum_i_de.load()
                                            update_exp_sub_reg = tla.add(
                                                update_exp_sub_reg,
                                                now_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_last_sum_i_de.store(
                                                update_exp_sub_reg, mask=pregFull
                                            )

                                    else:
                                        with tla.vec.func(mode="simd"):
                                            pregFull = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float32
                                            )
                                            preg_all_b16 = tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=tla.Float16
                                            )
                                            pregTailN, _ = tla.update_mask(
                                                tailN, dtype=tla.Float32
                                            )
                                            one_mask, _ = tla.update_mask(
                                                1, dtype=tla.Float32
                                            )
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
                                            minVreg = tla.full(
                                                MIN_VALUE, dtype=tla.Float32
                                            )
                                            for i in tla.range(m):
                                                ub_s_i0 = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(i, c0),
                                                )
                                                ub_now_max_i = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(i),
                                                )
                                                ub_s_reg0 = ub_s_i0.load()
                                                ub_s_reg0 = tla.mul(
                                                    ub_s_reg0,
                                                    scaleValue_,
                                                    mask=pregFull,
                                                )
                                                ub_s_reg0 = tla.where(
                                                    pregTailN, ub_s_reg0, minVreg
                                                )
                                                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                                                max_reg = ub_s_reg0.reduce(
                                                    tla.ReductionOp.MAX, mask=pregFull
                                                )
                                                ub_now_max_i.store(
                                                    max_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            ub_last_max_i_de = tla.tile_view(
                                                lastMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_now_max_i_de = tla.tile_view(
                                                nowMaxAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_last_sum_i_de = tla.tile_view(
                                                lastSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            ub_dm_i_de = tla.tile_view(
                                                expMaxUbAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_max_reg_de = ub_now_max_i_de.load()
                                            last_max_reg_de = ub_last_max_i_de.load()
                                            last_sum_reg = ub_last_sum_i_de.load()
                                            max_reg_de = tla.max(
                                                now_max_reg_de,
                                                last_max_reg_de,
                                                mask=pregFull,
                                            )
                                            exp_sub_max_reg = tla.exp(
                                                tla.sub(
                                                    last_max_reg_de,
                                                    max_reg_de,
                                                    mask=pregFull,
                                                ),
                                                mask=pregFull,
                                            )
                                            update_exp_sub_reg = tla.mul(
                                                exp_sub_max_reg,
                                                last_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_now_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_last_max_i_de.store(
                                                max_reg_de, mask=pregFull
                                            )
                                            ub_dm_i_de.store(
                                                exp_sub_max_reg, mask=pregFull
                                            )
                                            tla.local_mem_bar(
                                                tla.params.MemType.VEC_STORE,
                                                tla.params.MemType.VEC_LOAD,
                                            )
                                            for j in tla.range(m):
                                                ub_now_max_iDe = tla.tile_view(
                                                    nowMaxAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_now_sum_iDe = tla.tile_view(
                                                    nowSumAddr,
                                                    tla.make_shape(1),
                                                    tla.make_coord(j),
                                                )
                                                ub_s_i0De = tla.tile_view(
                                                    ub_s,
                                                    tla.make_shape(1, _VL_F32),
                                                    tla.make_coord(j, c0),
                                                )
                                                ub_p_zN_f16_i = tla.tile_view(
                                                    ub_p_zN,
                                                    tla.make_shape(1, _VL_F16),
                                                    tla.make_coord(j, c0),
                                                )

                                                max_regDe = ub_now_max_iDe.load(
                                                    params=tla.params.NormalLoadParams(
                                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                    )
                                                )
                                                ub_s_reg0De = ub_s_i0De.load()
                                                exp_reg0 = tla.exp(
                                                    tla.sub(
                                                        ub_s_reg0De,
                                                        max_regDe,
                                                        mask=pregFull,
                                                    ),
                                                    mask=pregFull,
                                                )
                                                exp_sum_reg = exp_reg0.reduce(
                                                    tla.ReductionOp.ADD, mask=pregFull
                                                )
                                                ub_now_sum_iDe.store(
                                                    exp_sum_reg,
                                                    tla.params.UnalignStoreParams(),
                                                    mask=one_mask,
                                                )
                                                exp_dst_reg0 = exp_reg0.to(
                                                    DTYPE_P,
                                                    cast_trait_zero,
                                                    mask=pregFull,
                                                )
                                                exp_dst_reg, zero_reg = (
                                                    tla.deinterleave(
                                                        exp_dst_reg0, exp_dst_reg0
                                                    )
                                                )
                                                ub_p_zN_f16_i.store(
                                                    exp_dst_reg,
                                                    params=tla.params.BlockStoreParams(
                                                        block_stride=65
                                                    ),
                                                    mask=preg_all_b16,
                                                )

                                            ub_now_sum_i_de = tla.tile_view(
                                                nowSumAddr,
                                                tla.make_shape(_VL_F32),
                                                tla.make_coord(0),
                                            )
                                            now_sum_reg = ub_now_sum_i_de.load()
                                            update_exp_sub_reg = tla.add(
                                                update_exp_sub_reg,
                                                now_sum_reg,
                                                mask=pregFull,
                                            )
                                            ub_last_sum_i_de.store(
                                                update_exp_sub_reg, mask=pregFull
                                            )
                            # ------------- vf end ----------------------------
                            if ubSBufId == c0:
                                tla.set_flag(p_ub_ready_l1_0)
                                tla.wait_flag(p_ub_ready_l1_0)
                                tla.cross_core_set_flag(mm1_ready_sm_0, tla.arch.VECTOR)
                            else:
                                tla.set_flag(p_ub_ready_l1_1)
                                tla.wait_flag(p_ub_ready_l1_1)
                                tla.cross_core_set_flag(mm1_ready_sm_1, tla.arch.VECTOR)
                            if fineMaskBit:
                                mutex_maskr.unlock(pipe=tla.arch.VECTOR)

                            # P(UB) -> L1
                            curNRound = (
                                (n + FRACTAL_ALIGN - 1) // FRACTAL_ALIGN * FRACTAL_ALIGN
                            )
                            ub_p_tile = tla.tile_view(
                                ub_p, tla.make_shape(m, n), tla.make_coord(c0, c0)
                            )
                            l1P_tile = tla.tile_view(
                                l1PTensorTla,
                                tla.make_shape(mHalf, n),
                                tla.make_coord(subIdxEff, c0),
                            )

                            if l1PBufId == c0:
                                tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
                            elif l1PBufId == c1:
                                tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
                            else:
                                tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)

                            tla.copy(l1P_tile, ub_p_zN)  # CopyPUbToPL1

                            # 通知 P buffer 已写完可复用
                            if ubSBufId == c0:
                                tla.set_flag(mte3_ready_softmax_0)
                            else:
                                tla.set_flag(mte3_ready_softmax_1)
                            # 前向跨核通知 PV
                            if l1PBufId == c0:
                                tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE3)
                            elif l1PBufId == c1:
                                tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE3)
                            else:
                                tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE3)

                    if is_move_mask:
                        is_move_mask = tla.as_numeric(False)
                    if fineMaskBit and is_first:
                        is_first = tla.as_numeric(False)

                # ==================== 后半 idx>=PRE_LAUNCH：PV(cube) + rescale O(vector) ====================
                if gatheredKvSTileIdx >= PRE_LAUNCH and cmp and launch_idx_2 >= 0:
                    gatheredKvSTileIdxDe = launch_idx_2
                    validIdxDe = validIdx - 2
                    if gatheredKvSTileIdxDe == kvSLoopNum - c1:
                        kvSTileSizeActDe = (
                            gatheredKvSeqlen - gatheredKvSTileIdxDe * kvBaseTile_
                        )
                    else:
                        kvSTileSizeActDe = tla.as_numeric(kvBaseTile_)
                    isFirstKvSTileDe = validIdxDe == c0
                    isLastKvSTileDe = gatheredKvSTileIdxDe == kvSLoopNum - c1
                    ubSBufIdDe = validIdxDe % UB_S_OTMP_BUF_STAGES
                    ubS_ptrDe = ubS_ptrs[0] if ubSBufIdDe == c0 else ubS_ptrs[1]

                    ubSTensorTlaDe = tla.make_tensor(
                        ubS_ptrDe,
                        tla.make_layout(
                            tla.make_shape(rowNum, kvSTileSizeActDe),
                            tla.make_stride(128, 1),
                        ),
                    )
                    ubOTmpBufId = validIdxDe % UB_S_OTMP_BUF_STAGES
                    ubOTmp_ptr = ubOTmp_ptrs[0] if ubOTmpBufId == c0 else ubOTmp_ptrs[1]
                    ubOTmpTensorTla = tla.make_tensor(
                        ubOTmp_ptr,
                        tla.make_layout(
                            tla.make_shape(rowNum, embed_), tla.make_stride(128, 1)
                        ),
                    )
                    l1PBufIdDe = validIdxDe % P_L1_BUF
                    l1P_ptrDe = (
                        l1P_ptrs[0]
                        if l1PBufIdDe == c0
                        else (l1P_ptrs[1] if l1PBufIdDe == c1 else l1P_ptrs[2])
                    )
                    l1PTensorTlaDe = tla.make_tensor_like(
                        l1P_ptrDe,
                        ubSTensorTlaDe,
                        tla.arch.zN,
                    )

                    # PV Mmad
                    with tla.cube():
                        kvShapeRowDe = curKvSeqlen
                        kvShapeColDe = tla.as_numeric(embed_)
                        if tla.const_expr(paged):
                            phys_v = blockTable[
                                curBatch * max_blocks_per_batch + gatheredKvSTileIdxDe
                            ]
                            v_tile_base = (
                                value.ptr
                                + (
                                    phys_v * kvHeads_ * kvBaseTile_
                                    + kvHeadIdx * kvBaseTile_
                                )
                                * embed_
                            )
                        else:
                            gm_v = tla.make_tensor(
                                value.ptr + gmOffsetV,
                                tla.make_layout(
                                    tla.make_shape(kvShapeRowDe, kvShapeColDe),
                                    tla.make_stride(vSOffset, 1),
                                    layoutTag=tla.arch.RowMajor,
                                ),
                            )
                        # 跨相 L0A/L0B 前缀和
                        prefixSumL0AStagesDe = (
                            (
                                (validIdxDe + c1 + PRE_LAUNCH) * mm1L0ATotalStages_
                                + validIdxDe * mm2L0ATotalStages_
                            )
                            if gatheredKvSTileIdx < kvSLoopNum - PRE_LAUNCH
                            else (
                                (validNum) * mm1L0ATotalStages_
                                + validIdxDe * mm2L0ATotalStages_
                            )
                        )
                        prefixSumL0BStagesDe = (
                            (
                                (validIdxDe + c1 + PRE_LAUNCH) * mm1L0BTotalStages_
                                + validIdxDe * mm2L0BTotalStages_
                            )
                            if gatheredKvSTileIdx < kvSLoopNum - PRE_LAUNCH
                            else (
                                (validNum) * mm1L0BTotalStages_
                                + validIdxDe * mm2L0BTotalStages_
                            )
                        )
                        # L1 buffer 选择
                        l1BvBufId = (
                            validIdxDe % V_L1_BUF
                        )  # l1BBufId = idxDe % vL1BufNum
                        l1V_ptr = l1V_ptrs[0] if l1BvBufId == c0 else l1V_ptrs[1]
                        if tla.const_expr(paged):
                            gm_v_tile = tla.make_tensor(
                                v_tile_base,
                                tla.make_layout(
                                    tla.make_shape(kvBaseTile_, embed_),
                                    tla.make_stride(embed_, 1),
                                    layoutTag=tla.arch.RowMajor,
                                ),
                            )
                        else:
                            gm_v_tile = tla.tile_view(
                                gm_v,
                                tla.make_shape(kvBaseTile_, embed_),
                                tla.make_coord(gatheredKvSTileIdxDe, c0),
                            )
                        l1_v_tile = tla.make_tensor_like(
                            l1V_ptr, gm_v_tile, layoutTag=tla.arch.zN
                        )
                        if l1BvBufId == c0:
                            tla.wait_flag(v_l0b_ready_l1_0)  # MTE1_MTE2
                        else:
                            tla.wait_flag(v_l0b_ready_l1_1)
                        tla.copy(l1_v_tile, gm_v_tile)
                        if l1BvBufId == c0:
                            tla.set_flag(v_l1_ready_l0_0)  # MTE2_MTE1
                            tla.wait_flag(v_l1_ready_l0_0)  # MTE2_MTE1
                        else:
                            tla.set_flag(v_l1_ready_l0_1)
                            tla.wait_flag(v_l1_ready_l0_1)
                        if l1PBufIdDe == c0:
                            tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE1)
                        elif l1PBufIdDe == c1:
                            tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE1)
                        else:
                            tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE1)
                        # copy L1 to l0
                        l0TileNActDe = tla.as_numeric(embed_)
                        nLoopCounter = validIdxDe
                        l0CBufIdDe = nLoopCounter % L0_STAGES  # l0C 只按 n 分 buffer
                        l0c_ptrDe = l0c_ptrs[2] if l0CBufIdDe == c0 else l0c_ptrs[3]
                        ub_o_tile = tla.tile_view(
                            ubOTmpTensorTla,
                            tla.make_shape(qBaseTile_, embed_),
                            tla.make_coord(c0, c0),
                        )
                        l0c_o = tla.make_tensor_like(
                            l0c_ptrDe, ub_o_tile, layoutTag=tla.arch.L0Clayout
                        )
                        l0TileMActDe = rowNum

                        # L0A/L0B buffer id = (跨相前缀和 + 本 mmad 内 stage 号) % L0_STAGES
                        l0ALoopCounterDe = prefixSumL0AStagesDe
                        l0BLoopCounterDe = prefixSumL0BStagesDe
                        l0TileKActDe = kvSTileSizeActDe
                        l0ABufIdDe = l0ALoopCounterDe % L0_STAGES
                        l0BBufIdDe = l0BLoopCounterDe % L0_STAGES
                        # V: L1 -> L0B
                        l0b_ptrDe = l0b_ptrs[0] if l0BBufIdDe == c0 else l0b_ptrs[1]
                        l0_b2 = tla.make_tensor_like(l0b_ptrDe, l1_v_tile, tla.arch.nZ)
                        if l0BBufIdDe == c0:
                            tla.wait_flag(mmad_ready_l0b_0)  # M_MTE1
                        else:
                            tla.wait_flag(mmad_ready_l0b_1)
                        tla.copy(l0_b2, l1_v_tile)  # copyL1ToL0B
                        if l0BBufIdDe == c0:
                            tla.set_flag(l0b_ready_mmad_0)  # MTE1_M
                        else:
                            tla.set_flag(l0b_ready_mmad_1)
                        if l1BvBufId == c0:
                            tla.set_flag(v_l0b_ready_l1_0)  # MTE1_MTE2
                        else:
                            tla.set_flag(v_l0b_ready_l1_1)
                        l1_p_l0 = tla.tile_view(
                            l1PTensorTlaDe,
                            tla.make_shape(128, 128),
                            tla.make_coord(c0, c0),
                        )
                        l0a_ptrDe = l0a_ptrs[0] if l0ABufIdDe == c0 else l0a_ptrs[1]
                        l0_a2 = tla.make_tensor_like(
                            l0a_ptrDe, l1PTensorTlaDe, tla.arch.zN
                        )
                        if l0ABufIdDe == c0:
                            tla.wait_flag(mmad_ready_l0a_0)  # M_MTE1
                        else:
                            tla.wait_flag(mmad_ready_l0a_1)
                        tla.copy(l0_a2, l1_p_l0)  # copyL1ToL0A
                        if l0ABufIdDe == c0:
                            tla.set_flag(l0a_ready_mmad_0)  # MTE1_M
                        else:
                            tla.set_flag(l0a_ready_mmad_1)
                        if l1PBufIdDe == c0:
                            tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1)
                        elif l1PBufIdDe == c1:
                            tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1)
                        else:
                            tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1)

                        l0TileMAligned = (
                            (l0TileMActDe + FRACTAL_ALIGN - 1)
                            // FRACTAL_ALIGN
                            * FRACTAL_ALIGN
                        )
                        if l0ABufIdDe == c0:
                            tla.wait_flag(l0a_ready_mmad_0)  # MTE1_M
                        else:
                            tla.wait_flag(l0a_ready_mmad_1)
                        if l0BBufIdDe == c0:
                            tla.wait_flag(l0b_ready_mmad_0)  # MTE1_M
                        else:
                            tla.wait_flag(l0b_ready_mmad_1)
                        if l0CBufIdDe == c0:
                            tla.wait_flag(
                                fix_ready_mmad_2
                            )  # WaitFlag(FIX_M, l0CEventId=l0CBufIdDe+2)
                        else:
                            tla.wait_flag(fix_ready_mmad_3)
                        tla.mmad(l0c_o, l0_a2, l0_b2, init_c=True)
                        if l0ABufIdDe == c0:
                            tla.set_flag(mmad_ready_l0a_0)  # M_MTE1
                        else:
                            tla.set_flag(mmad_ready_l0a_1)
                        if l0BBufIdDe == c0:
                            tla.set_flag(mmad_ready_l0b_0)  # M_MTE1
                        else:
                            tla.set_flag(mmad_ready_l0b_1)
                        if ubOTmpBufId == c0:
                            tla.cross_core_wait_flag(mm2_ready_re_0, tla.arch.FIX)
                        else:
                            tla.cross_core_wait_flag(mm2_ready_re_1, tla.arch.FIX)
                        if l0CBufIdDe == c0:
                            tla.set_flag(
                                mmad_ready_fix_2
                            )  # SetFlag(M_FIX, l0CEventId=l0CBufIdDe+2)
                            tla.wait_flag(mmad_ready_fix_2)  # M_FIX
                        else:
                            tla.set_flag(mmad_ready_fix_3)
                            tla.wait_flag(mmad_ready_fix_3)
                        tla.copy(
                            ubOTmpTensorTla,
                            l0c_o,
                            tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ),
                        )
                        if l0CBufIdDe == c0:
                            tla.set_flag(fix_ready_mmad_2)  # FIX_M
                        else:
                            tla.set_flag(fix_ready_mmad_3)
                        if ubOTmpBufId == c0:
                            tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.FIX)
                        else:
                            tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.FIX)

                    # rescale O
                    with tla.vector():
                        oShapeColDe = tla.as_numeric(embed_)
                        gm_o = tla.make_tensor(
                            attentionOut.ptr + qBOffset + qHeadIdx * qNOffset,
                            tla.make_layout(
                                tla.make_shape(curQSeqlen, oShapeColDe),
                                tla.make_stride(oSOffset, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        # operator() 标量计算
                        rowNumOri = rowNum
                        colNumOri = tla.as_numeric(embed_)
                        subBlockIdxDe = tla.arch.sub_block_idx()
                        subIdxEffDe = subBlockIdxDe if rowNumOri > 1 else c0
                        subBlockNum = tla.as_numeric(2)
                        colNumOriAligned8 = (
                            (colNumOri + 7) // 8 * 8
                        )  # RoundUp(colNumOri, 8)
                        rowNumSplit = (rowNumOri + 1) // subBlockNum
                        rowNumSplit = (
                            rowNumOri if rowNumOri < rowNumSplit else rowNumSplit
                        )
                        rowNumCurSubCore = (
                            rowNumSplit
                            if subIdxEffDe == c0
                            else (rowNumOri - rowNumSplit)
                        )
                        rowOffsetCurSubCore = rowNumSplit * subIdxEffDe
                        colNumCurSubCore = colNumOri
                        colStrideCurSubCore = colNumOriAligned8
                        gmO_tile = tla.tile_view(
                            gm_o,
                            tla.make_shape(qBaseTile_, colNumCurSubCore),
                            tla.make_coord(qSTileIdx, c0),
                        )
                        ubOTmpBufId = validIdxDe % UB_S_OTMP_BUF_STAGES

                        curTileMod = validIdxDe % P_L1_BUF
                        expMax_ptrDe = (
                            expMax_ptrs[0]
                            if curTileMod == c0
                            else (
                                expMax_ptrs[1] if curTileMod == c1 else expMax_ptrs[2]
                            )
                        )
                        # SubCoreCompute
                        if rowNumCurSubCore == c0:
                            if ubOTmpBufId == c0:
                                tla.cross_core_wait_flag(
                                    mm2_ready_re_0, tla.arch.VECTOR
                                )
                                tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.VECTOR)
                            else:
                                tla.cross_core_wait_flag(
                                    mm2_ready_re_1, tla.arch.VECTOR
                                )
                                tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.VECTOR)
                        else:
                            # SubCoreCompute 标量参数
                            mDe = rowNumCurSubCore
                            nDe = colNumCurSubCore
                            mRound = (
                                (mDe + FRACTAL_ALIGN - 1)
                                // FRACTAL_ALIGN
                                * FRACTAL_ALIGN
                            )
                            nRound = (
                                (nDe + FRACTAL_ALIGN - 1)
                                // FRACTAL_ALIGN
                                * FRACTAL_ALIGN
                            )
                            vlSizeDe = _VL_F32
                            colFullLoop = (nDe + vlSizeDe - 1) // vlSizeDe - 1
                            colTail = (nDe - 1) % vlSizeDe + 1
                            # UB 地址视图
                            loUb = tla.make_tensor(
                                ubOTmp_ptr,
                                tla.make_layout(
                                    tla.make_shape(mDe, 128), tla.make_stride(128, 1)
                                ),
                            )
                            goUb = tla.make_tensor(
                                ubO_ptr,
                                tla.make_layout(
                                    tla.make_shape(mDe, 128), tla.make_stride(128, 1)
                                ),
                            )
                            goUb16 = tla.make_tensor(
                                ubO16_ptr,
                                tla.make_layout(
                                    tla.make_shape(mDe, 128), tla.make_stride(128, 1)
                                ),
                            )
                            dmUb = tla.make_tensor(
                                expMax_ptrDe,
                                tla.make_layout(
                                    tla.make_shape(mDe), tla.make_stride(1)
                                ),
                            )
                            glUb = tla.make_tensor(
                                lastSum_ptr,
                                tla.make_layout(
                                    tla.make_shape(mDe), tla.make_stride(1)
                                ),
                            )
                            # 等 PV fixpipe 完成
                            if ubOTmpBufId == c0:
                                tla.cross_core_wait_flag(
                                    mm2_ready_re_0, tla.arch.VECTOR
                                )
                            else:
                                tla.cross_core_wait_flag(
                                    mm2_ready_re_1, tla.arch.VECTOR
                                )
                            tla.wait_flag(mte3_ready_rescale)
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
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i0, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i0, c0),
                                            )
                                            ub_go_1 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i0, c1),
                                            )
                                            ub_gl = tla.tile_view(
                                                glUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i0),
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
                                            div_reg_0 = tla.div(
                                                lo_reg_0, gl_reg, mask=pregFullDee
                                            )
                                            div_reg_1 = tla.div(
                                                lo_reg_1, gl_reg, mask=pregFullDee
                                            )
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
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i0, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i0, c0),
                                            )
                                            ub_gl = tla.tile_view(
                                                glUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i0),
                                            )
                                            lo_reg_0 = ub_lo_0.load()
                                            gl_reg = ub_gl.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            div_reg_0 = tla.div(
                                                lo_reg_0, gl_reg, mask=pregFullDee
                                            )
                                            ub_go_0.store(div_reg_0, mask=pregFullDee)
                            elif isFirstKvSTileDe and (not isLastKvSTileDe):
                                if nDe > 64:
                                    with tla.vec.func(mode="simd"):
                                        pregFullDee1 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        for i1 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i1, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i1, c0),
                                            )
                                            ub_go_1 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i1, c1),
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
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i1, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i1, c0),
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
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i2, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i2, c0),
                                            )
                                            ub_go_1 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i2, c1),
                                            )
                                            ub_gl = tla.tile_view(
                                                glUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i2),
                                            )
                                            ub_dm = tla.tile_view(
                                                dmUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i2),
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
                                            mul_reg_0 = tla.mul(
                                                go_reg_0, dm_reg, mask=pregFullDee2
                                            )
                                            mul_reg_1 = tla.mul(
                                                go_reg_1, dm_reg, mask=pregFullDee2
                                            )
                                            add_reg_0 = tla.add(
                                                mul_reg_0, lo_reg_0, mask=pregFullDee2
                                            )
                                            add_reg_1 = tla.add(
                                                mul_reg_1, lo_reg_1, mask=pregFullDee2
                                            )
                                            div_reg_0 = tla.div(
                                                add_reg_0, gl_reg, mask=pregFullDee2
                                            )
                                            div_reg_1 = tla.div(
                                                add_reg_1, gl_reg, mask=pregFullDee2
                                            )
                                            ub_go_0.store(div_reg_0, mask=pregFullDee2)
                                            ub_go_1.store(div_reg_1, mask=pregFullDee2)
                                else:
                                    with tla.vec.func(mode="simd"):
                                        pregFullDee2 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        for i2 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i2, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i2, c0),
                                            )
                                            ub_gl = tla.tile_view(
                                                glUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i2),
                                            )
                                            ub_dm = tla.tile_view(
                                                dmUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i2),
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
                                            mul_reg_0 = tla.mul(
                                                go_reg_0, dm_reg, mask=pregFullDee2
                                            )
                                            add_reg_0 = tla.add(
                                                mul_reg_0, lo_reg_0, mask=pregFullDee2
                                            )
                                            div_reg_0 = tla.div(
                                                add_reg_0, gl_reg, mask=pregFullDee2
                                            )
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
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i3, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i3, c0),
                                            )
                                            ub_go_1 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i3, c1),
                                            )
                                            ub_dm = tla.tile_view(
                                                dmUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i3),
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
                                            mul_reg_0 = tla.mul(
                                                go_reg_0, dm_reg, mask=pregFullDee3
                                            )
                                            mul_reg_1 = tla.mul(
                                                go_reg_1, dm_reg, mask=pregFullDee3
                                            )
                                            add_reg_0 = tla.add(
                                                mul_reg_0, lo_reg_0, mask=pregFullDee3
                                            )
                                            add_reg_1 = tla.add(
                                                mul_reg_1, lo_reg_1, mask=pregFullDee3
                                            )
                                            ub_go_0.store(add_reg_0, mask=pregFullDee3)
                                            ub_go_1.store(add_reg_1, mask=pregFullDee3)
                                else:
                                    with tla.vec.func(mode="simd"):
                                        pregFullDee3 = tla.create_mask(
                                            pattern=tla.mask.ALL, dtype=tla.Float32
                                        )
                                        for i3 in tla.range(rowNumCurSubCore):
                                            ub_lo_0 = tla.tile_view(
                                                loUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i3, c0),
                                            )
                                            ub_go_0 = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i3, c0),
                                            )
                                            ub_dm = tla.tile_view(
                                                dmUb,
                                                tla.make_shape(1),
                                                tla.make_coord(i3),
                                            )
                                            lo_reg_0 = ub_lo_0.load()
                                            go_reg_0 = ub_go_0.load()
                                            dm_reg = ub_dm.load(
                                                params=tla.params.NormalLoadParams(
                                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                                )
                                            )
                                            mul_reg_0 = tla.mul(
                                                go_reg_0, dm_reg, mask=pregFullDee3
                                            )
                                            add_reg_0 = tla.add(
                                                mul_reg_0, lo_reg_0, mask=pregFullDee3
                                            )
                                            ub_go_0.store(add_reg_0, mask=pregFullDee3)

                            if ubOTmpBufId == c0:
                                tla.cross_core_set_flag(mm2_ready_re_0, tla.arch.VECTOR)
                            else:
                                tla.cross_core_set_flag(mm2_ready_re_1, tla.arch.VECTOR)
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
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i4, c0),
                                            )
                                            ub_go_1_de = tla.tile_view(
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i4, c1),
                                            )
                                            ub_go_b16 = tla.tile_view(
                                                goUb16,
                                                tla.make_shape(1, _VL_F16),
                                                tla.make_coord(i4, c0),
                                            )
                                            go_reg_0_de = ub_go_0_de.load()
                                            go_reg_1_de = ub_go_1_de.load()
                                            out_reg0 = go_reg_0_de.to(
                                                DTYPE_Q,
                                                cast_trait_zero_de,
                                                mask=pregFullDee4,
                                            )
                                            out_reg1 = go_reg_1_de.to(
                                                DTYPE_Q,
                                                cast_trait_one_de,
                                                mask=pregFullDee4,
                                            )
                                            lo_reg_0_de = tla.bitwise_or(
                                                out_reg0, out_reg1, mask=pregAll_b16
                                            )
                                            ub_go_b16.store(
                                                lo_reg_0_de, mask=pregAll_b16
                                            )
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
                                                goUb,
                                                tla.make_shape(1, _VL_F32),
                                                tla.make_coord(i4, c0),
                                            )
                                            ub_go_b16 = tla.tile_view(
                                                goUb16,
                                                tla.make_shape(1, _VL_F16),
                                                tla.make_coord(i4, c0),
                                            )
                                            go_reg_0_de = ub_go_0_de.load()
                                            go_reg_0_de = go_reg_0_de.to(
                                                DTYPE_Q,
                                                cast_trait_zero_de,
                                                mask=pregFullDee4,
                                            )
                                            lo_reg_0_de, zero_reg = tla.deinterleave(
                                                go_reg_0_de, go_reg_0_de
                                            )
                                            ub_go_b16.store(
                                                lo_reg_0_de, mask=pregAll_b16
                                            )
                                tla.set_flag(p_ub_ready_l1_0)
                                tla.wait_flag(p_ub_ready_l1_0)
                                gm_out_tile = tla.tile_view(
                                    gmO_tile,
                                    tla.make_shape(rowNumSplit, nDe),
                                    tla.make_coord(subBlockIdxDe, c0),
                                )
                                tla.copy(gm_out_tile, goUb16)
                            tla.set_flag(mte3_ready_rescale)
                cmp = tla.as_numeric(True)
    # release
    with tla.cube():
        tla.wait_flag(q_l0a_ready_l1)
        tla.wait_flag(k_l0b_ready_l1_0)
        tla.wait_flag(k_l0b_ready_l1_1)
        tla.wait_flag(v_l0b_ready_l1_0)
        tla.wait_flag(v_l0b_ready_l1_1)
        tla.wait_flag(mmad_ready_l0a_0)
        tla.wait_flag(mmad_ready_l0a_1)
        tla.wait_flag(mmad_ready_l0b_0)
        tla.wait_flag(mmad_ready_l0b_1)
        tla.wait_flag(fix_ready_mmad_0)
        tla.wait_flag(fix_ready_mmad_1)
        tla.wait_flag(fix_ready_mmad_2)
        tla.wait_flag(fix_ready_mmad_3)
        tla.cross_core_wait_flag(mm1_ready_sm_0, tla.arch.FIX)
        tla.cross_core_wait_flag(mm1_ready_sm_1, tla.arch.FIX)
        tla.cross_core_wait_flag(mm2_ready_re_0, tla.arch.FIX)
        tla.cross_core_wait_flag(mm2_ready_re_1, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla.wait_flag(vec_ready_mte2_0)
        tla.wait_flag(vec_ready_mte2_1)
        tla.wait_flag(mte3_ready_mask_0)
        tla.wait_flag(mte3_ready_mask_1)
        tla.wait_flag(mte3_ready_softmax_0)
        tla.wait_flag(mte3_ready_softmax_1)
        tla.wait_flag(mte3_ready_rescale)
        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)

