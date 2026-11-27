# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""FlashAttentionScore (batch invariant) implemented with the CATLASS Python DSL (tla).

Computes the training-style forward attention score triple
``(attention_out, softmax_max, softmax_sum)``, where ``softmax_max`` /
``softmax_sum`` are the per-row FP32 online-softmax statistics broadcast to
8 columns, as consumed by the backward attention kernels.

Single-file adaptation of the ``tla_dsl/tla/kernel/attention/flash_attention_infer``
example, with ``softmax_max`` / ``softmax_sum`` outputs, a dense GM mask path
(``sparse_mode=1``), head-major BNSD addressing and a guarded softmax
denominator (fully masked rows output exactly 0) added on top.

Layouts (``input_layout``):
    0 = BNSD   -> head-major addressing, kernel reads/writes [B, N, S, D]
    1 = BSH    -> packed rows already match the BSND kernel stride model
    2 = TND    -> varlen packed rows (qFormat=0)
    3 = BSND   -> native

``sparse_mode``: 0 no mask, 1 dense ``atten_mask`` from GM, 2 leftUpCausal,
3 rightDownCausal.

Batch invariance: task partitioning is per ``(batch, head, q_block)`` and every
reduction (max/sum) stays inside a single row's KV extent — no cross-batch
normalization or shared scratch — so running ``B=N`` yields bit-identical
per-batch slices compared to ``N`` independent ``B=1`` runs; TND packed runs
are likewise bit-identical to unpacked per-sequence runs.

Module layout (top to bottom, each covered by a section comment below):
    fa_common      shared constants and the FaCtx / FaTask / FaBufs / FaSync records
    fa_tiling      FATilingData, compute_tiling, pack_tiling_int, make_actual_seqlen
    stage_qk       tiling load, flag/buffer setup, Q tile load, QK MMAD, score store-out
    stage_softmax  masking, row max/sum, rescale factors, per-row online-softmax stats
    stage_pv       P/V load and PV MMAD accumulation
    stage_rescale  output rescale/normalization, softmax_max / softmax_sum write-back
    kernel         flash_attention_score_kernel（sparse_mode 编译期特化，掩码不生成多余代码）
    host side      compile cache, mask packing, kernel launch, output restore
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import sys
import weakref
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

import catlass.tla as tla  # noqa: E402
from catlass.base_dsl.typing import Float32, Int32, Numeric, Pointer  # noqa: E402
from catlass.params import MaskLoadParams, MaskStoreParams  # noqa: E402
from catlass.tla.runtime import from_dlpack  # noqa: E402
from catlass.types import TlaCrossFlag, TlaFlag  # noqa: E402


# ---------------------------------------------------------------------------
# fa_common.py — 公共常量与 FaCtx / FaTask / FaBufs / FaSync 定义
# ---------------------------------------------------------------------------

# ---------------- 公共常量 ----------------
PRE_LAUNCH = 2
UB_S_OTMP_BUF_STAGES = 2
P_L1_BUF = 3
L0_STAGES = 2
VL_F32 = 64
VL_F16 = 128


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------- 参数打包 ----------------
@dataclass
class FaCtx:
    qHeads_: Int32
    kvHeads_: Int32
    maxQSeqlen_: Int32
    maxKvSeqlen_: Int32
    kvSeqlen_: Int32
    qFormat_: Int32
    kvFormat_: Int32
    scaleValue_: Float32
    mm1L0ATotalStages_: int
    mm1L0BTotalStages_: int
    mm2L0ATotalStages_: int
    mm2L0BTotalStages_: int
    firstBatchTaskNum_: Int32
    totalTaskNum_: Int32
    embed_: int
    embedV_: int
    qBaseTile_: int
    kvBaseTile_: int
    groupSize: Int32
    qNOffset: int
    kNOffset: int
    vNOffset: int
    oNOffset: int
    qSOffset: Int32
    kSOffset: Int32
    vSOffset: Int32
    oSOffset: Int32
    sNOffset: int = 8
    ubStages_: int = 2
    sSOffset: Int32 = 8
    statsBnsd_: Int32 = 0
    DTYPE_Q: type[Numeric] | None = None
    DTYPE_K: type[Numeric] | None = None
    DTYPE_P: type[Numeric] | None = None
    DTYPE_V: type[Numeric] | None = None


@dataclass
class FaTask:
    """task 级运行期标量；调用点按需构造部分字段，故保留 None 默认。"""

    curBatch: Int32 | None = None
    qHeadIdx: Int32 | None = None
    kvHeadIdx: Int32 | None = None
    qSTileIdx: Int32 | None = None
    qBOffset: Int32 | None = None
    kBOffset: Int32 | None = None
    vBOffset: Int32 | None = None
    oBOffset: Int32 | None = None
    curQSeqlen: Int32 | None = None
    curKvSeqlen: Int32 | None = None
    rowNum: Int32 | None = None
    gmOffsetV: Int32 | None = None


@dataclass
class FaBufs:
    l1Q_ptrs: list[Pointer]
    l1K_ptrs: list[Pointer]
    l1P_ptrs: list[Pointer]
    l1V_ptrs: list[Pointer]
    l0a_ptrs: list[Pointer]
    l0b_ptrs: list[Pointer]
    l0c_ptrs: list[Pointer]
    ubS_ptrs: list[Pointer]
    ubP_ptrs: list[Pointer]
    ubOTmp_ptrs: list[Pointer]
    ubO_ptr: Pointer
    ubO16_ptr: Pointer
    nowMax_ptr: Pointer
    expMax_ptrs: list[Pointer]
    nowSum_ptr: Pointer
    lastMax_ptr: Pointer
    lastSum_ptr: Pointer
    ubStatMax_ptr: Pointer
    ubStatSum_ptr: Pointer


@dataclass
class FaSync:
    q_l0a_data_ready: TlaFlag
    k_l0b_data_ready_0: TlaFlag
    k_l0b_data_ready_1: TlaFlag
    v_l0b_data_ready_0: TlaFlag
    v_l0b_data_ready_1: TlaFlag
    l0a_available_0: TlaFlag
    l0a_available_1: TlaFlag
    l0b_available_0: TlaFlag
    l0b_available_1: TlaFlag
    l0c_available_0: TlaFlag
    l0c_available_1: TlaFlag
    l0c_available_2: TlaFlag
    l0c_available_3: TlaFlag
    q_l1_data_ready: TlaFlag
    k_l1_data_ready_0: TlaFlag
    k_l1_data_ready_1: TlaFlag
    v_l1_data_ready_0: TlaFlag
    v_l1_data_ready_1: TlaFlag
    l0a_data_ready_0: TlaFlag
    l0a_data_ready_1: TlaFlag
    l0b_data_ready_0: TlaFlag
    l0b_data_ready_1: TlaFlag
    l0c_data_ready_0: TlaFlag
    l0c_data_ready_1: TlaFlag
    l0c_data_ready_2: TlaFlag
    l0c_data_ready_3: TlaFlag
    vec_ready_mte2_0: TlaFlag
    vec_ready_mte2_1: TlaFlag
    mte3_ready_mask_0: TlaFlag
    mte3_ready_mask_1: TlaFlag
    ub_p_available_0: TlaFlag
    ub_p_available_1: TlaFlag
    mte3_ready_rescale: TlaFlag
    ub_p_data_ready_0: TlaFlag
    ub_p_data_ready_1: TlaFlag
    ub_s_cross_ready_0: TlaCrossFlag
    ub_s_cross_ready_1: TlaCrossFlag
    ub_otmp_cross_ready_0: TlaCrossFlag
    ub_otmp_cross_ready_1: TlaCrossFlag
    ub_p_cross_ready_0: TlaCrossFlag
    ub_p_cross_ready_1: TlaCrossFlag
    ub_p_cross_ready_2: TlaCrossFlag


# ---------------------------------------------------------------------------
# fa_tiling.py — host 侧 tiling 计算与 int32 打包
# ---------------------------------------------------------------------------

Q_BASE_TILE = 128
KV_BASE_TILE = 128

QK_L1_TILE_M = 128
QK_L1_TILE_N = 256
QK_L1_TILE_K_LEFT = 192
QK_L1_TILE_K_RIGHT = 192

PV_L1_TILE_M = 128
PV_L1_TILE_N = 128
PV_L1_TILE_K_LEFT = 256
PV_L1_TILE_K_RIGHT = 128

Q_L1_BUF_NUM = 1
K_L1_BUF_NUM = 2
V_L1_BUF_NUM = 2
P_L1_BUF_NUM = 3


@dataclass
class FATilingData:
    batch: int = 0
    num_heads: int = 0
    kv_heads: int = 0
    max_q_seqlen: int = 0
    max_kv_seqlen: int = 0
    first_batch_task_num: int = 0
    total_task_num: int = 0
    head_dim: int = 128
    group_size: int = 0


def compute_tiling(
    batch: int,
    num_heads: int,
    kv_heads: int,
    q_seqlen_list: List[int],
    kv_seqlen_list: List[int],
    head_dim: int = 128,
) -> FATilingData:
    td = FATilingData()

    td.batch = batch
    td.num_heads = num_heads
    td.kv_heads = kv_heads
    td.head_dim = head_dim
    td.group_size = num_heads // kv_heads
    td.max_q_seqlen = max(q_seqlen_list)
    td.max_kv_seqlen = max(kv_seqlen_list)
    total_task_num = 0
    first_batch_task_num = 0
    for b_idx in range(batch):
        q_s = q_seqlen_list[b_idx]
        q_block_count_cur = (q_s + Q_BASE_TILE - 1) // Q_BASE_TILE
        cur_task_num = num_heads * q_block_count_cur
        if b_idx == 0:
            first_batch_task_num = cur_task_num
        total_task_num += cur_task_num
    td.first_batch_task_num = first_batch_task_num
    td.total_task_num = total_task_num
    return td


def pack_tiling_int(td: FATilingData) -> List[int]:
    return [
        td.batch,
        td.num_heads,
        td.kv_heads,
        td.max_q_seqlen,
        td.max_kv_seqlen,
        td.first_batch_task_num,
        td.total_task_num,
    ]


def make_actual_seqlen(seqlen_list: List[int]):
    return [0] + list(itertools.accumulate(seqlen_list))


# ---------------------------------------------------------------------------
# stage_qk.py — QK cube 阶段：tiling/标志/缓冲准备、Q tile 搬运、MMAD 与分数搬出
# ---------------------------------------------------------------------------

DTYPE_S = tla.Float32
DTYPE_OTMP = tla.Float32
DTYPE_ACC = tla.Float32
L0_TILE_M = 128
L0_TILE_N = 128
L0_TILE_K = 128
K_L1_BUF = 2


@tla.jit
def _read_tiling(tilingInt, tilingScale, head_dim, head_major):
    """tiling 读取 + 派生偏移，返回 FaCtx。"""
    _TI_BATCH = 0
    _TI_NUMHEADS = 1
    _TI_KVHEADS = 2
    _TI_MAXQ = 3
    _TI_MAXKV = 4
    _TI_FIRSTTASK = 5
    _TI_TOTALTASK = 6
    _TI_QFORMAT = 7
    _TI_KVFORMAT = 8
    _TI_SNOFFSET = 9
    _TI_SSOFFSET = 10
    _TI_STATSBNSD = 11
    qBaseTile_ = Q_BASE_TILE
    kvBaseTile_ = KV_BASE_TILE
    embed_ = head_dim
    embedV_ = head_dim
    qHeads_ = tilingInt[_TI_NUMHEADS]
    kvHeads_ = tilingInt[_TI_KVHEADS]
    maxQSeqlen_ = tilingInt[_TI_MAXQ]
    maxKvSeqlen_ = tilingInt[_TI_MAXKV]
    kvSeqlen_ = tilingInt[_TI_MAXKV]
    qFormat_ = tilingInt[_TI_QFORMAT]  # (0=TND, 1=BSND)
    kvFormat_ = tilingInt[_TI_KVFORMAT]
    sNOffset_ = tilingInt[_TI_SNOFFSET]
    sSOffset_ = tilingInt[_TI_SSOFFSET]
    statsBnsd_ = tilingInt[_TI_STATSBNSD]
    scaleValue_ = tilingScale[0]
    mm1L0ATotalStages_ = (qBaseTile_ // L0_TILE_M) * (128 // L0_TILE_K)
    mm1L0BTotalStages_ = (kvBaseTile_ // L0_TILE_N) * (128 // L0_TILE_K)
    mm2L0ATotalStages_ = (qBaseTile_ // L0_TILE_M) * (kvBaseTile_ // L0_TILE_K)
    mm2L0BTotalStages_ = (kvBaseTile_ // L0_TILE_K) * (128 // L0_TILE_N)
    firstBatchTaskNum_ = tilingInt[_TI_FIRSTTASK]  # qSTileNum_ * qHeads_
    totalTaskNum_ = tilingInt[_TI_TOTALTASK]  # batch_ * firstBatchTaskNum_
    groupSize = qHeads_ // kvHeads_
    ubStages_ = 1 if head_dim > 128 else UB_S_OTMP_BUF_STAGES
    qNOffset = embed_
    kNOffset = embed_
    vNOffset = embedV_
    oNOffset = embedV_
    qSOffset = qHeads_ * qNOffset
    kSOffset = kvHeads_ * kNOffset
    vSOffset = kvHeads_ * vNOffset
    oSOffset = qHeads_ * oNOffset
    if tla.const_expr(head_major):
        # head-major [B, N, S, D]：同一 head 的 S 行连续，head 之间相隔 S*D
        qNOffset = maxQSeqlen_ * embed_
        kNOffset = maxKvSeqlen_ * embedV_
        vNOffset = maxKvSeqlen_ * embedV_
        oNOffset = maxQSeqlen_ * embedV_
        qSOffset = embed_
        kSOffset = embedV_
        vSOffset = embedV_
        oSOffset = embedV_
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
        sNOffset=sNOffset_,
        sSOffset=sSOffset_,
        statsBnsd_=statsBnsd_,
        ubStages_=ubStages_,
    )


@tla.jit
def _declare_flags():
    """flag 声明。"""
    # --- L1->L0 完成 -> GM->L1 可开始 ---
    q_l0a_data_ready = tla.flag("q_l0a_data_ready", tla.pipes.MTE1, tla.pipes.MTE2)
    k_l0b_data_ready_0 = tla.flag("k_l0b_data_ready_0", tla.pipes.MTE1, tla.pipes.MTE2)
    k_l0b_data_ready_1 = tla.flag("k_l0b_data_ready_1", tla.pipes.MTE1, tla.pipes.MTE2)
    v_l0b_data_ready_0 = tla.flag("v_l0b_data_ready_0", tla.pipes.MTE1, tla.pipes.MTE2)
    v_l0b_data_ready_1 = tla.flag("v_l0b_data_ready_1", tla.pipes.MTE1, tla.pipes.MTE2)
    # --- CUBE MMAD 完成 -> L1->L0 可开始 ---
    l0a_available_0 = tla.flag("l0a_available_0", tla.pipes.CUBE, tla.pipes.MTE1)
    l0a_available_1 = tla.flag("l0a_available_1", tla.pipes.CUBE, tla.pipes.MTE1)
    l0b_available_0 = tla.flag("l0b_available_0", tla.pipes.CUBE, tla.pipes.MTE1)
    l0b_available_1 = tla.flag("l0b_available_1", tla.pipes.CUBE, tla.pipes.MTE1)
    # --- FIX (L0C->UB) 完成 -> CUBE 可开始 ---
    l0c_available_0 = tla.flag("l0c_available_0", tla.pipes.FIX, tla.pipes.CUBE)
    l0c_available_1 = tla.flag("l0c_available_1", tla.pipes.FIX, tla.pipes.CUBE)
    l0c_available_2 = tla.flag("l0c_available_2", tla.pipes.FIX, tla.pipes.CUBE)
    l0c_available_3 = tla.flag("l0c_available_3", tla.pipes.FIX, tla.pipes.CUBE)
    # --- GM->L1 加载完成 -> L1->L0 可开始 ---
    q_l1_data_ready = tla.flag("q_l1_data_ready", tla.pipes.MTE2, tla.pipes.MTE1)
    k_l1_data_ready_0 = tla.flag("k_l1_data_ready_0", tla.pipes.MTE2, tla.pipes.MTE1)
    k_l1_data_ready_1 = tla.flag("k_l1_data_ready_1", tla.pipes.MTE2, tla.pipes.MTE1)
    v_l1_data_ready_0 = tla.flag("v_l1_data_ready_0", tla.pipes.MTE2, tla.pipes.MTE1)
    v_l1_data_ready_1 = tla.flag("v_l1_data_ready_1", tla.pipes.MTE2, tla.pipes.MTE1)
    # --- L1->L0 完成 -> CUBE 可开始 ---
    l0a_data_ready_0 = tla.flag("l0a_data_ready_0", tla.pipes.MTE1, tla.pipes.CUBE)
    l0a_data_ready_1 = tla.flag("l0a_data_ready_1", tla.pipes.MTE1, tla.pipes.CUBE)
    l0b_data_ready_0 = tla.flag("l0b_data_ready_0", tla.pipes.MTE1, tla.pipes.CUBE)
    l0b_data_ready_1 = tla.flag("l0b_data_ready_1", tla.pipes.MTE1, tla.pipes.CUBE)
    # --- CUBE MMAD 完成 -> FIX (L0C -> UB) 可开始 ---
    l0c_data_ready_0 = tla.flag("l0c_data_ready_0", tla.pipes.CUBE, tla.pipes.FIX)
    l0c_data_ready_1 = tla.flag("l0c_data_ready_1", tla.pipes.CUBE, tla.pipes.FIX)
    l0c_data_ready_2 = tla.flag("l0c_data_ready_2", tla.pipes.CUBE, tla.pipes.FIX)
    l0c_data_ready_3 = tla.flag("l0c_data_ready_3", tla.pipes.CUBE, tla.pipes.FIX)
    # --- VECTOR 完成 -> gm -> UB ---
    vec_ready_mte2_0 = tla.flag("vec_ready_mte2_0", tla.pipes.VECTOR, tla.pipes.MTE2)
    vec_ready_mte2_1 = tla.flag("vec_ready_mte2_1", tla.pipes.VECTOR, tla.pipes.MTE2)
    # --- mask2index
    mte3_ready_mask_0 = tla.flag("mte3_ready_mask_0", tla.pipes.MTE3, tla.pipes.VECTOR)
    mte3_ready_mask_1 = tla.flag("mte3_ready_mask_1", tla.pipes.MTE3, tla.pipes.VECTOR)
    # --- softmax
    ub_p_available_0 = tla.flag("ub_p_available_0", tla.pipes.MTE3, tla.pipes.VECTOR)
    ub_p_available_1 = tla.flag("ub_p_available_1", tla.pipes.MTE3, tla.pipes.VECTOR)
    # --- rescale
    mte3_ready_rescale = tla.flag(
        "mte3_ready_rescale", tla.pipes.MTE3, tla.pipes.VECTOR
    )
    ub_p_data_ready_0 = tla.flag("ub_p_data_ready_0", tla.pipes.VECTOR, tla.pipes.MTE3)
    ub_p_data_ready_1 = tla.flag("ub_p_data_ready_1", tla.pipes.VECTOR, tla.pipes.MTE3)
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
        tla.cross_core_set_flag(sync.ub_p_cross_ready_0, tla.pipes.MTE1)
        tla.cross_core_set_flag(sync.ub_p_cross_ready_1, tla.pipes.MTE1)
        tla.cross_core_set_flag(sync.ub_p_cross_ready_2, tla.pipes.MTE1)
    with tla.vector():
        tla.set_flag(sync.vec_ready_mte2_0)
        tla.set_flag(sync.vec_ready_mte2_1)
        tla.set_flag(sync.mte3_ready_mask_0)
        tla.set_flag(sync.mte3_ready_mask_1)
        tla.set_flag(sync.ub_p_available_0)
        tla.set_flag(sync.ub_p_available_1)
        tla.set_flag(sync.mte3_ready_rescale)
        tla.cross_core_set_flag(sync.ub_s_cross_ready_0, tla.pipes.VECTOR)
        tla.cross_core_set_flag(sync.ub_s_cross_ready_1, tla.pipes.VECTOR)
        tla.cross_core_set_flag(sync.ub_otmp_cross_ready_0, tla.pipes.VECTOR)
        tla.cross_core_set_flag(sync.ub_otmp_cross_ready_1, tla.pipes.VECTOR)


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
    if tla.const_expr(ctx.ubStages_ > 1):
        ubS_ptrs = [
            tla.allocate(
                ctx.qBaseTile_ // 2 * ctx.kvBaseTile_,
                DTYPE_S,
                tla.AddressSpace.ub,
                256,
            ),
            tla.allocate(
                ctx.qBaseTile_ // 2 * ctx.kvBaseTile_,
                DTYPE_S,
                tla.AddressSpace.ub,
                256,
            ),
        ]
    else:
        ubS_ptrs = [
            tla.allocate(
                ctx.qBaseTile_ // 2 * ctx.kvBaseTile_,
                DTYPE_S,
                tla.AddressSpace.ub,
                256,
            )
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
    if tla.const_expr(ctx.ubStages_ > 1):
        ubOTmp_ptrs = [
            tla.allocate(
                ctx.qBaseTile_ // 2 * ctx.embed_,
                DTYPE_OTMP,
                tla.AddressSpace.ub,
                256,
            ),
            tla.allocate(
                ctx.qBaseTile_ // 2 * ctx.embed_,
                DTYPE_OTMP,
                tla.AddressSpace.ub,
                256,
            ),
        ]
    else:
        ubOTmp_ptrs = [
            tla.allocate(
                ctx.qBaseTile_ // 2 * ctx.embed_,
                DTYPE_OTMP,
                tla.AddressSpace.ub,
                256,
            )
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
    ubStatMax_ptr = tla.allocate(
        (ctx.qBaseTile_ // 2) * 8, tla.Float32, tla.AddressSpace.ub, 256
    )
    ubStatSum_ptr = tla.allocate(
        (ctx.qBaseTile_ // 2) * 8, tla.Float32, tla.AddressSpace.ub, 256
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
        ubStatMax_ptr,
        ubStatSum_ptr,
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
            tla.cross_core_wait_flag(sync.ub_s_cross_ready_0, tla.pipes.FIX)
        else:
            tla.cross_core_wait_flag(sync.ub_s_cross_ready_1, tla.pipes.FIX)
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
            tla.cross_core_set_flag(sync.ub_s_cross_ready_0, tla.pipes.FIX)
        else:
            tla.cross_core_set_flag(sync.ub_s_cross_ready_1, tla.pipes.FIX)
        if gatheredKvSTileIdx == kvSLoopNum - 1:
            tla.set_flag(sync.q_l0a_data_ready)  # MTE1_MTE2


# ---------------------------------------------------------------------------
# stage_softmax.py — Online Softmax Vector 阶段：掩码、行 max/行 sum、重缩放与逐行统计量
# ---------------------------------------------------------------------------

MIN_VALUE = -3.4028234663852886e38


@tla.jit
def _exp_shift(max_reg):
    """Shift used by exp; an all-masked row (max == MIN_VALUE) shifts by 0.

    exp(MIN_VALUE - 0) underflows to 0, so a fully masked row keeps sum == 0
    (reference contract) instead of producing exp(0) == 1 for every column.
    """
    return tla.where(
        tla.cmp(max_reg, MIN_VALUE * 0.5, mode="gt"),
        max_reg,
        tla.full(0.0, dtype=tla.Float32),
    )


@tla.jit
def _mask_pred(ub_mask_i, pregFull, dense_mask):
    """读取一行掩码谓词：因果路径为 bit-packed，dense 路径为 0/1 f32。"""
    if tla.const_expr(dense_mask):
        v = ub_mask_i.load()
        z = tla.sub(v, v, mask=pregFull)
        return tla.cmp(v, z, "gt", mask=pregFull)
    return ub_mask_i.load(MaskLoadParams())


@tla.jit
def _mask_sf_first(
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
    dense_mask,
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
                    mask_reg00 = _mask_pred(ub_mask_i0, pregFull, dense_mask)
                    mask_reg11 = _mask_pred(ub_mask_i1, pregFull, dense_mask)
                    ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                    ub_s_reg1 = tla.mul(ub_s_reg1, ctx.scaleValue_, mask=pregFull)
                    ub_s_reg0 = tla.where(mask_reg00, ub_s_reg0, minVreg)
                    ub_s_reg1 = tla.where(mask_reg11, ub_s_reg1, minVreg)
                    ub_s_i0.store(ub_s_reg0, mask=pregFull)
                    ub_s_i1.store(ub_s_reg1, mask=pregFull)
                    max_tmp_reg = tla.max(ub_s_reg0, ub_s_reg1, mask=pregFull)
                    max_reg = max_tmp_reg.reduce(tla.ReductionOp.MAX, mask=pregFull)
                    ub_last_max_i.store(
                        max_reg,
                        tla.params.NormalStoreParams(
                            store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                        ),
                    )
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
                        tla.sub(ub_s_reg0De, _exp_shift(max_regDe), mask=pregFull),
                        mask=pregFull,
                    )
                    ub_s_even_reg = tla.exp(
                        tla.sub(ub_s_reg1De, _exp_shift(max_regDe), mask=pregFull),
                        mask=pregFull,
                    )
                    exp_odd_reg0, exp_even_reg1 = tla.deinterleave(
                        ub_s_odd_reg, ub_s_even_reg
                    )
                    exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                    exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                    ub_last_sum_iDe.store(
                        exp_sum_reg,
                        tla.params.NormalStoreParams(
                            store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                        ),
                    )
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
                    ub_last_max_i.store(
                        max_reg,
                        tla.params.NormalStoreParams(
                            store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                        ),
                    )
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
                        tla.sub(ub_s_odd_reg, _exp_shift(max_regDe), mask=pregFull),
                        mask=pregFull,
                    )
                    exp_even_reg1 = tla.exp(
                        tla.sub(ub_s_even_reg, _exp_shift(max_regDe), mask=pregFull),
                        mask=pregFull,
                    )
                    exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                    exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                    ub_last_sum_iDe.store(
                        exp_sum_reg,
                        tla.params.NormalStoreParams(
                            store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                        ),
                    )
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
                    mask_reg00 = _mask_pred(ub_mask_i0, pregFull, dense_mask)
                    ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                    ub_s_reg0 = tla.where(mask_reg00, ub_s_reg0, minVreg)
                    ub_s_i0.store(ub_s_reg0, mask=pregFull)
                    max_reg = ub_s_reg0.reduce(tla.ReductionOp.MAX, mask=pregFull)
                    ub_last_max_i.store(
                        max_reg,
                        tla.params.NormalStoreParams(
                            store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                        ),
                    )
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
                        tla.sub(ub_s_reg0De, _exp_shift(max_regDe), mask=pregFull),
                        mask=pregFull,
                    )
                    exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                    ub_last_sum_iDe.store(
                        exp_sum_reg,
                        tla.params.NormalStoreParams(
                            store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                        ),
                    )
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
                    ub_last_max_i.store(
                        max_reg,
                        tla.params.NormalStoreParams(
                            store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                        ),
                    )
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
                        tla.sub(ub_s_reg0De, _exp_shift(max_regDe), mask=pregFull),
                        mask=pregFull,
                    )
                    exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                    ub_last_sum_iDe.store(
                        exp_sum_reg,
                        tla.params.NormalStoreParams(
                            store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                        ),
                    )
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
    dense_mask,
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
                mask_reg00 = _mask_pred(ub_mask_i0, pregFull, dense_mask)
                mask_reg11 = _mask_pred(ub_mask_i1, pregFull, dense_mask)
                ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                ub_s_reg1 = tla.mul(ub_s_reg1, ctx.scaleValue_, mask=pregFull)
                ub_s_reg0 = tla.where(mask_reg00, ub_s_reg0, minVreg)
                ub_s_reg1 = tla.where(mask_reg11, ub_s_reg1, minVreg)
                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                ub_s_i1.store(ub_s_reg1, mask=pregFull)
                max_tmp_reg = tla.max(ub_s_reg0, ub_s_reg1, mask=pregFull)
                max_reg = max_tmp_reg.reduce(tla.ReductionOp.MAX, mask=pregFull)
                ub_now_max_i.store(
                    max_reg,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                    ),
                )
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
                    tla.sub(ub_s_reg0De, _exp_shift(max_regDe), mask=pregFull),
                    mask=pregFull,
                )
                ub_s_even_reg = tla.exp(
                    tla.sub(ub_s_reg1De, _exp_shift(max_regDe), mask=pregFull),
                    mask=pregFull,
                )
                exp_odd_reg0, exp_even_reg1 = tla.deinterleave(
                    ub_s_odd_reg, ub_s_even_reg
                )
                exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                ub_now_sum_iDe.store(
                    exp_sum_reg,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                    ),
                )
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
                ub_now_max_i.store(
                    max_reg,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                    ),
                )
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
                    tla.sub(ub_s_odd_reg, _exp_shift(max_regDe), mask=pregFull),
                    mask=pregFull,
                )
                exp_even_reg1 = tla.exp(
                    tla.sub(ub_s_even_reg, _exp_shift(max_regDe), mask=pregFull),
                    mask=pregFull,
                )
                exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                ub_now_sum_iDe.store(
                    exp_sum_reg,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                    ),
                )
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
    dense_mask,
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
                mask_reg00 = _mask_pred(ub_mask_i0, pregFull, dense_mask)
                ub_s_reg0 = tla.mul(ub_s_reg0, ctx.scaleValue_, mask=pregFull)
                ub_s_reg0 = tla.where(mask_reg00, ub_s_reg0, minVreg)
                ub_s_i0.store(ub_s_reg0, mask=pregFull)
                max_reg = ub_s_reg0.reduce(tla.ReductionOp.MAX, mask=pregFull)
                ub_now_max_i.store(
                    max_reg,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                    ),
                )
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
                    tla.sub(ub_s_reg0De, _exp_shift(max_regDe), mask=pregFull),
                    mask=pregFull,
                )
                exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                ub_now_sum_iDe.store(
                    exp_sum_reg,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                    ),
                )
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
                ub_now_max_i.store(
                    max_reg,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                    ),
                )
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
                    tla.sub(ub_s_reg0De, _exp_shift(max_regDe), mask=pregFull),
                    mask=pregFull,
                )
                exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                ub_now_sum_iDe.store(
                    exp_sum_reg,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                    ),
                )
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
                ub_last_max_i.store(
                    max_reg,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                    ),
                )
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
                    tla.sub(ub_s_odd_reg, _exp_shift(max_regDe), mask=pregFull),
                    mask=pregFull,
                )
                exp_even_reg1 = tla.exp(
                    tla.sub(ub_s_even_reg, _exp_shift(max_regDe), mask=pregFull),
                    mask=pregFull,
                )
                exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
                exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
                ub_last_sum_iDe.store(
                    exp_sum_reg,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                    ),
                )
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
                ub_last_max_i.store(
                    max_reg,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                    ),
                )
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
                    tla.sub(ub_s_reg0De, _exp_shift(max_regDe), mask=pregFull),
                    mask=pregFull,
                )
                exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
                ub_last_sum_iDe.store(
                    exp_sum_reg,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                    ),
                )
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
            ub_now_max_i.store(
                max_reg,
                tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                ),
            )
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
                tla.sub(ub_s_odd_reg, _exp_shift(max_regDe), mask=pregFull),
                mask=pregFull,
            )
            exp_even_reg1 = tla.exp(
                tla.sub(ub_s_even_reg, _exp_shift(max_regDe), mask=pregFull),
                mask=pregFull,
            )
            exp_sum_reg = tla.add(exp_odd_reg0, exp_even_reg1, mask=pregFull)
            exp_sum_reg = exp_sum_reg.reduce(tla.ReductionOp.ADD, mask=pregFull)
            ub_now_sum_iDe.store(
                exp_sum_reg,
                tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                ),
            )
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
            ub_now_max_i.store(
                max_reg,
                tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                ),
            )
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
                tla.sub(ub_s_reg0De, _exp_shift(max_regDe), mask=pregFull),
                mask=pregFull,
            )
            exp_sum_reg = exp_reg0.reduce(tla.ReductionOp.ADD, mask=pregFull)
            ub_now_sum_iDe.store(
                exp_sum_reg,
                tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                ),
            )
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


# ---------------------------------------------------------------------------
# stage_pv.py — PV cube 阶段：P/V 搬运与 MMAD 累加
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# stage_rescale.py — 输出重缩放与归一化、softmax_max / softmax_sum 写回
# ---------------------------------------------------------------------------


@tla.jit
def _safe_den(den_reg):
    """Guard the softmax denominator: a fully masked row has sum == 0 -> 1."""
    return tla.where(
        tla.cmp(den_reg, 0.0, mode="gt"),
        den_reg,
        tla.full(1.0, dtype=tla.Float32),
    )


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
    softmaxMax,
    softmaxSum,
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
        ubOTmpBufId = validIdxDe % ctx.ubStages_
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
                            div_reg_0 = tla.div(
                                lo_reg_0, _safe_den(gl_reg), mask=pregFullDee
                            )
                            div_reg_1 = tla.div(
                                lo_reg_1, _safe_den(gl_reg), mask=pregFullDee
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
                            div_reg_0 = tla.div(
                                lo_reg_0, _safe_den(gl_reg), mask=pregFullDee
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
                            div_reg_0 = tla.div(
                                add_reg_0, _safe_den(gl_reg), mask=pregFullDee2
                            )
                            div_reg_1 = tla.div(
                                add_reg_1, _safe_den(gl_reg), mask=pregFullDee2
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
                            div_reg_0 = tla.div(
                                add_reg_0, _safe_den(gl_reg), mask=pregFullDee2
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
                # softmax_max / softmax_sum：每行广播到 8 列，先写 UB，
                # 再与 O 共用同一次 VEC->MTE3 同步后拷出
                rowInBatch = (
                    task.qSTileIdx * ctx.qBaseTile_ + subBlockIdxDe * rowNumSplit
                )
                sBase = tla.as_numeric(0)
                if ctx.statsBnsd_ != c0:
                    # (B, N, S, 8)：batch/head 步长显式展开，行内 stride=8，
                    # 这样 host 侧无需再 permute+contiguous。
                    sBase = (
                        task.curBatch * ctx.qHeads_ + task.qHeadIdx
                    ) * ctx.sNOffset + rowInBatch * ctx.sSOffset
                else:
                    qRowBase = task.qBOffset // ctx.qSOffset + rowInBatch
                    sBase = qRowBase * ctx.sSOffset + task.qHeadIdx * ctx.sNOffset
                gm_smax = tla.make_tensor(
                    softmaxMax.ptr + sBase,
                    tla.make_layout(
                        tla.make_shape(rowNumCurSubCore, 8),
                        tla.make_stride(ctx.sSOffset, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                gm_ssum = tla.make_tensor(
                    softmaxSum.ptr + sBase,
                    tla.make_layout(
                        tla.make_shape(rowNumCurSubCore, 8),
                        tla.make_stride(ctx.sSOffset, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                ub_stats_max = tla.make_tensor(
                    bufs.ubStatMax_ptr,
                    tla.make_layout(
                        tla.make_shape(rowNumCurSubCore, 8),
                        tla.make_stride(8, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                ub_stats_sum = tla.make_tensor(
                    bufs.ubStatSum_ptr,
                    tla.make_layout(
                        tla.make_shape(rowNumCurSubCore, 8),
                        tla.make_stride(8, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                lastMaxUb = tla.make_tensor(
                    bufs.lastMax_ptr,
                    tla.make_layout(
                        tla.make_shape(rowNumCurSubCore), tla.make_stride(1)
                    ),
                )
                lastSumUb = tla.make_tensor(
                    bufs.lastSum_ptr,
                    tla.make_layout(
                        tla.make_shape(rowNumCurSubCore), tla.make_stride(1)
                    ),
                )
                with tla.vec.func(mode="simd"):
                    pregStat, _ = tla.update_mask(8, dtype=tla.Float32)
                    for i5 in tla.range(rowNumCurSubCore):
                        ub_mx = tla.tile_view(
                            lastMaxUb, tla.make_shape(1), tla.make_coord(i5)
                        )
                        ub_sm = tla.tile_view(
                            lastSumUb, tla.make_shape(1), tla.make_coord(i5)
                        )
                        mx_reg = ub_mx.load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        sm_reg = ub_sm.load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        ub_stat_mx = tla.tile_view(
                            ub_stats_max,
                            tla.make_shape(1, 8),
                            tla.make_coord(i5, c0),
                        )
                        ub_stat_sm = tla.tile_view(
                            ub_stats_sum,
                            tla.make_shape(1, 8),
                            tla.make_coord(i5, c0),
                        )
                        ub_stat_mx.store(mx_reg, mask=pregStat)
                        ub_stat_sm.store(sm_reg, mask=pregStat)
                tla.set_flag(sync.ub_p_data_ready_0)
                tla.wait_flag(sync.ub_p_data_ready_0)
                gm_out_tile = tla.make_tensor(
                    attentionOut.ptr
                    + task.qBOffset
                    + task.qHeadIdx * ctx.qNOffset
                    + (task.qSTileIdx * ctx.qBaseTile_ + subBlockIdxDe * rowNumSplit)
                    * ctx.oSOffset,
                    tla.make_layout(
                        tla.make_shape(rowNumCurSubCore, nDe),
                        tla.make_stride(ctx.oSOffset, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(gm_out_tile, goUb16)
                tla.copy(gm_smax, ub_stats_max)
                tla.copy(gm_ssum, ub_stats_sum)
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


# ---------------------------------------------------------------------------
# kernel — sparse_mode 编译期特化：0 无掩码，1/2/3 带掩码
# ---------------------------------------------------------------------------


@tla.kernel
def flash_attention_score_kernel(
    query: tla.Tensor,  # gQ  [S1, D]（BSND/TND：packed [*,S,N,D] 的单 head 切片）
    key: tla.Tensor,  # gK  [D, S2]（ColumnMajor：直接给出 K^T）
    value: tla.Tensor,  # gV  [S2, D]
    attentionOut: tla.Tensor,  # gO  [S1, D]
    softmaxMax: tla.Tensor,  # gSMax [rows, 8]  fp32（每行广播 8 列）
    softmaxSum: tla.Tensor,  # gSSum [rows, 8]  fp32（每行广播 8 列）
    tilingInt: tla.Tensor,  # Tiling 整型包（Int32 1D）
    tilingScale: tla.Tensor,  # Tiling 浮点包（fp32 1D，[scaleValue]）
    actualQseqlen: tla.Tensor,  # gActualQseqlen：TND 为累加 [B+1]，BSND 占位 [B]
    actualKvseqlen: tla.Tensor,  # gActualKvseqlen：TND 累加 [B+1]，BSND 逐 batch [B]
    tileRange: tla.Tensor,  # TileMask tile_range [batch, Tq] int32（sparse=0 占位）
    tileCompute: tla.Tensor,  # TileMask tile_compute_bp [batch, Tq, Wk] int32（占位）
    fineMask: tla.Tensor,  # TileMask fine_mask_bp [batch, Tq, Wk] int32（占位）
    maskr: tla.Tensor,  # TileMask maskr [batch, Sq] int32（占位）
    maskGm: tla.Tensor,  # dense mask [S1, S2] fp32，1=可见 0=遮蔽（占位）
    is_fp16: tla.Constexpr[bool],  # True=fp16，False=bf16（决定 DTYPE_Q/K/V/P/O）
    uniform_q_seqlen: tla.Constexpr[int],  # 定长 Q 每 batch 序列长度（0=非 uniform）
    uniform_kv_seqlen: tla.Constexpr[int],  # 定长 KV 每 batch 序列长度（0=非 uniform）
    uniform_tasks_per_batch: tla.Constexpr[int],  # 每 batch 的 Q 任务数（0=非 uniform）
    head_dim: tla.Constexpr[int],  # head 维度（编译期）
    dense_mask: tla.Constexpr[bool],  # sparse=1：掩码从 GM 逐 tile 搬入
    mask_stride: tla.Constexpr[int],  # dense mask 行 stride（= S2 向上取整到 128）
    sparse: tla.Constexpr[int],  # 0=无掩码；1=dense mask；2/3=因果掩码
    head_major: tla.Constexpr[bool],  # True=输入输出为 [B,N,S,D] head-major 布局
) -> None:
    """训练前向 FA kernel；sparse=0 时掩码代码在编译期整体剔除。"""
    # 编译期根据 is_fp16 选择 Q/K/V/P/O 的数据类型；S/OTMP/ACC 恒为 Float32
    ctx = _read_tiling(tilingInt, tilingScale, head_dim, head_major)
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
    if tla.const_expr(sparse != 0):
        # mask 缓冲：dense 路径存 0/1 f32 掩码块，因果路径存 bit-packed 谓词
        mask_ub_ptr = tla.allocate(
            ctx.qBaseTile_ // 2 * head_dim, tla.Float32, tla.AddressSpace.ub, 256
        )
        maskr_ub_ptr = tla.allocate(64, tla.Int32, tla.AddressSpace.ub, 256)
        mutex_maskr = tla.mutex(resource="maskr_ub", id=16)
    maxQsBlockNum = (ctx.maxQSeqlen_ + ctx.qBaseTile_ - 1) // ctx.qBaseTile_
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
    task_range = tla.range(
        tla.arch.block_idx(), ctx.totalTaskNum_, tla.arch.block_num()
    )
    for taskSlotIdx in task_range:
        if tla.const_expr(uniform_tasks_per_batch != 0):
            curBatch = taskSlotIdx // uniform_tasks_per_batch
            taskIdxCurBatch = taskSlotIdx - curBatch * uniform_tasks_per_batch
            # batch 步长 = heads * S * D，打包与 head-major 两种布局同值
            qBOffset = curBatch * ctx.qHeads_ * uniform_q_seqlen * ctx.embed_
            kBOffset = curBatch * ctx.kvHeads_ * uniform_kv_seqlen * ctx.embedV_
            vBOffset = kBOffset
            oBOffset = curBatch * ctx.qHeads_ * uniform_q_seqlen * ctx.embedV_
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
        # 掩码路径的状态变量先无条件初始化：静态 if 只赋值 then 分支时，
        # 其 carried value 需要在调用点已绑定（sparse=0 也要走通）。
        trIdx = tla.as_numeric(0)
        tileRangeVal = tla.as_numeric(0)
        if tla.const_expr(sparse != 0):
            # TileMask tile_range 给出该 Q tile 需扫过的 KV 上界（以 KV tile 数计）
            trIdx = curBatch * maxQsBlockNum + qSTileIdx
            tileRangeVal = tileRange[trIdx]
            noSkipKvS = min(ctx.kvBaseTile_ * tileRangeVal, curKvSeqlen)
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
            validNum = tla.as_numeric(0)
            cmp = tla.as_numeric(True)
            maskMoved = tla.as_numeric(0)  # maskr 是否已搬入 UB（每 Q 块只搬一次）
            Tk = tla.as_numeric(0)
            Wk = tla.as_numeric(0)
            fineMaskWordBase = tla.as_numeric(0)
            maskHasFine = tla.as_numeric(False)
            is_move_mask = tla.as_numeric(False)
            if tla.const_expr(sparse != 0):
                Tk = (ctx.kvSeqlen_ + ctx.kvBaseTile_ - 1) // ctx.kvBaseTile_
                Wk = ceil_div(Tk, 32)
                fineMaskWordBase = trIdx * Wk
                maskHasFine = fineMask[fineMaskWordBase] != c0
                for fineMaskWordIdx in tla.range(c1, Wk, c1):
                    if fineMask[fineMaskWordBase + fineMaskWordIdx] != c0:
                        maskHasFine = tla.as_numeric(True)
                is_move_mask = maskHasFine
            for gatheredKvSTileIdx in KvS_range:
                if tla.const_expr(sparse != 0):
                    # tileCompute 标记该 KV tile 是否参与计算（全遮蔽 tile 直接跳过）
                    if gatheredKvSTileIdx < kvSLoopNum:
                        computeWordIdx = trIdx * Wk + gatheredKvSTileIdx // 32
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
                    ubSBufId = validIdx % ctx.ubStages_
                    if tla.const_expr(ctx.ubStages_ > 1):
                        ubS_ptr = (
                            bufs.ubS_ptrs[0] if ubSBufId == c0 else bufs.ubS_ptrs[1]
                        )
                    else:
                        ubS_ptr = bufs.ubS_ptrs[0]
                    kvSStartIdx = tla.as_numeric(0)
                    wordIdx = tla.as_numeric(0)
                    bitPos = tla.as_numeric(0)
                    fineMaskBit = tla.as_numeric(False)
                    if tla.const_expr(sparse != 0):
                        kvSStartIdx = gatheredKvSTileIdx * ctx.kvBaseTile_
                        wordIdx = trIdx * Wk + gatheredKvSTileIdx // 32
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
                    # online Softmax
                    if tla.const_expr(sparse != 0):
                        # maskr 只在首个 KV 迭代搬一次，后续复用 UB 中的掩码
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
                            maskGm,
                            dense_mask,
                            mask_stride,
                        )
                        if moveMaskNow:
                            maskMoved = c1
                    else:
                        _softmax_wrap_nomask(
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
                        )
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
                    ubSBufIdDe = validIdxDe % ctx.ubStages_
                    if tla.const_expr(ctx.ubStages_ > 1):
                        ubS_ptrDe = (
                            bufs.ubS_ptrs[0] if ubSBufIdDe == c0 else bufs.ubS_ptrs[1]
                        )
                    else:
                        ubS_ptrDe = bufs.ubS_ptrs[0]
                    ubSTensorTlaDe = tla.make_tensor(
                        ubS_ptrDe,
                        tla.make_layout(
                            tla.make_shape(rowNum, kvSTileSizeActDe),
                            tla.make_stride(128, 1),
                        ),
                    )
                    ubOTmpBufId = validIdxDe % ctx.ubStages_
                    if tla.const_expr(ctx.ubStages_ > 1):
                        ubOTmp_ptr = (
                            bufs.ubOTmp_ptrs[0]
                            if ubOTmpBufId == c0
                            else bufs.ubOTmp_ptrs[1]
                        )
                    else:
                        ubOTmp_ptr = bufs.ubOTmp_ptrs[0]
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
                            curBatch=curBatch,
                        ),
                        ctx,
                        bufs,
                        sync,
                        validIdxDe,
                        isLastKvSTileDe,
                        ubOTmp_ptr,
                        isFirstKvSTileDe,
                        attentionOut,
                        softmaxMax,
                        softmaxSum,
                    )
                cmp = tla.as_numeric(True)
    _drain(sync)


@tla.jit
def _softmax_wrap_nomask(
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
            # 标量参数
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
                _sf_first(ctx, n, tailN, m, ub_s, lastMaxAddr, lastSumAddr, ub_p_zN)
            else:
                if n > 64:
                    _sf_mid_n64(
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
                    )
                else:
                    _sf_mid_nlt64(
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
            tla.copy(l1P_tile, ub_p_zN)  # CopyPUbToPL1
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
    maskGm,
    dense_mask,
    mask_stride,
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
            if tla.const_expr(dense_mask):
                # 任意 mask（sparse_mode=1）：直接把 0/1 f32 掩码块搬入 UB
                gm_mask_row = qSTileIdx * ctx.qBaseTile_ + subIdxEff * mHalf
                ub_mask_tile = tla.tile_view(
                    ub_mask, tla.make_shape(m, 128), tla.make_coord(c0, c0)
                )
                gm_mask_tile = tla.make_tensor(
                    maskGm.ptr + gm_mask_row * mask_stride + kvSStartIdx,
                    tla.make_layout(
                        tla.make_shape(m, 128),
                        tla.make_stride(mask_stride, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                mutex_maskr.lock(pipe=tla.pipes.MTE2)
                tla.copy(ub_mask_tile, gm_mask_tile)
                mutex_maskr.unlock(pipe=tla.pipes.MTE2)
                if fineMaskBit:
                    # 与消费端(softmax)串行：下一 KV tile 的 MTE2 拷贝需等本 tile 读完
                    mutex_maskr.lock(pipe=tla.pipes.VECTOR)
            else:
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
                                preg0 = tla.bitwise_and(
                                    preg0, pregTailN0, mask=pregFull0
                                )
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
                    dense_mask,
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
                        dense_mask,
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
                        dense_mask,
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


# ---------------------------------------------------------------------------
# host side: layout/len normalization, mask packing, launch
# ---------------------------------------------------------------------------


_NPU_ARCH = "--npu-arch 3510"
_ARTIFACTS: dict = {}


def _tla_tensor(buf: torch.Tensor):
    contiguous = buf.contiguous()
    t = from_dlpack(contiguous, layout_tag=tla.arch.RowMajor)
    t._torch_storage = contiguous
    return t


# ``from_dlpack`` costs ~50us per tensor and takes ownership of the buffer, so
# binding a freshly allocated output on every call is both slow and uncacheable.
# Layout metadata only depends on (shape, dtype), so it is derived once per
# shape from a host buffer and combined with the real address per call.
_TEMPLATES: dict = {}


def _bind(buf: torch.Tensor):
    if not buf.is_contiguous():
        buf = buf.contiguous()
    key = (tuple(buf.shape), buf.dtype, str(buf.device))
    tpl = _TEMPLATES.get(key)
    if tpl is None:
        host = torch.empty(1, buf.shape[-1], dtype=buf.dtype)
        tpl = from_dlpack(
            host, layout_tag=tla.arch.RowMajor, origin_shape=tuple(buf.shape)
        )
        _cache_put(_TEMPLATES, key, tpl)
    t = copy.copy(tpl)
    t.data_ptr = int(buf.data_ptr())
    t._external_binding = True
    t._torch_storage = buf
    return t


_BLOCK_NUM: dict = {}


def _block_num(device: int = 0) -> int:
    cores = _BLOCK_NUM.get(int(device))
    if cores is None:
        props = torch.npu.get_device_properties(int(device))
        cores = max(1, int(props.cube_core_num))
        _BLOCK_NUM[int(device)] = cores
    return cores


# Immutable launch arguments (tiling words, scale, cumulative lengths, mask
# tile descriptors) are materialised once per distinct value and reused, so a
# steady-state call only has to bind the caller's tensors.
_BOUND: dict = {}
_TILING_CACHE: dict = {}
_TILE_CACHE: dict = {}
_MASK_CACHE: dict = {}
_CACHE_LIMIT = 64


def _cache_put(cache, key, value):
    if len(cache) >= _CACHE_LIMIT:
        cache.clear()
    cache[key] = value


def _cached_tla(key, build):
    bound = _BOUND.get(key)
    if bound is None:
        bound = _tla_tensor(build())
        _cache_put(_BOUND, key, bound)
    return bound


def _tiling_words(
    b, nq, nkv, q_lens, kv_lens, d, q_format, kv_format, sn_off, ss_off, s_mode
):
    key = (
        b,
        nq,
        nkv,
        tuple(q_lens),
        tuple(kv_lens),
        d,
        q_format,
        kv_format,
        sn_off,
        ss_off,
        s_mode,
    )
    words = _TILING_CACHE.get(key)
    if words is None:
        words = pack_tiling_int(
            compute_tiling(
                batch=b,
                num_heads=nq,
                kv_heads=nkv,
                q_seqlen_list=list(q_lens),
                kv_seqlen_list=list(kv_lens),
                head_dim=d,
            )
        ) + [q_format, kv_format, sn_off, ss_off, s_mode]
        _cache_put(_TILING_CACHE, key, words)
    return words


def _lens(actual_seq, uniform: int, batch: int):
    """Return (lens, cumulative) for one sequence dimension."""
    if actual_seq is None or len(list(actual_seq)) == 0:
        return [uniform] * batch, [uniform * (i + 1) for i in range(batch)]
    cum = [int(x) for x in actual_seq]
    prev = 0
    lens = []
    for v in cum:
        lens.append(v - prev)
        prev = v
    return lens, cum


def _compile(
    targs,
    is_fp16,
    uniform_q,
    uniform_kv,
    tasks_per_batch,
    head_dim,
    dense_mask,
    mask_stride,
    sparse,
    head_major,
):
    key = (
        tuple(tuple(t.shape) for t in targs),
        is_fp16,
        uniform_q,
        uniform_kv,
        tasks_per_batch,
        head_dim,
        dense_mask,
        mask_stride,
        sparse,
        head_major,
    )
    art = _ARTIFACTS.get(key)
    if art is None:
        art = tla.compile(
            flash_attention_score_kernel,
            *targs,
            is_fp16,
            uniform_q,
            uniform_kv,
            tasks_per_batch,
            head_dim,
            dense_mask,
            mask_stride,
            sparse,
            head_major,
            options=_NPU_ARCH,
        )
        _ARTIFACTS[key] = art
    return art


def _pack_dense_mask(atten_mask, pad_cols, device):
    """attention_mask [S1, S2] -> f32 [S1, pad_cols]，1=可见 0=遮蔽。

    pad_cols 取 128 的倍数：dense 路径每 tile 固定搬 128 列，
    尾部不足的列由零填充（不会被 softmax 消费）。
    """
    # id() 复用 + 原地改写会命中陈旧副本，因此把 tensor 版本号纳入 key
    key = (
        id(atten_mask),
        atten_mask._version,
        tuple(atten_mask.shape),
        pad_cols,
        str(device),
    )
    hit = _MASK_CACHE.get(key)
    if hit is not None and hit[0]() is atten_mask:
        return hit[1]
    vis = (atten_mask.to("cpu") == 0).to(torch.float32)
    if vis.shape[1] < pad_cols:
        vis = torch.nn.functional.pad(vis, (0, pad_cols - vis.shape[1]))
    packed = vis.to(device).contiguous()
    bound = _tla_tensor(packed)
    _cache_put(_MASK_CACHE, key, (weakref.ref(atten_mask), bound))
    return bound


def _mask_arrays(b, q_lens, kv_lens, sparse, max_q, max_kv, device):
    """构造 TileMask 打包数组（tile_range / tile_compute / fine_mask / maskr）。"""
    key = (b, tuple(q_lens), tuple(kv_lens), sparse, max_q, max_kv, str(device))
    cached = _TILE_CACHE.get(key)
    if cached is not None:
        return cached
    qb = 128
    max_qb = (max_q + qb - 1) // qb
    tk = (max_kv + qb - 1) // qb
    wk = (tk + 31) // 32
    tile_range = [0] * (b * max_qb)
    tile_compute = [0] * (b * max_qb * wk)
    fine_mask = [0] * (b * max_qb * wk)
    maskr = [0] * (b * max_q)
    for bi in range(b):
        s1 = int(q_lens[bi])
        s2 = int(kv_lens[bi])
        nq_t = (s1 + qb - 1) // qb
        for t in range(nq_t):
            if sparse == 1:
                need = s2
            else:
                imax = min(s1, (t + 1) * qb) - 1
                if sparse == 2:
                    need = min(s2, imax + 1)
                else:
                    need = max(0, min(s2, imax + (s2 - s1) + 1))
            tr = (need + qb - 1) // qb
            tile_range[bi * max_qb + t] = tr
            for k in range(tr):
                w = (bi * max_qb + t) * wk + k // 32
                tile_compute[w] |= 1 << (k % 32)
                fine_mask[w] |= 1 << (k % 32)
        if sparse != 1:
            for r in range(s1):
                if sparse == 2:
                    maskr[bi * max_q + r] = r + 1
                else:
                    maskr[bi * max_q + r] = r + (s2 - s1) + 1
    key = (b, tuple(q_lens), tuple(kv_lens), sparse, max_q, max_kv, str(device))
    built = _TILE_CACHE.get(key)
    if built is None:
        built = tuple(
            _tla_tensor(torch.tensor(values, dtype=torch.int32, device=device))
            for values in (tile_range, tile_compute, fine_mask, maskr)
        )
        _cache_put(_TILE_CACHE, key, built)
    return built


def _launch(
    q2d,
    k2d,
    v2d,
    b,
    nq,
    nkv,
    d,
    q_lens,
    kv_lens,
    cum_q,
    cum_kv,
    scale_value,
    uniform_q,
    uniform_kv,
    tasks_per_batch,
    q_format,
    kv_format,
    stats_shape,
    sn_off,
    ss_off,
    stats_bnsd=0,
    head_major=False,
    sparse=0,
    mask_gm=None,
    mask_stride=0,
    tile_arrays=None,
):
    """Launch the kernel on the packed [rows, D] representation."""
    device = q2d.device
    # The vendored kernel is validated for head_dim == 128; D < 128 is padded
    # with zeros (QK scores and V columns are unaffected), D > 128 uses a
    # smaller Q tile to fit the vector-core UB budget.
    d_in = d
    if d > 128:
        raise NotImplementedError("head_dim > 128 is not supported by this kernel")
    pad_d = 128 if d < 128 else d
    if pad_d != d:
        q2d = torch.nn.functional.pad(q2d, (0, pad_d - d))
        k2d = torch.nn.functional.pad(k2d, (0, pad_d - d))
        v2d = torch.nn.functional.pad(v2d, (0, pad_d - d))
    d = pad_d
    out = torch.empty_like(q2d)
    smax = torch.empty(stats_shape, dtype=torch.float32, device=device)
    ssum = torch.empty(stats_shape, dtype=torch.float32, device=device)

    tiling_int = _tiling_words(
        b, nq, nkv, q_lens, kv_lens, d, q_format, kv_format, sn_off, ss_off, stats_bnsd
    )
    dev = str(device)

    common = (
        _bind(q2d),
        _bind(k2d),
        _bind(v2d),
        _bind(out),
        _bind(smax.reshape(-1, 8)),
        _bind(ssum.reshape(-1, 8)),
        _cached_tla(
            ("tiling", dev, tuple(tiling_int)),
            lambda: torch.tensor(tiling_int, dtype=torch.int32, device=device),
        ),
        _cached_tla(
            ("scale", dev, float(scale_value)),
            lambda: torch.tensor(
                [float(scale_value)], dtype=torch.float32, device=device
            ),
        ),
        _cached_tla(
            ("cumq", dev, tuple(cum_q)),
            lambda: torch.tensor(cum_q, dtype=torch.int32, device=device),
        ),
        _cached_tla(
            ("cumkv", dev, tuple(cum_kv)),
            lambda: torch.tensor(cum_kv, dtype=torch.int32, device=device),
        ),
    )
    # sparse=0 时掩码分支在编译期被裁掉，这些 1 元素占位不会被读
    placeholder = _cached_tla(
        ("tile-placeholder", dev),
        lambda: torch.zeros(1, dtype=torch.int32, device=device),
    )
    tile_range = tile_compute = fine_mask = maskr = placeholder
    if tile_arrays is not None:
        tile_range, tile_compute, fine_mask, maskr = tile_arrays
    if mask_gm is None:
        mask_gm = _cached_tla(
            ("maskgm-empty", dev),
            lambda: torch.zeros(1, dtype=torch.float32, device=device),
        )
    args = common + (tile_range, tile_compute, fine_mask, maskr, mask_gm)
    art = _compile(
        args,
        q2d.dtype == torch.float16,
        uniform_q,
        uniform_kv,
        tasks_per_batch,
        d,
        sparse == 1,
        int(mask_stride),
        sparse,
        head_major,
    )
    if os.environ.get("FA_COMPILE_ONLY") == "1":
        return out[:, :d_in], smax, ssum
    art(*args, block_num=_block_num())
    return out[:, :d_in], smax, ssum


def run(
    query,
    key,
    value,
    atten_mask,
    actual_seq_qlen,
    actual_seq_kvlen,
    scale_value,
    head_num,
    input_layout,
    sparse_mode,
):
    """Return (attention_out, softmax_max, softmax_sum)."""
    layout = int(input_layout)
    sparse = int(sparse_mode)
    if sparse not in (0, 1, 2, 3):
        raise ValueError("unsupported sparse_mode: {}".format(sparse))

    if layout == 2:
        if sparse == 1:
            raise NotImplementedError("sparse_mode=1 is not supported with TND")
        t, nq, d = query.shape
        b = len(list(actual_seq_qlen))
        q_lens, cum_q = _lens(actual_seq_qlen, t, b)
        kv_lens, cum_kv = _lens(actual_seq_kvlen, key.shape[0], b)
        tile_arrays = None
        if sparse in (2, 3):
            tile_arrays = _mask_arrays(
                b,
                q_lens,
                kv_lens,
                sparse,
                max(q_lens),
                max(kv_lens),
                query.device,
            )
        # TND 的 actual_seqlen 约定为带前导 0 的累加数组（长 B+1）
        out, smax, ssum = _launch(
            query.reshape(-1, d),
            key.reshape(-1, d),
            value.reshape(-1, d),
            b,
            nq,
            key.shape[1],
            d,
            q_lens,
            kv_lens,
            [0] + cum_q,
            [0] + cum_kv,
            scale_value,
            0,
            0,
            0,
            0,
            0,
            (t, nq, 8),
            8,
            nq * 8,
            sparse=sparse,
            tile_arrays=tile_arrays,
        )
        return out.reshape(t, nq, d), smax, ssum

    if layout == 0:
        b, nq, s1, d = query.shape
        # BNSD 本身与 head-major 步长一一对应，kernel 直接按 [B,N,S,D] 读写，
        # 不再做 3 次输入 permute+contiguous 与 1 次输出 permute+contiguous。
        q4 = query.contiguous()
        k4 = key.contiguous()
        v4 = value.contiguous()
        nkv, s2 = key.shape[1], key.shape[2]
        head_major = True
    elif layout == 1:
        b, s1, hq = query.shape
        d = hq // int(head_num)
        nq = int(head_num)
        s2, hkv = key.shape[1], key.shape[2]
        nkv = hkv // d
        q4 = query.reshape(b, s1, nq, d).contiguous()
        k4 = key.reshape(b, s2, nkv, d).contiguous()
        v4 = value.reshape(b, s2, nkv, d).contiguous()
        head_major = False
    else:
        b, s1, nq, d = query.shape
        s2, nkv = key.shape[1], key.shape[2]
        q4, k4, v4 = query.contiguous(), key.contiguous(), value.contiguous()
        head_major = False

    q_lens = _lens(actual_seq_qlen, s1, b)[0]
    kv_lens = _lens(actual_seq_kvlen, s2, b)[0]
    # kernel 定长分支用 uniform_kv_seqlen 同时作 KV 批量步长与 KV 长度，q 定长而
    # kv 变长时会得到 kvSLoopNum==0、任务体被跳过、输出保持未初始化；变长分支的
    # 逐 batch 寻址又只对 TND 打包数据成立。参照 flash_attention_infer 用 is_uniform
    # （q 与 kv 同时定长）门控 q/kv：非 TND 下变长（或与张量维度不一致）的
    # actual_seq_* 直接报错，避免静默错误。
    is_uniform = len(set(q_lens)) == 1 and len(set(kv_lens)) == 1
    if not is_uniform or q_lens[0] != s1 or kv_lens[0] != s2:
        raise ValueError(
            "actual_seq_qlen/actual_seq_kvlen must be uniform and match the "
            "tensor dims for non-TND layouts; use TND for varlen sequences"
        )
    uniform_q = q_lens[0]
    uniform_kv = kv_lens[0]
    tasks_per_batch = nq * ((uniform_q + 127) // 128)
    if head_major and not (uniform_q and uniform_kv):
        # head-major 步长依赖固定的每 batch 序列长度，变长时退回打包布局
        q4 = q4.permute(0, 2, 1, 3).contiguous()
        k4 = k4.permute(0, 2, 1, 3).contiguous()
        v4 = v4.permute(0, 2, 1, 3).contiguous()
        head_major = False
    # (B, N, S, 8) 直接由 kernel 写出（batch/head 步长在 kernel 内展开），
    # 省掉 2 次 host 侧 permute+contiguous。非 uniform 时退回 (B, S, N, 8)。
    stats_bnsd = 1 if uniform_q else 0
    if stats_bnsd:
        stats_shape = (b, nq, s1, 8)
        sn_off, ss_off = s1 * 8, 8
    else:
        stats_shape = (b, s1, nq, 8)
        sn_off, ss_off = 8, nq * 8

    tile_arrays = None
    mask_gm = None
    mask_stride = 0
    if sparse in (1, 2, 3):
        if sparse == 1:
            if atten_mask is None:
                raise ValueError("sparse_mode=1 requires atten_mask")
            # attention mask: 非 0 = 遮蔽；kernel 内 1 = 可见
            pad_cols = ((int(atten_mask.shape[-1]) + 127) // 128) * 128
            mask_gm = _pack_dense_mask(atten_mask, pad_cols, query.device)
            mask_stride = pad_cols
        tile_arrays = _mask_arrays(
            b, q_lens, kv_lens, sparse, max(q_lens), max(kv_lens), query.device
        )

    out, smax, ssum = _launch(
        q4.reshape(-1, d),
        k4.reshape(-1, d),
        v4.reshape(-1, d),
        b,
        nq,
        nkv,
        d,
        q_lens,
        kv_lens,
        q_lens,
        kv_lens,
        scale_value,
        uniform_q,
        uniform_kv,
        tasks_per_batch,
        1,
        1,
        stats_shape,
        sn_off,
        ss_off,
        stats_bnsd=stats_bnsd,
        head_major=head_major,
        sparse=sparse,
        mask_gm=mask_gm,
        mask_stride=mask_stride,
        tile_arrays=tile_arrays,
    )
    if stats_bnsd:
        smax = smax.reshape(b, nq, s1, 8)
        ssum = ssum.reshape(b, nq, s1, 8)
    else:
        smax = smax.reshape(b, s1, nq, 8).permute(0, 2, 1, 3).contiguous()
        ssum = ssum.reshape(b, s1, nq, 8).permute(0, 2, 1, 3).contiguous()
    if layout == 0 and head_major:
        out = out.reshape(b, nq, s1, d)
    else:
        out = out.reshape(b, s1, nq, d)
        if layout == 0:
            out = out.permute(0, 2, 1, 3).contiguous()
        elif layout == 1:
            out = out.reshape(b, s1, nq * d)
    return out, smax, ssum


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlashAttentionScore (batch invariant) kernel"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--batch", type=int, default=2, help="batch size")
    parser.add_argument("--qseqlen", type=int, default=333, help="Q 序列长度之和")
    parser.add_argument("--kvseqlen", type=int, default=777, help="KV 序列长度之和")
    parser.add_argument("--headnum", type=int, default=8, help="Q 头数")
    parser.add_argument("--kvheadnum", type=int, default=2, help="KV 头数")
    parser.add_argument("--headdim", type=int, default=128, help="头维度，最大 128")
    parser.add_argument(
        "--layout", choices=("BNSD", "BSH", "TND", "BSND"), default="BNSD"
    )
    parser.add_argument("--sparse", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-batch-check",
        action="store_true",
        help="跳过整批与逐 batch 的逐位一致性比对",
    )
    return parser.parse_args()


def _split_lengths(total: int, parts: int) -> List[int]:
    """把 total 拆成 parts 段非等长序列（余数分给前几段），用于 TND 变长。"""
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _reference(q, k, v, mask, scale, sparse):
    """float32 参考实现，q/k/v 均为 (heads, seqlen, head_dim)。"""
    nq, s1, _ = q.shape
    nkv, s2, _ = k.shape
    rep = nq // nkv
    kk = k.repeat_interleave(rep, dim=0)
    vv = v.repeat_interleave(rep, dim=0)
    s = torch.matmul(q, kk.transpose(1, 2)) * scale
    hidden = None
    if sparse == 1:
        hidden = mask.bool().unsqueeze(0)
    elif sparse in (2, 3):
        i = torch.arange(s1, device=q.device).view(-1, 1)
        j = torch.arange(s2, device=q.device).view(1, -1)
        keep = j <= i if sparse == 2 else j <= i + (s2 - s1)
        hidden = ~keep.unsqueeze(0)
    if hidden is not None:
        s = s.masked_fill(hidden, float("-inf"))
    m = s.max(dim=-1).values
    p = torch.exp(s - m.unsqueeze(-1))
    lsum = p.sum(dim=-1)
    return torch.matmul(p, vv) / lsum.unsqueeze(-1), m, lsum


def _pack_tnd(t, lens: List[int]):
    """(B, heads, S, D) → (T, heads, D)，按 lens 逐 batch 打包。"""
    return torch.cat(
        [t[i, :, : lens[i]].transpose(0, 1) for i in range(len(lens))]
    ).contiguous()


def _unpack_tnd(t, lens: List[int]):
    """(T, heads, ...) → 逐 batch 的 (heads, len, ...) 列表。"""
    pieces, start = [], 0
    for n in lens:
        pieces.append(t[start : start + n].transpose(0, 1))
        start += n
    return pieces


def main() -> int:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = "npu:{}".format(args.device)
    torch.npu.set_device(device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    layout = {"BNSD": 0, "BSH": 1, "TND": 2, "BSND": 3}[args.layout]
    sparse = args.sparse
    b, s1, s2 = args.batch, args.qseqlen, args.kvseqlen
    nq, nkv, d = args.headnum, args.kvheadnum, args.headdim
    if nq % nkv:
        raise SystemExit("headnum 必须是 kvheadnum 的整数倍")
    if sparse == 1 and layout == 2:
        raise SystemExit("sparse_mode=1 不支持 TND 布局")
    scale = d**-0.5

    q0 = torch.randn(b, nq, s1, d, dtype=dtype, device=device)
    k0 = torch.randn(b, nkv, s2, d, dtype=dtype, device=device)
    v0 = torch.randn(b, nkv, s2, d, dtype=dtype, device=device)
    mask = None
    if sparse == 1:
        # 每行保留首列可见，避免出现整行全掩码（参考实现会得到 NaN）
        mask = (torch.rand(s1, s2, device=device) < 0.2).to(torch.uint8)
        mask[:, 0] = 0

    q_lens, kv_lens = [s1] * b, [s2] * b
    if layout == 2:
        q_lens, kv_lens = _split_lengths(s1, b), _split_lengths(s2, b)
        out, smax, ssum = run(
            _pack_tnd(q0, q_lens),
            _pack_tnd(k0, kv_lens),
            _pack_tnd(v0, kv_lens),
            None,
            list(itertools.accumulate(q_lens)),
            list(itertools.accumulate(kv_lens)),
            scale,
            nq,
            layout,
            sparse,
        )
        outs = _unpack_tnd(out, q_lens)
        smaxs = _unpack_tnd(smax, q_lens)
        ssums = _unpack_tnd(ssum, q_lens)
    else:
        if layout == 0:
            q, k, v = q0, k0, v0
        elif layout == 3:
            q = q0.transpose(1, 2).contiguous()
            k = k0.transpose(1, 2).contiguous()
            v = v0.transpose(1, 2).contiguous()
        else:
            q = q0.transpose(1, 2).reshape(b, s1, nq * d).contiguous()
            k = k0.transpose(1, 2).reshape(b, s2, nkv * d).contiguous()
            v = v0.transpose(1, 2).reshape(b, s2, nkv * d).contiguous()
        out, smax, ssum = run(q, k, v, mask, None, None, scale, nq, layout, sparse)
        if layout == 1:
            out = out.reshape(b, s1, nq, d).transpose(1, 2)
        elif layout == 3:
            out = out.transpose(1, 2)
        outs = [out[i] for i in range(b)]
        smaxs = [smax[i] for i in range(b)]
        ssums = [ssum[i] for i in range(b)]

    # 与参考实现的容差：attention_out 的量级受输入 dtype 的 P 转换精度限制；
    # 因果掩码下若某行只有 1~2 个可见列，单个 P 的舍入误差会直接体现在输出上，
    # 因此 bf16 的 out 容差需放到 1.5e-2（实测 8.9e-3），fp16 为 2e-3（实测 9.8e-4）。
    # smax/ssum 恒为 FP32，误差远小于 out。
    out_tol, max_tol, sum_tol = (
        (1.5e-2, 1e-4, 5e-3) if args.dtype == "bf16" else (2e-3, 1e-4, 5e-3)
    )
    accuracy_fail = 0
    for i in range(b):
        sq, sk = q_lens[i], kv_lens[i]
        o_ref, m_ref, l_ref = _reference(
            q0[i, :, :sq].float(),
            k0[i, :, :sk].float(),
            v0[i, :, :sk].float(),
            mask,
            scale,
            sparse,
        )
        out_err = (outs[i].float() - o_ref).abs().max().item()
        max_err = (smaxs[i].float()[:, :, 0] - m_ref).abs().max().item()
        sum_err = (ssums[i].float()[:, :, 0] - l_ref).abs().max().item()
        ok = out_err <= out_tol and max_err <= max_tol and sum_err <= sum_tol
        accuracy_fail += 0 if ok else 1
        print(
            "batch {}: out_err={:.6f} max_err={:.2e} sum_err={:.2e} {}".format(
                i, out_err, max_err, sum_err, "OK" if ok else "FAIL"
            ),
            flush=True,
        )
    print(
        "tolerance: out<={:.1e} max<={:.1e} sum<={:.1e}".format(
            out_tol, max_tol, sum_tol
        ),
        flush=True,
    )

    mistakes = accuracy_fail
    invariance_fail = 0
    if b > 1 and not args.skip_batch_check:
        for i in range(b):
            single = None
            if layout == 0:
                single = run(
                    q0[i : i + 1],
                    k0[i : i + 1],
                    v0[i : i + 1],
                    mask,
                    None,
                    None,
                    scale,
                    nq,
                    0,
                    sparse,
                )[0][0]
            elif layout == 3:
                qb = q0[i : i + 1].transpose(1, 2).contiguous()
                kb = k0[i : i + 1].transpose(1, 2).contiguous()
                vb = v0[i : i + 1].transpose(1, 2).contiguous()
                single = run(qb, kb, vb, mask, None, None, scale, nq, 3, sparse)[0][
                    0
                ].transpose(0, 1)
            if single is not None and not torch.equal(single, outs[i]):
                invariance_fail += 1
        mistakes += invariance_fail
        print("batch_invariance_mismatch={}".format(invariance_fail), flush=True)

    print(
        "layout={} sparse={} dtype={} shape=({},{},{},{},{})".format(
            args.layout, sparse, args.dtype, b, s1, s2, nq, d
        ),
        flush=True,
    )
    print("RESULT {}".format("PASS" if mistakes == 0 else "FAIL"), flush=True)
    return 0 if mistakes == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
