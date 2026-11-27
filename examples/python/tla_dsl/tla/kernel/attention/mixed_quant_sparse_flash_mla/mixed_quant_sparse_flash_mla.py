# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Mixed quantized sparse flash MLA attention (single-file CATLASS DSL, device + host).

Computes, for every query token,
    S = Q @ K^T * softmax_scale            (head_dim = 512, rope_head_dim = 64)
    O = softmax(S) @ V
with 608-byte quantized KV records: 64 bf16 rope elements, 448 FP8 E4M3 latent
elements and 7 per-token-group bf16 scales.

Modules:
  common       -- constants, buffer byte table, sync objects and tiling helpers
  dequant_kv   -- 608-byte record -> bf16 dequantization kernel (AIV)
  block_cube   -- cube side: Q/K loading, bmm1, bmm2, fixpipe
  block_vector -- vector side: softmax, vec1/vec2, output copy, sparse gather
  *_kernel     -- branch kernels: swa / hca / csa / ori_sparse / ori_cmp_sparse
  attention    -- shape dispatch, tiling / workspace / physical address table, executor cache
  run          -- dequantization orchestration, launch and view reshaping (public entry)
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
import sys

import torch
import torch_npu  # noqa: F401  -- registers torch.npu

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


def static_range(start, stop=None):
    if stop is None:
        return tuple(builtins.range(int(start)))
    return tuple(builtins.range(int(start), int(stop)))


HEAD_DIM = 512  # dSize
Q_ROWS = 64
KV_BLOCK = 128  # s2BaseSize（csa_kernel_arch35.h:160）
D_CHUNK = 128
NUM_D_CHUNKS = 4
V_SEGS = 4
Q_HALF = 256  # dSize >> 1（qHalfNum=2）
KV_SLOTS = 3
P_SLOTS = 2  # l1P[2]
L0_SLOTS = 2

KV_UNIT = 16
KV_COPY_IN = 8
KV_STAGE_SLOTS = KV_SLOTS

IDX_RING_SLOTS = 2
IDX_RING_STRIDE = 72

MIN_F = -3.4028234663852886e38  # NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF

INIT_POP_ELE = 512


@dataclass
class Sync:
    s_ready: object
    s_free: object
    p_ready: object
    o_ready: object
    o_free: object
    q_vector_ready: object
    q_vector_free: object
    kv_l1_free: object
    kv_stage_ready: object
    q_l1_ready: object  # MTE2 -> MTE1
    q_second_free: object
    q_second_ready: object
    q_l0a_free: object  # MTE1 -> MTE2
    k_l1_ready0: object
    k_l1_ready1: object
    k_l1_ready2: object
    k_l0b_free0: object
    k_l0b_free1: object
    k_l0b_free2: object
    s_mm_ready: object  # CUBE -> FIX
    s_mm_ready1: object
    s_fix_done: object  # FIX -> CUBE
    s_fix_done1: object
    o_mm_ready0: object
    o_mm_ready1: object
    o_fix_done0: object
    o_fix_done1: object
    p_data_ready: object
    lse_v_mte3: object
    lse_mte3_v: object
    kv_mte2_mte3_0: object
    kv_mte2_mte3_1: object
    kv_mte3_mte2_0: object
    kv_mte3_mte2_1: object
    init_mte3_v: object
    init_v_mte3: object
    phy_m2v: object
    phy_s2v: object
    phy_v_mte3: object
    phy_m3v: object
    phy_m3s: object
    phy_vs: object
    m_dq_rec0: object
    m_dq_rec1: object
    m_dq_out0: object
    m_dq_out1: object
    idx_m2s0: object
    idx_m2s1: object
    idx_s2m0: object
    idx_s2m1: object
    # F3 batch-consistency（OT INNERCORE_INTRALSE_MTE3_MTE2 / StageVec2PartialOAndWait
    bc_m3m2: object
    bc_m3v: object
    bc_m2v: object
    bc_v2m: object
    bc_v3m: object
    p_free: object
    m_l0a0: object
    m_l0a1: object
    m_l0b0: object
    m_l0b1: object
    m_o16: object
    m_p: object
    m_sink: object
    m_l0c_o0: object
    m_l0c_o1: object
    ab_ready0: object
    ab_ready1: object
    ab_free0: object
    ab_free1: object
    # CrossCoreSetFlag<0,PIPE_MTE2>(15) / CrossCoreWaitFlag<0,PIPE_MTE2>(15)）。
    # ``cross_core_set_flag(aic_ffts, FIX)`` + ``cross_core_wait_flag(aic_ffts, SCALAR)``
    s2_mte2: object
    s2_rendezvous: object
    decode_exchange: object


@dataclass
class Bufs:
    l1_p: object
    l1_q: object
    l1_k: object
    l1_ori: object
    l0a0: object
    l0a1: object
    l0b0: object
    l0b1: object
    l0c_s: object
    l0c_o: object
    ub_s: object
    ub_p: object
    ub_otmp_full: object
    ub_o: object
    ub_o16: object
    ub_sink: object
    ub_max: object
    ub_sum: object
    ub_dm: object
    ub_kv0: object
    ub_kv1: object
    ub_init: object
    ub_pv_stats: object
    ub_init_lse: object
    ub_phy_idx: object
    ub_phy_bt: object
    ub_phy_addr: object
    ub_idx_ring: object
    ub_bc_prev: object
    ub_bc_max: object  # staged prev lse（align8(rows) fp32，OT blockMaxUb slot0）
    ub_bc_sum: object  # staged prev sum（align8(rows) fp32，OT blockSumUb slot0）
    ub_bc_w: object


@tla.jit
def compute_need_init(cu_q, cu_kv, num_seqs, ori_mask_mode, ori_template):
    need_init = tla.as_numeric(0)
    if ori_template == 1:
        need_init = tla.as_numeric(1)
    else:
        if ori_mask_mode == 0:
            need_init = tla.as_numeric(1)
        else:
            for b in static_range(num_seqs):
                s1 = cu_q[b + 1] - cu_q[b]
                s2 = cu_kv[b + 1] - cu_kv[b]
                if s1 > s2:
                    need_init = tla.as_numeric(1)
    return need_init


@tla.jit
def declare_sync(
    need_rendezvous: bool = False,
    need_idx_ring: bool = False,
    need_lse_war: bool = False,
    need_fused_dequant: bool = False,
    two_q_slots: bool = False,
    direct_kv: bool = False,
    vector_query: bool = False,
) -> Sync:
    def cross(name, mode=4):
        return tla.cross_flag(name, mode=mode)

    return Sync(
        s_ready=cross("smla_s_ready"),
        s_free=cross("smla_s_free"),
        p_ready=cross("smla_p_ready"),
        o_ready=cross("smla_o_ready"),
        o_free=cross("smla_o_free"),
        kv_stage_ready=cross("smla_kv_stage_ready"),
        kv_l1_free=cross("smla_kv_l1_free") if direct_kv else None,
        q_vector_ready=cross("smla_q_vector_ready") if vector_query else None,
        q_vector_free=cross("smla_q_vector_free") if vector_query else None,
        s2_rendezvous=(
            cross("smla_s2_rendezvous", mode=0) if need_rendezvous else None
        ),
        #   dst=MTE1: q_l1_ready 0, k_l1_ready 1-3 / dst=MTE2: q_l0a 0, k_l0b
        #             1-3, kv_m3m2 4-5, bc_m3m2 6, bc_v2m 7 / dst=MTE3: kv_m2m3
        #             0-1, p_data 2,
        #             lse 3, init_v_mte3 4, phy_v_mte3 5, bc_v3m 6, p_free 7
        #   dst=V: init_mte3_v 0, phy_m2v 1, phy_s2v 2, bc_m2v 3, phy_m3v 4,
        #          o_fix_done 1-2 /
        #   dst=S: phy_m3s 0, phy_vs 1 /
        q_second_free=(
            tla.flag("q_second_free", tla.pipes.MTE1, tla.pipes.MTE2, id=4)
            if two_q_slots
            else None
        ),
        q_second_ready=(
            tla.flag("q_second_ready", tla.pipes.MTE2, tla.pipes.MTE1, id=6)
            if two_q_slots
            else None
        ),
        q_l1_ready=tla.flag("smla_q_l1_ready", tla.pipes.MTE2, tla.pipes.MTE1, id=0),
        q_l0a_free=tla.flag("smla_q_l0a_free", tla.pipes.MTE1, tla.pipes.MTE2, id=0),
        k_l1_ready0=tla.flag("smla_k_l1_ready0", tla.pipes.MTE2, tla.pipes.MTE1, id=1),
        k_l1_ready1=tla.flag("smla_k_l1_ready1", tla.pipes.MTE2, tla.pipes.MTE1, id=2),
        k_l1_ready2=tla.flag("smla_k_l1_ready2", tla.pipes.MTE2, tla.pipes.MTE1, id=3),
        k_l0b_free0=tla.flag("smla_k_l0b_free0", tla.pipes.MTE1, tla.pipes.MTE2, id=1),
        k_l0b_free1=tla.flag("smla_k_l0b_free1", tla.pipes.MTE1, tla.pipes.MTE2, id=2),
        k_l0b_free2=tla.flag("smla_k_l0b_free2", tla.pipes.MTE1, tla.pipes.MTE2, id=3),
        s_mm_ready=tla.flag("smla_s_mm_ready", tla.pipes.CUBE, tla.pipes.FIX, id=0),
        s_mm_ready1=tla.flag("smla_s_mm_ready1", tla.pipes.CUBE, tla.pipes.FIX, id=3),
        s_fix_done=tla.flag("smla_s_fix_done", tla.pipes.FIX, tla.pipes.CUBE, id=0),
        s_fix_done1=tla.flag("smla_s_fix_done1", tla.pipes.FIX, tla.pipes.CUBE, id=3),
        o_mm_ready0=tla.flag("smla_o_mm_ready0", tla.pipes.CUBE, tla.pipes.FIX, id=1),
        o_mm_ready1=tla.flag("smla_o_mm_ready1", tla.pipes.CUBE, tla.pipes.FIX, id=2),
        o_fix_done0=tla.flag("smla_o_fix_done0", tla.pipes.FIX, tla.pipes.CUBE, id=1),
        o_fix_done1=tla.flag("smla_o_fix_done1", tla.pipes.FIX, tla.pipes.CUBE, id=2),
        p_data_ready=tla.flag(
            "smla_p_data_ready", tla.pipes.VECTOR, tla.pipes.MTE3, id=2
        ),
        lse_v_mte3=tla.flag("smla_lse_v_mte3", tla.pipes.VECTOR, tla.pipes.MTE3, id=3),
        lse_mte3_v=(
            tla.flag("smla_lse_mte3_v", tla.pipes.MTE3, tla.pipes.VECTOR, id=6)
            if need_lse_war
            else None
        ),
        kv_mte2_mte3_0=tla.flag("smla_kv_m2m3_0", tla.pipes.MTE2, tla.pipes.MTE3, id=0),
        kv_mte2_mte3_1=tla.flag("smla_kv_m2m3_1", tla.pipes.MTE2, tla.pipes.MTE3, id=1),
        kv_mte3_mte2_0=tla.flag("smla_kv_m3m2_0", tla.pipes.MTE3, tla.pipes.MTE2, id=4),
        kv_mte3_mte2_1=tla.flag("smla_kv_m3m2_1", tla.pipes.MTE3, tla.pipes.MTE2, id=5),
        init_mte3_v=tla.flag(
            "smla_init_mte3_v", tla.pipes.MTE3, tla.pipes.VECTOR, id=0
        ),
        init_v_mte3=tla.flag(
            "smla_init_v_mte3", tla.pipes.VECTOR, tla.pipes.MTE3, id=4
        ),
        phy_m2v=tla.flag("smla_phy_m2v", tla.pipes.MTE2, tla.pipes.VECTOR, id=1),
        phy_s2v=tla.flag("smla_phy_s2v", tla.pipes.SCALAR, tla.pipes.VECTOR, id=2),
        phy_v_mte3=tla.flag("smla_phy_v_mte3", tla.pipes.VECTOR, tla.pipes.MTE3, id=5),
        phy_m3v=tla.flag("smla_phy_m3v", tla.pipes.MTE3, tla.pipes.VECTOR, id=4),
        phy_m3s=tla.flag("smla_phy_m3s", tla.pipes.MTE3, tla.pipes.SCALAR, id=0),
        phy_vs=tla.flag("smla_phy_vs", tla.pipes.VECTOR, tla.pipes.SCALAR, id=1),
        idx_m2s0=(
            tla.flag("smla_idx_m2s0", tla.pipes.MTE2, tla.pipes.SCALAR, id=2)
            if need_idx_ring
            else None
        ),
        idx_m2s1=(
            tla.flag("smla_idx_m2s1", tla.pipes.MTE2, tla.pipes.SCALAR, id=3)
            if need_idx_ring
            else None
        ),
        idx_s2m0=(
            tla.flag("smla_idx_s2m0", tla.pipes.SCALAR, tla.pipes.MTE2, id=6)
            if need_idx_ring
            else None
        ),
        idx_s2m1=(
            tla.flag("smla_idx_s2m1", tla.pipes.SCALAR, tla.pipes.MTE2, id=7)
            if need_idx_ring
            else None
        ),
        bc_m3m2=tla.flag("smla_bc_m3m2", tla.pipes.MTE3, tla.pipes.MTE2, id=6),
        bc_m3v=tla.flag("smla_bc_m3v", tla.pipes.MTE3, tla.pipes.VECTOR, id=5),
        bc_m2v=tla.flag("smla_bc_m2v", tla.pipes.MTE2, tla.pipes.VECTOR, id=3),
        bc_v2m=tla.flag("smla_bc_v2m", tla.pipes.VECTOR, tla.pipes.MTE2, id=7),
        p_free=cross("smla_p_free"),
        bc_v3m=tla.flag("smla_bc_v3m", tla.pipes.VECTOR, tla.pipes.MTE3, id=6),
        ab_ready0=tla.flag("smla_ab_ready0", tla.pipes.MTE1, tla.pipes.CUBE, id=4),
        ab_ready1=tla.flag("smla_ab_ready1", tla.pipes.MTE1, tla.pipes.CUBE, id=5),
        ab_free0=tla.flag("smla_ab_free0", tla.pipes.CUBE, tla.pipes.MTE1, id=4),
        ab_free1=tla.flag("smla_ab_free1", tla.pipes.CUBE, tla.pipes.MTE1, id=5),
        s2_mte2=cross("smla_s2_mte2", mode=0),
        decode_exchange=cross("smla_decode_exchange", mode=0),
        m_dq_rec0=(tla.mutex("smla_dq_rec0", id=9) if need_fused_dequant else None),
        m_dq_rec1=(tla.mutex("smla_dq_rec1", id=10) if need_fused_dequant else None),
        m_dq_out0=(tla.mutex("smla_dq_out0", id=11) if need_fused_dequant else None),
        m_dq_out1=(tla.mutex("smla_dq_out1", id=12) if need_fused_dequant else None),
        m_l0a0=tla.mutex("smla_l0a0", id=0),
        m_l0a1=tla.mutex("smla_l0a1", id=1),
        m_l0b0=tla.mutex("smla_l0b0", id=2),
        m_l0b1=tla.mutex("smla_l0b1", id=3),
        m_o16=tla.mutex("smla_o16", id=4),
        m_p=tla.mutex("smla_p", id=5),
        m_sink=tla.mutex("smla_sink", id=6),
        m_l0c_o0=tla.mutex("smla_l0c_o0", id=7),
        m_l0c_o1=tla.mutex("smla_l0c_o1", id=8),
    )


@tla.jit
def init_sync(
    sync: Sync,
    defer_ofree: bool = False,
    pre_phy_vs: bool = False,
    pre_bc: bool = False,
    pre_l0c2: bool = False,
    pre_idx_ring: bool = False,
) -> None:
    with tla.cube():
        tla.set_flag(sync.q_l0a_free)
        tla.set_flag(sync.k_l0b_free0)
        tla.set_flag(sync.k_l0b_free1)
        tla.set_flag(sync.k_l0b_free2)
        tla.set_flag(sync.ab_free0)
        tla.set_flag(sync.ab_free1)
        tla.set_flag(sync.s_fix_done)
        if pre_l0c2:
            tla.set_flag(sync.s_fix_done1)
        tla.set_flag(sync.o_fix_done0)
        tla.set_flag(sync.o_fix_done1)
    with tla.vector():
        tla.set_flag(sync.kv_mte3_mte2_0)
        tla.set_flag(sync.kv_mte3_mte2_1)
        if pre_phy_vs:
            tla.set_flag(sync.phy_vs)
        if pre_idx_ring:
            tla.set_flag(sync.idx_s2m0)
            tla.set_flag(sync.idx_s2m1)
        if pre_bc:
            tla.set_flag(sync.bc_v2m)
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=1)
        if not defer_ofree:
            tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=1)


@tla.jit
def init_sync_preload(sync: Sync) -> None:
    with tla.vector():
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=1)


@tla.jit
def alloc_buffers(
    dtype_in,
    row_cap=None,
    sink_cap=None,
    phy=0,
    phy_bt=0,
    bc=0,
    preload=False,
    idx_ring=False,
    rolling=False,
    kv_pad=0,
    kv_raw_pad=0,
    stat_pad=0,
    keep_q=False,
    gather32=False,
    cache_ori: tla.Constexpr[bool] = False,
) -> Bufs:
    # ---- L1：l1P[2] 32,768 | l1Q 3x32,768 | l1K(V) 3x131,072 = 524,288 B ----
    l1_p = tla.allocate(
        Q_ROWS * KV_BLOCK * (2 if preload else 1), dtype_in, tla.AddressSpace.l1, 512
    )
    l1_q = tla.allocate(
        (2 if cache_ori else (4 if rolling else 3)) * Q_ROWS * Q_HALF,
        dtype_in,
        tla.AddressSpace.l1,
        512,
    )
    l1_k = tla.allocate(
        (2 if rolling else KV_SLOTS) * HEAD_DIM * KV_BLOCK,
        dtype_in,
        tla.AddressSpace.l1,
        512,
    )
    l1_ori = (
        tla.allocate(D_CHUNK * KV_BLOCK, dtype_in, tla.AddressSpace.l1, 512)
        if cache_ori
        else None
    )
    # L0A: two working banks for Q/P, with two additional resident Q segments for folded CSA.
    l0a_size = Q_ROWS * D_CHUNK * (2 if keep_q else 1)
    l0a0 = tla.allocate(l0a_size, dtype_in, tla.AddressSpace.l0a, 512)
    l0a1 = tla.allocate(l0a_size, dtype_in, tla.AddressSpace.l0a, 512)
    l0b0 = tla.allocate(D_CHUNK * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
    l0b1 = tla.allocate(D_CHUNK * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
    l0c_s = tla.allocate(
        Q_ROWS * KV_BLOCK * (2 if preload else 1),
        tla.Float32,
        tla.AddressSpace.l0c,
        1024,
    )
    l0c_o = tla.allocate(Q_ROWS * HEAD_DIM, tla.Float32, tla.AddressSpace.l0c, 1024)
    half = Q_ROWS // 2 if row_cap is None else row_cap
    ub_s = tla.allocate(
        half * KV_BLOCK * (2 if preload else 1), tla.Float32, tla.AddressSpace.ub, 256
    )
    ub_p = tla.allocate(half * KV_BLOCK, dtype_in, tla.AddressSpace.ub, 256)
    ub_otmp_full = tla.allocate(half * HEAD_DIM, tla.Float32, tla.AddressSpace.ub, 256)
    ub_o = tla.allocate(half * HEAD_DIM, tla.Float32, tla.AddressSpace.ub, 256)
    ub_o16 = tla.allocate(half * HEAD_DIM, dtype_in, tla.AddressSpace.ub, 256)
    sink_len = half if sink_cap is None else sink_cap
    ub_sink = tla.allocate(sink_len, tla.Float32, tla.AddressSpace.ub, 256)
    ub_max = tla.allocate(
        half * (2 if preload else 1) + stat_pad, tla.Float32, tla.AddressSpace.ub, 256
    )
    # Folded CSA caches its initial denominators in a full SIMD register
    # after the two running-sum banks.
    sum_pad = VL if cache_ori else stat_pad
    ub_sum = tla.allocate(
        half * (2 if preload else 1) + sum_pad, tla.Float32, tla.AddressSpace.ub, 256
    )
    ub_dm = tla.allocate(
        half * (2 if preload else 1), tla.Float32, tla.AddressSpace.ub, 256
    )
    raw_rows = 2 * KV_UNIT if gather32 else KV_UNIT + kv_raw_pad
    ub_kv0 = tla.allocate(raw_rows * HEAD_DIM, dtype_in, tla.AddressSpace.ub, 256)
    ub_kv1 = (
        ub_o16
        if gather32
        else tla.allocate(
            (KV_UNIT + kv_pad) * HEAD_DIM, dtype_in, tla.AddressSpace.ub, 256
        )
    )
    ub_init = tla.allocate(INIT_POP_ELE, dtype_in, tla.AddressSpace.ub, 256)
    ub_init_lse = tla.allocate(INIT_POP_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_pv_stats = (
        tla.allocate(128, tla.Float32, tla.AddressSpace.ub, 256) if stat_pad else None
    )
    phy_cap = phy if phy > 0 else 1
    bt_cap = phy_bt if phy_bt > 0 else 1
    bt_cap = ((bt_cap + 63) // 64) * 64
    ub_phy_idx = tla.allocate(phy_cap, tla.Int32, tla.AddressSpace.ub, 256)
    ub_phy_bt = tla.allocate(bt_cap, tla.Int32, tla.AddressSpace.ub, 256)
    ub_phy_addr = tla.allocate(phy_cap * 2, tla.Int32, tla.AddressSpace.ub, 256)
    ub_idx_ring = (
        tla.allocate(
            IDX_RING_SLOTS * IDX_RING_STRIDE, tla.Int32, tla.AddressSpace.ub, 32
        )
        if idx_ring
        else None
    )
    bc_cap = half if bc == 1 else 1
    bc_cap8 = ((bc_cap + 7) // 8) * 8 if bc == 1 else 1
    ub_bc_prev = tla.allocate(bc_cap * D_CHUNK, tla.Float32, tla.AddressSpace.ub, 256)
    ub_bc_max = tla.allocate(bc_cap8, tla.Float32, tla.AddressSpace.ub, 256)
    ub_bc_sum = tla.allocate(bc_cap8, tla.Float32, tla.AddressSpace.ub, 256)
    ub_bc_w = tla.allocate(bc_cap8 * 2, tla.Float32, tla.AddressSpace.ub, 256)
    return Bufs(
        l1_p=l1_p,
        l1_q=l1_q,
        l1_k=l1_k,
        l1_ori=l1_ori,
        l0a0=l0a0,
        l0a1=l0a1,
        l0b0=l0b0,
        l0b1=l0b1,
        l0c_s=l0c_s,
        l0c_o=l0c_o,
        ub_s=ub_s,
        ub_p=ub_p,
        ub_otmp_full=ub_otmp_full,
        ub_o=ub_o,
        ub_o16=ub_o16,
        ub_sink=ub_sink,
        ub_max=ub_max,
        ub_sum=ub_sum,
        ub_dm=ub_dm,
        ub_kv0=ub_kv0,
        ub_kv1=ub_kv1,
        ub_init=ub_init,
        ub_init_lse=ub_init_lse,
        ub_pv_stats=ub_pv_stats,
        ub_phy_idx=ub_phy_idx,
        ub_phy_bt=ub_phy_bt,
        ub_phy_addr=ub_phy_addr,
        ub_idx_ring=ub_idx_ring,
        ub_bc_prev=ub_bc_prev,
        ub_bc_max=ub_bc_max,
        ub_bc_sum=ub_bc_sum,
        ub_bc_w=ub_bc_w,
    )


NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
REC_BYTES = 608
GROUP_SIZE = 64
SCALE_GROUPS = NOPE_DIM // GROUP_SIZE
SCALE_STRIDE = 8


@dataclass(frozen=True)
class DequantCfg:
    rows_per_block: tla.Constexpr[int] = 64


@dataclass(frozen=True)
class DequantPagesCfg:
    rows_per_block: tla.Constexpr[int] = 64
    rows_per_page: tla.Constexpr[int] = 128


@dataclass(frozen=True)
class DequantPagesPairCfg:
    rows_per_block_a: tla.Constexpr[int] = 64
    rows_per_page_a: tla.Constexpr[int] = 128
    rows_per_block_b: tla.Constexpr[int] = 64
    rows_per_page_b: tla.Constexpr[int] = 128
    # The page-list tensors include padding for aligned transfers. Decode each
    # referenced page once, before using the same storage for the index plan.
    pages_a: tla.Constexpr[int] = 0
    pages_b: tla.Constexpr[int] = 0


def _cast_slot(slot):
    return tla.params.CastParams(
        reg_slot=slot,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )


def _widen_slot(slot):
    return tla.params.CastParams(reg_slot=slot)


def _emit_group(ub_s, ub_o, row, group, val, preg, output_mask, cast_pack, pack_store):
    """Apply the group scale and store in the attention input dtype."""
    brc = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
    scale = tla.get_tile(ub_s, tla.make_coord(row, group), tla.make_shape(1, 1)).load(
        params=brc
    )
    scaled = tla.mul(val, scale, mask=preg)
    tla.get_tile(
        ub_o, tla.make_coord(row, group * GROUP_SIZE), tla.make_shape(1, GROUP_SIZE)
    ).store(
        scaled.to(ub_o.ptr.dtype, cast_pack, mask=preg), pack_store, mask=output_mask
    )


def _store_dequant_rope(ub_r, ub_o, row, fp32_mask, output_mask, cast, pack):
    """The record stores RoPE in BF16 even when Q and dequantized KV are FP16."""
    source = tla.get_tile(ub_r, tla.make_coord(row, 0), tla.make_shape(1, ROPE_DIM))
    target = tla.get_tile(
        ub_o,
        tla.make_coord(row, NOPE_DIM),
        tla.make_shape(1, ROPE_DIM),
    )
    if ub_o.ptr.dtype == tla.BFloat16:
        target.store(source.load())
    else:
        unpack = tla.params.NormalLoadParams(
            load_dist=tla.params.LoadDist.DIST_UNPACK_B16
        )
        bf16_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.BFloat16)
        expanded = source.load(params=unpack).to(
            tla.Float32, _widen_slot(tla.params.RegSlot.ZERO), mask=bf16_mask
        )
        target.store(
            expanded.to(ub_o.ptr.dtype, cast, mask=fp32_mask),
            pack,
            mask=output_mask,
        )


@tla.kernel
def dequant_kv_records(
    gm_nope: tla.Tensor,
    gm_rope: tla.Tensor,
    gm_scale: tla.Tensor,
    gm_out: tla.Tensor,
    _cfg: tla.Constexpr[DequantCfg],
) -> None:
    """gm_nope[L, 608] + gm_rope[L, 64] + gm_scale[L, 7] -> gm_out[L, 512] bf16。"""
    rows = _cfg.rows_per_block
    n_row = gm_out.origin_shape[0]
    n_blocks = n_row // rows

    m_nope = tla.mutex("dq_nope", id=0)
    m_rope = tla.mutex("dq_rope", id=1)
    m_scale = tla.mutex("dq_scale", id=2)
    m_out = tla.mutex("dq_out", id=3)

    ub_n = tla.make_tensor(
        tla.allocate(rows * REC_BYTES, gm_nope.ptr.dtype, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(rows, REC_BYTES), tla.make_stride(REC_BYTES, 1)),
    )
    ub_r = tla.make_tensor(
        tla.allocate(rows * ROPE_DIM, tla.BFloat16, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(rows, ROPE_DIM), tla.make_stride(ROPE_DIM, 1)),
    )
    ub_s = tla.make_tensor(
        tla.allocate(rows * SCALE_STRIDE, tla.Float32, tla.AddressSpace.ub, 256),
        tla.make_layout(
            tla.make_shape(rows, SCALE_STRIDE), tla.make_stride(SCALE_STRIDE, 1)
        ),
    )
    ub_o = tla.make_tensor(
        tla.allocate(rows * HEAD_DIM, gm_out.ptr.dtype, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)),
    )

    c0 = 0
    with tla.vector():
        for blk in tla.range(tla.arch.block_idx(), n_blocks, tla.arch.block_num()):
            r0 = blk * rows

            m_nope.lock(pipe=tla.pipes.MTE2)
            tla.copy(
                ub_n,
                tla.get_tile(
                    gm_nope,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, REC_BYTES),
                ),
            )
            m_nope.unlock(pipe=tla.pipes.MTE2)
            m_rope.lock(pipe=tla.pipes.MTE2)
            tla.copy(
                ub_r,
                tla.get_tile(
                    gm_rope,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, ROPE_DIM),
                ),
            )
            m_rope.unlock(pipe=tla.pipes.MTE2)
            m_scale.lock(pipe=tla.pipes.MTE2)
            tla.copy(
                ub_s,
                tla.get_tile(
                    gm_scale,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, SCALE_STRIDE),
                ),
            )
            m_scale.unlock(pipe=tla.pipes.MTE2)

            m_nope.lock(pipe=tla.pipes.VECTOR)
            m_rope.lock(pipe=tla.pipes.VECTOR)
            m_scale.lock(pipe=tla.pipes.VECTOR)
            m_out.lock(pipe=tla.pipes.VECTOR)
            with tla.vec.func(mode="simd"):
                m8 = tla.create_mask(pattern=tla.mask.ALL, dtype=gm_nope.ptr.dtype)
                preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                output_mask = tla.create_mask(
                    pattern=tla.mask.ALL, dtype=ub_o.ptr.dtype
                )
                cast0 = _widen_slot(tla.params.RegSlot.ZERO)
                cast1 = _widen_slot(tla.params.RegSlot.ONE)
                cast2 = _widen_slot(tla.params.RegSlot.TWO)
                cast3 = _widen_slot(tla.params.RegSlot.THREE)
                cast_pack = _cast_slot(tla.params.RegSlot.ZERO)
                pack_store = tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_PACK_B32
                )
                for r in tla.range(rows):
                    for part in tla.range_constexpr(2):
                        off = 128 + part * 256
                        count = 256 if part == 0 else 192
                        reg = tla.get_tile(
                            ub_n, tla.make_coord(r, off), tla.make_shape(1, count)
                        ).load()
                        v0 = reg.to(tla.Float32, cast0, mask=m8)
                        v1 = reg.to(tla.Float32, cast1, mask=m8)
                        v2 = reg.to(tla.Float32, cast2, mask=m8)
                        v3 = reg.to(tla.Float32, cast3, mask=m8)
                        a0, a1 = tla.interleave(v0, v2)
                        b0, b1 = tla.interleave(v1, v3)
                        c00, c01 = tla.interleave(a0, b0)
                        c10, c11 = tla.interleave(a1, b1)
                        group = part * 4
                        _emit_group(
                            ub_s,
                            ub_o,
                            r,
                            group,
                            c00,
                            preg,
                            output_mask,
                            cast_pack,
                            pack_store,
                        )
                        _emit_group(
                            ub_s,
                            ub_o,
                            r,
                            group + 1,
                            c01,
                            preg,
                            output_mask,
                            cast_pack,
                            pack_store,
                        )
                        _emit_group(
                            ub_s,
                            ub_o,
                            r,
                            group + 2,
                            c10,
                            preg,
                            output_mask,
                            cast_pack,
                            pack_store,
                        )
                        if part == 0:
                            _emit_group(
                                ub_s,
                                ub_o,
                                r,
                                group + 3,
                                c11,
                                preg,
                                output_mask,
                                cast_pack,
                                pack_store,
                            )
                    _store_dequant_rope(
                        ub_r, ub_o, r, preg, output_mask, cast_pack, pack_store
                    )
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
            m_out.unlock(pipe=tla.pipes.VECTOR)
            m_rope.unlock(pipe=tla.pipes.VECTOR)
            m_scale.unlock(pipe=tla.pipes.VECTOR)
            m_nope.unlock(pipe=tla.pipes.VECTOR)

            m_out.lock(pipe=tla.pipes.MTE3)
            tla.copy(
                tla.get_tile(
                    gm_out,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, HEAD_DIM),
                ),
                ub_o,
            )
            m_out.unlock(pipe=tla.pipes.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def dequant_kv_pages(
    gm_nope: tla.Tensor,
    gm_rope: tla.Tensor,
    gm_scale: tla.Tensor,
    gm_pages: tla.Tensor,
    gm_out: tla.Tensor,
    _cfg: tla.Constexpr[DequantPagesCfg],
) -> None:
    rows = _cfg.rows_per_block
    rpp = _cfg.rows_per_page
    n_sub = rpp // rows
    n_pages = gm_pages.origin_shape[0]
    total = n_pages * n_sub

    m_nope = tla.mutex("dq_nope", id=0)
    m_rope = tla.mutex("dq_rope", id=1)
    m_scale = tla.mutex("dq_scale", id=2)
    m_out = tla.mutex("dq_out", id=3)

    ub_n = tla.make_tensor(
        tla.allocate(rows * REC_BYTES, gm_nope.ptr.dtype, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(rows, REC_BYTES), tla.make_stride(REC_BYTES, 1)),
    )
    ub_r = tla.make_tensor(
        tla.allocate(rows * ROPE_DIM, tla.BFloat16, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(rows, ROPE_DIM), tla.make_stride(ROPE_DIM, 1)),
    )
    ub_s = tla.make_tensor(
        tla.allocate(rows * SCALE_STRIDE, tla.Float32, tla.AddressSpace.ub, 256),
        tla.make_layout(
            tla.make_shape(rows, SCALE_STRIDE), tla.make_stride(SCALE_STRIDE, 1)
        ),
    )
    ub_o = tla.make_tensor(
        tla.allocate(rows * HEAD_DIM, gm_out.ptr.dtype, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)),
    )

    c0 = 0
    with tla.vector():
        for t in tla.range(tla.arch.block_idx(), total, tla.arch.block_num()):
            p = t // n_sub
            s = t - p * n_sub
            page = gm_pages[p]
            if page < 0:
                page = 0
            r0 = page * rpp + s * rows
            rs = p * rpp + s * rows

            m_nope.lock(pipe=tla.pipes.MTE2)
            tla.copy(
                ub_n,
                tla.get_tile(
                    gm_nope,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, REC_BYTES),
                ),
            )
            m_nope.unlock(pipe=tla.pipes.MTE2)
            m_rope.lock(pipe=tla.pipes.MTE2)
            tla.copy(
                ub_r,
                tla.get_tile(
                    gm_rope,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, ROPE_DIM),
                ),
            )
            m_rope.unlock(pipe=tla.pipes.MTE2)
            m_scale.lock(pipe=tla.pipes.MTE2)
            tla.copy(
                ub_s,
                tla.get_tile(
                    gm_scale,
                    tla.make_coord(rs, c0),
                    tla.make_shape(rows, SCALE_STRIDE),
                ),
            )
            m_scale.unlock(pipe=tla.pipes.MTE2)

            m_nope.lock(pipe=tla.pipes.VECTOR)
            m_rope.lock(pipe=tla.pipes.VECTOR)
            m_scale.lock(pipe=tla.pipes.VECTOR)
            m_out.lock(pipe=tla.pipes.VECTOR)
            with tla.vec.func(mode="simd"):
                m8 = tla.create_mask(pattern=tla.mask.ALL, dtype=gm_nope.ptr.dtype)
                preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                output_mask = tla.create_mask(
                    pattern=tla.mask.ALL, dtype=ub_o.ptr.dtype
                )
                cast0 = _widen_slot(tla.params.RegSlot.ZERO)
                cast1 = _widen_slot(tla.params.RegSlot.ONE)
                cast2 = _widen_slot(tla.params.RegSlot.TWO)
                cast3 = _widen_slot(tla.params.RegSlot.THREE)
                cast_pack = _cast_slot(tla.params.RegSlot.ZERO)
                pack_store = tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_PACK_B32
                )
                for r in tla.range(rows):
                    for part in tla.range_constexpr(2):
                        off = 128 + part * 256
                        count = 256 if part == 0 else 192
                        reg = tla.get_tile(
                            ub_n, tla.make_coord(r, off), tla.make_shape(1, count)
                        ).load()
                        v0 = reg.to(tla.Float32, cast0, mask=m8)
                        v1 = reg.to(tla.Float32, cast1, mask=m8)
                        v2 = reg.to(tla.Float32, cast2, mask=m8)
                        v3 = reg.to(tla.Float32, cast3, mask=m8)
                        a0, a1 = tla.interleave(v0, v2)
                        b0, b1 = tla.interleave(v1, v3)
                        c00, c01 = tla.interleave(a0, b0)
                        c10, c11 = tla.interleave(a1, b1)
                        group = part * 4
                        _emit_group(
                            ub_s,
                            ub_o,
                            r,
                            group,
                            c00,
                            preg,
                            output_mask,
                            cast_pack,
                            pack_store,
                        )
                        _emit_group(
                            ub_s,
                            ub_o,
                            r,
                            group + 1,
                            c01,
                            preg,
                            output_mask,
                            cast_pack,
                            pack_store,
                        )
                        _emit_group(
                            ub_s,
                            ub_o,
                            r,
                            group + 2,
                            c10,
                            preg,
                            output_mask,
                            cast_pack,
                            pack_store,
                        )
                        if part == 0:
                            _emit_group(
                                ub_s,
                                ub_o,
                                r,
                                group + 3,
                                c11,
                                preg,
                                output_mask,
                                cast_pack,
                                pack_store,
                            )
                    _store_dequant_rope(
                        ub_r, ub_o, r, preg, output_mask, cast_pack, pack_store
                    )
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
            m_out.unlock(pipe=tla.pipes.VECTOR)
            m_rope.unlock(pipe=tla.pipes.VECTOR)
            m_scale.unlock(pipe=tla.pipes.VECTOR)
            m_nope.unlock(pipe=tla.pipes.VECTOR)

            m_out.lock(pipe=tla.pipes.MTE3)
            tla.copy(
                tla.get_tile(
                    gm_out,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, HEAD_DIM),
                ),
                ub_o,
            )
            m_out.unlock(pipe=tla.pipes.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.jit
def _dv_block(
    gm_nope,
    gm_rope,
    gm_scale,
    gm_out,
    ub_n,
    ub_r,
    ub_s,
    ub_o,
    r0,
    rs,
    rows,
    m_nope,
    m_rope,
    m_scale,
    m_out,
    raw_records: tla.Constexpr[bool] = False,
):
    c0 = 0
    m_nope.lock(pipe=tla.pipes.MTE2)
    tla.copy(
        ub_n,
        tla.get_tile(gm_nope, tla.make_coord(r0, c0), tla.make_shape(rows, REC_BYTES)),
    )
    m_nope.unlock(pipe=tla.pipes.MTE2)
    if not raw_records:
        m_rope.lock(pipe=tla.pipes.MTE2)
        tla.copy(
            ub_r,
            tla.get_tile(
                gm_rope, tla.make_coord(r0, c0), tla.make_shape(rows, ROPE_DIM)
            ),
        )
        m_rope.unlock(pipe=tla.pipes.MTE2)
        m_scale.lock(pipe=tla.pipes.MTE2)
        tla.copy(
            ub_s,
            tla.get_tile(
                gm_scale, tla.make_coord(rs, c0), tla.make_shape(rows, SCALE_STRIDE)
            ),
        )
        m_scale.unlock(pipe=tla.pipes.MTE2)

    m_nope.lock(pipe=tla.pipes.VECTOR)
    if not raw_records:
        m_rope.lock(pipe=tla.pipes.VECTOR)
        m_scale.lock(pipe=tla.pipes.VECTOR)
    m_out.lock(pipe=tla.pipes.VECTOR)
    if raw_records:
        _dequant_fp8_records_nd(
            tla.recast_ptr(ub_n.ptr, dtype=tla.UInt8), ub_o.ptr, rows
        )
    else:
        with tla.vec.func(mode="simd"):
            m8 = tla.create_mask(pattern=tla.mask.ALL, dtype=gm_nope.ptr.dtype)
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            output_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=ub_o.ptr.dtype)
            cast0 = _widen_slot(tla.params.RegSlot.ZERO)
            cast1 = _widen_slot(tla.params.RegSlot.ONE)
            cast2 = _widen_slot(tla.params.RegSlot.TWO)
            cast3 = _widen_slot(tla.params.RegSlot.THREE)
            cast_pack = _cast_slot(tla.params.RegSlot.ZERO)
            pack_store = tla.params.NormalStoreParams(
                store_dist=tla.params.StoreDist.DIST_PACK_B32
            )
            for r in tla.range(rows):
                for part in tla.range_constexpr(2):
                    off = 128 + part * 256
                    count = 256 if part == 0 else 192
                    reg = tla.get_tile(
                        ub_n, tla.make_coord(r, off), tla.make_shape(1, count)
                    ).load()
                    v0 = reg.to(tla.Float32, cast0, mask=m8)
                    v1 = reg.to(tla.Float32, cast1, mask=m8)
                    v2 = reg.to(tla.Float32, cast2, mask=m8)
                    v3 = reg.to(tla.Float32, cast3, mask=m8)
                    a0, a1 = tla.interleave(v0, v2)
                    b0, b1 = tla.interleave(v1, v3)
                    c00, c01 = tla.interleave(a0, b0)
                    c10, c11 = tla.interleave(a1, b1)
                    group = part * 4
                    _emit_group(
                        ub_s,
                        ub_o,
                        r,
                        group,
                        c00,
                        preg,
                        output_mask,
                        cast_pack,
                        pack_store,
                    )
                    _emit_group(
                        ub_s,
                        ub_o,
                        r,
                        group + 1,
                        c01,
                        preg,
                        output_mask,
                        cast_pack,
                        pack_store,
                    )
                    _emit_group(
                        ub_s,
                        ub_o,
                        r,
                        group + 2,
                        c10,
                        preg,
                        output_mask,
                        cast_pack,
                        pack_store,
                    )
                    if part == 0:
                        _emit_group(
                            ub_s,
                            ub_o,
                            r,
                            group + 3,
                            c11,
                            preg,
                            output_mask,
                            cast_pack,
                            pack_store,
                        )
                _store_dequant_rope(
                    ub_r, ub_o, r, preg, output_mask, cast_pack, pack_store
                )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    m_out.unlock(pipe=tla.pipes.VECTOR)
    if not raw_records:
        m_rope.unlock(pipe=tla.pipes.VECTOR)
        m_scale.unlock(pipe=tla.pipes.VECTOR)
    m_nope.unlock(pipe=tla.pipes.VECTOR)

    m_out.lock(pipe=tla.pipes.MTE3)
    tla.copy(
        tla.get_tile(gm_out, tla.make_coord(r0, c0), tla.make_shape(rows, HEAD_DIM)),
        ub_o,
    )
    m_out.unlock(pipe=tla.pipes.MTE3)


def _dv_buffers(rows, dtype_in, dtype_out, shared=None):
    if shared is None:
        data = tla.allocate(rows * REC_BYTES, dtype_in, tla.AddressSpace.ub, 256)
        rope = tla.allocate(rows * ROPE_DIM, tla.BFloat16, tla.AddressSpace.ub, 256)
        scale = tla.allocate(rows * SCALE_STRIDE, tla.Float32, tla.AddressSpace.ub, 256)
        output = tla.allocate(rows * HEAD_DIM, dtype_out, tla.AddressSpace.ub, 256)
    else:
        work = tla.recast_ptr(shared, dtype=tla.UInt8)
        data = tla.recast_ptr(work, dtype=dtype_in)
        rope = tla.recast_ptr(work + rows * REC_BYTES, dtype=tla.BFloat16)
        scale = tla.recast_ptr(
            work + rows * (REC_BYTES + 2 * ROPE_DIM), dtype=tla.Float32
        )
        output = tla.recast_ptr(
            work + rows * (REC_BYTES + 2 * ROPE_DIM + 4 * SCALE_STRIDE), dtype=dtype_out
        )
    ub_n = tla.make_tensor(
        data,
        tla.make_layout(tla.make_shape(rows, REC_BYTES), tla.make_stride(REC_BYTES, 1)),
    )
    ub_r = tla.make_tensor(
        rope,
        tla.make_layout(tla.make_shape(rows, ROPE_DIM), tla.make_stride(ROPE_DIM, 1)),
    )
    ub_s = tla.make_tensor(
        scale,
        tla.make_layout(
            tla.make_shape(rows, SCALE_STRIDE), tla.make_stride(SCALE_STRIDE, 1)
        ),
    )
    ub_o = tla.make_tensor(
        output,
        tla.make_layout(tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)),
    )
    return ub_n, ub_r, ub_s, ub_o


@tla.jit
def _dequant_page_pair(
    gm_nope_a: tla.Tensor,
    gm_rope_a: tla.Tensor,
    gm_scale_a: tla.Tensor,
    gm_pages_a: tla.Tensor,
    gm_out_a: tla.Tensor,
    gm_nope_b: tla.Tensor,
    gm_rope_b: tla.Tensor,
    gm_scale_b: tla.Tensor,
    gm_pages_b: tla.Tensor,
    gm_out_b: tla.Tensor,
    _cfg: tla.Constexpr[DequantPagesPairCfg],
    shared=None,
    fold_output_ready=None,
) -> None:
    rows_a = _cfg.rows_per_block_a
    rpp_a = _cfg.rows_per_page_a
    n_sub_a = rpp_a // rows_a
    total_a = (_cfg.pages_a or gm_pages_a.origin_shape[0]) * n_sub_a
    rows_b = _cfg.rows_per_block_b
    rpp_b = _cfg.rows_per_page_b
    n_sub_b = rpp_b // rows_b
    total_b = (_cfg.pages_b or gm_pages_b.origin_shape[0]) * n_sub_b
    total = total_a + total_b
    dtype_a = gm_nope_a.ptr.dtype
    dtype_b = gm_nope_b.ptr.dtype

    m_nope = tla.mutex("dq_nope", id=0)
    m_rope = tla.mutex("dq_rope", id=1)
    m_scale = tla.mutex("dq_scale", id=2)
    m_out = tla.mutex("dq_out", id=3)

    ub_na, ub_ra, ub_sa, ub_oa = _dv_buffers(
        rows_a, dtype_a, gm_out_a.ptr.dtype, shared
    )
    ub_nb, ub_rb, ub_sb, ub_ob = _dv_buffers(
        rows_b, dtype_b, gm_out_b.ptr.dtype, shared
    )

    with tla.vector():
        for t in tla.range(tla.arch.block_idx(), total, tla.arch.block_num()):
            if t < total_a:
                p = t // n_sub_a
                s = t - p * n_sub_a
                page = gm_pages_a[p]
                if page < 0:
                    page = 0
                r0 = page * rpp_a + s * rows_a
                rs = p * rpp_a + s * rows_a
                _dv_block(
                    gm_nope_a,
                    gm_rope_a,
                    gm_scale_a,
                    gm_out_a,
                    ub_na,
                    ub_ra,
                    ub_sa,
                    ub_oa,
                    r0,
                    rs,
                    rows_a,
                    m_nope,
                    m_rope,
                    m_scale,
                    m_out,
                    raw_records=shared is not None
                    and dtype_a == tla.Float8E4M3FN
                    and gm_out_a.ptr.dtype == tla.BFloat16,
                )
            else:
                t2 = t - total_a
                p = t2 // n_sub_b
                s = t2 - p * n_sub_b
                page = gm_pages_b[p]
                if page < 0:
                    page = 0
                r0 = page * rpp_b + s * rows_b
                rs = p * rpp_b + s * rows_b
                _dv_block(
                    gm_nope_b,
                    gm_rope_b,
                    gm_scale_b,
                    gm_out_b,
                    ub_nb,
                    ub_rb,
                    ub_sb,
                    ub_ob,
                    r0,
                    rs,
                    rows_b,
                    m_nope,
                    m_rope,
                    m_scale,
                    m_out,
                    raw_records=shared is not None
                    and dtype_b == tla.Float8E4M3FN
                    and gm_out_b.ptr.dtype == tla.BFloat16,
                )
        if fold_output_ready is None:
            tla.pipe_barrier(tla.pipes.ALL)
        else:
            tla.set_flag(fold_output_ready)


@tla.kernel
def dequant_kv_pages_pair(
    gm_nope_a: tla.Tensor,
    gm_rope_a: tla.Tensor,
    gm_scale_a: tla.Tensor,
    gm_pages_a: tla.Tensor,
    gm_out_a: tla.Tensor,
    gm_nope_b: tla.Tensor,
    gm_rope_b: tla.Tensor,
    gm_scale_b: tla.Tensor,
    gm_pages_b: tla.Tensor,
    gm_out_b: tla.Tensor,
    _cfg: tla.Constexpr[DequantPagesPairCfg],
) -> None:
    _dequant_page_pair(
        gm_nope_a,
        gm_rope_a,
        gm_scale_a,
        gm_pages_a,
        gm_out_a,
        gm_nope_b,
        gm_rope_b,
        gm_scale_b,
        gm_pages_b,
        gm_out_b,
        _cfg,
    )


@tla.jit
def l1_q_tensor(ptr, dtype, rows):
    return tla.make_tensor(
        ptr, tla.make_layout_with_tag((rows, Q_HALF), dtype, tla.arch.zN)
    )


@tla.jit
def l1_k_tensor(ptr, dtype):
    return tla.make_tensor(
        ptr, tla.make_layout_with_tag((HEAD_DIM, KV_BLOCK), dtype, tla.arch.nZ)
    )


@tla.jit
def l1_v_tensor(ptr, dtype):
    return tla.make_tensor(
        ptr, tla.make_layout_with_tag((KV_BLOCK, HEAD_DIM), dtype, tla.arch.zN)
    )


@tla.jit
def l1_v_seg_tensor(ptr, dtype):
    return tla.make_tensor(
        ptr, tla.make_layout_with_tag((KV_BLOCK, D_CHUNK), dtype, tla.arch.zN)
    )


@tla.jit
def l1_p_tensor(ptr, dtype, rows):
    return tla.make_tensor(
        ptr, tla.make_layout_with_tag((rows, KV_BLOCK), dtype, tla.arch.zN)
    )


@tla.jit
def load_q(sync, bufs, q_tensor, task, m, dtype, row_base=None, slot_a=0, slot_b=1):
    c0 = 0
    base = task * m if row_base is None else row_base
    with tla.cube():
        tla.wait_flag(sync.q_l0a_free)
        gm_q = tla.make_tensor(
            q_tensor.ptr + base * HEAD_DIM,
            tla.make_layout(
                tla.make_shape(m, HEAD_DIM),
                tla.make_stride(HEAD_DIM, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        gm0 = tla.tile_view(gm_q, tla.make_shape(m, Q_HALF), tla.make_coord(c0, c0))
        gm1 = tla.tile_view(gm_q, tla.make_shape(m, Q_HALF), tla.make_coord(c0, 1))
        l1q0 = l1_q_tensor(_q_slot_ptr(bufs, slot_a), dtype, m)
        l1q1 = l1_q_tensor(_q_slot_ptr(bufs, slot_b), dtype, m)
        tla.copy(l1q0, gm0)
        tla.copy(l1q1, gm1)
        tla.set_flag(sync.q_l1_ready)


@tla.jit
def _k_slot_wait(sync, slot):
    if slot == 0:
        tla.wait_flag(sync.k_l0b_free0)
    if slot == 1:
        tla.wait_flag(sync.k_l0b_free1)
    if slot == 2:
        tla.wait_flag(sync.k_l0b_free2)


@tla.jit
def _k_slot_free(sync, slot):
    if slot == 0:
        tla.set_flag(sync.k_l0b_free0)
    if slot == 1:
        tla.set_flag(sync.k_l0b_free1)
    if slot == 2:
        tla.set_flag(sync.k_l0b_free2)


@tla.jit
def _k_slot_ready_wait(sync, slot):
    if slot == 0:
        tla.wait_flag(sync.k_l1_ready0)
    if slot == 1:
        tla.wait_flag(sync.k_l1_ready1)
    if slot == 2:
        tla.wait_flag(sync.k_l1_ready2)


@tla.jit
def _k_slot_ready_set(sync, slot):
    if slot == 0:
        tla.set_flag(sync.k_l1_ready0)
    if slot == 1:
        tla.set_flag(sync.k_l1_ready1)
    if slot == 2:
        tla.set_flag(sync.k_l1_ready2)


@tla.jit
def _k_slot_ptr(bufs, slot):
    return bufs.l1_k + slot * (HEAD_DIM * KV_BLOCK)


@tla.jit
def _q_slot_ptr(bufs, slot):
    return bufs.l1_q + slot * (Q_ROWS * Q_HALF)


@tla.jit
def load_k_ptr(sync, bufs, src_ptr, n_load, slot, dtype):
    with tla.cube():
        _k_slot_wait(sync, slot)
        l1k = l1_k_tensor(_k_slot_ptr(bufs, slot), dtype)
        gm_k = tla.make_tensor(
            src_ptr,
            tla.make_layout(
                tla.make_shape(HEAD_DIM, n_load),
                tla.make_stride(1, HEAD_DIM),
                layoutTag=tla.arch.ColumnMajor,
            ),
        )
        tla.copy(
            tla.tile_view(l1k, tla.make_shape(HEAD_DIM, n_load), tla.make_coord(0, 0)),
            gm_k,
        )
        _k_slot_ready_set(sync, slot)


@tla.jit
def _load_k_ptr_nocredit(sync, bufs, src_ptr, n_load, slot, dtype):
    with tla.cube():
        _k_slot_wait(sync, slot)
        l1k = l1_k_tensor(_k_slot_ptr(bufs, slot), dtype)
        gm_k = tla.make_tensor(
            src_ptr,
            tla.make_layout(
                tla.make_shape(HEAD_DIM, n_load),
                tla.make_stride(1, HEAD_DIM),
                layoutTag=tla.arch.ColumnMajor,
            ),
        )
        tla.copy(
            tla.tile_view(l1k, tla.make_shape(HEAD_DIM, n_load), tla.make_coord(0, 0)),
            gm_k,
        )
        _k_slot_ready_set(sync, slot)


@tla.jit
def load_k(sync, bufs, kv_tensor, kv_row0, tile, n_load, slot, dtype, blk=KV_BLOCK):
    base = kv_row0 + tile * blk
    _load_k_ptr_nocredit(
        sync, bufs, kv_tensor.ptr + base * HEAD_DIM, n_load, slot, dtype
    )


@tla.jit
def load_k_pa(
    sync,
    bufs,
    kv_tensor,
    bt,
    seq,
    start_row,
    n_load,
    slot,
    dtype,
    pa_block_size,
    pa_stride,
    bt_max,
    pages_bound,
    fill_tail=0,
):
    with tla.cube():
        _k_slot_wait(sync, slot)
        l1k = l1_k_tensor(_k_slot_ptr(bufs, slot), dtype)
        done = tla.as_numeric(0)
        cur = start_row
        for _p in static_range(pages_bound):
            if done < n_load:
                page = cur // pa_block_size
                rem = cur - page * pa_block_size
                phys = bt[seq * bt_max + page]
                if phys < 0:
                    phys = tla.as_numeric(0)
                cnt = pa_block_size - rem
                if done + cnt > n_load:
                    cnt = n_load - done
                for c in (0, 1, 2, 3):
                    gm_k = tla.make_tensor(
                        kv_tensor.ptr + phys * pa_stride + rem * HEAD_DIM + c * D_CHUNK,
                        tla.make_layout(
                            tla.make_shape(D_CHUNK, cnt),
                            tla.make_stride(1, HEAD_DIM),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    k_tile = tla.get_tile(
                        l1k,
                        tla.make_coord(c * D_CHUNK, done),
                        tla.make_shape(D_CHUNK, cnt),
                    )
                    tla.copy(k_tile, gm_k)
                done = done + cnt
                cur = cur + cnt
        if fill_tail == 1:
            if n_load < KV_BLOCK:
                if n_load > 0:
                    lphys = bt[seq * bt_max]
                    if lphys < 0:
                        lphys = tla.as_numeric(0)
                    pad = KV_BLOCK - n_load
                    for c in (0, 1, 2, 3):
                        gm_z = tla.make_tensor(
                            kv_tensor.ptr + lphys * pa_stride + c * D_CHUNK,
                            tla.make_layout(
                                tla.make_shape(D_CHUNK, pad),
                                tla.make_stride(1, HEAD_DIM),
                                layoutTag=tla.arch.ColumnMajor,
                            ),
                        )
                        tla.copy(
                            tla.get_tile(
                                l1k,
                                tla.make_coord(c * D_CHUNK, n_load),
                                tla.make_shape(D_CHUNK, pad),
                            ),
                            gm_z,
                        )
        _k_slot_ready_set(sync, slot)


@tla.jit
def _pack_swa_query(
    packed: tla.Pointer[tla.UInt16, tla.AddressSpace.ub],
    raw: tla.Pointer[tla.UInt16, tla.AddressSpace.ub],
) -> None:
    raw_nd = tla.make_tensor(
        raw, tla.make_layout_with_tag((32, 512), raw.dtype, tla.arch.RowMajor)
    )
    packed_zn = tla.make_tensor(
        packed, tla.make_layout_with_tag((32, 512), packed.dtype, tla.arch.zN)
    )
    with tla.vec.func(mode="simd"):
        params = tla.params.BlockStoreParams(block_stride=32)
        for row in tla.range(32):
            for col in tla.range_constexpr(0, 512, 128):
                coord = tla.make_coord(row, col)
                shape = tla.make_shape(1, 128)
                data = tla.get_tile(raw_nd, coord, shape).load()
                tla.get_tile(packed_zn, coord, shape).store(data, params=params)


@tla.jit
def _swa_load_query_vector(sync, bufs, q, task, slot):
    with tla.vector():
        rows = Q_ROWS // 2
        sub = tla.arch.sub_block_idx()
        dt = q.ptr.dtype
        layout = tla.make_layout_with_tag((rows, HEAD_DIM), dt, tla.arch.RowMajor)
        raw = tla.recast_ptr(bufs.ub_o, dtype=dt)
        packed = raw + rows * HEAD_DIM
        gm = tla.make_tensor(q.ptr + (task * Q_ROWS + sub * rows) * HEAD_DIM, layout)
        ub = tla.make_tensor(raw, layout)
        raw_lock = sync.m_dq_rec0
        packed_lock = sync.m_dq_out0
        raw_lock.lock(pipe=tla.pipes.MTE2)
        tla.copy(ub, gm, cache_policy=tla.params.CachePolicy.DISABLE)
        raw_lock.unlock(pipe=tla.pipes.MTE2)
        raw_lock.lock(pipe=tla.pipes.VECTOR)
        packed_lock.lock(pipe=tla.pipes.VECTOR)
        _pack_swa_query(
            tla.recast_ptr(packed, dtype=tla.UInt16),
            tla.recast_ptr(raw, dtype=tla.UInt16),
        )
        packed_lock.unlock(pipe=tla.pipes.VECTOR)
        raw_lock.unlock(pipe=tla.pipes.VECTOR)
        packed_lock.lock(pipe=tla.pipes.MTE3)
        tla.cross_core_wait_flag(sync.q_vector_free, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_wait_flag(sync.q_vector_free, tla.pipes.MTE3, aiv_id=1)
        l1 = _q_slot_ptr(bufs, 2 * slot) + sub * rows * 16
        _single_query_insert_nz(
            tla.recast_ptr(l1, dtype=tla.UInt16),
            tla.recast_ptr(packed, dtype=tla.UInt16),
            rows,
            Q_ROWS,
            rows,
            HEAD_DIM,
        )
        packed_lock.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(sync.q_vector_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.q_vector_ready, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def load_rolling_q(sync, bufs, q, task, m, q_slot):
    """Two independent Q credits allow MTE2 to prefetch the next query."""
    with tla.cube():
        if q_slot == 0:
            tla.wait_flag(sync.q_l0a_free)
        else:
            tla.wait_flag(sync.q_second_free)
        src = tla.make_tensor(
            q.ptr + task * m * HEAD_DIM,
            tla.make_layout_with_tag((m, HEAD_DIM), q.ptr.dtype, tla.arch.RowMajor),
        )
        dst = tla.make_tensor(
            _q_slot_ptr(bufs, 2 * q_slot),
            tla.make_layout_with_tag((m, HEAD_DIM), q.ptr.dtype, tla.arch.zN),
        )
        tla.copy(dst, src)
        if q_slot == 0:
            tla.set_flag(sync.q_l1_ready)
        else:
            tla.set_flag(sync.q_second_ready)


@tla.jit
def load_rolling_window(
    sync, bufs, kv, bt, window_end, window_size, slot, reuse, page_size, page_stride
):
    """Each L1 slot retains its previous window. Update its incoming rows."""
    dt = kv.ptr.dtype
    with tla.cube():
        _k_slot_wait(sync, slot)
        dst = l1_k_tensor(_k_slot_ptr(bufs, slot), dt)
        remaining = window_size
        if reuse:
            remaining = 2 if remaining > 2 else remaining
        logical = window_end - remaining
        # A short initial window occupies a prefix. Fill the rest with finite
        # rows so the padded P@V accesses are valid even on a cold L1 slot.
        if not reuse:
            if window_size < KV_BLOCK:
                first_page = bt[0]
                first_page = first_page if first_page >= 0 else 0
                gm_pad = tla.make_tensor(
                    kv.ptr + first_page * page_stride,
                    tla.make_layout(
                        tla.make_shape(HEAD_DIM, KV_BLOCK),
                        tla.make_stride(1, HEAD_DIM),
                        layoutTag=tla.arch.ColumnMajor,
                    ),
                )
                tla.copy(dst, gm_pad)
        for p in (0, 1, 2):
            if remaining > 0:
                page = logical // page_size
                in_page = logical - page * page_size
                offset = logical % KV_BLOCK
                count = page_size - in_page
                if count > KV_BLOCK - offset:
                    count = KV_BLOCK - offset
                if count > remaining:
                    count = remaining
                physical = bt[page]
                physical = physical if physical >= 0 else 0
                src = tla.make_tensor(
                    kv.ptr + physical * page_stride + in_page * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(HEAD_DIM, count),
                        tla.make_stride(1, HEAD_DIM),
                        layoutTag=tla.arch.ColumnMajor,
                    ),
                )
                tla.copy(
                    tla.get_tile(
                        dst, tla.make_coord(0, offset), tla.make_shape(HEAD_DIM, count)
                    ),
                    src,
                )
                logical = logical + count
                remaining = remaining - count
        _k_slot_ready_set(sync, slot)


@tla.jit
def wait_kv_stage(sync):
    with tla.cube():
        tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE2, aiv_id=0)
        tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE2, aiv_id=1)


@tla.jit
def wait_p_ready(sync):
    with tla.cube():
        tla.cross_core_wait_flag(sync.p_ready, tla.pipes.MTE1, aiv_id=0)
        tla.cross_core_wait_flag(sync.p_ready, tla.pipes.MTE1, aiv_id=1)


@tla.jit
def _csa_prefetch_first_value(
    sync, bufs, slot, dtype, n_k, cached_ori, ori_free, columns: tla.Constexpr[int]
):
    head = _k_slot_ptr(bufs, slot)
    if cached_ori:
        head = bufs.l1_ori
    v_full = tla.make_tensor(
        head, tla.make_layout_with_tag((KV_BLOCK, columns), dtype, tla.arch.zN)
    )
    v_tile = tla.tile_view(v_full, tla.make_shape(n_k, columns), tla.make_coord(0, 0))
    v_l0 = tla.make_tensor(
        bufs.l0b0, tla.make_layout_with_tag((n_k, columns), dtype, tla.arch.nZ)
    )
    tla.wait_flag(sync.ab_free0)
    tla.copy(v_l0, v_tile)
    if cached_ori:
        tla.set_flag(ori_free)


@tla.jit
def iterate_bmm1(
    sync,
    bufs,
    slot,
    m,
    is_last_tile,
    first_tile,
    dtype,
    slot_s=0,
    q_slot_a=0,
    q_slot_b=1,
    n_blk=KV_BLOCK,
    two_q_slots=False,
    direct_kv=False,
    vector_query=False,
    phase0=None,
    phase1=None,
    phase0_free=None,
    phased=False,
    keep_q=False,
    q_prefix_free=None,
    cached_ori=False,
    reuse_k=False,
    unit_sync: tla.Constexpr[bool] = False,
    pv0=None,
    keep_q_suffix: tla.Constexpr[bool] = False,
):
    c0 = 0
    c1 = 1
    with tla.cube():
        if direct_kv:
            tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE1, aiv_id=0)
            tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE1, aiv_id=1)
        elif phase0 is not None and phased:
            if slot == 1:
                tla.wait_flag(phase1)
            else:
                tla.wait_flag(phase0)
                tla.set_flag(phase0_free)
        elif not reuse_k:
            _k_slot_ready_wait(sync, slot)
        if first_tile:
            if two_q_slots:
                if q_slot_a == 0:
                    tla.wait_flag(sync.q_l1_ready)
                elif vector_query:
                    tla.cross_core_wait_flag(
                        sync.q_vector_ready, tla.pipes.MTE1, aiv_id=0
                    )
                    tla.cross_core_wait_flag(
                        sync.q_vector_ready, tla.pipes.MTE1, aiv_id=1
                    )
                else:
                    tla.wait_flag(sync.q_second_ready)
            else:
                tla.wait_flag(sync.q_l1_ready)
        ub_s_dst = tla.make_tensor(
            bufs.ub_s + slot_s * (Q_ROWS // 2 * KV_BLOCK),
            tla.make_layout(
                tla.make_shape(m, n_blk),
                tla.make_stride(KV_BLOCK, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        l0c_s = tla.make_tensor(
            bufs.l0c_s + slot_s * (Q_ROWS * KV_BLOCK),
            tla.make_layout_with_tag((m, n_blk), tla.Float32, tla.arch.L0Clayout),
        )
        if slot_s == 0:
            tla.wait_flag(sync.s_fix_done)
        if slot_s == 1:
            tla.wait_flag(sync.s_fix_done1)
        for c in (0, 1, 2, 3):
            if phase0 is not None and phased and c == (2 if unit_sync else 3):
                _k_slot_ready_wait(sync, slot)
            half = c // 2
            l1q = l1_q_tensor(
                _q_slot_ptr(bufs, q_slot_a if half == 0 else q_slot_b), dtype, m
            )
            q_chunk = tla.tile_view(
                l1q, tla.make_shape(m, D_CHUNK), tla.make_coord(c0, c % 2)
            )
            k_base = _k_slot_ptr(bufs, slot)
            if c == 0 and cached_ori:
                k_base = bufs.l1_ori
            head = tla.make_tensor(
                k_base + c * (D_CHUNK * KV_BLOCK),
                tla.make_layout_with_tag((D_CHUNK, KV_BLOCK), dtype, tla.arch.nZ),
            )
            k_chunk = tla.tile_view(
                head, tla.make_shape(D_CHUNK, n_blk), tla.make_coord(0, 0)
            )
            slot_l0 = c % 2
            l0a_ptr = bufs.l0a0 if slot_l0 == 0 else bufs.l0a1
            l0b_ptr = bufs.l0b0 if slot_l0 == 0 else bufs.l0b1
            if keep_q and c < 2:
                # Reuse the first two Q segments for every tile in this query.
                l0a_ptr = l0a_ptr + Q_ROWS * D_CHUNK
            l0_q = tla.make_tensor_like(l0a_ptr, q_chunk)
            l0_k = tla.make_tensor_like(l0b_ptr, k_chunk)
            if slot_l0 == 0:
                tla.wait_flag(sync.ab_free0)
                if not keep_q or c >= 2 or first_tile:
                    tla.copy(l0_q, q_chunk)
                if keep_q_suffix and c == 2 and is_last_tile:
                    # The last segment is already resident, so c2 is the last L1 read.
                    tla.set_flag(sync.q_l0a_free)
                tla.copy(l0_k, k_chunk)
                tla.set_flag(sync.ab_ready0)
            else:
                tla.wait_flag(sync.ab_free1)
                if keep_q_suffix:
                    if c == 1 and first_tile:
                        tla.copy(l0_q, q_chunk)
                        q_suffix = l1_q_tensor(_q_slot_ptr(bufs, q_slot_b), dtype, m)
                        last_q = tla.tile_view(
                            q_suffix, tla.make_shape(m, D_CHUNK), tla.make_coord(0, 1)
                        )
                        tla.copy(tla.make_tensor_like(bufs.l0a1, last_q), last_q)
                elif not keep_q or c >= 2 or first_tile:
                    tla.copy(l0_q, q_chunk)
                if c == 1 and q_prefix_free is not None and first_tile:
                    # Both resident prefix chunks have left L1.
                    tla.set_flag(q_prefix_free)
                if c == 3 and keep_q and not keep_q_suffix and is_last_tile:
                    # The next suffix may fill L1 once its final read is issued.
                    tla.set_flag(sync.q_l0a_free)
                tla.copy(l0_k, k_chunk)
                tla.set_flag(sync.ab_ready1)
            if slot_l0 == 0:
                tla.wait_flag(sync.ab_ready0)
            else:
                tla.wait_flag(sync.ab_ready1)
            tla.mmad(
                l0c_s,
                l0_q,
                l0_k,
                init_c=(c == 0),
                unit_flag=(3 if c == 3 else 2) if unit_sync else 0,
            )
            if slot_l0 == 0:
                tla.set_flag(sync.ab_free0)
            else:
                tla.set_flag(sync.ab_free1)
        if pv0 is not None:
            pv_active, pv_slot, pv_count, pv_kind, pv_ori_free, pv_wide = pv0
            if pv_active:
                cached = pv_kind == -1
                if pv_wide and pv_kind == 1 and pv_count <= 64:
                    width = (pv_count + 15) // 16 * 16
                    _csa_prefetch_first_value(
                        sync,
                        bufs,
                        pv_slot,
                        dtype,
                        width,
                        False,
                        pv_ori_free,
                        HEAD_DIM // 2,
                    )
                elif pv_count > 112:
                    _csa_prefetch_first_value(
                        sync,
                        bufs,
                        pv_slot,
                        dtype,
                        KV_BLOCK,
                        cached,
                        pv_ori_free,
                        D_CHUNK,
                    )
                elif pv_count <= 16:
                    _csa_prefetch_first_value(
                        sync, bufs, pv_slot, dtype, 16, cached, pv_ori_free, D_CHUNK
                    )
                elif pv_count <= 32:
                    _csa_prefetch_first_value(
                        sync, bufs, pv_slot, dtype, 32, cached, pv_ori_free, D_CHUNK
                    )
                else:
                    width = (pv_count + 15) // 16 * 16
                    _csa_prefetch_first_value(
                        sync, bufs, pv_slot, dtype, width, cached, pv_ori_free, D_CHUNK
                    )
        if not unit_sync:
            if slot_s == 0:
                tla.set_flag(sync.s_mm_ready)
                tla.wait_flag(sync.s_mm_ready)
            if slot_s == 1:
                tla.set_flag(sync.s_mm_ready1)
                tla.wait_flag(sync.s_mm_ready1)
        tla.cross_core_wait_flag(sync.s_free, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_wait_flag(sync.s_free, tla.pipes.FIX, aiv_id=1)
        tla.copy(
            ub_s_dst,
            l0c_s,
            tla.params.CopyL0C2DstParams(
                unit_flag=3 if unit_sync else 0,
                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
            ),
        )
        if slot_s == 0:
            tla.set_flag(sync.s_fix_done)
        if slot_s == 1:
            tla.set_flag(sync.s_fix_done1)
        tla.cross_core_set_flag(sync.s_ready, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_set_flag(sync.s_ready, tla.pipes.FIX, aiv_id=1)
        if is_last_tile and not keep_q:
            if two_q_slots:
                if q_slot_a == 0:
                    tla.set_flag(sync.q_l0a_free)
                elif vector_query:
                    tla.cross_core_set_flag(
                        sync.q_vector_free, tla.pipes.MTE1, aiv_id=0
                    )
                    tla.cross_core_set_flag(
                        sync.q_vector_free, tla.pipes.MTE1, aiv_id=1
                    )
                else:
                    tla.set_flag(sync.q_second_free)
            else:
                tla.set_flag(sync.q_l0a_free)


@tla.jit
def iterate_bmm2_seg(
    sync,
    bufs,
    seg,
    slot_k,
    m,
    dtype,
    slot_p=0,
    n_k=KV_BLOCK,
    compact=False,
    preload_v=False,
    cached_ori=False,
    ori_free=None,
    split_fix: tla.Constexpr[bool] = False,
    single_p_free=None,
    columns: tla.Constexpr[int] = D_CHUNK,
    first_v_loaded=False,
    wait_single_p=True,
    signal_single_p=True,
):
    c0 = 0
    with tla.cube():
        l1p = l1_p_tensor(bufs.l1_p + slot_p * (Q_ROWS * KV_BLOCK), dtype, m)
        head = _k_slot_ptr(bufs, slot_k)
        if seg == 0 and cached_ori:
            head = bufs.l1_ori
        v_seg_full = tla.make_tensor(
            head + seg * (KV_BLOCK * columns),
            tla.make_layout_with_tag((KV_BLOCK, columns), dtype, tla.arch.zN),
        )
        p_tile = tla.tile_view(l1p, tla.make_shape(m, n_k), tla.make_coord(c0, c0))
        v_seg = tla.tile_view(
            v_seg_full, tla.make_shape(n_k, columns), tla.make_coord(c0, c0)
        )
        slot = seg % 2
        l0a_ptr = bufs.l0a0 if slot == 0 else bufs.l0a1
        l0b_ptr = bufs.l0b0 if slot == 0 else bufs.l0b1
        l0_p = tla.make_tensor_like(
            bufs.l0a0 if single_p_free is not None else l0a_ptr, p_tile
        )
        l0_v = tla.make_tensor_like(l0b_ptr, v_seg)
        if compact or n_k != KV_BLOCK:
            l0_v = tla.make_tensor(
                l0b_ptr,
                tla.make_layout_with_tag((n_k, columns), dtype, tla.arch.nZ),
            )
        # shape<(16,4),(16,8)> stride<(16,256),(1,1024)>，n-fract stride =
        m_f = ((m + 15) // 16) * 16
        l0c_seg = tla.make_tensor(
            bufs.l0c_o + seg * (m_f * columns),
            tla.make_layout_with_tag((m, columns), tla.Float32, tla.arch.L0Clayout),
        )
        if seg == 0:
            tla.wait_flag(sync.o_fix_done0)
        elif seg == HEAD_DIM // (2 * columns) and split_fix:
            tla.wait_flag(sync.o_fix_done1)
        if seg < 2:
            if slot == 0:
                if not first_v_loaded:
                    tla.wait_flag(sync.ab_free0)
                if preload_v:
                    if not first_v_loaded:
                        tla.copy(l0_v, v_seg)
                        if cached_ori:
                            tla.set_flag(ori_free)
                    tla.cross_core_wait_flag(sync.p_ready, tla.pipes.MTE1, aiv_id=0)
                    tla.cross_core_wait_flag(sync.p_ready, tla.pipes.MTE1, aiv_id=1)
                    if single_p_free is not None and wait_single_p:
                        tla.wait_flag(single_p_free)
                    tla.copy(l0_p, p_tile)
                else:
                    tla.copy(l0_p, p_tile)
                    tla.copy(l0_v, v_seg)
                tla.set_flag(sync.ab_ready0)
            else:
                tla.wait_flag(sync.ab_free1)
                if single_p_free is None:
                    tla.copy(l0_p, p_tile)
                tla.copy(l0_v, v_seg)
                tla.set_flag(sync.ab_ready1)
        else:
            if slot == 0:
                tla.wait_flag(sync.ab_free0)
                tla.copy(l0_v, v_seg)
                tla.set_flag(sync.ab_ready0)
            else:
                tla.wait_flag(sync.ab_free1)
                tla.copy(l0_v, v_seg)
                tla.set_flag(sync.ab_ready1)
        if slot == 0:
            tla.wait_flag(sync.ab_ready0)
        else:
            tla.wait_flag(sync.ab_ready1)
        tla.mmad(l0c_seg, l0_p, l0_v, init_c=True, unit_flag=3 if split_fix else 0)
        if slot == 0:
            tla.set_flag(sync.ab_free0)
        else:
            tla.set_flag(sync.ab_free1)
        if (
            seg == HEAD_DIM // columns - 1
            and single_p_free is not None
            and signal_single_p
        ):
            tla.set_flag(single_p_free)


@tla.jit
def iterate_bmm2_fixpipe(sync, bufs, m, start_output=False):
    c0 = 0
    with tla.cube():
        ub_full = tla.make_tensor(
            bufs.ub_o if start_output else bufs.ub_otmp_full,
            tla.make_layout(
                tla.make_shape(m, HEAD_DIM),
                tla.make_stride(HEAD_DIM, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        l0c_full = tla.make_tensor(
            bufs.l0c_o,
            tla.make_layout_with_tag((m, HEAD_DIM), tla.Float32, tla.arch.L0Clayout),
        )
        tla.set_flag(sync.o_mm_ready0)
        tla.wait_flag(sync.o_mm_ready0)
        tla.cross_core_wait_flag(sync.o_free, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_wait_flag(sync.o_free, tla.pipes.FIX, aiv_id=1)
        tla.copy(
            ub_full,
            l0c_full,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        tla.set_flag(sync.o_fix_done0)
        tla.cross_core_set_flag(sync.o_ready, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_set_flag(sync.o_ready, tla.pipes.FIX, aiv_id=1)


@tla.jit
def _csa_fix_pv_half(sync, bufs, m, rows, part, fresh, free_flag, ready_flag):
    with tla.cube():
        width = HEAD_DIM // 2
        gm_layout = tla.make_layout(
            tla.make_shape(m, width),
            tla.make_stride(HEAD_DIM, 1),
            layoutTag=tla.arch.RowMajor,
        )
        tmp_layout = tla.make_layout(
            tla.make_shape(m, width),
            tla.make_stride(width, 1),
            layoutTag=tla.arch.RowMajor,
        )
        acc = tla.make_tensor(
            bufs.l0c_o + part * m * width,
            tla.make_layout_with_tag((m, width), tla.Float32, tla.arch.L0Clayout),
        )
        done = sync.o_fix_done0 if part == 0 else sync.o_fix_done1
        tla.cross_core_wait_flag(free_flag, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_wait_flag(free_flag, tla.pipes.FIX, aiv_id=1)
        if fresh:
            result = tla.make_tensor(bufs.ub_o + part * width, gm_layout)
            tla.copy(
                result,
                acc,
                tla.params.CopyL0C2DstParams(
                    unit_flag=3, l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                ),
            )
        else:
            result = tla.make_tensor(
                bufs.ub_otmp_full + part * rows * width, tmp_layout
            )
            tla.copy(
                result,
                acc,
                tla.params.CopyL0C2DstParams(
                    unit_flag=3, l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                ),
            )
        tla.set_flag(done)
        tla.cross_core_set_flag(ready_flag, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_set_flag(ready_flag, tla.pipes.FIX, aiv_id=1)


@tla.jit
def release_k(sync, slot):
    with tla.cube():
        _k_slot_free(sync, slot)


@tla.jit
def split_g_mte2_pair(sync):
    with tla.cube():
        tla.cross_core_set_flag(sync.s2_mte2, tla.pipes.MTE2)
        tla.cross_core_wait_flag(sync.s2_mte2, tla.pipes.MTE2)


@tla.jit
def pair_rendezvous(sync):
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
        tla.cross_core_set_flag(sync.s2_rendezvous, tla.pipes.FIX)
        tla.cross_core_wait_flag(sync.s2_rendezvous, tla.pipes.SCALAR)


VL = 64
VL_B16 = 128


@tla.jit
def init_output_gm(
    sync, bufs, gm, base_elem, total_elem, pop_ele, fill_dtype, use_lse_buf=0
):
    src = bufs.ub_init_lse if use_lse_buf == 1 else bufs.ub_init
    ub = tla.make_tensor(
        src, tla.make_layout(tla.make_shape(pop_ele), tla.make_stride(1))
    )
    # OT：SetFlag<MTE3_V>(SYNC_ID); WaitFlag<MTE3_V>(SYNC_ID)
    tla.set_flag(sync.init_mte3_v)
    tla.wait_flag(sync.init_mte3_v)
    with tla.vec.func(mode="simd"):
        zero = tla.full(0.0, dtype=fill_dtype)
        zmask, _ = tla.update_mask(VL, dtype=fill_dtype)
        # AscendC::Duplicate(popBuffer, initValue, POP_BUF_ELE_SIZE)
        for c in tla.range(pop_ele // VL):
            tla.tile_view(ub, tla.make_shape(VL), tla.make_coord(c)).store(
                zero, params=tla.params.UnalignStoreParams()
            )
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    # OT：SetFlag<V_MTE3>(SYNC_ID); WaitFlag<V_MTE3>(SYNC_ID)
    tla.set_flag(sync.init_v_mte3)
    tla.wait_flag(sync.init_v_mte3)
    loop_cnt = total_elem // pop_ele
    tail = total_elem - loop_cnt * pop_ele
    for i in tla.range(loop_cnt):
        gm_blk = tla.make_tensor(
            gm.ptr + base_elem + i * pop_ele,
            tla.make_layout(tla.make_shape(pop_ele), tla.make_stride(1)),
        )
        tla.copy(gm_blk, ub)
    if tail > 0:
        gm_tail = tla.make_tensor(
            gm.ptr + base_elem + loop_cnt * pop_ele,
            tla.make_layout(tla.make_shape(tail), tla.make_stride(1)),
        )
        tla.copy(gm_tail, tla.tile_view(ub, tla.make_shape(tail), tla.make_coord(0)))
    tla.set_flag(sync.init_mte3_v)
    tla.wait_flag(sync.init_mte3_v)


def _lse_total_len(total_output, total_lse, head_dim):
    d = int(getattr(head_dim, "value", head_dim))
    if d == 0:
        return total_lse
    return total_output // d


@tla.jit
def init_output_single_core(
    sync,
    bufs,
    attention_out,
    lse,
    total_output,
    total_lse,
    enable_lse,
    dtype_in,
    head_dim: tla.Constexpr[int] = 0,
):
    vec_core_num = tla.arch.block_num() * 2  # CV_RATIO * GetBlockNum()
    vec_idx = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
    # ---- attentionOut ----
    single = (total_output + vec_core_num - 1) // vec_core_num
    off = vec_idx * single
    if off < total_output:
        actual = total_output - off
        if actual > single:
            actual = single
        init_output_gm(
            sync, bufs, attention_out, off, actual, INIT_POP_ELE, dtype_in, 0
        )
    if enable_lse == 1:
        lse_len = _lse_total_len(total_output, total_lse, head_dim)
        single_l = (lse_len + vec_core_num - 1) // vec_core_num
        off_l = vec_idx * single_l
        if off_l < lse_len:
            actual_l = lse_len - off_l
            if actual_l > single_l:
                actual_l = single_l
            init_output_gm(
                sync, bufs, lse, off_l, actual_l, INIT_POP_ELE, tla.Float32, 1
            )


@tla.jit
def clean_output(
    sync,
    bufs,
    attention_out,
    lse,
    need_init,
    total_output,
    total_lse,
    enable_lse,
    dtype_in,
    head_dim: tla.Constexpr[int] = 0,
):
    with tla.vector():
        if need_init == 1:
            init_output_single_core(
                sync,
                bufs,
                attention_out,
                lse,
                total_output,
                total_lse,
                enable_lse,
                dtype_in,
                head_dim=head_dim,
            )


@tla.jit
def _ub_s(bufs, rows, slot=0):
    return tla.make_tensor(
        bufs.ub_s + slot * (Q_ROWS // 2 * KV_BLOCK),
        tla.make_layout(tla.make_shape(rows, KV_BLOCK), tla.make_stride(KV_BLOCK, 1)),
    )


@tla.jit
def _ub_p(bufs, rows):
    ub_rm = tla.make_tensor(
        bufs.ub_p,
        tla.make_layout(tla.make_shape(rows, KV_BLOCK), tla.make_stride(KV_BLOCK, 1)),
    )
    return tla.make_tensor_like(bufs.ub_p, ub_rm, tla.arch.zN)


@tla.jit
def _dm(bufs, rows, slot=0):
    return tla.make_tensor(
        bufs.ub_dm + slot * _bc_rows8(rows),
        tla.make_layout(tla.make_shape(rows), tla.make_stride(1)),
    )


@tla.jit
def _sum(bufs, rows, slot=0):
    return tla.make_tensor(
        bufs.ub_sum + slot * _bc_rows8(rows),
        tla.make_layout(tla.make_shape(rows), tla.make_stride(1)),
    )


@tla.jit
def _max(bufs, rows, slot=0):
    return tla.make_tensor(
        bufs.ub_max + slot * _bc_rows8(rows),
        tla.make_layout(tla.make_shape(rows), tla.make_stride(1)),
    )


@tla.jit
def _stats(ptr, rows):
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
    )


@tla.jit
def vec_init_from_sinks(
    sync,
    bufs,
    sinks,
    task,
    m,
    rows,
    dtype_in,
    sink_off=0,
    has_sinks=1,
    sink_cap=0,
    slot_sum=0,
    slot_max=0,
    clear_o: tla.Constexpr[int] = 1,
):
    c0 = 0
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        sink_len = sink_cap if sink_cap > 0 else rows
        ub_sink = _stats(bufs.ub_sink, sink_len)
        gm_sink_all = tla.make_tensor(
            sinks.ptr, tla.make_layout(tla.make_shape(sink_len), tla.make_stride(1))
        )
        if has_sinks == 1:
            sync.m_sink.lock(pipe=tla.pipes.MTE2)
            if sink_cap == 0:
                s_off = sink_off + sub * rows
                if sub * rows >= m:
                    s_off = sink_off
                gm_sink = tla.make_tensor(
                    sinks.ptr + s_off,
                    tla.make_layout(tla.make_shape(rows), tla.make_stride(1)),
                )
                tla.copy(ub_sink, gm_sink)
            else:
                tla.copy(ub_sink, gm_sink_all)
            sync.m_sink.unlock(pipe=tla.pipes.MTE2)
        sync.m_sink.lock(pipe=tla.pipes.VECTOR)
        ub_max = _max(bufs, rows, slot_max)
        ub_sum = _sum(bufs, rows, slot_sum)
        ub_o = tla.make_tensor(
            bufs.ub_o,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            zero = tla.full(0.0, dtype=tla.Float32)
            one = tla.full(1.0, dtype=tla.Float32)
            minreg = tla.full(MIN_F, dtype=tla.Float32)
            for r in tla.range(rows):
                ub_max_r = tla.tile_view(ub_max, tla.make_shape(1), tla.make_coord(r))
                ub_sum_r = tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(r))
                if has_sinks == 1:
                    # （OT sinksOffset = goIdx + subBlockIdx*firstHalfMRealSize）。
                    sink_row = (sink_off + sub * rows + r) if sink_cap > 0 else r
                    ub_sink_r = tla.tile_view(
                        ub_sink, tla.make_shape(1), tla.make_coord(sink_row)
                    )
                    ub_max_r.store(
                        ub_sink_r.load(params=brc),
                        params=tla.params.UnalignStoreParams(),
                    )
                    ub_sum_r.store(one, params=tla.params.UnalignStoreParams())
                else:
                    ub_max_r.store(minreg, params=tla.params.UnalignStoreParams())
                    ub_sum_r.store(zero, params=tla.params.UnalignStoreParams())
                if clear_o == 1:
                    for c in tla.range(HEAD_DIM // VL):
                        ub_o_c = tla.tile_view(
                            ub_o, tla.make_shape(1, VL), tla.make_coord(r, c)
                        )
                        ub_o_c.store(zero)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        sync.m_sink.unlock(pipe=tla.pipes.VECTOR)


@tla.jit
def process_vec1(
    sync,
    bufs,
    scale,
    tail_n,
    m,
    rows,
    dtype_in,
    slot_s=0,
    slot_p=0,
    preload: tla.Constexpr[int] = 0,
    slot_sum_rd=0,
    slot_max_rd=0,
):
    c0 = 0
    c1 = 1
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        ub_s = _ub_s(bufs, rows, slot_s)
        ub_p = _ub_p(bufs, rows)
        # max(x-1)）。
        ub_max = _max(bufs, rows, slot_s)
        ub_max_rd = _max(bufs, rows, slot_max_rd)
        ub_sum = _sum(bufs, rows, slot_s)
        ub_sum_rd = _sum(bufs, rows, slot_sum_rd)
        ub_dm = _dm(bufs, rows, slot_s)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=1)
        sync.m_p.lock(pipe=tla.pipes.VECTOR)

        n0 = tail_n
        if n0 > VL:
            n0 = tla.as_numeric(VL)
        n1 = tail_n - VL
        if n1 > VL:
            n1 = tla.as_numeric(VL)
        if n1 < 0:
            n1 = tla.as_numeric(0)
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            preg_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype_in)
            preg0, _ = tla.update_mask(n0, dtype=tla.Float32)
            preg1, _ = tla.update_mask(n1, dtype=tla.Float32)
            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
            minreg = tla.full(MIN_F, dtype=tla.Float32)
            cast_zero = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            cast_one = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ONE,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            # vf_basic_block_{aligned128,unaligned64,unaligned128}_*_sfa.h。
            if tail_n == KV_BLOCK:
                # ---- pass 1: scale + mask + running max ----
                for r in tla.range(rows):
                    s0 = tla.tile_view(
                        ub_s, tla.make_shape(1, VL), tla.make_coord(r, 0)
                    )
                    s1 = tla.tile_view(
                        ub_s, tla.make_shape(1, VL), tla.make_coord(r, 1)
                    )
                    v0 = tla.mul(s0.load(), scale, mask=preg)
                    v1 = tla.mul(s1.load(), scale, mask=preg)
                    v0 = tla.where(preg0, v0, minreg)
                    v1 = tla.where(preg1, v1, minreg)
                    s0.store(v0)
                    s1.store(v1)
                    tmax = tla.max(v0, v1, mask=preg).reduce(
                        tla.ReductionOp.MAX, mask=preg
                    )
                    ub_max_r = tla.tile_view(
                        ub_max, tla.make_shape(1), tla.make_coord(r)
                    )
                    ub_dm_r = tla.tile_view(ub_dm, tla.make_shape(1), tla.make_coord(r))
                    old_max = tla.tile_view(
                        ub_max_rd, tla.make_shape(1), tla.make_coord(r)
                    ).load(params=brc)
                    new_max = tla.max(old_max, tmax, mask=preg)
                    ub_max_r.store(new_max, params=tla.params.UnalignStoreParams())
                    ub_dm_r.store(
                        tla.exp(tla.sub(old_max, new_max, mask=preg), mask=preg),
                        params=tla.params.UnalignStoreParams(),
                    )
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
                # ---- pass 2: exp(x - max), sum, cast -> P ----
                preg_i = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
                v = tla.arange(base=0, order="increase", dtype=tla.Int16)
                hi_sel = tla.shift_right(v, 6, mask=preg_i)
                perm_u = tla.sub(
                    tla.mul(v, 2, mask=preg_i),
                    tla.mul(hi_sel, 127, mask=preg_i),
                    mask=preg_i,
                ).bitcast(tla.UInt16)
                block_store = tla.params.BlockStoreParams(
                    block_stride=ub_p.stride[1][1] // ub_p.shape[1][0]
                )
                for r2 in tla.range(rows):
                    ub_max_b = tla.tile_view(
                        ub_max, tla.make_shape(1), tla.make_coord(r2)
                    )
                    ub_sum_b = tla.tile_view(
                        ub_sum, tla.make_shape(1), tla.make_coord(r2)
                    )
                    ub_dm_b = tla.tile_view(
                        ub_dm, tla.make_shape(1), tla.make_coord(r2)
                    )
                    mx = ub_max_b.load(params=brc)
                    dm = ub_dm_b.load(params=brc)
                    ps0 = tla.tile_view(
                        ub_s, tla.make_shape(1, VL), tla.make_coord(r2, 0)
                    )
                    e0 = tla.exp(tla.sub(ps0.load(), mx, mask=preg), mask=preg)
                    rec0 = e0.reduce(tla.ReductionOp.ADD, mask=preg)
                    ps1 = tla.tile_view(
                        ub_s, tla.make_shape(1, VL), tla.make_coord(r2, 1)
                    )
                    e1 = tla.exp(tla.sub(ps1.load(), mx, mask=preg), mask=preg)
                    rec1 = e1.reduce(tla.ReductionOp.ADD, mask=preg)
                    tsum = tla.add(rec0, rec1, mask=preg)
                    d0 = e0.to(dtype_in, cast_zero, mask=preg)
                    d1 = e1.to(dtype_in, cast_one, mask=preg)
                    merged = tla.bitwise_or(d0, d1, mask=preg_b16)
                    tight = tla.gather(merged, perm_u)
                    tla.tile_view(
                        ub_p, tla.make_shape(1, KV_BLOCK), tla.make_coord(r2, 0)
                    ).store(tight, params=block_store)
                    old_sum = tla.tile_view(
                        ub_sum_rd, tla.make_shape(1), tla.make_coord(r2)
                    ).load(params=brc)
                    new_sum = tla.add(tla.mul(old_sum, dm, mask=preg), tsum, mask=preg)
                    ub_sum_b.store(new_sum, params=tla.params.UnalignStoreParams())
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
            else:
                if tail_n <= VL:
                    for r in tla.range(rows):
                        s0 = tla.tile_view(
                            ub_s, tla.make_shape(1, VL), tla.make_coord(r, 0)
                        )
                        v0 = tla.mul(s0.load(), scale, mask=preg)
                        v0 = tla.where(preg0, v0, minreg)
                        s0.store(v0)
                        tmax = v0.reduce(tla.ReductionOp.MAX, mask=preg)
                        ub_max_r = tla.tile_view(
                            ub_max, tla.make_shape(1), tla.make_coord(r)
                        )
                        ub_dm_r = tla.tile_view(
                            ub_dm, tla.make_shape(1), tla.make_coord(r)
                        )
                        old_max = tla.tile_view(
                            ub_max_rd, tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        new_max = tla.max(old_max, tmax, mask=preg)
                        ub_max_r.store(new_max, params=tla.params.UnalignStoreParams())
                        ub_dm_r.store(
                            tla.exp(tla.sub(old_max, new_max, mask=preg), mask=preg),
                            params=tla.params.UnalignStoreParams(),
                        )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                    preg_i = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
                    v = tla.arange(base=0, order="increase", dtype=tla.Int16)
                    hi_sel = tla.shift_right(v, 6, mask=preg_i)
                    perm_u = tla.sub(
                        tla.mul(v, 2, mask=preg_i),
                        tla.mul(hi_sel, 127, mask=preg_i),
                        mask=preg_i,
                    ).bitcast(tla.UInt16)
                    block_store = tla.params.BlockStoreParams(
                        block_stride=ub_p.stride[1][1] // ub_p.shape[1][0]
                    )
                    for r2 in tla.range(rows):
                        ub_max_b = tla.tile_view(
                            ub_max, tla.make_shape(1), tla.make_coord(r2)
                        )
                        ub_sum_b = tla.tile_view(
                            ub_sum, tla.make_shape(1), tla.make_coord(r2)
                        )
                        ub_dm_b = tla.tile_view(
                            ub_dm, tla.make_shape(1), tla.make_coord(r2)
                        )
                        mx = ub_max_b.load(params=brc)
                        dm = ub_dm_b.load(params=brc)
                        ps0 = tla.tile_view(
                            ub_s, tla.make_shape(1, VL), tla.make_coord(r2, 0)
                        )
                        e0 = tla.exp(tla.sub(ps0.load(), mx, mask=preg), mask=preg)
                        rec0 = e0.reduce(tla.ReductionOp.ADD, mask=preg)
                        d0 = e0.to(dtype_in, cast_zero, mask=preg)
                        tight = tla.gather(d0, perm_u)
                        tla.tile_view(
                            ub_p, tla.make_shape(1, KV_BLOCK), tla.make_coord(r2, 0)
                        ).store(tight, params=block_store)
                        old_sum = tla.tile_view(
                            ub_sum_rd, tla.make_shape(1), tla.make_coord(r2)
                        ).load(params=brc)
                        new_sum = tla.add(
                            tla.mul(old_sum, dm, mask=preg), rec0, mask=preg
                        )
                        ub_sum_b.store(new_sum, params=tla.params.UnalignStoreParams())
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                else:
                    # ---- pass 1: scale + mask + running max ----
                    for r in tla.range(rows):
                        s0 = tla.tile_view(
                            ub_s, tla.make_shape(1, VL), tla.make_coord(r, 0)
                        )
                        s1 = tla.tile_view(
                            ub_s, tla.make_shape(1, VL), tla.make_coord(r, 1)
                        )
                        v0 = tla.mul(s0.load(), scale, mask=preg)
                        v1 = tla.mul(s1.load(), scale, mask=preg)
                        v1 = tla.where(preg1, v1, minreg)
                        s0.store(v0)
                        s1.store(v1)
                        tmax = tla.max(v0, v1, mask=preg).reduce(
                            tla.ReductionOp.MAX, mask=preg
                        )
                        ub_max_r = tla.tile_view(
                            ub_max, tla.make_shape(1), tla.make_coord(r)
                        )
                        ub_dm_r = tla.tile_view(
                            ub_dm, tla.make_shape(1), tla.make_coord(r)
                        )
                        old_max = tla.tile_view(
                            ub_max_rd, tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        new_max = tla.max(old_max, tmax, mask=preg)
                        ub_max_r.store(new_max, params=tla.params.UnalignStoreParams())
                        ub_dm_r.store(
                            tla.exp(tla.sub(old_max, new_max, mask=preg), mask=preg),
                            params=tla.params.UnalignStoreParams(),
                        )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                    # ---- pass 2: exp(x - max), sum, cast -> P ----
                    preg_i = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
                    v = tla.arange(base=0, order="increase", dtype=tla.Int16)
                    hi_sel = tla.shift_right(v, 6, mask=preg_i)
                    perm_u = tla.sub(
                        tla.mul(v, 2, mask=preg_i),
                        tla.mul(hi_sel, 127, mask=preg_i),
                        mask=preg_i,
                    ).bitcast(tla.UInt16)
                    block_store = tla.params.BlockStoreParams(
                        block_stride=ub_p.stride[1][1] // ub_p.shape[1][0]
                    )
                    for r2 in tla.range(rows):
                        ub_max_b = tla.tile_view(
                            ub_max, tla.make_shape(1), tla.make_coord(r2)
                        )
                        ub_sum_b = tla.tile_view(
                            ub_sum, tla.make_shape(1), tla.make_coord(r2)
                        )
                        ub_dm_b = tla.tile_view(
                            ub_dm, tla.make_shape(1), tla.make_coord(r2)
                        )
                        mx = ub_max_b.load(params=brc)
                        dm = ub_dm_b.load(params=brc)
                        ps0 = tla.tile_view(
                            ub_s, tla.make_shape(1, VL), tla.make_coord(r2, 0)
                        )
                        e0 = tla.exp(tla.sub(ps0.load(), mx, mask=preg), mask=preg)
                        rec0 = e0.reduce(tla.ReductionOp.ADD, mask=preg)
                        ps1 = tla.tile_view(
                            ub_s, tla.make_shape(1, VL), tla.make_coord(r2, 1)
                        )
                        e1 = tla.exp(tla.sub(ps1.load(), mx, mask=preg), mask=preg)
                        rec1 = e1.reduce(tla.ReductionOp.ADD, mask=preg)
                        tsum = tla.add(rec0, rec1, mask=preg)
                        d0 = e0.to(dtype_in, cast_zero, mask=preg)
                        d1 = e1.to(dtype_in, cast_one, mask=preg)
                        merged = tla.bitwise_or(d0, d1, mask=preg_b16)
                        tight = tla.gather(merged, perm_u)
                        tla.tile_view(
                            ub_p, tla.make_shape(1, KV_BLOCK), tla.make_coord(r2, 0)
                        ).store(tight, params=block_store)
                        old_sum = tla.tile_view(
                            ub_sum_rd, tla.make_shape(1), tla.make_coord(r2)
                        ).load(params=brc)
                        new_sum = tla.add(
                            tla.mul(old_sum, dm, mask=preg), tsum, mask=preg
                        )
                        ub_sum_b.store(new_sum, params=tla.params.UnalignStoreParams())
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=1)
        l1p = tla.make_tensor(
            bufs.l1_p + slot_p * (Q_ROWS * KV_BLOCK),
            tla.make_layout_with_tag((m, KV_BLOCK), dtype_in, tla.arch.zN),
        )
        l1p_rows = tla.tile_view(
            l1p, tla.make_shape(rows, KV_BLOCK), tla.make_coord(sub, 0)
        )
        sync.m_p.unlock(pipe=tla.pipes.VECTOR)
        sync.m_p.lock(pipe=tla.pipes.MTE3)
        tla.set_flag(sync.p_data_ready)
        tla.wait_flag(sync.p_data_ready)
        tla.copy(l1p_rows, ub_p)
        sync.m_p.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=1)


_FOLD_CMP_SOURCE = r"""
#include "kernel_operator.h"

using namespace AscendC::MicroAPI;
using AscendC::CMPMODE;
using AscendC::SpecialPurposeReg;
static constexpr CastTrait fold_float_cast = {RegLayout::UNKNOWN, SatMode::UNKNOWN,
    MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

__simd_vf__ void FindRuns(__ubuf__ int32_t* out, __ubuf__ int32_t* source,
                          uint32_t count) {
    auto full = CreateMask<int32_t, MaskPattern::ALL>();
    auto* previous = source - 1;
    UnalignRegForLoad prev_reg;
    UnalignRegForStore out_reg;
    LoadUnAlignPre(prev_reg, previous);
    uint32_t remaining = count;
    uint32_t first = 1;
    auto head = UpdateMask<int32_t>(first);
    ClearSpr<SpecialPurposeReg::AR>();
    for (uint16_t col = 0; col < count; col += 64) {
        RegTensor<int32_t> ids, prev, positions, packed;
        LoadAlign(ids, source + col);
        LoadUnAlign(prev, prev_reg, previous, 64);
        auto active = UpdateMask<int32_t>(remaining);
        MaskReg changed;
        Compare<int32_t, CMPMODE::NE>(changed, ids, prev, active);
        if (col == 0) {
            MaskOr(changed, changed, head, full);
        }
        Arange(positions, (int32_t)col);
        GatherMask<int32_t, GatherMaskMode::STORE_REG>(packed, positions, changed);
        StoreUnAlign(out, packed, out_reg);
    }
    StoreUnAlignPost(out, out_reg);
}

__simd_vf__ void EmitRuns(__ubuf__ int32_t* ids_out, __ubuf__ float* log_out,
                          __ubuf__ int32_t* positions, __ubuf__ int32_t* source,
                          uint32_t groups) {
    auto* next = positions + 1;
    UnalignRegForLoad next_reg;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LocalMemBar<MemType::SCALAR_STORE, MemType::VEC_LOAD>();
    LoadUnAlignPre(next_reg, next);
    uint32_t remaining = groups;
    for (uint16_t col = 0; col < groups; col += 64) {
        RegTensor<int32_t> starts, ends, n, ids;
        RegTensor<float> weights, logs;
        auto active = UpdateMask<int32_t>(remaining);
        LoadAlign(starts, positions + col);
        LoadUnAlign(ends, next_reg, next, 64);
        Gather(ids, source, (RegTensor<uint32_t>&)starts, active);
        Sub(n, ends, starts, active);
        Cast<float, int32_t, fold_float_cast>(weights, n, active);
        Ln(logs, weights, active);
        StoreAlign(ids_out + col, ids, active);
        StoreAlign(log_out + col, logs, active);
    }
}

extern "C" {
[aicore] __attribute__((always_inline)) void fold_one_cmp_row(
    uint64_t ids_in, uint64_t pos, uint64_t ids_out, uint64_t logs,
    uint64_t length, int32_t length_in) {
    auto* source = reinterpret_cast<__ubuf__ int32_t *>(ids_in);
    auto* positions = reinterpret_cast<__ubuf__ int32_t *>(pos);
    auto* count_out = reinterpret_cast<__ubuf__ int32_t *>(length);
    if (length_in == 0) {
        count_out[0] = 0;
        return;
    }
    FindRuns(positions, source, length_in);
    const uint32_t groups = AscendC::GetSpr<SpecialPurposeReg::AR>() / sizeof(int32_t);
    const int32_t count = static_cast<int32_t>(groups);
    count_out[0] = count == length_in ? -count : count;
    positions[groups] = length_in;
    EmitRuns(reinterpret_cast<__ubuf__ int32_t *>(ids_out),
             reinterpret_cast<__ubuf__ float *>(logs), positions, source, groups);
}
}
"""


@tla.extern(source=_FOLD_CMP_SOURCE, name="fold_one_cmp_row")
def _fold_one_cmp_row(
    source: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    positions: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    ids: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    log_counts: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    count: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    length: tla.Int32,
) -> None: ...


@tla.jit
def _fold_indices_batch(
    source,
    topk_in,
    ids,
    log_counts,
    topk_out,
    input_ids,
    input_lens,
    positions,
    output_ids,
    output_log,
    output_lens,
    width,
    start,
    count,
    limit,
    released,
    input_scalar,
    input_vector,
    output_scalar,
    output_vector,
):
    """Copy one group of index rows in and write its runs back in bulk."""
    row_shape = tla.make_layout(tla.make_shape(count), tla.make_stride(1))
    tile_shape = tla.make_layout(tla.make_shape(count * width), tla.make_stride(1))
    # Reuse this UB buffer after its three previous GM writes have completed.
    tla.wait_flag(released)
    tla.copy(
        tla.make_tensor(input_ids, tile_shape),
        tla.make_tensor(source.ptr + start * width, tile_shape),
    )
    tla.copy(
        tla.make_tensor(input_lens, row_shape),
        tla.make_tensor(topk_in.ptr + start, row_shape),
    )
    tla.set_flag(input_scalar)
    tla.set_flag(input_vector)
    tla.wait_flag(input_scalar)
    tla.wait_flag(input_vector)
    counts = tla.make_tensor(input_lens, row_shape)
    for r in tla.range(count):
        length = counts[r]
        length = limit if length > limit else length
        length = length if length > 0 else 0
        _fold_one_cmp_row(
            input_ids + r * width,
            positions + r * (width + 64),
            output_ids + r * width,
            output_log + r * width,
            output_lens + r,
            length,
        )
    tla.set_flag(output_scalar)
    tla.set_flag(output_vector)
    tla.wait_flag(output_scalar)
    tla.wait_flag(output_vector)
    tla.copy(
        tla.make_tensor(ids.ptr + start * width, tile_shape),
        tla.make_tensor(output_ids, tile_shape),
    )
    tla.copy(
        tla.make_tensor(log_counts.ptr + start * width, tile_shape),
        tla.make_tensor(output_log, tile_shape),
    )
    tla.copy(
        tla.make_tensor(topk_out.ptr + start, row_shape),
        tla.make_tensor(output_lens, row_shape),
    )
    tla.set_flag(released)


@tla.kernel
def fold_cmp_indices(
    source: tla.Tensor,
    topk_in: tla.Tensor,
    cu_cmp: tla.Tensor,
    ids: tla.Tensor,
    log_counts: tla.Tensor,
    topk_out: tla.Tensor,
    rows: tla.Constexpr[int],
    width: tla.Constexpr[int],
    chunk_rows: tla.Constexpr[int],
) -> None:
    size = chunk_rows * width
    record0 = tla.allocate(size + 16, tla.Int32, tla.AddressSpace.ub, 256) + 8
    record1 = tla.allocate(size + 16, tla.Int32, tla.AddressSpace.ub, 256) + 8
    lens0 = tla.allocate(chunk_rows, tla.Int32, tla.AddressSpace.ub, 256)
    lens1 = tla.allocate(chunk_rows, tla.Int32, tla.AddressSpace.ub, 256)
    positions = tla.allocate(
        chunk_rows * (width + 64), tla.Int32, tla.AddressSpace.ub, 256
    )
    indices0 = tla.allocate(size, tla.Int32, tla.AddressSpace.ub, 256)
    indices1 = tla.allocate(size, tla.Int32, tla.AddressSpace.ub, 256)
    weights0 = tla.allocate(size, tla.Float32, tla.AddressSpace.ub, 256)
    weights1 = tla.allocate(size, tla.Float32, tla.AddressSpace.ub, 256)
    output0 = tla.allocate(chunk_rows, tla.Int32, tla.AddressSpace.ub, 256)
    output1 = tla.allocate(chunk_rows, tla.Int32, tla.AddressSpace.ub, 256)
    released0 = tla.flag("cmp_fold_empty0", tla.pipes.MTE3, tla.pipes.SCALAR, id=0)
    released1 = tla.flag("cmp_fold_empty1", tla.pipes.MTE3, tla.pipes.SCALAR, id=1)
    input_ready = tla.flag(
        "cmp_fold_record_ready", tla.pipes.MTE2, tla.pipes.SCALAR, id=0
    )
    source_ready = tla.flag(
        "cmp_fold_source_ready", tla.pipes.MTE2, tla.pipes.VECTOR, id=0
    )
    lengths_ready = tla.flag(
        "cmp_fold_lengths_ready", tla.pipes.SCALAR, tla.pipes.MTE3, id=0
    )
    rows_ready = tla.flag("cmp_fold_rows_ready", tla.pipes.VECTOR, tla.pipes.MTE3, id=0)
    with tla.vector():
        limit = cu_cmp[1] - cu_cmp[0]
        limit = limit if limit < width else width
        tla.set_flag(released0)
        tla.set_flag(released1)
        steps = (rows + chunk_rows - 1) // chunk_rows
        per_core = (steps + tla.arch.block_num() - 1) // tla.arch.block_num()
        begin = tla.arch.block_idx() * per_core
        finish = begin + per_core
        finish = steps if finish > steps else finish
        for chunk in tla.range(begin, finish):
            start = chunk * chunk_rows
            count = rows - start
            count = chunk_rows if count > chunk_rows else count
            if chunk % 2 == 0:
                _fold_indices_batch(
                    source,
                    topk_in,
                    ids,
                    log_counts,
                    topk_out,
                    record0,
                    lens0,
                    positions,
                    indices0,
                    weights0,
                    output0,
                    width,
                    start,
                    count,
                    limit,
                    released0,
                    input_ready,
                    source_ready,
                    lengths_ready,
                    rows_ready,
                )
            else:
                _fold_indices_batch(
                    source,
                    topk_in,
                    ids,
                    log_counts,
                    topk_out,
                    record1,
                    lens1,
                    positions,
                    indices1,
                    weights1,
                    output1,
                    width,
                    start,
                    count,
                    limit,
                    released1,
                    input_ready,
                    source_ready,
                    lengths_ready,
                    rows_ready,
                )
        tla.wait_flag(released0)
        tla.wait_flag(released1)


_CSA_PAIR_PLAN_SOURCE = (
    _FOLD_CMP_SOURCE
    + r"""
#include "kernel_operator.h"
using namespace AscendC::MicroAPI;
using AscendC::CMPMODE;
using AscendC::SpecialPurposeReg;

__simd_vf__ void ClearPairMap(__ubuf__ int32_t* map, int32_t span) {
    auto all = CreateMask<int32_t, MaskPattern::ALL>();
    RegTensor<int32_t> zero;
    Duplicate(zero, 0);
    for (uint16_t k = 0; k < span; k += 64) {
        StoreAlign(map + k, zero, all);
    }
}

__simd_vf__ void FindSortedRuns(__ubuf__ int32_t* out, __ubuf__ float* unordered,
                                __ubuf__ int32_t* source, uint32_t len) {
    auto all = CreateMask<int32_t, MaskPattern::ALL>();
    uint32_t first = 1;
    uint32_t zero_lanes = 0;
    auto head = UpdateMask<int32_t>(first);
    auto bad = UpdateMask<int32_t>(zero_lanes);
    auto* previous = source - 1;
    UnalignRegForLoad prior;
    UnalignRegForStore emitted;
    LoadUnAlignPre(prior, previous);
    uint32_t remaining = len;
    ClearSpr<SpecialPurposeReg::AR>();
    for (uint16_t k = 0; k < len; k += 64) {
        RegTensor<int32_t> ids, prev, positions, packed;
        auto live = UpdateMask<int32_t>(remaining);
        LoadAlign(ids, source + k);
        LoadUnAlign(prev, prior, previous, 64);
        MaskReg changed, backwards;
        Compare<int32_t, CMPMODE::NE>(changed, ids, prev, live);
        Compare<int32_t, CMPMODE::LT>(backwards, ids, prev, live);
        if (k == 0) {
            auto tail = all;
            MaskNot(tail, head, all);
            MaskOr(changed, changed, head, all);
            MaskAnd(backwards, backwards, tail, all);
        }
        MaskOr(bad, bad, backwards, all);
        Arange(positions, static_cast<int32_t>(k));
        GatherMask<int32_t, GatherMaskMode::STORE_REG>(packed, positions, changed);
        StoreUnAlign(out, packed, emitted);
    }
    StoreUnAlignPost(out, emitted);
    RegTensor<float> flags, any;
    Duplicate(flags, 1.0f, bad);
    ReduceMax(any, flags, all);
    StoreAlign(unordered, any, head);
    LocalMemBar<MemType::VEC_STORE, MemType::SCALAR_LOAD>();
}

template <bool First>
__simd_vf__ void EmitPartnerRuns(__ubuf__ int32_t* ids_out, __ubuf__ float* log_out,
                                 __ubuf__ int32_t* positions, __ubuf__ int32_t* source,
                                 __ubuf__ int32_t* map, __ubuf__ int32_t* shared,
                                 int32_t stamp, uint32_t groups) {
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LocalMemBar<MemType::SCALAR_STORE, MemType::VEC_LOAD>();
    auto* next = positions + 1;
    UnalignRegForLoad next_reg;
    LoadUnAlignPre(next_reg, next);
    RegTensor<int32_t> current;
    Duplicate(current, stamp);
    UnalignRegForStore emitted;
    if constexpr (!First) ClearSpr<SpecialPurposeReg::AR>();
    uint32_t remaining = groups;
    for (uint16_t k = 0; k < groups; k += 64) {
        RegTensor<int32_t> starts, ends, n, ids, ordinal;
        RegTensor<float> weights, logs;
        auto live = UpdateMask<int32_t>(remaining);
        LoadAlign(starts, positions + k);
        LoadUnAlign(ends, next_reg, next, 64);
        Gather(ids, source, (RegTensor<uint32_t>&)starts, live);
        Sub(n, ends, starts, live);
        Cast<float, int32_t, fold_float_cast>(weights, n, live);
        Ln(logs, weights, live);
        StoreAlign(ids_out + k, ids, live);
        StoreAlign(log_out + k, logs, live);
        if constexpr (First) {
            Arange(ordinal, static_cast<int32_t>(stamp + k));
            Scatter(map, ordinal, (RegTensor<uint32_t>&)ids, live);
        } else {
            RegTensor<int32_t> found, chosen;
            MaskReg common;
            Gather(found, map, (RegTensor<uint32_t>&)ids, live);
            Compare<int32_t, CMPMODE::GE>(common, found, current, live);
            Arange(ordinal, static_cast<int32_t>(k));
            GatherMask<int32_t, GatherMaskMode::STORE_REG>(chosen, ordinal, common);
            StoreUnAlign(shared, chosen, emitted);
        }
    }
    if constexpr (!First) StoreUnAlignPost(shared, emitted);
}

template <bool First>
[aicore] __attribute__((always_inline)) bool FoldPartner(
    uint64_t raw, uint64_t positions, uint64_t out, uint64_t logs, uint64_t lengths,
    uint64_t map, uint64_t shared, int32_t stamp, int32_t len, int32_t width, int32_t keys) {
    auto* src = reinterpret_cast<__ubuf__ int32_t*>(raw);
    if (src[0] < 0 || src[len - 1] >= keys) {
        fold_one_cmp_row(raw, positions, out, logs, lengths, len);
        return false;
    }
    auto* starts = reinterpret_cast<__ubuf__ int32_t*>(positions);
    auto* unordered = reinterpret_cast<__ubuf__ float*>(starts + width + 8);
    FindSortedRuns(starts, unordered, src, len);
    const int32_t groups = AscendC::GetSpr<SpecialPurposeReg::AR>() / sizeof(int32_t);
    auto* count = reinterpret_cast<__ubuf__ int32_t*>(lengths);
    count[0] = groups == len ? -groups : groups;
    starts[groups] = len;
    auto* ids_out = reinterpret_cast<__ubuf__ int32_t*>(out);
    auto* log_out = reinterpret_cast<__ubuf__ float*>(logs);
    if (groups < 128 || unordered[0] != 0) {
        EmitRuns(ids_out, log_out, starts, src, groups);
        return false;
    }
    EmitPartnerRuns<First>(ids_out, log_out, starts, src,
                           reinterpret_cast<__ubuf__ int32_t*>(map),
                           reinterpret_cast<__ubuf__ int32_t*>(shared), stamp, groups);
    if constexpr (First) return true;
    return AscendC::GetSpr<SpecialPurposeReg::AR>() >= 128 * sizeof(int32_t);
}

__simd_vf__ void PutSharedRows(__ubuf__ int32_t* chosen, __ubuf__ int32_t* map,
                               __ubuf__ int32_t* b_ids, __ubuf__ float* a_weights,
                               __ubuf__ float* b_weights, __ubuf__ int32_t* a_out,
                               __ubuf__ int32_t* b_out, __ubuf__ float* a_log,
                               __ubuf__ float* b_log, int32_t stamp) {
    auto all = CreateMask<int32_t, MaskPattern::ALL>();
    RegTensor<int32_t> used;
    Duplicate(used, -stamp);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint16_t k = 0; k < 128; k += 64) {
        RegTensor<int32_t> keys, a_pos, b_pos;
        RegTensor<float> aw, bw;
        LoadAlign(b_pos, chosen + k);
        Gather(keys, b_ids, (RegTensor<uint32_t>&)b_pos, all);
        Gather(a_pos, map, (RegTensor<uint32_t>&)keys, all);
        Adds(a_pos, a_pos, -stamp, all);
        Gather(aw, a_weights, (RegTensor<uint32_t>&)a_pos, all);
        Gather(bw, b_weights, (RegTensor<uint32_t>&)b_pos, all);
        StoreAlign(a_out + k, keys, all);
        StoreAlign(b_out + k, keys, all);
        StoreAlign(a_log + k, aw, all);
        StoreAlign(b_log + k, bw, all);
        Scatter(map, used, (RegTensor<uint32_t>&)keys, all);
    }
}

__simd_vf__ void PutUniqueRows(__ubuf__ int32_t* ids_out, __ubuf__ float* logs_out,
                               __ubuf__ int32_t* ids, __ubuf__ float* values,
                               __ubuf__ int32_t* map, __ubuf__ int32_t* positions,
                               int32_t len, int32_t stamp) {
    RegTensor<int32_t> used;
    Duplicate(used, -stamp);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    uint32_t remaining = len;
    UnalignRegForStore emitted;
    auto* dest = positions;
    ClearSpr<SpecialPurposeReg::AR>();
    for (uint16_t k = 0; k < len; k += 64) {
        RegTensor<int32_t> key, ordinal, index, packed;
        MaskReg keep;
        auto live = UpdateMask<int32_t>(remaining);
        LoadAlign(key, ids + k);
        Gather(ordinal, map, (RegTensor<uint32_t>&)key, live);
        Compare<int32_t, CMPMODE::NE>(keep, ordinal, used, live);
        Arange(index, static_cast<int32_t>(k));
        GatherMask<int32_t, GatherMaskMode::STORE_REG>(packed, index, keep);
        StoreUnAlign(dest, packed, emitted);
    }
    StoreUnAlignPost(dest, emitted);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    const uint32_t unique = len - 128;
    remaining = unique;
    for (uint16_t k = 0; k < unique; k += 64) {
        RegTensor<uint32_t> from;
        RegTensor<int32_t> keys;
        RegTensor<float> logs;
        auto live = UpdateMask<int32_t>(remaining);
        LoadAlign(from, reinterpret_cast<__ubuf__ uint32_t*>(positions + k));
        Gather(keys, ids, from, live);
        Gather(logs, values, from, live);
        StoreAlign(ids_out + k, keys, live);
        StoreAlign(logs_out + k, logs, live);
    }
}

[aicore] __attribute__((always_inline)) void PlanFoldedPair(
    uint64_t rows, uint64_t logs, uint64_t lens, uint64_t matching,
    uint64_t ids_temp, uint64_t weights_temp, int32_t width, uint64_t maps, int32_t stamp,
    int32_t gap, int32_t mark_gap, uint64_t position_storage) {
    auto* p = reinterpret_cast<__ubuf__ int32_t*>(lens);
    const int32_t a = p[0] < 0 ? -p[0] : p[0];
    const int32_t b = p[mark_gap] < 0 ? -p[mark_gap] : p[mark_gap];
    auto* ia = reinterpret_cast<__ubuf__ int32_t*>(rows);
    auto* ib = ia + width;
    auto* wa = reinterpret_cast<__ubuf__ float*>(logs);
    auto* wb = wa + width;
    auto* shared = reinterpret_cast<__ubuf__ int32_t*>(matching);
    auto* map = reinterpret_cast<__ubuf__ int32_t*>(maps);
    auto* out_a = reinterpret_cast<__ubuf__ int32_t*>(ids_temp);
    auto* out_b = out_a + gap;
    auto* log_a = reinterpret_cast<__ubuf__ float*>(weights_temp);
    auto* log_b = log_a + gap;
    const int32_t first_tail = (a - 128 + 127) / 128 * 128;
    PutSharedRows(shared, map, ib, wa, wb, out_a + first_tail, out_b, log_a + first_tail, log_b, stamp);
    auto* positions = reinterpret_cast<__ubuf__ int32_t*>(position_storage);
    if (a > 128) PutUniqueRows(out_a, log_a, ia, wa, map, positions, a, stamp);
    if (b > 128) PutUniqueRows(out_b + 128, log_b + 128, ib, wb, map, positions, b, stamp);
    p[0] += p[0] < 0 ? -(width + 1) : width + 1;
    p[mark_gap] += p[mark_gap] < 0 ? -(width + 1) : width + 1;
}

__simd_vf__ void CopyFoldedRow(__ubuf__ int32_t* out_ids, __ubuf__ float* out_weights,
                              __ubuf__ int32_t* ids, __ubuf__ float* weights, int32_t count) {
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    uint32_t remaining = count;
    for (uint16_t i = 0; i < count; i += 64) {
        RegTensor<int32_t> keys;
        RegTensor<float> logs;
        auto live = UpdateMask<int32_t>(remaining);
        LoadAlign(keys, ids + i);
        LoadAlign(logs, weights + i);
        StoreAlign(out_ids + i, keys, live);
        StoreAlign(out_weights + i, logs, live);
    }
}

extern "C" [aicore] __attribute__((always_inline)) void prepare_ring_index_batch(
    uint64_t raw, uint64_t lengths, uint64_t final_ids, uint64_t final_weights, uint64_t marks,
    uint64_t positions, uint64_t pair_ids, uint64_t pair_weights, uint64_t matches, uint64_t maps,
    int32_t width, int32_t count, int32_t span, int32_t keys, int32_t step, int32_t stamp_base) {
    auto* input_lens = reinterpret_cast<__ubuf__ int32_t*>(lengths);
    auto* output_lens = reinterpret_cast<__ubuf__ int32_t*>(marks);
    const int32_t limit = keys < width ? keys : width;
    if (step == 2 && stamp_base == 0) ClearPairMap(reinterpret_cast<__ubuf__ int32_t*>(maps), span);
    for (int32_t row = 0; row < count; row += step) {
        const int32_t in_a = input_lens[row];
        const int32_t len_a = in_a > limit ? limit : in_a > 0 ? in_a : 0;
        const int32_t start = row * width;
        auto* ca = output_lens + 8 * row;
        const bool has_b = step == 2 && row + 1 < count;
        const int32_t in_b = has_b ? input_lens[row + 1] : 0;
        const int32_t len_b = in_b > limit ? limit : in_b > 0 ? in_b : 0;
        const int32_t stamp = (stamp_base + row / 2 + 1) * width;
        const uint64_t row_a = raw + start * 4;
        const uint64_t row_b = row_a + width * 4;
        bool mapped = false;
        if (len_a >= 128 && len_b >= 128) {
            mapped = FoldPartner<true>(row_a, positions, pair_ids, pair_weights, reinterpret_cast<uint64_t>(ca),
                                       maps, matches, stamp, len_a, width, keys);
        } else {
            fold_one_cmp_row(row_a, positions, pair_ids, pair_weights, reinterpret_cast<uint64_t>(ca), len_a);
        }
        if (has_b) {
            auto* cb = ca + 8;
            const uint64_t ids_b = pair_ids + width * 4;
            const uint64_t log_b = pair_weights + width * 4;
            if (mapped) {
                const bool matched = FoldPartner<false>(row_b, positions, ids_b, log_b, reinterpret_cast<uint64_t>(cb),
                                                        maps, matches, stamp, len_b, width, keys);
                if (matched) {
                    PlanFoldedPair(pair_ids, pair_weights, reinterpret_cast<uint64_t>(ca), matches,
                                   final_ids + start * 4, final_weights + start * 4, width,
                                   maps, stamp, width, 8, positions);
                }
            } else {
                fold_one_cmp_row(row_b, positions, ids_b, log_b, reinterpret_cast<uint64_t>(cb), len_b);
            }
        }
        if (ca[0] <= width && ca[0] >= -width) {
            auto* ids = reinterpret_cast<__ubuf__ int32_t*>(final_ids) + start;
            auto* weights = reinterpret_cast<__ubuf__ float*>(final_weights) + start;
            auto* folded = reinterpret_cast<__ubuf__ int32_t*>(pair_ids);
            auto* logs = reinterpret_cast<__ubuf__ float*>(pair_weights);
            const int32_t len = ca[0] < 0 ? -ca[0] : ca[0];
            CopyFoldedRow(ids, weights, folded, logs, len);
            if (has_b) {
                const int32_t cb = ca[8];
                CopyFoldedRow(ids + width, weights + width, folded + width, logs + width, cb < 0 ? -cb : cb);
            }
        }
    }
}
"""
)


@tla.extern(source=_CSA_PAIR_PLAN_SOURCE, name="prepare_ring_index_batch")
def _prepare_ring_index_batch(
    raw: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    lengths: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    output: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    weights: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    marks: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    positions: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    pair_ids: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    pair_weights: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    matches: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    maps: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    width: tla.Int32,
    count: tla.Int32,
    span: tla.Int32,
    keys: tla.Int32,
    step: tla.Int32,
    stamp_base: tla.Int32,
) -> None: ...


@tla.jit
def _prepare_ring_io(
    source,
    topk_in,
    ids,
    weights,
    topk_out,
    input_ids,
    input_lens,
    out_ids,
    out_weights,
    out_lens,
    positions,
    pair_ids,
    pair_weights,
    matches,
    maps,
    first,
    width,
    count,
    span,
    keys,
    step,
    stamp_base,
    released,
    input_s,
    input_v,
    output_s,
    output_v,
    prep_ready=None,
):
    _prefetch_ring_index_batch(
        source,
        topk_in,
        input_ids,
        input_lens,
        first,
        width,
        count,
        released,
        input_s,
        input_v,
        cold=prep_ready is not None,
    )
    if prep_ready is not None:
        tla.wait_flag(prep_ready)
    _consume_ring_index_batch(
        ids,
        weights,
        topk_out,
        input_ids,
        input_lens,
        out_ids,
        out_weights,
        out_lens,
        positions,
        pair_ids,
        pair_weights,
        matches,
        maps,
        first,
        width,
        count,
        span,
        keys,
        step,
        stamp_base,
        released,
        input_s,
        input_v,
        output_s,
        output_v,
    )


@tla.jit
def _prefetch_ring_index_batch(
    source,
    topk_in,
    input_ids,
    input_lens,
    first,
    width,
    count,
    released,
    input_s,
    input_v,
    cold: tla.Constexpr[bool] = False,
):
    index_layout = tla.make_layout(tla.make_shape(count * width), tla.make_stride(1))
    length_layout = tla.make_layout(tla.make_shape(count), tla.make_stride(1))
    if not cold:
        tla.wait_flag(released)
    tla.copy(
        tla.make_tensor(input_ids, index_layout),
        tla.make_tensor(source.ptr + first * width, index_layout),
    )
    tla.copy(
        tla.make_tensor(input_lens, length_layout),
        tla.make_tensor(topk_in.ptr + first, length_layout),
    )
    tla.set_flag(input_s)
    tla.set_flag(input_v)


@tla.jit
def _consume_ring_index_batch(
    ids,
    weights,
    topk_out,
    input_ids,
    input_lens,
    out_ids,
    out_weights,
    out_lens,
    positions,
    pair_ids,
    pair_weights,
    matches,
    maps,
    first,
    width,
    count,
    span,
    keys,
    step,
    stamp_base,
    released,
    input_s,
    input_v,
    output_s,
    output_v,
):
    index_layout = tla.make_layout(tla.make_shape(count * width), tla.make_stride(1))
    mark_layout = tla.make_layout(tla.make_shape(count * 8), tla.make_stride(1))
    tla.wait_flag(input_s)
    tla.wait_flag(input_v)
    _prepare_ring_index_batch(
        input_ids,
        input_lens,
        out_ids,
        out_weights,
        out_lens,
        positions,
        pair_ids,
        pair_weights,
        matches,
        maps,
        width,
        count,
        span,
        keys,
        step,
        stamp_base,
    )
    tla.set_flag(output_s)
    tla.set_flag(output_v)
    tla.wait_flag(output_s)
    tla.wait_flag(output_v)
    tla.copy(
        tla.make_tensor(ids.ptr + first * width, index_layout),
        tla.make_tensor(out_ids, index_layout),
    )
    tla.copy(
        tla.make_tensor(weights.ptr + first * width, index_layout),
        tla.make_tensor(out_weights, index_layout),
    )
    tla.copy(
        tla.make_tensor(topk_out.ptr + first * 8, mark_layout),
        tla.make_tensor(out_lens, mark_layout),
    )
    tla.set_flag(released)


@tla.jit
def _fold_and_pair_ring_indices(
    source: tla.Tensor,
    topk: tla.Tensor,
    cu_cmp: tla.Tensor,
    ids: tla.Tensor,
    weights: tla.Tensor,
    counts: tla.Tensor,
    rows: tla.Constexpr[int],
    width: tla.Constexpr[int],
    span: tla.Constexpr[int],
    batch_pairs: tla.Constexpr[int],
    prefix_rows: tla.Constexpr[int],
    storage,
    prep_ready=None,
    prime_second_batch: tla.Constexpr[bool] = False,
    standalone_at_tail: tla.Constexpr[bool] = False,
):
    size = 2 * batch_pairs * width
    start_raw = 2 * size if prep_ready is not None else 0
    raw0 = storage + start_raw + 8
    raw1 = storage + start_raw + size + 24
    lens0 = tla.allocate(2 * batch_pairs, tla.Int32, tla.AddressSpace.ub, 256)
    lens1 = tla.allocate(2 * batch_pairs, tla.Int32, tla.AddressSpace.ub, 256)
    start_out = 0 if prep_ready is not None else 2 * size + 64
    index0 = storage + start_out
    index1 = storage + start_out + size
    weights0 = tla.allocate(size, tla.Float32, tla.AddressSpace.ub, 256)
    weights1 = tla.allocate(size, tla.Float32, tla.AddressSpace.ub, 256)
    out0 = tla.allocate(16 * batch_pairs, tla.Int32, tla.AddressSpace.ub, 256)
    out1 = tla.allocate(16 * batch_pairs, tla.Int32, tla.AddressSpace.ub, 256)
    positions = tla.allocate(width + 64, tla.Int32, tla.AddressSpace.ub, 256)
    pair_ids = tla.allocate(2 * width + 16, tla.Int32, tla.AddressSpace.ub, 256) + 8
    pair_weights = tla.allocate(2 * width, tla.Float32, tla.AddressSpace.ub, 256)
    matches = tla.allocate(width, tla.Int32, tla.AddressSpace.ub, 256)
    maps = tla.allocate(span, tla.Int32, tla.AddressSpace.ub, 256)
    event0 = 0 if prep_ready is None else 4
    event1 = event0 + 1
    released0 = tla.flag(
        "fold_pair_buffer0", tla.pipes.MTE3, tla.pipes.SCALAR, id=event0
    )
    released1 = tla.flag(
        "fold_pair_buffer1", tla.pipes.MTE3, tla.pipes.SCALAR, id=event1
    )
    input_s = tla.flag("fold_pair_source", tla.pipes.MTE2, tla.pipes.SCALAR, id=event0)
    input_v = tla.flag(
        "fold_pair_vector_source", tla.pipes.MTE2, tla.pipes.VECTOR, id=event0
    )
    input_s1 = tla.flag(
        "fold_pair_source1", tla.pipes.MTE2, tla.pipes.SCALAR, id=event1
    )
    input_v1 = tla.flag(
        "fold_pair_vector_source1", tla.pipes.MTE2, tla.pipes.VECTOR, id=event1
    )
    output_s = tla.flag("fold_pair_marks", tla.pipes.SCALAR, tla.pipes.MTE3, id=event0)
    output_v = tla.flag(
        "fold_pair_results", tla.pipes.VECTOR, tla.pipes.MTE3, id=event0
    )
    with tla.vector():
        keys = cu_cmp[1] - cu_cmp[0]
        if prep_ready is None:
            tla.set_flag(released0)
        tla.set_flag(released1)
        batch_rows = 2 * batch_pairs
        core = tla.arch.block_idx()
        grid = tla.arch.block_num()
        prefix_blocks = (prefix_rows + batch_rows - 1) // batch_rows
        used0 = tla.as_numeric(0)
        if core < prefix_blocks:
            used0 = tla.as_numeric(1)
            single_first = core * batch_rows
            count = prefix_rows - single_first
            count = count if count < batch_rows else batch_rows
            first = (
                rows - prefix_rows + single_first
                if standalone_at_tail
                else single_first
            )
            _prepare_ring_io(
                source,
                topk,
                ids,
                weights,
                counts,
                raw0,
                lens0,
                index0,
                weights0,
                out0,
                positions,
                pair_ids,
                pair_weights,
                matches,
                maps,
                first,
                width,
                count,
                span,
                keys,
                1,
                0,
                released0,
                input_s,
                input_v,
                output_s,
                output_v,
                prep_ready,
            )
        else:
            paired_core = core - prefix_blocks
            paired_grid = grid - prefix_blocks
            num_pairs = (rows - prefix_rows + 1) // 2
            base_pairs = num_pairs // paired_grid
            remainder = num_pairs % paired_grid
            first_pair = paired_core * base_pairs + (
                paired_core if paired_core < remainder else remainder
            )
            pairs = base_pairs + (1 if paired_core < remainder else 0)
            pair_offset = 0 if standalone_at_tail else prefix_rows
            pair_end = rows - prefix_rows if standalone_at_tail else rows
            first_row = pair_offset + 2 * first_pair
            end = first_row + 2 * pairs
            end = end if end < pair_end else pair_end
            groups = (pairs + batch_pairs - 1) // batch_pairs
            # The source DMA for the next batch owns a separate bank. Its
            # credits let MTE2 advance while Vector prepares the current batch.
            if groups > 0:
                used0 = tla.as_numeric(1)
                primed_count = end - first_row
                primed_count = primed_count if primed_count < batch_rows else batch_rows
                _prefetch_ring_index_batch(
                    source,
                    topk,
                    raw0,
                    lens0,
                    first_row,
                    width,
                    primed_count,
                    released0,
                    input_s,
                    input_v,
                    cold=prep_ready is not None,
                )
            if prep_ready is not None:
                if prime_second_batch and groups > 1:
                    second_row = first_row + batch_rows
                    second_count = end - second_row
                    second_count = (
                        second_count if second_count < batch_rows else batch_rows
                    )
                    _prefetch_ring_index_batch(
                        source,
                        topk,
                        raw1,
                        lens1,
                        second_row,
                        width,
                        second_count,
                        released1,
                        input_s1,
                        input_v1,
                    )
                tla.wait_flag(prep_ready)
            for p in tla.range(groups):
                first = first_row + p * batch_rows
                left = end - first
                count = left if left < batch_rows else batch_rows
                next_first = first + batch_rows
                next_left = end - next_first
                next_count = next_left if next_left < batch_rows else batch_rows
                stamp_base = (p % 64) * batch_pairs
                if p % 2 == 0:
                    prime_next = (
                        p > 0 and p + 1 < groups
                        if prime_second_batch and prep_ready is not None
                        else p + 1 < groups
                    )
                    if prime_next:
                        _prefetch_ring_index_batch(
                            source,
                            topk,
                            raw1,
                            lens1,
                            next_first,
                            width,
                            next_count,
                            released1,
                            input_s1,
                            input_v1,
                        )
                    _consume_ring_index_batch(
                        ids,
                        weights,
                        counts,
                        raw0,
                        lens0,
                        index0,
                        weights0,
                        out0,
                        positions,
                        pair_ids,
                        pair_weights,
                        matches,
                        maps,
                        first,
                        width,
                        count,
                        span,
                        keys,
                        2,
                        stamp_base,
                        released0,
                        input_s,
                        input_v,
                        output_s,
                        output_v,
                    )
                else:
                    if p + 1 < groups:
                        _prefetch_ring_index_batch(
                            source,
                            topk,
                            raw0,
                            lens0,
                            next_first,
                            width,
                            next_count,
                            released0,
                            input_s,
                            input_v,
                        )
                    _consume_ring_index_batch(
                        ids,
                        weights,
                        counts,
                        raw1,
                        lens1,
                        index1,
                        weights1,
                        out1,
                        positions,
                        pair_ids,
                        pair_weights,
                        matches,
                        maps,
                        first,
                        width,
                        count,
                        span,
                        keys,
                        2,
                        stamp_base,
                        released1,
                        input_s1,
                        input_v1,
                        output_s,
                        output_v,
                    )
        if prep_ready is None:
            tla.wait_flag(released0)
        elif used0 == 1:
            tla.wait_flag(released0)
        tla.wait_flag(released1)


@tla.kernel
def fold_and_pair_ring_indices(
    source: tla.Tensor,
    topk: tla.Tensor,
    cu_cmp: tla.Tensor,
    ids: tla.Tensor,
    weights: tla.Tensor,
    counts: tla.Tensor,
    rows: tla.Constexpr[int],
    width: tla.Constexpr[int],
    span: tla.Constexpr[int],
    batch_pairs: tla.Constexpr[int],
    prefix_rows: tla.Constexpr[int],
    standalone_at_tail: tla.Constexpr[bool] = False,
):
    storage = tla.allocate(
        8 * batch_pairs * width + 64, tla.Int32, tla.AddressSpace.ub, 256
    )
    _fold_and_pair_ring_indices(
        source,
        topk,
        cu_cmp,
        ids,
        weights,
        counts,
        rows,
        width,
        span,
        batch_pairs,
        prefix_rows,
        storage,
        standalone_at_tail=standalone_at_tail,
    )


@tla.kernel
def dequant_and_fold_indices(
    gm_nope_a: tla.Tensor,
    gm_rope_a: tla.Tensor,
    gm_scale_a: tla.Tensor,
    gm_pages_a: tla.Tensor,
    gm_out_a: tla.Tensor,
    gm_nope_b: tla.Tensor,
    gm_rope_b: tla.Tensor,
    gm_scale_b: tla.Tensor,
    gm_pages_b: tla.Tensor,
    gm_out_b: tla.Tensor,
    source: tla.Tensor,
    topk: tla.Tensor,
    cu_cmp: tla.Tensor,
    ids: tla.Tensor,
    weights: tla.Tensor,
    counts: tla.Tensor,
    rows: tla.Constexpr[int],
    width: tla.Constexpr[int],
    span: tla.Constexpr[int],
    batch_pairs: tla.Constexpr[int],
    prefix_rows: tla.Constexpr[int],
    cfg: tla.Constexpr[DequantPagesPairCfg],
    prefetch_indices: tla.Constexpr[bool] = False,
    prime_second_batch: tla.Constexpr[bool] = False,
    standalone_at_tail: tla.Constexpr[bool] = False,
):
    packed_rows = max(cfg.rows_per_block_a, cfg.rows_per_block_b)
    bytes_per_row = REC_BYTES + 2 * ROPE_DIM + 4 * SCALE_STRIDE + 2 * HEAD_DIM
    storage = tla.allocate(
        max(8 * batch_pairs * width + 64, (packed_rows * bytes_per_row + 3) // 4),
        tla.Int32,
        tla.AddressSpace.ub,
        256,
    )
    # Raw input banks lie above the region used by dequantization. Start
    # their first fetch before handing the lower output region to the planner.
    overlap = (
        prefetch_indices and packed_rows * bytes_per_row <= 16 * batch_pairs * width
    )
    prep_ready = (
        tla.flag("dequant_for_fold", tla.pipes.MTE3, tla.pipes.SCALAR, id=7)
        if overlap
        else None
    )
    _dequant_page_pair(
        gm_nope_a,
        gm_rope_a,
        gm_scale_a,
        gm_pages_a,
        gm_out_a,
        gm_nope_b,
        gm_rope_b,
        gm_scale_b,
        gm_pages_b,
        gm_out_b,
        cfg,
        storage,
        prep_ready,
    )
    _fold_and_pair_ring_indices(
        source,
        topk,
        cu_cmp,
        ids,
        weights,
        counts,
        rows,
        width,
        span,
        batch_pairs,
        prefix_rows,
        storage,
        prep_ready,
        prime_second_batch,
        standalone_at_tail,
    )


@tla.jit
def _fold_kv_offset(bt, logical, shift, mask, page_size):
    pos = logical if logical >= 0 else 0
    page = bt[pos >> shift]
    physical = page if page >= 0 else tla.Int32(0)
    return (
        physical.to(tla.Int64) * tla.Int64(page_size) + (pos & mask).to(tla.Int64)
    ) * tla.Int64(1024)


@tla.jit
def _fold_gather_rows(
    dst: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    pool: tla.Pointer[tla.UInt8, tla.AddressSpace.gm],
    ids: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    table: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    start: tla.Int32,
    valid: tla.Int32,
    rows: tla.Int32,
    page_shift: tla.Int32,
) -> None:
    page_size = 1 << page_shift
    mask = page_size - 1
    ids_v = tla.make_tensor(
        ids, tla.make_layout(tla.make_shape(valid), tla.make_stride(1))
    )
    bt = tla.make_tensor(
        table,
        tla.make_layout(
            tla.make_shape((valid >> page_shift) + 2),
            tla.make_stride(1),
        ),
    )
    available = valid - start
    available = available if available < rows else rows
    paired = (available // 2) * 2 if available > 0 else 0
    pair_hi = tla.Int64(1 << 30)
    for i in tla.range(0, paired, 2):
        a = _fold_kv_offset(bt, ids_v[start + i], page_shift, mask, page_size)
        b = _fold_kv_offset(bt, ids_v[start + i + 1], page_shift, mask, page_size)
        if b > a and b - a < pair_hi:
            _copy_byte_rows(dst + i * 1024, pool + a, 2, 1024, b - a)
        else:
            _copy_byte_rows(dst + i * 1024, pool + a, 1, 1024, 1024)
            _copy_byte_rows(dst + i * 1024 + 1024, pool + b, 1, 1024, 1024)
    if paired < rows:
        tail = start + paired
        tail = tail if tail < valid else valid - 1
        last = _fold_kv_offset(bt, ids_v[tail], page_shift, mask, page_size)
        for j in tla.range(paired, rows):
            _copy_byte_rows(dst + j * 1024, pool + last, 1, 1024, 1024)


@tla.jit
def gather_folded_to_workspace(
    sync,
    bufs,
    ws,
    kv,
    ids,
    offset,
    valid,
    stored,
    slot,
    ws_base,
    shift,
    alignment: tla.Constexpr[int] = 8,
):
    """Keep each KV row paired with its multiplicity when pages are out of order."""
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        half = ((stored + 1) // 2 + (alignment - 1)) // alignment * alignment
        start = sub * half
        count = half if sub == 0 else stored - half
        ids_layout = tla.make_layout(tla.make_shape(KV_BLOCK), tla.make_stride(1))
        tla.copy(
            tla.make_tensor(bufs.ub_phy_idx, ids_layout),
            tla.make_tensor(ids.ptr + offset, ids_layout),
        )
        tla.set_flag(sync.idx_m2s0)
        tla.wait_flag(sync.idx_m2s0)
        chunk_rows = 2 * KV_UNIT
        output_lock = sync.m_o16
        first_rows = count if count < chunk_rows else chunk_rows
        if first_rows > 0:
            tla.wait_flag(sync.kv_mte3_mte2_0)
            _fold_gather_rows(
                tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                tla.recast_ptr(kv.ptr, dtype=tla.UInt8),
                bufs.ub_phy_idx,
                bufs.ub_phy_bt,
                start,
                valid,
                first_rows,
                shift,
            )
            tla.set_flag(sync.kv_mte2_mte3_0)
            tla.wait_flag(sync.kv_mte2_mte3_0)
            first_view = tla.make_layout(
                tla.make_shape(first_rows * HEAD_DIM), tla.make_stride(1)
            )
            first_dst = tla.make_tensor(
                ws.ptr + ws_base + slot * KV_STAGE_STRIDE + start * HEAD_DIM, first_view
            )
            tla.copy(first_dst, tla.make_tensor(bufs.ub_kv0, first_view))
            tla.set_flag(sync.kv_mte3_mte2_0)
        last_rows = count - first_rows
        if last_rows > 0:
            # The second gather chunk shares the output's BF16 staging buffer.
            start_last = start + chunk_rows
            output_lock.lock(pipe=tla.pipes.MTE2)
            _fold_gather_rows(
                tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
                tla.recast_ptr(kv.ptr, dtype=tla.UInt8),
                bufs.ub_phy_idx,
                bufs.ub_phy_bt,
                start_last,
                valid,
                last_rows,
                shift,
            )
            output_lock.unlock(pipe=tla.pipes.MTE2)
            output_lock.lock(pipe=tla.pipes.MTE3)
            last_view = tla.make_layout(
                tla.make_shape(last_rows * HEAD_DIM), tla.make_stride(1)
            )
            last_dst = tla.make_tensor(
                ws.ptr + ws_base + slot * KV_STAGE_STRIDE + start_last * HEAD_DIM,
                last_view,
            )
            tla.copy(last_dst, tla.make_tensor(bufs.ub_kv1, last_view))
            output_lock.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


_CSA_VEC1_SOURCE = r"""
#include "kernel_operator.h"
using namespace AscendC::MicroAPI;

static constexpr CastTrait csa_probs = {RegLayout::ZERO, SatMode::SAT,
                                MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

enum class CsaWidth { Narrow, Tail, Full };

template <bool Weighted, CsaWidth Width>
__simd_vf__ void CsaSoftmaxRows(
    __ubuf__ float* scores, __ubuf__ bfloat16_t* prob,
    __ubuf__ float* max_read, __ubuf__ float* sum_read,
    __ubuf__ float* max_write, __ubuf__ float* sum_write,
    __ubuf__ float* decay_write, __ubuf__ float* weights,
    int32_t count, float scale) {
    auto full = CreateMask<float, MaskPattern::ALL>();
    auto bf16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    uint32_t left = 1;
    auto one = UpdateMask<float>(left);
    uint32_t n0 = count < 64 ? count : 64;
    uint32_t n1 = count > 64 ? count - 64 : 0;
    auto mask0 = UpdateMask<float>(n0);
    auto mask1 = UpdateMask<float>(n1);
    RegTensor<float> neg_inf, scale_reg, w0, w1;
    Duplicate(neg_inf, -3.4028234663852886e38f);
    Duplicate(scale_reg, scale);
    if constexpr (Weighted) {
        LoadAlign(w0, weights);
        if constexpr (Width != CsaWidth::Narrow) {
            LoadAlign(w1, weights + 64);
        }
    }
    UnalignRegForStore max_out, sum_out, decay_out;
    for (uint16_t row = 0; row < 32; ++row) {
        RegTensor<float> prev_max, prev_sum, s0, s1, l0, l1;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(prev_max, max_read + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(prev_sum, sum_read + row);
        LoadAlign(s0, scores + row * 128);
        if constexpr (Weighted) {
            l0 = w0;
            MulAddDst(l0, s0, scale_reg, full);
        } else {
            Muls(l0, s0, scale, full);
        }
        if constexpr (Width == CsaWidth::Narrow) {
            Select(l0, l0, neg_inf, mask0);
        } else {
            LoadAlign(s1, scores + row * 128 + 64);
            if constexpr (Weighted) {
                l1 = w1;
                MulAddDst(l1, s1, scale_reg, full);
            } else {
                Muls(l1, s1, scale, full);
            }
            if constexpr (Width == CsaWidth::Tail) {
                Select(l1, l1, neg_inf, mask1);
            }
        }
        RegTensor<float> peak, decay, a, b, denom;
        if constexpr (Width == CsaWidth::Narrow) {
            ReduceMax(peak, l0, full);
        } else {
            Max(peak, l0, l1, full);
            ReduceMax(peak, peak, full);
        }
        Duplicate(peak, peak, full);
        Max(peak, peak, prev_max, full);
        ExpSub(decay, prev_max, peak, one);
        ExpSub(a, l0, peak, mask0);
        RegTensor<bfloat16_t> pa, pb, packed, unused;
        Cast<bfloat16_t, float, csa_probs>(pa, a, full);
        if constexpr (Width == CsaWidth::Narrow) {
            Duplicate(pb, 0);
            ReduceSum(denom, a, full);
        } else {
            ExpSub(b, l1, peak, mask1);
            Cast<bfloat16_t, float, csa_probs>(pb, b, full);
            Add(a, a, b, full);
            ReduceSum(denom, a, full);
        }
        MulAddDst(denom, prev_sum, decay, one);
        DeInterleave(packed, unused, pa, pb);
        auto* dst_row = prob + row * 16;
        StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            dst_row, packed, 32, 1, bf16);
        StoreUnAlign(max_write, peak, max_out, 1);
        StoreUnAlign(sum_write, denom, sum_out, 1);
        StoreUnAlign(decay_write, decay, decay_out, 1);
    }
    StoreUnAlignPost(max_write, max_out, 0);
    StoreUnAlignPost(sum_write, sum_out, 0);
    StoreUnAlignPost(decay_write, decay_out, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <bool Weighted>
[aicore] __attribute__((always_inline)) void CsaSelectSoftmax(
    __ubuf__ float* s, __ubuf__ bfloat16_t* p, __ubuf__ float* mr, __ubuf__ float* sr,
    __ubuf__ float* mw, __ubuf__ float* sw, __ubuf__ float* dm, __ubuf__ float* w,
    int32_t count, float scale) {
    if (count == 128) {
        CsaSoftmaxRows<Weighted, CsaWidth::Full>(s, p, mr, sr, mw, sw, dm, w, count, scale);
    } else if (count <= 64) {
        CsaSoftmaxRows<Weighted, CsaWidth::Narrow>(s, p, mr, sr, mw, sw, dm, w, count, scale);
    } else {
        CsaSoftmaxRows<Weighted, CsaWidth::Tail>(s, p, mr, sr, mw, sw, dm, w, count, scale);
    }
}

extern "C" [aicore] __attribute__((always_inline)) void csa_vec1(
    uint64_t scores, uint64_t probs, uint64_t max_read, uint64_t sum_read,
    uint64_t max_write, uint64_t sum_write, uint64_t decay, uint64_t weights,
    int32_t count, float scale, int32_t weighted) {
    auto* s = reinterpret_cast<__ubuf__ float*>(scores);
    auto* p = reinterpret_cast<__ubuf__ bfloat16_t*>(probs);
    auto* mr = reinterpret_cast<__ubuf__ float*>(max_read);
    auto* sr = reinterpret_cast<__ubuf__ float*>(sum_read);
    auto* mw = reinterpret_cast<__ubuf__ float*>(max_write);
    auto* sw = reinterpret_cast<__ubuf__ float*>(sum_write);
    auto* dm = reinterpret_cast<__ubuf__ float*>(decay);
    auto* w = reinterpret_cast<__ubuf__ float*>(weights);
    if (weighted != 0) {
        CsaSelectSoftmax<true>(s, p, mr, sr, mw, sw, dm, w, count, scale);
    } else {
        CsaSelectSoftmax<false>(s, p, mr, sr, mw, sw, dm, w, count, scale);
    }
}
"""


_CSA_FIXED_VEC1_SOURCE = r"""
#include "kernel_operator.h"
using namespace AscendC::MicroAPI;

constexpr CastTrait csa_fixed_probs = {RegLayout::ZERO, SatMode::SAT,
                                MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

// The bound check uses the unsigned magnitude bits of IEEE FP32. NaNs and
// infinities exceed the limit, so this also selects the stable replay for them.
// For |score| <= 32, subtracting 64*ln(2) keeps every live BF16 weight normal
// and keeps the sum of weights below 1 even at the maximum tile count.
constexpr float csa_fixed_anchor = 44.36141956f;

__simd_callee__ inline void TrackMagnitude(RegTensor<int32_t>& upper,
    RegTensor<float>& value, MaskReg& valid, MaskReg& full) {
    RegTensor<int32_t> bits, signless, zero;
    Duplicate(signless, 0x7fffffff);
    Duplicate(zero, 0);
    And(bits, reinterpret_cast<RegTensor<int32_t>&>(value), signless, full);
    Select(bits, bits, zero, valid);
    Max(upper, upper, bits, full);
}

__simd_callee__ inline void CommitMagnitude(__ubuf__ int32_t* flag, RegTensor<int32_t>& upper) {
    auto full = CreateMask<int32_t, MaskPattern::ALL>();
    uint32_t lanes = 1;
    auto one = UpdateMask<int32_t>(lanes);
    RegTensor<int32_t> limit, zero, bad, prev;
    Duplicate(limit, 0x42000000); // 32.0 in FP32
    Duplicate(zero, 0);
    Duplicate(bad, 1);
    ReduceMax(upper, upper, full);
    MaskReg outside;
    Compare<int32_t, AscendC::CMPMODE::GT>(outside, upper, limit, one);
    Select(bad, bad, zero, outside);
    LoadAlign(prev, flag);
    Max(prev, prev, bad, one);
    StoreAlign(flag, prev, one);
}

__simd_vf__ void CsaFixedInitialSums(__ubuf__ float* sinks, __ubuf__ float* sums,
                                    __ubuf__ float* maxima, int32_t has_sinks, __ubuf__ int32_t* guard) {
    uint32_t rows = 32;
    auto live = UpdateMask<float>(rows);
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> anchor, values, initial;
    Duplicate(anchor, csa_fixed_anchor);
    if (has_sinks) {
        LoadAlign(values, sinks);
        RegTensor<int32_t> magnitude;
        Duplicate(magnitude, 0);
        TrackMagnitude(magnitude, values, live, full);
        CommitMagnitude(guard, magnitude);
        ExpSub(initial, values, anchor, live);
        StoreAlign(sums, initial, live);
    } else {
        Duplicate(initial, 0.0f);
        StoreAlign(sums, initial, live);
    }
    StoreAlign(maxima, anchor, live);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

extern "C" [aicore] __attribute__((always_inline)) void csa_fixed_init(
    uint64_t sinks, uint64_t sums, uint64_t maxima, int32_t has_sinks, uint64_t guard) {
    CsaFixedInitialSums(reinterpret_cast<__ubuf__ float*>(sinks),
                        reinterpret_cast<__ubuf__ float*>(sums),
                        reinterpret_cast<__ubuf__ float*>(maxima), has_sinks,
                        reinterpret_cast<__ubuf__ int32_t*>(guard));
}

template <bool Wide>
__simd_vf__ void CsaFixedExponentRows(
    __ubuf__ float* scores, __ubuf__ bfloat16_t* prob, __ubuf__ float* sum_read,
    __ubuf__ float* sum_write, __ubuf__ float* max_write,
    int32_t count, float scale,
    __ubuf__ int32_t* guard) {
    auto full = CreateMask<float, MaskPattern::ALL>();
    auto bf16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    uint32_t left = 1;
    auto one = UpdateMask<float>(left);
    uint32_t n0 = count < 64 ? count : 64;
    uint32_t n1 = count > 64 ? count - 64 : 0;
    auto mask0 = UpdateMask<float>(n0);
    auto mask1 = UpdateMask<float>(n1);
    RegTensor<float> anchor;
    Duplicate(anchor, csa_fixed_anchor);
    UnalignRegForStore sum_out;
    RegTensor<int32_t> magnitude;
    Duplicate(magnitude, 0);
    for (uint16_t row = 0; row < 32; ++row) {
        RegTensor<float> prev, s0, s1, a, b, denom;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(prev, sum_read + row);
        LoadAlign(s0, scores + row * 128);
        Muls(s0, s0, scale, full);
        TrackMagnitude(magnitude, s0, mask0, full);
        ExpSub(a, s0, anchor, mask0);
        RegTensor<bfloat16_t> pa, pb, packed, unused;
        Cast<bfloat16_t, float, csa_fixed_probs>(pa, a, full);
        if constexpr (Wide) {
            LoadAlign(s1, scores + row * 128 + 64);
            Muls(s1, s1, scale, full);
            TrackMagnitude(magnitude, s1, mask1, full);
            ExpSub(b, s1, anchor, mask1);
            Cast<bfloat16_t, float, csa_fixed_probs>(pb, b, full);
            Add(a, a, b, full);
        } else {
            Duplicate(pb, 0);
        }
        ReduceSum(denom, a, full);
        Add(denom, denom, prev, one);
        DeInterleave(packed, unused, pa, pb);
        auto* dst_row = prob + row * 16;
        StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            dst_row, packed, 32, 1, bf16);
        StoreUnAlign(sum_write, denom, sum_out, 1);
    }
    StoreUnAlignPost(sum_write, sum_out, 0);
    uint32_t rows = 32;
    auto live = UpdateMask<float>(rows);
    StoreAlign(max_write, anchor, live);
    CommitMagnitude(guard, magnitude);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

extern "C" [aicore] __attribute__((always_inline)) void csa_fixed_exp(
    uint64_t scores, uint64_t probs, uint64_t sum_read, uint64_t sum_write,
    uint64_t max_write, int32_t count, float scale, uint64_t status) {
    auto* s = reinterpret_cast<__ubuf__ float*>(scores);
    auto* p = reinterpret_cast<__ubuf__ bfloat16_t*>(probs);
    auto* sr = reinterpret_cast<__ubuf__ float*>(sum_read);
    auto* sw = reinterpret_cast<__ubuf__ float*>(sum_write);
    auto* guard = reinterpret_cast<__ubuf__ int32_t*>(status);
    auto* mw = reinterpret_cast<__ubuf__ float*>(max_write);
    if (count > 64) CsaFixedExponentRows<true>(s, p, sr, sw, mw, count, scale, guard);
    else CsaFixedExponentRows<false>(s, p, sr, sw, mw, count, scale, guard);
}

"""


@tla.extern(source=_CSA_VEC1_SOURCE, name="csa_vec1")
def _csa_vec1(
    scores: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    probs: tla.Pointer[tla.BFloat16, tla.AddressSpace.ub],
    max_read: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    sum_read: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    max_write: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    sum_write: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    decay: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    weights: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    count: tla.Int32,
    scale: tla.Float32,
    weighted: tla.Int32,
) -> None: ...


@tla.jit
def _flat(ptr, count):
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(count), tla.make_stride(1))
    )


@tla.jit
def _csa_fixed_init(
    sinks: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    sums: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    maxima: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    has_sinks: tla.Int32,
    guard: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
) -> None:
    with tla.vec.func(mode="simd"):
        all32 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        live32, _ = tla.update_mask(32, dtype=tla.Float32)
        all_i32 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
        live_i32, _ = tla.update_mask(32, dtype=tla.Int32)
        anchor = tla.full(44.36141956, dtype=tla.Float32)
        zero32 = tla.full(0.0, dtype=tla.Float32)
        zero_i32 = tla.full(0, dtype=tla.Int32)
        initial = zero32
        if has_sinks == 1:
            values = _flat(sinks, 64).load()
            bits = tla.bitwise_and(
                values.bitcast(tla.Int32), tla.full(0x7FFFFFFF, dtype=tla.Int32)
            )
            bits = tla.where(live_i32, bits, zero_i32)
            lane_max = bits.reduce(tla.ReductionOp.MAX, mask=all_i32)
            outside = tla.cmp(
                tla.full(lane_max, dtype=tla.Int32),
                tla.full(0x42000000, dtype=tla.Int32),
                mode="gt",
            )
            bad = tla.where(outside, tla.full(1, dtype=tla.Int32), zero_i32)
            prev = _flat(guard, 1).load(
                params=tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                )
            )
            _flat(guard, 1).store(
                tla.max(prev, bad, mask=all_i32),
                params=tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                ),
            )
            initial = tla.where(
                live32,
                tla.exp(tla.sub(values, anchor, mask=all32), mask=all32),
                zero32,
            )
        _flat(sums, 32).store(initial, mask=live32)
        _flat(maxima, 32).store(anchor, mask=live32)
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.extern(source=_CSA_FIXED_VEC1_SOURCE, name="csa_fixed_exp")
def _csa_fixed_exp(
    scores: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    probs: tla.Pointer[tla.BFloat16, tla.AddressSpace.ub],
    sum_read: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    sum_write: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    max_write: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    count: tla.Int32,
    scale: tla.Float32,
    guard: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
) -> None: ...


@tla.jit
def _csa_copy_prob_width(source, target, rows, sub, width: tla.Constexpr[int]):
    shape = tla.make_shape(rows, width)
    src = tla.tile_view(source, shape, tla.make_coord(0, 0))
    dst = tla.tile_view(target, shape, tla.make_coord(sub, 0))
    tla.copy(dst, src)


@tla.jit
def process_vec1_csa(
    sync,
    bufs,
    scale,
    tail_n,
    m,
    rows,
    dtype_in,
    slot_s,
    slot_p,
    slot_rd,
    weights,
    weight_offset,
    is_cmp,
    use_weights: tla.Constexpr[bool] = True,
    cached_initial: tla.Constexpr[bool] = False,
    first_tile=False,
    compact_probabilities: tla.Constexpr[bool] = False,
):
    """Update per-head softmax state and apply optional log multiplicities."""
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        ub_s = _ub_s(bufs, rows, slot_s)
        ub_p = _ub_p(bufs, rows)
        ub_max = _max(bufs, rows, slot_s)
        max_read = _max(bufs, rows, slot_rd).ptr
        ub_sum = _sum(bufs, rows, slot_s)
        sum_read = _sum(bufs, rows, slot_rd).ptr
        if cached_initial and first_tile:
            max_read = bufs.ub_sink
            sum_read = bufs.ub_sum + 2 * rows
        ub_dm = _dm(bufs, rows, slot_s)
        weight_ptr = tla.recast_ptr(bufs.ub_phy_addr, dtype=tla.Float32)
        weight_shape = tla.make_layout(
            tla.make_shape(1, KV_BLOCK), tla.make_stride(KV_BLOCK, 1)
        )
        weight = tla.make_tensor(weight_ptr, weight_shape)
        if use_weights:
            sync.m_sink.lock(pipe=tla.pipes.MTE2)
            if is_cmp:
                tla.copy(
                    weight, tla.make_tensor(weights.ptr + weight_offset, weight_shape)
                )
            sync.m_sink.unlock(pipe=tla.pipes.MTE2)
            sync.m_sink.lock(pipe=tla.pipes.VECTOR)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=1)
        sync.m_p.lock(pipe=tla.pipes.VECTOR)
        _csa_vec1(
            ub_s.ptr,
            ub_p.ptr,
            max_read,
            sum_read,
            ub_max.ptr,
            ub_sum.ptr,
            ub_dm.ptr,
            weight.ptr,
            tail_n,
            scale,
            is_cmp.to(tla.Int32) if use_weights else tla.as_numeric(0),
        )
        if use_weights:
            sync.m_sink.unlock(pipe=tla.pipes.VECTOR)
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=1)
        l1p = l1_p_tensor(bufs.l1_p + slot_p * (Q_ROWS * KV_BLOCK), dtype_in, m)
        sink = tla.tile_view(
            l1p, tla.make_shape(rows, KV_BLOCK), tla.make_coord(sub, 0)
        )
        sync.m_p.unlock(pipe=tla.pipes.VECTOR)
        sync.m_p.lock(pipe=tla.pipes.MTE3)
        tla.set_flag(sync.p_data_ready)
        tla.wait_flag(sync.p_data_ready)
        if compact_probabilities:
            # PV only reads the probability columns covered by its K tile.
            if tail_n <= 16:
                _csa_copy_prob_width(ub_p, l1p, rows, sub, 16)
            elif tail_n <= 32:
                _csa_copy_prob_width(ub_p, l1p, rows, sub, 32)
            elif tail_n <= 64:
                _csa_copy_prob_width(ub_p, l1p, rows, sub, 64)
            else:
                tla.copy(sink, ub_p)
        else:
            tla.copy(sink, ub_p)
        sync.m_p.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=1)


_CSA_PV_SOURCE = r"""
#include "kernel_operator.h"
using namespace AscendC::MicroAPI;

constexpr CastTrait csa_narrow = {RegLayout::ZERO, SatMode::SAT,
                                 MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

template <bool AddPrevious, bool WriteOutput, bool WriteLse>
__simd_vf__ void FinishCsaPv(__ubuf__ float* previous, __ubuf__ float* tile,
                            __ubuf__ bfloat16_t* output, __ubuf__ float* decay,
                            __ubuf__ float* sums, __ubuf__ float* maxima,
                            __ubuf__ float* inverse, __ubuf__ float* lse) {
    constexpr uint16_t rows = 32;
    constexpr uint16_t width = 512;
    auto full = CreateMask<float, MaskPattern::ALL>();
    auto bf16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    if constexpr (WriteOutput) {
        uint32_t length = rows;
        auto active = UpdateMask<float>(length);
        RegTensor<float> divisor, one, inv;
        LoadAlign(divisor, sums);
        Duplicate(one, 1.0f);
        Div(inv, one, divisor, active);
        StoreAlign(inverse, inv, active);
        if constexpr (WriteLse) {
            RegTensor<float> maximum, logsum;
            LoadAlign(maximum, maxima);
            Ln(logsum, divisor, active);
            Add(logsum, logsum, maximum, active);
            StoreAlign(lse, logsum, active);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t row = 0; row < rows; ++row) {
        RegTensor<float> weight, inv;
        if constexpr (AddPrevious) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(weight, decay + row);
        }
        if constexpr (WriteOutput) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(inv, inverse + row);
        }
        #pragma unroll
        for (uint16_t col = 0; col < width; col += 64) {
            RegTensor<float> result;
            uint32_t offset = (uint32_t)row * width + col;
            LoadAlign(result, tile + offset);
            if constexpr (AddPrevious) {
                RegTensor<float> old;
                LoadAlign(old, previous + offset);
                MulAddDst(result, old, weight, full);
            }
            if constexpr (WriteOutput) {
                RegTensor<bfloat16_t> out;
                Mul(result, result, inv, full);
                Cast<bfloat16_t, float, csa_narrow>(out, result, full);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(output + offset, out, bf16);
            } else {
                StoreAlign(previous + offset, result, full);
            }
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

extern "C" [aicore] __attribute__((always_inline)) void csa_finish_pv(
    uint64_t previous, uint64_t current, uint64_t output, uint64_t decay,
    uint64_t sums, uint64_t maxima, uint64_t scratch, int32_t mode) {
    auto* prev = reinterpret_cast<__ubuf__ float*>(previous);
    auto* tile = reinterpret_cast<__ubuf__ float*>(current);
    auto* out = reinterpret_cast<__ubuf__ bfloat16_t*>(output);
    auto* alpha = reinterpret_cast<__ubuf__ float*>(decay);
    auto* denominator = reinterpret_cast<__ubuf__ float*>(sums);
    auto* maximum = reinterpret_cast<__ubuf__ float*>(maxima);
    auto* inverse = reinterpret_cast<__ubuf__ float*>(scratch);
    auto* lse = inverse + 64;
    switch (mode) {
      case 0: FinishCsaPv<true, false, false>(prev, tile, out, alpha, denominator, maximum, inverse, lse); break;
      case 1: FinishCsaPv<true, true, false>(prev, tile, out, alpha, denominator, maximum, inverse, lse); break;
      case 2: FinishCsaPv<true, true, true>(prev, tile, out, alpha, denominator, maximum, inverse, lse); break;
      case 3: FinishCsaPv<false, true, false>(prev, tile, out, alpha, denominator, maximum, inverse, lse); break;
      case 4: FinishCsaPv<false, true, true>(prev, tile, out, alpha, denominator, maximum, inverse, lse); break;
    }
}

template <bool AddPrevious, bool Output>
__simd_vf__ void FinishCsaPvHalf(__ubuf__ float* previous, __ubuf__ float* tile,
                                __ubuf__ bfloat16_t* output, __ubuf__ float* decay,
                                __ubuf__ float* sums, __ubuf__ float* maxima,
                                __ubuf__ float* inv, __ubuf__ float* lse,
                                uint16_t part, bool write_lse) {
    constexpr uint16_t rows = 32;
    constexpr uint16_t width = 256;
    auto all = CreateMask<float, MaskPattern::ALL>();
    auto bf16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    if constexpr (Output) {
        if (part == 0) {
            uint32_t count = rows;
            auto active = UpdateMask<float>(count);
            RegTensor<float> den, one, inverse;
            LoadAlign(den, sums);
            Duplicate(one, 1.0f);
            Div(inverse, one, den, active);
            StoreAlign(inv, inverse, active);
            if (write_lse) {
                RegTensor<float> peak, logsum;
                LoadAlign(peak, maxima);
                Ln(logsum, den, active);
                Add(logsum, logsum, peak, active);
                StoreAlign(lse, logsum, active);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }
    for (uint16_t row = 0; row < rows; ++row) {
        RegTensor<float> weight, inverse;
        if constexpr (AddPrevious) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(weight, decay + row);
        }
        if constexpr (Output) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(inverse, inv + row);
        }
        #pragma unroll
        for (uint16_t col = 0; col < width; col += 64) {
            const uint32_t loc = row * 512 + part * width + col;
            const uint32_t temp = AddPrevious ? row * width + col : loc;
            RegTensor<float> value;
            LoadAlign(value, tile + temp);
            if constexpr (AddPrevious) {
                RegTensor<float> old;
                LoadAlign(old, previous + loc);
                MulAddDst(value, old, weight, all);
            }
            if constexpr (Output) {
                RegTensor<bfloat16_t> narrowed;
                Mul(value, value, inverse, all);
                Cast<bfloat16_t, float, csa_narrow>(narrowed, value, all);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(output + loc, narrowed, bf16);
            } else {
                StoreAlign(previous + loc, value, all);
            }
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

extern "C" [aicore] __attribute__((always_inline)) void csa_finish_pv_half(
    uint64_t previous, uint64_t current, uint64_t output, uint64_t decay, uint64_t sums,
    uint64_t maxima, uint64_t scratch, int32_t part, int32_t mode, int32_t write_lse) {
    auto* prev = reinterpret_cast<__ubuf__ float*>(previous);
    auto* tile = reinterpret_cast<__ubuf__ float*>(current);
    auto* out = reinterpret_cast<__ubuf__ bfloat16_t*>(output);
    auto* alpha = reinterpret_cast<__ubuf__ float*>(decay);
    auto* den = reinterpret_cast<__ubuf__ float*>(sums);
    auto* peak = reinterpret_cast<__ubuf__ float*>(maxima);
    auto* inv = reinterpret_cast<__ubuf__ float*>(scratch);
    if (mode == 0) {
        FinishCsaPvHalf<true, false>(prev, tile, out, alpha, den, peak, inv, inv + 64, part, false);
    } else if (mode == 1) {
        FinishCsaPvHalf<true, true>(prev, tile, out, alpha, den, peak, inv, inv + 64, part, write_lse != 0);
    } else {
        FinishCsaPvHalf<false, true>(prev, tile, out, alpha, den, peak, inv, inv + 64, part, write_lse != 0);
    }
}

"""


@tla.extern(source=_CSA_PV_SOURCE, name="csa_finish_pv")
def _csa_finish_pv(
    running: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    tile: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    output: tla.Pointer[tla.BFloat16, tla.AddressSpace.ub],
    decay: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    sums: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    maxima: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    scratch: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    mode: tla.Int32,
) -> None: ...


@tla.jit
def _csa_pv_rows(
    bufs,
    source,
    rows,
    slot,
    dtype,
    add_previous: tla.Constexpr[bool],
    write_output: tla.Constexpr[bool],
    emit_lse: tla.Constexpr[bool] = False,
):
    mode = ((1 if add_previous else 3) + int(emit_lse)) if write_output else 0
    _csa_finish_pv(
        bufs.ub_o,
        source,
        bufs.ub_o16,
        _dm(bufs, rows, slot).ptr,
        _sum(bufs, rows, slot).ptr,
        _max(bufs, rows, slot).ptr,
        bufs.ub_pv_stats,
        tla.as_numeric(mode),
    )


@tla.jit
def _csa_pv_update(
    sync,
    bufs,
    out,
    lse,
    task,
    m,
    rows,
    slot,
    dtype,
    first_tile,
    write_output,
    bypass_cache,
    emit_lse: tla.Constexpr[bool],
):
    """Consume PV and return its Fixpipe credit before the GM write."""
    with tla.vector():
        output_lock = sync.m_o16
        tla.cross_core_wait_flag(sync.o_ready, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(sync.o_ready, tla.pipes.VECTOR, aiv_id=1)
        if write_output:
            output_lock.lock(pipe=tla.pipes.VECTOR)
            if first_tile:
                _csa_pv_rows(bufs, bufs.ub_o, rows, slot, dtype, False, True, emit_lse)
            else:
                _csa_pv_rows(
                    bufs, bufs.ub_otmp_full, rows, slot, dtype, True, True, emit_lse
                )
            output_lock.unlock(pipe=tla.pipes.VECTOR)
        else:
            if not first_tile:
                _csa_pv_rows(bufs, bufs.ub_otmp_full, rows, slot, dtype, True, False)
        tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=1)
        if write_output:
            sub = tla.arch.sub_block_idx()
            row = task * m + sub * rows
            layout = tla.make_layout_with_tag(
                (rows, HEAD_DIM), dtype, tla.arch.RowMajor
            )
            gm = tla.make_tensor(out.ptr + row * HEAD_DIM, layout)
            ub = tla.make_tensor(bufs.ub_o16, layout)
            output_lock.lock(pipe=tla.pipes.MTE3)
            if bypass_cache:
                tla.copy(gm, ub, cache_policy=tla.params.CachePolicy.DISABLE)
            else:
                tla.copy(gm, ub)
            if emit_lse:
                ld = tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
                gm_lse = tla.make_tensor(lse.ptr + row, ld)
                tla.copy(gm_lse, _stats(bufs.ub_pv_stats + VL, rows))
            output_lock.unlock(pipe=tla.pipes.MTE3)


@tla.extern(source=_CSA_PV_SOURCE, name="csa_finish_pv_half")
def _csa_finish_pv_half(
    previous: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    current: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    out: tla.Pointer[tla.BFloat16, tla.AddressSpace.ub],
    decay: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    sum: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    peak: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    scratch: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    part: tla.Int32,
    mode: tla.Int32,
    write_lse: tla.Int32,
) -> None: ...


# Terminal output is handed to MTE3 by the existing Vector event.
_CSA_PV_TERMINAL_SOURCE = _CSA_PV_SOURCE.replace(
    "    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();\n}\n\n"
    'extern "C" [aicore] __attribute__((always_inline)) void csa_finish_pv_half(',
    "    if (!Output) {\n        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();\n    }\n}\n\n"
    'extern "C" [aicore] __attribute__((always_inline)) void csa_finish_pv_half(',
)


@tla.extern(source=_CSA_PV_TERMINAL_SOURCE, name="csa_finish_pv_half")
def _csa_finish_pv_half_terminal(
    previous: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    current: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    out: tla.Pointer[tla.BFloat16, tla.AddressSpace.ub],
    decay: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    sum: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    peak: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    scratch: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    part: tla.Int32,
    mode: tla.Int32,
    write_lse: tla.Int32,
) -> None: ...


@tla.jit
def _csa_consume_pv_half(
    sync,
    bufs,
    out,
    lse,
    task,
    m,
    rows,
    slot,
    dtype,
    part,
    first,
    last,
    emit_lse,
    free_flag,
    ready_flag,
    stream_output: tla.Constexpr[bool] = False,
    early_lse: tla.Constexpr[bool] = False,
    terminal_output_event: tla.Constexpr[bool] = False,
):
    with tla.vector():
        output_lock = sync.m_o16
        tla.cross_core_wait_flag(ready_flag, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(ready_flag, tla.pipes.VECTOR, aiv_id=1)
        if last or not first:
            current = (
                bufs.ub_o
                if first
                else bufs.ub_otmp_full + part * rows * (HEAD_DIM // 2)
            )
            mode = 2 if first else (1 if last else 0)
            acquire_output = (last and part == 0) if stream_output else last
            if acquire_output:
                output_lock.lock(pipe=tla.pipes.VECTOR)
            if terminal_output_event:
                _csa_finish_pv_half_terminal(
                    bufs.ub_o,
                    current,
                    bufs.ub_o16,
                    _dm(bufs, rows, slot).ptr,
                    _sum(bufs, rows, slot).ptr,
                    _max(bufs, rows, slot).ptr,
                    bufs.ub_pv_stats,
                    part,
                    mode,
                    tla.as_numeric(int(emit_lse)),
                )
            else:
                _csa_finish_pv_half(
                    bufs.ub_o,
                    current,
                    bufs.ub_o16,
                    _dm(bufs, rows, slot).ptr,
                    _sum(bufs, rows, slot).ptr,
                    _max(bufs, rows, slot).ptr,
                    bufs.ub_pv_stats,
                    part,
                    mode,
                    tla.as_numeric(int(emit_lse)),
                )
            release_output = (last and part == 1) if stream_output else last
            if release_output:
                output_lock.unlock(pipe=tla.pipes.VECTOR)
        tla.cross_core_set_flag(free_flag, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(free_flag, tla.pipes.VECTOR, aiv_id=1)
        if stream_output and part == 0 and last:
            # The two Vector results occupy disjoint columns in UB. Start
            # the left copy while the right half is still being computed.
            # MTE3 releases m_o16 after both copies, before the gather stage
            # reuses the buffer.
            left_row = task * m + tla.arch.sub_block_idx() * rows
            tla.set_flag(sync.lse_v_mte3)
            tla.wait_flag(sync.lse_v_mte3)
            if emit_lse and early_lse:
                early_lse_layout = tla.make_layout(
                    tla.make_shape(rows), tla.make_stride(1)
                )
                tla.copy(
                    tla.make_tensor(lse.ptr + left_row, early_lse_layout),
                    _stats(bufs.ub_pv_stats + VL, rows),
                )
            left_layout = tla.make_layout(
                tla.make_shape(rows, HEAD_DIM // 2), tla.make_stride(HEAD_DIM, 1)
            )
            tla.copy(
                tla.make_tensor(out.ptr + left_row * HEAD_DIM, left_layout),
                tla.make_tensor(bufs.ub_o16, left_layout),
                cache_policy=tla.params.CachePolicy.DISABLE,
            )
        if part == 1 and last:
            row = task * m + tla.arch.sub_block_idx() * rows
            if stream_output:
                output_lock.lock(pipe=tla.pipes.MTE3)
                layout = tla.make_layout(
                    tla.make_shape(rows, HEAD_DIM // 2), tla.make_stride(HEAD_DIM, 1)
                )
                tla.copy(
                    tla.make_tensor(out.ptr + row * HEAD_DIM + HEAD_DIM // 2, layout),
                    tla.make_tensor(bufs.ub_o16 + HEAD_DIM // 2, layout),
                    cache_policy=tla.params.CachePolicy.DISABLE,
                )
            else:
                layout = tla.make_layout_with_tag(
                    (rows, HEAD_DIM), dtype, tla.arch.RowMajor
                )
                output_lock.lock(pipe=tla.pipes.MTE3)
                tla.copy(
                    tla.make_tensor(out.ptr + row * HEAD_DIM, layout),
                    tla.make_tensor(bufs.ub_o16, layout),
                    cache_policy=tla.params.CachePolicy.DISABLE,
                )
            if emit_lse and not early_lse:
                ld = tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
                tla.copy(
                    tla.make_tensor(lse.ptr + row, ld),
                    _stats(bufs.ub_pv_stats + VL, rows),
                )
            output_lock.unlock(pipe=tla.pipes.MTE3)


@tla.jit
def process_vec2_seg(sync, bufs, seg, m, rows, dtype_in, fresh=0, slot_dm=0):
    c0 = 0
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        ub_o = tla.make_tensor(
            bufs.ub_o,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        ub_dm = _dm(bufs, rows, slot_dm)
        ub_otmp = tla.make_tensor(
            bufs.ub_otmp_full,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        if seg == 0:
            tla.cross_core_wait_flag(sync.o_ready, tla.pipes.VECTOR, aiv_id=0)
            tla.cross_core_wait_flag(sync.o_ready, tla.pipes.VECTOR, aiv_id=1)
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            if fresh:
                for r in tla.range(rows):
                    for c in tla.range(D_CHUNK // VL):
                        col = seg * (D_CHUNK // VL) + c
                        dst = tla.tile_view(
                            ub_o, tla.make_shape(1, VL), tla.make_coord(r, col)
                        )
                        ot = tla.tile_view(
                            ub_otmp, tla.make_shape(1, VL), tla.make_coord(r, col)
                        )
                        dst.store(ot.load())
            else:
                for r in tla.range(rows):
                    dm = tla.tile_view(
                        ub_dm, tla.make_shape(1), tla.make_coord(r)
                    ).load(params=brc)
                    for c in tla.range(D_CHUNK // VL):
                        col = seg * (D_CHUNK // VL) + c
                        dst = tla.tile_view(
                            ub_o, tla.make_shape(1, VL), tla.make_coord(r, col)
                        )
                        ot = tla.tile_view(
                            ub_otmp, tla.make_shape(1, VL), tla.make_coord(r, col)
                        )
                        acc = dst.load()
                        acc = tla.mul(acc, dm, mask=preg)
                        acc = tla.add(acc, ot.load(), mask=preg)
                        dst.store(acc)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        if seg == 3:
            tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=1)


def _bc_rows8(rows):
    return (rows + 7) // 8 * 8


def _bc_slot_stride(rows):
    return rows * HEAD_DIM + _bc_rows8(rows) * 2


@tla.jit
def bc_reinit_stats(bufs, rows, slot=0):
    with tla.vector():
        ub_max = _max(bufs, rows, slot)
        ub_sum = _sum(bufs, rows, slot)
        with tla.vec.func(mode="simd"):
            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
            zero = tla.full(0.0, dtype=tla.Float32)
            minreg = tla.full(MIN_F, dtype=tla.Float32)
            for r in tla.range(rows):
                tla.tile_view(ub_max, tla.make_shape(1), tla.make_coord(r)).store(
                    minreg, params=tla.params.UnalignStoreParams()
                )
                tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(r)).store(
                    zero, params=tla.params.UnalignStoreParams()
                )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.jit
def bc_div_cur(bufs, rows, slot=0):
    with tla.vector():
        ub_o = tla.make_tensor(
            bufs.ub_o,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        ub_sum = _sum(bufs, rows, slot)
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            for r in tla.range(rows):
                den = tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(r)).load(
                    params=brc
                )
                for c2 in (0, 1, 2, 3, 4, 5, 6, 7):
                    dst = tla.tile_view(
                        ub_o, tla.make_shape(1, VL), tla.make_coord(r, c2)
                    )
                    dst.store(tla.div(dst.load(), den))
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.jit
def bc_stage_cur(sync, bufs, bc_ws, rows, slot=0):
    with tla.vector():
        slot_ws = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
        rows8 = _bc_rows8(rows)
        base = slot_ws * _bc_slot_stride(rows)
        tla.set_flag(sync.bc_v3m)
        tla.wait_flag(sync.bc_v3m)
        gm_o = tla.make_tensor(
            bc_ws.ptr + base,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        tla.copy(
            gm_o,
            tla.make_tensor(
                bufs.ub_o,
                tla.make_layout(
                    tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
                ),
            ),
        )
        gm_max = tla.make_tensor(
            bc_ws.ptr + base + rows * HEAD_DIM,
            tla.make_layout(tla.make_shape(rows8), tla.make_stride(1)),
        )
        tla.copy(gm_max, _max(bufs, rows8, slot))
        gm_sum = tla.make_tensor(
            bc_ws.ptr + base + rows * HEAD_DIM + rows8,
            tla.make_layout(tla.make_shape(rows8), tla.make_stride(1)),
        )
        tla.copy(gm_sum, _sum(bufs, rows8, slot))
        tla.set_flag(sync.bc_m3m2)
        tla.set_flag(sync.bc_m3v)
        tla.wait_flag(sync.bc_m3v)


@tla.jit
def bc_merge_seg(sync, bufs, bc_ws, rows, seg, rows8, base):
    tla.wait_flag(sync.bc_v2m)
    gm_seg = tla.make_tensor(
        bc_ws.ptr + base + seg * D_CHUNK,
        tla.make_layout(tla.make_shape(rows, D_CHUNK), tla.make_stride(HEAD_DIM, 1)),
    )
    tla.copy(
        tla.make_tensor(
            bufs.ub_bc_prev,
            tla.make_layout(tla.make_shape(rows, D_CHUNK), tla.make_stride(D_CHUNK, 1)),
        ),
        gm_seg,
    )
    tla.set_flag(sync.bc_m2v)
    tla.wait_flag(sync.bc_m2v)
    with tla.vec.func(mode="simd"):
        preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        brc = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
        ub_o = tla.make_tensor(
            bufs.ub_o,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        ub_bc_prev = tla.make_tensor(
            bufs.ub_bc_prev,
            tla.make_layout(tla.make_shape(rows, D_CHUNK), tla.make_stride(D_CHUNK, 1)),
        )
        ub_bc_w = _stats(bufs.ub_bc_w, rows8 * 2)
        for r in tla.range(rows):
            w_p = tla.tile_view(ub_bc_w, tla.make_shape(1), tla.make_coord(r)).load(
                params=brc
            )
            w_c = tla.tile_view(
                ub_bc_w, tla.make_shape(1), tla.make_coord(rows8 + r)
            ).load(params=brc)
            for c2 in (0, 1):
                col = seg * (D_CHUNK // VL) + c2
                dst = tla.tile_view(ub_o, tla.make_shape(1, VL), tla.make_coord(r, col))
                prev = tla.tile_view(
                    ub_bc_prev, tla.make_shape(1, VL), tla.make_coord(r, c2)
                ).load()
                cur = dst.load()
                part = tla.mul(w_p, prev, mask=preg)
                dst.store(
                    tla.add(part, tla.mul(w_c, cur, mask=preg), mask=preg),
                    mask=preg,
                )
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    tla.set_flag(sync.bc_v2m)


@tla.jit
def bc_merge_prev(sync, bufs, bc_ws, rows, slot=0):
    with tla.vector():
        slot_ws = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
        rows8 = _bc_rows8(rows)
        base = slot_ws * _bc_slot_stride(rows)
        ub_bc_max = _stats(bufs.ub_bc_max, rows8)
        ub_bc_sum = _stats(bufs.ub_bc_sum, rows8)
        ub_max = _max(bufs, rows, slot)
        ub_sum = _sum(bufs, rows, slot)
        tla.wait_flag(sync.bc_m3m2)
        gm_max = tla.make_tensor(
            bc_ws.ptr + base + rows * HEAD_DIM,
            tla.make_layout(tla.make_shape(rows8), tla.make_stride(1)),
        )
        gm_sum = tla.make_tensor(
            bc_ws.ptr + base + rows * HEAD_DIM + rows8,
            tla.make_layout(tla.make_shape(rows8), tla.make_stride(1)),
        )
        tla.copy(ub_bc_max, gm_max)
        tla.copy(ub_bc_sum, gm_sum)
        tla.set_flag(sync.bc_m2v)
        tla.wait_flag(sync.bc_m2v)
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
            one = tla.full(1.0, dtype=tla.Float32)
            ub_bc_w = _stats(bufs.ub_bc_w, rows8 * 2)
            for r in tla.range(rows):
                lse_p = tla.tile_view(
                    ub_bc_max, tla.make_shape(1), tla.make_coord(r)
                ).load(params=brc)
                s_p = tla.tile_view(
                    ub_bc_sum, tla.make_shape(1), tla.make_coord(r)
                ).load(params=brc)
                lse_c = tla.tile_view(
                    ub_max, tla.make_shape(1), tla.make_coord(r)
                ).load(params=brc)
                s_c = tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(r)).load(
                    params=brc
                )
                m_new = tla.max(lse_p, lse_c, mask=preg)
                e_p = tla.exp(tla.sub(lse_p, m_new, mask=preg), mask=preg)
                e_c = tla.exp(tla.sub(lse_c, m_new, mask=preg), mask=preg)
                num_p = tla.mul(s_p, e_p, mask=preg)
                num_c = tla.mul(s_c, e_c, mask=preg)
                z_sum = tla.add(num_p, num_c, mask=preg)
                w_p = tla.div(num_p, z_sum, mask=preg)
                w_c = tla.div(num_c, z_sum, mask=preg)
                merged_lse = tla.add(tla.log(z_sum, mask=preg), m_new, mask=preg)
                tla.tile_view(ub_max, tla.make_shape(1), tla.make_coord(r)).store(
                    merged_lse, params=tla.params.UnalignStoreParams()
                )
                tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(r)).store(
                    one, params=tla.params.UnalignStoreParams()
                )
                tla.tile_view(ub_bc_w, tla.make_shape(1), tla.make_coord(r)).store(
                    w_p, params=tla.params.UnalignStoreParams()
                )
                tla.tile_view(
                    ub_bc_w, tla.make_shape(1), tla.make_coord(rows8 + r)
                ).store(w_c, params=tla.params.UnalignStoreParams())
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        for seg in (0, 1, 2, 3):
            bc_merge_seg(sync, bufs, bc_ws, rows, seg, rows8, base)


@tla.jit
def bmm2_data_copy_out(
    sync,
    bufs,
    out_tensor,
    task,
    m,
    rows,
    dtype_in,
    row_base=None,
    valid_rows=None,
    odd_mode=0,
    slot_sum=0,
    fresh_from_cube=False,
):
    c0 = 0
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        if fresh_from_cube:
            tla.cross_core_wait_flag(sync.o_ready, tla.pipes.VECTOR, aiv_id=0)
            tla.cross_core_wait_flag(sync.o_ready, tla.pipes.VECTOR, aiv_id=1)
        ub_o = tla.make_tensor(
            bufs.ub_otmp_full if fresh_from_cube else bufs.ub_o,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        ub_o16 = tla.make_tensor(
            bufs.ub_o16,
            tla.make_layout_with_tag((rows, HEAD_DIM), dtype_in, tla.arch.RowMajor),
        )
        ub_sum = _sum(bufs, rows, slot_sum)
        sync.m_o16.lock(pipe=tla.pipes.VECTOR)
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            preg_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype_in)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            cast_zero = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            pack_store = tla.params.NormalStoreParams(
                store_dist=tla.params.StoreDist.DIST_PACK_B32
            )
            for rf in tla.range(rows):
                denf = tla.tile_view(
                    ub_sum, tla.make_shape(1), tla.make_coord(rf)
                ).load(params=brc)
                for cf in tla.range(HEAD_DIM // VL):
                    srcf = tla.tile_view(
                        ub_o, tla.make_shape(1, VL), tla.make_coord(rf, cf)
                    )
                    d16f = tla.tile_view(
                        ub_o16, tla.make_shape(1, VL), tla.make_coord(rf, cf)
                    )
                    d16f.store(
                        tla.div(srcf.load(), denf, mask=preg).to(
                            dtype_in, cast_zero, mask=preg
                        ),
                        pack_store,
                        mask=preg_b16,
                    )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        sync.m_o16.unlock(pipe=tla.pipes.VECTOR)
        if fresh_from_cube:
            tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=1)
        sync.m_o16.lock(pipe=tla.pipes.MTE3)
        if odd_mode == 0:
            if sub * rows < m:
                gm_o = tla.make_tensor(
                    out_tensor.ptr + task * (m * HEAD_DIM) + (sub * rows * HEAD_DIM),
                    tla.make_layout_with_tag(
                        (rows, HEAD_DIM), dtype_in, tla.arch.RowMajor
                    ),
                )
                if fresh_from_cube:
                    tla.copy(gm_o, ub_o16, cache_policy=tla.params.CachePolicy.DISABLE)
                else:
                    tla.copy(gm_o, ub_o16)
        else:
            #  attentionOutOffset + subBlockIdx*firstHalfMRealSize*dSizeV）。
            n_rows = valid_rows - sub * rows
            if n_rows > rows:
                n_rows = tla.as_numeric(rows)
            if n_rows < 0:
                n_rows = tla.as_numeric(0)
            for r in tla.range(n_rows):
                gm_r = tla.make_tensor(
                    out_tensor.ptr + (row_base + sub * rows + r) * HEAD_DIM,
                    tla.make_layout_with_tag(
                        (1, HEAD_DIM), dtype_in, tla.arch.RowMajor
                    ),
                )
                src_r = tla.tile_view(
                    ub_o16, tla.make_shape(1, HEAD_DIM), tla.make_coord(r, c0)
                )
                tla.copy(gm_r, src_r)
        sync.m_o16.unlock(pipe=tla.pipes.MTE3)


@tla.jit
def lse_copy_out(
    sync, bufs, lse_tensor, task, m, rows, slot=0, row_base=None, valid_rows=None
):
    c0 = 0
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        ub_max = _max(bufs, rows, slot)
        ub_sum = _sum(bufs, rows, slot)
        ub_lse = _dm(bufs, rows, slot)
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            for r in tla.range(rows):
                s = tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(r)).load(
                    params=brc
                )
                mx = tla.tile_view(ub_max, tla.make_shape(1), tla.make_coord(r)).load(
                    params=brc
                )
                val = tla.add(tla.log(s, mask=preg), mx, mask=preg)
                tla.tile_view(ub_lse, tla.make_shape(1), tla.make_coord(r)).store(
                    val, params=tla.params.UnalignStoreParams()
                )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        tla.set_flag(sync.lse_v_mte3)
        tla.wait_flag(sync.lse_v_mte3)
        base = task * m if row_base is None else row_base
        if valid_rows is None and m % 2 == 0:
            gm_lse = tla.make_tensor(
                lse_tensor.ptr + base + sub * rows,
                tla.make_layout(tla.make_shape(rows), tla.make_stride(1)),
            )
            tla.copy(gm_lse, ub_lse)
        else:
            n_rows = (m if valid_rows is None else valid_rows) - sub * rows
            if n_rows > rows:
                n_rows = tla.as_numeric(rows)
            if n_rows < 0:
                n_rows = tla.as_numeric(0)
            for r in tla.range(n_rows):
                gm_r = tla.make_tensor(
                    lse_tensor.ptr + base + sub * rows + r,
                    tla.make_layout(tla.make_shape(1), tla.make_stride(1)),
                )
                src_r = tla.tile_view(ub_lse, tla.make_shape(1), tla.make_coord(r))
                tla.copy(gm_r, src_r)
        tla.set_flag(sync.lse_mte3_v)
        tla.wait_flag(sync.lse_mte3_v)


PHY_ELEMS_PER_LOOP = 128
PHY_ELEMS_PER_REG = 64
PHY_ADDR_PER_LOOP = 256
PHY_ADDR_PER_REG = 128
PHY_INVALID = -1


# `SmlaCsaBlockVector::CopyInKvSparse`（sparse_flash_mla_csa_block_vector_arch35.h:495）：
_PAIR_ELEM_BYTES = 2
_SPARSE_BLOCK_SIZE = 1


class TraceFidelityError(RuntimeError):
    pass


# "dynamic Tla if requires variables used after the branch..."。
_TRACE_OWNER = None
_TRACE_GSPLIT = 0
_TRACE_PAIRED = False
_TRACE_STAGE = False
_TRACE_QUARTER = False
_TRACE_SLOTS = 0
_TRACE_EMITS = 0


def configure_trace(g_split, share_stage, share_quarter, kv_slots, owner):
    global _TRACE_OWNER, _TRACE_GSPLIT, _TRACE_PAIRED
    global _TRACE_STAGE, _TRACE_QUARTER, _TRACE_SLOTS, _TRACE_EMITS
    _TRACE_OWNER = None
    _TRACE_GSPLIT = 0
    _TRACE_PAIRED = False
    _TRACE_STAGE = False
    _TRACE_QUARTER = False
    _TRACE_SLOTS = 0
    _TRACE_EMITS = 0
    if g_split not in (1, 2):
        raise TraceFidelityError(
            f"[{owner}] refusing to compile: g_split={g_split}, only 1/2 pairing semantics exist"
        )
    paired = g_split == 2
    if not paired and (share_stage or share_quarter):
        raise TraceFidelityError(
            f"[{owner}] refusing to compile: shared-slot arm requested with g_split={g_split} "
            f"(share_stage={share_stage}, share_quarter={share_quarter}). "
            "Shared staging requires two neighbouring cores to cover one token: with "
            "g_split=1 each core writes only a quarter of the rows and the slot ring "
            "aliases across cores. Implement balancing and a slot ring first."
        )
    if share_quarter and not share_stage:
        raise TraceFidelityError(
            f"[{owner}] refusing to compile: quarter split needs shared staging "
            "(share_stage=False)"
        )
    _TRACE_OWNER = owner
    _TRACE_GSPLIT = g_split
    _TRACE_PAIRED = paired
    _TRACE_STAGE = bool(share_stage) and paired
    _TRACE_QUARTER = bool(share_quarter) and paired
    _TRACE_SLOTS = kv_slots * 2 if _TRACE_STAGE else kv_slots


def require_configured(site):
    if _TRACE_OWNER is None:
        raise TraceFidelityError(
            f"{site} caller did not call configure_trace: a kernel sharing this "
            "gather must declare the pairing intent at the body head"
        )


def require_share(site):
    global _TRACE_EMITS
    if not (_TRACE_PAIRED and _TRACE_STAGE):
        raise TraceFidelityError(
            f"[{_TRACE_OWNER or 'unset'}] {site}: shared-slot path traced on a "
            f"non-paired / disabled instance (paired={_TRACE_PAIRED}, stage={_TRACE_STAGE})"
        )
    _TRACE_EMITS += 1


def verify_trace_closed(owner):
    if _TRACE_OWNER != owner:
        raise TraceFidelityError(
            f"[{owner}] configure_trace was not called (current owner={_TRACE_OWNER})"
        )
    if _TRACE_STAGE and _TRACE_EMITS == 0:
        raise TraceFidelityError(
            f"[{owner}] shared-slot arm enabled but the trace never reached the gather"
        )


_PAIR_SOURCE = r"""
#include <cstdint>
#include "kernel_operator.h"

extern "C" {

[aicore] __attribute__((always_inline)) void tla_pair_gm_to_ub(
    uint64_t src_gm_addr, uint64_t dst_ub_addr, int64_t block_len_bytes,
    int64_t src_stride_bytes, int64_t dst_byte_off, int64_t block_count) {
  AscendC::DataCopyExtParams intri;
  intri.blockCount = static_cast<uint32_t>(block_count);
  intri.blockLen = static_cast<uint32_t>(block_len_bytes);
  intri.dstStride = 0;
  intri.srcStride = src_stride_bytes;
  AscendC::DataCopyPadExtParams<uint8_t> padParams;
  padParams.isPad = true;
  padParams.leftPadding = 0;
  padParams.rightPadding = 0;
  padParams.paddingValue = 0;
  AscendC::LocalTensor<uint8_t> dst(
      AscendC::TPosition::VECCALC,
      static_cast<uint32_t>(dst_ub_addr) + static_cast<uint32_t>(dst_byte_off),
      static_cast<uint32_t>(block_len_bytes));
  AscendC::GlobalTensor<uint8_t> src;
  src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(src_gm_addr),
                      static_cast<uint32_t>(block_len_bytes));
  AscendC::DataCopyPad(dst, src, intri, padParams);
}

}
"""


@tla.jit
def pair_gm_to_ub(
    src: tla.Pointer[tla.UInt8, tla.AddressSpace.gm],
    dst_base: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    block_len_bytes: tla.Int64,
    src_stride_bytes: tla.Int64,
    dst_byte_off: tla.Int64,
    block_count: tla.Int64,
) -> None:
    _copy_byte_rows(
        dst_base + dst_byte_off,
        src,
        block_count,
        block_len_bytes,
        block_len_bytes + src_stride_bytes,
    )


@tla.jit
def copy_sparse_idx_to_ub(ub_idx, idx_tensor, gm_base, ncols):
    for j in tla.range(ncols):
        ub_idx[j] = idx_tensor[gm_base + j]


@tla.jit
def copy_pa_table_to_ub(ub_bt, bt_tensor, bt_base, bt_cols):
    for j in tla.range(bt_cols):
        ub_bt[j] = bt_tensor[bt_base + j]


@tla.jit
def copy_phy_addr_to_gm(ub_addr, phy_table, table_off, ncols):
    n_i32 = ((ncols + PHY_ELEMS_PER_LOOP - 1) // PHY_ELEMS_PER_LOOP) * PHY_ADDR_PER_LOOP
    gm_i32 = tla.make_tensor(
        tla.recast_ptr(phy_table.ptr + table_off, dtype=tla.Int32),
        tla.make_layout(tla.make_shape(n_i32), tla.make_stride(1)),
    )
    tla.copy(
        gm_i32,
        tla.tile_view(ub_addr, tla.make_shape(n_i32), tla.make_coord(0)),
    )


@tla.jit
def gather_block_page(ub_bt, b, m, n_win):
    zero = tla.full(0, dtype=tla.Int32)
    top = tla.full(PHY_ELEMS_PER_REG - 1, dtype=tla.Int32)
    g = tla.full(0, dtype=tla.Int32)
    for w in static_range(n_win):
        lo = w * PHY_ELEMS_PER_REG
        lo_v = tla.full(lo, dtype=tla.Int32)
        bw = tla.sub(b, lo_v)
        bw = tla.where(tla.cmp(bw, zero, "lt"), zero, bw)
        bw = tla.where(tla.cmp(bw, top, "gt"), top, bw)
        win = tla.tile_view(ub_bt, tla.make_shape(PHY_ELEMS_PER_REG), tla.make_coord(w))
        gw = tla.gather(win, bw, mask=m)
        g = tla.where(tla.cmp(b, lo_v, "ge"), gw, g)
    return g


@tla.jit
def get_kv_phy_addr_vf_pa_impl(
    ub_idx,
    ub_bt,
    ub_addr,
    ncols,
    n_chunks,
    shift_right_num,
    pa_block_size,
    kv_dim,
    kv_stride,
    bt_max,
):
    with tla.vec.func(mode="simd"):
        zero = tla.full(0, dtype=tla.Int32)
        minus_one = tla.full(PHY_INVALID, dtype=tla.Int32)
        n_win = (bt_max + PHY_ELEMS_PER_REG - 1) // PHY_ELEMS_PER_REG
        for c in tla.range(n_chunks):
            rem = ncols - c * PHY_ELEMS_PER_LOOP
            n0 = rem
            if n0 > PHY_ELEMS_PER_REG:
                n0 = tla.as_numeric(PHY_ELEMS_PER_REG)
            if n0 < 0:
                n0 = tla.as_numeric(0)
            n1 = rem - PHY_ELEMS_PER_REG
            if n1 < 0:
                n1 = tla.as_numeric(0)
            m0, _ = tla.update_mask(n0, dtype=tla.Int32)
            m1, _ = tla.update_mask(n1, dtype=tla.Int32)
            # c*4+{0,1,2,3}（PHY_ADDR_PER_LOOP/PHY_ELEMS_PER_REG = 4）。
            addr_tile = c * (PHY_ADDR_PER_LOOP // PHY_ELEMS_PER_REG)
            s0 = tla.tile_view(
                ub_idx,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(c * (PHY_ELEMS_PER_LOOP // PHY_ELEMS_PER_REG)),
            ).load()
            s0 = tla.where(m0, s0, minus_one)
            s0 = tla.where(tla.cmp(s0, zero, "lt"), zero, s0)
            b0 = tla.shift_right(s0, shift_right_num)  # sidx >> log2(bs)
            t0 = tla.sub(s0, tla.mul(b0, pa_block_size))  # toff = sidx-boff
            p0 = tla.mul(t0, kv_dim)  # poff = toff*kvDim
            g0 = gather_block_page(ub_bt, b0, m0, n_win)  # DataCopyGather
            g0 = tla.where(tla.cmp(g0, zero, "lt"), zero, g0)
            a0 = tla.add(tla.mul(g0, kv_stride), p0)  # phys*kvStride+poff
            a0 = tla.where(m0, a0, minus_one)
            h0 = tla.where(m0, zero, minus_one)
            lo0, hi0 = tla.interleave(a0, h0)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile),
            ).store(lo0)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 1),
            ).store(hi0)
            s1 = tla.tile_view(
                ub_idx,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(c * (PHY_ELEMS_PER_LOOP // PHY_ELEMS_PER_REG) + 1),
            ).load()
            s1 = tla.where(m1, s1, minus_one)
            s1 = tla.where(tla.cmp(s1, zero, "lt"), zero, s1)
            b1 = tla.shift_right(s1, shift_right_num)
            t1 = tla.sub(s1, tla.mul(b1, pa_block_size))
            p1 = tla.mul(t1, kv_dim)
            g1 = gather_block_page(ub_bt, b1, m1, n_win)  # DataCopyGather
            g1 = tla.where(tla.cmp(g1, zero, "lt"), zero, g1)
            a1 = tla.add(tla.mul(g1, kv_stride), p1)  # phys*kvStride+poff
            a1 = tla.where(m1, a1, minus_one)
            h1 = tla.where(m1, zero, minus_one)
            lo1, hi1 = tla.interleave(a1, h1)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 2),
            ).store(lo1)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 3),
            ).store(hi1)


@tla.jit
def get_kv_phy_addr_vf_pa(
    ub_idx, ub_bt, ub_addr, ncols, n_chunks, pa_block_size, kv_dim, kv_stride, bt_max
):
    shift_right_num = int(pa_block_size).bit_length() - 1
    get_kv_phy_addr_vf_pa_impl(
        ub_idx,
        ub_bt,
        ub_addr,
        ncols,
        n_chunks,
        shift_right_num,
        pa_block_size,
        kv_dim,
        kv_stride,
        bt_max,
    )


@tla.jit
def get_kv_phy_addr_vf_tnd_impl(ub_idx, ub_addr, ncols, n_chunks, kv_prefix, kv_dim):
    with tla.vec.func(mode="simd"):
        zero = tla.full(0, dtype=tla.Int32)
        minus_one = tla.full(PHY_INVALID, dtype=tla.Int32)
        for c in tla.range(n_chunks):
            rem = ncols - c * PHY_ELEMS_PER_LOOP
            n0 = rem
            if n0 > PHY_ELEMS_PER_REG:
                n0 = tla.as_numeric(PHY_ELEMS_PER_REG)
            if n0 < 0:
                n0 = tla.as_numeric(0)
            n1 = rem - PHY_ELEMS_PER_REG
            if n1 < 0:
                n1 = tla.as_numeric(0)
            m0, _ = tla.update_mask(n0, dtype=tla.Int32)
            m1, _ = tla.update_mask(n1, dtype=tla.Int32)
            addr_tile = c * (PHY_ADDR_PER_LOOP // PHY_ELEMS_PER_REG)
            # ---- reg0 ----
            s0 = tla.tile_view(
                ub_idx,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(c * (PHY_ELEMS_PER_LOOP // PHY_ELEMS_PER_REG)),
            ).load()
            s0 = tla.where(m0, s0, minus_one)
            s0 = tla.where(tla.cmp(s0, zero, "lt"), zero, s0)
            a0 = tla.mul(tla.add(s0, kv_prefix), kv_dim)
            h0 = tla.where(m0, zero, minus_one)
            lo0, hi0 = tla.interleave(a0, h0)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile),
            ).store(lo0)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 1),
            ).store(hi0)
            # ---- reg1 ----
            s1 = tla.tile_view(
                ub_idx,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(c * (PHY_ELEMS_PER_LOOP // PHY_ELEMS_PER_REG) + 1),
            ).load()
            s1 = tla.where(m1, s1, minus_one)
            s1 = tla.where(tla.cmp(s1, zero, "lt"), zero, s1)
            a1 = tla.mul(tla.add(s1, kv_prefix), kv_dim)
            h1 = tla.where(m1, zero, minus_one)
            lo1, hi1 = tla.interleave(a1, h1)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 2),
            ).store(lo1)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 3),
            ).store(hi1)


@tla.jit
def get_kv_phy_addr_vf_tnd(ub_idx, ub_addr, ncols, n_chunks, kv_prefix, kv_dim):
    get_kv_phy_addr_vf_tnd_impl(ub_idx, ub_addr, ncols, n_chunks, kv_prefix, kv_dim)


@tla.jit
def get_kv_phy_addr_vf_bsnd_impl(ub_idx, ub_addr, ncols, n_chunks, b_s2_base, kv_dim):
    with tla.vec.func(mode="simd"):
        zero = tla.full(0, dtype=tla.Int32)
        minus_one = tla.full(PHY_INVALID, dtype=tla.Int32)
        for c in tla.range(n_chunks):
            rem = ncols - c * PHY_ELEMS_PER_LOOP
            n0 = rem
            if n0 > PHY_ELEMS_PER_REG:
                n0 = tla.as_numeric(PHY_ELEMS_PER_REG)
            if n0 < 0:
                n0 = tla.as_numeric(0)
            n1 = rem - PHY_ELEMS_PER_REG
            if n1 < 0:
                n1 = tla.as_numeric(0)
            m0, _ = tla.update_mask(n0, dtype=tla.Int32)
            m1, _ = tla.update_mask(n1, dtype=tla.Int32)
            addr_tile = c * (PHY_ADDR_PER_LOOP // PHY_ELEMS_PER_REG)
            # ---- reg0 ----
            s0 = tla.tile_view(
                ub_idx,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(c * (PHY_ELEMS_PER_LOOP // PHY_ELEMS_PER_REG)),
            ).load()
            s0 = tla.where(m0, s0, minus_one)
            s0 = tla.where(tla.cmp(s0, zero, "lt"), zero, s0)
            a0 = tla.add(tla.mul(s0, kv_dim), b_s2_base)
            h0 = tla.where(m0, zero, minus_one)
            lo0, hi0 = tla.interleave(a0, h0)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile),
            ).store(lo0)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 1),
            ).store(hi0)
            # ---- reg1 ----
            s1 = tla.tile_view(
                ub_idx,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(c * (PHY_ELEMS_PER_LOOP // PHY_ELEMS_PER_REG) + 1),
            ).load()
            s1 = tla.where(m1, s1, minus_one)
            s1 = tla.where(tla.cmp(s1, zero, "lt"), zero, s1)
            a1 = tla.add(tla.mul(s1, kv_dim), b_s2_base)
            h1 = tla.where(m1, zero, minus_one)
            lo1, hi1 = tla.interleave(a1, h1)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 2),
            ).store(lo1)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 3),
            ).store(hi1)


@tla.jit
def get_kv_phy_addr_vf_bsnd(ub_idx, ub_addr, ncols, n_chunks, kv_prefix, kv_dim):
    b_s2_base = kv_prefix * kv_dim
    get_kv_phy_addr_vf_bsnd_impl(ub_idx, ub_addr, ncols, n_chunks, b_s2_base, kv_dim)


@tla.jit
def get_kv_phy_addr_for_kv_type(
    sync,
    bufs,
    phy_table,
    idx,
    bt,
    table_row,
    idx_row_width,
    ncols,
    aligned_cols,
    kv_layout,
    bt_seq,
    bt_max,
    pa_block_size,
    pa_stride,
    kv_prefix,
):
    with tla.vector():
        tla.wait_flag(sync.phy_vs)
        ub_idx_t = tla.make_tensor(
            bufs.ub_phy_idx,
            tla.make_layout(tla.make_shape(aligned_cols), tla.make_stride(1)),
        )
        ub_addr_t = tla.make_tensor(
            bufs.ub_phy_addr,
            tla.make_layout(tla.make_shape(aligned_cols * 2), tla.make_stride(1)),
        )
        bt_view = (
            ((bt_max + PHY_ELEMS_PER_REG - 1) // PHY_ELEMS_PER_REG) * PHY_ELEMS_PER_REG
            if bt_max > 0
            else PHY_ELEMS_PER_REG
        )
        ub_bt_t = tla.make_tensor(
            bufs.ub_phy_bt,
            tla.make_layout(tla.make_shape(bt_view), tla.make_stride(1)),
        )
        #    （OT CopySparseIdxToUb）
        if kv_layout == 2:
            copy_pa_table_to_ub(ub_bt_t, bt, bt_seq * bt_max, bt_max)
        copy_sparse_idx_to_ub(ub_idx_t, idx, table_row * idx_row_width, ncols)
        tla.set_flag(sync.phy_s2v)
        tla.set_flag(sync.phy_m2v)
        tla.wait_flag(sync.phy_m2v)
        tla.wait_flag(sync.phy_s2v)
        n_chunks = (ncols + PHY_ELEMS_PER_LOOP - 1) // PHY_ELEMS_PER_LOOP
        if kv_layout == 2:
            get_kv_phy_addr_vf_pa(
                ub_idx_t,
                ub_bt_t,
                ub_addr_t,
                ncols,
                n_chunks,
                pa_block_size,
                HEAD_DIM,
                pa_stride,
                bt_max,
            )
        elif kv_layout == 1:
            get_kv_phy_addr_vf_tnd(
                ub_idx_t,
                ub_addr_t,
                ncols,
                n_chunks,
                kv_prefix,
                HEAD_DIM,
            )
        else:
            get_kv_phy_addr_vf_bsnd(
                ub_idx_t,
                ub_addr_t,
                ncols,
                n_chunks,
                kv_prefix,
                HEAD_DIM,
            )
        tla.set_flag(sync.phy_v_mte3)
        tla.set_flag(sync.phy_vs)
        tla.wait_flag(sync.phy_v_mte3)
        copy_phy_addr_to_gm(ub_addr_t, phy_table, table_row * aligned_cols, ncols)
        tla.set_flag(sync.phy_m3v)
        tla.set_flag(sync.phy_m3s)
        tla.wait_flag(sync.phy_m3v)
        tla.wait_flag(sync.phy_m3s)


@tla.jit
def get_kv_phy_addr(
    sync,
    bufs,
    phy_table,
    idx,
    bt,
    table_row,
    idx_row_width,
    ncols,
    aligned_cols,
    kv_layout,
    bt_seq,
    bt_max,
    pa_block_size,
    pa_stride,
    kv_prefix,
):
    get_kv_phy_addr_for_kv_type(
        sync,
        bufs,
        phy_table,
        idx,
        bt,
        table_row,
        idx_row_width,
        ncols,
        aligned_cols,
        kv_layout,
        bt_seq,
        bt_max,
        pa_block_size,
        pa_stride,
        kv_prefix,
    )


@tla.jit
def gather_kv_rows(
    sync,
    bufs,
    ws,
    kv_tensor,
    idx,
    token_row,
    width,
    base_line,
    n_rows,
    row_base,
    slot,
    slot_stride,
    ws_base=0,
    announce: bool = True,
    block_table=None,
    bt_seq=0,
    bt_max=0,
    pa_block_size=0,
    pa_stride=0,
    topk_value_mode=1,
):
    # dynamic Tla control flow"）。
    require_configured("gather_kv_rows")
    if _TRACE_STAGE:
        require_share("gather_kv_rows")
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        first = (n_rows + 1) // 2
        if _TRACE_QUARTER:
            par = tla.arch.block_idx() % 2
            h0 = (first + 1) // 2
            h1 = first - h0
            second = n_rows - first
            s0 = (second + 1) // 2
            s1 = second - s0
            my_start = 0
            my_rows = h0
            if sub == 1:
                my_start = h0
                my_rows = h1
            if par == 1:
                my_start = first
                my_rows = s0
                if sub == 1:
                    my_start = first + s0
                    my_rows = s1
        else:
            my_start = 0
            my_rows = first
            if sub == 1:
                my_start = first
                my_rows = n_rows - first
        n_chunks = (my_rows + KV_UNIT - 1) // KV_UNIT
        for c in tla.range(n_chunks):
            line0 = my_start + c * KV_UNIT
            rows_out = my_rows - c * KV_UNIT
            if rows_out > KV_UNIT:
                rows_out = tla.as_numeric(KV_UNIT)
            if rows_out < 0:
                rows_out = tla.as_numeric(0)
            if c % 2 == 0:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv0,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_mte3_mte2_0)
                for i in static_range(KV_UNIT):
                    line = line0 + i
                    if line > n_rows - 1:
                        line = n_rows - 1
                    if line < 0:
                        line = 0
                    src = idx[token_row * width + base_line + line]
                    if src < 0:
                        src = 0
                    row_off = (row_base + src) * HEAD_DIM
                    if pa_block_size > 0:
                        blk = src // pa_block_size
                        off = src - blk * pa_block_size
                        # OT GetkeyOffset（csa_block_vector_arch35.h:441-447）：
                        #   if constexpr (TOPK_VALUE_MODE == TOPK_INDEX_MODE)
                        #       physBlockIdx = blockTableGm.GetValue(boIdx*maxBlockNumPerBatch
                        #                                            + blkTableIdx);
                        phys = blk
                        if topk_value_mode == 1:
                            phys = block_table[bt_seq * bt_max + blk]
                            if phys < 0:
                                phys = 0
                        row_off = phys * pa_stride + off * HEAD_DIM
                    else:
                        row_off = (row_base + src) * HEAD_DIM
                    gm_row = tla.make_tensor(
                        kv_tensor.ptr + row_off,
                        tla.make_layout(
                            tla.make_shape(1, HEAD_DIM),
                            tla.make_stride(HEAD_DIM, 1),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    )
                    tla.copy(
                        tla.tile_view(
                            ub_kv, tla.make_shape(1, HEAD_DIM), tla.make_coord(i, 0)
                        ),
                        gm_row,
                    )
                tla.set_flag(sync.kv_mte2_mte3_0)
                tla.wait_flag(sync.kv_mte2_mte3_0)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_0)
            else:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv1,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_mte3_mte2_1)
                for i in static_range(KV_UNIT):
                    line = line0 + i
                    if line > n_rows - 1:
                        line = n_rows - 1
                    if line < 0:
                        line = 0
                    src = idx[token_row * width + base_line + line]
                    if src < 0:
                        src = 0
                    row_off = (row_base + src) * HEAD_DIM
                    if pa_block_size > 0:
                        blk = src // pa_block_size
                        off = src - blk * pa_block_size
                        # OT GetkeyOffset（csa_block_vector_arch35.h:441-447）：
                        #   if constexpr (TOPK_VALUE_MODE == TOPK_INDEX_MODE)
                        #       physBlockIdx = blockTableGm.GetValue(boIdx*maxBlockNumPerBatch
                        #                                            + blkTableIdx);
                        phys = blk
                        if topk_value_mode == 1:
                            phys = block_table[bt_seq * bt_max + blk]
                            if phys < 0:
                                phys = 0
                        row_off = phys * pa_stride + off * HEAD_DIM
                    else:
                        row_off = (row_base + src) * HEAD_DIM
                    gm_row = tla.make_tensor(
                        kv_tensor.ptr + row_off,
                        tla.make_layout(
                            tla.make_shape(1, HEAD_DIM),
                            tla.make_stride(HEAD_DIM, 1),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    )
                    tla.copy(
                        tla.tile_view(
                            ub_kv, tla.make_shape(1, HEAD_DIM), tla.make_coord(i, 0)
                        ),
                        gm_row,
                    )
                tla.set_flag(sync.kv_mte2_mte3_1)
                tla.wait_flag(sync.kv_mte2_mte3_1)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_1)
        if announce:
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def gather_kv_rows_pa_pairwise(
    sync,
    bufs,
    ws,
    kv_tensor,
    idx,
    token_row,
    width,
    base_line,
    n_rows,
    slot,
    slot_stride,
    ws_base=0,
    announce: bool = True,
    block_table=None,
    bt_seq=0,
    bt_max=0,
    pa_block_size=0,
    pa_stride=0,
    topk_value_mode=1,
):
    require_configured("gather_kv_rows_pa_pairwise")
    if _TRACE_STAGE:
        require_share("gather_kv_rows_pa_pairwise")
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        first = (n_rows + 1) // 2
        if _TRACE_QUARTER:
            par = tla.arch.block_idx() % 2
            h0 = (first + 1) // 2
            h1 = first - h0
            second = n_rows - first
            s0 = (second + 1) // 2
            s1 = second - s0
            my_start = 0
            my_rows = h0
            if sub == 1:
                my_start = h0
                my_rows = h1
            if par == 1:
                my_start = first
                my_rows = s0
                if sub == 1:
                    my_start = first + s0
                    my_rows = s1
        else:
            my_start = 0
            my_rows = first
            if sub == 1:
                my_start = first
                my_rows = n_rows - first
        n_chunks = (my_rows + KV_UNIT - 1) // KV_UNIT
        pair_zero = tla.Int64(0)
        pair_one = tla.Int64(1)
        pair_hi = tla.Int64(1073741824)
        for c in tla.range(n_chunks):
            line0 = my_start + c * KV_UNIT
            rows_out = my_rows - c * KV_UNIT
            if rows_out > KV_UNIT:
                rows_out = tla.as_numeric(KV_UNIT)
            if rows_out < 0:
                rows_out = tla.as_numeric(0)
            if c % 2 == 0:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv0,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_mte3_mte2_0)
                _pa_pairwise_fill(
                    ub_kv,
                    bufs.ub_kv0,
                    kv_tensor,
                    idx,
                    token_row,
                    width,
                    base_line,
                    n_rows,
                    line0,
                    block_table,
                    bt_seq,
                    bt_max,
                    pa_block_size,
                    pa_stride,
                    topk_value_mode,
                    pair_zero,
                    pair_one,
                    pair_hi,
                )
                tla.set_flag(sync.kv_mte2_mte3_0)
                tla.wait_flag(sync.kv_mte2_mte3_0)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_0)
            else:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv1,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_mte3_mte2_1)
                _pa_pairwise_fill(
                    ub_kv,
                    bufs.ub_kv1,
                    kv_tensor,
                    idx,
                    token_row,
                    width,
                    base_line,
                    n_rows,
                    line0,
                    block_table,
                    bt_seq,
                    bt_max,
                    pa_block_size,
                    pa_stride,
                    topk_value_mode,
                    pair_zero,
                    pair_one,
                    pair_hi,
                )
                tla.set_flag(sync.kv_mte2_mte3_1)
                tla.wait_flag(sync.kv_mte2_mte3_1)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_1)
        if announce:
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def _pa_pairwise_fill(
    ub_kv,
    ub_base,
    kv_tensor,
    idx,
    token_row,
    width,
    base_line,
    n_rows,
    line0,
    block_table,
    bt_seq,
    bt_max,
    pa_block_size,
    pa_stride,
    topk_value_mode,
    pair_zero,
    pair_one,
    pair_hi,
):
    for blk in static_range(KV_UNIT // 8):
        blk_base = 8 * blk
        addr_t = ()
        for ai in static_range(8):
            line = line0 + blk_base + ai
            if line > n_rows - 1:
                line = n_rows - 1
            src_raw = idx[token_row * width + base_line + line]
            src = src_raw
            if src < 0:
                src = 0
            blk_i = src // pa_block_size
            off_i = src - blk_i * pa_block_size
            phys = blk_i
            if topk_value_mode == 1:
                phys = block_table[bt_seq * bt_max + blk_i]
                if phys < 0:
                    phys = 0
            off = phys * pa_stride + off_i * HEAD_DIM
            if src_raw < 0:
                off = -1
            if line0 + blk_base + ai > n_rows - 1:
                off = -1
            addr_t = addr_t + (off.to(tla.Int64),)
        # CopyInKvSparse（:495-560）：i += 2
        for p in static_range(4):
            row_a = blk_base + 2 * p
            off0 = addr_t[2 * p]
            off1 = addr_t[2 * p + 1]
            v0 = (off0 >= pair_zero).to(tla.Int64)
            v1 = (off1 >= pair_zero).to(tla.Int64)
            if v0 + v1 > pair_zero:
                pair_delta = off1 - off0
                c_desc = (pair_delta < pair_zero).to(tla.Int64)
                pair_abs = pair_delta * (pair_one - c_desc - c_desc)
                sel = v1 * c_desc
                pair_base = off0 + sel * pair_delta
                block_count = pair_one + v1
                pair_stride = pair_abs * tla.Int64(_PAIR_ELEM_BYTES) - tla.Int64(
                    HEAD_DIM * _PAIR_ELEM_BYTES
                )
                if pair_stride >= pair_zero and pair_stride < pair_hi:
                    pair_gm_to_ub(
                        tla.recast_ptr(kv_tensor.ptr + pair_base, dtype=tla.UInt8),
                        tla.recast_ptr(ub_base, dtype=tla.UInt8),
                        tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                        pair_stride,
                        tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                        block_count,
                    )
                else:
                    if off0 >= pair_zero:
                        gm_row_a = tla.make_tensor(
                            kv_tensor.ptr + off0,
                            tla.make_layout(
                                tla.make_shape(1, HEAD_DIM),
                                tla.make_stride(HEAD_DIM, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.copy(
                            tla.tile_view(
                                ub_kv,
                                tla.make_shape(1, HEAD_DIM),
                                tla.make_coord(row_a, 0),
                            ),
                            gm_row_a,
                        )
                    if off1 >= pair_zero:
                        gm_row_b = tla.make_tensor(
                            kv_tensor.ptr + off1,
                            tla.make_layout(
                                tla.make_shape(1, HEAD_DIM),
                                tla.make_stride(HEAD_DIM, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.copy(
                            tla.tile_view(
                                ub_kv,
                                tla.make_shape(1, HEAD_DIM),
                                tla.make_coord(row_a + 1, 0),
                            ),
                            gm_row_b,
                        )


REC_ROWS = 8
SCALE_GROUPS = 7  # 448 / 64


@tla.jit
def _emit_group_rec(
    ub_rec_bf,
    ub_o,
    row,
    group,
    val,
    preg,
    preg_b16,
    cast_pack,
    pack_store,
    brc16,
    scale_col=288,
):
    sb = tla.get_tile(
        ub_rec_bf, tla.make_coord(row, scale_col + group), tla.make_shape(1, 1)
    ).load(params=brc16)
    scale = sb.to(tla.Float32, _widen_slot(tla.params.RegSlot.ZERO), mask=preg_b16)
    scaled = tla.mul(val, scale, mask=preg)
    tla.get_tile(
        ub_o, tla.make_coord(row, group * GROUP_SIZE), tla.make_shape(1, GROUP_SIZE)
    ).store(scaled.to(tla.BFloat16, cast_pack, mask=preg), pack_store, mask=preg_b16)


@tla.jit
def _dequant_rec_block(ub_rec, ub_rec_bf, ub_o, rec_dtype, rows, m_rec, m_out):
    m_out.lock(pipe=tla.pipes.VECTOR)
    with tla.vec.func(mode="simd"):
        m8 = tla.create_mask(pattern=tla.mask.ALL, dtype=rec_dtype)
        preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        preg_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.BFloat16)
        brc16 = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B16)
        cast0 = _widen_slot(tla.params.RegSlot.ZERO)
        cast1 = _widen_slot(tla.params.RegSlot.ONE)
        cast2 = _widen_slot(tla.params.RegSlot.TWO)
        cast3 = _widen_slot(tla.params.RegSlot.THREE)
        cast_pack = _cast_slot(tla.params.RegSlot.ZERO)
        pack_store = tla.params.NormalStoreParams(
            store_dist=tla.params.StoreDist.DIST_PACK_B32
        )
        for r in tla.range(rows):
            for part in tla.range_constexpr(2):
                off = 128 + part * 256
                count = 256 if part == 0 else 192
                reg = tla.get_tile(
                    ub_rec, tla.make_coord(r, off), tla.make_shape(1, count)
                ).load()
                v0 = reg.to(tla.Float32, cast0, mask=m8)
                v1 = reg.to(tla.Float32, cast1, mask=m8)
                v2 = reg.to(tla.Float32, cast2, mask=m8)
                v3 = reg.to(tla.Float32, cast3, mask=m8)
                a0, a1 = tla.interleave(v0, v2)
                b0, b1 = tla.interleave(v1, v3)
                c00, c01 = tla.interleave(a0, b0)
                c10, c11 = tla.interleave(a1, b1)
                group = part * 4
                _emit_group_rec(
                    ub_rec_bf,
                    ub_o,
                    r,
                    group,
                    c00,
                    preg,
                    preg_b16,
                    cast_pack,
                    pack_store,
                    brc16,
                )
                _emit_group_rec(
                    ub_rec_bf,
                    ub_o,
                    r,
                    group + 1,
                    c01,
                    preg,
                    preg_b16,
                    cast_pack,
                    pack_store,
                    brc16,
                )
                _emit_group_rec(
                    ub_rec_bf,
                    ub_o,
                    r,
                    group + 2,
                    c10,
                    preg,
                    preg_b16,
                    cast_pack,
                    pack_store,
                    brc16,
                )
                if part == 0:
                    _emit_group_rec(
                        ub_rec_bf,
                        ub_o,
                        r,
                        group + 3,
                        c11,
                        preg,
                        preg_b16,
                        cast_pack,
                        pack_store,
                        brc16,
                    )
            rreg = tla.get_tile(
                ub_rec_bf, tla.make_coord(r, 0), tla.make_shape(1, ROPE_DIM)
            ).load()
            tla.get_tile(
                ub_o,
                tla.make_coord(r, SCALE_GROUPS * GROUP_SIZE),
                tla.make_shape(1, ROPE_DIM),
            ).store(rreg)
    m_out.unlock(pipe=tla.pipes.VECTOR)


@tla.jit
def _pa_rec_fill(
    ub_rec,
    rec,
    idx,
    token_row,
    width,
    base_line,
    n_rows,
    line0,
    block_table,
    bt_seq,
    bt_max,
    pa_block_size,
    rec_stride_rows,
    is_cmp,
    start_row,
    topk_value_mode,
    pair_zero,
    pair_one,
    pair_hi,
    m_rec_s=None,
):
    for blk in static_range(REC_ROWS // 8):
        blk_base = 8 * blk
        addr_t = ()
        for ai in static_range(8):
            line = line0 + blk_base + ai
            if line > n_rows - 1:
                line = n_rows - 1
            valid = 1
            if line0 + blk_base + ai > n_rows - 1:
                valid = tla.as_numeric(0)
            off = tla.as_numeric(-1)
            if is_cmp == 1:
                src_raw = idx[token_row * width + base_line + line]
                src = src_raw
                if src < 0:
                    src = 0
                blk_i = src // pa_block_size
                off_i = src - blk_i * pa_block_size
                phys = blk_i
                if topk_value_mode == 1:
                    phys = block_table[bt_seq * bt_max + blk_i]
                    if phys < 0:
                        phys = 0
                off = phys * rec_stride_rows + off_i
                if src_raw < 0:
                    off = tla.as_numeric(-1)
            else:
                pos = start_row + line
                blk_i = pos // pa_block_size
                off_i = pos - blk_i * pa_block_size
                phys = block_table[bt_seq * bt_max + blk_i]
                if phys < 0:
                    phys = 0
                off = phys * rec_stride_rows + off_i
            if valid == 0:
                off = tla.as_numeric(-1)
            addr_t = addr_t + (off.to(tla.Int64),)
        for p in static_range(4):
            row_a = blk_base + 2 * p
            off0 = addr_t[2 * p]
            off1 = addr_t[2 * p + 1]
            v0 = (off0 >= pair_zero).to(tla.Int64)
            v1 = (off1 >= pair_zero).to(tla.Int64)
            if v0 + v1 > pair_zero:
                pair_delta = off1 - off0
                c_desc = (pair_delta < pair_zero).to(tla.Int64)
                pair_abs = pair_delta * (pair_one - c_desc - c_desc)
                sel = v1 * c_desc
                pair_base = off0 + sel * pair_delta
                block_count = pair_one + v1
                pair_stride = pair_abs * tla.Int64(REC_BYTES) - tla.Int64(REC_BYTES)
                if pair_stride >= pair_zero and pair_stride < pair_hi:
                    pair_gm_to_ub(
                        tla.recast_ptr(
                            rec.ptr + pair_base * tla.Int64(REC_BYTES), dtype=tla.UInt8
                        ),
                        tla.recast_ptr(ub_rec.ptr, dtype=tla.UInt8),
                        tla.Int64(REC_BYTES),
                        pair_stride,
                        tla.Int64(row_a * REC_BYTES),
                        block_count,
                    )
                else:
                    if off0 >= pair_zero:
                        gm_row_a = tla.make_tensor(
                            rec.ptr + off0 * tla.Int64(REC_BYTES),
                            tla.make_layout(
                                tla.make_shape(1, REC_BYTES),
                                tla.make_stride(REC_BYTES, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.copy(
                            tla.tile_view(
                                ub_rec,
                                tla.make_shape(1, REC_BYTES),
                                tla.make_coord(row_a, 0),
                            ),
                            gm_row_a,
                        )
                    if off1 >= pair_zero:
                        gm_row_b = tla.make_tensor(
                            rec.ptr + off1 * tla.Int64(REC_BYTES),
                            tla.make_layout(
                                tla.make_shape(1, REC_BYTES),
                                tla.make_stride(REC_BYTES, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.copy(
                            tla.tile_view(
                                ub_rec,
                                tla.make_shape(1, REC_BYTES),
                                tla.make_coord(row_a + 1, 0),
                            ),
                            gm_row_b,
                        )


@tla.jit
def _rec_chunk(
    ws,
    rec,
    idx,
    token_row,
    width,
    base_line,
    n_rows,
    block_table,
    bt_seq,
    bt_max,
    pa_block_size,
    rec_stride_rows,
    is_cmp,
    start_row,
    topk_value_mode,
    pair_zero,
    pair_one,
    pair_hi,
    my_start,
    my_rows,
    c,
    slot,
    slot_stride,
    ws_base,
    ub_rec,
    ub_rec_bf,
    ub_out,
    m_rec_s,
    m_out_s,
):
    line0 = my_start + c * REC_ROWS
    rows_out = my_rows - c * REC_ROWS
    if rows_out > REC_ROWS:
        rows_out = tla.as_numeric(REC_ROWS)
    if rows_out < 0:
        rows_out = tla.as_numeric(0)
    m_rec_s.lock(pipe=tla.pipes.MTE2)
    _pa_rec_fill(
        ub_rec,
        rec,
        idx,
        token_row,
        width,
        base_line,
        n_rows,
        line0,
        block_table,
        bt_seq,
        bt_max,
        pa_block_size,
        rec_stride_rows,
        is_cmp,
        start_row,
        topk_value_mode,
        pair_zero,
        pair_one,
        pair_hi,
        m_rec_s,
    )
    m_rec_s.unlock(pipe=tla.pipes.MTE2)
    m_rec_s.lock(pipe=tla.pipes.VECTOR)
    _dequant_rec_block(
        ub_rec, ub_rec_bf, ub_out, rec.ptr.dtype, REC_ROWS, m_rec_s, m_out_s
    )
    m_rec_s.unlock(pipe=tla.pipes.VECTOR)
    m_out_s.lock(pipe=tla.pipes.MTE3)
    gm_chunk = tla.make_tensor(
        ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
        tla.make_layout(
            tla.make_shape(rows_out, HEAD_DIM),
            tla.make_stride(HEAD_DIM, 1),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    tla.copy(
        gm_chunk,
        tla.tile_view(ub_out, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)),
    )
    m_out_s.unlock(pipe=tla.pipes.MTE3)


@tla.jit
def gather_kv_rows_pa_dequant(
    sync,
    bufs,
    ws,
    rec,
    idx,
    token_row,
    width,
    base_line,
    n_rows,
    slot,
    slot_stride,
    ws_base,
    block_table,
    bt_seq,
    bt_max,
    pa_block_size,
    pa_stride,
    is_cmp,
    start_row,
    topk_value_mode=1,
    announce: bool = True,
):
    require_configured("gather_kv_rows_pa_dequant")
    if _TRACE_STAGE:
        require_share("gather_kv_rows_pa_dequant")
    rec_stride_rows = pa_stride // REC_BYTES
    rec_dtype = rec.ptr.dtype
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        first = (n_rows + 1) // 2
        if _TRACE_QUARTER:
            par = tla.arch.block_idx() % 2
            h0 = (first + 1) // 2
            h1 = first - h0
            second = n_rows - first
            s0 = (second + 1) // 2
            s1 = second - s0
            my_start = 0
            my_rows = h0
            if sub == 1:
                my_start = h0
                my_rows = h1
            if par == 1:
                my_start = first
                my_rows = s0
                if sub == 1:
                    my_start = first + s0
                    my_rows = s1
        else:
            my_start = 0
            my_rows = first
            if sub == 1:
                my_start = first
                my_rows = n_rows - first
        rec_ptr = tla.recast_ptr(bufs.ub_kv0, dtype=rec_dtype)
        ub_rec = tla.make_tensor(
            rec_ptr,
            tla.make_layout(
                tla.make_shape(REC_ROWS, REC_BYTES),
                tla.make_stride(REC_BYTES, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ub_rec_bf = tla.make_tensor(
            tla.recast_ptr(bufs.ub_kv0, dtype=tla.BFloat16),
            tla.make_layout(
                tla.make_shape(REC_ROWS, REC_BYTES // 2),
                tla.make_stride(REC_BYTES // 2, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ub_out = tla.make_tensor(
            bufs.ub_kv1,
            tla.make_layout(
                tla.make_shape(REC_ROWS, HEAD_DIM),
                tla.make_stride(HEAD_DIM, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ub_rec_1 = tla.make_tensor(
            rec_ptr + tla.Int64(REC_ROWS * REC_BYTES),
            tla.make_layout(
                tla.make_shape(REC_ROWS, REC_BYTES),
                tla.make_stride(REC_BYTES, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ub_rec_bf_1 = tla.make_tensor(
            tla.recast_ptr(bufs.ub_kv0, dtype=tla.BFloat16)
            + tla.Int64(REC_ROWS * (REC_BYTES // 2)),
            tla.make_layout(
                tla.make_shape(REC_ROWS, REC_BYTES // 2),
                tla.make_stride(REC_BYTES // 2, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ub_out_1 = tla.make_tensor(
            bufs.ub_kv1 + REC_ROWS * HEAD_DIM,
            tla.make_layout(
                tla.make_shape(REC_ROWS, HEAD_DIM),
                tla.make_stride(HEAD_DIM, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        n_chunks = (my_rows + REC_ROWS - 1) // REC_ROWS
        n_pairs = (n_chunks + 1) // 2
        pair_zero = tla.Int64(0)
        pair_one = tla.Int64(1)
        pair_hi = tla.Int64(1073741824)
        for pp in tla.range(n_pairs):
            c0 = pp + pp
            _rec_chunk(
                ws,
                rec,
                idx,
                token_row,
                width,
                base_line,
                n_rows,
                block_table,
                bt_seq,
                bt_max,
                pa_block_size,
                rec_stride_rows,
                is_cmp,
                start_row,
                topk_value_mode,
                pair_zero,
                pair_one,
                pair_hi,
                my_start,
                my_rows,
                c0,
                slot,
                slot_stride,
                ws_base,
                ub_rec,
                ub_rec_bf,
                ub_out,
                sync.m_dq_rec0,
                sync.m_dq_out0,
            )
            c1 = c0 + 1
            if c1 < n_chunks:
                _rec_chunk(
                    ws,
                    rec,
                    idx,
                    token_row,
                    width,
                    base_line,
                    n_rows,
                    block_table,
                    bt_seq,
                    bt_max,
                    pa_block_size,
                    rec_stride_rows,
                    is_cmp,
                    start_row,
                    topk_value_mode,
                    pair_zero,
                    pair_one,
                    pair_hi,
                    my_start,
                    my_rows,
                    c1,
                    slot,
                    slot_stride,
                    ws_base,
                    ub_rec_1,
                    ub_rec_bf_1,
                    ub_out_1,
                    sync.m_dq_rec1,
                    sync.m_dq_out1,
                )
        if announce:
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def gather_kv_rows_vec(
    sync,
    bufs,
    ws,
    kv_tensor,
    phy_table,
    token_row,
    phy_width,
    base_line,
    n_rows,
    slot,
    slot_stride,
    ws_base=0,
    announce: bool = True,
):
    require_configured("gather_kv_rows_vec")
    if _TRACE_STAGE:
        require_share("gather_kv_rows_vec")
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        first = (n_rows + 1) // 2
        if _TRACE_QUARTER:
            par = tla.arch.block_idx() % 2
            h0 = (first + 1) // 2
            h1 = first - h0
            second = n_rows - first
            s0 = (second + 1) // 2
            s1 = second - s0
            my_start = 0
            my_rows = h0
            if sub == 1:
                my_start = h0
                my_rows = h1
            if par == 1:
                my_start = first
                my_rows = s0
                if sub == 1:
                    my_start = first + s0
                    my_rows = s1
        else:
            my_start = 0
            my_rows = first
            if sub == 1:
                my_start = first
                my_rows = n_rows - first
        n_chunks = (my_rows + KV_UNIT - 1) // KV_UNIT
        for c in tla.range(n_chunks):
            line0 = my_start + c * KV_UNIT
            rows_out = my_rows - c * KV_UNIT
            if rows_out > KV_UNIT:
                rows_out = tla.as_numeric(KV_UNIT)
            if rows_out < 0:
                rows_out = tla.as_numeric(0)
            if c % 2 == 0:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv0,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_mte3_mte2_0)
                if _SPARSE_BLOCK_SIZE == 1:
                    #   （`for i in 0..7: tokenData[i] = phyAddrGm64.GetValue(idx+i)`，
                    for blk in static_range(KV_UNIT // 8):
                        blk_base = 8 * blk
                        addr_t = ()
                        for ai in static_range(8):
                            aline = line0 + blk_base + ai
                            if aline > n_rows - 1:
                                aline = n_rows - 1
                            if aline < 0:
                                aline = 0
                            addr_t = addr_t + (
                                phy_table[token_row * phy_width + base_line + aline],
                            )
                        pair_lo = tla.Int64(HEAD_DIM)
                        pair_hi = tla.Int64(1073741824)
                        pair_zero = tla.Int64(0)
                        for p in static_range(4):
                            row_a = blk_base + 2 * p
                            off_a = addr_t[2 * p]
                            off_b = addr_t[2 * p + 1]
                            pair_delta = off_b - off_a
                            valid_a = row_a <= n_rows - 1 - line0
                            valid_b = valid_a and row_a + 1 <= n_rows - 1 - line0
                            if (
                                valid_b
                                and pair_delta >= pair_lo
                                and pair_delta < pair_hi
                            ):
                                pair_gm_to_ub(
                                    tla.recast_ptr(
                                        kv_tensor.ptr + off_a, dtype=tla.UInt8
                                    ),
                                    tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                    tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    pair_delta * tla.Int64(_PAIR_ELEM_BYTES)
                                    - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(2),
                                )
                            elif (
                                valid_b
                                and pair_zero - pair_delta >= pair_lo
                                and pair_zero - pair_delta < pair_hi
                            ):
                                pair_gm_to_ub(
                                    tla.recast_ptr(
                                        kv_tensor.ptr + off_b, dtype=tla.UInt8
                                    ),
                                    tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                    tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    (pair_zero - pair_delta)
                                    * tla.Int64(_PAIR_ELEM_BYTES)
                                    - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(2),
                                )
                            elif valid_b:
                                for ri in static_range(2):
                                    row_off = addr_t[2 * p + ri]
                                    gm_row = tla.make_tensor(
                                        kv_tensor.ptr + row_off,
                                        tla.make_layout(
                                            tla.make_shape(1, HEAD_DIM),
                                            tla.make_stride(HEAD_DIM, 1),
                                            layoutTag=tla.arch.RowMajor,
                                        ),
                                    )
                                    tla.copy(
                                        tla.tile_view(
                                            ub_kv,
                                            tla.make_shape(1, HEAD_DIM),
                                            tla.make_coord(row_a + ri, 0),
                                        ),
                                        gm_row,
                                    )
                            elif valid_a:
                                pair_gm_to_ub(
                                    tla.recast_ptr(
                                        kv_tensor.ptr + off_a, dtype=tla.UInt8
                                    ),
                                    tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                    tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    pair_zero,
                                    tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(1),
                                )
                else:
                    for i in static_range(KV_UNIT):
                        line = line0 + i
                        if line > n_rows - 1:
                            line = n_rows - 1
                        if line < 0:
                            line = 0
                        row_off = phy_table[token_row * phy_width + base_line + line]
                        gm_row = tla.make_tensor(
                            kv_tensor.ptr + row_off,
                            tla.make_layout(
                                tla.make_shape(1, HEAD_DIM),
                                tla.make_stride(HEAD_DIM, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.copy(
                            tla.tile_view(
                                ub_kv, tla.make_shape(1, HEAD_DIM), tla.make_coord(i, 0)
                            ),
                            gm_row,
                        )
                tla.set_flag(sync.kv_mte2_mte3_0)
                tla.wait_flag(sync.kv_mte2_mte3_0)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_0)
            else:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv1,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_mte3_mte2_1)
                if _SPARSE_BLOCK_SIZE == 1:
                    for blk in static_range(KV_UNIT // 8):
                        blk_base = 8 * blk
                        addr_t = ()
                        for ai in static_range(8):
                            aline = line0 + blk_base + ai
                            if aline > n_rows - 1:
                                aline = n_rows - 1
                            if aline < 0:
                                aline = 0
                            addr_t = addr_t + (
                                phy_table[token_row * phy_width + base_line + aline],
                            )
                        pair_lo = tla.Int64(HEAD_DIM)
                        pair_hi = tla.Int64(1073741824)
                        pair_zero = tla.Int64(0)
                        for p in static_range(4):
                            row_a = blk_base + 2 * p
                            off_a = addr_t[2 * p]
                            off_b = addr_t[2 * p + 1]
                            pair_delta = off_b - off_a
                            valid_a = row_a <= n_rows - 1 - line0
                            valid_b = valid_a and row_a + 1 <= n_rows - 1 - line0
                            if (
                                valid_b
                                and pair_delta >= pair_lo
                                and pair_delta < pair_hi
                            ):
                                pair_gm_to_ub(
                                    tla.recast_ptr(
                                        kv_tensor.ptr + off_a, dtype=tla.UInt8
                                    ),
                                    tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
                                    tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    pair_delta * tla.Int64(_PAIR_ELEM_BYTES)
                                    - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(2),
                                )
                            elif (
                                valid_b
                                and pair_zero - pair_delta >= pair_lo
                                and pair_zero - pair_delta < pair_hi
                            ):
                                pair_gm_to_ub(
                                    tla.recast_ptr(
                                        kv_tensor.ptr + off_b, dtype=tla.UInt8
                                    ),
                                    tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
                                    tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    (pair_zero - pair_delta)
                                    * tla.Int64(_PAIR_ELEM_BYTES)
                                    - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(2),
                                )
                            elif valid_b:
                                for ri in static_range(2):
                                    row_off = addr_t[2 * p + ri]
                                    gm_row = tla.make_tensor(
                                        kv_tensor.ptr + row_off,
                                        tla.make_layout(
                                            tla.make_shape(1, HEAD_DIM),
                                            tla.make_stride(HEAD_DIM, 1),
                                            layoutTag=tla.arch.RowMajor,
                                        ),
                                    )
                                    tla.copy(
                                        tla.tile_view(
                                            ub_kv,
                                            tla.make_shape(1, HEAD_DIM),
                                            tla.make_coord(row_a + ri, 0),
                                        ),
                                        gm_row,
                                    )
                            elif valid_a:
                                pair_gm_to_ub(
                                    tla.recast_ptr(
                                        kv_tensor.ptr + off_a, dtype=tla.UInt8
                                    ),
                                    tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
                                    tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    pair_zero,
                                    tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(1),
                                )
                else:
                    for i in static_range(KV_UNIT):
                        line = line0 + i
                        if line > n_rows - 1:
                            line = n_rows - 1
                        if line < 0:
                            line = 0
                        row_off = phy_table[token_row * phy_width + base_line + line]
                        gm_row = tla.make_tensor(
                            kv_tensor.ptr + row_off,
                            tla.make_layout(
                                tla.make_shape(1, HEAD_DIM),
                                tla.make_stride(HEAD_DIM, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.copy(
                            tla.tile_view(
                                ub_kv, tla.make_shape(1, HEAD_DIM), tla.make_coord(i, 0)
                            ),
                            gm_row,
                        )
                tla.set_flag(sync.kv_mte2_mte3_1)
                tla.wait_flag(sync.kv_mte2_mte3_1)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_1)
        if announce:
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def gather_kv_rows_ring(
    sync,
    bufs,
    ws,
    kv_tensor,
    idx,
    token_row,
    idx_width,
    row_base,
    base_line,
    n_rows,
    slot,
    slot_stride,
    idx_slot,
    ws_base=0,
    announce: bool = True,
):
    require_configured("gather_kv_rows_vec")
    if _TRACE_STAGE:
        require_share("gather_kv_rows_vec")
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        first = (n_rows + 1) // 2
        if _TRACE_QUARTER:
            par = tla.arch.block_idx() % 2
            h0 = (first + 1) // 2
            h1 = first - h0
            second = n_rows - first
            s0 = (second + 1) // 2
            s1 = second - s0
            my_start = 0
            my_rows = h0
            if sub == 1:
                my_start = h0
                my_rows = h1
            if par == 1:
                my_start = first
                my_rows = s0
                if sub == 1:
                    my_start = first + s0
                    my_rows = s1
        else:
            my_start = 0
            my_rows = first
            if sub == 1:
                my_start = first
                my_rows = n_rows - first
        n_chunks = (my_rows + KV_UNIT - 1) // KV_UNIT
        a0_abs = token_row * idx_width + base_line + my_start
        rem = a0_abs - (a0_abs // 8) * 8
        a0 = a0_abs - rem
        cover = n_chunks * KV_UNIT
        if cover > my_rows:
            cover = my_rows
        cover = ((cover + 7) // 8) * 8
        room = idx_width - (base_line + my_start)
        if cover > room:
            cover = room
        need = rem + cover
        if n_chunks > 0:
            if idx_slot == 0:
                tla.wait_flag(sync.idx_s2m0)
            else:
                tla.wait_flag(sync.idx_s2m1)
            tla.copy(
                tla.tile_view(
                    tla.make_tensor(
                        bufs.ub_idx_ring + idx_slot * IDX_RING_STRIDE,
                        tla.make_layout(
                            tla.make_shape(IDX_RING_STRIDE),
                            tla.make_stride(1),
                        ),
                    ),
                    tla.make_shape(need),
                    tla.make_coord(0),
                ),
                tla.make_tensor(
                    idx.ptr + a0,
                    tla.make_layout(
                        tla.make_shape(need),
                        tla.make_stride(1),
                    ),
                ),
            )
            if idx_slot == 0:
                tla.set_flag(sync.idx_m2s0)
                tla.wait_flag(sync.idx_m2s0)
            else:
                tla.set_flag(sync.idx_m2s1)
                tla.wait_flag(sync.idx_m2s1)
        ring_t = tla.make_tensor(
            bufs.ub_idx_ring,
            tla.make_layout(
                tla.make_shape(IDX_RING_SLOTS * IDX_RING_STRIDE),
                tla.make_stride(1),
            ),
        )
        ring_o = idx_slot * IDX_RING_STRIDE + rem - my_start
        last_row = my_start + cover - 1
        if last_row > n_rows - 1:
            last_row = n_rows - 1
        for c in tla.range(n_chunks):
            line0 = my_start + c * KV_UNIT
            rows_out = my_rows - c * KV_UNIT
            if rows_out > KV_UNIT:
                rows_out = tla.as_numeric(KV_UNIT)
            if rows_out < 0:
                rows_out = tla.as_numeric(0)
            if c % 2 == 0:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv0,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_mte3_mte2_0)
                for blk in static_range(KV_UNIT // 8):
                    blk_base = 8 * blk
                    addr_t = ()
                    for ai in static_range(8):
                        aline = line0 + blk_base + ai
                        if aline > last_row:
                            aline = last_row
                        if aline < 0:
                            aline = 0
                        v = ring_t[ring_o + aline]
                        if v < 0:
                            v = 0
                        addr_t = addr_t + (
                            (row_base + v).to(tla.Int64) * tla.Int64(HEAD_DIM),
                        )
                    pair_lo = tla.Int64(HEAD_DIM)
                    pair_hi = tla.Int64(1073741824)
                    pair_zero = tla.Int64(0)
                    for p in static_range(4):
                        row_a = blk_base + 2 * p
                        off_a = addr_t[2 * p]
                        off_b = addr_t[2 * p + 1]
                        pair_delta = off_b - off_a
                        valid_a = row_a <= n_rows - 1 - line0
                        valid_b = valid_a and row_a + 1 <= n_rows - 1 - line0
                        if valid_b and pair_delta >= pair_lo and pair_delta < pair_hi:
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_a, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                pair_delta * tla.Int64(_PAIR_ELEM_BYTES)
                                - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(2),
                            )
                        elif (
                            valid_b
                            and pair_zero - pair_delta >= pair_lo
                            and pair_zero - pair_delta < pair_hi
                        ):
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_b, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                (pair_zero - pair_delta) * tla.Int64(_PAIR_ELEM_BYTES)
                                - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(2),
                            )
                        elif valid_b:
                            for ri in static_range(2):
                                row_off = addr_t[2 * p + ri]
                                gm_row = tla.make_tensor(
                                    kv_tensor.ptr + row_off,
                                    tla.make_layout(
                                        tla.make_shape(1, HEAD_DIM),
                                        tla.make_stride(HEAD_DIM, 1),
                                        layoutTag=tla.arch.RowMajor,
                                    ),
                                )
                                tla.copy(
                                    tla.tile_view(
                                        ub_kv,
                                        tla.make_shape(1, HEAD_DIM),
                                        tla.make_coord(row_a + ri, 0),
                                    ),
                                    gm_row,
                                )
                        elif valid_a:
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_a, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                pair_zero,
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(1),
                            )
                tla.set_flag(sync.kv_mte2_mte3_0)
                tla.wait_flag(sync.kv_mte2_mte3_0)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_0)
            else:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv1,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_mte3_mte2_1)
                for blk in static_range(KV_UNIT // 8):
                    blk_base = 8 * blk
                    addr_t = ()
                    for ai in static_range(8):
                        aline = line0 + blk_base + ai
                        if aline > last_row:
                            aline = last_row
                        if aline < 0:
                            aline = 0
                        v = ring_t[ring_o + aline]
                        if v < 0:
                            v = 0
                        addr_t = addr_t + (
                            (row_base + v).to(tla.Int64) * tla.Int64(HEAD_DIM),
                        )
                    pair_lo = tla.Int64(HEAD_DIM)
                    pair_hi = tla.Int64(1073741824)
                    pair_zero = tla.Int64(0)
                    for p in static_range(4):
                        row_a = blk_base + 2 * p
                        off_a = addr_t[2 * p]
                        off_b = addr_t[2 * p + 1]
                        pair_delta = off_b - off_a
                        valid_a = row_a <= n_rows - 1 - line0
                        valid_b = valid_a and row_a + 1 <= n_rows - 1 - line0
                        if valid_b and pair_delta >= pair_lo and pair_delta < pair_hi:
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_a, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
                                tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                pair_delta * tla.Int64(_PAIR_ELEM_BYTES)
                                - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(2),
                            )
                        elif (
                            valid_b
                            and pair_zero - pair_delta >= pair_lo
                            and pair_zero - pair_delta < pair_hi
                        ):
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_b, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
                                tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                (pair_zero - pair_delta) * tla.Int64(_PAIR_ELEM_BYTES)
                                - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(2),
                            )
                        elif valid_b:
                            for ri in static_range(2):
                                row_off = addr_t[2 * p + ri]
                                gm_row = tla.make_tensor(
                                    kv_tensor.ptr + row_off,
                                    tla.make_layout(
                                        tla.make_shape(1, HEAD_DIM),
                                        tla.make_stride(HEAD_DIM, 1),
                                        layoutTag=tla.arch.RowMajor,
                                    ),
                                )
                                tla.copy(
                                    tla.tile_view(
                                        ub_kv,
                                        tla.make_shape(1, HEAD_DIM),
                                        tla.make_coord(row_a + ri, 0),
                                    ),
                                    gm_row,
                                )
                        elif valid_a:
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_a, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
                                tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                pair_zero,
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(1),
                            )
                tla.set_flag(sync.kv_mte2_mte3_1)
                tla.wait_flag(sync.kv_mte2_mte3_1)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_1)
        if n_chunks > 0:
            if idx_slot == 0:
                tla.set_flag(sync.idx_s2m0)
            else:
                tla.set_flag(sync.idx_s2m1)
        if announce:
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


KV_STAGE_STRIDE = KV_BLOCK * 512

WS_BLOCKS = 32

CSA_PRELOAD_NUM = 3


@tla.jit
def _decode_vector_barrier(sync) -> None:
    with tla.vector():
        tla.pipe_barrier(tla.pipes.MTE3)
        tla.cross_core_set_flag(sync.decode_exchange, tla.pipes.MTE3)
        tla.cross_core_wait_flag(sync.decode_exchange, tla.pipes.MTE2)


@tla.jit
def _publish_decode_partition(
    sync, bufs, partial, partial_stats, pair, rows, slot, nonempty
):
    """Publish FP32 numerator, maximum and denominator for the second partition."""
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        head_base = (pair * 2 + sub) * rows
        shape = tla.make_layout(
            tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
        )
        stat_layout = tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
        gm_o = tla.make_tensor(partial.ptr + head_base * HEAD_DIM, shape)
        gm_m = tla.make_tensor(partial_stats.ptr + 2 * head_base, stat_layout)
        gm_z = tla.make_tensor(partial_stats.ptr + 2 * head_base + rows, stat_layout)
        local_o = tla.make_tensor(bufs.ub_o, shape)
        local_m = _max(bufs, rows, slot)
        local_z = _sum(bufs, rows, slot)
        if not nonempty:
            with tla.vec.func(mode="simd"):
                mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                active, _ = tla.update_mask(rows, dtype=tla.Float32)
                zero = tla.full(0.0, dtype=tla.Float32)
                local_m.store(tla.full(MIN_F, dtype=tla.Float32), mask=active)
                local_z.store(zero, mask=active)
                for r in tla.range(rows):
                    for v in tla.range(HEAD_DIM // VL):
                        tla.tile_view(
                            local_o, tla.make_shape(1, VL), tla.make_coord(r, v)
                        ).store(
                            zero,
                            mask=mask,
                        )
        tla.set_flag(sync.lse_v_mte3)
        tla.wait_flag(sync.lse_v_mte3)
        tla.copy(gm_o, local_o)
        tla.copy(gm_m, local_m)
        tla.copy(gm_z, local_z)


@tla.jit
def _merge_decode_partition(
    sync,
    bufs,
    partial,
    partial_stats,
    out,
    lse,
    pair,
    task,
    rows,
    slot,
    has_lse,
    dtype,
):
    """Merge two independent online softmax states at the query owner."""
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        head_base = (pair * 2 + sub) * rows
        shape = tla.make_layout(
            tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
        )
        stat_layout = tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
        remote_o = tla.make_tensor(bufs.ub_otmp_full, shape)
        remote_m = tla.make_tensor(bufs.ub_init_lse, stat_layout)
        remote_z = tla.make_tensor(bufs.ub_init_lse + rows, stat_layout)
        both_stats = tla.make_layout(tla.make_shape(2 * rows), tla.make_stride(1))
        gm_o = tla.make_tensor(partial.ptr + head_base * HEAD_DIM, shape)
        # The final local PV update still reads this UB region on Vector.
        pv_consumed = tla.flag(
            "decode_local_pv_consumed", tla.pipes.VECTOR, tla.pipes.MTE2, id=0
        )
        tla.set_flag(pv_consumed)
        tla.wait_flag(pv_consumed)
        tla.copy(remote_o, gm_o)
        tla.copy(
            tla.make_tensor(bufs.ub_init_lse, both_stats),
            tla.make_tensor(partial_stats.ptr + 2 * head_base, both_stats),
        )
        tla.set_flag(sync.phy_m2v)
        tla.wait_flag(sync.phy_m2v)
        local_o = tla.make_tensor(bufs.ub_o, shape)
        local_m = _max(bufs, rows, slot)
        local_z = _sum(bufs, rows, slot)
        answer = tla.make_tensor(bufs.ub_o16, shape)
        log_z = _dm(bufs, rows, slot)
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            mask_out = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            cast = _cast_slot(tla.params.RegSlot.ZERO)
            pack = tla.params.NormalStoreParams(
                store_dist=tla.params.StoreDist.DIST_PACK_B32
            )
            active, _ = tla.update_mask(rows, dtype=tla.Float32)
            m0 = local_m.load()
            m1 = remote_m.load()
            z0 = local_z.load()
            z1 = remote_z.load()
            mx = tla.max(m0, m1, mask=active)
            e0 = tla.exp(tla.sub(m0, mx, mask=active), mask=active)
            e1 = tla.exp(tla.sub(m1, mx, mask=active), mask=active)
            den = tla.add(
                tla.mul(z0, e0, mask=active), tla.mul(z1, e1, mask=active), mask=active
            )
            remote_m.store(tla.div(e0, den, mask=active), mask=active)
            remote_z.store(tla.div(e1, den, mask=active), mask=active)
            if has_lse:
                log_z.store(
                    tla.add(tla.log(den, mask=active), mx, mask=active), mask=active
                )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            for r in tla.range(rows):
                w0 = tla.tile_view(remote_m, tla.make_shape(1), tla.make_coord(r)).load(
                    params=brc
                )
                w1 = tla.tile_view(remote_z, tla.make_shape(1), tla.make_coord(r)).load(
                    params=brc
                )
                for v in tla.range(HEAD_DIM // VL):
                    where = tla.make_coord(r, v)
                    a = tla.tile_view(local_o, tla.make_shape(1, VL), where).load()
                    b = tla.tile_view(remote_o, tla.make_shape(1, VL), where).load()
                    value = tla.add(
                        tla.mul(a, w0, mask=mask), tla.mul(b, w1, mask=mask), mask=mask
                    )
                    tla.tile_view(answer, tla.make_shape(1, VL), where).store(
                        value.to(dtype, cast, mask=mask),
                        pack,
                        mask=mask_out,
                    )
        tla.set_flag(sync.lse_v_mte3)
        tla.wait_flag(sync.lse_v_mte3)
        row = task * 2 * rows + sub * rows
        tla.copy(tla.make_tensor(out.ptr + row * HEAD_DIM, shape), answer)
        if has_lse:
            tla.copy(tla.make_tensor(lse.ptr + row, stat_layout), log_z)


@tla.jit
def _csa_prefetch_q_half(
    sync, bufs, q, task, slot, finish: tla.Constexpr[bool], q_prefix_free=None
):
    with tla.cube():
        if finish:
            tla.wait_flag(sync.q_l0a_free)
        elif q_prefix_free is not None:
            tla.wait_flag(q_prefix_free)
        dtype = q.ptr.dtype
        column = Q_HALF if finish else 0
        gm = tla.make_tensor(
            q.ptr + task * Q_ROWS * HEAD_DIM + column,
            tla.make_layout(
                tla.make_shape(Q_ROWS, Q_HALF),
                tla.make_stride(HEAD_DIM, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        target = l1_q_tensor(_q_slot_ptr(bufs, slot), dtype, Q_ROWS)
        tla.copy(target, gm, cache_policy=tla.params.CachePolicy.DISABLE)
        if finish:
            tla.set_flag(sync.q_l1_ready)


@tla.jit
def _csa_load_full_ori(sync, bufs, kv, bt, start, slot, dtype, bs, stride):
    # The folded prefill route has one sequence and a page size divisible by 128.
    # Cold short windows can safely fill the rest from the first physical page.
    with tla.cube():
        _k_slot_wait(sync, slot)
        dst = l1_k_tensor(_k_slot_ptr(bufs, slot), dtype)
        length = KV_BLOCK
        copied = tla.as_numeric(0)
        for _ in (0, 1):
            if copied < length:
                pos = start + copied
                page = pos // bs
                within = pos % bs
                real = bt[page]
                real = real if real >= 0 else 0
                remaining = length - copied
                count = bs - within
                count = count if count < remaining else remaining
                shape = tla.make_shape(HEAD_DIM, count)
                gm = tla.make_tensor(
                    kv.ptr + real * stride + within * HEAD_DIM,
                    tla.make_layout(
                        shape,
                        tla.make_stride(1, HEAD_DIM),
                        layoutTag=tla.arch.ColumnMajor,
                    ),
                )
                tla.copy(tla.get_tile(dst, tla.make_coord(0, copied), shape), gm)
                copied = copied + count
        _k_slot_ready_set(sync, slot)


@tla.jit
def _csa_load_k_head_halves(
    sync,
    bufs,
    gm,
    count,
    slot,
    dtype,
    phase0,
    phase1,
    phase0_free,
    prefix: tla.Constexpr[int] = 384,
    split_columns: tla.Constexpr[bool] = False,
    split_prefix: tla.Constexpr[bool] = False,
):
    """Publish the two K halves independently to overlap MTE2 with QK."""
    with tla.cube():
        _k_slot_wait(sync, slot)
        l1k = l1_k_tensor(_k_slot_ptr(bufs, slot), dtype)
        parts = (
            (0, D_CHUNK),
            (D_CHUNK, D_CHUNK),
            (2 * D_CHUNK, D_CHUNK),
            (3 * D_CHUNK, D_CHUNK),
        )
        prefix_parts = (
            (0, D_CHUNK),
            (D_CHUNK, prefix - D_CHUNK),
            (prefix, HEAD_DIM - prefix),
        )
        segments = (
            prefix_parts
            if split_prefix
            else (
                parts if split_columns else ((0, prefix), (prefix, HEAD_DIM - prefix))
            )
        )
        for pos, width in segments:
            shape = tla.make_shape(width, count)
            layout = tla.make_layout(
                shape, tla.make_stride(1, HEAD_DIM), layoutTag=tla.arch.ColumnMajor
            )
            src = tla.make_tensor(gm + pos, layout)
            dst = tla.get_tile(l1k, tla.make_coord(pos, 0), shape)
            tla.copy(dst, src)
            prefix_ready = (
                pos + width == prefix if split_columns or split_prefix else pos == 0
            )
            if prefix_ready:
                if slot == 1:
                    tla.set_flag(phase1)
                else:
                    tla.wait_flag(phase0_free)
                    tla.set_flag(phase0)
        _k_slot_ready_set(sync, slot)


@tla.jit
def _csa_copy_cyclic_ori(
    l1,
    kv,
    bt,
    begin,
    end,
    bs,
    stride,
    head_begin: tla.Constexpr[int],
    head_width: tla.Constexpr[int],
    dst_head: tla.Constexpr[int],
):
    pos = begin
    for _ in (0, 1):
        if pos < end:
            offset = pos % KV_BLOCK
            page = pos // bs
            in_page = pos % bs
            physical = bt[page]
            physical = physical if physical >= 0 else 0
            remaining = end - pos
            capacity = KV_BLOCK - offset
            count = remaining if remaining < capacity else capacity
            shape = tla.make_shape(head_width, count)
            gm_ptr = kv.ptr + physical * stride + in_page * HEAD_DIM + head_begin
            gm = tla.make_tensor(
                gm_ptr,
                tla.make_layout(
                    shape, tla.make_stride(1, HEAD_DIM), layoutTag=tla.arch.ColumnMajor
                ),
            )
            tla.copy(tla.get_tile(l1, tla.make_coord(dst_head, offset), shape), gm)
            pos = pos + count


@tla.jit
def _csa_load_cyclic_ori(
    sync, bufs, kv, bt, start, previous_end, slot, dtype, bs, stride, ori_free
):
    """Keep the first head segment of the original window in logical row order modulo 128."""
    with tla.cube():
        active_head = D_CHUNK
        end = start + KV_BLOCK
        loaded = previous_end if previous_end > start else start
        dst = l1_k_tensor(_k_slot_ptr(bufs, slot), dtype)
        rolling = tla.make_tensor(
            bufs.l1_ori,
            tla.make_layout_with_tag((active_head, KV_BLOCK), dtype, tla.arch.nZ),
        )
        _k_slot_wait(sync, slot)
        tla.wait_flag(ori_free)
        _csa_copy_cyclic_ori(
            rolling, kv, bt, loaded, end, bs, stride, 0, active_head, 0
        )
        _csa_copy_cyclic_ori(
            dst,
            kv,
            bt,
            start,
            end,
            bs,
            stride,
            active_head,
            HEAD_DIM - active_head,
            active_head,
        )
        _k_slot_ready_set(sync, slot)


@tla.jit
def _csa_blocked_query(
    step,
    task0,
    warm_steps,
    paired_prefix,
    pair_chunk,
    stripe_pairs,
    complete,
    tail_start,
):
    position = step - warm_steps
    group = position // (2 * pair_chunk)
    within = position - group * (2 * pair_chunk)
    first_pair = group * stripe_pairs + task0 * pair_chunk
    if group == complete:
        first_pair = tail_start
    task = paired_prefix + 2 * first_pair + within
    if paired_prefix and step == 0:
        task = task0
    return task


@tla.jit
def _csa_cache_initial_stats(sync, bufs, sinks, rows, has_sinks: tla.Constexpr[int]):
    _swa_cache_sinks(sync, bufs, sinks, rows, has_sinks)
    with tla.vector():
        base = _stats(bufs.ub_sum + 2 * rows, VL)
        with tla.vec.func(mode="simd"):
            start = tla.full(1.0 if has_sinks else 0.0, dtype=tla.Float32)
            base.store(start)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


_DECODE_RANGE_READ_AIC = r"""
#include "kernel_operator.h"
extern "C" [aicore] __attribute__((always_inline)) void refresh_decode_guard_aic(uint64_t base) {
    auto* address = reinterpret_cast<__gm__ float*>(base);
    AscendC::GlobalTensor<float> cell;
    cell.SetGlobalBuffer(address, 16);
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(cell);
}
"""

_DECODE_RANGE_READ_AIV = _DECODE_RANGE_READ_AIC.replace(
    "refresh_decode_guard_aic", "refresh_decode_guard_aiv"
)


@tla.extern(source=_DECODE_RANGE_READ_AIC, name="refresh_decode_guard_aic")
def _refresh_decode_guard_aic(
    addr: tla.Pointer[tla.Float32, tla.AddressSpace.gm],
) -> None: ...


@tla.extern(source=_DECODE_RANGE_READ_AIV, name="refresh_decode_guard_aiv")
def _refresh_decode_guard_aiv(
    addr: tla.Pointer[tla.Float32, tla.AddressSpace.gm],
) -> None: ...


@tla.jit
def _store_decode_guard(guard, shared, from_vector):
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        view = tla.make_layout(tla.make_shape(8), tla.make_stride(1))
        out = tla.recast_ptr(shared + 8 * sub, dtype=tla.Int32)
        ready = tla.flag("decode_guard_stored", tla.pipes.VECTOR, tla.pipes.MTE3, id=7)
        tla.set_flag(ready)
        tla.wait_flag(ready)
        tla.copy(tla.make_tensor(out, view), tla.make_tensor(guard, view))
        tla.cross_core_set_flag(from_vector, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(from_vector, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def _wait_decode_guard(shared, from_vector, to_vector):
    with tla.cube():
        tla.cross_core_wait_flag(from_vector, tla.pipes.SCALAR, aiv_id=0)
        tla.cross_core_wait_flag(from_vector, tla.pipes.SCALAR, aiv_id=1)
        _refresh_decode_guard_aic(shared)
        tla.cross_core_set_flag(to_vector, tla.pipes.SCALAR, aiv_id=0)
        tla.cross_core_set_flag(to_vector, tla.pipes.SCALAR, aiv_id=1)
    with tla.vector():
        tla.cross_core_wait_flag(to_vector, tla.pipes.SCALAR, aiv_id=0)
        tla.cross_core_wait_flag(to_vector, tla.pipes.SCALAR, aiv_id=1)
        _refresh_decode_guard_aiv(shared)


@tla.jit
def _csa_init_fixed_decode(sync, bufs, sinks, rows, slot, has_sinks, guard):
    with tla.vector():
        if has_sinks:
            head = tla.arch.sub_block_idx() * rows
            layout = tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
            tla.copy(
                tla.make_tensor(bufs.ub_sink, layout),
                tla.make_tensor(sinks.ptr + head, layout),
            )
            tla.set_flag(sync.phy_m2v)
            tla.wait_flag(sync.phy_m2v)
        _csa_fixed_init(
            bufs.ub_sink,
            _sum(bufs, rows, slot).ptr,
            _max(bufs, rows, slot).ptr,
            has_sinks.to(tla.Int32),
            guard,
        )


@tla.jit
def _csa_decode_fixed_vec1(sync, bufs, scale, count, slot, previous_slot, guard):
    with tla.vector():
        rows = Q_ROWS // 2
        sub = tla.arch.sub_block_idx()
        score = _ub_s(bufs, rows, slot)
        prob = _ub_p(bufs, rows)
        maximum = _max(bufs, rows, slot)
        total = _sum(bufs, rows, slot)
        previous = _sum(bufs, rows, previous_slot)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=1)
        sync.m_p.lock(pipe=tla.pipes.VECTOR)
        _csa_fixed_exp(
            score.ptr,
            prob.ptr,
            previous.ptr,
            total.ptr,
            maximum.ptr,
            count,
            scale,
            guard,
        )
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=1)
        target = l1_p_tensor(bufs.l1_p + slot * Q_ROWS * KV_BLOCK, tla.BFloat16, Q_ROWS)
        target_half = tla.tile_view(
            target, tla.make_shape(rows, KV_BLOCK), tla.make_coord(sub, 0)
        )
        sync.m_p.unlock(pipe=tla.pipes.VECTOR)
        sync.m_p.lock(pipe=tla.pipes.MTE3)
        tla.set_flag(sync.p_data_ready)
        tla.wait_flag(sync.p_data_ready)
        tla.copy(target_half, prob)
        sync.m_p.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def _csa_decode_fixed_pv(
    sync,
    bufs,
    seg: tla.Constexpr[int],
    k_slot,
    p_slot,
    width,
    first,
    compact: tla.Constexpr[bool] = False,
):
    with tla.cube():
        dtype = tla.BFloat16
        m = Q_ROWS
        probs = l1_p_tensor(bufs.l1_p + p_slot * Q_ROWS * KV_BLOCK, dtype, m)
        p_slice = tla.tile_view(probs, tla.make_shape(m, width), tla.make_coord(0, 0))
        v_base = _k_slot_ptr(bufs, k_slot) + seg * KV_BLOCK * D_CHUNK
        values = l1_v_seg_tensor(v_base, dtype)
        v_slice = tla.tile_view(
            values, tla.make_shape(width, D_CHUNK), tla.make_coord(0, 0)
        )
        a = bufs.l0a0 if seg % 2 == 0 else bufs.l0a1
        b = bufs.l0b0 if seg % 2 == 0 else bufs.l0b1
        free = sync.ab_free0 if seg % 2 == 0 else sync.ab_free1
        ready = sync.ab_ready0 if seg % 2 == 0 else sync.ab_ready1
        p_l0 = tla.make_tensor_like(a, p_slice)
        v_l0 = tla.make_tensor_like(b, v_slice)
        if compact:
            v_l0 = tla.make_tensor(
                b, tla.make_layout_with_tag((width, D_CHUNK), dtype, tla.arch.nZ)
            )
        output = tla.make_tensor(
            bufs.l0c_o + seg * m * D_CHUNK,
            tla.make_layout_with_tag((m, D_CHUNK), tla.Float32, tla.arch.L0Clayout),
        )
        if seg == 0:
            if first:
                tla.wait_flag(sync.o_fix_done0)
        tla.wait_flag(free)
        if seg == 0:
            tla.copy(v_l0, v_slice)
            tla.cross_core_wait_flag(sync.p_ready, tla.pipes.MTE1, aiv_id=0)
            tla.cross_core_wait_flag(sync.p_ready, tla.pipes.MTE1, aiv_id=1)
            tla.copy(p_l0, p_slice)
        else:
            if seg == 1:
                tla.copy(p_l0, p_slice)
            tla.copy(v_l0, v_slice)
        tla.set_flag(ready)
        tla.wait_flag(ready)
        tla.mmad(output, p_l0, v_l0, init_c=first, unit_flag=0)
        tla.set_flag(free)


@tla.jit
def _csa_decode_schedule(
    sync,
    bufs,
    q,
    ori_kv,
    cmp_kv,
    ori_bt,
    cmp_bt,
    idx,
    topk,
    cu_q,
    cu_kv,
    cu_cmp,
    sinks,
    out,
    lse,
    stats,
    scale,
    task0,
    core_count,
    split_cores,
    q_prefix_free,
    guard,
    guard_data,
    guard_written,
    fixed: tla.Constexpr[bool],
    num_seqs: tla.Constexpr[int],
    idx_width: tla.Constexpr[int],
    ori_page: tla.Constexpr[int],
    ori_pages: tla.Constexpr[int],
    cmp_page: tla.Constexpr[int],
    cmp_pages: tla.Constexpr[int],
    has_sinks: tla.Constexpr[int],
    enable_lse: tla.Constexpr[bool],
    q_span: tla.Constexpr[int],
    uniform_ori: tla.Constexpr[bool],
    uniform_cmp: tla.Constexpr[bool],
):
    """Decode one full query per core, then a half query on each participating core."""
    m = Q_ROWS
    rows = Q_ROWS // 2
    dtype = q.ptr.dtype
    odd = task0 % 2 == 1
    split_query = core_count + task0 // 2
    takes_split = task0 < split_cores
    steps = 2 if takes_split else 1
    ori_span = cu_kv[1] if uniform_ori else 0
    cmp_span = cu_cmp[1] if uniform_cmp else 0

    # The two queues hold metadata for the preceding QK and PV tiles.
    qk_task = tla.as_numeric(0)
    qk_index = tla.as_numeric(0)
    qk_tiles = tla.as_numeric(1)
    qk_count = tla.as_numeric(0)
    pv_task = tla.as_numeric(0)
    pv_index = tla.as_numeric(0)
    pv_tiles = tla.as_numeric(1)
    pv_count = tla.as_numeric(0)
    done = tla.as_numeric(0)
    local_slot = tla.as_numeric(0)
    has_partial = tla.as_numeric(0)

    for step in tla.range(steps + 1):
        task = task0 if step == 0 else split_query
        tiles = tla.as_numeric(0)
        seq = tla.as_numeric(0)
        lo = tla.as_numeric(0)
        visible = tla.as_numeric(0)
        ori_tiles = tla.as_numeric(0)
        cmp_count = tla.as_numeric(0)
        cmp_skip = tla.as_numeric(0)
        if step < steps:
            row = task
            q_length = tla.as_numeric(0)
            if q_span:
                seq = task // q_span
                row = task - seq * q_span
                q_length = tla.as_numeric(q_span)
            else:
                for b in static_range(1, num_seqs):
                    if task >= cu_q[b]:
                        seq = tla.as_numeric(b)
                        row = task - cu_q[b]
                q_length = cu_q[seq + 1] - cu_q[seq]
            ori_length = ori_span if uniform_ori else cu_kv[seq + 1] - cu_kv[seq]
            cmp_length = cmp_span if uniform_cmp else cu_cmp[seq + 1] - cu_cmp[seq]
            # Dispatch limits this path to causal windows of at most 128 keys
            # and guarantees that every query row has a visible original key.
            hi = ori_length - q_length + row + 1
            visible = hi if hi < KV_BLOCK else KV_BLOCK
            lo = hi - visible
            ori_tiles = tla.as_numeric(1)
            count = topk[task]
            count = count if count > 0 else 0
            count = count if count < idx_width else idx_width
            cmp_count = count if count < cmp_length else cmp_length
            if step == 1:
                # Bias the split for cheaper contiguous original-row loads.
                key_budget = cmp_count - visible // 2
                left_count = (key_budget + 32) // 64 * 32
                left_count = left_count if left_count > 0 else 0
                left_count = left_count if left_count < cmp_count else cmp_count
                if odd:
                    cmp_skip = left_count
                    cmp_count = cmp_count - left_count
                    ori_tiles = tla.as_numeric(0)
                    has_partial = (cmp_count > 0).to(tla.Int32)
                else:
                    cmp_count = left_count
            tiles = ori_tiles + (cmp_count + KV_BLOCK - 1) // KV_BLOCK

        iterations = 2 if step == steps else tiles
        for i in tla.range(iterations):
            x = done + i
            current_count = tla.as_numeric(0)
            if i < tiles:
                if x == 0:
                    load_q(sync, bufs, q, task, m, dtype)
                slot = x % KV_SLOTS
                if i < ori_tiles:
                    current_count = visible
                    gather_decode_kv_to_l1(
                        sync,
                        bufs,
                        ori_kv,
                        idx,
                        0,
                        ori_bt,
                        seq,
                        ori_pages,
                        ori_page,
                        lo,
                        visible,
                        slot,
                        False,
                    )
                else:
                    column = (i - ori_tiles) * KV_BLOCK
                    current_count = cmp_count - column
                    current_count = (
                        current_count if current_count < KV_BLOCK else KV_BLOCK
                    )
                    gather_decode_kv_to_l1(
                        sync,
                        bufs,
                        cmp_kv,
                        idx,
                        task * idx_width + column + cmp_skip,
                        cmp_bt,
                        seq,
                        cmp_pages,
                        cmp_page,
                        0,
                        current_count,
                        slot,
                        True,
                    )

            if x >= 1 and i < tiles + 1:
                score_slot = (x - 1) % P_SLOTS
                sum_slot = score_slot if qk_index == 0 else 1 - score_slot
                first = qk_index == 0
                last = qk_index + 1 == qk_tiles
                before_split = takes_split and qk_task < core_count
                if first:
                    if fixed:
                        owns_sink = (tla.as_numeric(has_sinks) == 1) and not (
                            odd and qk_task >= core_count
                        )
                        _csa_init_fixed_decode(
                            sync, bufs, sinks, rows, score_slot, owns_sink, guard
                        )
                    else:
                        if qk_task >= core_count and odd and has_sinks == 1:
                            vec_init_from_sinks(
                                sync,
                                bufs,
                                sinks,
                                qk_task,
                                m,
                                rows,
                                dtype,
                                0,
                                0,
                                slot_sum=score_slot,
                                slot_max=score_slot,
                                clear_o=0,
                            )
                        else:
                            vec_init_from_sinks(
                                sync,
                                bufs,
                                sinks,
                                qk_task,
                                m,
                                rows,
                                dtype,
                                0,
                                has_sinks,
                                slot_sum=score_slot,
                                slot_max=score_slot,
                                clear_o=0,
                            )
                if fixed:
                    _csa_decode_fixed_vec1(
                        sync, bufs, scale, qk_count, score_slot, sum_slot, guard
                    )
                else:
                    process_vec1_csa(
                        sync,
                        bufs,
                        scale,
                        qk_count,
                        m,
                        rows,
                        dtype,
                        score_slot,
                        score_slot,
                        sum_slot,
                        stats,
                        0,
                        False,
                        use_weights=False,
                    )
                qk_width = (qk_count + 15) // 16 * 16
                iterate_bmm1(
                    sync,
                    bufs,
                    (x - 1) % KV_SLOTS,
                    m,
                    last,
                    first,
                    dtype,
                    slot_s=score_slot,
                    n_blk=qk_width,
                    direct_kv=True,
                    keep_q=True,
                    q_prefix_free=q_prefix_free,
                )
                if first:
                    if before_split:
                        _csa_prefetch_q_half(
                            sync, bufs, q, split_query, 0, False, q_prefix_free
                        )
                    else:
                        with tla.cube():
                            tla.wait_flag(q_prefix_free)
                if last and i < tiles:
                    _csa_prefetch_q_half(sync, bufs, q, task, 1, True)

            if fixed:
                if step == steps and i == 0:
                    _store_decode_guard(guard, guard_data, guard_written)

            if x >= 2:
                value_slot = (x - 2) % KV_SLOTS
                prob_slot = (x - 2) % P_SLOTS
                start = pv_index == 0
                end = pv_index + 1 == pv_tiles
                for seg in (0, 1, 2, 3):
                    if fixed:
                        if pv_count <= 112:
                            pv_width = (pv_count + 15) // 16 * 16
                            _csa_decode_fixed_pv(
                                sync,
                                bufs,
                                seg,
                                value_slot,
                                prob_slot,
                                pv_width,
                                start,
                                compact=True,
                            )
                        else:
                            _csa_decode_fixed_pv(
                                sync, bufs, seg, value_slot, prob_slot, KV_BLOCK, start
                            )
                    else:
                        if pv_count <= 112:
                            replay_width = (pv_count + 15) // 16 * 16
                            iterate_bmm2_seg(
                                sync,
                                bufs,
                                seg,
                                value_slot,
                                m,
                                dtype,
                                slot_p=prob_slot,
                                n_k=replay_width,
                                compact=True,
                                preload_v=True,
                            )
                        else:
                            iterate_bmm2_seg(
                                sync,
                                bufs,
                                seg,
                                value_slot,
                                m,
                                dtype,
                                slot_p=prob_slot,
                                preload_v=True,
                            )
                with tla.cube():
                    tla.cross_core_set_flag(sync.kv_l1_free, tla.pipes.MTE1, aiv_id=0)
                    tla.cross_core_set_flag(sync.kv_l1_free, tla.pipes.MTE1, aiv_id=1)
                write_output = end and pv_task < core_count
                if fixed:
                    if end:
                        iterate_bmm2_fixpipe(sync, bufs, m, start_output=True)
                        _csa_pv_update(
                            sync,
                            bufs,
                            out,
                            lse,
                            pv_task,
                            m,
                            rows,
                            prob_slot,
                            dtype,
                            True,
                            write_output,
                            False,
                            enable_lse,
                        )
                else:
                    iterate_bmm2_fixpipe(sync, bufs, m, start_output=start)
                    _csa_pv_update(
                        sync,
                        bufs,
                        out,
                        lse,
                        pv_task,
                        m,
                        rows,
                        prob_slot,
                        dtype,
                        start,
                        write_output,
                        False,
                        enable_lse,
                    )
                if end and pv_task >= core_count:
                    local_slot = prob_slot

            pv_task = qk_task
            pv_index = qk_index
            pv_tiles = qk_tiles
            pv_count = qk_count
            if i < tiles:
                qk_task = task
                qk_index = i
                qk_tiles = tiles
                qk_count = current_count
        done = done + tiles
    return local_slot, has_partial


@tla.kernel
def sparse_flash_mla_csa_bounded_decode_kernel(
    q: tla.Tensor,
    ori_kv: tla.Tensor,
    cmp_kv: tla.Tensor,
    ori_bt: tla.Tensor,
    cmp_bt: tla.Tensor,
    idx: tla.Tensor,
    topk: tla.Tensor,
    cu_q: tla.Tensor,
    cu_kv: tla.Tensor,
    cu_cmp: tla.Tensor,
    sinks: tla.Tensor,
    out: tla.Tensor,
    lse: tla.Tensor,
    partials: tla.Tensor,
    stats: tla.Tensor,
    scale: tla.Float32,
    num_seqs: tla.Constexpr[int],
    idx_width: tla.Constexpr[int],
    ori_page: tla.Constexpr[int],
    ori_pages: tla.Constexpr[int],
    cmp_page: tla.Constexpr[int],
    cmp_pages: tla.Constexpr[int],
    has_sinks: tla.Constexpr[int],
    enable_lse: tla.Constexpr[bool],
    q_span: tla.Constexpr[int],
    uniform_ori: tla.Constexpr[bool],
    uniform_cmp: tla.Constexpr[bool],
):
    rows = Q_ROWS // 2
    dtype = q.ptr.dtype
    configure_trace(1, False, False, KV_SLOTS, "csa")
    sync = declare_sync(
        need_idx_ring=True,
        need_lse_war=enable_lse,
        need_fused_dequant=True,
        direct_kv=True,
    )
    bufs = alloc_buffers(
        dtype,
        phy=KV_BLOCK // 2,
        preload=True,
        kv_pad=2,
        sink_cap=VL,
        kv_raw_pad=_DECODE_RAW_ROWS - KV_UNIT,
        stat_pad=VL - rows,
        keep_q=True,
    )
    init_sync(sync, pre_phy_vs=False, pre_l0c2=True, pre_idx_ring=False)
    init_sync_preload(sync)
    q_prefix_free = tla.flag("csa_q_prefix_free", tla.pipes.MTE1, tla.pipes.MTE2, id=4)
    with tla.cube():
        for _ in tla.range_constexpr(KV_SLOTS):
            tla.cross_core_set_flag(sync.kv_l1_free, tla.pipes.MTE1, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_l1_free, tla.pipes.MTE1, aiv_id=1)

    guard = tla.allocate(64, tla.Int32, tla.AddressSpace.ub, 256)
    guard_written = tla.cross_flag("decode_range_written", mode=4)
    guard_visible = tla.cross_flag("decode_range_visible", mode=4)
    with tla.vector():
        guard_view = tla.make_tensor(
            guard, tla.make_layout(tla.make_shape(64), tla.make_stride(1))
        )
        with tla.vec.func(mode="simd"):
            guard_view.store(tla.full(0, dtype=tla.Int32))
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)

    core_count = tla.arch.block_num()
    task = tla.arch.block_idx()
    splits = cu_q[num_seqs] - core_count
    split_cores = 2 * splits
    # Each vector subblock writes its own 32-byte region after the merge data.
    guard_data = stats.ptr + splits * (2 * Q_ROWS) + task * 16
    slot, has_partial = _csa_decode_schedule(
        sync,
        bufs,
        q,
        ori_kv,
        cmp_kv,
        ori_bt,
        cmp_bt,
        idx,
        topk,
        cu_q,
        cu_kv,
        cu_cmp,
        sinks,
        out,
        lse,
        stats,
        scale,
        task,
        core_count,
        split_cores,
        q_prefix_free,
        guard,
        guard_data,
        guard_written,
        True,
        num_seqs,
        idx_width,
        ori_page,
        ori_pages,
        cmp_page,
        cmp_pages,
        has_sinks,
        enable_lse,
        q_span,
        uniform_ori,
        uniform_cmp,
    )
    _wait_decode_guard(guard_data, guard_written, guard_visible)
    guard_gm = tla.make_tensor(
        tla.recast_ptr(guard_data, dtype=tla.Int32),
        tla.make_layout(tla.make_shape(16), tla.make_stride(1)),
    )
    if (guard_gm[0] | guard_gm[8]) != 0:
        slot, has_partial = _csa_decode_schedule(
            sync,
            bufs,
            q,
            ori_kv,
            cmp_kv,
            ori_bt,
            cmp_bt,
            idx,
            topk,
            cu_q,
            cu_kv,
            cu_cmp,
            sinks,
            out,
            lse,
            stats,
            scale,
            task,
            core_count,
            split_cores,
            q_prefix_free,
            guard,
            guard_data,
            guard_written,
            False,
            num_seqs,
            idx_width,
            ori_page,
            ori_pages,
            cmp_page,
            cmp_pages,
            has_sinks,
            enable_lse,
            q_span,
            uniform_ori,
            uniform_cmp,
        )

    pair = task // 2
    if task < split_cores and task % 2 == 1:
        _publish_decode_partition(
            sync, bufs, partials, stats, pair, rows, slot, has_partial == 1
        )
    _decode_vector_barrier(sync)
    if task < split_cores and task % 2 == 0:
        _merge_decode_partition(
            sync,
            bufs,
            partials,
            stats,
            out,
            lse,
            pair,
            core_count + pair,
            rows,
            slot,
            enable_lse,
            dtype,
        )


@tla.jit
def _csa_issue_folded_pv_tile(
    sync,
    bufs,
    out,
    lse,
    task,
    m,
    rows,
    dtype,
    k_slot,
    p_slot,
    width,
    compact: tla.Constexpr[bool],
    cached_ori,
    ori_free,
    first,
    last,
    enable_lse: tla.Constexpr[bool],
    free_right,
    ready_right,
    single_p_free,
    columns: tla.Constexpr[int] = D_CHUNK,
    first_v_loaded=False,
    wait_single_p=True,
    signal_single_p=True,
    early_lse: tla.Constexpr[bool] = False,
    terminal_output_event: tla.Constexpr[bool] = False,
):
    segs_per_half = HEAD_DIM // (2 * columns)
    for seg in static_range(HEAD_DIM // columns):
        iterate_bmm2_seg(
            sync,
            bufs,
            seg,
            k_slot,
            m,
            dtype,
            slot_p=p_slot,
            n_k=width,
            compact=compact,
            preload_v=True,
            cached_ori=cached_ori,
            ori_free=ori_free,
            split_fix=True,
            single_p_free=single_p_free,
            columns=columns,
            first_v_loaded=first_v_loaded,
            wait_single_p=wait_single_p,
            signal_single_p=signal_single_p,
        )
        if seg % segs_per_half == segs_per_half - 1:
            part = seg // segs_per_half
            free_flag = sync.o_free if part == 0 else free_right
            ready_flag = sync.o_ready if part == 0 else ready_right
            _csa_fix_pv_half(sync, bufs, m, rows, part, first, free_flag, ready_flag)
            _csa_consume_pv_half(
                sync,
                bufs,
                out,
                lse,
                task,
                m,
                rows,
                p_slot,
                dtype,
                part,
                first,
                last,
                enable_lse,
                free_flag,
                ready_flag,
                stream_output=single_p_free is not None,
                early_lse=early_lse,
                terminal_output_event=terminal_output_event,
            )


@tla.kernel
def sparse_flash_mla_csa_kernel(
    q: tla.Tensor,
    ori_kv: tla.Tensor,
    cmp_kv: tla.Tensor,
    ori_bt: tla.Tensor,
    cmp_bt: tla.Tensor,
    idx: tla.Tensor,
    topk: tla.Tensor,  # [T] int32
    cu_q: tla.Tensor,  # [B+1] int32
    cu_kv: tla.Tensor,  # [B+1] int32
    cu_cmp: tla.Tensor,  # [B+1] int32
    resid: tla.Tensor,  # [B] int32
    sinks: tla.Tensor,  # [H] fp32
    out: tla.Tensor,  # [T*H, D]
    lse: tla.Tensor,
    tiling: tla.Tensor,
    ws: tla.Tensor,
    cmp_phy: tla.Tensor,
    softmax_scale: tla.Float32,
    head_num: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    num_seqs: tla.Constexpr[int],
    idx_width: tla.Constexpr[int],
    ori_mask_mode: tla.Constexpr[int],
    ori_win_left: tla.Constexpr[int],
    ori_win_right: tla.Constexpr[int],
    cmp_mask_mode: tla.Constexpr[int],
    cmp_ratio: tla.Constexpr[int],
    ori_pa_block_size: tla.Constexpr[int],
    ori_pa_stride: tla.Constexpr[int],
    ori_pa_max_blocks: tla.Constexpr[int],
    ori_pa_pages: tla.Constexpr[int],
    cmp_pa_block_size: tla.Constexpr[int],
    cmp_pa_stride: tla.Constexpr[int],
    cmp_pa_max_blocks: tla.Constexpr[int],
    g_split: tla.Constexpr[int],
    has_sinks: tla.Constexpr[int],
    enable_lse: tla.Constexpr[int],
    topk_value_mode: tla.Constexpr[int],  # F5: 1=TOPK_INDEX, 2=TOPK_OFFSET
    kv_layout: tla.Constexpr[int],
    is_vec_s2phyaddr: tla.Constexpr[int],
    cmp_phy_width: tla.Constexpr[int],
    fused_dequant: tla.Constexpr[int] = 0,
    local_init: tla.Constexpr[bool] = False,
    direct_kv: tla.Constexpr[bool] = False,
    folded: tla.Constexpr[bool] = False,
    fold_page_shift: tla.Constexpr[int] = 0,
    split_decode: tla.Constexpr[bool] = False,
    paired_fold: tla.Constexpr[bool] = False,
    paired_prefix: tla.Constexpr[int] = 0,
    overlap_folded_pv: tla.Constexpr[bool] = False,
    compact_probabilities: tla.Constexpr[bool] = False,
    single_probability_bank: tla.Constexpr[bool] = False,
    contiguous_pairs: tla.Constexpr[bool] = False,
    keep_folded_q_suffix: tla.Constexpr[bool] = False,
    prefetch_folded_v: tla.Constexpr[bool] = False,
    linear_pair_rows: tla.Constexpr[int] = 0,
    linear_ori_rows: tla.Constexpr[int] = 0,
    linear_cmp_rows: tla.Constexpr[int] = 0,
    split_tail_queries: tla.Constexpr[bool] = False,
    center_pair_tail: tla.Constexpr[bool] = False,
    leading_tail_queries: tla.Constexpr[bool] = False,
    arithmetic_tail_setup: tla.Constexpr[bool] = False,
    trailing_tail_queries: tla.Constexpr[bool] = False,
) -> None:
    dtype_in = q.ptr.dtype
    m = head_num // g_split
    rows = m - m // 2
    half_g = head_num // g_split if head_num % g_split == 0 else head_num // g_split + 1

    configure_trace(g_split, False, False, KV_SLOTS, "csa")

    use_vec_gather = (is_vec_s2phyaddr == 1) and (kv_layout != 2)
    idx_ring_on = use_vec_gather and kv_layout != 2

    sync = declare_sync(
        need_idx_ring=idx_ring_on or direct_kv or folded,
        need_lse_war=enable_lse == 1,
        need_fused_dequant=(fused_dequant & 1) == 1,
        direct_kv=direct_kv,
    )
    bufs = alloc_buffers(
        dtype_in,
        phy=KV_BLOCK
        if folded
        else (KV_BLOCK // 2 if direct_kv else (cmp_phy_width if use_vec_gather else 0)),
        phy_bt=cmp_pa_max_blocks if use_vec_gather or folded else 0,
        preload=True,
        idx_ring=idx_ring_on,
        kv_pad=2 if direct_kv else 0,
        kv_raw_pad=_DECODE_RAW_ROWS - KV_UNIT if direct_kv else 0,
        stat_pad=VL - rows if folded or direct_kv else 0,
        keep_q=folded,
        gather32=folded,
        cache_ori=folded,
    )
    init_sync(
        sync,
        pre_phy_vs=use_vec_gather and not idx_ring_on,
        pre_l0c2=True,
        pre_idx_ring=idx_ring_on,
    )
    init_sync_preload(sync)
    o_free_right = (
        tla.cross_flag("csa_o_free_right", mode=4) if overlap_folded_pv else None
    )
    o_ready_right = (
        tla.cross_flag("csa_o_ready_right", mode=4) if overlap_folded_pv else None
    )
    single_p_free = (
        tla.flag("csa_single_p_free", tla.pipes.CUBE, tla.pipes.MTE1, id=6)
        if single_probability_bank
        else None
    )
    early_lse = (
        linear_pair_rows > 0
        and not split_tail_queries
        and linear_ori_rows - linear_pair_rows >= 3 * KV_BLOCK // 4
    )
    if single_probability_bank and not linear_pair_rows:
        with tla.cube():
            tla.set_flag(single_p_free)
    if overlap_folded_pv:
        with tla.vector():
            tla.cross_core_set_flag(o_free_right, tla.pipes.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(o_free_right, tla.pipes.VECTOR, aiv_id=1)
    q_prefix_free = (
        tla.flag("csa_q_prefix_free", tla.pipes.MTE1, tla.pipes.MTE2, id=4)
        if folded
        else None
    )
    ori_free = (
        tla.flag("csa_ori_free", tla.pipes.MTE1, tla.pipes.MTE2, id=5)
        if folded
        else None
    )
    if folded:
        with tla.cube():
            tla.set_flag(ori_free)
    phase0 = (
        tla.flag("csa_k_first_half0", tla.pipes.MTE2, tla.pipes.MTE1, id=6)
        if folded
        else None
    )
    phase1 = (
        tla.flag("csa_k_first_half1", tla.pipes.MTE2, tla.pipes.MTE1, id=7)
        if folded
        else None
    )
    # Slots 0 and 2 share one early event. Publish it after its previous receiver consumes it.
    phase0_free = (
        tla.flag("csa_k_first_half_free", tla.pipes.MTE1, tla.pipes.MTE2, id=6)
        if folded
        else None
    )
    if folded:
        with tla.cube():
            tla.set_flag(phase0_free)
    if folded:
        _csa_cache_initial_stats(sync, bufs, sinks, rows, has_sinks)
        with tla.vector():
            bt_view = tla.make_layout(
                tla.make_shape(cmp_pa_max_blocks), tla.make_stride(1)
            )
            tla.copy(
                tla.make_tensor(bufs.ub_phy_bt, bt_view),
                tla.make_tensor(cmp_bt.ptr, bt_view),
            )
            tla.set_flag(sync.idx_m2s0)
            tla.wait_flag(sync.idx_m2s0)
    if direct_kv:
        # One initial credit for each ring slot. MTE1 returns it after PV has
        # loaded all V segments, and MTE3 consumes it before reusing the slot.
        with tla.cube():
            for _ in tla.range_constexpr(KV_SLOTS):
                tla.cross_core_set_flag(sync.kv_l1_free, tla.pipes.MTE1, aiv_id=0)
                tla.cross_core_set_flag(sync.kv_l1_free, tla.pipes.MTE1, aiv_id=1)
    if not local_init:
        need_init = compute_need_init(cu_q, cu_kv, num_seqs, ori_mask_mode, 0)
        clean_output(
            sync,
            bufs,
            out,
            lse,
            need_init,
            cu_q[num_seqs] * head_num * head_dim,
            0,
            enable_lse,
            dtype_in,
            head_dim=head_dim,
        )

        init_aiv_arrive = tla.cross_flag("smla_init_aiv_arrive", mode=4)
        init_aic_ffts = tla.cross_flag("smla_init_aic_ffts", mode=0)
        init_aic_done = tla.cross_flag("smla_init_aic_done", mode=2)
        if need_init == 1:
            with tla.cube():
                tla.pipe_barrier(tla.pipes.ALL)
                tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=0)
                tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=1)
                tla.cross_core_set_flag(init_aic_ffts, tla.pipes.FIX)
                tla.cross_core_wait_flag(init_aic_ffts, tla.pipes.SCALAR)
                tla.cross_core_set_flag(init_aic_done, tla.pipes.SCALAR)
            with tla.vector():
                tla.pipe_barrier(tla.pipes.ALL)
                if tla.arch.sub_block_idx() == 0:
                    tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=0)
                else:
                    tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=1)
                tla.cross_core_wait_flag(init_aic_done, tla.pipes.SCALAR)

    total_tasks = (
        tla.as_numeric(linear_pair_rows)
        if linear_pair_rows
        else cu_q[num_seqs] * g_split
    )
    task0 = tla.arch.block_idx()
    q_bn = tla.arch.block_num()

    s2_budget = tla.as_numeric(0)
    c_one = 1
    if g_split == 2:
        s2_budget = tiling[c_one]

    # n_valid / n_load / kv_row0 / is_cmp / seq。
    r0_task = tla.as_numeric(0)
    r0_j = tla.as_numeric(0)
    r0_nt = tla.as_numeric(1)
    r0_nvalid = tla.as_numeric(0)
    r0_nload = tla.as_numeric(0)
    r0_kvrow0 = tla.as_numeric(0)
    r0_iscmp = tla.as_numeric(0)
    r0_seq = tla.as_numeric(0)
    r0_start = tla.as_numeric(0)
    r1_task = tla.as_numeric(0)
    r1_j = tla.as_numeric(0)
    r1_nt = tla.as_numeric(1)
    r1_nvalid = tla.as_numeric(0)
    r1_nload = tla.as_numeric(0)
    r1_kvrow0 = tla.as_numeric(0)
    r1_iscmp = tla.as_numeric(0)
    r1_seq = tla.as_numeric(0)
    r1_start = tla.as_numeric(0)
    r2_task = tla.as_numeric(0)
    r2_j = tla.as_numeric(0)
    r2_nt = tla.as_numeric(1)
    r2_nvalid = tla.as_numeric(0)
    r2_nload = tla.as_numeric(0)
    r2_kvrow0 = tla.as_numeric(0)
    r2_iscmp = tla.as_numeric(0)
    r2_seq = tla.as_numeric(0)
    r2_start = tla.as_numeric(0)
    r3_task = tla.as_numeric(0)
    r3_j = tla.as_numeric(0)
    r3_nt = tla.as_numeric(1)
    r3_nvalid = tla.as_numeric(0)
    r3_nload = tla.as_numeric(0)
    r3_kvrow0 = tla.as_numeric(0)
    r3_iscmp = tla.as_numeric(0)
    r3_seq = tla.as_numeric(0)
    r3_start = tla.as_numeric(0)

    g_done = tla.as_numeric(0)
    k_cursor = tla.as_numeric(0)
    r0_slot = tla.as_numeric(0)
    r1_slot = tla.as_numeric(0)
    r2_slot = tla.as_numeric(0)
    r3_slot = tla.as_numeric(0)
    r0_link = tla.as_numeric(0)
    r1_link = tla.as_numeric(0)
    r2_link = tla.as_numeric(0)
    r3_link = tla.as_numeric(0)
    ws_base = tla.arch.block_idx() * (KV_SLOTS * KV_STAGE_STRIDE)

    n_steps = (total_tasks - task0 + q_bn - 1) // q_bn
    if n_steps < 0:
        n_steps = tla.as_numeric(0)

    pair_chunk = 128 if contiguous_pairs else (32 if overlap_folded_pv else 8)
    stripe_pairs = pair_chunk * q_bn
    complete = tla.as_numeric(0)
    tail_start = tla.as_numeric(0)
    tail_end = tla.as_numeric(0)
    first_single = (
        task0 - (q_bn - paired_prefix)
        if trailing_tail_queries
        else (
            (
                task0
                if leading_tail_queries
                else (task0 * paired_prefix + q_bn // 2) // q_bn
            )
            if split_tail_queries
            else 0
        )
    )
    warm_steps = (
        (first_single >= 0).to(tla.Int32)
        if trailing_tail_queries
        else (
            (
                (task0 < paired_prefix).to(tla.Int32)
                if leading_tail_queries
                else ((task0 + 1) * paired_prefix + q_bn // 2) // q_bn - first_single
            )
            if split_tail_queries
            else (1 if paired_prefix else 0)
        )
    )
    linear_begin = tla.as_numeric(0)
    if trailing_tail_queries:
        # Give each core the same number of adjacent pairs. The last
        # few cores each take one of the remaining single queries.
        paired_queries = (linear_pair_rows - paired_prefix) // q_bn
        linear_begin = task0 * paired_queries
        n_steps = warm_steps + paired_queries
    elif split_tail_queries:
        paired_queries = (linear_pair_rows - paired_prefix) // q_bn
        linear_begin = paired_prefix + task0 * paired_queries
        n_steps = warm_steps + paired_queries
    elif linear_pair_rows:
        pairs = (linear_pair_rows + 1) // 2
        linear_begin = 2 * (task0 * pairs // q_bn)
        limit = 2 * ((task0 + 1) * pairs // q_bn)
        limit = limit if limit < linear_pair_rows else linear_pair_rows
        n_steps = limit - linear_begin
    elif paired_fold:
        # Keep a short group of adjacent pairs on each Cube, and distribute
        # the partial stripe across all of the Cube cores.
        pair_rows = total_tasks - paired_prefix
        complete = pair_rows // (2 * stripe_pairs)
        tail_pairs = (pair_rows - 2 * complete * stripe_pairs + 1) // 2
        if center_pair_tail:
            pair_bias = q_bn // 2
            tail_start = (
                complete * stripe_pairs + (task0 * tail_pairs + pair_bias) // q_bn
            )
            tail_end = (
                complete * stripe_pairs + ((task0 + 1) * tail_pairs + pair_bias) // q_bn
            )
        else:
            tail_start = complete * stripe_pairs + task0 * tail_pairs // q_bn
            tail_end = complete * stripe_pairs + (task0 + 1) * tail_pairs // q_bn
        tail_limit = 2 * tail_end if 2 * tail_end < pair_rows else pair_rows
        tail_rows = tail_limit - 2 * tail_start
        n_steps = (
            warm_steps + 2 * complete * pair_chunk + (tail_rows if tail_rows > 0 else 0)
        )

    cached_ori_end = tla.as_numeric(-1)
    local_slot = tla.as_numeric(0)
    second_partition_nonempty = tla.as_numeric(0)
    split_cores = 2 * (total_tasks - q_bn)
    if split_decode:
        n_steps = tla.as_numeric(1)
        if task0 < split_cores:
            n_steps = tla.as_numeric(2)
    qk_lag = 1 if direct_kv else CSA_PRELOAD_NUM - 1
    for step in tla.range(n_steps + 1):
        task = task0 + step * q_bn
        next_query_hint = total_tasks
        if trailing_tail_queries:
            paired_task = linear_begin + step
            paired_steps_this_core = n_steps - warm_steps
            single_query = linear_pair_rows - paired_prefix + first_single
            is_single = (step >= paired_steps_this_core).to(tla.Int32)
            task = paired_task + is_single * (single_query - paired_task)
        elif arithmetic_tail_setup:
            paired_task = linear_begin + step - warm_steps
            is_single = (step < warm_steps).to(tla.Int32)
            task = paired_task + is_single * (task0 - paired_task)
        elif split_tail_queries:
            task = linear_begin + step - warm_steps
            if step < warm_steps:
                task = first_single
            if step + 1 < n_steps:
                next_query_hint = linear_begin + step + 1 - warm_steps
        elif linear_pair_rows:
            task = linear_begin + step
            if step + 1 < n_steps:
                next_query_hint = task + 1
        elif paired_fold:
            task = _csa_blocked_query(
                step,
                task0,
                warm_steps,
                paired_prefix,
                pair_chunk,
                stripe_pairs,
                complete,
                tail_start,
            )
            if step + 1 < n_steps:
                next_query_hint = _csa_blocked_query(
                    step + 1,
                    task0,
                    warm_steps,
                    paired_prefix,
                    pair_chunk,
                    stripe_pairs,
                    complete,
                    tail_start,
                )
        if split_decode and step == 1:
            task = q_bn + task0 // 2

        n_tiles = tla.as_numeric(0)
        has_multiplicities = tla.as_numeric(0)
        pair_role = tla.as_numeric(0)
        seq = tla.as_numeric(0)
        tok = tla.as_numeric(0)
        rowq = tla.as_numeric(0)
        start_idx = tla.as_numeric(0)
        vis = tla.as_numeric(0)
        kv_len = tla.as_numeric(0)
        n_ori_tiles = tla.as_numeric(0)
        cmp_lines = tla.as_numeric(0)
        cmp_skip = tla.as_numeric(0)
        kv_row0 = tla.as_numeric(0)
        cmp_row0 = tla.as_numeric(0)

        if step < n_steps:
            if linear_pair_rows:
                tok = task
                end = tok + (linear_ori_rows - linear_pair_rows + 1)
                vis = end if end < KV_BLOCK else KV_BLOCK
                start_idx = end - vis
                n_ori_tiles = tla.as_numeric(1)
                marked = topk[tok * 8]
                has_multiplicities = (marked > 0).to(tla.Int32)
                sparse_count = marked if marked >= 0 else -marked
                if trailing_tail_queries:
                    matched = (sparse_count > idx_width).to(tla.Int32)
                    pair_role = matched * (1 + step % 2)
                    cmp_lines = sparse_count - matched * (idx_width + 1)
                elif arithmetic_tail_setup:
                    matched = (sparse_count > idx_width).to(tla.Int32)
                    cmp_lines = sparse_count - matched * (idx_width + 1)
                    pair_role = matched * (1 + (step - warm_steps) % 2)
                else:
                    if sparse_count > idx_width:
                        sparse_count = sparse_count - idx_width - 1
                        pair_step = step - warm_steps if split_tail_queries else step
                        pair_role = 1 + pair_step % 2
                    cmp_lines = (
                        sparse_count
                        if sparse_count < linear_cmp_rows
                        else linear_cmp_rows
                    )
                n_tiles = 1 + (cmp_lines + KV_BLOCK - 1) // KV_BLOCK
            else:
                tok = task // g_split
                rowq = tok
                for bb in static_range(1, num_seqs):
                    if tok >= cu_q[bb]:
                        seq = bb
                        rowq = tok - cu_q[bb]
                q_len = cu_q[seq + 1] - cu_q[seq]
                kv_len = cu_kv[seq + 1] - cu_kv[seq]
                cmp_len = cu_cmp[seq + 1] - cu_cmp[seq]

                #      start = clip(row - pre, 0, kv)、end = clip(row + next + 1, 0, kv) ----
                causal_offset = kv_len - q_len
                next_tok = kv_len
                pre_tok = q_len
                if ori_mask_mode == 3:
                    next_tok = causal_offset
                    pre_tok = q_len
                elif ori_mask_mode == 4:
                    if ori_win_left == -1:
                        pre_tok = q_len
                    else:
                        pre_tok = ori_win_left - causal_offset
                    if ori_win_right == -1:
                        next_tok = kv_len
                    else:
                        next_tok = causal_offset + ori_win_right
                    if pre_tok > q_len:
                        pre_tok = q_len
                end_idx = rowq + next_tok + 1
                if end_idx < 0:
                    end_idx = tla.as_numeric(0)
                if end_idx > kv_len:
                    end_idx = kv_len
                start_idx = rowq - pre_tok
                if start_idx < 0:
                    start_idx = tla.as_numeric(0)
                if start_idx > kv_len:
                    start_idx = kv_len
                vis = end_idx - start_idx
                if vis < 0:
                    vis = tla.as_numeric(0)
                n_ori_tiles = (vis + KV_BLOCK - 1) // KV_BLOCK

                sparse_count = topk[tok * 8] if paired_fold else topk[tok]
                if folded:
                    has_multiplicities = (sparse_count > 0).to(tla.Int32)
                    if sparse_count < 0:
                        sparse_count = -sparse_count
                    if paired_fold and sparse_count > idx_width:
                        sparse_count = sparse_count - idx_width - 1
                        pair_role = 1 + (step - (1 if paired_prefix else 0)) % 2
                if sparse_count > idx_width:
                    sparse_count = tla.as_numeric(idx_width)
                if sparse_count < 0:
                    sparse_count = tla.as_numeric(0)
                next_cmp = cmp_len * cmp_ratio
                if cmp_mask_mode != 0:
                    next_cmp = cmp_len * cmp_ratio + resid[seq] - q_len
                cmp_end = (rowq + 1 + next_cmp) // cmp_ratio
                if cmp_end < 0:
                    cmp_end = tla.as_numeric(0)
                if cmp_end > cmp_len:
                    cmp_end = cmp_len
                cmp_lines = cmp_end
                if cmp_lines > sparse_count:
                    cmp_lines = sparse_count

                if split_decode and step == 1:
                    # Dense window pages are loaded in larger bursts than sparse KV rows.
                    # Bias the partition toward the window owner and align the sparse DMA offset.
                    key_budget = cmp_lines - vis // 2
                    half_cmp = (key_budget + 32) // 64 * 32
                    half_cmp = half_cmp if half_cmp > 0 else 0
                    half_cmp = half_cmp if half_cmp < cmp_lines else cmp_lines
                    if task0 % 2 == 0:
                        cmp_lines = half_cmp
                    else:
                        cmp_skip = half_cmp
                        cmp_lines = cmp_lines - half_cmp if cmp_lines > half_cmp else 0
                        n_ori_tiles = tla.as_numeric(0)
                        second_partition_nonempty = (cmp_lines > 0).to(tla.Int32)
                n_tiles = n_ori_tiles + (cmp_lines + KV_BLOCK - 1) // KV_BLOCK
                kv_row0 = cu_kv[seq] + start_idx
                cmp_row0 = cu_cmp[seq]
            if local_init:
                if n_tiles == 0:
                    with tla.vector():
                        head_base = task * m + tla.arch.sub_block_idx() * rows
                        init_output_gm(
                            sync,
                            bufs,
                            out,
                            head_base * HEAD_DIM,
                            rows * HEAD_DIM,
                            INIT_POP_ELE,
                            dtype_in,
                        )
                        if enable_lse == 1:
                            init_output_gm(
                                sync,
                                bufs,
                                lse,
                                head_base,
                                rows,
                                INIT_POP_ELE,
                                tla.Float32,
                                use_lse_buf=1,
                            )

        n_body = n_tiles
        if step == n_steps:
            if g_done > 0:
                n_body = n_tiles + qk_lag + 1

        for i in tla.range(n_body):
            x = g_done + i

            # ================================================================
            # ================================================================
            if i < n_tiles:
                j0 = i
                nv0 = tla.as_numeric(0)
                nl0 = tla.as_numeric(0)
                is_ori_tile = j0 < n_ori_tiles
                cmp_tile = j0 - n_ori_tiles
                ori_tile = j0
                if folded and pair_role == 2:
                    is_ori_tile = j0 > 0 and j0 <= n_ori_tiles
                    cmp_tile = 0 if j0 == 0 else j0 - n_ori_tiles
                    ori_tile = j0 - 1
                if is_ori_tile:
                    nv0 = vis - ori_tile * KV_BLOCK
                    if nv0 > KV_BLOCK:
                        nv0 = tla.as_numeric(KV_BLOCK)
                    nl0 = vis - ori_tile * KV_BLOCK
                    if nl0 < 0:
                        nl0 = tla.as_numeric(0)
                    if nl0 > KV_BLOCK:
                        nl0 = tla.as_numeric(KV_BLOCK)
                else:
                    nv0 = cmp_lines - cmp_tile * KV_BLOCK
                    if folded and pair_role == 1:
                        nv0 = (
                            tla.as_numeric(KV_BLOCK)
                            if j0 + 1 == n_tiles
                            else nv0 - KV_BLOCK
                        )
                    if nv0 > KV_BLOCK:
                        nv0 = tla.as_numeric(KV_BLOCK)
                    nl0 = nv0
                    if nl0 < 0:
                        nl0 = tla.as_numeric(0)
                r0_task = task
                r0_j = j0
                r0_nt = n_tiles
                r0_nvalid = nv0
                r0_nload = nl0
                # The dense-row offset is unused on this paged folded route.
                if not arithmetic_tail_setup and not trailing_tail_queries:
                    r0_kvrow0 = next_query_hint if paired_fold else kv_row0
                r0_start = start_idx
                r0_seq = seq
                r0_iscmp = tla.as_numeric(0)
                # Hold the original panel only when a CMP tile follows the
                # original window. The next query can then wait for its release.
                if (
                    folded
                    and is_ori_tile
                    and cmp_lines > 0
                    and (pair_role != 2 or cmp_lines > KV_BLOCK)
                ):
                    r0_iscmp = tla.as_numeric(-1)
                if folded:
                    r0_link = tla.as_numeric(0)
                    if pair_role == 1 and j0 + 1 == n_tiles:
                        r0_link = tla.as_numeric(1)
                    if pair_role == 2 and j0 == 0:
                        r0_link = tla.as_numeric(2)
                        r0_slot = (k_cursor + KV_SLOTS - 1) % KV_SLOTS
                    else:
                        r0_slot = k_cursor
                        k_cursor = (k_cursor + 1) % KV_SLOTS
                if not is_ori_tile:
                    r0_iscmp = tla.as_numeric(1)
                    if folded:
                        # Each log-weight row matches the original column order
                        # of its compacted indices. -1 skips the weights for distinct keys.
                        r0_start = (
                            cmp_tile * KV_BLOCK if has_multiplicities == 1 else -1
                        )
                        r0_nload = KV_BLOCK if x < KV_SLOTS else nv0

                if direct_kv:
                    if x == 0:
                        q_first = 2 * ((task - task0) // q_bn)
                        load_q(
                            sync,
                            bufs,
                            q,
                            task,
                            m,
                            dtype_in,
                            slot_a=q_first % 3,
                            slot_b=(q_first + 1) % 3,
                        )
                    slot0 = x % KV_SLOTS
                    if j0 >= n_ori_tiles:
                        idx0 = (
                            tok * idx_width + (j0 - n_ori_tiles) * KV_BLOCK + cmp_skip
                        )
                        gather_decode_kv_to_l1(
                            sync,
                            bufs,
                            cmp_kv,
                            idx,
                            idx0,
                            cmp_bt,
                            seq,
                            cmp_pa_max_blocks,
                            cmp_pa_block_size,
                            0,
                            nl0,
                            slot0,
                            True,
                        )
                    else:
                        gather_decode_kv_to_l1(
                            sync,
                            bufs,
                            ori_kv,
                            idx,
                            0,
                            ori_bt,
                            seq,
                            ori_pa_max_blocks,
                            ori_pa_block_size,
                            start_idx + j0 * KV_BLOCK,
                            nl0,
                            slot0,
                            False,
                        )
                elif not is_ori_tile:
                    base_line0 = cmp_tile * KV_BLOCK
                    slot0 = r0_slot if folded else x % KV_SLOTS
                    islot = x - (x // 2) * 2
                    if kv_layout == 2 and (fused_dequant & 1) == 1:
                        gather_kv_rows_pa_dequant(
                            sync,
                            bufs,
                            ws,
                            cmp_kv,
                            idx,
                            tok,
                            idx_width,
                            base_line0,
                            nl0,
                            slot0,
                            KV_STAGE_STRIDE,
                            ws_base,
                            cmp_bt,
                            seq,
                            cmp_pa_max_blocks,
                            cmp_pa_block_size,
                            cmp_pa_stride,
                            tla.as_numeric(1),
                            tla.as_numeric(0),
                            topk_value_mode=topk_value_mode,
                        )
                    elif folded:
                        if r0_link != 2:
                            gather_folded_to_workspace(
                                sync,
                                bufs,
                                ws,
                                cmp_kv,
                                idx,
                                tok * idx_width + base_line0,
                                nv0,
                                r0_nload,
                                slot0,
                                ws_base,
                                fold_page_shift,
                                alignment=2 if center_pair_tail else 8,
                            )
                    elif kv_layout == 2 and not use_vec_gather:
                        gather_kv_rows_pa_pairwise(
                            sync,
                            bufs,
                            ws,
                            cmp_kv,
                            idx,
                            tok,
                            idx_width,
                            base_line0,
                            nl0,
                            slot0,
                            KV_STAGE_STRIDE,
                            ws_base,
                            block_table=cmp_bt,
                            bt_seq=seq,
                            bt_max=cmp_pa_max_blocks,
                            pa_block_size=cmp_pa_block_size,
                            pa_stride=cmp_pa_stride,
                            topk_value_mode=topk_value_mode,
                        )
                    elif is_vec_s2phyaddr == 1:
                        gather_kv_rows_ring(
                            sync,
                            bufs,
                            ws,
                            cmp_kv,
                            idx,
                            tok,
                            idx_width,
                            cmp_row0,
                            base_line0,
                            nl0,
                            slot0,
                            KV_STAGE_STRIDE,
                            islot,
                            ws_base,
                        )
                    else:
                        gather_kv_rows(
                            sync,
                            bufs,
                            ws,
                            cmp_kv,
                            idx,
                            tok,
                            idx_width,
                            base_line0,
                            nl0,
                            cmp_row0,
                            slot0,
                            KV_STAGE_STRIDE,
                            ws_base,
                            block_table=cmp_bt,
                            bt_seq=seq,
                            bt_max=cmp_pa_max_blocks,
                            pa_block_size=cmp_pa_block_size,
                            pa_stride=cmp_pa_stride,
                            topk_value_mode=topk_value_mode,
                        )
                elif (fused_dequant & 2) == 2:
                    slot0 = x - (x // KV_SLOTS) * KV_SLOTS
                    gather_kv_rows_pa_dequant(
                        sync,
                        bufs,
                        ws,
                        ori_kv,
                        idx,
                        tok,
                        idx_width,
                        0,
                        nl0,
                        slot0,
                        KV_STAGE_STRIDE,
                        ws_base,
                        ori_bt,
                        seq,
                        ori_pa_max_blocks,
                        ori_pa_block_size,
                        ori_pa_stride,
                        tla.as_numeric(0),
                        start_idx + j0 * KV_BLOCK,
                        topk_value_mode=topk_value_mode,
                    )

            # ================================================================
            # OT IterateLoadQK(runInfo[(taskId+3)%4], isFirstLoop)（:628），
            # ================================================================
            if not direct_kv and x >= 1:
                if i < n_tiles + 1:
                    xg1 = x - 1
                    slot_k1 = r1_slot if folded else xg1 % KV_SLOTS
                    if xg1 == 0:
                        qn1 = (r1_task - task0) // q_bn
                        q2n1 = qn1 + qn1
                        qa1 = q2n1 - (q2n1 // 3) * 3
                        q2n1b = q2n1 + 1
                        qb1 = q2n1b - (q2n1b // 3) * 3
                        load_q(
                            sync,
                            bufs,
                            q,
                            r1_task,
                            m,
                            dtype_in,
                            slot_a=0 if folded else qa1,
                            slot_b=1 if folded else qb1,
                        )
                    if g_split == 2:
                        if s2_budget > 0:
                            s2_budget = s2_budget - 1
                            split_g_mte2_pair(sync)
                    if (fused_dequant & 2) == 2:
                        wait_kv_stage(sync)
                        load_k_ptr(
                            sync,
                            bufs,
                            ws.ptr + ws_base + slot_k1 * KV_STAGE_STRIDE,
                            r1_nload,
                            slot_k1,
                            dtype_in,
                        )
                    elif r1_iscmp == 1:
                        if not folded or r1_link != 2:
                            wait_kv_stage(sync)
                            gm = ws.ptr + ws_base + slot_k1 * KV_STAGE_STRIDE
                            if folded and r1_nvalid > 64:
                                _csa_load_k_head_halves(
                                    sync,
                                    bufs,
                                    gm,
                                    r1_nload,
                                    slot_k1,
                                    dtype_in,
                                    phase0,
                                    phase1,
                                    phase0_free,
                                    256 if overlap_folded_pv else 384,
                                    split_columns=center_pair_tail,
                                    split_prefix=trailing_tail_queries,
                                )
                            else:
                                load_k_ptr(sync, bufs, gm, r1_nload, slot_k1, dtype_in)
                    else:
                        if folded and r1_iscmp == -1:
                            _csa_load_cyclic_ori(
                                sync,
                                bufs,
                                ori_kv,
                                ori_bt,
                                r1_start,
                                cached_ori_end,
                                slot_k1,
                                dtype_in,
                                ori_pa_block_size,
                                ori_pa_stride,
                                ori_free,
                            )
                            cached_ori_end = r1_start + KV_BLOCK
                        elif folded:
                            _csa_load_full_ori(
                                sync,
                                bufs,
                                ori_kv,
                                ori_bt,
                                r1_start,
                                slot_k1,
                                dtype_in,
                                ori_pa_block_size,
                                ori_pa_stride,
                            )
                        elif ori_pa_block_size > 0:
                            load_k_pa(
                                sync,
                                bufs,
                                ori_kv,
                                ori_bt,
                                r1_seq,
                                r1_start + r1_j * KV_BLOCK,
                                r1_nload,
                                slot_k1,
                                dtype_in,
                                ori_pa_block_size,
                                ori_pa_stride,
                                ori_pa_max_blocks,
                                ori_pa_pages,
                                fill_tail=(xg1 < KV_SLOTS).to(tla.Int32)
                                if folded
                                else tla.as_numeric(0),
                            )
                        else:
                            load_k(
                                sync,
                                bufs,
                                ori_kv,
                                r1_kvrow0,
                                r1_j,
                                r1_nload,
                                slot_k1,
                                dtype_in,
                            )
                else:
                    if g_split == 2:
                        if s2_budget > 0:
                            s2_budget = s2_budget - 1
                            split_g_mte2_pair(sync)

            # ================================================================
            # ================================================================
            if x >= qk_lag:
                if i < n_tiles + qk_lag:
                    xg2 = x - qk_lag
                    slot_s2 = xg2 - (xg2 // P_SLOTS) * P_SLOTS
                    slot_k2 = r2_slot if folded else xg2 % KV_SLOTS
                    slot_sum_rd = slot_s2
                    if r2_j > 0:
                        slot_sum_rd = 1 - slot_s2
                    q_nta2 = (
                        (r2_task >= q_bn).to(tla.Int32)
                        if split_decode
                        else (r2_task - task0) // q_bn
                    )
                    q_two2 = q_nta2 + q_nta2
                    qa2 = q_two2 - (q_two2 // 3) * 3
                    q_two2b = q_two2 + 1
                    qb2 = q_two2b - (q_two2b // 3) * 3
                    if r2_j == 0 and not folded:
                        goff2 = (r2_task - (r2_task // g_split) * g_split) * m
                        if (
                            split_decode
                            and has_sinks == 1
                            and r2_task >= q_bn
                            and task0 % 2 == 1
                        ):
                            vec_init_from_sinks(
                                sync,
                                bufs,
                                sinks,
                                r2_task,
                                m,
                                rows,
                                dtype_in,
                                goff2,
                                0,
                                slot_sum=slot_s2,
                                slot_max=slot_s2,
                                clear_o=0,
                            )
                        else:
                            vec_init_from_sinks(
                                sync,
                                bufs,
                                sinks,
                                r2_task,
                                m,
                                rows,
                                dtype_in,
                                goff2,
                                has_sinks,
                                slot_sum=slot_s2,
                                slot_max=slot_s2,
                                clear_o=0,
                            )
                    if folded:
                        process_vec1_csa(
                            sync,
                            bufs,
                            softmax_scale,
                            r2_nvalid,
                            m,
                            rows,
                            dtype_in,
                            slot_s2,
                            slot_s2,
                            slot_sum_rd,
                            cmp_phy,
                            r2_task * idx_width
                            + (r2_start if r2_start >= 0 else tla.as_numeric(0)),
                            r2_iscmp == 1 and r2_start >= 0,
                            cached_initial=True,
                            first_tile=r2_j == 0,
                            compact_probabilities=compact_probabilities,
                        )
                    elif direct_kv:
                        process_vec1_csa(
                            sync,
                            bufs,
                            softmax_scale,
                            r2_nvalid,
                            m,
                            rows,
                            dtype_in,
                            slot_s2,
                            slot_s2,
                            slot_sum_rd,
                            cmp_phy,
                            0,
                            False,
                            use_weights=False,
                        )
                    else:
                        process_vec1(
                            sync,
                            bufs,
                            softmax_scale,
                            r2_nvalid,
                            m,
                            rows,
                            dtype_in,
                            slot_s=slot_s2,
                            slot_p=slot_s2,
                            preload=1,
                            slot_sum_rd=slot_sum_rd,
                            slot_max_rd=slot_sum_rd,
                        )
                    phased = (
                        folded and r2_iscmp == 1 and r2_nvalid > 64 and r2_link != 2
                    )
                    pv0 = (
                        (
                            x >= qk_lag + 1,
                            r3_slot,
                            r3_nvalid,
                            r3_iscmp,
                            ori_free,
                            single_probability_bank,
                        )
                        if prefetch_folded_v
                        else None
                    )
                    if (folded or direct_kv) and r2_nvalid <= 112:
                        width_2 = (r2_nvalid + 15) // 16 * 16
                        iterate_bmm1(
                            sync,
                            bufs,
                            slot_k2,
                            m,
                            r2_j + 1 == r2_nt,
                            r2_j == 0,
                            dtype_in,
                            slot_s=slot_s2,
                            q_slot_a=0 if folded else qa2,
                            q_slot_b=1 if folded else qb2,
                            n_blk=width_2,
                            direct_kv=direct_kv,
                            phase0_free=phase0_free,
                            phase0=phase0,
                            phase1=phase1,
                            phased=phased,
                            keep_q=folded,
                            reuse_k=folded and r2_link == 2,
                            q_prefix_free=q_prefix_free,
                            cached_ori=folded and r2_iscmp == -1,
                            unit_sync=overlap_folded_pv,
                            pv0=pv0,
                            keep_q_suffix=keep_folded_q_suffix,
                        )
                    else:
                        iterate_bmm1(
                            sync,
                            bufs,
                            slot_k2,
                            m,
                            r2_j + 1 == r2_nt,
                            r2_j == 0,
                            dtype_in,
                            slot_s=slot_s2,
                            q_slot_a=0 if folded else qa2,
                            q_slot_b=1 if folded else qb2,
                            direct_kv=direct_kv,
                            phase0_free=phase0_free,
                            phase0=phase0,
                            phase1=phase1,
                            phased=phased,
                            keep_q=folded,
                            reuse_k=folded and r2_link == 2,
                            q_prefix_free=q_prefix_free,
                            cached_ori=folded and r2_iscmp == -1,
                            unit_sync=overlap_folded_pv,
                            pv0=pv0,
                            keep_q_suffix=keep_folded_q_suffix,
                        )
                    following_task = total_tasks
                    if trailing_tail_queries:
                        last_paired_query = linear_begin + n_steps - warm_steps - 1
                        following_single = total_tasks - paired_prefix + first_single
                        following_task = (
                            r2_task + 1
                            if r2_task < last_paired_query
                            else (
                                following_single
                                if warm_steps > 0 and r2_task == last_paired_query
                                else total_tasks
                            )
                        )
                    else:
                        following_task = (
                            (linear_begin if r2_task < paired_prefix else r2_task + 1)
                            if arithmetic_tail_setup
                            else (r2_kvrow0 if paired_fold else r2_task + q_bn)
                        )
                    prefetch_limit = (
                        linear_begin + n_steps - warm_steps
                        if arithmetic_tail_setup
                        else total_tasks
                    )
                    if folded and r2_j == 0 and following_task < prefetch_limit:
                        _csa_prefetch_q_half(
                            sync, bufs, q, following_task, 0, False, q_prefix_free
                        )
                    if r2_j + 1 == r2_nt:
                        if i < n_tiles + qk_lag - 1:
                            next_task = r1_task if split_decode else following_task
                            qn_p = q_nta2 + 1
                            if direct_kv:
                                next_task = r0_task
                                if not split_decode:
                                    qn_p = (next_task - task0) // q_bn
                            q2n_p = qn_p + qn_p
                            qap = q2n_p - (q2n_p // 3) * 3
                            q2np_b = q2n_p + 1
                            qbp = q2np_b - (q2np_b // 3) * 3
                            if folded:
                                _csa_prefetch_q_half(sync, bufs, q, next_task, 1, True)
                            else:
                                load_q(
                                    sync,
                                    bufs,
                                    q,
                                    next_task,
                                    m,
                                    dtype_in,
                                    slot_a=qap,
                                    slot_b=qbp,
                                )

            # ================================================================
            # ================================================================
            if x >= qk_lag + 1:
                xg3 = x - (qk_lag + 1)
                slot_k3 = r3_slot if folded else xg3 % KV_SLOTS
                slot_p3 = xg3 - (xg3 // P_SLOTS) * P_SLOTS
                if not folded:
                    wait_p_ready(sync)
                first_pv = r3_j == 0
                finish_pv = (r3_j + 1 == r3_nt) and (not split_decode or r3_task < q_bn)
                if overlap_folded_pv:
                    first_v_loaded = (
                        (i < n_tiles + qk_lag) if prefetch_folded_v else False
                    )
                    wait_single_p = True
                    signal_single_p = True
                    if linear_pair_rows:
                        drain_pair = step == n_steps and g_done > 1
                        wait_single_p = drain_pair and i == n_body - 1
                        signal_single_p = drain_pair and i == n_body - 2
                    if single_probability_bank and r3_iscmp == 1 and r3_nvalid <= 64:
                        narrow_width = (r3_nvalid + 15) // 16 * 16
                        _csa_issue_folded_pv_tile(
                            sync,
                            bufs,
                            out,
                            lse,
                            r3_task,
                            m,
                            rows,
                            dtype_in,
                            slot_k3,
                            slot_p3,
                            narrow_width,
                            True,
                            False,
                            ori_free,
                            first_pv,
                            finish_pv,
                            enable_lse == 1,
                            o_free_right,
                            o_ready_right,
                            single_p_free,
                            first_v_loaded=first_v_loaded,
                            wait_single_p=wait_single_p,
                            signal_single_p=signal_single_p,
                            columns=HEAD_DIM // 2,
                            early_lse=early_lse,
                            terminal_output_event=center_pair_tail,
                        )
                    elif r3_nvalid > 112:
                        _csa_issue_folded_pv_tile(
                            sync,
                            bufs,
                            out,
                            lse,
                            r3_task,
                            m,
                            rows,
                            dtype_in,
                            slot_k3,
                            slot_p3,
                            KV_BLOCK,
                            False,
                            r3_iscmp == -1,
                            ori_free,
                            first_pv,
                            finish_pv,
                            enable_lse == 1,
                            o_free_right,
                            o_ready_right,
                            single_p_free,
                            first_v_loaded=first_v_loaded,
                            wait_single_p=wait_single_p,
                            signal_single_p=signal_single_p,
                            early_lse=early_lse,
                            terminal_output_event=center_pair_tail,
                        )
                    elif r3_nvalid <= 16:
                        _csa_issue_folded_pv_tile(
                            sync,
                            bufs,
                            out,
                            lse,
                            r3_task,
                            m,
                            rows,
                            dtype_in,
                            slot_k3,
                            slot_p3,
                            16,
                            True,
                            r3_iscmp == -1,
                            ori_free,
                            first_pv,
                            finish_pv,
                            enable_lse == 1,
                            o_free_right,
                            o_ready_right,
                            single_p_free,
                            first_v_loaded=first_v_loaded,
                            wait_single_p=wait_single_p,
                            signal_single_p=signal_single_p,
                            early_lse=early_lse,
                            terminal_output_event=center_pair_tail,
                        )
                    elif r3_nvalid <= 32:
                        _csa_issue_folded_pv_tile(
                            sync,
                            bufs,
                            out,
                            lse,
                            r3_task,
                            m,
                            rows,
                            dtype_in,
                            slot_k3,
                            slot_p3,
                            32,
                            True,
                            r3_iscmp == -1,
                            ori_free,
                            first_pv,
                            finish_pv,
                            enable_lse == 1,
                            o_free_right,
                            o_ready_right,
                            single_p_free,
                            first_v_loaded=first_v_loaded,
                            wait_single_p=wait_single_p,
                            signal_single_p=signal_single_p,
                            early_lse=early_lse,
                            terminal_output_event=center_pair_tail,
                        )
                    else:
                        folded_width = (r3_nvalid + 15) // 16 * 16
                        _csa_issue_folded_pv_tile(
                            sync,
                            bufs,
                            out,
                            lse,
                            r3_task,
                            m,
                            rows,
                            dtype_in,
                            slot_k3,
                            slot_p3,
                            folded_width,
                            True,
                            r3_iscmp == -1,
                            ori_free,
                            first_pv,
                            finish_pv,
                            enable_lse == 1,
                            o_free_right,
                            o_ready_right,
                            single_p_free,
                            first_v_loaded=first_v_loaded,
                            wait_single_p=wait_single_p,
                            signal_single_p=signal_single_p,
                            early_lse=early_lse,
                            terminal_output_event=center_pair_tail,
                        )
                else:
                    for ordinary_pv_segment in (0, 1, 2, 3):
                        if folded and r3_nvalid <= 16:
                            iterate_bmm2_seg(
                                sync,
                                bufs,
                                ordinary_pv_segment,
                                slot_k3,
                                m,
                                dtype_in,
                                slot_p=slot_p3,
                                n_k=16,
                                compact=True,
                                preload_v=True,
                                cached_ori=r3_iscmp == -1,
                                ori_free=ori_free,
                            )
                        elif folded and r3_nvalid <= 32:
                            iterate_bmm2_seg(
                                sync,
                                bufs,
                                ordinary_pv_segment,
                                slot_k3,
                                m,
                                dtype_in,
                                slot_p=slot_p3,
                                n_k=32,
                                compact=True,
                                preload_v=True,
                                cached_ori=r3_iscmp == -1,
                                ori_free=ori_free,
                            )
                        elif (folded or direct_kv) and r3_nvalid <= 112:
                            width_3 = (r3_nvalid + 15) // 16 * 16
                            iterate_bmm2_seg(
                                sync,
                                bufs,
                                ordinary_pv_segment,
                                slot_k3,
                                m,
                                dtype_in,
                                slot_p=slot_p3,
                                n_k=width_3,
                                compact=True,
                                preload_v=folded,
                                cached_ori=folded and r3_iscmp == -1,
                                ori_free=ori_free,
                            )
                        else:
                            iterate_bmm2_seg(
                                sync,
                                bufs,
                                ordinary_pv_segment,
                                slot_k3,
                                m,
                                dtype_in,
                                slot_p=slot_p3,
                                preload_v=folded,
                                cached_ori=folded and r3_iscmp == -1,
                                ori_free=ori_free,
                            )
                if direct_kv:
                    with tla.cube():
                        tla.cross_core_set_flag(
                            sync.kv_l1_free, tla.pipes.MTE1, aiv_id=0
                        )
                        tla.cross_core_set_flag(
                            sync.kv_l1_free, tla.pipes.MTE1, aiv_id=1
                        )
                if folded or direct_kv:
                    if folded and r3_link != 1:
                        release_k(sync, slot_k3)
                    if not overlap_folded_pv:
                        iterate_bmm2_fixpipe(sync, bufs, m, start_output=first_pv)
                        _csa_pv_update(
                            sync,
                            bufs,
                            out,
                            lse,
                            r3_task,
                            m,
                            rows,
                            slot_p3,
                            dtype_in,
                            first_pv,
                            finish_pv,
                            folded,
                            enable_lse == 1,
                        )
                else:
                    iterate_bmm2_fixpipe(sync, bufs, m)
                    # Bool predicate, got i32）
                    fresh_v3 = r3_j == 0
                    for seg in (0, 1, 2, 3):
                        process_vec2_seg(
                            sync,
                            bufs,
                            seg,
                            m,
                            rows,
                            dtype_in,
                            fresh_v3,
                            slot_dm=slot_p3,
                        )
                    release_k(sync, slot_k3)
                if r3_j + 1 == r3_nt:
                    if split_decode and r3_task >= q_bn:
                        local_slot = slot_p3
                    else:
                        if not (folded or direct_kv):
                            bmm2_data_copy_out(
                                sync,
                                bufs,
                                out,
                                r3_task,
                                m,
                                rows,
                                dtype_in,
                                slot_sum=slot_p3,
                            )
                        if enable_lse == 1 and not (folded or direct_kv):
                            lse_copy_out(
                                sync,
                                bufs,
                                lse,
                                r3_task,
                                m,
                                rows,
                                slot=slot_p3,
                                row_base=(r3_task // g_split) * head_num
                                + (r3_task - (r3_task // g_split) * g_split) * half_g,
                            )

            rt = r2_task
            r3_task = rt
            rt = r2_j
            r3_j = rt
            rt = r2_nt
            r3_nt = rt
            rt = r2_nvalid
            r3_nvalid = rt
            rt = r2_nload
            r3_nload = rt
            rt = r2_kvrow0
            r3_kvrow0 = rt
            rt = r2_iscmp
            r3_iscmp = rt
            r3_slot = r2_slot
            r3_link = r2_link
            rt = r2_start
            r3_start = rt
            rt = r2_seq
            r3_seq = rt
            if direct_kv:
                if i < n_tiles:
                    r2_task = r0_task
                    r2_j = r0_j
                    r2_nt = r0_nt
                    r2_nvalid = r0_nvalid
            else:
                r2_task = r1_task
                r2_j = r1_j
                r2_nt = r1_nt
                r2_nvalid = r1_nvalid
                r2_nload = r1_nload
                r2_kvrow0 = r1_kvrow0
                r2_start = r1_start
                r2_iscmp = r1_iscmp
                r2_slot = r1_slot
                r2_link = r1_link
                r2_seq = r1_seq
                if i < n_tiles:
                    r1_task = r0_task
                    r1_j = r0_j
                    r1_nt = r0_nt
                    r1_nvalid = r0_nvalid
                    r1_nload = r0_nload
                    r1_kvrow0 = r0_kvrow0
                    r1_start = r0_start
                    r1_iscmp = r0_iscmp
                    r1_slot = r0_slot
                    r1_link = r0_link
                    r1_seq = r0_seq

        g_done = g_done + n_tiles

    if split_decode:
        pair = task0 // 2
        if task0 < split_cores and task0 % 2 == 1:
            _publish_decode_partition(
                sync,
                bufs,
                ws,
                cmp_phy,
                pair,
                rows,
                local_slot,
                second_partition_nonempty == 1,
            )
        _decode_vector_barrier(sync)
        if task0 < split_cores and task0 % 2 == 0:
            _merge_decode_partition(
                sync,
                bufs,
                ws,
                cmp_phy,
                out,
                lse,
                pair,
                q_bn + pair,
                rows,
                local_slot,
                enable_lse == 1,
                dtype_in,
            )

    if g_split == 2:
        for _k in tla.range(s2_budget):
            split_g_mte2_pair(sync)


HCA_PRELOAD_NUM = 2


@tla.kernel
def sparse_flash_mla_hca_kernel(
    q: tla.Tensor,
    ori_kv: tla.Tensor,
    cmp_kv: tla.Tensor,
    ori_bt: tla.Tensor,
    cmp_bt: tla.Tensor,
    cu_q: tla.Tensor,  # [B+1] int32
    cu_kv: tla.Tensor,  # [B+1] int32
    cu_cmp: tla.Tensor,  # [B+1] int32
    resid: tla.Tensor,  # [B] int32
    sinks: tla.Tensor,  # [H] fp32
    out: tla.Tensor,  # [T*H, D]
    lse: tla.Tensor,
    tiling: tla.Tensor,
    softmax_scale: tla.Float32,
    head_num: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    num_seqs: tla.Constexpr[int],
    cmp_ratio: tla.Constexpr[int],
    cmp_mask_mode: tla.Constexpr[int],
    ori_mask_mode: tla.Constexpr[int],
    ori_win_left: tla.Constexpr[int],
    ori_win_right: tla.Constexpr[int],
    ori_pa_block_size: tla.Constexpr[int],
    ori_pa_stride: tla.Constexpr[int],
    ori_pa_max_blocks: tla.Constexpr[int],
    ori_pa_pages: tla.Constexpr[int],
    cmp_pa_block_size: tla.Constexpr[int],
    cmp_pa_stride: tla.Constexpr[int],
    cmp_pa_max_blocks: tla.Constexpr[int],
    cmp_pa_pages: tla.Constexpr[int],
    g_split: tla.Constexpr[int],
    has_sinks: tla.Constexpr[int],
    enable_lse: tla.Constexpr[int],
    cmp_blk: tla.Constexpr[int],
) -> None:
    c1 = 1
    c2 = 2
    dtype_in = q.ptr.dtype
    #
    #   halfG = (gSize+1)/2 = 33，goIdx = (aicIdx%2==0)?0:halfG
    #   （common/smla_kernel_common_arch35.h:271-278）；
    #   fixpipe mSize = Align2Func(mRealSize)（csa_block_cube_arch35.h:440/493），
    odd_g = (g_split == 2) and (head_num % g_split != 0)
    half_g = (head_num + 1) // 2 if g_split == 2 else head_num
    m = (half_g + (half_g % 2)) if odd_g else (head_num // g_split)
    rows = m - m // 2
    row_cap = m if odd_g else None
    odd_mode = 1 if odd_g else 0
    sink_cap = ((head_num + 7) // 8) * 8 if odd_g else 0

    sync = declare_sync(need_lse_war=enable_lse == 1)
    bufs = alloc_buffers(
        dtype_in, row_cap, (sink_cap if sink_cap > 0 else None), preload=True
    )
    init_sync(sync, pre_l0c2=True)
    init_sync_preload(sync)
    need_init = compute_need_init(cu_q, cu_kv, num_seqs, ori_mask_mode, 0)
    clean_output(
        sync,
        bufs,
        out,
        lse,
        need_init,
        cu_q[num_seqs] * head_num * head_dim,
        0,
        enable_lse,
        dtype_in,
        head_dim=head_dim,
    )

    if need_init == 1:
        init_aiv_arrive = tla.cross_flag("smla_init_aiv_arrive", mode=4)
        init_aic_ffts = tla.cross_flag("smla_init_aic_ffts", mode=0)
        init_aic_done = tla.cross_flag("smla_init_aic_done", mode=2)
        with tla.cube():
            tla.pipe_barrier(tla.pipes.ALL)
            tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=0)
            tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=1)
            tla.cross_core_set_flag(init_aic_ffts, tla.pipes.FIX)
            tla.cross_core_wait_flag(init_aic_ffts, tla.pipes.SCALAR)
            tla.cross_core_set_flag(init_aic_done, tla.pipes.SCALAR)
        with tla.vector():
            tla.pipe_barrier(tla.pipes.ALL)
            if tla.arch.sub_block_idx() == 0:
                tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=0)
            else:
                tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=1)
            tla.cross_core_wait_flag(init_aic_done, tla.pipes.SCALAR)

    total_tasks = cu_q[num_seqs] * g_split
    task0 = tla.arch.block_idx()
    q_bn = tla.arch.block_num()
    g_k = tla.as_numeric(0)
    p_task = task0
    p_nt = tla.as_numeric(1)
    p_no = tla.as_numeric(1)
    p_cend = tla.as_numeric(0)
    p_ol = tla.as_numeric(KV_BLOCK)
    p_rb = tla.as_numeric(0)
    p_vm = tla.as_numeric(0)
    p_goff = tla.as_numeric(0)
    pp_task = task0
    pp_nt = tla.as_numeric(1)
    pp_no = tla.as_numeric(1)
    pp_rb = tla.as_numeric(0)
    pp_vm = tla.as_numeric(0)
    for task in tla.range(task0, total_tasks, q_bn):
        tok = task // g_split
        seq = tla.as_numeric(0)
        rowq = tok
        for bb in static_range(1, num_seqs):
            if tok >= cu_q[bb]:
                seq = bb
                rowq = tok - cu_q[bb]
        q_len = cu_q[seq + 1] - cu_q[seq]
        kv_len = cu_kv[seq + 1] - cu_kv[seq]
        cmp_len = cu_cmp[seq + 1] - cu_cmp[seq]

        causal_offset = kv_len - q_len
        next_tok = kv_len
        pre_tok = q_len
        if ori_mask_mode == 3:
            next_tok = causal_offset
            pre_tok = q_len
        elif ori_mask_mode == 4:
            if ori_win_left == -1:
                pre_tok = q_len
            else:
                pre_tok = ori_win_left - causal_offset
            if ori_win_right == -1:
                next_tok = kv_len
            else:
                next_tok = causal_offset + ori_win_right
            if pre_tok > q_len:
                pre_tok = q_len
        ori_start = rowq - pre_tok
        if ori_start < 0:
            ori_start = tla.as_numeric(0)
        if ori_start > kv_len:
            ori_start = kv_len
        ori_end = rowq + next_tok + 1
        if ori_end < 0:
            ori_end = tla.as_numeric(0)
        if ori_end > kv_len:
            ori_end = kv_len
        ori_lines = ori_end - ori_start
        if ori_lines < 0:
            ori_lines = tla.as_numeric(0)
        n_ori_tiles = (ori_lines + KV_BLOCK - 1) // KV_BLOCK

        next_cmp = cmp_len * cmp_ratio
        if cmp_mask_mode != 0:
            next_cmp = cmp_len * cmp_ratio + resid[seq] - q_len
        cmp_end = (rowq + 1 + next_cmp) // cmp_ratio
        if cmp_end < 0:
            cmp_end = tla.as_numeric(0)
        if cmp_end > cmp_len:
            cmp_end = cmp_len
        n_cmp_tiles = (cmp_end + cmp_blk - 1) // cmp_blk

        n_tiles = n_ori_tiles + n_cmp_tiles
        kv_row0 = cu_kv[seq] + ori_start
        cmp_row0 = cu_cmp[seq]

        #           csa_block_cube_arch35.h:267-275、kvcache.h:236-250）。
        half = task - tok * g_split
        g_off = half * half_g
        valid_m = half_g
        if half != 0:
            valid_m = head_num - half_g
        q_row_base = tok * head_num + g_off

        #   stage2 = tile x-2 （AIC bmm2×4+fixpipe + AIV vec2×4 + release；

        n_task = (task - task0) // q_bn
        q_two_n = n_task + n_task
        q_slot_a = q_two_n - (q_two_n // 3) * 3
        q_two_n1 = q_two_n + 1
        q_slot_b = q_two_n1 - (q_two_n1 // 3) * 3
        nx_task = task + q_bn
        nx_done = 0
        if nx_task < total_tasks:
            nx_done = 1
        if g_k == 0:
            load_q(
                sync,
                bufs,
                q,
                task,
                m,
                dtype_in,
                row_base=q_row_base,
                slot_a=q_slot_a,
                slot_b=q_slot_b,
            )

        drain = 0
        if nx_done == 0:
            drain = HCA_PRELOAD_NUM
        for i in tla.range(n_tiles + drain):
            x = g_k + i
            if i < n_tiles:
                slot_k0 = x - (x // KV_SLOTS) * KV_SLOTS
                n_load0 = tla.as_numeric(0)
                if i < n_ori_tiles:
                    n_load0 = ori_lines - i * KV_BLOCK
                    if n_load0 < 0:
                        n_load0 = tla.as_numeric(0)
                    if n_load0 > KV_BLOCK:
                        n_load0 = tla.as_numeric(KV_BLOCK)
                    if ori_pa_block_size > 0:
                        load_k_pa(
                            sync,
                            bufs,
                            ori_kv,
                            ori_bt,
                            seq,
                            ori_start + i * KV_BLOCK,
                            n_load0,
                            slot_k0,
                            dtype_in,
                            ori_pa_block_size,
                            ori_pa_stride,
                            ori_pa_max_blocks,
                            ori_pa_pages,
                        )
                    else:
                        load_k(
                            sync, bufs, ori_kv, kv_row0, i, n_load0, slot_k0, dtype_in
                        )
                else:
                    n_load0 = cmp_end - (i - n_ori_tiles) * cmp_blk
                    if n_load0 < 0:
                        n_load0 = tla.as_numeric(0)
                    if n_load0 > cmp_blk:
                        n_load0 = tla.as_numeric(cmp_blk)
                    if cmp_pa_block_size > 0:
                        load_k_pa(
                            sync,
                            bufs,
                            cmp_kv,
                            cmp_bt,
                            seq,
                            (i - n_ori_tiles) * cmp_blk,
                            n_load0,
                            slot_k0,
                            dtype_in,
                            cmp_pa_block_size,
                            cmp_pa_stride,
                            cmp_pa_max_blocks,
                            cmp_pa_pages,
                        )
                    else:
                        load_k(
                            sync,
                            bufs,
                            cmp_kv,
                            cmp_row0,
                            i - n_ori_tiles,
                            n_load0,
                            slot_k0,
                            dtype_in,
                            blk=cmp_blk,
                        )
            if x >= c1:
                if i != n_tiles + c1:
                    j1 = i - c1
                    nt1 = n_tiles
                    no1 = n_ori_tiles
                    cend1 = cmp_end
                    ol1 = ori_lines
                    task1 = task
                    nta1 = n_task
                    goff1 = g_off
                    pft_task = nx_task
                    pft_ok = nx_done
                    if i < c1:
                        j1 = p_nt - 1
                        nt1 = p_nt
                        no1 = p_no
                        cend1 = p_cend
                        ol1 = p_ol
                        task1 = p_task
                        nta1 = (p_task - task0) // q_bn
                        goff1 = p_goff
                        pft_task = task
                        pft_ok = 1
                    n_valid1 = tla.as_numeric(0)
                    if j1 < no1:
                        n_valid1 = ol1 - j1 * KV_BLOCK
                        if n_valid1 > KV_BLOCK:
                            n_valid1 = tla.as_numeric(KV_BLOCK)
                    else:
                        n_valid1 = cend1 - (j1 - no1) * cmp_blk
                        if n_valid1 > cmp_blk:
                            n_valid1 = tla.as_numeric(cmp_blk)
                    qa1 = nta1 + nta1
                    q_slot_a1 = qa1 - (qa1 // 3) * 3
                    qb1 = qa1 + 1
                    q_slot_b1 = qb1 - (qb1 // 3) * 3
                    xg1 = x - c1
                    slot_k1 = xg1 - (xg1 // KV_SLOTS) * KV_SLOTS
                    slot_s1 = xg1 - (xg1 // HCA_PRELOAD_NUM) * HCA_PRELOAD_NUM
                    slot_sum_rd = slot_s1
                    if j1 != 0:
                        slot_sum_rd = slot_s1 + 1
                        if slot_sum_rd > 1:
                            slot_sum_rd = slot_sum_rd - 2
                    if j1 == 0:
                        vec_init_from_sinks(
                            sync,
                            bufs,
                            sinks,
                            task1,
                            m,
                            rows,
                            dtype_in,
                            goff1,
                            has_sinks,
                            sink_cap,
                            slot_sum=slot_s1,
                            slot_max=slot_s1,
                            clear_o=0,
                        )
                    process_vec1(
                        sync,
                        bufs,
                        softmax_scale,
                        n_valid1,
                        m,
                        rows,
                        dtype_in,
                        slot_s=slot_s1,
                        slot_p=slot_s1,
                        preload=1,
                        slot_sum_rd=slot_sum_rd,
                        slot_max_rd=slot_sum_rd,
                    )
                    is_first1 = j1 == 0
                    is_last1 = j1 + c1 == nt1
                    if j1 < no1:
                        iterate_bmm1(
                            sync,
                            bufs,
                            slot_k1,
                            m,
                            is_last1,
                            is_first1,
                            dtype_in,
                            slot_s=slot_s1,
                            q_slot_a=q_slot_a1,
                            q_slot_b=q_slot_b1,
                        )
                    else:
                        iterate_bmm1(
                            sync,
                            bufs,
                            slot_k1,
                            m,
                            is_last1,
                            is_first1,
                            dtype_in,
                            slot_s=slot_s1,
                            q_slot_a=q_slot_a1,
                            q_slot_b=q_slot_b1,
                            n_blk=cmp_blk,
                        )
                    if is_last1:
                        if pft_ok == 1:
                            pft_n = (pft_task - task0) // q_bn
                            pft_2n = pft_n + pft_n
                            pft_sa = pft_2n - (pft_2n // 3) * 3
                            pft_2n1 = pft_2n + 1
                            pft_sb = pft_2n1 - (pft_2n1 // 3) * 3
                            pft_tok = pft_task // g_split
                            pft_half = pft_task - pft_tok * g_split
                            load_q(
                                sync,
                                bufs,
                                q,
                                pft_task,
                                m,
                                dtype_in,
                                row_base=pft_tok * head_num + pft_half * half_g,
                                slot_a=pft_sa,
                                slot_b=pft_sb,
                            )
            if x >= c2:
                j2 = i - c2
                nt2 = n_tiles
                no2 = n_ori_tiles
                task2 = task
                rb2 = q_row_base
                vm2 = valid_m
                if i < c2:
                    if i == c1:
                        j2 = p_nt - 1
                        nt2 = p_nt
                        no2 = p_no
                        task2 = p_task
                        rb2 = p_rb
                        vm2 = p_vm
                    else:
                        if p_nt >= 2:
                            j2 = p_nt - 2
                            nt2 = p_nt
                            no2 = p_no
                            task2 = p_task
                            rb2 = p_rb
                            vm2 = p_vm
                        else:
                            j2 = pp_nt - 1
                            nt2 = pp_nt
                            no2 = pp_no
                            task2 = pp_task
                            rb2 = pp_rb
                            vm2 = pp_vm
                xg2 = x - c2
                slot_k2 = xg2 - (xg2 // KV_SLOTS) * KV_SLOTS
                slot_p2 = xg2 - (xg2 // HCA_PRELOAD_NUM) * HCA_PRELOAD_NUM
                wait_p_ready(sync)
                if j2 < no2:
                    iterate_bmm2_seg(
                        sync, bufs, 0, slot_k2, m, dtype_in, slot_p=slot_p2
                    )
                    iterate_bmm2_seg(
                        sync, bufs, 1, slot_k2, m, dtype_in, slot_p=slot_p2
                    )
                    iterate_bmm2_seg(
                        sync, bufs, 2, slot_k2, m, dtype_in, slot_p=slot_p2
                    )
                    iterate_bmm2_seg(
                        sync, bufs, 3, slot_k2, m, dtype_in, slot_p=slot_p2
                    )
                else:
                    iterate_bmm2_seg(
                        sync, bufs, 0, slot_k2, m, dtype_in, slot_p=slot_p2, n_k=cmp_blk
                    )
                    iterate_bmm2_seg(
                        sync, bufs, 1, slot_k2, m, dtype_in, slot_p=slot_p2, n_k=cmp_blk
                    )
                    iterate_bmm2_seg(
                        sync, bufs, 2, slot_k2, m, dtype_in, slot_p=slot_p2, n_k=cmp_blk
                    )
                    iterate_bmm2_seg(
                        sync, bufs, 3, slot_k2, m, dtype_in, slot_p=slot_p2, n_k=cmp_blk
                    )
                iterate_bmm2_fixpipe(sync, bufs, m)
                fresh_v2 = j2 == 0
                for seg in (0, 1, 2, 3):
                    process_vec2_seg(
                        sync,
                        bufs,
                        seg,
                        m,
                        rows,
                        dtype_in,
                        fresh_v2,
                        slot_dm=slot_p2,
                    )
                release_k(sync, slot_k2)
                if j2 + c1 == nt2:
                    bmm2_data_copy_out(
                        sync,
                        bufs,
                        out,
                        task2,
                        m,
                        rows,
                        dtype_in,
                        row_base=rb2,
                        valid_rows=vm2,
                        odd_mode=odd_mode,
                        slot_sum=slot_p2,
                    )
                    if enable_lse == 1:
                        lse_copy_out(
                            sync,
                            bufs,
                            lse,
                            task2,
                            m,
                            rows,
                            slot=slot_p2,
                            row_base=rb2,
                            valid_rows=(vm2 if odd_mode == 1 else None),
                        )
        pp_task = p_task
        pp_nt = p_nt
        pp_no = p_no
        pp_rb = p_rb
        pp_vm = p_vm
        p_task = task
        p_nt = n_tiles
        p_no = n_ori_tiles
        p_cend = cmp_end
        p_ol = ori_lines
        p_rb = q_row_base
        p_vm = valid_m
        p_goff = g_off
        g_k = g_k + n_tiles


SWA_PRELOAD_NUM = 2


@tla.jit
def _task_vis(
    cu_q,
    cu_kv,
    ct,
    g_split: tla.Constexpr[int],
    num_seqs: tla.Constexpr[int],
    ori_mask_mode: tla.Constexpr[int],
    ori_win_left: tla.Constexpr[int],
    ori_win_right: tla.Constexpr[int],
):
    tok = ct // g_split
    seq = tla.as_numeric(0)
    rowq = tok
    for bb in static_range(1, num_seqs):
        if tok >= cu_q[bb]:
            seq = bb
            rowq = tok - cu_q[bb]
    q_len = cu_q[seq + 1] - cu_q[seq]
    kv_len = cu_kv[seq + 1] - cu_kv[seq]
    causal_offset = kv_len - q_len
    next_tok = kv_len
    pre_tok = q_len
    if ori_mask_mode == 3:
        next_tok = causal_offset
        pre_tok = q_len
    elif ori_mask_mode == 4:
        if ori_win_left == -1:
            pre_tok = q_len
        else:
            pre_tok = ori_win_left - causal_offset
        if ori_win_right == -1:
            next_tok = kv_len
        else:
            next_tok = causal_offset + ori_win_right
        if pre_tok > q_len:
            pre_tok = q_len
    end_idx = rowq + next_tok + 1
    if end_idx < 0:
        end_idx = tla.as_numeric(0)
    if end_idx > kv_len:
        end_idx = kv_len
    start_idx = rowq - pre_tok
    if start_idx < 0:
        start_idx = tla.as_numeric(0)
    if start_idx > kv_len:
        start_idx = kv_len
    vis = end_idx - start_idx
    if vis < 0:
        vis = tla.as_numeric(0)
    return vis


@tla.jit
def _swa_load_quantized_window(
    sync, bufs, kv, bt, window_end, window_size, slot, reuse, page_size
):
    """Decode the initial window and the two new rows on each later visit."""
    with tla.vector():
        group = 16
        pitch = group + 1
        # The launcher requires at least 128 physical rows. During startup
        # the whole first page is finite, and softmax masks the unused rows.
        remaining = KV_BLOCK
        logical = window_end - window_size
        if reuse:
            remaining = 2
            logical = window_end - remaining
        sub = tla.arch.sub_block_idx()
        half = remaining // 2
        start = logical + sub * half
        out = _k_slot_ptr(bufs, slot)
        raw = tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8)
        packed = tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt16)
        input_lock = sync.m_dq_rec1
        packed_lock = sync.m_dq_out1
        done = tla.as_numeric(0)
        for step in tla.range((half + group - 1) // group + 1):
            if done < half:
                offset = (start + done) % KV_BLOCK
                count = half - done
                count = count if count < group else group
                count = count if count < KV_BLOCK - offset else KV_BLOCK - offset
                input_lock.lock(pipe=tla.pipes.MTE2)
                _single_query_load_records(
                    raw, kv.ptr, bt.ptr, start + done, count, page_size
                )
                input_lock.unlock(pipe=tla.pipes.MTE2)
                input_lock.lock(pipe=tla.pipes.VECTOR)
                packed_lock.lock(pipe=tla.pipes.VECTOR)
                _single_query_dequant_nz(packed, raw, count, pitch)
                input_lock.unlock(pipe=tla.pipes.VECTOR)
                packed_lock.unlock(pipe=tla.pipes.VECTOR)
                packed_lock.lock(pipe=tla.pipes.MTE3)
                if step == 0:
                    tla.cross_core_wait_flag(sync.kv_l1_free, tla.pipes.MTE3, aiv_id=0)
                    tla.cross_core_wait_flag(sync.kv_l1_free, tla.pipes.MTE3, aiv_id=1)
                _single_query_insert_nz(
                    tla.recast_ptr(out + offset * 16, dtype=tla.UInt16),
                    packed,
                    count,
                    KV_BLOCK,
                    pitch,
                    HEAD_DIM,
                )
                packed_lock.unlock(pipe=tla.pipes.MTE3)
                done = done + count
        tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


_SWA_NORMALIZED_SOURCE = r"""
#include "kernel_operator.h"
using namespace AscendC::MicroAPI;

static constexpr CastTrait swa_probs = {RegLayout::ZERO, SatMode::SAT,
                                       MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

template <bool FullWindow, bool LastHalf>
__simd_vf__ void NormalizeSwaRows(
    __ubuf__ float* scores, __ubuf__ float* sinks, __ubuf__ bfloat16_t* prob,
    __ubuf__ float* maxima, __ubuf__ float* sums, int32_t count, float scale) {
    constexpr uint16_t rows = 32;
    auto all = CreateMask<float, MaskPattern::ALL>();
    auto bf16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    uint32_t one_lane = 1;
    auto one = UpdateMask<float>(one_lane);
    uint32_t n0 = count < 64 ? count : 64;
    uint32_t n1 = count > 64 ? count - 64 : 0;
    auto mask0 = UpdateMask<float>(n0);
    auto mask1 = UpdateMask<float>(n1);
    RegTensor<float> neg_inf, unity;
    Duplicate(neg_inf, -3.4028234663852886e38f);
    Duplicate(unity, 1.0f);
    UnalignRegForStore max_out, sum_out;
    for (uint16_t row = 0; row < rows; ++row) {
        RegTensor<float> sink, s0, s1, peak, finite, e0, e1, denom, inv;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(sink, sinks + row);
        LoadAlign(s0, scores + row * 128);
        Muls(s0, s0, scale, all);
        if constexpr (LastHalf) {
            LoadAlign(s1, scores + row * 128 + 64);
            Muls(s1, s1, scale, all);
            if constexpr (!FullWindow) Select(s1, s1, neg_inf, mask1);
            Max(finite, s0, s1, all);
        } else {
            Select(finite, s0, neg_inf, mask0);
        }
        ReduceMax(peak, finite, all);
        Duplicate(peak, peak, all);
        Max(peak, peak, sink, all);
        ExpSub(e0, s0, peak, mask0);
        if constexpr (LastHalf) {
            RegTensor<float> joined;
            ExpSub(e1, s1, peak, mask1);
            Add(joined, e0, e1, all);
            ReduceSum(denom, joined, all);
        } else {
            ReduceSum(denom, e0, all);
        }
        RegTensor<float> sink_prob;
        ExpSub(sink_prob, sink, peak, one);
        Add(denom, denom, sink_prob, one);
        Div(inv, unity, denom, one);
        Duplicate(inv, inv, all);
        RegTensor<bfloat16_t> p0, p1, packed, unused;
        Mul(e0, e0, inv, all);
        Cast<bfloat16_t, float, swa_probs>(p0, e0, all);
        if constexpr (LastHalf) {
            Mul(e1, e1, inv, all);
            Cast<bfloat16_t, float, swa_probs>(p1, e1, all);
        } else {
            Duplicate(p1, 0);
        }
        DeInterleave(packed, unused, p0, p1);
        auto* dst_row = prob + row * 16;
        StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            dst_row, packed, rows, 1, bf16);
        StoreUnAlign(maxima, peak, max_out, 1);
        StoreUnAlign(sums, denom, sum_out, 1);
    }
    StoreUnAlignPost(maxima, max_out, 0);
    StoreUnAlignPost(sums, sum_out, 0);
}

extern "C" [aicore] __attribute__((always_inline)) void swa_normalize(
    uint64_t scores, uint64_t sink, uint64_t probs, uint64_t maxima, uint64_t sums,
    int32_t count, float scale) {
    auto* s = reinterpret_cast<__ubuf__ float*>(scores);
    auto* h = reinterpret_cast<__ubuf__ float*>(sink);
    auto* p = reinterpret_cast<__ubuf__ bfloat16_t*>(probs);
    auto* m = reinterpret_cast<__ubuf__ float*>(maxima);
    auto* z = reinterpret_cast<__ubuf__ float*>(sums);
    if (count == 128) {
        NormalizeSwaRows<true, true>(s, h, p, m, z, count, scale);
    } else if (count > 64) {
        NormalizeSwaRows<false, true>(s, h, p, m, z, count, scale);
    } else {
        NormalizeSwaRows<false, false>(s, h, p, m, z, count, scale);
    }
}
"""


@tla.extern(source=_SWA_NORMALIZED_SOURCE, name="swa_normalize")
def _swa_normalize(
    scores: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    sinks: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    probs: tla.Pointer[tla.BFloat16, tla.AddressSpace.ub],
    maxima: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    sums: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    count: tla.Int32,
    scale: tla.Float32,
) -> None: ...


@tla.jit
def _swa_cache_sinks(sync, bufs, sinks, rows, has_sinks: tla.Constexpr[int]):
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        dst = _stats(bufs.ub_sink, rows)
        if has_sinks:
            src = _stats(sinks.ptr + sub * rows, rows)
            tla.copy(dst, src)
            tla.set_flag(sync.phy_m2v)
            tla.wait_flag(sync.phy_m2v)
        else:
            with tla.vec.func(mode="simd"):
                live, _ = tla.update_mask(rows, dtype=tla.Float32)
                dst.store(tla.full(MIN_F, dtype=tla.Float32), mask=live)


@tla.jit
def _swa_process_normalized(sync, bufs, scale, valid, m, rows, dt, slot):
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        prob = _ub_p(bufs, rows)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=1)
        sync.m_p.lock(pipe=tla.pipes.VECTOR)
        _swa_normalize(
            _ub_s(bufs, rows, slot).ptr,
            bufs.ub_sink,
            prob.ptr,
            _max(bufs, rows, slot).ptr,
            _sum(bufs, rows, slot).ptr,
            valid,
            scale,
        )
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=1)
        sync.m_p.unlock(pipe=tla.pipes.VECTOR)
        l1p = l1_p_tensor(bufs.l1_p + slot * Q_ROWS * KV_BLOCK, dt, m)
        dst = tla.tile_view(l1p, tla.make_shape(rows, KV_BLOCK), tla.make_coord(sub, 0))
        sync.m_p.lock(pipe=tla.pipes.MTE3)
        tla.set_flag(sync.p_data_ready)
        tla.wait_flag(sync.p_data_ready)
        tla.copy(dst, prob)
        sync.m_p.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def _swa_store_normalized_lse(sync, bufs, lse, task, m, rows, slot):
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        stats_layout = tla.make_layout(tla.make_shape(VL), tla.make_stride(1))
        mx = tla.make_tensor(bufs.ub_max + slot * rows, stats_layout)
        z = tla.make_tensor(bufs.ub_sum + slot * rows, stats_layout)
        ub_lse = tla.make_tensor(bufs.ub_dm + slot * rows, stats_layout)
        with tla.vec.func(mode="simd"):
            active, _ = tla.update_mask(rows, dtype=tla.Float32)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            lse_vec = tla.add(tla.log(z.load(), mask=active), mx.load(), mask=active)
            ub_lse.store(lse_vec, mask=active)
        tla.set_flag(sync.lse_v_mte3)
        tla.wait_flag(sync.lse_v_mte3)
        target = _stats(lse.ptr + task * m + sub * rows, rows)
        tla.copy(target, _dm(bufs, rows, slot))
        tla.set_flag(sync.lse_mte3_v)
        tla.wait_flag(sync.lse_mte3_v)


@tla.jit
def _swa_store_normalized_output(sync, bufs, out, task, m):
    with tla.cube():
        src = tla.make_tensor(
            bufs.l0c_o,
            tla.make_layout_with_tag((m, HEAD_DIM), tla.Float32, tla.arch.L0Clayout),
        )
        dst = tla.make_tensor(
            out.ptr + task * m * HEAD_DIM,
            tla.make_layout_with_tag((m, HEAD_DIM), out.ptr.dtype, tla.arch.RowMajor),
        )
        tla.set_flag(sync.o_mm_ready0)
        tla.wait_flag(sync.o_mm_ready0)
        tla.copy(dst, src, cache_policy=tla.params.CachePolicy.DISABLE)
        tla.set_flag(sync.o_fix_done0)


@tla.kernel
def sparse_flash_mla_swa_kernel(
    q: tla.Tensor,
    kv: tla.Tensor,
    ori_bt: tla.Tensor,
    sinks: tla.Tensor,  # [H] fp32
    cu_q: tla.Tensor,  # [B+1] int32
    cu_kv: tla.Tensor,  # [B+1] int32
    out: tla.Tensor,  # [T, H, D] TND
    lse: tla.Tensor,
    tiling: tla.Tensor,
    bc_ws: tla.Tensor,
    softmax_scale: tla.Float32,
    head_num: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    num_seqs: tla.Constexpr[int],
    ori_mask_mode: tla.Constexpr[int],
    ori_win_left: tla.Constexpr[int],
    ori_win_right: tla.Constexpr[int],
    ori_pa_block_size: tla.Constexpr[int],
    ori_pa_stride: tla.Constexpr[int],
    ori_pa_max_blocks: tla.Constexpr[int],
    ori_pa_pages: tla.Constexpr[int],
    g_split: tla.Constexpr[int],
    has_sinks: tla.Constexpr[int],
    enable_lse: tla.Constexpr[int],
    is_batch_consistency: tla.Constexpr[int],
    rolling: tla.Constexpr[bool] = False,
    rolling_records: tla.Constexpr[bool] = False,
    balance_task_ranges: tla.Constexpr[bool] = False,
) -> None:
    c0 = 0
    c1 = 1
    c2 = 2
    dtype_in = q.ptr.dtype
    m = head_num // g_split
    rows = m - m // 2
    half_g = head_num // g_split if head_num % g_split == 0 else head_num // g_split + 1

    sync = declare_sync(
        need_lse_war=enable_lse == 1,
        two_q_slots=rolling,
        vector_query=rolling,
        need_fused_dequant=rolling,
        direct_kv=rolling_records,
    )
    bufs = alloc_buffers(
        dtype_in,
        bc=is_batch_consistency,
        preload=True,
        rolling=rolling,
        kv_pad=1 if rolling_records else 0,
        stat_pad=VL if rolling_records else 0,
    )
    init_sync(sync, pre_bc=(is_batch_consistency == 1), pre_l0c2=True)
    init_sync_preload(sync)
    if rolling:
        with tla.cube():
            tla.cross_core_set_flag(sync.q_vector_free, tla.pipes.MTE1, aiv_id=0)
            tla.cross_core_set_flag(sync.q_vector_free, tla.pipes.MTE1, aiv_id=1)
    if rolling_records:
        with tla.cube():
            for _ in tla.range_constexpr(2):
                tla.cross_core_set_flag(sync.kv_l1_free, tla.pipes.MTE1, aiv_id=0)
                tla.cross_core_set_flag(sync.kv_l1_free, tla.pipes.MTE1, aiv_id=1)
    if rolling_records:
        _swa_cache_sinks(sync, bufs, sinks, rows, has_sinks)
    # F4（OT swa_kernel_arch35.h:154 CleanOutput）
    need_init = compute_need_init(cu_q, cu_kv, num_seqs, ori_mask_mode, 0)
    clean_output(
        sync,
        bufs,
        out,
        lse,
        need_init,
        cu_q[num_seqs] * head_num * head_dim,
        0,
        enable_lse,
        dtype_in,
        head_dim=head_dim,
    )

    if need_init == 1:
        init_aiv_arrive = tla.cross_flag("smla_init_aiv_arrive", mode=4)
        init_aic_ffts = tla.cross_flag("smla_init_aic_ffts", mode=0)
        init_aic_done = tla.cross_flag("smla_init_aic_done", mode=2)
        with tla.cube():
            tla.pipe_barrier(tla.pipes.ALL)
            tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=0)
            tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=1)
            tla.cross_core_set_flag(init_aic_ffts, tla.pipes.FIX)
            tla.cross_core_wait_flag(init_aic_ffts, tla.pipes.SCALAR)
            tla.cross_core_set_flag(init_aic_done, tla.pipes.SCALAR)
        with tla.vector():
            tla.pipe_barrier(tla.pipes.ALL)
            if tla.arch.sub_block_idx() == 0:
                tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=0)
            else:
                tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=1)
            tla.cross_core_wait_flag(init_aic_done, tla.pipes.SCALAR)

    total_tasks = cu_q[num_seqs] * g_split
    task0 = tla.arch.block_idx()
    q_bn = tla.arch.block_num()
    if balance_task_ranges:
        # Distribute the remaining tasks across cores to balance contiguous slices.
        end_task = total_tasks * (task0 + 1) // q_bn
        task0 = total_tasks * task0 // q_bn
        total_tasks = end_task
        q_bn = tla.as_numeric(1)
    elif rolling:
        tasks_per_core = (total_tasks + q_bn - 1) // q_bn
        task0 = task0 * tasks_per_core
        end_task = task0 + tasks_per_core
        if end_task < total_tasks:
            total_tasks = end_task
        q_bn = tla.as_numeric(1)
    task_range = tla.range(task0, total_tasks, q_bn)
    g_k = tla.as_numeric(0)
    p_task = task0
    p_nt = tla.as_numeric(1)
    p_vis = tla.as_numeric(KV_BLOCK)
    p_bbn = tla.as_numeric(1)
    p_goff = tla.as_numeric(0)
    pp_task = task0
    pp_nt = tla.as_numeric(1)
    pp_vis = tla.as_numeric(KV_BLOCK)
    pp_bbn = tla.as_numeric(1)
    pp_goff = tla.as_numeric(0)
    for task in task_range:
        tok = task // g_split
        seq = tla.as_numeric(0)
        rowq = tok
        for bb in static_range(1, num_seqs):
            if tok >= cu_q[bb]:
                seq = bb
                rowq = tok - cu_q[bb]
        q_len = cu_q[seq + 1] - cu_q[seq]
        kv_len = cu_kv[seq + 1] - cu_kv[seq]
        causal_offset = kv_len - q_len
        next_tok = kv_len  # mode 0（full）
        pre_tok = q_len
        if ori_mask_mode == 3:
            next_tok = causal_offset
            pre_tok = q_len
        elif ori_mask_mode == 4:
            if ori_win_left == -1:
                pre_tok = q_len
            else:
                pre_tok = ori_win_left - causal_offset
            if ori_win_right == -1:
                next_tok = kv_len
            else:
                next_tok = causal_offset + ori_win_right
            if pre_tok > q_len:
                pre_tok = q_len
        end_idx = rowq + next_tok + 1
        if end_idx < 0:
            end_idx = tla.as_numeric(0)
        if end_idx > kv_len:
            end_idx = kv_len
        start_idx = rowq - pre_tok
        if start_idx < 0:
            start_idx = tla.as_numeric(0)
        if start_idx > kv_len:
            start_idx = kv_len
        vis = end_idx - start_idx
        if vis < 0:
            vis = tla.as_numeric(0)
        num_tiles = (vis + KV_BLOCK - 1) // KV_BLOCK
        # （swa_kernel_arch35.h:428-437：raw=ceil(totalLoad/32)、
        raw_rb = (vis + 31) // 32
        rb = ((raw_rb + 127) // 128) * 128
        base_block_num = rb // 128
        if rb == 0:
            base_block_num = tla.as_numeric(1)
        kv_row0 = cu_kv[seq] + start_idx
        g_off = (task - tok * g_split) * m

        any_after = 0
        nx_task = task + q_bn
        nx_done = 0
        if rolling:
            if nx_task < total_tasks:
                nx_done = 1
                any_after = 1
        else:
            n_probe = (total_tasks - task + q_bn - 1) // q_bn
            for k in tla.range(1, n_probe):
                if nx_done == 0:
                    ct_p = task + k * q_bn
                    if ct_p < total_tasks:
                        if (
                            _task_vis(
                                cu_q,
                                cu_kv,
                                ct_p,
                                g_split,
                                num_seqs,
                                ori_mask_mode,
                                ori_win_left,
                                ori_win_right,
                            )
                            > 0
                        ):
                            nx_task = ct_p
                            nx_done = 1
                            any_after = 1

        n_task = (task - task0) // q_bn
        q_two_n = n_task + n_task
        q_slot_a = q_two_n - (q_two_n // 3) * 3
        q_two_n1 = q_two_n + 1
        q_slot_b = q_two_n1 - (q_two_n1 // 3) * 3
        if not rolling and g_k == 0:
            if num_tiles > 0:
                load_q(
                    sync, bufs, q, task, m, dtype_in, slot_a=q_slot_a, slot_b=q_slot_b
                )

        drain = 0
        if num_tiles > 0:
            if any_after == 0:
                drain = SWA_PRELOAD_NUM
        slots = 2 if rolling else KV_SLOTS
        for i in tla.range(num_tiles + drain):
            x = g_k + i
            if i < num_tiles:
                n_load0 = vis - i * KV_BLOCK
                if n_load0 < 0:
                    n_load0 = tla.as_numeric(0)
                if n_load0 > KV_BLOCK:
                    n_load0 = tla.as_numeric(KV_BLOCK)
                slot_k0 = x - (x // slots) * slots
                if rolling:
                    if slot_k0 == 0:
                        load_rolling_q(sync, bufs, q, task, m, 0)
                    else:
                        _swa_load_query_vector(sync, bufs, q, task, slot_k0)
                    if rolling_records:
                        _swa_load_quantized_window(
                            sync,
                            bufs,
                            kv,
                            ori_bt,
                            end_idx,
                            n_load0,
                            slot_k0,
                            x >= slots,
                            ori_pa_block_size,
                        )
                    else:
                        load_rolling_window(
                            sync,
                            bufs,
                            kv,
                            ori_bt,
                            end_idx,
                            n_load0,
                            slot_k0,
                            x >= slots,
                            ori_pa_block_size,
                            ori_pa_stride,
                        )
                elif ori_pa_block_size > 0:
                    ft = c1
                    if i > 0:
                        ft = tla.as_numeric(0)
                    load_k_pa(
                        sync,
                        bufs,
                        kv,
                        ori_bt,
                        seq,
                        start_idx + i * KV_BLOCK,
                        n_load0,
                        slot_k0,
                        dtype_in,
                        ori_pa_block_size,
                        ori_pa_stride,
                        ori_pa_max_blocks,
                        ori_pa_pages,
                        fill_tail=ft,
                    )
                else:
                    load_k(sync, bufs, kv, kv_row0, i, n_load0, slot_k0, dtype_in)
            if x >= c1:
                if i != num_tiles + c1:
                    j1 = i - c1
                    nt1 = num_tiles
                    vis1 = vis
                    bbn1 = base_block_num
                    goff1 = g_off
                    task1 = task
                    nta1 = n_task
                    pft_task = nx_task
                    pft_ok = nx_done
                    if i < c1:
                        j1 = p_nt - 1
                        nt1 = p_nt
                        vis1 = p_vis
                        bbn1 = p_bbn
                        goff1 = p_goff
                        task1 = p_task
                        nta1 = (p_task - task0) // q_bn
                        pft_task = task
                        pft_ok = 1
                    qa1 = nta1 + nta1
                    q_slot_a1 = qa1 - (qa1 // 3) * 3
                    qb1 = qa1 + 1
                    q_slot_b1 = qb1 - (qb1 // 3) * 3
                    n_valid1 = vis1 - j1 * KV_BLOCK
                    if n_valid1 > KV_BLOCK:
                        n_valid1 = tla.as_numeric(KV_BLOCK)
                    xg1 = x - c1
                    slot_k1 = xg1 - (xg1 // slots) * slots
                    if rolling:
                        q_slot_a1 = slot_k1 * 2
                        q_slot_b1 = q_slot_a1 + 1
                    slot_s1 = xg1 - (xg1 // SWA_PRELOAD_NUM) * SWA_PRELOAD_NUM
                    slot_sum_rd = slot_s1
                    if j1 != 0:
                        slot_sum_rd = slot_s1 + 1
                        if slot_sum_rd > 1:
                            slot_sum_rd = slot_sum_rd - 2
                        if is_batch_consistency == 1:
                            if j1 - (j1 // bbn1) * bbn1 == 0:
                                slot_sum_rd = slot_s1
                    is_first1 = j1 == 0
                    is_last1 = j1 + c1 == nt1
                    # （OT InitVec1SoftmaxFromSinks / ProcessVec1 isFirstSoftmaxBase
                    if not rolling_records and is_first1:
                        vec_init_from_sinks(
                            sync,
                            bufs,
                            sinks,
                            task1,
                            m,
                            rows,
                            dtype_in,
                            goff1,
                            has_sinks,
                            slot_sum=slot_s1,
                            slot_max=slot_s1,
                            clear_o=0,
                        )
                    tid_in_block1 = j1
                    is_first_base1 = j1 == 0
                    reduce_block_id1 = j1
                    if is_batch_consistency == 1:
                        tid_in_block1 = j1 - (j1 // bbn1) * bbn1
                        is_first_base1 = tid_in_block1 == 0
                        reduce_block_id1 = j1 // bbn1
                        if is_first_base1:
                            if j1 != 0:
                                bc_reinit_stats(bufs, rows, slot_s1)
                    if rolling_records:
                        _swa_process_normalized(
                            sync,
                            bufs,
                            softmax_scale,
                            n_valid1,
                            m,
                            rows,
                            dtype_in,
                            slot_s1,
                        )
                    else:
                        process_vec1(
                            sync,
                            bufs,
                            softmax_scale,
                            n_valid1,
                            m,
                            rows,
                            dtype_in,
                            slot_s=slot_s1,
                            slot_p=slot_s1,
                            preload=1,
                            slot_sum_rd=slot_sum_rd,
                            slot_max_rd=slot_sum_rd,
                        )
                    iterate_bmm1(
                        sync,
                        bufs,
                        slot_k1,
                        m,
                        is_last1,
                        is_first1,
                        dtype_in,
                        slot_s=slot_s1,
                        q_slot_a=q_slot_a1,
                        q_slot_b=q_slot_b1,
                        two_q_slots=rolling,
                        vector_query=rolling,
                        direct_kv=rolling_records,
                    )
                    if not rolling and is_last1:
                        if pft_ok == 1:
                            pft_n = (pft_task - task0) // q_bn
                            pft_2n = pft_n + pft_n
                            pft_sa = pft_2n - (pft_2n // 3) * 3
                            pft_2n1 = pft_2n + 1
                            pft_sb = pft_2n1 - (pft_2n1 // 3) * 3
                            load_q(
                                sync,
                                bufs,
                                q,
                                pft_task,
                                m,
                                dtype_in,
                                slot_a=pft_sa,
                                slot_b=pft_sb,
                            )
            # local（j2=i-2）；i==1 → x-2 =
            if x >= c2:
                j2 = i - c2
                nt2 = num_tiles
                vis2 = vis
                bbn2 = base_block_num
                goff2 = g_off
                task2 = task
                if i < c2:
                    if i == c1:
                        j2 = p_nt - 1
                        nt2 = p_nt
                        vis2 = p_vis
                        bbn2 = p_bbn
                        goff2 = p_goff
                        task2 = p_task
                    else:
                        if p_nt >= 2:
                            j2 = p_nt - 2
                            nt2 = p_nt
                            vis2 = p_vis
                            bbn2 = p_bbn
                            goff2 = p_goff
                            task2 = p_task
                        else:
                            j2 = pp_nt - 1
                            nt2 = pp_nt
                            vis2 = pp_vis
                            bbn2 = pp_bbn
                            goff2 = pp_goff
                            task2 = pp_task
                n_valid2 = vis2 - j2 * KV_BLOCK
                if n_valid2 > KV_BLOCK:
                    n_valid2 = tla.as_numeric(KV_BLOCK)
                xg2 = x - c2
                slot_k2 = xg2 - (xg2 // slots) * slots
                slot_p2 = xg2 - (xg2 // SWA_PRELOAD_NUM) * SWA_PRELOAD_NUM
                tid_in_block2 = j2
                is_first_base2 = j2 == 0
                reduce_block_id2 = j2
                fresh_v2 = is_first_base2
                if is_batch_consistency == 1:
                    tid_in_block2 = j2 - (j2 // bbn2) * bbn2
                    is_first_base2 = tid_in_block2 == 0
                    reduce_block_id2 = j2 // bbn2
                    # smlaIsFirstVec2Base=isFirstBase，
                    fresh_v2 = is_first_base2
                wait_p_ready(sync)
                for seg in (0, 1, 2, 3):
                    iterate_bmm2_seg(
                        sync, bufs, seg, slot_k2, m, dtype_in, slot_p=slot_p2
                    )
                if rolling_records:
                    with tla.cube():
                        tla.cross_core_set_flag(
                            sync.kv_l1_free, tla.pipes.MTE1, aiv_id=0
                        )
                        tla.cross_core_set_flag(
                            sync.kv_l1_free, tla.pipes.MTE1, aiv_id=1
                        )
                if rolling_records:
                    _swa_store_normalized_output(sync, bufs, out, task2, m)
                else:
                    iterate_bmm2_fixpipe(sync, bufs, m)
                if not rolling:
                    for seg in (0, 1, 2, 3):
                        process_vec2_seg(
                            sync,
                            bufs,
                            seg,
                            m,
                            rows,
                            dtype_in,
                            fresh_v2,
                            slot_dm=slot_p2,
                        )
                if is_batch_consistency == 1:
                    is_last_base2 = tid_in_block2 == bbn2 - 1
                    if j2 + c1 == nt2:
                        is_last_base2 = j2 + c1 == nt2
                    if is_last_base2:
                        if nt2 != 1:
                            bc_div_cur(bufs, rows, slot_p2)
                        if reduce_block_id2 != 0:
                            bc_merge_prev(sync, bufs, bc_ws, rows, slot_p2)
                        if j2 + c1 != nt2:
                            bc_stage_cur(sync, bufs, bc_ws, rows, slot_p2)
                if not rolling_records:
                    release_k(sync, slot_k2)
                if j2 + c1 == nt2:
                    if not rolling_records:
                        bmm2_data_copy_out(
                            sync,
                            bufs,
                            out,
                            task2,
                            m,
                            rows,
                            dtype_in,
                            slot_sum=slot_p2,
                            fresh_from_cube=rolling,
                        )
                    if enable_lse == 1:
                        if rolling_records:
                            _swa_store_normalized_lse(
                                sync, bufs, lse, task2, m, rows, slot_p2
                            )
                        else:
                            lse_copy_out(
                                sync,
                                bufs,
                                lse,
                                task2,
                                m,
                                rows,
                                slot=slot_p2,
                                row_base=(task2 // g_split) * head_num
                                + (task2 - (task2 // g_split) * g_split) * half_g,
                            )

        if num_tiles > 0:
            pp_task = p_task
            pp_nt = p_nt
            pp_vis = p_vis
            pp_bbn = p_bbn
            pp_goff = p_goff
            p_task = task
            p_nt = num_tiles
            p_vis = vis
            p_bbn = base_block_num
            p_goff = g_off
        g_k = g_k + num_tiles


ORI_SPARSE_PRELOAD_NUM = 3


@tla.kernel
def sparse_flash_mla_ori_sparse_kernel(
    q: tla.Tensor,
    ori_kv: tla.Tensor,
    ori_bt: tla.Tensor,
    ori_idx: tla.Tensor,
    ori_topk: tla.Tensor,  # [T] int32
    cu_q: tla.Tensor,  # [B+1] int32
    cu_kv: tla.Tensor,  # [B+1] int32
    sinks: tla.Tensor,  # [H] fp32
    out: tla.Tensor,  # [T*H, D]
    lse: tla.Tensor,
    tiling: tla.Tensor,
    ws: tla.Tensor,
    ori_phy: tla.Tensor,
    softmax_scale: tla.Float32,
    head_num: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    num_seqs: tla.Constexpr[int],
    idx_width: tla.Constexpr[int],
    ori_mask_mode: tla.Constexpr[int],
    ori_win_left: tla.Constexpr[int],
    ori_win_right: tla.Constexpr[int],
    enable_lse: tla.Constexpr[int],
    ori_pa_block_size: tla.Constexpr[int],
    ori_pa_stride: tla.Constexpr[int],
    ori_pa_max_blocks: tla.Constexpr[int],
    g_split: tla.Constexpr[int],
    has_sinks: tla.Constexpr[int],
    topk_value_mode: tla.Constexpr[int],  # F5: 1=TOPK_INDEX, 2=TOPK_OFFSET
    kv_layout: tla.Constexpr[int],
    is_vec_s2phyaddr: tla.Constexpr[int],
    ori_phy_width: tla.Constexpr[int],
) -> None:
    dtype_in = q.ptr.dtype
    # IS_SPLIT_G（OT: tiling splitG = gSize > 64，common/smla_kernel_common_arch35.h:273-277）：
    m = head_num // g_split
    rows = m - m // 2
    half_g = head_num // g_split if head_num % g_split == 0 else head_num // g_split + 1

    configure_trace(g_split, False, False, KV_SLOTS, "os")

    use_vec_gather = (is_vec_s2phyaddr == 1) and (kv_layout != 2)
    idx_ring_on = use_vec_gather and kv_layout != 2

    sync = declare_sync(need_idx_ring=idx_ring_on, need_lse_war=enable_lse == 1)
    bufs = alloc_buffers(
        dtype_in,
        phy=ori_phy_width if use_vec_gather else 0,
        phy_bt=ori_pa_max_blocks if use_vec_gather else 0,
        preload=True,
        idx_ring=idx_ring_on,
    )
    init_sync(
        sync,
        pre_phy_vs=use_vec_gather and not idx_ring_on,
        pre_l0c2=True,
        pre_idx_ring=idx_ring_on,
    )
    init_sync_preload(sync)
    need_init = compute_need_init(cu_q, cu_kv, num_seqs, ori_mask_mode, 1)
    clean_output(
        sync,
        bufs,
        out,
        lse,
        need_init,
        cu_q[num_seqs] * head_num * head_dim,
        cu_q[num_seqs] * head_num,
        enable_lse,
        dtype_in,
    )

    if is_vec_s2phyaddr == 1:
        init_aiv_arrive = tla.cross_flag("smla_init_aiv_arrive", mode=4)
        init_aic_ffts = tla.cross_flag("smla_init_aic_ffts", mode=0)
        init_aic_done = tla.cross_flag("smla_init_aic_done", mode=2)
        with tla.cube():
            tla.pipe_barrier(tla.pipes.ALL)
            tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=0)
            tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=1)
            tla.cross_core_set_flag(init_aic_ffts, tla.pipes.FIX)
            tla.cross_core_wait_flag(init_aic_ffts, tla.pipes.SCALAR)
            tla.cross_core_set_flag(init_aic_done, tla.pipes.SCALAR)
        with tla.vector():
            tla.pipe_barrier(tla.pipes.ALL)
            if tla.arch.sub_block_idx() == 0:
                tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=0)
            else:
                tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=1)
            tla.cross_core_wait_flag(init_aic_done, tla.pipes.SCALAR)
    else:
        init_aiv_arrive = tla.cross_flag("smla_init_aiv_arrive", mode=4)
        init_aic_ffts = tla.cross_flag("smla_init_aic_ffts", mode=0)
        init_aic_done = tla.cross_flag("smla_init_aic_done", mode=2)
        if need_init == 1:
            with tla.cube():
                tla.pipe_barrier(tla.pipes.ALL)
                tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=0)
                tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=1)
                tla.cross_core_set_flag(init_aic_ffts, tla.pipes.FIX)
                tla.cross_core_wait_flag(init_aic_ffts, tla.pipes.SCALAR)
                tla.cross_core_set_flag(init_aic_done, tla.pipes.SCALAR)
            with tla.vector():
                tla.pipe_barrier(tla.pipes.ALL)
                if tla.arch.sub_block_idx() == 0:
                    tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=0)
                else:
                    tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=1)
                tla.cross_core_wait_flag(init_aic_done, tla.pipes.SCALAR)

    total_tasks = cu_q[num_seqs] * g_split
    task0 = tla.arch.block_idx()
    q_bn = tla.arch.block_num()

    s2_budget = tla.as_numeric(0)
    c_one = 1
    if g_split == 2:
        s2_budget = tiling[c_one]

    r0_task = tla.as_numeric(0)
    r0_j = tla.as_numeric(0)
    r0_nt = tla.as_numeric(1)
    r0_nvalid = tla.as_numeric(0)
    r0_nload = tla.as_numeric(0)
    r1_task = tla.as_numeric(0)
    r1_j = tla.as_numeric(0)
    r1_nt = tla.as_numeric(1)
    r1_nvalid = tla.as_numeric(0)
    r1_nload = tla.as_numeric(0)
    r2_task = tla.as_numeric(0)
    r2_j = tla.as_numeric(0)
    r2_nt = tla.as_numeric(1)
    r2_nvalid = tla.as_numeric(0)
    r2_nload = tla.as_numeric(0)
    r3_task = tla.as_numeric(0)
    r3_j = tla.as_numeric(0)
    r3_nt = tla.as_numeric(1)
    r3_nvalid = tla.as_numeric(0)
    r3_nload = tla.as_numeric(0)

    g_done = tla.as_numeric(0)
    ws_base = tla.arch.block_idx() * (KV_SLOTS * KV_STAGE_STRIDE)

    n_steps = (total_tasks - task0 + q_bn - 1) // q_bn
    if n_steps < 0:
        n_steps = tla.as_numeric(0)

    for step in tla.range(n_steps + 1):
        task = task0 + step * q_bn

        n_tiles = tla.as_numeric(0)
        seq = tla.as_numeric(0)
        tok = tla.as_numeric(0)
        rowq = tla.as_numeric(0)
        kv_len = tla.as_numeric(0)
        lines = tla.as_numeric(0)
        kv_row0 = tla.as_numeric(0)

        if step < n_steps:
            tok = task // g_split
            rowq = tok
            for bb in static_range(1, num_seqs):
                if tok >= cu_q[bb]:
                    seq = bb
                    rowq = tok - cu_q[bb]
            q_len = cu_q[seq + 1] - cu_q[seq]
            kv_len = cu_kv[seq + 1] - cu_kv[seq]

            # cubeSOuterOffset ≡ rowq，s1RealSize ≡ 1：
            #   end   ← clip(rowq + nextTokensPerBatchOri + 1, 0, kv_len)
            #   start ← clip(rowq - preTokensPerBatchOri,     0, kv_len)
            #   mode 3（RightDownCausal）：next = kv-q, pre = q
            #                   next = (win_right==-1) ? kv : (kv-q)+win_right
            #   mode 0（full）：next = kv, pre = q
            causal_offset = kv_len - q_len
            next_tok = kv_len
            pre_tok = q_len
            if ori_mask_mode == 3:
                next_tok = causal_offset
                pre_tok = q_len
            elif ori_mask_mode == 4:
                if ori_win_left == -1:
                    pre_tok = q_len
                else:
                    pre_tok = ori_win_left - causal_offset
                if ori_win_right == -1:
                    next_tok = kv_len
                else:
                    next_tok = causal_offset + ori_win_right
                if pre_tok > q_len:
                    pre_tok = q_len
            else:
                next_tok = kv_len
                pre_tok = q_len
            end_idx = rowq + next_tok + 1
            if end_idx < 0:
                end_idx = tla.as_numeric(0)
            if end_idx > kv_len:
                end_idx = kv_len
            start_idx = rowq - pre_tok
            if start_idx < 0:
                start_idx = tla.as_numeric(0)
            if start_idx > kv_len:
                start_idx = kv_len
            range_len = end_idx - start_idx
            if range_len < 0:
                range_len = tla.as_numeric(0)

            sparse_count = ori_topk[tok]
            if sparse_count > idx_width:
                sparse_count = tla.as_numeric(idx_width)
            if sparse_count < 0:
                sparse_count = tla.as_numeric(0)
            lines = range_len
            if lines > sparse_count:
                lines = sparse_count
            if lines > kv_len:
                lines = kv_len
            if lines < 0:
                lines = tla.as_numeric(0)
            n_tiles = (lines + KV_BLOCK - 1) // KV_BLOCK

            kv_row0 = cu_kv[seq]

        n_body = n_tiles
        if step == n_steps:
            if g_done > 0:
                n_body = n_tiles + ORI_SPARSE_PRELOAD_NUM

        for i in tla.range(n_body):
            x = g_done + i

            # ================================================================
            # OT ProcessVec0(runInfo[smlaRunInfoIdx])（:606）+ SetRunInfo
            # ================================================================
            if i < n_tiles:
                j0 = i
                nv0 = lines - j0 * KV_BLOCK
                if nv0 > KV_BLOCK:
                    nv0 = tla.as_numeric(KV_BLOCK)
                nl0 = nv0
                if nl0 < 0:
                    nl0 = tla.as_numeric(0)
                r0_task = task
                r0_j = j0
                r0_nt = n_tiles
                r0_nvalid = nv0
                r0_nload = nl0
                base_line0 = j0 * KV_BLOCK
                slot0 = x - (x // KV_SLOTS) * KV_SLOTS
                islot = x - (x // 2) * 2
                if kv_layout == 2 and not use_vec_gather:
                    gather_kv_rows_pa_pairwise(
                        sync,
                        bufs,
                        ws,
                        ori_kv,
                        ori_idx,
                        tok,
                        idx_width,
                        base_line0,
                        nl0,
                        slot0,
                        KV_STAGE_STRIDE,
                        ws_base,
                        block_table=ori_bt,
                        bt_seq=seq,
                        bt_max=ori_pa_max_blocks,
                        pa_block_size=ori_pa_block_size,
                        pa_stride=ori_pa_stride,
                        topk_value_mode=topk_value_mode,
                    )
                elif is_vec_s2phyaddr == 1:
                    gather_kv_rows_ring(
                        sync,
                        bufs,
                        ws,
                        ori_kv,
                        ori_idx,
                        tok,
                        idx_width,
                        kv_row0,
                        base_line0,
                        nl0,
                        slot0,
                        KV_STAGE_STRIDE,
                        islot,
                        ws_base,
                    )
                else:
                    gather_kv_rows(
                        sync,
                        bufs,
                        ws,
                        ori_kv,
                        ori_idx,
                        tok,
                        idx_width,
                        base_line0,
                        nl0,
                        kv_row0,
                        slot0,
                        KV_STAGE_STRIDE,
                        ws_base,
                        block_table=ori_bt,
                        bt_seq=seq,
                        bt_max=ori_pa_max_blocks,
                        pa_block_size=ori_pa_block_size,
                        pa_stride=ori_pa_stride,
                        topk_value_mode=topk_value_mode,
                    )

            # ================================================================
            # OT IterateLoadQK(runInfo[(taskId+3)%4], isFirstLoop)（:628），
            # ================================================================
            if x >= 1:
                if i < n_tiles + 1:
                    xg1 = x - 1
                    slot_k1 = xg1 - (xg1 // KV_SLOTS) * KV_SLOTS
                    if xg1 == 0:
                        qn1 = (r1_task - task0) // q_bn
                        q2n1 = qn1 + qn1
                        qa1 = q2n1 - (q2n1 // 3) * 3
                        q2n1b = q2n1 + 1
                        qb1 = q2n1b - (q2n1b // 3) * 3
                        load_q(
                            sync, bufs, q, r1_task, m, dtype_in, slot_a=qa1, slot_b=qb1
                        )
                    if g_split == 2:
                        if s2_budget > 0:
                            s2_budget = s2_budget - 1
                            split_g_mte2_pair(sync)
                    wait_kv_stage(sync)
                    load_k_ptr(
                        sync,
                        bufs,
                        ws.ptr + ws_base + slot_k1 * KV_STAGE_STRIDE,
                        r1_nload,
                        slot_k1,
                        dtype_in,
                    )
                else:
                    if g_split == 2:
                        if s2_budget > 0:
                            s2_budget = s2_budget - 1
                            split_g_mte2_pair(sync)

            # ================================================================
            # IterateBmm1(bmm1Slot, smlaNotLastTwoLoop, Next=(taskId+3)%4,
            # ================================================================
            if x >= 2:
                if i < n_tiles + 2:
                    xg2 = x - 2
                    slot_s2 = xg2 - (xg2 // P_SLOTS) * P_SLOTS
                    slot_k2 = xg2 - (xg2 // KV_SLOTS) * KV_SLOTS
                    slot_sum_rd = slot_s2
                    if r2_j > 0:
                        slot_sum_rd = 1 - slot_s2
                    q_nta2 = (r2_task - task0) // q_bn
                    q_two2 = q_nta2 + q_nta2
                    qa2 = q_two2 - (q_two2 // 3) * 3
                    q_two2b = q_two2 + 1
                    qb2 = q_two2b - (q_two2b // 3) * 3
                    if r2_j == 0:
                        goff2 = (r2_task - (r2_task // g_split) * g_split) * m
                        vec_init_from_sinks(
                            sync,
                            bufs,
                            sinks,
                            r2_task,
                            m,
                            rows,
                            dtype_in,
                            goff2,
                            has_sinks,
                            slot_sum=slot_s2,
                            slot_max=slot_s2,
                            clear_o=0,
                        )
                    process_vec1(
                        sync,
                        bufs,
                        softmax_scale,
                        r2_nvalid,
                        m,
                        rows,
                        dtype_in,
                        slot_s=slot_s2,
                        slot_p=slot_s2,
                        preload=1,
                        slot_sum_rd=slot_sum_rd,
                        slot_max_rd=slot_sum_rd,
                    )
                    iterate_bmm1(
                        sync,
                        bufs,
                        slot_k2,
                        m,
                        r2_j + 1 == r2_nt,
                        r2_j == 0,
                        dtype_in,
                        slot_s=slot_s2,
                        q_slot_a=qa2,
                        q_slot_b=qb2,
                    )
                    if r2_j + 1 == r2_nt:
                        if i < n_tiles + 1:
                            qn_p = q_nta2 + 1
                            q2n_p = qn_p + qn_p
                            qap = q2n_p - (q2n_p // 3) * 3
                            q2np_b = q2n_p + 1
                            qbp = q2np_b - (q2np_b // 3) * 3
                            load_q(
                                sync,
                                bufs,
                                q,
                                r2_task + q_bn,
                                m,
                                dtype_in,
                                slot_a=qap,
                                slot_b=qbp,
                            )

            # ================================================================
            # ================================================================
            if x >= 3:
                xg3 = x - 3
                slot_k3 = xg3 - (xg3 // KV_SLOTS) * KV_SLOTS
                slot_p3 = xg3 - (xg3 // P_SLOTS) * P_SLOTS
                wait_p_ready(sync)
                for seg in (0, 1, 2, 3):
                    iterate_bmm2_seg(
                        sync, bufs, seg, slot_k3, m, dtype_in, slot_p=slot_p3
                    )
                iterate_bmm2_fixpipe(sync, bufs, m)
                # Bool predicate, got i32）
                fresh_v3 = r3_j == 0
                for seg in (0, 1, 2, 3):
                    process_vec2_seg(
                        sync, bufs, seg, m, rows, dtype_in, fresh_v3, slot_dm=slot_p3
                    )
                release_k(sync, slot_k3)
                if r3_j + 1 == r3_nt:
                    bmm2_data_copy_out(
                        sync, bufs, out, r3_task, m, rows, dtype_in, slot_sum=slot_p3
                    )
                    if enable_lse == 1:
                        lse_copy_out(
                            sync,
                            bufs,
                            lse,
                            r3_task,
                            m,
                            rows,
                            slot=slot_p3,
                            row_base=(r3_task // g_split) * head_num
                            + (r3_task - (r3_task // g_split) * g_split) * half_g,
                        )

            rt = r2_task
            r3_task = rt
            rt = r2_j
            r3_j = rt
            rt = r2_nt
            r3_nt = rt
            rt = r2_nvalid
            r3_nvalid = rt
            rt = r2_nload
            r3_nload = rt
            r2_task = r1_task
            r2_j = r1_j
            r2_nt = r1_nt
            r2_nvalid = r1_nvalid
            r2_nload = r1_nload
            if i < n_tiles:
                r1_task = r0_task
                r1_j = r0_j
                r1_nt = r0_nt
                r1_nvalid = r0_nvalid
                r1_nload = r0_nload

        g_done = g_done + n_tiles

    if g_split == 2:
        for _k in tla.range(s2_budget):
            split_g_mte2_pair(sync)


ORI_CMP_PRELOAD_NUM = 3


@tla.kernel
def sparse_flash_mla_ori_cmp_sparse_kernel(
    q: tla.Tensor,
    ori_kv: tla.Tensor,
    cmp_kv: tla.Tensor,
    ori_bt: tla.Tensor,
    cmp_bt: tla.Tensor,
    ori_idx: tla.Tensor,
    ori_topk: tla.Tensor,  # [T] int32
    cmp_idx: tla.Tensor,
    cmp_topk: tla.Tensor,  # [T] int32
    cu_q: tla.Tensor,  # [B+1] int32
    cu_kv: tla.Tensor,  # [B+1] int32
    cu_cmp: tla.Tensor,  # [B+1] int32
    resid: tla.Tensor,  # [B] int32
    sinks: tla.Tensor,  # [H] fp32
    out: tla.Tensor,  # [T*H, D]
    lse: tla.Tensor,
    tiling: tla.Tensor,
    ws: tla.Tensor,  # WS_BLOCKS * KV_SLOTS * 128 * 512
    ori_phy: tla.Tensor,
    cmp_phy: tla.Tensor,
    softmax_scale: tla.Float32,
    head_num: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    num_seqs: tla.Constexpr[int],
    ori_idx_width: tla.Constexpr[int],
    cmp_idx_width: tla.Constexpr[int],
    ori_mask_mode: tla.Constexpr[int],
    ori_win_left: tla.Constexpr[int],
    ori_win_right: tla.Constexpr[int],
    cmp_mask_mode: tla.Constexpr[int],
    cmp_ratio: tla.Constexpr[int],
    #   phys ← block_table[seq*max_blocks + logical_pos // block_size]
    #   addr ← phys*pa_stride + (logical_pos % block_size)*HEAD_DIM
    ori_pa_block_size: tla.Constexpr[int],
    ori_pa_stride: tla.Constexpr[int],
    ori_pa_max_blocks: tla.Constexpr[int],
    cmp_pa_block_size: tla.Constexpr[int],
    cmp_pa_stride: tla.Constexpr[int],
    cmp_pa_max_blocks: tla.Constexpr[int],
    g_split: tla.Constexpr[int],
    has_sinks: tla.Constexpr[int],
    enable_lse: tla.Constexpr[int],
    topk_value_mode: tla.Constexpr[int],  # F5: 1=TOPK_INDEX, 2=TOPK_OFFSET
    kv_layout: tla.Constexpr[int],
    is_vec_s2phyaddr: tla.Constexpr[int],
    ori_phy_width: tla.Constexpr[int],
    cmp_phy_width: tla.Constexpr[int],
) -> None:
    dtype_in = q.ptr.dtype
    # IS_SPLIT_G（OT: tiling splitG = gSize > 64，common/smla_kernel_common_arch35.h:273-277）：
    m = head_num // g_split
    rows = m - m // 2
    half_g = head_num // g_split if head_num % g_split == 0 else head_num // g_split + 1

    _PAIRED = g_split == 2
    configure_trace(g_split, _PAIRED, _PAIRED, KV_SLOTS, "ocs")

    use_vec_gather = (is_vec_s2phyaddr == 1) and (kv_layout != 2)
    idx_ring_on = use_vec_gather and kv_layout != 2

    sync = declare_sync(
        need_rendezvous=_TRACE_STAGE,
        need_idx_ring=idx_ring_on,
        need_lse_war=enable_lse == 1,
    )
    phy_cap = ori_phy_width if ori_phy_width >= cmp_phy_width else cmp_phy_width
    bt_cap = (
        ori_pa_max_blocks
        if ori_pa_max_blocks >= cmp_pa_max_blocks
        else cmp_pa_max_blocks
    )
    bufs = alloc_buffers(
        dtype_in,
        phy=phy_cap if use_vec_gather else 0,
        phy_bt=bt_cap if use_vec_gather else 0,
        preload=True,
        idx_ring=idx_ring_on,
    )
    init_sync(
        sync,
        pre_phy_vs=use_vec_gather and not idx_ring_on,
        pre_l0c2=True,
        pre_idx_ring=idx_ring_on,
    )
    init_sync_preload(sync)
    need_init = compute_need_init(cu_q, cu_kv, num_seqs, ori_mask_mode, 1)
    clean_output(
        sync,
        bufs,
        out,
        lse,
        need_init,
        cu_q[num_seqs] * head_num * head_dim,
        0,
        enable_lse,
        dtype_in,
        head_dim=head_dim,
    )

    init_aiv_arrive = tla.cross_flag("smla_init_aiv_arrive", mode=4)
    init_aic_ffts = tla.cross_flag("smla_init_aic_ffts", mode=0)
    init_aic_done = tla.cross_flag("smla_init_aic_done", mode=2)
    if is_vec_s2phyaddr == 1:
        with tla.cube():
            tla.pipe_barrier(tla.pipes.ALL)
            tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=0)
            tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=1)
            tla.cross_core_set_flag(init_aic_ffts, tla.pipes.FIX)
            tla.cross_core_wait_flag(init_aic_ffts, tla.pipes.SCALAR)
            tla.cross_core_set_flag(init_aic_done, tla.pipes.SCALAR)
        with tla.vector():
            tla.pipe_barrier(tla.pipes.ALL)
            if tla.arch.sub_block_idx() == 0:
                tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=0)
            else:
                tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=1)
            tla.cross_core_wait_flag(init_aic_done, tla.pipes.SCALAR)
    else:
        if need_init == 1:
            with tla.cube():
                tla.pipe_barrier(tla.pipes.ALL)
                tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=0)
                tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=1)
                tla.cross_core_set_flag(init_aic_ffts, tla.pipes.FIX)
                tla.cross_core_wait_flag(init_aic_ffts, tla.pipes.SCALAR)
                tla.cross_core_set_flag(init_aic_done, tla.pipes.SCALAR)
            with tla.vector():
                tla.pipe_barrier(tla.pipes.ALL)
                if tla.arch.sub_block_idx() == 0:
                    tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=0)
                else:
                    tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=1)
                tla.cross_core_wait_flag(init_aic_done, tla.pipes.SCALAR)

    total_tasks = cu_q[num_seqs] * g_split
    task0 = tla.arch.block_idx()
    q_bn = tla.arch.block_num()

    # `GetKVPhyAddr(hasLoad, bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, …)`
    if is_vec_s2phyaddr == 1:
        with tla.cube():
            tla.pipe_barrier(tla.pipes.ALL)
            tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=0)
            tla.cross_core_wait_flag(init_aiv_arrive, tla.pipes.SCALAR, aiv_id=1)
            tla.cross_core_set_flag(init_aic_ffts, tla.pipes.FIX)
            tla.cross_core_wait_flag(init_aic_ffts, tla.pipes.SCALAR)
            tla.cross_core_set_flag(init_aic_done, tla.pipes.SCALAR)
        with tla.vector():
            tla.pipe_barrier(tla.pipes.ALL)
            if tla.arch.sub_block_idx() == 0:
                tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=0)
            else:
                tla.cross_core_set_flag(init_aiv_arrive, tla.pipes.MTE3, aiv_id=1)
            tla.cross_core_wait_flag(init_aic_done, tla.pipes.SCALAR)

    s2_budget = tla.as_numeric(0)
    c_one = 1
    if _PAIRED:
        s2_budget = tiling[c_one]

    r0_task = tla.as_numeric(0)
    r0_j = tla.as_numeric(0)
    r0_nt = tla.as_numeric(1)
    r0_nvalid = tla.as_numeric(0)
    r0_nload = tla.as_numeric(0)
    r1_task = tla.as_numeric(0)
    r1_j = tla.as_numeric(0)
    r1_nt = tla.as_numeric(1)
    r1_nvalid = tla.as_numeric(0)
    r1_nload = tla.as_numeric(0)
    r2_task = tla.as_numeric(0)
    r2_j = tla.as_numeric(0)
    r2_nt = tla.as_numeric(1)
    r2_nvalid = tla.as_numeric(0)
    r2_nload = tla.as_numeric(0)
    r3_task = tla.as_numeric(0)
    r3_j = tla.as_numeric(0)
    r3_nt = tla.as_numeric(1)
    r3_nvalid = tla.as_numeric(0)
    r3_nload = tla.as_numeric(0)

    g_done = tla.as_numeric(0)
    ws_base = tla.arch.block_idx() * (KV_SLOTS * KV_STAGE_STRIDE)
    #
    SHARE_SLOTS = _TRACE_SLOTS
    if _TRACE_STAGE:
        ws_base = (tla.arch.block_idx() // 2) * (SHARE_SLOTS * KV_STAGE_STRIDE)

    n_steps = (total_tasks - task0 + q_bn - 1) // q_bn
    if n_steps < 0:
        n_steps = tla.as_numeric(0)

    if _TRACE_STAGE:
        n_steps_g = (total_tasks + q_bn - 1) // q_bn
        if n_steps_g < 0:
            n_steps_g = tla.as_numeric(0)
        if _PAIRED:
            s2_budget = tiling[c_one] * n_steps_g

    for step in tla.range(n_steps + 1):
        task = task0 + step * q_bn

        n_tiles = tla.as_numeric(0)
        n_ori_tiles = tla.as_numeric(0)
        seq = tla.as_numeric(0)
        tok = tla.as_numeric(0)
        rowq = tla.as_numeric(0)
        kv_len = tla.as_numeric(0)
        ori_lines = tla.as_numeric(0)
        cmp_lines = tla.as_numeric(0)
        kv_row0 = tla.as_numeric(0)
        cmp_row0 = tla.as_numeric(0)

        if step < n_steps:
            tok = task // g_split
            rowq = tok
            for bb in static_range(1, num_seqs):
                if tok >= cu_q[bb]:
                    seq = bb
                    rowq = tok - cu_q[bb]
            q_len = cu_q[seq + 1] - cu_q[seq]
            kv_len = cu_kv[seq + 1] - cu_kv[seq]
            cmp_len = cu_cmp[seq + 1] - cu_cmp[seq]

            causal_offset = kv_len - q_len
            next_tok = kv_len
            pre_tok = q_len
            if ori_mask_mode == 3:
                next_tok = causal_offset
                pre_tok = q_len
            elif ori_mask_mode == 4:
                if ori_win_left == -1:
                    pre_tok = q_len
                else:
                    pre_tok = ori_win_left - causal_offset
                if ori_win_right == -1:
                    next_tok = kv_len
                else:
                    next_tok = causal_offset + ori_win_right
                if pre_tok > q_len:
                    pre_tok = q_len
            end_idx = rowq + next_tok + 1
            if end_idx < 0:
                end_idx = tla.as_numeric(0)
            if end_idx > kv_len:
                end_idx = kv_len
            start_idx = rowq - pre_tok
            if start_idx < 0:
                start_idx = tla.as_numeric(0)
            if start_idx > kv_len:
                start_idx = kv_len
            range_len = end_idx - start_idx
            if range_len < 0:
                range_len = tla.as_numeric(0)

            ori_count = ori_topk[tok]
            if ori_count > ori_idx_width:
                ori_count = tla.as_numeric(ori_idx_width)
            if ori_count < 0:
                ori_count = tla.as_numeric(0)
            ori_lines = range_len
            if ori_lines > ori_count:
                ori_lines = ori_count
            if ori_lines > kv_len:
                ori_lines = kv_len
            if ori_lines < 0:
                ori_lines = tla.as_numeric(0)
            n_ori_tiles = (ori_lines + KV_BLOCK - 1) // KV_BLOCK

            cmp_count = cmp_topk[tok]
            if cmp_count > cmp_idx_width:
                cmp_count = tla.as_numeric(cmp_idx_width)
            if cmp_count < 0:
                cmp_count = tla.as_numeric(0)
            # nextTokensPerBatchCmp（kvcache.h:86-93）：
            next_cmp = cmp_len * cmp_ratio
            if cmp_mask_mode != 0:
                next_cmp = cmp_len * cmp_ratio + resid[seq] - q_len
            cmp_end = (rowq + 1 + next_cmp) // cmp_ratio
            if cmp_end < 0:
                cmp_end = tla.as_numeric(0)
            if cmp_end > cmp_len:
                cmp_end = cmp_len
            cmp_lines = cmp_end
            if cmp_lines > cmp_count:
                cmp_lines = cmp_count
            n_cmp_tiles = (cmp_lines + KV_BLOCK - 1) // KV_BLOCK

            n_tiles = n_ori_tiles + n_cmp_tiles
            kv_row0 = cu_kv[seq]
            cmp_row0 = cu_cmp[seq]

        n_body = n_tiles
        if step == n_steps:
            if g_done > 0:
                n_body = n_tiles + ORI_CMP_PRELOAD_NUM

        for i in tla.range(n_body):
            x = g_done + i

            # ================================================================
            # OT ProcessVec0(runInfo[smlaRunInfoIdx])（:606）+ SetRunInfo
            # ================================================================
            if i < n_tiles:
                j0 = i
                nv0 = tla.as_numeric(0)
                nl0 = tla.as_numeric(0)
                slot0 = x - (x // SHARE_SLOTS) * SHARE_SLOTS
                islot = x - (x // 2) * 2
                if j0 < n_ori_tiles:
                    base_line0 = j0 * KV_BLOCK
                    nv0 = ori_lines - base_line0
                    if nv0 > KV_BLOCK:
                        nv0 = tla.as_numeric(KV_BLOCK)
                    nl0 = nv0
                    if nl0 < 0:
                        nl0 = tla.as_numeric(0)
                    if kv_layout == 2 and not use_vec_gather:
                        gather_kv_rows_pa_pairwise(
                            sync,
                            bufs,
                            ws,
                            ori_kv,
                            ori_idx,
                            tok,
                            ori_idx_width,
                            base_line0,
                            nl0,
                            slot0,
                            KV_STAGE_STRIDE,
                            ws_base,
                            block_table=ori_bt,
                            bt_seq=seq,
                            bt_max=ori_pa_max_blocks,
                            pa_block_size=ori_pa_block_size,
                            pa_stride=ori_pa_stride,
                            topk_value_mode=topk_value_mode,
                        )
                    elif is_vec_s2phyaddr == 1:
                        gather_kv_rows_ring(
                            sync,
                            bufs,
                            ws,
                            ori_kv,
                            ori_idx,
                            tok,
                            ori_idx_width,
                            kv_row0,
                            base_line0,
                            nl0,
                            slot0,
                            KV_STAGE_STRIDE,
                            islot,
                            ws_base,
                        )
                    else:
                        gather_kv_rows(
                            sync,
                            bufs,
                            ws,
                            ori_kv,
                            ori_idx,
                            tok,
                            ori_idx_width,
                            base_line0,
                            nl0,
                            kv_row0,
                            slot0,
                            KV_STAGE_STRIDE,
                            ws_base,
                            block_table=ori_bt,
                            bt_seq=seq,
                            bt_max=ori_pa_max_blocks,
                            pa_block_size=ori_pa_block_size,
                            pa_stride=ori_pa_stride,
                            topk_value_mode=topk_value_mode,
                        )
                else:
                    ct0 = j0 - n_ori_tiles
                    base_line0 = ct0 * KV_BLOCK
                    nv0 = cmp_lines - base_line0
                    if nv0 > KV_BLOCK:
                        nv0 = tla.as_numeric(KV_BLOCK)
                    nl0 = nv0
                    if nl0 < 0:
                        nl0 = tla.as_numeric(0)
                    if kv_layout == 2 and not use_vec_gather:
                        gather_kv_rows_pa_pairwise(
                            sync,
                            bufs,
                            ws,
                            cmp_kv,
                            cmp_idx,
                            tok,
                            cmp_idx_width,
                            base_line0,
                            nl0,
                            slot0,
                            KV_STAGE_STRIDE,
                            ws_base,
                            block_table=cmp_bt,
                            bt_seq=seq,
                            bt_max=cmp_pa_max_blocks,
                            pa_block_size=cmp_pa_block_size,
                            pa_stride=cmp_pa_stride,
                            topk_value_mode=topk_value_mode,
                        )
                    elif is_vec_s2phyaddr == 1:
                        gather_kv_rows_ring(
                            sync,
                            bufs,
                            ws,
                            cmp_kv,
                            cmp_idx,
                            tok,
                            cmp_idx_width,
                            cmp_row0,
                            base_line0,
                            nl0,
                            slot0,
                            KV_STAGE_STRIDE,
                            islot,
                            ws_base,
                        )
                    else:
                        gather_kv_rows(
                            sync,
                            bufs,
                            ws,
                            cmp_kv,
                            cmp_idx,
                            tok,
                            cmp_idx_width,
                            base_line0,
                            nl0,
                            cmp_row0,
                            slot0,
                            KV_STAGE_STRIDE,
                            ws_base,
                            block_table=cmp_bt,
                            bt_seq=seq,
                            bt_max=cmp_pa_max_blocks,
                            pa_block_size=cmp_pa_block_size,
                            pa_stride=cmp_pa_stride,
                            topk_value_mode=topk_value_mode,
                        )
                r0_task = task
                r0_j = j0
                r0_nt = n_tiles
                r0_nvalid = nv0
                r0_nload = nl0

            # ================================================================
            # OT IterateLoadQK(runInfo[(taskId+3)%4], isFirstLoop)（:628），
            # ================================================================
            if x >= 1:
                if i < n_tiles + 1:
                    xg1 = x - 1
                    slot_k1 = xg1 - (xg1 // KV_SLOTS) * KV_SLOTS
                    slot_st1 = slot_k1
                    if SHARE_SLOTS != KV_SLOTS:
                        slot_st1 = xg1 - (xg1 // SHARE_SLOTS) * SHARE_SLOTS
                    if xg1 == 0:
                        qn1 = (r1_task - task0) // q_bn
                        q2n1 = qn1 + qn1
                        qa1 = q2n1 - (q2n1 // 3) * 3
                        q2n1b = q2n1 + 1
                        qb1 = q2n1b - (q2n1b // 3) * 3
                        load_q(
                            sync, bufs, q, r1_task, m, dtype_in, slot_a=qa1, slot_b=qb1
                        )
                    if _TRACE_STAGE:
                        wait_kv_stage(sync)
                        if s2_budget > 0:
                            s2_budget = s2_budget - 1
                            split_g_mte2_pair(sync)
                    else:
                        if _PAIRED:
                            if s2_budget > 0:
                                s2_budget = s2_budget - 1
                                split_g_mte2_pair(sync)
                        wait_kv_stage(sync)
                    load_k_ptr(
                        sync,
                        bufs,
                        ws.ptr + ws_base + slot_st1 * KV_STAGE_STRIDE,
                        r1_nload,
                        slot_k1,
                        dtype_in,
                    )
                else:
                    if _PAIRED:
                        if s2_budget > 0:
                            s2_budget = s2_budget - 1
                            split_g_mte2_pair(sync)

            # ================================================================
            # IterateBmm1(bmm1Slot, smlaNotLastTwoLoop, Next=(taskId+3)%4,
            # ================================================================
            if x >= 2:
                if i < n_tiles + 2:
                    xg2 = x - 2
                    slot_s2 = xg2 - (xg2 // P_SLOTS) * P_SLOTS
                    slot_k2 = xg2 - (xg2 // KV_SLOTS) * KV_SLOTS
                    slot_sum_rd = slot_s2
                    if r2_j > 0:
                        slot_sum_rd = 1 - slot_s2
                    q_nta2 = (r2_task - task0) // q_bn
                    q_two2 = q_nta2 + q_nta2
                    qa2 = q_two2 - (q_two2 // 3) * 3
                    q_two2b = q_two2 + 1
                    qb2 = q_two2b - (q_two2b // 3) * 3
                    if r2_j == 0:
                        goff2 = (r2_task - (r2_task // g_split) * g_split) * m
                        vec_init_from_sinks(
                            sync,
                            bufs,
                            sinks,
                            r2_task,
                            m,
                            rows,
                            dtype_in,
                            goff2,
                            has_sinks,
                            slot_sum=slot_s2,
                            slot_max=slot_s2,
                            clear_o=0,
                        )
                    process_vec1(
                        sync,
                        bufs,
                        softmax_scale,
                        r2_nvalid,
                        m,
                        rows,
                        dtype_in,
                        slot_s=slot_s2,
                        slot_p=slot_s2,
                        preload=1,
                        slot_sum_rd=slot_sum_rd,
                        slot_max_rd=slot_sum_rd,
                    )
                    iterate_bmm1(
                        sync,
                        bufs,
                        slot_k2,
                        m,
                        r2_j + 1 == r2_nt,
                        r2_j == 0,
                        dtype_in,
                        slot_s=slot_s2,
                        q_slot_a=qa2,
                        q_slot_b=qb2,
                    )
                    if r2_j + 1 == r2_nt:
                        if i < n_tiles + 1:
                            qn_p = q_nta2 + 1
                            q2n_p = qn_p + qn_p
                            qap = q2n_p - (q2n_p // 3) * 3
                            q2np_b = q2n_p + 1
                            qbp = q2np_b - (q2np_b // 3) * 3
                            load_q(
                                sync,
                                bufs,
                                q,
                                r2_task + q_bn,
                                m,
                                dtype_in,
                                slot_a=qap,
                                slot_b=qbp,
                            )

            # ================================================================
            # ================================================================
            if x >= 3:
                xg3 = x - 3
                slot_k3 = xg3 - (xg3 // KV_SLOTS) * KV_SLOTS
                slot_p3 = xg3 - (xg3 // P_SLOTS) * P_SLOTS
                wait_p_ready(sync)
                for seg in (0, 1, 2, 3):
                    iterate_bmm2_seg(
                        sync, bufs, seg, slot_k3, m, dtype_in, slot_p=slot_p3
                    )
                iterate_bmm2_fixpipe(sync, bufs, m)
                # Bool predicate, got i32）
                fresh_v3 = r3_j == 0
                for seg in (0, 1, 2, 3):
                    process_vec2_seg(
                        sync, bufs, seg, m, rows, dtype_in, fresh_v3, slot_dm=slot_p3
                    )
                release_k(sync, slot_k3)
                if r3_j + 1 == r3_nt:
                    bmm2_data_copy_out(
                        sync, bufs, out, r3_task, m, rows, dtype_in, slot_sum=slot_p3
                    )
                    if enable_lse == 1:
                        lse_copy_out(
                            sync,
                            bufs,
                            lse,
                            r3_task,
                            m,
                            rows,
                            slot=slot_p3,
                            row_base=(r3_task // g_split) * head_num
                            + (r3_task - (r3_task // g_split) * g_split) * half_g,
                        )

            rt = r2_task
            r3_task = rt
            rt = r2_j
            r3_j = rt
            rt = r2_nt
            r3_nt = rt
            rt = r2_nvalid
            r3_nvalid = rt
            rt = r2_nload
            r3_nload = rt
            r2_task = r1_task
            r2_j = r1_j
            r2_nt = r1_nt
            r2_nvalid = r1_nvalid
            r2_nload = r1_nload
            if i < n_tiles:
                r1_task = r0_task
                r1_j = r0_j
                r1_nt = r0_nt
                r1_nvalid = r0_nvalid
                r1_nload = r0_nload

        g_done = g_done + n_tiles

    if _PAIRED:
        for _k in tla.range(s2_budget):
            if _TRACE_STAGE:
                pair_rendezvous(sync)
            else:
                split_g_mte2_pair(sync)

    with tla.vector():
        tla.set_flag(sync.init_mte3_v)
        tla.wait_flag(sync.init_mte3_v)

    verify_trace_closed("ocs")


_SINGLE_QUERY_KV_SOURCE = r"""
#include "kernel_operator.h"

using namespace AscendC::MicroAPI;

static constexpr CastTrait expand = {RegLayout::ZERO, SatMode::UNKNOWN,
                                  MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr CastTrait narrow = {RegLayout::ZERO, SatMode::NO_SAT,
                                  MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

__simd_callee__ inline RegTensor<bfloat16_t> DecodeGroup(
    __ubuf__ uint8_t* record, int32_t group, MaskReg f32Mask) {
    RegTensor<fp8_e4m3fn_t> fp8;
    RegTensor<bfloat16_t> scale16, packed;
    RegTensor<float> value, scale;
    LoadAlign<fp8_e4m3fn_t, LoadDist::DIST_UNPACK4_B8>(
        fp8, reinterpret_cast<__ubuf__ fp8_e4m3fn_t *>(record + 128 + group * 64));
    LoadAlign<bfloat16_t, LoadDist::DIST_BRC_B16>(
        scale16, reinterpret_cast<__ubuf__ bfloat16_t *>(record + 576) + group);
    Cast<float, fp8_e4m3fn_t, expand>(value, fp8, f32Mask);
    Cast<float, bfloat16_t, expand>(scale, scale16, f32Mask);
    Mul(value, value, scale, f32Mask);
    Cast<bfloat16_t, float, narrow>(packed, value, f32Mask);
    return packed;
}

__simd_vf__ void DequantFp8RowsNd(__ubuf__ uint8_t* raw, __ubuf__ bfloat16_t* output, int32_t rows) {
    auto all = CreateMask<bfloat16_t, MaskPattern::ALL>();
    auto half = CreateMask<bfloat16_t, MaskPattern::H>();
    auto width = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < rows; ++row) {
        auto* record = raw + row * 608;
        auto* out = output + row * 512;
        #pragma unroll
        for (uint16_t group = 0; group < 7; ++group) {
            auto values = DecodeGroup(record, group, width);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(out + group * 64, values, all);
        }
        RegTensor<bfloat16_t> rope;
        LoadAlign(rope, reinterpret_cast<__ubuf__ bfloat16_t*>(record));
        StoreAlign(out + 448, rope, half);
    }
}

extern "C" [aicore] __attribute__((always_inline)) void dequant_fp8_records_nd(
    uint64_t records, uint64_t result, int32_t rows) {
    DequantFp8RowsNd(reinterpret_cast<__ubuf__ uint8_t*>(records),
                     reinterpret_cast<__ubuf__ bfloat16_t*>(result), rows);
}

__simd_callee__ inline void PackFp8Pair(
    __ubuf__ bfloat16_t* out, __ubuf__ uint8_t* record,
    uint16_t group, uint16_t row, uint16_t pitch, MaskReg& f32Mask, MaskReg& all) {
    auto lo = DecodeGroup(record, group, f32Mask);
    auto hi = DecodeGroup(record, group + 1, f32Mask);
    RegTensor<bfloat16_t> value, unused;
    DeInterleave(value, unused, lo, hi);
    auto* dst = out + group * 64 * pitch + row * 16;
    StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
        dst, value, pitch, 1, all);
}

template <bool UnrollGroups = false>
__simd_vf__ void PackFp8Kv(
    __ubuf__ bfloat16_t* out, __ubuf__ uint8_t* in,
    uint16_t rows, uint16_t pitch) {
    auto all = CreateMask<bfloat16_t, MaskPattern::ALL>();
    auto half = CreateMask<bfloat16_t, MaskPattern::H>();
    auto f32Mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<bfloat16_t> zero;
    Duplicate(zero, 0);
    for (uint16_t r = 0; r < rows; ++r) {
        auto* record = in + r * 608;
        if constexpr (UnrollGroups) {
            PackFp8Pair(out, record, 0, r, pitch, f32Mask, all);
            PackFp8Pair(out, record, 2, r, pitch, f32Mask, all);
            PackFp8Pair(out, record, 4, r, pitch, f32Mask, all);
        } else {
            for (uint16_t group = 0; group < 6; group += 2) {
                PackFp8Pair(out, record, group, r, pitch, f32Mask, all);
            }
        }
        RegTensor<bfloat16_t> value, unused, rope;
        auto tail = DecodeGroup(record, 6, f32Mask);
        DeInterleave(value, unused, tail, zero);
        auto* dstTail = out + 384 * pitch + r * 16;
        StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            dstTail, value, pitch, 1, half);
        LoadAlign(rope, reinterpret_cast<__ubuf__ bfloat16_t *>(record));
        auto* dstRope = out + 448 * pitch + r * 16;
        StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            dstRope, rope, pitch, 1, half);
    }
    for (uint16_t r = rows; r < pitch - 1; ++r) {
        for (uint16_t c = 0; c < 512; c += 128) {
            auto* dst = out + c * pitch + r * 16;
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                dst, zero, pitch, 1, all);
        }
    }
}

__simd_vf__ void PackFp8Columns(
    __ubuf__ bfloat16_t* out, __ubuf__ uint8_t* in,
    uint16_t rows, uint16_t pitch, uint16_t group) {
    auto all = CreateMask<bfloat16_t, MaskPattern::ALL>();
    auto half = CreateMask<bfloat16_t, MaskPattern::H>();
    auto f32Mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<bfloat16_t> zero;
    Duplicate(zero, 0);
    if (group < 6) {
        for (uint16_t r = 0; r < rows; ++r) {
            auto* record = in + r * 608;
            auto lo = DecodeGroup(record, group, f32Mask);
            auto hi = DecodeGroup(record, group + 1, f32Mask);
            RegTensor<bfloat16_t> value, unused;
            DeInterleave(value, unused, lo, hi);
            auto* dst = out + r * 16;
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                dst, value, pitch, 1, all);
        }
    } else {
        for (uint16_t r = 0; r < rows; ++r) {
            auto* record = in + r * 608;
            RegTensor<bfloat16_t> value, unused, rope;
            auto tail = DecodeGroup(record, 6, f32Mask);
            DeInterleave(value, unused, tail, zero);
            auto* dstTail = out + r * 16;
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                dstTail, value, pitch, 1, half);
            LoadAlign(rope, reinterpret_cast<__ubuf__ bfloat16_t *>(record));
            auto* dstRope = out + 64 * pitch + r * 16;
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                dstRope, rope, pitch, 1, half);
        }
    }
    for (uint16_t r = rows; r < pitch - 1; ++r) {
        auto* dst = out + r * 16;
        StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            dst, zero, pitch, 1, all);
    }
}

extern "C" {
[aicore] __attribute__((always_inline)) void single_query_insert_nz(
    uint64_t dst, uint64_t src, int32_t n, int32_t dst_rows,
    int32_t src_rows, int32_t cols) {
    copy_ubuf_to_cbuf(reinterpret_cast<__cbuf__ uint16_t *>(dst),
                     reinterpret_cast<__ubuf__ uint16_t *>(src),
                     0, cols / 16, n, src_rows - n, dst_rows - n);
}

[aicore] __attribute__((always_inline)) void pack_fp8_columns(
    uint64_t dst, uint64_t src, int32_t rows, int32_t pitch, int32_t group) {
    PackFp8Columns(reinterpret_cast<__ubuf__ bfloat16_t *>(dst),
                   reinterpret_cast<__ubuf__ uint8_t *>(src), rows, pitch, group);
}

[aicore] __attribute__((always_inline)) void single_query_dequant_nz(
    uint64_t dst, uint64_t src, int32_t rows, int32_t pitch) {
    PackFp8Kv(reinterpret_cast<__ubuf__ bfloat16_t *>(dst),
              reinterpret_cast<__ubuf__ uint8_t *>(src), rows, pitch);
}

[aicore] __attribute__((always_inline)) void decode_fp8_to_nz(
    uint64_t dst, uint64_t src, int32_t rows, int32_t pitch) {
    PackFp8Kv<true>(reinterpret_cast<__ubuf__ bfloat16_t *>(dst),
                   reinterpret_cast<__ubuf__ uint8_t *>(src), rows, pitch);
}

[aicore] __attribute__((always_inline)) void single_query_load_records(
    uint64_t dst, uint64_t src, uint64_t table, int32_t start, int32_t rows,
    int32_t page_size) {
    auto* out = reinterpret_cast<__ubuf__ uint8_t *>(dst);
    auto* in = reinterpret_cast<__gm__ uint8_t *>(src);
    auto* bt = reinterpret_cast<__gm__ int32_t *>(table);
    constexpr uint32_t row_bytes = 608;
    set_loop_size_outtoub(1ULL << 21 | 1ULL);
    for (int32_t done = 0; done < rows;) {
        int32_t logical_row = start + done;
        int32_t page = logical_row / page_size;
        int32_t row = logical_row - page * page_size;
        int32_t count = rows - done;
        if (count > page_size - row) {
            count = page_size - row;
        }
        int32_t physical_page = bt[page] >= 0 ? bt[page] : 0;
        uint64_t offset = (uint64_t)physical_page * page_size + row;
        uint32_t bytes = count * row_bytes;
        copy_gm_to_ubuf_align_v2(out + done * row_bytes, in + offset * row_bytes,
                                0, 1, bytes, 0, 0, false, 0, bytes, bytes);
        done += count;
    }
}

[aicore] __attribute__((always_inline)) uint64_t record_offset(
    __ubuf__ int32_t* index, __gm__ int32_t* pages, int32_t i, int32_t page_size) {
    int32_t position = index[i];
    position = position >= 0 ? position : 0;
    int32_t page = position / page_size;
    int32_t physical = pages[page];
    physical = physical >= 0 ? physical : 0;
    return ((uint64_t)physical * page_size + position - page * page_size) * 608;
}

[aicore] __attribute__((always_inline)) void gather_fp8_records(
    uint64_t dst, uint64_t src, uint64_t indices, uint64_t table,
    int32_t rows, int32_t page_size) {
    auto* output = reinterpret_cast<__ubuf__ uint8_t *>(dst);
    auto* input = reinterpret_cast<__gm__ uint8_t *>(src);
    auto* index = reinterpret_cast<__ubuf__ int32_t *>(indices);
    auto* pages = reinterpret_cast<__gm__ int32_t *>(table);
    constexpr uint32_t bytes = 608;
    set_loop_size_outtoub(1ULL << 21 | 1ULL);
    for (int32_t i = 0; i + 1 < rows; i += 2) {
        uint64_t a = record_offset(index, pages, i, page_size);
        uint64_t b = record_offset(index, pages, i + 1, page_size);
        auto* to = output + i * bytes;
        if (b > a && b - a < (1ULL << 30)) {
            copy_gm_to_ubuf_align_v2(to, input + a, 0, 2, bytes, 0, 0, false, 0, b - a, bytes);
        } else {
            copy_gm_to_ubuf_align_v2(to, input + a, 0, 1, bytes, 0, 0, false, 0, bytes, bytes);
            copy_gm_to_ubuf_align_v2(to + bytes, input + b, 0, 1, bytes, 0, 0, false, 0, bytes, bytes);
        }
    }
    if (rows & 1) {
        int32_t i = rows - 1;
        uint64_t row = record_offset(index, pages, i, page_size);
        copy_gm_to_ubuf_align_v2(output + i * bytes, input + row, 0, 1, bytes, 0, 0, false, 0, bytes, bytes);
    }
}
}
"""


@tla.extern(source=_SINGLE_QUERY_KV_SOURCE, name="dequant_fp8_records_nd")
def _dequant_fp8_records_nd(
    raw: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    out: tla.Pointer[tla.BFloat16, tla.AddressSpace.ub],
    rows: tla.Int32,
) -> None: ...


@tla.jit
def _single_query_insert_nz(
    dst: tla.Pointer[tla.UInt16, tla.AddressSpace.l1],
    src: tla.Pointer[tla.UInt16, tla.AddressSpace.ub],
    rows: tla.Int32,
    dst_pitch: tla.Int32,
    src_pitch: tla.Int32,
    cols: tla.Int32,
) -> None:
    dst_b16 = tla.recast_ptr(dst, dtype=tla.BFloat16)
    src_b16 = tla.recast_ptr(src, dtype=tla.BFloat16)
    ub_parent = tla.make_tensor(
        src_b16,
        tla.make_layout_with_tag((src_pitch, cols), tla.BFloat16, tla.arch.zNUnAlign),
    )
    l1_parent = tla.make_tensor(
        dst_b16,
        tla.make_layout_with_tag((dst_pitch, cols), tla.BFloat16, tla.arch.zN),
    )
    origin = tla.make_coord(0, 0)
    extent = tla.make_shape(rows, cols)
    tla.copy(
        tla.get_tile(l1_parent, origin, extent),
        tla.get_tile(ub_parent, origin, extent),
    )


@tla.extern(source=_SINGLE_QUERY_KV_SOURCE, name="pack_fp8_columns")
def _pack_fp8_columns(
    output: tla.Pointer[tla.UInt16, tla.AddressSpace.ub],
    input: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    rows: tla.Int32,
    pitch: tla.Int32,
    group: tla.Int32,
) -> None: ...


@tla.extern(source=_SINGLE_QUERY_KV_SOURCE, name="single_query_dequant_nz")
def _single_query_dequant_nz(
    dst: tla.Pointer[tla.UInt16, tla.AddressSpace.ub],
    src: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    rows: tla.Int32,
    pitch: tla.Int32,
) -> None: ...


@tla.jit
def _copy_byte_rows(dst, src, count, row_bytes, src_pitch):
    shape = tla.make_shape(count, row_bytes)
    gm = tla.make_tensor(
        src,
        tla.make_layout(
            shape, tla.make_stride(src_pitch, 1), layoutTag=tla.arch.RowMajor
        ),
    )
    ub = tla.make_tensor(
        dst,
        tla.make_layout(
            shape, tla.make_stride(row_bytes, 1), layoutTag=tla.arch.RowMajor
        ),
    )
    tla.copy(ub, gm)


@tla.jit
def _record_offset(table, logical, page_size, row_bytes):
    logical = logical if logical >= 0 else 0
    physical = table[logical // page_size]
    physical = physical if physical >= 0 else tla.Int32(0)
    return (
        physical.to(tla.Int64) * tla.Int64(page_size)
        + (logical % page_size).to(tla.Int64)
    ) * tla.Int64(row_bytes)


@tla.jit
def _single_query_load_records(
    dst: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    src: tla.Pointer[tla.UInt8, tla.AddressSpace.gm],
    page_table: tla.Pointer[tla.Int32, tla.AddressSpace.gm],
    start: tla.Int32,
    rows: tla.Int32,
    page_size: tla.Int32,
) -> None:
    pages_view = tla.make_tensor(
        page_table,
        tla.make_layout(
            tla.make_shape(rows // page_size + 2),
            tla.make_stride(1),
        ),
    )
    ps64 = tla.Int64(page_size)
    c608 = tla.Int64(608)
    done = tla.as_numeric(0)
    while done < rows:
        logical = start + done
        page = logical // page_size
        in_page = logical % page_size
        take = rows - done
        room = page_size - in_page
        take = take if take < room else room
        physical = pages_view[page]
        physical = physical if physical >= 0 else tla.Int32(0)
        offset = (physical.to(tla.Int64) * ps64 + in_page.to(tla.Int64)) * c608
        nbytes = take * 608
        _copy_byte_rows(dst + done * 608, src + offset, 1, nbytes, nbytes)
        done = done + take


@tla.jit
def _gather_fp8_records(
    dst: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    pool: tla.Pointer[tla.UInt8, tla.AddressSpace.gm],
    idx: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    table: tla.Pointer[tla.Int32, tla.AddressSpace.gm],
    rows: tla.Int32,
    page_size: tla.Int32,
) -> None:
    ids = tla.make_tensor(
        idx, tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
    )
    pages = tla.make_tensor(
        table,
        tla.make_layout(
            tla.make_shape(rows // page_size + 2),
            tla.make_stride(1),
        ),
    )
    pair_hi = tla.Int64(1 << 30)
    for i in tla.range(0, rows - 1, 2):
        a = _record_offset(pages, ids[i], page_size, 608)
        b = _record_offset(pages, ids[i + 1], page_size, 608)
        if b > a and b - a < pair_hi:
            _copy_byte_rows(dst + i * 608, pool + a, 2, 608, b - a)
        else:
            _copy_byte_rows(dst + i * 608, pool + a, 1, 608, 608)
            _copy_byte_rows(dst + i * 608 + 608, pool + b, 1, 608, 608)
    if rows % 2 == 1:
        tail = _record_offset(pages, ids[rows - 1], page_size, 608)
        _copy_byte_rows(dst + (rows - 1) * 608, pool + tail, 1, 608, 608)


_DECODE_CHUNK_ROWS = 16
_DECODE_CHUNK_PITCH = _DECODE_CHUNK_ROWS + 1
_DECODE_RAW_ROWS = (2 * _DECODE_CHUNK_ROWS * REC_BYTES + 2 * HEAD_DIM - 1) // (
    2 * HEAD_DIM
)


@tla.extern(source=_SINGLE_QUERY_KV_SOURCE, name="decode_fp8_to_nz")
def _decode_fp8_to_nz(
    output: tla.Pointer[tla.UInt16, tla.AddressSpace.ub],
    raw: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    rows: tla.Int32,
    pitch: tla.Int32,
) -> None: ...


@tla.jit
def _insert_decode_chunk(
    l1,
    input_ptr,
    packed_ptr,
    kv,
    indices,
    table,
    page_size,
    start,
    valid,
    offset,
    row,
    store_rows,
    sparse,
    record_lock,
    packed_lock,
    slot_free,
    first_chunk,
):
    count = valid - row
    count = count if count < store_rows else store_rows
    count = count if count > 0 else 0
    record = tla.recast_ptr(input_ptr, dtype=tla.UInt8)
    source = tla.recast_ptr(kv.ptr, dtype=tla.UInt8)
    packed = tla.recast_ptr(packed_ptr, dtype=tla.UInt16)
    record_lock.lock(pipe=tla.pipes.MTE2)
    if sparse:
        _gather_fp8_records(record, source, indices + offset, table, count, page_size)
    else:
        _single_query_load_records(record, source, table, start + row, count, page_size)
    record_lock.unlock(pipe=tla.pipes.MTE2)
    record_lock.lock(pipe=tla.pipes.VECTOR)
    packed_lock.lock(pipe=tla.pipes.VECTOR)
    _decode_fp8_to_nz(packed, record, count, _DECODE_CHUNK_PITCH)
    record_lock.unlock(pipe=tla.pipes.VECTOR)
    packed_lock.unlock(pipe=tla.pipes.VECTOR)
    packed_lock.lock(pipe=tla.pipes.MTE3)
    if first_chunk:
        tla.cross_core_wait_flag(slot_free, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_wait_flag(slot_free, tla.pipes.MTE3, aiv_id=1)
    _single_query_insert_nz(
        tla.recast_ptr(l1 + row * 16, dtype=tla.UInt16),
        packed,
        store_rows,
        KV_BLOCK,
        _DECODE_CHUNK_PITCH,
        HEAD_DIM,
    )
    packed_lock.unlock(pipe=tla.pipes.MTE3)


@tla.jit
def gather_decode_kv_to_l1(
    sync,
    bufs,
    kv,
    idx,
    idx0,
    bt,
    seq,
    bt_width,
    page_size,
    start,
    valid,
    slot,
    sparse,
):
    """Each AIV fills one half of the K/V slot, including its padded rows."""
    with tla.vector():
        half = (valid + 15) // 16 * 8
        first = tla.arch.sub_block_idx() * half
        l1 = _k_slot_ptr(bufs, slot)
        table = bt.ptr + seq * bt_width
        if sparse:
            local_valid = valid - first
            local_valid = local_valid if local_valid < half else half
            if local_valid > 0:
                copy_rows = (local_valid + 7) // 8 * 8
                layout = tla.make_layout(tla.make_shape(copy_rows), tla.make_stride(1))
                tla.copy(
                    tla.make_tensor(bufs.ub_phy_idx, layout),
                    tla.make_tensor(idx.ptr + idx0 + first, layout),
                )
                tla.set_flag(sync.idx_m2s0)
                tla.wait_flag(sync.idx_m2s0)
        for chunk in tla.range((half + _DECODE_CHUNK_ROWS - 1) // _DECODE_CHUNK_ROWS):
            offset = chunk * _DECODE_CHUNK_ROWS
            store_rows = half - offset
            store_rows = (
                store_rows if store_rows < _DECODE_CHUNK_ROWS else _DECODE_CHUNK_ROWS
            )
            if chunk % 2 == 0:
                _insert_decode_chunk(
                    l1,
                    bufs.ub_kv0,
                    bufs.ub_kv1,
                    kv,
                    bufs.ub_phy_idx,
                    table,
                    page_size,
                    start,
                    valid,
                    offset,
                    first + offset,
                    store_rows,
                    sparse,
                    sync.m_dq_rec0,
                    sync.m_dq_out0,
                    sync.kv_l1_free,
                    chunk == 0,
                )
            else:
                # The second packed tile shares GM-output staging. Its MTE3
                # transfer and the PV epilogue are ordered by the same mutex.
                _insert_decode_chunk(
                    l1,
                    bufs.ub_kv0 + _DECODE_CHUNK_ROWS * (REC_BYTES // 2),
                    bufs.ub_o16,
                    kv,
                    bufs.ub_phy_idx,
                    table,
                    page_size,
                    start,
                    valid,
                    offset,
                    first + offset,
                    store_rows,
                    sparse,
                    sync.m_dq_rec1,
                    sync.m_o16,
                    sync.kv_l1_free,
                    False,
                )
        tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def _single_query_load_segment(
    kv,
    table,
    page_size,
    start,
    valid,
    capacity,
    n,
    base,
    l1,
    ub_record,
    ub_nz,
    sync,
    table_start=0,
):
    with tla.vector():
        half = capacity // 2
        offset = tla.arch.sub_block_idx() * half
        count = valid - offset
        count = count if count < half else half
        count = count if count > 0 else 0
        sync.m_dq_rec0.lock(pipe=tla.pipes.MTE2)
        _single_query_load_records(
            ub_record,
            kv.ptr,
            table.ptr + table_start,
            start + offset,
            count,
            page_size,
        )
        sync.m_dq_rec0.unlock(pipe=tla.pipes.MTE2)
        sync.m_dq_rec0.lock(pipe=tla.pipes.VECTOR)
        sync.m_dq_out0.lock(pipe=tla.pipes.VECTOR)
        _single_query_dequant_nz(
            tla.recast_ptr(ub_nz, dtype=tla.UInt16), ub_record, count, half + 1
        )
        sync.m_dq_rec0.unlock(pipe=tla.pipes.VECTOR)
        sync.m_dq_out0.unlock(pipe=tla.pipes.VECTOR)
        sync.m_dq_out0.lock(pipe=tla.pipes.MTE3)
        _single_query_insert_nz(
            tla.recast_ptr(l1 + (base + offset) * 16, dtype=tla.UInt16),
            tla.recast_ptr(ub_nz, dtype=tla.UInt16),
            half,
            n,
            half + 1,
            HEAD_DIM,
        )
        sync.m_dq_out0.unlock(pipe=tla.pipes.MTE3)


@tla.jit
def _single_query_softmax(
    l1p,
    scores,
    prob_ub,
    sums,
    maxima,
    sinks,
    sink_ub,
    head_group,
    m,
    n,
    n_ori,
    n_cmp,
    scale,
    has_sinks,
    sync,
    shared_sinks=False,
    release_scores=False,
):
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        rows = m // 2
        dt = l1p.dtype
        p_pitch = ((rows + 15) // 16) * 16
        pitch = (n + 63) // 64 * 64
        ub_p = tla.make_tensor(
            prob_ub, tla.make_layout_with_tag((rows, pitch), dt, tla.arch.zN)
        )
        pc = tla.make_tensor(l1p, tla.make_layout_with_tag((m, n), dt, tla.arch.zN))
        sink_tile = tla.make_tensor(
            sink_ub, tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
        )
        if has_sinks and not shared_sinks:
            sink_gm = tla.make_tensor(
                sinks.ptr + head_group * m + sub * rows,
                tla.make_layout(tla.make_shape(rows), tla.make_stride(1)),
            )
            tla.copy(sink_tile, sink_gm)
            tla.set_flag(sync.phy_m2v)
            tla.wait_flag(sync.phy_m2v)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=1)
        sync.m_p.lock(pipe=tla.pipes.VECTOR)
        sb = tla.make_tensor(
            scores,
            tla.make_layout(tla.make_shape(rows, pitch), tla.make_stride(pitch, 1)),
        )
        ub_sum = _stats(sums, rows)
        ub_max = _stats(maxima, rows)
        nc0 = n_ori if n_ori < 64 else 64
        nc1 = n_ori - nc0
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            one_lane, _ = tla.update_mask(1, dtype=tla.Float32)
            mask0, _ = tla.update_mask(nc0, dtype=tla.Float32)
            mask1, _ = tla.update_mask(nc1, dtype=tla.Float32)
            mask2, _ = tla.update_mask(n_cmp, dtype=tla.Float32)
            minreg = tla.full(MIN_F, dtype=tla.Float32)
            cast0 = _cast_slot(tla.params.RegSlot.ZERO)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            pack_store = tla.params.BlockStoreParams(block_stride=p_pitch)
            with_tail = n > KV_BLOCK
            for r in tla.range(rows):
                sink = minreg
                if has_sinks:
                    sink = tla.tile_view(
                        sink_tile, tla.make_shape(1), tla.make_coord(r)
                    ).load(params=brc)
                s0 = tla.tile_view(
                    sb, tla.make_shape(1, 64), tla.make_coord(r, 0)
                ).load()
                s1 = tla.tile_view(
                    sb, tla.make_shape(1, 64), tla.make_coord(r, 1)
                ).load()
                v0 = tla.where(mask0, tla.mul(s0, scale, mask=preg), minreg)
                v1 = tla.where(mask1, tla.mul(s1, scale, mask=preg), minreg)
                vmax = tla.max(v0, v1, mask=preg)
                v2 = minreg
                if with_tail:
                    s2 = tla.tile_view(
                        sb, tla.make_shape(1, 64), tla.make_coord(r, 2)
                    ).load()
                    v2 = tla.where(mask2, tla.mul(s2, scale, mask=preg), minreg)
                    vmax = tla.max(vmax, v2, mask=preg)
                max_lane = vmax.reduce(tla.ReductionOp.MAX, mask=preg)
                max_val = tla.max(
                    sink, tla.full(max_lane, dtype=tla.Float32), mask=preg
                )
                e0 = tla.exp(tla.sub(v0, max_val, mask=preg), mask=preg)
                e1 = tla.exp(tla.sub(v1, max_val, mask=preg), mask=preg)
                e0 = tla.where(mask0, e0, tla.full(0.0, dtype=tla.Float32))
                e1 = tla.where(mask1, e1, tla.full(0.0, dtype=tla.Float32))
                total = tla.add(
                    e0.reduce(tla.ReductionOp.ADD, mask=preg),
                    e1.reduce(tla.ReductionOp.ADD, mask=preg),
                    mask=preg,
                )
                e2 = tla.full(0.0, dtype=tla.Float32)
                if with_tail:
                    zero32 = tla.full(0.0, dtype=tla.Float32)
                    e2 = tla.where(
                        mask2,
                        tla.exp(tla.sub(v2, max_val, mask=preg), mask=preg),
                        zero32,
                    )
                    total = tla.add(
                        total, e2.reduce(tla.ReductionOp.ADD, mask=preg), mask=preg
                    )
                denom = tla.full(total, dtype=tla.Float32)
                if has_sinks:
                    denom = tla.add(
                        denom,
                        tla.exp(tla.sub(sink, max_val, mask=preg), mask=preg),
                        mask=preg,
                    )
                q0 = tla.div(e0, denom, mask=preg)
                q1 = tla.div(e1, denom, mask=preg)
                p0, _ = tla.deinterleave(
                    q0.to(dt, cast0, mask=preg), q1.to(dt, cast0, mask=preg)
                )
                tla.tile_view(ub_p, tla.make_shape(1, 128), tla.make_coord(r, 0)).store(
                    p0, params=pack_store
                )
                if with_tail:
                    q2 = tla.div(e2, denom, mask=preg)
                    p2, _ = tla.deinterleave(
                        q2.to(dt, cast0, mask=preg), e2.to(dt, cast0, mask=preg)
                    )
                    half_mask, _ = tla.update_mask(64, dtype=dt)
                    tla.tile_view(
                        ub_p, tla.make_shape(1, 64), tla.make_coord(r, 2)
                    ).store(
                        p2,
                        params=pack_store,
                        mask=half_mask,
                    )
                tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(r)).store(
                    denom, tla.params.UnalignStoreParams()
                )
                tla.tile_view(ub_max, tla.make_shape(1), tla.make_coord(r)).store(
                    max_val, tla.params.UnalignStoreParams()
                )
        if release_scores:
            tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=1)
        sync.m_p.unlock(pipe=tla.pipes.VECTOR)
        sync.m_p.lock(pipe=tla.pipes.MTE3)
        _single_query_insert_nz(
            tla.recast_ptr(pc.ptr + sub * rows * 16, dtype=tla.UInt16),
            tla.recast_ptr(ub_p.ptr, dtype=tla.UInt16),
            rows,
            m,
            p_pitch,
            n,
        )
        sync.m_p.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=1)


@dataclass
class SingleQueryStats:
    ub_sum: object
    ub_max: object
    ub_dm: object


@tla.kernel
def sparse_flash_mla_single_query_kernel(
    q: tla.Tensor,
    ori: tla.Tensor,
    cmp: tla.Tensor,
    bt_o: tla.Tensor,
    bt_c: tla.Tensor,
    cu_o: tla.Tensor,
    cu_c: tla.Tensor,
    sinks: tla.Tensor,
    out: tla.Tensor,
    lse: tla.Tensor,
    scale: tla.Float32,
    n: tla.Constexpr[int],
    page_o: tla.Constexpr[int],
    page_c: tla.Constexpr[int],
    has_sink: tla.Constexpr[bool],
    enable_lse: tla.Constexpr[bool],
) -> None:
    m = 16
    dt = q.ptr.dtype
    rows = m // 2
    cap_cmp = n - KV_BLOCK
    half = max(KV_BLOCK // 2, cap_cmp // 2)
    dq_in = tla.allocate(half * REC_BYTES, tla.UInt8, tla.AddressSpace.ub, 256)
    dq_out = tla.allocate((half + 1) * HEAD_DIM, dt, tla.AddressSpace.ub, 256)
    q_mem = tla.allocate(m * HEAD_DIM, dt, tla.AddressSpace.l1, 512)
    kv_mem = tla.allocate(n * HEAD_DIM, dt, tla.AddressSpace.l1, 512)
    p_mem = tla.allocate(m * n, dt, tla.AddressSpace.l1, 512)
    score_pitch = (n + 63) // 64 * 64
    s_mem = tla.allocate(rows * score_pitch, tla.Float32, tla.AddressSpace.ub, 256)
    sink_mem = tla.allocate(rows, tla.Float32, tla.AddressSpace.ub, 256)
    ub_p = tla.allocate(
        ((rows + 15) // 16) * 16 * score_pitch, dt, tla.AddressSpace.ub, 256
    )
    stats = SingleQueryStats(
        ub_sum=tla.allocate(rows, tla.Float32, tla.AddressSpace.ub, 256),
        ub_max=tla.allocate(rows, tla.Float32, tla.AddressSpace.ub, 256),
        ub_dm=tla.allocate(rows, tla.Float32, tla.AddressSpace.ub, 256),
    )
    sync = declare_sync(need_lse_war=enable_lse, need_fused_dequant=True)
    # One A/B buffer. For 128 keys, each MMAD covers 256 features.
    d_step = 256 if n == KV_BLOCK else 128
    l0a = tla.allocate(m * max(n, d_step), dt, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(n * d_step, dt, tla.AddressSpace.l0b, 512)
    l0s = tla.allocate(m * n, tla.Float32, tla.AddressSpace.l0c, 1024)
    l0o = tla.allocate(m * HEAD_DIM, tla.Float32, tla.AddressSpace.l0c, 1024)
    head_group = tla.arch.block_idx()
    end = cu_o[1]
    start = end - KV_BLOCK if end > KV_BLOCK else 0
    visible = end - start
    compressed = cu_c[1] if cap_cmp else tla.as_numeric(0)
    _single_query_load_segment(
        ori,
        bt_o,
        page_o,
        start,
        visible,
        KV_BLOCK,
        n,
        0,
        kv_mem,
        dq_in,
        dq_out,
        sync,
    )
    if cap_cmp:
        _single_query_load_segment(
            cmp,
            bt_c,
            page_c,
            0,
            compressed,
            cap_cmp,
            n,
            KV_BLOCK,
            kv_mem,
            dq_in,
            dq_out,
            sync,
        )
    with tla.vector():
        tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)
    with tla.cube():
        q_l1 = tla.make_tensor(
            q_mem, tla.make_layout_with_tag((m, HEAD_DIM), dt, tla.arch.zN)
        )
        q_gm = tla.make_tensor(
            q.ptr + head_group * m * HEAD_DIM,
            tla.make_layout_with_tag((m, HEAD_DIM), dt, tla.arch.RowMajor),
        )
        k_l1 = tla.make_tensor(
            kv_mem, tla.make_layout_with_tag((HEAD_DIM, n), dt, tla.arch.nZ)
        )
        s_l0 = tla.make_tensor(
            l0s, tla.make_layout_with_tag((m, n), tla.Float32, tla.arch.L0Clayout)
        )
        s_ub = tla.make_tensor(
            s_mem,
            tla.make_layout(
                tla.make_shape(m, n),
                tla.make_stride(score_pitch, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        tla.set_flag(sync.ab_free0)
        tla.copy(q_l1, q_gm)
        tla.set_flag(sync.q_l1_ready)
        tla.wait_flag(sync.q_l1_ready)
        tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE1, aiv_id=0)
        tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE1, aiv_id=1)
        for d in static_range(HEAD_DIM // d_step):
            a = tla.tile_view(q_l1, tla.make_shape(m, d_step), tla.make_coord(0, d))
            b = tla.tile_view(k_l1, tla.make_shape(d_step, n), tla.make_coord(d, 0))
            qa = tla.make_tensor_like(l0a, a)
            kb = tla.make_tensor_like(l0b, b)
            tla.wait_flag(sync.ab_free0)
            tla.copy(qa, a)
            tla.copy(kb, b)
            tla.set_flag(sync.ab_ready0)
            tla.wait_flag(sync.ab_ready0)
            tla.mmad(s_l0, qa, kb, init_c=(d == 0))
            tla.set_flag(sync.ab_free0)
        tla.set_flag(sync.s_mm_ready)
        tla.wait_flag(sync.s_mm_ready)
        tla.copy(
            s_ub,
            s_l0,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        tla.cross_core_set_flag(sync.s_ready, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_set_flag(sync.s_ready, tla.pipes.FIX, aiv_id=1)

    _single_query_softmax(
        p_mem,
        s_mem,
        ub_p,
        stats.ub_sum,
        stats.ub_max,
        sinks,
        sink_mem,
        head_group,
        m,
        n,
        visible,
        compressed,
        scale,
        has_sink,
        sync,
    )
    if enable_lse:
        lse_copy_out(sync, stats, lse, head_group, m, rows)
    wait_p_ready(sync)
    with tla.cube():
        p_l1 = tla.make_tensor(p_mem, tla.make_layout_with_tag((m, n), dt, tla.arch.zN))
        pa = tla.make_tensor_like(l0a, p_l1)
        v_l1 = tla.make_tensor(
            kv_mem, tla.make_layout_with_tag((n, HEAD_DIM), dt, tla.arch.zN)
        )
        for d in static_range(HEAD_DIM // d_step):
            v = tla.tile_view(v_l1, tla.make_shape(n, d_step), tla.make_coord(0, d))
            vb = tla.make_tensor_like(l0b, v)
            ob = tla.make_tensor(
                l0o + d * m * d_step,
                tla.make_layout_with_tag((m, d_step), tla.Float32, tla.arch.L0Clayout),
            )
            tla.wait_flag(sync.ab_free0)
            if d == 0:
                tla.copy(pa, p_l1)
            tla.copy(vb, v)
            tla.set_flag(sync.ab_ready0)
            tla.wait_flag(sync.ab_ready0)
            tla.mmad(ob, pa, vb, init_c=True)
            tla.set_flag(sync.ab_free0)
        o_l0 = tla.make_tensor(
            l0o,
            tla.make_layout_with_tag((m, HEAD_DIM), tla.Float32, tla.arch.L0Clayout),
        )
        o_gm = tla.make_tensor(
            out.ptr + head_group * m * HEAD_DIM,
            tla.make_layout_with_tag((m, HEAD_DIM), dt, tla.arch.RowMajor),
        )
        tla.set_flag(sync.o_mm_ready0)
        tla.wait_flag(sync.o_mm_ready0)
        tla.copy(o_gm, o_l0)


@tla.jit
def _single_query_load_csa(
    ori,
    cmp,
    ori_table,
    cmp_table,
    indices,
    page_o,
    page_c,
    start,
    visible,
    compressed,
    cap_cmp,
    n,
    l1,
    raw,
    nz,
    ub_index,
    raw_index,
    positions,
    log_counts,
    compact_count,
    sync,
):
    """Gather run representatives and place their BF16 K/V in the combined tile."""
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        ori_half = KV_BLOCK // 2
        cmp_half = cap_cmp // 2
        ori_offset = sub * ori_half
        cmp_offset = sub * cmp_half
        if compressed > 0:
            full_index = tla.make_layout(tla.make_shape(cap_cmp), tla.make_stride(1))
            tla.copy(
                tla.make_tensor(raw_index, full_index),
                tla.make_tensor(indices.ptr, full_index),
            )
        ori_count = visible - ori_offset
        ori_count = ori_count if ori_count < ori_half else ori_half
        ori_count = ori_count if ori_count > 0 else 0
        raw_cmp = raw + ori_half * REC_BYTES
        ori_rows = ori_half + 1
        cmp_rows = cmp_half + 1
        nz_o = tla.recast_ptr(nz, dtype=tla.UInt16)
        nz_c = nz_o + ori_rows * HEAD_DIM
        sync.m_dq_rec0.lock(pipe=tla.pipes.MTE2)
        _single_query_load_records(
            raw, ori.ptr, ori_table.ptr, start + ori_offset, ori_count, page_o
        )
        groups = tla.as_numeric(0)
        if compressed > 0:
            tla.set_flag(sync.phy_m2v)
            tla.wait_flag(sync.phy_m2v)
            _fold_one_cmp_row(
                raw_index, positions, ub_index, log_counts, compact_count, compressed
            )
            tla.set_flag(sync.phy_vs)
            tla.wait_flag(sync.phy_vs)
            count = tla.make_tensor(
                compact_count, tla.make_layout(tla.make_shape(1), tla.make_stride(1))
            )[0]
            groups = count if count >= 0 else -count
        cmp_count = groups - cmp_offset
        cmp_count = cmp_count if cmp_count < cmp_half else cmp_half
        cmp_count = cmp_count if cmp_count > 0 else 0
        if cmp_count > 0:
            _gather_fp8_records(
                raw_cmp,
                cmp.ptr,
                ub_index + cmp_offset,
                cmp_table.ptr,
                cmp_count,
                page_c,
            )
        sync.m_dq_rec0.unlock(pipe=tla.pipes.MTE2)
        sync.m_dq_rec0.lock(pipe=tla.pipes.VECTOR)
        sync.m_dq_out0.lock(pipe=tla.pipes.VECTOR)
        _single_query_dequant_nz(nz_o, raw, ori_count, ori_rows)
        _single_query_dequant_nz(nz_c, raw_cmp, cmp_count, cmp_rows)
        sync.m_dq_rec0.unlock(pipe=tla.pipes.VECTOR)
        sync.m_dq_out0.unlock(pipe=tla.pipes.VECTOR)
        sync.m_dq_out0.lock(pipe=tla.pipes.MTE3)
        _single_query_insert_nz(
            tla.recast_ptr(l1 + ori_offset * 16, dtype=tla.UInt16),
            nz_o,
            ori_half,
            n,
            ori_rows,
            HEAD_DIM,
        )
        _single_query_insert_nz(
            tla.recast_ptr(l1 + (KV_BLOCK + cmp_offset) * 16, dtype=tla.UInt16),
            nz_c,
            cmp_half,
            n,
            cmp_rows,
            HEAD_DIM,
        )
        sync.m_dq_out0.unlock(pipe=tla.pipes.MTE3)


_CSA_SHORT_NORM = r"""
#include "kernel_operator.h"
using namespace AscendC::MicroAPI;

static constexpr CastTrait prob_bf16 = {RegLayout::ZERO, SatMode::SAT,
                                       MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

__simd_vf__ void NormalizeShortRows(__ubuf__ bfloat16_t* probs, __ubuf__ float* scores,
                                  __ubuf__ float* sinks, __ubuf__ float* sums, __ubuf__ float* maxima,
                                  __ubuf__ float* log_counts, int32_t n_ori, int32_t n_cmp,
                                  float scale, int32_t has_sink) {
    auto all = CreateMask<float, MaskPattern::ALL>();
    auto bf16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    auto bf16_half = CreateMask<bfloat16_t, MaskPattern::H>();
    uint32_t size0 = n_ori < 64 ? n_ori : 64;
    uint32_t size1 = n_ori > 64 ? n_ori - 64 : 0;
    uint32_t size2 = n_cmp;
    uint32_t single = 1;
    auto valid0 = UpdateMask<float>(size0);
    auto valid1 = UpdateMask<float>(size1);
    auto valid2 = UpdateMask<float>(size2);
    auto one = UpdateMask<float>(single);
    RegTensor<float> neg, unity, logs, scale_reg;
    RegTensor<bfloat16_t> zero;
    if (n_cmp > 0) {
        LoadAlign(logs, log_counts);
    } else {
        Duplicate(logs, 0.0f);
    }
    Duplicate(scale_reg, scale);
    Duplicate(neg, -3.4028234663852886e38f);
    Duplicate(unity, 1.0f);
    Duplicate(zero, 0);
    UnalignRegForStore max_store, sum_store;
    for (uint16_t row = 0; row < 8; ++row) {
        auto* data = scores + row * 192;
        RegTensor<float> a, b, c, peak, sink, joined, denom, element, inv;
        LoadAlign(a, data);
        LoadAlign(b, data + 64);
        LoadAlign(c, data + 128);
        Muls(a, a, scale, all);
        Muls(b, b, scale, all);
        RegTensor<float> weighted = logs;
        MulAddDst(weighted, c, scale_reg, all);
        c = weighted;
        Select(a, a, neg, valid0);
        Select(b, b, neg, valid1);
        Select(c, c, neg, valid2);
        Max(peak, a, b, all);
        Max(peak, peak, c, all);
        ReduceMax(peak, peak, all);
        Duplicate(peak, peak, all);
        if (has_sink) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(sink, sinks + row);
            Max(peak, peak, sink, all);
        }
        ExpSub(a, a, peak, valid0);
        ExpSub(b, b, peak, valid1);
        ExpSub(c, c, peak, valid2);
        Add(joined, a, b, all);
        Add(joined, joined, c, all);
        ReduceSum(denom, joined, all);
        if (has_sink) {
            ExpSub(element, sink, peak, one);
            Add(denom, denom, element, one);
        }
        Div(inv, unity, denom, one);
        Duplicate(inv, inv, all);
        Mul(a, a, inv, all);
        Mul(b, b, inv, all);
        Mul(c, c, inv, all);
        RegTensor<bfloat16_t> p0, p1, p2, packed, last, unused;
        Cast<bfloat16_t, float, prob_bf16>(p0, a, all);
        Cast<bfloat16_t, float, prob_bf16>(p1, b, all);
        Cast<bfloat16_t, float, prob_bf16>(p2, c, all);
        DeInterleave(packed, unused, p0, p1);
        DeInterleave(last, unused, p2, zero);
        auto* row_start = probs + row * 16;
        auto* tail = row_start + 128 * 16;
        StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            row_start, packed, 16, 1, bf16);
        StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            tail, last, 16, 1, bf16_half);
        StoreUnAlign(maxima, peak, max_store, 1);
        StoreUnAlign(sums, denom, sum_store, 1);
    }
    StoreUnAlignPost(maxima, max_store, 0);
    StoreUnAlignPost(sums, sum_store, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

extern "C" [aicore] __attribute__((always_inline)) void csa_short_norm(
    uint64_t p, uint64_t s, uint64_t h, uint64_t z, uint64_t m, uint64_t logs,
    int32_t n_ori, int32_t n_cmp, float scale, int32_t has_sink) {
    NormalizeShortRows(reinterpret_cast<__ubuf__ bfloat16_t*>(p),
                       reinterpret_cast<__ubuf__ float*>(s),
                       reinterpret_cast<__ubuf__ float*>(h),
                       reinterpret_cast<__ubuf__ float*>(z),
                       reinterpret_cast<__ubuf__ float*>(m), reinterpret_cast<__ubuf__ float*>(logs),
                       n_ori, n_cmp, scale, has_sink);
}
"""


@tla.extern(source=_CSA_SHORT_NORM, name="csa_short_norm")
def _csa_short_norm(
    p: tla.Pointer[tla.BFloat16, tla.AddressSpace.ub],
    s: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    h: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    z: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    m: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    logs: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    n_ori: tla.Int32,
    n_cmp: tla.Int32,
    scale: tla.Float32,
    has_sink: tla.Int32,
) -> None: ...


@tla.jit
def _normalize_csa_short(
    l1p,
    scores,
    prob_ub,
    sums,
    maxima,
    sink_ub,
    m,
    n,
    n_ori,
    n_cmp,
    compact_count,
    logs,
    scale,
    has_sinks,
    sync,
):
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        rows = m // 2
        dt = l1p.dtype
        p_pitch = ((rows + 15) // 16) * 16
        pitch = (n + 63) // 64 * 64
        ub_p = tla.make_tensor(
            prob_ub, tla.make_layout_with_tag((rows, pitch), dt, tla.arch.zN)
        )
        pc = tla.make_tensor(l1p, tla.make_layout_with_tag((m, n), dt, tla.arch.zN))
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=1)
        groups = tla.as_numeric(0)
        if n_cmp > 0:
            count = tla.make_tensor(
                compact_count, tla.make_layout(tla.make_shape(1), tla.make_stride(1))
            )[0]
            groups = count if count >= 0 else -count
        sync.m_p.lock(pipe=tla.pipes.VECTOR)
        _csa_short_norm(
            ub_p.ptr,
            scores,
            sink_ub,
            sums,
            maxima,
            logs,
            n_ori,
            groups,
            scale,
            tla.as_numeric(int(has_sinks)),
        )
        sync.m_p.unlock(pipe=tla.pipes.VECTOR)
        sync.m_p.lock(pipe=tla.pipes.MTE3)
        _single_query_insert_nz(
            tla.recast_ptr(pc.ptr + sub * rows * 16, dtype=tla.UInt16),
            tla.recast_ptr(ub_p.ptr, dtype=tla.UInt16),
            rows,
            m,
            p_pitch,
            n,
        )
        sync.m_p.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=1)


@tla.kernel
def sparse_flash_mla_csa_short_kernel(
    q: tla.Tensor,
    ori: tla.Tensor,
    cmp: tla.Tensor,
    bt_o: tla.Tensor,
    bt_c: tla.Tensor,
    cu_o: tla.Tensor,
    cu_c: tla.Tensor,
    sinks: tla.Tensor,
    out: tla.Tensor,
    lse: tla.Tensor,
    sparse_idx: tla.Tensor,
    topk: tla.Tensor,
    scale: tla.Float32,
    n: tla.Constexpr[int],
    page_o: tla.Constexpr[int],
    page_c: tla.Constexpr[int],
    has_sink: tla.Constexpr[bool],
    enable_lse: tla.Constexpr[bool],
) -> None:
    m = 16
    dt = q.ptr.dtype
    rows = m // 2
    cap_cmp = n - KV_BLOCK
    half = n // 2
    local_idx = tla.allocate(cap_cmp, tla.Int32, tla.AddressSpace.ub, 256)
    # Run detection reads a full vector and the element before the first index.
    raw_index = tla.allocate(cap_cmp + 128, tla.Int32, tla.AddressSpace.ub, 256) + 64
    positions = tla.allocate(cap_cmp + 64, tla.Int32, tla.AddressSpace.ub, 256)
    log_counts = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    compact_count = tla.allocate(8, tla.Int32, tla.AddressSpace.ub, 256)
    dq_in = tla.allocate(half * REC_BYTES, tla.UInt8, tla.AddressSpace.ub, 256)
    dq_out = tla.allocate((half + 2) * HEAD_DIM, dt, tla.AddressSpace.ub, 256)
    q_mem = tla.allocate(m * HEAD_DIM, dt, tla.AddressSpace.l1, 512)
    kv_mem = tla.allocate(n * HEAD_DIM, dt, tla.AddressSpace.l1, 512)
    p_mem = tla.allocate(m * n, dt, tla.AddressSpace.l1, 512)
    score_pitch = (n + 63) // 64 * 64
    s_mem = tla.allocate(rows * score_pitch, tla.Float32, tla.AddressSpace.ub, 256)
    sink_mem = tla.allocate(rows, tla.Float32, tla.AddressSpace.ub, 256)
    ub_p = tla.allocate(
        ((rows + 15) // 16) * 16 * score_pitch, dt, tla.AddressSpace.ub, 256
    )
    stats = SingleQueryStats(
        ub_sum=tla.allocate(rows, tla.Float32, tla.AddressSpace.ub, 256),
        ub_max=tla.allocate(rows, tla.Float32, tla.AddressSpace.ub, 256),
        ub_dm=tla.allocate(rows, tla.Float32, tla.AddressSpace.ub, 256),
    )
    sync = declare_sync(need_lse_war=enable_lse, need_fused_dequant=True)
    # Two operand banks of 128 and 64 columns fit in 64 KiB of L0B.
    l0a = tla.allocate(m * n, dt, tla.AddressSpace.l0a, 512)
    l0a1 = tla.allocate(m * 64, dt, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(n * (128 + 64), dt, tla.AddressSpace.l0b, 512)
    l0s = tla.allocate(m * n, tla.Float32, tla.AddressSpace.l0c, 1024)
    l0o = tla.allocate(m * HEAD_DIM, tla.Float32, tla.AddressSpace.l0c, 1024)
    head_group = tla.arch.block_idx()
    end = cu_o[1]
    start = end - KV_BLOCK if end > KV_BLOCK else 0
    visible = end - start
    compressed = cu_c[1]
    requested = topk[0]
    if requested < compressed:
        compressed = requested if requested > 0 else tla.as_numeric(0)
    if has_sink:
        with tla.vector():
            sub = tla.arch.sub_block_idx()
            sink_view = tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
            tla.copy(
                tla.make_tensor(sink_mem, sink_view),
                tla.make_tensor(sinks.ptr + head_group * m + sub * rows, sink_view),
            )
    _single_query_load_csa(
        ori,
        cmp,
        bt_o,
        bt_c,
        sparse_idx,
        page_o,
        page_c,
        start,
        visible,
        compressed,
        cap_cmp,
        n,
        kv_mem,
        dq_in,
        dq_out,
        local_idx,
        raw_index,
        positions,
        log_counts,
        compact_count,
        sync,
    )
    with tla.vector():
        tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)
    with tla.cube():
        q_l1 = tla.make_tensor(
            q_mem, tla.make_layout_with_tag((m, HEAD_DIM), dt, tla.arch.zN)
        )
        q_gm = tla.make_tensor(
            q.ptr + head_group * m * HEAD_DIM,
            tla.make_layout_with_tag((m, HEAD_DIM), dt, tla.arch.RowMajor),
        )
        k_l1 = tla.make_tensor(
            kv_mem, tla.make_layout_with_tag((HEAD_DIM, n), dt, tla.arch.nZ)
        )
        s_l0 = tla.make_tensor(
            l0s, tla.make_layout_with_tag((m, n), tla.Float32, tla.arch.L0Clayout)
        )
        s_ub = tla.make_tensor(
            s_mem,
            tla.make_layout(
                tla.make_shape(m, n),
                tla.make_stride(score_pitch, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        tla.set_flag(sync.ab_free0)
        tla.set_flag(sync.ab_free1)
        tla.copy(q_l1, q_gm)
        tla.set_flag(sync.q_l1_ready)
        tla.wait_flag(sync.q_l1_ready)
        tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE1, aiv_id=0)
        tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE1, aiv_id=1)
        for offset, depth, alternate in (
            (0, 128, False),
            (128, 64, True),
            (192, 128, False),
            (320, 64, True),
            (384, 128, False),
        ):
            a = tla.get_tile(q_l1, tla.make_coord(0, offset), tla.make_shape(m, depth))
            b = tla.get_tile(k_l1, tla.make_coord(offset, 0), tla.make_shape(depth, n))
            free = sync.ab_free1 if alternate else sync.ab_free0
            ready = sync.ab_ready1 if alternate else sync.ab_ready0
            qa = tla.make_tensor_like(l0a1 if alternate else l0a, a)
            kb = tla.make_tensor_like(l0b + n * 128 if alternate else l0b, b)
            tla.wait_flag(free)
            tla.copy(qa, a)
            tla.copy(kb, b)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            tla.mmad(s_l0, qa, kb, init_c=offset == 0)
            tla.set_flag(free)
        tla.set_flag(sync.s_mm_ready)
        tla.wait_flag(sync.s_mm_ready)
        tla.copy(
            s_ub,
            s_l0,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        tla.cross_core_set_flag(sync.s_ready, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_set_flag(sync.s_ready, tla.pipes.FIX, aiv_id=1)

    _normalize_csa_short(
        p_mem,
        s_mem,
        ub_p,
        stats.ub_sum,
        stats.ub_max,
        sink_mem,
        m,
        n,
        visible,
        compressed,
        compact_count,
        log_counts,
        scale,
        has_sink,
        sync,
    )
    if enable_lse:
        lse_copy_out(sync, stats, lse, head_group, m, rows)
    with tla.cube():
        p_l1 = tla.make_tensor(p_mem, tla.make_layout_with_tag((m, n), dt, tla.arch.zN))
        pa = tla.make_tensor_like(l0a, p_l1)
        v_l1 = tla.make_tensor(
            kv_mem, tla.make_layout_with_tag((n, HEAD_DIM), dt, tla.arch.zN)
        )
        # Start with the second V bank so its first load overlaps the final QK block.
        for offset, depth, alternate in (
            (128, 64, True),
            (0, 128, False),
            (320, 64, True),
            (192, 128, False),
            (384, 128, False),
        ):
            v = tla.get_tile(v_l1, tla.make_coord(0, offset), tla.make_shape(n, depth))
            vb = tla.make_tensor_like(l0b + n * 128 if alternate else l0b, v)
            ob = tla.make_tensor(
                l0o + offset * m,
                tla.make_layout_with_tag((m, depth), tla.Float32, tla.arch.L0Clayout),
            )
            free = sync.ab_free1 if alternate else sync.ab_free0
            ready = sync.ab_ready1 if alternate else sync.ab_ready0
            tla.wait_flag(free)
            tla.copy(vb, v)
            if offset == 128:
                tla.cross_core_wait_flag(sync.p_ready, tla.pipes.MTE1, aiv_id=0)
                tla.cross_core_wait_flag(sync.p_ready, tla.pipes.MTE1, aiv_id=1)
                tla.copy(pa, p_l1)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            tla.mmad(ob, pa, vb, init_c=True)
            tla.set_flag(free)
        o_l0 = tla.make_tensor(
            l0o,
            tla.make_layout_with_tag((m, HEAD_DIM), tla.Float32, tla.arch.L0Clayout),
        )
        o_gm = tla.make_tensor(
            out.ptr + head_group * m * HEAD_DIM,
            tla.make_layout_with_tag((m, HEAD_DIM), dt, tla.arch.RowMajor),
        )
        tla.set_flag(sync.o_mm_ready0)
        tla.wait_flag(sync.o_mm_ready0)
        tla.copy(o_gm, o_l0)


@dataclass
class RollingHcaBuffers:
    l1_q: object
    l1_k: object
    l1_p: object
    l0a: object
    l0b: object
    l0c_s: object
    l0c_o: object
    ub_s: object
    ub_p: object
    ub_sink: object
    ub_sum: object
    ub_max: object
    ub_dm: object


@tla.jit
def _load_hca_shared_cmp(kv, table, ptr, valid, page, stride, width, sync):
    """Keep the compressed prefix resident across all queries assigned to a core."""
    dt = kv.ptr.dtype
    with tla.cube():
        dst = tla.make_tensor(
            ptr, tla.make_layout_with_tag((HEAD_DIM, width), dt, tla.arch.nZ)
        )
        pos = tla.as_numeric(0)
        for offset in tla.range(0, valid, page):
            count = valid - offset
            count = count if count < page else page
            phys = table[offset // page]
            phys = phys if phys >= 0 else 0
            src_view = tla.make_layout(
                tla.make_shape(HEAD_DIM, count),
                tla.make_stride(1, HEAD_DIM),
                layoutTag=tla.arch.ColumnMajor,
            )
            src = tla.make_tensor(kv.ptr + phys * stride, src_view)
            tla.copy(
                tla.get_tile(
                    dst, tla.make_coord(0, offset), tla.make_shape(HEAD_DIM, count)
                ),
                src,
            )
            pos = offset + count
        # Use a finite, valid row for padding so masked PV columns multiply by zero.
        first = table[0]
        first = first if first >= 0 else 0
        row_view = tla.make_layout(
            tla.make_shape(HEAD_DIM, 1),
            tla.make_stride(1, HEAD_DIM),
            layoutTag=tla.arch.ColumnMajor,
        )
        row = tla.make_tensor(kv.ptr + first * stride, row_view)
        for padding in tla.range(pos, width):
            tla.copy(
                tla.get_tile(
                    dst, tla.make_coord(0, padding), tla.make_shape(HEAD_DIM, 1)
                ),
                row,
            )
        tla.set_flag(sync.k_l1_ready2)
        tla.wait_flag(sync.k_l1_ready2)


@tla.jit
def _rolling_hca_qk(buf, slot, cmp_width, sync):
    m = Q_ROWS
    rows = m // 2
    width = KV_BLOCK + cmp_width
    pitch = (width + VL - 1) // VL * VL
    dt = buf.l1_q.dtype
    with tla.cube():
        q_l1 = tla.make_tensor(
            _q_slot_ptr(buf, 2 * slot),
            tla.make_layout_with_tag((m, HEAD_DIM), dt, tla.arch.zN),
        )
        acc = buf.l0c_s + slot * m * width
        s_l0 = tla.make_tensor(
            acc, tla.make_layout_with_tag((m, width), tla.Float32, tla.arch.L0Clayout)
        )
        s_ub = tla.make_tensor(
            buf.ub_s + slot * rows * pitch,
            tla.make_layout(
                tla.make_shape(m, width),
                tla.make_stride(pitch, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        if slot == 0:
            tla.wait_flag(sync.q_l1_ready)
            tla.wait_flag(sync.s_fix_done)
        else:
            tla.wait_flag(sync.q_second_ready)
            tla.wait_flag(sync.s_fix_done1)
        _k_slot_ready_wait(sync, slot)
        for part in (0, 1):
            n = KV_BLOCK if part == 0 else cmp_width
            mem = _k_slot_ptr(buf, slot) if part == 0 else _k_slot_ptr(buf, 2)
            k_l1 = tla.make_tensor(
                mem, tla.make_layout_with_tag((HEAD_DIM, n), dt, tla.arch.nZ)
            )
            frag = tla.make_tensor(
                acc + part * m * KV_BLOCK,
                tla.make_layout_with_tag((m, n), tla.Float32, tla.arch.L0Clayout),
            )
            for d in (0, 1, 2, 3):
                ab = d % 2
                free = sync.ab_free0 if ab == 0 else sync.ab_free1
                ready = sync.ab_ready0 if ab == 0 else sync.ab_ready1
                a = tla.tile_view(
                    q_l1, tla.make_shape(m, D_CHUNK), tla.make_coord(0, d)
                )
                b = tla.tile_view(
                    k_l1, tla.make_shape(D_CHUNK, n), tla.make_coord(d, 0)
                )
                qa = tla.make_tensor_like(buf.l0a + ab * m * D_CHUNK, a)
                kb = tla.make_tensor_like(buf.l0b + ab * D_CHUNK * KV_BLOCK, b)
                tla.wait_flag(free)
                tla.copy(qa, a)
                tla.copy(kb, b)
                tla.set_flag(ready)
                tla.wait_flag(ready)
                tla.mmad(frag, qa, kb, init_c=(d == 0))
                tla.set_flag(free)
        if slot == 0:
            tla.set_flag(sync.q_l0a_free)
            tla.set_flag(sync.s_mm_ready)
        else:
            tla.set_flag(sync.q_second_free)
            tla.set_flag(sync.s_mm_ready1)
        tla.cross_core_wait_flag(sync.s_free, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_wait_flag(sync.s_free, tla.pipes.FIX, aiv_id=1)
        if slot == 0:
            tla.wait_flag(sync.s_mm_ready)
        else:
            tla.wait_flag(sync.s_mm_ready1)
        tla.copy(
            s_ub,
            s_l0,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        if slot == 0:
            tla.set_flag(sync.s_fix_done)
        else:
            tla.set_flag(sync.s_fix_done1)
        tla.cross_core_set_flag(sync.s_ready, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_set_flag(sync.s_ready, tla.pipes.FIX, aiv_id=1)


@tla.jit
def _rolling_hca_pv(buf, out, token, slot, cmp_width, sync):
    m = Q_ROWS
    n = KV_BLOCK + cmp_width
    dt = buf.l1_q.dtype
    wait_p_ready(sync)
    with tla.cube():
        pc = tla.make_tensor(
            buf.l1_p + slot * m * n, tla.make_layout_with_tag((m, n), dt, tla.arch.zN)
        )
        tla.wait_flag(sync.o_fix_done0)
        for part in (0, 1):
            width = KV_BLOCK if part == 0 else cmp_width
            mem = _k_slot_ptr(buf, slot) if part == 0 else _k_slot_ptr(buf, 2)
            vv = tla.make_tensor(
                mem, tla.make_layout_with_tag((width, HEAD_DIM), dt, tla.arch.zN)
            )
            pa = tla.get_tile(
                pc, tla.make_coord(0, part * KV_BLOCK), tla.make_shape(m, width)
            )
            for d in (0, 1, 2, 3):
                ab = d % 2
                free = sync.ab_free0 if ab == 0 else sync.ab_free1
                ready = sync.ab_ready0 if ab == 0 else sync.ab_ready1
                vp = tla.tile_view(
                    vv, tla.make_shape(width, D_CHUNK), tla.make_coord(0, d)
                )
                p0 = tla.make_tensor_like(buf.l0a + ab * m * D_CHUNK, pa)
                v0 = tla.make_tensor_like(buf.l0b + ab * D_CHUNK * KV_BLOCK, vp)
                oo = tla.make_tensor(
                    buf.l0c_o + d * m * D_CHUNK,
                    tla.make_layout_with_tag(
                        (m, D_CHUNK), tla.Float32, tla.arch.L0Clayout
                    ),
                )
                tla.wait_flag(free)
                if d < 2:
                    tla.copy(p0, pa)
                tla.copy(v0, vp)
                tla.set_flag(ready)
                tla.wait_flag(ready)
                tla.mmad(oo, p0, v0, init_c=(part == 0))
                tla.set_flag(free)
            if part == 0:
                _k_slot_free(sync, slot)
        acc = tla.make_tensor(
            buf.l0c_o,
            tla.make_layout_with_tag((m, HEAD_DIM), tla.Float32, tla.arch.L0Clayout),
        )
        dst = tla.make_tensor(
            out.ptr + token * m * HEAD_DIM,
            tla.make_layout_with_tag((m, HEAD_DIM), dt, tla.arch.RowMajor),
        )
        tla.set_flag(sync.o_mm_ready0)
        tla.wait_flag(sync.o_mm_ready0)
        tla.copy(dst, acc, cache_policy=tla.params.CachePolicy.DISABLE)
        tla.set_flag(sync.o_fix_done0)


@tla.kernel
def sparse_flash_mla_rolling_hca_kernel(
    q: tla.Tensor,
    ori_kv: tla.Tensor,
    cmp_kv: tla.Tensor,
    ori_bt: tla.Tensor,
    cmp_bt: tla.Tensor,
    cu_q: tla.Tensor,
    cu_kv: tla.Tensor,
    cu_cmp: tla.Tensor,
    sinks: tla.Tensor,
    out: tla.Tensor,
    lse: tla.Tensor,
    softmax_scale: tla.Float32,
    ori_pa_block_size: tla.Constexpr[int],
    ori_pa_stride: tla.Constexpr[int],
    cmp_pa_block_size: tla.Constexpr[int],
    cmp_pa_stride: tla.Constexpr[int],
    cmp_blk: tla.Constexpr[int],
    has_sinks: tla.Constexpr[bool],
    enable_lse: tla.Constexpr[bool],
) -> None:
    """Pipeline adjacent queries with a sliding original KV and a resident compressed prefix."""
    m = Q_ROWS
    rows = m // 2
    width = KV_BLOCK + cmp_blk
    pitch = (width + VL - 1) // VL * VL
    dt = q.ptr.dtype
    buf = RollingHcaBuffers(
        l1_q=tla.allocate(2 * m * HEAD_DIM, dt, tla.AddressSpace.l1, 512),
        l1_k=tla.allocate(
            (2 * KV_BLOCK + cmp_blk) * HEAD_DIM, dt, tla.AddressSpace.l1, 512
        ),
        l1_p=tla.allocate(2 * m * width, dt, tla.AddressSpace.l1, 512),
        l0a=tla.allocate(2 * m * D_CHUNK, dt, tla.AddressSpace.l0a, 512),
        l0b=tla.allocate(2 * KV_BLOCK * D_CHUNK, dt, tla.AddressSpace.l0b, 512),
        l0c_s=tla.allocate(2 * m * width, tla.Float32, tla.AddressSpace.l0c, 1024),
        l0c_o=tla.allocate(m * HEAD_DIM, tla.Float32, tla.AddressSpace.l0c, 1024),
        ub_s=tla.allocate(2 * rows * pitch, tla.Float32, tla.AddressSpace.ub, 256),
        ub_p=tla.allocate(rows * pitch, dt, tla.AddressSpace.ub, 256),
        ub_sink=tla.allocate(rows, tla.Float32, tla.AddressSpace.ub, 256),
        ub_sum=tla.allocate(rows, tla.Float32, tla.AddressSpace.ub, 256),
        ub_max=tla.allocate(rows, tla.Float32, tla.AddressSpace.ub, 256),
        ub_dm=tla.allocate(rows, tla.Float32, tla.AddressSpace.ub, 256),
    )
    sync = declare_sync(need_lse_war=enable_lse, two_q_slots=True)
    init_sync(sync, pre_l0c2=True)
    init_sync_preload(sync)
    with tla.cube():
        tla.set_flag(sync.q_second_free)
    with tla.vector():
        if has_sinks:
            shape = tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
            src = tla.make_tensor(sinks.ptr + tla.arch.sub_block_idx() * rows, shape)
            tla.copy(tla.make_tensor(buf.ub_sink, shape), src)
            tla.set_flag(sync.phy_m2v)
            tla.wait_flag(sync.phy_m2v)
    q_len = cu_q[1]
    ori_len = cu_kv[1]
    cmp_len = cu_cmp[1]
    quota = (q_len + tla.arch.block_num() - 1) // tla.arch.block_num()
    first = tla.arch.block_idx() * quota
    count = q_len - first
    count = count if count < quota else quota
    if count > 0:
        _load_hca_shared_cmp(
            cmp_kv,
            cmp_bt,
            _k_slot_ptr(buf, 2),
            cmp_len,
            cmp_pa_block_size,
            cmp_pa_stride,
            cmp_blk,
            sync,
        )
        for step in tla.range(count + 2):
            if step < count:
                token = first + step
                slot = step % 2
                window_end = ori_len - q_len + token + 1
                visible = window_end if window_end < KV_BLOCK else KV_BLOCK
                load_rolling_q(sync, buf, q, token, m, slot)
                load_rolling_window(
                    sync,
                    buf,
                    ori_kv,
                    ori_bt,
                    window_end,
                    visible,
                    slot,
                    step >= 2,
                    ori_pa_block_size,
                    ori_pa_stride,
                )
            if step > 0 and step <= count:
                query = first + step - 1
                compute_slot = (step - 1) % 2
                compute_end = ori_len - q_len + query + 1
                compute_visible = compute_end if compute_end < KV_BLOCK else KV_BLOCK
                _rolling_hca_qk(buf, compute_slot, cmp_blk, sync)
                _single_query_softmax(
                    buf.l1_p + compute_slot * m * width,
                    buf.ub_s + compute_slot * rows * pitch,
                    buf.ub_p,
                    buf.ub_sum,
                    buf.ub_max,
                    sinks,
                    buf.ub_sink,
                    0,
                    m,
                    width,
                    compute_visible,
                    cmp_len,
                    softmax_scale,
                    has_sinks,
                    sync,
                    True,
                    True,
                )
                if enable_lse:
                    lse_copy_out(sync, buf, lse, query, m, rows)
            if step >= 2:
                _rolling_hca_pv(
                    buf, out, first + step - 2, (step - 2) % 2, cmp_blk, sync
                )


@tla.jit
def _load_group_query(q_l1, q_ptr, first_query, group, q_count, dt):
    one_query = tla.make_layout_with_tag((16, HEAD_DIM), dt, tla.arch.RowMajor)
    for j in static_range(q_count):
        offset = ((first_query + j) * 64 + group * 16) * HEAD_DIM
        gm_heads = tla.make_tensor(q_ptr + offset, one_query)
        l1_heads = tla.get_tile(
            q_l1, tla.make_coord(j * 16, 0), tla.make_shape(16, HEAD_DIM)
        )
        tla.copy(l1_heads, gm_heads)


@tla.jit
def _load_grouped_hca(
    kv,
    cmp,
    table,
    cmp_bt,
    start,
    n_ori,
    n_cmp,
    seq,
    page_size,
    table_width,
    cmp_page_size,
    cmp_width,
    n,
    l1,
    record,
    packed,
    sync,
):
    """Prepare the union of original and compressed KV in 128-column slabs."""
    with tla.vector():
        half = n // 2
        row = tla.arch.sub_block_idx() * half
        n_first = n_ori - row
        n_first = n_first if n_first < half else half
        n_last = 0
        if row + n_first == n_ori:
            n_last = n_cmp if n_cmp < half - n_first else half - n_first
        sync.m_dq_rec0.lock(pipe=tla.pipes.MTE2)
        _single_query_load_records(
            record,
            kv.ptr,
            table.ptr + seq * table_width,
            start + row,
            n_first,
            page_size,
        )
        if n_last > 0:
            _single_query_load_records(
                record + n_first * REC_BYTES,
                cmp.ptr,
                cmp_bt.ptr + seq * cmp_width,
                0,
                n_last,
                cmp_page_size,
            )
        sync.m_dq_rec0.unlock(pipe=tla.pipes.MTE2)
        sync.m_dq_rec0.lock(pipe=tla.pipes.VECTOR)
        for slab in static_range(HEAD_DIM // D_CHUNK):
            slot = slab % 2
            lease = sync.m_dq_out0 if slot == 0 else sync.m_dq_out1
            dst = tla.recast_ptr(packed + slot * (half + 1) * D_CHUNK, dtype=tla.UInt16)
            lease.lock(pipe=tla.pipes.VECTOR)
            _pack_fp8_columns(dst, record, n_first + n_last, half + 1, slab * 2)
            lease.unlock(pipe=tla.pipes.VECTOR)
            lease.lock(pipe=tla.pipes.MTE3)
            _single_query_insert_nz(
                tla.recast_ptr(l1 + slab * n * D_CHUNK + row * 16, dtype=tla.UInt16),
                dst,
                half,
                n,
                half + 1,
                D_CHUNK,
            )
            lease.unlock(pipe=tla.pipes.MTE3)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)
        sync.m_dq_rec0.unlock(pipe=tla.pipes.VECTOR)


@tla.jit
def _grouped_softmax(
    prob_l1,
    prob_ub,
    scores_ptr,
    sinks_ub,
    lse_ub,
    sinks,
    lse,
    group,
    first_query,
    q_count,
    m,
    n,
    scale,
    has_sinks,
    enable_lse,
    sync,
    has_cmp: tla.Constexpr[bool] = False,
    compressed=0,
):
    with tla.vector():
        rows = m // 2
        sub = tla.arch.sub_block_idx()
        row0 = sub * rows
        pitch = (n + 63) // 64 * 64
        src = tla.make_tensor(
            scores_ptr,
            tla.make_layout(tla.make_shape(rows, pitch), tla.make_stride(pitch, 1)),
        )
        p_pitch = ((rows + 15) // 16) * 16
        dst = tla.make_tensor(
            prob_ub, tla.make_layout_with_tag((rows, pitch), prob_l1.dtype, tla.arch.zN)
        )
        svec = _stats(sinks_ub, 16)
        if has_sinks:
            sgm = tla.make_tensor(
                sinks.ptr + group * 16,
                tla.make_layout(tla.make_shape(16), tla.make_stride(1)),
            )
            tla.copy(svec, sgm)
            tla.set_flag(sync.phy_m2v)
            tla.wait_flag(sync.phy_m2v)
        stats = _stats(lse_ub, rows)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=1)
        with tla.vec.func(mode="simd"):
            all32 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            one, _ = tla.update_mask(1, dtype=tla.Float32)
            zero = tla.full(0.0, dtype=tla.Float32)
            low = tla.full(MIN_F, dtype=tla.Float32)
            lanes = tla.arange(base=0, order="increase", dtype=tla.Int32)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            cast = _cast_slot(tla.params.RegSlot.ZERO)
            block_store = tla.params.BlockStoreParams(block_stride=p_pitch)
            for query in tla.range(q_count // 2):
                token = sub * (q_count // 2) + query
                masks = ()
                for c in tla.range_constexpr(pitch // 64):
                    ids = tla.add(lanes, c * 64)
                    mask = tla.bitwise_and(
                        tla.cmp(ids, token, "ge"),
                        tla.cmp(ids, token + KV_BLOCK, "lt"),
                    )
                    if has_cmp:
                        ori_count = q_count + KV_BLOCK - 1
                        mask = tla.bitwise_or(
                            mask,
                            tla.bitwise_and(
                                tla.cmp(ids, ori_count, "ge"),
                                tla.cmp(ids, ori_count + compressed, "lt"),
                            ),
                        )
                    masks += (mask,)
                for head in tla.range(16):
                    r = query * 16 + head
                    seed = low
                    if has_sinks:
                        seed = tla.tile_view(
                            svec, tla.make_shape(1), tla.make_coord(head)
                        ).load(params=brc)
                    max_lane = seed
                    values = ()
                    for c in tla.range_constexpr(pitch // 64):
                        ss = tla.tile_view(
                            src, tla.make_shape(1, 64), tla.make_coord(r, c)
                        ).load()
                        val = tla.where(masks[c], tla.mul(ss, scale, mask=all32), low)
                        values += (val,)
                        max_lane = tla.max(max_lane, val, mask=all32)
                    max_v = tla.full(
                        max_lane.reduce(tla.ReductionOp.MAX, mask=all32),
                        dtype=tla.Float32,
                    )
                    exps = ()
                    sums = zero
                    for c in tla.range_constexpr(pitch // 64):
                        val = tla.where(
                            masks[c],
                            tla.exp(tla.sub(values[c], max_v, mask=all32), mask=all32),
                            zero,
                        )
                        exps += (val,)
                        sums = tla.add(sums, val, mask=all32)
                    part_sum = tla.full(
                        sums.reduce(tla.ReductionOp.ADD, mask=all32), dtype=tla.Float32
                    )
                    if has_sinks:
                        part_sum = tla.add(
                            part_sum,
                            tla.exp(tla.sub(seed, max_v, mask=all32), mask=all32),
                            mask=all32,
                        )
                    inv = tla.div(
                        tla.full(1.0, dtype=tla.Float32), part_sum, mask=all32
                    )
                    for c in tla.range_constexpr((pitch + 127) // 128):
                        e0 = tla.mul(exps[2 * c], inv, mask=all32).to(
                            prob_l1.dtype, cast, mask=all32
                        )
                        if 2 * c + 1 < pitch // 64:
                            e1 = tla.mul(exps[2 * c + 1], inv, mask=all32).to(
                                prob_l1.dtype, cast, mask=all32
                            )
                            p0, _ = tla.deinterleave(e0, e1)
                            tla.tile_view(
                                dst, tla.make_shape(1, 128), tla.make_coord(r, c)
                            ).store(p0, params=block_store)
                        else:
                            p1, _ = tla.deinterleave(e0, e0)
                            half_mask, _ = tla.update_mask(64, dtype=prob_l1.dtype)
                            tla.tile_view(
                                dst, tla.make_shape(1, 64), tla.make_coord(r, 2 * c)
                            ).store(p1, params=block_store, mask=half_mask)
                    if enable_lse:
                        tla.tile_view(
                            stats, tla.make_shape(1), tla.make_coord(r)
                        ).store(
                            tla.add(tla.log(part_sum, mask=all32), max_v, mask=all32),
                            tla.params.UnalignStoreParams(),
                        )
        tla.set_flag(sync.p_data_ready)
        tla.wait_flag(sync.p_data_ready)
        _single_query_insert_nz(
            tla.recast_ptr(prob_l1 + row0 * 16, dtype=tla.UInt16),
            tla.recast_ptr(prob_ub, dtype=tla.UInt16),
            rows,
            m,
            p_pitch,
            n,
        )
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=1)
        if enable_lse:
            shape = tla.make_shape(q_count // 2, 16)
            dest = tla.make_tensor(
                lse.ptr + (first_query + row0 // 16) * 64 + group * 16,
                tla.make_layout(shape, tla.make_stride(64, 1)),
            )
            source = tla.make_tensor(
                lse_ub, tla.make_layout(shape, tla.make_stride(16, 1))
            )
            tla.copy(dest, source)


@tla.kernel
def sparse_flash_mla_grouped_dense_kernel(
    q: tla.Tensor,
    kv: tla.Tensor,
    cmp: tla.Tensor,
    page_table: tla.Tensor,
    cmp_bt: tla.Tensor,
    cu_kv: tla.Tensor,
    cu_cmp: tla.Tensor,
    sinks: tla.Tensor,
    out: tla.Tensor,
    lse: tla.Tensor,
    scale: tla.Float32,
    q_count: tla.Constexpr[int],
    page_size: tla.Constexpr[int],
    table_width: tla.Constexpr[int],
    cmp_page_size: tla.Constexpr[int],
    cmp_table_width: tla.Constexpr[int],
    has_sinks: tla.Constexpr[bool],
    enable_lse: tla.Constexpr[bool],
) -> None:
    m = 16 * q_count
    dt = q.ptr.dtype
    rows = m // 2
    ori_count = KV_BLOCK + q_count - 1
    has_cmp = cmp_page_size > 0
    n = KV_BLOCK + 16 + (64 if has_cmp else 0)
    raw_half = n // 2
    score_pitch = (n + 63) // 64 * 64
    dq_in = tla.allocate(raw_half * REC_BYTES, tla.UInt8, tla.AddressSpace.ub, 256)
    dq_out = tla.allocate(
        (raw_half + 1) * (2 * D_CHUNK if has_cmp else HEAD_DIM),
        dt,
        tla.AddressSpace.ub,
        256,
    )
    s_ub_ptr = tla.allocate(rows * score_pitch, tla.Float32, tla.AddressSpace.ub, 256)
    p_ub = tla.allocate(
        ((rows + 15) // 16) * 16 * score_pitch, dt, tla.AddressSpace.ub, 256
    )
    sinks_ub = tla.allocate(16, tla.Float32, tla.AddressSpace.ub, 256)
    lse_ub = tla.allocate(rows, tla.Float32, tla.AddressSpace.ub, 256)
    q_l1_ptr = tla.allocate(m * HEAD_DIM, dt, tla.AddressSpace.l1, 512)
    k_l1_ptr = tla.allocate(n * HEAD_DIM, dt, tla.AddressSpace.l1, 512)
    p_l1_ptr = tla.allocate(m * n, dt, tla.AddressSpace.l1, 512)
    d_step = 64
    l0a = tla.allocate(m * n, dt, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(2 * n * d_step, dt, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(m * HEAD_DIM, tla.Float32, tla.AddressSpace.l0c, 1024)
    sync = declare_sync(need_fused_dequant=True)

    task = tla.arch.block_idx()
    group = task % 4
    seq = task // 4
    first_query = seq * q_count
    end = cu_kv[seq + 1] - cu_kv[seq]
    compressed = tla.as_numeric(0)
    if has_cmp:
        compressed = cu_cmp[seq + 1] - cu_cmp[seq]
        _load_grouped_hca(
            kv,
            cmp,
            page_table,
            cmp_bt,
            end - ori_count,
            ori_count,
            compressed,
            seq,
            page_size,
            table_width,
            cmp_page_size,
            cmp_table_width,
            n,
            k_l1_ptr,
            dq_in,
            dq_out,
            sync,
        )
    else:
        _single_query_load_segment(
            kv,
            page_table,
            page_size,
            end - ori_count,
            ori_count,
            n,
            n,
            0,
            k_l1_ptr,
            dq_in,
            dq_out,
            sync,
            table_start=seq * table_width,
        )
        with tla.vector():
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)
    with tla.cube():
        q_l1 = tla.make_tensor(
            q_l1_ptr, tla.make_layout_with_tag((m, HEAD_DIM), dt, tla.arch.zN)
        )
        k_l1 = tla.make_tensor(
            k_l1_ptr, tla.make_layout_with_tag((HEAD_DIM, n), dt, tla.arch.nZ)
        )
        scores = tla.make_tensor(
            l0c, tla.make_layout_with_tag((m, n), tla.Float32, tla.arch.L0Clayout)
        )
        s_ub = tla.make_tensor(
            s_ub_ptr,
            tla.make_layout(
                tla.make_shape(m, n),
                tla.make_stride(score_pitch, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        tla.set_flag(sync.ab_free0)
        tla.set_flag(sync.ab_free1)
        _load_group_query(q_l1, q.ptr, first_query, group, q_count, dt)
        tla.set_flag(sync.q_l1_ready)
        tla.wait_flag(sync.q_l1_ready)
        if not has_cmp:
            tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE1, aiv_id=0)
            tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE1, aiv_id=1)
        for d in static_range(HEAD_DIM // d_step):
            # Vector publishes one slab as soon as its segment is in L1.
            if has_cmp and d * d_step % D_CHUNK == 0:
                tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE1, aiv_id=0)
                tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE1, aiv_id=1)
            a = tla.tile_view(q_l1, tla.make_shape(m, d_step), tla.make_coord(0, d))
            b = tla.tile_view(k_l1, tla.make_shape(d_step, n), tla.make_coord(d, 0))
            slot = d % 2
            qa = tla.make_tensor_like(l0a + slot * m * d_step, a)
            kb = tla.make_tensor_like(l0b + slot * n * d_step, b)
            free = sync.ab_free0 if slot == 0 else sync.ab_free1
            ready = sync.ab_ready0 if slot == 0 else sync.ab_ready1
            tla.wait_flag(free)
            tla.copy(qa, a)
            tla.copy(kb, b)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            tla.mmad(scores, qa, kb, init_c=(d == 0))
            tla.set_flag(free)
        tla.set_flag(sync.s_mm_ready)
        tla.wait_flag(sync.s_mm_ready)
        tla.copy(
            s_ub,
            scores,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        tla.cross_core_set_flag(sync.s_ready, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_set_flag(sync.s_ready, tla.pipes.FIX, aiv_id=1)
    _grouped_softmax(
        p_l1_ptr,
        p_ub,
        s_ub_ptr,
        sinks_ub,
        lse_ub,
        sinks,
        lse,
        group,
        first_query,
        q_count,
        m,
        n,
        scale,
        has_sinks,
        enable_lse,
        sync,
        has_cmp=has_cmp,
        compressed=compressed,
    )
    wait_p_ready(sync)
    with tla.cube():
        p_l1 = tla.make_tensor(
            p_l1_ptr, tla.make_layout_with_tag((m, n), dt, tla.arch.zN)
        )
        v_l1 = tla.make_tensor(
            k_l1_ptr, tla.make_layout_with_tag((n, HEAD_DIM), dt, tla.arch.zN)
        )
        for d in static_range(HEAD_DIM // d_step):
            acc = tla.make_tensor(
                l0c + d * m * d_step,
                tla.make_layout_with_tag((m, d_step), tla.Float32, tla.arch.L0Clayout),
            )
            slot = d % 2
            v = tla.tile_view(v_l1, tla.make_shape(n, d_step), tla.make_coord(0, d))
            pa = tla.make_tensor(l0a, tla.make_layout_with_tag((m, n), dt, tla.arch.zN))
            vb = tla.make_tensor(
                l0b + slot * n * d_step,
                tla.make_layout_with_tag((n, d_step), dt, tla.arch.nZ),
            )
            free = sync.ab_free0 if slot == 0 else sync.ab_free1
            ready = sync.ab_ready0 if slot == 0 else sync.ab_ready1
            tla.wait_flag(free)
            if d == 0:
                # The P tile occupies both QK A buffers. QK is finished
                # before wait_p_ready returns, so it can reuse the storage.
                tla.wait_flag(sync.ab_free1)
                tla.copy(pa, p_l1)
                tla.set_flag(sync.ab_free1)
            tla.copy(vb, v)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            tla.mmad(acc, pa, vb, init_c=True)
            tla.set_flag(free)
        tla.set_flag(sync.o_mm_ready0)
        tla.wait_flag(sync.o_mm_ready0)
        full_out = tla.make_tensor(
            l0c,
            tla.make_layout_with_tag((m, HEAD_DIM), tla.Float32, tla.arch.L0Clayout),
        )
        for t in static_range(q_count):
            g = tla.make_tensor(
                out.ptr + ((first_query + t) * 64 + group * 16) * HEAD_DIM,
                tla.make_layout_with_tag((16, HEAD_DIM), dt, tla.arch.RowMajor),
            )
            a = tla.tile_view(
                full_out, tla.make_shape(16, HEAD_DIM), tla.make_coord(t, 0)
            )
            tla.copy(g, a)


NPU_ARCH = "--npu-arch 3510"
UB_RELEASE_FLAGS = (
    " --cce-disable-asc-reserved-ubuf --cce-disable-vf-stack-reserved-ubuf"
)
KV_STAGE_BYTES_EL = 3 * 128 * 512

_LAYOUT_CONST = {"BSND": 0, "TND": 1, "PA_BBND": 2}


_OUT_POOL = {}
_LSE_POOL = {}
_ZERO_LEN = {}
_OUT_POOL_SLOTS = 4

_EXECUTOR_CACHE = {}
_EXECUTOR_CACHE_LIMIT = 16

_BINDING_CACHE = {}
_BINDING_CACHE_LIMIT = 512

_DUMMY_CACHE = {}
_DEV_PROP = {}


def _pooled_output(q):
    key = (str(q.device), tuple(q.shape), q.dtype)
    state = _OUT_POOL.get(key)
    if state is None:
        state = {"bufs": [], "i": 0}
        _OUT_POOL[key] = state
    while len(state["bufs"]) < _OUT_POOL_SLOTS:
        state["bufs"].append(torch.empty_like(q))
    out = state["bufs"][state["i"] % _OUT_POOL_SLOTS]
    state["i"] += 1
    return out


def _pooled_lse(device, shape):
    key = (str(device), tuple(shape))
    state = _LSE_POOL.get(key)
    if state is None:
        state = {"bufs": [], "i": 0}
        _LSE_POOL[key] = state
    while len(state["bufs"]) < _OUT_POOL_SLOTS:
        state["bufs"].append(
            torch.empty(tuple(shape), dtype=torch.float32, device=device)
        )
    out = state["bufs"][state["i"] % _OUT_POOL_SLOTS]
    state["i"] += 1
    return out


def _zero_len(device):
    key = str(device)
    cached = _ZERO_LEN.get(key)
    if cached is None:
        cached = torch.empty((0,), dtype=torch.float32, device=device)
        _ZERO_LEN[key] = cached
    return cached


def _cube_cores(device):
    key = str(device)
    cores = _DEV_PROP.get(key)
    if cores is None:
        cores = int(torch_npu.npu.get_device_properties(device).cube_core_num)
        if cores < 1:
            raise RuntimeError("NPU device reports no available Cube cores")
        _DEV_PROP[key] = cores
    return cores


def _tla_tensor(value, layout_tag=None, element_type=None):
    key = (
        value.data_ptr(),
        tuple(value.shape),
        value.dtype,
        layout_tag or tla.arch.RowMajor,
        element_type,
    )
    hit = _BINDING_CACHE.get(key)
    if hit is not None:
        return hit[1]
    contiguous = value.contiguous()
    bound = from_dlpack(
        contiguous,
        layout_tag=layout_tag or tla.arch.RowMajor,
        element_type=element_type,
    )
    bound._torch_storage = contiguous
    if len(_BINDING_CACHE) >= _BINDING_CACHE_LIMIT:
        _BINDING_CACHE.clear()
    _BINDING_CACHE[key] = (value, bound)
    return bound


def _dummy(device, dtype):
    key = (str(device), str(dtype))
    cached = _DUMMY_CACHE.get(key)
    if cached is None:
        cached = torch.zeros(1, dtype=dtype, device=device)
        _DUMMY_CACHE[key] = cached
    return cached


def _align128(n):
    return (int(n) + 127) // 128 * 128


def _check_tensor(name, value, dtype, ndim=None):
    if value is None:
        raise ValueError(f"{name} is required")
    if value.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D")


_CU_CACHE = {}
_BT_CACHE = {}
_LAUNCH_PLAN = {}
_PLAN_LIMIT = 64
_SCRATCH = {}


def _scratch(device, n, dtype, zero=True):
    key = (str(device), int(n), str(dtype), bool(zero))
    buf = _SCRATCH.get(key)
    if buf is None:
        if len(_SCRATCH) >= 32:
            _SCRATCH.clear()
        buf = (torch.zeros if zero else torch.empty)(int(n), dtype=dtype, device=device)
        _SCRATCH[key] = buf
    return buf


def _metadata_cache_key(value):
    version = (
        value._version
        if isinstance(value, torch.Tensor) and not value.is_inference()
        else None
    )
    return id(value), version


def _build_cu(cu, seqused, batch, per_batch, device):
    key = (
        _metadata_cache_key(cu),
        _metadata_cache_key(seqused),
        batch,
        per_batch,
        str(device),
    )
    hit = _CU_CACHE.get(key)
    if hit is not None and hit[0] is cu and hit[1] is seqused:
        return hit[2]
    result = _build_cu_impl(cu, seqused, batch, per_batch, device)
    if len(_CU_CACHE) >= 64:
        _CU_CACHE.clear()
    _CU_CACHE[key] = (cu, seqused, result)
    return result


def _build_cu_impl(cu, seqused, batch, per_batch, device):
    i32 = torch.int32
    if cu is not None:
        _check_tensor("cu_seqlens", cu, i32, 1)
        t = cu.to(i32).contiguous()
        return t, [int(v) for v in t.reshape(-1).tolist()]
    if seqused is not None:
        _check_tensor("seqused", seqused, i32, 1)
        vals = [int(v) for v in seqused.reshape(-1).tolist()]
        acc = [0]
        for v in vals:
            acc.append(acc[-1] + v)
        return torch.tensor(acc, dtype=i32, device=device), acc
    if per_batch is not None:
        n, step = int(per_batch[0]), int(per_batch[1])
        acc = [i * step for i in range(n + 1)]
        return torch.tensor(acc, dtype=i32, device=device), acc
    if batch is None:
        raise ValueError("cannot derive cu_seqlens: pass cu_seqlens_* or seqused_*")
    acc = [0] * (int(batch) + 1)
    return torch.tensor(acc, dtype=i32, device=device), acc


def _block_table(bt, kv, device):
    key = (
        _metadata_cache_key(bt),
        id(kv),
        None if kv is None else tuple(kv.shape),
        str(device),
    )
    hit = _BT_CACHE.get(key)
    if hit is not None and hit[0] is bt and hit[1] is kv:
        return hit[2]
    result = _block_table_impl(bt, kv, device)
    if len(_BT_CACHE) >= 64:
        _BT_CACHE.clear()
    _BT_CACHE[key] = (bt, kv, result)
    return result


def _block_table_impl(bt, kv, device):
    i32 = torch.int32
    if bt is None:
        return torch.zeros(1, dtype=i32, device=device), 0, 0, 0, 1
    _check_tensor("block_table", bt, i32, 2)
    b = bt.to(i32).contiguous()
    block_size = int(kv.shape[1])
    pa_stride = int(kv.numel() // kv.shape[0])
    max_blocks = int(b.shape[1])
    pages = 1 + (KV_BLOCK - 1 + max(block_size, 1) - 1) // max(block_size, 1)
    return b.reshape(-1), block_size, pa_stride, max_blocks, pages


def _vectorize_flag(route, layout_kv, ori_kv, cmp_kv, ori_bt, cmp_bt, ori_w, cmp_w):
    if route not in ("csa", "os", "ocs"):
        return 0
    if layout_kv == "PA_BBND":
        ori_bs = int(ori_kv.shape[1])
        ori_ok = int(ori_bs & (ori_bs - 1) == 0)
        if route == "os":
            blocksize_flag = ori_ok
            pa = int(ori_bt.shape[1]) * 4 if ori_bt is not None else 0
        else:
            cmp_bs = int(cmp_kv.shape[1])
            cmp_ok = int(cmp_bs & (cmp_bs - 1) == 0)
            blocksize_flag = int(ori_ok and cmp_ok)
            max_blocks = 0
            if ori_bt is not None:
                max_blocks = max(max_blocks, int(ori_bt.shape[1]))
            if cmp_bt is not None:
                max_blocks = max(max_blocks, int(cmp_bt.shape[1]))
            pa = max_blocks * 4
    else:
        blocksize_flag = 1
        pa = 0
    vec_ub = max(pa + _align128(cmp_w) * 12, pa + _align128(ori_w) * 12)
    if blocksize_flag == 0 or vec_ub > 184 * 1024:
        return 0
    return 1


def _fa_s2_max_num(
    route,
    cu_q,
    cu_kv,
    cu_cmp,
    ori_topk,
    cmp_topk,
    ori_w,
    cmp_w,
    ori_mask_mode,
    win_l,
    win_r,
    cmp_mask_mode,
    cmp_ratio,
    resid,
    block_num,
    g_split,
):
    nseq = len(cu_q) - 1
    kv_block = KV_BLOCK

    def _ceil_div(a, b):
        return (a + b - 1) // b if a > 0 else 0

    tiles = []
    for tok in range(len(cu_q) - 1):
        seq = 0
        rowq = tok
        for bb in range(1, nseq):
            if tok >= cu_q[bb]:
                seq = bb
                rowq = tok - cu_q[bb]
        q_len = cu_q[seq + 1] - cu_q[seq]
        kv_len = cu_kv[seq + 1] - cu_kv[seq]
        causal = kv_len - q_len
        next_tok = kv_len
        pre_tok = q_len
        if ori_mask_mode == 3:
            next_tok = causal
        elif ori_mask_mode == 4:
            pre_tok = q_len if win_l == -1 else win_l - causal
            next_tok = kv_len if win_r == -1 else causal + win_r
            if pre_tok > q_len:
                pre_tok = q_len
        end = max(0, min(rowq + next_tok + 1, kv_len))
        start = max(0, min(rowq - pre_tok, kv_len))
        range_len = max(0, end - start)

        n = 0
        cmp_lines = 0
        if route in ("csa", "ocs"):
            cmp_len = cu_cmp[seq + 1] - cu_cmp[seq]
            next_cmp = cmp_len * cmp_ratio
            if cmp_mask_mode != 0:
                next_cmp = cmp_len * cmp_ratio + resid[seq] - q_len
            cmp_lines = min(max(0, (rowq + 1 + next_cmp) // cmp_ratio), cmp_len)
            if cmp_topk is not None:
                cmp_lines = min(cmp_lines, max(0, min(cmp_topk[tok], cmp_w)))
        if route == "csa":
            n += _ceil_div(range_len, kv_block)
            n += _ceil_div(cmp_lines, kv_block)
        elif route == "os":
            lines = min(range_len, kv_len)
            if ori_topk is not None:
                lines = min(lines, max(0, min(ori_topk[tok], ori_w)))
            n += _ceil_div(lines, kv_block)
        else:  # ocs
            ori_lines = min(range_len, kv_len)
            if ori_topk is not None:
                ori_lines = min(ori_lines, max(0, min(ori_topk[tok], ori_w)))
            n += _ceil_div(ori_lines, kv_block)
            n += _ceil_div(cmp_lines, kv_block)
        tiles.append(n)

    per_core = [0] * max(block_num, 1)
    for t in range(len(tiles) * g_split):
        per_core[t % max(block_num, 1)] += tiles[t // g_split]
    return max(per_core) if per_core else 0


def attention_run(
    q,
    ori_kv,
    cmp_kv,
    ori_sparse_indices,
    cmp_sparse_indices,
    ori_block_table,
    cmp_block_table,
    cu_seqlens_q,
    cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv,
    seqused_q,
    seqused_ori_kv,
    seqused_cmp_kv,
    cmp_residual_kv,
    ori_topk_length,
    cmp_topk_length,
    sinks,
    metadata,
    softmax_scale,
    cmp_ratio,
    ori_mask_mode,
    cmp_mask_mode,
    ori_win_left,
    ori_win_right,
    layout_q,
    layout_kv,
    return_softmax_lse,
    *,
    compile_only=False,
    fused_dequant=0,
    single_query_kv=0,
    grouped_window_rows=0,
    rolling_records=False,
    dequant_pair=None,
):
    stream = None
    if (
        not compile_only
        and fused_dequant == 3
        and cmp_sparse_indices is not None
        and layout_q == "TND"
        and q.device.type == "npu"
    ):
        stream = torch_npu.npu.current_stream(q.device).npu_stream
    sig = (
        stream,
        id(q),
        id(ori_kv),
        id(cmp_kv),
        id(ori_sparse_indices),
        id(cmp_sparse_indices),
        _metadata_cache_key(ori_block_table),
        _metadata_cache_key(cmp_block_table),
        _metadata_cache_key(cu_seqlens_q),
        _metadata_cache_key(cu_seqlens_ori_kv),
        _metadata_cache_key(cu_seqlens_cmp_kv),
        _metadata_cache_key(seqused_q),
        _metadata_cache_key(seqused_ori_kv),
        _metadata_cache_key(seqused_cmp_kv),
        id(cmp_residual_kv),
        id(ori_topk_length),
        id(cmp_topk_length),
        id(sinks),
        id(metadata),
        None if softmax_scale is None else float(softmax_scale),
        int(cmp_ratio),
        int(ori_mask_mode),
        int(cmp_mask_mode),
        int(ori_win_left),
        int(ori_win_right),
        str(layout_q),
        str(layout_kv),
        bool(return_softmax_lse),
        bool(compile_only),
        int(fused_dequant),
        int(single_query_kv),
        int(grouped_window_rows),
        bool(rolling_records),
        dequant_pair[3] if dequant_pair is not None else None,
    )
    plan = _LAUNCH_PLAN.get(sig)
    if plan is not None:
        plan[0](dequant_pair)
        plan[1]()
        return plan[2], plan[3]

    dtype = q.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q must be float16 or bfloat16")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    if int(q.shape[-1]) != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}")
    heads = int(q.shape[-2])
    if heads < 1:
        raise ValueError("q must have at least one head")
    if layout_q not in ("TND", "BSND"):
        raise ValueError("layout_q must be TND or BSND")
    if layout_kv not in ("TND", "BSND", "PA_BBND"):
        raise ValueError("layout_kv must be TND, BSND or PA_BBND")
    if layout_kv != "PA_BBND" and layout_q != layout_kv:
        raise ValueError("non-paged Q and KV layouts must match")

    ori_records = (int(fused_dequant) & 2) == 2
    cmp_records = (int(fused_dequant) & 1) == 1
    if ori_records or cmp_records:
        for _name, _kv, _flag in (
            ("ori_kv", ori_kv, ori_records),
            ("cmp_kv", cmp_kv, cmp_records),
        ):
            if _kv is None or not _flag:
                continue
            if _kv.dtype not in (torch.uint8, torch.int8):
                raise ValueError(f"{_name} records must be uint8/int8")
            if not _kv.is_contiguous():
                raise ValueError(f"{_name} records must be contiguous")
            if int(_kv.shape[-1]) != REC_BYTES:
                raise ValueError(f"{_name} records must have {REC_BYTES} bytes per row")
    if not ori_records:
        _check_tensor("ori_kv", ori_kv, dtype)
    if int(ori_kv.shape[-2]) != 1:
        raise ValueError("only a single KV head (num_kv_heads == 1) is supported")

    ori_mask_mode = int(ori_mask_mode)
    cmp_mask_mode = int(cmp_mask_mode)
    cmp_ratio = int(cmp_ratio)
    if ori_mask_mode not in (0, 3, 4):
        raise ValueError("ori_mask_mode must be 0, 3 or 4")
    if cmp_mask_mode not in (0, 3):
        raise ValueError("cmp_mask_mode must be 0 or 3")
    if cmp_ratio < 1:
        raise ValueError("cmp_ratio must be positive")
    if cmp_kv is None and cmp_mask_mode != 0:
        raise ValueError("cmp_mask_mode must be 0 without compressed KV")
    if cmp_kv is None and cmp_sparse_indices is not None:
        raise ValueError("compressed sparse indices require compressed KV")
    if metadata is None or metadata.dtype != torch.int32:
        raise ValueError("metadata must be int32")
    if metadata.ndim != 1:
        raise ValueError("metadata must be 1-D")
    scale_value = float(softmax_scale)
    want_lse = bool(return_softmax_lse)

    has_cmp = cmp_kv is not None
    has_ori_idx = ori_sparse_indices is not None
    has_cmp_idx = cmp_sparse_indices is not None

    paged = layout_kv == "PA_BBND"
    if layout_q == "TND":
        if q.ndim != 3:
            raise ValueError("TND query layout requires q [T1, N1, 512]")
        if cu_seqlens_q is None and seqused_q is None:
            raise ValueError("TND query layout requires int32 cu_seqlens_q")
        batch = (
            int(cu_seqlens_q.numel()) - 1
            if cu_seqlens_q is not None
            else int(seqused_q.numel())
        )
        total_rows = int(q.shape[0])
        s1_len = 0
    else:
        if q.ndim != 4:
            raise ValueError("BSND query layout requires q [B, S1, N1, 512]")
        batch = int(q.shape[0])
        s1_len = int(q.shape[1])
        total_rows = batch * s1_len

    if paged:
        if ori_kv.ndim != 4:
            raise ValueError("PA_BBND KV requires [blocks, page, 1, 512]")
        kv_bs = 0
        if seqused_ori_kv is None:
            raise ValueError("paged KV requires seqused_ori_kv")
        if ori_block_table is None or ori_block_table.dtype != torch.int32:
            raise ValueError("paged KV requires int32 ori_block_table")
        _check_tensor("ori_block_table", ori_block_table, torch.int32, 2)
        ori_bts = int(ori_block_table.shape[1])
        if int(ori_block_table.shape[0]) != batch:
            raise ValueError("ori_block_table batch mismatch")
    elif layout_kv == "TND":
        if ori_kv.ndim != 3:
            raise ValueError("TND KV requires [T2, 1, 512]")
        kv_bs = 0
        ori_bts = 0
        if cu_seqlens_ori_kv is None and seqused_ori_kv is None:
            raise ValueError("TND KV requires int32 cu_seqlens_ori_kv")
    else:
        if ori_kv.ndim != 4:
            raise ValueError("BSND KV requires [B, S2, 1, 512]")
        kv_bs = int(ori_kv.shape[1])
        ori_bts = 0

    cmp_bs = 0
    cmp_bts = 0
    if has_cmp:
        if not cmp_records:
            _check_tensor("cmp_kv", cmp_kv, dtype)
        if paged:
            if cmp_kv.ndim != 4:
                raise ValueError("paged compressed KV requires [blocks, page, 1, 512]")
            if seqused_cmp_kv is None:
                raise ValueError("paged compressed KV requires seqused_cmp_kv")
            if cmp_block_table is None or cmp_block_table.dtype != torch.int32:
                raise ValueError("paged compressed KV requires int32 cmp_block_table")
            cmp_bts = int(cmp_block_table.shape[1])
        elif layout_kv == "BSND":
            if cmp_kv.ndim != 4:
                raise ValueError("BSND compressed KV requires [B, S2c, 1, 512]")
            cmp_bs = int(cmp_kv.shape[1])
        else:
            if cmp_kv.ndim != 3:
                raise ValueError("TND compressed KV requires [T2c, 1, 512]")
            if cu_seqlens_cmp_kv is None and seqused_cmp_kv is None:
                raise ValueError("TND compressed KV requires int32 cu_seqlens_cmp_kv")

    for name, value in (
        ("ori_sparse_indices", ori_sparse_indices),
        ("cmp_sparse_indices", cmp_sparse_indices),
        ("ori_topk_length", ori_topk_length),
        ("cmp_topk_length", cmp_topk_length),
        ("seqused_q", seqused_q),
        ("seqused_ori_kv", seqused_ori_kv),
        ("seqused_cmp_kv", seqused_cmp_kv),
        ("cmp_residual_kv", cmp_residual_kv),
    ):
        if value is not None:
            _check_tensor(name, value, torch.int32)
    if sinks is not None:
        if sinks.dtype != torch.float32 or not sinks.is_contiguous():
            raise ValueError("sinks must be contiguous float32")
        if int(sinks.numel()) != heads:
            raise ValueError("sinks must have N1 elements")

    ori_w = int(ori_sparse_indices.shape[-1]) if has_ori_idx else 0
    cmp_w = int(cmp_sparse_indices.shape[-1]) if has_cmp_idx else 0
    if has_ori_idx:
        if int(ori_sparse_indices.numel()) != total_rows * ori_w:
            raise ValueError("ori_sparse_indices shape does not match the query rows")
    if has_cmp_idx:
        if int(cmp_sparse_indices.numel()) != total_rows * cmp_w:
            raise ValueError("cmp_sparse_indices shape does not match the query rows")

    if has_cmp_idx and has_ori_idx:
        route = "ocs"
    elif has_cmp_idx:
        route = "csa"
    elif has_ori_idx:
        if has_cmp:
            raise ValueError(
                "ori sparse indices with compressed KV require "
                "cmp_sparse_indices (ORI_CMP_SPARSE route)"
            )
        route = "os"
    elif has_cmp:
        route = "hca"
    else:
        route = "swa"

    if (
        (ori_records or cmp_records)
        and route != "csa"
        and not (single_query_kv or grouped_window_rows or rolling_records)
    ):
        raise ValueError("fused_dequant is only implemented on the CSA route")
    if route == "csa" and cmp_topk_length is None:
        raise ValueError("CSA route requires cmp_topk_length")
    if route == "os" and ori_topk_length is None:
        raise ValueError("ORI_SPARSE route requires ori_topk_length")
    if route == "ocs" and (ori_topk_length is None or cmp_topk_length is None):
        raise ValueError("ORI_CMP_SPARSE route requires both *_topk_length")

    device = q.device
    i32 = torch.int32
    f32 = torch.float32
    zero_len = _zero_len(device)

    if total_rows == 0:
        output = _pooled_output(q)
        lse = (
            _pooled_lse(
                device,
                (1, total_rows, heads)
                if layout_q == "TND"
                else (batch, 1, s1_len, heads),
            )
            if want_lse
            else zero_len
        )
        return output, lse

    per_q = (batch, s1_len) if layout_q == "BSND" else None
    per_kv = (batch, kv_bs) if layout_kv == "BSND" else None
    cu_q_t, cu_q = _build_cu(cu_seqlens_q, seqused_q, batch, per_q, device)
    cu_kv_t, cu_kv = _build_cu(cu_seqlens_ori_kv, seqused_ori_kv, batch, per_kv, device)
    num_seqs = len(cu_q) - 1

    def _pack_index(lengths, step):
        parts = [
            torch.arange(b * step, b * step + v, device=device)
            for b, v in enumerate(lengths)
            if v > 0
        ]
        return torch.cat(parts).to(torch.long) if parts else None

    q_rows_idx = kv_rows_idx = cmp_rows_idx = None
    if layout_q == "BSND" and seqused_q is not None:
        q_lens = [int(v) for v in seqused_q.reshape(-1).tolist()]
        if len(q_lens) == batch and any(v != s1_len for v in q_lens):
            q_rows_idx = _pack_index(q_lens, s1_len)
    if not paged and layout_kv == "BSND" and seqused_ori_kv is not None:
        kv_lens = [int(v) for v in seqused_ori_kv.reshape(-1).tolist()]
        if len(kv_lens) == batch and any(v != kv_bs for v in kv_lens):
            kv_rows_idx = _pack_index(kv_lens, kv_bs)
    if not paged and has_cmp and layout_kv == "BSND" and seqused_cmp_kv is not None:
        cmp_lens = [int(v) for v in seqused_cmp_kv.reshape(-1).tolist()]
        if len(cmp_lens) == batch and any(v != cmp_bs for v in cmp_lens):
            cmp_rows_idx = _pack_index(cmp_lens, cmp_bs)
    run_rows = int(q_rows_idx.numel()) if q_rows_idx is not None else total_rows

    if has_cmp:
        cmp_per_batch = (batch, cmp_bs) if layout_kv == "BSND" else None
        cu_cmp_t, cu_cmp = _build_cu(
            cu_seqlens_cmp_kv, seqused_cmp_kv, batch, cmp_per_batch, device
        )
    else:
        cu_cmp_t, cu_cmp = None, [0, 0]

    bt_o_t, pa_o_bs, pa_o_stride, pa_o_max, pa_o_pages = _block_table(
        ori_block_table, ori_kv, device
    )
    if has_cmp:
        bt_c_t, pa_c_bs, pa_c_stride, pa_c_max, pa_c_pages = _block_table(
            cmp_block_table, cmp_kv, device
        )
    else:
        bt_c_t = _dummy(device, i32)
        pa_c_bs = pa_c_stride = pa_c_max = pa_c_pages = 0

    g_split = 2 if heads > 64 else 1  # OT: splitG = gSize > 64
    if g_split == 1 and heads % 4 == 0 and int(cu_q[-1]) == 1:
        g_split = 4 if route in ("swa", "hca") else 2

    block_num = min(_cube_cores(device), WS_BLOCKS)
    _n_task = int(cu_q[-1]) * g_split if cu_q else 0
    if 0 < _n_task < block_num:
        block_num = _n_task

    direct_kv = (
        route == "csa"
        and fused_dequant == 3
        and run_rows > 1
        and heads == Q_ROWS
        and dtype == torch.bfloat16
        and ori_kv.dtype == cmp_kv.dtype == torch.uint8
        and pa_o_bs > 0
        and pa_c_bs > 0
        and cmp_w % KV_BLOCK == 0
    )
    split_decode = (
        not compile_only
        and direct_kv
        and paged
        and layout_q == "TND"
        and cmp_mask_mode == 0
        and ori_mask_mode == 4
        and ori_win_left == KV_BLOCK - 1
        and ori_win_right == 0
        and cu_q[0] == 0
        and cu_q[-1] == run_rows
        and block_num < run_rows <= block_num * 3 // 2
        and cmp_w >= 2 * KV_BLOCK
        and cmp_w % (2 * KV_BLOCK) == 0
        and all(
            0 <= cu_q[i + 1] - cu_q[i] <= cu_kv[i + 1] - cu_kv[i]
            for i in range(num_seqs)
        )
    )

    bounded_decode = split_decode and g_split == 1 and cmp_w <= 1024

    # The split plan owns its outputs and partials for its stream.
    output = torch.empty_like(q) if split_decode else _pooled_output(q)
    if want_lse:
        lse_shape = (
            (1, total_rows, heads) if layout_q == "TND" else (batch, 1, s1_len, heads)
        )
        lse = (
            torch.empty(lse_shape, dtype=f32, device=device)
            if split_decode
            else _pooled_lse(device, lse_shape)
        )
        lse_flat = (
            torch.empty(run_rows * heads, dtype=f32, device=device)
            if q_rows_idx is not None
            else lse.reshape(-1)
        )
    else:
        lse = zero_len
        lse_flat = torch.empty(run_rows * heads, dtype=f32, device=device)

    q_rows_flat = q.reshape(total_rows, heads * HEAD_DIM)
    if q_rows_idx is not None:
        q_rows_flat = q_rows_flat.index_select(0, q_rows_idx)
    q2 = q_rows_flat.reshape(-1, HEAD_DIM).contiguous()
    kv_width = REC_BYTES if ori_records else HEAD_DIM
    cmp_width_r = REC_BYTES if cmp_records else HEAD_DIM
    kv_rows_flat = ori_kv.reshape(-1, kv_width)
    if kv_rows_idx is not None:
        kv_rows_flat = kv_rows_flat.index_select(0, kv_rows_idx)
    kv2 = kv_rows_flat.contiguous()
    if q_rows_idx is not None:
        out_pack = torch.empty(
            run_rows, heads * HEAD_DIM, dtype=output.dtype, device=device
        )
        out2 = out_pack.reshape(-1, HEAD_DIM)
    else:
        out_pack = None
        out2 = output.reshape(total_rows * heads, HEAD_DIM)
    if has_cmp:
        cmp_rows_flat = cmp_kv.reshape(-1, cmp_width_r)
        if cmp_rows_idx is not None:
            cmp_rows_flat = cmp_rows_flat.index_select(0, cmp_rows_idx)
        cmp2 = cmp_rows_flat.contiguous()
    else:
        cmp2 = None

    sinks_use = (
        sinks if sinks is not None else torch.zeros(heads, dtype=f32, device=device)
    )
    has_sinks = 0 if sinks is None else 1

    resid = (
        cmp_residual_kv.reshape(-1).to(i32).contiguous()
        if (cmp_residual_kv is not None and cmp_ratio > 1 and cmp_mask_mode != 0)
        else torch.zeros(num_seqs, dtype=i32, device=device)
    )

    kv_layout = _LAYOUT_CONST[layout_kv]
    is_vec = _vectorize_flag(
        route,
        layout_kv,
        ori_kv,
        cmp_kv if has_cmp else ori_kv,
        ori_block_table,
        cmp_block_table if has_cmp else None,
        ori_w,
        cmp_w,
    )

    tiling = _scratch(device, 4, i32, zero=True)
    tiling.zero_()
    if route in ("csa", "os", "ocs") and g_split == 2:
        ori_topk = (
            [int(v) for v in ori_topk_length.reshape(-1).tolist()]
            if ori_topk_length is not None
            else None
        )
        cmp_topk = (
            [int(v) for v in cmp_topk_length.reshape(-1).tolist()]
            if cmp_topk_length is not None
            else None
        )
        resid_list = [int(v) for v in resid.reshape(-1).tolist()]
        tiling[1] = _fa_s2_max_num(
            route,
            cu_q,
            cu_kv,
            cu_cmp,
            ori_topk,
            cmp_topk,
            ori_w,
            cmp_w,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            cmp_mask_mode,
            cmp_ratio,
            resid_list,
            block_num,
            g_split,
        )

    scale_t = tla.Float32(scale_value)

    if route in ("os", "ocs"):
        ws = _scratch(device, 2 * WS_BLOCKS * KV_STAGE_BYTES_EL, dtype)
    else:
        ws = _dummy(device, dtype)

    fold_executor = None
    combined_preprocess = False
    if grouped_window_rows:
        cmp_input = cmp2 if has_cmp else kv2
        cmp_lengths = cu_cmp_t if has_cmp else cu_kv_t
        inputs = (
            q2,
            kv2,
            cmp_input,
            bt_o_t,
            bt_c_t,
            cu_kv_t,
            cmp_lengths,
            sinks_use,
            out2,
            lse_flat,
        )
        tensor_args = tuple(_tla_tensor(t) for t in inputs)
        tensor_shapes = [tuple(t.shape) for t in inputs]
        const_args = (
            grouped_window_rows,
            pa_o_bs,
            pa_o_max,
            pa_c_bs,
            pa_c_max,
            bool(has_sinks),
            want_lse,
        )
        block_num = num_seqs * (heads // 16)
        kernel_fn = sparse_flash_mla_grouped_dense_kernel
    elif single_query_kv:
        cmp_input = cmp2 if has_cmp else kv2
        cmp_lengths = cu_cmp_t if has_cmp else cu_kv_t
        inputs = (
            q2,
            kv2,
            cmp_input,
            bt_o_t,
            bt_c_t,
            cu_kv_t,
            cmp_lengths,
            sinks_use,
            out2,
            lse_flat,
        )
        kernel_fn = sparse_flash_mla_single_query_kernel
        if cmp_sparse_indices is not None:
            inputs += (
                cmp_sparse_indices.reshape(-1).contiguous(),
                cmp_topk_length.reshape(-1).contiguous(),
            )
            kernel_fn = sparse_flash_mla_csa_short_kernel
        tensor_args = tuple(_tla_tensor(t) for t in inputs)
        tensor_shapes = [tuple(t.shape) for t in inputs]
        const_args = (single_query_kv, pa_o_bs, pa_c_bs, bool(has_sinks), want_lse)
        block_num = heads // 16
    elif route == "swa":
        t_kv = (
            _tla_tensor(kv2, element_type=tla.UInt8)
            if rolling_records
            else _tla_tensor(kv2)
        )
        m_rows = heads // g_split - (heads // g_split) // 2
        rows8 = (m_rows + 7) // 8 * 8
        bc_stride = m_rows * HEAD_DIM + rows8 * 2
        bc_ws = _scratch(device, block_num * 2 * bc_stride, f32)
        tensor_args = (
            _tla_tensor(q2),
            t_kv,
            _tla_tensor(bt_o_t),
            _tla_tensor(sinks_use),
            _tla_tensor(cu_q_t),
            _tla_tensor(cu_kv_t),
            _tla_tensor(out2),
            _tla_tensor(lse_flat),
            _tla_tensor(tiling),
            _tla_tensor(bc_ws),
        )
        tensor_shapes = [
            tuple(t.shape)
            for t in (
                q2,
                kv2,
                bt_o_t,
                sinks_use,
                cu_q_t,
                cu_kv_t,
                out2,
                lse_flat,
                tiling,
                bc_ws,
            )
        ]
        # With a small remainder, rounding every slice up leaves one short
        # final slice. Distribute the remainder while keeping each core's KV
        # window contiguous.
        balance_task_ranges = (
            rolling_records and 0 < _n_task % block_num <= block_num // 4
        )
        const_args = (
            heads,
            HEAD_DIM,
            num_seqs,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            pa_o_bs,
            pa_o_stride,
            pa_o_max,
            pa_o_pages,
            g_split,
            has_sinks,
            1 if want_lse else 0,
            0,
            paged
            and num_seqs == 1
            and heads == 64
            and run_rows >= KV_BLOCK
            and ori_mask_mode == 4
            and ori_win_left == KV_BLOCK - 1
            and ori_win_right == 0
            and pa_o_bs >= KV_BLOCK
            and pa_o_bs % KV_BLOCK == 0
            and cu_kv[-1] >= KV_BLOCK,
            rolling_records,
            balance_task_ranges,
        )
        kernel_fn = sparse_flash_mla_swa_kernel
    elif route == "hca":
        cmp_len_max = max(
            (cu_cmp[i + 1] - cu_cmp[i] for i in range(len(cu_cmp) - 1)), default=0
        )
        cmp_blk = 32 if cmp_len_max <= 32 else (64 if cmp_len_max <= 64 else 128)
        rolling_hca = (
            paged
            and layout_q == "TND"
            and num_seqs == 1
            and heads == Q_ROWS
            and dtype == torch.bfloat16
            and run_rows >= KV_BLOCK
            and cmp_mask_mode == 0
            and cu_q == [0, run_rows]
            and cu_kv[0] == cu_cmp[0] == 0
            and cu_kv[-1] >= run_rows
            and 0 < cmp_len_max <= 64
            and ori_mask_mode == 4
            and ori_win_left == KV_BLOCK - 1
            and ori_win_right == 0
            and pa_o_bs >= KV_BLOCK
            and pa_o_bs % KV_BLOCK == 0
            and pa_c_bs > 0
        )
        if rolling_hca:
            inputs = (
                q2,
                kv2,
                cmp2,
                bt_o_t,
                bt_c_t,
                cu_q_t,
                cu_kv_t,
                cu_cmp_t,
                sinks_use,
                out2,
                lse_flat,
            )
            tensor_args = tuple(_tla_tensor(t) for t in inputs)
            tensor_shapes = [tuple(t.shape) for t in inputs]
            const_args = (
                pa_o_bs,
                pa_o_stride,
                pa_c_bs,
                pa_c_stride,
                cmp_blk,
                bool(has_sinks),
                want_lse,
            )
            kernel_fn = sparse_flash_mla_rolling_hca_kernel
        else:
            q_hca = q2
            if g_split == 2 and heads % 2 == 1:
                half_g = (heads + 1) // 2
                m_cube = half_g + (half_g % 2)
                pad_rows = max(0, m_cube - (heads - half_g))
                if pad_rows:
                    q_pad = torch.zeros(
                        (q2.shape[0] + pad_rows, HEAD_DIM),
                        dtype=q2.dtype,
                        device=device,
                    )
                    q_pad[: q2.shape[0]] = q2
                    q_hca = q_pad
            sinks_hca = sinks_use
            if g_split == 2 and heads % 2 == 1:
                sink_pad = ((heads + 7) // 8) * 8 - heads
                if sink_pad:
                    s_pad = torch.zeros(heads + sink_pad, dtype=f32, device=device)
                    s_pad[:heads] = sinks_use
                    sinks_hca = s_pad
            tensor_args = (
                _tla_tensor(q_hca),
                _tla_tensor(kv2),
                _tla_tensor(cmp2),
                _tla_tensor(bt_o_t),
                _tla_tensor(bt_c_t),
                _tla_tensor(cu_q_t),
                _tla_tensor(cu_kv_t),
                _tla_tensor(cu_cmp_t),
                _tla_tensor(resid),
                _tla_tensor(sinks_hca),
                _tla_tensor(out2),
                _tla_tensor(lse_flat),
                _tla_tensor(tiling),
            )
            tensor_shapes = [
                tuple(t.shape)
                for t in (
                    q_hca,
                    kv2,
                    cmp2,
                    bt_o_t,
                    bt_c_t,
                    cu_q_t,
                    cu_kv_t,
                    cu_cmp_t,
                    resid,
                    sinks_hca,
                    out2,
                    lse_flat,
                    tiling,
                )
            ]
            const_args = (
                heads,
                HEAD_DIM,
                num_seqs,
                cmp_ratio,
                cmp_mask_mode,
                ori_mask_mode,
                int(ori_win_left),
                int(ori_win_right),
                pa_o_bs,
                pa_o_stride,
                pa_o_max,
                pa_o_pages,
                pa_c_bs,
                pa_c_stride,
                pa_c_max,
                pa_c_pages,
                g_split,
                has_sinks,
                1 if want_lse else 0,
                int(cmp_blk),
            )
            kernel_fn = sparse_flash_mla_hca_kernel
    elif route == "csa":
        idx = cmp_sparse_indices.reshape(total_rows, cmp_w)
        topk = cmp_topk_length.reshape(-1)
        if q_rows_idx is not None:
            idx = idx.index_select(0, q_rows_idx)
            topk = topk.index_select(0, q_rows_idx)
        idx = idx.reshape(-1).to(i32).contiguous()
        topk = topk.to(i32).contiguous()
        phy_w = _align128(cmp_w) if is_vec == 1 else 1
        if not split_decode:
            ws = _scratch(device, 2 * WS_BLOCKS * KV_STAGE_BYTES_EL, dtype)
        fold_runs = (
            not compile_only
            and not fused_dequant
            and paged
            and layout_q == "TND"
            and num_seqs == 1
            and heads == Q_ROWS
            and run_rows >= KV_BLOCK
            and dtype == torch.bfloat16
            and cmp_mask_mode == 0
            and ori_mask_mode == 4
            and ori_win_left == KV_BLOCK - 1
            and ori_win_right == 0
            and pa_o_bs >= KV_BLOCK
            and pa_o_bs % KV_BLOCK == 0
            and pa_c_bs > 0
            and (pa_c_bs & (pa_c_bs - 1)) == 0
            and pa_c_max <= 256
            and cu_q == [0, run_rows]
            and cu_kv[0] == cu_cmp[0] == 0
            and cu_kv[-1] >= max(KV_BLOCK, run_rows - block_num)
            and 0 < cmp_w <= 1024
            and cmp_w % KV_BLOCK == 0
        )
        mark_span = (cu_cmp[-1] + 63) // 64 * 64
        paired_fold = fold_runs and 128 <= mark_span <= 2048
        overlap_folded_pv = paired_fold
        compact_probabilities = overlap_folded_pv and run_rows > 96 * block_num
        paired_prefix = block_num if paired_fold and cu_kv[-1] < run_rows else 0
        single_probability_bank = paired_fold and run_rows > 96 * block_num
        # Retain the last Q segment in A1 for medium groups with enough original
        # context to keep the leading windows from being dominated by short tiles.
        warm_query_groups = (
            single_probability_bank
            and cmp_w <= 512
            and cu_kv[-1] - run_rows >= 32
            and 118 * block_num < run_rows <= 128 * block_num
        )
        keep_folded_q_suffix = warm_query_groups or (
            single_probability_bank and run_rows > 120 * block_num
        )
        prefetch_folded_v = warm_query_groups and run_rows <= 122 * block_num
        # Long groups with a complete leading window schedule one contiguous
        # range per core. Keeping pair boundaries intact preserves KV reuse.
        linear_pairs = (
            keep_folded_q_suffix
            and not paired_prefix
            and 136 * block_num < run_rows <= 256 * block_num
        )
        contiguous_pairs = (
            single_probability_bank
            and not paired_prefix
            and (
                112 * block_num < run_rows <= 120 * block_num
                or (warm_query_groups and not prefetch_folded_v)
                or (
                    keep_folded_q_suffix
                    and not warm_query_groups
                    and run_rows <= 136 * block_num
                )
                or linear_pairs
            )
        )
        # Spread the remainder as individual queries, then assign the same
        # number of adjacent pairs to every core.
        tail_queries = run_rows % (2 * block_num)
        arithmetic_tail_setup = (
            contiguous_pairs
            and cmp_w <= 512
            and 120 * block_num < run_rows <= 132 * block_num
            and 0 < tail_queries <= 5
        )
        leading_tail_queries = arithmetic_tail_setup or (
            contiguous_pairs
            and warm_query_groups
            and cu_kv[-1] - run_rows >= 96
            and 0 < tail_queries <= 8
        )
        split_tail_queries = leading_tail_queries or (
            contiguous_pairs
            and 132 * block_num < run_rows <= 256 * block_num
            and 16 <= tail_queries <= block_num
        )
        trailing_tail_queries = (
            linear_pairs
            and cmp_w <= 512
            and run_rows > 140 * block_num
            and cu_kv[-1] - run_rows < KV_BLOCK
            and 0 < tail_queries <= 8
        )
        if split_tail_queries or trailing_tail_queries:
            paired_prefix = tail_queries
            linear_pairs = True
        center_pair_tail = (
            contiguous_pairs
            and not linear_pairs
            and tail_queries == 2
            and 132 * block_num < run_rows
        )
        if fold_runs:
            folded_idx = torch.empty_like(idx)
            folded_topk = (
                torch.empty(run_rows * 8, dtype=i32, device=device)
                if paired_fold
                else torch.empty_like(topk)
            )
            cmp_phy = torch.empty_like(idx, dtype=f32)
            fold_tensors = (idx, topk, cu_cmp_t, folded_idx, cmp_phy, folded_topk)
            fold_args = tuple(_tla_tensor(t) for t in fold_tensors)
            chunk_rows = 8 if cmp_w <= 512 else 4
            if paired_fold and cmp_w <= 512:
                chunk_rows = 9
            if paired_fold:
                batch_rows = 2 * chunk_rows
                fold_steps = (paired_prefix + batch_rows - 1) // batch_rows + (
                    run_rows - paired_prefix + batch_rows - 1
                ) // batch_rows
                fold_grid = _grid_for(device, fold_steps)
                fold_config = (run_rows, cmp_w, mark_span, chunk_rows, paired_prefix)
                fold_fn = fold_and_pair_ring_indices
            else:
                fold_grid = _grid_for(device, (run_rows + chunk_rows - 1) // chunk_rows)
                fold_config = (run_rows, cmp_w, chunk_rows)
                fold_fn = fold_cmp_indices
            fold_key = (
                "cmp_runs",
                str(device),
                tuple(tuple(t.shape) for t in fold_tensors),
                paired_fold,
                fold_config,
            )
            compile_fold_args = fold_args
            if paired_fold and dequant_pair is not None:
                dq_args, dq_config, dq_blocks, dq_key = dequant_pair
                fold_key += (dq_key, overlap_folded_pv, center_pair_tail)
                fold_config += (dq_config, overlap_folded_pv, center_pair_tail)
                fold_grid = max(fold_grid, _grid_for(device, dq_blocks))
                compile_fold_args = dq_args + fold_args
                fold_fn = dequant_and_fold_indices
                combined_preprocess = True
            if trailing_tail_queries:
                fold_config += (True,)
                fold_key += (True,)
            fold_executor = _EXECUTOR_CACHE.get(fold_key)
            if fold_executor is None:
                fold_executor = tla.compile(
                    fold_fn, *compile_fold_args, *fold_config, options=NPU_ARCH
                )
                _EXECUTOR_CACHE[fold_key] = fold_executor
            idx, topk = folded_idx, folded_topk
        elif split_decode:
            splits = run_rows - block_num
            ws = torch.empty(
                splits * Q_ROWS * HEAD_DIM, dtype=torch.float32, device=device
            )
            stats_size = splits * 2 * Q_ROWS + (block_num * 16 if bounded_decode else 0)
            cmp_phy = torch.empty(stats_size, dtype=torch.float32, device=device)
        else:
            cmp_phy = torch.zeros(run_rows * phy_w, dtype=torch.int64, device=device)
        ori_etype = None
        if ori_records:
            ori_etype = tla.Int8 if kv2.dtype == torch.int8 else tla.Float8E4M3FN
        cmp_etype = None
        if cmp_records:
            cmp_etype = tla.Int8 if cmp2.dtype == torch.int8 else tla.Float8E4M3FN
        tensor_args = (
            _tla_tensor(q2),
            _tla_tensor(kv2, element_type=ori_etype),
            _tla_tensor(cmp2, element_type=cmp_etype),
            _tla_tensor(bt_o_t),
            _tla_tensor(bt_c_t),
            _tla_tensor(idx),
            _tla_tensor(topk),
            _tla_tensor(cu_q_t),
            _tla_tensor(cu_kv_t),
            _tla_tensor(cu_cmp_t),
            _tla_tensor(resid),
            _tla_tensor(sinks_use),
            _tla_tensor(out2),
            _tla_tensor(lse_flat),
            _tla_tensor(tiling),
            _tla_tensor(ws),
            _tla_tensor(cmp_phy),
        )
        tensor_shapes = [
            tuple(t.shape)
            for t in (
                q2,
                kv2,
                cmp2,
                bt_o_t,
                bt_c_t,
                idx,
                topk,
                cu_q_t,
                cu_kv_t,
                cu_cmp_t,
                resid,
                sinks_use,
                out2,
                lse_flat,
                tiling,
                ws,
                cmp_phy,
            )
        ]
        const_args = (
            heads,
            HEAD_DIM,
            num_seqs,
            cmp_w,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            cmp_mask_mode,
            cmp_ratio,
            pa_o_bs,
            pa_o_stride,
            pa_o_max,
            pa_o_pages,
            pa_c_bs,
            pa_c_stride,
            pa_c_max,
            g_split,
            has_sinks,
            1 if want_lse else 0,
            1,
            kv_layout,
            is_vec,
            phy_w,
            int(fused_dequant),
            paged
            and heads % (2 * g_split) == 0
            and ori_mask_mode == 4
            and any(
                cu_q[i + 1] - cu_q[i] > cu_kv[i + 1] - cu_kv[i] for i in range(num_seqs)
            ),
            direct_kv,
            fold_runs,
            pa_c_bs.bit_length() - 1 if fold_runs else 0,
            split_decode,
            paired_fold,
            paired_prefix,
            overlap_folded_pv,
            compact_probabilities,
            single_probability_bank,
            contiguous_pairs,
            keep_folded_q_suffix,
            prefetch_folded_v,
            run_rows if linear_pairs else 0,
            cu_kv[-1] if linear_pairs else 0,
            cu_cmp[-1] if linear_pairs else 0,
            split_tail_queries,
            center_pair_tail,
            leading_tail_queries,
            arithmetic_tail_setup,
            trailing_tail_queries,
        )
        kernel_fn = sparse_flash_mla_csa_kernel
        if bounded_decode:
            q_span = cu_q[1]
            if not all(value == i * q_span for i, value in enumerate(cu_q)):
                q_span = 0
            uniform_ori = all(value == i * cu_kv[1] for i, value in enumerate(cu_kv))
            uniform_cmp = all(value == i * cu_cmp[1] for i, value in enumerate(cu_cmp))
            tensor_args = tensor_args[:10] + tensor_args[11:14] + tensor_args[15:]
            tensor_shapes = (
                tensor_shapes[:10] + tensor_shapes[11:14] + tensor_shapes[15:]
            )
            const_args = (
                num_seqs,
                cmp_w,
                pa_o_bs,
                pa_o_max,
                pa_c_bs,
                pa_c_max,
                has_sinks,
                want_lse,
                q_span,
                uniform_ori,
                uniform_cmp,
            )
            kernel_fn = sparse_flash_mla_csa_bounded_decode_kernel
    elif route == "os":
        idx = ori_sparse_indices.reshape(total_rows, ori_w)
        topk = ori_topk_length.reshape(-1)
        if q_rows_idx is not None:
            idx = idx.index_select(0, q_rows_idx)
            topk = topk.index_select(0, q_rows_idx)
        idx = idx.reshape(-1).to(i32).contiguous()
        topk = topk.to(i32).contiguous()
        phy_w = _align128(ori_w) if is_vec == 1 else 1
        ori_phy = torch.zeros(run_rows * phy_w, dtype=torch.int64, device=device)
        tensor_args = (
            _tla_tensor(q2),
            _tla_tensor(kv2),
            _tla_tensor(bt_o_t),
            _tla_tensor(idx),
            _tla_tensor(topk),
            _tla_tensor(cu_q_t),
            _tla_tensor(cu_kv_t),
            _tla_tensor(sinks_use),
            _tla_tensor(out2),
            _tla_tensor(lse_flat),
            _tla_tensor(tiling),
            _tla_tensor(ws),
            _tla_tensor(ori_phy),
        )
        tensor_shapes = [
            tuple(t.shape)
            for t in (
                q2,
                kv2,
                bt_o_t,
                idx,
                topk,
                cu_q_t,
                cu_kv_t,
                sinks_use,
                out2,
                lse_flat,
                tiling,
                ws,
                ori_phy,
            )
        ]
        const_args = (
            heads,
            HEAD_DIM,
            num_seqs,
            ori_w,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            1 if want_lse else 0,
            pa_o_bs,
            pa_o_stride,
            pa_o_max,
            g_split,
            has_sinks,
            1,
            kv_layout,
            is_vec,
            phy_w,
        )
        kernel_fn = sparse_flash_mla_ori_sparse_kernel
    else:  # ocs
        ori_idx = ori_sparse_indices.reshape(total_rows, ori_w)
        ori_topk = ori_topk_length.reshape(-1)
        cmp_idx = cmp_sparse_indices.reshape(total_rows, cmp_w)
        cmp_topk = cmp_topk_length.reshape(-1)
        if q_rows_idx is not None:
            ori_idx = ori_idx.index_select(0, q_rows_idx)
            ori_topk = ori_topk.index_select(0, q_rows_idx)
            cmp_idx = cmp_idx.index_select(0, q_rows_idx)
            cmp_topk = cmp_topk.index_select(0, q_rows_idx)
        ori_idx = ori_idx.reshape(-1).to(i32).contiguous()
        ori_topk = ori_topk.to(i32).contiguous()
        cmp_idx = cmp_idx.reshape(-1).to(i32).contiguous()
        cmp_topk = cmp_topk.to(i32).contiguous()
        ori_phy_w = _align128(ori_w) if is_vec == 1 else 1
        cmp_phy_w = _align128(cmp_w) if is_vec == 1 else 1
        ori_phy = torch.zeros(run_rows * ori_phy_w, dtype=torch.int64, device=device)
        cmp_phy = torch.zeros(run_rows * cmp_phy_w, dtype=torch.int64, device=device)
        tensor_args = (
            _tla_tensor(q2),
            _tla_tensor(kv2),
            _tla_tensor(cmp2),
            _tla_tensor(bt_o_t),
            _tla_tensor(bt_c_t),
            _tla_tensor(ori_idx),
            _tla_tensor(ori_topk),
            _tla_tensor(cmp_idx),
            _tla_tensor(cmp_topk),
            _tla_tensor(cu_q_t),
            _tla_tensor(cu_kv_t),
            _tla_tensor(cu_cmp_t),
            _tla_tensor(resid),
            _tla_tensor(sinks_use),
            _tla_tensor(out2),
            _tla_tensor(lse_flat),
            _tla_tensor(tiling),
            _tla_tensor(ws),
            _tla_tensor(ori_phy),
            _tla_tensor(cmp_phy),
        )
        tensor_shapes = [
            tuple(t.shape)
            for t in (
                q2,
                kv2,
                cmp2,
                bt_o_t,
                bt_c_t,
                ori_idx,
                ori_topk,
                cmp_idx,
                cmp_topk,
                cu_q_t,
                cu_kv_t,
                cu_cmp_t,
                resid,
                sinks_use,
                out2,
                lse_flat,
                tiling,
                ws,
                ori_phy,
                cmp_phy,
            )
        ]
        const_args = (
            heads,
            HEAD_DIM,
            num_seqs,
            ori_w,
            cmp_w,
            ori_mask_mode,
            int(ori_win_left),
            int(ori_win_right),
            cmp_mask_mode,
            cmp_ratio,
            pa_o_bs,
            pa_o_stride,
            pa_o_max,
            pa_c_bs,
            pa_c_stride,
            pa_c_max,
            g_split,
            has_sinks,
            1 if want_lse else 0,
            1,
            kv_layout,
            is_vec,
            ori_phy_w,
            cmp_phy_w,
        )
        kernel_fn = sparse_flash_mla_ori_cmp_sparse_kernel

    key = (
        id(kernel_fn),
        str(device),
        str(dtype),
        tuple(tensor_shapes),
        tuple(int(v) for v in const_args),
    )
    executor = _EXECUTOR_CACHE.get(key)
    if executor is None:
        compile_args = tensor_args + (scale_t,) + const_args
        try:
            executor = tla.compile(
                kernel_fn,
                *compile_args,
                options=NPU_ARCH,
            )
        except Exception as exc:  # noqa: BLE001
            if "253952" not in str(exc):
                raise
            executor = tla.compile(
                kernel_fn,
                *compile_args,
                options=NPU_ARCH + UB_RELEASE_FLAGS,
            )
        if len(_EXECUTOR_CACHE) >= _EXECUTOR_CACHE_LIMIT:
            _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE[key] = executor

    if compile_only:
        return output, lse

    def _launch(dequant_pair):
        if combined_preprocess:
            fold_executor(*dequant_pair[0], *fold_args, block_num=fold_grid)
        else:
            if dequant_pair is not None:
                _execute_dequant_pair(dequant_pair, device)
            if fold_executor is not None:
                fold_executor(*fold_args, block_num=fold_grid)
        executor(*tensor_args, scale_t, block_num=block_num)

    def _post():
        if out_pack is None:
            return
        output.zero_()
        output.reshape(total_rows, heads * HEAD_DIM).index_copy_(
            0, q_rows_idx, out_pack
        )
        if want_lse:
            lse.zero_()
            lse.reshape(total_rows, heads).index_copy_(
                0, q_rows_idx, lse_flat.reshape(run_rows, heads)
            )

    if len(_LAUNCH_PLAN) >= _PLAN_LIMIT:
        _LAUNCH_PLAN.clear()
    _LAUNCH_PLAN[sig] = (_launch, _post, output, lse)
    _launch(dequant_pair)
    _post()
    return output, lse


SCALE_GROUPS = 7
NOPE_BYTES = 448
NOPE_OFF = 128
SUPPORTED_QUANT_MODE = 1
METADATA_FA_SIZE = 1024

INPUT_ORDER = (
    "q",
    "quant_mode",
    "ori_kv",
    "cmp_kv",
    "ori_sparse_indices",
    "cmp_sparse_indices",
    "ori_block_table",
    "cmp_block_table",
    "cu_seqlens_q",
    "cu_seqlens_ori_kv",
    "cu_seqlens_cmp_kv",
    "seqused_q",
    "seqused_ori_kv",
    "seqused_cmp_kv",
    "cmp_residual_kv",
    "ori_topk_length",
    "cmp_topk_length",
    "sinks",
    "metadata",
    "rope_head_dim",
    "softmax_scale",
    "cmp_ratio",
    "ori_mask_mode",
    "cmp_mask_mode",
    "ori_win_left",
    "ori_win_right",
    "layout_q",
    "layout_kv",
    "topk_value_mode",
    "return_softmax_lse",
)

_KV_POOL = {}
_KV_POOL_SLOTS = 2
_DEQUANT_EXEC = {}
_DEQUANT_PAGES_EXEC = {}
_PLANE_POOL = {}
_RAW_SCALE = {}
_DERIVED = {}
_OUT_VIEW = {}
_BIND = {}
_BIND_LIMIT = 64
_META_VIEW = {}
_CORE_NUM = {}
_PAGE_CACHE = {}
_PAGES_T = {}
_CPLANE_CACHE = {}

_FUSED_DEQUANT = True
_FUSED_ORI = True
_FUSED_MAX_Q_ROWS = 128
_FUSED_MAX_WIN = 512


def _fused_dequant_ok(
    q,
    ori_kv,
    cmp_kv,
    layout_kv,
    ori_sparse_indices,
    cmp_sparse_indices,
    ori_mask_mode,
    ori_win_left,
    topk_value_mode,
):
    if not _FUSED_DEQUANT:
        return False
    if layout_kv != "PA_BBND" or ori_kv is None or cmp_kv is None:
        return False
    if ori_sparse_indices is not None or cmp_sparse_indices is None:
        return False
    if int(topk_value_mode) != 1:
        return False
    if int(ori_mask_mode) != 4 or not (0 <= int(ori_win_left) <= _FUSED_MAX_WIN):
        return False
    for kv in (ori_kv, cmp_kv):
        if int(kv.shape[-1]) != REC_BYTES:
            return False
        if kv.dtype not in (torch.uint8, torch.int8):
            return False
    n_q = int(q.shape[0]) if q.ndim == 3 else int(q.shape[0]) * int(q.shape[1])
    if not 0 < n_q <= _FUSED_MAX_Q_ROWS:
        return 0
    return 3 if _FUSED_ORI else 1


def _pooled_kv(device, rows, dtype):
    key = (str(device), rows, str(dtype))
    state = _KV_POOL.get(key)
    if state is None:
        state = {"bufs": [], "i": 0}
        _KV_POOL[key] = state
    while len(state["bufs"]) < _KV_POOL_SLOTS:
        state["bufs"].append(torch.empty(rows, HEAD_DIM, dtype=dtype, device=device))
    buf = state["bufs"][state["i"] % _KV_POOL_SLOTS]
    state["i"] += 1
    return buf


def _dequant_blocks(device):
    key = str(device)
    n = _CORE_NUM.get(key)
    if n is None:
        try:
            props = torch.npu.get_device_properties(device)
            n = int(getattr(props, "vector_core_num", 0)) or int(props.cube_core_num)
        except Exception:  # noqa: BLE001
            n = 64
        n = max(1, min(int(n), 64))
        _CORE_NUM[key] = n
    return n


def _blocks_for(rows):
    block = 64
    while block > 1 and rows % block:
        block //= 2
    return block


def _rows_for_pages(bs, n_pages, device):
    cores = _dequant_blocks(device)
    cands = [c for c in (16, 32, 64) if bs % c == 0]
    if not cands:
        best = 1
        while best * 2 <= bs and bs % (best * 2) == 0:
            best *= 2
        return best
    for cand in cands:
        if n_pages * (bs // cand) <= 4 * cores:
            return cand
    return cands[-1]


def _scale_plane(rows, raw_cached, key):
    pkey = (str(rows.device), rows.shape[0])
    plane = _PLANE_POOL.get(pkey)
    if plane is None:
        plane = torch.zeros(
            rows.shape[0], SCALE_STRIDE, dtype=torch.float32, device=rows.device
        )
        _PLANE_POOL[pkey] = plane
    raw = raw_cached
    if raw is None:
        raw = rows[:, 576:590].contiguous().view(torch.bfloat16)
    plane[:, :SCALE_GROUPS] = raw.float()
    return plane, raw


def _bind(tensor, role, element_type=None):
    key = (id(tensor), role)
    hit = _BIND.get(key)
    if hit is not None and hit[0] is tensor:
        return hit[1]
    bound = from_dlpack(tensor, layout_tag=tla.arch.RowMajor, element_type=element_type)
    bound._torch_storage = tensor
    if len(_BIND) >= _BIND_LIMIT:
        _BIND.clear()
    _BIND[key] = (tensor, bound)
    return bound


def _record_views(src_kv):
    key = (id(src_kv), str(src_kv.device), tuple(src_kv.shape))
    hit = _DERIVED.get(key)
    if hit is not None and hit[0] is src_kv:
        return hit[1], hit[4], hit[5], hit[6]
    rows = src_kv.reshape(-1, REC_BYTES)
    nope = rows.view(torch.int8)
    rope = rows.view(torch.bfloat16)
    element_type = tla.Int8 if src_kv.dtype == torch.int8 else tla.Float8E4M3FN
    t_nope = from_dlpack(nope, layout_tag=tla.arch.RowMajor, element_type=element_type)
    t_rope = from_dlpack(rope, layout_tag=tla.arch.RowMajor)
    t_nope._torch_storage = nope
    t_rope._torch_storage = rope
    if len(_DERIVED) >= _BIND_LIMIT:
        _DERIVED.clear()
    entry = (src_kv, rows, nope, rope, t_nope, t_rope, element_type)
    _DERIVED[key] = entry
    return rows, t_nope, t_rope, element_type


def _used_pages(
    src_kv,
    layout_kv,
    seqused,
    block_table,
    ratio=1,
    residual=None,
    side="ori",
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    act_q=0,
    has_sparse=False,
):
    if layout_kv != "PA_BBND" or seqused is None or block_table is None:
        return None
    key = (
        _metadata_cache_key(block_table),
        _metadata_cache_key(seqused),
        _metadata_cache_key(residual),
        tuple(block_table.shape),
        int(ratio),
        side,
        int(mask_mode),
        int(win_left),
        int(win_right),
        int(act_q),
        bool(has_sparse),
    )
    hit = _PAGE_CACHE.get(key)
    if (
        hit is not None
        and hit[0] is block_table
        and hit[1] is seqused
        and hit[2] is residual
    ):
        return hit[3]
    bs = int(src_kv.shape[1])
    acts = [int(v) for v in seqused.reshape(-1).tolist()]
    if residual is not None:
        acts = [
            a + (int(r) + ratio - 1) // ratio
            for a, r in zip(acts, residual.reshape(-1).tolist())
        ]
    narrow = (
        side == "ori"
        and not has_sparse
        and int(mask_mode) == 4
        and int(win_left) >= 0
        and int(act_q) > 0
    )
    pages = []
    seen = set()
    bt_w = int(block_table.shape[1])
    for b, a in enumerate(acts):
        lo, hi = 0, a
        if narrow:
            lo = max(0, a - int(act_q) - int(win_left) - 1)
            if int(win_right) != -1:
                hi = min(a, a + int(win_right))
        if hi <= lo:
            continue
        p0 = lo // bs
        p1 = min((hi - 1) // bs, bt_w - 1)
        if p1 < p0:
            continue
        for phys in block_table[b, p0 : p1 + 1].cpu().tolist():
            if phys >= 0 and phys not in seen:
                seen.add(phys)
                pages.append(int(phys))
    if len(_PAGE_CACHE) >= 64:
        _PAGE_CACHE.clear()
    _PAGE_CACHE[key] = (block_table, seqused, residual, pages)
    return pages


def _pages_tensor(pages, device):
    cap = ((len(pages) + 7) // 8) * 8
    plist = list(pages) + [pages[-1]] * (cap - len(pages))
    key = (str(device), tuple(plist))
    hit = _PAGES_T.get(key)
    if hit is not None:
        return hit
    tensor = torch.tensor(plist, dtype=torch.int32, device=device)
    if len(_PAGES_T) >= 64:
        _PAGES_T.clear()
    _PAGES_T[key] = tensor
    return tensor


def _compact_plane(rows, pages_t, bs):
    dev = rows.device
    ckey = (_metadata_cache_key(rows), id(pages_t))
    hit = _CPLANE_CACHE.get(ckey)
    if hit is not None and hit[0] is rows and hit[1] is pages_t:
        return hit[2]
    n = int(pages_t.numel()) * bs
    # Keep a separate scale plane for each cached record view and page list.
    # Original and compressed KV can have the same padded row count.
    plane = torch.zeros(n, SCALE_STRIDE, dtype=torch.float32, device=dev)
    used = (
        pages_t.to(torch.long).reshape(-1, 1) * bs
        + torch.arange(bs, device=dev).reshape(1, -1)
    ).reshape(-1)
    gathered = rows.index_select(0, used)
    raw = gathered[:, 576:590].contiguous().view(torch.bfloat16)
    plane[:, :SCALE_GROUPS] = raw.float()
    if len(_CPLANE_CACHE) >= 64:
        _CPLANE_CACHE.clear()
    _CPLANE_CACHE[ckey] = (rows, pages_t, plane)
    return plane


def _grid_for(device, n_blocks):
    return max(1, min(_dequant_blocks(device), int(n_blocks)))


def _launch_dequant_pages(
    src_kv, dtype_q, shape, rows, t_nope, t_rope, element_type, pages
):
    dev = rows.device
    bs = int(src_kv.shape[1])
    block = _rows_for_pages(bs, ((len(pages) + 7) // 8) * 8, dev)
    out = _pooled_kv(dev, rows.shape[0], dtype_q)
    pages_t = _pages_tensor(pages, dev)
    t_scale = _bind(_compact_plane(rows, pages_t, bs), "cscale")
    t_pages = _bind(pages_t, "pages")
    t_out = _bind(out, "out")
    cfg = DequantPagesCfg(rows_per_block=block, rows_per_page=bs)
    key = (
        str(dev),
        dtype_q,
        str(element_type),
        rows.shape[0],
        int(pages_t.numel()),
        bs,
        block,
    )
    executor = _DEQUANT_PAGES_EXEC.get(key)
    if executor is None:
        executor = tla.compile(
            dequant_kv_pages,
            t_nope,
            t_rope,
            t_scale,
            t_pages,
            t_out,
            cfg,
            options=NPU_ARCH,
        )
        _DEQUANT_PAGES_EXEC[key] = executor
    executor(
        t_nope,
        t_rope,
        t_scale,
        t_pages,
        t_out,
        block_num=_grid_for(dev, int(pages_t.numel()) * (bs // block)),
    )
    return _out_view(out, shape)


def _prepare_dequant_pair(src_a, pages_a, src_b, pages_b, dtype_q):
    dev = src_a.device
    rows_a, t_nope_a, t_rope_a, et_a = _record_views(src_a)
    rows_b, t_nope_b, t_rope_b, et_b = _record_views(src_b)
    bs_a = int(src_a.shape[1])
    bs_b = int(src_b.shape[1])
    block_a = _rows_for_pages(bs_a, ((len(pages_a) + 7) // 8) * 8, dev)
    block_b = _rows_for_pages(bs_b, ((len(pages_b) + 7) // 8) * 8, dev)
    if (
        dtype_q == torch.bfloat16
        and et_a == et_b == tla.Float8E4M3FN
        and bs_b % 32 == 0
    ):
        block_b = 32
    out_a = _pooled_kv(dev, rows_a.shape[0], dtype_q)
    out_b = _pooled_kv(dev, rows_b.shape[0], dtype_q)
    pt_a = _pages_tensor(pages_a, dev)
    pt_b = _pages_tensor(pages_b, dev)
    t_pages_a = _bind(pt_a, "pages")
    t_pages_b = _bind(pt_b, "pages")
    t_scale_a = _bind(_compact_plane(rows_a, pt_a, bs_a), "cscale")
    t_scale_b = _bind(_compact_plane(rows_b, pt_b, bs_b), "cscale")
    t_out_a = _bind(out_a, "out")
    t_out_b = _bind(out_b, "out")
    cfg = DequantPagesPairCfg(block_a, bs_a, block_b, bs_b, len(pages_a), len(pages_b))
    key = (
        "pair",
        str(dev),
        dtype_q,
        str(et_a),
        str(et_b),
        rows_a.shape[0],
        rows_b.shape[0],
        int(pt_a.numel()),
        int(pt_b.numel()),
        block_a,
        bs_a,
        block_b,
        bs_b,
        len(pages_a),
        len(pages_b),
    )
    gms = (
        t_nope_a,
        t_rope_a,
        t_scale_a,
        t_pages_a,
        t_out_a,
        t_nope_b,
        t_rope_b,
        t_scale_b,
        t_pages_b,
        t_out_b,
    )
    blocks = len(pages_a) * (bs_a // block_a) + len(pages_b) * (bs_b // block_b)
    return (
        _out_view(out_a, src_a.shape),
        _out_view(out_b, src_b.shape),
        (gms, cfg, blocks, key),
    )


def _execute_dequant_pair(plan, device):
    gms, cfg, blocks, key = plan
    executor = _DEQUANT_PAGES_EXEC.get(key)
    if executor is None:
        executor = tla.compile(dequant_kv_pages_pair, *gms, cfg, options=NPU_ARCH)
        _DEQUANT_PAGES_EXEC[key] = executor
    executor(*gms, block_num=_grid_for(device, blocks))


def _launch_dequant(
    src_kv,
    dtype_q,
    layout_kv=None,
    seqused=None,
    block_table=None,
    ratio=1,
    residual=None,
    side="ori",
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    act_q=0,
    has_sparse=False,
):
    shape = src_kv.shape
    rows, t_nope, t_rope, element_type = _record_views(src_kv)
    n_rows = rows.shape[0]
    if n_rows == 0:
        return torch.zeros(*shape[:-1], HEAD_DIM, dtype=dtype_q, device=src_kv.device)
    pages = _used_pages(
        src_kv,
        layout_kv,
        seqused,
        block_table,
        ratio,
        residual,
        side,
        mask_mode,
        win_left,
        win_right,
        act_q,
        has_sparse,
    )
    if pages:
        return _launch_dequant_pages(
            src_kv, dtype_q, shape, rows, t_nope, t_rope, element_type, pages
        )
    out = _pooled_kv(rows.device, n_rows, dtype_q)

    scale_key = (id(src_kv), "raw")
    raw_hit = _RAW_SCALE.get(scale_key)
    raw_cached = (
        raw_hit[1]
        if raw_hit is not None
        and raw_hit[0] is src_kv
        and raw_hit[1].shape[0] == rows.shape[0]
        else None
    )
    scales, raw = _scale_plane(rows, raw_cached, scale_key)
    if raw_cached is None:
        _RAW_SCALE[scale_key] = (src_kv, raw)

    t_scale = _bind(scales, "scale")
    t_out = _bind(out, "out")

    block = _blocks_for(n_rows)
    cfg = DequantCfg(rows_per_block=block)
    key = (str(rows.device), dtype_q, str(element_type), n_rows, block)
    executor = _DEQUANT_EXEC.get(key)
    if executor is None:
        executor = tla.compile(
            dequant_kv_records,
            t_nope,
            t_rope,
            t_scale,
            t_out,
            cfg,
            options=NPU_ARCH,
        )
        _DEQUANT_EXEC[key] = executor
    executor(
        t_nope,
        t_rope,
        t_scale,
        t_out,
        block_num=_grid_for(rows.device, n_rows // block),
    )
    return _out_view(out, shape)


def _out_view(out, shape):
    key = (id(out), tuple(shape))
    hit = _OUT_VIEW.get(key)
    if hit is not None and hit[0] is out:
        return hit[1]
    view = out.reshape(*shape[:-1], HEAD_DIM)
    _OUT_VIEW[key] = (out, view)
    return view


_TOPK_FULL = {}


def _topk_length(indices, given, dtype=torch.int32):
    if given is not None:
        return given
    if indices is None:
        return None
    shape = tuple(indices.shape[:-1])
    key = (str(indices.device), shape, indices.shape[-1])
    cached = _TOPK_FULL.get(key)
    if cached is None:
        cached = torch.full(
            shape, indices.shape[-1], dtype=dtype, device=indices.device
        )
        _TOPK_FULL[key] = cached
    return cached


def _single_query_tile_width(
    q,
    ori_kv,
    cmp_kv,
    ori_indices,
    cmp_indices,
    layout_q,
    layout_kv,
    cu_q,
    used_q,
    cu_ori,
    used_ori,
    cu_cmp,
    used_cmp,
    ori_mask,
    cmp_mask,
    win_left,
    win_right,
):
    """Use one combined tile for a BF16 query over a short PA_BBND window."""
    if (
        layout_q != "TND"
        or layout_kv != "PA_BBND"
        or tuple(q.shape) != (1, 64, HEAD_DIM)
        or q.dtype != torch.bfloat16
        or ori_indices is not None
        or cmp_indices is not None
        or cmp_mask != 0
        or ori_mask not in (3, 4)
        or (ori_mask == 4 and (win_left != KV_BLOCK - 1 or win_right != 0))
    ):
        return 0

    for kv in (ori_kv, cmp_kv):
        if kv is not None and (
            kv.ndim != 4
            or kv.shape[-2:] != (1, REC_BYTES)
            or not kv.shape[0]
            or not kv.shape[1]
            or kv.dtype != torch.uint8
            or not kv.is_contiguous()
        ):
            return 0

    def length(cu, used):
        if cu is not None and cu.ndim == 1 and cu.numel() == 2:
            begin, end = cu.tolist()
            if begin == 0:
                return int(end)
        if cu is None and used is not None and used.ndim == 1 and used.numel() == 1:
            return int(used.item())
        return 0

    if ori_kv is None or length(cu_q, used_q) != 1:
        return 0
    n_ori = length(cu_ori, used_ori)
    if n_ori <= 0 or (ori_mask == 3 and n_ori > KV_BLOCK):
        return 0
    if cmp_kv is None:
        return KV_BLOCK
    n_cmp = length(cu_cmp, used_cmp)
    if not 0 < n_cmp <= 64:
        return 0
    return KV_BLOCK + (n_cmp + 15) // 16 * 16


def _grouped_window_rows(
    q,
    kv,
    ori_sparse_indices,
    cmp_kv,
    cmp_sparse_indices,
    cu_q,
    cu_kv,
    cu_cmp,
    layout_q,
    layout_kv,
    ori_mask,
    cmp_mask,
    win_left,
    win_right,
):
    """Reuse the sliding window and optional dense CMP arm across queries."""
    if (
        layout_q != "TND"
        or layout_kv != "PA_BBND"
        or q.ndim != 3
        or q.shape[1:] != (64, HEAD_DIM)
        or q.dtype != torch.bfloat16
        or ori_sparse_indices is not None
        or cmp_sparse_indices is not None
        or cmp_mask != 0
        or not 1 < q.shape[0] <= 64
        or ori_mask != 4
        or win_left != KV_BLOCK - 1
        or win_right != 0
        or cu_q is None
        or cu_kv is None
        or cu_q.ndim != 1
        or cu_kv.ndim != 1
        or cu_q.numel() < 2
        or cu_kv.numel() != cu_q.numel()
    ):
        return 0
    if (
        kv is None
        or kv.ndim != 4
        or kv.shape[-2:] != (1, REC_BYTES)
        or not kv.shape[0]
        or not kv.shape[1]
        or kv.dtype != torch.uint8
        or not kv.is_contiguous()
    ):
        return 0
    if (cu_q.numel() - 1) * 4 > _cube_cores(q.device):
        return 0

    q_offsets = cu_q.tolist()
    if q_offsets[0] != 0 or q_offsets[-1] != q.shape[0]:
        return 0
    rows = q_offsets[1]
    if rows not in (2, 4, 6) or any(
        right - left != rows for left, right in zip(q_offsets, q_offsets[1:])
    ):
        return 0
    kv_offsets = cu_kv.tolist()
    if kv_offsets[0] != 0 or any(
        right - left < KV_BLOCK + rows - 1
        for left, right in zip(kv_offsets, kv_offsets[1:])
    ):
        return 0
    if cmp_kv is not None:
        if (
            cmp_kv.ndim != 4
            or cmp_kv.shape[-2:] != (1, REC_BYTES)
            or not cmp_kv.shape[0]
            or not cmp_kv.shape[1]
            or cmp_kv.dtype != torch.uint8
            or not cmp_kv.is_contiguous()
            or cu_cmp is None
            or cu_cmp.ndim != 1
            or cu_cmp.numel() != len(q_offsets)
        ):
            return 0
        cmp_offsets = cu_cmp.tolist()
        if cmp_offsets[0] != 0 or not all(
            0 < right - left <= 64 for left, right in zip(cmp_offsets, cmp_offsets[1:])
        ):
            return 0
    return rows


def run(
    q,
    quant_mode,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    rope_head_dim=None,
    softmax_scale=None,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False,
):
    if int(quant_mode) != SUPPORTED_QUANT_MODE:
        raise NotImplementedError(
            "mixed_quant_sparse_flash_mla DSL implements quant_mode=1 only"
        )
    if rope_head_dim is not None and int(rope_head_dim) != ROPE_DIM:
        raise NotImplementedError("only rope_head_dim=64 is implemented")
    if int(topk_value_mode) == 2 and layout_kv == "PA_BBND":
        raise NotImplementedError(
            "topk_value_mode=2 with PA_BBND needs physical->logical remap"
        )

    dtype_q = q.dtype
    if seqused_q is not None:
        act_q_max = max(int(v) for v in seqused_q.reshape(-1).tolist())
    elif cu_seqlens_q is not None:
        cu = [int(v) for v in cu_seqlens_q.reshape(-1).tolist()]
        act_q_max = max((cu[i + 1] - cu[i] for i in range(len(cu) - 1)), default=0)
    else:
        act_q_max = int(q.shape[1]) if q.ndim == 4 else 0
    ori_k = None
    cmp_k = None
    dequant_pair = None
    fused = _fused_dequant_ok(
        q,
        ori_kv,
        cmp_kv,
        layout_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_mask_mode,
        ori_win_left,
        topk_value_mode,
    )
    sparse_short = (
        cmp_kv is not None
        and cmp_sparse_indices is not None
        and cmp_topk_length is not None
        and cmp_sparse_indices.dtype == cmp_topk_length.dtype == torch.int32
        and cmp_sparse_indices.numel() >= 64
        and cmp_topk_length.numel() == 1
        and int(topk_value_mode) == 1
    )
    single_query_kv = _single_query_tile_width(
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        None if sparse_short else cmp_sparse_indices,
        layout_q,
        layout_kv,
        cu_seqlens_q,
        seqused_q,
        cu_seqlens_ori_kv,
        seqused_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_cmp_kv,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
    )
    if sparse_short and single_query_kv > KV_BLOCK + 32:
        single_query_kv = 0
    grouped_window_rows = _grouped_window_rows(
        q,
        ori_kv,
        ori_sparse_indices,
        cmp_kv,
        cmp_sparse_indices,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        layout_q,
        layout_kv,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
    )
    rolling_records = (
        layout_q == "TND"
        and layout_kv == "PA_BBND"
        and q.ndim == 3
        and q.shape[1:] == (Q_ROWS, HEAD_DIM)
        and q.shape[0] >= KV_BLOCK
        and q.dtype == torch.bfloat16
        and ori_kv is not None
        and ori_kv.ndim == 4
        and ori_kv.shape[-2:] == (1, REC_BYTES)
        and ori_kv.dtype == torch.uint8
        and ori_kv.is_contiguous()
        and ori_block_table is not None
        and ori_block_table.ndim == 2
        and ori_block_table.shape[0] == 1
        and ori_kv.shape[1] >= KV_BLOCK
        and ori_kv.shape[1] % KV_BLOCK == 0
        and cmp_kv is None
        and ori_sparse_indices is None
        and cmp_sparse_indices is None
        and (ori_mask_mode, ori_win_left, ori_win_right) == (4, KV_BLOCK - 1, 0)
    )
    if rolling_records:
        _, q_lengths = _build_cu(cu_seqlens_q, seqused_q, 1, None, q.device)
        _, ori_lengths = _build_cu(cu_seqlens_ori_kv, seqused_ori_kv, 1, None, q.device)
        rolling_records = (
            q_lengths == [0, q.shape[0]]
            and len(ori_lengths) == 2
            and ori_lengths[0] == 0
            and ori_lengths[1] >= KV_BLOCK
        )
    if single_query_kv or grouped_window_rows or rolling_records:
        fused = 2 if cmp_kv is None else 3
    fused_ori = (fused & 2) == 2
    fused_cmp = (fused & 1) == 1
    ori_kv_pages = (
        _used_pages(
            ori_kv,
            layout_kv,
            seqused_ori_kv,
            ori_block_table,
            side="ori",
            mask_mode=int(ori_mask_mode),
            win_left=int(ori_win_left),
            win_right=int(ori_win_right),
            act_q=act_q_max,
            has_sparse=ori_sparse_indices is not None,
        )
        if ori_kv is not None and not fused_ori
        else None
    )
    cmp_kv_pages = (
        _used_pages(
            cmp_kv,
            layout_kv,
            seqused_cmp_kv,
            cmp_block_table,
            int(cmp_ratio),
            cmp_residual_kv,
            side="cmp",
        )
        if cmp_kv is not None and not fused_cmp
        else None
    )
    if single_query_kv or grouped_window_rows or rolling_records:
        ori_k, cmp_k = ori_kv, cmp_kv
    elif fused_cmp:
        cmp_k = cmp_kv
        if fused_ori:
            ori_k = ori_kv
        else:
            ori_k = (
                _launch_dequant(
                    ori_kv,
                    dtype_q,
                    layout_kv,
                    seqused_ori_kv,
                    ori_block_table,
                    side="ori",
                    mask_mode=int(ori_mask_mode),
                    win_left=int(ori_win_left),
                    win_right=int(ori_win_right),
                    act_q=act_q_max,
                    has_sparse=ori_sparse_indices is not None,
                )
                if ori_kv is not None
                else None
            )
    elif ori_kv_pages and cmp_kv_pages:
        ori_k, cmp_k, dequant_pair = _prepare_dequant_pair(
            ori_kv, ori_kv_pages, cmp_kv, cmp_kv_pages, dtype_q
        )
    else:
        ori_k = (
            _launch_dequant(
                ori_kv,
                dtype_q,
                layout_kv,
                seqused_ori_kv,
                ori_block_table,
                side="ori",
                mask_mode=int(ori_mask_mode),
                win_left=int(ori_win_left),
                win_right=int(ori_win_right),
                act_q=act_q_max,
                has_sparse=ori_sparse_indices is not None,
            )
            if ori_kv is not None
            else None
        )
        cmp_k = (
            _launch_dequant(
                cmp_kv,
                dtype_q,
                layout_kv,
                seqused_cmp_kv,
                cmp_block_table,
                int(cmp_ratio),
                cmp_residual_kv,
                side="cmp",
            )
            if cmp_kv is not None
            else None
        )

    meta = metadata
    if meta is None:
        meta = torch.zeros(METADATA_FA_SIZE, dtype=torch.int32, device=q.device)
    else:
        key = id(metadata)
        hit = _META_VIEW.get(key)
        if hit is not None and hit[0] is metadata:
            meta = hit[1]
        else:
            meta = metadata.reshape(-1)[:METADATA_FA_SIZE]
            if not meta.is_contiguous():
                meta = meta.contiguous()
            _META_VIEW[key] = (metadata, meta)

    scale = 512**-0.5 if softmax_scale is None else float(softmax_scale)

    return attention_run(
        q,
        ori_k,
        cmp_k,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        _topk_length(ori_sparse_indices, ori_topk_length),
        _topk_length(cmp_sparse_indices, cmp_topk_length),
        sinks,
        meta,
        scale,
        int(cmp_ratio),
        int(ori_mask_mode),
        int(cmp_mask_mode),
        int(ori_win_left),
        int(ori_win_right),
        layout_q,
        layout_kv,
        bool(return_softmax_lse),
        fused_dequant=fused,
        single_query_kv=single_query_kv,
        grouped_window_rows=grouped_window_rows,
        rolling_records=rolling_records,
        dequant_pair=dequant_pair,
    )
