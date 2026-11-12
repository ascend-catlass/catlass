# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Transposed QK: K rows in L0A, resident Q columns in L0B, SPLIT_N to UB.

The caller executes these stages inside ``tla.cube()`` and owns the paired
score ready/free flags. Each AIV receives half of the logical Query columns
at physical UB row stride ``UB_SCORE_STRIDE``, including short Query tiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import catlass.tla as tla
from catlass.base_dsl.typing import Pointer
from catlass.types import TlaMutex
from helpers import TilingParams


TILING = TilingParams(l1_tm=224, l1_tn=256, l1_tk=256, l0_tm=224, l0_tn=256, l0_tk=64)
M1 = TILING.l1_tm
N1 = TILING.l1_tn
K1 = TILING.l1_tk
K0 = TILING.l0_tk
C_SLICE_ROWS = M1 // 2
UB_SCORE_STRIDE = N1 // 2
WORKSPACE_ROW_STRIDE = N1
QK_ALIGNMENT_BYTES = 512
# SPLIT_N requires each AIV Query half to align to 16 columns.
QUERY_HALF_ALIGNMENT_TOKENS = 16
QUERY_ALIGNMENT_TOKENS = 2 * QUERY_HALF_ALIGNMENT_TOKENS


@dataclass
class QkBuffers:
    q0: Pointer
    q1: Pointer
    k0: Pointer
    k1: Pointer
    a0: Pointer
    a1: Pointer
    b0: Pointer
    b1: Pointer
    c: Pointer
    c1: Pointer
    mc0: TlaMutex
    mc1: TlaMutex
    mq0: TlaMutex
    mq1: TlaMutex
    mk0: TlaMutex
    mk1: TlaMutex
    ma0: TlaMutex
    ma1: TlaMutex
    mb0: TlaMutex
    mb1: TlaMutex


@tla.jit
def allocate_qk():
    """Default tiling: 480 KiB L1, 56/64 KiB L0A/B, two 112 KiB L0C slices.

    mq protects Q in L1 (nZ); mk protects K in L1 (zN). ma protects
    streamed K in L0A; mb protects Q in L0B; mc transfers each L0C
    slice between CUBE and FIX. Each L1→L0 operation locks its L1 mutex
    before its L0 mutex. MMAD locks ma before mb.
    """
    return QkBuffers(
        tla.allocate(K1 * N1, tla.BFloat16, tla.AddressSpace.l1, QK_ALIGNMENT_BYTES),
        tla.allocate(K1 * N1, tla.BFloat16, tla.AddressSpace.l1, QK_ALIGNMENT_BYTES),
        tla.allocate(M1 * K1, tla.BFloat16, tla.AddressSpace.l1, QK_ALIGNMENT_BYTES),
        tla.allocate(M1 * K1, tla.BFloat16, tla.AddressSpace.l1, QK_ALIGNMENT_BYTES),
        tla.allocate(M1 * K0, tla.BFloat16, tla.AddressSpace.l0a, QK_ALIGNMENT_BYTES),
        tla.allocate(M1 * K0, tla.BFloat16, tla.AddressSpace.l0a, QK_ALIGNMENT_BYTES),
        tla.allocate(K0 * N1, tla.BFloat16, tla.AddressSpace.l0b, QK_ALIGNMENT_BYTES),
        tla.allocate(K0 * N1, tla.BFloat16, tla.AddressSpace.l0b, QK_ALIGNMENT_BYTES),
        tla.allocate(
            C_SLICE_ROWS * N1, tla.Float32, tla.AddressSpace.l0c, QK_ALIGNMENT_BYTES
        ),
        tla.allocate(
            C_SLICE_ROWS * N1, tla.Float32, tla.AddressSpace.l0c, QK_ALIGNMENT_BYTES
        ),
        tla.mutex(resource="block_mean_attention_score_l0c_0", id=8),
        tla.mutex(resource="block_mean_attention_score_l0c_1", id=9),
        tla.mutex(resource="block_mean_attention_score_q_l1_0", id=0),
        tla.mutex(resource="block_mean_attention_score_q_l1_1", id=1),
        tla.mutex(resource="block_mean_attention_score_k_l1_0", id=2),
        tla.mutex(resource="block_mean_attention_score_k_l1_1", id=3),
        tla.mutex(resource="block_mean_attention_score_k_l0a_0", id=4),
        tla.mutex(resource="block_mean_attention_score_k_l0a_1", id=5),
        tla.mutex(resource="block_mean_attention_score_q_l0b_0", id=6),
        tla.mutex(resource="block_mean_attention_score_q_l0b_1", id=7),
    )


@tla.jit
def _load_query(
    q, q_ptr, mutex, head, query_start, nq, nq_pad, s, d, feature_start, dk
):
    """Read Qᵀ directly from [S,D]; duplicate its first block for odd tails."""
    q_gm = tla.make_tensor(
        q.ptr + (head * s + query_start) * d + feature_start,
        tla.make_layout(
            tla.make_shape(dk, nq),
            tla.make_stride(1, d),
            layoutTag=tla.arch.ColumnMajor,
        ),
    )
    q_l1 = tla.make_tensor(
        q_ptr, tla.make_layout_with_tag((dk, nq_pad), tla.BFloat16, tla.arch.nZ)
    )
    q_real = tla.tile_view(q_l1, tla.make_shape(dk, nq), tla.make_coord(0, 0))
    mutex.lock(pipe=tla.arch.MTE2)
    tla.copy(q_real, q_gm)
    if nq_pad > nq:
        q_duplicate = tla.tile_view(
            q_gm, tla.make_shape(dk, QUERY_HALF_ALIGNMENT_TOKENS), tla.make_coord(0, 0)
        )
        q_padding = tla.tile_view(
            q_l1,
            tla.make_shape(dk, QUERY_HALF_ALIGNMENT_TOKENS),
            tla.make_coord(0, nq // QUERY_HALF_ALIGNMENT_TOKENS),
        )
        tla.copy(q_padding, q_duplicate)
    mutex.unlock(pipe=tla.arch.MTE2)


@tla.jit
def prepare_query(q, bufs, head, query_start, nq, nq_pad, s, d):
    """Keep Qᵀ in L1 for D<=K1, and in both L0B buffers for D<=2*K0."""
    if tla.const_expr(d <= K1):
        _load_query(q, bufs.q0, bufs.mq0, head, query_start, nq, nq_pad, s, d, 0, d)
        if tla.const_expr(d <= 2 * K0):
            mq = bufs.mq0
            q_l1 = tla.make_tensor(
                bufs.q0,
                tla.make_layout_with_tag((d, nq_pad), tla.BFloat16, tla.arch.nZ),
            )
            for part in tla.range(0, (d + K0 - 1) // K0, 1):
                q_part = tla.tile_view(
                    q_l1, tla.make_shape(K0, nq_pad), tla.make_coord(part, 0)
                )
                q_l0 = tla.make_tensor_like(
                    bufs.b0 if part == 0 else bufs.b1, q_part, tla.arch.nZ
                )
                mb = bufs.mb0 if part == 0 else bufs.mb1
                mq.lock(pipe=tla.arch.MTE1)
                mb.lock(pipe=tla.arch.MTE1)
                tla.copy(q_l0, q_part)
                mb.unlock(pipe=tla.arch.MTE1)
                mq.unlock(pipe=tla.arch.MTE1)


@tla.jit
def produce_qk(
    q, k, bufs, head, query_start, key_start, nq, nq_pad, mk, s, d, tile_serial
):
    """Accumulate one Key slice while FIX can drain the other L0C slot."""
    cptr = bufs.c if tile_serial % 2 == 0 else bufs.c1
    mutex_c = bufs.mc0 if tile_serial % 2 == 0 else bufs.mc1
    mutex_c.lock(pipe=tla.arch.CUBE)
    score = tla.make_tensor(
        cptr, tla.make_layout_with_tag((mk, nq_pad), tla.Float32, tla.arch.L0Clayout)
    )
    for feature_tile in tla.range(0, (d + K1 - 1) // K1, 1):
        feature_start = feature_tile * K1
        dk = d - feature_start if d - feature_start < K1 else K1
        l1_index = (tile_serial * ((d + K1 - 1) // K1) + feature_tile) % 2
        k_ptr = bufs.k0 if l1_index == 0 else bufs.k1
        mutex_k = bufs.mk0 if l1_index == 0 else bufs.mk1
        q_ptr = bufs.q0
        mutex_q = bufs.mq0
        if tla.const_expr(d > K1):
            q_ptr = bufs.q0 if l1_index == 0 else bufs.q1
            mutex_q = bufs.mq0 if l1_index == 0 else bufs.mq1
            _load_query(
                q,
                q_ptr,
                mutex_q,
                head,
                query_start,
                nq,
                nq_pad,
                s,
                d,
                feature_start,
                dk,
            )
        q_l1 = tla.make_tensor(
            q_ptr, tla.make_layout_with_tag((dk, nq_pad), tla.BFloat16, tla.arch.nZ)
        )
        k_gm = tla.make_tensor(
            k.ptr + (head * s + key_start) * d + feature_start,
            tla.make_layout(tla.make_shape(mk, dk), tla.make_stride(d, 1)),
        )
        k_l1 = tla.make_tensor_like(k_ptr, k_gm, tla.arch.zN)
        mutex_k.lock(pipe=tla.arch.MTE2)
        tla.copy(k_l1, k_gm)
        mutex_k.unlock(pipe=tla.arch.MTE2)

        for part in tla.range(0, (dk + K0 - 1) // K0, 1):
            l0_index = (
                tile_serial * ((d + K0 - 1) // K0) + feature_tile * (K1 // K0) + part
            ) % 2
            k_part = tla.tile_view(
                k_l1, tla.make_shape(mk, K0), tla.make_coord(0, part)
            )
            q_part = tla.tile_view(
                q_l1, tla.make_shape(K0, nq_pad), tla.make_coord(part, 0)
            )
            mb = bufs.mb0 if l0_index == 0 else bufs.mb1
            b_ptr = bufs.b0 if l0_index == 0 else bufs.b1
            if tla.const_expr(d <= 2 * K0):
                mb = bufs.mb0 if part == 0 else bufs.mb1
                b_ptr = bufs.b0 if part == 0 else bufs.b1
            q_l0 = tla.make_tensor_like(b_ptr, q_part, tla.arch.nZ)
            if tla.const_expr(d > 2 * K0):
                mutex_q.lock(pipe=tla.arch.MTE1)
                mb.lock(pipe=tla.arch.MTE1)
                tla.copy(q_l0, q_part)
                mb.unlock(pipe=tla.arch.MTE1)
                mutex_q.unlock(pipe=tla.arch.MTE1)

            ma = bufs.ma0 if l0_index == 0 else bufs.ma1
            k_l0 = tla.make_tensor_like(
                bufs.a0 if l0_index == 0 else bufs.a1, k_part, tla.arch.zN
            )
            mutex_k.lock(pipe=tla.arch.MTE1)
            ma.lock(pipe=tla.arch.MTE1)
            tla.copy(k_l0, k_part)
            ma.unlock(pipe=tla.arch.MTE1)
            mutex_k.unlock(pipe=tla.arch.MTE1)
            ma.lock(pipe=tla.arch.CUBE)
            mb.lock(pipe=tla.arch.CUBE)
            init_c = True if feature_tile == 0 and part == 0 else False
            tla.mmad(score, k_l0, q_l0, init_c=init_c, unit_flag=0)
            mb.unlock(pipe=tla.arch.CUBE)
            ma.unlock(pipe=tla.arch.CUBE)
    mutex_c.unlock(pipe=tla.arch.CUBE)


@tla.jit
def copy_scores(bufs, score_ub_ptr, mk, nq_pad, tile_serial):
    """Consume one L0C slice under FIX ownership, then release its slot."""
    cptr = bufs.c if tile_serial % 2 == 0 else bufs.c1
    mutex_c = bufs.mc0 if tile_serial % 2 == 0 else bufs.mc1
    mutex_c.lock(pipe=tla.arch.FIX)
    score = tla.make_tensor(
        cptr, tla.make_layout_with_tag((mk, nq_pad), tla.Float32, tla.arch.L0Clayout)
    )
    score_ub = tla.make_tensor(
        score_ub_ptr,
        tla.make_layout(
            tla.make_shape(mk, nq_pad), tla.make_stride(UB_SCORE_STRIDE, 1)
        ),
    )
    tla.copy(
        score_ub,
        score,
        tla.params.CopyL0C2DstParams(
            unit_flag=0, l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N
        ),
    )
    mutex_c.unlock(pipe=tla.arch.FIX)
