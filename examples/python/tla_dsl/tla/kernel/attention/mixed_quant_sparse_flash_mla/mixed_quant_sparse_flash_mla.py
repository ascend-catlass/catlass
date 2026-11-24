# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Mixed-quant sparse flash MLA forward, CATLASS DSL (device + host).

Each KV token is a 608-byte quantised record: 64 RoPE values in bf16, 448 FP8
latent values with seven per-token group scales, and 7 padding bytes. The kernel
dequantises the records it reads, then runs sparse MLA over two column sets: a
dense causal window on the original KV and a top-k selection on the compressed
KV.

.. math::

    S_{r,j} = \frac{Q_r \cdot K_j}{\sqrt{512}}, \qquad
    O_r = \sum_j \operatorname{softmax}(S)_{r,j} \cdot V_j

Five shapes share one tiling: ``SWA`` (sliding window), ``HCA`` (window plus all
compressed keys), ``CSA`` (window plus top-k compressed keys), ``ORI_SPARSE`` and
``ORI_CMP_SPARSE`` (explicit index lists). KV supports ``BSND`` / ``TND`` /
``PA_BBND`` layouts. The public entry point is ``run(...)`` at the end of the
file.

Supported product: Ascend 950PR / 950DT.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
import sys

import torch
import torch_npu

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


def static_range(start, stop=None):
    if stop is None:
        return tuple(builtins.range(int(start)))
    return tuple(builtins.range(int(start), int(stop)))


_BLOCK_SYNC_AIC_SOURCE = r"""
#include "kernel_operator.h"

extern "C" {
[aicore] __attribute__((always_inline)) void tla_sync_all_cores_aic() {
  AscendC::SyncAll<false>();
}
}
"""

_BLOCK_SYNC_AIV_SOURCE = r"""
#include "kernel_operator.h"

extern "C" {
[aicore] __attribute__((always_inline)) void tla_sync_all_cores_aiv() {
  AscendC::SyncAll<false>();
}
}
"""


@tla.extern(source=_BLOCK_SYNC_AIC_SOURCE, name="tla_sync_all_cores_aic")
def sync_all_cores_aic() -> None: ...


@tla.extern(source=_BLOCK_SYNC_AIV_SOURCE, name="tla_sync_all_cores_aiv")
def sync_all_cores_aiv() -> None: ...


HEAD_DIM = 512
Q_ROWS = 64
KV_BLOCK = 128
D_CHUNK = 128
Q_HALF = 256
KV_SLOTS = 3
P_SLOTS = 2

KV_UNIT = 16

IDX_RING_SLOTS = 2
IDX_RING_STRIDE = 72

MIN_F = -3.4028234663852886e38

INIT_POP_ELE = 512


@dataclass
class Sync:
    s_ready: object
    s_free: object
    p_ready: object
    o_ready: object
    o_free: object
    kv_stage_ready: object

    m_q: object
    m_s: object
    m_s1: object
    m_o0: object
    m_kv0: object
    m_kv1: object
    m_k0: object
    m_k1: object
    m_k2: object
    m_lse: object
    m_init: object

    m_idx0: object
    m_idx1: object

    bc_m3m2: object
    bc_m3v: object
    bc_m2v: object
    bc_v2m: object
    bc_v3m: object

    p_free: object

    m_o16: object
    m_p: object
    m_sink: object

    m_ab0: object  # L0A/L0B slot 0: MTE1 <-> CUBE
    m_ab1: object  # L0A/L0B slot 1: MTE1 <-> CUBE

    s2_mte2: object

    s2_rendezvous: object


@dataclass
class Bufs:
    l1_p: object

    l1_q: object

    l1_k: object
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
    ub_init_lse: object

    ub_idx_ring: object
    ub_bc_prev: object
    ub_bc_max: object
    ub_bc_sum: object
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
        s2_rendezvous=(
            cross("smla_s2_rendezvous", mode=0) if need_rendezvous else None
        ),
        m_q=tla.mutex("smla_m_q", id=0),
        m_s=tla.mutex("smla_m_s", id=7),
        m_s1=tla.mutex("smla_m_s1", id=8),
        m_o0=tla.mutex("smla_m_o0", id=13),
        m_kv0=tla.mutex("smla_m_kv0", id=15),
        m_kv1=tla.mutex("smla_m_kv1", id=16),
        m_k0=tla.mutex("smla_m_k0", id=9),
        m_k1=tla.mutex("smla_m_k1", id=10),
        m_k2=tla.mutex("smla_m_k2", id=11),
        m_lse=tla.mutex("smla_m_lse", id=12),
        m_init=tla.mutex("smla_m_init", id=14),
        m_idx0=(tla.mutex("smla_m_idx0", id=26) if need_idx_ring else None),
        m_idx1=(tla.mutex("smla_m_idx1", id=27) if need_idx_ring else None),
        bc_m3m2=tla.flag("smla_bc_m3m2", tla.arch.MTE3, tla.arch.MTE2, id=6),
        bc_m3v=tla.flag("smla_bc_m3v", tla.arch.MTE3, tla.arch.VECTOR, id=5),
        bc_m2v=tla.flag("smla_bc_m2v", tla.arch.MTE2, tla.arch.VECTOR, id=3),
        bc_v2m=tla.flag("smla_bc_v2m", tla.arch.VECTOR, tla.arch.MTE2, id=7),
        p_free=cross("smla_p_free"),
        bc_v3m=tla.flag("smla_bc_v3m", tla.arch.VECTOR, tla.arch.MTE3, id=6),
        m_ab0=tla.mutex("smla_m_ab0", id=1),
        m_ab1=tla.mutex("smla_m_ab1", id=2),
        s2_mte2=cross("smla_s2_mte2", mode=0),
        m_o16=tla.mutex("smla_o16", id=4),
        m_p=tla.mutex("smla_p", id=5),
        m_sink=tla.mutex("smla_sink", id=6),
    )


@tla.jit
def init_sync(
    sync: Sync,
    defer_ofree: bool = False,
    pre_bc: bool = False,
) -> None:
    with tla.vector():
        if pre_bc:
            tla.set_flag(sync.bc_v2m)
        tla.cross_core_set_flag(sync.s_free, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(sync.s_free, tla.arch.VECTOR, aiv_id=1)
        if not defer_ofree:
            tla.cross_core_set_flag(sync.o_free, tla.arch.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(sync.o_free, tla.arch.VECTOR, aiv_id=1)


@tla.jit
def init_sync_preload(sync: Sync) -> None:
    with tla.vector():
        tla.cross_core_set_flag(sync.s_free, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(sync.s_free, tla.arch.VECTOR, aiv_id=1)


@tla.jit
def alloc_buffers(
    dtype_in,
    row_cap=None,
    sink_cap=None,
    bc=0,
    preload=False,
    idx_ring=False,
) -> Bufs:
    # ---- L1: l1P[2] 32,768 | l1Q 3x32,768 | l1K(V) 3x131,072 = 524,288 B ----

    l1_p = tla.allocate(
        Q_ROWS * KV_BLOCK * (2 if preload else 1), dtype_in, tla.AddressSpace.l1, 512
    )

    l1_q = tla.allocate(3 * Q_ROWS * Q_HALF, dtype_in, tla.AddressSpace.l1, 512)
    l1_k = tla.allocate(
        KV_SLOTS * HEAD_DIM * KV_BLOCK, dtype_in, tla.AddressSpace.l1, 512
    )
    # ---- L0A: 32,768 B, Q tile + P tiles ----
    l0a0 = tla.allocate(Q_ROWS * D_CHUNK, dtype_in, tla.AddressSpace.l0a, 512)
    l0a1 = tla.allocate(Q_ROWS * D_CHUNK, dtype_in, tla.AddressSpace.l0a, 512)
    # ---- L0B: 65,536 B, K and V tiles ----
    l0b0 = tla.allocate(D_CHUNK * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
    l0b1 = tla.allocate(D_CHUNK * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
    # ---- L0C: 262,144 B, S and O halves ----

    l0c_s = tla.allocate(
        Q_ROWS * KV_BLOCK * (2 if preload else 1),
        tla.Float32,
        tla.AddressSpace.l0c,
        1024,
    )

    l0c_o = tla.allocate(Q_ROWS * HEAD_DIM, tla.Float32, tla.AddressSpace.l0c, 1024)
    # ---- UB per AIV ----
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
        half * (2 if preload else 1), tla.Float32, tla.AddressSpace.ub, 256
    )

    ub_sum = tla.allocate(
        half * (2 if preload else 1), tla.Float32, tla.AddressSpace.ub, 256
    )

    ub_dm = tla.allocate(
        half * (2 if preload else 1), tla.Float32, tla.AddressSpace.ub, 256
    )

    ub_kv0 = tla.allocate(
        KV_UNIT * HEAD_DIM,
        dtype_in,
        tla.AddressSpace.ub,
        256,
    )
    ub_kv1 = tla.allocate(KV_UNIT * HEAD_DIM, dtype_in, tla.AddressSpace.ub, 256)

    ub_init = tla.allocate(INIT_POP_ELE, dtype_in, tla.AddressSpace.ub, 256)
    ub_init_lse = tla.allocate(INIT_POP_ELE, tla.Float32, tla.AddressSpace.ub, 256)

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


def _cast_slot(slot):
    return tla.params.CastParams(
        reg_slot=slot,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )


def _emit_group(ub_s, ub_o, row, group, val, preg, preg_b16, cast_pack, pack_store):
    brc = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
    scale = tla.get_tile(ub_s, tla.make_coord(row, group), tla.make_shape(1, 1)).load(
        params=brc
    )
    scaled = tla.mul(val, scale, mask=preg)
    tla.get_tile(
        ub_o, tla.make_coord(row, group * GROUP_SIZE), tla.make_shape(1, GROUP_SIZE)
    ).store(scaled.to(tla.BFloat16, cast_pack, mask=preg), pack_store, mask=preg_b16)


@tla.kernel
def dequant_kv_records(
    gm_nope: tla.Tensor,
    gm_rope: tla.Tensor,
    gm_scale: tla.Tensor,
    gm_out: tla.Tensor,
    _cfg: tla.Constexpr[DequantCfg],
) -> None:
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
        tla.allocate(rows * HEAD_DIM, tla.BFloat16, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)),
    )

    c0 = 0
    with tla.vector():
        for blk in tla.range(tla.arch.block_idx(), n_blocks, tla.arch.block_num()):
            r0 = blk * rows

            m_nope.lock(pipe=tla.arch.MTE2)
            tla.copy(
                ub_n,
                tla.get_tile(
                    gm_nope,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, REC_BYTES),
                ),
            )
            m_nope.unlock(pipe=tla.arch.MTE2)
            m_rope.lock(pipe=tla.arch.MTE2)
            tla.copy(
                ub_r,
                tla.get_tile(
                    gm_rope,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, ROPE_DIM),
                ),
            )
            m_rope.unlock(pipe=tla.arch.MTE2)
            m_scale.lock(pipe=tla.arch.MTE2)
            tla.copy(
                ub_s,
                tla.get_tile(
                    gm_scale,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, SCALE_STRIDE),
                ),
            )
            m_scale.unlock(pipe=tla.arch.MTE2)

            m_nope.lock(pipe=tla.arch.VECTOR)
            m_rope.lock(pipe=tla.arch.VECTOR)
            m_scale.lock(pipe=tla.arch.VECTOR)
            m_out.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                m8 = tla.create_mask(pattern=tla.mask.ALL, dtype=gm_nope.ptr.dtype)
                preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                preg_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.BFloat16)
                cast0 = _cast_slot(tla.params.RegSlot.ZERO)
                cast1 = _cast_slot(tla.params.RegSlot.ONE)
                cast2 = _cast_slot(tla.params.RegSlot.TWO)
                cast3 = _cast_slot(tla.params.RegSlot.THREE)
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
                            preg_b16,
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
                            preg_b16,
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
                            preg_b16,
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
                                preg_b16,
                                cast_pack,
                                pack_store,
                            )

                    rreg = tla.get_tile(
                        ub_r, tla.make_coord(r, c0), tla.make_shape(1, ROPE_DIM)
                    ).load()
                    tla.get_tile(
                        ub_o,
                        tla.make_coord(r, SCALE_GROUPS * GROUP_SIZE),
                        tla.make_shape(1, ROPE_DIM),
                    ).store(rreg)
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
            m_out.unlock(pipe=tla.arch.VECTOR)
            m_rope.unlock(pipe=tla.arch.VECTOR)
            m_scale.unlock(pipe=tla.arch.VECTOR)
            m_nope.unlock(pipe=tla.arch.VECTOR)

            m_out.lock(pipe=tla.arch.MTE3)
            tla.copy(
                tla.get_tile(
                    gm_out,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, HEAD_DIM),
                ),
                ub_o,
            )
            m_out.unlock(pipe=tla.arch.MTE3)
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
        tla.allocate(rows * HEAD_DIM, tla.BFloat16, tla.AddressSpace.ub, 256),
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

            m_nope.lock(pipe=tla.arch.MTE2)
            tla.copy(
                ub_n,
                tla.get_tile(
                    gm_nope,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, REC_BYTES),
                ),
            )
            m_nope.unlock(pipe=tla.arch.MTE2)
            m_rope.lock(pipe=tla.arch.MTE2)
            tla.copy(
                ub_r,
                tla.get_tile(
                    gm_rope,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, ROPE_DIM),
                ),
            )
            m_rope.unlock(pipe=tla.arch.MTE2)
            m_scale.lock(pipe=tla.arch.MTE2)
            tla.copy(
                ub_s,
                tla.get_tile(
                    gm_scale,
                    tla.make_coord(rs, c0),
                    tla.make_shape(rows, SCALE_STRIDE),
                ),
            )
            m_scale.unlock(pipe=tla.arch.MTE2)

            m_nope.lock(pipe=tla.arch.VECTOR)
            m_rope.lock(pipe=tla.arch.VECTOR)
            m_scale.lock(pipe=tla.arch.VECTOR)
            m_out.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                m8 = tla.create_mask(pattern=tla.mask.ALL, dtype=gm_nope.ptr.dtype)
                preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                preg_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.BFloat16)
                cast0 = _cast_slot(tla.params.RegSlot.ZERO)
                cast1 = _cast_slot(tla.params.RegSlot.ONE)
                cast2 = _cast_slot(tla.params.RegSlot.TWO)
                cast3 = _cast_slot(tla.params.RegSlot.THREE)
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
                            preg_b16,
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
                            preg_b16,
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
                            preg_b16,
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
                                preg_b16,
                                cast_pack,
                                pack_store,
                            )
                    rreg = tla.get_tile(
                        ub_r, tla.make_coord(r, c0), tla.make_shape(1, ROPE_DIM)
                    ).load()
                    tla.get_tile(
                        ub_o,
                        tla.make_coord(r, SCALE_GROUPS * GROUP_SIZE),
                        tla.make_shape(1, ROPE_DIM),
                    ).store(rreg)
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
            m_out.unlock(pipe=tla.arch.VECTOR)
            m_rope.unlock(pipe=tla.arch.VECTOR)
            m_scale.unlock(pipe=tla.arch.VECTOR)
            m_nope.unlock(pipe=tla.arch.VECTOR)

            m_out.lock(pipe=tla.arch.MTE3)
            tla.copy(
                tla.get_tile(
                    gm_out,
                    tla.make_coord(r0, c0),
                    tla.make_shape(rows, HEAD_DIM),
                ),
                ub_o,
            )
            m_out.unlock(pipe=tla.arch.MTE3)
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
):
    c0 = 0
    m_nope.lock(pipe=tla.arch.MTE2)
    tla.copy(
        ub_n,
        tla.get_tile(gm_nope, tla.make_coord(r0, c0), tla.make_shape(rows, REC_BYTES)),
    )
    m_nope.unlock(pipe=tla.arch.MTE2)
    m_rope.lock(pipe=tla.arch.MTE2)
    tla.copy(
        ub_r,
        tla.get_tile(gm_rope, tla.make_coord(r0, c0), tla.make_shape(rows, ROPE_DIM)),
    )
    m_rope.unlock(pipe=tla.arch.MTE2)
    m_scale.lock(pipe=tla.arch.MTE2)
    tla.copy(
        ub_s,
        tla.get_tile(
            gm_scale, tla.make_coord(rs, c0), tla.make_shape(rows, SCALE_STRIDE)
        ),
    )
    m_scale.unlock(pipe=tla.arch.MTE2)

    m_nope.lock(pipe=tla.arch.VECTOR)
    m_rope.lock(pipe=tla.arch.VECTOR)
    m_scale.lock(pipe=tla.arch.VECTOR)
    m_out.lock(pipe=tla.arch.VECTOR)
    with tla.vec.func(mode="simd"):
        m8 = tla.create_mask(pattern=tla.mask.ALL, dtype=gm_nope.ptr.dtype)
        preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        preg_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.BFloat16)
        cast0 = _cast_slot(tla.params.RegSlot.ZERO)
        cast1 = _cast_slot(tla.params.RegSlot.ONE)
        cast2 = _cast_slot(tla.params.RegSlot.TWO)
        cast3 = _cast_slot(tla.params.RegSlot.THREE)
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
                    ub_s, ub_o, r, group, c00, preg, preg_b16, cast_pack, pack_store
                )
                _emit_group(
                    ub_s, ub_o, r, group + 1, c01, preg, preg_b16, cast_pack, pack_store
                )
                _emit_group(
                    ub_s, ub_o, r, group + 2, c10, preg, preg_b16, cast_pack, pack_store
                )
                if part == 0:
                    _emit_group(
                        ub_s,
                        ub_o,
                        r,
                        group + 3,
                        c11,
                        preg,
                        preg_b16,
                        cast_pack,
                        pack_store,
                    )
            rreg = tla.get_tile(
                ub_r, tla.make_coord(r, c0), tla.make_shape(1, ROPE_DIM)
            ).load()
            tla.get_tile(
                ub_o,
                tla.make_coord(r, SCALE_GROUPS * GROUP_SIZE),
                tla.make_shape(1, ROPE_DIM),
            ).store(rreg)
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    m_out.unlock(pipe=tla.arch.VECTOR)
    m_rope.unlock(pipe=tla.arch.VECTOR)
    m_scale.unlock(pipe=tla.arch.VECTOR)
    m_nope.unlock(pipe=tla.arch.VECTOR)

    m_out.lock(pipe=tla.arch.MTE3)
    tla.copy(
        tla.get_tile(gm_out, tla.make_coord(r0, c0), tla.make_shape(rows, HEAD_DIM)),
        ub_o,
    )
    m_out.unlock(pipe=tla.arch.MTE3)


def _dv_buffers(rows, dtype_in):
    ub_n = tla.make_tensor(
        tla.allocate(rows * REC_BYTES, dtype_in, tla.AddressSpace.ub, 256),
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
        tla.allocate(rows * HEAD_DIM, tla.BFloat16, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)),
    )
    return ub_n, ub_r, ub_s, ub_o


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
    rows_a = _cfg.rows_per_block_a
    rpp_a = _cfg.rows_per_page_a
    n_sub_a = rpp_a // rows_a
    total_a = gm_pages_a.origin_shape[0] * n_sub_a
    rows_b = _cfg.rows_per_block_b
    rpp_b = _cfg.rows_per_page_b
    n_sub_b = rpp_b // rows_b
    total_b = gm_pages_b.origin_shape[0] * n_sub_b
    total = total_a + total_b
    dtype_a = gm_nope_a.ptr.dtype
    dtype_b = gm_nope_b.ptr.dtype

    m_nope = tla.mutex("dq_nope", id=0)
    m_rope = tla.mutex("dq_rope", id=1)
    m_scale = tla.mutex("dq_scale", id=2)
    m_out = tla.mutex("dq_out", id=3)

    ub_na, ub_ra, ub_sa, ub_oa = _dv_buffers(rows_a, dtype_a)
    ub_nb, ub_rb, ub_sb, ub_ob = _dv_buffers(rows_b, dtype_b)

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
                )
        tla.pipe_barrier(tla.pipes.ALL)


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
    m_q = sync.m_q
    c0 = 0
    base = task * m if row_base is None else row_base
    with tla.cube():
        m_q.lock(pipe=tla.arch.MTE2)
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
        m_q.unlock(pipe=tla.arch.MTE2)


@tla.jit
def _k_slot_wait(k0, k1, k2, slot):
    if slot == 0:
        k0.lock(pipe=tla.arch.MTE2)
    if slot == 1:
        k1.lock(pipe=tla.arch.MTE2)
    if slot == 2:
        k2.lock(pipe=tla.arch.MTE2)


@tla.jit
def _k_slot_free(k0, k1, k2, slot):
    if slot == 0:
        k0.unlock(pipe=tla.arch.MTE1)
    if slot == 1:
        k1.unlock(pipe=tla.arch.MTE1)
    if slot == 2:
        k2.unlock(pipe=tla.arch.MTE1)


@tla.jit
def _k_slot_ready_wait(k0, k1, k2, slot):
    if slot == 0:
        k0.lock(pipe=tla.arch.MTE1)
    if slot == 1:
        k1.lock(pipe=tla.arch.MTE1)
    if slot == 2:
        k2.lock(pipe=tla.arch.MTE1)


@tla.jit
def _k_slot_ready_set(k0, k1, k2, slot):
    if slot == 0:
        k0.unlock(pipe=tla.arch.MTE2)
    if slot == 1:
        k1.unlock(pipe=tla.arch.MTE2)
    if slot == 2:
        k2.unlock(pipe=tla.arch.MTE2)


@tla.jit
def _k_slot_ptr(bufs, slot):
    return bufs.l1_k + slot * (HEAD_DIM * KV_BLOCK)


@tla.jit
def _q_slot_ptr(bufs, slot):
    return bufs.l1_q + slot * (Q_ROWS * Q_HALF)


@tla.jit
def load_k_ptr(sync, bufs, src_ptr, n_load, slot, dtype):
    with tla.cube():
        _k_slot_wait(sync.m_k0, sync.m_k1, sync.m_k2, slot)
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
        _k_slot_ready_set(sync.m_k0, sync.m_k1, sync.m_k2, slot)


@tla.jit
def _load_k_ptr_nocredit(sync, bufs, src_ptr, n_load, slot, dtype):
    with tla.cube():
        _k_slot_wait(sync.m_k0, sync.m_k1, sync.m_k2, slot)
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
        _k_slot_ready_set(sync.m_k0, sync.m_k1, sync.m_k2, slot)


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
        _k_slot_wait(sync.m_k0, sync.m_k1, sync.m_k2, slot)
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
        _k_slot_ready_set(sync.m_k0, sync.m_k1, sync.m_k2, slot)


@tla.jit
def wait_kv_stage(sync):
    with tla.cube():
        tla.cross_core_wait_flag(sync.kv_stage_ready, tla.arch.MTE2, aiv_id=0)
        tla.cross_core_wait_flag(sync.kv_stage_ready, tla.arch.MTE2, aiv_id=1)


@tla.jit
def wait_p_ready(sync):
    with tla.cube():
        tla.cross_core_wait_flag(sync.p_ready, tla.arch.MTE1, aiv_id=0)
        tla.cross_core_wait_flag(sync.p_ready, tla.arch.MTE1, aiv_id=1)


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
):
    m_ab0 = sync.m_ab0
    m_ab1 = sync.m_ab1
    m_q = sync.m_q
    m_s = sync.m_s
    m_s1 = sync.m_s1
    c0 = 0
    c1 = 1
    with tla.cube():
        _k_slot_ready_wait(sync.m_k0, sync.m_k1, sync.m_k2, slot)
        if first_tile:
            m_q.lock(pipe=tla.arch.MTE1)
        l1k = l1_k_tensor(_k_slot_ptr(bufs, slot), dtype)
        ub_s_full = tla.make_tensor(
            bufs.ub_s + slot_s * (Q_ROWS // 2 * KV_BLOCK),
            tla.make_layout(
                tla.make_shape(m, KV_BLOCK),
                tla.make_stride(KV_BLOCK, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        l0c_s = tla.make_tensor_like(
            bufs.l0c_s + slot_s * (Q_ROWS * KV_BLOCK), ub_s_full, tla.arch.L0Clayout
        )

        ub_s_dst = ub_s_full
        if n_blk != KV_BLOCK:
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
            m_s.lock(pipe=tla.arch.CUBE)
        if slot_s == 1:
            m_s1.lock(pipe=tla.arch.CUBE)
        for c in (0, 1, 2, 3):
            half = c // 2
            l1q = l1_q_tensor(
                _q_slot_ptr(bufs, q_slot_a if half == 0 else q_slot_b), dtype, m
            )
            q_chunk = tla.tile_view(
                l1q, tla.make_shape(m, D_CHUNK), tla.make_coord(c0, c % 2)
            )
            k_chunk = tla.tile_view(
                l1k, tla.make_shape(D_CHUNK, n_blk), tla.make_coord(c, 0)
            )
            slot_l0 = c % 2
            l0a_ptr = bufs.l0a0 if slot_l0 == 0 else bufs.l0a1
            l0b_ptr = bufs.l0b0 if slot_l0 == 0 else bufs.l0b1
            l0_q = tla.make_tensor_like(l0a_ptr, q_chunk)
            l0_k = tla.make_tensor_like(l0b_ptr, k_chunk)

            if slot_l0 == 0:
                m_ab0.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_q, q_chunk)
                tla.copy(l0_k, k_chunk)
                m_ab0.unlock(pipe=tla.arch.MTE1)
            else:
                m_ab1.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_q, q_chunk)
                tla.copy(l0_k, k_chunk)
                m_ab1.unlock(pipe=tla.arch.MTE1)
            if slot_l0 == 0:
                m_ab0.lock(pipe=tla.arch.CUBE)
            else:
                m_ab1.lock(pipe=tla.arch.CUBE)
            tla.mmad(
                l0c_s,
                l0_q,
                l0_k,
                init_c=(c == 0),
            )
            if slot_l0 == 0:
                m_ab0.unlock(pipe=tla.arch.CUBE)
            else:
                m_ab1.unlock(pipe=tla.arch.CUBE)
        if slot_s == 0:
            m_s.unlock(pipe=tla.arch.CUBE)
            m_s.lock(pipe=tla.arch.FIX)
        if slot_s == 1:
            m_s1.unlock(pipe=tla.arch.CUBE)
            m_s1.lock(pipe=tla.arch.FIX)
        tla.cross_core_wait_flag(sync.s_free, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(sync.s_free, tla.arch.FIX, aiv_id=1)
        tla.copy(
            ub_s_dst,
            l0c_s,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        if slot_s == 0:
            m_s.unlock(pipe=tla.arch.FIX)
        if slot_s == 1:
            m_s1.unlock(pipe=tla.arch.FIX)
        tla.cross_core_set_flag(sync.s_ready, tla.arch.FIX, aiv_id=0)
        tla.cross_core_set_flag(sync.s_ready, tla.arch.FIX, aiv_id=1)
        if is_last_tile:
            m_q.unlock(pipe=tla.arch.MTE1)


@tla.jit
def iterate_bmm2_seg(sync, bufs, seg, slot_k, m, dtype, slot_p=0, n_k=KV_BLOCK):
    m_ab0 = sync.m_ab0
    m_ab1 = sync.m_ab1
    m_o0 = sync.m_o0
    c0 = 0
    with tla.cube():
        l1p = l1_p_tensor(bufs.l1_p + slot_p * (Q_ROWS * KV_BLOCK), dtype, m)
        v_seg_full = l1_v_seg_tensor(
            _k_slot_ptr(bufs, slot_k) + seg * (KV_BLOCK * D_CHUNK), dtype
        )
        p_tile = tla.tile_view(l1p, tla.make_shape(m, n_k), tla.make_coord(c0, c0))

        v_seg = v_seg_full
        if n_k != KV_BLOCK:
            v_seg = tla.tile_view(
                v_seg_full, tla.make_shape(n_k, D_CHUNK), tla.make_coord(c0, c0)
            )
        slot = seg % 2
        l0a_ptr = bufs.l0a0 if slot == 0 else bufs.l0a1
        l0b_ptr = bufs.l0b0 if slot == 0 else bufs.l0b1
        l0_p = tla.make_tensor_like(l0a_ptr, p_tile)
        l0_v = tla.make_tensor_like(l0b_ptr, v_seg)
        if n_k != KV_BLOCK:
            l0_v = tla.make_tensor(
                l0b_ptr, tla.make_layout_with_tag((n_k, D_CHUNK), dtype, tla.arch.nZ)
            )

        m_f = ((m + 15) // 16) * 16
        l0c_seg = tla.make_tensor(
            bufs.l0c_o + seg * (m_f * D_CHUNK),
            tla.make_layout_with_tag((m, D_CHUNK), tla.Float32, tla.arch.L0Clayout),
        )

        if seg == 0:
            m_o0.lock(pipe=tla.arch.CUBE)

        if seg < 2:
            if slot == 0:
                m_ab0.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_p, p_tile)
                tla.copy(l0_v, v_seg)
                m_ab0.unlock(pipe=tla.arch.MTE1)
            else:
                m_ab1.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_p, p_tile)
                tla.copy(l0_v, v_seg)
                m_ab1.unlock(pipe=tla.arch.MTE1)
        else:
            if slot == 0:
                m_ab0.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_v, v_seg)
                m_ab0.unlock(pipe=tla.arch.MTE1)
            else:
                m_ab1.lock(pipe=tla.arch.MTE1)
                tla.copy(l0_v, v_seg)
                m_ab1.unlock(pipe=tla.arch.MTE1)
        if slot == 0:
            m_ab0.lock(pipe=tla.arch.CUBE)
        else:
            m_ab1.lock(pipe=tla.arch.CUBE)
        tla.mmad(l0c_seg, l0_p, l0_v, init_c=True, unit_flag=0)
        if slot == 0:
            m_ab0.unlock(pipe=tla.arch.CUBE)
        else:
            m_ab1.unlock(pipe=tla.arch.CUBE)


@tla.jit
def iterate_bmm2_fixpipe(sync, bufs, m):
    m_o0 = sync.m_o0
    c0 = 0
    with tla.cube():
        ub_full = tla.make_tensor(
            bufs.ub_otmp_full,
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
        m_o0.unlock(pipe=tla.arch.CUBE)
        m_o0.lock(pipe=tla.arch.FIX)
        tla.cross_core_wait_flag(sync.o_free, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(sync.o_free, tla.arch.FIX, aiv_id=1)
        tla.copy(
            ub_full,
            l0c_full,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        m_o0.unlock(pipe=tla.arch.FIX)
        tla.cross_core_set_flag(sync.o_ready, tla.arch.FIX, aiv_id=0)
        tla.cross_core_set_flag(sync.o_ready, tla.arch.FIX, aiv_id=1)


@tla.jit
def release_k(sync, slot):
    with tla.cube():
        _k_slot_free(sync.m_k0, sync.m_k1, sync.m_k2, slot)


@tla.jit
def split_g_mte2_pair(sync):
    with tla.cube():
        tla.cross_core_set_flag(sync.s2_mte2, tla.arch.MTE2)
        tla.cross_core_wait_flag(sync.s2_mte2, tla.arch.MTE2)


@tla.jit
def pair_rendezvous(sync):
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
        tla.cross_core_set_flag(sync.s2_rendezvous, tla.arch.FIX)
        tla.cross_core_wait_flag(sync.s2_rendezvous, tla.arch.SCALAR)


VL = 64


@tla.jit
def init_output_gm(
    sync, bufs, gm, base_elem, total_elem, pop_ele, fill_dtype, use_lse_buf=0
):
    src = bufs.ub_init_lse if use_lse_buf == 1 else bufs.ub_init
    ub = tla.make_tensor(
        src, tla.make_layout(tla.make_shape(pop_ele), tla.make_stride(1))
    )
    sync.m_init.lock(pipe=tla.arch.VECTOR)
    with tla.vec.func(mode="simd"):
        zero = tla.full(0.0, dtype=fill_dtype)
        zmask, _ = tla.update_mask(VL, dtype=fill_dtype)

        for c in tla.range(pop_ele // VL):
            tla.tile_view(ub, tla.make_shape(VL), tla.make_coord(c)).store(
                zero, params=tla.params.UnalignStoreParams()
            )
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    sync.m_init.unlock(pipe=tla.arch.VECTOR)
    sync.m_init.lock(pipe=tla.arch.MTE3)
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
    sync.m_init.unlock(pipe=tla.arch.MTE3)
    sync.m_init.lock(pipe=tla.arch.VECTOR)
    sync.m_init.unlock(pipe=tla.arch.VECTOR)


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
    vec_core_num = tla.arch.block_num() * 2
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
    # ---- softmaxLse (when isSoftmaxLseEnable) ----
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
    m_sink = sync.m_sink
    c0 = 0
    with tla.vector():
        sub = tla.arch.sub_block_idx()

        sink_len = sink_cap if sink_cap > 0 else rows
        ub_sink = _stats(bufs.ub_sink, sink_len)
        gm_sink_all = tla.make_tensor(
            sinks.ptr, tla.make_layout(tla.make_shape(sink_len), tla.make_stride(1))
        )
        if has_sinks == 1:
            m_sink.lock(pipe=tla.arch.MTE2)
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
            m_sink.unlock(pipe=tla.arch.MTE2)
        m_sink.lock(pipe=tla.arch.VECTOR)

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
                        ub_o_c.store(zero, mask=preg)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        m_sink.unlock(pipe=tla.arch.VECTOR)


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
    m_p = sync.m_p
    c0 = 0
    c1 = 1
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        ub_s = _ub_s(bufs, rows, slot_s)
        ub_p = _ub_p(bufs, rows)

        ub_max = _max(bufs, rows, slot_s)
        ub_max_rd = _max(bufs, rows, slot_max_rd)

        ub_sum = _sum(bufs, rows, slot_s)
        ub_sum_rd = _sum(bufs, rows, slot_sum_rd)

        ub_dm = _dm(bufs, rows, slot_s)
        tla.cross_core_wait_flag(sync.s_ready, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(sync.s_ready, tla.arch.VECTOR, aiv_id=1)

        m_p.lock(pipe=tla.arch.VECTOR)

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
            # ---- vec1 softmax: three s2RealSize specialisations ----

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
                    s0.store(v0, mask=preg)
                    s1.store(v1, mask=preg)
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
                        s0.store(v0, mask=preg)
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
                    # ---- arm 2, pass 2: single-slice exp/sum/cast + gather ----

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
                        s0.store(v0, mask=preg)
                        s1.store(v1, mask=preg)
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

        tla.cross_core_set_flag(sync.s_free, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(sync.s_free, tla.arch.VECTOR, aiv_id=1)
        # ---- UB P -> L1 P (each AIV writes half of M) ----

        l1p = tla.make_tensor(
            bufs.l1_p + slot_p * (Q_ROWS * KV_BLOCK),
            tla.make_layout_with_tag((m, KV_BLOCK), dtype_in, tla.arch.zN),
        )

        l1p_rows = tla.tile_view(
            l1p, tla.make_shape(rows, KV_BLOCK), tla.make_coord(sub, 0)
        )
        m_p.unlock(pipe=tla.arch.VECTOR)
        m_p.lock(pipe=tla.arch.MTE3)
        tla.copy(l1p_rows, ub_p)
        m_p.unlock(pipe=tla.arch.MTE3)
        tla.cross_core_set_flag(sync.p_ready, tla.arch.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.p_ready, tla.arch.MTE3, aiv_id=1)


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
            tla.cross_core_wait_flag(sync.o_ready, tla.arch.VECTOR, aiv_id=0)
            tla.cross_core_wait_flag(sync.o_ready, tla.arch.VECTOR, aiv_id=1)
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
            tla.cross_core_set_flag(sync.o_free, tla.arch.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(sync.o_free, tla.arch.VECTOR, aiv_id=1)


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
                    dst.store(tla.div(dst.load(), den, mask=preg), mask=preg)
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
        # ---- wait for the staged copy-out (MTE3->MTE2), then load prev max/sum ----
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
        # ---- per-row merge weights; write merged_lse / 1.0 back to ub_max / ub_sum ----
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
        # ---- per-segment prevO load + weighted merge (four segments unrolled) ----

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
):
    m_o16 = sync.m_o16
    c0 = 0
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        ub_o = tla.make_tensor(
            bufs.ub_o,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        ub_o16 = tla.make_tensor(
            bufs.ub_o16,
            tla.make_layout_with_tag((rows, HEAD_DIM), dtype_in, tla.arch.RowMajor),
        )

        ub_sum = _sum(bufs, rows, slot_sum)
        m_o16.lock(pipe=tla.arch.VECTOR)
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
        m_o16.unlock(pipe=tla.arch.VECTOR)
        m_o16.lock(pipe=tla.arch.MTE3)
        if odd_mode == 0:
            if sub * rows < m:
                gm_o = tla.make_tensor(
                    out_tensor.ptr + task * (m * HEAD_DIM) + (sub * rows * HEAD_DIM),
                    tla.make_layout_with_tag(
                        (rows, HEAD_DIM), dtype_in, tla.arch.RowMajor
                    ),
                )
                tla.copy(gm_o, ub_o16)
        else:
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
        m_o16.unlock(pipe=tla.arch.MTE3)


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
        sync.m_lse.lock(pipe=tla.arch.VECTOR)
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
        sync.m_lse.unlock(pipe=tla.arch.VECTOR)
        sync.m_lse.lock(pipe=tla.arch.MTE3)
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

        sync.m_lse.unlock(pipe=tla.arch.MTE3)
        sync.m_lse.lock(pipe=tla.arch.VECTOR)
        sync.m_lse.unlock(pipe=tla.arch.VECTOR)


_PAIR_ELEM_BYTES = 2

# ---------------------------------------------------------------------------


class TraceFidelityError(RuntimeError):
    pass


_TRACE_OWNER = None
_TRACE_PAIRED = False
_TRACE_STAGE = False
_TRACE_QUARTER = False
_TRACE_SLOTS = 0
_TRACE_EMITS = 0


def configure_trace(g_split, share_stage, share_quarter, kv_slots, owner):
    global _TRACE_OWNER, _TRACE_PAIRED
    global _TRACE_STAGE, _TRACE_QUARTER, _TRACE_SLOTS, _TRACE_EMITS
    _TRACE_OWNER = None
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
            "(share_stage=False)."
        )
    _TRACE_OWNER = owner
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

// Both use blockLen = dSizeVInput * sizeof(KV_T) bytes; a uint8 view plus a
// byte length lets one extern cover both dtypes with identical descriptors.
[aicore] __attribute__((always_inline)) void tla_pair_gm_to_ub(
    uint64_t src_gm_addr, uint64_t dst_ub_addr, int64_t block_len_bytes,
    int64_t src_stride_bytes, int64_t dst_byte_off, int64_t block_count) {
  AscendC::DataCopyExtParams intri;
  intri.blockCount = static_cast<uint32_t>(block_count);
  intri.blockLen = static_cast<uint32_t>(block_len_bytes);
  intri.dstStride = 0;
  intri.srcStride = src_stride_bytes;  // dav-c310: int64_t field
  AscendC::DataCopyPadExtParams<uint8_t> padParams;
  padParams.isPad = true;
  padParams.leftPadding = 0;
  padParams.rightPadding = 0;   // one row is already 32B aligned
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


@tla.extern(source=_PAIR_SOURCE, name="tla_pair_gm_to_ub")
def pair_gm_to_ub(
    src: tla.Pointer[tla.UInt8, tla.AddressSpace.gm],
    dst_base: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    block_len_bytes: tla.Int64,
    src_stride_bytes: tla.Int64,
    dst_byte_off: tla.Int64,
    block_count: tla.Int64,
) -> None: ...


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
    m_kv0 = sync.m_kv0
    m_kv1 = sync.m_kv1

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

                m_kv0.lock(pipe=tla.arch.MTE2)
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
                m_kv0.unlock(pipe=tla.arch.MTE2)
                m_kv0.lock(pipe=tla.arch.MTE3)
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
                m_kv0.unlock(pipe=tla.arch.MTE3)
            else:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv1,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                m_kv1.lock(pipe=tla.arch.MTE2)
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
                m_kv1.unlock(pipe=tla.arch.MTE2)
                m_kv1.lock(pipe=tla.arch.MTE3)
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
                m_kv1.unlock(pipe=tla.arch.MTE3)
        if announce:
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.arch.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.arch.MTE3, aiv_id=1)


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
    m_kv0 = sync.m_kv0
    m_kv1 = sync.m_kv1
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
                m_kv0.lock(pipe=tla.arch.MTE2)
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
                m_kv0.unlock(pipe=tla.arch.MTE2)
                m_kv0.lock(pipe=tla.arch.MTE3)
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
                m_kv0.unlock(pipe=tla.arch.MTE3)
            else:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv1,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                m_kv1.lock(pipe=tla.arch.MTE2)
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
                m_kv1.unlock(pipe=tla.arch.MTE2)
                m_kv1.lock(pipe=tla.arch.MTE3)
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
                m_kv1.unlock(pipe=tla.arch.MTE3)
        if announce:
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.arch.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.arch.MTE3, aiv_id=1)


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
                        HEAD_DIM * _PAIR_ELEM_BYTES,
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


SCALE_GROUPS = 7


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
    m_idx0 = sync.m_idx0
    m_idx1 = sync.m_idx1
    m_kv0 = sync.m_kv0
    m_kv1 = sync.m_kv1
    require_configured("gather_kv_rows_ring")
    if _TRACE_STAGE:
        require_share("gather_kv_rows_ring")
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
        # ---- index-row DMA (once per tile; two-slot ring rotating on x % 2) ----

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
                m_idx0.lock(pipe=tla.arch.MTE2)
            else:
                m_idx1.lock(pipe=tla.arch.MTE2)
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
                m_idx0.unlock(pipe=tla.arch.MTE2)
                m_idx0.lock(pipe=tla.arch.SCALAR)
            else:
                m_idx1.unlock(pipe=tla.arch.MTE2)
                m_idx1.lock(pipe=tla.arch.SCALAR)

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

                m_kv0.lock(pipe=tla.arch.MTE2)

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
                                HEAD_DIM * _PAIR_ELEM_BYTES,
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
                                HEAD_DIM * _PAIR_ELEM_BYTES,
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
                                HEAD_DIM * _PAIR_ELEM_BYTES,
                                pair_zero,
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(1),
                            )
                # ==== UB -> ws staging (GM round-trip): start ====
                m_kv0.unlock(pipe=tla.arch.MTE2)
                m_kv0.lock(pipe=tla.arch.MTE3)
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
                m_kv0.unlock(pipe=tla.arch.MTE3)
                # ==== UB -> ws staging: end ====
            else:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv1,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )

                m_kv1.lock(pipe=tla.arch.MTE2)

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
                                HEAD_DIM * _PAIR_ELEM_BYTES,
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
                                HEAD_DIM * _PAIR_ELEM_BYTES,
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
                                HEAD_DIM * _PAIR_ELEM_BYTES,
                                pair_zero,
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(1),
                            )
                # ==== UB -> ws staging (GM round-trip): start ====
                m_kv1.unlock(pipe=tla.arch.MTE2)
                m_kv1.lock(pipe=tla.arch.MTE3)
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
                m_kv1.unlock(pipe=tla.arch.MTE3)
                # ==== UB -> ws staging: end ====

        if n_chunks > 0:
            if idx_slot == 0:
                m_idx0.unlock(pipe=tla.arch.SCALAR)
            else:
                m_idx1.unlock(pipe=tla.arch.SCALAR)
        if announce:
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.arch.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.arch.MTE3, aiv_id=1)


KV_STAGE_STRIDE = KV_BLOCK * 512

WS_BLOCKS = 32

CSA_PRELOAD_NUM = 3


@tla.kernel
def sparse_flash_mla_csa_kernel(
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
    resid: tla.Tensor,
    sinks: tla.Tensor,
    out: tla.Tensor,
    lse: tla.Tensor,
    tiling: tla.Tensor,
    ws: tla.Tensor,
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
    topk_value_mode: tla.Constexpr[int],
    kv_layout: tla.Constexpr[int],
    is_vec_s2phyaddr: tla.Constexpr[int],
) -> None:
    dtype_in = q.ptr.dtype

    m = head_num // g_split

    rows = m - m // 2

    half_g = head_num // g_split if head_num % g_split == 0 else head_num // g_split + 1

    configure_trace(g_split, False, False, KV_SLOTS, "csa")

    use_vec_gather = (is_vec_s2phyaddr == 1) and (kv_layout != 2)
    idx_ring_on = use_vec_gather and kv_layout != 2

    sync = declare_sync(need_idx_ring=idx_ring_on)

    bufs = alloc_buffers(
        dtype_in,
        preload=True,
        idx_ring=idx_ring_on,
    )
    init_sync(sync)
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
        with tla.cube():
            sync_all_cores_aic()
        with tla.vector():
            sync_all_cores_aiv()

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
        start_idx = tla.as_numeric(0)
        vis = tla.as_numeric(0)
        kv_len = tla.as_numeric(0)
        n_ori_tiles = tla.as_numeric(0)
        cmp_lines = tla.as_numeric(0)
        kv_row0 = tla.as_numeric(0)
        cmp_row0 = tla.as_numeric(0)

        if step < n_steps:
            # ---- locate (seq, row); sparse indices use token row tok = task // g_split ----
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
            vis = end_idx - start_idx
            if vis < 0:
                vis = tla.as_numeric(0)
            n_ori_tiles = (vis + KV_BLOCK - 1) // KV_BLOCK

            sparse_count = topk[tok]
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

            n_tiles = n_ori_tiles + (cmp_lines + KV_BLOCK - 1) // KV_BLOCK
            kv_row0 = cu_kv[seq] + start_idx
            cmp_row0 = cu_cmp[seq]

        n_body = n_tiles
        if step == n_steps:
            if g_done > 0:
                n_body = n_tiles + CSA_PRELOAD_NUM

        for i in tla.range(n_body):
            x = g_done + i

            # ================================================================

            # ================================================================
            if i < n_tiles:
                j0 = i
                nv0 = tla.as_numeric(0)
                nl0 = tla.as_numeric(0)
                if j0 < n_ori_tiles:
                    nv0 = vis - j0 * KV_BLOCK
                    if nv0 > KV_BLOCK:
                        nv0 = tla.as_numeric(KV_BLOCK)

                    nl0 = vis - j0 * KV_BLOCK
                    if nl0 < 0:
                        nl0 = tla.as_numeric(0)
                    if nl0 > KV_BLOCK:
                        nl0 = tla.as_numeric(KV_BLOCK)
                else:
                    nv0 = cmp_lines - (j0 - n_ori_tiles) * KV_BLOCK
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
                r0_kvrow0 = kv_row0
                r0_start = start_idx
                r0_seq = seq
                r0_iscmp = tla.as_numeric(0)
                if j0 >= n_ori_tiles:
                    r0_iscmp = tla.as_numeric(1)

                if j0 >= n_ori_tiles:
                    base_line0 = (j0 - n_ori_tiles) * KV_BLOCK
                    slot0 = x - (x // KV_SLOTS) * KV_SLOTS
                    islot = x - (x // 2) * 2

                    if kv_layout == 2 and not use_vec_gather:
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

            # ================================================================

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
                    if r1_iscmp == 1:
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
                        if ori_pa_block_size > 0:
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

            # ================================================================

            # ================================================================

            # ---- ring rotation (end of body): r3<-r2, r2<-r1; r1<-r0 on a real tile ----
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
            rt = r2_start
            r3_start = rt
            rt = r2_seq
            r3_seq = rt
            r2_task = r1_task
            r2_j = r1_j
            r2_nt = r1_nt
            r2_nvalid = r1_nvalid
            r2_nload = r1_nload
            r2_kvrow0 = r1_kvrow0
            r2_start = r1_start
            r2_iscmp = r1_iscmp
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
                r1_seq = r0_seq

        g_done = g_done + n_tiles

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
    cu_q: tla.Tensor,
    cu_kv: tla.Tensor,
    cu_cmp: tla.Tensor,
    resid: tla.Tensor,
    sinks: tla.Tensor,
    out: tla.Tensor,
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

    odd_g = (g_split == 2) and (head_num % g_split != 0)
    half_g = (head_num + 1) // 2 if g_split == 2 else head_num

    m = (half_g + (half_g % 2)) if odd_g else (head_num // g_split)

    rows = m - m // 2
    row_cap = m if odd_g else None

    odd_mode = 1 if odd_g else 0
    sink_cap = ((head_num + 7) // 8) * 8 if odd_g else 0

    sync = declare_sync()

    bufs = alloc_buffers(
        dtype_in, row_cap, (sink_cap if sink_cap > 0 else None), preload=True
    )
    init_sync(sync)
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
        with tla.cube():
            sync_all_cores_aic()
        with tla.vector():
            sync_all_cores_aiv()

    total_tasks = cu_q[num_seqs] * g_split
    task0 = tla.arch.block_idx()
    q_bn = tla.arch.block_num()
    # ---- gate 3: global gtile pipeline across tasks (as in swa_kernel) ----

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
        # ---- locate (seq, row); the row index is the token (tok = task // g_split) ----
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

        half = task - tok * g_split
        g_off = half * half_g
        valid_m = half_g
        if half != 0:
            valid_m = head_num - half_g
        q_row_base = tok * head_num + g_off

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
            # ---- stage 0 (AIC): load tile i ----
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
            # ---- stage 1: vec1 of tile i-1 (AIV) and bmm1 (AIC) ----
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
            # ---- stage 2: bmm2 + fixpipe + vec2 of tile i-2 (+ release / copy_out) ----
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

        # ---- loop-carry advance after this task: roll P/PP and the gtile prefix ----
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


@tla.kernel
def sparse_flash_mla_swa_kernel(
    q: tla.Tensor,
    kv: tla.Tensor,
    ori_bt: tla.Tensor,
    sinks: tla.Tensor,
    cu_q: tla.Tensor,
    cu_kv: tla.Tensor,
    out: tla.Tensor,
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
) -> None:
    c0 = 0
    c1 = 1
    c2 = 2
    dtype_in = q.ptr.dtype

    m = head_num // g_split

    rows = m - m // 2
    half_g = head_num // g_split if head_num % g_split == 0 else head_num // g_split + 1

    sync = declare_sync()

    bufs = alloc_buffers(dtype_in, bc=is_batch_consistency, preload=True)

    init_sync(sync, pre_bc=(is_batch_consistency == 1))
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
        with tla.cube():
            sync_all_cores_aic()
        with tla.vector():
            sync_all_cores_aiv()

    total_tasks = cu_q[num_seqs] * g_split

    task0 = tla.arch.block_idx()
    q_bn = tla.arch.block_num()
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
        # ---- locate (seq, row); TND uses the cu_seqlens_q prefix table ----
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
        num_tiles = (vis + KV_BLOCK - 1) // KV_BLOCK

        raw_rb = (vis + 31) // 32
        rb = ((raw_rb + 127) // 128) * 128
        base_block_num = rb // 128
        if rb == 0:
            base_block_num = tla.as_numeric(1)
        kv_row0 = cu_kv[seq] + start_idx
        g_off = (task - tok * g_split) * m

        # ---- gate 3 prefix quantities (loop-carry, see the loop head) ----

        any_after = 0
        nx_task = task + q_bn
        nx_done = 0
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

        if g_k == 0:
            if num_tiles > 0:
                load_q(
                    sync, bufs, q, task, m, dtype_in, slot_a=q_slot_a, slot_b=q_slot_b
                )

        drain = 0
        if num_tiles > 0:
            if any_after == 0:
                drain = SWA_PRELOAD_NUM
        for i in tla.range(num_tiles + drain):
            x = g_k + i
            # ---- stage 0 (AIC): tile x is the i-th tile of this task ----
            if i < num_tiles:
                n_load0 = vis - i * KV_BLOCK
                if n_load0 < 0:
                    n_load0 = tla.as_numeric(0)
                if n_load0 > KV_BLOCK:
                    n_load0 = tla.as_numeric(KV_BLOCK)
                slot_k0 = x - (x // KV_SLOTS) * KV_SLOTS

                if ori_pa_block_size > 0:
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
                    slot_k1 = xg1 - (xg1 // KV_SLOTS) * KV_SLOTS
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

                    if is_first1:
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
                    )

                    if is_last1:
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
                slot_k2 = xg2 - (xg2 // KV_SLOTS) * KV_SLOTS
                slot_p2 = xg2 - (xg2 // SWA_PRELOAD_NUM) * SWA_PRELOAD_NUM

                tid_in_block2 = j2
                is_first_base2 = j2 == 0
                reduce_block_id2 = j2

                fresh_v2 = is_first_base2
                if is_batch_consistency == 1:
                    tid_in_block2 = j2 - (j2 // bbn2) * bbn2
                    is_first_base2 = tid_in_block2 == 0
                    reduce_block_id2 = j2 // bbn2

                    fresh_v2 = is_first_base2
                wait_p_ready(sync)

                for seg in (0, 1, 2, 3):
                    iterate_bmm2_seg(
                        sync, bufs, seg, slot_k2, m, dtype_in, slot_p=slot_p2
                    )
                iterate_bmm2_fixpipe(sync, bufs, m)
                for seg in (0, 1, 2, 3):
                    process_vec2_seg(
                        sync, bufs, seg, m, rows, dtype_in, fresh_v2, slot_dm=slot_p2
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
                release_k(sync, slot_k2)

                if j2 + c1 == nt2:
                    bmm2_data_copy_out(
                        sync, bufs, out, task2, m, rows, dtype_in, slot_sum=slot_p2
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
                            row_base=(task2 // g_split) * head_num
                            + (task2 - (task2 // g_split) * g_split) * half_g,
                        )

        # ---- loop-carry state advance after this task: roll P/PP and the gtile prefix ----

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
    ori_topk: tla.Tensor,
    cu_q: tla.Tensor,
    cu_kv: tla.Tensor,
    sinks: tla.Tensor,
    out: tla.Tensor,
    lse: tla.Tensor,
    tiling: tla.Tensor,
    ws: tla.Tensor,
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
    topk_value_mode: tla.Constexpr[int],
    kv_layout: tla.Constexpr[int],
    is_vec_s2phyaddr: tla.Constexpr[int],
) -> None:
    dtype_in = q.ptr.dtype

    m = head_num // g_split

    rows = m - m // 2

    half_g = head_num // g_split if head_num % g_split == 0 else head_num // g_split + 1

    configure_trace(g_split, False, False, KV_SLOTS, "os")

    use_vec_gather = (is_vec_s2phyaddr == 1) and (kv_layout != 2)
    idx_ring_on = use_vec_gather and kv_layout != 2

    sync = declare_sync(need_idx_ring=idx_ring_on)

    bufs = alloc_buffers(
        dtype_in,
        preload=True,
        idx_ring=idx_ring_on,
    )
    init_sync(sync)
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
        with tla.cube():
            sync_all_cores_aic()
        with tla.vector():
            sync_all_cores_aiv()
    else:
        if need_init == 1:
            with tla.cube():
                sync_all_cores_aic()
            with tla.vector():
                sync_all_cores_aiv()

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
            # ---- locate (seq, row); sparse indices use token row tok = task // g_split ----
            tok = task // g_split
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

            # ---- rows: Min(rangeLen, Min(topk, idxWidth)), then Min(actualS2OriSize) ----
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

            # ---- ring rotation (end of body): r3<-r2, r2<-r1; r1<-r0 on a real tile ----
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
    ori_topk: tla.Tensor,
    cmp_idx: tla.Tensor,
    cmp_topk: tla.Tensor,
    cu_q: tla.Tensor,
    cu_kv: tla.Tensor,
    cu_cmp: tla.Tensor,
    resid: tla.Tensor,
    sinks: tla.Tensor,
    out: tla.Tensor,
    lse: tla.Tensor,
    tiling: tla.Tensor,
    ws: tla.Tensor,
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
    ori_pa_block_size: tla.Constexpr[int],
    ori_pa_stride: tla.Constexpr[int],
    ori_pa_max_blocks: tla.Constexpr[int],
    cmp_pa_block_size: tla.Constexpr[int],
    cmp_pa_stride: tla.Constexpr[int],
    cmp_pa_max_blocks: tla.Constexpr[int],
    g_split: tla.Constexpr[int],
    has_sinks: tla.Constexpr[int],
    enable_lse: tla.Constexpr[int],
    topk_value_mode: tla.Constexpr[int],
    kv_layout: tla.Constexpr[int],
    is_vec_s2phyaddr: tla.Constexpr[int],
) -> None:
    dtype_in = q.ptr.dtype

    m = head_num // g_split

    rows = m - m // 2
    half_g = head_num // g_split if head_num % g_split == 0 else head_num // g_split + 1

    # ---- single entry point for pairing semantics ----

    _PAIRED = g_split == 2
    configure_trace(g_split, _PAIRED, _PAIRED, KV_SLOTS, "ocs")

    use_vec_gather = (is_vec_s2phyaddr == 1) and (kv_layout != 2)
    idx_ring_on = use_vec_gather and kv_layout != 2

    sync = declare_sync(
        need_rendezvous=_TRACE_STAGE,
        need_idx_ring=idx_ring_on,
    )
    bufs = alloc_buffers(
        dtype_in,
        preload=True,
        idx_ring=idx_ring_on,
    )
    init_sync(sync)
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

    if is_vec_s2phyaddr == 1:
        with tla.cube():
            sync_all_cores_aic()
        with tla.vector():
            sync_all_cores_aiv()
    else:
        if need_init == 1:
            with tla.cube():
                sync_all_cores_aic()
            with tla.vector():
                sync_all_cores_aiv()

    total_tasks = cu_q[num_seqs] * g_split
    task0 = tla.arch.block_idx()
    q_bn = tla.arch.block_num()

    if is_vec_s2phyaddr == 1:
        with tla.cube():
            sync_all_cores_aic()
        with tla.vector():
            sync_all_cores_aiv()

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

    SHARE_SLOTS = _TRACE_SLOTS
    if _TRACE_STAGE:
        ws_base = (tla.arch.block_idx() // 2) * (SHARE_SLOTS * KV_STAGE_STRIDE)

    n_steps = (total_tasks - task0 + q_bn - 1) // q_bn
    if n_steps < 0:
        n_steps = tla.as_numeric(0)

    # ---- shared staging: the cross-block quota counts global steps ----

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
            # ---- locate (seq, row); sparse indices / topk use the token row (tok), not the G half ----
            tok = task // g_split
            rowq = tok
            for bb in static_range(1, num_seqs):
                if tok >= cu_q[bb]:
                    seq = bb
                    rowq = tok - cu_q[bb]
            q_len = cu_q[seq + 1] - cu_q[seq]
            kv_len = cu_kv[seq + 1] - cu_kv[seq]
            cmp_len = cu_cmp[seq + 1] - cu_cmp[seq]

            # ---- ori visible range (ComputeParamBatch :96-114) ----
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

            # ---- ori row count (ORI_CMP_SPARSE branch: start=0, end=Min(...)) ----
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

            # ---- cmp row count (:86-93 + :410-420) ----
            cmp_count = cmp_topk[tok]
            if cmp_count > cmp_idx_width:
                cmp_count = tla.as_numeric(cmp_idx_width)
            if cmp_count < 0:
                cmp_count = tla.as_numeric(0)

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

            # ================================================================
            if i < n_tiles:
                j0 = i
                nv0 = tla.as_numeric(0)
                nl0 = tla.as_numeric(0)
                slot0 = x - (x // SHARE_SLOTS) * SHARE_SLOTS
                # ---- dual-index phasing: why ori and cmp can share one event pair ----

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

            # ---- ring rotation (end of body): r3<-r2, r2<-r1; r1<-r0 on a real tile ----
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
        sync.m_init.lock(pipe=tla.arch.MTE3)
        sync.m_init.unlock(pipe=tla.arch.MTE3)
        sync.m_init.lock(pipe=tla.arch.VECTOR)

    # ---- trace-time closure check: a promised shared arm that never fired fails ----
    verify_trace_closed("ocs")


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


def _build_cu(cu, seqused, batch, per_batch, device):
    key = (
        id(cu) if cu is not None else 0,
        id(seqused) if seqused is not None else 0,
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
        id(bt) if bt is not None else 0,
        id(kv) if kv is not None else 0,
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
        else:
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
):
    # ---- repeat-call fast path: one launch plan per identical argument set ----
    sig = (
        id(q),
        id(ori_kv),
        id(cmp_kv),
        id(ori_sparse_indices),
        id(cmp_sparse_indices),
        id(ori_block_table),
        id(cmp_block_table),
        id(cu_seqlens_q),
        id(cu_seqlens_ori_kv),
        id(cu_seqlens_cmp_kv),
        id(seqused_q),
        id(seqused_ori_kv),
        id(seqused_cmp_kv),
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
    )
    plan = _LAUNCH_PLAN.get(sig)
    if plan is not None:
        plan[0]()
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

    output = _pooled_output(q)
    if want_lse:
        lse = _pooled_lse(
            device,
            (1, total_rows, heads) if layout_q == "TND" else (batch, 1, s1_len, heads),
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
    kv_width = HEAD_DIM
    cmp_width_r = HEAD_DIM
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

    g_split = 2 if heads > 64 else 1

    if g_split == 1 and heads % 4 == 0 and int(cu_q[-1]) == 1:
        g_split = 2
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

    block_num = min(_cube_cores(device), WS_BLOCKS)
    _n_task = int(cu_q[-1]) * g_split if cu_q else 0
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

    if route in ("csa", "os", "ocs"):
        ws = _scratch(device, WS_BLOCKS * KV_STAGE_BYTES_EL, dtype)
    else:
        ws = _dummy(device, dtype)

    if route == "swa":
        t_kv = _tla_tensor(kv2)
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
        )
        kernel_fn = sparse_flash_mla_swa_kernel
    elif route == "hca":
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

        cmp_len_max = max(
            (cu_cmp[i + 1] - cu_cmp[i] for i in range(len(cu_cmp) - 1)), default=0
        )
        cmp_blk = 32 if cmp_len_max <= 32 else (64 if cmp_len_max <= 64 else 128)
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
        tensor_args = (
            _tla_tensor(q2),
            _tla_tensor(kv2),
            _tla_tensor(cmp2),
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
        )
        kernel_fn = sparse_flash_mla_csa_kernel
    elif route == "os":
        idx = ori_sparse_indices.reshape(total_rows, ori_w)
        topk = ori_topk_length.reshape(-1)
        if q_rows_idx is not None:
            idx = idx.index_select(0, q_rows_idx)
            topk = topk.index_select(0, q_rows_idx)
        idx = idx.reshape(-1).to(i32).contiguous()
        topk = topk.to(i32).contiguous()
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
        )
        kernel_fn = sparse_flash_mla_ori_sparse_kernel
    else:
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
        )
        kernel_fn = sparse_flash_mla_ori_cmp_sparse_kernel

    key = (
        route,
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
        except Exception as exc:
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

    def _launch():
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
    _launch()
    _post()
    return output, lse


SCALE_GROUPS = 7
SUPPORTED_QUANT_MODE = 1
METADATA_FA_SIZE = 1024

_KV_POOL = {}
_KV_POOL_SLOTS = 2
_DEQUANT_EXEC = {}
_DEQUANT_PAGES_EXEC = {}
_PLANE_POOL = {}
_CPLANE_POOL = {}
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

# ---------------------------------------------------------------------------


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
        except Exception:
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
        id(block_table),
        id(seqused),
        id(residual),
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
    ckey = (id(rows), id(pages_t))
    hit = _CPLANE_CACHE.get(ckey)
    if hit is not None and hit[0] is rows and hit[1] is pages_t:
        return hit[2]
    n = int(pages_t.numel()) * bs
    pkey = (str(dev), n)
    plane = _CPLANE_POOL.get(pkey)
    if plane is None:
        plane = torch.zeros(n, SCALE_STRIDE, dtype=torch.float32, device=dev)
        _CPLANE_POOL[pkey] = plane
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
    key = (str(dev), str(element_type), rows.shape[0], int(pages_t.numel()), bs, block)
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


def _launch_dequant_pair(src_a, pages_a, src_b, pages_b, dtype_q):
    dev = src_a.device
    rows_a, t_nope_a, t_rope_a, et_a = _record_views(src_a)
    rows_b, t_nope_b, t_rope_b, et_b = _record_views(src_b)
    bs_a = int(src_a.shape[1])
    bs_b = int(src_b.shape[1])
    block_a = _rows_for_pages(bs_a, ((len(pages_a) + 7) // 8) * 8, dev)
    block_b = _rows_for_pages(bs_b, ((len(pages_b) + 7) // 8) * 8, dev)
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
    cfg = DequantPagesPairCfg(block_a, bs_a, block_b, bs_b)
    key = (
        "pair",
        str(dev),
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
    )
    executor = _DEQUANT_PAGES_EXEC.get(key)
    if executor is None:
        executor = tla.compile(
            dequant_kv_pages_pair,
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
            cfg,
            options=NPU_ARCH,
        )
        _DEQUANT_PAGES_EXEC[key] = executor
    n_blocks = int(pt_a.numel()) * (bs_a // block_a) + int(pt_b.numel()) * (
        bs_b // block_b
    )
    executor(
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
        block_num=_grid_for(dev, n_blocks),
    )
    return _out_view(out_a, src_a.shape), _out_view(out_b, src_b.shape)


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
    key = (str(rows.device), str(element_type), n_rows, block)
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
    """Run the mixed-quant sparse flash MLA forward pass.

    ``q`` is ``[T1, N1, 512]`` (TND) or ``[B, S1, N1, 512]`` (BSND); ``ori_kv`` and
    ``cmp_kv`` hold 608-byte quantised records in the layout selected by
    ``layout_kv``. ``cmp_sparse_indices`` / ``ori_sparse_indices`` select the
    compressed / sparse columns per query row. Returns ``(out, lse)`` where
    ``out`` matches the query shape and dtype and ``lse`` is f32
    ``[T1, N1]``-equivalent (``None`` when ``return_softmax_lse`` is false).
    """
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
        if ori_kv is not None
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
        if cmp_kv is not None
        else None
    )
    if ori_kv_pages and cmp_kv_pages:
        ori_k, cmp_k = _launch_dequant_pair(
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
    )
