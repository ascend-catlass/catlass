# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device-side CATLASS DSL kernels for ``flash_attn_fp8_fullquant``.

This file is the canonical Device-side source.  ``build_solution.py`` bundles
it together with the Host-side module into the runnable single-file solution
expected by the v5 benchmark harness.

The final kernel is a mixed AIC/AIV FP8 flash attention:
  QK^T (E4M3 x E4M3 -> FP32) -> online softmax -> P quantized to E4M3FN
  -> PV (E4M3 x E4M3 -> FP32) -> FP16 output.

Complexity: one task per (batch, q-head, q-block); each task scans every KV
page once.  Per token-head the work is O(seq_kv * D) for QK and O(seq_kv * D)
for PV; there is no per-V re-scan of K.
"""

from __future__ import annotations

import catlass.tla as tla


@tla.jit
def _p_store64(p_reg, ub_p, row, chunk):
    """64 FP32 probabilities -> 64 packed E4M3FN bytes (4x slot narrowing)."""
    full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
    cast0 = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    cast1 = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ONE,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    cast2 = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.TWO,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    cast3 = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.THREE,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    even, odd = tla.deinterleave(p_reg, p_reg)
    ee, eo = tla.deinterleave(even, even)
    oe, oo = tla.deinterleave(odd, odd)
    p0 = ee.to(tla.Float8E4M3FN, cast0, mask=full).bitcast(tla.UInt8)
    p1 = oe.to(tla.Float8E4M3FN, cast1, mask=full).bitcast(tla.UInt8)
    p2 = eo.to(tla.Float8E4M3FN, cast2, mask=full).bitcast(tla.UInt8)
    p3 = oo.to(tla.Float8E4M3FN, cast3, mask=full).bitcast(tla.UInt8)
    packed = tla.bitwise_or(tla.bitwise_or(p0, p1), tla.bitwise_or(p2, p3))
    tla.tile_view(ub_p, tla.make_shape(1, 64), tla.make_coord(row, chunk)).store(
        packed.bitcast(tla.Float8E4M3FN)
    )


@tla.jit
def _sf_tile(
    ub_s,
    ub_p,
    ub_qs,
    ub_ks,
    last_max,
    last_sum,
    dm_buf,
    m,
    kv_len,
    q_row_base,
    page_base,
    softmax_scale,
    p_scale,
    causal,
    full_visible,
    first,
):
    """One KV page of online softmax with per-token Q/K descale + P quantize.

    ``ub_s`` is (m, 128) FP32, ``ub_p`` is (56, 128) E4M3FN row-major,
    ``ub_qs`` is the half's per-token q_scale, ``ub_ks`` the page's k_scale.
    Masked columns use -3e38 so ``exp`` underflows to exactly 0 on mid tiles;
    causal is bottom-right aligned and only masks a prefix of each row.
    """
    CHUNK = 64
    with tla.vec.func(mode="simd"):
        full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        one, _ = tla.update_mask(1, dtype=tla.Float32)
        minv = tla.full(-3.0e38, tla.Float32)
        zero = tla.full(0.0, tla.Float32)
        k0 = tla.tile_view(ub_ks, tla.make_shape(1, 64), tla.make_coord(0, 0)).load()
        k1 = tla.tile_view(ub_ks, tla.make_shape(1, 64), tla.make_coord(0, 1)).load()
        for i in tla.range(m):
            qs = tla.tile_view(ub_qs, tla.make_shape(1), tla.make_coord(i)).load(
                params=tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                )
            )
            qsc = tla.mul(qs, softmax_scale, mask=full)
            s0 = tla.tile_view(
                ub_s, tla.make_shape(1, CHUNK), tla.make_coord(i, 0)
            ).load()
            s1 = tla.tile_view(
                ub_s, tla.make_shape(1, CHUNK), tla.make_coord(i, 1)
            ).load()
            s0 = tla.mul(tla.mul(s0, qsc, mask=full), k0, mask=full)
            s1 = tla.mul(tla.mul(s1, qsc, mask=full), k1, mask=full)
            if full_visible == 0:
                lim = kv_len if causal == 0 else q_row_base + i + 1
                lim = kv_len if lim > kv_len else lim
                c0 = lim - page_base
                c0 = 64 if c0 > 64 else c0
                c0 = 0 if c0 < 0 else c0
                c1 = lim - page_base - 64
                c1 = 64 if c1 > 64 else c1
                c1 = 0 if c1 < 0 else c1
                mask0, _ = tla.update_mask(c0, dtype=tla.Float32)
                mask1, _ = tla.update_mask(c1, dtype=tla.Float32)
                s0 = tla.where(mask0, s0, minv)
                s1 = tla.where(mask1, s1, minv)
            mx = tla.max(s0, s1, mask=full)
            mxr = mx.reduce(tla.ReductionOp.MAX, mask=full)
            tla.tile_view(dm_buf, tla.make_shape(1), tla.make_coord(i)).store(
                mxr, tla.params.UnalignStoreParams(), mask=one
            )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            tm_brc = tla.tile_view(dm_buf, tla.make_shape(1), tla.make_coord(i)).load(
                params=tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                )
            )
            lm = tla.tile_view(last_max, tla.make_shape(1), tla.make_coord(i)).load(
                params=tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                )
            )
            ls = tla.tile_view(last_sum, tla.make_shape(1), tla.make_coord(i)).load(
                params=tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                )
            )
            new_max = tm_brc
            dmr = tla.full(1.0, tla.Float32)
            if first != 0:
                # A fully masked first page has no finite row max.  Mapping the
                # sentinel to zero prevents exp(minv - minv) == 1 from leaking
                # probability mass into an otherwise empty causal row.
                new_max = tla.where(tla.cmp(tm_brc, -1.0e38, mode="gt"), tm_brc, zero)
            if first == 0:
                new_max = tla.max(tm_brc, lm, mask=full)
                dmr = tla.exp(tla.sub(lm, new_max, mask=full), mask=full)
            p0 = tla.exp(tla.sub(s0, new_max, mask=full), mask=full)
            p1 = tla.exp(tla.sub(s1, new_max, mask=full), mask=full)
            row_sum = tla.add(
                p0.reduce(tla.ReductionOp.ADD, mask=full),
                p1.reduce(tla.ReductionOp.ADD, mask=full),
                mask=full,
            )
            new_sum = row_sum
            if first == 0:
                new_sum = tla.add(row_sum, tla.mul(ls, dmr, mask=full), mask=full)
            tla.tile_view(last_max, tla.make_shape(1), tla.make_coord(i)).store(
                new_max, tla.params.UnalignStoreParams(), mask=one
            )
            tla.tile_view(last_sum, tla.make_shape(1), tla.make_coord(i)).store(
                new_sum, tla.params.UnalignStoreParams(), mask=one
            )
            tla.tile_view(dm_buf, tla.make_shape(1), tla.make_coord(i)).store(
                dmr, tla.params.UnalignStoreParams(), mask=one
            )
            _p_store64(tla.mul(p0, p_scale, mask=full), ub_p, i, 0)
            _p_store64(tla.mul(p1, p_scale, mask=full), ub_p, i, 1)
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.kernel
def fafq_kernel(
    gm_q: tla.Tensor,
    gm_qscale: tla.Tensor,
    gm_k: tla.Tensor,
    gm_kscale: tla.Tensor,
    gm_v: tla.Tensor,
    gm_vscale: tla.Tensor,
    gm_bt: tla.Tensor,
    gm_cuq: tla.Tensor,
    gm_seqq: tla.Tensor,
    gm_seqkv: tla.Tensor,
    gm_out: tla.Tensor,
    hq: tla.Int32,
    hkv: tla.Int32,
    batch: tla.Int32,
    max_blocks: tla.Int32,
    max_q_blocks: tla.Int32,
    softmax_scale: tla.Float32,
    causal: tla.Int32,
    p_scale: tla.Float32,
    q_pad_rows: tla.Int32,
    out_tokens: tla.Int32,
) -> None:
    """Paged FP8 fully-quantized GQA flash attention (single mixed kernel)."""
    Q_BLOCK = 64
    HALF = 32
    D = 128
    PAGE = 128

    l1_q = tla.allocate(Q_BLOCK * D, tla.Float8E4M3FN, tla.AddressSpace.l1, 512)
    l1_k = tla.allocate(D * PAGE, tla.Float8E4M3FN, tla.AddressSpace.l1, 512)
    l1_v = tla.allocate(PAGE * D, tla.Float8E4M3FN, tla.AddressSpace.l1, 512)
    l1_p = tla.allocate(Q_BLOCK * PAGE, tla.Float8E4M3FN, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(Q_BLOCK * PAGE, tla.Float8E4M3FN, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(PAGE * D, tla.Float8E4M3FN, tla.AddressSpace.l0b, 512)
    l0c_s = tla.allocate(Q_BLOCK * PAGE, tla.Float32, tla.AddressSpace.l0c, 1024)
    l0c_o = tla.allocate(Q_BLOCK * D, tla.Float32, tla.AddressSpace.l0c, 1024)

    ub_s = tla.allocate(HALF * PAGE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_o = tla.allocate(HALF * D, tla.Float32, tla.AddressSpace.ub, 256)
    ub_acc = tla.allocate(HALF * D, tla.Float32, tla.AddressSpace.ub, 256)
    ub_p = tla.allocate(HALF * PAGE, tla.Float8E4M3FN, tla.AddressSpace.ub, 256)
    ub_qs = tla.allocate(Q_BLOCK, tla.Float32, tla.AddressSpace.ub, 256)
    ub_ks = tla.allocate(PAGE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_lm = tla.allocate(HALF, tla.Float32, tla.AddressSpace.ub, 256)
    ub_ls = tla.allocate(HALF, tla.Float32, tla.AddressSpace.ub, 256)
    ub_dm = tla.allocate(HALF, tla.Float32, tla.AddressSpace.ub, 256)
    ub_out = tla.allocate(HALF * D, tla.Float16, tla.AddressSpace.ub, 256)

    q_l0a_ready = tla.flag("q_l0a_ready", tla.arch.MTE1, tla.arch.MTE2)
    q_l1_ready = tla.flag("q_l1_ready", tla.arch.MTE2, tla.arch.MTE1)
    k_l0b_ready = tla.flag("k_l0b_ready", tla.arch.MTE1, tla.arch.MTE2)
    k_l1_ready = tla.flag("k_l1_ready", tla.arch.MTE2, tla.arch.MTE1)
    v_l0b_ready = tla.flag("v_l0b_ready", tla.arch.MTE1, tla.arch.MTE2)
    v_l1_ready = tla.flag("v_l1_ready", tla.arch.MTE2, tla.arch.MTE1)
    l0a_avail = tla.flag("l0a_avail", tla.arch.CUBE, tla.arch.MTE1)
    l0a_ready = tla.flag("l0a_ready", tla.arch.MTE1, tla.arch.CUBE)
    l0b_avail = tla.flag("l0b_avail", tla.arch.CUBE, tla.arch.MTE1)
    l0b_ready = tla.flag("l0b_ready", tla.arch.MTE1, tla.arch.CUBE)
    l0cs_avail = tla.flag("l0cs_avail", tla.arch.FIX, tla.arch.CUBE)
    l0cs_ready = tla.flag("l0cs_ready", tla.arch.CUBE, tla.arch.FIX)
    l0co_avail = tla.flag("l0co_avail", tla.arch.FIX, tla.arch.CUBE)
    l0co_ready = tla.flag("l0co_ready", tla.arch.CUBE, tla.arch.FIX)
    ub_p_avail = tla.flag("ub_p_avail", tla.arch.MTE3, tla.arch.VECTOR)
    ub_p_ready = tla.flag("ub_p_ready", tla.arch.VECTOR, tla.arch.MTE3)
    sc_ready = tla.flag("sc_ready", tla.arch.MTE2, tla.arch.VECTOR)
    sc_free = tla.flag("sc_free", tla.arch.VECTOR, tla.arch.MTE2)
    out_ready = tla.flag("out_ready", tla.arch.VECTOR, tla.arch.MTE3)
    s_cross = tla.cross_flag("s_cross")
    p_cross = tla.cross_flag("p_cross")
    o_cross = tla.cross_flag("o_cross")
    task_c_done = tla.cross_flag("task_c_done")
    task_v_done = tla.cross_flag("task_v_done")

    with tla.cube():
        tla.set_flag(q_l0a_ready)
        tla.set_flag(k_l0b_ready)
        tla.set_flag(v_l0b_ready)
        tla.set_flag(l0a_avail)
        tla.set_flag(l0b_avail)
        tla.set_flag(l0cs_avail)
        tla.set_flag(l0co_avail)
        tla.cross_core_set_flag(p_cross, tla.arch.MTE1)
        tla.cross_core_set_flag(task_c_done, tla.arch.CUBE)
    with tla.vector():
        tla.set_flag(ub_p_avail)
        tla.set_flag(sc_free)
        tla.cross_core_set_flag(s_cross, tla.arch.VECTOR)
        tla.cross_core_set_flag(o_cross, tla.arch.VECTOR)
        tla.cross_core_set_flag(task_v_done, tla.arch.VECTOR)

    total_tasks = batch * hq * max_q_blocks
    for task in tla.range(tla.arch.block_idx(), total_tasks, tla.arch.block_num()):
        b = task // (hq * max_q_blocks)
        rem = task - b * hq * max_q_blocks
        h = rem // max_q_blocks
        qb = rem - h * max_q_blocks
        if (h % 2) != 0:
            qb = max_q_blocks - 1 - qb
        q_len_b = gm_seqq[b]
        qb_start = qb * Q_BLOCK
        rows = q_len_b - qb_start
        if rows > 0:
            rows = Q_BLOCK if rows > Q_BLOCK else rows
            pipe_rows = 2 if rows < 2 else rows
            q_start = gm_cuq[b] + qb_start
            kv_len = gm_seqkv[b]
            n_pages = (kv_len + PAGE - 1) // PAGE
            if causal != 0:
                # Causal window: only pages up to the last row's visible limit.
                q_last = qb_start + rows - 1 + (kv_len - q_len_b)
                visible = q_last + 1
                visible = kv_len if visible > kv_len else visible
                visible = 0 if visible < 0 else visible
                n_pages = (visible + PAGE - 1) // PAGE
            first_lim = kv_len
            if causal != 0:
                first_lim = qb_start + (kv_len - q_len_b) + 1
                first_lim = kv_len if first_lim > kv_len else first_lim
                first_lim = 0 if first_lim < 0 else first_lim
            group = hq // hkv
            kv_head = h // group
            gm_q_t = tla.make_tensor(
                gm_q.ptr + (h * q_pad_rows + q_start) * D,
                tla.make_layout(tla.make_shape(Q_BLOCK, D), tla.make_stride(D, 1)),
            )
            with tla.cube():
                tla.cross_core_wait_flag(task_v_done, tla.arch.CUBE)
            with tla.vector():
                tla.cross_core_wait_flag(task_c_done, tla.arch.VECTOR)
            with tla.vector():
                sub0 = tla.arch.sub_block_idx()
                m_half0 = (pipe_rows + 1) // 2
                m_sub0 = m_half0 if sub0 == 0 else pipe_rows - m_half0
                if m_sub0 > 0:
                    lm0 = tla.make_tensor(
                        ub_lm, tla.make_layout(tla.make_shape(HALF), tla.make_stride(1))
                    )
                    ls0 = tla.make_tensor(
                        ub_ls, tla.make_layout(tla.make_shape(HALF), tla.make_stride(1))
                    )
                    dm0 = tla.make_tensor(
                        ub_dm, tla.make_layout(tla.make_shape(HALF), tla.make_stride(1))
                    )
                    acc0 = tla.make_tensor(
                        ub_acc,
                        tla.make_layout(tla.make_shape(HALF, D), tla.make_stride(D, 1)),
                    )
                    with tla.vec.func(mode="simd"):
                        full0 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        one0, _ = tla.update_mask(1, dtype=tla.Float32)
                        minv0 = tla.full(-3.0e38, tla.Float32)
                        zero0 = tla.full(0.0, tla.Float32)
                        onef0 = tla.full(1.0, tla.Float32)
                        for j in tla.range(m_sub0):
                            tla.tile_view(
                                lm0, tla.make_shape(1), tla.make_coord(j)
                            ).store(minv0, tla.params.UnalignStoreParams(), mask=one0)
                            tla.tile_view(
                                ls0, tla.make_shape(1), tla.make_coord(j)
                            ).store(zero0, tla.params.UnalignStoreParams(), mask=one0)
                            tla.tile_view(
                                dm0, tla.make_shape(1), tla.make_coord(j)
                            ).store(onef0, tla.params.UnalignStoreParams(), mask=one0)
                            tla.tile_view(
                                acc0,
                                tla.make_shape(1, 64),
                                tla.make_coord(j, 0),
                            ).store(zero0, mask=full0)
                            tla.tile_view(
                                acc0,
                                tla.make_shape(1, 64),
                                tla.make_coord(j, 1),
                            ).store(zero0, mask=full0)

            for p in tla.range(0, n_pages):
                phys = gm_bt[b * max_blocks + p]
                page_base = p * PAGE
                full_vis = 1 if page_base + PAGE <= first_lim else 0
                kv_off = (phys * PAGE) * hkv * D + kv_head * D
                # ---- QK: Q x K^T -> S (FP8 MMAD, FP32 accumulate) ----
                with tla.cube():
                    tla.wait_flag(q_l0a_ready)
                    l1_q_t2 = tla.make_tensor_like(l1_q, gm_q_t, tla.arch.zN)
                    tla.copy(l1_q_t2, gm_q_t)
                    tla.set_flag(q_l1_ready)
                    tla.wait_flag(q_l1_ready)
                    l0a_q = tla.make_tensor_like(l0a, l1_q_t2, tla.arch.zN)
                    tla.wait_flag(l0a_avail)
                    tla.copy(l0a_q, l1_q_t2)
                    tla.set_flag(l0a_ready)
                    gm_k_t = tla.make_tensor(
                        gm_k.ptr + kv_off,
                        tla.make_layout(
                            tla.make_shape(D, PAGE),
                            tla.make_stride(1, hkv * D),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    l1_k_t = tla.make_tensor_like(l1_k, gm_k_t, tla.arch.nZ)
                    tla.wait_flag(k_l0b_ready)
                    tla.copy(l1_k_t, gm_k_t)
                    tla.set_flag(k_l1_ready)
                    l0b_k = tla.make_tensor_like(l0b, l1_k_t)
                    tla.wait_flag(l0b_avail)
                    tla.wait_flag(k_l1_ready)
                    tla.copy(l0b_k, l1_k_t)
                    tla.set_flag(l0b_ready)
                    tla.set_flag(k_l0b_ready)
                    ub_s_full = tla.make_tensor(
                        ub_s,
                        tla.make_layout(
                            tla.make_shape(Q_BLOCK, PAGE),
                            tla.make_stride(PAGE, 1),
                        ),
                    )
                    l0c_s_t = tla.make_tensor_like(l0c_s, ub_s_full, tla.arch.L0Clayout)
                    tla.wait_flag(l0a_ready)
                    tla.wait_flag(l0b_ready)
                    tla.wait_flag(l0cs_avail)
                    tla.mmad(l0c_s_t, l0a_q, l0b_k, init_c=True)
                    tla.set_flag(l0a_avail)
                    tla.set_flag(l0b_avail)
                    tla.set_flag(q_l0a_ready)
                    tla.cross_core_wait_flag(s_cross, tla.arch.FIX)
                    tla.set_flag(l0cs_ready)
                    tla.wait_flag(l0cs_ready)
                    ub_s_view = tla.make_tensor(
                        ub_s,
                        tla.make_layout(
                            tla.make_shape(pipe_rows, PAGE), tla.make_stride(PAGE, 1)
                        ),
                    )
                    tla.copy(
                        ub_s_view,
                        l0c_s_t,
                        tla.params.CopyL0C2DstParams(
                            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                        ),
                    )
                    tla.set_flag(l0cs_avail)
                    tla.cross_core_set_flag(s_cross, tla.arch.FIX)

                # ---- online softmax + P quantize (AIV halves rows) ----
                with tla.vector():
                    sub = tla.arch.sub_block_idx()
                    m_half = (pipe_rows + 1) // 2
                    m_sub = m_half if sub == 0 else pipe_rows - m_half
                    if m_sub > 0:
                        tla.cross_core_wait_flag(s_cross, tla.arch.VECTOR)
                        tla.wait_flag(ub_p_avail)
                        tla.wait_flag(sc_free)
                        gm_qs = tla.make_tensor(
                            gm_qscale.ptr + q_start + sub * m_half,
                            tla.make_layout(tla.make_shape(m_sub), tla.make_stride(1)),
                        )
                        gm_ks = tla.make_tensor(
                            gm_kscale.ptr + phys * PAGE,
                            tla.make_layout(
                                tla.make_shape(1, PAGE),
                                tla.make_stride(PAGE, 1),
                            ),
                        )
                        ub_ks_t = tla.make_tensor(
                            ub_ks,
                            tla.make_layout(
                                tla.make_shape(1, PAGE),
                                tla.make_stride(PAGE, 1),
                            ),
                        )
                        ub_qs_t = tla.make_tensor(
                            ub_qs,
                            tla.make_layout(tla.make_shape(m_sub), tla.make_stride(1)),
                        )
                        tla.copy(ub_qs_t, gm_qs)
                        tla.copy(ub_ks_t, gm_ks)
                        tla.set_flag(sc_ready)
                        tla.wait_flag(sc_ready)
                        ub_s_sub = tla.make_tensor(
                            ub_s,
                            tla.make_layout(
                                tla.make_shape(m_sub, PAGE),
                                tla.make_stride(PAGE, 1),
                            ),
                        )
                        ub_p_view = tla.make_tensor(
                            ub_p,
                            tla.make_layout(
                                tla.make_shape(HALF, PAGE),
                                tla.make_stride(PAGE, 1),
                            ),
                        )
                        ub_lm_t = tla.make_tensor(
                            ub_lm,
                            tla.make_layout(tla.make_shape(HALF), tla.make_stride(1)),
                        )
                        ub_ls_t = tla.make_tensor(
                            ub_ls,
                            tla.make_layout(tla.make_shape(HALF), tla.make_stride(1)),
                        )
                        ub_dm_t = tla.make_tensor(
                            ub_dm,
                            tla.make_layout(tla.make_shape(HALF), tla.make_stride(1)),
                        )
                        first = 1 if p == 0 else 0
                        _sf_tile(
                            ub_s_sub,
                            ub_p_view,
                            ub_qs_t,
                            ub_ks_t,
                            ub_lm_t,
                            ub_ls_t,
                            ub_dm_t,
                            m_sub,
                            kv_len,
                            qb_start + (kv_len - q_len_b) + sub * m_half,
                            page_base,
                            softmax_scale,
                            p_scale,
                            causal,
                            full_vis,
                            first,
                        )
                        tla.set_flag(sc_free)
                        tla.cross_core_set_flag(s_cross, tla.arch.VECTOR)
                        tla.set_flag(ub_p_ready)
                        tla.wait_flag(ub_p_ready)
                        l1_p_ref = tla.make_tensor(
                            ub_s,
                            tla.make_layout(
                                tla.make_shape(Q_BLOCK, PAGE),
                                tla.make_stride(PAGE, 1),
                            ),
                        )
                        l1_p_full = tla.make_tensor_like(l1_p, l1_p_ref, tla.arch.zN)
                        l1_p_tile = tla.tile_view(
                            l1_p_full,
                            tla.make_shape(m_half, PAGE),
                            tla.make_coord(sub, 0),
                        )
                        ub_p_send = tla.tile_view(
                            ub_p_view,
                            tla.make_shape(m_sub, PAGE),
                            tla.make_coord(0, 0),
                        )
                        tla.cross_core_wait_flag(p_cross, tla.arch.MTE3)
                        tla.copy(l1_p_tile, ub_p_send)
                        tla.set_flag(ub_p_avail)
                        tla.cross_core_set_flag(p_cross, tla.arch.MTE3)
                    else:
                        tla.cross_core_wait_flag(s_cross, tla.arch.VECTOR)
                        tla.cross_core_set_flag(s_cross, tla.arch.VECTOR)

                # ---- PV: P x V -> O (FP8 MMAD, FP32 accumulate) ----
                with tla.cube():
                    gm_v_t = tla.make_tensor(
                        gm_v.ptr + kv_off,
                        tla.make_layout(
                            tla.make_shape(PAGE, D),
                            tla.make_stride(hkv * D, 1),
                        ),
                    )
                    l1_v_t = tla.make_tensor_like(l1_v, gm_v_t, tla.arch.zN)
                    tla.wait_flag(v_l0b_ready)
                    tla.copy(l1_v_t, gm_v_t)
                    tla.set_flag(v_l1_ready)
                    l0b_v = tla.make_tensor_like(l0b, l1_v_t, tla.arch.nZ)
                    tla.wait_flag(l0b_avail)
                    tla.wait_flag(v_l1_ready)
                    tla.copy(l0b_v, l1_v_t)
                    tla.set_flag(l0b_ready)
                    tla.set_flag(v_l0b_ready)
                    tla.cross_core_wait_flag(p_cross, tla.arch.MTE1)
                    l1_p_ref2 = tla.make_tensor(
                        ub_s,
                        tla.make_layout(
                            tla.make_shape(Q_BLOCK, PAGE),
                            tla.make_stride(PAGE, 1),
                        ),
                    )
                    l1_p_full2 = tla.make_tensor_like(l1_p, l1_p_ref2, tla.arch.zN)
                    l0a_p = tla.make_tensor_like(l0a, l1_p_full2, tla.arch.zN)
                    tla.wait_flag(l0a_avail)
                    tla.copy(l0a_p, l1_p_full2)
                    tla.set_flag(l0a_ready)
                    tla.cross_core_set_flag(p_cross, tla.arch.MTE1)
                    ub_o_full = tla.make_tensor(
                        ub_o,
                        tla.make_layout(
                            tla.make_shape(Q_BLOCK, D), tla.make_stride(D, 1)
                        ),
                    )
                    l0c_o_t = tla.make_tensor_like(l0c_o, ub_o_full, tla.arch.L0Clayout)
                    tla.wait_flag(l0a_ready)
                    tla.wait_flag(l0b_ready)
                    tla.wait_flag(l0co_avail)
                    tla.mmad(l0c_o_t, l0a_p, l0b_v, init_c=True)
                    tla.set_flag(l0a_avail)
                    tla.set_flag(l0b_avail)
                    tla.cross_core_wait_flag(o_cross, tla.arch.FIX)
                    tla.set_flag(l0co_ready)
                    tla.wait_flag(l0co_ready)
                    ub_o_view = tla.make_tensor(
                        ub_o,
                        tla.make_layout(
                            tla.make_shape(pipe_rows, D), tla.make_stride(D, 1)
                        ),
                    )
                    tla.copy(
                        ub_o_view,
                        l0c_o_t,
                        tla.params.CopyL0C2DstParams(
                            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                        ),
                    )
                    tla.set_flag(l0co_avail)
                    tla.cross_core_set_flag(o_cross, tla.arch.FIX)

                # ---- rescale: acc = acc*dm + O ----
                with tla.vector():
                    sub_r = tla.arch.sub_block_idx()
                    m_half_r = (pipe_rows + 1) // 2
                    m_sub_r = m_half_r if sub_r == 0 else pipe_rows - m_half_r
                    if m_sub_r > 0:
                        tla.cross_core_wait_flag(o_cross, tla.arch.VECTOR)
                        ub_o_sub = tla.make_tensor(
                            ub_o,
                            tla.make_layout(
                                tla.make_shape(m_sub_r, D),
                                tla.make_stride(D, 1),
                            ),
                        )
                        ub_acc_sub = tla.make_tensor(
                            ub_acc,
                            tla.make_layout(
                                tla.make_shape(m_sub_r, D),
                                tla.make_stride(D, 1),
                            ),
                        )
                        ub_dm_r = tla.make_tensor(
                            ub_dm,
                            tla.make_layout(tla.make_shape(HALF), tla.make_stride(1)),
                        )
                        with tla.vec.func(mode="simd"):
                            full_r = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            for ri in tla.range(m_sub_r):
                                dmv = tla.tile_view(
                                    ub_dm_r,
                                    tla.make_shape(1),
                                    tla.make_coord(ri),
                                ).load(
                                    params=tla.params.NormalLoadParams(
                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                    )
                                )
                                r_ov0 = tla.tile_view(
                                    ub_o_sub,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(ri, 0),
                                ).load()
                                r_av0 = r_ov0
                                if p != 0:
                                    r_av0 = tla.add(
                                        tla.mul(
                                            tla.tile_view(
                                                ub_acc_sub,
                                                tla.make_shape(1, 64),
                                                tla.make_coord(ri, 0),
                                            ).load(),
                                            dmv,
                                            mask=full_r,
                                        ),
                                        r_ov0,
                                        mask=full_r,
                                    )
                                tla.tile_view(
                                    ub_acc_sub,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(ri, 0),
                                ).store(r_av0, mask=full_r)
                                r_ov1 = tla.tile_view(
                                    ub_o_sub,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(ri, 1),
                                ).load()
                                r_av1 = r_ov1
                                if p != 0:
                                    r_av1 = tla.add(
                                        tla.mul(
                                            tla.tile_view(
                                                ub_acc_sub,
                                                tla.make_shape(1, 64),
                                                tla.make_coord(ri, 1),
                                            ).load(),
                                            dmv,
                                            mask=full_r,
                                        ),
                                        r_ov1,
                                        mask=full_r,
                                    )
                                tla.tile_view(
                                    ub_acc_sub,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(ri, 1),
                                ).store(r_av1, mask=full_r)
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )
                        tla.cross_core_set_flag(o_cross, tla.arch.VECTOR)
                    else:
                        tla.cross_core_wait_flag(o_cross, tla.arch.VECTOR)
                        tla.cross_core_set_flag(o_cross, tla.arch.VECTOR)

            with tla.cube():
                tla.cross_core_set_flag(task_c_done, tla.arch.CUBE)
            # ---- final normalize + FP16 output ----
            with tla.vector():
                sub_f = tla.arch.sub_block_idx()
                m_half_f = (rows + 1) // 2
                m_sub_f = m_half_f if sub_f == 0 else rows - m_half_f
                if m_sub_f > 0:
                    ub_acc_f = tla.make_tensor(
                        ub_acc,
                        tla.make_layout(tla.make_shape(HALF, D), tla.make_stride(D, 1)),
                    )
                    ub_ls_f = tla.make_tensor(
                        ub_ls,
                        tla.make_layout(tla.make_shape(HALF), tla.make_stride(1)),
                    )
                    ub_out_f = tla.make_tensor(
                        ub_out,
                        tla.make_layout(tla.make_shape(HALF, D), tla.make_stride(D, 1)),
                    )
                    cast_z = tla.params.CastParams(
                        reg_slot=tla.params.RegSlot.ZERO,
                        sat_mode=tla.params.SatMode.NOSAT,
                        round_mode=tla.params.RoundMode.CAST_ROUND,
                    )
                    cast_o = tla.params.CastParams(
                        reg_slot=tla.params.RegSlot.ONE,
                        sat_mode=tla.params.SatMode.NOSAT,
                        round_mode=tla.params.RoundMode.CAST_ROUND,
                    )
                    with tla.vec.func(mode="simd"):
                        full_f = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        onef = tla.full(1.0, tla.Float32)
                        vs = gm_vscale[kv_head]
                        for fi in tla.range(m_sub_f):
                            lsv = tla.tile_view(
                                ub_ls_f,
                                tla.make_shape(1),
                                tla.make_coord(fi),
                            ).load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            den = tla.mul(lsv, p_scale, mask=full_f)
                            den_safe = tla.where(
                                tla.cmp(den, 0.0, mode="gt"), den, onef
                            )
                            inv = tla.div(onef, den_safe, mask=full_f)
                            scale_f = tla.mul(inv, vs, mask=full_f)
                            f_av0 = tla.tile_view(
                                ub_acc_f,
                                tla.make_shape(1, 64),
                                tla.make_coord(fi, 0),
                            ).load()
                            f_ov0 = tla.mul(f_av0, scale_f, mask=full_f)
                            f_be0, f_bo0 = tla.deinterleave(f_ov0, f_ov0)
                            f_pack0 = tla.bitwise_or(
                                f_be0.to(tla.Float16, cast_z, mask=full_f),
                                f_bo0.to(tla.Float16, cast_o, mask=full_f),
                            )
                            tla.tile_view(
                                ub_out_f,
                                tla.make_shape(1, 64),
                                tla.make_coord(fi, 0),
                            ).store(f_pack0)
                            f_av1 = tla.tile_view(
                                ub_acc_f,
                                tla.make_shape(1, 64),
                                tla.make_coord(fi, 1),
                            ).load()
                            f_ov1 = tla.mul(f_av1, scale_f, mask=full_f)
                            f_be1, f_bo1 = tla.deinterleave(f_ov1, f_ov1)
                            f_pack1 = tla.bitwise_or(
                                f_be1.to(tla.Float16, cast_z, mask=full_f),
                                f_bo1.to(tla.Float16, cast_o, mask=full_f),
                            )
                            tla.tile_view(
                                ub_out_f,
                                tla.make_shape(1, 64),
                                tla.make_coord(fi, 1),
                            ).store(f_pack1)
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE,
                            tla.params.MemType.VEC_LOAD,
                        )
                    tla.set_flag(out_ready)
                    tla.wait_flag(out_ready)
                    gm_o = tla.make_tensor(
                        gm_out.ptr + (h * out_tokens + q_start + sub_f * m_half_f) * D,
                        tla.make_layout(
                            tla.make_shape(m_sub_f, D),
                            tla.make_stride(D, 1),
                        ),
                    )
                    ub_out_g = tla.make_tensor(
                        ub_out,
                        tla.make_layout(
                            tla.make_shape(m_sub_f, D), tla.make_stride(D, 1)
                        ),
                    )
                    tla.copy(gm_o, ub_out_g)
            with tla.vector():
                tla.cross_core_set_flag(task_v_done, tla.arch.VECTOR)

    with tla.cube():
        tla.wait_flag(q_l0a_ready)
        tla.wait_flag(k_l0b_ready)
        tla.wait_flag(v_l0b_ready)
        tla.wait_flag(l0a_avail)
        tla.wait_flag(l0b_avail)
        tla.wait_flag(l0cs_avail)
        tla.wait_flag(l0co_avail)
        tla.cross_core_wait_flag(s_cross, tla.arch.FIX)
        tla.cross_core_wait_flag(o_cross, tla.arch.FIX)
        tla.cross_core_wait_flag(task_v_done, tla.arch.CUBE)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla.wait_flag(ub_p_avail)
        tla.wait_flag(sc_free)
        tla.cross_core_wait_flag(p_cross, tla.arch.MTE3)
        tla.cross_core_wait_flag(task_c_done, tla.arch.VECTOR)
        tla.pipe_barrier(tla.pipes.ALL)


import math

import torch
import torch_npu  # noqa: F401
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


HEAD_DIM = 128
PAGE_SIZE = 128
M_BLOCK = 64
_ARTIFACT_CACHE = {}


def _tla_tensor(buf, layout_tag, element_type=None):
    """torch NPU buffer -> tla.Tensor; FP8 is carried via an int8 view."""
    if element_type is not None:
        buf = buf.view(torch.int8)
    buf = buf.contiguous()
    tensor = from_dlpack(buf, layout_tag=layout_tag, element_type=element_type)
    tensor._torch_storage = buf
    return tensor


def _cube_core_num(device=0):
    props = torch.npu.get_device_properties(int(device))
    return int(props.cube_core_num)


def _compile_cached(key, kernel, args, options):
    artifact = _ARTIFACT_CACHE.get(key)
    if artifact is None:
        artifact = tla.compile(kernel, *args, options=options)
        _ARTIFACT_CACHE[key] = artifact
    return artifact


def _ceil_div(a, b):
    return (a + b - 1) // b


def run(
    q,
    q_scale,
    k_cache,
    k_scale,
    v_cache,
    v_scale,
    block_table,
    seq_lens_q,
    seq_lens_kv,
    softmax_scale,
    causal,
    p_scale,
):
    """Launch the FP8 fully-quantized paged GQA flash attention kernel."""
    if q.ndim != 3 or q.shape[2] != HEAD_DIM:
        raise ValueError("q must be [T, H_q, 128]")
    if (
        k_cache.ndim != 4
        or k_cache.shape[1] != PAGE_SIZE
        or k_cache.shape[3] != HEAD_DIM
    ):
        raise ValueError("k_cache must be [num_blocks, 128, H_kv, 128]")
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache must match k_cache shape")
    if (
        q.dtype != torch.uint8
        or k_cache.dtype != torch.uint8
        or v_cache.dtype != torch.uint8
    ):
        raise TypeError("Q/K/V FP8 data must be carried as uint8 E4M3FN bit patterns")
    if q_scale.ndim != 1 or q_scale.numel() != q.shape[0]:
        raise ValueError("q_scale must have one value per query token")
    if q_scale.dtype != torch.float32:
        raise TypeError("q_scale must be float32")
    if k_scale.ndim != 2 or k_scale.shape[1] != PAGE_SIZE:
        raise ValueError("k_scale must be [num_blocks, 128]")
    if v_scale.ndim != 1:
        raise ValueError("v_scale must be [H_kv]")
    if k_scale.dtype != torch.float32 or v_scale.dtype != torch.float32:
        raise TypeError("k_scale and v_scale must be float32")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("block_table must be a two-dimensional int32 tensor")
    if seq_lens_q.ndim != 1 or seq_lens_q.dtype != torch.int32:
        raise ValueError("seq_lens_q must be a one-dimensional int32 tensor")
    if seq_lens_kv.ndim != 1 or seq_lens_kv.dtype != torch.int32:
        raise ValueError("seq_lens_kv must be a one-dimensional int32 tensor")
    if not math.isfinite(float(softmax_scale)) or softmax_scale <= 0.0:
        raise ValueError("softmax_scale must be finite and positive")
    if not math.isfinite(float(p_scale)) or p_scale <= 0.0:
        raise ValueError("p_scale must be finite and positive")

    device = q.device
    total_tokens, num_q_heads, _ = q.shape
    num_kv_heads = k_cache.shape[2]
    num_blocks = k_cache.shape[0]
    batch = block_table.shape[0]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if k_scale.shape[0] != num_blocks:
        raise ValueError("k_scale must have one row per K/V cache block")
    if v_scale.numel() != num_kv_heads:
        raise ValueError("v_scale must have one value per KV head")
    inputs = (
        q_scale,
        k_cache,
        k_scale,
        v_cache,
        v_scale,
        block_table,
        seq_lens_q,
        seq_lens_kv,
    )
    if any(tensor.device != device for tensor in inputs):
        raise ValueError("all tensors must be on the same device as q")

    q_lens = [int(x) for x in seq_lens_q.to(torch.int64).tolist()]
    kv_lens = [int(x) for x in seq_lens_kv.to(torch.int64).tolist()]
    if len(q_lens) != batch or len(kv_lens) != batch:
        raise ValueError("seq_lens must have batch entries")
    if any(length < 0 for length in q_lens) or any(length < 0 for length in kv_lens):
        raise ValueError("sequence lengths must be non-negative")
    if sum(q_lens) != total_tokens:
        raise ValueError("sum(seq_lens_q) must equal T")
    max_blocks = int(block_table.shape[1])
    if any(length > max_blocks * PAGE_SIZE for length in kv_lens):
        raise ValueError("seq_lens_kv exceeds block_table page capacity")
    if any(kv_lens):
        if num_blocks == 0:
            raise ValueError("K/V cache must contain referenced pages")
    else:
        return torch.zeros(
            (total_tokens, num_q_heads, HEAD_DIM),
            dtype=torch.float16,
            device=device,
        )
    if total_tokens == 0 or batch == 0:
        return torch.empty(
            (total_tokens, num_q_heads, HEAD_DIM), dtype=torch.float16, device=device
        )

    # Pad Q / q_scale by one M block so every per-head strided GM view is in-bounds.
    q_head = q.permute(1, 0, 2).contiguous().view(torch.int8)
    q_pad = torch.zeros(
        (num_q_heads, total_tokens + M_BLOCK, HEAD_DIM),
        dtype=torch.int8,
        device=device,
    )
    q_pad[:, :total_tokens] = q_head
    q_scale_pad = torch.ones(
        (total_tokens + M_BLOCK,), dtype=torch.float32, device=device
    )
    q_scale_pad[:total_tokens] = q_scale

    cu_q = torch.zeros((batch + 1,), dtype=torch.int32, device=device)
    for i, length in enumerate(q_lens):
        cu_q[i + 1] = cu_q[i] + length

    max_q_blocks = max(_ceil_div(length, M_BLOCK) for length in q_lens)
    out_head = torch.zeros(
        (num_q_heads, total_tokens, HEAD_DIM), dtype=torch.float16, device=device
    )

    t_q = _tla_tensor(q_pad.contiguous().view(-1), tla.arch.RowMajor, tla.Float8E4M3FN)
    t_qscale = _tla_tensor(q_scale_pad, tla.arch.RowMajor)
    t_k = _tla_tensor(
        k_cache.contiguous().view(-1), tla.arch.RowMajor, tla.Float8E4M3FN
    )
    t_kscale = _tla_tensor(k_scale.contiguous().view(-1), tla.arch.RowMajor)
    t_v = _tla_tensor(
        v_cache.contiguous().view(-1), tla.arch.RowMajor, tla.Float8E4M3FN
    )
    t_vscale = _tla_tensor(v_scale.contiguous().view(-1), tla.arch.RowMajor)
    t_bt = _tla_tensor(block_table.contiguous().view(-1), tla.arch.RowMajor)
    t_cuq = _tla_tensor(cu_q, tla.arch.RowMajor)
    t_seqq = _tla_tensor(seq_lens_q.contiguous().view(-1), tla.arch.RowMajor)
    t_seqkv = _tla_tensor(seq_lens_kv.contiguous().view(-1), tla.arch.RowMajor)
    t_out = _tla_tensor(out_head.view(-1), tla.arch.RowMajor)

    args = (
        t_q,
        t_qscale,
        t_k,
        t_kscale,
        t_v,
        t_vscale,
        t_bt,
        t_cuq,
        t_seqq,
        t_seqkv,
        t_out,
        tla.Int32(num_q_heads),
        tla.Int32(num_kv_heads),
        tla.Int32(batch),
        tla.Int32(max_blocks),
        tla.Int32(max_q_blocks),
        tla.Float32(float(softmax_scale)),
        tla.Int32(int(bool(causal))),
        tla.Float32(float(p_scale)),
        tla.Int32(total_tokens + M_BLOCK),
        tla.Int32(total_tokens),
    )
    cube_num = _cube_core_num(device.index or 0)
    key = (
        "fafq",
        tuple(q.shape),
        tuple(k_cache.shape),
        tuple(block_table.shape),
        tuple(q_lens),
        tuple(kv_lens),
        float(softmax_scale),
        int(bool(causal)),
        float(p_scale),
        str(q.dtype),
        str(k_cache.dtype),
    )
    artifact = _compile_cached(key, fafq_kernel, args, "--npu-arch 3510")
    artifact(*args, block_num=cube_num)
    return out_head.permute(1, 0, 2).contiguous()
