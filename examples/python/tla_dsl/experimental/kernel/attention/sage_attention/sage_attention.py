# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sage-style causal attention for Ascend 950: INT8 QK, FP8 PV.

One CATLASS DSL AIC/AIV mixed kernel fuses QK GEMM, online softmax in
FP32, E4M3FN P quantization and PV GEMM. Q/K are pre-quantized INT8 with
per-128 query and per-64 key scales; V is pre-quantized E4M3FN with a
per-channel scale. Every pipeline stage is TLA (register-level SIMD
included); no AscendC extern source is compiled. Dependencies: CATLASS
DSL, PyTorch, and torch_npu.
"""

import catlass.tla as tla

_EXECUTORS = {}


def _sage_score_row(
    s_src, s_scr, kss, qpos, colsvec, key_start, row, neg, mask32, cast
):
    """Scale and mask one key row of INT32 scores, then stash it as FP32."""
    raw = tla.tile_view(s_src, tla.make_shape(1, 64), tla.make_coord(row, 0)).load()
    v = tla.mul(raw.to(tla.Float32, cast), kss, mask=mask32)
    causal = tla.cmp(qpos, key_start + row, "ge")
    blockok = tla.cmp(tla.arange(row, dtype=tla.Int32), colsvec, "lt")
    v = tla.where(tla.bitwise_and(causal, blockok), v, neg)
    tla.tile_view(s_scr, tla.make_shape(1, 64), tla.make_coord(row, 0)).store(
        v, mask=mask32
    )
    return v


def _sage_pack_prob_group(
    s_scr, p0v, p1v, group, maximum, denom, traits, mask32, mask256, m32, m_hi
):
    """Exponentiate four key rows, accumulate the denominator, then quantize to
    E4M3 and store the two 32-lane planes of one key group.

    Four rows share one 256-byte register: cast row q into byte q of every
    32-bit slot, then deinterleave twice to bring phase q (row q, all columns)
    into the low lanes. Lanes 0..31 are the first plane, 32..63 the second.
    """
    packed = None
    for q in range(4):
        row = group * 4 + q
        val = tla.tile_view(s_scr, tla.make_shape(1, 64), tla.make_coord(row, 0)).load()
        e = tla.exp(tla.sub(val, maximum, mask=mask32), mask=mask32)
        denom = tla.add(denom, e, mask=mask32)
        e = tla.mul(e, 448.0, mask=mask32)
        f8 = e.to(tla.Float8E4M3FN, traits[q]).bitcast(tla.Int8)
        if packed is None:
            packed = f8
        else:
            packed = tla.bitwise_or(packed, f8, mask=mask256)
    even, odd = tla.deinterleave(packed, packed)
    ph0, ph2 = tla.deinterleave(even, even)
    ph1, ph3 = tla.deinterleave(odd, odd)
    for q in range(4):
        row = group * 4 + q
        phase = (ph0, ph1, ph2, ph3)[q]
        tla.tile_view(p0v, tla.make_shape(1, 64), tla.make_coord(row, 0)).store(
            phase, mask=m32
        )
        tla.tile_view(p1v, tla.make_shape(1, 64), tla.make_coord(row, 0)).store(
            phase, mask=m_hi
        )
    return denom


@tla.kernel
def sage_attention_kernel(
    q: tla.Tensor,
    k: tla.Tensor,
    v: tla.Tensor,
    out: tla.Tensor,
    qs: tla.Tensor,
    ks: tla.Tensor,
    vs: tla.Tensor,
    cq: tla.Tensor,
    ck: tla.Tensor,
    cqs: tla.Tensor,
    cks: tla.Tensor,
    lse: tla.Tensor,
    batch: tla.Int32,
    hq: tla.Int32,
    hk: tla.Int32,
    nq: tla.Int32,
    nk: tla.Int32,
    dim: tla.Constexpr[int],
    packed: tla.Constexpr[bool],
    nhd: tla.Constexpr[bool],
    want_lse: tla.Constexpr[bool],
):
    bm = 128
    # Eight per64 K scales share one online-softmax/PV tile. L0 GEMMs
    # use two256-key subtiles; P is quantized once over all512 keys.
    bn = 512
    kc = 256
    half = bm // 2
    pv_dtype = v.ptr.dtype
    out_dtype = out.ptr.dtype
    cast = tla.params.CastParams()
    bcast = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
    us = tla.params.UnalignStoreParams()
    split = tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M)
    split_s = tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N)
    q1p = tla.allocate(bm * dim, tla.Int8, tla.AddressSpace.l1, 512)
    k1p = tla.allocate(bn * dim, tla.Int8, tla.AddressSpace.l1, 512)
    v1p = tla.allocate(bn * dim, pv_dtype, tla.AddressSpace.l1, 512)
    p1p = tla.allocate(2 * bm * bn, pv_dtype, tla.AddressSpace.l1, 512)
    q0p = tla.allocate(bm * dim, tla.Int8, tla.AddressSpace.l0b, 512)
    k0p = tla.allocate(kc * dim, tla.Int8, tla.AddressSpace.l0a, 512)
    p0p = tla.allocate(bm * kc, pv_dtype, tla.AddressSpace.l0a, 512)
    v0p = tla.allocate(kc * dim, pv_dtype, tla.AddressSpace.l0b, 512)
    # Independent QK chunks overlap MMAD and Fixpipe. PV reuses retired slot0.
    s0p = tla.allocate(2 * bm * kc, tla.Int32, tla.AddressSpace.l0c, 1024)
    o0p = tla.recast_ptr(s0p, dtype=tla.Float32)
    # SPLIT_N writes each AIV's64 columns with physical stride64.
    sp = tla.allocate(half * bn, tla.Int32, tla.AddressSpace.ub, 512)
    # P MTE3 completes before V-tail zeroing can reuse this scratch.
    pp = tla.allocate((bn + 2) * half, tla.Int8, tla.AddressSpace.ub, 512)
    vzp = pp
    # Each score is converted in place only after its INT32 load.
    bits_ptr = tla.recast_ptr(sp, dtype=tla.Float32)
    vs_ptr = tla.allocate(dim, tla.Float32, tla.AddressSpace.ub, 256)
    op = tla.allocate(half * dim, tla.Float32, tla.AddressSpace.ub, 512)
    ap = tla.allocate(half * dim, tla.Float32, tla.AddressSpace.ub, 512)
    # Final normalization consumes accumulator rows in forward order.
    yp = tla.recast_ptr(ap, dtype=out_dtype)
    mp = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 512)
    lp = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 512)
    rp = tla.allocate(2 * half, tla.Float32, tla.AddressSpace.ub, 512)
    ep = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 512)
    qready = tla.cross_flag("qready")
    pready = tla.cross_flag("pready")
    pready_alt = tla.cross_flag("pready_alt")
    oready = tla.cross_flag("oready")
    vzero_ready = tla.cross_flag("vzero_ready")
    # In-core buffer ownership is expressed with mutexes; cross-core handoff
    # stays on cross_flag. Cube and vector sides keep separate mutex id spaces.
    m_q1 = tla.mutex("q1", id=0)
    m_q0 = tla.mutex("q0", id=1)
    m_k1 = tla.mutex("k1", id=2)
    m_k0 = tla.mutex("k0", id=3)
    m_v1 = tla.mutex("v1", id=4)
    m_v0 = tla.mutex("v0", id=5)
    m_p0 = tla.mutex("p0", id=6)
    m_s0 = tla.mutex("s0", id=7)
    m_s1 = tla.mutex("s1", id=8)
    m_vscale = tla.mutex("vscale", id=0)
    m_pp = tla.mutex("pp", id=1)
    m_out = tla.mutex("out", id=2)
    q1ref = tla.make_tensor(
        q1p,
        tla.make_layout(
            tla.make_shape(dim, bm),
            tla.make_stride(1, dim),
            layoutTag=tla.arch.ColumnMajor,
        ),
    )
    q1 = tla.make_tensor_like(q1p, q1ref, tla.arch.nZ)
    k1ref = tla.make_tensor(
        k1p, tla.make_layout(tla.make_shape(bn, dim), tla.make_stride(dim, 1))
    )
    k1 = tla.make_tensor_like(k1p, k1ref, tla.arch.zN)
    v1ref = tla.make_tensor(
        v1p, tla.make_layout(tla.make_shape(bn, dim), tla.make_stride(dim, 1))
    )
    v1 = tla.make_tensor_like(v1p, v1ref, tla.arch.zN)
    p1ref = tla.make_tensor(
        p1p, tla.make_layout(tla.make_shape(bn, bm), tla.make_stride(bm, 1))
    )
    p1 = tla.make_tensor_like(p1p, p1ref, tla.arch.zN)
    p1t_ref = tla.make_tensor(
        p1p,
        tla.make_layout(
            tla.make_shape(bm, bn),
            tla.make_stride(1, bm),
            layoutTag=tla.arch.ColumnMajor,
        ),
    )
    p1t = tla.make_tensor_like(p1p, p1t_ref, tla.arch.nZ)
    p1storagep = tla.recast_ptr(p1p, dtype=tla.Float16)
    p1storage_ref = tla.make_tensor(
        p1storagep,
        tla.make_layout(tla.make_shape(bn, bm // 2), tla.make_stride(bm // 2, 1)),
    )
    p1storage = tla.make_tensor_like(p1storagep, p1storage_ref, tla.arch.zN)
    v1storagep = tla.recast_ptr(v1p, dtype=tla.Float16)
    v1storage_ref = tla.make_tensor(
        v1storagep,
        tla.make_layout(tla.make_shape(bn, dim // 2), tla.make_stride(dim // 2, 1)),
    )
    v1storage = tla.make_tensor_like(v1storagep, v1storage_ref, tla.arch.zN)
    q0 = tla.make_tensor_like(q0p, q1, tla.arch.nZ)
    k0ref = tla.make_tensor(
        k0p, tla.make_layout(tla.make_shape(kc, dim), tla.make_stride(dim, 1))
    )
    k0 = tla.make_tensor_like(k0p, k0ref, tla.arch.zN)
    p0ref = tla.make_tensor(
        p0p,
        tla.make_layout(
            tla.make_shape(bm, kc),
            tla.make_stride(1, bm),
            layoutTag=tla.arch.ColumnMajor,
        ),
    )
    p0 = tla.make_tensor_like(p0p, p0ref, tla.arch.zN)
    v0ref = tla.make_tensor(
        v0p, tla.make_layout(tla.make_shape(kc, dim), tla.make_stride(dim, 1))
    )
    v0 = tla.make_tensor_like(v0p, v0ref, tla.arch.nZ)
    sref = tla.make_tensor(
        s0p, tla.make_layout(tla.make_shape(kc, bm), tla.make_stride(bm, 1))
    )
    s0 = tla.make_tensor_like(s0p, sref, tla.arch.L0Clayout)
    oref = tla.make_tensor(
        o0p, tla.make_layout(tla.make_shape(bm, dim), tla.make_stride(dim, 1))
    )
    o0 = tla.make_tensor_like(o0p, oref, tla.arch.L0Clayout)
    ofix = tla.make_tensor(
        op, tla.make_layout(tla.make_shape(bm, dim), tla.make_stride(dim, 1))
    )
    pp16 = tla.recast_ptr(pp, dtype=tla.Float16)
    pu16ref = tla.make_tensor(
        pp16,
        tla.make_layout(
            tla.make_shape(bn + 1, half // 2), tla.make_stride(half // 2, 1)
        ),
    )
    pu16 = tla.make_tensor_like(pp16, pu16ref, tla.arch.zNUnAlign)
    vz16p = tla.recast_ptr(vzp, dtype=tla.Float16)
    vz16ref = tla.make_tensor(
        vz16p,
        tla.make_layout(
            tla.make_shape(bn // 2 + 1, dim // 2), tla.make_stride(dim // 2, 1)
        ),
    )
    vz16 = tla.make_tensor_like(vz16p, vz16ref, tla.arch.zNUnAlign)
    vscale = tla.make_tensor(
        vs_ptr, tla.make_layout(tla.make_shape(1, dim), tla.make_stride(dim, 1))
    )
    ou = tla.make_tensor(
        op, tla.make_layout(tla.make_shape(half, dim), tla.make_stride(dim, 1))
    )
    acc = tla.make_tensor(
        ap, tla.make_layout(tla.make_shape(half, dim), tla.make_stride(dim, 1))
    )
    yu = tla.make_tensor(
        yp, tla.make_layout(tla.make_shape(half, dim), tla.make_stride(dim, 1))
    )
    maxima = tla.make_tensor(
        mp, tla.make_layout(tla.make_shape(half), tla.make_stride(1))
    )
    sums = tla.make_tensor(
        lp, tla.make_layout(tla.make_shape(half), tla.make_stride(1))
    )
    rescale = tla.make_tensor(
        rp, tla.make_layout(tla.make_shape(half), tla.make_stride(1))
    )
    elu = tla.make_tensor(ep, tla.make_layout(tla.make_shape(half), tla.make_stride(1)))
    # INT32 QK scores, their FP32 in-place rescale, and the two 32-lane FP8
    # probability planes consumed by the UB to L1 copy.
    s_src = tla.make_tensor(
        sp, tla.make_layout(tla.make_shape(bn, half), tla.make_stride(half, 1))
    )
    s_scr = tla.make_tensor(
        bits_ptr, tla.make_layout(tla.make_shape(bn, half), tla.make_stride(half, 1))
    )
    pf0 = tla.make_tensor(
        pp, tla.make_layout(tla.make_shape(bn, 64), tla.make_stride(32, 1))
    )
    pf1 = tla.make_tensor(
        pp + ((bn + 1) * 32 - 32),
        tla.make_layout(tla.make_shape(bn, 64), tla.make_stride(32, 1)),
    )
    blocks = (nq + bm - 1) // bm
    total = batch * hq * blocks
    with tla.vector():
        tla.cross_core_set_flag(qready, tla.arch.VECTOR)
        tla.cross_core_set_flag(oready, tla.arch.VECTOR)
    with tla.cube():
        tla.cross_core_set_flag(pready, tla.arch.MTE1)
        tla.cross_core_set_flag(pready_alt, tla.arch.MTE1)
    for task in tla.range(tla.arch.block_idx(), total, tla.arch.block_num()):
        # Zig-zag complete waves, then visit high-causal-cost tiles first.
        wave = task // tla.arch.block_num()
        order = task
        if wave % 2 == 1:
            if (wave + 1) * tla.arch.block_num() <= total:
                order = (
                    wave * tla.arch.block_num()
                    + tla.arch.block_num()
                    - 1
                    - task % tla.arch.block_num()
                )
        order = total - 1 - order
        if packed:
            order = task
        b = order // (hq * blocks)
        qb = (order // hq) % blocks
        h = order % hq
        kh = h // (hq // hk)
        qlen = nq
        klen = nk
        qstart = tla.as_numeric(0)
        kstart = tla.as_numeric(0)
        qsoff = (b * hq + h) * ((nq + 127) // 128) + (qb * bm) // 128
        ksoff = (b * hk + kh) * ((nk + 63) // 64)
        ksstride = tla.as_numeric(1)
        if packed:
            qstart = cq[b].to(tla.Int32)
            kstart = ck[b].to(tla.Int32)
            qlen = cq[b + 1].to(tla.Int32) - qstart
            klen = ck[b + 1].to(tla.Int32) - kstart
            qsoff = (cqs[b].to(tla.Int32) + (qb * bm) // 128) * hq + h
            ksoff = cks[b].to(tla.Int32) * hk + kh
            ksstride = hk
        if qb * bm < qlen:
            qstride = tla.as_numeric(dim)
            kstride = tla.as_numeric(dim)
            qbase = (b * hq + h) * nq * dim
            kbase = (b * hk + kh) * nk * dim
            if packed:
                qstride = hq * dim
                kstride = hk * dim
                qbase = qstart * qstride + h * dim
                kbase = kstart * kstride + kh * dim
            elif nhd:
                qstride = hq * dim
                kstride = hk * dim
                qbase = b * nq * qstride + h * dim
                kbase = b * nk * kstride + kh * dim
            rows = bm if qlen - qb * bm >= bm else qlen - qb * bm
            kvstop = klen if klen < (qb + 1) * bm else (qb + 1) * bm
            steps = (kvstop + bn - 1) // bn
            qscale = qs[qsoff]
            sub = tla.arch.sub_block_idx()
            valid = rows - sub * half
            valid = half if valid > half else valid
            valid = 0 if valid < 0 else valid
            with tla.cube():
                qgm = tla.make_tensor(
                    q.ptr + qbase + qb * bm * qstride,
                    tla.make_layout(
                        tla.make_shape(dim, rows),
                        tla.make_stride(1, qstride),
                        layoutTag=tla.arch.ColumnMajor,
                    ),
                )
                qdst = tla.tile_view(
                    q1, tla.make_shape(dim, rows), tla.make_coord(0, 0)
                )
                m_q1.lock(pipe=tla.arch.MTE2)
                tla.copy(qdst, qgm)
                m_q1.unlock(pipe=tla.arch.MTE2)
                m_q1.lock(pipe=tla.arch.MTE1)
                m_q0.lock(pipe=tla.arch.MTE1)
                tla.copy(q0, q1)
                m_q0.unlock(pipe=tla.arch.MTE1)
                m_q1.unlock(pipe=tla.arch.MTE1)
            with tla.vector():
                vsgm = tla.make_tensor(
                    vs.ptr + (b * hk + kh) * dim,
                    tla.make_layout(tla.make_shape(1, dim), tla.make_stride(dim, 1)),
                )
                m_vscale.lock(pipe=tla.arch.MTE2)
                tla.copy(vscale, vsgm)
                m_vscale.unlock(pipe=tla.arch.MTE2)
                # The final epilogue reads the scale and the accumulator tile;
                # hold the output mutex until both stores have retired.
                m_out.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    init_full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    zero = tla.full(0.0, dtype=tla.Float32)
                    initial_max = tla.full(-3.0e38, dtype=tla.Float32)
                    maxima.store(
                        initial_max, mask=tla.update_mask(half, dtype=tla.Float32)[0]
                    )
                    sums.store(zero, mask=tla.update_mask(half, dtype=tla.Float32)[0])
                    for ir in tla.range(half):
                        for idim in tla.range_constexpr(dim // 64):
                            tla.tile_view(
                                acc, tla.make_shape(1, 64), tla.make_coord(ir, idim)
                            ).store(zero, mask=init_full)
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                m_out.unlock(pipe=tla.arch.VECTOR)
            # One-block lookahead: QK/softmax(t), then PV/rescale(t-1).
            # P and alpha retain their own generation until delayed consumption.
            for ki in tla.range(steps + 1):
                if ki < steps:
                    write_slot = ki % 2
                    cols = bn if klen - ki * bn >= bn else klen - ki * bn
                    kscale0 = ks[ksoff + ki * (bn // 64) * ksstride]
                    kscale1 = tla.as_numeric(0.0)
                    kscale2 = tla.as_numeric(0.0)
                    kscale3 = tla.as_numeric(0.0)
                    kscale4 = tla.as_numeric(0.0)
                    kscale5 = tla.as_numeric(0.0)
                    kscale6 = tla.as_numeric(0.0)
                    kscale7 = tla.as_numeric(0.0)
                    if cols > 64:
                        kscale1 = ks[ksoff + (ki * (bn // 64) + 1) * ksstride]
                    if cols > 128:
                        kscale2 = ks[ksoff + (ki * (bn // 64) + 2) * ksstride]
                    if cols > 192:
                        kscale3 = ks[ksoff + (ki * (bn // 64) + 3) * ksstride]
                    if cols > 256:
                        kscale4 = ks[ksoff + (ki * (bn // 64) + 4) * ksstride]
                    if cols > 320:
                        kscale5 = ks[ksoff + (ki * (bn // 64) + 5) * ksstride]
                    if cols > 384:
                        kscale6 = ks[ksoff + (ki * (bn // 64) + 6) * ksstride]
                    if cols > 448:
                        kscale7 = ks[ksoff + (ki * (bn // 64) + 7) * ksstride]
                    # Dequantization is block-invariant: fold q_scale and log2(e)
                    # into each per-64 key scale once per K step.
                    kss0 = (qscale * kscale0) * 0.6931471805599453
                    kss1 = (qscale * kscale1) * 0.6931471805599453
                    kss2 = (qscale * kscale2) * 0.6931471805599453
                    kss3 = (qscale * kscale3) * 0.6931471805599453
                    kss4 = (qscale * kscale4) * 0.6931471805599453
                    kss5 = (qscale * kscale5) * 0.6931471805599453
                    kss6 = (qscale * kscale6) * 0.6931471805599453
                    kss7 = (qscale * kscale7) * 0.6931471805599453
                    kss_list = [kss0, kss1, kss2, kss3, kss4, kss5, kss6, kss7]
                    with tla.cube():
                        kgm = tla.make_tensor(
                            k.ptr + kbase + ki * bn * kstride,
                            tla.make_layout(
                                tla.make_shape(cols, dim), tla.make_stride(kstride, 1)
                            ),
                        )
                        m_k1.lock(pipe=tla.arch.MTE2)
                        tla.copy(
                            tla.tile_view(
                                k1, tla.make_shape(cols, dim), tla.make_coord(0, 0)
                            ),
                            kgm,
                        )
                        m_k1.unlock(pipe=tla.arch.MTE2)
                        tla.cross_core_wait_flag(qready, tla.arch.FIX)
                        m_k1.lock(pipe=tla.arch.MTE1)
                        for qpart in tla.range_constexpr(bn // kc):
                            schunk = tla.make_tensor_like(
                                s0p + qpart * bm * kc, s0, tla.arch.L0Clayout
                            )
                            m_k0.lock(pipe=tla.arch.MTE1)
                            tla.copy(
                                k0,
                                tla.tile_view(
                                    k1,
                                    tla.make_shape(kc, dim),
                                    tla.make_coord(qpart, 0),
                                ),
                            )
                            m_k0.unlock(pipe=tla.arch.MTE1)
                            m_q0.lock(pipe=tla.arch.CUBE)
                            m_k0.lock(pipe=tla.arch.CUBE)
                            # Each L0C chunk owns its own mutex, so the other
                            # chunk's Fixpipe can drain concurrently.
                            schunk_lock = m_s0 if qpart == 0 else m_s1
                            schunk_lock.lock(pipe=tla.arch.CUBE)
                            tla.mmad(schunk, k0, q0, init_c=True)
                            schunk_lock.unlock(pipe=tla.arch.CUBE)
                            m_k0.unlock(pipe=tla.arch.CUBE)
                            m_q0.unlock(pipe=tla.arch.CUBE)
                            # SPLIT_N writes 64 query columns per AIV.
                            chunk_sfix = tla.make_tensor(
                                sp + qpart * kc * half,
                                tla.make_layout(
                                    tla.make_shape(kc, bm), tla.make_stride(half, 1)
                                ),
                            )
                            schunk_lock.lock(pipe=tla.arch.FIX)
                            tla.copy(chunk_sfix, schunk, split_s)
                            schunk_lock.unlock(pipe=tla.arch.FIX)
                        m_k1.unlock(pipe=tla.arch.MTE1)
                        tla.cross_core_set_flag(qready, tla.arch.FIX)
                    with tla.vector():
                        tla.cross_core_wait_flag(qready, tla.arch.VECTOR)
                        # Bounded-register native softmax + FP8 quantization.
                        m_pp.lock(pipe=tla.arch.VECTOR)
                        alu = tla.make_tensor(
                            rp + write_slot * half,
                            tla.make_layout(tla.make_shape(half), tla.make_stride(1)),
                        )
                        with tla.vec.func(mode="simd"):
                            mask32 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            mask256 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Int8
                            )
                            m32, _ = tla.update_mask(32, dtype=tla.Int8)
                            m64, _ = tla.update_mask(64, dtype=tla.Int8)
                            m_hi = tla.bitwise_and(
                                m64, tla.bitwise_not(m32, mask=mask256), mask=mask256
                            )
                            trait0 = tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.ZERO,
                                sat_mode=tla.params.SatMode.NOSAT,
                                round_mode=tla.params.RoundMode.CAST_ROUND,
                            )
                            trait1 = tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.ONE,
                                sat_mode=tla.params.SatMode.NOSAT,
                                round_mode=tla.params.RoundMode.CAST_ROUND,
                            )
                            trait2 = tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.TWO,
                                sat_mode=tla.params.SatMode.NOSAT,
                                round_mode=tla.params.RoundMode.CAST_ROUND,
                            )
                            trait3 = tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.THREE,
                                sat_mode=tla.params.SatMode.NOSAT,
                                round_mode=tla.params.RoundMode.CAST_ROUND,
                            )
                            traits = [trait0, trait1, trait2, trait3]
                            neg = tla.full(-3.0e38, dtype=tla.Float32)
                            qpos = tla.arange(qb * bm + sub * half, dtype=tla.Int32)
                            colsvec = tla.arange(cols, dtype=tla.Int32)
                            old = maxima.load()
                            maximum = old
                            # Pass 1: scale, causal/padding mask, running max.
                            for kg in tla.range_constexpr(bn // 64):
                                kss = kss_list[kg]
                                for r in tla.range(64):
                                    row = kg * 64 + r
                                    sv = _sage_score_row(
                                        s_src,
                                        s_scr,
                                        kss,
                                        qpos,
                                        colsvec,
                                        ki * bn,
                                        row,
                                        neg,
                                        mask32,
                                        cast,
                                    )
                                    maximum = tla.max(maximum, sv, mask=mask32)
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )
                            alpha = tla.exp(
                                tla.sub(old, maximum, mask=mask32), mask=mask32
                            )
                            denom = tla.mul(sums.load(), alpha, mask=mask32)
                            alu.store(alpha, mask=mask32)
                            maxima.store(maximum, mask=mask32)
                            sums.store(denom, mask=mask32)
                            # Pass 2: exp, denominator, FP8 quantization and pack.
                            for g in tla.range_constexpr(bn // 4):
                                denom = _sage_pack_prob_group(
                                    s_scr,
                                    pf0,
                                    pf1,
                                    g,
                                    maximum,
                                    denom,
                                    traits,
                                    mask32,
                                    mask256,
                                    m32,
                                    m_hi,
                                )
                            sums.store(denom, mask=mask32)
                        m_pp.unlock(pipe=tla.arch.VECTOR)
                        tla.cross_core_set_flag(qready, tla.arch.VECTOR)
                        if write_slot == 0:
                            tla.cross_core_wait_flag(pready, tla.arch.MTE3)
                        else:
                            tla.cross_core_wait_flag(pready_alt, tla.arch.MTE3)
                        pslot_storage = tla.make_tensor_like(
                            tla.recast_ptr(
                                p1p + write_slot * bm * bn, dtype=tla.Float16
                            ),
                            p1storage,
                            tla.arch.zN,
                        )
                        m_pp.lock(pipe=tla.arch.MTE3)
                        tla.copy(
                            tla.tile_view(
                                pslot_storage,
                                tla.make_shape(bn, half // 2),
                                tla.make_coord(0, sub),
                            ),
                            tla.tile_view(
                                pu16,
                                tla.make_shape(bn, half // 2),
                                tla.make_coord(0, 0),
                            ),
                        )
                        m_pp.unlock(pipe=tla.arch.MTE3)
                        if write_slot == 0:
                            tla.cross_core_set_flag(pready, tla.arch.MTE3)
                        else:
                            tla.cross_core_set_flag(pready_alt, tla.arch.MTE3)
                if ki > 0:
                    read_ki = ki - 1
                    read_slot = read_ki % 2
                    pv_cols = bn if klen - read_ki * bn >= bn else klen - read_ki * bn
                    # Zero the whole physical V tile on a partial K block. A
                    # masked zero P must never multiply uninitialized NaN padding.
                    if pv_cols < bn:
                        with tla.vector():
                            m_pp.lock(pipe=tla.arch.VECTOR)
                            with tla.vec.func(mode="simd"):
                                vzmask = tla.update_mask(dim // 2, dtype=tla.Float16)[0]
                                vzreg = tla.full(0.0, dtype=tla.Float16)
                                for vzrow in tla.range(bn // 2):
                                    tla.tile_view(
                                        vz16,
                                        tla.make_shape(1, dim // 2),
                                        tla.make_coord(vzrow, 0),
                                    ).store(
                                        vzreg,
                                        mask=vzmask,
                                        params=tla.params.BlockStoreParams(
                                            block_stride=bn // 2 + 1
                                        ),
                                    )
                            m_pp.unlock(pipe=tla.arch.VECTOR)
                            m_pp.lock(pipe=tla.arch.MTE3)
                            tla.copy(
                                tla.tile_view(
                                    v1storage,
                                    tla.make_shape(bn // 2, dim // 2),
                                    tla.make_coord(sub, 0),
                                ),
                                tla.tile_view(
                                    vz16,
                                    tla.make_shape(bn // 2, dim // 2),
                                    tla.make_coord(0, 0),
                                ),
                            )
                            m_pp.unlock(pipe=tla.arch.MTE3)
                            tla.cross_core_set_flag(vzero_ready, tla.arch.MTE3)
                    with tla.cube():
                        if pv_cols < bn:
                            tla.cross_core_wait_flag(vzero_ready, tla.arch.MTE2)
                        pv_vgm = tla.make_tensor(
                            v.ptr + kbase + read_ki * bn * kstride,
                            tla.make_layout(
                                tla.make_shape(pv_cols, dim),
                                tla.make_stride(kstride, 1),
                            ),
                        )
                        m_v1.lock(pipe=tla.arch.MTE2)
                        tla.copy(
                            tla.tile_view(
                                v1, tla.make_shape(pv_cols, dim), tla.make_coord(0, 0)
                            ),
                            pv_vgm,
                        )
                        m_v1.unlock(pipe=tla.arch.MTE2)
                        if read_slot == 0:
                            tla.cross_core_wait_flag(pready, tla.arch.MTE1)
                        else:
                            tla.cross_core_wait_flag(pready_alt, tla.arch.MTE1)
                        pslot_read = tla.make_tensor_like(
                            p1p + read_slot * bm * bn, p1t, tla.arch.nZ
                        )
                        m_v1.lock(pipe=tla.arch.MTE1)
                        for ppart in tla.range_constexpr(bn // kc):
                            m_v0.lock(pipe=tla.arch.MTE1)
                            m_p0.lock(pipe=tla.arch.MTE1)
                            tla.copy(
                                v0,
                                tla.tile_view(
                                    v1,
                                    tla.make_shape(kc, dim),
                                    tla.make_coord(ppart, 0),
                                ),
                            )
                            tla.copy(
                                p0,
                                tla.tile_view(
                                    pslot_read,
                                    tla.make_shape(bm, kc),
                                    tla.make_coord(0, ppart),
                                ),
                            )
                            m_p0.unlock(pipe=tla.arch.MTE1)
                            m_v0.unlock(pipe=tla.arch.MTE1)
                            m_v0.lock(pipe=tla.arch.CUBE)
                            m_p0.lock(pipe=tla.arch.CUBE)
                            m_s0.lock(pipe=tla.arch.CUBE)
                            tla.mmad(o0, p0, v0, init_c=(ppart == 0))
                            m_s0.unlock(pipe=tla.arch.CUBE)
                            m_p0.unlock(pipe=tla.arch.CUBE)
                            m_v0.unlock(pipe=tla.arch.CUBE)
                        m_v1.unlock(pipe=tla.arch.MTE1)
                        if read_slot == 0:
                            tla.cross_core_set_flag(pready, tla.arch.MTE1)
                        else:
                            tla.cross_core_set_flag(pready_alt, tla.arch.MTE1)
                        tla.cross_core_wait_flag(oready, tla.arch.FIX)
                        m_s0.lock(pipe=tla.arch.FIX)
                        tla.copy(ofix, o0, split)
                        m_s0.unlock(pipe=tla.arch.FIX)
                        tla.cross_core_set_flag(oready, tla.arch.FIX)
                    with tla.vector():
                        tla.cross_core_wait_flag(oready, tla.arch.VECTOR)
                        rread = tla.make_tensor(
                            rp + read_slot * half,
                            tla.make_layout(tla.make_shape(half), tla.make_stride(1)),
                        )
                        with tla.vec.func(mode="simd"):
                            acc_full = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            for ar in tla.range(valid):
                                prev_alpha = tla.tile_view(
                                    rread, tla.make_shape(1), tla.make_coord(ar)
                                ).load(bcast)
                                for d in tla.range_constexpr(dim // 64):
                                    cur = tla.tile_view(
                                        ou, tla.make_shape(1, 64), tla.make_coord(ar, d)
                                    ).load()
                                    at = tla.tile_view(
                                        acc,
                                        tla.make_shape(1, 64),
                                        tla.make_coord(ar, d),
                                    )
                                    cur = at.load() * prev_alpha + cur
                                    at.store(cur, mask=acc_full)
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )
                        tla.cross_core_set_flag(oready, tla.arch.VECTOR)
            with tla.vector():
                # acc aliases yu, and the scale is read one last time here; both
                # must wait for the previous task's MTE3 stores to retire.
                m_out.lock(pipe=tla.arch.VECTOR)
                m_vscale.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    output_mask = tla.update_mask(64, dtype=out_dtype)[0]
                    for yr in tla.range(valid):
                        final_den = tla.tile_view(
                            sums, tla.make_shape(1), tla.make_coord(yr)
                        ).load(bcast)
                        for yd in tla.range_constexpr(dim // 64):
                            y = (
                                tla.tile_view(
                                    acc, tla.make_shape(1, 64), tla.make_coord(yr, yd)
                                ).load()
                                / final_den
                            )
                            channel_scale = tla.tile_view(
                                vscale, tla.make_shape(1, 64), tla.make_coord(0, yd)
                            ).load()
                            y = y * (channel_scale * 0.002232142857142857)
                            if out_dtype == tla.Float32:
                                tla.tile_view(
                                    yu, tla.make_shape(1, 64), tla.make_coord(yr, yd)
                                ).store(y)
                            else:
                                yh = y.to(out_dtype, cast)
                                ypck, unused_y = tla.deinterleave(yh, yh)
                                tla.tile_view(
                                    yu, tla.make_shape(1, 64), tla.make_coord(yr, yd)
                                ).store(ypck, mask=output_mask)
                m_vscale.unlock(pipe=tla.arch.VECTOR)
                m_out.unlock(pipe=tla.arch.VECTOR)
                if valid > 0:
                    ogm = tla.make_tensor(
                        out.ptr + qbase + (qb * bm + sub * half) * qstride,
                        tla.make_layout(
                            tla.make_shape(valid, dim), tla.make_stride(qstride, 1)
                        ),
                    )
                    m_out.lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        ogm,
                        tla.tile_view(
                            yu, tla.make_shape(valid, dim), tla.make_coord(0, 0)
                        ),
                    )
                    m_out.unlock(pipe=tla.arch.MTE3)
                    if want_lse:
                        m_out.lock(pipe=tla.arch.VECTOR)
                        with tla.vec.func(mode="simd"):
                            for lr in tla.range(valid):
                                m = tla.tile_view(
                                    maxima, tla.make_shape(1), tla.make_coord(lr)
                                ).load(bcast)
                                psum = tla.tile_view(
                                    sums, tla.make_shape(1), tla.make_coord(lr)
                                ).load(bcast)
                                e = (m + tla.log(psum)) * 1.4426950408889634
                                tla.tile_view(
                                    elu, tla.make_shape(1), tla.make_coord(lr)
                                ).store(e, params=us)
                        m_out.unlock(pipe=tla.arch.VECTOR)
                        egm = tla.make_tensor(
                            lse.ptr + (b * hq + h) * nq + qb * bm + sub * half,
                            tla.make_layout(tla.make_shape(valid), tla.make_stride(1)),
                        )
                        m_out.lock(pipe=tla.arch.MTE3)
                        tla.copy(
                            egm,
                            tla.tile_view(
                                elu, tla.make_shape(valid), tla.make_coord(0)
                            ),
                        )
                        m_out.unlock(pipe=tla.arch.MTE3)
    with tla.cube():
        tla.cross_core_wait_flag(qready, tla.arch.FIX)
        tla.cross_core_wait_flag(oready, tla.arch.FIX)
    with tla.vector():
        tla.cross_core_wait_flag(pready, tla.arch.MTE3)
        tla.cross_core_wait_flag(pready_alt, tla.arch.MTE3)


def run(
    q,
    k,
    v,
    q_scale,
    k_scale,
    v_scale,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    cu_seqlens_q_scale=None,
    cu_seqlens_k_scale=None,
    tensor_layout="HND",
    output_dtype="float16",
    return_lse=False,
):
    """Launch attention only; Q/K/V are already quantized.

    Device-data preconditions (not scanned or synchronized on the host):
    prefixes start at zero, are monotone, end at their tensor lengths, and
    match per128/per64 scale blocks; max_seqlen_q bounds each query length.
    v_scale must be finite and nonnegative. D=64/128 and signed-Int32
    addressable sizes are supported. Empty KV produces the oracle's NaNs.
    """
    import torch
    import torch_npu

    if isinstance(output_dtype, str):
        output_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[output_dtype]
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("unsupported output dtype")
    meta = (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        cu_seqlens_q_scale,
        cu_seqlens_k_scale,
    )
    packed = all(x is not None for x in meta)
    if any(x is not None for x in meta) != packed:
        raise ValueError("all five varlen arguments must be supplied together")
    if tensor_layout not in ("HND", "NHD") or (
        packed and (tensor_layout != "HND" or return_lse)
    ):
        raise ValueError("invalid layout/LSE configuration")
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.float8_e4m3fn:
        raise ValueError("this kernel supports INT8 Q/K and E4M3FN V")
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("incompatible Q/K/V shapes")
    if q.device.type != "npu":
        raise ValueError("NPU inputs required")
    for x in (q, k, v, q_scale, k_scale, v_scale):
        if x.device != q.device or not x.is_contiguous():
            raise ValueError("inputs must be contiguous on one NPU")
    if any(x.dtype != torch.float32 for x in (q_scale, k_scale, v_scale)):
        raise ValueError("scales must be float32")
    if packed:
        if q.dim() != 3 or k.dim() != 3:
            raise ValueError("packed input must be TND")
        batch, hq, hk = cu_seqlens_q.numel() - 1, q.size(1), k.size(1)
        nq, nk = int(max_seqlen_q), k.size(0)
        for x in (cu_seqlens_q, cu_seqlens_k, cu_seqlens_q_scale, cu_seqlens_k_scale):
            if (
                x.device != q.device
                or x.dtype not in (torch.int32, torch.int64)
                or x.dim() != 1
                or x.numel() != batch + 1
                or not x.is_contiguous()
            ):
                raise ValueError("invalid prefix-sum metadata")
        if (
            q_scale.dim() != 2
            or k_scale.dim() != 2
            or q_scale.size(1) != hq
            or k_scale.size(1) != hk
        ):
            raise ValueError("invalid packed scale shape")
    else:
        if q.dim() != 4 or k.dim() != 4 or q.size(0) != k.size(0):
            raise ValueError("fixed input must have matching batches and rank four")
        batch = q.size(0)
        haxis, naxis = (1, 2) if tensor_layout == "HND" else (2, 1)
        hq, hk, nq, nk = q.size(haxis), k.size(haxis), q.size(naxis), k.size(naxis)
        if nq != nk:
            raise ValueError("fixed causal input lengths must match")
        if q_scale.shape != (batch, hq, (nq + 127) // 128) or k_scale.shape != (
            batch,
            hk,
            (nk + 63) // 64,
        ):
            raise ValueError("invalid fixed scale shape")
        cu_seqlens_q = cu_seqlens_k = cu_seqlens_q_scale = cu_seqlens_k_scale = q
    dim = q.size(-1)
    if dim not in (64, 128) or batch < 1 or hq < 1 or hk < 1 or hq % hk or nq < 1:
        raise ValueError(
            "requires D=64/128, positive batch/heads/max query length and Hq divisible by Hkv"
        )
    if v_scale.shape != (batch, hk, dim):
        raise ValueError("v_scale must have shape [B, Hkv, D]")
    # Prefix values and derived offsets are bounded by these storage sizes.
    task_count = batch * hq * ((nq + 127) // 128)
    if (
        any(x.numel() >= 2**31 for x in (q, k, v, q_scale, k_scale, v_scale))
        or nq + 127 >= 2**31
        or task_count >= 2**31
    ):
        raise ValueError("kernel requires signed-int32 addressable input sizes")
    out = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    lse = torch.empty(
        (batch, hq, nq) if return_lse else (0,),
        dtype=torch.float32,
        device=q.device if return_lse else "cpu",
    )
    tensors = (
        q,
        k,
        v,
        out,
        q_scale,
        k_scale,
        v_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_q_scale,
        cu_seqlens_k_scale,
        lse if return_lse else out,
    )
    # CATLASS rejects null DLPack pointers. Empty KV has steps=0, so
    # these allocated bindings are never read; no fill or extra kernel.
    tensors = tuple(
        x
        if x.numel()
        else torch.empty(
            tuple(max(1, n) for n in x.shape), dtype=x.dtype, device=x.device
        )
        for x in tensors
    )
    args = tuple(
        tla.from_dlpack(
            x.view(torch.int8) if x.dtype == torch.float8_e4m3fn else x,
            element_type=tla.Float8E4M3FN if x.dtype == torch.float8_e4m3fn else None,
            origin_shape=(x.numel(),),
            layout_tag=tla.arch.RowMajor,
        ).mark_layout_dynamic()
        for x in tensors
    )
    scalars = tuple(tla.Int32(x) for x in (batch, hq, hk, nq, nk))
    key = (
        str(q.device),
        str(v.dtype),
        str(output_dtype),
        dim,
        packed,
        tensor_layout,
        return_lse,
        tuple(str(x.dtype) for x in tensors[7:11]),
    )
    executor = _EXECUTORS.get(key)
    if executor is None:
        executor = tla.compile(
            sage_attention_kernel,
            *args,
            *scalars,
            dim,
            packed,
            tensor_layout == "NHD",
            return_lse,
            options="--npu-arch 3510",
        )
        _EXECUTORS[key] = executor
    cores = int(torch_npu.npu.get_device_properties(q.device).cube_core_num)
    executor(*args, *scalars, block_num=min(cores, task_count))
    return out if packed else (out, lse)
