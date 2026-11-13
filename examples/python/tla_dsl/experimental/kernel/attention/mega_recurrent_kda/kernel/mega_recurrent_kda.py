# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Mega Recurrent KDA stages and one fused C1V2 entry."""

import catlass.tla as tla


# AIV helpers, Conv and recurrent stages


@tla.jit
def flat(ptr, count):
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(count), tla.make_stride(1))
    )


@tla.jit
def load_bf64(tile):
    value = tile.load(
        tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_UNPACK_B16)
    )
    return value.to(
        tla.Float32,
        tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO,
            sat_mode=tla.params.SatMode.NOSAT,
            round_mode=tla.params.RoundMode.CAST_ROUND,
        ),
    )


@tla.jit
def store_bf64(tile, value):
    mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
    bits = value.bitcast(tla.Int32)
    parity = tla.bitwise_and(tla.shift_right(bits, 16), tla.full(1, tla.Int32))
    rounded = tla.bitwise_and(
        bits + tla.full(32767, tla.Int32) + parity, tla.full(-65536, tla.Int32)
    )
    special = tla.cmp(
        tla.bitwise_and(bits, tla.full(2147483647, tla.Int32)), 2139095040, mode="ge"
    )
    rounded = tla.where(special, bits, rounded)
    converted = rounded.bitcast(tla.Float32).to(
        tla.BFloat16,
        tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO,
            sat_mode=tla.params.SatMode.NOSAT,
            round_mode=tla.params.RoundMode.CAST_TRUNC,
        ),
        mask,
    )
    tile.store(
        converted,
        tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK_B32),
    )


@tla.jit
def segment(base, start, count):
    # All segment starts are multiples of count (64 SIMD lanes, or 1 scalar).
    return tla.tile_view(base, tla.make_shape(count), tla.make_coord(start // count))


@tla.jit
def conv_consume(
    rawp,
    arena,
    dst,
    cw,
    cs,
    ci,
    ca,
    first_row,
    half,
    channel,
    slot,
    L,
    loaded,
    scalar_ready,
    done,
    available,
    RAW_PITCH: tla.Constexpr[int] = 128,
    COMPACT_AUX: tla.Constexpr[bool] = False,
):
    channels = cs.origin_shape[1]
    batches = half // 8
    bf = tla.recast_ptr(arena, dtype=tla.BFloat16)
    histp = bf + 32768  # UB byte 65536, up to 6144B.
    weightp = bf + 35840  # UB byte 71680, 1024B.
    metap = tla.recast_ptr(arena + 18432, dtype=tla.Int32)  # byte73728
    outputp = bf + 40960 if COMPACT_AUX else bf + 40960 + slot * 8192
    castp = bf + 49152 if COMPACT_AUX else bf + 57344
    metadata = flat(metap, 8)
    raw_all = flat(rawp, 63 * RAW_PITCH + 128)
    hist_all = flat(histp, 3072)
    weight_all = flat(weightp, 512)
    out_all = flat(outputp, 8192)
    cast_all = flat(castp, 8192)
    tla.wait_flag(available)
    meta_src = flat(ci.ptr + first_row // 8, batches)
    meta_dst = tla.make_tensor(
        metap,
        tla.make_layout(
            tla.make_shape(8), tla.make_stride(1), origin_shape=tla.make_shape(batches)
        ),
    )
    tla.copy(meta_dst, meta_src)
    for load_batch in tla.range(batches):
        batch = first_row // 8 + load_batch
        load_idx = tla.Int64(ci[batch])
        accepted = ca[batch]
        off = accepted - 1 if accepted > 0 else 0
        history = tla.make_tensor(
            cs.ptr
            + (load_idx * tla.Int64(L) + tla.Int64(off)) * tla.Int64(channels)
            + tla.Int64(channel),
            tla.make_layout(tla.make_shape(3, 128), tla.make_stride(channels, 1)),
        )
        hist = tla.make_tensor(
            histp + load_batch * 384,
            tla.make_layout(tla.make_shape(3, 128), tla.make_stride(128, 1)),
        )
        tla.copy(hist, history)
    weights = tla.make_tensor(
        cw.ptr + (channel // 128) * 512,
        tla.make_layout(tla.make_shape(4, 128), tla.make_stride(128, 1)),
    )
    wu = tla.make_tensor(
        weightp, tla.make_layout(tla.make_shape(4, 128), tla.make_stride(128, 1))
    )
    tla.copy(wu, weights)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    with tla.vec.func(mode="simd"):
        for lane in tla.range_constexpr(2):
            col = lane * 64
            weight0 = load_bf64(segment(weight_all, col, 64))
            weight1 = load_bf64(segment(weight_all, 128 + col, 64))
            weight2 = load_bf64(segment(weight_all, 256 + col, 64))
            weight3 = load_bf64(segment(weight_all, 384 + col, 64))
            for compute_batch in tla.range(batches):
                compute_idx = segment(metadata, compute_batch, 1).load(
                    tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                null_slot = tla.cmp(compute_idx, 0, mode="eq")
                history0 = load_bf64(segment(hist_all, compute_batch * 384 + col, 64))
                history1 = load_bf64(
                    segment(hist_all, compute_batch * 384 + 128 + col, 64)
                )
                history2 = load_bf64(
                    segment(hist_all, compute_batch * 384 + 256 + col, 64)
                )
                for t in tla.range(8):
                    current_value = segment(
                        raw_all, (compute_batch * 8 + t) * RAW_PITCH + col, 64
                    ).load()
                    total = tla.full(0.0, tla.Float32) + history0 * weight0
                    total = total + history1 * weight1
                    total = total + history2 * weight2
                    total = total + current_value * weight3
                    result = total / (
                        tla.full(1.0, tla.Float32) + tla.exp(tla.neg(total))
                    )
                    result = tla.where(null_slot, tla.full(0.0, tla.Float32), result)
                    store_bf64(
                        segment(out_all, (compute_batch * 8 + t) * 128 + col, 64),
                        result,
                    )
                    store_bf64(
                        segment(cast_all, (compute_batch * 8 + t) * 128 + col, 64),
                        current_value,
                    )
                    # Keep the new history in FP32; only the saved pool is BF16.
                    history0 = history1
                    history1 = history2
                    history2 = current_value
    tla.set_flag(done)
    tla.wait_flag(done)
    output = tla.make_tensor(
        outputp, tla.make_layout(tla.make_shape(half, 128), tla.make_stride(128, 1))
    )
    tla.copy(dst, output)
    for store_batch in tla.range(batches):
        store_idx = ci[first_row // 8 + store_batch]
        if store_idx != 0:
            previous = tla.make_tensor(
                histp + store_batch * 384 + 128,
                tla.make_layout(tla.make_shape(2, 128), tla.make_stride(128, 1)),
            )
            hist_dst = tla.make_tensor(
                cs.ptr
                + tla.Int64(store_idx) * tla.Int64(L) * tla.Int64(channels)
                + tla.Int64(channel),
                tla.make_layout(tla.make_shape(2, 128), tla.make_stride(channels, 1)),
            )
            tla.copy(hist_dst, previous)
            current = tla.make_tensor(
                castp + store_batch * 1024,
                tla.make_layout(tla.make_shape(8, 128), tla.make_stride(128, 1)),
            )
            raw_dst = tla.make_tensor(
                cs.ptr
                + (tla.Int64(store_idx) * tla.Int64(L) + tla.Int64(2))
                * tla.Int64(channels)
                + tla.Int64(channel),
                tla.make_layout(tla.make_shape(8, 128), tla.make_stride(channels, 1)),
            )
            tla.copy(raw_dst, current)
    tla.set_flag(available)


@tla.jit
def matrix(ptr, rows, cols, pitch):
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(rows, cols), tla.make_stride(pitch, 1))
    )


@tla.jit
def broadcast_f32(base, index):
    return segment(base, index, 1).load(
        tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
    )


@tla.jit
def broadcast_bf16(base, index):
    v = segment(base, index, 1).load(
        tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B16)
    )
    return v.to(
        tla.Float32,
        tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO,
            sat_mode=tla.params.SatMode.NOSAT,
            round_mode=tla.params.RoundMode.CAST_ROUND,
        ),
    )


@tla.jit
def sum128(x0, x1, mask):
    return tla.full((x0 + x1).reduce(tla.ReductionOp.ADD, mask=mask), tla.Float32)


def allocate_recurrent_resources(arena):
    loaded = tla.flag("rec_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("rec_done", tla.arch.VECTOR, tla.arch.MTE3)
    sf0 = tla.flag("snapshot_free0", tla.arch.MTE3, tla.arch.VECTOR)
    sf1 = tla.flag("snapshot_free1", tla.arch.MTE3, tla.arch.VECTOR)
    finished = tla.cross_flag("rec_finished", mode=2)
    chunk_ready1 = tla.cross_flag("rec_chunk_ready1", mode=2)
    chunk_ack0 = tla.cross_flag("rec_chunk_ack0", mode=2)
    chunk_ack1 = tla.cross_flag("rec_chunk_ack1", mode=2)
    return (
        arena,
        loaded,
        done,
        sf0,
        sf1,
        finished,
        chunk_ready1,
        chunk_ack0,
        chunk_ack1,
    )


@tla.jit
def recurrent_stage(
    qkv: tla.Tensor,
    g: tla.Tensor,
    beta: tla.Tensor,
    z: tla.Tensor,
    state: tla.Tensor,
    indices: tla.Tensor,
    accepted: tla.Tensor,
    alog: tla.Tensor,
    bias: tla.Tensor,
    gamma: tla.Tensor,
    u: tla.Tensor,
    resources,
    N: tla.Constexpr[int],
    SCALE: tla.Constexpr[float],
    LOWER: tla.Constexpr[float],
    EPS: tla.Constexpr[float],
    PIPELINED: tla.Constexpr[bool] = False,
):
    (arena, loaded, done, sf0, sf1, finished, chunk_ready1, chunk_ack0, chunk_ack1) = (
        resources
    )
    bf = tla.recast_ptr(arena, dtype=tla.BFloat16)
    B = qkv.origin_shape[0] // 8
    P = N * 128
    NB = beta.origin_shape[1]
    with tla.cube():
        if tla.const_expr(PIPELINED):
            tla.cross_core_set_flag(chunk_ack0, tla.arch.MTE2)
            tla.cross_core_set_flag(chunk_ack1, tla.arch.MTE2)
        else:
            tla.cross_core_wait_flag(finished, tla.arch.SCALAR)
            tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        for bh in tla.range(
            2 * tla.arch.block_idx() + tla.arch.sub_block_idx(),
            B * N,
            2 * tla.arch.block_num(),
        ):
            b = bh // N
            h = bh % N
            work_index = bh // (2 * tla.arch.block_num())
            active_state = arena if work_index % 2 == 0 else arena + 40960
            prefetch_state = arena + 40960 if work_index % 2 == 0 else arena
            qptr = bf + 65536
            kptr = qptr + 1024
            vptr = qptr + 2048
            gptr = qptr + 3072
            zptr = qptr + 4096
            betap = qptr + 5120
            dtp = arena + 35584
            gp = arena + 35712
            alp = arena + 35840
            op = arena + 35904
            up = bf + 72192
            qa = flat(qptr, 1024)
            ka = flat(kptr, 1024)
            va = flat(vptr, 1024)
            ga = flat(gptr, 1024)
            za = flat(zptr, 1024)
            ba = flat(betap, 8 * NB)
            dta = flat(dtp, 128)
            gma = flat(gp, 128)
            ala = flat(alp, 8)
            oa = flat(op, 128)
            ua = flat(up, 1024)
            delta_rows = flat(arena + 36864, 128)
            tla.copy(
                matrix(qptr, 8, 128, 128),
                matrix(qkv.ptr + b * 8 * 3 * P + h * 128, 8, 128, 3 * P),
            )
            tla.copy(
                matrix(kptr, 8, 128, 128),
                matrix(qkv.ptr + b * 8 * 3 * P + P + h * 128, 8, 128, 3 * P),
            )
            tla.copy(
                matrix(vptr, 8, 128, 128),
                matrix(qkv.ptr + b * 8 * 3 * P + 2 * P + h * 128, 8, 128, 3 * P),
            )
            tla.copy(
                matrix(gptr, 8, 128, 128),
                matrix(g.ptr + b * 8 * P + h * 128, 8, 128, P),
            )
            tla.copy(
                matrix(zptr, 8, 128, 128),
                matrix(z.ptr + b * 8 * P + h * 128, 8, 128, P),
            )
            tla.copy(matrix(betap, 8, NB, NB), matrix(beta.ptr + b * 8 * NB, 8, NB, NB))
            tla.copy(matrix(dtp, 1, 128, 128), matrix(bias.ptr + h * 128, 1, 128, 128))
            tla.copy(matrix(gp, 1, 128, 128), matrix(gamma.ptr, 1, 128, 128))
            # A physical eight-element UB buffer with logical one element.
            al_dst = tla.make_tensor(
                alp,
                tla.make_layout(
                    tla.make_shape(1, 8),
                    tla.make_stride(8, 1),
                    origin_shape=tla.make_shape(1, 1),
                ),
            )
            tla.copy(al_dst, matrix(alog.ptr + h, 1, 1, 1))
            if work_index == 0:
                naccept = accepted[b]
                initial_t = naccept - 1 if naccept > 0 else 0
                initial_slot = tla.Int64(indices[b * 8 + initial_t])
                tla.copy(
                    matrix(active_state, 128, 128, 128),
                    matrix(
                        state.ptr
                        + (initial_slot * tla.Int64(N) + tla.Int64(h))
                        * tla.Int64(16384),
                        128,
                        128,
                        128,
                    ),
                )
            tla.set_flag(loaded)
            # loaded precedes this DMA, so current VECTOR work can overlap it.
            next_bh = bh + 2 * tla.arch.block_num()
            if next_bh < B * N:
                next_batch = next_bh // N
                next_head = next_bh % N
                next_accept = accepted[next_batch]
                next_t = next_accept - 1 if next_accept > 0 else 0
                next_slot = tla.Int64(indices[next_batch * 8 + next_t])
                tla.copy(
                    matrix(prefetch_state, 128, 128, 128),
                    matrix(
                        state.ptr
                        + (next_slot * tla.Int64(N) + tla.Int64(next_head))
                        * tla.Int64(16384),
                        128,
                        128,
                        128,
                    ),
                )
            tla.wait_flag(loaded)
            tla.set_flag(sf0)
            tla.set_flag(sf1)
            for t in tla.range(8):
                parity = t % 2
                srcp = active_state if parity == 0 else arena + 16384
                dstp = arena + 16384 if parity == 0 else active_state
                old = flat(srcp, 16384)
                new = flat(dstp, 16384)
                if parity == 0:
                    tla.wait_flag(sf1)
                else:
                    tla.wait_flag(sf0)
                with tla.vec.func(mode="simd"):
                    mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    q0 = load_bf64(segment(qa, t * 128, 64))
                    q1 = load_bf64(segment(qa, t * 128 + 64, 64))
                    k0 = load_bf64(segment(ka, t * 128, 64))
                    k1 = load_bf64(segment(ka, t * 128 + 64, 64))
                    qnorm = tla.sqrt(
                        sum128(q0 * q0, q1 * q1, mask) + tla.full(1.0e-6, tla.Float32)
                    )
                    knorm = tla.sqrt(
                        sum128(k0 * k0, k1 * k1, mask) + tla.full(1.0e-6, tla.Float32)
                    )
                    q0 = q0 / qnorm * tla.full(SCALE, tla.Float32)
                    q1 = q1 / qnorm * tla.full(SCALE, tla.Float32)
                    k0 = k0 / knorm
                    k1 = k1 / knorm
                    one = tla.full(1.0, tla.Float32)
                    ea = tla.exp(broadcast_f32(ala, 0))
                    gate0 = (
                        load_bf64(segment(ga, t * 128, 64)) + segment(dta, 0, 64).load()
                    )
                    gate1 = (
                        load_bf64(segment(ga, t * 128 + 64, 64))
                        + segment(dta, 64, 64).load()
                    )
                    d0 = tla.exp(
                        tla.full(LOWER, tla.Float32)
                        / (one + tla.exp(tla.neg(ea * gate0)))
                    )
                    d1 = tla.exp(
                        tla.full(LOWER, tla.Float32)
                        / (one + tla.exp(tla.neg(ea * gate1)))
                    )
                    be = one / (one + tla.exp(tla.neg(broadcast_bf16(ba, t * NB + h))))
                    decay_key0 = d0 * k0
                    decay_key1 = d1 * k1
                    # Reduce all predictions before the vector-wide delta step.
                    for row_group in tla.range(32):
                        for row_lane in tla.range_constexpr(4):
                            predict_row = row_group * 4 + row_lane
                            weighted0 = (
                                segment(old, predict_row * 128, 64).load() * decay_key0
                            )
                            weighted1 = (
                                segment(old, predict_row * 128 + 64, 64).load()
                                * decay_key1
                            )
                            pred = (weighted0 + weighted1).reduce(
                                tla.ReductionOp.ADD, mask=mask
                            )
                            segment(delta_rows, predict_row, 1).store(
                                pred,
                                tla.params.NormalStoreParams(
                                    store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                                ),
                            )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                    for delta_lane in tla.range_constexpr(2):
                        values = load_bf64(segment(va, t * 128 + delta_lane * 64, 64))
                        predictions = segment(delta_rows, delta_lane * 64, 64).load()
                        segment(delta_rows, delta_lane * 64, 64).store(
                            be * (values - predictions)
                        )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                    for update_group in tla.range(32):
                        for update_lane in tla.range_constexpr(4):
                            r = update_group * 4 + update_lane
                            a0 = segment(old, r * 128, 64).load() * d0
                            a1 = segment(old, r * 128 + 64, 64).load() * d1
                            delta = broadcast_f32(delta_rows, r)
                            n0 = a0 + delta * k0
                            n1 = a1 + delta * k1
                            segment(new, r * 128, 64).store(n0)
                            segment(new, r * 128 + 64, 64).store(n1)
                            oval = (n0 * q0 + n1 * q1).reduce(
                                tla.ReductionOp.ADD, mask=mask
                            )
                            segment(oa, r, 1).store(
                                oval,
                                tla.params.NormalStoreParams(
                                    store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                                ),
                            )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                    o0 = segment(oa, 0, 64).load()
                    o1 = segment(oa, 64, 64).load()
                    denom = tla.sqrt(
                        sum128(o0 * o0, o1 * o1, mask)
                        * tla.full(1.0 / 128, tla.Float32)
                        + tla.full(EPS, tla.Float32)
                    )
                    z0 = load_bf64(segment(za, t * 128, 64))
                    z1 = load_bf64(segment(za, t * 128 + 64, 64))
                    u0 = (
                        o0
                        / denom
                        * segment(gma, 0, 64).load()
                        / (one + tla.exp(tla.neg(z0)))
                    )
                    u1 = (
                        o1
                        / denom
                        * segment(gma, 64, 64).load()
                        / (one + tla.exp(tla.neg(z1)))
                    )
                    store_bf64(segment(ua, t * 128, 64), u0)
                    store_bf64(segment(ua, t * 128 + 64, 64), u1)
                tla.set_flag(done)
                tla.wait_flag(done)
                target_slot = tla.Int64(indices[b * 8 + t])
                tla.copy(
                    matrix(
                        state.ptr
                        + (target_slot * tla.Int64(N) + tla.Int64(h))
                        * tla.Int64(16384),
                        128,
                        128,
                        128,
                    ),
                    matrix(dstp, 128, 128, 128),
                )
                if parity == 0:
                    tla.set_flag(sf1)
                else:
                    tla.set_flag(sf0)
            tla.copy(
                matrix(u.ptr + b * 8 * P + h * 128, 8, 128, P), matrix(up, 8, 128, 128)
            )
            tla.wait_flag(sf0)
            tla.wait_flag(sf1)
            tla.pipe_barrier(tla.pipes.ALL)
            if tla.const_expr(PIPELINED):
                if next_bh >= B * N or next_bh // (16 * N) != bh // (16 * N):
                    publish_slot = (bh // (16 * N)) % 2
                    if publish_slot == 0:
                        tla.cross_core_wait_flag(chunk_ack0, tla.arch.MTE3)
                        tla.cross_core_set_flag(finished, tla.arch.MTE3)
                    else:
                        tla.cross_core_wait_flag(chunk_ack1, tla.arch.MTE3)
                        tla.cross_core_set_flag(chunk_ready1, tla.arch.MTE3)
        if tla.const_expr(PIPELINED):
            tla.cross_core_wait_flag(chunk_ack0, tla.arch.MTE3)
            tla.cross_core_wait_flag(chunk_ack1, tla.arch.MTE3)
        else:
            tla.cross_core_set_flag(finished, tla.arch.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def recurrent_kernel(
    qkv: tla.Tensor,
    g: tla.Tensor,
    beta: tla.Tensor,
    z: tla.Tensor,
    state: tla.Tensor,
    indices: tla.Tensor,
    accepted: tla.Tensor,
    alog: tla.Tensor,
    bias: tla.Tensor,
    gamma: tla.Tensor,
    u: tla.Tensor,
    N: tla.Constexpr[int],
    SCALE: tla.Constexpr[float],
    LOWER: tla.Constexpr[float],
    EPS: tla.Constexpr[float],
):
    arena = tla.allocate(57344, tla.Float32, tla.AddressSpace.ub, 256)
    resources = allocate_recurrent_resources(arena)
    recurrent_stage(
        qkv,
        g,
        beta,
        z,
        state,
        indices,
        accepted,
        alog,
        bias,
        gamma,
        u,
        resources,
        N,
        SCALE,
        LOWER,
        EPS,
    )


# AIC resources, projection tiles and stage scheduling


@tla.jit
def resident_projection_tile(
    x,
    w_raw,
    out,
    resources,
    mi,
    macro,
    work,
    TM: tla.Constexpr[int],
    CV: tla.Constexpr[bool],
):
    (
        l1p,
        l0ap,
        l0bp,
        l0cp,
        ld0,
        ld1,
        lf0,
        lf1,
        af0,
        af1,
        ad,
        cf,
        ready0,
        ready1,
        free0,
        free1,
        rawp,
        conv_loaded,
        conv_scalar,
        conv_done,
        conv_free,
        bf0,
        bf1,
        panel_ready0,
        panel_ready1,
        panel_ready2,
    ) = resources
    K = x.origin_shape[1]
    N = out.origin_shape[1]
    cvslot = (work // tla.arch.block_num()) % 2
    tla.wait_flag(cf)
    for kb in tla.range((K + 255) // 256):
        slot = kb % 2
        a_src = tla.tile_view(x, tla.make_shape(TM, 256), tla.make_coord(mi, kb))
        w_src = tla.make_tensor(
            w_raw.ptr + (kb * 16 * N + macro * 384) * 16,
            tla.make_layout(
                tla.make_shape((384, 1), (16, a_src.origin_shape[1] // 16)),
                tla.make_stride((16, N * 16), (1, N * 16)),
                origin_shape=tla.make_shape(384, a_src.origin_shape[1]),
                layoutTag=tla.arch.zNUnAlign,
            ),
        )
        a_l1 = tla.make_tensor_like(l1p + slot * 32768, a_src, tla.arch.zN)
        wp = l1p + 65536 + slot * 98304
        w_l1 = tla.make_tensor(
            wp,
            tla.make_layout_with_tag(
                (w_src.origin_shape[0], w_src.origin_shape[1]),
                tla.BFloat16,
                tla.arch.zNUnAlign,
            ),
        )
        wt_l1 = tla.make_tensor(
            wp,
            tla.make_layout_with_tag(
                (w_src.origin_shape[1], w_src.origin_shape[0]),
                tla.BFloat16,
                tla.arch.nZ,
            ),
        )
        if slot == 0:
            tla.wait_flag(lf0)
        else:
            tla.wait_flag(lf1)
        tla.copy(a_l1, a_src)
        tla.copy(w_l1, w_src)
        if slot == 0:
            tla.set_flag(ld0)
        else:
            tla.set_flag(ld1)
        kcount = (a_src.origin_shape[1] + 127) // 128
        for ki in tla.range(kcount):
            aslot = ki % 2
            aa = tla.tile_view(a_l1, tla.make_shape(TM, 128), tla.make_coord(0, ki))
            al0 = tla.make_tensor_like(l0ap + aslot * 16384, aa, tla.arch.zN)
            if ki == 0:
                if slot == 0:
                    tla.wait_flag(ld0)
                else:
                    tla.wait_flag(ld1)
            if aslot == 0:
                tla.wait_flag(af0)
            else:
                tla.wait_flag(af1)
            tla.copy(al0, aa)
            for panel in tla.range_constexpr(3):
                bslot = (ki * 3 + panel) % 2
                bb = tla.tile_view(
                    wt_l1, tla.make_shape(128, 128), tla.make_coord(ki, panel)
                )
                bl0 = tla.make_tensor_like(l0bp + bslot * 16384, bb, tla.arch.nZ)
                if bslot == 0:
                    tla.wait_flag(bf0)
                else:
                    tla.wait_flag(bf1)
                tla.copy(bl0, bb)
                if tla.const_expr(panel == 2):
                    if ki == kcount - 1:
                        if slot == 0:
                            tla.set_flag(lf0)
                        else:
                            tla.set_flag(lf1)
                tla.set_flag(ad)
                tla.wait_flag(ad)
                target = tla.tile_view(
                    out, tla.make_shape(TM, 128), tla.make_coord(mi, macro * 3 + panel)
                )
                acc = tla.make_tensor_like(
                    l0cp + panel * 16384, target, tla.arch.L0Clayout
                )
                tla.mmad(acc, al0, bl0, init_c=(kb == 0 and ki == 0), unit_flag=0)
                if bslot == 0:
                    tla.set_flag(bf0)
                else:
                    tla.set_flag(bf1)
                if tla.const_expr(panel == 2):
                    if aslot == 0:
                        tla.set_flag(af0)
                    else:
                        tla.set_flag(af1)
                if kb == (K + 255) // 256 - 1 and ki == kcount - 1:
                    if tla.const_expr(panel == 0):
                        tla.set_flag(panel_ready0)
                        tla.wait_flag(panel_ready0)
                    elif tla.const_expr(panel == 1):
                        tla.set_flag(panel_ready1)
                        tla.wait_flag(panel_ready1)
                    else:
                        tla.set_flag(panel_ready2)
                        tla.wait_flag(panel_ready2)
                    if tla.const_expr(CV):
                        if tla.const_expr(panel == 0):
                            if cvslot == 0:
                                tla.cross_core_wait_flag(free0, tla.arch.FIX)
                            else:
                                tla.cross_core_wait_flag(free1, tla.arch.FIX)
                        raw = tla.make_tensor_like(
                            rawp + cvslot * 24576 + panel * 8192, acc, tla.arch.RowMajor
                        )
                        tla.copy(
                            raw,
                            acc,
                            tla.params.CopyL0C2DstParams(
                                unit_flag=0, l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ),
                        )
                    else:
                        tla.copy(target, acc, tla.params.CopyL0C2DstParams(unit_flag=0))
    if tla.const_expr(CV):
        if cvslot == 0:
            tla.cross_core_set_flag(ready0, tla.arch.FIX)
        else:
            tla.cross_core_set_flag(ready1, tla.arch.FIX)
    tla.set_flag(cf)


@tla.jit
def dual_projection_tile(
    x,
    w_raw,
    out,
    resources,
    mi,
    macro,
    HEADS: tla.Constexpr[int],
    TN: tla.Constexpr[int],
    CV: tla.Constexpr[bool],
):
    (
        l1p,
        l0ap,
        l0bp,
        l0cp,
        ld0,
        ld1,
        lf0,
        lf1,
        af0,
        af1,
        ad,
        cf,
        ready0,
        ready1,
        free0,
        free1,
        rawp,
        conv_loaded,
        conv_scalar,
        conv_done,
        conv_free,
        bf0,
        bf1,
        panel_ready0,
        panel_ready1,
        panel_ready2,
    ) = resources
    M = x.origin_shape[0]
    K = x.origin_shape[1]
    N = out.origin_shape[1]
    second = (mi + 1) * 128 < M
    tla.wait_flag(cf)
    for kb in tla.range((K + 255) // 256):
        slot = kb % 2
        a_src = tla.tile_view(x, tla.make_shape(128, 256), tla.make_coord(mi, kb))
        a_l1 = tla.make_tensor_like(l1p + slot * 65536, a_src, tla.arch.zN)
        w_src = tla.make_tensor(
            w_raw.ptr + (kb * 16 * N + macro * HEADS * TN) * 16,
            tla.make_layout(
                tla.make_shape((HEADS * TN, 1), (16, a_src.origin_shape[1] // 16)),
                tla.make_stride((16, N * 16), (1, N * 16)),
                origin_shape=tla.make_shape(HEADS * TN, a_src.origin_shape[1]),
                layoutTag=tla.arch.zNUnAlign,
            ),
        )
        wp = l1p + 131072 + slot * 65536
        w_l1 = tla.make_tensor(
            wp,
            tla.make_layout_with_tag(
                (w_src.origin_shape[0], w_src.origin_shape[1]),
                tla.BFloat16,
                tla.arch.zNUnAlign,
            ),
        )
        wt_l1 = tla.make_tensor(
            wp,
            tla.make_layout_with_tag(
                (w_src.origin_shape[1], w_src.origin_shape[0]),
                tla.BFloat16,
                tla.arch.nZ,
            ),
        )
        if slot == 0:
            tla.wait_flag(lf0)
        else:
            tla.wait_flag(lf1)
        tla.copy(a_l1, a_src)
        if second:
            gm_source1 = tla.tile_view(
                x, tla.make_shape(128, 256), tla.make_coord(mi + 1, kb)
            )
            gm_panel1 = tla.make_tensor_like(
                l1p + slot * 65536 + 32768, gm_source1, tla.arch.zN
            )
            tla.copy(gm_panel1, gm_source1)
        tla.copy(w_l1, w_src)
        if slot == 0:
            tla.set_flag(ld0)
        else:
            tla.set_flag(ld1)
        kcount = (a_src.origin_shape[1] + 127) // 128
        for ki in tla.range(kcount):
            if ki == 0:
                if slot == 0:
                    tla.wait_flag(ld0)
                else:
                    tla.wait_flag(ld1)
            aa = tla.tile_view(a_l1, tla.make_shape(128, 128), tla.make_coord(0, ki))
            al0 = tla.make_tensor_like(l0ap, aa, tla.arch.zN)
            tla.wait_flag(af0)
            tla.copy(al0, aa)
            tla.wait_flag(af1)
            if second:
                load_source1 = tla.tile_view(
                    x, tla.make_shape(128, 256), tla.make_coord(mi + 1, kb)
                )
                load_panel1 = tla.make_tensor_like(
                    l1p + slot * 65536 + 32768, load_source1, tla.arch.zN
                )
                load_aa1 = tla.tile_view(
                    load_panel1, tla.make_shape(128, 128), tla.make_coord(0, ki)
                )
                load_al1 = tla.make_tensor_like(l0ap + 16384, load_aa1, tla.arch.zN)
                tla.copy(load_al1, load_aa1)
            for head in tla.range_constexpr(HEADS):
                bb = tla.tile_view(
                    wt_l1, tla.make_shape(128, TN), tla.make_coord(ki, head)
                )
                bl0 = tla.make_tensor_like(l0bp + head * 16384, bb, tla.arch.nZ)
                if tla.const_expr(head == 0):
                    tla.wait_flag(bf0)
                else:
                    tla.wait_flag(bf1)
                tla.copy(bl0, bb)
                if tla.const_expr(head == HEADS - 1):
                    if ki == kcount - 1:
                        if slot == 0:
                            tla.set_flag(lf0)
                        else:
                            tla.set_flag(lf1)
                tla.set_flag(ad)
                tla.wait_flag(ad)
                target = tla.tile_view(
                    out,
                    tla.make_shape(128, TN),
                    tla.make_coord(mi, macro * HEADS + head),
                )
                acc = tla.make_tensor_like(
                    l0cp + head * 16384, target, tla.arch.L0Clayout
                )
                tla.mmad(acc, al0, bl0, init_c=(kb == 0 and ki == 0), unit_flag=0)
                if tla.const_expr(head == HEADS - 1):
                    tla.set_flag(af0)
                if second:
                    # Recreate descriptors in this region; no values escape the load branch.
                    mma_source1 = tla.tile_view(
                        x, tla.make_shape(128, 256), tla.make_coord(mi + 1, kb)
                    )
                    mma_panel1 = tla.make_tensor_like(
                        l1p + slot * 65536 + 32768, mma_source1, tla.arch.zN
                    )
                    mma_aa1 = tla.tile_view(
                        mma_panel1, tla.make_shape(128, 128), tla.make_coord(0, ki)
                    )
                    mma_al1 = tla.make_tensor_like(l0ap + 16384, mma_aa1, tla.arch.zN)
                    target1 = tla.tile_view(
                        out,
                        tla.make_shape(128, TN),
                        tla.make_coord(mi + 1, macro * HEADS + head),
                    )
                    acc1 = tla.make_tensor_like(
                        l0cp + (2 + head) * 16384, target1, tla.arch.L0Clayout
                    )
                    tla.mmad(
                        acc1, mma_al1, bl0, init_c=(kb == 0 and ki == 0), unit_flag=0
                    )
                if tla.const_expr(head == HEADS - 1):
                    tla.set_flag(af1)
                if tla.const_expr(head == 0):
                    tla.set_flag(bf0)
                else:
                    tla.set_flag(bf1)
    tla.set_flag(panel_ready0)
    tla.wait_flag(panel_ready0)
    for row_panel in tla.range_constexpr(2):
        if (mi + row_panel) * 128 < M:
            if tla.const_expr(CV):
                if tla.const_expr(row_panel == 0):
                    tla.cross_core_wait_flag(free0, tla.arch.FIX)
                else:
                    tla.cross_core_wait_flag(free1, tla.arch.FIX)
            for output_head in tla.range_constexpr(HEADS):
                output_target = tla.tile_view(
                    out,
                    tla.make_shape(128, TN),
                    tla.make_coord(mi + row_panel, macro * HEADS + output_head),
                )
                output_acc = tla.make_tensor_like(
                    l0cp + (row_panel * 2 + output_head) * 16384,
                    output_target,
                    tla.arch.L0Clayout,
                )
                if tla.const_expr(CV):
                    raw = tla.make_tensor_like(
                        rawp + row_panel * 16384 + output_head * 8192,
                        output_acc,
                        tla.arch.RowMajor,
                    )
                    tla.copy(
                        raw,
                        output_acc,
                        tla.params.CopyL0C2DstParams(
                            unit_flag=0, l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                        ),
                    )
                else:
                    tla.copy(
                        output_target,
                        output_acc,
                        tla.params.CopyL0C2DstParams(unit_flag=0),
                    )
            if tla.const_expr(CV):
                if tla.const_expr(row_panel == 0):
                    tla.cross_core_set_flag(ready0, tla.arch.FIX)
                else:
                    tla.cross_core_set_flag(ready1, tla.arch.FIX)
    tla.set_flag(cf)


def allocate_projection_resources(max_n=128, dual_m=False):
    # K512/K128: 512 KiB L1, 64 KiB L0A/B, 64 KiB L0C.
    l1p = tla.allocate(262144, tla.BFloat16, tla.AddressSpace.l1, 512)
    l0ap = tla.allocate(32768, tla.BFloat16, tla.AddressSpace.l0a, 512)
    l0bp = tla.allocate(32768, tla.BFloat16, tla.AddressSpace.l0b, 512)
    l0cp = tla.allocate(
        65536 if dual_m else 128 * max_n, tla.Float32, tla.AddressSpace.l0c, 512
    )
    ld0 = tla.flag("l1_loaded_0", tla.arch.MTE2, tla.arch.MTE1)
    ld1 = tla.flag("l1_loaded_1", tla.arch.MTE2, tla.arch.MTE1)
    lf0 = tla.flag("l1_free_0", tla.arch.MTE1, tla.arch.MTE2)
    lf1 = tla.flag("l1_free_1", tla.arch.MTE1, tla.arch.MTE2)
    af0 = tla.flag("l0_free_0", tla.arch.CUBE, tla.arch.MTE1)
    af1 = tla.flag("l0_free_1", tla.arch.CUBE, tla.arch.MTE1)
    ad = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)
    cf = tla.flag("l0c_free", tla.arch.FIX, tla.arch.CUBE)
    ready0 = tla.cross_flag("cv_ready0", mode=2)
    ready1 = tla.cross_flag("cv_ready1", mode=2)
    free0 = tla.cross_flag("cv_free0", mode=2)
    free1 = tla.cross_flag("cv_free1", mode=2)
    # S0/S1 wide-CV prefix and S3's three states reuse this arena by phase.
    rawp = tla.allocate(
        61440 if max_n == 384 and not dual_m else 57344,
        tla.Float32,
        tla.AddressSpace.ub,
        256,
    )
    conv_loaded = tla.flag("conv_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    conv_scalar = tla.flag("conv_scalar", tla.arch.SCALAR, tla.arch.VECTOR)
    conv_done = tla.flag("conv_done", tla.arch.VECTOR, tla.arch.MTE3)
    conv_free = tla.flag("conv_free", tla.arch.MTE3, tla.arch.MTE2)
    bf0 = tla.flag("resident_b_free0", tla.arch.CUBE, tla.arch.MTE1)
    bf1 = tla.flag("resident_b_free1", tla.arch.CUBE, tla.arch.MTE1)
    panel_ready0 = tla.flag("resident_panel0", tla.arch.CUBE, tla.arch.FIX)
    panel_ready1 = tla.flag("resident_panel1", tla.arch.CUBE, tla.arch.FIX)
    panel_ready2 = tla.flag("resident_panel2", tla.arch.CUBE, tla.arch.FIX)
    return (
        l1p,
        l0ap,
        l0bp,
        l0cp,
        ld0,
        ld1,
        lf0,
        lf1,
        af0,
        af1,
        ad,
        cf,
        ready0,
        ready1,
        free0,
        free1,
        rawp,
        conv_loaded,
        conv_scalar,
        conv_done,
        conv_free,
        bf0,
        bf1,
        panel_ready0,
        panel_ready1,
        panel_ready2,
    )


@tla.jit
def projection_tile(
    x,
    w_raw,
    out,
    resources,
    mi,
    ni,
    work,
    TM: tla.Constexpr[int],
    TN: tla.Constexpr[int],
    CV: tla.Constexpr[bool],
    RESIDENT_ARENA: tla.Constexpr[bool] = False,
):
    (
        l1p,
        l0ap,
        l0bp,
        l0cp,
        ld0,
        ld1,
        lf0,
        lf1,
        af0,
        af1,
        ad,
        cf,
        ready0,
        ready1,
        free0,
        free1,
        rawp,
        conv_loaded,
        conv_scalar,
        conv_done,
        conv_free,
        bf0,
        bf1,
        panel_ready0,
        panel_ready1,
        panel_ready2,
    ) = resources
    K = x.origin_shape[1]
    N = out.origin_shape[1]
    TK = 256 if TN == 256 or RESIDENT_ARENA else 512
    K0 = 64 if TN == 256 else 128
    target = tla.tile_view(out, tla.make_shape(TM, TN), tla.make_coord(mi, ni))
    acc = tla.make_tensor_like(l0cp, target, tla.arch.L0Clayout)
    tla.wait_flag(cf)
    for kb in tla.range((K + TK - 1) // TK):
        slot = kb % 2
        a_src = tla.tile_view(x, tla.make_shape(TM, TK), tla.make_coord(mi, kb))
        w_src = tla.make_tensor(
            w_raw.ptr + (kb * (TK // 16) * N + ni * TN) * 16,
            tla.make_layout(
                tla.make_shape((TN, 1), (16, a_src.origin_shape[1] // 16)),
                tla.make_stride((16, N * 16), (1, N * 16)),
                origin_shape=tla.make_shape(TN, a_src.origin_shape[1]),
                layoutTag=tla.arch.zNUnAlign,
            ),
        )
        ap = l1p + slot * (32768 if RESIDENT_ARENA else 65536)
        wp = (
            l1p
            + (65536 if RESIDENT_ARENA else 131072)
            + slot * (98304 if RESIDENT_ARENA else 65536)
        )
        a_l1 = tla.make_tensor_like(ap, a_src, tla.arch.zN)
        w_l1 = tla.make_tensor(
            wp,
            tla.make_layout_with_tag(
                (w_src.origin_shape[0], w_src.origin_shape[1]),
                tla.BFloat16,
                tla.arch.zNUnAlign,
            ),
        )
        # zN(W[N,K]) equals nZ(W.T[K,N]) in physical storage.
        wt_l1 = tla.make_tensor(
            wp,
            tla.make_layout_with_tag(
                (w_src.origin_shape[1], w_src.origin_shape[0]),
                tla.BFloat16,
                tla.arch.nZ,
            ),
        )
        if slot == 0:
            tla.wait_flag(lf0)
        else:
            tla.wait_flag(lf1)
        tla.copy(a_l1, a_src)
        tla.copy(w_l1, w_src)
        if slot == 0:
            tla.set_flag(ld0)
        else:
            tla.set_flag(ld1)
        kcount = (a_src.origin_shape[1] + K0 - 1) // K0
        for ki in tla.range(kcount):
            lslot = ki % 2
            aa = tla.tile_view(a_l1, tla.make_shape(TM, K0), tla.make_coord(0, ki))
            bb = tla.tile_view(wt_l1, tla.make_shape(K0, TN), tla.make_coord(ki, 0))
            al0 = tla.make_tensor_like(l0ap + lslot * 16384, aa, tla.arch.zN)
            bl0 = tla.make_tensor_like(l0bp + lslot * 16384, bb, tla.arch.nZ)
            if ki == 0:
                if slot == 0:
                    tla.wait_flag(ld0)
                else:
                    tla.wait_flag(ld1)
            if lslot == 0:
                tla.wait_flag(af0)
            else:
                tla.wait_flag(af1)
            tla.copy(al0, aa)
            if tla.const_expr(RESIDENT_ARENA):
                if lslot == 0:
                    tla.wait_flag(bf0)
                else:
                    tla.wait_flag(bf1)
            tla.copy(bl0, bb)
            if ki == kcount - 1:
                if slot == 0:
                    tla.set_flag(lf0)
                else:
                    tla.set_flag(lf1)
            tla.set_flag(ad)
            tla.wait_flag(ad)
            last = (kb == (K + TK - 1) // TK - 1) and (ki == kcount - 1)
            uf = 3 if last else 2
            init = (kb == 0) and (ki == 0)
            tla.mmad(acc, al0, bl0, init_c=init, unit_flag=uf)
            if tla.const_expr(RESIDENT_ARENA):
                if lslot == 0:
                    tla.set_flag(bf0)
                else:
                    tla.set_flag(bf1)
            if lslot == 0:
                tla.set_flag(af0)
            else:
                tla.set_flag(af1)
    if tla.const_expr(CV):
        cvslot = (work // tla.arch.block_num()) % 2
        if cvslot == 0:
            tla.cross_core_wait_flag(free0, tla.arch.FIX)
        else:
            tla.cross_core_wait_flag(free1, tla.arch.FIX)
        raw = tla.make_tensor_like(rawp + cvslot * 64 * TN, acc, tla.arch.RowMajor)
        tla.copy(
            raw,
            acc,
            tla.params.CopyL0C2DstParams(
                unit_flag=3, l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
            ),
        )
        if cvslot == 0:
            tla.cross_core_set_flag(ready0, tla.arch.FIX)
        else:
            tla.cross_core_set_flag(ready1, tla.arch.FIX)
    else:
        tla.copy(target, acc, tla.params.CopyL0C2DstParams(unit_flag=3))
    tla.set_flag(cf)


@tla.jit
def projection_stage(
    x: tla.Tensor,
    w_raw: tla.Tensor,
    out: tla.Tensor,
    cw: tla.Tensor,
    cs: tla.Tensor,
    ci: tla.Tensor,
    ca: tla.Tensor,
    resources,
    TM: tla.Constexpr[int],
    TN: tla.Constexpr[int],
    CV: tla.Constexpr[bool],
    CONV: tla.Constexpr[bool],
    L: tla.Constexpr[int],
):
    (
        l1p,
        l0ap,
        l0bp,
        l0cp,
        ld0,
        ld1,
        lf0,
        lf1,
        af0,
        af1,
        ad,
        cf,
        ready0,
        ready1,
        free0,
        free1,
        rawp,
        conv_loaded,
        conv_scalar,
        conv_done,
        conv_free,
        bf0,
        bf1,
        panel_ready0,
        panel_ready1,
        panel_ready2,
    ) = resources
    M = x.origin_shape[0]
    K = x.origin_shape[1]
    N = out.origin_shape[1]
    gridn = (N + TN - 1) // TN
    nwork = ((M + TM - 1) // TM) * gridn

    with tla.cube():
        tla.set_flag(lf0)
        tla.set_flag(lf1)
        tla.set_flag(af0)
        tla.set_flag(af1)
        tla.set_flag(cf)
        if tla.const_expr(TN == 384):
            tla.set_flag(bf0)
            tla.set_flag(bf1)
        for work in tla.range(tla.arch.block_idx(), nwork, tla.arch.block_num()):
            mi = work // gridn
            ni = work % gridn
            if tla.const_expr(TN == 384):
                resident_projection_tile(x, w_raw, out, resources, mi, ni, work, TM, CV)
            else:
                projection_tile(x, w_raw, out, resources, mi, ni, work, TM, TN, CV)
        tla.wait_flag(lf0)
        tla.wait_flag(lf1)
        tla.wait_flag(af0)
        tla.wait_flag(af1)
        tla.wait_flag(cf)
        if tla.const_expr(TN == 384):
            tla.wait_flag(bf0)
            tla.wait_flag(bf1)
        if tla.const_expr(CV):
            tla.cross_core_wait_flag(free0, tla.arch.FIX)
            tla.cross_core_wait_flag(free1, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

    if tla.const_expr(CV):
        with tla.vector():
            tla.cross_core_set_flag(free0, tla.arch.MTE3)
            tla.cross_core_set_flag(free1, tla.arch.MTE3)
            if tla.const_expr(CONV):
                tla.set_flag(conv_free)
            for work in tla.range(tla.arch.block_idx(), nwork, tla.arch.block_num()):
                mi = work // gridn
                ni = work % gridn
                cvslot = (work // tla.arch.block_num()) % 2
                remaining = M - mi * TM
                valid = TM if remaining >= TM else remaining
                half = valid // 2
                dst = tla.make_tensor(
                    out.ptr + (mi * TM + tla.arch.sub_block_idx() * half) * N + ni * TN,
                    tla.make_layout(tla.make_shape(half, TN), tla.make_stride(N, 1)),
                )
                raw = tla.make_tensor_like(
                    rawp + cvslot * 64 * TN, dst, tla.arch.RowMajor
                )
                if tla.const_expr(CONV):
                    if cvslot == 0:
                        tla.cross_core_wait_flag(ready0, tla.arch.VECTOR)
                    else:
                        tla.cross_core_wait_flag(ready1, tla.arch.VECTOR)
                    first_row = mi * TM + tla.arch.sub_block_idx() * half
                    for head_lane in tla.range_constexpr(TN // 128):
                        conv_dst = tla.make_tensor(
                            out.ptr + first_row * N + ni * TN + head_lane * 128,
                            tla.make_layout(
                                tla.make_shape(half, 128), tla.make_stride(N, 1)
                            ),
                        )
                        if tla.const_expr(TN == 384):
                            conv_consume(
                                rawp + cvslot * 24576 + head_lane * 8192,
                                rawp + 32768,
                                conv_dst,
                                cw,
                                cs,
                                ci,
                                ca,
                                first_row,
                                half,
                                ni * TN + head_lane * 128,
                                cvslot,
                                L,
                                conv_loaded,
                                conv_scalar,
                                conv_done,
                                conv_free,
                                128,
                                True,
                            )
                        else:
                            conv_consume(
                                rawp + cvslot * 64 * TN + head_lane * 128,
                                rawp + (TN // 128 - 1) * 16384,
                                conv_dst,
                                cw,
                                cs,
                                ci,
                                ca,
                                first_row,
                                half,
                                ni * TN + head_lane * 128,
                                cvslot,
                                L,
                                conv_loaded,
                                conv_scalar,
                                conv_done,
                                conv_free,
                                TN,
                            )
                    if cvslot == 0:
                        tla.cross_core_set_flag(free0, tla.arch.VECTOR)
                    else:
                        tla.cross_core_set_flag(free1, tla.arch.VECTOR)
                else:
                    if cvslot == 0:
                        tla.cross_core_wait_flag(ready0, tla.arch.MTE3)
                    else:
                        tla.cross_core_wait_flag(ready1, tla.arch.MTE3)
                    if tla.const_expr(TN == 384):
                        for output_panel in tla.range_constexpr(3):
                            raw_panel = tla.make_tensor(
                                rawp + cvslot * 24576 + output_panel * 8192,
                                tla.make_layout(
                                    tla.make_shape(half, 128), tla.make_stride(128, 1)
                                ),
                            )
                            dst_panel = tla.make_tensor(
                                out.ptr
                                + (mi * TM + tla.arch.sub_block_idx() * half) * N
                                + ni * TN
                                + output_panel * 128,
                                tla.make_layout(
                                    tla.make_shape(half, 128), tla.make_stride(N, 1)
                                ),
                            )
                            tla.copy(dst_panel, raw_panel)
                    else:
                        tla.copy(dst, raw)
                    if cvslot == 0:
                        tla.cross_core_set_flag(free0, tla.arch.MTE3)
                    else:
                        tla.cross_core_set_flag(free1, tla.arch.MTE3)
            if tla.const_expr(CONV):
                tla.wait_flag(conv_free)
            tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def projection_kernel(
    x: tla.Tensor,
    w_raw: tla.Tensor,
    out: tla.Tensor,
    cw: tla.Tensor,
    cs: tla.Tensor,
    ci: tla.Tensor,
    ca: tla.Tensor,
    TM: tla.Constexpr[int],
    TN: tla.Constexpr[int],
    CV: tla.Constexpr[bool],
    CONV: tla.Constexpr[bool],
    L: tla.Constexpr[int],
):
    resources = allocate_projection_resources(max(128, TN))
    projection_stage(x, w_raw, out, cw, cs, ci, ca, resources, TM, TN, CV, CONV, L)


@tla.jit
def initial_stage(
    x,
    wqkv,
    wa,
    wbeta,
    wz,
    qkv,
    a,
    beta,
    z,
    cw,
    cs,
    ci,
    ca,
    resources,
    N: tla.Constexpr[int],
    L: tla.Constexpr[int],
    TM: tla.Constexpr[int],
    NB: tla.Constexpr[int],
    INIT_N: tla.Constexpr[int],
):
    (
        l1p,
        l0ap,
        l0bp,
        l0cp,
        ld0,
        ld1,
        lf0,
        lf1,
        af0,
        af1,
        ad,
        cf,
        ready0,
        ready1,
        free0,
        free1,
        rawp,
        conv_loaded,
        conv_scalar,
        conv_done,
        conv_free,
        bf0,
        bf1,
        panel_ready0,
        panel_ready1,
        panel_ready2,
    ) = resources
    M = x.origin_shape[0]
    P = N * 128
    heads_per_tile = INIT_N // 128
    qtiles = 3 * N // heads_per_tile
    group = 4 * N // heads_per_tile + 2
    works = ((M + TM - 1) // TM) * group
    with tla.cube():
        tla.set_flag(lf0)
        tla.set_flag(lf1)
        tla.set_flag(af0)
        tla.set_flag(af1)
        tla.set_flag(cf)
        if tla.const_expr(INIT_N == 384):
            tla.set_flag(bf0)
            tla.set_flag(bf1)
        for work in tla.range(tla.arch.block_idx(), works, tla.arch.block_num()):
            mi = work // group
            channel = work % group
            if channel < qtiles:
                if tla.const_expr(INIT_N == 384):
                    resident_projection_tile(
                        x, wqkv, qkv, resources, mi, channel, work, TM, True
                    )
                else:
                    projection_tile(
                        x, wqkv, qkv, resources, mi, channel, work, TM, INIT_N, True
                    )
            elif channel == qtiles:
                projection_tile(
                    x, wa, a, resources, mi, 0, work, TM, 128, False, INIT_N == 384
                )
            elif channel == qtiles + 1:
                projection_tile(
                    x, wbeta, beta, resources, mi, 0, work, TM, NB, False, INIT_N == 384
                )
            else:
                if tla.const_expr(INIT_N == 384):
                    resident_projection_tile(
                        x, wz, z, resources, mi, channel - qtiles - 2, work, TM, False
                    )
                else:
                    projection_tile(
                        x,
                        wz,
                        z,
                        resources,
                        mi,
                        channel - qtiles - 2,
                        work,
                        TM,
                        INIT_N,
                        False,
                    )
        tla.wait_flag(lf0)
        tla.wait_flag(lf1)
        tla.wait_flag(af0)
        tla.wait_flag(af1)
        tla.wait_flag(cf)
        if tla.const_expr(INIT_N == 384):
            tla.wait_flag(bf0)
            tla.wait_flag(bf1)
        tla.cross_core_wait_flag(free0, tla.arch.FIX)
        tla.cross_core_wait_flag(free1, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla.cross_core_set_flag(free0, tla.arch.MTE3)
        tla.cross_core_set_flag(free1, tla.arch.MTE3)
        tla.set_flag(conv_free)
        for vwork in tla.range(tla.arch.block_idx(), works, tla.arch.block_num()):
            vmi = vwork // group
            vchannel = vwork % group
            if vchannel < qtiles:
                cvslot = (vwork // tla.arch.block_num()) % 2
                remaining = M - vmi * TM
                valid = TM if remaining >= TM else remaining
                half = valid // 2
                first_row = vmi * TM + tla.arch.sub_block_idx() * half
                if cvslot == 0:
                    tla.cross_core_wait_flag(ready0, tla.arch.VECTOR)
                else:
                    tla.cross_core_wait_flag(ready1, tla.arch.VECTOR)
                for head_lane in tla.range_constexpr(heads_per_tile):
                    channel_start = vchannel * INIT_N + head_lane * 128
                    dst = tla.make_tensor(
                        qkv.ptr + first_row * 3 * P + channel_start,
                        tla.make_layout(
                            tla.make_shape(half, 128), tla.make_stride(3 * P, 1)
                        ),
                    )
                    if tla.const_expr(INIT_N == 384):
                        conv_consume(
                            rawp + cvslot * 24576 + head_lane * 8192,
                            rawp + 32768,
                            dst,
                            cw,
                            cs,
                            ci,
                            ca,
                            first_row,
                            half,
                            channel_start,
                            cvslot,
                            L,
                            conv_loaded,
                            conv_scalar,
                            conv_done,
                            conv_free,
                            128,
                            True,
                        )
                    else:
                        conv_consume(
                            rawp + cvslot * 64 * INIT_N + head_lane * 128,
                            rawp + (heads_per_tile - 1) * 16384,
                            dst,
                            cw,
                            cs,
                            ci,
                            ca,
                            first_row,
                            half,
                            channel_start,
                            cvslot,
                            L,
                            conv_loaded,
                            conv_scalar,
                            conv_done,
                            conv_free,
                            INIT_N,
                        )
                if cvslot == 0:
                    tla.cross_core_set_flag(free0, tla.arch.VECTOR)
                else:
                    tla.cross_core_set_flag(free1, tla.arch.VECTOR)
        tla.wait_flag(conv_free)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.jit
def prime_dual(resources):
    tla.set_flag(resources[6])
    tla.set_flag(resources[7])
    tla.set_flag(resources[8])
    tla.set_flag(resources[9])
    tla.set_flag(resources[11])
    tla.set_flag(resources[21])
    tla.set_flag(resources[22])


@tla.jit
def drain_dual(resources, CV: tla.Constexpr[bool] = True):
    tla.wait_flag(resources[6])
    tla.wait_flag(resources[7])
    tla.wait_flag(resources[8])
    tla.wait_flag(resources[9])
    tla.wait_flag(resources[11])
    tla.wait_flag(resources[21])
    tla.wait_flag(resources[22])
    if tla.const_expr(CV):
        tla.cross_core_wait_flag(resources[14], tla.arch.FIX)
        tla.cross_core_wait_flag(resources[15], tla.arch.FIX)
    tla.pipe_barrier(tla.pipes.ALL)


@tla.jit
def consume_dual(
    out,
    cw,
    cs,
    ci,
    ca,
    resources,
    mi,
    macro,
    HEADS: tla.Constexpr[int],
    TN: tla.Constexpr[int],
    L: tla.Constexpr[int],
    CONV: tla.Constexpr[bool],
):
    rawp = resources[16]
    M = out.origin_shape[0]
    width = out.origin_shape[1]
    for row_panel in tla.range_constexpr(2):
        if (mi + row_panel) * 128 < M:
            remaining = M - (mi + row_panel) * 128
            valid = 128 if remaining >= 128 else remaining
            half = valid // 2
            first_row = (mi + row_panel) * 128 + tla.arch.sub_block_idx() * half
            if tla.const_expr(row_panel == 0):
                tla.cross_core_wait_flag(
                    resources[12], tla.arch.VECTOR if CONV else tla.arch.MTE3
                )
            else:
                tla.cross_core_wait_flag(
                    resources[13], tla.arch.VECTOR if CONV else tla.arch.MTE3
                )
            for head in tla.range_constexpr(HEADS):
                channel = (macro * HEADS + head) * TN
                dst = tla.make_tensor(
                    out.ptr + first_row * width + channel,
                    tla.make_layout(
                        tla.make_shape(half, TN), tla.make_stride(width, 1)
                    ),
                )
                if tla.const_expr(CONV):
                    conv_consume(
                        rawp + row_panel * 16384 + head * 8192,
                        rawp + 16384,
                        dst,
                        cw,
                        cs,
                        ci,
                        ca,
                        first_row,
                        half,
                        channel,
                        row_panel,
                        L,
                        resources[17],
                        resources[18],
                        resources[19],
                        resources[20],
                        128,
                        True,
                    )
                else:
                    raw = tla.make_tensor(
                        rawp + row_panel * 16384 + head * 8192,
                        tla.make_layout(
                            tla.make_shape(half, TN), tla.make_stride(TN, 1)
                        ),
                    )
                    tla.copy(dst, raw)
            if tla.const_expr(row_panel == 0):
                tla.cross_core_set_flag(
                    resources[14], tla.arch.VECTOR if CONV else tla.arch.MTE3
                )
            else:
                tla.cross_core_set_flag(
                    resources[15], tla.arch.VECTOR if CONV else tla.arch.MTE3
                )


@tla.jit
def initial_dual_stage(
    x,
    wqkv,
    wa,
    wbeta,
    wz,
    qkv,
    a,
    beta,
    z,
    cw,
    cs,
    ci,
    ca,
    resources,
    N: tla.Constexpr[int],
    L: tla.Constexpr[int],
    NB: tla.Constexpr[int],
):
    m_groups = (x.origin_shape[0] + 255) // 256
    qtiles = 3 * N // 2
    group = 2 * N + 2
    works = m_groups * group
    with tla.cube():
        prime_dual(resources)
        for work in tla.range(tla.arch.block_idx(), works, tla.arch.block_num()):
            mi = 2 * (work % m_groups)
            channel = work // m_groups
            if channel < qtiles:
                dual_projection_tile(x, wqkv, qkv, resources, mi, channel, 2, 128, True)
            elif channel == qtiles:
                dual_projection_tile(x, wa, a, resources, mi, 0, 1, 128, False)
            elif channel == qtiles + 1:
                dual_projection_tile(x, wbeta, beta, resources, mi, 0, 1, NB, False)
            else:
                dual_projection_tile(
                    x, wz, z, resources, mi, channel - qtiles - 2, 2, 128, False
                )
        drain_dual(resources)
    with tla.vector():
        tla.cross_core_set_flag(resources[14], tla.arch.MTE3)
        tla.cross_core_set_flag(resources[15], tla.arch.MTE3)
        tla.set_flag(resources[20])
        for work in tla.range(tla.arch.block_idx(), works, tla.arch.block_num()):
            mi = 2 * (work % m_groups)
            channel = work // m_groups
            if channel < qtiles:
                consume_dual(
                    qkv, cw, cs, ci, ca, resources, mi, channel, 2, 128, L, True
                )
        tla.wait_flag(resources[20])
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def dual_projection_kernel(
    x: tla.Tensor,
    w: tla.Tensor,
    out: tla.Tensor,
    cw: tla.Tensor,
    cs: tla.Tensor,
    ci: tla.Tensor,
    ca: tla.Tensor,
    HEADS: tla.Constexpr[int],
    TN: tla.Constexpr[int],
    CV: tla.Constexpr[bool],
    CONV: tla.Constexpr[bool],
    L: tla.Constexpr[int],
):
    resources = allocate_projection_resources(256, True)
    m_groups = (x.origin_shape[0] + 255) // 256
    works = m_groups * (out.origin_shape[1] // (HEADS * TN))
    with tla.cube():
        prime_dual(resources)
        for work in tla.range(tla.arch.block_idx(), works, tla.arch.block_num()):
            dual_projection_tile(
                x,
                w,
                out,
                resources,
                2 * (work % m_groups),
                work // m_groups,
                HEADS,
                TN,
                CV,
            )
        drain_dual(resources, CV)
    if tla.const_expr(CV):
        with tla.vector():
            tla.cross_core_set_flag(resources[14], tla.arch.MTE3)
            tla.cross_core_set_flag(resources[15], tla.arch.MTE3)
            if tla.const_expr(CONV):
                tla.set_flag(resources[20])
            for work in tla.range(tla.arch.block_idx(), works, tla.arch.block_num()):
                consume_dual(
                    out,
                    cw,
                    cs,
                    ci,
                    ca,
                    resources,
                    2 * (work % m_groups),
                    work // m_groups,
                    HEADS,
                    TN,
                    L,
                    CONV,
                )
            if tla.const_expr(CONV):
                tla.wait_flag(resources[20])
            tla.pipe_barrier(tla.pipes.ALL)


@tla.jit
def prefetch_output_weight(w, resources):
    width = w.origin_shape[0]
    ni = tla.arch.block_idx()
    source = tla.make_tensor(
        w.ptr + ni * 256 * 16,
        tla.make_layout(
            tla.make_shape((256, 1), (16, 48)),
            tla.make_stride((16, width * 16), (1, width * 16)),
            origin_shape=tla.make_shape(256, 768),
            layoutTag=tla.arch.zNUnAlign,
        ),
    )
    resident = tla.make_tensor(
        resources[0] + 65536,
        tla.make_layout_with_tag((256, 768), tla.BFloat16, tla.arch.zNUnAlign),
    )
    tla.copy(resident, source)
    tla.set_flag(resources[4])
    tla.wait_flag(resources[4])


@tla.jit
def output_resident_tile(x, out, resources, mi):
    l1p = resources[0]
    l0ap = resources[1]
    l0bp = resources[2]
    l0cp = resources[3]
    ld0 = resources[4]
    ld1 = resources[5]
    lf0 = resources[6]
    lf1 = resources[7]
    af0 = resources[8]
    af1 = resources[9]
    ad = resources[10]
    cf = resources[11]
    target = tla.tile_view(
        out, tla.make_shape(128, 256), tla.make_coord(mi, tla.arch.block_idx())
    )
    acc = tla.make_tensor_like(l0cp, target, tla.arch.L0Clayout)
    weight = tla.make_tensor(
        l1p + 65536, tla.make_layout_with_tag((768, 256), tla.BFloat16, tla.arch.nZ)
    )
    tla.wait_flag(cf)
    for kb in tla.range(3):
        slot = kb % 2
        source = tla.tile_view(x, tla.make_shape(128, 256), tla.make_coord(mi, kb))
        panel = tla.make_tensor_like(l1p + slot * 32768, source, tla.arch.zN)
        if slot == 0:
            tla.wait_flag(lf0)
        else:
            tla.wait_flag(lf1)
        tla.copy(panel, source)
        if slot == 0:
            tla.set_flag(ld0)
        else:
            tla.set_flag(ld1)
        for ki in tla.range(4):
            lslot = ki % 2
            a = tla.tile_view(panel, tla.make_shape(128, 64), tla.make_coord(0, ki))
            b = tla.tile_view(
                weight, tla.make_shape(64, 256), tla.make_coord(kb * 4 + ki, 0)
            )
            al0 = tla.make_tensor_like(l0ap + lslot * 16384, a, tla.arch.zN)
            bl0 = tla.make_tensor_like(l0bp + lslot * 16384, b, tla.arch.nZ)
            if ki == 0:
                if slot == 0:
                    tla.wait_flag(ld0)
                else:
                    tla.wait_flag(ld1)
            if lslot == 0:
                tla.wait_flag(af0)
            else:
                tla.wait_flag(af1)
            tla.copy(al0, a)
            tla.copy(bl0, b)
            if ki == 3:
                if slot == 0:
                    tla.set_flag(lf0)
                else:
                    tla.set_flag(lf1)
            tla.set_flag(ad)
            tla.wait_flag(ad)
            last = kb == 2 and ki == 3
            tla.mmad(
                acc, al0, bl0, init_c=(kb == 0 and ki == 0), unit_flag=3 if last else 2
            )
            if lslot == 0:
                tla.set_flag(af0)
            else:
                tla.set_flag(af1)
    tla.copy(target, acc, tla.params.CopyL0C2DstParams(unit_flag=3))
    tla.set_flag(cf)


@tla.kernel
def output_resident_kernel(
    x: tla.Tensor,
    w: tla.Tensor,
    out: tla.Tensor,
    cw: tla.Tensor,
    cs: tla.Tensor,
    ci: tla.Tensor,
    ca: tla.Tensor,
    TM: tla.Constexpr[int],
    TN: tla.Constexpr[int],
    CV: tla.Constexpr[bool],
    CONV: tla.Constexpr[bool],
    L: tla.Constexpr[int],
):
    resources = allocate_projection_resources(256)
    with tla.cube():
        prefetch_output_weight(w, resources)
        tla.set_flag(resources[6])
        tla.set_flag(resources[7])
        tla.set_flag(resources[8])
        tla.set_flag(resources[9])
        tla.set_flag(resources[11])
        for mi in tla.range((x.origin_shape[0] + 127) // 128):
            output_resident_tile(x, out, resources, mi)
        tla.wait_flag(resources[6])
        tla.wait_flag(resources[7])
        tla.wait_flag(resources[8])
        tla.wait_flag(resources[9])
        tla.wait_flag(resources[11])
        tla.pipe_barrier(tla.pipes.ALL)


@tla.jit
def projection_chunks(
    u,
    w,
    out,
    pr,
    rr,
    cube_bar,
    TN: tla.Constexpr[int] = 128,
    RESIDENT_WEIGHT: tla.Constexpr[bool] = False,
):
    lf0 = pr[6]
    lf1 = pr[7]
    af0 = pr[8]
    af1 = pr[9]
    cf = pr[11]
    ready0 = rr[5]
    ready1 = rr[6]
    ack0 = rr[7]
    ack1 = rr[8]
    M = u.origin_shape[0]
    gridn = out.origin_shape[1] // TN
    with tla.cube():
        if tla.const_expr(RESIDENT_WEIGHT):
            prefetch_output_weight(w, pr)
        # MTE1 primes lf only after consuming the resident weight's ld0.
        tla.set_flag(lf0)
        tla.set_flag(lf1)
        tla.set_flag(af0)
        tla.set_flag(af1)
        tla.set_flag(cf)
        for chunk in tla.range((M + 127) // 128):
            if chunk % 2 == 0:
                tla.cross_core_wait_flag(ready0, tla.arch.MTE2)
            else:
                tla.cross_core_wait_flag(ready1, tla.arch.MTE2)
            tla.cross_core_set_flag(cube_bar, tla.arch.MTE2)
            tla.cross_core_wait_flag(cube_bar, tla.arch.MTE2)
            # The full GM u buffer is never reused; ack frees only this notification.
            if chunk % 2 == 0:
                tla.cross_core_set_flag(ack0, tla.arch.MTE2)
            else:
                tla.cross_core_set_flag(ack1, tla.arch.MTE2)
            if tla.const_expr(RESIDENT_WEIGHT):
                output_resident_tile(u, out, pr, chunk)
            else:
                # Preserve the flat cyclic assignment for either output tile width.
                first_ni = (
                    tla.arch.block_idx()
                    + tla.arch.block_num()
                    - (chunk * gridn) % tla.arch.block_num()
                ) % tla.arch.block_num()
                for ni in tla.range(first_ni, gridn, tla.arch.block_num()):
                    projection_tile(
                        u, w, out, pr, chunk, ni, chunk * gridn + ni, 128, TN, False
                    )
        tla.wait_flag(lf0)
        tla.wait_flag(lf1)
        tla.wait_flag(af0)
        tla.wait_flag(af1)
        tla.wait_flag(cf)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def backend_kernel(
    qkv: tla.Tensor,
    g: tla.Tensor,
    beta: tla.Tensor,
    z: tla.Tensor,
    state: tla.Tensor,
    indices: tla.Tensor,
    accepted: tla.Tensor,
    alog: tla.Tensor,
    bias: tla.Tensor,
    gamma: tla.Tensor,
    u: tla.Tensor,
    w: tla.Tensor,
    out: tla.Tensor,
    N: tla.Constexpr[int],
    SCALE: tla.Constexpr[float],
    LOWER: tla.Constexpr[float],
    EPS: tla.Constexpr[float],
    OUTPUT_N: tla.Constexpr[int] = 128,
    RESIDENT_WEIGHT: tla.Constexpr[bool] = False,
):
    pr = allocate_projection_resources(max(128, OUTPUT_N))
    rr = allocate_recurrent_resources(pr[16])
    cube_bar = tla.cross_flag("backend_cube_barrier", mode=0)
    recurrent_stage(
        qkv,
        g,
        beta,
        z,
        state,
        indices,
        accepted,
        alog,
        bias,
        gamma,
        u,
        rr,
        N,
        SCALE,
        LOWER,
        EPS,
        True,
    )
    projection_chunks(u, w, out, pr, rr, cube_bar, OUTPUT_N, RESIDENT_WEIGHT)


# Fused C1V2 entry


@tla.kernel
def mega_kernel(
    x: tla.Tensor,
    wqkv: tla.Tensor,
    wa: tla.Tensor,
    wb: tla.Tensor,
    wbeta: tla.Tensor,
    wz: tla.Tensor,
    gamma: tla.Tensor,
    wo: tla.Tensor,
    cw: tla.Tensor,
    cs: tla.Tensor,
    state: tla.Tensor,
    ci: tla.Tensor,
    indices: tla.Tensor,
    ca: tla.Tensor,
    accepted: tla.Tensor,
    alog: tla.Tensor,
    bias: tla.Tensor,
    qkv: tla.Tensor,
    a: tla.Tensor,
    g: tla.Tensor,
    beta: tla.Tensor,
    z: tla.Tensor,
    u: tla.Tensor,
    out: tla.Tensor,
    N: tla.Constexpr[int],
    L: tla.Constexpr[int],
    TM: tla.Constexpr[int],
    NB: tla.Constexpr[int],
    INIT_N: tla.Constexpr[int],
    PIPELINED: tla.Constexpr[bool],
    OUTPUT_N: tla.Constexpr[int],
    DUAL_M: tla.Constexpr[bool],
    OUTPUT_RESIDENT: tla.Constexpr[bool],
    SCALE: tla.Constexpr[float],
    LOWER: tla.Constexpr[float],
    EPS: tla.Constexpr[float],
):
    pr = allocate_projection_resources(INIT_N, DUAL_M)
    rr = allocate_recurrent_resources(pr[16])
    cube_bar = tla.cross_flag("cube_phase_barrier", mode=0)
    vector_bar = tla.cross_flag("vector_phase_barrier", mode=0)
    projections_done = tla.cross_flag("projections_done", mode=2)
    if tla.const_expr(DUAL_M):
        initial_dual_stage(
            x, wqkv, wa, wbeta, wz, qkv, a, beta, z, cw, cs, ci, ca, pr, N, L, NB
        )
    else:
        initial_stage(
            x,
            wqkv,
            wa,
            wbeta,
            wz,
            qkv,
            a,
            beta,
            z,
            cw,
            cs,
            ci,
            ca,
            pr,
            N,
            L,
            TM,
            NB,
            INIT_N,
        )
    with tla.cube():
        tla.cross_core_set_flag(cube_bar, tla.arch.MTE2)
        tla.cross_core_wait_flag(cube_bar, tla.arch.MTE2)
    projection_stage(a, wb, g, cw, cs, ci, ca, pr, TM, 128, False, False, L)
    with tla.cube():
        tla.cross_core_set_flag(cube_bar, tla.arch.MTE2)
        tla.cross_core_wait_flag(cube_bar, tla.arch.MTE2)
        tla.cross_core_set_flag(projections_done, tla.arch.MTE2)
    with tla.vector():
        tla.cross_core_wait_flag(projections_done, tla.arch.MTE2)
        tla.cross_core_set_flag(vector_bar, tla.arch.MTE2)
        tla.cross_core_wait_flag(vector_bar, tla.arch.MTE2)
    recurrent_stage(
        qkv,
        g,
        beta,
        z,
        state,
        indices,
        accepted,
        alog,
        bias,
        gamma,
        u,
        rr,
        N,
        SCALE,
        LOWER,
        EPS,
        PIPELINED,
    )
    if tla.const_expr(PIPELINED):
        projection_chunks(u, wo, out, pr, rr, cube_bar, OUTPUT_N, OUTPUT_RESIDENT)
    else:
        with tla.cube():
            tla.cross_core_set_flag(cube_bar, tla.arch.MTE2)
            tla.cross_core_wait_flag(cube_bar, tla.arch.MTE2)
        projection_stage(u, wo, out, cw, cs, ci, ca, pr, TM, 128, False, False, L)
