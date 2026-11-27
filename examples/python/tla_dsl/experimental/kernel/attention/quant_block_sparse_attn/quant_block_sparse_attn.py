# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Self-contained MXFP8 block sparse attention for Ascend 950.

All production kernels, input-norm checks, layout adaptation, and dispatch
live in this file. TLA implements each pipeline stage. The example requires
CATLASS DSL, PyTorch, and torch_npu.
"""

import math
import torch
import catlass.tla as tla

# Shared DMA, buffer initialization, and Fixpipe helpers.


@tla.jit
def mx_zero_ub(ptr, bytes: tla.Constexpr[int]):
    # Full 256-byte vectors plus an unaligned tail; UnalignStoreParams handles
    # the tail without an explicit mask.
    chunk = 256
    units = bytes // chunk
    tail = bytes % chunk
    target = tla.make_tensor(
        tla.recast_ptr(ptr, dtype=tla.Int8),
        tla.make_layout(tla.make_shape(bytes), tla.make_stride(1)),
    )
    with tla.vec.func():
        zero = tla.full(0, tla.Int8)
        for i in tla.range_constexpr(units):
            # Tile coordinates are in units of the tile extent.
            tla.tile_view(target, tla.make_shape(chunk), tla.make_coord(i)).store(zero)
        if tail != 0:
            tla.tile_view(target, tla.make_shape(tail), tla.make_coord(units)).store(
                zero, tla.params.UnalignStoreParams()
            )


@tla.jit
def mx_zero_scale_l1(dst, src, bytes: tla.Constexpr[int]):
    # One contiguous DMA; FP16 is only a 16-bit storage view of E8M0 bytes.
    layout = tla.make_layout_with_tag((bytes // 32, 16), tla.Float16, tla.arch.zN)
    target = tla.make_tensor(tla.recast_ptr(dst, dtype=tla.Float16), layout)
    source = tla.make_tensor(tla.recast_ptr(src, dtype=tla.Float16), layout)
    tla.copy(target, source)


@tla.jit
def _mx_ps_copy(dst, src, count):
    source_layout = tla.make_layout(
        tla.make_shape((8, 1), (16, count)),
        tla.make_stride((16, 128), (1, 128)),
        origin_shape=tla.make_shape(8, count * 16),
        layoutTag=tla.arch.zNUnAlign,
    )
    target_layout = tla.make_layout(
        tla.make_shape((16, 1), (16, count)),
        tla.make_stride((16, 256), (1, 256)),
        origin_shape=tla.make_shape(8, count * 16),
        layoutTag=tla.arch.zN,
    )
    tla.copy(tla.make_tensor(dst, target_layout), tla.make_tensor(src, source_layout))


@tla.jit
def mx_ps64_to_l1(dst, src, start, count, ring: tla.Constexpr[int], sub):
    d = tla.recast_ptr(dst, dtype=tla.Float16) + sub * 128
    s = tla.recast_ptr(src, dtype=tla.Float16)
    first = count if count < ring - start else ring - start
    _mx_ps_copy(d + start * 256, s, first)
    if first < count:
        _mx_ps_copy(d, s + first * 128, count - first)


@tla.jit
def _mx_score_copy(dst, src, scale, unit_flag: tla.Constexpr[int]):
    # L0C scores (128 keys x 128 queries) to half scores, each AIV owning one
    # 64-query half. Both halves use the same per-tensor K/Q dequantization,
    # so the exponent fold stays in the fixpipe and the softmax sees FP16.
    for part in tla.range_constexpr(2):
        source = tla.make_tensor(
            src + part * 8192,
            tla.make_layout_with_tag((128, 64), tla.Float32, tla.arch.L0Clayout),
        )
        target = tla.make_tensor(
            dst,
            tla.make_layout(tla.make_shape(128, 64), tla.make_stride(64, 1)),
        )
        tla.copy(
            target,
            source,
            tla.params.CopyL0C2DstParams(
                unit_flag=unit_flag,
                quant_mode=tla.params.QuantMode.PER_TENSOR,
                quant_scale=scale,
                l0c2ub_mode=(
                    tla.params.L0C2UBMode.NO_SPLIT_VEC_0
                    if part == 0
                    else tla.params.L0C2UBMode.NO_SPLIT_VEC_1
                ),
            ),
        )


@tla.jit
def mx_score_unit(dst, src, scale):
    # The first fixpipe starts off the MMAD's unit flag, so no CUBE/FIX event.
    _mx_score_copy(dst, src, scale, 0b11)


@tla.jit
def mx_score_fix(dst, src, scale):
    # Plain fixpipe: the caller orders it behind an explicit CUBE/FIX event.
    _mx_score_copy(dst, src, scale, 0b00)


@tla.jit
def _mx_load_row(scores, row):
    # One key row of 64 FP16 scores; DIST_UNPACK_B16 widens to a full register.
    return tla.tile_view(scores, tla.make_shape(64), tla.make_coord(row)).load(
        tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_UNPACK_B16)
    )


@tla.jit
def _mx_prob_value(scores, row, snap, maskh):
    # One key row of 64 queries: exp(score - snap).
    # Use separate operations for compatibility with CATLASS 2.2.1.
    # The fused form is tla.arch.sub_exp(_mx_load_row(scores, row), snap, mask=maskh).
    diff = tla.sub(_mx_load_row(scores, row), snap, mask=maskh)
    exp_f16 = tla.exp(diff, mask=maskh)
    return exp_f16.to(
        tla.Float32,
        tla.params.CastParams(),
        mask=maskh,
    )


@tla.jit
def _mx_prob_lane(scores, row, snap, mf, mh, e4):
    # One key row of 64 queries, cast into the byte slot of every 32-bit lane
    # selected by `e4`. The saturating FP8 cast caps the value at the E4M3
    # maximum.
    return (
        _mx_prob_value(scores, row, snap, mh)
        .to(tla.Float8E4M3FN, e4, mask=mf)
        .bitcast(tla.Int8)
    )


@tla.jit
def _mx_softmax_encode(score, prob, maximum, scratch, cols):
    # Group32 softmax over the FP16 scores: [128 keys, 64 queries] in, the
    # E4M3 probabilities as two [128 keys, 32 queries] planes out, and one
    # E8M0 exponent per 32-key group per query.
    scores = tla.make_tensor(
        score, tla.make_layout(tla.make_shape(128 * 64), tla.make_stride(1))
    )
    prob0 = tla.make_tensor(
        tla.recast_ptr(prob, dtype=tla.Int8),
        tla.make_layout(tla.make_shape(128, 32), tla.make_stride(32, 1)),
    )
    prob1 = tla.make_tensor(
        tla.recast_ptr(prob + 129 * 32, dtype=tla.Int8),
        tla.make_layout(tla.make_shape(128, 32), tla.make_stride(32, 1)),
    )
    events = _mx_vector_tile(scratch, 256)
    maxt = _mx_vector_tile(maximum, 64)
    with tla.vec.func():
        mf = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        mh = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
        floor_i32 = tla.params.CastParams(
            sat_mode=tla.params.SatMode.NOSAT,
            round_mode=tla.params.RoundMode.CAST_FLOOR,
        )
        plain = tla.params.CastParams()
        narrow = tla.params.CastParams(sat_mode=tla.params.SatMode.NOSAT)
        e4 = tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO,
            sat_mode=tla.params.SatMode.SAT,
            round_mode=tla.params.RoundMode.CAST_RINT,
        )
        e4b = tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ONE,
            sat_mode=tla.params.SatMode.SAT,
            round_mode=tla.params.RoundMode.CAST_RINT,
        )
        e4c = tla.params.CastParams(
            reg_slot=tla.params.RegSlot.TWO,
            sat_mode=tla.params.SatMode.SAT,
            round_mode=tla.params.RoundMode.CAST_RINT,
        )
        e4d = tla.params.CastParams(
            reg_slot=tla.params.RegSlot.THREE,
            sat_mode=tla.params.SatMode.SAT,
            round_mode=tla.params.RoundMode.CAST_RINT,
        )
        # Byte b of a packed four-row block belongs to key row b // 32 and
        # query b % 32; the second plane repeats the block 128 bytes later.
        m8 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int8)
        m128, _ = tla.update_mask(128, dtype=tla.Int8)
        u8 = tla.arange(0, dtype=tla.Int8)
        ilo = tla.bitwise_or(
            tla.shift_left(
                tla.bitwise_and(u8, tla.full(31, tla.Int8), mask=m8), 2, mask=m8
            ),
            tla.shift_right(u8, 5, mask=m8),
            mask=m8,
        )
        ihi = tla.add(ilo, tla.full(-128, tla.Int8), mask=m8)
        log2e = tla.full(1.4423828125, tla.Float16)
        bias = tla.full(7.8046875, tla.Float16)
        ln2 = tla.full(0.693359375, tla.Float16)
        current = maxt.load()
        if cols < 128:
            # Keys past the real K extent stay parked outside the score range:
            # they must neither win a group maximum nor reach the denominator.
            parked = tla.full(-65504.0, tla.Float16)
            mh64, _ = tla.update_mask(64, dtype=tla.Float16)
            for pad in tla.range(cols, 128):
                tla.tile_view(scores, tla.make_shape(64), tla.make_coord(pad)).store(
                    parked, mask=mh64
                )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        for grid in tla.range_constexpr(4):
            best = tla.full(-65504.0, tla.Float16)
            for mrow in tla.range(8):
                krow = grid * 32 + mrow * 4
                # Four rows per trip: the halving tree shortens the
                # dependency chain of the group maximum without spilling.
                m0 = tla.max(
                    _mx_load_row(scores, krow),
                    _mx_load_row(scores, krow + 1),
                    mask=mh,
                )
                m1 = tla.max(
                    _mx_load_row(scores, krow + 2),
                    _mx_load_row(scores, krow + 3),
                    mask=mh,
                )
                best = tla.max(best, tla.max(m0, m1, mask=mh), mask=mh)
            # The original scales and floors in FP16, so keep that rounding.
            scaled = tla.mul(best, log2e, mask=mh)
            # FP16 -> FP32 -> Int32 keeps the FP16 product's rounding.
            scaled32 = scaled.to(tla.Float32, plain, mask=mh)
            exponent = scaled32.to(tla.Int32, floor_i32, mask=mf)
            ef = exponent.to(tla.Float32, plain, mask=mf)
            tla.tile_view(events, tla.make_shape(64), tla.make_coord(grid)).store(ef)
            current = tla.max(current, ef, mask=mf)
            # Shift this group's probabilities up to 2**7.8 for E4M3.
            snap = tla.mul(
                tla.sub(ef.to(tla.Float16, narrow, mask=mf), bias, mask=mh),
                ln2,
                mask=mh,
            )
            # One packed four-row block per trip: each row lands in one byte of
            # every 32-bit lane, so the two gathers emit the two query planes.
            for quad in tla.range(8):
                prow = grid * 32 + quad * 4
                pack = tla.bitwise_or(
                    tla.bitwise_or(
                        _mx_prob_lane(scores, prow, snap, mf, mh, e4),
                        _mx_prob_lane(scores, prow + 1, snap, mf, mh, e4b),
                        mask=m8,
                    ),
                    tla.bitwise_or(
                        _mx_prob_lane(scores, prow + 2, snap, mf, mh, e4c),
                        _mx_prob_lane(scores, prow + 3, snap, mf, mh, e4d),
                        mask=m8,
                    ),
                    mask=m8,
                )
                tla.tile_view(
                    prob0, tla.make_shape(4, 32), tla.make_coord(grid * 8 + quad, 0)
                ).store(tla.gather(pack, ilo), mask=m128)
                tla.tile_view(
                    prob1, tla.make_shape(4, 32), tla.make_coord(grid * 8 + quad, 0)
                ).store(tla.gather(pack, ihi), mask=m128)
        maxt.store(current)


@tla.jit
def mx_softmax(
    score, prob, ps, maximum, denominator, rescale, scratch, scale, cols, saved
):
    # Single-tile path: keep the previous maximum for the PV rescale, emit the
    # scale bytes around the refreshed probabilities.
    _mx_group_scales_basic(ps, maximum, saved, rescale, scratch, -1)
    _mx_softmax_encode(score, prob, maximum, scratch, cols)
    _mx_group_scales_basic(ps, maximum, saved, rescale, scratch, 1)


@tla.jit
def mx_pad(v, s):
    # The probability pad channel stays the E4M3 encoding of 1.0 (0x38) and
    # the MX scale pad stays the E8M0 encoding of 1.0 (0x7f).
    with tla.vec.func():
        one_e4m3 = tla.full(0x38, tla.Int8)
        values = tla.make_tensor(
            tla.recast_ptr(v, dtype=tla.Int8),
            tla.make_layout(tla.make_shape(2048), tla.make_stride(1)),
        )
        for i in tla.range_constexpr(8):
            tla.tile_view(values, tla.make_shape(256), tla.make_coord(i)).store(
                one_e4m3
            )
        scales = tla.make_tensor(
            tla.recast_ptr(s, dtype=tla.Int8),
            tla.make_layout(tla.make_shape(64), tla.make_stride(1)),
        )
        tla.tile_view(scales, tla.make_shape(64), tla.make_coord(0)).store(
            tla.full(127, tla.Int8), tla.params.UnalignStoreParams()
        )


# Distinct bindings prevent the basic/fullstore/bounded variants from aliasing.


@tla.jit
def _mx_vector_tile(ptr, lanes: tla.Constexpr[int]):
    return tla.make_tensor(
        ptr,
        tla.make_layout(
            tla.make_shape(lanes), tla.make_stride(1), layoutTag=tla.arch.VectorLayout
        ),
    )


@tla.jit
def _mx_scale_bits(groups, index, current, mask):
    value = (
        tla.tile_view(groups, tla.make_shape(64), tla.make_coord(index)).load()
        - current
    )
    value = tla.max(value + 119.0, 0.0)
    # Both exponents are integral, so FLOOR and RINT coincide here.
    return value.to(
        tla.Int32,
        tla.params.CastParams(
            sat_mode=tla.params.SatMode.SAT, round_mode=tla.params.RoundMode.CAST_FLOOR
        ),
        mask=mask,
    )


@tla.jit
def _mx_group_scales_dsl(
    ps,
    maximum,
    saved,
    alpha,
    groups,
    count,
    fullstore: tla.Constexpr[bool],
    bounded: tla.Constexpr[bool],
):
    mt = _mx_vector_tile(maximum, 64)
    st = _mx_vector_tile(saved, 64)
    at = _mx_vector_tile(alpha, 64)
    gt = _mx_vector_tile(groups, 4096)
    pt = _mx_vector_tile(tla.recast_ptr(ps, dtype=tla.Int8), 4096)
    ph0 = _mx_vector_tile(tla.recast_ptr(ps, dtype=tla.Int16), 2048)
    ph1 = _mx_vector_tile(tla.recast_ptr(ps + 32, dtype=tla.Int16), 2048)
    if count == -2:
        with tla.vec.func():
            initial = 48.0 if bounded else -65504.0
            mt.store(tla.full(initial, tla.Float32))
    elif count < 0:
        with tla.vec.func():
            st.store(mt.load())
    else:
        with tla.vec.func():
            mf = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            old = st.load()
            current = mt.load()
            delta = tla.max(old - current + 127.0, 0.0)
            bits = delta.to(
                tla.Int32,
                tla.params.CastParams(
                    sat_mode=tla.params.SatMode.SAT,
                    round_mode=tla.params.RoundMode.CAST_FLOOR,
                ),
                mask=mf,
            )
            at.store(tla.shift_left(bits, 23, mask=mf).bitcast(tla.Float32), mask=mf)
            if fullstore:
                mb = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int8)
                u = tla.arange(0, dtype=tla.Int8)
                # Mask before signed shifts; preserve the original byte permutation.
                ix = tla.bitwise_and(u, tla.full(1, tla.Int8), mask=mb)
                a = tla.shift_left(
                    tla.bitwise_and(u, tla.full(30, tla.Int8), mask=mb), 1, mask=mb
                )
                ix = tla.bitwise_or(ix, a, mask=mb)
                a = tla.shift_right(
                    tla.bitwise_and(u, tla.full(32, tla.Int8), mask=mb), 4, mask=mb
                )
                ix = tla.bitwise_or(ix, a, mask=mb)
                ix = tla.bitwise_or(
                    ix, tla.bitwise_and(u, tla.full(-64, tla.Int8), mask=mb), mask=mb
                )
                for t in tla.range(count):
                    b0 = (
                        _mx_scale_bits(gt, t * 4, current, mf)
                        .bitcast(tla.UInt32)
                        .to(
                            tla.UInt8,
                            tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.ZERO,
                                sat_mode=tla.params.SatMode.SAT,
                            ),
                            mask=mf,
                        )
                        .bitcast(tla.Int8)
                    )
                    b1 = (
                        _mx_scale_bits(gt, t * 4 + 1, current, mf)
                        .bitcast(tla.UInt32)
                        .to(
                            tla.UInt8,
                            tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.ONE,
                                sat_mode=tla.params.SatMode.SAT,
                            ),
                            mask=mf,
                        )
                        .bitcast(tla.Int8)
                    )
                    b2 = (
                        _mx_scale_bits(gt, t * 4 + 2, current, mf)
                        .bitcast(tla.UInt32)
                        .to(
                            tla.UInt8,
                            tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.TWO,
                                sat_mode=tla.params.SatMode.SAT,
                            ),
                            mask=mf,
                        )
                        .bitcast(tla.Int8)
                    )
                    b3 = (
                        _mx_scale_bits(gt, t * 4 + 3, current, mf)
                        .bitcast(tla.UInt32)
                        .to(
                            tla.UInt8,
                            tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.THREE,
                                sat_mode=tla.params.SatMode.SAT,
                            ),
                            mask=mf,
                        )
                        .bitcast(tla.Int8)
                    )
                    packed = tla.bitwise_or(
                        tla.bitwise_or(tla.bitwise_or(b0, b1, mask=mb), b2, mask=mb),
                        b3,
                        mask=mb,
                    )
                    tla.tile_view(pt, tla.make_shape(256), tla.make_coord(t)).store(
                        tla.gather(packed, ix), mask=mb
                    )
            else:
                mh = tla.create_mask(pattern=tla.mask.VL64, dtype=tla.Int16)
                for t in tla.range(count):
                    for pair in tla.range_constexpr(2):
                        lo = _mx_scale_bits(gt, t * 4 + pair * 2, current, mf)
                        hi = _mx_scale_bits(gt, t * 4 + pair * 2 + 1, current, mf)
                        joined = tla.bitwise_or(
                            lo, tla.shift_left(hi, 8, mask=mf), mask=mf
                        )
                        words = joined.to(
                            tla.Int16,
                            tla.params.CastParams(sat_mode=tla.params.SatMode.SAT),
                            mask=mf,
                        )
                        packed, unused = tla.deinterleave(words, words)
                        if pair == 0:
                            tla.tile_view(
                                ph0, tla.make_shape(128), tla.make_coord(t)
                            ).store(
                                packed,
                                tla.params.BlockStoreParams(block_stride=2),
                                mask=mh,
                            )
                        else:
                            tla.tile_view(
                                ph1, tla.make_shape(128), tla.make_coord(t)
                            ).store(
                                packed,
                                tla.params.BlockStoreParams(block_stride=2),
                                mask=mh,
                            )


@tla.jit
def _mx_group_scales_basic(ps, maximum, saved, alpha, groups, count):
    _mx_group_scales_dsl(ps, maximum, saved, alpha, groups, count, False, False)


@tla.jit
def _mx_group_scales_fullstore(ps, maximum, saved, alpha, groups, count):
    _mx_group_scales_dsl(ps, maximum, saved, alpha, groups, count, True, False)


@tla.jit
def _mx_group_scales_bounded(ps, maximum, saved, alpha, groups, count):
    _mx_group_scales_dsl(ps, maximum, saved, alpha, groups, count, True, True)


@tla.jit
def _mx_finish_pair(at, db, i, inv, mask, od, slot0, slot1):
    # Normalize a feature pair and pack both columns' 16-bit values into one
    # 32-bit lane, ready for the interleave transpose network.
    lo = (
        tla.tile_view(at, tla.make_shape(64), tla.make_coord(db * 16 + 2 * i)).load()
        * inv
    ).to(od, slot0, mask=mask)
    hi = (
        tla.tile_view(
            at, tla.make_shape(64), tla.make_coord(db * 16 + 2 * i + 1)
        ).load()
        * inv
    ).to(od, slot1, mask=mask)
    return tla.bitwise_or(lo.bitcast(tla.Int32), hi.bitcast(tla.Int32), mask=mask)


@tla.jit
def _mx_finish_store(o16, reg, k, db, params, mask):
    # One transposed register holds eight query rows of sixteen features; the
    # block store walks the padded output rows on their 32-byte pitch.
    tla.tile_view(o16, tla.make_shape(8, 16), tla.make_coord(k, db)).store(
        reg.bitcast(tla.Int16), params, mask=mask
    )


@tla.jit
def mx_epilogue(
    acc,
    pv,
    maximum,
    denominator,
    rescale,
    output,
    lse,
    dim: tla.Constexpr[int],
    rows,
    op: tla.Constexpr[int],
    dtype: tla.Constexpr[int],
):
    if op == 0:
        at = _mx_vector_tile(acc, (dim + 1) * 64)
        mt = _mx_vector_tile(maximum, 64)
        dt = _mx_vector_tile(denominator, 64)
        with tla.vec.func():
            zero = tla.full(0.0, tla.Float32)
            initial = tla.full(-65504.0, tla.Float32)
            mt.store(initial)
            dt.store(zero)
            for d in tla.range(dim + 1):
                tla.tile_view(at, tla.make_shape(64), tla.make_coord(d)).store(zero)
    elif op == 1:
        at = _mx_vector_tile(acc, (dim + 1) * 64)
        pt = _mx_vector_tile(pv, (dim + 1) * 64)
        rt = _mx_vector_tile(rescale, 64)
        with tla.vec.func():
            alpha = rt.load()
            for d in tla.range(dim + 1):
                tile = tla.tile_view(at, tla.make_shape(64), tla.make_coord(d))
                old = tile.load()
                current = tla.tile_view(
                    pt, tla.make_shape(64), tla.make_coord(d)
                ).load()
                scaled = old * alpha
                tile.store(current + scaled)
    else:
        # Finish: normalize the transposed PV rows, emit LSE, and write the
        # query-major output. The accumulator is feature-major (one 64-query
        # register per feature), so 16 features are transposed in registers
        # with a three-round interleave network before the block stores.
        at = _mx_vector_tile(acc, (dim + 1) * 64)
        mt = _mx_vector_tile(maximum, 64)
        lt = _mx_vector_tile(lse, 64)
        ostride = dim + (8 if dtype == 0 else 16)
        values = tla.make_tensor(
            tla.recast_ptr(output, dtype=tla.Float32),
            tla.make_layout(tla.make_shape(64 * ostride), tla.make_stride(1)),
        )
        o16 = tla.make_tensor(
            tla.recast_ptr(output, dtype=tla.Int16),
            tla.make_layout(tla.make_shape(64, dim + 16), tla.make_stride(ostride, 1)),
        )
        with tla.vec.func():
            mf = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            plain = tla.params.CastParams(sat_mode=tla.params.SatMode.NOSAT)
            den = tla.tile_view(
                at, tla.make_shape(64), tla.make_coord(tla.as_numeric(dim))
            ).load()
            total = tla.log(den) + mt.load() * 0.6931471805599453 + 0.13538030870311432
            lt.store(total, mask=mf)
            inv = tla.div(
                tla.full(1.0, tla.Float32),
                tla.max(den, 1e-30, mask=mf),
                mask=mf,
            )
            if dtype == 0:
                # Four-byte output: scatter each feature column on its stride.
                for d in tla.range(dim):
                    value = (
                        tla.tile_view(at, tla.make_shape(64), tla.make_coord(d)).load()
                        * inv
                    )
                    # Offsets are non-negative, so the bit pattern is the value.
                    index = (tla.arange(0, dtype=tla.Int32) * ostride + d).bitcast(
                        tla.UInt32
                    )
                    tla.scatter(values, value, index, mask=mf)
            else:
                od = tla.Float16 if dtype == 1 else tla.BFloat16
                slot0 = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    sat_mode=tla.params.SatMode.NOSAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                )
                slot1 = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ONE,
                    sat_mode=tla.params.SatMode.NOSAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                )
                mb = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
                blocks = tla.params.BlockStoreParams(block_stride=ostride // 16)
                for db in tla.range(dim // 16):
                    r0 = _mx_finish_pair(at, db, 0, inv, mf, od, slot0, slot1)
                    r1 = _mx_finish_pair(at, db, 1, inv, mf, od, slot0, slot1)
                    r2 = _mx_finish_pair(at, db, 2, inv, mf, od, slot0, slot1)
                    r3 = _mx_finish_pair(at, db, 3, inv, mf, od, slot0, slot1)
                    r4 = _mx_finish_pair(at, db, 4, inv, mf, od, slot0, slot1)
                    r5 = _mx_finish_pair(at, db, 5, inv, mf, od, slot0, slot1)
                    r6 = _mx_finish_pair(at, db, 6, inv, mf, od, slot0, slot1)
                    r7 = _mx_finish_pair(at, db, 7, inv, mf, od, slot0, slot1)
                    r0, r4 = tla.interleave(r0, r4)
                    r1, r5 = tla.interleave(r1, r5)
                    r2, r6 = tla.interleave(r2, r6)
                    r3, r7 = tla.interleave(r3, r7)
                    r0, r2 = tla.interleave(r0, r2)
                    r1, r3 = tla.interleave(r1, r3)
                    r4, r6 = tla.interleave(r4, r6)
                    r5, r7 = tla.interleave(r5, r7)
                    r0, r1 = tla.interleave(r0, r1)
                    r2, r3 = tla.interleave(r2, r3)
                    r4, r5 = tla.interleave(r4, r5)
                    r6, r7 = tla.interleave(r6, r7)
                    _mx_finish_store(o16, r0, 0, db, blocks, mb)
                    _mx_finish_store(o16, r1, 1, db, blocks, mb)
                    _mx_finish_store(o16, r2, 2, db, blocks, mb)
                    _mx_finish_store(o16, r3, 3, db, blocks, mb)
                    _mx_finish_store(o16, r4, 4, db, blocks, mb)
                    _mx_finish_store(o16, r5, 5, db, blocks, mb)
                    _mx_finish_store(o16, r6, 6, db, blocks, mb)
                    _mx_finish_store(o16, r7, 7, db, blocks, mb)


@tla.jit
def _mx_group_softmax_store_basic(
    score, prob, ps, maximum, denominator, rescale, scratch, scale, cols
):
    # Group pipelines emit the scale bytes from the group helper, so the
    # softmax only refreshes the probabilities and the running maximum.
    _mx_softmax_encode(score, prob, maximum, scratch, cols)


@tla.jit
def _mx_group_softmax_store_fast(
    score, prob, ps, maximum, denominator, rescale, scratch, scale, cols
):
    _mx_softmax_encode(score, prob, maximum, scratch, cols)


# Attention kernel: solution.
@tla.kernel
def quant_block_sparse_attention_kernel(
    q: tla.Tensor,
    k: tla.Tensor,
    v: tla.Tensor,
    out: tla.Tensor,
    qs: tla.Tensor,
    ks: tla.Tensor,
    vs: tla.Tensor,
    indices: tla.Tensor,
    lse: tla.Tensor,
    batch: tla.Int32,
    hq: tla.Int32,
    hk: tla.Int32,
    nq: tla.Int32,
    nk: tla.Int32,
    mh: tla.Int32,
    selected: tla.Int32,
    scale: tla.Constexpr[float],
    dim: tla.Constexpr[int],
    xb: tla.Constexpr[int],
    yb: tla.Constexpr[int],
    want_lse: tla.Constexpr[bool],
):
    # Transposed QK/PV: each AIV owns 64 query lanes. PV adds a
    # constant-one channel for the quantized probability denominator.
    bm = 128
    bn = 128
    dv = dim + 32
    pv_m = dim + 16
    half = 64
    qm = 64 if xb % 128 != 0 else 128
    fp8 = tla.Float8E4M3FN
    e8m0 = tla.Float8E8M0
    od = out.ptr.dtype
    output_stride = dim + (8 if od == tla.Float32 else 16)
    od_code = 0 if od == tla.Float32 else (1 if od == tla.Float16 else 2)
    q1p = tla.allocate(bm * dim, fp8, tla.AddressSpace.l1, 512)
    k1p = tla.allocate(bn * dim, fp8, tla.AddressSpace.l1, 512)
    v1p = tla.allocate(bn * dv, fp8, tla.AddressSpace.l1, 512)
    # Two P slots let softmax(t) overlap PV(t-1).
    p1p = tla.allocate(2 * bm * bn, fp8, tla.AddressSpace.l1, 512)
    qs1p = tla.allocate(bm * (dim // 32), e8m0, tla.AddressSpace.l1, 512)
    ks1p = tla.allocate(bn * (dim // 32), e8m0, tla.AddressSpace.l1, 512)
    vs1p = tla.allocate(dv * (bn // 32), e8m0, tla.AddressSpace.l1, 512)
    ps1p = tla.allocate(2 * bm * (bn // 32), e8m0, tla.AddressSpace.l1, 512)
    q0p = tla.allocate(bm * dim, fp8, tla.AddressSpace.l0b, 1024)
    k0p = tla.allocate(bn * dim, fp8, tla.AddressSpace.l0a, 1024)
    p0p = tla.allocate(bm * bn, fp8, tla.AddressSpace.l0b, 1024)
    v0p = tla.allocate(bn * dv, fp8, tla.AddressSpace.l0a, 1024)
    s0p = tla.allocate(bm * bn, tla.Float32, tla.AddressSpace.l0c, 1024)
    o0p = tla.allocate(bm * dv, tla.Float32, tla.AddressSpace.l0c, 1024)
    # Separate score loads and probability stores into different 32 KiB UB
    # regions. The second half is spacing, not an additional score tile.
    sp = tla.allocate(2 * half * bn, tla.Float16, tla.AddressSpace.ub, 512)
    pp = tla.allocate(half * (bn + 1), tla.UInt8, tla.AddressSpace.ub, 512)
    psp = tla.allocate(half * (bn // 32), tla.UInt8, tla.AddressSpace.ub, 256)
    mp = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)
    lp = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)
    rp = tla.allocate(2 * half, tla.Float32, tla.AddressSpace.ub, 256)
    svp = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)
    gp = tla.allocate(half * (bn // 32), tla.Float32, tla.AddressSpace.ub, 256)
    op = tla.allocate(half * pv_m, tla.Float32, tla.AddressSpace.ub, 512)
    ap = tla.allocate(half * pv_m, tla.Float32, tla.AddressSpace.ub, 512)
    yp = tla.allocate(half * output_stride, od, tla.AddressSpace.ub, 512)
    ep = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)

    q1 = tla.make_tensor(q1p, tla.make_layout_with_tag((dim, bm), fp8, tla.arch.nZ))
    k1 = tla.make_tensor(k1p, tla.make_layout_with_tag((bn, dim), fp8, tla.arch.zN))
    v1 = tla.make_tensor(v1p, tla.make_layout_with_tag((dv, bn), fp8, tla.arch.nZ))
    p1 = tla.make_tensor(p1p, tla.make_layout_with_tag((bn, bm), fp8, tla.arch.zN))
    qs1 = tla.make_tensor(
        qs1p, tla.make_layout_with_tag((dim // 32, bm), e8m0, tla.arch.nNMxScale)
    )
    ks1 = tla.make_tensor(
        ks1p, tla.make_layout_with_tag((bn, dim // 32), e8m0, tla.arch.zZMxScale)
    )
    vs1 = tla.make_tensor(
        vs1p, tla.make_layout_with_tag((dv, bn // 32), e8m0, tla.arch.zZMxScale)
    )
    ps1 = tla.make_tensor(
        ps1p, tla.make_layout_with_tag((bn // 32, bm), e8m0, tla.arch.nNMxScale)
    )
    q0 = tla.make_tensor(q0p, tla.make_layout_with_tag((dim, bm), fp8, tla.arch.nZ))
    k0 = tla.make_tensor(k0p, tla.make_layout_with_tag((bn, dim), fp8, tla.arch.zN))
    p0 = tla.make_tensor(p0p, tla.make_layout_with_tag((bn, bm), fp8, tla.arch.nZ))
    v0 = tla.make_tensor(v0p, tla.make_layout_with_tag((dv, bn), fp8, tla.arch.zN))
    s0 = tla.make_tensor(
        s0p, tla.make_layout_with_tag((bn, bm), tla.Float32, tla.arch.L0Clayout)
    )
    o0 = tla.make_tensor(
        o0p, tla.make_layout_with_tag((dv, bm), tla.Float32, tla.arch.L0Clayout)
    )
    sfix = tla.make_tensor(
        sp, tla.make_layout(tla.make_shape(bn, bm), tla.make_stride(half, 1))
    )
    ofix = tla.make_tensor(
        op, tla.make_layout(tla.make_shape(pv_m, bm), tla.make_stride(half, 1))
    )
    p16ref = tla.make_tensor(
        tla.recast_ptr(pp, dtype=tla.Float16),
        tla.make_layout(
            tla.make_shape(bn + 1, half // 2), tla.make_stride(half // 2, 1)
        ),
    )
    pu = tla.make_tensor_like(p16ref.ptr, p16ref, tla.arch.zNUnAlign)
    p1storage = tla.make_tensor(
        tla.recast_ptr(p1p, dtype=tla.Float16),
        tla.make_layout_with_tag((bn, bm // 2), tla.Float16, tla.arch.zN),
    )
    psuref = tla.make_tensor(
        tla.recast_ptr(psp, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(8, 16), tla.make_stride(16, 1)),
    )
    psu = tla.make_tensor_like(psuref.ptr, psuref, tla.arch.zNUnAlign)
    psflat = tla.make_tensor(
        tla.recast_ptr(ps1p, dtype=tla.Float16),
        tla.make_layout_with_tag((16, 16), tla.Float16, tla.arch.zN),
    )
    padp = tla.allocate(2048, tla.UInt8, tla.AddressSpace.ub, 512)
    pads = tla.allocate(64, tla.UInt8, tla.AddressSpace.ub, 256)
    padref = tla.make_tensor(
        tla.recast_ptr(padp, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(64, 16), tla.make_stride(16, 1)),
    )
    padu = tla.make_tensor_like(padref.ptr, padref, tla.arch.zNUnAlign)
    pad1 = tla.make_tensor(
        tla.recast_ptr(v1p + dim * bn, dtype=tla.Float16),
        tla.make_layout_with_tag((128, 16), tla.Float16, tla.arch.zN),
    )
    padsref = tla.make_tensor(
        tla.recast_ptr(pads, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(2, 16), tla.make_stride(16, 1)),
    )
    padsu = tla.make_tensor_like(padsref.ptr, padsref, tla.arch.zNUnAlign)
    pads1 = tla.make_tensor(
        tla.recast_ptr(vs1p + dim * 4, dtype=tla.Float16),
        tla.make_layout_with_tag((4, 16), tla.Float16, tla.arch.zN),
    )
    yu = tla.make_tensor(
        yp,
        tla.make_layout(tla.make_shape(half, dim), tla.make_stride(output_stride, 1)),
    )
    eu = tla.make_tensor(ep, tla.make_layout(tla.make_shape(half), tla.make_stride(1)))

    # In-core buffer ownership is expressed with mutexes. Cube-side and
    # vector-side mutex ids live in separate hardware id spaces.
    m_q1 = tla.mutex("q1", id=0)
    m_q0 = tla.mutex("q0", id=1)
    m_k1 = tla.mutex("k1", id=2)
    m_k0 = tla.mutex("k0", id=3)
    m_s0 = tla.mutex("s0", id=4)
    m_v1 = tla.mutex("v1", id=5)
    m_pv0 = tla.mutex("pv0", id=6)
    m_o0 = tla.mutex("o0", id=7)
    m_pad = tla.mutex("pad", id=8)
    m_p1 = tla.mutex("p1", id=9)
    m_out = tla.mutex("out", id=10)
    sr = tla.cross_flag("score_ready")
    pr = tla.cross_flag("prob_ready")
    pr_alt = tla.cross_flag("prob_ready_alt")
    ore = tla.cross_flag("output_ready")
    split_o = tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N)
    padready = tla.cross_flag("pad_ready")
    with tla.vector():
        # Partial Q/K/V copies leave some MX scale bytes untouched. E8M0
        # 0xff is NaN, so zero data alone does not make padding harmless.
        m_p1.lock(pipe=tla.pipes.VECTOR)
        mx_zero_ub(pp, half * (bn + 1))
        m_p1.unlock(pipe=tla.pipes.VECTOR)
        m_pad.lock(pipe=tla.pipes.VECTOR)
        mx_pad(padp, pads)
        m_pad.unlock(pipe=tla.pipes.VECTOR)
        m_p1.lock(pipe=tla.pipes.MTE3)
        if tla.arch.sub_block_idx() == 0:
            mx_zero_scale_l1(qs1p, pp, bm * (dim // 32))
            mx_zero_scale_l1(ks1p, pp, bn * (dim // 32))
            mx_zero_scale_l1(vs1p, pp, dv * (bn // 32))
        m_p1.unlock(pipe=tla.pipes.MTE3)
        m_pad.lock(pipe=tla.pipes.MTE3)
        tla.copy(
            tla.tile_view(
                pad1,
                tla.make_shape(64, 16),
                tla.make_coord(tla.arch.sub_block_idx(), 0),
            ),
            padu,
        )
        tla.copy(
            tla.tile_view(
                pads1,
                tla.make_shape(2, 16),
                tla.make_coord(tla.arch.sub_block_idx(), 0),
            ),
            padsu,
        )
        m_pad.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(padready, tla.pipes.MTE3)
    with tla.cube():
        tla.cross_core_wait_flag(padready, tla.pipes.MTE2)
    with tla.vector():
        tla.cross_core_set_flag(sr, tla.pipes.VECTOR)
        tla.cross_core_set_flag(ore, tla.pipes.VECTOR)
    with tla.cube():
        tla.cross_core_set_flag(pr, tla.pipes.MTE1)
        tla.cross_core_set_flag(pr_alt, tla.pipes.MTE1)
    blocks = (nq + xb - 1) // xb
    subtiles = xb // qm
    total = batch * hq * blocks * subtiles
    for task in tla.range(tla.arch.block_idx(), total, tla.arch.block_num()):
        h = task % hq
        qb = (task // hq // subtiles) % blocks
        qi = (task // hq) % subtiles
        b = task // (hq * blocks * subtiles)
        kh = h // (hq // hk)
        qstart = qb * xb + qi * qm
        if qstart < nq:
            rows = qm if nq - qstart >= qm else nq - qstart
            sub = tla.arch.sub_block_idx()
            valid = rows - sub * half
            valid = half if valid > half else valid
            valid = 0 if valid < 0 else valid
            qbase = ((b * hq + h) * nq + qstart) * dim
            kbase = (b * hk + kh) * nk * dim
            vb = (b * hk + kh) * ((nk + 63) // 64) * dim * 2
            mask_head = 0 if mh == 1 else h
            maskbase = ((b * mh + mask_head) * blocks + qb) * selected
            # Common sorted indices use O(K) traversal; unordered input uses
            # successive minima, retaining reference sorting/dedup semantics.
            ordered = tla.as_numeric(1)
            previous = tla.as_numeric(-1)
            for scan in tla.range(selected):
                value = indices[maskbase + scan].to(tla.Int32)
                if value >= 0:
                    if value < previous:
                        ordered = tla.as_numeric(0)
                    previous = value
            with tla.cube():
                qgm = tla.make_tensor(
                    q.ptr + qbase,
                    tla.make_layout(
                        tla.make_shape(dim, rows),
                        tla.make_stride(1, dim),
                        layoutTag=tla.arch.ColumnMajor,
                    ),
                )
                m_q1.lock(pipe=tla.pipes.MTE2)
                tla.copy(
                    tla.tile_view(q1, tla.make_shape(dim, rows), tla.make_coord(0, 0)),
                    qgm,
                )
                qsgm = tla.get_tile(
                    qs, tla.make_coord(0, qbase // dim), tla.make_shape(dim // 32, rows)
                )
                tla.copy(
                    tla.tile_view(
                        qs1, tla.make_shape(dim // 32, rows), tla.make_coord(0, 0)
                    ),
                    qsgm,
                )
                m_q1.unlock(pipe=tla.pipes.MTE2)
                m_q1.lock(pipe=tla.pipes.MTE1)
                m_q0.lock(pipe=tla.pipes.MTE1)
                tla.copy(q0, q1, scale=qs1)
                m_q0.unlock(pipe=tla.pipes.MTE1)
                m_q1.unlock(pipe=tla.pipes.MTE1)
            with tla.vector():
                mx_epilogue(
                    ap,
                    op,
                    mp,
                    lp,
                    rp,
                    tla.recast_ptr(yp, dtype=tla.UInt8),
                    ep,
                    dim,
                    valid,
                    0,
                    od_code,
                )
            count = tla.as_numeric(0)
            pending_start = tla.as_numeric(0)
            pending_cols = tla.as_numeric(0)
            last = tla.as_numeric(-1)
            for si in tla.range(selected):
                block = indices[maskbase + si].to(tla.Int32)
                if ordered == 0:
                    block = (nk + yb - 1) // yb
                    for sj in tla.range(selected):
                        candidate = indices[maskbase + sj].to(tla.Int32)
                        if candidate > last:
                            if candidate < block:
                                block = candidate
                if block > last:
                    last = block
                    for ki in tla.range((yb + bn - 1) // bn):
                        start = block * yb + ki * bn
                        if start < nk:
                            cols = bn if yb - ki * bn >= bn else yb - ki * bn
                            cols = cols if nk - start >= cols else nk - start
                            with tla.cube():
                                m_k1.lock(pipe=tla.pipes.MTE2)
                                if cols < bn:
                                    k1.fill(0)
                                kgm = tla.make_tensor(
                                    k.ptr + kbase + start * dim,
                                    tla.make_layout(
                                        tla.make_shape(cols, dim),
                                        tla.make_stride(dim, 1),
                                    ),
                                )
                                tla.copy(
                                    tla.tile_view(
                                        k1,
                                        tla.make_shape(cols, dim),
                                        tla.make_coord(0, 0),
                                    ),
                                    kgm,
                                )
                                ksgm = tla.get_tile(
                                    ks,
                                    tla.make_coord(kbase // dim + start, 0),
                                    tla.make_shape(cols, dim // 32),
                                )
                                tla.copy(
                                    tla.tile_view(
                                        ks1,
                                        tla.make_shape(cols, dim // 32),
                                        tla.make_coord(0, 0),
                                    ),
                                    ksgm,
                                )
                                m_k1.unlock(pipe=tla.pipes.MTE2)
                                m_k1.lock(pipe=tla.pipes.MTE1)
                                m_k0.lock(pipe=tla.pipes.MTE1)
                                tla.copy(k0, k1, scale=ks1)
                                m_k0.unlock(pipe=tla.pipes.MTE1)
                                m_k1.unlock(pipe=tla.pipes.MTE1)
                                m_q0.lock(pipe=tla.pipes.CUBE)
                                m_k0.lock(pipe=tla.pipes.CUBE)
                                m_s0.lock(pipe=tla.pipes.CUBE)
                                tla.mmad_mx(s0, k0, q0, init_c=True)
                                m_s0.unlock(pipe=tla.pipes.CUBE)
                                m_k0.unlock(pipe=tla.pipes.CUBE)
                                m_q0.unlock(pipe=tla.pipes.CUBE)
                                tla.cross_core_wait_flag(sr, tla.pipes.FIX)
                                m_s0.lock(pipe=tla.pipes.FIX)
                                mx_score_fix(sp, s0p, scale)
                                m_s0.unlock(pipe=tla.pipes.FIX)
                                tla.cross_core_set_flag(sr, tla.pipes.FIX)
                            with tla.vector():
                                tla.cross_core_wait_flag(sr, tla.pipes.VECTOR)
                                m_p1.lock(pipe=tla.pipes.VECTOR)
                                mx_softmax(
                                    sp,
                                    pp,
                                    psp,
                                    mp,
                                    lp,
                                    rp + (count % 2) * half,
                                    gp,
                                    scale,
                                    cols,
                                    svp,
                                )
                                m_p1.unlock(pipe=tla.pipes.VECTOR)
                                tla.cross_core_set_flag(sr, tla.pipes.VECTOR)
                                if count % 2 == 0:
                                    tla.cross_core_wait_flag(pr, tla.pipes.MTE3)
                                else:
                                    tla.cross_core_wait_flag(pr_alt, tla.pipes.MTE3)
                                write_p = tla.make_tensor_like(
                                    tla.recast_ptr(
                                        p1p + (count % 2) * bm * bn, dtype=tla.Float16
                                    ),
                                    p1storage,
                                    tla.arch.zN,
                                )
                                write_ps = tla.make_tensor_like(
                                    tla.recast_ptr(
                                        ps1p + (count % 2) * bm * 4, dtype=tla.Float16
                                    ),
                                    psflat,
                                    tla.arch.zN,
                                )
                                m_p1.lock(pipe=tla.pipes.MTE3)
                                tla.copy(
                                    tla.tile_view(
                                        write_p,
                                        tla.make_shape(bn, half // 2),
                                        tla.make_coord(0, sub),
                                    ),
                                    tla.tile_view(
                                        pu,
                                        tla.make_shape(bn, half // 2),
                                        tla.make_coord(0, 0),
                                    ),
                                )
                                tla.copy(
                                    tla.tile_view(
                                        write_ps,
                                        tla.make_shape(8, 16),
                                        tla.make_coord(sub, 0),
                                    ),
                                    psu,
                                )
                                m_p1.unlock(pipe=tla.pipes.MTE3)
                                if count % 2 == 0:
                                    tla.cross_core_set_flag(pr, tla.pipes.MTE3)
                                else:
                                    tla.cross_core_set_flag(pr_alt, tla.pipes.MTE3)
                            if count > 0:
                                with tla.cube():
                                    m_v1.lock(pipe=tla.pipes.MTE2)
                                    if pending_cols < bn:
                                        tla.get_tile(
                                            v1,
                                            tla.make_coord(0, 0),
                                            tla.make_shape(dim, bn),
                                        ).fill(0)
                                    vgm_pv = tla.make_tensor(
                                        v.ptr + kbase + pending_start * dim,
                                        tla.make_layout(
                                            tla.make_shape(dim, pending_cols),
                                            tla.make_stride(1, dim),
                                            layoutTag=tla.arch.ColumnMajor,
                                        ),
                                    )
                                    tla.copy(
                                        tla.tile_view(
                                            v1,
                                            tla.make_shape(dim, pending_cols),
                                            tla.make_coord(0, 0),
                                        ),
                                        vgm_pv,
                                    )
                                    vgroups_pv = ((pending_cols + 63) // 64) * 2
                                    vsgm_pv = tla.get_tile(
                                        vs,
                                        tla.make_coord(
                                            0, vb // dim + pending_start // 32
                                        ),
                                        tla.make_shape(dim, vgroups_pv),
                                    )
                                    tla.copy(
                                        tla.tile_view(
                                            vs1,
                                            tla.make_shape(dim, vgroups_pv),
                                            tla.make_coord(0, 0),
                                        ),
                                        vsgm_pv,
                                    )
                                    m_v1.unlock(pipe=tla.pipes.MTE2)
                                    m_v1.lock(pipe=tla.pipes.MTE1)
                                    m_pv0.lock(pipe=tla.pipes.MTE1)
                                    read_slot = (count - 1) % 2
                                    if read_slot == 0:
                                        tla.cross_core_wait_flag(pr, tla.pipes.MTE1)
                                    else:
                                        tla.cross_core_wait_flag(pr_alt, tla.pipes.MTE1)
                                    read_p = tla.make_tensor_like(
                                        p1p + read_slot * bm * bn, p1, tla.arch.zN
                                    )
                                    read_ps = tla.make_tensor_like(
                                        ps1p + read_slot * bm * 4,
                                        ps1,
                                        tla.arch.nNMxScale,
                                    )
                                    tla.copy(p0, read_p, scale=read_ps)
                                    tla.copy(v0, v1, scale=vs1)
                                    if read_slot == 0:
                                        tla.cross_core_set_flag(pr, tla.pipes.MTE1)
                                    else:
                                        tla.cross_core_set_flag(pr_alt, tla.pipes.MTE1)
                                    m_pv0.unlock(pipe=tla.pipes.MTE1)
                                    m_v1.unlock(pipe=tla.pipes.MTE1)
                                    m_pv0.lock(pipe=tla.pipes.CUBE)
                                    m_o0.lock(pipe=tla.pipes.CUBE)
                                    tla.mmad_mx(o0, v0, p0, init_c=True)
                                    m_o0.unlock(pipe=tla.pipes.CUBE)
                                    m_pv0.unlock(pipe=tla.pipes.CUBE)
                                    tla.cross_core_wait_flag(ore, tla.pipes.FIX)
                                    m_o0.lock(pipe=tla.pipes.FIX)
                                    tla.copy(ofix, o0, split_o)
                                    m_o0.unlock(pipe=tla.pipes.FIX)
                                    tla.cross_core_set_flag(ore, tla.pipes.FIX)
                                with tla.vector():
                                    tla.cross_core_wait_flag(ore, tla.pipes.VECTOR)
                                    mx_epilogue(
                                        ap,
                                        op,
                                        mp,
                                        lp,
                                        rp + ((count - 1) % 2) * half,
                                        tla.recast_ptr(yp, dtype=tla.UInt8),
                                        ep,
                                        dim,
                                        valid,
                                        1,
                                        od_code,
                                    )
                                    tla.cross_core_set_flag(ore, tla.pipes.VECTOR)
                            pending_start = start
                            pending_cols = cols
                            count = count + 1
            # Drain the final PV tile left by the one-tile lookahead.
            if count > 0:
                with tla.cube():
                    m_v1.lock(pipe=tla.pipes.MTE2)
                    if pending_cols < bn:
                        tla.get_tile(
                            v1, tla.make_coord(0, 0), tla.make_shape(dim, bn)
                        ).fill(0)
                    vgm_drain = tla.make_tensor(
                        v.ptr + kbase + pending_start * dim,
                        tla.make_layout(
                            tla.make_shape(dim, pending_cols),
                            tla.make_stride(1, dim),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    tla.copy(
                        tla.tile_view(
                            v1, tla.make_shape(dim, pending_cols), tla.make_coord(0, 0)
                        ),
                        vgm_drain,
                    )
                    vgroups_drain = ((pending_cols + 63) // 64) * 2
                    vsgm_drain = tla.get_tile(
                        vs,
                        tla.make_coord(0, vb // dim + pending_start // 32),
                        tla.make_shape(dim, vgroups_drain),
                    )
                    tla.copy(
                        tla.tile_view(
                            vs1,
                            tla.make_shape(dim, vgroups_drain),
                            tla.make_coord(0, 0),
                        ),
                        vsgm_drain,
                    )
                    m_v1.unlock(pipe=tla.pipes.MTE2)
                    m_v1.lock(pipe=tla.pipes.MTE1)
                    m_pv0.lock(pipe=tla.pipes.MTE1)
                    drain_slot = (count - 1) % 2
                    if drain_slot == 0:
                        tla.cross_core_wait_flag(pr, tla.pipes.MTE1)
                    else:
                        tla.cross_core_wait_flag(pr_alt, tla.pipes.MTE1)
                    drain_p = tla.make_tensor_like(
                        p1p + drain_slot * bm * bn, p1, tla.arch.zN
                    )
                    drain_ps = tla.make_tensor_like(
                        ps1p + drain_slot * bm * 4, ps1, tla.arch.nNMxScale
                    )
                    tla.copy(p0, drain_p, scale=drain_ps)
                    tla.copy(v0, v1, scale=vs1)
                    if drain_slot == 0:
                        tla.cross_core_set_flag(pr, tla.pipes.MTE1)
                    else:
                        tla.cross_core_set_flag(pr_alt, tla.pipes.MTE1)
                    m_pv0.unlock(pipe=tla.pipes.MTE1)
                    m_v1.unlock(pipe=tla.pipes.MTE1)
                    m_pv0.lock(pipe=tla.pipes.CUBE)
                    m_o0.lock(pipe=tla.pipes.CUBE)
                    tla.mmad_mx(o0, v0, p0, init_c=True)
                    m_o0.unlock(pipe=tla.pipes.CUBE)
                    m_pv0.unlock(pipe=tla.pipes.CUBE)
                    tla.cross_core_wait_flag(ore, tla.pipes.FIX)
                    m_o0.lock(pipe=tla.pipes.FIX)
                    tla.copy(ofix, o0, split_o)
                    m_o0.unlock(pipe=tla.pipes.FIX)
                    tla.cross_core_set_flag(ore, tla.pipes.FIX)
                with tla.vector():
                    tla.cross_core_wait_flag(ore, tla.pipes.VECTOR)
                    mx_epilogue(
                        ap,
                        op,
                        mp,
                        lp,
                        rp + ((count - 1) % 2) * half,
                        tla.recast_ptr(yp, dtype=tla.UInt8),
                        ep,
                        dim,
                        valid,
                        1,
                        od_code,
                    )
                    tla.cross_core_set_flag(ore, tla.pipes.VECTOR)
            with tla.vector():
                m_out.lock(pipe=tla.pipes.VECTOR)
                mx_epilogue(
                    ap,
                    op,
                    mp,
                    lp,
                    rp,
                    tla.recast_ptr(yp, dtype=tla.UInt8),
                    ep,
                    dim,
                    valid,
                    2,
                    od_code,
                )
                m_out.unlock(pipe=tla.pipes.VECTOR)
                if valid > 0:
                    m_out.lock(pipe=tla.pipes.MTE3)
                    ogm = tla.make_tensor(
                        out.ptr + qbase + sub * half * dim,
                        tla.make_layout(
                            tla.make_shape(valid, dim), tla.make_stride(dim, 1)
                        ),
                    )
                    tla.copy(
                        ogm,
                        tla.tile_view(
                            yu, tla.make_shape(valid, dim), tla.make_coord(0, 0)
                        ),
                    )
                    if want_lse:
                        egm = tla.make_tensor(
                            lse.ptr + qbase // dim + sub * half,
                            tla.make_layout(tla.make_shape(valid), tla.make_stride(1)),
                        )
                        tla.copy(
                            egm,
                            tla.tile_view(eu, tla.make_shape(valid), tla.make_coord(0)),
                        )
                    m_out.unlock(pipe=tla.pipes.MTE3)
    with tla.cube():
        tla.cross_core_wait_flag(sr, tla.pipes.FIX)
        tla.cross_core_wait_flag(ore, tla.pipes.FIX)
    with tla.vector():
        tla.cross_core_wait_flag(pr, tla.pipes.MTE3)
        tla.cross_core_wait_flag(pr_alt, tla.pipes.MTE3)


# Attention kernel: pipeline_mxfp8.
@tla.kernel
def quant_block_sparse_attention_mxfp8_kernel(
    q: tla.Tensor,
    k: tla.Tensor,
    v: tla.Tensor,
    out: tla.Tensor,
    qs: tla.Tensor,
    ks: tla.Tensor,
    vs: tla.Tensor,
    indices: tla.Tensor,
    lse: tla.Tensor,
    batch: tla.Constexpr[int],
    hq: tla.Constexpr[int],
    hk: tla.Constexpr[int],
    nq: tla.Constexpr[int],
    nk: tla.Constexpr[int],
    mh: tla.Constexpr[int],
    selected: tla.Constexpr[int],
    scale: tla.Constexpr[float],
    dim: tla.Constexpr[int],
    xb: tla.Constexpr[int],
    yb: tla.Constexpr[int],
    want_lse: tla.Constexpr[bool],
):
    # Transposed QK/PV: each AIV owns 64 query lanes. PV adds a
    # constant-one channel for the quantized probability denominator.
    group = min(16, selected)
    ring = min(group + 2, max(1, selected))
    delay = ring
    vring = 2
    bm = 128
    bn = 128
    dv = dim + 32
    pv_m = dim + 16
    half = 64
    qm = 64 if xb % 128 != 0 else 128
    fp8 = tla.Float8E4M3FN
    e8m0 = tla.Float8E8M0
    od = out.ptr.dtype
    output_stride = dim + (8 if od == tla.Float32 else 16)
    od_code = 0 if od == tla.Float32 else (1 if od == tla.Float16 else 2)
    q1p = tla.allocate(bm * dim, fp8, tla.AddressSpace.l1, 512)
    k1p = tla.allocate(2 * bn * dim, fp8, tla.AddressSpace.l1, 512)
    v1p = tla.allocate(vring * bn * dv, fp8, tla.AddressSpace.l1, 512)
    # Keep the current group and two look-ahead P tiles in L1.
    p1p = tla.allocate(ring * bm * bn, fp8, tla.AddressSpace.l1, 512)
    qs1p = tla.allocate(bm * (dim // 32), e8m0, tla.AddressSpace.l1, 512)
    ks1p = tla.allocate(2 * bn * (dim // 32), e8m0, tla.AddressSpace.l1, 512)
    vs1p = tla.allocate(vring * dv * (bn // 32), e8m0, tla.AddressSpace.l1, 512)
    ps1p = tla.allocate(ring * bm * (bn // 32), e8m0, tla.AddressSpace.l1, 512)
    q0p = tla.allocate(bm * dim, fp8, tla.AddressSpace.l0b, 1024)
    k0p = tla.allocate(bn * dim, fp8, tla.AddressSpace.l0a, 1024)
    p0p = tla.allocate(2 * bm * bn, fp8, tla.AddressSpace.l0b, 1024)
    v0p = tla.allocate(2 * bn * dv, fp8, tla.AddressSpace.l0a, 1024)
    s0p = tla.allocate(2 * bm * bn, tla.Float32, tla.AddressSpace.l0c, 1024)
    o0p = tla.allocate(bm * dv, tla.Float32, tla.AddressSpace.l0c, 1024)
    # Separate score loads and probability stores into different 32 KiB UB
    # regions. The second half is spacing, not an additional score tile.
    sp = tla.allocate(4 * half * bn, tla.Float16, tla.AddressSpace.ub, 512)
    pp_stride = 32768
    pp = tla.allocate(pp_stride + half * (bn + 1), tla.UInt8, tla.AddressSpace.ub, 512)
    psp = tla.allocate(16 * half * (bn // 32), tla.UInt8, tla.AddressSpace.ub, 256)
    mp = tla.allocate(2 * half, tla.Float32, tla.AddressSpace.ub, 256)
    lp = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)
    rp = tla.allocate(2 * half, tla.Float32, tla.AddressSpace.ub, 256)
    gp = tla.allocate(16 * half * (bn // 32), tla.Float32, tla.AddressSpace.ub, 256)
    op = tla.allocate(half * pv_m, tla.Float32, tla.AddressSpace.ub, 512)
    ap = tla.allocate(half * pv_m, tla.Float32, tla.AddressSpace.ub, 512)
    yp = tla.allocate(half * output_stride, od, tla.AddressSpace.ub, 512)
    tmp_alpha = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)
    ep = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)

    q1 = tla.make_tensor(q1p, tla.make_layout_with_tag((dim, bm), fp8, tla.arch.nZ))
    k1 = tla.make_tensor(k1p, tla.make_layout_with_tag((bn, dim), fp8, tla.arch.zN))
    v1 = tla.make_tensor(v1p, tla.make_layout_with_tag((dv, bn), fp8, tla.arch.nZ))
    p1 = tla.make_tensor(p1p, tla.make_layout_with_tag((bn, bm), fp8, tla.arch.zN))
    qs1 = tla.make_tensor(
        qs1p, tla.make_layout_with_tag((dim // 32, bm), e8m0, tla.arch.nNMxScale)
    )
    ks1 = tla.make_tensor(
        ks1p, tla.make_layout_with_tag((bn, dim // 32), e8m0, tla.arch.zZMxScale)
    )
    vs1 = tla.make_tensor(
        vs1p, tla.make_layout_with_tag((dv, bn // 32), e8m0, tla.arch.zZMxScale)
    )
    ps1 = tla.make_tensor(
        ps1p, tla.make_layout_with_tag((bn // 32, bm), e8m0, tla.arch.nNMxScale)
    )
    q0 = tla.make_tensor(q0p, tla.make_layout_with_tag((dim, bm), fp8, tla.arch.nZ))
    k0 = tla.make_tensor(k0p, tla.make_layout_with_tag((bn, dim), fp8, tla.arch.zN))
    p0 = tla.make_tensor(p0p, tla.make_layout_with_tag((bn, bm), fp8, tla.arch.nZ))
    v0 = tla.make_tensor(v0p, tla.make_layout_with_tag((pv_m, bn), fp8, tla.arch.zN))
    s0 = tla.make_tensor(
        s0p, tla.make_layout_with_tag((bn, bm), tla.Float32, tla.arch.L0Clayout)
    )
    o0 = tla.make_tensor(
        o0p, tla.make_layout_with_tag((pv_m, bm), tla.Float32, tla.arch.L0Clayout)
    )
    sfix = tla.make_tensor(
        sp, tla.make_layout(tla.make_shape(bn, bm), tla.make_stride(half, 1))
    )
    afix = tla.make_tensor(
        ap, tla.make_layout(tla.make_shape(pv_m, bm), tla.make_stride(half, 1))
    )
    ofix = tla.make_tensor(
        op, tla.make_layout(tla.make_shape(pv_m, bm), tla.make_stride(half, 1))
    )
    p16ref = tla.make_tensor(
        tla.recast_ptr(pp, dtype=tla.Float16),
        tla.make_layout(
            tla.make_shape(bn + 1, half // 2), tla.make_stride(half // 2, 1)
        ),
    )
    pu = tla.make_tensor_like(p16ref.ptr, p16ref, tla.arch.zNUnAlign)
    p1storage = tla.make_tensor(
        tla.recast_ptr(p1p, dtype=tla.Float16),
        tla.make_layout_with_tag((bn, bm // 2), tla.Float16, tla.arch.zN),
    )
    psuref = tla.make_tensor(
        tla.recast_ptr(psp, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(8, 16), tla.make_stride(16, 1)),
    )
    psu = tla.make_tensor_like(psuref.ptr, psuref, tla.arch.zNUnAlign)
    psflat = tla.make_tensor(
        tla.recast_ptr(ps1p, dtype=tla.Float16),
        tla.make_layout_with_tag((16, 16), tla.Float16, tla.arch.zN),
    )
    padp = tla.allocate(2048, tla.UInt8, tla.AddressSpace.ub, 512)
    pads = tla.allocate(64, tla.UInt8, tla.AddressSpace.ub, 256)
    padref = tla.make_tensor(
        tla.recast_ptr(padp, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(64, 16), tla.make_stride(16, 1)),
    )
    padu = tla.make_tensor_like(padref.ptr, padref, tla.arch.zNUnAlign)
    pad1 = tla.make_tensor(
        tla.recast_ptr(v1p + dim * bn, dtype=tla.Float16),
        tla.make_layout_with_tag((128, 16), tla.Float16, tla.arch.zN),
    )
    padsref = tla.make_tensor(
        tla.recast_ptr(pads, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(2, 16), tla.make_stride(16, 1)),
    )
    padsu = tla.make_tensor_like(padsref.ptr, padsref, tla.arch.zNUnAlign)
    pads1 = tla.make_tensor(
        tla.recast_ptr(vs1p + dim * 4, dtype=tla.Float16),
        tla.make_layout_with_tag((4, 16), tla.Float16, tla.arch.zN),
    )
    yu = tla.make_tensor(
        yp,
        tla.make_layout(tla.make_shape(half, dim), tla.make_stride(output_stride, 1)),
    )
    eu = tla.make_tensor(ep, tla.make_layout(tla.make_shape(half), tla.make_stride(1)))

    # In-core buffer ownership is expressed with mutexes.
    m_q1 = tla.mutex("q1", id=0)
    m_q0 = tla.mutex("q0", id=1)
    m_k1_0 = tla.mutex("k1_0", id=2)
    m_k1_1 = tla.mutex("k1_1", id=3)
    m_k0 = tla.mutex("k0", id=4)
    m_v1_0 = tla.mutex("v1_0", id=5)
    m_v1_1 = tla.mutex("v1_1", id=6)
    m_pv_0 = tla.mutex("pv_0", id=7)
    m_pv_1 = tla.mutex("pv_1", id=8)
    m_o0 = tla.mutex("o0", id=9)
    m_pad = tla.mutex("pad", id=10)
    m_pp_0 = tla.mutex("pp_0", id=11)
    m_pp_1 = tla.mutex("pp_1", id=12)
    m_ps = tla.mutex("ps", id=13)
    m_out = tla.mutex("out", id=14)
    sr = tla.cross_flag("score_ready")
    sr_second = tla.cross_flag("score_ready_second")
    ore = tla.cross_flag("output_ready")
    psready = tla.cross_flag("scales_ready")
    pfree = tla.cross_flag("prob_consumed")
    padready = tla.cross_flag("pad_ready")
    split_o = tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N)
    with tla.vector():
        m_pp_0.lock(pipe=tla.pipes.VECTOR)
        m_pp_1.lock(pipe=tla.pipes.VECTOR)
        mx_zero_ub(pp, half * (bn + 1))
        mx_zero_ub(pp + pp_stride, half * (bn + 1))
        m_pp_1.unlock(pipe=tla.pipes.VECTOR)
        m_pp_0.unlock(pipe=tla.pipes.VECTOR)
        m_pad.lock(pipe=tla.pipes.VECTOR)
        mx_pad(padp, pads)
        m_pad.unlock(pipe=tla.pipes.VECTOR)
        m_pp_0.lock(pipe=tla.pipes.MTE3)
        m_pp_1.lock(pipe=tla.pipes.MTE3)
        m_pad.lock(pipe=tla.pipes.MTE3)
        if tla.arch.sub_block_idx() == 0:
            mx_zero_scale_l1(ks1p, pp, 2 * bn * (dim // 32))
            mx_zero_scale_l1(vs1p, pp, vring * dv * 4)
            for seed_slot in tla.range(vring):
                seed_v1 = tla.make_tensor_like(
                    tla.recast_ptr(
                        v1p + seed_slot * bn * dv + dim * bn, dtype=tla.Float16
                    ),
                    pad1,
                    tla.arch.zN,
                )
                seed_vs1 = tla.make_tensor_like(
                    tla.recast_ptr(
                        vs1p + seed_slot * dv * 4 + dim * 4, dtype=tla.Float16
                    ),
                    pads1,
                    tla.arch.zN,
                )
                for seed_sub in tla.range(2):
                    tla.copy(
                        tla.tile_view(
                            seed_v1, tla.make_shape(64, 16), tla.make_coord(seed_sub, 0)
                        ),
                        padu,
                    )
                    tla.copy(
                        tla.tile_view(
                            seed_vs1, tla.make_shape(2, 16), tla.make_coord(seed_sub, 0)
                        ),
                        padsu,
                    )
        m_pad.unlock(pipe=tla.pipes.MTE3)
        m_pp_1.unlock(pipe=tla.pipes.MTE3)
        m_pp_0.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(padready, tla.pipes.MTE3)
        tla.cross_core_set_flag(sr, tla.pipes.VECTOR)
        tla.cross_core_set_flag(sr_second, tla.pipes.VECTOR)
        tla.cross_core_set_flag(ore, tla.pipes.VECTOR)
    with tla.cube():
        tla.cross_core_wait_flag(padready, tla.pipes.MTE2)
    blocks = (nq + 127) // 128
    total = batch * hq * blocks
    core = tla.arch.block_idx()
    cores = tla.arch.block_num()
    task_count = (total - core + cores - 1) // cores
    total_tiles = task_count * selected
    sub = tla.arch.sub_block_idx()
    qordered = tla.as_numeric(1)
    pordered = tla.as_numeric(1)
    qlast = tla.as_numeric(-1)
    vlast = tla.as_numeric(-1)
    qbase = tla.as_numeric(0)
    kbase = tla.as_numeric(0)
    maskbase = tla.as_numeric(0)
    pqbase = tla.as_numeric(0)
    pkbase = tla.as_numeric(0)
    vb = tla.as_numeric(0)
    pmaskbase = tla.as_numeric(0)
    rows = 128 if nq % 128 == 0 else tla.as_numeric(128)
    valid = 64 if nq % 128 == 0 else tla.as_numeric(64)
    for step in tla.range(total_tiles + delay):
        pi = step - delay
        qtile = step % selected
        ptile = pi % selected
        if step < total_tiles and qtile == 0:
            task = core + (step // selected) * cores
            h = (task // blocks) % hq
            qb = task % blocks
            b = task // (hq * blocks)
            kh = h // (hq // hk)
            qstart = qb * 128
            if nq % 128 != 0:
                rows = 128 if nq - qstart >= 128 else nq - qstart
            qbase = ((b * hq + h) * nq + qstart) * dim
            kbase = (b * hk + kh) * nk * dim
            mask_head = 0 if mh == 1 else h
            maskbase = ((b * mh + mask_head) * blocks + qb) * selected
        if pi >= 0 and ptile == 0:
            ptask = core + (pi // selected) * cores
            ph = (ptask // blocks) % hq
            pqb = ptask % blocks
            pb = ptask // (hq * blocks)
            pkh = ph // (hq // hk)
            pqstart = pqb * 128
            if nq % 128 != 0:
                prows = 128 if nq - pqstart >= 128 else nq - pqstart
                valid = prows - sub * half
                valid = half if valid > half else valid
                valid = 0 if valid < 0 else valid
            pqbase = ((pb * hq + ph) * nq + pqstart) * dim
            pkbase = (pb * hk + pkh) * nk * dim
            vb = (pb * hk + pkh) * ((nk + 63) // 64) * dim * 2
            pmask_head = 0 if mh == 1 else ph
            pmaskbase = ((pb * mh + pmask_head) * blocks + pqb) * selected
        qmp = mp + ((step // selected) % 2) * half
        pmp = mp + ((pi // selected) % 2) * half
        if step < total_tiles:
            if qtile == 0:
                qordered = tla.as_numeric(1)
                qlast = tla.as_numeric(-1)
                qprevious = tla.as_numeric(-1)
                for qscan in tla.range(selected):
                    qvalue = indices[maskbase + qscan].to(tla.Int32)
                    if qvalue >= 0:
                        if qvalue < qprevious:
                            qordered = tla.as_numeric(0)
                        qprevious = qvalue
                with tla.cube():
                    m_q0.lock(pipe=tla.pipes.MTE1)
                    m_q1.lock(pipe=tla.pipes.MTE2)
                    qgm = tla.make_tensor(
                        q.ptr + qbase,
                        tla.make_layout(
                            tla.make_shape(dim, rows),
                            tla.make_stride(1, dim),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    tla.copy(
                        tla.tile_view(
                            q1, tla.make_shape(dim, rows), tla.make_coord(0, 0)
                        ),
                        qgm,
                    )
                    qsgm = tla.get_tile(
                        qs,
                        tla.make_coord(0, qbase // dim),
                        tla.make_shape(dim // 32, rows),
                    )
                    tla.copy(
                        tla.tile_view(
                            qs1, tla.make_shape(dim // 32, rows), tla.make_coord(0, 0)
                        ),
                        qsgm,
                    )
                    m_q1.unlock(pipe=tla.pipes.MTE2)
                    m_q1.lock(pipe=tla.pipes.MTE1)
                    tla.copy(q0, q1, scale=qs1)
                    m_q1.unlock(pipe=tla.pipes.MTE1)
                    m_q0.unlock(pipe=tla.pipes.MTE1)
                    m_q0.lock(pipe=tla.pipes.CUBE)
                with tla.vector():
                    _mx_group_scales_fullstore(psp, qmp, lp, rp, gp, -2)
        if pi >= 0:
            if ptile == 0:
                pordered = tla.as_numeric(1)
                vlast = tla.as_numeric(-1)
                pprevious = tla.as_numeric(-1)
                for pscan in tla.range(selected):
                    pvalue = indices[pmaskbase + pscan].to(tla.Int32)
                    if pvalue >= 0:
                        if pvalue < pprevious:
                            pordered = tla.as_numeric(0)
                        pprevious = pvalue
        qcols = tla.as_numeric(0)
        qstart_k = tla.as_numeric(0)
        vcols = tla.as_numeric(0)
        vstart_k = tla.as_numeric(0)
        if step < total_tiles:
            qblock = indices[maskbase + qtile].to(tla.Int32)
            if qordered == 0:
                qblock = (nk + bn - 1) // bn
                for qi_scan in tla.range(selected):
                    qcandidate = indices[maskbase + qi_scan].to(tla.Int32)
                    if qcandidate > qlast:
                        if qcandidate < qblock:
                            qblock = qcandidate
            if qblock > qlast:
                qlast = qblock
                if qblock * bn < nk:
                    qstart_k = qblock * bn
                    qcols = bn if nk - qstart_k >= bn else nk - qstart_k
        if pi >= 0:
            pblock = indices[pmaskbase + ptile].to(tla.Int32)
            if pordered == 0:
                pblock = (nk + bn - 1) // bn
                for vi_scan in tla.range(selected):
                    vcandidate = indices[pmaskbase + vi_scan].to(tla.Int32)
                    if vcandidate > vlast:
                        if vcandidate < pblock:
                            pblock = vcandidate
            if pblock > vlast:
                vlast = pblock
                if pblock * bn < nk:
                    vstart_k = pblock * bn
                    vcols = bn if nk - vstart_k >= bn else nk - vstart_k
        with tla.cube():
            if step < total_tiles:
                if step % 2 == 0:
                    m_k1_0.lock(pipe=tla.pipes.MTE2)
                else:
                    m_k1_1.lock(pipe=tla.pipes.MTE2)
                current_k1 = tla.make_tensor_like(
                    k1p + (step % 2) * bn * dim, k1, tla.arch.zN
                )
                current_ks1 = tla.make_tensor_like(
                    ks1p + (step % 2) * bn * (dim // 32), ks1, tla.arch.zZMxScale
                )
                if qcols < bn:
                    current_k1.fill(0)
                if qcols > 0:
                    k_copy_cols = bn if nk % bn == 0 else qcols
                    kgm = tla.make_tensor(
                        k.ptr + kbase + qstart_k * dim,
                        tla.make_layout(
                            tla.make_shape(k_copy_cols, dim), tla.make_stride(dim, 1)
                        ),
                    )
                    ksgm = tla.get_tile(
                        ks,
                        tla.make_coord(kbase // dim + qstart_k, 0),
                        tla.make_shape(k_copy_cols, dim // 32),
                    )
                    tla.copy(
                        tla.tile_view(
                            current_k1,
                            tla.make_shape(k_copy_cols, dim),
                            tla.make_coord(0, 0),
                        ),
                        kgm,
                    )
                    tla.copy(
                        tla.tile_view(
                            current_ks1,
                            tla.make_shape(k_copy_cols, dim // 32),
                            tla.make_coord(0, 0),
                        ),
                        ksgm,
                    )
                if step % 2 == 0:
                    m_k1_0.unlock(pipe=tla.pipes.MTE2)
                else:
                    m_k1_1.unlock(pipe=tla.pipes.MTE2)
            if pi >= 0:
                if pi % 2 == 0:
                    m_v1_0.lock(pipe=tla.pipes.MTE2)
                else:
                    m_v1_1.lock(pipe=tla.pipes.MTE2)
                preload_vlong = tla.make_tensor_like(
                    v1p + (pi % 2) * bn * dv, v1, tla.arch.nZ
                )
                preload_vslong = tla.make_tensor_like(
                    vs1p + (pi % 2) * dv * 4, vs1, tla.arch.zZMxScale
                )
                if vcols < bn:
                    tla.get_tile(
                        preload_vlong, tla.make_coord(0, 0), tla.make_shape(dim, bn)
                    ).fill(0)
                if vcols > 0:
                    v_copy_cols = bn if nk % bn == 0 else vcols
                    long_vgm = tla.make_tensor(
                        v.ptr + pkbase + vstart_k * dim,
                        tla.make_layout(
                            tla.make_shape(dim, v_copy_cols),
                            tla.make_stride(1, dim),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    long_groups = ((v_copy_cols + 63) // 64) * 2
                    long_vsgm = tla.get_tile(
                        vs,
                        tla.make_coord(0, vb // dim + vstart_k // 32),
                        tla.make_shape(dim, long_groups),
                    )
                    tla.copy(
                        tla.tile_view(
                            preload_vlong,
                            tla.make_shape(dim, v_copy_cols),
                            tla.make_coord(0, 0),
                        ),
                        long_vgm,
                    )
                    tla.copy(
                        tla.tile_view(
                            preload_vslong,
                            tla.make_shape(dim, long_groups),
                            tla.make_coord(0, 0),
                        ),
                        long_vsgm,
                    )
                if pi % 2 == 0:
                    m_v1_0.unlock(pipe=tla.pipes.MTE2)
                else:
                    m_v1_1.unlock(pipe=tla.pipes.MTE2)
            if step < total_tiles:
                if step % 2 == 0:
                    m_k1_0.lock(pipe=tla.pipes.MTE1)
                else:
                    m_k1_1.lock(pipe=tla.pipes.MTE1)
                m_k0.lock(pipe=tla.pipes.MTE1)
                current_k0 = tla.make_tensor_like(k0p, k0, tla.arch.zN)
                read_k1 = tla.make_tensor_like(
                    k1p + (step % 2) * bn * dim, k1, tla.arch.zN
                )
                read_ks1 = tla.make_tensor_like(
                    ks1p + (step % 2) * bn * (dim // 32), ks1, tla.arch.zZMxScale
                )
                tla.copy(current_k0, read_k1, scale=read_ks1)
                if step % 2 == 0:
                    m_k1_0.unlock(pipe=tla.pipes.MTE1)
                else:
                    m_k1_1.unlock(pipe=tla.pipes.MTE1)
                m_k0.unlock(pipe=tla.pipes.MTE1)
                m_k0.lock(pipe=tla.pipes.CUBE)
                current_s0 = tla.make_tensor_like(
                    s0p + (step % 2) * bm * bn, s0, tla.arch.L0Clayout
                )
                # Matched with unitFlag=3 in mx_score_unit; synchronization
                # is per L0C unit instead of a whole-tile CUBE/FIX event.
                tla.mmad_mx(current_s0, current_k0, q0, init_c=True, unit_flag=3)
                m_k0.unlock(pipe=tla.pipes.CUBE)
                if qtile == selected - 1:
                    m_q0.unlock(pipe=tla.pipes.CUBE)
            if pi >= 0:
                if pi % 2 == 0:
                    m_v1_0.lock(pipe=tla.pipes.MTE1)
                else:
                    m_v1_1.lock(pipe=tla.pipes.MTE1)
                if pi % 2 == 0:
                    m_pv_0.lock(pipe=tla.pipes.MTE1)
                else:
                    m_pv_1.lock(pipe=tla.pipes.MTE1)
                pv_v0 = tla.make_tensor_like(v0p + (pi % 2) * bn * dv, v0, tla.arch.zN)
                pv_p0 = tla.make_tensor_like(p0p + (pi % 2) * bm * bn, p0, tla.arch.nZ)
                read_v1 = tla.make_tensor_like(
                    v1p + (pi % vring) * bn * dv, v1, tla.arch.nZ
                )
                read_vs1 = tla.make_tensor_like(
                    vs1p + (pi % vring) * dv * 4, vs1, tla.arch.zZMxScale
                )
                read_p1 = tla.make_tensor_like(
                    p1p + (pi % ring) * bm * bn, p1, tla.arch.zN
                )
                read_ps1 = tla.make_tensor_like(
                    ps1p + (pi % ring) * bm * 4, ps1, tla.arch.nNMxScale
                )
                # V does not depend on P scales. Release its L1 slot immediately
                # after MTE1 has read it, before waiting for the group scales.
                tla.copy(
                    pv_v0,
                    tla.get_tile(
                        read_v1, tla.make_coord(0, 0), tla.make_shape(pv_m, bn)
                    ),
                    scale=tla.get_tile(
                        read_vs1, tla.make_coord(0, 0), tla.make_shape(pv_m, bn // 32)
                    ),
                )
                if pi % 2 == 0:
                    m_v1_0.unlock(pipe=tla.pipes.MTE1)
                else:
                    m_v1_1.unlock(pipe=tla.pipes.MTE1)
                if ptile % group == 0:
                    tla.cross_core_wait_flag(psready, tla.pipes.MTE1)
                tla.copy(pv_p0, read_p1, scale=read_ps1)
                tla.cross_core_set_flag(pfree, tla.pipes.MTE1)
                if pi % 2 == 0:
                    m_pv_0.unlock(pipe=tla.pipes.MTE1)
                    m_pv_0.lock(pipe=tla.pipes.CUBE)
                else:
                    m_pv_1.unlock(pipe=tla.pipes.MTE1)
                    m_pv_1.lock(pipe=tla.pipes.CUBE)
                if ptile % group == 0:
                    m_o0.lock(pipe=tla.pipes.CUBE)
                tla.mmad_mx(o0, pv_v0, pv_p0, init_c=ptile % group == 0)
                if pi % 2 == 0:
                    m_pv_0.unlock(pipe=tla.pipes.CUBE)
                else:
                    m_pv_1.unlock(pipe=tla.pipes.CUBE)
                if ptile % group == group - 1 or ptile == selected - 1:
                    m_o0.unlock(pipe=tla.pipes.CUBE)
            if step < total_tiles:
                if step % 2 == 0:
                    tla.cross_core_wait_flag(sr, tla.pipes.FIX)
                else:
                    tla.cross_core_wait_flag(sr_second, tla.pipes.FIX)
                mx_score_unit(
                    sp + (step % 2) * 2 * half * bn, s0p + (step % 2) * bm * bn, scale
                )
                if step % 2 == 0:
                    tla.cross_core_set_flag(sr, tla.pipes.FIX)
                else:
                    tla.cross_core_set_flag(sr_second, tla.pipes.FIX)
            if pi >= 0:
                if ptile % group == group - 1 or ptile == selected - 1:
                    m_o0.lock(pipe=tla.pipes.FIX)
                    tla.cross_core_wait_flag(ore, tla.pipes.FIX)
                    if ptile < group:
                        tla.copy(afix, o0, split_o)
                    else:
                        tla.copy(ofix, o0, split_o)
                    tla.cross_core_set_flag(ore, tla.pipes.FIX)
                    m_o0.unlock(pipe=tla.pipes.FIX)
        with tla.vector():
            if step < total_tiles:
                if step % 2 == 0:
                    tla.cross_core_wait_flag(sr, tla.pipes.VECTOR)
                else:
                    tla.cross_core_wait_flag(sr_second, tla.pipes.VECTOR)
                if step % 2 == 0:
                    m_pp_0.lock(pipe=tla.pipes.VECTOR)
                else:
                    m_pp_1.lock(pipe=tla.pipes.VECTOR)
                if qtile % group == 0:
                    _mx_group_scales_fullstore(psp, qmp, lp, rp, gp, -1)
                _mx_group_softmax_store_fast(
                    sp + (step % 2) * 2 * half * bn,
                    pp + (step % 2) * pp_stride,
                    psp,
                    qmp,
                    lp,
                    tmp_alpha,
                    gp + (qtile % group) * 256,
                    scale,
                    qcols,
                )
                if step % 2 == 0:
                    tla.cross_core_set_flag(sr, tla.pipes.VECTOR)
                else:
                    tla.cross_core_set_flag(sr_second, tla.pipes.VECTOR)
            if pi >= 0:
                tla.cross_core_wait_flag(pfree, tla.pipes.MTE3)
            if step < total_tiles:
                if step % 2 == 0:
                    m_pp_0.unlock(pipe=tla.pipes.VECTOR)
                    m_pp_0.lock(pipe=tla.pipes.MTE3)
                else:
                    m_pp_1.unlock(pipe=tla.pipes.VECTOR)
                    m_pp_1.lock(pipe=tla.pipes.MTE3)
                write_p = tla.make_tensor_like(
                    tla.recast_ptr(p1p + (step % ring) * bm * bn, dtype=tla.Float16),
                    p1storage,
                    tla.arch.zN,
                )
                puc = tla.make_tensor_like(
                    tla.recast_ptr(pp + (step % 2) * pp_stride, dtype=tla.Float16),
                    pu,
                    tla.arch.zNUnAlign,
                )
                tla.copy(
                    tla.tile_view(
                        write_p,
                        tla.make_shape(bn, half // 2),
                        tla.make_coord(0, sub),
                    ),
                    tla.tile_view(
                        puc, tla.make_shape(bn, half // 2), tla.make_coord(0, 0)
                    ),
                )
                if step % 2 == 0:
                    m_pp_0.unlock(pipe=tla.pipes.MTE3)
                else:
                    m_pp_1.unlock(pipe=tla.pipes.MTE3)
                if qtile % group == group - 1 or qtile == selected - 1:
                    group_count = qtile % group + 1
                    group_start = step - qtile % group
                    m_ps.lock(pipe=tla.pipes.VECTOR)
                    _mx_group_scales_fullstore(
                        psp,
                        qmp,
                        lp,
                        rp
                        + (
                            (
                                (step // selected) * ((selected + group - 1) // group)
                                + qtile // group
                            )
                            % 2
                        )
                        * half,
                        gp,
                        group_count,
                    )
                    m_ps.unlock(pipe=tla.pipes.VECTOR)
                    m_ps.lock(pipe=tla.pipes.MTE3)
                    mx_ps64_to_l1(ps1p, psp, group_start % ring, group_count, ring, sub)
                    m_ps.unlock(pipe=tla.pipes.MTE3)
                    tla.cross_core_set_flag(psready, tla.pipes.MTE3)
            if pi >= 0:
                if ptile % group == group - 1 or ptile == selected - 1:
                    tla.cross_core_wait_flag(ore, tla.pipes.VECTOR)
                    if ptile >= group:
                        mx_epilogue(
                            ap,
                            op,
                            pmp,
                            lp,
                            rp
                            + (
                                (
                                    (pi // selected) * ((selected + group - 1) // group)
                                    + ptile // group
                                )
                                % 2
                            )
                            * half,
                            tla.recast_ptr(yp, dtype=tla.UInt8),
                            ep,
                            dim,
                            valid,
                            1,
                            od_code,
                        )
                    if ptile == selected - 1:
                        m_out.lock(pipe=tla.pipes.VECTOR)
                        mx_epilogue(
                            ap,
                            op,
                            pmp,
                            lp,
                            rp,
                            tla.recast_ptr(yp, dtype=tla.UInt8),
                            ep,
                            dim,
                            valid,
                            2,
                            od_code,
                        )
                        m_out.unlock(pipe=tla.pipes.VECTOR)
                        if valid > 0:
                            m_out.lock(pipe=tla.pipes.MTE3)
                            ogm = tla.make_tensor(
                                out.ptr + pqbase + sub * half * dim,
                                tla.make_layout(
                                    tla.make_shape(valid, dim), tla.make_stride(dim, 1)
                                ),
                            )
                            tla.copy(
                                ogm,
                                tla.tile_view(
                                    yu, tla.make_shape(valid, dim), tla.make_coord(0, 0)
                                ),
                            )
                            if want_lse:
                                egm = tla.make_tensor(
                                    lse.ptr + pqbase // dim + sub * half,
                                    tla.make_layout(
                                        tla.make_shape(valid), tla.make_stride(1)
                                    ),
                                )
                                tla.copy(
                                    egm,
                                    tla.tile_view(
                                        eu, tla.make_shape(valid), tla.make_coord(0)
                                    ),
                                )
                            m_out.unlock(pipe=tla.pipes.MTE3)
                    tla.cross_core_set_flag(ore, tla.pipes.VECTOR)
    with tla.cube():
        tla.cross_core_wait_flag(sr, tla.pipes.FIX)
        tla.cross_core_wait_flag(sr_second, tla.pipes.FIX)
        tla.cross_core_wait_flag(ore, tla.pipes.FIX)


# Attention kernel: pipeline_bounded.
@tla.kernel
def quant_block_sparse_attention_bounded_kernel(
    q: tla.Tensor,
    k: tla.Tensor,
    v: tla.Tensor,
    out: tla.Tensor,
    qs: tla.Tensor,
    ks: tla.Tensor,
    vs: tla.Tensor,
    indices: tla.Tensor,
    lse: tla.Tensor,
    batch: tla.Constexpr[int],
    hq: tla.Constexpr[int],
    hk: tla.Constexpr[int],
    nq: tla.Constexpr[int],
    nk: tla.Constexpr[int],
    mh: tla.Constexpr[int],
    selected: tla.Constexpr[int],
    scale: tla.Constexpr[float],
    dim: tla.Constexpr[int],
    xb: tla.Constexpr[int],
    yb: tla.Constexpr[int],
    want_lse: tla.Constexpr[bool],
):
    # Transposed QK/PV: each AIV owns 64 query lanes. PV adds a
    # constant-one channel for the quantized probability denominator.
    group = min(16, selected)
    ring = min(group + 2, max(1, selected))
    delay = ring
    vring = 2
    bm = 128
    bn = 128
    dv = dim + 32
    pv_m = dim + 16
    half = 64
    qm = 64 if xb % 128 != 0 else 128
    fp8 = tla.Float8E4M3FN
    e8m0 = tla.Float8E8M0
    od = out.ptr.dtype
    output_stride = dim + (8 if od == tla.Float32 else 16)
    od_code = 0 if od == tla.Float32 else (1 if od == tla.Float16 else 2)
    q1p = tla.allocate(bm * dim, fp8, tla.AddressSpace.l1, 512)
    k1p = tla.allocate(2 * bn * dim, fp8, tla.AddressSpace.l1, 512)
    v1p = tla.allocate(vring * bn * dv, fp8, tla.AddressSpace.l1, 512)
    # Keep the current group and two look-ahead P tiles in L1.
    p1p = tla.allocate(ring * bm * bn, fp8, tla.AddressSpace.l1, 512)
    qs1p = tla.allocate(bm * (dim // 32), e8m0, tla.AddressSpace.l1, 512)
    ks1p = tla.allocate(2 * bn * (dim // 32), e8m0, tla.AddressSpace.l1, 512)
    vs1p = tla.allocate(vring * dv * (bn // 32), e8m0, tla.AddressSpace.l1, 512)
    ps1p = tla.allocate(ring * bm * (bn // 32), e8m0, tla.AddressSpace.l1, 512)
    q0p = tla.allocate(bm * dim, fp8, tla.AddressSpace.l0b, 1024)
    k0p = tla.allocate(bn * dim, fp8, tla.AddressSpace.l0a, 1024)
    p0p = tla.allocate(2 * bm * bn, fp8, tla.AddressSpace.l0b, 1024)
    v0p = tla.allocate(2 * bn * dv, fp8, tla.AddressSpace.l0a, 1024)
    s0p = tla.allocate(2 * bm * bn, tla.Float32, tla.AddressSpace.l0c, 1024)
    o0p = tla.allocate(bm * dv, tla.Float32, tla.AddressSpace.l0c, 1024)
    # Separate score loads and probability stores into different 32 KiB UB
    # regions. The second half is spacing, not an additional score tile.
    sp = tla.allocate(4 * half * bn, tla.Float16, tla.AddressSpace.ub, 512)
    pp_stride = 32768
    pp = tla.allocate(pp_stride + half * (bn + 1), tla.UInt8, tla.AddressSpace.ub, 512)
    psp = tla.allocate(16 * half * (bn // 32), tla.UInt8, tla.AddressSpace.ub, 256)
    mp = tla.allocate(2 * half, tla.Float32, tla.AddressSpace.ub, 256)
    lp = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)
    rp = tla.allocate(2 * half, tla.Float32, tla.AddressSpace.ub, 256)
    gp = tla.allocate(16 * half * (bn // 32), tla.Float32, tla.AddressSpace.ub, 256)
    op = tla.allocate(half * pv_m, tla.Float32, tla.AddressSpace.ub, 512)
    ap = tla.allocate(half * pv_m, tla.Float32, tla.AddressSpace.ub, 512)
    yp = tla.allocate(half * output_stride, od, tla.AddressSpace.ub, 512)
    tmp_alpha = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)
    ep = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)

    q1 = tla.make_tensor(q1p, tla.make_layout_with_tag((dim, bm), fp8, tla.arch.nZ))
    k1 = tla.make_tensor(k1p, tla.make_layout_with_tag((bn, dim), fp8, tla.arch.zN))
    v1 = tla.make_tensor(v1p, tla.make_layout_with_tag((dv, bn), fp8, tla.arch.nZ))
    p1 = tla.make_tensor(p1p, tla.make_layout_with_tag((bn, bm), fp8, tla.arch.zN))
    qs1 = tla.make_tensor(
        qs1p, tla.make_layout_with_tag((dim // 32, bm), e8m0, tla.arch.nNMxScale)
    )
    ks1 = tla.make_tensor(
        ks1p, tla.make_layout_with_tag((bn, dim // 32), e8m0, tla.arch.zZMxScale)
    )
    vs1 = tla.make_tensor(
        vs1p, tla.make_layout_with_tag((dv, bn // 32), e8m0, tla.arch.zZMxScale)
    )
    ps1 = tla.make_tensor(
        ps1p, tla.make_layout_with_tag((bn // 32, bm), e8m0, tla.arch.nNMxScale)
    )
    q0 = tla.make_tensor(q0p, tla.make_layout_with_tag((dim, bm), fp8, tla.arch.nZ))
    k0 = tla.make_tensor(k0p, tla.make_layout_with_tag((bn, dim), fp8, tla.arch.zN))
    p0 = tla.make_tensor(p0p, tla.make_layout_with_tag((bn, bm), fp8, tla.arch.nZ))
    v0 = tla.make_tensor(v0p, tla.make_layout_with_tag((pv_m, bn), fp8, tla.arch.zN))
    s0 = tla.make_tensor(
        s0p, tla.make_layout_with_tag((bn, bm), tla.Float32, tla.arch.L0Clayout)
    )
    o0 = tla.make_tensor(
        o0p, tla.make_layout_with_tag((pv_m, bm), tla.Float32, tla.arch.L0Clayout)
    )
    sfix = tla.make_tensor(
        sp, tla.make_layout(tla.make_shape(bn, bm), tla.make_stride(half, 1))
    )
    afix = tla.make_tensor(
        ap, tla.make_layout(tla.make_shape(pv_m, bm), tla.make_stride(half, 1))
    )
    ofix = tla.make_tensor(
        op, tla.make_layout(tla.make_shape(pv_m, bm), tla.make_stride(half, 1))
    )
    p16ref = tla.make_tensor(
        tla.recast_ptr(pp, dtype=tla.Float16),
        tla.make_layout(
            tla.make_shape(bn + 1, half // 2), tla.make_stride(half // 2, 1)
        ),
    )
    pu = tla.make_tensor_like(p16ref.ptr, p16ref, tla.arch.zNUnAlign)
    p1storage = tla.make_tensor(
        tla.recast_ptr(p1p, dtype=tla.Float16),
        tla.make_layout_with_tag((bn, bm // 2), tla.Float16, tla.arch.zN),
    )
    psuref = tla.make_tensor(
        tla.recast_ptr(psp, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(8, 16), tla.make_stride(16, 1)),
    )
    psu = tla.make_tensor_like(psuref.ptr, psuref, tla.arch.zNUnAlign)
    psflat = tla.make_tensor(
        tla.recast_ptr(ps1p, dtype=tla.Float16),
        tla.make_layout_with_tag((16, 16), tla.Float16, tla.arch.zN),
    )
    padp = tla.allocate(2048, tla.UInt8, tla.AddressSpace.ub, 512)
    pads = tla.allocate(64, tla.UInt8, tla.AddressSpace.ub, 256)
    padref = tla.make_tensor(
        tla.recast_ptr(padp, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(64, 16), tla.make_stride(16, 1)),
    )
    padu = tla.make_tensor_like(padref.ptr, padref, tla.arch.zNUnAlign)
    pad1 = tla.make_tensor(
        tla.recast_ptr(v1p + dim * bn, dtype=tla.Float16),
        tla.make_layout_with_tag((128, 16), tla.Float16, tla.arch.zN),
    )
    padsref = tla.make_tensor(
        tla.recast_ptr(pads, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(2, 16), tla.make_stride(16, 1)),
    )
    padsu = tla.make_tensor_like(padsref.ptr, padsref, tla.arch.zNUnAlign)
    pads1 = tla.make_tensor(
        tla.recast_ptr(vs1p + dim * 4, dtype=tla.Float16),
        tla.make_layout_with_tag((4, 16), tla.Float16, tla.arch.zN),
    )
    yu = tla.make_tensor(
        yp,
        tla.make_layout(tla.make_shape(half, dim), tla.make_stride(output_stride, 1)),
    )
    eu = tla.make_tensor(ep, tla.make_layout(tla.make_shape(half), tla.make_stride(1)))

    # In-core buffer ownership is expressed with mutexes.
    m_q1 = tla.mutex("q1", id=0)
    m_q0 = tla.mutex("q0", id=1)
    m_k1_0 = tla.mutex("k1_0", id=2)
    m_k1_1 = tla.mutex("k1_1", id=3)
    m_k0 = tla.mutex("k0", id=4)
    m_v1_0 = tla.mutex("v1_0", id=5)
    m_v1_1 = tla.mutex("v1_1", id=6)
    m_pv_0 = tla.mutex("pv_0", id=7)
    m_pv_1 = tla.mutex("pv_1", id=8)
    m_o0 = tla.mutex("o0", id=9)
    m_pad = tla.mutex("pad", id=10)
    m_pp_0 = tla.mutex("pp_0", id=11)
    m_pp_1 = tla.mutex("pp_1", id=12)
    m_ps = tla.mutex("ps", id=13)
    m_out = tla.mutex("out", id=14)
    sr = tla.cross_flag("score_ready")
    sr_second = tla.cross_flag("score_ready_second")
    ore = tla.cross_flag("output_ready")
    psready = tla.cross_flag("scales_ready")
    pfree = tla.cross_flag("prob_consumed")
    padready = tla.cross_flag("pad_ready")
    split_o = tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N)
    with tla.vector():
        m_pp_0.lock(pipe=tla.pipes.VECTOR)
        m_pp_1.lock(pipe=tla.pipes.VECTOR)
        mx_zero_ub(pp, half * (bn + 1))
        mx_zero_ub(pp + pp_stride, half * (bn + 1))
        m_pp_1.unlock(pipe=tla.pipes.VECTOR)
        m_pp_0.unlock(pipe=tla.pipes.VECTOR)
        m_pad.lock(pipe=tla.pipes.VECTOR)
        mx_pad(padp, pads)
        m_pad.unlock(pipe=tla.pipes.VECTOR)
        m_pp_0.lock(pipe=tla.pipes.MTE3)
        m_pp_1.lock(pipe=tla.pipes.MTE3)
        m_pad.lock(pipe=tla.pipes.MTE3)
        if tla.arch.sub_block_idx() == 0:
            mx_zero_scale_l1(ks1p, pp, 2 * bn * (dim // 32))
            mx_zero_scale_l1(vs1p, pp, vring * dv * 4)
            for seed_slot in tla.range(vring):
                seed_v1 = tla.make_tensor_like(
                    tla.recast_ptr(
                        v1p + seed_slot * bn * dv + dim * bn, dtype=tla.Float16
                    ),
                    pad1,
                    tla.arch.zN,
                )
                seed_vs1 = tla.make_tensor_like(
                    tla.recast_ptr(
                        vs1p + seed_slot * dv * 4 + dim * 4, dtype=tla.Float16
                    ),
                    pads1,
                    tla.arch.zN,
                )
                for seed_sub in tla.range(2):
                    tla.copy(
                        tla.tile_view(
                            seed_v1, tla.make_shape(64, 16), tla.make_coord(seed_sub, 0)
                        ),
                        padu,
                    )
                    tla.copy(
                        tla.tile_view(
                            seed_vs1, tla.make_shape(2, 16), tla.make_coord(seed_sub, 0)
                        ),
                        padsu,
                    )
        m_pad.unlock(pipe=tla.pipes.MTE3)
        m_pp_1.unlock(pipe=tla.pipes.MTE3)
        m_pp_0.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(padready, tla.pipes.MTE3)
        tla.cross_core_set_flag(sr, tla.pipes.VECTOR)
        tla.cross_core_set_flag(sr_second, tla.pipes.VECTOR)
        tla.cross_core_set_flag(ore, tla.pipes.VECTOR)
    with tla.cube():
        tla.cross_core_wait_flag(padready, tla.pipes.MTE2)
    blocks = (nq + 127) // 128
    total = batch * hq * blocks
    core = tla.arch.block_idx()
    cores = tla.arch.block_num()
    task_count = (total - core + cores - 1) // cores
    total_tiles = task_count * selected
    sub = tla.arch.sub_block_idx()
    qordered = tla.as_numeric(1)
    pordered = tla.as_numeric(1)
    qlast = tla.as_numeric(-1)
    vlast = tla.as_numeric(-1)
    qbase = tla.as_numeric(0)
    kbase = tla.as_numeric(0)
    maskbase = tla.as_numeric(0)
    pqbase = tla.as_numeric(0)
    pkbase = tla.as_numeric(0)
    vb = tla.as_numeric(0)
    pmaskbase = tla.as_numeric(0)
    rows = 128 if nq % 128 == 0 else tla.as_numeric(128)
    valid = 64 if nq % 128 == 0 else tla.as_numeric(64)
    for step in tla.range(total_tiles + delay):
        pi = step - delay
        qtile = step % selected
        ptile = pi % selected
        if step < total_tiles and qtile == 0:
            task = core + (step // selected) * cores
            h = (task // blocks) % hq
            qb = task % blocks
            b = task // (hq * blocks)
            kh = h // (hq // hk)
            qstart = qb * 128
            if nq % 128 != 0:
                rows = 128 if nq - qstart >= 128 else nq - qstart
            qbase = ((b * hq + h) * nq + qstart) * dim
            kbase = (b * hk + kh) * nk * dim
            mask_head = 0 if mh == 1 else h
            maskbase = ((b * mh + mask_head) * blocks + qb) * selected
        if pi >= 0 and ptile == 0:
            ptask = core + (pi // selected) * cores
            ph = (ptask // blocks) % hq
            pqb = ptask % blocks
            pb = ptask // (hq * blocks)
            pkh = ph // (hq // hk)
            pqstart = pqb * 128
            if nq % 128 != 0:
                prows = 128 if nq - pqstart >= 128 else nq - pqstart
                valid = prows - sub * half
                valid = half if valid > half else valid
                valid = 0 if valid < 0 else valid
            pqbase = ((pb * hq + ph) * nq + pqstart) * dim
            pkbase = (pb * hk + pkh) * nk * dim
            vb = (pb * hk + pkh) * ((nk + 63) // 64) * dim * 2
            pmask_head = 0 if mh == 1 else ph
            pmaskbase = ((pb * mh + pmask_head) * blocks + pqb) * selected
        qmp = mp + ((step // selected) % 2) * half
        pmp = mp + ((pi // selected) % 2) * half
        if step < total_tiles:
            if qtile == 0:
                qordered = tla.as_numeric(1)
                qlast = tla.as_numeric(-1)
                qprevious = tla.as_numeric(-1)
                for qscan in tla.range(selected):
                    qvalue = indices[maskbase + qscan].to(tla.Int32)
                    if qvalue >= 0:
                        if qvalue < qprevious:
                            qordered = tla.as_numeric(0)
                        qprevious = qvalue
                with tla.cube():
                    m_q0.lock(pipe=tla.pipes.MTE1)
                    m_q1.lock(pipe=tla.pipes.MTE2)
                    qgm = tla.make_tensor(
                        q.ptr + qbase,
                        tla.make_layout(
                            tla.make_shape(dim, rows),
                            tla.make_stride(1, dim),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    tla.copy(
                        tla.tile_view(
                            q1, tla.make_shape(dim, rows), tla.make_coord(0, 0)
                        ),
                        qgm,
                    )
                    qsgm = tla.get_tile(
                        qs,
                        tla.make_coord(0, qbase // dim),
                        tla.make_shape(dim // 32, rows),
                    )
                    tla.copy(
                        tla.tile_view(
                            qs1, tla.make_shape(dim // 32, rows), tla.make_coord(0, 0)
                        ),
                        qsgm,
                    )
                    m_q1.unlock(pipe=tla.pipes.MTE2)
                    m_q1.lock(pipe=tla.pipes.MTE1)
                    tla.copy(q0, q1, scale=qs1)
                    m_q1.unlock(pipe=tla.pipes.MTE1)
                    m_q0.unlock(pipe=tla.pipes.MTE1)
                    m_q0.lock(pipe=tla.pipes.CUBE)
                with tla.vector():
                    _mx_group_scales_bounded(psp, qmp, lp, rp, gp, -2)
        if pi >= 0:
            if ptile == 0:
                pordered = tla.as_numeric(1)
                vlast = tla.as_numeric(-1)
                pprevious = tla.as_numeric(-1)
                for pscan in tla.range(selected):
                    pvalue = indices[pmaskbase + pscan].to(tla.Int32)
                    if pvalue >= 0:
                        if pvalue < pprevious:
                            pordered = tla.as_numeric(0)
                        pprevious = pvalue
        qcols = tla.as_numeric(0)
        qstart_k = tla.as_numeric(0)
        vcols = tla.as_numeric(0)
        vstart_k = tla.as_numeric(0)
        if step < total_tiles:
            qblock = indices[maskbase + qtile].to(tla.Int32)
            if qordered == 0:
                qblock = (nk + bn - 1) // bn
                for qi_scan in tla.range(selected):
                    qcandidate = indices[maskbase + qi_scan].to(tla.Int32)
                    if qcandidate > qlast:
                        if qcandidate < qblock:
                            qblock = qcandidate
            if qblock > qlast:
                qlast = qblock
                if qblock * bn < nk:
                    qstart_k = qblock * bn
                    qcols = bn if nk - qstart_k >= bn else nk - qstart_k
        if pi >= 0:
            pblock = indices[pmaskbase + ptile].to(tla.Int32)
            if pordered == 0:
                pblock = (nk + bn - 1) // bn
                for vi_scan in tla.range(selected):
                    vcandidate = indices[pmaskbase + vi_scan].to(tla.Int32)
                    if vcandidate > vlast:
                        if vcandidate < pblock:
                            pblock = vcandidate
            if pblock > vlast:
                vlast = pblock
                if pblock * bn < nk:
                    vstart_k = pblock * bn
                    vcols = bn if nk - vstart_k >= bn else nk - vstart_k
        with tla.cube():
            if step < total_tiles:
                if step % 2 == 0:
                    m_k1_0.lock(pipe=tla.pipes.MTE2)
                else:
                    m_k1_1.lock(pipe=tla.pipes.MTE2)
                current_k1 = tla.make_tensor_like(
                    k1p + (step % 2) * bn * dim, k1, tla.arch.zN
                )
                current_ks1 = tla.make_tensor_like(
                    ks1p + (step % 2) * bn * (dim // 32), ks1, tla.arch.zZMxScale
                )
                if qcols < bn:
                    current_k1.fill(0)
                if qcols > 0:
                    k_copy_cols = bn if nk % bn == 0 else qcols
                    kgm = tla.make_tensor(
                        k.ptr + kbase + qstart_k * dim,
                        tla.make_layout(
                            tla.make_shape(k_copy_cols, dim), tla.make_stride(dim, 1)
                        ),
                    )
                    ksgm = tla.get_tile(
                        ks,
                        tla.make_coord(kbase // dim + qstart_k, 0),
                        tla.make_shape(k_copy_cols, dim // 32),
                    )
                    tla.copy(
                        tla.tile_view(
                            current_k1,
                            tla.make_shape(k_copy_cols, dim),
                            tla.make_coord(0, 0),
                        ),
                        kgm,
                    )
                    tla.copy(
                        tla.tile_view(
                            current_ks1,
                            tla.make_shape(k_copy_cols, dim // 32),
                            tla.make_coord(0, 0),
                        ),
                        ksgm,
                    )
                if step % 2 == 0:
                    m_k1_0.unlock(pipe=tla.pipes.MTE2)
                else:
                    m_k1_1.unlock(pipe=tla.pipes.MTE2)
            if pi >= 0:
                if pi % 2 == 0:
                    m_v1_0.lock(pipe=tla.pipes.MTE2)
                else:
                    m_v1_1.lock(pipe=tla.pipes.MTE2)
                preload_vlong = tla.make_tensor_like(
                    v1p + (pi % 2) * bn * dv, v1, tla.arch.nZ
                )
                preload_vslong = tla.make_tensor_like(
                    vs1p + (pi % 2) * dv * 4, vs1, tla.arch.zZMxScale
                )
                if vcols < bn:
                    tla.get_tile(
                        preload_vlong, tla.make_coord(0, 0), tla.make_shape(dim, bn)
                    ).fill(0)
                if vcols > 0:
                    v_copy_cols = bn if nk % bn == 0 else vcols
                    long_vgm = tla.make_tensor(
                        v.ptr + pkbase + vstart_k * dim,
                        tla.make_layout(
                            tla.make_shape(dim, v_copy_cols),
                            tla.make_stride(1, dim),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    long_groups = ((v_copy_cols + 63) // 64) * 2
                    long_vsgm = tla.get_tile(
                        vs,
                        tla.make_coord(0, vb // dim + vstart_k // 32),
                        tla.make_shape(dim, long_groups),
                    )
                    tla.copy(
                        tla.tile_view(
                            preload_vlong,
                            tla.make_shape(dim, v_copy_cols),
                            tla.make_coord(0, 0),
                        ),
                        long_vgm,
                    )
                    tla.copy(
                        tla.tile_view(
                            preload_vslong,
                            tla.make_shape(dim, long_groups),
                            tla.make_coord(0, 0),
                        ),
                        long_vsgm,
                    )
                if pi % 2 == 0:
                    m_v1_0.unlock(pipe=tla.pipes.MTE2)
                else:
                    m_v1_1.unlock(pipe=tla.pipes.MTE2)
            if step < total_tiles:
                if step % 2 == 0:
                    m_k1_0.lock(pipe=tla.pipes.MTE1)
                else:
                    m_k1_1.lock(pipe=tla.pipes.MTE1)
                m_k0.lock(pipe=tla.pipes.MTE1)
                current_k0 = tla.make_tensor_like(k0p, k0, tla.arch.zN)
                read_k1 = tla.make_tensor_like(
                    k1p + (step % 2) * bn * dim, k1, tla.arch.zN
                )
                read_ks1 = tla.make_tensor_like(
                    ks1p + (step % 2) * bn * (dim // 32), ks1, tla.arch.zZMxScale
                )
                tla.copy(current_k0, read_k1, scale=read_ks1)
                if step % 2 == 0:
                    m_k1_0.unlock(pipe=tla.pipes.MTE1)
                else:
                    m_k1_1.unlock(pipe=tla.pipes.MTE1)
                m_k0.unlock(pipe=tla.pipes.MTE1)
                m_k0.lock(pipe=tla.pipes.CUBE)
                current_s0 = tla.make_tensor_like(
                    s0p + (step % 2) * bm * bn, s0, tla.arch.L0Clayout
                )
                # Matched with unitFlag=3 in mx_score_unit; synchronization
                # is per L0C unit instead of a whole-tile CUBE/FIX event.
                tla.mmad_mx(current_s0, current_k0, q0, init_c=True, unit_flag=3)
                m_k0.unlock(pipe=tla.pipes.CUBE)
                if qtile == selected - 1:
                    m_q0.unlock(pipe=tla.pipes.CUBE)
            if pi >= 0:
                if pi % 2 == 0:
                    m_v1_0.lock(pipe=tla.pipes.MTE1)
                else:
                    m_v1_1.lock(pipe=tla.pipes.MTE1)
                if pi % 2 == 0:
                    m_pv_0.lock(pipe=tla.pipes.MTE1)
                else:
                    m_pv_1.lock(pipe=tla.pipes.MTE1)
                pv_v0 = tla.make_tensor_like(v0p + (pi % 2) * bn * dv, v0, tla.arch.zN)
                pv_p0 = tla.make_tensor_like(p0p + (pi % 2) * bm * bn, p0, tla.arch.nZ)
                read_v1 = tla.make_tensor_like(
                    v1p + (pi % vring) * bn * dv, v1, tla.arch.nZ
                )
                read_vs1 = tla.make_tensor_like(
                    vs1p + (pi % vring) * dv * 4, vs1, tla.arch.zZMxScale
                )
                read_p1 = tla.make_tensor_like(
                    p1p + (pi % ring) * bm * bn, p1, tla.arch.zN
                )
                read_ps1 = tla.make_tensor_like(
                    ps1p + (pi % ring) * bm * 4, ps1, tla.arch.nNMxScale
                )
                # V does not depend on P scales. Release its L1 slot immediately
                # after MTE1 has read it, before waiting for the group scales.
                tla.copy(
                    pv_v0,
                    tla.get_tile(
                        read_v1, tla.make_coord(0, 0), tla.make_shape(pv_m, bn)
                    ),
                    scale=tla.get_tile(
                        read_vs1, tla.make_coord(0, 0), tla.make_shape(pv_m, bn // 32)
                    ),
                )
                if pi % 2 == 0:
                    m_v1_0.unlock(pipe=tla.pipes.MTE1)
                else:
                    m_v1_1.unlock(pipe=tla.pipes.MTE1)
                if ptile % group == 0:
                    tla.cross_core_wait_flag(psready, tla.pipes.MTE1)
                tla.copy(pv_p0, read_p1, scale=read_ps1)
                tla.cross_core_set_flag(pfree, tla.pipes.MTE1)
                if pi % 2 == 0:
                    m_pv_0.unlock(pipe=tla.pipes.MTE1)
                    m_pv_0.lock(pipe=tla.pipes.CUBE)
                else:
                    m_pv_1.unlock(pipe=tla.pipes.MTE1)
                    m_pv_1.lock(pipe=tla.pipes.CUBE)
                if ptile == 0:
                    m_o0.lock(pipe=tla.pipes.CUBE)
                tla.mmad_mx(o0, pv_v0, pv_p0, init_c=ptile == 0)
                if pi % 2 == 0:
                    m_pv_0.unlock(pipe=tla.pipes.CUBE)
                else:
                    m_pv_1.unlock(pipe=tla.pipes.CUBE)
                if ptile == selected - 1:
                    m_o0.unlock(pipe=tla.pipes.CUBE)
            if step < total_tiles:
                if step % 2 == 0:
                    tla.cross_core_wait_flag(sr, tla.pipes.FIX)
                else:
                    tla.cross_core_wait_flag(sr_second, tla.pipes.FIX)
                mx_score_unit(
                    sp + (step % 2) * 2 * half * bn, s0p + (step % 2) * bm * bn, scale
                )
                if step % 2 == 0:
                    tla.cross_core_set_flag(sr, tla.pipes.FIX)
                else:
                    tla.cross_core_set_flag(sr_second, tla.pipes.FIX)
            if pi >= 0:
                if ptile == selected - 1:
                    m_o0.lock(pipe=tla.pipes.FIX)
                    tla.cross_core_wait_flag(ore, tla.pipes.FIX)
                    tla.copy(afix, o0, split_o)
                    tla.cross_core_set_flag(ore, tla.pipes.FIX)
                    m_o0.unlock(pipe=tla.pipes.FIX)
        with tla.vector():
            if step < total_tiles:
                if step % 2 == 0:
                    tla.cross_core_wait_flag(sr, tla.pipes.VECTOR)
                else:
                    tla.cross_core_wait_flag(sr_second, tla.pipes.VECTOR)
                if step % 2 == 0:
                    m_pp_0.lock(pipe=tla.pipes.VECTOR)
                else:
                    m_pp_1.lock(pipe=tla.pipes.VECTOR)
                if qtile % group == 0:
                    _mx_group_scales_bounded(psp, qmp, lp, rp, gp, -1)
                _mx_group_softmax_store_fast(
                    sp + (step % 2) * 2 * half * bn,
                    pp + (step % 2) * pp_stride,
                    psp,
                    qmp,
                    lp,
                    tmp_alpha,
                    gp + (qtile % group) * 256,
                    scale,
                    qcols,
                )
                if step % 2 == 0:
                    tla.cross_core_set_flag(sr, tla.pipes.VECTOR)
                else:
                    tla.cross_core_set_flag(sr_second, tla.pipes.VECTOR)
            if pi >= 0:
                tla.cross_core_wait_flag(pfree, tla.pipes.MTE3)
            if step < total_tiles:
                if step % 2 == 0:
                    m_pp_0.unlock(pipe=tla.pipes.VECTOR)
                    m_pp_0.lock(pipe=tla.pipes.MTE3)
                else:
                    m_pp_1.unlock(pipe=tla.pipes.VECTOR)
                    m_pp_1.lock(pipe=tla.pipes.MTE3)
                write_p = tla.make_tensor_like(
                    tla.recast_ptr(p1p + (step % ring) * bm * bn, dtype=tla.Float16),
                    p1storage,
                    tla.arch.zN,
                )
                puc = tla.make_tensor_like(
                    tla.recast_ptr(pp + (step % 2) * pp_stride, dtype=tla.Float16),
                    pu,
                    tla.arch.zNUnAlign,
                )
                tla.copy(
                    tla.tile_view(
                        write_p,
                        tla.make_shape(bn, half // 2),
                        tla.make_coord(0, sub),
                    ),
                    tla.tile_view(
                        puc, tla.make_shape(bn, half // 2), tla.make_coord(0, 0)
                    ),
                )
                if step % 2 == 0:
                    m_pp_0.unlock(pipe=tla.pipes.MTE3)
                else:
                    m_pp_1.unlock(pipe=tla.pipes.MTE3)
                if qtile % group == group - 1 or qtile == selected - 1:
                    group_count = qtile % group + 1
                    group_start = step - qtile % group
                    m_ps.lock(pipe=tla.pipes.VECTOR)
                    _mx_group_scales_bounded(
                        psp,
                        qmp,
                        lp,
                        rp
                        + (
                            (
                                (step // selected) * ((selected + group - 1) // group)
                                + qtile // group
                            )
                            % 2
                        )
                        * half,
                        gp,
                        group_count,
                    )
                    m_ps.unlock(pipe=tla.pipes.VECTOR)
                    m_ps.lock(pipe=tla.pipes.MTE3)
                    mx_ps64_to_l1(ps1p, psp, group_start % ring, group_count, ring, sub)
                    m_ps.unlock(pipe=tla.pipes.MTE3)
                    tla.cross_core_set_flag(psready, tla.pipes.MTE3)
            if pi >= 0:
                if ptile == selected - 1:
                    tla.cross_core_wait_flag(ore, tla.pipes.VECTOR)
                    if ptile == selected - 1:
                        m_out.lock(pipe=tla.pipes.VECTOR)
                        mx_epilogue(
                            ap,
                            op,
                            pmp,
                            lp,
                            rp,
                            tla.recast_ptr(yp, dtype=tla.UInt8),
                            ep,
                            dim,
                            valid,
                            2,
                            od_code,
                        )
                        m_out.unlock(pipe=tla.pipes.VECTOR)
                        if valid > 0:
                            m_out.lock(pipe=tla.pipes.MTE3)
                            ogm = tla.make_tensor(
                                out.ptr + pqbase + sub * half * dim,
                                tla.make_layout(
                                    tla.make_shape(valid, dim), tla.make_stride(dim, 1)
                                ),
                            )
                            tla.copy(
                                ogm,
                                tla.tile_view(
                                    yu, tla.make_shape(valid, dim), tla.make_coord(0, 0)
                                ),
                            )
                            if want_lse:
                                egm = tla.make_tensor(
                                    lse.ptr + pqbase // dim + sub * half,
                                    tla.make_layout(
                                        tla.make_shape(valid), tla.make_stride(1)
                                    ),
                                )
                                tla.copy(
                                    egm,
                                    tla.tile_view(
                                        eu, tla.make_shape(valid), tla.make_coord(0)
                                    ),
                                )
                            m_out.unlock(pipe=tla.pipes.MTE3)
                    tla.cross_core_set_flag(ore, tla.pipes.VECTOR)
    with tla.cube():
        tla.cross_core_wait_flag(sr, tla.pipes.FIX)
        tla.cross_core_wait_flag(sr_second, tla.pipes.FIX)
        tla.cross_core_wait_flag(ore, tla.pipes.FIX)


# Attention kernel: pipeline_continuous.
@tla.kernel
def quant_block_sparse_attention_continuous_kernel(
    q: tla.Tensor,
    k: tla.Tensor,
    v: tla.Tensor,
    out: tla.Tensor,
    qs: tla.Tensor,
    ks: tla.Tensor,
    vs: tla.Tensor,
    indices: tla.Tensor,
    lse: tla.Tensor,
    batch: tla.Constexpr[int],
    hq: tla.Constexpr[int],
    hk: tla.Constexpr[int],
    nq: tla.Constexpr[int],
    nk: tla.Constexpr[int],
    mh: tla.Constexpr[int],
    selected: tla.Constexpr[int],
    scale: tla.Constexpr[float],
    dim: tla.Constexpr[int],
    xb: tla.Constexpr[int],
    yb: tla.Constexpr[int],
    want_lse: tla.Constexpr[bool],
):
    # Transposed QK/PV: each AIV owns 64 query lanes. PV adds a
    # constant-one channel for the quantized probability denominator.
    group = min(16, selected)
    ring = min(group + 2, max(1, selected))
    delay = ring
    vring = 2
    bm = 128
    bn = 128
    dv = dim + 32
    pv_m = dim + 16
    half = 64
    qm = 64 if xb % 128 != 0 else 128
    fp8 = tla.Float8E4M3FN
    e8m0 = tla.Float8E8M0
    od = out.ptr.dtype
    output_stride = dim + (8 if od == tla.Float32 else 16)
    od_code = 0 if od == tla.Float32 else (1 if od == tla.Float16 else 2)
    q1p = tla.allocate(bm * dim, fp8, tla.AddressSpace.l1, 512)
    k1p = tla.allocate(2 * bn * dim, fp8, tla.AddressSpace.l1, 512)
    v1p = tla.allocate(vring * bn * dv, fp8, tla.AddressSpace.l1, 512)
    # Two P slots let softmax(t) overlap PV(t-1).
    p1p = tla.allocate(ring * bm * bn, fp8, tla.AddressSpace.l1, 512)
    qs1p = tla.allocate(bm * (dim // 32), e8m0, tla.AddressSpace.l1, 512)
    ks1p = tla.allocate(2 * bn * (dim // 32), e8m0, tla.AddressSpace.l1, 512)
    vs1p = tla.allocate(vring * dv * (bn // 32), e8m0, tla.AddressSpace.l1, 512)
    ps1p = tla.allocate(ring * bm * (bn // 32), e8m0, tla.AddressSpace.l1, 512)
    q0p = tla.allocate(bm * dim, fp8, tla.AddressSpace.l0b, 1024)
    k0p = tla.allocate(bn * dim, fp8, tla.AddressSpace.l0a, 1024)
    p0p = tla.allocate(2 * bm * bn, fp8, tla.AddressSpace.l0b, 1024)
    v0p = tla.allocate(2 * bn * dv, fp8, tla.AddressSpace.l0a, 1024)
    s0p = tla.allocate(2 * bm * bn, tla.Float32, tla.AddressSpace.l0c, 1024)
    o0p = tla.allocate(bm * dv, tla.Float32, tla.AddressSpace.l0c, 1024)
    # Separate score loads and probability stores into different 32 KiB UB
    # regions. The second half is spacing, not an additional score tile.
    sp = tla.allocate(4 * half * bn, tla.Float16, tla.AddressSpace.ub, 512)
    pp = tla.allocate(half * (bn + 1), tla.UInt8, tla.AddressSpace.ub, 512)
    psp = tla.allocate(16 * half * (bn // 32), tla.UInt8, tla.AddressSpace.ub, 256)
    mp = tla.allocate(2 * half, tla.Float32, tla.AddressSpace.ub, 256)
    lp = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)
    rp = tla.allocate(2 * half, tla.Float32, tla.AddressSpace.ub, 256)
    gp = tla.allocate(16 * half * (bn // 32), tla.Float32, tla.AddressSpace.ub, 256)
    op = tla.allocate(half * pv_m, tla.Float32, tla.AddressSpace.ub, 512)
    ap = tla.allocate(half * pv_m, tla.Float32, tla.AddressSpace.ub, 512)
    yp = tla.allocate(half * output_stride, od, tla.AddressSpace.ub, 512)
    tmp_alpha = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)
    ep = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)

    q1 = tla.make_tensor(q1p, tla.make_layout_with_tag((dim, bm), fp8, tla.arch.nZ))
    k1 = tla.make_tensor(k1p, tla.make_layout_with_tag((bn, dim), fp8, tla.arch.zN))
    v1 = tla.make_tensor(v1p, tla.make_layout_with_tag((dv, bn), fp8, tla.arch.nZ))
    p1 = tla.make_tensor(p1p, tla.make_layout_with_tag((bn, bm), fp8, tla.arch.zN))
    qs1 = tla.make_tensor(
        qs1p, tla.make_layout_with_tag((dim // 32, bm), e8m0, tla.arch.nNMxScale)
    )
    ks1 = tla.make_tensor(
        ks1p, tla.make_layout_with_tag((bn, dim // 32), e8m0, tla.arch.zZMxScale)
    )
    vs1 = tla.make_tensor(
        vs1p, tla.make_layout_with_tag((dv, bn // 32), e8m0, tla.arch.zZMxScale)
    )
    ps1 = tla.make_tensor(
        ps1p, tla.make_layout_with_tag((bn // 32, bm), e8m0, tla.arch.nNMxScale)
    )
    q0 = tla.make_tensor(q0p, tla.make_layout_with_tag((dim, bm), fp8, tla.arch.nZ))
    k0 = tla.make_tensor(k0p, tla.make_layout_with_tag((bn, dim), fp8, tla.arch.zN))
    p0 = tla.make_tensor(p0p, tla.make_layout_with_tag((bn, bm), fp8, tla.arch.nZ))
    v0 = tla.make_tensor(v0p, tla.make_layout_with_tag((pv_m, bn), fp8, tla.arch.zN))
    s0 = tla.make_tensor(
        s0p, tla.make_layout_with_tag((bn, bm), tla.Float32, tla.arch.L0Clayout)
    )
    o0 = tla.make_tensor(
        o0p, tla.make_layout_with_tag((pv_m, bm), tla.Float32, tla.arch.L0Clayout)
    )
    sfix = tla.make_tensor(
        sp, tla.make_layout(tla.make_shape(bn, bm), tla.make_stride(half, 1))
    )
    afix = tla.make_tensor(
        ap, tla.make_layout(tla.make_shape(pv_m, bm), tla.make_stride(half, 1))
    )
    ofix = tla.make_tensor(
        op, tla.make_layout(tla.make_shape(pv_m, bm), tla.make_stride(half, 1))
    )
    p16ref = tla.make_tensor(
        tla.recast_ptr(pp, dtype=tla.Float16),
        tla.make_layout(
            tla.make_shape(bn + 1, half // 2), tla.make_stride(half // 2, 1)
        ),
    )
    pu = tla.make_tensor_like(p16ref.ptr, p16ref, tla.arch.zNUnAlign)
    p1storage = tla.make_tensor(
        tla.recast_ptr(p1p, dtype=tla.Float16),
        tla.make_layout_with_tag((bn, bm // 2), tla.Float16, tla.arch.zN),
    )
    psuref = tla.make_tensor(
        tla.recast_ptr(psp, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(8, 16), tla.make_stride(16, 1)),
    )
    psu = tla.make_tensor_like(psuref.ptr, psuref, tla.arch.zNUnAlign)
    psflat = tla.make_tensor(
        tla.recast_ptr(ps1p, dtype=tla.Float16),
        tla.make_layout_with_tag((16, 16), tla.Float16, tla.arch.zN),
    )
    padp = tla.allocate(2048, tla.UInt8, tla.AddressSpace.ub, 512)
    pads = tla.allocate(64, tla.UInt8, tla.AddressSpace.ub, 256)
    padref = tla.make_tensor(
        tla.recast_ptr(padp, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(64, 16), tla.make_stride(16, 1)),
    )
    padu = tla.make_tensor_like(padref.ptr, padref, tla.arch.zNUnAlign)
    pad1 = tla.make_tensor(
        tla.recast_ptr(v1p + dim * bn, dtype=tla.Float16),
        tla.make_layout_with_tag((128, 16), tla.Float16, tla.arch.zN),
    )
    padsref = tla.make_tensor(
        tla.recast_ptr(pads, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(2, 16), tla.make_stride(16, 1)),
    )
    padsu = tla.make_tensor_like(padsref.ptr, padsref, tla.arch.zNUnAlign)
    pads1 = tla.make_tensor(
        tla.recast_ptr(vs1p + dim * 4, dtype=tla.Float16),
        tla.make_layout_with_tag((4, 16), tla.Float16, tla.arch.zN),
    )
    yu = tla.make_tensor(
        yp,
        tla.make_layout(tla.make_shape(half, dim), tla.make_stride(output_stride, 1)),
    )
    eu = tla.make_tensor(ep, tla.make_layout(tla.make_shape(half), tla.make_stride(1)))

    # In-core buffer ownership is expressed with mutexes.
    m_q1 = tla.mutex("q1", id=0)
    m_q0 = tla.mutex("q0", id=1)
    m_k1_0 = tla.mutex("k1_0", id=2)
    m_k1_1 = tla.mutex("k1_1", id=3)
    m_k0 = tla.mutex("k0", id=4)
    m_s0_0 = tla.mutex("s0_0", id=5)
    m_s0_1 = tla.mutex("s0_1", id=6)
    m_v1_0 = tla.mutex("v1_0", id=7)
    m_v1_1 = tla.mutex("v1_1", id=8)
    m_pv_0 = tla.mutex("pv_0", id=9)
    m_pv_1 = tla.mutex("pv_1", id=10)
    m_o0 = tla.mutex("o0", id=11)
    m_pad = tla.mutex("pad", id=12)
    m_pp = tla.mutex("pp", id=13)
    m_ps = tla.mutex("ps", id=14)
    m_out = tla.mutex("out", id=15)
    sr = tla.cross_flag("score_ready")
    sr_second = tla.cross_flag("score_ready_second")
    ore = tla.cross_flag("output_ready")
    psready = tla.cross_flag("scales_ready")
    pfree = tla.cross_flag("prob_consumed")
    padready = tla.cross_flag("pad_ready")
    split_o = tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N)
    with tla.vector():
        m_pp.lock(pipe=tla.pipes.VECTOR)
        mx_zero_ub(pp, half * (bn + 1))
        m_pp.unlock(pipe=tla.pipes.VECTOR)
        m_pad.lock(pipe=tla.pipes.VECTOR)
        mx_pad(padp, pads)
        m_pad.unlock(pipe=tla.pipes.VECTOR)
        m_pad.lock(pipe=tla.pipes.MTE3)
        m_pp.lock(pipe=tla.pipes.MTE3)
        if tla.arch.sub_block_idx() == 0:
            mx_zero_scale_l1(ks1p, pp, 2 * bn * (dim // 32))
            mx_zero_scale_l1(vs1p, pp, vring * dv * 4)
            for seed_slot in tla.range(vring):
                seed_v1 = tla.make_tensor_like(
                    tla.recast_ptr(
                        v1p + seed_slot * bn * dv + dim * bn, dtype=tla.Float16
                    ),
                    pad1,
                    tla.arch.zN,
                )
                seed_vs1 = tla.make_tensor_like(
                    tla.recast_ptr(
                        vs1p + seed_slot * dv * 4 + dim * 4, dtype=tla.Float16
                    ),
                    pads1,
                    tla.arch.zN,
                )
                for seed_sub in tla.range(2):
                    tla.copy(
                        tla.tile_view(
                            seed_v1, tla.make_shape(64, 16), tla.make_coord(seed_sub, 0)
                        ),
                        padu,
                    )
                    tla.copy(
                        tla.tile_view(
                            seed_vs1, tla.make_shape(2, 16), tla.make_coord(seed_sub, 0)
                        ),
                        padsu,
                    )
        m_pp.unlock(pipe=tla.pipes.MTE3)
        m_pad.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(padready, tla.pipes.MTE3)
        tla.cross_core_set_flag(sr, tla.pipes.VECTOR)
        tla.cross_core_set_flag(sr_second, tla.pipes.VECTOR)
        tla.cross_core_set_flag(ore, tla.pipes.VECTOR)
    with tla.cube():
        tla.cross_core_wait_flag(padready, tla.pipes.MTE2)
    blocks = (nq + 127) // 128
    total = batch * hq * blocks
    core = tla.arch.block_idx()
    cores = tla.arch.block_num()
    task_count = (total - core + cores - 1) // cores
    total_tiles = task_count * selected
    sub = tla.arch.sub_block_idx()
    qordered = tla.as_numeric(1)
    pordered = tla.as_numeric(1)
    qlast = tla.as_numeric(-1)
    vlast = tla.as_numeric(-1)
    for step in tla.range(total_tiles + delay):
        pi = step - delay
        qtile = step % selected
        ptile = pi % selected
        task = core + (step // selected) * cores
        ptask = core + (pi // selected) * cores
        h = (task // blocks) % hq
        qb = task % blocks
        b = task // (hq * blocks)
        kh = h // (hq // hk)
        qstart = qb * 128
        rows = 128 if nq % 128 == 0 else (128 if nq - qstart >= 128 else nq - qstart)
        qbase = ((b * hq + h) * nq + qstart) * dim
        kbase = (b * hk + kh) * nk * dim
        mask_head = 0 if mh == 1 else h
        maskbase = ((b * mh + mask_head) * blocks + qb) * selected
        ph = (ptask // blocks) % hq
        pqb = ptask % blocks
        pb = ptask // (hq * blocks)
        pkh = ph // (hq // hk)
        pqstart = pqb * 128
        prows = 128 if nq % 128 == 0 else (128 if nq - pqstart >= 128 else nq - pqstart)
        valid = prows - sub * half
        valid = half if valid > half else valid
        valid = 0 if valid < 0 else valid
        pqbase = ((pb * hq + ph) * nq + pqstart) * dim
        pkbase = (pb * hk + pkh) * nk * dim
        vb = (pb * hk + pkh) * ((nk + 63) // 64) * dim * 2
        pmask_head = 0 if mh == 1 else ph
        pmaskbase = ((pb * mh + pmask_head) * blocks + pqb) * selected
        qmp = mp + ((step // selected) % 2) * half
        pmp = mp + ((pi // selected) % 2) * half
        if step < total_tiles:
            if qtile == 0:
                qordered = tla.as_numeric(1)
                qlast = tla.as_numeric(-1)
                qprevious = tla.as_numeric(-1)
                for qscan in tla.range(selected):
                    qvalue = indices[maskbase + qscan].to(tla.Int32)
                    if qvalue >= 0:
                        if qvalue < qprevious:
                            qordered = tla.as_numeric(0)
                        qprevious = qvalue
                with tla.cube():
                    m_q0.lock(pipe=tla.pipes.MTE1)
                    m_q1.lock(pipe=tla.pipes.MTE2)
                    qgm = tla.make_tensor(
                        q.ptr + qbase,
                        tla.make_layout(
                            tla.make_shape(dim, rows),
                            tla.make_stride(1, dim),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    tla.copy(
                        tla.tile_view(
                            q1, tla.make_shape(dim, rows), tla.make_coord(0, 0)
                        ),
                        qgm,
                    )
                    qsgm = tla.get_tile(
                        qs,
                        tla.make_coord(0, qbase // dim),
                        tla.make_shape(dim // 32, rows),
                    )
                    tla.copy(
                        tla.tile_view(
                            qs1, tla.make_shape(dim // 32, rows), tla.make_coord(0, 0)
                        ),
                        qsgm,
                    )
                    m_q1.unlock(pipe=tla.pipes.MTE2)
                    m_q1.lock(pipe=tla.pipes.MTE1)
                    tla.copy(q0, q1, scale=qs1)
                    m_q1.unlock(pipe=tla.pipes.MTE1)
                    m_q0.unlock(pipe=tla.pipes.MTE1)
                    m_q0.lock(pipe=tla.pipes.CUBE)
                with tla.vector():
                    _mx_group_scales_basic(psp, qmp, lp, rp, gp, -2)
        if pi >= 0:
            if ptile == 0:
                pordered = tla.as_numeric(1)
                vlast = tla.as_numeric(-1)
                pprevious = tla.as_numeric(-1)
                for pscan in tla.range(selected):
                    pvalue = indices[pmaskbase + pscan].to(tla.Int32)
                    if pvalue >= 0:
                        if pvalue < pprevious:
                            pordered = tla.as_numeric(0)
                        pprevious = pvalue
        qcols = tla.as_numeric(0)
        qstart_k = tla.as_numeric(0)
        vcols = tla.as_numeric(0)
        vstart_k = tla.as_numeric(0)
        if step < total_tiles:
            qblock = indices[maskbase + qtile].to(tla.Int32)
            if qordered == 0:
                qblock = (nk + bn - 1) // bn
                for qi_scan in tla.range(selected):
                    qcandidate = indices[maskbase + qi_scan].to(tla.Int32)
                    if qcandidate > qlast:
                        if qcandidate < qblock:
                            qblock = qcandidate
            if qblock > qlast:
                qlast = qblock
                if qblock * bn < nk:
                    qstart_k = qblock * bn
                    qcols = bn if nk - qstart_k >= bn else nk - qstart_k
        if pi >= 0:
            pblock = indices[pmaskbase + ptile].to(tla.Int32)
            if pordered == 0:
                pblock = (nk + bn - 1) // bn
                for vi_scan in tla.range(selected):
                    vcandidate = indices[pmaskbase + vi_scan].to(tla.Int32)
                    if vcandidate > vlast:
                        if vcandidate < pblock:
                            pblock = vcandidate
            if pblock > vlast:
                vlast = pblock
                if pblock * bn < nk:
                    vstart_k = pblock * bn
                    vcols = bn if nk - vstart_k >= bn else nk - vstart_k
        with tla.cube():
            if step < total_tiles:
                if step % 2 == 0:
                    m_k1_0.lock(pipe=tla.pipes.MTE2)
                else:
                    m_k1_1.lock(pipe=tla.pipes.MTE2)
                current_k1 = tla.make_tensor_like(
                    k1p + (step % 2) * bn * dim, k1, tla.arch.zN
                )
                current_ks1 = tla.make_tensor_like(
                    ks1p + (step % 2) * bn * (dim // 32), ks1, tla.arch.zZMxScale
                )
                if qcols < bn:
                    current_k1.fill(0)
                if qcols > 0:
                    k_copy_cols = bn if nk % bn == 0 else qcols
                    kgm = tla.make_tensor(
                        k.ptr + kbase + qstart_k * dim,
                        tla.make_layout(
                            tla.make_shape(k_copy_cols, dim), tla.make_stride(dim, 1)
                        ),
                    )
                    ksgm = tla.get_tile(
                        ks,
                        tla.make_coord(kbase // dim + qstart_k, 0),
                        tla.make_shape(k_copy_cols, dim // 32),
                    )
                    tla.copy(
                        tla.tile_view(
                            current_k1,
                            tla.make_shape(k_copy_cols, dim),
                            tla.make_coord(0, 0),
                        ),
                        kgm,
                    )
                    tla.copy(
                        tla.tile_view(
                            current_ks1,
                            tla.make_shape(k_copy_cols, dim // 32),
                            tla.make_coord(0, 0),
                        ),
                        ksgm,
                    )
                if step % 2 == 0:
                    m_k1_0.unlock(pipe=tla.pipes.MTE2)
                else:
                    m_k1_1.unlock(pipe=tla.pipes.MTE2)
            if pi >= 0:
                if pi % 2 == 0:
                    m_v1_0.lock(pipe=tla.pipes.MTE2)
                else:
                    m_v1_1.lock(pipe=tla.pipes.MTE2)
                preload_vlong = tla.make_tensor_like(
                    v1p + (pi % 2) * bn * dv, v1, tla.arch.nZ
                )
                preload_vslong = tla.make_tensor_like(
                    vs1p + (pi % 2) * dv * 4, vs1, tla.arch.zZMxScale
                )
                if vcols < bn:
                    tla.get_tile(
                        preload_vlong, tla.make_coord(0, 0), tla.make_shape(dim, bn)
                    ).fill(0)
                if vcols > 0:
                    v_copy_cols = bn if nk % bn == 0 else vcols
                    long_vgm = tla.make_tensor(
                        v.ptr + pkbase + vstart_k * dim,
                        tla.make_layout(
                            tla.make_shape(dim, v_copy_cols),
                            tla.make_stride(1, dim),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    long_groups = ((v_copy_cols + 63) // 64) * 2
                    long_vsgm = tla.get_tile(
                        vs,
                        tla.make_coord(0, vb // dim + vstart_k // 32),
                        tla.make_shape(dim, long_groups),
                    )
                    tla.copy(
                        tla.tile_view(
                            preload_vlong,
                            tla.make_shape(dim, v_copy_cols),
                            tla.make_coord(0, 0),
                        ),
                        long_vgm,
                    )
                    tla.copy(
                        tla.tile_view(
                            preload_vslong,
                            tla.make_shape(dim, long_groups),
                            tla.make_coord(0, 0),
                        ),
                        long_vsgm,
                    )
                if pi % 2 == 0:
                    m_v1_0.unlock(pipe=tla.pipes.MTE2)
                else:
                    m_v1_1.unlock(pipe=tla.pipes.MTE2)
            if step < total_tiles:
                if step % 2 == 0:
                    m_k1_0.lock(pipe=tla.pipes.MTE1)
                else:
                    m_k1_1.lock(pipe=tla.pipes.MTE1)
                m_k0.lock(pipe=tla.pipes.MTE1)
                current_k0 = tla.make_tensor_like(k0p, k0, tla.arch.zN)
                read_k1 = tla.make_tensor_like(
                    k1p + (step % 2) * bn * dim, k1, tla.arch.zN
                )
                read_ks1 = tla.make_tensor_like(
                    ks1p + (step % 2) * bn * (dim // 32), ks1, tla.arch.zZMxScale
                )
                tla.copy(current_k0, read_k1, scale=read_ks1)
                if step % 2 == 0:
                    m_k1_0.unlock(pipe=tla.pipes.MTE1)
                else:
                    m_k1_1.unlock(pipe=tla.pipes.MTE1)
                m_k0.unlock(pipe=tla.pipes.MTE1)
                m_k0.lock(pipe=tla.pipes.CUBE)
                if step % 2 == 0:
                    m_s0_0.lock(pipe=tla.pipes.CUBE)
                else:
                    m_s0_1.lock(pipe=tla.pipes.CUBE)
                current_s0 = tla.make_tensor_like(
                    s0p + (step % 2) * bm * bn, s0, tla.arch.L0Clayout
                )
                tla.mmad_mx(current_s0, current_k0, q0, init_c=True)
                m_k0.unlock(pipe=tla.pipes.CUBE)
                if step % 2 == 0:
                    m_s0_0.unlock(pipe=tla.pipes.CUBE)
                else:
                    m_s0_1.unlock(pipe=tla.pipes.CUBE)
                if qtile == selected - 1:
                    m_q0.unlock(pipe=tla.pipes.CUBE)
            if pi >= 0:
                if pi >= 0:
                    if pi % 2 == 0:
                        m_v1_0.lock(pipe=tla.pipes.MTE1)
                    else:
                        m_v1_1.lock(pipe=tla.pipes.MTE1)
                if ptile % group == 0:
                    tla.cross_core_wait_flag(psready, tla.pipes.MTE1)
                if pi % 2 == 0:
                    m_pv_0.lock(pipe=tla.pipes.MTE1)
                else:
                    m_pv_1.lock(pipe=tla.pipes.MTE1)
                pv_v0 = tla.make_tensor_like(v0p + (pi % 2) * bn * dv, v0, tla.arch.zN)
                pv_p0 = tla.make_tensor_like(p0p + (pi % 2) * bm * bn, p0, tla.arch.nZ)
                read_v1 = tla.make_tensor_like(
                    v1p + (pi % vring) * bn * dv, v1, tla.arch.nZ
                )
                read_vs1 = tla.make_tensor_like(
                    vs1p + (pi % vring) * dv * 4, vs1, tla.arch.zZMxScale
                )
                read_p1 = tla.make_tensor_like(
                    p1p + (pi % ring) * bm * bn, p1, tla.arch.zN
                )
                read_ps1 = tla.make_tensor_like(
                    ps1p + (pi % ring) * bm * 4, ps1, tla.arch.nNMxScale
                )
                tla.copy(
                    pv_v0,
                    tla.get_tile(
                        read_v1, tla.make_coord(0, 0), tla.make_shape(pv_m, bn)
                    ),
                    scale=tla.get_tile(
                        read_vs1, tla.make_coord(0, 0), tla.make_shape(pv_m, bn // 32)
                    ),
                )
                tla.copy(pv_p0, read_p1, scale=read_ps1)
                if pi % 2 == 0:
                    m_v1_0.unlock(pipe=tla.pipes.MTE1)
                else:
                    m_v1_1.unlock(pipe=tla.pipes.MTE1)
                tla.cross_core_set_flag(pfree, tla.pipes.MTE1)
                if pi % 2 == 0:
                    m_pv_0.unlock(pipe=tla.pipes.MTE1)
                    m_pv_0.lock(pipe=tla.pipes.CUBE)
                else:
                    m_pv_1.unlock(pipe=tla.pipes.MTE1)
                    m_pv_1.lock(pipe=tla.pipes.CUBE)
                if ptile % group == 0:
                    m_o0.lock(pipe=tla.pipes.CUBE)
                tla.mmad_mx(o0, pv_v0, pv_p0, init_c=ptile % group == 0)
                if pi % 2 == 0:
                    m_pv_0.unlock(pipe=tla.pipes.CUBE)
                else:
                    m_pv_1.unlock(pipe=tla.pipes.CUBE)
                if ptile % group == group - 1 or ptile == selected - 1:
                    m_o0.unlock(pipe=tla.pipes.CUBE)
            if step < total_tiles:
                if step % 2 == 0:
                    m_s0_0.lock(pipe=tla.pipes.FIX)
                else:
                    m_s0_1.lock(pipe=tla.pipes.FIX)
                if step % 2 == 0:
                    tla.cross_core_wait_flag(sr, tla.pipes.FIX)
                else:
                    tla.cross_core_wait_flag(sr_second, tla.pipes.FIX)
                mx_score_fix(
                    sp + (step % 2) * 2 * half * bn, s0p + (step % 2) * bm * bn, scale
                )
                if step % 2 == 0:
                    tla.cross_core_set_flag(sr, tla.pipes.FIX)
                else:
                    tla.cross_core_set_flag(sr_second, tla.pipes.FIX)
                if step % 2 == 0:
                    m_s0_0.unlock(pipe=tla.pipes.FIX)
                else:
                    m_s0_1.unlock(pipe=tla.pipes.FIX)
            if pi >= 0:
                if ptile % group == group - 1 or ptile == selected - 1:
                    m_o0.lock(pipe=tla.pipes.FIX)
                    tla.cross_core_wait_flag(ore, tla.pipes.FIX)
                    if ptile < group:
                        tla.copy(afix, o0, split_o)
                    else:
                        tla.copy(ofix, o0, split_o)
                    tla.cross_core_set_flag(ore, tla.pipes.FIX)
                    m_o0.unlock(pipe=tla.pipes.FIX)
        with tla.vector():
            if step < total_tiles:
                if step % 2 == 0:
                    tla.cross_core_wait_flag(sr, tla.pipes.VECTOR)
                else:
                    tla.cross_core_wait_flag(sr_second, tla.pipes.VECTOR)
                m_pp.lock(pipe=tla.pipes.VECTOR)
                if qtile % group == 0:
                    _mx_group_scales_basic(psp, qmp, lp, rp, gp, -1)
                _mx_group_softmax_store_basic(
                    sp + (step % 2) * 2 * half * bn,
                    pp,
                    psp,
                    qmp,
                    lp,
                    tmp_alpha,
                    gp + (qtile % group) * 256,
                    scale,
                    qcols,
                )
                if step % 2 == 0:
                    tla.cross_core_set_flag(sr, tla.pipes.VECTOR)
                else:
                    tla.cross_core_set_flag(sr_second, tla.pipes.VECTOR)
            if pi >= 0:
                tla.cross_core_wait_flag(pfree, tla.pipes.MTE3)
            if step < total_tiles:
                m_pp.unlock(pipe=tla.pipes.VECTOR)
                m_pp.lock(pipe=tla.pipes.MTE3)
                write_p = tla.make_tensor_like(
                    tla.recast_ptr(p1p + (step % ring) * bm * bn, dtype=tla.Float16),
                    p1storage,
                    tla.arch.zN,
                )
                tla.copy(
                    tla.tile_view(
                        write_p,
                        tla.make_shape(bn, half // 2),
                        tla.make_coord(0, sub),
                    ),
                    tla.tile_view(
                        pu, tla.make_shape(bn, half // 2), tla.make_coord(0, 0)
                    ),
                )
                m_pp.unlock(pipe=tla.pipes.MTE3)
                if qtile % group == group - 1 or qtile == selected - 1:
                    group_count = qtile % group + 1
                    group_start = step - qtile % group
                    m_ps.lock(pipe=tla.pipes.VECTOR)
                    _mx_group_scales_basic(
                        psp,
                        qmp,
                        lp,
                        rp
                        + (
                            (
                                (step // selected) * ((selected + group - 1) // group)
                                + qtile // group
                            )
                            % 2
                        )
                        * half,
                        gp,
                        group_count,
                    )
                    m_ps.unlock(pipe=tla.pipes.VECTOR)
                    m_ps.lock(pipe=tla.pipes.MTE3)
                    mx_ps64_to_l1(ps1p, psp, group_start % ring, group_count, ring, sub)
                    m_ps.unlock(pipe=tla.pipes.MTE3)
                    tla.cross_core_set_flag(psready, tla.pipes.MTE3)
            if pi >= 0:
                if ptile % group == group - 1 or ptile == selected - 1:
                    tla.cross_core_wait_flag(ore, tla.pipes.VECTOR)
                    if ptile >= group:
                        mx_epilogue(
                            ap,
                            op,
                            pmp,
                            lp,
                            rp
                            + (
                                (
                                    (pi // selected) * ((selected + group - 1) // group)
                                    + ptile // group
                                )
                                % 2
                            )
                            * half,
                            tla.recast_ptr(yp, dtype=tla.UInt8),
                            ep,
                            dim,
                            valid,
                            1,
                            od_code,
                        )
                    if ptile == selected - 1:
                        m_out.lock(pipe=tla.pipes.VECTOR)
                        mx_epilogue(
                            ap,
                            op,
                            pmp,
                            lp,
                            rp,
                            tla.recast_ptr(yp, dtype=tla.UInt8),
                            ep,
                            dim,
                            valid,
                            2,
                            od_code,
                        )
                        m_out.unlock(pipe=tla.pipes.VECTOR)
                        if valid > 0:
                            m_out.lock(pipe=tla.pipes.MTE3)
                            ogm = tla.make_tensor(
                                out.ptr + pqbase + sub * half * dim,
                                tla.make_layout(
                                    tla.make_shape(valid, dim), tla.make_stride(dim, 1)
                                ),
                            )
                            tla.copy(
                                ogm,
                                tla.tile_view(
                                    yu, tla.make_shape(valid, dim), tla.make_coord(0, 0)
                                ),
                            )
                            if want_lse:
                                egm = tla.make_tensor(
                                    lse.ptr + pqbase // dim + sub * half,
                                    tla.make_layout(
                                        tla.make_shape(valid), tla.make_stride(1)
                                    ),
                                )
                                tla.copy(
                                    egm,
                                    tla.tile_view(
                                        eu, tla.make_shape(valid), tla.make_coord(0)
                                    ),
                                )
                            m_out.unlock(pipe=tla.pipes.MTE3)
                    tla.cross_core_set_flag(ore, tla.pipes.VECTOR)
    with tla.cube():
        tla.cross_core_wait_flag(sr, tla.pipes.FIX)
        tla.cross_core_wait_flag(sr_second, tla.pipes.FIX)
        tla.cross_core_wait_flag(ore, tla.pipes.FIX)


# Attention kernel: pipeline_short.
@tla.kernel
def quant_block_sparse_attention_short_kernel(
    q: tla.Tensor,
    k: tla.Tensor,
    v: tla.Tensor,
    out: tla.Tensor,
    qs: tla.Tensor,
    ks: tla.Tensor,
    vs: tla.Tensor,
    indices: tla.Tensor,
    lse: tla.Tensor,
    batch: tla.Constexpr[int],
    hq: tla.Constexpr[int],
    hk: tla.Constexpr[int],
    nq: tla.Constexpr[int],
    nk: tla.Constexpr[int],
    mh: tla.Constexpr[int],
    selected: tla.Constexpr[int],
    scale: tla.Constexpr[float],
    dim: tla.Constexpr[int],
    xb: tla.Constexpr[int],
    yb: tla.Constexpr[int],
    want_lse: tla.Constexpr[bool],
):
    # Transposed QK/PV: each AIV owns 64 query lanes. PV adds a
    # constant-one channel for the quantized probability denominator.
    group = 2 if selected <= 8 else 4
    ring = min(group, max(1, selected))
    delay = ring
    vring = max(1, selected) if selected <= 1 else 2
    bm = 128
    bn = 128
    dv = dim + 32
    pv_m = dim + 16
    half = 64
    qm = 64 if xb % 128 != 0 else 128
    fp8 = tla.Float8E4M3FN
    e8m0 = tla.Float8E8M0
    od = out.ptr.dtype
    output_stride = dim + (8 if od == tla.Float32 else 16)
    od_code = 0 if od == tla.Float32 else (1 if od == tla.Float16 else 2)
    q1p = tla.allocate(bm * dim, fp8, tla.AddressSpace.l1, 512)
    k1p = tla.allocate(2 * bn * dim, fp8, tla.AddressSpace.l1, 512)
    v1p = tla.allocate(vring * bn * dv, fp8, tla.AddressSpace.l1, 512)
    # Two P slots let softmax(t) overlap PV(t-1).
    p1p = tla.allocate(ring * bm * bn, fp8, tla.AddressSpace.l1, 512)
    qs1p = tla.allocate(bm * (dim // 32), e8m0, tla.AddressSpace.l1, 512)
    ks1p = tla.allocate(2 * bn * (dim // 32), e8m0, tla.AddressSpace.l1, 512)
    vs1p = tla.allocate(vring * dv * (bn // 32), e8m0, tla.AddressSpace.l1, 512)
    ps1p = tla.allocate(ring * bm * (bn // 32), e8m0, tla.AddressSpace.l1, 512)
    q0p = tla.allocate(bm * dim, fp8, tla.AddressSpace.l0b, 1024)
    k0p = tla.allocate(2 * bn * dim, fp8, tla.AddressSpace.l0a, 1024)
    p0p = tla.allocate(bm * bn, fp8, tla.AddressSpace.l0b, 1024)
    v0p = tla.allocate(bn * dv, fp8, tla.AddressSpace.l0a, 1024)
    s0p = tla.allocate(bm * bn, tla.Float32, tla.AddressSpace.l0c, 1024)
    o0p = tla.allocate(bm * dv, tla.Float32, tla.AddressSpace.l0c, 1024)
    # Separate score loads and probability stores into different 32 KiB UB
    # regions. The second half is spacing, not an additional score tile.
    sp = tla.allocate(2 * half * bn, tla.Float16, tla.AddressSpace.ub, 512)
    pp = tla.allocate(half * (bn + 1), tla.UInt8, tla.AddressSpace.ub, 512)
    psp = tla.allocate(16 * half * (bn // 32), tla.UInt8, tla.AddressSpace.ub, 256)
    mp = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)
    lp = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)
    rp = tla.allocate(2 * half, tla.Float32, tla.AddressSpace.ub, 256)
    gp = tla.allocate(16 * half * (bn // 32), tla.Float32, tla.AddressSpace.ub, 256)
    op = tla.allocate(half * pv_m, tla.Float32, tla.AddressSpace.ub, 512)
    ap = tla.allocate(half * pv_m, tla.Float32, tla.AddressSpace.ub, 512)
    yp = tla.allocate(half * output_stride, od, tla.AddressSpace.ub, 512)
    tmp_alpha = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)
    ep = tla.allocate(half, tla.Float32, tla.AddressSpace.ub, 256)

    q1 = tla.make_tensor(q1p, tla.make_layout_with_tag((dim, bm), fp8, tla.arch.nZ))
    k1 = tla.make_tensor(k1p, tla.make_layout_with_tag((bn, dim), fp8, tla.arch.zN))
    v1 = tla.make_tensor(v1p, tla.make_layout_with_tag((dv, bn), fp8, tla.arch.nZ))
    p1 = tla.make_tensor(p1p, tla.make_layout_with_tag((bn, bm), fp8, tla.arch.zN))
    qs1 = tla.make_tensor(
        qs1p, tla.make_layout_with_tag((dim // 32, bm), e8m0, tla.arch.nNMxScale)
    )
    ks1 = tla.make_tensor(
        ks1p, tla.make_layout_with_tag((bn, dim // 32), e8m0, tla.arch.zZMxScale)
    )
    vs1 = tla.make_tensor(
        vs1p, tla.make_layout_with_tag((dv, bn // 32), e8m0, tla.arch.zZMxScale)
    )
    ps1 = tla.make_tensor(
        ps1p, tla.make_layout_with_tag((bn // 32, bm), e8m0, tla.arch.nNMxScale)
    )
    q0 = tla.make_tensor(q0p, tla.make_layout_with_tag((dim, bm), fp8, tla.arch.nZ))
    k0 = tla.make_tensor(k0p, tla.make_layout_with_tag((bn, dim), fp8, tla.arch.zN))
    p0 = tla.make_tensor(p0p, tla.make_layout_with_tag((bn, bm), fp8, tla.arch.nZ))
    v0 = tla.make_tensor(v0p, tla.make_layout_with_tag((pv_m, bn), fp8, tla.arch.zN))
    s0 = tla.make_tensor(
        s0p, tla.make_layout_with_tag((bn, bm), tla.Float32, tla.arch.L0Clayout)
    )
    o0 = tla.make_tensor(
        o0p, tla.make_layout_with_tag((pv_m, bm), tla.Float32, tla.arch.L0Clayout)
    )
    sfix = tla.make_tensor(
        sp, tla.make_layout(tla.make_shape(bn, bm), tla.make_stride(half, 1))
    )
    afix = tla.make_tensor(
        ap, tla.make_layout(tla.make_shape(pv_m, bm), tla.make_stride(half, 1))
    )
    ofix = tla.make_tensor(
        op, tla.make_layout(tla.make_shape(pv_m, bm), tla.make_stride(half, 1))
    )
    p16ref = tla.make_tensor(
        tla.recast_ptr(pp, dtype=tla.Float16),
        tla.make_layout(
            tla.make_shape(bn + 1, half // 2), tla.make_stride(half // 2, 1)
        ),
    )
    pu = tla.make_tensor_like(p16ref.ptr, p16ref, tla.arch.zNUnAlign)
    p1storage = tla.make_tensor(
        tla.recast_ptr(p1p, dtype=tla.Float16),
        tla.make_layout_with_tag((bn, bm // 2), tla.Float16, tla.arch.zN),
    )
    psuref = tla.make_tensor(
        tla.recast_ptr(psp, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(8, 16), tla.make_stride(16, 1)),
    )
    psu = tla.make_tensor_like(psuref.ptr, psuref, tla.arch.zNUnAlign)
    psflat = tla.make_tensor(
        tla.recast_ptr(ps1p, dtype=tla.Float16),
        tla.make_layout_with_tag((16, 16), tla.Float16, tla.arch.zN),
    )
    padp = tla.allocate(2048, tla.UInt8, tla.AddressSpace.ub, 512)
    pads = tla.allocate(64, tla.UInt8, tla.AddressSpace.ub, 256)
    padref = tla.make_tensor(
        tla.recast_ptr(padp, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(64, 16), tla.make_stride(16, 1)),
    )
    padu = tla.make_tensor_like(padref.ptr, padref, tla.arch.zNUnAlign)
    pad1 = tla.make_tensor(
        tla.recast_ptr(v1p + dim * bn, dtype=tla.Float16),
        tla.make_layout_with_tag((128, 16), tla.Float16, tla.arch.zN),
    )
    padsref = tla.make_tensor(
        tla.recast_ptr(pads, dtype=tla.Float16),
        tla.make_layout(tla.make_shape(2, 16), tla.make_stride(16, 1)),
    )
    padsu = tla.make_tensor_like(padsref.ptr, padsref, tla.arch.zNUnAlign)
    pads1 = tla.make_tensor(
        tla.recast_ptr(vs1p + dim * 4, dtype=tla.Float16),
        tla.make_layout_with_tag((4, 16), tla.Float16, tla.arch.zN),
    )
    yu = tla.make_tensor(
        yp,
        tla.make_layout(tla.make_shape(half, dim), tla.make_stride(output_stride, 1)),
    )
    eu = tla.make_tensor(ep, tla.make_layout(tla.make_shape(half), tla.make_stride(1)))

    # In-core buffer ownership is expressed with mutexes.
    m_q1 = tla.mutex("q1", id=0)
    m_q0 = tla.mutex("q0", id=1)
    m_k1_0 = tla.mutex("k1_0", id=2)
    m_k1_1 = tla.mutex("k1_1", id=3)
    m_k0_0 = tla.mutex("k0_0", id=4)
    m_k0_1 = tla.mutex("k0_1", id=5)
    m_s0 = tla.mutex("s0", id=6)
    m_v1_0 = tla.mutex("v1_0", id=7)
    m_v1_1 = tla.mutex("v1_1", id=8)
    m_v1s = tla.mutex("v1_short", id=9)
    m_pv = tla.mutex("pv", id=10)
    m_o0 = tla.mutex("o0", id=11)
    m_pad = tla.mutex("pad", id=12)
    m_pp = tla.mutex("pp", id=13)
    m_ps = tla.mutex("ps", id=14)
    m_out = tla.mutex("out", id=15)
    sr = tla.cross_flag("score_ready")
    ore = tla.cross_flag("output_ready")
    psready = tla.cross_flag("scales_ready")
    pfree = tla.cross_flag("prob_consumed")
    padready = tla.cross_flag("pad_ready")
    split_o = tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N)
    with tla.vector():
        m_pp.lock(pipe=tla.pipes.VECTOR)
        mx_zero_ub(pp, half * (bn + 1))
        m_pp.unlock(pipe=tla.pipes.VECTOR)
        m_pad.lock(pipe=tla.pipes.VECTOR)
        mx_pad(padp, pads)
        m_pad.unlock(pipe=tla.pipes.VECTOR)
        m_pad.lock(pipe=tla.pipes.MTE3)
        m_pp.lock(pipe=tla.pipes.MTE3)
        if tla.arch.sub_block_idx() == 0:
            mx_zero_scale_l1(ks1p, pp, 2 * bn * (dim // 32))
            mx_zero_scale_l1(vs1p, pp, vring * dv * 4)
            for seed_slot in tla.range(vring):
                seed_v1 = tla.make_tensor_like(
                    tla.recast_ptr(
                        v1p + seed_slot * bn * dv + dim * bn, dtype=tla.Float16
                    ),
                    pad1,
                    tla.arch.zN,
                )
                seed_vs1 = tla.make_tensor_like(
                    tla.recast_ptr(
                        vs1p + seed_slot * dv * 4 + dim * 4, dtype=tla.Float16
                    ),
                    pads1,
                    tla.arch.zN,
                )
                for seed_sub in tla.range(2):
                    tla.copy(
                        tla.tile_view(
                            seed_v1, tla.make_shape(64, 16), tla.make_coord(seed_sub, 0)
                        ),
                        padu,
                    )
                    tla.copy(
                        tla.tile_view(
                            seed_vs1, tla.make_shape(2, 16), tla.make_coord(seed_sub, 0)
                        ),
                        padsu,
                    )
        m_pp.unlock(pipe=tla.pipes.MTE3)
        m_pad.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(padready, tla.pipes.MTE3)
        tla.cross_core_set_flag(sr, tla.pipes.VECTOR)
        tla.cross_core_set_flag(ore, tla.pipes.VECTOR)
    with tla.cube():
        tla.cross_core_wait_flag(padready, tla.pipes.MTE2)
    blocks = (nq + xb - 1) // xb
    subtiles = xb // qm
    total = batch * hq * blocks * subtiles
    for task in tla.range(tla.arch.block_idx(), total, tla.arch.block_num()):
        h = (task // (blocks * subtiles)) % hq
        qb = (task // subtiles) % blocks
        qi = task % subtiles
        b = task // (hq * blocks * subtiles)
        kh = h // (hq // hk)
        qstart = qb * xb + qi * qm
        if qstart < nq:
            rows = qm if nq % qm == 0 else (qm if nq - qstart >= qm else nq - qstart)
            sub = tla.arch.sub_block_idx()
            valid = rows - sub * half
            valid = half if valid > half else valid
            valid = 0 if valid < 0 else valid
            qbase = ((b * hq + h) * nq + qstart) * dim
            kbase = (b * hk + kh) * nk * dim
            vb = (b * hk + kh) * ((nk + 63) // 64) * dim * 2
            mask_head = 0 if mh == 1 else h
            maskbase = ((b * mh + mask_head) * blocks + qb) * selected
            # Common sorted indices use O(K) traversal; unordered input uses
            # successive minima, retaining reference sorting/dedup semantics.
            ordered = tla.as_numeric(1)
            previous = tla.as_numeric(-1)
            for scan in tla.range(selected):
                value = indices[maskbase + scan].to(tla.Int32)
                if value >= 0:
                    if value < previous:
                        ordered = tla.as_numeric(0)
                    previous = value
            with tla.cube():
                qgm = tla.make_tensor(
                    q.ptr + qbase,
                    tla.make_layout(
                        tla.make_shape(dim, rows),
                        tla.make_stride(1, dim),
                        layoutTag=tla.arch.ColumnMajor,
                    ),
                )
                m_q1.lock(pipe=tla.pipes.MTE2)
                tla.copy(
                    tla.tile_view(q1, tla.make_shape(dim, rows), tla.make_coord(0, 0)),
                    qgm,
                )
                qsgm = tla.get_tile(
                    qs, tla.make_coord(0, qbase // dim), tla.make_shape(dim // 32, rows)
                )
                tla.copy(
                    tla.tile_view(
                        qs1, tla.make_shape(dim // 32, rows), tla.make_coord(0, 0)
                    ),
                    qsgm,
                )
                m_q1.unlock(pipe=tla.pipes.MTE2)
                m_q1.lock(pipe=tla.pipes.MTE1)
                m_q0.lock(pipe=tla.pipes.MTE1)
                tla.copy(q0, q1, scale=qs1)
                m_q0.unlock(pipe=tla.pipes.MTE1)
                m_q1.unlock(pipe=tla.pipes.MTE1)
            with tla.vector():
                _mx_group_scales_basic(psp, mp, lp, rp, gp, -2)
            if selected > 0 and selected <= 1:
                with tla.cube():
                    m_v1s.lock(pipe=tla.pipes.MTE2)
            qlast = tla.as_numeric(-1)
            vlast = tla.as_numeric(-1)
            for step in tla.range_constexpr(selected + delay):
                pi = step - delay
                qcols = tla.as_numeric(0)
                qstart_k = tla.as_numeric(0)
                vcols = tla.as_numeric(0)
                vstart_k = tla.as_numeric(0)
                if step < selected:
                    qblock = indices[maskbase + step].to(tla.Int32)
                    if ordered == 0:
                        qblock = (nk + bn - 1) // bn
                        for qi_scan in tla.range(selected):
                            qcandidate = indices[maskbase + qi_scan].to(tla.Int32)
                            if qcandidate > qlast:
                                if qcandidate < qblock:
                                    qblock = qcandidate
                    if qblock > qlast:
                        qlast = qblock
                        if qblock * bn < nk:
                            qstart_k = qblock * bn
                            qcols = bn if nk - qstart_k >= bn else nk - qstart_k
                if pi >= 0:
                    pblock = indices[maskbase + pi].to(tla.Int32)
                    if ordered == 0:
                        pblock = (nk + bn - 1) // bn
                        for vi_scan in tla.range(selected):
                            vcandidate = indices[maskbase + vi_scan].to(tla.Int32)
                            if vcandidate > vlast:
                                if vcandidate < pblock:
                                    pblock = vcandidate
                    if pblock > vlast:
                        vlast = pblock
                        if pblock * bn < nk:
                            vstart_k = pblock * bn
                            vcols = bn if nk - vstart_k >= bn else nk - vstart_k
                with tla.cube():
                    if step < selected:
                        if step % 2 == 0:
                            m_k1_0.lock(pipe=tla.pipes.MTE2)
                        else:
                            m_k1_1.lock(pipe=tla.pipes.MTE2)
                        current_k1 = tla.make_tensor_like(
                            k1p + (step % 2) * bn * dim, k1, tla.arch.zN
                        )
                        current_ks1 = tla.make_tensor_like(
                            ks1p + (step % 2) * bn * (dim // 32),
                            ks1,
                            tla.arch.zZMxScale,
                        )
                        if qcols < bn:
                            current_k1.fill(0)
                        if qcols > 0:
                            k_copy_cols = bn if nk % bn == 0 else qcols
                            kgm = tla.make_tensor(
                                k.ptr + kbase + qstart_k * dim,
                                tla.make_layout(
                                    tla.make_shape(k_copy_cols, dim),
                                    tla.make_stride(dim, 1),
                                ),
                            )
                            ksgm = tla.get_tile(
                                ks,
                                tla.make_coord(kbase // dim + qstart_k, 0),
                                tla.make_shape(k_copy_cols, dim // 32),
                            )
                            tla.copy(
                                tla.tile_view(
                                    current_k1,
                                    tla.make_shape(k_copy_cols, dim),
                                    tla.make_coord(0, 0),
                                ),
                                kgm,
                            )
                            tla.copy(
                                tla.tile_view(
                                    current_ks1,
                                    tla.make_shape(k_copy_cols, dim // 32),
                                    tla.make_coord(0, 0),
                                ),
                                ksgm,
                            )
                        if step % 2 == 0:
                            m_k1_0.unlock(pipe=tla.pipes.MTE2)
                        else:
                            m_k1_1.unlock(pipe=tla.pipes.MTE2)
                    if selected <= 1:
                        if step < selected:
                            preload_v = tla.make_tensor_like(
                                v1p + step * bn * dv, v1, tla.arch.nZ
                            )
                            preload_vs = tla.make_tensor_like(
                                vs1p + step * dv * 4, vs1, tla.arch.zZMxScale
                            )
                            if qcols < bn:
                                tla.get_tile(
                                    preload_v,
                                    tla.make_coord(0, 0),
                                    tla.make_shape(dim, bn),
                                ).fill(0)
                            if qcols > 0:
                                short_vgm = tla.make_tensor(
                                    v.ptr + kbase + qstart_k * dim,
                                    tla.make_layout(
                                        tla.make_shape(dim, qcols),
                                        tla.make_stride(1, dim),
                                        layoutTag=tla.arch.ColumnMajor,
                                    ),
                                )
                                short_groups = ((qcols + 63) // 64) * 2
                                short_vsgm = tla.get_tile(
                                    vs,
                                    tla.make_coord(0, vb // dim + qstart_k // 32),
                                    tla.make_shape(dim, short_groups),
                                )
                                tla.copy(
                                    tla.tile_view(
                                        preload_v,
                                        tla.make_shape(dim, qcols),
                                        tla.make_coord(0, 0),
                                    ),
                                    short_vgm,
                                )
                                tla.copy(
                                    tla.tile_view(
                                        preload_vs,
                                        tla.make_shape(dim, short_groups),
                                        tla.make_coord(0, 0),
                                    ),
                                    short_vsgm,
                                )
                        if step == selected - 1:
                            m_v1s.unlock(pipe=tla.pipes.MTE2)
                    else:
                        if pi >= 0:
                            if pi % 2 == 0:
                                m_v1_0.lock(pipe=tla.pipes.MTE2)
                            else:
                                m_v1_1.lock(pipe=tla.pipes.MTE2)
                            preload_vlong = tla.make_tensor_like(
                                v1p + (pi % 2) * bn * dv, v1, tla.arch.nZ
                            )
                            preload_vslong = tla.make_tensor_like(
                                vs1p + (pi % 2) * dv * 4, vs1, tla.arch.zZMxScale
                            )
                            if vcols < bn:
                                tla.get_tile(
                                    preload_vlong,
                                    tla.make_coord(0, 0),
                                    tla.make_shape(dim, bn),
                                ).fill(0)
                            if vcols > 0:
                                v_copy_cols = bn if nk % bn == 0 else vcols
                                long_vgm = tla.make_tensor(
                                    v.ptr + kbase + vstart_k * dim,
                                    tla.make_layout(
                                        tla.make_shape(dim, v_copy_cols),
                                        tla.make_stride(1, dim),
                                        layoutTag=tla.arch.ColumnMajor,
                                    ),
                                )
                                long_groups = ((v_copy_cols + 63) // 64) * 2
                                long_vsgm = tla.get_tile(
                                    vs,
                                    tla.make_coord(0, vb // dim + vstart_k // 32),
                                    tla.make_shape(dim, long_groups),
                                )
                                tla.copy(
                                    tla.tile_view(
                                        preload_vlong,
                                        tla.make_shape(dim, v_copy_cols),
                                        tla.make_coord(0, 0),
                                    ),
                                    long_vgm,
                                )
                                tla.copy(
                                    tla.tile_view(
                                        preload_vslong,
                                        tla.make_shape(dim, long_groups),
                                        tla.make_coord(0, 0),
                                    ),
                                    long_vsgm,
                                )
                            if pi % 2 == 0:
                                m_v1_0.unlock(pipe=tla.pipes.MTE2)
                            else:
                                m_v1_1.unlock(pipe=tla.pipes.MTE2)
                    if step < selected:
                        if step % 2 == 0:
                            m_k1_0.lock(pipe=tla.pipes.MTE1)
                        else:
                            m_k1_1.lock(pipe=tla.pipes.MTE1)
                        if step % 2 == 0:
                            m_k0_0.lock(pipe=tla.pipes.MTE1)
                        else:
                            m_k0_1.lock(pipe=tla.pipes.MTE1)
                        current_k0 = tla.make_tensor_like(
                            k0p + (step % 2) * bn * dim, k0, tla.arch.zN
                        )
                        read_k1 = tla.make_tensor_like(
                            k1p + (step % 2) * bn * dim, k1, tla.arch.zN
                        )
                        read_ks1 = tla.make_tensor_like(
                            ks1p + (step % 2) * bn * (dim // 32),
                            ks1,
                            tla.arch.zZMxScale,
                        )
                        tla.copy(current_k0, read_k1, scale=read_ks1)
                        if step % 2 == 0:
                            m_k1_0.unlock(pipe=tla.pipes.MTE1)
                        else:
                            m_k1_1.unlock(pipe=tla.pipes.MTE1)
                        if step % 2 == 0:
                            m_k0_0.unlock(pipe=tla.pipes.MTE1)
                        else:
                            m_k0_1.unlock(pipe=tla.pipes.MTE1)
                        if step % 2 == 0:
                            m_k0_0.lock(pipe=tla.pipes.CUBE)
                        else:
                            m_k0_1.lock(pipe=tla.pipes.CUBE)
                        m_q0.lock(pipe=tla.pipes.CUBE)
                        m_s0.lock(pipe=tla.pipes.CUBE)
                        tla.mmad_mx(s0, current_k0, q0, init_c=True)
                        m_s0.unlock(pipe=tla.pipes.CUBE)
                        m_q0.unlock(pipe=tla.pipes.CUBE)
                        if step % 2 == 0:
                            m_k0_0.unlock(pipe=tla.pipes.CUBE)
                        else:
                            m_k0_1.unlock(pipe=tla.pipes.CUBE)
                    if pi >= 0:
                        if selected > 1 or pi == 0:
                            if pi % 2 == 0 and selected > 1:
                                m_v1_0.lock(pipe=tla.pipes.MTE1)
                            elif selected > 1:
                                m_v1_1.lock(pipe=tla.pipes.MTE1)
                            else:
                                m_v1s.lock(pipe=tla.pipes.MTE1)
                        if pi % group == 0:
                            tla.cross_core_wait_flag(psready, tla.pipes.MTE1)
                        m_pv.lock(pipe=tla.pipes.MTE1)
                        read_v1 = tla.make_tensor_like(
                            v1p + (pi % vring) * bn * dv, v1, tla.arch.nZ
                        )
                        read_vs1 = tla.make_tensor_like(
                            vs1p + (pi % vring) * dv * 4, vs1, tla.arch.zZMxScale
                        )
                        read_p1 = tla.make_tensor_like(
                            p1p + (pi % ring) * bm * bn, p1, tla.arch.zN
                        )
                        read_ps1 = tla.make_tensor_like(
                            ps1p + (pi % ring) * bm * 4, ps1, tla.arch.nNMxScale
                        )
                        tla.copy(
                            v0,
                            tla.get_tile(
                                read_v1, tla.make_coord(0, 0), tla.make_shape(pv_m, bn)
                            ),
                            scale=tla.get_tile(
                                read_vs1,
                                tla.make_coord(0, 0),
                                tla.make_shape(pv_m, bn // 32),
                            ),
                        )
                        tla.copy(p0, read_p1, scale=read_ps1)
                        if selected > 1:
                            if pi % 2 == 0:
                                m_v1_0.unlock(pipe=tla.pipes.MTE1)
                            else:
                                m_v1_1.unlock(pipe=tla.pipes.MTE1)
                        elif pi == selected - 1:
                            m_v1s.unlock(pipe=tla.pipes.MTE1)
                        tla.cross_core_set_flag(pfree, tla.pipes.MTE1)
                        m_pv.unlock(pipe=tla.pipes.MTE1)
                        m_pv.lock(pipe=tla.pipes.CUBE)
                        if pi % group == 0:
                            m_o0.lock(pipe=tla.pipes.CUBE)
                        tla.mmad_mx(o0, v0, p0, init_c=pi % group == 0)
                        m_pv.unlock(pipe=tla.pipes.CUBE)
                        if pi % group == group - 1 or pi == selected - 1:
                            m_o0.unlock(pipe=tla.pipes.CUBE)
                    if step < selected:
                        m_s0.lock(pipe=tla.pipes.FIX)
                        tla.cross_core_wait_flag(sr, tla.pipes.FIX)
                        mx_score_fix(sp, s0p, scale)
                        tla.cross_core_set_flag(sr, tla.pipes.FIX)
                        m_s0.unlock(pipe=tla.pipes.FIX)
                    if pi >= 0:
                        if pi % group == group - 1 or pi == selected - 1:
                            m_o0.lock(pipe=tla.pipes.FIX)
                            tla.cross_core_wait_flag(ore, tla.pipes.FIX)
                            if pi < group:
                                tla.copy(afix, o0, split_o)
                            else:
                                tla.copy(ofix, o0, split_o)
                            tla.cross_core_set_flag(ore, tla.pipes.FIX)
                            m_o0.unlock(pipe=tla.pipes.FIX)
                with tla.vector():
                    if step < selected:
                        tla.cross_core_wait_flag(sr, tla.pipes.VECTOR)
                        m_pp.lock(pipe=tla.pipes.VECTOR)
                        if step % group == 0:
                            _mx_group_scales_basic(psp, mp, lp, rp, gp, -1)
                        _mx_group_softmax_store_basic(
                            sp,
                            pp,
                            psp,
                            mp,
                            lp,
                            tmp_alpha,
                            gp + (step % group) * 256,
                            scale,
                            qcols,
                        )
                        tla.cross_core_set_flag(sr, tla.pipes.VECTOR)
                    if pi >= 0:
                        tla.cross_core_wait_flag(pfree, tla.pipes.MTE3)
                    if step < selected:
                        m_pp.unlock(pipe=tla.pipes.VECTOR)
                        m_pp.lock(pipe=tla.pipes.MTE3)
                        write_p = tla.make_tensor_like(
                            tla.recast_ptr(
                                p1p + (step % ring) * bm * bn, dtype=tla.Float16
                            ),
                            p1storage,
                            tla.arch.zN,
                        )
                        tla.copy(
                            tla.tile_view(
                                write_p,
                                tla.make_shape(bn, half // 2),
                                tla.make_coord(0, sub),
                            ),
                            tla.tile_view(
                                pu,
                                tla.make_shape(bn, half // 2),
                                tla.make_coord(0, 0),
                            ),
                        )
                        m_pp.unlock(pipe=tla.pipes.MTE3)
                        if step % group == group - 1 or step == selected - 1:
                            group_count = step % group + 1
                            group_start = step - step % group
                            m_ps.lock(pipe=tla.pipes.VECTOR)
                            _mx_group_scales_basic(
                                psp,
                                mp,
                                lp,
                                rp + ((step // group) % 2) * half,
                                gp,
                                group_count,
                            )
                            m_ps.unlock(pipe=tla.pipes.VECTOR)
                            m_ps.lock(pipe=tla.pipes.MTE3)
                            for scale_slot in tla.range(group_count):
                                write_ps = tla.make_tensor_like(
                                    tla.recast_ptr(
                                        ps1p
                                        + ((group_start + scale_slot) % ring) * bm * 4,
                                        dtype=tla.Float16,
                                    ),
                                    psflat,
                                    tla.arch.zN,
                                )
                                read_ps = tla.make_tensor_like(
                                    tla.recast_ptr(
                                        psp + scale_slot * 256, dtype=tla.Float16
                                    ),
                                    psu,
                                    tla.arch.zNUnAlign,
                                )
                                tla.copy(
                                    tla.tile_view(
                                        write_ps,
                                        tla.make_shape(8, 16),
                                        tla.make_coord(sub, 0),
                                    ),
                                    read_ps,
                                )
                            m_ps.unlock(pipe=tla.pipes.MTE3)
                            tla.cross_core_set_flag(psready, tla.pipes.MTE3)
                    if pi >= 0:
                        if pi % group == group - 1 or pi == selected - 1:
                            tla.cross_core_wait_flag(ore, tla.pipes.VECTOR)
                            if pi >= group:
                                mx_epilogue(
                                    ap,
                                    op,
                                    mp,
                                    lp,
                                    rp + ((pi // group) % 2) * half,
                                    tla.recast_ptr(yp, dtype=tla.UInt8),
                                    ep,
                                    dim,
                                    valid,
                                    1,
                                    od_code,
                                )
                            tla.cross_core_set_flag(ore, tla.pipes.VECTOR)
            with tla.vector():
                m_out.lock(pipe=tla.pipes.VECTOR)
                if selected == 0:
                    mx_epilogue(
                        ap,
                        op,
                        mp,
                        lp,
                        rp,
                        tla.recast_ptr(yp, dtype=tla.UInt8),
                        ep,
                        dim,
                        valid,
                        0,
                        od_code,
                    )
                mx_epilogue(
                    ap,
                    op,
                    mp,
                    lp,
                    rp,
                    tla.recast_ptr(yp, dtype=tla.UInt8),
                    ep,
                    dim,
                    valid,
                    2,
                    od_code,
                )
                m_out.unlock(pipe=tla.pipes.VECTOR)
                if valid > 0:
                    m_out.lock(pipe=tla.pipes.MTE3)
                    ogm = tla.make_tensor(
                        out.ptr + qbase + sub * half * dim,
                        tla.make_layout(
                            tla.make_shape(valid, dim), tla.make_stride(dim, 1)
                        ),
                    )
                    tla.copy(
                        ogm,
                        tla.tile_view(
                            yu, tla.make_shape(valid, dim), tla.make_coord(0, 0)
                        ),
                    )
                    if want_lse:
                        egm = tla.make_tensor(
                            lse.ptr + qbase // dim + sub * half,
                            tla.make_layout(tla.make_shape(valid), tla.make_stride(1)),
                        )
                        tla.copy(
                            egm,
                            tla.tile_view(eu, tla.make_shape(valid), tla.make_coord(0)),
                        )
                    m_out.unlock(pipe=tla.pipes.MTE3)
    with tla.cube():
        tla.cross_core_wait_flag(sr, tla.pipes.FIX)
        tla.cross_core_wait_flag(ore, tla.pipes.FIX)


# Runtime score bound for the fixed-exponent specialization.

_NORM_TILE = 128
_NORM_ROWS = 128


@tla.jit
def mx_guard_load(dst, src, bytes):
    layout = tla.make_layout(tla.make_shape(bytes), tla.make_stride(1))
    target = tla.make_tensor(tla.recast_ptr(dst, dtype=tla.Int8), layout)
    source = tla.make_tensor(tla.recast_ptr(src, dtype=tla.Int8), layout)
    tla.copy(target, source)


@tla.jit
def mx_guard_store(dst, src):
    # A one-element row keeps the GM write at four bytes, including the tail.
    target = tla.make_tensor(
        dst, tla.make_layout(tla.make_shape(1, 1), tla.make_stride(1, 1))
    )
    source = tla.make_tensor(
        src, tla.make_layout(tla.make_shape(1, 1), tla.make_stride(8, 1))
    )
    tla.copy(target, source)


@tla.jit
def _mx_row_norm(xp, sp, op, count):
    # Decoded squared norm of each MXFP8 row, reduced to one block maximum.
    # A row is four E8M0 groups of 32 channels; the four scale bytes arrive as
    # one broadcast 32-bit word and each lane picks its own byte.
    data = tla.make_tensor(
        tla.recast_ptr(xp, dtype=tla.Float8E4M3FN),
        tla.make_layout(tla.make_shape(_NORM_TILE * 128), tla.make_stride(1)),
    )
    words = tla.make_tensor(
        tla.recast_ptr(sp, dtype=tla.Int32),
        tla.make_layout(tla.make_shape(_NORM_TILE), tla.make_stride(1)),
    )
    outt = _mx_vector_tile(op, 64)
    with tla.vec.func():
        mf = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        plain = tla.params.CastParams()
        brc = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
        unpack = tla.params.NormalLoadParams(
            load_dist=tla.params.LoadDist.DIST_UNPACK4_B8
        )
        # Byte offset of a lane's group inside the packed scale word; the upper
        # half of a row uses the two groups sixteen bits further up.
        lane = tla.arange(0, dtype=tla.Int32)
        shift_lo = tla.shift_left(tla.shift_right(lane, 5, mask=mf), 3, mask=mf)
        shift_hi = tla.add(shift_lo, tla.full(16, tla.Int32), mask=mf)
        byte = tla.full(255, tla.Int32)
        one_e = tla.full(0x00400000, tla.Int32)
        best = tla.full(0.0, tla.Float32)
        for r in tla.range(count):
            bits = tla.tile_view(words, tla.make_shape(1), tla.make_coord(r)).load(
                params=brc
            )
            # 2**(byte-127) straight into the float exponent field, floored at
            # 2**-126 so a zero byte cannot collapse the bound.
            lo_scale = tla.max(
                tla.shift_left(
                    tla.bitwise_and(
                        tla.shift_right(bits, shift_lo, mask=mf), byte, mask=mf
                    ),
                    23,
                    mask=mf,
                ),
                one_e,
                mask=mf,
            ).bitcast(tla.Float32)
            hi_scale = tla.max(
                tla.shift_left(
                    tla.bitwise_and(
                        tla.shift_right(bits, shift_hi, mask=mf), byte, mask=mf
                    ),
                    23,
                    mask=mf,
                ),
                one_e,
                mask=mf,
            ).bitcast(tla.Float32)
            # The FP8 decode wants the 256-lane source mask, so leave the
            # cast unmasked; UNPACK4_B8 zeroed every other byte already.
            lo = (
                tla.tile_view(data, tla.make_shape(64), tla.make_coord(r * 2))
                .load(params=unpack)
                .to(tla.Float32, plain)
            )
            hi = (
                tla.tile_view(data, tla.make_shape(64), tla.make_coord(r * 2 + 1))
                .load(params=unpack)
                .to(tla.Float32, plain)
            )
            lo = tla.mul(lo, lo_scale, mask=mf)
            hi = tla.mul(hi, hi_scale, mask=mf)
            square = tla.add(
                tla.mul(lo, lo, mask=mf), tla.mul(hi, hi, mask=mf), mask=mf
            )
            total = square.reduce(tla.ReductionOp.ADD, mask=mf)
            best = tla.max(best, tla.full(total, tla.Float32), mask=mf)
        outt.store(best)


@tla.kernel
def row_norm_kernel(
    x: tla.Tensor, scales: tla.Tensor, out: tla.Tensor, rows: tla.Constexpr[int]
):
    xp = tla.allocate(_NORM_TILE * 128 + 256, tla.UInt8, tla.AddressSpace.ub, 512)
    sp = tla.allocate(_NORM_TILE * 4 + 256, tla.UInt8, tla.AddressSpace.ub, 512)
    op = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    mutex_x = tla.mutex("norm_x", id=0)
    mutex_s = tla.mutex("norm_s", id=1)
    mutex_o = tla.mutex("norm_o", id=2)
    with tla.vector():
        for t in tla.range(
            tla.arch.block_idx(),
            (rows + _NORM_TILE - 1) // _NORM_TILE,
            tla.arch.block_num(),
        ):
            count = (
                _NORM_TILE
                if rows - t * _NORM_TILE >= _NORM_TILE
                else rows - t * _NORM_TILE
            )
            mutex_x.lock(pipe=tla.pipes.MTE2)
            mutex_s.lock(pipe=tla.pipes.MTE2)
            mx_guard_load(xp, x.ptr + t * _NORM_TILE * 128, count * 128)
            mx_guard_load(sp, scales.ptr + t * _NORM_TILE * 4, count * 4)
            mutex_s.unlock(pipe=tla.pipes.MTE2)
            mutex_x.unlock(pipe=tla.pipes.MTE2)
            mutex_x.lock(pipe=tla.pipes.VECTOR)
            mutex_s.lock(pipe=tla.pipes.VECTOR)
            mutex_o.lock(pipe=tla.pipes.VECTOR)
            _mx_row_norm(xp, sp, op, count)
            mutex_o.unlock(pipe=tla.pipes.VECTOR)
            mutex_s.unlock(pipe=tla.pipes.VECTOR)
            mutex_x.unlock(pipe=tla.pipes.VECTOR)
            mutex_o.lock(pipe=tla.pipes.MTE3)
            mx_guard_store(out.ptr + t, op)
            mutex_o.unlock(pipe=tla.pipes.MTE3)


_NORM_EXECUTORS = {}


def max_squared_norm(x, scales):
    """Maximum decoded row norm squared, conservatively floored at 2**-100.

    This bound only selects a kernel specialization, so it runs as one fused
    device kernel: widen the E4M3 channels, apply the E8M0 group scales, then
    square and reduce to one value per row block.
    """
    import torch_npu

    if x.shape[-1] != 128:
        raise ValueError("row norm bound expects 128-channel rows")
    rows = x.numel() // 128
    out = torch.empty(
        ((rows + _NORM_ROWS - 1) // _NORM_ROWS,), dtype=torch.float32, device=x.device
    )
    args = [
        tla.from_dlpack(
            t.view(torch.uint8).flatten() if i < 2 else t, layout_tag=tla.arch.RowMajor
        )
        for i, t in enumerate((x, scales, out))
    ]
    key = (str(x.device), rows)
    if key not in _NORM_EXECUTORS:
        _NORM_EXECUTORS[key] = tla.compile(
            row_norm_kernel, *args, rows, options="--npu-arch 3510"
        )
    _NORM_EXECUTORS[key](
        *args,
        block_num=min(
            int(torch_npu.npu.get_device_properties(x.device).vector_core_num),
            out.numel(),
        ),
    )
    return out.amax().clamp_min(2.0**-100).reshape(1, 1)


def scores_are_bounded(q, k, qs, ks, scale):
    if abs(scale) > 1e10:
        return False
    q2 = max_squared_norm(q, qs)
    k2 = max_squared_norm(k, ks)
    bound2 = q2 * k2 * (scale * scale)
    # One small host read chooses the compiled specialization. All preceding
    # device kernels are part of the operator and its benchmark timing.
    return bool((torch.isfinite(bound2) & (bound2 <= 900.0)).all().item())


# Layout and variable-length adaptation; all copies stay on the NPU.


def _sequence_lengths(value, size, batch, packed):
    if value is None:
        if packed:
            raise ValueError("TND requires cumulative sequence ends")
        return [size] * batch
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
            raise ValueError("lengths must be a 1D integer tensor")
        value = value.tolist()
    if len(value) != batch or any(int(x) != x for x in value):
        raise ValueError("one integer length is required per batch")
    ends = list(map(int, value))
    result = [b - a for a, b in zip([0] + ends, ends)] if packed else ends
    if any(x < 0 or x > size for x in result) or (packed and ends[-1] != size):
        raise ValueError("invalid sequence lengths/cumulative ends")
    return result


def _run_layouts(
    core,
    q,
    k,
    v,
    block_indices,
    block_shape,
    q_dequant_scale,
    k_dequant_scale,
    v_dequant_scale,
    q_input_layout="BNSD",
    kv_input_layout=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    softmax_scale=None,
    output_dtype="bfloat16",
    return_lse=False,
):
    if softmax_scale is not None and not math.isfinite(float(softmax_scale)):
        raise ValueError("softmax_scale must be finite")
    layout = q_input_layout
    if layout not in ("BNSD", "BSND", "TND") or kv_input_layout not in (None, layout):
        raise ValueError("Q/KV layouts must match and be BNSD, BSND, or TND")
    if (actual_seq_lengths is None) != (actual_seq_lengths_kv is None):
        raise ValueError("both sequence-length arguments must be supplied together")
    packed = layout == "TND"
    if any(
        t.ndim != (3 if packed else 4) or t.dtype != torch.float8_e4m3fn
        for t in (q, k, v)
    ):
        raise ValueError("invalid Q/K/V rank or FP8 dtype")
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("incompatible Q/K/V shapes")
    if block_indices.ndim not in (3, 4) or block_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("block_indices must be 3D/4D integers")
    batch = block_indices.shape[0]
    if batch < 1 or (not packed and (q.shape[0] != batch or k.shape[0] != batch)):
        raise ValueError("invalid batch dimensions")
    dim = q.shape[-1]
    ha, sa = (1, 0) if packed else ((1, 2) if layout == "BNSD" else (2, 1))
    hq, hk = q.shape[ha], k.shape[ha]
    if dim not in (64, 128) or hq < 1 or hk < 1 or hq % hk:
        raise ValueError("invalid heads/head dimension")
    qlens = _sequence_lengths(actual_seq_lengths, q.shape[sa], batch, packed)
    klens = _sequence_lengths(actual_seq_lengths_kv, k.shape[sa], batch, packed)
    if len(block_shape) != 2 or any(
        x <= 0 or int(x) != x or x % 64 for x in block_shape
    ):
        raise ValueError("block dimensions must be positive multiples of 64")
    xb = int(block_shape[0])
    qmax = max(qlens) if packed else q.shape[sa]
    if block_indices.shape[-2] != (qmax + xb - 1) // xb:
        raise ValueError("invalid block_indices query dimension")
    if block_indices.ndim == 4 and block_indices.shape[1] not in (1, hq):
        raise ValueError("invalid block_indices head dimension")
    vshape = (
        (sum((n + 63) // 64 for n in klens), hk, dim, 2)
        if packed
        else (
            (batch, hk, (k.shape[sa] + 63) // 64, dim, 2)
            if layout == "BNSD"
            else (batch, (k.shape[sa] + 63) // 64, hk, dim, 2)
        )
    )
    for t, shape in zip(
        (q_dequant_scale, k_dequant_scale, v_dequant_scale),
        (q.shape[:-1] + (dim // 64, 2), k.shape[:-1] + (dim // 64, 2), vshape),
    ):
        if t.shape != shape or t.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise ValueError("invalid MX scale shape/dtype")
    tensors = (
        q,
        k,
        v,
        block_indices,
        q_dequant_scale,
        k_dequant_scale,
        v_dequant_scale,
    )
    if q.device.type != "npu" or any(t.device != q.device for t in tensors):
        raise ValueError("all inputs must be on one NPU")
    od = (
        getattr(torch, output_dtype, None)
        if isinstance(output_dtype, str)
        else output_dtype
    )
    if od not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("unsupported output dtype")
    if (
        layout == "BNSD"
        and q.shape[2] > 0
        and k.shape[2] > 0
        and qlens == [q.shape[2]] * batch
        and klens == [k.shape[2]] * batch
    ):
        return core(
            q,
            k,
            v,
            block_indices,
            block_shape,
            q_dequant_scale,
            k_dequant_scale,
            v_dequant_scale,
            softmax_scale=softmax_scale,
            output_dtype=od,
            return_lse=return_lse,
        )
    out = torch.zeros(q.shape, dtype=od, device=q.device)
    lse = (
        torch.full(
            q.shape[:-1] + (1,), -torch.inf, dtype=torch.float32, device=q.device
        )
        if return_lse
        else None
    )
    raw = [
        t.view(torch.uint8)
        for t in (q, k, v, q_dequant_scale, k_dequant_scale, v_dequant_scale)
    ]
    qo = ko = vo = 0
    for b, (qn, kn) in enumerate(zip(qlens, klens)):
        if qn and kn:
            if packed:
                parts = [
                    raw[0][qo : qo + qn].transpose(0, 1),
                    raw[1][ko : ko + kn].transpose(0, 1),
                    raw[2][ko : ko + kn].transpose(0, 1),
                    raw[3][qo : qo + qn].transpose(0, 1),
                    raw[4][ko : ko + kn].transpose(0, 1),
                    raw[5][vo : vo + (kn + 63) // 64].transpose(0, 1),
                ]
                target = out[qo : qo + qn].transpose(0, 1)
                lt = lse[qo : qo + qn].transpose(0, 1) if return_lse else None
            else:
                if layout == "BNSD":
                    parts = [t[b] for t in raw]
                else:
                    parts = [t[b].transpose(0, 1) for t in raw]
                parts = [
                    t[:, :n]
                    for t, n in zip(parts, (qn, kn, kn, qn, kn, (kn + 63) // 64))
                ]
                target = (
                    out[b, :, :qn] if layout == "BNSD" else out[b, :qn].transpose(0, 1)
                )
                lt = (
                    (
                        lse[b, :, :qn]
                        if layout == "BNSD"
                        else lse[b, :qn].transpose(0, 1)
                    )
                    if return_lse
                    else None
                )
            parts = [t.unsqueeze(0).contiguous() for t in parts]
            qq, kk, vv = [t.view(torch.float8_e4m3fn) for t in parts[:3]]
            bi = block_indices[b : b + 1, ..., : (qn + xb - 1) // xb, :].contiguous()
            result = core(
                qq,
                kk,
                vv,
                bi,
                block_shape,
                *parts[3:],
                softmax_scale=softmax_scale,
                output_dtype=od,
                return_lse=return_lse,
            )
            if return_lse:
                target.copy_(result[0][0])
                lt.copy_(result[1][0])
            else:
                target.copy_(result[0])
        qo += qn
        ko += kn
        vo += (kn + 63) // 64
    return (out, lse) if return_lse else out


# Public entry point and compiled attention cache.
_EXECUTORS = {}


def _run_bnsd(
    q,
    k,
    v,
    block_indices,
    block_shape,
    q_dequant_scale,
    k_dequant_scale,
    v_dequant_scale,
    q_input_layout="BNSD",
    kv_input_layout=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    softmax_scale=None,
    output_dtype="bfloat16",
    return_lse=False,
):
    import torch_npu

    if q_input_layout != "BNSD" or kv_input_layout not in (None, "BNSD"):
        raise ValueError("currently requires BNSD layout")
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("incompatible Q/K/V shapes")
    b, hq, nq, dim = q.shape
    bk, hk, nk, _ = k.shape
    if b != bk or b < 1 or hq < 1 or hk < 1 or hq % hk or dim not in (64, 128):
        raise ValueError("invalid batch/heads/head dimension")
    if actual_seq_lengths is not None and list(actual_seq_lengths) != [nq] * b:
        raise ValueError("variable lengths are not yet supported")
    if actual_seq_lengths_kv is not None and list(actual_seq_lengths_kv) != [nk] * b:
        raise ValueError("variable lengths are not yet supported")
    if len(block_shape) != 2 or any(
        x <= 0 or int(x) != x or x % 64 for x in block_shape
    ):
        raise ValueError("block_shape must be two positive multiples of 64")
    xb, yb = map(int, block_shape)
    if block_indices.ndim not in (3, 4) or block_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("block_indices must be 3D/4D int32/int64")
    mh = 1 if block_indices.ndim == 3 else block_indices.shape[1]
    if (
        block_indices.shape[0] != b
        or mh not in (1, hq)
        or block_indices.shape[-2] != (nq + xb - 1) // xb
    ):
        raise ValueError("invalid block_indices shape")
    if isinstance(output_dtype, str):
        output_dtype = getattr(torch, output_dtype, None)
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("unsupported output dtype")
    if any(x.dtype != torch.float8_e4m3fn for x in (q, k, v)):
        raise ValueError("Q/K/V must be E4M3FN")
    scales = (q_dequant_scale, k_dequant_scale, v_dequant_scale)
    shapes = (
        q.shape[:-1] + (dim // 64, 2),
        k.shape[:-1] + (dim // 64, 2),
        (b, hk, (nk + 63) // 64, dim, 2),
    )
    for x, shape in zip(scales, shapes):
        if x.shape != shape or x.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise ValueError("invalid scale shape/dtype")
    inputs = (q, k, v, *scales, block_indices)
    if q.device.type != "npu" or any(
        x.device != q.device or not x.is_contiguous() for x in inputs
    ):
        raise ValueError("inputs must be contiguous on one NPU")
    scale = dim**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")
    out = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    lse = (
        torch.empty(q.shape[:-1] + (1,), dtype=torch.float32, device=q.device)
        if return_lse
        else out
    )
    tensors = (q, k, v, out, *scales, block_indices, lse)
    grouped = xb == 128 and yb == 128
    bindings = []
    for i, x in enumerate(tensors):
        element = (
            tla.Float8E4M3FN if i < 3 else (tla.Float8E8M0 if 4 <= i <= 6 else None)
        )
        if not x.numel():
            x = torch.empty((1,), dtype=x.dtype, device=x.device)
        raw = x.view(torch.uint8) if element is not None else x
        tag = tla.arch.RowMajor
        shape = (raw.numel(),)
        if i == 4:
            tag = tla.arch.ColMajorMxScaleB
            shape = (dim // 32, raw.numel() // (dim // 32))
        elif i == 5:
            tag = tla.arch.RowMajorMxScaleA
            shape = (raw.numel() // (dim // 32), dim // 32)
        elif i == 6:
            tag = tla.arch.ColMajorMxScaleA
            shape = (dim, raw.numel() // dim)
        binding = tla.from_dlpack(
            raw, element_type=element, origin_shape=shape, layout_tag=tag
        )
        # MX scale slicing currently requires dynamic layout metadata in TLA.
        bindings.append(
            binding if grouped and not 4 <= i <= 6 else binding.mark_layout_dynamic()
        )
    scalars = tuple(
        tla.Int32(x) for x in (b, hq, hk, nq, nk, mh, block_indices.shape[-1])
    )
    grouped = xb == 128 and yb == 128
    if grouped:
        scalars = (b, hq, hk, nq, nk, mh, block_indices.shape[-1])
    bounded = False
    if (
        grouped
        and dim == 128
        and nq >= 8192
        and nk > 0
        and block_indices.shape[-1] >= 64
    ):
        bounded = scores_are_bounded(q, k, *scales[:2], scale)
    key = (
        str(q.device),
        output_dtype,
        block_indices.dtype,
        dim,
        xb,
        yb,
        return_lse,
        scale,
        bounded,
        tuple(int(x) for x in scalars) if grouped else (),
    )
    executor = _EXECUTORS.get(key)
    if executor is None:
        kernel = quant_block_sparse_attention_kernel
        if grouped:
            if block_indices.shape[-1] > 1:
                kernel = quant_block_sparse_attention_mxfp8_kernel
                if bounded:
                    kernel = quant_block_sparse_attention_bounded_kernel
            elif block_indices.shape[-1] > 0:
                kernel = quant_block_sparse_attention_continuous_kernel
            else:
                kernel = quant_block_sparse_attention_short_kernel
        executor = tla.compile(
            kernel,
            *bindings,
            *scalars,
            scale,
            dim,
            xb,
            yb,
            return_lse,
            options="--npu-arch 3510",
        )
        _EXECUTORS[key] = executor
    tasks = b * hq * ((nq + xb - 1) // xb) * (xb // (64 if xb % 128 else 128))
    cores = int(torch_npu.npu.get_device_properties(q.device).cube_core_num)
    executor(
        *bindings, *(() if grouped else scalars), block_num=min(cores, max(1, tasks))
    )
    return (out, lse) if return_lse else out


def quant_block_sparse_attn(
    q,
    k,
    v,
    block_indices,
    block_shape,
    q_dequant_scale,
    k_dequant_scale,
    v_dequant_scale,
    q_input_layout="BNSD",
    kv_input_layout=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    softmax_scale=None,
    output_dtype="bfloat16",
    return_lse=False,
):
    """MXFP8 sparse attention; block indices may be unsorted or repeated.

    Fixed contiguous BNSD inputs use one fused kernel. BSND/TND and padded
    batches use NPU layout copies around the same kernel.
    E8M0 scales and nonnegative indices must contain valid, in-range values.
    """
    return _run_layouts(
        _run_bnsd,
        q,
        k,
        v,
        block_indices,
        block_shape,
        q_dequant_scale,
        k_dequant_scale,
        v_dequant_scale,
        q_input_layout,
        kv_input_layout,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        softmax_scale,
        output_dtype,
        return_lse,
    )
