# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared C64 preparation and C53 recurrent device kernels."""

from catlass import tla


def cube_publish(cf):
    tla.cross_core_set_flag(cf, tla.arch.FIX, aiv_id=0)
    tla.cross_core_set_flag(cf, tla.arch.FIX, aiv_id=1)


def cube_set(cf):
    tla.cross_core_set_flag(cf, tla.arch.MTE1, aiv_id=0)
    tla.cross_core_set_flag(cf, tla.arch.MTE1, aiv_id=1)


def cube_wait(cf):
    tla.cross_core_wait_flag(cf, tla.arch.MTE1, aiv_id=0)
    tla.cross_core_wait_flag(cf, tla.arch.MTE1, aiv_id=1)


def gm_view(x, offset, rows, cols, pitch):
    return tla.make_tensor(
        x.ptr + offset,
        tla.make_layout(
            tla.make_shape(rows, cols),
            tla.make_stride(pitch, 1),
            layoutTag=tla.arch.RowMajor,
        ),
    )


COMPONENT_CAST = tla.params.CastParams(round_mode=tla.params.RoundMode.CAST_ROUND)

PACK16 = tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK_B32)


def component_high(value):
    limit = 3.3895313892515355e38
    return tla.min(tla.max(value, -limit), limit).to(tla.BFloat16, COMPONENT_CAST)


def store_components(value, high, low):
    # Clamp only the high component; keep the original value in the residual.
    hi = component_high(value)
    lo = (value - hi.to(tla.Float32, tla.params.CastParams())).to(
        tla.BFloat16, COMPONENT_CAST
    )
    high.store(hi, PACK16)
    low.store(lo, PACK16)


def sub(x, row, col, rows, cols):
    return tla.get_tile(x, tla.make_coord(row, col), tla.make_shape(rows, cols))


def tensor(ptr, rows, cols, dtype=tla.Float32, tag=tla.arch.RowMajor):
    return tla.make_tensor(
        tla.recast_ptr(ptr, dtype=dtype),
        tla.make_layout_with_tag((rows, cols), dtype, tag),
    )


@tla.jit
def export_w(whout, wlout, ub, f, ready):
    result = tensor(ub + 140 * 256, 32, 128)
    high = tensor(ub + 156 * 256, 32, 128, tla.BFloat16)
    low = tensor(ub + 164 * 256, 32, 128, tla.BFloat16)
    with tla.vec.func(mode="simd"):
        for row in tla.range(32):
            for part in tla.range_constexpr(2):
                value = (
                    tla.full(0.0, tla.Float32)
                    - sub(result, row, part * 64, 1, 64).load()
                )
                store_components(
                    value,
                    sub(high, row, part * 64, 1, 64),
                    sub(low, row, part * 64, 1, 64),
                )
    tla.set_flag(ready)
    tla.wait_flag(ready)
    off = tla.Int64(f) * tla.Int64(8192) + tla.Int64(
        tla.arch.sub_block_idx()
    ) * tla.Int64(4096)
    tla.copy(gm_view(whout, off, 32, 128, 128), high)
    tla.copy(gm_view(wlout, off, 32, 128, 128), low)


FIRST32 = tla.params.NormalStoreParams(
    store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
)

ZERO_CAST = tla.params.CastParams(round_mode=tla.params.RoundMode.CAST_ROUND)

UNPACK16 = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_UNPACK_B16)


def read_f32(x, bf16):
    if bf16:
        return x.load(UNPACK16).to(tla.Float32, tla.params.CastParams())
    return x.load()


def store_native_pair(value, high, low, row):
    hi = component_high(value)
    lo = (value - hi.to(tla.Float32, tla.params.CastParams())).to(
        tla.BFloat16, COMPONENT_CAST
    )
    native_hi, unused_hi = tla.deinterleave(hi, hi)
    native_lo, unused_lo = tla.deinterleave(lo, lo)
    active = tla.create_mask(pattern=tla.mask.VL64, dtype=tla.BFloat16)
    sub(high, 0, row * 16, 1, 128).store(
        native_hi, tla.params.BlockStoreParams(block_stride=64), mask=active
    )
    sub(low, 0, row * 16, 1, 128).store(
        native_lo, tla.params.BlockStoreParams(block_stride=64), mask=active
    )


@tla.jit
def s0_bg(k, b, ub, lp, koff, boff, valid, slot, b_bf16, ld, vd, sf, raw_reuse):
    col = tla.arch.sub_block_idx() * 64
    kr = tensor(ub, 64, 64, tla.BFloat16)
    br = tensor(ub + 8 * 256, 64, 64, tla.BFloat16 if b_bf16 else tla.Float32)
    prefix = tensor(ub + 24 * 256, 64, 64)
    original = tensor(ub + 56 * 256, 1, 4096)
    high = tensor(ub + 72 * 256, 1, 4096, tla.BFloat16)
    low = tensor(ub + 80 * 256, 1, 4096, tla.BFloat16)
    control = tensor(ub + 173 * 256, 1, 64)
    if valid < 64:
        with tla.vec.func(mode="simd"):
            zero = tla.full(0.0, tla.Float32)
            for pad_row in tla.range(valid, 64):
                sub(kr, pad_row, 0, 1, 64).store(
                    zero.to(tla.BFloat16, ZERO_CAST), PACK16
                )
                if b_bf16:
                    sub(br, pad_row, 0, 1, 64).store(
                        zero.to(tla.BFloat16, ZERO_CAST), PACK16
                    )
                else:
                    sub(br, pad_row, 0, 1, 64).store(zero)
    # Q/G and K/B intentionally share raw storage. Q reads must finish.
    tla.set_flag(raw_reuse)
    tla.wait_flag(raw_reuse)
    tla.copy(
        sub(kr, 0, 0, valid, 64), gm_view(k, koff + tla.Int64(col), valid, 64, 128)
    )
    tla.copy(
        sub(br, 0, 0, valid, 64), gm_view(b, boff + tla.Int64(col), valid, 64, 128)
    )
    tla.set_flag(ld)
    tla.wait_flag(ld)
    with tla.vec.func(mode="simd"):
        largest = tla.full(0.0, tla.Float32)
        for bg_row in tla.range(64):
            key = read_f32(sub(kr, bg_row, 0, 1, 64), True)
            beta = read_f32(sub(br, bg_row, 0, 1, 64), b_bf16)
            p = sub(prefix, bg_row, 0, 1, 64).load()
            value = (beta * key) * p
            sub(original, 0, bg_row * 8, 1, 64).store(
                value, tla.params.BlockStoreParams(block_stride=64)
            )
            store_native_pair(value, high, low, bg_row)
            largest = tla.max(largest, tla.abs(key))
        mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        sub(control, 0, slot * 4 + 2, 1, 1).store(
            largest.reduce(tla.ReductionOp.MAX, mask=mask), FIRST32
        )
    tla.set_flag(vd)
    tla.wait_flag(vd)
    original_dest = tensor(lp + (368 + 32 * slot) * 256, 64, 128, tag=tla.arch.zN)
    high_dest = tensor(lp + (184 * slot + 32) * 256, 64, 128, tla.BFloat16, tla.arch.zN)
    low_dest = tensor(lp + (184 * slot + 48) * 256, 64, 128, tla.BFloat16, tla.arch.zN)
    tla.copy(
        sub(original_dest, 0, col, 64, 64),
        tensor(ub + 56 * 256, 64, 64, tag=tla.arch.zN),
    )
    tla.copy(
        sub(high_dest, 0, col, 64, 64),
        tensor(ub + 72 * 256, 64, 64, tla.BFloat16, tla.arch.zN),
    )
    tla.copy(
        sub(low_dest, 0, col, 64, 64),
        tensor(ub + 80 * 256, 64, 64, tla.BFloat16, tla.arch.zN),
    )
    tla.set_flag(sf)
    tla.wait_flag(sf)


@tla.jit
def _safe_tile(ub, lp, slot, value, vd, sf):
    flag = tensor(ub + 136 * 256, 16, 8)
    with tla.vec.func(mode="simd"):
        zero = tla.full(0.0, tla.Float32)
        for flag_row in tla.range(1, 16):
            sub(flag, flag_row, 0, 1, 8).store(zero)
        indices = tla.arange(0, dtype=tla.Int32)
        sub(flag, 0, 0, 1, 8).store(
            tla.where(tla.cmp(indices, 0, "eq"), tla.full(value, tla.Float32), zero)
        )
    tla.set_flag(vd)
    tla.wait_flag(vd)
    bridge = tensor(lp + (slot * 184 + 128) * 256, 16, 16, tag=tla.arch.zN)
    tla.copy(sub(bridge, 0, tla.arch.sub_block_idx() * 8, 16, 8), flag)
    tla.set_flag(sf)
    tla.wait_flag(sf)


@tla.jit
def s0_ki(ub, lp, slot, vd, sf, vs):
    col = tla.arch.sub_block_idx() * 64
    kr = tensor(ub, 64, 64, tla.BFloat16)
    prefix = tensor(ub + 24 * 256, 64, 64)
    control = tensor(ub + 173 * 256, 1, 64)
    high = tensor(ub + 56 * 256, 1, 4096, tla.BFloat16)
    low = tensor(ub + 64 * 256, 1, 4096, tla.BFloat16)
    tla.set_flag(vs)
    tla.wait_flag(vs)
    safe = (
        (control[0, slot * 4] >= tla.Float32(2**-16))
        & (control[0, slot * 4 + 1] <= tla.Float32(2**30))
        & (control[0, slot * 4 + 2] <= tla.Float32(2**30))
    )
    if safe:
        with tla.vec.func(mode="simd"):
            for inv_row in tla.range(64):
                key = read_f32(sub(kr, inv_row, 0, 1, 64), True)
                p = sub(prefix, inv_row, 0, 1, 64).load()
                store_native_pair(key / p, high, low, inv_row)
    else:
        with tla.vec.func(mode="simd"):
            zero = tla.full(0.0, tla.Float32)
            for inv_zero_row in tla.range(64):
                store_native_pair(zero, high, low, inv_zero_row)
    tla.set_flag(vd)
    tla.wait_flag(vd)
    high_dest = tensor(lp + (184 * slot + 64) * 256, 64, 128, tla.BFloat16, tla.arch.zN)
    low_dest = tensor(lp + (184 * slot + 80) * 256, 64, 128, tla.BFloat16, tla.arch.zN)
    tla.copy(
        sub(high_dest, 0, col, 64, 64),
        tensor(ub + 56 * 256, 64, 64, tla.BFloat16, tla.arch.zN),
    )
    tla.copy(
        sub(low_dest, 0, col, 64, 64),
        tensor(ub + 64 * 256, 64, 64, tla.BFloat16, tla.arch.zN),
    )
    tla.set_flag(sf)
    tla.wait_flag(sf)
    if safe:
        _safe_tile(ub, lp, slot, 1.0, vd, sf)
    else:
        _safe_tile(ub, lp, slot, 0.0, vd, sf)


@tla.jit
def s0_kt(ub, lp, slot, vd, sf):
    kr = tensor(ub, 64, 64, tla.BFloat16)
    decay = tensor(ub + 40 * 256, 64, 64)
    high = tensor(ub + 72 * 256, 64, 64, tla.BFloat16)
    low = tensor(ub + 80 * 256, 64, 64, tla.BFloat16)
    with tla.vec.func(mode="simd"):
        suffix = tla.full(1.0, tla.Float32)
        for reverse in tla.range_constexpr(64):
            row = 63 - reverse
            value = read_f32(sub(kr, row, 0, 1, 64), True) * suffix
            hi = component_high(value)
            lo = (value - hi.to(tla.Float32, tla.params.CastParams())).to(
                tla.BFloat16, COMPONENT_CAST
            )
            sub(high, row, 0, 1, 64).store(hi, PACK16)
            sub(low, row, 0, 1, 64).store(lo, PACK16)
            suffix = suffix * sub(decay, row, 0, 1, 64).load()
    tla.set_flag(vd)
    tla.wait_flag(vd)


@tla.jit
def s0_qg(q, g, ub, lp, qoff, goff, valid, slot, g_bf16, ld, vd, sf, raw_reuse):
    col = tla.arch.sub_block_idx() * 64
    qr = tensor(ub, 64, 64, tla.BFloat16)
    gr = tensor(ub + 8 * 256, 64, 64, tla.BFloat16 if g_bf16 else tla.Float32)
    prefix = tensor(ub + 24 * 256, 64, 64)
    decay = tensor(ub + 40 * 256, 64, 64)
    packed = tensor(ub + 56 * 256, 64, 64, tla.BFloat16)
    native = tensor(ub + 176 * 256, 1, 4096, tla.BFloat16)
    end = tensor(ub + 172 * 256 + slot * 128, 1, 128)
    control = tensor(ub + 173 * 256, 1, 64)
    if valid < 64:
        with tla.vec.func(mode="simd"):
            zero = tla.full(0.0, tla.Float32)
            for pad_row in tla.range(valid, 64):
                sub(qr, pad_row, 0, 1, 64).store(
                    zero.to(tla.BFloat16, ZERO_CAST), PACK16
                )
                if g_bf16:
                    sub(gr, pad_row, 0, 1, 64).store(
                        zero.to(tla.BFloat16, ZERO_CAST), PACK16
                    )
                else:
                    sub(gr, pad_row, 0, 1, 64).store(zero)
    tla.set_flag(raw_reuse)
    tla.wait_flag(raw_reuse)
    tla.copy(
        sub(qr, 0, 0, valid, 64), gm_view(q, qoff + tla.Int64(col), valid, 64, 128)
    )
    tla.copy(
        sub(gr, 0, 0, valid, 64), gm_view(g, goff + tla.Int64(col), valid, 64, 128)
    )
    tla.set_flag(ld)
    tla.wait_flag(ld)
    with tla.vec.func(mode="simd"):
        for exp_row in tla.range(64):
            gate = read_f32(sub(gr, exp_row, 0, 1, 64), g_bf16)
            sub(decay, exp_row, 0, 1, 64).store(tla.exp(gate))
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        cumulative = tla.full(1.0, tla.Float32)
        for prefix_row in tla.range(64):
            cumulative = cumulative * sub(decay, prefix_row, 0, 1, 64).load()
            sub(prefix, prefix_row, 0, 1, 64).store(cumulative)
        sub(end, 0, col, 1, 64).store(cumulative)
        mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        sub(control, 0, slot * 4, 1, 1).store(
            cumulative.reduce(tla.ReductionOp.MIN, mask=mask), FIRST32
        )
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        largest = tla.full(0.0, tla.Float32)
        for query_row in tla.range(64):
            query = read_f32(sub(qr, query_row, 0, 1, 64), True)
            p = sub(prefix, query_row, 0, 1, 64).load()
            largest = tla.max(largest, tla.abs(query))
            encoded = component_high(query * p)
            sub(packed, query_row, 0, 1, 64).store(encoded, PACK16)
            native_value, unused = tla.deinterleave(encoded, encoded)
            active = tla.create_mask(pattern=tla.mask.VL64, dtype=tla.BFloat16)
            sub(native, 0, query_row * 16, 1, 128).store(
                native_value, tla.params.BlockStoreParams(block_stride=64), mask=active
            )
        sub(control, 0, slot * 4 + 1, 1, 1).store(
            largest.reduce(tla.ReductionOp.MAX, mask=mask), FIRST32
        )
    tla.set_flag(vd)
    tla.wait_flag(vd)
    dest = tensor(lp + slot * 184 * 256, 64, 128, tla.BFloat16, tla.arch.zN)
    tla.copy(
        sub(dest, 0, col, 64, 64),
        tensor(ub + 176 * 256, 64, 64, tla.BFloat16, tla.arch.zN),
    )
    tla.set_flag(sf)
    tla.wait_flag(sf)


@tla.jit
def prepare_factors(
    q,
    k,
    g,
    b,
    qout,
    kt0,
    kt1,
    endout,
    ub,
    lp,
    f,
    slot,
    length,
    hk,
    hf,
    hg,
    hb,
    g_bf16,
    b_bf16,
    ld,
    vd,
    sf,
    vs,
    raw_reuse,
    phase=0,
):
    blocks = (length + 63) // 64
    block = f % blocks
    head = f // blocks
    batch = head // hf
    fh = head % hf
    kh = fh // (hf // hk)
    gh = kh if hg == hk else fh
    bh = kh if hb == hk else fh
    start = block * 64
    valid = length - start
    if valid > 64:
        valid = 64
    qoff = (
        tla.Int64(batch) * tla.Int64(hk) * tla.Int64(length)
        + tla.Int64(kh) * tla.Int64(length)
        + tla.Int64(start)
    ) * tla.Int64(128)
    goff = (
        tla.Int64(batch) * tla.Int64(hg) * tla.Int64(length)
        + tla.Int64(gh) * tla.Int64(length)
        + tla.Int64(start)
    ) * tla.Int64(128)
    boff = (
        tla.Int64(batch) * tla.Int64(hb) * tla.Int64(length)
        + tla.Int64(bh) * tla.Int64(length)
        + tla.Int64(start)
    ) * tla.Int64(128)
    col = tla.arch.sub_block_idx() * 64
    off = tla.Int64(f) * tla.Int64(8192) + tla.Int64(col)
    if phase == 0 or phase == 1:
        s0_qg(q, g, ub, lp, qoff, goff, valid, slot, g_bf16, ld, vd, sf, raw_reuse)
        tla.copy(
            gm_view(qout, off, 64, 64, 128), tensor(ub + 56 * 256, 64, 64, tla.BFloat16)
        )

        tla.set_flag(sf)
        tla.wait_flag(sf)

    if phase == 0 or phase == 2:
        s0_bg(k, b, ub, lp, qoff, boff, valid, slot, b_bf16, ld, vd, sf, raw_reuse)
    if phase == 0 or phase == 3:
        s0_ki(ub, lp, slot, vd, sf, vs)
        s0_kt(ub, lp, slot, vd, sf)
        tla.copy(
            gm_view(kt0, off, 64, 64, 128), tensor(ub + 72 * 256, 64, 64, tla.BFloat16)
        )
        tla.copy(
            gm_view(kt1, off, 64, 64, 128), tensor(ub + 80 * 256, 64, 64, tla.BFloat16)
        )
        end = tensor(ub + 172 * 256 + slot * 128, 1, 128)
        tla.copy(
            gm_view(endout, tla.Int64(f) * tla.Int64(128) + tla.Int64(col), 1, 64, 128),
            sub(end, 0, col, 1, 64),
        )
        tla.set_flag(sf)
        tla.wait_flag(sf)


@tla.jit
def prepare_w(ub, lp, ap, bp, cp, slot, m1m, mm1, mf, fm):
    a = tensor(ap, 64, 64, tag=tla.arch.zN)
    b = tensor(bp, 64, 128, tag=tla.arch.nZ)
    c = tensor(cp, 64, 128, tag=tla.arch.L0Clayout)
    tla.copy(a, tensor(lp + (slot * 184 + 164) * 256, 64, 64, tag=tla.arch.zN))
    tla.copy(b, tensor(lp + (368 + 32 * slot) * 256, 64, 128, tag=tla.arch.zN))
    tla.set_flag(m1m)
    tla.wait_flag(m1m)
    tla.mmad(c, a, b, init_c=True, hf32_mode=tla.params.HF32Mode.HF32_DISABLE)
    tla.set_flag(mf)
    tla.wait_flag(mf)
    tla.copy(
        tensor(ub + 140 * 256, 64, 128),
        c,
        tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
    )
    tla.set_flag(mm1)
    tla.wait_flag(mm1)
    tla.set_flag(fm)
    tla.wait_flag(fm)


def prepare_component_mmad(
    lhs_ptr, rhs_ptr, ap, bp, cp, transpose_rhs, m1m, mm1, terms=3, init_first=True
):
    """C64xK128 times K128xN64 with two BF16 planes of 16 KiB each."""
    c = tensor(cp, 64, 64, tag=tla.arch.L0Clayout)
    for term in tla.range_constexpr(terms):
        left_part = term // 2
        right_part = term % 2
        lhs = tensor(lhs_ptr + left_part * 16 * 256, 64, 128, tla.BFloat16, tla.arch.zN)
        rhs = tensor(
            rhs_ptr + right_part * 16 * 256,
            128,
            64,
            tla.BFloat16,
            tla.arch.nZ if transpose_rhs else tla.arch.zN,
        )
        a = tensor(ap, 64, 128, tla.BFloat16, tla.arch.zN)
        b = tensor(bp, 128, 64, tla.BFloat16, tla.arch.nZ)
        tla.copy(a, lhs)
        tla.copy(b, rhs)
        tla.set_flag(m1m)
        tla.wait_flag(m1m)
        tla.mmad(
            c,
            a,
            b,
            init_c=(init_first and term == 0),
            hf32_mode=tla.params.HF32Mode.HF32_DISABLE,
        )
        tla.set_flag(mm1)
        tla.wait_flag(mm1)


@tla.jit
def s1_scores(ub, lp, ap, bp, cp, slot, m1m, mf, mm1, fm):
    for product in tla.range_constexpr(2):
        c = tensor(cp + product * 32 * 256, 64, 64, tag=tla.arch.L0Clayout)
        score = tensor(ub + (88 + product * 16) * 256, 64, 64)
        prepare_component_mmad(
            lp + (slot * 184 + product * 32) * 256,
            lp + (slot * 184 + 64) * 256,
            ap + product * 16 * 256,
            bp + product * 32 * 256,
            cp + product * 32 * 256,
            True,
            m1m,
            mm1,
            1 if product == 0 else 3,
        )
        tla.set_flag(mf)
        tla.wait_flag(mf)
        tla.copy(
            score,
            c,
            tla.params.CopyL0C2DstParams(
                l0c2ub_mode=tla.params.L0C2UBMode.NO_SPLIT_VEC_0
            ),
        )
        tla.copy(
            score,
            c,
            tla.params.CopyL0C2DstParams(
                l0c2ub_mode=tla.params.L0C2UBMode.NO_SPLIT_VEC_1
            ),
        )
        tla.set_flag(fm)
        tla.wait_flag(fm)
    fa = tensor(ap, 16, 16, tag=tla.arch.zN)
    fb = tensor(bp, 16, 16, tag=tla.arch.nZ)
    fc = tensor(cp, 16, 16, tag=tla.arch.L0Clayout)
    tla.copy(fa, tensor(lp + (slot * 184 + 128) * 256, 16, 16, tag=tla.arch.zN))
    tla.copy(fb, tensor(lp + (slot * 184 + 128) * 256, 16, 16, tag=tla.arch.nZ))
    tla.set_flag(m1m)
    tla.wait_flag(m1m)
    tla.mmad(fc, fa, fb, init_c=True, hf32_mode=tla.params.HF32Mode.HF32_DISABLE)
    tla.set_flag(mf)
    tla.wait_flag(mf)
    flags = tensor(ub + 136 * 256, 16, 16)
    tla.copy(
        flags,
        fc,
        tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.NO_SPLIT_VEC_0),
    )
    tla.copy(
        flags,
        fc,
        tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.NO_SPLIT_VEC_1),
    )
    tla.set_flag(mm1)
    tla.wait_flag(mm1)
    tla.set_flag(fm)
    tla.wait_flag(fm)


BROADCAST32 = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)


@tla.jit
def s2_leaves(ub, lp, slot, ready):
    lower = tensor(ub + 104 * 256, 64, 64)
    inverse = tensor(ub + 120 * 256, 64, 64)
    row0 = tla.arch.sub_block_idx() * 32
    native_lower = tensor(ub + 176 * 256, 1, 4096)
    native_inverse = tensor(ub + 192 * 256, 1, 2048)
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_LOAD, tla.params.MemType.VEC_STORE)
        indices = tla.arange(0, dtype=tla.Int32)
        zero = tla.full(0.0, tla.Float32)
        one = tla.full(1.0, tla.Float32)
        upper_mask = tla.cmp(indices, row0 + 16, "lt")
        for leaf_row in tla.range(16):
            inverse_row = row0 + leaf_row
            result = tla.where(
                tla.cmp(indices, inverse_row, "eq"),
                one,
                tla.where(tla.cmp(indices, inverse_row + 16, "eq"), one, zero),
            )
            for previous in tla.range(leaf_row):
                previous_row = row0 + previous
                c0 = sub(lower, inverse_row, previous_row, 1, 1).load(BROADCAST32)
                c1 = sub(lower, inverse_row + 16, previous_row + 16, 1, 1).load(
                    BROADCAST32
                )
                coefficient = tla.where(upper_mask, c0, c1)
                row_values = sub(inverse, previous_row, 0, 1, 64).load()
                result = result - coefficient * row_values
            sub(inverse, inverse_row, 0, 1, 64).store(result)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        for expand_row in tla.range(16):
            packed = sub(inverse, row0 + expand_row, 0, 1, 64).load()
            upper = tla.where(upper_mask, packed, zero)
            lower_leaf = tla.where(upper_mask, zero, packed)
            tla.local_mem_bar(tla.params.MemType.VEC_LOAD, tla.params.MemType.VEC_STORE)
            sub(inverse, row0 + expand_row, 0, 1, 64).store(upper)
            sub(inverse, row0 + 16 + expand_row, 0, 1, 64).store(lower_leaf)
            sub(native_inverse, 0, expand_row * 8, 1, 64).store(
                upper, tla.params.BlockStoreParams(block_stride=32)
            )
            sub(native_inverse, 0, (expand_row + 16) * 8, 1, 64).store(
                lower_leaf, tla.params.BlockStoreParams(block_stride=32)
            )
        if row0 == 0:
            for lower_row in tla.range(64):
                values = sub(lower, lower_row, 0, 1, 64).load()
                sub(native_lower, 0, lower_row * 8, 1, 64).store(
                    tla.where(
                        tla.cmp(indices, (lower_row // 16) * 16, "lt"),
                        zero - values,
                        values,
                    ),
                    tla.params.BlockStoreParams(block_stride=64),
                )
    tla.set_flag(ready)
    tla.wait_flag(ready)
    if row0 == 0:
        tla.copy(
            tensor(lp + (slot * 184 + 148) * 256, 64, 64, tag=tla.arch.zN),
            tensor(ub + 176 * 256, 64, 64, tag=tla.arch.zN),
        )
    dest = tensor(lp + (slot * 184 + 164) * 256, 64, 64, tag=tla.arch.zN)
    tla.copy(
        sub(dest, row0, 0, 32, 64), tensor(ub + 192 * 256, 32, 64, tag=tla.arch.zN)
    )


@tla.jit
def s2_output_score(ub, lp, slot):
    if tla.arch.sub_block_idx() == 0:
        tla.copy(
            tensor(lp + (slot * 184 + 132) * 256, 64, 64, tag=tla.arch.zN),
            tensor(ub + 88 * 256, 64, 64),
        )


@tla.jit
def s2_scores(
    q, k, g, b, ub, f, length, hk, hf, hg, hb, g_bf16, b_bf16, vs, ld, raw_reuse
):
    aq = tensor(ub + 88 * 256, 64, 64)
    lower = tensor(ub + 104 * 256, 64, 64)
    flags = tensor(ub + 136 * 256, 16, 16)
    tla.set_flag(vs)
    tla.wait_flag(vs)
    safe = flags[0, 0] == tla.Float32(2.0)
    if not safe:
        # The preceding W export may still read 140--172 KiB on MTE3.
        # The next work's prefix/decay/control have separate, live storage.
        tla.pipe_barrier(tla.pipes.ALL)
        blocks = (length + 63) // 64
        block = f % blocks
        head = f // blocks
        batch = head // hf
        fh = head % hf
        kh = fh // (hf // hk)
        gh = kh if hg == hk else fh
        bh = kh if hb == hk else fh
        valid = length - block * 64
        if valid > 64:
            valid = 64
        qoff = (
            (tla.Int64(batch) * tla.Int64(hk) + tla.Int64(kh)) * tla.Int64(length)
            + tla.Int64(block) * tla.Int64(64)
        ) * tla.Int64(128)
        goff = (
            (tla.Int64(batch) * tla.Int64(hg) + tla.Int64(gh)) * tla.Int64(length)
            + tla.Int64(block) * tla.Int64(64)
        ) * tla.Int64(128)
        boff = (
            (tla.Int64(batch) * tla.Int64(hb) + tla.Int64(bh)) * tla.Int64(length)
            + tla.Int64(block) * tla.Int64(64)
        ) * tla.Int64(128)
        kr = tensor(ub, 64, 128, tla.BFloat16)
        qr = tensor(ub + 176 * 256, 64, 128, tla.BFloat16)
        br = tensor(ub + 192 * 256, 64, 128, tla.BFloat16 if b_bf16 else tla.Float32)
        gr = tensor(ub + 140 * 256, 64, 128, tla.BFloat16 if g_bf16 else tla.Float32)
        decay = tensor(ub + 140 * 256, 64, 128)
        if valid < 64:
            with tla.vec.func(mode="simd"):
                fallback_zero = tla.full(0.0, tla.Float32)
                for pad_row in tla.range(valid, 64):
                    for pad_part in tla.range_constexpr(2):
                        sub(kr, pad_row, pad_part * 64, 1, 64).store(
                            fallback_zero.to(tla.BFloat16, ZERO_CAST), PACK16
                        )
                        sub(qr, pad_row, pad_part * 64, 1, 64).store(
                            fallback_zero.to(tla.BFloat16, ZERO_CAST), PACK16
                        )
                        if b_bf16:
                            sub(br, pad_row, pad_part * 64, 1, 64).store(
                                fallback_zero.to(tla.BFloat16, ZERO_CAST), PACK16
                            )
                        else:
                            sub(br, pad_row, pad_part * 64, 1, 64).store(fallback_zero)
                        if g_bf16:
                            sub(gr, pad_row, pad_part * 64, 1, 64).store(
                                fallback_zero.to(tla.BFloat16, ZERO_CAST), PACK16
                            )
                        else:
                            sub(gr, pad_row, pad_part * 64, 1, 64).store(fallback_zero)
        tla.set_flag(raw_reuse)
        tla.wait_flag(raw_reuse)
        tla.copy(sub(kr, 0, 0, valid, 128), gm_view(k, qoff, valid, 128, 128))
        tla.copy(sub(qr, 0, 0, valid, 128), gm_view(q, qoff, valid, 128, 128))
        tla.copy(sub(br, 0, 0, valid, 128), gm_view(b, boff, valid, 128, 128))
        tla.copy(sub(gr, 0, 0, valid, 128), gm_view(g, goff, valid, 128, 128))
        tla.set_flag(ld)
        tla.wait_flag(ld)
        with tla.vec.func(mode="simd"):
            for reverse_row in tla.range(64):
                gate_row = 63 - reverse_row
                gate0 = read_f32(sub(gr, gate_row, 0, 1, 64), g_bf16)
                gate1 = read_f32(sub(gr, gate_row, 64, 1, 64), g_bf16)
                if g_bf16:
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_LOAD, tla.params.MemType.VEC_STORE
                    )
                sub(decay, gate_row, 0, 1, 64).store(tla.exp(gate0))
                sub(decay, gate_row, 64, 1, 64).store(tla.exp(gate1))
                if g_bf16:
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
        with tla.vec.func(mode="simd"):
            all_lanes = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            for fallback_row in tla.range(64):
                for fallback_part in tla.range_constexpr(2):
                    qrow = read_f32(
                        sub(qr, fallback_row, fallback_part * 64, 1, 64), True
                    )
                    krow = read_f32(
                        sub(kr, fallback_row, fallback_part * 64, 1, 64), True
                    )
                    brow = read_f32(
                        sub(br, fallback_row, fallback_part * 64, 1, 64), b_bf16
                    )
                    bk = brow * krow
                    product = tla.full(1.0, tla.Float32)
                    for reverse_j in tla.range(fallback_row + 1):
                        j = fallback_row - reverse_j
                        kj = read_f32(sub(kr, j, fallback_part * 64, 1, 64), True)
                        weighted = kj * product
                        qp = (qrow * weighted).reduce(
                            tla.ReductionOp.ADD, mask=all_lanes
                        )
                        lp = (bk * weighted).reduce(tla.ReductionOp.ADD, mask=all_lanes)
                        if fallback_part == 0:
                            sub(aq, fallback_row, j, 1, 1).store(qp, FIRST32)
                            sub(lower, fallback_row, j, 1, 1).store(lp, FIRST32)
                        else:
                            oldq = sub(aq, fallback_row, j, 1, 1).load(BROADCAST32)
                            oldl = sub(lower, fallback_row, j, 1, 1).load(BROADCAST32)
                            sub(aq, fallback_row, j, 1, 1).store(
                                oldq + tla.full(qp, tla.Float32), FIRST32
                            )
                            sub(lower, fallback_row, j, 1, 1).store(
                                oldl + tla.full(lp, tla.Float32), FIRST32
                            )
                        product = (
                            product * sub(decay, j, fallback_part * 64, 1, 64).load()
                        )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
    with tla.vec.func(mode="simd"):
        indices = tla.arange(0, dtype=tla.Int32)
        zeros = tla.full(0.0, tla.Float32)
        for mask_row in tla.range(64):
            qvalue = sub(aq, mask_row, 0, 1, 64).load()
            lvalue = sub(lower, mask_row, 0, 1, 64).load()
            sub(aq, mask_row, 0, 1, 64).store(
                tla.where(tla.cmp(indices, mask_row, "le"), qvalue, zeros)
            )
            sub(lower, mask_row, 0, 1, 64).store(
                tla.where(tla.cmp(indices, mask_row, "lt"), lvalue, zeros)
            )


@tla.jit
def s3_inverse_products(lp, ap, bp, cp, slot, m1m, mf, mm1, fm, fix_l1):
    inverse = tensor(lp + (slot * 184 + 164) * 256, 64, 64, tag=tla.arch.zN)
    lower = tensor(lp + (slot * 184 + 148) * 256, 64, 64, tag=tla.arch.zN)
    y = tensor(lp + (slot * 184 + 180) * 256, 32, 32, tag=tla.arch.zN)
    for pair in tla.range_constexpr(2):
        start = pair * 32
        a = tensor(ap, 16, 16, tag=tla.arch.zN)
        b = tensor(bp, 16, 16, tag=tla.arch.nZ)
        c = tensor(cp, 16, 16, tag=tla.arch.L0Clayout)
        tla.copy(a, sub(inverse, start + 16, start + 16, 16, 16))
        tla.copy(b, sub(lower, start + 16, start, 16, 16))
        tla.set_flag(m1m)
        tla.wait_flag(m1m)
        tla.mmad(c, a, b, init_c=True, hf32_mode=tla.params.HF32Mode.HF32_DISABLE)
        tla.set_flag(mf)
        tla.wait_flag(mf)
        tla.copy(sub(y, pair * 16, pair * 16, 16, 16), c)
        tla.set_flag(fix_l1)
        tla.wait_flag(fix_l1)
        tla.set_flag(mm1)
        tla.wait_flag(mm1)
        tla.set_flag(fm)
        tla.wait_flag(fm)


@tla.jit
def s4_merge_pairs(lp, ap, bp, cp, slot, aout, f, m1m, mf, mm1, fm, fix_l1):
    inverse = tensor(lp + (slot * 184 + 164) * 256, 64, 64, tag=tla.arch.zN)
    y = tensor(lp + (slot * 184 + 180) * 256, 32, 32, tag=tla.arch.zN)
    for pair in tla.range_constexpr(2):
        start = pair * 32
        a = tensor(ap, 16, 16, tag=tla.arch.zN)
        b = tensor(bp, 16, 16, tag=tla.arch.nZ)
        c = tensor(cp, 16, 16, tag=tla.arch.L0Clayout)
        tla.copy(a, sub(y, pair * 16, pair * 16, 16, 16))
        tla.copy(b, sub(inverse, start, start, 16, 16))
        tla.set_flag(m1m)
        tla.wait_flag(m1m)
        tla.mmad(c, a, b, init_c=True, hf32_mode=tla.params.HF32Mode.HF32_DISABLE)
        tla.set_flag(mf)
        tla.wait_flag(mf)
        tla.copy(sub(inverse, start + 16, start, 16, 16), c)
        offset = tla.Int64(f) * tla.Int64(4096) + tla.Int64((start + 16) * 64 + start)
        tla.copy(gm_view(aout, offset, 16, 16, 64), c)
        tla.set_flag(fix_l1)
        tla.wait_flag(fix_l1)
        tla.set_flag(mm1)
        tla.wait_flag(mm1)
        tla.set_flag(fm)
        tla.wait_flag(fm)


@tla.jit
def s5_inverse_product(lp, ap, bp, cp, slot, m1m, mf, mm1, fm, fix_l1):
    inverse = tensor(lp + (slot * 184 + 164) * 256, 64, 64, tag=tla.arch.zN)
    lower = tensor(lp + (slot * 184 + 148) * 256, 64, 64, tag=tla.arch.zN)
    y = tensor(lp + (slot * 184 + 180) * 256, 32, 32, tag=tla.arch.zN)
    a = tensor(ap, 32, 32, tag=tla.arch.zN)
    b = tensor(bp, 32, 32, tag=tla.arch.nZ)
    c = tensor(cp, 32, 32, tag=tla.arch.L0Clayout)
    tla.copy(a, sub(inverse, 32, 32, 32, 32))
    tla.copy(b, sub(lower, 32, 0, 32, 32))
    tla.set_flag(m1m)
    tla.wait_flag(m1m)
    tla.mmad(c, a, b, init_c=True, hf32_mode=tla.params.HF32Mode.HF32_DISABLE)
    tla.set_flag(mf)
    tla.wait_flag(mf)
    tla.copy(y, c)
    tla.set_flag(fix_l1)
    tla.wait_flag(fix_l1)
    tla.set_flag(mm1)
    tla.wait_flag(mm1)
    tla.set_flag(fm)
    tla.wait_flag(fm)


@tla.jit
def s6_merge_inverse(lp, ap, bp, cp, slot, aout, f, m1m, mf, mm1, fm, fix_l1):
    inverse = tensor(lp + (slot * 184 + 164) * 256, 64, 64, tag=tla.arch.zN)
    y = tensor(lp + (slot * 184 + 180) * 256, 32, 32, tag=tla.arch.zN)
    a = tensor(ap, 32, 32, tag=tla.arch.zN)
    b = tensor(bp, 32, 32, tag=tla.arch.nZ)
    c = tensor(cp, 32, 32, tag=tla.arch.L0Clayout)
    tla.copy(a, y)
    tla.copy(b, sub(inverse, 0, 0, 32, 32))
    tla.set_flag(m1m)
    tla.wait_flag(m1m)
    tla.mmad(c, a, b, init_c=True, hf32_mode=tla.params.HF32Mode.HF32_DISABLE)
    tla.set_flag(mf)
    tla.wait_flag(mf)
    tla.copy(sub(inverse, 32, 0, 32, 32), c)
    tla.copy(
        gm_view(aout, tla.Int64(f) * tla.Int64(4096) + tla.Int64(2048), 32, 32, 64), c
    )
    tla.set_flag(fix_l1)
    tla.wait_flag(fix_l1)
    tla.set_flag(mm1)
    tla.wait_flag(mm1)
    tla.set_flag(fm)
    tla.wait_flag(fm)


@tla.jit
def vector_set(cf):
    if tla.arch.sub_block_idx() == 0:
        tla.cross_core_set_flag(cf, tla.arch.MTE3, aiv_id=0)
    else:
        tla.cross_core_set_flag(cf, tla.arch.MTE3, aiv_id=1)


@tla.jit
def vector_wait(cf):
    if tla.arch.sub_block_idx() == 0:
        tla.cross_core_wait_flag(cf, tla.arch.VECTOR, aiv_id=0)
    else:
        tla.cross_core_wait_flag(cf, tla.arch.VECTOR, aiv_id=1)


@tla.kernel
def prepare_kernel(
    q: tla.Tensor,
    k: tla.Tensor,
    g: tla.Tensor,
    b: tla.Tensor,
    qout: tla.Tensor,
    kt0: tla.Tensor,
    kt1: tla.Tensor,
    endout: tla.Tensor,
    aqout: tla.Tensor,
    aout: tla.Tensor,
    whout: tla.Tensor,
    wlout: tla.Tensor,
    length: tla.Constexpr[int],
    hk: tla.Constexpr[int],
    hf: tla.Constexpr[int],
    hg: tla.Constexpr[int],
    hb: tla.Constexpr[int],
    factors: tla.Constexpr[int],
    g_bf16: tla.Constexpr[bool],
    b_bf16: tla.Constexpr[bool],
):
    ub = tla.allocate(224 * 256, tla.Float32, tla.AddressSpace.ub, 256)
    lp = tla.allocate(432 * 256, tla.Float32, tla.AddressSpace.l1, 512)
    ap = tla.allocate(32 * 256, tla.Float32, tla.AddressSpace.l0a, 512)
    bp = tla.allocate(64 * 256, tla.Float32, tla.AddressSpace.l0b, 512)
    cp = tla.allocate(64 * 256, tla.Float32, tla.AddressSpace.l0c, 512)
    m1m = tla.flag("m1m", tla.arch.MTE1, tla.arch.CUBE)
    mm1 = tla.flag("mm1", tla.arch.CUBE, tla.arch.MTE1)
    mf = tla.flag("mf", tla.arch.CUBE, tla.arch.FIX)
    fm = tla.flag("fm", tla.arch.FIX, tla.arch.CUBE)
    fix_l1 = tla.flag("fix_l1", tla.arch.FIX, tla.arch.MTE1)
    p0 = tla.cross_flag("P0", mode=4)
    p1 = tla.cross_flag("P1", mode=4)
    qf = tla.cross_flag("Q", mode=4)
    l0 = tla.cross_flag("leaf0", mode=4)
    l1 = tla.cross_flag("leaf1", mode=4)
    wf = tla.cross_flag("W", mode=4)
    score_ready = tla.flag("score_ready", tla.arch.VECTOR, tla.arch.MTE3)
    w_ready = tla.flag("w_ready", tla.arch.VECTOR, tla.arch.MTE3)
    ld = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vd = tla.flag("vectors_done", tla.arch.VECTOR, tla.arch.MTE3)
    sf = tla.flag("staging_free", tla.arch.MTE3, tla.arch.VECTOR)
    vs = tla.flag("branch_ready", tla.arch.VECTOR, tla.arch.SCALAR)
    raw_reuse = tla.flag("raw_reuse", tla.arch.VECTOR, tla.arch.MTE2)
    count = (
        factors - tla.arch.block_idx() + tla.arch.block_num() - 1
    ) // tla.arch.block_num()
    with tla.cube():
        cube_set(p0)
        cube_set(p1)
        for iteration in tla.range(count):
            slot = iteration % 2
            if slot == 0:
                cube_wait(p0)
            else:
                cube_wait(p1)
            cube_wait(qf)
            s1_scores(ub, lp, ap, bp, cp, slot, m1m, mf, mm1, fm)
            cube_publish(qf)
            if slot == 0:
                cube_wait(l0)
            else:
                cube_wait(l1)
            f = tla.arch.block_idx() + iteration * tla.arch.block_num()
            s3_inverse_products(lp, ap, bp, cp, slot, m1m, mf, mm1, fm, fix_l1)
            s4_merge_pairs(lp, ap, bp, cp, slot, aout, f, m1m, mf, mm1, fm, fix_l1)
            s5_inverse_product(lp, ap, bp, cp, slot, m1m, mf, mm1, fm, fix_l1)
            s6_merge_inverse(lp, ap, bp, cp, slot, aout, f, m1m, mf, mm1, fm, fix_l1)
            cube_wait(wf)
            prepare_w(ub, lp, ap, bp, cp, slot, m1m, mm1, mf, fm)
            cube_publish(wf)
            if iteration + 1 < count:
                if slot == 0:
                    cube_set(p0)
                else:
                    cube_set(p1)
        cube_wait(qf)
        cube_wait(wf)
        if (count - 1) % 2 == 0:
            cube_set(p0)
        else:
            cube_set(p1)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        vector_set(qf)
        vector_set(wf)
        vector_wait(p0)
        prepare_factors(
            q,
            k,
            g,
            b,
            qout,
            kt0,
            kt1,
            endout,
            ub,
            lp,
            tla.arch.block_idx(),
            0,
            length,
            hk,
            hf,
            hg,
            hb,
            g_bf16,
            b_bf16,
            ld,
            vd,
            sf,
            vs,
            raw_reuse,
        )
        vector_set(p0)
        for iteration in tla.range(count):
            next_f = tla.arch.block_idx() + (iteration + 1) * tla.arch.block_num()
            ns = (iteration + 1) % 2
            if iteration + 1 < count:
                if ns == 0:
                    vector_wait(p0)
                else:
                    vector_wait(p1)
                prepare_factors(
                    q,
                    k,
                    g,
                    b,
                    qout,
                    kt0,
                    kt1,
                    endout,
                    ub,
                    lp,
                    next_f,
                    ns,
                    length,
                    hk,
                    hf,
                    hg,
                    hb,
                    g_bf16,
                    b_bf16,
                    ld,
                    vd,
                    sf,
                    vs,
                    raw_reuse,
                    1,
                )
            vector_wait(qf)
            f = tla.arch.block_idx() + iteration * tla.arch.block_num()
            s2_scores(
                q,
                k,
                g,
                b,
                ub,
                f,
                length,
                hk,
                hf,
                hg,
                hb,
                g_bf16,
                b_bf16,
                vs,
                ld,
                raw_reuse,
            )
            slot = iteration % 2
            s2_leaves(ub, lp, slot, score_ready)
            row = tla.arch.sub_block_idx() * 32
            aoffset = tla.Int64(f) * tla.Int64(4096) + tla.Int64(row) * tla.Int64(64)
            tla.copy(
                gm_view(aout, aoffset, 32, 64, 64),
                sub(tensor(ub + 120 * 256, 64, 64), row, 0, 32, 64),
            )
            if row == 0:
                tla.copy(
                    gm_view(aqout, tla.Int64(f) * tla.Int64(4096), 64, 64, 64),
                    tensor(ub + 88 * 256, 64, 64),
                )
            if slot == 0:
                vector_set(l0)
            else:
                vector_set(l1)
            s2_output_score(ub, lp, slot)
            vector_set(qf)
            if iteration + 1 < count:
                prepare_factors(
                    q,
                    k,
                    g,
                    b,
                    qout,
                    kt0,
                    kt1,
                    endout,
                    ub,
                    lp,
                    next_f,
                    ns,
                    length,
                    hk,
                    hf,
                    hg,
                    hb,
                    g_bf16,
                    b_bf16,
                    ld,
                    vd,
                    sf,
                    vs,
                    raw_reuse,
                    2,
                )
            if iteration + 1 < count:
                prepare_factors(
                    q,
                    k,
                    g,
                    b,
                    qout,
                    kt0,
                    kt1,
                    endout,
                    ub,
                    lp,
                    next_f,
                    ns,
                    length,
                    hk,
                    hf,
                    hg,
                    hb,
                    g_bf16,
                    b_bf16,
                    ld,
                    vd,
                    sf,
                    vs,
                    raw_reuse,
                    3,
                )
                if ns == 0:
                    vector_set(p0)
                else:
                    vector_set(p1)
            vector_wait(wf)
            export_w(whout, wlout, ub, f, w_ready)
            vector_set(wf)
        vector_wait(p0)
        vector_wait(p1)
        tla.pipe_barrier(tla.pipes.ALL)


def recurrent_component_mmad(
    lhs_ptr, rhs_ptr, ap, bp, cp, width, m1m, mm1, terms=4, init_first=False
):
    c = tensor(cp, 64, width, tag=tla.arch.L0Clayout)
    for term in tla.range_constexpr(terms):
        lhs = tensor(
            lhs_ptr + (term // 2) * 16 * 256, 64, 128, tla.BFloat16, tla.arch.zN
        )
        rhs = tensor(
            rhs_ptr + (term % 2) * (width // 4) * 256,
            128,
            width,
            tla.BFloat16,
            tla.arch.zN,
        )
        a = tensor(ap, 64, 128, tla.BFloat16, tla.arch.zN)
        b = tensor(bp, 128, width, tla.BFloat16, tla.arch.nZ)
        tla.copy(a, lhs)
        tla.copy(b, rhs)
        tla.set_flag(m1m)
        tla.wait_flag(m1m)
        tla.mmad(
            c,
            a,
            b,
            init_c=(init_first and term == 0),
            hf32_mode=tla.params.HF32Mode.HF32_DISABLE,
        )
        tla.set_flag(mm1)
        tla.wait_flag(mm1)


@tla.jit
def compute_output(ub, lp, ap, bp, cp, slot, width, m1m, mf, mm1, fm):
    c = tensor(cp, 64, width, tag=tla.arch.L0Clayout)
    recurrent_component_mmad(
        lp + slot * 128 * 256, lp + (256) * 256, ap, bp, cp, width, m1m, mm1, 1, True
    )
    a = tensor(ap, 64, 64, tag=tla.arch.zN)
    b = tensor(bp, 64, width, tag=tla.arch.nZ)
    tla.copy(a, tensor(lp + (slot * 128 + 96) * 256, 64, 64, tag=tla.arch.zN))
    tla.copy(b, tensor(lp + (384 + 32 * slot) * 256, 64, width, tag=tla.arch.zN))
    tla.set_flag(m1m)
    tla.wait_flag(m1m)
    tla.mmad(c, a, b, init_c=False, hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN)
    tla.set_flag(mf)
    tla.wait_flag(mf)
    tla.copy(
        tensor(ub + 208 * 256, 64, width),
        c,
        tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
    )
    tla.set_flag(mm1)
    tla.wait_flag(mm1)
    tla.set_flag(fm)
    tla.wait_flag(fm)


@tla.jit
def compute_state_product(ub, lp, ap, bp, cp, slot, width, m1m, mf, mm1, fm):
    # Both Kt and all three delta planes remain resident through all five terms.
    for plane in tla.range_constexpr(3):
        if plane < 2:
            left = tensor(ap + plane * 16 * 256, 128, 64, tla.BFloat16, tla.arch.zN)
            tla.copy(
                left,
                tensor(
                    lp + (slot * 128 + 48 + plane * 16) * 256,
                    128,
                    64,
                    tla.BFloat16,
                    tla.arch.nZ,
                ),
            )
        right = tensor(
            bp + plane * (width // 8) * 256, 64, width, tla.BFloat16, tla.arch.nZ
        )
        tla.copy(
            right,
            tensor(
                lp + (448 + plane * (width // 8)) * 256,
                64,
                width,
                tla.BFloat16,
                tla.arch.zN,
            ),
        )
    tla.set_flag(m1m)
    tla.wait_flag(m1m)
    c = tensor(cp + 32 * 256, 128, width, tag=tla.arch.L0Clayout)
    for term in tla.range_constexpr(4):
        a = tensor(ap + (term // 2) * 16 * 256, 128, 64, tla.BFloat16, tla.arch.zN)
        b = tensor(
            bp + (term % 2) * (width // 8) * 256, 64, width, tla.BFloat16, tla.arch.nZ
        )
        tla.mmad(
            c, a, b, init_c=(term == 0), hf32_mode=tla.params.HF32Mode.HF32_DISABLE
        )
    a0 = tensor(ap, 128, 64, tla.BFloat16, tla.arch.zN)
    b2 = tensor(bp + 2 * (width // 8) * 256, 64, width, tla.BFloat16, tla.arch.nZ)
    tla.mmad(c, a0, b2, init_c=False, hf32_mode=tla.params.HF32Mode.HF32_DISABLE)
    tla.set_flag(mf)
    tla.wait_flag(mf)
    tla.copy(
        tensor(ub + 176 * 256, 128, width),
        c,
        tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
    )
    tla.set_flag(mm1)
    tla.wait_flag(mm1)
    tla.set_flag(fm)
    tla.wait_flag(fm)


@tla.jit
def compute_u(lp, ap, bp, cp, slot, width, m1m, mm1):
    a = tensor(ap, 64, 64, tag=tla.arch.zN)
    b = tensor(bp, 64, width, tag=tla.arch.nZ)
    c = tensor(cp, 64, width, tag=tla.arch.L0Clayout)
    tla.copy(a, tensor(lp + (slot * 128 + 112) * 256, 64, 64, tag=tla.arch.zN))
    tla.copy(b, tensor(lp + (320 + 32 * slot) * 256, 64, width, tag=tla.arch.zN))
    tla.set_flag(m1m)
    tla.wait_flag(m1m)
    tla.mmad(c, a, b, init_c=True, hf32_mode=tla.params.HF32Mode.HF32_DISABLE)
    tla.set_flag(mm1)
    tla.wait_flag(mm1)


def factor_index(head, block, length, hk, hv, hf):
    value_head = head % hv
    batch = head // hv
    fh = value_head if hf == hv else value_head // (hv // hk)
    return (tla.Int64(batch) * tla.Int64(hf) + tla.Int64(fh)) * tla.Int64(
        (length + 63) // 64
    ) + tla.Int64(block)


@tla.jit
def load_factors(qg, kt0, kt1, aq, ainv, wh, wl, lp, f, slot, free, loaded):
    tla.wait_flag(free)
    base = lp + slot * 128 * 256
    off = tla.Int64(f) * tla.Int64(8192)
    soff = tla.Int64(f) * tla.Int64(4096)
    tla.copy(
        tensor(base, 64, 128, tla.BFloat16, tla.arch.zN), gm_view(qg, off, 64, 128, 128)
    )
    tla.copy(
        tensor(base + 16 * 256, 64, 128, tla.BFloat16, tla.arch.zN),
        gm_view(wh, off, 64, 128, 128),
    )
    tla.copy(
        tensor(base + 32 * 256, 64, 128, tla.BFloat16, tla.arch.zN),
        gm_view(wl, off, 64, 128, 128),
    )
    tla.copy(
        tensor(base + 48 * 256, 64, 128, tla.BFloat16, tla.arch.zN),
        gm_view(kt0, off, 64, 128, 128),
    )
    tla.copy(
        tensor(base + 64 * 256, 64, 128, tla.BFloat16, tla.arch.zN),
        gm_view(kt1, off, 64, 128, 128),
    )
    tla.copy(
        tensor(base + 96 * 256, 64, 64, tag=tla.arch.zN), gm_view(aq, soff, 64, 64, 64)
    )
    tla.copy(
        tensor(base + 112 * 256, 64, 64, tag=tla.arch.zN),
        gm_view(ainv, soff, 64, 64, 64),
    )
    tla.set_flag(loaded)


@tla.jit
def predict(lp, ap, bp, cp, slot, width, m1m, mm1):
    # Keep both planes in L0 until all four consumers finish. P may reuse
    # the upper B bank only after the final CUBE -> MTE1 acknowledgement.
    for plane in tla.range_constexpr(2):
        a = tensor(ap + plane * 16 * 256, 64, 128, tla.BFloat16, tla.arch.zN)
        b = tensor(
            bp + plane * (width // 4) * 256, 128, width, tla.BFloat16, tla.arch.nZ
        )
        tla.copy(
            a,
            tensor(
                lp + (slot * 128 + 16 + plane * 16) * 256,
                64,
                128,
                tla.BFloat16,
                tla.arch.zN,
            ),
        )
        tla.copy(
            b,
            tensor(
                lp + (256 + plane * (width // 4)) * 256,
                128,
                width,
                tla.BFloat16,
                tla.arch.zN,
            ),
        )
    tla.set_flag(m1m)
    tla.wait_flag(m1m)
    c = tensor(cp, 64, width, tag=tla.arch.L0Clayout)
    for term in tla.range_constexpr(4):
        left = tensor(ap + (term // 2) * 16 * 256, 64, 128, tla.BFloat16, tla.arch.zN)
        right = tensor(
            bp + (term % 2) * (width // 4) * 256, 128, width, tla.BFloat16, tla.arch.nZ
        )
        tla.mmad(
            c, left, right, init_c=False, hf32_mode=tla.params.HF32Mode.HF32_DISABLE
        )
    tla.set_flag(mm1)
    tla.wait_flag(mm1)


@tla.jit
def publish_delta(ub, lp, cp, slot, width, mf, fm, fix_l1):
    tla.set_flag(mf)
    tla.wait_flag(mf)
    source = tensor(cp, 64, width, tag=tla.arch.L0Clayout)
    tla.copy(tensor(lp + (384 + 32 * slot) * 256, 64, width, tag=tla.arch.zN), source)
    tla.copy(
        tensor(ub + 176 * 256, 64, width),
        source,
        tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
    )
    tla.set_flag(fix_l1)
    tla.wait_flag(fix_l1)
    tla.set_flag(fm)
    tla.wait_flag(fm)


@tla.jit
def cube_group(
    qg,
    kt0,
    kt1,
    aq,
    ainv,
    wh,
    wl,
    ub,
    lp,
    ap,
    bp,
    cp,
    head,
    width,
    length,
    hk,
    hv,
    hf,
    f0,
    f1,
    r0,
    r1,
    m1m,
    mm1,
    mf,
    fm,
    fix_l1,
    wv0,
    wv1,
    hflag,
    dflag,
    oflag,
    done,
):
    blocks = (length + 63) // 64
    tla.set_flag(f0)
    tla.set_flag(f1)
    f = factor_index(head, 0, length, hk, hv, hf)
    load_factors(qg, kt0, kt1, aq, ainv, wh, wl, lp, f, 0, f0, r0)
    for block in tla.range(blocks):
        slot = block % 2
        if slot == 0:
            tla.wait_flag(r0)
        else:
            tla.wait_flag(r1)
        if block + 1 < blocks:
            nf = factor_index(head, block + 1, length, hk, hv, hf)
            if slot == 0:
                load_factors(qg, kt0, kt1, aq, ainv, wh, wl, lp, nf, 1, f1, r1)
            else:
                load_factors(qg, kt0, kt1, aq, ainv, wh, wl, lp, nf, 0, f0, r0)
        if slot == 0:
            cube_wait(wv0)
        else:
            cube_wait(wv1)
        compute_u(lp, ap, bp, cp, slot, width, m1m, mm1)
        cube_wait(hflag)
        predict(lp, ap, bp, cp, slot, width, m1m, mm1)
        publish_delta(ub, lp, cp, slot, width, mf, fm, fix_l1)
        cube_publish(dflag)
        compute_output(ub, lp, ap, bp, cp, slot, width, m1m, mf, mm1, fm)
        cube_publish(oflag)
        cube_wait(dflag)
        compute_state_product(ub, lp, ap, bp, cp, slot, width, m1m, mf, mm1, fm)
        cube_publish(hflag)
        if slot == 0:
            tla.set_flag(f0)
        else:
            tla.set_flag(f1)
    cube_wait(hflag)
    tla.wait_flag(f0)
    tla.wait_flag(f1)
    tla.pipe_barrier(tla.pipes.ALL)
    cube_set(done)


def triplet(value):
    high = component_high(value)
    residual = value - high.to(tla.Float32, tla.params.CastParams())
    middle = residual.to(tla.BFloat16, COMPONENT_CAST)
    low = (residual - middle.to(tla.Float32, tla.params.CastParams())).to(
        tla.BFloat16, COMPONENT_CAST
    )
    return high, middle, low


def store_native_triplet(value, high, middle, low, row, part, width):
    hi, mid, lo = triplet(value)
    hi_packed, hi_unused = tla.deinterleave(hi, hi)
    mid_packed, mid_unused = tla.deinterleave(mid, mid)
    lo_packed, lo_unused = tla.deinterleave(lo, lo)
    active = tla.create_mask(
        pattern=tla.mask.VL32 if width == 32 else tla.mask.VL64, dtype=tla.BFloat16
    )
    offset = row * 16 + part * 64 * 32
    sub(high, 0, offset, 1, 128).store(
        hi_packed, tla.params.BlockStoreParams(block_stride=32), mask=active
    )
    sub(middle, 0, offset, 1, 128).store(
        mid_packed, tla.params.BlockStoreParams(block_stride=32), mask=active
    )
    sub(low, 0, offset, 1, 128).store(
        lo_packed, tla.params.BlockStoreParams(block_stride=32), mask=active
    )


@tla.jit
def encode_delta_v128(ub, lp, block, width, ready):
    # The current WV has consumed this raw slot. Its n+2 refill is after
    # the current P/H round trip. Delta lives in the previous, now dead P pool.
    raw = ub + (block % 2) * 48 * 256
    source = tensor(ub + 176 * 256, 1, 4096)
    high = tensor(raw, 1, 4096, tla.BFloat16)
    middle = tensor(raw + 8 * 256, 1, 4096, tla.BFloat16)
    low = tensor(raw + 16 * 256, 1, 4096, tla.BFloat16)
    with tla.vec.func(mode="simd"):
        for encode_row in tla.range(32):
            for encode_part in tla.range_constexpr((width + 63) // 64):
                value = sub(
                    source, 0, encode_row * width + encode_part * 64, 1, 64
                ).load()
                store_native_triplet(
                    value, high, middle, low, encode_row, encode_part, width
                )
    tla.set_flag(ready)
    tla.wait_flag(ready)
    row0 = tla.arch.sub_block_idx() * 32
    for plane in tla.range_constexpr(3):
        dest = tensor(
            lp + (448 + plane * (width // 8)) * 256,
            64,
            width,
            tla.BFloat16,
            tla.arch.zN,
        )
        tla.copy(
            sub(dest, row0, 0, 32, width),
            tensor(raw + plane * 8 * 256, 32, width, tla.BFloat16, tla.arch.zN),
        )


def store_two_bf16_rows(encoded, dest, row, prefix_elements, stride, first, second):
    native, unused = tla.deinterleave(encoded, encoded)
    sub(dest, 0, prefix_elements + row * 16, 1, 128).store(
        native, tla.params.BlockStoreParams(block_stride=stride), mask=first
    )
    sub(dest, 0, (row + 1) * 16, 1, 128).store(
        native, tla.params.BlockStoreParams(block_stride=stride), mask=second
    )


@tla.jit
def encode_delta(ub, lp, block, width, ready):
    if width == 32:
        raw = ub + (block % 2) * 48 * 256
        source = tensor(ub + 176 * 256, 1, 4096)
        high = tensor(raw, 1, 8192, tla.BFloat16)
        middle = tensor(raw + 8 * 256, 1, 8192, tla.BFloat16)
        low = tensor(raw + 16 * 256, 1, 8192, tla.BFloat16)
        with tla.vec.func(mode="simd"):
            first = tla.create_mask(pattern=tla.mask.VL32, dtype=tla.BFloat16)
            second = tla.bitwise_xor(
                tla.create_mask(pattern=tla.mask.VL64, dtype=tla.BFloat16), first
            )
            for pair in tla.range(16):
                value = sub(source, 0, pair * 64, 1, 64).load()
                hi, mid, lo = triplet(value)
                store_two_bf16_rows(hi, high, pair * 2, 1024, 32, first, second)
                store_two_bf16_rows(mid, middle, pair * 2, 1024, 32, first, second)
                store_two_bf16_rows(lo, low, pair * 2, 1024, 32, first, second)
        tla.set_flag(ready)
        tla.wait_flag(ready)
        row0 = tla.arch.sub_block_idx() * 32
        for plane in tla.range_constexpr(3):
            dest = tensor(
                lp + (448 + plane * 4) * 256, 64, 32, tla.BFloat16, tla.arch.zN
            )
            tla.copy(
                sub(dest, row0, 0, 32, 32),
                tensor(raw + (plane * 8 + 2) * 256, 32, 32, tla.BFloat16, tla.arch.zN),
            )
    else:
        encode_delta_v128(ub, lp, block, width, ready)


def store_native_components(value, high, low, row, part, width):
    hi = component_high(value)
    lo = (value - hi.to(tla.Float32, tla.params.CastParams())).to(
        tla.BFloat16, COMPONENT_CAST
    )
    packed_hi, unused_hi = tla.deinterleave(hi, hi)
    packed_lo, unused_lo = tla.deinterleave(lo, lo)
    active = tla.create_mask(
        pattern=tla.mask.VL32 if width == 32 else tla.mask.VL64, dtype=tla.BFloat16
    )
    offset = row * 16 + part * 64 * 64
    sub(high, 0, offset, 1, 128).store(
        packed_hi, tla.params.BlockStoreParams(block_stride=64), mask=active
    )
    sub(low, 0, offset, 1, 128).store(
        packed_lo, tla.params.BlockStoreParams(block_stride=64), mask=active
    )


@tla.jit
def initialize_state(
    initial, ub, lp, head, vcol, width, has_initial, loaded, vector_done, staging_free
):
    lanes = 32 if width == 32 else 64
    tla.pipe_barrier(tla.pipes.ALL)
    row0 = tla.arch.sub_block_idx() * 64
    hold = tensor(ub + 112 * 256, 64, width)
    hold_flat = tensor(ub + 112 * 256, 1, 8192)
    high = tensor(ub + 144 * 256, 1, 8192, tla.BFloat16)
    low = tensor(ub + 160 * 256, 1, 8192, tla.BFloat16)
    if has_initial:
        off = (tla.Int64(head) * tla.Int64(128) + tla.Int64(row0)) * tla.Int64(
            128
        ) + tla.Int64(vcol)
        tla.copy(hold, gm_view(initial, off, 64, width, 128))
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            for init_row in tla.range(64):
                for init_part in tla.range_constexpr((width + 63) // 64):
                    value = sub(
                        hold_flat, 0, init_row * width + init_part * 64, 1, 64
                    ).load()
                    store_native_components(
                        value, high, low, init_row, init_part, width
                    )
    else:
        with tla.vec.func(mode="simd"):
            zeros = tla.full(0.0, tla.Float32)
            for zero_row in tla.range(64):
                for zero_part in tla.range_constexpr((width + 63) // 64):
                    sub(hold, zero_row, zero_part * 64, 1, lanes).store(zeros)
                    store_native_components(
                        zeros, high, low, zero_row, zero_part, width
                    )
    tla.set_flag(vector_done)
    tla.wait_flag(vector_done)
    dest_high = tensor(lp + 256 * 256, 128, width, tla.BFloat16, tla.arch.zN)
    dest_low = tensor(
        lp + (256 + width // 4) * 256, 128, width, tla.BFloat16, tla.arch.zN
    )
    tla.copy(
        sub(dest_high, row0, 0, 64, width),
        tensor(ub + 144 * 256, 64, width, tla.BFloat16, tla.arch.zN),
    )
    tla.copy(
        sub(dest_low, row0, 0, 64, width),
        tensor(ub + 160 * 256, 64, width, tla.BFloat16, tla.arch.zN),
    )
    tla.set_flag(staging_free)
    tla.wait_flag(staging_free)


@tla.jit
def load_values_pair(
    v,
    w,
    end_gm,
    ub,
    lp,
    head,
    vcol,
    width,
    block,
    f,
    length,
    w_bf16,
    loaded,
    raw_reuse,
    ready,
    staging_free,
):
    lanes = 32 if width == 32 else 64
    slot = block % 2
    valid = tla.Int32(length) - tla.Int32(block) * tla.Int32(64)
    if valid > 64:
        valid = 64
    key0 = tla.arch.sub_block_idx() * 64
    row0 = tla.arch.sub_block_idx() * 32
    raw = ub + slot * 48 * 256
    vr = tensor(raw, 64, width, tla.BFloat16)
    wr = tensor(raw + 16 * 256, 64, width, tla.BFloat16 if w_bf16 else tla.Float32)
    end = tensor(ub + 232 * 256 + slot * 128, 1, 128)
    if valid < 64:
        with tla.vec.func(mode="simd"):
            zeros = tla.full(0.0, tla.Float32)
            for pad_row in tla.range(valid, 64):
                for pad_part in tla.range_constexpr((width + 63) // 64):
                    sub(vr, pad_row, pad_part * 64, 1, lanes).store(
                        zeros.to(tla.BFloat16, ZERO_CAST), PACK16
                    )
                    if w_bf16:
                        sub(wr, pad_row, pad_part * 64, 1, lanes).store(
                            zeros.to(tla.BFloat16, ZERO_CAST), PACK16
                        )
                    else:
                        sub(wr, pad_row, pad_part * 64, 1, lanes).store(zeros)
    tla.set_flag(raw_reuse)
    tla.wait_flag(raw_reuse)
    tla.copy(
        sub(end, 0, key0, 1, 64),
        gm_view(end_gm, tla.Int64(f) * tla.Int64(128) + tla.Int64(key0), 1, 64, 128),
    )
    off = (
        tla.Int64(head) * tla.Int64(length) + tla.Int64(block) * tla.Int64(64)
    ) * tla.Int64(128) + tla.Int64(vcol)
    tla.copy(sub(vr, 0, 0, valid, width), gm_view(v, off, valid, width, 128))
    tla.copy(sub(wr, 0, 0, valid, width), gm_view(w, off, valid, width, 128))
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    v_flat = tensor(raw, 1, 8192, tla.BFloat16)
    w_flat = tensor(raw + 16 * 256, 1, 8192, tla.BFloat16 if w_bf16 else tla.Float32)
    native_staging = tensor(ub + 92 * 256, 1, 4096)
    with tla.vec.func(mode="simd"):
        first = tla.create_mask(pattern=tla.mask.VL32, dtype=tla.Float32)
        second = tla.bitwise_xor(
            tla.create_mask(pattern=tla.mask.VL64, dtype=tla.Float32), first
        )
        for pair in tla.range(16):
            value = read_f32(sub(v_flat, 0, (row0 + pair * 2) * 32, 1, 64), True)
            weight = read_f32(sub(w_flat, 0, (row0 + pair * 2) * 32, 1, 64), w_bf16)
            result = value * weight
            sub(native_staging, 0, 1024 + pair * 16, 1, 64).store(
                result, tla.params.BlockStoreParams(block_stride=32), mask=first
            )
            sub(native_staging, 0, (pair * 2 + 1) * 8, 1, 64).store(
                result, tla.params.BlockStoreParams(block_stride=32), mask=second
            )
    tla.set_flag(ready)
    tla.wait_flag(ready)
    dest = tensor(lp + (320 + 32 * slot) * 256, 64, width, tag=tla.arch.zN)
    tla.copy(
        sub(dest, row0, 0, 32, width), tensor(ub + 96 * 256, 32, width, tag=tla.arch.zN)
    )
    tla.set_flag(staging_free)
    tla.wait_flag(staging_free)


@tla.jit
def load_values_v128(
    v,
    w,
    end_gm,
    ub,
    lp,
    head,
    vcol,
    width,
    block,
    f,
    length,
    w_bf16,
    loaded,
    raw_reuse,
    ready,
    staging_free,
):
    lanes = 32 if width == 32 else 64
    slot = block % 2
    valid = tla.Int32(length) - tla.Int32(block) * tla.Int32(64)
    if valid > 64:
        valid = 64
    key0 = tla.arch.sub_block_idx() * 64
    row0 = tla.arch.sub_block_idx() * 32
    raw = ub + slot * 48 * 256
    vr = tensor(raw, 64, width, tla.BFloat16)
    wr = tensor(raw + 16 * 256, 64, width, tla.BFloat16 if w_bf16 else tla.Float32)
    end = tensor(ub + 232 * 256 + slot * 128, 1, 128)
    if valid < 64:
        with tla.vec.func(mode="simd"):
            zeros = tla.full(0.0, tla.Float32)
            for pad_row in tla.range(valid, 64):
                for pad_part in tla.range_constexpr((width + 63) // 64):
                    sub(vr, pad_row, pad_part * 64, 1, lanes).store(
                        zeros.to(tla.BFloat16, ZERO_CAST), PACK16
                    )
                    if w_bf16:
                        sub(wr, pad_row, pad_part * 64, 1, lanes).store(
                            zeros.to(tla.BFloat16, ZERO_CAST), PACK16
                        )
                    else:
                        sub(wr, pad_row, pad_part * 64, 1, lanes).store(zeros)
    tla.set_flag(raw_reuse)
    tla.wait_flag(raw_reuse)
    tla.copy(
        sub(end, 0, key0, 1, 64),
        gm_view(end_gm, tla.Int64(f) * tla.Int64(128) + tla.Int64(key0), 1, 64, 128),
    )
    off = (
        tla.Int64(head) * tla.Int64(length) + tla.Int64(block) * tla.Int64(64)
    ) * tla.Int64(128) + tla.Int64(vcol)
    tla.copy(sub(vr, 0, 0, valid, width), gm_view(v, off, valid, width, 128))
    tla.copy(sub(wr, 0, 0, valid, width), gm_view(w, off, valid, width, 128))
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    v_flat = tensor(raw, 1, 8192, tla.BFloat16)
    w_flat = tensor(raw + 16 * 256, 1, 8192, tla.BFloat16 if w_bf16 else tla.Float32)
    native_staging = tensor(ub + 96 * 256, 1, 4096)
    with tla.vec.func(mode="simd"):
        for row in tla.range(32):
            for part in tla.range_constexpr((width + 63) // 64):
                value = read_f32(
                    sub(v_flat, 0, (row0 + row) * width + part * 64, 1, 64), True
                )
                weight = read_f32(
                    sub(w_flat, 0, (row0 + row) * width + part * 64, 1, 64), w_bf16
                )
                target = sub(native_staging, 0, row * 8 + part * 64 * 32, 1, 64)
                if width == 32:
                    active = tla.create_mask(pattern=tla.mask.VL32, dtype=tla.Float32)
                    target.store(
                        value * weight,
                        tla.params.BlockStoreParams(block_stride=32),
                        mask=active,
                    )
                else:
                    target.store(
                        value * weight, tla.params.BlockStoreParams(block_stride=32)
                    )
    tla.set_flag(ready)
    tla.wait_flag(ready)
    dest = tensor(lp + (320 + 32 * slot) * 256, 64, width, tag=tla.arch.zN)
    tla.copy(
        sub(dest, row0, 0, 32, width), tensor(ub + 96 * 256, 32, width, tag=tla.arch.zN)
    )
    tla.set_flag(staging_free)
    tla.wait_flag(staging_free)


@tla.jit
def load_values(
    v,
    w,
    end_gm,
    ub,
    lp,
    head,
    vcol,
    width,
    block,
    f,
    length,
    w_bf16,
    loaded,
    raw_reuse,
    ready,
    staging_free,
):
    if width == 32:
        load_values_pair(
            v,
            w,
            end_gm,
            ub,
            lp,
            head,
            vcol,
            width,
            block,
            f,
            length,
            w_bf16,
            loaded,
            raw_reuse,
            ready,
            staging_free,
        )
    else:
        load_values_v128(
            v,
            w,
            end_gm,
            ub,
            lp,
            head,
            vcol,
            width,
            block,
            f,
            length,
            w_bf16,
            loaded,
            raw_reuse,
            ready,
            staging_free,
        )


@tla.jit
def encode_state(ub, width, block):
    lanes = 32 if width == 32 else 64
    key0 = tla.arch.sub_block_idx() * 64
    hold = tensor(ub + 112 * 256, 64, width)
    hold_flat = tensor(ub + 112 * 256, 1, 8192)
    high = tensor(ub + 144 * 256, 1, 8192, tla.BFloat16)
    low = tensor(ub + 160 * 256, 1, 8192, tla.BFloat16)
    end = tensor(ub + 232 * 256 + (block % 2) * 128, 1, 128)
    product = tensor(ub + 176 * 256, 1, 8192)
    with tla.vec.func(mode="simd"):
        for update_row in tla.range(64):
            gate = sub(end, 0, key0 + update_row, 1, 1).load(BROADCAST32)
            for update_part in tla.range_constexpr((width + 63) // 64):
                previous = sub(
                    hold_flat, 0, update_row * width + update_part * 64, 1, 64
                ).load()
                increment = sub(
                    product, 0, update_row * width + update_part * 64, 1, 64
                ).load()
                current = previous * gate + increment
                sub(hold, update_row, update_part * 64, 1, lanes).store(current)
                store_native_components(
                    current, high, low, update_row, update_part, width
                )


@tla.jit
def encode_state_pair(ub, block):
    key0 = tla.arch.sub_block_idx() * 64
    hold = tensor(ub + 112 * 256, 1, 8192)
    product = tensor(ub + 176 * 256, 1, 8192)
    end = tensor(ub + 232 * 256 + (block % 2) * 128, 1, 128)
    high = tensor(ub + 140 * 256, 1, 8192, tla.BFloat16)
    low = tensor(ub + 156 * 256, 1, 8192, tla.BFloat16)
    with tla.vec.func(mode="simd"):
        first = tla.create_mask(pattern=tla.mask.VL32, dtype=tla.BFloat16)
        second = tla.bitwise_xor(
            tla.create_mask(pattern=tla.mask.VL64, dtype=tla.BFloat16), first
        )
        first_row = tla.cmp(tla.arange(0, dtype=tla.Int32), 32, "lt")
        for pair in tla.range(32):
            row = pair * 2
            g0 = sub(end, 0, key0 + row, 1, 1).load(BROADCAST32)
            g1 = sub(end, 0, key0 + row + 1, 1, 1).load(BROADCAST32)
            gate = tla.where(first_row, g0, g1)
            previous = sub(hold, 0, row * 32, 1, 64).load()
            increment = sub(product, 0, row * 32, 1, 64).load()
            current = previous * gate + increment
            sub(hold, 0, row * 32, 1, 64).store(current)
            hi = component_high(current)
            lo = (current - hi.to(tla.Float32, tla.params.CastParams())).to(
                tla.BFloat16, COMPONENT_CAST
            )
            store_two_bf16_rows(hi, high, row, 2048, 64, first, second)
            store_two_bf16_rows(lo, low, row, 2048, 64, first, second)


@tla.jit
def update_state(
    final_state, ub, lp, head, vcol, width, block, length, write_final, state_ready
):
    key0 = tla.arch.sub_block_idx() * 64
    hold = tensor(ub + 112 * 256, 64, width)
    if width == 32:
        encode_state_pair(ub, block)
    else:
        encode_state(ub, width, block)
    tla.set_flag(state_ready)
    tla.wait_flag(state_ready)
    next_high = tensor(lp + 256 * 256, 128, width, tla.BFloat16, tla.arch.zN)
    next_low = tensor(
        lp + (256 + width // 4) * 256, 128, width, tla.BFloat16, tla.arch.zN
    )
    tla.copy(
        sub(next_high, key0, 0, 64, width),
        tensor(ub + 144 * 256, 64, width, tla.BFloat16, tla.arch.zN),
    )
    tla.copy(
        sub(next_low, key0, 0, 64, width),
        tensor(ub + 160 * 256, 64, width, tla.BFloat16, tla.arch.zN),
    )
    if write_final:
        if block == (length + 63) // 64 - 1:
            state_offset = (
                tla.Int64(head) * tla.Int64(128) + tla.Int64(key0)
            ) * tla.Int64(128) + tla.Int64(vcol)
            tla.copy(gm_view(final_state, state_offset, 64, width, 128), hold)


def bf16_rne(value):
    bits = value.bitcast(tla.Int32)
    upper = tla.shift_right(bits, 16)
    odd = tla.bitwise_and(upper, tla.full(1, tla.Int32))
    rounded = tla.shift_right(bits + tla.full(0x7FFF, tla.Int32) + odd, 16)
    magnitude = tla.bitwise_and(bits, tla.full(0x7FFFFFFF, tla.Int32))
    # Inf stays Inf. Preserve the sign and make every NaN a quiet BF16 NaN,
    # including inputs whose payload exists only in the discarded low bits.
    special = tla.where(
        tla.cmp(magnitude, 0x7F800000, "gt"),
        tla.bitwise_or(upper, tla.full(0x40, tla.Int32)),
        upper,
    )
    result = tla.where(tla.cmp(magnitude, 0x7F800000, "ge"), special, rounded)
    short = result.to(
        tla.Int16,
        tla.params.CastParams(
            round_mode=tla.params.RoundMode.CAST_TRUNC,
            sat_mode=tla.params.SatMode.NOSAT,
            reg_slot=tla.params.RegSlot.ZERO,
        ),
    )
    return short.bitcast(tla.BFloat16)


@tla.jit
def encode_output_v128(ub, width, block, scale):
    lanes = 32 if width == 32 else 64
    raw_output = tensor(ub + 208 * 256, 32, width)
    output = tensor(ub + 224 * 256, 32, width, tla.BFloat16)
    with tla.vec.func(mode="simd"):
        multiplier = tla.full(scale, tla.Float32)
        for output_row in tla.range(32):
            for output_part in tla.range_constexpr((width + 63) // 64):
                raw = sub(raw_output, output_row, output_part * 64, 1, lanes).load()
                sub(output, output_row, output_part * 64, 1, lanes).store(
                    bf16_rne(raw * multiplier), PACK16
                )


@tla.jit
def encode_output(ub, width, block, scale):
    if width == 32:
        source = tensor(ub + 208 * 256, 1, 1024)
        output = tensor(ub + 224 * 256, 1, 1024, tla.BFloat16)
        with tla.vec.func(mode="simd"):
            multiplier = tla.full(scale, tla.Float32)
            for pair in tla.range(16):
                raw = sub(source, 0, pair * 64, 1, 64).load()
                sub(output, 0, pair * 64, 1, 64).store(
                    bf16_rne(raw * multiplier), PACK16
                )
    else:
        encode_output_v128(ub, width, block, scale)


@tla.jit
def write_output(out, ub, head, vcol, width, block, length, scale, output_ready):
    row0 = tla.arch.sub_block_idx() * 32
    output = tensor(ub + 224 * 256, 32, width, tla.BFloat16)
    encode_output(ub, width, block, scale)
    tla.set_flag(output_ready)
    tla.wait_flag(output_ready)
    valid = length - block * 64 - row0
    if valid > 32:
        valid = 32
    if valid > 0:
        offset = (
            tla.Int64(head) * tla.Int64(length)
            + tla.Int64(block) * tla.Int64(64)
            + tla.Int64(row0)
        ) * tla.Int64(128) + tla.Int64(vcol)
        tla.copy(
            gm_view(out, offset, valid, width, 128), sub(output, 0, 0, valid, width)
        )


@tla.jit
def vector_group(
    v,
    w,
    end,
    initial,
    out,
    final_state,
    ub,
    lp,
    head,
    vcol,
    width,
    length,
    hk,
    hv,
    hf,
    w_bf16,
    has_initial,
    write_final,
    scale,
    ld,
    rr,
    vr,
    sf,
    init_ready,
    state_ready,
    delta_ready,
    output_ready,
    wv0,
    wv1,
    hflag,
    dflag,
    oflag,
    done,
):
    blocks = (length + 63) // 64
    initialize_state(
        initial, ub, lp, head, vcol, width, has_initial, ld, init_ready, sf
    )
    vector_set(hflag)
    f = factor_index(head, 0, length, hk, hv, hf)
    load_values(
        v, w, end, ub, lp, head, vcol, width, 0, f, length, w_bf16, ld, rr, vr, sf
    )
    vector_set(wv0)
    for block in tla.range(blocks):
        if block + 1 < blocks:
            nf = factor_index(head, block + 1, length, hk, hv, hf)
            load_values(
                v,
                w,
                end,
                ub,
                lp,
                head,
                vcol,
                width,
                block + 1,
                nf,
                length,
                w_bf16,
                ld,
                rr,
                vr,
                sf,
            )
            if (block + 1) % 2 == 0:
                vector_set(wv0)
            else:
                vector_set(wv1)
        vector_wait(dflag)
        encode_delta(ub, lp, block, width, delta_ready)
        vector_set(dflag)
        vector_wait(oflag)
        write_output(out, ub, head, vcol, width, block, length, scale, output_ready)
        vector_wait(hflag)
        update_state(
            final_state,
            ub,
            lp,
            head,
            vcol,
            width,
            block,
            length,
            write_final,
            state_ready,
        )
        vector_set(hflag)
    vector_wait(done)


@tla.kernel
def recurrent_kernel(
    v: tla.Tensor,
    w: tla.Tensor,
    qg: tla.Tensor,
    kt0: tla.Tensor,
    kt1: tla.Tensor,
    end: tla.Tensor,
    aq: tla.Tensor,
    ainv: tla.Tensor,
    wh: tla.Tensor,
    wl: tla.Tensor,
    initial: tla.Tensor,
    out: tla.Tensor,
    final_state: tla.Tensor,
    length: tla.Constexpr[int],
    hk: tla.Constexpr[int],
    hv: tla.Constexpr[int],
    hf: tla.Constexpr[int],
    groups: tla.Constexpr[int],
    full_heads: tla.Constexpr[int],
    w_bf16: tla.Constexpr[bool],
    has_initial: tla.Constexpr[bool],
    write_final: tla.Constexpr[bool],
    scale: tla.Constexpr[float],
):
    ub = tla.allocate(240 * 256, tla.Float32, tla.AddressSpace.ub, 256)
    lp = tla.allocate(496 * 256, tla.Float32, tla.AddressSpace.l1, 512)
    ap = tla.allocate(64 * 256, tla.Float32, tla.AddressSpace.l0a, 512)
    bp = tla.allocate(64 * 256, tla.Float32, tla.AddressSpace.l0b, 512)
    cp = tla.allocate(96 * 256, tla.Float32, tla.AddressSpace.l0c, 512)
    wv0 = tla.cross_flag("WV0", mode=4)
    wv1 = tla.cross_flag("WV1", mode=4)
    hflag = tla.cross_flag("H", mode=4)
    dflag = tla.cross_flag("D", mode=4)
    oflag = tla.cross_flag("O_ready", mode=4)
    done = tla.cross_flag("group_done", mode=4)
    f0 = tla.flag("factor0_free", tla.arch.MTE1, tla.arch.MTE2)
    f1 = tla.flag("factor1_free", tla.arch.MTE1, tla.arch.MTE2)
    r0 = tla.flag("factor0_ready", tla.arch.MTE2, tla.arch.MTE1)
    r1 = tla.flag("factor1_ready", tla.arch.MTE2, tla.arch.MTE1)
    ld = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    rr = tla.flag("raw_reuse", tla.arch.VECTOR, tla.arch.MTE2)
    vr = tla.flag("wv_ready", tla.arch.VECTOR, tla.arch.MTE3)
    init_ready = tla.flag("init_ready", tla.arch.VECTOR, tla.arch.MTE3)
    state_ready = tla.flag("state_ready", tla.arch.VECTOR, tla.arch.MTE3)
    delta_ready = tla.flag("delta_ready", tla.arch.VECTOR, tla.arch.MTE3)
    output_ready = tla.flag("output_ready", tla.arch.VECTOR, tla.arch.MTE3)
    sf = tla.flag("staging_free", tla.arch.MTE3, tla.arch.VECTOR)
    m1m = tla.flag("m1m", tla.arch.MTE1, tla.arch.CUBE)
    mm1 = tla.flag("mm1", tla.arch.CUBE, tla.arch.MTE1)
    mf = tla.flag("mf", tla.arch.CUBE, tla.arch.FIX)
    fm = tla.flag("fm", tla.arch.FIX, tla.arch.CUBE)
    fix_l1 = tla.flag("fix_l1", tla.arch.FIX, tla.arch.MTE1)
    with tla.cube():
        if full_heads > 0:
            for cg_full in tla.range(
                tla.arch.block_idx(), full_heads, tla.arch.block_num()
            ):
                cube_group(
                    qg,
                    kt0,
                    kt1,
                    aq,
                    ainv,
                    wh,
                    wl,
                    ub,
                    lp,
                    ap,
                    bp,
                    cp,
                    cg_full,
                    128,
                    length,
                    hk,
                    hv,
                    hf,
                    f0,
                    f1,
                    r0,
                    r1,
                    m1m,
                    mm1,
                    mf,
                    fm,
                    fix_l1,
                    wv0,
                    wv1,
                    hflag,
                    dflag,
                    oflag,
                    done,
                )
        if full_heads < groups:
            for cg_tail in tla.range(
                full_heads + tla.arch.block_idx(), groups, tla.arch.block_num()
            ):
                ch_tail = full_heads + (cg_tail - full_heads) // 4
                cube_group(
                    qg,
                    kt0,
                    kt1,
                    aq,
                    ainv,
                    wh,
                    wl,
                    ub,
                    lp,
                    ap,
                    bp,
                    cp,
                    ch_tail,
                    32,
                    length,
                    hk,
                    hv,
                    hf,
                    f0,
                    f1,
                    r0,
                    r1,
                    m1m,
                    mm1,
                    mf,
                    fm,
                    fix_l1,
                    wv0,
                    wv1,
                    hflag,
                    dflag,
                    oflag,
                    done,
                )
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        if full_heads > 0:
            for vg_full in tla.range(
                tla.arch.block_idx(), full_heads, tla.arch.block_num()
            ):
                vector_group(
                    v,
                    w,
                    end,
                    initial,
                    out,
                    final_state,
                    ub,
                    lp,
                    vg_full,
                    0,
                    128,
                    length,
                    hk,
                    hv,
                    hf,
                    w_bf16,
                    has_initial,
                    write_final,
                    scale,
                    ld,
                    rr,
                    vr,
                    sf,
                    init_ready,
                    state_ready,
                    delta_ready,
                    output_ready,
                    wv0,
                    wv1,
                    hflag,
                    dflag,
                    oflag,
                    done,
                )
        if full_heads < groups:
            for vg_tail in tla.range(
                full_heads + tla.arch.block_idx(), groups, tla.arch.block_num()
            ):
                vh_tail = full_heads + (vg_tail - full_heads) // 4
                vc_tail = (vg_tail - full_heads) % 4 * 32
                vector_group(
                    v,
                    w,
                    end,
                    initial,
                    out,
                    final_state,
                    ub,
                    lp,
                    vh_tail,
                    vc_tail,
                    32,
                    length,
                    hk,
                    hv,
                    hf,
                    w_bf16,
                    has_initial,
                    write_final,
                    scale,
                    ld,
                    rr,
                    vr,
                    sf,
                    init_ready,
                    state_ready,
                    delta_ready,
                    output_ready,
                    wv0,
                    wv1,
                    hflag,
                    dflag,
                    oflag,
                    done,
                )
        tla.pipe_barrier(tla.pipes.ALL)
