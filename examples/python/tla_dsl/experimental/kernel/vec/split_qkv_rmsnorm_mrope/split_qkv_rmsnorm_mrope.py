# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fused QKV split, Q/K RMSNorm and multimodal rotary position embedding."""

from functools import lru_cache
import math

import torch
import torch_npu  # noqa: F401 -- registers torch.npu

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

# ---------------------------------------------------------------------------
# Device kernel
# ---------------------------------------------------------------------------


def round_to_even(value, dtype):
    """Convert FP32 with round-to-nearest-even without double rounding."""
    cast = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    bits = value.bitcast(tla.Int32)
    zeros = bits - bits
    one = zeros + 1
    magnitude = tla.bitwise_and(bits, (zeros + 2147483647))
    if dtype == tla.BFloat16:
        midpoint = tla.cmp(tla.bitwise_and(bits, zeros + 131071), 32768, "eq")
        fixed = tla.where(midpoint, bits - one, bits)
        return fixed.bitcast(tla.Float32).to(dtype, cast)
    exponent = tla.shift_right(magnitude, 23)
    # F32 addition's RN-even rounds to the F16 grid; subtraction is exact.
    scale_exp = tla.min(tla.max(exponent, zeros + 113), zeros + 142) + 13
    magic = tla.shift_left(scale_exp, 23).bitcast(tla.Float32)
    absolute = magnitude.bitcast(tla.Float32)
    rounded = (absolute + magic) - magic
    finite_bits = tla.where(
        tla.cmp(exponent, 143, "ge"), magnitude, rounded.bitcast(tla.Int32)
    )
    result = tla.bitwise_or(finite_bits, tla.bitwise_and(bits, zeros + -2147483648))
    return result.bitcast(tla.Float32).to(dtype, cast)


@tla.kernel
def split_qkv_rmsnorm_mrope_kernel(
    packed: tla.Tensor,
    qw: tla.Tensor,
    kw: tla.Tensor,
    cs: tla.Tensor,
    qb: tla.Tensor,
    kb: tla.Tensor,
    oq: tla.Tensor,
    ok: tla.Tensor,
    ov: tla.Tensor,
    og: tla.Tensor,
    tokens: tla.Int32,
    hq: tla.Constexpr,
    hk: tla.Constexpr,
    d: tla.Constexpr,
    r: tla.Constexpr,
    epsilon: tla.Constexpr,
    sections: tla.Constexpr,
    interleaved: tla.Constexpr,
    bias: tla.Constexpr,
    gate: tla.Constexpr,
    io_dtype: tla.Constexpr,
    narrow: tla.Constexpr,
    group: tla.Constexpr,
):
    dtype = io_dtype
    gq = min(hq, group)
    gk = min(hk, group)
    nq = hq // gq
    nk = hk // gk
    qstride = 2 if gate else 1
    width = (qstride * hq + 2 * hk) * d
    layout_d = tla.make_layout(
        tla.make_shape(d), tla.make_stride(1), layoutTag=tla.arch.RowMajor
    )
    layout_q = tla.make_layout(
        tla.make_shape(gq * d * qstride),
        tla.make_stride(1),
        layoutTag=tla.arch.RowMajor,
    )
    layout_yq = tla.make_layout(
        tla.make_shape(gq * d), tla.make_stride(1), layoutTag=tla.arch.RowMajor
    )
    layout_yk = tla.make_layout(
        tla.make_shape(gk * d), tla.make_stride(1), layoutTag=tla.arch.RowMajor
    )
    layout_cs = tla.make_layout(
        tla.make_shape(r), tla.make_stride(1), layoutTag=tla.arch.RowMajor
    )
    x_storage = tla.allocate(3 * gq * d * qstride, dtype, tla.AddressSpace.ub, 256)
    layout_w = tla.make_layout(
        tla.make_shape(2 * d), tla.make_stride(1), layoutTag=tla.arch.RowMajor
    )
    w = tla.make_tensor(tla.allocate(2 * d, dtype, tla.AddressSpace.ub, 256), layout_w)
    b = tla.make_tensor(tla.allocate(2 * d, dtype, tla.AddressSpace.ub, 256), layout_w)
    layout_c = tla.make_layout(
        tla.make_shape(3 * r), tla.make_stride(1), layoutTag=tla.arch.RowMajor
    )
    c_storage = tla.allocate(3 * 3 * r, dtype, tla.AddressSpace.ub, 256)
    y_storage = tla.allocate(3 * gq * d, dtype, tla.AddressSpace.ub, 256)
    aux_storage = tla.allocate(3 * gq * d, dtype, tla.AddressSpace.ub, 256)
    # A real Tensor view exposes ptr; offset arithmetic uses element counts.
    px = tla.tile_view(packed, tla.make_shape(1), tla.make_coord(0))
    pc = tla.tile_view(cs, tla.make_shape(1), tla.make_coord(0))
    pq = tla.tile_view(oq, tla.make_shape(1), tla.make_coord(0))
    pk = tla.tile_view(ok, tla.make_shape(1), tla.make_coord(0))
    pv = tla.tile_view(ov, tla.make_shape(1), tla.make_coord(0))
    pg = tla.tile_view(og, tla.make_shape(1), tla.make_coord(0))
    pqw = tla.tile_view(qw, tla.make_shape(1), tla.make_coord(0))
    pkw = tla.tile_view(kw, tla.make_shape(1), tla.make_coord(0))
    pqb = tla.tile_view(qb, tla.make_shape(1), tla.make_coord(0))
    pkb = tla.tile_view(kb, tla.make_shape(1), tla.make_coord(0))
    loaded = tla.flag("weights_loaded", tla.arch.MTE2, tla.arch.VECTOR, id=7)
    ready0 = tla.flag("ready0", tla.arch.MTE2, tla.arch.VECTOR, id=0)
    done0 = tla.flag("done0", tla.arch.VECTOR, tla.arch.MTE3, id=0)
    free0 = tla.flag("free0", tla.arch.MTE3, tla.arch.MTE2, id=0)
    ready1 = tla.flag("ready1", tla.arch.MTE2, tla.arch.VECTOR, id=1)
    done1 = tla.flag("done1", tla.arch.VECTOR, tla.arch.MTE3, id=1)
    free1 = tla.flag("free1", tla.arch.MTE3, tla.arch.MTE2, id=1)
    ready2 = tla.flag("ready2", tla.arch.MTE2, tla.arch.VECTOR, id=2)
    done2 = tla.flag("done2", tla.arch.VECTOR, tla.arch.MTE3, id=2)
    free2 = tla.flag("free2", tla.arch.MTE3, tla.arch.MTE2, id=2)
    unpack = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_UNPACK_B16)
    pack = tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK_B32)
    cast = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    with tla.vector():
        # Q/K parameters are immutable for the whole invocation and private per AIV.
        tla.copy(
            tla.tile_view(w, tla.make_shape(d), tla.make_coord(0)),
            tla.make_tensor(pqw.ptr, layout_d),
        )
        tla.copy(
            tla.tile_view(w, tla.make_shape(d), tla.make_coord(1)),
            tla.make_tensor(pkw.ptr, layout_d),
        )
        if tla.const_expr(bias):
            tla.copy(
                tla.tile_view(b, tla.make_shape(d), tla.make_coord(0)),
                tla.make_tensor(pqb.ptr, layout_d),
            )
            tla.copy(
                tla.tile_view(b, tla.make_shape(d), tla.make_coord(1)),
                tla.make_tensor(pkb.ptr, layout_d),
            )
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.set_flag(free0)
        tla.set_flag(free1)
        tla.set_flag(free2)
        buffer = 0
        # O(T*(HQ+HK)*D). Groups own disjoint complete heads, including all D lanes.
        for task in tla.range(
            tla.arch.block_idx(), tokens * (nq + nk), tla.arch.block_num()
        ):
            # Disjoint slots are released by their final user (MTE3).
            if buffer == 0:
                tla.wait_flag(free0)
            elif buffer == 1:
                tla.wait_flag(free1)
            else:
                tla.wait_flag(free2)
            x = tla.make_tensor(x_storage + buffer * gq * d * qstride, layout_q)
            c = tla.make_tensor(c_storage + buffer * 3 * r, layout_c)
            y = tla.make_tensor(y_storage + buffer * gq * d, layout_yq)
            aux = tla.make_tensor(aux_storage + buffer * gq * d, layout_yq)
            # Contiguous Q then K task spaces prevent a grid stride divisible
            # by 2 from pinning half the AIVs to the much lighter KV group.
            token = task // nq
            slot = task % nq
            if task >= tokens * nq:
                token = (task - tokens * nq) // nk
                slot = nq + (task - tokens * nq) % nk
            woff = d
            heads = gk
            stride = 1
            base = 0
            xoff = token * width
            if slot < nq:
                woff = 0
                heads = gq
                stride = qstride
                base = slot * gq
                xoff = xoff + base * d * qstride
                tla.copy(x, tla.make_tensor(px.ptr + xoff, layout_q))
            else:
                base = (slot - nq) * gk
                xoff = xoff + hq * d * qstride + base * d
                tla.copy(
                    tla.tile_view(x, tla.make_shape(gk * d), tla.make_coord(0)),
                    tla.make_tensor(px.ptr + xoff, layout_yk),
                )
                tla.copy(
                    tla.tile_view(aux, tla.make_shape(gk * d), tla.make_coord(0)),
                    tla.make_tensor(px.ptr + xoff + hk * d, layout_yk),
                )
            for axis in tla.range_constexpr(3):
                tla.copy(
                    tla.tile_view(c, tla.make_shape(r), tla.make_coord(axis)),
                    tla.make_tensor(pc.ptr + (axis * tokens + token) * r, layout_cs),
                )
            if buffer == 0:
                tla.set_flag(ready0)
                tla.wait_flag(ready0)
            elif buffer == 1:
                tla.set_flag(ready1)
                tla.wait_flag(ready1)
            else:
                tla.set_flag(ready2)
                tla.wait_flag(ready2)
            with tla.vec.func(mode="simd"):
                # Accumulate all D lanes in FP32 before normalizing complete heads.
                if tla.const_expr(r == 64):
                    lane = tla.arange(0, dtype=tla.Int32)
                    freq = tla.bitwise_and(lane, tla.full(31, dtype=tla.Int32))
                    swap_index = tla.bitwise_xor(lane, tla.full(32, dtype=tla.Int32))
                    if tla.const_expr(interleaved):
                        remainder = (
                            freq
                            - tla.shift_right(freq * tla.full(171, dtype=tla.Int32), 9)
                            * 3
                        )
                        mh = tla.bitwise_and(
                            tla.cmp(remainder, 1, "eq"),
                            tla.cmp(freq, 3 * sections[1], "lt"),
                        )
                        mw = tla.bitwise_and(
                            tla.cmp(remainder, 2, "eq"),
                            tla.cmp(freq, 3 * sections[2], "lt"),
                        )
                    else:
                        mh = tla.bitwise_and(
                            tla.cmp(freq, sections[0], "ge"),
                            tla.cmp(freq, sections[0] + sections[1], "lt"),
                        )
                        mw = tla.cmp(freq, sections[0] + sections[1], "ge")
                    t_axis = (
                        tla.tile_view(c, tla.make_shape(64), tla.make_coord(0))
                        .load(unpack)
                        .to(tla.Float32, cast)
                    )
                    h_axis = (
                        tla.tile_view(c, tla.make_shape(64), tla.make_coord(1))
                        .load(unpack)
                        .to(tla.Float32, cast)
                    )
                    w_axis = (
                        tla.tile_view(c, tla.make_shape(64), tla.make_coord(2))
                        .load(unpack)
                        .to(tla.Float32, cast)
                    )
                    ct = tla.gather(t_axis, freq)
                    ch = tla.gather(h_axis, freq)
                    cw = tla.gather(w_axis, freq)
                    st = tla.gather(t_axis, freq + 32)
                    sh = tla.gather(h_axis, freq + 32)
                    sw = tla.gather(w_axis, freq + 32)
                    cosine0 = tla.where(mh, ch, tla.where(mw, cw, ct))
                    sine0 = tla.where(mh, sh, tla.where(mw, sw, st))
                    sine_signed0 = tla.where(
                        tla.cmp(lane, 32, "lt"), tla.neg(sine0), sine0
                    )
                else:
                    for trig_part in tla.range_constexpr((r // 2 + 63) // 64):
                        j = tla.arange(trig_part * 64, dtype=tla.Int32)
                        if tla.const_expr(interleaved):
                            remainder = (
                                j
                                - tla.shift_right(j * tla.full(171, dtype=tla.Int32), 9)
                                * 3
                            )
                            mh = tla.bitwise_and(
                                tla.cmp(remainder, 1, "eq"),
                                tla.cmp(j, 3 * sections[1], "lt"),
                            )
                            mw = tla.bitwise_and(
                                tla.cmp(remainder, 2, "eq"),
                                tla.cmp(j, 3 * sections[2], "lt"),
                            )
                        else:
                            mh = tla.bitwise_and(
                                tla.cmp(j, sections[0], "ge"),
                                tla.cmp(j, sections[0] + sections[1], "lt"),
                            )
                            mw = tla.cmp(j, sections[0] + sections[1], "ge")
                        ct = (
                            tla.tile_view(
                                c,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(trig_part * 64 // min(64, r // 2)),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        ch = (
                            tla.tile_view(
                                c,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord((r + trig_part * 64) // min(64, r // 2)),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        cw = (
                            tla.tile_view(
                                c,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (2 * r + trig_part * 64) // min(64, r // 2)
                                ),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        st = (
                            tla.tile_view(
                                c,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (r // 2 + trig_part * 64) // min(64, r // 2)
                                ),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        sh = (
                            tla.tile_view(
                                c,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (r + r // 2 + trig_part * 64) // min(64, r // 2)
                                ),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        sw = (
                            tla.tile_view(
                                c,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (2 * r + r // 2 + trig_part * 64) // min(64, r // 2)
                                ),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        if tla.const_expr(trig_part == 0):
                            cosine0 = tla.where(mh, ch, tla.where(mw, cw, ct))
                            sine0 = tla.where(mh, sh, tla.where(mw, sw, st))
                        else:
                            cosine1 = tla.where(mh, ch, tla.where(mw, cw, ct))
                            sine1 = tla.where(mh, sh, tla.where(mw, sw, st))
                # Four independent complete heads; H2/H1 preserve smaller groups.
                for quad in tla.range(heads // 4):
                    q0_h1 = quad * 4 + 0
                    q1_h1 = quad * 4 + 1
                    q2_h1 = quad * 4 + 2
                    q3_h1 = quad * 4 + 3
                    q0_sums = tla.full(0.0, dtype=tla.Float32)
                    for q0_part in tla.range_constexpr(d // 64):
                        q0_v = (
                            tla.tile_view(
                                x,
                                tla.make_shape(64),
                                tla.make_coord(q0_h1 * stride * d // 64 + q0_part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        if tla.const_expr(q0_part == 0):
                            q0_x0 = q0_v
                        elif tla.const_expr(q0_part == 1):
                            q0_x1 = q0_v
                        elif tla.const_expr(q0_part == 2):
                            q0_x2 = q0_v
                        elif tla.const_expr(q0_part == 3):
                            q0_x3 = q0_v
                        q0_sums = q0_sums + q0_v * q0_v
                    q0_total = q0_sums.reduce(
                        tla.ReductionOp.ADD,
                        mask=tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32),
                    )
                    q0_variance = (
                        tla.full(q0_total, dtype=tla.Float32) * (1.0 / d) + epsilon
                    )
                    q0_inv = tla.div(
                        tla.full(1.0, dtype=tla.Float32), tla.sqrt(q0_variance)
                    )
                    q1_sums = tla.full(0.0, dtype=tla.Float32)
                    for q1_part in tla.range_constexpr(d // 64):
                        q1_v = (
                            tla.tile_view(
                                x,
                                tla.make_shape(64),
                                tla.make_coord(q1_h1 * stride * d // 64 + q1_part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        if tla.const_expr(q1_part == 0):
                            q1_x0 = q1_v
                        elif tla.const_expr(q1_part == 1):
                            q1_x1 = q1_v
                        elif tla.const_expr(q1_part == 2):
                            q1_x2 = q1_v
                        elif tla.const_expr(q1_part == 3):
                            q1_x3 = q1_v
                        q1_sums = q1_sums + q1_v * q1_v
                    q1_total = q1_sums.reduce(
                        tla.ReductionOp.ADD,
                        mask=tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32),
                    )
                    q1_variance = (
                        tla.full(q1_total, dtype=tla.Float32) * (1.0 / d) + epsilon
                    )
                    q1_inv = tla.div(
                        tla.full(1.0, dtype=tla.Float32), tla.sqrt(q1_variance)
                    )
                    q2_sums = tla.full(0.0, dtype=tla.Float32)
                    for q2_part in tla.range_constexpr(d // 64):
                        q2_v = (
                            tla.tile_view(
                                x,
                                tla.make_shape(64),
                                tla.make_coord(q2_h1 * stride * d // 64 + q2_part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        if tla.const_expr(q2_part == 0):
                            q2_x0 = q2_v
                        elif tla.const_expr(q2_part == 1):
                            q2_x1 = q2_v
                        elif tla.const_expr(q2_part == 2):
                            q2_x2 = q2_v
                        elif tla.const_expr(q2_part == 3):
                            q2_x3 = q2_v
                        q2_sums = q2_sums + q2_v * q2_v
                    q2_total = q2_sums.reduce(
                        tla.ReductionOp.ADD,
                        mask=tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32),
                    )
                    q2_variance = (
                        tla.full(q2_total, dtype=tla.Float32) * (1.0 / d) + epsilon
                    )
                    q2_inv = tla.div(
                        tla.full(1.0, dtype=tla.Float32), tla.sqrt(q2_variance)
                    )
                    q3_sums = tla.full(0.0, dtype=tla.Float32)
                    for q3_part in tla.range_constexpr(d // 64):
                        q3_v = (
                            tla.tile_view(
                                x,
                                tla.make_shape(64),
                                tla.make_coord(q3_h1 * stride * d // 64 + q3_part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        if tla.const_expr(q3_part == 0):
                            q3_x0 = q3_v
                        elif tla.const_expr(q3_part == 1):
                            q3_x1 = q3_v
                        elif tla.const_expr(q3_part == 2):
                            q3_x2 = q3_v
                        elif tla.const_expr(q3_part == 3):
                            q3_x3 = q3_v
                        q3_sums = q3_sums + q3_v * q3_v
                    q3_total = q3_sums.reduce(
                        tla.ReductionOp.ADD,
                        mask=tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32),
                    )
                    q3_variance = (
                        tla.full(q3_total, dtype=tla.Float32) * (1.0 / d) + epsilon
                    )
                    q3_inv = tla.div(
                        tla.full(1.0, dtype=tla.Float32), tla.sqrt(q3_variance)
                    )
                    # Tail is normalized once; the rotary prefix stays in FP32 registers.
                    for q0_part in tla.range_constexpr((d - r) // 64):
                        if tla.const_expr(r // 64 + q0_part == 0):
                            q0_v = q0_x0
                        elif tla.const_expr(r // 64 + q0_part == 1):
                            q0_v = q0_x1
                        elif tla.const_expr(r // 64 + q0_part == 2):
                            q0_v = q0_x2
                        elif tla.const_expr(r // 64 + q0_part == 3):
                            q0_v = q0_x3
                        q0_weight = (
                            tla.tile_view(
                                w,
                                tla.make_shape(64),
                                tla.make_coord(woff // 64 + r // 64 + q0_part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        q0_norm = (q0_v * q0_inv) * q0_weight
                        if tla.const_expr(bias):
                            q0_bv = (
                                tla.tile_view(
                                    b,
                                    tla.make_shape(64),
                                    tla.make_coord(woff // 64 + r // 64 + q0_part),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q0_norm = q0_norm + q0_bv
                        tla.tile_view(
                            y,
                            tla.make_shape(64),
                            tla.make_coord((q0_h1 * d + r) // 64 + q0_part),
                        ).store(narrow(q0_norm, dtype), pack)
                    if tla.const_expr(gate):
                        if slot < nq:
                            for q0_cp in tla.range_constexpr(d // 128):
                                q0_raw = tla.tile_view(
                                    x,
                                    tla.make_shape(128),
                                    tla.make_coord(
                                        ((q0_h1 * 2 + 1) * d) // 128 + q0_cp
                                    ),
                                ).load()
                                tla.tile_view(
                                    aux,
                                    tla.make_shape(128),
                                    tla.make_coord(q0_h1 * d // 128 + q0_cp),
                                ).store(q0_raw)
                    if tla.const_expr(r == 64):
                        q0_full_weight = (
                            tla.tile_view(
                                w, tla.make_shape(64), tla.make_coord(woff // 64)
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        q0_full_norm = (q0_x0 * q0_inv) * q0_full_weight
                        if tla.const_expr(bias):
                            q0_full_bias = (
                                tla.tile_view(
                                    b, tla.make_shape(64), tla.make_coord(woff // 64)
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q0_full_norm = q0_full_norm + q0_full_bias
                        q0_partner = tla.gather(q0_full_norm, swap_index)
                        q0_full_out = q0_full_norm * cosine0 + q0_partner * sine_signed0
                        tla.tile_view(
                            y, tla.make_shape(64), tla.make_coord(q0_h1 * d // 64)
                        ).store(narrow(q0_full_out, dtype), pack)
                    else:
                        for q0_rp in tla.range_constexpr((r // 2 + 63) // 64):
                            if tla.const_expr(q0_rp == 0):
                                q0_cosine = cosine0
                                q0_sine = sine0
                            else:
                                q0_cosine = cosine1
                                q0_sine = sine1
                            if tla.const_expr(r >= 128):
                                if tla.const_expr(q0_rp == 0):
                                    q0_a = q0_x0
                                elif tla.const_expr(q0_rp == 1):
                                    q0_a = q0_x1
                                elif tla.const_expr(q0_rp == 2):
                                    q0_a = q0_x2
                                elif tla.const_expr(q0_rp == 3):
                                    q0_a = q0_x3
                                if tla.const_expr(r // 128 + q0_rp == 0):
                                    q0_z = q0_x0
                                elif tla.const_expr(r // 128 + q0_rp == 1):
                                    q0_z = q0_x1
                                elif tla.const_expr(r // 128 + q0_rp == 2):
                                    q0_z = q0_x2
                                elif tla.const_expr(r // 128 + q0_rp == 3):
                                    q0_z = q0_x3
                            else:
                                q0_a = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (q0_h1 * stride * d + q0_rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                q0_z = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (q0_h1 * stride * d + r // 2 + q0_rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                            q0_wa = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + q0_rp * 64 // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q0_wz = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + (r // 2 + q0_rp * 64) // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q0_a = (q0_a * q0_inv) * q0_wa
                            q0_z = (q0_z * q0_inv) * q0_wz
                            if tla.const_expr(bias):
                                q0_ba = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + q0_rp * 64 // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                q0_bz = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + (r // 2 + q0_rp * 64) // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                q0_a = q0_a + q0_ba
                                q0_z = q0_z + q0_bz
                            q0_left = q0_a * q0_cosine - q0_z * q0_sine
                            q0_right = q0_z * q0_cosine + q0_a * q0_sine
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (q0_h1 * d + q0_rp * 64) // min(64, r // 2)
                                ),
                            ).store(narrow(q0_left, dtype), pack)
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (q0_h1 * d + r // 2 + q0_rp * 64) // min(64, r // 2)
                                ),
                            ).store(narrow(q0_right, dtype), pack)
                    # Tail is normalized once; the rotary prefix stays in FP32 registers.
                    for q1_part in tla.range_constexpr((d - r) // 64):
                        if tla.const_expr(r // 64 + q1_part == 0):
                            q1_v = q1_x0
                        elif tla.const_expr(r // 64 + q1_part == 1):
                            q1_v = q1_x1
                        elif tla.const_expr(r // 64 + q1_part == 2):
                            q1_v = q1_x2
                        elif tla.const_expr(r // 64 + q1_part == 3):
                            q1_v = q1_x3
                        q1_weight = (
                            tla.tile_view(
                                w,
                                tla.make_shape(64),
                                tla.make_coord(woff // 64 + r // 64 + q1_part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        q1_norm = (q1_v * q1_inv) * q1_weight
                        if tla.const_expr(bias):
                            q1_bv = (
                                tla.tile_view(
                                    b,
                                    tla.make_shape(64),
                                    tla.make_coord(woff // 64 + r // 64 + q1_part),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q1_norm = q1_norm + q1_bv
                        tla.tile_view(
                            y,
                            tla.make_shape(64),
                            tla.make_coord((q1_h1 * d + r) // 64 + q1_part),
                        ).store(narrow(q1_norm, dtype), pack)
                    if tla.const_expr(gate):
                        if slot < nq:
                            for q1_cp in tla.range_constexpr(d // 128):
                                q1_raw = tla.tile_view(
                                    x,
                                    tla.make_shape(128),
                                    tla.make_coord(
                                        ((q1_h1 * 2 + 1) * d) // 128 + q1_cp
                                    ),
                                ).load()
                                tla.tile_view(
                                    aux,
                                    tla.make_shape(128),
                                    tla.make_coord(q1_h1 * d // 128 + q1_cp),
                                ).store(q1_raw)
                    if tla.const_expr(r == 64):
                        q1_full_weight = (
                            tla.tile_view(
                                w, tla.make_shape(64), tla.make_coord(woff // 64)
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        q1_full_norm = (q1_x0 * q1_inv) * q1_full_weight
                        if tla.const_expr(bias):
                            q1_full_bias = (
                                tla.tile_view(
                                    b, tla.make_shape(64), tla.make_coord(woff // 64)
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q1_full_norm = q1_full_norm + q1_full_bias
                        q1_partner = tla.gather(q1_full_norm, swap_index)
                        q1_full_out = q1_full_norm * cosine0 + q1_partner * sine_signed0
                        tla.tile_view(
                            y, tla.make_shape(64), tla.make_coord(q1_h1 * d // 64)
                        ).store(narrow(q1_full_out, dtype), pack)
                    else:
                        for q1_rp in tla.range_constexpr((r // 2 + 63) // 64):
                            if tla.const_expr(q1_rp == 0):
                                q1_cosine = cosine0
                                q1_sine = sine0
                            else:
                                q1_cosine = cosine1
                                q1_sine = sine1
                            if tla.const_expr(r >= 128):
                                if tla.const_expr(q1_rp == 0):
                                    q1_a = q1_x0
                                elif tla.const_expr(q1_rp == 1):
                                    q1_a = q1_x1
                                elif tla.const_expr(q1_rp == 2):
                                    q1_a = q1_x2
                                elif tla.const_expr(q1_rp == 3):
                                    q1_a = q1_x3
                                if tla.const_expr(r // 128 + q1_rp == 0):
                                    q1_z = q1_x0
                                elif tla.const_expr(r // 128 + q1_rp == 1):
                                    q1_z = q1_x1
                                elif tla.const_expr(r // 128 + q1_rp == 2):
                                    q1_z = q1_x2
                                elif tla.const_expr(r // 128 + q1_rp == 3):
                                    q1_z = q1_x3
                            else:
                                q1_a = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (q1_h1 * stride * d + q1_rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                q1_z = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (q1_h1 * stride * d + r // 2 + q1_rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                            q1_wa = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + q1_rp * 64 // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q1_wz = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + (r // 2 + q1_rp * 64) // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q1_a = (q1_a * q1_inv) * q1_wa
                            q1_z = (q1_z * q1_inv) * q1_wz
                            if tla.const_expr(bias):
                                q1_ba = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + q1_rp * 64 // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                q1_bz = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + (r // 2 + q1_rp * 64) // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                q1_a = q1_a + q1_ba
                                q1_z = q1_z + q1_bz
                            q1_left = q1_a * q1_cosine - q1_z * q1_sine
                            q1_right = q1_z * q1_cosine + q1_a * q1_sine
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (q1_h1 * d + q1_rp * 64) // min(64, r // 2)
                                ),
                            ).store(narrow(q1_left, dtype), pack)
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (q1_h1 * d + r // 2 + q1_rp * 64) // min(64, r // 2)
                                ),
                            ).store(narrow(q1_right, dtype), pack)
                    # Tail is normalized once; the rotary prefix stays in FP32 registers.
                    for q2_part in tla.range_constexpr((d - r) // 64):
                        if tla.const_expr(r // 64 + q2_part == 0):
                            q2_v = q2_x0
                        elif tla.const_expr(r // 64 + q2_part == 1):
                            q2_v = q2_x1
                        elif tla.const_expr(r // 64 + q2_part == 2):
                            q2_v = q2_x2
                        elif tla.const_expr(r // 64 + q2_part == 3):
                            q2_v = q2_x3
                        q2_weight = (
                            tla.tile_view(
                                w,
                                tla.make_shape(64),
                                tla.make_coord(woff // 64 + r // 64 + q2_part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        q2_norm = (q2_v * q2_inv) * q2_weight
                        if tla.const_expr(bias):
                            q2_bv = (
                                tla.tile_view(
                                    b,
                                    tla.make_shape(64),
                                    tla.make_coord(woff // 64 + r // 64 + q2_part),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q2_norm = q2_norm + q2_bv
                        tla.tile_view(
                            y,
                            tla.make_shape(64),
                            tla.make_coord((q2_h1 * d + r) // 64 + q2_part),
                        ).store(narrow(q2_norm, dtype), pack)
                    if tla.const_expr(gate):
                        if slot < nq:
                            for q2_cp in tla.range_constexpr(d // 128):
                                q2_raw = tla.tile_view(
                                    x,
                                    tla.make_shape(128),
                                    tla.make_coord(
                                        ((q2_h1 * 2 + 1) * d) // 128 + q2_cp
                                    ),
                                ).load()
                                tla.tile_view(
                                    aux,
                                    tla.make_shape(128),
                                    tla.make_coord(q2_h1 * d // 128 + q2_cp),
                                ).store(q2_raw)
                    if tla.const_expr(r == 64):
                        q2_full_weight = (
                            tla.tile_view(
                                w, tla.make_shape(64), tla.make_coord(woff // 64)
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        q2_full_norm = (q2_x0 * q2_inv) * q2_full_weight
                        if tla.const_expr(bias):
                            q2_full_bias = (
                                tla.tile_view(
                                    b, tla.make_shape(64), tla.make_coord(woff // 64)
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q2_full_norm = q2_full_norm + q2_full_bias
                        q2_partner = tla.gather(q2_full_norm, swap_index)
                        q2_full_out = q2_full_norm * cosine0 + q2_partner * sine_signed0
                        tla.tile_view(
                            y, tla.make_shape(64), tla.make_coord(q2_h1 * d // 64)
                        ).store(narrow(q2_full_out, dtype), pack)
                    else:
                        for q2_rp in tla.range_constexpr((r // 2 + 63) // 64):
                            if tla.const_expr(q2_rp == 0):
                                q2_cosine = cosine0
                                q2_sine = sine0
                            else:
                                q2_cosine = cosine1
                                q2_sine = sine1
                            if tla.const_expr(r >= 128):
                                if tla.const_expr(q2_rp == 0):
                                    q2_a = q2_x0
                                elif tla.const_expr(q2_rp == 1):
                                    q2_a = q2_x1
                                elif tla.const_expr(q2_rp == 2):
                                    q2_a = q2_x2
                                elif tla.const_expr(q2_rp == 3):
                                    q2_a = q2_x3
                                if tla.const_expr(r // 128 + q2_rp == 0):
                                    q2_z = q2_x0
                                elif tla.const_expr(r // 128 + q2_rp == 1):
                                    q2_z = q2_x1
                                elif tla.const_expr(r // 128 + q2_rp == 2):
                                    q2_z = q2_x2
                                elif tla.const_expr(r // 128 + q2_rp == 3):
                                    q2_z = q2_x3
                            else:
                                q2_a = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (q2_h1 * stride * d + q2_rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                q2_z = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (q2_h1 * stride * d + r // 2 + q2_rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                            q2_wa = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + q2_rp * 64 // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q2_wz = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + (r // 2 + q2_rp * 64) // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q2_a = (q2_a * q2_inv) * q2_wa
                            q2_z = (q2_z * q2_inv) * q2_wz
                            if tla.const_expr(bias):
                                q2_ba = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + q2_rp * 64 // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                q2_bz = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + (r // 2 + q2_rp * 64) // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                q2_a = q2_a + q2_ba
                                q2_z = q2_z + q2_bz
                            q2_left = q2_a * q2_cosine - q2_z * q2_sine
                            q2_right = q2_z * q2_cosine + q2_a * q2_sine
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (q2_h1 * d + q2_rp * 64) // min(64, r // 2)
                                ),
                            ).store(narrow(q2_left, dtype), pack)
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (q2_h1 * d + r // 2 + q2_rp * 64) // min(64, r // 2)
                                ),
                            ).store(narrow(q2_right, dtype), pack)
                    # Tail is normalized once; the rotary prefix stays in FP32 registers.
                    for q3_part in tla.range_constexpr((d - r) // 64):
                        if tla.const_expr(r // 64 + q3_part == 0):
                            q3_v = q3_x0
                        elif tla.const_expr(r // 64 + q3_part == 1):
                            q3_v = q3_x1
                        elif tla.const_expr(r // 64 + q3_part == 2):
                            q3_v = q3_x2
                        elif tla.const_expr(r // 64 + q3_part == 3):
                            q3_v = q3_x3
                        q3_weight = (
                            tla.tile_view(
                                w,
                                tla.make_shape(64),
                                tla.make_coord(woff // 64 + r // 64 + q3_part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        q3_norm = (q3_v * q3_inv) * q3_weight
                        if tla.const_expr(bias):
                            q3_bv = (
                                tla.tile_view(
                                    b,
                                    tla.make_shape(64),
                                    tla.make_coord(woff // 64 + r // 64 + q3_part),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q3_norm = q3_norm + q3_bv
                        tla.tile_view(
                            y,
                            tla.make_shape(64),
                            tla.make_coord((q3_h1 * d + r) // 64 + q3_part),
                        ).store(narrow(q3_norm, dtype), pack)
                    if tla.const_expr(gate):
                        if slot < nq:
                            for q3_cp in tla.range_constexpr(d // 128):
                                q3_raw = tla.tile_view(
                                    x,
                                    tla.make_shape(128),
                                    tla.make_coord(
                                        ((q3_h1 * 2 + 1) * d) // 128 + q3_cp
                                    ),
                                ).load()
                                tla.tile_view(
                                    aux,
                                    tla.make_shape(128),
                                    tla.make_coord(q3_h1 * d // 128 + q3_cp),
                                ).store(q3_raw)
                    if tla.const_expr(r == 64):
                        q3_full_weight = (
                            tla.tile_view(
                                w, tla.make_shape(64), tla.make_coord(woff // 64)
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        q3_full_norm = (q3_x0 * q3_inv) * q3_full_weight
                        if tla.const_expr(bias):
                            q3_full_bias = (
                                tla.tile_view(
                                    b, tla.make_shape(64), tla.make_coord(woff // 64)
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q3_full_norm = q3_full_norm + q3_full_bias
                        q3_partner = tla.gather(q3_full_norm, swap_index)
                        q3_full_out = q3_full_norm * cosine0 + q3_partner * sine_signed0
                        tla.tile_view(
                            y, tla.make_shape(64), tla.make_coord(q3_h1 * d // 64)
                        ).store(narrow(q3_full_out, dtype), pack)
                    else:
                        for q3_rp in tla.range_constexpr((r // 2 + 63) // 64):
                            if tla.const_expr(q3_rp == 0):
                                q3_cosine = cosine0
                                q3_sine = sine0
                            else:
                                q3_cosine = cosine1
                                q3_sine = sine1
                            if tla.const_expr(r >= 128):
                                if tla.const_expr(q3_rp == 0):
                                    q3_a = q3_x0
                                elif tla.const_expr(q3_rp == 1):
                                    q3_a = q3_x1
                                elif tla.const_expr(q3_rp == 2):
                                    q3_a = q3_x2
                                elif tla.const_expr(q3_rp == 3):
                                    q3_a = q3_x3
                                if tla.const_expr(r // 128 + q3_rp == 0):
                                    q3_z = q3_x0
                                elif tla.const_expr(r // 128 + q3_rp == 1):
                                    q3_z = q3_x1
                                elif tla.const_expr(r // 128 + q3_rp == 2):
                                    q3_z = q3_x2
                                elif tla.const_expr(r // 128 + q3_rp == 3):
                                    q3_z = q3_x3
                            else:
                                q3_a = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (q3_h1 * stride * d + q3_rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                q3_z = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (q3_h1 * stride * d + r // 2 + q3_rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                            q3_wa = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + q3_rp * 64 // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q3_wz = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + (r // 2 + q3_rp * 64) // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            q3_a = (q3_a * q3_inv) * q3_wa
                            q3_z = (q3_z * q3_inv) * q3_wz
                            if tla.const_expr(bias):
                                q3_ba = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + q3_rp * 64 // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                q3_bz = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + (r // 2 + q3_rp * 64) // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                q3_a = q3_a + q3_ba
                                q3_z = q3_z + q3_bz
                            q3_left = q3_a * q3_cosine - q3_z * q3_sine
                            q3_right = q3_z * q3_cosine + q3_a * q3_sine
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (q3_h1 * d + q3_rp * 64) // min(64, r // 2)
                                ),
                            ).store(narrow(q3_left, dtype), pack)
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (q3_h1 * d + r // 2 + q3_rp * 64) // min(64, r // 2)
                                ),
                            ).store(narrow(q3_right, dtype), pack)
                if heads == 2:
                    p0_h1 = 0
                    p1_h1 = 1
                    p0_sums = tla.full(0.0, dtype=tla.Float32)
                    for p0_part in tla.range_constexpr(d // 64):
                        p0_v = (
                            tla.tile_view(
                                x,
                                tla.make_shape(64),
                                tla.make_coord(p0_h1 * stride * d // 64 + p0_part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        if tla.const_expr(p0_part == 0):
                            p0_x0 = p0_v
                        elif tla.const_expr(p0_part == 1):
                            p0_x1 = p0_v
                        elif tla.const_expr(p0_part == 2):
                            p0_x2 = p0_v
                        elif tla.const_expr(p0_part == 3):
                            p0_x3 = p0_v
                        p0_sums = p0_sums + p0_v * p0_v
                    p0_total = p0_sums.reduce(
                        tla.ReductionOp.ADD,
                        mask=tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32),
                    )
                    p0_variance = (
                        tla.full(p0_total, dtype=tla.Float32) * (1.0 / d) + epsilon
                    )
                    p0_inv = tla.div(
                        tla.full(1.0, dtype=tla.Float32), tla.sqrt(p0_variance)
                    )
                    p1_sums = tla.full(0.0, dtype=tla.Float32)
                    for p1_part in tla.range_constexpr(d // 64):
                        p1_v = (
                            tla.tile_view(
                                x,
                                tla.make_shape(64),
                                tla.make_coord(p1_h1 * stride * d // 64 + p1_part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        if tla.const_expr(p1_part == 0):
                            p1_x0 = p1_v
                        elif tla.const_expr(p1_part == 1):
                            p1_x1 = p1_v
                        elif tla.const_expr(p1_part == 2):
                            p1_x2 = p1_v
                        elif tla.const_expr(p1_part == 3):
                            p1_x3 = p1_v
                        p1_sums = p1_sums + p1_v * p1_v
                    p1_total = p1_sums.reduce(
                        tla.ReductionOp.ADD,
                        mask=tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32),
                    )
                    p1_variance = (
                        tla.full(p1_total, dtype=tla.Float32) * (1.0 / d) + epsilon
                    )
                    p1_inv = tla.div(
                        tla.full(1.0, dtype=tla.Float32), tla.sqrt(p1_variance)
                    )
                    # Tail is normalized once; the rotary prefix stays in FP32 registers.
                    for p0_part in tla.range_constexpr((d - r) // 64):
                        if tla.const_expr(r // 64 + p0_part == 0):
                            p0_v = p0_x0
                        elif tla.const_expr(r // 64 + p0_part == 1):
                            p0_v = p0_x1
                        elif tla.const_expr(r // 64 + p0_part == 2):
                            p0_v = p0_x2
                        elif tla.const_expr(r // 64 + p0_part == 3):
                            p0_v = p0_x3
                        p0_weight = (
                            tla.tile_view(
                                w,
                                tla.make_shape(64),
                                tla.make_coord(woff // 64 + r // 64 + p0_part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        p0_norm = (p0_v * p0_inv) * p0_weight
                        if tla.const_expr(bias):
                            p0_bv = (
                                tla.tile_view(
                                    b,
                                    tla.make_shape(64),
                                    tla.make_coord(woff // 64 + r // 64 + p0_part),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            p0_norm = p0_norm + p0_bv
                        tla.tile_view(
                            y,
                            tla.make_shape(64),
                            tla.make_coord((p0_h1 * d + r) // 64 + p0_part),
                        ).store(narrow(p0_norm, dtype), pack)
                    if tla.const_expr(gate):
                        if slot < nq:
                            for p0_cp in tla.range_constexpr(d // 128):
                                p0_raw = tla.tile_view(
                                    x,
                                    tla.make_shape(128),
                                    tla.make_coord(
                                        ((p0_h1 * 2 + 1) * d) // 128 + p0_cp
                                    ),
                                ).load()
                                tla.tile_view(
                                    aux,
                                    tla.make_shape(128),
                                    tla.make_coord(p0_h1 * d // 128 + p0_cp),
                                ).store(p0_raw)
                    if tla.const_expr(r == 64):
                        p0_full_weight = (
                            tla.tile_view(
                                w, tla.make_shape(64), tla.make_coord(woff // 64)
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        p0_full_norm = (p0_x0 * p0_inv) * p0_full_weight
                        if tla.const_expr(bias):
                            p0_full_bias = (
                                tla.tile_view(
                                    b, tla.make_shape(64), tla.make_coord(woff // 64)
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            p0_full_norm = p0_full_norm + p0_full_bias
                        p0_partner = tla.gather(p0_full_norm, swap_index)
                        p0_full_out = p0_full_norm * cosine0 + p0_partner * sine_signed0
                        tla.tile_view(
                            y, tla.make_shape(64), tla.make_coord(p0_h1 * d // 64)
                        ).store(narrow(p0_full_out, dtype), pack)
                    else:
                        for p0_rp in tla.range_constexpr((r // 2 + 63) // 64):
                            if tla.const_expr(p0_rp == 0):
                                p0_cosine = cosine0
                                p0_sine = sine0
                            else:
                                p0_cosine = cosine1
                                p0_sine = sine1
                            if tla.const_expr(r >= 128):
                                if tla.const_expr(p0_rp == 0):
                                    p0_a = p0_x0
                                elif tla.const_expr(p0_rp == 1):
                                    p0_a = p0_x1
                                elif tla.const_expr(p0_rp == 2):
                                    p0_a = p0_x2
                                elif tla.const_expr(p0_rp == 3):
                                    p0_a = p0_x3
                                if tla.const_expr(r // 128 + p0_rp == 0):
                                    p0_z = p0_x0
                                elif tla.const_expr(r // 128 + p0_rp == 1):
                                    p0_z = p0_x1
                                elif tla.const_expr(r // 128 + p0_rp == 2):
                                    p0_z = p0_x2
                                elif tla.const_expr(r // 128 + p0_rp == 3):
                                    p0_z = p0_x3
                            else:
                                p0_a = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (p0_h1 * stride * d + p0_rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                p0_z = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (p0_h1 * stride * d + r // 2 + p0_rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                            p0_wa = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + p0_rp * 64 // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            p0_wz = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + (r // 2 + p0_rp * 64) // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            p0_a = (p0_a * p0_inv) * p0_wa
                            p0_z = (p0_z * p0_inv) * p0_wz
                            if tla.const_expr(bias):
                                p0_ba = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + p0_rp * 64 // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                p0_bz = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + (r // 2 + p0_rp * 64) // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                p0_a = p0_a + p0_ba
                                p0_z = p0_z + p0_bz
                            p0_left = p0_a * p0_cosine - p0_z * p0_sine
                            p0_right = p0_z * p0_cosine + p0_a * p0_sine
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (p0_h1 * d + p0_rp * 64) // min(64, r // 2)
                                ),
                            ).store(narrow(p0_left, dtype), pack)
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (p0_h1 * d + r // 2 + p0_rp * 64) // min(64, r // 2)
                                ),
                            ).store(narrow(p0_right, dtype), pack)
                    # Tail is normalized once; the rotary prefix stays in FP32 registers.
                    for p1_part in tla.range_constexpr((d - r) // 64):
                        if tla.const_expr(r // 64 + p1_part == 0):
                            p1_v = p1_x0
                        elif tla.const_expr(r // 64 + p1_part == 1):
                            p1_v = p1_x1
                        elif tla.const_expr(r // 64 + p1_part == 2):
                            p1_v = p1_x2
                        elif tla.const_expr(r // 64 + p1_part == 3):
                            p1_v = p1_x3
                        p1_weight = (
                            tla.tile_view(
                                w,
                                tla.make_shape(64),
                                tla.make_coord(woff // 64 + r // 64 + p1_part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        p1_norm = (p1_v * p1_inv) * p1_weight
                        if tla.const_expr(bias):
                            p1_bv = (
                                tla.tile_view(
                                    b,
                                    tla.make_shape(64),
                                    tla.make_coord(woff // 64 + r // 64 + p1_part),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            p1_norm = p1_norm + p1_bv
                        tla.tile_view(
                            y,
                            tla.make_shape(64),
                            tla.make_coord((p1_h1 * d + r) // 64 + p1_part),
                        ).store(narrow(p1_norm, dtype), pack)
                    if tla.const_expr(gate):
                        if slot < nq:
                            for p1_cp in tla.range_constexpr(d // 128):
                                p1_raw = tla.tile_view(
                                    x,
                                    tla.make_shape(128),
                                    tla.make_coord(
                                        ((p1_h1 * 2 + 1) * d) // 128 + p1_cp
                                    ),
                                ).load()
                                tla.tile_view(
                                    aux,
                                    tla.make_shape(128),
                                    tla.make_coord(p1_h1 * d // 128 + p1_cp),
                                ).store(p1_raw)
                    if tla.const_expr(r == 64):
                        p1_full_weight = (
                            tla.tile_view(
                                w, tla.make_shape(64), tla.make_coord(woff // 64)
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        p1_full_norm = (p1_x0 * p1_inv) * p1_full_weight
                        if tla.const_expr(bias):
                            p1_full_bias = (
                                tla.tile_view(
                                    b, tla.make_shape(64), tla.make_coord(woff // 64)
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            p1_full_norm = p1_full_norm + p1_full_bias
                        p1_partner = tla.gather(p1_full_norm, swap_index)
                        p1_full_out = p1_full_norm * cosine0 + p1_partner * sine_signed0
                        tla.tile_view(
                            y, tla.make_shape(64), tla.make_coord(p1_h1 * d // 64)
                        ).store(narrow(p1_full_out, dtype), pack)
                    else:
                        for p1_rp in tla.range_constexpr((r // 2 + 63) // 64):
                            if tla.const_expr(p1_rp == 0):
                                p1_cosine = cosine0
                                p1_sine = sine0
                            else:
                                p1_cosine = cosine1
                                p1_sine = sine1
                            if tla.const_expr(r >= 128):
                                if tla.const_expr(p1_rp == 0):
                                    p1_a = p1_x0
                                elif tla.const_expr(p1_rp == 1):
                                    p1_a = p1_x1
                                elif tla.const_expr(p1_rp == 2):
                                    p1_a = p1_x2
                                elif tla.const_expr(p1_rp == 3):
                                    p1_a = p1_x3
                                if tla.const_expr(r // 128 + p1_rp == 0):
                                    p1_z = p1_x0
                                elif tla.const_expr(r // 128 + p1_rp == 1):
                                    p1_z = p1_x1
                                elif tla.const_expr(r // 128 + p1_rp == 2):
                                    p1_z = p1_x2
                                elif tla.const_expr(r // 128 + p1_rp == 3):
                                    p1_z = p1_x3
                            else:
                                p1_a = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (p1_h1 * stride * d + p1_rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                p1_z = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (p1_h1 * stride * d + r // 2 + p1_rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                            p1_wa = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + p1_rp * 64 // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            p1_wz = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + (r // 2 + p1_rp * 64) // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            p1_a = (p1_a * p1_inv) * p1_wa
                            p1_z = (p1_z * p1_inv) * p1_wz
                            if tla.const_expr(bias):
                                p1_ba = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + p1_rp * 64 // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                p1_bz = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + (r // 2 + p1_rp * 64) // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                p1_a = p1_a + p1_ba
                                p1_z = p1_z + p1_bz
                            p1_left = p1_a * p1_cosine - p1_z * p1_sine
                            p1_right = p1_z * p1_cosine + p1_a * p1_sine
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (p1_h1 * d + p1_rp * 64) // min(64, r // 2)
                                ),
                            ).store(narrow(p1_left, dtype), pack)
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (p1_h1 * d + r // 2 + p1_rp * 64) // min(64, r // 2)
                                ),
                            ).store(narrow(p1_right, dtype), pack)
                if heads == 1:
                    h1 = 0
                    sums = tla.full(0.0, dtype=tla.Float32)
                    for part in tla.range_constexpr(d // 64):
                        v = (
                            tla.tile_view(
                                x,
                                tla.make_shape(64),
                                tla.make_coord(h1 * stride * d // 64 + part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        if tla.const_expr(part == 0):
                            x0 = v
                        elif tla.const_expr(part == 1):
                            x1 = v
                        elif tla.const_expr(part == 2):
                            x2 = v
                        elif tla.const_expr(part == 3):
                            x3 = v
                        sums = sums + v * v
                    total = sums.reduce(
                        tla.ReductionOp.ADD,
                        mask=tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32),
                    )
                    variance = tla.full(total, dtype=tla.Float32) * (1.0 / d) + epsilon
                    inv = tla.div(tla.full(1.0, dtype=tla.Float32), tla.sqrt(variance))
                    # Tail is normalized once; the rotary prefix stays in FP32 registers.
                    for part in tla.range_constexpr((d - r) // 64):
                        if tla.const_expr(r // 64 + part == 0):
                            v = x0
                        elif tla.const_expr(r // 64 + part == 1):
                            v = x1
                        elif tla.const_expr(r // 64 + part == 2):
                            v = x2
                        elif tla.const_expr(r // 64 + part == 3):
                            v = x3
                        weight = (
                            tla.tile_view(
                                w,
                                tla.make_shape(64),
                                tla.make_coord(woff // 64 + r // 64 + part),
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        norm = (v * inv) * weight
                        if tla.const_expr(bias):
                            bv = (
                                tla.tile_view(
                                    b,
                                    tla.make_shape(64),
                                    tla.make_coord(woff // 64 + r // 64 + part),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            norm = norm + bv
                        tla.tile_view(
                            y,
                            tla.make_shape(64),
                            tla.make_coord((h1 * d + r) // 64 + part),
                        ).store(narrow(norm, dtype), pack)
                    if tla.const_expr(gate):
                        if slot < nq:
                            for cp in tla.range_constexpr(d // 128):
                                raw = tla.tile_view(
                                    x,
                                    tla.make_shape(128),
                                    tla.make_coord(((h1 * 2 + 1) * d) // 128 + cp),
                                ).load()
                                tla.tile_view(
                                    aux,
                                    tla.make_shape(128),
                                    tla.make_coord(h1 * d // 128 + cp),
                                ).store(raw)
                    if tla.const_expr(r == 64):
                        full_weight = (
                            tla.tile_view(
                                w, tla.make_shape(64), tla.make_coord(woff // 64)
                            )
                            .load(unpack)
                            .to(tla.Float32, cast)
                        )
                        full_norm = (x0 * inv) * full_weight
                        if tla.const_expr(bias):
                            full_bias = (
                                tla.tile_view(
                                    b, tla.make_shape(64), tla.make_coord(woff // 64)
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            full_norm = full_norm + full_bias
                        partner = tla.gather(full_norm, swap_index)
                        full_out = full_norm * cosine0 + partner * sine_signed0
                        tla.tile_view(
                            y, tla.make_shape(64), tla.make_coord(h1 * d // 64)
                        ).store(narrow(full_out, dtype), pack)
                    else:
                        for rp in tla.range_constexpr((r // 2 + 63) // 64):
                            if tla.const_expr(rp == 0):
                                cosine = cosine0
                                sine = sine0
                            else:
                                cosine = cosine1
                                sine = sine1
                            if tla.const_expr(r >= 128):
                                if tla.const_expr(rp == 0):
                                    a = x0
                                elif tla.const_expr(rp == 1):
                                    a = x1
                                elif tla.const_expr(rp == 2):
                                    a = x2
                                elif tla.const_expr(rp == 3):
                                    a = x3
                                if tla.const_expr(r // 128 + rp == 0):
                                    z = x0
                                elif tla.const_expr(r // 128 + rp == 1):
                                    z = x1
                                elif tla.const_expr(r // 128 + rp == 2):
                                    z = x2
                                elif tla.const_expr(r // 128 + rp == 3):
                                    z = x3
                            else:
                                a = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (h1 * stride * d + rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                z = (
                                    tla.tile_view(
                                        x,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            (h1 * stride * d + r // 2 + rp * 64)
                                            // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                            wa = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + rp * 64 // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            wz = (
                                tla.tile_view(
                                    w,
                                    tla.make_shape(min(64, r // 2)),
                                    tla.make_coord(
                                        woff // min(64, r // 2)
                                        + (r // 2 + rp * 64) // min(64, r // 2)
                                    ),
                                )
                                .load(unpack)
                                .to(tla.Float32, cast)
                            )
                            a = (a * inv) * wa
                            z = (z * inv) * wz
                            if tla.const_expr(bias):
                                ba = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + rp * 64 // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                bz = (
                                    tla.tile_view(
                                        b,
                                        tla.make_shape(min(64, r // 2)),
                                        tla.make_coord(
                                            woff // min(64, r // 2)
                                            + (r // 2 + rp * 64) // min(64, r // 2)
                                        ),
                                    )
                                    .load(unpack)
                                    .to(tla.Float32, cast)
                                )
                                a = a + ba
                                z = z + bz
                            left = a * cosine - z * sine
                            right = z * cosine + a * sine
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord((h1 * d + rp * 64) // min(64, r // 2)),
                            ).store(narrow(left, dtype), pack)
                            tla.tile_view(
                                y,
                                tla.make_shape(min(64, r // 2)),
                                tla.make_coord(
                                    (h1 * d + r // 2 + rp * 64) // min(64, r // 2)
                                ),
                            ).store(narrow(right, dtype), pack)
            if buffer == 0:
                tla.set_flag(done0)
                tla.wait_flag(done0)
            elif buffer == 1:
                tla.set_flag(done1)
                tla.wait_flag(done1)
            else:
                tla.set_flag(done2)
                tla.wait_flag(done2)
            if slot < nq:
                tla.copy(
                    tla.make_tensor(pq.ptr + (token * hq + base) * d, layout_yq), y
                )
                if tla.const_expr(gate):
                    tla.copy(
                        tla.make_tensor(pg.ptr + (token * hq + base) * d, layout_yq),
                        aux,
                    )
            else:
                tla.copy(
                    tla.make_tensor(pk.ptr + (token * hk + base) * d, layout_yk),
                    tla.tile_view(y, tla.make_shape(gk * d), tla.make_coord(0)),
                )
                tla.copy(
                    tla.make_tensor(pv.ptr + (token * hk + base) * d, layout_yk),
                    tla.tile_view(aux, tla.make_shape(gk * d), tla.make_coord(0)),
                )
            if buffer == 0:
                tla.set_flag(free0)
            elif buffer == 1:
                tla.set_flag(free1)
            else:
                tla.set_flag(free2)
            buffer = buffer + 1
            if buffer == 3:
                buffer = 0
        tla.wait_flag(free0)
        tla.wait_flag(free1)
        tla.wait_flag(free2)
        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host: validation, compilation cache and launch
# ---------------------------------------------------------------------------


def _tla_tensor(tensor):
    return from_dlpack(
        tensor.view(-1), layout_tag=tla.arch.RowMajor
    ).mark_compact_shape_dynamic(0)


@lru_cache(maxsize=128)
def _compiled(dtype_name, config, group, device_index):
    # device_index keeps cached executors separate for each device.
    dtype = tla.Float16 if dtype_name == "float16" else tla.BFloat16
    samples = [
        tla.make_fake_tensor(
            dtype, (256,), (1,), layout_tag=tla.arch.RowMajor
        ).mark_compact_shape_dynamic(0)
        for _ in range(10)
    ]
    return tla.compile(
        split_qkv_rmsnorm_mrope_kernel,
        *samples,
        tla.Int32(1),
        *config,
        dtype,
        round_to_even,
        group,
        options="--npu-arch 3510",
    )


def _check(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    num_q_heads,
    num_kv_heads,
    head_size,
    eps,
    mrope_section,
    is_interleaved,
    rope_dim,
    q_bias,
    k_bias,
    has_gate,
):
    for name, v in [
        ("num_q_heads", num_q_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_size", head_size),
    ]:
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0 or v & (v - 1):
            raise ValueError(f"{name} must be a positive power of two")
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if not isinstance(is_interleaved, bool) or not isinstance(has_gate, bool):
        raise ValueError("is_interleaved and has_gate must be bool")
    if not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    r = head_size if rope_dim is None else rope_dim
    if (
        not isinstance(r, int)
        or isinstance(r, bool)
        or not 2 <= r <= head_size
        or r & (r - 1)
    ):
        raise ValueError("rope_dim must be a power of two in [2, head_size]")
    # SIMD implementation explicitly supports the README workload domain.
    if head_size not in (128, 256) or r not in (64, 128, 256):
        raise ValueError("supported head_size: 128/256; rope_dim: 64/128/256")
    if (
        not isinstance(mrope_section, (list, tuple))
        or len(mrope_section) != 3
        or any(
            not isinstance(s, int) or isinstance(s, bool) or s < 0
            for s in mrope_section
        )
        or 2 * sum(mrope_section) != r
    ):
        raise ValueError(
            "mrope_section must be three nonnegative ints summing to rope_dim/2"
        )
    if not isinstance(qkv, torch.Tensor) or qkv.ndim != 2 or qkv.shape[0] <= 0:
        raise ValueError("qkv must have shape [T>0, packed_size]")
    if qkv.device.type != "npu" or qkv.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("qkv must be an f16/bf16 Ascend NPU tensor")
    if qkv.device.index != torch.npu.current_device():
        raise ValueError(
            "qkv must be on the current Ascend NPU; call torch.npu.set_device first"
        )
    packed = ((2 if has_gate else 1) * num_q_heads + 2 * num_kv_heads) * head_size
    if qkv.shape[1] != packed:
        raise ValueError("qkv packed width does not match configuration")
    if (q_bias is None) != (k_bias is None):
        raise ValueError("q_bias and k_bias must both be provided or both be None")
    for name, tensor in [
        ("qkv", qkv),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("cos_sin", cos_sin),
        ("q_bias", q_bias),
        ("k_bias", k_bias),
    ]:
        if tensor is None and name in ("q_bias", "k_bias"):
            continue
        if (
            not isinstance(tensor, torch.Tensor)
            or not tensor.is_contiguous()
            or tensor.device != qkv.device
            or tensor.dtype != qkv.dtype
        ):
            raise ValueError(f"{name} must be contiguous with qkv dtype and device")
        if name not in ("qkv", "cos_sin") and tuple(tensor.shape) != (head_size,):
            raise ValueError(f"{name} must have shape [head_size]")
    if tuple(cos_sin.shape) != (3, qkv.shape[0], r):
        raise ValueError("cos_sin must have shape [3,T,rope_dim]")
    return r


@lru_cache(maxsize=16)
def _vector_cores(device_index):
    props = torch.npu.get_device_properties(device_index)
    count = int(props.vector_core_num)
    if count <= 0:
        raise RuntimeError("cannot determine physical vector core count")
    return count


def run(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    num_q_heads,
    num_kv_heads,
    head_size,
    eps,
    mrope_section,
    is_interleaved,
    rope_dim=None,
    q_bias=None,
    k_bias=None,
    has_gate=False,
):
    """Return Q, K, V and gate on the input device and current stream."""
    r = _check(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        num_q_heads,
        num_kv_heads,
        head_size,
        eps,
        mrope_section,
        is_interleaved,
        rope_dim,
        q_bias,
        k_bias,
        has_gate,
    )
    t = qkv.shape[0]
    q = torch.empty((t, num_q_heads * head_size), dtype=qkv.dtype, device=qkv.device)
    k = torch.empty((t, num_kv_heads * head_size), dtype=qkv.dtype, device=qkv.device)
    v = torch.empty((t, num_kv_heads * head_size), dtype=qkv.dtype, device=qkv.device)
    g = torch.empty(
        (t, num_q_heads * head_size if has_gate else 0),
        dtype=qkv.dtype,
        device=qkv.device,
    )
    config = (
        num_q_heads,
        num_kv_heads,
        head_size,
        r,
        float(eps),
        tuple(mrope_section),
        is_interleaved,
        q_bias is not None,
        has_gate,
    )
    group = num_q_heads if t >= 32 else 1
    artifact = _compiled(str(qkv.dtype).split(".")[-1], config, group, qkv.device.index)
    # Absent bias and empty gate are never dereferenced; pass a valid dummy descriptor.
    tensors = (
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        q_bias if q_bias is not None else q_weight,
        k_bias if k_bias is not None else k_weight,
        q,
        k,
        v,
        g if has_gate else q,
    )
    tasks = t * (
        num_q_heads // min(num_q_heads, group)
        + num_kv_heads // min(num_kv_heads, group)
    )
    blocks = min(_vector_cores(qkv.device.index), tasks)
    views = tuple(_tla_tensor(tensor) for tensor in tensors)
    artifact(*views, tla.Int32(t), block_num=blocks)
    return q, k, v, g
