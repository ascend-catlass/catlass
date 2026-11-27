# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ascend 950 Flash KDA forward device kernels.

Sections: shared address/Vector/Cube/state helpers; preprocessing and scores;
staged producer, recurrence and kernel; persistent math, pipeline and kernel.
Host-only ABI4 scheduling is implemented in the host section at the end of this file.
"""

import math
from collections.abc import Sequence
from threading import Lock
from types import SimpleNamespace

import torch
from torch import Tensor
from catlass import tla


@tla.jit
def mutex_event(mutex, pipe):
    mutex.lock(pipe=pipe)
    mutex.unlock(pipe=pipe)


# Shared address and layout helpers.


@tla.jit
def i64(value):
    return tla.Int64(value)


@tla.jit
def token_offset(batch, head, token, heads, length, layout):
    result = ((batch * i64(heads) + head) * i64(length) + token) * i64(128)
    if tla.const_expr(layout != "BNSD"):
        result = ((batch * i64(length) + token) * i64(heads) + head) * i64(128)
    return result


@tla.jit
def strided_rows(ptr, rows, columns, stride):
    return tla.make_tensor(
        ptr,
        tla.make_layout(
            tla.make_shape(rows, columns),
            tla.make_stride(stride, 1),
            layoutTag=tla.arch.RowMajor,
        ),
    )


# -----------------------------------------------------------------------------
# Common Vector primitives.
# -----------------------------------------------------------------------------


@tla.jit
def flat(ptr, count):
    return tla.make_tensor(
        ptr,
        tla.make_layout(
            tla.make_shape(count), tla.make_stride(1), layoutTag=tla.arch.RowMajor
        ),
    )


@tla.jit
def at(tensor, offset, count):
    # All flat accesses have offsets that are multiples of the requested extent.
    # Keep the allocation-backed tensor through VF lowering; do not rebuild a
    # tensor from a dynamically shifted UB pointer inside SIMD.
    return tla.tile_view(tensor, tla.make_shape(count), tla.make_coord(offset // count))


@tla.jit
def padded_nz(ptr, rows, columns=64):
    return tla.make_tensor(
        ptr,
        tla.make_layout(
            tla.make_shape((16, rows // 16), (16, columns // 16)),
            tla.make_stride((16, 256), (1, rows * 16 + 16)),
            origin_shape=tla.make_shape(rows, columns),
            layoutTag=tla.arch.zN,
        ),
    )


@tla.jit
def cast_bf16(value):
    return value.to(
        tla.BFloat16,
        tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO,
            sat_mode=tla.params.SatMode.NOSAT,
            round_mode=tla.params.RoundMode.CAST_RINT,
        ),
    )


@tla.jit
def load_bf16_f32(tile):
    return tile.load(
        tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_UNPACK_B16)
    ).to(
        tla.Float32,
        tla.params.CastParams(),
    )


@tla.jit
def broadcast(tile):
    return tile.load(
        tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
    )


@tla.jit
def store_bf16(tile, value, count=64):
    # Slot-zero BF16 register has two predicate lanes per compact destination
    # element (TLA test_store_dist.py). Extent is proven by each caller.
    mask, remaining = tla.update_mask(count * 2, dtype=tla.BFloat16)
    tile.store(
        cast_bf16(value),
        tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK_B32),
        mask=mask,
    )


@tla.jit
def cast_and_pack_bf16(value):
    # Source: vcast + vreinterpret_lanes(u32) + vpack(u16) + vreinterpret(bf16).
    # TLA bitcast cannot change lane width. Extract the even halfword lanes
    # after the native slot-zero cast. Only the low 64 lanes are consumed.
    converted = cast_bf16(value)
    packed, unused = tla.deinterleave(converted, converted)
    return packed


@tla.jit
def prepare_gate_input(raw, converted, valid=64):
    """The source public adapter first casts FP32 g to BF16."""
    with tla.vec.func(mode="simd"):
        for row in tla.range(64):
            value = tla.full(0.0, dtype=tla.Float32)
            if row < valid:
                if tla.const_expr(raw.ptr.dtype == tla.Float32):
                    value = at(raw, row * 64, 64).load()
                else:
                    value = load_bf16_f32(at(raw, row * 64, 64))
            store_bf16(at(converted, row * 64, 64), value)


@tla.jit
def zero_qkv_tail(q, k, v, valid):
    with tla.vec.func(mode="simd"):
        zero = tla.full(0.0, dtype=tla.Float32)
        for row in tla.range(valid, 64):
            store_bf16(at(q, row * 64, 64), zero)
            store_bf16(at(k, row * 64, 64), zero)
            store_bf16(at(v, row * 64, 64), zero)


@tla.jit
def prepare_beta_input(raw, converted, valid, pitch):
    with tla.vec.func(mode="simd"):
        full, unused = tla.update_mask(64, dtype=tla.Float32)
        if tla.const_expr(raw.ptr.dtype == tla.Float32):
            # Gather's result lane count follows the source *view*, while
            # offsets still address the full allocation (64*pitch elements).
            value = tla.gather(at(raw, 0, 64), tla.arange(0, dtype=tla.Int32) * pitch)
            mask, remaining = tla.update_mask(valid, dtype=tla.Float32)
            converted.store(
                tla.where(mask, value, tla.full(0.0, dtype=tla.Float32)), mask=full
            )
        else:
            for row in tla.range(64):
                value = tla.full(0.0, dtype=tla.Float32)
                if row < valid:
                    raw_value = at(raw, row * pitch, 1).load(
                        tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_BRC_B16
                        )
                    )
                    value = raw_value.to(
                        tla.Float32,
                        tla.params.CastParams(),
                    )
                at(converted, row, 1).store(
                    value,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                    ),
                )


@tla.jit
def beta_to_column(active, column):
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        full, unused = tla.update_mask(64, dtype=tla.Float32)
        for group in tla.range(8):
            values = at(active, group * 8, 8).load(
                tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_E2B_B32)
            )
            at(column, group * 64, 64).store(values, mask=full)


@tla.jit
def compute_qk_partial_sums(q_half, k_half, partials):
    """Source lines 1005–1018: 64 rows, 64 dimensions per AIV, unroll=4."""
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        lane_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        for group in tla.range(16):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                q = load_bf16_f32(at(q_half, row * 64, 64))
                k = load_bf16_f32(at(k_half, row * 64, 64))
                q_sum = (q * q).reduce(tla.ReductionOp.ADD, mask=lane_mask)
                k_sum = (k * k).reduce(tla.ReductionOp.ADD, mask=lane_mask)
                first = tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                )
                at(partials, row, 1).store(q_sum, first)
                at(partials, 64 + row, 1).store(k_sum, first)


@tla.jit
def compute_qk_inverse_vectors(local, peer, inverse):
    """Source lines 1021–1032: two full-register inverse-norm computations."""
    with tla.vec.func(mode="simd"):
        one = tla.full(1.0, dtype=tla.Float32)
        full, unused = tla.update_mask(64, dtype=tla.Float32)
        q_sum = at(local, 0, 64).load() + at(peer, 0, 64).load()
        k_sum = at(local, 64, 64).load() + at(peer, 64, 64).load()
        at(inverse, 0, 64).store(
            tla.min(one / tla.sqrt(q_sum + 1e-6), 1000.0), mask=full
        )
        at(inverse, 64, 64).store(
            tla.min(one / tla.sqrt(k_sum + 1e-6), 1000.0), mask=full
        )


@tla.jit
def normalize_local_qk(q_half, k_half, inverse, q_target=None, k_target=None):
    """Source normalization, optionally retaining distinct raw prefetch storage."""
    out_q = q_half
    out_k = k_half
    if tla.const_expr(q_target is not None):
        out_q = q_target
    if tla.const_expr(k_target is not None):
        out_k = k_target
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        for group in tla.range(16):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                q = load_bf16_f32(at(q_half, row * 64, 64))
                k = load_bf16_f32(at(k_half, row * 64, 64))
                store_bf16(at(out_q, row * 64, 64), q * broadcast(at(inverse, row, 1)))
                store_bf16(
                    at(out_k, row * 64, 64), k * broadcast(at(inverse, 64 + row, 1))
                )


@tla.jit
def normalize_and_snapshot(q, k, local, peer, inverse, k_snapshot):
    """Same norm arithmetic and RNE, fuse inverse and the delayed K snapshot."""
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        one = tla.full(1.0, dtype=tla.Float32)
        full, unused = tla.update_mask(64, dtype=tla.Float32)
        q_sum = at(local, 0, 64).load() + at(peer, 0, 64).load()
        k_sum = at(local, 64, 64).load() + at(peer, 64, 64).load()
        at(inverse, 0, 64).store(
            tla.min(one / tla.sqrt(q_sum + 1e-6), 1000.0), mask=full
        )
        at(inverse, 64, 64).store(
            tla.min(one / tla.sqrt(k_sum + 1e-6), 1000.0), mask=full
        )
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        for group in tla.range(16):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                qv = load_bf16_f32(at(q, row * 64, 64)) * broadcast(at(inverse, row, 1))
                kv = load_bf16_f32(at(k, row * 64, 64)) * broadcast(
                    at(inverse, 64 + row, 1)
                )
                store_bf16(at(q, row * 64, 64), qv)
                store_bf16(at(k, row * 64, 64), kv)
                store_bf16(at(k_snapshot, row * 64, 64), kv)


@tla.jit
def copy_bf16_qk(q, k, out_q, out_k):
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        mask, remaining = tla.update_mask(64, dtype=tla.BFloat16)
        for row in tla.range(64):
            at(out_q, row * 64, 64).store(at(q, row * 64, 64).load(), mask=mask)
            at(out_k, row * 64, 64).store(at(k, row * 64, 64).load(), mask=mask)


@tla.jit
def gate_row(raw, row, rate, dt, mode, lower_bound):
    if tla.const_expr(raw.ptr.dtype == tla.Float32):
        value = cast_bf16(at(raw, row * 64, 64).load()).to(
            tla.Float32,
            tla.params.CastParams(),
        )
    else:
        value = load_bf16_f32(at(raw, row * 64, 64))
    gate = value
    if tla.const_expr(mode == 1):
        gate = tla.full(lower_bound, dtype=tla.Float32) / (
            1.0 + tla.exp((-1.0 * rate) * (value + dt))
        )
    elif tla.const_expr(mode == 2):
        x = value + dt
        gate = (-1.0 * rate) * (
            tla.max(x, 0.0) + tla.log(1.0 + tla.exp(-1.0 * tla.abs(x)))
        )
    return gate


@tla.jit
def activate_gate(raw, bias, alpha, snapshot, valid_rows, mode, lower_bound, has_bias):
    """Source 1575/1602 and preprocessed branch; sequential FP32 cumsum."""
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        full, unused = tla.update_mask(64, dtype=tla.Float32)
        rate = tla.exp(broadcast(alpha))
        dt = tla.full(0.0, dtype=tla.Float32)
        if tla.const_expr(has_bias):
            dt = bias.load()
        prefix = tla.full(0.0, dtype=tla.Float32)
        groups = valid_rows // 4
        for group in tla.range(groups):
            for lane in tla.range_constexpr(4):
                group_row = group * 4 + lane
                prefix = prefix + gate_row(raw, group_row, rate, dt, mode, lower_bound)
                at(snapshot, group_row * 64, 64).store(prefix, mask=full)
        for remainder_row in tla.range(groups * 4, valid_rows):
            prefix = prefix + gate_row(raw, remainder_row, rate, dt, mode, lower_bound)
            at(snapshot, remainder_row * 64, 64).store(prefix, mask=full)
        for tail_row in tla.range(valid_rows, 64):
            at(snapshot, tail_row * 64, 64).store(prefix, mask=full)


@tla.jit
def activate_beta(raw, beta, valid_rows, use_sigmoid, allow_negative):
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        value = raw.load()
        if tla.const_expr(use_sigmoid):
            value = tla.full(1.0, dtype=tla.Float32) / (1.0 + tla.exp(-1.0 * value))
            if tla.const_expr(allow_negative):
                value = value * 2.0
        mask, remaining = tla.update_mask(valid_rows, dtype=tla.Float32)
        full, unused = tla.update_mask(64, dtype=tla.Float32)
        beta.store(tla.where(mask, value, tla.full(0.0, dtype=tla.Float32)), mask=full)


@tla.jit
def activate_beta_f32_input(raw, beta, valid_rows, pitch, use_sigmoid, allow_negative):
    """Load the same FP32 logits directly; caller protects the DMA input lifetime."""
    with tla.vec.func(mode="simd"):
        if tla.const_expr(pitch == 1):
            value = at(raw, 0, 64).load()
        else:
            value = tla.gather(at(raw, 0, 64), tla.arange(0, dtype=tla.Int32) * pitch)
        if tla.const_expr(use_sigmoid):
            value = tla.full(1.0, dtype=tla.Float32) / (1.0 + tla.exp(-1.0 * value))
            if tla.const_expr(allow_negative):
                value = value * 2.0
        mask, remaining = tla.update_mask(valid_rows, dtype=tla.Float32)
        full, unused = tla.update_mask(64, dtype=tla.Float32)
        beta.store(tla.where(mask, value, tla.full(0.0, dtype=tla.Float32)), mask=full)


@tla.jit
def store_shared_score_operand(value, destination, tile, row):
    low, remaining = tla.update_mask(64, dtype=tla.BFloat16)
    tla.tile_view(
        destination, tla.make_shape(1, 128), tla.make_coord(tile * 16 + row, 0)
    ).store(
        cast_and_pack_bf16(value),
        tla.params.BlockStoreParams(block_stride=225),
        mask=low,
    )


@tla.jit
def pack_shared_score_panel(q_half, k_half, prefix, destination, panel_idx, scale):
    """Source lines 1762–1825; 14 panels in padded NZ UB, row unroll=2."""
    packed_destination = padded_nz(destination.ptr, 224)
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        front = at(prefix, 16 * 64, 64).load()
        back = at(prefix, 48 * 64, 64).load()
        reference = front
        if tla.const_expr(panel_idx == 1):
            reference = back
        for half in tla.range_constexpr(2):
            col_block = panel_idx * 2 + half
            for group in tla.range(8):
                for lane in tla.range_constexpr(2):
                    row = group * 2 + lane
                    logical_row = col_block * 16 + row
                    q = load_bf16_f32(at(q_half, logical_row * 64, 64))
                    k = load_bf16_f32(at(k_half, logical_row * 64, 64))
                    gate = at(prefix, logical_row * 64, 64).load()
                    decay = tla.exp(gate - reference)
                    store_shared_score_operand(
                        k * decay, packed_destination, 2 * col_block, row
                    )
                    store_shared_score_operand(
                        (q * decay) * scale, packed_destination, 2 * col_block + 1, row
                    )
                    for row_group in tla.range_constexpr(2):
                        if tla.const_expr(col_block < 2 * (row_group + 1)):
                            right_ref = front
                            if tla.const_expr(row_group == 1):
                                right_ref = back
                            store_shared_score_operand(
                                k * tla.exp(right_ref - gate),
                                packed_destination,
                                8 + row_group * 2 + col_block,
                                row,
                            )


@tla.jit
def nd2nz_tile(source, destination, rows, columns):
    """Source staged.py:167, native 16-bit ND -> padded NZ vector stores."""
    src = tla.make_tensor(
        source.ptr,
        tla.make_layout(
            tla.make_shape(rows, columns),
            tla.make_stride(columns, 1),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    dst = padded_nz(destination.ptr, rows, columns)
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        for row in tla.range(rows):
            for col in tla.range_constexpr((columns + 127) // 128):
                mask, remaining = tla.update_mask(
                    columns - col * 128, dtype=source.ptr.dtype
                )
                value = tla.tile_view(
                    src, tla.make_shape(1, 128), tla.make_coord(row, col)
                ).load()
                tla.tile_view(
                    dst, tla.make_shape(1, 128), tla.make_coord(row, col)
                ).store(
                    value, tla.params.BlockStoreParams(block_stride=rows + 1), mask=mask
                )


@tla.jit
def apply_beta_and_finish_t(kkt, beta, negative_t, aiv):
    """Source 1287–1306: FP32 beta product, strict causal mask, FP16 handoff."""
    with tla.vec.func(mode="simd"):
        full, unused = tla.update_mask(128, dtype=tla.Float16)
        for group in tla.range(8):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                value = at(kkt, row * 64, 64).load() * broadcast(
                    at(beta, aiv * 32 + row, 1)
                )
                mask, remaining = tla.update_mask(aiv * 32 + row, dtype=tla.Float32)
                value = tla.where(mask, value, tla.full(0.0, dtype=tla.Float32)) * -1.0
                half = value.to(
                    tla.Float16,
                    tla.params.CastParams(
                        reg_slot=tla.params.RegSlot.ZERO,
                        sat_mode=tla.params.SatMode.NOSAT,
                        round_mode=tla.params.RoundMode.CAST_RINT,
                    ),
                )
                at(negative_t, row * 64, 64).store(
                    half,
                    tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_PACK_B32
                    ),
                    mask=full,
                )


@tla.jit
def mask_mqk(mqk, aqk, aiv):
    with tla.vec.func(mode="simd"):
        for group in tla.range(8):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                value = at(mqk, row * 64, 64).load()
                mask, remaining = tla.update_mask(aiv * 32 + row + 1, dtype=tla.Float32)
                store_bf16(
                    at(aqk, row * 64, 64),
                    tla.where(mask, value, tla.full(0.0, dtype=tla.Float32)),
                )


@tla.jit
def persistent_negative_t(kkt, beta, negative, packed, aiv):
    """Source persistent apply_beta_and_mask_kkt: one FP16 rounding point."""
    destination = padded_nz(packed.ptr, 32, 64)
    with tla.vec.func(mode="simd"):
        full, unused = tla.update_mask(64, dtype=tla.Float32)
        low, unused = tla.update_mask(64, dtype=tla.Float16)
        for group in tla.range(8):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                value = at(kkt, row * 64, 64).load() * broadcast(
                    at(beta, aiv * 32 + row, 1)
                )
                causal, remaining = tla.update_mask(aiv * 32 + row, dtype=tla.Float32)
                value = tla.where(causal, value, tla.full(0.0, dtype=tla.Float32)) * (
                    -1.0
                )
                half = value.to(
                    tla.Float16,
                    tla.params.CastParams(
                        reg_slot=tla.params.RegSlot.ZERO,
                        sat_mode=tla.params.SatMode.NOSAT,
                        round_mode=tla.params.RoundMode.CAST_RINT,
                    ),
                )
                restored = half.to(
                    tla.Float32,
                    tla.params.CastParams(),
                )
                at(negative, row * 64, 64).store(restored, mask=full)
                compact, ignored = tla.deinterleave(half, half)
                tla.tile_view(
                    destination, tla.make_shape(1, 128), tla.make_coord(row, 0)
                ).store(compact, tla.params.BlockStoreParams(block_stride=33), mask=low)


@tla.jit
def mask_mqk_nz(mqk, public, packed, aiv, save_matrices=True):
    """Source persistent causal QK: shared BF16 value for ND and NZ outputs."""
    destination = padded_nz(packed.ptr, 32, 64)
    with tla.vec.func(mode="simd"):
        low, unused = tla.update_mask(64, dtype=tla.BFloat16)
        for group in tla.range(8):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                mask, remaining = tla.update_mask(aiv * 32 + row + 1, dtype=tla.Float32)
                value = tla.where(
                    mask, at(mqk, row * 64, 64).load(), tla.full(0.0, dtype=tla.Float32)
                )
                if tla.const_expr(save_matrices):
                    store_bf16(at(public, row * 64, 64), value)
                tla.tile_view(
                    destination, tla.make_shape(1, 128), tla.make_coord(row, 0)
                ).store(
                    cast_and_pack_bf16(value),
                    tla.params.BlockStoreParams(block_stride=33),
                    mask=low,
                )


@tla.jit
def prepare_decayed_q(q, prefix, qg, qs, scale):
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        for group in tla.range(16):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                value = load_bf16_f32(at(q, row * 64, 64)) * tla.exp(
                    at(prefix, row * 64, 64).load()
                )
                store_bf16(at(qg, row * 64, 64), value)
                store_bf16(at(qs, row * 64, 64), value * scale)


@tla.jit
def beta_weighted_operand(raw, beta, prefix, result, decayed, negate=False):
    """Source 1652: the K decay precedes beta; V has no decay."""
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        for group in tla.range(16):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                value = load_bf16_f32(at(raw, row * 64, 64))
                if tla.const_expr(decayed):
                    value = value * tla.exp(at(prefix, row * 64, 64).load())
                weighted = value * broadcast(at(beta, row, 1))
                if tla.const_expr(negate):
                    weighted = weighted * (-1.0)
                store_bf16(at(result, row * 64, 64), weighted)


@tla.jit
def beta_weighted_nz(raw, beta, prefix, destination, decayed):
    """Source BF16 rounding followed immediately by direct padded NZ store."""
    packed = padded_nz(destination.ptr, 64, 64)
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        mask, remaining = tla.update_mask(64, dtype=tla.BFloat16)
        for group in tla.range(16):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                value = load_bf16_f32(at(raw, row * 64, 64))
                if tla.const_expr(decayed):
                    value = value * tla.exp(at(prefix, row * 64, 64).load())
                weighted = value * broadcast(at(beta, row, 1))
                tla.tile_view(
                    packed, tla.make_shape(1, 128), tla.make_coord(row, 0)
                ).store(
                    cast_and_pack_bf16(weighted),
                    tla.params.BlockStoreParams(block_stride=65),
                    mask=mask,
                )


@tla.jit
def finish_t_nz(kkt, beta, destination, aiv):
    """Strict lower -T, FP16 RNE, direct NZ; upper entries remain zero."""
    packed = padded_nz(destination.ptr, 32, 64)
    with tla.vec.func(mode="simd"):
        half_mask, unused = tla.update_mask(64, dtype=tla.Float16)
        for group in tla.range(8):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                value = at(kkt, row * 64, 64).load() * broadcast(
                    at(beta, aiv * 32 + row, 1)
                )
                mask, remaining = tla.update_mask(aiv * 32 + row, dtype=tla.Float32)
                value = tla.where(mask, value, tla.full(0.0, dtype=tla.Float32)) * (
                    -1.0
                )
                half = value.to(
                    tla.Float16,
                    tla.params.CastParams(
                        reg_slot=tla.params.RegSlot.ZERO,
                        sat_mode=tla.params.SatMode.NOSAT,
                        round_mode=tla.params.RoundMode.CAST_RINT,
                    ),
                )
                compact, ignored = tla.deinterleave(half, half)
                tla.tile_view(
                    packed, tla.make_shape(1, 128), tla.make_coord(row, 0)
                ).store(
                    compact,
                    tla.params.BlockStoreParams(block_stride=33),
                    mask=half_mask,
                )


@tla.jit
def restored_state_inputs(k, prefix, kg, gamma):
    with tla.vec.func(mode="simd"):
        end = at(prefix, 63 * 64, 64).load()
        for group in tla.range(16):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                value = load_bf16_f32(at(k, row * 64, 64)) * tla.exp(
                    end - at(prefix, row * 64, 64).load()
                )
                store_bf16(at(kg, row * 64, 64), value)
        full, unused = tla.update_mask(64, dtype=tla.Float32)
        gamma.store(tla.exp(end), mask=full)


# -----------------------------------------------------------------------------
# Common Cube primitives.
# -----------------------------------------------------------------------------


@tla.jit
def row_major(ptr, rows, columns):
    return tla.make_tensor(
        ptr,
        tla.make_layout(
            tla.make_shape(rows, columns),
            tla.make_stride(columns, 1),
            layoutTag=tla.arch.RowMajor,
        ),
    )


@tla.jit
def shared_score_l1(ptr):
    return tla.make_tensor_like(ptr, row_major(ptr, 224, 128), tla.arch.zN)


@tla.jit
def shared_right_transpose(l1_ptr, index):
    # Transpose of a 16x128 panel in the *whole* 224x128 zN cache.
    # Keep the parent's 3584-element column-fractal stride.
    return tla.make_tensor(
        l1_ptr + (8 + index) * 256,
        tla.make_layout(
            tla.make_shape((16, 8), (16, 1)),
            tla.make_stride((1, 3584), (16, 256)),
            origin_shape=tla.make_shape(128, 16),
            layoutTag=tla.arch.nZ,
        ),
    )


@tla.jit
def issue_cached_score_mmads(
    cache, left_ptr, right_ptr, ck_ptr, cq_ptr, loaded, left_free
):
    """Source 856–875: cache six RHS in L0B; each of eight LHS loads once."""
    _mutex_left_free = tla.mutex("issue_cached_score_mmads_731_left_free", id=0)
    _mutex_loaded = tla.mutex("issue_cached_score_mmads_731_loaded", id=1)
    for index in tla.range_constexpr(6):
        source = shared_right_transpose(cache.ptr, index)
        right = tla.make_tensor_like(
            right_ptr + index * 2048, row_major(right_ptr, 128, 16), tla.arch.nZ
        )
        tla.copy(right, source)
    for kind in tla.range_constexpr(2):
        for row in tla.range_constexpr(4):
            source = tla.tile_view(
                cache, tla.make_shape(16, 128), tla.make_coord(2 * row + kind, 0)
            )
            left = tla.make_tensor_like(
                left_ptr + kind * 2048, row_major(left_ptr, 16, 128), tla.arch.zN
            )
            if tla.const_expr(row != 0):
                mutex_event(_mutex_left_free, tla.pipes.MTE1)
            tla.copy(left, source)
            mutex_event(_mutex_loaded, tla.pipes.MTE1)
            mutex_event(_mutex_loaded, tla.pipes.CUBE)
            for col in tla.range_constexpr(row + 1):
                index = col
                if tla.const_expr(row >= 2):
                    index = 2 + col
                right = tla.make_tensor_like(
                    right_ptr + index * 2048, row_major(right_ptr, 128, 16), tla.arch.nZ
                )
                result_ptr = ck_ptr
                if tla.const_expr(kind == 1):
                    result_ptr = cq_ptr
                # Source constructs a compact NZ result alias for each 16x16
                # block; use the same full64 fractal offset explicitly.
                result = tla.make_tensor_like(
                    result_ptr + row * 256 + col * 1024,
                    row_major(result_ptr, 16, 16),
                    tla.arch.L0Clayout,
                )
                tla.mmad(result, left, right, init_c=True)
            if tla.const_expr(row != 3):
                mutex_event(_mutex_left_free, tla.pipes.CUBE)


@tla.jit
def matrix64(ptr, layout):
    return tla.make_tensor_like(ptr, row_major(ptr, 64, 64), layout)


@tla.jit
def load_inverse_operands(left, right, a, b, zero, ready, free, diagonal_only, clear):
    """Reference full64 double-buffer load, optionally replace diagonal fractals."""
    _mutex_free = tla.mutex("load_inverse_operands_782_free", id=0)
    _mutex_ready = tla.mutex("load_inverse_operands_782_ready", id=1)
    mutex_event(_mutex_free, tla.pipes.MTE1)
    if tla.const_expr(diagonal_only):
        if tla.const_expr(clear):
            tla.copy(a, zero)
            tla.copy(b, zero)
        for i in tla.range_constexpr(4):
            tla.copy(
                tla.tile_view(a, tla.make_shape(16, 16), tla.make_coord(i, i)),
                tla.tile_view(left, tla.make_shape(16, 16), tla.make_coord(i, i)),
            )
        for i in tla.range_constexpr(4):
            tla.copy(
                tla.tile_view(b, tla.make_shape(16, 16), tla.make_coord(i, i)),
                tla.tile_view(right, tla.make_shape(16, 16), tla.make_coord(i, i)),
            )
    else:
        tla.copy(a, left)
        tla.copy(b, right)
    mutex_event(_mutex_ready, tla.pipes.MTE1)


@tla.jit
def inverse_mmad(c, a, b, ready, free, initialize, unit):
    _mutex_ready = tla.mutex("inverse_mmad_806_ready", id=0)
    _mutex_free = tla.mutex("inverse_mmad_806_free", id=1)
    mutex_event(_mutex_ready, tla.pipes.CUBE)
    tla.mmad(c, a, b, init_c=initialize, unit_flag=unit)
    mutex_event(_mutex_free, tla.pipes.CUBE)


@tla.jit
def inverse_fix(destination, c, fix_ready, accumulator_free):
    _mutex_fix_ready = tla.mutex("inverse_fix_813_fix_ready", id=0)
    _mutex_accumulator_free = tla.mutex("inverse_fix_813_accumulator_free", id=1)
    tla.copy(destination, c, tla.params.CopyL0C2DstParams(unit_flag=3))
    mutex_event(_mutex_fix_ready, tla.pipes.FIX)
    mutex_event(_mutex_fix_ready, tla.pipes.MTE1)
    mutex_event(_mutex_accumulator_free, tla.pipes.FIX)
    mutex_event(_mutex_accumulator_free, tla.pipes.CUBE)


@tla.jit
def neumann_packed64_leaves(
    negative_t,
    identity,
    zero,
    scratch,
    power2,
    power4,
    inverse,
    a0,
    a1,
    b0,
    b1,
    c_inverse,
    c_power,
    ready0,
    ready1,
    free0,
    free1,
    fix_ready,
    accumulator_free,
):
    """Source 547–592: preserve FP32 c_inverse between FP16 feedback updates."""
    # I + (-T): preload both banks, then accumulate without rounding in between.
    load_inverse_operands(
        negative_t, identity, a0, b0, zero, ready0, free0, False, False
    )
    load_inverse_operands(identity, identity, a1, b1, zero, ready1, free1, False, False)
    inverse_mmad(c_inverse, a0, b0, ready0, free0, True, 2)
    inverse_mmad(c_inverse, a1, b1, ready1, free1, False, 3)
    inverse_fix(scratch, c_inverse, fix_ready, accumulator_free)
    # Four independent 16x16 powers embedded in one 64x64 MMAD.
    load_inverse_operands(
        negative_t, negative_t, a0, b0, zero, ready0, free0, True, True
    )
    inverse_mmad(c_power, a0, b0, ready0, free0, True, 3)
    inverse_fix(power2, c_power, fix_ready, accumulator_free)
    load_inverse_operands(scratch, identity, a1, b1, zero, ready1, free1, True, False)
    load_inverse_operands(scratch, power2, a0, b0, zero, ready0, free0, True, False)
    inverse_mmad(c_inverse, a1, b1, ready1, free1, True, 2)
    inverse_mmad(c_inverse, a0, b0, ready0, free0, False, 3)
    inverse_fix(inverse, c_inverse, fix_ready, accumulator_free)
    load_inverse_operands(power2, power2, a1, b1, zero, ready1, free1, True, False)
    inverse_mmad(c_power, a1, b1, ready1, free1, True, 3)
    inverse_fix(power4, c_power, fix_ready, accumulator_free)
    load_inverse_operands(inverse, power4, a0, b0, zero, ready0, free0, True, False)
    inverse_mmad(c_inverse, a0, b0, ready0, free0, False, 3)
    inverse_fix(inverse, c_inverse, fix_ready, accumulator_free)
    load_inverse_operands(power4, power4, a1, b1, zero, ready1, free1, True, False)
    inverse_mmad(c_power, a1, b1, ready1, free1, True, 3)
    inverse_fix(power2, c_power, fix_ready, accumulator_free)
    load_inverse_operands(inverse, power2, a0, b0, zero, ready0, free0, True, False)
    inverse_mmad(c_inverse, a0, b0, ready0, free0, False, 3)
    inverse_fix(inverse, c_inverse, fix_ready, accumulator_free)


@tla.jit
def copy_selected_fractals(destination, source, mode):
    """Logical block selection; TLA performs the L0A/L0B physical mapping."""
    if tla.const_expr(mode == 0):
        tla.copy(destination, source)
    else:
        for row in tla.range_constexpr(4):
            for col in tla.range_constexpr(4):
                if tla.const_expr(
                    (mode == 1 and row == col and row % 2 == 0)
                    or (mode == 2 and row == col and row % 2 == 1)
                    or (mode == 3 and row % 2 == 1 and col == row - 1)
                    or (mode == 4 and row < 2 and col < 2)
                    or (mode == 5 and row >= 2 and col < 2)
                ):
                    tla.copy(
                        tla.tile_view(
                            destination,
                            tla.make_shape(16, 16),
                            tla.make_coord(row, col),
                        ),
                        tla.tile_view(
                            source, tla.make_shape(16, 16), tla.make_coord(row, col)
                        ),
                    )


@tla.jit
def load_selected_pair(left, right, a, b, zero, ready, free, left_mode, right_mode):
    _mutex_free = tla.mutex("load_selected_pair_905_free", id=0)
    _mutex_ready = tla.mutex("load_selected_pair_905_ready", id=1)
    mutex_event(_mutex_free, tla.pipes.MTE1)
    tla.copy(a, zero)
    tla.copy(b, zero)
    copy_selected_fractals(a, left, left_mode)
    copy_selected_fractals(b, right, right_mode)
    mutex_event(_mutex_ready, tla.pipes.MTE1)


@tla.jit
def compose_packed64_inverse(
    scratch,
    inverse,
    temporary,
    zero,
    a0,
    a1,
    b0,
    b1,
    c_inverse,
    c_power,
    ready0,
    ready1,
    free0,
    free1,
    fix_ready,
    accumulator_free,
):
    """Source 594–669: two packed16 merges then one packed32 merge."""
    _mutex_free0 = tla.mutex("compose_packed64_inverse_915_free0", id=0)
    _mutex_ready0 = tla.mutex("compose_packed64_inverse_915_ready0", id=1)
    load_selected_pair(scratch, inverse, a1, b1, zero, ready1, free1, 3, 1)
    inverse_mmad(c_power, a1, b1, ready1, free1, True, 3)
    inverse_fix(temporary, c_power, fix_ready, accumulator_free)
    load_selected_pair(inverse, temporary, a0, b0, zero, ready0, free0, 2, 3)
    inverse_mmad(c_inverse, a0, b0, ready0, free0, False, 3)
    inverse_fix(inverse, c_inverse, fix_ready, accumulator_free)
    load_selected_pair(scratch, inverse, a1, b1, zero, ready1, free1, 0, 4)
    # Preserve source overlap: prepare the second pair's independent LHS before
    # the first MMAD and FIX produce its RHS.
    mutex_event(_mutex_free0, tla.pipes.MTE1)
    tla.copy(a0, zero)
    tla.copy(b0, zero)
    tla.copy(a0, inverse)
    inverse_mmad(c_power, a1, b1, ready1, free1, True, 3)
    inverse_fix(temporary, c_power, fix_ready, accumulator_free)
    copy_selected_fractals(b0, temporary, 5)
    mutex_event(_mutex_ready0, tla.pipes.MTE1)
    inverse_mmad(c_inverse, a0, b0, ready0, free0, False, 3)


# -----------------------------------------------------------------------------
# FP32 state and matrix snapshots.
# -----------------------------------------------------------------------------


@tla.jit
def cast_state_snapshot(state, snapshot, dv):
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        for row in tla.range(dv):
            store_bf16(at(snapshot, row * 64, 64), at(state, row * 64, 64).load())


@tla.jit
def finish_u_row(
    u, delta, output, destination, row, dv, width, mask, save_vnew, valid_row
):
    if tla.const_expr(dv == 128):
        value0 = tla.full(0.0, dtype=tla.Float32)
        value1 = tla.full(0.0, dtype=tla.Float32)
        if tla.const_expr(valid_row):
            value0 = (
                load_bf16_f32(
                    tla.tile_view(u, tla.make_shape(1, 64), tla.make_coord(row, 0))
                )
                - tla.tile_view(
                    delta, tla.make_shape(1, 64), tla.make_coord(row, 0)
                ).load()
            )
            value1 = (
                load_bf16_f32(
                    tla.tile_view(u, tla.make_shape(1, 64), tla.make_coord(row, 1))
                )
                - tla.tile_view(
                    delta, tla.make_shape(1, 64), tla.make_coord(row, 1)
                ).load()
            )
        compact, ignored = tla.deinterleave(cast_bf16(value0), cast_bf16(value1))
        if tla.const_expr(save_vnew):
            tla.tile_view(output, tla.make_shape(1, 128), tla.make_coord(row, 0)).store(
                compact, mask=mask
            )
        tla.tile_view(
            destination, tla.make_shape(1, 128), tla.make_coord(row, 0)
        ).store(compact, tla.params.BlockStoreParams(block_stride=33), mask=mask)
    else:
        value = tla.full(0.0, dtype=tla.Float32)
        if tla.const_expr(valid_row):
            value = (
                load_bf16_f32(
                    tla.tile_view(u, tla.make_shape(1, 64), tla.make_coord(row, 0))
                )
                - tla.tile_view(
                    delta, tla.make_shape(1, 64), tla.make_coord(row, 0)
                ).load()
            )
        if tla.const_expr(save_vnew):
            store_bf16(
                tla.tile_view(output, tla.make_shape(1, 64), tla.make_coord(row, 0)),
                value,
                width,
            )
        tla.tile_view(
            destination, tla.make_shape(1, 128), tla.make_coord(row, 0)
        ).store(
            cast_and_pack_bf16(value),
            tla.params.BlockStoreParams(block_stride=33),
            mask=mask,
        )


@tla.jit
def finish_u_delta_nz(u, delta, output, packed, dv, aiv, valid, save_vnew=True):
    """Source valid-row loops, preserving fused ND/NZ and exact BF16 boundary."""
    width = 64
    if tla.const_expr(dv < 64):
        width = dv
    destination = padded_nz(packed.ptr, 32, dv)
    rows = valid - aiv * 32
    if rows > 32:
        rows = 32
    if rows < 0:
        rows = 0
    with tla.vec.func(mode="simd"):
        mask, remaining = tla.update_mask(dv, dtype=tla.BFloat16)
        groups = rows // 2
        for group in tla.range(groups):
            for lane in tla.range_constexpr(2):
                group_row = group * 2 + lane
                finish_u_row(
                    u,
                    delta,
                    output,
                    destination,
                    group_row,
                    dv,
                    width,
                    mask,
                    save_vnew,
                    True,
                )
        for remainder_row in tla.range(groups * 2, rows):
            finish_u_row(
                u,
                delta,
                output,
                destination,
                remainder_row,
                dv,
                width,
                mask,
                save_vnew,
                True,
            )
        for tail_row in tla.range(rows, 32):
            finish_u_row(
                u,
                delta,
                output,
                destination,
                tail_row,
                dv,
                width,
                mask,
                save_vnew,
                False,
            )


@tla.jit
def scale_and_add_delta(state, delta, gamma, dv):
    with tla.vec.func(mode="simd"):
        decay = gamma.load()
        full, unused = tla.update_mask(64, dtype=tla.Float32)
        for group in tla.range(dv // 4):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                scaled = at(state, row * 64, 64).load() * decay
                updated = scaled + at(delta, row * 64, 64).load()
                at(state, row * 64, 64).store(updated, mask=full)


@tla.jit
def cast_state_nz(state, packed, dv):
    """One BF16 RNE conversion, direct padded NZ store; FP32 state is intact."""
    destination = padded_nz(packed.ptr, dv, 64)
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        mask, remaining = tla.update_mask(64, dtype=tla.BFloat16)
        for group in tla.range(dv // 4):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                value = cast_and_pack_bf16(at(state, row * 64, 64).load())
                tla.tile_view(
                    destination, tla.make_shape(1, 128), tla.make_coord(row, 0)
                ).store(
                    value, tla.params.BlockStoreParams(block_stride=dv + 1), mask=mask
                )


@tla.jit
def update_state_nz(state, delta, gamma, packed, dv):
    """Same separate FP32 multiply/add; also publish its BF16 NZ snapshot."""
    destination = padded_nz(packed.ptr, dv, 64)
    with tla.vec.func(mode="simd"):
        decay = gamma.load()
        full, unused = tla.update_mask(64, dtype=tla.Float32)
        mask, remaining = tla.update_mask(64, dtype=tla.BFloat16)
        for group in tla.range(dv // 4):
            for lane in tla.range_constexpr(4):
                row = group * 4 + lane
                scaled = at(state, row * 64, 64).load() * decay
                updated = scaled + at(delta, row * 64, 64).load()
                at(state, row * 64, 64).store(updated, mask=full)
                tla.tile_view(
                    destination, tla.make_shape(1, 128), tla.make_coord(row, 0)
                ).store(
                    cast_and_pack_bf16(updated),
                    tla.params.BlockStoreParams(block_stride=dv + 1),
                    mask=mask,
                )


@tla.jit
def compute_state_projections(
    q,
    w,
    state,
    a0_ptr,
    a1_ptr,
    b_ptr,
    cu_ptr,
    co_ptr,
    dv,
    loaded,
    projection_done=None,
    output_free=None,
):
    _mutex_loaded = tla.mutex("compute_state_projections_1138_loaded", id=0)
    _mutex_projection_done = tla.mutex(
        "compute_state_projections_1138_projection_done", id=1
    )
    _mutex_output_free = tla.mutex("compute_state_projections_1138_output_free", id=2)
    a0 = tla.make_tensor_like(a0_ptr, row_major(a0_ptr, 64, 128), tla.arch.zN)
    a1 = tla.make_tensor_like(a1_ptr, row_major(a1_ptr, 64, 128), tla.arch.zN)
    # State is stored [V,K]. Its zN bytes are a transpose nZ view [K,V].
    state_transposed = tla.make_tensor_like(
        state.ptr, row_major(state.ptr, 128, dv), tla.arch.nZ
    )
    b = tla.make_tensor_like(b_ptr, row_major(b_ptr, 128, dv), tla.arch.nZ)
    cu = tla.make_tensor_like(cu_ptr, row_major(cu_ptr, 64, dv), tla.arch.L0Clayout)
    co = tla.make_tensor_like(co_ptr, row_major(co_ptr, 64, dv), tla.arch.L0Clayout)
    tla.copy(b, state_transposed)
    tla.copy(a0, w)
    mutex_event(_mutex_loaded, tla.pipes.MTE1)
    mutex_event(_mutex_loaded, tla.pipes.CUBE)
    tla.mmad(cu, a0, b, init_c=True)
    if tla.const_expr(projection_done is not None):
        mutex_event(_mutex_projection_done, tla.pipes.CUBE)
    tla.copy(a1, q)
    mutex_event(_mutex_loaded, tla.pipes.MTE1)
    mutex_event(_mutex_loaded, tla.pipes.CUBE)
    if tla.const_expr(output_free is not None):
        mutex_event(_mutex_output_free, tla.pipes.CUBE)
    tla.mmad(co, a1, b, init_c=True)


@tla.jit
def load_recurrent_operands(kg, aqk, ka_ptr, aq_ptr, dv, loaded):
    _mutex_loaded = tla.mutex("load_recurrent_operands_1177_loaded", id=0)
    kb = tla.make_tensor_like(ka_ptr, row_major(ka_ptr, 64, 128), tla.arch.nZ)
    qa = tla.make_tensor_like(aq_ptr, row_major(aq_ptr, 64, 64), tla.arch.zN)
    tla.copy(kb, kg)
    tla.copy(qa, aqk)
    mutex_event(_mutex_loaded, tla.pipes.MTE1)


@tla.jit
def compute_state_delta(vnew, ka_ptr, ua_ptr, ch_ptr, dv, loaded):
    """Source update_state: publish this result before the output MMAD."""
    _mutex_loaded = tla.mutex("compute_state_delta_1186_loaded", id=0)
    vt = tla.make_tensor_like(vnew.ptr, row_major(vnew.ptr, dv, 64), tla.arch.nZ)
    ua = tla.make_tensor_like(ua_ptr, row_major(ua_ptr, dv, 64), tla.arch.zN)
    kb = tla.make_tensor_like(ka_ptr, row_major(ka_ptr, 64, 128), tla.arch.nZ)
    ch = tla.make_tensor_like(ch_ptr, row_major(ch_ptr, dv, 128), tla.arch.L0Clayout)
    tla.copy(ua, vt)
    mutex_event(_mutex_loaded, tla.pipes.MTE1)
    mutex_event(_mutex_loaded, tla.pipes.CUBE)
    tla.mmad(ch, ua, kb, init_c=True)


@tla.jit
def accumulate_output(vnew, aq_ptr, ub_ptr, co_ptr, dv, loaded):
    """Source accumulate_mqk_u; independent of the state-delta FIX write."""
    _mutex_loaded = tla.mutex("accumulate_output_1199_loaded", id=0)
    ub = tla.make_tensor_like(ub_ptr, row_major(ub_ptr, 64, dv), tla.arch.nZ)
    qa = tla.make_tensor_like(aq_ptr, row_major(aq_ptr, 64, 64), tla.arch.zN)
    co = tla.make_tensor_like(co_ptr, row_major(co_ptr, 64, dv), tla.arch.L0Clayout)
    tla.copy(ub, vnew)
    mutex_event(_mutex_loaded, tla.pipes.MTE1)
    mutex_event(_mutex_loaded, tla.pipes.CUBE)
    tla.mmad(co, qa, ub, init_c=False)


# -----------------------------------------------------------------------------
# Input preprocessing and prefix.
# -----------------------------------------------------------------------------


@tla.jit
def load_raw_prefix_inputs(
    q,
    k,
    v,
    g,
    beta,
    alpha,
    bias,
    input_q,
    input_k,
    uv,
    raw_gate,
    beta_io,
    bias_ub,
    alpha_ub,
    valid,
    q_stride,
    v_stride,
    beta_stride,
    beta_pitch,
    first_chunk,
    load_head=True,
):
    aiv = tla.arch.sub_block_idx()
    if first_chunk:
        tla.copy(
            row_major(input_q.ptr, valid, 64),
            strided_rows(q.ptr + aiv * 64, valid, 64, q_stride),
        )
        tla.copy(
            row_major(input_k.ptr, valid, 64),
            strided_rows(k.ptr + aiv * 64, valid, 64, q_stride),
        )
    tla.copy(
        row_major(uv.ptr, valid, 64),
        strided_rows(v.ptr + aiv * 64, valid, 64, v_stride),
    )
    tla.copy(
        row_major(raw_gate.ptr, valid, 64),
        strided_rows(g.ptr + aiv * 64, valid, 64, v_stride),
    )
    if tla.const_expr(beta_stride == 1):
        tla.copy(row_major(beta_io.ptr, 1, valid), row_major(beta.ptr, 1, valid))
    else:
        tla.copy(
            strided_rows(beta_io.ptr, valid, 1, beta_pitch),
            strided_rows(beta.ptr, valid, 1, beta_stride),
        )
    if tla.const_expr(load_head):
        tla.copy(row_major(bias_ub.ptr, 1, 64), row_major(bias.ptr + aiv * 64, 1, 64))
        tla.copy(row_major(alpha_ub.ptr, 1, 1), row_major(alpha.ptr, 1, 1))


@tla.jit
def prefix_inputs(
    q,
    k,
    v,
    g,
    beta,
    alpha,
    bias,
    exchange,
    qn,
    kn,
    rstd,
    gcache,
    bact,
    uq,
    uk,
    uv,
    ug,
    raw_gate,
    gate_bf16,
    raw_beta,
    beta_io,
    beta_column,
    bias_ub,
    alpha_ub,
    stats,
    ub,
    valid,
    mode,
    lower,
    allow_negative,
    normalize,
    q_stride,
    v_stride,
    beta_stride,
    beta_pitch,
    norm_stride,
    beta_out_stride,
    norm_owner,
    rstd_k_offset,
    rstd_stride,
    gm_loaded,
    v_done,
    cast_free,
    peer_loaded,
    published,
    consumed,
    beta_sigmoid=True,
    save_cache=True,
    save_norm=True,
    save_beta=True,
    qraw=None,
    kraw=None,
    first_chunk=True,
    defer_norm=False,
    inputs_prefetched=False,
):
    _mutex_gm_loaded = tla.mutex("prefix_inputs_1270_gm_loaded", id=0)
    _mutex_v_done = tla.mutex("prefix_inputs_1270_v_done", id=1)
    _mutex_peer_loaded = tla.mutex("prefix_inputs_1270_peer_loaded", id=2)
    _mutex_cast_free = tla.mutex("prefix_inputs_1270_cast_free", id=3)
    input_q = uq
    input_k = uk
    if tla.const_expr(qraw is not None):
        input_q = qraw
    if tla.const_expr(kraw is not None):
        input_k = kraw
    with tla.vector():
        aiv = tla.arch.sub_block_idx()
        if tla.const_expr(not inputs_prefetched):
            load_raw_prefix_inputs(
                q,
                k,
                v,
                g,
                beta,
                alpha,
                bias,
                input_q,
                input_k,
                uv,
                raw_gate,
                beta_io,
                bias_ub,
                alpha_ub,
                valid,
                q_stride,
                v_stride,
                beta_stride,
                beta_pitch,
                first_chunk,
            )
        mutex_event(_mutex_gm_loaded, tla.pipes.MTE2)
        mutex_event(_mutex_gm_loaded, tla.pipes.VECTOR)
        if valid < 64:
            zero_qkv_tail(input_q, input_k, uv, valid)
        if tla.const_expr(beta.ptr.dtype != tla.Float32):
            prepare_beta_input(beta_io, raw_beta, valid, beta_pitch)
        if tla.const_expr(normalize):
            compute_qk_partial_sums(input_q, input_k, at(stats, 0, 128))
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            tla.copy(
                row_major(exchange.ptr + aiv * 128, 1, 128),
                row_major(stats.ptr, 1, 128),
            )
            tla.cross_core_set_flag(published, tla.pipes.MTE3)
            tla.cross_core_wait_flag(published, tla.pipes.MTE2)
            tla.copy(
                row_major(stats.ptr + 128, 1, 128),
                row_major(exchange.ptr + (1 - aiv) * 128, 1, 128),
            )
            mutex_event(_mutex_peer_loaded, tla.pipes.MTE2)
        if valid == 64:
            activate_gate(raw_gate, bias_ub, alpha_ub, ug, 64, mode, lower, True)
        else:
            activate_gate(raw_gate, bias_ub, alpha_ub, ug, valid, mode, lower, True)
        if tla.const_expr(normalize):
            mutex_event(_mutex_peer_loaded, tla.pipes.VECTOR)
            tla.cross_core_set_flag(consumed, tla.pipes.MTE2)
            tla.cross_core_wait_flag(consumed, tla.pipes.MTE3)
            compute_qk_inverse_vectors(
                at(stats, 0, 128), at(stats, 128, 128), at(stats, 256, 128)
            )
            if tla.const_expr(not defer_norm):
                normalize_local_qk(input_q, input_k, at(stats, 256, 128), uq, uk)
        elif tla.const_expr(qraw is not None and not defer_norm):
            copy_bf16_qk(input_q, input_k, uq, uk)
        if tla.const_expr(beta.ptr.dtype == tla.Float32):
            activate_beta_f32_input(
                beta_io, ub, valid, beta_pitch, beta_sigmoid, allow_negative
            )
        else:
            activate_beta(raw_beta, ub, valid, beta_sigmoid, allow_negative)
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        if tla.const_expr(save_norm and not defer_norm):
            if norm_owner:
                tla.copy(
                    strided_rows(qn.ptr + aiv * 64, valid, 64, norm_stride),
                    row_major(uq.ptr, valid, 64),
                )
                tla.copy(
                    strided_rows(kn.ptr + aiv * 64, valid, 64, norm_stride),
                    row_major(uk.ptr, valid, 64),
                )
        # Public cache padding is zero-initialized by the host adapter. Keep
        # repeated last-prefix rows internal, where gamma needs them.
        if tla.const_expr(save_cache):
            tla.copy(
                strided_rows(gcache.ptr + aiv * 64, valid, 64, 128),
                row_major(ug.ptr, valid, 64),
            )
        if aiv == 0:
            if tla.const_expr(save_beta):
                if tla.const_expr(beta_out_stride == 1):
                    tla.copy(row_major(bact.ptr, 1, valid), row_major(ub.ptr, 1, valid))
                else:
                    beta_to_column(ub, beta_column)
                    mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                    mutex_event(_mutex_v_done, tla.pipes.MTE3)
                    tla.copy(
                        strided_rows(bact.ptr, valid, 1, beta_out_stride),
                        strided_rows(beta_column.ptr, valid, 1, 8),
                    )
            if tla.const_expr(normalize and save_norm):
                if norm_owner:
                    if tla.const_expr(rstd_stride == 1):
                        tla.copy(
                            row_major(rstd.ptr, 1, valid),
                            row_major(stats.ptr + 256, 1, valid),
                        )
                        tla.copy(
                            row_major(rstd.ptr + rstd_k_offset, 1, valid),
                            row_major(stats.ptr + 320, 1, valid),
                        )
                    else:
                        mutex_event(_mutex_cast_free, tla.pipes.MTE3)
                        mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
                        beta_to_column(at(stats, 256, 64), beta_column)
                        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                        mutex_event(_mutex_v_done, tla.pipes.MTE3)
                        tla.copy(
                            strided_rows(rstd.ptr, valid, 1, rstd_stride),
                            strided_rows(beta_column.ptr, valid, 1, 8),
                        )
                        mutex_event(_mutex_cast_free, tla.pipes.MTE3)
                        mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
                        beta_to_column(at(stats, 320, 64), beta_column)
                        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                        mutex_event(_mutex_v_done, tla.pipes.MTE3)
                        tla.copy(
                            strided_rows(
                                rstd.ptr + rstd_k_offset, valid, 1, rstd_stride
                            ),
                            strided_rows(beta_column.ptr, valid, 1, 8),
                        )


# -----------------------------------------------------------------------------
# General and safe-diagonal scores.
# -----------------------------------------------------------------------------


@tla.jit
def build_left_k(k, g, decay, nd, row_block):
    with tla.vec.func(mode="simd"):
        ref = at(g, row_block * 1024, 64).load()
        full, unused = tla.update_mask(64, dtype=tla.Float32)
        for group in tla.range(8):
            for lane in tla.range_constexpr(2):
                row = group * 2 + lane
                d = tla.exp(at(g, (row_block * 16 + row) * 64, 64).load() - ref)
                at(decay, row * 64, 64).store(d, mask=full)
                store_bf16(
                    at(nd, row * 64, 64),
                    load_bf16_f32(at(k, (row_block * 16 + row) * 64, 64)) * d,
                )


@tla.jit
def build_left_q(q, decay, nd, row_block, scale):
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        for group in tla.range(8):
            for lane in tla.range_constexpr(2):
                row = group * 2 + lane
                value = (
                    load_bf16_f32(at(q, (row_block * 16 + row) * 64, 64))
                    * at(decay, row * 64, 64).load()
                )
                store_bf16(at(nd, row * 64, 64), value * scale)


@tla.jit
def build_right(k, g, nd, row_block, col_block):
    with tla.vec.func(mode="simd"):
        ref = at(g, row_block * 1024, 64).load()
        for group in tla.range(8):
            for lane in tla.range_constexpr(2):
                row = group * 2 + lane
                d = tla.exp(ref - at(g, (col_block * 16 + row) * 64, 64).load())
                store_bf16(
                    at(nd, row * 64, 64),
                    load_bf16_f32(at(k, (col_block * 16 + row) * 64, 64)) * d,
                )


@tla.jit
def clear_safe(left_k, left_q, right_k):
    with tla.vec.func(mode="simd"):
        zero = tla.full(0.0, dtype=tla.Float32)
        for row in tla.range(16):
            store_bf16(at(left_k, row * 64, 64), zero)
            store_bf16(at(left_q, row * 64, 64), zero)
            store_bf16(at(right_k, row * 64, 64), zero)


@tla.jit
def update_safe_left(k, q, left_k, left_q, row_block, active, scale):
    with tla.vec.func(mode="simd"):
        if active > 0:
            zero = tla.full(0.0, dtype=tla.Float32)
            store_bf16(at(left_k, (active - 1) * 64, 64), zero)
            store_bf16(at(left_q, (active - 1) * 64, 64), zero)
        store_bf16(
            at(left_k, active * 64, 64),
            load_bf16_f32(at(k, (row_block * 16 + active) * 64, 64)),
        )
        store_bf16(
            at(left_q, active * 64, 64),
            load_bf16_f32(at(q, (row_block * 16 + active) * 64, 64)) * scale,
        )


@tla.jit
def update_safe_right(k, g, right, row_block, active):
    with tla.vec.func(mode="simd"):
        ref = at(g, (row_block * 16 + active) * 64, 64).load()
        for row in tla.range(active + 1):
            decay = tla.exp(ref - at(g, (row_block * 16 + row) * 64, 64).load())
            store_bf16(
                at(right, row * 64, 64),
                load_bf16_f32(at(k, (row_block * 16 + row) * 64, 64)) * decay,
            )


@tla.jit
def publish_operand(nd, nz_ptr, l1_ptr, step, ready0, ready1, free0, free1, v_done):
    _mutex_v_done = tla.mutex("publish_operand_1554_v_done", id=0)
    slot = step % 2
    if slot == 0:
        tla.cross_core_wait_flag(free0, tla.pipes.VECTOR)
    else:
        tla.cross_core_wait_flag(free1, tla.pipes.VECTOR)
    nz = flat(nz_ptr + slot * 1088, 1088)
    nd2nz_tile(nd, nz, 16, 64)
    mutex_event(_mutex_v_done, tla.pipes.VECTOR)
    mutex_event(_mutex_v_done, tla.pipes.MTE3)
    l1 = tla.make_tensor_like(
        l1_ptr + slot * 2048, row_major(l1_ptr, 16, 128), tla.arch.zN
    )
    tla.copy(
        tla.tile_view(
            l1, tla.make_shape(16, 64), tla.make_coord(0, tla.arch.sub_block_idx())
        ),
        padded_nz(nz.ptr, 16),
    )
    if slot == 0:
        tla.cross_core_set_flag(ready0, tla.pipes.MTE3)
    else:
        tla.cross_core_set_flag(ready1, tla.pipes.MTE3)


@tla.jit
def consume_operand(dst, l1_ptr, step, ready0, ready1, free0, free1, transpose):
    slot = step % 2
    if slot == 0:
        tla.cross_core_wait_flag(ready0, tla.pipes.MTE1)
    else:
        tla.cross_core_wait_flag(ready1, tla.pipes.MTE1)
    src = tla.make_tensor_like(
        l1_ptr + slot * 2048, row_major(l1_ptr, 16, 128), tla.arch.zN
    )
    if tla.const_expr(transpose):
        src = tla.make_tensor_like(
            l1_ptr + slot * 2048, row_major(l1_ptr, 128, 16), tla.arch.nZ
        )
    tla.copy(dst, src)
    if slot == 0:
        tla.cross_core_set_flag(free0, tla.pipes.MTE1)
    else:
        tla.cross_core_set_flag(free1, tla.pipes.MTE1)


@tla.jit
def per_d_scores(
    q,
    k,
    g,
    nd,
    decay,
    safe_k,
    safe_q,
    safe_r,
    nz_ptr,
    l1_ptr,
    a_ptr,
    b_ptr,
    ck,
    cq,
    scale,
    bounded,
):
    """Keep reference block order and FP32 active-row accumulation order.

    Two L1 handoff slots; each holds both AIVs' 16x64 halves. The safe diagonal
    only evaluates non-positive gate differences; it is not the shared path.
    """
    _mutex_op_free = tla.mutex("per_d_scores_1601_op_free", id=0)
    _mutex_loaded = tla.mutex("per_d_scores_1601_loaded", id=1)
    ready0 = tla.cross_flag("pd_ready0", mode=2)
    ready1 = tla.cross_flag("pd_ready1", mode=2)
    free0 = tla.cross_flag("pd_free0", mode=2)
    free1 = tla.cross_flag("pd_free1", mode=2)
    v_done = tla.flag("pd_v_done", tla.pipes.VECTOR, tla.pipes.MTE3)
    loaded = tla.flag("pd_loaded", tla.pipes.MTE1, tla.pipes.CUBE)
    op_free = tla.flag("pd_op_free", tla.pipes.CUBE, tla.pipes.MTE1)
    with tla.cube():
        ka = tla.make_tensor_like(a_ptr, row_major(a_ptr, 16, 128), tla.arch.zN)
        qa = tla.make_tensor_like(a_ptr + 2048, row_major(a_ptr, 16, 128), tla.arch.zN)
        kb = tla.make_tensor_like(b_ptr, row_major(b_ptr, 128, 16), tla.arch.nZ)
        tla.cross_core_set_flag(free0, tla.pipes.MTE1)
        tla.cross_core_set_flag(free1, tla.pipes.MTE1)
        mutex_event(_mutex_op_free, tla.pipes.CUBE)
        step = 0
        for rb in tla.range_constexpr(4):
            if tla.const_expr(rb > 0 or bounded):
                mutex_event(_mutex_op_free, tla.pipes.MTE1)
                consume_operand(ka, l1_ptr, step, ready0, ready1, free0, free1, False)
                step = step + 1
                consume_operand(qa, l1_ptr, step, ready0, ready1, free0, free1, False)
                step = step + 1
                for cb in tla.range_constexpr(rb + int(bounded)):
                    if tla.const_expr(cb > 0):
                        mutex_event(_mutex_op_free, tla.pipes.MTE1)
                    consume_operand(
                        kb, l1_ptr, step, ready0, ready1, free0, free1, True
                    )
                    step = step + 1
                    mutex_event(_mutex_loaded, tla.pipes.MTE1)
                    mutex_event(_mutex_loaded, tla.pipes.CUBE)
                    c0 = tla.make_tensor_like(
                        ck + rb * 256 + cb * 1024,
                        row_major(ck, 16, 16),
                        tla.arch.L0Clayout,
                    )
                    c1 = tla.make_tensor_like(
                        cq + rb * 256 + cb * 1024,
                        row_major(cq, 16, 16),
                        tla.arch.L0Clayout,
                    )
                    tla.mmad(c0, ka, kb, init_c=True)
                    tla.mmad(c1, qa, kb, init_c=True)
                    mutex_event(_mutex_op_free, tla.pipes.CUBE)
            if tla.const_expr(not bounded):
                diag0 = tla.make_tensor_like(
                    ck + rb * 256 + rb * 1024, row_major(ck, 16, 16), tla.arch.L0Clayout
                )
                diag1 = tla.make_tensor_like(
                    cq + rb * 256 + rb * 1024, row_major(cq, 16, 16), tla.arch.L0Clayout
                )
                for active in tla.range(16):
                    mutex_event(_mutex_op_free, tla.pipes.MTE1)
                    consume_operand(
                        ka, l1_ptr, step, ready0, ready1, free0, free1, False
                    )
                    step = step + 1
                    consume_operand(
                        qa, l1_ptr, step, ready0, ready1, free0, free1, False
                    )
                    step = step + 1
                    consume_operand(
                        kb, l1_ptr, step, ready0, ready1, free0, free1, True
                    )
                    step = step + 1
                    mutex_event(_mutex_loaded, tla.pipes.MTE1)
                    mutex_event(_mutex_loaded, tla.pipes.CUBE)
                    tla.mmad(diag0, ka, kb, init_c=active == 0)
                    tla.mmad(diag1, qa, kb, init_c=active == 0)
                    mutex_event(_mutex_op_free, tla.pipes.CUBE)
        mutex_event(_mutex_op_free, tla.pipes.MTE1)
    with tla.vector():
        step = 0
        for rb in tla.range_constexpr(4):
            if tla.const_expr(rb > 0 or bounded):
                build_left_k(k, g, decay, nd, rb)
                publish_operand(
                    nd, nz_ptr, l1_ptr, step, ready0, ready1, free0, free1, v_done
                )
                step = step + 1
                build_left_q(q, decay, nd, rb, scale)
                publish_operand(
                    nd, nz_ptr, l1_ptr, step, ready0, ready1, free0, free1, v_done
                )
                step = step + 1
                for cb in tla.range_constexpr(rb + int(bounded)):
                    build_right(k, g, nd, rb, cb)
                    publish_operand(
                        nd, nz_ptr, l1_ptr, step, ready0, ready1, free0, free1, v_done
                    )
                    step = step + 1
            if tla.const_expr(not bounded):
                clear_safe(safe_k, safe_q, safe_r)
                for active in tla.range(16):
                    update_safe_left(k, q, safe_k, safe_q, rb, active, scale)
                    publish_operand(
                        safe_k,
                        nz_ptr,
                        l1_ptr,
                        step,
                        ready0,
                        ready1,
                        free0,
                        free1,
                        v_done,
                    )
                    step = step + 1
                    publish_operand(
                        safe_q,
                        nz_ptr,
                        l1_ptr,
                        step,
                        ready0,
                        ready1,
                        free0,
                        free1,
                        v_done,
                    )
                    step = step + 1
                    update_safe_right(k, g, safe_r, rb, active)
                    publish_operand(
                        safe_r,
                        nz_ptr,
                        l1_ptr,
                        step,
                        ready0,
                        ready1,
                        free0,
                        free1,
                        v_done,
                    )
                    step = step + 1
        tla.cross_core_wait_flag(free0, tla.pipes.VECTOR)
        tla.cross_core_wait_flag(free1, tla.pipes.VECTOR)


# -----------------------------------------------------------------------------
# Inverse projection.
# -----------------------------------------------------------------------------


@tla.jit
def publish_resident_inverse(
    acc, public_inverse, resident, save, cube_done, resident_ready
):
    # Source first FIX uses unit_flag=2 (wait without consuming). The current
    # TLA FIX API exposes only 0/3. An explicit CUBE->FIX edge closes that first
    # copy; the final unit_flag=3 copy retains the source's consuming boundary.
    _mutex_cube_done = tla.mutex("publish_resident_inverse_1766_cube_done", id=0)
    _mutex_resident_ready = tla.mutex(
        "publish_resident_inverse_1766_resident_ready", id=1
    )
    if tla.const_expr(save):
        mutex_event(_mutex_cube_done, tla.pipes.CUBE)
        mutex_event(_mutex_cube_done, tla.pipes.FIX)
        tla.copy(public_inverse, acc)
    tla.copy(resident, acc, tla.params.CopyL0C2DstParams(unit_flag=3))
    mutex_event(_mutex_resident_ready, tla.pipes.FIX)


@tla.jit
def compute_u_pre_and_w(
    resident,
    k_beta,
    v_beta,
    out_w,
    out_u,
    a_ptr,
    b_ptr,
    c_ptr,
    resident_ready,
    k_ready,
    v_ready,
    loaded,
    b_free,
    result_ready,
):
    """Keep inverse resident in L0A; preserve positive W and BF16 U_pre exits."""
    _mutex_resident_ready = tla.mutex("compute_u_pre_and_w_1781_resident_ready", id=0)
    _mutex_loaded = tla.mutex("compute_u_pre_and_w_1781_loaded", id=1)
    _mutex_b_free = tla.mutex("compute_u_pre_and_w_1781_b_free", id=2)
    _mutex_result_ready = tla.mutex("compute_u_pre_and_w_1781_result_ready", id=3)
    a = tla.make_tensor_like(a_ptr, row_major(a_ptr, 64, 64), tla.arch.zN)
    b = tla.make_tensor_like(b_ptr, row_major(b_ptr, 64, 128), tla.arch.nZ)
    cw = tla.make_tensor_like(c_ptr, row_major(c_ptr, 64, 128), tla.arch.L0Clayout)
    cu = tla.make_tensor_like(
        c_ptr + 8192, row_major(c_ptr, 64, 128), tla.arch.L0Clayout
    )
    mutex_event(_mutex_resident_ready, tla.pipes.MTE1)
    tla.copy(a, resident)
    tla.cross_core_wait_flag(k_ready, tla.pipes.MTE1)
    tla.copy(b, k_beta)
    mutex_event(_mutex_loaded, tla.pipes.MTE1)
    mutex_event(_mutex_loaded, tla.pipes.CUBE)
    tla.mmad(cw, a, b, init_c=True)
    mutex_event(_mutex_b_free, tla.pipes.CUBE)
    mutex_event(_mutex_result_ready, tla.pipes.MTE1)
    mutex_event(_mutex_result_ready, tla.pipes.CUBE)
    tla.copy(out_w, cw)
    mutex_event(_mutex_b_free, tla.pipes.MTE1)
    tla.cross_core_wait_flag(v_ready, tla.pipes.MTE1)
    tla.copy(b, v_beta)
    mutex_event(_mutex_loaded, tla.pipes.MTE1)
    mutex_event(_mutex_loaded, tla.pipes.CUBE)
    tla.mmad(cu, a, b, init_c=True)
    mutex_event(_mutex_result_ready, tla.pipes.MTE1)
    mutex_event(_mutex_result_ready, tla.pipes.CUBE)
    tla.copy(out_u, cu)


# -----------------------------------------------------------------------------
# Staged single-chunk producer.
# -----------------------------------------------------------------------------


@tla.jit
def stage_one_chunk(
    ubase,
    lbase,
    abase,
    bbase,
    cbase,
    q,
    k,
    v,
    g,
    beta,
    alpha,
    bias,
    exchange,
    identity16,
    aqk,
    akk,
    w,
    u,
    qg,
    kg,
    qs,
    gamma,
    qn,
    kn,
    rstd,
    gcache,
    bact,
    valid,
    mode,
    lower,
    allow_negative,
    normalize,
    scale,
    shared,
    phase_flags=None,
    q_stride=128,
    v_stride=128,
    beta_stride=1,
    norm_stride=128,
    beta_out_stride=1,
    output_rows=64,
    norm_owner=1,
    rstd_k_offset=64,
    rstd_stride=1,
    beta_sigmoid=True,
):
    _mutex_gm_ready = tla.mutex("stage_one_chunk_1832_gm_ready", id=0)
    _mutex_gm_loaded = tla.mutex("stage_one_chunk_1832_gm_loaded", id=1)
    _mutex_v_done = tla.mutex("stage_one_chunk_1832_v_done", id=2)
    _mutex_peer_loaded = tla.mutex("stage_one_chunk_1832_peer_loaded", id=3)
    _mutex_cast_free = tla.mutex("stage_one_chunk_1832_cast_free", id=4)
    _mutex_cube_done = tla.mutex("stage_one_chunk_1832_cube_done", id=5)
    _mutex_free0 = tla.mutex("stage_one_chunk_1832_free0", id=6)
    _mutex_free1 = tla.mutex("stage_one_chunk_1832_free1", id=7)
    uq = flat(tla.recast_ptr(ubase + 0, dtype=tla.BFloat16), 4096)
    uk = flat(tla.recast_ptr(ubase + 2048, dtype=tla.BFloat16), 4096)
    ug = flat(tla.recast_ptr(ubase + 4096, dtype=tla.Float32), 4096)
    ub = flat(tla.recast_ptr(ubase + 8192, dtype=tla.Float32), 64)
    packed = flat(tla.recast_ptr(ubase + 8256, dtype=tla.BFloat16), 14400)
    kkt = flat(tla.recast_ptr(ubase + 15488, dtype=tla.Float32), 2048)
    mqk = flat(tla.recast_ptr(ubase + 17536, dtype=tla.Float32), 2048)
    nt = flat(tla.recast_ptr(ubase + 19584, dtype=tla.Float16), 2048)
    ntnz = flat(tla.recast_ptr(ubase + 20608, dtype=tla.Float16), 2112)
    aq = flat(tla.recast_ptr(ubase + 21696, dtype=tla.BFloat16), 2048)
    score_ptr = tla.recast_ptr(lbase + 0, dtype=tla.BFloat16)
    lp = tla.recast_ptr(lbase + 28672, dtype=tla.Float16)
    left = tla.recast_ptr(abase + 0, dtype=tla.BFloat16)
    right = tla.recast_ptr(bbase + 0, dtype=tla.BFloat16)
    ap = tla.recast_ptr(abase + 4096, dtype=tla.Float16)
    bp = tla.recast_ptr(bbase + 12288, dtype=tla.Float16)
    cp = tla.recast_ptr(cbase + 0, dtype=tla.Float32)
    uv = flat(tla.recast_ptr(ubase + 22720, dtype=tla.BFloat16), 4096)
    cast = flat(tla.recast_ptr(ubase + 24768, dtype=tla.BFloat16), 4096)
    nz0 = flat(tla.recast_ptr(ubase + 26816, dtype=tla.BFloat16), 4160)
    nz1 = flat(tla.recast_ptr(ubase + 28928, dtype=tla.BFloat16), 4160)
    uqg = flat(tla.recast_ptr(ubase + 31040, dtype=tla.BFloat16), 4096)
    uqs = flat(tla.recast_ptr(ubase + 33088, dtype=tla.BFloat16), 4096)
    ukg = flat(tla.recast_ptr(ubase + 35136, dtype=tla.BFloat16), 4096)
    ugamma = flat(tla.recast_ptr(ubase + 37184, dtype=tla.Float32), 64)
    resident_ptr = tla.recast_ptr(lbase + 57344, dtype=tla.BFloat16)
    kb_ptr = tla.recast_ptr(lbase + 61440, dtype=tla.BFloat16)
    vb_ptr = tla.recast_ptr(lbase + 69632, dtype=tla.BFloat16)
    pa = tla.recast_ptr(abase + 12288, dtype=tla.BFloat16)
    pb = tla.recast_ptr(bbase + 20480, dtype=tla.BFloat16)
    pc = tla.recast_ptr(cbase + 8192, dtype=tla.Float32)
    if tla.const_expr(phase_flags is None):
        kb_ready = tla.cross_flag("kb_ready", mode=2)
        vb_ready = tla.cross_flag("vb_ready", mode=2)
        score_ready = tla.cross_flag("score_ready", mode=2)
        score_done = tla.cross_flag("score_done", mode=2)
        nt_ready = tla.cross_flag("nt_ready", mode=2)
    else:
        kb_ready, vb_ready, score_ready, score_done, nt_ready = phase_flags
    resident_ready = tla.flag("resident_ready", tla.pipes.FIX, tla.pipes.MTE1)
    cast_free = tla.flag("cast_free", tla.pipes.MTE3, tla.pipes.VECTOR)
    gm_loaded = tla.flag("gm_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    v_done = tla.flag("v_done", tla.pipes.VECTOR, tla.pipes.MTE3)
    l0_loaded = tla.flag("l0_loaded", tla.pipes.MTE1, tla.pipes.CUBE)
    left_free = tla.flag("left_free", tla.pipes.CUBE, tla.pipes.MTE1)
    cube_done = tla.flag("cube_done", tla.pipes.CUBE, tla.pipes.FIX)
    gm_ready = tla.flag("gm_ready", tla.pipes.MTE2, tla.pipes.MTE1)
    ready0 = tla.flag("ready0", tla.pipes.MTE1, tla.pipes.CUBE)
    ready1 = tla.flag("ready1", tla.pipes.MTE1, tla.pipes.CUBE)
    free0 = tla.flag("free0", tla.pipes.CUBE, tla.pipes.MTE1)
    free1 = tla.flag("free1", tla.pipes.CUBE, tla.pipes.MTE1)
    fix_ready = tla.flag("fix_ready", tla.pipes.FIX, tla.pipes.MTE1)
    acc_free = tla.flag("acc_free", tla.pipes.FIX, tla.pipes.CUBE)
    raw_gate = flat(tla.recast_ptr(ubase + 37248, dtype=g.ptr.dtype), 4096)
    gate_bf16 = flat(tla.recast_ptr(ubase + 41344, dtype=tla.BFloat16), 4096)
    raw_beta = flat(ubase + 43392, 64)
    bias_ub = flat(ubase + 43456, 64)
    alpha_ub = flat(ubase + 43520, 1)
    stats = flat(ubase + 43584, 384)
    beta_io = flat(tla.recast_ptr(ubase + 43968, dtype=beta.ptr.dtype), 1024)
    beta_column = flat(ubase + 43968, 512)
    beta_pitch = 1
    if tla.const_expr(beta_stride != 1):
        if tla.const_expr(beta.ptr.dtype == tla.Float32):
            beta_pitch = 8
        else:
            beta_pitch = 16
    peer_loaded = tla.flag("s1_peer_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    published = tla.cross_flag("s1_partial_ready", mode=1)
    consumed = tla.cross_flag("s1_partial_free", mode=1)
    negative = matrix64(lp, tla.arch.zN)
    identity = matrix64(lp + 4096, tla.arch.zN)
    zero = matrix64(lp + 8192, tla.arch.zN)
    scratch = matrix64(lp + 12288, tla.arch.zN)
    p2 = matrix64(lp + 16384, tla.arch.zN)
    p4 = matrix64(lp + 20480, tla.arch.zN)
    inverse = matrix64(lp + 24576, tla.arch.zN)
    a0 = matrix64(ap, tla.arch.zN)
    a1 = matrix64(ap + 4096, tla.arch.zN)
    b0 = matrix64(bp, tla.arch.nZ)
    b1 = matrix64(bp + 4096, tla.arch.nZ)
    ci = matrix64(cp, tla.arch.L0Clayout)
    ct = matrix64(cp + 4096, tla.arch.L0Clayout)
    with tla.cube():
        identity.fill(0)
        zero.fill(0)
        tla.pipe_barrier(tla.pipes.MTE2)
        for i in tla.range_constexpr(4):
            tla.copy(
                tla.tile_view(identity, tla.make_shape(16, 16), tla.make_coord(i, i)),
                row_major(identity16.ptr, 16, 16),
            )
        mutex_event(_mutex_gm_ready, tla.pipes.MTE2)
    with tla.vector():
        aiv = tla.arch.sub_block_idx()
        tla.copy(
            row_major(uq.ptr, valid, 64),
            strided_rows(q.ptr + aiv * 64, valid, 64, q_stride),
        )
        tla.copy(
            row_major(uk.ptr, valid, 64),
            strided_rows(k.ptr + aiv * 64, valid, 64, q_stride),
        )
        tla.copy(
            row_major(uv.ptr, valid, 64),
            strided_rows(v.ptr + aiv * 64, valid, 64, v_stride),
        )
        tla.copy(
            row_major(raw_gate.ptr, valid, 64),
            strided_rows(g.ptr + aiv * 64, valid, 64, v_stride),
        )
        if tla.const_expr(beta_stride == 1):
            tla.copy(row_major(beta_io.ptr, 1, valid), row_major(beta.ptr, 1, valid))
        else:
            tla.copy(
                strided_rows(beta_io.ptr, valid, 1, beta_pitch),
                strided_rows(beta.ptr, valid, 1, beta_stride),
            )
        tla.copy(row_major(bias_ub.ptr, 1, 64), row_major(bias.ptr + aiv * 64, 1, 64))
        tla.copy(row_major(alpha_ub.ptr, 1, 1), row_major(alpha.ptr, 1, 1))
        mutex_event(_mutex_gm_loaded, tla.pipes.MTE2)
        mutex_event(_mutex_gm_loaded, tla.pipes.VECTOR)
        zero_qkv_tail(uq, uk, uv, valid)
        prepare_gate_input(raw_gate, gate_bf16, valid)
        prepare_beta_input(beta_io, raw_beta, valid, beta_pitch)
        if tla.const_expr(normalize):
            compute_qk_partial_sums(uq, uk, at(stats, 0, 128))
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            tla.copy(
                row_major(exchange.ptr + aiv * 128, 1, 128),
                row_major(stats.ptr, 1, 128),
            )
            tla.cross_core_set_flag(published, tla.pipes.MTE3)
            tla.cross_core_wait_flag(published, tla.pipes.MTE2)
            tla.copy(
                row_major(stats.ptr + 128, 1, 128),
                row_major(exchange.ptr + (1 - aiv) * 128, 1, 128),
            )
            mutex_event(_mutex_peer_loaded, tla.pipes.MTE2)
        activate_gate(gate_bf16, bias_ub, alpha_ub, ug, valid, mode, lower, True)
        if tla.const_expr(normalize):
            mutex_event(_mutex_peer_loaded, tla.pipes.VECTOR)
            tla.cross_core_set_flag(consumed, tla.pipes.MTE2)
            tla.cross_core_wait_flag(consumed, tla.pipes.MTE3)
            compute_qk_inverse_vectors(
                at(stats, 0, 128), at(stats, 128, 128), at(stats, 256, 128)
            )
            normalize_local_qk(uq, uk, at(stats, 256, 128))
        activate_beta(raw_beta, ub, valid, beta_sigmoid, allow_negative)
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        if norm_owner:
            tla.copy(
                strided_rows(qn.ptr + aiv * 64, valid, 64, norm_stride),
                row_major(uq.ptr, valid, 64),
            )
            tla.copy(
                strided_rows(kn.ptr + aiv * 64, valid, 64, norm_stride),
                row_major(uk.ptr, valid, 64),
            )
        tla.copy(
            strided_rows(gcache.ptr + aiv * 64, valid, 64, 128),
            row_major(ug.ptr, valid, 64),
        )
        if aiv == 0:
            if tla.const_expr(beta_out_stride == 1):
                tla.copy(row_major(bact.ptr, 1, valid), row_major(ub.ptr, 1, valid))
            else:
                beta_to_column(ub, beta_column)
                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                tla.copy(
                    strided_rows(bact.ptr, valid, 1, beta_out_stride),
                    strided_rows(beta_column.ptr, valid, 1, 8),
                )
            if tla.const_expr(normalize):
                if norm_owner:
                    if tla.const_expr(rstd_stride == 1):
                        tla.copy(
                            row_major(rstd.ptr, 1, valid),
                            row_major(stats.ptr + 256, 1, valid),
                        )
                        tla.copy(
                            row_major(rstd.ptr + rstd_k_offset, 1, valid),
                            row_major(stats.ptr + 320, 1, valid),
                        )
                    else:
                        mutex_event(_mutex_cast_free, tla.pipes.MTE3)
                        mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
                        beta_to_column(at(stats, 256, 64), beta_column)
                        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                        mutex_event(_mutex_v_done, tla.pipes.MTE3)
                        tla.copy(
                            strided_rows(rstd.ptr, valid, 1, rstd_stride),
                            strided_rows(beta_column.ptr, valid, 1, 8),
                        )
                        mutex_event(_mutex_cast_free, tla.pipes.MTE3)
                        mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
                        beta_to_column(at(stats, 320, 64), beta_column)
                        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                        mutex_event(_mutex_v_done, tla.pipes.MTE3)
                        tla.copy(
                            strided_rows(
                                rstd.ptr + rstd_k_offset, valid, 1, rstd_stride
                            ),
                            strided_rows(beta_column.ptr, valid, 1, 8),
                        )
    if tla.const_expr(shared):
        with tla.cube():
            tla.cross_core_wait_flag(score_ready, tla.pipes.MTE1)
            issue_cached_score_mmads(
                shared_score_l1(score_ptr),
                left,
                right,
                cp,
                cp + 4096,
                l0_loaded,
                left_free,
            )
        with tla.vector():
            aiv = tla.arch.sub_block_idx()
            pack_shared_score_panel(uq, uk, ug, packed, 0, scale)
            pack_shared_score_panel(uq, uk, ug, packed, 1, scale)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            tla.copy(
                tla.tile_view(
                    shared_score_l1(score_ptr),
                    tla.make_shape(224, 64),
                    tla.make_coord(0, aiv),
                ),
                padded_nz(packed.ptr, 224),
            )
            tla.cross_core_set_flag(score_ready, tla.pipes.MTE3)
    else:
        per_d_scores(
            uq,
            uk,
            ug,
            flat(packed.ptr, 1024),
            flat(tla.recast_ptr(packed.ptr + 4096, dtype=tla.Float32), 1024),
            flat(packed.ptr + 1024, 1024),
            flat(packed.ptr + 2048, 1024),
            flat(packed.ptr + 3072, 1024),
            packed.ptr + 6144,
            score_ptr,
            left,
            right,
            cp,
            cp + 4096,
            scale,
            mode == 1,
        )
    with tla.cube():
        mutex_event(_mutex_cube_done, tla.pipes.CUBE)
        mutex_event(_mutex_cube_done, tla.pipes.FIX)
        tla.copy(
            row_major(kkt.ptr, 64, 64),
            ci,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        tla.copy(
            row_major(mqk.ptr, 64, 64),
            ct,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        tla.cross_core_set_flag(score_done, tla.pipes.FIX)
        tla.cross_core_wait_flag(nt_ready, tla.pipes.MTE1)
        mutex_event(_mutex_gm_ready, tla.pipes.MTE1)
        mutex_event(_mutex_free0, tla.pipes.CUBE)
        mutex_event(_mutex_free1, tla.pipes.CUBE)
        neumann_packed64_leaves(
            negative,
            identity,
            zero,
            scratch,
            p2,
            p4,
            inverse,
            a0,
            a1,
            b0,
            b1,
            ci,
            ct,
            ready0,
            ready1,
            free0,
            free1,
            fix_ready,
            acc_free,
        )
        compose_packed64_inverse(
            scratch,
            inverse,
            p2,
            zero,
            a0,
            a1,
            b0,
            b1,
            ci,
            ct,
            ready0,
            ready1,
            free0,
            free1,
            fix_ready,
            acc_free,
        )
        resident = matrix64(resident_ptr, tla.arch.zN)
        publish_resident_inverse(
            ci,
            row_major(akk.ptr, output_rows, 64),
            resident,
            True,
            cube_done,
            resident_ready,
        )
        kb = tla.make_tensor_like(kb_ptr, row_major(kb_ptr, 64, 128), tla.arch.zN)
        vb = tla.make_tensor_like(vb_ptr, row_major(vb_ptr, 64, 128), tla.arch.zN)
        compute_u_pre_and_w(
            resident,
            kb,
            vb,
            strided_rows(w.ptr, output_rows, 128, v_stride),
            strided_rows(u.ptr, output_rows, 128, v_stride),
            pa,
            pb,
            pc,
            resident_ready,
            kb_ready,
            vb_ready,
            l0_loaded,
            left_free,
            cube_done,
        )
        mutex_event(_mutex_free0, tla.pipes.MTE1)
        mutex_event(_mutex_free1, tla.pipes.MTE1)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        aiv = tla.arch.sub_block_idx()
        tla.cross_core_wait_flag(score_done, tla.pipes.VECTOR)
        apply_beta_and_finish_t(kkt, ub, nt, aiv)
        nd2nz_tile(nt, ntnz, 32, 64)
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        tla.copy(
            tla.tile_view(
                matrix64(lp, tla.arch.zN),
                tla.make_shape(32, 64),
                tla.make_coord(aiv, 0),
            ),
            padded_nz(ntnz.ptr, 32),
        )
        tla.cross_core_set_flag(nt_ready, tla.pipes.MTE3)
        beta_weighted_operand(uk, ub, ug, cast, True)
        nd2nz_tile(cast, nz0, 64, 64)
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        kb = tla.make_tensor_like(kb_ptr, row_major(kb_ptr, 64, 128), tla.arch.zN)
        vb = tla.make_tensor_like(vb_ptr, row_major(vb_ptr, 64, 128), tla.arch.zN)
        tla.copy(
            tla.tile_view(kb, tla.make_shape(64, 64), tla.make_coord(0, aiv)),
            padded_nz(nz0.ptr, 64),
        )
        tla.cross_core_set_flag(kb_ready, tla.pipes.MTE3)
        mutex_event(_mutex_cast_free, tla.pipes.MTE3)
        mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
        beta_weighted_operand(uv, ub, ug, cast, False)
        nd2nz_tile(cast, nz1, 64, 64)
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        tla.copy(
            tla.tile_view(vb, tla.make_shape(64, 64), tla.make_coord(0, aiv)),
            padded_nz(nz1.ptr, 64),
        )
        tla.cross_core_set_flag(vb_ready, tla.pipes.MTE3)
        prepare_decayed_q(uq, ug, uqg, uqs, scale)
        restored_state_inputs(uk, ug, ukg, ugamma)
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        tla.copy(
            strided_rows(qg.ptr + aiv * 64, output_rows, 64, v_stride),
            row_major(uqg.ptr, output_rows, 64),
        )
        tla.copy(
            strided_rows(kg.ptr + aiv * 64, output_rows, 64, v_stride),
            row_major(ukg.ptr, output_rows, 64),
        )
        tla.copy(
            tla.tile_view(
                row_major(qs.ptr, 64, 128),
                tla.make_shape(64, 64),
                tla.make_coord(0, aiv),
            ),
            row_major(uqs.ptr, 64, 64),
        )
        tla.copy(row_major(gamma.ptr + aiv * 64, 1, 64), row_major(ugamma.ptr, 1, 64))
        mask_mqk(mqk, aq, aiv)
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        rows = output_rows - aiv * 32
        if rows > 32:
            rows = 32
        if rows > 0:
            tla.copy(
                row_major(aqk.ptr + aiv * 2048, rows, 64), row_major(aq.ptr, rows, 64)
            )
        tla.pipe_barrier(tla.pipes.ALL)


# -----------------------------------------------------------------------------
# Staged first-stage pipeline.
# -----------------------------------------------------------------------------


@tla.jit
def copy_k_snapshot(source, destination):
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        mask, remaining = tla.update_mask(64, dtype=tla.BFloat16)
        for row in tla.range(64):
            value = at(source, row * 64, 64).load()
            at(destination, row * 64, 64).store(value, mask=mask)


@tla.jit
def task_addresses(
    task,
    prefix,
    group,
    count,
    logical_batch,
    storage_batch,
    token_start,
    length,
    nq,
    nv,
    storage_length,
    chunk_capacity,
    layout,
):
    local = task - prefix
    head = local // count
    chunk = group + local % count
    qhead = head // tla.Int64(nv // nq)
    valid = tla.Int64(64)
    if (chunk + tla.Int64(1)) * tla.Int64(64) > length:
        valid = length - chunk * tla.Int64(64)
    token = token_start + chunk * tla.Int64(64)
    qo = (
        (storage_batch * tla.Int64(nq) + qhead) * tla.Int64(storage_length) + token
    ) * tla.Int64(128)
    vo = (
        (storage_batch * tla.Int64(nv) + head) * tla.Int64(storage_length) + token
    ) * tla.Int64(128)
    if tla.const_expr(layout != "BNSD"):
        qo = (
            (storage_batch * tla.Int64(storage_length) + token) * tla.Int64(nq) + qhead
        ) * tla.Int64(128)
        vo = (
            (storage_batch * tla.Int64(storage_length) + token) * tla.Int64(nv) + head
        ) * tla.Int64(128)
    no = (
        (storage_batch * tla.Int64(nq) + qhead) * tla.Int64(storage_length) + token
    ) * tla.Int64(128)
    slot = logical_batch * tla.Int64(chunk_capacity) + chunk
    if tla.const_expr(layout == "TND"):
        no = (token * tla.Int64(nq) + qhead) * tla.Int64(128)
        slot = token_start // tla.Int64(64) + logical_batch + chunk
    co = slot * tla.Int64(nv) + head
    bo = (storage_batch * tla.Int64(storage_length) + token) * tla.Int64(nv) + head
    return head, valid.to(tla.Int32), qo, vo, no, co, bo


@tla.jit
def stage_one_pipeline(
    ubase,
    lbase,
    abase,
    bbase,
    cbase,
    q,
    k,
    v,
    g,
    beta,
    alpha,
    bias,
    exchange,
    identity16,
    aqk,
    akk,
    w,
    u,
    qg,
    kg,
    qs,
    gamma,
    qn,
    kn,
    rstd,
    gcache,
    bact,
    lo,
    hi,
    prefix,
    group,
    count,
    logical_batch,
    storage_batch,
    token_start,
    length,
    nq,
    nv,
    storage_length,
    chunk_capacity,
    layout,
    mode,
    lower,
    allow_negative,
    normalize,
    scale,
    shared,
    phase_flags,
    q_stride,
    v_stride,
    beta_stride,
    norm_stride,
    rstd_k_offset,
    rstd_stride,
    beta_sigmoid,
    save_matrices=True,
    save_cache=True,
    save_norm=True,
    save_beta=True,
):
    # 236800 bytes: raw Q/K rings, G/K snapshot rings, MQK ring, then temporaries.
    _mutex_gm_ready = tla.mutex("stage_one_pipeline_2357_gm_ready", id=0)
    _mutex_v_input_free = tla.mutex("stage_one_pipeline_2357_v_input_free", id=1)
    _mutex_v_done = tla.mutex("stage_one_pipeline_2357_v_done", id=2)
    _mutex_cast_free = tla.mutex("stage_one_pipeline_2357_cast_free", id=3)
    _mutex_free0 = tla.mutex("stage_one_pipeline_2357_free0", id=4)
    _mutex_free1 = tla.mutex("stage_one_pipeline_2357_free1", id=5)
    _mutex_acc_free = tla.mutex("stage_one_pipeline_2357_acc_free", id=6)
    _mutex_gm_loaded = tla.mutex("stage_one_pipeline_2357_gm_loaded", id=7)
    _mutex_peer_loaded = tla.mutex("stage_one_pipeline_2357_peer_loaded", id=8)
    _mutex_qk_input_free = tla.mutex("stage_one_pipeline_2357_qk_input_free", id=9)
    _mutex_cube_done = tla.mutex("stage_one_pipeline_2357_cube_done", id=10)
    ub = flat(ubase + 26624, 64)
    kkt = flat(ubase + 24576, 2048)
    uv = flat(tla.recast_ptr(ubase + 26688, dtype=tla.BFloat16), 4096)
    packed = flat(tla.recast_ptr(ubase + 28736, dtype=tla.BFloat16), 14400)
    ntnz = flat(tla.recast_ptr(ubase + 36960, dtype=tla.Float16), 2112)
    aq = flat(tla.recast_ptr(ubase + 38016, dtype=tla.BFloat16), 2048)
    nz0 = flat(tla.recast_ptr(ubase + 41088, dtype=tla.BFloat16), 4160)
    nz1 = flat(tla.recast_ptr(ubase + 43168, dtype=tla.BFloat16), 4160)
    uqg = flat(tla.recast_ptr(ubase + 45248, dtype=tla.BFloat16), 4096)
    uqs = flat(tla.recast_ptr(ubase + 47296, dtype=tla.BFloat16), 4096)
    ukg = flat(tla.recast_ptr(ubase + 49344, dtype=tla.BFloat16), 4096)
    ugamma = flat(ubase + 51392, 64)
    raw_gate = flat(tla.recast_ptr(ubase + 51456, dtype=g.ptr.dtype), 4096)
    stats = flat(ubase + 57600, 384)
    raw_beta = flat(ubase + 57984, 64)
    bias_ub = flat(ubase + 58048, 64)
    alpha_ub = flat(ubase + 58112, 1)
    beta_io = flat(tla.recast_ptr(ubase + 58176, dtype=beta.ptr.dtype), 1024)
    beta_column = flat(ubase + 58176, 512)
    beta_pitch = 1
    direct_beta = beta.ptr.dtype == tla.Float32 and (
        rstd_stride == 1 or not (normalize and save_norm)
    )
    if tla.const_expr(beta_stride != 1):
        if tla.const_expr(beta.ptr.dtype == tla.Float32):
            beta_pitch = 8
        else:
            beta_pitch = 16
    score_ptr = tla.recast_ptr(lbase, dtype=tla.BFloat16)
    lp = tla.recast_ptr(lbase + 28672, dtype=tla.Float16)
    left = tla.recast_ptr(abase, dtype=tla.BFloat16)
    right = tla.recast_ptr(bbase, dtype=tla.BFloat16)
    ap = tla.recast_ptr(abase + 4096, dtype=tla.Float16)
    bp = tla.recast_ptr(bbase + 12288, dtype=tla.Float16)
    cp = cbase
    resident_ptr = tla.recast_ptr(lbase + 57344, dtype=tla.BFloat16)
    kb_ptr = tla.recast_ptr(lbase + 61440, dtype=tla.BFloat16)
    vb_ptr = tla.recast_ptr(lbase + 69632, dtype=tla.BFloat16)
    pa = tla.recast_ptr(abase + 12288, dtype=tla.BFloat16)
    pb = tla.recast_ptr(bbase + 20480, dtype=tla.BFloat16)
    pc = cbase + 8192
    kb_ready, vb_ready, score_ready, score_done, nt_ready = phase_flags
    c3_free = tla.cross_flag("pipeline_c3_free", mode=2)
    published = tla.cross_flag("pipeline_partial_ready", mode=1)
    consumed = tla.cross_flag("pipeline_partial_free", mode=1)
    resident_ready = tla.flag("pipeline_resident", tla.pipes.FIX, tla.pipes.MTE1)
    cast_free = tla.flag("pipeline_cast_free", tla.pipes.MTE3, tla.pipes.VECTOR)
    gm_loaded = tla.flag("pipeline_gm_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    v_done = tla.flag("pipeline_v_done", tla.pipes.VECTOR, tla.pipes.MTE3)
    v_input_free = tla.flag("pipeline_v_input_free", tla.pipes.VECTOR, tla.pipes.MTE2)
    qk_input_free = tla.flag("pipeline_qk_input_free", tla.pipes.MTE3, tla.pipes.MTE2)
    l0_loaded = tla.flag("pipeline_l0_loaded", tla.pipes.MTE1, tla.pipes.CUBE)
    left_free = tla.flag("pipeline_left_free", tla.pipes.CUBE, tla.pipes.MTE1)
    cube_done = tla.flag("pipeline_cube_done", tla.pipes.CUBE, tla.pipes.FIX)
    gm_ready = tla.flag("pipeline_gm_ready", tla.pipes.MTE2, tla.pipes.MTE1)
    ready0 = tla.flag("pipeline_ready0", tla.pipes.MTE1, tla.pipes.CUBE)
    ready1 = tla.flag("pipeline_ready1", tla.pipes.MTE1, tla.pipes.CUBE)
    free0 = tla.flag("pipeline_free0", tla.pipes.CUBE, tla.pipes.MTE1)
    free1 = tla.flag("pipeline_free1", tla.pipes.CUBE, tla.pipes.MTE1)
    fix_ready = tla.flag("pipeline_fix_ready", tla.pipes.FIX, tla.pipes.MTE1)
    acc_free = tla.flag("pipeline_acc_free", tla.pipes.FIX, tla.pipes.CUBE)
    peer_loaded = tla.flag("pipeline_peer_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    negative = matrix64(lp, tla.arch.zN)
    identity = matrix64(lp + 4096, tla.arch.zN)
    zero = matrix64(lp + 8192, tla.arch.zN)
    scratch = matrix64(lp + 12288, tla.arch.zN)
    p2 = matrix64(lp + 16384, tla.arch.zN)
    p4 = matrix64(lp + 20480, tla.arch.zN)
    inverse = matrix64(lp + 24576, tla.arch.zN)
    a0 = matrix64(ap, tla.arch.zN)
    a1 = matrix64(ap + 4096, tla.arch.zN)
    b0 = matrix64(bp, tla.arch.nZ)
    b1 = matrix64(bp + 4096, tla.arch.nZ)
    ci = matrix64(cp, tla.arch.L0Clayout)
    ct = matrix64(cp + 4096, tla.arch.L0Clayout)
    resident = matrix64(resident_ptr, tla.arch.zN)
    kb = tla.make_tensor_like(kb_ptr, row_major(kb_ptr, 64, 128), tla.arch.zN)
    vb = tla.make_tensor_like(vb_ptr, row_major(vb_ptr, 64, 128), tla.arch.zN)
    with tla.cube():
        identity.fill(0)
        zero.fill(0)
        tla.pipe_barrier(tla.pipes.MTE2)
        for i in tla.range_constexpr(4):
            tla.copy(
                tla.tile_view(identity, tla.make_shape(16, 16), tla.make_coord(i, i)),
                row_major(identity16.ptr, 16, 16),
            )
        mutex_event(_mutex_gm_ready, tla.pipes.MTE2)
        mutex_event(_mutex_gm_ready, tla.pipes.MTE1)
    for tick in tla.range(lo.to(tla.Int32), (hi + tla.Int64(1)).to(tla.Int32)):
        task = tla.Int64(tick)
        parity = (task % tla.Int64(2)).to(tla.Int32)
        old_parity = 1 - parity
        uq = flat(tla.recast_ptr(ubase + parity * 2048, dtype=tla.BFloat16), 4096)
        uk = flat(
            tla.recast_ptr(ubase + 4096 + parity * 2048, dtype=tla.BFloat16), 4096
        )
        ug = flat(ubase + 8192 + parity * 4096, 4096)
        uk_snapshot = flat(
            tla.recast_ptr(ubase + 16384 + parity * 2048, dtype=tla.BFloat16), 4096
        )
        old_k = flat(
            tla.recast_ptr(ubase + 16384 + old_parity * 2048, dtype=tla.BFloat16), 4096
        )
        old_g = flat(ubase + 8192 + old_parity * 4096, 4096)
        mqk = flat(ubase + 20480 + parity * 2048, 2048)
        old_mqk = flat(ubase + 20480 + old_parity * 2048, 2048)
        # Address expressions for the empty epilogue are never dereferenced.
        head, valid, qo, vo, no, co, bo = task_addresses(
            task,
            prefix,
            group,
            count,
            logical_batch,
            storage_batch,
            token_start,
            length,
            nq,
            nv,
            storage_length,
            chunk_capacity,
            layout,
        )
        old_head, old_valid, _, old_vo, _, old_co, _ = task_addresses(
            task - tla.Int64(1),
            prefix,
            group,
            count,
            logical_batch,
            storage_batch,
            token_start,
            length,
            nq,
            nv,
            storage_length,
            chunk_capacity,
            layout,
        )
        if task < hi:
            with tla.vector():
                aiv_0 = tla.arch.sub_block_idx()
                # Raw gate arrives while the delayed inverse runs on Cube.
                mutex_event(_mutex_v_input_free, tla.pipes.VECTOR)
                mutex_event(_mutex_v_input_free, tla.pipes.MTE2)
                tla.copy(
                    row_major(raw_gate.ptr, valid, 64),
                    strided_rows(g.ptr + vo + aiv_0 * 64, valid, 64, v_stride),
                )
                if task == lo:
                    tla.copy(
                        row_major(bias_ub.ptr, 1, 64),
                        row_major(bias.ptr + head * tla.Int64(128) + aiv_0 * 64, 1, 64),
                    )
                    tla.copy(
                        row_major(alpha_ub.ptr, 1, 1), row_major(alpha.ptr + head, 1, 1)
                    )
                elif head != old_head:
                    tla.copy(
                        row_major(bias_ub.ptr, 1, 64),
                        row_major(bias.ptr + head * tla.Int64(128) + aiv_0 * 64, 1, 64),
                    )
                    tla.copy(
                        row_major(alpha_ub.ptr, 1, 1), row_major(alpha.ptr + head, 1, 1)
                    )
                if task == lo:
                    tla.copy(
                        row_major(uq.ptr, valid, 64),
                        strided_rows(q.ptr + qo + aiv_0 * 64, valid, 64, q_stride),
                    )
                    tla.copy(
                        row_major(uk.ptr, valid, 64),
                        strided_rows(k.ptr + qo + aiv_0 * 64, valid, 64, q_stride),
                    )
        if task > lo:
            with tla.vector():
                aiv_1 = tla.arch.sub_block_idx()
                tla.cross_core_wait_flag(score_done, tla.pipes.VECTOR)
                if task > lo + tla.Int64(1):
                    tla.cross_core_wait_flag(c3_free, tla.pipes.MTE3)
                finish_t_nz(kkt, ub, ntnz, aiv_1)
                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                tla.copy(
                    tla.tile_view(
                        negative, tla.make_shape(32, 64), tla.make_coord(aiv_1, 0)
                    ),
                    padded_nz(ntnz.ptr, 32),
                )
                tla.cross_core_set_flag(nt_ready, tla.pipes.MTE3)
                mutex_event(_mutex_cast_free, tla.pipes.MTE3)
                mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
                beta_weighted_nz(old_k, ub, old_g, nz0, True)
                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                tla.copy(
                    tla.tile_view(kb, tla.make_shape(64, 64), tla.make_coord(0, aiv_1)),
                    padded_nz(nz0.ptr, 64),
                )
                tla.cross_core_set_flag(kb_ready, tla.pipes.MTE3)
                mutex_event(_mutex_cast_free, tla.pipes.MTE3)
                mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
                beta_weighted_nz(uv, ub, old_g, nz1, False)
                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                tla.copy(
                    tla.tile_view(vb, tla.make_shape(64, 64), tla.make_coord(0, aiv_1)),
                    padded_nz(nz1.ptr, 64),
                )
                tla.cross_core_set_flag(vb_ready, tla.pipes.MTE3)
                mutex_event(_mutex_v_input_free, tla.pipes.VECTOR)
                mutex_event(_mutex_v_input_free, tla.pipes.MTE2)
            with tla.cube():
                tla.cross_core_wait_flag(nt_ready, tla.pipes.MTE1)
                mutex_event(_mutex_free0, tla.pipes.CUBE)
                mutex_event(_mutex_free1, tla.pipes.CUBE)
                neumann_packed64_leaves(
                    negative,
                    identity,
                    zero,
                    scratch,
                    p2,
                    p4,
                    inverse,
                    a0,
                    a1,
                    b0,
                    b1,
                    ci,
                    ct,
                    ready0,
                    ready1,
                    free0,
                    free1,
                    fix_ready,
                    acc_free,
                )
                compose_packed64_inverse(
                    scratch,
                    inverse,
                    p2,
                    zero,
                    a0,
                    a1,
                    b0,
                    b1,
                    ci,
                    ct,
                    ready0,
                    ready1,
                    free0,
                    free1,
                    fix_ready,
                    acc_free,
                )
                publish_resident_inverse(
                    ci,
                    row_major(akk.ptr + old_co * tla.Int64(4096), old_valid, 64),
                    resident,
                    save_matrices,
                    cube_done,
                    resident_ready,
                )
                mutex_event(_mutex_free0, tla.pipes.MTE1)
                mutex_event(_mutex_free1, tla.pipes.MTE1)
                mutex_event(_mutex_acc_free, tla.pipes.FIX)
                mutex_event(_mutex_acc_free, tla.pipes.CUBE)
        if task < hi:
            with tla.vector():
                aiv_2 = tla.arch.sub_block_idx()
                tla.copy(
                    row_major(uv.ptr, valid, 64),
                    strided_rows(v.ptr + vo + aiv_2 * 64, valid, 64, v_stride),
                )
                if tla.const_expr(beta_stride == 1):
                    tla.copy(
                        row_major(beta_io.ptr, 1, valid),
                        row_major(beta.ptr + vo // tla.Int64(128), 1, valid),
                    )
                else:
                    tla.copy(
                        strided_rows(beta_io.ptr, valid, 1, beta_pitch),
                        strided_rows(
                            beta.ptr + vo // tla.Int64(128), valid, 1, beta_stride
                        ),
                    )
                mutex_event(_mutex_gm_loaded, tla.pipes.MTE2)
                mutex_event(_mutex_gm_loaded, tla.pipes.VECTOR)
                if valid < 64:
                    zero_qkv_tail(uq, uk, uv, valid)
                if tla.const_expr(not direct_beta):
                    prepare_beta_input(beta_io, raw_beta, valid, beta_pitch)
                if tla.const_expr(normalize):
                    compute_qk_partial_sums(uq, uk, at(stats, 0, 128))
                    mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                    mutex_event(_mutex_v_done, tla.pipes.MTE3)
                    tla.copy(
                        row_major(exchange.ptr + aiv_2 * 128, 1, 128),
                        row_major(stats.ptr, 1, 128),
                    )
                    tla.cross_core_set_flag(published, tla.pipes.MTE3)
                    tla.cross_core_wait_flag(published, tla.pipes.MTE2)
                    tla.copy(
                        row_major(stats.ptr + 128, 1, 128),
                        row_major(exchange.ptr + (1 - aiv_2) * 128, 1, 128),
                    )
                    mutex_event(_mutex_peer_loaded, tla.pipes.MTE2)
                if valid == 64:
                    activate_gate(
                        raw_gate, bias_ub, alpha_ub, ug, 64, mode, lower, True
                    )
                else:
                    activate_gate(
                        raw_gate, bias_ub, alpha_ub, ug, valid, mode, lower, True
                    )
                if tla.const_expr(normalize):
                    mutex_event(_mutex_peer_loaded, tla.pipes.VECTOR)
                    tla.cross_core_set_flag(consumed, tla.pipes.MTE2)
                    tla.cross_core_wait_flag(consumed, tla.pipes.MTE3)
                    normalize_and_snapshot(
                        uq,
                        uk,
                        at(stats, 0, 128),
                        at(stats, 128, 128),
                        at(stats, 256, 128),
                        uk_snapshot,
                    )
                else:
                    copy_k_snapshot(uk, uk_snapshot)
                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                if tla.const_expr(save_norm):
                    if head % tla.Int64(nv // nq) == tla.Int64(0):
                        tla.copy(
                            strided_rows(
                                qn.ptr + no + aiv_2 * 64, valid, 64, norm_stride
                            ),
                            row_major(uq.ptr, valid, 64),
                        )
                        tla.copy(
                            strided_rows(
                                kn.ptr + no + aiv_2 * 64, valid, 64, norm_stride
                            ),
                            row_major(uk.ptr, valid, 64),
                        )
                if tla.const_expr(save_cache):
                    tla.copy(
                        strided_rows(
                            gcache.ptr + co * tla.Int64(8192) + aiv_2 * 64,
                            valid,
                            64,
                            128,
                        ),
                        row_major(ug.ptr, valid, 64),
                    )
                if tla.const_expr(normalize and save_norm):
                    if aiv_2 == 0:
                        if head % tla.Int64(nv // nq) == tla.Int64(0):
                            if tla.const_expr(rstd_stride == 1):
                                tla.copy(
                                    row_major(
                                        rstd.ptr + no // tla.Int64(128), 1, valid
                                    ),
                                    row_major(stats.ptr + 256, 1, valid),
                                )
                                tla.copy(
                                    row_major(
                                        rstd.ptr
                                        + no // tla.Int64(128)
                                        + tla.Int64(rstd_k_offset),
                                        1,
                                        valid,
                                    ),
                                    row_major(stats.ptr + 320, 1, valid),
                                )
                            else:
                                mutex_event(_mutex_cast_free, tla.pipes.MTE3)
                                mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
                                beta_to_column(at(stats, 256, 64), beta_column)
                                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                                tla.copy(
                                    strided_rows(
                                        rstd.ptr + no // tla.Int64(128),
                                        valid,
                                        1,
                                        rstd_stride,
                                    ),
                                    strided_rows(beta_column.ptr, valid, 1, 8),
                                )
                                mutex_event(_mutex_cast_free, tla.pipes.MTE3)
                                mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
                                beta_to_column(at(stats, 320, 64), beta_column)
                                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                                tla.copy(
                                    strided_rows(
                                        rstd.ptr
                                        + no // tla.Int64(128)
                                        + tla.Int64(rstd_k_offset),
                                        valid,
                                        1,
                                        rstd_stride,
                                    ),
                                    strided_rows(beta_column.ptr, valid, 1, 8),
                                )
                mutex_event(_mutex_cast_free, tla.pipes.MTE3)
                mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
                prepare_decayed_q(uq, ug, uqg, uqs, scale)
                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                if tla.const_expr(save_cache):
                    tla.copy(
                        strided_rows(qg.ptr + vo + aiv_2 * 64, valid, 64, v_stride),
                        row_major(uqg.ptr, valid, 64),
                    )
                tla.copy(
                    tla.tile_view(
                        row_major(qs.ptr + task * tla.Int64(8192), 64, 128),
                        tla.make_shape(64, 64),
                        tla.make_coord(0, aiv_2),
                    ),
                    row_major(uqs.ptr, 64, 64),
                )
                if task + tla.Int64(1) < hi:
                    (
                        _,
                        next_valid,
                        next_qo,
                        _,
                        _,
                        _,
                        _,
                    ) = task_addresses(
                        task + tla.Int64(1),
                        prefix,
                        group,
                        count,
                        logical_batch,
                        storage_batch,
                        token_start,
                        length,
                        nq,
                        nv,
                        storage_length,
                        chunk_capacity,
                        layout,
                    )
                    next_q = flat(
                        tla.recast_ptr(ubase + old_parity * 2048, dtype=tla.BFloat16),
                        4096,
                    )
                    next_k = flat(
                        tla.recast_ptr(
                            ubase + 4096 + old_parity * 2048, dtype=tla.BFloat16
                        ),
                        4096,
                    )
                    mutex_event(_mutex_qk_input_free, tla.pipes.MTE3)
                    mutex_event(_mutex_qk_input_free, tla.pipes.MTE2)
                    tla.copy(
                        row_major(next_q.ptr, next_valid, 64),
                        strided_rows(
                            q.ptr + next_qo + aiv_2 * 64, next_valid, 64, q_stride
                        ),
                    )
                    tla.copy(
                        row_major(next_k.ptr, next_valid, 64),
                        strided_rows(
                            k.ptr + next_qo + aiv_2 * 64, next_valid, 64, q_stride
                        ),
                    )
            if tla.const_expr(shared):
                with tla.cube():
                    tla.cross_core_wait_flag(score_ready, tla.pipes.MTE1)
                    issue_cached_score_mmads(
                        shared_score_l1(score_ptr),
                        left,
                        right,
                        cp,
                        cp + 4096,
                        l0_loaded,
                        left_free,
                    )
                with tla.vector():
                    aiv_3 = tla.arch.sub_block_idx()
                    pack_shared_score_panel(uq, uk, ug, packed, 0, scale)
                    pack_shared_score_panel(uq, uk, ug, packed, 1, scale)
                    mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                    mutex_event(_mutex_v_done, tla.pipes.MTE3)
                    tla.copy(
                        tla.tile_view(
                            shared_score_l1(score_ptr),
                            tla.make_shape(224, 64),
                            tla.make_coord(0, aiv_3),
                        ),
                        padded_nz(packed.ptr, 224),
                    )
                    tla.cross_core_set_flag(score_ready, tla.pipes.MTE3)
            else:
                per_d_scores(
                    uq,
                    uk,
                    ug,
                    flat(packed.ptr, 1024),
                    flat(tla.recast_ptr(packed.ptr + 4096, dtype=tla.Float32), 1024),
                    flat(packed.ptr + 1024, 1024),
                    flat(packed.ptr + 2048, 1024),
                    flat(packed.ptr + 3072, 1024),
                    packed.ptr + 6144,
                    score_ptr,
                    left,
                    right,
                    cp,
                    cp + 4096,
                    scale,
                    mode == 1,
                )
            with tla.cube():
                mutex_event(_mutex_cube_done, tla.pipes.CUBE)
                mutex_event(_mutex_cube_done, tla.pipes.FIX)
                tla.copy(
                    row_major(kkt.ptr, 64, 64),
                    ci,
                    tla.params.CopyL0C2DstParams(
                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                    ),
                )
                tla.copy(
                    row_major(mqk.ptr, 64, 64),
                    ct,
                    tla.params.CopyL0C2DstParams(
                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                    ),
                )
                tla.cross_core_set_flag(score_done, tla.pipes.FIX)
            with tla.vector():
                aiv_4 = tla.arch.sub_block_idx()
                if tla.const_expr(direct_beta):
                    activate_beta_f32_input(
                        beta_io, ub, valid, beta_pitch, beta_sigmoid, allow_negative
                    )
                else:
                    activate_beta(raw_beta, ub, valid, beta_sigmoid, allow_negative)
                if tla.const_expr(save_beta):
                    if aiv_4 == 0:
                        mutex_event(_mutex_cast_free, tla.pipes.MTE3)
                        mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
                        beta_to_column(ub, beta_column)
                        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                        mutex_event(_mutex_v_done, tla.pipes.MTE3)
                        tla.copy(
                            strided_rows(bact.ptr + bo, valid, 1, nv),
                            strided_rows(beta_column.ptr, valid, 1, 8),
                        )
        if task > lo:
            with tla.cube():
                compute_u_pre_and_w(
                    resident,
                    kb,
                    vb,
                    strided_rows(w.ptr + old_vo, old_valid, 128, v_stride),
                    strided_rows(u.ptr + old_vo, old_valid, 128, v_stride),
                    pa,
                    pb,
                    pc,
                    resident_ready,
                    kb_ready,
                    vb_ready,
                    l0_loaded,
                    left_free,
                    cube_done,
                )
                tla.cross_core_set_flag(c3_free, tla.pipes.MTE1)
                mutex_event(_mutex_acc_free, tla.pipes.FIX)
                mutex_event(_mutex_acc_free, tla.pipes.CUBE)
            with tla.vector():
                aiv_5 = tla.arch.sub_block_idx()
                restored_state_inputs(old_k, old_g, ukg, ugamma)
                mask_mqk(old_mqk, aq, aiv_5)
                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                tla.copy(
                    strided_rows(kg.ptr + old_vo + aiv_5 * 64, old_valid, 64, v_stride),
                    row_major(ukg.ptr, old_valid, 64),
                )
                tla.copy(
                    row_major(
                        gamma.ptr + (task - tla.Int64(1)) * tla.Int64(128) + aiv_5 * 64,
                        1,
                        64,
                    ),
                    row_major(ugamma.ptr, 1, 64),
                )
                rows = old_valid - aiv_5 * 32
                if rows > 32:
                    rows = 32
                if rows > 0:
                    tla.copy(
                        row_major(
                            aqk.ptr + old_co * tla.Int64(4096) + aiv_5 * 2048, rows, 64
                        ),
                        row_major(aq.ptr, rows, 64),
                    )
    with tla.vector():
        if hi > lo:
            tla.cross_core_wait_flag(c3_free, tla.pipes.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)


# -----------------------------------------------------------------------------
# Staged state recurrence.
# -----------------------------------------------------------------------------


@tla.jit
def load_chunk_l1(destination, source, valid, columns, stride):
    """Source _load_chunk_rows: zero a tail slot, then copy each 16-column tile."""
    if valid < 64:
        destination.fill(0)
        tla.pipe_barrier(tla.pipes.MTE2)
        for col in tla.range_constexpr(columns // 16):
            tla.copy(
                tla.tile_view(
                    destination, tla.make_shape(64, 16), tla.make_coord(0, col)
                ),
                strided_rows(source + col * 16, valid, 16, stride),
            )
    else:
        tla.copy(destination, strided_rows(source, 64, columns, stride))


@tla.jit
def load_u_rows(destination, source, valid, dv, dv_idx, aiv, stride):
    rows = valid - aiv * 32
    if rows > 32:
        rows = 32
    if rows > 0:
        tla.copy(
            row_major(destination, rows, dv),
            strided_rows(source + aiv * 32 * stride + dv_idx * dv, rows, dv, stride),
        )


@tla.jit
def state_local_flags():
    return (
        tla.flag("s2_gm_l1", tla.pipes.MTE2, tla.pipes.MTE1),
        tla.flag("s2_gm_ub", tla.pipes.MTE2, tla.pipes.VECTOR),
        tla.flag("s2_input_free0", tla.pipes.VECTOR, tla.pipes.MTE2),
        tla.flag("s2_input_free1", tla.pipes.VECTOR, tla.pipes.MTE2),
        tla.flag("s2_l1_free0", tla.pipes.MTE1, tla.pipes.MTE2),
        tla.flag("s2_l1_free1", tla.pipes.MTE1, tla.pipes.MTE2),
        tla.flag("s2_v_done", tla.pipes.VECTOR, tla.pipes.MTE3),
        tla.flag("s2_history_free", tla.pipes.MTE3, tla.pipes.VECTOR),
        tla.flag("s2_loaded", tla.pipes.MTE1, tla.pipes.CUBE),
        tla.flag("s2_cube_done", tla.pipes.CUBE, tla.pipes.FIX),
        tla.flag("s2_acc_free", tla.pipes.FIX, tla.pipes.CUBE),
        tla.flag("s2_output_operands_free", tla.pipes.CUBE, tla.pipes.MTE1),
    )


@tla.jit
def stage_two_group(
    ubase,
    lbase,
    abase,
    bbase,
    cbase,
    qs,
    w,
    kg,
    aqk,
    u,
    initial,
    gamma,
    out,
    vnew,
    history,
    final,
    chunks,
    valid_last,
    dv,
    dv_idx,
    output_stride,
    save_history,
    phase_flags=None,
    token_stride=128,
    matrix_chunk_stride=4096,
    history_chunk_stride=16384,
    local_flags=None,
    save_vnew=True,
):
    """All pointer arguments address this group's first chunk of one value head.

    Scratch operands are padded [chunk,64,128]; Aqk [chunk,64,64], gamma
    [chunk,128]. Output may be strided BSND/TND. State is [V,K].
    """
    _mutex_gm_l1 = tla.mutex("stage_two_group_3083_gm_l1", id=0)
    _mutex_acc_free = tla.mutex("stage_two_group_3083_acc_free", id=1)
    _mutex_output_operands_free = tla.mutex(
        "stage_two_group_3083_output_operands_free", id=2
    )
    _mutex_loaded = tla.mutex("stage_two_group_3083_loaded", id=3)
    _mutex_l1_free0 = tla.mutex("stage_two_group_3083_l1_free0", id=4)
    _mutex_l1_free1 = tla.mutex("stage_two_group_3083_l1_free1", id=5)
    _mutex_cube_done = tla.mutex("stage_two_group_3083_cube_done", id=6)
    _mutex_gm_ub = tla.mutex("stage_two_group_3083_gm_ub", id=7)
    _mutex_v_done = tla.mutex("stage_two_group_3083_v_done", id=8)
    _mutex_history_free = tla.mutex("stage_two_group_3083_history_free", id=9)
    _mutex_input_free0 = tla.mutex("stage_two_group_3083_input_free0", id=10)
    _mutex_input_free1 = tla.mutex("stage_two_group_3083_input_free1", id=11)
    state = flat(ubase, dv * 64)
    delta = flat(ubase + 8192, dv * 64)
    snz = flat(tla.recast_ptr(ubase + 20480, dtype=tla.BFloat16), (dv + 1) * 64)
    uin_ptr = tla.recast_ptr(ubase + 24640, dtype=tla.BFloat16)
    du = ubase + 28736
    uv = tla.recast_ptr(ubase + 32832, dtype=tla.BFloat16)
    unz = flat(tla.recast_ptr(ubase + 34880, dtype=tla.BFloat16), 33 * dv)
    gamma_ptr = ubase + 36992
    history_copy = ubase + 37120
    a0 = abase
    a1 = abase + 8192
    ua = abase + 16384
    qa = abase + 24576
    hb = bbase
    kb = bbase + 16384
    vb = bbase + 24576
    cu = cbase
    co = cbase + 8192
    ch = cbase + 16384
    if tla.const_expr(local_flags is None):
        local_flags = state_local_flags()
    (
        gm_l1,
        gm_ub,
        input_free0,
        input_free1,
        l1_free0,
        l1_free1,
        v_done,
        history_free,
        loaded,
        cube_done,
        acc_free,
        output_operands_free,
    ) = local_flags
    if tla.const_expr(phase_flags is None):
        h_ready = tla.cross_flag("s2_h_ready", mode=2)
        du_ready = tla.cross_flag("s2_du_ready", mode=2)
        u_ready = tla.cross_flag("s2_u_ready", mode=2)
        dh_ready = tla.cross_flag("s2_dh_ready", mode=2)
    else:
        h_ready, du_ready, u_ready, dh_ready = phase_flags
    with tla.cube():
        ql = tla.make_tensor_like(lbase, row_major(lbase, 64, 128), tla.arch.zN)
        wl = tla.make_tensor_like(lbase + 16384, row_major(lbase, 64, 128), tla.arch.zN)
        kl = tla.make_tensor_like(lbase + 32768, row_major(lbase, 64, 128), tla.arch.zN)
        al = tla.make_tensor_like(lbase + 49152, row_major(lbase, 64, 64), tla.arch.zN)
        first_valid = 64
        if chunks == 1:
            first_valid = valid_last
        tla.copy(ql, row_major(qs, 64, 128))
        load_chunk_l1(wl, w, first_valid, 128, token_stride)
        load_chunk_l1(kl, kg, first_valid, 128, token_stride)
        load_chunk_l1(al, aqk, first_valid, 64, 64)
        mutex_event(_mutex_gm_l1, tla.pipes.MTE2)
        mutex_event(_mutex_acc_free, tla.pipes.FIX)
        mutex_event(_mutex_output_operands_free, tla.pipes.CUBE)
        for chunk in tla.range(chunks):
            slot = chunk % 2
            next_slot = (chunk + 1) % 2
            next_valid = 64
            if chunk + 2 == chunks:
                next_valid = valid_last
            ql = tla.make_tensor_like(
                lbase + slot * 8192, row_major(lbase, 64, 128), tla.arch.zN
            )
            wl = tla.make_tensor_like(
                lbase + 16384 + slot * 8192, row_major(lbase, 64, 128), tla.arch.zN
            )
            kl = tla.make_tensor_like(
                lbase + 32768 + slot * 8192, row_major(lbase, 64, 128), tla.arch.zN
            )
            al = tla.make_tensor_like(
                lbase + 49152 + slot * 4096, row_major(lbase, 64, 64), tla.arch.zN
            )
            hl = tla.make_tensor_like(
                lbase + 57344 + slot * 16384, row_major(lbase, dv, 128), tla.arch.zN
            )
            ul = tla.make_tensor_like(
                lbase + 90112 + slot * 8192, row_major(lbase, 64, dv), tla.arch.zN
            )
            mutex_event(_mutex_gm_l1, tla.pipes.MTE1)
            tla.cross_core_wait_flag(h_ready, tla.pipes.MTE1)
            compute_state_projections(
                ql, wl, hl, a0, a1, hb, cu, co, dv, loaded, cube_done, acc_free
            )
            mutex_event(_mutex_output_operands_free, tla.pipes.MTE1)
            load_recurrent_operands(kl, al, kb, qa, dv, loaded)
            mutex_event(_mutex_loaded, tla.pipes.CUBE)
            if chunk + 2 < chunks:
                if slot == 0:
                    mutex_event(_mutex_l1_free0, tla.pipes.MTE1)
                else:
                    mutex_event(_mutex_l1_free1, tla.pipes.MTE1)
            if chunk + 1 < chunks:
                if chunk > 0:
                    if next_slot == 0:
                        mutex_event(_mutex_l1_free0, tla.pipes.MTE2)
                    else:
                        mutex_event(_mutex_l1_free1, tla.pipes.MTE2)
                next_q = tla.make_tensor_like(
                    lbase + next_slot * 8192, row_major(lbase, 64, 128), tla.arch.zN
                )
                next_w = tla.make_tensor_like(
                    lbase + 16384 + next_slot * 8192,
                    row_major(lbase, 64, 128),
                    tla.arch.zN,
                )
                tla.copy(next_q, row_major(qs + (chunk + 1) * 8192, 64, 128))
                load_chunk_l1(
                    next_w,
                    w + (chunk + 1) * 64 * token_stride,
                    next_valid,
                    128,
                    token_stride,
                )
            mutex_event(_mutex_cube_done, tla.pipes.FIX)
            tla.copy(
                row_major(du, 64, dv),
                tla.make_tensor_like(cu, row_major(cu, 64, dv), tla.arch.L0Clayout),
                tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
            )
            tla.cross_core_set_flag(du_ready, tla.pipes.FIX)
            tla.cross_core_wait_flag(u_ready, tla.pipes.MTE1)
            compute_state_delta(ul, kb, ua, ch, dv, loaded)
            mutex_event(_mutex_cube_done, tla.pipes.CUBE)
            mutex_event(_mutex_cube_done, tla.pipes.FIX)
            tla.copy(
                strided_rows(delta.ptr, dv, 128, 64),
                tla.make_tensor_like(ch, row_major(ch, dv, 128), tla.arch.L0Clayout),
                tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N),
            )
            tla.cross_core_set_flag(dh_ready, tla.pipes.FIX)
            accumulate_output(ul, qa, vb, co, dv, loaded)
            mutex_event(_mutex_output_operands_free, tla.pipes.CUBE)
            mutex_event(_mutex_cube_done, tla.pipes.CUBE)
            if chunk + 1 < chunks:
                next_k = tla.make_tensor_like(
                    lbase + 32768 + next_slot * 8192,
                    row_major(lbase, 64, 128),
                    tla.arch.zN,
                )
                next_a = tla.make_tensor_like(
                    lbase + 49152 + next_slot * 4096,
                    row_major(lbase, 64, 64),
                    tla.arch.zN,
                )
                load_chunk_l1(
                    next_k,
                    kg + (chunk + 1) * 64 * token_stride,
                    next_valid,
                    128,
                    token_stride,
                )
                load_chunk_l1(
                    next_a, aqk + (chunk + 1) * matrix_chunk_stride, next_valid, 64, 64
                )
                mutex_event(_mutex_gm_l1, tla.pipes.MTE2)
            valid = 64
            if chunk + 1 == chunks:
                valid = valid_last
            mutex_event(_mutex_cube_done, tla.pipes.FIX)
            tla.copy(
                strided_rows(
                    out + chunk * 64 * output_stride + dv_idx * dv,
                    valid,
                    dv,
                    output_stride,
                ),
                tla.make_tensor_like(co, row_major(co, 64, dv), tla.arch.L0Clayout),
            )
            mutex_event(_mutex_acc_free, tla.pipes.FIX)
        mutex_event(_mutex_acc_free, tla.pipes.CUBE)
        mutex_event(_mutex_output_operands_free, tla.pipes.MTE1)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        aiv = tla.arch.sub_block_idx()
        tla.copy(
            row_major(state.ptr, dv, 64),
            strided_rows(initial + dv_idx * dv * 128 + aiv * 64, dv, 64, 128),
        )
        tla.copy(row_major(gamma_ptr, 1, 64), row_major(gamma + aiv * 64, 1, 64))
        first_valid = 64
        if chunks == 1:
            first_valid = valid_last
        load_u_rows(uin_ptr, u, first_valid, dv, dv_idx, aiv, token_stride)
        mutex_event(_mutex_gm_ub, tla.pipes.MTE2)
        mutex_event(_mutex_gm_ub, tla.pipes.VECTOR)
        cast_state_nz(state, snz, dv)
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        hl = tla.make_tensor_like(lbase + 57344, row_major(lbase, dv, 128), tla.arch.zN)
        tla.copy(
            tla.tile_view(hl, tla.make_shape(dv, 64), tla.make_coord(0, aiv)),
            padded_nz(snz.ptr, dv),
        )
        tla.cross_core_set_flag(h_ready, tla.pipes.MTE3)
        mutex_event(_mutex_history_free, tla.pipes.MTE3)
        for chunk in tla.range(chunks):
            slot = chunk % 2
            next_slot = (chunk + 1) % 2
            valid = 64
            if chunk + 1 == chunks:
                valid = valid_last
            next_valid = 64
            if chunk + 2 == chunks:
                next_valid = valid_last
            if chunk > 0:
                mutex_event(_mutex_gm_ub, tla.pipes.VECTOR)
            if chunk + 1 < chunks:
                if chunk > 0:
                    if next_slot == 0:
                        mutex_event(_mutex_input_free0, tla.pipes.MTE2)
                    else:
                        mutex_event(_mutex_input_free1, tla.pipes.MTE2)
                tla.copy(
                    row_major(gamma_ptr + next_slot * 64, 1, 64),
                    row_major(gamma + (chunk + 1) * 128 + aiv * 64, 1, 64),
                )
                load_u_rows(
                    uin_ptr + next_slot * 4096,
                    u + (chunk + 1) * 64 * token_stride,
                    next_valid,
                    dv,
                    dv_idx,
                    aiv,
                    token_stride,
                )
                mutex_event(_mutex_gm_ub, tla.pipes.MTE2)
            tla.cross_core_wait_flag(du_ready, tla.pipes.VECTOR)
            finish_u_delta_nz(
                row_major(uin_ptr + slot * 4096, 32, dv),
                row_major(du, 32, dv),
                row_major(uv, 32, dv),
                unz,
                dv,
                aiv,
                valid,
                save_vnew,
            )
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            ul = tla.make_tensor_like(
                lbase + 90112 + slot * 8192, row_major(lbase, 64, dv), tla.arch.zN
            )
            tla.copy(
                tla.tile_view(ul, tla.make_shape(32, dv), tla.make_coord(aiv, 0)),
                padded_nz(unz.ptr, 32, dv),
            )
            tla.cross_core_set_flag(u_ready, tla.pipes.MTE3)
            rows = valid - aiv * 32
            if rows > 32:
                rows = 32
            if rows < 0:
                rows = 0
            if tla.const_expr(save_vnew):
                if rows > 0:
                    tla.copy(
                        strided_rows(
                            vnew
                            + (chunk * 64 + aiv * 32) * output_stride
                            + dv_idx * dv,
                            rows,
                            dv,
                            output_stride,
                        ),
                        row_major(uv, rows, dv),
                    )
            tla.cross_core_wait_flag(dh_ready, tla.pipes.VECTOR)
            mutex_event(_mutex_history_free, tla.pipes.VECTOR)
            if tla.const_expr(save_history):
                tla.copy(row_major(history_copy, dv, 64), row_major(state.ptr, dv, 64))
                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                tla.copy(
                    strided_rows(
                        history
                        + chunk * history_chunk_stride
                        + dv_idx * dv * 128
                        + aiv * 64,
                        dv,
                        64,
                        128,
                    ),
                    row_major(history_copy, dv, 64),
                )
            mutex_event(_mutex_history_free, tla.pipes.MTE3)
            update_state_nz(state, delta, flat(gamma_ptr + slot * 64, 64), snz, dv)
            if chunk + 2 < chunks:
                if slot == 0:
                    mutex_event(_mutex_input_free0, tla.pipes.VECTOR)
                else:
                    mutex_event(_mutex_input_free1, tla.pipes.VECTOR)
            if chunk + 1 < chunks:
                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                hl = tla.make_tensor_like(
                    lbase + 57344 + next_slot * 16384,
                    row_major(lbase, dv, 128),
                    tla.arch.zN,
                )
                tla.copy(
                    tla.tile_view(hl, tla.make_shape(dv, 64), tla.make_coord(0, aiv)),
                    padded_nz(snz.ptr, dv),
                )
                tla.cross_core_set_flag(h_ready, tla.pipes.MTE3)
        mutex_event(_mutex_history_free, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        tla.copy(
            strided_rows(final + dv_idx * dv * 128 + aiv * 64, dv, 64, 128),
            row_major(state.ptr, dv, 64),
        )
        tla.pipe_barrier(tla.pipes.ALL)


# -----------------------------------------------------------------------------
# Staged cross-core boundary.
# -----------------------------------------------------------------------------


@tla.jit
def stage_boundary(aiv_ready, aic_all, aic_release):
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
        tla.cross_core_wait_flag(aiv_ready, tla.pipes.SCALAR)
        tla.cross_core_set_flag(aic_all, tla.pipes.FIX)
        tla.cross_core_wait_flag(aic_all, tla.pipes.SCALAR)
        tla.cross_core_set_flag(aic_release, tla.pipes.MTE3)
    with tla.vector():
        tla.pipe_barrier(tla.pipes.ALL)
        tla.cross_core_set_flag(aiv_ready, tla.pipes.MTE3)
        tla.cross_core_wait_flag(aic_release, tla.pipes.SCALAR)


# -----------------------------------------------------------------------------
# Staged kernel and task scheduler.
# -----------------------------------------------------------------------------


@tla.jit
def state_items(
    dv,
    ub,
    l1,
    a,
    b,
    c,
    qs,
    w,
    kg,
    aqk,
    u,
    initial,
    gamma,
    out,
    vnew,
    history,
    final,
    meta,
    active_batches,
    batch_base,
    storage_base,
    token_base,
    length_base,
    group_base,
    count_base,
    prefix1_base,
    prefix2_base,
    start,
    end,
    phase_flags,
    local_flags,
    nv,
    storage_length,
    chunk_capacity,
    token_stride,
    layout,
    save_history=True,
    save_cache=True,
):
    for active in tla.range(active_batches.to(tla.Int32)):
        ai = i64(active)
        prefix = meta[prefix2_base + ai]
        stop = meta[prefix2_base + ai + i64(1)]
        lo = start
        hi = end
        if lo < prefix:
            lo = prefix
        if hi > stop:
            hi = stop
        logical_batch = meta[batch_base + ai]
        storage_batch = meta[storage_base + ai]
        token_start = meta[token_base + ai]
        length = meta[length_base + ai]
        group = meta[group_base + ai]
        count = meta[count_base + ai]
        total = (length + i64(63)) // i64(64)
        last = i64(64)
        if group + count == total:
            last = length - (total - i64(1)) * i64(64)
        for task in tla.range(lo.to(tla.Int32), hi.to(tla.Int32)):
            item = i64(task) - prefix
            head = item // i64(128 // dv)
            dv_idx = (item % i64(128 // dv)).to(tla.Int32)
            linear = meta[prefix1_base + ai] + head * count
            vo = token_offset(
                storage_batch,
                head,
                token_start + group * i64(64),
                nv,
                storage_length,
                layout,
            )
            chunk_slot = logical_batch * i64(chunk_capacity) + group
            if tla.const_expr(layout == "TND"):
                chunk_slot = token_start // i64(64) + logical_batch + group
            hc = (chunk_slot * i64(nv) + head) * i64(16384)
            mc = (chunk_slot * i64(nv) + head) * i64(4096)
            si = (logical_batch * i64(nv) + head) * i64(16384)
            source = initial.ptr + si
            if group > i64(0):
                source = history.ptr + hc
            destination = final.ptr + si
            if group + count < total:
                destination = history.ptr + hc + count * i64(nv * 16384)
            stage_two_group(
                ub,
                l1,
                a,
                b,
                c,
                qs.ptr + linear * i64(8192),
                w.ptr + vo,
                kg.ptr + vo,
                aqk.ptr + mc,
                u.ptr + vo,
                source,
                gamma.ptr + linear * i64(128),
                out.ptr + vo,
                vnew.ptr + vo,
                history.ptr + hc,
                destination,
                count.to(tla.Int32),
                last.to(tla.Int32),
                dv,
                dv_idx,
                token_stride,
                save_history,
                phase_flags,
                token_stride,
                nv * 4096,
                nv * 16384,
                local_flags,
                save_cache,
            )


class StagedMetadataKernel:
    def __init__(
        self,
        batch,
        length,
        nq,
        nv,
        layout,
        mode=1,
        negative=False,
        lower=-0.1,
        scale=128**-0.5,
        normalize=True,
        beta_sigmoid=True,
        pipeline=False,
        save_matrices=True,
        save_cache=True,
        save_history=True,
        save_norm=True,
        save_beta=True,
    ):
        if layout not in ("BNSD", "BSND", "TND"):
            raise ValueError("invalid layout")
        self.batch, self.length, self.nq, self.nv, self.layout = (
            batch,
            length,
            nq,
            nv,
            layout,
        )
        self.mode, self.negative, self.lower, self.scale = mode, negative, lower, scale
        self.normalize, self.beta_sigmoid = normalize, beta_sigmoid
        self.pipeline = pipeline
        self.save_matrices, self.save_cache = save_matrices, save_cache
        self.save_history, self.save_norm, self.save_beta = (
            save_history,
            save_norm,
            save_beta,
        )
        self.shared = mode == 1 and not negative and abs(scale) <= 0.125 and lower >= -5
        self.chunk_capacity = (length + 63) // 64
        self.token_stride = 128 if layout == "BNSD" else nv * 128
        self.q_stride = 128 if layout == "BNSD" else nq * 128
        self.beta_stride = 1 if layout == "BNSD" else nv
        self.norm_stride = nq * 128 if layout == "TND" else 128
        self.rstd_stride = nq if layout == "TND" else 1
        self.rstd_k_offset = (1 if layout == "TND" else batch) * nq * length

    @tla.jit
    def __call__(
        self,
        q,
        k,
        v,
        g,
        beta,
        alpha,
        bias,
        exchange,
        identity16,
        initial,
        meta,
        aqk,
        akk,
        w,
        u,
        qg,
        kg,
        qs,
        gamma,
        qn,
        kn,
        rstd,
        gc,
        bact,
        out,
        vnew,
        history,
        final,
    ):
        self.run(
            q,
            k,
            v,
            g,
            beta,
            alpha,
            bias,
            exchange,
            identity16,
            initial,
            meta,
            aqk,
            akk,
            w,
            u,
            qg,
            kg,
            qs,
            gamma,
            qn,
            kn,
            rstd,
            gc,
            bact,
            out,
            vnew,
            history,
            final,
        )

    @tla.kernel
    def run(
        self,
        q,
        k,
        v,
        g,
        beta,
        alpha,
        bias,
        exchange,
        identity16,
        initial,
        meta,
        aqk,
        akk,
        w,
        u,
        qg,
        kg,
        qs,
        gamma,
        qn,
        kn,
        rstd,
        gc,
        bact,
        out,
        vnew,
        history,
        final,
    ):
        ub = tla.allocate(61440, tla.Float32, tla.AddressSpace.ub, 256)
        l1 = tla.allocate(131072, tla.BFloat16, tla.AddressSpace.l1, 512)
        a = tla.allocate(32768, tla.BFloat16, tla.AddressSpace.l0a, 512)
        b = tla.allocate(32768, tla.BFloat16, tla.AddressSpace.l0b, 512)
        c = tla.allocate(32768, tla.Float32, tla.AddressSpace.l0c, 1024)
        kb = tla.cross_flag("phase_kb", mode=2)
        vb = tla.cross_flag("phase_vb", mode=2)
        scores_in = tla.cross_flag("phase_scores_in", mode=2)
        scores_out = tla.cross_flag("phase_scores_out", mode=2)
        nt = tla.cross_flag("phase_nt", mode=2)
        av_ready = tla.cross_flag("boundary_aiv", mode=2)
        ac_all = tla.cross_flag("boundary_all", mode=0)
        ac_release = tla.cross_flag("boundary_release", mode=2)
        state_flags = state_local_flags()
        phase1 = (kb, vb, scores_in, scores_out, nt)
        phase2 = (scores_in, scores_out, nt, kb)
        block = i64(tla.arch.block_idx())
        rounds = meta[i64(4)].to(tla.Int32)
        begin = tla.Int32(0)
        end_round = rounds
        for round_idx in tla.range(begin, end_round):
            record = meta[i64(11) + i64(round_idx)]
            active_batches = meta[record + i64(1)]
            batch_base = record + i64(2)
            storage_base = batch_base + active_batches
            token_base = storage_base + active_batches
            length_base = token_base + active_batches
            group_base = length_base + active_batches + i64(1)
            count_base = group_base + active_batches
            prefix1_base = count_base + active_batches
            ranges1 = prefix1_base + active_batches + i64(1)
            split_base = ranges1 + i64(64)
            prefix2_base = split_base + active_batches
            ranges2 = prefix2_base + active_batches + i64(1)
            start1 = meta[ranges1 + block * i64(2)]
            end1 = meta[ranges1 + block * i64(2) + i64(1)]
            for active in tla.range(active_batches.to(tla.Int32)):
                ai = i64(active)
                prefix = meta[prefix1_base + ai]
                stop = meta[prefix1_base + ai + i64(1)]
                lo = start1
                hi = end1
                if lo < prefix:
                    lo = prefix
                if hi > stop:
                    hi = stop
                logical_batch = meta[batch_base + ai]
                storage_batch = meta[storage_base + ai]
                token_start = meta[token_base + ai]
                length = meta[length_base + ai]
                group = meta[group_base + ai]
                count = meta[count_base + ai]
                if tla.const_expr(self.pipeline):
                    stage_one_pipeline(
                        ub,
                        l1,
                        a,
                        b,
                        c,
                        q,
                        k,
                        v,
                        g,
                        beta,
                        alpha,
                        bias,
                        flat(exchange.ptr + block * i64(256), 256),
                        identity16,
                        aqk,
                        akk,
                        w,
                        u,
                        qg,
                        kg,
                        qs,
                        gamma,
                        qn,
                        kn,
                        rstd,
                        gc,
                        bact,
                        lo,
                        hi,
                        prefix,
                        group,
                        count,
                        logical_batch,
                        storage_batch,
                        token_start,
                        length,
                        self.nq,
                        self.nv,
                        self.length,
                        self.chunk_capacity,
                        self.layout,
                        self.mode,
                        self.lower,
                        self.negative,
                        self.normalize,
                        self.scale,
                        self.shared,
                        phase1,
                        self.q_stride,
                        self.token_stride,
                        self.beta_stride,
                        self.norm_stride,
                        self.rstd_k_offset,
                        self.rstd_stride,
                        self.beta_sigmoid,
                        self.save_matrices,
                        self.save_cache,
                        self.save_norm,
                        self.save_beta,
                    )
                if tla.const_expr(not self.pipeline):
                    for task in tla.range(lo.to(tla.Int32), hi.to(tla.Int32)):
                        local = i64(task) - prefix
                        head = local // count
                        chunk = group + local % count
                        qhead = head // i64(self.nv // self.nq)
                        valid = i64(64)
                        if (chunk + i64(1)) * i64(64) > length:
                            valid = length - chunk * i64(64)
                        token = token_start + chunk * i64(64)
                        qo = token_offset(
                            storage_batch,
                            qhead,
                            token,
                            self.nq,
                            self.length,
                            self.layout,
                        )
                        vo = token_offset(
                            storage_batch,
                            head,
                            token,
                            self.nv,
                            self.length,
                            self.layout,
                        )
                        bo = vo // i64(128)
                        no = (
                            (storage_batch * i64(self.nq) + qhead) * i64(self.length)
                            + token
                        ) * i64(128)
                        if tla.const_expr(self.layout == "TND"):
                            no = (token * i64(self.nq) + qhead) * i64(128)
                        ro = no // i64(128)
                        chunk_slot = logical_batch * i64(self.chunk_capacity) + chunk
                        if tla.const_expr(self.layout == "TND"):
                            chunk_slot = token_start // i64(64) + logical_batch + chunk
                        co = chunk_slot * i64(self.nv) + head
                        workspace = prefix + local
                        stage_one_chunk(
                            ub,
                            l1,
                            a,
                            b,
                            c,
                            flat(q.ptr + qo, 1),
                            flat(k.ptr + qo, 1),
                            flat(v.ptr + vo, 1),
                            flat(g.ptr + vo, 1),
                            flat(beta.ptr + bo, 1),
                            flat(alpha.ptr + head, 1),
                            flat(bias.ptr + head * i64(128), 128),
                            flat(exchange.ptr + block * i64(256), 256),
                            identity16,
                            flat(aqk.ptr + co * i64(4096), 4096),
                            flat(akk.ptr + co * i64(4096), 4096),
                            flat(w.ptr + vo, 1),
                            flat(u.ptr + vo, 1),
                            flat(qg.ptr + vo, 1),
                            flat(kg.ptr + vo, 1),
                            flat(qs.ptr + workspace * i64(8192), 8192),
                            flat(gamma.ptr + workspace * i64(128), 128),
                            flat(qn.ptr + no, 1),
                            flat(kn.ptr + no, 1),
                            flat(rstd.ptr + ro, 1),
                            flat(gc.ptr + co * i64(8192), 8192),
                            flat(
                                bact.ptr
                                + (storage_batch * i64(self.length) + token)
                                * i64(self.nv)
                                + head,
                                1,
                            ),
                            valid.to(tla.Int32),
                            self.mode,
                            self.lower,
                            self.negative,
                            self.normalize,
                            self.scale,
                            self.shared,
                            phase1,
                            self.q_stride,
                            self.token_stride,
                            self.beta_stride,
                            self.norm_stride,
                            self.nv,
                            valid.to(tla.Int32),
                            head % i64(self.nv // self.nq) == i64(0),
                            self.rstd_k_offset,
                            self.rstd_stride,
                            self.beta_sigmoid,
                        )
            stage_boundary(av_ready, ac_all, ac_release)
            start2 = meta[ranges2 + block * i64(2)]
            end2 = meta[ranges2 + block * i64(2) + i64(1)]
            splits = meta[split_base]
            if splits == i64(1):
                state_items(
                    128,
                    ub,
                    l1,
                    a,
                    b,
                    c,
                    qs,
                    w,
                    kg,
                    aqk,
                    u,
                    initial,
                    gamma,
                    out,
                    vnew,
                    history,
                    final,
                    meta,
                    active_batches,
                    batch_base,
                    storage_base,
                    token_base,
                    length_base,
                    group_base,
                    count_base,
                    prefix1_base,
                    prefix2_base,
                    start2,
                    end2,
                    phase2,
                    state_flags,
                    self.nv,
                    self.length,
                    self.chunk_capacity,
                    self.token_stride,
                    self.layout,
                    self.save_history,
                    self.save_cache,
                )
            elif splits == i64(2):
                state_items(
                    64,
                    ub,
                    l1,
                    a,
                    b,
                    c,
                    qs,
                    w,
                    kg,
                    aqk,
                    u,
                    initial,
                    gamma,
                    out,
                    vnew,
                    history,
                    final,
                    meta,
                    active_batches,
                    batch_base,
                    storage_base,
                    token_base,
                    length_base,
                    group_base,
                    count_base,
                    prefix1_base,
                    prefix2_base,
                    start2,
                    end2,
                    phase2,
                    state_flags,
                    self.nv,
                    self.length,
                    self.chunk_capacity,
                    self.token_stride,
                    self.layout,
                    self.save_history,
                    self.save_cache,
                )
            elif splits == i64(4):
                state_items(
                    32,
                    ub,
                    l1,
                    a,
                    b,
                    c,
                    qs,
                    w,
                    kg,
                    aqk,
                    u,
                    initial,
                    gamma,
                    out,
                    vnew,
                    history,
                    final,
                    meta,
                    active_batches,
                    batch_base,
                    storage_base,
                    token_base,
                    length_base,
                    group_base,
                    count_base,
                    prefix1_base,
                    prefix2_base,
                    start2,
                    end2,
                    phase2,
                    state_flags,
                    self.nv,
                    self.length,
                    self.chunk_capacity,
                    self.token_stride,
                    self.layout,
                    self.save_history,
                    self.save_cache,
                )
            else:
                state_items(
                    16,
                    ub,
                    l1,
                    a,
                    b,
                    c,
                    qs,
                    w,
                    kg,
                    aqk,
                    u,
                    initial,
                    gamma,
                    out,
                    vnew,
                    history,
                    final,
                    meta,
                    active_batches,
                    batch_base,
                    storage_base,
                    token_base,
                    length_base,
                    group_base,
                    count_base,
                    prefix1_base,
                    prefix2_base,
                    start2,
                    end2,
                    phase2,
                    state_flags,
                    self.nv,
                    self.length,
                    self.chunk_capacity,
                    self.token_stride,
                    self.layout,
                    self.save_history,
                    self.save_cache,
                )
            if round_idx + 1 < rounds:
                stage_boundary(av_ready, ac_all, ac_release)


# -----------------------------------------------------------------------------
# Persistent inverse.
# -----------------------------------------------------------------------------


@tla.jit
def cast_pack_fp16(value):
    converted = value.to(
        tla.Float16,
        tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO,
            sat_mode=tla.params.SatMode.NOSAT,
            round_mode=tla.params.RoundMode.CAST_RINT,
        ),
    )
    packed, unused = tla.deinterleave(converted, converted)
    return packed


@tla.jit
def solve_prefix(negative, prefix, destination, aiv):
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        lane = tla.arange(0, dtype=tla.Int32)
        col = tla.bitwise_and(lane, tla.full(15, dtype=tla.Int32))
        block = tla.bitwise_and(lane, tla.full(16, dtype=tla.Int32))
        base = block * 64 + block + aiv * 32
        mask, unused = tla.update_mask(32, dtype=tla.Float32)
        low, unused = tla.update_mask(32, dtype=tla.Float16)
        zero = tla.full(0.0, dtype=tla.Float32)
        one = tla.full(1.0, dtype=tla.Float32)
        r0 = tla.where(tla.cmp(col, 0, mode="eq"), one, zero)
        r1 = tla.where(tla.cmp(col, 1, mode="eq"), one, zero)
        r2 = tla.where(tla.cmp(col, 2, mode="eq"), one, zero)
        r3 = tla.where(tla.cmp(col, 3, mode="eq"), one, zero)
        r4 = tla.where(tla.cmp(col, 4, mode="eq"), one, zero)
        r5 = tla.where(tla.cmp(col, 5, mode="eq"), one, zero)
        r6 = tla.where(tla.cmp(col, 6, mode="eq"), one, zero)
        r7 = tla.where(tla.cmp(col, 7, mode="eq"), one, zero)
        r8 = tla.where(tla.cmp(col, 8, mode="eq"), one, zero)
        r9 = tla.where(tla.cmp(col, 9, mode="eq"), one, zero)
        r10 = tla.where(tla.cmp(col, 10, mode="eq"), one, zero)
        at(prefix, 0, 32).store(r0, mask=mask)
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(0, 0)).store(
            cast_pack_fp16(r0), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 64, mask=mask)
        r1 = x * r0 + r1
        at(prefix, 32, 32).store(r1, mask=mask)
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(1, 0)).store(
            cast_pack_fp16(r1), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 128, mask=mask)
        r2 = x * r0 + r2
        x = tla.gather(at(negative, 0, 64), base + 129, mask=mask)
        r2 = x * r1 + r2
        at(prefix, 64, 32).store(r2, mask=mask)
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(2, 0)).store(
            cast_pack_fp16(r2), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 192, mask=mask)
        r3 = x * r0 + r3
        x = tla.gather(at(negative, 0, 64), base + 193, mask=mask)
        r3 = x * r1 + r3
        x = tla.gather(at(negative, 0, 64), base + 194, mask=mask)
        r3 = x * r2 + r3
        at(prefix, 96, 32).store(r3, mask=mask)
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(3, 0)).store(
            cast_pack_fp16(r3), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 256, mask=mask)
        r4 = x * r0 + r4
        x = tla.gather(at(negative, 0, 64), base + 257, mask=mask)
        r4 = x * r1 + r4
        x = tla.gather(at(negative, 0, 64), base + 258, mask=mask)
        r4 = x * r2 + r4
        x = tla.gather(at(negative, 0, 64), base + 259, mask=mask)
        r4 = x * r3 + r4
        at(prefix, 128, 32).store(r4, mask=mask)
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(4, 0)).store(
            cast_pack_fp16(r4), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 320, mask=mask)
        r5 = x * r0 + r5
        x = tla.gather(at(negative, 0, 64), base + 321, mask=mask)
        r5 = x * r1 + r5
        x = tla.gather(at(negative, 0, 64), base + 322, mask=mask)
        r5 = x * r2 + r5
        x = tla.gather(at(negative, 0, 64), base + 323, mask=mask)
        r5 = x * r3 + r5
        x = tla.gather(at(negative, 0, 64), base + 324, mask=mask)
        r5 = x * r4 + r5
        at(prefix, 160, 32).store(r5, mask=mask)
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(5, 0)).store(
            cast_pack_fp16(r5), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 384, mask=mask)
        r6 = x * r0 + r6
        x = tla.gather(at(negative, 0, 64), base + 385, mask=mask)
        r6 = x * r1 + r6
        x = tla.gather(at(negative, 0, 64), base + 386, mask=mask)
        r6 = x * r2 + r6
        x = tla.gather(at(negative, 0, 64), base + 387, mask=mask)
        r6 = x * r3 + r6
        x = tla.gather(at(negative, 0, 64), base + 388, mask=mask)
        r6 = x * r4 + r6
        x = tla.gather(at(negative, 0, 64), base + 389, mask=mask)
        r6 = x * r5 + r6
        at(prefix, 192, 32).store(r6, mask=mask)
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(6, 0)).store(
            cast_pack_fp16(r6), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 448, mask=mask)
        r7 = x * r0 + r7
        x = tla.gather(at(negative, 0, 64), base + 449, mask=mask)
        r7 = x * r1 + r7
        x = tla.gather(at(negative, 0, 64), base + 450, mask=mask)
        r7 = x * r2 + r7
        x = tla.gather(at(negative, 0, 64), base + 451, mask=mask)
        r7 = x * r3 + r7
        x = tla.gather(at(negative, 0, 64), base + 452, mask=mask)
        r7 = x * r4 + r7
        x = tla.gather(at(negative, 0, 64), base + 453, mask=mask)
        r7 = x * r5 + r7
        x = tla.gather(at(negative, 0, 64), base + 454, mask=mask)
        r7 = x * r6 + r7
        at(prefix, 224, 32).store(r7, mask=mask)
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(7, 0)).store(
            cast_pack_fp16(r7), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 512, mask=mask)
        r8 = x * r0 + r8
        x = tla.gather(at(negative, 0, 64), base + 513, mask=mask)
        r8 = x * r1 + r8
        x = tla.gather(at(negative, 0, 64), base + 514, mask=mask)
        r8 = x * r2 + r8
        x = tla.gather(at(negative, 0, 64), base + 515, mask=mask)
        r8 = x * r3 + r8
        x = tla.gather(at(negative, 0, 64), base + 516, mask=mask)
        r8 = x * r4 + r8
        x = tla.gather(at(negative, 0, 64), base + 517, mask=mask)
        r8 = x * r5 + r8
        x = tla.gather(at(negative, 0, 64), base + 518, mask=mask)
        r8 = x * r6 + r8
        x = tla.gather(at(negative, 0, 64), base + 519, mask=mask)
        r8 = x * r7 + r8
        at(prefix, 256, 32).store(r8, mask=mask)
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(8, 0)).store(
            cast_pack_fp16(r8), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 576, mask=mask)
        r9 = x * r0 + r9
        x = tla.gather(at(negative, 0, 64), base + 577, mask=mask)
        r9 = x * r1 + r9
        x = tla.gather(at(negative, 0, 64), base + 578, mask=mask)
        r9 = x * r2 + r9
        x = tla.gather(at(negative, 0, 64), base + 579, mask=mask)
        r9 = x * r3 + r9
        x = tla.gather(at(negative, 0, 64), base + 580, mask=mask)
        r9 = x * r4 + r9
        x = tla.gather(at(negative, 0, 64), base + 581, mask=mask)
        r9 = x * r5 + r9
        x = tla.gather(at(negative, 0, 64), base + 582, mask=mask)
        r9 = x * r6 + r9
        x = tla.gather(at(negative, 0, 64), base + 583, mask=mask)
        r9 = x * r7 + r9
        x = tla.gather(at(negative, 0, 64), base + 584, mask=mask)
        r9 = x * r8 + r9
        at(prefix, 288, 32).store(r9, mask=mask)
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(9, 0)).store(
            cast_pack_fp16(r9), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 640, mask=mask)
        r10 = x * r0 + r10
        x = tla.gather(at(negative, 0, 64), base + 641, mask=mask)
        r10 = x * r1 + r10
        x = tla.gather(at(negative, 0, 64), base + 642, mask=mask)
        r10 = x * r2 + r10
        x = tla.gather(at(negative, 0, 64), base + 643, mask=mask)
        r10 = x * r3 + r10
        x = tla.gather(at(negative, 0, 64), base + 644, mask=mask)
        r10 = x * r4 + r10
        x = tla.gather(at(negative, 0, 64), base + 645, mask=mask)
        r10 = x * r5 + r10
        x = tla.gather(at(negative, 0, 64), base + 646, mask=mask)
        r10 = x * r6 + r10
        x = tla.gather(at(negative, 0, 64), base + 647, mask=mask)
        r10 = x * r7 + r10
        x = tla.gather(at(negative, 0, 64), base + 648, mask=mask)
        r10 = x * r8 + r10
        x = tla.gather(at(negative, 0, 64), base + 649, mask=mask)
        r10 = x * r9 + r10
        at(prefix, 320, 32).store(r10, mask=mask)
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(10, 0)).store(
            cast_pack_fp16(r10), tla.params.BlockStoreParams(block_stride=49), mask=low
        )


@tla.jit
def solve_tail(negative, prefix, destination, aiv):
    prefix_rows = tla.make_tensor(
        prefix.ptr,
        tla.make_layout(
            tla.make_shape(11, 64), tla.make_stride(32, 1), layoutTag=tla.arch.RowMajor
        ),
    )
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        lane = tla.arange(0, dtype=tla.Int32)
        col = tla.bitwise_and(lane, tla.full(15, dtype=tla.Int32))
        block = tla.bitwise_and(lane, tla.full(16, dtype=tla.Int32))
        base = block * 64 + block + aiv * 32
        mask, unused = tla.update_mask(32, dtype=tla.Float32)
        low, unused = tla.update_mask(32, dtype=tla.Float16)
        zero = tla.full(0.0, dtype=tla.Float32)
        one = tla.full(1.0, dtype=tla.Float32)
        r11 = tla.where(tla.cmp(col, 11, mode="eq"), one, zero)
        r12 = tla.where(tla.cmp(col, 12, mode="eq"), one, zero)
        r13 = tla.where(tla.cmp(col, 13, mode="eq"), one, zero)
        r14 = tla.where(tla.cmp(col, 14, mode="eq"), one, zero)
        r15 = tla.where(tla.cmp(col, 15, mode="eq"), one, zero)
        for previous in tla.range(11):
            prior = tla.tile_view(
                prefix_rows, tla.make_shape(1, 64), tla.make_coord(previous, 0)
            ).load()
            prefix_x = tla.gather(at(negative, 0, 64), base + 704 + previous, mask=mask)
            r11 = prefix_x * prior + r11
            prefix_x = tla.gather(at(negative, 0, 64), base + 768 + previous, mask=mask)
            r12 = prefix_x * prior + r12
            prefix_x = tla.gather(at(negative, 0, 64), base + 832 + previous, mask=mask)
            r13 = prefix_x * prior + r13
            prefix_x = tla.gather(at(negative, 0, 64), base + 896 + previous, mask=mask)
            r14 = prefix_x * prior + r14
            prefix_x = tla.gather(at(negative, 0, 64), base + 960 + previous, mask=mask)
            r15 = prefix_x * prior + r15
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(11, 0)).store(
            cast_pack_fp16(r11), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 779, mask=mask)
        r12 = x * r11 + r12
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(12, 0)).store(
            cast_pack_fp16(r12), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 843, mask=mask)
        r13 = x * r11 + r13
        x = tla.gather(at(negative, 0, 64), base + 844, mask=mask)
        r13 = x * r12 + r13
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(13, 0)).store(
            cast_pack_fp16(r13), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 907, mask=mask)
        r14 = x * r11 + r14
        x = tla.gather(at(negative, 0, 64), base + 908, mask=mask)
        r14 = x * r12 + r14
        x = tla.gather(at(negative, 0, 64), base + 909, mask=mask)
        r14 = x * r13 + r14
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(14, 0)).store(
            cast_pack_fp16(r14), tla.params.BlockStoreParams(block_stride=49), mask=low
        )
        x = tla.gather(at(negative, 0, 64), base + 971, mask=mask)
        r15 = x * r11 + r15
        x = tla.gather(at(negative, 0, 64), base + 972, mask=mask)
        r15 = x * r12 + r15
        x = tla.gather(at(negative, 0, 64), base + 973, mask=mask)
        r15 = x * r13 + r15
        x = tla.gather(at(negative, 0, 64), base + 974, mask=mask)
        r15 = x * r14 + r15
        tla.tile_view(destination, tla.make_shape(1, 64), tla.make_coord(15, 0)).store(
            cast_pack_fp16(r15), tla.params.BlockStoreParams(block_stride=49), mask=low
        )


@tla.jit
def persistent_diagonal_inverse(negative, prefix, nz, aiv, initialize=True):
    if initialize:
        with tla.vec.func(mode="simd"):
            low, unused = tla.update_mask(64, dtype=tla.Float16)
            zero = cast_pack_fp16(tla.full(0.0, dtype=tla.Float32))
            for row in tla.range(32):
                tla.tile_view(nz, tla.make_shape(1, 64), tla.make_coord(row, 0)).store(
                    zero, tla.params.BlockStoreParams(block_stride=33), mask=low
                )
    # Each AIV owns 32 rows, with two diagonal 16x16 tiles at its global columns.
    diagonal = padded_nz(nz.ptr + aiv * 1056, 32)
    solve_prefix(negative, prefix, diagonal, aiv)
    solve_tail(negative, prefix, diagonal, aiv)


@tla.jit
def compose_persistent_inverse(
    negative,
    diagonal,
    identity,
    temporary,
    inverse,
    zero,
    a0,
    a1,
    b0,
    b1,
    ci,
    ct,
    ready0,
    ready1,
    free0,
    free1,
    fix_ready,
    acc_free,
):
    """Source four full64 MMADs: odd/even16 composition, then lower32 update."""
    _mutex_free0 = tla.mutex("compose_persistent_inverse_4386_free0", id=0)
    _mutex_ready0 = tla.mutex("compose_persistent_inverse_4386_ready0", id=1)
    load_selected_pair(negative, diagonal, a1, b1, zero, ready1, free1, 3, 1)
    inverse_mmad(ct, a1, b1, ready1, free1, True, 3)
    inverse_fix(temporary, ct, fix_ready, acc_free)
    mutex_event(_mutex_free0, tla.pipes.MTE1)
    tla.copy(a0, diagonal)
    tla.copy(b0, identity)
    copy_selected_fractals(b0, temporary, 3)
    mutex_event(_mutex_ready0, tla.pipes.MTE1)
    inverse_mmad(ci, a0, b0, ready0, free0, True, 3)
    inverse_fix(inverse, ci, fix_ready, acc_free)
    load_selected_pair(negative, inverse, a1, b1, zero, ready1, free1, 0, 4)
    mutex_event(_mutex_free0, tla.pipes.MTE1)
    tla.copy(a0, inverse)
    tla.copy(b0, zero)
    inverse_mmad(ct, a1, b1, ready1, free1, True, 3)
    inverse_fix(temporary, ct, fix_ready, acc_free)
    copy_selected_fractals(b0, temporary, 5)
    mutex_event(_mutex_ready0, tla.pipes.MTE1)
    inverse_mmad(ci, a0, b0, ready0, free0, False, 3)


# -----------------------------------------------------------------------------
# Persistent recurrent operands.
# -----------------------------------------------------------------------------


@tla.jit
def recurrent_l1(ptr):
    return tla.make_tensor_like(ptr, row_major(ptr, 256, 128), tla.arch.zN)


@tla.jit
def store_recurrent(value, destination, operand, row, mask):
    tla.tile_view(
        destination, tla.make_shape(1, 128), tla.make_coord(operand * 64 + row, 0)
    ).store(
        cast_and_pack_bf16(value),
        tla.params.BlockStoreParams(block_stride=257),
        mask=mask,
    )


@tla.jit
def pack_recurrent(q, k, v, g, beta, packed, gamma, qg, kg, scale, save_cache):
    destination = padded_nz(packed.ptr, 256)
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        low, remaining = tla.update_mask(64, dtype=tla.BFloat16)
        full, unused = tla.update_mask(64, dtype=tla.Float32)
        last = at(g, 63 * 64, 64).load()
        gamma.store(tla.exp(last), mask=full)
        for group in tla.range(32):
            for lane in tla.range_constexpr(2):
                row = group * 2 + lane
                gate = at(g, row * 64, 64).load()
                decay = tla.exp(gate)
                restore = tla.exp(last - gate)
                qv = load_bf16_f32(at(q, row * 64, 64))
                kv = load_bf16_f32(at(k, row * 64, 64))
                vv = load_bf16_f32(at(v, row * 64, 64))
                bv = broadcast(at(beta, row, 1))
                qg_value = qv * decay
                kg_value = kv * restore
                if tla.const_expr(save_cache):
                    store_bf16(at(qg, row * 64, 64), qg_value)
                    store_bf16(at(kg, row * 64, 64), kg_value)
                store_recurrent(qg_value * scale, destination, 0, row, low)
                store_recurrent(kg_value, destination, 1, row, low)
                # API-062: exact sign mapping to avoid unsupported scaled FIX to L1.
                store_recurrent(((kv * decay) * bv) * (-1.0), destination, 2, row, low)
                store_recurrent(vv * bv, destination, 3, row, low)


# -----------------------------------------------------------------------------
# Persistent input prefetch.
# -----------------------------------------------------------------------------


@tla.jit
def prefetch_half(q, k, qraw, kraw, next_valid, stride, half, free):
    _mutex_free = tla.mutex("prefetch_half_4486_free", id=0)
    with tla.vector():
        aiv = tla.arch.sub_block_idx()
        rows = next_valid - half * 32
        if rows > 32:
            rows = 32
        if rows > 0:
            # Raw Q/K have already been consumed by normalization/panel packing.
            mutex_event(_mutex_free, tla.pipes.VECTOR)
            mutex_event(_mutex_free, tla.pipes.MTE2)
            source = tla.Int64((64 + half * 32) * stride) + tla.Int64(aiv * 64)
            tla.copy(
                row_major(qraw.ptr + half * 2048, rows, 64),
                strided_rows(q.ptr + source, rows, 64, stride),
            )
            tla.copy(
                row_major(kraw.ptr + half * 2048, rows, 64),
                strided_rows(k.ptr + source, rows, 64, stride),
            )


# -----------------------------------------------------------------------------
# Persistent scores.
# -----------------------------------------------------------------------------


@tla.jit
def score_l1(ptr, shared=False):
    return tla.make_tensor_like(
        ptr, row_major(ptr, 224 if shared else 288, 128), tla.arch.zN
    )


@tla.jit
def store_operand(value, destination, tile, row, panel_rows=288):
    low, unused = tla.update_mask(64, dtype=tla.BFloat16)
    tla.tile_view(
        destination, tla.make_shape(1, 128), tla.make_coord(tile * 16 + row, 0)
    ).store(
        cast_and_pack_bf16(value),
        tla.params.BlockStoreParams(block_stride=panel_rows + 1),
        mask=low,
    )


@tla.jit
def pack_persistent_panel(
    q, k, g, packed, panel, scale, raw_q=None, raw_k=None, inverse=None, shared=False
):
    panel_rows = 224 if shared else 288
    destination = padded_nz(packed.ptr, panel_rows)
    input_q = q
    input_k = k
    if tla.const_expr(raw_q is not None):
        input_q = raw_q
    if tla.const_expr(raw_k is not None):
        input_k = raw_k
    with tla.vec.func(mode="simd"):
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        norm_mask, unused = tla.update_mask(128, dtype=tla.BFloat16)
        for half in tla.range_constexpr(2):
            block = panel * 2 + half
            for group in tla.range(8):
                for lane in tla.range_constexpr(2):
                    row = group * 2 + lane
                    logical = block * 16 + row
                    qv = load_bf16_f32(at(input_q, logical * 64, 64))
                    kv = load_bf16_f32(at(input_k, logical * 64, 64))
                    if tla.const_expr(raw_q is not None):
                        if tla.const_expr(inverse is not None):
                            qv = qv * broadcast(at(inverse, logical, 1))
                            kv = kv * broadcast(at(inverse, 64 + logical, 1))
                        qb = cast_bf16(qv)
                        kb = cast_bf16(kv)
                        at(q, logical * 64, 64).store(
                            qb,
                            tla.params.NormalStoreParams(
                                store_dist=tla.params.StoreDist.DIST_PACK_B32
                            ),
                            mask=norm_mask,
                        )
                        at(k, logical * 64, 64).store(
                            kb,
                            tla.params.NormalStoreParams(
                                store_dist=tla.params.StoreDist.DIST_PACK_B32
                            ),
                            mask=norm_mask,
                        )
                        qv = qb.to(
                            tla.Float32,
                            tla.params.CastParams(),
                        )
                        kv = kb.to(
                            tla.Float32,
                            tla.params.CastParams(),
                        )
                    gate = at(g, logical * 64, 64).load()
                    reference = at(
                        g, ((panel * 32 + 16) if shared else block * 16) * 64, 64
                    ).load()
                    decay = tla.exp(gate - reference)
                    store_operand(kv * decay, destination, 2 * block, row, panel_rows)
                    store_operand(
                        (qv * decay) * scale,
                        destination,
                        2 * block + 1,
                        row,
                        panel_rows,
                    )
                    if tla.const_expr(shared):
                        for row_group in tla.range_constexpr(2):
                            if tla.const_expr(block < 2 * (row_group + 1)):
                                right_ref = at(g, (row_group * 32 + 16) * 64, 64).load()
                                store_operand(
                                    kv * tla.exp(right_ref - gate),
                                    destination,
                                    8 + row_group * 2 + block,
                                    row,
                                    panel_rows,
                                )
                    else:
                        for rb in tla.range_constexpr(block, 4):
                            right_ref = at(g, rb * 16 * 64, 64).load()
                            store_operand(
                                kv * tla.exp(right_ref - gate),
                                destination,
                                8 + rb * (rb + 1) // 2 + block,
                                row,
                            )


@tla.jit
def right_transpose(ptr, index, shared=False):
    return tla.make_tensor(
        ptr + (8 + index) * 256,
        tla.make_layout(
            tla.make_shape((16, 8), (16, 1)),
            tla.make_stride((1, 3584 if shared else 4608), (16, 256)),
            origin_shape=tla.make_shape(128, 16),
            layoutTag=tla.arch.nZ,
        ),
    )


@tla.jit
def persistent_score_mmads(
    cache,
    left_ptr,
    right_ptr,
    ck,
    cq,
    loaded,
    kk=None,
    mq=None,
    kk_ready=None,
    mq_ready=None,
    cube_done=None,
    shared=False,
):
    """Source RHS loading batches and next-row LHS prefetch, two L0A banks."""
    _mutex_free0 = tla.mutex("persistent_score_mmads_4639_free0", id=0)
    _mutex_free1 = tla.mutex("persistent_score_mmads_4639_free1", id=1)
    _mutex_loaded = tla.mutex("persistent_score_mmads_4639_loaded", id=2)
    _mutex_ready0 = tla.mutex("persistent_score_mmads_4639_ready0", id=3)
    _mutex_ready1 = tla.mutex("persistent_score_mmads_4639_ready1", id=4)
    _mutex_cube_done = tla.mutex("persistent_score_mmads_4639_cube_done", id=5)
    ready0 = tla.flag("ps_left_ready0", tla.pipes.MTE1, tla.pipes.CUBE)
    ready1 = tla.flag("ps_left_ready1", tla.pipes.MTE1, tla.pipes.CUBE)
    free0 = tla.flag("ps_left_free0", tla.pipes.CUBE, tla.pipes.MTE1)
    free1 = tla.flag("ps_left_free1", tla.pipes.CUBE, tla.pipes.MTE1)
    mutex_event(_mutex_free0, tla.pipes.CUBE)
    mutex_event(_mutex_free1, tla.pipes.CUBE)
    for kind in tla.range_constexpr(2):
        # This CUBE->MTE1 edge also releases the previous chunk's output
        # operands at bp[0:8192]. All RHS preloads must follow it.
        mutex_event(_mutex_free0, tla.pipes.MTE1)
        if tla.const_expr(shared and kind == 0):
            for j in tla.range_constexpr(6):
                right = tla.make_tensor_like(
                    right_ptr + j * 2048, row_major(right_ptr, 128, 16), tla.arch.nZ
                )
                tla.copy(right, right_transpose(cache.ptr, j, True))
            mutex_event(_mutex_loaded, tla.pipes.MTE1)
            mutex_event(_mutex_loaded, tla.pipes.CUBE)
        left0 = tla.make_tensor_like(
            left_ptr, row_major(left_ptr, 16, 128), tla.arch.zN
        )
        tla.copy(
            left0,
            tla.tile_view(cache, tla.make_shape(16, 128), tla.make_coord(kind, 0)),
        )
        mutex_event(_mutex_ready0, tla.pipes.MTE1)
        for row in tla.range_constexpr(4):
            left = tla.make_tensor_like(
                left_ptr + (row % 2) * 2048, row_major(left_ptr, 16, 128), tla.arch.zN
            )
            if tla.const_expr(row % 2 == 0):
                mutex_event(_mutex_ready0, tla.pipes.CUBE)
            else:
                mutex_event(_mutex_ready1, tla.pipes.CUBE)
            for col in tla.range_constexpr(row + 1):
                index = (
                    (col if row < 2 else 2 + col)
                    if shared
                    else row * (row + 1) // 2 + col
                )
                if tla.const_expr(kind == 0 and not shared):
                    if tla.const_expr(
                        index == 0
                        or index == 2
                        or index == 4
                        or index == 5
                        or index == 6
                    ):
                        count = 1
                        if tla.const_expr(index == 0 or index == 2):
                            count = 2
                        if tla.const_expr(index == 6):
                            count = 4
                        for j in tla.range_constexpr(index, index + count):
                            right = tla.make_tensor_like(
                                right_ptr + j * 2048,
                                row_major(right_ptr, 128, 16),
                                tla.arch.nZ,
                            )
                            tla.copy(right, right_transpose(cache.ptr, j))
                        mutex_event(_mutex_loaded, tla.pipes.MTE1)
                        mutex_event(_mutex_loaded, tla.pipes.CUBE)
                right = tla.make_tensor_like(
                    right_ptr + index * 2048, row_major(right_ptr, 128, 16), tla.arch.nZ
                )
                result = ck
                if tla.const_expr(kind == 1):
                    result = cq
                acc = tla.make_tensor_like(
                    result + row * 256 + col * 1024,
                    row_major(result, 16, 16),
                    tla.arch.L0Clayout,
                )
                tla.mmad(acc, left, right, init_c=True)
                if tla.const_expr(row < 3 and col == 0):
                    if tla.const_expr(row % 2 == 0):
                        mutex_event(_mutex_free1, tla.pipes.MTE1)
                    else:
                        mutex_event(_mutex_free0, tla.pipes.MTE1)
                    next_left = tla.make_tensor_like(
                        left_ptr + ((row + 1) % 2) * 2048,
                        row_major(left_ptr, 16, 128),
                        tla.arch.zN,
                    )
                    tla.copy(
                        next_left,
                        tla.tile_view(
                            cache,
                            tla.make_shape(16, 128),
                            tla.make_coord(2 * (row + 1) + kind, 0),
                        ),
                    )
                    if tla.const_expr(row % 2 == 0):
                        mutex_event(_mutex_ready1, tla.pipes.MTE1)
                    else:
                        mutex_event(_mutex_ready0, tla.pipes.MTE1)
            if tla.const_expr(row % 2 == 0):
                mutex_event(_mutex_free0, tla.pipes.CUBE)
            else:
                mutex_event(_mutex_free1, tla.pipes.CUBE)
        if tla.const_expr(kk is not None):
            result = ck
            output = kk
            notification = kk_ready
            if tla.const_expr(kind == 1):
                result = cq
                output = mq
                notification = mq_ready
            mutex_event(_mutex_cube_done, tla.pipes.CUBE)
            mutex_event(_mutex_cube_done, tla.pipes.FIX)
            tla.copy(
                row_major(output.ptr, 64, 64),
                tla.make_tensor_like(
                    result, row_major(result, 64, 64), tla.arch.L0Clayout
                ),
                tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
            )
            tla.cross_core_set_flag(notification, tla.pipes.FIX)
    mutex_event(_mutex_free0, tla.pipes.MTE1)
    mutex_event(_mutex_free1, tla.pipes.MTE1)


# -----------------------------------------------------------------------------
# Persistent Cube recurrence.
# -----------------------------------------------------------------------------


@tla.jit
def persistent_products_and_state(
    resident,
    kbeta,
    vbeta,
    qs,
    kg,
    aqk,
    state,
    negative_w,
    vnew,
    out_w,
    out_u,
    out_vnew,
    out,
    delta,
    ap,
    bp,
    cp,
    valid,
    stride,
    resident_ready,
    k_ready,
    v_ready,
    state_ready,
    recurrent_ready,
    delta_ready,
    loaded,
    cube_done,
    save_cache=True,
    inputs_ready=False,
    output_overlap=False,
    first_chunk=True,
    last_chunk=True,
):
    _mutex_resident_ready = tla.mutex(
        "persistent_products_and_state_4782_resident_ready", id=0
    )
    _mutex_loaded = tla.mutex("persistent_products_and_state_4782_loaded", id=1)
    _mutex_b_free = tla.mutex("persistent_products_and_state_4782_b_free", id=2)
    _mutex_cube_done = tla.mutex("persistent_products_and_state_4782_cube_done", id=3)
    _mutex_w_ready = tla.mutex("persistent_products_and_state_4782_w_ready", id=4)
    _mutex_output_read = tla.mutex(
        "persistent_products_and_state_4782_output_read", id=5
    )
    _mutex_output_free = tla.mutex(
        "persistent_products_and_state_4782_output_free", id=6
    )
    _mutex_vnew_ready = tla.mutex("persistent_products_and_state_4782_vnew_ready", id=7)
    _mutex_delta_read = tla.mutex("persistent_products_and_state_4782_delta_read", id=8)
    w_ready = tla.flag("pr_w_ready", tla.pipes.FIX, tla.pipes.MTE1)
    vnew_ready = tla.flag("pr_vnew_ready", tla.pipes.FIX, tla.pipes.MTE1)
    output_read = tla.flag("pr_output_read", tla.pipes.FIX, tla.pipes.CUBE)
    output_done = tla.flag("pr_output_done", tla.pipes.CUBE, tla.pipes.FIX)
    b_free = tla.flag("pr_b_free", tla.pipes.CUBE, tla.pipes.MTE1)
    output_free = tla.flag("pr_output_free", tla.pipes.FIX, tla.pipes.CUBE)
    delta_read = tla.flag("pr_delta_read", tla.pipes.FIX, tla.pipes.CUBE)
    inverse_a = tla.make_tensor_like(ap + 12288, row_major(ap, 64, 64), tla.arch.zN)
    beta_b = tla.make_tensor_like(bp, row_major(bp, 64, 128), tla.arch.nZ)
    cw = tla.make_tensor_like(cp, row_major(cp, 64, 128), tla.arch.L0Clayout)
    cu = tla.make_tensor_like(cp + 8192, row_major(cp, 64, 128), tla.arch.L0Clayout)
    co = tla.make_tensor_like(cp + 16384, row_major(cp, 64, 128), tla.arch.L0Clayout)
    mutex_event(_mutex_resident_ready, tla.pipes.MTE1)
    tla.copy(inverse_a, resident)
    if tla.const_expr(not inputs_ready):
        tla.cross_core_wait_flag(k_ready, tla.pipes.MTE1)
    tla.copy(beta_b, kbeta)
    mutex_event(_mutex_loaded, tla.pipes.MTE1)
    mutex_event(_mutex_loaded, tla.pipes.CUBE)
    tla.mmad(cw, inverse_a, beta_b, init_c=True)
    mutex_event(_mutex_b_free, tla.pipes.CUBE)
    mutex_event(_mutex_cube_done, tla.pipes.CUBE)
    mutex_event(_mutex_cube_done, tla.pipes.FIX)
    if tla.const_expr(save_cache):
        tla.copy(
            strided_rows(out_w, valid, 128, stride),
            cw,
            tla.params.CopyL0C2DstParams(
                quant_mode=tla.params.QuantMode.PER_TENSOR, quant_scale=-1.0
            ),
        )
    tla.copy(negative_w, cw)
    mutex_event(_mutex_w_ready, tla.pipes.FIX)
    mutex_event(_mutex_b_free, tla.pipes.MTE1)
    if tla.const_expr(not inputs_ready):
        tla.cross_core_wait_flag(v_ready, tla.pipes.MTE1)
    tla.copy(beta_b, vbeta)
    mutex_event(_mutex_loaded, tla.pipes.MTE1)
    mutex_event(_mutex_loaded, tla.pipes.CUBE)
    tla.mmad(cu, inverse_a, beta_b, init_c=True)
    if tla.const_expr(save_cache):
        mutex_event(_mutex_cube_done, tla.pipes.CUBE)
        mutex_event(_mutex_cube_done, tla.pipes.FIX)
        tla.copy(strided_rows(out_u, valid, 128, stride), cu)
        mutex_event(_mutex_output_read, tla.pipes.FIX)
    state_b = tla.make_tensor_like(bp + 8192, row_major(bp, 128, 128), tla.arch.nZ)
    state_t = tla.make_tensor_like(
        state.ptr, row_major(state.ptr, 128, 128), tla.arch.nZ
    )
    wa = tla.make_tensor_like(ap + 16384, row_major(ap, 64, 128), tla.arch.zN)
    qa = tla.make_tensor_like(ap + 24576, row_major(ap, 64, 128), tla.arch.zN)
    tla.cross_core_wait_flag(state_ready, tla.pipes.MTE1)
    tla.copy(state_b, state_t)
    mutex_event(_mutex_w_ready, tla.pipes.MTE1)
    tla.copy(wa, negative_w)
    mutex_event(_mutex_loaded, tla.pipes.MTE1)
    mutex_event(_mutex_loaded, tla.pipes.CUBE)
    if tla.const_expr(save_cache):
        mutex_event(_mutex_output_read, tla.pipes.CUBE)
    tla.mmad(cu, wa, state_b, init_c=False)
    mutex_event(_mutex_cube_done, tla.pipes.CUBE)
    tla.copy(qa, qs)
    mutex_event(_mutex_loaded, tla.pipes.MTE1)
    mutex_event(_mutex_loaded, tla.pipes.CUBE)
    if tla.const_expr(output_overlap):
        if first_chunk:
            mutex_event(_mutex_output_free, tla.pipes.FIX)
        mutex_event(_mutex_output_free, tla.pipes.CUBE)
    tla.mmad(co, qa, state_b, init_c=True)
    mutex_event(_mutex_cube_done, tla.pipes.FIX)
    if tla.const_expr(save_cache):
        tla.copy(strided_rows(out_vnew, valid, 128, stride), cu)
    tla.copy(vnew, cu)
    mutex_event(_mutex_vnew_ready, tla.pipes.FIX)
    kb = tla.make_tensor_like(bp + 24576, row_major(bp, 64, 128), tla.arch.nZ)
    aq = tla.make_tensor_like(ap + 8192, row_major(ap, 64, 64), tla.arch.zN)
    vt = tla.make_tensor_like(vnew.ptr, row_major(vnew.ptr, 128, 64), tla.arch.nZ)
    va = tla.make_tensor_like(ap, row_major(ap, 128, 64), tla.arch.zN)
    vb = tla.make_tensor_like(bp, row_major(bp, 64, 128), tla.arch.nZ)
    tla.copy(kb, kg)
    mutex_event(_mutex_vnew_ready, tla.pipes.MTE1)
    tla.copy(va, vt)
    mutex_event(_mutex_loaded, tla.pipes.MTE1)
    mutex_event(_mutex_loaded, tla.pipes.CUBE)
    # vnew_ready -> MTE1 loads -> loaded already closes FIX's cu reader.
    ch = tla.make_tensor_like(cp, row_major(cp, 128, 128), tla.arch.L0Clayout)
    tla.mmad(ch, va, kb, init_c=True)
    mutex_event(_mutex_cube_done, tla.pipes.CUBE)
    mutex_event(_mutex_cube_done, tla.pipes.FIX)
    tla.copy(
        strided_rows(delta, 128, 128, 64),
        ch,
        tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N),
    )
    tla.cross_core_set_flag(delta_ready, tla.pipes.FIX)
    if tla.const_expr(output_overlap):
        mutex_event(_mutex_delta_read, tla.pipes.FIX)
    # Only Aqk depends on recur_ready; Qs and Kg were acquired in the bulk bank.
    tla.cross_core_wait_flag(recurrent_ready, tla.pipes.MTE1)
    tla.copy(aq, aqk)
    tla.copy(vb, vnew)
    mutex_event(_mutex_loaded, tla.pipes.MTE1)
    mutex_event(_mutex_loaded, tla.pipes.CUBE)
    _mutex_output_path = tla.mutex("pr_output_path", id=0)
    _mutex_output_path.lock(pipe=tla.pipes.CUBE)
    tla.mmad(co, aq, vb, init_c=False)
    _mutex_output_path.unlock(pipe=tla.pipes.CUBE)
    _mutex_output_path.lock(pipe=tla.pipes.FIX)
    tla.copy(strided_rows(out, valid, 128, stride), co)
    _mutex_output_path.unlock(pipe=tla.pipes.FIX)
    if tla.const_expr(output_overlap):
        mutex_event(_mutex_delta_read, tla.pipes.CUBE)


# -----------------------------------------------------------------------------
# Persistent single-chunk path.
# -----------------------------------------------------------------------------


@tla.jit
def persistent_chunk(
    ubase,
    lp,
    ap,
    bp,
    cp,
    q,
    k,
    v,
    g,
    beta,
    alpha,
    bias,
    exchange,
    identity16,
    aqk,
    akk,
    w,
    u,
    qg,
    kg,
    qs,
    gamma,
    qn,
    kn,
    rstd,
    gcache,
    bact,
    out,
    vnew,
    history,
    valid,
    mode,
    lower,
    negative,
    normalize,
    scale,
    q_stride=128,
    v_stride=128,
    beta_stride=1,
    norm_stride=128,
    beta_out_stride=1,
    norm_owner=1,
    rstd_k_offset=64,
    rstd_stride=1,
    beta_sigmoid=True,
    save_matrices=True,
    save_cache=True,
    save_history=True,
    save_norm=True,
    save_beta=True,
    prefetch=False,
    first_chunk=True,
    next_valid=0,
    bulk_operands=False,
):
    # State [128,64] occupies [0,8192) FP32 for the entire sequence.
    _mutex_identity_ready = tla.mutex("persistent_chunk_4936_identity_ready", id=0)
    _mutex_v_done = tla.mutex("persistent_chunk_4936_v_done", id=1)
    _mutex_cast_free = tla.mutex("persistent_chunk_4936_cast_free", id=2)
    _mutex_cube_done = tla.mutex("persistent_chunk_4936_cube_done", id=3)
    _mutex_free0 = tla.mutex("persistent_chunk_4936_free0", id=4)
    _mutex_free1 = tla.mutex("persistent_chunk_4936_free1", id=5)
    state = flat(ubase, 8192)
    uq = flat(tla.recast_ptr(ubase + 8192, dtype=tla.BFloat16), 4096)
    uk = flat(tla.recast_ptr(ubase + 10240, dtype=tla.BFloat16), 4096)
    ug = flat(ubase + 12288, 4096)
    uv = flat(tla.recast_ptr(ubase + 16384, dtype=tla.BFloat16), 4096)
    ub = flat(ubase + 18432, 64)
    packed = flat(tla.recast_ptr(ubase + 18496, dtype=tla.BFloat16), 18560)
    kk = flat(ubase + 27776, 2048)
    mq = flat(ubase + 29824, 2048)
    nt = flat(tla.recast_ptr(ubase + 31872, dtype=tla.Float16), 2048)
    ntnz = flat(tla.recast_ptr(ubase + 32896, dtype=tla.Float16), 2112)
    dnz = flat(tla.recast_ptr(ubase + 33952, dtype=tla.Float16), 2112)
    prefix = flat(ubase + 35008, 384)
    raw_gate = flat(tla.recast_ptr(ubase + 35392, dtype=g.ptr.dtype), 4096)
    gate_bf16 = flat(tla.recast_ptr(ubase + 39488, dtype=tla.BFloat16), 4096)
    beta_io = flat(tla.recast_ptr(ubase + 41568, dtype=beta.ptr.dtype), 1024)
    beta_column = flat(ubase + 41568, 512)
    stats = flat(ubase + 42592, 384)
    bias_ub = flat(ubase + 42976, 64)
    alpha_ub = flat(ubase + 43040, 1)
    raw_beta = flat(ubase + 43104, 64)
    ugamma = flat(ubase + 43168, 64)
    raw_q = None
    raw_k = None
    if tla.const_expr(prefetch):
        raw_q = flat(tla.recast_ptr(ubase + 43232, dtype=tla.BFloat16), 4096)
        raw_k = flat(tla.recast_ptr(ubase + 45280, dtype=tla.BFloat16), 4096)
        if tla.const_expr(bulk_operands):
            raw_q = flat(tla.recast_ptr(ubase + 47776, dtype=tla.BFloat16), 4096)
            raw_k = flat(tla.recast_ptr(ubase + 49824, dtype=tla.BFloat16), 4096)
    prefetch_free = tla.flag("pc_prefetch_free", tla.pipes.VECTOR, tla.pipes.MTE2)
    # Packed score bytes are free after score_done; leaf reads only nt32.
    nt32 = flat(ubase + 18496, 2048)
    # Raw gate is dead after prefix; these vector exits share its storage.
    cast = flat(tla.recast_ptr(ubase + 35392, dtype=tla.BFloat16), 4096)
    decayed = flat(tla.recast_ptr(ubase + 37440, dtype=tla.BFloat16), 4096)
    nz = flat(tla.recast_ptr(ubase + 39488, dtype=tla.BFloat16), 4160)
    # After the leaf/score readers finish, reuse score scratch for H snapshot.
    snapshot = flat(tla.recast_ptr(ubase + 18496, dtype=tla.BFloat16), 8192)
    snz = flat(tla.recast_ptr(ubase + 22592, dtype=tla.BFloat16), 8256)
    delta = flat(ubase + 8192, 8192)
    public = flat(tla.recast_ptr(nt.ptr, dtype=tla.BFloat16), 2048)
    aq_nz = flat(tla.recast_ptr(ntnz.ptr, dtype=tla.BFloat16), 2112)
    invptr = tla.recast_ptr(lp + 36864, dtype=tla.Float16)
    a = tla.recast_ptr(ap + 4096, dtype=tla.Float16)
    b = tla.recast_ptr(bp + 20480, dtype=tla.Float16)
    nt_l1 = matrix64(invptr, tla.arch.zN)
    diagonal = matrix64(invptr + 4096, tla.arch.zN)
    identity = matrix64(invptr + 8192, tla.arch.zN)
    zero = matrix64(invptr + 12288, tla.arch.zN)
    temporary = matrix64(invptr + 16384, tla.arch.zN)
    inverse = matrix64(invptr + 20480, tla.arch.zN)
    a0 = matrix64(a, tla.arch.zN)
    a1 = matrix64(a + 4096, tla.arch.zN)
    b0 = matrix64(b, tla.arch.nZ)
    b1 = matrix64(b + 4096, tla.arch.nZ)
    ci = matrix64(cp, tla.arch.L0Clayout)
    ct = matrix64(cp + 4096, tla.arch.L0Clayout)
    resident = matrix64(lp + 61440, tla.arch.zN)
    kl = tla.make_tensor_like(lp + 65536, row_major(lp, 64, 128), tla.arch.zN)
    vl = tla.make_tensor_like(lp + 73728, row_major(lp, 64, 128), tla.arch.zN)
    ql = tla.make_tensor_like(lp + 81920, row_major(lp, 64, 128), tla.arch.zN)
    kgl = tla.make_tensor_like(lp + 90112, row_major(lp, 64, 128), tla.arch.zN)
    al = matrix64(lp + 98304, tla.arch.zN)
    hl = tla.make_tensor_like(lp, row_major(lp, 128, 128), tla.arch.zN)
    wl = tla.make_tensor_like(lp + 16384, row_major(lp, 64, 128), tla.arch.zN)
    vnl = tla.make_tensor_like(lp + 24576, row_major(lp, 64, 128), tla.arch.zN)
    if tla.const_expr(bulk_operands):
        recurrent = recurrent_l1(lp + 65536)
        ql = tla.tile_view(recurrent, tla.make_shape(64, 128), tla.make_coord(0, 0))
        kgl = tla.tile_view(recurrent, tla.make_shape(64, 128), tla.make_coord(1, 0))
        kl = tla.tile_view(recurrent, tla.make_shape(64, 128), tla.make_coord(2, 0))
        vl = tla.tile_view(recurrent, tla.make_shape(64, 128), tla.make_coord(3, 0))
        recurrent_nz = flat(tla.recast_ptr(ubase + 35392, dtype=tla.BFloat16), 16448)
        ugamma = flat(ubase + 43616, 64)
        saved_qg = flat(tla.recast_ptr(ubase + 43680, dtype=tla.BFloat16), 4096)
        saved_kg = flat(tla.recast_ptr(ubase + 45728, dtype=tla.BFloat16), 4096)
    gm_loaded = tla.flag("pc_gm_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    cast_free = tla.flag("pc_cast_free", tla.pipes.MTE3, tla.pipes.VECTOR)
    peer_loaded = tla.flag("pc_peer_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    published = tla.cross_flag("pc_partial_ready", mode=1)
    consumed = tla.cross_flag("pc_partial_free", mode=1)
    identity_ready = tla.flag("pc_identity_ready", tla.pipes.MTE2, tla.pipes.MTE1)
    loaded = tla.flag("pc_loaded", tla.pipes.MTE1, tla.pipes.CUBE)
    cube_done = tla.flag("pc_cube_done", tla.pipes.CUBE, tla.pipes.FIX)
    v_done = tla.flag("pc_v_done", tla.pipes.VECTOR, tla.pipes.MTE3)
    ready0 = tla.flag("pc_ready0", tla.pipes.MTE1, tla.pipes.CUBE)
    ready1 = tla.flag("pc_ready1", tla.pipes.MTE1, tla.pipes.CUBE)
    free0 = tla.flag("pc_free0", tla.pipes.CUBE, tla.pipes.MTE1)
    free1 = tla.flag("pc_free1", tla.pipes.CUBE, tla.pipes.MTE1)
    fix_ready = tla.flag("pc_fix_ready", tla.pipes.FIX, tla.pipes.MTE1)
    acc_free = tla.flag("pc_acc_free", tla.pipes.FIX, tla.pipes.CUBE)
    resident_ready = tla.flag("pc_resident_ready", tla.pipes.FIX, tla.pipes.MTE1)
    scores_ready = tla.cross_flag("pc_scores_ready", mode=2)
    scores_done = tla.cross_flag("pc_scores_done", mode=2)
    inverse_ready = tla.cross_flag("pc_inverse_ready", mode=2)
    k_ready = tla.cross_flag("pc_k_ready", mode=2)
    v_ready = tla.cross_flag("pc_v_ready", mode=2)
    h_ready = tla.cross_flag("pc_h_ready", mode=2)
    recur_ready = tla.cross_flag("pc_recur_ready", mode=2)
    delta_ready = tla.cross_flag("pc_delta_ready", mode=2)
    beta_pitch = 1
    if tla.const_expr(beta_stride != 1):
        if tla.const_expr(beta.ptr.dtype == tla.Float32):
            beta_pitch = 8
        else:
            beta_pitch = 16
    with tla.cube():
        identity.fill(0)
        zero.fill(0)
        tla.pipe_barrier(tla.pipes.MTE2)
        for j in tla.range_constexpr(4):
            tla.copy(
                tla.tile_view(identity, tla.make_shape(16, 16), tla.make_coord(j, j)),
                row_major(identity16.ptr, 16, 16),
            )
        mutex_event(_mutex_identity_ready, tla.pipes.MTE2)
    with tla.vector():
        aiv = tla.arch.sub_block_idx()
        # Incoming state is ready on V, so publish the transitive MTE2->V->MTE3
        # edge before a history DMA reads the same UB allocation.
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        if tla.const_expr(save_history):
            tla.copy(
                strided_rows(history.ptr + aiv * 64, 128, 64, 128),
                row_major(state.ptr, 128, 64),
            )
    prefix_inputs(
        q,
        k,
        v,
        g,
        beta,
        alpha,
        bias,
        exchange,
        qn,
        kn,
        rstd,
        gcache,
        bact,
        uq,
        uk,
        uv,
        ug,
        raw_gate,
        gate_bf16,
        raw_beta,
        beta_io,
        beta_column,
        bias_ub,
        alpha_ub,
        stats,
        ub,
        valid,
        mode,
        lower,
        negative,
        normalize,
        q_stride,
        v_stride,
        beta_stride,
        beta_pitch,
        norm_stride,
        beta_out_stride,
        norm_owner,
        rstd_k_offset,
        rstd_stride,
        gm_loaded,
        v_done,
        cast_free,
        peer_loaded,
        published,
        consumed,
        beta_sigmoid,
        save_cache,
        save_norm,
        save_beta,
        raw_q,
        raw_k,
        first_chunk,
    )
    if tla.const_expr(mode == 1):
        with tla.vector():
            aiv = tla.arch.sub_block_idx()
            pack_persistent_panel(uq, uk, ug, packed, 0, scale)
        if tla.const_expr(prefetch):
            prefetch_half(q, k, raw_q, raw_k, next_valid, q_stride, 0, prefetch_free)
        with tla.vector():
            aiv = tla.arch.sub_block_idx()
            pack_persistent_panel(uq, uk, ug, packed, 1, scale)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            tla.copy(
                tla.tile_view(
                    score_l1(lp), tla.make_shape(288, 64), tla.make_coord(0, aiv)
                ),
                padded_nz(packed.ptr, 288),
            )
            tla.cross_core_set_flag(scores_ready, tla.pipes.MTE3)
        if tla.const_expr(prefetch):
            prefetch_half(q, k, raw_q, raw_k, next_valid, q_stride, 1, prefetch_free)
        with tla.cube():
            tla.cross_core_wait_flag(scores_ready, tla.pipes.MTE1)
            persistent_score_mmads(score_l1(lp), ap, bp, cp, cp + 4096, loaded)
    else:
        per_d_scores(
            uq,
            uk,
            ug,
            flat(packed.ptr, 1024),
            flat(tla.recast_ptr(packed.ptr + 4096, dtype=tla.Float32), 1024),
            flat(packed.ptr + 1024, 1024),
            flat(packed.ptr + 2048, 1024),
            flat(packed.ptr + 3072, 1024),
            packed.ptr + 6144,
            lp,
            ap,
            bp,
            cp,
            cp + 4096,
            scale,
            False,
        )
    if tla.const_expr(bulk_operands):
        with tla.vector():
            aiv_recurrent = tla.arch.sub_block_idx()
            mutex_event(_mutex_cast_free, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
            pack_recurrent(
                uq,
                uk,
                uv,
                ug,
                ub,
                recurrent_nz,
                ugamma,
                saved_qg,
                saved_kg,
                scale,
                save_cache,
            )
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            tla.copy(
                tla.tile_view(
                    recurrent, tla.make_shape(256, 64), tla.make_coord(0, aiv_recurrent)
                ),
                padded_nz(recurrent_nz.ptr, 256),
            )
            tla.cross_core_set_flag(k_ready, tla.pipes.MTE3)
            tla.cross_core_set_flag(v_ready, tla.pipes.MTE3)
            if tla.const_expr(save_cache):
                tla.copy(
                    strided_rows(qg.ptr + aiv_recurrent * 64, valid, 64, v_stride),
                    row_major(saved_qg.ptr, valid, 64),
                )
                tla.copy(
                    strided_rows(kg.ptr + aiv_recurrent * 64, valid, 64, v_stride),
                    row_major(saved_kg.ptr, valid, 64),
                )
    with tla.cube():
        mutex_event(_mutex_cube_done, tla.pipes.CUBE)
        mutex_event(_mutex_cube_done, tla.pipes.FIX)
        tla.copy(
            row_major(kk.ptr, 64, 64),
            ci,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        tla.copy(
            row_major(mq.ptr, 64, 64),
            ct,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        tla.cross_core_set_flag(scores_done, tla.pipes.FIX)
        tla.cross_core_wait_flag(inverse_ready, tla.pipes.MTE1)
        mutex_event(_mutex_identity_ready, tla.pipes.MTE1)
        mutex_event(_mutex_free0, tla.pipes.CUBE)
        mutex_event(_mutex_free1, tla.pipes.CUBE)
        compose_persistent_inverse(
            nt_l1,
            diagonal,
            identity,
            temporary,
            inverse,
            zero,
            a0,
            a1,
            b0,
            b1,
            ci,
            ct,
            ready0,
            ready1,
            free0,
            free1,
            fix_ready,
            acc_free,
        )
        publish_resident_inverse(
            ci,
            row_major(akk.ptr, valid, 64),
            resident,
            save_matrices,
            cube_done,
            resident_ready,
        )
        mutex_event(_mutex_free0, tla.pipes.MTE1)
        mutex_event(_mutex_free1, tla.pipes.MTE1)
        persistent_products_and_state(
            resident,
            kl,
            vl,
            ql,
            kgl,
            al,
            hl,
            wl,
            vnl,
            w.ptr,
            u.ptr,
            vnew.ptr,
            out.ptr,
            delta.ptr,
            ap,
            bp,
            cp,
            valid,
            v_stride,
            resident_ready,
            k_ready,
            v_ready,
            h_ready,
            recur_ready,
            delta_ready,
            loaded,
            cube_done,
            save_cache,
        )
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        aiv = tla.arch.sub_block_idx()
        tla.cross_core_wait_flag(scores_done, tla.pipes.VECTOR)
        apply_beta_and_finish_t(kk, ub, nt, aiv)
        nd2nz_tile(nt, ntnz, 32, 64)
        with tla.vec.func(mode="simd"):
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            full, unused = tla.update_mask(64, dtype=tla.Float32)
            for row in tla.range(32):
                at(nt32, row * 64, 64).store(
                    load_bf16_f32(at(nt, row * 64, 64)), mask=full
                )
        persistent_diagonal_inverse(nt32, prefix, padded_nz(dnz.ptr, 32), aiv)
    if tla.const_expr(prefetch and mode != 1):
        prefetch_half(q, k, raw_q, raw_k, next_valid, q_stride, 0, prefetch_free)
        prefetch_half(q, k, raw_q, raw_k, next_valid, q_stride, 1, prefetch_free)
    with tla.vector():
        aiv = tla.arch.sub_block_idx()
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        tla.copy(
            tla.tile_view(nt_l1, tla.make_shape(32, 64), tla.make_coord(aiv, 0)),
            padded_nz(ntnz.ptr, 32),
        )
        tla.copy(
            tla.tile_view(diagonal, tla.make_shape(32, 64), tla.make_coord(aiv, 0)),
            padded_nz(dnz.ptr, 32),
        )
        tla.cross_core_set_flag(inverse_ready, tla.pipes.MTE3)
        if tla.const_expr(not bulk_operands):
            # API-062: no scaled FIX to L1. Negate K_beta exactly before BF16
            # packing, so Cube produces -W; the public W FIX reverses the sign.
            beta_weighted_operand(uk, ub, ug, cast, True, True)
            nd2nz_tile(cast, nz, 64, 64)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            tla.copy(
                tla.tile_view(kl, tla.make_shape(64, 64), tla.make_coord(0, aiv)),
                padded_nz(nz.ptr, 64),
            )
            tla.cross_core_set_flag(k_ready, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
            beta_weighted_operand(uv, ub, ug, cast, False)
            nd2nz_tile(cast, nz, 64, 64)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            tla.copy(
                tla.tile_view(vl, tla.make_shape(64, 64), tla.make_coord(0, aiv)),
                padded_nz(nz.ptr, 64),
            )
            tla.cross_core_set_flag(v_ready, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
            prepare_decayed_q(uq, ug, cast, decayed, scale)
            nd2nz_tile(decayed, nz, 64, 64)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            if tla.const_expr(save_cache):
                tla.copy(
                    strided_rows(qg.ptr + aiv * 64, valid, 64, v_stride),
                    row_major(cast.ptr, valid, 64),
                )
            tla.copy(
                tla.tile_view(ql, tla.make_shape(64, 64), tla.make_coord(0, aiv)),
                padded_nz(nz.ptr, 64),
            )
            mutex_event(_mutex_cast_free, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
            restored_state_inputs(uk, ug, cast, ugamma)
            nd2nz_tile(cast, nz, 64, 64)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            if tla.const_expr(save_cache):
                tla.copy(
                    strided_rows(kg.ptr + aiv * 64, valid, 64, v_stride),
                    row_major(cast.ptr, valid, 64),
                )
            tla.copy(
                tla.tile_view(kgl, tla.make_shape(64, 64), tla.make_coord(0, aiv)),
                padded_nz(nz.ptr, 64),
            )
            mutex_event(_mutex_cast_free, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
        mask_mqk(mq, public, aiv)
        nd2nz_tile(public, aq_nz, 32, 64)
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        rows = valid - aiv * 32
        if rows > 32:
            rows = 32
        if tla.const_expr(save_matrices):
            if rows > 0:
                tla.copy(
                    row_major(aqk.ptr + aiv * 2048, rows, 64),
                    row_major(public.ptr, rows, 64),
                )
        tla.copy(
            tla.tile_view(al, tla.make_shape(32, 64), tla.make_coord(aiv, 0)),
            padded_nz(aq_nz.ptr, 32),
        )
        tla.cross_core_set_flag(recur_ready, tla.pipes.MTE3)
        cast_state_snapshot(state, snapshot, 128)
        nd2nz_tile(snapshot, snz, 128, 64)
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        tla.copy(
            tla.tile_view(hl, tla.make_shape(128, 64), tla.make_coord(0, aiv)),
            padded_nz(snz.ptr, 128),
        )
        tla.cross_core_set_flag(h_ready, tla.pipes.MTE3)
        tla.cross_core_wait_flag(delta_ready, tla.pipes.VECTOR)
        scale_and_add_delta(state, delta, ugamma, 128)
        tla.pipe_barrier(tla.pipes.ALL)


# -----------------------------------------------------------------------------
# Persistent chunk pipeline.
# -----------------------------------------------------------------------------


@tla.jit
def persistent_pipeline_chunk(
    ubase,
    lp,
    ap,
    bp,
    cp,
    q,
    k,
    v,
    g,
    beta,
    alpha,
    bias,
    exchange,
    identity16,
    aqk,
    akk,
    w,
    u,
    qg,
    kg,
    qs,
    gamma,
    qn,
    kn,
    rstd,
    gcache,
    bact,
    out,
    vnew,
    history,
    valid,
    mode,
    lower,
    negative,
    normalize,
    scale,
    q_stride=128,
    v_stride=128,
    beta_stride=1,
    norm_stride=128,
    beta_out_stride=1,
    norm_owner=1,
    rstd_k_offset=64,
    rstd_stride=1,
    beta_sigmoid=True,
    save_matrices=True,
    save_cache=True,
    save_history=True,
    save_norm=True,
    save_beta=True,
    prefetch=True,
    first_chunk=True,
    next_valid=0,
    bulk_operands=True,
    following_valid=0,
    parity=0,
    cache_heads=1,
    shared_scores=False,
):
    # State [128,64] occupies [0,8192) FP32 for the entire sequence.
    _mutex_identity_ready = tla.mutex(
        "persistent_pipeline_chunk_5456_identity_ready", id=0
    )
    _mutex_v_done = tla.mutex("persistent_pipeline_chunk_5456_v_done", id=1)
    _mutex_cast_free = tla.mutex("persistent_pipeline_chunk_5456_cast_free", id=2)
    _mutex_input_dma_free = tla.mutex(
        "persistent_pipeline_chunk_5456_input_dma_free", id=3
    )
    _mutex_cube_done = tla.mutex("persistent_pipeline_chunk_5456_cube_done", id=4)
    _mutex_free0 = tla.mutex("persistent_pipeline_chunk_5456_free0", id=5)
    _mutex_free1 = tla.mutex("persistent_pipeline_chunk_5456_free1", id=6)
    score_rows = 224 if shared_scores else 288
    state = flat(ubase, 8192)
    uq = flat(tla.recast_ptr(ubase + 8192, dtype=tla.BFloat16), 4096)
    uk = flat(tla.recast_ptr(ubase + 10240, dtype=tla.BFloat16), 4096)
    ug = flat(ubase + 12288, 4096)
    uv = flat(tla.recast_ptr(ubase + 16384, dtype=tla.BFloat16), 4096)
    ub = flat(ubase + 18432, 64)
    packed = flat(tla.recast_ptr(ubase + 18496, dtype=tla.BFloat16), 18560)
    kk = flat(ubase + 27776, 2048)
    mq = flat(ubase + 29824, 2048)
    nt = flat(tla.recast_ptr(ubase + 31872, dtype=tla.Float16), 2048)
    ntnz = flat(tla.recast_ptr(ubase + 32896, dtype=tla.Float16), 2112)
    dnz = flat(tla.recast_ptr(ubase + 33952, dtype=tla.Float16), 2112)
    prefix = flat(ubase + 35008, 384)
    raw_gate = flat(tla.recast_ptr(ubase + 35392, dtype=g.ptr.dtype), 4096)
    gate_bf16 = flat(tla.recast_ptr(ubase + 39488, dtype=tla.BFloat16), 4096)
    beta_io = flat(tla.recast_ptr(ubase + 41568, dtype=beta.ptr.dtype), 1024)
    beta_column = flat(ubase + 41568, 512)
    stats = flat(ubase + 42592, 384)
    bias_ub = flat(ubase + 51936, 64)
    alpha_ub = flat(ubase + 52000, 1)
    raw_beta = flat(ubase + 43104, 64)
    ugamma = flat(ubase + 43168, 64)
    raw_q = None
    raw_k = None
    if tla.const_expr(prefetch):
        raw_q = flat(tla.recast_ptr(ubase + 43232, dtype=tla.BFloat16), 4096)
        raw_k = flat(tla.recast_ptr(ubase + 45280, dtype=tla.BFloat16), 4096)
        if tla.const_expr(bulk_operands):
            raw_q = flat(tla.recast_ptr(ubase + 47840, dtype=tla.BFloat16), 4096)
            raw_k = flat(tla.recast_ptr(ubase + 49888, dtype=tla.BFloat16), 4096)
    prefetch_free = tla.flag("pc_prefetch_free", tla.pipes.VECTOR, tla.pipes.MTE2)
    # Packed score bytes are free after score_done; leaf reads only nt32.
    nt32 = flat(ubase + 18496, 2048)
    # Raw gate is dead after prefix; these vector exits share its storage.
    cast = flat(tla.recast_ptr(ubase + 35392, dtype=tla.BFloat16), 4096)
    decayed = flat(tla.recast_ptr(ubase + 37440, dtype=tla.BFloat16), 4096)
    nz = flat(tla.recast_ptr(ubase + 39488, dtype=tla.BFloat16), 4160)
    # After the leaf/score readers finish, reuse score scratch for H snapshot.
    snz = flat(tla.recast_ptr(ubase + 22592, dtype=tla.BFloat16), 8256)
    delta = flat(ubase + 53248, 8192)
    public = flat(tla.recast_ptr(nt.ptr, dtype=tla.BFloat16), 2048)
    aq_nz = flat(tla.recast_ptr(ntnz.ptr, dtype=tla.BFloat16), 2112)
    if tla.const_expr(bulk_operands):
        aq_nz = flat(tla.recast_ptr(ubase + 52064, dtype=tla.BFloat16), 2112)
    invptr = tla.recast_ptr(lp + 36864, dtype=tla.Float16)
    a = tla.recast_ptr(ap + 4096, dtype=tla.Float16)
    b = tla.recast_ptr(bp + 20480, dtype=tla.Float16)
    nt_l1 = matrix64(invptr, tla.arch.zN)
    diagonal = matrix64(invptr + 4096, tla.arch.zN)
    identity = matrix64(invptr + 8192, tla.arch.zN)
    zero = matrix64(invptr + 12288, tla.arch.zN)
    temporary = matrix64(invptr + 16384, tla.arch.zN)
    inverse = matrix64(invptr + 20480, tla.arch.zN)
    a0 = matrix64(a, tla.arch.zN)
    a1 = matrix64(a + 4096, tla.arch.zN)
    b0 = matrix64(b, tla.arch.nZ)
    b1 = matrix64(b + 4096, tla.arch.nZ)
    ci = matrix64(cp, tla.arch.L0Clayout)
    ct = matrix64(cp + 4096, tla.arch.L0Clayout)
    resident = matrix64(lp + 61440, tla.arch.zN)
    kl = tla.make_tensor_like(lp + 65536, row_major(lp, 64, 128), tla.arch.zN)
    vl = tla.make_tensor_like(lp + 73728, row_major(lp, 64, 128), tla.arch.zN)
    ql = tla.make_tensor_like(lp + 81920, row_major(lp, 64, 128), tla.arch.zN)
    kgl = tla.make_tensor_like(lp + 90112, row_major(lp, 64, 128), tla.arch.zN)
    al = matrix64(lp + 131072, tla.arch.zN)
    hl = tla.make_tensor_like(lp + 135168, row_major(lp, 128, 128), tla.arch.zN)
    wl = tla.make_tensor_like(lp + 151552, row_major(lp, 64, 128), tla.arch.zN)
    vnl = tla.make_tensor_like(lp + 159744, row_major(lp, 64, 128), tla.arch.zN)
    if tla.const_expr(bulk_operands):
        recurrent = recurrent_l1(lp + 65536 + parity * 32768)
        ql = tla.tile_view(recurrent, tla.make_shape(64, 128), tla.make_coord(0, 0))
        kgl = tla.tile_view(recurrent, tla.make_shape(64, 128), tla.make_coord(1, 0))
        kl = tla.tile_view(recurrent, tla.make_shape(64, 128), tla.make_coord(2, 0))
        vl = tla.tile_view(recurrent, tla.make_shape(64, 128), tla.make_coord(3, 0))
        recurrent_nz = flat(tla.recast_ptr(ubase + 35392, dtype=tla.BFloat16), 16448)
        ugamma = flat(ubase + 43616 + parity * 64, 64)
        saved_qg = flat(tla.recast_ptr(ubase + 43744, dtype=tla.BFloat16), 4096)
        saved_kg = flat(tla.recast_ptr(ubase + 45792, dtype=tla.BFloat16), 4096)
    gm_loaded = tla.flag("pc_gm_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    cast_free = tla.flag("pc_cast_free", tla.pipes.MTE3, tla.pipes.VECTOR)
    input_dma_free = tla.flag("pc_input_dma_free", tla.pipes.MTE3, tla.pipes.MTE2)
    peer_loaded = tla.flag("pc_peer_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    published = tla.cross_flag("pc_partial_ready", mode=1)
    consumed = tla.cross_flag("pc_partial_free", mode=1)
    identity_ready = tla.flag("pc_identity_ready", tla.pipes.MTE2, tla.pipes.MTE1)
    loaded = tla.flag("pc_loaded", tla.pipes.MTE1, tla.pipes.CUBE)
    cube_done = tla.flag("pc_cube_done", tla.pipes.CUBE, tla.pipes.FIX)
    v_done = tla.flag("pc_v_done", tla.pipes.VECTOR, tla.pipes.MTE3)
    ready0 = tla.flag("pc_ready0", tla.pipes.MTE1, tla.pipes.CUBE)
    ready1 = tla.flag("pc_ready1", tla.pipes.MTE1, tla.pipes.CUBE)
    free0 = tla.flag("pc_free0", tla.pipes.CUBE, tla.pipes.MTE1)
    free1 = tla.flag("pc_free1", tla.pipes.CUBE, tla.pipes.MTE1)
    fix_ready = tla.flag("pc_fix_ready", tla.pipes.FIX, tla.pipes.MTE1)
    acc_free = tla.flag("pc_acc_free", tla.pipes.FIX, tla.pipes.CUBE)
    resident_ready = tla.flag("pc_resident_ready", tla.pipes.FIX, tla.pipes.MTE1)
    scores_ready = tla.cross_flag("pc_scores_ready", mode=2)
    scores_done = tla.cross_flag("pc_scores_done", mode=2)
    mq_done = tla.cross_flag("pc_mq_done", mode=2)
    inverse_ready = tla.cross_flag("pc_inverse_ready", mode=2)
    k_ready = tla.cross_flag("pc_k_ready", mode=2)
    v_ready = tla.cross_flag("pc_v_ready", mode=2)
    h_ready = tla.cross_flag("pc_h_ready", mode=2)
    recur_ready = tla.cross_flag("pc_recur_ready", mode=2)
    delta_ready = tla.cross_flag("pc_delta_ready", mode=2)
    beta_pitch = 1
    if tla.const_expr(beta_stride != 1):
        if tla.const_expr(beta.ptr.dtype == tla.Float32):
            beta_pitch = 8
        else:
            beta_pitch = 16
    inverse_norm = None
    if tla.const_expr(normalize):
        inverse_norm = at(stats, 256, 128)
    if first_chunk:
        with tla.cube():
            identity.fill(0)
            zero.fill(0)
            tla.pipe_barrier(tla.pipes.MTE2)
            for j in tla.range_constexpr(4):
                tla.copy(
                    tla.tile_view(
                        identity, tla.make_shape(16, 16), tla.make_coord(j, j)
                    ),
                    row_major(identity16.ptr, 16, 16),
                )
            mutex_event(_mutex_identity_ready, tla.pipes.MTE2)
    with tla.vector():
        aiv_entry = tla.arch.sub_block_idx()
        # Incoming state is ready on V, so publish the transitive MTE2->V->MTE3
        # edge before a history DMA reads the same UB allocation.
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        if tla.const_expr(save_history):
            tla.copy(
                strided_rows(history.ptr + aiv_entry * 64, 128, 64, 128),
                row_major(state.ptr, 128, 64),
            )
        if first_chunk:
            cast_state_nz(state, snz, 128)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            tla.copy(
                tla.tile_view(
                    hl, tla.make_shape(128, 64), tla.make_coord(0, aiv_entry)
                ),
                padded_nz(snz.ptr, 128),
            )
            tla.cross_core_set_flag(h_ready, tla.pipes.MTE3)
    if first_chunk:
        prefix_inputs(
            q,
            k,
            v,
            g,
            beta,
            alpha,
            bias,
            exchange,
            qn,
            kn,
            rstd,
            gcache,
            bact,
            uq,
            uk,
            uv,
            ug,
            raw_gate,
            gate_bf16,
            raw_beta,
            beta_io,
            beta_column,
            bias_ub,
            alpha_ub,
            stats,
            ub,
            valid,
            mode,
            lower,
            negative,
            normalize,
            q_stride,
            v_stride,
            beta_stride,
            beta_pitch,
            norm_stride,
            beta_out_stride,
            norm_owner,
            rstd_k_offset,
            rstd_stride,
            gm_loaded,
            v_done,
            cast_free,
            peer_loaded,
            published,
            consumed,
            beta_sigmoid,
            save_cache,
            save_norm,
            save_beta,
            raw_q,
            raw_k,
            first_chunk,
            mode == 1 and prefetch,
        )
    if first_chunk:
        if tla.const_expr(mode == 1):
            with tla.vector():
                mutex_event(_mutex_cast_free, tla.pipes.MTE3)
                mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
                pack_persistent_panel(
                    uq,
                    uk,
                    ug,
                    packed,
                    0,
                    scale,
                    raw_q,
                    raw_k,
                    inverse_norm,
                    shared_scores,
                )
            if tla.const_expr(prefetch):
                prefetch_half(
                    q, k, raw_q, raw_k, next_valid, q_stride, 0, prefetch_free
                )
            with tla.vector():
                aiv_score = tla.arch.sub_block_idx()
                pack_persistent_panel(
                    uq,
                    uk,
                    ug,
                    packed,
                    1,
                    scale,
                    raw_q,
                    raw_k,
                    inverse_norm,
                    shared_scores,
                )
                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                tla.copy(
                    tla.tile_view(
                        score_l1(lp, shared_scores),
                        tla.make_shape(score_rows, 64),
                        tla.make_coord(0, aiv_score),
                    ),
                    padded_nz(packed.ptr, score_rows),
                )
                tla.cross_core_set_flag(scores_ready, tla.pipes.MTE3)
                if tla.const_expr(save_norm):
                    if norm_owner:
                        tla.copy(
                            strided_rows(
                                qn.ptr + aiv_score * 64, valid, 64, norm_stride
                            ),
                            row_major(uq.ptr, valid, 64),
                        )
                        tla.copy(
                            strided_rows(
                                kn.ptr + aiv_score * 64, valid, 64, norm_stride
                            ),
                            row_major(uk.ptr, valid, 64),
                        )
            if tla.const_expr(prefetch):
                prefetch_half(
                    q, k, raw_q, raw_k, next_valid, q_stride, 1, prefetch_free
                )
    if tla.const_expr(mode == 1):
        with tla.cube():
            tla.cross_core_wait_flag(scores_ready, tla.pipes.MTE1)
            persistent_score_mmads(
                score_l1(lp, shared_scores),
                ap,
                bp,
                cp,
                cp + 4096,
                loaded,
                kk,
                mq,
                scores_done,
                mq_done,
                cube_done,
                shared_scores,
            )
    else:
        with tla.vector():
            mutex_event(_mutex_cast_free, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
        per_d_scores(
            uq,
            uk,
            ug,
            flat(packed.ptr, 1024),
            flat(tla.recast_ptr(packed.ptr + 4096, dtype=tla.Float32), 1024),
            flat(packed.ptr + 1024, 1024),
            flat(packed.ptr + 2048, 1024),
            flat(packed.ptr + 3072, 1024),
            packed.ptr + 6144,
            lp,
            ap,
            bp,
            cp,
            cp + 4096,
            scale,
            False,
        )
    if first_chunk:
        if tla.const_expr(bulk_operands):
            with tla.vector():
                aiv_recurrent = tla.arch.sub_block_idx()
                mutex_event(_mutex_cast_free, tla.pipes.MTE3)
                mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
                pack_recurrent(
                    uq,
                    uk,
                    uv,
                    ug,
                    ub,
                    recurrent_nz,
                    ugamma,
                    saved_qg,
                    saved_kg,
                    scale,
                    save_cache,
                )
                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                tla.copy(
                    tla.tile_view(
                        recurrent,
                        tla.make_shape(256, 64),
                        tla.make_coord(0, aiv_recurrent),
                    ),
                    padded_nz(recurrent_nz.ptr, 256),
                )
                if parity == 0:
                    tla.cross_core_set_flag(k_ready, tla.pipes.MTE3)
                else:
                    tla.cross_core_set_flag(v_ready, tla.pipes.MTE3)
                if tla.const_expr(save_cache):
                    tla.copy(
                        strided_rows(qg.ptr + aiv_recurrent * 64, valid, 64, v_stride),
                        row_major(saved_qg.ptr, valid, 64),
                    )
                    tla.copy(
                        strided_rows(kg.ptr + aiv_recurrent * 64, valid, 64, v_stride),
                        row_major(saved_kg.ptr, valid, 64),
                    )
    if next_valid > 0:
        with tla.vector():
            # Current raw V/g/beta have already become independent L1 operands.
            # Close all previous MTE3 readers before reusing their UB aliases.
            mutex_event(_mutex_input_dma_free, tla.pipes.MTE3)
            mutex_event(_mutex_input_dma_free, tla.pipes.MTE2)
            load_raw_prefix_inputs(
                q,
                k,
                flat(v.ptr + tla.Int64(64 * v_stride), 1),
                flat(g.ptr + tla.Int64(64 * v_stride), 1),
                flat(beta.ptr + tla.Int64(64 * beta_stride), 1),
                alpha,
                bias,
                uq,
                uk,
                uv,
                raw_gate,
                beta_io,
                bias_ub,
                alpha_ub,
                next_valid,
                q_stride,
                v_stride,
                beta_stride,
                beta_pitch,
                False,
                False,
            )
    with tla.cube():
        if tla.const_expr(mode != 1):
            mutex_event(_mutex_cube_done, tla.pipes.CUBE)
            mutex_event(_mutex_cube_done, tla.pipes.FIX)
            tla.copy(
                row_major(kk.ptr, 64, 64),
                ci,
                tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
            )
            tla.copy(
                row_major(mq.ptr, 64, 64),
                ct,
                tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
            )
            tla.cross_core_set_flag(scores_done, tla.pipes.FIX)
            tla.cross_core_set_flag(mq_done, tla.pipes.FIX)
        tla.cross_core_wait_flag(inverse_ready, tla.pipes.MTE1)
        if first_chunk:
            mutex_event(_mutex_identity_ready, tla.pipes.MTE1)
        mutex_event(_mutex_free0, tla.pipes.CUBE)
        mutex_event(_mutex_free1, tla.pipes.CUBE)
        compose_persistent_inverse(
            nt_l1,
            diagonal,
            identity,
            temporary,
            inverse,
            zero,
            a0,
            a1,
            b0,
            b1,
            ci,
            ct,
            ready0,
            ready1,
            free0,
            free1,
            fix_ready,
            acc_free,
        )
        publish_resident_inverse(
            ci,
            row_major(akk.ptr, valid, 64),
            resident,
            save_matrices,
            cube_done,
            resident_ready,
        )
        mutex_event(_mutex_free0, tla.pipes.MTE1)
        mutex_event(_mutex_free1, tla.pipes.MTE1)
        if parity == 0:
            tla.cross_core_wait_flag(k_ready, tla.pipes.MTE1)
        else:
            tla.cross_core_wait_flag(v_ready, tla.pipes.MTE1)
        persistent_products_and_state(
            resident,
            kl,
            vl,
            ql,
            kgl,
            al,
            hl,
            wl,
            vnl,
            w.ptr,
            u.ptr,
            vnew.ptr,
            out.ptr,
            delta.ptr,
            ap,
            bp,
            cp,
            valid,
            v_stride,
            resident_ready,
            k_ready,
            v_ready,
            h_ready,
            recur_ready,
            delta_ready,
            loaded,
            cube_done,
            save_cache,
            True,
            True,
            first_chunk,
            next_valid == 0,
        )
    with tla.vector():
        aiv_leaf = tla.arch.sub_block_idx()
        tla.cross_core_wait_flag(scores_done, tla.pipes.VECTOR)
        # Snapshot NZ and leaf workspace alias; close its MTE3 reader first.
        mutex_event(_mutex_cast_free, tla.pipes.MTE3)
        mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
        persistent_negative_t(kk, ub, nt32, ntnz, aiv_leaf)
        persistent_diagonal_inverse(
            nt32, prefix, padded_nz(dnz.ptr, 32), aiv_leaf, first_chunk
        )
        # QK must finish before inverse reuses L0C or next score replaces L1.
        tla.cross_core_wait_flag(mq_done, tla.pipes.VECTOR)
    if tla.const_expr(prefetch and mode != 1):
        prefetch_half(q, k, raw_q, raw_k, next_valid, q_stride, 0, prefetch_free)
        prefetch_half(q, k, raw_q, raw_k, next_valid, q_stride, 1, prefetch_free)
    with tla.vector():
        aiv_inverse = tla.arch.sub_block_idx()
        mutex_event(_mutex_v_done, tla.pipes.VECTOR)
        mutex_event(_mutex_v_done, tla.pipes.MTE3)
        tla.copy(
            tla.tile_view(
                nt_l1, tla.make_shape(32, 64), tla.make_coord(aiv_inverse, 0)
            ),
            padded_nz(ntnz.ptr, 32),
        )
        tla.copy(
            tla.tile_view(
                diagonal, tla.make_shape(32, 64), tla.make_coord(aiv_inverse, 0)
            ),
            padded_nz(dnz.ptr, 32),
        )
        tla.cross_core_set_flag(inverse_ready, tla.pipes.MTE3)
    if tla.const_expr(bulk_operands):
        with tla.vector():
            aiv_aq = tla.arch.sub_block_idx()
            mask_mqk_nz(mq, public, aq_nz, aiv_aq, save_matrices)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            rows = valid - aiv_aq * 32
            if rows > 32:
                rows = 32
            if tla.const_expr(save_matrices):
                if rows > 0:
                    tla.copy(
                        row_major(aqk.ptr + aiv_aq * 2048, rows, 64),
                        row_major(public.ptr, rows, 64),
                    )
            tla.copy(
                tla.tile_view(al, tla.make_shape(32, 64), tla.make_coord(aiv_aq, 0)),
                padded_nz(aq_nz.ptr, 32),
            )
            tla.cross_core_set_flag(recur_ready, tla.pipes.MTE3)
    if next_valid > 0:
        prefix_inputs(
            flat(q.ptr + tla.Int64(64 * q_stride), 1),
            flat(k.ptr + tla.Int64(64 * q_stride), 1),
            flat(v.ptr + tla.Int64(64 * v_stride), 1),
            flat(g.ptr + tla.Int64(64 * v_stride), 1),
            flat(beta.ptr + tla.Int64(64 * beta_stride), 1),
            alpha,
            bias,
            exchange,
            flat(qn.ptr + tla.Int64(64 * norm_stride), 1),
            flat(kn.ptr + tla.Int64(64 * norm_stride), 1),
            flat(rstd.ptr + tla.Int64(64 * rstd_stride), 1),
            flat(gcache.ptr + tla.Int64(cache_heads * 8192), 1),
            flat(bact.ptr + tla.Int64(64 * beta_out_stride), 1),
            uq,
            uk,
            uv,
            ug,
            raw_gate,
            gate_bf16,
            raw_beta,
            beta_io,
            beta_column,
            bias_ub,
            alpha_ub,
            stats,
            ub,
            next_valid,
            mode,
            lower,
            negative,
            normalize,
            q_stride,
            v_stride,
            beta_stride,
            beta_pitch,
            norm_stride,
            beta_out_stride,
            norm_owner,
            rstd_k_offset,
            rstd_stride,
            gm_loaded,
            v_done,
            cast_free,
            peer_loaded,
            published,
            consumed,
            beta_sigmoid,
            save_cache,
            save_norm,
            save_beta,
            raw_q,
            raw_k,
            False,
            mode == 1 and prefetch,
            True,
        )
        if tla.const_expr(mode == 1):
            with tla.vector():
                pack_persistent_panel(
                    uq,
                    uk,
                    ug,
                    packed,
                    0,
                    scale,
                    raw_q,
                    raw_k,
                    inverse_norm,
                    shared_scores,
                )
            if tla.const_expr(prefetch):
                prefetch_half(
                    flat(q.ptr + tla.Int64(64 * q_stride), 1),
                    flat(k.ptr + tla.Int64(64 * q_stride), 1),
                    raw_q,
                    raw_k,
                    following_valid,
                    q_stride,
                    0,
                    prefetch_free,
                )
            with tla.vector():
                aiv_next = tla.arch.sub_block_idx()
                pack_persistent_panel(
                    uq,
                    uk,
                    ug,
                    packed,
                    1,
                    scale,
                    raw_q,
                    raw_k,
                    inverse_norm,
                    shared_scores,
                )
                mutex_event(_mutex_v_done, tla.pipes.VECTOR)
                mutex_event(_mutex_v_done, tla.pipes.MTE3)
                tla.copy(
                    tla.tile_view(
                        score_l1(lp, shared_scores),
                        tla.make_shape(score_rows, 64),
                        tla.make_coord(0, aiv_next),
                    ),
                    padded_nz(packed.ptr, score_rows),
                )
                tla.cross_core_set_flag(scores_ready, tla.pipes.MTE3)
                if tla.const_expr(save_norm):
                    if norm_owner:
                        tla.copy(
                            strided_rows(
                                qn.ptr + tla.Int64(64 * norm_stride) + aiv_next * 64,
                                next_valid,
                                64,
                                norm_stride,
                            ),
                            row_major(uq.ptr, next_valid, 64),
                        )
                        tla.copy(
                            strided_rows(
                                kn.ptr + tla.Int64(64 * norm_stride) + aiv_next * 64,
                                next_valid,
                                64,
                                norm_stride,
                            ),
                            row_major(uk.ptr, next_valid, 64),
                        )
            if tla.const_expr(prefetch):
                prefetch_half(
                    flat(q.ptr + tla.Int64(64 * q_stride), 1),
                    flat(k.ptr + tla.Int64(64 * q_stride), 1),
                    raw_q,
                    raw_k,
                    following_valid,
                    q_stride,
                    1,
                    prefetch_free,
                )
    with tla.vector():
        aiv_tail = tla.arch.sub_block_idx()
        if tla.const_expr(not bulk_operands):
            # API-062: no scaled FIX to L1. Negate K_beta exactly before BF16
            # packing, so Cube produces -W; the public W FIX reverses the sign.
            beta_weighted_operand(uk, ub, ug, cast, True, True)
            nd2nz_tile(cast, nz, 64, 64)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            tla.copy(
                tla.tile_view(kl, tla.make_shape(64, 64), tla.make_coord(0, aiv_tail)),
                padded_nz(nz.ptr, 64),
            )
            tla.cross_core_set_flag(k_ready, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
            beta_weighted_operand(uv, ub, ug, cast, False)
            nd2nz_tile(cast, nz, 64, 64)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            tla.copy(
                tla.tile_view(vl, tla.make_shape(64, 64), tla.make_coord(0, aiv_tail)),
                padded_nz(nz.ptr, 64),
            )
            tla.cross_core_set_flag(v_ready, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
            prepare_decayed_q(uq, ug, cast, decayed, scale)
            nd2nz_tile(decayed, nz, 64, 64)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            if tla.const_expr(save_cache):
                tla.copy(
                    strided_rows(qg.ptr + aiv_tail * 64, valid, 64, v_stride),
                    row_major(cast.ptr, valid, 64),
                )
            tla.copy(
                tla.tile_view(ql, tla.make_shape(64, 64), tla.make_coord(0, aiv_tail)),
                padded_nz(nz.ptr, 64),
            )
            mutex_event(_mutex_cast_free, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
            restored_state_inputs(uk, ug, cast, ugamma)
            nd2nz_tile(cast, nz, 64, 64)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            if tla.const_expr(save_cache):
                tla.copy(
                    strided_rows(kg.ptr + aiv_tail * 64, valid, 64, v_stride),
                    row_major(cast.ptr, valid, 64),
                )
            tla.copy(
                tla.tile_view(kgl, tla.make_shape(64, 64), tla.make_coord(0, aiv_tail)),
                padded_nz(nz.ptr, 64),
            )
            mutex_event(_mutex_cast_free, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
        if tla.const_expr(not bulk_operands):
            mask_mqk_nz(mq, public, aq_nz, aiv_tail, save_matrices)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            rows = valid - aiv_tail * 32
            if rows > 32:
                rows = 32
            if tla.const_expr(save_matrices):
                if rows > 0:
                    tla.copy(
                        row_major(aqk.ptr + aiv_tail * 2048, rows, 64),
                        row_major(public.ptr, rows, 64),
                    )
            tla.copy(
                tla.tile_view(al, tla.make_shape(32, 64), tla.make_coord(aiv_tail, 0)),
                padded_nz(aq_nz.ptr, 32),
            )
            tla.cross_core_set_flag(recur_ready, tla.pipes.MTE3)
    if next_valid > 0:
        next_recurrent = recurrent_l1(lp + 65536 + (1 - parity) * 32768)
        next_gamma = flat(ubase + 43616 + (1 - parity) * 64, 64)
        with tla.vector():
            aiv_pack = tla.arch.sub_block_idx()
            mutex_event(_mutex_cast_free, tla.pipes.MTE3)
            mutex_event(_mutex_cast_free, tla.pipes.VECTOR)
            pack_recurrent(
                uq,
                uk,
                uv,
                ug,
                ub,
                recurrent_nz,
                next_gamma,
                saved_qg,
                saved_kg,
                scale,
                save_cache,
            )
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            tla.copy(
                tla.tile_view(
                    next_recurrent, tla.make_shape(256, 64), tla.make_coord(0, aiv_pack)
                ),
                padded_nz(recurrent_nz.ptr, 256),
            )
            if parity == 0:
                tla.cross_core_set_flag(v_ready, tla.pipes.MTE3)
            else:
                tla.cross_core_set_flag(k_ready, tla.pipes.MTE3)
            if tla.const_expr(save_cache):
                tla.copy(
                    strided_rows(
                        qg.ptr + tla.Int64(64 * v_stride) + aiv_pack * 64,
                        next_valid,
                        64,
                        v_stride,
                    ),
                    row_major(saved_qg.ptr, next_valid, 64),
                )
                tla.copy(
                    strided_rows(
                        kg.ptr + tla.Int64(64 * v_stride) + aiv_pack * 64,
                        next_valid,
                        64,
                        v_stride,
                    ),
                    row_major(saved_kg.ptr, next_valid, 64),
                )
    with tla.vector():
        tla.cross_core_wait_flag(delta_ready, tla.pipes.VECTOR)
        if next_valid > 0:
            aiv_update = tla.arch.sub_block_idx()
            # cast_free before next recurrent packing has released score's
            # MTE3 read of the SNZ alias; the recurrent buffer is disjoint.
            update_state_nz(state, delta, ugamma, snz, 128)
            mutex_event(_mutex_v_done, tla.pipes.VECTOR)
            mutex_event(_mutex_v_done, tla.pipes.MTE3)
            tla.copy(
                tla.tile_view(
                    hl, tla.make_shape(128, 64), tla.make_coord(0, aiv_update)
                ),
                padded_nz(snz.ptr, 128),
            )
            tla.cross_core_set_flag(h_ready, tla.pipes.MTE3)
        else:
            scale_and_add_delta(state, delta, ugamma, 128)


# -----------------------------------------------------------------------------
# Persistent kernel and task scheduler.
# -----------------------------------------------------------------------------


class PersistentMetadataKernel:
    def __init__(
        self,
        batch,
        length,
        nq,
        nv,
        layout,
        mode=1,
        negative=False,
        lower=-0.1,
        scale=128**-0.5,
        normalize=True,
        beta_sigmoid=True,
        save_matrices=True,
        save_cache=True,
        save_history=True,
        save_norm=True,
        save_beta=True,
        prefetch=False,
        bulk_operands=False,
        pipeline=False,
    ):
        self.batch, self.length, self.nq, self.nv, self.layout = (
            batch,
            length,
            nq,
            nv,
            layout,
        )
        self.mode, self.negative, self.lower, self.scale = mode, negative, lower, scale
        self.normalize, self.beta_sigmoid = normalize, beta_sigmoid
        self.save_matrices, self.save_cache = save_matrices, save_cache
        self.save_history, self.save_norm, self.save_beta = (
            save_history,
            save_norm,
            save_beta,
        )
        self.prefetch = prefetch
        self.bulk_operands = bulk_operands
        self.pipeline = pipeline
        self.shared_scores = (
            not save_matrices
            and normalize
            and mode == 1
            and not negative
            and lower == -5
            and abs(scale) <= 0.125
        )
        self.chunk_capacity = (length + 63) // 64
        self.token_stride = 128 if layout == "BNSD" else nv * 128
        self.q_stride = 128 if layout == "BNSD" else nq * 128
        self.beta_stride = 1 if layout == "BNSD" else nv
        self.norm_stride = nq * 128 if layout == "TND" else 128
        self.rstd_stride = nq if layout == "TND" else 1
        self.rstd_k_offset = (1 if layout == "TND" else batch) * nq * length

    @tla.jit
    def __call__(
        self,
        q,
        k,
        v,
        g,
        beta,
        alpha,
        bias,
        exchange,
        identity16,
        initial,
        meta,
        aqk,
        akk,
        w,
        u,
        qg,
        kg,
        qs,
        gamma,
        qn,
        kn,
        rstd,
        gc,
        bact,
        out,
        vnew,
        history,
        final,
    ):
        self.run(
            q,
            k,
            v,
            g,
            beta,
            alpha,
            bias,
            exchange,
            identity16,
            initial,
            meta,
            aqk,
            akk,
            w,
            u,
            qg,
            kg,
            qs,
            gamma,
            qn,
            kn,
            rstd,
            gc,
            bact,
            out,
            vnew,
            history,
            final,
        )

    @tla.kernel
    def run(
        self,
        q,
        k,
        v,
        g,
        beta,
        alpha,
        bias,
        exchange,
        identity16,
        initial,
        meta,
        aqk,
        akk,
        w,
        u,
        qg,
        kg,
        qs,
        gamma,
        qn,
        kn,
        rstd,
        gc,
        bact,
        out,
        vnew,
        history,
        final,
    ):
        _mutex_load_state = tla.mutex("run_6449_load_state", id=0)
        _mutex_save_state = tla.mutex("run_6449_save_state", id=1)
        ub = tla.allocate(61440, tla.Float32, tla.AddressSpace.ub, 256)
        l1 = tla.allocate(
            172032 if self.pipeline else 131072, tla.BFloat16, tla.AddressSpace.l1, 512
        )
        a = tla.allocate(32768, tla.BFloat16, tla.AddressSpace.l0a, 512)
        b = tla.allocate(32768, tla.BFloat16, tla.AddressSpace.l0b, 512)
        c = tla.allocate(32768, tla.Float32, tla.AddressSpace.l0c, 1024)
        load_state = tla.flag("ps_load_state", tla.pipes.MTE2, tla.pipes.VECTOR)
        save_state = tla.flag("ps_save_state", tla.pipes.VECTOR, tla.pipes.MTE3)
        block = i64(tla.arch.block_idx())
        task = meta[i64(11) + block]
        while task != i64(0):
            logical_batch = meta[task]
            storage_batch = meta[task + i64(1)]
            token_start = meta[task + i64(2)]
            length = meta[task + i64(3)]
            head = meta[task + i64(4)]
            qhead = head // i64(self.nv // self.nq)
            si = (logical_batch * i64(self.nv) + head) * i64(16384)
            total = ((length + i64(63)) // i64(64)).to(tla.Int32)
            with tla.vector():
                aiv = tla.arch.sub_block_idx()
                tla.copy(
                    row_major(ub, 128, 64),
                    strided_rows(initial.ptr + si + i64(aiv * 64), 128, 64, 128),
                )
                mutex_event(_mutex_load_state, tla.pipes.MTE2)
                mutex_event(_mutex_load_state, tla.pipes.VECTOR)
            for chunk in tla.range(total):
                ci = i64(chunk)
                valid = i64(64)
                if (ci + i64(1)) * i64(64) > length:
                    valid = length - ci * i64(64)
                token = token_start + ci * i64(64)
                qo = token_offset(
                    storage_batch, qhead, token, self.nq, self.length, self.layout
                )
                vo = token_offset(
                    storage_batch, head, token, self.nv, self.length, self.layout
                )
                bo = vo // i64(128)
                no = (
                    (storage_batch * i64(self.nq) + qhead) * i64(self.length) + token
                ) * i64(128)
                if tla.const_expr(self.layout == "TND"):
                    no = (token * i64(self.nq) + qhead) * i64(128)
                chunk_slot = logical_batch * i64(self.chunk_capacity) + ci
                if tla.const_expr(self.layout == "TND"):
                    chunk_slot = token_start // i64(64) + logical_batch + ci
                co = chunk_slot * i64(self.nv) + head
                first_chunk = True
                next_valid = tla.Int32(0)
                if tla.const_expr(self.prefetch):
                    first_chunk = chunk == 0
                    if chunk + 1 < total:
                        next_valid = (length - (ci + i64(1)) * i64(64)).to(tla.Int32)
                        if next_valid > 64:
                            next_valid = 64
                following_valid = tla.Int32(0)
                if chunk + 2 < total:
                    following_valid = (length - (ci + i64(2)) * i64(64)).to(tla.Int32)
                    if following_valid > 64:
                        following_valid = 64
                if tla.const_expr(self.pipeline):
                    persistent_pipeline_chunk(
                        ub,
                        l1,
                        a,
                        b,
                        c,
                        flat(q.ptr + qo, 1),
                        flat(k.ptr + qo, 1),
                        flat(v.ptr + vo, 1),
                        flat(g.ptr + vo, 1),
                        flat(beta.ptr + bo, 1),
                        flat(alpha.ptr + head, 1),
                        flat(bias.ptr + head * i64(128), 128),
                        flat(exchange.ptr + block * i64(256), 256),
                        identity16,
                        flat(aqk.ptr + co * i64(4096), 4096),
                        flat(akk.ptr + co * i64(4096), 4096),
                        flat(w.ptr + vo, 1),
                        flat(u.ptr + vo, 1),
                        flat(qg.ptr + vo, 1),
                        flat(kg.ptr + vo, 1),
                        flat(qs.ptr + block * i64(8192), 8192),
                        flat(gamma.ptr + block * i64(128), 128),
                        flat(qn.ptr + no, 1),
                        flat(kn.ptr + no, 1),
                        flat(rstd.ptr + no // i64(128), 1),
                        flat(gc.ptr + co * i64(8192), 8192),
                        flat(
                            bact.ptr
                            + (storage_batch * i64(self.length) + token) * i64(self.nv)
                            + head,
                            1,
                        ),
                        flat(out.ptr + vo, 1),
                        flat(vnew.ptr + vo, 1),
                        flat(history.ptr + co * i64(16384), 16384),
                        valid.to(tla.Int32),
                        self.mode,
                        self.lower,
                        self.negative,
                        self.normalize,
                        self.scale,
                        self.q_stride,
                        self.token_stride,
                        self.beta_stride,
                        self.norm_stride,
                        self.nv,
                        head % i64(self.nv // self.nq) == i64(0),
                        self.rstd_k_offset,
                        self.rstd_stride,
                        self.beta_sigmoid,
                        self.save_matrices,
                        self.save_cache,
                        self.save_history,
                        self.save_norm,
                        self.save_beta,
                        self.prefetch,
                        first_chunk,
                        next_valid,
                        self.bulk_operands,
                        following_valid,
                        chunk % 2,
                        self.nv,
                        self.shared_scores,
                    )
                else:
                    persistent_chunk(
                        ub,
                        l1,
                        a,
                        b,
                        c,
                        flat(q.ptr + qo, 1),
                        flat(k.ptr + qo, 1),
                        flat(v.ptr + vo, 1),
                        flat(g.ptr + vo, 1),
                        flat(beta.ptr + bo, 1),
                        flat(alpha.ptr + head, 1),
                        flat(bias.ptr + head * i64(128), 128),
                        flat(exchange.ptr + block * i64(256), 256),
                        identity16,
                        flat(aqk.ptr + co * i64(4096), 4096),
                        flat(akk.ptr + co * i64(4096), 4096),
                        flat(w.ptr + vo, 1),
                        flat(u.ptr + vo, 1),
                        flat(qg.ptr + vo, 1),
                        flat(kg.ptr + vo, 1),
                        flat(qs.ptr + block * i64(8192), 8192),
                        flat(gamma.ptr + block * i64(128), 128),
                        flat(qn.ptr + no, 1),
                        flat(kn.ptr + no, 1),
                        flat(rstd.ptr + no // i64(128), 1),
                        flat(gc.ptr + co * i64(8192), 8192),
                        flat(
                            bact.ptr
                            + (storage_batch * i64(self.length) + token) * i64(self.nv)
                            + head,
                            1,
                        ),
                        flat(out.ptr + vo, 1),
                        flat(vnew.ptr + vo, 1),
                        flat(history.ptr + co * i64(16384), 16384),
                        valid.to(tla.Int32),
                        self.mode,
                        self.lower,
                        self.negative,
                        self.normalize,
                        self.scale,
                        self.q_stride,
                        self.token_stride,
                        self.beta_stride,
                        self.norm_stride,
                        self.nv,
                        head % i64(self.nv // self.nq) == i64(0),
                        self.rstd_k_offset,
                        self.rstd_stride,
                        self.beta_sigmoid,
                        self.save_matrices,
                        self.save_cache,
                        self.save_history,
                        self.save_norm,
                        self.save_beta,
                        self.prefetch,
                        first_chunk,
                        next_valid,
                        self.bulk_operands,
                    )
            with tla.vector():
                aiv = tla.arch.sub_block_idx()
                mutex_event(_mutex_save_state, tla.pipes.VECTOR)
                mutex_event(_mutex_save_state, tla.pipes.MTE3)
                tla.copy(
                    strided_rows(final.ptr + si + i64(aiv * 64), 128, 64, 128),
                    row_major(ub, 128, 64),
                )
                tla.pipe_barrier(tla.pipes.ALL)
            with tla.cube():
                tla.pipe_barrier(tla.pipes.ALL)
            task = meta[task + i64(5)]


# =============================================================================
# Host side: parameter checks, ABI4 integer scheduling, allocation and launch.
# =============================================================================

# Ascend 950 native CATLASS DSL Flash KDA forward host interface.
#
# Input validation, ABI4 integer scheduling and forward launch live here.
# See README.md. Device arithmetic lives in the section above.

__all__ = ["flash_kda_fwd"]


# -----------------------------------------------------------------------------
# Host input validation.
# -----------------------------------------------------------------------------

D = 128


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _tensor(name, value, dtype, shape=None, rank=None):
    _require(isinstance(value, Tensor), f"{name} must be a Tensor")
    _require(value.device.type == "npu", f"{name} must be on NPU")
    _require(value.dtype in dtype, f"{name}: unsupported dtype {value.dtype}")
    _require(value.is_contiguous(), f"{name} must be contiguous")
    if rank is not None:
        _require(value.ndim == rank, f"{name}: expected rank {rank}")
    if shape is not None:
        _require(tuple(value.shape) == tuple(shape), f"{name}: expected shape {shape}")


def _validate(
    q,
    k,
    v,
    g,
    beta,
    metadata,
    initial_state,
    A_log,
    dt_bias,
    cu_seqlens,
    seqused,
    *,
    scale,
    layout,
    flags,
    lower_bound,
):
    _require(layout in ("BNSD", "BSND", "TND"), "unsupported layout")
    _require(metadata is None, "metadata must be None in this interface version")
    for name, flag in flags.items():
        _require(isinstance(flag, bool), f"{name} must be bool")
    _require(
        scale is None
        or (
            isinstance(scale, (int, float))
            and not isinstance(scale, bool)
            and math.isfinite(scale)
        ),
        "scale must be finite",
    )
    rank = 3 if layout == "TND" else 4
    for name, x in (("q", q), ("k", k), ("v", v)):
        _tensor(name, x, (torch.bfloat16,), rank=rank)
    _tensor("g", g, (torch.bfloat16, torch.float32), v.shape)
    _tensor("beta", beta, (torch.bfloat16, torch.float32), v.shape[:-1])
    _require(k.shape == q.shape, "q/k shapes must match")
    _require(q.shape[-1] == D and v.shape[-1] == D, "D must be 128")
    if layout == "TND":
        _tensor("cu_seqlens", cu_seqlens, (torch.int32,), rank=1)
        batch, length, nq, nv = (
            cu_seqlens.numel() - 1,
            q.shape[0],
            q.shape[1],
            v.shape[1],
        )
        expected_v = (length, nv, D)
    else:
        batch = q.shape[0]
        length = q.shape[2 if layout == "BNSD" else 1]
        nq, nv = (
            q.shape[1 if layout == "BNSD" else 2],
            v.shape[1 if layout == "BNSD" else 2],
        )
        expected_v = (
            (batch, nv, length, D) if layout == "BNSD" else (batch, length, nv, D)
        )
    _require(batch > 0 and length > 0, "batch and storage length must be positive")
    _require(1 <= nq <= nv <= 96 and nv % nq == 0, "invalid GQA head counts")
    _require(tuple(v.shape) == expected_v, "v shape does not match q")
    lengths = None
    if cu_seqlens is not None:
        _tensor("cu_seqlens", cu_seqlens, (torch.int32,), (batch + 1,))
        prefix = cu_seqlens.cpu().tolist()
        _require(prefix[0] == 0, "cu_seqlens must start at 0")
        lengths = [b - a for a, b in zip(prefix, prefix[1:])]
        _require(all(0 < x <= length for x in lengths), "invalid sequence lengths")
        if layout == "TND":
            _require(prefix[-1] == length, "cu_seqlens endpoint must equal T")
    if seqused is not None:
        _tensor("seqused", seqused, (torch.int32,), (batch,))
        used = seqused.cpu().tolist()
        _require(all(0 < x <= length for x in used), "invalid seqused")
        _require(lengths is None or lengths == used, "inconsistent length tensors")
        lengths = used
    lengths = lengths if lengths is not None else [length] * batch
    _require(sum(lengths) <= 2**31 - 1, "length prefix exceeds INT32")
    if initial_state is not None:
        _tensor("initial_state", initial_state, (torch.float32,), (batch, nv, D, D))
    if flags["use_gate_in_kernel"]:
        _tensor("A_log", A_log, (torch.float32,), (nv,))
        if dt_bias is not None:
            _tensor("dt_bias", dt_bias, (torch.float32,), (nv, D))
        _require(
            lower_bound is None
            or (
                isinstance(lower_bound, (int, float))
                and not isinstance(lower_bound, bool)
                and math.isfinite(lower_bound)
                and -5 <= lower_bound < 0
            ),
            "invalid lower_bound",
        )
    _require(
        not flags["allow_neg_eigval"] or flags["use_beta_sigmoid_in_kernel"],
        "allow_neg_eigval requires beta sigmoid",
    )
    for name, tensor in (
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("initial_state", initial_state),
        ("A_log", A_log if flags["use_gate_in_kernel"] else None),
        ("dt_bias", dt_bias if flags["use_gate_in_kernel"] else None),
        ("cu_seqlens", cu_seqlens),
        ("seqused", seqused),
    ):
        if tensor is not None:
            _require(
                isinstance(tensor, Tensor) and tensor.device == q.device,
                f"{name} must be on the same NPU as q",
            )
    return batch, nq, nv, length, lengths


# -----------------------------------------------------------------------------
# Host ABI4 scheduling.
# -----------------------------------------------------------------------------

CHUNK_SIZE = 64
MAX_VALUE_HEADS = 96
WORKSPACE_SLOTS = 1024
MAX_CHUNK_GROUP = 512
HEADER_WORDS = 11
STATUS_OK = 0
I64 = int


def zeros(dtype, count):
    return [0] * count


MAGIC = 0x4B444132

ABI_VERSION = 4

MAX_AIC_CORES = 32

ROUND_FIXED_WORDS = 133

ROUND_WORDS_PER_BATCH = 9

HEADER_MAGIC = 0

HEADER_ABI_VERSION = 1

HEADER_STATUS = 2

HEADER_CORE_NUM = 3

HEADER_STAGE12_ROUND_NUM = 4

HEADER_BATCH = 5

HEADER_VALUE_HEADS = 6

HEADER_HAS_CU = 7

HEADER_WORKSPACE_SLOTS = 8

HEADER_TEMPLATE = 9

HEADER_CHUNK_TASK_NUM = 10

TEMPLATE_AUTO = 0

TEMPLATE_STAGED = 1

TEMPLATE_PERSISTENT = 2

CHUNK_TASK_WORDS = 6

CHUNK_TASK_BATCH = 0

CHUNK_TASK_STORAGE_BATCH = 1

CHUNK_TASK_TOKEN_START = 2

CHUNK_TASK_LENGTH = 3

CHUNK_TASK_VALUE_HEAD = 4

CHUNK_TASK_NEXT = 5

STATUS_BAD_CORE_NUM = 1

STATUS_BAD_WORKSPACE_SLOTS = 2

STATUS_BAD_FLAGS = 3

STATUS_BAD_SHAPE = 4

STATUS_BAD_CU_START = 5

STATUS_BAD_LENGTH = 6

STATUS_BAD_PADDED_LENGTH = 7

STATUS_BAD_PACKED_END = 8

STATUS_BAD_METADATA_CAPACITY = 9

SHORT_TRAIN_CHUNK_GROUP = 13

SHORT_TRAIN_MIN_LENGTH = 8192

SHORT_TRAIN_HEADS = 32

INT64_MAX = 2**63 - 1

ROUTE_COST_SCALE = 64

MAX_SCALED_QUANTITY = INT64_MAX // ROUTE_COST_SCALE

ROUTE_PARAMETER_WORDS = 16

ROUTE_SCHEMA_INDEX = 13

ROUTE_MARGIN_INDEX = 14

PARTS_PER_MILLION = 1_000_000

PICOSECONDS_PER_NANOSECOND = 1000

FULL_CHUNK_COST_NS = 4930

CALIBRATED_MIN_CHUNKS = 64

CALIBRATED_MAX_CHUNKS = 256

CALIBRATED_GROUP_LIMIT = 16

MAX_ACTIVE_BATCHES = WORKSPACE_SLOTS

MAX_STAGE1_TASKS = WORKSPACE_SLOTS

MAX_STAGE2_TASKS = WORKSPACE_SLOTS * 8


def _interpolate_stage1_task_cost(
    valid_row_num,
    left_row_num,
    left_cost,
    right_row_num,
    right_cost,
):
    # 拟合系数单位为皮秒，调度成本单位为纳秒。
    row_span = right_row_num - left_row_num
    numerator = left_cost * row_span + (valid_row_num - left_row_num) * (
        right_cost - left_cost
    )
    denominator = row_span * PICOSECONDS_PER_NANOSECOND
    return (numerator + denominator // 2) // denominator


def _stage1_task_cost(valid_row_num):
    cost = 0
    if valid_row_num > 0:
        if valid_row_num <= 16:
            cost = _interpolate_stage1_task_cost(valid_row_num, 1, 4892380, 16, 4920438)
        elif valid_row_num <= 32:
            cost = _interpolate_stage1_task_cost(
                valid_row_num, 16, 4920438, 32, 4970718
            )
        elif valid_row_num <= 48:
            cost = _interpolate_stage1_task_cost(
                valid_row_num, 32, 4970718, 48, 4992347
            )
        elif valid_row_num <= 63:
            cost = _interpolate_stage1_task_cost(
                valid_row_num, 48, 4992347, 63, 4952972
            )
        else:
            cost = FULL_CHUNK_COST_NS
    return cost


def _stage2_task_cost(candidate_idx, chunk_num):
    item_init = 352
    chunk_cost = 1839
    if candidate_idx == 1:
        item_init = 133
        chunk_cost = 1186
    elif candidate_idx == 2:
        item_init = 19
        chunk_cost = 884
    elif candidate_idx == 3:
        item_init = 65
        chunk_cost = 729
    return item_init + chunk_num * chunk_cost


def round_record_words(active_batch_num: int) -> int:
    """返回一条变长记录占用的字数。"""

    active_batch_num = int(active_batch_num)
    if active_batch_num <= 0:
        raise ValueError("active_batch_num must be positive")
    return ROUND_FIXED_WORDS + ROUND_WORDS_PER_BATCH * active_batch_num


def metadata_capacity_upper_bound(
    batch: int,
    storage_length: int,
    *,
    is_packed: bool,
    value_heads: int = MAX_VALUE_HEADS,
) -> int:
    """返回覆盖所有合法轮次划分的安全容量上界。"""

    batch = int(batch)
    storage_length = int(storage_length)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if storage_length <= 0:
        raise ValueError("storage_length must be positive")
    storage_chunk_num = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    if is_packed:
        chunk_entries = storage_chunk_num + batch - 1
    else:
        chunk_entries = batch * storage_chunk_num
    if not 1 <= int(value_heads) <= MAX_VALUE_HEADS:
        raise ValueError("value_heads must be in [1, 96]")
    # 每个可能的 chunk 条目预留一个目录字和一条单批次记录。
    group_capacity = (
        HEADER_WORDS
        + 1
        + (1 + ROUND_FIXED_WORDS + ROUND_WORDS_PER_BATCH) * chunk_entries
    )
    # chunk 调度器在任务记录后使用 B 个临时批次索引。
    chunk_capacity = (
        HEADER_WORDS
        + MAX_AIC_CORES
        + CHUNK_TASK_WORDS * batch * int(value_heads)
        + batch
    )
    return max(group_capacity, chunk_capacity)


def _select_group_num(
    remaining_chunk_num,
    active_batch_num,
    value_heads,
    workspace_slots,
    max_chunk_group,
):
    max_remaining_chunk_num = 0
    for active_idx in range(0, active_batch_num):
        if remaining_chunk_num[active_idx] > max_remaining_chunk_num:
            max_remaining_chunk_num = remaining_chunk_num[active_idx]

    selected_group_num = 1
    lower_group_num = 1
    upper_group_num = max_remaining_chunk_num
    if upper_group_num > max_chunk_group:
        upper_group_num = max_chunk_group
    workspace_group_limit = workspace_slots // value_heads
    if upper_group_num > workspace_group_limit:
        upper_group_num = workspace_group_limit
    # 槽位用量随组大小单调增长。选取最大可行分组，减少长序列跨越全局阶段屏障的次数。
    while lower_group_num <= upper_group_num:
        candidate_group_num = (lower_group_num + upper_group_num) // 2
        slot_num = 0
        for active_idx in range(0, active_batch_num):
            current_chunk_num = remaining_chunk_num[active_idx]
            if current_chunk_num > candidate_group_num:
                current_chunk_num = candidate_group_num
            slot_num += value_heads * current_chunk_num
        if slot_num <= workspace_slots:
            selected_group_num = candidate_group_num
            lower_group_num = candidate_group_num + 1
        else:
            upper_group_num = candidate_group_num - 1
    return selected_group_num


def _partition_costs(
    task_costs,
    task_num,
    core_num,
    core_ranges,
    core_costs,
):
    for core_idx in range(0, MAX_AIC_CORES):
        core_ranges[core_idx * 2] = 0
        core_ranges[core_idx * 2 + 1] = 0
        core_costs[core_idx] = 0

    if task_num > 0:
        active_core_num = task_num
        if active_core_num > core_num:
            active_core_num = core_num
        remaining_cost = 0
        for task_idx in range(0, task_num):
            remaining_cost += task_costs[task_idx]

        start = 0
        for core_idx in range(0, active_core_num):
            remaining_core_num = active_core_num - core_idx
            if remaining_core_num == 1:
                end = task_num
            else:
                max_end = task_num - (remaining_core_num - 1)
                end = start + 1
                segment_cost = task_costs[start]
                while end < max_end:
                    next_cost = segment_cost + task_costs[end]
                    next_error = abs(next_cost * remaining_core_num - remaining_cost)
                    current_error = abs(
                        segment_cost * remaining_core_num - remaining_cost
                    )
                    if next_error <= current_error:
                        segment_cost = next_cost
                        end += 1
                    else:
                        break

            segment_cost = 0
            for task_idx in range(start, end):
                segment_cost += task_costs[task_idx]
            core_ranges[core_idx * 2] = start
            core_ranges[core_idx * 2 + 1] = end
            core_costs[core_idx] = segment_cost
            remaining_cost -= segment_cost
            start = end


def _chunk_batch_precedes(cu_seqlens, left_batch, right_batch):
    left_length = cu_seqlens[left_batch + 1] - cu_seqlens[left_batch]
    right_length = cu_seqlens[right_batch + 1] - cu_seqlens[right_batch]
    left_chunks = (left_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    right_chunks = (right_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    precedes = 0
    if left_chunks > right_chunks:
        precedes = 1
    elif left_chunks == right_chunks and left_batch < right_batch:
        precedes = 1
    return precedes


def _chunk_heap_sift_down(cu_seqlens, metadata, heap_base, heap_num, root_idx):
    # 最大堆先按 chunk 数排序，再按原始批次索引排序。
    while root_idx * 2 + 1 < heap_num:
        child_idx = root_idx * 2 + 1
        if child_idx + 1 < heap_num:
            if (
                _chunk_batch_precedes(
                    cu_seqlens,
                    metadata[heap_base + child_idx + 1],
                    metadata[heap_base + child_idx],
                )
                == 1
            ):
                child_idx += 1
        root_batch = metadata[heap_base + root_idx]
        child_batch = metadata[heap_base + child_idx]
        if _chunk_batch_precedes(cu_seqlens, child_batch, root_batch) == 0:
            break
        metadata[heap_base + root_idx] = child_batch
        metadata[heap_base + child_idx] = root_batch
        root_idx = child_idx


def _training_group_limit(schedule_args, max_chunks):
    requested = schedule_args.max_chunk_group
    if schedule_args.calibration_enabled == 0:
        return requested
    if (
        schedule_args.layout_kind == 1
        and schedule_args.core_num == MAX_AIC_CORES
        and schedule_args.batch == 1
        and schedule_args.value_heads > 32
        and schedule_args.value_heads <= MAX_VALUE_HEADS
        and max_chunks >= CALIBRATED_MIN_CHUNKS
        and max_chunks <= CALIBRATED_MAX_CHUNKS
    ):
        return min(requested, CALIBRATED_GROUP_LIMIT)
    return requested


def _route_add_cost(time, cost, quantity):
    if time < 0 or cost < 0 or quantity < 0:
        return -1
    if quantity > 0 and cost > (INT64_MAX - time) // quantity:
        return -1
    return time + cost * quantity


def _route_rounds(schedule_args, max_group, remaining_chunks):
    rounds = 0
    batch_start = 0
    batches_per_set = schedule_args.workspace_slots // schedule_args.value_heads
    while batch_start < schedule_args.batch:
        batch_count = min(batches_per_set, schedule_args.batch - batch_start)
        longest = 0
        for local_batch in range(batch_count):
            batch = batch_start + local_batch
            length = schedule_args.storage_length
            if schedule_args.has_cu == 1:
                length = (
                    schedule_args.cu_seqlens[batch + 1]
                    - schedule_args.cu_seqlens[batch]
                )
            chunks = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
            remaining_chunks[local_batch] = chunks
            longest = max(longest, chunks)
        while longest > 0:
            low = 1
            high = min(max_group, min(longest, batches_per_set))
            while low < high:
                middle = (low + high + 1) // 2
                chunk_count = 0
                for local_batch in range(batch_count):
                    chunk_count += min(remaining_chunks[local_batch], middle)
                if (
                    chunk_count
                    <= schedule_args.workspace_slots // schedule_args.value_heads
                ):
                    low = middle
                else:
                    high = middle - 1
            longest = 0
            for local_batch in range(batch_count):
                chunks = max(0, remaining_chunks[local_batch] - low)
                remaining_chunks[local_batch] = chunks
                longest = max(longest, chunks)
            rounds += 1
        batch_start += batch_count
    return rounds


def _route_persistent_loads(schedule_args, uniform_chunks, core_chunks, core_rows):
    # 使用与执行调度相同的堆及按 chunk 数排序的 LPT 顺序。
    task_count = schedule_args.batch * schedule_args.value_heads
    heap_base = HEADER_WORDS + MAX_AIC_CORES + CHUNK_TASK_WORDS * task_count
    if heap_base + schedule_args.batch > schedule_args.metadata_word_capacity:
        return -1
    heap_count = schedule_args.batch
    if uniform_chunks == 0:
        for batch in range(schedule_args.batch):
            schedule_args.metadata[heap_base + batch] = batch
        root = heap_count // 2
        while root > 0:
            root -= 1
            _chunk_heap_sift_down(
                schedule_args.cu_seqlens,
                schedule_args.metadata,
                heap_base,
                heap_count,
                root,
            )
    peak = 0
    for scheduled_batch in range(schedule_args.batch):
        batch = scheduled_batch
        if uniform_chunks == 0:
            batch = schedule_args.metadata[heap_base]
            heap_count -= 1
            if heap_count > 0:
                schedule_args.metadata[heap_base] = schedule_args.metadata[
                    heap_base + heap_count
                ]
                _chunk_heap_sift_down(
                    schedule_args.cu_seqlens,
                    schedule_args.metadata,
                    heap_base,
                    heap_count,
                    0,
                )
        length = schedule_args.storage_length
        if schedule_args.has_cu == 1:
            length = (
                schedule_args.cu_seqlens[batch + 1] - schedule_args.cu_seqlens[batch]
            )
        chunks = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
        for head in range(schedule_args.value_heads):
            selected_core = 0
            for core in range(1, schedule_args.core_num):
                if core_chunks[core] < core_chunks[selected_core]:
                    selected_core = core
            core_chunks[selected_core] += chunks
            core_rows[selected_core] += length
            peak = max(peak, core_chunks[selected_core])
    return peak


def _select_route_from_model(
    schedule_args,
    staged_peak,
    total_chunks,
    total_rows,
    query_rows,
    persistent_peak,
    core_rows,
    max_group,
    remaining_chunks,
):
    # 非负成本单位为飞秒，两种时间均乘以 64 后比较。
    if (
        staged_peak < 1
        or total_chunks < 1
        or total_rows < 1
        or query_rows < 1
        or persistent_peak < 1
    ):
        return TEMPLATE_STAGED
    parallel_chunks = (total_chunks - 1) // schedule_args.core_num + 1
    head_count = schedule_args.batch * schedule_args.value_heads
    rounds = _route_rounds(schedule_args, max_group, remaining_chunks)
    if (
        parallel_chunks > MAX_SCALED_QUANTITY
        or staged_peak > MAX_SCALED_QUANTITY
        or persistent_peak > MAX_SCALED_QUANTITY
        or rounds > MAX_SCALED_QUANTITY
        or head_count > MAX_SCALED_QUANTITY
    ):
        return TEMPLATE_STAGED
    base = schedule_args.layout_kind * ROUTE_PARAMETER_WORDS
    schema = schedule_args.route_coefficients[base + ROUTE_SCHEMA_INDEX]
    if schema != 6 and schema != 7:
        return TEMPLATE_STAGED
    for index in range(ROUTE_SCHEMA_INDEX):
        if schedule_args.route_coefficients[base + index] < 0:
            return TEMPLATE_STAGED
    staged = 0
    for index in range(5):
        quantity = ROUTE_COST_SCALE
        if index == 1:
            quantity = parallel_chunks * ROUTE_COST_SCALE
        elif index == 2:
            quantity = staged_peak * ROUTE_COST_SCALE
        elif index == 3:
            quantity = rounds * ROUTE_COST_SCALE
        elif index == 4:
            quantity = query_rows
        staged = _route_add_cost(
            staged, schedule_args.route_coefficients[base + index], quantity
        )
        if staged < 0:
            return TEMPLATE_STAGED
    compute = _route_add_cost(
        0,
        schedule_args.route_coefficients[base + 5],
        parallel_chunks * ROUTE_COST_SCALE,
    )
    memory = _route_add_cost(0, schedule_args.route_coefficients[base + 6], total_rows)
    if compute < 0 or memory < 0:
        return TEMPLATE_STAGED
    staged = _route_add_cost(staged, max(compute, memory), 1)
    if staged < 0:
        return TEMPLATE_STAGED
    persistent = 0
    for index in range(7, 11):
        quantity = ROUTE_COST_SCALE
        if index == 8:
            quantity = persistent_peak * ROUTE_COST_SCALE
        elif index == 9:
            quantity = head_count * ROUTE_COST_SCALE
        elif index == 10:
            quantity = query_rows
        persistent = _route_add_cost(
            persistent, schedule_args.route_coefficients[base + index], quantity
        )
        if persistent < 0:
            return TEMPLATE_STAGED
    previous_rows = 0
    for phase in range(schedule_args.core_num):
        next_rows = 0
        active_cores = 0
        for core in range(schedule_args.core_num):
            rows = core_rows[core]
            if rows > previous_rows:
                active_cores += 1
                if next_rows == 0 or rows < next_rows:
                    next_rows = rows
        if next_rows == 0:
            break
        memory = _route_add_cost(
            0, schedule_args.route_coefficients[base + 12], active_cores
        )
        if memory < 0:
            return TEMPLATE_STAGED
        cost = max(schedule_args.route_coefficients[base + 11], memory)
        persistent = _route_add_cost(persistent, cost, next_rows - previous_rows)
        if persistent < 0:
            return TEMPLATE_STAGED
        previous_rows = next_rows
    margin = schedule_args.route_coefficients[base + ROUTE_MARGIN_INDEX]
    if margin < 0 or margin >= PARTS_PER_MILLION:
        return TEMPLATE_STAGED
    required_gain = (staged // PARTS_PER_MILLION) * margin + (
        staged % PARTS_PER_MILLION
    ) * margin // PARTS_PER_MILLION
    if persistent < staged - required_gain:
        return TEMPLATE_PERSISTENT
    return TEMPLATE_STAGED


def flash_kda_fwd_metadata_kernel(a):
    # Source annotation is postponed and never evaluated; callers pass SimpleNamespace.
    valid_seq_len_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    token_start_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    chunk_num_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    tail_row_num_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    active_local_batch_idx = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    remaining_chunk_num = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    chunk_num_per_group = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    stage1_task_costs = zeros(I64, MAX_STAGE1_TASKS)  # noqa: F821 - AICPU 编译器内建函数。
    stage1_core_ranges = zeros(I64, MAX_AIC_CORES * 2)  # noqa: F821 - AICPU 编译器内建函数。
    stage1_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    stage2_task_costs = zeros(I64, MAX_STAGE2_TASKS)  # noqa: F821 - AICPU 编译器内建函数。
    candidate_stage2_core_ranges = zeros(I64, MAX_AIC_CORES * 2)  # noqa: F821 - AICPU 编译器内建函数。
    candidate_stage2_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    selected_stage2_core_ranges = zeros(I64, MAX_AIC_CORES * 2)  # noqa: F821 - AICPU 编译器内建函数。
    route_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    route_lpt_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    route_valid_rows = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    chunk_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    chunk_core_tails = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。

    if a.metadata_word_capacity < HEADER_WORDS:
        # AICPU 将非零内核返回值视为任务异常（ACL 507005）。
        # 诊断码写入 metadata 头部，不作为内核返回值。
        return STATUS_OK

    a.metadata[HEADER_MAGIC] = MAGIC
    a.metadata[HEADER_ABI_VERSION] = ABI_VERSION
    a.metadata[HEADER_STATUS] = STATUS_OK
    a.metadata[HEADER_CORE_NUM] = a.core_num
    a.metadata[HEADER_STAGE12_ROUND_NUM] = 0
    a.metadata[HEADER_BATCH] = a.batch
    a.metadata[HEADER_VALUE_HEADS] = a.value_heads
    a.metadata[HEADER_HAS_CU] = a.has_cu
    a.metadata[HEADER_WORKSPACE_SLOTS] = a.workspace_slots
    a.metadata[HEADER_TEMPLATE] = TEMPLATE_AUTO
    a.metadata[HEADER_CHUNK_TASK_NUM] = 0
    if a.metadata_word_capacity > HEADER_WORDS:
        a.metadata[HEADER_WORDS] = HEADER_WORDS + 1

    status = STATUS_OK
    if a.core_num < 1 or a.core_num > MAX_AIC_CORES:
        status = STATUS_BAD_CORE_NUM
    elif (
        a.template_kind != TEMPLATE_AUTO
        and a.template_kind != TEMPLATE_STAGED
        and a.template_kind != TEMPLATE_PERSISTENT
    ):
        status = STATUS_BAD_FLAGS
    elif a.calibration_enabled != 0 and a.calibration_enabled != 1:
        status = STATUS_BAD_FLAGS
    elif a.layout_kind != 0 and a.layout_kind != 1:
        status = STATUS_BAD_FLAGS
    elif a.workspace_slots != WORKSPACE_SLOTS:
        status = STATUS_BAD_WORKSPACE_SLOTS
    elif a.max_chunk_group < 1 or a.max_chunk_group > MAX_CHUNK_GROUP:
        status = STATUS_BAD_FLAGS
    elif not (a.has_cu == 0 or a.has_cu == 1):
        status = STATUS_BAD_FLAGS
    elif not (a.is_packed == 0 or a.is_packed == 1):
        status = STATUS_BAD_FLAGS
    elif a.is_packed == 1 and a.has_cu == 0:
        status = STATUS_BAD_FLAGS
    elif a.batch < 1 or a.storage_length < 1 or a.value_heads < 1:
        status = STATUS_BAD_SHAPE
    elif a.value_heads > MAX_VALUE_HEADS:
        status = STATUS_BAD_SHAPE
    elif a.is_packed == 1 and a.physical_batches != 1:
        status = STATUS_BAD_SHAPE
    elif a.is_packed == 0 and a.physical_batches != a.batch:
        status = STATUS_BAD_SHAPE
    if status != STATUS_OK:
        a.metadata[HEADER_STATUS] = status
        return STATUS_OK

    if a.has_cu == 1:
        if a.cu_seqlens[0] != 0:
            a.metadata[HEADER_STATUS] = STATUS_BAD_CU_START
            return STATUS_OK

    # 发布选路结果或任务前，检查所有实际序列长度。
    uniform_chunk_num = 1
    first_chunk_num = 0
    route_max_chunk_num = 0
    route_total_chunk_num = 0
    route_peak_chunk_num = 0
    route_total_rows = 0
    route_query_rows = 0
    evaluate_route = 0
    if a.template_kind == TEMPLATE_AUTO and a.calibration_enabled == 1:
        evaluate_route = 1
    for batch_idx in range(0, a.batch):
        valid_seq_len = a.storage_length
        if a.has_cu == 1:
            valid_seq_len = a.cu_seqlens[batch_idx + 1] - a.cu_seqlens[batch_idx]
        if valid_seq_len <= 0:
            status = STATUS_BAD_LENGTH
        elif a.is_packed == 0 and valid_seq_len > a.storage_length:
            status = STATUS_BAD_PADDED_LENGTH
        else:
            current_chunk_num = (valid_seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            if batch_idx == 0:
                first_chunk_num = current_chunk_num
            elif current_chunk_num != first_chunk_num:
                uniform_chunk_num = 0
            if current_chunk_num > route_max_chunk_num:
                route_max_chunk_num = current_chunk_num
            if evaluate_route == 1:
                if current_chunk_num > INT64_MAX // a.value_heads:
                    evaluate_route = 0
                else:
                    batch_chunk_num = a.value_heads * current_chunk_num
                    if route_total_chunk_num > INT64_MAX - batch_chunk_num:
                        evaluate_route = 0
                    else:
                        route_total_chunk_num += batch_chunk_num
                        route_total_rows = _route_add_cost(
                            route_total_rows, a.value_heads, valid_seq_len
                        )
                        route_query_rows = _route_add_cost(
                            route_query_rows, a.query_heads, valid_seq_len
                        )
                        if route_total_rows < 0 or route_query_rows < 0:
                            evaluate_route = 0
                        # 每个核的成本不超过已检查的总成本。
                        for value_head_idx in range(0, a.value_heads):
                            cyclic_core = (
                                batch_idx * a.value_heads + value_head_idx
                            ) % a.core_num
                            route_core_costs[cyclic_core] += current_chunk_num
    if status == STATUS_OK and a.is_packed == 1:
        if a.cu_seqlens[a.batch] != a.storage_length:
            status = STATUS_BAD_PACKED_END
    if status != STATUS_OK:
        a.metadata[HEADER_STATUS] = status
        return STATUS_OK
    selected_template = a.template_kind
    if selected_template == TEMPLATE_AUTO:
        selected_template = TEMPLATE_STAGED
        if evaluate_route == 1:
            for core_idx in range(0, a.core_num):
                if route_core_costs[core_idx] > route_peak_chunk_num:
                    route_peak_chunk_num = route_core_costs[core_idx]
            persistent_peak = _route_persistent_loads(
                a, uniform_chunk_num, route_lpt_costs, route_valid_rows
            )
            selected_template = _select_route_from_model(
                a,
                route_peak_chunk_num,
                route_total_chunk_num,
                route_total_rows,
                route_query_rows,
                persistent_peak,
                route_valid_rows,
                _training_group_limit(a, route_max_chunk_num),
                remaining_chunk_num,
            )
            # 短多序列训练保留直接 U 路径的缓存。
            if a.has_cu == 1 and a.batch > 1 and route_max_chunk_num <= 2:
                selected_template = TEMPLATE_PERSISTENT
    a.metadata[HEADER_TEMPLATE] = selected_template
    if selected_template == TEMPLATE_PERSISTENT:
        task_num = a.batch * a.value_heads
        task_base = HEADER_WORDS + MAX_AIC_CORES
        used_words = task_base + CHUNK_TASK_WORDS * task_num
        heap_base = used_words
        if used_words + a.batch > a.metadata_word_capacity:
            a.metadata[HEADER_STATUS] = STATUS_BAD_METADATA_CAPACITY
            return STATUS_OK
        for core_idx in range(0, MAX_AIC_CORES):
            a.metadata[HEADER_WORDS + core_idx] = 0
        # 等长序列保持原始批次和头顺序；变长批次采用 O(B log B) 的 LPT 调度。
        # 使用 B 个 GM 临时字，避免固定本地数组和反复扫描全部批次。
        heap_num = a.batch
        if uniform_chunk_num == 0:
            for batch_idx in range(0, a.batch):
                a.metadata[heap_base + batch_idx] = batch_idx
            heap_root = a.batch // 2
            while heap_root > 0:
                heap_root -= 1
                _chunk_heap_sift_down(
                    a.cu_seqlens, a.metadata, heap_base, heap_num, heap_root
                )
        task_cursor = task_base
        for scheduled_batch_idx in range(0, a.batch):
            batch_idx = scheduled_batch_idx
            if uniform_chunk_num == 0:
                batch_idx = a.metadata[heap_base]
                heap_num -= 1
                if heap_num > 0:
                    a.metadata[heap_base] = a.metadata[heap_base + heap_num]
                    _chunk_heap_sift_down(
                        a.cu_seqlens, a.metadata, heap_base, heap_num, 0
                    )
            storage_batch_idx = batch_idx
            token_start = 0
            valid_seq_len = a.storage_length
            if a.has_cu == 1:
                token_start = a.cu_seqlens[batch_idx]
                valid_seq_len = a.cu_seqlens[batch_idx + 1] - token_start
            if a.is_packed == 1:
                storage_batch_idx = 0
            else:
                token_start = 0
            chunk_num = (valid_seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            for value_head_idx in range(0, a.value_heads):
                selected_core = 0
                for core_idx in range(1, a.core_num):
                    if chunk_core_costs[core_idx] < chunk_core_costs[selected_core]:
                        selected_core = core_idx
                a.metadata[task_cursor + CHUNK_TASK_BATCH] = batch_idx
                a.metadata[task_cursor + CHUNK_TASK_STORAGE_BATCH] = storage_batch_idx
                a.metadata[task_cursor + CHUNK_TASK_TOKEN_START] = token_start
                a.metadata[task_cursor + CHUNK_TASK_LENGTH] = valid_seq_len
                a.metadata[task_cursor + CHUNK_TASK_VALUE_HEAD] = value_head_idx
                a.metadata[task_cursor + CHUNK_TASK_NEXT] = 0
                previous_task = chunk_core_tails[selected_core]
                if previous_task == 0:
                    a.metadata[HEADER_WORDS + selected_core] = task_cursor
                else:
                    a.metadata[previous_task + CHUNK_TASK_NEXT] = task_cursor
                chunk_core_tails[selected_core] = task_cursor
                chunk_core_costs[selected_core] += chunk_num
                task_cursor += CHUNK_TASK_WORDS
        a.metadata[HEADER_CHUNK_TASK_NUM] = task_num
        return STATUS_OK

    max_chunk_group = a.max_chunk_group
    if a.template_kind == TEMPLATE_AUTO:
        max_chunk_group = _training_group_limit(a, route_max_chunk_num)
    batch_num_per_set = a.workspace_slots // a.value_heads
    stage12_round_num = 0
    total_active_batch_entries = 0
    batch_start = 0
    while batch_start < a.batch:
        batch_num_in_set = a.batch - batch_start
        if batch_num_in_set > batch_num_per_set:
            batch_num_in_set = batch_num_per_set
        for local_batch_idx in range(0, batch_num_in_set):
            batch_idx = batch_start + local_batch_idx
            token_start = 0
            valid_seq_len = a.storage_length
            if a.has_cu == 1:
                token_start = a.cu_seqlens[batch_idx]
                token_end = a.cu_seqlens[batch_idx + 1]
                valid_seq_len = token_end - token_start
            valid_seq_len_by_batch[local_batch_idx] = valid_seq_len
            token_start_by_batch[local_batch_idx] = token_start
            if valid_seq_len <= 0:
                status = STATUS_BAD_LENGTH
                chunk_num_by_batch[local_batch_idx] = 0
                tail_row_num_by_batch[local_batch_idx] = 0
            elif a.is_packed == 0 and valid_seq_len > a.storage_length:
                status = STATUS_BAD_PADDED_LENGTH
                chunk_num_by_batch[local_batch_idx] = 0
                tail_row_num_by_batch[local_batch_idx] = 0
            else:
                chunk_num_by_batch[local_batch_idx] = (
                    valid_seq_len + CHUNK_SIZE - 1
                ) // CHUNK_SIZE
                tail_row_num_by_batch[local_batch_idx] = (
                    (valid_seq_len - 1) % CHUNK_SIZE
                ) + 1

        chunk_start = 0
        while True:
            active_batch_num = 0
            for local_batch_idx in range(0, batch_num_in_set):
                if chunk_num_by_batch[local_batch_idx] > chunk_start:
                    active_local_batch_idx[active_batch_num] = local_batch_idx
                    remaining_chunk_num[active_batch_num] = (
                        chunk_num_by_batch[local_batch_idx] - chunk_start
                    )
                    active_batch_num += 1
            if active_batch_num == 0:
                break
            group_num = _select_group_num(
                remaining_chunk_num,
                active_batch_num,
                a.value_heads,
                a.workspace_slots,
                max_chunk_group,
            )
            stage12_round_num += 1
            total_active_batch_entries += active_batch_num
            chunk_start += group_num
        batch_start += batch_num_in_set

    if status == STATUS_OK and a.is_packed == 1:
        if a.cu_seqlens[a.batch] != a.storage_length:
            status = STATUS_BAD_PACKED_END
    if status != STATUS_OK:
        a.metadata[HEADER_STATUS] = status
        return STATUS_OK

    record_begin = HEADER_WORDS + stage12_round_num + 1
    used_words = (
        record_begin
        + ROUND_FIXED_WORDS * stage12_round_num
        + ROUND_WORDS_PER_BATCH * total_active_batch_entries
    )
    if used_words > a.metadata_word_capacity:
        a.metadata[HEADER_STATUS] = STATUS_BAD_METADATA_CAPACITY
        return STATUS_OK

    a.metadata[HEADER_STAGE12_ROUND_NUM] = stage12_round_num
    record_cursor = record_begin
    global_round_idx = 0
    batch_start = 0
    while batch_start < a.batch:
        batch_num_in_set = a.batch - batch_start
        if batch_num_in_set > batch_num_per_set:
            batch_num_in_set = batch_num_per_set
        for local_batch_idx in range(0, batch_num_in_set):
            batch_idx = batch_start + local_batch_idx
            token_start = 0
            valid_seq_len = a.storage_length
            if a.has_cu == 1:
                token_start = a.cu_seqlens[batch_idx]
                token_end = a.cu_seqlens[batch_idx + 1]
                valid_seq_len = token_end - token_start
            valid_seq_len_by_batch[local_batch_idx] = valid_seq_len
            token_start_by_batch[local_batch_idx] = token_start
            chunk_num_by_batch[local_batch_idx] = (
                valid_seq_len + CHUNK_SIZE - 1
            ) // CHUNK_SIZE
            tail_row_num_by_batch[local_batch_idx] = (
                (valid_seq_len - 1) % CHUNK_SIZE
            ) + 1

        chunk_start = 0
        while True:
            active_batch_num = 0
            for local_batch_idx in range(0, batch_num_in_set):
                if chunk_num_by_batch[local_batch_idx] > chunk_start:
                    active_local_batch_idx[active_batch_num] = local_batch_idx
                    remaining_chunk_num[active_batch_num] = (
                        chunk_num_by_batch[local_batch_idx] - chunk_start
                    )
                    active_batch_num += 1
            if active_batch_num == 0:
                break

            group_num = _select_group_num(
                remaining_chunk_num,
                active_batch_num,
                a.value_heads,
                a.workspace_slots,
                max_chunk_group,
            )
            for active_idx in range(0, active_batch_num):
                current_chunk_num = remaining_chunk_num[active_idx]
                if current_chunk_num > group_num:
                    current_chunk_num = group_num
                chunk_num_per_group[active_idx] = current_chunk_num

            a.metadata[HEADER_WORDS + global_round_idx] = record_cursor
            cursor = record_cursor
            a.metadata[cursor] = global_round_idx
            a.metadata[cursor + 1] = active_batch_num
            cursor += 2
            for active_idx in range(0, active_batch_num):
                a.metadata[cursor + active_idx] = (
                    batch_start + active_local_batch_idx[active_idx]
                )
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                logical_batch_idx = batch_start + active_local_batch_idx[active_idx]
                if a.is_packed == 1:
                    a.metadata[cursor + active_idx] = 0
                else:
                    a.metadata[cursor + active_idx] = logical_batch_idx
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                local_batch_idx = active_local_batch_idx[active_idx]
                if a.is_packed == 1:
                    a.metadata[cursor + active_idx] = token_start_by_batch[
                        local_batch_idx
                    ]
                else:
                    a.metadata[cursor + active_idx] = 0
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                local_batch_idx = active_local_batch_idx[active_idx]
                a.metadata[cursor + active_idx] = valid_seq_len_by_batch[
                    local_batch_idx
                ]
            cursor += active_batch_num
            a.metadata[cursor] = group_num
            cursor += 1
            for active_idx in range(0, active_batch_num):
                a.metadata[cursor + active_idx] = chunk_start
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                a.metadata[cursor + active_idx] = chunk_num_per_group[active_idx]
            cursor += active_batch_num

            stage1_task_num = 0
            a.metadata[cursor] = 0
            for active_idx in range(0, active_batch_num):
                stage1_task_num += a.value_heads * chunk_num_per_group[active_idx]
                a.metadata[cursor + active_idx + 1] = stage1_task_num
            cursor += active_batch_num + 1

            stage1_task_idx = 0
            for active_idx in range(0, active_batch_num):
                local_batch_idx = active_local_batch_idx[active_idx]
                for _ in range(0, a.value_heads):
                    for local_chunk_idx in range(0, chunk_num_per_group[active_idx]):
                        global_chunk_idx = chunk_start + local_chunk_idx
                        stage1_cost = FULL_CHUNK_COST_NS
                        if (
                            global_chunk_idx + 1 == chunk_num_by_batch[local_batch_idx]
                            and tail_row_num_by_batch[local_batch_idx] < CHUNK_SIZE
                        ):
                            stage1_cost = _stage1_task_cost(
                                tail_row_num_by_batch[local_batch_idx]
                            )
                        stage1_task_costs[stage1_task_idx] = stage1_cost
                        stage1_task_idx += 1
            _partition_costs(
                stage1_task_costs,
                stage1_task_num,
                a.core_num,
                stage1_core_ranges,
                stage1_core_costs,
            )
            for range_idx in range(0, MAX_AIC_CORES * 2):
                a.metadata[cursor + range_idx] = stage1_core_ranges[range_idx]
            cursor += MAX_AIC_CORES * 2

            selected_dv_splits_num = 1
            selected_max_core_cost = 0
            selected_total_cost = 0
            selected_task_num = 0
            selected_is_set = 0
            for candidate_idx in range(0, 4):
                candidate_dv_splits_num = 1 << candidate_idx
                candidate_task_num = 0
                candidate_total_cost = 0
                for active_idx in range(0, active_batch_num):
                    task_cost = (
                        _stage2_task_cost(
                            candidate_idx, chunk_num_per_group[active_idx]
                        )
                        + a.stage2_overhead_ns * chunk_num_per_group[active_idx]
                    )
                    batch_task_num = a.value_heads * candidate_dv_splits_num
                    for _ in range(0, batch_task_num):
                        stage2_task_costs[candidate_task_num] = task_cost
                        candidate_task_num += 1
                        candidate_total_cost += task_cost
                _partition_costs(
                    stage2_task_costs,
                    candidate_task_num,
                    a.core_num,
                    candidate_stage2_core_ranges,
                    candidate_stage2_core_costs,
                )
                candidate_max_core_cost = 0
                for core_idx in range(0, MAX_AIC_CORES):
                    if candidate_stage2_core_costs[core_idx] > candidate_max_core_cost:
                        candidate_max_core_cost = candidate_stage2_core_costs[core_idx]

                candidate_is_better = 0
                if selected_is_set == 0:
                    candidate_is_better = 1
                elif candidate_max_core_cost < selected_max_core_cost:
                    candidate_is_better = 1
                elif candidate_max_core_cost == selected_max_core_cost:
                    if candidate_total_cost < selected_total_cost:
                        candidate_is_better = 1
                    elif candidate_total_cost == selected_total_cost:
                        if candidate_task_num < selected_task_num:
                            candidate_is_better = 1
                        elif candidate_task_num == selected_task_num:
                            if candidate_dv_splits_num < selected_dv_splits_num:
                                candidate_is_better = 1
                if candidate_is_better == 1:
                    selected_is_set = 1
                    selected_dv_splits_num = candidate_dv_splits_num
                    selected_max_core_cost = candidate_max_core_cost
                    selected_total_cost = candidate_total_cost
                    selected_task_num = candidate_task_num
                    for range_idx in range(0, MAX_AIC_CORES * 2):
                        selected_stage2_core_ranges[range_idx] = (
                            candidate_stage2_core_ranges[range_idx]
                        )

            for active_idx in range(0, active_batch_num):
                a.metadata[cursor + active_idx] = selected_dv_splits_num
            cursor += active_batch_num
            stage2_task_num = 0
            a.metadata[cursor] = 0
            for active_idx in range(0, active_batch_num):
                stage2_task_num += a.value_heads * selected_dv_splits_num
                a.metadata[cursor + active_idx + 1] = stage2_task_num
            cursor += active_batch_num + 1
            for range_idx in range(0, MAX_AIC_CORES * 2):
                a.metadata[cursor + range_idx] = selected_stage2_core_ranges[range_idx]
            cursor += MAX_AIC_CORES * 2

            record_cursor = cursor
            global_round_idx += 1
            chunk_start += group_num
        batch_start += batch_num_in_set

    a.metadata[HEADER_WORDS + stage12_round_num] = record_cursor
    return STATUS_OK


def _take(words: list[int], cursor: int, count: int) -> tuple[list[int], int]:
    end = cursor + count
    if cursor < 0 or count < 0 or end > len(words):
        raise ValueError("metadata record extends past available words")
    return words[cursor:end], end


def _decode_ranges(flat_words: list[int]) -> list[tuple[int, int]]:
    if len(flat_words) != MAX_AIC_CORES * 2:
        raise ValueError("core range block must contain 64 words")
    return [
        (flat_words[index], flat_words[index + 1])
        for index in range(0, len(flat_words), 2)
    ]


def _validate_prefix(prefix: list[int], name: str) -> None:
    if not prefix or prefix[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if any(end < start for start, end in zip(prefix, prefix[1:])):
        raise ValueError(f"{name} must be nondecreasing")


def _validate_ranges(ranges: list[tuple[int, int]], total: int, name: str) -> None:
    cursor = 0
    saw_empty = False
    for start, end in ranges:
        if (start, end) == (0, 0):
            saw_empty = True
            continue
        if saw_empty:
            raise ValueError(f"{name} has work after an unused core")
        if start != cursor or end <= start or end > total:
            raise ValueError(f"{name} is not a contiguous partition")
        cursor = end
    if cursor != total:
        raise ValueError(f"{name} does not cover its task total")


def _decode_round(
    words: list[int],
    start: int,
    end: int,
    *,
    batch: int,
    value_heads: int,
) -> dict[str, object]:
    if start < 0 or end > len(words) or start + 2 > end:
        raise ValueError("metadata round record is truncated")
    # 将后续所有 _take 和标量访问限制在当前记录范围内。
    words = words[start:end]
    start, end = 0, end - start
    cursor = 0
    stage12_round_idx, active_batch_num = words[cursor : cursor + 2]
    cursor += 2
    if not 1 <= active_batch_num <= min(batch, WORKSPACE_SLOTS):
        raise ValueError("metadata active_batch_num outside batch/workspace bounds")
    if start + round_record_words(active_batch_num) != end:
        raise ValueError("metadata round record length does not match active batches")

    batch_idx, cursor = _take(words, cursor, active_batch_num)
    storage_batch_idx, cursor = _take(words, cursor, active_batch_num)
    token_start, cursor = _take(words, cursor, active_batch_num)
    valid_seq_len, cursor = _take(words, cursor, active_batch_num)
    group_num = words[cursor]
    cursor += 1
    chunk_start_per_group, cursor = _take(words, cursor, active_batch_num)
    chunk_num_per_group, cursor = _take(words, cursor, active_batch_num)
    stage1_task_prefix, cursor = _take(words, cursor, active_batch_num + 1)
    stage1_ranges_flat, cursor = _take(words, cursor, MAX_AIC_CORES * 2)
    dv_splits_num, cursor = _take(words, cursor, active_batch_num)
    stage2_task_prefix, cursor = _take(words, cursor, active_batch_num + 1)
    stage2_ranges_flat, cursor = _take(words, cursor, MAX_AIC_CORES * 2)
    if cursor != end:
        raise ValueError("metadata round decoder did not consume the full record")

    if group_num <= 0:
        raise ValueError("group_num must be positive")
    if any(index < 0 or index >= batch for index in batch_idx):
        raise ValueError("batch_idx is outside logical batch")
    if any(length <= 0 for length in valid_seq_len):
        raise ValueError("valid_seq_len must be positive")
    if any(count <= 0 for count in chunk_num_per_group):
        raise ValueError("chunk_num_per_group must be positive")
    if any(value not in (1, 2, 4, 8) for value in dv_splits_num):
        raise ValueError("dv_splits_num must be 1, 2, 4, or 8")

    _validate_prefix(stage1_task_prefix, "stage1_task_prefix")
    _validate_prefix(stage2_task_prefix, "stage2_task_prefix")
    for index, chunk_num in enumerate(chunk_num_per_group):
        if stage1_task_prefix[index + 1] - stage1_task_prefix[index] != (
            value_heads * chunk_num
        ):
            raise ValueError("stage1_task_prefix does not match chunk tasks")
        if stage2_task_prefix[index + 1] - stage2_task_prefix[index] != (
            value_heads * dv_splits_num[index]
        ):
            raise ValueError("stage2_task_prefix does not match DV tasks")
    if stage1_task_prefix[-1] > WORKSPACE_SLOTS:
        raise ValueError("stage1 task total exceeds workspace slots")
    if stage2_task_prefix[-1] > WORKSPACE_SLOTS * 8:
        raise ValueError("stage2 task total exceeds workspace slots")

    stage1_core_ranges = _decode_ranges(stage1_ranges_flat)
    stage2_core_ranges = _decode_ranges(stage2_ranges_flat)
    _validate_ranges(stage1_core_ranges, stage1_task_prefix[-1], "stage1_core_ranges")
    _validate_ranges(stage2_core_ranges, stage2_task_prefix[-1], "stage2_core_ranges")
    return {
        "stage12_round_idx": stage12_round_idx,
        "active_batch_num": active_batch_num,
        "batch_idx": batch_idx,
        "storage_batch_idx": storage_batch_idx,
        "token_start": token_start,
        "valid_seq_len": valid_seq_len,
        "group_num": group_num,
        "chunk_start_per_group": chunk_start_per_group,
        "chunk_num_per_group": chunk_num_per_group,
        "stage1_task_prefix": stage1_task_prefix,
        "stage1_core_ranges": stage1_core_ranges,
        "dv_splits_num": dv_splits_num,
        "stage2_task_prefix": stage2_task_prefix,
        "stage2_core_ranges": stage2_core_ranges,
    }


def decode_metadata(raw: Sequence[int]) -> dict[str, object]:
    """解码并检查 metadata，供测试和诊断使用。"""

    words = [int(value) for value in raw]
    if len(words) < HEADER_WORDS:
        raise ValueError("metadata is shorter than its header")
    if words[HEADER_MAGIC] != MAGIC:
        raise ValueError("metadata magic does not match")
    if words[HEADER_ABI_VERSION] != ABI_VERSION:
        raise ValueError("metadata ABI version does not match")

    status = words[HEADER_STATUS]
    header = {
        "magic": words[HEADER_MAGIC],
        "abi_version": words[HEADER_ABI_VERSION],
        "status": status,
        "core_num": words[HEADER_CORE_NUM],
        "stage12_round_num": words[HEADER_STAGE12_ROUND_NUM],
        "batch": words[HEADER_BATCH],
        "value_heads": words[HEADER_VALUE_HEADS],
        "has_cu": words[HEADER_HAS_CU],
        "workspace_slots": words[HEADER_WORKSPACE_SLOTS],
        "template": {TEMPLATE_STAGED: "staged", TEMPLATE_PERSISTENT: "persistent"}.get(
            words[HEADER_TEMPLATE], "unknown"
        ),
    }
    if status != STATUS_OK:
        header["used_words"] = HEADER_WORDS
        header["stage12_rounds"] = []
        return header
    if not 1 <= header["core_num"] <= MAX_AIC_CORES:
        raise ValueError("metadata core_num must be in [1, 32]")
    if header["workspace_slots"] != WORKSPACE_SLOTS:
        raise ValueError(f"metadata workspace_slots must be {WORKSPACE_SLOTS}")
    if header["batch"] <= 0:
        raise ValueError("successful metadata requires positive batch")
    if not 1 <= header["value_heads"] <= MAX_VALUE_HEADS:
        raise ValueError("metadata value_heads must be in [1, 96]")

    if header["has_cu"] not in (0, 1):
        raise ValueError("metadata has_cu must be 0 or 1")
    if header["template"] == "unknown":
        raise ValueError("metadata template is invalid")
    if header["template"] == "persistent":
        task_num = words[HEADER_CHUNK_TASK_NUM]
        task_base = HEADER_WORDS + MAX_AIC_CORES
        used_words = task_base + CHUNK_TASK_WORDS * task_num
        if (
            header["stage12_round_num"] != 0
            or task_num != header["batch"] * header["value_heads"]
        ):
            raise ValueError("chunk metadata task count does not match its shape")
        if used_words > len(words):
            raise ValueError("chunk metadata tasks are truncated")
        seen_offsets = set()
        seen_jobs = set()
        tasks_by_core = []
        for core_idx in range(MAX_AIC_CORES):
            cursor = words[HEADER_WORDS + core_idx]
            if core_idx >= header["core_num"] and cursor:
                raise ValueError("inactive core has chunk tasks")
            tasks = []
            while cursor:
                if cursor in seen_offsets:
                    raise ValueError("chunk task chain contains a cycle or duplicate")
                if (
                    cursor < task_base
                    or cursor + CHUNK_TASK_WORDS > used_words
                    or (cursor - task_base) % CHUNK_TASK_WORDS
                ):
                    raise ValueError("chunk task pointer is out of bounds or unaligned")
                seen_offsets.add(cursor)
                batch_idx = words[cursor + CHUNK_TASK_BATCH]
                head_idx = words[cursor + CHUNK_TASK_VALUE_HEAD]
                if (
                    not 0 <= batch_idx < header["batch"]
                    or not 0 <= head_idx < header["value_heads"]
                ):
                    raise ValueError("chunk task batch/head index is out of bounds")
                if (batch_idx, head_idx) in seen_jobs:
                    raise ValueError("chunk sequence/head task is duplicated")
                seen_jobs.add((batch_idx, head_idx))
                length = words[cursor + CHUNK_TASK_LENGTH]
                storage_batch = words[cursor + CHUNK_TASK_STORAGE_BATCH]
                token_start = words[cursor + CHUNK_TASK_TOKEN_START]
                if (
                    length <= 0
                    or token_start < 0
                    or storage_batch not in (0, batch_idx)
                ):
                    raise ValueError("chunk task has invalid sequence coordinates")
                tasks.append(
                    dict(
                        batch_idx=batch_idx,
                        storage_batch_idx=storage_batch,
                        token_start=token_start,
                        sequence_length=length,
                        value_head_idx=head_idx,
                    ),
                )
                cursor = words[cursor + CHUNK_TASK_NEXT]
            tasks_by_core.append(tasks)
        if len(seen_jobs) != task_num:
            raise ValueError("chunk metadata does not cover every task")
        header.update(
            used_words=used_words,
            stage12_rounds=[],
            chunk_task_num=task_num,
            chunk_tasks_by_core=tasks_by_core,
        )
        return header

    if words[HEADER_CHUNK_TASK_NUM] != 0:
        raise ValueError("staged metadata must not contain persistent tasks")
    stage12_round_num = int(header["stage12_round_num"])
    if stage12_round_num < 1:
        raise ValueError("staged metadata round count must be positive")
    offset_begin = HEADER_WORDS
    offset_end = offset_begin + stage12_round_num + 1
    if offset_end > len(words):
        raise ValueError("metadata offset table is truncated")
    min_record_words = round_record_words(1)
    if offset_end + stage12_round_num * min_record_words > len(words):
        raise ValueError("metadata capacity is insufficient for round records")
    offsets = words[offset_begin:offset_end]
    if any(offset < 0 for offset in offsets):
        raise ValueError("metadata offset must be nonnegative")
    if offsets[0] < offset_end:
        raise ValueError("metadata first record offset overlaps the offset table")
    if offsets[-1] > len(words):
        raise ValueError("metadata final offset exceeds capacity")
    if any(end <= start for start, end in zip(offsets, offsets[1:])):
        raise ValueError("metadata offsets must be strictly increasing")

    header["used_words"] = offsets[-1]
    header["stage12_rounds"] = [
        _decode_round(
            words,
            offsets[index],
            offsets[index + 1],
            batch=int(header["batch"]),
            value_heads=int(header["value_heads"]),
        )
        for index in range(stage12_round_num)
    ]
    return header


_ROUTE_PARAMETERS = (
    0,
    2711857704,
    699286307,
    16555661144,
    28359460,
    94263834,
    94263834,
    12013349865,
    1045656019,
    84358156,
    12842477,
    2527878117,
    181898838,
    6,
    8441,
    0,
    0,
    692059912,
    851406583,
    21510815509,
    73490570,
    81791118,
    166178779,
    12029971961,
    2552246171,
    459139222,
    45086212,
    2303638937,
    143977434,
    6,
    18992,
    0,
)

_DT_ROUTE_PARAMETERS = (
    4135271144,
    193981546,
    366448654,
    12881511041,
    29417161,
    4351059785,
    158980371,
    10652650460,
    3206978745,
    332321804,
    0,
    419242242,
    29945874,
    7,
    0,
    0,
    20317858430,
    1160598799,
    1018161115,
    1015361233,
    14389959,
    933231823,
    115749683,
    11499084207,
    4373896639,
    413928806,
    41555011,
    0,
    0,
    7,
    0,
    0,
)


def build_schedule(
    *,
    batch,
    nheads_qk,
    nheads_v,
    storage_length,
    layout,
    lengths=None,
    enable_grad=False,
    core_num=28,
    template="auto",
    device_kind="PR",
    stage2_overhead_ns=0,
):
    """ABI4 scheduling with the native workspace capacity.

    The original 861-slot capacity is checked separately against fixed-source
    ABI words; production uses WORKSPACE_SLOTS consistently for allocations.

    Lengths are already validated host metadata; None means unpadded uniform input.
    The source has no empty-sequence route. Do not silently invent one here.
    A nonzero native per-chunk Stage2 overhead changes only its DV cost ranking;
    all Stage1 groups, numerical formulas and sequence ownership remain intact.
    """
    if layout not in ("BNSD", "BSND", "TND"):
        raise ValueError("invalid layout")
    if not 0 < nheads_qk <= nheads_v <= 96 or nheads_v % nheads_qk:
        raise ValueError("invalid GQA heads")
    if template not in ("auto", "staged", "persistent"):
        raise ValueError("invalid template")
    has_cu = lengths is not None
    if layout == "TND" and not has_cu:
        raise ValueError("TND requires lengths")
    lengths = [storage_length] * batch if lengths is None else list(lengths)
    if len(lengths) != batch or any(x <= 0 for x in lengths):
        raise ValueError("source schedule requires positive sequence lengths")
    if layout == "TND" and sum(lengths) != storage_length:
        raise ValueError("packed lengths must cover storage")
    prefix = [0]
    for size in lengths:
        prefix.append(prefix[-1] + size)
    capacity = metadata_capacity_upper_bound(
        batch, storage_length, is_packed=layout == "TND", value_heads=nheads_v
    )
    short = (
        enable_grad
        and layout == "BNSD"
        and batch == 1
        and nheads_qk == nheads_v == SHORT_TRAIN_HEADS
        and storage_length >= SHORT_TRAIN_MIN_LENGTH
        and storage_length % CHUNK_SIZE == 0
        and not has_cu
        and core_num == MAX_AIC_CORES
    )
    words = [0] * capacity
    args = SimpleNamespace(
        cu_seqlens=prefix,
        route_coefficients=_DT_ROUTE_PARAMETERS
        if device_kind == "DT"
        else _ROUTE_PARAMETERS,
        calibration_enabled=1,
        layout_kind=0 if layout == "BNSD" else 1,
        metadata=words,
        metadata_word_capacity=capacity,
        batch=batch,
        physical_batches=1 if layout == "TND" else batch,
        storage_length=storage_length,
        value_heads=nheads_v,
        query_heads=nheads_qk,
        has_cu=int(has_cu),
        is_packed=int(layout == "TND"),
        core_num=core_num,
        workspace_slots=WORKSPACE_SLOTS,
        template_kind={"auto": 0, "staged": 1, "persistent": 2}[template],
        max_chunk_group=SHORT_TRAIN_CHUNK_GROUP if short else MAX_CHUNK_GROUP,
        stage2_overhead_ns=stage2_overhead_ns,
    )
    status = flash_kda_fwd_metadata_kernel(args)
    if status != STATUS_OK:
        raise RuntimeError(f"metadata kernel returned status {status}")
    decoded = decode_metadata(words)
    if decoded["status"]:
        raise ValueError(f"source schedule status {decoded['status']}")
    return words[: decoded["used_words"]], decoded


# -----------------------------------------------------------------------------
# Runtime caches and public forward entry.
# -----------------------------------------------------------------------------

_COMPILED = {}
_LOCK = Lock()
_IDENTITIES = {}


def _identity16(device):
    """Cache only the immutable constant; publish it safely across streams."""
    stream = torch.npu.current_stream(device)
    with _LOCK:
        entry = _IDENTITIES.get(device.index)
        if entry is None:
            value = torch.eye(16, dtype=torch.float16, device=device)
            ready = stream.record_event()
            _IDENTITIES[device.index] = (value, ready, stream)
        else:
            value, ready, owner = entry
            if stream != owner:
                stream.wait_event(ready)
    return value


def flash_kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    metadata=None,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    *,
    scale=128**-0.5,
    output_final_state=False,
    return_intermediate_states=False,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False,
    disable_recompute=False,
    lower_bound=None,
    layout="TND",
):
    flags = dict(
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
    )
    batch, nq, nv, length, lengths = _validate(
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
        scale=scale,
        layout=layout,
        flags=flags,
        lower_bound=lower_bound,
    )
    training = not torch.is_inference_mode_enabled()
    save = training and disable_recompute
    norm = training and use_qk_l2norm_in_kernel
    save_beta = training and use_beta_sigmoid_in_kernel
    scale = 128**-0.5 if scale is None else float(scale)
    mode = 0 if not use_gate_in_kernel else (2 if lower_bound is None else 1)
    lower = -0.1 if mode != 1 else float(lower_bound)
    packed = layout == "TND"
    chunks = (length + 63) // 64
    chunk_shape = (chunks + batch,) if packed else (batch, chunks)
    norm_shape = (length, nq) if packed else (batch, nq, length)
    beta_shape = (length, nv) if packed else (batch, length, nv)
    with torch.npu.device(q.device):
        properties = torch.npu.get_device_properties(q.device)
        cores = properties.cube_core_num
        native_template = "auto"
        if (
            not training
            and use_qk_l2norm_in_kernel
            and mode == 1
            and lower == -5
            and abs(scale) <= 0.125
            and all(n % 64 == 0 for n in lengths)
        ):
            native_template = "persistent"
        try:
            schedule_result = build_schedule(
                batch=batch,
                nheads_qk=nq,
                nheads_v=nv,
                storage_length=length,
                layout=layout,
                lengths=lengths
                if cu_seqlens is not None or seqused is not None
                else None,
                enable_grad=training,
                core_num=cores,
                device_kind="DT" if "DT" in properties.name else "PR",
                stage2_overhead_ns=1000,
                template=native_template,
            )
        except ValueError as error:
            raise ValueError(
                f"Flash KDA schedule validation failed: {error}"
            ) from error
        except RuntimeError as error:
            raise RuntimeError(
                f"Flash KDA schedule generation failed: {error}"
            ) from error
        if not isinstance(schedule_result, tuple) or len(schedule_result) != 2:
            raise RuntimeError("Flash KDA schedule returned an invalid result")
        words, decoded = schedule_result
        if decoded["status"] != STATUS_OK:
            raise RuntimeError(
                f"Flash KDA schedule returned status {decoded['status']}"
            )

        def allocate(shape, dtype=torch.bfloat16, zero=False):
            element_size = 2 if dtype == torch.bfloat16 or dtype == torch.float16 else 4
            _require(
                math.prod(shape) <= (2**63 - 1) // element_size,
                "tensor byte size exceeds signed 64-bit addressing",
            )
            factory = torch.zeros if zero else torch.empty
            return factory(shape, dtype=dtype, device=q.device)

        initial = (
            initial_state
            if initial_state is not None
            else allocate((batch, nv, 128, 128), torch.float32, True)
        )
        alpha = A_log if use_gate_in_kernel else None
        bias = dt_bias if use_gate_in_kernel else None
        if alpha is None:
            alpha = allocate((nv,), torch.float32, True)
        if bias is None:
            bias = allocate((nv, 128), torch.float32, True)
        exchange = allocate((cores, 256), torch.float32)
        identity = _identity16(q.device)
        schedule = torch.tensor(words, dtype=torch.int64, device=q.device)
        staged = decoded["template"] == "staged"
        aq = allocate(chunk_shape + (nv, 64, 64) if training or staged else (1,))
        ak = allocate(chunk_shape + (nv, 64, 64) if training else (1,))
        cache_shape = tuple(v.shape) if save or staged else (1,)
        w, u, kg = [allocate(cache_shape) for _ in range(3)]
        qg, vnew = [allocate(tuple(v.shape) if save else (1,)) for _ in range(2)]
        out = allocate(
            tuple(v.shape), zero=not packed and any(n < length for n in lengths)
        )
        qs = allocate((WORKSPACE_SLOTS, 64, 128) if staged else (1,))
        gamma = allocate((WORKSPACE_SLOTS, 128) if staged else (1,), torch.float32)
        qn, kn = [allocate(norm_shape + (128,) if norm else (1,)) for _ in range(2)]
        rstd = allocate((2,) + norm_shape if norm else (1,), torch.float32)
        # Packed capacity includes unused chunk slots; padded seqused may leave
        # whole unused chunks even when each valid length is a multiple of 64.
        gc = allocate(
            chunk_shape + (nv, 64, 128) if save else (1,),
            torch.float32,
            save and (packed or any(n % 64 or n < length for n in lengths)),
        )
        bact = allocate(beta_shape if save_beta else (1,), torch.float32)
        history = allocate(
            chunk_shape + (nv, 128, 128)
            if return_intermediate_states or staged
            else (1,),
            torch.float32,
        )
        final = allocate((batch, nv, 128, 128), torch.float32)
        tensors = [
            q,
            k,
            v,
            g,
            beta,
            alpha,
            bias,
            exchange,
            identity,
            initial,
            schedule,
            aq,
            ak,
            w,
            u,
            qg,
            kg,
            qs,
            gamma,
            qn,
            kn,
            rstd,
            gc,
            bact,
            out,
            vnew,
            history,
            final,
        ]
        bound = []
        for tensor in tensors:
            bound.append(
                tla.from_dlpack(tensor.flatten(), layout_tag=tla.arch.RowMajor)
            )
        config = (
            batch,
            length,
            nq,
            nv,
            layout,
            mode,
            allow_neg_eigval,
            lower,
            scale,
            use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel,
        )
        options = dict(
            save_matrices=training,
            save_cache=save,
            save_history=return_intermediate_states,
            save_norm=norm,
            save_beta=save_beta,
        )
        if staged:
            options["pipeline"] = True
        else:
            options.update(prefetch=True, bulk_operands=True, pipeline=True)
        key = (
            decoded["template"],
            config,
            tuple(sorted(options.items())),
            tuple((x.numel(), x.dtype) for x in tensors),
        )
        with _LOCK:
            binary = _COMPILED.get(key)
            if binary is None:
                cls = (
                    PersistentMetadataKernel
                    if decoded["template"] == "persistent"
                    else StagedMetadataKernel
                )
                binary = tla.compile(
                    cls(*config, **options), *bound, options="--npu-arch 3510"
                )
                if len(_COMPILED) >= 128:
                    _COMPILED.pop(next(iter(_COMPILED)))
                _COMPILED[key] = binary
        stream = torch.npu.current_stream(q.device)
        for tensor in tensors:
            tensor.record_stream(stream)
        binary(*bound, block_num=cores, stream=stream.npu_stream)
    return (
        out,
        final if output_final_state else None,
        history if return_intermediate_states else None,
        aq if training else None,
        ak if training else None,
        w if save else None,
        u if save else None,
        qg if save else None,
        kg if save else None,
        vnew if save else None,
        gc if save else None,
        qn if norm else None,
        rstd[0] if norm else None,
        kn if norm else None,
        rstd[1] if norm else None,
        bact if training and use_beta_sigmoid_in_kernel else None,
    )
