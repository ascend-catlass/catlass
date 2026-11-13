# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ascend 950 Flash KDA forward device kernels.

Adapted from cann/cannbot-dsl fc438d3ef03129929eb8faae3906365b3162f163,
samples/attention/flash_kda_fwd/. See NOTICE and README.md.

Sections: shared address/Vector/Cube/state helpers; preprocessing and scores;
staged producer, recurrence and kernel; persistent math, pipeline and kernel.
Host-only ABI4 scheduling is implemented in __init__.py.
"""

from catlass import tla


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
        tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO,
            sat_mode=tla.params.SatMode.NOSAT,
            round_mode=tla.params.RoundMode.CAST_RINT,
        ),
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
                        tla.params.CastParams(
                            reg_slot=tla.params.RegSlot.ZERO,
                            sat_mode=tla.params.SatMode.NOSAT,
                            round_mode=tla.params.RoundMode.CAST_RINT,
                        ),
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
            tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.NOSAT,
                round_mode=tla.params.RoundMode.CAST_RINT,
            ),
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
                    tla.params.CastParams(
                        reg_slot=tla.params.RegSlot.ZERO,
                        sat_mode=tla.params.SatMode.NOSAT,
                        round_mode=tla.params.RoundMode.CAST_RINT,
                    ),
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
                tla.wait_flag(left_free)
            tla.copy(left, source)
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
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
                tla.set_flag(left_free)


@tla.jit
def matrix64(ptr, layout):
    return tla.make_tensor_like(ptr, row_major(ptr, 64, 64), layout)


@tla.jit
def load_inverse_operands(left, right, a, b, zero, ready, free, diagonal_only, clear):
    """Reference full64 double-buffer load, optionally replace diagonal fractals."""
    tla.wait_flag(free)
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
    tla.set_flag(ready)


@tla.jit
def inverse_mmad(c, a, b, ready, free, initialize, unit):
    tla.wait_flag(ready)
    tla.mmad(c, a, b, init_c=initialize, unit_flag=unit)
    tla.set_flag(free)


@tla.jit
def inverse_fix(destination, c, fix_ready, accumulator_free):
    tla.copy(destination, c, tla.params.CopyL0C2DstParams(unit_flag=3))
    tla.set_flag(fix_ready)
    tla.wait_flag(fix_ready)
    tla.set_flag(accumulator_free)
    tla.wait_flag(accumulator_free)


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
    tla.wait_flag(free)
    tla.copy(a, zero)
    tla.copy(b, zero)
    copy_selected_fractals(a, left, left_mode)
    copy_selected_fractals(b, right, right_mode)
    tla.set_flag(ready)


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
    load_selected_pair(scratch, inverse, a1, b1, zero, ready1, free1, 3, 1)
    inverse_mmad(c_power, a1, b1, ready1, free1, True, 3)
    inverse_fix(temporary, c_power, fix_ready, accumulator_free)
    load_selected_pair(inverse, temporary, a0, b0, zero, ready0, free0, 2, 3)
    inverse_mmad(c_inverse, a0, b0, ready0, free0, False, 3)
    inverse_fix(inverse, c_inverse, fix_ready, accumulator_free)
    load_selected_pair(scratch, inverse, a1, b1, zero, ready1, free1, 0, 4)
    # Preserve source overlap: prepare the second pair's independent LHS before
    # the first MMAD and FIX produce its RHS.
    tla.wait_flag(free0)
    tla.copy(a0, zero)
    tla.copy(b0, zero)
    tla.copy(a0, inverse)
    inverse_mmad(c_power, a1, b1, ready1, free1, True, 3)
    inverse_fix(temporary, c_power, fix_ready, accumulator_free)
    copy_selected_fractals(b0, temporary, 5)
    tla.set_flag(ready0)
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
def finish_u_delta(u, delta, output, dv, aiv, valid):
    """Staged source consumes BF16 u, subtracts FP32 W@H, then casts once."""
    width = 64
    if tla.const_expr(dv < 64):
        width = dv
    with tla.vec.func(mode="simd"):
        for group in tla.range(16):
            for lane in tla.range_constexpr(2):
                row = group * 2 + lane
                for col in tla.range_constexpr((dv + 63) // 64):
                    value = tla.full(0.0, dtype=tla.Float32)
                    if aiv * 32 + row < valid:
                        raw = load_bf16_f32(
                            tla.tile_view(
                                u, tla.make_shape(1, 64), tla.make_coord(row, col)
                            )
                        )
                        change = tla.tile_view(
                            delta, tla.make_shape(1, 64), tla.make_coord(row, col)
                        ).load()
                        value = raw - change
                    store_bf16(
                        tla.tile_view(
                            output, tla.make_shape(1, 64), tla.make_coord(row, col)
                        ),
                        value,
                        width,
                    )


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
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    tla.mmad(cu, a0, b, init_c=True)
    if tla.const_expr(projection_done is not None):
        tla.set_flag(projection_done)
    tla.copy(a1, q)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    if tla.const_expr(output_free is not None):
        tla.wait_flag(output_free)
    tla.mmad(co, a1, b, init_c=True)


@tla.jit
def load_recurrent_operands(kg, aqk, ka_ptr, aq_ptr, dv, loaded):
    kb = tla.make_tensor_like(ka_ptr, row_major(ka_ptr, 64, 128), tla.arch.nZ)
    qa = tla.make_tensor_like(aq_ptr, row_major(aq_ptr, 64, 64), tla.arch.zN)
    tla.copy(kb, kg)
    tla.copy(qa, aqk)
    tla.set_flag(loaded)


@tla.jit
def compute_output_and_state_delta(
    vnew, ka_ptr, aq_ptr, ua_ptr, ub_ptr, co_ptr, ch_ptr, dv, loaded
):
    # Source uses the same resident v_new for both delta H and output.
    vt = tla.make_tensor_like(vnew.ptr, row_major(vnew.ptr, dv, 64), tla.arch.nZ)
    ua = tla.make_tensor_like(ua_ptr, row_major(ua_ptr, dv, 64), tla.arch.zN)
    ub = tla.make_tensor_like(ub_ptr, row_major(ub_ptr, 64, dv), tla.arch.nZ)
    kb = tla.make_tensor_like(ka_ptr, row_major(ka_ptr, 64, 128), tla.arch.nZ)
    qa = tla.make_tensor_like(aq_ptr, row_major(aq_ptr, 64, 64), tla.arch.zN)
    co = tla.make_tensor_like(co_ptr, row_major(co_ptr, 64, dv), tla.arch.L0Clayout)
    ch = tla.make_tensor_like(ch_ptr, row_major(ch_ptr, dv, 128), tla.arch.L0Clayout)
    tla.copy(ua, vt)
    tla.copy(ub, vnew)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    tla.mmad(ch, ua, kb, init_c=True)
    tla.mmad(co, qa, ub, init_c=False)


@tla.jit
def compute_state_delta(vnew, ka_ptr, ua_ptr, ch_ptr, dv, loaded):
    """Source update_state: publish this result before the output MMAD."""
    vt = tla.make_tensor_like(vnew.ptr, row_major(vnew.ptr, dv, 64), tla.arch.nZ)
    ua = tla.make_tensor_like(ua_ptr, row_major(ua_ptr, dv, 64), tla.arch.zN)
    kb = tla.make_tensor_like(ka_ptr, row_major(ka_ptr, 64, 128), tla.arch.nZ)
    ch = tla.make_tensor_like(ch_ptr, row_major(ch_ptr, dv, 128), tla.arch.L0Clayout)
    tla.copy(ua, vt)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    tla.mmad(ch, ua, kb, init_c=True)


@tla.jit
def accumulate_output(vnew, aq_ptr, ub_ptr, co_ptr, dv, loaded):
    """Source accumulate_mqk_u; independent of the state-delta FIX write."""
    ub = tla.make_tensor_like(ub_ptr, row_major(ub_ptr, 64, dv), tla.arch.nZ)
    qa = tla.make_tensor_like(aq_ptr, row_major(aq_ptr, 64, 64), tla.arch.zN)
    co = tla.make_tensor_like(co_ptr, row_major(co_ptr, 64, dv), tla.arch.L0Clayout)
    tla.copy(ub, vnew)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
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
        tla.set_flag(gm_loaded)
        tla.wait_flag(gm_loaded)
        if valid < 64:
            zero_qkv_tail(input_q, input_k, uv, valid)
        if tla.const_expr(beta.ptr.dtype != tla.Float32):
            prepare_beta_input(beta_io, raw_beta, valid, beta_pitch)
        if tla.const_expr(normalize):
            compute_qk_partial_sums(input_q, input_k, at(stats, 0, 128))
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            tla.copy(
                row_major(exchange.ptr + aiv * 128, 1, 128),
                row_major(stats.ptr, 1, 128),
            )
            tla.cross_core_set_flag(published, tla.arch.MTE3)
            tla.cross_core_wait_flag(published, tla.arch.MTE2)
            tla.copy(
                row_major(stats.ptr + 128, 1, 128),
                row_major(exchange.ptr + (1 - aiv) * 128, 1, 128),
            )
            tla.set_flag(peer_loaded)
        if valid == 64:
            activate_gate(raw_gate, bias_ub, alpha_ub, ug, 64, mode, lower, True)
        else:
            activate_gate(raw_gate, bias_ub, alpha_ub, ug, valid, mode, lower, True)
        if tla.const_expr(normalize):
            tla.wait_flag(peer_loaded)
            tla.cross_core_set_flag(consumed, tla.arch.MTE2)
            tla.cross_core_wait_flag(consumed, tla.arch.MTE3)
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
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
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
                    tla.set_flag(v_done)
                    tla.wait_flag(v_done)
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
                        tla.set_flag(cast_free)
                        tla.wait_flag(cast_free)
                        beta_to_column(at(stats, 256, 64), beta_column)
                        tla.set_flag(v_done)
                        tla.wait_flag(v_done)
                        tla.copy(
                            strided_rows(rstd.ptr, valid, 1, rstd_stride),
                            strided_rows(beta_column.ptr, valid, 1, 8),
                        )
                        tla.set_flag(cast_free)
                        tla.wait_flag(cast_free)
                        beta_to_column(at(stats, 320, 64), beta_column)
                        tla.set_flag(v_done)
                        tla.wait_flag(v_done)
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
    slot = step % 2
    if slot == 0:
        tla.cross_core_wait_flag(free0, tla.arch.VECTOR)
    else:
        tla.cross_core_wait_flag(free1, tla.arch.VECTOR)
    nz = flat(nz_ptr + slot * 1088, 1088)
    nd2nz_tile(nd, nz, 16, 64)
    tla.set_flag(v_done)
    tla.wait_flag(v_done)
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
        tla.cross_core_set_flag(ready0, tla.arch.MTE3)
    else:
        tla.cross_core_set_flag(ready1, tla.arch.MTE3)


@tla.jit
def consume_operand(dst, l1_ptr, step, ready0, ready1, free0, free1, transpose):
    slot = step % 2
    if slot == 0:
        tla.cross_core_wait_flag(ready0, tla.arch.MTE1)
    else:
        tla.cross_core_wait_flag(ready1, tla.arch.MTE1)
    src = tla.make_tensor_like(
        l1_ptr + slot * 2048, row_major(l1_ptr, 16, 128), tla.arch.zN
    )
    if tla.const_expr(transpose):
        src = tla.make_tensor_like(
            l1_ptr + slot * 2048, row_major(l1_ptr, 128, 16), tla.arch.nZ
        )
    tla.copy(dst, src)
    if slot == 0:
        tla.cross_core_set_flag(free0, tla.arch.MTE1)
    else:
        tla.cross_core_set_flag(free1, tla.arch.MTE1)


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
    ready0 = tla.cross_flag("pd_ready0", mode=2)
    ready1 = tla.cross_flag("pd_ready1", mode=2)
    free0 = tla.cross_flag("pd_free0", mode=2)
    free1 = tla.cross_flag("pd_free1", mode=2)
    v_done = tla.flag("pd_v_done", tla.arch.VECTOR, tla.arch.MTE3)
    loaded = tla.flag("pd_loaded", tla.arch.MTE1, tla.arch.CUBE)
    op_free = tla.flag("pd_op_free", tla.arch.CUBE, tla.arch.MTE1)
    with tla.cube():
        ka = tla.make_tensor_like(a_ptr, row_major(a_ptr, 16, 128), tla.arch.zN)
        qa = tla.make_tensor_like(a_ptr + 2048, row_major(a_ptr, 16, 128), tla.arch.zN)
        kb = tla.make_tensor_like(b_ptr, row_major(b_ptr, 128, 16), tla.arch.nZ)
        tla.cross_core_set_flag(free0, tla.arch.MTE1)
        tla.cross_core_set_flag(free1, tla.arch.MTE1)
        tla.set_flag(op_free)
        step = 0
        for rb in tla.range_constexpr(4):
            if tla.const_expr(rb > 0 or bounded):
                tla.wait_flag(op_free)
                consume_operand(ka, l1_ptr, step, ready0, ready1, free0, free1, False)
                step = step + 1
                consume_operand(qa, l1_ptr, step, ready0, ready1, free0, free1, False)
                step = step + 1
                for cb in tla.range_constexpr(rb + int(bounded)):
                    if tla.const_expr(cb > 0):
                        tla.wait_flag(op_free)
                    consume_operand(
                        kb, l1_ptr, step, ready0, ready1, free0, free1, True
                    )
                    step = step + 1
                    tla.set_flag(loaded)
                    tla.wait_flag(loaded)
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
                    tla.set_flag(op_free)
            if tla.const_expr(not bounded):
                diag0 = tla.make_tensor_like(
                    ck + rb * 256 + rb * 1024, row_major(ck, 16, 16), tla.arch.L0Clayout
                )
                diag1 = tla.make_tensor_like(
                    cq + rb * 256 + rb * 1024, row_major(cq, 16, 16), tla.arch.L0Clayout
                )
                for active in tla.range(16):
                    tla.wait_flag(op_free)
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
                    tla.set_flag(loaded)
                    tla.wait_flag(loaded)
                    tla.mmad(diag0, ka, kb, init_c=active == 0)
                    tla.mmad(diag1, qa, kb, init_c=active == 0)
                    tla.set_flag(op_free)
        tla.wait_flag(op_free)
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
        tla.cross_core_wait_flag(free0, tla.arch.VECTOR)
        tla.cross_core_wait_flag(free1, tla.arch.VECTOR)


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
    if tla.const_expr(save):
        tla.set_flag(cube_done)
        tla.wait_flag(cube_done)
        tla.copy(public_inverse, acc)
    tla.copy(resident, acc, tla.params.CopyL0C2DstParams(unit_flag=3))
    tla.set_flag(resident_ready)


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
    a = tla.make_tensor_like(a_ptr, row_major(a_ptr, 64, 64), tla.arch.zN)
    b = tla.make_tensor_like(b_ptr, row_major(b_ptr, 64, 128), tla.arch.nZ)
    cw = tla.make_tensor_like(c_ptr, row_major(c_ptr, 64, 128), tla.arch.L0Clayout)
    cu = tla.make_tensor_like(
        c_ptr + 8192, row_major(c_ptr, 64, 128), tla.arch.L0Clayout
    )
    tla.wait_flag(resident_ready)
    tla.copy(a, resident)
    tla.cross_core_wait_flag(k_ready, tla.arch.MTE1)
    tla.copy(b, k_beta)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    tla.mmad(cw, a, b, init_c=True)
    tla.set_flag(b_free)
    tla.set_flag(result_ready)
    tla.wait_flag(result_ready)
    tla.copy(out_w, cw)
    tla.wait_flag(b_free)
    tla.cross_core_wait_flag(v_ready, tla.arch.MTE1)
    tla.copy(b, v_beta)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    tla.mmad(cu, a, b, init_c=True)
    tla.set_flag(result_ready)
    tla.wait_flag(result_ready)
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
    resident_ready = tla.flag("resident_ready", tla.arch.FIX, tla.arch.MTE1)
    cast_free = tla.flag("cast_free", tla.arch.MTE3, tla.arch.VECTOR)
    gm_loaded = tla.flag("gm_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    v_done = tla.flag("v_done", tla.arch.VECTOR, tla.arch.MTE3)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)
    left_free = tla.flag("left_free", tla.arch.CUBE, tla.arch.MTE1)
    cube_done = tla.flag("cube_done", tla.arch.CUBE, tla.arch.FIX)
    gm_ready = tla.flag("gm_ready", tla.arch.MTE2, tla.arch.MTE1)
    ready0 = tla.flag("ready0", tla.arch.MTE1, tla.arch.CUBE)
    ready1 = tla.flag("ready1", tla.arch.MTE1, tla.arch.CUBE)
    free0 = tla.flag("free0", tla.arch.CUBE, tla.arch.MTE1)
    free1 = tla.flag("free1", tla.arch.CUBE, tla.arch.MTE1)
    fix_ready = tla.flag("fix_ready", tla.arch.FIX, tla.arch.MTE1)
    acc_free = tla.flag("acc_free", tla.arch.FIX, tla.arch.CUBE)
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
    peer_loaded = tla.flag("s1_peer_loaded", tla.arch.MTE2, tla.arch.VECTOR)
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
        tla.pipe_barrier(tla.arch.MTE2)
        for i in tla.range_constexpr(4):
            tla.copy(
                tla.tile_view(identity, tla.make_shape(16, 16), tla.make_coord(i, i)),
                row_major(identity16.ptr, 16, 16),
            )
        tla.set_flag(gm_ready)
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
        tla.set_flag(gm_loaded)
        tla.wait_flag(gm_loaded)
        zero_qkv_tail(uq, uk, uv, valid)
        prepare_gate_input(raw_gate, gate_bf16, valid)
        prepare_beta_input(beta_io, raw_beta, valid, beta_pitch)
        if tla.const_expr(normalize):
            compute_qk_partial_sums(uq, uk, at(stats, 0, 128))
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            tla.copy(
                row_major(exchange.ptr + aiv * 128, 1, 128),
                row_major(stats.ptr, 1, 128),
            )
            tla.cross_core_set_flag(published, tla.arch.MTE3)
            tla.cross_core_wait_flag(published, tla.arch.MTE2)
            tla.copy(
                row_major(stats.ptr + 128, 1, 128),
                row_major(exchange.ptr + (1 - aiv) * 128, 1, 128),
            )
            tla.set_flag(peer_loaded)
        activate_gate(gate_bf16, bias_ub, alpha_ub, ug, valid, mode, lower, True)
        if tla.const_expr(normalize):
            tla.wait_flag(peer_loaded)
            tla.cross_core_set_flag(consumed, tla.arch.MTE2)
            tla.cross_core_wait_flag(consumed, tla.arch.MTE3)
            compute_qk_inverse_vectors(
                at(stats, 0, 128), at(stats, 128, 128), at(stats, 256, 128)
            )
            normalize_local_qk(uq, uk, at(stats, 256, 128))
        activate_beta(raw_beta, ub, valid, beta_sigmoid, allow_negative)
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
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
                tla.set_flag(v_done)
                tla.wait_flag(v_done)
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
                        tla.set_flag(cast_free)
                        tla.wait_flag(cast_free)
                        beta_to_column(at(stats, 256, 64), beta_column)
                        tla.set_flag(v_done)
                        tla.wait_flag(v_done)
                        tla.copy(
                            strided_rows(rstd.ptr, valid, 1, rstd_stride),
                            strided_rows(beta_column.ptr, valid, 1, 8),
                        )
                        tla.set_flag(cast_free)
                        tla.wait_flag(cast_free)
                        beta_to_column(at(stats, 320, 64), beta_column)
                        tla.set_flag(v_done)
                        tla.wait_flag(v_done)
                        tla.copy(
                            strided_rows(
                                rstd.ptr + rstd_k_offset, valid, 1, rstd_stride
                            ),
                            strided_rows(beta_column.ptr, valid, 1, 8),
                        )
    if tla.const_expr(shared):
        with tla.cube():
            tla.cross_core_wait_flag(score_ready, tla.arch.MTE1)
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
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            tla.copy(
                tla.tile_view(
                    shared_score_l1(score_ptr),
                    tla.make_shape(224, 64),
                    tla.make_coord(0, aiv),
                ),
                padded_nz(packed.ptr, 224),
            )
            tla.cross_core_set_flag(score_ready, tla.arch.MTE3)
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
        tla.set_flag(cube_done)
        tla.wait_flag(cube_done)
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
        tla.cross_core_set_flag(score_done, tla.arch.FIX)
        tla.cross_core_wait_flag(nt_ready, tla.arch.MTE1)
        tla.wait_flag(gm_ready)
        tla.set_flag(free0)
        tla.set_flag(free1)
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
        tla.wait_flag(free0)
        tla.wait_flag(free1)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        aiv = tla.arch.sub_block_idx()
        tla.cross_core_wait_flag(score_done, tla.arch.VECTOR)
        apply_beta_and_finish_t(kkt, ub, nt, aiv)
        nd2nz_tile(nt, ntnz, 32, 64)
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
        tla.copy(
            tla.tile_view(
                matrix64(lp, tla.arch.zN),
                tla.make_shape(32, 64),
                tla.make_coord(aiv, 0),
            ),
            padded_nz(ntnz.ptr, 32),
        )
        tla.cross_core_set_flag(nt_ready, tla.arch.MTE3)
        beta_weighted_operand(uk, ub, ug, cast, True)
        nd2nz_tile(cast, nz0, 64, 64)
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
        kb = tla.make_tensor_like(kb_ptr, row_major(kb_ptr, 64, 128), tla.arch.zN)
        vb = tla.make_tensor_like(vb_ptr, row_major(vb_ptr, 64, 128), tla.arch.zN)
        tla.copy(
            tla.tile_view(kb, tla.make_shape(64, 64), tla.make_coord(0, aiv)),
            padded_nz(nz0.ptr, 64),
        )
        tla.cross_core_set_flag(kb_ready, tla.arch.MTE3)
        tla.set_flag(cast_free)
        tla.wait_flag(cast_free)
        beta_weighted_operand(uv, ub, ug, cast, False)
        nd2nz_tile(cast, nz1, 64, 64)
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
        tla.copy(
            tla.tile_view(vb, tla.make_shape(64, 64), tla.make_coord(0, aiv)),
            padded_nz(nz1.ptr, 64),
        )
        tla.cross_core_set_flag(vb_ready, tla.arch.MTE3)
        prepare_decayed_q(uq, ug, uqg, uqs, scale)
        restored_state_inputs(uk, ug, ukg, ugamma)
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
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
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
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
    resident_ready = tla.flag("pipeline_resident", tla.arch.FIX, tla.arch.MTE1)
    cast_free = tla.flag("pipeline_cast_free", tla.arch.MTE3, tla.arch.VECTOR)
    gm_loaded = tla.flag("pipeline_gm_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    v_done = tla.flag("pipeline_v_done", tla.arch.VECTOR, tla.arch.MTE3)
    v_input_free = tla.flag("pipeline_v_input_free", tla.arch.VECTOR, tla.arch.MTE2)
    qk_input_free = tla.flag("pipeline_qk_input_free", tla.arch.MTE3, tla.arch.MTE2)
    l0_loaded = tla.flag("pipeline_l0_loaded", tla.arch.MTE1, tla.arch.CUBE)
    left_free = tla.flag("pipeline_left_free", tla.arch.CUBE, tla.arch.MTE1)
    cube_done = tla.flag("pipeline_cube_done", tla.arch.CUBE, tla.arch.FIX)
    gm_ready = tla.flag("pipeline_gm_ready", tla.arch.MTE2, tla.arch.MTE1)
    ready0 = tla.flag("pipeline_ready0", tla.arch.MTE1, tla.arch.CUBE)
    ready1 = tla.flag("pipeline_ready1", tla.arch.MTE1, tla.arch.CUBE)
    free0 = tla.flag("pipeline_free0", tla.arch.CUBE, tla.arch.MTE1)
    free1 = tla.flag("pipeline_free1", tla.arch.CUBE, tla.arch.MTE1)
    fix_ready = tla.flag("pipeline_fix_ready", tla.arch.FIX, tla.arch.MTE1)
    acc_free = tla.flag("pipeline_acc_free", tla.arch.FIX, tla.arch.CUBE)
    peer_loaded = tla.flag("pipeline_peer_loaded", tla.arch.MTE2, tla.arch.VECTOR)
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
        tla.pipe_barrier(tla.arch.MTE2)
        for i in tla.range_constexpr(4):
            tla.copy(
                tla.tile_view(identity, tla.make_shape(16, 16), tla.make_coord(i, i)),
                row_major(identity16.ptr, 16, 16),
            )
        tla.set_flag(gm_ready)
        tla.wait_flag(gm_ready)
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
                tla.set_flag(v_input_free)
                tla.wait_flag(v_input_free)
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
                tla.cross_core_wait_flag(score_done, tla.arch.VECTOR)
                if task > lo + tla.Int64(1):
                    tla.cross_core_wait_flag(c3_free, tla.arch.MTE3)
                finish_t_nz(kkt, ub, ntnz, aiv_1)
                tla.set_flag(v_done)
                tla.wait_flag(v_done)
                tla.copy(
                    tla.tile_view(
                        negative, tla.make_shape(32, 64), tla.make_coord(aiv_1, 0)
                    ),
                    padded_nz(ntnz.ptr, 32),
                )
                tla.cross_core_set_flag(nt_ready, tla.arch.MTE3)
                tla.set_flag(cast_free)
                tla.wait_flag(cast_free)
                beta_weighted_nz(old_k, ub, old_g, nz0, True)
                tla.set_flag(v_done)
                tla.wait_flag(v_done)
                tla.copy(
                    tla.tile_view(kb, tla.make_shape(64, 64), tla.make_coord(0, aiv_1)),
                    padded_nz(nz0.ptr, 64),
                )
                tla.cross_core_set_flag(kb_ready, tla.arch.MTE3)
                tla.set_flag(cast_free)
                tla.wait_flag(cast_free)
                beta_weighted_nz(uv, ub, old_g, nz1, False)
                tla.set_flag(v_done)
                tla.wait_flag(v_done)
                tla.copy(
                    tla.tile_view(vb, tla.make_shape(64, 64), tla.make_coord(0, aiv_1)),
                    padded_nz(nz1.ptr, 64),
                )
                tla.cross_core_set_flag(vb_ready, tla.arch.MTE3)
                tla.set_flag(v_input_free)
                tla.wait_flag(v_input_free)
            with tla.cube():
                tla.cross_core_wait_flag(nt_ready, tla.arch.MTE1)
                tla.set_flag(free0)
                tla.set_flag(free1)
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
                tla.wait_flag(free0)
                tla.wait_flag(free1)
                tla.set_flag(acc_free)
                tla.wait_flag(acc_free)
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
                tla.set_flag(gm_loaded)
                tla.wait_flag(gm_loaded)
                if valid < 64:
                    zero_qkv_tail(uq, uk, uv, valid)
                if tla.const_expr(not direct_beta):
                    prepare_beta_input(beta_io, raw_beta, valid, beta_pitch)
                if tla.const_expr(normalize):
                    compute_qk_partial_sums(uq, uk, at(stats, 0, 128))
                    tla.set_flag(v_done)
                    tla.wait_flag(v_done)
                    tla.copy(
                        row_major(exchange.ptr + aiv_2 * 128, 1, 128),
                        row_major(stats.ptr, 1, 128),
                    )
                    tla.cross_core_set_flag(published, tla.arch.MTE3)
                    tla.cross_core_wait_flag(published, tla.arch.MTE2)
                    tla.copy(
                        row_major(stats.ptr + 128, 1, 128),
                        row_major(exchange.ptr + (1 - aiv_2) * 128, 1, 128),
                    )
                    tla.set_flag(peer_loaded)
                if valid == 64:
                    activate_gate(
                        raw_gate, bias_ub, alpha_ub, ug, 64, mode, lower, True
                    )
                else:
                    activate_gate(
                        raw_gate, bias_ub, alpha_ub, ug, valid, mode, lower, True
                    )
                if tla.const_expr(normalize):
                    tla.wait_flag(peer_loaded)
                    tla.cross_core_set_flag(consumed, tla.arch.MTE2)
                    tla.cross_core_wait_flag(consumed, tla.arch.MTE3)
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
                tla.set_flag(v_done)
                tla.wait_flag(v_done)
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
                                tla.set_flag(cast_free)
                                tla.wait_flag(cast_free)
                                beta_to_column(at(stats, 256, 64), beta_column)
                                tla.set_flag(v_done)
                                tla.wait_flag(v_done)
                                tla.copy(
                                    strided_rows(
                                        rstd.ptr + no // tla.Int64(128),
                                        valid,
                                        1,
                                        rstd_stride,
                                    ),
                                    strided_rows(beta_column.ptr, valid, 1, 8),
                                )
                                tla.set_flag(cast_free)
                                tla.wait_flag(cast_free)
                                beta_to_column(at(stats, 320, 64), beta_column)
                                tla.set_flag(v_done)
                                tla.wait_flag(v_done)
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
                tla.set_flag(cast_free)
                tla.wait_flag(cast_free)
                prepare_decayed_q(uq, ug, uqg, uqs, scale)
                tla.set_flag(v_done)
                tla.wait_flag(v_done)
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
                    tla.set_flag(qk_input_free)
                    tla.wait_flag(qk_input_free)
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
                    tla.cross_core_wait_flag(score_ready, tla.arch.MTE1)
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
                    tla.set_flag(v_done)
                    tla.wait_flag(v_done)
                    tla.copy(
                        tla.tile_view(
                            shared_score_l1(score_ptr),
                            tla.make_shape(224, 64),
                            tla.make_coord(0, aiv_3),
                        ),
                        padded_nz(packed.ptr, 224),
                    )
                    tla.cross_core_set_flag(score_ready, tla.arch.MTE3)
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
                tla.set_flag(cube_done)
                tla.wait_flag(cube_done)
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
                tla.cross_core_set_flag(score_done, tla.arch.FIX)
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
                        tla.set_flag(cast_free)
                        tla.wait_flag(cast_free)
                        beta_to_column(ub, beta_column)
                        tla.set_flag(v_done)
                        tla.wait_flag(v_done)
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
                tla.cross_core_set_flag(c3_free, tla.arch.MTE1)
                tla.set_flag(acc_free)
                tla.wait_flag(acc_free)
            with tla.vector():
                aiv_5 = tla.arch.sub_block_idx()
                restored_state_inputs(old_k, old_g, ukg, ugamma)
                mask_mqk(old_mqk, aq, aiv_5)
                tla.set_flag(v_done)
                tla.wait_flag(v_done)
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
            tla.cross_core_wait_flag(c3_free, tla.arch.MTE3)
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
        tla.pipe_barrier(tla.arch.MTE2)
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
        tla.flag("s2_gm_l1", tla.arch.MTE2, tla.arch.MTE1),
        tla.flag("s2_gm_ub", tla.arch.MTE2, tla.arch.VECTOR),
        tla.flag("s2_input_free0", tla.arch.VECTOR, tla.arch.MTE2),
        tla.flag("s2_input_free1", tla.arch.VECTOR, tla.arch.MTE2),
        tla.flag("s2_l1_free0", tla.arch.MTE1, tla.arch.MTE2),
        tla.flag("s2_l1_free1", tla.arch.MTE1, tla.arch.MTE2),
        tla.flag("s2_v_done", tla.arch.VECTOR, tla.arch.MTE3),
        tla.flag("s2_history_free", tla.arch.MTE3, tla.arch.VECTOR),
        tla.flag("s2_loaded", tla.arch.MTE1, tla.arch.CUBE),
        tla.flag("s2_cube_done", tla.arch.CUBE, tla.arch.FIX),
        tla.flag("s2_acc_free", tla.arch.FIX, tla.arch.CUBE),
        tla.flag("s2_output_operands_free", tla.arch.CUBE, tla.arch.MTE1),
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
        tla.set_flag(gm_l1)
        tla.set_flag(acc_free)
        tla.set_flag(output_operands_free)
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
            tla.wait_flag(gm_l1)
            tla.cross_core_wait_flag(h_ready, tla.arch.MTE1)
            compute_state_projections(
                ql, wl, hl, a0, a1, hb, cu, co, dv, loaded, cube_done, acc_free
            )
            tla.wait_flag(output_operands_free)
            load_recurrent_operands(kl, al, kb, qa, dv, loaded)
            tla.wait_flag(loaded)
            if chunk + 2 < chunks:
                if slot == 0:
                    tla.set_flag(l1_free0)
                else:
                    tla.set_flag(l1_free1)
            if chunk + 1 < chunks:
                if chunk > 0:
                    if next_slot == 0:
                        tla.wait_flag(l1_free0)
                    else:
                        tla.wait_flag(l1_free1)
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
            tla.wait_flag(cube_done)
            tla.copy(
                row_major(du, 64, dv),
                tla.make_tensor_like(cu, row_major(cu, 64, dv), tla.arch.L0Clayout),
                tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
            )
            tla.cross_core_set_flag(du_ready, tla.arch.FIX)
            tla.cross_core_wait_flag(u_ready, tla.arch.MTE1)
            compute_state_delta(ul, kb, ua, ch, dv, loaded)
            tla.set_flag(cube_done)
            tla.wait_flag(cube_done)
            tla.copy(
                strided_rows(delta.ptr, dv, 128, 64),
                tla.make_tensor_like(ch, row_major(ch, dv, 128), tla.arch.L0Clayout),
                tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N),
            )
            tla.cross_core_set_flag(dh_ready, tla.arch.FIX)
            accumulate_output(ul, qa, vb, co, dv, loaded)
            tla.set_flag(output_operands_free)
            tla.set_flag(cube_done)
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
                tla.set_flag(gm_l1)
            valid = 64
            if chunk + 1 == chunks:
                valid = valid_last
            tla.wait_flag(cube_done)
            tla.copy(
                strided_rows(
                    out + chunk * 64 * output_stride + dv_idx * dv,
                    valid,
                    dv,
                    output_stride,
                ),
                tla.make_tensor_like(co, row_major(co, 64, dv), tla.arch.L0Clayout),
            )
            tla.set_flag(acc_free)
        tla.wait_flag(acc_free)
        tla.wait_flag(output_operands_free)
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
        tla.set_flag(gm_ub)
        tla.wait_flag(gm_ub)
        cast_state_nz(state, snz, dv)
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
        hl = tla.make_tensor_like(lbase + 57344, row_major(lbase, dv, 128), tla.arch.zN)
        tla.copy(
            tla.tile_view(hl, tla.make_shape(dv, 64), tla.make_coord(0, aiv)),
            padded_nz(snz.ptr, dv),
        )
        tla.cross_core_set_flag(h_ready, tla.arch.MTE3)
        tla.set_flag(history_free)
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
                tla.wait_flag(gm_ub)
            if chunk + 1 < chunks:
                if chunk > 0:
                    if next_slot == 0:
                        tla.wait_flag(input_free0)
                    else:
                        tla.wait_flag(input_free1)
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
                tla.set_flag(gm_ub)
            tla.cross_core_wait_flag(du_ready, tla.arch.VECTOR)
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
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            ul = tla.make_tensor_like(
                lbase + 90112 + slot * 8192, row_major(lbase, 64, dv), tla.arch.zN
            )
            tla.copy(
                tla.tile_view(ul, tla.make_shape(32, dv), tla.make_coord(aiv, 0)),
                padded_nz(unz.ptr, 32, dv),
            )
            tla.cross_core_set_flag(u_ready, tla.arch.MTE3)
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
            tla.cross_core_wait_flag(dh_ready, tla.arch.VECTOR)
            tla.wait_flag(history_free)
            if tla.const_expr(save_history):
                tla.copy(row_major(history_copy, dv, 64), row_major(state.ptr, dv, 64))
                tla.set_flag(v_done)
                tla.wait_flag(v_done)
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
            tla.set_flag(history_free)
            update_state_nz(state, delta, flat(gamma_ptr + slot * 64, 64), snz, dv)
            if chunk + 2 < chunks:
                if slot == 0:
                    tla.set_flag(input_free0)
                else:
                    tla.set_flag(input_free1)
            if chunk + 1 < chunks:
                tla.set_flag(v_done)
                tla.wait_flag(v_done)
                hl = tla.make_tensor_like(
                    lbase + 57344 + next_slot * 16384,
                    row_major(lbase, dv, 128),
                    tla.arch.zN,
                )
                tla.copy(
                    tla.tile_view(hl, tla.make_shape(dv, 64), tla.make_coord(0, aiv)),
                    padded_nz(snz.ptr, dv),
                )
                tla.cross_core_set_flag(h_ready, tla.arch.MTE3)
        tla.wait_flag(history_free)
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
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
        tla.cross_core_wait_flag(aiv_ready, tla.arch.SCALAR)
        tla.cross_core_set_flag(aic_all, tla.arch.FIX)
        tla.cross_core_wait_flag(aic_all, tla.arch.SCALAR)
        tla.cross_core_set_flag(aic_release, tla.arch.MTE3)
    with tla.vector():
        tla.pipe_barrier(tla.pipes.ALL)
        tla.cross_core_set_flag(aiv_ready, tla.arch.MTE3)
        tla.cross_core_wait_flag(aic_release, tla.arch.SCALAR)


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
        diagnostic_phase=0,
        diagnostic_round=-1,
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
        self.diagnostic_phase, self.diagnostic_round = (
            diagnostic_phase,
            diagnostic_round,
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
        c = tla.allocate(32768, tla.Float32, tla.AddressSpace.l0c, 512)
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
        if tla.const_expr(self.diagnostic_round >= 0):
            begin = tla.Int32(self.diagnostic_round)
            end_round = tla.Int32(self.diagnostic_round + 1)
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
            if tla.const_expr(self.diagnostic_phase != 2):
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
                                (storage_batch * i64(self.nq) + qhead)
                                * i64(self.length)
                                + token
                            ) * i64(128)
                            if tla.const_expr(self.layout == "TND"):
                                no = (token * i64(self.nq) + qhead) * i64(128)
                            ro = no // i64(128)
                            chunk_slot = (
                                logical_batch * i64(self.chunk_capacity) + chunk
                            )
                            if tla.const_expr(self.layout == "TND"):
                                chunk_slot = (
                                    token_start // i64(64) + logical_batch + chunk
                                )
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
            if tla.const_expr(self.diagnostic_phase != 1):
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
    load_selected_pair(negative, diagonal, a1, b1, zero, ready1, free1, 3, 1)
    inverse_mmad(ct, a1, b1, ready1, free1, True, 3)
    inverse_fix(temporary, ct, fix_ready, acc_free)
    tla.wait_flag(free0)
    tla.copy(a0, diagonal)
    tla.copy(b0, identity)
    copy_selected_fractals(b0, temporary, 3)
    tla.set_flag(ready0)
    inverse_mmad(ci, a0, b0, ready0, free0, True, 3)
    inverse_fix(inverse, ci, fix_ready, acc_free)
    load_selected_pair(negative, inverse, a1, b1, zero, ready1, free1, 0, 4)
    tla.wait_flag(free0)
    tla.copy(a0, inverse)
    tla.copy(b0, zero)
    inverse_mmad(ct, a1, b1, ready1, free1, True, 3)
    inverse_fix(temporary, ct, fix_ready, acc_free)
    copy_selected_fractals(b0, temporary, 5)
    tla.set_flag(ready0)
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
    with tla.vector():
        aiv = tla.arch.sub_block_idx()
        rows = next_valid - half * 32
        if rows > 32:
            rows = 32
        if rows > 0:
            # Raw Q/K have already been consumed by normalization/panel packing.
            tla.set_flag(free)
            tla.wait_flag(free)
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
                            tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.ZERO,
                                sat_mode=tla.params.SatMode.NOSAT,
                                round_mode=tla.params.RoundMode.CAST_RINT,
                            ),
                        )
                        kv = kb.to(
                            tla.Float32,
                            tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.ZERO,
                                sat_mode=tla.params.SatMode.NOSAT,
                                round_mode=tla.params.RoundMode.CAST_RINT,
                            ),
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
    ready0 = tla.flag("ps_left_ready0", tla.arch.MTE1, tla.arch.CUBE)
    ready1 = tla.flag("ps_left_ready1", tla.arch.MTE1, tla.arch.CUBE)
    free0 = tla.flag("ps_left_free0", tla.arch.CUBE, tla.arch.MTE1)
    free1 = tla.flag("ps_left_free1", tla.arch.CUBE, tla.arch.MTE1)
    tla.set_flag(free0)
    tla.set_flag(free1)
    for kind in tla.range_constexpr(2):
        # This CUBE->MTE1 edge also releases the previous chunk's output
        # operands at bp[0:8192]. All RHS preloads must follow it.
        tla.wait_flag(free0)
        if tla.const_expr(shared and kind == 0):
            for j in tla.range_constexpr(6):
                right = tla.make_tensor_like(
                    right_ptr + j * 2048, row_major(right_ptr, 128, 16), tla.arch.nZ
                )
                tla.copy(right, right_transpose(cache.ptr, j, True))
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
        left0 = tla.make_tensor_like(
            left_ptr, row_major(left_ptr, 16, 128), tla.arch.zN
        )
        tla.copy(
            left0,
            tla.tile_view(cache, tla.make_shape(16, 128), tla.make_coord(kind, 0)),
        )
        tla.set_flag(ready0)
        for row in tla.range_constexpr(4):
            left = tla.make_tensor_like(
                left_ptr + (row % 2) * 2048, row_major(left_ptr, 16, 128), tla.arch.zN
            )
            if tla.const_expr(row % 2 == 0):
                tla.wait_flag(ready0)
            else:
                tla.wait_flag(ready1)
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
                        tla.set_flag(loaded)
                        tla.wait_flag(loaded)
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
                        tla.wait_flag(free1)
                    else:
                        tla.wait_flag(free0)
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
                        tla.set_flag(ready1)
                    else:
                        tla.set_flag(ready0)
            if tla.const_expr(row % 2 == 0):
                tla.set_flag(free0)
            else:
                tla.set_flag(free1)
        if tla.const_expr(kk is not None):
            result = ck
            output = kk
            notification = kk_ready
            if tla.const_expr(kind == 1):
                result = cq
                output = mq
                notification = mq_ready
            tla.set_flag(cube_done)
            tla.wait_flag(cube_done)
            tla.copy(
                row_major(output.ptr, 64, 64),
                tla.make_tensor_like(
                    result, row_major(result, 64, 64), tla.arch.L0Clayout
                ),
                tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
            )
            tla.cross_core_set_flag(notification, tla.arch.FIX)
    tla.wait_flag(free0)
    tla.wait_flag(free1)


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
    w_ready = tla.flag("pr_w_ready", tla.arch.FIX, tla.arch.MTE1)
    vnew_ready = tla.flag("pr_vnew_ready", tla.arch.FIX, tla.arch.MTE1)
    output_read = tla.flag("pr_output_read", tla.arch.FIX, tla.arch.CUBE)
    output_done = tla.flag("pr_output_done", tla.arch.CUBE, tla.arch.FIX)
    b_free = tla.flag("pr_b_free", tla.arch.CUBE, tla.arch.MTE1)
    output_free = tla.flag("pr_output_free", tla.arch.FIX, tla.arch.CUBE)
    delta_read = tla.flag("pr_delta_read", tla.arch.FIX, tla.arch.CUBE)
    inverse_a = tla.make_tensor_like(ap + 12288, row_major(ap, 64, 64), tla.arch.zN)
    beta_b = tla.make_tensor_like(bp, row_major(bp, 64, 128), tla.arch.nZ)
    cw = tla.make_tensor_like(cp, row_major(cp, 64, 128), tla.arch.L0Clayout)
    cu = tla.make_tensor_like(cp + 8192, row_major(cp, 64, 128), tla.arch.L0Clayout)
    co = tla.make_tensor_like(cp + 16384, row_major(cp, 64, 128), tla.arch.L0Clayout)
    tla.wait_flag(resident_ready)
    tla.copy(inverse_a, resident)
    if tla.const_expr(not inputs_ready):
        tla.cross_core_wait_flag(k_ready, tla.arch.MTE1)
    tla.copy(beta_b, kbeta)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    tla.mmad(cw, inverse_a, beta_b, init_c=True)
    tla.set_flag(b_free)
    tla.set_flag(cube_done)
    tla.wait_flag(cube_done)
    if tla.const_expr(save_cache):
        tla.copy(
            strided_rows(out_w, valid, 128, stride),
            cw,
            tla.params.CopyL0C2DstParams(
                quant_mode=tla.params.QuantMode.PER_TENSOR, quant_scale=-1.0
            ),
        )
    tla.copy(negative_w, cw)
    tla.set_flag(w_ready)
    tla.wait_flag(b_free)
    if tla.const_expr(not inputs_ready):
        tla.cross_core_wait_flag(v_ready, tla.arch.MTE1)
    tla.copy(beta_b, vbeta)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    tla.mmad(cu, inverse_a, beta_b, init_c=True)
    if tla.const_expr(save_cache):
        tla.set_flag(cube_done)
        tla.wait_flag(cube_done)
        tla.copy(strided_rows(out_u, valid, 128, stride), cu)
        tla.set_flag(output_read)
    state_b = tla.make_tensor_like(bp + 8192, row_major(bp, 128, 128), tla.arch.nZ)
    state_t = tla.make_tensor_like(
        state.ptr, row_major(state.ptr, 128, 128), tla.arch.nZ
    )
    wa = tla.make_tensor_like(ap + 16384, row_major(ap, 64, 128), tla.arch.zN)
    qa = tla.make_tensor_like(ap + 24576, row_major(ap, 64, 128), tla.arch.zN)
    tla.cross_core_wait_flag(state_ready, tla.arch.MTE1)
    tla.copy(state_b, state_t)
    tla.wait_flag(w_ready)
    tla.copy(wa, negative_w)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    if tla.const_expr(save_cache):
        tla.wait_flag(output_read)
    tla.mmad(cu, wa, state_b, init_c=False)
    tla.set_flag(cube_done)
    tla.copy(qa, qs)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    if tla.const_expr(output_overlap):
        if first_chunk:
            tla.set_flag(output_free)
        tla.wait_flag(output_free)
    tla.mmad(co, qa, state_b, init_c=True)
    tla.wait_flag(cube_done)
    if tla.const_expr(save_cache):
        tla.copy(strided_rows(out_vnew, valid, 128, stride), cu)
    tla.copy(vnew, cu)
    tla.set_flag(vnew_ready)
    kb = tla.make_tensor_like(bp + 24576, row_major(bp, 64, 128), tla.arch.nZ)
    aq = tla.make_tensor_like(ap + 8192, row_major(ap, 64, 64), tla.arch.zN)
    vt = tla.make_tensor_like(vnew.ptr, row_major(vnew.ptr, 128, 64), tla.arch.nZ)
    va = tla.make_tensor_like(ap, row_major(ap, 128, 64), tla.arch.zN)
    vb = tla.make_tensor_like(bp, row_major(bp, 64, 128), tla.arch.nZ)
    tla.copy(kb, kg)
    tla.wait_flag(vnew_ready)
    tla.copy(va, vt)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    # vnew_ready -> MTE1 loads -> loaded already closes FIX's cu reader.
    ch = tla.make_tensor_like(cp, row_major(cp, 128, 128), tla.arch.L0Clayout)
    tla.mmad(ch, va, kb, init_c=True)
    tla.set_flag(cube_done)
    tla.wait_flag(cube_done)
    tla.copy(
        strided_rows(delta, 128, 128, 64),
        ch,
        tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_N),
    )
    tla.cross_core_set_flag(delta_ready, tla.arch.FIX)
    if tla.const_expr(output_overlap):
        tla.set_flag(delta_read)
    # Only Aqk depends on recur_ready; Qs and Kg were acquired in the bulk bank.
    tla.cross_core_wait_flag(recurrent_ready, tla.arch.MTE1)
    tla.copy(aq, aqk)
    tla.copy(vb, vnew)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    tla.mmad(co, aq, vb, init_c=False)
    tla.set_flag(output_done)
    tla.wait_flag(output_done)
    tla.copy(strided_rows(out, valid, 128, stride), co)
    if tla.const_expr(output_overlap):
        tla.set_flag(output_free)
        tla.wait_flag(delta_read)
        if last_chunk:
            tla.wait_flag(output_free)


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
    prefetch_free = tla.flag("pc_prefetch_free", tla.arch.VECTOR, tla.arch.MTE2)
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
    gm_loaded = tla.flag("pc_gm_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    cast_free = tla.flag("pc_cast_free", tla.arch.MTE3, tla.arch.VECTOR)
    peer_loaded = tla.flag("pc_peer_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    published = tla.cross_flag("pc_partial_ready", mode=1)
    consumed = tla.cross_flag("pc_partial_free", mode=1)
    identity_ready = tla.flag("pc_identity_ready", tla.arch.MTE2, tla.arch.MTE1)
    loaded = tla.flag("pc_loaded", tla.arch.MTE1, tla.arch.CUBE)
    cube_done = tla.flag("pc_cube_done", tla.arch.CUBE, tla.arch.FIX)
    v_done = tla.flag("pc_v_done", tla.arch.VECTOR, tla.arch.MTE3)
    ready0 = tla.flag("pc_ready0", tla.arch.MTE1, tla.arch.CUBE)
    ready1 = tla.flag("pc_ready1", tla.arch.MTE1, tla.arch.CUBE)
    free0 = tla.flag("pc_free0", tla.arch.CUBE, tla.arch.MTE1)
    free1 = tla.flag("pc_free1", tla.arch.CUBE, tla.arch.MTE1)
    fix_ready = tla.flag("pc_fix_ready", tla.arch.FIX, tla.arch.MTE1)
    acc_free = tla.flag("pc_acc_free", tla.arch.FIX, tla.arch.CUBE)
    resident_ready = tla.flag("pc_resident_ready", tla.arch.FIX, tla.arch.MTE1)
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
        tla.pipe_barrier(tla.arch.MTE2)
        for j in tla.range_constexpr(4):
            tla.copy(
                tla.tile_view(identity, tla.make_shape(16, 16), tla.make_coord(j, j)),
                row_major(identity16.ptr, 16, 16),
            )
        tla.set_flag(identity_ready)
    with tla.vector():
        aiv = tla.arch.sub_block_idx()
        # Incoming state is ready on V, so publish the transitive MTE2->V->MTE3
        # edge before a history DMA reads the same UB allocation.
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
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
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            tla.copy(
                tla.tile_view(
                    score_l1(lp), tla.make_shape(288, 64), tla.make_coord(0, aiv)
                ),
                padded_nz(packed.ptr, 288),
            )
            tla.cross_core_set_flag(scores_ready, tla.arch.MTE3)
        if tla.const_expr(prefetch):
            prefetch_half(q, k, raw_q, raw_k, next_valid, q_stride, 1, prefetch_free)
        with tla.cube():
            tla.cross_core_wait_flag(scores_ready, tla.arch.MTE1)
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
            tla.set_flag(cast_free)
            tla.wait_flag(cast_free)
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
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            tla.copy(
                tla.tile_view(
                    recurrent, tla.make_shape(256, 64), tla.make_coord(0, aiv_recurrent)
                ),
                padded_nz(recurrent_nz.ptr, 256),
            )
            tla.cross_core_set_flag(k_ready, tla.arch.MTE3)
            tla.cross_core_set_flag(v_ready, tla.arch.MTE3)
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
        tla.set_flag(cube_done)
        tla.wait_flag(cube_done)
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
        tla.cross_core_set_flag(scores_done, tla.arch.FIX)
        tla.cross_core_wait_flag(inverse_ready, tla.arch.MTE1)
        tla.wait_flag(identity_ready)
        tla.set_flag(free0)
        tla.set_flag(free1)
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
        tla.wait_flag(free0)
        tla.wait_flag(free1)
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
        tla.cross_core_wait_flag(scores_done, tla.arch.VECTOR)
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
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
        tla.copy(
            tla.tile_view(nt_l1, tla.make_shape(32, 64), tla.make_coord(aiv, 0)),
            padded_nz(ntnz.ptr, 32),
        )
        tla.copy(
            tla.tile_view(diagonal, tla.make_shape(32, 64), tla.make_coord(aiv, 0)),
            padded_nz(dnz.ptr, 32),
        )
        tla.cross_core_set_flag(inverse_ready, tla.arch.MTE3)
        if tla.const_expr(not bulk_operands):
            # API-062: no scaled FIX to L1. Negate K_beta exactly before BF16
            # packing, so Cube produces -W; the public W FIX reverses the sign.
            beta_weighted_operand(uk, ub, ug, cast, True, True)
            nd2nz_tile(cast, nz, 64, 64)
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            tla.copy(
                tla.tile_view(kl, tla.make_shape(64, 64), tla.make_coord(0, aiv)),
                padded_nz(nz.ptr, 64),
            )
            tla.cross_core_set_flag(k_ready, tla.arch.MTE3)
            tla.set_flag(cast_free)
            tla.wait_flag(cast_free)
            beta_weighted_operand(uv, ub, ug, cast, False)
            nd2nz_tile(cast, nz, 64, 64)
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            tla.copy(
                tla.tile_view(vl, tla.make_shape(64, 64), tla.make_coord(0, aiv)),
                padded_nz(nz.ptr, 64),
            )
            tla.cross_core_set_flag(v_ready, tla.arch.MTE3)
            tla.set_flag(cast_free)
            tla.wait_flag(cast_free)
            prepare_decayed_q(uq, ug, cast, decayed, scale)
            nd2nz_tile(decayed, nz, 64, 64)
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            if tla.const_expr(save_cache):
                tla.copy(
                    strided_rows(qg.ptr + aiv * 64, valid, 64, v_stride),
                    row_major(cast.ptr, valid, 64),
                )
            tla.copy(
                tla.tile_view(ql, tla.make_shape(64, 64), tla.make_coord(0, aiv)),
                padded_nz(nz.ptr, 64),
            )
            tla.set_flag(cast_free)
            tla.wait_flag(cast_free)
            restored_state_inputs(uk, ug, cast, ugamma)
            nd2nz_tile(cast, nz, 64, 64)
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            if tla.const_expr(save_cache):
                tla.copy(
                    strided_rows(kg.ptr + aiv * 64, valid, 64, v_stride),
                    row_major(cast.ptr, valid, 64),
                )
            tla.copy(
                tla.tile_view(kgl, tla.make_shape(64, 64), tla.make_coord(0, aiv)),
                padded_nz(nz.ptr, 64),
            )
            tla.set_flag(cast_free)
            tla.wait_flag(cast_free)
        mask_mqk(mq, public, aiv)
        nd2nz_tile(public, aq_nz, 32, 64)
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
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
        tla.cross_core_set_flag(recur_ready, tla.arch.MTE3)
        cast_state_snapshot(state, snapshot, 128)
        nd2nz_tile(snapshot, snz, 128, 64)
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
        tla.copy(
            tla.tile_view(hl, tla.make_shape(128, 64), tla.make_coord(0, aiv)),
            padded_nz(snz.ptr, 128),
        )
        tla.cross_core_set_flag(h_ready, tla.arch.MTE3)
        tla.cross_core_wait_flag(delta_ready, tla.arch.VECTOR)
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
    prefetch_free = tla.flag("pc_prefetch_free", tla.arch.VECTOR, tla.arch.MTE2)
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
    gm_loaded = tla.flag("pc_gm_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    cast_free = tla.flag("pc_cast_free", tla.arch.MTE3, tla.arch.VECTOR)
    input_dma_free = tla.flag("pc_input_dma_free", tla.arch.MTE3, tla.arch.MTE2)
    peer_loaded = tla.flag("pc_peer_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    published = tla.cross_flag("pc_partial_ready", mode=1)
    consumed = tla.cross_flag("pc_partial_free", mode=1)
    identity_ready = tla.flag("pc_identity_ready", tla.arch.MTE2, tla.arch.MTE1)
    loaded = tla.flag("pc_loaded", tla.arch.MTE1, tla.arch.CUBE)
    cube_done = tla.flag("pc_cube_done", tla.arch.CUBE, tla.arch.FIX)
    v_done = tla.flag("pc_v_done", tla.arch.VECTOR, tla.arch.MTE3)
    ready0 = tla.flag("pc_ready0", tla.arch.MTE1, tla.arch.CUBE)
    ready1 = tla.flag("pc_ready1", tla.arch.MTE1, tla.arch.CUBE)
    free0 = tla.flag("pc_free0", tla.arch.CUBE, tla.arch.MTE1)
    free1 = tla.flag("pc_free1", tla.arch.CUBE, tla.arch.MTE1)
    fix_ready = tla.flag("pc_fix_ready", tla.arch.FIX, tla.arch.MTE1)
    acc_free = tla.flag("pc_acc_free", tla.arch.FIX, tla.arch.CUBE)
    resident_ready = tla.flag("pc_resident_ready", tla.arch.FIX, tla.arch.MTE1)
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
            tla.pipe_barrier(tla.arch.MTE2)
            for j in tla.range_constexpr(4):
                tla.copy(
                    tla.tile_view(
                        identity, tla.make_shape(16, 16), tla.make_coord(j, j)
                    ),
                    row_major(identity16.ptr, 16, 16),
                )
            tla.set_flag(identity_ready)
    with tla.vector():
        aiv_entry = tla.arch.sub_block_idx()
        # Incoming state is ready on V, so publish the transitive MTE2->V->MTE3
        # edge before a history DMA reads the same UB allocation.
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
        if tla.const_expr(save_history):
            tla.copy(
                strided_rows(history.ptr + aiv_entry * 64, 128, 64, 128),
                row_major(state.ptr, 128, 64),
            )
        if first_chunk:
            cast_state_nz(state, snz, 128)
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            tla.copy(
                tla.tile_view(
                    hl, tla.make_shape(128, 64), tla.make_coord(0, aiv_entry)
                ),
                padded_nz(snz.ptr, 128),
            )
            tla.cross_core_set_flag(h_ready, tla.arch.MTE3)
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
                tla.set_flag(cast_free)
                tla.wait_flag(cast_free)
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
                tla.set_flag(v_done)
                tla.wait_flag(v_done)
                tla.copy(
                    tla.tile_view(
                        score_l1(lp, shared_scores),
                        tla.make_shape(score_rows, 64),
                        tla.make_coord(0, aiv_score),
                    ),
                    padded_nz(packed.ptr, score_rows),
                )
                tla.cross_core_set_flag(scores_ready, tla.arch.MTE3)
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
            tla.cross_core_wait_flag(scores_ready, tla.arch.MTE1)
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
            tla.set_flag(cast_free)
            tla.wait_flag(cast_free)
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
                tla.set_flag(cast_free)
                tla.wait_flag(cast_free)
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
                tla.set_flag(v_done)
                tla.wait_flag(v_done)
                tla.copy(
                    tla.tile_view(
                        recurrent,
                        tla.make_shape(256, 64),
                        tla.make_coord(0, aiv_recurrent),
                    ),
                    padded_nz(recurrent_nz.ptr, 256),
                )
                if parity == 0:
                    tla.cross_core_set_flag(k_ready, tla.arch.MTE3)
                else:
                    tla.cross_core_set_flag(v_ready, tla.arch.MTE3)
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
            tla.set_flag(input_dma_free)
            tla.wait_flag(input_dma_free)
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
            tla.set_flag(cube_done)
            tla.wait_flag(cube_done)
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
            tla.cross_core_set_flag(scores_done, tla.arch.FIX)
            tla.cross_core_set_flag(mq_done, tla.arch.FIX)
        tla.cross_core_wait_flag(inverse_ready, tla.arch.MTE1)
        if first_chunk:
            tla.wait_flag(identity_ready)
        tla.set_flag(free0)
        tla.set_flag(free1)
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
        tla.wait_flag(free0)
        tla.wait_flag(free1)
        if parity == 0:
            tla.cross_core_wait_flag(k_ready, tla.arch.MTE1)
        else:
            tla.cross_core_wait_flag(v_ready, tla.arch.MTE1)
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
        tla.cross_core_wait_flag(scores_done, tla.arch.VECTOR)
        # Snapshot NZ and leaf workspace alias; close its MTE3 reader first.
        tla.set_flag(cast_free)
        tla.wait_flag(cast_free)
        persistent_negative_t(kk, ub, nt32, ntnz, aiv_leaf)
        persistent_diagonal_inverse(
            nt32, prefix, padded_nz(dnz.ptr, 32), aiv_leaf, first_chunk
        )
        # QK must finish before inverse reuses L0C or next score replaces L1.
        tla.cross_core_wait_flag(mq_done, tla.arch.VECTOR)
    if tla.const_expr(prefetch and mode != 1):
        prefetch_half(q, k, raw_q, raw_k, next_valid, q_stride, 0, prefetch_free)
        prefetch_half(q, k, raw_q, raw_k, next_valid, q_stride, 1, prefetch_free)
    with tla.vector():
        aiv_inverse = tla.arch.sub_block_idx()
        tla.set_flag(v_done)
        tla.wait_flag(v_done)
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
        tla.cross_core_set_flag(inverse_ready, tla.arch.MTE3)
    if tla.const_expr(bulk_operands):
        with tla.vector():
            aiv_aq = tla.arch.sub_block_idx()
            mask_mqk_nz(mq, public, aq_nz, aiv_aq, save_matrices)
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
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
            tla.cross_core_set_flag(recur_ready, tla.arch.MTE3)
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
                tla.set_flag(v_done)
                tla.wait_flag(v_done)
                tla.copy(
                    tla.tile_view(
                        score_l1(lp, shared_scores),
                        tla.make_shape(score_rows, 64),
                        tla.make_coord(0, aiv_next),
                    ),
                    padded_nz(packed.ptr, score_rows),
                )
                tla.cross_core_set_flag(scores_ready, tla.arch.MTE3)
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
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            tla.copy(
                tla.tile_view(kl, tla.make_shape(64, 64), tla.make_coord(0, aiv_tail)),
                padded_nz(nz.ptr, 64),
            )
            tla.cross_core_set_flag(k_ready, tla.arch.MTE3)
            tla.set_flag(cast_free)
            tla.wait_flag(cast_free)
            beta_weighted_operand(uv, ub, ug, cast, False)
            nd2nz_tile(cast, nz, 64, 64)
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            tla.copy(
                tla.tile_view(vl, tla.make_shape(64, 64), tla.make_coord(0, aiv_tail)),
                padded_nz(nz.ptr, 64),
            )
            tla.cross_core_set_flag(v_ready, tla.arch.MTE3)
            tla.set_flag(cast_free)
            tla.wait_flag(cast_free)
            prepare_decayed_q(uq, ug, cast, decayed, scale)
            nd2nz_tile(decayed, nz, 64, 64)
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            if tla.const_expr(save_cache):
                tla.copy(
                    strided_rows(qg.ptr + aiv_tail * 64, valid, 64, v_stride),
                    row_major(cast.ptr, valid, 64),
                )
            tla.copy(
                tla.tile_view(ql, tla.make_shape(64, 64), tla.make_coord(0, aiv_tail)),
                padded_nz(nz.ptr, 64),
            )
            tla.set_flag(cast_free)
            tla.wait_flag(cast_free)
            restored_state_inputs(uk, ug, cast, ugamma)
            nd2nz_tile(cast, nz, 64, 64)
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            if tla.const_expr(save_cache):
                tla.copy(
                    strided_rows(kg.ptr + aiv_tail * 64, valid, 64, v_stride),
                    row_major(cast.ptr, valid, 64),
                )
            tla.copy(
                tla.tile_view(kgl, tla.make_shape(64, 64), tla.make_coord(0, aiv_tail)),
                padded_nz(nz.ptr, 64),
            )
            tla.set_flag(cast_free)
            tla.wait_flag(cast_free)
        if tla.const_expr(not bulk_operands):
            mask_mqk_nz(mq, public, aq_nz, aiv_tail, save_matrices)
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
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
            tla.cross_core_set_flag(recur_ready, tla.arch.MTE3)
    if next_valid > 0:
        next_recurrent = recurrent_l1(lp + 65536 + (1 - parity) * 32768)
        next_gamma = flat(ubase + 43616 + (1 - parity) * 64, 64)
        with tla.vector():
            aiv_pack = tla.arch.sub_block_idx()
            tla.set_flag(cast_free)
            tla.wait_flag(cast_free)
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
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            tla.copy(
                tla.tile_view(
                    next_recurrent, tla.make_shape(256, 64), tla.make_coord(0, aiv_pack)
                ),
                padded_nz(recurrent_nz.ptr, 256),
            )
            if parity == 0:
                tla.cross_core_set_flag(v_ready, tla.arch.MTE3)
            else:
                tla.cross_core_set_flag(k_ready, tla.arch.MTE3)
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
        tla.cross_core_wait_flag(delta_ready, tla.arch.VECTOR)
        if next_valid > 0:
            aiv_update = tla.arch.sub_block_idx()
            # cast_free before next recurrent packing has released score's
            # MTE3 read of the SNZ alias; the recurrent buffer is disjoint.
            update_state_nz(state, delta, ugamma, snz, 128)
            tla.set_flag(v_done)
            tla.wait_flag(v_done)
            tla.copy(
                tla.tile_view(
                    hl, tla.make_shape(128, 64), tla.make_coord(0, aiv_update)
                ),
                padded_nz(snz.ptr, 128),
            )
            tla.cross_core_set_flag(h_ready, tla.arch.MTE3)
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
        ub = tla.allocate(61440, tla.Float32, tla.AddressSpace.ub, 256)
        l1 = tla.allocate(
            172032 if self.pipeline else 131072, tla.BFloat16, tla.AddressSpace.l1, 512
        )
        a = tla.allocate(32768, tla.BFloat16, tla.AddressSpace.l0a, 512)
        b = tla.allocate(32768, tla.BFloat16, tla.AddressSpace.l0b, 512)
        c = tla.allocate(32768, tla.Float32, tla.AddressSpace.l0c, 512)
        load_state = tla.flag("ps_load_state", tla.arch.MTE2, tla.arch.VECTOR)
        save_state = tla.flag("ps_save_state", tla.arch.VECTOR, tla.arch.MTE3)
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
                tla.set_flag(load_state)
                tla.wait_flag(load_state)
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
                tla.set_flag(save_state)
                tla.wait_flag(save_state)
                tla.copy(
                    strided_rows(final.ptr + si + i64(aiv * 64), 128, 64, 128),
                    row_major(ub, 128, 64),
                )
                tla.pipe_barrier(tla.pipes.ALL)
            with tla.cube():
                tla.pipe_barrier(tla.pipes.ALL)
            task = meta[task + i64(5)]
