# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Deferred normalization for K @ Q.T tiles split along the Query axis.

Score slots use ``UB_SCORE_STRIDE`` FP32 elements per AIV row. Independent staging
slots carry per-Key-block sums and their tile maxima to and from GM. Their mutexes
cover both arrays because they always have the same owner and lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass

import catlass.tla as tla
from catlass.base_dsl.typing import Pointer
from catlass.types import TlaMutex

from .constants import BLOCK_TOKENS, align_up
from .stage_qk import M1, UB_SCORE_STRIDE, WORKSPACE_ROW_STRIDE


NEG_INF = float("-inf")
NAN = float("nan")
# Static UB allocation budget per AIV. Alignment is in bytes, not tile elements.
UB_CAPACITY_BYTES = 248 * 1024
UB_ALIGNMENT_BYTES = 256
# Each counts load is one 64-lane Int32 SIMD vector.
COUNTS_LOAD_ELEMS = 64
# Fixed power-of-two block indexing used by the SIMD masks.
BLOCK_TOKEN_SHIFT = BLOCK_TOKENS.bit_length() - 1
BLOCK_TOKEN_MASK = BLOCK_TOKENS - 1
FP32_VECTOR_LANES = 64
QUERY_VECTORS_PER_AIV = UB_SCORE_STRIDE // FP32_VECTOR_LANES
QUERY_BLOCKS_PER_VECTOR = FP32_VECTOR_LANES // BLOCK_TOKENS
QUERY_BLOCKS_PER_AIV = UB_SCORE_STRIDE // BLOCK_TOKENS
# One physical predicate register is 32 bytes, or eight INT32 words.
PREDICATE_WORDS = 8
QUERY_MASK_WORDS = QUERY_VECTORS_PER_AIV * PREDICATE_WORDS
# Pad BF16 output rows to 32 bytes. This alignment is independent of BLOCK_TOKENS.
OUTPUT_ROW_ALIGNMENT_ELEMS = 16
OUTPUT_ROW_STRIDE = align_up(M1 // BLOCK_TOKENS, OUTPUT_ROW_ALIGNMENT_ELEMS)
OUTPUT_ROWS_PER_VECTOR = FP32_VECTOR_LANES // OUTPUT_ROW_STRIDE
OUTPUT_ROW_SHIFT = OUTPUT_ROW_STRIDE.bit_length() - 1
OUTPUT_KEY_MASK = OUTPUT_ROW_STRIDE - 1
if OUTPUT_ROW_STRIDE != 1 << OUTPUT_ROW_SHIFT:
    raise ValueError("OUTPUT_ROW_STRIDE must be a power of two")
if QUERY_BLOCKS_PER_AIV != 2 * OUTPUT_ROWS_PER_VECTOR:
    raise ValueError("Query blocks per AIV must match the two packed output halves")
# Resource identifiers, independent of token geometry and mask sizes.
WORKSPACE_MUTEX_ID = 15
STATISTICS_MUTEX_ID = 16


def vector_fixed_bytes(key_rows: int = M1) -> int:
    """Explicit UB allocation excluding cached or streamed counts."""
    return (
        2 * key_rows * UB_SCORE_STRIDE * 4
        + 2 * (key_rows // BLOCK_TOKENS) * UB_SCORE_STRIDE * 4
        + 2 * UB_SCORE_STRIDE * 4
        + 2 * UB_SCORE_STRIDE * 4
        + 2 * QUERY_BLOCKS_PER_AIV * OUTPUT_ROW_STRIDE * 2
        + UB_ALIGNMENT_BYTES
    )


# A fixed counts allocation allows all S to reuse one artifact. The streaming
# path uses its first and last SIMD vectors for Key and Query respectively.
COUNTS_CACHE_ELEMS = (
    (UB_CAPACITY_BYTES - vector_fixed_bytes()) // UB_ALIGNMENT_BYTES
) * (UB_ALIGNMENT_BYTES // 4)


@dataclass
class VectorBuffers:
    score0: Pointer
    score1: Pointer
    qcounts: Pointer
    kcounts: Pointer
    stage_a0: Pointer
    stage_a1: Pointer
    stage_m0: Pointer
    stage_m1: Pointer
    run_max: Pointer
    run_sum: Pointer
    predicates: Pointer
    out0: Pointer
    out1: Pointer
    mq: TlaMutex
    mk: TlaMutex
    mo0: TlaMutex
    mo1: TlaMutex
    ms0: TlaMutex
    ms1: TlaMutex
    mw: TlaMutex
    mr: TlaMutex


@tla.jit
def allocate_vector(
    _cache_elems: tla.Constexpr[int],
    _m1: tla.Constexpr[int],
):
    qcounts = tla.allocate(
        _cache_elems, tla.Int32, tla.AddressSpace.ub, UB_ALIGNMENT_BYTES
    )
    # Scalar loads inside a VF require the root allocation as their Key base.
    kcounts = qcounts
    query_mutex = tla.mutex(resource="block_mean_attention_score_q_counts", id=9)
    return VectorBuffers(
        tla.allocate(
            _m1 * UB_SCORE_STRIDE, tla.Float32, tla.AddressSpace.ub, UB_ALIGNMENT_BYTES
        ),
        tla.allocate(
            _m1 * UB_SCORE_STRIDE, tla.Float32, tla.AddressSpace.ub, UB_ALIGNMENT_BYTES
        ),
        qcounts,
        kcounts,
        tla.allocate(
            (_m1 // BLOCK_TOKENS) * UB_SCORE_STRIDE,
            tla.Float32,
            tla.AddressSpace.ub,
            UB_ALIGNMENT_BYTES,
        ),
        tla.allocate(
            (_m1 // BLOCK_TOKENS) * UB_SCORE_STRIDE,
            tla.Float32,
            tla.AddressSpace.ub,
            UB_ALIGNMENT_BYTES,
        ),
        tla.allocate(
            UB_SCORE_STRIDE, tla.Float32, tla.AddressSpace.ub, UB_ALIGNMENT_BYTES
        ),
        tla.allocate(
            UB_SCORE_STRIDE, tla.Float32, tla.AddressSpace.ub, UB_ALIGNMENT_BYTES
        ),
        tla.allocate(
            UB_SCORE_STRIDE, tla.Float32, tla.AddressSpace.ub, UB_ALIGNMENT_BYTES
        ),
        tla.allocate(
            UB_SCORE_STRIDE, tla.Float32, tla.AddressSpace.ub, UB_ALIGNMENT_BYTES
        ),
        tla.allocate(
            UB_ALIGNMENT_BYTES // 4, tla.Int32, tla.AddressSpace.ub, UB_ALIGNMENT_BYTES
        ),
        tla.allocate(
            QUERY_BLOCKS_PER_AIV * OUTPUT_ROW_STRIDE,
            tla.BFloat16,
            tla.AddressSpace.ub,
            UB_ALIGNMENT_BYTES,
        ),
        tla.allocate(
            QUERY_BLOCKS_PER_AIV * OUTPUT_ROW_STRIDE,
            tla.BFloat16,
            tla.AddressSpace.ub,
            UB_ALIGNMENT_BYTES,
        ),
        query_mutex,
        tla.mutex(resource="block_mean_attention_score_k_counts", id=10),
        tla.mutex(resource="block_mean_attention_score_output_0", id=11),
        tla.mutex(resource="block_mean_attention_score_output_1", id=12),
        tla.mutex(resource="block_mean_attention_score_staging_0", id=13),
        tla.mutex(resource="block_mean_attention_score_staging_1", id=14),
        tla.mutex(
            resource="block_mean_attention_score_workspace", id=WORKSPACE_MUTEX_ID
        ),
        tla.mutex(
            resource="block_mean_attention_score_statistics", id=STATISTICS_MUTEX_ID
        ),
    )


@tla.jit
def _vector(ptr, size):
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(size), tla.make_stride(1))
    )


@tla.jit
def _matrix(ptr, rows, cols, stride):
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(rows, cols), tla.make_stride(stride, 1))
    )


@tla.jit
def _load_counts(counts, ptr, mutex, start, count):
    mutex.lock(pipe=tla.arch.MTE2)
    tla.copy(_vector(ptr, count), _vector(counts.ptr + start, count))
    mutex.unlock(pipe=tla.arch.MTE2)
    mutex.lock(pipe=tla.arch.VECTOR)
    full = _vector(ptr, COUNTS_LOAD_ELEMS)
    with tla.vec.func(mode="simd"):
        keep = tla.cmp(tla.arange(0, dtype=tla.Int32), count, "lt")
        full.store(tla.where(keep, full.load(), tla.full(0, dtype=tla.Int32)))
    mutex.unlock(pipe=tla.arch.VECTOR)


@tla.jit
def initialize_counts(
    counts,
    bufs,
    _blocks,
    _cache_elems,
    _cached,
):
    mutex = bufs.mq
    qcounts = bufs.qcounts
    if _cached:
        mutex.lock(pipe=tla.arch.MTE2)
        tla.copy(_vector(qcounts, _blocks), _vector(counts.ptr, _blocks))
        mutex.unlock(pipe=tla.arch.MTE2)
        mutex.lock(pipe=tla.arch.VECTOR)
        cache = _vector(qcounts, _cache_elems)
        with tla.vec.func(mode="simd"):
            lanes = tla.arange(0, dtype=tla.Int32)
            zero = tla.full(0, dtype=tla.Int32)
            # DMA initialized all valid counts. Only the final partial vector
            # and guard vectors need masking; leave the full prefix untouched.
            for group in tla.range(
                _blocks // COUNTS_LOAD_ELEMS, _cache_elems // COUNTS_LOAD_ELEMS
            ):
                part = tla.tile_view(
                    cache, tla.make_shape(COUNTS_LOAD_ELEMS), tla.make_coord(group)
                )
                keep = tla.cmp(lanes + group * COUNTS_LOAD_ELEMS, _blocks, "lt")
                part.store(tla.where(keep, part.load(), zero))
        mutex.unlock(pipe=tla.arch.VECTOR)
        # Later scalar tile-count sums must observe the initialized cache.
        mutex.lock(pipe=tla.arch.SCALAR)
        mutex.unlock(pipe=tla.arch.SCALAR)


@tla.jit
def _query_counts(bufs, query_start, _cached):
    offset = (
        query_start // BLOCK_TOKENS
        if _cached
        else COUNTS_CACHE_ELEMS - COUNTS_LOAD_ELEMS
    )
    return _vector(bufs.qcounts + offset, COUNTS_LOAD_ELEMS)


@tla.jit
def _key_counts(bufs, key_start, _cached):
    offset = key_start // BLOCK_TOKENS if _cached else 0
    return _vector(bufs.kcounts + offset, COUNTS_LOAD_ELEMS)


@tla.jit
def prepare_vector(counts, bufs, query_start, nq, nq_pad, _cached):
    if not _cached:
        _load_counts(
            counts,
            bufs.qcounts + (COUNTS_CACHE_ELEMS - COUNTS_LOAD_ELEMS),
            bufs.mq,
            query_start // BLOCK_TOKENS,
            nq // BLOCK_TOKENS,
        )
    qcounts = _query_counts(bufs, query_start, _cached)
    run_max = _vector(bufs.run_max, UB_SCORE_STRIDE)
    run_sum = _vector(bufs.run_sum, UB_SCORE_STRIDE)
    masks = _vector(bufs.predicates, QUERY_MASK_WORDS)
    half = nq_pad // 2
    qoffset = tla.arch.sub_block_idx() * half
    bufs.mq.lock(pipe=tla.arch.VECTOR)
    bufs.mr.lock(pipe=tla.arch.VECTOR)
    with tla.vec.func(mode="simd"):
        lanes = tla.arange(0, dtype=tla.Int32)
        zero = tla.full(0.0, dtype=tla.Float32)
        block = tla.shift_right(lanes, BLOCK_TOKEN_SHIFT)
        token = tla.bitwise_and(lanes, tla.full(BLOCK_TOKEN_MASK, dtype=tla.Int32))
        for part in tla.range(QUERY_VECTORS_PER_AIV):
            local_lane = lanes + part * FP32_VECTOR_LANES
            valid = tla.cmp(
                token,
                tla.gather(
                    qcounts,
                    block + part * QUERY_BLOCKS_PER_VECTOR + qoffset // BLOCK_TOKENS,
                ),
                "lt",
            )
            valid = tla.bitwise_and(valid, tla.cmp(local_lane, half, "lt"))
            valid = tla.bitwise_and(valid, tla.cmp(local_lane + qoffset, nq, "lt"))
            # A physical predicate register occupies 32 B, or eight INT32s.
            tla.tile_view(
                masks, tla.make_shape(PREDICATE_WORDS), tla.make_coord(part)
            ).store(valid, tla.params.MaskStoreParams())
            tla.tile_view(
                run_max, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(part)
            ).store(zero)
            tla.tile_view(
                run_sum, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(part)
            ).store(zero)
    bufs.mr.unlock(pipe=tla.arch.VECTOR)
    bufs.mq.unlock(pipe=tla.arch.VECTOR)


@tla.jit
def _load_score_row(scores, row, part):
    # The fixed score pitch permits a statically sized row view. A tile_view
    # of a dynamic matrix can otherwise infer a dynamic VectorSSA width.
    values = tla.make_tensor(
        scores.ptr,
        tla.make_layout(tla.make_shape(FP32_VECTOR_LANES), tla.make_stride(1)),
        tla.make_coord(row * UB_SCORE_STRIDE + part * FP32_VECTOR_LANES),
    )
    return values.load()


@tla.jit
def _store_score_row(scores, row, part, value):
    # row is bounded by mk // BLOCK_TOKENS; both 64-lane halves are allocated.
    destination = tla.make_tensor(
        scores.ptr,
        tla.make_layout(tla.make_shape(FP32_VECTOR_LANES), tla.make_stride(1)),
        tla.make_coord(row * UB_SCORE_STRIDE + part * FP32_VECTOR_LANES),
    )
    destination.store(value)


@tla.jit
def _exact_tile_max(scores, scalar_counts, key_offset, mk, scale):
    """Scan exactly the valid Key rows with one chain per 64-lane half."""
    neg_inf = tla.full(NEG_INF, dtype=tla.Float32)
    max0 = neg_inf
    max1 = neg_inf
    for max_kb in tla.range(mk // BLOCK_TOKENS):
        max_count = scalar_counts[key_offset + max_kb]
        for max_kr in tla.range(max_count):
            max_row = max_kb * BLOCK_TOKENS + max_kr
            max0 = tla.max(max0, _load_score_row(scores, max_row, 0))
            max1 = tla.max(max1, _load_score_row(scores, max_row, 1))
    # Preserve the existing final -Inf comparisons for nonfinite inputs.
    tile_max0 = tla.max(tla.max(max0, neg_inf), tla.max(neg_inf, neg_inf)) * scale
    tile_max1 = tla.max(tla.max(max1, neg_inf), tla.max(neg_inf, neg_inf)) * scale
    return tile_max0, tile_max1


@tla.jit
def _accumulate_tile(
    scores, scalar_counts, key_offset, mk, scale, stage_a, tile_max0, tile_max1
):
    """Use one directly initialized chain per half for count-bounded sums."""
    zero = tla.full(0.0, dtype=tla.Float32)
    tile_sum0 = zero
    tile_sum1 = zero
    for sum_kb in tla.range(mk // BLOCK_TOKENS):
        sum0 = zero
        sum1 = zero
        sum_count = scalar_counts[key_offset + sum_kb]
        if sum_count > 0:
            sum_row = sum_kb * BLOCK_TOKENS
            sum0 = tla.exp(_load_score_row(scores, sum_row, 0) * scale - tile_max0)
            sum1 = tla.exp(_load_score_row(scores, sum_row, 1) * scale - tile_max1)
            for sum_kr in tla.range(1, sum_count):
                sum_row = sum_kb * BLOCK_TOKENS + sum_kr
                sum0 = sum0 + tla.exp(
                    _load_score_row(scores, sum_row, 0) * scale - tile_max0
                )
                sum1 = sum1 + tla.exp(
                    _load_score_row(scores, sum_row, 1) * scale - tile_max1
                )
        _store_score_row(stage_a, sum_kb, 0, sum0)
        _store_score_row(stage_a, sum_kb, 1, sum1)
        tile_sum0 = tile_sum0 + sum0
        tile_sum1 = tile_sum1 + sum1
    return tile_sum0, tile_sum1


@tla.jit
def _merge_statistics_ssa(max_view, sum_view, current_max, current_sum):
    """Merge the live tile values without a UB store/reload round trip."""
    old_max = max_view.load()
    old_sum = sum_view.load()
    have_old = tla.cmp(old_sum, 0.0, "gt")
    new_max = tla.where(have_old, tla.max(old_max, current_max), current_max)
    # A nonempty tile can contain only -Inf scores. Its exact maximum stays
    # -Inf and its sum is zero; avoid forming -Inf - -Inf while merging it.
    # NaN sums deliberately remain sticky across all subsequent Key tiles.
    reference = tla.where(
        tla.cmp(new_max, NEG_INF, "gt"), new_max, tla.full(0.0, dtype=tla.Float32)
    )
    safe_old = tla.where(have_old, old_max, reference)
    safe_current = tla.where(tla.cmp(current_sum, 0.0, "gt"), current_max, reference)
    new_sum = old_sum * tla.exp(safe_old - reference) + current_sum * tla.exp(
        safe_current - reference
    )
    max_view.store(new_max)
    sum_view.store(new_sum)


@tla.jit
def _phase_one_exact(
    counts,
    ws,
    bufs,
    score_ptr,
    key_start,
    mk,
    scale,
    key_tile,
    serial,
    blocks,
    key_tiles,
    _total,
    _cached,
    _m1: tla.Constexpr[int],
):
    """Compute an exact max for every nonempty Key tile before accumulation."""
    # Scalar loads within the VF must use a fixed allocation base; an offset
    # pointer would have to be materialized inside the outlined Vector function.
    scalar_counts = _vector(bufs.kcounts, COUNTS_CACHE_ELEMS)
    key_offset = key_start // BLOCK_TOKENS if _cached else 0
    stage_a_ptr = bufs.stage_a0 if serial % 2 == 0 else bufs.stage_a1
    stage_m_ptr = bufs.stage_m0 if serial % 2 == 0 else bufs.stage_m1
    staging_mutex = bufs.ms0 if serial % 2 == 0 else bufs.ms1
    scores = _matrix(score_ptr, _m1, UB_SCORE_STRIDE, UB_SCORE_STRIDE)
    stage_a = _matrix(
        stage_a_ptr, _m1 // BLOCK_TOKENS, UB_SCORE_STRIDE, UB_SCORE_STRIDE
    )
    stage_m = _vector(stage_m_ptr, UB_SCORE_STRIDE)
    run_max = _vector(bufs.run_max, UB_SCORE_STRIDE)
    run_sum = _vector(bufs.run_sum, UB_SCORE_STRIDE)
    bufs.mk.lock(pipe=tla.arch.VECTOR)
    staging_mutex.lock(pipe=tla.arch.VECTOR)
    bufs.mr.lock(pipe=tla.arch.VECTOR)
    with tla.vec.func(mode="simd"):
        zero = tla.full(0.0, dtype=tla.Float32)
        neg_inf = tla.full(NEG_INF, dtype=tla.Float32)
        total = _total
        if total > 0:
            tile_max0, tile_max1 = _exact_tile_max(
                scores, scalar_counts, key_offset, mk, scale
            )
            # A tile containing only -Inf contributes zero when another
            # tile has finite scores. Keep its exact -Inf workspace max,
            # but avoid exp(-Inf - -Inf) while forming the partial sums.
            sum_reference0 = tla.where(
                tla.cmp(tile_max0, NEG_INF, "eq"), zero, tile_max0
            )
            sum_reference1 = tla.where(
                tla.cmp(tile_max1, NEG_INF, "eq"), zero, tile_max1
            )
            tile_sum0, tile_sum1 = _accumulate_tile(
                scores,
                scalar_counts,
                key_offset,
                mk,
                scale,
                stage_a,
                sum_reference0,
                sum_reference1,
            )
            tla.tile_view(
                stage_m, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(0)
            ).store(tile_max0)
            tla.tile_view(
                stage_m, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(1)
            ).store(tile_max1)
            _merge_statistics_ssa(
                tla.tile_view(
                    run_max, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(0)
                ),
                tla.tile_view(
                    run_sum, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(0)
                ),
                tile_max0,
                tile_sum0,
            )
            _merge_statistics_ssa(
                tla.tile_view(
                    run_max, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(1)
                ),
                tla.tile_view(
                    run_sum, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(1)
                ),
                tile_max1,
                tile_sum1,
            )
        else:
            for empty_kb in tla.range(mk // BLOCK_TOKENS):
                _store_score_row(stage_a, empty_kb, 0, zero)
                _store_score_row(stage_a, empty_kb, 1, zero)
            for empty_part in tla.range(QUERY_VECTORS_PER_AIV):
                tla.tile_view(
                    stage_m,
                    tla.make_shape(FP32_VECTOR_LANES),
                    tla.make_coord(empty_part),
                ).store(neg_inf)
    bufs.mr.unlock(pipe=tla.arch.VECTOR)
    staging_mutex.unlock(pipe=tla.arch.VECTOR)
    bufs.mk.unlock(pipe=tla.arch.VECTOR)
    # Workspace segments use UB_SCORE_STRIDE lanes, including short Query
    # tiles whose logical AIV offset is only nq_pad/2.
    ws_base = (
        tla.arch.block_idx() * (blocks + key_tiles) * WORKSPACE_ROW_STRIDE
        + tla.arch.sub_block_idx() * UB_SCORE_STRIDE
    )
    gm_a = _matrix(
        ws.ptr + ws_base + (key_start // BLOCK_TOKENS) * WORKSPACE_ROW_STRIDE,
        mk // BLOCK_TOKENS,
        UB_SCORE_STRIDE,
        WORKSPACE_ROW_STRIDE,
    )
    gm_m = _vector(
        ws.ptr + ws_base + (blocks + key_tile) * WORKSPACE_ROW_STRIDE, UB_SCORE_STRIDE
    )
    staging_mutex.lock(pipe=tla.arch.MTE3)
    tla.copy(
        gm_a, _matrix(stage_a_ptr, mk // BLOCK_TOKENS, UB_SCORE_STRIDE, UB_SCORE_STRIDE)
    )
    tla.copy(gm_m, stage_m)
    staging_mutex.unlock(pipe=tla.arch.MTE3)


@tla.jit
def phase_one(
    counts,
    ws,
    bufs,
    score_ptr,
    key_start,
    mk,
    scale,
    key_tile,
    serial,
    blocks,
    key_tiles,
    _cached,
    _m1: tla.Constexpr[int],
):
    """Load Key counts and run the exact count-bounded tile computation."""
    key_mutex = bufs.mk
    if not _cached:
        _load_counts(
            counts, bufs.kcounts, bufs.mk, key_start // BLOCK_TOKENS, mk // BLOCK_TOKENS
        )
        key_mutex.lock(pipe=tla.arch.SCALAR)
    scalar_counts = _vector(bufs.kcounts, COUNTS_CACHE_ELEMS)
    key_offset = key_start // BLOCK_TOKENS if _cached else 0
    total = tla.as_numeric(0)
    for total_kb in tla.range(mk // BLOCK_TOKENS):
        total = total + scalar_counts[key_offset + total_kb]
    if not _cached:
        key_mutex.unlock(pipe=tla.arch.SCALAR)
    _phase_one_exact(
        counts,
        ws,
        bufs,
        score_ptr,
        key_start,
        mk,
        scale,
        key_tile,
        serial,
        blocks,
        key_tiles,
        total,
        _cached,
        _m1,
    )


@tla.jit
def finish_statistics(bufs, query_start, nq_pad, _cached):
    """Fold Query-block averaging into the final normalization inverse."""
    sums = _vector(bufs.run_sum, UB_SCORE_STRIDE)
    qcounts = _query_counts(bufs, query_start, _cached)
    masks = _vector(bufs.predicates, QUERY_MASK_WORDS)
    qoffset = tla.arch.sub_block_idx() * (nq_pad // 2)
    # Counts are constant across all Key tiles consumed by this task.
    bufs.mr.lock(pipe=tla.arch.VECTOR)
    with tla.vec.func(mode="simd"):
        zero = tla.full(0.0, dtype=tla.Float32)
        one = tla.full(1.0, dtype=tla.Float32)
        nan = tla.full(NAN, dtype=tla.Float32)
        for part in tla.range(QUERY_VECTORS_PER_AIV):
            view = tla.tile_view(
                sums, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(part)
            )
            value = view.load()
            valid = tla.cmp(value, 0.0, "gt")
            safe_value = tla.where(valid, value, one)
            query_valid = tla.tile_view(
                masks, tla.make_shape(PREDICATE_WORDS), tla.make_coord(part)
            ).load(tla.params.MaskLoadParams())
            query_blocks = tla.shift_right(
                tla.arange(0, dtype=tla.Int32) + part * FP32_VECTOR_LANES + qoffset,
                BLOCK_TOKEN_SHIFT,
            )
            divisor = tla.gather(qcounts, query_blocks).to(
                tla.Float32,
                tla.params.CastParams(round_mode=tla.params.RoundMode.CAST_ROUND),
            )
            safe_divisor = tla.where(tla.cmp(divisor, 0.0, "gt"), divisor, one)
            reciprocal = tla.where(valid, one / (safe_value * safe_divisor), nan)
            reciprocal = tla.where(query_valid, reciprocal, zero)
            # A negative sentinel marks every lane of an empty Query block.
            # Finalize propagates its required NaN without scalar QB branches.
            view.store(
                tla.where(
                    tla.cmp(divisor, 0.0, "gt"),
                    reciprocal,
                    tla.full(-1.0, dtype=tla.Float32),
                )
            )
    bufs.mr.unlock(pipe=tla.arch.VECTOR)


@tla.jit
def _load_workspace_tile(ws, bufs, tile, blocks, key_tiles, _m1: tla.Constexpr[int]):
    key_start = tile * _m1
    mk = blocks * BLOCK_TOKENS - key_start
    mk = mk if mk < _m1 else _m1
    a_ptr = bufs.stage_a0 if tile % 2 == 0 else bufs.stage_a1
    m_ptr = bufs.stage_m0 if tile % 2 == 0 else bufs.stage_m1
    mutex = bufs.ms0 if tile % 2 == 0 else bufs.ms1
    base = (
        tla.arch.block_idx() * (blocks + key_tiles) * WORKSPACE_ROW_STRIDE
        + tla.arch.sub_block_idx() * UB_SCORE_STRIDE
    )
    gm_a = _matrix(
        ws.ptr + base + (key_start // BLOCK_TOKENS) * WORKSPACE_ROW_STRIDE,
        mk // BLOCK_TOKENS,
        UB_SCORE_STRIDE,
        WORKSPACE_ROW_STRIDE,
    )
    gm_m = _vector(
        ws.ptr + base + (blocks + tile) * WORKSPACE_ROW_STRIDE, UB_SCORE_STRIDE
    )
    mutex.lock(pipe=tla.arch.MTE2)
    tla.copy(_matrix(a_ptr, mk // BLOCK_TOKENS, UB_SCORE_STRIDE, UB_SCORE_STRIDE), gm_a)
    tla.copy(_vector(m_ptr, UB_SCORE_STRIDE), gm_m)
    mutex.unlock(pipe=tla.arch.MTE2)


@tla.jit
def _finalize_impl(
    counts,
    ws,
    out,
    bufs,
    head,
    query_start,
    nq,
    nq_pad,
    blocks,
    key_tiles,
    _cached,
    _m1: tla.Constexpr[int],
):
    """Normalize deferred sums and reduce groups of sixteen Query lanes."""
    # The task-wide resource transfers the GM workspace from phase-one
    # MTE3 writes to finalize's MTE2 reads.
    bufs.mw.lock(pipe=tla.arch.MTE2)
    _load_workspace_tile(ws, bufs, 0, blocks, key_tiles, _m1)
    half = nq_pad // 2
    qbase = tla.arch.sub_block_idx() * half // BLOCK_TOKENS
    local_blocks = nq // BLOCK_TOKENS - qbase
    local_blocks = local_blocks if local_blocks > 0 else 0
    local_blocks = (
        local_blocks if local_blocks < half // BLOCK_TOKENS else half // BLOCK_TOKENS
    )
    masks = _vector(bufs.predicates, QUERY_MASK_WORDS)
    final_max = _vector(bufs.run_max, UB_SCORE_STRIDE)
    inv_sum = _vector(bufs.run_sum, UB_SCORE_STRIDE)
    query_mutex = bufs.mq
    key_mutex = bufs.mk
    statistics_mutex = bufs.mr
    for tile in tla.range(key_tiles):
        # Read into the other staging slot while VECTOR consumes this one.
        if tile + 1 < key_tiles:
            _load_workspace_tile(ws, bufs, tile + 1, blocks, key_tiles, _m1)
        key_start = tile * _m1
        mk = blocks * BLOCK_TOKENS - key_start
        mk = mk if mk < _m1 else _m1
        if not _cached:
            _load_counts(
                counts,
                bufs.kcounts,
                key_mutex,
                key_start // BLOCK_TOKENS,
                mk // BLOCK_TOKENS,
            )
        kcounts = _key_counts(bufs, key_start, _cached)
        a_ptr = bufs.stage_a0 if tile % 2 == 0 else bufs.stage_a1
        m_ptr = bufs.stage_m0 if tile % 2 == 0 else bufs.stage_m1
        mutex = bufs.ms0 if tile % 2 == 0 else bufs.ms1
        output_ptr = bufs.out0 if tile % 2 == 0 else bufs.out1
        mo = bufs.mo0 if tile % 2 == 0 else bufs.mo1
        stage_a = _matrix(a_ptr, _m1 // BLOCK_TOKENS, UB_SCORE_STRIDE, UB_SCORE_STRIDE)
        stage_m = _vector(m_ptr, UB_SCORE_STRIDE)
        packed_output = _vector(output_ptr, QUERY_BLOCKS_PER_AIV * OUTPUT_ROW_STRIDE)
        mutex.lock(pipe=tla.arch.VECTOR)
        query_mutex.lock(pipe=tla.arch.VECTOR)
        key_mutex.lock(pipe=tla.arch.VECTOR)
        statistics_mutex.lock(pipe=tla.arch.VECTOR)
        mo.lock(pipe=tla.arch.VECTOR)
        with tla.vec.func(mode="simd"):
            lanes = tla.arange(0, dtype=tla.Int32)
            zero = tla.full(0.0, dtype=tla.Float32)
            nan = tla.full(NAN, dtype=tla.Float32)
            valid0 = tla.tile_view(
                masks, tla.make_shape(PREDICATE_WORDS), tla.make_coord(0)
            ).load(tla.params.MaskLoadParams())
            valid1 = tla.tile_view(
                masks, tla.make_shape(PREDICATE_WORDS), tla.make_coord(1)
            ).load(tla.params.MaskLoadParams())
            m0 = tla.tile_view(
                stage_m, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(0)
            ).load()
            m1 = tla.tile_view(
                stage_m, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(1)
            ).load()
            final0 = tla.tile_view(
                final_max, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(0)
            ).load()
            final1 = tla.tile_view(
                final_max, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(1)
            ).load()
            inv0 = tla.tile_view(
                inv_sum, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(0)
            ).load()
            inv1 = tla.tile_view(
                inv_sum, tla.make_shape(FP32_VECTOR_LANES), tla.make_coord(1)
            ).load()
            have0 = tla.bitwise_and(valid0, tla.cmp(inv0, 0.0, "gt"))
            have1 = tla.bitwise_and(valid1, tla.cmp(inv1, 0.0, "gt"))
            have0 = tla.bitwise_and(have0, tla.cmp(m0, NEG_INF, "gt"))
            have1 = tla.bitwise_and(have1, tla.cmp(m1, NEG_INF, "gt"))
            # Select both exponent operands before subtracting: where is not
            # short-circuiting, and invalid Query slots may contain NaN/Inf.
            weight0 = tla.exp(
                tla.where(have0, m0, zero) - tla.where(have0, final0, zero)
            )
            weight1 = tla.exp(
                tla.where(have1, m1, zero) - tla.where(have1, final1, zero)
            )
            weight0 = tla.where(have0, weight0 * inv0, zero)
            weight1 = tla.where(have1, weight1 * inv1, zero)
            # Empty Query blocks and bad valid logits use NaN weights. Mask A
            # before multiplying:
            # padding in a positive-count block must contribute finite zero.
            weight0 = tla.where(tla.cmp(inv0, 0.0, "ge"), weight0, nan)
            weight1 = tla.where(tla.cmp(inv1, 0.0, "ge"), weight1, nan)
            # Pack complete output rows directly in registers, keeping the
            # reduced per-Key values live until the final BF16 conversion.
            output0 = zero
            output1 = zero
            key_slots = tla.bitwise_and(
                lanes, tla.full(OUTPUT_KEY_MASK, dtype=tla.Int32)
            )
            query_groups = tla.shift_right(lanes, OUTPUT_ROW_SHIFT)
            # A Key tile advances by 14 counts (56 B); the cached view
            # therefore requires an unaligned load.
            key_divisors = kcounts.load(tla.params.UnalignLoadParams()).to(
                tla.Float32,
                tla.params.CastParams(round_mode=tla.params.RoundMode.CAST_ROUND),
            )
            positive_key = tla.cmp(key_divisors, 0.0, "gt")
            one = tla.full(1.0, dtype=tla.Float32)
            key_recip = tla.where(
                positive_key, one / tla.where(positive_key, key_divisors, one), nan
            )
            output_recip = tla.gather(key_recip, key_slots)
            for kb in tla.range(mk // BLOCK_TOKENS):
                value0 = _load_score_row(stage_a, kb, 0)
                value1 = _load_score_row(stage_a, kb, 1)
                # Padding must contribute zero before applying normalization.
                value0 = tla.where(valid0, value0, zero) * weight0
                value1 = tla.where(valid1, value1, zero) * weight1
                reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                block_sums0 = value0.reduce(
                    tla.ReductionOp.ADD,
                    mode=tla.ReduceMode.INTRA_BLOCK,
                    mask=reduce_mask,
                )
                block_sums1 = value1.reduce(
                    tla.ReductionOp.ADD,
                    mode=tla.ReduceMode.INTRA_BLOCK,
                    mask=reduce_mask,
                )
                lo, hi = tla.deinterleave(block_sums0, block_sums1)
                grouped = lo + hi
                selected0 = tla.gather(grouped, query_groups)
                # Packed results from the second register start at lane 32.
                selected1 = tla.gather(grouped, query_groups + FP32_VECTOR_LANES // 2)
                is_key_slot = tla.cmp(key_slots, kb, "eq")
                output0 = tla.where(is_key_slot, selected0, output0)
                output1 = tla.where(is_key_slot, selected1, output1)
            output0 = output0 * output_recip
            output1 = output1 * output_recip
            output_store_mask, _ = tla.update_mask(
                FP32_VECTOR_LANES, dtype=tla.BFloat16
            )
            narrowed0 = output0.to(
                tla.BFloat16,
                tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                ),
            )
            narrowed1 = output1.to(
                tla.BFloat16,
                tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                ),
            )
            packed0, _ = tla.deinterleave(narrowed0, narrowed0)
            packed1, _ = tla.deinterleave(narrowed1, narrowed1)
            tla.tile_view(
                packed_output,
                tla.make_shape(FP32_VECTOR_LANES),
                tla.make_coord(0),
            ).store(packed0, mask=output_store_mask)
            tla.tile_view(
                packed_output,
                tla.make_shape(FP32_VECTOR_LANES),
                tla.make_coord(1),
            ).store(packed1, mask=output_store_mask)
        mo.unlock(pipe=tla.arch.VECTOR)
        statistics_mutex.unlock(pipe=tla.arch.VECTOR)
        key_mutex.unlock(pipe=tla.arch.VECTOR)
        query_mutex.unlock(pipe=tla.arch.VECTOR)
        mutex.unlock(pipe=tla.arch.VECTOR)
        if local_blocks > 0:
            gm_out = _matrix(
                out.ptr
                + (head * blocks + query_start // BLOCK_TOKENS + qbase) * blocks
                + key_start // BLOCK_TOKENS,
                local_blocks,
                mk // BLOCK_TOKENS,
                blocks,
            )
            ub_out = _matrix(
                output_ptr, local_blocks, mk // BLOCK_TOKENS, OUTPUT_ROW_STRIDE
            )
            mo.lock(pipe=tla.arch.MTE3)
            tla.copy(gm_out, ub_out)
            mo.unlock(pipe=tla.arch.MTE3)
    bufs.mw.unlock(pipe=tla.arch.MTE2)


@tla.jit
def finalize(
    counts,
    ws,
    out,
    bufs,
    head,
    query_start,
    nq,
    nq_pad,
    blocks,
    key_tiles,
    _cached,
    _m1: tla.Constexpr[int],
):
    finish_statistics(bufs, query_start, nq_pad, _cached)
    _finalize_impl(
        counts,
        ws,
        out,
        bufs,
        head,
        query_start,
        nq,
        nq_pad,
        blocks,
        key_tiles,
        _cached,
        _m1,
    )
