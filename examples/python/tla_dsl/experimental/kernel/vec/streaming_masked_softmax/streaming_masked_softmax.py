# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Streaming row-wise softmax with optional additive mask/bias.

Short rows stay resident in UB. Long, low-parallel rows use one launch with
parallel chunk statistics, a device-wide synchronization point, compact row
statistics, and parallel normalization. UB ownership is expressed with mutexes.
"""

from __future__ import annotations

import math
import weakref

import catlass.tla as tla


@tla.kernel
def streaming_masked_softmax_kernel(
    gm_x: tla.Tensor,  # [M, N] fp32
    gm_add: tla.Tensor,  # [M, N] fp32; never read when HAS_ADD == 0
    gm_y: tla.Tensor,  # [M, N] fp32
    N: tla.Constexpr[int],
    CHUNK: tla.Constexpr[int],
    TAIL: tla.Constexpr[int],
    SCALE: tla.Constexpr[float],
    HAS_ADD: tla.Constexpr[int],
) -> None:
    vl_ele = 64
    min_float = -3.4028234663852886e38
    lanes = CHUNK // vl_ele
    n_chunks = N // CHUNK
    tail_blocks = (TAIL + vl_ele - 1) // vl_ele
    tail_storage = tail_blocks * vl_ele
    row_resident = TAIL == 0 and N <= CHUNK

    # One mutex owns one physical UB buffer. MTE2, VECTOR and MTE3 hand the
    # buffer between them without integer flag bookkeeping.
    mutex_x = tla.mutex(resource="streaming_masked_softmax_x", id=0)
    mutex_add = tla.mutex(resource="streaming_masked_softmax_add", id=1)

    # Long rows need only one slice buffer per operand. The x buffer is reused
    # for scaled logits and normalized output to keep the UB footprint small.
    x_ptr = tla.allocate(CHUNK, tla.Float32, tla.AddressSpace.ub, 256)
    add_ptr = tla.allocate(CHUNK, tla.Float32, tla.AddressSpace.ub, 256)
    max_ptr = tla.allocate(vl_ele, tla.Float32, tla.AddressSpace.ub, 256)
    sum_ptr = tla.allocate(vl_ele, tla.Float32, tla.AddressSpace.ub, 256)

    chunk_layout = tla.make_layout(
        tla.make_shape(1, CHUNK),
        tla.make_stride(CHUNK, 1),
        origin_shape=tla.make_shape(1, CHUNK),
    )
    vector_layout = tla.make_layout(
        tla.make_shape(vl_ele),
        tla.make_stride(1),
    )
    ub_x = tla.make_tensor(x_ptr, chunk_layout)
    ub_add = tla.make_tensor(add_ptr, chunk_layout)
    ub_max = tla.make_tensor(max_ptr, vector_layout)
    ub_sum = tla.make_tensor(sum_ptr, vector_layout)

    # Tail aliases the start of x/add. DMA copies only TAIL elements; the
    # vector view exposes whole 64-lane blocks and masks the last reduction.
    tail_extent = TAIL if TAIL else 1
    tail_storage_extent = tail_storage if tail_storage else vl_ele
    tail_dma_layout = tla.make_layout(
        tla.make_shape(1, tail_extent),
        tla.make_stride(1, 1),
        origin_shape=tla.make_shape(1, tail_extent),
    )
    tail_storage_layout = tla.make_layout(
        tla.make_shape(1, tail_storage_extent),
        tla.make_stride(tail_storage_extent, 1),
        origin_shape=tla.make_shape(1, tail_storage_extent),
    )
    tail_x_dma = tla.make_tensor(x_ptr, tail_dma_layout)
    tail_a_dma = tla.make_tensor(add_ptr, tail_dma_layout)
    tail_x_vec = tla.make_tensor(x_ptr, tail_storage_layout)
    tail_a_vec = tla.make_tensor(add_ptr, tail_storage_layout)

    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()

    with tla.vector():
        for row in tla.range(gid, gm_x.origin_shape[0], aiv_num):
            row_off = row.to(tla.Int64) * tla.Int64(N)
            gm_row_x = tla.make_tensor(gm_x.ptr + row_off, chunk_layout)
            gm_row_y = tla.make_tensor(gm_y.ptr + row_off, chunk_layout)

            # Short-row fast path: one GM read, then max, exp/sum and normalize
            # all reuse the resident UB row.
            if tla.const_expr(row_resident):
                mutex_x.lock(pipe=tla.arch.MTE2)
                tla.copy(ub_x, gm_row_x)
                mutex_x.unlock(pipe=tla.arch.MTE2)
                if tla.const_expr(HAS_ADD == 1):
                    gm_row_add = tla.make_tensor(gm_add.ptr + row_off, chunk_layout)
                    mutex_add.lock(pipe=tla.arch.MTE2)
                    tla.copy(ub_add, gm_row_add)
                    mutex_add.unlock(pipe=tla.arch.MTE2)

                mutex_x.lock(pipe=tla.arch.VECTOR)
                if tla.const_expr(HAS_ADD == 1):
                    mutex_add.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    running_max = tla.full(min_float, dtype=tla.Float32)

                    # Apply bias and scale before max so max and sum share one
                    # numerical domain.
                    for row_lane_1 in tla.range_constexpr(lanes):
                        row_xc_1 = tla.tile_view(
                            ub_x,
                            tla.make_shape(1, vl_ele),
                            tla.make_coord(0, row_lane_1),
                        )
                        row_value_1 = row_xc_1.load()
                        if tla.const_expr(HAS_ADD == 1):
                            row_ac_1 = tla.tile_view(
                                ub_add,
                                tla.make_shape(1, vl_ele),
                                tla.make_coord(0, row_lane_1),
                            )
                            row_value_1 = tla.add(
                                row_value_1, row_ac_1.load(), mask=full
                            )
                        row_value_1 = tla.mul(row_value_1, SCALE, mask=full)
                        row_xc_1.store(row_value_1)
                        running_max = tla.max(running_max, row_value_1, mask=full)

                    row_max = tla.full(
                        running_max.reduce(tla.ReductionOp.MAX, mask=full),
                        dtype=tla.Float32,
                    )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE,
                        tla.params.MemType.VEC_LOAD,
                    )

                    # Exponentials overwrite the scaled row in place. The sum
                    # stays in one vector accumulator and reduces only once.
                    acc = tla.full(0.0, dtype=tla.Float32)
                    for row_lane_2 in tla.range_constexpr(lanes):
                        row_xc_2 = tla.tile_view(
                            ub_x,
                            tla.make_shape(1, vl_ele),
                            tla.make_coord(0, row_lane_2),
                        )
                        row_value_2 = tla.exp(
                            tla.sub(row_xc_2.load(), row_max, mask=full),
                            mask=full,
                        )
                        row_xc_2.store(row_value_2)
                        acc = tla.add(acc, row_value_2, mask=full)
                    row_sum = tla.full(
                        acc.reduce(tla.ReductionOp.ADD, mask=full),
                        dtype=tla.Float32,
                    )
                    inv_sum = tla.div(
                        tla.full(1.0, dtype=tla.Float32),
                        row_sum,
                        mask=full,
                    )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE,
                        tla.params.MemType.VEC_LOAD,
                    )

                    # One reciprocal replaces N divisions by N multiplies.
                    for row_lane_3 in tla.range_constexpr(lanes):
                        row_xc_3 = tla.tile_view(
                            ub_x,
                            tla.make_shape(1, vl_ele),
                            tla.make_coord(0, row_lane_3),
                        )
                        row_xc_3.store(tla.mul(row_xc_3.load(), inv_sum, mask=full))

                if tla.const_expr(HAS_ADD == 1):
                    mutex_add.unlock(pipe=tla.arch.VECTOR)
                mutex_x.unlock(pipe=tla.arch.VECTOR)

                # MTE3 owns x until the output copy drains. The following row
                # can refill the buffer immediately after this unlock.
                mutex_x.lock(pipe=tla.arch.MTE3)
                tla.copy(gm_row_y, ub_x)
                mutex_x.unlock(pipe=tla.arch.MTE3)

            else:
                # Long-row pass 1: build online row max/sum over fixed slices.
                with tla.vec.func(mode="simd"):
                    ub_max.store(tla.full(min_float, dtype=tla.Float32))
                    ub_sum.store(tla.full(0.0, dtype=tla.Float32))

                for first_chunk in tla.range(0, n_chunks, 1):
                    first_gm_chunk_x = tla.make_tensor(
                        gm_x.ptr + row_off + first_chunk * CHUNK,
                        chunk_layout,
                    )
                    mutex_x.lock(pipe=tla.arch.MTE2)
                    tla.copy(ub_x, first_gm_chunk_x)
                    mutex_x.unlock(pipe=tla.arch.MTE2)
                    if tla.const_expr(HAS_ADD == 1):
                        first_gm_chunk_add = tla.make_tensor(
                            gm_add.ptr + row_off + first_chunk * CHUNK,
                            chunk_layout,
                        )
                        mutex_add.lock(pipe=tla.arch.MTE2)
                        tla.copy(ub_add, first_gm_chunk_add)
                        mutex_add.unlock(pipe=tla.arch.MTE2)

                    mutex_x.lock(pipe=tla.arch.VECTOR)
                    if tla.const_expr(HAS_ADD == 1):
                        mutex_add.lock(pipe=tla.arch.VECTOR)
                    with tla.vec.func(mode="simd"):
                        first_full = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        first_old_max = ub_max.load()
                        first_old_sum = ub_sum.load()
                        first_tile_max = tla.full(min_float, dtype=tla.Float32)

                        # Scale+bias in place once, then reuse the scaled slice
                        # for exp. This avoids re-reading the same UB chunk.
                        for first_lane in tla.range_constexpr(lanes):
                            first_xc = tla.tile_view(
                                ub_x,
                                tla.make_shape(1, vl_ele),
                                tla.make_coord(0, first_lane),
                            )
                            first_value = first_xc.load()
                            if tla.const_expr(HAS_ADD == 1):
                                first_ac = tla.tile_view(
                                    ub_add,
                                    tla.make_shape(1, vl_ele),
                                    tla.make_coord(0, first_lane),
                                )
                                first_value = tla.add(
                                    first_value,
                                    first_ac.load(),
                                    mask=first_full,
                                )
                            first_value = tla.mul(first_value, SCALE, mask=first_full)
                            first_xc.store(first_value)
                            first_tile_max = tla.max(
                                first_tile_max,
                                first_value,
                                mask=first_full,
                            )

                        first_new_max = tla.max(
                            first_old_max,
                            tla.full(
                                first_tile_max.reduce(
                                    tla.ReductionOp.MAX, mask=first_full
                                ),
                                dtype=tla.Float32,
                            ),
                            mask=first_full,
                        )
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE,
                            tla.params.MemType.VEC_LOAD,
                        )

                        first_tile_acc = tla.full(0.0, dtype=tla.Float32)
                        for first_lane in tla.range_constexpr(lanes):
                            first_xc = tla.tile_view(
                                ub_x,
                                tla.make_shape(1, vl_ele),
                                tla.make_coord(0, first_lane),
                            )
                            first_tile_acc = tla.add(
                                first_tile_acc,
                                tla.exp(
                                    tla.sub(
                                        first_xc.load(),
                                        first_new_max,
                                        mask=first_full,
                                    ),
                                    mask=first_full,
                                ),
                                mask=first_full,
                            )
                        first_new_sum = tla.add(
                            tla.mul(
                                first_old_sum,
                                tla.exp(
                                    tla.sub(
                                        first_old_max,
                                        first_new_max,
                                        mask=first_full,
                                    ),
                                    mask=first_full,
                                ),
                                mask=first_full,
                            ),
                            tla.full(
                                first_tile_acc.reduce(
                                    tla.ReductionOp.ADD, mask=first_full
                                ),
                                dtype=tla.Float32,
                            ),
                            mask=first_full,
                        )
                        ub_max.store(first_new_max)
                        ub_sum.store(first_new_sum)

                    if tla.const_expr(HAS_ADD == 1):
                        mutex_add.unlock(pipe=tla.arch.VECTOR)
                    mutex_x.unlock(pipe=tla.arch.VECTOR)

                # Pass 1 tail: process 64-lane blocks and narrow only the last
                # reduction. Every tail block updates the running statistics.
                if tla.const_expr(TAIL != 0):
                    tail_offset = n_chunks * CHUNK
                    gm_tail_x = tla.make_tensor(
                        gm_x.ptr + row_off + tail_offset, tail_dma_layout
                    )
                    mutex_x.lock(pipe=tla.arch.MTE2)
                    tla.copy(tail_x_dma, gm_tail_x)
                    mutex_x.unlock(pipe=tla.arch.MTE2)
                    if tla.const_expr(HAS_ADD == 1):
                        gm_tail_add = tla.make_tensor(
                            gm_add.ptr + row_off + tail_offset,
                            tail_dma_layout,
                        )
                        mutex_add.lock(pipe=tla.arch.MTE2)
                        tla.copy(tail_a_dma, gm_tail_add)
                        mutex_add.unlock(pipe=tla.arch.MTE2)

                    mutex_x.lock(pipe=tla.arch.VECTOR)
                    if tla.const_expr(HAS_ADD == 1):
                        mutex_add.lock(pipe=tla.arch.VECTOR)
                    with tla.vec.func(mode="simd"):
                        full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        running_max = ub_max.load()
                        running_sum = ub_sum.load()
                        for block_idx in tla.range_constexpr(tail_blocks):
                            remainder = TAIL - block_idx * vl_ele
                            valid = vl_ele if remainder >= vl_ele else remainder
                            reduce_mask, _ = tla.update_mask(valid, dtype=tla.Float32)
                            xc = tla.tile_view(
                                tail_x_vec,
                                tla.make_shape(1, vl_ele),
                                tla.make_coord(0, block_idx),
                            )
                            value = xc.load()
                            if tla.const_expr(HAS_ADD == 1):
                                ac = tla.tile_view(
                                    tail_a_vec,
                                    tla.make_shape(1, vl_ele),
                                    tla.make_coord(0, block_idx),
                                )
                                value = tla.add(value, ac.load(), mask=full)
                            value = tla.mul(value, SCALE, mask=full)
                            xc.store(value)
                            new_max = tla.max(
                                running_max,
                                tla.full(
                                    value.reduce(
                                        tla.ReductionOp.MAX,
                                        mask=reduce_mask,
                                    ),
                                    dtype=tla.Float32,
                                ),
                                mask=full,
                            )
                            exp_value = tla.exp(
                                tla.sub(value, new_max, mask=full),
                                mask=full,
                            )
                            running_sum = tla.add(
                                tla.mul(
                                    running_sum,
                                    tla.exp(
                                        tla.sub(running_max, new_max, mask=full),
                                        mask=full,
                                    ),
                                    mask=full,
                                ),
                                tla.full(
                                    exp_value.reduce(
                                        tla.ReductionOp.ADD,
                                        mask=reduce_mask,
                                    ),
                                    dtype=tla.Float32,
                                ),
                                mask=full,
                            )
                            running_max = new_max
                        ub_max.store(running_max)
                        ub_sum.store(running_sum)

                    if tla.const_expr(HAS_ADD == 1):
                        mutex_add.unlock(pipe=tla.arch.VECTOR)
                    mutex_x.unlock(pipe=tla.arch.VECTOR)

                # Long-row pass 2: re-read each slice and normalize with the
                # final stats. The output alias avoids any probability buffer.
                for second_chunk in tla.range(0, n_chunks, 1):
                    second_gm_chunk_x = tla.make_tensor(
                        gm_x.ptr + row_off + second_chunk * CHUNK,
                        chunk_layout,
                    )
                    second_gm_chunk_y = tla.make_tensor(
                        gm_y.ptr + row_off + second_chunk * CHUNK,
                        chunk_layout,
                    )
                    mutex_x.lock(pipe=tla.arch.MTE2)
                    tla.copy(ub_x, second_gm_chunk_x)
                    mutex_x.unlock(pipe=tla.arch.MTE2)
                    if tla.const_expr(HAS_ADD == 1):
                        second_gm_chunk_add = tla.make_tensor(
                            gm_add.ptr + row_off + second_chunk * CHUNK,
                            chunk_layout,
                        )
                        mutex_add.lock(pipe=tla.arch.MTE2)
                        tla.copy(ub_add, second_gm_chunk_add)
                        mutex_add.unlock(pipe=tla.arch.MTE2)

                    mutex_x.lock(pipe=tla.arch.VECTOR)
                    if tla.const_expr(HAS_ADD == 1):
                        mutex_add.lock(pipe=tla.arch.VECTOR)
                    with tla.vec.func(mode="simd"):
                        second_full = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        second_row_max = ub_max.load()
                        second_inv_sum = tla.div(
                            tla.full(1.0, dtype=tla.Float32),
                            ub_sum.load(),
                            mask=second_full,
                        )
                        for second_lane in tla.range_constexpr(lanes):
                            second_xc = tla.tile_view(
                                ub_x,
                                tla.make_shape(1, vl_ele),
                                tla.make_coord(0, second_lane),
                            )
                            second_value = second_xc.load()
                            if tla.const_expr(HAS_ADD == 1):
                                second_ac = tla.tile_view(
                                    ub_add,
                                    tla.make_shape(1, vl_ele),
                                    tla.make_coord(0, second_lane),
                                )
                                second_value = tla.add(
                                    second_value,
                                    second_ac.load(),
                                    mask=second_full,
                                )
                            second_value = tla.mul(
                                second_value, SCALE, mask=second_full
                            )
                            second_xc.store(
                                tla.mul(
                                    tla.exp(
                                        tla.sub(
                                            second_value,
                                            second_row_max,
                                            mask=second_full,
                                        ),
                                        mask=second_full,
                                    ),
                                    second_inv_sum,
                                    mask=second_full,
                                )
                            )

                    if tla.const_expr(HAS_ADD == 1):
                        mutex_add.unlock(pipe=tla.arch.VECTOR)
                    mutex_x.unlock(pipe=tla.arch.VECTOR)

                    mutex_x.lock(pipe=tla.arch.MTE3)
                    tla.copy(second_gm_chunk_y, ub_x)
                    mutex_x.unlock(pipe=tla.arch.MTE3)

                # Pass 2 tail: normalize the same 64-lane blocks and copy only
                # the valid TAIL elements back to GM.
                if tla.const_expr(TAIL != 0):
                    tail_offset = n_chunks * CHUNK
                    gm_tail_x = tla.make_tensor(
                        gm_x.ptr + row_off + tail_offset, tail_dma_layout
                    )
                    gm_tail_y = tla.make_tensor(
                        gm_y.ptr + row_off + tail_offset, tail_dma_layout
                    )
                    mutex_x.lock(pipe=tla.arch.MTE2)
                    tla.copy(tail_x_dma, gm_tail_x)
                    mutex_x.unlock(pipe=tla.arch.MTE2)
                    if tla.const_expr(HAS_ADD == 1):
                        gm_tail_add = tla.make_tensor(
                            gm_add.ptr + row_off + tail_offset,
                            tail_dma_layout,
                        )
                        mutex_add.lock(pipe=tla.arch.MTE2)
                        tla.copy(tail_a_dma, gm_tail_add)
                        mutex_add.unlock(pipe=tla.arch.MTE2)

                    mutex_x.lock(pipe=tla.arch.VECTOR)
                    if tla.const_expr(HAS_ADD == 1):
                        mutex_add.lock(pipe=tla.arch.VECTOR)
                    with tla.vec.func(mode="simd"):
                        full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        row_max = ub_max.load()
                        inv_sum = tla.div(
                            tla.full(1.0, dtype=tla.Float32),
                            ub_sum.load(),
                            mask=full,
                        )
                        for block_idx in tla.range_constexpr(tail_blocks):
                            xc = tla.tile_view(
                                tail_x_vec,
                                tla.make_shape(1, vl_ele),
                                tla.make_coord(0, block_idx),
                            )
                            value = xc.load()
                            if tla.const_expr(HAS_ADD == 1):
                                ac = tla.tile_view(
                                    tail_a_vec,
                                    tla.make_shape(1, vl_ele),
                                    tla.make_coord(0, block_idx),
                                )
                                value = tla.add(value, ac.load(), mask=full)
                            value = tla.mul(value, SCALE, mask=full)
                            xc.store(
                                tla.mul(
                                    tla.exp(
                                        tla.sub(value, row_max, mask=full),
                                        mask=full,
                                    ),
                                    inv_sum,
                                    mask=full,
                                )
                            )

                    if tla.const_expr(HAS_ADD == 1):
                        mutex_add.unlock(pipe=tla.arch.VECTOR)
                    mutex_x.unlock(pipe=tla.arch.VECTOR)

                    mutex_x.lock(pipe=tla.arch.MTE3)
                    tla.copy(gm_tail_y, tail_x_dma)
                    mutex_x.unlock(pipe=tla.arch.MTE3)


_SYNC_ALL_SOURCE = r"""
#include "kernel_operator.h"
extern "C" {
[aicore] __attribute__((always_inline)) void streaming_tla_sync_all() {
    AscendC::SyncAll();
}
}
"""


@tla.extern(source=_SYNC_ALL_SOURCE, name="streaming_tla_sync_all")
def streaming_tla_sync_all() -> None: ...


@tla.kernel
def streaming_softmax_split_kernel(
    gm_x: tla.Tensor,
    gm_add: tla.Tensor,
    gm_partial: tla.Tensor,
    gm_y: tla.Tensor,
    N: tla.Constexpr[int],
    CHUNK: tla.Constexpr[int],
    SCALE: tla.Constexpr[float],
    HAS_ADD: tla.Constexpr[int],
) -> None:
    """Two-stage, one-launch split-row softmax.

    Stage 1 computes chunk max and exp-sum in parallel. A device-wide AIV
    barrier publishes scalar statistics to GM. Stage 2 scans the compact
    statistics, computes the row max/sum, and normalizes each chunk.
    """
    vl_ele = 64
    min_float = -3.4028234663852886e38
    lanes = CHUNK // vl_ele
    parts = N // CHUNK
    padded_parts = ((parts + vl_ele - 1) // vl_ele) * vl_ele
    stat_blocks = padded_parts // vl_ele
    parts_i32 = tla.Int32(parts)
    rows = gm_x.origin_shape[0]
    total_tasks = rows * parts

    mutex_x = tla.mutex(resource="streaming_split_x", id=0)
    mutex_add = tla.mutex(resource="streaming_split_add", id=1)
    mutex_part = tla.mutex(resource="streaming_split_part", id=2)
    mutex_stat = tla.mutex(resource="streaming_split_stat", id=3)

    x_ptr = tla.allocate(CHUNK, tla.Float32, tla.AddressSpace.ub, 256)
    add_ptr = tla.allocate(CHUNK, tla.Float32, tla.AddressSpace.ub, 256)
    stat_max_ptr = tla.allocate(vl_ele, tla.Float32, tla.AddressSpace.ub, 256)
    stat_sum_ptr = tla.allocate(vl_ele, tla.Float32, tla.AddressSpace.ub, 256)
    row_max_ptr = tla.allocate(vl_ele, tla.Float32, tla.AddressSpace.ub, 256)
    row_sum_ptr = tla.allocate(vl_ele, tla.Float32, tla.AddressSpace.ub, 256)
    load_max_ptr = tla.allocate(vl_ele, tla.Float32, tla.AddressSpace.ub, 256)
    load_sum_ptr = tla.allocate(vl_ele, tla.Float32, tla.AddressSpace.ub, 256)

    chunk_layout = tla.make_layout(
        tla.make_shape(1, CHUNK),
        tla.make_stride(CHUNK, 1),
        origin_shape=tla.make_shape(1, CHUNK),
    )
    vector_layout = tla.make_layout(
        tla.make_shape(1, vl_ele),
        tla.make_stride(1, 1),
        origin_shape=tla.make_shape(1, vl_ele),
    )
    scalar_layout = tla.make_layout(
        tla.make_shape(1),
        tla.make_stride(1),
        origin_shape=tla.make_shape(1),
    )
    ub_x = tla.make_tensor(x_ptr, chunk_layout)
    ub_add = tla.make_tensor(add_ptr, chunk_layout)
    ub_stat_max = tla.make_tensor(stat_max_ptr, vector_layout)
    ub_stat_sum = tla.make_tensor(stat_sum_ptr, vector_layout)
    ub_stat_max_one = tla.make_tensor(stat_max_ptr, scalar_layout)
    ub_stat_sum_one = tla.make_tensor(stat_sum_ptr, scalar_layout)
    ub_row_max = tla.make_tensor(row_max_ptr, vector_layout)
    ub_row_sum = tla.make_tensor(row_sum_ptr, vector_layout)
    ub_load_max = tla.make_tensor(load_max_ptr, vector_layout)
    ub_load_sum = tla.make_tensor(load_sum_ptr, vector_layout)

    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()
    with tla.vector():
        # Stage 1: compute one max/sum pair per chunk.
        for split_task_1 in tla.range(gid, total_tasks, aiv_num):
            split_task_i32_1 = split_task_1.to(tla.Int32)
            split_row_1 = split_task_i32_1 // parts_i32
            split_part_1 = split_task_i32_1 % parts_i32
            split_row_off_1 = split_row_1.to(tla.Int64) * tla.Int64(N)
            split_off_1 = split_row_off_1 + split_part_1.to(tla.Int64) * tla.Int64(
                CHUNK
            )
            split_x_gm_1 = tla.make_tensor(gm_x.ptr + split_off_1, chunk_layout)
            mutex_x.lock(pipe=tla.arch.MTE2)
            tla.copy(ub_x, split_x_gm_1)
            mutex_x.unlock(pipe=tla.arch.MTE2)
            if tla.const_expr(HAS_ADD == 1):
                split_add_gm_1 = tla.make_tensor(gm_add.ptr + split_off_1, chunk_layout)
                mutex_add.lock(pipe=tla.arch.MTE2)
                tla.copy(ub_add, split_add_gm_1)
                mutex_add.unlock(pipe=tla.arch.MTE2)

            mutex_x.lock(pipe=tla.arch.VECTOR)
            if tla.const_expr(HAS_ADD == 1):
                mutex_add.lock(pipe=tla.arch.VECTOR)
            mutex_stat.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                split_full_1 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                split_chunk_max_1 = tla.full(min_float, dtype=tla.Float32)
                for split_lane_1 in tla.range_constexpr(lanes):
                    split_xc_1 = tla.tile_view(
                        ub_x,
                        tla.make_shape(1, vl_ele),
                        tla.make_coord(0, split_lane_1),
                    )
                    split_value_1 = split_xc_1.load()
                    if tla.const_expr(HAS_ADD == 1):
                        split_ac_1 = tla.tile_view(
                            ub_add,
                            tla.make_shape(1, vl_ele),
                            tla.make_coord(0, split_lane_1),
                        )
                        split_value_1 = tla.add(
                            split_value_1, split_ac_1.load(), mask=split_full_1
                        )
                    split_value_1 = tla.mul(split_value_1, SCALE, mask=split_full_1)
                    split_xc_1.store(split_value_1)
                    split_chunk_max_1 = tla.max(
                        split_chunk_max_1, split_value_1, mask=split_full_1
                    )
                split_max_b_1 = tla.full(
                    split_chunk_max_1.reduce(tla.ReductionOp.MAX, mask=split_full_1),
                    dtype=tla.Float32,
                )
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE,
                    tla.params.MemType.VEC_LOAD,
                )
                split_acc_1 = tla.full(0.0, dtype=tla.Float32)
                for split_lane_2 in tla.range_constexpr(lanes):
                    split_xc_2 = tla.tile_view(
                        ub_x,
                        tla.make_shape(1, vl_ele),
                        tla.make_coord(0, split_lane_2),
                    )
                    split_acc_1 = tla.add(
                        split_acc_1,
                        tla.exp(
                            tla.sub(
                                split_xc_2.load(),
                                split_max_b_1,
                                mask=split_full_1,
                            ),
                            mask=split_full_1,
                        ),
                        mask=split_full_1,
                    )
                split_sum_b_1 = tla.full(
                    split_acc_1.reduce(tla.ReductionOp.ADD, mask=split_full_1),
                    dtype=tla.Float32,
                )
                ub_stat_max_one.store(split_max_b_1)
                ub_stat_sum_one.store(split_sum_b_1)

            if tla.const_expr(HAS_ADD == 1):
                mutex_add.unlock(pipe=tla.arch.VECTOR)
            mutex_x.unlock(pipe=tla.arch.VECTOR)
            mutex_stat.unlock(pipe=tla.arch.VECTOR)

            split_row_offset_stats_1 = split_row_1.to(tla.Int64) * tla.Int64(
                padded_parts
            ) + split_part_1.to(tla.Int64)
            split_sum_offset_1 = (
                tla.Int64(rows * padded_parts) + split_row_offset_stats_1
            )
            split_max_gm_1 = tla.make_tensor(
                gm_partial.ptr + split_row_offset_stats_1, scalar_layout
            )
            split_sum_gm_1 = tla.make_tensor(
                gm_partial.ptr + split_sum_offset_1, scalar_layout
            )
            mutex_stat.lock(pipe=tla.arch.MTE3)
            tla.copy(split_max_gm_1, ub_stat_max_one)
            tla.copy(split_sum_gm_1, ub_stat_sum_one)
            mutex_stat.unlock(pipe=tla.arch.MTE3)

        tla.pipe_barrier(tla.pipes.ALL)
        streaming_tla_sync_all()
        tla.pipe_barrier(tla.pipes.ALL)

        # Stage 2: one task per row computes final (max, sum) from chunk
        # partials and publishes compact row statistics.
        final_base = 2 * rows * padded_parts
        for split_row_3 in tla.range(gid, rows, aiv_num):
            with tla.vec.func(mode="simd"):
                ub_row_max.store(tla.full(min_float, dtype=tla.Float32))
                ub_row_sum.store(tla.full(0.0, dtype=tla.Float32))

            for split_sb_3 in tla.range(0, stat_blocks, 1):
                split_stat_base_3 = split_row_3.to(tla.Int64) * tla.Int64(
                    padded_parts
                ) + split_sb_3.to(tla.Int64) * tla.Int64(vl_ele)
                split_block_max_gm_3 = tla.make_tensor(
                    gm_partial.ptr + split_stat_base_3, vector_layout
                )
                mutex_part.lock(pipe=tla.arch.MTE2)
                tla.copy(ub_load_max, split_block_max_gm_3)
                mutex_part.unlock(pipe=tla.arch.MTE2)
                mutex_part.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    split_remain_3 = parts - split_sb_3 * vl_ele
                    split_valid_3 = (
                        vl_ele if split_remain_3 >= vl_ele else split_remain_3
                    )
                    split_mask_3, _ = tla.update_mask(split_valid_3, dtype=tla.Float32)
                    split_block_max_3 = tla.full(
                        ub_load_max.load().reduce(
                            tla.ReductionOp.MAX, mask=split_mask_3
                        ),
                        dtype=tla.Float32,
                    )
                    split_max_3 = tla.max(
                        ub_row_max.load(),
                        split_block_max_3,
                        mask=tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32),
                    )
                    ub_row_max.store(split_max_3)
                mutex_part.unlock(pipe=tla.arch.VECTOR)

            for split_sb_4 in tla.range(0, stat_blocks, 1):
                split_stat_base_4 = split_row_3.to(tla.Int64) * tla.Int64(
                    padded_parts
                ) + split_sb_4.to(tla.Int64) * tla.Int64(vl_ele)
                split_stat_sum_base_4 = split_stat_base_4 + tla.Int64(
                    rows * padded_parts
                )
                split_block_max_gm_4 = tla.make_tensor(
                    gm_partial.ptr + split_stat_base_4, vector_layout
                )
                split_block_sum_gm_4 = tla.make_tensor(
                    gm_partial.ptr + split_stat_sum_base_4, vector_layout
                )
                mutex_part.lock(pipe=tla.arch.MTE2)
                tla.copy(ub_load_max, split_block_max_gm_4)
                tla.copy(ub_load_sum, split_block_sum_gm_4)
                mutex_part.unlock(pipe=tla.arch.MTE2)
                mutex_part.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    split_remain_4 = parts - split_sb_4 * vl_ele
                    split_valid_4 = (
                        vl_ele if split_remain_4 >= vl_ele else split_remain_4
                    )
                    split_mask_4, _ = tla.update_mask(split_valid_4, dtype=tla.Float32)
                    split_full_4 = tla.create_mask(
                        pattern=tla.mask.ALL, dtype=tla.Float32
                    )
                    split_contrib_4 = tla.mul(
                        ub_load_sum.load(),
                        tla.exp(
                            tla.sub(
                                ub_load_max.load(),
                                ub_row_max.load(),
                                mask=split_full_4,
                            ),
                            mask=split_full_4,
                        ),
                        mask=split_full_4,
                    )
                    split_sum_4 = tla.add(
                        ub_row_sum.load(),
                        tla.full(
                            split_contrib_4.reduce(
                                tla.ReductionOp.ADD, mask=split_mask_4
                            ),
                            dtype=tla.Float32,
                        ),
                        mask=split_full_4,
                    )
                    ub_row_sum.store(split_sum_4)
                mutex_part.unlock(pipe=tla.arch.VECTOR)

            mutex_stat.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                final_one, _ = tla.update_mask(1, dtype=tla.Float32)
                final_full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                ub_stat_max_one.store(
                    tla.full(
                        ub_row_max.load().reduce(tla.ReductionOp.MAX, mask=final_full),
                        dtype=tla.Float32,
                    )
                )
                ub_stat_sum_one.store(
                    tla.full(
                        ub_row_sum.load().reduce(tla.ReductionOp.ADD, mask=final_one),
                        dtype=tla.Float32,
                    )
                )
            mutex_stat.unlock(pipe=tla.arch.VECTOR)

            split_final_max_gm = tla.make_tensor(
                gm_partial.ptr + tla.Int64(final_base) + split_row_3.to(tla.Int64),
                scalar_layout,
            )
            split_final_sum_gm = tla.make_tensor(
                gm_partial.ptr
                + tla.Int64(final_base + rows)
                + split_row_3.to(tla.Int64),
                scalar_layout,
            )
            mutex_stat.lock(pipe=tla.arch.MTE3)
            tla.copy(split_final_max_gm, ub_stat_max_one)
            tla.copy(split_final_sum_gm, ub_stat_sum_one)
            mutex_stat.unlock(pipe=tla.arch.MTE3)

        tla.pipe_barrier(tla.pipes.ALL)
        streaming_tla_sync_all()
        tla.pipe_barrier(tla.pipes.ALL)

        # Stage 3: normalize every chunk with the compact row statistics.
        for split_task_5 in tla.range(gid, total_tasks, aiv_num):
            split_task_i32_5 = split_task_5.to(tla.Int32)
            split_row_5 = split_task_i32_5 // parts_i32
            split_part_5 = split_task_i32_5 % parts_i32
            split_row_off_5 = split_row_5.to(tla.Int64) * tla.Int64(N)
            split_off_5 = split_row_off_5 + split_part_5.to(tla.Int64) * tla.Int64(
                CHUNK
            )
            split_final_max_gm_5 = tla.make_tensor(
                gm_partial.ptr + tla.Int64(final_base) + split_row_5.to(tla.Int64),
                scalar_layout,
            )
            split_final_sum_gm_5 = tla.make_tensor(
                gm_partial.ptr
                + tla.Int64(final_base + rows)
                + split_row_5.to(tla.Int64),
                scalar_layout,
            )
            mutex_stat.lock(pipe=tla.arch.MTE2)
            tla.copy(ub_stat_max_one, split_final_max_gm_5)
            tla.copy(ub_stat_sum_one, split_final_sum_gm_5)
            mutex_stat.unlock(pipe=tla.arch.MTE2)

            split_x_gm_5 = tla.make_tensor(gm_x.ptr + split_off_5, chunk_layout)
            split_y_gm_5 = tla.make_tensor(gm_y.ptr + split_off_5, chunk_layout)
            mutex_x.lock(pipe=tla.arch.MTE2)
            tla.copy(ub_x, split_x_gm_5)
            mutex_x.unlock(pipe=tla.arch.MTE2)
            if tla.const_expr(HAS_ADD == 1):
                split_add_gm_5 = tla.make_tensor(gm_add.ptr + split_off_5, chunk_layout)
                mutex_add.lock(pipe=tla.arch.MTE2)
                tla.copy(ub_add, split_add_gm_5)
                mutex_add.unlock(pipe=tla.arch.MTE2)

            mutex_stat.lock(pipe=tla.arch.VECTOR)
            mutex_x.lock(pipe=tla.arch.VECTOR)
            if tla.const_expr(HAS_ADD == 1):
                mutex_add.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                split_full_5 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                split_one_5, _ = tla.update_mask(1, dtype=tla.Float32)
                split_row_max_5 = tla.full(
                    ub_stat_max.load().reduce(tla.ReductionOp.MAX, mask=split_one_5),
                    dtype=tla.Float32,
                )
                split_row_sum_5 = tla.full(
                    ub_stat_sum.load().reduce(tla.ReductionOp.ADD, mask=split_one_5),
                    dtype=tla.Float32,
                )
                split_inv_5 = tla.div(
                    tla.full(1.0, dtype=tla.Float32),
                    split_row_sum_5,
                    mask=split_full_5,
                )
                for split_lane_5 in tla.range_constexpr(lanes):
                    split_xc_5 = tla.tile_view(
                        ub_x,
                        tla.make_shape(1, vl_ele),
                        tla.make_coord(0, split_lane_5),
                    )
                    split_value_5 = split_xc_5.load()
                    if tla.const_expr(HAS_ADD == 1):
                        split_ac_5 = tla.tile_view(
                            ub_add,
                            tla.make_shape(1, vl_ele),
                            tla.make_coord(0, split_lane_5),
                        )
                        split_value_5 = tla.add(
                            split_value_5, split_ac_5.load(), mask=split_full_5
                        )
                    split_value_5 = tla.mul(split_value_5, SCALE, mask=split_full_5)
                    split_xc_5.store(
                        tla.mul(
                            tla.exp(
                                tla.sub(
                                    split_value_5,
                                    split_row_max_5,
                                    mask=split_full_5,
                                ),
                                mask=split_full_5,
                            ),
                            split_inv_5,
                            mask=split_full_5,
                        )
                    )

            if tla.const_expr(HAS_ADD == 1):
                mutex_add.unlock(pipe=tla.arch.VECTOR)
            mutex_x.unlock(pipe=tla.arch.VECTOR)
            mutex_stat.unlock(pipe=tla.arch.VECTOR)

            mutex_x.lock(pipe=tla.arch.MTE3)
            tla.copy(split_y_gm_5, ub_x)
            mutex_x.unlock(pipe=tla.arch.MTE3)


VL_ELE = 64
MAX_CHUNK = 2048
ROW_RESIDENT_LIMIT = 2048

_COMPILED = {}
_NULL_ADD = {}
_INPUT_WRAPPERS = {}


def _pick_chunk(n):
    """Choose a row-resident size or the largest streaming slice."""
    if n <= ROW_RESIDENT_LIMIT and n % VL_ELE == 0:
        return n
    chunk = min(n, MAX_CHUNK)
    chunk -= chunk % VL_ELE
    return max(VL_ELE, chunk)


def _pick_split_chunk(n, m, cores):
    """Use split-row kernels when one row cannot saturate the AIV grid."""
    if m >= cores or n <= ROW_RESIDENT_LIMIT or n % VL_ELE != 0 or m * n < 8192:
        return None
    target_tasks = max(1, cores)
    for chunk in (1024, 512):
        if n % chunk == 0 and m * (n // chunk) >= target_tasks:
            return chunk
    return None


def _vector_cores(index):
    import torch

    return int(torch.npu.get_device_properties(index).vector_core_num)


def _tla_input(tensor, layout_tag):
    """Cache descriptors for stable inputs to reduce per-call host work."""
    from catlass.tla.runtime import from_dlpack

    key = id(tensor)
    signature = (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        str(tensor.device),
        layout_tag,
    )
    cached = _INPUT_WRAPPERS.get(key)
    if cached is not None and cached[0]() is tensor and cached[1] == signature:
        return cached[2]

    wrapped = from_dlpack(tensor, layout_tag=layout_tag)
    wrappers = _INPUT_WRAPPERS
    cache_key = key

    def drop(dead_ref):
        entry = wrappers.get(cache_key)
        if entry is not None and entry[0] is dead_ref:
            wrappers.pop(cache_key, None)

    wrappers[cache_key] = (weakref.ref(tensor, drop), signature, wrapped)
    return wrapped


def _prepare_inputs(x, add, scale):
    import torch

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.dim() != 2:
        raise ValueError("x must be 2-D [M, N]")
    if x.dtype != torch.float32:
        raise ValueError("x must be float32")
    m, n = x.shape
    if n < VL_ELE:
        raise ValueError("N must be >= {}, got {}".format(VL_ELE, n))

    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")

    x = x.contiguous()
    has_add = 0 if add is None else 1
    if has_add:
        if not isinstance(add, torch.Tensor):
            raise TypeError("add must be a torch.Tensor or None")
        if add.shape != x.shape:
            raise ValueError("add must have the same shape as x")
        if add.dtype != torch.float32:
            raise ValueError("add must be float32")
        add = add.contiguous()

    device_index = torch.npu.current_device()
    if not has_add:
        add = _NULL_ADD.get(device_index)
        if add is None:
            add = torch.zeros((1, 1), dtype=torch.float32, device=x.device)
            _NULL_ADD[device_index] = add

    return x, add, scale, m, n, has_add, device_index


def _ensure_plan(x, add, scale, m, n, has_add, device_index):
    """Return a cached launch plan for already validated inputs."""
    import torch

    if m == 0:
        return None
    cores = _vector_cores(device_index)
    split_chunk = _pick_split_chunk(n, m, cores)
    chunk = split_chunk if split_chunk is not None else _pick_chunk(n)
    tail = 0 if split_chunk is not None else n - (n // chunk) * chunk
    path = "split" if split_chunk is not None else "single"
    key = (m, n, path, chunk, tail, scale, has_add, str(x.device))
    if key not in _COMPILED:
        output = torch.empty((m, n), dtype=torch.float32, device=x.device)
        x_t = _tla_input(x, tla.arch.RowMajor)
        add_t = _tla_input(add, tla.arch.RowMajor)
        y_t = _tla_input(output, tla.arch.RowMajor)
        if split_chunk is not None:
            parts = n // split_chunk
            padded_parts = ((parts + VL_ELE - 1) // VL_ELE) * VL_ELE
            partial = torch.empty(
                (2 * m * padded_parts + 2 * m,),
                dtype=torch.float32,
                device=x.device,
            )
            partial_t = _tla_input(partial, tla.arch.RowMajor)
            split_artifact = tla.compile(
                streaming_softmax_split_kernel,
                x_t,
                add_t,
                partial_t,
                y_t,
                n,
                split_chunk,
                scale,
                has_add,
                options="--npu-arch 3510",
            )
            plan = {
                "kind": "split",
                "cores": cores,
                "chunk": split_chunk,
                "parts": parts,
                "partial": partial,
                "artifact": split_artifact,
            }
        else:
            artifact = tla.compile(
                streaming_masked_softmax_kernel,
                x_t,
                add_t,
                y_t,
                n,
                chunk,
                tail,
                scale,
                has_add,
                options="--npu-arch 3510",
            )
            plan = {"kind": "single", "cores": cores, "artifact": artifact}
        _COMPILED[key] = plan
    return key


def precompile(x, add=None, scale=1.0):
    """Compile and cache the kernel without launching it on the device."""
    prepared = _prepare_inputs(x, add, scale)
    return _ensure_plan(*prepared)


def run(x, add=None, scale=1.0):
    """Return softmax(scale * (x + add)) over the last axis."""
    import torch

    x, add, scale, m, n, has_add, device_index = _prepare_inputs(x, add, scale)
    output = torch.empty_like(x)
    if m == 0:
        return output
    key = _ensure_plan(x, add, scale, m, n, has_add, device_index)
    plan = _COMPILED[key]
    if plan["kind"] == "split":
        plan["artifact"](
            x,
            add,
            plan["partial"],
            output,
            block_num=plan["cores"],
        )
    else:
        plan["artifact"](x, add, output, block_num=plan["cores"])
    return output


def main():
    import argparse

    import torch
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    device = f"npu:{args.device}"
    shapes = ((64, 64), (64, 128), (4, 100), (4, 2100), (4, 3000), (1, 1048576))
    failed = False
    for m, n in shapes:
        x = torch.randn(m, n, dtype=torch.float32, device=device)
        for has_add in (False, True):
            add = None
            if has_add:
                add = torch.randn_like(x)
                if n > 7:
                    add[:, n - 7 :] -= 1.0e9
            scale = 0.125
            output = run(x, add, scale)
            logits = x * scale if add is None else (x + add) * scale
            expected = torch.softmax(logits, dim=-1)
            torch.npu.synchronize()
            max_abs = float((output - expected).abs().max().item())
            ok = bool(torch.isfinite(output).all().item()) and max_abs <= 1.0e-4
            failed |= not ok
            print(
                "shape=({}, {}) add={} max_abs_err={:.9g} passed={}".format(
                    m, n, int(has_add), max_abs, ok
                )
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
