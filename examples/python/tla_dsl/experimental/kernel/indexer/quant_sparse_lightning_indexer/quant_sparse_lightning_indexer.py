# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MXFP4 sparse Lightning Indexer for Ascend 950.

Performance design:

* Prepare uses a block-private UB bitmap to deduplicate candidates without a
  global sort, then distributes one row over several grid-stride AIV splits.
  Packed K and E8M0 scales move as aligned Int32 words.
* QK batches two query rows per Cube launch and uses three coarse explicit
  mutex groups: L1 staging (MTE2/MTE1), L0 staging (MTE1/CUBE), and
  L0C/Fixpipe.  This avoids per-buffer ownership churn while preserving the
  GM->L1->L0->MMAD->Fixpipe dependency chain.
* Large TopK problems reduce 32 heads into BF16 scores in parallel token
  chunks, sort each chunk in UB, and merge only the top half at every tree
  level.  Launch-bound shapes retain the serial merge path.
* Host-side shape/workspace/compiled-artifact/DLPack-template caches remove
  repeated Python and descriptor construction while preserving fresh outputs.

The public entry point is ``run``.
"""

from __future__ import annotations

import copy

import torch
import torch_npu  # noqa: F401

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


Q_HEADS = 32
HEAD_DIM = 128
PACKED_DIM = HEAD_DIM // 2
SCALE_GROUPS = HEAD_DIM // 32
TOKEN_BLOCK = 8
RECORD_BYTES = PACKED_DIM * TOKEN_BLOCK + SCALE_GROUPS * TOKEN_BLOCK
CANDIDATE_CAPACITY = 2048
BN = 128
BM = 32
BK = 128
SK = 32
TOPK_CHUNK = 512


@tla.kernel
def prepare_metadata_kernel(
    candidate_indices: tla.Tensor,
    candidate_length: tla.Tensor,
    cu_seqlens_q: tla.Tensor,
    seqused_q: tla.Tensor,
    seqused_k: tla.Tensor,
    cmp_residual_k: tla.Tensor,
    block_ids: tla.Tensor,
    block_count: tla.Tensor,
    token_count: tla.Tensor,
    visible_out: tla.Tensor,
    block_capacity: tla.Constexpr[int],
    has_residual: tla.Constexpr[bool],
    mask_mode: tla.Constexpr[int],
    cmp_ratio: tla.Constexpr[int],
) -> None:
    """Build canonical ascending unique blocks once per query row."""
    zero = 0
    one = 1
    zero8 = tla.Int8(0)
    one8 = tla.Int8(1)
    threads = 64
    bitmap_ub = tla.make_tensor(
        tla.allocate(block_capacity, tla.Int8, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(block_capacity), tla.make_stride(1)),
    )
    chunk_blocks_ub = tla.make_tensor(
        tla.allocate(threads, tla.Int32, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(threads), tla.make_stride(1)),
    )
    chunk_tokens_ub = tla.make_tensor(
        tla.allocate(threads, tla.Int32, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(threads), tla.make_stride(1)),
    )
    chunk_block_offsets_ub = tla.make_tensor(
        tla.allocate(threads + 1, tla.Int32, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(threads + 1), tla.make_stride(1)),
    )
    chunk_token_offsets_ub = tla.make_tensor(
        tla.allocate(threads + 1, tla.Int32, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(threads + 1), tla.make_stride(1)),
    )
    chunk_slots = (block_capacity + threads - 1) // threads

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=threads):
            tid, _, _ = tla.arch.thread_idx()
            thread_count, _, _ = tla.arch.thread_block_dim()
            block = tla.arch.block_idx()
            row = block
            for slot in tla.range(tid, block_capacity, thread_count):
                bitmap_ub[slot] = zero8
            tla.arch.sync_threads()

            batch = zero
            batch_count = cu_seqlens_q.origin_shape[0] - one
            for q_batch in tla.range_constexpr(zero, batch_count, one):
                if row >= cu_seqlens_q[q_batch] and row < cu_seqlens_q[q_batch + one]:
                    batch = q_batch
            row_in_batch = row - cu_seqlens_q[batch]
            visible = zero
            if row_in_batch < seqused_q[batch]:
                if mask_mode == zero:
                    visible = seqused_k[batch]
                else:
                    residual = zero
                    if has_residual:
                        residual = cmp_residual_k[batch]
                    visible = (
                        seqused_k[batch] * cmp_ratio
                        + residual
                        - seqused_q[batch]
                        + row_in_batch
                        + one
                    ) // cmp_ratio
                    if visible < zero:
                        visible = zero
                    if visible > seqused_k[batch]:
                        visible = seqused_k[batch]
            visible_out[row] = visible

            length = candidate_length[row, zero]
            if length > candidate_indices.origin_shape[1]:
                length = candidate_indices.origin_shape[1]
            if length < zero:
                length = zero
            for c in tla.range(tid, length, thread_count):
                if candidate_indices[row, c] >= zero:
                    if candidate_indices[row, c] < block_capacity:
                        if candidate_indices[row, c] * TOKEN_BLOCK < visible:
                            bitmap_ub[candidate_indices[row, c]] = one8
            tla.arch.sync_threads()

            window_start = tid * chunk_slots
            local_blocks = zero
            local_tokens = zero
            for chunk_offset in tla.range_constexpr(0, chunk_slots, 1):
                scan_block = window_start + chunk_offset
                if scan_block < block_capacity:
                    if bitmap_ub[scan_block] != zero8:
                        local_blocks = local_blocks + one
                        remain = visible_out[row] - scan_block * TOKEN_BLOCK
                        if remain > TOKEN_BLOCK:
                            remain = TOKEN_BLOCK
                        local_tokens = local_tokens + remain
            chunk_blocks_ub[tid] = local_blocks
            chunk_tokens_ub[tid] = local_tokens
            tla.arch.sync_threads()

            chunk_block_offsets_ub[tid] = local_blocks
            chunk_token_offsets_ub[tid] = local_tokens
            tla.arch.sync_threads()

            for scan_step in tla.range_constexpr(0, 6, 1):
                distance = 1 << scan_step
                add_blocks = zero
                add_tokens = zero
                if tid >= distance:
                    add_blocks = chunk_block_offsets_ub[tid - distance]
                    add_tokens = chunk_token_offsets_ub[tid - distance]
                tla.arch.sync_threads()
                chunk_block_offsets_ub[tid] = local_blocks + add_blocks
                chunk_token_offsets_ub[tid] = local_tokens + add_tokens
                local_blocks = chunk_block_offsets_ub[tid]
                local_tokens = chunk_token_offsets_ub[tid]
                tla.arch.sync_threads()

            if tid == threads - one:
                block_count[row] = local_blocks
                token_count[row] = local_tokens
            rank = local_blocks - chunk_blocks_ub[tid]
            for chunk_offset in tla.range_constexpr(0, chunk_slots, 1):
                scan_block = window_start + chunk_offset
                if scan_block < block_capacity:
                    if bitmap_ub[scan_block] != zero8:
                        block_ids[row, rank] = scan_block
                        rank = rank + one

        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def pack_candidates_kernel(
    k_flat: tla.Tensor,
    descale_k_flat: tla.Tensor,
    cu_seqlens_q: tla.Tensor,
    cu_seqlens_k: tla.Tensor,
    block_table: tla.Tensor,
    block_ids: tla.Tensor,
    token_count: tla.Tensor,
    token_k: tla.Tensor,
    token_scale: tla.Tensor,
    rows: tla.Constexpr[int],
    pack_splits: tla.Constexpr[int],
    max_tokens: tla.Constexpr[int],
    is_paged: tla.Constexpr[bool],
    page_size: tla.Constexpr[int],
    records_per_page: tla.Constexpr[int],
) -> None:
    """Pack K records from canonical candidate metadata into dense row storage."""
    zero = 0
    one = 1
    threads = 256
    k_u32 = tla.make_tensor(
        tla.recast_ptr(k_flat.ptr, dtype=tla.Int32),
        tla.make_layout(
            tla.make_shape(k_flat.origin_shape[0] // 4), tla.make_stride(1)
        ),
    )
    dk_u32 = tla.make_tensor(
        tla.recast_ptr(descale_k_flat.ptr, dtype=tla.Int32),
        tla.make_layout(
            tla.make_shape(descale_k_flat.origin_shape[0] // 4),
            tla.make_stride(1),
        ),
    )
    token_k_u32 = tla.make_tensor(
        tla.recast_ptr(token_k.ptr, dtype=tla.Int32),
        tla.make_layout(
            tla.make_shape(rows * max_tokens * PACKED_DIM // 4),
            tla.make_stride(1),
        ),
    )
    token_scale_u32 = tla.make_tensor(
        tla.recast_ptr(token_scale.ptr, dtype=tla.Int32),
        tla.make_layout(
            tla.make_shape(rows * max_tokens),
            tla.make_stride(1),
        ),
    )

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=threads):
            tid, _, _ = tla.arch.thread_idx()
            thread_count, _, _ = tla.arch.thread_block_dim()
            block = tla.arch.block_idx()
            row = block // pack_splits
            pack_split = block % pack_splits
            total = token_count[row]
            batch = zero
            batch_count = cu_seqlens_q.origin_shape[0] - one
            for q_batch in tla.range_constexpr(zero, batch_count, one):
                if row >= cu_seqlens_q[q_batch] and row < cu_seqlens_q[q_batch + one]:
                    batch = q_batch

            for dense in tla.range(
                tid + pack_split * thread_count,
                total,
                thread_count * pack_splits,
            ):
                block_rank = dense // TOKEN_BLOCK
                tail = dense % TOKEN_BLOCK
                logical = block_ids[row, block_rank] * TOKEN_BLOCK + tail
                source = zero
                scale_source = zero
                if is_paged:
                    logical_page = logical // page_size
                    page_offset = logical % page_size
                    physical_page = block_table[batch, logical_page]
                    record = page_offset // TOKEN_BLOCK
                    record_start = (
                        physical_page * records_per_page + record
                    ) * RECORD_BYTES
                    source = record_start + tail * PACKED_DIM
                    scale_source = (
                        record_start + PACKED_DIM * TOKEN_BLOCK + tail * SCALE_GROUPS
                    )
                else:
                    physical = cu_seqlens_k[batch] + logical
                    source = physical * PACKED_DIM
                    scale_source = physical * SCALE_GROUPS

                flat_token = row * max_tokens + dense
                destination_word = flat_token * PACKED_DIM // 4
                source_word = source // 4
                for word in tla.range_constexpr(0, PACKED_DIM // 4, 1):
                    token_k_u32[destination_word + word] = k_u32[source_word + word]
                token_scale_u32[flat_token] = (
                    dk_u32[scale_source // 4]
                    if not is_paged
                    else k_u32[scale_source // 4]
                )

        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def qk_kernel(
    q_pack: tla.Tensor,
    q_scale: tla.Tensor,
    k_pack: tla.Tensor,
    k_scale: tla.Tensor,
    block_count: tla.Tensor,
    qk_out: tla.Tensor,
    rows: tla.Constexpr[int],
    tokens_per_row: tla.Constexpr[int],
) -> None:
    """Odd-row QK; explicit mutexes own L1, L0 and L0C staging lifetimes."""
    zero = 0
    one = 1
    tiles_per_row = tokens_per_row // BN
    total_tasks = rows * tiles_per_row
    dtype_a = q_pack.ptr.dtype
    dtype_b = k_pack.ptr.dtype
    mutex_l1 = tla.mutex(resource="qk_l1", id=0)
    mutex_l0 = tla.mutex(resource="qk_l0", id=1)
    mutex_l0c = tla.mutex(resource="qk_l0c", id=2)

    l1a = tla.allocate(BM * BK, dtype_a, tla.AddressSpace.l1, 512)
    l1b = tla.allocate(BK * BN, dtype_b, tla.AddressSpace.l1, 512)
    l1sa = tla.allocate(BM * SCALE_GROUPS, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb = tla.allocate(SCALE_GROUPS * BN, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(BM * BK, dtype_a, tla.AddressSpace.l0a, 1024)
    l0b = tla.allocate(BK * BN, dtype_b, tla.AddressSpace.l0b, 1024)
    l0c = tla.allocate(BM * BN, tla.Float32, tla.AddressSpace.l0c, 1024)

    with tla.cube():
        for task in tla.range(tla.arch.block_idx(), total_tasks, tla.arch.block_num()):
            row = task // tiles_per_row
            nt = task % tiles_per_row
            valid_tokens = block_count[row] * TOKEN_BLOCK
            if nt * BN < valid_tokens:
                gm_a = tla.tile_view(
                    q_pack,
                    tla.make_shape(BM, BK),
                    tla.make_coord(row * Q_HEADS, zero),
                )
                global_nt = row * tiles_per_row + nt
                gm_b = tla.tile_view(
                    k_pack,
                    tla.make_shape(BK, BN),
                    tla.make_coord(zero, global_nt),
                )
                gm_sa = tla.tile_view(
                    q_scale,
                    tla.make_shape(BM, SCALE_GROUPS),
                    tla.make_coord(row * Q_HEADS, zero),
                )
                gm_sb = tla.tile_view(
                    k_scale,
                    tla.make_shape(SCALE_GROUPS, BN),
                    tla.make_coord(zero, global_nt),
                )
                gm_c = tla.tile_view(
                    qk_out,
                    tla.make_shape(BM, BN),
                    tla.make_coord(row * Q_HEADS, nt),
                )

                n_valid = gm_b.origin_shape[1]
                t_l1a = tla.make_tensor(
                    l1a, tla.make_layout_with_tag((BM, BK), dtype_a, tla.arch.zN)
                )
                t_l1b = tla.make_tensor(
                    l1b, tla.make_layout_with_tag((BK, n_valid), dtype_b, tla.arch.nZ)
                )
                t_l1sa = tla.make_tensor_like(l1sa, gm_sa, tla.arch.zZMxScale)
                t_l1sb = tla.make_tensor_like(l1sb, gm_sb, tla.arch.nNMxScale)
                t_l0a = tla.make_tensor_like(l0a, t_l1a, tla.arch.zN)
                t_l0b = tla.make_tensor_like(l0b, t_l1b, tla.arch.nZ)
                t_l0c = tla.make_tensor_like(l0c, gm_c)

                mutex_l1.lock(pipe=tla.pipes.MTE2)
                tla.copy(t_l1a, gm_a)
                tla.copy(t_l1b, gm_b)
                tla.copy(t_l1sa, gm_sa)
                tla.copy(t_l1sb, gm_sb)
                mutex_l1.unlock(pipe=tla.pipes.MTE2)

                mutex_l1.lock(pipe=tla.pipes.MTE1)
                mutex_l0.lock(pipe=tla.pipes.MTE1)
                tla.copy(t_l0a, t_l1a, scale=t_l1sa)
                tla.copy(t_l0b, t_l1b, scale=t_l1sb)
                mutex_l0.unlock(pipe=tla.pipes.MTE1)
                mutex_l1.unlock(pipe=tla.pipes.MTE1)

                mutex_l0.lock(pipe=tla.pipes.CUBE)
                mutex_l0c.lock(pipe=tla.pipes.CUBE)
                tla.mmad_mx(t_l0c, t_l0a, t_l0b, init_c=True, unit_flag=0b11)
                mutex_l0c.unlock(pipe=tla.pipes.CUBE)
                mutex_l0.unlock(pipe=tla.pipes.CUBE)

                mutex_l0c.lock(pipe=tla.pipes.FIX)
                tla.copy(
                    gm_c,
                    t_l0c,
                    tla.params.CopyL0C2DstParams(unit_flag=0b11),
                )
                mutex_l0c.unlock(pipe=tla.pipes.FIX)


@tla.kernel
def qk2_kernel(
    q_pack: tla.Tensor,
    q_scale0: tla.Tensor,
    k_pack0: tla.Tensor,
    k_scale0: tla.Tensor,
    block_count0: tla.Tensor,
    qk_out0: tla.Tensor,
    q_scale1: tla.Tensor,
    k_pack1: tla.Tensor,
    k_scale1: tla.Tensor,
    block_count1: tla.Tensor,
    qk_out1: tla.Tensor,
    tokens_per_row: tla.Constexpr[int],
    pair_base: tla.Constexpr[int],
) -> None:
    """Two-row QK launch with row-private descriptors and explicit mutexes."""
    zero = 0
    tiles_per_row = tokens_per_row // BN
    total_tasks = 2 * tiles_per_row
    dtype_a = q_pack.ptr.dtype
    dtype_b = k_pack0.ptr.dtype
    mutex_l1 = tla.mutex(resource="qk2_l1", id=0)
    mutex_l0 = tla.mutex(resource="qk2_l0", id=1)
    mutex_l0c = tla.mutex(resource="qk2_l0c", id=2)

    l1a = tla.allocate(BM * BK, dtype_a, tla.AddressSpace.l1, 512)
    l1b = tla.allocate(BK * BN, dtype_b, tla.AddressSpace.l1, 512)
    l1sa = tla.allocate(BM * SCALE_GROUPS, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb = tla.allocate(SCALE_GROUPS * BN, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(BM * BK, dtype_a, tla.AddressSpace.l0a, 1024)
    l0b = tla.allocate(BK * BN, dtype_b, tla.AddressSpace.l0b, 1024)
    l0c = tla.allocate(BM * BN, tla.Float32, tla.AddressSpace.l0c, 1024)
    q_row_ptr = tla.recast_ptr(
        tla.recast_ptr(q_pack.ptr, dtype=tla.UInt8) + pair_base * Q_HEADS * PACKED_DIM,
        dtype=tla.Float4E2M1,
    )
    q_row0 = tla.make_tensor(
        q_row_ptr,
        tla.make_layout(
            tla.make_shape(Q_HEADS, HEAD_DIM),
            tla.make_stride(HEAD_DIM, 1),
            origin_shape=tla.make_shape(Q_HEADS, HEAD_DIM),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    q_row1_ptr = tla.recast_ptr(
        tla.recast_ptr(q_pack.ptr, dtype=tla.UInt8)
        + (pair_base + 1) * Q_HEADS * PACKED_DIM,
        dtype=tla.Float4E2M1,
    )
    q_row1 = tla.make_tensor(
        q_row1_ptr,
        tla.make_layout(
            tla.make_shape(Q_HEADS, HEAD_DIM),
            tla.make_stride(HEAD_DIM, 1),
            origin_shape=tla.make_shape(Q_HEADS, HEAD_DIM),
            layoutTag=tla.arch.RowMajor,
        ),
    )

    with tla.cube():
        for task in tla.range(tla.arch.block_idx(), total_tasks, tla.arch.block_num()):
            local_row = task // tiles_per_row
            nt = task % tiles_per_row
            if local_row == zero:
                gm_a = tla.tile_view(
                    q_row0,
                    tla.make_shape(BM, BK),
                    tla.make_coord(zero, zero),
                )
                valid_tokens = block_count0[zero] * TOKEN_BLOCK
                if nt * BN < valid_tokens:
                    gm_b = tla.tile_view(
                        k_pack0,
                        tla.make_shape(BK, BN),
                        tla.make_coord(zero, nt),
                    )
                    gm_sa = tla.tile_view(
                        q_scale0,
                        tla.make_shape(BM, SCALE_GROUPS),
                        tla.make_coord(zero, zero),
                    )
                    gm_sb = tla.tile_view(
                        k_scale0,
                        tla.make_shape(SCALE_GROUPS, BN),
                        tla.make_coord(zero, nt),
                    )
                    gm_c = tla.tile_view(
                        qk_out0,
                        tla.make_shape(BM, BN),
                        tla.make_coord(zero, nt),
                    )
                    t_l1a = tla.make_tensor(
                        l1a,
                        tla.make_layout_with_tag((BM, BK), dtype_a, tla.arch.zN),
                    )
                    t_l1b = tla.make_tensor(
                        l1b,
                        tla.make_layout_with_tag((BK, BN), dtype_b, tla.arch.nZ),
                    )
                    t_l1sa = tla.make_tensor_like(l1sa, gm_sa, tla.arch.zZMxScale)
                    t_l1sb = tla.make_tensor_like(l1sb, gm_sb, tla.arch.nNMxScale)
                    t_l0a = tla.make_tensor_like(l0a, t_l1a, tla.arch.zN)
                    t_l0b = tla.make_tensor_like(l0b, t_l1b, tla.arch.nZ)
                    t_l0c = tla.make_tensor_like(l0c, gm_c)
                    mutex_l1.lock(pipe=tla.pipes.MTE2)
                    tla.copy(t_l1a, gm_a)
                    tla.copy(t_l1b, gm_b)
                    tla.copy(t_l1sa, gm_sa)
                    tla.copy(t_l1sb, gm_sb)
                    mutex_l1.unlock(pipe=tla.pipes.MTE2)
                    mutex_l1.lock(pipe=tla.pipes.MTE1)
                    mutex_l0.lock(pipe=tla.pipes.MTE1)
                    tla.copy(t_l0a, t_l1a, scale=t_l1sa)
                    tla.copy(t_l0b, t_l1b, scale=t_l1sb)
                    mutex_l0.unlock(pipe=tla.pipes.MTE1)
                    mutex_l1.unlock(pipe=tla.pipes.MTE1)
                    mutex_l0.lock(pipe=tla.pipes.CUBE)
                    mutex_l0c.lock(pipe=tla.pipes.CUBE)
                    tla.mmad_mx(t_l0c, t_l0a, t_l0b, init_c=True, unit_flag=0b11)
                    mutex_l0c.unlock(pipe=tla.pipes.CUBE)
                    mutex_l0.unlock(pipe=tla.pipes.CUBE)
                    mutex_l0c.lock(pipe=tla.pipes.FIX)
                    tla.copy(
                        gm_c,
                        t_l0c,
                        tla.params.CopyL0C2DstParams(unit_flag=0b11),
                    )
                    mutex_l0c.unlock(pipe=tla.pipes.FIX)
            else:
                gm_a = tla.tile_view(
                    q_row1,
                    tla.make_shape(BM, BK),
                    tla.make_coord(zero, zero),
                )
                valid_tokens = block_count1[zero] * TOKEN_BLOCK
                if nt * BN < valid_tokens:
                    gm_b = tla.tile_view(
                        k_pack1,
                        tla.make_shape(BK, BN),
                        tla.make_coord(zero, nt),
                    )
                    gm_sa = tla.tile_view(
                        q_scale1,
                        tla.make_shape(BM, SCALE_GROUPS),
                        tla.make_coord(zero, zero),
                    )
                    gm_sb = tla.tile_view(
                        k_scale1,
                        tla.make_shape(SCALE_GROUPS, BN),
                        tla.make_coord(zero, nt),
                    )
                    gm_c = tla.tile_view(
                        qk_out1,
                        tla.make_shape(BM, BN),
                        tla.make_coord(zero, nt),
                    )
                    t_l1a = tla.make_tensor(
                        l1a,
                        tla.make_layout_with_tag((BM, BK), dtype_a, tla.arch.zN),
                    )
                    t_l1b = tla.make_tensor(
                        l1b,
                        tla.make_layout_with_tag((BK, BN), dtype_b, tla.arch.nZ),
                    )
                    t_l1sa = tla.make_tensor_like(l1sa, gm_sa, tla.arch.zZMxScale)
                    t_l1sb = tla.make_tensor_like(l1sb, gm_sb, tla.arch.nNMxScale)
                    t_l0a = tla.make_tensor_like(l0a, t_l1a, tla.arch.zN)
                    t_l0b = tla.make_tensor_like(l0b, t_l1b, tla.arch.nZ)
                    t_l0c = tla.make_tensor_like(l0c, gm_c)
                    mutex_l1.lock(pipe=tla.pipes.MTE2)
                    tla.copy(t_l1a, gm_a)
                    tla.copy(t_l1b, gm_b)
                    tla.copy(t_l1sa, gm_sa)
                    tla.copy(t_l1sb, gm_sb)
                    mutex_l1.unlock(pipe=tla.pipes.MTE2)
                    mutex_l1.lock(pipe=tla.pipes.MTE1)
                    mutex_l0.lock(pipe=tla.pipes.MTE1)
                    tla.copy(t_l0a, t_l1a, scale=t_l1sa)
                    tla.copy(t_l0b, t_l1b, scale=t_l1sb)
                    mutex_l0.unlock(pipe=tla.pipes.MTE1)
                    mutex_l1.unlock(pipe=tla.pipes.MTE1)
                    mutex_l0.lock(pipe=tla.pipes.CUBE)
                    mutex_l0c.lock(pipe=tla.pipes.CUBE)
                    tla.mmad_mx(t_l0c, t_l0a, t_l0b, init_c=True, unit_flag=0b11)
                    mutex_l0c.unlock(pipe=tla.pipes.CUBE)
                    mutex_l0.unlock(pipe=tla.pipes.CUBE)
                    mutex_l0c.lock(pipe=tla.pipes.FIX)
                    tla.copy(
                        gm_c,
                        t_l0c,
                        tla.params.CopyL0C2DstParams(unit_flag=0b11),
                    )
                    mutex_l0c.unlock(pipe=tla.pipes.FIX)


@tla.kernel
def qk2_stream_kernel(
    q_pack: tla.Tensor,
    q_scale0: tla.Tensor,
    k_pack0: tla.Tensor,
    k_scale0: tla.Tensor,
    block_count0: tla.Tensor,
    qk_out0: tla.Tensor,
    q_scale1: tla.Tensor,
    k_pack1: tla.Tensor,
    k_scale1: tla.Tensor,
    block_count1: tla.Tensor,
    qk_out1: tla.Tensor,
    tokens_per_row: tla.Constexpr[int],
    pair_base: tla.Constexpr[int],
    shards_per_row: tla.Constexpr[int],
) -> None:
    """Two-row QK streaming kernel with Q retained in L0A per CTA."""
    zero = 0
    tiles_per_row = tokens_per_row // BN
    total_tasks = 2 * shards_per_row
    dtype_a = q_pack.ptr.dtype
    dtype_b = k_pack0.ptr.dtype
    mutex_l1 = tla.mutex(resource="qk2s_l1", id=0)
    mutex_l0 = tla.mutex(resource="qk2s_l0", id=1)
    mutex_l0c = tla.mutex(resource="qk2s_l0c", id=2)

    l1a = tla.allocate(BM * BK, dtype_a, tla.AddressSpace.l1, 512)
    l1b = tla.allocate(BK * BN, dtype_b, tla.AddressSpace.l1, 512)
    l1sa = tla.allocate(BM * SCALE_GROUPS, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb = tla.allocate(SCALE_GROUPS * BN, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(BM * BK, dtype_a, tla.AddressSpace.l0a, 1024)
    l0b = tla.allocate(BK * BN, dtype_b, tla.AddressSpace.l0b, 1024)
    l0c = tla.allocate(BM * BN, tla.Float32, tla.AddressSpace.l0c, 1024)
    q_row_ptr = tla.recast_ptr(
        tla.recast_ptr(q_pack.ptr, dtype=tla.UInt8) + pair_base * Q_HEADS * PACKED_DIM,
        dtype=tla.Float4E2M1,
    )
    q_row0 = tla.make_tensor(
        q_row_ptr,
        tla.make_layout(
            tla.make_shape(Q_HEADS, HEAD_DIM),
            tla.make_stride(HEAD_DIM, 1),
            origin_shape=tla.make_shape(Q_HEADS, HEAD_DIM),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    q_row1_ptr = tla.recast_ptr(
        tla.recast_ptr(q_pack.ptr, dtype=tla.UInt8)
        + (pair_base + 1) * Q_HEADS * PACKED_DIM,
        dtype=tla.Float4E2M1,
    )
    q_row1 = tla.make_tensor(
        q_row1_ptr,
        tla.make_layout(
            tla.make_shape(Q_HEADS, HEAD_DIM),
            tla.make_stride(HEAD_DIM, 1),
            origin_shape=tla.make_shape(Q_HEADS, HEAD_DIM),
            layoutTag=tla.arch.RowMajor,
        ),
    )

    with tla.cube():
        for task in tla.range(tla.arch.block_idx(), total_tasks, tla.arch.block_num()):
            local_row = task // shards_per_row
            shard = task % shards_per_row
            if local_row == zero:
                gm_a = tla.tile_view(
                    q_row0,
                    tla.make_shape(BM, BK),
                    tla.make_coord(zero, zero),
                )
                gm_sa = tla.tile_view(
                    q_scale0,
                    tla.make_shape(BM, SCALE_GROUPS),
                    tla.make_coord(zero, zero),
                )
                t_l1a = tla.make_tensor(
                    l1a,
                    tla.make_layout_with_tag((BM, BK), dtype_a, tla.arch.zN),
                )
                t_l1sa = tla.make_tensor_like(l1sa, gm_sa, tla.arch.zZMxScale)
                t_l0a = tla.make_tensor_like(l0a, t_l1a, tla.arch.zN)
                mutex_l1.lock(pipe=tla.arch.MTE2)
                tla.copy(t_l1a, gm_a)
                tla.copy(t_l1sa, gm_sa)
                mutex_l1.unlock(pipe=tla.arch.MTE2)
                mutex_l1.lock(pipe=tla.arch.MTE1)
                mutex_l0.lock(pipe=tla.arch.MTE1)
                tla.copy(t_l0a, t_l1a, scale=t_l1sa)
                mutex_l0.unlock(pipe=tla.arch.MTE1)
                mutex_l1.unlock(pipe=tla.arch.MTE1)
                valid_tokens = block_count0[zero] * TOKEN_BLOCK
                for nt in tla.range(shard, tiles_per_row, shards_per_row):
                    if nt * BN < valid_tokens:
                        gm_b = tla.tile_view(
                            k_pack0,
                            tla.make_shape(BK, BN),
                            tla.make_coord(zero, nt),
                        )
                        gm_sb = tla.tile_view(
                            k_scale0,
                            tla.make_shape(SCALE_GROUPS, BN),
                            tla.make_coord(zero, nt),
                        )
                        gm_c = tla.tile_view(
                            qk_out0,
                            tla.make_shape(BM, BN),
                            tla.make_coord(zero, nt),
                        )
                        t_l1b = tla.make_tensor(
                            l1b,
                            tla.make_layout_with_tag((BK, BN), dtype_b, tla.arch.nZ),
                        )
                        t_l1sb = tla.make_tensor_like(l1sb, gm_sb, tla.arch.nNMxScale)
                        t_l0b = tla.make_tensor_like(l0b, t_l1b, tla.arch.nZ)
                        t_l0c = tla.make_tensor_like(l0c, gm_c)
                        mutex_l1.lock(pipe=tla.arch.MTE2)
                        tla.copy(t_l1b, gm_b)
                        tla.copy(t_l1sb, gm_sb)
                        mutex_l1.unlock(pipe=tla.arch.MTE2)
                        mutex_l1.lock(pipe=tla.arch.MTE1)
                        mutex_l0.lock(pipe=tla.arch.MTE1)
                        tla.copy(t_l0b, t_l1b, scale=t_l1sb)
                        mutex_l0.unlock(pipe=tla.arch.MTE1)
                        mutex_l1.unlock(pipe=tla.arch.MTE1)
                        mutex_l0.lock(pipe=tla.arch.CUBE)
                        mutex_l0c.lock(pipe=tla.arch.CUBE)
                        tla.mmad_mx(t_l0c, t_l0a, t_l0b, init_c=True, unit_flag=0b11)
                        mutex_l0c.unlock(pipe=tla.arch.CUBE)
                        mutex_l0.unlock(pipe=tla.arch.CUBE)
                        mutex_l0c.lock(pipe=tla.arch.FIX)
                        tla.copy(
                            gm_c,
                            t_l0c,
                            tla.params.CopyL0C2DstParams(unit_flag=0b11),
                        )
                        mutex_l0c.unlock(pipe=tla.arch.FIX)
            else:
                gm_a = tla.tile_view(
                    q_row1,
                    tla.make_shape(BM, BK),
                    tla.make_coord(zero, zero),
                )
                gm_sa = tla.tile_view(
                    q_scale1,
                    tla.make_shape(BM, SCALE_GROUPS),
                    tla.make_coord(zero, zero),
                )
                t_l1a = tla.make_tensor(
                    l1a,
                    tla.make_layout_with_tag((BM, BK), dtype_a, tla.arch.zN),
                )
                t_l1sa = tla.make_tensor_like(l1sa, gm_sa, tla.arch.zZMxScale)
                t_l0a = tla.make_tensor_like(l0a, t_l1a, tla.arch.zN)
                mutex_l1.lock(pipe=tla.arch.MTE2)
                tla.copy(t_l1a, gm_a)
                tla.copy(t_l1sa, gm_sa)
                mutex_l1.unlock(pipe=tla.arch.MTE2)
                mutex_l1.lock(pipe=tla.arch.MTE1)
                mutex_l0.lock(pipe=tla.arch.MTE1)
                tla.copy(t_l0a, t_l1a, scale=t_l1sa)
                mutex_l0.unlock(pipe=tla.arch.MTE1)
                mutex_l1.unlock(pipe=tla.arch.MTE1)
                valid_tokens = block_count1[zero] * TOKEN_BLOCK
                for nt in tla.range(shard, tiles_per_row, shards_per_row):
                    if nt * BN < valid_tokens:
                        gm_b = tla.tile_view(
                            k_pack1,
                            tla.make_shape(BK, BN),
                            tla.make_coord(zero, nt),
                        )
                        gm_sb = tla.tile_view(
                            k_scale1,
                            tla.make_shape(SCALE_GROUPS, BN),
                            tla.make_coord(zero, nt),
                        )
                        gm_c = tla.tile_view(
                            qk_out1,
                            tla.make_shape(BM, BN),
                            tla.make_coord(zero, nt),
                        )
                        t_l1b = tla.make_tensor(
                            l1b,
                            tla.make_layout_with_tag((BK, BN), dtype_b, tla.arch.nZ),
                        )
                        t_l1sb = tla.make_tensor_like(l1sb, gm_sb, tla.arch.nNMxScale)
                        t_l0b = tla.make_tensor_like(l0b, t_l1b, tla.arch.nZ)
                        t_l0c = tla.make_tensor_like(l0c, gm_c)
                        mutex_l1.lock(pipe=tla.arch.MTE2)
                        tla.copy(t_l1b, gm_b)
                        tla.copy(t_l1sb, gm_sb)
                        mutex_l1.unlock(pipe=tla.arch.MTE2)
                        mutex_l1.lock(pipe=tla.arch.MTE1)
                        mutex_l0.lock(pipe=tla.arch.MTE1)
                        tla.copy(t_l0b, t_l1b, scale=t_l1sb)
                        mutex_l0.unlock(pipe=tla.arch.MTE1)
                        mutex_l1.unlock(pipe=tla.arch.MTE1)
                        mutex_l0.lock(pipe=tla.arch.CUBE)
                        mutex_l0c.lock(pipe=tla.arch.CUBE)
                        tla.mmad_mx(t_l0c, t_l0a, t_l0b, init_c=True, unit_flag=0b11)
                        mutex_l0c.unlock(pipe=tla.arch.CUBE)
                        mutex_l0.unlock(pipe=tla.arch.CUBE)
                        mutex_l0c.lock(pipe=tla.arch.FIX)
                        tla.copy(
                            gm_c,
                            t_l0c,
                            tla.params.CopyL0C2DstParams(unit_flag=0b11),
                        )
                        mutex_l0c.unlock(pipe=tla.arch.FIX)


@tla.kernel
def reduce_topk_kernel(
    qk: tla.Tensor,
    weights: tla.Tensor,
    block_ids: tla.Tensor,
    token_count: tla.Tensor,
    output_offset: tla.Tensor,
    indices_out: tla.Tensor,
    values_out: tla.Tensor,
    rows: tla.Constexpr[int],
    max_tokens: tla.Constexpr[int],
    sort_size: tla.Constexpr[int],
    log_sort: tla.Constexpr[int],
    topk: tla.Constexpr[int],
    has_offset: tla.Constexpr[bool],
    write_values: tla.Constexpr[bool],
) -> None:
    """32-head sequential BF16 reduction followed by deterministic TopK."""
    zero = 0
    one = 1
    scores = tla.make_tensor(
        tla.allocate(sort_size, tla.BFloat16, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(sort_size), tla.make_stride(1)),
    )
    sort_tokens = tla.make_tensor(
        tla.allocate(sort_size, tla.Int32, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(sort_size), tla.make_stride(1)),
    )
    lanes = 256

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=lanes):
            tid, _, _ = tla.arch.thread_idx()
            thread_count, _, _ = tla.arch.thread_block_dim()
            row = tla.arch.block_idx()
            total = token_count[row]
            for score_token in tla.range(tid, total, thread_count):
                acc = tla.Float32(0.0)
                for head in tla.range_constexpr(0, Q_HEADS, 1):
                    qk_bf16 = qk[row * Q_HEADS + head, score_token].to(tla.BFloat16)
                    qk_f32 = qk_bf16.to(tla.Float32)
                    weight_bf16 = weights[row, head].to(tla.BFloat16)
                    weight_f32 = weight_bf16.to(tla.Float32)
                    relu = qk_f32
                    if relu < tla.Float32(0.0):
                        relu = tla.Float32(0.0)
                    summed = acc + relu * weight_f32
                    acc = summed.to(tla.BFloat16).to(tla.Float32)
                scores[score_token] = acc.to(tla.BFloat16)
                sort_tokens[score_token] = (
                    block_ids[row, score_token // TOKEN_BLOCK] * TOKEN_BLOCK
                    + score_token % TOKEN_BLOCK
                )

            neg_inf = tla.Float32(float("-inf"))
            bf16_neg_inf = neg_inf.to(tla.BFloat16)
            for pad_token in tla.range(total + tid, sort_size, thread_count):
                scores[pad_token] = bf16_neg_inf
                sort_tokens[pad_token] = tla.Int32(2147483647)
            tla.arch.sync_threads()

            for level in tla.range_constexpr(1, log_sort + 1, 1):
                stage_size = 1 << level
                for step in tla.range_constexpr(0, level, 1):
                    distance = 1 << (level - step - 1)
                    for pos in tla.range(tid, sort_size, thread_count):
                        bit = (pos // distance) % 2
                        partner = pos + distance
                        if bit != 0:
                            partner = pos - distance
                        if partner > pos:
                            score_a = scores[pos].to(tla.Float32)
                            score_b = scores[partner].to(tla.Float32)
                            token_a = sort_tokens[pos]
                            token_b = sort_tokens[partner]
                            a_better = zero
                            b_better = zero
                            if score_a > score_b:
                                a_better = one
                            if score_b > score_a:
                                b_better = one
                            if score_a == score_b:
                                if token_a < token_b:
                                    a_better = one
                                if token_b < token_a:
                                    b_better = one
                            ascending = (pos // stage_size) % 2 == 0
                            swap_pair = zero
                            if ascending:
                                if a_better == zero and b_better != zero:
                                    swap_pair = one
                            else:
                                if a_better != zero and b_better == zero:
                                    swap_pair = one
                            if swap_pair != zero:
                                scores[pos] = score_b.to(tla.BFloat16)
                                scores[partner] = score_a.to(tla.BFloat16)
                                sort_tokens[pos] = token_b
                                sort_tokens[partner] = token_a
                    tla.arch.sync_threads()

            row_offset = zero
            if has_offset:
                row_offset = output_offset[row]
            for rank in tla.range(tid, topk, thread_count):
                indices_out[row * topk + rank] = tla.Int32(-1)
                if write_values:
                    values_out[row * topk + rank] = bf16_neg_inf
                if rank < total:
                    indices_out[row * topk + rank] = sort_tokens[rank] + row_offset
                    if write_values:
                        values_out[row * topk + rank] = scores[rank]

        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def reduce_chunk_topk_kernel(
    qk: tla.Tensor,
    weights: tla.Tensor,
    block_ids: tla.Tensor,
    token_count: tla.Tensor,
    chunk_keys: tla.Tensor,
    rows: tla.Constexpr[int],
    max_tokens: tla.Constexpr[int],
    sort_size: tla.Constexpr[int],
    chunk_size: tla.Constexpr[int],
    log_chunk: tla.Constexpr[int],
    chunk_count: tla.Constexpr[int],
) -> None:
    """Reduce and bitonically sort independent row/token chunks in parallel."""
    zero = 0
    one = 1
    lanes = 256
    sort_keys = tla.make_tensor(
        tla.allocate(chunk_size, tla.Int32, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(chunk_size), tla.make_stride(1)),
    )
    total_tasks = rows * chunk_count
    bf16_neg_inf = tla.Float32(float("-inf")).to(tla.BFloat16)
    neg_inf_bits = bf16_neg_inf.bitcast(tla.Int16)
    neg_inf_order = tla.Int32(-32768) - neg_inf_bits.to(tla.Int32)
    padding_key = neg_inf_order << 16

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=lanes):
            tid, _, _ = tla.arch.thread_idx()
            thread_count, _, _ = tla.arch.thread_block_dim()
            for task in tla.range(
                tla.arch.block_idx(), total_tasks, tla.arch.block_num()
            ):
                row = task // chunk_count
                chunk = task % chunk_count
                chunk_start = chunk * chunk_size
                total = token_count[row]
                for local_token in tla.range(tid, chunk_size, thread_count):
                    token = chunk_start + local_token
                    if token < total:
                        acc = tla.Float32(0.0)
                        for head in tla.range_constexpr(0, Q_HEADS, 1):
                            qk_bf16 = qk[row * Q_HEADS + head, token].to(tla.BFloat16)
                            qk_f32 = qk_bf16.to(tla.Float32)
                            weight_bf16 = weights[row, head].to(tla.BFloat16)
                            weight_f32 = weight_bf16.to(tla.Float32)
                            relu = qk_f32
                            if relu < tla.Float32(0.0):
                                relu = tla.Float32(0.0)
                            acc = (
                                (acc + relu * weight_f32)
                                .to(tla.BFloat16)
                                .to(tla.Float32)
                            )
                        reduction_bits = acc.to(tla.BFloat16).bitcast(tla.Int16)
                        reduction_order = reduction_bits.to(tla.Int32)
                        if reduction_bits < tla.Int16(0):
                            reduction_order = tla.Int32(-32768) - reduction_order
                        reduction_token = (
                            block_ids[row, token // TOKEN_BLOCK] * TOKEN_BLOCK
                            + token % TOKEN_BLOCK
                        )
                        sort_keys[local_token] = (reduction_order << 16) | (
                            tla.Int32(65535) - reduction_token
                        )
                    else:
                        sort_keys[local_token] = padding_key
                tla.arch.sync_threads()

                if chunk_start < total:
                    for level in tla.range_constexpr(1, log_chunk + 1, 1):
                        stage_size = 1 << level
                        for step in tla.range_constexpr(0, level, 1):
                            distance = 1 << (level - step - 1)
                            for sort_pos in tla.range(tid, chunk_size, thread_count):
                                bit = (sort_pos // distance) % 2
                                partner = sort_pos + distance
                                if bit != 0:
                                    partner = sort_pos - distance
                                if partner > sort_pos:
                                    ascending = (sort_pos // stage_size) % 2 == 0
                                    key_a = sort_keys[sort_pos]
                                    key_b = sort_keys[partner]
                                    swap_pair = zero
                                    if ascending:
                                        if key_a > key_b:
                                            swap_pair = one
                                    else:
                                        if key_a < key_b:
                                            swap_pair = one
                                    if swap_pair != zero:
                                        sort_keys[sort_pos] = key_b
                                        sort_keys[partner] = key_a
                            tla.arch.sync_threads()

                chunk_base = row * sort_size + chunk * chunk_size
                for store_pos in tla.range(tid, chunk_size, thread_count):
                    chunk_keys[chunk_base + store_pos] = sort_keys[store_pos]
                tla.arch.sync_threads()

        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def merge_topk_kernel(
    chunk_keys: tla.Tensor,
    output_offset: tla.Tensor,
    indices_out: tla.Tensor,
    values_out: tla.Tensor,
    rows: tla.Constexpr[int],
    sort_size: tla.Constexpr[int],
    chunk_size: tla.Constexpr[int],
    log_chunk: tla.Constexpr[int],
    chunk_count: tla.Constexpr[int],
    log_chunk_count: tla.Constexpr[int],
    pairs_round_0: tla.Constexpr[int],
    pairs_round_1: tla.Constexpr[int],
    pairs_round_2: tla.Constexpr[int],
    pairs_round_3: tla.Constexpr[int],
    pairs_round_4: tla.Constexpr[int],
    topk: tla.Constexpr[int],
    has_offset: tla.Constexpr[bool],
    write_values: tla.Constexpr[bool],
) -> None:
    """Serial top-K-preserving merge selected for launch-bound shapes."""
    zero = 0
    one = 1
    lanes = 256
    merge_width = chunk_size * 2
    bf16_neg_inf = tla.Float32(float("-inf")).to(tla.BFloat16)
    keys = tla.make_tensor(
        tla.allocate(sort_size, tla.Int32, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(sort_size), tla.make_stride(1)),
    )
    work_keys = tla.make_tensor(
        tla.allocate(merge_width, tla.Int32, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(merge_width), tla.make_stride(1)),
    )
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=lanes):
            tid, _, _ = tla.arch.thread_idx()
            thread_count, _, _ = tla.arch.thread_block_dim()
            row = tla.arch.block_idx()
            result_base = row * sort_size
            for load_chunk_pos in tla.range(tid, sort_size, thread_count):
                keys[load_chunk_pos] = chunk_keys[result_base + load_chunk_pos]
            tla.arch.sync_threads()

            for merge_round in range(0, 5, 1):
                pair_count = pairs_round_0
                if merge_round == 1:
                    pair_count = pairs_round_1
                if merge_round == 2:
                    pair_count = pairs_round_2
                if merge_round == 3:
                    pair_count = pairs_round_3
                if merge_round == 4:
                    pair_count = pairs_round_4
                for pair in tla.range(0, pair_count, 1):
                    left_array = pair * 2
                    right_array = left_array + 1
                    left_base = left_array * chunk_size
                    right_base = right_array * chunk_size
                    for load_pos in tla.range(tid, chunk_size, thread_count):
                        work_keys[load_pos] = keys[left_base + load_pos]
                        reverse_pos = chunk_size - one - load_pos
                        work_keys[chunk_size + load_pos] = keys[
                            right_base + reverse_pos
                        ]
                    tla.arch.sync_threads()

                    for step in tla.range_constexpr(0, log_chunk + 1, 1):
                        distance = chunk_size >> step
                        for merge_pos in tla.range(tid, chunk_size, thread_count):
                            partner = merge_pos + distance
                            key_a = work_keys[merge_pos]
                            key_b = work_keys[partner]
                            if key_a < key_b:
                                work_keys[merge_pos] = key_b
                                work_keys[partner] = key_a
                        tla.arch.sync_threads()

                    output_base = pair * chunk_size
                    for merge_store_pos in tla.range(tid, chunk_size, thread_count):
                        keys[output_base + merge_store_pos] = work_keys[merge_store_pos]
                    tla.arch.sync_threads()

            row_offset = zero
            if has_offset:
                row_offset = output_offset[row]
            for rank in tla.range(tid, topk, thread_count):
                indices_out[row * topk + rank] = tla.Int32(-1)
                if write_values:
                    values_out[row * topk + rank] = bf16_neg_inf
                key = keys[rank]
                token = tla.Int32(65535) - (key & tla.Int32(65535))
                score_order = key >> 16
                score_bits = score_order
                if score_order < 0:
                    score_bits = tla.Int32(-32768) - score_order
                if token != tla.Int32(65535):
                    indices_out[row * topk + rank] = token + row_offset
                    if write_values:
                        values_out[row * topk + rank] = tla.Int16(score_bits).bitcast(
                            tla.BFloat16
                        )

        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def merge_stage_topk_kernel(
    input_keys: tla.Tensor,
    output_keys: tla.Tensor,
    output_offset: tla.Tensor,
    indices_out: tla.Tensor,
    values_out: tla.Tensor,
    rows: tla.Constexpr[int],
    chunk_size: tla.Constexpr[int],
    output_chunk_size: tla.Constexpr[int],
    log_chunk: tla.Constexpr[int],
    source_chunks: tla.Constexpr[int],
    topk: tla.Constexpr[int],
    has_offset: tla.Constexpr[bool],
    write_values: tla.Constexpr[bool],
    final_stage: tla.Constexpr[bool],
) -> None:
    """Merge one tree level and retain its exact requested top capacity."""
    zero = 0
    one = 1
    lanes = 256
    pair_count = source_chunks // 2
    total_tasks = rows * pair_count
    merge_width = chunk_size * 2
    bf16_neg_inf = tla.Float32(float("-inf")).to(tla.BFloat16)
    keys = tla.make_tensor(
        tla.allocate(output_chunk_size, tla.Int32, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(output_chunk_size), tla.make_stride(1)),
    )
    work_keys = tla.make_tensor(
        tla.allocate(merge_width, tla.Int32, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(merge_width), tla.make_stride(1)),
    )

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=lanes):
            tid, _, _ = tla.arch.thread_idx()
            thread_count, _, _ = tla.arch.thread_block_dim()
            for task in tla.range(
                tla.arch.block_idx(), total_tasks, tla.arch.block_num()
            ):
                row = task // pair_count
                pair = task % pair_count
                left_base = row * source_chunks * chunk_size + pair * merge_width
                right_base = left_base + chunk_size
                for load_pos in tla.range(tid, chunk_size, thread_count):
                    work_keys[load_pos] = input_keys[left_base + load_pos]
                    reverse_pos = chunk_size - one - load_pos
                    work_keys[chunk_size + load_pos] = input_keys[
                        right_base + reverse_pos
                    ]
                tla.arch.sync_threads()

                for step in tla.range_constexpr(0, log_chunk + 1, 1):
                    distance = chunk_size >> step
                    for merge_pos in tla.range(tid, chunk_size, thread_count):
                        partner = merge_pos + distance
                        key_a = work_keys[merge_pos]
                        key_b = work_keys[partner]
                        if key_a < key_b:
                            work_keys[merge_pos] = key_b
                            work_keys[partner] = key_a
                    tla.arch.sync_threads()

                output_base = (
                    row * pair_count * output_chunk_size + pair * output_chunk_size
                )
                for store_pos in tla.range(tid, output_chunk_size, thread_count):
                    keys[store_pos] = work_keys[store_pos]
                    output_keys[output_base + store_pos] = work_keys[store_pos]
                tla.arch.sync_threads()

                if final_stage:
                    row_offset = zero
                    if has_offset:
                        row_offset = output_offset[row]
                    for rank in tla.range(tid, topk, thread_count):
                        indices_out[row * topk + rank] = tla.Int32(-1)
                        if write_values:
                            values_out[row * topk + rank] = bf16_neg_inf
                        key = keys[rank]
                        token = tla.Int32(65535) - (key & tla.Int32(65535))
                        score_order = key >> 16
                        score_bits = score_order
                        if score_order < 0:
                            score_bits = tla.Int32(-32768) - score_order
                        if token != tla.Int32(65535):
                            indices_out[row * topk + rank] = token + row_offset
                            if write_values:
                                values_out[row * topk + rank] = tla.Int16(
                                    score_bits
                                ).bitcast(tla.BFloat16)

        tla.pipe_barrier(tla.pipes.ALL)


_PREP_CACHE = {}
_QK_CACHE = {}
_TOPK_CACHE = {}
_CHUNK_CACHE = {}
_WORKSPACE_CACHE = {}
_BINDING_CACHE = {}


def _device_index(device):
    text = str(device)
    return int(text.split(":")[-1]) if ":" in text else 0


def _tensor(x, tag, origin_shape=None, element_type=None):
    binding_key = (
        tuple(x.shape),
        tuple(x.stride()),
        str(x.dtype),
        str(tag),
        origin_shape,
        str(element_type),
    )
    template = _BINDING_CACHE.get(binding_key)
    if template is not None:
        # Reuse layout metadata; only storage pointer and producer lifetime vary.
        bound = copy.copy(template)
        bound.data_ptr = int(x.data_ptr())
        bound._dlpack_source = x
        bound._external_binding = True
        return bound
    kwargs = {"layout_tag": tag}
    if origin_shape is not None:
        kwargs["origin_shape"] = origin_shape
    if element_type is not None:
        kwargs["element_type"] = element_type
    template = from_dlpack(x, **kwargs)
    _BINDING_CACHE[binding_key] = template
    return template


def _core_counts(device_index):
    props = torch.npu.get_device_properties(device_index)
    cube = int(getattr(props, "cube_core_num", 28) or 28)
    vector = int(getattr(props, "vector_core_num", 56) or 56)
    return cube, vector


def run(
    q,
    k,
    w,
    descale_q,
    descale_k,
    candidate_block_indices,
    candidate_block_length,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    cmp_residual_k,
    block_table,
    output_idx_offset,
    topk,
    candidate_block_size,
    mask_mode,
    cmp_ratio,
    return_value,
):
    """Return ``(topk_indices, topk_values)`` on the input NPU device."""
    if int(candidate_block_size) != TOKEN_BLOCK:
        raise ValueError("candidate_block_size must be 8")
    rows = int(q.shape[0])
    topk = int(topk)
    mask_mode = int(mask_mode)
    cmp_ratio = 1 if mask_mode == 0 else int(cmp_ratio)
    is_paged = block_table is not None
    has_residual = cmp_residual_k is not None
    has_offset = output_idx_offset is not None
    write_values = bool(return_value)
    device = q.device
    device_index = _device_index(device)
    cube_cores, vector_cores = _core_counts(device_index)

    if is_paged:
        page_size = int(k.shape[1]) * TOKEN_BLOCK
        records_per_page = int(k.shape[1])
        k_capacity = int(k.shape[0]) * page_size
    else:
        page_size = TOKEN_BLOCK
        records_per_page = 1
        k_capacity = int(k.shape[0])
    block_capacity = min(CANDIDATE_CAPACITY, max(1, k_capacity // TOKEN_BLOCK))
    max_tokens = min(CANDIDATE_CAPACITY * TOKEN_BLOCK, max(k_capacity, BN))
    # Keep the per-row QK/output stride 1 KiB aligned for Fixpipe.
    max_tokens = ((max_tokens + 255) // 256) * 256
    flat_tokens = rows * max_tokens
    available_splits = max(1, vector_cores // rows)
    if max_tokens >= 2048:
        pack_splits = min(32, available_splits)
    elif max_tokens >= 512:
        pack_splits = min(8, available_splits)
    else:
        pack_splits = 1

    workspace_key = (rows, block_capacity, max_tokens, device_index)
    workspace = _WORKSPACE_CACHE.get(workspace_key)
    if workspace is None:
        # Shape-stable scratch storage is reused to avoid allocator churn and
        # keeps cached descriptors/artifacts valid across calls.
        workspace = {
            "token_k": torch.zeros(
                (flat_tokens, PACKED_DIM), dtype=torch.uint8, device=device
            ),
            "token_scale": torch.zeros(
                (flat_tokens, SCALE_GROUPS), dtype=torch.uint8, device=device
            ),
            "bitmap": torch.empty(
                (rows, block_capacity), dtype=torch.int8, device=device
            ),
            "block_ids": torch.empty(
                (rows, block_capacity), dtype=torch.int32, device=device
            ),
            "block_count": torch.empty((rows,), dtype=torch.int32, device=device),
            "token_count": torch.empty((rows,), dtype=torch.int32, device=device),
            "visible": torch.empty((rows,), dtype=torch.int32, device=device),
            "qk": torch.empty(
                (rows * Q_HEADS, max_tokens), dtype=torch.float32, device=device
            ),
            "dummy_i32": torch.zeros((1,), dtype=torch.int32, device=device),
            "dummy_bt": torch.zeros((1, 1), dtype=torch.int32, device=device),
            "dummy_off": torch.zeros((1, 1), dtype=torch.int32, device=device),
        }
        _WORKSPACE_CACHE[workspace_key] = workspace
    token_k = workspace["token_k"]
    token_scale = workspace["token_scale"]
    bitmap = workspace["bitmap"]
    block_ids = workspace["block_ids"]
    block_count = workspace["block_count"]
    token_count = workspace["token_count"]
    visible = workspace["visible"]
    dummy_i32 = workspace["dummy_i32"]
    dummy_bt = workspace["dummy_bt"]
    dummy_off = workspace["dummy_off"]
    qk = workspace["qk"]
    cached = workspace.get("static_descriptors")
    if cached is None:
        # Descriptors carry dynamic layout metadata; launches rebind pointers
        # without rebuilding capsule metadata.
        k_rows = token_k.reshape(rows, max_tokens, PACKED_DIM)
        ks_rows = token_scale.reshape(rows, max_tokens, SCALE_GROUPS)
        qk_rows = qk.reshape(rows, Q_HEADS, max_tokens)
        qk_static = []
        for row_index in range(rows):
            qk_static.append(
                (
                    _tensor(
                        k_rows[row_index],
                        tla.arch.ColumnMajor,
                        origin_shape=(HEAD_DIM, max_tokens),
                        element_type=tla.Float4E2M1,
                    ).mark_layout_dynamic(),
                    _tensor(
                        ks_rows[row_index],
                        tla.arch.ColMajorMxScaleB,
                        origin_shape=(SCALE_GROUPS, max_tokens),
                        element_type=tla.Float8E8M0,
                    ).mark_layout_dynamic(),
                    _tensor(
                        block_count[row_index : row_index + 1],
                        tla.arch.RowMajor,
                    ),
                    _tensor(qk_rows[row_index], tla.arch.RowMajor),
                )
            )
        cached = {
            "prep": [
                _tensor(bitmap, tla.arch.RowMajor),
                _tensor(block_ids, tla.arch.RowMajor),
                _tensor(block_count, tla.arch.RowMajor),
                _tensor(token_count, tla.arch.RowMajor),
                _tensor(visible, tla.arch.RowMajor),
                _tensor(token_k, tla.arch.RowMajor),
                _tensor(token_scale, tla.arch.RowMajor),
            ],
            "qk": qk_static,
            "topk": (
                _tensor(qk, tla.arch.RowMajor),
                _tensor(block_ids, tla.arch.RowMajor),
                _tensor(token_count, tla.arch.RowMajor),
            ),
            "dummy_off": _tensor(dummy_off, tla.arch.RowMajor),
        }
        workspace["static_descriptors"] = cached
    descale_k_flat = (
        descale_k.reshape(-1)
        if descale_k is not None
        else torch.zeros((4,), dtype=torch.uint8, device=device)
    )
    k_flat = k.reshape(-1)

    prep_key = (
        rows,
        pack_splits,
        block_capacity,
        max_tokens,
        is_paged,
        has_residual,
        mask_mode,
        cmp_ratio,
        page_size,
        records_per_page,
        device_index,
    )
    cu_q_descriptor = _tensor(cu_seqlens_q.reshape(-1), tla.arch.RowMajor)
    cu_k_descriptor = _tensor(
        (cu_seqlens_k if cu_seqlens_k is not None else dummy_i32).reshape(-1),
        tla.arch.RowMajor,
    )
    block_table_descriptor = _tensor(
        block_table if block_table is not None else dummy_bt,
        tla.arch.RowMajor,
    )
    metadata_tensors = [
        _tensor(candidate_block_indices.reshape(rows, -1), tla.arch.RowMajor),
        _tensor(candidate_block_length, tla.arch.RowMajor),
        cu_q_descriptor,
        _tensor(seqused_q.reshape(-1), tla.arch.RowMajor),
        _tensor(seqused_k.reshape(-1), tla.arch.RowMajor),
        _tensor(
            (cmp_residual_k if cmp_residual_k is not None else dummy_i32).reshape(-1),
            tla.arch.RowMajor,
        ),
        cached["prep"][1],
        cached["prep"][2],
        cached["prep"][3],
        cached["prep"][4],
    ]
    metadata_args = metadata_tensors + [
        block_capacity,
        has_residual,
        mask_mode,
        cmp_ratio,
    ]
    pack_tensors = [
        _tensor(k_flat, tla.arch.RowMajor),
        _tensor(descale_k_flat, tla.arch.RowMajor),
        cu_q_descriptor,
        cu_k_descriptor,
        block_table_descriptor,
        cached["prep"][1],
        cached["prep"][3],
        cached["prep"][5],
        cached["prep"][6],
    ]
    pack_args = pack_tensors + [
        rows,
        pack_splits,
        max_tokens,
        is_paged,
        page_size,
        records_per_page,
    ]
    prep = _PREP_CACHE.get(prep_key)
    if prep is None:
        metadata = tla.compile(
            prepare_metadata_kernel, *metadata_args, options="--npu-arch 3510"
        )
        pack = tla.compile(
            pack_candidates_kernel, *pack_args, options="--npu-arch 3510"
        )
        prep = (metadata, pack)
        _PREP_CACHE[prep_key] = prep

    # Packed MXFP4 storage is rebound as logical FP4 tensors with zero copy.
    q_pack = q.reshape(rows * Q_HEADS, PACKED_DIM)
    q_scale = descale_q.reshape(rows * Q_HEADS, SCALE_GROUPS)
    q_rows = q_pack.reshape(rows, Q_HEADS, PACKED_DIM)
    qs_rows = q_scale.reshape(rows, Q_HEADS, SCALE_GROUPS)

    def q_scale_tensor(row_index):
        return _tensor(
            qs_rows[row_index],
            tla.arch.RowMajorMxScaleA,
            origin_shape=(Q_HEADS, SCALE_GROUPS),
            element_type=tla.Float8E8M0,
        ).mark_layout_dynamic()

    def qk_row_tensors(row_index):
        static = cached["qk"][row_index]
        return [
            _tensor(
                q_rows[row_index],
                tla.arch.RowMajor,
                origin_shape=(Q_HEADS, HEAD_DIM),
                element_type=tla.Float4E2M1,
            ).mark_layout_dynamic(),
            q_scale_tensor(row_index),
            static[0],
            static[1],
            static[2],
            static[3],
        ]

    qk_key = (max_tokens, device_index)
    qk_artifact = None
    if rows % 2:
        qk_artifact = _QK_CACHE.get(qk_key)
        qk_compile_args = qk_row_tensors(0) + [1, max_tokens]
        if qk_artifact is None:
            qk_artifact = tla.compile(
                qk_kernel, *qk_compile_args, options="--npu-arch 3510"
            )
            _QK_CACHE[qk_key] = qk_artifact

    q_pack_descriptor = _tensor(
        q_pack,
        tla.arch.RowMajor,
        origin_shape=(rows * Q_HEADS, HEAD_DIM),
        element_type=tla.Float4E2M1,
    ).mark_layout_dynamic()
    qk2_artifacts = []
    qk2_arg_lists = []
    qk2_shards = min(max_tokens // BN, max(1, cube_cores // 2))
    # One Cube launch handles a row pair; row-private K/K-scale/QK descriptors
    # avoid MX-scale aliasing. Row-local shards keep Q resident in L0A.
    for pair_base in range(0, rows - 1, 2):
        qk2_key = (
            rows,
            max_tokens,
            pair_base,
            qk2_shards,
            device_index,
            "qk2-stream",
        )
        qk2_artifact = _QK_CACHE.get(qk2_key)
        qk2_tensors = [
            q_pack_descriptor,
            q_scale_tensor(pair_base),
            cached["qk"][pair_base][0],
            cached["qk"][pair_base][1],
            cached["qk"][pair_base][2],
            cached["qk"][pair_base][3],
            q_scale_tensor(pair_base + 1),
            cached["qk"][pair_base + 1][0],
            cached["qk"][pair_base + 1][1],
            cached["qk"][pair_base + 1][2],
            cached["qk"][pair_base + 1][3],
        ]
        if qk2_artifact is None:
            qk2_artifact = tla.compile(
                qk2_stream_kernel,
                *(qk2_tensors + [max_tokens, pair_base, qk2_shards]),
                options="--npu-arch 3510",
            )
            _QK_CACHE[qk2_key] = qk2_artifact
        qk2_artifacts.append(qk2_artifact)
        qk2_arg_lists.append(qk2_tensors)

    indices = torch.empty((rows, 1, topk), dtype=torch.int32, device=device)
    values_work = torch.empty((rows, 1, topk), dtype=torch.bfloat16, device=device)
    sort_size = 1 << (max_tokens - 1).bit_length()
    log_sort = sort_size.bit_length() - 1
    topk_capacity = 1 << (topk - 1).bit_length()
    base_chunk_size = min(sort_size, max(TOPK_CHUNK, topk_capacity))
    # A small top-K cannot receive more than K entries from any disjoint
    # subset. Use enough small leaves to occupy the row's AIV budget, then
    # prune every merge-tree node to the next-power-of-two K capacity.
    small_tree = topk_capacity <= 64 and sort_size > base_chunk_size
    chunk_size = base_chunk_size
    if small_tree:
        task_slots = max(2, vector_cores // rows)
        desired_chunks = 1 << (task_slots.bit_length() - 1)
        desired_chunks = min(desired_chunks, sort_size // topk_capacity)
        chunk_size = sort_size // max(2, desired_chunks)
    use_chunked_topk = sort_size > chunk_size
    topk_key = (
        "chunked" if use_chunked_topk else "whole",
        rows,
        max_tokens,
        block_capacity,
        sort_size,
        topk,
        has_offset,
        write_values,
        device_index,
    )
    offset_descriptor = (
        _tensor(output_idx_offset.reshape(-1), tla.arch.RowMajor)
        if output_idx_offset is not None
        else cached["dummy_off"]
    )
    weight_descriptor = _tensor(w.reshape(rows, Q_HEADS), tla.arch.RowMajor)
    indices_descriptor = _tensor(indices.reshape(-1), tla.arch.RowMajor)
    values_descriptor = _tensor(values_work.reshape(-1), tla.arch.RowMajor)
    topk_artifacts = _TOPK_CACHE.get(topk_key)
    if use_chunked_topk:
        chunk_count = sort_size // chunk_size
        log_chunk = chunk_size.bit_length() - 1
        merge_log_capacity = topk_capacity.bit_length() - 1
        chunk_buffers = _CHUNK_CACHE.get((rows, sort_size, device_index))
        if chunk_buffers is None:
            chunk_keys = torch.empty(
                (rows * sort_size,), dtype=torch.int32, device=device
            )
            merge_keys_a = torch.empty(
                (rows * sort_size,), dtype=torch.int32, device=device
            )
            merge_keys_b = torch.empty(
                (rows * sort_size,), dtype=torch.int32, device=device
            )
            chunk_buffers = {
                "keys": chunk_keys,
                "key_descriptor": _tensor(chunk_keys, tla.arch.RowMajor),
                "merge_keys_a_descriptor": _tensor(merge_keys_a, tla.arch.RowMajor),
                "merge_keys_b_descriptor": _tensor(merge_keys_b, tla.arch.RowMajor),
            }
            _CHUNK_CACHE[(rows, sort_size, device_index)] = chunk_buffers
        chunk_tensors = [
            cached["topk"][0],
            weight_descriptor,
            cached["topk"][1],
            cached["topk"][2],
            chunk_buffers["key_descriptor"],
            rows,
            max_tokens,
            sort_size,
            chunk_size,
            log_chunk,
            chunk_count,
        ]
        # Expose merge-tree levels only where extra AIV parallelism amortizes
        # the additional launches; launch-bound shapes use the serial kernel.
        use_parallel_merge = small_tree or (
            (chunk_count <= 4 and rows <= 5)
            or (chunk_count >= 8 and rows <= 4)
            or (chunk_count >= 8 and rows == 8)
        )
        merge_arg_lists = []
        merge_block_nums = []
        merge_tensors = []
        if use_parallel_merge:
            source_chunks = chunk_count
            source_key_descriptor = chunk_buffers["key_descriptor"]
            input_chunk_size = chunk_size
            input_log_chunk = log_chunk
            output_chunk_size = topk_capacity if small_tree else chunk_size
            stage = 0
            while source_chunks > 1:
                destination_suffix = "a" if stage % 2 == 0 else "b"
                merge_arg_lists.append(
                    [
                        source_key_descriptor,
                        chunk_buffers[
                            "merge_keys_{}_descriptor".format(destination_suffix)
                        ],
                        offset_descriptor,
                        indices_descriptor,
                        values_descriptor,
                        rows,
                        input_chunk_size,
                        output_chunk_size,
                        input_log_chunk,
                        source_chunks,
                        topk,
                        has_offset,
                        write_values,
                        source_chunks == 2,
                    ]
                )
                merge_block_nums.append(rows * (source_chunks // 2))
                source_key_descriptor = chunk_buffers[
                    "merge_keys_{}_descriptor".format(destination_suffix)
                ]
                source_chunks //= 2
                stage += 1
                if small_tree:
                    input_chunk_size = topk_capacity
                    output_chunk_size = topk_capacity
                    input_log_chunk = merge_log_capacity
        else:
            round_pairs = [chunk_count >> (round_index + 1) for round_index in range(5)]
            merge_tensors = [
                chunk_buffers["key_descriptor"],
                offset_descriptor,
                indices_descriptor,
                values_descriptor,
                rows,
                sort_size,
                chunk_size,
                log_chunk,
                chunk_count,
                chunk_count.bit_length() - 1,
                *round_pairs,
                topk,
                has_offset,
                write_values,
            ]
        if topk_artifacts is None:
            chunk_artifact = tla.compile(
                reduce_chunk_topk_kernel,
                *chunk_tensors,
                options="--npu-arch 3510",
            )
            if use_parallel_merge:
                merge_payload = tuple(
                    tla.compile(
                        merge_stage_topk_kernel,
                        *merge_args,
                        options="--npu-arch 3510",
                    )
                    for merge_args in merge_arg_lists
                )
            else:
                merge_payload = tla.compile(
                    merge_topk_kernel,
                    *merge_tensors,
                    options="--npu-arch 3510",
                )
            topk_artifacts = (
                chunk_artifact,
                use_parallel_merge,
                merge_payload,
            )
            _TOPK_CACHE[topk_key] = topk_artifacts
        chunk_artifact, use_parallel_merge, merge_payload = topk_artifacts
    else:
        topk_tensors = [
            cached["topk"][0],
            weight_descriptor,
            cached["topk"][1],
            cached["topk"][2],
            offset_descriptor,
            indices_descriptor,
            values_descriptor,
            rows,
            max_tokens,
            sort_size,
            log_sort,
            topk,
            has_offset,
            write_values,
        ]
        if topk_artifacts is None:
            topk_artifacts = tla.compile(
                reduce_topk_kernel, *topk_tensors, options="--npu-arch 3510"
            )
            _TOPK_CACHE[topk_key] = topk_artifacts
        topk_artifact = topk_artifacts
    metadata, pack = prep
    metadata(
        *metadata_tensors,
        block_num=max(1, min(rows, vector_cores)),
    )
    pack(
        *pack_tensors,
        block_num=max(1, min(rows * pack_splits, vector_cores)),
    )
    torch.npu.synchronize()
    for pair_index in range(len(qk2_artifacts)):
        qk2_artifacts[pair_index](
            *qk2_arg_lists[pair_index],
            block_num=2 * qk2_shards,
        )
    if rows % 2:
        qk_artifact(
            *qk_row_tensors(rows - 1),
            block_num=cube_cores,
        )
    torch.npu.synchronize()
    if use_chunked_topk:
        chunk_artifact(
            *chunk_tensors[:5],
            block_num=max(1, min(rows * chunk_count, vector_cores)),
        )
        torch.npu.synchronize()
        if use_parallel_merge:
            for stage, merge_artifact in enumerate(merge_payload):
                merge_artifact(
                    *merge_arg_lists[stage][:5],
                    block_num=max(1, min(merge_block_nums[stage], vector_cores)),
                )
        else:
            merge_payload(
                *merge_tensors[:4],
                block_num=max(1, min(rows, vector_cores)),
            )
    else:
        topk_artifact(*topk_tensors[:7], block_num=max(1, min(rows, vector_cores)))
    torch.npu.synchronize()

    if write_values:
        values = values_work
    else:
        values = torch.empty((0,), dtype=torch.bfloat16, device=device)
    return indices, values
