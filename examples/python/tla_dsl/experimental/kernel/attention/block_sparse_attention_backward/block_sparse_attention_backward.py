# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import catlass.tla as tla


_EXECUTOR_CACHE = {}


_SYNC_ALL_SOURCE = r"""
#include "kernel_operator.h"
extern "C" {
[aicore] __attribute__((always_inline)) void tla_sync_all() {
    AscendC::SyncAll();
}
}
"""


@tla.extern(source=_SYNC_ALL_SOURCE, name="tla_sync_all")
def tla_sync_all() -> None: ...


@tla.kernel
def block_sparse_attention_backward_kernel(
    mem_q: tla.Tensor,
    mem_k: tla.Tensor,
    mem_v: tla.Tensor,
    mem_do: tla.Tensor,
    mem_o: tla.Tensor,
    mem_lse: tla.Tensor,
    mem_dq: tla.Tensor,
    mem_dk_acc: tla.Tensor,
    mem_dv_acc: tla.Tensor,
    mem_dk: tla.Tensor,
    mem_dv: tla.Tensor,
    scratch_p: tla.Tensor,
    scratch_ds: tla.Tensor,
    block_indices: tla.Tensor,
    cu_seqlens_q: tla.Tensor,
    cu_seqlens_k: tla.Tensor,
    softmax_scale: tla.Float32,
    batch_size: tla.Int32,
    q_sequence_length: tla.Int32,
    kv_sequence_length: tla.Int32,
    total_q_rows: tla.Int32,
    total_k_rows: tla.Int32,
    mask_heads: tla.Int32,
    use_varlen: tla.Int32,
    direct_pds: tla.Constexpr[int],
    inline_tail: tla.Constexpr[int],
    q_heads: tla.Constexpr[int],
    kv_heads: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    q_block: tla.Constexpr[int],
    kv_block: tla.Constexpr[int],
) -> None:
    """Single-launch, per-head-owner sparse attention backward."""

    c0 = 0
    c1 = 1
    half_q = q_block // 2
    fp32_lanes = 64
    zero_tile_rows = 64
    alignment = 512
    # L0C accumulators are fed to tla.mmad, which needs 1024-byte alignment.
    l0c_alignment = 1024
    io_dtype = mem_q.ptr.dtype
    selected_count = block_indices.origin_shape[1]
    q_block_count = (q_sequence_length + q_block - 1) // q_block
    group_size = q_heads // kv_heads
    tasks_per_batch = q_heads * q_block_count
    total_q_tasks = batch_size * tasks_per_batch
    batch_scan = tla.as_numeric(c0)
    q_start_scan = tla.as_numeric(c0)
    q_end_scan = tla.as_numeric(c0)
    q_length_scan = tla.as_numeric(c0)
    q_blocks_scan = tla.as_numeric(c0)
    batch_tasks_scan = tla.as_numeric(c0)
    if use_varlen != c0:
        total_q_tasks = tla.as_numeric(c0)
        for batch_scan in tla.range(batch_size):
            q_start_scan = cu_seqlens_q[batch_scan]
            q_end_scan = cu_seqlens_q[batch_scan + c1]
            q_length_scan = q_end_scan - q_start_scan
            q_blocks_scan = (q_length_scan + q_block - c1) // q_block
            total_q_tasks = total_q_tasks + q_heads * q_blocks_scan
    cv_ready_0 = tla.cross_flag("cv_ready_0", mode=4)
    cv_ready_1 = tla.cross_flag("cv_ready_1", mode=4)
    cv_free_0 = tla.cross_flag("cv_free_0", mode=4)
    cv_free_1 = tla.cross_flag("cv_free_1", mode=4)
    pds_ready_0 = tla.cross_flag("pds_ready_0", mode=4)
    pds_ready_1 = tla.cross_flag("pds_ready_1", mode=4)
    pds_free_0 = tla.cross_flag("pds_free_0", mode=4)
    pds_free_1 = tla.cross_flag("pds_free_1", mode=4)
    output_zeroed_0 = tla.cross_flag("output_zeroed_0", mode=4)
    output_zeroed_1 = tla.cross_flag("output_zeroed_1", mode=4)
    gradient_ready_0 = tla.cross_flag("gradient_ready_0", mode=4)
    gradient_ready_1 = tla.cross_flag("gradient_ready_1", mode=4)
    dq_out_ready = tla.cross_flag("dq_out_ready", mode=4)
    dq_out_free = tla.cross_flag("dq_out_free", mode=4)

    ab_free = tla.flag("ab_free", tla.arch.CUBE, tla.arch.MTE1)
    ab_ready = tla.flag("ab_ready", tla.arch.MTE1, tla.arch.CUBE)
    aux_ab_free = tla.flag("aux_ab_free", tla.arch.CUBE, tla.arch.MTE1)
    aux_ab_ready = tla.flag("aux_ab_ready", tla.arch.MTE1, tla.arch.CUBE)
    l1_ready = tla.flag("l1_ready", tla.arch.MTE2, tla.arch.MTE1)
    qdo_l1_free = tla.flag("qdo_l1_free", tla.arch.MTE1, tla.arch.MTE2)
    qdo_l1_ready = tla.flag("qdo_l1_ready", tla.arch.MTE2, tla.arch.MTE1)
    k_l1_free = tla.flag("k_l1_free", tla.arch.MTE1, tla.arch.MTE2)
    v_l1_free = tla.flag("v_l1_free", tla.arch.MTE1, tla.arch.MTE2)
    c_free = tla.flag("c_free", tla.arch.FIX, tla.arch.CUBE)
    c_ready = tla.flag("c_ready", tla.arch.CUBE, tla.arch.FIX)
    aux_c_free = tla.flag("aux_c_free", tla.arch.FIX, tla.arch.CUBE)
    aux_c_ready = tla.flag("aux_c_ready", tla.arch.CUBE, tla.arch.FIX)
    dq_c_free = tla.flag("dq_c_free", tla.arch.FIX, tla.arch.CUBE)
    dq_c_ready = tla.flag("dq_c_ready", tla.arch.CUBE, tla.arch.FIX)

    l0a_ptr = tla.allocate(
        q_block * head_dim, io_dtype, tla.AddressSpace.l0a, alignment
    )
    l0b_ptr = tla.allocate(
        head_dim * kv_block, io_dtype, tla.AddressSpace.l0b, alignment
    )
    l0a_aux_ptr = tla.allocate(
        q_block * head_dim, io_dtype, tla.AddressSpace.l0a, alignment
    )
    l0b_aux_ptr = tla.allocate(
        head_dim * kv_block, io_dtype, tla.AddressSpace.l0b, alignment
    )
    l0c_tmp_ptr = tla.allocate(
        q_block * kv_block, tla.Float32, tla.AddressSpace.l0c, l0c_alignment
    )
    l0c_aux_ptr = tla.allocate(
        q_block * kv_block, tla.Float32, tla.AddressSpace.l0c, l0c_alignment
    )
    l0c_dq_ptr = tla.allocate(
        q_block * head_dim, tla.Float32, tla.AddressSpace.l0c, l0c_alignment
    )

    l1_q_ptr = tla.allocate(
        q_block * head_dim, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_do_ptr = tla.allocate(
        q_block * head_dim, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_k_ptr = tla.allocate(
        kv_block * head_dim, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_v_ptr = tla.allocate(
        kv_block * head_dim, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_ds_ptr = tla.allocate(
        2 * q_block * kv_block, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_pt_ptr = tla.allocate(
        2 * kv_block * q_block, io_dtype, tla.AddressSpace.l1, alignment
    )

    score_ptr = tla.allocate(
        2 * half_q * kv_block, tla.Float32, tla.AddressSpace.ub, alignment
    )
    dp_ptr = tla.allocate(
        2 * half_q * kv_block, tla.Float32, tla.AddressSpace.ub, alignment
    )
    d_ptr = tla.allocate(half_q, tla.Float32, tla.AddressSpace.ub, alignment)
    lse_ptr = tla.allocate(half_q, tla.Float32, tla.AddressSpace.ub, alignment)
    # GM->UB DMA pads each packed LSE row to a 32-byte boundary.
    lse_ub_head_stride = ((q_heads + 7) // 8) * 8
    lse_gather_ptr = tla.allocate(
        half_q * lse_ub_head_stride, tla.Float32, tla.AddressSpace.ub, alignment
    )
    p_ptr = tla.allocate(
        (half_q + c1) * kv_block, io_dtype, tla.AddressSpace.ub, alignment
    )
    ds_ptr = tla.allocate(
        (half_q + c1) * kv_block, io_dtype, tla.AddressSpace.ub, alignment
    )
    zero_ptr = tla.allocate(
        zero_tile_rows * head_dim,
        tla.Float32,
        tla.AddressSpace.ub,
        alignment,
    )
    post_out_ptr = tla.allocate(
        zero_tile_rows * head_dim,
        io_dtype,
        tla.AddressSpace.ub,
        alignment,
    )
    cast_zero = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    cast_one = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ONE,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )

    with tla.cube():
        tla.set_flag(ab_free)
        tla.set_flag(aux_ab_free)
        tla.set_flag(c_free)
        tla.set_flag(aux_c_free)
        tla.set_flag(dq_c_free)
        tla.set_flag(qdo_l1_free)
        tla.set_flag(k_l1_free)
        tla.set_flag(v_l1_free)
        if tla.const_expr(direct_pds == 0):
            tla.cross_core_set_flag(pds_free_0, tla.arch.MTE2, aiv_id=0)
            tla.cross_core_set_flag(pds_free_1, tla.arch.MTE2, aiv_id=0)
            tla.cross_core_set_flag(pds_free_0, tla.arch.MTE2, aiv_id=1)
            tla.cross_core_set_flag(pds_free_1, tla.arch.MTE2, aiv_id=1)
        else:
            tla.cross_core_set_flag(pds_free_0, tla.arch.MTE1, aiv_id=0)
            tla.cross_core_set_flag(pds_free_1, tla.arch.MTE1, aiv_id=0)
            tla.cross_core_set_flag(pds_free_0, tla.arch.MTE1, aiv_id=1)
            tla.cross_core_set_flag(pds_free_1, tla.arch.MTE1, aiv_id=1)

    with tla.vector():
        init_vec_idx = tla.arch.sub_block_idx()
        if init_vec_idx == c0:
            tla.cross_core_set_flag(cv_free_0, tla.arch.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(cv_free_1, tla.arch.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(dq_out_free, tla.arch.VECTOR, aiv_id=0)
        else:
            tla.cross_core_set_flag(cv_free_0, tla.arch.VECTOR, aiv_id=1)
            tla.cross_core_set_flag(cv_free_1, tla.arch.VECTOR, aiv_id=1)
            tla.cross_core_set_flag(dq_out_free, tla.arch.VECTOR, aiv_id=1)

    with tla.vector():
        zero_vec_idx = tla.arch.sub_block_idx()
        zero_worker_idx = tla.arch.block_idx() * 2 + zero_vec_idx
        zero_worker_count = tla.arch.block_num() * 2
        zero_ub = tla.make_tensor(
            zero_ptr,
            tla.make_layout(
                tla.make_shape(zero_tile_rows, head_dim),
                tla.make_stride(head_dim, c1),
            ),
        )
        with tla.vec.func(mode="simd"):
            zero_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            for zero_row in tla.range(zero_tile_rows):
                for zero_col in tla.range_constexpr(head_dim // fp32_lanes):
                    tla.tile_view(
                        zero_ub,
                        tla.make_shape(c1, fp32_lanes),
                        tla.make_coord(zero_row, zero_col),
                    ).store(
                        tla.full(0.0, dtype=tla.Float32),
                        mask=zero_mask,
                    )
        tla.pipe_barrier(tla.pipes.ALL)
        for zero_k_row in tla.range(
            zero_worker_idx * zero_tile_rows,
            total_k_rows,
            zero_worker_count * zero_tile_rows,
        ):
            zero_k_rows = total_k_rows - zero_k_row
            if zero_k_rows > zero_tile_rows:
                zero_k_rows = tla.as_numeric(zero_tile_rows)
            dk_zero_tile = tla.make_tensor(
                mem_dk_acc.ptr + zero_k_row * head_dim,
                tla.make_layout(
                    tla.make_shape(zero_k_rows, head_dim),
                    tla.make_stride(head_dim, c1),
                ),
            )
            dv_zero_tile = tla.make_tensor(
                mem_dv_acc.ptr + zero_k_row * head_dim,
                tla.make_layout(
                    tla.make_shape(zero_k_rows, head_dim),
                    tla.make_stride(head_dim, c1),
                ),
            )
            zero_ub_tile = tla.tile_view(
                zero_ub,
                tla.make_shape(zero_k_rows, head_dim),
                tla.make_coord(c0, c0),
            )
            tla.copy(dk_zero_tile, zero_ub_tile)
            tla.copy(dv_zero_tile, zero_ub_tile)

    with tla.vector():
        barrier_vec_idx = tla.arch.sub_block_idx()
        tla.pipe_barrier(tla.pipes.ALL)
        tla_sync_all()
        if barrier_vec_idx == c0:
            tla.cross_core_set_flag(output_zeroed_0, tla.arch.MTE3, aiv_id=0)
        else:
            tla.cross_core_set_flag(output_zeroed_1, tla.arch.MTE3, aiv_id=1)

    with tla.cube():
        tla.cross_core_wait_flag(output_zeroed_0, tla.arch.MTE2, aiv_id=0)
        tla.cross_core_wait_flag(output_zeroed_1, tla.arch.MTE2, aiv_id=1)

    task_range = tla.range(tla.arch.block_idx(), total_q_tasks, tla.arch.block_num())
    for task in task_range:
        batch_idx = task // tasks_per_batch
        task_in_batch = task - batch_idx * tasks_per_batch

        q_start = tla.as_numeric(0)
        k_start = tla.as_numeric(0)
        q_length = q_sequence_length
        k_length = kv_sequence_length
        if use_varlen != 0:
            task_prefix = tla.as_numeric(c0)
            for batch_scan in tla.range(batch_size):
                q_start_scan = cu_seqlens_q[batch_scan]
                q_end_scan = cu_seqlens_q[batch_scan + c1]
                q_length_scan = q_end_scan - q_start_scan
                q_blocks_scan = (q_length_scan + q_block - c1) // q_block
                batch_tasks_scan = q_heads * q_blocks_scan
                if task >= task_prefix and task < task_prefix + batch_tasks_scan:
                    batch_idx = batch_scan
                    task_in_batch = task - task_prefix
                task_prefix = task_prefix + batch_tasks_scan
            q_start = cu_seqlens_q[batch_idx]
            k_start = cu_seqlens_k[batch_idx]
            q_length = cu_seqlens_q[batch_idx + 1] - q_start
            k_length = cu_seqlens_k[batch_idx + 1] - k_start

        local_q_blocks = (q_length + q_block - c1) // q_block
        head_idx = task_in_batch // local_q_blocks
        task_q_block = task_in_batch - head_idx * local_q_blocks
        if use_varlen != c0 and q_sequence_length > 2048:
            if tla.const_expr(q_heads % 4 == 0):
                head_cluster = task_in_batch // (local_q_blocks * 4)
                cluster_task = task_in_batch - head_cluster * local_q_blocks * 4
                head_idx = head_cluster * 4 + cluster_task % 4
                task_q_block = cluster_task // 4
        kv_head_idx = head_idx // group_size

        q_stride = head_dim
        k_stride = head_dim
        q_base = (batch_idx * q_heads + head_idx) * q_sequence_length * head_dim
        k_base = (batch_idx * kv_heads + kv_head_idx) * kv_sequence_length * head_dim
        lse_base = (batch_idx * q_heads + head_idx) * q_sequence_length
        lse_stride = c1
        lse_local_head = tla.as_numeric(c0)
        if use_varlen != 0:
            q_stride = q_heads * head_dim
            k_stride = kv_heads * head_dim
            q_base = q_start * q_stride + head_idx * head_dim
            k_base = k_start * k_stride + kv_head_idx * head_dim
            lse_base = q_start * q_heads + head_idx
            lse_stride = q_heads
            lse_local_head = head_idx

        task_q_block_end = task_q_block
        if task_q_block < local_q_blocks:
            task_q_block_end = task_q_block + c1
        for q_block_idx in tla.range(task_q_block, task_q_block_end, c1):
            q_remaining = q_length - q_block_idx * q_block
            q_size = tla.as_numeric(q_block)
            if q_remaining < q_block:
                q_size = q_remaining

            q_gm = tla.make_tensor(
                mem_q.ptr + q_base + q_block_idx * q_block * q_stride,
                tla.make_layout(
                    tla.make_shape(q_size, head_dim), tla.make_stride(q_stride, c1)
                ),
            )
            do_gm = tla.make_tensor(
                mem_do.ptr + q_base + q_block_idx * q_block * q_stride,
                tla.make_layout(
                    tla.make_shape(q_size, head_dim), tla.make_stride(q_stride, c1)
                ),
            )
            l1_q_ref = tla.make_tensor(
                l1_q_ptr,
                tla.make_layout(
                    tla.make_shape(q_size, head_dim), tla.make_stride(head_dim, c1)
                ),
            )
            l1_do_ref = tla.make_tensor(
                l1_do_ptr,
                tla.make_layout(
                    tla.make_shape(q_size, head_dim), tla.make_stride(head_dim, c1)
                ),
            )
            l1_q = tla.make_tensor_like(l1_q_ptr, l1_q_ref, tla.arch.zN)
            l1_do = tla.make_tensor_like(l1_do_ptr, l1_do_ref, tla.arch.zN)
            with tla.cube():
                tla.wait_flag(qdo_l1_free)
                tla.copy(l1_q, q_gm)
                tla.copy(l1_do, do_gm)
                tla.set_flag(qdo_l1_ready)
                tla.wait_flag(qdo_l1_ready)

            with tla.vector():
                vec_idx = tla.arch.sub_block_idx()
                q_first_half = (q_size + c1) // 2
                q_sub = q_first_half if vec_idx == c0 else q_size - q_first_half
                q_sub_offset = vec_idx * half_q
                lse_gm = tla.make_tensor(
                    mem_lse.ptr
                    + lse_base
                    + (q_block_idx * q_block + q_sub_offset) * lse_stride,
                    tla.make_layout(tla.make_shape(q_sub), tla.make_stride(lse_stride)),
                )
                lse_ub = tla.make_tensor(
                    lse_gather_ptr,
                    tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
                )
                lse_gather_ub = tla.make_tensor(
                    lse_gather_ptr,
                    tla.make_layout(
                        tla.make_shape(q_sub, q_heads),
                        tla.make_stride(lse_ub_head_stride, c1),
                    ),
                )
                if use_varlen != 0:
                    lse_gather_gm = tla.make_tensor(
                        mem_lse.ptr
                        + (
                            lse_base
                            - head_idx
                            + (q_block_idx * q_block + q_sub_offset) * q_heads
                        ),
                        tla.make_layout(
                            tla.make_shape(q_sub, q_heads),
                            tla.make_stride(q_heads, c1),
                        ),
                    )
                    tla.copy(lse_gather_ub, lse_gather_gm)
                else:
                    tla.copy(lse_ub, lse_gm)
                cache_row_offset = q_block_idx * q_block + q_sub_offset
                do_cache_gm = tla.make_tensor(
                    mem_do.ptr + q_base + cache_row_offset * q_stride,
                    tla.make_layout(
                        tla.make_shape(q_sub, head_dim),
                        tla.make_stride(q_stride, c1),
                    ),
                )
                o_cache_gm = tla.make_tensor(
                    mem_o.ptr + q_base + cache_row_offset * q_stride,
                    tla.make_layout(
                        tla.make_shape(q_sub, head_dim),
                        tla.make_stride(q_stride, c1),
                    ),
                )
                do_cache_ub = tla.make_tensor(
                    p_ptr,
                    tla.make_layout(
                        tla.make_shape(q_sub, head_dim),
                        tla.make_stride(head_dim, c1),
                    ),
                )
                o_cache_ub = tla.make_tensor(
                    post_out_ptr,
                    tla.make_layout(
                        tla.make_shape(q_sub, head_dim),
                        tla.make_stride(head_dim, c1),
                    ),
                )
                tla.copy(do_cache_ub, do_cache_gm)
                tla.copy(o_cache_ub, o_cache_gm)
                tla.pipe_barrier(tla.pipes.ALL)
                with tla.vec.func(mode="simd"):
                    f32_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    compute_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=io_dtype)
                    io_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=io_dtype)
                    one_mask = tla.update_mask(c1, dtype=tla.Float32)[0]
                    d_cache_ub = tla.make_tensor(
                        d_ptr,
                        tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
                    )
                    for cache_row in tla.range(q_sub):
                        do_cache_row = tla.tile_view(
                            do_cache_ub,
                            tla.make_shape(c1, head_dim),
                            tla.make_coord(cache_row, c0),
                        ).load()
                        o_cache_row = tla.tile_view(
                            o_cache_ub,
                            tla.make_shape(c1, head_dim),
                            tla.make_coord(cache_row, c0),
                        ).load()
                        do_odd = do_cache_row.to(
                            tla.Float32, cast_zero, mask=compute_mask
                        )
                        do_even = do_cache_row.to(
                            tla.Float32, cast_one, mask=compute_mask
                        )
                        o_odd = o_cache_row.to(tla.Float32, cast_zero, mask=io_mask)
                        o_even = o_cache_row.to(tla.Float32, cast_one, mask=io_mask)
                        dot_odd = tla.mul(do_odd, o_odd, mask=f32_mask).reduce(
                            tla.ReductionOp.ADD, mask=f32_mask
                        )
                        dot_even = tla.mul(do_even, o_even, mask=f32_mask).reduce(
                            tla.ReductionOp.ADD, mask=f32_mask
                        )
                        dot = tla.add(
                            tla.full(dot_odd, dtype=tla.Float32),
                            tla.full(dot_even, dtype=tla.Float32),
                            mask=f32_mask,
                        )
                        tla.tile_view(
                            d_cache_ub,
                            tla.make_shape(c1),
                            tla.make_coord(cache_row),
                        ).store(
                            dot,
                            params=tla.params.UnalignStoreParams(),
                            mask=one_mask,
                        )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE,
                        tla.params.MemType.VEC_LOAD,
                    )
                tla.pipe_barrier(tla.pipes.ALL)

            dq_ref = tla.make_tensor(
                l0c_dq_ptr,
                tla.make_layout(
                    tla.make_shape(q_size, head_dim),
                    tla.make_stride(head_dim, c1),
                ),
            )
            l0_dq_tile = tla.make_tensor_like(l0c_dq_ptr, dq_ref, tla.arch.L0Clayout)

            for _single_pass in tla.range_constexpr(1):
                pass_idx = c1
                # Keep MM12 for the current sparse tile one item ahead of the
                # MM345 consumers.  The final iteration carries zero-sized
                # producer tiles so it only flushes the lagged MM345 stage;
                # readiness tokens are still balanced by the existing drain.
                for sparse_idx in tla.range(c0, selected_count + c1, c1):
                    current_sparse_idx = sparse_idx
                    if current_sparse_idx == selected_count:
                        current_sparse_idx = selected_count - c1
                    mask_head = c0
                    if mask_heads != c1:
                        mask_head = head_idx
                    mask_row = (
                        batch_idx * mask_heads + mask_head
                    ) * q_block_count + q_block_idx
                    physical_block = block_indices[mask_row, current_sparse_idx]
                    k_remaining = k_length - physical_block * kv_block
                    k_size = tla.as_numeric(kv_block)
                    if k_remaining < kv_block:
                        k_size = k_remaining
                    producer_q_size = q_size
                    if sparse_idx == selected_count:
                        producer_q_size = tla.as_numeric(c0)
                        k_size = tla.as_numeric(c0)

                    k_rows_gm = tla.make_tensor(
                        mem_k.ptr + k_base + physical_block * kv_block * k_stride,
                        tla.make_layout(
                            tla.make_shape(k_size, head_dim),
                            tla.make_stride(k_stride, c1),
                        ),
                    )
                    k_cols_gm = tla.make_tensor(
                        mem_k.ptr + k_base + physical_block * kv_block * k_stride,
                        tla.make_layout(
                            tla.make_shape(head_dim, k_size),
                            tla.make_stride(c1, k_stride),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    v_cols_gm = tla.make_tensor(
                        mem_v.ptr + k_base + physical_block * kv_block * k_stride,
                        tla.make_layout(
                            tla.make_shape(head_dim, k_size),
                            tla.make_stride(c1, k_stride),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )

                    cv_slot = sparse_idx % 2
                    cv_score_ptr = score_ptr + cv_slot * half_q * kv_block
                    cv_dp_ptr = dp_ptr + cv_slot * half_q * kv_block
                    cv_ds_ptr = l1_ds_ptr + cv_slot * q_block * kv_block
                    cv_p_ptr = l1_pt_ptr + cv_slot * q_block * kv_block
                    with tla.cube():
                        l1_k_col_ref = tla.make_tensor(
                            l1_k_ptr,
                            tla.make_layout(
                                tla.make_shape(head_dim, k_size),
                                tla.make_stride(c1, head_dim),
                                layoutTag=tla.arch.ColumnMajor,
                            ),
                        )
                        l1_k_col = tla.make_tensor_like(
                            l1_k_ptr, l1_k_col_ref, tla.arch.nZ
                        )
                        tla.wait_flag(k_l1_free)
                        tla.copy(l1_k_col, k_cols_gm)
                        tla.set_flag(l1_ready)
                        l1_v_col_ref = tla.make_tensor(
                            l1_v_ptr,
                            tla.make_layout(
                                tla.make_shape(head_dim, k_size),
                                tla.make_stride(c1, head_dim),
                                layoutTag=tla.arch.ColumnMajor,
                            ),
                        )
                        l1_v_col = tla.make_tensor_like(
                            l1_v_ptr, l1_v_col_ref, tla.arch.nZ
                        )
                        tla.wait_flag(v_l1_free)
                        tla.copy(l1_v_col, v_cols_gm)
                        tla.set_flag(qdo_l1_ready)
                        tla.wait_flag(l1_ready)
                        l0_q = tla.make_tensor_like(l0a_ptr, l1_q, tla.arch.zN)
                        l0_kt = tla.make_tensor_like(l0b_ptr, l1_k_col, tla.arch.nZ)
                        score_ref = tla.make_tensor(
                            l0c_tmp_ptr,
                            tla.make_layout(
                                tla.make_shape(producer_q_size, k_size),
                                tla.make_stride(k_size, c1),
                            ),
                        )
                        l0_score = tla.make_tensor_like(
                            l0c_tmp_ptr, score_ref, tla.arch.L0Clayout
                        )
                        tla.wait_flag(ab_free)
                        tla.copy(l0_q, l1_q)
                        tla.copy(l0_kt, l1_k_col)
                        tla.set_flag(k_l1_free)
                        tla.set_flag(ab_ready)
                        tla.wait_flag(ab_ready)
                        tla.wait_flag(c_free)
                        tla.mmad(l0_score, l0_q, l0_kt, init_c=True)
                        tla.set_flag(ab_free)
                        tla.set_flag(c_ready)
                        tla.wait_flag(c_ready)
                        if cv_slot == c0:
                            tla.cross_core_wait_flag(cv_free_0, tla.arch.FIX, aiv_id=0)
                        else:
                            tla.cross_core_wait_flag(cv_free_1, tla.arch.FIX, aiv_id=0)
                        if cv_slot == c0:
                            tla.cross_core_wait_flag(cv_free_0, tla.arch.FIX, aiv_id=1)
                        else:
                            tla.cross_core_wait_flag(cv_free_1, tla.arch.FIX, aiv_id=1)
                        score_ub = tla.make_tensor(
                            cv_score_ptr,
                            tla.make_layout(
                                tla.make_shape(producer_q_size, k_size),
                                tla.make_stride(kv_block, c1),
                            ),
                        )
                        tla.copy(
                            score_ub,
                            l0_score,
                            tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ),
                        )
                        tla.set_flag(c_free)

                    with tla.cube():
                        l1_v_col_ref = tla.make_tensor(
                            l1_v_ptr,
                            tla.make_layout(
                                tla.make_shape(head_dim, k_size),
                                tla.make_stride(c1, head_dim),
                                layoutTag=tla.arch.ColumnMajor,
                            ),
                        )
                        l1_v_col = tla.make_tensor_like(
                            l1_v_ptr, l1_v_col_ref, tla.arch.nZ
                        )
                        tla.wait_flag(qdo_l1_ready)
                        l0_do = tla.make_tensor_like(l0a_aux_ptr, l1_do, tla.arch.zN)
                        l0_vt = tla.make_tensor_like(l0b_aux_ptr, l1_v_col, tla.arch.nZ)
                        dp_ref = tla.make_tensor(
                            l0c_aux_ptr,
                            tla.make_layout(
                                tla.make_shape(producer_q_size, k_size),
                                tla.make_stride(k_size, c1),
                            ),
                        )
                        l0_dp = tla.make_tensor_like(
                            l0c_aux_ptr, dp_ref, tla.arch.L0Clayout
                        )
                        tla.wait_flag(aux_ab_free)
                        tla.copy(l0_do, l1_do)
                        tla.copy(l0_vt, l1_v_col)
                        tla.set_flag(v_l1_free)
                        tla.set_flag(aux_ab_ready)
                        tla.wait_flag(aux_ab_ready)
                        tla.wait_flag(aux_c_free)
                        tla.mmad(l0_dp, l0_do, l0_vt, init_c=True)
                        tla.set_flag(aux_ab_free)
                        tla.set_flag(aux_c_ready)
                        tla.wait_flag(aux_c_ready)
                        dp_ub = tla.make_tensor(
                            cv_dp_ptr,
                            tla.make_layout(
                                tla.make_shape(producer_q_size, k_size),
                                tla.make_stride(kv_block, c1),
                            ),
                        )
                        tla.copy(
                            dp_ub,
                            l0_dp,
                            tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ),
                        )
                        tla.set_flag(aux_c_free)
                        if cv_slot == c0:
                            tla.cross_core_set_flag(cv_ready_0, tla.arch.FIX, aiv_id=0)
                        else:
                            tla.cross_core_set_flag(cv_ready_1, tla.arch.FIX, aiv_id=0)
                        if cv_slot == c0:
                            tla.cross_core_set_flag(cv_ready_0, tla.arch.FIX, aiv_id=1)
                        else:
                            tla.cross_core_set_flag(cv_ready_1, tla.arch.FIX, aiv_id=1)

                    # Keep P in FP32 vector registers through dS; quantize
                    # each Cube operand once to the native input dtype.
                    with tla.vector():
                        vec_idx = tla.arch.sub_block_idx()
                        if vec_idx == c0:
                            if cv_slot == c0:
                                tla.cross_core_wait_flag(
                                    cv_ready_0, tla.arch.VECTOR, aiv_id=0
                                )
                            else:
                                tla.cross_core_wait_flag(
                                    cv_ready_1, tla.arch.VECTOR, aiv_id=0
                                )
                        else:
                            if cv_slot == c0:
                                tla.cross_core_wait_flag(
                                    cv_ready_0, tla.arch.VECTOR, aiv_id=1
                                )
                            else:
                                tla.cross_core_wait_flag(
                                    cv_ready_1, tla.arch.VECTOR, aiv_id=1
                                )
                        q_first_half = (producer_q_size + c1) // 2
                        q_sub = (
                            q_first_half
                            if vec_idx == c0
                            else producer_q_size - q_first_half
                        )
                        score_ub = tla.make_tensor(
                            cv_score_ptr,
                            tla.make_layout(
                                tla.make_shape(q_sub, kv_block),
                                tla.make_stride(kv_block, c1),
                            ),
                        )
                        lse_ub = tla.make_tensor(
                            lse_gather_ptr + lse_local_head,
                            tla.make_layout(
                                tla.make_shape(q_sub),
                                tla.make_stride(
                                    lse_ub_head_stride if use_varlen != 0 else c1
                                ),
                            ),
                        )
                        lse_gather_compute = tla.make_tensor(
                            lse_gather_ptr,
                            tla.make_layout(
                                tla.make_shape(q_sub, q_heads),
                                tla.make_stride(lse_ub_head_stride, c1),
                            ),
                        )
                        p_base = tla.make_tensor(
                            p_ptr,
                            tla.make_layout(
                                tla.make_shape(half_q, kv_block),
                                tla.make_stride(kv_block, c1),
                            ),
                        )
                        p_direct_row = tla.make_tensor(
                            p_ptr,
                            tla.make_layout(
                                tla.make_shape(half_q + c1, kv_block),
                                tla.make_stride(kv_block, c1),
                            ),
                        )
                        p_direct_base = tla.make_tensor_like(
                            p_ptr, p_direct_row, tla.arch.zNUnAlign
                        )

                        vec_idx = tla.arch.sub_block_idx()
                        q_first_half = (producer_q_size + c1) // 2
                        q_sub = (
                            q_first_half
                            if vec_idx == c0
                            else producer_q_size - q_first_half
                        )
                        dp_ub = tla.make_tensor(
                            cv_dp_ptr,
                            tla.make_layout(
                                tla.make_shape(q_sub, kv_block),
                                tla.make_stride(kv_block, c1),
                            ),
                        )
                        d_ub = tla.make_tensor(
                            d_ptr,
                            tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
                        )
                        p_fp32_base = tla.make_tensor(
                            cv_score_ptr,
                            tla.make_layout(
                                tla.make_shape(q_sub, kv_block),
                                tla.make_stride(kv_block, c1),
                            ),
                        )
                        p_b16_base = tla.make_tensor(
                            p_ptr,
                            tla.make_layout(
                                tla.make_shape(half_q, kv_block),
                                tla.make_stride(kv_block, c1),
                            ),
                        )
                        ds_base = tla.make_tensor(
                            ds_ptr,
                            tla.make_layout(
                                tla.make_shape(half_q, kv_block),
                                tla.make_stride(kv_block, c1),
                            ),
                        )
                        p_direct_row = tla.make_tensor(
                            p_ptr,
                            tla.make_layout(
                                tla.make_shape(half_q + c1, kv_block),
                                tla.make_stride(kv_block, c1),
                            ),
                        )
                        p_direct_base = tla.make_tensor_like(
                            p_ptr, p_direct_row, tla.arch.zNUnAlign
                        )
                        ds_direct_row = tla.make_tensor(
                            ds_ptr,
                            tla.make_layout(
                                tla.make_shape(half_q + c1, kv_block),
                                tla.make_stride(kv_block, c1),
                            ),
                        )
                        ds_direct_base = tla.make_tensor_like(
                            ds_ptr, ds_direct_row, tla.arch.zNUnAlign
                        )
                        with tla.vec.func(mode="simd"):
                            f32_mask = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            b16_mask = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=io_dtype
                            )
                            if tla.const_expr(inline_tail == 0):
                                tail = (k_size - c1) % fp32_lanes + c1
                                tail_mask = tla.update_mask(tail, dtype=tla.Float32)[0]
                                min_reg = tla.full(-3.0e38, dtype=tla.Float32)
                                if k_size != kv_block:
                                    for row_pad in tla.range(q_sub):
                                        if k_size > fp32_lanes:
                                            s_tail = tla.tile_view(
                                                score_ub,
                                                tla.make_shape(c1, fp32_lanes),
                                                tla.make_coord(row_pad, c1),
                                            )
                                            s_tail.store(
                                                tla.where(
                                                    tail_mask, s_tail.load(), min_reg
                                                ),
                                                mask=f32_mask,
                                            )
                                        else:
                                            s_first = tla.tile_view(
                                                score_ub,
                                                tla.make_shape(c1, fp32_lanes),
                                                tla.make_coord(row_pad, c0),
                                            )
                                            s_first.store(
                                                tla.where(
                                                    tail_mask, s_first.load(), min_reg
                                                ),
                                                mask=f32_mask,
                                            )
                                            s_second = tla.tile_view(
                                                score_ub,
                                                tla.make_shape(c1, fp32_lanes),
                                                tla.make_coord(row_pad, c1),
                                            )
                                            s_second.store(min_reg, mask=f32_mask)
                                    tla.local_mem_bar(
                                        tla.params.MemType.VEC_STORE,
                                        tla.params.MemType.VEC_LOAD,
                                    )
                            else:
                                valid_first = tla.update_mask(
                                    (k_size + c1) // 2, dtype=tla.Float32
                                )[0]
                                valid_second = tla.update_mask(
                                    k_size // 2, dtype=tla.Float32
                                )[0]
                                inline_zero = tla.full(0.0, dtype=tla.Float32)
                            one_mask = tla.update_mask(c1, dtype=tla.Float32)[0]
                            if tla.const_expr(inline_tail == 0):
                                dp_pad_tail = (k_size - c1) % fp32_lanes + c1
                                dp_pad_tail_mask = tla.update_mask(
                                    dp_pad_tail, dtype=tla.Float32
                                )[0]
                                dp_pad_zero_reg = tla.full(0.0, dtype=tla.Float32)
                                if k_size != kv_block:
                                    for dp_pad_row_pad in tla.range(q_sub):
                                        if k_size > fp32_lanes:
                                            dp_pad_p_tail = tla.tile_view(
                                                dp_ub,
                                                tla.make_shape(c1, fp32_lanes),
                                                tla.make_coord(dp_pad_row_pad, c1),
                                            )
                                            dp_pad_p_tail.store(
                                                tla.where(
                                                    dp_pad_tail_mask,
                                                    dp_pad_p_tail.load(),
                                                    dp_pad_zero_reg,
                                                ),
                                                mask=f32_mask,
                                            )
                                        else:
                                            dp_pad_p_first = tla.tile_view(
                                                dp_ub,
                                                tla.make_shape(c1, fp32_lanes),
                                                tla.make_coord(dp_pad_row_pad, c0),
                                            )
                                            dp_pad_p_first.store(
                                                tla.where(
                                                    dp_pad_tail_mask,
                                                    dp_pad_p_first.load(),
                                                    dp_pad_zero_reg,
                                                ),
                                                mask=f32_mask,
                                            )
                                            dp_pad_p_second = tla.tile_view(
                                                dp_ub,
                                                tla.make_shape(c1, fp32_lanes),
                                                tla.make_coord(dp_pad_row_pad, c1),
                                            )
                                            dp_pad_p_second.store(
                                                dp_pad_zero_reg, mask=f32_mask
                                            )
                                    tla.local_mem_bar(
                                        tla.params.MemType.VEC_STORE,
                                        tla.params.MemType.VEC_LOAD,
                                    )
                            for row_pair in tla.range(q_sub // 4):
                                for row_slot in tla.range_constexpr(4):
                                    row = row_pair * 4 + row_slot
                                    score_row = tla.tile_view(
                                        score_ub,
                                        tla.make_shape(c1, 2 * fp32_lanes),
                                        tla.make_coord(row, c0),
                                    )
                                    s_odd, s_even = score_row.load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                        )
                                    )
                                    lse = tla.tile_view(
                                        lse_ub,
                                        tla.make_shape(c1),
                                        tla.make_coord(row),
                                    ).load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    p_odd = tla.exp(
                                        tla.sub(
                                            tla.mul(
                                                s_odd, softmax_scale, mask=f32_mask
                                            ),
                                            lse,
                                            mask=f32_mask,
                                        ),
                                        mask=f32_mask,
                                    )
                                    p_even = tla.exp(
                                        tla.sub(
                                            tla.mul(
                                                s_even, softmax_scale, mask=f32_mask
                                            ),
                                            lse,
                                            mask=f32_mask,
                                        ),
                                        mask=f32_mask,
                                    )
                                    if tla.const_expr(inline_tail != 0):
                                        p_odd = tla.where(
                                            valid_first, p_odd, inline_zero
                                        )
                                        p_even = tla.where(
                                            valid_second, p_even, inline_zero
                                        )
                                    p_b16 = tla.bitwise_or(
                                        p_odd.to(io_dtype, cast_zero, mask=f32_mask),
                                        p_even.to(io_dtype, cast_one, mask=f32_mask),
                                        mask=b16_mask,
                                    )
                                    if tla.const_expr(direct_pds == 0):
                                        tla.tile_view(
                                            p_base,
                                            tla.make_shape(c1, 2 * fp32_lanes),
                                            tla.make_coord(row, c0),
                                        ).store(p_b16, mask=b16_mask)
                                    else:
                                        tla.tile_view(
                                            p_direct_base,
                                            tla.make_shape(c1, 2 * fp32_lanes),
                                            tla.make_coord(row, c0),
                                        ).store(
                                            p_b16,
                                            params=tla.params.BlockStoreParams(
                                                block_stride=half_q + c1
                                            ),
                                            mask=b16_mask,
                                        )
                                    dp_row = tla.tile_view(
                                        dp_ub,
                                        tla.make_shape(c1, 2 * fp32_lanes),
                                        tla.make_coord(row, c0),
                                    )
                                    dp_odd, dp_even = dp_row.load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                        )
                                    )
                                    if pass_idx == c0:
                                        dot_odd = tla.mul(
                                            p_odd, dp_odd, mask=f32_mask
                                        ).reduce(tla.ReductionOp.ADD, mask=f32_mask)
                                        dot_even = tla.mul(
                                            p_even, dp_even, mask=f32_mask
                                        ).reduce(tla.ReductionOp.ADD, mask=f32_mask)
                                        dot = tla.add(
                                            tla.full(dot_odd, dtype=tla.Float32),
                                            tla.full(dot_even, dtype=tla.Float32),
                                            mask=f32_mask,
                                        )
                                        d_row = tla.tile_view(
                                            d_ub,
                                            tla.make_shape(c1),
                                            tla.make_coord(row),
                                        )
                                        old_d = d_row.load(
                                            params=tla.params.NormalLoadParams(
                                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                                            )
                                        )
                                        d_row.store(
                                            tla.add(old_d, dot, mask=f32_mask),
                                            params=tla.params.UnalignStoreParams(),
                                            mask=one_mask,
                                        )
                                    else:
                                        d_value = tla.tile_view(
                                            d_ub,
                                            tla.make_shape(c1),
                                            tla.make_coord(row),
                                        ).load(
                                            params=tla.params.NormalLoadParams(
                                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                                            )
                                        )
                                        ds_odd = tla.mul(
                                            p_odd,
                                            tla.sub(dp_odd, d_value, mask=f32_mask),
                                            mask=f32_mask,
                                        )
                                        ds_even = tla.mul(
                                            p_even,
                                            tla.sub(dp_even, d_value, mask=f32_mask),
                                            mask=f32_mask,
                                        )
                                        if tla.const_expr(inline_tail != 0):
                                            ds_odd = tla.where(
                                                valid_first, ds_odd, inline_zero
                                            )
                                            ds_even = tla.where(
                                                valid_second, ds_even, inline_zero
                                            )
                                        ds_b16 = tla.bitwise_or(
                                            ds_odd.to(
                                                io_dtype, cast_zero, mask=f32_mask
                                            ),
                                            ds_even.to(
                                                io_dtype, cast_one, mask=f32_mask
                                            ),
                                            mask=b16_mask,
                                        )
                                        if tla.const_expr(direct_pds == 0):
                                            tla.tile_view(
                                                ds_base,
                                                tla.make_shape(c1, 2 * fp32_lanes),
                                                tla.make_coord(row, c0),
                                            ).store(
                                                ds_b16,
                                                mask=b16_mask,
                                            )
                                        else:
                                            tla.tile_view(
                                                ds_direct_base,
                                                tla.make_shape(c1, 2 * fp32_lanes),
                                                tla.make_coord(row, c0),
                                            ).store(
                                                ds_b16,
                                                params=tla.params.BlockStoreParams(
                                                    block_stride=half_q + c1
                                                ),
                                                mask=b16_mask,
                                            )
                            for tail_row in tla.range((q_sub // 4) * 4, q_sub, c1):
                                tail_score_row = tla.tile_view(
                                    score_ub,
                                    tla.make_shape(c1, 2 * fp32_lanes),
                                    tla.make_coord(tail_row, c0),
                                )
                                tail_s_odd, tail_s_even = tail_score_row.load(
                                    params=tla.params.NormalLoadParams(
                                        load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                    )
                                )
                                tail_lse = tla.tile_view(
                                    lse_ub,
                                    tla.make_shape(c1),
                                    tla.make_coord(tail_row),
                                ).load(
                                    params=tla.params.NormalLoadParams(
                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                    )
                                )
                                tail_p_odd = tla.exp(
                                    tla.sub(
                                        tla.mul(
                                            tail_s_odd, softmax_scale, mask=f32_mask
                                        ),
                                        tail_lse,
                                        mask=f32_mask,
                                    ),
                                    mask=f32_mask,
                                )
                                tail_p_even = tla.exp(
                                    tla.sub(
                                        tla.mul(
                                            tail_s_even, softmax_scale, mask=f32_mask
                                        ),
                                        tail_lse,
                                        mask=f32_mask,
                                    ),
                                    mask=f32_mask,
                                )
                                if tla.const_expr(inline_tail != 0):
                                    tail_p_odd = tla.where(
                                        valid_first, tail_p_odd, inline_zero
                                    )
                                    tail_p_even = tla.where(
                                        valid_second, tail_p_even, inline_zero
                                    )
                                tail_p_b16 = tla.bitwise_or(
                                    tail_p_odd.to(io_dtype, cast_zero, mask=f32_mask),
                                    tail_p_even.to(io_dtype, cast_one, mask=f32_mask),
                                    mask=b16_mask,
                                )
                                if tla.const_expr(direct_pds == 0):
                                    tla.tile_view(
                                        p_base,
                                        tla.make_shape(c1, 2 * fp32_lanes),
                                        tla.make_coord(tail_row, c0),
                                    ).store(tail_p_b16, mask=b16_mask)
                                else:
                                    tla.tile_view(
                                        p_direct_base,
                                        tla.make_shape(c1, 2 * fp32_lanes),
                                        tla.make_coord(tail_row, c0),
                                    ).store(
                                        tail_p_b16,
                                        params=tla.params.BlockStoreParams(
                                            block_stride=half_q + c1
                                        ),
                                        mask=b16_mask,
                                    )
                                tail_dp_row = tla.tile_view(
                                    dp_ub,
                                    tla.make_shape(c1, 2 * fp32_lanes),
                                    tla.make_coord(tail_row, c0),
                                )
                                tail_dp_odd, tail_dp_even = tail_dp_row.load(
                                    params=tla.params.NormalLoadParams(
                                        load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                    )
                                )
                                if pass_idx == c0:
                                    tail_dot_odd = tla.mul(
                                        tail_p_odd, tail_dp_odd, mask=f32_mask
                                    ).reduce(tla.ReductionOp.ADD, mask=f32_mask)
                                    tail_dot_even = tla.mul(
                                        tail_p_even, tail_dp_even, mask=f32_mask
                                    ).reduce(tla.ReductionOp.ADD, mask=f32_mask)
                                    tail_dot = tla.add(
                                        tla.full(tail_dot_odd, dtype=tla.Float32),
                                        tla.full(tail_dot_even, dtype=tla.Float32),
                                        mask=f32_mask,
                                    )
                                    tail_d_row = tla.tile_view(
                                        d_ub,
                                        tla.make_shape(c1),
                                        tla.make_coord(tail_row),
                                    )
                                    tail_old_d = tail_d_row.load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    tail_d_row.store(
                                        tla.add(tail_old_d, tail_dot, mask=f32_mask),
                                        params=tla.params.UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                else:
                                    tail_d_value = tla.tile_view(
                                        d_ub,
                                        tla.make_shape(c1),
                                        tla.make_coord(tail_row),
                                    ).load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    tail_ds_odd = tla.mul(
                                        tail_p_odd,
                                        tla.sub(
                                            tail_dp_odd, tail_d_value, mask=f32_mask
                                        ),
                                        mask=f32_mask,
                                    )
                                    tail_ds_even = tla.mul(
                                        tail_p_even,
                                        tla.sub(
                                            tail_dp_even, tail_d_value, mask=f32_mask
                                        ),
                                        mask=f32_mask,
                                    )
                                    if tla.const_expr(inline_tail != 0):
                                        tail_ds_odd = tla.where(
                                            valid_first, tail_ds_odd, inline_zero
                                        )
                                        tail_ds_even = tla.where(
                                            valid_second, tail_ds_even, inline_zero
                                        )
                                    tail_ds_b16 = tla.bitwise_or(
                                        tail_ds_odd.to(
                                            io_dtype, cast_zero, mask=f32_mask
                                        ),
                                        tail_ds_even.to(
                                            io_dtype, cast_one, mask=f32_mask
                                        ),
                                        mask=b16_mask,
                                    )
                                    if tla.const_expr(direct_pds == 0):
                                        tla.tile_view(
                                            ds_base,
                                            tla.make_shape(c1, 2 * fp32_lanes),
                                            tla.make_coord(tail_row, c0),
                                        ).store(
                                            tail_ds_b16,
                                            mask=b16_mask,
                                        )
                                    else:
                                        tla.tile_view(
                                            ds_direct_base,
                                            tla.make_shape(c1, 2 * fp32_lanes),
                                            tla.make_coord(tail_row, c0),
                                        ).store(
                                            tail_ds_b16,
                                            params=tla.params.BlockStoreParams(
                                                block_stride=half_q + c1
                                            ),
                                            mask=b16_mask,
                                        )
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )

                        tla.pipe_barrier(tla.pipes.ALL)
                        if vec_idx == c0:
                            if cv_slot == c0:
                                tla.cross_core_set_flag(
                                    cv_free_0, tla.arch.VECTOR, aiv_id=0
                                )
                            else:
                                tla.cross_core_set_flag(
                                    cv_free_1, tla.arch.VECTOR, aiv_id=0
                                )
                        else:
                            if cv_slot == c0:
                                tla.cross_core_set_flag(
                                    cv_free_0, tla.arch.VECTOR, aiv_id=1
                                )
                            else:
                                tla.cross_core_set_flag(
                                    cv_free_1, tla.arch.VECTOR, aiv_id=1
                                )
                        if pass_idx != c0:
                            if vec_idx == c0:
                                if cv_slot == c0:
                                    tla.cross_core_wait_flag(
                                        pds_free_0, tla.arch.MTE3, aiv_id=0
                                    )
                                else:
                                    tla.cross_core_wait_flag(
                                        pds_free_1, tla.arch.MTE3, aiv_id=0
                                    )
                            else:
                                if cv_slot == c0:
                                    tla.cross_core_wait_flag(
                                        pds_free_0, tla.arch.MTE3, aiv_id=1
                                    )
                                else:
                                    tla.cross_core_wait_flag(
                                        pds_free_1, tla.arch.MTE3, aiv_id=1
                                    )
                            if tla.const_expr(direct_pds == 0):
                                scratch_offset = (
                                    tla.arch.block_idx() * q_block * kv_block
                                    + vec_idx * half_q * kv_block
                                )
                                p_gm = tla.make_tensor(
                                    scratch_p.ptr + scratch_offset,
                                    tla.make_layout(
                                        tla.make_shape(q_sub, k_size),
                                        tla.make_stride(kv_block, c1),
                                    ),
                                )
                                ds_gm = tla.make_tensor(
                                    scratch_ds.ptr + scratch_offset,
                                    tla.make_layout(
                                        tla.make_shape(q_sub, k_size),
                                        tla.make_stride(kv_block, c1),
                                    ),
                                )
                                tla.copy(
                                    ds_gm,
                                    tla.tile_view(
                                        ds_base,
                                        tla.make_shape(q_sub, k_size),
                                        tla.make_coord(c0, c0),
                                    ),
                                )
                            else:
                                l1_ds_ref = tla.make_tensor(
                                    cv_ds_ptr,
                                    tla.make_layout(
                                        tla.make_shape(q_block, k_size),
                                        tla.make_stride(kv_block, c1),
                                    ),
                                )
                                l1_ds_full = tla.make_tensor_like(
                                    cv_ds_ptr, l1_ds_ref, tla.arch.zN
                                )
                                l1_ds_half = tla.tile_view(
                                    l1_ds_full,
                                    tla.make_shape(half_q, k_size),
                                    tla.make_coord(vec_idx, c0),
                                )
                                tla.copy(
                                    tla.tile_view(
                                        l1_ds_half,
                                        tla.make_shape(q_sub, k_size),
                                        tla.make_coord(c0, c0),
                                    ),
                                    tla.tile_view(
                                        ds_direct_base,
                                        tla.make_shape(q_sub, k_size),
                                        tla.make_coord(c0, c0),
                                    ),
                                )
                            tla.pipe_barrier(tla.pipes.ALL)
                            if tla.const_expr(direct_pds == 0):
                                tla.copy(
                                    p_gm,
                                    tla.tile_view(
                                        p_b16_base,
                                        tla.make_shape(q_sub, k_size),
                                        tla.make_coord(c0, c0),
                                    ),
                                )
                            else:
                                l1_p_ref = tla.make_tensor(
                                    cv_p_ptr,
                                    tla.make_layout(
                                        tla.make_shape(q_block, k_size),
                                        tla.make_stride(kv_block, c1),
                                    ),
                                )
                                l1_p_full = tla.make_tensor_like(
                                    cv_p_ptr, l1_p_ref, tla.arch.zN
                                )
                                l1_p_half = tla.tile_view(
                                    l1_p_full,
                                    tla.make_shape(half_q, k_size),
                                    tla.make_coord(vec_idx, c0),
                                )
                                tla.copy(
                                    tla.tile_view(
                                        l1_p_half,
                                        tla.make_shape(q_sub, k_size),
                                        tla.make_coord(c0, c0),
                                    ),
                                    tla.tile_view(
                                        p_direct_base,
                                        tla.make_shape(q_sub, k_size),
                                        tla.make_coord(c0, c0),
                                    ),
                                )
                            tla.pipe_barrier(tla.pipes.ALL)
                            if vec_idx == c0:
                                if cv_slot == c0:
                                    tla.cross_core_set_flag(
                                        pds_ready_0, tla.arch.MTE3, aiv_id=0
                                    )
                                else:
                                    tla.cross_core_set_flag(
                                        pds_ready_1, tla.arch.MTE3, aiv_id=0
                                    )
                            else:
                                if cv_slot == c0:
                                    tla.cross_core_set_flag(
                                        pds_ready_0, tla.arch.MTE3, aiv_id=1
                                    )
                                else:
                                    tla.cross_core_set_flag(
                                        pds_ready_1, tla.arch.MTE3, aiv_id=1
                                    )

                    if pass_idx != c0 and sparse_idx != c0:
                        previous_sparse_idx = sparse_idx - c1
                        consume_slot = previous_sparse_idx % 2
                        consume_ds_ptr = l1_ds_ptr + consume_slot * q_block * kv_block
                        consume_p_ptr = l1_pt_ptr + consume_slot * q_block * kv_block
                        previous_physical_block = block_indices[
                            mask_row, previous_sparse_idx
                        ]
                        previous_k_remaining = (
                            k_length - previous_physical_block * kv_block
                        )
                        previous_k_size = tla.as_numeric(kv_block)
                        if previous_k_remaining < kv_block:
                            previous_k_size = previous_k_remaining
                        previous_k_rows_gm = tla.make_tensor(
                            mem_k.ptr
                            + k_base
                            + previous_physical_block * kv_block * k_stride,
                            tla.make_layout(
                                tla.make_shape(previous_k_size, head_dim),
                                tla.make_stride(k_stride, c1),
                            ),
                        )
                        with tla.cube():
                            if tla.const_expr(direct_pds == 0):
                                if consume_slot == c0:
                                    tla.cross_core_wait_flag(
                                        pds_ready_0, tla.arch.MTE2, aiv_id=0
                                    )
                                else:
                                    tla.cross_core_wait_flag(
                                        pds_ready_1, tla.arch.MTE2, aiv_id=0
                                    )
                                if consume_slot == c0:
                                    tla.cross_core_wait_flag(
                                        pds_ready_0, tla.arch.MTE2, aiv_id=1
                                    )
                                else:
                                    tla.cross_core_wait_flag(
                                        pds_ready_1, tla.arch.MTE2, aiv_id=1
                                    )
                                scratch_offset = (
                                    tla.arch.block_idx() * q_block * kv_block
                                )
                                ds_scratch_row = tla.make_tensor(
                                    scratch_ds.ptr + scratch_offset,
                                    tla.make_layout(
                                        tla.make_shape(q_size, previous_k_size),
                                        tla.make_stride(kv_block, c1),
                                    ),
                                )
                                l1_ds = tla.make_tensor_like(
                                    consume_ds_ptr, ds_scratch_row, tla.arch.zN
                                )
                                tla.copy(l1_ds, ds_scratch_row)
                            else:
                                if consume_slot == c0:
                                    tla.cross_core_wait_flag(
                                        pds_ready_0, tla.arch.MTE1, aiv_id=0
                                    )
                                else:
                                    tla.cross_core_wait_flag(
                                        pds_ready_1, tla.arch.MTE1, aiv_id=0
                                    )
                                if consume_slot == c0:
                                    tla.cross_core_wait_flag(
                                        pds_ready_0, tla.arch.MTE1, aiv_id=1
                                    )
                                else:
                                    tla.cross_core_wait_flag(
                                        pds_ready_1, tla.arch.MTE1, aiv_id=1
                                    )
                                l1_ds_row_ref = tla.make_tensor(
                                    consume_ds_ptr,
                                    tla.make_layout(
                                        tla.make_shape(q_block, previous_k_size),
                                        tla.make_stride(kv_block, c1),
                                    ),
                                )
                                l1_ds_storage = tla.make_tensor_like(
                                    consume_ds_ptr, l1_ds_row_ref, tla.arch.zN
                                )
                                l1_ds = tla.tile_view(
                                    l1_ds_storage,
                                    tla.make_shape(q_size, previous_k_size),
                                    tla.make_coord(c0, c0),
                                )

                            l1_k_row_ref = tla.make_tensor(
                                l1_k_ptr,
                                tla.make_layout(
                                    tla.make_shape(previous_k_size, head_dim),
                                    tla.make_stride(head_dim, c1),
                                ),
                            )
                            l1_k_row = tla.make_tensor_like(
                                l1_k_ptr, l1_k_row_ref, tla.arch.zN
                            )
                            tla.wait_flag(k_l1_free)
                            tla.copy(l1_k_row, previous_k_rows_gm)
                            tla.set_flag(l1_ready)
                            tla.wait_flag(l1_ready)
                            l0_ds_ref = tla.make_tensor(
                                l0a_ptr,
                                tla.make_layout(
                                    tla.make_shape(q_size, previous_k_size),
                                    tla.make_stride(kv_block, c1),
                                ),
                            )
                            l0_ds = tla.make_tensor_like(
                                l0a_ptr, l0_ds_ref, tla.arch.zN
                            )
                            l0_k = tla.make_tensor_like(l0b_ptr, l1_k_row, tla.arch.nZ)
                            if sparse_idx == c1:
                                tla.wait_flag(dq_c_free)
                            tla.wait_flag(ab_free)
                            tla.copy(l0_ds, l1_ds)
                            tla.copy(l0_k, l1_k_row)
                            tla.set_flag(ab_ready)
                            tla.wait_flag(ab_ready)
                            tla.mmad(
                                l0_dq_tile,
                                l0_ds,
                                l0_k,
                                init_c=(previous_sparse_idx == c0),
                            )
                            tla.set_flag(ab_free)
                            tla.set_flag(k_l1_free)

                        with tla.cube():
                            if tla.const_expr(direct_pds == 0):
                                scratch_offset = (
                                    tla.arch.block_idx() * q_block * kv_block
                                )
                                ds_scratch_t = tla.make_tensor(
                                    scratch_ds.ptr + scratch_offset,
                                    tla.make_layout(
                                        tla.make_shape(previous_k_size, q_size),
                                        tla.make_stride(c1, kv_block),
                                        layoutTag=tla.arch.ColumnMajor,
                                    ),
                                )
                                l1_dst = tla.make_tensor_like(
                                    consume_ds_ptr, ds_scratch_t, tla.arch.nZ
                                )
                            else:
                                l1_ds_t_ref = tla.make_tensor(
                                    consume_ds_ptr,
                                    tla.make_layout(
                                        tla.make_shape(previous_k_size, q_block),
                                        tla.make_stride(c1, kv_block),
                                        layoutTag=tla.arch.ColumnMajor,
                                    ),
                                )
                                l1_dst_storage = tla.make_tensor_like(
                                    consume_ds_ptr, l1_ds_t_ref, tla.arch.nZ
                                )
                                l1_dst = tla.tile_view(
                                    l1_dst_storage,
                                    tla.make_shape(previous_k_size, q_size),
                                    tla.make_coord(c0, c0),
                                )
                            l0_dst = tla.make_tensor_like(l0a_aux_ptr, l1_dst)
                            l0_q_b = tla.make_tensor_like(
                                l0b_aux_ptr, l1_q, tla.arch.nZ
                            )
                            dk_ref = tla.make_tensor(
                                l0c_tmp_ptr,
                                tla.make_layout(
                                    tla.make_shape(previous_k_size, head_dim),
                                    tla.make_stride(head_dim, c1),
                                ),
                            )
                            l0_dk = tla.make_tensor_like(
                                l0c_tmp_ptr, dk_ref, tla.arch.L0Clayout
                            )
                            tla.wait_flag(aux_ab_free)
                            tla.copy(l0_dst, l1_dst)
                            tla.copy(l0_q_b, l1_q)
                            tla.set_flag(aux_ab_ready)
                            tla.wait_flag(aux_ab_ready)
                            tla.wait_flag(c_free)
                            tla.mmad(
                                l0_dk,
                                l0_dst,
                                l0_q_b,
                                init_c=True,
                            )
                            tla.set_flag(aux_ab_free)
                            tla.set_flag(c_ready)
                            tla.wait_flag(c_ready)
                            dk_acc_gm = tla.make_tensor(
                                mem_dk_acc.ptr
                                + k_base
                                + previous_physical_block * kv_block * k_stride,
                                tla.make_layout(
                                    tla.make_shape(previous_k_size, head_dim),
                                    tla.make_stride(k_stride, c1),
                                ),
                            )
                            tla.copy(
                                dk_acc_gm,
                                l0_dk,
                                tla.params.CopyL0C2DstParams(
                                    atomic_mode=tla.params.AtomicMode.ADD
                                ),
                            )
                            tla.set_flag(c_free)

                        with tla.cube():
                            if tla.const_expr(direct_pds == 0):
                                scratch_offset = (
                                    tla.arch.block_idx() * q_block * kv_block
                                )
                                p_scratch_t = tla.make_tensor(
                                    scratch_p.ptr + scratch_offset,
                                    tla.make_layout(
                                        tla.make_shape(previous_k_size, q_size),
                                        tla.make_stride(c1, kv_block),
                                        layoutTag=tla.arch.ColumnMajor,
                                    ),
                                )
                                l1_pt = tla.make_tensor_like(consume_p_ptr, p_scratch_t)
                                tla.copy(l1_pt, p_scratch_t)
                                if consume_slot == c0:
                                    tla.cross_core_set_flag(
                                        pds_free_0, tla.arch.MTE2, aiv_id=0
                                    )
                                else:
                                    tla.cross_core_set_flag(
                                        pds_free_1, tla.arch.MTE2, aiv_id=0
                                    )
                                if consume_slot == c0:
                                    tla.cross_core_set_flag(
                                        pds_free_0, tla.arch.MTE2, aiv_id=1
                                    )
                                else:
                                    tla.cross_core_set_flag(
                                        pds_free_1, tla.arch.MTE2, aiv_id=1
                                    )
                            else:
                                l1_pt_ref = tla.make_tensor(
                                    consume_p_ptr,
                                    tla.make_layout(
                                        tla.make_shape(previous_k_size, q_block),
                                        tla.make_stride(c1, kv_block),
                                        layoutTag=tla.arch.ColumnMajor,
                                    ),
                                )
                                l1_pt_storage = tla.make_tensor_like(
                                    consume_p_ptr, l1_pt_ref, tla.arch.nZ
                                )
                                l1_pt = tla.tile_view(
                                    l1_pt_storage,
                                    tla.make_shape(previous_k_size, q_size),
                                    tla.make_coord(c0, c0),
                                )
                            tla.set_flag(l1_ready)
                            tla.wait_flag(l1_ready)
                            l0_pt = tla.make_tensor_like(l0a_ptr, l1_pt)
                            l0_do_b = tla.make_tensor_like(l0b_ptr, l1_do, tla.arch.nZ)
                            dv_ref = tla.make_tensor(
                                l0c_aux_ptr,
                                tla.make_layout(
                                    tla.make_shape(previous_k_size, head_dim),
                                    tla.make_stride(head_dim, c1),
                                ),
                            )
                            l0_dv = tla.make_tensor_like(
                                l0c_aux_ptr, dv_ref, tla.arch.L0Clayout
                            )
                            tla.wait_flag(ab_free)
                            tla.copy(l0_pt, l1_pt)
                            tla.copy(l0_do_b, l1_do)
                            if tla.const_expr(direct_pds != 0):
                                if consume_slot == c0:
                                    tla.cross_core_set_flag(
                                        pds_free_0, tla.arch.MTE1, aiv_id=0
                                    )
                                else:
                                    tla.cross_core_set_flag(
                                        pds_free_1, tla.arch.MTE1, aiv_id=0
                                    )
                                if consume_slot == c0:
                                    tla.cross_core_set_flag(
                                        pds_free_0, tla.arch.MTE1, aiv_id=1
                                    )
                                else:
                                    tla.cross_core_set_flag(
                                        pds_free_1, tla.arch.MTE1, aiv_id=1
                                    )
                            tla.set_flag(ab_ready)
                            tla.wait_flag(ab_ready)
                            tla.wait_flag(aux_c_free)
                            tla.mmad(
                                l0_dv,
                                l0_pt,
                                l0_do_b,
                                init_c=True,
                            )
                            tla.set_flag(ab_free)
                            if sparse_idx == selected_count:
                                tla.set_flag(qdo_l1_free)
                            tla.set_flag(aux_c_ready)
                            tla.wait_flag(aux_c_ready)
                            dv_acc_gm = tla.make_tensor(
                                mem_dv_acc.ptr
                                + k_base
                                + previous_physical_block * kv_block * k_stride,
                                tla.make_layout(
                                    tla.make_shape(previous_k_size, head_dim),
                                    tla.make_stride(k_stride, c1),
                                ),
                            )
                            tla.copy(
                                dv_acc_gm,
                                l0_dv,
                                tla.params.CopyL0C2DstParams(
                                    atomic_mode=tla.params.AtomicMode.ADD
                                ),
                            )
                            tla.set_flag(aux_c_free)

            # The flush iteration produces one duplicate P/dS tile.  Drain
            # its readiness tokens and return the single scratch slot before
            # advancing to the next query task.
            with tla.cube():
                if tla.const_expr(direct_pds == 0):
                    if selected_count % 2 == c0:
                        tla.cross_core_wait_flag(pds_ready_0, tla.arch.MTE2, aiv_id=0)
                    else:
                        tla.cross_core_wait_flag(pds_ready_1, tla.arch.MTE2, aiv_id=0)
                    if selected_count % 2 == c0:
                        tla.cross_core_wait_flag(pds_ready_0, tla.arch.MTE2, aiv_id=1)
                    else:
                        tla.cross_core_wait_flag(pds_ready_1, tla.arch.MTE2, aiv_id=1)
                    if selected_count % 2 == c0:
                        tla.cross_core_set_flag(pds_free_0, tla.arch.MTE2, aiv_id=0)
                    else:
                        tla.cross_core_set_flag(pds_free_1, tla.arch.MTE2, aiv_id=0)
                    if selected_count % 2 == c0:
                        tla.cross_core_set_flag(pds_free_0, tla.arch.MTE2, aiv_id=1)
                    else:
                        tla.cross_core_set_flag(pds_free_1, tla.arch.MTE2, aiv_id=1)
                else:
                    if selected_count % 2 == c0:
                        tla.cross_core_wait_flag(pds_ready_0, tla.arch.MTE1, aiv_id=0)
                    else:
                        tla.cross_core_wait_flag(pds_ready_1, tla.arch.MTE1, aiv_id=0)
                    if selected_count % 2 == c0:
                        tla.cross_core_wait_flag(pds_ready_0, tla.arch.MTE1, aiv_id=1)
                    else:
                        tla.cross_core_wait_flag(pds_ready_1, tla.arch.MTE1, aiv_id=1)
                    if selected_count % 2 == c0:
                        tla.cross_core_set_flag(pds_free_0, tla.arch.MTE1, aiv_id=0)
                    else:
                        tla.cross_core_set_flag(pds_free_1, tla.arch.MTE1, aiv_id=0)
                    if selected_count % 2 == c0:
                        tla.cross_core_set_flag(pds_free_0, tla.arch.MTE1, aiv_id=1)
                    else:
                        tla.cross_core_set_flag(pds_free_1, tla.arch.MTE1, aiv_id=1)

            with tla.cube():
                tla.set_flag(dq_c_ready)
                tla.wait_flag(dq_c_ready)
                tla.cross_core_wait_flag(dq_out_free, tla.arch.FIX, aiv_id=0)
                tla.cross_core_wait_flag(dq_out_free, tla.arch.FIX, aiv_id=1)
                dq_out_ub = tla.make_tensor(
                    zero_ptr,
                    tla.make_layout(
                        tla.make_shape(q_size, head_dim),
                        tla.make_stride(head_dim, c1),
                    ),
                )
                tla.copy(
                    dq_out_ub,
                    l0_dq_tile,
                    tla.params.CopyL0C2DstParams(
                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                    ),
                )
                tla.set_flag(dq_c_free)
                tla.cross_core_set_flag(dq_out_ready, tla.arch.FIX, aiv_id=0)
                tla.cross_core_set_flag(dq_out_ready, tla.arch.FIX, aiv_id=1)

            with tla.vector():
                dq_vec_idx = tla.arch.sub_block_idx()
                if dq_vec_idx == c0:
                    tla.cross_core_wait_flag(dq_out_ready, tla.arch.VECTOR, aiv_id=0)
                else:
                    tla.cross_core_wait_flag(dq_out_ready, tla.arch.VECTOR, aiv_id=1)
                dq_first_half = (q_size + c1) // 2
                dq_sub = dq_first_half if dq_vec_idx == c0 else q_size - dq_first_half
                dq_sub_offset = dq_vec_idx * dq_first_half
                dq_acc_ub = tla.make_tensor(
                    zero_ptr,
                    tla.make_layout(
                        tla.make_shape(dq_sub, head_dim),
                        tla.make_stride(head_dim, c1),
                    ),
                )
                dq_out_ub = tla.make_tensor(
                    post_out_ptr,
                    tla.make_layout(
                        tla.make_shape(dq_sub, head_dim),
                        tla.make_stride(head_dim, c1),
                    ),
                )
                dq_acc_flat = tla.make_tensor(
                    zero_ptr,
                    tla.make_layout(
                        tla.make_shape(dq_sub * head_dim), tla.make_stride(c1)
                    ),
                )
                dq_out_flat = tla.make_tensor(
                    post_out_ptr,
                    tla.make_layout(
                        tla.make_shape(dq_sub * head_dim), tla.make_stride(c1)
                    ),
                )
                dq_pack_store = tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_PACK_B32
                )
                with tla.vec.func(mode="simd"):
                    dq_f32_mask = tla.create_mask(
                        pattern=tla.mask.ALL, dtype=tla.Float32
                    )
                    dq_b16_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=io_dtype)
                    for dq_vec in tla.range(c0, dq_sub * head_dim // fp32_lanes, c1):
                        dq_src = tla.tile_view(
                            dq_acc_flat,
                            tla.make_shape(fp32_lanes),
                            tla.make_coord(dq_vec),
                        )
                        dq_dst = tla.tile_view(
                            dq_out_flat,
                            tla.make_shape(fp32_lanes),
                            tla.make_coord(dq_vec),
                        )
                        dq_value = tla.mul(
                            dq_src.load(), softmax_scale, mask=dq_f32_mask
                        ).to(io_dtype, cast_zero, mask=dq_f32_mask)
                        dq_dst.store(dq_value, dq_pack_store, mask=dq_b16_mask)
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE,
                        tla.params.MemType.VEC_LOAD,
                    )
                tla.pipe_barrier(tla.pipes.ALL)
                dq_out_gm = tla.make_tensor(
                    mem_dq.ptr
                    + q_base
                    + (q_block_idx * q_block + dq_sub_offset) * q_stride,
                    tla.make_layout(
                        tla.make_shape(dq_sub, head_dim),
                        tla.make_stride(q_stride, c1),
                    ),
                )
                tla.copy(dq_out_gm, dq_out_ub)
                tla.pipe_barrier(tla.pipes.ALL)
                if dq_vec_idx == c0:
                    tla.cross_core_set_flag(dq_out_free, tla.arch.VECTOR, aiv_id=0)
                else:
                    tla.cross_core_set_flag(dq_out_free, tla.arch.VECTOR, aiv_id=1)

    with tla.cube():
        tla.wait_flag(ab_free)
        tla.wait_flag(aux_ab_free)
        tla.wait_flag(c_free)
        tla.wait_flag(aux_c_free)
        tla.wait_flag(dq_c_free)
        tla.wait_flag(qdo_l1_free)
        tla.wait_flag(k_l1_free)
        tla.wait_flag(v_l1_free)
        tla.cross_core_wait_flag(cv_free_0, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(cv_free_1, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(cv_free_0, tla.arch.FIX, aiv_id=1)
        tla.cross_core_wait_flag(cv_free_1, tla.arch.FIX, aiv_id=1)
        tla.cross_core_wait_flag(dq_out_free, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(dq_out_free, tla.arch.FIX, aiv_id=1)
        # Publish completion only after every FIX write/atomic is visible.
        tla.pipe_barrier(tla.pipes.ALL)
        tla.cross_core_set_flag(gradient_ready_0, tla.arch.FIX, aiv_id=0)
        tla.cross_core_set_flag(gradient_ready_1, tla.arch.FIX, aiv_id=1)

    with tla.vector():
        final_vec_idx = tla.arch.sub_block_idx()
        if final_vec_idx == c0:
            tla.cross_core_wait_flag(pds_free_0, tla.arch.MTE3, aiv_id=0)
            tla.cross_core_wait_flag(pds_free_1, tla.arch.MTE3, aiv_id=0)
            tla.cross_core_wait_flag(gradient_ready_0, tla.arch.VECTOR, aiv_id=0)
        else:
            tla.cross_core_wait_flag(pds_free_0, tla.arch.MTE3, aiv_id=1)
            tla.cross_core_wait_flag(pds_free_1, tla.arch.MTE3, aiv_id=1)
            tla.cross_core_wait_flag(gradient_ready_1, tla.arch.VECTOR, aiv_id=1)

        # All dK/dV atomics target FP32 GM workspaces.  Make every update
        # system-visible before the AIV workers cast disjoint output rows.
        tla.pipe_barrier(tla.pipes.ALL)
        tla_sync_all()

        post_worker_idx = tla.arch.block_idx() * 2 + final_vec_idx
        post_worker_count = tla.arch.block_num() * 2
        # Main-loop score/P storage is dead after the global gradient barrier.
        # Reuse it for dV so the two outputs share the post-phase barriers.
        post_dk_ub = tla.make_tensor(
            zero_ptr,
            tla.make_layout(
                tla.make_shape(zero_tile_rows, head_dim), tla.make_stride(head_dim, c1)
            ),
        )
        post_dv_ub = tla.make_tensor(
            score_ptr,
            tla.make_layout(
                tla.make_shape(zero_tile_rows, head_dim), tla.make_stride(head_dim, c1)
            ),
        )
        post_dk_out_ub = tla.make_tensor(
            post_out_ptr,
            tla.make_layout(
                tla.make_shape(zero_tile_rows, head_dim), tla.make_stride(head_dim, c1)
            ),
        )
        post_dv_out_ub = tla.make_tensor(
            p_ptr,
            tla.make_layout(
                tla.make_shape(zero_tile_rows, head_dim), tla.make_stride(head_dim, c1)
            ),
        )
        post_pack_store = tla.params.NormalStoreParams(
            store_dist=tla.params.StoreDist.DIST_PACK_B32
        )
        for post_row in tla.range(
            post_worker_idx * zero_tile_rows,
            total_k_rows,
            post_worker_count * zero_tile_rows,
        ):
            post_rows = total_k_rows - post_row
            if post_rows > zero_tile_rows:
                post_rows = tla.as_numeric(zero_tile_rows)
            post_shape = tla.make_shape(post_rows, head_dim)
            post_layout = tla.make_layout(post_shape, tla.make_stride(head_dim, c1))
            post_dk_acc_gm = tla.make_tensor(
                mem_dk_acc.ptr + post_row * head_dim, post_layout
            )
            post_dv_acc_gm = tla.make_tensor(
                mem_dv_acc.ptr + post_row * head_dim, post_layout
            )
            post_dk_out_gm = tla.make_tensor(
                mem_dk.ptr + post_row * head_dim, post_layout
            )
            post_dv_out_gm = tla.make_tensor(
                mem_dv.ptr + post_row * head_dim, post_layout
            )
            post_dk_tile = tla.tile_view(post_dk_ub, post_shape, tla.make_coord(c0, c0))
            post_dv_tile = tla.tile_view(post_dv_ub, post_shape, tla.make_coord(c0, c0))
            post_dk_out_tile = tla.tile_view(
                post_dk_out_ub, post_shape, tla.make_coord(c0, c0)
            )
            post_dv_out_tile = tla.tile_view(
                post_dv_out_ub, post_shape, tla.make_coord(c0, c0)
            )
            tla.copy(post_dk_tile, post_dk_acc_gm)
            tla.copy(post_dv_tile, post_dv_acc_gm)
            tla.pipe_barrier(tla.pipes.ALL)
            post_elems = post_rows * head_dim
            post_flat_layout = tla.make_layout(
                tla.make_shape(post_elems), tla.make_stride(c1)
            )
            post_dk_flat = tla.make_tensor(zero_ptr, post_flat_layout)
            post_dv_flat = tla.make_tensor(score_ptr, post_flat_layout)
            post_dk_out_flat = tla.make_tensor(post_out_ptr, post_flat_layout)
            post_dv_out_flat = tla.make_tensor(p_ptr, post_flat_layout)
            with tla.vec.func(mode="simd"):
                post_f32_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                post_b16_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=io_dtype)
                for cast_group in tla.range((post_elems // fp32_lanes) // 4):
                    for cast_slot in tla.range_constexpr(4):
                        cast_vec = cast_group * 4 + cast_slot
                        cast_shape = tla.make_shape(fp32_lanes)
                        cast_coord = tla.make_coord(cast_vec)
                        dk_values = tla.tile_view(
                            post_dk_flat, cast_shape, cast_coord
                        ).load()
                        dv_values = tla.tile_view(
                            post_dv_flat, cast_shape, cast_coord
                        ).load()
                        dk_values = tla.mul(
                            dk_values, softmax_scale, mask=post_f32_mask
                        )
                        dk_cast = dk_values.to(io_dtype, cast_zero, mask=post_f32_mask)
                        dv_cast = dv_values.to(io_dtype, cast_zero, mask=post_f32_mask)
                        tla.tile_view(post_dk_out_flat, cast_shape, cast_coord).store(
                            dk_cast,
                            post_pack_store,
                            mask=post_b16_mask,
                        )
                        tla.tile_view(post_dv_out_flat, cast_shape, cast_coord).store(
                            dv_cast,
                            post_pack_store,
                            mask=post_b16_mask,
                        )
                for tail_cast_vec in tla.range(
                    ((post_elems // fp32_lanes) // 4) * 4, post_elems // fp32_lanes, c1
                ):
                    tail_cast_shape = tla.make_shape(fp32_lanes)
                    tail_cast_coord = tla.make_coord(tail_cast_vec)
                    tail_dk_values = tla.tile_view(
                        post_dk_flat, tail_cast_shape, tail_cast_coord
                    ).load()
                    tail_dv_values = tla.tile_view(
                        post_dv_flat, tail_cast_shape, tail_cast_coord
                    ).load()
                    tail_dk_values = tla.mul(
                        tail_dk_values, softmax_scale, mask=post_f32_mask
                    )
                    tail_dk_cast = tail_dk_values.to(
                        io_dtype, cast_zero, mask=post_f32_mask
                    )
                    tail_dv_cast = tail_dv_values.to(
                        io_dtype, cast_zero, mask=post_f32_mask
                    )
                    tla.tile_view(
                        post_dk_out_flat, tail_cast_shape, tail_cast_coord
                    ).store(
                        tail_dk_cast,
                        post_pack_store,
                        mask=post_b16_mask,
                    )
                    tla.tile_view(
                        post_dv_out_flat, tail_cast_shape, tail_cast_coord
                    ).store(
                        tail_dv_cast,
                        post_pack_store,
                        mask=post_b16_mask,
                    )
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(post_dk_out_gm, post_dk_out_tile)
            tla.copy(post_dv_out_gm, post_dv_out_tile)
            tla.pipe_barrier(tla.pipes.ALL)


def block_sparse_attention_backward(
    q,
    k,
    v,
    grad_output,
    attention_out,
    softmax_lse,
    block_indices,
    block_shape,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    softmax_scale=None,
    is_causal=False,
):
    import torch
    import torch_npu  # noqa: F401

    use_varlen = q.dim() == 3
    expected_rank = 3 if use_varlen else 4
    if (
        q.dim() != expected_rank
        or k.dim() != expected_rank
        or v.dim() != expected_rank
        or grad_output.dim() != expected_rank
        or attention_out.dim() != expected_rank
    ):
        raise ValueError(
            "q, k, v, grad_output, and attention_out must all use BNSD or TND"
        )
    if (
        not q.is_contiguous()
        or not k.is_contiguous()
        or not v.is_contiguous()
        or not grad_output.is_contiguous()
        or not attention_out.is_contiguous()
    ):
        raise ValueError("q, k, v, grad_output, and attention_out must be contiguous")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q must use float16 or bfloat16")
    if (
        k.dtype != q.dtype
        or v.dtype != q.dtype
        or grad_output.dtype != q.dtype
        or attention_out.dtype != q.dtype
        or k.device != q.device
        or v.device != q.device
        or grad_output.device != q.device
        or attention_out.device != q.device
    ):
        raise ValueError(
            "q, k, v, grad_output, and attention_out must share dtype and device"
        )
    if tuple(attention_out.shape) != tuple(grad_output.shape):
        raise ValueError("attention_out shape must match grad_output")
    if softmax_lse.dtype != torch.float32 or softmax_lse.device != q.device:
        raise ValueError("softmax_lse must be FP32 on q's device")
    if not softmax_lse.is_contiguous():
        raise ValueError("softmax_lse must be contiguous")
    if is_causal:
        raise ValueError("causal mode is outside the frozen workload")

    if use_varlen:
        batch_size = int(cu_seqlens_q.numel()) - 1
        q_heads = int(q.size(1))
        kv_heads = int(k.size(1))
        q_sequence_length = int(max_seqlen_q)
        kv_sequence_length = int(max_seqlen_k)
        head_dim = int(q.size(2))
        expected_lse = (int(q.size(0)), q_heads, 1)
    else:
        batch_size = int(q.size(0))
        q_heads = int(q.size(1))
        kv_heads = int(k.size(1))
        q_sequence_length = int(q.size(2))
        kv_sequence_length = int(k.size(2))
        head_dim = int(q.size(3))
        expected_lse = (batch_size, q_heads, q_sequence_length, 1)
        cu_seqlens_q = torch.empty((1,), dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.empty((1,), dtype=torch.int32, device=q.device)

    if tuple(softmax_lse.shape) != expected_lse:
        raise ValueError("softmax_lse shape does not match q")
    if head_dim != 128:
        raise ValueError("head_dim must be 128")
    if q_heads < 1 or kv_heads < 1 or q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if block_indices.dtype != torch.int32 or not block_indices.is_contiguous():
        raise ValueError("block_indices must be contiguous int32")
    if block_indices.device != q.device or block_indices.dim() not in (3, 4):
        raise ValueError("block_indices must be rank 3/4 on q's device")
    if block_shape.dtype != torch.int64 or block_shape.numel() != 2:
        raise ValueError("block_shape must be int64 [128, 128]")

    q_block = 128
    kv_block = 128
    q_block_count = (q_sequence_length + q_block - 1) // q_block
    mask_heads = 1 if block_indices.dim() == 3 else int(block_indices.size(1))
    if int(block_indices.size(0)) != batch_size:
        raise ValueError("block_indices batch does not match q")
    if int(block_indices.size(-2)) != q_block_count:
        raise ValueError("block_indices query-block dimension is invalid")
    if mask_heads not in (1, q_heads):
        raise ValueError("block_indices head dimension must be 1 or q_heads")

    try:
        scale_value = float(softmax_scale)
    except TypeError:
        scale_value = head_dim**-0.5

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    # Vendor-scale workspace: full FP32 dK/dV accumulators only. dQ is cast
    # directly from its owner L0C tile; P/dS scratch is fixed per physical core.
    dk_acc = torch.empty_like(k, dtype=torch.float32)
    dv_acc = torch.empty_like(v, dtype=torch.float32)
    physical_blocks = (26 if use_varlen else 22) if q_sequence_length <= 2048 else 28
    scratch_shape = (physical_blocks, q_block, kv_block)
    scratch_p_tensor = torch.empty(scratch_shape, dtype=q.dtype, device=q.device)
    scratch_ds_tensor = torch.empty(scratch_shape, dtype=q.dtype, device=q.device)

    def as_flat(tensor):
        return tla.from_dlpack(
            tensor, layout_tag=tla.arch.RowMajor, origin_shape=(tensor.numel(),)
        ).mark_layout_dynamic()

    q_t = as_flat(q)
    k_t = as_flat(k)
    v_t = as_flat(v)
    do_t = as_flat(grad_output)
    o_t = as_flat(attention_out)
    lse_t = as_flat(softmax_lse)
    dq_t = as_flat(dq)
    dk_acc_t = as_flat(dk_acc)
    dv_acc_t = as_flat(dv_acc)
    dk_t = as_flat(dk)
    dv_t = as_flat(dv)
    scratch_p_t = as_flat(scratch_p_tensor)
    scratch_ds_t = as_flat(scratch_ds_tensor)
    indices_t = tla.from_dlpack(
        block_indices,
        layout_tag=tla.arch.RowMajor,
        origin_shape=(
            batch_size * mask_heads * q_block_count,
            int(block_indices.size(-1)),
        ),
    ).mark_layout_dynamic()
    cu_q_t = as_flat(cu_seqlens_q)
    cu_k_t = as_flat(cu_seqlens_k)

    scalar_args = (
        tla.Float32(scale_value),
        tla.Int32(batch_size),
        tla.Int32(q_sequence_length),
        tla.Int32(kv_sequence_length),
        tla.Int32(int(q.numel()) // head_dim),
        tla.Int32(int(k.numel()) // head_dim),
        tla.Int32(mask_heads),
        tla.Int32(1 if use_varlen else 0),
    )
    executor_key = (
        str(q.dtype),
        str(q.device),
        q_heads,
        kv_heads,
        int(k.numel()),
        int(block_indices.size(-1)),
        use_varlen,
        (
            q_sequence_length <= 2048
            and (use_varlen or kv_sequence_length % kv_block != 0)
        ),
    )
    executor = _EXECUTOR_CACHE.get(executor_key)
    if executor is None:
        executor = tla.compile(
            block_sparse_attention_backward_kernel,
            q_t,
            k_t,
            v_t,
            do_t,
            o_t,
            lse_t,
            dq_t,
            dk_acc_t,
            dv_acc_t,
            dk_t,
            dv_t,
            scratch_p_t,
            scratch_ds_t,
            indices_t,
            cu_q_t,
            cu_k_t,
            *scalar_args,
            1,
            1
            if (
                q_sequence_length <= 2048
                and (use_varlen or kv_sequence_length % kv_block != 0)
            )
            else 0,
            q_heads,
            kv_heads,
            head_dim,
            q_block,
            kv_block,
            options="--npu-arch 3510",
        )
        _EXECUTOR_CACHE[executor_key] = executor

    executor(
        q_t,
        k_t,
        v_t,
        do_t,
        o_t,
        lse_t,
        dq_t,
        dk_acc_t,
        dv_acc_t,
        dk_t,
        dv_t,
        scratch_p_t,
        scratch_ds_t,
        indices_t,
        cu_q_t,
        cu_k_t,
        *scalar_args,
        block_num=physical_blocks,
    )
    return dq, dk, dv
