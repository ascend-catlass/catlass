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


# A compiled object owns the lazily loaded device executor.  Keeping one object
# per dtype/device/output mode lets every dynamic batch/sequence/mask shape reuse that
# executor instead of re-entering the backend compiler on each public call.
_EXECUTOR_CACHE = {}


@tla.kernel
def block_sparse_attention_kernel(
    mem_q: tla.Tensor,
    mem_k: tla.Tensor,
    mem_v: tla.Tensor,
    mem_o: tla.Tensor,
    mem_lse: tla.Tensor,
    block_indices: tla.Tensor,
    cu_seqlens_q: tla.Tensor,
    cu_seqlens_k: tla.Tensor,
    softmax_scale: tla.Float32,
    batch_size: tla.Int32,
    q_heads: tla.Int32,
    kv_heads: tla.Int32,
    q_sequence_length: tla.Int32,
    kv_sequence_length: tla.Int32,
    head_dim: tla.Constexpr[int],
    q_block: tla.Constexpr[int],
    kv_block: tla.Constexpr[int],
    mask_heads: tla.Int32,
    use_varlen: tla.Int32,
    return_lse: tla.Constexpr[bool],
):
    q_block_sub = q_block // 2
    alignment = 512
    # L0C accumulators are fed to tla.mmad, which needs 1024-byte alignment.
    l0c_alignment = 1024
    l0_half_size = 32768
    vl_float_ele = 64
    min_value = -3.0e38
    pre_launch = 2
    ub_s_otmp_buf_stages = 2
    l0_stages = 2
    k_l1_buf = 2
    v_l1_buf = 2
    p_l1_buf = 3
    mm1_l0a_total_stages = 1
    mm1_l0b_total_stages = 1
    mm2_l0a_total_stages = 1
    mm2_l0b_total_stages = 1

    group_size = q_heads // kv_heads
    q_block_count = (q_sequence_length + q_block - 1) // q_block
    selected_block_count = block_indices.origin_shape[1]
    tasks_per_batch = q_heads * q_block_count
    total_tasks = batch_size * tasks_per_batch
    batch_scan = tla.as_numeric(0)
    q_start_scan = tla.as_numeric(0)
    q_end_scan = tla.as_numeric(0)
    q_length_scan = tla.as_numeric(0)
    q_blocks_scan = tla.as_numeric(0)
    batch_tasks_scan = tla.as_numeric(0)
    if use_varlen != 0:
        total_tasks = tla.as_numeric(0)
        for batch_scan in tla.range(batch_size):
            q_start_scan = cu_seqlens_q[batch_scan]
            q_end_scan = cu_seqlens_q[batch_scan + 1]
            q_length_scan = q_end_scan - q_start_scan
            q_blocks_scan = (q_length_scan + q_block - 1) // q_block
            total_tasks = total_tasks + q_heads * q_blocks_scan
    # QK->softmax (qkReadyFlag, id=0/1, AIC FIX set/wait <-> AIV V wait/set)
    qk_ready_0 = tla.cross_flag("qk_ready_0")
    qk_ready_1 = tla.cross_flag("qk_ready_1")
    # softmax->PV (softmaxReadyFlag, id=2/3/4, AIV MTE3 set <-> AIC MTE1 wait)
    sm_ready_mm2_0 = tla.cross_flag("sm_ready_mm2_0")
    sm_ready_mm2_1 = tla.cross_flag("sm_ready_mm2_1")
    sm_ready_mm2_2 = tla.cross_flag("sm_ready_mm2_2")
    # PV->rescale (pvReadyFlag, id=5/6, AIC FIX set <-> AIV V wait) — PV fixpipe L0C->UB 后通知 rescale
    pv_ready_0 = tla.cross_flag("pv_ready_0")
    pv_ready_1 = tla.cross_flag("pv_ready_1")

    q_l0a_ready_l1 = tla.flag("q_l0a_ready_l1", tla.arch.MTE1, tla.arch.MTE2)
    q_l1_ready_l0 = tla.flag("q_l1_ready_l0", tla.arch.MTE2, tla.arch.MTE1)

    k_l0b_ready_l1_0 = tla.flag("k_l0b_ready_l1_0", tla.arch.MTE1, tla.arch.MTE2)
    k_l0b_ready_l1_1 = tla.flag("k_l0b_ready_l1_1", tla.arch.MTE1, tla.arch.MTE2)
    k_l1_ready_l0_0 = tla.flag("k_l1_ready_l0_0", tla.arch.MTE2, tla.arch.MTE1)
    k_l1_ready_l0_1 = tla.flag("k_l1_ready_l0_1", tla.arch.MTE2, tla.arch.MTE1)

    mmad_ready_l0a_0 = tla.flag("mmad_ready_l0a_0", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0a_1 = tla.flag("mmad_ready_l0a_1", tla.arch.CUBE, tla.arch.MTE1)
    l0a_ready_mmad_0 = tla.flag("l0a_ready_mmad_0", tla.arch.MTE1, tla.arch.CUBE)
    l0a_ready_mmad_1 = tla.flag("l0a_ready_mmad_1", tla.arch.MTE1, tla.arch.CUBE)

    mmad_ready_l0b_0 = tla.flag("mmad_ready_l0b_0", tla.arch.CUBE, tla.arch.MTE1)
    mmad_ready_l0b_1 = tla.flag("mmad_ready_l0b_1", tla.arch.CUBE, tla.arch.MTE1)
    l0b_ready_mmad_0 = tla.flag("l0b_ready_mmad_0", tla.arch.MTE1, tla.arch.CUBE)
    l0b_ready_mmad_1 = tla.flag("l0b_ready_mmad_1", tla.arch.MTE1, tla.arch.CUBE)

    fix_ready_mmad_qk_0 = tla.flag("fix_ready_mmad_qk_0", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_qk_1 = tla.flag("fix_ready_mmad_qk_1", tla.arch.FIX, tla.arch.CUBE)
    mmad_ready_fix_qk_0 = tla.flag("mmad_ready_fix_qk_0", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_qk_1 = tla.flag("mmad_ready_fix_qk_1", tla.arch.CUBE, tla.arch.FIX)

    fix_ready_mmad_pv_0 = tla.flag("fix_ready_mmad_pv_0", tla.arch.FIX, tla.arch.CUBE)
    fix_ready_mmad_pv_1 = tla.flag("fix_ready_mmad_pv_1", tla.arch.FIX, tla.arch.CUBE)
    mmad_ready_fix_pv_0 = tla.flag("mmad_ready_fix_pv_0", tla.arch.CUBE, tla.arch.FIX)
    mmad_ready_fix_pv_1 = tla.flag("mmad_ready_fix_pv_1", tla.arch.CUBE, tla.arch.FIX)

    v_l0b_ready_l1_0 = tla.flag("v_l0b_ready_l1_0", tla.arch.MTE1, tla.arch.MTE2)
    v_l0b_ready_l1_1 = tla.flag("v_l0b_ready_l1_1", tla.arch.MTE1, tla.arch.MTE2)
    v_l1_ready_l0_0 = tla.flag("v_l1_ready_l0_0", tla.arch.MTE2, tla.arch.MTE1)
    v_l1_ready_l0_1 = tla.flag("v_l1_ready_l0_1", tla.arch.MTE2, tla.arch.MTE1)

    #   V_MTE3(ubSBufId)         EVENT_ID0/1 — softmax V 完 -> 触发 P->L1 MTE3
    #   MTE3_V(ubSBufId + 2)     EVENT_ID2/3 — 上一轮 P->L1 MTE3 完成 -> 本轮 softmax 可动
    mte3_ready_softmax_0 = tla.flag(
        "mte3_ready_softmax_0", tla.arch.MTE3, tla.arch.VECTOR
    )
    mte3_ready_softmax_1 = tla.flag(
        "mte3_ready_softmax_1", tla.arch.MTE3, tla.arch.VECTOR
    )
    v_mte3_0 = tla.flag("v_mte3_0", tla.arch.VECTOR, tla.arch.MTE3)
    v_mte3_1 = tla.flag("v_mte3_1", tla.arch.VECTOR, tla.arch.MTE3)
    # mask copy (MTE2->V)
    mask_copy_end = tla.flag("mask_copy_end", tla.arch.MTE2, tla.arch.VECTOR)
    # rescale MTE3→V : 上一轮 O->GM MTE3 完成后本轮 rescale 可动
    mte3_ready_rescale = tla.flag("mte3_ready_rescale", tla.arch.MTE3, tla.arch.VECTOR)
    # rescale V→MTE3 : cast 写完 ub_out_f16 后才 O->GM MTE3
    rescale_v_mte3 = tla.flag("rescale_v_mte3", tla.arch.VECTOR, tla.arch.MTE3)
    if tla.const_expr(return_lse):
        lse_free = tla.flag("lse_free", tla.arch.MTE3, tla.arch.VECTOR)
        lse_ready = tla.flag("lse_ready", tla.arch.VECTOR, tla.arch.MTE3)

    c0 = 0
    c1 = 1

    # Q/K/V/O 均为 b16（f16/bf16），中间 S/PV/acc 为 f32
    io_dtype = mem_q.ptr.dtype

    l1_q_ptr = tla.allocate(
        q_block * head_dim, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_k_ping_ptr = tla.allocate(
        head_dim * kv_block, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_k_pong_ptr = tla.allocate(
        head_dim * kv_block, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_v_ping_ptr = tla.allocate(
        kv_block * head_dim, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_v_pong_ptr = tla.allocate(
        kv_block * head_dim, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_p_0_ptr = tla.allocate(
        q_block * kv_block, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_p_1_ptr = tla.allocate(
        q_block * kv_block, io_dtype, tla.AddressSpace.l1, alignment
    )
    l1_p_2_ptr = tla.allocate(
        q_block * kv_block, io_dtype, tla.AddressSpace.l1, alignment
    )

    l0a_ping_ptr = tla.allocate(
        l0_half_size // 2, io_dtype, tla.AddressSpace.l0a, alignment
    )
    l0a_pong_ptr = tla.allocate(
        l0_half_size // 2, io_dtype, tla.AddressSpace.l0a, alignment
    )
    l0b_ping_ptr = tla.allocate(
        l0_half_size // 2, io_dtype, tla.AddressSpace.l0b, alignment
    )
    l0b_pong_ptr = tla.allocate(
        l0_half_size // 2, io_dtype, tla.AddressSpace.l0b, alignment
    )

    l0_s_ping_ptr = tla.allocate(
        q_block * kv_block, tla.Float32, tla.AddressSpace.l0c, l0c_alignment
    )
    l0_s_pong_ptr = tla.allocate(
        q_block * kv_block, tla.Float32, tla.AddressSpace.l0c, l0c_alignment
    )
    l0_pv_ping_ptr = tla.allocate(
        q_block * head_dim, tla.Float32, tla.AddressSpace.l0c, l0c_alignment
    )
    l0_pv_pong_ptr = tla.allocate(
        q_block * head_dim, tla.Float32, tla.AddressSpace.l0c, l0c_alignment
    )

    ub_s_ping_ptr = tla.allocate(
        q_block_sub * kv_block, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_s_pong_ptr = tla.allocate(
        q_block_sub * kv_block, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_p_f16_ping_ptr = tla.allocate(
        (q_block_sub + 1) * kv_block, io_dtype, tla.AddressSpace.ub, alignment
    )
    ub_p_f16_pong_ptr = tla.allocate(
        (q_block_sub + 1) * kv_block, io_dtype, tla.AddressSpace.ub, alignment
    )
    ub_pv_ping_ptr = tla.allocate(
        q_block_sub * head_dim, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_pv_pong_ptr = tla.allocate(
        q_block_sub * head_dim, tla.Float32, tla.AddressSpace.ub, alignment
    )

    ub_acc_ptr = tla.allocate(
        q_block_sub * head_dim, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_out_f16_ptr = tla.recast_ptr(ub_acc_ptr, dtype=io_dtype)

    ub_now_max_ptr = tla.allocate(
        q_block_sub, tla.Float32, tla.AddressSpace.ub, alignment
    )
    if tla.const_expr(return_lse):
        # Packed TND needs one 32-byte-aligned UB row for each scalar DMA block.
        ub_lse_ptr = tla.allocate(
            q_block_sub * 9, tla.Float32, tla.AddressSpace.ub, alignment
        )
    ub_last_max_0_ptr = tla.allocate(
        q_block_sub, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_last_max_1_ptr = tla.allocate(
        q_block_sub, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_last_max_2_ptr = tla.allocate(
        q_block_sub, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_sum_0_ptr = tla.allocate(
        q_block_sub, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_sum_1_ptr = tla.allocate(
        q_block_sub, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_sum_2_ptr = tla.allocate(
        q_block_sub, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_tmp_ptr = tla.allocate(
        2 * vl_float_ele, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_mask_ptr = tla.allocate(
        q_block_sub * kv_block, tla.Int8, tla.AddressSpace.ub, alignment
    )
    ub_exp_sum_ptr = tla.allocate(
        q_block_sub, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_exp_max_0_ptr = tla.allocate(
        q_block_sub, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_exp_max_1_ptr = tla.allocate(
        q_block_sub, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_exp_max_2_ptr = tla.allocate(
        q_block_sub, tla.Float32, tla.AddressSpace.ub, alignment
    )

    cast_trait_zero = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    cast_trait_one = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ONE,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )

    with tla.cube():
        tla.set_flag(q_l0a_ready_l1)
        tla.set_flag(k_l0b_ready_l1_0)
        tla.set_flag(k_l0b_ready_l1_1)
        tla.set_flag(v_l0b_ready_l1_0)
        tla.set_flag(v_l0b_ready_l1_1)
        tla.set_flag(mmad_ready_l0a_0)
        tla.set_flag(mmad_ready_l0a_1)
        tla.set_flag(mmad_ready_l0b_0)
        tla.set_flag(mmad_ready_l0b_1)
        tla.set_flag(fix_ready_mmad_qk_0)
        tla.set_flag(fix_ready_mmad_qk_1)
        tla.set_flag(fix_ready_mmad_pv_0)
        tla.set_flag(fix_ready_mmad_pv_1)

        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1)
        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1)
        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1)

    with tla.vector():
        tla.cross_core_set_flag(qk_ready_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(qk_ready_1, tla.arch.VECTOR)

        tla.cross_core_set_flag(pv_ready_0, tla.arch.VECTOR)
        tla.cross_core_set_flag(pv_ready_1, tla.arch.VECTOR)

        tla.set_flag(mte3_ready_softmax_0)
        tla.set_flag(mte3_ready_softmax_1)

        tla.set_flag(mte3_ready_rescale)
        if tla.const_expr(return_lse):
            tla.set_flag(lse_free)

    pipeline_iter = tla.as_numeric(-1)
    qk_stage_count = tla.as_numeric(0)
    pv_stage_count = tla.as_numeric(0)
    q_task_serial = tla.as_numeric(-1)
    stage_0_sparse_block = tla.as_numeric(0)
    stage_1_sparse_block = tla.as_numeric(0)
    stage_2_sparse_block = tla.as_numeric(0)
    stage_0_kv_iter = tla.as_numeric(0)
    stage_1_kv_iter = tla.as_numeric(0)
    stage_2_kv_iter = tla.as_numeric(0)
    stage_0_q_block_idx = tla.as_numeric(0)
    stage_1_q_block_idx = tla.as_numeric(0)
    stage_2_q_block_idx = tla.as_numeric(0)
    stage_0_batch_idx = tla.as_numeric(0)
    stage_1_batch_idx = tla.as_numeric(0)
    stage_2_batch_idx = tla.as_numeric(0)
    stage_0_head_idx = tla.as_numeric(0)
    stage_1_head_idx = tla.as_numeric(0)
    stage_2_head_idx = tla.as_numeric(0)
    stage_0_q_state_idx = tla.as_numeric(0)
    stage_1_q_state_idx = tla.as_numeric(0)
    stage_2_q_state_idx = tla.as_numeric(0)
    stage_0_q_tile_size = tla.as_numeric(q_block)
    stage_1_q_tile_size = tla.as_numeric(q_block)
    stage_2_q_tile_size = tla.as_numeric(q_block)
    stage_0_kv_tile_size = tla.as_numeric(kv_block)
    stage_1_kv_tile_size = tla.as_numeric(kv_block)
    stage_2_kv_tile_size = tla.as_numeric(kv_block)

    task_range = tla.range(
        tla.arch.block_idx(),
        total_tasks,
        tla.arch.block_num(),
    )
    for task in task_range:
        q_task_serial = q_task_serial + c1
        q_state_idx = q_task_serial % p_l1_buf
        ub_last_max_ptr = ub_last_max_0_ptr
        ub_sum_ptr = ub_sum_0_ptr
        if q_state_idx == c1:
            ub_last_max_ptr = ub_last_max_1_ptr
            ub_sum_ptr = ub_sum_1_ptr
        elif q_state_idx == 2:
            ub_last_max_ptr = ub_last_max_2_ptr
            ub_sum_ptr = ub_sum_2_ptr
        batch_idx = task // tasks_per_batch
        task_in_batch = task - batch_idx * tasks_per_batch
        q_sequence_start = tla.as_numeric(0)
        kv_sequence_start = tla.as_numeric(0)
        q_length = q_sequence_length
        kv_length = kv_sequence_length
        if use_varlen != 0:
            task_prefix = tla.as_numeric(0)
            for batch_scan in tla.range(batch_size):
                q_start_scan = cu_seqlens_q[batch_scan]
                q_end_scan = cu_seqlens_q[batch_scan + 1]
                q_length_scan = q_end_scan - q_start_scan
                q_blocks_scan = (q_length_scan + q_block - 1) // q_block
                batch_tasks_scan = q_heads * q_blocks_scan
                if task >= task_prefix and task < task_prefix + batch_tasks_scan:
                    batch_idx = batch_scan
                    task_in_batch = task - task_prefix
                task_prefix = task_prefix + batch_tasks_scan
            q_sequence_start = cu_seqlens_q[batch_idx]
            q_length = cu_seqlens_q[batch_idx + 1] - q_sequence_start
            kv_sequence_start = cu_seqlens_k[batch_idx]
            kv_length = cu_seqlens_k[batch_idx + 1] - kv_sequence_start
        q_block_idx = task_in_batch // q_heads
        head_idx = task_in_batch - q_block_idx * q_heads
        kv_head_idx = head_idx // group_size
        mask_head_idx = c0
        if mask_heads != 1:
            mask_head_idx = head_idx
        mask_row = (
            batch_idx * mask_heads + mask_head_idx
        ) * q_block_count + q_block_idx

        q_tile_size = tla.as_numeric(q_block)
        q_remaining = q_length - q_block_idx * q_block
        if q_remaining < q_block:
            q_tile_size = q_remaining

        q_token_stride = head_dim
        kv_token_stride = head_dim
        q_base = (batch_idx * q_heads + head_idx) * q_sequence_length * head_dim
        k_base = (batch_idx * kv_heads + kv_head_idx) * kv_sequence_length * head_dim
        if use_varlen != 0:
            q_token_stride = q_heads * head_dim
            kv_token_stride = kv_heads * head_dim
            q_base = q_sequence_start * q_token_stride + head_idx * head_dim
            k_base = kv_sequence_start * kv_token_stride + kv_head_idx * head_dim

        mem_q_block = tla.make_tensor(
            mem_q.ptr + q_base,
            tla.make_layout(
                tla.make_shape(q_length, head_dim),
                tla.make_stride(q_token_stride, 1),
            ),
        )
        mem_k_block = tla.make_tensor(
            mem_k.ptr + k_base,
            tla.make_layout(
                tla.make_shape(head_dim, kv_length),
                tla.make_stride(1, kv_token_stride),
                layoutTag=tla.arch.ColumnMajor,
            ),
        )

        with tla.cube():
            q_gm = tla.make_tensor(
                mem_q.ptr + q_base + q_block_idx * q_block * q_token_stride,
                tla.make_layout(
                    tla.make_shape(q_tile_size, head_dim),
                    tla.make_stride(q_token_stride, 1),
                ),
            )
            l1_q_ref = tla.make_tensor(
                l1_q_ptr,
                tla.make_layout(
                    tla.make_shape(q_block, head_dim),
                    tla.make_stride(head_dim, 1),
                ),
            )
            l1_q = tla.make_tensor_like(l1_q_ptr, l1_q_ref)
            l1_q_valid = tla.tile_view(
                l1_q,
                tla.make_shape(q_tile_size, head_dim),
                tla.make_coord(c0, c0),
            )
            tla.wait_flag(q_l0a_ready_l1)
            tla.copy(l1_q_valid, q_gm)
            tla.set_flag(q_l1_ready_l0)
            tla.wait_flag(q_l1_ready_l0)

        pipeline_drain = (
            pre_launch if task + tla.arch.block_num() >= total_tasks else c0
        )
        for kv_iter in tla.range(c0, selected_block_count + pipeline_drain, c1):
            pipeline_iter = pipeline_iter + c1
            stage_2_sparse_block = stage_1_sparse_block
            stage_1_sparse_block = stage_0_sparse_block
            stage_2_kv_iter = stage_1_kv_iter
            stage_1_kv_iter = stage_0_kv_iter
            stage_2_q_block_idx = stage_1_q_block_idx
            stage_1_q_block_idx = stage_0_q_block_idx
            stage_2_batch_idx = stage_1_batch_idx
            stage_1_batch_idx = stage_0_batch_idx
            stage_2_head_idx = stage_1_head_idx
            stage_1_head_idx = stage_0_head_idx
            stage_2_q_state_idx = stage_1_q_state_idx
            stage_1_q_state_idx = stage_0_q_state_idx
            stage_2_q_tile_size = stage_1_q_tile_size
            stage_1_q_tile_size = stage_0_q_tile_size
            stage_2_kv_tile_size = stage_1_kv_tile_size
            stage_1_kv_tile_size = stage_0_kv_tile_size
            if kv_iter < selected_block_count:
                qk_stage_idx = qk_stage_count
                ubSBufId = qk_stage_idx % ub_s_otmp_buf_stages
                l1PBufId_qk = qk_stage_idx % p_l1_buf
                l1BBufId = qk_stage_idx % k_l1_buf
                l0CBufId = qk_stage_idx % l0_stages

                sparse_kv_block = block_indices[mask_row, kv_iter]
                kv_tile_size = tla.as_numeric(kv_block)
                kv_remaining = kv_length - sparse_kv_block * kv_block
                if kv_remaining < kv_block:
                    kv_tile_size = kv_remaining
                stage_0_sparse_block = sparse_kv_block
                stage_0_kv_iter = kv_iter
                stage_0_q_block_idx = q_block_idx
                stage_0_batch_idx = batch_idx
                stage_0_head_idx = head_idx
                stage_0_q_state_idx = q_state_idx
                stage_0_q_tile_size = q_tile_size
                stage_0_kv_tile_size = kv_tile_size

                # L0A/L0B ping/pong flag
                prefixSumL0AStages = (
                    qk_stage_count * mm1_l0a_total_stages
                    + pv_stage_count * mm2_l0a_total_stages
                )
                prefixSumL0BStages = (
                    qk_stage_count * mm1_l0b_total_stages
                    + pv_stage_count * mm2_l0b_total_stages
                )

                with tla.cube():
                    l1_k_ptr = l1_k_ping_ptr if l1BBufId == c0 else l1_k_pong_ptr
                    l0_s_ptr = l0_s_ping_ptr if l0CBufId == c0 else l0_s_pong_ptr
                    ub_s_ptr = ub_s_ping_ptr if ubSBufId == c0 else ub_s_pong_ptr

                    k_gm = tla.make_tensor(
                        mem_k.ptr
                        + k_base
                        + sparse_kv_block * kv_block * kv_token_stride,
                        tla.make_layout(
                            tla.make_shape(head_dim, kv_tile_size),
                            tla.make_stride(1, kv_token_stride),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    q_gm = tla.make_tensor(
                        mem_q.ptr + q_base + q_block_idx * q_block * q_token_stride,
                        tla.make_layout(
                            tla.make_shape(q_tile_size, head_dim),
                            tla.make_stride(q_token_stride, 1),
                        ),
                    )
                    l1_q_ref = tla.make_tensor(
                        l1_q_ptr,
                        tla.make_layout(
                            tla.make_shape(q_block, head_dim),
                            tla.make_stride(head_dim, 1),
                        ),
                    )
                    l1_q = tla.make_tensor_like(l1_q_ptr, l1_q_ref)
                    l1_k_ref = tla.make_tensor(
                        l1_k_ptr,
                        tla.make_layout(
                            tla.make_shape(head_dim, kv_block),
                            tla.make_stride(1, head_dim),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    l1_k = tla.make_tensor_like(
                        l1_k_ptr, l1_k_ref, layoutTag=tla.arch.nZ
                    )
                    l1_k_valid = tla.tile_view(
                        l1_k,
                        tla.make_shape(head_dim, kv_tile_size),
                        tla.make_coord(c0, c0),
                    )

                    # QK L0: 扁平化（QK_NL0_LOOP_NUM=QK_ML0_LOOP_NUM=QK_KL0_LOOP_NUM=1）
                    l0c_s_ref = tla.make_tensor(
                        l0_s_ptr,
                        tla.make_layout(
                            tla.make_shape(q_block, kv_block),
                            tla.make_stride(kv_block, 1),
                        ),
                    )
                    l0c_s = tla.make_tensor_like(
                        l0_s_ptr, l0c_s_ref, tla.arch.L0Clayout
                    )
                    s_ub = tla.make_tensor(
                        ub_s_ptr,
                        tla.make_layout(
                            tla.make_shape(q_block, kv_block),
                            tla.make_stride(kv_block, 1),
                        ),
                    )
                    ub_s_fix = tla.tile_view(
                        s_ub,
                        tla.make_shape(q_block, kv_block),
                        tla.make_coord(c0, c0),
                    )

                    if l1BBufId == c0:
                        tla.wait_flag(k_l0b_ready_l1_0)
                    else:
                        tla.wait_flag(k_l0b_ready_l1_1)
                    tla.copy(l1_k_valid, k_gm)
                    if l1BBufId == c0:
                        tla.set_flag(k_l1_ready_l0_0)
                        tla.wait_flag(k_l1_ready_l0_0)
                    else:
                        tla.set_flag(k_l1_ready_l0_1)
                        tla.wait_flag(k_l1_ready_l0_1)

                    l0ABufId = prefixSumL0AStages % l0_stages
                    l0BBufId_l0 = prefixSumL0BStages % l0_stages
                    l0_q_ptr = l0a_ping_ptr if l0ABufId == c0 else l0a_pong_ptr
                    l0_k_ptr = l0b_ping_ptr if l0BBufId_l0 == c0 else l0b_pong_ptr
                    l0_q = tla.make_tensor_like(l0_q_ptr, l1_q)
                    l0_k = tla.make_tensor_like(l0_k_ptr, l1_k)

                    if l0BBufId_l0 == c0:
                        tla.wait_flag(mmad_ready_l0b_0)
                    else:
                        tla.wait_flag(mmad_ready_l0b_1)
                    tla.copy(l0_k, l1_k)
                    if l0BBufId_l0 == c0:
                        tla.set_flag(l0b_ready_mmad_0)
                    else:
                        tla.set_flag(l0b_ready_mmad_1)
                    if l1BBufId == c0:
                        tla.set_flag(k_l0b_ready_l1_0)
                    else:
                        tla.set_flag(k_l0b_ready_l1_1)

                    if l0ABufId == c0:
                        tla.wait_flag(mmad_ready_l0a_0)
                    else:
                        tla.wait_flag(mmad_ready_l0a_1)
                    tla.copy(l0_q, l1_q)
                    if l0ABufId == c0:
                        tla.set_flag(l0a_ready_mmad_0)
                    else:
                        tla.set_flag(l0a_ready_mmad_1)

                    if l0ABufId == c0:
                        tla.wait_flag(l0a_ready_mmad_0)
                    else:
                        tla.wait_flag(l0a_ready_mmad_1)
                    if l0BBufId_l0 == c0:
                        tla.wait_flag(l0b_ready_mmad_0)
                    else:
                        tla.wait_flag(l0b_ready_mmad_1)
                    if l0CBufId == c0:
                        tla.wait_flag(fix_ready_mmad_qk_0)
                    else:
                        tla.wait_flag(fix_ready_mmad_qk_1)
                    tla.mmad(l0c_s, l0_q, l0_k, init_c=True)
                    if l0ABufId == c0:
                        tla.set_flag(mmad_ready_l0a_0)
                    else:
                        tla.set_flag(mmad_ready_l0a_1)
                    if l0BBufId_l0 == c0:
                        tla.set_flag(mmad_ready_l0b_0)
                    else:
                        tla.set_flag(mmad_ready_l0b_1)

                    if ubSBufId == c0:
                        tla.cross_core_wait_flag(qk_ready_0, tla.arch.FIX)
                    else:
                        tla.cross_core_wait_flag(qk_ready_1, tla.arch.FIX)
                    if l0CBufId == c0:
                        tla.set_flag(mmad_ready_fix_qk_0)
                        tla.wait_flag(mmad_ready_fix_qk_0)
                    else:
                        tla.set_flag(mmad_ready_fix_qk_1)
                        tla.wait_flag(mmad_ready_fix_qk_1)
                    tla.copy(
                        ub_s_fix,
                        l0c_s,
                        tla.params.CopyL0C2DstParams(
                            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                        ),
                    )
                    if l0CBufId == c0:
                        tla.set_flag(fix_ready_mmad_qk_0)
                    else:
                        tla.set_flag(fix_ready_mmad_qk_1)
                    if ubSBufId == c0:
                        tla.cross_core_set_flag(qk_ready_0, tla.arch.FIX)
                    else:
                        tla.cross_core_set_flag(qk_ready_1, tla.arch.FIX)
                    if kv_iter == (selected_block_count - c1):
                        tla.set_flag(q_l0a_ready_l1)

                with tla.vector():
                    if ubSBufId == c0:
                        tla.cross_core_wait_flag(qk_ready_0, tla.arch.VECTOR)
                    else:
                        tla.cross_core_wait_flag(qk_ready_1, tla.arch.VECTOR)
                    if ubSBufId == c0:
                        tla.wait_flag(mte3_ready_softmax_0)
                    else:
                        tla.wait_flag(mte3_ready_softmax_1)

                    vec_idx = tla.arch.sub_block_idx()
                    update = kv_iter != c0

                    # SPLIT_M transfers the fixed 128-row tile in 64-row halves,
                    # even when the logical query tail contains fewer rows.
                    q_tile_size_half = q_block_sub
                    q_tile_size_sub = q_tile_size - vec_idx * q_block_sub
                    if q_tile_size_sub > q_block_sub:
                        q_tile_size_sub = tla.as_numeric(q_block_sub)
                    if q_tile_size_sub < c0:
                        q_tile_size_sub = tla.as_numeric(c0)

                    ub_s_ptr = ub_s_ping_ptr if ubSBufId == c0 else ub_s_pong_ptr
                    ub_p_f16_ptr = (
                        ub_p_f16_ping_ptr if ubSBufId == c0 else ub_p_f16_pong_ptr
                    )
                    l1_p_ptr_qk = l1_p_0_ptr
                    if l1PBufId_qk == c0:
                        l1_p_ptr_qk = l1_p_0_ptr
                    elif l1PBufId_qk == c1:
                        l1_p_ptr_qk = l1_p_1_ptr
                    else:
                        l1_p_ptr_qk = l1_p_2_ptr
                    ub_exp_max_ptr_qk = ub_exp_max_0_ptr
                    if l1PBufId_qk == c0:
                        ub_exp_max_ptr_qk = ub_exp_max_0_ptr
                    elif l1PBufId_qk == c1:
                        ub_exp_max_ptr_qk = ub_exp_max_1_ptr
                    else:
                        ub_exp_max_ptr_qk = ub_exp_max_2_ptr

                    ub_s = tla.make_tensor(
                        ub_s_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub, kv_block),
                            tla.make_stride(kv_block, 1),
                        ),
                    )

                    ub_now_max = tla.make_tensor(
                        ub_now_max_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub), tla.make_stride(1)
                        ),
                    )
                    ub_last_max = tla.make_tensor(
                        ub_last_max_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub), tla.make_stride(1)
                        ),
                    )
                    ub_sum = tla.make_tensor(
                        ub_sum_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub), tla.make_stride(1)
                        ),
                    )
                    ub_exp_sum = tla.make_tensor(
                        ub_exp_sum_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub), tla.make_stride(1)
                        ),
                    )
                    ub_exp_max_qk = tla.make_tensor(
                        ub_exp_max_ptr_qk,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub), tla.make_stride(1)
                        ),
                    )
                    # P UB layout: RowMajor (65, 128), zN
                    ub_p_f16 = tla.make_tensor(
                        ub_p_f16_ptr,
                        tla.make_layout(
                            tla.make_shape(q_block_sub + 1, kv_block),
                            tla.make_stride(kv_block, 1),
                        ),
                    )
                    ub_p_zN_full = tla.make_tensor_like(
                        ub_p_f16_ptr, ub_p_f16, tla.arch.zNUnAlign
                    )
                    ub_p_zN = tla.tile_view(
                        ub_p_zN_full,
                        tla.make_shape(q_tile_size_sub, kv_tile_size),
                        tla.make_coord(c0, c0),
                    )
                    ub_tmp = tla.make_tensor(
                        ub_tmp_ptr,
                        tla.make_layout(
                            tla.make_shape(2 * vl_float_ele), tla.make_stride(1)
                        ),
                    )

                    tailN = (kv_tile_size - 1) % vl_float_ele + 1
                    if kv_iter == c0:
                        if kv_tile_size > vl_float_ele:
                            with tla.vec.func(mode="simd"):
                                pregFull = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=tla.Float32
                                )
                                preg_all_b16 = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=io_dtype
                                )
                                one_mask = tla.update_mask(1, dtype=tla.Float32)[0]
                                mask_tail_n = tla.update_mask(tailN, dtype=tla.Float32)[
                                    0
                                ]
                                min_reg = tla.full(min_value, dtype=tla.Float32)
                                for i0 in tla.range(q_tile_size_sub):
                                    ub_s_i0_p1 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i0, c0),
                                    )
                                    ub_s_i1_p1 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i0, c1),
                                    )
                                    ub_s_reg0_p1 = ub_s_i0_p1.load()
                                    ub_s_reg1_p1 = ub_s_i1_p1.load()
                                    ub_s_reg0_p1 = tla.mul(
                                        ub_s_reg0_p1, softmax_scale, mask=pregFull
                                    )
                                    ub_s_reg1_p1 = tla.mul(
                                        ub_s_reg1_p1, softmax_scale, mask=pregFull
                                    )
                                    ub_s_reg1_p1 = tla.where(
                                        mask_tail_n, ub_s_reg1_p1, min_reg
                                    )
                                    ub_s_i0_p1.store(ub_s_reg0_p1, mask=pregFull)
                                    ub_s_i1_p1.store(ub_s_reg1_p1, mask=pregFull)
                                    tmp_reg_p1 = tla.max(
                                        ub_s_reg0_p1, ub_s_reg1_p1, mask=pregFull
                                    )
                                    max_reg_p1 = tmp_reg_p1.reduce(
                                        tla.ReductionOp.MAX, mask=pregFull
                                    )
                                    ub_max_dst_p1 = tla.tile_view(
                                        ub_now_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i0),
                                    )
                                    ub_max_dst_p1.store(
                                        max_reg_p1,
                                        params=tla.params.UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                for i2 in tla.range(q_tile_size_sub):
                                    ub_now_max_i_p2 = tla.tile_view(
                                        ub_now_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i2),
                                    )
                                    max_reg_p2 = ub_now_max_i_p2.load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    ub_s_row_p2 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, 2 * vl_float_ele),
                                        tla.make_coord(i2, c0),
                                    )
                                    s_odd_p2, s_even_p2 = ub_s_row_p2.load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                        )
                                    )
                                    exp_odd_p2 = tla.exp(
                                        tla.sub(s_odd_p2, max_reg_p2, mask=pregFull),
                                        mask=pregFull,
                                    )
                                    exp_even_p2 = tla.exp(
                                        tla.sub(s_even_p2, max_reg_p2, mask=pregFull),
                                        mask=pregFull,
                                    )
                                    tmp_reg_p2 = tla.add(
                                        exp_odd_p2, exp_even_p2, mask=pregFull
                                    )
                                    block_sum_p2 = tmp_reg_p2.reduce(
                                        tla.ReductionOp.ADD, mask=pregFull
                                    )
                                    ub_sum_i_p2 = tla.tile_view(
                                        ub_sum, tla.make_shape(1), tla.make_coord(i2)
                                    )
                                    ub_sum_i_p2.store(
                                        block_sum_p2,
                                        params=tla.params.UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                    p_even_p2 = exp_even_p2.to(
                                        io_dtype, cast_trait_one, mask=pregFull
                                    )
                                    p_odd_p2 = exp_odd_p2.to(
                                        io_dtype, cast_trait_zero, mask=pregFull
                                    )
                                    p_zn_p2 = tla.bitwise_or(
                                        p_even_p2, p_odd_p2, mask=preg_all_b16
                                    )
                                    ub_p_zn_i0_p2 = tla.tile_view(
                                        ub_p_zN,
                                        tla.make_shape(1, 2 * vl_float_ele),
                                        tla.make_coord(i2, c0),
                                    )
                                    ub_p_zn_i0_p2.store(
                                        p_zn_p2,
                                        params=tla.params.BlockStoreParams(
                                            block_stride=q_block_sub + 1
                                        ),
                                        mask=preg_all_b16,
                                    )
                        else:
                            with tla.vec.func(mode="simd"):
                                pregFull = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=tla.Float32
                                )
                                preg_all_b16 = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=io_dtype
                                )
                                one_mask = tla.update_mask(1, dtype=tla.Float32)[0]
                                mask_tail_n = tla.update_mask(tailN, dtype=tla.Float32)[
                                    0
                                ]
                                min_reg = tla.full(min_value, dtype=tla.Float32)
                                for i0 in tla.range(q_tile_size_sub):
                                    ub_s_i0_p1 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i0, c0),
                                    )
                                    ub_s_reg0_p1 = ub_s_i0_p1.load()
                                    ub_s_reg0_p1 = tla.mul(
                                        ub_s_reg0_p1, softmax_scale, mask=pregFull
                                    )
                                    ub_s_reg0_p1 = tla.where(
                                        mask_tail_n, ub_s_reg0_p1, min_reg
                                    )
                                    ub_s_i0_p1.store(ub_s_reg0_p1, mask=pregFull)
                                    max_reg_p1 = ub_s_reg0_p1.reduce(
                                        tla.ReductionOp.MAX, mask=pregFull
                                    )
                                    ub_max_dst_p1 = tla.tile_view(
                                        ub_now_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i0),
                                    )
                                    ub_max_dst_p1.store(
                                        max_reg_p1,
                                        params=tla.params.UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                for i2 in tla.range(q_tile_size_sub):
                                    ub_now_max_i_p2 = tla.tile_view(
                                        ub_now_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i2),
                                    )
                                    max_reg_p2 = ub_now_max_i_p2.load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    ub_s_i0_p2 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i2, c0),
                                    )
                                    ub_s_reg0_p2 = ub_s_i0_p2.load()
                                    exp_reg0_p2 = tla.exp(
                                        tla.sub(
                                            ub_s_reg0_p2, max_reg_p2, mask=pregFull
                                        ),
                                        mask=pregFull,
                                    )
                                    block_sum_p2 = exp_reg0_p2.reduce(
                                        tla.ReductionOp.ADD, mask=pregFull
                                    )
                                    ub_sum_i_p2 = tla.tile_view(
                                        ub_sum, tla.make_shape(1), tla.make_coord(i2)
                                    )
                                    ub_sum_i_p2.store(
                                        block_sum_p2,
                                        params=tla.params.UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                    p_reg0_p2 = exp_reg0_p2.to(
                                        io_dtype, cast_trait_zero, mask=pregFull
                                    )
                                    # PV consumes all 128 lanes; the upper 64 are padding.
                                    r0_qk_p2, _ = tla.deinterleave(
                                        p_reg0_p2, tla.full(0, dtype=io_dtype)
                                    )
                                    ub_p_zn_i0_p2 = tla.tile_view(
                                        ub_p_zN,
                                        tla.make_shape(1, 2 * vl_float_ele),
                                        tla.make_coord(i2, c0),
                                    )
                                    ub_p_zn_i0_p2.store(
                                        r0_qk_p2,
                                        params=tla.params.BlockStoreParams(
                                            block_stride=q_block_sub + 1
                                        ),
                                        mask=preg_all_b16,
                                    )
                    else:
                        if kv_tile_size > vl_float_ele:
                            with tla.vec.func(mode="simd"):
                                pregFull = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=tla.Float32
                                )
                                preg_all_b16 = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=io_dtype
                                )
                                one_mask = tla.update_mask(1, dtype=tla.Float32)[0]
                                mask_tail_n = tla.update_mask(tailN, dtype=tla.Float32)[
                                    0
                                ]
                                min_reg = tla.full(min_value, dtype=tla.Float32)
                                for i0 in tla.range(q_tile_size_sub):
                                    ub_s_i0_p1 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i0, c0),
                                    )
                                    ub_s_i1_p1 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i0, c1),
                                    )
                                    ub_s_reg0_p1 = ub_s_i0_p1.load()
                                    ub_s_reg1_p1 = ub_s_i1_p1.load()
                                    ub_s_reg0_p1 = tla.mul(
                                        ub_s_reg0_p1, softmax_scale, mask=pregFull
                                    )
                                    ub_s_reg1_p1 = tla.mul(
                                        ub_s_reg1_p1, softmax_scale, mask=pregFull
                                    )
                                    ub_s_reg1_p1 = tla.where(
                                        mask_tail_n, ub_s_reg1_p1, min_reg
                                    )
                                    ub_s_i0_p1.store(ub_s_reg0_p1, mask=pregFull)
                                    ub_s_i1_p1.store(ub_s_reg1_p1, mask=pregFull)
                                    tmp_reg_p1 = tla.max(
                                        ub_s_reg0_p1, ub_s_reg1_p1, mask=pregFull
                                    )
                                    max_reg_p1 = tmp_reg_p1.reduce(
                                        tla.ReductionOp.MAX, mask=pregFull
                                    )
                                    ub_max_dst_p1 = tla.tile_view(
                                        ub_last_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i0),
                                    )
                                    ub_max_dst_p1.store(
                                        max_reg_p1,
                                        params=tla.params.UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                ub_now_max_vu = tla.tile_view(
                                    ub_now_max,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                )
                                ub_last_max_vu = tla.tile_view(
                                    ub_last_max,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                )
                                ub_sum_vu = tla.tile_view(
                                    ub_sum,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                )
                                ub_exp_max_vu = tla.tile_view(
                                    ub_exp_max_qk,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                )
                                now_max_vu = (
                                    ub_now_max_vu.load()
                                )  # 旧 global max（上一轮存入）
                                blk_max_vu = (
                                    ub_last_max_vu.load()
                                )  # 本块 max（i0 存入）
                                global_max_vu = tla.max(
                                    now_max_vu, blk_max_vu, mask=pregFull
                                )
                                dm_vu = tla.exp(
                                    tla.sub(now_max_vu, global_max_vu, mask=pregFull),
                                    mask=pregFull,
                                )  # exp(旧-新)
                                sum_vu = ub_sum_vu.load()  # 旧 global sum
                                update_vu = tla.mul(sum_vu, dm_vu, mask=pregFull)
                                ub_now_max_vu.store(global_max_vu, mask=pregFull)
                                ub_exp_max_vu.store(dm_vu, mask=pregFull)
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                for i2 in tla.range(q_tile_size_sub):
                                    ub_now_max_i_p2 = tla.tile_view(
                                        ub_now_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i2),
                                    )
                                    max_reg_p2 = ub_now_max_i_p2.load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    ub_s_row_p2 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, 2 * vl_float_ele),
                                        tla.make_coord(i2, c0),
                                    )
                                    s_odd_p2, s_even_p2 = ub_s_row_p2.load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                        )
                                    )
                                    exp_odd_p2 = tla.exp(
                                        tla.sub(s_odd_p2, max_reg_p2, mask=pregFull),
                                        mask=pregFull,
                                    )
                                    exp_even_p2 = tla.exp(
                                        tla.sub(s_even_p2, max_reg_p2, mask=pregFull),
                                        mask=pregFull,
                                    )
                                    tmp_reg_p2 = tla.add(
                                        exp_odd_p2, exp_even_p2, mask=pregFull
                                    )
                                    block_sum_p2 = tmp_reg_p2.reduce(
                                        tla.ReductionOp.ADD, mask=pregFull
                                    )
                                    ub_exp_sum_i_p2 = tla.tile_view(
                                        ub_exp_sum,
                                        tla.make_shape(1),
                                        tla.make_coord(i2),
                                    )
                                    ub_exp_sum_i_p2.store(
                                        block_sum_p2,
                                        params=tla.params.UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                    p_even_p2 = exp_even_p2.to(
                                        io_dtype, cast_trait_one, mask=pregFull
                                    )
                                    p_odd_p2 = exp_odd_p2.to(
                                        io_dtype, cast_trait_zero, mask=pregFull
                                    )
                                    p_zn_p2 = tla.bitwise_or(
                                        p_even_p2, p_odd_p2, mask=preg_all_b16
                                    )
                                    ub_p_zn_i0_p2 = tla.tile_view(
                                        ub_p_zN,
                                        tla.make_shape(1, 2 * vl_float_ele),
                                        tla.make_coord(i2, c0),
                                    )
                                    ub_p_zn_i0_p2.store(
                                        p_zn_p2,
                                        params=tla.params.BlockStoreParams(
                                            block_stride=q_block_sub + 1
                                        ),
                                        mask=preg_all_b16,
                                    )
                                # 向量化 final add（替代 i3 逐行 sum-merge；update_vu + 本块 sum → 新 global sum）
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                ub_exp_sum_vf = tla.tile_view(
                                    ub_exp_sum,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                )
                                blk_sum_vf = ub_exp_sum_vf.load()
                                new_sum_vf = tla.add(
                                    update_vu, blk_sum_vf, mask=pregFull
                                )
                                ub_sum_vf = tla.tile_view(
                                    ub_sum,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                )
                                ub_sum_vf.store(new_sum_vf, mask=pregFull)
                        else:
                            with tla.vec.func(mode="simd"):
                                pregFull = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=tla.Float32
                                )
                                preg_all_b16 = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=io_dtype
                                )
                                one_mask = tla.update_mask(1, dtype=tla.Float32)[0]
                                mask_tail_n = tla.update_mask(tailN, dtype=tla.Float32)[
                                    0
                                ]
                                min_reg = tla.full(min_value, dtype=tla.Float32)
                                for i0 in tla.range(q_tile_size_sub):
                                    ub_s_i0_p1 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i0, c0),
                                    )
                                    ub_s_reg0_p1 = ub_s_i0_p1.load()
                                    ub_s_reg0_p1 = tla.mul(
                                        ub_s_reg0_p1, softmax_scale, mask=pregFull
                                    )
                                    ub_s_reg0_p1 = tla.where(
                                        mask_tail_n, ub_s_reg0_p1, min_reg
                                    )
                                    ub_s_i0_p1.store(ub_s_reg0_p1, mask=pregFull)
                                    max_reg_p1 = ub_s_reg0_p1.reduce(
                                        tla.ReductionOp.MAX, mask=pregFull
                                    )
                                    ub_max_dst_p1 = tla.tile_view(
                                        ub_last_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i0),
                                    )
                                    ub_max_dst_p1.store(
                                        max_reg_p1,
                                        params=tla.params.UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                ub_now_max_vu = tla.tile_view(
                                    ub_now_max,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                )
                                ub_last_max_vu = tla.tile_view(
                                    ub_last_max,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                )
                                ub_sum_vu = tla.tile_view(
                                    ub_sum,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                )
                                ub_exp_max_vu = tla.tile_view(
                                    ub_exp_max_qk,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                )
                                now_max_vu = (
                                    ub_now_max_vu.load()
                                )  # 旧 global max（上一轮存入）
                                blk_max_vu = (
                                    ub_last_max_vu.load()
                                )  # 本块 max（i0 存入）
                                global_max_vu = tla.max(
                                    now_max_vu, blk_max_vu, mask=pregFull
                                )
                                dm_vu = tla.exp(
                                    tla.sub(now_max_vu, global_max_vu, mask=pregFull),
                                    mask=pregFull,
                                )  # exp(旧-新)
                                sum_vu = ub_sum_vu.load()  # 旧 global sum
                                update_vu = tla.mul(sum_vu, dm_vu, mask=pregFull)
                                ub_now_max_vu.store(global_max_vu, mask=pregFull)
                                ub_exp_max_vu.store(dm_vu, mask=pregFull)
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                for i2 in tla.range(q_tile_size_sub):
                                    ub_now_max_i_p2 = tla.tile_view(
                                        ub_now_max,
                                        tla.make_shape(1),
                                        tla.make_coord(i2),
                                    )
                                    max_reg_p2 = ub_now_max_i_p2.load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    ub_s_i0_p2 = tla.tile_view(
                                        ub_s,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i2, c0),
                                    )
                                    ub_s_reg0_p2 = ub_s_i0_p2.load()
                                    exp_reg0_p2 = tla.exp(
                                        tla.sub(
                                            ub_s_reg0_p2, max_reg_p2, mask=pregFull
                                        ),
                                        mask=pregFull,
                                    )
                                    block_sum_p2 = exp_reg0_p2.reduce(
                                        tla.ReductionOp.ADD, mask=pregFull
                                    )
                                    ub_exp_sum_i_p2 = tla.tile_view(
                                        ub_exp_sum,
                                        tla.make_shape(1),
                                        tla.make_coord(i2),
                                    )
                                    ub_exp_sum_i_p2.store(
                                        block_sum_p2,
                                        params=tla.params.UnalignStoreParams(),
                                        mask=one_mask,
                                    )
                                    p_reg0_p2 = exp_reg0_p2.to(
                                        io_dtype, cast_trait_zero, mask=pregFull
                                    )
                                    # PV consumes all 128 lanes; the upper 64 are padding.
                                    r0_qk_p2, _ = tla.deinterleave(
                                        p_reg0_p2, tla.full(0, dtype=io_dtype)
                                    )
                                    ub_p_zn_i0_p2 = tla.tile_view(
                                        ub_p_zN,
                                        tla.make_shape(1, 2 * vl_float_ele),
                                        tla.make_coord(i2, c0),
                                    )
                                    ub_p_zn_i0_p2.store(
                                        r0_qk_p2,
                                        params=tla.params.BlockStoreParams(
                                            block_stride=q_block_sub + 1
                                        ),
                                        mask=preg_all_b16,
                                    )
                                # 向量化 final add（替代 i3 逐行 sum-merge；update_vu + 本块 sum → 新 global sum）
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                ub_exp_sum_vf = tla.tile_view(
                                    ub_exp_sum,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                )
                                blk_sum_vf = ub_exp_sum_vf.load()
                                new_sum_vf = tla.add(
                                    update_vu, blk_sum_vf, mask=pregFull
                                )
                                ub_sum_vf = tla.tile_view(
                                    ub_sum,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                )
                                ub_sum_vf.store(new_sum_vf, mask=pregFull)

                    if tla.const_expr(return_lse):
                        # Capture the final max here: later QK tasks may overwrite
                        # ub_now_max before this task's delayed PV finishes.
                        if kv_iter == selected_block_count - c1 and q_tile_size_sub > 0:
                            tla.wait_flag(lse_free)
                            ub_lse = tla.make_tensor(
                                ub_lse_ptr,
                                tla.make_layout(
                                    tla.make_shape(q_block_sub * 8), tla.make_stride(1)
                                ),
                            )
                            ub_lse_compact = tla.make_tensor(
                                ub_lse_ptr + q_block_sub * 8,
                                tla.make_layout(
                                    tla.make_shape(vl_float_ele), tla.make_stride(1)
                                ),
                            )
                            with tla.vec.func(mode="simd"):
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                lse_mask = tla.update_mask(
                                    q_tile_size_sub, dtype=tla.Float32
                                )[0]
                                lse_max = tla.tile_view(
                                    ub_now_max,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                ).load()
                                lse_sum = tla.tile_view(
                                    ub_sum,
                                    tla.make_shape(vl_float_ele),
                                    tla.make_coord(0),
                                ).load()
                                lse_reg = tla.add(
                                    lse_max,
                                    tla.log(lse_sum, mask=lse_mask),
                                    mask=lse_mask,
                                )
                                if use_varlen != 0:
                                    # Replicate each scalar into an aligned 8-float row;
                                    # DMA writes only its first element to [token, head].
                                    ub_lse_compact.store(lse_reg, mask=lse_mask)
                                    tla.local_mem_bar(
                                        tla.params.MemType.VEC_STORE,
                                        tla.params.MemType.VEC_LOAD,
                                    )
                                    lanes = tla.arange(0, dtype=tla.Int32)
                                    indices = tla.shift_right(lanes, 3)
                                    for lse_chunk in tla.range(8):
                                        expanded = tla.gather(
                                            ub_lse_compact,
                                            tla.add(indices, lse_chunk * 8),
                                        )
                                        tla.tile_view(
                                            ub_lse,
                                            tla.make_shape(vl_float_ele),
                                            tla.make_coord(lse_chunk),
                                        ).store(expanded)
                                else:
                                    tla.tile_view(
                                        ub_lse,
                                        tla.make_shape(vl_float_ele),
                                        tla.make_coord(0),
                                    ).store(lse_reg, mask=lse_mask)
                            tla.set_flag(lse_ready)
                            tla.wait_flag(lse_ready)
                            lse_row = q_block_idx * q_block + vec_idx * q_tile_size_half
                            if use_varlen != 0:
                                lse_gm = tla.make_tensor(
                                    mem_lse.ptr
                                    + (q_sequence_start + lse_row) * q_heads
                                    + head_idx,
                                    tla.make_layout(
                                        tla.make_shape(q_tile_size_sub, 1),
                                        tla.make_stride(q_heads, 1),
                                    ),
                                )
                                lse_src = tla.make_tensor(
                                    ub_lse_ptr,
                                    tla.make_layout(
                                        tla.make_shape(q_tile_size_sub, 1),
                                        tla.make_stride(8, 1),
                                    ),
                                )
                                tla.copy(lse_gm, lse_src)
                            else:
                                lse_gm = tla.make_tensor(
                                    mem_lse.ptr
                                    + (batch_idx * q_heads + head_idx)
                                    * q_sequence_length
                                    + lse_row,
                                    tla.make_layout(
                                        tla.make_shape(q_tile_size_sub),
                                        tla.make_stride(1),
                                    ),
                                )
                                lse_src = tla.make_tensor(
                                    ub_lse_ptr,
                                    tla.make_layout(
                                        tla.make_shape(q_tile_size_sub),
                                        tla.make_stride(1),
                                    ),
                                )
                                tla.copy(lse_gm, lse_src)
                            tla.set_flag(lse_free)

                    if ubSBufId == c0:
                        tla.set_flag(v_mte3_0)
                        tla.wait_flag(v_mte3_0)
                    else:
                        tla.set_flag(v_mte3_1)
                        tla.wait_flag(v_mte3_1)
                    if ubSBufId == c0:
                        tla.cross_core_set_flag(qk_ready_0, tla.arch.VECTOR)
                    else:
                        tla.cross_core_set_flag(qk_ready_1, tla.arch.VECTOR)

                    # softmaxReadyFlag = l1PBufId + 2; 等 PV 用完上一轮 P buffer
                    if l1PBufId_qk == c0:
                        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
                    elif l1PBufId_qk == c1:
                        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
                    else:
                        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)
                    ub_p_zN_copy = tla.tile_view(
                        ub_p_zN_full,
                        tla.make_shape(q_block_sub, kv_block),
                        tla.make_coord(c0, c0),
                    )
                    l1_p_ref = tla.make_tensor(
                        l1_p_ptr_qk,
                        tla.make_layout(
                            tla.make_shape(q_block, kv_block),
                            tla.make_stride(kv_block, 1),
                        ),
                    )
                    l1_p_qk = tla.make_tensor_like(l1_p_ptr_qk, l1_p_ref, tla.arch.zN)
                    l1_p_sub = tla.tile_view(
                        l1_p_qk,
                        tla.make_shape(q_block_sub, kv_block),
                        tla.make_coord(vec_idx, c0),
                    )
                    tla.copy(l1_p_sub, ub_p_zN_copy)
                    # 释放 ub_s 给下一轮 softmax
                    if ubSBufId == c0:
                        tla.set_flag(mte3_ready_softmax_0)
                    else:
                        tla.set_flag(mte3_ready_softmax_1)

                    if l1PBufId_qk == c0:
                        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE3)
                    elif l1PBufId_qk == c1:
                        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE3)
                    else:
                        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE3)

                qk_stage_count = qk_stage_count + c1

            if pipeline_iter >= pre_launch:
                kv_pv = stage_2_kv_iter
                q_block_idx_pv = stage_2_q_block_idx
                batch_idx_pv = stage_2_batch_idx
                head_idx_pv = stage_2_head_idx
                kv_head_idx_pv = head_idx_pv // group_size
                q_state_idx_pv = stage_2_q_state_idx
                q_tile_size_pv = stage_2_q_tile_size
                kv_tile_size_pv = stage_2_kv_tile_size
                ub_sum_ptr_pv = ub_sum_0_ptr
                if q_state_idx_pv == c1:
                    ub_sum_ptr_pv = ub_sum_1_ptr
                elif q_state_idx_pv == 2:
                    ub_sum_ptr_pv = ub_sum_2_ptr

                sparse_v_block = stage_2_sparse_block
                ubOTmpBufId = pv_stage_count % ub_s_otmp_buf_stages
                l1PBufId_pv = pv_stage_count % p_l1_buf
                l1BBufId_pv = pv_stage_count % v_l1_buf
                l0CBufId_pv = pv_stage_count % l0_stages

                prefixSumL0AStages_pv = (
                    qk_stage_count * mm1_l0a_total_stages
                    + pv_stage_count * mm2_l0a_total_stages
                )
                prefixSumL0BStages_pv = (
                    qk_stage_count * mm1_l0b_total_stages
                    + pv_stage_count * mm2_l0b_total_stages
                )

                q_sequence_start_pv = tla.as_numeric(0)
                kv_sequence_start_pv = tla.as_numeric(0)
                q_length_pv = q_sequence_length
                kv_length_pv = kv_sequence_length
                q_token_stride_pv = head_dim
                kv_token_stride_pv = head_dim
                v_base_pv = (
                    (batch_idx_pv * kv_heads + kv_head_idx_pv)
                    * kv_sequence_length
                    * head_dim
                )
                o_base_pv = (
                    (batch_idx_pv * q_heads + head_idx_pv)
                    * q_sequence_length
                    * head_dim
                )
                if use_varlen != 0:
                    q_sequence_start_pv = cu_seqlens_q[batch_idx_pv]
                    q_length_pv = cu_seqlens_q[batch_idx_pv + 1] - q_sequence_start_pv
                    kv_sequence_start_pv = cu_seqlens_k[batch_idx_pv]
                    kv_length_pv = cu_seqlens_k[batch_idx_pv + 1] - kv_sequence_start_pv
                    q_token_stride_pv = q_heads * head_dim
                    kv_token_stride_pv = kv_heads * head_dim
                    v_base_pv = (
                        kv_sequence_start_pv * kv_token_stride_pv
                        + kv_head_idx_pv * head_dim
                    )
                    o_base_pv = (
                        q_sequence_start_pv * q_token_stride_pv + head_idx_pv * head_dim
                    )

                mem_v_block_pv = tla.make_tensor(
                    mem_v.ptr + v_base_pv,
                    tla.make_layout(
                        tla.make_shape(kv_length_pv, head_dim),
                        tla.make_stride(kv_token_stride_pv, 1),
                    ),
                )
                mem_o_block_pv = tla.make_tensor(
                    mem_o.ptr + o_base_pv,
                    tla.make_layout(
                        tla.make_shape(q_length_pv, head_dim),
                        tla.make_stride(q_token_stride_pv, 1),
                    ),
                )

                with tla.cube():
                    l1_v_ptr = l1_v_ping_ptr if l1BBufId_pv == c0 else l1_v_pong_ptr
                    l0_pv_ptr = l0_pv_ping_ptr if l0CBufId_pv == c0 else l0_pv_pong_ptr

                    v_gm = tla.make_tensor(
                        mem_v.ptr
                        + v_base_pv
                        + sparse_v_block * kv_block * kv_token_stride_pv,
                        tla.make_layout(
                            tla.make_shape(kv_tile_size_pv, head_dim),
                            tla.make_stride(kv_token_stride_pv, 1),
                        ),
                    )
                    l1_v_ref = tla.make_tensor(
                        l1_v_ptr,
                        tla.make_layout(
                            tla.make_shape(kv_block, head_dim),
                            tla.make_stride(head_dim, 1),
                        ),
                    )
                    l1_v = tla.make_tensor_like(l1_v_ptr, l1_v_ref)
                    l1_v_valid = tla.tile_view(
                        l1_v,
                        tla.make_shape(kv_tile_size_pv, head_dim),
                        tla.make_coord(c0, c0),
                    )

                    if l1BBufId_pv == c0:
                        tla.wait_flag(v_l0b_ready_l1_0)
                    else:
                        tla.wait_flag(v_l0b_ready_l1_1)
                    tla.copy(l1_v_valid, v_gm)
                    if l1BBufId_pv == c0:
                        tla.set_flag(v_l1_ready_l0_0)
                        tla.wait_flag(v_l1_ready_l0_0)
                    else:
                        tla.set_flag(v_l1_ready_l0_1)
                        tla.wait_flag(v_l1_ready_l0_1)

                    if l1PBufId_pv == c0:
                        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE1)
                    elif l1PBufId_pv == c1:
                        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE1)
                    else:
                        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE1)

                    l1_p_ptr_pv = l1_p_0_ptr
                    if l1PBufId_pv == c0:
                        l1_p_ptr_pv = l1_p_0_ptr
                    elif l1PBufId_pv == c1:
                        l1_p_ptr_pv = l1_p_1_ptr
                    else:
                        l1_p_ptr_pv = l1_p_2_ptr
                    ub_p_full_pv = tla.make_tensor(
                        l1_p_ptr_pv,
                        tla.make_layout(
                            tla.make_shape(q_block, kv_block),
                            tla.make_stride(kv_block, 1),
                        ),
                    )
                    l1_p_pv = tla.make_tensor_like(
                        l1_p_ptr_pv, ub_p_full_pv, tla.arch.zN
                    )

                    # PV L0: 扁平化（PV_NL0_LOOP_NUM=PV_ML0_LOOP_NUM=PV_KL0_LOOP_NUM=1）
                    l0c_pv_ref = tla.make_tensor(
                        l0_pv_ptr,
                        tla.make_layout(
                            tla.make_shape(q_block, head_dim),
                            tla.make_stride(head_dim, 1),
                        ),
                    )
                    l0c_pv = tla.make_tensor_like(
                        l0_pv_ptr, l0c_pv_ref, tla.arch.L0Clayout
                    )

                    l0ABufId_pv = prefixSumL0AStages_pv % l0_stages
                    l0BBufId_pv = prefixSumL0BStages_pv % l0_stages
                    l0_p_ptr = l0a_ping_ptr if l0ABufId_pv == c0 else l0a_pong_ptr
                    l0_v_ptr = l0b_ping_ptr if l0BBufId_pv == c0 else l0b_pong_ptr
                    l0_p = tla.make_tensor_like(l0_p_ptr, l1_p_pv)
                    l0_v = tla.make_tensor_like(l0_v_ptr, l1_v)

                    if l0BBufId_pv == c0:
                        tla.wait_flag(mmad_ready_l0b_0)
                    else:
                        tla.wait_flag(mmad_ready_l0b_1)
                    tla.copy(l0_v, l1_v)
                    if l0BBufId_pv == c0:
                        tla.set_flag(l0b_ready_mmad_0)
                    else:
                        tla.set_flag(l0b_ready_mmad_1)
                    if l1BBufId_pv == c0:
                        tla.set_flag(v_l0b_ready_l1_0)
                    else:
                        tla.set_flag(v_l0b_ready_l1_1)

                    if l0ABufId_pv == c0:
                        tla.wait_flag(mmad_ready_l0a_0)
                    else:
                        tla.wait_flag(mmad_ready_l0a_1)
                    tla.copy(l0_p, l1_p_pv)
                    if l0ABufId_pv == c0:
                        tla.set_flag(l0a_ready_mmad_0)
                    else:
                        tla.set_flag(l0a_ready_mmad_1)
                    if l1PBufId_pv == c0:
                        tla.cross_core_set_flag(sm_ready_mm2_0, tla.arch.MTE1)
                    elif l1PBufId_pv == c1:
                        tla.cross_core_set_flag(sm_ready_mm2_1, tla.arch.MTE1)
                    else:
                        tla.cross_core_set_flag(sm_ready_mm2_2, tla.arch.MTE1)

                    if l0ABufId_pv == c0:
                        tla.wait_flag(l0a_ready_mmad_0)
                    else:
                        tla.wait_flag(l0a_ready_mmad_1)
                    if l0BBufId_pv == c0:
                        tla.wait_flag(l0b_ready_mmad_0)
                    else:
                        tla.wait_flag(l0b_ready_mmad_1)
                    if l0CBufId_pv == c0:
                        tla.wait_flag(fix_ready_mmad_pv_0)
                    else:
                        tla.wait_flag(fix_ready_mmad_pv_1)
                    tla.mmad(l0c_pv, l0_p, l0_v, init_c=True)
                    if l0ABufId_pv == c0:
                        tla.set_flag(mmad_ready_l0a_0)
                    else:
                        tla.set_flag(mmad_ready_l0a_1)
                    if l0BBufId_pv == c0:
                        tla.set_flag(mmad_ready_l0b_0)
                    else:
                        tla.set_flag(mmad_ready_l0b_1)

                    if ubOTmpBufId == c0:
                        tla.cross_core_wait_flag(pv_ready_0, tla.arch.FIX)
                    else:
                        tla.cross_core_wait_flag(pv_ready_1, tla.arch.FIX)
                    if l0CBufId_pv == c0:
                        tla.set_flag(mmad_ready_fix_pv_0)
                        tla.wait_flag(mmad_ready_fix_pv_0)
                    else:
                        tla.set_flag(mmad_ready_fix_pv_1)
                        tla.wait_flag(mmad_ready_fix_pv_1)
                    ub_pv_ptr_pv = (
                        ub_pv_ping_ptr if ubOTmpBufId == c0 else ub_pv_pong_ptr
                    )
                    ub_pv_fix = tla.make_tensor(
                        ub_pv_ptr_pv,
                        tla.make_layout(
                            tla.make_shape(q_block, head_dim),
                            tla.make_stride(head_dim, 1),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    )
                    tla.copy(
                        ub_pv_fix,
                        l0c_pv,
                        tla.params.CopyL0C2DstParams(
                            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                        ),
                    )
                    if l0CBufId_pv == c0:
                        tla.set_flag(fix_ready_mmad_pv_0)
                    else:
                        tla.set_flag(fix_ready_mmad_pv_1)
                    if ubOTmpBufId == c0:
                        tla.cross_core_set_flag(pv_ready_0, tla.arch.FIX)
                    else:
                        tla.cross_core_set_flag(pv_ready_1, tla.arch.FIX)

                with tla.vector():
                    if ubOTmpBufId == c0:
                        tla.cross_core_wait_flag(pv_ready_0, tla.arch.VECTOR)
                    else:
                        tla.cross_core_wait_flag(pv_ready_1, tla.arch.VECTOR)
                    tla.wait_flag(mte3_ready_rescale)

                    ub_pv_ptr_re = (
                        ub_pv_ping_ptr if ubOTmpBufId == c0 else ub_pv_pong_ptr
                    )
                    ub_exp_max_ptr_re = ub_exp_max_0_ptr
                    if l1PBufId_pv == c0:
                        ub_exp_max_ptr_re = ub_exp_max_0_ptr
                    elif l1PBufId_pv == c1:
                        ub_exp_max_ptr_re = ub_exp_max_1_ptr
                    else:
                        ub_exp_max_ptr_re = ub_exp_max_2_ptr

                    vec_idx_re = tla.arch.sub_block_idx()
                    is_first_pv = kv_pv == c0
                    is_last_pv = kv_pv == (selected_block_count - c1)

                    q_tile_size_half_re = q_block_sub
                    q_tile_size_sub_re = q_tile_size_pv - vec_idx_re * q_block_sub
                    if q_tile_size_sub_re > q_block_sub:
                        q_tile_size_sub_re = tla.as_numeric(q_block_sub)
                    if q_tile_size_sub_re < c0:
                        q_tile_size_sub_re = tla.as_numeric(c0)

                    ub_pv_re = tla.make_tensor(
                        ub_pv_ptr_re,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub_re, head_dim),
                            tla.make_stride(head_dim, 1),
                        ),
                    )
                    ub_acc_re = tla.make_tensor(
                        ub_acc_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub_re, head_dim),
                            tla.make_stride(head_dim, 1),
                        ),
                    )
                    ub_out_f16_re = tla.make_tensor(
                        ub_out_f16_ptr,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub_re, head_dim),
                            tla.make_stride(head_dim, 1),
                        ),
                    )
                    ub_sum_re = tla.make_tensor(
                        ub_sum_ptr_pv,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub_re), tla.make_stride(1)
                        ),
                    )
                    ub_exp_max_re = tla.make_tensor(
                        ub_exp_max_ptr_re,
                        tla.make_layout(
                            tla.make_shape(q_tile_size_sub_re), tla.make_stride(1)
                        ),
                    )

                    if is_first_pv and is_last_pv:
                        # Single block: O = PV / sum → cast → GM (1 个 vec.func + local_mem_bar)
                        with tla.vec.func(mode="simd"):
                            pregFull_re0 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            preg_all_b16_re0 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=io_dtype
                            )
                            for i_re0 in tla.range(q_tile_size_sub_re):
                                ub_sum_i_re0 = tla.tile_view(
                                    ub_sum_re, tla.make_shape(1), tla.make_coord(i_re0)
                                )
                                sum_reg_re0 = ub_sum_i_re0.load(
                                    params=tla.params.NormalLoadParams(
                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                    )
                                )
                                for j_re0 in tla.range(
                                    c0, head_dim // vl_float_ele, c1
                                ):
                                    cur_tile_re0 = tla.tile_view(
                                        ub_pv_re,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i_re0, j_re0),
                                    )
                                    cur_reg_re0 = cur_tile_re0.load()
                                    div_reg_re0 = tla.div(
                                        cur_reg_re0, sum_reg_re0, mask=pregFull_re0
                                    )
                                    acc_tile_re0 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i_re0, j_re0),
                                    )
                                    acc_tile_re0.store(div_reg_re0, mask=pregFull_re0)
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )
                            for i_re1 in tla.range(q_tile_size_sub_re):
                                for j_re1 in tla.range(
                                    c0, head_dim // (2 * vl_float_ele), c1
                                ):
                                    acc_tile_re1_0 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i_re1, 2 * j_re1),
                                    )
                                    acc_tile_re1_1 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i_re1, 2 * j_re1 + c1),
                                    )
                                    out_tile_re1 = tla.tile_view(
                                        ub_out_f16_re,
                                        tla.make_shape(1, 2 * vl_float_ele),
                                        tla.make_coord(i_re1, j_re1),
                                    )
                                    acc_reg_re1_0 = acc_tile_re1_0.load()
                                    acc_reg_re1_1 = acc_tile_re1_1.load()
                                    out_reg_re1_0 = acc_reg_re1_0.to(
                                        io_dtype, cast_trait_zero, mask=pregFull_re0
                                    )
                                    out_reg_re1_1 = acc_reg_re1_1.to(
                                        io_dtype, cast_trait_zero, mask=pregFull_re0
                                    )
                                    r0_re1, _ = tla.deinterleave(
                                        out_reg_re1_0, out_reg_re1_1
                                    )
                                    out_tile_re1.store(r0_re1, mask=preg_all_b16_re0)
                        tla.set_flag(rescale_v_mte3)
                        tla.wait_flag(rescale_v_mte3)
                        o_row_start_re0 = (
                            q_block_idx_pv * q_block + vec_idx_re * q_tile_size_half_re
                        )
                        o_sub_gm_re0 = tla.make_tensor(
                            mem_o.ptr + o_base_pv + o_row_start_re0 * q_token_stride_pv,
                            tla.make_layout(
                                tla.make_shape(q_tile_size_sub_re, head_dim),
                                tla.make_stride(q_token_stride_pv, 1),
                            ),
                        )
                        if q_tile_size_sub_re > 0:
                            tla.copy(o_sub_gm_re0, ub_out_f16_re)

                    elif is_first_pv and (not is_last_pv):
                        # First block: copy PV to acc
                        with tla.vec.func(mode="simd"):
                            pregFull_re2 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            for i_re2 in tla.range(q_tile_size_sub_re):
                                for j_re2 in tla.range(
                                    c0, head_dim // vl_float_ele, c1
                                ):
                                    cur_tile_re2 = tla.tile_view(
                                        ub_pv_re,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i_re2, j_re2),
                                    )
                                    acc_tile_re2 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i_re2, j_re2),
                                    )
                                    cur_reg_re2 = cur_tile_re2.load()
                                    acc_tile_re2.store(cur_reg_re2, mask=pregFull_re2)

                    elif (not is_first_pv) and (not is_last_pv):
                        with tla.vec.func(mode="simd"):
                            pregFull_re3 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            for i_re3 in tla.range(q_tile_size_sub_re):
                                ub_exp_max_i_re3 = tla.tile_view(
                                    ub_exp_max_re,
                                    tla.make_shape(1),
                                    tla.make_coord(i_re3),
                                )
                                exp_max_reg_re3 = ub_exp_max_i_re3.load(
                                    params=tla.params.NormalLoadParams(
                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                    )
                                )
                                for j_re3 in tla.range(
                                    c0, head_dim // vl_float_ele, c1
                                ):
                                    pre_tile_re3 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i_re3, j_re3),
                                    )
                                    cur_tile_re3 = tla.tile_view(
                                        ub_pv_re,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i_re3, j_re3),
                                    )
                                    pre_reg_re3 = pre_tile_re3.load()
                                    cur_reg_re3 = cur_tile_re3.load()
                                    mul_reg_re3 = tla.mul(
                                        exp_max_reg_re3, pre_reg_re3, mask=pregFull_re3
                                    )
                                    add_reg_re3 = tla.add(
                                        mul_reg_re3, cur_reg_re3, mask=pregFull_re3
                                    )
                                    pre_tile_re3.store(add_reg_re3, mask=pregFull_re3)

                    else:
                        # Last block: O = (O*expMax + PV) / sum → cast → GM (1 个 vec.func + local_mem_bar)
                        with tla.vec.func(mode="simd"):
                            pregFull_re4 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            preg_all_b16_re4 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=io_dtype
                            )
                            for i_re4 in tla.range(q_tile_size_sub_re):
                                ub_exp_max_i_re4 = tla.tile_view(
                                    ub_exp_max_re,
                                    tla.make_shape(1),
                                    tla.make_coord(i_re4),
                                )
                                exp_max_reg_re4 = ub_exp_max_i_re4.load(
                                    params=tla.params.NormalLoadParams(
                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                    )
                                )
                                ub_sum_i_re4 = tla.tile_view(
                                    ub_sum_re, tla.make_shape(1), tla.make_coord(i_re4)
                                )
                                sum_reg_re4 = ub_sum_i_re4.load(
                                    params=tla.params.NormalLoadParams(
                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                    )
                                )
                                for j_re4 in tla.range(
                                    c0, head_dim // vl_float_ele, c1
                                ):
                                    pre_tile_re4 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i_re4, j_re4),
                                    )
                                    cur_tile_re4 = tla.tile_view(
                                        ub_pv_re,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i_re4, j_re4),
                                    )
                                    pre_reg_re4 = pre_tile_re4.load()
                                    cur_reg_re4 = cur_tile_re4.load()
                                    mul_reg_re4 = tla.mul(
                                        exp_max_reg_re4, pre_reg_re4, mask=pregFull_re4
                                    )
                                    add_reg_re4 = tla.add(
                                        mul_reg_re4, cur_reg_re4, mask=pregFull_re4
                                    )
                                    div_reg_re4 = tla.div(
                                        add_reg_re4, sum_reg_re4, mask=pregFull_re4
                                    )
                                    pre_tile_re4.store(div_reg_re4, mask=pregFull_re4)
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )
                            for i_re5 in tla.range(q_tile_size_sub_re):
                                for j_re5 in tla.range(
                                    c0, head_dim // (2 * vl_float_ele), c1
                                ):
                                    acc_tile_re5_0 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i_re5, 2 * j_re5),
                                    )
                                    acc_tile_re5_1 = tla.tile_view(
                                        ub_acc_re,
                                        tla.make_shape(1, vl_float_ele),
                                        tla.make_coord(i_re5, 2 * j_re5 + c1),
                                    )
                                    out_tile_re5 = tla.tile_view(
                                        ub_out_f16_re,
                                        tla.make_shape(1, 2 * vl_float_ele),
                                        tla.make_coord(i_re5, j_re5),
                                    )
                                    acc_reg_re5_0 = acc_tile_re5_0.load()
                                    acc_reg_re5_1 = acc_tile_re5_1.load()
                                    out_reg_re5_0 = acc_reg_re5_0.to(
                                        io_dtype, cast_trait_zero, mask=pregFull_re4
                                    )
                                    out_reg_re5_1 = acc_reg_re5_1.to(
                                        io_dtype, cast_trait_zero, mask=pregFull_re4
                                    )
                                    r0_re5, _ = tla.deinterleave(
                                        out_reg_re5_0, out_reg_re5_1
                                    )
                                    out_tile_re5.store(r0_re5, mask=preg_all_b16_re4)
                        tla.set_flag(rescale_v_mte3)
                        tla.wait_flag(rescale_v_mte3)
                        o_row_start_re5 = (
                            q_block_idx_pv * q_block + vec_idx_re * q_tile_size_half_re
                        )
                        o_sub_gm_re5 = tla.make_tensor(
                            mem_o.ptr + o_base_pv + o_row_start_re5 * q_token_stride_pv,
                            tla.make_layout(
                                tla.make_shape(q_tile_size_sub_re, head_dim),
                                tla.make_stride(q_token_stride_pv, 1),
                            ),
                        )
                        if q_tile_size_sub_re > 0:
                            tla.copy(o_sub_gm_re5, ub_out_f16_re)

                    if ubOTmpBufId == c0:
                        tla.cross_core_set_flag(pv_ready_0, tla.arch.VECTOR)
                    else:
                        tla.cross_core_set_flag(pv_ready_1, tla.arch.VECTOR)
                    # O->GM copy 完成, 释放给下一轮 rescale
                    tla.set_flag(mte3_ready_rescale)

                pv_stage_count = pv_stage_count + c1

    with tla.cube():
        tla.wait_flag(q_l0a_ready_l1)
        tla.wait_flag(k_l0b_ready_l1_0)
        tla.wait_flag(k_l0b_ready_l1_1)
        tla.wait_flag(v_l0b_ready_l1_0)
        tla.wait_flag(v_l0b_ready_l1_1)
        tla.wait_flag(mmad_ready_l0a_0)
        tla.wait_flag(mmad_ready_l0a_1)
        tla.wait_flag(mmad_ready_l0b_0)
        tla.wait_flag(mmad_ready_l0b_1)
        tla.wait_flag(fix_ready_mmad_qk_0)
        tla.wait_flag(fix_ready_mmad_qk_1)
        tla.wait_flag(fix_ready_mmad_pv_0)
        tla.wait_flag(fix_ready_mmad_pv_1)

        tla.cross_core_wait_flag(qk_ready_0, tla.arch.FIX)
        tla.cross_core_wait_flag(qk_ready_1, tla.arch.FIX)

        tla.cross_core_wait_flag(pv_ready_0, tla.arch.FIX)
        tla.cross_core_wait_flag(pv_ready_1, tla.arch.FIX)

    with tla.vector():
        tla.cross_core_wait_flag(sm_ready_mm2_0, tla.arch.MTE3)
        tla.cross_core_wait_flag(sm_ready_mm2_1, tla.arch.MTE3)
        tla.cross_core_wait_flag(sm_ready_mm2_2, tla.arch.MTE3)
        tla.wait_flag(mte3_ready_softmax_0)
        tla.wait_flag(mte3_ready_softmax_1)
        tla.wait_flag(mte3_ready_rescale)
        if tla.const_expr(return_lse):
            tla.wait_flag(lse_free)


def block_sparse_attention(
    q,
    k,
    v,
    block_indices,
    block_shape,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    softmax_scale=None,
    is_causal=False,
    return_lse=False,
):
    """Return O, or (O, FP32 natural-log LSE) when return_lse=True."""
    import torch
    import torch_npu  # noqa: F401 -- registers torch.npu device APIs

    use_varlen = q.dim() == 3
    expected_rank = 3 if use_varlen else 4
    if q.dim() != expected_rank or k.dim() != expected_rank or v.dim() != expected_rank:
        layout_name = "packed TND rank-3" if use_varlen else "BNSD rank-4"
        raise ValueError(f"q, k, and v must use contiguous {layout_name} layouts")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("q, k, and v must be contiguous")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q, k, and v must use float16 or bfloat16")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same NPU device")

    if use_varlen:
        if cu_seqlens_q.dim() != 1 or cu_seqlens_k.dim() != 1:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be rank-1 tensors")
        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must use int32")
        if (
            cu_seqlens_q.device != q.device
            or cu_seqlens_k.device != q.device
            or not cu_seqlens_q.is_contiguous()
            or not cu_seqlens_k.is_contiguous()
        ):
            raise ValueError(
                "cu_seqlens_q and cu_seqlens_k must be contiguous on q's NPU device"
            )
        if cu_seqlens_q.numel() != cu_seqlens_k.numel() or cu_seqlens_q.numel() < 2:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must define the same batch")
        batch_size = int(cu_seqlens_q.numel()) - 1
        q_heads = int(q.size(1))
        q_sequence_length = int(max_seqlen_q)
        kv_sequence_length = int(max_seqlen_k)
        head_dim = int(q.size(2))
        if int(k.size(0)) != int(v.size(0)) or int(k.size(1)) != int(v.size(1)):
            raise ValueError("packed k and v must have matching token/head dimensions")
        kv_heads = int(k.size(1))
        if int(k.size(2)) != head_dim or int(v.size(2)) != head_dim:
            raise ValueError("q, k, and v head/value dimensions must all be 128")
    else:
        batch_size = int(q.size(0))
        q_heads = int(q.size(1))
        q_sequence_length = int(q.size(2))
        kv_sequence_length = int(k.size(2))
        head_dim = int(q.size(3))
        if int(k.size(0)) != batch_size or int(v.size(0)) != batch_size:
            raise ValueError("q, k, and v must have the same batch size")
        kv_heads = int(k.size(1))
        if tuple(v.shape[:3]) != tuple(k.shape[:3]):
            raise ValueError("k and v must have matching batch/head/sequence shapes")
        if int(k.size(3)) != head_dim or int(v.size(3)) != head_dim:
            raise ValueError("q, k, and v head/value dimensions must all be 128")
        cu_seqlens_q = torch.empty((1,), dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.empty((1,), dtype=torch.int32, device=q.device)

    if batch_size < 1 or q_heads < 1 or q_sequence_length < 1:
        raise ValueError(
            "batch, query-head count, and maximum query length must be positive"
        )
    if kv_sequence_length < 1:
        raise ValueError("maximum KV sequence length must be positive")
    if head_dim != 128:
        raise ValueError("head_dim must be 128")
    if kv_heads < 1 or q_heads % kv_heads != 0:
        raise ValueError("query-head count must be divisible by KV-head count")

    if block_shape.dim() != 1 or block_shape.numel() != 2:
        raise ValueError("block_shape must be a one-dimensional [128, 128] tensor")
    if block_shape.dtype != torch.int64:
        raise ValueError(
            "block_shape must use int64 and the supported value [128, 128]"
        )
    # Reading an NPU Tensor value in the host wrapper would launch an extra
    # Torch operation. The public contract fixes both entries to 128, and the
    # kernel receives those supported block sizes explicitly as constexprs.
    q_block = 128
    kv_block = 128

    if is_causal:
        raise ValueError("causal attention is not supported")

    if block_indices.dim() not in (3, 4):
        raise ValueError("block_indices must have rank 3 or 4")
    if block_indices.dtype != torch.int32:
        raise ValueError("block_indices must use int32")
    if block_indices.device != q.device or not block_indices.is_contiguous():
        raise ValueError("block_indices must be contiguous on the same NPU device as q")
    q_block_count = (q_sequence_length + q_block - 1) // q_block
    if int(block_indices.size(0)) != batch_size:
        raise ValueError("block_indices batch dimension must match q")
    if int(block_indices.size(-2)) != q_block_count:
        raise ValueError("block_indices query-block shape is invalid")
    mask_heads = 1 if block_indices.dim() == 3 else int(block_indices.size(1))
    if mask_heads not in (1, q_heads):
        raise ValueError("block_indices mask-head count must be 1 or q_heads")
    selected_block_count = int(block_indices.size(-1))
    if selected_block_count < 1:
        raise ValueError("selected block count must be positive")

    try:
        scale_value = float(softmax_scale)
    except TypeError:
        scale_value = head_dim**-0.5
    scale_t = tla.Float32(scale_value)
    output = torch.empty_like(q)
    softmax_lse = (
        torch.empty((*q.shape[:-1], 1), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    q_t = tla.from_dlpack(
        q, layout_tag=tla.arch.RowMajor, origin_shape=(q.numel(),)
    ).mark_layout_dynamic()
    k_t = tla.from_dlpack(
        k, layout_tag=tla.arch.RowMajor, origin_shape=(k.numel(),)
    ).mark_layout_dynamic()
    v_t = tla.from_dlpack(
        v, layout_tag=tla.arch.RowMajor, origin_shape=(v.numel(),)
    ).mark_layout_dynamic()
    output_t = tla.from_dlpack(
        output, layout_tag=tla.arch.RowMajor, origin_shape=(output.numel(),)
    ).mark_layout_dynamic()
    # The disabled specialization never dereferences this placeholder.
    lse_tensor = softmax_lse if return_lse else output
    lse_t = tla.from_dlpack(
        lse_tensor, layout_tag=tla.arch.RowMajor, origin_shape=(lse_tensor.numel(),)
    ).mark_layout_dynamic()
    indices_t = tla.from_dlpack(
        block_indices,
        layout_tag=tla.arch.RowMajor,
        origin_shape=(
            batch_size * mask_heads * q_block_count,
            selected_block_count,
        ),
    ).mark_layout_dynamic()
    cu_q_t = tla.from_dlpack(
        cu_seqlens_q,
        layout_tag=tla.arch.RowMajor,
        origin_shape=(cu_seqlens_q.numel(),),
    ).mark_layout_dynamic()
    cu_k_t = tla.from_dlpack(
        cu_seqlens_k,
        layout_tag=tla.arch.RowMajor,
        origin_shape=(cu_seqlens_k.numel(),),
    ).mark_layout_dynamic()
    batch_size_t = tla.Int32(batch_size)
    q_heads_t = tla.Int32(q_heads)
    kv_heads_t = tla.Int32(kv_heads)
    q_sequence_length_t = tla.Int32(q_sequence_length)
    kv_sequence_length_t = tla.Int32(kv_sequence_length)
    mask_heads_t = tla.Int32(mask_heads)
    use_varlen_t = tla.Int32(1 if use_varlen else 0)
    executor_key = (str(q.dtype), str(q.device), bool(return_lse))
    executor = _EXECUTOR_CACHE.get(executor_key)
    if executor is None:
        executor = tla.compile(
            block_sparse_attention_kernel,
            q_t,
            k_t,
            v_t,
            output_t,
            lse_t,
            indices_t,
            cu_q_t,
            cu_k_t,
            scale_t,
            batch_size_t,
            q_heads_t,
            kv_heads_t,
            q_sequence_length_t,
            kv_sequence_length_t,
            head_dim,
            q_block,
            kv_block,
            mask_heads_t,
            use_varlen_t,
            bool(return_lse),
            options="--npu-arch 3510",
        )
        _EXECUTOR_CACHE[executor_key] = executor
    total_tasks = batch_size * q_heads * q_block_count
    device_properties = torch_npu.npu.get_device_properties(q.device)
    cube_core_num = int(device_properties.cube_core_num)
    if cube_core_num < 1:
        raise RuntimeError("NPU device reports no available Cube cores")
    launch_blocks = min(total_tasks, cube_core_num)
    executor(
        q_t,
        k_t,
        v_t,
        output_t,
        lse_t,
        indices_t,
        cu_q_t,
        cu_k_t,
        scale_t,
        batch_size_t,
        q_heads_t,
        kv_heads_t,
        q_sequence_length_t,
        kv_sequence_length_t,
        mask_heads_t,
        use_varlen_t,
        block_num=launch_blocks,
    )
    return (output, softmax_lse) if return_lse else output
