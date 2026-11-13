# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Standalone operator: matmul + row-wise exp + selected-column scatter-add.

For ``x [M, H]``, ``weight [V, H]``, ``row_logsumexp [M]``,
``global_col_ids [M, S]`` and ``values [M, S]``, compute:

    C = bf16_rne(x @ weight.T)                  # fp32 accumulation
    E = exp(float32(C) * inv_temperature - row_logsumexp[:, None])
    R = -E * values.sum(dim=-1, keepdim=True)
    R[m, global_col_ids[m, s] - col_start] += values[m, s]
    Y = bf16_rne(R * inv_temperature)           # [M, V] bf16

Here ``S = num_selected``, ``inv_temperature = 1 / temperature`` and
``col_start = shard_index * V``.  The scatter-add applies only to global column
ids in ``[col_start, col_start + V)``; repeated ids accumulate in slot order.
All ``values`` participate in the row sum, including entries whose column ids
are outside this shard.  The operator accepts arbitrary selected columns,
including ``S = 0``, and does not compute or require a top-k ordering.

``row_logsumexp`` is supplied by the caller; the operator does not compute it.
It is a per-row log-normalization offset for the temperature-scaled matmul
result.  When it covers the full column range, ``E`` is the corresponding
column shard of that normalized exponential; its local row sum need not be 1.

Public API
----------
``run(x, weight, global_col_ids, row_logsumexp, values, temperature,
shard_index=0, out=None)`` -> ``Y [M, V] bf16``.

Both matrix operands are bf16, column ids are int32, and ``row_logsumexp`` and
``values`` are fp32.  ``weight`` stores one column shard as ``[V, H]``; the
matmul uses its transpose.  ``run`` returns a view into a padded output buffer,
which may be noncontiguous.  Optional ``out`` supplies that full
``[M_pad, V_pad]`` bf16 buffer.

The host layer dispatches padding, the negative row sum, global-to-local column
id conversion, compilation (cached per shape/device) and launch.
``prepare``/``launch`` expose the same work split for benchmarking the main
fused device kernel without preparation overhead.

The matmul accumulates in fp32 and is rounded to bf16 (ties to even) before
the fp32 epilogue.  The output is rounded to bf16 again after the final scale.

Kernel structure
----------------
The device code is one fused kernel split into a cube half and a vector half
that hand a tile over to each other through a single cross-core flag:

    cube half    for each (m, v) tile:  K-blocked matmul into L0C
                                        -> fixpipe copy L0C -> UB (fp32)
    vector half  for each (m, v) tile:  epilogue in place in UB
                                        -> narrow + store Y to GM

Both halves walk the same task list, so they stay in lock step.  Every stage
holds exactly one tile (one L0C accumulator, one UB staging tile), which gives
a one-matmul lookahead: the vector epilogue of tile ``t-1`` runs while the cube
executes the MMAD chain of tile ``t``, and the fixpipe copy of tile ``t`` then
waits for that epilogue to release UB.
"""

import torch
import torch_npu  # noqa: F401

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

# --------------------------------------------------------------------------- #
# Tile configuration
# --------------------------------------------------------------------------- #
# Matmul dimensions are M x N x K, with N = V and K = H.  The selected-column
# count S is separate and is passed as num_selected.
TILE_M = 256  # C tile rows; also the M padding granularity
TILE_N = 256  # C tile columns; also the V padding granularity
TILE_K = 256  # depth of one K block staged in L1; also the H padding granularity
K_STEP = 64  # depth of one L0A/L0B fragment inside a K block

# Super-tile swizzle of the (m, v) tile grid: SWIZZLE_M x SWIZZLE_N tiles per
# super-tile, walked row-major.  Measured on the target device over the
# net-00..13 shapes; the original 32x2 mapping left 3/4 of the cores idle at
# small M.
SWIZZLE_M = 2
SWIZZLE_N = 8

_COMPILE_OPTS = (
    "--npu-arch 3510"
    " --cce-disable-asc-reserved-ubuf"
    " --cce-disable-vf-stack-reserved-ubuf"
)


def _pad_to(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


@tla.kernel
def matmul_row_exp_scatter_add_kernel(
    gm_x: tla.Tensor,  # [M_pad, H_pad] bf16, row-major; X
    gm_weight: tla.Tensor,  # logical [H_pad, V_pad] column-major; W^T
    gm_row_logsumexp: tla.Tensor,  # [M_pad, 1] fp32; row_logsumexp[m]
    gm_neg_row_sum: tla.Tensor,  # [M_pad, 1] fp32; -sum_s values[m, s]
    gm_local_col_ids: tla.Tensor,  # [M_pad, max(S, 1)] int32; local column ids
    gm_values: tla.Tensor,  # [M_pad, max(S, 1)] fp32; values[m, s]
    gm_out: tla.Tensor,  # [M_pad, V_pad] bf16, row-major; Y
    inv_temperature: float,
    num_selected: tla.Constexpr[int],
) -> None:
    zero, one = 0, 1
    tile_m, tile_n, tile_k, k_step = TILE_M, TILE_N, TILE_K, K_STEP
    # The fixpipe splits the C tile over the two AIVs of the cube core, so each
    # AIV owns half of the rows.
    rows_per_aiv = tile_m // 2
    simd_lanes = 64  # fp32 lanes per SIMD fragment
    swizzle_m, swizzle_n = SWIZZLE_M, SWIZZLE_N

    m = gm_x.origin_shape[0]
    v = gm_out.origin_shape[1]
    m_tiles = (m + tile_m - 1) // tile_m
    v_tiles = (v + tile_n - 1) // tile_n
    k_blocks = (gm_x.origin_shape[1] + tile_k - 1) // tile_k
    k_chunks = tile_k // k_step
    m_supers = (m_tiles + swizzle_m - 1) // swizzle_m
    v_supers = (v_tiles + swizzle_n - 1) // swizzle_n
    task_count = m_supers * v_supers * swizzle_m * swizzle_n
    in_dtype = gm_x.ptr.dtype

    # ------------------------------------------------------------------ sync
    # tile_ready is the only cube <-> vector hand-off: the cube half sets it
    # once UB holds a fresh fp32 C tile.  The vector half sets it back once
    # the output has been read out of UB, ping-ponging at tile granularity.
    tile_ready = tla.cross_flag("tile_ready")
    # Load/compute mutexes, one per staging slot.  Ids are fixed literals; they
    # are hardware resources, not handles.
    mutex_l1_a0 = tla.mutex("l1_a0", id=0)
    mutex_l1_b0 = tla.mutex("l1_b0", id=1)
    mutex_l1_a1 = tla.mutex("l1_a1", id=2)
    mutex_l1_b1 = tla.mutex("l1_b1", id=3)
    mutex_l0_a0 = tla.mutex("l0_a0", id=4)
    mutex_l0_b0 = tla.mutex("l0_b0", id=5)
    mutex_l0_a1 = tla.mutex("l0_a1", id=6)
    mutex_l0_b1 = tla.mutex("l0_b1", id=7)
    mutex_l0_c = tla.mutex("l0_c", id=8)
    # Guards the final ``* invT`` pass against a concurrent fixpipe fill.
    mutex_epilogue = tla.mutex("epilogue", id=10)

    # -------------------------------------------------------------- staging
    # L1 holds two ping-pong slots of the A and B operands, one K block each.
    l1_a0_ptr = tla.allocate(tile_m * tile_k, in_dtype, tla.AddressSpace.l1, 512)
    l1_a1_ptr = tla.allocate(tile_m * tile_k, in_dtype, tla.AddressSpace.l1, 512)
    l1_b0_ptr = tla.allocate(tile_k * tile_n, in_dtype, tla.AddressSpace.l1, 512)
    l1_b1_ptr = tla.allocate(tile_k * tile_n, in_dtype, tla.AddressSpace.l1, 512)
    # L0A/L0B hold one K_STEP fragment, ping-ponged inside a K block.
    l0_a0_ptr = tla.allocate(tile_m * k_step, in_dtype, tla.AddressSpace.l0a, 512)
    l0_a1_ptr = tla.allocate(tile_m * k_step, in_dtype, tla.AddressSpace.l0a, 512)
    l0_b0_ptr = tla.allocate(k_step * tile_n, in_dtype, tla.AddressSpace.l0b, 512)
    l0_b1_ptr = tla.allocate(k_step * tile_n, in_dtype, tla.AddressSpace.l0b, 512)
    # One fp32 accumulator; the fixpipe copies it out whole, so it cannot be
    # double buffered (L0C on this part is exactly one TILE_M x TILE_N tile).
    l0_c_ptr = tla.allocate(tile_m * tile_n, tla.Float32, tla.AddressSpace.l0c, 1024)
    # UB: the fp32 C tile (one AIV half) plus the per-row epilogue scalars.
    ub_c_ptr = tla.allocate(
        rows_per_aiv * tile_n, tla.Float32, tla.AddressSpace.ub, 256
    )
    ub_row_logsumexp_ptr = tla.allocate(
        rows_per_aiv, tla.Float32, tla.AddressSpace.ub, 256
    )
    ub_neg_row_sum_ptr = tla.allocate(
        rows_per_aiv, tla.Float32, tla.AddressSpace.ub, 256
    )

    # ======================================================================= #
    # Cube half: matmul, then hand the tile over to the vector half.
    # ======================================================================= #
    with tla.cube():
        # Logical views used as layout templates; the fixpipe writes the AIV
        # half of ``ub_c_full`` at offset 0 of its own UB.
        ub_c_full = tla.make_tensor(
            ub_c_ptr,
            tla.make_layout(tla.make_shape(tile_m, tile_n), tla.make_stride(tile_n, 1)),
        )
        l0_c = tla.make_tensor_like(l0_c_ptr, ub_c_full, tla.arch.L0Clayout)

        for task in tla.range(tla.arch.block_idx(), task_count, tla.arch.block_num()):
            # Task -> (m tile, v tile) through the super-tile swizzle.
            super_tile = task // (swizzle_m * swizzle_n)
            slot = task % (swizzle_m * swizzle_n)
            m_tile = (super_tile // v_supers) * swizzle_m + slot // swizzle_n
            v_tile = (super_tile % v_supers) * swizzle_n + slot % swizzle_n
            if m_tile < m_tiles and v_tile < v_tiles:
                # Every tile's first block is loaded into slot 0.  Reset the
                # phase here; an odd block count leaves the previous tile at 1.
                l1_phase = tla.as_numeric(zero)
                # Stage the first K block before entering the pipeline.
                gm_a_head = tla.tile_view(
                    gm_x,
                    tla.make_shape(tile_m, tile_k),
                    tla.make_coord(m_tile, zero),
                )
                gm_b_head = tla.tile_view(
                    gm_weight,
                    tla.make_shape(tile_k, tile_n),
                    tla.make_coord(zero, v_tile),
                )
                l1_a_head = tla.make_tensor_like(l1_a0_ptr, gm_a_head, tla.arch.zN)
                l1_b_head = tla.make_tensor_like(l1_b0_ptr, gm_b_head, tla.arch.nZ)
                mutex_l1_a0.lock(pipe=tla.arch.MTE2)
                tla.copy(l1_a_head, gm_a_head)
                mutex_l1_a0.unlock(pipe=tla.arch.MTE2)
                mutex_l1_b0.lock(pipe=tla.arch.MTE2)
                tla.copy(l1_b_head, gm_b_head)
                mutex_l1_b0.unlock(pipe=tla.arch.MTE2)

                mutex_l0_c.lock(pipe=tla.arch.CUBE)

                for k_block in tla.range(zero, k_blocks, one):
                    if l1_phase == zero:
                        mutex_l1_a0.lock(pipe=tla.arch.MTE1)
                        mutex_l1_b0.lock(pipe=tla.arch.MTE1)
                    else:
                        mutex_l1_a1.lock(pipe=tla.arch.MTE1)
                        mutex_l1_b1.lock(pipe=tla.arch.MTE1)

                    # Prefetch the next K block into the other L1 slot.
                    if k_block + one < k_blocks:
                        gm_a_next = tla.tile_view(
                            gm_x,
                            tla.make_shape(tile_m, tile_k),
                            tla.make_coord(m_tile, k_block + one),
                        )
                        gm_b_next = tla.tile_view(
                            gm_weight,
                            tla.make_shape(tile_k, tile_n),
                            tla.make_coord(k_block + one, v_tile),
                        )
                        next_a_ptr = l1_a1_ptr if l1_phase == zero else l1_a0_ptr
                        next_b_ptr = l1_b1_ptr if l1_phase == zero else l1_b0_ptr
                        l1_a_next = tla.make_tensor_like(
                            next_a_ptr, gm_a_next, tla.arch.zN
                        )
                        l1_b_next = tla.make_tensor_like(
                            next_b_ptr, gm_b_next, tla.arch.nZ
                        )
                        if l1_phase == zero:
                            mutex_l1_a1.lock(pipe=tla.arch.MTE2)
                            mutex_l1_b1.lock(pipe=tla.arch.MTE2)
                        else:
                            mutex_l1_a0.lock(pipe=tla.arch.MTE2)
                            mutex_l1_b0.lock(pipe=tla.arch.MTE2)
                        tla.copy(l1_a_next, gm_a_next)
                        tla.copy(l1_b_next, gm_b_next)
                        if l1_phase == zero:
                            mutex_l1_a1.unlock(pipe=tla.arch.MTE2)
                            mutex_l1_b1.unlock(pipe=tla.arch.MTE2)
                        else:
                            mutex_l1_a0.unlock(pipe=tla.arch.MTE2)
                            mutex_l1_b0.unlock(pipe=tla.arch.MTE2)

                    cur_a_ptr = l1_a0_ptr if l1_phase == zero else l1_a1_ptr
                    cur_b_ptr = l1_b0_ptr if l1_phase == zero else l1_b1_ptr
                    gm_a_cur = tla.tile_view(
                        gm_x,
                        tla.make_shape(tile_m, tile_k),
                        tla.make_coord(m_tile, k_block),
                    )
                    gm_b_cur = tla.tile_view(
                        gm_weight,
                        tla.make_shape(tile_k, tile_n),
                        tla.make_coord(k_block, v_tile),
                    )
                    l1_a = tla.make_tensor_like(cur_a_ptr, gm_a_cur, tla.arch.zN)
                    l1_b = tla.make_tensor_like(cur_b_ptr, gm_b_cur, tla.arch.nZ)

                    # Prime the L0 fragments with the first K_STEP of the block.
                    l1_a_first = tla.tile_view(
                        l1_a,
                        tla.make_shape(tile_m, k_step),
                        tla.make_coord(zero, zero),
                    )
                    l1_b_first = tla.tile_view(
                        l1_b,
                        tla.make_shape(k_step, tile_n),
                        tla.make_coord(zero, zero),
                    )
                    l0_a_first = tla.make_tensor_like(
                        l0_a0_ptr, l1_a_first, tla.arch.zN
                    )
                    l0_b_first = tla.make_tensor_like(
                        l0_b0_ptr, l1_b_first, tla.arch.nZ
                    )
                    mutex_l0_a0.lock(pipe=tla.arch.MTE1)
                    mutex_l0_b0.lock(pipe=tla.arch.MTE1)
                    tla.copy(l0_a_first, l1_a_first)
                    tla.copy(l0_b_first, l1_b_first)
                    mutex_l0_a0.unlock(pipe=tla.arch.MTE1)
                    mutex_l0_b0.unlock(pipe=tla.arch.MTE1)

                    l0_phase = tla.as_numeric(zero)
                    for k_chunk in tla.range(zero, k_chunks, one):
                        if l0_phase == zero:
                            mutex_l0_a0.lock(pipe=tla.arch.CUBE)
                            mutex_l0_b0.lock(pipe=tla.arch.CUBE)
                        else:
                            mutex_l0_a1.lock(pipe=tla.arch.CUBE)
                            mutex_l0_b1.lock(pipe=tla.arch.CUBE)

                        # Load the next L0 fragment into the other slot.
                        l0_src_a = tla.tile_view(
                            l1_a,
                            tla.make_shape(tile_m, k_step),
                            tla.make_coord(zero, k_chunk + one),
                        )
                        l0_src_b = tla.tile_view(
                            l1_b,
                            tla.make_shape(k_step, tile_n),
                            tla.make_coord(k_chunk + one, zero),
                        )
                        next_l0_a_ptr = l0_a1_ptr if l0_phase == zero else l0_a0_ptr
                        next_l0_b_ptr = l0_b1_ptr if l0_phase == zero else l0_b0_ptr
                        l0_a_next = tla.make_tensor_like(
                            next_l0_a_ptr, l0_src_a, tla.arch.zN
                        )
                        l0_b_next = tla.make_tensor_like(
                            next_l0_b_ptr, l0_src_b, tla.arch.nZ
                        )
                        if k_chunk + one < k_chunks:
                            if l0_phase == zero:
                                mutex_l0_a1.lock(pipe=tla.arch.MTE1)
                                mutex_l0_b1.lock(pipe=tla.arch.MTE1)
                            else:
                                mutex_l0_a0.lock(pipe=tla.arch.MTE1)
                                mutex_l0_b0.lock(pipe=tla.arch.MTE1)
                            tla.copy(l0_a_next, l0_src_a)
                            tla.copy(l0_b_next, l0_src_b)
                            if l0_phase == zero:
                                mutex_l0_a1.unlock(pipe=tla.arch.MTE1)
                                mutex_l0_b1.unlock(pipe=tla.arch.MTE1)
                            else:
                                mutex_l0_a0.unlock(pipe=tla.arch.MTE1)
                                mutex_l0_b0.unlock(pipe=tla.arch.MTE1)

                        cur_l0_a_ptr = l0_a0_ptr if l0_phase == zero else l0_a1_ptr
                        cur_l0_b_ptr = l0_b0_ptr if l0_phase == zero else l0_b1_ptr
                        l1_a_cur = tla.tile_view(
                            l1_a,
                            tla.make_shape(tile_m, k_step),
                            tla.make_coord(zero, k_chunk),
                        )
                        l1_b_cur = tla.tile_view(
                            l1_b,
                            tla.make_shape(k_step, tile_n),
                            tla.make_coord(k_chunk, zero),
                        )
                        l0_a = tla.make_tensor_like(cur_l0_a_ptr, l1_a_cur, tla.arch.zN)
                        l0_b = tla.make_tensor_like(cur_l0_b_ptr, l1_b_cur, tla.arch.nZ)

                        # First fragment of the tile initialises the accumulator;
                        # the last one releases L0C to the fixpipe.
                        init_c = True if k_block == zero and k_chunk == zero else False
                        unit_flag = (
                            0b11
                            if (k_block + one >= k_blocks)
                            and (k_chunk + one >= k_chunks)
                            else 0b10
                        )
                        tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
                        if l0_phase == zero:
                            mutex_l0_a0.unlock(pipe=tla.arch.CUBE)
                            mutex_l0_b0.unlock(pipe=tla.arch.CUBE)
                        else:
                            mutex_l0_a1.unlock(pipe=tla.arch.CUBE)
                            mutex_l0_b1.unlock(pipe=tla.arch.CUBE)
                        l0_phase = one - l0_phase

                    if l1_phase == zero:
                        mutex_l1_a0.unlock(pipe=tla.arch.MTE1)
                        mutex_l1_b0.unlock(pipe=tla.arch.MTE1)
                    else:
                        mutex_l1_a1.unlock(pipe=tla.arch.MTE1)
                        mutex_l1_b1.unlock(pipe=tla.arch.MTE1)
                    l1_phase = one - l1_phase

                # Accumulator is complete: hand the fp32 tile to the vector half.
                # The fixpipe waits for the last MMAD (unit_flag bit 0) and splits
                # the tile over the two AIVs.
                mutex_l0_c.unlock(pipe=tla.arch.CUBE)
                mutex_l0_c.lock(pipe=tla.arch.FIX)
                tla.cross_core_wait_flag(tile_ready, tla.arch.FIX)
                tla.copy(
                    ub_c_full,
                    l0_c,
                    tla.params.CopyL0C2DstParams(
                        unit_flag=3,
                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
                    ),
                )
                tla.cross_core_set_flag(tile_ready, tla.arch.FIX)
                mutex_l0_c.unlock(pipe=tla.arch.FIX)

    # ======================================================================= #
    # Vector half: epilogue in place, then narrow and store the output.
    # ======================================================================= #
    with tla.vector():
        aiv = tla.arch.sub_block_idx()
        ub_c = tla.make_tensor(
            ub_c_ptr,
            tla.make_layout(
                tla.make_shape(rows_per_aiv, tile_n), tla.make_stride(tile_n, 1)
            ),
        )
        ub_row_logsumexp = tla.make_tensor(
            ub_row_logsumexp_ptr,
            tla.make_layout(tla.make_shape(rows_per_aiv), tla.make_stride(1)),
        )
        ub_neg_row_sum = tla.make_tensor(
            ub_neg_row_sum_ptr,
            tla.make_layout(tla.make_shape(rows_per_aiv), tla.make_stride(1)),
        )
        # Per-row scalar loads are broadcast to every SIMD lane.
        broadcast_load = tla.params.NormalLoadParams(
            load_dist=tla.params.LoadDist.DIST_BRC_B32
        )
        # 2.2.0 has no CAST_RINT (only CEIL/FLOOR/ROUND/TRUNC), so both
        # ties-to-even bf16 roundings below are emulated on the f32 bit pattern:
        # (x + 0x7FFF + ((x >> 16) & 1)) & ~0xFFFF.

        tla.cross_core_set_flag(tile_ready, tla.arch.VECTOR)

        for task in tla.range(tla.arch.block_idx(), task_count, tla.arch.block_num()):
            super_tile = task // (swizzle_m * swizzle_n)
            slot = task % (swizzle_m * swizzle_n)
            m_tile = (super_tile // v_supers) * swizzle_m + slot // swizzle_n
            v_tile = (super_tile % v_supers) * swizzle_n + slot % swizzle_n
            if m_tile < m_tiles and v_tile < v_tiles:
                row_base = m_tile * tile_m + aiv * rows_per_aiv
                col_base = v_tile * tile_n
                tla.cross_core_wait_flag(tile_ready, tla.arch.VECTOR)

                # Step 1 - per-row lse and negative row sum.  A 4-byte pitch
                # is not a legal DMA tile, so use one scalar GM read per row.
                with tla.vec.func(mode="simt", thread_block_dim=rows_per_aiv):
                    row_thread, _, _ = tla.arch.thread_idx()
                    ub_row_logsumexp[row_thread] = gm_row_logsumexp[
                        row_base + row_thread, zero
                    ]
                    ub_neg_row_sum[row_thread] = gm_neg_row_sum[
                        row_base + row_thread, zero
                    ]
                tla.pipe_barrier(tla.pipes.ALL)

                # Step 2 - matmul result -> bf16 -> exp, then scale by
                # the negative row sum:
                #   R = exp(bf16_rne(C) * invT - lse) * (-sum_s values)
                # Preparation computes the negative row sum once, avoiding
                # a sign flip in the inner loop.
                with tla.vec.func(mode="simd"):
                    all_lanes = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    lsb_one = tla.full(1, dtype=tla.Int32)
                    round_bias = tla.full(32767, dtype=tla.Int32)
                    bf16_keep = tla.full(-65536, dtype=tla.Int32)
                    for row in tla.range(rows_per_aiv):
                        neg_row_sum = tla.tile_view(
                            ub_neg_row_sum,
                            tla.make_shape(1),
                            tla.make_coord(row),
                        ).load(params=broadcast_load)
                        logsumexp_row = tla.tile_view(
                            ub_row_logsumexp,
                            tla.make_shape(1),
                            tla.make_coord(row),
                        ).load(params=broadcast_load)
                        for col_block in tla.range(zero, tile_n // simd_lanes, one):
                            c_frag = tla.tile_view(
                                ub_c,
                                tla.make_shape(1, simd_lanes),
                                tla.make_coord(row, col_block),
                            ).load()
                            # Round the matmul result to bf16 before the fp32
                            # epilogue (ties to even).
                            c_frag = c_frag.bitcast(tla.Int32)
                            lsb = tla.bitwise_and(
                                tla.shift_right(c_frag, 16),
                                lsb_one,
                                mask=all_lanes,
                            )
                            c_frag = tla.add(
                                tla.add(c_frag, round_bias, mask=all_lanes),
                                lsb,
                                mask=all_lanes,
                            )
                            c_frag = tla.bitwise_and(c_frag, bf16_keep, mask=all_lanes)
                            c_frag = c_frag.bitcast(tla.Float32)
                            c_frag = tla.mul(c_frag, inv_temperature, mask=all_lanes)
                            c_frag = tla.sub(c_frag, logsumexp_row, mask=all_lanes)
                            c_frag = tla.exp(c_frag, mask=all_lanes)
                            c_frag = tla.mul(c_frag, neg_row_sum, mask=all_lanes)
                            tla.tile_view(
                                ub_c,
                                tla.make_shape(1, simd_lanes),
                                tla.make_coord(row, col_block),
                            ).store(c_frag, mask=all_lanes)

                tla.pipe_barrier(tla.pipes.ALL)

                # Step 3 - selected-column scatter-add: one thread per row,
                # one read-modify-write per selected slot.  Repeated ids add
                # sequentially.  ``gm_local_col_ids`` holds shard-local column
                # ids; preparation maps ids outside this shard to -1.
                with tla.vec.func(mode="simt", thread_block_dim=rows_per_aiv):
                    row_thread, _, _ = tla.arch.thread_idx()
                    global_row = row_base + row_thread
                    for selected_idx in tla.range(zero, num_selected, one):
                        local_col = gm_local_col_ids[global_row, selected_idx]
                        if local_col >= zero:
                            if local_col >= col_base:
                                if local_col < col_base + tile_n:
                                    ub_c[row_thread, local_col - col_base] = (
                                        ub_c[row_thread, local_col - col_base]
                                        + gm_values[global_row, selected_idx]
                                    )

                tla.pipe_barrier(tla.pipes.ALL)

                # Step 4 - Y = bf16_rne(R * invT), in place.  The value is left
                # as an fp32 number that already sits exactly on the bf16 grid,
                # so step 5's narrowing cannot round a second time.
                mutex_epilogue.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    out_lanes = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    out_lsb_one = tla.full(1, dtype=tla.Int32)
                    out_round_bias = tla.full(32767, dtype=tla.Int32)
                    out_bf16_mask = tla.full(-65536, dtype=tla.Int32)
                    for out_round_row in tla.range(rows_per_aiv):
                        for out_col_block in tla.range(zero, tile_n // simd_lanes, one):
                            out_frag = tla.tile_view(
                                ub_c,
                                tla.make_shape(1, simd_lanes),
                                tla.make_coord(out_round_row, out_col_block),
                            ).load()
                            out_frag = tla.mul(
                                out_frag, inv_temperature, mask=out_lanes
                            )
                            out_frag = out_frag.bitcast(tla.Int32)
                            out_lsb = tla.bitwise_and(
                                tla.shift_right(out_frag, 16),
                                out_lsb_one,
                                mask=out_lanes,
                            )
                            out_frag = tla.add(
                                tla.add(out_frag, out_round_bias, mask=out_lanes),
                                out_lsb,
                                mask=out_lanes,
                            )
                            out_frag = tla.bitwise_and(
                                out_frag, out_bf16_mask, mask=out_lanes
                            )
                            out_frag = out_frag.bitcast(tla.Float32)
                            tla.tile_view(
                                ub_c,
                                tla.make_shape(1, simd_lanes),
                                tla.make_coord(out_round_row, out_col_block),
                            ).store(out_frag, mask=out_lanes)
                mutex_epilogue.unlock(pipe=tla.arch.VECTOR)

                tla.pipe_barrier(tla.pipes.ALL)

                # Step 5 - output tile -> GM.  bf16 has no legal UB -> GM cast on
                # 2.2.0, so the bf16-exact fp32 value is narrowed per lane and
                # stored scalar.  One thread per column: the tile_n lanes write
                # one contiguous GM row, so the stores coalesce.
                with tla.vec.func(mode="simt", thread_block_dim=tile_n):
                    col_thread, _, _ = tla.arch.thread_idx()
                    for out_row in tla.range(zero, rows_per_aiv, one):
                        gm_out[row_base + out_row, col_base + col_thread] = ub_c[
                            out_row, col_thread
                        ].to(tla.BFloat16)
                # Hand UB back once all output values have been read out.
                tla.pipe_barrier(tla.pipes.ALL)
                tla.cross_core_set_flag(tile_ready, tla.arch.VECTOR)

        tla.pipe_barrier(tla.pipes.ALL)


# ----------------------------------------------------------------------------------------------
# Host layer
# ----------------------------------------------------------------------------------------------
def _check_2d(name, t, dtype, rows=None, cols=None):
    if t.dim() != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(t.shape)}")
    if t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {t.dtype}")
    if rows is not None and t.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows, got {t.shape[0]}")
    if cols is not None and t.shape[1] != cols:
        raise ValueError(f"{name} must have {cols} columns, got {t.shape[1]}")


class _Prepared:
    __slots__ = (
        "compiled",
        "launch",
        "blocks",
        "m_pad",
        "h_pad",
        "v_pad",
        "m",
        "v",
        "x_pad",
        "weight_pad",
        "lse_pad",
        "neg_row_sum_pad",
        "col_ids_pad",
        "values_pad",
        "out_buf",
    )


_COMPILE_CACHE = {}
_STAGE_CACHE = {}


def _stage(device, name, shape, dtype):
    """Cached staging buffer, zero-initialized on its first allocation."""
    key = (device, name, shape, dtype)
    buf = _STAGE_CACHE.get(key)
    if buf is None:
        buf = torch.zeros(shape, dtype=dtype, device=f"npu:{device}")
        _STAGE_CACHE[key] = buf
    return buf


def _copy_padded(dst, src):
    """Copy ``src`` into the top-left corner and clear the rest of ``dst``."""
    rows, cols = src.shape
    dst[:rows, :cols].copy_(src)
    if dst.shape[0] > rows:
        dst[rows:].zero_()
    if dst.dim() == 2 and dst.shape[1] > cols:
        dst[:rows, cols:].zero_()


def _copy_padded_1d(dst, src):
    n = src.shape[0]
    dst[:n].copy_(src)
    if dst.shape[0] > n:
        dst[n:].zero_()


def _compiled_entry(device, m_pad, h_pad, v_pad, num_selected, sample):
    key = (device, m_pad, h_pad, v_pad, num_selected)
    entry = _COMPILE_CACHE.get(key)
    if entry is None:
        x_t, weight_t, lse_t, neg_row_sum_t, col_ids_t, values_t, out_t = sample
        # Only shapes and dtypes are specialised; the scalars below are runtime
        # arguments, so any valid ones compile the same kernel.
        entry = tla.compile(
            matmul_row_exp_scatter_add_kernel,
            x_t,
            weight_t,
            lse_t,
            neg_row_sum_t,
            col_ids_t,
            values_t,
            out_t,
            1.0,
            num_selected,
            options=_COMPILE_OPTS,
        )
        _COMPILE_CACHE[key] = entry
    return entry


def _to_tla(
    x_pad, weight_pad, lse_pad, neg_row_sum_pad, col_ids_pad, values_pad, out_buf
):
    return (
        from_dlpack(x_pad, layout_tag=tla.arch.RowMajor),
        from_dlpack(weight_pad, layout_tag=tla.arch.ColumnMajor),
        from_dlpack(lse_pad.reshape(-1, 1), layout_tag=tla.arch.RowMajor),
        from_dlpack(neg_row_sum_pad.reshape(-1, 1), layout_tag=tla.arch.RowMajor),
        from_dlpack(col_ids_pad, layout_tag=tla.arch.RowMajor),
        from_dlpack(values_pad, layout_tag=tla.arch.RowMajor),
        from_dlpack(out_buf, layout_tag=tla.arch.RowMajor),
    )


def prepare(
    x,
    weight,
    global_col_ids,
    row_logsumexp,
    values,
    temperature,
    shard_index=0,
    out=None,
):
    """Pad inputs, compute negative row sums and compile the kernel.

    Uses the same input contract as ``run``.  Returns an object whose
    ``launch()`` runs the main device kernel only, so preparation cost can be
    excluded from timing.  Optional ``out`` supplies a preallocated
    ``[M_pad, V_pad]`` bf16 buffer.

    Staging buffers are shared by device, role and padded shape.  A later
    preparation can overwrite inputs held by an earlier prepared object.
    """
    m, h = x.shape
    v = weight.shape[0]
    num_selected = global_col_ids.shape[1]
    device = x.device.index or torch.npu.current_device()

    col_start = int(shard_index) * v
    m_pad = _pad_to(m, TILE_M)
    h_pad = _pad_to(h, TILE_K)
    v_pad = _pad_to(v, TILE_N)

    x_pad = _stage(device, "x", (m_pad, h_pad), torch.bfloat16)
    _copy_padded(x_pad, x)
    weight_pad = _stage(device, "w", (v_pad, h_pad), torch.bfloat16)
    _copy_padded(weight_pad, weight)
    lse_pad = _stage(device, "lse", (m_pad,), torch.float32)
    _copy_padded_1d(lse_pad, row_logsumexp)
    # Global column ids -> this shard's column ids.  Out-of-shard ids become
    # -1 and skip scatter-add, while their values still enter the row sum.
    local_col_ids = global_col_ids - col_start
    local_col_ids.masked_fill_((local_col_ids < 0) | (local_col_ids >= v), -1)
    # The runtime rejects zero-width RowMajor views.  Keep one dummy staging
    # column for S = 0; num_selected stays zero, so the kernel never reads it.
    selected_cols_pad = max(1, num_selected)
    col_ids_pad = _stage(device, "col_ids", (m_pad, selected_cols_pad), torch.int32)
    _copy_padded(col_ids_pad, local_col_ids)
    values_pad = _stage(device, "values", (m_pad, selected_cols_pad), torch.float32)
    _copy_padded(values_pad, values)
    # Precompute the negative sum over all selected values in each row,
    # including values whose ids are outside this shard.
    neg_row_sum_pad = _stage(device, "neg_row_sum", (m_pad,), torch.float32)
    _copy_padded_1d(neg_row_sum_pad, -values.sum(dim=-1))
    if out is None:
        out_buf = torch.empty(
            (m_pad, v_pad), dtype=torch.bfloat16, device=f"npu:{device}"
        )
    else:
        if tuple(out.shape) != (m_pad, v_pad) or out.dtype != torch.bfloat16:
            raise ValueError(
                f"out must be bf16 [{m_pad}, {v_pad}], got {out.dtype} "
                f"{tuple(out.shape)}"
            )
        if out.device.type != "npu" or not out.is_contiguous():
            raise ValueError("out must be a contiguous NPU tensor")
        out_buf = out

    inv_temperature = 1.0 / float(temperature)
    sample = _to_tla(
        x_pad, weight_pad, lse_pad, neg_row_sum_pad, col_ids_pad, values_pad, out_buf
    )
    entry = _compiled_entry(device, m_pad, h_pad, v_pad, num_selected, sample)
    blocks = max(1, int(torch.npu.get_device_properties(device).cube_core_num))

    p = _Prepared()
    p.compiled = (entry, sample, inv_temperature, blocks)
    p.blocks = blocks
    p.m_pad, p.h_pad, p.v_pad, p.m, p.v = m_pad, h_pad, v_pad, m, v
    p.x_pad, p.weight_pad, p.lse_pad, p.neg_row_sum_pad = (
        x_pad,
        weight_pad,
        lse_pad,
        neg_row_sum_pad,
    )
    p.col_ids_pad, p.values_pad, p.out_buf = col_ids_pad, values_pad, out_buf

    def launch():
        entry, sample, inv_t, blocks_ = p.compiled
        entry(*sample, inv_t, block_num=blocks_)

    p.launch = launch
    return p


def run(
    x,
    weight,
    global_col_ids,
    row_logsumexp,
    values,
    temperature,
    shard_index=0,
    out=None,
):
    """Compute matmul, row-wise exp/scaling and selected-column scatter-add.

    Inputs are ``x [M, H]`` and ``weight [V, H]`` in bf16,
    ``global_col_ids [M, S]`` in int32, and ``row_logsumexp [M]`` and
    ``values [M, S]`` in fp32.  Columns in this shard cover
    ``[shard_index*V, (shard_index+1)*V)``.  Out-of-shard ids skip scatter-add;
    all values still contribute to their row sums.  Repeated ids accumulate.

    Returns ``Y [M, V] bf16`` as a possibly noncontiguous view.  Optional
    ``out`` is the full contiguous ``[M_pad, V_pad]`` bf16 output buffer;
    reusing one avoids a per-call allocation.
    """
    if x.device.type != "npu":
        raise ValueError("x must be on an NPU device")
    _check_2d("x", x, torch.bfloat16)
    _check_2d("weight", weight, torch.bfloat16, cols=x.shape[1])
    _check_2d("global_col_ids", global_col_ids, torch.int32, rows=x.shape[0])
    if row_logsumexp.dim() != 1 or row_logsumexp.dtype != torch.float32:
        raise ValueError("row_logsumexp must be 1-D float32")
    if row_logsumexp.shape[0] != x.shape[0]:
        raise ValueError("row_logsumexp must have M entries")
    _check_2d(
        "values", values, torch.float32, rows=x.shape[0], cols=global_col_ids.shape[1]
    )
    if not float(temperature) > 0:
        raise ValueError("temperature must be positive")
    if int(shard_index) < 0:
        raise ValueError("shard_index must be non-negative")

    m, v = x.shape[0], weight.shape[0]
    if m == 0 or v == 0:
        return torch.empty((m, v), dtype=torch.bfloat16, device=x.device)

    p = prepare(
        x,
        weight,
        global_col_ids,
        row_logsumexp,
        values,
        temperature,
        shard_index,
        out=out,
    )
    p.launch()
    torch.npu.synchronize()
    return p.out_buf[:m, :v]
