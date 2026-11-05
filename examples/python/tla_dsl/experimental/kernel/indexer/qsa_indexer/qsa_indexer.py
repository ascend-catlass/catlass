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


@tla.jit
def _qsa_indexer_all_visible_stage(
    query_positions: tla.Tensor,
    output: tla.Tensor,
    rows: tla.Constexpr[int],
    output_width: tla.Constexpr[int],
) -> None:
    SIMT_THREADS = 256

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
            thread_idx, _, _ = tla.arch.thread_idx()
            thread_count, _, _ = tla.arch.thread_block_dim()
            block_idx = tla.arch.block_idx()
            block_count = tla.arch.block_num()
            linear = block_idx * thread_count + thread_idx
            stride = block_count * thread_count
            element_count = rows * output_width

            for offset in tla.range(linear, element_count, stride):
                row = offset // output_width
                column = offset % output_width
                position = query_positions[row]
                if column == output_width - 1:
                    output[row, column] = position + 1
                else:
                    if column <= position:
                        output[row, column] = column.to(tla.Int32)
                    else:
                        output[row, column] = -1

        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def qsa_indexer(
    q: tla.Tensor,
    compressed_k: tla.Tensor,
    block_table: tla.Tensor,
    actual_seq: tla.Tensor,
    query_positions: tla.Tensor,
    output: tla.Tensor,
    raw_workspace: tla.Tensor,
    score_workspace: tla.Tensor,
    compress_ratio: tla.Constexpr[int],
    topk_limit: tla.Constexpr[int],
    query_heads: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    page_size: tla.Constexpr[int],
    kv_heads: tla.Constexpr[int],
    run_topk: tla.Constexpr[bool],
    rows: tla.Constexpr[int],
    output_width: tla.Constexpr[int],
) -> None:
    if run_topk:
        _qsa_indexer_mixed_topk_stage(
            q,
            compressed_k,
            block_table,
            actual_seq,
            query_positions,
            output,
            raw_workspace,
            score_workspace,
            compress_ratio,
            topk_limit,
            query_heads,
            head_dim,
            page_size,
            kv_heads,
        )
    else:
        _qsa_indexer_all_visible_stage(
            query_positions,
            output,
            rows,
            output_width,
        )


@tla.jit
def _qsa_indexer_mixed_topk_stage(
    q: tla.Tensor,
    compressed_k: tla.Tensor,
    block_table: tla.Tensor,
    actual_seq: tla.Tensor,
    query_positions: tla.Tensor,
    output: tla.Tensor,
    raw_workspace: tla.Tensor,
    score_workspace: tla.Tensor,
    compress_ratio: tla.Constexpr[int],
    topk_limit: tla.Constexpr[int],
    query_heads: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    page_size: tla.Constexpr[int],
    kv_heads: tla.Constexpr[int],
) -> None:
    SIMT_THREADS = 256
    MIXED_TILE_M = 32
    SCORE_REG_WIDTH = 64

    rows = output.origin_shape[0]
    max_pages = block_table.origin_shape[1]
    max_blocks = max_pages * page_size
    max_blocks_aligned = ((max_blocks + 255) // 256) * 256
    output_width = compress_ratio * (topk_limit + 1)
    group_size = query_heads // kv_heads
    score_dtype = q.ptr.dtype
    task_count = (rows + MIXED_TILE_M - 1) // MIXED_TILE_M

    aic_to_aiv = tla.cross_flag("qsa_score_ready", mode=4)
    aiv_to_aic = tla.cross_flag("qsa_score_free", mode=4)

    l1_q_loaded = tla.flag("qsa_l1_q_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l1_k_loaded = tla.flag("qsa_l1_k_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l1_q_available = tla.flag("qsa_l1_q_available", tla.arch.MTE1, tla.arch.MTE2)
    l1_k_available = tla.flag("qsa_l1_k_available", tla.arch.MTE1, tla.arch.MTE2)
    l0_q_loaded = tla.flag("qsa_l0_q_loaded", tla.arch.MTE1, tla.arch.CUBE)
    l0_k_loaded = tla.flag("qsa_l0_k_loaded", tla.arch.MTE1, tla.arch.CUBE)
    l0_q_available = tla.flag("qsa_l0_q_available", tla.arch.CUBE, tla.arch.MTE1)
    l0_k_available = tla.flag("qsa_l0_k_available", tla.arch.CUBE, tla.arch.MTE1)
    mmad_done = tla.flag("qsa_mmad_done", tla.arch.CUBE, tla.arch.FIX)
    l0_c_available = tla.flag("qsa_l0_c_available", tla.arch.FIX, tla.arch.CUBE)

    l1_q_ptr = tla.allocate(
        MIXED_TILE_M * query_heads * head_dim,
        score_dtype,
        tla.AddressSpace.l1,
        512,
    )
    l1_k_ptr = tla.allocate(
        head_dim * page_size,
        score_dtype,
        tla.AddressSpace.l1,
        512,
    )
    l0_q_ptr = tla.allocate(
        MIXED_TILE_M * query_heads * head_dim,
        score_dtype,
        tla.AddressSpace.l0a,
        512,
    )
    l0_k_ptr = tla.allocate(
        head_dim * page_size,
        score_dtype,
        tla.AddressSpace.l0b,
        512,
    )
    l0_c_ptr = tla.allocate(
        MIXED_TILE_M * query_heads * page_size,
        tla.Float32,
        tla.AddressSpace.l0c,
        512,
    )
    raw_ub_ptr = tla.allocate(
        MIXED_TILE_M * query_heads * page_size,
        tla.Float32,
        tla.AddressSpace.ub,
        256,
    )
    score_ub_ptr = tla.allocate(
        (MIXED_TILE_M // 2) * page_size,
        score_dtype,
        tla.AddressSpace.ub,
        256,
    )
    selected_lane_ptr = tla.allocate(640, tla.Float16, tla.AddressSpace.ub, 256)
    partial_ptr = tla.allocate(1536, tla.Int32, tla.AddressSpace.ub, 256)
    with tla.cube():
        tla.set_flag(l1_q_available)
        tla.set_flag(l1_k_available)
        tla.set_flag(l0_q_available)
        tla.set_flag(l0_k_available)
        tla.set_flag(l0_c_available)
        for cube_task in tla.range(
            tla.arch.block_idx(), task_count, tla.arch.block_num()
        ):
            cube_row_start = cube_task * MIXED_TILE_M
            cube_task_tile = tla.tile_view(
                raw_workspace,
                tla.make_shape(MIXED_TILE_M, page_size),
                tla.make_coord(cube_task, 0),
            )
            cube_actual_m = cube_task_tile.origin_shape[0]
            cube_last_row = cube_row_start + cube_actual_m - 1
            cube_max_visible = (query_positions[cube_last_row] + 1) // compress_ratio
            cube_page_count = (cube_max_visible + page_size - 1) // page_size
            for cube_page in tla.range(0, cube_page_count, 1):
                physical_page = block_table[0, cube_page]
                tla.cross_core_wait_flag(aiv_to_aic, tla.arch.FIX, aiv_id=0)
                tla.cross_core_wait_flag(aiv_to_aic, tla.arch.FIX, aiv_id=1)

                raw_gm_tile = tla.tile_view(
                    raw_workspace,
                    tla.make_shape(MIXED_TILE_M, page_size),
                    tla.make_coord(cube_task, cube_page),
                )
                actual_m = raw_gm_tile.origin_shape[0]
                matrix_m = actual_m * query_heads
                q_offset = cube_row_start * query_heads * head_dim
                k_offset = physical_page * page_size * kv_heads * head_dim
                gm_q = tla.make_tensor(
                    q.ptr + q_offset,
                    tla.make_layout(
                        tla.make_shape(matrix_m, head_dim),
                        tla.make_stride(head_dim, 1),
                        origin_shape=tla.make_shape(matrix_m, head_dim),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                gm_k = tla.make_tensor(
                    compressed_k.ptr + k_offset,
                    tla.make_layout(
                        tla.make_shape(head_dim, page_size),
                        tla.make_stride(1, kv_heads * head_dim),
                        origin_shape=tla.make_shape(head_dim, page_size),
                        layoutTag=tla.arch.ColumnMajor,
                    ),
                )
                raw_matrix = tla.make_tensor(
                    raw_workspace.ptr
                    + cube_row_start * max_blocks
                    + cube_page * page_size,
                    tla.make_layout(
                        tla.make_shape(matrix_m, page_size),
                        tla.make_stride(page_size, 1),
                        origin_shape=tla.make_shape(matrix_m, page_size),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                l1_q = tla.make_tensor_like(l1_q_ptr, gm_q, tla.arch.zN)
                l1_k = tla.make_tensor_like(l1_k_ptr, gm_k, tla.arch.nZ)
                tla.wait_flag(l1_q_available)
                tla.copy(l1_q, gm_q)
                tla.set_flag(l1_q_loaded)
                tla.wait_flag(l1_k_available)
                tla.copy(l1_k, gm_k)
                tla.set_flag(l1_k_loaded)

                l0_q = tla.make_tensor_like(l0_q_ptr, l1_q, tla.arch.zN)
                l0_k = tla.make_tensor_like(l0_k_ptr, l1_k, tla.arch.nZ)
                tla.wait_flag(l1_q_loaded)
                tla.wait_flag(l0_q_available)
                tla.copy(l0_q, l1_q)
                tla.set_flag(l1_q_available)
                tla.set_flag(l0_q_loaded)
                tla.wait_flag(l1_k_loaded)
                tla.wait_flag(l0_k_available)
                tla.copy(l0_k, l1_k)
                tla.set_flag(l1_k_available)
                tla.set_flag(l0_k_loaded)
                tla.wait_flag(l0_q_loaded)
                tla.wait_flag(l0_k_loaded)
                tla.wait_flag(l0_c_available)

                l0_c = tla.make_tensor_like(l0_c_ptr, raw_matrix, tla.arch.L0Clayout)
                tla.mmad(l0_c, l0_q, l0_k, init_c=True)
                tla.set_flag(l0_q_available)
                tla.set_flag(l0_k_available)
                tla.set_flag(mmad_done)
                tla.wait_flag(mmad_done)

                raw_ub = tla.make_tensor(
                    raw_ub_ptr,
                    tla.make_layout(
                        tla.make_shape(matrix_m, page_size),
                        tla.make_stride(page_size, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    raw_ub,
                    l0_c,
                    tla.params.CopyL0C2DstParams(
                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                    ),
                )
                tla.set_flag(l0_c_available)
                tla.cross_core_set_flag(aic_to_aiv, tla.arch.FIX, aiv_id=0)
                tla.cross_core_set_flag(aic_to_aiv, tla.arch.FIX, aiv_id=1)

        tla.wait_flag(l1_q_available)
        tla.wait_flag(l1_k_available)
        tla.wait_flag(l0_q_available)
        tla.wait_flag(l0_k_available)
        tla.wait_flag(l0_c_available)
        tla.cross_core_wait_flag(aiv_to_aic, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(aiv_to_aic, tla.arch.FIX, aiv_id=1)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        aiv_subblock = tla.arch.sub_block_idx()
        tla.cross_core_set_flag(aiv_to_aic, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(aiv_to_aic, tla.arch.VECTOR, aiv_id=1)

        for vector_task in tla.range(
            tla.arch.block_idx(), task_count, tla.arch.block_num()
        ):
            vector_row_start = vector_task * MIXED_TILE_M
            raw_gm_tile = tla.tile_view(
                raw_workspace,
                tla.make_shape(MIXED_TILE_M, page_size),
                tla.make_coord(vector_task, 0),
            )
            actual_m = raw_gm_tile.origin_shape[0]
            aiv_m_request = (actual_m + 1) // 2
            aiv_row_start = vector_row_start + aiv_subblock * aiv_m_request
            aiv_gm_tile = tla.tile_view(
                raw_gm_tile,
                tla.make_shape(aiv_m_request, page_size),
                tla.make_coord(aiv_subblock, 0),
            )
            aiv_m = aiv_gm_tile.origin_shape[0]
            score_ub = tla.make_tensor(
                score_ub_ptr,
                tla.make_layout(
                    tla.make_shape(MIXED_TILE_M // 2, page_size),
                    tla.make_stride(page_size, 1),
                    origin_shape=tla.make_shape(aiv_m, page_size),
                ),
            )
            vector_last_row = vector_row_start + actual_m - 1
            vector_max_visible = (
                query_positions[vector_last_row] + 1
            ) // compress_ratio
            vector_page_count = (vector_max_visible + page_size - 1) // page_size

            for vector_page in tla.range(0, vector_page_count, 1):
                tla.cross_core_wait_flag(aic_to_aiv, tla.arch.VECTOR, aiv_id=0)
                tla.cross_core_wait_flag(aic_to_aiv, tla.arch.VECTOR, aiv_id=1)
                raw_aiv = tla.make_tensor(
                    raw_ub_ptr,
                    tla.make_layout(
                        tla.make_shape(
                            (MIXED_TILE_M // 2) * query_heads,
                            page_size,
                        ),
                        tla.make_stride(page_size, 1),
                        origin_shape=tla.make_shape(aiv_m * query_heads, page_size),
                    ),
                )

                with tla.vec.func(mode="simd"):
                    zero_reg = tla.full(0.0, tla.Float32)
                    score_cast = tla.params.CastParams(
                        reg_slot=tla.params.RegSlot.ZERO,
                        sat_mode=tla.params.SatMode.NOSAT,
                        round_mode=tla.params.RoundMode.CAST_RINT,
                    )
                    score_pack_store = tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_PACK_B32
                    )
                    for score_row in tla.range(0, aiv_m, 1):
                        for score_chunk in tla.range_constexpr(
                            0,
                            (page_size + SCORE_REG_WIDTH - 1) // SCORE_REG_WIDTH,
                            1,
                        ):
                            active_width = page_size - (score_chunk * SCORE_REG_WIDTH)
                            if active_width > SCORE_REG_WIDTH:
                                active_width = SCORE_REG_WIDTH
                            active_mask, _ = tla.update_mask(active_width, tla.Float32)
                            score_store_mask, _ = tla.update_mask(
                                active_width * 2, score_dtype
                            )
                            raw_head = tla.tile_view(
                                raw_aiv,
                                tla.make_shape(1, SCORE_REG_WIDTH),
                                tla.make_coord(score_row * query_heads, score_chunk),
                            )
                            score_reg = tla.max(
                                raw_head.load(), zero_reg, mask=active_mask
                            )
                            if query_heads == 4:
                                raw_head = tla.tile_view(
                                    raw_aiv,
                                    tla.make_shape(1, SCORE_REG_WIDTH),
                                    tla.make_coord(
                                        score_row * query_heads + 1,
                                        score_chunk,
                                    ),
                                )
                                score_reg = tla.add(
                                    score_reg,
                                    tla.max(
                                        raw_head.load(),
                                        zero_reg,
                                        mask=active_mask,
                                    ),
                                    mask=active_mask,
                                )
                                raw_head = tla.tile_view(
                                    raw_aiv,
                                    tla.make_shape(1, SCORE_REG_WIDTH),
                                    tla.make_coord(
                                        score_row * query_heads + 2,
                                        score_chunk,
                                    ),
                                )
                                score_reg_high = tla.max(
                                    raw_head.load(), zero_reg, mask=active_mask
                                )
                                raw_head = tla.tile_view(
                                    raw_aiv,
                                    tla.make_shape(1, SCORE_REG_WIDTH),
                                    tla.make_coord(
                                        score_row * query_heads + 3,
                                        score_chunk,
                                    ),
                                )
                                score_reg_high = tla.add(
                                    score_reg_high,
                                    tla.max(
                                        raw_head.load(),
                                        zero_reg,
                                        mask=active_mask,
                                    ),
                                    mask=active_mask,
                                )
                                score_reg = tla.add(
                                    score_reg,
                                    score_reg_high,
                                    mask=active_mask,
                                )
                            elif query_heads == 8:
                                raw_head = tla.tile_view(
                                    raw_aiv,
                                    tla.make_shape(1, SCORE_REG_WIDTH),
                                    tla.make_coord(
                                        score_row * query_heads + 1,
                                        score_chunk,
                                    ),
                                )
                                score_reg = tla.add(
                                    score_reg,
                                    tla.max(
                                        raw_head.load(),
                                        zero_reg,
                                        mask=active_mask,
                                    ),
                                    mask=active_mask,
                                )
                                raw_head = tla.tile_view(
                                    raw_aiv,
                                    tla.make_shape(1, SCORE_REG_WIDTH),
                                    tla.make_coord(
                                        score_row * query_heads + 2,
                                        score_chunk,
                                    ),
                                )
                                score_reg_high = tla.max(
                                    raw_head.load(), zero_reg, mask=active_mask
                                )
                                raw_head = tla.tile_view(
                                    raw_aiv,
                                    tla.make_shape(1, SCORE_REG_WIDTH),
                                    tla.make_coord(
                                        score_row * query_heads + 3,
                                        score_chunk,
                                    ),
                                )
                                score_reg_high = tla.add(
                                    score_reg_high,
                                    tla.max(
                                        raw_head.load(),
                                        zero_reg,
                                        mask=active_mask,
                                    ),
                                    mask=active_mask,
                                )
                                score_reg = tla.add(
                                    score_reg,
                                    score_reg_high,
                                    mask=active_mask,
                                )
                                raw_head = tla.tile_view(
                                    raw_aiv,
                                    tla.make_shape(1, SCORE_REG_WIDTH),
                                    tla.make_coord(
                                        score_row * query_heads + 4,
                                        score_chunk,
                                    ),
                                )
                                score_reg_high = tla.max(
                                    raw_head.load(), zero_reg, mask=active_mask
                                )
                                raw_head = tla.tile_view(
                                    raw_aiv,
                                    tla.make_shape(1, SCORE_REG_WIDTH),
                                    tla.make_coord(
                                        score_row * query_heads + 5,
                                        score_chunk,
                                    ),
                                )
                                score_reg_high = tla.add(
                                    score_reg_high,
                                    tla.max(
                                        raw_head.load(),
                                        zero_reg,
                                        mask=active_mask,
                                    ),
                                    mask=active_mask,
                                )
                                raw_head = tla.tile_view(
                                    raw_aiv,
                                    tla.make_shape(1, SCORE_REG_WIDTH),
                                    tla.make_coord(
                                        score_row * query_heads + 6,
                                        score_chunk,
                                    ),
                                )
                                score_reg_tail = tla.max(
                                    raw_head.load(), zero_reg, mask=active_mask
                                )
                                raw_head = tla.tile_view(
                                    raw_aiv,
                                    tla.make_shape(1, SCORE_REG_WIDTH),
                                    tla.make_coord(
                                        score_row * query_heads + 7,
                                        score_chunk,
                                    ),
                                )
                                score_reg_tail = tla.add(
                                    score_reg_tail,
                                    tla.max(
                                        raw_head.load(),
                                        zero_reg,
                                        mask=active_mask,
                                    ),
                                    mask=active_mask,
                                )
                                score_reg_high = tla.add(
                                    score_reg_high,
                                    score_reg_tail,
                                    mask=active_mask,
                                )
                                score_reg = tla.add(
                                    score_reg,
                                    score_reg_high,
                                    mask=active_mask,
                                )
                            else:
                                for vector_head in tla.range_constexpr(
                                    1, query_heads, 1
                                ):
                                    raw_head = tla.tile_view(
                                        raw_aiv,
                                        tla.make_shape(1, SCORE_REG_WIDTH),
                                        tla.make_coord(
                                            score_row * query_heads + vector_head,
                                            score_chunk,
                                        ),
                                    )
                                    score_reg = tla.add(
                                        score_reg,
                                        tla.max(
                                            raw_head.load(),
                                            zero_reg,
                                            mask=active_mask,
                                        ),
                                        mask=active_mask,
                                    )
                            score_chunk_ub = tla.tile_view(
                                score_ub,
                                tla.make_shape(1, SCORE_REG_WIDTH),
                                tla.make_coord(score_row, score_chunk),
                            )
                            score_chunk_ub.store(
                                score_reg.to(
                                    score_dtype,
                                    score_cast,
                                    active_mask,
                                ),
                                score_pack_store,
                                mask=score_store_mask,
                            )

                score_ub_bits_ptr = tla.recast_ptr(score_ub_ptr, dtype=tla.Int16)
                score_ub_bits = tla.make_tensor(
                    score_ub_bits_ptr,
                    tla.make_layout(
                        tla.make_shape(aiv_m, page_size),
                        tla.make_stride(page_size, 1),
                    ),
                )
                score_gm_page = tla.make_tensor(
                    score_workspace.ptr
                    + aiv_row_start * max_blocks
                    + vector_page * page_size,
                    tla.make_layout(
                        tla.make_shape(aiv_m, page_size),
                        tla.make_stride(max_blocks, 1),
                        origin_shape=tla.make_shape(aiv_m, page_size),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.pipe_barrier(tla.pipes.ALL)
                tla.copy(score_gm_page, score_ub_bits)
                tla.pipe_barrier(tla.pipes.ALL)

                if vector_page + 1 < vector_page_count:
                    tla.cross_core_set_flag(aiv_to_aic, tla.arch.VECTOR, aiv_id=0)
                    tla.cross_core_set_flag(aiv_to_aic, tla.arch.VECTOR, aiv_id=1)

            tla.pipe_barrier(tla.pipes.ALL)

            for aiv_row in tla.range(0, aiv_m, 1):
                row = aiv_row_start + aiv_row
                position = query_positions[row]
                visible = (position + 1) // compress_ratio
                tail = (position + 1) % compress_ratio
                keep = visible
                if visible > topk_limit:
                    keep = tla.Int32(topk_limit)
                topk_input_ptr = tla.recast_ptr(raw_ub_ptr, dtype=tla.Int16)
                topk_input = tla.make_tensor(
                    topk_input_ptr,
                    tla.make_layout(
                        tla.make_shape(max_blocks_aligned),
                        tla.make_stride(1),
                    ),
                )
                topk_input_f16 = tla.make_tensor(
                    tla.recast_ptr(topk_input_ptr, dtype=tla.Float16),
                    tla.make_layout(
                        tla.make_shape(max_blocks_aligned),
                        tla.make_stride(1),
                    ),
                )
                selected_lanes = tla.make_tensor(
                    selected_lane_ptr,
                    tla.make_layout(tla.make_shape(640), tla.make_stride(1)),
                )
                histogram_state = tla.make_tensor(
                    tla.recast_ptr(partial_ptr, dtype=tla.Int16),
                    tla.make_layout(tla.make_shape(3072), tla.make_stride(1)),
                )
                histogram_state_u16 = tla.make_tensor(
                    tla.recast_ptr(partial_ptr, dtype=tla.UInt16),
                    tla.make_layout(tla.make_shape(3072), tla.make_stride(1)),
                )

                if visible > topk_limit:
                    topk_source = tla.make_tensor(
                        score_workspace.ptr + row * max_blocks,
                        tla.make_layout(
                            tla.make_shape(max_blocks),
                            tla.make_stride(1),
                        ),
                    )
                    tla.copy(topk_input, topk_source)
                    tla.pipe_barrier(tla.pipes.ALL)

                    if tla.const_expr(True):
                        visible_aligned = ((visible + 255) // 256) * 256
                        histogram_chunks = visible_aligned // 256
                        with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                            pad_thread_idx, _, _ = tla.arch.thread_idx()
                            pad_thread_count, _, _ = tla.arch.thread_block_dim()
                            for pad_index in tla.range(
                                visible + pad_thread_idx,
                                visible_aligned,
                                pad_thread_count,
                            ):
                                topk_input[pad_index] = 0
                            selected_lanes[pad_thread_idx] = -1.0
                            selected_lanes[pad_thread_idx + pad_thread_count] = -1.0
                            if pad_thread_idx == 0:
                                histogram_state[256] = (visible_aligned - keep + 1).to(
                                    tla.Int16
                                )
                        tla.pipe_barrier(tla.pipes.ALL)

                        with tla.vec.func(mode="simd"):
                            all_u8 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.UInt8
                            )
                            all_u16 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.UInt16
                            )
                            low_half_u8 = tla.create_mask(
                                pattern=tla.mask.H, dtype=tla.UInt8
                            )
                            histogram_low = tla.full(0, tla.Int16).bitcast(tla.UInt16)
                            histogram_high = tla.full(0, tla.Int16).bitcast(tla.UInt16)
                            high_histogram_low_params = tla.params.HistogramsParams(
                                mode=tla.params.HistogramsBinType.BIN0,
                                type=tla.params.HistogramsType.ACCUMULATE,
                            )
                            high_histogram_high_params = tla.params.HistogramsParams(
                                mode=tla.params.HistogramsBinType.BIN1,
                                type=tla.params.HistogramsType.ACCUMULATE,
                            )
                            for histogram_chunk in tla.range(0, histogram_chunks, 1):
                                score_low = tla.tile_view(
                                    topk_input,
                                    tla.make_shape(128),
                                    tla.make_coord(histogram_chunk * 2),
                                ).load()
                                score_high = tla.tile_view(
                                    topk_input,
                                    tla.make_shape(128),
                                    tla.make_coord(histogram_chunk * 2 + 1),
                                ).load()
                                high_byte_low = tla.pack(
                                    tla.shift_right(score_low, 8),
                                    part=tla.RegPart.LOW,
                                )
                                high_byte_high = tla.pack(
                                    tla.shift_right(score_high, 8),
                                    part=tla.RegPart.HIGH,
                                )
                                high_bytes = tla.where(
                                    low_half_u8,
                                    high_byte_low,
                                    high_byte_high,
                                )
                                histogram_low = tla.histograms(
                                    high_bytes,
                                    histogram_low,
                                    high_histogram_low_params,
                                    mask=all_u8,
                                )
                                histogram_high = tla.histograms(
                                    high_bytes,
                                    histogram_high,
                                    high_histogram_high_params,
                                    mask=all_u8,
                                )
                            tla.tile_view(
                                histogram_state_u16,
                                tla.make_shape(128),
                                tla.make_coord(0),
                            ).store(histogram_low)
                            tla.tile_view(
                                histogram_state_u16,
                                tla.make_shape(128),
                                tla.make_coord(1),
                            ).store(histogram_high)
                            rank = (
                                tla.tile_view(
                                    histogram_state,
                                    tla.make_shape(1),
                                    tla.make_coord(256),
                                )
                                .load(
                                    tla.params.NormalLoadParams(
                                        load_dist=tla.params.LoadDist.DIST_BRC_B16
                                    )
                                )
                                .bitcast(tla.UInt16)
                            )
                            signed_lanes = tla.arange(dtype=tla.Int16)
                            lanes = signed_lanes.bitcast(tla.UInt16)
                            sentinel = tla.full(-1, tla.Int16).bitcast(tla.UInt16)
                            low_candidates = tla.where(
                                tla.cmp(histogram_low, rank, "ge"),
                                lanes,
                                sentinel,
                            )
                            high_candidates = tla.where(
                                tla.cmp(histogram_high, rank, "ge"),
                                tla.add(signed_lanes, 128).bitcast(tla.UInt16),
                                sentinel,
                            )
                            high_threshold = tla.min(
                                low_candidates, high_candidates
                            ).reduce(tla.ReductionOp.MIN, mask=all_u16)
                            tla.tile_view(
                                histogram_state_u16,
                                tla.make_shape(1),
                                tla.make_coord(272),
                            ).store(high_threshold)
                        tla.pipe_barrier(tla.pipes.ALL)

                        with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                            rank_thread_idx, _, _ = tla.arch.thread_idx()
                            if rank_thread_idx == 0:
                                high_threshold_scalar = histogram_state[272].to(
                                    tla.Int32
                                )
                                previous_count = tla.Int32(0)
                                if high_threshold_scalar > 0:
                                    previous_count = histogram_state[
                                        high_threshold_scalar - 1
                                    ].to(tla.Int32)
                                histogram_state[288] = (
                                    visible_aligned - keep + 1 - previous_count
                                ).to(tla.Int16)
                        tla.pipe_barrier(tla.pipes.ALL)

                        with tla.vec.func(mode="simd"):
                            all_u8 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.UInt8
                            )
                            all_u16 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.UInt16
                            )
                            low_half_u8 = tla.create_mask(
                                pattern=tla.mask.H, dtype=tla.UInt8
                            )
                            low_histogram_low = tla.full(0, tla.Int16).bitcast(
                                tla.UInt16
                            )
                            low_histogram_high = tla.full(0, tla.Int16).bitcast(
                                tla.UInt16
                            )
                            high_threshold_register = tla.tile_view(
                                histogram_state,
                                tla.make_shape(1),
                                tla.make_coord(272),
                            ).load(
                                tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B16
                                )
                            )
                            low_rank = (
                                tla.tile_view(
                                    histogram_state,
                                    tla.make_shape(1),
                                    tla.make_coord(288),
                                )
                                .load(
                                    tla.params.NormalLoadParams(
                                        load_dist=tla.params.LoadDist.DIST_BRC_B16
                                    )
                                )
                                .bitcast(tla.UInt16)
                            )
                            high_threshold_low = tla.pack(
                                high_threshold_register,
                                part=tla.RegPart.LOW,
                            )
                            high_threshold_high = tla.pack(
                                high_threshold_register,
                                part=tla.RegPart.HIGH,
                            )
                            high_threshold_bytes = tla.where(
                                low_half_u8,
                                high_threshold_low,
                                high_threshold_high,
                            )
                            low_histogram_low_params = tla.params.HistogramsParams(
                                mode=tla.params.HistogramsBinType.BIN0,
                                type=tla.params.HistogramsType.ACCUMULATE,
                            )
                            low_histogram_high_params = tla.params.HistogramsParams(
                                mode=tla.params.HistogramsBinType.BIN1,
                                type=tla.params.HistogramsType.ACCUMULATE,
                            )
                            for histogram_chunk in tla.range(0, histogram_chunks, 1):
                                score_low = tla.tile_view(
                                    topk_input,
                                    tla.make_shape(128),
                                    tla.make_coord(histogram_chunk * 2),
                                ).load()
                                score_high = tla.tile_view(
                                    topk_input,
                                    tla.make_shape(128),
                                    tla.make_coord(histogram_chunk * 2 + 1),
                                ).load()
                                low_byte_low = tla.pack(score_low, part=tla.RegPart.LOW)
                                low_byte_high = tla.pack(
                                    score_high, part=tla.RegPart.HIGH
                                )
                                low_bytes = tla.where(
                                    low_half_u8, low_byte_low, low_byte_high
                                )
                                high_byte_low = tla.pack(
                                    tla.shift_right(score_low, 8),
                                    part=tla.RegPart.LOW,
                                )
                                high_byte_high = tla.pack(
                                    tla.shift_right(score_high, 8),
                                    part=tla.RegPart.HIGH,
                                )
                                high_bytes = tla.where(
                                    low_half_u8,
                                    high_byte_low,
                                    high_byte_high,
                                )
                                matching_high = tla.cmp(
                                    high_bytes,
                                    high_threshold_bytes,
                                    "eq",
                                )
                                low_histogram_low = tla.histograms(
                                    low_bytes,
                                    low_histogram_low,
                                    low_histogram_low_params,
                                    mask=matching_high,
                                )
                                low_histogram_high = tla.histograms(
                                    low_bytes,
                                    low_histogram_high,
                                    low_histogram_high_params,
                                    mask=matching_high,
                                )
                            signed_lanes = tla.arange(dtype=tla.Int16)
                            lanes = signed_lanes.bitcast(tla.UInt16)
                            sentinel = tla.full(-1, tla.Int16).bitcast(tla.UInt16)
                            low_candidates = tla.where(
                                tla.cmp(low_histogram_low, low_rank, "ge"),
                                lanes,
                                sentinel,
                            )
                            high_candidates = tla.where(
                                tla.cmp(low_histogram_high, low_rank, "ge"),
                                tla.add(signed_lanes, 128).bitcast(tla.UInt16),
                                sentinel,
                            )
                            low_threshold = tla.min(
                                low_candidates, high_candidates
                            ).reduce(tla.ReductionOp.MIN, mask=all_u16)
                            tla.tile_view(
                                histogram_state_u16,
                                tla.make_shape(1),
                                tla.make_coord(304),
                            ).store(low_threshold)
                        tla.pipe_barrier(tla.pipes.ALL)

                        with tla.vec.func(mode="simd"):
                            all_i16 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Int16
                            )
                            high_threshold_register = tla.tile_view(
                                histogram_state,
                                tla.make_shape(1),
                                tla.make_coord(272),
                            ).load(
                                tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B16
                                )
                            )
                            low_threshold_register = tla.tile_view(
                                histogram_state,
                                tla.make_shape(1),
                                tla.make_coord(304),
                            ).load(
                                tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B16
                                )
                            )
                            threshold_register = tla.add(
                                tla.shift_left(high_threshold_register, 8),
                                low_threshold_register,
                                mask=all_i16,
                            ).bitcast(tla.UInt16)
                            one_register = tla.full(1, tla.Int16)
                            zero_register = tla.full(0, tla.Int16)
                            first_i16_store = tla.params.NormalStoreParams(
                                store_dist=(tla.params.StoreDist.DIST_FIRST_ELEMENT_B16)
                            )
                            gather_chunks = visible_aligned // 128
                            for gather_chunk in tla.range(0, gather_chunks, 1):
                                score_register = (
                                    tla.tile_view(
                                        topk_input,
                                        tla.make_shape(128),
                                        tla.make_coord(gather_chunk),
                                    )
                                    .load()
                                    .bitcast(tla.UInt16)
                                )
                                valid_width = visible - gather_chunk * 128
                                if valid_width > 128:
                                    valid_width = tla.Int32(128)
                                valid_mask, _ = tla.update_mask(valid_width, tla.Int16)
                                greater_mask = tla.bitwise_and(
                                    tla.cmp(
                                        score_register,
                                        threshold_register,
                                        "gt",
                                    ),
                                    valid_mask,
                                )
                                equal_mask = tla.bitwise_and(
                                    tla.cmp(
                                        score_register,
                                        threshold_register,
                                        "eq",
                                    ),
                                    valid_mask,
                                )
                                greater_count = tla.where(
                                    greater_mask,
                                    one_register,
                                    zero_register,
                                ).reduce(
                                    tla.ReductionOp.ADD,
                                    mask=all_i16,
                                )
                                equal_count = tla.where(
                                    equal_mask,
                                    one_register,
                                    zero_register,
                                ).reduce(
                                    tla.ReductionOp.ADD,
                                    mask=all_i16,
                                )
                                tla.tile_view(
                                    histogram_state,
                                    tla.make_shape(1),
                                    tla.make_coord(512 + gather_chunk),
                                ).store(greater_count, first_i16_store)
                                tla.tile_view(
                                    histogram_state,
                                    tla.make_shape(1),
                                    tla.make_coord(768 + gather_chunk),
                                ).store(equal_count, first_i16_store)
                        tla.pipe_barrier(tla.pipes.ALL)

                        with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                            gather_thread_idx, _, _ = tla.arch.thread_idx()
                            gather_thread_count, _, _ = tla.arch.thread_block_dim()
                            gather_chunks = visible_aligned // 128
                            if gather_thread_idx == 0:
                                total_greater = tla.Int32(0)
                                for count_chunk in tla.range(0, gather_chunks, 1):
                                    total_greater = total_greater + histogram_state[
                                        512 + count_chunk
                                    ].to(tla.Int32)
                                output_offset = tla.Int32(0)
                                remaining_equal = keep - total_greater
                                histogram_state[320] = tla.Int16(-1)
                                for prefix_chunk in tla.range(0, gather_chunks, 1):
                                    chunk_greater = histogram_state[
                                        512 + prefix_chunk
                                    ].to(tla.Int32)
                                    available_equal = histogram_state[
                                        768 + prefix_chunk
                                    ].to(tla.Int32)
                                    chunk_equal = available_equal
                                    if chunk_equal > remaining_equal:
                                        chunk_equal = remaining_equal
                                    if (
                                        chunk_equal > 0
                                        and chunk_equal < available_equal
                                    ):
                                        histogram_state[320] = prefix_chunk.to(
                                            tla.Int16
                                        )
                                    histogram_state[1280 + prefix_chunk] = (
                                        chunk_equal.to(tla.Int16)
                                    )
                                    histogram_state[1024 + prefix_chunk] = (
                                        output_offset.to(tla.Int16)
                                    )
                                    output_offset = (
                                        output_offset + chunk_greater + chunk_equal
                                    )
                                    remaining_equal = remaining_equal - chunk_equal
                                if tla.const_expr(max_blocks > 512):
                                    partial_chunk = histogram_state[320].to(tla.Int32)
                                    if partial_chunk >= 0:
                                        partial_threshold = (
                                            histogram_state[272].to(tla.Int32) << 8
                                        ) + histogram_state[304].to(tla.Int32)
                                        partial_write_offset = histogram_state[
                                            1024 + partial_chunk
                                        ].to(tla.Int32)
                                        partial_equal_quota = histogram_state[
                                            1280 + partial_chunk
                                        ].to(tla.Int32)
                                        partial_start = partial_chunk * 128
                                        partial_end = partial_start + 128
                                        if partial_end > visible:
                                            partial_end = visible
                                        for partial_candidate in tla.range(
                                            partial_start, partial_end, 1
                                        ):
                                            partial_score = topk_input[
                                                partial_candidate
                                            ].to(tla.Int32)
                                            partial_take = (
                                                partial_score > partial_threshold
                                            )
                                            if (
                                                partial_score == partial_threshold
                                                and partial_equal_quota > 0
                                            ):
                                                partial_take = True
                                                partial_equal_quota = (
                                                    partial_equal_quota - 1
                                                )
                                            if partial_take:
                                                selected_lanes[partial_write_offset] = (
                                                    partial_candidate - partial_start
                                                ).to(tla.Float16)
                                                partial_write_offset = (
                                                    partial_write_offset + 1
                                                )
                            tla.arch.sync_threads()

                            if tla.const_expr(max_blocks <= 512):
                                threshold = (
                                    histogram_state[272].to(tla.Int32) << 8
                                ) + histogram_state[304].to(tla.Int32)
                                for write_chunk in tla.range(
                                    gather_thread_idx,
                                    gather_chunks,
                                    gather_thread_count,
                                ):
                                    write_offset = histogram_state[
                                        1024 + write_chunk
                                    ].to(tla.Int32)
                                    equal_quota = histogram_state[
                                        1280 + write_chunk
                                    ].to(tla.Int32)
                                    write_start = write_chunk * 128
                                    write_end = write_start + 128
                                    if write_end > visible:
                                        write_end = visible
                                    for write_candidate in tla.range(
                                        write_start, write_end, 1
                                    ):
                                        write_score_bits = topk_input[
                                            write_candidate
                                        ].to(tla.Int32)
                                        take = write_score_bits > threshold
                                        if (
                                            write_score_bits == threshold
                                            and equal_quota > 0
                                        ):
                                            take = True
                                            equal_quota = equal_quota - 1
                                        if take:
                                            selected_lanes[write_offset] = (
                                                write_candidate.to(tla.Float16)
                                            )
                                            write_offset = write_offset + 1
                            tla.arch.sync_threads()
                    tla.pipe_barrier(tla.pipes.ALL)

                    if tla.const_expr(max_blocks > 512):
                        with tla.vec.func(mode="simd"):
                            all_i16 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Int16
                            )
                            cast_i16_to_f16 = tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.ZERO,
                                sat_mode=tla.params.SatMode.NOSAT,
                                round_mode=tla.params.RoundMode.CAST_ROUND,
                            )
                            high_threshold_register = tla.tile_view(
                                histogram_state,
                                tla.make_shape(1),
                                tla.make_coord(272),
                            ).load(
                                tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B16
                                )
                            )
                            low_threshold_register = tla.tile_view(
                                histogram_state,
                                tla.make_shape(1),
                                tla.make_coord(304),
                            ).load(
                                tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B16
                                )
                            )
                            threshold_register = tla.add(
                                tla.shift_left(high_threshold_register, 8),
                                low_threshold_register,
                                mask=all_i16,
                            ).bitcast(tla.UInt16)
                            lane_indices = tla.arange(dtype=tla.Int16).to(
                                tla.Float16,
                                cast_i16_to_f16,
                                all_i16,
                            )
                            gather_chunks = visible_aligned // 128
                            for squeeze_chunk in tla.range(0, gather_chunks, 1):
                                score_register = (
                                    tla.tile_view(
                                        topk_input,
                                        tla.make_shape(128),
                                        tla.make_coord(squeeze_chunk),
                                    )
                                    .load()
                                    .bitcast(tla.UInt16)
                                )
                                valid_width = visible - squeeze_chunk * 128
                                if valid_width > 128:
                                    valid_width = tla.Int32(128)
                                valid_mask, _ = tla.update_mask(valid_width, tla.Int16)
                                greater_mask = tla.bitwise_and(
                                    tla.cmp(
                                        score_register,
                                        threshold_register,
                                        "gt",
                                    ),
                                    valid_mask,
                                )
                                equal_mask = tla.bitwise_and(
                                    tla.cmp(
                                        score_register,
                                        threshold_register,
                                        "eq",
                                    ),
                                    valid_mask,
                                )
                                squeeze_equal_quota = histogram_state[
                                    1280 + squeeze_chunk
                                ].to(tla.Int32)
                                take_mask = greater_mask
                                if squeeze_equal_quota > 0:
                                    take_mask = tla.bitwise_or(greater_mask, equal_mask)
                                packed_lanes = tla.squeeze(lane_indices, take_mask)
                                squeeze_greater_count = histogram_state[
                                    512 + squeeze_chunk
                                ].to(tla.Int32)
                                squeeze_selected_count = (
                                    squeeze_greater_count + squeeze_equal_quota
                                )
                                selected_store_mask, _ = tla.update_mask(
                                    squeeze_selected_count, tla.Float16
                                )
                                tla.tile_view(
                                    topk_input_f16,
                                    tla.make_shape(128),
                                    tla.make_coord(squeeze_chunk),
                                ).store(
                                    packed_lanes,
                                    mask=selected_store_mask,
                                )
                        tla.pipe_barrier(tla.pipes.ALL)

                        with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                            chunk_thread_idx, _, _ = tla.arch.thread_idx()
                            chunk_thread_count, _, _ = tla.arch.thread_block_dim()
                            for large_init_column in tla.range(
                                chunk_thread_idx,
                                output_width,
                                chunk_thread_count,
                            ):
                                output[row, large_init_column] = -1
                            tla.arch.sync_threads()

                            gather_chunks = visible_aligned // 128
                            partial_chunk = histogram_state[320].to(tla.Int32)
                            for selected_chunk in tla.range(
                                chunk_thread_idx,
                                gather_chunks,
                                chunk_thread_count,
                            ):
                                chunk_output_offset = histogram_state[
                                    1024 + selected_chunk
                                ].to(tla.Int32)
                                chunk_selected_count = histogram_state[
                                    512 + selected_chunk
                                ].to(tla.Int32) + histogram_state[
                                    1280 + selected_chunk
                                ].to(tla.Int32)
                                chunk_stage_start = selected_chunk * 128
                                for chunk_rank in tla.range(0, chunk_selected_count, 1):
                                    selected_lane = topk_input_f16[
                                        chunk_stage_start + chunk_rank
                                    ]
                                    if selected_chunk == partial_chunk:
                                        selected_lane = selected_lanes[
                                            chunk_output_offset + chunk_rank
                                        ]
                                    selected_block = (
                                        selected_chunk * 128
                                        + selected_lane.to(tla.Int32)
                                    )
                                    selected_rank = chunk_output_offset + chunk_rank
                                    for selected_index in tla.range_constexpr(
                                        0, compress_ratio, 1
                                    ):
                                        output[
                                            row,
                                            selected_rank * compress_ratio
                                            + selected_index,
                                        ] = (
                                            selected_block * compress_ratio
                                            + selected_index
                                        ).to(tla.Int32)
                            if chunk_thread_idx == 0:
                                tail_start = visible * compress_ratio
                                for tail_index in tla.range_constexpr(
                                    0, compress_ratio - 1, 1
                                ):
                                    if tail_index < tail:
                                        output[
                                            row,
                                            keep * compress_ratio + tail_index,
                                        ] = (tail_start + tail_index).to(tla.Int32)
                                output[row, output_width - 1] = (
                                    keep * compress_ratio + tail
                                ).to(tla.Int32)
                            tla.arch.sync_threads()
                        tla.pipe_barrier(tla.pipes.ALL)

                if visible <= topk_limit:
                    with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                        visible_thread_idx, _, _ = tla.arch.thread_idx()
                        visible_thread_count, _, _ = tla.arch.thread_block_dim()
                        for visible_init_column in tla.range(
                            visible_thread_idx,
                            output_width,
                            visible_thread_count,
                        ):
                            output[row, visible_init_column] = -1
                        tla.arch.sync_threads()

                        for visible_column in tla.range(
                            visible_thread_idx,
                            output_width - 1,
                            visible_thread_count,
                        ):
                            if visible_column <= position:
                                output[row, visible_column] = visible_column.to(
                                    tla.Int32
                                )
                        if visible_thread_idx == 0:
                            output[row, output_width - 1] = (
                                keep * compress_ratio + tail
                            ).to(tla.Int32)
                        tla.arch.sync_threads()
                    tla.pipe_barrier(tla.pipes.ALL)
                else:
                    if tla.const_expr(max_blocks <= 512):
                        with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                            topk_thread_idx, _, _ = tla.arch.thread_idx()
                            topk_thread_count, _, _ = tla.arch.thread_block_dim()
                            for topk_init_column in tla.range(
                                topk_thread_idx,
                                output_width,
                                topk_thread_count,
                            ):
                                output[row, topk_init_column] = -1
                            tla.arch.sync_threads()

                            for selected in tla.range(
                                topk_thread_idx,
                                keep,
                                topk_thread_count,
                            ):
                                small_selected_block = selected_lanes[selected].to(
                                    tla.Int32
                                )
                                for small_selected_index in tla.range_constexpr(
                                    0, compress_ratio, 1
                                ):
                                    output[
                                        row,
                                        selected * compress_ratio
                                        + small_selected_index,
                                    ] = (
                                        small_selected_block * compress_ratio
                                        + small_selected_index
                                    ).to(tla.Int32)

                            if topk_thread_idx == 0:
                                small_tail_start = visible * compress_ratio
                                for small_tail_index in tla.range_constexpr(
                                    0, compress_ratio - 1, 1
                                ):
                                    if small_tail_index < tail:
                                        output[
                                            row,
                                            keep * compress_ratio + small_tail_index,
                                        ] = (small_tail_start + small_tail_index).to(
                                            tla.Int32
                                        )
                                output[row, output_width - 1] = (
                                    keep * compress_ratio + tail
                                ).to(tla.Int32)
                            tla.arch.sync_threads()
                        tla.pipe_barrier(tla.pipes.ALL)
            tla.cross_core_set_flag(aiv_to_aic, tla.arch.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(aiv_to_aic, tla.arch.VECTOR, aiv_id=1)
        tla.pipe_barrier(tla.pipes.ALL)
