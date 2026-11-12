# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""StreamK MMAD demo: kernel + host in one file.

Builds and launches the single mixed kernel: the AIC cube runs the full-K normal
tiles and the StreamK partial sums, the AIV vector section reduces the workspace
into GM C, and the result is verified for accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import argparse
from typing import Any, Literal

import catlass.tla as tla
import torch
import torch_npu  # noqa: F401

from helpers import (
    TilingParams,
    SwizzleParams,
)

# Compile-time knobs (formerly streamk_config).
AIV_TILE_M = 16
AIV_SUB_BLOCK_NUM = 2
AIV_REG_M = 1
AIV_REG_N = 64

DESCRIPTION = "StreamK MMAD; dynamic GM."


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.kernel
def streamk_mmad_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    gm_workspace: tla.Tensor,
    _tiling: tla.Constexpr[TilingParams],
    _swizzle: tla.Constexpr[SwizzleParams],
    block_dim: tla.Constexpr[int],
) -> None:
    c0 = 0
    c1 = 1

    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype
    dtype_gm_c = gm_c.ptr.dtype
    DTYPE_C = tla.Float32  # L0C accumulator only

    aiv_m_chunks = AIV_TILE_M // AIV_REG_M
    aiv_n_chunks = _tiling.l1_tn // AIV_REG_N

    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    k = gm_a.origin_shape[1]

    loops_m = (m + _tiling.l1_tm - 1) // _tiling.l1_tm
    loops_n = (n + _tiling.l1_tn - 1) // _tiling.l1_tn
    loops_k = (k + _tiling.l1_tk - 1) // _tiling.l1_tk
    total_mn = loops_m * loops_n
    streamk_blocks = total_mn % block_dim
    normal_blocks = total_mn - streamk_blocks
    k_tile_num_per_core = (streamk_blocks * loops_k) // block_dim
    k_tile_remain = (streamk_blocks * loops_k) % block_dim
    core_loops = (total_mn // block_dim) * block_dim + min(
        streamk_blocks * loops_k, block_dim
    )
    streamk_cores = core_loops - normal_blocks

    l1a0_data_ready = tla.flag("l1a0_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1a1_data_ready = tla.flag("l1a1_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1b0_data_ready = tla.flag("l1b0_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1b1_data_ready = tla.flag("l1b1_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1a0_available = tla.flag("l1a0_available", tla.arch.MTE1, tla.arch.MTE2)
    l1a1_available = tla.flag("l1a1_available", tla.arch.MTE1, tla.arch.MTE2)
    l1b0_available = tla.flag("l1b0_available", tla.arch.MTE1, tla.arch.MTE2)
    l1b1_available = tla.flag("l1b1_available", tla.arch.MTE1, tla.arch.MTE2)
    l0a0_available = tla.flag("l0a0_available", tla.arch.CUBE, tla.arch.MTE1)
    l0a1_available = tla.flag("l0a1_available", tla.arch.CUBE, tla.arch.MTE1)
    l0b0_available = tla.flag("l0b0_available", tla.arch.CUBE, tla.arch.MTE1)
    l0b1_available = tla.flag("l0b1_available", tla.arch.CUBE, tla.arch.MTE1)
    l0_ab_data_ready = tla.flag("l0_ab_data_ready", tla.arch.MTE1, tla.arch.CUBE)
    l0c_available = tla.flag("l0c_available", tla.arch.FIX, tla.arch.CUBE)
    # AIC→AIV handoff (mode 2) plus an all-AIV barrier (mode 0).
    aic_finish = tla.cross_flag("aic_finish", mode=2)
    aiv_ibarrier = tla.cross_flag("aiv_ibarrier", mode=0)

    l1a0_ptr = tla.allocate(
        _tiling.l1_tm * _tiling.l1_tk, dtype_a, tla.AddressSpace.l1, 512
    )
    l1a1_ptr = tla.allocate(
        _tiling.l1_tm * _tiling.l1_tk, dtype_a, tla.AddressSpace.l1, 512
    )
    l1b0_ptr = tla.allocate(
        _tiling.l1_tk * _tiling.l1_tn, dtype_b, tla.AddressSpace.l1, 512
    )
    l1b1_ptr = tla.allocate(
        _tiling.l1_tk * _tiling.l1_tn, dtype_b, tla.AddressSpace.l1, 512
    )

    l0a0_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tk, dtype_a, tla.AddressSpace.l0a, 512
    )
    l0a1_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tk, dtype_a, tla.AddressSpace.l0a, 512
    )
    l0b0_ptr = tla.allocate(
        _tiling.l0_tk * _tiling.l0_tn, dtype_b, tla.AddressSpace.l0b, 512
    )
    l0b1_ptr = tla.allocate(
        _tiling.l0_tk * _tiling.l0_tn, dtype_b, tla.AddressSpace.l0b, 512
    )

    l0c_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tn, DTYPE_C, tla.AddressSpace.l0c, 1024
    )

    aiv_ub_tile_elems = AIV_TILE_M * _tiling.l1_tn
    aiv_acc_ptr = tla.allocate(aiv_ub_tile_elems, DTYPE_C, tla.AddressSpace.ub, 256)
    aiv_temp_ptr = tla.allocate(aiv_ub_tile_elems, DTYPE_C, tla.AddressSpace.ub, 256)
    aiv_out_ptr = tla.allocate(aiv_ub_tile_elems, dtype_gm_c, tla.AddressSpace.ub, 256)
    aiv_ub_layout = tla.make_layout(
        tla.make_shape(AIV_TILE_M, _tiling.l1_tn),
        tla.make_stride(_tiling.l1_tn, c1),
        layoutTag=tla.arch.RowMajor,
    )
    aiv_acc_ub = tla.make_tensor(aiv_acc_ptr, aiv_ub_layout)
    aiv_temp_ub = tla.make_tensor(aiv_temp_ptr, aiv_ub_layout)
    aiv_out_ub = tla.make_tensor(aiv_out_ptr, aiv_ub_layout)

    aiv_loaded = tla.flag("aiv_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    aiv_vec_to_mte2 = tla.flag("aiv_vec_to_mte2", tla.arch.VECTOR, tla.arch.MTE2)
    aiv_done = tla.flag("aiv_done", tla.arch.VECTOR, tla.arch.MTE3)

    with tla.cube():
        # Drain pipe state left behind by a previous launch.
        tla.pipe_barrier(tla.pipes.ALL)

        tla.set_flag(l1a0_available)
        tla.set_flag(l1a1_available)
        tla.set_flag(l1b0_available)
        tla.set_flag(l1b1_available)
        tla.set_flag(l0a0_available)
        tla.set_flag(l0a1_available)
        tla.set_flag(l0b0_available)
        tla.set_flag(l0b1_available)
        tla.set_flag(l0c_available)

        l1_buf_idx = c0
        l0_buf_idx = c0
        block_idx = tla.arch.block_idx()

        # Cores without a StreamK task write only GM C, so nothing in the vector
        # section depends on them: release the paired AIV up front.
        if block_idx >= streamk_cores:
            tla.cross_core_set_flag(aic_finish, tla.arch.FIX)

        # Unified AIC schedule matching C++ streamk_matmul_tla:
        # remap loopIdx → actualLoopIdx (swap last normal with StreamK), then
        # GetStreamkBlockDec; CrossCoreSetFlag on PIPE_FIX after SK (no barrier).
        loop_range = tla.range(block_idx, core_loops, tla.arch.block_num())
        for loop_idx in loop_range:
            # actualLoopIdx remap (same conditions as streamk_matmul_tla.hpp).
            actual_loop_idx = loop_idx
            if normal_blocks > 0:
                if block_idx < streamk_cores:
                    swap_at = normal_blocks - tla.arch.block_num() + block_idx
                    if loop_idx == swap_at:
                        actual_loop_idx = normal_blocks + block_idx
                    elif loop_idx >= normal_blocks:
                        actual_loop_idx = swap_at
                elif loop_idx >= normal_blocks:
                    actual_loop_idx = normal_blocks - tla.arch.block_num() + block_idx
            # Same branch as GetStreamkBlockDec: SK when actual >= normal_blocks
            # (after remap, deferred normals land below normal_blocks again).
            is_sk_now = False
            if normal_blocks > 0:
                if actual_loop_idx >= normal_blocks:
                    is_sk_now = True
            else:
                is_sk_now = True

            # Prepare swizzle coordinates for normal and StreamK blocks.
            # Swizzle Direction fixed to RowMajor (0).
            swizzle_span = _swizzle.SWIZZLE_OFFSET * loops_n
            swizzle_tb_loop = (
                loops_m + _swizzle.SWIZZLE_OFFSET - 1
            ) // _swizzle.SWIZZLE_OFFSET
            tile_block_idx = actual_loop_idx // swizzle_span
            in_tile = actual_loop_idx % swizzle_span
            n_row = _swizzle.SWIZZLE_OFFSET
            if tile_block_idx == (swizzle_tb_loop - 1):
                n_row = loops_m - _swizzle.SWIZZLE_OFFSET * tile_block_idx
            block_row = tile_block_idx * _swizzle.SWIZZLE_OFFSET + in_tile % n_row
            block_col = in_tile // n_row
            if (tile_block_idx % 2) == 1:
                block_col = loops_n - block_col - 1

            # Pre-init: normal / SK-non-cross use sk_slot_count=1; SK cross uses 2.
            block_k = c0
            block_actual_k = k
            streamk_block_row = block_row
            streamk_block_col = block_col
            streamk_block_k = c0
            streamk_actual_k = c0
            sk_task_id = block_idx
            sk_slot_count = 1

            if is_sk_now:
                # GetStreamkBlockDec StreamK path — coords only here.
                # blockCoord = current MN / K-tail; streamkBlockCoord = next MN / K-head.
                rel = actual_loop_idx - normal_blocks
                cur_k_tile_num = k_tile_num_per_core
                k_tile_idx = rel * k_tile_num_per_core + k_tile_remain
                if rel < k_tile_remain:
                    cur_k_tile_num = k_tile_num_per_core + 1
                    k_tile_idx = rel * cur_k_tile_num

                # --- blockCoord: override hoisted coords with current SK tile ---
                streamk_block_idx = k_tile_idx // loops_k
                block_linear = normal_blocks + streamk_block_idx
                block_tb_idx = block_linear // swizzle_span
                block_in_tile = block_linear % swizzle_span
                block_n_row = _swizzle.SWIZZLE_OFFSET
                if block_tb_idx == (swizzle_tb_loop - 1):
                    block_n_row = loops_m - _swizzle.SWIZZLE_OFFSET * block_tb_idx
                block_row = (
                    block_tb_idx * _swizzle.SWIZZLE_OFFSET + block_in_tile % block_n_row
                )
                block_col = block_in_tile // block_n_row
                if (block_tb_idx % 2) == 1:
                    block_col = loops_n - block_col - 1
                block_k = k_tile_idx % loops_k
                block_actual_k = cur_k_tile_num * _tiling.l1_tk
                if (k_tile_idx % loops_k + cur_k_tile_num) * _tiling.l1_tk > k:
                    block_actual_k = k - (k_tile_idx % loops_k) * _tiling.l1_tk

                # --- streamkBlockCoord: next tile K-head on cross ---
                streamk_block_row = block_row
                streamk_block_col = block_col
                streamk_block_k = c0
                streamk_actual_k = c0
                if k_tile_idx % loops_k + cur_k_tile_num > loops_k:
                    sk_slot_count = 2
                    next_streamk_block_idx = (k_tile_idx + cur_k_tile_num) // loops_k
                    streamk_linear = normal_blocks + next_streamk_block_idx
                    streamk_tb_idx = streamk_linear // swizzle_span
                    streamk_in_tile = streamk_linear % swizzle_span
                    streamk_n_row = _swizzle.SWIZZLE_OFFSET
                    if streamk_tb_idx == (swizzle_tb_loop - 1):
                        streamk_n_row = (
                            loops_m - _swizzle.SWIZZLE_OFFSET * streamk_tb_idx
                        )
                    streamk_block_row = (
                        streamk_tb_idx * _swizzle.SWIZZLE_OFFSET
                        + streamk_in_tile % streamk_n_row
                    )
                    streamk_block_col = streamk_in_tile // streamk_n_row
                    if (streamk_tb_idx % 2) == 1:
                        streamk_block_col = loops_n - streamk_block_col - 1
                    streamk_block_k = c0
                    streamk_actual_k = (
                        (k_tile_idx + cur_k_tile_num) % loops_k
                    ) * _tiling.l1_tk

            # One mmad pipeline for: normal (count=1), SK non-cross (count=1),
            # and SK cross (count=2: blockCoord then streamkBlockCoord).
            sk_slots = tla.range(c0, sk_slot_count, c1)
            for sk_slot in sk_slots:
                sk_slot_row = block_row if sk_slot == 0 else streamk_block_row
                sk_slot_col = block_col if sk_slot == 0 else streamk_block_col
                sk_slot_block_k = block_k if sk_slot == 0 else streamk_block_k
                sk_slot_actual_k = block_actual_k if sk_slot == 0 else streamk_actual_k
                sk_slot_ws_row = sk_task_id * 2 + sk_slot

                gm_a_by_core = tla.tile_view(
                    gm_a,
                    tla.make_shape(_tiling.l1_tm, k),
                    tla.make_coord(sk_slot_row, c0),
                )
                gm_b_by_core = tla.tile_view(
                    gm_b,
                    tla.make_shape(k, _tiling.l1_tn),
                    tla.make_coord(c0, sk_slot_col),
                )
                gm_c_by_core = tla.tile_view(
                    gm_c,
                    tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                    tla.make_coord(sk_slot_row, sk_slot_col),
                )
                gm_ws_by_core = tla.tile_view(
                    gm_workspace,
                    tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                    tla.make_coord(sk_slot_ws_row, c0),
                )
                l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

                k_l1_count = (sk_slot_actual_k + _tiling.l1_tk - 1) // _tiling.l1_tk
                k_l1_range = tla.range(c0, k_l1_count, c1)

                for k_l1_i in k_l1_range:
                    k_l1 = sk_slot_block_k + k_l1_i
                    gm_a_l1 = tla.tile_view(
                        gm_a_by_core,
                        tla.make_shape(_tiling.l1_tm, _tiling.l1_tk),
                        tla.make_coord(c0, k_l1),
                    )
                    gm_b_l1 = tla.tile_view(
                        gm_b_by_core,
                        tla.make_shape(_tiling.l1_tk, _tiling.l1_tn),
                        tla.make_coord(k_l1, c0),
                    )

                    l1_a = tla.make_tensor_like(
                        l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_l1
                    )
                    l1_b = tla.make_tensor_like(
                        l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_l1
                    )
                    if l1_buf_idx == c0:
                        tla.wait_flag(l1a0_available)
                    else:
                        tla.wait_flag(l1a1_available)
                    tla.copy(l1_a, gm_a_l1)
                    if l1_buf_idx == c0:
                        tla.set_flag(l1a0_data_ready)
                    else:
                        tla.set_flag(l1a1_data_ready)

                    if l1_buf_idx == c0:
                        tla.wait_flag(l1b0_available)
                    else:
                        tla.wait_flag(l1b1_available)
                    tla.copy(l1_b, gm_b_l1)
                    if l1_buf_idx == c0:
                        tla.set_flag(l1b0_data_ready)
                    else:
                        tla.set_flag(l1b1_data_ready)

                    k_l0_count = (
                        l1_a.origin_shape[1] + _tiling.l0_tk - 1
                    ) // _tiling.l0_tk
                    k_l0_range = tla.range(c0, k_l0_count, c1)

                    for k_l0 in k_l0_range:
                        l1_a_l0 = tla.tile_view(
                            l1_a,
                            tla.make_shape(_tiling.l0_tm, _tiling.l0_tk),
                            tla.make_coord(c0, k_l0),
                        )
                        l1_b_l0 = tla.tile_view(
                            l1_b,
                            tla.make_shape(_tiling.l0_tk, _tiling.l0_tn),
                            tla.make_coord(k_l0, c0),
                        )

                        l0_a = tla.make_tensor_like(
                            l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_l0
                        )
                        l0_b = tla.make_tensor_like(
                            l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_l0
                        )
                        if k_l0 == 0:
                            if l1_buf_idx == c0:
                                tla.wait_flag(l1a0_data_ready)
                            else:
                                tla.wait_flag(l1a1_data_ready)

                        if l0_buf_idx == c0:
                            tla.wait_flag(l0a0_available)
                        else:
                            tla.wait_flag(l0a1_available)
                        tla.copy(l0_a, l1_a_l0)
                        if k_l0 == k_l0_count - 1:
                            if l1_buf_idx == c0:
                                tla.set_flag(l1a0_available)
                            else:
                                tla.set_flag(l1a1_available)

                        if k_l0 == 0:
                            if l1_buf_idx == c0:
                                tla.wait_flag(l1b0_data_ready)
                            else:
                                tla.wait_flag(l1b1_data_ready)
                        if l0_buf_idx == c0:
                            tla.wait_flag(l0b0_available)
                        else:
                            tla.wait_flag(l0b1_available)
                        tla.copy(l0_b, l1_b_l0)
                        if k_l0 == k_l0_count - 1:
                            if l1_buf_idx == c0:
                                tla.set_flag(l1b0_available)
                            else:
                                tla.set_flag(l1b1_available)

                        tla.set_flag(l0_ab_data_ready)
                        tla.wait_flag(l0_ab_data_ready)

                        unit_flag = (
                            0b11
                            if (k_l1_i == k_l1_count - 1) and (k_l0 == k_l0_count - 1)
                            else 0b10
                        )
                        init_c = True if k_l1_i == 0 and k_l0 == 0 else False
                        tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
                        if l0_buf_idx == c0:
                            tla.set_flag(l0a0_available)
                            tla.set_flag(l0b0_available)
                        else:
                            tla.set_flag(l0a1_available)
                            tla.set_flag(l0b1_available)
                        l0_buf_idx = c1 - l0_buf_idx
                    l1_buf_idx = c1 - l1_buf_idx

                if is_sk_now:
                    tla.copy(
                        gm_ws_by_core,
                        l0_c,
                        tla.params.CopyL0C2DstParams(unit_flag=0b11),
                    )
                else:
                    tla.copy(
                        gm_c_by_core,
                        l0_c,
                        tla.params.CopyL0C2DstParams(unit_flag=0b11),
                    )

            # Match C++: CrossCoreSetFlag on PIPE_FIX after SK (swap slot
            # or every SK when normal_blocks==0). No PipeBarrier before the
            # flag so AIV can overlap the deferred last normal.
            if is_sk_now:
                if normal_blocks > 0:
                    if block_idx < streamk_cores:
                        if loop_idx == normal_blocks - tla.arch.block_num() + block_idx:
                            tla.cross_core_set_flag(aic_finish, tla.arch.FIX)
                if normal_blocks == 0:
                    tla.cross_core_set_flag(aic_finish, tla.arch.FIX)

        tla.wait_flag(l1a0_available)
        tla.wait_flag(l1a1_available)
        tla.wait_flag(l1b0_available)
        tla.wait_flag(l1b1_available)
        tla.wait_flag(l0a0_available)
        tla.wait_flag(l0a1_available)
        tla.wait_flag(l0b0_available)
        tla.wait_flag(l0b1_available)
        tla.wait_flag(l0c_available)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        # Wait for the paired AIC, then for every AIC through the all-AIV barrier.
        tla.cross_core_wait_flag(aic_finish, tla.arch.MTE2)
        tla.cross_core_set_flag(aiv_ibarrier, tla.arch.MTE2)
        tla.cross_core_wait_flag(aiv_ibarrier, tla.arch.MTE2)
        # In a mix kernel block_idx() is the AIC id and sub_block_idx() the AIV half.
        aiv_id = tla.arch.block_idx()
        aiv_sub = tla.arch.sub_block_idx()
        aiv_global = aiv_id * AIV_SUB_BLOCK_NUM + aiv_sub

        # tla.range keeps the loop dynamic; a Python range would unroll it.
        for aiv_sk_id in tla.range(c0, streamk_blocks, c1):
            # GetCoreIdx / IsCross (block_swizzle.hpp) — same as C++ AIV reduce.
            # Fold kTileNumPerCore==0 at Python level to avoid %/÷ 0 in IR.
            aiv_start_core = c0
            aiv_end_core = c0
            aiv_head_cross = False
            aiv_tail_cross = False
            if k_tile_num_per_core == 0:
                aiv_start_core = aiv_sk_id * loops_k
                aiv_end_core = (aiv_sk_id + 1) * loops_k
                aiv_head_cross = False
                aiv_tail_cross = False
            else:
                aiv_threshold = k_tile_remain * (k_tile_num_per_core + 1)
                aiv_start_core = aiv_sk_id * loops_k // (k_tile_num_per_core + 1)
                if aiv_sk_id * loops_k > aiv_threshold:
                    aiv_start_core = (
                        k_tile_remain
                        + (aiv_sk_id * loops_k - aiv_threshold) // k_tile_num_per_core
                    )
                aiv_end_core = (aiv_sk_id + 1) * loops_k // (k_tile_num_per_core + 1)
                if (aiv_sk_id + 1) * loops_k > aiv_threshold:
                    aiv_end_core = (
                        k_tile_remain
                        + ((aiv_sk_id + 1) * loops_k - aiv_threshold)
                        // k_tile_num_per_core
                    )
                aiv_head_cross = (aiv_sk_id * loops_k) % (k_tile_num_per_core + 1) != 0
                if aiv_sk_id * loops_k > aiv_threshold:
                    aiv_head_numer = aiv_sk_id * loops_k - aiv_threshold
                    aiv_head_cross = aiv_head_numer % k_tile_num_per_core != 0
                aiv_tail_cross = ((aiv_sk_id + 1) * loops_k) % (
                    k_tile_num_per_core + 1
                ) != 0
                if (aiv_sk_id + 1) * loops_k > aiv_threshold:
                    aiv_tail_numer = (aiv_sk_id + 1) * loops_k - aiv_threshold
                    aiv_tail_cross = aiv_tail_numer % k_tile_num_per_core != 0
            # Workers are the cores that own a K slice of this tile. A trailing
            # cross core contributes one more slice to reduce but no worker.
            aiv_end_core_raw = aiv_end_core
            aiv_labor = (aiv_end_core_raw - aiv_start_core) * AIV_SUB_BLOCK_NUM
            if aiv_tail_cross:
                aiv_end_core = aiv_end_core + 1

            aiv_linear = normal_blocks + aiv_sk_id
            aiv_span = _swizzle.SWIZZLE_OFFSET * loops_n
            aiv_tb_loop = (
                loops_m + _swizzle.SWIZZLE_OFFSET - 1
            ) // _swizzle.SWIZZLE_OFFSET
            aiv_tb_idx = aiv_linear // aiv_span
            aiv_in_tile = aiv_linear % aiv_span
            aiv_n_row = _swizzle.SWIZZLE_OFFSET
            if aiv_tb_idx == (aiv_tb_loop - 1):
                aiv_n_row = loops_m - _swizzle.SWIZZLE_OFFSET * aiv_tb_idx
            aiv_block_row = (
                aiv_tb_idx * _swizzle.SWIZZLE_OFFSET + aiv_in_tile % aiv_n_row
            )
            aiv_block_col = aiv_in_tile // aiv_n_row
            if (aiv_tb_idx % 2) == 1:
                aiv_block_col = loops_n - aiv_block_col - 1

            aiv_tile_m = _tiling.l1_tm
            if aiv_block_row == (loops_m - 1):
                aiv_tile_m = m - aiv_block_row * _tiling.l1_tm

            aiv_slice_count = aiv_end_core - aiv_start_core
            aiv_m_loops = (aiv_tile_m + AIV_TILE_M - 1) // AIV_TILE_M
            aiv_rows_per_slot = _tiling.l1_tm // AIV_TILE_M

            # Only AIVs paired with the producing AICs reduce this tile; they
            # split its row chunks between them.
            if aiv_id >= aiv_start_core:
                if aiv_id < aiv_end_core_raw:
                    aiv_loop_start = aiv_global - aiv_start_core * AIV_SUB_BLOCK_NUM
                    # tla.range needs a constexpr step, so hand each worker a
                    # contiguous [lo, hi) instead of striding by worker count.
                    aiv_chunk_per = (aiv_m_loops + aiv_labor - 1) // aiv_labor
                    aiv_chunk_lo = aiv_loop_start * aiv_chunk_per
                    aiv_chunk_hi = aiv_chunk_lo + aiv_chunk_per
                    if aiv_chunk_lo > aiv_m_loops:
                        aiv_chunk_lo = aiv_m_loops
                    if aiv_chunk_hi > aiv_m_loops:
                        aiv_chunk_hi = aiv_m_loops
                    for aiv_m_idx in tla.range(aiv_chunk_lo, aiv_chunk_hi, c1):
                        aiv_c_row = aiv_block_row * aiv_rows_per_slot + aiv_m_idx
                        aiv_c_col = aiv_block_col
                        aiv_gm_c_tile = tla.tile_view(
                            gm_c,
                            tla.make_shape(AIV_TILE_M, _tiling.l1_tn),
                            tla.make_coord(aiv_c_row, aiv_c_col),
                        )

                        aiv_init_pingpong = 0
                        if aiv_head_cross:
                            aiv_init_pingpong = 1
                        aiv_init_row = (
                            aiv_start_core * 2 + aiv_init_pingpong
                        ) * aiv_rows_per_slot + aiv_m_idx
                        aiv_ws_init = tla.tile_view(
                            gm_workspace,
                            tla.make_shape(AIV_TILE_M, _tiling.l1_tn),
                            tla.make_coord(aiv_init_row, c0),
                        )
                        tla.copy(aiv_acc_ub, aiv_ws_init)
                        tla.set_flag(aiv_loaded)
                        tla.wait_flag(aiv_loaded)

                        aiv_slice_range = tla.range(c1, aiv_slice_count, c1)
                        for aiv_slice_idx in aiv_slice_range:
                            aiv_core = aiv_start_core + aiv_slice_idx
                            aiv_ws_row = (aiv_core * 2) * aiv_rows_per_slot + aiv_m_idx
                            aiv_ws_tile = tla.tile_view(
                                gm_workspace,
                                tla.make_shape(AIV_TILE_M, _tiling.l1_tn),
                                tla.make_coord(aiv_ws_row, c0),
                            )
                            tla.copy(aiv_temp_ub, aiv_ws_tile)
                            tla.set_flag(aiv_loaded)
                            tla.wait_flag(aiv_loaded)
                            with tla.vec.func(mode="simd"):
                                for _aiv_rm in tla.range(0, aiv_m_chunks, 1):
                                    for _aiv_rn in tla.range(0, aiv_n_chunks, 1):
                                        aiv_acc_chunk = tla.tile_view(
                                            aiv_acc_ub,
                                            tla.make_shape(AIV_REG_M, AIV_REG_N),
                                            tla.make_coord(_aiv_rm, _aiv_rn),
                                        )
                                        aiv_temp_chunk = tla.tile_view(
                                            aiv_temp_ub,
                                            tla.make_shape(AIV_REG_M, AIV_REG_N),
                                            tla.make_coord(_aiv_rm, _aiv_rn),
                                        )
                                        aiv_acc_chunk.store(
                                            tla.add(
                                                aiv_acc_chunk.load(),
                                                aiv_temp_chunk.load(),
                                                mask=tla.create_mask(
                                                    pattern=tla.mask.ALL, dtype=DTYPE_C
                                                ),
                                            ),
                                            mask=tla.create_mask(
                                                pattern=tla.mask.ALL, dtype=DTYPE_C
                                            ),
                                        )
                            tla.set_flag(aiv_vec_to_mte2)
                            tla.wait_flag(aiv_vec_to_mte2)

                        aiv_store_ub = aiv_acc_ub
                        if tla.const_expr(dtype_gm_c != DTYPE_C):
                            # multi_core_splitk pattern: 1D VL chunks, even-cast
                            # with f32 mask, then DIST_PACK_B32 densify.
                            if tla.const_expr(dtype_gm_c == tla.Float16):
                                aiv_cast_even = tla.params.CastParams(
                                    reg_slot=tla.params.RegSlot.ZERO,
                                    sat_mode=tla.params.SatMode.NOSAT,
                                    round_mode=tla.params.RoundMode.CAST_FLOOR,
                                )
                            else:
                                aiv_cast_even = tla.params.CastParams(
                                    reg_slot=tla.params.RegSlot.ZERO,
                                    sat_mode=tla.params.SatMode.NOSAT,
                                    round_mode=tla.params.RoundMode.CAST_ROUND,
                                )
                            aiv_pack_store = tla.params.NormalStoreParams(
                                store_dist=tla.params.StoreDist.DIST_PACK_B32
                            )
                            aiv_cast_vl_loops = aiv_m_chunks * aiv_n_chunks
                            aiv_acc_1d = tla.make_tensor(
                                aiv_acc_ptr,
                                tla.make_layout(
                                    tla.make_shape(aiv_ub_tile_elems),
                                    tla.make_stride(c1),
                                ),
                            )
                            aiv_out_1d = tla.make_tensor(
                                aiv_out_ptr,
                                tla.make_layout(
                                    tla.make_shape(aiv_ub_tile_elems),
                                    tla.make_stride(c1),
                                ),
                            )
                            with tla.vec.func(mode="simd"):
                                aiv_cast_mask = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=DTYPE_C
                                )
                                aiv_store_mask = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=dtype_gm_c
                                )
                                for aiv_cast_vl in tla.range(0, aiv_cast_vl_loops, 1):
                                    aiv_cast_src = tla.tile_view(
                                        aiv_acc_1d,
                                        tla.make_shape(AIV_REG_N),
                                        tla.make_coord(aiv_cast_vl),
                                    )
                                    aiv_cast_dst = tla.tile_view(
                                        aiv_out_1d,
                                        tla.make_shape(AIV_REG_N),
                                        tla.make_coord(aiv_cast_vl),
                                    )
                                    aiv_cast_h = aiv_cast_src.load().to(
                                        dtype_gm_c,
                                        aiv_cast_even,
                                        aiv_cast_mask,
                                    )
                                    aiv_cast_dst.store(
                                        aiv_cast_h,
                                        aiv_pack_store,
                                        mask=aiv_store_mask,
                                    )
                            aiv_store_ub = aiv_out_ub

                        tla.set_flag(aiv_done)
                        tla.wait_flag(aiv_done)
                        tla.copy(aiv_gm_c_tile, aiv_store_ub)
                        tla.pipe_barrier(tla.pipes.ALL)

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

ElemDType = Literal["f16", "bf16", "f32"]

# Relative tolerance for result compare: tighter when K is below this threshold.
_COMPARE_RTOL_K_THRESHOLD = 2048
_COMPARE_RTOL_NUMERATOR = 1.0
_COMPARE_RTOL_DENOM_SMALL_K = 256
_COMPARE_RTOL_DENOM_LARGE_K = 128
# Extra pass gate: allow a small fraction of elements outside atol/rtol.
_COMPARE_MISMATCH_RATIO_NARROW = 0.001  # f16 / bf16: <= 0.1%
_COMPARE_MISMATCH_RATIO_F32 = 0.0001  # f32: <= 0.01%


def _comparison_atol(dtype_c: ElemDType, args: argparse.Namespace) -> float:
    if dtype_c in ("f16", "bf16"):
        return max(float(args.atol), 5e-3)
    return float(args.atol)


def _comparison_rtol(k_val: int) -> float:
    """Pick relative tolerance from K: ``1/256`` if ``k < 2048``, else ``1/128``."""
    if k_val < _COMPARE_RTOL_K_THRESHOLD:
        return _COMPARE_RTOL_NUMERATOR / _COMPARE_RTOL_DENOM_SMALL_K
    return _COMPARE_RTOL_NUMERATOR / _COMPARE_RTOL_DENOM_LARGE_K


def _mismatch_ratio_budget(dtype_c: ElemDType) -> float:
    """Max fraction of out-of-tolerance elements still counted as pass."""
    if dtype_c in ("f16", "bf16"):
        return _COMPARE_MISMATCH_RATIO_NARROW
    return _COMPARE_MISMATCH_RATIO_F32


def _compare_expected_torch(
    actual: Any, expected: Any, *, rtol: float, atol: float, dtype_c: ElemDType
) -> dict[str, Any]:
    """Compare against golden with atol/rtol plus a small mismatch-ratio budget."""
    close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    total = int(actual.numel())
    mismatch_count = int((~close).sum().item())
    mismatch_ratio = (mismatch_count / total) if total else 0.0
    budget = _mismatch_ratio_budget(dtype_c)
    all_close = mismatch_count == 0
    within_budget = mismatch_ratio <= budget
    ok = all_close or within_budget
    first_mismatch: dict[str, Any] | None = None
    if mismatch_count > 0:
        row, col = (
            int(value) for value in close.logical_not().nonzero(as_tuple=False)[0]
        )
        first_mismatch = {
            "index": [row, col],
            "actual": float(actual[row, col].item()),
            "expected": float(expected[row, col].item()),
        }
    return {
        "ok": ok,
        "all_close": all_close,
        "within_budget": within_budget,
        "mismatch_count": mismatch_count,
        "mismatch_ratio": mismatch_ratio,
        "mismatch_budget": budget,
        "total": total,
        "first_mismatch": first_mismatch,
    }


def run(args: argparse.Namespace) -> int:
    from helpers import (
        get_block_num,
        create_tla_tensor,
    )

    torch.npu.set_device(args.device)
    print(
        f"--- mnk=({args.m},{args.n},{args.k}) "
        f"layout={args.layout_a}/{args.layout_b} "
        f"dtype={args.dtype_a}/{args.dtype_b}/{args.dtype_c} ---"
    )
    torch.manual_seed(0)
    dtypes = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
    dtype_a = dtypes[args.dtype_a]
    dtype_b = dtypes[args.dtype_b]
    dtype_c = dtypes[args.dtype_c]

    block_num = get_block_num(args.block_num, args.device, kind="cube")

    tiling = TilingParams()
    pad_m = (args.m + tiling.l1_tm - 1) // tiling.l1_tm * tiling.l1_tm
    pad_n = (args.n + tiling.l1_tn - 1) // tiling.l1_tn * tiling.l1_tn

    a = torch.zeros(pad_m, args.k, dtype=dtype_a, device="cpu")
    b = torch.zeros(args.k, pad_n, dtype=dtype_b, device="cpu")
    a[: args.m, :] = (
        torch.rand(args.m, args.k, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    ).to(dtype_a)
    b[:, : args.n] = (
        torch.rand(args.k, args.n, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    ).to(dtype_b)
    c = torch.rand(pad_m, pad_n, dtype=dtype_c, device="cpu") * 10.0 - 5.0
    ref = a[: args.m, :].float() @ b[:, : args.n].float()
    if dtype_c in (torch.float16, torch.bfloat16):
        ref = ref.to(dtype_c).float()

    ws_rows = tiling.l1_tm * 2 * block_num
    w = torch.zeros(ws_rows, tiling.l1_tn, dtype=torch.float32, device="cpu")

    a = (
        a.contiguous() if args.layout_a == "row" else a.permute(1, 0).contiguous()
    ).npu()
    b = (
        b.contiguous() if args.layout_b == "row" else b.permute(1, 0).contiguous()
    ).npu()
    c = c.contiguous().npu()
    w = w.contiguous().npu()
    a_tensor = create_tla_tensor(a, args.layout_a)
    b_tensor = create_tla_tensor(b, args.layout_b)
    c_tensor = create_tla_tensor(c, "row")
    w_tensor = create_tla_tensor(w, "row")

    artifact = tla.compile(
        streamk_mmad_kernel,
        a_tensor,
        b_tensor,
        c_tensor,
        w_tensor,
        tiling,
        SwizzleParams(),
        block_num,
        options="--npu-arch 3510",
    )
    artifact(a_tensor, b_tensor, c_tensor, w_tensor, block_num=block_num)
    torch.npu.synchronize()

    atol = _comparison_atol(args.dtype_c, args)
    rtol = _comparison_rtol(args.k)
    cmp = _compare_expected_torch(
        c[: args.m, : args.n].detach().cpu().float(),
        ref,
        rtol=rtol,
        atol=atol,
        dtype_c=args.dtype_c,
    )
    passed = cmp["ok"]
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--layout-a", choices=("row", "col"), default="row")
    parser.add_argument("--layout-b", choices=("row", "col"), default="row")
    parser.add_argument("--dtype-a", choices=("f16", "bf16", "f32"), default="f16")
    parser.add_argument("--dtype-b", choices=("f16", "bf16", "f32"), default="f16")
    parser.add_argument("--dtype-c", choices=("f16", "bf16", "f32"), default="f32")
    parser.add_argument("--block-num", type=int, default=-1)
    parser.add_argument(
        "--atol", type=float, default=1e-3, help="Comparison tolerance."
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
