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

import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import argparse

import catlass.tla as tla
import torch
import torch_npu  # noqa: F401

from catlass.types import dtype_size_bytes

from helpers import TilingParams, SwizzleParams, compute_ub_slot_elems

# ---- kernel constants + @tla.kernel ----
UB_SIZE = tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub)
BYTE_PER_C0 = 32
VECTOR_ELE = 256

# One SIMD register = 256B; lane count = 256 / sizeof(ElementC). REG_M=1.
REG_M = 1

# After this many MN tiles per core, AIC/AIV exchange the reverse cross-core flag.
REVERSE_DEPTH = 15

ENABLE_UNIT_FLAG = True

# UB nodes counted for sizing: load Acc and compute (store reuses Out).
EVG_UB_NODES = 2
# EVG UB multi-buffer depth (1 or 2 physical slots). Manual edit only, like _tiling.l1_tn.
EVG_UB_STAGES = 2

# LeakyReLU negative slope.
LEAKY_RELU_ALPHA = 0.1


@tla.kernel
def matmul_evg_leaky_relu_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_d: tla.Tensor,
    gm_workspace: tla.Tensor,
    _tiling: tla.Constexpr[TilingParams],
    _swizzle: tla.Constexpr[SwizzleParams],
    UB_SLOT_ELEMS: tla.Constexpr[int],
    SIMD_LANES: tla.Constexpr[int],
) -> None:
    """Cube GEMM to GM workspace; vector epilogue D = leaky_relu(Acc, alpha)."""
    c0 = 0
    c1 = 1

    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype
    dtype_gm_c = gm_d.ptr.dtype
    DTYPE_C = tla.Float32  # L0C accumulator only

    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    k = gm_a.origin_shape[1]

    # ---- soft-flags (cube pingpong) ----
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
    l0c_data_ready = tla.flag("l0c_data_ready", tla.arch.CUBE, tla.arch.FIX)
    l0c_available = tla.flag("l0c_available", tla.arch.FIX, tla.arch.CUBE)

    # ---- EVG/UB soft-flags ----
    ub_acc0_data_ready = tla.flag("ub_acc0_data_ready", tla.arch.MTE2, tla.arch.VECTOR)
    ub_acc1_data_ready = tla.flag("ub_acc1_data_ready", tla.arch.MTE2, tla.arch.VECTOR)
    ub_out0_data_ready = tla.flag("ub_out0_data_ready", tla.arch.VECTOR, tla.arch.MTE3)
    ub_out1_data_ready = tla.flag("ub_out1_data_ready", tla.arch.VECTOR, tla.arch.MTE3)
    ub_acc0_available = tla.flag("ub_acc0_available", tla.arch.VECTOR, tla.arch.MTE2)
    ub_acc1_available = tla.flag("ub_acc1_available", tla.arch.VECTOR, tla.arch.MTE2)
    ub_out0_available = tla.flag("ub_out0_available", tla.arch.MTE3, tla.arch.VECTOR)
    ub_out1_available = tla.flag("ub_out1_available", tla.arch.MTE3, tla.arch.VECTOR)

    # ---- cross-core flags ----
    # AIC FixPipe → set forward; AIV MTE2 → wait forward; every REVERSE_DEPTH tiles reverse.
    # No per-tile aiv_finish, so FixPipe(N+1) can overlap epilogue(N).
    aic_finish = tla.cross_flag("aic_finish", mode=2)
    aic_finish_rv = tla.cross_flag("aic_finish_rv", mode=2)

    # ---- L1/L0/L0C allocates ----
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

    # ---- UB allocates ----
    ub_acc_ptr0 = tla.allocate(UB_SLOT_ELEMS, dtype_gm_c, tla.AddressSpace.ub, 256)
    ub_out_ptr0 = tla.allocate(UB_SLOT_ELEMS, dtype_gm_c, tla.AddressSpace.ub, 256)
    ub_acc_ptr1 = ub_acc_ptr0
    ub_out_ptr1 = ub_out_ptr0
    if tla.const_expr(EVG_UB_STAGES >= 2):
        ub_acc_ptr1 = tla.allocate(UB_SLOT_ELEMS, dtype_gm_c, tla.AddressSpace.ub, 256)
        ub_out_ptr1 = tla.allocate(UB_SLOT_ELEMS, dtype_gm_c, tla.AddressSpace.ub, 256)

    # ---- grid / swizzle setup ----
    grid_m = (m + _tiling.l1_tm - 1) // _tiling.l1_tm
    grid_n = (n + _tiling.l1_tn - 1) // _tiling.l1_tn
    total_blocks = grid_m * grid_n
    # Folded at compile time via _swizzle.SWIZZLE_DIRECTION (host-set).
    if tla.const_expr(_swizzle.SWIZZLE_DIRECTION == 0):
        swizzle_tile_count = (
            grid_m + _swizzle.SWIZZLE_OFFSET - 1
        ) // _swizzle.SWIZZLE_OFFSET
    else:
        swizzle_tile_count = (
            grid_n + _swizzle.SWIZZLE_OFFSET - 1
        ) // _swizzle.SWIZZLE_OFFSET

    # ---- cube: prime flags; MN tile loop; K pingpong; FixPipe; handoff; drain ----
    with tla.cube():
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

        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )
        for block_linear in block_range:
            # Map linear MN task id → (block_row, block_col) with swizzle (must stay in kernel AST).
            block_row = c0
            block_col = c0
            if tla.const_expr(_swizzle.SWIZZLE_DIRECTION == 0):
                swizzle_tile_idx = block_linear // (_swizzle.SWIZZLE_OFFSET * grid_n)
                in_tile_idx = block_linear % (_swizzle.SWIZZLE_OFFSET * grid_n)
                swizzle_n_rows = (
                    _swizzle.SWIZZLE_OFFSET
                    if swizzle_tile_idx != (swizzle_tile_count - 1)
                    else (grid_m - _swizzle.SWIZZLE_OFFSET * swizzle_tile_idx)
                )
                block_row = (
                    swizzle_tile_idx * _swizzle.SWIZZLE_OFFSET
                    + in_tile_idx % swizzle_n_rows
                )
                block_col = in_tile_idx // swizzle_n_rows
                block_col = (
                    (grid_n - block_col - 1)
                    if (swizzle_tile_idx % 2 == 1)
                    else block_col
                )
            else:
                swizzle_tile_idx = block_linear // (_swizzle.SWIZZLE_OFFSET * grid_m)
                in_tile_idx = block_linear % (_swizzle.SWIZZLE_OFFSET * grid_m)
                swizzle_n_cols = (
                    _swizzle.SWIZZLE_OFFSET
                    if swizzle_tile_idx != (swizzle_tile_count - 1)
                    else (grid_n - _swizzle.SWIZZLE_OFFSET * swizzle_tile_idx)
                )
                block_row = in_tile_idx // swizzle_n_cols
                block_col = (
                    swizzle_tile_idx * _swizzle.SWIZZLE_OFFSET
                    + in_tile_idx % swizzle_n_cols
                )
                block_row = (
                    (grid_m - block_row - 1)
                    if (swizzle_tile_idx % 2 == 1)
                    else block_row
                )
            gm_a_by_core = tla.tile_view(
                gm_a, tla.make_shape(_tiling.l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                gm_b, tla.make_shape(k, _tiling.l1_tn), tla.make_coord(c0, block_col)
            )
            gm_workspace_by_core = tla.tile_view(
                gm_workspace,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(block_row, block_col),
            )

            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + _tiling.l1_tk - 1) // _tiling.l1_tk
            k_l1_range = tla.range(c0, k_l1_count, c1)

            l0_c = tla.make_tensor_like(l0c_ptr, gm_workspace_by_core)

            if tla.const_expr(not ENABLE_UNIT_FLAG):
                tla.wait_flag(l0c_available)
            for k_l1 in k_l1_range:
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

                k_l0_count = (l1_a.origin_shape[1] + _tiling.l0_tk - 1) // _tiling.l0_tk
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

                    unit_flag = 0
                    if tla.const_expr(ENABLE_UNIT_FLAG):
                        if (k_l1 == k_l1_count - 1) and (k_l0 == k_l0_count - 1):
                            unit_flag = 0b11
                        else:
                            unit_flag = 0b10
                    init_c = True if k_l1 == 0 and k_l0 == 0 else False
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
                    if l0_buf_idx == c0:
                        tla.set_flag(l0a0_available)
                        tla.set_flag(l0b0_available)
                    else:
                        tla.set_flag(l0a1_available)
                        tla.set_flag(l0b1_available)
                    l0_buf_idx = c1 - l0_buf_idx
                l1_buf_idx = c1 - l1_buf_idx

            # FixPipe: write L0C tile to GM workspace (full tile, no SPLIT_M).
            if tla.const_expr(not ENABLE_UNIT_FLAG):
                tla.set_flag(l0c_data_ready)
                tla.wait_flag(l0c_data_ready)
                tla.copy(gm_workspace_by_core, l0_c)
                tla.set_flag(l0c_available)
            else:
                tla.copy(
                    gm_workspace_by_core,
                    l0_c,
                    tla.params.CopyL0C2DstParams(unit_flag=0b11),
                )

            # Notify AIV that this MN tile is ready in GM workspace; reverse every REVERSE_DEPTH tiles.
            tla.cross_core_set_flag(aic_finish, tla.arch.FIX)
            per_core_tile_idx = (
                block_linear - tla.arch.block_idx()
            ) // tla.arch.block_num()
            if (per_core_tile_idx + 1) % REVERSE_DEPTH == 0:
                tla.cross_core_wait_flag(aic_finish_rv, tla.arch.FIX)

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

    # ---- vector: prime flags; MN loop; load / compute / store epilogue ----
    with tla.vector():
        aiv_sub_idx = tla.arch.sub_block_idx()
        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )

        tla.set_flag(ub_acc0_available)
        tla.set_flag(ub_out0_available)
        if tla.const_expr(EVG_UB_STAGES >= 2):
            tla.set_flag(ub_acc1_available)
            tla.set_flag(ub_out1_available)

        for block_linear in block_range:
            # Map linear MN task id → (block_row, block_col) with swizzle (must stay in kernel AST).
            block_row = c0
            block_col = c0
            if tla.const_expr(_swizzle.SWIZZLE_DIRECTION == 0):
                swizzle_tile_idx = block_linear // (_swizzle.SWIZZLE_OFFSET * grid_n)
                in_tile_idx = block_linear % (_swizzle.SWIZZLE_OFFSET * grid_n)
                swizzle_n_rows = (
                    _swizzle.SWIZZLE_OFFSET
                    if swizzle_tile_idx != (swizzle_tile_count - 1)
                    else (grid_m - _swizzle.SWIZZLE_OFFSET * swizzle_tile_idx)
                )
                block_row = (
                    swizzle_tile_idx * _swizzle.SWIZZLE_OFFSET
                    + in_tile_idx % swizzle_n_rows
                )
                block_col = in_tile_idx // swizzle_n_rows
                block_col = (
                    (grid_n - block_col - 1)
                    if (swizzle_tile_idx % 2 == 1)
                    else block_col
                )
            else:
                swizzle_tile_idx = block_linear // (_swizzle.SWIZZLE_OFFSET * grid_m)
                in_tile_idx = block_linear % (_swizzle.SWIZZLE_OFFSET * grid_m)
                swizzle_n_cols = (
                    _swizzle.SWIZZLE_OFFSET
                    if swizzle_tile_idx != (swizzle_tile_count - 1)
                    else (grid_n - _swizzle.SWIZZLE_OFFSET * swizzle_tile_idx)
                )
                block_row = in_tile_idx // swizzle_n_cols
                block_col = (
                    swizzle_tile_idx * _swizzle.SWIZZLE_OFFSET
                    + in_tile_idx % swizzle_n_cols
                )
                block_row = (
                    (grid_m - block_row - 1)
                    if (swizzle_tile_idx % 2 == 1)
                    else block_row
                )

            # Wait until AIC has written this tile to GM workspace (MTE2).
            # Every REVERSE_DEPTH tiles, set the reverse cross-core flag.
            tla.cross_core_wait_flag(aic_finish, tla.arch.MTE2)
            per_core_tile_idx = (
                block_linear - tla.arch.block_idx()
            ) // tla.arch.block_num()
            if (per_core_tile_idx + 1) % REVERSE_DEPTH == 0:
                tla.cross_core_set_flag(aic_finish_rv, tla.arch.MTE2)

            gm_workspace_by_core = tla.tile_view(
                gm_workspace,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(block_row, block_col),
            )
            gm_d_by_core = tla.tile_view(
                gm_d,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(block_row, block_col),
            )

            # Split the MN tile on M across the two AIV sub-blocks.
            actual_m = gm_d_by_core.origin_shape[0]
            actual_n = gm_d_by_core.origin_shape[1]
            # SPLIT_M: request ceil(M/2) rows × full N; AIV1 may clip when M is odd.
            aiv_m_req = (actual_m + 1) // 2
            gm_acc = tla.tile_view(
                gm_workspace_by_core,
                tla.make_shape(aiv_m_req, actual_n),
                tla.make_coord(aiv_sub_idx, c0),
            )
            gm_result = tla.tile_view(
                gm_d_by_core,
                tla.make_shape(aiv_m_req, actual_n),
                tla.make_coord(aiv_sub_idx, c0),
            )
            aiv_m = gm_result.origin_shape[0]
            aiv_n = gm_result.origin_shape[1]

            ub_stage_idx = c0
            if aiv_n <= UB_SLOT_ELEMS:
                ub_row_stride = ((aiv_n + BYTE_PER_C0 - 1) // BYTE_PER_C0) * BYTE_PER_C0
                max_rows = UB_SLOT_ELEMS // ub_row_stride
                if max_rows < 1:
                    max_rows = 1
                n_epilogue_row_tiles = (aiv_m + max_rows - 1) // max_rows

                for tile_i in tla.range(c0, n_epilogue_row_tiles, c1):
                    gm_c_tile = tla.tile_view(
                        gm_acc,
                        tla.make_shape(max_rows, aiv_n),
                        tla.make_coord(tile_i, c0),
                    )
                    gm_d_tile = tla.tile_view(
                        gm_result,
                        tla.make_shape(max_rows, aiv_n),
                        tla.make_coord(tile_i, c0),
                    )
                    tile_rows = gm_c_tile.origin_shape[0]
                    tile_cols = gm_c_tile.origin_shape[1]
                    ub_layout = tla.make_layout(
                        tla.make_shape(tile_rows, tile_cols),
                        tla.make_stride(ub_row_stride, 1),
                        layoutTag=tla.arch.RowMajor,
                    )
                    n_simd_strips = (tile_cols + SIMD_LANES - 1) // SIMD_LANES
                    if tla.const_expr(EVG_UB_STAGES >= 2):
                        use_ub_stage1 = ub_stage_idx != c0
                    else:
                        use_ub_stage1 = False
                    ub_acc = tla.make_tensor(
                        ub_acc_ptr1 if use_ub_stage1 else ub_acc_ptr0, ub_layout
                    )
                    ub_result = tla.make_tensor(
                        ub_out_ptr1 if use_ub_stage1 else ub_out_ptr0, ub_layout
                    )

                    # Wait UB slot free → load accumulator → signal compute may start.
                    if use_ub_stage1:
                        tla.wait_flag(ub_acc1_available)
                    else:
                        tla.wait_flag(ub_acc0_available)
                    tla.copy(ub_acc, gm_c_tile)
                    if use_ub_stage1:
                        tla.set_flag(ub_acc1_data_ready)
                        tla.wait_flag(ub_acc1_data_ready)
                        tla.wait_flag(ub_out1_available)
                    else:
                        tla.set_flag(ub_acc0_data_ready)
                        tla.wait_flag(ub_acc0_data_ready)
                        tla.wait_flag(ub_out0_available)

                    with tla.vec.func(mode="simd"):
                        for row_i in tla.range(c0, tile_rows, c1):
                            remaining = tile_cols
                            for col_j in tla.range(c0, n_simd_strips, c1):
                                c_chunk = tla.tile_view(
                                    ub_acc,
                                    tla.make_shape(REG_M, SIMD_LANES),
                                    tla.make_coord(row_i, col_j),
                                )
                                result_chunk = tla.tile_view(
                                    ub_result,
                                    tla.make_shape(REG_M, SIMD_LANES),
                                    tla.make_coord(row_i, col_j),
                                )
                                tail, remaining = tla.update_mask(
                                    remaining, dtype=dtype_gm_c
                                )
                                # LeakyRelu(x, a) = x if x>=0 else a*x
                                # = max(x,0) + a*min(x,0)
                                xv = c_chunk.load()
                                pos = tla.max(xv, 0.0, mask=tail)
                                neg = tla.min(xv, 0.0, mask=tail)
                                result_chunk.store(
                                    tla.add(
                                        pos,
                                        tla.mul(neg, LEAKY_RELU_ALPHA, mask=tail),
                                        mask=tail,
                                    ),
                                    mask=tail,
                                )

                    if use_ub_stage1:
                        tla.set_flag(ub_acc1_available)
                        tla.set_flag(ub_out1_data_ready)
                        tla.wait_flag(ub_out1_data_ready)
                    else:
                        tla.set_flag(ub_acc0_available)
                        tla.set_flag(ub_out0_data_ready)
                        tla.wait_flag(ub_out0_data_ready)
                    tla.copy(gm_d_tile, ub_result)
                    if use_ub_stage1:
                        tla.set_flag(ub_out1_available)
                    else:
                        tla.set_flag(ub_out0_available)

                    if tla.const_expr(EVG_UB_STAGES >= 2):
                        ub_stage_idx = c1 - ub_stage_idx
            else:
                n_col_tiles = (aiv_n + UB_SLOT_ELEMS - 1) // UB_SLOT_ELEMS
                for row_i in tla.range(c0, aiv_m, c1):
                    for col_tile in tla.range(c0, n_col_tiles, c1):
                        gm_c_tile = tla.tile_view(
                            gm_acc,
                            tla.make_shape(c1, UB_SLOT_ELEMS),
                            tla.make_coord(row_i, col_tile),
                        )
                        gm_d_tile = tla.tile_view(
                            gm_result,
                            tla.make_shape(c1, UB_SLOT_ELEMS),
                            tla.make_coord(row_i, col_tile),
                        )
                        tile_rows = gm_c_tile.origin_shape[0]
                        tile_cols = gm_c_tile.origin_shape[1]
                        ub_row_stride = (
                            (tile_cols + BYTE_PER_C0 - 1) // BYTE_PER_C0
                        ) * BYTE_PER_C0
                        ub_layout = tla.make_layout(
                            tla.make_shape(tile_rows, tile_cols),
                            tla.make_stride(ub_row_stride, 1),
                            layoutTag=tla.arch.RowMajor,
                        )
                        n_simd_strips = (tile_cols + SIMD_LANES - 1) // SIMD_LANES
                        if tla.const_expr(EVG_UB_STAGES >= 2):
                            use_ub_stage1 = ub_stage_idx != c0
                        else:
                            use_ub_stage1 = False
                        ub_acc = tla.make_tensor(
                            ub_acc_ptr1 if use_ub_stage1 else ub_acc_ptr0, ub_layout
                        )
                        ub_result = tla.make_tensor(
                            ub_out_ptr1 if use_ub_stage1 else ub_out_ptr0, ub_layout
                        )

                        if use_ub_stage1:
                            tla.wait_flag(ub_acc1_available)
                        else:
                            tla.wait_flag(ub_acc0_available)
                        tla.copy(ub_acc, gm_c_tile)
                        if use_ub_stage1:
                            tla.set_flag(ub_acc1_data_ready)
                            tla.wait_flag(ub_acc1_data_ready)
                            tla.wait_flag(ub_out1_available)
                        else:
                            tla.set_flag(ub_acc0_data_ready)
                            tla.wait_flag(ub_acc0_data_ready)
                            tla.wait_flag(ub_out0_available)

                        with tla.vec.func(mode="simd"):
                            for r_i in tla.range(c0, tile_rows, c1):
                                remaining = tile_cols
                                for col_j in tla.range(c0, n_simd_strips, c1):
                                    c_chunk = tla.tile_view(
                                        ub_acc,
                                        tla.make_shape(REG_M, SIMD_LANES),
                                        tla.make_coord(r_i, col_j),
                                    )
                                    result_chunk = tla.tile_view(
                                        ub_result,
                                        tla.make_shape(REG_M, SIMD_LANES),
                                        tla.make_coord(r_i, col_j),
                                    )
                                    tail, remaining = tla.update_mask(
                                        remaining, dtype=dtype_gm_c
                                    )
                                    # LeakyRelu(x, a) = max(x,0) + a*min(x,0)
                                    xv = c_chunk.load()
                                    pos = tla.max(xv, 0.0, mask=tail)
                                    neg = tla.min(xv, 0.0, mask=tail)
                                    result_chunk.store(
                                        tla.add(
                                            pos,
                                            tla.mul(neg, LEAKY_RELU_ALPHA, mask=tail),
                                            mask=tail,
                                        ),
                                        mask=tail,
                                    )

                        if use_ub_stage1:
                            tla.set_flag(ub_acc1_available)
                            tla.set_flag(ub_out1_data_ready)
                            tla.wait_flag(ub_out1_data_ready)
                        else:
                            tla.set_flag(ub_acc0_available)
                            tla.set_flag(ub_out0_data_ready)
                            tla.wait_flag(ub_out0_data_ready)
                        tla.copy(gm_d_tile, ub_result)
                        if use_ub_stage1:
                            tla.set_flag(ub_out1_available)
                        else:
                            tla.set_flag(ub_out0_available)

                    if tla.const_expr(EVG_UB_STAGES >= 2):
                        ub_stage_idx = c1 - ub_stage_idx

        tla.wait_flag(ub_acc0_available)
        tla.wait_flag(ub_out0_available)
        if tla.const_expr(EVG_UB_STAGES >= 2):
            tla.wait_flag(ub_acc1_available)
            tla.wait_flag(ub_out1_available)

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = EXAMPLE_DIR / "artifacts" / "runtime-cache"
DESCRIPTION = (
    "Matmul EVG leaky_relu: D=leaky_relu(A@B) via L0C→GM workspace + AIV; dynamic GM."
)


def _compute_ub_slots(dtype_c: str) -> tuple[int, int]:
    """Compute UB_SLOT_ELEMS / SIMD_LANES according to the stages and element data type."""
    return (
        compute_ub_slot_elems(dtype_c, nodes=EVG_UB_NODES, stages=EVG_UB_STAGES),
        VECTOR_ELE // dtype_size_bytes(dtype_c),
    )


def run(args: argparse.Namespace) -> int:
    from helpers import (
        get_block_num,
        create_tla_tensor,
        compare,
    )

    torch.npu.set_device(args.device)
    print(
        f"--- mnk=({args.m},{args.n},{args.k}) "
        f"layout={args.layout_a}/{args.layout_b} "
        f"dtype={args.dtype_a}/{args.dtype_b}/{args.dtype_c} ---"
    )

    torch.npu.manual_seed(0)
    dtypes = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
    dtype_a = dtypes[args.dtype_a]
    dtype_b = dtypes[args.dtype_b]
    dtype_c = dtypes[args.dtype_c]

    a = torch.rand(args.m, args.k, dtype=dtype_a, device="npu") * 10.0 - 5.0
    b = torch.rand(args.k, args.n, dtype=dtype_b, device="npu") * 10.0 - 5.0
    d = torch.zeros(args.m, args.n, dtype=dtype_c, device="npu")
    workspace = torch.zeros(args.m, args.n, dtype=dtype_c, device="npu")
    ref = torch.nn.functional.leaky_relu(
        a.float() @ b.float(), negative_slope=float(LEAKY_RELU_ALPHA)
    )
    if dtype_c in (torch.float16, torch.bfloat16):
        ref = ref.to(dtype_c).float()

    a = a.contiguous() if args.layout_a == "row" else a.permute(1, 0).contiguous()
    b = b.contiguous() if args.layout_b == "row" else b.permute(1, 0).contiguous()
    a_tensor = create_tla_tensor(a, args.layout_a)
    b_tensor = create_tla_tensor(b, args.layout_b)
    d_tensor = create_tla_tensor(d, "row")
    wks_tensor = create_tla_tensor(workspace, "row")

    _ub_slot_elements, _simd_lanes = _compute_ub_slots(args.dtype_c)
    artifact = tla.compile(
        matmul_evg_leaky_relu_kernel,
        a_tensor,
        b_tensor,
        d_tensor,
        wks_tensor,
        TilingParams(),
        SwizzleParams(
            SWIZZLE_DIRECTION=0 if int(args.m) > int(args.n) else 1,
            SWIZZLE_OFFSET=3,
        ),
        _ub_slot_elements,
        _simd_lanes,
        # The slots below divide the whole 256 KB, which only fits once the
        # compiler stops holding back its 8 KB reserve. The kernel does not spill,
        # so releasing the VF stack half is safe here.
        options=(
            "--npu-arch 3510"
            " --cce-disable-asc-reserved-ubuf"
            " --cce-disable-vf-stack-reserved-ubuf"
        ),
    )
    block_num = get_block_num(args.block_num, args.device, kind="cube")
    artifact(a_tensor, b_tensor, d_tensor, wks_tensor, block_num=block_num)
    torch.npu.synchronize()

    passed = compare(d.detach().cpu(), ref.cpu(), args.k)
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--layout-a", choices=("row", "col"), default="row")
    p.add_argument("--layout-b", choices=("row", "col"), default="row")
    p.add_argument("--dtype-a", choices=("f16", "bf16", "f32"), default="f32")
    p.add_argument("--dtype-b", choices=("f16", "bf16", "f32"), default="f32")
    p.add_argument("--dtype-c", choices=("f16", "f32"), default="f32")
    p.add_argument("--block-num", type=int, default=-1)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
