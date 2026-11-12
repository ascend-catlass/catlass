# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tail multi-core split-K matmul: Kernel + Host in one file.

Normal M×N tiles: full-K → GM C. Tail tiles: split-K → workspace + AIV ReduceAdd.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

from dataclasses import dataclass
import argparse

import catlass.tla as tla
import torch
import torch_npu  # noqa: F401
from catlass.params import NormalStoreParams, StoreDist

from helpers import (
    TilingParams,
    SwizzleParams,
)

SUB_BLOCK_NUM = 2
ELE_PER_VECTOR_BLOCK = 64
ELE_NUM_ALIGN = 8
COMPUTE_LENGTH = 192 * 1024 // 4

DESCRIPTION = "Tail multi-core split-K matmul; dynamic GM."


@dataclass(frozen=True)
class TailSplitKParams:
    normal_block_num: tla.Constexpr[int]
    tail_block_num: tla.Constexpr[int]
    splitk_factor: tla.Constexpr[int]
    core_loops: tla.Constexpr[int]
    tile_per_core: tla.Constexpr[int]
    ub_row_stride: tla.Constexpr[int]
    reduce_vl_loops: tla.Constexpr[int]
    chunk_elems: tla.Constexpr[int]


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.kernel
def tail_multi_core_splitk_mmad_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    gm_w: tla.Tensor,
    _tiling: tla.Constexpr[TilingParams],
    _swizzle: tla.Constexpr[SwizzleParams],
    _tail_splitk: tla.Constexpr[TailSplitKParams],
) -> None:
    """Normal tiles write GM C; tail tiles split-K via workspace and AIV ReduceAdd."""
    c0 = 0
    c1 = 1

    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype
    dtype_w = gm_w.ptr.dtype
    dtype_gm_c = gm_c.ptr.dtype
    DTYPE_C = tla.Float32  # L0C accumulator only

    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    k = gm_a.origin_shape[1]

    # AIC pipeline flags: MTE2 ↔ MTE1 ↔ Cube ↔ FIX
    l1a0_data_ready = tla.flag("l1a0_copy_end", tla.arch.MTE2, tla.arch.MTE1)
    l1a1_data_ready = tla.flag("l1a1_copy_end", tla.arch.MTE2, tla.arch.MTE1)
    l1b0_data_ready = tla.flag("l1b0_copy_end", tla.arch.MTE2, tla.arch.MTE1)
    l1b1_data_ready = tla.flag("l1b1_copy_end", tla.arch.MTE2, tla.arch.MTE1)
    l1a0_available = tla.flag("l1a0_copy_start", tla.arch.MTE1, tla.arch.MTE2)
    l1a1_available = tla.flag("l1a1_copy_start", tla.arch.MTE1, tla.arch.MTE2)
    l1b0_available = tla.flag("l1b0_copy_start", tla.arch.MTE1, tla.arch.MTE2)
    l1b1_available = tla.flag("l1b1_copy_start", tla.arch.MTE1, tla.arch.MTE2)
    l0a0_available = tla.flag("l0a0_copy_start", tla.arch.CUBE, tla.arch.MTE1)
    l0a1_available = tla.flag("l0a1_copy_start", tla.arch.CUBE, tla.arch.MTE1)
    l0b0_available = tla.flag("l0b0_copy_start", tla.arch.CUBE, tla.arch.MTE1)
    l0b1_available = tla.flag("l0b1_copy_start", tla.arch.CUBE, tla.arch.MTE1)
    l0_ab_data_ready = tla.flag("l0_copy_end", tla.arch.MTE1, tla.arch.CUBE)
    l0c_available = tla.flag("fix_done", tla.arch.FIX, tla.arch.CUBE)

    # Cross-core: paired AIC→AIV done (mode 2); AIV all-block barrier (mode 0).
    cross_aic_to_aiv_done = tla.cross_flag("aic_finish", mode=2)
    cross_aiv_barrier = tla.cross_flag("aiv_ibarrier", mode=0)

    # AIV reduce pipeline: MTE3 → MTE2 → vector → MTE3
    reduce_mte3_mte2 = tla.flag("reduce_mte3_mte2", tla.arch.MTE3, tla.arch.MTE2)
    reduce_mte2_v = tla.flag("reduce_mte2_v", tla.arch.MTE2, tla.arch.VECTOR)
    reduce_v_mte3 = tla.flag("reduce_v_mte3", tla.arch.VECTOR, tla.arch.MTE3)

    # L1 ping-pong tiles for A (M×K) and B (K×N)
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
    # L0 ping-pong tiles inside Cube
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

    # UB: split-K gather rows for tail-tile reduce.
    ub_reduce_ptr = tla.allocate(COMPUTE_LENGTH, dtype_w, tla.AddressSpace.ub, 256)
    if tla.const_expr(dtype_gm_c != tla.Float32):
        ub_cast_ptr = tla.allocate(
            _tail_splitk.chunk_elems, dtype_gm_c, tla.AddressSpace.ub, 256
        )

    # Narrowing cast to GM C: f16 uses floor rounding, bf16 uses round-to-nearest.
    if tla.const_expr(dtype_gm_c != tla.Float32):
        if tla.const_expr(dtype_gm_c == tla.Float16):
            cast_trait_zero = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.NOSAT,
                round_mode=tla.params.RoundMode.CAST_FLOOR,
            )
        else:
            cast_trait_zero = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.NOSAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )

    # Tile grid over M×N and K slices per AIC
    grid_m = (m + _tiling.l1_tm - 1) // _tiling.l1_tm
    grid_n = (n + _tiling.l1_tn - 1) // _tiling.l1_tn
    k_tile_num = (k + _tiling.l1_tk - 1) // _tiling.l1_tk
    # Tail band width (equals _swizzle.SWIZZLE_OFFSET when grid divides evenly).
    last_n_row = grid_m - _swizzle.SWIZZLE_OFFSET * (
        (grid_m + _swizzle.SWIZZLE_OFFSET - 1) // _swizzle.SWIZZLE_OFFSET - 1
    )
    last_n_col = grid_n - _swizzle.SWIZZLE_OFFSET * (
        (grid_n + _swizzle.SWIZZLE_OFFSET - 1) // _swizzle.SWIZZLE_OFFSET - 1
    )

    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)

        # Initial flag state so the first loop wait does not block.
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
        bid = tla.arch.block_idx()
        bdim = tla.arch.block_num()
        tail_cores = _tail_splitk.core_loops - _tail_splitk.normal_block_num

        # Normal-only AICs signal completion once before the task loop.
        if bid >= tail_cores:
            tla.cross_core_set_flag(cross_aic_to_aiv_done, tla.arch.FIX)

        task_range = tla.range(bid, _tail_splitk.core_loops, bdim)
        for loop_idx in task_range:
            # Remap loop index: normal tiles first, then tail tiles per split-K slice.
            actual = loop_idx
            if _tail_splitk.normal_block_num > 0:
                if (
                    loop_idx == _tail_splitk.normal_block_num - bdim + bid
                    and bid < tail_cores
                ):
                    actual = _tail_splitk.normal_block_num + bid
                elif loop_idx >= _tail_splitk.normal_block_num:
                    actual = _tail_splitk.normal_block_num - bdim + bid

            inner = actual % _tail_splitk.core_loops
            is_tail = 1 if inner >= _tail_splitk.normal_block_num else 0

            # Uneven K split for tail tiles: first (k % factor) slices get one extra tile.
            base_block = inner
            k_start = 0
            slice_tiles = k_tile_num
            rem = k_tile_num % _tail_splitk.splitk_factor
            quot = k_tile_num // _tail_splitk.splitk_factor
            if is_tail == 1:
                base_block = (
                    _tail_splitk.normal_block_num
                    + (inner - _tail_splitk.normal_block_num)
                    // _tail_splitk.splitk_factor
                )
                slice_in_group = (
                    inner - _tail_splitk.normal_block_num
                ) % _tail_splitk.splitk_factor
                k_start = slice_in_group * quot + rem
                slice_tiles = quot
                if slice_in_group < rem:
                    k_start = (quot + 1) * slice_in_group
                    slice_tiles = quot + 1

            # Map linear tile index to (block_row, block_col) via Zn/Nz swizzle.
            # _swizzle.SWIZZLE_DIRECTION is host-set before compile (0=Zn when m>n, else Nz).
            if tla.const_expr(_swizzle.SWIZZLE_DIRECTION == 0):
                # Zn: bands along M, serpentine N
                tile_block_loop = (
                    grid_m + _swizzle.SWIZZLE_OFFSET - 1
                ) // _swizzle.SWIZZLE_OFFSET
                tile_block_idx = base_block // (_swizzle.SWIZZLE_OFFSET * grid_n)
                in_tile = base_block % (_swizzle.SWIZZLE_OFFSET * grid_n)
                n_row = (
                    last_n_row
                    if tile_block_idx == tile_block_loop - 1
                    else _swizzle.SWIZZLE_OFFSET
                )
                block_row = tile_block_idx * _swizzle.SWIZZLE_OFFSET + in_tile % n_row
                block_col = in_tile // n_row
                odd = tile_block_idx % 2
                block_col = block_col + odd * (grid_n - 1 - 2 * block_col)
            else:
                # Nz: bands along N, serpentine M
                tile_block_loop = (
                    grid_n + _swizzle.SWIZZLE_OFFSET - 1
                ) // _swizzle.SWIZZLE_OFFSET
                tile_block_idx = base_block // (_swizzle.SWIZZLE_OFFSET * grid_m)
                in_tile = base_block % (_swizzle.SWIZZLE_OFFSET * grid_m)
                n_col = (
                    last_n_col
                    if tile_block_idx == tile_block_loop - 1
                    else _swizzle.SWIZZLE_OFFSET
                )
                block_row = in_tile // n_col
                block_col = tile_block_idx * _swizzle.SWIZZLE_OFFSET + in_tile % n_col
                odd = tile_block_idx % 2
                block_row = block_row + odd * (grid_m - 1 - 2 * block_row)

            # GM views for this M×N tile.
            gm_a_by_core = tla.tile_view(
                gm_a, tla.make_shape(_tiling.l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                gm_b, tla.make_shape(k, _tiling.l1_tn), tla.make_coord(c0, block_col)
            )
            gm_c_by_core = tla.tile_view(
                gm_c,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(block_row, block_col),
            )
            # Per-AIC workspace slot, cropped to actual tile shape.
            gm_w_by_core = tla.tile_view(
                gm_w,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(bid, c0),
            )
            gm_w_by_tile = tla.tile_view(
                gm_w_by_core,
                tla.make_shape(
                    gm_c_by_core.origin_shape[0], gm_c_by_core.origin_shape[1]
                ),
                tla.make_coord(c0, c0),
            )

            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

            # L1 K loop: copy K tiles from GM into ping-pong L1 buffers.
            k_l1_range = tla.range(c0, slice_tiles, c1)
            for k_local in k_l1_range:
                k_l1 = k_start + k_local
                gm_a_by_l1 = tla.tile_view(
                    gm_a_by_core,
                    tla.make_shape(_tiling.l1_tm, _tiling.l1_tk),
                    tla.make_coord(c0, k_l1),
                )
                gm_b_by_l1 = tla.tile_view(
                    gm_b_by_core,
                    tla.make_shape(_tiling.l1_tk, _tiling.l1_tn),
                    tla.make_coord(k_l1, c0),
                )

                l1_a = tla.make_tensor_like(
                    l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_by_l1
                )
                l1_b = tla.make_tensor_like(
                    l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_by_l1
                )

                # L1 A: wait for free buffer, GM→L1, release load-done flag.
                if l1_buf_idx == c0:
                    tla.wait_flag(l1a0_available)
                else:
                    tla.wait_flag(l1a1_available)
                tla.copy(l1_a, gm_a_by_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1a0_data_ready)
                else:
                    tla.set_flag(l1a1_data_ready)

                # L1 B: same handshake as A.
                if l1_buf_idx == c0:
                    tla.wait_flag(l1b0_available)
                else:
                    tla.wait_flag(l1b1_available)
                tla.copy(l1_b, gm_b_by_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1b0_data_ready)
                else:
                    tla.set_flag(l1b1_data_ready)

                # L0 K loop: L1→L0 ping-pong, MMAD into L0C.
                k_l0_count = (l1_a.origin_shape[1] + _tiling.l0_tk - 1) // _tiling.l0_tk
                k_l0_range = tla.range(c0, k_l0_count, c1)
                for k_l0 in k_l0_range:
                    l1_a_by_l0 = tla.tile_view(
                        l1_a,
                        tla.make_shape(_tiling.l0_tm, _tiling.l0_tk),
                        tla.make_coord(c0, k_l0),
                    )
                    l1_b_by_l0 = tla.tile_view(
                        l1_b,
                        tla.make_shape(_tiling.l0_tk, _tiling.l0_tn),
                        tla.make_coord(k_l0, c0),
                    )

                    l0_a = tla.make_tensor_like(
                        l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_by_l0
                    )
                    l0_b = tla.make_tensor_like(
                        l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_by_l0
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
                    tla.copy(l0_a, l1_a_by_l0)
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
                    tla.copy(l0_b, l1_b_by_l0)
                    if k_l0 == k_l0_count - 1:
                        if l1_buf_idx == c0:
                            tla.set_flag(l1b0_available)
                        else:
                            tla.set_flag(l1b1_available)

                    tla.set_flag(l0_ab_data_ready)
                    tla.wait_flag(l0_ab_data_ready)

                    unit_flag = (
                        0b11
                        if (k_local == slice_tiles - 1) and (k_l0 == k_l0_count - 1)
                        else 0b10
                    )
                    init_c = True if k_local == 0 and k_l0 == 0 else False
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)

                    if l0_buf_idx == c0:
                        tla.set_flag(l0a0_available)
                        tla.set_flag(l0b0_available)
                    else:
                        tla.set_flag(l0a1_available)
                        tla.set_flag(l0b1_available)
                    l0_buf_idx = c1 - l0_buf_idx

                l1_buf_idx = c1 - l1_buf_idx

            # Normal tiles → GM C; tail tiles → per-AIC workspace slot.
            if is_tail == 1:
                tla.copy(
                    gm_w_by_tile,
                    l0_c,
                    tla.params.CopyL0C2DstParams(unit_flag=0b11),
                )
            else:
                tla.copy(
                    gm_c_by_core,
                    l0_c,
                    tla.params.CopyL0C2DstParams(unit_flag=0b11),
                )

            # Tail AICs signal done after workspace store; normal-only AICs already signaled.
            if tla.const_expr(_tail_splitk.normal_block_num > 0):
                if loop_idx == _tail_splitk.normal_block_num - bdim + bid:
                    if bid < tail_cores:
                        tla.pipe_barrier(tla.pipes.ALL)
                        tla.cross_core_set_flag(cross_aic_to_aiv_done, tla.arch.FIX)
            else:
                tla.pipe_barrier(tla.pipes.ALL)
                tla.cross_core_set_flag(cross_aic_to_aiv_done, tla.arch.FIX)

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
        tla.cross_core_wait_flag(cross_aic_to_aiv_done, tla.arch.MTE2)
        tla.cross_core_set_flag(cross_aiv_barrier, tla.arch.MTE2)
        tla.cross_core_wait_flag(cross_aiv_barrier, tla.arch.MTE2)

        sub = tla.arch.sub_block_idx()
        aic_id = tla.arch.block_idx()
        aiv_id = aic_id * SUB_BLOCK_NUM + sub
        tail_limit = _tail_splitk.tail_block_num * _tail_splitk.splitk_factor

        do_reduce = 1 if aic_id < tail_limit else 0
        if do_reduce == 1:
            start_core = (
                aic_id // _tail_splitk.splitk_factor
            ) * _tail_splitk.splitk_factor
            base_mn = (
                _tail_splitk.normal_block_num + start_core // _tail_splitk.splitk_factor
            )
            labor_core_num = _tail_splitk.splitk_factor * SUB_BLOCK_NUM
            loop_start = aiv_id - start_core * SUB_BLOCK_NUM

            # Swizzle for the tail M×N tile (same Zn/Nz as cube path).
            if tla.const_expr(_swizzle.SWIZZLE_DIRECTION == 0):
                tile_block_loop = (
                    grid_m + _swizzle.SWIZZLE_OFFSET - 1
                ) // _swizzle.SWIZZLE_OFFSET
                tile_block_idx = base_mn // (_swizzle.SWIZZLE_OFFSET * grid_n)
                in_tile = base_mn % (_swizzle.SWIZZLE_OFFSET * grid_n)
                n_row = (
                    last_n_row
                    if tile_block_idx == tile_block_loop - 1
                    else _swizzle.SWIZZLE_OFFSET
                )
                block_row = tile_block_idx * _swizzle.SWIZZLE_OFFSET + in_tile % n_row
                block_col = in_tile // n_row
                odd = tile_block_idx % 2
                block_col = block_col + odd * (grid_n - 1 - 2 * block_col)
            else:
                tile_block_loop = (
                    grid_n + _swizzle.SWIZZLE_OFFSET - 1
                ) // _swizzle.SWIZZLE_OFFSET
                tile_block_idx = base_mn // (_swizzle.SWIZZLE_OFFSET * grid_m)
                in_tile = base_mn % (_swizzle.SWIZZLE_OFFSET * grid_m)
                n_col = (
                    last_n_col
                    if tile_block_idx == tile_block_loop - 1
                    else _swizzle.SWIZZLE_OFFSET
                )
                block_row = in_tile // n_col
                block_col = tile_block_idx * _swizzle.SWIZZLE_OFFSET + in_tile % n_col
                odd = tile_block_idx % 2
                block_row = block_row + odd * (grid_m - 1 - 2 * block_row)

            gm_c_by_core = tla.tile_view(
                gm_c,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(block_row, block_col),
            )
            m_act = gm_c_by_core.origin_shape[0]
            n_act = gm_c_by_core.origin_shape[1]
            c_plane = tla.tile_view(
                gm_c,
                tla.make_shape(m, n),
                tla.make_coord(c0, c0),
            )
            c_base_ptr = c_plane.ptr
            ws_plane = tla.tile_view(
                gm_w,
                tla.make_shape(tla.arch.block_num() * _tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(c0, c0),
            )
            ws_base_ptr = ws_plane.ptr

            ub_reduce_acc = tla.make_tensor(
                ub_reduce_ptr,
                tla.make_layout(
                    tla.make_shape(_tail_splitk.chunk_elems), tla.make_stride(1)
                ),
            )

            tla.set_flag(reduce_mte3_mte2)
            loops_num = (
                m_act + _tail_splitk.tile_per_core - 1
            ) // _tail_splitk.tile_per_core
            chunk_range = tla.range(loop_start, loops_num, labor_core_num)
            for loop_idx in chunk_range:
                row_off = loop_idx * _tail_splitk.tile_per_core
                tiles_actual = _tail_splitk.tile_per_core
                remaining = m_act - row_off
                if remaining < _tail_splitk.tile_per_core:
                    tiles_actual = remaining

                tla.wait_flag(reduce_mte3_mte2)

                # Per-slice 2D gather: workspace rows are padded to L1_N, so a flat
                # contiguous read of tiles_actual*n_act is wrong when n_act < L1_N.
                gm_ws_row_layout = tla.make_layout(
                    tla.make_shape(_tail_splitk.tile_per_core, _tiling.l1_tn),
                    tla.make_stride(_tiling.l1_tn, 1),
                    origin_shape=tla.make_shape(tiles_actual, n_act),
                )
                ub_ws_row_layout = tla.make_layout(
                    tla.make_shape(
                        _tail_splitk.tile_per_core, _tail_splitk.ub_row_stride
                    ),
                    tla.make_stride(_tail_splitk.ub_row_stride, 1),
                    origin_shape=tla.make_shape(tiles_actual, n_act),
                )
                for gather_sk_idx in tla.range(_tail_splitk.splitk_factor):
                    gm_ws_slice = tla.make_tensor(
                        ws_base_ptr
                        + (start_core + gather_sk_idx) * _tiling.l1_tm * _tiling.l1_tn
                        + row_off * _tiling.l1_tn,
                        gm_ws_row_layout,
                    )
                    ub_ws_slice = tla.make_tensor(
                        ub_reduce_ptr + gather_sk_idx * _tail_splitk.chunk_elems,
                        ub_ws_row_layout,
                    )
                    tla.copy(ub_ws_slice, gm_ws_slice)

                # Flat UB view over gathered slices (each row = one padded chunk).
                ub_ws_gather = tla.make_tensor(
                    ub_reduce_ptr,
                    tla.make_layout(
                        tla.make_shape(
                            _tail_splitk.splitk_factor, _tail_splitk.chunk_elems
                        ),
                        tla.make_stride(_tail_splitk.chunk_elems, 1),
                    ),
                )

                tla.set_flag(reduce_mte2_v)
                tla.wait_flag(reduce_mte2_v)

                # Sum split-K UB rows; optionally cast narrow types and densify stores.
                with tla.vec.func(mode="simd"):
                    acc_chunk_shape = tla.make_shape(ELE_PER_VECTOR_BLOCK)
                    src_chunk_shape = tla.make_shape(1, ELE_PER_VECTOR_BLOCK)
                    for sk_idx in tla.range(1, _tail_splitk.splitk_factor):
                        for vl_idx in tla.range(_tail_splitk.reduce_vl_loops):
                            reduce_acc_chunk = tla.tile_view(
                                ub_reduce_acc,
                                acc_chunk_shape,
                                tla.make_coord(vl_idx),
                            )
                            reduce_src_chunk = tla.tile_view(
                                ub_ws_gather,
                                src_chunk_shape,
                                tla.make_coord(sk_idx, vl_idx),
                            )
                            reduce_acc_chunk.store(
                                reduce_acc_chunk.load() + reduce_src_chunk.load(),
                            )

                    if tla.const_expr(dtype_gm_c != tla.Float32):
                        # f32→f16/bf16 cast leaves values in low-16 of each B32 slot;
                        # DIST_PACK_B32 packs those halves densely (replaces deinterleave).
                        cast_mask = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        store_mask = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=dtype_gm_c
                        )
                        pack_store = NormalStoreParams(
                            store_dist=StoreDist.DIST_PACK_B32
                        )
                        ub_out_1d = tla.make_tensor(
                            ub_cast_ptr,
                            tla.make_layout(
                                tla.make_shape(_tail_splitk.chunk_elems),
                                tla.make_stride(1),
                            ),
                        )
                        for cast_vl_idx in tla.range(_tail_splitk.reduce_vl_loops):
                            cast_acc_chunk = tla.tile_view(
                                ub_reduce_acc,
                                acc_chunk_shape,
                                tla.make_coord(cast_vl_idx),
                            )
                            cast_out_chunk = tla.tile_view(
                                ub_out_1d,
                                acc_chunk_shape,
                                tla.make_coord(cast_vl_idx),
                            )
                            acc_v = cast_acc_chunk.load()
                            out_v = acc_v.to(dtype_gm_c, cast_trait_zero, cast_mask)
                            cast_out_chunk.store(out_v, pack_store, mask=store_mask)

                tla.set_flag(reduce_v_mte3)
                tla.wait_flag(reduce_v_mte3)

                gm_c_row_layout = tla.make_layout(
                    tla.make_shape(_tail_splitk.tile_per_core, _tiling.l1_tn),
                    tla.make_stride(n, 1),
                    origin_shape=tla.make_shape(tiles_actual, n_act),
                )
                gm_out_ptr = (
                    c_base_ptr
                    + (block_row * _tiling.l1_tm + row_off) * n
                    + block_col * _tiling.l1_tn
                )
                ub_row_layout = tla.make_layout(
                    tla.make_shape(
                        _tail_splitk.tile_per_core, _tail_splitk.ub_row_stride
                    ),
                    tla.make_stride(_tail_splitk.ub_row_stride, 1),
                    origin_shape=tla.make_shape(tiles_actual, n_act),
                )
                if tla.const_expr(dtype_gm_c != tla.Float32):
                    gm_out = tla.make_tensor(gm_out_ptr, gm_c_row_layout)
                    ub_cast = tla.make_tensor(ub_cast_ptr, ub_row_layout)
                    tla.copy(gm_out, ub_cast)
                else:
                    gm_out = tla.make_tensor(gm_out_ptr, gm_c_row_layout)
                    ub_out_fp32 = tla.make_tensor(ub_reduce_ptr, ub_row_layout)
                    tla.copy(gm_out, ub_out_fp32)

                tla.set_flag(reduce_mte3_mte2)

            tla.wait_flag(reduce_mte3_mte2)

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


def compute_tail_scheduler(
    m_val: int, n_val: int, k_val: int, tiling_params: TilingParams, core_num: int
) -> TailSplitKParams:
    """Split M×N grid into normal blocks (full-K → C) and tail blocks (split-K)."""
    if core_num <= 0:
        raise ValueError(f"core_num must be positive; got {core_num}")
    grid_m = (m_val + tiling_params.l1_tm - 1) // tiling_params.l1_tm
    grid_n = (n_val + tiling_params.l1_tn - 1) // tiling_params.l1_tn
    k_tile_num = (k_val + tiling_params.l1_tk - 1) // tiling_params.l1_tk
    mn_blocks = grid_m * grid_n
    t_num = mn_blocks % core_num
    n_num = mn_blocks - t_num
    factor = 1
    if t_num > 0:
        factor = core_num // t_num
    factor = min(factor, k_tile_num)
    loops = n_num + t_num * factor

    # Tail tile row-chunk size, UB stride, and vector loop count per AIV.
    labor = factor * 2
    tile_len_align = (
        (tiling_params.l1_tn + ELE_NUM_ALIGN - 1) // ELE_NUM_ALIGN
    ) * ELE_NUM_ALIGN
    tile_per_core_max = (COMPUTE_LENGTH // labor) // tile_len_align
    if tile_per_core_max == 0:
        tile_per_core_max = 1
    tpc = (tiling_params.l1_tm + labor - 1) // labor
    if tpc > tile_per_core_max:
        tpc = tile_per_core_max
    if tpc > tiling_params.l1_tm:
        tpc = tiling_params.l1_tm
    if tpc == 0:
        tpc = 1
    ub_stride = tile_len_align
    chunk = tpc * ub_stride
    while factor * chunk > COMPUTE_LENGTH and tpc > 1:
        tpc -= 1
        chunk = tpc * ub_stride
    if factor * chunk > COMPUTE_LENGTH:
        raise ValueError(
            f"tail reduce UB overflow: factor={factor} chunk={chunk} "
            f"compute_length={COMPUTE_LENGTH}"
        )

    return TailSplitKParams(
        normal_block_num=n_num,
        tail_block_num=t_num,
        splitk_factor=factor,
        core_loops=loops,
        tile_per_core=tpc,
        ub_row_stride=ub_stride,
        reduce_vl_loops=(chunk + ELE_PER_VECTOR_BLOCK - 1) // ELE_PER_VECTOR_BLOCK,
        chunk_elems=chunk,
    )


def workspace_shape(tiling_params: TilingParams, aic: int) -> tuple[int, int]:
    """Per-AIC L1 tile row in workspace; floor ≥10 MB."""
    min_elems = (10 * 1024 * 1024) // 4
    need = aic * tiling_params.l1_tm * tiling_params.l1_tn
    elems = max(min_elems, need)
    rows = max(
        aic * tiling_params.l1_tm,
        (elems + tiling_params.l1_tn - 1) // tiling_params.l1_tn,
    )
    return rows, tiling_params.l1_tn


def run(args: argparse.Namespace) -> int:
    from helpers import (
        get_block_num,
        create_tla_tensor,
        tolerance,
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

    # Pre define params in need
    _tiling_params = TilingParams()
    _swizzle_params = SwizzleParams(
        SWIZZLE_DIRECTION=0 if args.m > args.n else 1,
        SWIZZLE_OFFSET=3,
    )

    # Prepare tail splitk arguments
    block_num = get_block_num(args.block_num, args.device, kind="cube")
    _tail_splitk_params = compute_tail_scheduler(
        args.m, args.n, args.k, _tiling_params, block_num
    )

    a = torch.rand(args.m, args.k, dtype=dtype_a, device="cpu") * 10.0 - 5.0
    b = torch.rand(args.k, args.n, dtype=dtype_b, device="cpu") * 10.0 - 5.0
    c = torch.rand(args.m, args.n, dtype=dtype_c, device="cpu") * 10.0 - 5.0
    ref = a.float() @ b.float()
    if dtype_c in (torch.float16, torch.bfloat16):
        ref = ref.to(dtype_c).float()

    ws_rows, ws_cols = workspace_shape(_tiling_params, block_num)
    w = torch.zeros(ws_rows, ws_cols, dtype=torch.float32, device="cpu")

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
        tail_multi_core_splitk_mmad_kernel,
        a_tensor,
        b_tensor,
        c_tensor,
        w_tensor,
        _tiling_params,
        _swizzle_params,
        _tail_splitk_params,
        options="--npu-arch 3510",
    )
    artifact(a_tensor, b_tensor, c_tensor, w_tensor, block_num=block_num)
    torch.npu.synchronize()

    budget = 1.0 / 10000.0 if args.dtype_c == "f32" else 1.0 / 1000.0
    result = c.detach().cpu().float()
    thr = tolerance(ref, args.k, bf16=(args.dtype_c == "bf16"))
    bad = (result - ref).abs() > thr
    bad = bad | torch.isnan(result) | torch.isinf(result)
    n_total = int(ref.numel())
    n_bad = int(bad.sum().item())
    mismatch_ratio = (n_bad / n_total) if n_total else 0.0
    passed = mismatch_ratio <= budget
    print(
        f"passed={passed} mismatch={100.0 * mismatch_ratio:.4f}% "
        f"(budget={100.0 * budget:.4f}%) cache_key={artifact.cache_key}"
    )
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--layout-a", choices=("row", "col"), default="row")
    parser.add_argument("--layout-b", choices=("row", "col"), default="row")
    parser.add_argument("--dtype-a", choices=("f16", "bf16", "f32"), default="f16")
    parser.add_argument("--dtype-b", choices=("f16", "bf16", "f32"), default="f16")
    parser.add_argument("--dtype-c", choices=("f16", "bf16", "f32"), default="f16")
    parser.add_argument("--block-num", type=int, default=-1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
