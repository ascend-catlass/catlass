# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Multi-core split-K matmul: Kernel + Host in one file.

AIC partial MMAD → workspace; AIV ReduceAdd → C. Dynamic GM.
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


DESCRIPTION = "Multi-core split-K matmul; dynamic GM."


@dataclass(frozen=True)
class SplitKParams:
    splitk_factor: tla.Constexpr[int]
    element_count: tla.Constexpr[int]
    task_per_aiv: tla.Constexpr[int]
    reduce_loops: tla.Constexpr[int]
    ub_row_stride: tla.Constexpr[int]
    reduce_vl_loops: tla.Constexpr[int]


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.kernel
def multi_core_splitk_mmad_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    gm_workspace: tla.Tensor,
    _tiling: tla.Constexpr[TilingParams],
    _swizzle: tla.Constexpr[SwizzleParams],
    _splitk: tla.Constexpr[SplitKParams],
) -> None:
    """AIC writes partial GEMM tiles to workspace; AIV reduces split-K slices to C."""
    c0 = 0
    c1 = 1

    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype
    dtype_w = gm_workspace.ptr.dtype
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

    # UB: split-K gather rows; optional narrow-type cast staging.
    ub_reduce_ptr = tla.allocate(COMPUTE_LENGTH, dtype_w, tla.AddressSpace.ub, 256)

    # Narrowing cast to GM C: f16 uses floor rounding, bf16 uses round-to-nearest.
    if tla.const_expr(dtype_gm_c != tla.Float32):
        ub_cast_ptr = tla.allocate(
            _splitk.task_per_aiv, dtype_gm_c, tla.AddressSpace.ub, 256
        )
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
    mn_loops = grid_m * grid_n
    core_loops = mn_loops * _splitk.splitk_factor
    # Tail band width (equals _swizzle.SWIZZLE_OFFSET when grid divides evenly).
    last_n_row = grid_m - _swizzle.SWIZZLE_OFFSET * (
        (grid_m + _swizzle.SWIZZLE_OFFSET - 1) // _swizzle.SWIZZLE_OFFSET - 1
    )
    last_n_col = grid_n - _swizzle.SWIZZLE_OFFSET * (
        (grid_n + _swizzle.SWIZZLE_OFFSET - 1) // _swizzle.SWIZZLE_OFFSET - 1
    )

    with tla.cube():
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

        task_range = tla.range(tla.arch.block_idx(), core_loops, tla.arch.block_num())
        for task in task_range:
            # Map linear task id → (split-K slice, M×N tile).
            slice_idx = (task % core_loops) // mn_loops
            inner_idx = task % mn_loops

            # Map linear tile index to (block_row, block_col) via Zn/Nz swizzle.
            # _swizzle.SWIZZLE_DIRECTION is host-set before compile (0=Zn when m>n, else Nz).
            if tla.const_expr(_swizzle.SWIZZLE_DIRECTION == 0):
                # Zn: bands along M, serpentine N
                tile_block_loop = (
                    grid_m + _swizzle.SWIZZLE_OFFSET - 1
                ) // _swizzle.SWIZZLE_OFFSET
                tile_block_idx = inner_idx // (_swizzle.SWIZZLE_OFFSET * grid_n)
                in_tile = inner_idx % (_swizzle.SWIZZLE_OFFSET * grid_n)
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
                tile_block_idx = inner_idx // (_swizzle.SWIZZLE_OFFSET * grid_m)
                in_tile = inner_idx % (_swizzle.SWIZZLE_OFFSET * grid_m)
                n_col = (
                    last_n_col
                    if tile_block_idx == tile_block_loop - 1
                    else _swizzle.SWIZZLE_OFFSET
                )
                block_row = in_tile // n_col
                block_col = tile_block_idx * _swizzle.SWIZZLE_OFFSET + in_tile % n_col
                odd = tile_block_idx % 2
                block_row = block_row + odd * (grid_m - 1 - 2 * block_row)

            # Uneven K split: first (k % factor) slices get one extra K tile.
            rem = k_tile_num % _splitk.splitk_factor
            quot = k_tile_num // _splitk.splitk_factor
            k_start = slice_idx * quot + rem
            slice_tiles = quot
            if slice_idx < rem:
                k_start = (quot + 1) * slice_idx
                slice_tiles = quot + 1

            # GM views for this M×N tile and workspace slice row.
            gm_a_by_core = tla.tile_view(
                gm_a, tla.make_shape(_tiling.l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                gm_b, tla.make_shape(k, _tiling.l1_tn), tla.make_coord(c0, block_col)
            )
            gm_ws_plane = tla.tile_view(
                gm_workspace, tla.make_shape(m, n), tla.make_coord(slice_idx, c0)
            )
            gm_w_by_core = tla.tile_view(
                gm_ws_plane,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(block_row, block_col),
            )

            l0_c = tla.make_tensor_like(l0c_ptr, gm_w_by_core)

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

                # Select ping or pong L1 buffer for this K step.
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

                    # L0 A/B ping-pong is independent of L1 selection.
                    l0_a = tla.make_tensor_like(
                        l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_by_l0
                    )
                    l0_b = tla.make_tensor_like(
                        l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_by_l0
                    )

                    # First L0 sub-tile: wait until L1 load completes.
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

                    # MMAD; unit_flag marks last micro-tile when enabled.
                    tla.set_flag(l0_ab_data_ready)
                    tla.wait_flag(l0_ab_data_ready)

                    unit_flag = (
                        0b11
                        if (k_local == slice_tiles - 1) and (k_l0 == k_l0_count - 1)
                        else 0b10
                    )
                    init_c = True if k_local == 0 and k_l0 == 0 else False
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)

                    # Release L0 ping-pong for the next L0 sub-tile.
                    if l0_buf_idx == c0:
                        tla.set_flag(l0a0_available)
                        tla.set_flag(l0b0_available)
                    else:
                        tla.set_flag(l0a1_available)
                        tla.set_flag(l0b1_available)
                    l0_buf_idx = c1 - l0_buf_idx

                l1_buf_idx = c1 - l1_buf_idx

            tla.copy(
                gm_w_by_core,
                l0_c,
                tla.params.CopyL0C2DstParams(unit_flag=0b11),
            )

        # Drain L1/L0 pipelines before cross-core handoff.
        tla.wait_flag(l1a0_available)
        tla.wait_flag(l1a1_available)
        tla.wait_flag(l1b0_available)
        tla.wait_flag(l1b1_available)
        tla.wait_flag(l0a0_available)
        tla.wait_flag(l0a1_available)
        tla.wait_flag(l0b0_available)
        tla.wait_flag(l0b1_available)
        tla.wait_flag(l0c_available)

        tla.cross_core_set_flag(cross_aic_to_aiv_done, tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        # Wait for paired AIC, then all-AIV barrier.
        tla.cross_core_wait_flag(cross_aic_to_aiv_done, tla.arch.MTE2)
        tla.cross_core_set_flag(cross_aiv_barrier, tla.arch.MTE2)
        tla.cross_core_wait_flag(cross_aiv_barrier, tla.arch.MTE2)

        sub = tla.arch.sub_block_idx()
        aiv_id = tla.arch.block_idx() * SUB_BLOCK_NUM + sub
        aiv_num = tla.arch.block_num() * SUB_BLOCK_NUM

        gm_ws_plane = tla.tile_view(
            gm_workspace, tla.make_shape(m, n), tla.make_coord(c0, c0)
        )
        ws_base_ptr = gm_ws_plane.ptr
        c_plane = tla.tile_view(gm_c, tla.make_shape(m, n), tla.make_coord(c0, c0))
        c_base_ptr = c_plane.ptr

        ub_row_layout = tla.make_layout(
            tla.make_shape(_splitk.ub_row_stride), tla.make_stride(1)
        )
        ub_reduce_acc = tla.make_tensor(ub_reduce_ptr, ub_row_layout)

        tla.set_flag(reduce_mte3_mte2)

        loop_range = tla.range(aiv_id, _splitk.reduce_loops, aiv_num)

        _task_per_aiv = tla.as_numeric(_splitk.task_per_aiv).to(tla.Int64)
        for loop_idx in loop_range:
            src_off = loop_idx.to(tla.Int64) * _task_per_aiv
            remaining = tla.as_numeric(_splitk.element_count).to(
                tla.Int64
            ) - src_off.to(tla.Int64)
            actual_tile_len = _task_per_aiv
            if remaining < _task_per_aiv:
                actual_tile_len = remaining

            tla.wait_flag(reduce_mte3_mte2)

            # Gather all split-K workspace rows for this flat M×N chunk into UB.
            gm_ws_gather = tla.make_tensor(
                ws_base_ptr + src_off.to(tla.Int64),
                tla.make_layout(
                    tla.make_shape(_splitk.splitk_factor, _splitk.task_per_aiv),
                    tla.make_stride(_splitk.element_count, 1),
                    origin_shape=tla.make_shape(_splitk.splitk_factor, actual_tile_len),
                ),
            )
            ub_ws_gather = tla.make_tensor(
                ub_reduce_ptr,
                tla.make_layout(
                    tla.make_shape(_splitk.splitk_factor, _splitk.ub_row_stride),
                    tla.make_stride(_splitk.ub_row_stride, 1),
                    origin_shape=tla.make_shape(_splitk.splitk_factor, actual_tile_len),
                ),
            )
            tla.copy(ub_ws_gather, gm_ws_gather)

            tla.set_flag(reduce_mte2_v)
            tla.wait_flag(reduce_mte2_v)

            # Sum split-K UB rows; optionally cast narrow types and densify stores.
            with tla.vec.func(mode="simd"):
                # Add workspace rows 1..factor-1 into row 0.
                vec_chunk_shape = tla.make_shape(1, ELE_PER_VECTOR_BLOCK)
                for sk_idx in tla.range(1, _splitk.splitk_factor):
                    for vl_idx in tla.range(_splitk.reduce_vl_loops):
                        reduce_acc_chunk = tla.tile_view(
                            ub_reduce_acc,
                            tla.make_shape(ELE_PER_VECTOR_BLOCK),
                            tla.make_coord(vl_idx),
                        )
                        reduce_src_chunk = tla.tile_view(
                            ub_ws_gather,
                            vec_chunk_shape,
                            tla.make_coord(sk_idx, vl_idx),
                        )
                        reduce_acc_chunk.store(
                            reduce_acc_chunk.load() + reduce_src_chunk.load(),
                        )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )

                if tla.const_expr(dtype_gm_c != tla.Float32):
                    # f32→f16/bf16 cast leaves values in low-16 of each B32 slot;
                    # DIST_PACK_B32 packs those halves densely (replaces deinterleave).
                    cast_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    store_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype_gm_c)
                    pack_store = NormalStoreParams(store_dist=StoreDist.DIST_PACK_B32)
                    ub_out_1d = tla.make_tensor(
                        ub_cast_ptr,
                        tla.make_layout(
                            tla.make_shape(_splitk.task_per_aiv), tla.make_stride(1)
                        ),
                    )
                    for cast_vl_idx in tla.range(_splitk.reduce_vl_loops):
                        cast_acc_chunk = tla.tile_view(
                            ub_reduce_acc,
                            tla.make_shape(ELE_PER_VECTOR_BLOCK),
                            tla.make_coord(cast_vl_idx),
                        )
                        cast_out_chunk = tla.tile_view(
                            ub_out_1d,
                            tla.make_shape(ELE_PER_VECTOR_BLOCK),
                            tla.make_coord(cast_vl_idx),
                        )
                        acc_v = cast_acc_chunk.load()
                        out_v = acc_v.to(dtype_gm_c, cast_trait_zero, cast_mask)
                        cast_out_chunk.store(out_v, pack_store, mask=store_mask)

            tla.set_flag(reduce_v_mte3)
            tla.wait_flag(reduce_v_mte3)

            if tla.const_expr(dtype_gm_c != tla.Float32):
                gm_out = tla.make_tensor(
                    c_base_ptr + src_off,
                    tla.make_layout(
                        tla.make_shape(1, _splitk.task_per_aiv),
                        tla.make_stride(n, 1),
                        origin_shape=tla.make_shape(1, actual_tile_len),
                    ),
                )
                ub_cast = tla.make_tensor(
                    ub_cast_ptr,
                    tla.make_layout(
                        tla.make_shape(1, _splitk.task_per_aiv),
                        tla.make_stride(_splitk.task_per_aiv, 1),
                        origin_shape=tla.make_shape(1, actual_tile_len),
                    ),
                )
                tla.copy(gm_out, ub_cast)
            else:
                gm_out = tla.make_tensor(
                    c_base_ptr + src_off,
                    tla.make_layout(
                        tla.make_shape(1, _splitk.task_per_aiv),
                        tla.make_stride(n, 1),
                        origin_shape=tla.make_shape(1, actual_tile_len),
                    ),
                )
                ub_out_fp32 = tla.make_tensor(
                    ub_reduce_ptr,
                    tla.make_layout(
                        tla.make_shape(1, _splitk.task_per_aiv),
                        tla.make_stride(_splitk.ub_row_stride, 1),
                        origin_shape=tla.make_shape(1, actual_tile_len),
                    ),
                )
                tla.copy(gm_out, ub_out_fp32)

            tla.set_flag(reduce_mte3_mte2)

        tla.wait_flag(reduce_mte3_mte2)
        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


def get_splitk_factor(
    m_val: int,
    n_val: int,
    k_val: int,
    tiling_params: TilingParams,
    aic_core_num: int,
) -> int:
    factor = 2
    block_num = ((m_val + tiling_params.l1_tm - 1) // tiling_params.l1_tm) * (
        (n_val + tiling_params.l1_tn - 1) // tiling_params.l1_tn
    )
    k_tile_num = (k_val + tiling_params.l1_tk - 1) // tiling_params.l1_tk
    if aic_core_num // block_num > 0:
        factor = aic_core_num // block_num
    return min(factor, k_tile_num)


def workspace_elems(m_val: int, n_val: int, factor: int) -> int:
    min_elems = (2 * 1024 * 1024) // 4
    if factor * m_val * n_val >= min_elems:
        return factor * m_val * n_val
    rows = max(factor * m_val, (min_elems + n_val - 1) // n_val)
    return rows * n_val


def workspace_shape(m_val: int, n_val: int, factor: int) -> tuple[int, int]:
    elems = workspace_elems(m_val, n_val, factor)
    rows = max(factor * m_val, (elems + n_val - 1) // n_val)
    if rows % m_val != 0:
        rows = ((rows + m_val - 1) // m_val) * m_val
    return rows, n_val


def compute_reduce_tiling(
    m_val: int, n_val: int, factor: int, aic_core_num: int
) -> SplitKParams:
    if aic_core_num <= 0:
        raise ValueError(f"aic_core_num must be positive; got {aic_core_num}")
    elem_count = m_val * n_val
    aiv_num = aic_core_num * 2
    per_aiv = (elem_count + aiv_num - 1) // aiv_num
    task = (
        (per_aiv + ELE_PER_VECTOR_BLOCK - 1) // ELE_PER_VECTOR_BLOCK
    ) * ELE_PER_VECTOR_BLOCK
    task_vl_cap = 512 * ELE_PER_VECTOR_BLOCK
    task_stage_cap = (
        COMPUTE_LENGTH // factor // ELE_PER_VECTOR_BLOCK * ELE_PER_VECTOR_BLOCK
    )
    task = min(task, task_stage_cap, task_vl_cap)
    if task == 0:
        task = ELE_PER_VECTOR_BLOCK
    ub_stride = ((task + ELE_NUM_ALIGN - 1) // ELE_NUM_ALIGN) * ELE_NUM_ALIGN
    if factor * ub_stride > COMPUTE_LENGTH:
        raise ValueError(
            f"splitk reduce UB overflow: factor={factor} ub_row_stride={ub_stride}"
        )
    loops = (elem_count + task - 1) // task
    return SplitKParams(
        splitk_factor=factor,
        element_count=elem_count,
        task_per_aiv=task,
        reduce_loops=loops,
        ub_row_stride=ub_stride,
        reduce_vl_loops=(task + ELE_PER_VECTOR_BLOCK - 1) // ELE_PER_VECTOR_BLOCK,
    )


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

    # Prepare multi-core splitk arguments
    block_num = get_block_num(args.block_num, args.device, kind="cube")
    factor = get_splitk_factor(args.m, args.n, args.k, _tiling_params, block_num)
    _splitk_params = compute_reduce_tiling(args.m, args.n, factor, block_num)

    a = torch.rand(args.m, args.k, dtype=dtype_a, device="cpu") * 10.0 - 5.0
    b = torch.rand(args.k, args.n, dtype=dtype_b, device="cpu") * 10.0 - 5.0
    c = torch.rand(args.m, args.n, dtype=dtype_c, device="cpu") * 10.0 - 5.0
    ref = a.float() @ b.float()
    if dtype_c in (torch.float16, torch.bfloat16):
        ref = ref.to(dtype_c).float()

    ws_rows, ws_cols = workspace_shape(args.m, args.n, factor)
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
        multi_core_splitk_mmad_kernel,
        a_tensor,
        b_tensor,
        c_tensor,
        w_tensor,
        _tiling_params,
        _swizzle_params,
        _splitk_params,
        options=(
            "--npu-arch 3510"
            " --cce-disable-asc-reserved-ubuf"
            " --cce-disable-vf-stack-reserved-ubuf"
        ),
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
