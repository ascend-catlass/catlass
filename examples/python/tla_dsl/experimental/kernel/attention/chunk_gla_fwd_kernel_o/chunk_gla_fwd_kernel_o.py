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
import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Any
import catlass.tla as tla


@tla.kernel
def chunk_gla_fwd_kernel_o_mixed_kernel(
    mem_q: tla.Tensor,
    mem_g: tla.Tensor,
    mem_h: tla.Tensor,
    mem_h_cast: tla.Tensor,
    mem_a: tla.Tensor,
    mem_v: tla.Tensor,
    mem_qg: tla.Tensor,
    mem_a_masked: tla.Tensor,
    mem_state: tla.Tensor,
    mem_intra: tla.Tensor,
    mem_o: tla.Tensor,
) -> None:
    """Compute the complete operator in one paired AIC/AIV launch."""
    inputs_ready = tla.cross_flag("gla_inputs_ready", mode=4)
    l1_free = tla.flag("gla_l1_free", tla.arch.MTE1, tla.arch.MTE2)
    l1_loaded = tla.flag("gla_l1_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l0_free = tla.flag("gla_l0_free", tla.arch.CUBE, tla.arch.MTE1)
    l0_loaded = tla.flag("gla_l0_loaded", tla.arch.MTE1, tla.arch.CUBE)
    mm_done = tla.flag("gla_mm_done", tla.arch.CUBE, tla.arch.FIX)
    fix_free = tla.flag("gla_fix_free", tla.arch.FIX, tla.arch.CUBE)
    q_ub_free = tla.flag("gla_q_ub_free", tla.arch.MTE3, tla.arch.MTE2)
    q_ub_loaded = tla.flag("gla_q_ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    q_vec_done = tla.flag("gla_q_vec_done", tla.arch.VECTOR, tla.arch.MTE3)
    a_ub_free = tla.flag("gla_a_ub_free", tla.arch.MTE3, tla.arch.MTE2)
    a_ub_loaded = tla.flag("gla_a_ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    a_vec_done = tla.flag("gla_a_vec_done", tla.arch.VECTOR, tla.arch.MTE3)
    h_ub_free = tla.flag("gla_h_ub_free", tla.arch.MTE3, tla.arch.MTE2)
    h_ub_loaded = tla.flag("gla_h_ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    h_vec_done = tla.flag("gla_h_vec_done", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.cube():
        l1a_ptr = tla.recast_ptr(
            tla.allocate(
                CHUNK * max(CHUNK, DIM) * INPUT_BYTES,
                tla.UInt8,
                tla.AddressSpace.l1,
                512,
            ),
            dtype=INPUT_DTYPE,
        )
        l1b_ptr = tla.recast_ptr(
            tla.allocate(
                DIM * max(CHUNK, DIM) * INPUT_BYTES, tla.UInt8, tla.AddressSpace.l1, 512
            ),
            dtype=INPUT_DTYPE,
        )
        l0a_ptr = tla.recast_ptr(
            tla.allocate(
                CHUNK * max(CHUNK, DIM) * INPUT_BYTES,
                tla.UInt8,
                tla.AddressSpace.l0a,
                512,
            ),
            dtype=INPUT_DTYPE,
        )
        l0b_ptr = tla.recast_ptr(
            tla.allocate(
                DIM * max(CHUNK, DIM) * INPUT_BYTES,
                tla.UInt8,
                tla.AddressSpace.l0b,
                512,
            ),
            dtype=INPUT_DTYPE,
        )
        l0c_ptr = tla.recast_ptr(
            tla.allocate(
                CHUNK * DIM * F32_BYTES, tla.UInt8, tla.AddressSpace.l0c, 1024
            ),
            dtype=tla.Float32,
        )
        tla.set_flag(l1_free)
        tla.set_flag(l0_free)
        tla.set_flag(fix_free)
        for task_id in tla.range(tla.arch.block_idx(), TASKS, tla.arch.block_num()):
            q_base = task_id * Q_TASK_ELEMS
            h_base = task_id * DIM * DIM
            a_base = task_id * A_TASK_ELEMS
            v_base = task_id * V_TASK_ELEMS
            tla.cross_core_wait_flag(inputs_ready, tla.arch.MTE2, aiv_id=0)
            tla.cross_core_wait_flag(inputs_ready, tla.arch.MTE2, aiv_id=1)
            q_gm = tla.make_tensor(mem_qg.ptr + q_base, _gm_matrix(CHUNK, DIM, DIM))
            h_gm = tla.make_tensor(mem_h_cast.ptr + h_base, _gm_matrix(DIM, DIM, DIM))
            out_gm = tla.make_tensor(mem_o.ptr + v_base, _gm_matrix(CHUNK, DIM, DIM))
            out_l0 = tla.make_tensor_like(l0c_ptr, out_gm, tla.arch.L0Clayout)
            q_l1 = tla.make_tensor_like(l1a_ptr, q_gm)
            h_l1 = tla.make_tensor_like(l1b_ptr, h_gm)
            tla.wait_flag(fix_free)
            tla.wait_flag(l1_free)
            tla.copy(q_l1, q_gm)
            tla.copy(h_l1, h_gm)
            tla.set_flag(l1_loaded)
            tla.wait_flag(l1_loaded)
            q_l0 = tla.make_tensor_like(l0a_ptr, q_l1, tla.arch.zN)
            h_l0 = tla.make_tensor_like(l0b_ptr, h_l1, tla.arch.nZ)
            tla.wait_flag(l0_free)
            tla.copy(q_l0, q_l1)
            tla.copy(h_l0, h_l1)
            tla.set_flag(l1_free)
            tla.set_flag(l0_loaded)
            tla.wait_flag(l0_loaded)
            tla.mmad(out_l0, q_l0, h_l0, init_c=True)
            tla.set_flag(l0_free)
            a_gm = tla.make_tensor(
                mem_a_masked.ptr + a_base, _gm_matrix(CHUNK, CHUNK, CHUNK)
            )
            v_gm = tla.make_tensor(mem_v.ptr + v_base, _gm_matrix(CHUNK, DIM, DIM))
            a_l1 = tla.make_tensor_like(l1a_ptr, a_gm)
            v_l1 = tla.make_tensor_like(l1b_ptr, v_gm)
            tla.wait_flag(l1_free)
            tla.copy(a_l1, a_gm)
            tla.copy(v_l1, v_gm)
            tla.set_flag(l1_loaded)
            tla.wait_flag(l1_loaded)
            a_l0 = tla.make_tensor_like(l0a_ptr, a_l1, tla.arch.zN)
            v_l0 = tla.make_tensor_like(l0b_ptr, v_l1, tla.arch.nZ)
            tla.wait_flag(l0_free)
            tla.copy(a_l0, a_l1)
            tla.copy(v_l0, v_l1)
            tla.set_flag(l1_free)
            tla.set_flag(l0_loaded)
            tla.wait_flag(l0_loaded)
            tla.mmad(out_l0, a_l0, v_l0, init_c=False)
            tla.set_flag(l0_free)
            tla.set_flag(mm_done)
            tla.wait_flag(mm_done)
            tla.copy(out_gm, out_l0)
            tla.set_flag(fix_free)
        tla.wait_flag(l1_free)
        tla.wait_flag(l0_free)
        tla.wait_flag(fix_free)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        q_ub_ptr = tla.recast_ptr(
            tla.allocate(
                Q_VECTOR_ELEMS * INPUT_BYTES, tla.UInt8, tla.AddressSpace.ub, 256
            ),
            dtype=INPUT_DTYPE,
        )
        g_ub_ptr = tla.recast_ptr(
            tla.allocate(
                Q_VECTOR_ELEMS * INPUT_BYTES, tla.UInt8, tla.AddressSpace.ub, 256
            ),
            dtype=INPUT_DTYPE,
        )
        qg_ub_ptr = tla.recast_ptr(
            tla.allocate(
                Q_VECTOR_ELEMS * INPUT_BYTES, tla.UInt8, tla.AddressSpace.ub, 256
            ),
            dtype=INPUT_DTYPE,
        )
        a_ub_ptr = tla.recast_ptr(
            tla.allocate(A_WORK_ELEMS * F32_BYTES, tla.UInt8, tla.AddressSpace.ub, 256),
            dtype=tla.Float32,
        )
        a_masked_ub_ptr = tla.recast_ptr(
            tla.allocate(
                A_WORK_ELEMS * INPUT_BYTES, tla.UInt8, tla.AddressSpace.ub, 256
            ),
            dtype=INPUT_DTYPE,
        )
        a_rne_bits_ub_ptr = tla.recast_ptr(
            tla.allocate(A_WORK_ELEMS * F32_BYTES, tla.UInt8, tla.AddressSpace.ub, 256),
            dtype=tla.Int32,
        )
        a_bits_ub_ptr = tla.recast_ptr(a_ub_ptr, dtype=tla.Int32)
        a_rne_ub_ptr = tla.recast_ptr(a_rne_bits_ub_ptr, dtype=tla.Float32)
        h_ub_ptr = tla.recast_ptr(
            tla.allocate(
                H_VECTOR_ELEMS * F32_BYTES, tla.UInt8, tla.AddressSpace.ub, 256
            ),
            dtype=tla.Float32,
        )
        h_cast_ub_ptr = tla.recast_ptr(
            tla.allocate(
                H_VECTOR_ELEMS * INPUT_BYTES, tla.UInt8, tla.AddressSpace.ub, 256
            ),
            dtype=INPUT_DTYPE,
        )
        tla.set_flag(q_ub_free)
        tla.set_flag(a_ub_free)
        tla.set_flag(h_ub_free)
        sub_block = tla.arch.sub_block_idx()
        for task_id in tla.range(tla.arch.block_idx(), TASKS, tla.arch.block_num()):
            q_offset = task_id * Q_TASK_ELEMS + sub_block * Q_VECTOR_ELEMS
            a_offset = task_id * A_TASK_ELEMS + sub_block * A_VECTOR_ELEMS
            q_gm = tla.make_tensor(mem_q.ptr + q_offset, _flat_layout(Q_VECTOR_ELEMS))
            g_gm = tla.make_tensor(mem_g.ptr + q_offset, _flat_layout(Q_VECTOR_ELEMS))
            qg_gm = tla.make_tensor(mem_qg.ptr + q_offset, _flat_layout(Q_VECTOR_ELEMS))
            q_ub = tla.make_tensor_like(q_ub_ptr, q_gm, tla.arch.RowMajor)
            g_ub = tla.make_tensor_like(g_ub_ptr, g_gm, tla.arch.RowMajor)
            qg_ub = tla.make_tensor_like(qg_ub_ptr, q_gm, tla.arch.RowMajor)
            a_ub = tla.make_tensor(a_ub_ptr, _flat_layout(A_WORK_ELEMS))
            a_bits_ub = tla.make_tensor(a_bits_ub_ptr, _flat_layout(A_WORK_ELEMS))
            a_rne_bits_ub = tla.make_tensor(
                a_rne_bits_ub_ptr, _flat_layout(A_WORK_ELEMS)
            )
            a_rne_ub = tla.make_tensor(a_rne_ub_ptr, _flat_layout(A_WORK_ELEMS))
            a_masked_ub = tla.make_tensor(a_masked_ub_ptr, _flat_layout(A_WORK_ELEMS))
            for _bt128_setup in tla.range_constexpr(0, NUM_BT_TILES - 1, 1):
                full_row = TILE + sub_block * (TILE // 2)
                diag_start = sub_block * TILE
                a_full_offset = task_id * A_TASK_ELEMS + full_row * CHUNK
                a_diag_offset = task_id * A_TASK_ELEMS + diag_start * CHUNK + diag_start
                a_full_gm = tla.make_tensor(
                    mem_a.ptr + a_full_offset, _row_block(TILE // 2, CHUNK)
                )
                a_diag_gm = tla.make_tensor(
                    mem_a.ptr + a_diag_offset, _row_block(TILE, CHUNK)
                )
                a_full_masked_gm = tla.make_tensor(
                    mem_a_masked.ptr + a_full_offset,
                    _row_block(TILE // 2, CHUNK),
                )
                a_diag_masked_gm = tla.make_tensor(
                    mem_a_masked.ptr + a_diag_offset,
                    _row_block(TILE, CHUNK),
                )
                a_full_ub = tla.make_tensor(a_ub_ptr, _row_block(TILE // 2, TILE))
                a_diag_ub = tla.make_tensor(
                    a_ub_ptr + (TILE // 2) * TILE, _row_block(TILE, TILE)
                )
                a_full_masked_ub = tla.make_tensor(
                    a_masked_ub_ptr, _row_block(TILE // 2, TILE)
                )
                a_diag_masked_ub = tla.make_tensor(
                    a_masked_ub_ptr + (TILE // 2) * TILE,
                    _row_block(TILE, TILE),
                )
            for _bt64_setup in tla.range_constexpr(NUM_BT_TILES - 1, 1, 1):
                a_gm = tla.make_tensor(
                    mem_a.ptr + a_offset, _flat_layout(A_VECTOR_ELEMS)
                )
                a_masked_gm = tla.make_tensor(
                    mem_a_masked.ptr + a_offset, _flat_layout(A_VECTOR_ELEMS)
                )
            h_offset = task_id * DIM * DIM + sub_block * H_VECTOR_ELEMS
            h_gm = tla.make_tensor(mem_h.ptr + h_offset, _flat_layout(H_VECTOR_ELEMS))
            h_cast_shape = tla.make_tensor(
                mem_h_cast.ptr + h_offset, _flat_layout(H_VECTOR_ELEMS)
            )
            h_ub = tla.make_tensor_like(h_ub_ptr, h_gm, tla.arch.RowMajor)
            h_cast_ub = tla.make_tensor_like(
                h_cast_ub_ptr, h_cast_shape, tla.arch.RowMajor
            )
            tla.wait_flag(q_ub_free)
            tla.copy(q_ub, q_gm)
            tla.copy(g_ub, g_gm)
            tla.set_flag(q_ub_loaded)
            tla.wait_flag(h_ub_free)
            tla.copy(h_ub, h_gm)
            tla.set_flag(h_ub_loaded)
            tla.wait_flag(a_ub_free)
            for _bt128_load in tla.range_constexpr(0, NUM_BT_TILES - 1, 1):
                tla.copy(a_full_ub, a_full_gm)
                tla.copy(a_diag_ub, a_diag_gm)
            for _bt64_load in tla.range_constexpr(NUM_BT_TILES - 1, 1, 1):
                tla.copy(a_ub, a_gm)
            tla.set_flag(a_ub_loaded)
            tla.wait_flag(q_ub_loaded)
            with tla.vec.func(mode="simd"):
                for tile_id in tla.range_constexpr(
                    0, Q_VECTOR_ELEMS // INPUT_VECTOR_ELEMS, 1
                ):
                    q_tile = tla.tile_view(
                        q_ub,
                        tla.make_shape(INPUT_VECTOR_ELEMS),
                        tla.make_coord(tile_id),
                    )
                    g_tile = tla.tile_view(
                        g_ub,
                        tla.make_shape(INPUT_VECTOR_ELEMS),
                        tla.make_coord(tile_id),
                    )
                    qg_tile = tla.tile_view(
                        qg_ub,
                        tla.make_shape(INPUT_VECTOR_ELEMS),
                        tla.make_coord(tile_id),
                    )
                    q_reg = q_tile.load()
                    g_reg = g_tile.load()
                    q_value_0, q_value_1 = tla.interleave(
                        q_reg.to(tla.Float32, _CAST_WIDEN),
                        q_reg.to(tla.Float32, _CAST_WIDEN_ODD),
                    )
                    g_value_0, g_value_1 = tla.interleave(
                        g_reg.to(tla.Float32, _CAST_WIDEN),
                        g_reg.to(tla.Float32, _CAST_WIDEN_ODD),
                    )
                    qg_value_0 = q_value_0 * tla.exp(g_value_0 * LN2) * SCALE
                    qg_value_1 = q_value_1 * tla.exp(g_value_1 * LN2) * SCALE
                    qg_even, qg_odd = tla.deinterleave(qg_value_0, qg_value_1)
                    qg_tile.store(
                        qg_even.to(INPUT_DTYPE, _CAST_TO_INPUT_EVEN)
                        + qg_odd.to(INPUT_DTYPE, _CAST_TO_INPUT_ODD)
                    )
            tla.set_flag(q_vec_done)
            tla.wait_flag(q_vec_done)
            tla.copy(qg_gm, qg_ub)
            tla.set_flag(q_ub_free)
            tla.wait_flag(h_ub_loaded)
            with tla.vec.func(mode="simd"):
                for tile_id in tla.range_constexpr(
                    0, H_VECTOR_ELEMS // F32_VECTOR_ELEMS, 1
                ):
                    h_tile = tla.tile_view(
                        h_ub,
                        tla.make_shape(F32_VECTOR_ELEMS),
                        tla.make_coord(tile_id),
                    )
                    h_cast_tile = tla.tile_view(
                        h_cast_ub,
                        tla.make_shape(F32_VECTOR_ELEMS),
                        tla.make_coord(tile_id),
                    )
                    h_value = h_tile.load()
                    h_even, h_odd = tla.deinterleave(h_value, h_value)
                    h_cast_tile.store(
                        h_even.to(INPUT_DTYPE, _CAST_TO_INPUT_EVEN)
                        + h_odd.to(INPUT_DTYPE, _CAST_TO_INPUT_ODD)
                    )
            tla.set_flag(h_vec_done)
            tla.wait_flag(h_vec_done)
            tla.copy(h_cast_shape, h_cast_ub)
            tla.set_flag(h_ub_free)
            tla.wait_flag(a_ub_loaded)
            for _rne_pass in tla.range_constexpr(0, RNE_PASSES, 1):
                with tla.vec.func(mode="simd"):
                    zero_i = tla.arange(0, dtype=tla.Int32)
                    zero_i = zero_i - zero_i
                    midpoint_even_mask = tla.full(131071, tla.Int32)
                    midpoint = tla.full(32768, tla.Int32)
                    one_i = tla.full(1, tla.Int32)
                    for tile_id in tla.range_constexpr(
                        0, A_WORK_ELEMS // F32_VECTOR_ELEMS, 1
                    ):
                        bits_tile = tla.tile_view(
                            a_bits_ub,
                            tla.make_shape(F32_VECTOR_ELEMS),
                            tla.make_coord(tile_id),
                        )
                        rne_bits_tile = tla.tile_view(
                            a_rne_bits_ub,
                            tla.make_shape(F32_VECTOR_ELEMS),
                            tla.make_coord(tile_id),
                        )
                        bits = bits_tile.load()
                        tie_to_even = tla.cmp(
                            tla.bitwise_and(bits, midpoint_even_mask),
                            zero_i + midpoint,
                            "eq",
                        )
                        corrected = tla.where(tie_to_even, bits - one_i, bits)
                        rne_bits_tile.store(corrected)
            with tla.vec.func(mode="simd"):
                for _bt128_cast in tla.range_constexpr(0, NUM_BT_TILES - 1, 1):
                    for row in tla.range_constexpr(0, TILE // 2, 1):
                        tile_id = row
                        a_tile = tla.tile_view(
                            a_ub,
                            tla.make_shape(F32_VECTOR_ELEMS),
                            tla.make_coord(tile_id),
                        )
                        rne_tile = tla.tile_view(
                            a_rne_ub,
                            tla.make_shape(F32_VECTOR_ELEMS),
                            tla.make_coord(tile_id),
                        )
                        masked_tile = tla.tile_view(
                            a_masked_ub,
                            tla.make_shape(F32_VECTOR_ELEMS),
                            tla.make_coord(tile_id),
                        )
                        value = rne_tile.load() if RNE_PASSES else a_tile.load()
                        value_even, value_odd = tla.deinterleave(value, value)
                        masked_tile.store(
                            value_even.to(INPUT_DTYPE, _CAST_TO_INPUT_EVEN)
                            + value_odd.to(INPUT_DTYPE, _CAST_TO_INPUT_ODD)
                        )
                    for row in tla.range_constexpr(0, TILE, 1):
                        tile_id = TILE // 2 + row
                        a_tile = tla.tile_view(
                            a_ub,
                            tla.make_shape(F32_VECTOR_ELEMS),
                            tla.make_coord(tile_id),
                        )
                        rne_tile = tla.tile_view(
                            a_rne_ub,
                            tla.make_shape(F32_VECTOR_ELEMS),
                            tla.make_coord(tile_id),
                        )
                        masked_tile = tla.tile_view(
                            a_masked_ub,
                            tla.make_shape(F32_VECTOR_ELEMS),
                            tla.make_coord(tile_id),
                        )
                        cols = tla.arange(0, dtype=tla.Int32)
                        causal = tla.cmp(cols, cols - cols + row, "le")
                        value = rne_tile.load() if RNE_PASSES else a_tile.load()
                        zero = value - value
                        masked = tla.where(causal, value, zero)
                        masked_even, masked_odd = tla.deinterleave(masked, masked)
                        masked_tile.store(
                            masked_even.to(INPUT_DTYPE, _CAST_TO_INPUT_EVEN)
                            + masked_odd.to(INPUT_DTYPE, _CAST_TO_INPUT_ODD)
                        )
                for _bt64_cast in tla.range_constexpr(NUM_BT_TILES - 1, 1, 1):
                    for row in tla.range_constexpr(0, VECTOR_ROWS, 1):
                        global_row = sub_block * VECTOR_ROWS + row
                        tile_id = row
                        a_tile = tla.tile_view(
                            a_ub,
                            tla.make_shape(F32_VECTOR_ELEMS),
                            tla.make_coord(tile_id),
                        )
                        rne_tile = tla.tile_view(
                            a_rne_ub,
                            tla.make_shape(F32_VECTOR_ELEMS),
                            tla.make_coord(tile_id),
                        )
                        masked_tile = tla.tile_view(
                            a_masked_ub,
                            tla.make_shape(F32_VECTOR_ELEMS),
                            tla.make_coord(tile_id),
                        )
                        cols = tla.arange(0, dtype=tla.Int32)
                        causal = tla.cmp(cols, cols - cols + global_row, "le")
                        value = rne_tile.load() if RNE_PASSES else a_tile.load()
                        zero = value - value
                        masked = tla.where(causal, value, zero)
                        masked_even, masked_odd = tla.deinterleave(masked, masked)
                        masked_tile.store(
                            masked_even.to(INPUT_DTYPE, _CAST_TO_INPUT_EVEN)
                            + masked_odd.to(INPUT_DTYPE, _CAST_TO_INPUT_ODD)
                        )
            tla.set_flag(a_vec_done)
            tla.wait_flag(a_vec_done)
            for _bt128_store in tla.range_constexpr(0, NUM_BT_TILES - 1, 1):
                tla.copy(a_full_masked_gm, a_full_masked_ub)
                tla.copy(a_diag_masked_gm, a_diag_masked_ub)
            for _bt64_store in tla.range_constexpr(NUM_BT_TILES - 1, 1, 1):
                tla.copy(a_masked_gm, a_masked_ub)
            tla.set_flag(a_ub_free)
            tla.cross_core_set_flag(inputs_ready, tla.arch.MTE3, aiv_id=0)
            tla.cross_core_set_flag(inputs_ready, tla.arch.MTE3, aiv_id=1)
        tla.wait_flag(q_ub_free)
        tla.wait_flag(a_ub_free)
        tla.wait_flag(h_ub_free)
        tla.pipe_barrier(tla.pipes.ALL)


# DIM is a compile-time specialization used directly by the wide Cube MMA.
DIM = 64
TILE = 64
LN2 = 0.6931471805599453
SCALE = 0.125
INPUT_BYTES = 2
F32_BYTES = 4
F32_VECTOR_ELEMS = 64
INPUT_VECTOR_ELEMS = 128
TASKS = 1
CHUNK = 64
NUM_BT_TILES = 1
VECTOR_ROWS = 32
Q_TASK_ELEMS = CHUNK * DIM
A_TASK_ELEMS = CHUNK * CHUNK
V_TASK_ELEMS = CHUNK * DIM
Q_VECTOR_ELEMS = VECTOR_ROWS * DIM
A_VECTOR_ELEMS = VECTOR_ROWS * CHUNK
A_WORK_ELEMS = A_VECTOR_ELEMS
# Each AIV owns 32 rows of one 64x64 H tile at a time.  Keeping this buffer
# tile-sized avoids exhausting per-sub-block UB when DIM=128.
H_VECTOR_ELEMS = (TILE // 2) * TILE
H_GROUPS = 1
INPUT_DTYPE = tla.BFloat16
INPUT_DTYPE_NAME = "bfloat16"
RNE_PASSES = 1
_CAST_WIDEN = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
_CAST_WIDEN_ODD = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ONE,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
_CAST_TO_INPUT_EVEN = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
_CAST_TO_INPUT_ODD = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ONE,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)


def configure_problem(
    num_tasks: int, chunk_size: int, dim: int, input_dtype: str, scale: float
) -> None:
    """Set compile-time specialization values without changing the ABI."""
    global TASKS, CHUNK, DIM, NUM_BT_TILES, VECTOR_ROWS
    global Q_TASK_ELEMS, A_TASK_ELEMS, V_TASK_ELEMS
    global Q_VECTOR_ELEMS, A_VECTOR_ELEMS, A_WORK_ELEMS
    global INPUT_DTYPE, INPUT_DTYPE_NAME, SCALE
    global RNE_PASSES, H_VECTOR_ELEMS, H_GROUPS
    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be positive, got {num_tasks}")
    if chunk_size not in (64, 128):
        raise ValueError(f"chunk_size must be 64 or 128, got {chunk_size}")
    if dim not in (64, 128):
        raise ValueError(f"D/V must be 64 or 128, got {dim}")
    if input_dtype not in ("float16", "bfloat16"):
        raise ValueError(f"unsupported input dtype: {input_dtype}")
    TASKS = num_tasks
    CHUNK = chunk_size
    DIM = dim
    NUM_BT_TILES = CHUNK // TILE
    VECTOR_ROWS = CHUNK // 2
    Q_TASK_ELEMS = CHUNK * DIM
    A_TASK_ELEMS = CHUNK * CHUNK
    V_TASK_ELEMS = CHUNK * DIM
    Q_VECTOR_ELEMS = VECTOR_ROWS * DIM
    A_VECTOR_ELEMS = VECTOR_ROWS * CHUNK
    A_WORK_ELEMS = 3 * TILE * TILE // 2 if CHUNK == 128 else A_VECTOR_ELEMS
    H_VECTOR_ELEMS = DIM * DIM // 2
    H_GROUPS = 1
    INPUT_DTYPE_NAME = input_dtype
    INPUT_DTYPE = tla.Float16 if input_dtype == "float16" else tla.BFloat16
    RNE_PASSES = 1 if input_dtype == "bfloat16" else 0
    SCALE = float(scale)


def _flat_layout(elements: int):
    return tla.make_layout(
        tla.make_shape(elements), tla.make_stride(1), layoutTag=tla.arch.RowMajor
    )


def _gm_row_tile(row_stride: int):
    return tla.make_layout(
        tla.make_shape(TILE, TILE),
        tla.make_stride(row_stride, 1),
        layoutTag=tla.arch.RowMajor,
    )


def _row_block(rows: int, row_stride: int):
    return tla.make_layout(
        tla.make_shape(rows, TILE),
        tla.make_stride(row_stride, 1),
        layoutTag=tla.arch.RowMajor,
    )


def _gm_matrix(rows: int, cols: int, row_stride: int):
    return tla.make_layout(
        tla.make_shape(rows, cols),
        tla.make_stride(row_stride, 1),
        layoutTag=tla.arch.RowMajor,
    )


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = DEMO_DIR / "artifacts" / "runtime-cache" / "mixed-generalized"
KERNEL = chunk_gla_fwd_kernel_o_mixed_kernel


@dataclass(frozen=True)
class ProblemSpec:
    uuid: str
    mode: str
    input_dtype: str
    q_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    g_shape: tuple[int, ...]
    a_shape: tuple[int, ...]
    h_shape: tuple[int, ...]
    internal_q_bnsd_shape: tuple[int, int, int, int]
    internal_v_bnsd_shape: tuple[int, int, int, int]
    internal_a_bnsd_shape: tuple[int, int, int, int]
    batch: int
    sequence_length: int
    total_tokens: int
    heads: int
    dim: int
    chunk_size: int
    scale: float
    state_v_first: bool
    cu_seqlens: tuple[int, ...] | None
    canonical_chunk_indices: tuple[tuple[int, int], ...] | None
    num_chunks: int
    num_tasks: int
    q_elements: int
    a_elements: int
    v_elements: int
    h_elements: int


def _numel(shape: tuple[int, ...]) -> int:
    value = 1
    for dim in shape:
        value *= dim
    return value


def _diff(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(end - start for start, end in zip(values, values[1:]))


def _validate_cu_values(values: tuple[int, ...], total_tokens: int) -> None:
    if len(values) < 2 or values[0] != 0 or values[-1] != total_tokens:
        raise ValueError(
            f"cu_seqlens must start at 0 and end at {total_tokens}, got {values}"
        )
    if any(end <= start for start, end in zip(values, values[1:])):
        raise ValueError(f"cu_seqlens must be strictly increasing: {values}")


def _prepare_chunk_indices_values(
    cu_values: tuple[int, ...], chunk_size: int
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for sequence_id, length in enumerate(_diff(cu_values)):
        count = (length + chunk_size - 1) // chunk_size
        pairs.extend((sequence_id, local_id) for local_id in range(count))
    return tuple(pairs)


def prepare_chunk_indices_compatible(
    torch: Any, cu_seqlens: Any, chunk_size: int
) -> Any:
    values = tuple(int(x) for x in cu_seqlens.detach().cpu().tolist())
    pairs = _prepare_chunk_indices_values(values, chunk_size)
    return torch.tensor(pairs, dtype=cu_seqlens.dtype, device=cu_seqlens.device)


def _canonicalize_chunk_indices(
    torch: Any, cu_seqlens: Any, chunk_indices: Any, chunk_size: int
) -> tuple[Any, tuple[tuple[int, int], ...]]:
    values = tuple(int(x) for x in cu_seqlens.detach().cpu().tolist())
    canonical = _prepare_chunk_indices_values(values, chunk_size)
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices_compatible(torch, cu_seqlens, chunk_size)
    actual = tuple(
        (int(row[0]), int(row[1])) for row in chunk_indices.detach().cpu().tolist()
    )
    if set(actual) != set(canonical):
        raise ValueError("chunk_indices must cover the canonical chunk set")
    return chunk_indices, canonical


def _validate_direct_tnd_alignment(cu_values: tuple[int, ...], chunk_size: int) -> None:
    lengths = _diff(cu_values)
    if any(value % chunk_size for value in cu_values):
        raise ValueError(
            "direct B=1 BNSD path requires every cu_seqlens boundary aligned "
            f"to {chunk_size}: {cu_values}"
        )
    if any(length % chunk_size for length in lengths):
        raise ValueError(
            "direct B=1 BNSD path requires every sequence length divisible "
            f"by {chunk_size}: {lengths}"
        )


def _parse_problem(case: dict[str, Any]) -> ProblemSpec:
    uuid = str(case["uuid"])
    inputs = case["inputs"]
    q_shape = tuple(int(x) for x in inputs["q"]["shape"])
    v_shape = tuple(int(x) for x in inputs["v"]["shape"])
    g_shape = tuple(int(x) for x in inputs["g"]["shape"])
    a_shape = tuple(int(x) for x in inputs["A"]["shape"])
    h_shape = tuple(int(x) for x in inputs["h"]["shape"])
    input_dtype = str(inputs["q"]["dtype"])
    chunk_size = int(inputs["chunk_size"]["value"])
    scale = float(inputs["scale"]["value"])
    state_v_first = bool(inputs["state_v_first"]["value"])
    if input_dtype not in ("float16", "bfloat16"):
        raise ValueError(f"unsupported dtype in {uuid}: {input_dtype}")
    if inputs["v"]["dtype"] != input_dtype or inputs["g"]["dtype"] != input_dtype:
        raise ValueError(f"q/v/g dtype mismatch in {uuid}")
    if inputs["A"]["dtype"] != "float32" or inputs["h"]["dtype"] != "float32":
        raise ValueError(f"A/h must be float32 in {uuid}")
    if not math.isfinite(scale):
        raise ValueError(f"scale must be finite in {uuid}: {scale}")
    if state_v_first:
        raise ValueError("state_v_first=True is not covered by current workloads")
    if chunk_size not in (64, 128):
        raise ValueError(f"unsupported chunk_size in {uuid}: {chunk_size}")
    if q_shape != v_shape or g_shape != v_shape:
        raise ValueError(f"q/v/g shape mismatch in {uuid}")
    cu_spec = inputs["cu_seqlens"]
    mode = "bsnd" if cu_spec is None else "tnd"
    if mode == "bsnd":
        if len(v_shape) != 4 or len(a_shape) != 4:
            raise ValueError(f"BSND tensors must be rank 4 in {uuid}")
        batch, sequence_length, heads, dim = v_shape
        if sequence_length % chunk_size:
            raise ValueError(f"BSND T must be divisible by BT in {uuid}")
        num_chunks = batch * sequence_length // chunk_size
        cu_values = None
        canonical = None
    else:
        if len(v_shape) != 3 or len(a_shape) != 3:
            raise ValueError(f"TND tensors must be rank 3 in {uuid}")
        sequence_length, heads, dim = v_shape
        batch = 1
        cu_values = tuple(int(x) for x in cu_spec["value"])
        _validate_cu_values(cu_values, sequence_length)
        _validate_direct_tnd_alignment(cu_values, chunk_size)
        canonical = _prepare_chunk_indices_values(cu_values, chunk_size)
        num_chunks = len(canonical)
    if dim not in (64, 128):
        raise ValueError(f"D/V must be 64 or 128 in {uuid}, got {dim}")
    if a_shape[:-1] != v_shape[:-1] or a_shape[-1] != chunk_size:
        raise ValueError(f"unexpected A shape in {uuid}: {a_shape}")
    if h_shape != (num_chunks, heads, dim, dim):
        raise ValueError(f"unexpected h shape in {uuid}: {h_shape}")
    if inputs["chunk_indices"] is not None:
        raise ValueError(f"workload chunk_indices must be null in {uuid}")
    num_tasks = num_chunks * heads
    q_elements = num_tasks * chunk_size * dim
    a_elements = num_tasks * chunk_size * chunk_size
    v_elements = num_tasks * chunk_size * dim
    h_elements = num_tasks * dim * dim
    if q_elements != _numel(q_shape) or v_elements != _numel(v_shape):
        raise ValueError(f"task stream size mismatch in {uuid}")
    return ProblemSpec(
        uuid=uuid,
        mode=mode,
        input_dtype=input_dtype,
        q_shape=q_shape,
        v_shape=v_shape,
        g_shape=g_shape,
        a_shape=a_shape,
        h_shape=h_shape,
        internal_q_bnsd_shape=(batch, heads, sequence_length, dim),
        internal_v_bnsd_shape=(batch, heads, sequence_length, dim),
        internal_a_bnsd_shape=(batch, heads, sequence_length, chunk_size),
        batch=batch,
        sequence_length=sequence_length,
        total_tokens=batch * sequence_length,
        heads=heads,
        dim=dim,
        chunk_size=chunk_size,
        scale=scale,
        state_v_first=state_v_first,
        cu_seqlens=cu_values,
        canonical_chunk_indices=canonical,
        num_chunks=num_chunks,
        num_tasks=num_tasks,
        q_elements=q_elements,
        a_elements=a_elements,
        v_elements=v_elements,
        h_elements=h_elements,
    )


def _meta_tensor(elements: int, dtype: Any) -> Any:
    from catlass import runtime as runtime_mod

    with runtime_mod._eager_capture():
        shape = tla.make_shape(elements)
        return tla.Tensor(
            shape,
            dtype,
            origin_shape=shape,
            coord=tla.make_coord(0),
            stride=tla.make_stride(1),
            layout_tag=tla.arch.RowMajor,
        )


def _tla_input_dtype(name: str) -> Any:
    return tla.Float16 if name == "float16" else tla.BFloat16


def _type_args(spec: ProblemSpec) -> tuple[Any, ...]:
    configure_problem(
        spec.num_tasks, spec.chunk_size, spec.dim, spec.input_dtype, spec.scale
    )
    input_dtype = _tla_input_dtype(spec.input_dtype)
    return (
        _meta_tensor(spec.q_elements, input_dtype),
        _meta_tensor(spec.q_elements, input_dtype),
        _meta_tensor(spec.h_elements, tla.Float32),
        _meta_tensor(spec.h_elements, input_dtype),
        _meta_tensor(spec.a_elements, tla.Float32),
        _meta_tensor(spec.v_elements, input_dtype),
        _meta_tensor(spec.q_elements, input_dtype),
        _meta_tensor(spec.a_elements, input_dtype),
        _meta_tensor(spec.v_elements, tla.Float32),
        _meta_tensor(spec.v_elements, tla.Float32),
        _meta_tensor(spec.v_elements, input_dtype),
    )


def _runtime_kwargs(
    cache_dir: str | Path, *, force_recompile: bool, cache: bool = True
) -> dict[str, Any]:
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        bisheng_bin = str(Path(ascend_home) / "tools" / "bisheng_compiler" / "bin")
        path = os.environ.get("PATH", "")
        if Path(bisheng_bin, "bisheng").is_file() and bisheng_bin not in path.split(
            os.pathsep
        ):
            os.environ["PATH"] = bisheng_bin + os.pathsep + path
    os.environ["CATLASS_DSL_CACHE"] = "1" if cache else "0"
    # Per-case kernel caches must not override the shared BC cache location.
    os.environ.pop("CATLASS_DSL_CACHE_DIR", None)
    os.environ["CATLASS_DSL_FORCE_RECOMPILE"] = "1" if force_recompile else "0"
    return {
        "arch_scope": "aic.c310",
        "cache": cache,
        "cache_dir": str(Path(cache_dir).expanduser().resolve()),
        "force_recompile": force_recompile,
    }


def _case_cache_dir(base: str | Path, spec: ProblemSpec) -> Path:
    key = (
        f"{spec.uuid}-{spec.mode}-{spec.input_dtype}-bt{spec.chunk_size}-"
        f"d{spec.dim}-scale{spec.scale:g}-tasks{spec.num_tasks}"
    )
    return Path(base).expanduser().resolve() / key


def dump_tlair(spec: ProblemSpec) -> str:
    return KERNEL.dump_mlir(type_args=_type_args(spec))


def build_kernel(args: argparse.Namespace, spec: ProblemSpec) -> int:
    artifact = tla.compile(
        KERNEL,
        *_type_args(spec),
        **_runtime_kwargs(
            _case_cache_dir(args.cache_dir, spec),
            force_recompile=args.force_recompile,
            cache=not args.no_cache,
        ),
    )
    print(
        f"build_ok=True uuid={spec.uuid} mode={spec.mode} "
        f"dtype={spec.input_dtype} bt={spec.chunk_size} tasks={spec.num_tasks} "
        f"kernel=chunk_gla_fwd_kernel_o_mixed_kernel path={artifact.kernel_binary_path}"
    )
    return 0


def _require_torch_npu(device: int) -> Any:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    return torch


def _torch_input_dtype(torch: Any, name: str) -> Any:
    return torch.float16 if name == "float16" else torch.bfloat16


def _random_input(
    torch: Any, shape: tuple[int, ...], dtype: Any, generator: Any
) -> Any:
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)


def make_inputs(
    torch: Any, spec: ProblemSpec, seed: int
) -> tuple[tuple[Any, Any, Any, Any, Any], Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dtype = _torch_input_dtype(torch, spec.input_dtype)
    q = _random_input(torch, spec.q_shape, dtype, generator)
    v = _random_input(torch, spec.v_shape, dtype, generator)
    g = _random_input(torch, spec.g_shape, dtype, generator)
    a = _random_input(torch, spec.a_shape, torch.float32, generator)
    h = _random_input(torch, spec.h_shape, torch.float32, generator)
    cu_seqlens = None
    if spec.cu_seqlens is not None:
        cu_seqlens = torch.tensor(spec.cu_seqlens, dtype=torch.int64)
    return (q, v, g, a, h), cu_seqlens


def _to_bnsd(tensor: Any, mode: str) -> Any:
    if mode == "tnd":
        tensor = tensor.unsqueeze(0)
    return tensor.permute(0, 2, 1, 3).contiguous()


def _from_bnsd(tensor: Any, mode: str) -> Any:
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    return tensor.squeeze(0) if mode == "tnd" else tensor


def _h_to_task_stream(h: Any, spec: ProblemSpec) -> Any:
    return h.reshape(spec.num_chunks, spec.heads, spec.dim, spec.dim)


def _pack_inputs(
    torch: Any, spec: ProblemSpec, q: Any, v: Any, g: Any, a: Any, h: Any
) -> tuple[Any, Any, Any, Any, Any]:
    q_bnsd = _to_bnsd(q, spec.mode)
    g_bnsd = _to_bnsd(g, spec.mode)
    v_bnsd = _to_bnsd(v, spec.mode)
    a_bnsd = _to_bnsd(a, spec.mode)
    a_tasks = a_bnsd.reshape(spec.num_tasks, spec.chunk_size, spec.chunk_size)
    h_task = (
        _h_to_task_stream(h, spec)
        .reshape(spec.num_tasks, spec.dim, spec.dim)
        .contiguous()
    )
    return (
        q_bnsd.reshape(spec.num_tasks, spec.chunk_size, spec.dim).contiguous(),
        g_bnsd.reshape(spec.num_tasks, spec.chunk_size, spec.dim).contiguous(),
        h_task,
        a_tasks.contiguous(),
        v_bnsd.reshape(spec.num_tasks, spec.chunk_size, spec.dim).contiguous(),
    )


def cpu_reference(
    torch: Any, spec: ProblemSpec, q: Any, v: Any, g: Any, a: Any, h: Any
) -> Any:
    q_tasks, g_tasks, h_tasks, a_tasks, v_tasks = _pack_inputs(
        torch, spec, q, v, g, a, h
    )
    qg_tasks = (q_tasks.float() * torch.exp2(g_tasks.float())).to(q.dtype)
    causal = torch.tril(
        torch.ones(
            spec.chunk_size,
            spec.chunk_size,
            dtype=torch.float32,
            device=a_tasks.device,
        )
    )
    a_tasks = (a_tasks * causal).to(q.dtype)
    h_tasks = h_tasks.to(q.dtype)
    state = torch.bmm(qg_tasks.float(), h_tasks.float()) * spec.scale
    intra = torch.bmm(a_tasks.float(), v_tasks.float())
    out_tasks = (state + intra).to(q.dtype)
    out_bnsd = out_tasks.reshape(spec.internal_v_bnsd_shape)
    return _from_bnsd(out_bnsd, spec.mode)


def _to_tla(dev_tensor: Any) -> Any:
    return tla.from_dlpack(
        dev_tensor.reshape(-1).contiguous(), layout_tag=tla.arch.RowMajor
    )


def chunk_gla_fwd_kernel_o(
    q: Any,
    v: Any,
    g: Any,
    A: Any,
    h: Any,
    scale: float,
    state_v_first: bool = False,
    cu_seqlens: Any = None,
    chunk_size: int = 64,
    chunk_indices: Any = None,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_recompile: bool = False,
    cache: bool = True,
    block: int = 28,
    return_perf: bool = False,
) -> Any:
    """Run BSND or aligned TND through one CATLASS mixed kernel launch."""
    import torch

    case = {
        "uuid": "runtime",
        "inputs": {
            "q": {
                "shape": list(q.shape),
                "dtype": "float16" if q.dtype == torch.float16 else "bfloat16",
            },
            "v": {
                "shape": list(v.shape),
                "dtype": "float16" if v.dtype == torch.float16 else "bfloat16",
            },
            "g": {
                "shape": list(g.shape),
                "dtype": "float16" if g.dtype == torch.float16 else "bfloat16",
            },
            "A": {"shape": list(A.shape), "dtype": "float32"},
            "h": {"shape": list(h.shape), "dtype": "float32"},
            "scale": {"value": float(scale)},
            "state_v_first": {"value": bool(state_v_first)},
            "cu_seqlens": None
            if cu_seqlens is None
            else {"value": [int(x) for x in cu_seqlens.detach().cpu().tolist()]},
            "chunk_size": {"value": int(chunk_size)},
            "chunk_indices": None,
        },
    }
    spec = _parse_problem(case)
    if chunk_indices is not None and cu_seqlens is not None:
        actual_indices = tuple(
            (int(row[0]), int(row[1])) for row in chunk_indices.detach().cpu().tolist()
        )
        if set(actual_indices) != set(spec.canonical_chunk_indices):
            raise ValueError("chunk_indices must cover the canonical chunk set")
    q_tasks, g_tasks, h_tasks, a_tasks, v_tasks = _pack_inputs(
        torch, spec, q, v, g, A, h
    )
    h_cast_tmp = torch.empty(h_tasks.shape, dtype=q.dtype, device=h.device)
    qg_tmp = torch.empty_like(q_tasks)
    # BT128 leaves the upper-right 64x64 tile unwritten, but Cube reads it.
    allocate_masked = torch.zeros if spec.chunk_size == 128 else torch.empty
    a_masked_tmp = allocate_masked(a_tasks.shape, dtype=q.dtype, device=A.device)
    state_tmp = torch.empty(v_tasks.shape, dtype=torch.float32, device=v.device)
    intra_tmp = torch.empty_like(state_tmp)
    out_tasks = torch.empty_like(v_tasks)
    real_args = (
        _to_tla(q_tasks),
        _to_tla(g_tasks),
        _to_tla(h_tasks),
        _to_tla(h_cast_tmp),
        _to_tla(a_tasks),
        _to_tla(v_tasks),
        _to_tla(qg_tmp),
        _to_tla(a_masked_tmp),
        _to_tla(state_tmp),
        _to_tla(intra_tmp),
        _to_tla(out_tasks),
    )
    configure_problem(
        spec.num_tasks, spec.chunk_size, spec.dim, spec.input_dtype, spec.scale
    )
    artifact = tla.compile(
        KERNEL,
        *real_args,
        **_runtime_kwargs(
            _case_cache_dir(cache_dir, spec),
            force_recompile=force_recompile,
            cache=cache,
        ),
    )
    from time import perf_counter

    # Task-stream packing is queued asynchronously; exclude it from kernel timing.
    torch.npu.synchronize()
    launch_start = perf_counter()
    artifact(*real_args, block_num=block)
    torch.npu.synchronize()
    kernel_us = (perf_counter() - launch_start) * 1_000_000.0
    output = _from_bnsd(out_tasks.reshape(spec.internal_v_bnsd_shape), spec.mode)
    if return_perf:
        return output, {"catlass_kernel_us": kernel_us}
    return output


def _compare(
    torch: Any,
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float,
    required_matched_ratio: float = 1.0,
    max_error_cap: float = math.inf,
) -> bool:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    # Avoid near-zero expected values dominating the diagnostic relative error.
    rel = diff / expected_f.abs().clamp_min(1.0)
    close = torch.isclose(actual_f, expected_f, rtol=rtol, atol=atol)
    mismatch_count = int((~close).sum())
    matched_ratio = 1.0 - mismatch_count / close.numel()
    max_abs = float(diff.max())
    passed = matched_ratio >= required_matched_ratio and max_abs <= max_error_cap
    mismatch = close.logical_not().nonzero(as_tuple=False)
    first = None
    if mismatch.numel():
        index = tuple(int(x) for x in mismatch[0].tolist())
        first = {
            "index": index,
            "actual": float(actual_f[index]),
            "expected": float(expected_f[index]),
            "abs_diff": float(diff[index]),
        }
    print(
        f"accuracy name=o pass={passed} "
        f"max_abs={max_abs} mean_abs={float(diff.mean())} "
        f"max_rel={float(rel.max())} "
        f"mismatch={mismatch_count}/{close.numel()} "
        f"matched_ratio={matched_ratio:.9f} "
        f"first_mismatch={first}"
    )
    return passed


def run_validation(
    args: argparse.Namespace, uuid: str, *, manage_runtime: bool = True
) -> int:
    case = args.direct_case
    spec = _parse_problem(case)
    try:
        torch = _require_torch_npu(args.device)
        tensors, cu_cpu = make_inputs(torch, spec, args.seed)
        q, v, g, a, h = tensors
        print(
            f"case_start uuid={uuid} mode={spec.mode} dtype={spec.input_dtype} "
            f"bt={spec.chunk_size} tasks={spec.num_tasks} V={spec.v_shape}"
        )
        print(f"cpu_reference_start=True uuid={uuid}")
        o_ref = cpu_reference(torch, spec, q, v, g, a, h)
        print(f"cpu_reference_done=True uuid={uuid}")
        q_npu, v_npu, g_npu, a_npu, h_npu = (tensor.npu() for tensor in tensors)
        cu_npu = cu_cpu.npu() if cu_cpu is not None else None
        if cu_npu is not None:
            generated, canonical = _canonicalize_chunk_indices(
                torch, cu_npu, None, spec.chunk_size
            )
            print(
                f"prepare_chunk_indices count={len(canonical)} "
                f"first={canonical[:3]} last={canonical[-3:]} "
                f"dtype={generated.dtype}"
            )
        o_npu, perf = chunk_gla_fwd_kernel_o(
            q_npu,
            v_npu,
            g_npu,
            a_npu,
            h_npu,
            scale=spec.scale,
            state_v_first=spec.state_v_first,
            cu_seqlens=cu_npu,
            chunk_size=spec.chunk_size,
            chunk_indices=None,
            cache_dir=args.cache_dir,
            force_recompile=args.force_recompile,
            cache=not args.no_cache,
            block=args.block,
            return_perf=True,
        )
        print(f"kernel_time_us={perf['catlass_kernel_us']:.3f}")
        o_actual = o_npu.cpu()
        finite = bool(torch.isfinite(o_actual).all())
        print(
            f"shape_dtype uuid={uuid} o={tuple(o_actual.shape)}/{o_actual.dtype} "
            f"finite={finite}"
        )
        tolerance = case.get("tolerance", {})
        rtol = (
            args.rtol
            if args.rtol is not None
            else float(tolerance.get("max_rtol", 2e-2))
        )
        atol = (
            args.atol
            if args.atol is not None
            else float(tolerance.get("max_atol", 2e-2))
        )
        o_ok = _compare(
            torch,
            o_actual,
            o_ref,
            rtol=rtol,
            atol=atol,
            required_matched_ratio=float(tolerance.get("required_matched_ratio", 1.0)),
            max_error_cap=float(tolerance.get("max_error_cap", math.inf)),
        )
        passed = finite and o_ok
        print(f"validation uuid={uuid} pass={passed}")
        return 0 if passed else 1
    finally:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate chunk_gla_fwd_kernel_o from B T H D [deviceId]."
    )
    parser.add_argument("batch", type=int, help="BSND batch size; use 1 for TND")
    parser.add_argument("sequence_length", type=int, help="sequence length T")
    parser.add_argument("heads", type=int, help="number of heads H")
    parser.add_argument("dim", type=int, choices=(64, 128), help="feature dimension D")
    parser.add_argument(
        "device_pos", nargs="?", type=int, default=0, help="NPU device ID (default: 0)"
    )
    parser.add_argument("--mode", choices=("bsnd", "tnd"), default="bsnd")
    parser.add_argument("--chunk-size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--scale", type=float, default=0.125)
    parser.add_argument("--cu-seqlens", help="TND cumulative lengths, comma-separated")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dump-tlair", action="store_true")
    mode.add_argument("--build-only", action="store_true")
    parser.add_argument("--block", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--rtol", type=float, default=0.04)
    parser.add_argument("--atol", type=float, default=0.04)
    return parser


def _direct_case(args: argparse.Namespace) -> dict[str, Any]:
    dims = (args.batch, args.sequence_length, args.heads, args.dim)
    if any(value is None for value in dims) or any(value <= 0 for value in dims):
        raise ValueError("provide positive dimensions: B T H D [deviceId]")
    if args.device_pos < 0:
        raise ValueError("deviceId must be nonnegative")
    if args.mode == "bsnd":
        if args.cu_seqlens is not None:
            raise ValueError("--cu-seqlens requires --mode tnd")
        shape = [args.batch, args.sequence_length, args.heads, args.dim]
        chunks = args.batch * args.sequence_length // args.chunk_size
        cu = None
    else:
        if args.batch != 1:
            raise ValueError("TND mode requires B=1; T is the total token count")
        if args.cu_seqlens is None:
            raise ValueError("TND mode requires --cu-seqlens, for example 0,64,128")
        shape = [args.sequence_length, args.heads, args.dim]
        chunks = args.sequence_length // args.chunk_size
        cu = {"value": [int(value) for value in args.cu_seqlens.split(",")]}
    inputs = {name: {"shape": shape, "dtype": args.dtype} for name in ("q", "v", "g")}
    inputs.update(
        {
            "A": {"shape": shape[:-1] + [args.chunk_size], "dtype": "float32"},
            "h": {
                "shape": [chunks, args.heads, args.dim, args.dim],
                "dtype": "float32",
            },
            "scale": {"value": args.scale},
            "state_v_first": {"value": False},
            "cu_seqlens": cu,
            "chunk_size": {"value": args.chunk_size},
            "chunk_indices": None,
        }
    )
    case = {
        "uuid": f"direct-{args.mode}-{args.dtype}-bt{args.chunk_size}",
        "inputs": inputs,
    }
    args.device = args.device_pos
    args.direct_case = case
    return case


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    case = _direct_case(args)
    spec = _parse_problem(case)
    if args.dump_tlair:
        print(dump_tlair(spec))
        return 0
    if args.build_only:
        return build_kernel(args, spec)
    return run_validation(args, case["uuid"])


if __name__ == "__main__":
    raise SystemExit(main())
