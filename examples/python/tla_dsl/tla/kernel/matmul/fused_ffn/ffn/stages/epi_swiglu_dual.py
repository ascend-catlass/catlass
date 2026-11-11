# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# SwiGLU dual epilogue: RAW FP32 store (gate) + silu(gate)*up (CPP FusionSwiglu).
from __future__ import annotations

import catlass.tla as tla

REG_M = 1

_CAST_ACC_TO_OUT = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
_PACK_STORE = tla.params.NormalStoreParams(
    store_dist=tla.params.StoreDist.DIST_PACK_B32
)


@tla.jit
def raw_store_vl_1d(
    acc_1d,
    out_1d,
    n_elems,
    SIMD_LANES: tla.Constexpr[int],
) -> None:
    """Copy contiguous f32 acc → f32 GM/out (no activation)."""
    c0 = 0
    c1 = 1
    n_vl_full = n_elems // SIMD_LANES
    n_vl = (n_elems + SIMD_LANES - 1) // SIMD_LANES
    aligned = (n_elems % SIMD_LANES) == c0
    with tla.vec.func(mode="simd"):
        if aligned:
            mask_f = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            for vl_i in tla.range(c0, n_vl_full, c1):
                c_chunk = tla.tile_view(
                    acc_1d,
                    tla.make_shape(SIMD_LANES),
                    tla.make_coord(vl_i),
                )
                result_chunk = tla.tile_view(
                    out_1d,
                    tla.make_shape(SIMD_LANES),
                    tla.make_coord(vl_i),
                )
                xv = c_chunk.load()
                result_chunk.store(xv, mask=mask_f)
        else:
            remaining = n_elems
            for vl_i in tla.range(c0, n_vl, c1):
                c_chunk = tla.tile_view(
                    acc_1d,
                    tla.make_shape(SIMD_LANES),
                    tla.make_coord(vl_i),
                )
                result_chunk = tla.tile_view(
                    out_1d,
                    tla.make_shape(SIMD_LANES),
                    tla.make_coord(vl_i),
                )
                tail, remaining = tla.update_mask(remaining, dtype=tla.Float32)
                xv = c_chunk.load()
                result_chunk.store(xv, mask=tail)


@tla.jit
def raw_store_vl_2d(
    ub_c_tile,
    ub_result,
    tile_rows,
    tile_cols,
    SIMD_LANES: tla.Constexpr[int],
) -> None:
    """Copy pitched f32 acc tile → f32 out."""
    c0 = 0
    c1 = 1
    n_simd_strips = (tile_cols + SIMD_LANES - 1) // SIMD_LANES
    with tla.vec.func(mode="simd"):
        for row_i in tla.range(c0, tile_rows, c1):
            remaining = tile_cols
            for col_j in tla.range(c0, n_simd_strips, c1):
                c_chunk = tla.tile_view(
                    ub_c_tile,
                    tla.make_shape(REG_M, SIMD_LANES),
                    tla.make_coord(row_i, col_j),
                )
                result_chunk = tla.tile_view(
                    ub_result,
                    tla.make_shape(REG_M, SIMD_LANES),
                    tla.make_coord(row_i, col_j),
                )
                tail, remaining = tla.update_mask(remaining, dtype=tla.Float32)
                xv = c_chunk.load()
                result_chunk.store(xv, mask=tail)


@tla.jit
def dual_silu_inplace_vl_1d(
    gate_1d,
    n_elems,
    SIMD_LANES: tla.Constexpr[int],
) -> None:
    """Pass-1: ``gate = silu(gate)`` in one UB buffer (alone in ``vec.func``)."""
    c0 = 0
    c1 = 1
    n_vl_full = n_elems // SIMD_LANES
    n_vl = (n_elems + SIMD_LANES - 1) // SIMD_LANES
    aligned = (n_elems % SIMD_LANES) == c0
    with tla.vec.func(mode="simd"):
        if aligned:
            mask_f = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            for vl_i in tla.range(c0, n_vl_full, c1):
                g_chunk = tla.tile_view(
                    gate_1d,
                    tla.make_shape(SIMD_LANES),
                    tla.make_coord(vl_i),
                )
                gv = g_chunk.load()
                den = 1.0 + tla.exp(tla.neg(gv, mask=mask_f), mask=mask_f)
                sv = tla.div(gv, den, mask=mask_f)
                g_chunk.store(sv, mask=mask_f)
        else:
            remaining = n_elems
            for vl_i in tla.range(c0, n_vl, c1):
                g_chunk = tla.tile_view(
                    gate_1d,
                    tla.make_shape(SIMD_LANES),
                    tla.make_coord(vl_i),
                )
                tail, remaining = tla.update_mask(remaining, dtype=tla.Float32)
                gv = g_chunk.load()
                den = 1.0 + tla.exp(tla.neg(gv, mask=tail), mask=tail)
                sv = tla.div(gv, den, mask=tail)
                g_chunk.store(sv, mask=tail)


@tla.jit
def dual_mul_pack_vl_1d(
    acc_1d,
    gate_1d,
    out_1d,
    n_elems,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
) -> None:
    """Pass-2: ``out = silu_gate * up`` (gate already silu'd)."""
    c0 = 0
    c1 = 1
    n_vl_full = n_elems // SIMD_LANES
    n_vl = (n_elems + SIMD_LANES - 1) // SIMD_LANES
    aligned = (n_elems % SIMD_LANES) == c0
    with tla.vec.func(mode="simd"):
        if aligned:
            mask_f = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            mask_h = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            for vl_i in tla.range(c0, n_vl_full, c1):
                c_chunk = tla.tile_view(
                    acc_1d,
                    tla.make_shape(SIMD_LANES),
                    tla.make_coord(vl_i),
                )
                g_chunk = tla.tile_view(
                    gate_1d,
                    tla.make_shape(SIMD_LANES),
                    tla.make_coord(vl_i),
                )
                result_chunk = tla.tile_view(
                    out_1d,
                    tla.make_shape(SIMD_LANES),
                    tla.make_coord(vl_i),
                )
                sv = g_chunk.load()
                uv = c_chunk.load()
                yv = tla.mul(sv, uv, mask=mask_f)
                if tla.const_expr(dtype_gm_c != tla.Float32):
                    yv16 = yv.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=mask_f)
                    result_chunk.store(yv16, _PACK_STORE, mask=mask_h)
                else:
                    result_chunk.store(yv, mask=mask_f)
        else:
            remaining = n_elems
            for vl_i in tla.range(c0, n_vl, c1):
                c_chunk = tla.tile_view(
                    acc_1d,
                    tla.make_shape(SIMD_LANES),
                    tla.make_coord(vl_i),
                )
                g_chunk = tla.tile_view(
                    gate_1d,
                    tla.make_shape(SIMD_LANES),
                    tla.make_coord(vl_i),
                )
                result_chunk = tla.tile_view(
                    out_1d,
                    tla.make_shape(SIMD_LANES),
                    tla.make_coord(vl_i),
                )
                tail, remaining = tla.update_mask(remaining, dtype=tla.Float32)
                sv = g_chunk.load()
                uv = c_chunk.load()
                yv = tla.mul(sv, uv, mask=tail)
                if tla.const_expr(dtype_gm_c != tla.Float32):
                    yv16 = yv.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=tail)
                    result_chunk.store(yv16, _PACK_STORE)
                else:
                    result_chunk.store(yv, mask=tail)


@tla.jit
def swiglu_dual_apply_vl_1d(
    acc_1d,
    gate_1d,
    out_1d,
    n_elems,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
) -> None:
    """``out = silu(gate) * up`` — two ``vec.func`` to avoid UB helper alias."""
    dual_silu_inplace_vl_1d(gate_1d, n_elems, SIMD_LANES)
    dual_mul_pack_vl_1d(acc_1d, gate_1d, out_1d, n_elems, SIMD_LANES, dtype_gm_c)


@tla.jit
def dual_silu_inplace_vl_2d(
    ub_gate_tile,
    tile_rows,
    tile_cols,
    SIMD_LANES: tla.Constexpr[int],
) -> None:
    """Pass-1 pitched: ``gate = silu(gate)``."""
    c0 = 0
    c1 = 1
    n_simd_strips = (tile_cols + SIMD_LANES - 1) // SIMD_LANES
    with tla.vec.func(mode="simd"):
        for row_i in tla.range(c0, tile_rows, c1):
            remaining = tile_cols
            for col_j in tla.range(c0, n_simd_strips, c1):
                g_chunk = tla.tile_view(
                    ub_gate_tile,
                    tla.make_shape(REG_M, SIMD_LANES),
                    tla.make_coord(row_i, col_j),
                )
                tail, remaining = tla.update_mask(remaining, dtype=tla.Float32)
                gv = g_chunk.load()
                den = 1.0 + tla.exp(tla.neg(gv, mask=tail), mask=tail)
                sv = tla.div(gv, den, mask=tail)
                g_chunk.store(sv, mask=tail)


@tla.jit
def dual_mul_pack_vl_2d(
    ub_c_tile,
    ub_gate_tile,
    ub_result,
    tile_rows,
    tile_cols,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
) -> None:
    """Pass-2 pitched: ``out = silu_gate * up``."""
    c0 = 0
    c1 = 1
    n_simd_strips = (tile_cols + SIMD_LANES - 1) // SIMD_LANES
    with tla.vec.func(mode="simd"):
        for row_i in tla.range(c0, tile_rows, c1):
            remaining = tile_cols
            for col_j in tla.range(c0, n_simd_strips, c1):
                c_chunk = tla.tile_view(
                    ub_c_tile,
                    tla.make_shape(REG_M, SIMD_LANES),
                    tla.make_coord(row_i, col_j),
                )
                g_chunk = tla.tile_view(
                    ub_gate_tile,
                    tla.make_shape(REG_M, SIMD_LANES),
                    tla.make_coord(row_i, col_j),
                )
                result_chunk = tla.tile_view(
                    ub_result,
                    tla.make_shape(REG_M, SIMD_LANES),
                    tla.make_coord(row_i, col_j),
                )
                tail, remaining = tla.update_mask(remaining, dtype=tla.Float32)
                sv = g_chunk.load()
                uv = c_chunk.load()
                yv = tla.mul(sv, uv, mask=tail)
                if tla.const_expr(dtype_gm_c != tla.Float32):
                    yv16 = yv.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=tail)
                    result_chunk.store(yv16, _PACK_STORE)
                else:
                    result_chunk.store(yv, mask=tail)


@tla.jit
def swiglu_dual_apply_vl_2d(
    ub_c_tile,
    ub_gate_tile,
    ub_result,
    tile_rows,
    tile_cols,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
) -> None:
    """``out = silu(gate) * up`` over pitched 2D tiles (two ``vec.func``)."""
    dual_silu_inplace_vl_2d(ub_gate_tile, tile_rows, tile_cols, SIMD_LANES)
    dual_mul_pack_vl_2d(
        ub_c_tile,
        ub_gate_tile,
        ub_result,
        tile_rows,
        tile_cols,
        SIMD_LANES,
        dtype_gm_c,
    )
