# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Phase-1 SiLU epilogue helpers (RegSiluB16-style VL).
from __future__ import annotations

import catlass.tla as tla

REG_M = 1

# f32 UB-acc → dtype_gm_c; CAST_RINT + DIST_PACK_B32 matches RegSiluB16-style pack.
_CAST_ACC_TO_OUT = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
_PACK_STORE = tla.params.NormalStoreParams(
    store_dist=tla.params.StoreDist.DIST_PACK_B32
)


@tla.jit
def silu_apply_vl_1d(
    acc_1d,
    out_1d,
    n_elems,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
) -> None:
    """SiLU over a contiguous 1D f32 acc view → out (f16 pack or f32)."""
    c0 = 0
    c1 = 1
    n_vl_full = n_elems // SIMD_LANES
    n_vl = (n_elems + SIMD_LANES - 1) // SIMD_LANES
    aligned = (n_elems % SIMD_LANES) == c0
    with tla.vec.func(mode="simd"):
        if aligned:
            mask_f = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            # PACK store needs f16 ALL; omitting regenerates plt.b16 every VL.
            mask_h = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
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
                den = 1.0 + tla.exp(tla.neg(xv, mask=mask_f), mask=mask_f)
                yv = tla.div(xv, den, mask=mask_f)
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
                result_chunk = tla.tile_view(
                    out_1d,
                    tla.make_shape(SIMD_LANES),
                    tla.make_coord(vl_i),
                )
                tail, remaining = tla.update_mask(remaining, dtype=tla.Float32)
                xv = c_chunk.load()
                den = 1.0 + tla.exp(tla.neg(xv, mask=tail), mask=tail)
                yv = tla.div(xv, den, mask=tail)
                if tla.const_expr(dtype_gm_c != tla.Float32):
                    yv16 = yv.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=tail)
                    result_chunk.store(yv16, _PACK_STORE)
                else:
                    result_chunk.store(yv, mask=tail)


@tla.jit
def silu_apply_vl_2d(
    ub_c_tile,
    ub_result,
    tile_rows,
    tile_cols,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
) -> None:
    """SiLU over a 2D row×col tile (non-contiguous / strip path)."""
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
                den = 1.0 + tla.exp(tla.neg(xv, mask=tail), mask=tail)
                yv = tla.div(xv, den, mask=tail)
                if tla.const_expr(dtype_gm_c != tla.Float32):
                    yv16 = yv.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=tail)
                    result_chunk.store(yv16, _PACK_STORE)
                else:
                    result_chunk.store(yv, mask=tail)
