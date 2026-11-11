# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Phase-1 S2: GELU (erf) epilogue — Padé aligned with CPP fusion_regbase_act.h
from __future__ import annotations

import catlass.tla as tla

REG_M = 1


def _gelu_dual_enabled() -> bool:
    """CPP RegGeluErfB16 always pairs (Chunk2)."""
    return True


# CPP GeluErfChunkB16 coefficients (fusion_regbase_act.h).
_P0 = 0.29639384698e5
_P1 = 0.50637915060e4
_P2 = 0.13938061484e4
_P3 = 0.10162808918e3
_P4 = 0.75517016694e1
_P5 = 0.053443748819
_Q0 = 0.26267224157e5
_Q1 = 0.13243365831e5
_Q2 = 0.30231248150e4
_Q3 = 0.39856963806e3
_Q4 = 0.31212858877e2
_INV_SQRT2 = 0.70710678118654752
_CLIP = 3.92

_CAST_ACC_TO_OUT = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
_PACK_STORE = tla.params.NormalStoreParams(
    store_dist=tla.params.StoreDist.DIST_PACK_B32
)


def _gelu_erf_reg(xv, tail):
    """Scalar-register GELU(erf-Padé) on a masked SIMD chunk."""
    t = tla.mul(xv, _INV_SQRT2, mask=tail)
    clip = tla.max(tla.min(t, _CLIP, mask=tail), -_CLIP, mask=tail)
    x2 = tla.mul(clip, clip, mask=tail)
    pq = tla.mul(x2, _P5, mask=tail)
    pq = tla.add(pq, _P4, mask=tail)
    pq = tla.mul(pq, x2, mask=tail)
    pq = tla.add(pq, _P3, mask=tail)
    pq = tla.mul(pq, x2, mask=tail)
    pq = tla.add(pq, _P2, mask=tail)
    pq = tla.mul(pq, x2, mask=tail)
    pq = tla.add(pq, _P1, mask=tail)
    pq = tla.mul(pq, x2, mask=tail)
    pq = tla.add(pq, _P0, mask=tail)
    pq = tla.mul(pq, clip, mask=tail)
    y = tla.add(x2, _Q4, mask=tail)
    y = tla.mul(y, x2, mask=tail)
    y = tla.add(y, _Q3, mask=tail)
    y = tla.mul(y, x2, mask=tail)
    y = tla.add(y, _Q2, mask=tail)
    y = tla.mul(y, x2, mask=tail)
    y = tla.add(y, _Q1, mask=tail)
    y = tla.mul(y, x2, mask=tail)
    y = tla.add(y, _Q0, mask=tail)
    erf_t = tla.div(pq, y, mask=tail)
    y = tla.add(erf_t, 1.0, mask=tail)
    y = tla.mul(y, 0.5, mask=tail)
    return tla.mul(y, xv, mask=tail)


def _gelu_erf_reg_dual(xv0, xv1, mask):
    """Two independent Padé chains, ops interleaved (CPP GeluErfChunk2B16)."""
    t0 = tla.mul(xv0, _INV_SQRT2, mask=mask)
    t1 = tla.mul(xv1, _INV_SQRT2, mask=mask)
    clip0 = tla.max(tla.min(t0, _CLIP, mask=mask), -_CLIP, mask=mask)
    clip1 = tla.max(tla.min(t1, _CLIP, mask=mask), -_CLIP, mask=mask)
    x20 = tla.mul(clip0, clip0, mask=mask)
    x21 = tla.mul(clip1, clip1, mask=mask)
    pq0 = tla.mul(x20, _P5, mask=mask)
    pq1 = tla.mul(x21, _P5, mask=mask)
    pq0 = tla.add(pq0, _P4, mask=mask)
    pq1 = tla.add(pq1, _P4, mask=mask)
    pq0 = tla.mul(pq0, x20, mask=mask)
    pq1 = tla.mul(pq1, x21, mask=mask)
    pq0 = tla.add(pq0, _P3, mask=mask)
    pq1 = tla.add(pq1, _P3, mask=mask)
    pq0 = tla.mul(pq0, x20, mask=mask)
    pq1 = tla.mul(pq1, x21, mask=mask)
    pq0 = tla.add(pq0, _P2, mask=mask)
    pq1 = tla.add(pq1, _P2, mask=mask)
    pq0 = tla.mul(pq0, x20, mask=mask)
    pq1 = tla.mul(pq1, x21, mask=mask)
    pq0 = tla.add(pq0, _P1, mask=mask)
    pq1 = tla.add(pq1, _P1, mask=mask)
    pq0 = tla.mul(pq0, x20, mask=mask)
    pq1 = tla.mul(pq1, x21, mask=mask)
    pq0 = tla.add(pq0, _P0, mask=mask)
    pq1 = tla.add(pq1, _P0, mask=mask)
    pq0 = tla.mul(pq0, clip0, mask=mask)
    pq1 = tla.mul(pq1, clip1, mask=mask)
    y0 = tla.add(x20, _Q4, mask=mask)
    y1 = tla.add(x21, _Q4, mask=mask)
    y0 = tla.mul(y0, x20, mask=mask)
    y1 = tla.mul(y1, x21, mask=mask)
    y0 = tla.add(y0, _Q3, mask=mask)
    y1 = tla.add(y1, _Q3, mask=mask)
    y0 = tla.mul(y0, x20, mask=mask)
    y1 = tla.mul(y1, x21, mask=mask)
    y0 = tla.add(y0, _Q2, mask=mask)
    y1 = tla.add(y1, _Q2, mask=mask)
    y0 = tla.mul(y0, x20, mask=mask)
    y1 = tla.mul(y1, x21, mask=mask)
    y0 = tla.add(y0, _Q1, mask=mask)
    y1 = tla.add(y1, _Q1, mask=mask)
    y0 = tla.mul(y0, x20, mask=mask)
    y1 = tla.mul(y1, x21, mask=mask)
    y0 = tla.add(y0, _Q0, mask=mask)
    y1 = tla.add(y1, _Q0, mask=mask)
    erf0 = tla.div(pq0, y0, mask=mask)
    erf1 = tla.div(pq1, y1, mask=mask)
    y0 = tla.add(erf0, 1.0, mask=mask)
    y1 = tla.add(erf1, 1.0, mask=mask)
    y0 = tla.mul(y0, 0.5, mask=mask)
    y1 = tla.mul(y1, 0.5, mask=mask)
    return tla.mul(y0, xv0, mask=mask), tla.mul(y1, xv1, mask=mask)


@tla.jit
def gelu_apply_vl_1d(
    acc_1d,
    out_1d,
    n_elems,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
) -> None:
    """GELU(erf-Padé) over contiguous 1D f32 acc → out.

    PACK store must pass an f16 ALL mask on the aligned hot path; omitting it
    regenerates ``plt.b16`` every VL (same tax as SiLU/SwiGLU before mask_h).
    """
    c0 = 0
    c1 = 1
    n_vl_full = n_elems // SIMD_LANES
    n_vl = (n_elems + SIMD_LANES - 1) // SIMD_LANES
    aligned = (n_elems % SIMD_LANES) == c0
    with tla.vec.func(mode="simd"):
        if aligned:
            mask_f = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            mask_h = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            use_dual = tla.const_expr(_gelu_dual_enabled()) and ((n_vl_full % 2) == c0)
            if use_dual:
                for vl_i in tla.range(c0, n_vl_full, 2):
                    c0v = tla.tile_view(
                        acc_1d, tla.make_shape(SIMD_LANES), tla.make_coord(vl_i)
                    )
                    c1v = tla.tile_view(
                        acc_1d,
                        tla.make_shape(SIMD_LANES),
                        tla.make_coord(vl_i + c1),
                    )
                    o0v = tla.tile_view(
                        out_1d, tla.make_shape(SIMD_LANES), tla.make_coord(vl_i)
                    )
                    o1v = tla.tile_view(
                        out_1d,
                        tla.make_shape(SIMD_LANES),
                        tla.make_coord(vl_i + c1),
                    )
                    xv0 = c0v.load()
                    xv1 = c1v.load()
                    y0, y1 = _gelu_erf_reg_dual(xv0, xv1, mask_f)
                    if tla.const_expr(dtype_gm_c != tla.Float32):
                        y0_16 = y0.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=mask_f)
                        y1_16 = y1.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=mask_f)
                        o0v.store(y0_16, _PACK_STORE, mask=mask_h)
                        o1v.store(y1_16, _PACK_STORE, mask=mask_h)
                    else:
                        o0v.store(y0, mask=mask_f)
                        o1v.store(y1, mask=mask_f)
            else:
                for vl_i in tla.range(c0, n_vl_full, c1):
                    c_chunk = tla.tile_view(
                        acc_1d, tla.make_shape(SIMD_LANES), tla.make_coord(vl_i)
                    )
                    result_chunk = tla.tile_view(
                        out_1d, tla.make_shape(SIMD_LANES), tla.make_coord(vl_i)
                    )
                    xv = c_chunk.load()
                    yv = _gelu_erf_reg(xv, mask_f)
                    if tla.const_expr(dtype_gm_c != tla.Float32):
                        yv16 = yv.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=mask_f)
                        result_chunk.store(yv16, _PACK_STORE, mask=mask_h)
                    else:
                        result_chunk.store(yv, mask=mask_f)
        else:
            remaining = n_elems
            for vl_i in tla.range(c0, n_vl, c1):
                c_chunk = tla.tile_view(
                    acc_1d, tla.make_shape(SIMD_LANES), tla.make_coord(vl_i)
                )
                result_chunk = tla.tile_view(
                    out_1d, tla.make_shape(SIMD_LANES), tla.make_coord(vl_i)
                )
                # Capture f32 lane count for this VL before update_mask consumes it.
                rem_before = remaining
                tail, remaining = tla.update_mask(remaining, dtype=tla.Float32)
                xv = c_chunk.load()
                yv = _gelu_erf_reg(xv, tail)
                if tla.const_expr(dtype_gm_c != tla.Float32):
                    yv16 = yv.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=tail)
                    # f16 PACK needs 2× f32 lanes (see SwiGLU rem pack fix).
                    rem_h, _ = tla.update_mask(rem_before * 2, dtype=tla.Float16)
                    result_chunk.store(yv16, _PACK_STORE, mask=rem_h)
                else:
                    result_chunk.store(yv, mask=tail)


@tla.jit
def gelu_apply_vl_2d(
    ub_c_tile,
    ub_result,
    tile_rows,
    tile_cols,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
) -> None:
    """GELU(erf-Padé) over a 2D row×col tile."""
    c0 = 0
    c1 = 1
    n_simd_strips = (tile_cols + SIMD_LANES - 1) // SIMD_LANES
    aligned = (tile_cols % SIMD_LANES) == c0
    with tla.vec.func(mode="simd"):
        if aligned:
            mask_f = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            mask_h = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            use_dual = tla.const_expr(_gelu_dual_enabled()) and (
                (n_simd_strips % 2) == c0
            )
            if use_dual:
                for row_i in tla.range(c0, tile_rows, c1):
                    for col_j in tla.range(c0, n_simd_strips, 2):
                        c0v = tla.tile_view(
                            ub_c_tile,
                            tla.make_shape(REG_M, SIMD_LANES),
                            tla.make_coord(row_i, col_j),
                        )
                        c1v = tla.tile_view(
                            ub_c_tile,
                            tla.make_shape(REG_M, SIMD_LANES),
                            tla.make_coord(row_i, col_j + c1),
                        )
                        o0v = tla.tile_view(
                            ub_result,
                            tla.make_shape(REG_M, SIMD_LANES),
                            tla.make_coord(row_i, col_j),
                        )
                        o1v = tla.tile_view(
                            ub_result,
                            tla.make_shape(REG_M, SIMD_LANES),
                            tla.make_coord(row_i, col_j + c1),
                        )
                        xv0 = c0v.load()
                        xv1 = c1v.load()
                        y0, y1 = _gelu_erf_reg_dual(xv0, xv1, mask_f)
                        if tla.const_expr(dtype_gm_c != tla.Float32):
                            y0_16 = y0.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=mask_f)
                            y1_16 = y1.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=mask_f)
                            o0v.store(y0_16, _PACK_STORE, mask=mask_h)
                            o1v.store(y1_16, _PACK_STORE, mask=mask_h)
                        else:
                            o0v.store(y0, mask=mask_f)
                            o1v.store(y1, mask=mask_f)
            else:
                for row_i in tla.range(c0, tile_rows, c1):
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
                        xv = c_chunk.load()
                        yv = _gelu_erf_reg(xv, mask_f)
                        if tla.const_expr(dtype_gm_c != tla.Float32):
                            yv16 = yv.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=mask_f)
                            result_chunk.store(yv16, _PACK_STORE, mask=mask_h)
                        else:
                            result_chunk.store(yv, mask=mask_f)
        else:
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
                    rem_before = remaining
                    tail, remaining = tla.update_mask(remaining, dtype=tla.Float32)
                    xv = c_chunk.load()
                    yv = _gelu_erf_reg(xv, tail)
                    if tla.const_expr(dtype_gm_c != tla.Float32):
                        yv16 = yv.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=tail)
                        rem_h, _ = tla.update_mask(rem_before * 2, dtype=tla.Float16)
                        result_chunk.store(yv16, _PACK_STORE, mask=rem_h)
                    else:
                        result_chunk.store(yv, mask=tail)
