# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Phase-1 S4: SwiGLU single epilogue — row layout [g|u], y = silu(g)*u (half width).
from __future__ import annotations

import catlass.tla as tla

_CAST_ACC_TO_OUT = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
_PACK_STORE = tla.params.NormalStoreParams(
    store_dist=tla.params.StoreDist.DIST_PACK_B32
)


def _store_pack_b16(tile, yv, mask_f, mask_h, dtype_gm_c) -> None:
    """f32→b16 PACK store with an explicit f16 ALL mask.

    Without ``mask=`` on the PACK store, TLA lowering invents a per-VL
    ``plt.b16`` from dynamic remaining (camodel: ~3.5 PLT/VEXP + heavy
    RV_SXOR/SSUB/SCMP). CPP ``StoreAlign`` reuses the VL mask; pass
    ``mask_h`` (``create_mask(ALL, f16)``) so the hot path keeps ``pge`` only.
    """
    if tla.const_expr(dtype_gm_c != tla.Float32):
        tile.store(
            yv.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=mask_f),
            _PACK_STORE,
            mask=mask_h,
        )
    else:
        tile.store(yv, mask=mask_f)


@tla.jit
def swiglu_apply_vl_1d(
    acc_1d,
    out_1d,
    rows,
    cols_padded,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
    USE_ALL_MASK: tla.Constexpr[bool] = True,
    TAIL_LANES: tla.Constexpr[int] = 0,
    HOIST_N_VL: tla.Constexpr[int] = 0,
    BAKE_TN: tla.Constexpr[int] = 0,
) -> None:
    """Flat row-major acc ``rows*cols_padded`` with per-row [g|u]; out is half-wide."""
    _ = (HOIST_N_VL, BAKE_TN)  # call-site ABI retained; hoist experiment removed
    c0 = 0
    c1 = 1
    c2 = 2
    half_w = cols_padded // 2
    half_vl = half_w // SIMD_LANES
    row_vl = cols_padded // SIMD_LANES
    n_vl_tail = (half_w + SIMD_LANES - 1) // SIMD_LANES
    use_hybrid_tail = (not USE_ALL_MASK) and (TAIL_LANES > 0)
    with tla.vec.func(mode="simd"):
        # Constexpr-only: do not OR with runtime ``aligned`` (dual-path VF).
        if tla.const_expr(USE_ALL_MASK):
            mask_f = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            mask_h = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            # CPP RegSwigluSingleB16 shape: one VL chain, reuse temps.
            for r in tla.range(c0, rows, c1):
                base = r * row_vl
                out_base = r * half_vl
                for i in tla.range(c0, half_vl, c1):
                    g_chunk = tla.tile_view(
                        acc_1d,
                        tla.make_shape(SIMD_LANES),
                        tla.make_coord(base + i),
                    )
                    u_chunk = tla.tile_view(
                        acc_1d,
                        tla.make_shape(SIMD_LANES),
                        tla.make_coord(base + half_vl + i),
                    )
                    o_chunk = tla.tile_view(
                        out_1d,
                        tla.make_shape(SIMD_LANES),
                        tla.make_coord(out_base + i),
                    )
                    gv = g_chunk.load()
                    uv = u_chunk.load()
                    t = tla.mul(gv, -1.0, mask=mask_f)
                    t = tla.exp(t, mask=mask_f)
                    t = tla.add(t, 1.0, mask=mask_f)
                    t = tla.div(gv, t, mask=mask_f)
                    yv = tla.mul(t, uv, mask=mask_f)
                    _store_pack_b16(o_chunk, yv, mask_f, mask_h, dtype_gm_c)
        elif tla.const_expr(use_hybrid_tail):
            # half_w chunk views (coord r*2 / r*2+1): safe when cols%SIMD!=0
            # (flat ``row_vl=cols//SIMD`` truncates and mis-aligns up for half=80).
            # Body: ALL(pge); one UpdateMask(plt) tail. half_vl may be 0
            # (half=48, SIMD=64 → single UpdateMask(48), matches CPP loops=1).
            # TAIL_LANES is bake-time. Do NOT gate on runtime rem (nan in micro).
            mask_f = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            mask_h = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            for r in tla.range(c0, rows, c1):
                g_row = tla.tile_view(
                    acc_1d, tla.make_shape(half_w), tla.make_coord(r * c2)
                )
                u_row = tla.tile_view(
                    acc_1d, tla.make_shape(half_w), tla.make_coord(r * c2 + c1)
                )
                o_row = tla.tile_view(out_1d, tla.make_shape(half_w), tla.make_coord(r))
                for vl_i in tla.range(c0, half_vl, c1):
                    g_chunk = tla.tile_view(
                        g_row, tla.make_shape(SIMD_LANES), tla.make_coord(vl_i)
                    )
                    u_chunk = tla.tile_view(
                        u_row, tla.make_shape(SIMD_LANES), tla.make_coord(vl_i)
                    )
                    o_chunk = tla.tile_view(
                        o_row, tla.make_shape(SIMD_LANES), tla.make_coord(vl_i)
                    )
                    gv = g_chunk.load()
                    uv = u_chunk.load()
                    t = tla.mul(gv, -1.0, mask=mask_f)
                    t = tla.exp(t, mask=mask_f)
                    t = tla.add(t, 1.0, mask=mask_f)
                    t = tla.div(gv, t, mask=mask_f)
                    yv = tla.mul(t, uv, mask=mask_f)
                    _store_pack_b16(o_chunk, yv, mask_f, mask_h, dtype_gm_c)
                g_tail = tla.tile_view(
                    g_row, tla.make_shape(SIMD_LANES), tla.make_coord(half_vl)
                )
                u_tail = tla.tile_view(
                    u_row, tla.make_shape(SIMD_LANES), tla.make_coord(half_vl)
                )
                o_tail = tla.tile_view(
                    o_row, tla.make_shape(SIMD_LANES), tla.make_coord(half_vl)
                )
                rem_t = TAIL_LANES
                tail_t, rem_t = tla.update_mask(rem_t, dtype=tla.Float32)
                gv_t = g_tail.load()
                uv_t = u_tail.load()
                t_t = tla.mul(gv_t, -1.0, mask=tail_t)
                t_t = tla.exp(t_t, mask=tail_t)
                t_t = tla.add(t_t, 1.0, mask=tail_t)
                t_t = tla.div(gv_t, t_t, mask=tail_t)
                yv_t = tla.mul(t_t, uv_t, mask=tail_t)
                if tla.const_expr(dtype_gm_c != tla.Float32):
                    # TLA PACK needs f16 mask (128 lanes). f16 VL is 2× f32 —
                    # use 2*TAIL so all packed lanes are enabled.
                    rem_h = TAIL_LANES * 2
                    tail_h, rem_h = tla.update_mask(rem_h, dtype=tla.Float16)
                    o_tail.store(
                        yv_t.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=tail_t),
                        _PACK_STORE,
                        mask=tail_h,
                    )
                else:
                    o_tail.store(yv_t, mask=tail_t)
        else:
            # Pure UpdateMask per row.
            # TLA PACK requires f16 mask — cannot reuse float mask like CPP.
            for r in tla.range(c0, rows, c1):
                remaining = half_w
                remaining_h = half_w * 2
                g_row = tla.tile_view(
                    acc_1d, tla.make_shape(half_w), tla.make_coord(r * c2)
                )
                u_row = tla.tile_view(
                    acc_1d, tla.make_shape(half_w), tla.make_coord(r * c2 + c1)
                )
                o_row = tla.tile_view(out_1d, tla.make_shape(half_w), tla.make_coord(r))
                for vl_i in tla.range(c0, n_vl_tail, c1):
                    g_chunk = tla.tile_view(
                        g_row, tla.make_shape(SIMD_LANES), tla.make_coord(vl_i)
                    )
                    u_chunk = tla.tile_view(
                        u_row, tla.make_shape(SIMD_LANES), tla.make_coord(vl_i)
                    )
                    result_chunk = tla.tile_view(
                        o_row, tla.make_shape(SIMD_LANES), tla.make_coord(vl_i)
                    )
                    tail, remaining = tla.update_mask(remaining, dtype=tla.Float32)
                    gv = g_chunk.load()
                    uv = u_chunk.load()
                    t = tla.mul(gv, -1.0, mask=tail)
                    t = tla.exp(t, mask=tail)
                    t = tla.add(t, 1.0, mask=tail)
                    t = tla.div(gv, t, mask=tail)
                    yv = tla.mul(t, uv, mask=tail)
                    if tla.const_expr(dtype_gm_c != tla.Float32):
                        tail_h, remaining_h = tla.update_mask(
                            remaining_h, dtype=tla.Float16
                        )
                        result_chunk.store(
                            yv.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=tail),
                            _PACK_STORE,
                            mask=tail_h,
                        )
                    else:
                        result_chunk.store(yv, mask=tail)


@tla.jit
def swiglu_apply_vl_2d(
    acc_2d,
    out_2d,
    rows,
    cols_padded,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
) -> None:
    """2D acc with pitch ≥ ``cols_padded``; per-row [g|u] → half-wide out."""
    c0 = 0
    c1 = 1
    half_w = cols_padded // 2
    n_vl = (half_w + SIMD_LANES - 1) // SIMD_LANES
    aligned = (half_w % SIMD_LANES) == c0
    gate_2d = tla.tile_view(
        acc_2d, tla.make_shape(rows, half_w), tla.make_coord(c0, c0)
    )
    up_2d = tla.tile_view(acc_2d, tla.make_shape(rows, half_w), tla.make_coord(c0, c1))
    with tla.vec.func(mode="simd"):
        if aligned:
            mask_f = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            mask_h = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            half_vl = half_w // SIMD_LANES
            for r in tla.range(c0, rows, c1):
                for vl_i in tla.range(c0, half_vl, c1):
                    g_chunk = tla.tile_view(
                        gate_2d,
                        tla.make_shape(c1, SIMD_LANES),
                        tla.make_coord(r, vl_i),
                    )
                    u_chunk = tla.tile_view(
                        up_2d,
                        tla.make_shape(c1, SIMD_LANES),
                        tla.make_coord(r, vl_i),
                    )
                    result_chunk = tla.tile_view(
                        out_2d,
                        tla.make_shape(c1, SIMD_LANES),
                        tla.make_coord(r, vl_i),
                    )
                    gv = g_chunk.load()
                    uv = u_chunk.load()
                    t = tla.mul(gv, -1.0, mask=mask_f)
                    t = tla.exp(t, mask=mask_f)
                    t = tla.add(t, 1.0, mask=mask_f)
                    t = tla.div(gv, t, mask=mask_f)
                    yv = tla.mul(t, uv, mask=mask_f)
                    _store_pack_b16(result_chunk, yv, mask_f, mask_h, dtype_gm_c)
        else:
            for r in tla.range(c0, rows, c1):
                remaining = half_w
                remaining_h = half_w * 2
                for vl_i in tla.range(c0, n_vl, c1):
                    g_chunk = tla.tile_view(
                        gate_2d,
                        tla.make_shape(c1, SIMD_LANES),
                        tla.make_coord(r, vl_i),
                    )
                    u_chunk = tla.tile_view(
                        up_2d,
                        tla.make_shape(c1, SIMD_LANES),
                        tla.make_coord(r, vl_i),
                    )
                    result_chunk = tla.tile_view(
                        out_2d,
                        tla.make_shape(c1, SIMD_LANES),
                        tla.make_coord(r, vl_i),
                    )
                    tail, remaining = tla.update_mask(remaining, dtype=tla.Float32)
                    gv = g_chunk.load()
                    uv = u_chunk.load()
                    t = tla.mul(gv, -1.0, mask=tail)
                    t = tla.exp(t, mask=tail)
                    t = tla.add(t, 1.0, mask=tail)
                    t = tla.div(gv, t, mask=tail)
                    yv = tla.mul(t, uv, mask=tail)
                    if tla.const_expr(dtype_gm_c != tla.Float32):
                        tail_h, remaining_h = tla.update_mask(
                            remaining_h, dtype=tla.Float16
                        )
                        result_chunk.store(
                            yv.to(dtype_gm_c, _CAST_ACC_TO_OUT, mask=tail),
                            _PACK_STORE,
                            mask=tail_h,
                        )
                    else:
                        result_chunk.store(yv, mask=tail)
