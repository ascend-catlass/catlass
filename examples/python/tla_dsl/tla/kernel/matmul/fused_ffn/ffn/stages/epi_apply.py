# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Phase-1: dispatch SiLU / GELU / SwiGLU / RAW / SwiGLU-dual VL apply by ACT_*.
from __future__ import annotations

import catlass.tla as tla

from ffn.config import ACT_GELU, ACT_RAW, ACT_SILU, ACT_SWIGLU, ACT_SWIGLU_DUAL
from ffn.stages.epi_gelu_erf import gelu_apply_vl_1d, gelu_apply_vl_2d
from ffn.stages.epi_silu import silu_apply_vl_1d, silu_apply_vl_2d
from ffn.stages.epi_swiglu_dual import (
    raw_store_vl_1d,
    raw_store_vl_2d,
    swiglu_dual_apply_vl_1d,
    swiglu_dual_apply_vl_2d,
)
from ffn.stages.epi_swiglu_single import swiglu_apply_vl_1d, swiglu_apply_vl_2d


@tla.jit
def epi_apply_vl_1d(
    ACT: tla.Constexpr[int],
    acc_1d,
    out_1d,
    n_elems,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
) -> None:
    """SiLU/GELU/RAW: ``n_elems`` is flat length. SwiGLU single uses ``epi_apply_swiglu_vl_1d``."""
    if tla.const_expr(ACT == ACT_SILU):
        silu_apply_vl_1d(acc_1d, out_1d, n_elems, SIMD_LANES, dtype_gm_c)
    elif tla.const_expr(ACT == ACT_GELU):
        gelu_apply_vl_1d(acc_1d, out_1d, n_elems, SIMD_LANES, dtype_gm_c)
    elif tla.const_expr(ACT == ACT_RAW):
        raw_store_vl_1d(acc_1d, out_1d, n_elems, SIMD_LANES)


@tla.jit
def epi_apply_swiglu_vl_1d(
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
    swiglu_apply_vl_1d(
        acc_1d,
        out_1d,
        rows,
        cols_padded,
        SIMD_LANES,
        dtype_gm_c,
        USE_ALL_MASK,
        TAIL_LANES,
        HOIST_N_VL,
        BAKE_TN,
    )


@tla.jit
def epi_apply_swiglu_dual_vl_1d(
    acc_1d,
    gate_1d,
    out_1d,
    n_elems,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
) -> None:
    swiglu_dual_apply_vl_1d(acc_1d, gate_1d, out_1d, n_elems, SIMD_LANES, dtype_gm_c)


@tla.jit
def epi_apply_vl_2d(
    ACT: tla.Constexpr[int],
    ub_c_tile,
    ub_result,
    tile_rows,
    tile_cols,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
) -> None:
    if tla.const_expr(ACT == ACT_SILU):
        silu_apply_vl_2d(
            ub_c_tile, ub_result, tile_rows, tile_cols, SIMD_LANES, dtype_gm_c
        )
    elif tla.const_expr(ACT == ACT_GELU):
        gelu_apply_vl_2d(
            ub_c_tile, ub_result, tile_rows, tile_cols, SIMD_LANES, dtype_gm_c
        )
    elif tla.const_expr(ACT == ACT_SWIGLU):
        swiglu_apply_vl_2d(
            ub_c_tile, ub_result, tile_rows, tile_cols, SIMD_LANES, dtype_gm_c
        )
    elif tla.const_expr(ACT == ACT_RAW):
        raw_store_vl_2d(ub_c_tile, ub_result, tile_rows, tile_cols, SIMD_LANES)


@tla.jit
def epi_apply_swiglu_dual_vl_2d(
    ub_c_tile,
    ub_gate_tile,
    ub_result,
    tile_rows,
    tile_cols,
    SIMD_LANES: tla.Constexpr[int],
    dtype_gm_c,
) -> None:
    swiglu_dual_apply_vl_2d(
        ub_c_tile,
        ub_gate_tile,
        ub_result,
        tile_rows,
        tile_cols,
        SIMD_LANES,
        dtype_gm_c,
    )
