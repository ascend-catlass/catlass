# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Down A-full-load (CPP FULL_LOAD_MODE=1): hidden 常驻 L1；按 N 分核流式搬 B。
from __future__ import annotations

import catlass.tla as tla

from ffn.gm_l1 import gm_l1_abs, gm_l1_abs_1d, gm_l1_tile
from ffn.tiling import SwizzleParams, TilingParams


@tla.jit
def stage_down_full_a(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    l1a0_ptr,
    l1a1_ptr,
    l1b0_ptr,
    l1b1_ptr,
    l0a0_ptr,
    l0a1_ptr,
    l0b0_ptr,
    l0b1_ptr,
    l0c_ptr,
    _tiling: tla.Constexpr[TilingParams],
    _swizzle: tla.Constexpr[SwizzleParams],
    L1_SINGLE: tla.Constexpr[bool] = False,
    BUF_L1_TM: tla.Constexpr[int] = 0,
    BUF_L1_TN: tla.Constexpr[int] = 0,
    BUF_L1_TK: tla.Constexpr[int] = 0,
    L0_SINGLE: tla.Constexpr[bool] = False,
    gm_bias: tla.Tensor = None,  # noqa: RUF013
    l1_bias_ptr=None,
    l0_bias_ptr=None,
    USE_BIAS: tla.Constexpr[bool] = False,
    l1_bias_data_ready=None,
    l1_bias_available=None,
    l0_bias_available=None,
) -> None:
    """CPP A-full-load: prologue CopyInA1; tile along N; stream B@K.

    Host ``TilingParams`` (from DoAL1FullLoad):
      l1_tm=align(M), l1_tn=baseN, l1_tk=stream_kl1,
      l0_tm/tn/tk=baseMN/K, resident_k=align(K), l1_buf_num=2|4.

    L1 layout (CPP): AL1_full | BL1Ping[|…|BL1×buf] | Bias.
    """
    c0 = 0
    c1 = 1
    del l1a1_ptr

    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    k = gm_a.origin_shape[1]

    l0_0_available = tla.flag("dnfa_l0_0_av", tla.arch.CUBE, tla.arch.MTE1)
    l0_1_available = tla.flag("dnfa_l0_1_av", tla.arch.CUBE, tla.arch.MTE1)
    l0_0_data_ready = tla.flag("dnfa_l0_0_dr", tla.arch.MTE1, tla.arch.CUBE)
    l0_1_data_ready = tla.flag("dnfa_l0_1_dr", tla.arch.MTE1, tla.arch.CUBE)
    l1_0_data_ready = tla.flag("dnfa_l1_0_dr", tla.arch.MTE2, tla.arch.MTE1)
    l1_1_data_ready = tla.flag("dnfa_l1_1_dr", tla.arch.MTE2, tla.arch.MTE1)
    l1_0_available = tla.flag("dnfa_l1_0_av", tla.arch.MTE1, tla.arch.MTE2)
    l1_1_available = tla.flag("dnfa_l1_1_av", tla.arch.MTE1, tla.arch.MTE2)
    l1a_data_ready = tla.flag("dnfa_l1a_dr", tla.arch.MTE2, tla.arch.MTE1)
    l1a_available = tla.flag("dnfa_l1a_av", tla.arch.MTE1, tla.arch.MTE2)
    l0c_data_ready = tla.flag("dnfa_l0c_dr", tla.arch.CUBE, tla.arch.FIX)
    l0c_available = tla.flag("dnfa_l0c_av", tla.arch.FIX, tla.arch.CUBE)

    l1_tm = _tiling.l1_tm
    l1_tn = _tiling.l1_tn
    l1_tk = _tiling.l1_tk
    l0_tm = _tiling.l0_tm
    l0_tn = _tiling.l0_tn
    l0_tk = _tiling.l0_tk
    a_full = _tiling.l1_tm * _tiling.resident_k
    b_one = _tiling.l1_tk * _tiling.l1_tn
    buf_n = _tiling.l1_buf_num if _tiling.l1_buf_num >= 2 else 2
    dual_b = tla.const_expr(buf_n >= 2)
    buf_unified = tla.const_expr(
        L1_SINGLE and BUF_L1_TM > 0 and BUF_L1_TK == 1 and BUF_L1_TN == 0
    )
    if tla.const_expr(buf_unified):
        l1_base = l1a0_ptr
        l1a_c = l1_base
        l1b0_c = l1_base + a_full
        l1b1_c = l1_base + a_full + b_one
    elif tla.const_expr(L1_SINGLE):
        l1a_c = l1a0_ptr
        l1b0_c = l1b0_ptr
        l1b1_c = l1b0_ptr
        dual_b = tla.const_expr(False)
    else:
        l1a_c = l1a0_ptr
        l1b0_c = l1b0_ptr
        l1b1_c = l1b1_ptr

    # CPP ASWT: mTileNum×nTileNum on mL1×nL1 (A-full: mL1≈M → mTileNum=1).
    grid_m = (m + l1_tm - 1) // l1_tm
    grid_n = (n + l1_tn - 1) // l1_tn
    total_blocks = grid_m * grid_n
    m_l0_count = (l1_tm + l0_tm - 1) // l0_tm
    n_l0_count = (l1_tn + l0_tn - 1) // l0_tn
    swizzle_tile_count = (
        grid_m + _swizzle.SWIZZLE_OFFSET - 1
    ) // _swizzle.SWIZZLE_OFFSET

    with tla.cube():
        tla.set_flag(l1_0_available)
        tla.set_flag(l1_1_available)
        tla.set_flag(l1a_available)
        tla.set_flag(l0_0_available)
        tla.set_flag(l0_1_available)
        tla.set_flag(l0c_available)
        if tla.const_expr(USE_BIAS):
            tla.set_flag(l1_bias_available)
            tla.set_flag(l0_bias_available)

        tla.wait_flag(l1a_available)
        gm_a_full = gm_l1_tile(gm_a, m, k, c0, c0)
        l1_a_full = tla.make_tensor_like(l1a_c, gm_a_full)
        tla.copy(l1_a_full, gm_a_full)
        tla.set_flag(l1a_data_ready)

        block_idx = tla.arch.block_idx()
        block_num = tla.arch.block_num()
        l1_buf_idx = tla.as_numeric(0)
        l0_buf_idx = tla.as_numeric(0)
        # CPP abL1LoopCnt_ & 0x1 → biasBt_[baseN_ * biasBufId] ping-pong.
        # A-full: baseN = l0_tn (L0-N strips), not l1_tn.
        bias_buf_idx = tla.as_numeric(0)
        l0_bias_ptr1 = l0_bias_ptr
        if tla.const_expr(USE_BIAS):
            l0_bias_ptr1 = l0_bias_ptr + l0_tn

        for task_idx in tla.range(block_idx, total_blocks, block_num):
            swizzle_tile_idx = task_idx // (_swizzle.SWIZZLE_OFFSET * grid_n)
            in_tile_idx = task_idx % (_swizzle.SWIZZLE_OFFSET * grid_n)
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
                (grid_n - block_col - 1) if (swizzle_tile_idx % 2 == 1) else block_col
            )

            # N-tail width for this scheduler tile (bias/B DMA must not use full l1_tn).
            work_tn = l1_tn
            if block_col == (grid_n - c1):
                work_tn = n - (grid_n - c1) * l1_tn
            # Loop only over L0-N strips that cover work_tn; global C index still
            # uses published n_l0_count (fixed stride per nL1 tile).
            n_l0_eff = (work_tn + l0_tn - 1) // l0_tn

            if tla.const_expr(USE_BIAS):
                gm_bias_tile = gm_l1_abs_1d(gm_bias, work_tn, block_col * l1_tn)
                l1_bias_tile = tla.make_tensor_like(
                    l1_bias_ptr, gm_bias_tile, tla.arch.VectorLayout
                )
                tla.wait_flag(l1_bias_available)
                tla.copy(l1_bias_tile, gm_bias_tile)
                tla.set_flag(l1_bias_data_ready)

            # Resident A: handshake once per scheduler tile.
            tla.wait_flag(l1a_data_ready)
            # Bias L1→L0: wait SoftFlag once per nL1 tile (not per L0-N),
            # then strip by n_i. Re-arm data_ready after the last L0 strip.
            if tla.const_expr(USE_BIAS):
                tla.wait_flag(l1_bias_data_ready)
            for m_i in tla.range(c0, m_l0_count, c1):
                for n_i in tla.range(c0, n_l0_eff, c1):
                    gm_c_tile = tla.tile_view(
                        gm_c,
                        tla.make_shape(l0_tm, l0_tn),
                        tla.make_coord(
                            block_row * m_l0_count + m_i,
                            block_col * n_l0_count + n_i,
                        ),
                    )
                    # origin_shape crops M/N tails; tile shape stays l0_tm×l0_tn
                    # for coord stride (see B/bias views below).
                    l0_c = tla.make_tensor_like(l0c_ptr, gm_c_tile)
                    tla.wait_flag(l0c_available)

                    l0_bias = l0c_ptr
                    if tla.const_expr(USE_BIAS):
                        # Fixed l0_tn for coord stride (tail crops via origin);
                        # never index by runtime work_tn (would mis-address tail).
                        l1_bias_use = tla.tile_view(
                            l1_bias_tile,
                            tla.make_shape(l0_tn),
                            tla.make_coord(n_i),
                        )
                        cur_l0_bias_ptr = (
                            l0_bias_ptr if bias_buf_idx == c0 else l0_bias_ptr1
                        )
                        l0_bias = tla.make_tensor_like(
                            cur_l0_bias_ptr, l1_bias_use, tla.arch.VectorLayout
                        )
                        tla.wait_flag(l0_bias_available)
                        tla.copy(l0_bias, l1_bias_use)

                    k_l1_count = (k + l1_tk - 1) // l1_tk
                    for k_l1 in tla.range(c0, k_l1_count, c1):
                        # CPP DoAFullLoad CopyInB1: GM→L1 at curNL1=work_tn
                        # (N-tail clipped). Absolute N start = block_col*l1_tn
                        # (not block_col*work_tn — tile stride stays published).
                        gm_b_l1 = gm_l1_abs(
                            gm_b, l1_tk, work_tn, k_l1 * l1_tk, block_col * l1_tn
                        )
                        cur_tk = gm_b_l1.origin_shape[0]
                        l1_b = tla.make_tensor_like(
                            l1b0_c if (l1_buf_idx == c0) else l1b1_c, gm_b_l1
                        )
                        if l1_buf_idx == c0:
                            tla.wait_flag(l1_0_available)
                        else:
                            tla.wait_flag(l1_1_available)
                        tla.copy(l1_b, gm_b_l1)
                        if l1_buf_idx == c0:
                            tla.set_flag(l1_0_data_ready)
                        else:
                            tla.set_flag(l1_1_data_ready)

                        l1_a_strip = tla.tile_view(
                            l1_a_full,
                            tla.make_shape(l0_tm, l1_tk),
                            tla.make_coord(block_row * m_l0_count + m_i, k_l1),
                        )
                        # CopyInB2: L1→L0 strip at curNL0=l0_tn (may be
                        # L0B-capped < l1_tn). Tile-coord n_i into L1 B.
                        l1_b_strip = tla.tile_view(
                            l1_b,
                            tla.make_shape(l1_tk, l0_tn),
                            tla.make_coord(c0, n_i),
                        )

                        if l1_buf_idx == c0:
                            tla.wait_flag(l1_0_data_ready)
                        else:
                            tla.wait_flag(l1_1_data_ready)

                        k_l0_count = (cur_tk + l0_tk - 1) // l0_tk
                        for k_l0 in tla.range(c0, k_l0_count, c1):
                            l1_a_k = tla.tile_view(
                                l1_a_strip,
                                tla.make_shape(l0_tm, l0_tk),
                                tla.make_coord(c0, k_l0),
                            )
                            l1_b_k = tla.tile_view(
                                l1_b_strip,
                                tla.make_shape(l0_tk, l0_tn),
                                tla.make_coord(k_l0, c0),
                            )
                            l0_a = tla.make_tensor_like(
                                l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr,
                                l1_a_k,
                            )
                            l0_b = tla.make_tensor_like(
                                l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr,
                                l1_b_k,
                            )
                            if l0_buf_idx == c0:
                                tla.wait_flag(l0_0_available)
                            else:
                                tla.wait_flag(l0_1_available)
                            tla.copy(l0_a, l1_a_k)
                            tla.copy(l0_b, l1_b_k)
                            if l0_buf_idx == c0:
                                tla.set_flag(l0_0_data_ready)
                                tla.wait_flag(l0_0_data_ready)
                            else:
                                tla.set_flag(l0_1_data_ready)
                                tla.wait_flag(l0_1_data_ready)
                            init_c = True if (k_l1 == c0 and k_l0 == c0) else False
                            if tla.const_expr(USE_BIAS):
                                if init_c:
                                    tla.mmad(
                                        l0_c,
                                        l0_a,
                                        l0_b,
                                        init_c=True,
                                        bias=l0_bias,
                                    )
                                else:
                                    tla.mmad(l0_c, l0_a, l0_b, init_c=False)
                            else:
                                tla.mmad(l0_c, l0_a, l0_b, init_c=init_c)
                            if l0_buf_idx == c0:
                                tla.set_flag(l0_0_available)
                            else:
                                tla.set_flag(l0_1_available)
                            if tla.const_expr(not L0_SINGLE):
                                l0_buf_idx = c1 - l0_buf_idx

                        if l1_buf_idx == c0:
                            tla.set_flag(l1_0_available)
                        else:
                            tla.set_flag(l1_1_available)
                        if tla.const_expr(dual_b):
                            l1_buf_idx = c1 - l1_buf_idx

                    if tla.const_expr(USE_BIAS):
                        tla.set_flag(l0_bias_available)
                        # Advance BT slot once per L0C strip (CopyInC2 once).
                        bias_buf_idx = c1 - bias_buf_idx
                    tla.set_flag(l0c_data_ready)
                    tla.wait_flag(l0c_data_ready)
                    tla.copy(
                        gm_c_tile,
                        l0_c,
                        tla.params.CopyL0C2DstParams(relu_enable=False),
                    )
                    tla.set_flag(l0c_available)

            tla.set_flag(l1a_data_ready)
            if tla.const_expr(USE_BIAS):
                tla.set_flag(l1_bias_available)

        tla.wait_flag(l1_0_available)
        tla.wait_flag(l1_1_available)
        tla.wait_flag(l0_0_available)
        tla.wait_flag(l0_1_available)
        tla.wait_flag(l0c_available)
        tla.set_flag(l1a_available)
        if tla.const_expr(USE_BIAS):
            tla.wait_flag(l1_bias_available)
            tla.wait_flag(l0_bias_available)
