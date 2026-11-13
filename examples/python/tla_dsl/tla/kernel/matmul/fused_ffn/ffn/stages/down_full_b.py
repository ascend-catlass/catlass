# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Down B-full-load (CPP FULL_LOAD_MODE=2): weight2 常驻 L1；按 M 分核流式搬 A。
from __future__ import annotations

import catlass.tla as tla

from ffn.gm_l1 import gm_l1_abs_1d, gm_l1_tile
from ffn.tiling import SwizzleParams, TilingParams


@tla.jit
def stage_down_full_b(
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
    # CPP BType::isTrans: Linear ColumnMajor=True; Canonical RowMajor=False.
    B_IS_TRANS: tla.Constexpr[bool] = True,
) -> None:
    """CPP B-full-load: prologue CopyInB1(+bias); tile along M; stream A@K.

    Host ``TilingParams`` (from DoBL1FullLoad):
      l1_tm=baseM, l1_tn=align(N), l1_tk=stream_kl1,
      l0_tm/tn/tk=baseMN/K, resident_k=align(K), l1_buf_num=2|4.

    Control flow (CPP ``kernel_matmul_without_que`` B_FULL_LOAD):
      ASWT tileIdx over mL1×nL1 (nTileNum≈1); inner nOffset=l0_tn then K.
    """
    c0 = 0
    c1 = 1
    del l1b1_ptr

    m = gm_a.origin_shape[0]
    # C is always [M,N]. Linear B is GM [N,K] (isTrans); canonical B is [K,N].
    # Never read N from gm_b.origin_shape[1] — that is K under linear.
    n = gm_c.origin_shape[1]
    k = gm_a.origin_shape[1]

    l0_0_available = tla.flag("dnfb_l0_0_av", tla.arch.CUBE, tla.arch.MTE1)
    l0_1_available = tla.flag("dnfb_l0_1_av", tla.arch.CUBE, tla.arch.MTE1)
    l0_0_data_ready = tla.flag("dnfb_l0_0_dr", tla.arch.MTE1, tla.arch.CUBE)
    l0_1_data_ready = tla.flag("dnfb_l0_1_dr", tla.arch.MTE1, tla.arch.CUBE)
    l1_0_data_ready = tla.flag("dnfb_l1_0_dr", tla.arch.MTE2, tla.arch.MTE1)
    l1_1_data_ready = tla.flag("dnfb_l1_1_dr", tla.arch.MTE2, tla.arch.MTE1)
    l1_0_available = tla.flag("dnfb_l1_0_av", tla.arch.MTE1, tla.arch.MTE2)
    l1_1_available = tla.flag("dnfb_l1_1_av", tla.arch.MTE1, tla.arch.MTE2)
    l1b_data_ready = tla.flag("dnfb_l1b_dr", tla.arch.MTE2, tla.arch.MTE1)
    l1b_available = tla.flag("dnfb_l1b_av", tla.arch.MTE1, tla.arch.MTE2)
    l0c_data_ready = tla.flag("dnfb_l0c_dr", tla.arch.CUBE, tla.arch.FIX)
    l0c_available = tla.flag("dnfb_l0c_av", tla.arch.FIX, tla.arch.CUBE)

    l1_tm = _tiling.l1_tm
    l1_tk = _tiling.l1_tk
    l1_tn = _tiling.l1_tn
    l0_tn = _tiling.l0_tn
    l0_tk = _tiling.l0_tk
    c0_elems = 16
    a_one = _tiling.l1_tm * _tiling.l1_tk
    carve_buf_n = _tiling.l1_buf_num if _tiling.l1_buf_num >= 2 else 2
    dual_a = tla.const_expr(carve_buf_n >= 2)
    buf_unified = tla.const_expr(
        L1_SINGLE and BUF_L1_TM > 0 and BUF_L1_TK == 1 and BUF_L1_TN == 0
    )
    if tla.const_expr(buf_unified):
        l1_base = l1a0_ptr
        l1a0_c = l1_base
        l1a1_c = l1_base + a_one
        l1b_c = l1_base + a_one * carve_buf_n
    elif tla.const_expr(L1_SINGLE):
        l1a0_c = l1a0_ptr
        l1a1_c = l1a0_ptr
        l1b_c = l1b0_ptr
        dual_a = tla.const_expr(False)
    else:
        l1a0_c = l1a0_ptr
        l1a1_c = l1a1_ptr
        l1b_c = l1b0_ptr

    # CPP: nTileNum = CeilDiv(n, nL1); L0-N is inner nOffset (not ASWT tasks).
    grid_m = (m + l1_tm - 1) // l1_tm
    grid_n = (n + l1_tn - 1) // l1_tn
    total_blocks = grid_m * grid_n
    n_l0_count = (l1_tn + l0_tn - 1) // l0_tn
    swizzle_tile_count = (
        grid_m + _swizzle.SWIZZLE_OFFSET - 1
    ) // _swizzle.SWIZZLE_OFFSET

    with tla.cube():
        tla.set_flag(l1_0_available)
        tla.set_flag(l1_1_available)
        tla.set_flag(l1b_available)
        tla.set_flag(l0_0_available)
        tla.set_flag(l0_1_available)
        tla.set_flag(l0c_available)
        if tla.const_expr(USE_BIAS):
            tla.set_flag(l1_bias_available)
            tla.set_flag(l0_bias_available)

        tla.wait_flag(l1b_available)
        if tla.const_expr(B_IS_TRANS):
            # Linear: create_tla_tensor(..., "col") exposes ColumnMajor
            # logical (K, N) over physical contiguous [N,K]. CPP isTrans
            # CopyInB1 Nd2Nz uses nValue=N, dValue=K, srcDValue=K —
            # ColumnMajor→nZ reads get<1>=N, get<0>=K. Tile must be (K,N),
            # never (N,K) (#0688 mean_rel blow-up).
            gm_b_full = gm_l1_tile(gm_b, k, n, c0, c0)
            l1_b_full = tla.make_tensor_like(l1b_c, gm_b_full, layoutTag=tla.arch.nZ)
        else:
            # Canonical RowMajor [K,N] → zN (CPP Nd2Nz nValue=K,
            # dstNzC0Stride=Align(K,16)).
            gm_b_full = gm_l1_tile(gm_b, k, n, c0, c0)
            l1_b_full = tla.make_tensor_like(l1b_c, gm_b_full, layoutTag=tla.arch.zN)
        tla.copy(l1_b_full, gm_b_full)
        tla.set_flag(l1b_data_ready)

        if tla.const_expr(USE_BIAS):
            gm_bias_full = gm_l1_abs_1d(gm_bias, n, c0)
            l1_bias_full = tla.make_tensor_like(
                l1_bias_ptr, gm_bias_full, tla.arch.VectorLayout
            )
            tla.wait_flag(l1_bias_available)
            tla.copy(l1_bias_full, gm_bias_full)
            tla.set_flag(l1_bias_data_ready)

        block_idx = tla.arch.block_idx()
        block_num = tla.arch.block_num()
        l1_buf_idx = tla.as_numeric(0)
        l0_buf_idx = tla.as_numeric(0)
        # CPP abL1LoopCnt_ & 0x1 → biasBt_[baseN_ * biasBufId] ping-pong.
        bias_buf_idx = tla.as_numeric(0)
        l0_bias_ptr1 = l0_bias_ptr
        if tla.const_expr(USE_BIAS):
            l0_bias_ptr1 = l0_bias_ptr + l0_tn

        tla.wait_flag(l1b_data_ready)
        for task_idx in tla.range(block_idx, total_blocks, block_num):
            # ASWT on mL1 × nL1 (CPP UpdateMNTileIdx; nTileNum usually 1).
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
            gm_a_row = tla.tile_view(
                gm_a, tla.make_shape(l1_tm, k), tla.make_coord(block_row, c0)
            )
            k_block = gm_a_row.origin_shape[1]
            k_l1_count = (k_block + l1_tk - 1) // l1_tk

            # CPP inner nOffset loop (baseN / l0_tn).
            for n_i in tla.range(c0, n_l0_count, c1):
                gm_c_tile = tla.tile_view(
                    gm_c,
                    tla.make_shape(l1_tm, l0_tn),
                    tla.make_coord(block_row, block_col * n_l0_count + n_i),
                )
                l0_c = tla.make_tensor_like(l0c_ptr, gm_c_tile)
                tla.wait_flag(l0c_available)

                l0_bias = l0c_ptr
                if tla.const_expr(USE_BIAS):
                    l1_bias_tile = tla.tile_view(
                        l1_bias_full,
                        tla.make_shape(l0_tn),
                        tla.make_coord(block_col * n_l0_count + n_i),
                    )
                    cur_l0_bias_ptr = (
                        l0_bias_ptr if bias_buf_idx == c0 else l0_bias_ptr1
                    )
                    l0_bias = tla.make_tensor_like(
                        cur_l0_bias_ptr, l1_bias_tile, tla.arch.VectorLayout
                    )
                    tla.wait_flag(l1_bias_data_ready)
                    tla.wait_flag(l0_bias_available)
                    tla.copy(l0_bias, l1_bias_tile)

                for k_l1 in tla.range(c0, k_l1_count, c1):
                    gm_a_l1 = gm_l1_tile(gm_a_row, l1_tm, l1_tk, c0, k_l1)
                    cur_tk = gm_a_l1.origin_shape[1]
                    l1_a = tla.make_tensor_like(
                        l1a0_c if (l1_buf_idx == c0) else l1a1_c, gm_a_l1
                    )
                    if l1_buf_idx == c0:
                        tla.wait_flag(l1_0_available)
                    else:
                        tla.wait_flag(l1_1_available)
                    tla.copy(l1_a, gm_a_l1)
                    if l1_buf_idx == c0:
                        tla.set_flag(l1_0_data_ready)
                    else:
                        tla.set_flag(l1_1_data_ready)
                    if l1_buf_idx == c0:
                        tla.wait_flag(l1_0_data_ready)
                    else:
                        tla.wait_flag(l1_1_data_ready)

                    # CPP DoBFullLoad CopyInA2/B2: fold fractal offsets into
                    # L1 pointers (mStart=kStart=0). tile_view leaves m/kStart
                    # nonzero → Load2D L0 OOB (same class as A-full #0031).
                    # #0688 linear isTrans: L0B must NOT make_tensor_like(L1B)
                    # — that inherits resident srcStride (l1_tn) as L0 dst and
                    # writes past L0B; CPP dstStride=mStep from curNL0 only.
                    l1a_ptr = l1a0_c if (l1_buf_idx == c0) else l1a1_c
                    # Nd2Nz packs this L1 A tile at Align(curML1, 16). Edge
                    # tiles are shorter than baseM (silu#0228: M=8192,
                    # l1_tm=288 → last tile 128). Pitching K-fractals by
                    # l1_tm reads the wrong K and blows up that M strip.
                    cur_m = gm_a_l1.origin_shape[0]
                    ml1_align = (cur_m + 15) // 16 * 16
                    k_l0_count = (cur_tk + l0_tk - 1) // l0_tk
                    for k_l0 in tla.range(c0, k_l0_count, c1):
                        # CPP CopyInB2 uses curNL0/curK0 (rem), not baseN/K.
                        # #0604: N=257,l0_tn=128 → n_off=256 rem=1; loading
                        # published l0_tn=128 walks past l1_tn=272 → L1 OOB.
                        n_off = (block_col * n_l0_count + n_i) * l0_tn
                        remain_n = n - n_off
                        cur_n = l0_tn
                        if remain_n < l0_tn:
                            cur_n = remain_n
                        cur_n_align = (cur_n + 15) // 16 * 16
                        k_rel = k_l0 * l0_tk
                        k_off = k_l1 * l1_tk + k_rel
                        remain_k = cur_tk - k_rel
                        cur_k = l0_tk
                        if remain_k < l0_tk:
                            cur_k = remain_k
                        # CPP !isTrans A: offsetAl1 += ml1Align*baseK;
                        # srcStride=CeilDiv(curML1,16); m/kStep from curML0/K0.
                        nz_off_a = ml1_align * k_rel
                        l1_a_k = tla.make_tensor(
                            l1a_ptr + nz_off_a,
                            tla.make_layout(
                                tla.make_shape(
                                    (16, l1_tm // 16),
                                    (16, l0_tk // 16),
                                ),
                                tla.make_stride(
                                    (16, 256),
                                    (1, ml1_align * 16),
                                ),
                                origin_shape=tla.make_shape(cur_m, l0_tk),
                                layoutTag=tla.arch.zN,
                            ),
                        )
                        if tla.const_expr(B_IS_TRANS):
                            # B-full isTrans: L1 origin N is resident l1_tn;
                            # rem N only affects L0 dest mStep.
                            nz_off = n_off * c0_elems + k_off * l1_tn
                            l1_b_k = tla.make_tensor(
                                l1b_c + nz_off,
                                tla.make_layout(
                                    tla.make_shape(
                                        (c0_elems, l0_tk // c0_elems),
                                        (16, l0_tn // 16),
                                    ),
                                    tla.make_stride(
                                        (1, l1_tn * c0_elems),
                                        (c0_elems, 256),
                                    ),
                                    origin_shape=tla.make_shape(l0_tk, l1_tn),
                                    layoutTag=tla.arch.nZ,
                                ),
                            )
                        else:
                            # !isTrans: resident zN pitch is kAlign; rem
                            # kL0/nL0 set L0 mStep/kStep.
                            k_align = (
                                _tiling.resident_k
                                if _tiling.resident_k > 0
                                else ((k + 15) // 16) * 16
                            )
                            nz_off = n_off * k_align + k_off * c0_elems
                            l1_b_k = tla.make_tensor(
                                l1b_c + nz_off,
                                tla.make_layout(
                                    tla.make_shape(
                                        (c0_elems, l0_tk // c0_elems),
                                        (16, l0_tn // 16),
                                    ),
                                    tla.make_stride(
                                        (c0_elems, 256),
                                        (1, k_align * c0_elems),
                                    ),
                                    origin_shape=tla.make_shape(l0_tk, cur_n_align),
                                    layoutTag=tla.arch.zN,
                                ),
                            )
                        # CPP CopyInA2: dstStride=mStep=CeilDiv(curML0,16).
                        # Same pitch as the L1 tile (ml1_align), not baseM.
                        cur_m_align = ml1_align
                        l0_a = tla.make_tensor(
                            l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr,
                            tla.make_layout(
                                tla.make_shape(
                                    (16, l1_tm // 16),
                                    (16, l0_tk // 16),
                                ),
                                tla.make_stride(
                                    (16, 256),
                                    (1, cur_m_align * 16),
                                ),
                                origin_shape=tla.make_shape(cur_m, cur_k),
                                layoutTag=tla.arch.zN,
                            ),
                        )
                        # L0B HW is always nZ; !isTrans CopyInB2 uses
                        # ifTranspose=true (zN L1 → nZ L0). isTrans also
                        # builds compact L0 (dstStride=mStep), never inherits
                        # resident L1 stride.
                        l0_b = tla.make_tensor(
                            l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr,
                            tla.make_layout(
                                tla.make_shape(
                                    (c0_elems, l0_tk // c0_elems),
                                    (16, l0_tn // 16),
                                ),
                                tla.make_stride(
                                    (1, cur_n_align * c0_elems),
                                    (c0_elems, 256),
                                ),
                                origin_shape=tla.make_shape(cur_k, cur_n),
                                layoutTag=tla.arch.nZ,
                            ),
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
                                tla.mmad(l0_c, l0_a, l0_b, init_c=True, bias=l0_bias)
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
                    if tla.const_expr(dual_a):
                        l1_buf_idx = c1 - l1_buf_idx

                if tla.const_expr(USE_BIAS):
                    tla.set_flag(l0_bias_available)
                    tla.set_flag(l1_bias_data_ready)
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

        tla.set_flag(l1b_data_ready)
        if tla.const_expr(USE_BIAS):
            tla.set_flag(l1_bias_available)
        tla.wait_flag(l1_0_available)
        tla.wait_flag(l1_1_available)
        tla.wait_flag(l0_0_available)
        tla.wait_flag(l0_1_available)
        tla.wait_flag(l0c_available)
        tla.set_flag(l1b_available)
        if tla.const_expr(USE_BIAS):
            tla.wait_flag(l1_bias_available)
            tla.wait_flag(l0_bias_available)
