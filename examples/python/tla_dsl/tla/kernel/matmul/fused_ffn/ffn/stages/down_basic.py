# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# FFN Down stage: Basic cube GEMM (C = A @ B), CPP ``DoBFullLoadOrAswt`` shaped.
from __future__ import annotations

import os

import catlass.tla as tla

from ffn.gm_l1 import gm_l1_abs_1d, gm_l1_tile
from ffn.tiling import SwizzleParams, TilingParams, _HALF_L1_ELEMS

# Soft M↔FIX when l0cDB>1 (CPP enableL0cPingPong_). When l0cDB==1 CPP uses
# unit_flag 2/3 instead of hot-path M↔FIX SoftFlag (FINAL=3, NON_FINAL=2).
HALF_L1_ELEMS = _HALF_L1_ELEMS
L1_SLOT_PAD = 0
_UNIT_NON_FINAL = 2
_UNIT_FINAL = 3
# Skinny M: L0-period K-prefetch only (FH itself follows CPP kL1/baseK≥2).
SKINNY_DOWN_L1_TM = 16


@tla.jit
def stage_down_basic(
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
    gm_bias: tla.Tensor = None,  # noqa: RUF013 — TLA; pass gm_c when unused
    l1_bias_ptr=None,
    l0_bias_ptr=None,
    USE_BIAS: tla.Constexpr[bool] = False,
    l1_bias_data_ready=None,
    l1_bias_available=None,
    l0_bias_available=None,
    PROB_M: tla.Constexpr[int] = 0,
    PROB_N: tla.Constexpr[int] = 0,
    PROB_K: tla.Constexpr[int] = 0,
    BLOCK_DIM: tla.Constexpr[int] = 0,
) -> None:
    """Cube-only GEMM: C = A @ B with dynamic M,N,K from origin_shape.

    On-chip L1/L0/L0C must be supplied by the caller so Up+Down can share one set.
    ``L1_SINGLE`` must match allocate_gemm_pingpong (alias l1*1→l1*0).
    ``BUF_L1_*`` = merge alloc box (0 → same as stage); used to split PP.
    Cube BT (``USE_BIAS``): GM→L1→biasbuf on first-K ``mmad(..., bias=)``.
    """
    c0 = 0
    c1 = 1

    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    k = gm_a.origin_shape[1]

    # SoftFlags: same HardEvent IDs as Up / CPP BlockMmad WithoutQue.
    # CPP ctor: MTE1_MTE2 0..3, M_MTE1 6/7, FIX_M 0/1. Up ~drain before
    # SyncAll frees them for Down (sequential BlockMmad lifetimes).
    l0_0_available = tla.flag("down_l0_0_available", tla.arch.CUBE, tla.arch.MTE1, id=6)
    l0_1_available = tla.flag("down_l0_1_available", tla.arch.CUBE, tla.arch.MTE1, id=7)
    l0_0_data_ready = tla.flag(
        "down_l0_0_data_ready", tla.arch.MTE1, tla.arch.CUBE, id=6
    )
    l0_1_data_ready = tla.flag(
        "down_l0_1_data_ready", tla.arch.MTE1, tla.arch.CUBE, id=7
    )
    # A∥B L1: one ID per buf covers A+B; IDs 2/3 credit-only (CPP SECOND/THIRD).
    l1_0_data_ready = tla.flag(
        "down_l1_0_data_ready", tla.arch.MTE2, tla.arch.MTE1, id=0
    )
    l1_1_data_ready = tla.flag(
        "down_l1_1_data_ready", tla.arch.MTE2, tla.arch.MTE1, id=1
    )
    l1_0_available = tla.flag("down_l1_0_available", tla.arch.MTE1, tla.arch.MTE2, id=0)
    l1_1_available = tla.flag("down_l1_1_available", tla.arch.MTE1, tla.arch.MTE2, id=1)
    l1_2_available = tla.flag("down_l1_2_available", tla.arch.MTE1, tla.arch.MTE2, id=2)
    l1_3_available = tla.flag("down_l1_3_available", tla.arch.MTE1, tla.arch.MTE2, id=3)
    # CPP ctor always Sets FIX_M(0) and FIX_M(1).
    _l0c_db = tla.const_expr(_tiling.l0c_db > 1)
    # CPP: unitFlag = l0cDB ? 0 : (last ? 3 : 2); FixPipe unitFlag = l0cDB ? 0 : 3.
    # Assign plain ints under const_expr branches (ternary→const_expr yielded bool).
    _uf_nf = _UNIT_NON_FINAL
    _uf_fin = _UNIT_FINAL
    if tla.const_expr(_l0c_db):
        _uf_nf = 0
        _uf_fin = 0
    _l0c_slot_elems = _tiling.l0_tm * _tiling.l0_tn
    l0c0_data_ready = tla.flag(
        "down_l0c0_data_ready", tla.arch.CUBE, tla.arch.FIX, id=0
    )
    l0c0_available = tla.flag("down_l0c0_available", tla.arch.FIX, tla.arch.CUBE, id=0)
    l0c1_data_ready = tla.flag(
        "down_l0c1_data_ready", tla.arch.CUBE, tla.arch.FIX, id=1
    )
    l0c1_available = tla.flag("down_l0c1_available", tla.arch.FIX, tla.arch.CUBE, id=1)
    # CPP WithoutQue: one SoftFlag ID per L1 buf covers A+B (no separate A/B
    # HardEvents). Always alias l1a/l1b → l1_0/l1_1.
    l1a0_data_ready = l1_0_data_ready
    l1a1_data_ready = l1_1_data_ready
    l1b0_data_ready = l1_0_data_ready
    l1b1_data_ready = l1_1_data_ready
    l1a0_available = l1_0_available
    l1a1_available = l1_1_available
    l1b0_available = l1_0_available
    l1b1_available = l1_1_available

    grid_m = (m + _tiling.l1_tm - 1) // _tiling.l1_tm
    grid_n = (n + _tiling.l1_tn - 1) // _tiling.l1_tn
    total_blocks = grid_m * grid_n
    # CPP ASWT: always M-window (``choose_swizzle_*`` sets DIRECTION=0).
    swizzle_tile_count = (
        grid_m + _swizzle.SWIZZLE_OFFSET - 1
    ) // _swizzle.SWIZZLE_OFFSET
    m_tail = _swizzle.M_TAIL_CNT
    n_tail = _swizzle.N_TAIL_CNT
    tail_cnt = m_tail * n_tail
    use_mtail = tla.const_expr(tail_cnt > 1)
    # CPP GetBlockShape rem: split = CeilDiv(tailL1Last, mTailCnt) at runtime.
    # Host M_TAIL_SPLIT is debug/fallback only — device does not bake it in.
    l1_single = tla.const_expr(L1_SINGLE)
    # Split merge single buffer into Down ping-pong when 2×stage A/B fit.
    # Unified pool ``(total, tk=1, tn=0)``: carve pingA|pingB|pongA|pongB.
    buf_unified = tla.const_expr(BUF_L1_TK == 1 and BUF_L1_TN == 0 and BUF_L1_TM > 0)
    buf_tm = BUF_L1_TM if BUF_L1_TM > 0 else _tiling.l1_tm
    buf_tn = BUF_L1_TN if BUF_L1_TN > 0 else _tiling.l1_tn
    buf_tk = BUF_L1_TK if BUF_L1_TK > 0 else _tiling.l1_tk
    # Buffer carve: dump kL1 (may >K). MTE2/FH: CPP Min(K,kL1) working quantum.
    _pub_tk = _tiling.l1_tk
    _work_tk = _tiling.work_l1_tk if tla.const_expr(_tiling.work_l1_tk > 0) else _pub_tk
    stage_a = _tiling.l1_tm * _pub_tk
    stage_b = _pub_tk * _tiling.l1_tn
    # CPP isFirstLoopKL1Half: CeilAlign(kL1/2, BLOCK_CUBE=16), rest may be smaller.
    half_tk = ((_work_tk // 2 + 15) // 16) * 16
    rest_tk = _work_tk - half_tk
    can_split = tla.const_expr(
        L1_SINGLE
        and (
            (buf_unified and (2 * (stage_a + stage_b) <= BUF_L1_TM))
            or (
                (not buf_unified)
                and (2 * stage_a <= buf_tm * buf_tk)
                and (2 * stage_b <= buf_tk * buf_tn)
            )
        )
    )
    slot_pad = 0
    use_half_l1 = tla.const_expr(
        buf_unified and can_split and (HALF_L1_ELEMS + stage_a + stage_b <= BUF_L1_TM)
    )
    if tla.const_expr(
        buf_unified
        and can_split
        and (not use_half_l1)
        and L1_SLOT_PAD > 0
        and (2 * (stage_a + stage_b) + L1_SLOT_PAD <= BUF_L1_TM)
    ):
        slot_pad = L1_SLOT_PAD
    if tla.const_expr(can_split):
        if tla.const_expr(buf_unified):
            l1_base = l1a0_ptr
            if tla.const_expr(use_half_l1):
                l1b0_ptr = l1_base + stage_a
                l1a1_ptr = l1_base + HALF_L1_ELEMS
                l1b1_ptr = l1_base + HALF_L1_ELEMS + stage_a
            else:
                l1b0_ptr = l1_base + stage_a
                l1a1_ptr = l1_base + stage_a + stage_b + slot_pad
                l1b1_ptr = l1_base + stage_a + stage_b + slot_pad + stage_a
        else:
            l1a1_ptr = l1a0_ptr + stage_a
            l1b1_ptr = l1b0_ptr + stage_b
        l1_single = tla.const_expr(False)
    # Steady MTE2 quantum = full working kL1 (CPP after FH). No permanent half-K.
    l1_tk_step = min(_work_tk, k)
    skinny_down = tla.const_expr(_tiling.l1_tm <= SKINNY_DOWN_L1_TM)
    # CPP DoBFullLoadOrAswt isFirstLoopKL1Half: dual-L1 ping-pong path only,
    # isFirstTile && (kL1_/baseK_ >= 2). rest_tk>0 excludes degenerate
    # CeilAlign(half) that ate the whole kL1. After can_split carve, l1_single
    # is False iff dual slots exist (same gate as CPP l1BufNum==2).
    # Env FFN_DOWN_FH=0|off disables (A/B); unset/1 keeps CPP gate.
    _fh_raw = os.environ.get("FFN_DOWN_FH", "1").strip().lower()
    _fh_off = _fh_raw in ("0", "false", "off", "no")
    enable_first_half = tla.const_expr(
        (not _fh_off)
        and (not l1_single)
        and (_work_tk // _tiling.l0_tk >= 2)
        and (rest_tk > 0)
        and (half_tk > 0)
    )
    # FH is always 2 L1 fills. When each half fits one L0 (e.g. work=512,
    # l0_tk=256), unroll to SoftFlag-fixed straight line — TLA has no
    # WaitFlag(runtime_id) like CPP ``(l0PingPong_&1)+SIXTH_FLAG``.
    _fh_l0_one = tla.const_expr(
        enable_first_half
        and (not L0_SINGLE)
        and (half_tk <= _tiling.l0_tk)
        and (rest_tk <= _tiling.l0_tk)
    )
    # ≤1 MN tile/core: L0C DB unused (CPP still ctor-primes FIX_M(0/1);
    # loop WaitFlag(l0cPingPong_&1) stays on 0). Kill FixPipe SoftFlag if/else.
    # Primary: tiles ≤ BLOCK_DIM. Also allow tiles ≤ 15 (same bound as Up CV
    # id0) when BLOCK_DIM was historically compiled as 1.
    _dn_grid_m = (PROB_M + _tiling.l1_tm - 1) // _tiling.l1_tm if PROB_M > 0 else 0
    _dn_grid_n = (PROB_N + _tiling.l1_tn - 1) // _tiling.l1_tn if PROB_N > 0 else 0
    _dn_tiles = _dn_grid_m * _dn_grid_n
    _L0C_SLOT0_TASK_MAX = 15
    _l0c_slot0_only = tla.const_expr(
        (PROB_M > 0)
        and (PROB_N > 0)
        and (_dn_tiles > 0)
        and (not use_mtail)
        and (
            (_dn_tiles <= _L0C_SLOT0_TASK_MAX)
            or ((BLOCK_DIM > 0) and (_dn_tiles <= BLOCK_DIM))
        )
    )
    # Host grid in one swizzle group → kill last-tile rem + zigzag SSA.
    _dn_swizzle_count = (
        _dn_grid_m + _swizzle.SWIZZLE_OFFSET - 1
    ) // _swizzle.SWIZZLE_OFFSET
    _swizzle_one = tla.const_expr(
        (PROB_M > 0) and (PROB_N > 0) and (_dn_swizzle_count == 1)
    )
    # FH covers the only kL1 when work ≥ problem K (SwiGLU Down K=H).
    # Skip the empty k_l1 for that follows (removes fh_done/k_l1==0 tax).
    _fh_covers_all = tla.const_expr(
        _fh_l0_one and (PROB_K > 0) and (_work_tk >= PROB_K)
    )
    with tla.cube():
        # CPP BlockMmad ctor: Set MTE1_MTE2(0..3), M_MTE1(6/7), FIX_M(0/1).
        tla.set_flag(l1_0_available)
        tla.set_flag(l1_1_available)
        tla.set_flag(l1_2_available)
        tla.set_flag(l1_3_available)
        if tla.const_expr(USE_BIAS):
            # Credit-only: CPP folds CopyInC1/C2 under A/B HardEvents.
            tla.set_flag(l1_bias_data_ready)
            tla.set_flag(l1_bias_available)
            tla.set_flag(l0_bias_available)
        tla.set_flag(l0_0_available)
        tla.set_flag(l0_1_available)
        tla.set_flag(l0c0_available)
        tla.set_flag(l0c1_available)

        runtime_zero = tla.as_numeric(0)
        l1_buf_idx = runtime_zero
        l0_buf_idx = runtime_zero
        l0c_slot = runtime_zero
        # CPP abL1LoopCnt_ & 0x1 → biasBt_[baseN_ * biasBufId] ping-pong.
        bias_buf_idx = runtime_zero
        l1_bias_ptr1 = l1_bias_ptr
        l0_bias_ptr1 = l0_bias_ptr
        if tla.const_expr(USE_BIAS):
            l1_bias_ptr1 = l1_bias_ptr + _tiling.l1_tn
            l0_bias_ptr1 = l0_bias_ptr + _tiling.l1_tn
        # 0 = first MN tile on this core (matches prefetch target).
        mn_tile_i = runtime_zero
        block_idx = tla.arch.block_idx()
        block_num = tla.arch.block_num()
        # CPP last-wave mTail (see Up); rem tiles only, hetero full/split join.
        rem_tiles = total_blocks % block_num
        main_tiles = total_blocks - rem_tiles
        if tla.const_expr(use_mtail):
            expanded = main_tiles + rem_tiles * tail_cnt
        else:
            expanded = total_blocks
        block_range = tla.range(block_idx, expanded, block_num)
        for task_idx in block_range:
            block_linear = task_idx
            m_si = c0
            n_si = c0
            if tla.const_expr(use_mtail):
                if task_idx >= main_tiles:
                    local = task_idx - main_tiles
                    block_linear = main_tiles + local // tail_cnt
                    split_id = local % tail_cnt
                    m_si = split_id % m_tail
                    n_si = split_id // m_tail
            # M-window zig-zag (CPP ``UpdateMNTileIdx``); was row-major.
            if tla.const_expr(_swizzle_one):
                block_row = block_linear % grid_m
                block_col = block_linear // grid_m
            else:
                swizzle_tile_idx = block_linear // (_swizzle.SWIZZLE_OFFSET * grid_n)
                in_tile_idx = block_linear % (_swizzle.SWIZZLE_OFFSET * grid_n)
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
                    (grid_n - block_col - 1)
                    if (swizzle_tile_idx % 2 == 1)
                    else block_col
                )
            gm_a_full = tla.tile_view(
                gm_a, tla.make_shape(_tiling.l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_full = tla.tile_view(
                gm_b, tla.make_shape(k, _tiling.l1_tn), tla.make_coord(c0, block_col)
            )
            gm_c_full = tla.tile_view(
                gm_c,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(block_row, block_col),
            )
            if tla.const_expr(use_mtail):
                cur_tm = tla.as_numeric(_tiling.l1_tm)
                cur_tn = tla.as_numeric(_tiling.l1_tn)
                m_idx = c0
                n_idx = c0
                if task_idx >= main_tiles:
                    # CPP: CeilDiv(actual edge L1, mTailCnt/nTailCnt).
                    blk_m = gm_a_full.origin_shape[0]
                    blk_n = gm_c_full.origin_shape[1]
                    cur_tm = (blk_m + m_tail - 1) // m_tail
                    cur_tn = (blk_n + n_tail - 1) // n_tail
                    m_idx = m_si
                    n_idx = n_si
                gm_a_by_core = tla.tile_view(
                    gm_a_full, tla.make_shape(cur_tm, k), tla.make_coord(m_idx, c0)
                )
                gm_b_by_core = tla.tile_view(
                    gm_b_full, tla.make_shape(k, cur_tn), tla.make_coord(c0, n_idx)
                )
                gm_c_by_core = tla.tile_view(
                    gm_c_full,
                    tla.make_shape(cur_tm, cur_tn),
                    tla.make_coord(m_idx, n_idx),
                )
            else:
                gm_a_by_core = gm_a_full
                gm_b_by_core = gm_b_full
                gm_c_by_core = gm_c_full

            work_tm = gm_a_by_core.origin_shape[0]
            work_tn = gm_c_by_core.origin_shape[1]
            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + l1_tk_step - 1) // l1_tk_step

            l0c_ptr1 = l0c_ptr
            if tla.const_expr(_l0c_db and (not _l0c_slot0_only)):
                l0c_ptr1 = l0c_ptr + _l0c_slot_elems
            cur_l0c_ptr = l0c_ptr
            if tla.const_expr(not _l0c_slot0_only):
                cur_l0c_ptr = l0c_ptr if l0c_slot == c0 else l0c_ptr1
            l0_c = tla.make_tensor_like(cur_l0c_ptr, gm_c_by_core)
            # CPP DoBFullLoadOrAswt: Wait FIX_M only if enableL0cPingPong_
            # (l0cDB>1). Up DoAllLoc2Ub always waits — Down must not.
            if tla.const_expr(_l0c_db):
                if tla.const_expr(_l0c_slot0_only):
                    tla.wait_flag(l0c0_available)
                elif l0c_slot == c0:
                    tla.wait_flag(l0c0_available)
                else:
                    tla.wait_flag(l0c1_available)

            # Bias tensors bound here; CopyInC1 runs after CopyInA1 on first
            # kL1 (CPP DoBFullLoadOrAswt: A → bias → B under one SoftFlag).
            if tla.const_expr(USE_BIAS):
                # Absolute elem offset (same ABI as up gm_l1_abs_1d — avoid
                # tile_view rank-4 vs make_tensor rank-1 stub lock).
                # Pass clipped work_tn/cur_tn so N-tail DMA stays in-bounds.
                # Slot = abL1LoopCnt&1 (bias_buf_idx); stride = baseN (=l1_tn).
                _bias_col0 = block_col * _tiling.l1_tn
                if tla.const_expr(use_mtail):
                    gm_bias_by_core = gm_l1_abs_1d(
                        gm_bias, cur_tn, _bias_col0 + n_idx * cur_tn
                    )
                else:
                    gm_bias_by_core = gm_l1_abs_1d(gm_bias, work_tn, _bias_col0)
                cur_l1_bias_ptr = l1_bias_ptr if bias_buf_idx == c0 else l1_bias_ptr1
                cur_l0_bias_ptr = l0_bias_ptr if bias_buf_idx == c0 else l0_bias_ptr1
                l1_bias = tla.make_tensor_like(
                    cur_l1_bias_ptr, gm_bias_by_core, tla.arch.VectorLayout
                )
                l0_bias = tla.make_tensor_like(
                    cur_l0_bias_ptr, l1_bias, tla.arch.VectorLayout
                )

            # A∥B L1 ping-pong (CPP DoBFullLoadOrAswt): MTE2 copy → eager MTE2_MTE1
            # drain → L0 mmad loop; skinny-M may prefetch next K during L0.
            do_k_prefetch = tla.const_expr(skinny_down and (not l1_single))
            next_prefetched = runtime_zero
            # CPP isFirstLoopKL1Half: two fills (CeilAlign(kL1/2), rest),
            # then skip full k_l1==0. tile_view(shape=half_tk, coord=1) starts
            # at half_tk; origin crops to rest_tk (always rest<=half for CeilAlign).
            fh_done = runtime_zero
            # CPP isFirstLoopKL1Half: once per MN tile (not once per core).
            # Prior ``if mn_tile_i == c0`` was a dynamic SoftFlag/SSA tax
            # (HIVM icmp + empty else) and latent miss on 2nd+ tiles when
            # ``_fh_covers_all`` also skips the k_l1 for.
            if tla.const_expr(enable_first_half):
                if tla.const_expr(_fh_l0_one):
                    # ---- FH half0 @ L1/L0 slot0 (init_c) ----
                    gm_a_h0 = gm_l1_tile(gm_a_by_core, work_tm, half_tk, c0, c0)
                    gm_b_h0 = gm_l1_tile(gm_b_by_core, half_tk, work_tn, c0, c0)
                    l1_a_h0 = tla.make_tensor_like(l1a0_ptr, gm_a_h0)
                    l1_b_h0 = tla.make_tensor_like(l1b0_ptr, gm_b_h0)
                    tla.wait_flag(l1_0_available)
                    tla.copy(l1_a_h0, gm_a_h0)
                    if tla.const_expr(USE_BIAS):
                        tla.copy(l1_bias, gm_bias_by_core)
                    tla.copy(l1_b_h0, gm_b_h0)
                    tla.set_flag(l1_0_data_ready)
                    tla.wait_flag(l1_0_data_ready)
                    l1_a_h0_l0 = tla.tile_view(
                        l1_a_h0,
                        tla.make_shape(work_tm, _tiling.l0_tk),
                        tla.make_coord(c0, c0),
                    )
                    l1_b_h0_l0 = tla.tile_view(
                        l1_b_h0,
                        tla.make_shape(_tiling.l0_tk, work_tn),
                        tla.make_coord(c0, c0),
                    )
                    l0_a_h0 = tla.make_tensor_like(l0a0_ptr, l1_a_h0_l0)
                    l0_b_h0 = tla.make_tensor_like(l0b0_ptr, l1_b_h0_l0)
                    tla.wait_flag(l0_0_available)
                    tla.copy(l0_a_h0, l1_a_h0_l0)
                    if tla.const_expr(USE_BIAS):
                        tla.copy(l0_bias, l1_bias)
                    tla.copy(l0_b_h0, l1_b_h0_l0)
                    tla.set_flag(l0_0_data_ready)
                    tla.wait_flag(l0_0_data_ready)
                    if tla.const_expr(USE_BIAS):
                        tla.mmad(
                            l0_c,
                            l0_a_h0,
                            l0_b_h0,
                            init_c=True,
                            unit_flag=_uf_nf,
                            bias=l0_bias,
                        )
                    else:
                        tla.mmad(l0_c, l0_a_h0, l0_b_h0, init_c=True, unit_flag=_uf_nf)
                    tla.set_flag(l0_0_available)
                    tla.set_flag(l1_0_available)
                    # ---- FH half1 @ L1/L0 slot1 (accum) ----
                    gm_a_h1 = gm_l1_tile(gm_a_by_core, work_tm, half_tk, c0, c1)
                    gm_b_h1 = gm_l1_tile(gm_b_by_core, half_tk, work_tn, c1, c0)
                    l1_a_h1 = tla.make_tensor_like(l1a1_ptr, gm_a_h1)
                    l1_b_h1 = tla.make_tensor_like(l1b1_ptr, gm_b_h1)
                    tla.wait_flag(l1_1_available)
                    tla.copy(l1_a_h1, gm_a_h1)
                    tla.copy(l1_b_h1, gm_b_h1)
                    tla.set_flag(l1_1_data_ready)
                    tla.wait_flag(l1_1_data_ready)
                    l1_a_h1_l0 = tla.tile_view(
                        l1_a_h1,
                        tla.make_shape(work_tm, _tiling.l0_tk),
                        tla.make_coord(c0, c0),
                    )
                    l1_b_h1_l0 = tla.tile_view(
                        l1_b_h1,
                        tla.make_shape(_tiling.l0_tk, work_tn),
                        tla.make_coord(c0, c0),
                    )
                    l0_a_h1 = tla.make_tensor_like(l0a1_ptr, l1_a_h1_l0)
                    l0_b_h1 = tla.make_tensor_like(l0b1_ptr, l1_b_h1_l0)
                    tla.wait_flag(l0_1_available)
                    tla.copy(l0_a_h1, l1_a_h1_l0)
                    tla.copy(l0_b_h1, l1_b_h1_l0)
                    tla.set_flag(l0_1_data_ready)
                    tla.wait_flag(l0_1_data_ready)
                    if tla.const_expr(_fh_covers_all):
                        tla.mmad(
                            l0_c,
                            l0_a_h1,
                            l0_b_h1,
                            init_c=False,
                            unit_flag=_uf_fin,
                        )
                    else:
                        tla.mmad(
                            l0_c,
                            l0_a_h1,
                            l0_b_h1,
                            init_c=False,
                            unit_flag=_uf_nf,
                        )
                    tla.set_flag(l0_1_available)
                    tla.set_flag(l1_1_available)
                    # Two toggles land back on slot0 (matches loop FH).
                    l1_buf_idx = c0
                    if tla.const_expr(not L0_SINGLE):
                        l0_buf_idx = c0
                    fh_done = c1
                else:
                    for k_half in tla.range(c0, 2, c1):
                        gm_a_by_l1 = gm_l1_tile(
                            gm_a_by_core, work_tm, half_tk, c0, k_half
                        )
                        gm_b_by_l1 = gm_l1_tile(
                            gm_b_by_core, half_tk, work_tn, k_half, c0
                        )
                        l1_a = tla.make_tensor_like(
                            l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr,
                            gm_a_by_l1,
                        )
                        l1_b = tla.make_tensor_like(
                            l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr,
                            gm_b_by_l1,
                        )
                        if l1_buf_idx == c0:
                            tla.wait_flag(l1_0_available)
                        else:
                            tla.wait_flag(l1_1_available)
                        tla.copy(l1_a, gm_a_by_l1)
                        # CPP: CopyInC1 on iter0 only, same MTE1_MTE2 credit.
                        if tla.const_expr(USE_BIAS):
                            if k_half == c0:
                                tla.copy(l1_bias, gm_bias_by_core)
                        tla.copy(l1_b, gm_b_by_l1)
                        if l1_buf_idx == c0:
                            tla.set_flag(l1_0_data_ready)
                        else:
                            tla.set_flag(l1_1_data_ready)
                        if l1_buf_idx == c0:
                            tla.wait_flag(l1_0_data_ready)
                        else:
                            tla.wait_flag(l1_1_data_ready)
                        k_l0_count_h = (
                            l1_a.origin_shape[1] + _tiling.l0_tk - 1
                        ) // _tiling.l0_tk
                        for k_l0 in tla.range(c0, k_l0_count_h, c1):
                            l1_a_by_l0 = tla.tile_view(
                                l1_a,
                                tla.make_shape(work_tm, _tiling.l0_tk),
                                tla.make_coord(c0, k_l0),
                            )
                            l1_b_by_l0 = tla.tile_view(
                                l1_b,
                                tla.make_shape(_tiling.l0_tk, work_tn),
                                tla.make_coord(k_l0, c0),
                            )
                            l0_a = tla.make_tensor_like(
                                l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr,
                                l1_a_by_l0,
                            )
                            l0_b = tla.make_tensor_like(
                                l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr,
                                l1_b_by_l0,
                            )
                            # CPP: Wait M_MTE1 → A2 → C2(NeedBias) → B2 →
                            # Set/Wait MTE1_M → Mmad (no bias SoftFlag).
                            if l0_buf_idx == c0:
                                tla.wait_flag(l0_0_available)
                            else:
                                tla.wait_flag(l0_1_available)
                            tla.copy(l0_a, l1_a_by_l0)
                            init_c = True if k_half == 0 and k_l0 == 0 else False
                            if tla.const_expr(USE_BIAS):
                                if init_c:
                                    tla.copy(l0_bias, l1_bias)
                            tla.copy(l0_b, l1_b_by_l0)
                            if l0_buf_idx == c0:
                                tla.set_flag(l0_0_data_ready)
                                tla.wait_flag(l0_0_data_ready)
                            else:
                                tla.set_flag(l0_1_data_ready)
                                tla.wait_flag(l0_1_data_ready)
                            uf = _uf_nf
                            if tla.const_expr(_fh_covers_all):
                                if k_half == c1 and (k_l0 + c1) >= k_l0_count_h:
                                    uf = _uf_fin
                            if tla.const_expr(USE_BIAS):
                                if init_c:
                                    tla.mmad(
                                        l0_c,
                                        l0_a,
                                        l0_b,
                                        init_c=init_c,
                                        unit_flag=uf,
                                        bias=l0_bias,
                                    )
                                else:
                                    tla.mmad(
                                        l0_c,
                                        l0_a,
                                        l0_b,
                                        init_c=init_c,
                                        unit_flag=uf,
                                    )
                            else:
                                tla.mmad(
                                    l0_c,
                                    l0_a,
                                    l0_b,
                                    init_c=init_c,
                                    unit_flag=uf,
                                )
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
                        l1_buf_idx = c1 - l1_buf_idx
                        # CPP abL1LoopCnt_++ (each FH half is one L1 iter).
                        if tla.const_expr(USE_BIAS):
                            bias_buf_idx = c1 - bias_buf_idx
                    fh_done = c1

            # When FH already ate the only kL1, skip this for (no fh_done tax).
            if tla.const_expr(not _fh_covers_all):
                for k_l1 in tla.range(c0, k_l1_count, c1):
                    if fh_done == c1 and k_l1 == c0:
                        pass
                    else:
                        gm_a_by_l1 = gm_l1_tile(
                            gm_a_by_core, work_tm, l1_tk_step, c0, k_l1
                        )
                        gm_b_by_l1 = gm_l1_tile(
                            gm_b_by_core, l1_tk_step, work_tn, k_l1, c0
                        )
                        l1_a = tla.make_tensor_like(
                            l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_by_l1
                        )
                        l1_b = tla.make_tensor_like(
                            l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_by_l1
                        )
                        if next_prefetched == c0:
                            if l1_buf_idx == c0:
                                tla.wait_flag(l1_0_available)
                            else:
                                tla.wait_flag(l1_1_available)
                            # CPP: CopyInA1 → CopyInC1(iter0) → CopyInB1.
                            tla.copy(l1_a, gm_a_by_l1)
                            if tla.const_expr(USE_BIAS):
                                if k_l1 == c0:
                                    tla.copy(l1_bias, gm_bias_by_core)
                            tla.copy(l1_b, gm_b_by_l1)
                            # CPP: Set+Wait MTE2_MTE1 immediately (eager).
                            if l1_buf_idx == c0:
                                tla.set_flag(l1_0_data_ready)
                            else:
                                tla.set_flag(l1_1_data_ready)
                            if l1_buf_idx == c0:
                                tla.wait_flag(l1_0_data_ready)
                            else:
                                tla.wait_flag(l1_1_data_ready)
                        else:
                            # Only when (non-CPP).
                            if l1_buf_idx == c0:
                                tla.wait_flag(l1_0_data_ready)
                            else:
                                tla.wait_flag(l1_1_data_ready)
                            next_prefetched = runtime_zero

                        k_l0_count = (
                            l1_a.origin_shape[1] + _tiling.l0_tk - 1
                        ) // _tiling.l0_tk
                        k_l0_range = tla.range(c0, k_l0_count, c1)

                        for k_l0 in k_l0_range:
                            l1_a_by_l0 = tla.tile_view(
                                l1_a,
                                tla.make_shape(work_tm, _tiling.l0_tk),
                                tla.make_coord(c0, k_l0),
                            )
                            l1_b_by_l0 = tla.tile_view(
                                l1_b,
                                tla.make_shape(_tiling.l0_tk, work_tn),
                                tla.make_coord(k_l0, c0),
                            )

                            l0_a = tla.make_tensor_like(
                                l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_by_l0
                            )
                            l0_b = tla.make_tensor_like(
                                l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_by_l0
                            )
                            # CPP: Wait M_MTE1 → A2 → C2(NeedBias) → B2 →
                            # Set/Wait MTE1_M → Mmad (no bias SoftFlag).
                            # When first-half ran, k_l1==0 is skipped so init_c stays
                            # False here; BT was already folded on k_half==0.
                            if l0_buf_idx == c0:
                                tla.wait_flag(l0_0_available)
                            else:
                                tla.wait_flag(l0_1_available)
                            tla.copy(l0_a, l1_a_by_l0)
                            init_c = True if k_l1 == 0 and k_l0 == 0 else False
                            if tla.const_expr(USE_BIAS):
                                if init_c:
                                    tla.copy(l0_bias, l1_bias)
                            tla.copy(l0_b, l1_b_by_l0)
                            if l0_buf_idx == c0:
                                tla.set_flag(l0_0_data_ready)
                                tla.wait_flag(l0_0_data_ready)
                            else:
                                tla.set_flag(l0_1_data_ready)
                                tla.wait_flag(l0_1_data_ready)
                            uf = _uf_nf
                            if (k_l1 + c1) >= k_l1_count and (k_l0 + c1) >= k_l0_count:
                                uf = _uf_fin
                            if tla.const_expr(USE_BIAS):
                                if init_c:
                                    tla.mmad(
                                        l0_c,
                                        l0_a,
                                        l0_b,
                                        init_c=init_c,
                                        unit_flag=uf,
                                        bias=l0_bias,
                                    )
                                else:
                                    tla.mmad(
                                        l0_c, l0_a, l0_b, init_c=init_c, unit_flag=uf
                                    )
                            else:
                                tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=uf)
                            if l0_buf_idx == c0:
                                tla.set_flag(l0_0_available)
                            else:
                                tla.set_flag(l0_1_available)
                            if tla.const_expr(not L0_SINGLE):
                                l0_buf_idx = c1 - l0_buf_idx

                            # Non-CPP opt-in: prefetch K+1 during first L0 of K.
                            if (
                                tla.const_expr(do_k_prefetch)
                                and k_l0 == c0
                                and (k_l1 + c1) < k_l1_count
                            ):
                                k_next = k_l1 + c1
                                other = c1 - l1_buf_idx
                                gm_a_next = gm_l1_tile(
                                    gm_a_by_core, work_tm, l1_tk_step, c0, k_next
                                )
                                gm_b_next = gm_l1_tile(
                                    gm_b_by_core, l1_tk_step, work_tn, k_next, c0
                                )
                                l1_a_next = tla.make_tensor_like(
                                    l1a0_ptr if (other == c0) else l1a1_ptr, gm_a_next
                                )
                                l1_b_next = tla.make_tensor_like(
                                    l1b0_ptr if (other == c0) else l1b1_ptr, gm_b_next
                                )
                                if other == c0:
                                    tla.wait_flag(l1_0_available)
                                else:
                                    tla.wait_flag(l1_1_available)
                                tla.copy(l1_a_next, gm_a_next)
                                tla.copy(l1_b_next, gm_b_next)
                                # Set only — consumer waits on next k_l1 (∥ L0).
                                if other == c0:
                                    tla.set_flag(l1_0_data_ready)
                                else:
                                    tla.set_flag(l1_1_data_ready)
                                next_prefetched = c1

                        # CPP: SetFlag MTE1_MTE2(l1BufId) after full kL0Iter.
                        if l1_buf_idx == c0:
                            tla.set_flag(l1_0_available)
                        else:
                            tla.set_flag(l1_1_available)
                        l1_buf_idx = c1 - l1_buf_idx
                        # CPP abL1LoopCnt_++ → next tile/kL1 uses other BT slot.
                        if tla.const_expr(USE_BIAS):
                            bias_buf_idx = c1 - bias_buf_idx

            # CPP DoBFullLoadOrAswt: Set/Wait M_FIX + Set FIX_M only when
            # enableL0cPingPong_. !l0cDB → bare FixPipe (ctor/dtor still
            # prime/drain FIX_M 0/1).
            if tla.const_expr(_l0c_db):
                if tla.const_expr(_l0c_slot0_only):
                    tla.set_flag(l0c0_data_ready)
                    tla.wait_flag(l0c0_data_ready)
                elif l0c_slot == c0:
                    tla.set_flag(l0c0_data_ready)
                    tla.wait_flag(l0c0_data_ready)
                else:
                    tla.set_flag(l0c1_data_ready)
                    tla.wait_flag(l0c1_data_ready)
            tla.copy(
                gm_c_by_core,
                l0_c,
                tla.params.CopyL0C2DstParams(relu_enable=False, unit_flag=_uf_fin),
            )
            if tla.const_expr(_l0c_db):
                if tla.const_expr(_l0c_slot0_only):
                    tla.set_flag(l0c0_available)
                elif l0c_slot == c0:
                    tla.set_flag(l0c0_available)
                else:
                    tla.set_flag(l0c1_available)
                if tla.const_expr(not _l0c_slot0_only):
                    l0c_slot = c1 - l0c_slot
            mn_tile_i = mn_tile_i + c1

        # CPP ~BlockMmad: Wait MTE1_MTE2(0..3), M_MTE1(6/7), FIX_M(0/1).
        tla.wait_flag(l1_0_available)
        tla.wait_flag(l1_1_available)
        tla.wait_flag(l1_2_available)
        tla.wait_flag(l1_3_available)
        tla.wait_flag(l0_0_available)
        tla.wait_flag(l0_1_available)
        tla.wait_flag(l0c0_available)
        tla.wait_flag(l0c1_available)
        if tla.const_expr(USE_BIAS):
            tla.wait_flag(l1_bias_data_ready)
            tla.wait_flag(l1_bias_available)
            tla.wait_flag(l0_bias_available)
