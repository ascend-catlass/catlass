# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# FFN Down StreamK — full-port CPP BuiltInStreamKScheduler (tileIdx / offsetWorkspace / AIV).
from __future__ import annotations

import catlass.tla as tla

from ffn.gm_l1 import gm_l1_abs_1d
from ffn.tiling import SwizzleParams, TilingParams, _HALF_L1_ELEMS

# Compile-time AIV reduce knobs (match TLA streamk demo).
AIV_TILE_M = int("16")
if AIV_TILE_M not in (8, 16, 32, 64):
    AIV_TILE_M = 16
AIV_SUB_BLOCK_NUM = 2
AIV_REG_M = 1
AIV_REG_N = 64
# CPP StreamK workspace tile (ffn_arch35_tiling / BLOCK_BASE_M×N).
SK_WS_BASE_M = 256
SK_WS_BASE_N = 256
# Pack-kCnt UB budget (CPP BlockEpilogueStreamK). 112KB → 7×16KB @ tn=256
# for wave-0; when k_cnt > slots, remaining banks stream through pack[1]
# with MTE2_V + V_MTE2 (pre-pack_v2 one-slice loop). Cast/out overlays
# pack[1+] after reduce; leaves room beside Up C_UB pad.
SK_AIV_BULK_BUDGET_BYTES = 112 * 1024


def _aiv_n_pad(l1_tn: int) -> int:
    """Ceil-align N to AIV_REG_N (CPP baseN may be only 16-aligned)."""
    return ((int(l1_tn) + AIV_REG_N - 1) // AIV_REG_N) * AIV_REG_N


def _aiv_pack_slots(l1_tn: int) -> int:
    tile = AIV_TILE_M * _aiv_n_pad(l1_tn)
    slots = SK_AIV_BULK_BUDGET_BYTES // max(1, tile * 4)
    return max(2, int(slots))


# Host must reserve this many UB bytes from Up when Down StreamK is live.
# Pack f32 slots + cast/out tile (device still allocates out; CPP overlays).
def sk_aiv_ub_reserve_bytes(l1_tn: int, out_elem_bytes: int = 2, l1_tm: int = 0) -> int:
    del l1_tm
    tile = AIV_TILE_M * _aiv_n_pad(l1_tn)
    slots = _aiv_pack_slots(l1_tn)
    raw = slots * tile * 4 + tile * int(out_elem_bytes)
    return (raw + 255) & ~255


HALF_L1_ELEMS = _HALF_L1_ELEMS


@tla.jit
def stage_down_streamk(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    gm_workspace: tla.Tensor,
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
    block_dim: tla.Constexpr[int],
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
) -> None:
    """Down StreamK: CPP ``BuiltInStreamKScheduler`` AIC + AIV epilogue.

    Device-side mirror of ``BlockSchedulerStreamKBuiltIn`` / cmct
    ``kernel_matmul_streamk`` (tileIdx, DP+SK swap, offsetWorkspace) and
    ``block_epilogue_streamk`` (UpdateAivBasicIndex / bank reduce). Workspace
    is ``usedCoreNum × 256 × 256`` (1× banks; no linearized dual-slot).
    ``block_dim`` must equal launch ``block_num``. MN window uses
    ``_swizzle.SWIZZLE_OFFSET`` (= CPP ``WINDOW_LEN`` / ``MAIN_WINDOW`` = 4).
    """
    c0 = 0
    c1 = 1
    # CPP MAIN_WINDOW / BlockSchedulerStreamKBuiltIn.WINDOW_LEN.
    sk_window = _swizzle.SWIZZLE_OFFSET

    dtype_gm_c = gm_c.ptr.dtype
    DTYPE_C = tla.Float32  # L0C accumulator only

    # Carve dual L1 from single merge alloc (G1: 256×256×256 in 256×256×512 box).
    # Unified pool ``(total, tk=1, tn=0)``: carve pingA|pingB|pongA|pongB.
    buf_unified = tla.const_expr(BUF_L1_TK == 1 and BUF_L1_TN == 0 and BUF_L1_TM > 0)
    buf_tm = BUF_L1_TM if BUF_L1_TM > 0 else _tiling.l1_tm
    buf_tn = BUF_L1_TN if BUF_L1_TN > 0 else _tiling.l1_tn
    buf_tk = BUF_L1_TK if BUF_L1_TK > 0 else _tiling.l1_tk
    stage_a = _tiling.l1_tm * _tiling.l1_tk
    stage_b = _tiling.l1_tk * _tiling.l1_tn
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
    # CPP StreamK / ASWT: dual only with full mL1*kL1 slots; no permanent half-K.
    use_half_l1 = tla.const_expr(
        buf_unified and can_split and (HALF_L1_ELEMS + stage_a + stage_b <= BUF_L1_TM)
    )
    if tla.const_expr(can_split):
        if tla.const_expr(buf_unified):
            l1_base = l1a0_ptr
            if tla.const_expr(use_half_l1):
                l1b0_ptr = l1_base + stage_a
                l1a1_ptr = l1_base + HALF_L1_ELEMS
                l1b1_ptr = l1_base + HALF_L1_ELEMS + stage_a
            else:
                l1b0_ptr = l1_base + stage_a
                l1a1_ptr = l1_base + stage_a + stage_b
                l1b1_ptr = l1_base + stage_a + stage_b + stage_a
        else:
            l1a1_ptr = l1a0_ptr + stage_a
            l1b1_ptr = l1b0_ptr + stage_b
    use_l1_pp = tla.const_expr((not L1_SINGLE) or can_split)
    l1_tk_step = _tiling.l1_tk

    aiv_m_chunks = AIV_TILE_M // AIV_REG_M
    # CPP baseN is 16-aligned; AIV_REG_N=64 — pad UB width, keep published l1_tn.
    aiv_n_pad = ((_tiling.l1_tn + AIV_REG_N - 1) // AIV_REG_N) * AIV_REG_N
    aiv_n_chunks = aiv_n_pad // AIV_REG_N

    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    k = gm_a.origin_shape[1]

    # CPP BuiltInStreamKScheduler tile accounting (device inline).
    m_tile_num = (m + _tiling.l1_tm - 1) // _tiling.l1_tm
    n_tile_num = (n + _tiling.l1_tn - 1) // _tiling.l1_tn
    sk_single = _tiling.sk_single_core_k
    sk_k_tile_num = (k + sk_single - 1) // sk_single
    used = block_dim
    total_mn = m_tile_num * n_tile_num
    tail_mn = total_mn % used
    total_mn_dp = total_mn - tail_mn
    tail_sk_total = tail_mn * sk_k_tile_num
    tile_num = total_mn_dp + tail_sk_total
    last_loop = (tile_num + used - 1) // used
    # All-SK scene when the first tile already lands in the last loop.
    is_sk_scene_0 = ((c0 + c1 + used - 1) // used) == last_loop
    get_block_num = tile_num if tile_num < used else used
    aiv_rows_per_tile = _tiling.l1_tm // AIV_TILE_M

    # SoftFlags: reuse WithoutQue HardEvent IDs after Up SyncAll (=CPP
    # sequential BlockMmad lifetimes; same pattern as down_basic).
    # SK keeps A∥B as *distinct* SoftFlags (cannot alias A+B under one ID —
    # unlike Basic — or the independent schedule hangs). Map the 4 L1 flags
    # onto ctor IDs 0..3; L0 A/B onto 4/5 + WithoutQue 6/7. Bias trio is
    # parked at L1 id=4 / L0 id=0 in the kernel (does not collide).
    l1a0_data_ready = tla.flag("sk_l1a0_data_ready", tla.arch.MTE2, tla.arch.MTE1, id=0)
    l1a1_data_ready = tla.flag("sk_l1a1_data_ready", tla.arch.MTE2, tla.arch.MTE1, id=1)
    l1b0_data_ready = tla.flag("sk_l1b0_data_ready", tla.arch.MTE2, tla.arch.MTE1, id=2)
    l1b1_data_ready = tla.flag("sk_l1b1_data_ready", tla.arch.MTE2, tla.arch.MTE1, id=3)
    l1a0_available = tla.flag("sk_l1a0_available", tla.arch.MTE1, tla.arch.MTE2, id=0)
    l1a1_available = tla.flag("sk_l1a1_available", tla.arch.MTE1, tla.arch.MTE2, id=1)
    l1b0_available = tla.flag("sk_l1b0_available", tla.arch.MTE1, tla.arch.MTE2, id=2)
    l1b1_available = tla.flag("sk_l1b1_available", tla.arch.MTE1, tla.arch.MTE2, id=3)
    l0a0_available = tla.flag("sk_l0a0_available", tla.arch.CUBE, tla.arch.MTE1, id=6)
    l0a1_available = tla.flag("sk_l0a1_available", tla.arch.CUBE, tla.arch.MTE1, id=7)
    l0b0_available = tla.flag("sk_l0b0_available", tla.arch.CUBE, tla.arch.MTE1, id=4)
    l0b1_available = tla.flag("sk_l0b1_available", tla.arch.CUBE, tla.arch.MTE1, id=5)
    # Self-fence after A+B L0 copy; must NOT share WithoutQue 6/7 (or SK
    # L0 A/B available 4/5) — concurrent Set/Wait would corrupt credits.
    l0_ab_data_ready = tla.flag(
        "sk_l0_ab_data_ready", tla.arch.MTE1, tla.arch.CUBE, id=1
    )
    l0c_available = tla.flag("sk_l0c_available", tla.arch.FIX, tla.arch.CUBE, id=0)
    # Cube BT SoftFlags come from the kernel (shared Up↔Down; SoftFlag budget).
    # AIC→AIV handoff (mode 2) plus an all-AIV barrier (mode 0).
    aic_finish = tla.cross_flag("ffn_sk_aic_finish", mode=2)
    aiv_ibarrier = tla.cross_flag("ffn_sk_aiv_ibarrier", mode=0)

    # CPP BlockEpilogueStreamK: pack kCnt GM→UB then ONE MTE2_V, then Add.
    aiv_ub_n = aiv_n_pad
    aiv_ub_tile_elems = AIV_TILE_M * aiv_ub_n
    aiv_pack_slots = SK_AIV_BULK_BUDGET_BYTES // (aiv_ub_tile_elems * 4)
    if tla.const_expr(aiv_pack_slots < 2):
        aiv_pack_slots = 2
    aiv_pack_ptr = tla.allocate(
        aiv_pack_slots * aiv_ub_tile_elems, DTYPE_C, tla.AddressSpace.ub, 256
    )
    # Cast/out overlays pack[1] after reduce (CPP reuses packed UB).
    aiv_out_ptr = tla.allocate(aiv_ub_tile_elems, dtype_gm_c, tla.AddressSpace.ub, 256)
    aiv_ub_layout = tla.make_layout(
        tla.make_shape(AIV_TILE_M, aiv_ub_n),
        tla.make_stride(aiv_ub_n, c1),
        layoutTag=tla.arch.RowMajor,
    )
    # Slot-0 is the accumulator (CPP ubAddTensor); remaining slots are pack.
    aiv_acc_ub = tla.make_tensor(aiv_pack_ptr, aiv_ub_layout)
    aiv_out_ub = tla.make_tensor(aiv_out_ptr, aiv_ub_layout)

    # Per AIV-sub SoftFlags (MIX 1:2): dual subs must not share.
    aiv_loaded_s0 = tla.flag("sk_aiv_loaded_s0", tla.arch.MTE2, tla.arch.VECTOR, id=0)
    aiv_loaded_s1 = tla.flag("sk_aiv_loaded_s1", tla.arch.MTE2, tla.arch.VECTOR, id=1)
    # V→MTE2: required between Add and the next GM→UB when streaming
    # remaining K banks one-by-one (pre-pack_v2 aiv_v2m; pack_v2 wave-0
    # needs only one MTE2_V so it omits this).
    aiv_v2m_s0 = tla.flag("sk_aiv_v2m_s0", tla.arch.VECTOR, tla.arch.MTE2, id=0)
    aiv_v2m_s1 = tla.flag("sk_aiv_v2m_s1", tla.arch.VECTOR, tla.arch.MTE2, id=1)
    aiv_done_s0 = tla.flag("sk_aiv_done_s0", tla.arch.VECTOR, tla.arch.MTE3, id=0)
    aiv_done_s1 = tla.flag("sk_aiv_done_s1", tla.arch.VECTOR, tla.arch.MTE3, id=1)

    with tla.cube():
        tla.set_flag(l1a0_available)
        tla.set_flag(l1a1_available)
        tla.set_flag(l1b1_available)
        tla.set_flag(l0a0_available)
        tla.set_flag(l0a1_available)
        tla.set_flag(l0b0_available)
        tla.set_flag(l0b1_available)
        tla.set_flag(l0c_available)
        if tla.const_expr(USE_BIAS):
            tla.set_flag(l1_bias_available)
            tla.set_flag(l0_bias_available)

        l1_buf_idx = c0
        l0_buf_idx = c0
        # CPP abL1LoopCnt_ & 0x1 → biasBt_[baseN_ * biasBufId] ping-pong.
        bias_buf_idx = c0
        l1_bias_ptr1 = l1_bias_ptr
        l0_bias_ptr1 = l0_bias_ptr
        if tla.const_expr(USE_BIAS):
            l1_bias_ptr1 = l1_bias_ptr + _tiling.l1_tn
            l0_bias_ptr1 = l0_bias_ptr + _tiling.l1_tn
        block_idx = tla.arch.block_idx()
        tla.set_flag(l1b0_available)

        # Idle cores: release paired AIV; still drain SoftFlags at end.
        if block_idx >= get_block_num:
            tla.cross_core_set_flag(aic_finish, tla.arch.FIX)
        else:
            tile_range = tla.range(block_idx, tile_num, used)
            for tile_idx in tile_range:
                # DP+SK preload swap (kernel_matmul_streamk) when not all-SK.
                tmp = tile_idx
                if not is_sk_scene_0:
                    cur_loop = (tile_idx + c1 + used - 1) // used
                    if tile_idx % used < tail_sk_total:
                        if cur_loop == (last_loop - c1):
                            tmp = tile_idx + used
                        elif cur_loop == last_loop:
                            tmp = tile_idx - used

                # Whether this mapped tile index falls in the StreamK scene.
                is_sk_now = ((tmp + c1 + used - 1) // used) == last_loop
                cur_k_tile_num = sk_k_tile_num if is_sk_now else c1

                # MN tile index from mapped tmp; WINDOW_LEN=4 (CPP MAIN_WINDOW).
                # Pre-init for TLA dynamic-if SSA.
                k_tile_idx = c0
                mn_idx = tmp // cur_k_tile_num
                if is_sk_now:
                    k_tile_idx = (tmp % used) % cur_k_tile_num
                    mn_idx = (tmp % used) // cur_k_tile_num + total_mn_dp

                main_window = sk_window if m_tile_num > sk_window else m_tile_num
                main_row = m_tile_num // main_window - c1
                tail_window = m_tile_num - main_row * main_window
                row_idx = mn_idx // n_tile_num // main_window
                m_tile_idx = row_idx * main_window + mn_idx % main_window
                n_tile_idx = (mn_idx // main_window) % n_tile_num
                if row_idx >= main_row:
                    row_idx = main_row
                    tail_index = mn_idx - main_row * main_window * n_tile_num
                    m_tile_idx = main_row * main_window + tail_index % tail_window
                    n_tile_idx = (tail_index // tail_window) % n_tile_num
                if row_idx % 2 != 0:
                    n_tile_idx = n_tile_num - c1 - n_tile_idx

                # real_k / k_start (get_single_core_shape + sk K origin).
                tail_single_k = k - (cur_k_tile_num - c1) * sk_single
                real_k = sk_single
                if k_tile_idx == (cur_k_tile_num - c1):
                    real_k = tail_single_k
                # CPP ptr+offset: k_start = k_tile_idx * sk_single (may be
                # unaligned to l1_tk). Express with tla.get_tile (element coord).
                k_start = k_tile_idx * sk_single

                # offsetWorkspace bank (SK only).
                ws_bank = ((tmp % used) // sk_k_tile_num) * sk_k_tile_num + k_tile_idx
                gm_ws_slot = tla.tile_view(
                    gm_workspace,
                    tla.make_shape(SK_WS_BASE_M, SK_WS_BASE_N),
                    tla.make_coord(ws_bank, c0),
                )

                gm_a_by_core = tla.tile_view(
                    gm_a,
                    tla.make_shape(_tiling.l1_tm, k),
                    tla.make_coord(m_tile_idx, c0),
                )
                gm_b_by_core = tla.tile_view(
                    gm_b,
                    tla.make_shape(k, _tiling.l1_tn),
                    tla.make_coord(c0, n_tile_idx),
                )
                gm_c_by_core = tla.tile_view(
                    gm_c,
                    tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                    tla.make_coord(m_tile_idx, n_tile_idx),
                )
                gm_ws_by_core = tla.tile_view(
                    gm_ws_slot,
                    tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                    tla.make_coord(c0, c0),
                )
                l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

                k_l1_count = (real_k + l1_tk_step - 1) // l1_tk_step
                k_l1_range = tla.range(c0, k_l1_count, c1)
                do_k_prefetch = tla.const_expr(use_l1_pp)
                next_prefetched = c0

                # NeedBias only on first K of this MN (k_tile_idx==0).
                if tla.const_expr(USE_BIAS):
                    # Clip to this MN tile's N (origin may be < l1_tn on N-tail).
                    # Slot = abL1LoopCnt&1 (bias_buf_idx); stride = baseN.
                    _sk_bias_tn = gm_c_by_core.origin_shape[1]
                    gm_bias_by_core = gm_l1_abs_1d(
                        gm_bias,
                        _sk_bias_tn,
                        n_tile_idx * _tiling.l1_tn,
                    )
                    cur_l1_bias_ptr = (
                        l1_bias_ptr if bias_buf_idx == c0 else l1_bias_ptr1
                    )
                    cur_l0_bias_ptr = (
                        l0_bias_ptr if bias_buf_idx == c0 else l0_bias_ptr1
                    )
                    l1_bias = tla.make_tensor_like(
                        cur_l1_bias_ptr, gm_bias_by_core, tla.arch.VectorLayout
                    )
                    l0_bias = tla.make_tensor_like(
                        cur_l0_bias_ptr, l1_bias, tla.arch.VectorLayout
                    )
                    if k_tile_idx == 0:
                        tla.wait_flag(l1_bias_available)
                        tla.copy(l1_bias, gm_bias_by_core)
                        tla.set_flag(l1_bias_data_ready)

                for k_l1_i in k_l1_range:
                    k_elem = k_start + k_l1_i * l1_tk_step
                    # CPP block_mmad_streamk: last L1 uses rem K (blkK - iter*kL1).
                    cur_l1_k = l1_tk_step
                    if k_l1_i == (k_l1_count - c1):
                        cur_l1_k = real_k - (k_l1_count - c1) * l1_tk_step
                    # Published l1_tm/tn (constexpr). get_tile crops rem via origin;
                    # do not pass origin_shape SSA extents here — that broke SK.
                    gm_a_l1 = tla.get_tile(
                        gm_a_by_core,
                        tla.make_coord(c0, k_elem),
                        tla.make_shape(_tiling.l1_tm, cur_l1_k),
                    )
                    gm_b_l1 = tla.get_tile(
                        gm_b_by_core,
                        tla.make_coord(k_elem, c0),
                        tla.make_shape(cur_l1_k, _tiling.l1_tn),
                    )

                    l1_a = tla.make_tensor_like(
                        l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_l1
                    )
                    l1_b = tla.make_tensor_like(
                        l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_l1
                    )
                    l1_drain_pending = c0
                    if next_prefetched == c0:
                        if l1_buf_idx == c0:
                            tla.wait_flag(l1a0_available)
                        else:
                            tla.wait_flag(l1a1_available)
                        tla.copy(l1_a, gm_a_l1)
                        if l1_buf_idx == c0:
                            tla.set_flag(l1a0_data_ready)
                        else:
                            tla.set_flag(l1a1_data_ready)

                        if l1_buf_idx == c0:
                            tla.wait_flag(l1b0_available)
                        else:
                            tla.wait_flag(l1b1_available)
                        tla.copy(l1_b, gm_b_l1)
                        if l1_buf_idx == c0:
                            tla.set_flag(l1b0_data_ready)
                        else:
                            tla.set_flag(l1b1_data_ready)
                        l1_drain_pending = c1
                    else:
                        # Prefetched during prior K's first L0; one drain only
                        # (do not wait again at k_l0==0 — flag already cleared).
                        if l1_buf_idx == c0:
                            tla.wait_flag(l1a0_data_ready)
                            tla.wait_flag(l1b0_data_ready)
                        else:
                            tla.wait_flag(l1a1_data_ready)
                            tla.wait_flag(l1b1_data_ready)
                        next_prefetched = c0

                    k_l0_count = (
                        l1_a.origin_shape[1] + _tiling.l0_tk - 1
                    ) // _tiling.l0_tk
                    k_l0_range = tla.range(c0, k_l0_count, c1)

                    for k_l0 in k_l0_range:
                        l1_a_l0 = tla.tile_view(
                            l1_a,
                            tla.make_shape(_tiling.l0_tm, _tiling.l0_tk),
                            tla.make_coord(c0, k_l0),
                        )
                        l1_b_l0 = tla.tile_view(
                            l1_b,
                            tla.make_shape(_tiling.l0_tk, _tiling.l0_tn),
                            tla.make_coord(k_l0, c0),
                        )

                        l0_a = tla.make_tensor_like(
                            l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_l0
                        )
                        l0_b = tla.make_tensor_like(
                            l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_l0
                        )
                        if k_l0 == 0 and l1_drain_pending == c1:
                            if l1_buf_idx == c0:
                                tla.wait_flag(l1a0_data_ready)
                            else:
                                tla.wait_flag(l1a1_data_ready)

                        if l0_buf_idx == c0:
                            tla.wait_flag(l0a0_available)
                        else:
                            tla.wait_flag(l0a1_available)
                        tla.copy(l0_a, l1_a_l0)
                        if k_l0 == k_l0_count - 1:
                            if l1_buf_idx == c0:
                                tla.set_flag(l1a0_available)
                            else:
                                tla.set_flag(l1a1_available)

                        if k_l0 == 0 and l1_drain_pending == c1:
                            if l1_buf_idx == c0:
                                tla.wait_flag(l1b0_data_ready)
                            else:
                                tla.wait_flag(l1b1_data_ready)
                            l1_drain_pending = c0
                        if l0_buf_idx == c0:
                            tla.wait_flag(l0b0_available)
                        else:
                            tla.wait_flag(l0b1_available)
                        tla.copy(l0_b, l1_b_l0)
                        if k_l0 == k_l0_count - 1:
                            if l1_buf_idx == c0:
                                tla.set_flag(l1b0_available)
                            else:
                                tla.set_flag(l1b1_available)

                        tla.set_flag(l0_ab_data_ready)
                        tla.wait_flag(l0_ab_data_ready)

                        unit_flag = (
                            0b11
                            if (k_l1_i == k_l1_count - 1) and (k_l0 == k_l0_count - 1)
                            else 0b10
                        )
                        init_c = True if k_l1_i == 0 and k_l0 == 0 else False
                        # CPP NeedBias: BT only on k_tile_idx==0 + first K.
                        if tla.const_expr(USE_BIAS):
                            if k_tile_idx == 0 and init_c:
                                tla.wait_flag(l1_bias_data_ready)
                                tla.wait_flag(l0_bias_available)
                                tla.copy(l0_bias, l1_bias)
                                tla.set_flag(l1_bias_available)
                                tla.mmad(
                                    l0_c,
                                    l0_a,
                                    l0_b,
                                    init_c=init_c,
                                    unit_flag=unit_flag,
                                    bias=l0_bias,
                                )
                                tla.set_flag(l0_bias_available)
                            else:
                                tla.mmad(
                                    l0_c,
                                    l0_a,
                                    l0_b,
                                    init_c=init_c,
                                    unit_flag=unit_flag,
                                )
                        else:
                            tla.mmad(
                                l0_c,
                                l0_a,
                                l0_b,
                                init_c=init_c,
                                unit_flag=unit_flag,
                            )
                        if l0_buf_idx == c0:
                            tla.set_flag(l0a0_available)
                            tla.set_flag(l0b0_available)
                        else:
                            tla.set_flag(l0a1_available)
                            tla.set_flag(l0b1_available)
                        if tla.const_expr(not L0_SINGLE):
                            l0_buf_idx = c1 - l0_buf_idx

                        # Overlap next K GM→L1 with current L0/mmad (PP only).
                        if (
                            tla.const_expr(do_k_prefetch)
                            and k_l0 == c0
                            and (k_l1_i + c1) < k_l1_count
                        ):
                            k_elem_n = k_start + (k_l1_i + c1) * l1_tk_step
                            other = c1 - l1_buf_idx
                            cur_l1_k_n = l1_tk_step
                            if (k_l1_i + c1) == (k_l1_count - c1):
                                cur_l1_k_n = real_k - (k_l1_count - c1) * l1_tk_step
                            gm_a_next = tla.get_tile(
                                gm_a_by_core,
                                tla.make_coord(c0, k_elem_n),
                                tla.make_shape(_tiling.l1_tm, cur_l1_k_n),
                            )
                            gm_b_next = tla.get_tile(
                                gm_b_by_core,
                                tla.make_coord(k_elem_n, c0),
                                tla.make_shape(cur_l1_k_n, _tiling.l1_tn),
                            )
                            l1_a_next = tla.make_tensor_like(
                                l1a0_ptr if (other == c0) else l1a1_ptr, gm_a_next
                            )
                            l1_b_next = tla.make_tensor_like(
                                l1b0_ptr if (other == c0) else l1b1_ptr, gm_b_next
                            )
                            if other == c0:
                                tla.wait_flag(l1a0_available)
                                tla.wait_flag(l1b0_available)
                            else:
                                tla.wait_flag(l1a1_available)
                                tla.wait_flag(l1b1_available)
                            tla.copy(l1_a_next, gm_a_next)
                            tla.copy(l1_b_next, gm_b_next)
                            if other == c0:
                                tla.set_flag(l1a0_data_ready)
                                tla.set_flag(l1b0_data_ready)
                            else:
                                tla.set_flag(l1a1_data_ready)
                                tla.set_flag(l1b1_data_ready)
                            next_prefetched = c1
                    if tla.const_expr(use_l1_pp):
                        l1_buf_idx = c1 - l1_buf_idx
                    # CPP abL1LoopCnt_++ → next tile/kL1 uses other BT slot.
                    if tla.const_expr(USE_BIAS):
                        bias_buf_idx = c1 - bias_buf_idx

                if is_sk_now:
                    tla.copy(
                        gm_ws_by_core,
                        l0_c,
                        tla.params.CopyL0C2DstParams(unit_flag=0b11),
                    )
                else:
                    tla.copy(
                        gm_c_by_core,
                        l0_c,
                        tla.params.CopyL0C2DstParams(unit_flag=0b11),
                    )

                # Last tile for this core → wake AIV (PIPE_FIX).
                if tmp + used >= tile_num:
                    tla.cross_core_set_flag(aic_finish, tla.arch.FIX)

        tla.wait_flag(l1a0_available)
        tla.wait_flag(l1a1_available)
        tla.wait_flag(l1b0_available)
        tla.wait_flag(l1b1_available)
        tla.wait_flag(l0a0_available)
        tla.wait_flag(l0a1_available)
        tla.wait_flag(l0b0_available)
        tla.wait_flag(l0b1_available)
        tla.wait_flag(l0c_available)
        if tla.const_expr(USE_BIAS):
            tla.wait_flag(l1_bias_available)
            tla.wait_flag(l0_bias_available)

    with tla.vector():
        # Wait for the paired AIC, then for every AIC through the all-AIV barrier.
        tla.cross_core_wait_flag(aic_finish, tla.arch.MTE2)
        tla.cross_core_set_flag(aiv_ibarrier, tla.arch.MTE2)
        tla.cross_core_wait_flag(aiv_ibarrier, tla.arch.MTE2)
        # MIX: block_idx=AIC id, sub_block_idx=AIV half → CPP GetBlockIdx().
        aiv_id = tla.arch.block_idx()
        aiv_sub = tla.arch.sub_block_idx()
        aiv_global = aiv_id * AIV_SUB_BLOCK_NUM + aiv_sub

        # lastLoopTotalCnt * taskRation — only SK-tail AIVs reduce.
        last_loop_total = (total_mn % used) * sk_k_tile_num
        if aiv_global < last_loop_total * AIV_SUB_BLOCK_NUM:
            k_cnt = sk_k_tile_num
            labor = AIV_SUB_BLOCK_NUM * k_cnt
            index_params = aiv_global // labor
            k_cnt_index = aiv_global % labor

            # UpdateAivBasicIndex MN swizzle (batch=1).
            mn_index = (index_params + (total_mn - total_mn % used)) % total_mn
            main_window = sk_window if m_tile_num > sk_window else m_tile_num
            main_row = m_tile_num // main_window - c1
            tail_window = m_tile_num - main_row * main_window
            row_idx = mn_index // n_tile_num // main_window
            m_tile_idx = row_idx * main_window + mn_index % main_window
            n_tile_idx = (mn_index // main_window) % n_tile_num
            if row_idx >= main_row:
                row_idx = main_row
                tail_index = mn_index - main_row * main_window * n_tile_num
                m_tile_idx = main_row * main_window + tail_index % tail_window
                n_tile_idx = (tail_index // tail_window) % n_tile_num
            if row_idx % 2 != 0:
                n_tile_idx = n_tile_num - c1 - n_tile_idx

            cur_m = _tiling.l1_tm
            if m_tile_idx == (m_tile_num - c1):
                cur_m = m - (m_tile_num - c1) * _tiling.l1_tm
            # N-tail: keep store/GM tile at published l1_tn (constexpr). A dynamic
            # cur_n extent on copy broke StreamK even when rem==l1_tn (SSA extent).
            # Tail safety is origin crop on this tile_view (same as pre-fix).

            base_bank = index_params * k_cnt
            m_loops = (cur_m + AIV_TILE_M - 1) // AIV_TILE_M
            chunk_per = (m_loops + labor - 1) // labor
            # Prefer min over assignment-if (avoids scf.if SSA tax; microbench).
            # AST-rewritten scalar min (arith.minsi); not vector tla.min.
            chunk_lo = min(k_cnt_index * chunk_per, m_loops)
            chunk_hi = min(chunk_lo + chunk_per, m_loops)

            for aiv_m_idx in tla.range(chunk_lo, chunk_hi, c1):
                aiv_c_row = m_tile_idx * aiv_rows_per_tile + aiv_m_idx
                aiv_c_col = n_tile_idx
                aiv_gm_c_tile = tla.tile_view(
                    gm_c,
                    tla.make_shape(AIV_TILE_M, _tiling.l1_tn),
                    tla.make_coord(aiv_c_row, aiv_c_col),
                )

                # CPP BlockEpilogueStreamK: pack → one MTE2_V → Add for the
                # first min(k_cnt, slots) banks. When k_cnt < slots, only the
                # loaded slots are reduced (do not Add unloaded pack[i]).
                # Remaining banks [slots, k_cnt): one-slice loop with MTE2_V +
                # V_MTE2 (range empty when k_cnt≤slots).
                wave0_n = min(aiv_pack_slots, k_cnt)
                for aiv_si in tla.range(0, wave0_n, 1):
                    _pk_ws = tla.tile_view(
                        gm_workspace,
                        tla.make_shape(SK_WS_BASE_M, SK_WS_BASE_N),
                        tla.make_coord(base_bank + aiv_si, c0),
                    )
                    _pk_tile = tla.tile_view(
                        _pk_ws,
                        tla.make_shape(AIV_TILE_M, aiv_ub_n),
                        tla.make_coord(aiv_m_idx, c0),
                    )
                    _pk_dst = tla.make_tensor(
                        aiv_pack_ptr + aiv_si * aiv_ub_tile_elems,
                        aiv_ub_layout,
                    )
                    tla.copy(_pk_dst, _pk_tile)

                if aiv_sub == c0:
                    tla.set_flag(aiv_loaded_s0)
                    tla.wait_flag(aiv_loaded_s0)
                else:
                    tla.set_flag(aiv_loaded_s1)
                    tla.wait_flag(aiv_loaded_s1)

                for aiv_ai in tla.range(c1, wave0_n, 1):
                    _pk_src = tla.make_tensor(
                        aiv_pack_ptr + aiv_ai * aiv_ub_tile_elems,
                        aiv_ub_layout,
                    )
                    with tla.vec.func(mode="simd"):
                        for _aiv_rm in tla.range(0, aiv_m_chunks, 1):
                            for _aiv_rn in tla.range(0, aiv_n_chunks, 1):
                                aiv_acc_chunk = tla.tile_view(
                                    aiv_acc_ub,
                                    tla.make_shape(AIV_REG_M, AIV_REG_N),
                                    tla.make_coord(_aiv_rm, _aiv_rn),
                                )
                                aiv_src_chunk = tla.tile_view(
                                    _pk_src,
                                    tla.make_shape(AIV_REG_M, AIV_REG_N),
                                    tla.make_coord(_aiv_rm, _aiv_rn),
                                )
                                aiv_acc_chunk.store(
                                    tla.add(
                                        aiv_acc_chunk.load(),
                                        aiv_src_chunk.load(),
                                        mask=tla.create_mask(
                                            pattern=tla.mask.ALL,
                                            dtype=DTYPE_C,
                                        ),
                                    ),
                                    mask=tla.create_mask(
                                        pattern=tla.mask.ALL, dtype=DTYPE_C
                                    ),
                                )

                # Fence wave-0 Add before tail reuses pack[1] as MTE2 dst.
                if aiv_pack_slots < k_cnt:
                    if aiv_sub == c0:
                        tla.set_flag(aiv_v2m_s0)
                        tla.wait_flag(aiv_v2m_s0)
                    else:
                        tla.set_flag(aiv_v2m_s1)
                        tla.wait_flag(aiv_v2m_s1)

                # Tail banks [slots, k_cnt): pack[1] scratch + V_MTE2 fence.
                for aiv_ki in tla.range(aiv_pack_slots, k_cnt, 1):
                    _xw_ws = tla.tile_view(
                        gm_workspace,
                        tla.make_shape(SK_WS_BASE_M, SK_WS_BASE_N),
                        tla.make_coord(base_bank + aiv_ki, c0),
                    )
                    _xw_tile = tla.tile_view(
                        _xw_ws,
                        tla.make_shape(AIV_TILE_M, aiv_ub_n),
                        tla.make_coord(aiv_m_idx, c0),
                    )
                    _xw_dst = tla.make_tensor(
                        aiv_pack_ptr + c1 * aiv_ub_tile_elems,
                        aiv_ub_layout,
                    )
                    tla.copy(_xw_dst, _xw_tile)
                    if aiv_sub == c0:
                        tla.set_flag(aiv_loaded_s0)
                        tla.wait_flag(aiv_loaded_s0)
                    else:
                        tla.set_flag(aiv_loaded_s1)
                        tla.wait_flag(aiv_loaded_s1)
                    _xw_src = tla.make_tensor(
                        aiv_pack_ptr + c1 * aiv_ub_tile_elems,
                        aiv_ub_layout,
                    )
                    with tla.vec.func(mode="simd"):
                        for _xw_rm in tla.range(0, aiv_m_chunks, 1):
                            for _xw_rn in tla.range(0, aiv_n_chunks, 1):
                                _xw_acc = tla.tile_view(
                                    aiv_acc_ub,
                                    tla.make_shape(AIV_REG_M, AIV_REG_N),
                                    tla.make_coord(_xw_rm, _xw_rn),
                                )
                                _xw_sc = tla.tile_view(
                                    _xw_src,
                                    tla.make_shape(AIV_REG_M, AIV_REG_N),
                                    tla.make_coord(_xw_rm, _xw_rn),
                                )
                                _xw_acc.store(
                                    tla.add(
                                        _xw_acc.load(),
                                        _xw_sc.load(),
                                        mask=tla.create_mask(
                                            pattern=tla.mask.ALL,
                                            dtype=DTYPE_C,
                                        ),
                                    ),
                                    mask=tla.create_mask(
                                        pattern=tla.mask.ALL, dtype=DTYPE_C
                                    ),
                                )
                    if aiv_sub == c0:
                        tla.set_flag(aiv_v2m_s0)
                        tla.wait_flag(aiv_v2m_s0)
                    else:
                        tla.set_flag(aiv_v2m_s1)
                        tla.wait_flag(aiv_v2m_s1)

                aiv_store_ub = aiv_acc_ub
                if tla.const_expr(dtype_gm_c != DTYPE_C):
                    if tla.const_expr(dtype_gm_c == tla.Float16):
                        aiv_cast_even = tla.params.CastParams(
                            reg_slot=tla.params.RegSlot.ZERO,
                            sat_mode=tla.params.SatMode.NOSAT,
                            round_mode=tla.params.RoundMode.CAST_FLOOR,
                        )
                    else:
                        aiv_cast_even = tla.params.CastParams(
                            reg_slot=tla.params.RegSlot.ZERO,
                            sat_mode=tla.params.SatMode.NOSAT,
                            round_mode=tla.params.RoundMode.CAST_ROUND,
                        )
                    aiv_pack_store = tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_PACK_B32
                    )
                    aiv_cast_vl_loops = aiv_m_chunks * aiv_n_chunks
                    aiv_acc_1d = tla.make_tensor(
                        aiv_pack_ptr,
                        tla.make_layout(
                            tla.make_shape(aiv_ub_tile_elems),
                            tla.make_stride(c1),
                        ),
                    )
                    aiv_out_1d = tla.make_tensor(
                        aiv_out_ptr,
                        tla.make_layout(
                            tla.make_shape(aiv_ub_tile_elems),
                            tla.make_stride(c1),
                        ),
                    )
                    with tla.vec.func(mode="simd"):
                        aiv_cast_mask = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=DTYPE_C
                        )
                        aiv_store_mask = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=dtype_gm_c
                        )
                        for aiv_cast_vl in tla.range(0, aiv_cast_vl_loops, 1):
                            aiv_cast_src = tla.tile_view(
                                aiv_acc_1d,
                                tla.make_shape(AIV_REG_N),
                                tla.make_coord(aiv_cast_vl),
                            )
                            aiv_cast_dst = tla.tile_view(
                                aiv_out_1d,
                                tla.make_shape(AIV_REG_N),
                                tla.make_coord(aiv_cast_vl),
                            )
                            aiv_cast_h = aiv_cast_src.load().to(
                                dtype_gm_c,
                                aiv_cast_even,
                                aiv_cast_mask,
                            )
                            aiv_cast_dst.store(
                                aiv_cast_h,
                                aiv_pack_store,
                                mask=aiv_store_mask,
                            )
                    aiv_store_ub = aiv_out_ub

                if aiv_sub == c0:
                    tla.set_flag(aiv_done_s0)
                    tla.wait_flag(aiv_done_s0)
                else:
                    tla.set_flag(aiv_done_s1)
                    tla.wait_flag(aiv_done_s1)
                aiv_store_n = tla.tile_view(
                    aiv_store_ub,
                    tla.make_shape(AIV_TILE_M, _tiling.l1_tn),
                    tla.make_coord(c0, c0),
                )
                tla.copy(aiv_gm_c_tile, aiv_store_n)
                # Multi-strip labor (m_loops > labor → chunk≥2, l1_tm≳240):
                # next strip reuses pack/out UB; without this fence MTE3 store
                # races the next cast/load (basic_mmad_streamk same pattern).
                # Single-strip AIVs (M≤224 @ k_cnt=7) are unaffected functionally.
                tla.pipe_barrier(tla.pipes.ALL)

        # Light fence: last MTE3 store visible before kernel exit.
        tla.pipe_barrier(tla.pipes.MTE3)
