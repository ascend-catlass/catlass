# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# FFN Phase-1 main MIX kernel: Up(+Act) → hard SyncAll → Down Basic / StreamK.
from __future__ import annotations

import catlass.tla as tla

from ffn.buffers import (
    allocate_gemm_pingpong,
    merge_tiling,
    _merge_tiling_seq_capacity,
)
from ffn.config import (
    ACT_GELU,
    ACT_RAW,
    ACT_SILU,
    ACT_SWIGLU,
    ACT_SWIGLU_DUAL,
    BIAS_CUBE_BT,
    BIAS_NONE,
    DOWN_FULL_A,
    DOWN_FULL_B,
    DOWN_STREAMK,
    LAYOUT_LINEAR,
    SWIGLU_DUAL,
    SWIGLU_SINGLE,
    UP_L0C2UB,
    FfnConfig,
)
from ffn.stages.down_basic import stage_down_basic
from ffn.stages.down_full_a import stage_down_full_a
from ffn.stages.down_full_b import stage_down_full_b
from ffn.stages.down_streamk import stage_down_streamk
from ffn.stages.up_l0c2ub_silu import (
    ACC_SLOT_ELEMS,
    C_UB_ELEMS,
    EVG_UB_STAGES,
    stage_up_l0c2ub_silu,
)
from ffn.sync_hard import stage_barrier_hard
from ffn.tiling import SwizzleParams, TilingParams, l0_alloc_is_single


@tla.kernel
def ffn_basic_kernel(
    gm_x: tla.Tensor,
    gm_w1_t: tla.Tensor,
    gm_w1_up: tla.Tensor,
    gm_w2_t: tla.Tensor,
    gm_y: tla.Tensor,
    gm_hidden: tla.Tensor,
    gm_wks_up: tla.Tensor,
    gm_gate: tla.Tensor,
    gm_sk_ws: tla.Tensor,
    gm_b1: tla.Tensor,
    gm_b1_up: tla.Tensor,
    gm_b2: tla.Tensor,
    cfg: tla.Constexpr[FfnConfig],
    tiling_up: tla.Constexpr[TilingParams],
    tiling_down: tla.Constexpr[TilingParams],
    swizzle_up: tla.Constexpr[SwizzleParams],
    swizzle_down: tla.Constexpr[SwizzleParams],
    UB_SLOT_ELEMS: tla.Constexpr[int],
    SIMD_LANES: tla.Constexpr[int],
    BLOCK_DIM: tla.Constexpr[int],
    ACC_ELEMS: tla.Constexpr[int] = 0,
) -> None:
    """Single MIX kernel FFN (act via ``cfg``; Cube BT when ``BIAS_CUBE_BT``).

    Tensor convention (Linear, host supplies transposed weights as row-major):
      gm_x:      [M, K]
      gm_w1_t:   [K, H] / SwiGLU single [K,2H] packed / dual gate half [K,H]
      gm_w1_up:  SwiGLU dual up half [K,H] (dummy alias of gm_w1_t otherwise)
      gm_w2_t:   [H, N]
      gm_hidden: [M, H]
      gm_wks_up: [M, 2H] SwiGLU single L0C geom (dummy otherwise)
      gm_gate:   [M, H] FP32 SwiGLU dual gate (dummy otherwise)
      gm_sk_ws:  StreamK FP32 scratch (dummy if Basic)
      gm_b1:     Up bias [H] / SwiGLU single [2H] packed / dual gate [H]
      gm_b1_up:  SwiGLU dual up bias [H] (dummy alias of gm_b1 otherwise)
      gm_b2:     Down bias [N]
      gm_y:      [M, N]

    Host always passes bias tensors (1-elem dummy when ``BIAS_NONE``). Cube BT
    path: GM→L1→biasbuf on first-K ``mmad(..., bias=)`` per MN block.
    """
    aiv_arrive = tla.cross_flag("ffn_aiv_arrive", mode=4)
    aic_ffts = tla.cross_flag("ffn_aic_ffts", mode=0)
    aic_done = tla.cross_flag("ffn_aic_done", mode=2)
    # Mid-barrier flags only for SwiGLU dual (avoid burning SoftFlag IDs on single).
    if tla.const_expr(cfg.act == ACT_SWIGLU and cfg.swiglu_mode == SWIGLU_DUAL):
        aiv_arrive_dual = tla.cross_flag("ffn_aiv_arrive_dual", mode=4)
        aic_ffts_dual = tla.cross_flag("ffn_aic_ffts_dual", mode=0)
        aic_done_dual = tla.cross_flag("ffn_aic_done_dual", mode=2)
    else:
        aiv_arrive_dual = aiv_arrive
        aic_ffts_dual = aic_ffts
        aic_done_dual = aic_done

    # Sequential Up→Down: unified pool max(stage dual/single) (keeps stage K);
    # fall back to rectangular merge when the pair does not fit that way.
    tiling_buf_seq = _merge_tiling_seq_capacity(tiling_up, tiling_down)
    tiling_buf = (
        tiling_buf_seq
        if tiling_buf_seq is not None
        else merge_tiling(tiling_up, tiling_down)
    )
    # Host-side: matches allocate_gemm_pingpong L0 alias (large-M l0_tk=128).
    l0_single = (
        l0_alloc_is_single(tiling_buf)
        or l0_alloc_is_single(tiling_up)
        or (l0_alloc_is_single(tiling_down))
    )
    l0c_elems = max(
        int(tiling_up.l0_tm) * int(tiling_up.l0_tn),
        int(tiling_down.l0_tm) * int(tiling_down.l0_tn),
    )
    # CPP dbL0C=2: shared L0C half-pingpong when 2×max(tile) fits L0C.
    # Prefer host-published stage flags; fall back to capacity if either is 1
    # because the shared max tile may still fit (or not).
    from ffn.tiling_aswt import _DB as _L0C_DB_FACTOR
    from ffn.tiling_aswt import _FP32 as _L0C_FP32
    from ffn.tiling_aswt import _L0C as _L0C_BYTES

    # Alloc capacity (shared L0C pool). SoftFlag policy stays per-stage like
    # CPP upL0cDB / downL0cDB — never force Down into M↔FIX pingpong solely
    # because Up needed db=2 (or vice versa).
    l0c_db = (
        2
        if (
            max(int(tiling_up.l0c_db), int(tiling_down.l0c_db)) > 1
            and l0c_elems * _L0C_FP32 * _L0C_DB_FACTOR <= _L0C_BYTES
        )
        else 1
    )
    from dataclasses import replace as _replace_tiling

    def _l0c_sf(stage_db: int) -> int:
        return 2 if (int(stage_db) > 1 and l0c_db > 1) else 1

    tiling_up = _replace_tiling(tiling_up, l0c_db=_l0c_sf(tiling_up.l0c_db))
    tiling_down = _replace_tiling(tiling_down, l0c_db=_l0c_sf(tiling_down.l0c_db))
    USE_BIAS = tla.const_expr(cfg.bias_mode == BIAS_CUBE_BT)
    # SoftFlag IDs are global per pipe-pair in one MIX kernel (max 7/8).
    # Up completes before Down (SyncAll), so share one Cube-BT bias trio.
    # Basic WithoutQue path: trio is entry/exit credit only (CopyInC1/C2 ride
    # under A/B SoftFlags). full_a/full_b/SK may still use them mid-loop.
    #
    # Park bias SoftFlags on free IDs so they never steal WithoutQue ctor
    # slots (MTE1_MTE2 0..3, M_MTE1 6/7) that Up/Down Basic/SK reuse after
    # SyncAll — mirrors CPP sequential BlockMmad HardEvent lifetimes.
    if tla.const_expr(USE_BIAS):
        l1_bias_data_ready = tla.flag(
            "ffn_bias_l1_data_ready", tla.arch.MTE2, tla.arch.MTE1, id=4
        )
        l1_bias_available = tla.flag(
            "ffn_bias_l1_available", tla.arch.MTE1, tla.arch.MTE2, id=4
        )
        l0_bias_available = tla.flag(
            "ffn_bias_l0_available", tla.arch.CUBE, tla.arch.MTE1, id=0
        )
    else:
        l1_bias_data_ready = None
        l1_bias_available = None
        l0_bias_available = None
    bias_elems = (
        max(int(tiling_up.l1_tn), int(tiling_down.l1_tn))
        if tla.const_expr(USE_BIAS)
        else 0
    )
    (
        l1a0_ptr,
        l1a1_ptr,
        l1b0_ptr,
        l1b1_ptr,
        l0a0_ptr,
        l0a1_ptr,
        l0b0_ptr,
        l0b1_ptr,
        l0c_ptr,
        l1_single,
        l1_bias_ptr,
        l0_bias_ptr,
    ) = allocate_gemm_pingpong(
        gm_x.ptr.dtype,
        gm_w1_t.ptr.dtype,
        tiling_buf,
        L0C_ELEMS=l0c_elems if tiling_buf_seq is not None else 0,
        BIAS_L1_ELEMS=bias_elems,
        L0C_DB=l0c_db,
    )
    # Stages carve against actual pool bytes. Unified alloc shrinks by
    # ``bias_elems*4`` (2× fp32 slots → fp16 elem units); pass as BUF_L1_TM.
    _buf_l1_tm = tiling_buf.l1_tm
    if tla.const_expr(
        USE_BIAS and tiling_buf.l1_tk == 1 and tiling_buf.l1_tn == 0 and bias_elems > 0
    ):
        _buf_l1_tm = tiling_buf.l1_tm - bias_elems * 4

    # Shared Up UB (one pool). Stage does not allocate — TLA counts every
    # allocate in the MIX kernel (dual Raw+Up would otherwise OOM).
    _need_l0c2ub = tla.const_expr(
        cfg.up_path == UP_L0C2UB
        and (cfg.act == ACT_SILU or cfg.act == ACT_GELU or cfg.act == ACT_SWIGLU)
    )
    if tla.const_expr(
        _need_l0c2ub and cfg.act == ACT_SWIGLU and cfg.swiglu_mode == SWIGLU_DUAL
    ):
        # Straight allocates only (no if/else) — TLA sums dead-branch allocs.
        _acc_slot = ACC_ELEMS if tla.const_expr(ACC_ELEMS > 0) else ACC_SLOT_ELEMS
        _ub_acc0 = tla.allocate(_acc_slot, tla.Float32, tla.AddressSpace.ub, 256)
        _ub_acc1 = _ub_acc0
        _ub_out0 = tla.allocate(
            UB_SLOT_ELEMS, gm_hidden.ptr.dtype, tla.AddressSpace.ub, 256
        )
        _ub_out1 = _ub_out0
        _ub_gate = tla.allocate(_acc_slot, tla.Float32, tla.AddressSpace.ub, 256)
        _use_out_pp = False
    elif tla.const_expr(_need_l0c2ub):
        # Single FixPipe-acc (=CPP WithoutQue).
        _acc_slot = ACC_SLOT_ELEMS
        if tla.const_expr(ACC_ELEMS > 0):
            _acc_slot = ACC_ELEMS
        _ub_acc0 = tla.allocate(_acc_slot, tla.Float32, tla.AddressSpace.ub, 256)
        _ub_acc1 = _ub_acc0
        _use_out_pp = tla.const_expr(
            EVG_UB_STAGES >= 2 and (ACC_ELEMS <= 0 or ACC_ELEMS <= C_UB_ELEMS)
        )
        _ub_out0 = tla.allocate(
            UB_SLOT_ELEMS, gm_hidden.ptr.dtype, tla.AddressSpace.ub, 256
        )
        _ub_out1 = _ub_out0
        if tla.const_expr(_use_out_pp):
            _ub_out1 = tla.allocate(
                UB_SLOT_ELEMS, gm_hidden.ptr.dtype, tla.AddressSpace.ub, 256
            )
        _ub_gate = _ub_acc0

    if tla.const_expr(cfg.act == ACT_SILU):
        if tla.const_expr(cfg.up_path == UP_L0C2UB):
            stage_up_l0c2ub_silu(
                gm_x,
                gm_w1_t,
                gm_hidden,
                gm_hidden,
                l1a0_ptr,
                l1a1_ptr,
                l1b0_ptr,
                l1b1_ptr,
                l0a0_ptr,
                l0a1_ptr,
                l0b0_ptr,
                l0b1_ptr,
                l0c_ptr,
                tiling_up,
                swizzle_up,
                UB_SLOT_ELEMS,
                SIMD_LANES,
                l1_single,
                ACT_SILU,
                _buf_l1_tm,
                tiling_buf.l1_tn,
                tiling_buf.l1_tk,
                ACC_ELEMS,
                l0_single,
                gm_gate,
                _ub_acc0,
                _ub_acc1,
                _ub_out0,
                _ub_out1,
                _ub_gate,
                _use_out_pp,
                0,
                gm_b1,
                l1_bias_ptr,
                l0_bias_ptr,
                USE_BIAS,
                l1_bias_data_ready,
                l1_bias_available,
                l0_bias_available,
            )
    elif tla.const_expr(cfg.act == ACT_GELU):
        if tla.const_expr(cfg.up_path == UP_L0C2UB):
            stage_up_l0c2ub_silu(
                gm_x,
                gm_w1_t,
                gm_hidden,
                gm_hidden,
                l1a0_ptr,
                l1a1_ptr,
                l1b0_ptr,
                l1b1_ptr,
                l0a0_ptr,
                l0a1_ptr,
                l0b0_ptr,
                l0b1_ptr,
                l0c_ptr,
                tiling_up,
                swizzle_up,
                UB_SLOT_ELEMS,
                SIMD_LANES,
                l1_single,
                ACT_GELU,
                _buf_l1_tm,
                tiling_buf.l1_tn,
                tiling_buf.l1_tk,
                ACC_ELEMS,
                l0_single,
                gm_gate,
                _ub_acc0,
                _ub_acc1,
                _ub_out0,
                _ub_out1,
                _ub_gate,
                _use_out_pp,
                0,
                gm_b1,
                l1_bias_ptr,
                l0_bias_ptr,
                USE_BIAS,
                l1_bias_data_ready,
                l1_bias_available,
                l0_bias_available,
            )
    elif tla.const_expr(cfg.act == ACT_SWIGLU):
        if tla.const_expr(cfg.up_path == UP_L0C2UB):
            if tla.const_expr(cfg.swiglu_mode == SWIGLU_DUAL):
                # CPP Linear FFN dual path (ASCEND_IS_AIC Down):
                #   RawGateUp → SyncAll → SwigluUp → SyncAll → if AIC: Down
                # DSL: stage0 (ACT_RAW) → barrier_hard (SyncAll #1) →
                #   stage1 (ACT_SWIGLU_DUAL) → barrier before Down (SyncAll #2).
                # SoftFlag≤8: dual stages share one SoftFlag set (single-buffer).
                # CrossCore pool max 16: one shared CV finish set for both Ups
                # (plus SyncAll flags above) — do not let each stage allocate 6.
                up_aic0 = tla.cross_flag("ffn_dual_up_aic0", mode=4)
                up_aic1 = tla.cross_flag("ffn_dual_up_aic1", mode=4)
                up_aic2 = tla.cross_flag("ffn_dual_up_aic2", mode=4)
                up_aiv0 = tla.cross_flag("ffn_dual_up_aiv0", mode=4)
                up_aiv1 = tla.cross_flag("ffn_dual_up_aiv1", mode=4)
                up_aiv2 = tla.cross_flag("ffn_dual_up_aiv2", mode=4)
                stage_up_l0c2ub_silu(
                    gm_x,
                    gm_w1_t,
                    gm_gate,
                    gm_gate,
                    l1a0_ptr,
                    l1a1_ptr,
                    l1b0_ptr,
                    l1b1_ptr,
                    l0a0_ptr,
                    l0a1_ptr,
                    l0b0_ptr,
                    l0b1_ptr,
                    l0c_ptr,
                    tiling_up,
                    swizzle_up,
                    UB_SLOT_ELEMS,
                    SIMD_LANES,
                    l1_single,
                    ACT_RAW,
                    _buf_l1_tm,
                    tiling_buf.l1_tn,
                    tiling_buf.l1_tk,
                    ACC_ELEMS,
                    l0_single,
                    gm_gate,
                    _ub_acc0,
                    _ub_acc1,
                    _ub_out0,
                    _ub_out1,
                    _ub_gate,
                    _use_out_pp,
                    0,
                    gm_b1,
                    l1_bias_ptr,
                    l0_bias_ptr,
                    USE_BIAS,
                    l1_bias_data_ready,
                    l1_bias_available,
                    l0_bias_available,
                    cfg.swiglu_h,
                    cfg.swiglu_k,
                    cfg.swiglu_m,
                    BLOCK_DIM,
                    True,
                    up_aic0,
                    up_aic1,
                    up_aic2,
                    up_aiv0,
                    up_aiv1,
                    up_aiv2,
                )
                stage_barrier_hard(aiv_arrive_dual, aic_ffts_dual, aic_done_dual)
                stage_up_l0c2ub_silu(
                    gm_x,
                    gm_w1_up,
                    gm_hidden,
                    gm_hidden,
                    l1a0_ptr,
                    l1a1_ptr,
                    l1b0_ptr,
                    l1b1_ptr,
                    l0a0_ptr,
                    l0a1_ptr,
                    l0b0_ptr,
                    l0b1_ptr,
                    l0c_ptr,
                    tiling_up,
                    swizzle_up,
                    UB_SLOT_ELEMS,
                    SIMD_LANES,
                    l1_single,
                    ACT_SWIGLU_DUAL,
                    _buf_l1_tm,
                    tiling_buf.l1_tn,
                    tiling_buf.l1_tk,
                    ACC_ELEMS,
                    l0_single,
                    gm_gate,
                    _ub_acc0,
                    _ub_acc1,
                    _ub_out0,
                    _ub_out1,
                    _ub_gate,
                    _use_out_pp,
                    1,
                    gm_b1_up,
                    l1_bias_ptr,
                    l0_bias_ptr,
                    USE_BIAS,
                    l1_bias_data_ready,
                    l1_bias_available,
                    l0_bias_available,
                    cfg.swiglu_h,
                    cfg.swiglu_k,
                    cfg.swiglu_m,
                    BLOCK_DIM,
                    True,
                    up_aic0,
                    up_aic1,
                    up_aic2,
                    up_aiv0,
                    up_aiv1,
                    up_aiv2,
                )
            elif tla.const_expr(cfg.swiglu_mode == SWIGLU_SINGLE):
                stage_up_l0c2ub_silu(
                    gm_x,
                    gm_w1_t,
                    gm_hidden,
                    gm_wks_up,
                    l1a0_ptr,
                    l1a1_ptr,
                    l1b0_ptr,
                    l1b1_ptr,
                    l0a0_ptr,
                    l0a1_ptr,
                    l0b0_ptr,
                    l0b1_ptr,
                    l0c_ptr,
                    tiling_up,
                    swizzle_up,
                    UB_SLOT_ELEMS,
                    SIMD_LANES,
                    l1_single,
                    ACT_SWIGLU,
                    _buf_l1_tm,
                    tiling_buf.l1_tn,
                    tiling_buf.l1_tk,
                    ACC_ELEMS,
                    l0_single,
                    gm_gate,
                    _ub_acc0,
                    _ub_acc1,
                    _ub_out0,
                    _ub_out1,
                    _ub_gate,
                    _use_out_pp,
                    0,
                    gm_b1,
                    l1_bias_ptr,
                    l0_bias_ptr,
                    USE_BIAS,
                    l1_bias_data_ready,
                    l1_bias_available,
                    l0_bias_available,
                    cfg.swiglu_h,
                    cfg.swiglu_k,
                    cfg.swiglu_m,
                    BLOCK_DIM,
                )

    # Down after Up SyncAll. SwiGLU (single/dual): CPP ``if ASCEND_IS_AIC``
    # Down only — ``stage_down_{basic,full_*}`` are cube-only (AIV no-op).
    # Non-SwiGLU StreamK keeps AIC+AIV reduce; basic/full are also cube-only.
    # Match CPP WithoutQue: no SyncAll B-prefetch (CopyInB1 always in-loop).
    stage_barrier_hard(aiv_arrive, aic_ffts, aic_done)
    if tla.const_expr(cfg.down_mode == DOWN_STREAMK):
        stage_down_streamk(
            gm_hidden,
            gm_w2_t,
            gm_y,
            gm_sk_ws,
            l1a0_ptr,
            l1a1_ptr,
            l1b0_ptr,
            l1b1_ptr,
            l0a0_ptr,
            l0a1_ptr,
            l0b0_ptr,
            l0b1_ptr,
            l0c_ptr,
            tiling_down,
            swizzle_down,
            BLOCK_DIM,
            l1_single,
            _buf_l1_tm,
            tiling_buf.l1_tn,
            tiling_buf.l1_tk,
            l0_single,
            gm_b2,
            l1_bias_ptr,
            l0_bias_ptr,
            USE_BIAS,
            l1_bias_data_ready,
            l1_bias_available,
            l0_bias_available,
        )
    elif tla.const_expr(cfg.down_mode == DOWN_FULL_A):
        stage_down_full_a(
            gm_hidden,
            gm_w2_t,
            gm_y,
            l1a0_ptr,
            l1a1_ptr,
            l1b0_ptr,
            l1b1_ptr,
            l0a0_ptr,
            l0a1_ptr,
            l0b0_ptr,
            l0b1_ptr,
            l0c_ptr,
            tiling_down,
            swizzle_down,
            l1_single,
            _buf_l1_tm,
            tiling_buf.l1_tn,
            tiling_buf.l1_tk,
            l0_single,
            gm_b2,
            l1_bias_ptr,
            l0_bias_ptr,
            USE_BIAS,
            l1_bias_data_ready,
            l1_bias_available,
            l0_bias_available,
            B_IS_TRANS=(cfg.layout == LAYOUT_LINEAR),
        )
    elif tla.const_expr(cfg.down_mode == DOWN_FULL_B):
        stage_down_full_b(
            gm_hidden,
            gm_w2_t,
            gm_y,
            l1a0_ptr,
            l1a1_ptr,
            l1b0_ptr,
            l1b1_ptr,
            l0a0_ptr,
            l0a1_ptr,
            l0b0_ptr,
            l0b1_ptr,
            l0c_ptr,
            tiling_down,
            swizzle_down,
            l1_single,
            _buf_l1_tm,
            tiling_buf.l1_tn,
            tiling_buf.l1_tk,
            l0_single,
            gm_b2,
            l1_bias_ptr,
            l0_bias_ptr,
            USE_BIAS,
            l1_bias_data_ready,
            l1_bias_available,
            l0_bias_available,
            B_IS_TRANS=(cfg.layout == LAYOUT_LINEAR),
        )
    else:
        stage_down_basic(
            gm_hidden,
            gm_w2_t,
            gm_y,
            l1a0_ptr,
            l1a1_ptr,
            l1b0_ptr,
            l1b1_ptr,
            l0a0_ptr,
            l0a1_ptr,
            l0b0_ptr,
            l0b1_ptr,
            l0c_ptr,
            tiling_down,
            swizzle_down,
            l1_single,
            _buf_l1_tm,
            tiling_buf.l1_tn,
            tiling_buf.l1_tk,
            l0_single,
            gm_b2,
            l1_bias_ptr,
            l0_bias_ptr,
            USE_BIAS,
            l1_bias_data_ready,
            l1_bias_available,
            l0_bias_available,
            cfg.swiglu_m,
            cfg.swiglu_n,
            cfg.swiglu_h,
            BLOCK_DIM,
        )

    # CPP Linear FFN: after Down, ``if ASCEND_IS_AIC { PipeBarrier<PIPE_ALL>(); }``
    # (SwiGLU / SiLU / GELU / StreamK). Strict AIC drain before kernel exit.
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
