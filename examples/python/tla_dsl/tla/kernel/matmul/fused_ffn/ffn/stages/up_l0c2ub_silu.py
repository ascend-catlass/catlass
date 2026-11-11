# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# FFN Up stage (L0C→UB path): GEMM FixPipe → UB + act epilogue (Phase-1).
from __future__ import annotations

import catlass.tla as tla

from ffn.config import ACT_GELU, ACT_RAW, ACT_SILU, ACT_SWIGLU, ACT_SWIGLU_DUAL
from ffn.gm_l1 import gm_l1_abs, gm_l1_abs_1d, gm_l1_tile
from ffn.stages.epi_apply import (
    epi_apply_swiglu_dual_vl_1d,
    epi_apply_swiglu_dual_vl_2d,
    epi_apply_swiglu_vl_1d,
    epi_apply_vl_1d,
    epi_apply_vl_2d,
)
from ffn.tiling import _HALF_L1_ELEMS, SwizzleParams, TilingParams

UB_SIZE = tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub)
L0C_SIZE = tla.arch.get_capacity_in_bytes(tla.AddressSpace.l0c)
BYTE_PER_C0 = 32
# CPP DoubleCopyOutDualSplitM: SiLU/GELU/SwiGLU need16Align → Align(n, fp16_C0=16).
# RAW (OP_TYPE_EMPTY): need16Align=false → float C0=8 (misaligned H otherwise
# shifts rows by 32B). Old pitch=l1_tn padded N-tail blocked oneshot.
_FIXPIPE_DST_ALIGN_F16 = 16
_FIXPIPE_DST_ALIGN_F32 = 8
# CPP FusionSwigluSingle::Init: stageSize = min(lastUB/stageNum/sizeof/N*N, M*N)
# with stageNum=2 (UB-driven auto row height).
_SWIGLU_STAGE_NUM = 2
# Host UB sizing: FixPipe acc reserves at least L0C/2 (C_UB_ELEMS); when ASWT
# emits dbL0C=1 tiles, host passes ACC_ELEMS=ceil(tm/2)×tn (SPLIT_M per AIV).
# Remaining UB budget → out slot(s) / inplace pack headroom.
EVG_UB_NODES = 1
EVG_UB_STAGES = 2
# Default FixPipe C workspace: L0C/2 in fp32 elements (matches matmul_add_ub).
C_UB_ELEMS = L0C_SIZE // 2 // 4
ACC_UB_STAGES = 1
ACC_SLOT_ELEMS = C_UB_ELEMS


def up_fixpipe_acc_elems(l0_tm: int, l0_tn: int, *, compact: bool = False) -> int:
    """fp32 UB-acc elems **per slot** (SPLIT_M half-M strip).

    Default (single-acc SiLU/GELU/SwiGLU single): ``max(L0C/2, tile)`` —
    WithoutQue pad so oneshot FixPipe strip fits a C_UB-sized slot.

    ``compact=True`` (SwiGLU dual: separate acc + gate FP32 slots): exact
    tile, 256B-aligned. Padding each slot to L0C/2 would make
    ``2 × 128KiB`` fill the whole UB with no room for out.
    """
    half_m = (int(l0_tm) + 1) // 2
    tile = max(1, half_m * int(l0_tn))
    if compact:
        align = max(1, 256 // 4)
        return ((tile + align - 1) // align) * align
    return max(C_UB_ELEMS, tile)


def swiglu_dual_ub_bytes(l0_tm: int, l0_tn: int) -> int:
    """Bytes for dual's two compact FP32 slots (acc + gate), no out."""
    return up_fixpipe_acc_elems(l0_tm, l0_tn, compact=True) * 4 * 2


HALF_L1_ELEMS = _HALF_L1_ELEMS
L1_SLOT_PAD = 0
# CPP KernelMatmulMixWithoutQue: countId = count / 15 % 3 over 3 flag IDs.
_CV_COUNT_ID_MAX = 15
_CV_COUNT_FLAG = 3


@tla.jit
def _cv_wait_aiv(
    aiv0,
    aiv1,
    aiv2,
    count,
    pipe,
    ID0_ONLY: tla.Constexpr[bool] = False,
) -> None:
    """AIC CrossCoreWait AIV (mode=4 ×2). countId = count/15%3.

    ``ID0_ONLY``: tiles/core never reaches 15 → cid≡0; skip div/mod + 3-way if
    (TLA SoftFlag objects cannot do CPP ``base+countId`` runtime index).
    """
    if tla.const_expr(ID0_ONLY):
        tla.cross_core_wait_flag(aiv0, pipe, aiv_id=0)
        tla.cross_core_wait_flag(aiv0, pipe, aiv_id=1)
    else:
        cid = (count // _CV_COUNT_ID_MAX) % _CV_COUNT_FLAG
        if cid == 0:
            tla.cross_core_wait_flag(aiv0, pipe, aiv_id=0)
            tla.cross_core_wait_flag(aiv0, pipe, aiv_id=1)
        elif cid == 1:
            tla.cross_core_wait_flag(aiv1, pipe, aiv_id=0)
            tla.cross_core_wait_flag(aiv1, pipe, aiv_id=1)
        else:
            tla.cross_core_wait_flag(aiv2, pipe, aiv_id=0)
            tla.cross_core_wait_flag(aiv2, pipe, aiv_id=1)


@tla.jit
def _cv_set_aic(
    aic0,
    aic1,
    aic2,
    count,
    pipe,
    ID0_ONLY: tla.Constexpr[bool] = False,
) -> None:
    """AIC CrossCoreSet → AIV (mode=4 ×2). countId = count/15%3."""
    if tla.const_expr(ID0_ONLY):
        tla.cross_core_set_flag(aic0, pipe, aiv_id=0)
        tla.cross_core_set_flag(aic0, pipe, aiv_id=1)
    else:
        cid = (count // _CV_COUNT_ID_MAX) % _CV_COUNT_FLAG
        if cid == 0:
            tla.cross_core_set_flag(aic0, pipe, aiv_id=0)
            tla.cross_core_set_flag(aic0, pipe, aiv_id=1)
        elif cid == 1:
            tla.cross_core_set_flag(aic1, pipe, aiv_id=0)
            tla.cross_core_set_flag(aic1, pipe, aiv_id=1)
        else:
            tla.cross_core_set_flag(aic2, pipe, aiv_id=0)
            tla.cross_core_set_flag(aic2, pipe, aiv_id=1)


@tla.jit
def _cv_wait_aic(
    aic0,
    aic1,
    aic2,
    count,
    aiv_sub,
    ID0_ONLY: tla.Constexpr[bool] = False,
) -> None:
    """AIV CrossCoreWait AIC on PIPE_V. One channel per sub (CPP mode=4)."""
    if tla.const_expr(ID0_ONLY):
        if aiv_sub == 0:
            tla.cross_core_wait_flag(aic0, tla.arch.VECTOR, aiv_id=0)
        else:
            tla.cross_core_wait_flag(aic0, tla.arch.VECTOR, aiv_id=1)
    else:
        cid = (count // _CV_COUNT_ID_MAX) % _CV_COUNT_FLAG
        if cid == 0:
            if aiv_sub == 0:
                tla.cross_core_wait_flag(aic0, tla.arch.VECTOR, aiv_id=0)
            else:
                tla.cross_core_wait_flag(aic0, tla.arch.VECTOR, aiv_id=1)
        elif cid == 1:
            if aiv_sub == 0:
                tla.cross_core_wait_flag(aic1, tla.arch.VECTOR, aiv_id=0)
            else:
                tla.cross_core_wait_flag(aic1, tla.arch.VECTOR, aiv_id=1)
        else:
            if aiv_sub == 0:
                tla.cross_core_wait_flag(aic2, tla.arch.VECTOR, aiv_id=0)
            else:
                tla.cross_core_wait_flag(aic2, tla.arch.VECTOR, aiv_id=1)


@tla.jit
def _cv_set_aiv(
    aiv0,
    aiv1,
    aiv2,
    count,
    aiv_sub,
    pipe,
    ID0_ONLY: tla.Constexpr[bool] = False,
) -> None:
    """AIV CrossCoreSet → AIC (CPP PIPE_MTE3 after epi / empty)."""
    if tla.const_expr(ID0_ONLY):
        if aiv_sub == 0:
            tla.cross_core_set_flag(aiv0, pipe, aiv_id=0)
        else:
            tla.cross_core_set_flag(aiv0, pipe, aiv_id=1)
    else:
        cid = (count // _CV_COUNT_ID_MAX) % _CV_COUNT_FLAG
        if cid == 0:
            if aiv_sub == 0:
                tla.cross_core_set_flag(aiv0, pipe, aiv_id=0)
            else:
                tla.cross_core_set_flag(aiv0, pipe, aiv_id=1)
        elif cid == 1:
            if aiv_sub == 0:
                tla.cross_core_set_flag(aiv1, pipe, aiv_id=0)
            else:
                tla.cross_core_set_flag(aiv1, pipe, aiv_id=1)
        else:
            if aiv_sub == 0:
                tla.cross_core_set_flag(aiv2, pipe, aiv_id=0)
            else:
                tla.cross_core_set_flag(aiv2, pipe, aiv_id=1)


# SwiGLU B MTE2: CPP CopyInB1 always dual half-N Nd2Nz (no host-pack path).
_NZ_C0_F16 = 16
# GM→L1: plain tla.copy — layoutTag selects Nd2Nz; keep AscendC Nd2Nz
# matrix-stride ABI inside the DSL (no user-facing copy params).
# SwiGLU dual half-N: parent L1 pitch via nested tile_view.


def swiglu_b_cpp_eligible(
    l1_tn: int,
    l1_tk_step: int,
    n_cols: int,
    *,
    use_bias: bool = False,
    use_mtail: bool = False,
) -> bool:
    """Validate SwiGLU single tiling for CPP dual half-N B (always required).

    Raises if geometry cannot run device CopyInB1. ``use_bias`` / ``use_mtail``
    are ignored (CPP enableSwigluSingle is unconditional). Returns True on OK
    so call sites stay boolean; there is **no** host-pack fallback.
    """
    del use_bias, use_mtail
    tn = int(l1_tn)
    tk = int(l1_tk_step)
    n_cols = int(n_cols)
    if tn < 32 or tn % 32 != 0:
        raise ValueError(
            f"SwiGLU single requires 32-aligned tile N (CPP); got l1_tn={tn}"
        )
    half = tn // 2
    if half % _NZ_C0_F16 != 0:
        raise ValueError(f"SwiGLU single half-N must be C0-aligned; half={half}")
    if tk % _NZ_C0_F16 != 0:
        raise ValueError(f"SwiGLU single l1_tk must be C0-aligned; got {tk}")
    if n_cols < 2 or n_cols % 2 != 0:
        raise ValueError(f"SwiGLU single needs even 2H columns; got n_cols={n_cols}")
    return True


# NOTE: bf16 productization is sample/e2e work, not missing tla.exp(bf16).
# epi_silu already mirrors CPP RegSiluB16 (f32 math → cast/PACK). 3510/CPP
# have no native bf16 vexp — see ffn/docs/ffn_dsl_support_and_perf.md §1.2.1
# and ffn/docs/ffn_dsl_cpp_gap_status.md (G-BF16).


@tla.jit
def stage_up_l0c2ub_silu(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_d: tla.Tensor,
    gm_c_geom: tla.Tensor,
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
    UB_SLOT_ELEMS: tla.Constexpr[int],
    SIMD_LANES: tla.Constexpr[int],
    L1_SINGLE: tla.Constexpr[bool] = False,
    ACT: tla.Constexpr[int] = ACT_SILU,
    BUF_L1_TM: tla.Constexpr[int] = 0,
    BUF_L1_TN: tla.Constexpr[int] = 0,
    BUF_L1_TK: tla.Constexpr[int] = 0,
    ACC_ELEMS: tla.Constexpr[int] = 0,
    L0_SINGLE: tla.Constexpr[bool] = False,
    gm_gate: tla.Tensor = None,
    ub_acc_ptr0=None,
    ub_acc_ptr1=None,
    ub_out_ptr0=None,
    ub_out_ptr1=None,
    ub_gate_slot_ptr=None,
    USE_OUT_PP: tla.Constexpr[bool] = False,
    FLAG_TAG: tla.Constexpr[int] = 0,
    gm_bias: tla.Tensor = None,
    l1_bias_ptr=None,
    l0_bias_ptr=None,
    USE_BIAS: tla.Constexpr[bool] = False,
    l1_bias_data_ready=None,
    l1_bias_available=None,
    l0_bias_available=None,
    SWIGLU_H: tla.Constexpr[int] = 0,
    SWIGLU_K: tla.Constexpr[int] = 0,
    SWIGLU_M: tla.Constexpr[int] = 0,
    BLOCK_DIM: tla.Constexpr[int] = 0,
    # SwiGLU dual: kernel creates one CV finish set; both Raw+Up phases reuse
    # it (CrossCore pool max 16). Non-dual leaves SHARED_CV=False and creates
    # finishes locally under ``_fp``.
    SHARED_CV: tla.Constexpr[bool] = False,
    aic_finish0=None,
    aic_finish1=None,
    aic_finish2=None,
    aiv_finish0=None,
    aiv_finish1=None,
    aiv_finish2=None,
) -> None:
    """Cube GEMM FixPipe into UB; vector epilogue D = act(Acc) (SiLU/GELU/SwiGLU).

    On-chip L1/L0/L0C must be supplied by the caller so Up+Down can share one set.
    UB for FixPipe acc + act out is allocated here (Down does not use UB).
    Soft-flag names use the ``upl0c_`` prefix to avoid clashes with Down.
    ``L1_SINGLE`` must match allocate_gemm_pingpong (alias l1*1→l1*0).
    ``BUF_L1_*`` = merge alloc box (0 → same as stage); used to split PP.
    ``ACT`` selects VL epilogue. ``gm_c_geom`` sizes AIC L0C/FixPipe (``[M,H]``
    for SiLU/GELU/dual; ``[M,2H]`` for SwiGLU single). ``gm_d`` is AIV store.
    ``gm_gate`` is FP32 gate GM for ``ACT_SWIGLU_DUAL`` (ignored otherwise).
    """
    c0 = 0
    c1 = 1

    dtype_gm_c = gm_d.ptr.dtype
    # Dual Raw+Up: FLAG_TAG 0/1 uniquifies SoftFlag names across SyncAll
    # phases. HardEvent max 8/pipe-pair — Raw+Up+Down share the pool. Dual
    # phases use *single-buffer* SoftFlags (slot0 only; no ping-pong) so
    # 2×Up L1 = 4 and Down still fits. Skip EVENT_67 pad + dual-UB finish.
    # CrossCore finishes for dual are injected via SHARED_CV (kernel-owned).
    _fp = "upl0c" if FLAG_TAG == 0 else "upl0cB"
    _dual_phase = tla.const_expr(ACT == ACT_RAW or ACT == ACT_SWIGLU_DUAL)
    # RAW float FixPipe: C0=8; act paths: fp16 C0=16 (CPP need16Align).
    _fixpipe_align = (
        _FIXPIPE_DST_ALIGN_F32
        if tla.const_expr(ACT == ACT_RAW)
        else _FIXPIPE_DST_ALIGN_F16
    )

    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    k = gm_a.origin_shape[1]
    # CPP tiling dump may set kL1 > K (L1 slot); device then
    # ``kL1_ = Min(K, kL1)`` for MTE2 quantum. Buffers keep published ``l1_tk``.
    # Host ``work_l1_tk`` is that Min; SWIGLU_K backs it when work field is 0.
    _pub_l1_tk = _tiling.l1_tk
    _work_l1_tk = (
        _tiling.work_l1_tk if tla.const_expr(_tiling.work_l1_tk > 0) else _pub_l1_tk
    )
    if tla.const_expr(
        _tiling.work_l1_tk == 0 and SWIGLU_K > 0 and SWIGLU_K < _pub_l1_tk
    ):
        _work_l1_tk = SWIGLU_K
    # Bake SwiGLU mask. CPP RegSwigluSingleB16 always UpdateMask (plt).
    # Do NOT bake ALL from full-tile l1_tn//2: N-rem half_raw may still be
    # non-SIMD-aligned while (l1_tn//2)%SIMD==0, which would corrupt rem cols.
    # Always UM (TAIL_LANES=0) for CPP parity / rem safety.
    _swiglu_half_w = _tiling.l1_tn // 2
    _swiglu_tail_lanes = 0
    _swiglu_use_all_mask = False
    _swiglu_n_vl = (_swiglu_half_w + SIMD_LANES - 1) // SIMD_LANES
    # Nested UpdateMask / ALL path only (ptr-hoist experiment disabled).

    # ---- soft-flags (CPP BlockMmad WithoutQue HardEvent IDs) ----
    # CPP ctor ALWAYS: MTE1_MTE2 0..3, FIX_M 0/1, M_MTE1 6/7. Loop dual-L1
    # only uses l1BufId 0/1 (A+B under one ID); IDs 2/3 are credit-only.
    # Explicit ids: Up ~drain → Down reuses same HW events (CPP sequential
    # BlockMmad lifetimes).
    l1_0_data_ready = tla.flag(
        _fp + "_l1_0_data_ready", tla.arch.MTE2, tla.arch.MTE1, id=0
    )
    l1_0_available = tla.flag(
        _fp + "_l1_0_available", tla.arch.MTE1, tla.arch.MTE2, id=0
    )
    # Credit-only (CPP SECOND/THIRD); never Wait in the L1 loop.
    l1_2_available = tla.flag(
        _fp + "_l1_2_available", tla.arch.MTE1, tla.arch.MTE2, id=2
    )
    l1_3_available = tla.flag(
        _fp + "_l1_3_available", tla.arch.MTE1, tla.arch.MTE2, id=3
    )
    if tla.const_expr(_dual_phase):
        # Single-buffer loop: alias slot-1 → slot-0. ID 1 still primed in ctor
        # via a distinct SoftFlag (CPP always Sets FIRST_FLAG).
        l1_1_data_ready = l1_0_data_ready
        l1_1_available = l1_0_available
        l1_1_ctor = tla.flag(_fp + "_l1_1_ctor", tla.arch.MTE1, tla.arch.MTE2, id=1)
    else:
        l1_1_data_ready = tla.flag(
            _fp + "_l1_1_data_ready", tla.arch.MTE2, tla.arch.MTE1, id=1
        )
        l1_1_available = tla.flag(
            _fp + "_l1_1_available", tla.arch.MTE1, tla.arch.MTE2, id=1
        )
        l1_1_ctor = l1_1_available
    l0_0_available = tla.flag(
        _fp + "_l0_0_available", tla.arch.CUBE, tla.arch.MTE1, id=6
    )
    l0_0_data_ready = tla.flag(
        _fp + "_l0_0_data_ready", tla.arch.MTE1, tla.arch.CUBE, id=6
    )
    if tla.const_expr(_dual_phase):
        l0_1_available = l0_0_available
        l0_1_data_ready = l0_0_data_ready
        l0_1_ctor = tla.flag(_fp + "_l0_1_ctor", tla.arch.CUBE, tla.arch.MTE1, id=7)
    else:
        l0_1_available = tla.flag(
            _fp + "_l0_1_available", tla.arch.CUBE, tla.arch.MTE1, id=7
        )
        l0_1_data_ready = tla.flag(
            _fp + "_l0_1_data_ready", tla.arch.MTE1, tla.arch.CUBE, id=7
        )
        l0_1_ctor = l0_1_available
    # CPP ctor always Sets FIX_M(0) and FIX_M(1) (even if dbL0C=1 at runtime).
    _l0c_db = tla.const_expr(_tiling.l0c_db > 1)
    _l0c_slot_elems = _tiling.l0_tm * _tiling.l0_tn
    l0c0_data_ready = tla.flag(
        _fp + "_l0c0_data_ready", tla.arch.CUBE, tla.arch.FIX, id=0
    )
    l0c0_available = tla.flag(
        _fp + "_l0c0_available", tla.arch.FIX, tla.arch.CUBE, id=0
    )
    # id=1 always allocated (ctor/dtor). Loop only toggles it when dbL0C=2;
    # when dbL0C=1 the loop stays on slot0 (CPP still primes FIX_M(1)).
    l0c1_data_ready = tla.flag(
        _fp + "_l0c1_data_ready", tla.arch.CUBE, tla.arch.FIX, id=1
    )
    l0c1_available = tla.flag(
        _fp + "_l0c1_available", tla.arch.FIX, tla.arch.CUBE, id=1
    )
    # Cube BT SoftFlags come from the kernel (shared Up↔Down; SoftFlag budget).

    # UB comes from caller (kernel allocates once). No allocate here —
    # TLA sums every allocate in the MIX kernel (dual Raw+Up would OOM).
    use_out_pp = tla.const_expr(USE_OUT_PP)
    # SoftFlag for out-slot ping-pong only when out-PP path is live.
    # CPP Elementwise: only V_MTE3(0); FusionSwigluSingle: only MTE3_V(0).
    _ub_pp_sf = tla.const_expr(use_out_pp)

    # ---- UB SoftFlags: CPP self-fences id=0; optional out-PP on 1/2 ----
    mte3_v_fence = tla.flag(_fp + "_mte3_v_fence", tla.arch.MTE3, tla.arch.VECTOR, id=0)
    ub_out0_data_ready = tla.flag(
        _fp + "_ub_out0_data_ready", tla.arch.VECTOR, tla.arch.MTE3, id=0
    )
    if tla.const_expr(_ub_pp_sf):
        ub_out0_available = tla.flag(
            _fp + "_ub_out0_available", tla.arch.MTE3, tla.arch.VECTOR, id=1
        )
        if tla.const_expr(_dual_phase):
            ub_out1_data_ready = ub_out0_data_ready
            ub_out1_available = ub_out0_available
        else:
            ub_out1_data_ready = tla.flag(
                _fp + "_ub_out1_data_ready", tla.arch.VECTOR, tla.arch.MTE3, id=1
            )
            ub_out1_available = tla.flag(
                _fp + "_ub_out1_available", tla.arch.MTE3, tla.arch.VECTOR, id=2
            )
    else:
        # Alias to V_MTE3(0) so dead branches type-check; never Wait/Set as credit.
        ub_out0_available = ub_out0_data_ready
        ub_out1_data_ready = ub_out0_data_ready
        ub_out1_available = ub_out0_data_ready
    # SwiGLU dual: MTE2 gate load → VECTOR silu*up (CPP FusionSwiglu MTE2_V).
    if tla.const_expr(ACT == ACT_SWIGLU_DUAL):
        gate_mte2_v = tla.flag(_fp + "_gate_mte2_v", tla.arch.MTE2, tla.arch.VECTOR)
    else:
        gate_mte2_v = ub_out0_data_ready

    # AIC↔AIV done flags. CPP countId = count/15%3 over 3 IDs (FLAG 5/8 + id).
    if tla.const_expr(not SHARED_CV):
        aic_finish0 = tla.cross_flag(_fp + "_aic_finish0", mode=4)
        aic_finish1 = tla.cross_flag(_fp + "_aic_finish1", mode=4)
        aic_finish2 = tla.cross_flag(_fp + "_aic_finish2", mode=4)
        aiv_finish0 = tla.cross_flag(_fp + "_aiv_finish0", mode=4)
        aiv_finish1 = tla.cross_flag(_fp + "_aiv_finish1", mode=4)
        aiv_finish2 = tla.cross_flag(_fp + "_aiv_finish2", mode=4)
    # Out pool is hidden dtype (f16). RAW oneshot stores via acc; strip path
    # recasts the same bytes to f32 (half the element count).
    if tla.const_expr(ACT == ACT_RAW):
        ub_out_ptr0 = tla.recast_ptr(ub_out_ptr0, dtype=tla.Float32)
        ub_out_ptr1 = tla.recast_ptr(ub_out_ptr1, dtype=tla.Float32)

    # L1_SINGLE comes from allocate_gemm_pingpong(tiling_buf); do not recompute
    # from _tiling alone — FFN merge can alias L1 while stage tiling still looks dual.
    l1_single = tla.const_expr(L1_SINGLE)
    # Split merge single buffer into Up ping-pong when 2×stage A/B fit in alloc.
    # Unified pool ``(total, tk=1, tn=0)``: carve pingA|pingB|pongA|pongB.
    buf_unified = tla.const_expr(BUF_L1_TK == 1 and BUF_L1_TN == 0 and BUF_L1_TM > 0)
    buf_tm = BUF_L1_TM if BUF_L1_TM > 0 else _tiling.l1_tm
    buf_tn = BUF_L1_TN if BUF_L1_TN > 0 else _tiling.l1_tn
    buf_tk = BUF_L1_TK if BUF_L1_TK > 0 else _tiling.l1_tk
    # Buffer carve uses published kL1 (CPP aL1OneBuffer_ = mL1 * kL1 dump).
    stage_a = _tiling.l1_tm * _pub_l1_tk
    stage_b = _pub_l1_tk * _tiling.l1_tn
    # CPP DoAllLoc2UbAswt: dual L1 only when 2×full (mL1*kL1 + kL1*nL1) fits;
    # each slot holds full aL1OneBuffer_/bL1OneBuffer_; MTE2 quantum = full
    # kL1_ (Min(K,kL1)). No permanent half-K dual (old can_split_half removed).
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
                # Align CPP HALF_L1 dual-slot layout: B follows A in each half.
                l1b0_ptr = l1_base + stage_a
                l1a1_ptr = l1_base + HALF_L1_ELEMS
                l1b1_ptr = l1_base + HALF_L1_ELEMS + stage_a
            else:
                # Unified pool: ping A/B, optional pad, then pong A/B.
                l1b0_ptr = l1_base + stage_a
                l1a1_ptr = l1_base + stage_a + stage_b + slot_pad
                l1b1_ptr = l1_base + stage_a + stage_b + slot_pad + stage_a
        else:
            l1a1_ptr = l1a0_ptr + stage_a
            l1b1_ptr = l1b0_ptr + stage_b
        l1_single = tla.const_expr(False)

    # CPP enableSwigluSingle: always dual half-N B (+ CopyInC1). No pack path.
    _swiglu_half_tn = _tiling.l1_tn // 2
    _swiglu_b_cpp = tla.const_expr(
        (ACT == ACT_SWIGLU) and (SWIGLU_H > 0) and (SWIGLU_K > 0)
    )
    # MTE2 K step = CPP curKL1: working quantum (post-Min), then Min(K, ·).
    # Prefer host SWIGLU_K so step is constexpr (kills runtime Min/sdiv).
    if tla.const_expr(_swiglu_b_cpp and SWIGLU_K > 0):
        l1_tk_step = min(_work_l1_tk, SWIGLU_K)
    else:
        l1_tk_step = min(_work_l1_tk, k)

    # Host-published SwiGLU grid (constexpr) — avoid GM origin_shape CeilDiv.
    _cv_up_n = 2 * SWIGLU_H
    _cv_grid_m = (SWIGLU_M + _tiling.l1_tm - 1) // _tiling.l1_tm
    _cv_grid_n = (_cv_up_n + _tiling.l1_tn - 1) // _tiling.l1_tn
    _cv_tiles = _cv_grid_m * _cv_grid_n
    if tla.const_expr(_swiglu_b_cpp and SWIGLU_M > 0 and SWIGLU_H > 0):
        grid_m = _cv_grid_m
        grid_n = _cv_grid_n
        total_blocks = _cv_tiles
    else:
        grid_m = (m + _tiling.l1_tm - 1) // _tiling.l1_tm
        grid_n = (n + _tiling.l1_tn - 1) // _tiling.l1_tn
        total_blocks = grid_m * grid_n
    if tla.const_expr(_swizzle.SWIZZLE_DIRECTION == 0):
        swizzle_tile_count = (
            grid_m + _swizzle.SWIZZLE_OFFSET - 1
        ) // _swizzle.SWIZZLE_OFFSET
    else:
        swizzle_tile_count = (
            grid_n + _swizzle.SWIZZLE_OFFSET - 1
        ) // _swizzle.SWIZZLE_OFFSET
    # CPP ASWT last-wave mTail: expand task count; split MN on last round.
    m_tail = _swizzle.M_TAIL_CNT
    n_tail = _swizzle.N_TAIL_CNT
    tail_cnt = m_tail * n_tail
    use_mtail = tla.const_expr(tail_cnt > 1)
    # CPP GetBlockShape rem: CeilDiv(tailL1Last, mTailCnt) at runtime.
    # Host M_TAIL_SPLIT is debug/fallback only.
    # Single K-L1 when work covers full K (CPP curKL1Iter==1). SwiGLU: host
    # publishes SWIGLU_K; work=Min(K,kL1). Kill L1 SoftFlag if/else → slot0.
    _single_kl1 = tla.const_expr(
        _swiglu_b_cpp and (SWIGLU_K > 0) and (_work_l1_tk >= SWIGLU_K)
    )
    # Exactly 2 L0 under one L1 (CPP baseK=128, work=256 → 2×128). SoftFlag-fixed
    # straight line — same as CPP ``l0PingPong_&1`` without TLA if/else.
    _dual_l0 = tla.const_expr(
        _single_kl1
        and (not L0_SINGLE)
        and (not _dual_phase)
        and ((_work_l1_tk + _tiling.l0_tk - 1) // _tiling.l0_tk == 2)
    )
    # countId=count/15%3 stays 0 while tiles/core < 15.
    _cv_tasks = _cv_tiles
    if tla.const_expr(use_mtail):
        _cv_tasks = _cv_tiles * tail_cnt
    _cv_id0_only = tla.const_expr(
        _swiglu_b_cpp
        and (SWIGLU_M > 0)
        and (SWIGLU_H > 0)
        and (
            (_cv_tasks <= _CV_COUNT_ID_MAX)
            or ((BLOCK_DIM > 0) and (_cv_tasks <= BLOCK_DIM))
        )
    )
    # ≤1 tile/core: L0C DB unused (CPP still ctor-primes both ids). Lock slot0.
    _l0c_slot0_only = tla.const_expr(_cv_id0_only)
    # Host grid fits in one swizzle group → kill last-tile rem + zigzag SSA.
    if tla.const_expr(_swizzle.SWIZZLE_DIRECTION == 0):
        _cv_swizzle_count = (
            _cv_grid_m + _swizzle.SWIZZLE_OFFSET - 1
        ) // _swizzle.SWIZZLE_OFFSET
    else:
        _cv_swizzle_count = (
            _cv_grid_n + _swizzle.SWIZZLE_OFFSET - 1
        ) // _swizzle.SWIZZLE_OFFSET
    _swizzle_one = tla.const_expr(
        _swiglu_b_cpp and (SWIGLU_M > 0) and (SWIGLU_H > 0) and (_cv_swizzle_count == 1)
    )
    # AIC always uses origin_shape extents (former ``_exact_l1`` bake removed).
    # M divides l1_tm (N may still rem): bake AIV M extents. Prior AIC-side
    # ``_exact_m`` did not move aic_s; AIV origin_shape tax still shows in
    # aiv_s (+0.85µs vs CPP on swiglu128) and can stall AIC CrossCore.
    _exact_m = tla.const_expr(
        _swiglu_b_cpp
        and (SWIGLU_M > 0)
        and (not use_mtail)
        and (SWIGLU_M % _tiling.l1_tm == 0)
        and (_tiling.l1_tm % 2 == 0)  # equal dual-AIV split; odd needs rem crop
    )
    _n_rem = (_cv_up_n % _tiling.l1_tn) if (_cv_up_n > 0) else 0
    _has_n_rem = tla.const_expr(_swiglu_b_cpp and (_n_rem != 0))

    with tla.cube():
        # CPP BlockMmad ctor: Set MTE1_MTE2(0..3), M_MTE1(6/7), FIX_M(0/1).
        # Dual-phase loop aliases slot1→slot0, but ctor still primes distinct
        # HW ids via l1_1_ctor / l0_1_ctor (never double-Set the same object).
        tla.set_flag(l1_0_available)
        tla.set_flag(l1_1_ctor)
        tla.set_flag(l1_2_available)
        tla.set_flag(l1_3_available)
        tla.set_flag(l0_0_available)
        tla.set_flag(l0_1_ctor)
        tla.set_flag(l0c0_available)
        tla.set_flag(l0c1_available)
        if tla.const_expr(USE_BIAS):
            # Credit-only: CPP folds CopyInC1/C2 under A/B HardEvents (no
            # separate bias SoftFlag in the hot path).
            tla.set_flag(l1_bias_data_ready)
            tla.set_flag(l1_bias_available)
            tla.set_flag(l0_bias_available)

        l1_buf_idx = c0
        l0_buf_idx = c0
        l0c_slot = c0
        # CPP abL1LoopCnt_ & 0x1 → biasBt_[baseN_ * biasBufId] ping-pong.
        bias_buf_idx = c0
        l1_bias_ptr1 = l1_bias_ptr
        l0_bias_ptr1 = l0_bias_ptr
        if tla.const_expr(USE_BIAS):
            l1_bias_ptr1 = l1_bias_ptr + _tiling.l1_tn
            l0_bias_ptr1 = l0_bias_ptr + _tiling.l1_tn
        block_idx = tla.arch.block_idx()
        block_num = tla.arch.block_num()
        # CPP ASWT last-wave mTail: ``tileNum += (tailCnt-1)*(T%C)``; only the
        # rem tiles of the final round are MN-split (hetero full vs split views
        # joined via dynamic-if + dynamic tensor types).
        rem_tiles = total_blocks % block_num
        main_tiles = total_blocks - rem_tiles
        if tla.const_expr(use_mtail):
            expanded = main_tiles + rem_tiles * tail_cnt
        else:
            expanded = total_blocks
        runtime_zero = tla.as_numeric(0)
        # CPP enableCVSync + count: false/0 until first tile; then countId=count/15%3.
        enable_cv_sync = runtime_zero
        cv_count = runtime_zero
        _wp = tla.arch.FIX  # AIC waits aiv_finish on FIX (CPP default)
        # Allow with mTail: runtime still gates next_task < main_tiles so we
        # only prefetch full→full (rem splits have hetero M/N geometry).
        block_range = tla.range(block_idx, expanded, block_num)
        for task_idx in block_range:
            # Default: full tile. Last-wave rem tasks overwrite via dynamic if.
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
            block_row = c0
            block_col = c0
            if tla.const_expr(_swizzle_one):
                # One swizzle group: tile_idx≡0, n_rows≡grid_m, no zigzag flip.
                if tla.const_expr(_swizzle.SWIZZLE_DIRECTION == 0):
                    block_row = block_linear % grid_m
                    block_col = block_linear // grid_m
                else:
                    block_row = block_linear // grid_n
                    block_col = block_linear % grid_n
            elif tla.const_expr(_swizzle.SWIZZLE_DIRECTION == 0):
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
            else:
                swizzle_tile_idx = block_linear // (_swizzle.SWIZZLE_OFFSET * grid_m)
                in_tile_idx = block_linear % (_swizzle.SWIZZLE_OFFSET * grid_m)
                swizzle_n_cols = (
                    _swizzle.SWIZZLE_OFFSET
                    if swizzle_tile_idx != (swizzle_tile_count - 1)
                    else (grid_n - _swizzle.SWIZZLE_OFFSET * swizzle_tile_idx)
                )
                block_row = in_tile_idx // swizzle_n_cols
                block_col = (
                    swizzle_tile_idx * _swizzle.SWIZZLE_OFFSET
                    + in_tile_idx % swizzle_n_cols
                )
                block_row = (
                    (grid_m - block_row - 1)
                    if (swizzle_tile_idx % 2 == 1)
                    else block_row
                )
            # Geometric tile extent (edge M/N may be < L1).
            blk_m = m - block_row * _tiling.l1_tm
            blk_n = n - block_col * _tiling.l1_tn
            if blk_m > _tiling.l1_tm:
                blk_m = tla.as_numeric(_tiling.l1_tm)
            if blk_n > _tiling.l1_tn:
                blk_n = tla.as_numeric(_tiling.l1_tn)
            # CPP rem: CeilDiv(tailL1Last, mTailCnt); empty if offset past blk.
            split_tm_rt = (blk_m + m_tail - 1) // m_tail
            split_tn_rt = (blk_n + n_tail - 1) // n_tail
            split_empty = c0
            if tla.const_expr(use_mtail):
                if task_idx >= main_tiles:
                    if (m_si * split_tm_rt) >= blk_m:
                        split_empty = c1
                    if (n_si * split_tn_rt) >= blk_n:
                        split_empty = c1

            # Prefer host K (constexpr) for SwiGLU tile roots.
            _tile_k = k
            if tla.const_expr(_swiglu_b_cpp and SWIGLU_K > 0):
                _tile_k = SWIGLU_K
            gm_a_full = tla.tile_view(
                gm_a,
                tla.make_shape(_tiling.l1_tm, _tile_k),
                tla.make_coord(block_row, c0),
            )
            # Contiguous B tile root; SwiGLU single rebuilds dual half-N
            # from (K,H) inside the K loop (CPP CopyInB1; no host-pack).
            gm_b_full = tla.tile_view(
                gm_b,
                tla.make_shape(_tile_k, _tiling.l1_tn),
                tla.make_coord(c0, block_col),
            )
            gm_c_full = tla.tile_view(
                gm_c_geom,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(block_row, block_col),
            )
            if tla.const_expr(use_mtail):
                # Runtime extents so full vs split share one tile_view type.
                cur_tm = tla.as_numeric(_tiling.l1_tm)
                cur_tn = tla.as_numeric(_tiling.l1_tn)
                m_idx = c0
                n_idx = c0
                if task_idx >= main_tiles:
                    cur_tm = split_tm_rt
                    cur_tn = split_tn_rt
                    m_idx = m_si
                    n_idx = n_si
                gm_a_by_core = tla.tile_view(
                    gm_a_full,
                    tla.make_shape(cur_tm, _tile_k),
                    tla.make_coord(m_idx, c0),
                )
                gm_b_by_core = tla.tile_view(
                    gm_b_full,
                    tla.make_shape(_tile_k, cur_tn),
                    tla.make_coord(c0, n_idx),
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

            # Nested L1/L0 must follow by_core extents (split tiles are << l1_tm).
            # Exact full-L1 SwiGLU: bake extents (kill origin_shape smin/sdiv).
            work_tm = gm_a_by_core.origin_shape[0]
            work_tn = gm_c_by_core.origin_shape[1]
            # CPP ProcessSingleTile (enableSwigluSingle): when nL0Raw%32!=0,
            # mmad/UB frame = Align(raw,32); B/bias load halfW=raw/2 into
            # positions 0 and curNL1/2 (padded half). Else cube_tn==work_tn.
            cube_tn = work_tn
            half_raw = work_tn // 2
            if tla.const_expr(_swiglu_b_cpp):
                cube_tn = ((work_tn + 31) // 32) * 32
                half_raw = work_tn // 2
            half_pad = cube_tn // 2
            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + l1_tk_step - 1) // l1_tk_step
            k_l1_range = tla.range(c0, k_l1_count, c1)

            # CPP l0cOffset = (l0cPingPong_ & 1) * HALF_L0C; Wait FIX_M(slot).
            # Init before const_expr if (TLA dynamic-if SSA).
            l0c_ptr1 = l0c_ptr
            if tla.const_expr(_l0c_db and (not _l0c_slot0_only)):
                l0c_ptr1 = l0c_ptr + _l0c_slot_elems
            cur_l0c_ptr = l0c_ptr
            if tla.const_expr(not _l0c_slot0_only):
                cur_l0c_ptr = l0c_ptr if l0c_slot == c0 else l0c_ptr1
            if tla.const_expr(_swiglu_b_cpp):
                # On-chip extent may exceed GM tile (pad); make_layout_with_tag
                # takes caller extent (same pattern as MX 64-K pad).
                l0_c = tla.make_tensor(
                    cur_l0c_ptr,
                    tla.make_layout_with_tag(
                        (work_tm, cube_tn), tla.Float32, tla.arch.L0Clayout
                    ),
                )
            else:
                l0_c = tla.make_tensor_like(cur_l0c_ptr, gm_c_by_core)
            ub_acc_ptr = ub_acc_ptr0

            if split_empty != c0:
                # Empty last-wave split: handshake only (no MTE2/MMAD/FixPipe).
                # CPP ProcessAicMmad: wait when enableCVSync; count++; Set.
                if enable_cv_sync != c0:
                    _cv_wait_aiv(
                        aiv_finish0,
                        aiv_finish1,
                        aiv_finish2,
                        cv_count,
                        _wp,
                        _cv_id0_only,
                    )
                cv_count = cv_count + c1
                _cv_set_aic(
                    aic_finish0,
                    aic_finish1,
                    aic_finish2,
                    cv_count,
                    tla.arch.FIX,
                    _cv_id0_only,
                )
                enable_cv_sync = c1
                # Empty split: no L0C Wait/Set — keep l0c_slot (unlike ub CV).
            else:
                # CPP enableCVSync: skip AIV wait on first MN tile.
                # ≤1 tile/core: that skip is permanent (no prior tile) — no dynamic if.
                if tla.const_expr(not _l0c_slot0_only):
                    if enable_cv_sync != c0:
                        _cv_wait_aiv(
                            aiv_finish0,
                            aiv_finish1,
                            aiv_finish2,
                            cv_count,
                            _wp,
                            _cv_id0_only,
                        )

                if tla.const_expr(_l0c_slot0_only):
                    tla.wait_flag(l0c0_available)
                elif l0c_slot == c0:
                    tla.wait_flag(l0c0_available)
                else:
                    tla.wait_flag(l0c1_available)
                # CPP CopyInB1 dual half-N: hoist GM (K,H) roots + half index
                # outside K (invariant per MN tile). L1 half windows still
                # bind per k_l1 (ping-pong ptr / tk step).
                if tla.const_expr(_swiglu_b_cpp):
                    # Absolute half-column start in H (not ``// half_raw`` tile
                    # index — N-tail rem makes that floor to the wrong column).
                    _half_col0 = block_col * _swiglu_half_tn
                    gm_gate_root = tla.tile_view(
                        gm_b,
                        tla.make_shape(k, SWIGLU_H),
                        tla.make_coord(c0, c0),
                    )
                    gm_up_root = tla.tile_view(
                        gm_b,
                        tla.make_shape(k, SWIGLU_H),
                        tla.make_coord(c0, c1),
                    )
                # Bias tensors bound here; CopyInC1 runs after CopyInA1 on
                # first kL1 (CPP DoAllLoc2UbAswt: A → bias → B).
                # Slot = abL1LoopCnt&1 (bias_buf_idx); stride = baseN (=l1_tn).
                if tla.const_expr(USE_BIAS):
                    cur_l1_bias_ptr = (
                        l1_bias_ptr if bias_buf_idx == c0 else l1_bias_ptr1
                    )
                    cur_l0_bias_ptr = (
                        l0_bias_ptr if bias_buf_idx == c0 else l0_bias_ptr1
                    )
                    if tla.const_expr(_swiglu_b_cpp):
                        # Absolute half-col from full bias (gate @0, up @H).
                        gm_bias_gate = gm_l1_abs_1d(gm_bias, half_raw, _half_col0)
                        gm_bias_up = gm_l1_abs_1d(
                            gm_bias, half_raw, SWIGLU_H + _half_col0
                        )
                        l1_bias = tla.make_tensor(
                            cur_l1_bias_ptr,
                            tla.make_layout(
                                tla.make_shape(cube_tn),
                                tla.make_stride(c1),
                                layoutTag=tla.arch.VectorLayout,
                            ),
                        )
                        l0_bias = tla.make_tensor(
                            cur_l0_bias_ptr,
                            tla.make_layout(
                                tla.make_shape(cube_tn),
                                tla.make_stride(c1),
                                layoutTag=tla.arch.VectorLayout,
                            ),
                        )
                        l1_bias_gate = tla.tile_view(
                            l1_bias,
                            tla.make_shape(half_raw),
                            tla.make_coord(c0),
                        )
                        # up @ half_pad: mid tile then crop to half_raw
                        # (CPP cl1Local[curNL1/2] with nValue=halfW).
                        l1_bias_up_mid = tla.tile_view(
                            l1_bias,
                            tla.make_shape(half_pad),
                            tla.make_coord(c1),
                        )
                        l1_bias_up = tla.tile_view(
                            l1_bias_up_mid,
                            tla.make_shape(half_raw),
                            tla.make_coord(c0),
                        )
                    else:
                        _bias_col0 = block_col * _tiling.l1_tn
                        if tla.const_expr(use_mtail):
                            gm_bias_by_core = gm_l1_abs_1d(
                                gm_bias, cur_tn, _bias_col0 + n_idx * cur_tn
                            )
                        else:
                            # Use clipped work_tn (not full l1_tn) so N-tail
                            # GM→L1 bias DMA stays inside the allocated vector.
                            gm_bias_by_core = gm_l1_abs_1d(gm_bias, work_tn, _bias_col0)
                        l1_bias = tla.make_tensor_like(
                            cur_l1_bias_ptr, gm_bias_by_core, tla.arch.VectorLayout
                        )
                        l0_bias = tla.make_tensor_like(
                            cur_l0_bias_ptr, l1_bias, tla.arch.VectorLayout
                        )
                if tla.const_expr(_dual_l0):
                    # trip-1 kL1: no scf.for shell (CPP curKL1Iter==1).
                    k_l1 = c0
                    gm_a_l1 = gm_l1_tile(gm_a_by_core, work_tm, l1_tk_step, c0, k_l1)
                    gm_b_l1 = gm_l1_tile(gm_b_by_core, l1_tk_step, work_tn, k_l1, c0)
                    if tla.const_expr(_single_kl1):
                        _l1a_ptr = l1a0_ptr
                        _l1b_ptr = l1b0_ptr
                    else:
                        _l1a_ptr = l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr
                        _l1b_ptr = l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr
                    l1_a = tla.make_tensor_like(_l1a_ptr, gm_a_l1)
                    if tla.const_expr(_swiglu_b_cpp):
                        # Parent L1 N = cube_tn (padded); halves load half_raw.
                        l1_b = tla.make_tensor(
                            _l1b_ptr,
                            tla.make_layout_with_tag(
                                (l1_tk_step, cube_tn),
                                gm_b.ptr.dtype,
                                tla.arch.nZ,
                            ),
                        )
                    else:
                        l1_b = tla.make_tensor_like(_l1b_ptr, gm_b_l1)
                    # CPP DoAllLoc2UbAswt: Wait MTE1_MTE2 → CopyA+B →
                    # Set+Wait MTE2_MTE1 (eager drain).
                    if tla.const_expr(_single_kl1):
                        tla.wait_flag(l1_0_available)
                    elif l1_buf_idx == c0:
                        tla.wait_flag(l1_0_available)
                    else:
                        tla.wait_flag(l1_1_available)
                    if tla.const_expr(_swiglu_b_cpp):
                        # CPP CopyInB1 enableSwigluSingle (field-level):
                        #   nValue=halfW (=half_raw), dValue=curKL1,
                        #   srcNdMatrixStride=0, dstNzMatrixStride=0,
                        #   dstNzC0Stride=Align(curNL1)=cube_tn (parent pitch),
                        #   gate @ L1+0; up @ L1[(curNL1/2)*C0] (=half_pad*C0).
                        # GM: gate root + tile; up root = biasGlobal[n_/2] style
                        # via gm_up_root (coord 1 × SWIGLU_H).
                        gm_b_gate = gm_l1_abs(
                            gm_gate_root,
                            l1_tk_step,
                            half_raw,
                            k_l1 * l1_tk_step,
                            _half_col0,
                        )
                        gm_b_up = gm_l1_abs(
                            gm_up_root,
                            l1_tk_step,
                            half_raw,
                            k_l1 * l1_tk_step,
                            _half_col0,
                        )
                        l1_b_gate = tla.tile_view(
                            l1_b,
                            tla.make_shape(l1_tk_step, half_raw),
                            tla.make_coord(c0, c0),
                        )
                        # Nest: mid @ half_pad (=curNL1/2), then crop to half_raw
                        # (no-pad: half_raw==half_pad → identity).
                        l1_b_up_mid = tla.tile_view(
                            l1_b,
                            tla.make_shape(l1_tk_step, half_pad),
                            tla.make_coord(c0, c1),
                        )
                        l1_b_up = tla.tile_view(
                            l1_b_up_mid,
                            tla.make_shape(l1_tk_step, half_raw),
                            tla.make_coord(c0, c0),
                        )
                        # CPP DoAllLoc2UbAswt: CopyInA1 → CopyInC1(iter0) →
                        # CopyInB1 under one MTE1_MTE2 credit (no bias SoftFlag).
                        # A then B halves (plain copy; half-N keeps parent L1 pitch via tile_view).
                        tla.copy(l1_a, gm_a_l1)
                        if tla.const_expr(USE_BIAS):
                            if k_l1 == c0:
                                tla.copy(l1_bias_gate, gm_bias_gate)
                                tla.copy(l1_bias_up, gm_bias_up)
                        tla.copy(l1_b_gate, gm_b_gate)
                        tla.copy(l1_b_up, gm_b_up)
                    else:
                        # CPP order: A → bias(iter0) → B (same l1 SoftFlag).
                        tla.copy(l1_a, gm_a_l1)
                        if tla.const_expr(USE_BIAS):
                            if k_l1 == c0:
                                tla.copy(l1_bias, gm_bias_by_core)
                        tla.copy(l1_b, gm_b_l1)
                    if tla.const_expr(_single_kl1):
                        tla.set_flag(l1_0_data_ready)
                    elif l1_buf_idx == c0:
                        tla.set_flag(l1_0_data_ready)
                    else:
                        tla.set_flag(l1_1_data_ready)
                    if tla.const_expr(_single_kl1):
                        tla.wait_flag(l1_0_data_ready)
                    elif l1_buf_idx == c0:
                        tla.wait_flag(l1_0_data_ready)
                    else:
                        tla.wait_flag(l1_1_data_ready)

                    if tla.const_expr(_dual_l0):
                        # ---- L0-0 @ slot0 (init_c when first kL1) ----
                        l1_a_l0_0 = tla.tile_view(
                            l1_a,
                            tla.make_shape(work_tm, _tiling.l0_tk),
                            tla.make_coord(c0, c0),
                        )
                        l1_b_l0_0 = tla.tile_view(
                            l1_b,
                            tla.make_shape(_tiling.l0_tk, cube_tn),
                            tla.make_coord(c0, c0),
                        )
                        l0_a_0 = tla.make_tensor_like(l0a0_ptr, l1_a_l0_0)
                        l0_b_0 = tla.make_tensor_like(l0b0_ptr, l1_b_l0_0)
                        tla.wait_flag(l0_0_available)
                        tla.copy(l0_a_0, l1_a_l0_0)
                        # _dual_l0 ⇒ _single_kl1 ⇒ this is always first K (init_c).
                        if tla.const_expr(USE_BIAS):
                            tla.copy(l0_bias, l1_bias)
                        tla.copy(l0_b_0, l1_b_l0_0)
                        tla.set_flag(l0_0_data_ready)
                        tla.wait_flag(l0_0_data_ready)
                        if tla.const_expr(USE_BIAS):
                            tla.mmad(
                                l0_c,
                                l0_a_0,
                                l0_b_0,
                                init_c=True,
                                unit_flag=0,
                                bias=l0_bias,
                            )
                        else:
                            tla.mmad(
                                l0_c,
                                l0_a_0,
                                l0_b_0,
                                init_c=True,
                                unit_flag=0,
                            )
                        tla.set_flag(l0_0_available)
                        # ---- L0-1 @ slot1 (accum; rem K via origin crop) ----
                        l1_a_l0_1 = tla.tile_view(
                            l1_a,
                            tla.make_shape(work_tm, _tiling.l0_tk),
                            tla.make_coord(c0, c1),
                        )
                        l1_b_l0_1 = tla.tile_view(
                            l1_b,
                            tla.make_shape(_tiling.l0_tk, cube_tn),
                            tla.make_coord(c1, c0),
                        )
                        l0_a_1 = tla.make_tensor_like(l0a1_ptr, l1_a_l0_1)
                        l0_b_1 = tla.make_tensor_like(l0b1_ptr, l1_b_l0_1)
                        tla.wait_flag(l0_1_available)
                        tla.copy(l0_a_1, l1_a_l0_1)
                        tla.copy(l0_b_1, l1_b_l0_1)
                        tla.set_flag(l0_1_data_ready)
                        tla.wait_flag(l0_1_data_ready)
                        tla.mmad(l0_c, l0_a_1, l0_b_1, init_c=False, unit_flag=0)
                        tla.set_flag(l0_1_available)
                        l0_buf_idx = c0
                    else:
                        k_l0_count = (
                            l1_a.origin_shape[1] + _tiling.l0_tk - 1
                        ) // _tiling.l0_tk
                        for k_l0 in tla.range(c0, k_l0_count, c1):
                            # L0 MN follows work tile (full l1 or last-wave split);
                            # current FFN tilings keep l0_tm/tn == l1_tm/tn.
                            l1_a_l0 = tla.tile_view(
                                l1_a,
                                tla.make_shape(work_tm, _tiling.l0_tk),
                                tla.make_coord(c0, k_l0),
                            )
                            l1_b_l0 = tla.tile_view(
                                l1_b,
                                tla.make_shape(_tiling.l0_tk, cube_tn),
                                tla.make_coord(k_l0, c0),
                            )

                            l0_a = tla.make_tensor_like(
                                l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr,
                                l1_a_l0,
                            )
                            l0_b = tla.make_tensor_like(
                                l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr,
                                l1_b_l0,
                            )
                            # CPP: Wait M_MTE1(slot) → CopyA2+B2 → Set/Wait MTE1_M
                            # → Mmad → Set M_MTE1. Distinct IDs keep L0 DB live.
                            if l0_buf_idx == c0:
                                tla.wait_flag(l0_0_available)
                            else:
                                tla.wait_flag(l0_1_available)
                            # CPP: Wait M_MTE1 → A2 → C2(NeedBias) → B2 →
                            # Set/Wait MTE1_M → Mmad (no bias SoftFlag).
                            tla.copy(l0_a, l1_a_l0)
                            init_c = True if k_l1 == 0 and k_l0 == 0 else False
                            if tla.const_expr(USE_BIAS):
                                if init_c:
                                    tla.copy(l0_bias, l1_bias)
                            tla.copy(l0_b, l1_b_l0)
                            if l0_buf_idx == c0:
                                tla.set_flag(l0_0_data_ready)
                                tla.wait_flag(l0_0_data_ready)
                            else:
                                tla.set_flag(l0_1_data_ready)
                                tla.wait_flag(l0_1_data_ready)
                            if tla.const_expr(USE_BIAS):
                                if init_c:
                                    tla.mmad(
                                        l0_c,
                                        l0_a,
                                        l0_b,
                                        init_c=init_c,
                                        unit_flag=0,
                                        bias=l0_bias,
                                    )
                                else:
                                    tla.mmad(
                                        l0_c,
                                        l0_a,
                                        l0_b,
                                        init_c=init_c,
                                        unit_flag=0,
                                    )
                            else:
                                tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=0)
                            if l0_buf_idx == c0:
                                tla.set_flag(l0_0_available)
                            else:
                                tla.set_flag(l0_1_available)
                            if tla.const_expr((not L0_SINGLE) and (not _dual_phase)):
                                l0_buf_idx = c1 - l0_buf_idx

                    # CPP DoAllLoc2UbAswt: Set MTE1_MTE2(l1BufId) after full kL0Iter.
                    if tla.const_expr(_single_kl1):
                        tla.set_flag(l1_0_available)
                    elif l1_buf_idx == c0:
                        tla.set_flag(l1_0_available)
                    else:
                        tla.set_flag(l1_1_available)
                    if tla.const_expr(
                        (not l1_single) and (not _dual_phase) and (not _single_kl1)
                    ):
                        l1_buf_idx = c1 - l1_buf_idx
                    # CPP abL1LoopCnt_++ → next tile/kL1 uses other BT slot.
                    if tla.const_expr(USE_BIAS):
                        bias_buf_idx = c1 - bias_buf_idx

                else:
                    for k_l1 in k_l1_range:
                        gm_a_l1 = gm_l1_tile(
                            gm_a_by_core, work_tm, l1_tk_step, c0, k_l1
                        )
                        gm_b_l1 = gm_l1_tile(
                            gm_b_by_core, l1_tk_step, work_tn, k_l1, c0
                        )
                        if tla.const_expr(_single_kl1):
                            _l1a_ptr = l1a0_ptr
                            _l1b_ptr = l1b0_ptr
                        else:
                            _l1a_ptr = l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr
                            _l1b_ptr = l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr
                        l1_a = tla.make_tensor_like(_l1a_ptr, gm_a_l1)
                        if tla.const_expr(_swiglu_b_cpp):
                            # Parent L1 N = cube_tn (padded); halves load half_raw.
                            l1_b = tla.make_tensor(
                                _l1b_ptr,
                                tla.make_layout_with_tag(
                                    (l1_tk_step, cube_tn),
                                    gm_b.ptr.dtype,
                                    tla.arch.nZ,
                                ),
                            )
                        else:
                            l1_b = tla.make_tensor_like(_l1b_ptr, gm_b_l1)
                        # CPP DoAllLoc2UbAswt: Wait MTE1_MTE2 → CopyA+B →
                        # Set+Wait MTE2_MTE1 (eager drain).
                        if tla.const_expr(_single_kl1):
                            tla.wait_flag(l1_0_available)
                        elif l1_buf_idx == c0:
                            tla.wait_flag(l1_0_available)
                        else:
                            tla.wait_flag(l1_1_available)
                        if tla.const_expr(_swiglu_b_cpp):
                            # CPP CopyInB1 enableSwigluSingle (field-level):
                            #   nValue=halfW (=half_raw), dValue=curKL1,
                            #   srcNdMatrixStride=0, dstNzMatrixStride=0,
                            #   dstNzC0Stride=Align(curNL1)=cube_tn (parent pitch),
                            #   gate @ L1+0; up @ L1[(curNL1/2)*C0] (=half_pad*C0).
                            # GM: gate root + tile; up root = biasGlobal[n_/2] style
                            # via gm_up_root (coord 1 × SWIGLU_H).
                            gm_b_gate = gm_l1_abs(
                                gm_gate_root,
                                l1_tk_step,
                                half_raw,
                                k_l1 * l1_tk_step,
                                _half_col0,
                            )
                            gm_b_up = gm_l1_abs(
                                gm_up_root,
                                l1_tk_step,
                                half_raw,
                                k_l1 * l1_tk_step,
                                _half_col0,
                            )
                            l1_b_gate = tla.tile_view(
                                l1_b,
                                tla.make_shape(l1_tk_step, half_raw),
                                tla.make_coord(c0, c0),
                            )
                            # Nest: mid @ half_pad (=curNL1/2), then crop to half_raw
                            # (no-pad: half_raw==half_pad → identity).
                            l1_b_up_mid = tla.tile_view(
                                l1_b,
                                tla.make_shape(l1_tk_step, half_pad),
                                tla.make_coord(c0, c1),
                            )
                            l1_b_up = tla.tile_view(
                                l1_b_up_mid,
                                tla.make_shape(l1_tk_step, half_raw),
                                tla.make_coord(c0, c0),
                            )
                            # CPP DoAllLoc2UbAswt: CopyInA1 → CopyInC1(iter0) →
                            # CopyInB1 under one MTE1_MTE2 credit (no bias SoftFlag).
                            # A then B halves (plain copy; half-N keeps parent L1 pitch via tile_view).
                            tla.copy(l1_a, gm_a_l1)
                            if tla.const_expr(USE_BIAS):
                                if k_l1 == c0:
                                    tla.copy(l1_bias_gate, gm_bias_gate)
                                    tla.copy(l1_bias_up, gm_bias_up)
                            tla.copy(l1_b_gate, gm_b_gate)
                            tla.copy(l1_b_up, gm_b_up)
                        else:
                            # CPP order: A → bias(iter0) → B (same l1 SoftFlag).
                            tla.copy(l1_a, gm_a_l1)
                            if tla.const_expr(USE_BIAS):
                                if k_l1 == c0:
                                    tla.copy(l1_bias, gm_bias_by_core)
                            tla.copy(l1_b, gm_b_l1)
                        if tla.const_expr(_single_kl1):
                            tla.set_flag(l1_0_data_ready)
                        elif l1_buf_idx == c0:
                            tla.set_flag(l1_0_data_ready)
                        else:
                            tla.set_flag(l1_1_data_ready)
                        if tla.const_expr(_single_kl1):
                            tla.wait_flag(l1_0_data_ready)
                        elif l1_buf_idx == c0:
                            tla.wait_flag(l1_0_data_ready)
                        else:
                            tla.wait_flag(l1_1_data_ready)

                        if tla.const_expr(_dual_l0):
                            # ---- L0-0 @ slot0 (init_c when first kL1) ----
                            l1_a_l0_0 = tla.tile_view(
                                l1_a,
                                tla.make_shape(work_tm, _tiling.l0_tk),
                                tla.make_coord(c0, c0),
                            )
                            l1_b_l0_0 = tla.tile_view(
                                l1_b,
                                tla.make_shape(_tiling.l0_tk, cube_tn),
                                tla.make_coord(c0, c0),
                            )
                            l0_a_0 = tla.make_tensor_like(l0a0_ptr, l1_a_l0_0)
                            l0_b_0 = tla.make_tensor_like(l0b0_ptr, l1_b_l0_0)
                            tla.wait_flag(l0_0_available)
                            tla.copy(l0_a_0, l1_a_l0_0)
                            # _dual_l0 ⇒ _single_kl1 ⇒ this is always first K (init_c).
                            if tla.const_expr(USE_BIAS):
                                tla.copy(l0_bias, l1_bias)
                            tla.copy(l0_b_0, l1_b_l0_0)
                            tla.set_flag(l0_0_data_ready)
                            tla.wait_flag(l0_0_data_ready)
                            if tla.const_expr(USE_BIAS):
                                tla.mmad(
                                    l0_c,
                                    l0_a_0,
                                    l0_b_0,
                                    init_c=True,
                                    unit_flag=0,
                                    bias=l0_bias,
                                )
                            else:
                                tla.mmad(
                                    l0_c,
                                    l0_a_0,
                                    l0_b_0,
                                    init_c=True,
                                    unit_flag=0,
                                )
                            tla.set_flag(l0_0_available)
                            # ---- L0-1 @ slot1 (accum; rem K via origin crop) ----
                            l1_a_l0_1 = tla.tile_view(
                                l1_a,
                                tla.make_shape(work_tm, _tiling.l0_tk),
                                tla.make_coord(c0, c1),
                            )
                            l1_b_l0_1 = tla.tile_view(
                                l1_b,
                                tla.make_shape(_tiling.l0_tk, cube_tn),
                                tla.make_coord(c1, c0),
                            )
                            l0_a_1 = tla.make_tensor_like(l0a1_ptr, l1_a_l0_1)
                            l0_b_1 = tla.make_tensor_like(l0b1_ptr, l1_b_l0_1)
                            tla.wait_flag(l0_1_available)
                            tla.copy(l0_a_1, l1_a_l0_1)
                            tla.copy(l0_b_1, l1_b_l0_1)
                            tla.set_flag(l0_1_data_ready)
                            tla.wait_flag(l0_1_data_ready)
                            tla.mmad(l0_c, l0_a_1, l0_b_1, init_c=False, unit_flag=0)
                            tla.set_flag(l0_1_available)
                            l0_buf_idx = c0
                        else:
                            k_l0_count = (
                                l1_a.origin_shape[1] + _tiling.l0_tk - 1
                            ) // _tiling.l0_tk
                            for k_l0 in tla.range(c0, k_l0_count, c1):
                                # L0 MN follows work tile (full l1 or last-wave split);
                                # current FFN tilings keep l0_tm/tn == l1_tm/tn.
                                l1_a_l0 = tla.tile_view(
                                    l1_a,
                                    tla.make_shape(work_tm, _tiling.l0_tk),
                                    tla.make_coord(c0, k_l0),
                                )
                                l1_b_l0 = tla.tile_view(
                                    l1_b,
                                    tla.make_shape(_tiling.l0_tk, cube_tn),
                                    tla.make_coord(k_l0, c0),
                                )

                                l0_a = tla.make_tensor_like(
                                    l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr,
                                    l1_a_l0,
                                )
                                l0_b = tla.make_tensor_like(
                                    l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr,
                                    l1_b_l0,
                                )
                                # CPP: Wait M_MTE1(slot) → CopyA2+B2 → Set/Wait MTE1_M
                                # → Mmad → Set M_MTE1. Distinct IDs keep L0 DB live.
                                if l0_buf_idx == c0:
                                    tla.wait_flag(l0_0_available)
                                else:
                                    tla.wait_flag(l0_1_available)
                                # CPP: Wait M_MTE1 → A2 → C2(NeedBias) → B2 →
                                # Set/Wait MTE1_M → Mmad (no bias SoftFlag).
                                tla.copy(l0_a, l1_a_l0)
                                init_c = True if k_l1 == 0 and k_l0 == 0 else False
                                if tla.const_expr(USE_BIAS):
                                    if init_c:
                                        tla.copy(l0_bias, l1_bias)
                                tla.copy(l0_b, l1_b_l0)
                                if l0_buf_idx == c0:
                                    tla.set_flag(l0_0_data_ready)
                                    tla.wait_flag(l0_0_data_ready)
                                else:
                                    tla.set_flag(l0_1_data_ready)
                                    tla.wait_flag(l0_1_data_ready)
                                if tla.const_expr(USE_BIAS):
                                    if init_c:
                                        tla.mmad(
                                            l0_c,
                                            l0_a,
                                            l0_b,
                                            init_c=init_c,
                                            unit_flag=0,
                                            bias=l0_bias,
                                        )
                                    else:
                                        tla.mmad(
                                            l0_c,
                                            l0_a,
                                            l0_b,
                                            init_c=init_c,
                                            unit_flag=0,
                                        )
                                else:
                                    tla.mmad(
                                        l0_c, l0_a, l0_b, init_c=init_c, unit_flag=0
                                    )
                                if l0_buf_idx == c0:
                                    tla.set_flag(l0_0_available)
                                else:
                                    tla.set_flag(l0_1_available)
                                if tla.const_expr(
                                    (not L0_SINGLE) and (not _dual_phase)
                                ):
                                    l0_buf_idx = c1 - l0_buf_idx

                        # CPP DoAllLoc2UbAswt: Set MTE1_MTE2(l1BufId) after full kL0Iter.
                        if tla.const_expr(_single_kl1):
                            tla.set_flag(l1_0_available)
                        elif l1_buf_idx == c0:
                            tla.set_flag(l1_0_available)
                        else:
                            tla.set_flag(l1_1_available)
                        if tla.const_expr(
                            (not l1_single) and (not _dual_phase) and (not _single_kl1)
                        ):
                            l1_buf_idx = c1 - l1_buf_idx
                        # CPP abL1LoopCnt_++ → next tile/kL1 uses other BT slot.
                        if tla.const_expr(USE_BIAS):
                            bias_buf_idx = c1 - bias_buf_idx

                # FixPipe L0C→UB (SPLIT_M).
                actual_m = gm_c_by_core.origin_shape[0]
                # FixPipe N follows cube frame (padded for SwiGLU B_CPP tail).
                actual_n = cube_tn
                ub_pitch = (
                    (actual_n + (_fixpipe_align - 1)) // _fixpipe_align
                ) * _fixpipe_align
                ub_c = tla.make_tensor(
                    ub_acc_ptr,
                    tla.make_layout(
                        tla.make_shape(actual_m, actual_n),
                        tla.make_stride(ub_pitch, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                # CPP: Set/Wait M_FIX(slot) then FixPipe; Set FIX_M(slot) no wait.
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
                    ub_c,
                    l0_c,
                    tla.params.CopyL0C2DstParams(
                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                    ),
                )
                if tla.const_expr(_l0c_slot0_only):
                    tla.set_flag(l0c0_available)
                elif l0c_slot == c0:
                    tla.set_flag(l0c0_available)
                else:
                    tla.set_flag(l0c1_available)
                if tla.const_expr(_l0c_db and (not _l0c_slot0_only)):
                    l0c_slot = c1 - l0c_slot

                # CPP: count++; Set aic_finish(countId); enableCVSync=true.
                cv_count = cv_count + c1
                _cv_set_aic(
                    aic_finish0,
                    aic_finish1,
                    aic_finish2,
                    cv_count,
                    tla.arch.FIX,
                    _cv_id0_only,
                )
                enable_cv_sync = c1

        # CPP ProcessTiles end: ProcessAicSync(enableCVSync, count, countId).
        # Must stay dynamic: idle cores (tiles < block_dim) never Set aic_finish
        # / aiv_finish — unconditional Wait deadlocks (aicore timeout).
        if enable_cv_sync != c0:
            _cv_wait_aiv(
                aiv_finish0, aiv_finish1, aiv_finish2, cv_count, _wp, _cv_id0_only
            )

        # CPP ~BlockMmad: Wait MTE1_MTE2(0..3), M_MTE1(6/7), FIX_M(0/1).
        # Bias trio is DSL credit-only (CPP folds CopyInC under A/B).
        tla.wait_flag(l1_0_available)
        tla.wait_flag(l1_1_ctor)
        tla.wait_flag(l1_2_available)
        tla.wait_flag(l1_3_available)
        tla.wait_flag(l0_0_available)
        tla.wait_flag(l0_1_ctor)
        tla.wait_flag(l0c0_available)
        tla.wait_flag(l0c1_available)
        if tla.const_expr(USE_BIAS):
            tla.wait_flag(l1_bias_data_ready)
            tla.wait_flag(l1_bias_available)
            tla.wait_flag(l0_bias_available)

    with tla.vector():
        aiv_sub_idx = tla.arch.sub_block_idx()
        block_idx = tla.arch.block_idx()
        block_num = tla.arch.block_num()
        rem_tiles = total_blocks % block_num
        main_tiles = total_blocks - rem_tiles
        if tla.const_expr(use_mtail):
            expanded = main_tiles + rem_tiles * tail_cnt
        else:
            expanded = total_blocks
        block_range = tla.range(block_idx, expanded, block_num)

        # CPP Elementwise has no UB available SoftFlag ctor — only out-PP
        # paths prime credit.
        if tla.const_expr(_ub_pp_sf):
            tla.set_flag(ub_out0_available)
            if tla.const_expr(use_out_pp):
                tla.set_flag(ub_out1_available)

        # CPP ProcessAivEpilogue: independent count synced with AIC by tile order.
        cv_count = c0
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
            block_row = c0
            block_col = c0
            if tla.const_expr(_swizzle_one):
                if tla.const_expr(_swizzle.SWIZZLE_DIRECTION == 0):
                    block_row = block_linear % grid_m
                    block_col = block_linear // grid_m
                else:
                    block_row = block_linear // grid_n
                    block_col = block_linear % grid_n
            elif tla.const_expr(_swizzle.SWIZZLE_DIRECTION == 0):
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
            else:
                swizzle_tile_idx = block_linear // (_swizzle.SWIZZLE_OFFSET * grid_m)
                in_tile_idx = block_linear % (_swizzle.SWIZZLE_OFFSET * grid_m)
                swizzle_n_cols = (
                    _swizzle.SWIZZLE_OFFSET
                    if swizzle_tile_idx != (swizzle_tile_count - 1)
                    else (grid_n - _swizzle.SWIZZLE_OFFSET * swizzle_tile_idx)
                )
                block_row = in_tile_idx // swizzle_n_cols
                block_col = (
                    swizzle_tile_idx * _swizzle.SWIZZLE_OFFSET
                    + in_tile_idx % swizzle_n_cols
                )
                block_row = (
                    (grid_m - block_row - 1)
                    if (swizzle_tile_idx % 2 == 1)
                    else block_row
                )

            blk_m = m - block_row * _tiling.l1_tm
            blk_n = n - block_col * _tiling.l1_tn
            if blk_m > _tiling.l1_tm:
                blk_m = tla.as_numeric(_tiling.l1_tm)
            if blk_n > _tiling.l1_tn:
                blk_n = tla.as_numeric(_tiling.l1_tn)
            split_tm_rt = (blk_m + m_tail - 1) // m_tail
            split_tn_rt = (blk_n + n_tail - 1) // n_tail
            split_empty = c0
            if tla.const_expr(use_mtail):
                if task_idx >= main_tiles:
                    if (m_si * split_tm_rt) >= blk_m:
                        split_empty = c1
                    if (n_si * split_tn_rt) >= blk_n:
                        split_empty = c1

            # CPP: count++; Wait aic_finish(countId) on PIPE_V.
            cv_count = cv_count + c1
            _cv_wait_aic(
                aic_finish0,
                aic_finish1,
                aic_finish2,
                cv_count,
                aiv_sub_idx,
                _cv_id0_only,
            )
            ub_acc_ptr = ub_acc_ptr0

            if split_empty != c0:
                # CPP empty AIV: always CrossCoreSetFlag@PIPE_MTE3 (incl. last tile).
                _cv_set_aiv(
                    aiv_finish0,
                    aiv_finish1,
                    aiv_finish2,
                    cv_count,
                    aiv_sub_idx,
                    tla.arch.MTE3,
                    _cv_id0_only,
                )
            else:
                # Acc geometry from gm_c_geom (full Up N); store into gm_d (H, or H after SwiGLU).
                gm_c_full = tla.tile_view(
                    gm_c_geom,
                    tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                    tla.make_coord(block_row, block_col),
                )
                # Single SwiGLU: FixPipe N=2H, store N=H. RAW/dual/SiLU/GELU: N=H both.
                if tla.const_expr(ACT == ACT_SWIGLU):
                    out_tn = _tiling.l1_tn // 2
                    out_split = split_tn_rt // 2
                else:
                    out_tn = _tiling.l1_tn
                    out_split = split_tn_rt
                gm_out_full = tla.tile_view(
                    gm_d,
                    tla.make_shape(_tiling.l1_tm, out_tn),
                    tla.make_coord(block_row, block_col),
                )
                if tla.const_expr(ACT == ACT_SWIGLU_DUAL):
                    gm_gate_full = tla.tile_view(
                        gm_gate,
                        tla.make_shape(_tiling.l1_tm, out_tn),
                        tla.make_coord(block_row, block_col),
                    )
                else:
                    gm_gate_full = gm_out_full
                if tla.const_expr(use_mtail):
                    cur_tm = tla.as_numeric(_tiling.l1_tm)
                    cur_tn = tla.as_numeric(_tiling.l1_tn)
                    cur_out_tn = tla.as_numeric(out_tn)
                    m_idx = c0
                    n_idx = c0
                    out_n_idx = c0
                    if task_idx >= main_tiles:
                        cur_tm = split_tm_rt
                        cur_tn = split_tn_rt
                        cur_out_tn = out_split
                        m_idx = m_si
                        n_idx = n_si
                        out_n_idx = n_si
                    gm_c_by_core = tla.tile_view(
                        gm_c_full,
                        tla.make_shape(cur_tm, cur_tn),
                        tla.make_coord(m_idx, n_idx),
                    )
                    gm_out_by_core = tla.tile_view(
                        gm_out_full,
                        tla.make_shape(cur_tm, cur_out_tn),
                        tla.make_coord(m_idx, out_n_idx),
                    )
                    gm_gate_by_core = tla.tile_view(
                        gm_gate_full,
                        tla.make_shape(cur_tm, cur_out_tn),
                        tla.make_coord(m_idx, out_n_idx),
                    )
                else:
                    gm_c_by_core = gm_c_full
                    gm_out_by_core = gm_out_full
                    gm_gate_by_core = gm_gate_full
                # Bake M when host M%tm==0 (even tm → equal AIV split). N: bake
                # full vs rem via block_col when host knows rem; else origin_shape.
                if tla.const_expr(_exact_m):
                    actual_m = _tiling.l1_tm
                    aiv_m_req = _tiling.l1_tm // 2
                    if tla.const_expr(_has_n_rem):
                        actual_n = (
                            _tiling.l1_tn if block_col != (grid_n - 1) else _n_rem
                        )
                    else:
                        actual_n = gm_c_by_core.origin_shape[1]
                    out_n_tile = (
                        actual_n // 2 if tla.const_expr(ACT == ACT_SWIGLU) else actual_n
                    )
                else:
                    actual_m = gm_c_by_core.origin_shape[0]
                    actual_n = gm_c_by_core.origin_shape[1]
                    aiv_m_req = (actual_m + 1) // 2
                    out_n_tile = gm_out_by_core.origin_shape[1]
                gm_result = tla.tile_view(
                    gm_out_by_core,
                    tla.make_shape(aiv_m_req, out_n_tile),
                    tla.make_coord(aiv_sub_idx, c0),
                )
                gm_gate_result = tla.tile_view(
                    gm_gate_by_core,
                    tla.make_shape(aiv_m_req, out_n_tile),
                    tla.make_coord(aiv_sub_idx, c0),
                )
                # Exact-M: both AIV own aiv_m_req rows (tm even). Else crop.
                aiv_m = aiv_m_req
                if tla.const_expr(not _exact_m):
                    aiv_m = gm_result.origin_shape[0]
                # FixPipe / UB-acc width: CPP Align(raw,32) when SwiGLU B_CPP
                # N-tail %32!=0; else raw tile N (2H for SwiGLU).
                aiv_n = actual_n
                if tla.const_expr(_swiglu_b_cpp):
                    aiv_n = ((actual_n + 31) // 32) * 32
                aiv_out_n = out_n_tile
                if tla.const_expr(not _exact_m):
                    aiv_out_n = gm_result.origin_shape[1]
                # SwiGLU epi packs half of FixPipe width per row; GM store
                # truncates to raw half (CPP writeWidth = min(outWidth, valid)).
                aiv_half_pad = aiv_n // 2
                aiv_out_store_stride = aiv_out_n
                if tla.const_expr(ACT == ACT_SWIGLU):
                    aiv_out_store_stride = aiv_half_pad

                ub_pitch = (
                    (aiv_n + (_fixpipe_align - 1)) // _fixpipe_align
                ) * _fixpipe_align
                ub_c_full = tla.make_tensor(
                    ub_acc_ptr,
                    tla.make_layout(
                        tla.make_shape(aiv_m, aiv_n),
                        tla.make_stride(ub_pitch, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )

                ub_stage_idx = c0
                # Contiguous when pitch==width → flatten to 1D VL (Reg*B16-style).
                acc_contiguous = aiv_n == ub_pitch
                # CPP block_epilogue / empty: always CrossCoreSetFlag@PIPE_MTE3.
                # Do not gate on finish_on_mte3 — TLA dynamic-if SSA can leave that
                # flag 0 after oneshot store and publish on VECTOR while MTE3 copy
                # still runs (aic_mte2/-scalar bubble; aiv_mte3_instr −1 vs CPP).
                if aiv_n <= UB_SLOT_ELEMS:
                    out_row_stride = (
                        (aiv_out_n + BYTE_PER_C0 - 1) // BYTE_PER_C0
                    ) * BYTE_PER_C0
                    # Row budget from out slot (fallback path); oneshot packs into acc.
                    max_rows = UB_SLOT_ELEMS // out_row_stride
                    if max_rows < 1:
                        max_rows = 1
                    n_epilogue_row_tiles = (aiv_m + max_rows - 1) // max_rows
                    out_contiguous = out_row_stride == aiv_out_n
                    # Strip/2d fallback flat VL needs both acc and out-slot pitches.
                    use_flat_silu = acc_contiguous and out_contiguous
                    tile_cap = max_rows * aiv_n
                    flat_elems = aiv_m * aiv_n
                    # Epi half-width follows FixPipe frame (half_pad); GM is raw.
                    flat_out_elems = aiv_m * aiv_out_store_stride
                    ub_acc_1d = tla.make_tensor(
                        ub_acc_ptr,
                        tla.make_layout(
                            tla.make_shape(flat_elems),
                            tla.make_stride(c1),
                        ),
                    )
                    # Oneshot: flat layout → inplace pack (CPP Reg*B16), MTE3→GM.
                    # Do NOT gate on out-slot C0 pad: half_w=80 → Align(80,32)=96
                    # made out_contiguous=False and forced pitched 2d (aiv_vec~7µs).
                    can_oneshot = acc_contiguous
                    # CPP FusionSwigluSingle staging: only when stageSize < aiv strip.
                    # lastUB after FixPipe strip; stageNum=2 (see fusion_swiglu_single.h).
                    # Host L1 tile that already guarantees stage≥strip (common ASWT
                    # SwiGLU): skip per-tile div/mod (AIV scalar + AIC CrossCore stall).
                    # Large tiles keep the runtime stage path below.
                    _max_aiv_m = (_tiling.l1_tm + 1) // 2
                    _max_aiv_n = max(int(_tiling.l1_tn), 1)
                    _max_fp = int(_max_aiv_m) * int(_max_aiv_n)
                    _min_last = int(UB_SIZE) - _max_fp * 4
                    _min_stage = (
                        _min_last // _SWIGLU_STAGE_NUM // 4 // _max_aiv_n
                    ) * _max_aiv_n
                    _swiglu_tile_oneshot = (
                        ACT == ACT_SWIGLU and _max_fp > 0 and _min_stage >= _max_fp
                    )
                    if tla.const_expr(_swiglu_tile_oneshot):
                        use_stage_inplace = False
                        use_oneshot = can_oneshot
                        _stage_rows_use = max_rows
                    else:
                        # Branchless for TLA dynamic-if SSA (init before runtime if).
                        _fixpipe_elems = aiv_m * aiv_n
                        _last_ub = UB_SIZE - _fixpipe_elems * 4
                        _cpp_stage_elems = (
                            _last_ub // _SWIGLU_STAGE_NUM // 4 // aiv_n
                        ) * aiv_n
                        if _cpp_stage_elems > _fixpipe_elems:
                            _cpp_stage_elems = _fixpipe_elems
                        if _cpp_stage_elems < aiv_n:
                            _cpp_stage_elems = aiv_n
                        _cpp_stage_rows = _cpp_stage_elems // aiv_n
                        _stage_rows_use = _cpp_stage_rows
                        if _stage_rows_use < 1:
                            _stage_rows_use = 1
                        if _stage_rows_use > max_rows:
                            _stage_rows_use = max_rows
                        use_stage_inplace = (
                            can_oneshot
                            and tla.const_expr(ACT == ACT_SWIGLU)
                            and (_stage_rows_use < aiv_m)
                        )
                        use_oneshot = can_oneshot and (not use_stage_inplace)

                    if use_oneshot:
                        # Oneshot: inplace pack into FixPipe acc (CPP).
                        tla.set_flag(mte3_v_fence)
                        tla.wait_flag(mte3_v_fence)
                        out_ptr = tla.recast_ptr(ub_acc_ptr, dtype=dtype_gm_c)
                        out_1d = tla.make_tensor(
                            out_ptr,
                            tla.make_layout(
                                tla.make_shape(flat_out_elems),
                                tla.make_stride(c1),
                            ),
                        )
                        if tla.const_expr(ACT == ACT_SWIGLU):
                            epi_apply_swiglu_vl_1d(
                                ub_acc_1d,
                                out_1d,
                                aiv_m,
                                aiv_n,
                                SIMD_LANES,
                                dtype_gm_c,
                                _swiglu_use_all_mask,
                                _swiglu_tail_lanes,
                                0,
                                _tiling.l1_tn,
                            )
                        elif tla.const_expr(ACT == ACT_SWIGLU_DUAL):
                            # Gate → dedicated f32 slot; out PACK inplace on acc.
                            ub_gate_full = tla.make_tensor(
                                ub_gate_slot_ptr,
                                tla.make_layout(
                                    tla.make_shape(aiv_m, aiv_out_n),
                                    tla.make_stride(aiv_out_n, 1),
                                    layoutTag=tla.arch.RowMajor,
                                ),
                            )
                            tla.copy(dst=ub_gate_full, src=gm_gate_result)
                            tla.set_flag(gate_mte2_v)
                            tla.wait_flag(gate_mte2_v)
                            ub_gate_1d = tla.make_tensor(
                                ub_gate_slot_ptr,
                                tla.make_layout(
                                    tla.make_shape(flat_out_elems),
                                    tla.make_stride(c1),
                                ),
                            )
                            out_ptr = tla.recast_ptr(ub_acc_ptr, dtype=dtype_gm_c)
                            out_1d = tla.make_tensor(
                                out_ptr,
                                tla.make_layout(
                                    tla.make_shape(flat_out_elems),
                                    tla.make_stride(c1),
                                ),
                            )
                            epi_apply_swiglu_dual_vl_1d(
                                ub_acc_1d,
                                ub_gate_1d,
                                out_1d,
                                flat_out_elems,
                                SIMD_LANES,
                                dtype_gm_c,
                            )
                        else:
                            epi_apply_vl_1d(
                                ACT,
                                ub_acc_1d,
                                out_1d,
                                flat_elems,
                                SIMD_LANES,
                                dtype_gm_c,
                            )
                        ub_out_full = tla.make_tensor(
                            out_ptr,
                            tla.make_layout(
                                tla.make_shape(aiv_m, aiv_out_n),
                                tla.make_stride(aiv_out_store_stride, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.set_flag(ub_out0_data_ready)
                        tla.wait_flag(ub_out0_data_ready)
                        tla.copy(gm_result, ub_out_full)
                    elif use_stage_inplace:
                        # CPP stage height (UB-driven or override).
                        stage_rows = _stage_rows_use
                        n_stage = (aiv_m + stage_rows - 1) // stage_rows
                        acc_tile_cap = stage_rows * aiv_n
                        out_tile_cap = stage_rows * aiv_out_store_stride
                        for tile_i in tla.range(c0, n_stage, c1):
                            gm_d_tile = tla.tile_view(
                                gm_result,
                                tla.make_shape(stage_rows, aiv_out_n),
                                tla.make_coord(tile_i, c0),
                            )
                            tile_rows = gm_d_tile.origin_shape[0]
                            # Inplace into this stage's fp32 strip (CPP cLocal_[stageOffset]).
                            stage_acc_ptr = ub_acc_ptr + tile_i * acc_tile_cap
                            acc_tile_1d = tla.make_tensor(
                                stage_acc_ptr,
                                tla.make_layout(
                                    tla.make_shape(acc_tile_cap),
                                    tla.make_stride(c1),
                                ),
                            )
                            stage_out_ptr = tla.recast_ptr(
                                stage_acc_ptr, dtype=dtype_gm_c
                            )
                            out_tile_1d = tla.make_tensor(
                                stage_out_ptr,
                                tla.make_layout(
                                    tla.make_shape(out_tile_cap),
                                    tla.make_stride(c1),
                                ),
                            )
                            # CPP FusionSwigluSingle: Set+Wait MTE3_V(0) before VF;
                            # Elementwise: Set+Wait V_MTE3(0) before store. No
                            # available SoftFlag between stages (=CPP).
                            tla.set_flag(mte3_v_fence)
                            tla.wait_flag(mte3_v_fence)
                            if tla.const_expr(_ub_pp_sf):
                                tla.wait_flag(ub_out0_available)
                            epi_apply_swiglu_vl_1d(
                                acc_tile_1d,
                                out_tile_1d,
                                tile_rows,
                                aiv_n,
                                SIMD_LANES,
                                dtype_gm_c,
                                _swiglu_use_all_mask,
                                _swiglu_tail_lanes,
                                0,
                                _tiling.l1_tn,
                            )
                            ub_out_tile = tla.make_tensor(
                                stage_out_ptr,
                                tla.make_layout(
                                    tla.make_shape(tile_rows, aiv_out_n),
                                    tla.make_stride(aiv_out_store_stride, 1),
                                    layoutTag=tla.arch.RowMajor,
                                ),
                            )
                            tla.set_flag(ub_out0_data_ready)
                            tla.wait_flag(ub_out0_data_ready)
                            tla.copy(gm_d_tile, ub_out_tile)
                            if tla.const_expr(_ub_pp_sf):
                                tla.set_flag(ub_out0_available)
                    else:
                        for tile_i in tla.range(c0, n_epilogue_row_tiles, c1):
                            gm_d_tile = tla.tile_view(
                                gm_result,
                                tla.make_shape(max_rows, aiv_out_n),
                                tla.make_coord(tile_i, c0),
                            )
                            tile_rows = gm_d_tile.origin_shape[0]
                            tile_cols = gm_d_tile.origin_shape[1]
                            ub_out_layout = tla.make_layout(
                                tla.make_shape(tile_rows, tile_cols),
                                tla.make_stride(out_row_stride, 1),
                                layoutTag=tla.arch.RowMajor,
                            )
                            if tla.const_expr(use_out_pp):
                                use_ub_stage1 = ub_stage_idx != c0
                            else:
                                use_ub_stage1 = False
                            ub_out_ptr = ub_out_ptr1 if use_ub_stage1 else ub_out_ptr0
                            ub_result = tla.make_tensor(ub_out_ptr, ub_out_layout)

                            if tla.const_expr(_ub_pp_sf):
                                if use_ub_stage1:
                                    tla.wait_flag(ub_out1_available)
                                else:
                                    tla.wait_flag(ub_out0_available)
                            # CPP FusionSilu/Gelu/SwigluSingle: MTE3_V(0) before VF.
                            tla.set_flag(mte3_v_fence)
                            tla.wait_flag(mte3_v_fence)

                            if tla.const_expr(ACT == ACT_SWIGLU):
                                if use_flat_silu:
                                    # Densified acc strip → half-wide out
                                    # (half_pad frame; GM copy truncates).
                                    acc_tile_cap = max_rows * aiv_n
                                    out_tile_cap = max_rows * aiv_out_store_stride
                                    acc_tile_1d = tla.tile_view(
                                        ub_acc_1d,
                                        tla.make_shape(acc_tile_cap),
                                        tla.make_coord(tile_i),
                                    )
                                    out_tile_1d = tla.make_tensor(
                                        ub_out_ptr,
                                        tla.make_layout(
                                            tla.make_shape(out_tile_cap),
                                            tla.make_stride(c1),
                                        ),
                                    )
                                    epi_apply_swiglu_vl_1d(
                                        acc_tile_1d,
                                        out_tile_1d,
                                        tile_rows,
                                        aiv_n,
                                        SIMD_LANES,
                                        dtype_gm_c,
                                        _swiglu_use_all_mask,
                                        _swiglu_tail_lanes,
                                        0,
                                        _tiling.l1_tn,
                                    )
                                    # Override ub_result for pad truncate store.
                                    ub_result = tla.make_tensor(
                                        ub_out_ptr,
                                        tla.make_layout(
                                            tla.make_shape(tile_rows, aiv_out_n),
                                            tla.make_stride(aiv_out_store_stride, 1),
                                            layoutTag=tla.arch.RowMajor,
                                        ),
                                    )
                                else:
                                    # Pitched FixPipe (N-tail): keep stride=l1_tn.
                                    ub_c_tile = tla.tile_view(
                                        ub_c_full,
                                        tla.make_shape(max_rows, aiv_n),
                                        tla.make_coord(tile_i, c0),
                                    )
                                    epi_apply_vl_2d(
                                        ACT,
                                        ub_c_tile,
                                        ub_result,
                                        tile_rows,
                                        aiv_n,
                                        SIMD_LANES,
                                        dtype_gm_c,
                                    )
                            elif tla.const_expr(ACT == ACT_SWIGLU_DUAL):
                                gm_gate_tile = tla.tile_view(
                                    gm_gate_result,
                                    tla.make_shape(max_rows, aiv_out_n),
                                    tla.make_coord(tile_i, c0),
                                )
                                ub_gate_tile = tla.make_tensor(
                                    ub_gate_slot_ptr,
                                    tla.make_layout(
                                        tla.make_shape(tile_rows, tile_cols),
                                        tla.make_stride(out_row_stride, 1),
                                        layoutTag=tla.arch.RowMajor,
                                    ),
                                )
                                tla.copy(dst=ub_gate_tile, src=gm_gate_tile)
                                tla.set_flag(gate_mte2_v)
                                tla.wait_flag(gate_mte2_v)
                                ub_c_tile = tla.tile_view(
                                    ub_c_full,
                                    tla.make_shape(max_rows, aiv_n),
                                    tla.make_coord(tile_i, c0),
                                )
                                ub_result = tla.make_tensor(ub_out_ptr, ub_out_layout)
                                epi_apply_swiglu_dual_vl_2d(
                                    ub_c_tile,
                                    ub_gate_tile,
                                    ub_result,
                                    tile_rows,
                                    aiv_n,
                                    SIMD_LANES,
                                    dtype_gm_c,
                                )
                            elif use_flat_silu:
                                # 1D VL over densified acc (matches streamk cast / RegSilu loop).
                                tile_elems = tile_rows * aiv_n
                                acc_tile_1d = tla.tile_view(
                                    ub_acc_1d,
                                    tla.make_shape(tile_cap),
                                    tla.make_coord(tile_i),
                                )
                                out_tile_1d = tla.make_tensor(
                                    ub_out_ptr,
                                    tla.make_layout(
                                        tla.make_shape(tile_cap),
                                        tla.make_stride(c1),
                                    ),
                                )
                                epi_apply_vl_1d(
                                    ACT,
                                    acc_tile_1d,
                                    out_tile_1d,
                                    tile_elems,
                                    SIMD_LANES,
                                    dtype_gm_c,
                                )
                            else:
                                ub_c_tile = tla.tile_view(
                                    ub_c_full,
                                    tla.make_shape(max_rows, aiv_n),
                                    tla.make_coord(tile_i, c0),
                                )
                                epi_apply_vl_2d(
                                    ACT,
                                    ub_c_tile,
                                    ub_result,
                                    tile_rows,
                                    tile_cols,
                                    SIMD_LANES,
                                    dtype_gm_c,
                                )

                            if use_ub_stage1:
                                tla.set_flag(ub_out1_data_ready)
                                tla.wait_flag(ub_out1_data_ready)
                            else:
                                tla.set_flag(ub_out0_data_ready)
                                tla.wait_flag(ub_out0_data_ready)
                            tla.copy(gm_d_tile, ub_result)
                            if tla.const_expr(_ub_pp_sf):
                                if use_ub_stage1:
                                    tla.set_flag(ub_out1_available)
                                else:
                                    tla.set_flag(ub_out0_available)

                            if tla.const_expr(use_out_pp):
                                ub_stage_idx = c1 - ub_stage_idx
                else:
                    n_col_tiles = (aiv_n + UB_SLOT_ELEMS - 1) // UB_SLOT_ELEMS
                    for row_i in tla.range(c0, aiv_m, c1):
                        for col_tile in tla.range(c0, n_col_tiles, c1):
                            gm_d_tile = tla.tile_view(
                                gm_result,
                                tla.make_shape(c1, UB_SLOT_ELEMS),
                                tla.make_coord(row_i, col_tile),
                            )
                            ub_c_tile = tla.tile_view(
                                ub_c_full,
                                tla.make_shape(c1, UB_SLOT_ELEMS),
                                tla.make_coord(row_i, col_tile),
                            )
                            tile_rows = gm_d_tile.origin_shape[0]
                            tile_cols = gm_d_tile.origin_shape[1]
                            ub_row_stride = (
                                (tile_cols + BYTE_PER_C0 - 1) // BYTE_PER_C0
                            ) * BYTE_PER_C0
                            ub_out_layout = tla.make_layout(
                                tla.make_shape(tile_rows, tile_cols),
                                tla.make_stride(ub_row_stride, 1),
                                layoutTag=tla.arch.RowMajor,
                            )
                            if tla.const_expr(use_out_pp):
                                use_ub_stage1 = ub_stage_idx != c0
                            else:
                                use_ub_stage1 = False
                            ub_result = tla.make_tensor(
                                ub_out_ptr1 if use_ub_stage1 else ub_out_ptr0,
                                ub_out_layout,
                            )

                            if tla.const_expr(_ub_pp_sf):
                                if use_ub_stage1:
                                    tla.wait_flag(ub_out1_available)
                                else:
                                    tla.wait_flag(ub_out0_available)
                            tla.set_flag(mte3_v_fence)
                            tla.wait_flag(mte3_v_fence)

                            epi_apply_vl_2d(
                                ACT,
                                ub_c_tile,
                                ub_result,
                                tile_rows,
                                tile_cols,
                                SIMD_LANES,
                                dtype_gm_c,
                            )

                            if use_ub_stage1:
                                tla.set_flag(ub_out1_data_ready)
                                tla.wait_flag(ub_out1_data_ready)
                            else:
                                tla.set_flag(ub_out0_data_ready)
                                tla.wait_flag(ub_out0_data_ready)
                            tla.copy(gm_d_tile, ub_result)
                            if tla.const_expr(_ub_pp_sf):
                                if use_ub_stage1:
                                    tla.set_flag(ub_out1_available)
                                else:
                                    tla.set_flag(ub_out0_available)

                        if tla.const_expr(use_out_pp):
                            ub_stage_idx = c1 - ub_stage_idx

                # CPP: always CrossCoreSetFlag@PIPE_MTE3 after epi (incl. last
                # tile / empty); AIC ProcessAicSync drains it before SyncAll.
                _cv_set_aiv(
                    aiv_finish0,
                    aiv_finish1,
                    aiv_finish2,
                    cv_count,
                    aiv_sub_idx,
                    tla.arch.MTE3,
                    _cv_id0_only,
                )

        # CPP SyncAllImpl: AIV arrive is published inside stage_barrier_hard
        # after PipeBarrier<ALL>. aiv_finish already on PIPE_MTE3; SyncAll does ALL.
