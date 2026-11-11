# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Shared on-chip GEMM ping-pong buffers for sequential Up/Down in one MIX kernel.
from __future__ import annotations

import catlass.tla as tla

from ffn.tiling import (
    TilingParams,
    _HALF_L0_ELEMS,
    _L0AB_BYTES,
    _L0_DTYPE_BYTES,
    _L1_BYTES,
    _L1_DTYPE_BYTES,
    _L1_FIT_MARGIN,
    _l0_tk_cap,
    _l1_tk_prefer_full_k,
    fit_l1_bbox,
)


def is_seq_unified_l1(t: TilingParams) -> bool:
    """Seq-capacity buf: one L1 blob ``(total, tk=1, tn=0)`` (CPP-style reuse)."""
    return int(t.l1_tk) == 1 and int(t.l1_tn) == 0 and int(t.l1_tm) > 0


def seq_l1_pool_elems(t: TilingParams) -> int:
    """Total L1 elements allocated for a seq-capacity / merge buf."""
    if is_seq_unified_l1(t):
        return int(t.l1_tm)
    return int(t.l1_tm) * int(t.l1_tk) + int(t.l1_tk) * int(t.l1_tn)


def down_full_l1_elems(t: TilingParams) -> int:
    """CPP WithoutQue L1 footprint for Down A/B full-load (fp16 elems)."""
    rk = int(getattr(t, "resident_k", 0) or 0)
    mode = int(getattr(t, "full_load", 0) or 0)
    if rk <= 0 or mode not in (1, 2):
        return int(t.l1_tm) * int(t.l1_tk) + int(t.l1_tk) * int(t.l1_tn)
    buf_n = int(getattr(t, "l1_buf_num", 0) or 0)
    buf_n = 2 if buf_n < 2 else buf_n
    if mode == 1:
        # AL1 resident + BL1Ping×buf_n
        a = int(t.l1_tm) * rk
        b_one = int(t.l1_tk) * int(t.l1_tn)
        return a + b_one * buf_n
    # BL1 resident + AL1Ping×buf_n
    a_one = int(t.l1_tm) * int(t.l1_tk)
    b = int(t.l1_tn) * rk
    return a_one * buf_n + b


@tla.jit
def allocate_gemm_pingpong(
    dtype_a,
    dtype_b,
    _tiling: tla.Constexpr[TilingParams],
    L0C_ELEMS: tla.Constexpr[int] = 0,
    BIAS_L1_ELEMS: tla.Constexpr[int] = 0,
    L0C_DB: tla.Constexpr[int] = 1,
):
    """Allocate L1/L0/L0C. L1 is single-buffer when 2×(A+B) would exceed L1.

    ``L0C_ELEMS``: when >0, size one L0C slot by this count (seq-capacity buf
    encodes L0A/B as ``(Ea,1,Eb)`` which must not be used as L0C ``tm×tn``).

    ``L0C_DB``: CPP ``dbL0C`` — when 2, allocate two slots (half-L0C pingpong)
    so next-tile MMAD overlaps this-tile FixPipe (``FIX_M`` 0/1).

    ``BIAS_L1_ELEMS``: fp32 element count for one Cube-BT / L1-bias slot
    (``= max Up/Down l1_tn`` = CPP ``baseN_``). When >0, allocates **two**
    L1-bias slots and **two** biasbuf (BT) slots — CPP
    ``biasBt_[baseN_*(abL1LoopCnt&1)]`` / BiasPing-BiasPong. Unified-pool
    carve is ``elems*4`` fp16 units; callers pass
    ``BUF_L1_TM = l1_tm - elems*4``.

    Seq unified pool: ``(tm=total, tk=1, tn=0)`` — one L1 blob; stages carve
    ping A|B|pong A|B. Legacy ``(Ea,1,Eb)`` keeps separate A/B single pools.
    """
    # Unified: one contiguous pool (CPP sequential L1 reuse).
    # Cube BT L1 is carved from the pool tail (fp32 view) so a second L1
    # allocate cannot push past the 512 KiB chip after a near-full A/B pool.
    # Two slots (DB): each fp32 elem occupies 2 fp16 units → *4.
    _bias_as_a = BIAS_L1_ELEMS * 4 if tla.const_expr(BIAS_L1_ELEMS > 0) else 0
    if tla.const_expr(_tiling.l1_tk == 1 and _tiling.l1_tn == 0):
        l1_base = tla.allocate(_tiling.l1_tm, dtype_a, tla.AddressSpace.l1, 512)
        l1a0_ptr = l1_base
        l1a1_ptr = l1_base
        l1b0_ptr = l1_base
        l1b1_ptr = l1_base
        l1_single = tla.const_expr(True)
    else:
        l1_elems_a = _tiling.l1_tm * _tiling.l1_tk
        l1_elems_b = _tiling.l1_tk * _tiling.l1_tn
        # 2*(A+B)*2B > L1-margin → alias second buffer (full-K / near-full-K).
        # Exact 512 KiB ping-pong leaves no room for allocate(512) rounding.
        # Legacy seq ``(Ea, tk=1, Eb)``: keep single; stages carve ping-pong.
        l1_single = tla.const_expr(
            _tiling.l1_tk == 1
            or 2 * (l1_elems_a + l1_elems_b) * _L1_DTYPE_BYTES
            > (_L1_BYTES - _L1_FIT_MARGIN)
        )
        l1a0_ptr = tla.allocate(l1_elems_a, dtype_a, tla.AddressSpace.l1, 512)
        l1a1_ptr = l1a0_ptr
        if tla.const_expr(not l1_single):
            l1a1_ptr = tla.allocate(l1_elems_a, dtype_a, tla.AddressSpace.l1, 512)
        l1b0_ptr = tla.allocate(l1_elems_b, dtype_b, tla.AddressSpace.l1, 512)
        l1b1_ptr = l1b0_ptr
        if tla.const_expr(not l1_single):
            l1b1_ptr = tla.allocate(l1_elems_b, dtype_b, tla.AddressSpace.l1, 512)
    # CPP WithoutQue: ``HALF_L0_SIZE = L0A_SIZE/2/sizeof(A_T)``;
    # ``l0Offset = HALF_L0_SIZE * (l0PingPong_ & 1)`` for both L0A and L0B.
    # Dual iff tile fits one half (``l0_pingpong_fits`` / ``≤64KiB`` 2×).
    # Slot stride is always ``_HALF_L0_ELEMS`` (16384 fp16), never tile elems —
    # smaller tiles still sit at 0 / HALF_L0 like CPP (not packed adjacent).
    l0_a_elems = _tiling.l0_tm * _tiling.l0_tk
    l0_b_elems = _tiling.l0_tk * _tiling.l0_tn
    l0_single = tla.const_expr(
        2 * l0_a_elems * _L0_DTYPE_BYTES > _L0AB_BYTES
        or 2 * l0_b_elems * _L0_DTYPE_BYTES > _L0AB_BYTES
    )
    if tla.const_expr(not l0_single):
        l0a0_ptr = tla.allocate(_HALF_L0_ELEMS * 2, dtype_a, tla.AddressSpace.l0a, 512)
        l0a1_ptr = l0a0_ptr + _HALF_L0_ELEMS
        l0b0_ptr = tla.allocate(_HALF_L0_ELEMS * 2, dtype_b, tla.AddressSpace.l0b, 512)
        l0b1_ptr = l0b0_ptr + _HALF_L0_ELEMS
    else:
        l0a0_ptr = tla.allocate(l0_a_elems, dtype_a, tla.AddressSpace.l0a, 512)
        l0a1_ptr = l0a0_ptr
        l0b0_ptr = tla.allocate(l0_b_elems, dtype_b, tla.AddressSpace.l0b, 512)
        l0b1_ptr = l0b0_ptr
    l0c_slot = (
        L0C_ELEMS if tla.const_expr(L0C_ELEMS > 0) else _tiling.l0_tm * _tiling.l0_tn
    )
    # CPP HALF_L0C pingpong: two slots when dbL0C=2 (tile×2 ≤ L0C).
    l0c_n = l0c_slot * L0C_DB if tla.const_expr(L0C_DB > 1) else l0c_slot
    l0c_ptr = tla.allocate(l0c_n, tla.Float32, tla.AddressSpace.l0c, 512)
    # Cube BT: GM→L1 (fp32 VectorLayout) → biasbuf ping-pong (2×baseN).
    # Unused when elems==0. BT HW is 4096B (=1024 fp32); ASWT caps baseN
    # via btSize/DB/fp32 so 2×slot fits.
    if tla.const_expr(BIAS_L1_ELEMS > 0):
        _bias_db = BIAS_L1_ELEMS * 2
        if tla.const_expr(_tiling.l1_tk == 1 and _tiling.l1_tn == 0):
            # Tail of unified pool: fp16 elems → fp32 view (×2 per slot, ×2 DB).
            l1_bias_ptr = tla.recast_ptr(
                l1a0_ptr + (_tiling.l1_tm - _bias_as_a), dtype=tla.Float32
            )
        else:
            l1_bias_ptr = tla.allocate(_bias_db, tla.Float32, tla.AddressSpace.l1, 512)
        l0_bias_ptr = tla.allocate(_bias_db, tla.Float32, tla.AddressSpace.biasbuf, 512)
    else:
        l1_bias_ptr = l1a0_ptr
        l0_bias_ptr = l0c_ptr
    return (
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
    )


def merge_tiling(a: TilingParams, b: TilingParams) -> TilingParams:
    """One buffer set covering both stages (sequential Up then Down).

    Round-2: size L1 by ``max(stage A/B elems)`` so a small-MN / large-tk Down
    is not crushed by ``max(tm)×max(tk)`` with Up (that forced Down tk→~480–1024
    even when 32×32×4096 elems fit the same 512 KiB). Fall back to dim-max +
    shrink-K when capacity merge cannot encode both footprints.

    Undersized MN vs a stage tiling is still invalid for the dim-max path
    (SwiGLU Up ``l1_tn`` must not run in a smaller merged box).
    """
    from ffn.tiling import _L0_TK_MIN, _L1_BYTES, _L1_FIT_MARGIN, l1_single_bytes
    from ffn.tiling import _L1_DTYPE_BYTES, _align_down

    elems_a = max(int(a.l1_tm) * int(a.l1_tk), int(b.l1_tm) * int(b.l1_tk))
    elems_b = max(int(a.l1_tk) * int(a.l1_tn), int(b.l1_tk) * int(b.l1_tn))
    if (elems_a + elems_b) * _L1_DTYPE_BYTES <= _L1_BYTES:
        l1_tm = max(int(a.l1_tm), int(b.l1_tm))
        l1_tn = max(int(a.l1_tn), int(b.l1_tn))
        need_tk = max(
            (elems_a + l1_tm - 1) // max(1, l1_tm),
            (elems_b + l1_tn - 1) // max(1, l1_tn),
        )
        l1_tk = max(_L0_TK_MIN, _align_down(int(need_tk), _L0_TK_MIN, _L0_TK_MIN))
        # Align-down may undershoot element capacity; bump by L0 step.
        while (l1_tm * l1_tk < elems_a or l1_tk * l1_tn < elems_b) and (
            l1_tm * l1_tk + l1_tk * l1_tn
        ) * _L1_DTYPE_BYTES <= _L1_BYTES:
            l1_tk += _L0_TK_MIN
        if (
            l1_tm * l1_tk >= elems_a
            and l1_tk * l1_tn >= elems_b
            and (l1_tm * l1_tk + l1_tk * l1_tn) * _L1_DTYPE_BYTES <= _L1_BYTES
        ):
            l0_tm = min(max(int(a.l0_tm), int(b.l0_tm)), l1_tm)
            l0_tn = min(max(int(a.l0_tn), int(b.l0_tn)), l1_tn)
            l0_tk = _l0_tk_cap(l0_tm, l0_tn, l1_tk, max(int(a.l0_tk), int(b.l0_tk)))
            return TilingParams(
                l1_tm=l1_tm,
                l1_tn=l1_tn,
                l1_tk=l1_tk,
                l0_tm=l0_tm,
                l0_tn=l0_tn,
                l0_tk=l0_tk,
            )

    l1_tm = max(int(a.l1_tm), int(b.l1_tm))
    l1_tn = max(int(a.l1_tn), int(b.l1_tn))
    want_tk = max(int(a.l1_tk), int(b.l1_tk))
    # Exact single-buffer full-K may fill L1 (128×128×1024=512KiB); margin is
    # for ping-pong only. Prefer keeping MN + want_tk when single-buffer fits.
    if l1_single_bytes(l1_tm, l1_tn, want_tk) <= _L1_BYTES:
        l1_tk = want_tk
    else:
        limit = _L1_BYTES - _L1_FIT_MARGIN
        denom = 2 * (l1_tm + l1_tn)
        max_tk = _align_down(limit // max(1, denom), _L0_TK_MIN, _L0_TK_MIN)
        l1_tk = min(want_tk, max_tk)
        if l1_single_bytes(l1_tm, l1_tn, l1_tk) > limit:
            # Last resort: old MN shrink.
            l1_tm, l1_tn, l1_tk = fit_l1_bbox(l1_tm, l1_tn, want_tk)
            if l1_single_bytes(l1_tm, l1_tn, want_tk) <= _L1_BYTES:
                l1_tk = want_tk
            else:
                l1_tk = _l1_tk_prefer_full_k(l1_tm, l1_tn, want_tk)
                l1_tm, l1_tn, l1_tk = fit_l1_bbox(l1_tm, l1_tn, l1_tk)
    l0_tm = min(max(int(a.l0_tm), int(b.l0_tm)), l1_tm)
    l0_tn = min(max(int(a.l0_tn), int(b.l0_tn)), l1_tn)
    l0_tk = _l0_tk_cap(l0_tm, l0_tn, l1_tk, max(int(a.l0_tk), int(b.l0_tk)))
    return TilingParams(
        l1_tm=l1_tm,
        l1_tn=l1_tn,
        l1_tk=l1_tk,
        l0_tm=l0_tm,
        l0_tn=l0_tn,
        l0_tk=l0_tk,
    )


def _clamp_l0_tk(t: TilingParams, l0_tk: int) -> TilingParams:
    if int(t.l0_tk) <= int(l0_tk):
        return t
    from dataclasses import replace

    return replace(t, l0_tk=l0_tk)


def _clamp_l1_bbox(t: TilingParams, buf: TilingParams) -> TilingParams:
    """Clamp stage into allocated L1 capacity (elem counts), not max-dim only.

    Capacity merge may encode ``max(A/B elems)`` as a wide shallow box (e.g.
    128×128×1024) while Down keeps 32×32×4096 — same A elems. Dim-min clamp
    would wrongly crush Down tk to 1024.
    """
    from ffn.tiling import _L0_TK_MIN, _align_down

    buf_a = int(buf.l1_tm) * int(buf.l1_tk)
    buf_b = int(buf.l1_tk) * int(buf.l1_tn)
    stage_a = int(t.l1_tm) * int(t.l1_tk)
    stage_b = int(t.l1_tk) * int(t.l1_tn)
    l1_tm, l1_tn, l1_tk = int(t.l1_tm), int(t.l1_tn), int(t.l1_tk)
    if stage_a > buf_a or stage_b > buf_b:
        max_tk = min(
            l1_tk,
            buf_a // max(1, l1_tm),
            buf_b // max(1, l1_tn),
        )
        l1_tk = max(_L0_TK_MIN, _align_down(int(max_tk), _L0_TK_MIN, _L0_TK_MIN))
        # Last resort: also shrink MN into the buf box.
        if l1_tm * l1_tk > buf_a or l1_tk * l1_tn > buf_b:
            l1_tm = min(l1_tm, int(buf.l1_tm))
            l1_tn = min(l1_tn, int(buf.l1_tn))
            l1_tk = min(l1_tk, int(buf.l1_tk))
    l0_tm = min(int(t.l0_tm), int(buf.l0_tm), l1_tm)
    l0_tn = min(int(t.l0_tn), int(buf.l0_tn), l1_tn)
    # Seq-capacity buf encodes L0 as ``(Ea,1,Eb)`` — do not min with buf.l0_tk=1.
    want_l0 = int(t.l0_tk)
    if not is_seq_unified_l1(buf):
        want_l0 = min(want_l0, int(buf.l0_tk))
    l0_tk = _l0_tk_cap(l0_tm, l0_tn, l1_tk, want_l0)
    if (
        l1_tm == t.l1_tm
        and l1_tn == t.l1_tn
        and l1_tk == t.l1_tk
        and l0_tm == t.l0_tm
        and l0_tn == t.l0_tn
        and l0_tk == t.l0_tk
    ):
        return t
    from dataclasses import replace

    # Keep CPP dump/work split: clamp work_l1_tk into new published l1_tk.
    work = int(t.work_l1_tk)
    if work > 0:
        work = min(work, l1_tk)
    return replace(
        t,
        l1_tm=l1_tm,
        l1_tn=l1_tn,
        l1_tk=l1_tk,
        l0_tm=l0_tm,
        l0_tn=l0_tn,
        l0_tk=l0_tk,
        work_l1_tk=work,
    )


def _shrink_stage_tk_for_l1_split(t: TilingParams, buf: TilingParams) -> TilingParams:
    """Shrink stage ``l1_tk`` so 2×full A/B fit in ``buf`` (L1 dual PP).

    Only used when Down is already multi-K (full-K must stay single-buffer).
    CPP DoAllLoc2Ub dual slots hold full ``mL1*kL1`` / ``kL1*nL1`` — never
    permanent half-K. If 2×full does not fit, shrink ``tk`` until it does
    (or leave single-buffer with original ``tk`` when shrink is impossible).
    """
    from ffn.tiling import _L0_TK_MIN, _align_down

    tm, tn, tk = int(t.l1_tm), int(t.l1_tn), int(t.l1_tk)
    if is_seq_unified_l1(buf):
        pool = seq_l1_pool_elems(buf)
        if 2 * (tm * tk + tk * tn) <= pool:
            return t
        max_tk = min(tk, pool // (2 * max(1, tm + tn)))
    else:
        buf_a = int(buf.l1_tm) * int(buf.l1_tk)
        buf_b = int(buf.l1_tk) * int(buf.l1_tn)
        if 2 * tm * tk <= buf_a and 2 * tk * tn <= buf_b:
            return t
        max_tk = min(
            tk,
            buf_a // (2 * max(1, tm)),
            buf_b // (2 * max(1, tn)),
        )
    max_tk = max(_L0_TK_MIN, _align_down(int(max_tk), _L0_TK_MIN, _L0_TK_MIN))
    if max_tk >= tk:
        return t
    from dataclasses import replace

    l0_tm = min(int(t.l0_tm), tm)
    l0_tn = min(int(t.l0_tn), tn)
    work = int(t.work_l1_tk)
    if work > 0:
        work = min(work, max_tk)
    return replace(
        t,
        l1_tm=tm,
        l1_tn=tn,
        l1_tk=max_tk,
        l0_tm=l0_tm,
        l0_tn=l0_tn,
        l0_tk=_l0_tk_cap(l0_tm, l0_tn, max_tk, int(t.l0_tk)),
        work_l1_tk=work,
    )


def _merge_tiling_seq_capacity(a: TilingParams, b: TilingParams) -> TilingParams | None:
    """Sequential Up→SyncAll→Down L1: ``max(stage_footprint)`` unified pool.

    CPP reuses one L1 between stages; dual ping-pong is ``2*(A+B)`` laid out
    as pingA|pingB|pongA|pongB — not fixed cross-max pools
    ``max(A_up,A_dn)+max(B_up,B_dn)`` (that false-OOM's asymmetric A/B).

    Encoding ``(tm=total_elems, tk=1, tn=0)``: ``allocate_gemm_pingpong``
    allocates one blob; stages carve. Prefer both-dual → up-dual → dn-dual
    → both-single. Returns ``None`` if either stage alone exceeds L1.
    """
    from ffn.tiling import _L1_DTYPE_BYTES, _L1_FIT_MARGIN, l1_single_bytes

    if l1_single_bytes(int(a.l1_tm), int(a.l1_tn), int(a.l1_tk)) > _L1_BYTES:
        return None
    dn_full = int(getattr(b, "full_load", 0) or 0) in (1, 2)
    if (
        not dn_full
        and l1_single_bytes(int(b.l1_tm), int(b.l1_tn), int(b.l1_tk)) > _L1_BYTES
    ):
        return None
    if dn_full and down_full_l1_elems(b) * _L1_DTYPE_BYTES > _L1_BYTES:
        return None
    up_a = int(a.l1_tm) * int(a.l1_tk)
    up_b = int(a.l1_tk) * int(a.l1_tn)
    if dn_full:
        # Full-load: streaming side may be dual; resident is single. Use the
        # CPP WithoutQue footprint (not naive (tm+tn)*tk).
        dn_single = down_full_l1_elems(b)
        dn_dual = dn_single  # already includes streaming×l1_buf_num
    else:
        dn_a = int(b.l1_tm) * int(b.l1_tk)
        dn_b = int(b.l1_tk) * int(b.l1_tn)
        dn_single = dn_a + dn_b
        dn_dual = 2 * dn_single
    up_single = up_a + up_b
    up_dual = 2 * up_single
    # Unified seq pool is **one** L1 blob (stages carve ping/pong). FIT_MARGIN
    # exists for 4-slot ``allocate`` rounding — applying it here falsely rejects
    # exact full-chip dual (e.g. S49 up 64×64×1024 → 2×131072 = 262144 elems)
    # and would force single-buffer or shrink-tk (never permanent half-K).
    _l1_limit = _L1_BYTES
    # AscendC WithoutQue 2-buffer: expand pool so pong lives at HALF_L1.
    # Default ``half`` must match ``up_l0c2ub_silu`` / ``down_*`` carve default.
    # Old ``abab`` default + tight 2*(A+B) left use_half_l1=False on s128
    # (pool 81920 < HALF_L1+stage) — stable aic_mte2/scalar vs CPP.
    # CPP places ping/pong at HALF_L1*id → need full L1 (2*HALF_L1 elems).
    _half_l1 = _L1_BYTES // (2 * _L1_DTYPE_BYTES)
    _carve_half = "half" in (
        "half",
        "half_l1",
        "cpp",
    )
    # Sequential reuse: pool = max(up_need, dn_need), not sum / cross-max.
    if max(up_dual, dn_dual) * _L1_DTYPE_BYTES <= _l1_limit:
        pool = max(up_dual, dn_dual)
        if _carve_half:
            # Prefer full L1 so every stage's (HALF_L1 + A + B) fits; fallback
            # to HALF_L1 + max(stage single) when margin clips full chip.
            full_l1 = 2 * _half_l1
            if full_l1 * _L1_DTYPE_BYTES <= _L1_BYTES:
                pool = max(pool, full_l1)
            else:
                half_need = _half_l1 + max(up_single, dn_single)
                if half_need * _L1_DTYPE_BYTES <= _L1_BYTES:
                    pool = max(pool, half_need)
    elif max(up_dual, dn_single) * _L1_DTYPE_BYTES <= _l1_limit:
        pool = max(up_dual, dn_single)
    elif max(up_single, dn_dual) * _L1_DTYPE_BYTES <= _l1_limit:
        pool = max(up_single, dn_dual)
    elif max(up_single, dn_single) * _L1_DTYPE_BYTES <= _l1_limit:
        pool = max(up_single, dn_single)
    else:
        return None
    # L0A/L0B are also sequential — max elems, not max(tn)×max(tk).
    l0_elems_a = max(int(a.l0_tm) * int(a.l0_tk), int(b.l0_tm) * int(b.l0_tk))
    l0_elems_b = max(int(a.l0_tk) * int(a.l0_tn), int(b.l0_tk) * int(b.l0_tn))
    return TilingParams(
        l1_tm=pool,
        l1_tn=0,
        l1_tk=1,
        l0_tm=l0_elems_a,
        l0_tn=l0_elems_b,
        l0_tk=1,
    )


def share_tilings(
    up: TilingParams,
    down: TilingParams,
    *,
    m: int | None = None,
    h: int | None = None,
    n: int | None = None,
    target_bd: int = 16,
    down_k: int | None = None,
    balance: bool | None = None,
) -> tuple[TilingParams, TilingParams, TilingParams]:
    """Merge alloc tiling; clamp stages into L1/L0 **capacity** (not same MN).

    Phase-1.1 (W2): default **does not** call ``balance_syncall_tilings`` when
    Host is on ASWT — that pass crushed CPP baseMN to pack SyncAll. Physical
    L1 is still one shared buffer sized by ``merge_tiling`` = max footprints.

    ASWT sequential path: when each stage fits L1 alone, keep stage MN/K and
    allocate a unified pool ``max(stage dual/single feet)`` (see
    ``_merge_tiling_seq_capacity``).

    Default ``balance=False`` (ASWT / CPP parity: never crush logical MN for
    SyncAll packing). Pass ``balance=True`` only for legacy MN shrink toward
    ``target_bd``.

    ``down_k``: when Down is multi-K, may shrink ``l1_tk`` for L1 split PP.
    """
    if balance is None:
        balance = False

    # W2 sequential reuse (default ASWT): do not reshape both stages into one
    # (max_tm, max_tn, tk) box — that couples Up tn into Down tk.
    if not balance:
        seq = _merge_tiling_seq_capacity(up, down)
        if seq is not None:
            up_c, down_c, buf = up, down, seq
            if down_k is not None and int(down_c.l1_tk) < int(down_k):
                down_c = _shrink_stage_tk_for_l1_split(down_c, buf)
            return up_c, down_c, buf

    buf = merge_tiling(up, down)
    up_c = _clamp_l1_bbox(up, buf)
    down_c = _clamp_l1_bbox(down, buf)
    if balance and m is not None and h is not None and n is not None:
        from ffn.tiling import balance_syncall_tilings

        up_c, down_c = balance_syncall_tilings(
            int(m), int(h), int(n), up_c, down_c, target_bd=int(target_bd)
        )
    if down_k is not None and int(down_c.l1_tk) < int(down_k):
        down_c = _shrink_stage_tk_for_l1_split(down_c, buf)
    return up_c, down_c, buf


def apply_streamk_tilings(
    up: TilingParams,
    *,
    m: int,
    h: int,
    n: int,
    down_k: int | None = None,
    has_bias: bool = False,
    aic_num: int = 28,
) -> tuple[TilingParams, TilingParams, TilingParams]:
    """Install CPP DoOpTiling Down SK tile; keep Up ASWT as-is (no DSL boost).

    ``has_bias``: CalL1 bias deduct + stepK=3 when totalMNCnt > aic.
    ``h`` / ``down_k``: Down K (hidden); ``h`` is also Up logical N for callers.
    """
    from ffn.tiling import streamk_down_tiling

    down = streamk_down_tiling(
        int(m),
        int(down_k if down_k is not None else h),
        int(n),
        has_bias=has_bias,
        aic_num=int(aic_num),
    )
    # Prefer seq-capacity merge (no MN crush). Fallback merge+clamp.
    seq = _merge_tiling_seq_capacity(up, down)
    if seq is not None:
        return up, down, seq
    buf = merge_tiling(up, down)
    up_c = _clamp_l0_tk(_clamp_l1_bbox(up, buf), buf.l0_tk)
    down_c = _clamp_l0_tk(_clamp_l1_bbox(down, buf), buf.l0_tk)
    return up_c, down_c, buf
