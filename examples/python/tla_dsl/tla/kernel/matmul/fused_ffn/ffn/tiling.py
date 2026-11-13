# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# FFN Phase-0 tiling / swizzle dataclasses (aligned with master e2e common.params).
from __future__ import annotations

from dataclasses import dataclass

import catlass.tla as tla

_CUBE = 16
_MAX_L1 = 256
_MAX_L0 = 256
# Minimum L0-K step (cube). Cap must cover CPP GetBaseK (e.g. #0164 baseK=512
# at 16×32). Capacity is still gated by L0A/L0B ping-pong in ``_l0_tk_for``.
_L0_TK_MIN = 32
_L0_TK_MAX = 512
# Soft cap for L1-K. Was 1024 (fix17 full-K); raised so small-MN stages=1
# can cut huge-H Down trips (e.g. 64×64 → tk≈2048) without inventing >L1 sizes.
_L1_TK_MAX = 4096
# Ascend950 L1 = 512 KiB. Leave room for tla.allocate(..., 512) rounding
# on up to 4 L1 slots (A0/A1/B0/B1); exact 128×128×1024 single-buffer is 512 KiB
# and LOAD2D then reports L0A/L0B OOB on large-H merges.
_L1_BYTES = 524288
_L1_ALIGN = 512
_L1_FIT_MARGIN = _L1_ALIGN * 4
# Ascend950 L0A/L0B = 64 KiB each; GEMM uses 2-stage ping-pong → 32 KiB/slot.
_L0AB_BYTES = 65536
_L0AB_STAGES = 2
# Operand A/B byte width for L1/L0 capacity + ping-pong slot stride.
# Product path matches CPP: x is f16/bf16 only (2B). Host rejects f32.
_L1_DTYPE_BYTES = 2
_L0_DTYPE_BYTES = 2
# CPP ``HALF_L0_SIZE = L0A_SIZE / 2 / sizeof(A_T)`` (WithoutQue DoAllLoc2Ub).
_HALF_L0_ELEMS = _L0AB_BYTES // (_L0AB_STAGES * _L0_DTYPE_BYTES)  # 16384 @2B
# Dual-L1 half-pool: L1/2/sizeof(A_T) — stages carve pong at this offset.
_HALF_L1_ELEMS = _L1_BYTES // (2 * _L1_DTYPE_BYTES)  # 131072 @2B


def _align_down(x: int, align: int, lo: int) -> int:
    v = (int(x) // align) * align
    return max(lo, v)


def _l0_tk_for(l0_tm: int, l0_tn: int, l1_tk: int, *, stages: int | None = None) -> int:
    """Largest cube-aligned L0-K that fits L0A/L0B and ``l1_tk``.

    ``stages=2`` (default): ping-pong → 32 KiB/slot. ``stages=1``: full 64 KiB
    per buffer (unlocks ``l0_tk=128`` at ``l0_tm=256`` for MTE1 quantum).

    When ``l1_tk`` / work-K is below ``_L0_TK_MIN`` (e.g. K=2), return that
    short length — CPP ``mmad.k=curK0`` allows it; forcing 32 would L1→L0 OOB.
    """
    l0_tm = max(1, int(l0_tm))
    l0_tn = max(1, int(l0_tn))
    stages = _L0AB_STAGES if stages is None else max(1, int(stages))
    slot = _L0AB_BYTES // stages
    max_by_a = slot // max(1, l0_tm * _L0_DTYPE_BYTES)
    max_by_b = slot // max(1, l0_tn * _L0_DTYPE_BYTES)
    # Do not apply ``_MAX_L0`` (MN legacy box) to K — that crushed CPP
    # baseK=512 → 256 (swiglu#0164).
    max_tk = min(_L0_TK_MAX, int(l1_tk), max_by_a, max_by_b)
    if max_tk < _L0_TK_MIN:
        return max(1, max_tk)
    return max(
        _L0_TK_MIN,
        _align_down(max_tk, _L0_TK_MIN, _L0_TK_MIN),
    )


def l0_pingpong_fits(l0_tm: int, l0_tn: int, l0_tk: int) -> bool:
    """Whether 2-stage L0A/L0B ping-pong fits in 64 KiB each."""
    a, b = l0_pingpong_bytes(l0_tm, l0_tn, l0_tk)
    return a <= _L0AB_BYTES and b <= _L0AB_BYTES


def _clamp_l0_tk_to_work(l0_tk: int, work_l1_tk: int) -> int:
    """Ensure L0-K never exceeds published work-K (L1 resid / problem K)."""
    w = int(work_l1_tk)
    if w <= 0:
        return max(1, int(l0_tk))
    return max(1, min(int(l0_tk), w))


def _l0_tk_cap(l0_tm: int, l0_tn: int, l1_tk: int, want_tk: int) -> int:
    """Cap ``want_tk`` to L0 capacity; keep single-buf K when ping-pong cannot fit."""
    # Align to C0=16 so CPP baseK=48/80 is not snapped to 32/64.
    want = min(int(want_tk), int(l1_tk), _L0_TK_MAX)
    if want < _CUBE:
        return max(1, want)
    want = max(_CUBE, _align_down(want, _CUBE, _CUBE))
    if l0_pingpong_fits(l0_tm, l0_tn, want):
        return want
    return min(want, _l0_tk_for(l0_tm, l0_tn, l1_tk, stages=1))


def l0_alloc_is_single(buf: TilingParams) -> bool:
    """Whether L0A/L0B must be single-buffer (tile does not fit CPP HALF_L0).

    CPP always ping-pongs at ``HALF_L0_SIZE`` when the L0 tile fits one half
    (``l0_pingpong_fits``). Exact fill (== half) is dual. Device alloc uses
    fixed ``_HALF_L0_ELEMS`` slot stride (see buffers.py), not tile elems.
    """
    return not l0_pingpong_fits(int(buf.l0_tm), int(buf.l0_tn), int(buf.l0_tk))


def _l1_tk_for(
    tm: int, tn: int, k: int, *, stages: int = 2, margin: int | None = None
) -> int:
    """Largest L0-aligned ``l1_tk`` that fits ``stages`` copies of A+B in L1.

    ``stages=2`` is L1 ping-pong; ``stages=1`` is single-buffer (full-K when A+B fit).
    Full-K (stages=1) uses margin 0 so 128×128×1024 still loads once; ping-pong
    keeps ``_L1_FIT_MARGIN`` so 4-slot exact-fill is not selected.
    """
    tm = max(1, int(tm))
    tn = max(1, int(tn))
    k = max(1, int(k))
    stages = max(1, int(stages))
    if margin is None:
        margin = 0 if stages == 1 else _L1_FIT_MARGIN
    # Capacity: ping-pong stages times A+B footprints must fit in L1 after margin.
    denom = stages * (tm + tn) * _L1_DTYPE_BYTES
    budget = max(denom, max(0, _L1_BYTES - int(margin)))
    max_by_l1 = budget // denom
    max_tk = min(_L1_TK_MAX, max_by_l1, k)
    return min(
        _L1_TK_MAX,
        max(_L0_TK_MIN, _align_down(max_tk, _L0_TK_MIN, _L0_TK_MIN)),
    )


def l1_single_bytes(tm: int, tn: int, tk: int) -> int:
    # Seq unified pool ``(total, tk=1, tn=0)``.
    if int(tk) == 1 and int(tn) == 0:
        return int(tm) * _L1_DTYPE_BYTES
    return (int(tm) + int(tn)) * int(tk) * _L1_DTYPE_BYTES


def _next_smaller_side(x: int) -> int:
    """Step down toward cube; prefer power-of-two ladder, else −16 (ASWT 160/80)."""
    x = int(x)
    for s in (256, 128, 64, 32, 16):
        if s < x:
            return s
    return max(_CUBE, x - _CUBE)


def fit_l1_bbox(tm: int, tn: int, tk: int) -> tuple[int, int, int]:
    """Shrink MN on the 16/32/64/128 ladder so single-buffer A+B has margin.

    128×128×1024 is exactly 512 KiB and traps MTE/L1. Intermediate sizes
    like 112 (16-aligned but not in the GEMM ladder) also MTE-fault.
    Keep ``tk`` (one L1 K-step) and reduce N then M.
    """
    tm = max(_CUBE, int(tm))
    tn = max(_CUBE, int(tn))
    tk = max(_L0_TK_MIN, int(tk))
    limit = _L1_BYTES - _L1_FIT_MARGIN
    while l1_single_bytes(tm, tn, tk) > limit:
        if tn >= tm and tn > _CUBE:
            tn = _next_smaller_side(tn)
        elif tm > _CUBE:
            tm = _next_smaller_side(tm)
        else:
            break
    return tm, tn, tk


def _l1_tk_prefer_full_k(tm: int, tn: int, k: int) -> int:
    """Prefer stages=1 full-K / a_full_load; else L1 ping-pong stages=2.

    Board note (Round-2): picking stages=1 tk=1024 for huge H disabled merge
    ``can_split`` and *regressed* vs ping-pong 480 (G1 181µs vs R1 140µs).
    Only take the large stages=1 chunk when it yields ≤2 K-trips (a_full_load
    lite). Otherwise keep stages=2 so Down L1 split PP can overlap MTE2∥MMAD.
    """
    k = max(1, int(k))
    tk1 = _l1_tk_for(tm, tn, k, stages=1)
    if tk1 >= k:
        return tk1
    tk2 = _l1_tk_for(tm, tn, k, stages=2)
    if tk1 <= 0:
        return tk2
    trips1 = (k + tk1 - 1) // max(1, tk1)
    if trips1 <= 2:
        return tk1
    # Huge-K: pack ping-pong tighter (no fit margin) → 128² tk 480→512.
    if k >= 2048:
        tk2_tight = _l1_tk_for(tm, tn, k, stages=2, margin=0)
        if tk2_tight > tk2:
            return tk2_tight
    return tk2 if tk2 > 0 else tk1


@dataclass(frozen=True)
class TilingParams:
    l1_tm: tla.Constexpr[int] = 256
    l1_tn: tla.Constexpr[int] = 256
    l1_tk: tla.Constexpr[int] = 128
    l0_tm: tla.Constexpr[int] = 256
    l0_tn: tla.Constexpr[int] = 256
    l0_tk: tla.Constexpr[int] = 32
    # Down full-load only: aligned K of the resident operand (0 = NONE / Basic).
    # A-full → m_aligned×resident_k; B-full → n_aligned×resident_k.
    resident_k: tla.Constexpr[int] = 0
    # CPP ``l1BufNum`` for the streaming side (2 or 4); 0 → treat as 2.
    l1_buf_num: tla.Constexpr[int] = 0
    # CPP ``MatMulV3FullLoad``: 0 NONE, 1 A, 2 B (mirrors host detect).
    full_load: tla.Constexpr[int] = 0
    # CPP device ``Min(K, kL1)`` working quantum for MTE2 / first-half-K.
    # Host dump ``l1_tk`` may exceed K (L1 slot); 0 → treat as ``l1_tk``.
    work_l1_tk: tla.Constexpr[int] = 0
    # CPP ``MatMulV3BasicTilingData`` host-dump fields (ResetBase / mixInfo).
    # Basic ASWT keeps split-cnt=1 → device uses edge tail, not mTailMain.
    m_base_tail_split_cnt: tla.Constexpr[int] = 1
    n_base_tail_split_cnt: tla.Constexpr[int] = 1
    m_tail_main: tla.Constexpr[int] = 0
    n_tail_main: tla.Constexpr[int] = 0
    # CPP ``ubDB``: 2 iff baseM×baseN×fp32 ≤ UB.
    ub_db: tla.Constexpr[int] = 1
    # CPP ``dbL0C``: 2 iff baseM×baseN×fp32×DB ≤ L0C (half-L0C pingpong).
    l0c_db: tla.Constexpr[int] = 1
    # CPP runInfo stepKa / depthA1 (kL1 = stepKa×baseK; depthA1 = stepKa×DB).
    step_ka: tla.Constexpr[int] = 1
    depth_a1: tla.Constexpr[int] = 2
    # CPP ``skSingleCoreK`` (StreamK K chunk per tile). 0 → not SK / unused.
    sk_single_core_k: tla.Constexpr[int] = 0


@dataclass(frozen=True)
class SwizzleParams:
    SWIZZLE_DIRECTION: tla.Constexpr[int] = 0
    SWIZZLE_OFFSET: tla.Constexpr[int] = 4
    # CPP ASWT ``mTailCnt`` / ``nTailCnt`` (last-wave MN tile split). 1 = off.
    M_TAIL_CNT: tla.Constexpr[int] = 1
    N_TAIL_CNT: tla.Constexpr[int] = 1
    # Aligned split extents used with tile-index coords (0 → derive ceil(L1/cnt)).
    M_TAIL_SPLIT: tla.Constexpr[int] = 0
    N_TAIL_SPLIT: tla.Constexpr[int] = 0


def l0_pingpong_bytes(l0_tm: int, l0_tn: int, l0_tk: int) -> tuple[int, int]:
    """(L0A bytes, L0B bytes) for 2-stage ping-pong fp16/bf16 tiles."""
    a = 2 * int(l0_tm) * int(l0_tk) * _L0_DTYPE_BYTES
    b = 2 * int(l0_tk) * int(l0_tn) * _L0_DTYPE_BYTES
    return a, b


def maybe_force_dual_full_l0_tk(tiling_up: TilingParams) -> TilingParams:
    """SwiGLU dual: one L0-K over full work-K when single-buf L0A/L0B fit.

    Dual Up aliases SoftFlag slot1→slot0; multi L0-K with those flags can
    double-apply the last K slice. Prefer ``l0_tk=work_l1_tk`` (or ``l1_tk``
    when work unset) when a *single* L0A/L0B of that K fits in 64 KiB;
    otherwise keep natural l0_tk.

    Never expand past ``work_l1_tk`` — published ``l1_tk`` may exceed problem
    K (swiglu#0044: l1=64, work=16 → old force l0=64 L1→L0 OOB / AIC trap).
    """
    work = int(tiling_up.work_l1_tk)
    want_l0_tk = min(int(tiling_up.l1_tk), work) if work > 0 else int(tiling_up.l1_tk)
    want_l0_tk = max(1, want_l0_tk)
    if want_l0_tk <= int(tiling_up.l0_tk):
        return tiling_up
    l0_tm_i, l0_tn_i = int(tiling_up.l0_tm), int(tiling_up.l0_tn)
    # l0_pingpong_bytes is 2×; single-buf size is half.
    l0a_pp, l0b_pp = l0_pingpong_bytes(l0_tm_i, l0_tn_i, want_l0_tk)
    if l0a_pp // 2 > _L0AB_BYTES or l0b_pp // 2 > _L0AB_BYTES:
        return tiling_up
    from dataclasses import replace

    return replace(tiling_up, l0_tm=l0_tm_i, l0_tn=l0_tn_i, l0_tk=want_l0_tk)


def mn_blocks(m: int, n: int, tiling: TilingParams) -> int:
    grid_m = (max(1, int(m)) + tiling.l1_tm - 1) // tiling.l1_tm
    grid_n = (max(1, int(n)) + tiling.l1_tn - 1) // tiling.l1_tn
    return max(1, grid_m * grid_n)


def shrink_tiling_for_tiles(
    m: int,
    n: int,
    tiling: TilingParams,
    want_tiles: int,
    *,
    min_side: int = 64,
    strict_min_side: bool = False,
) -> TilingParams:
    """Shrink L1/L0 MN (keep ``l1_tk``) so ``mn_blocks >= want_tiles`` when possible.

    Used after merge clamp: skinny-N Down often has only 8 tiles while Up has
    32 — SyncAll-safe ``block_num`` then caps at 8 and TD regresses. Prefer
    ``min(tm,tn) >= min_side`` (default 64) to avoid FixPipe/SiLU strip thrash.
    ``strict_min_side`` refuses candidates below ``min_side`` (pack pass).
    """
    m = max(1, int(m))
    n = max(1, int(n))
    want = max(1, int(want_tiles))
    cur = mn_blocks(m, n, tiling)
    if cur >= want:
        return tiling

    lo = max(_CUBE, int(min_side))
    # Candidates: prefixes of current stage bbox (must stay inside shared alloc).
    sides_m = sorted(
        {
            max(_CUBE, _align_down(min(m, s, tiling.l1_tm), _CUBE, _CUBE))
            for s in (128, 64, 32, 16)
            if s <= tiling.l1_tm
        },
        reverse=True,
    )
    sides_n = sorted(
        {
            max(_CUBE, _align_down(min(n, s, tiling.l1_tn), _CUBE, _CUBE))
            for s in (128, 64, 32, 16)
            if s <= tiling.l1_tn
        },
        reverse=True,
    )
    # Non-strict: once ``want`` is met, prefer healthy large tiles (not max grid).
    # Strict pack: among min_side≥lo, prefer more tiles then larger area.
    best: tuple | None = None
    for tm in sides_m:
        for tn in sides_n:
            ms = min(tm, tn)
            if strict_min_side and ms < lo:
                continue
            grid = ((m + tm - 1) // tm) * ((n + tn - 1) // tn)
            if grid < cur:
                continue
            hits = 1 if grid >= want else 0
            pen_ms = ms if ms >= lo else ms - 1000
            area = tm * tn
            aspect = -abs(tm - tn)
            if strict_min_side:
                key = (hits, grid, pen_ms, area, aspect, tm, tn)
            else:
                key = (hits, pen_ms, area, aspect, tm, tn)
            if best is None or key > best:
                best = key
    if best is None:
        return tiling
    tm, tn = best[-2], best[-1]
    if tm == tiling.l1_tm and tn == tiling.l1_tn:
        return tiling
    from dataclasses import replace

    l0_tm = min(tiling.l0_tm, tm)
    l0_tn = min(tiling.l0_tn, tn)
    return replace(
        tiling,
        l1_tm=tm,
        l1_tn=tn,
        l0_tm=l0_tm,
        l0_tn=l0_tn,
        l0_tk=min(tiling.l0_tk, _l0_tk_for(l0_tm, l0_tn, tiling.l1_tk)),
    )


def balance_syncall_tilings(
    m: int,
    h: int,
    n: int,
    tiling_up: TilingParams,
    tiling_down: TilingParams,
    *,
    target_bd: int = 16,
) -> tuple[TilingParams, TilingParams]:
    """Raise the weaker stage's tile count toward ``target_bd`` (SyncAll-safe).

    Only the bd-headroom pass (default 16). A further ``pack≥56`` shrink to
    64×64 was A/B'd (``sweep_pack56_v1``) and **regressed** large shapes
    (e.g. ``320…`` 0.97→1.07, ``384,768…`` 1.06→1.26) — do not re-enable
    without shape-gated evidence.
    """
    target = max(1, int(target_bd))
    up, down = tiling_up, tiling_down
    for _ in range(2):
        up_t = mn_blocks(m, h, up)
        dn_t = mn_blocks(m, n, down)
        if min(up_t, dn_t) >= target:
            break
        if dn_t < up_t:
            down = shrink_tiling_for_tiles(m, n, down, min(target, up_t))
        elif up_t < dn_t:
            up = shrink_tiling_for_tiles(m, h, up, min(target, dn_t))
        else:
            down = shrink_tiling_for_tiles(m, n, down, target)
            up = shrink_tiling_for_tiles(m, h, up, target)
            break
    return up, down


def swiglu_single_ok(tiling_up: TilingParams) -> bool:
    """CPP ``swigluSingle`` gate after Up tiling for ``n=2H``.

    Mirrors ``ffn_arch35_tiling.cpp``::
      swigluSingle = (up.n == n1 && nL1%32==0 && baseN%32==0 && nL1==baseN)
    DSL: evaluate on **raw** ASWT ``l1_tn``/``l0_tn`` **before**
    ``set_swiglu_up_tn`` (which forces ``l0_tn==l1_tn`` and would hide dual).
    AIC uses origin_shape extents for SwiGLU (no constexpr L1 bake).
    """
    n_l1 = int(tiling_up.l1_tn)
    n_l0 = int(tiling_up.l0_tn)
    return n_l1 % 32 == 0 and n_l0 % 32 == 0 and n_l1 == n_l0


def resolve_swiglu_mode(
    tiling_up: TilingParams,
    *,
    force: str | None = None,
) -> int:
    """Pick ``SWIGLU_SINGLE`` / ``SWIGLU_DUAL`` (CPP auto, optional force).

    ``force``: ``None``/``auto`` → tiling gate; ``single``/``dual`` → override
    (smoke / debug; CPP has no CLI but dual is the same kernel fallback).
    """
    from ffn.config import SWIGLU_DUAL, SWIGLU_SINGLE

    f = (force or "auto").strip().lower()
    if f == "dual":
        return SWIGLU_DUAL
    if f == "single":
        return SWIGLU_SINGLE
    return SWIGLU_SINGLE if swiglu_single_ok(tiling_up) else SWIGLU_DUAL


def set_swiglu_up_tn(tiling: TilingParams, tn: int) -> TilingParams:
    """Set Up ``l1_tn`` exactly (SwiGLU; also match ``l0_tn``).

    Phase-1.1: keep ASWT/CPP ``baseN`` (even only for ``[g|u]`` interleave).
    Do **not** snap to a DSL-only set — if the kernel fails on CPP tiles,
    fix the epi. Stage loops L0 over K only → ``l0_tn`` must equal ``l1_tn``.
    Call **after** ``resolve_swiglu_mode`` on the single path only.
    """
    # Allow ASWT baseN > legacy ``_MAX_L1`` (256); L0 must track l1_tn for SwiGLU.
    from dataclasses import replace

    from ffn.tiling_aswt import _MAX_L1_ASWT

    tn_v = min(_MAX_L1_ASWT, max(2, int(tn)))
    if tn_v % 2 != 0:
        tn_v = min(_MAX_L1_ASWT, tn_v + 1)
    l0_tn = tn_v
    l0_tm = min(int(tiling.l0_tm), int(tiling.l1_tm))
    return replace(
        tiling,
        l1_tn=tn_v,
        l0_tm=l0_tm,
        l0_tn=l0_tn,
    )


def select_swiglu_tilings(
    m: int,
    k: int,
    h: int,
    n: int,
    tiling_up: TilingParams,
    tiling_down: TilingParams,
    *,
    cube_max: int = 28,
) -> tuple[TilingParams, TilingParams]:
    """Pass-through after ASWT Up(n=2H); dual/single decided by caller.

    Do **not** call ``set_swiglu_up_tn`` here — that would force ``l0==l1``
    and mask the CPP ``swigluSingle`` gate (``nL1==baseN``).
    """
    del m, k, h, n, cube_max
    return tiling_up, tiling_down


def choose_ffn_host_tilings(
    m: int,
    k: int,
    h: int,
    n: int,
    *,
    cube_max: int = 28,
    up_n: int | None = None,
    has_bias: bool = False,
    dtype_bytes: int = 2,
) -> tuple[TilingParams, TilingParams, int]:
    """Up+Down host tilings via ASWT (CPP MatMulV3 rebalance).

    Returns ``(tiling_up, tiling_down, down_full_load)`` where
    ``down_full_load`` is 0/1/2 (CPP ``MatMulV3FullLoad``).
    ``dtype_bytes`` is A/B element width (f16/bf16=2, f32=4) for L0/L1.
    """
    from ffn.tiling_aswt import aswt_to_tiling_params, dsl_safe_aswt

    db = max(1, int(dtype_bytes))
    up_n_v = int(h if up_n is None else up_n)
    swiglu_n32 = up_n_v == 2 * int(h)
    up = dsl_safe_aswt(
        m,
        k,
        up_n_v,
        aic_num=cube_max,
        prefer_ub_db2=True,
        swiglu_single_n_align32=swiglu_n32,
        has_bias=has_bias,
        dtype_bytes=db,
    )
    down = dsl_safe_aswt(
        m,
        h,
        n,
        aic_num=cube_max,
        prefer_ub_db2=True,
        has_bias=has_bias,
        dtype_bytes=db,
    )
    tu, _, _ = aswt_to_tiling_params(
        up,
        k,
        allow_full_load=False,
        has_bias=has_bias,
        m=m,
        n=up_n_v,
        aic_num=cube_max,
        dtype_bytes=db,
    )
    td, down_res, fl_detail = aswt_to_tiling_params(
        down,
        h,
        allow_full_load=True,
        has_bias=has_bias,
        m=m,
        n=n,
        aic_num=cube_max,
        dtype_bytes=db,
    )
    del down_res
    full = int(fl_detail.full_load) if fl_detail is not None else 0
    return tu, td, full


def choose_swizzle_up(
    m: int,
    n: int,
    *,
    l1_tm: int = 0,
    l1_tn: int = 0,
    block_num: int = 28,
    forbid_n_split: bool = False,
) -> SwizzleParams:
    """CPP ASWT: M-window (``WINDOW_LEN=4``) + optional last-wave ``mTailCnt``.

    Previously ``DIRECTION=1 when m<=n`` windowed on N; that diverges from
    ``block_scheduler_aswt.h::UpdateMNTileIdx``. Pass ``l1_tm/tn`` + ``block_num``
    to fill ``M_TAIL_CNT``/``N_TAIL_CNT`` (else 1/1).
    """
    m_tail, n_tail = 1, 1
    m_split, n_split = 0, 0
    tm, tn = int(l1_tm), int(l1_tn)
    # CPP ASWT last-wave mTail. Kernel path is last-wave-only (not always-split).
    if tm > 0 and tn > 0:
        from ffn.tiling_aswt import calc_tail_basic_block, fit_tail_split_for_edge

        gm = (max(1, int(m)) + tm - 1) // tm
        gn = (max(1, int(n)) + tn - 1) // tn
        raw_m, raw_n = calc_tail_basic_block(
            gm, gn, int(block_num), forbid_n_split=forbid_n_split
        )
        m_tail, m_split = fit_tail_split_for_edge(int(m), tm, raw_m)
        if forbid_n_split:
            n_tail, n_split = 1, 0
        else:
            n_tail, n_split = fit_tail_split_for_edge(int(n), tn, raw_n)
    return SwizzleParams(
        SWIZZLE_DIRECTION=0,
        SWIZZLE_OFFSET=4,
        M_TAIL_CNT=int(m_tail),
        N_TAIL_CNT=int(n_tail),
        M_TAIL_SPLIT=int(m_split),
        N_TAIL_SPLIT=int(n_split),
    )


def choose_swizzle_down(
    m: int,
    n: int,
    *,
    l1_tm: int = 0,
    l1_tn: int = 0,
    block_num: int = 28,
    forbid_n_split: bool = False,
) -> SwizzleParams:
    """Same as Up: M-window + ``WINDOW_LEN=4`` + optional mTail."""
    return choose_swizzle_up(
        m,
        n,
        l1_tm=l1_tm,
        l1_tn=l1_tn,
        block_num=block_num,
        forbid_n_split=forbid_n_split,
    )


def prefer_down_streamk(
    m: int,
    h: int,
    n: int,
    tiling_down: TilingParams,
    cube_max: int,
    activation: str,
    *,
    has_bias: bool = False,
) -> bool:
    """Mirror CPP ``ffn_arch35_tiling`` Down StreamK — same shape → same template.

    CPP host (``ffn_arch35_tiling.cpp``):
      1. ``!isSwiglu`` (tiling_key: swiglu → MODE_BASIC only)
      2. basic ``baseM < align16(M)``
      3. StreamK tiling succeeds **and** ``sk.baseM >= fullM``
      4. ``CheckStreamKSKTilingDav3510`` (fp16):
         ``CeilAlign(K,256) >= max(8192, aic*256)/2`` → K≥4096 @ aic=28;
         ``ceil(M/256)*ceil(N/256) <= aic/2``

    SwiGLU StreamK is not CPP-aligned; keep Basic for SwiGLU.
    Device uses CPP ``BlockSchedulerStreamKBuiltIn`` (no linearized cross bypass).
    """
    if activation == "swiglu":
        return False
    full_m = ((int(m) + 15) // 16) * 16
    # (2) basic already full-M → CPP never tries StreamK.
    if int(tiling_down.l1_tm) >= full_m:
        return False
    # (4) K floor: CeilAlign(K,256) >= max(8192, aic*256) / dtype_bytes(fp16=2).
    aic = max(1, int(cube_max))
    k_align = ((int(h) + 255) // 256) * 256
    k_floor = max(8192, aic * 256) // 2
    if k_align < k_floor:
        return False
    # (4) 256-aligned MN tile count ≤ aic/2.
    m_cnt = (int(m) + 255) // 256
    n_cnt = (int(n) + 255) // 256
    if m_cnt * n_cnt > aic // 2:
        return False
    # (3) SK must expose baseM ≥ fullM (DoOpTiling port).
    sk = streamk_down_tiling(m, h, n, aic_num=aic, has_bias=has_bias)
    if int(sk.l1_tm) < full_m:
        return False
    return True


def streamk_down_tiling(
    m: int,
    h: int,
    n: int,
    *,
    has_bias: bool = False,
    aic_num: int = 28,
    dtype_bytes: int = 2,
) -> TilingParams:
    """Port of ``MatMulV3BasicStreamKTiling::DoOpTiling`` (DAV_3510 / FFN Down).

    FFN only enables the SKTiling gate (``totalMNCnt <= aic/2``), so this
    follows the first branch of DoOpTiling. L1-K from CalL1 (bias deduct);
    when ``totalMNCnt > aic && hasBias`` CPP forces ``stepKa=stepKb=3``.

    L0 MN == L1 MN (DSL SK demo contract). ``tn`` snapped to 64 for AIV reduce
    when CPP baseN is not already 64-aligned (device constraint).
    """
    from ffn.tiling_aswt import _cal_l1_tk_cpp, _ceil_align, _ceil_div, _floor_align

    m = max(1, int(m))
    k = max(1, int(h))
    n = max(1, int(n))
    aic = max(1, int(aic_num))
    dtype_bytes = max(1, int(dtype_bytes))

    # ResetBaseDav3510
    base_m = 256
    base_n = 256
    m_cnt = _ceil_div(m, base_m)
    n_cnt = _ceil_div(n, base_n)
    total = m_cnt * n_cnt
    if total <= aic // 2:
        if aic // 3 < m_cnt < aic // 2:
            m_cnt = aic // 2
        if aic // 3 < n_cnt < aic // 2:
            n_cnt = aic // 2
        total = m_cnt * n_cnt
        base_m = _ceil_align(_ceil_div(m, m_cnt), 16)
        base_n = _ceil_align(_ceil_div(n, n_cnt), 16)
        k_cnt = aic // max(1, total)
        single_core_k = _ceil_div(k, max(1, k_cnt))
    else:
        # DPSK remainder path (FFN SK gate usually excludes this).
        rem = total % aic
        k_cnt = aic // max(1, rem)
        sk_single = _ceil_div(k, max(1, k_cnt))
        k_cnt = _ceil_div(k, max(1, sk_single))
        single_core_k = sk_single

    # CPP DoOpTiling (non-ND B): CeilAlign(singleCoreK, 16) for fp16/bf16.
    # DSL GM→L1 is always Nd2Nz into cube L1; unaligned sk/l1_tk (e.g. 147
    # @ H=4096/aic=28) leaves L0 rem K not 16-aligned and corrupts SK reduce
    # (H=4480→160 passes; 4096→147 fails). Always align here for B16.
    if dtype_bytes <= 2:
        single_core_k = _ceil_align(int(single_core_k), 16)
    else:
        single_core_k = _ceil_align(int(single_core_k), 8)

    # FFN Linear: !isATrans || isBTrans → align 128B / dtype (CPP DoOpTiling).
    base_k_align = max(16, 128 // dtype_bytes)
    raw_max = (65536 // 2) // dtype_bytes // max(base_m, base_n)
    k_value_max = max(base_k_align, _floor_align(raw_max, base_k_align))
    base_k = min(int(single_core_k), int(k_value_max))
    if base_k < 16:
        base_k = 16

    l1_tk = _cal_l1_tk_cpp(
        base_m, base_n, base_k, k, has_bias=has_bias, dtype_bytes=dtype_bytes
    )
    # CPP: (totalMNCnt > aic) && hasBias → stepKa=stepKb=3.
    if total > aic and has_bias:
        l1_tk = min(k, 3 * base_k)

    # Device tile_view needs k_start = k_tile*sk_single aligned to l1_tk.
    # CPP keeps them independent (ptr+offset); TLA GM→L1 f16 copy cannot yet
    # express unaligned make_tensor offsets (2D vs 4D stub). When sk_single
    # does not divide into l1_tk, raise l1_tk to sk_single if dual L1 fits.
    sk_sc = int(single_core_k)
    if sk_sc > 0 and l1_tk > 0 and (sk_sc % int(l1_tk) != 0):
        dual_elems = 2 * (int(base_m) * sk_sc + sk_sc * int(base_n))
        if dual_elems * int(dtype_bytes) <= _L1_BYTES:
            l1_tk = sk_sc

    # CPP DoOpTiling: baseN stays 16-aligned (no host pad to 64).
    # Device AIV reduce pads UB width to AIV_REG_N=64 without changing baseN.
    l0_tm = min(_MAX_L0, base_m)
    l0_tn = min(_MAX_L0, base_n)
    # Prefer CPP baseK; capacity via ping-pong. Device Min(baseK, remainK).
    # Align to cube C0=16 (not _L0_TK_MIN=32) so baseK=48/80 survive.
    l0_tk = min(base_k, _L0_TK_MAX)
    if not l0_pingpong_fits(l0_tm, l0_tn, l0_tk):
        l0_tk = _l0_tk_cap(l0_tm, l0_tn, max(l1_tk, l0_tk), l0_tk)
    if l0_tk >= _CUBE:
        l0_tk = max(_CUBE, _align_down(l0_tk, _CUBE, _CUBE))
    else:
        l0_tk = max(1, l0_tk)
    work_l1_tk = min(int(l1_tk), int(k))
    from ffn.tiling_aswt import _DB, _FP32, _L0C, _UB

    ub_db = 2 if (int(base_m) * int(base_n) * _FP32) <= _UB else 1
    # CPP basic_aswt: dbL0C = baseM×baseN×fp32×DB ≤ l0CSize ? 2 : 1.
    l0c_db = 2 if (int(base_m) * int(base_n) * _FP32 * _DB) <= _L0C else 1
    step_ka = max(1, int(l1_tk) // max(1, int(base_k)))
    return TilingParams(
        l1_tm=base_m,
        l1_tn=base_n,
        l1_tk=l1_tk,
        l0_tm=l0_tm,
        l0_tn=l0_tn,
        l0_tk=l0_tk,
        work_l1_tk=work_l1_tk,
        m_base_tail_split_cnt=1,
        n_base_tail_split_cnt=1,
        m_tail_main=0,
        n_tail_main=0,
        ub_db=ub_db,
        l0c_db=l0c_db,
        step_ka=step_ka,
        depth_a1=step_ka * _DB,
        sk_single_core_k=sk_sc,
    )


def sk_workspace_elems(block_num: int, tiling_down: TilingParams | None = None) -> int:
    """FP32 workspace elems — CPP ``usedCoreNum × 256 × 256``.

    Bank index from CPP ``offsetWorkspace`` / ``BlockSchedulerStreamKBuiltIn``
    (1× usedCore; no dual-slot linearized fallback).
    """
    del tiling_down
    return int(block_num) * 256 * 256


def choose_block_num(
    m: int,
    k: int,
    h: int,
    n: int,
    tiling_up: TilingParams,
    tiling_down: TilingParams,
    cube_max: int,
    *,
    down_streamk: bool = False,
    down_k: int | None = None,
    swiglu_single_n_align32: bool = False,
    has_bias: bool = False,
) -> int:
    """CPP ``blockDim = max(up.usedCoreNum, down.usedCoreNum)``.

    Basic + StreamK: same formula. SK Down ``usedCoreNum = aicNum`` (ResetBase).
    Idle Up cores take empty ``tla.range`` then SyncAll (same as Basic idle).
    """
    from ffn.tiling_aswt import apply_down_full_load, dsl_safe_aswt

    del tiling_up, tiling_down
    up_n = int(h)
    dk = int(h if down_k is None else down_k)
    up = dsl_safe_aswt(
        m,
        k,
        up_n,
        aic_num=cube_max,
        prefer_ub_db2=True,
        swiglu_single_n_align32=swiglu_single_n_align32,
        has_bias=has_bias,
    )
    if down_streamk:
        down_used = max(1, int(cube_max))
    else:
        down = dsl_safe_aswt(
            m, dk, n, aic_num=cube_max, prefer_ub_db2=True, has_bias=has_bias
        )
        down = apply_down_full_load(m, dk, n, down, aic_num=cube_max, has_bias=has_bias)
        down_used = int(down.used_cores)
    return max(1, min(int(cube_max), max(int(up.used_cores), down_used)))
