# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Python port of Linear FFN + MatMulV3 Basic ASWT host tiling (Ascend950 / DAV_3510).
from __future__ import annotations

import math
from dataclasses import dataclass

from ffn.tiling import (
    TilingParams,
    _L0_TK_MAX,
    _L0_TK_MIN,
    _align_down,
    _l0_tk_for,
)

# Hardware / ASWT constants (matmul_v3_common_advanced.h + tiling_helper defaults).
_DB = 2
_BASIC_16 = 16
_BASIC_32 = 32
_BASIC_64 = 64
_BASIC_128 = 128
_BASIC_256 = 256
_K_128_BYTE = 128
_K_256_BYTE = 256
_K_512_BYTE = 512
_FP32 = 4
_CUBE_BOUND_RATIO = 0.85
_EPSILON = 1e-9
_BALANCE_EDGE = 0.9
_MIN_TAIL_BLOCK = 4096
# GetCoreFreq returns cube_freq/1000 (default 1650 → 1.65).
_CORE_FREQ = 1650.0 / 1000.0
_AI_CORE_CNT = 32.0  # SoCInfo ai_core_cnt fallback (BW formula only)
_DDR_RATE = 31.0
_L2_RATE = 100.0
_KB = 1024.0
# Ascend950PR / DAV_3510 on-chip (InitCompileInfo / GetCoreMemSize).
# Measured on this box: ubSize=253952, l1=524288, l0C=262144, l2=134217728.
# Wrong _UB=196608 falsely triggered preferUbDB2 N-halve (G540 272×192→96).
_L0A = 65536
_L0C = 262144
_L1 = 524288
_UB = 253952
_L2 = 128 * 1024 * 1024
# Ascend950 Bias Table (CoreMemType::BT). GetMaxBaseWithLimit hasBias N cap:
#   btSize / DB / fp32  → 4096 / 2 / 4 = 512.
_BT_SIZE = 4096
_BIAS_TABLE_NUM = 256  # L1 reserve elems (fp32) when hasBias (CPP ASWT)


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def calc_tail_basic_block(
    m_tiles: int,
    n_tiles: int,
    aic_num: int,
    *,
    forbid_n_split: bool = False,
) -> tuple[int, int]:
    """Port of ``MatMulV3AswTiling::CalcTailBasicBlock`` → ``(mTailCnt, nTailCnt)``.

    When ``T > C`` and ``T % C != 0``, grow M (then N) so that
    ``mCnt * nCnt * (T % C) <= C``. SwiGLU single forbids N-split.
    """
    m_tiles = max(1, int(m_tiles))
    n_tiles = max(1, int(n_tiles))
    aic_num = max(1, int(aic_num))
    mn = m_tiles * n_tiles
    tail = 0 if mn <= aic_num else mn % aic_num
    m_cnt, n_cnt = 1, 1
    if tail == 0:
        return m_cnt, n_cnt
    while (m_cnt + 1) * n_cnt * tail <= aic_num:
        m_cnt += 1
        if (not forbid_n_split) and m_cnt * (n_cnt + 1) * tail <= aic_num:
            n_cnt += 1
    return m_cnt, n_cnt


def fit_tail_split_for_edge(
    extent: int,
    tile: int,
    tail_cnt: int,
    *,
    align: int = 16,
) -> tuple[int, int]:
    """Return ``(tail_cnt, split)`` for last-wave tile-index splits.

    ``split = align_up(ceil(tile / raw_tail), align)``. Keep ``raw_tail`` so
    full L1 tiles stay covered; edge tiles may produce empty trailing pieces
    (``m_si * split >= edge``) — kernel must sync-only those (CPP returns
    ``{0,0}`` shape).
    """
    tile = max(1, int(tile))
    raw = max(1, int(tail_cnt))
    align = max(1, int(align))
    if raw <= 1:
        return 1, 0
    split = (tile + raw - 1) // raw
    split = ((split + align - 1) // align) * align
    if split <= 0:
        return 1, 0
    return int(raw), int(split)


def _ceil_align(x: int, align: int) -> int:
    align = max(1, int(align))
    return _ceil_div(x, align) * align


def _floor_align(x: int, align: int) -> int:
    align = max(1, int(align))
    return (int(x) // align) * align


def _hbm_bw() -> float:
    return _CORE_FREQ * _AI_CORE_CNT * _DDR_RATE / _KB


def _l2_bw() -> float:
    return _CORE_FREQ * _AI_CORE_CNT * _L2_RATE / _KB


def _balance_rate(m: int, n: int, used: int, base_m: int, base_n: int) -> float:
    """GetBalanceRateWithTail (batch=1, no batchInfo)."""
    used = max(1, int(used))
    total = _ceil_div(m, base_m) * _ceil_div(n, base_n)
    main = _ceil_div(total, used) - 1
    rem = total - used * main
    rem = max(1, rem)
    tail_split = used // rem
    bm, bn = base_m, base_n
    if m <= _BASIC_16:
        bm = m
    if n <= _BASIC_16:
        bn = n
    if main == 0 or (bm * bn) // max(1, tail_split) < _MIN_TAIL_BLOCK:
        return (m * n / used) / ((main + 1) * bm * bn)
    sqrt_t = int(math.sqrt(tail_split))
    offset = (tail_split - sqrt_t * sqrt_t) // max(1, sqrt_t) + 1
    tail_round = 1.0 / (sqrt_t * (sqrt_t + offset - 1))
    return (m * n / used) / ((main + tail_round) * bm * bn)


def _max_base_with_limit(
    shape: int,
    k: int,
    dtype_bytes: int,
    base_mn_limit: int,
    align: int,
    *,
    is_n: bool,
    memory_bound: bool,
    is_a_trans: bool,
    is_b_trans: bool,
    has_bias: bool = False,
) -> int:
    """GetMaxBaseWithLimit for ND fp16."""
    k_align = _ceil_align(k, _BASIC_16)
    k_limit = _BASIC_16 if memory_bound else (_K_128_BYTE // dtype_bytes)
    min_kl0 = min(k_limit, k_align) * dtype_bytes
    max_mn_buf = base_mn_limit // _FP32 // _BASIC_16
    max_block = min(_L0A // _DB // max(1, min_kl0), max_mn_buf)
    if is_n and has_bias:
        # CPP GetMaxBaseWithLimit: maxBaseBlock = min(..., btSize/DB/fp32).
        max_block = min(max_block, _BT_SIZE // _DB // _FP32)
    # !isATrans || isBTrans → 256B (mem) / 512B (else); else 16
    if (not is_a_trans) or is_b_trans:
        k_align_unit = (_K_256_BYTE if memory_bound else _K_512_BYTE) // dtype_bytes
    else:
        k_align_unit = _BASIC_16
    max_k_inner = _L1 // (_DB * _DB * dtype_bytes * min(k_align_unit, k_align))
    max_block = min(max_block, max_k_inner)
    max_block = min(_ceil_align(shape, align), _floor_align(max_block, align))
    if shape < align:
        max_block = min(max_block, _ceil_align(shape, _BASIC_16))
    return max_block


def _get_base_k(
    base_m: int,
    base_n: int,
    k: int,
    dtype_bytes: int,
    *,
    is_a_trans: bool,
    is_b_trans: bool,
) -> int:
    """GetBaseK."""
    k_align = _ceil_align(k, _BASIC_16)
    max_base_k = _L0A // _DB // dtype_bytes // max(base_m, base_n)
    if k_align <= max_base_k:
        return int(k_align)
    if is_a_trans and not is_b_trans:
        return int(_floor_align(max_base_k, _BASIC_16))
    if max_base_k * dtype_bytes >= _K_256_BYTE:
        return int(_floor_align(max_base_k, _K_256_BYTE // dtype_bytes))
    for cand in (128, 64, 32, 16):
        if max_base_k >= cand:
            return int(cand)
    return _BASIC_16


@dataclass(frozen=True)
class AswtResult:
    base_m: int
    base_n: int
    base_k: int
    used_cores: int
    allow_full_load: bool
    # 0=NONE, 1=A_FULL_LOAD, 2=B_FULL_LOAD (MatMulV3FullLoad).
    full_load: int = 0
    cube_bound_param: float = 0.0
    cube_bound_edge: float = 0.0


def aswt_rebalance(
    m: int,
    k: int,
    n: int,
    *,
    aic_num: int = 28,
    dtype_bytes: int = 2,
    is_a_trans: bool = False,
    is_b_trans: bool = True,  # FFN Linear [N,K] → kernel transB
    prefer_ub_db2: bool = True,
    prefer_l0c_db2: bool = False,
    prefer_l0c_m_split_db2: bool = False,
    prefer_no_m_split: bool = False,
    has_bias: bool = False,
    l2_size: int = _L2,
    swiglu_single_n_align32: bool = False,
) -> AswtResult:
    """Port of GetRebalanceBlock (+ GetBaseK) for FFN Linear fp16 ND.

    ``swiglu_single_n_align32`` mirrors CPP ``args.swigluSingleNAlign32``:
    when searching Up for SwiGLU single (n=2H), raise ``baseN`` step/bound to
    32 so epilogue half-width stays 32B-aligned. Missing this flag produced
    illegal-for-single tiles (e.g. S49 64×496) that CPP never selects.
    """
    m = max(1, int(m))
    k = max(1, int(k))
    n = max(1, int(n))
    aic_num = max(1, int(aic_num))
    used = aic_num

    hbm = _hbm_bw()
    l2 = _l2_bw()
    compute = (_CORE_FREQ * 8.0) * aic_num
    cmr = (m + n) / (m * n)
    l2_usage = max((m + n) * k * dtype_bytes / max(1, int(l2_size)), 1.0)
    cube_edge = (l2 / compute) + l2_usage * (1 - l2 / hbm) * cmr - (1 + l2 / hbm) / k

    base_mn_limit = _L0C if used == aic_num else min(_L0C, _UB)
    if prefer_l0c_db2:
        base_mn_limit = _L0C // 2

    base_m_best = min(_ceil_align(m, _BASIC_16), _BASIC_256)
    base_n_best = max(
        _BASIC_16,
        min(
            _ceil_align(n, _BASIC_16),
            _floor_align(base_mn_limit // _FP32 // base_m_best, _BASIC_16),
        ),
    )
    cube_param_best = (1.0 / base_m_best) + (1.0 / base_n_best)
    memory_bound = cube_param_best > cube_edge
    inner_align = _BASIC_128 if memory_bound else _BASIC_64

    fixp_edge = (m * n * hbm) / ((m + n) * l2)
    base_m_align = (inner_align // dtype_bytes) if is_a_trans else _BASIC_16
    if k < fixp_edge:
        base_n_align = _K_256_BYTE // dtype_bytes
    elif is_b_trans:
        base_n_align = _BASIC_16
    else:
        base_n_align = inner_align // dtype_bytes
    if swiglu_single_n_align32:
        # CPP matmul_v3_tiling_helper: baseNAlignUnit = max(..., 32).
        base_n_align = max(base_n_align, _BASIC_32)

    max_base_m = _max_base_with_limit(
        m,
        k,
        dtype_bytes,
        base_mn_limit,
        base_m_align,
        is_n=False,
        memory_bound=memory_bound,
        is_a_trans=is_a_trans,
        is_b_trans=is_b_trans,
        has_bias=has_bias,
    )
    max_base_n = _max_base_with_limit(
        n,
        k,
        dtype_bytes,
        base_mn_limit,
        base_n_align,
        is_n=True,
        memory_bound=memory_bound,
        is_a_trans=is_a_trans,
        is_b_trans=is_b_trans,
        has_bias=has_bias,
    )

    base_m = max(_BASIC_16, min(max_base_m, _BASIC_256))
    base_n = max(
        _BASIC_16,
        min(
            max_base_n,
            _floor_align(base_mn_limit // _FP32 // max(1, base_m), base_n_align),
        ),
    )
    cube_param = (1.0 / base_m) + (1.0 / base_n)
    cube_edge = cube_edge * _CUBE_BOUND_RATIO
    balance = _balance_rate(m, n, used, base_m, base_n)

    cur_m = max_base_m
    while cur_m >= 1:
        if prefer_no_m_split and cur_m < _ceil_align(m, _BASIC_16):
            if cur_m <= base_m_align:
                break
            cur_m -= base_m_align
            continue
        cur_max_n = min(
            max_base_n,
            _floor_align(base_mn_limit // _FP32 // max(1, cur_m), base_n_align),
        )
        cur_n = cur_max_n
        while cur_n >= 1:
            cur_cube = (1.0 / cur_m) + (1.0 / cur_n)
            cur_bal = _balance_rate(m, n, used, cur_m, cur_n)
            skip = (
                balance >= _BALANCE_EDGE
                and cur_cube > cube_param
                and cur_cube > cube_edge
                and cube_edge > 0
            )
            if not skip:
                cube_bound_cond = cur_cube <= cube_edge and cur_bal > balance
                ratio = cur_cube / cur_bal
                best_ratio = cube_param / balance
                balance_cond = ratio < best_ratio or (
                    abs(ratio - best_ratio) < _EPSILON and cur_bal > balance
                )
                if cube_bound_cond or balance_cond:
                    if cube_bound_cond:
                        cube_edge = cur_cube
                    base_m, base_n = cur_m, cur_n
                    cube_param = cur_cube
                    balance = cur_bal
            if cur_n <= base_n_align:
                break
            cur_n -= base_n_align
        if cur_m <= base_m_align:
            break
        cur_m -= base_m_align

    base_m = min(_ceil_align(m, _BASIC_16), base_m)
    base_n = min(_ceil_align(n, _BASIC_16), base_n)

    if (
        prefer_l0c_m_split_db2
        and base_m > _BASIC_16
        and base_m * base_n * _FP32 * _DB > _L0C
    ):
        base_m = max(_BASIC_16, _floor_align(base_m // 2, _BASIC_16))

    if prefer_ub_db2 and base_n > _BASIC_32 and base_m * base_n * _FP32 > _UB:
        base_n = max(_BASIC_32, _floor_align(base_n // 2, _BASIC_32))

    base_k = _get_base_k(
        base_m,
        base_n,
        k,
        dtype_bytes,
        is_a_trans=is_a_trans,
        is_b_trans=is_b_trans,
    )
    m_core = _ceil_div(m, base_m)
    n_core = _ceil_div(n, base_n)
    used = min(m_core * n_core, used)
    return AswtResult(
        base_m,
        base_n,
        base_k,
        used,
        allow_full_load=False,
        full_load=0,
        cube_bound_param=float(cube_param),
        cube_bound_edge=float(cube_edge),
    )


def _dsl_guard_enabled() -> bool:
    """Product: DSL guard off (was FFN_ASWT_DSL_GUARD opt-in)."""
    return False


def dsl_safe_aswt(
    m: int,
    k: int,
    n: int,
    *,
    aic_num: int = 28,
    prefer_ub_db2: bool = True,
    is_b_trans: bool = True,
    swiglu_single_n_align32: bool = False,
    has_bias: bool = False,
    dtype_bytes: int = 2,
    dsl_guard: bool | None = None,
) -> AswtResult:
    """ASWT result; optional DSL runtime guard (off by default for CPP parity).

    When ``FFN_ASWT_DSL_GUARD=1`` (or ``dsl_guard=True``), grow pure-ASWT
    **16×16** tiles to 32 on one side — historical L0C2UB trap workaround.
    ``has_bias`` forwards to ASWT N-cap (``btSize/DB/fp32``).
    ``dtype_bytes`` sizes L0/L1 A/B (f16/bf16=2; f32=4).
    """
    res = aswt_rebalance(
        m,
        k,
        n,
        aic_num=aic_num,
        dtype_bytes=dtype_bytes,
        prefer_ub_db2=prefer_ub_db2,
        is_b_trans=is_b_trans,
        swiglu_single_n_align32=swiglu_single_n_align32,
        has_bias=has_bias,
    )
    use_guard = _dsl_guard_enabled() if dsl_guard is None else bool(dsl_guard)
    if not use_guard:
        return res
    bm, bn = int(res.base_m), int(res.base_n)
    m = max(1, int(m))
    n = max(1, int(n))
    k = max(1, int(k))
    if bm <= _BASIC_16 and bn <= _BASIC_16:
        if m > _BASIC_16:
            bm = min(_BASIC_32, _ceil_align(m, _BASIC_16))
        elif n > _BASIC_16:
            bn = min(_BASIC_32, _ceil_align(n, _BASIC_16))
        bm = min(bm, _ceil_align(m, _BASIC_16))
        bn = min(bn, _ceil_align(n, _BASIC_16))
        base_k = _get_base_k(bm, bn, k, 2, is_a_trans=False, is_b_trans=is_b_trans)
        used = min(_ceil_div(m, bm) * _ceil_div(n, bn), max(1, int(aic_num)))
        return AswtResult(
            bm,
            bn,
            base_k,
            used,
            allow_full_load=False,
            full_load=0,
            cube_bound_param=res.cube_bound_param,
            cube_bound_edge=res.cube_bound_edge,
        )
    return res


# CPP ASWT may emit baseM/N > 256 (e.g. G540 272×192, SwiGLU single 64×512). Do not
# crush to the legacy heuristic L1 box; L0 still caps at ``_MAX_L0`` and loops
# within L1.
_MAX_L1_ASWT = 512
_L1_SINGLE_SIZE_LIMIT = 48 * 1024  # matmul_v3_common_advanced.h


def _cal_l1_tk_cpp(
    base_m: int,
    base_n: int,
    base_k: int,
    k: int,
    *,
    dtype_bytes: int = 2,
    is_a_trans: bool = False,
    is_b_trans: bool = True,
    has_bias: bool = False,
) -> int:
    """Port of ``MatMulV3TilingHelper::CalL1TilingDefault`` → ``kL1`` (resKL1).

    ``has_bias`` deducts ``baseN * DB * fp32`` from L1 (CPP CalL1TilingDefault).
    """
    base_m = max(1, int(base_m))
    base_n = max(1, int(base_n))
    base_k = max(1, int(base_k))
    k = max(1, int(k))
    dtype_bytes = max(1, int(dtype_bytes))
    is_k_inner = (not is_a_trans) or is_b_trans
    total_l1 = _L1 - (base_n * _DB * _FP32 if has_bias else 0)
    max_step_k = min(_ceil_div(k, base_k), 8)
    k_align_unit = (_K_512_BYTE // dtype_bytes) if is_k_inner else _BASIC_16
    res_kl1 = 0
    single_mte = 0
    for step_k in range(1, max_step_k + 1):
        cur_kl1 = base_k * step_k
        a_l1 = base_m * cur_kl1 * dtype_bytes
        b_l1 = base_n * cur_kl1 * dtype_bytes
        if (a_l1 + b_l1) * _DB > total_l1 or max(a_l1, b_l1) * _DB * 2 > _L1:
            break
        cond_no_res = res_kl1 == 0
        cond_k_align_256b = cur_kl1 % (_K_256_BYTE // dtype_bytes) == 0
        cond_k_align = res_kl1 % k_align_unit != 0 and (
            cond_k_align_256b
            or ((not cond_k_align_256b) and single_mte < _L1_SINGLE_SIZE_LIMIT)
        )
        cond_mte_size = (
            res_kl1 % k_align_unit == 0
            and cur_kl1 % k_align_unit == 0
            and single_mte < _L1_SINGLE_SIZE_LIMIT
        )
        if cond_no_res or cond_k_align or cond_mte_size:
            res_kl1 = cur_kl1
            single_mte = max(a_l1, b_l1)
    if res_kl1 <= 0:
        res_kl1 = base_k
    return min(k, max(base_k, res_kl1))


# CPP GetTilingDataProcess: stepKa = min(STEPKA_THERSHOLD, min(stepKa, stepKb)).
_STEPKA_THRESHOLD = 4


def _ffn_onthefly_kl1(
    base_m: int,
    base_n: int,
    base_k: int,
    k: int,
    *,
    has_bias: bool = False,
    dtype_bytes: int = 2,
) -> int:
    """FFN Basic ASWT ON_THE_FLY → kL1 matching CPP **tiling dump** (may be >K).

    Mirrors ``FfnMatMulV3Tiling::DoOpTiling`` + ``GetTilingDataProcess``::

      remain = L1 - (hasBias ? BIAS_TABLE_NUM * fp32 : 0)
      stepKa = remain / 2 / ((baseM+baseN)*baseK) / aDtypeSize
      stepKa = min(4, stepKa)   # STEPKA_THERSHOLD
      kL1 = stepKa * baseK      # NOT min(K, ·) — L1 slot size in dump

    Device CPP later does ``kL1_ = Min(K, kL1)`` for MTE2; DSL stages must
    mirror that Min on the copy/step path, while buffers keep this published
    value. ``k`` kept for call-site compat only.
    """
    del k
    base_m = max(1, int(base_m))
    base_n = max(1, int(base_n))
    base_k = max(1, int(base_k))
    dtype_bytes = max(1, int(dtype_bytes))
    remain = _L1 - (_BIAS_TABLE_NUM * _FP32 if has_bias else 0)
    denom = 2 * (base_m + base_n) * base_k * dtype_bytes
    step_ka = max(1, remain // denom) if denom > 0 else 1
    step_ka = min(_STEPKA_THRESHOLD, step_ka)
    return max(base_k, step_ka * base_k)


def _check_al1_full_load(
    m: int,
    k: int,
    n: int,
    res: AswtResult,
    *,
    aic_num: int = 28,
    has_bias: bool = False,
    dtype_bytes: int = 2,
    bias_dtype_bytes: int = 4,
) -> bool:
    """Port of ``CheckAL1FullLoad`` (l0C2Out fixed ON_THE_FLY)."""
    if res.cube_bound_param <= res.cube_bound_edge:
        return False
    single_m = max(1, int(res.base_m))
    single_n = max(1, int(res.base_n))
    m_cnt = _ceil_div(m, single_m)
    n_cnt = _ceil_div(n, single_n)
    if n_cnt <= aic_num:
        return False
    if int(k) <= _BASIC_128 and m_cnt != 1:
        return False
    m_al = _ceil_align(m, _BASIC_16)
    k_al = _ceil_align(k, _BASIC_16)
    a_l1 = m_al * k_al * dtype_bytes
    bias_sz = int(res.base_n) * _DB * bias_dtype_bytes if has_bias else 0
    return (a_l1 + bias_sz) <= (_L1 * 3) // 4


def _check_bl1_full_load(
    m: int,
    k: int,
    n: int,
    res: AswtResult,
    *,
    aic_num: int = 28,
    has_bias: bool = False,
    dtype_bytes: int = 2,
    bias_dtype_bytes: int = 4,
) -> bool:
    """Port of ``CheckBL1FullLoad``."""
    if res.cube_bound_param <= res.cube_bound_edge:
        return False
    single_m = max(1, int(res.base_m))
    m_cnt = _ceil_div(m, single_m)
    if m_cnt <= aic_num:
        return False
    n_al = _ceil_align(n, _BASIC_16)
    k_al = _ceil_align(k, _BASIC_16)
    b_l1 = n_al * k_al * dtype_bytes
    bias_sz = n_al * bias_dtype_bytes if has_bias else 0
    return (b_l1 + bias_sz) <= (_L1 * 3) // 4


def detect_down_full_load(
    m: int,
    k: int,
    n: int,
    res: AswtResult,
    *,
    aic_num: int = 28,
    has_bias: bool = False,
    dtype_bytes: int = 2,
) -> int:
    """Return 0/1/2 for Down full-load decision (A preferred over B)."""
    if _check_al1_full_load(
        m, k, n, res, aic_num=aic_num, has_bias=has_bias, dtype_bytes=dtype_bytes
    ):
        return 1
    if _check_bl1_full_load(
        m, k, n, res, aic_num=aic_num, has_bias=has_bias, dtype_bytes=dtype_bytes
    ):
        return 2
    return 0


@dataclass(frozen=True)
class FullLoadDetail:
    """Extra fields after DoAL1/DoBL1 (streaming kL1 + single-core extents)."""

    full_load: int  # 1=A, 2=B
    base_m: int
    base_n: int
    base_k: int
    used_cores: int
    # Streaming-side kL1 (= baseK * stepK); resident side spans full K in L1.
    stream_kl1: int
    single_core_m: int
    single_core_n: int
    m_aligned: int
    n_aligned: int
    k_aligned: int
    l1_buffer_num: int  # 2 or 4


def apply_down_full_load(
    m: int,
    k: int,
    n: int,
    res: AswtResult,
    *,
    aic_num: int = 28,
    has_bias: bool = False,
    dtype_bytes: int = 2,
    bias_dtype_bytes: int = 4,
) -> AswtResult:
    """Host port of DoAL1/DoBL1FullLoad → updated baseMN/K + usedCoreNum."""
    detail = do_down_full_load_detail(
        m,
        k,
        n,
        res,
        aic_num=aic_num,
        has_bias=has_bias,
        dtype_bytes=dtype_bytes,
        bias_dtype_bytes=bias_dtype_bytes,
    )
    if detail is None:
        return res
    return AswtResult(
        detail.base_m,
        detail.base_n,
        detail.base_k,
        detail.used_cores,
        allow_full_load=True,
        full_load=detail.full_load,
        cube_bound_param=res.cube_bound_param,
        cube_bound_edge=res.cube_bound_edge,
    )


def do_down_full_load_detail(
    m: int,
    k: int,
    n: int,
    res: AswtResult,
    *,
    aic_num: int = 28,
    has_bias: bool = False,
    dtype_bytes: int = 2,
    bias_dtype_bytes: int = 4,
) -> FullLoadDetail | None:
    """Full port of ``DoAL1FullLoad`` / ``DoBL1FullLoad`` (incl. stepK)."""
    full = detect_down_full_load(
        m, k, n, res, aic_num=aic_num, has_bias=has_bias, dtype_bytes=dtype_bytes
    )
    if full == 0:
        return None
    base_m = int(res.base_m)
    base_n = int(res.base_n)
    base_k = int(res.base_k)
    m_al = _ceil_align(m, _BASIC_16)
    n_al = _ceil_align(n, _BASIC_16)
    k_al = _ceil_align(k, _BASIC_16)

    if full == 1:
        # DoAL1FullLoad: A resident; stream B; usedCore along N.
        a_l1 = m_al * k_al * dtype_bytes
        bias_sz = base_n * _DB * bias_dtype_bytes if has_bias else 0
        remain = max(0, _L1 - (a_l1 + bias_sz))
        max_n_l1 = remain // max(1, base_k * dtype_bytes * _DB)
        max_n_l0c = _L0C // max(1, base_m * _FP32 * _DB)
        max_n = _floor_align(min(max_n_l1, max_n_l0c), _BASIC_16)
        bal_n = _ceil_align(_ceil_div(n, aic_num), _BASIC_16)
        if max_n > 0:
            base_n = min(base_n, max_n, bal_n)
        # CPP DoAL1FullLoad: FloorAlign(baseN, 128B/dtype) only when !isBTrans.
        # FFN Linear is always isBTrans → publish raw baseN (no host pad/floor).
        max_step = min(
            _ceil_div(k, base_k),
            remain // max(1, base_k * base_n * _DB * dtype_bytes),
            4,
        )
        max_step = max(1, max_step)
        # CPP DoAL1FullLoad: isBTrans + kL1 256B miss → baseK<<=1 and
        # clamp baseN by L0B. FFN Linear is always isBTrans.
        if (
            max_step == remain // max(1, base_k * base_n * _DB * dtype_bytes)
            and (max_step * base_k) % (_K_256_BYTE // dtype_bytes) != 0
            and (base_k * 2) * base_m * dtype_bytes * _DB <= _L0A
        ):
            base_k = base_k * 2
            max_n_l1 = remain // max(1, base_k * dtype_bytes * _DB)
            max_n_l0b = _L0A // max(1, base_k * dtype_bytes * _DB)
            max_n_l0c = _L0C // max(1, base_m * _FP32 * _DB)
            base_n = min(
                base_n,
                _floor_align(min(max_n_l1, max_n_l0b, max_n_l0c), _BASIC_16),
            )
            max_step = min(
                _ceil_div(k, base_k),
                remain // max(1, base_k * base_n * _DB * dtype_bytes),
                4,
            )
            max_step = max(1, max_step)
        step_k = 1
        for sk in range(1, max_step + 1):
            cur = base_k * sk
            used = base_n * cur * dtype_bytes
            if used < _L1_SINGLE_SIZE_LIMIT:
                continue
            if cur % (_K_256_BYTE // dtype_bytes) != 0:
                continue
            step_k = sk
            break
        else:
            step_k = max_step
        stream_kl1 = min(k, base_k * step_k)
        used = min(_ceil_div(n, max(1, base_n)), aic_num)
        b4 = base_k * step_k * base_n * dtype_bytes * 4
        bias4 = (bias_sz // _DB) * 4 if has_bias else 0
        l1_buf = 2 if (b4 + a_l1 + bias4) > _L1 else 4
        return FullLoadDetail(
            1,
            base_m,
            base_n,
            base_k,
            used,
            stream_kl1,
            m,  # singleCoreM = M
            base_n,
            m_al,
            n_al,
            k_al,
            l1_buf,
        )

    # DoBL1FullLoad: B resident; stream A; usedCore along M.
    b_l1 = n_al * k_al * dtype_bytes
    bias_sz = n_al * bias_dtype_bytes if has_bias else 0
    remain = max(0, _L1 - (b_l1 + bias_sz))
    max_m_l1 = remain // max(1, base_k * dtype_bytes * _DB)
    max_m_l0c = _L0C // max(1, base_n * _FP32 * _DB)
    max_m = _floor_align(min(max_m_l1, max_m_l0c), _BASIC_16)
    bal_m = _ceil_align(_ceil_div(m, aic_num), _BASIC_16)
    if max_m > 0:
        base_m = min(base_m, max_m, bal_m)
    max_step = min(
        _ceil_div(k, base_k),
        remain // max(1, base_k * base_m * _DB * dtype_bytes),
        4,
    )
    max_step = max(1, max_step)
    step_k = 1
    for sk in range(1, max_step + 1):
        cur = base_k * sk
        used = base_m * cur * dtype_bytes
        if used < _L1_SINGLE_SIZE_LIMIT:
            continue
        if cur % (_K_256_BYTE // dtype_bytes) != 0:
            continue
        step_k = sk
        break
    else:
        step_k = max_step
    # A not-trans: try baseK<<=1 when kL1 not 256B-aligned at maxStep.
    if (
        max_step == remain // max(1, base_k * base_m * _DB * dtype_bytes)
        and (max_step * base_k) % (_K_256_BYTE // dtype_bytes) != 0
        and (base_k * 2) * base_n * dtype_bytes * _DB <= _L0A
    ):
        base_k = base_k * 2
        max_m_l1 = remain // max(1, base_k * dtype_bytes * _DB)
        max_m_l0a = _L0A // max(1, base_k * dtype_bytes * _DB)
        base_m = min(base_m, _floor_align(min(max_m_l1, max_m_l0a), _BASIC_16))
        step_k = 1
        max_step = min(
            _ceil_div(k, base_k),
            remain // max(1, base_k * base_m * _DB * dtype_bytes),
            4,
        )
        max_step = max(1, max_step)
        for sk in range(1, max_step + 1):
            cur = base_k * sk
            used = base_m * cur * dtype_bytes
            if used < _L1_SINGLE_SIZE_LIMIT:
                continue
            if cur % (_K_256_BYTE // dtype_bytes) != 0:
                continue
            step_k = sk
            break
        else:
            step_k = max_step
    stream_kl1 = min(k, base_k * step_k)
    used = min(_ceil_div(m, max(1, base_m)), aic_num)
    a4 = base_k * step_k * base_m * dtype_bytes * 4
    bias4 = (bias_sz // _DB) * 4 if has_bias else 0
    l1_buf = 2 if (a4 + b_l1 + bias4) > _L1 else 4
    return FullLoadDetail(
        2,
        base_m,
        base_n,
        base_k,
        used,
        stream_kl1,
        base_m,
        n,  # singleCoreN = N
        m_al,
        n_al,
        k_al,
        l1_buf,
    )


def _l0c2out_is_onthefly(
    m: int,
    k: int,
    n: int,
    res: AswtResult,
    *,
    aic_num: int = 28,
    dtype_bytes: int = 2,
    aiv_num: int | None = None,
) -> bool:
    """Port of ``GetL0C2OutDav3510`` → True iff ON_THE_FLY (FFN default)."""
    aiv = int(aiv_num) if aiv_num is not None else int(aic_num) * 2
    if aiv != int(aic_num) * 2:
        return True
    # isValidMKN: k<=256 && m>=256
    if int(k) > _BASIC_256 or int(m) < _BASIC_256:
        return True
    m_cnt = _ceil_div(m, max(1, int(res.base_m)))
    n_cnt = _ceil_div(n, max(1, int(res.base_n)))
    if m_cnt * n_cnt <= aic_num:
        return True
    # Unaligned N: n*cDtype % 128 != 0 && n*cDtype > 256
    n_bytes = int(n) * dtype_bytes
    if n_bytes % 128 == 0 or n_bytes <= _BASIC_256:
        return True
    # Would be ND_FIXPIPE_1_1 (fp16/bf16); FFN then skips ON_THE_FLY stepKa.
    return False


def aswt_to_tiling_params(
    res: AswtResult,
    k: int,
    *,
    allow_full_load: bool,
    has_bias: bool = False,
    m: int | None = None,
    n: int | None = None,
    aic_num: int = 28,
    dtype_bytes: int = 2,
) -> tuple[TilingParams, AswtResult, FullLoadDetail | None]:
    """Map ASWT → TilingParams + possibly full-load AswtResult/detail.

    Full-load (CPP):
      A: l1_tm=align(M), l1_tn=baseN, l1_tk=stream_kl1 (B streams)
      B: l1_tm=baseM, l1_tn=align(N), l1_tk=stream_kl1 (A streams)
    NONE: ON_THE_FLY / CalL1 as before. ``l0_tm/tn`` track the L1 MN tile
    used by mmad (baseM×baseN for full-load; equal to l1 for NONE).
    ``dtype_bytes``: A/B element width for L0/L1 capacity (f16/bf16=2, f32=4).
    """
    db = max(1, int(dtype_bytes))
    detail: FullLoadDetail | None = None
    res_out = res
    if allow_full_load and m is not None and n is not None:
        detail = do_down_full_load_detail(
            int(m),
            max(1, int(k)),
            int(n),
            res,
            aic_num=aic_num,
            has_bias=has_bias,
            dtype_bytes=db,
        )
        if detail is not None:
            res_out = AswtResult(
                detail.base_m,
                detail.base_n,
                detail.base_k,
                detail.used_cores,
                allow_full_load=True,
                full_load=detail.full_load,
                cube_bound_param=res.cube_bound_param,
                cube_bound_edge=res.cube_bound_edge,
            )

    k = max(1, int(k))
    if detail is not None and detail.full_load == 1:
        # A full: resident A spans full M×K; tile MN = (baseM for L0, baseN).
        tm = max(_BASIC_16, int(detail.base_m))
        tn = max(_BASIC_16, int(detail.base_n))
        l1_tm = detail.m_aligned  # published mL1 ≈ full M
        l1_tn = tn
        l1_tk = max(_L0_TK_MIN, int(detail.stream_kl1))
        l0_tm, l0_tn = tm, tn
    elif detail is not None and detail.full_load == 2:
        tm = max(_BASIC_16, int(detail.base_m))
        tn = max(_BASIC_16, int(detail.base_n))
        l1_tm = tm
        l1_tn = detail.n_aligned  # published nL1 ≈ full N
        l1_tk = max(_L0_TK_MIN, int(detail.stream_kl1))
        l0_tm, l0_tn = tm, tn
    else:
        tm = min(_MAX_L1_ASWT, max(_BASIC_16, int(res_out.base_m)))
        tn = min(_MAX_L1_ASWT, max(_BASIC_16, int(res_out.base_n)))
        onthefly = True
        if m is not None and n is not None:
            onthefly = _l0c2out_is_onthefly(int(m), k, int(n), res_out, aic_num=aic_num)
        if onthefly:
            l1_tk = _ffn_onthefly_kl1(
                tm,
                tn,
                int(res_out.base_k),
                k,
                has_bias=has_bias,
                dtype_bytes=db,
            )
        else:
            l1_tk = _cal_l1_tk_cpp(
                tm,
                tn,
                int(res_out.base_k),
                k,
                has_bias=has_bias,
                dtype_bytes=db,
            )
        l1_tm, l1_tn = tm, tn
        l0_tm, l0_tn = tm, tn

    l0_tk = max(
        _L0_TK_MIN,
        min(
            int(res_out.base_k),
            l1_tk,
            _L0_TK_MAX,
            _l0_tk_for(l0_tm, l0_tn, l1_tk),
        ),
    )
    l0_tk = _align_down(l0_tk, _L0_TK_MIN, _L0_TK_MIN)
    # A-full: expand L0 M when capacity allows.
    # B-full: keep baseN as l0_tn; device loads full-N nZ once, then
    # CPP-style ptr+parent-stride L0 strips (resident_cpp).
    if detail is not None and detail.full_load == 1:
        if 2 * l1_tm * l0_tk * db <= _L0A and 2 * l0_tk * l0_tn * db <= _L0A:
            l0_tm = l1_tm
    # Cap L0 N only (never rewrite l1_tn / CPP baseN). Device A-full does
    # CopyInB1 at l1_tn then L1→L0 strips at l0_tn (CPP DoAFullLoad).
    # Guard is ``_DB * l0_{m|n} * l0_tk * dtype_bytes``; shrink must use the
    # same dual-buf formula (old ``L0/(tk*2)`` was single-buf and left A-full
    # ``l0_tn=304`` overflowing L0B → silent Down corruption).
    if l0_tn > _MAX_L1_ASWT or not (
        _DB * l0_tm * l0_tk * db <= _L0A and _DB * l0_tk * l0_tn * db <= _L0A
    ):
        # Prefer keeping l0_tm; shrink l0_tn to fit dual-buf L0B.
        max_tn = max(
            _BASIC_16,
            _floor_align(_L0A // max(1, _DB * l0_tk * db), _BASIC_16),
        )
        l0_tn = min(l0_tn, max_tn, _MAX_L1_ASWT)
        l0_tk = max(
            _L0_TK_MIN,
            min(l0_tk, _l0_tk_for(l0_tm, l0_tn, l1_tk)),
        )
        l0_tk = _align_down(l0_tk, _L0_TK_MIN, _L0_TK_MIN)
    resident_k = 0
    l1_buf_num = 0
    full_load = 0
    if detail is not None and detail.full_load in (1, 2):
        resident_k = int(detail.k_aligned)
        l1_buf_num = int(detail.l1_buffer_num)
        full_load = int(detail.full_load)
    # CPP GetTilingData may publish kL1 > K; device Min(K, kL1) for MTE2/FH.
    work_l1_tk = min(int(l1_tk), int(k))
    # CPP mixInfo.ubDB = (baseM×baseN×fp32 ≤ ubSize) ? 2 : 1 — published
    # baseMN (res_out / DoAL1), not L0-capped l0_tm×l0_tn.
    ub_base_m = int(res_out.base_m)
    ub_base_n = int(res_out.base_n)
    ub_db = 2 if (ub_base_m * ub_base_n * _FP32) <= _UB else 1
    # CPP basic_aswt: dbL0C = baseM×baseN×fp32×DB ≤ l0CSize ? 2 : 1.
    # Search already sizes baseMN under ``max_n_l0c = L0C/(m×fp32×DB)``.
    l0c_db = 2 if (ub_base_m * ub_base_n * _FP32 * _DB) <= _L0C else 1
    # stepKa from published kL1 / baseK; depthA1 = stepKa × DB (basic PP).
    base_k = max(1, int(res_out.base_k))
    step_ka = max(1, int(l1_tk) // base_k)
    depth_a1 = step_ka * _DB
    return (
        TilingParams(
            l1_tm=l1_tm,
            l1_tn=l1_tn,
            l1_tk=l1_tk,
            l0_tm=l0_tm,
            l0_tn=l0_tn,
            l0_tk=l0_tk,
            resident_k=resident_k,
            l1_buf_num=l1_buf_num,
            full_load=full_load,
            work_l1_tk=work_l1_tk,
            # ResetBase: mBaseTailSplitCnt=1, mTailMain=0 (no base-tail redistrib).
            m_base_tail_split_cnt=1,
            n_base_tail_split_cnt=1,
            m_tail_main=0,
            n_tail_main=0,
            ub_db=ub_db,
            l0c_db=l0c_db,
            step_ka=step_ka,
            depth_a1=depth_a1,
        ),
        res_out,
        detail,
    )
