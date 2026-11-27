# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Batch-invariant mm: Kernel + Host in one file.

Matches torch.mm on rank-2 tensors. Each Cube Core owns one (M, N) tile
and reduces the whole K.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)
if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import argparse

import catlass.tla as tla
import torch
import torch_npu  # noqa: F401

from catlass.tla.runtime import from_dlpack
from helpers import TilingParams, get_block_num

_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32)
_COMPILE_CACHE: dict[tuple, object] = {}
_AIC_CACHE: dict[int, int] = {}
_TILING_CACHE: dict[tuple, TilingParams] = {}
_BIND_CACHE: dict[tuple, list] = {}

_L0A_SIZE = 64 * 1024
_L1_SIZE = 512 * 1024
_L0C_SIZE = 256 * 1024
_DB = 2
_ALIGN16 = 16
_ALIGN64 = 64
_ALIGN128 = 128
_ALIGN256 = 256
_MIN_BASE = 112
_MAX_BASE = 336
_LOAD_BALANCE_LIMIT = 0.88
_MAX_SINGLE_CORE_ROUND = 10.0
_LOAD_BALANCING_THRESHOLD = 0.98
_BALANCE_REDUNDANT_THRESHOLD = 0.97
_MIN_EQUALIZATION = 1.02
_MAX_LOOP = 20
_K128_BYTE = 128
_K256_BYTE = 256

# Load-balance table entries: (block on the larger MN side, the other MN side, K, cost).
_BLOCK_TABLE = (
    (336, 192, 48, 1.0),
    (336, 176, 48, 1.0),
    (336, 160, 48, 1.0),
    (336, 144, 48, 1.073),
    (336, 128, 48, 1.167),
    (336, 112, 48, 1.287),
    (320, 192, 48, 1.0),
    (320, 176, 48, 1.0),
    (320, 160, 48, 1.014),
    (320, 144, 48, 1.089),
    (320, 128, 48, 1.183),
    (304, 208, 48, 1.0),
    (304, 192, 48, 1.0),
    (304, 176, 48, 1.0),
    (304, 160, 48, 1.032),
    (304, 144, 48, 1.107),
    (304, 128, 48, 1.201),
    (288, 224, 48, 1.0),
    (288, 208, 48, 1.0),
    (288, 192, 48, 1.0),
    (288, 176, 48, 1.0),
    (288, 160, 48, 1.051),
    (288, 144, 48, 1.126),
    (288, 128, 48, 1.220),
    (272, 240, 48, 1.0),
    (272, 224, 48, 1.0),
    (272, 208, 48, 1.0),
    (272, 192, 48, 1.0),
    (272, 176, 48, 1.012),
    (272, 160, 48, 1.073),
    (272, 144, 48, 1.148),
    (272, 128, 48, 1.242),
    (256, 256, 64, 1.0),
    (256, 240, 64, 1.0),
    (256, 224, 64, 1.0),
    (256, 208, 64, 1.0),
    (256, 192, 64, 1.0),
    (256, 176, 64, 1.037),
    (256, 160, 64, 1.098),
    (256, 144, 64, 1.173),
    (256, 128, 64, 1.267),
    (240, 272, 48, 1.0),
    (240, 256, 64, 1.0),
    (240, 240, 64, 1.0),
    (240, 224, 64, 1.0),
    (240, 208, 64, 1.0),
    (240, 192, 64, 1.014),
    (240, 176, 64, 1.065),
    (240, 160, 64, 1.126),
    (240, 144, 64, 1.201),
    (240, 128, 64, 1.295),
    (224, 288, 48, 1.0),
    (224, 272, 48, 1.0),
    (224, 256, 64, 1.0),
    (224, 240, 64, 1.0),
    (224, 224, 64, 1.0),
    (224, 208, 64, 1.003),
    (224, 192, 64, 1.046),
    (224, 176, 64, 1.097),
    (224, 160, 64, 1.159),
    (224, 144, 64, 1.234),
    (208, 304, 48, 1.0),
    (208, 288, 48, 1.0),
    (208, 272, 48, 1.0),
    (208, 256, 64, 1.0),
    (208, 240, 64, 1.0),
    (208, 224, 64, 1.003),
    (208, 208, 64, 1.04),
    (208, 192, 64, 1.083),
    (208, 176, 64, 1.134),
    (208, 160, 64, 1.196),
    (208, 144, 64, 1.271),
    (192, 336, 48, 1.0),
    (192, 320, 48, 1.0),
    (192, 304, 48, 1.0),
    (192, 288, 48, 1.0),
    (192, 272, 48, 1.0),
    (192, 256, 64, 1.0),
    (192, 240, 64, 1.014),
    (192, 224, 64, 1.046),
    (192, 208, 64, 1.083),
    (192, 192, 80, 1.126),
    (192, 176, 80, 1.178),
    (192, 160, 80, 1.239),
    (176, 336, 48, 1.0),
    (176, 320, 48, 1.0),
    (176, 304, 48, 1.0),
    (176, 288, 48, 1.0),
    (176, 272, 48, 1.012),
    (176, 256, 64, 1.037),
    (176, 240, 64, 1.065),
    (176, 224, 64, 1.097),
    (176, 208, 64, 1.134),
    (176, 192, 80, 1.178),
    (176, 176, 80, 1.229),
    (176, 160, 80, 1.290),
    (160, 336, 48, 1.0),
    (160, 320, 48, 1.014),
    (160, 304, 48, 1.032),
    (160, 288, 48, 1.051),
    (160, 272, 48, 1.073),
    (160, 256, 64, 1.098),
    (160, 240, 64, 1.126),
    (160, 224, 64, 1.159),
    (160, 208, 64, 1.196),
    (160, 192, 80, 1.239),
    (160, 176, 80, 1.290),
    (144, 336, 48, 1.073),
    (144, 320, 48, 1.089),
    (144, 304, 48, 1.107),
    (144, 288, 48, 1.126),
    (144, 272, 48, 1.148),
    (144, 256, 64, 1.173),
    (144, 240, 64, 1.201),
    (144, 224, 64, 1.234),
    (144, 208, 64, 1.271),
    (128, 336, 48, 1.167),
    (128, 320, 48, 1.183),
    (128, 304, 48, 1.201),
    (128, 288, 48, 1.220),
    (128, 272, 48, 1.242),
    (128, 256, 64, 1.267),
    (128, 240, 64, 1.295),
    (112, 336, 48, 1.287),
)
_BLOCK_LOOKUP = {
    336: (6, 0),
    320: (5, 6),
    304: (6, 11),
    288: (7, 17),
    272: (8, 24),
    256: (9, 32),
    240: (10, 41),
    224: (10, 51),
    208: (11, 61),
    192: (12, 72),
    176: (12, 84),
    160: (11, 96),
    144: (9, 107),
    128: (7, 116),
    112: (1, 123),
}


def _ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        return 0
    return (int(value) + int(divisor) - 1) // int(divisor)


def _ceil_align(value: int, align: int) -> int:
    return _ceil_div(value, align) * align


def _floor_align(value: int, align: int) -> int:
    if align <= 0:
        return int(value)
    return (int(value) // int(align)) * int(align)


def _elem_bytes(dtype: torch.dtype) -> int:
    return 4 if dtype == torch.float32 else 2


def _multi_core_balance(
    m: int, n: int, aic_num: int, base_m: int, base_n: int
) -> float:
    if base_m <= 0 or base_n <= 0 or aic_num <= 0:
        return 0.0
    m_cnt = _ceil_div(m, base_m)
    n_cnt = _ceil_div(n, base_n)
    m_tail = m % base_m
    n_tail = n % base_n
    m_main = m // base_m
    n_main = n // base_n
    total = m_cnt * n_cnt
    total_main = m_main * n_main
    if total == 0:
        return 0.0
    avg = (m * n) / aic_num
    single_max_tail = max(base_m * n_tail, base_n * m_tail)
    core_tail = total % aic_num
    tail_ratio = aic_num // core_tail if core_tail else 1

    def _div(base: int) -> int:
        if base % _ALIGN256 == 0:
            return _ALIGN16
        if base % _ALIGN128 == 0:
            return _ALIGN16 // 2
        if base % _ALIGN64 == 0:
            return 4
        if base % 32 == 0:
            return 2
        return 1

    tail_ratio = min(tail_ratio, _div(base_m) * _div(base_n))
    tail_num = _ceil_div(total, aic_num) - _ceil_div(total_main, aic_num)
    no_div_tail = tail_num - 1 if tail_num >= 2 else 0
    if tail_num == 0:
        extra = _ceil_div(total_main, aic_num) - total_main // aic_num
        max_load = (
            total_main // aic_num
        ) * base_m * base_n + extra * base_m * base_n / tail_ratio
    else:
        max_load = (
            _ceil_div(total_main, aic_num) * base_m * base_n
            + no_div_tail * single_max_tail
            + (tail_num - no_div_tail) * single_max_tail / tail_ratio
        )
    if max_load <= 0:
        return 0.0
    return avg / max_load


def _redundant_move(base_m: int, base_n: int, m: int, n: int) -> int:
    return _ceil_div(m, base_m) * n + _ceil_div(n, base_n) * m - (m + n)


def _calc_basic_block(
    m: int, n: int, base_m: int, base_n: int, used_cores: int
) -> tuple[int, int]:
    m_core = _ceil_div(m, base_m)
    n_core = _ceil_div(n, base_n)
    if m_core == 0 or n_core == 0 or used_cores <= 0:
        return base_m, base_n
    if m_core <= n_core:
        base_m = _ceil_align(_ceil_div(m, m_core), _ALIGN16)
        m_core = _ceil_div(m, base_m)
        n_core = max(used_cores // max(m_core, 1), 1)
        base_n = _ceil_align(_ceil_div(n, n_core), _ALIGN16)
    else:
        base_n = _ceil_align(_ceil_div(n, n_core), _ALIGN16)
        n_core = _ceil_div(n, base_n)
        m_core = max(used_cores // max(n_core, 1), 1)
        base_m = _ceil_align(_ceil_div(m, m_core), _ALIGN16)
    while base_n >= base_m * 2 and n_core < used_cores // 2:
        n_core = n_core * 2
        m_core = max(used_cores // n_core, 1)
        base_m = _ceil_align(_ceil_div(m, m_core), _ALIGN16)
        base_n = _ceil_align(_ceil_div(n, n_core), _ALIGN16)
        m_core = _ceil_div(m, base_m)
        n_core = _ceil_div(n, base_n)
        if m_core <= 0 or n_core <= 0:
            break
    while base_m >= base_n * 2 and m_core < used_cores // 2:
        m_core = m_core * 2
        n_core = max(used_cores // m_core, 1)
        base_m = _ceil_align(_ceil_div(m, m_core), _ALIGN16)
        base_n = _ceil_align(_ceil_div(n, n_core), _ALIGN16)
        m_core = _ceil_div(m, base_m)
        n_core = _ceil_div(n, base_n)
        if m_core <= 0 or n_core <= 0:
            break
    return max(base_m, _ALIGN16), max(base_n, _ALIGN16)


def _update_base_block(
    base_block: int, is_m_larger: bool, dtype_size: int, k: int
) -> int:
    if not is_m_larger:
        align = _ALIGN128 // dtype_size if k <= _ALIGN256 else _ALIGN64 // dtype_size
        return _ceil_align(base_block, max(align, 1))
    return _ceil_align(base_block, _ALIGN16)


def _handle_large_single_side(
    m: int,
    n: int,
    min_mn: int,
    max_mn: int,
    is_m_larger: bool,
    aic_num: int,
    dtype_size: int,
    k: int,
) -> tuple[int, int]:
    db_l0c = 1
    if is_m_larger:
        base_n = min_mn
        base_m = _floor_align(_L0C_SIZE // db_l0c // max(base_n, 1) // 4, _ALIGN16)
        target = base_m
    else:
        base_m = min_mn
        base_n = _floor_align(_L0C_SIZE // db_l0c // max(base_m, 1) // 4, 32)
        target = base_n
    min_cores = (aic_num + 1) * 9 // 10
    found = target
    for tmp_cores in range(aic_num, min_cores - 1, -1):
        for loop in range(1, _MAX_LOOP + 1):
            block = _update_base_block(
                _ceil_div(max_mn, tmp_cores * loop), is_m_larger, dtype_size, k
            )
            tile = block * min_mn * 4
            if _MIN_BASE <= block <= _MAX_BASE and tile <= _L0C_SIZE:
                found = block
                if is_m_larger:
                    return max(found, _ALIGN16), max(base_n, _ALIGN16)
                return max(base_m, _ALIGN16), max(found, _ALIGN16)
    if is_m_larger:
        return max(base_m, _ALIGN16), max(base_n, _ALIGN16)
    return max(base_m, _ALIGN16), max(base_n, _ALIGN16)


def _n_not_aligned(base_n: int, dtype_size: int, k: int) -> bool:
    if k <= _ALIGN256:
        return base_n * dtype_size % _ALIGN128 != 0
    return base_n * dtype_size % _ALIGN64 != 0


def _handle_large_both_sides(
    m: int,
    n: int,
    k: int,
    aic_num: int,
    dtype_size: int,
    is_fp32: bool,
    is_m_larger: bool,
    higher_x: int,
    lower_x: int,
    min_mn: int,
    default_balance: float,
    redundant: int,
) -> tuple[int, int]:
    init_k = _K128_BYTE // dtype_size
    best = {
        "balance": default_balance,
        "m": _K256_BYTE,
        "n": _K256_BYTE,
        "k": init_k,
        "hit": False,
    }

    def search(start: int, end: int, negative: bool) -> bool:
        base_x = start
        while (base_x >= end) if negative else (base_x <= end):
            lookup = _BLOCK_LOOKUP.get(base_x)
            if lookup is None:
                base_x = base_x - _ALIGN16 if negative else base_x + _ALIGN16
                continue
            count, start_index = lookup
            for i in range(count):
                x1, x2, x3, x4 = _BLOCK_TABLE[start_index + i]
                if x3 > k:
                    continue
                if is_fp32 and (x3 // 2) % _ALIGN16 != 0:
                    continue
                cur_m = x2 if is_m_larger else x1
                cur_n = x1 if is_m_larger else x2
                if _n_not_aligned(cur_n, dtype_size, k):
                    continue
                if cur_m * dtype_size % _ALIGN64 != 0:
                    continue
                balance = _multi_core_balance(m, n, aic_num, cur_m, cur_n) / x4
                remove = _redundant_move(cur_m, cur_n, m, n) / max(redundant, 1)
                if balance - default_balance <= remove - _BALANCE_REDUNDANT_THRESHOLD:
                    continue
                if balance > _LOAD_BALANCING_THRESHOLD:
                    best["m"] = cur_m
                    best["n"] = cur_n
                    best["hit"] = True
                    return True
                if (
                    balance > best["balance"]
                    and balance > _MIN_EQUALIZATION * default_balance
                ):
                    best["balance"] = balance
                    best["m"] = cur_m
                    best["n"] = cur_n
                    best["hit"] = True
            base_x = base_x - _ALIGN16 if negative else base_x + _ALIGN16
        return False

    if search(lower_x, _MIN_BASE, True):
        return best["m"], best["n"]
    second = dict(best)
    if search(higher_x, min(min_mn, _MAX_BASE), False):
        return best["m"], best["n"]
    if second["hit"] or best["hit"]:
        pick = second if second["balance"] >= best["balance"] else best
        return pick["m"], pick["n"]
    return best["m"], best["n"]


def _cal_l1_stepk(
    m: int, n: int, k: int, base_m: int, base_n: int, base_k: int, dtype_size: int
) -> int:
    half_l1 = _L1_SIZE // 2
    depth_a = max(half_l1 // max(base_m * base_k * dtype_size, 1), 1)
    depth_b = max(half_l1 // max(base_n * base_k * dtype_size, 1), 1)
    if (
        depth_a * base_m * base_k * dtype_size + depth_b * base_n * base_k * dtype_size
        > half_l1 * 2
    ):
        if base_m <= base_n:
            depth_a = max(depth_a // 2, 1)
        else:
            depth_b = max(depth_b // 2, 1)
    step_ka = max(depth_a // _DB, 1)
    step_kb = max(depth_b // _DB, 1)
    if (
        base_m == _ALIGN256
        and base_n == _ALIGN256
        and m % _ALIGN16 == 0
        and n % _ALIGN16 == 0
        and k % _ALIGN16 == 0
        and k <= _ALIGN256
    ):
        step_ka = min(step_ka, 2)
        step_kb = min(step_kb, 2)
    if step_ka >= step_kb:
        step_ka = max(step_ka // step_kb * step_kb, 1)
    else:
        step_kb = max(step_kb // step_ka * step_ka, 1)
    return min(step_ka, step_kb)


def tiling_for_shape(
    m: int, n: int, k: int, dtype: torch.dtype, aic_num: int
) -> TilingParams:
    """Select L1/L0 tiles for one GEMM on DAV_3510. Batch count is not an input."""
    m = max(int(m), 1)
    n = max(int(n), 1)
    k = max(int(k), 1)
    aic_num = max(int(aic_num), 1)
    dtype_size = _elem_bytes(dtype)
    is_fp32 = dtype == torch.float32
    base_m = _ALIGN256
    base_n = _ALIGN256
    used_cores = aic_num
    base_m = min(_ceil_align(m, _ALIGN16), base_m)
    base_n = min(_ceil_align(n, _ALIGN16), base_n)
    default_balance = _multi_core_balance(m, n, aic_num, base_m, base_n)
    redundant = max(_redundant_move(base_m, base_n, m, n), 1)
    m_core = _ceil_div(m, base_m)
    n_core = _ceil_div(n, base_n)
    single_blocks = (m_core * n_core) // aic_num
    if (
        1.0 <= single_blocks <= _MAX_SINGLE_CORE_ROUND
        and default_balance < _LOAD_BALANCE_LIMIT
    ):
        data = float(m * n) / aic_num
        while data > float(_ALIGN256 * _ALIGN256):
            data /= 2.0
        best = math.sqrt(data)
        higher = _ceil_align(int(math.ceil(best)), _ALIGN16)
        lower = _floor_align(int(math.floor(best)), _ALIGN16)
        min_mn = _ceil_align(min(m, n), _ALIGN16)
        max_mn = _ceil_align(max(m, n), _ALIGN16)
        is_m_larger = m > n
        if lower >= min_mn:
            base_m, base_n = _handle_large_single_side(
                m, n, min_mn, max_mn, is_m_larger, aic_num, dtype_size, k
            )
        else:
            base_m, base_n = _handle_large_both_sides(
                m,
                n,
                k,
                aic_num,
                dtype_size,
                is_fp32,
                is_m_larger,
                higher,
                lower,
                min_mn,
                default_balance,
                redundant,
            )
    if single_blocks < 1:
        base_m, base_n = _calc_basic_block(m, n, base_m, base_n, used_cores)
    base_m = max(_ceil_align(base_m, _ALIGN16), _ALIGN16)
    base_n = max(_ceil_align(base_n, _ALIGN16), _ALIGN16)
    k_align = _ALIGN16
    k_max = _floor_align(_L0A_SIZE // _DB // dtype_size // max(base_m, base_n), k_align)
    base_k = min(_ceil_align(k, k_align), max(k_max, k_align))
    step_k = _cal_l1_stepk(m, n, k, base_m, base_n, base_k, dtype_size)
    l1_tk = max(step_k * base_k, base_k)
    l1_tk = min(l1_tk, max(_ceil_align(k, base_k), base_k))
    l0a = base_m * base_k * dtype_size * _DB
    l0b = base_k * base_n * dtype_size * _DB
    l0c = base_m * base_n * 4
    l1 = (base_m * l1_tk + l1_tk * base_n) * dtype_size * _DB
    while (
        l0a > _L0A_SIZE or l0b > _L0A_SIZE or l0c > _L0C_SIZE or l1 > _L1_SIZE
    ) and base_k > _ALIGN16:
        base_k = max(_floor_align(base_k // 2, _ALIGN16), _ALIGN16)
        step_k = _cal_l1_stepk(m, n, k, base_m, base_n, base_k, dtype_size)
        l1_tk = max(step_k * base_k, base_k)
        l0a = base_m * base_k * dtype_size * _DB
        l0b = base_k * base_n * dtype_size * _DB
        l1 = (base_m * l1_tk + l1_tk * base_n) * dtype_size * _DB
    while l1 > _L1_SIZE and l1_tk > base_k:
        l1_tk -= base_k
        l1 = (base_m * l1_tk + l1_tk * base_n) * dtype_size * _DB
    l1_tk = min(max(l1_tk, base_k), max(_ceil_align(k, base_k), base_k))
    return TilingParams(
        l1_tm=base_m,
        l1_tn=base_n,
        l1_tk=l1_tk,
        l0_tm=base_m,
        l0_tn=base_n,
        l0_tk=base_k,
    )


@tla.kernel
def mm_batchinvariant_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    _tiling: tla.Constexpr[TilingParams],
) -> None:
    """One Cube Core owns each (M, N) tile and reduces the whole K."""
    c0 = 0
    c1 = 1

    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype

    k_dim = gm_a.origin_shape[1]
    n_dim = gm_c.origin_shape[1]
    batch_cnt = gm_b.origin_shape[0] // k_dim
    m_dim = gm_a.origin_shape[0] // batch_cnt

    mutex_l1a0 = tla.mutex(resource="l1a0", id=0)
    mutex_l1a1 = tla.mutex(resource="l1a1", id=1)
    mutex_l1b0 = tla.mutex(resource="l1b0", id=2)
    mutex_l1b1 = tla.mutex(resource="l1b1", id=3)
    mutex_l0a0 = tla.mutex(resource="l0a0", id=4)
    mutex_l0a1 = tla.mutex(resource="l0a1", id=5)
    mutex_l0b0 = tla.mutex(resource="l0b0", id=6)
    mutex_l0b1 = tla.mutex(resource="l0b1", id=7)

    l1a0_ptr = tla.allocate(
        _tiling.l1_tm * _tiling.l1_tk, dtype_a, tla.AddressSpace.l1, 512
    )
    l1a1_ptr = tla.allocate(
        _tiling.l1_tm * _tiling.l1_tk, dtype_a, tla.AddressSpace.l1, 512
    )
    l1b0_ptr = tla.allocate(
        _tiling.l1_tk * _tiling.l1_tn, dtype_b, tla.AddressSpace.l1, 512
    )
    l1b1_ptr = tla.allocate(
        _tiling.l1_tk * _tiling.l1_tn, dtype_b, tla.AddressSpace.l1, 512
    )
    l0a0_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tk, dtype_a, tla.AddressSpace.l0a, 512
    )
    l0a1_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tk, dtype_a, tla.AddressSpace.l0a, 512
    )
    l0b0_ptr = tla.allocate(
        _tiling.l0_tk * _tiling.l0_tn, dtype_b, tla.AddressSpace.l0b, 512
    )
    l0b1_ptr = tla.allocate(
        _tiling.l0_tk * _tiling.l0_tn, dtype_b, tla.AddressSpace.l0b, 512
    )
    l0c_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tn, tla.Float32, tla.AddressSpace.l0c, 1024
    )

    grid_m = (m_dim + _tiling.l1_tm - 1) // _tiling.l1_tm
    grid_n = (n_dim + _tiling.l1_tn - 1) // _tiling.l1_tn
    mn_blocks = grid_m * grid_n
    total_blocks = batch_cnt * mn_blocks

    with tla.cube():
        runtime_zero = tla.as_numeric(0)
        l1_buf_idx = runtime_zero
        l0_buf_idx = runtime_zero
        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )
        for loop_idx in block_range:
            batch_idx = loop_idx // mn_blocks
            mn_linear = loop_idx % mn_blocks
            block_row = mn_linear // grid_n
            block_col = mn_linear % grid_n

            # Batch view first, so an M tile cannot cross into the next batch.
            gm_a_batch = tla.tile_view(
                gm_a, tla.make_shape(m_dim, k_dim), tla.make_coord(batch_idx, c0)
            )
            gm_b_batch = tla.tile_view(
                gm_b, tla.make_shape(k_dim, n_dim), tla.make_coord(batch_idx, c0)
            )
            gm_c_batch = tla.tile_view(
                gm_c, tla.make_shape(m_dim, n_dim), tla.make_coord(batch_idx, c0)
            )
            gm_a_by_core = tla.tile_view(
                gm_a_batch,
                tla.make_shape(_tiling.l1_tm, k_dim),
                tla.make_coord(block_row, c0),
            )
            gm_b_by_core = tla.tile_view(
                gm_b_batch,
                tla.make_shape(k_dim, _tiling.l1_tn),
                tla.make_coord(c0, block_col),
            )
            gm_c_by_core = tla.tile_view(
                gm_c_batch,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(block_row, block_col),
            )

            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + _tiling.l1_tk - 1) // _tiling.l1_tk
            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

            for k_l1 in tla.range(c0, k_l1_count, c1):
                gm_a_by_l1 = tla.tile_view(
                    gm_a_by_core,
                    tla.make_shape(_tiling.l1_tm, _tiling.l1_tk),
                    tla.make_coord(c0, k_l1),
                )
                gm_b_by_l1 = tla.tile_view(
                    gm_b_by_core,
                    tla.make_shape(_tiling.l1_tk, _tiling.l1_tn),
                    tla.make_coord(k_l1, c0),
                )
                l1_a = tla.make_tensor_like(
                    l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_by_l1
                )
                l1_b = tla.make_tensor_like(
                    l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_by_l1
                )
                mutex_l1a = mutex_l1a0 if (l1_buf_idx == c0) else mutex_l1a1
                mutex_l1a.lock(pipe=tla.pipes.MTE2)
                tla.copy(l1_a, gm_a_by_l1)
                mutex_l1a.unlock(pipe=tla.pipes.MTE2)
                mutex_l1b = mutex_l1b0 if (l1_buf_idx == c0) else mutex_l1b1
                mutex_l1b.lock(pipe=tla.pipes.MTE2)
                tla.copy(l1_b, gm_b_by_l1)
                mutex_l1b.unlock(pipe=tla.pipes.MTE2)

                k_l0_count = (l1_a.origin_shape[1] + _tiling.l0_tk - 1) // _tiling.l0_tk
                for k_l0 in tla.range(c0, k_l0_count, c1):
                    l1_a_by_l0 = tla.tile_view(
                        l1_a,
                        tla.make_shape(_tiling.l0_tm, _tiling.l0_tk),
                        tla.make_coord(c0, k_l0),
                    )
                    l1_b_by_l0 = tla.tile_view(
                        l1_b,
                        tla.make_shape(_tiling.l0_tk, _tiling.l0_tn),
                        tla.make_coord(k_l0, c0),
                    )
                    l0_a = tla.make_tensor_like(
                        l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_by_l0
                    )
                    l0_b = tla.make_tensor_like(
                        l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_by_l0
                    )
                    mutex_l0a = mutex_l0a0 if (l0_buf_idx == c0) else mutex_l0a1
                    mutex_l1a.lock(pipe=tla.pipes.MTE1)
                    mutex_l0a.lock(pipe=tla.pipes.MTE1)
                    tla.copy(l0_a, l1_a_by_l0)
                    mutex_l0a.unlock(pipe=tla.pipes.MTE1)
                    mutex_l1a.unlock(pipe=tla.pipes.MTE1)
                    mutex_l0b = mutex_l0b0 if (l0_buf_idx == c0) else mutex_l0b1
                    mutex_l1b.lock(pipe=tla.pipes.MTE1)
                    mutex_l0b.lock(pipe=tla.pipes.MTE1)
                    tla.copy(l0_b, l1_b_by_l0)
                    mutex_l0b.unlock(pipe=tla.pipes.MTE1)
                    mutex_l1b.unlock(pipe=tla.pipes.MTE1)

                    mutex_l0a.lock(pipe=tla.pipes.CUBE)
                    mutex_l0b.lock(pipe=tla.pipes.CUBE)
                    init_c = True if k_l1 == 0 and k_l0 == 0 else False
                    unit_flag = (
                        0b11
                        if (k_l1 == k_l1_count - 1) and (k_l0 == k_l0_count - 1)
                        else 0b10
                    )
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
                    mutex_l0b.unlock(pipe=tla.pipes.CUBE)
                    mutex_l0a.unlock(pipe=tla.pipes.CUBE)
                    l0_buf_idx = c1 - l0_buf_idx
                l1_buf_idx = c1 - l1_buf_idx

            tla.copy(
                gm_c_by_core,
                l0_c,
                tla.params.CopyL0C2DstParams(unit_flag=0b11),
            )


def _cube_cores(device_index: int) -> int:
    cached = _AIC_CACHE.get(device_index)
    if cached is None:
        cached = get_block_num(-1, device_index, kind="cube")
        _AIC_CACHE[device_index] = cached
    return cached


def _cached_tiling(
    m_dim: int, n_dim: int, k_dim: int, dtype: torch.dtype, aic_num: int
) -> TilingParams:
    key = (m_dim, n_dim, k_dim, dtype, aic_num)
    tiling = _TILING_CACHE.get(key)
    if tiling is None:
        tiling = tiling_for_shape(m_dim, n_dim, k_dim, dtype, aic_num)
        _TILING_CACHE[key] = tiling
    return tiling


def _to_npu_contiguous(buf: torch.Tensor) -> torch.Tensor:
    if buf.device.type != "npu":
        buf = buf.to("npu")
    if not buf.is_contiguous():
        buf = buf.contiguous()
    return buf


def _tla_row(buf: torch.Tensor):
    return from_dlpack(buf, layout_tag=tla.arch.RowMajor)


def _bind_tensors(a_packed, b_packed, c_packed, cache_key: tuple):
    bound = _BIND_CACHE.get(cache_key)
    if bound is None:
        bound = [_tla_row(a_packed), _tla_row(b_packed), _tla_row(c_packed)]
        _BIND_CACHE[cache_key] = bound
        return bound
    for tla_t, buf in zip(bound, (a_packed, b_packed, c_packed)):
        tla_t.data_ptr = int(buf.data_ptr())
        tla_t._launch_keep = buf
    return bound


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("mm expects two rank-2 tensors")
    source = a.device
    if a.dtype != b.dtype or a.dtype not in _SUPPORTED:
        raise ValueError("a and b must share float16, bfloat16, or float32")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("contracted K dimensions differ")

    m_dim = int(a.shape[0])
    k_dim = int(a.shape[1])
    n_dim = int(b.shape[1])
    if m_dim == 0 or n_dim == 0:
        return a.new_empty(m_dim, n_dim).to(source)
    if k_dim == 0:
        return a.new_zeros(m_dim, n_dim).to(source)

    current = torch.npu.current_device()
    npu_index = a.device.index if a.device.type == "npu" else current
    if current != npu_index:
        torch.npu.set_device(npu_index)
    aic_num = _cube_cores(npu_index)
    a_packed = _to_npu_contiguous(a)
    b_packed = _to_npu_contiguous(b)
    c_packed = torch.empty(m_dim, n_dim, dtype=a.dtype, device="npu")

    tiling = _cached_tiling(m_dim, n_dim, k_dim, a.dtype, aic_num)
    cache_key = (
        a.dtype,
        m_dim,
        n_dim,
        k_dim,
        tiling.l1_tm,
        tiling.l1_tn,
        tiling.l1_tk,
        tiling.l0_tm,
        tiling.l0_tn,
        tiling.l0_tk,
        aic_num,
    )
    tensors = _bind_tensors(a_packed, b_packed, c_packed, cache_key)
    artifact = _COMPILE_CACHE.get(cache_key)
    if artifact is None:
        artifact = tla.compile(
            mm_batchinvariant_kernel,
            *tensors,
            tiling,
            options="--npu-arch 3510",
        )
        _COMPILE_CACHE[cache_key] = artifact
    artifact(*tensors, block_num=aic_num)

    out = c_packed
    if source.type != "npu":
        out = out.to(source)
    return out


_DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}


def main() -> int:
    from helpers import compare

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--dtype", choices=("f16", "bf16", "f32"), default="f16")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    print(f"--- mnk=({args.m},{args.n},{args.k}) dtype={args.dtype} ---")
    torch.manual_seed(0)
    dtype = _DTYPES[args.dtype]
    a_cpu = torch.rand(args.m, args.k, dtype=dtype) * 10.0 - 5.0
    b_cpu = torch.rand(args.k, args.n, dtype=dtype) * 10.0 - 5.0
    ref = a_cpu.float() @ b_cpu.float()
    if dtype in (torch.float16, torch.bfloat16):
        ref = ref.to(dtype).float()
    got = run(a_cpu.npu(), b_cpu.npu())
    torch.npu.synchronize()
    passed = compare(got.detach().cpu(), ref, args.k)
    artifact = next(reversed(_COMPILE_CACHE.values()))
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
