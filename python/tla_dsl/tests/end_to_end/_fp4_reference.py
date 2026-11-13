# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Independent scalar FP4 codebook and rounding reference."""

import math


def decode_fp4(code: int, exponent_bits: int, bias: int) -> float:
    if not 0 <= code < 16 or exponent_bits not in (1, 2):
        raise ValueError("Expected a four-bit code and E1M2 or E2M1")
    sign = -1 if code & 8 else 1
    mantissa_bits = 3 - exponent_bits
    exponent = (code & 7) >> mantissa_bits
    mantissa = code & ((1 << mantissa_bits) - 1)
    significand = (1 if exponent else 0) + mantissa / (1 << mantissa_bits)
    return sign * significand * 2.0 ** (max(exponent, 1) - bias)


def nearest_code(value: float, codebook: list[float], *, ties_away: bool) -> int:
    """Finite nearest conversion by exhaustive distances, preserving zero sign."""
    if not math.isfinite(value):
        raise ValueError("NaN and infinity policy must be checked separately")
    sign = math.copysign(1, value)
    zeros = [(i, x) for i, x in enumerate(codebook) if x == 0]
    matching_zero = next(
        (i for i, x in zeros if math.copysign(1, x) == sign), zeros[0][0]
    )
    candidates = [
        (abs(value - x), i, x)
        for i, x in enumerate(codebook)
        if math.isfinite(x)
        and (
            (x == 0 and i == matching_zero) or (x != 0 and math.copysign(1, x) == sign)
        )
    ]
    distance = min(item[0] for item in candidates)
    tied = [(i, x) for d, i, x in candidates if d == distance]
    if ties_away:
        return max(tied, key=lambda item: abs(item[1]))[0]
    return min(tied, key=lambda item: (item[0] & 1, item[0]))[0]
