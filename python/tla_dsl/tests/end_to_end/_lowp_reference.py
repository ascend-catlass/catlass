# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Independent scalar codebooks for low-precision device regressions."""

from bisect import bisect_right
import math
import struct


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


def decode_hif8(code: int) -> float:
    if not 0 <= code < 256:
        raise ValueError("HiF8 code must fit in a byte")
    sign = -1 if code & 128 else 1
    bits = code & 127
    if bits < 8:
        if bits == 0:
            return math.nan if sign < 0 else 0.0
        return sign * 2.0 ** (bits - 23)
    if bits < 16:
        return sign * (1 + (bits & 7) / 8)
    dot = 1 if bits < 32 else 2 if bits < 64 else 3 if bits < 96 else 4
    mantissa_bits = 3 if dot <= 2 else 5 - dot
    exponent_bits = (bits >> mantissa_bits) & ((1 << dot) - 1)
    magnitude = (1 << (dot - 1)) | (exponent_bits & ((1 << (dot - 1)) - 1))
    exponent = -magnitude if exponent_bits & (1 << (dot - 1)) else magnitude
    mantissa = bits & ((1 << mantissa_bits) - 1)
    if exponent == 15 and mantissa == 1:
        return sign * math.inf
    return sign * 2.0**exponent * (1 + mantissa / (1 << mantissa_bits))


def encode_hif8_rna(value: float, *, saturate: bool) -> int:
    """Independent HiF8 nearest-away codebook reference on Ascend 950.

    The next magnitude after 32768 would be 49152 (reserved for infinity),
    hence the non-saturating overflow midpoint is 40960. HiF8 has one zero;
    byte 128 is NaN, not negative zero.
    """
    if math.isnan(value):
        return 0 if saturate else 128
    sign = 128 if value < 0 else 0
    if math.isinf(value) or abs(value) >= 40960:
        return sign | (110 if saturate else 111)
    book = [decode_hif8(code) for code in range(256)]
    return nearest_code(value, book, ties_away=True)


def boundary_samples(codebook):
    """Finite and overflow midpoints, their neighbours and special values."""
    import numpy as np

    finite = sorted(
        {value for value in codebook if math.isfinite(value) and value >= 0}
    )
    values = [0.0, -0.0, math.nan, math.inf, -math.inf, finite[-1] * 2, -finite[-1] * 2]
    # Include the finite-to-NaN/Inf boundary as well as finite-to-finite ties.
    # HiF8 reserves the would-be value 49152 for infinity.
    overflow = (
        40960.0 if finite[-1] == 32768.0 else finite[-1] + (finite[-1] - finite[-2]) / 2
    )
    midpoints = [overflow]
    midpoints.extend((lower + upper) / 2 for lower, upper in zip(finite, finite[1:]))
    for midpoint in midpoints:
        midpoint = np.float32(midpoint)
        for value in (
            np.nextafter(midpoint, np.float32(-np.inf)),
            midpoint,
            np.nextafter(midpoint, np.float32(np.inf)),
        ):
            values.extend((value, -value))
    return np.asarray(values, np.float32)


_HIF8_POSITIVE = sorted(
    (decode_hif8(code), code) for code in range(128) if math.isfinite(decode_hif8(code))
)
_HIF8_MAGNITUDES = [value for value, _ in _HIF8_POSITIVE]


def encode_hif8_hybrid(value: float, *, saturate: bool) -> int:
    """FP32 HYBRID reference from Ascend C Cast's published bit rules.

    The source's low 14 bits provide the threshold; this is deterministic,
    including the documented frac == threshold == 0 increment. Only exponents
    -3 through 3 use nearest-away. Code selection uses the decoded codebook.
    """
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    exponent = ((bits >> 23) & 255) - 127
    mantissa = bits & 0x7FFFFF
    threshold = bits & 0x3FFF
    sign = (bits >> 24) & 128
    if math.isnan(value):
        return 0 if saturate else 128
    if math.isinf(value) or exponent > 15:
        return sign | (110 if saturate else 111)
    if exponent < -23:
        return 0
    if exponent == -23:
        return sign | 1 if (8192 | (mantissa >> 10)) >= threshold else 0
    if abs(exponent) <= 3:
        return encode_hif8_rna(value, saturate=saturate)
    shift = 7 if abs(exponent) <= 7 else 8 if exponent >= -15 else 9
    fraction = (mantissa >> shift) & 0x3FFF
    index = bisect_right(_HIF8_MAGNITUDES, abs(value)) - 1
    if fraction >= threshold:
        index += 1
    code = _HIF8_POSITIVE[index][1] if index < len(_HIF8_POSITIVE) else 111
    return sign | (110 if saturate and code == 111 else code)


def hif8_hybrid_samples():
    """Exponent regions and threshold <, ==, > fraction, including zero ties."""
    import numpy as np

    bits = [0, 0x80000000, 1, 0x80000001, 0x7F800000, 0xFF800000, 0x7FC00000]
    mantissas = {0, 1, 63, 0x3FFF, 0x40000, 0x7FFFF, 0x80000, 0x1FFFFF, 0x7FFFFF}
    for shift in (7, 8, 9):
        equality = (1 << (2 * shift)) + (1 << shift) + 1
        mantissas.update((equality - 1, equality, equality + 1))
    # At exponent -23 the fraction includes the implicit leading significand.
    mantissas.update((8199, 8200, 8201))
    for exponent in range(-25, 18):
        for mantissa in sorted(mantissas):
            magnitude = ((exponent + 127) << 23) | mantissa
            bits.extend((magnitude, magnitude | 0x80000000))
    return np.asarray(bits, np.uint32).view(np.float32)
