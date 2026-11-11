# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Host: SwiGLU epi SIMD lane pick (device uses unpacked dual half-N B, no W1 pack).
from __future__ import annotations

# Preferred f32 VL width (Ascend950 VL=256B → 64 f32; 32 also validated).
SWIGLU_SIMD_LANES = 64


def swiglu_simd_lanes(l1_tn: int) -> int:
    """SIMD lanes for SwiGLU epi.

    Only ``{64, 32}`` are used: the f32 + ``DIST_PACK_B32`` path produces wrong
    results with VL=16/8.

    Prefer a width that **evenly divides** ``half_w = l1_tn/2`` so the epi can
    bake ``USE_ALL_MASK`` (no tail). Example: ``tn=192`` → half=96 → SIMD=32.

    When nothing divides, match CPP ``RegSwigluSingleB16`` (``E = GetVecLen()/4``
    = 64): keep SIMD=64 and let the epi use one ``UpdateMask`` / hybrid tail.
    Old fallback ``largest simd <= half`` picked 32 for half=48 (ASWT tn=96),
    splitting into VL 32+16 vs CPP's single ``UpdateMask(48)`` — ~2× ``aiv_vec``
    on S5 m128.
    """
    half = max(1, int(l1_tn) // 2)
    for simd in (64, 32):
        if simd <= half and half % simd == 0:
            return simd
    # half < 32: SIMD=32 + UpdateMask (epi ptr+elem windows for u at +half_w).
    if half < 32:
        return 32
    return 64
