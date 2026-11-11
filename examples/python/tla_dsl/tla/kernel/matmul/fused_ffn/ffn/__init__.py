# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FFN DSL package (Phase-0 SiLU + Phase-1 act routing)."""

from ffn.config import (
    ACT_GELU,
    ACT_MAP,
    ACT_RAW,
    ACT_SILU,
    ACT_SWIGLU,
    ACT_SWIGLU_DUAL,
    BIAS_AIV_VEC,
    BIAS_CUBE_BT,
    BIAS_NONE,
    DOWN_BASIC,
    DOWN_FULL_A,
    DOWN_FULL_B,
    DOWN_STREAMK,
    LAYOUT_CANONICAL,
    LAYOUT_LINEAR,
    SWIGLU_DUAL,
    SWIGLU_SINGLE,
    UP_L0C2UB,
    FfnConfig,
    build_ffn_config,
    validate_combo,
)
from ffn.tiling import SwizzleParams, TilingParams

__all__ = [
    "ACT_GELU",
    "ACT_MAP",
    "ACT_RAW",
    "ACT_SILU",
    "ACT_SWIGLU",
    "ACT_SWIGLU_DUAL",
    "BIAS_AIV_VEC",
    "BIAS_CUBE_BT",
    "BIAS_NONE",
    "DOWN_BASIC",
    "DOWN_FULL_A",
    "DOWN_FULL_B",
    "DOWN_STREAMK",
    "LAYOUT_CANONICAL",
    "LAYOUT_LINEAR",
    "SWIGLU_DUAL",
    "SWIGLU_SINGLE",
    "UP_L0C2UB",
    "FfnConfig",
    "SwizzleParams",
    "TilingParams",
    "build_ffn_config",
    "validate_combo",
]
