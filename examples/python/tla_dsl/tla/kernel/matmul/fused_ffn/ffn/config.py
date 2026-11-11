# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# FFN compile-time strategy enums / config (Phase-0 + Phase-1 routing).
from __future__ import annotations

from dataclasses import dataclass

import catlass.tla as tla

# Activation tags (SILU/GELU/SWIGLU); RAW/DUAL are DSL Up epi tags
ACT_SILU = 0
ACT_GELU = 1
ACT_SWIGLU = 2  # SwiGLU single (2H interleave)
ACT_RAW = 3  # dual stage-1: store FP32 acc → gate GM (no act)
ACT_SWIGLU_DUAL = 4  # dual stage-2: silu(gate)*up → hidden

# Bias
BIAS_NONE = 0
BIAS_AIV_VEC = 1
BIAS_CUBE_BT = 2

# Up data path (product: L0C→UB FixPipe only). ``UP_EVG_GM`` kept so host
# ``--up-path evg`` can still branch/reject; device no longer implements it.
UP_EVG_GM = 0
UP_L0C2UB = 1

# Down mode — CPP FFNArch35 / MatMulV3FullLoad (0 NONE, 1 A, 2 B) + StreamK=3
DOWN_BASIC = 0
DOWN_FULL_A = 1  # hidden 常驻 L1；按 N 分核
DOWN_FULL_B = 2  # weight2 常驻 L1；按 M 分核
DOWN_STREAMK = 3

# Layout
LAYOUT_LINEAR = 0
LAYOUT_CANONICAL = 1

# SwiGLU mode (P1 default single)
SWIGLU_SINGLE = 0
SWIGLU_DUAL = 1

ACT_MAP = {"silu": ACT_SILU, "gelu": ACT_GELU, "swiglu": ACT_SWIGLU}


@dataclass(frozen=True)
class FfnConfig:
    """Compile-time FFN strategy knobs (Constexpr-friendly ints)."""

    act: tla.Constexpr[int] = ACT_SILU
    bias_mode: tla.Constexpr[int] = BIAS_NONE
    up_path: tla.Constexpr[int] = UP_L0C2UB
    down_mode: tla.Constexpr[int] = DOWN_BASIC
    layout: tla.Constexpr[int] = LAYOUT_LINEAR
    swiglu_mode: tla.Constexpr[int] = SWIGLU_SINGLE
    # Host-computed: unpacked dual half-N B (CPP CopyInB1) for this shape.
    swiglu_b_cpp: tla.Constexpr[bool] = False
    # Hidden size H (SwiGLU); 0 = unused. Enables constexpr (K,H) B roots.
    swiglu_h: tla.Constexpr[int] = 0
    # Input K (SwiGLU B ColMajor srcD); 0 = unused.
    swiglu_k: tla.Constexpr[int] = 0
    # Batch M (SwiGLU); 0 = unused. Enables constexpr tile-grid for CV id0.
    swiglu_m: tla.Constexpr[int] = 0
    # Output N (SwiGLU Down); 0 = unused. With swiglu_m+BLOCK_DIM → 1-tile/core.
    swiglu_n: tla.Constexpr[int] = 0


def validate_combo(
    act: int,
    layout: int = LAYOUT_LINEAR,
    swiglu_mode: int = SWIGLU_SINGLE,
    *,
    bias_mode: int = BIAS_NONE,
    require_implemented: bool = True,
) -> None:
    """Raise ``ValueError`` for illegal or not-yet-implemented combos."""
    from ffn.policies.act import get_act_policy
    from ffn.policies.bias import get_bias_policy
    from ffn.policies.swiglu import SWIGLU_POLICIES

    pol = get_act_policy(act)
    if int(layout) == LAYOUT_CANONICAL and not pol.supports_canonical:
        raise ValueError(f"activation {pol.name!r} does not support Canonical layout")
    if int(act) == ACT_SWIGLU:
        mode_name = "dual" if int(swiglu_mode) == SWIGLU_DUAL else "single"
        sm = SWIGLU_POLICIES[mode_name]
        if require_implemented and not sm.implemented:
            raise ValueError(
                f"swiglu mode {mode_name!r} not implemented yet (Phase-1 S4/P1.b)"
            )
    if require_implemented and not pol.implemented:
        raise ValueError(
            f"activation {pol.name!r} device path not implemented yet (see ffn_phase1_design.md)"
        )
    bpol = get_bias_policy(bias_mode)
    if require_implemented and not bpol.implemented:
        raise ValueError(
            f"bias mode {bpol.name!r} not implemented yet (see ffn.policies.bias next-PR checklist)"
        )


def build_ffn_config(
    activation: str = "silu",
    *,
    up_path: int = UP_L0C2UB,
    bias_mode: int = BIAS_NONE,
    down_mode: int = DOWN_BASIC,
    layout: int = LAYOUT_LINEAR,
    swiglu_mode: int = SWIGLU_SINGLE,
    require_implemented: bool = True,
) -> FfnConfig:
    """Host helper: activation string → validated ``FfnConfig``."""
    from ffn.policies.act import parse_activation

    if int(up_path) != UP_L0C2UB:
        raise ValueError(
            f"up_path={up_path!r} unsupported; product path is L0C→UB only (UP_L0C2UB={UP_L0C2UB})"
        )
    pol = parse_activation(activation)
    validate_combo(
        pol.act_id,
        layout,
        swiglu_mode,
        bias_mode=bias_mode,
        require_implemented=require_implemented,
    )
    return FfnConfig(
        act=pol.act_id,
        bias_mode=bias_mode,
        up_path=UP_L0C2UB,
        down_mode=down_mode,
        layout=layout,
        swiglu_mode=swiglu_mode,
    )
