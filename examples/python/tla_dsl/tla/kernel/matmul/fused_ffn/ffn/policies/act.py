# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ActPolicy registry: host-side activation routing (gelu / silu / swiglu).
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ffn.config import ACT_GELU, ACT_SILU, ACT_SWIGLU


@dataclass(frozen=True)
class ActPolicy:
    """Host + documentation contract for one activation path."""

    name: str
    act_id: int
    up_out_width: Callable[[int], int]  # H → Up N (H or 2H)
    needs_gate_workspace: bool
    supports_canonical: bool
    epi_stage: str
    implemented: bool  # False until device epi lands (S2/S4)


def _w_h(h: int) -> int:
    return int(h)


def _w_2h(h: int) -> int:
    return 2 * int(h)


ACT_POLICIES: dict[str, ActPolicy] = {
    "silu": ActPolicy(
        name="silu",
        act_id=ACT_SILU,
        up_out_width=_w_h,
        needs_gate_workspace=False,
        supports_canonical=True,
        epi_stage="epi_silu.silu_apply_vl_*",  # VL in epi_silu; handshake in up_l0c2ub
        implemented=True,
    ),
    "gelu": ActPolicy(
        name="gelu",
        act_id=ACT_GELU,
        up_out_width=_w_h,
        needs_gate_workspace=False,
        supports_canonical=True,
        epi_stage="epi_gelu_erf.gelu_apply_vl_*",
        implemented=True,  # S2: L0C2UB + erf-Padé
    ),
    "swiglu": ActPolicy(
        name="swiglu",
        act_id=ACT_SWIGLU,
        up_out_width=_w_2h,
        needs_gate_workspace=False,  # single; dual flips this
        supports_canonical=False,
        epi_stage="epi_swiglu_single.swiglu_apply_vl_1d",
        implemented=True,  # L0C2UB + CPP dual half-N B (no host-pack)
    ),
}


def parse_activation(name: str) -> ActPolicy:
    key = str(name).strip().lower()
    if key not in ACT_POLICIES:
        raise ValueError(
            f"unsupported activation {name!r}; choose from {sorted(ACT_POLICIES)}"
        )
    return ACT_POLICIES[key]


def get_act_policy(act_id: int) -> ActPolicy:
    for p in ACT_POLICIES.values():
        if p.act_id == int(act_id):
            return p
    raise ValueError(f"unknown act_id={act_id}")
