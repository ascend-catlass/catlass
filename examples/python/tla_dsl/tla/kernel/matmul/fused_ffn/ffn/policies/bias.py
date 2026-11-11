# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# BiasPolicy registry. Default NONE; cube_bt wires GM→L1→biasbuf→mmad(bias=).
from __future__ import annotations

from dataclasses import dataclass

from ffn.config import BIAS_AIV_VEC, BIAS_CUBE_BT, BIAS_NONE


@dataclass(frozen=True)
class BiasPolicy:
    name: str
    bias_id: int
    implemented: bool


BIAS_POLICIES: dict[str, BiasPolicy] = {
    "none": BiasPolicy("none", BIAS_NONE, implemented=True),
    "aiv_vec": BiasPolicy("aiv_vec", BIAS_AIV_VEC, implemented=False),
    # Cube Bias Table: first-K mmad(..., bias=) on Up/Down (SK: slice0 only).
    "cube_bt": BiasPolicy("cube_bt", BIAS_CUBE_BT, implemented=True),
}


def parse_bias(name: str) -> BiasPolicy:
    key = str(name).strip().lower()
    if key not in BIAS_POLICIES:
        raise ValueError(
            f"unsupported bias mode {name!r}; choose from {sorted(BIAS_POLICIES)}"
        )
    return BIAS_POLICIES[key]


def get_bias_policy(bias_id: int) -> BiasPolicy:
    for p in BIAS_POLICIES.values():
        if p.bias_id == int(bias_id):
            return p
    raise ValueError(f"unknown bias_id={bias_id}")
