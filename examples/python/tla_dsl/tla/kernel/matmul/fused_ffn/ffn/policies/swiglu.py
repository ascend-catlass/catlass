# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# SwiGLU geometry policy (P1 default SINGLE; DUAL is P1.b skeleton).
from __future__ import annotations

from dataclasses import dataclass

from ffn.config import SWIGLU_DUAL as _MODE_DUAL
from ffn.config import SWIGLU_SINGLE as _MODE_SINGLE


@dataclass(frozen=True)
class SwigluPolicy:
    name: str
    mode_id: int
    needs_gate_gm: bool
    implemented: bool


SWIGLU_SINGLE = SwigluPolicy(
    name="single",
    mode_id=_MODE_SINGLE,
    needs_gate_gm=False,
    implemented=True,  # S4: L0C2UB single
)
SWIGLU_DUAL = SwigluPolicy(
    name="dual",
    mode_id=_MODE_DUAL,
    needs_gate_gm=True,
    implemented=True,  # CPP-aligned fallback: RawGate + SyncAll + SwigluUp
)

SWIGLU_POLICIES: dict[str, SwigluPolicy] = {
    "single": SWIGLU_SINGLE,
    "dual": SWIGLU_DUAL,
}
