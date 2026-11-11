# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Golden reference for fused FFN.
from __future__ import annotations

import torch
import torch.nn.functional as F


def _up_act(
    activation: str,
    xf: torch.Tensor,
    w1f: torch.Tensor,
    b1: torch.Tensor | None,
) -> torch.Tensor:
    act = str(activation).strip().lower()
    if act == "silu":
        return F.silu(F.linear(xf, w1f, b1))
    if act == "gelu":
        # torch.nn.GELU(approximate='none') == erf definition (device AIV path)
        return F.gelu(F.linear(xf, w1f, b1), approximate="none")
    if act == "swiglu":
        # W1 rows: [gate H | up H]; hidden = silu(gate) * up
        up = F.linear(xf, w1f, b1)  # [M, 2H]
        h_dim = up.shape[-1] // 2
        gate, up_h = up[..., :h_dim], up[..., h_dim:]
        return F.silu(gate) * up_h
    raise ValueError(f"unsupported activation {activation!r}")


def golden_ffn(
    activation: str,
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    out_dtype: torch.dtype,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    *,
    device_iso: bool = True,
) -> torch.Tensor:
    """Reference FFN for ``silu`` / ``gelu`` (erf) / ``swiglu`` (single).

    Weight layouts:
      silu/gelu: w1 [H,K], w2 [N,H]
      swiglu:    w1 [2H,K] (gate|up), w2 [N,H]

    Optional ``bias1`` / ``bias2`` are paired Cube BT bias tensors:
      silu/gelu: bias1 [H], bias2 [N]
      swiglu:    bias1 [2H] (gate|up), bias2 [N]

    Default ``device_iso=True`` matches the fused kernel data path: Up+Act in
    fp32, **cast Hidden to ``out_dtype``** (GM materialization), then Down in
    fp32 and cast output. Set ``device_iso=False`` for a full fp32 chain that
    casts only at the end.
    """
    xf = x.float()
    w1f = w1.float()
    w2f = w2.float()
    b1 = None if bias1 is None else bias1.float()
    b2 = None if bias2 is None else bias2.float()
    h = _up_act(activation, xf, w1f, b1)
    if device_iso and out_dtype in (torch.float16, torch.bfloat16):
        # Device: AIV act often in f32 then Cast+PACK → Hidden GM (B16).
        h = h.to(dtype=out_dtype).float()
    return F.linear(h, w2f, b2).to(dtype=out_dtype)


def golden_ffn_full_fp32(
    activation: str,
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    out_dtype: torch.dtype,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full fp32 reference chain; cast to ``out_dtype`` only at the end."""
    return golden_ffn(
        activation,
        x,
        w1,
        w2,
        out_dtype,
        bias1=bias1,
        bias2=bias2,
        device_iso=False,
    )
