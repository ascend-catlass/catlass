# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from collections.abc import Iterable

import torch
from torch import Tensor


def ascend950_conv2d_bias_silu(
    fmap: Tensor,
    filter: Tensor,
    bias: Tensor,
    stride: Iterable[int] = (1, 1),
    padding: Iterable[int] = (0, 0, 0, 0),
    dilation: Iterable[int] = (1, 1),
) -> Tensor:
    """Run CATLASS Ascend950 Conv2D + bias + SiLU fusion on NPU tensors.

    Source: example 56_ascend950_basic_conv2d_tla(conv2d_bias_silu).

    Args:
        fmap: Input feature map tensor in NC1HWC0 layout ``(N, C1, H, W, C0)``.
        filter: Filter tensor in CI1KHKWCOCI0 layout ``(Cin1, KH, KW, OC, C0)``.
        bias: Bias tensor with ``CoutRound`` (= ceil(OC / 16) * 16) elements.
        stride: Convolution strides ``[H, W]``.
        padding: Convolution padding ``[left, right, top, bottom]``.
        dilation: Convolution dilation ``[H, W]``.

    Returns:
        Output tensor in flat NC1HWC0 layout, SiLU(conv2d(fmap, filter) + bias).
    """
    return torch.ops.catlass.ascend950_conv2d_bias_silu(
        fmap,
        filter,
        bias,
        stride,
        padding,
        dilation,
    )
