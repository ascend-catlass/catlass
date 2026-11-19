# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import torch
from torch import Tensor
from typing import List


def ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant(
    mat1: Tensor,
    mat2: Tensor,
    mx_scale_a: Tensor,
    mx_scale_b: Tensor,
    group_list: Tensor,
    transB: bool = True,
) -> List[Tensor]:
    """Run CATLASS Ascend950 Grouped MX FP8 matmul + SwiGLU + MX quant on NPU.

    Source: example 65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant.

    Computes grouped MX FP8 matmul with slice-M, applies SwiGLU activation,
    then performs MX FP8 online quantization. Outputs quantized data Q and
    quantization scale Q_scale.

    Pipeline per group g:
        C[g] = (MxScaleA * A[g]) @ (MxScaleB * B[g])   # (m_g, N)
        act = C[:, :N/2], gate = C[:, N/2:]
        Y = swish(act) * gate                             # (m_g, N/2)
        Q, Q_scale = MX_FP8_quant(Y, block_size=32)

    Args:
        mat1: Left input (float8_e4m3fn), shape ``(M, K)``.
        mat2: Right input (float8_e4m3fn), shape ``(G, N, K)`` when ``transB=True``.
        mx_scale_a: MX scale for A (float8_e8m0fnu).
        mx_scale_b: MX scale for B (float8_e8m0fnu).
        group_list: Non-cumulative group sizes (int64), shape ``(G,)``.
        transB: Whether to read ``mat2`` as transposed (default matches example).

    Returns:
        List of two tensors:
        - Q: float8_e4m3fn output, shape ``(M, N/2)``.
        - Q_scale: float8_e8m0fnu output scale, shape ``(M, ceil(N/2/32))``.
    """
    return torch.ops.catlass.ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant(
        mat1, mat2, mx_scale_a, mx_scale_b, group_list, transB
    )
