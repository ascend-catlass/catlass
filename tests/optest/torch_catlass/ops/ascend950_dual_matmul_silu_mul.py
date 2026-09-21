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


def _resolve_dtype(out_dtype: str | torch.dtype | None, default: torch.dtype) -> torch.dtype:
    if out_dtype is None:
        return default
    if isinstance(out_dtype, torch.dtype):
        return out_dtype
    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    dtype = aliases.get(out_dtype.lower())
    if dtype is None:
        raise ValueError(f"{out_dtype} is not a supported DualMatmul output dtype")
    return dtype


def ascend950_dual_matmul_silu_mul(
    x: Tensor,
    b0: Tensor,
    b1: Tensor,
    outDType: str | torch.dtype | None = None,
) -> Tensor:
    """Run fused ``SiLU(x @ b0.T) * (x @ b1.T)`` on Ascend950.

    Source: example 78_ascend950_dual_matmul_silu_mul.

    Args:
        x: Contiguous input tensor with physical shape ``(M, K)``.
        b0: Contiguous input tensor with physical shape ``(N, K)``. CATLASS
            interprets it as a ColumnMajor logical ``(K, N)`` matrix.
        b1: Contiguous input tensor with the same physical shape and dtype as
            ``b0``.
        outDType: Output dtype. Accepted values are FP16/BF16 torch dtypes or
            their string aliases. Defaults to the input dtype.

    Returns:
        NPU tensor with shape ``(M, N)`` containing
        ``SiLU(x @ b0.T) * (x @ b1.T)``.
    """
    out_dtype = _resolve_dtype(outDType, x.dtype)
    return torch.ops.catlass.ascend950_dual_matmul_silu_mul(x, b0, b1, out_dtype)
