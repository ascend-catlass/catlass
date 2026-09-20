# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

import torch
from torch import Tensor


def ascend950_matmul_gather_scatter(a: Tensor, b: Tensor, indices: Tensor) -> Tensor:
    """Compute ``D[indices, :] = A[indices, :] @ B`` in one Ascend 950 kernel.

    Source: experimental/matmul/ascend950_matmul_gather_scatter.

    ``a`` and ``b`` must be contiguous FP16 NPU matrices with shapes ``(M, K)``
    and ``(K, N)``. ``indices`` must be a contiguous INT32 NPU vector of length
    ``J``. The caller guarantees that indices are in range and contain no
    duplicates. Rows not selected by ``indices`` are returned as exact zeros.
    """
    return torch.ops.catlass.ascend950_matmul_gather_scatter(a, b, indices)
