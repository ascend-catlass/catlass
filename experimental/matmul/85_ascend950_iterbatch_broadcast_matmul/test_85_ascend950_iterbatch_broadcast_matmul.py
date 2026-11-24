# This program is free software, you can redistribute it and/or modify
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License root for the full text of the License.

import pytest
import torch
import torch_npu
import torch_catlass

from common import only_on_3510


@only_on_3510
def test_iterbatch_broadcast_matmul_bcast_b():
    """Broadcast B: A is per-batch, B is a single matrix shared by all batches."""
    B, m, n, k = 8, 128, 128, 64
    a = torch.randn(B, m, k, dtype=torch.float32, device="npu")
    b = torch.randn(k, n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float32")
    expected = torch.matmul(a, b)

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@only_on_3510
def test_iterbatch_broadcast_matmul_bcast_b_shared_bias():
    """Broadcast B with a bias vector of shape (N,) shared by all batches."""
    B, m, n, k = 8, 128, 128, 64
    a = torch.randn(B, m, k, dtype=torch.float32, device="npu")
    b = torch.randn(k, n, dtype=torch.float32, device="npu")
    bias = torch.randn(n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float32", bias=bias)
    expected = torch.matmul(a, b) + bias

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@only_on_3510
def test_iterbatch_broadcast_matmul_bcast_b_per_batch_bias():
    """Broadcast B with a per-batch bias of shape (B, N)."""
    B, m, n, k = 8, 128, 128, 64
    a = torch.randn(B, m, k, dtype=torch.float32, device="npu")
    b = torch.randn(k, n, dtype=torch.float32, device="npu")
    bias = torch.randn(B, n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float32", bias=bias)
    expected = torch.matmul(a, b) + bias.unsqueeze(1)

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@only_on_3510
def test_iterbatch_broadcast_matmul_bcast_a_fp16():
    """Broadcast A with fp16 inputs: A is a single matrix, B is per-batch."""
    B, m, n, k = 6, 128, 64, 128
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(B, k, n, dtype=torch.float16, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float16")
    expected = (a.float() @ b.float()).half()

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@only_on_3510
def test_iterbatch_broadcast_matmul_no_broadcast():
    """No broadcast: both operands are per-batch, equivalent to torch.bmm."""
    B, m, n, k = 5, 64, 128, 64
    a = torch.randn(B, m, k, dtype=torch.float32, device="npu")
    b = torch.randn(B, k, n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float32")
    expected = torch.bmm(a, b)

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@only_on_3510
def test_iterbatch_broadcast_matmul_bcast_both():
    """Both operands broadcast: every output batch equals A @ B."""
    B, m, n, k = 8, 128, 128, 64
    a = torch.randn(m, k, dtype=torch.float32, device="npu")
    b = torch.randn(k, n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float32", batch=B)
    expected = torch.matmul(a, b).expand(B, m, n)

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@only_on_3510
def test_iterbatch_broadcast_matmul_trans_a():
    """transA: A is passed as RowMajor (B, K, M), the kernel sees ColumnMajor (M, K), i.e. A^T."""
    B, m, n, k = 6, 128, 64, 64
    a = torch.randn(B, k, m, dtype=torch.float32, device="npu")
    b = torch.randn(k, n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float32", transA=True)
    expected = torch.matmul(a.transpose(-2, -1), b)

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@only_on_3510
def test_iterbatch_broadcast_matmul_trans_b():
    """transB: B is passed as RowMajor (N, K), the kernel sees ColumnMajor (K, N), i.e. B^T."""
    B, m, n, k = 6, 64, 128, 64
    a = torch.randn(B, m, k, dtype=torch.float32, device="npu")
    b = torch.randn(n, k, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float32", transB=True)
    expected = torch.matmul(a, b.transpose(-2, -1))

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@only_on_3510
def test_iterbatch_broadcast_matmul_bcast_both_trans_both():
    """Double broadcast combined with double transpose: A^T @ B^T replicated over batches."""
    B, m, n, k = 4, 128, 128, 64
    a = torch.randn(k, m, dtype=torch.float32, device="npu")
    b = torch.randn(n, k, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(
        a, b, "float32", transA=True, transB=True, batch=B)
    expected = torch.matmul(a.transpose(-2, -1), b.transpose(-2, -1)).expand(B, m, n)

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@only_on_3510
def test_iterbatch_broadcast_matmul_large_batch_group_prefetch():
    """B=128, both operands per-batch: groupCount (43) exceeds the 32 AIC cores, so the
    grid-stride loop runs a second round and prefetches the next group's A/B into L1
    (nextGroupIdx = groupIdx + blockNum path). Capacity-bound auto plan picks B1=B0=3
    (per-batch L1 slots leave room for 3 batches), and 128 = 42*3 + 2, so the last L1
    group holds 2 batches and its last L0 sub-group is also a remainder (2 < 3)."""
    B, m, n, k = 128, 128, 128, 64
    a = torch.randn(B, m, k, dtype=torch.float32, device="npu")
    b = torch.randn(B, k, n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float32")
    expected = torch.bmm(a, b)

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@only_on_3510
def test_iterbatch_broadcast_matmul_large_batch_per_batch_bias():
    """B=128 with per-batch bias: same prefetch/remainder coverage as the no-bias large
    batch case (B1=B0=3, groupCount=43), plus the next group's bias rows are prefetched
    (nextTensorBias) and the bias row index must stay correct inside the 2-batch
    remainder group."""
    B, m, n, k = 128, 128, 128, 64
    a = torch.randn(B, m, k, dtype=torch.float32, device="npu")
    b = torch.randn(B, k, n, dtype=torch.float32, device="npu")
    bias = torch.randn(B, n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float32", bias=bias)
    expected = torch.bmm(a, b) + bias.unsqueeze(1)

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@only_on_3510
def test_iterbatch_broadcast_matmul_bcast_b_batch64():
    """B=64, A per-batch, B broadcast: the core-count bound lifts auto B1 to 2, so each
    L1 group physically holds one broadcast B and two per-batch A matrices. 64 divides
    evenly by B1=B0=2, giving the exact-division counterpart to the remainder cases."""
    B, m, n, k = 64, 128, 128, 64
    a = torch.randn(B, m, k, dtype=torch.float32, device="npu")
    b = torch.randn(k, n, dtype=torch.float32, device="npu")
    bias = torch.randn(n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float32", bias=bias)
    expected = torch.matmul(a, b) + bias

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@only_on_3510
def test_iterbatch_broadcast_matmul_k_tail():
    """k=40 is not a multiple of the fp32 L0 tile K (L0_TILE_K=16), so the K loop runs
    three bricks with a partial last brick (kActual = 8). The broadcast B is also
    ragged on its K rows in GM (40 rounds up to 48 rows in the L1 zN slot)."""
    B, m, n, k = 64, 64, 64, 40
    a = torch.randn(B, m, k, dtype=torch.float32, device="npu")
    b = torch.randn(k, n, dtype=torch.float32, device="npu")
    bias = torch.randn(n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float32", bias=bias)
    expected = torch.matmul(a, b) + bias

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@only_on_3510
def test_iterbatch_broadcast_matmul_unaligned_mn():
    """m=100 and n=52 are not 16-aligned: the L1 fractal slots round A up to 112 rows and
    B up to 56 columns, and the fixpipe copy-out must mask the padded edges back to the
    exact (100, 52) output shape."""
    B, m, n, k = 64, 100, 52, 64
    a = torch.randn(B, m, k, dtype=torch.float32, device="npu")
    b = torch.randn(B, k, n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float32")
    expected = torch.bmm(a, b)

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@only_on_3510
def test_iterbatch_broadcast_matmul_large_batch_fp16():
    """fp16 counterpart of the large batch case: with the fp16 tile config
    (L1_TILE_K=128, L0_TILE_K=32) the capacity bound still yields B1=3 and
    groupCount=43 > 32, covering the prefetch and remainder paths under a
    different L0 brick geometry."""
    B, m, n, k = 128, 128, 128, 128
    a = torch.randn(B, m, k, dtype=torch.float16, device="npu")
    b = torch.randn(B, k, n, dtype=torch.float16, device="npu")

    result = torch_catlass.ascend950_iterbatch_broadcast_matmul(a, b, "float16")
    expected = torch.bmm(a, b)

    assert result.shape == (B, m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
