# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import pytest
import torch

from common import only_on_3510


@pytest.mark.parametrize(
    "dtype,out_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
    ],
)
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1, 31),
        (64, 17, 128),
        (128, 128, 2048),
        (256, 128, 1024),
        (1024, 1024, 512),
        (512, 1024, 512),
    ],
)
@only_on_3510
def test_ascend950_dual_matmul_silu_mul(dtype, out_dtype, m, n, k):
    import torch_catlass

    x = torch.randn(m, k, dtype=dtype, device="npu")
    b0 = torch.randn(n, k, dtype=dtype, device="npu")
    b1 = torch.randn(n, k, dtype=dtype, device="npu")

    result = torch_catlass.ascend950_dual_matmul_silu_mul(x, b0, b1, out_dtype)
    x_cpu = x.cpu().float()
    b0_cpu = b0.cpu().float()
    b1_cpu = b1.cpu().float()
    expected = (
        torch.nn.functional.silu(torch.matmul(x_cpu, b0_cpu.t()))
        * torch.matmul(x_cpu, b1_cpu.t())
    )
    actual = result.cpu().float()
    normalized_error = (actual - expected).abs() / expected.abs().clamp_min(1.0)
    threshold_exponent = -10 if out_dtype == torch.float16 else -7
    threshold = 2 ** threshold_exponent
    mere = normalized_error.mean().item()
    mare = normalized_error.max().item()

    assert result.shape == (m, n)
    assert result.dtype == out_dtype
    assert result.device.type == "npu"
    assert mere < threshold and mare < 10 * threshold, (
        f"Precision check failed: MERE={mere}, MARE={mare}, threshold={threshold}"
    )


@only_on_3510
def test_ascend950_dual_matmul_silu_mul_rejects_invalid_inputs():
    import torch_catlass

    x = torch.randn(16, 32, dtype=torch.float16, device="npu")
    b0 = torch.randn(8, 32, dtype=torch.float16, device="npu")
    b1 = torch.randn(8, 32, dtype=torch.float16, device="npu")

    with pytest.raises(RuntimeError, match="same dtype"):
        torch_catlass.ascend950_dual_matmul_silu_mul(x, b0.bfloat16(), b1, torch.float16)
    with pytest.raises(RuntimeError, match="same physical shape"):
        torch_catlass.ascend950_dual_matmul_silu_mul(x, b0, b1[:7], torch.float16)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        torch_catlass.ascend950_dual_matmul_silu_mul(x[:, ::2], b0, b1, torch.float16)
    with pytest.raises(RuntimeError, match="output dtype"):
        torch_catlass.ascend950_dual_matmul_silu_mul(x, b0, b1, torch.float32)
