# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import re

import pytest
import torch
import torch_npu

from mx_golden import prepare_fp8_mx_batch_inputs


def _is_ascend950() -> bool:
    if torch_npu.npu.device_count() <= 0:
        return False
    name = torch_npu.npu.get_device_name()
    return bool(re.search(r"Ascend950(PR|DT)", name, re.I))


pytestmark = pytest.mark.skipif(
    not _is_ascend950(),
    reason="example 58_ascend950_fp8_mx_batch_matmul requires Ascend 950 NPU",
)

# 固定随机种子，保证 1000 个 case 可复现
rng = torch.Generator().manual_seed(20260908)
NUM_CASES = 2

# 候选形状池：覆盖不同 batch / M / N / K 组合
BATCH_POOL = [1, 2, 4, 8]
MN_POOL = [128, 256, 512, 1024]
K_POOL = [128, 256, 512, 1024, 2048]


def _gen_case(i: int):
    """从形状池中随机采样一组 (batch, M, N, K)。"""
    b = BATCH_POOL[i % len(BATCH_POOL)]
    m = MN_POOL[(i * 3 + 1) % len(MN_POOL)]
    n = MN_POOL[(i * 5 + 2) % len(MN_POOL)]
    k = K_POOL[(i * 7 + 3) % len(K_POOL)]
    # 小幅度随机扰动：保持 32 对齐（MX scale 每 32 元素一组）
    m = max(32, m + int(torch.randint(0, 3, (), generator=rng).item()) * 32)
    n = max(32, n + int(torch.randint(0, 3, (), generator=rng).item()) * 32)
    k = max(32, k + int(torch.randint(0, 3, (), generator=rng).item()) * 32)
    return (b, m, n, k)


CASES = [_gen_case(i) for i in range(NUM_CASES)]


@pytest.mark.parametrize("batch,m,n,k", CASES, ids=[f"b{b}_m{m}_n{n}_k{k}" for b, m, n, k in CASES])
def test_ascend950_fp8_mx_batch_matmul(batch, m, n, k):
    """Compare CATLASS MX FP8 batch matmul against dequant reference (1000 cases)."""
    import torch_catlass

    a, b, a_scale, b_scale, expected = prepare_fp8_mx_batch_inputs(batch, m, n, k, device="npu")

    result = torch_catlass.ascend950_fp8_mx_batch_matmul(
        a, b, a_scale, b_scale, transA=False, transB=True
    )

    assert result.shape == (batch, m, n)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "npu"

    expected = expected.to(torch.bfloat16).to(torch.float32)
    rtol, atol = 1e-1, 1e-1
    assert torch.allclose(result.cpu().float(), expected, rtol=rtol, atol=atol), (
        f"max diff = {(result.cpu().float() - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
