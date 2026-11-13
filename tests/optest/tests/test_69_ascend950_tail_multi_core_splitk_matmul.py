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
import torch_catlass
from torch_catlass.ops.tail_multi_core_splitk_matmul import MAX_K


# 5*8 output tiles satisfy C < blocks < 1.5*C for 28, 32 and 36 AI cores.
TEST_M: int = 1280
TEST_N: int = 2048


def _is_ascend950() -> bool:
    if torch_npu.npu.device_count() <= 0:
        return False
    name = torch_npu.npu.get_device_name()
    return bool(re.search(r"Ascend950(PR|DT)", name, re.I))


pytestmark = pytest.mark.skipif(
    not _is_ascend950(),
    reason="example 69_ascend950_tail_multi_core_splitk_matmul requires Ascend 950 NPU",
)


def test_ascend950_tail_multi_core_splitk_matmul():
    m, n = TEST_M, TEST_N
    k = 2048
    a = torch.randn(m, k, dtype=torch.float32, device="npu")
    b = torch.randn(k, n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_tail_multi_core_splitk_matmul(
        a, b, "float32", False, False, False, False
    )
    expected = torch.matmul(a.cpu().double(), b.cpu().double()).float().npu()

    assert result.shape == (m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_k_limits(dtype, trans_a, trans_b, offset):
    m, n = TEST_M, TEST_N
    k = MAX_K + offset
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)
    a = torch.ones(a_shape, dtype=dtype)
    b = torch.ones(b_shape, dtype=dtype)
    if offset > 0:
        # 包装层在 CPU 上直接拒绝超限输入，不进入设备派发。
        with pytest.raises(ValueError, match=f"K <= {MAX_K}"):
            torch_catlass.ascend950_tail_multi_core_splitk_matmul(a, b, dtype, trans_a, trans_b)
        a_npu, b_npu = a.npu(), b.npu()
        torch.npu.synchronize()
        # 同步派发使原生拒绝在当前用例上报，公共封装只保留 ACL 错误。
        rejection_stream = torch_npu.npu.SyncLaunchStream()
        with torch_npu.npu.stream(rejection_stream):
            with pytest.raises(RuntimeError):
                torch.ops.catlass.ascend950_tail_multi_core_splitk_matmul(
                    a_npu, b_npu, dtype, trans_a, trans_b, False, False
                )
                rejection_stream.synchronize()
        return
    # Retain device inputs until result.cpu() completes the queued computation.
    a_npu, b_npu = a.npu(), b.npu()
    result = torch_catlass.ascend950_tail_multi_core_splitk_matmul(
        a_npu, b_npu, dtype, trans_a, trans_b
    )
    expected = torch.full((m, n), k, dtype=dtype)
    assert torch.equal(result.cpu(), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
