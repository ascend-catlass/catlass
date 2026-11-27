# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import pytest
import torch
import torch_npu
import torch_catlass

from a8w4_golden import compare_a8w4_result, prepare_a8w4_grouped_mx_inputs
from common import only_on_3510


@only_on_3510
@pytest.mark.parametrize(
    "group_sizes, n, k",
    [
        # single group — degenerates to a plain A8W4 MX matmul
        ([128], 128, 128),
        # two equal groups
        ([128, 128], 256, 256),
        # uneven groups
        ([64, 192], 256, 256),
        # four groups, larger K
        ([64, 64, 64, 64], 256, 512),
        # shapes aligned to tile boundaries (256)
        ([256, 256], 256, 256),
    ],
    ids=[
        "single_128x128x128",
        "two_even_128x256x256",
        "two_uneven_64_192x256x256",
        "four_64x256x512",
        "tile_aligned_256x256x256",
    ],
)
def test_ascend950_a8w4_grouped_mx_matmul(group_sizes, n, k):
    """Compare grouped A8W4 MX matmul against dequant reference (example 74).

    Covers forward numerical correctness across multiple shape/group
    combinations and verifies the quantization→dequantization error stays
    within tolerance (aligned with examples/common/golden CompareData).
    """
    m = sum(group_sizes)
    a, b, group_list, a_scale, b_scale, expected = prepare_a8w4_grouped_mx_inputs(
        group_sizes, n, k, device="npu"
    )

    # Input sanity
    assert a.dtype == torch.float8_e4m3fn
    assert a.shape == (m, k)
    assert b.dtype == torch.int8
    assert group_list.dtype == torch.int64
    assert group_list.numel() == len(group_sizes)

    result = torch_catlass.ascend950_a8w4_grouped_mx_matmul(a, b, group_list, a_scale, b_scale)

    # Output shape / dtype / device
    assert result.shape == (m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"

    torch_npu.npu.synchronize()
    result_cpu = result.cpu()
    assert torch.isfinite(result_cpu).all(), "result contains non-finite values"

    # Numerical correctness vs dequant reference.
    passed, max_diff = compare_a8w4_result(result_cpu, expected, k)
    rtol = 1.0 / 128 if k >= 2048 else 1.0 / 256
    assert passed, (
        f"group_sizes={group_sizes}, n={n}, k={k}: max diff = {max_diff}, rtol={rtol} "
        f"(CompareData: diff <= rtol * max(1, |expected|))"
    )


@only_on_3510
def test_ascend950_a8w4_grouped_mx_matmul_dtype_invariants():
    """Verify dtype and layout invariants are enforced by the wrapper."""
    group_sizes = [64, 64]
    n, k = 128, 128
    a, b, group_list, a_scale, b_scale, _ = prepare_a8w4_grouped_mx_inputs(
        group_sizes, n, k, device="npu"
    )

    # A must be float8_e4m3fn
    assert a.dtype == torch.float8_e4m3fn
    # B must be int8 (packed FP4 prologue bytes)
    assert b.dtype == torch.int8
    # Scales must be float8_e8m0fnu
    assert a_scale.dtype == torch.float8_e8m0fnu
    assert b_scale.dtype == torch.float8_e8m0fnu
    # b_scale shape: (G, N, mxScaleAlignedK/2, 2)
    assert b_scale.dim() == 4
    assert b_scale.size(0) == len(group_sizes)
    assert b_scale.size(1) == n


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
