# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import pytest
import torch
import torch.nn.functional as F
from conv_common import (
    C0,
    cihw_to_ci1khkwcoci0,
    get_howo,
    nc1hwc0_to_nchw,
    nchw_to_nc1hwc0,
)

import torch_catlass
from common import only_on_3510


def _assert_close(result_nchw: torch.Tensor, ref_output: torch.Tensor) -> None:
    """Assert the NPU result matches the fp32 golden reference."""
    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(result_nchw.float(), ref_output, rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result_nchw.float() - ref_output).abs().max().item()}"
    )


def _run_ascend950_conv2d_bias_silu(N, C, H, W, OC, KH, KW, stride, padding, dilation, seed=1):
    """Run ascend950_conv2d_bias_silu on NPU and return (result_nchw, reference_nchw).

    Implementation taken from example 56_conv2d_bias_silu; prebuilt kernel
    is CatlassKernel::Ascend950Conv2dBiasSilu.
    """
    torch.manual_seed(seed)
    cout_round = (OC + C0 - 1) // C0 * C0

    fmap_cpu = torch.randn(N, C, H, W, dtype=torch.float16)
    filter_cpu = torch.randn(OC, C, KH, KW, dtype=torch.float16)
    bias_cpu = torch.randn(cout_round, dtype=torch.float16)

    fmap_nc1hwc0 = nchw_to_nc1hwc0(fmap_cpu)
    filter_ci1khkwcoci0 = cihw_to_ci1khkwcoci0(filter_cpu)

    fmap_npu = fmap_nc1hwc0.npu()
    filter_npu = filter_ci1khkwcoci0.npu()
    bias_npu = bias_cpu.npu()

    result = torch_catlass.ascend950_conv2d_bias_silu(
        fmap_npu,
        filter_npu,
        bias_npu,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    Ho, Wo = get_howo((H, W), (KH, KW), padding, dilation, stride)
    expected_size = N * (cout_round // C0) * Ho * Wo * C0

    assert result.numel() == expected_size
    assert result.dtype == torch.float16
    assert result.device.type == "npu"

    # Golden: SiLU(conv2d(fmap, filter) + bias), computed in fp32.
    # F.pad order: (padLeft, padRight, padTop, padBottom) for last 2 dims
    fmap_padded = F.pad(fmap_cpu.float(), (padding[0], padding[1], padding[2], padding[3]))
    conv_out = F.conv2d(
        fmap_padded,
        filter_cpu.float(),
        stride=stride,
        padding=0,
        dilation=dilation,
    )
    conv_out = conv_out + bias_cpu[:OC].float().view(1, -1, 1, 1)
    ref_output = conv_out * torch.sigmoid(conv_out)

    result_nchw = nc1hwc0_to_nchw(result.cpu(), N, OC, Ho, Wo)
    return result_nchw, ref_output


@only_on_3510
def test_ascend950_conv2d_bias_silu():
    N, C, H, W = 1, 16, 32, 32
    OC, KH, KW = 32, 3, 3
    stride = [1, 1]
    padding = [1, 1, 1, 1]
    dilation = [1, 1]

    result_nchw, ref_output = _run_ascend950_conv2d_bias_silu(
        N, C, H, W, OC, KH, KW, stride, padding, dilation, seed=3
    )
    _assert_close(result_nchw, ref_output)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
