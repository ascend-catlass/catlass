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
    conv_can_implement,
    get_howo,
    nc1hwc0_to_nchw,
    nchw_to_nc1hwc0,
)

import torch_catlass
from common import only_on_2201


def _assert_close(result_nchw: torch.Tensor, ref_output: torch.Tensor) -> None:
    """Assert the NPU result matches the fp32 golden reference."""
    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(result_nchw.float(), ref_output, rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result_nchw.float() - ref_output).abs().max().item()}"
    )


def _run_conv2d_bias_silu(N, C, H, W, OC, KH, KW, stride, padding, dilation, seed=1):
    """Run conv2d_bias_silu (AtlasA2/2201) on NPU and return (result_nchw, reference_nchw).

    Implementation taken from example 33_conv2d_bias_silu; prebuilt kernel
    is CatlassKernel::Conv2dBiasSilu.
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

    result = torch_catlass.conv2d_bias_silu(
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


@only_on_2201
def test_conv2d_bias_silu():
    N, C, H, W = 1, 16, 8, 8
    OC, KH, KW = 32, 3, 3
    stride = [1, 1]
    padding = [1, 2, 1, 0]
    dilation = [1, 1]

    result_nchw, ref_output = _run_conv2d_bias_silu(
        N, C, H, W, OC, KH, KW, stride, padding, dilation, seed=1
    )
    _assert_close(result_nchw, ref_output)


@only_on_2201
def test_conv2d_bias_silu_stride2_cout48():
    # 2x2 kernel: with the AtlasA2 tile config (FmapL1<8,12,8>/FilterL1<96,8>),
    # a 3x3 kernel with stride=2 needs 659968B of L1 (>512K) and is rejected by
    # CanImplement; 2x2 with stride=2 fits (393216B, 75%).
    N, C, H, W = 2, 16, 16, 16
    OC, KH, KW = 48, 2, 2
    stride = [2, 2]
    padding = [1, 1, 1, 1]
    dilation = [1, 1]

    result_nchw, ref_output = _run_conv2d_bias_silu(
        N, C, H, W, OC, KH, KW, stride, padding, dilation, seed=2
    )
    _assert_close(result_nchw, ref_output)


@only_on_2201
def test_conv2d_bias_silu_exceeds_cache():
    torch.manual_seed(4)
    N, C, H, W = 1, 16, 8, 8
    OC, KH, KW = 128, 5, 5
    stride = [1, 1]
    padding = [1, 2, 1, 0]
    dilation = [1, 1]
    # AtlasA2 默认参数（L1 512K / L0C 128K）与 conv2d_bias_silu 的 TileShape 一致
    if conv_can_implement((KH, KW), dilation, stride):
        pytest.skip(
            "This case aims to test the case that the conv2d problem exceeds the on-chip buffers."
        )

    with pytest.raises(RuntimeError, match=r"Conv2d\+SiLU op cannot be implemented.*"):
        cout_round = (OC + C0 - 1) // C0 * C0
        fmap_cpu = torch.randn(N, C, H, W, dtype=torch.float16)
        filter_cpu = torch.randn(OC, C, KH, KW, dtype=torch.float16)
        bias_cpu = torch.randn(cout_round, dtype=torch.float16)

        fmap_nc1hwc0 = nchw_to_nc1hwc0(fmap_cpu)
        filter_ci1khkwcoci0 = cihw_to_ci1khkwcoci0(filter_cpu)

        fmap_npu = fmap_nc1hwc0.npu()
        filter_npu = filter_ci1khkwcoci0.npu()
        bias_npu = bias_cpu.npu()

        _ = torch_catlass.conv2d_bias_silu(
            fmap_npu,
            filter_npu,
            bias_npu,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
