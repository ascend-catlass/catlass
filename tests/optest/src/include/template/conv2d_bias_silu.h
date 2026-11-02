/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_CONV2D_BIAS_SILU_H
#define OPTEST_CONV2D_BIAS_SILU_H

#include <stdexcept>
#include <string>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_prebuilt.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"

namespace CatlassKernelWrapper {

// AtlasA2 (2201) 专用 Conv2D + Bias + SiLU 融合算子。
// 实现取自 example 33_conv2d_bias_silu ；prebuilt kernel 由
// kernels/33_basic_conv2d/conv2d_bias_silu.cpp 提供
// (CatlassKernel::Conv2dBiasSilu)。
struct Conv2dBiasSiluOp {
    using OutputType = at::Tensor;

    static OutputType Run(
        const at::Tensor& fmap, const at::Tensor& filter, const at::Tensor& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation)
    {
        CatlassKernel::ConvParams params;

        aclDataType dtype = TorchDtypeToAclDtype(fmap.scalar_type());
        TORCH_CHECK(dtype == TorchDtypeToAclDtype(filter.scalar_type()), "fmap and filter must have the same dtype");
        TORCH_CHECK(dtype == TorchDtypeToAclDtype(bias.scalar_type()), "fmap and bias must have the same dtype");
        TORCH_CHECK(dtype == ACL_FLOAT16, "conv2d_bias_silu supports float16 only (in current prebuilt mode)");

        TORCH_CHECK(fmap.dim() == 5, "fmap must be 5-D NC1HWC0 (N, C1, H, W, C0)");
        TORCH_CHECK(filter.dim() == 5, "filter must be 5-D CI1KHKWCOCI0 (Cin1, KH, KW, OC, C0)");
        TORCH_CHECK(bias.dim() == 1, "bias must be 1-D with CoutRound elements");

        int64_t N = fmap.size(0);
        int64_t C1 = fmap.size(1);
        int64_t H = fmap.size(2);
        int64_t W = fmap.size(3);
        int64_t C0 = fmap.size(4);

        int64_t Cin1 = filter.size(0);
        int64_t KH = filter.size(1);
        int64_t KW = filter.size(2);
        int64_t OC = filter.size(3);
        int64_t filterC0 = filter.size(4);

        int64_t coutRound = (OC + C0 - 1) / C0 * C0;

        TORCH_CHECK(C0 == 16, "fmap C0 must be 16");
        TORCH_CHECK(C0 == filterC0, "fmap C0 must match filter C0");
        TORCH_CHECK(C1 == Cin1, "fmap C1 must match filter Cin1");
        TORCH_CHECK(bias.numel() == coutRound, "bias must have CoutRound elements");

        int64_t C = C1 * C0;

        params.inputDataType = dtype;
        params.outputDataType = dtype;

        params.fmapRelated = {
            static_cast<uint32_t>(N), static_cast<uint32_t>(H),  static_cast<uint32_t>(W),
            static_cast<uint32_t>(C), static_cast<uint32_t>(OC),
        };

        params.filterRelated = {
            static_cast<uint32_t>(KH),
            static_cast<uint32_t>(KW),
        };

        params.strideList = {
            static_cast<uint32_t>(stride.at(0)),
            static_cast<uint32_t>(stride.at(1)),
        };

        params.padList = {
            static_cast<uint32_t>(padding.at(0)),
            static_cast<uint32_t>(padding.at(1)),
            static_cast<uint32_t>(padding.at(2)),
            static_cast<uint32_t>(padding.at(3)),
        };

        params.dilationList = {
            static_cast<uint32_t>(dilation.at(0)),
            static_cast<uint32_t>(dilation.at(1)),
        };

        // data_ptr() 返回 tensor 实际起始地址
        // contiguous() 保证输入内存连续
        at::Tensor fmapContiguous = fmap.contiguous();
        at::Tensor filterContiguous = filter.contiguous();
        at::Tensor biasContiguous = bias.contiguous();

        params.inputAddr.resize(3);
        params.inputAddr[0] = static_cast<uint8_t*>(fmapContiguous.data_ptr());
        params.inputAddr[1] = static_cast<uint8_t*>(filterContiguous.data_ptr());
        params.inputAddr[2] = static_cast<uint8_t*>(biasContiguous.data_ptr());

        uint32_t ho =
            (H + params.padList[2] + params.padList[3] - static_cast<uint32_t>(dilation.at(0)) * (KH - 1) - 1) /
                params.strideList[0] +
            1;
        uint32_t wo =
            (W + params.padList[0] + params.padList[1] - static_cast<uint32_t>(dilation.at(1)) * (KW - 1) - 1) /
                params.strideList[1] +
            1;
        int64_t outputElements = static_cast<int64_t>(N) * (coutRound / C0) * ho * wo * C0;

        OutputType output = GetOutputTensor({outputElements}, AclDtypeToTorchDtype(dtype));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        CatlassKernel::Conv2dBiasSilu(aicCoreNum, stream, params);

        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
