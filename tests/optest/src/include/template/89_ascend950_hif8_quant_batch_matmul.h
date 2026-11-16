/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef OPTEST_QUANT_BATCH_MATMUL_FIXPIPE_DEQUANT_H
#define OPTEST_QUANT_BATCH_MATMUL_FIXPIPE_DEQUANT_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using QuantBatchMatmulKernelFn = void (*)(
    const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

template <QuantBatchMatmulKernelFn KernelFunc>
struct QuantBatchMatmulFixpipeDequantLike {
    using OutputType = at::Tensor;

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& perChannelScale, double scalePerTensor,
        const at::Tensor& bias, int64_t quantMode,
        const c10::ScalarType& outDType, bool transA = false, bool transB = false,
        bool useNzA = false, bool useNzB = false)
    {
        TORCH_CHECK(!transA && !transB, "transposed layouts are not supported");
        TORCH_CHECK(!useNzA && !useNzB, "NZ block layout is not supported");
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;

        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.element["QUANT_MODE"] = static_cast<aclDataType>(quantMode);
        tParams.transpose["A"] = false;
        tParams.transpose["B"] = false;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        auto mat1Sizes = mat1.sizes();
        auto mat2Sizes = mat2.sizes();
        TORCH_CHECK(mat1.dim() == 3, "mat1 must be 3-D (B, M, K)");
        TORCH_CHECK(mat2.dim() == 3, "mat2 must be 3-D (B, K, N)");
        int64_t B = mat1Sizes[0];
        int64_t M = mat1Sizes[1];
        int64_t k1 = mat1Sizes[2];
        int64_t k2 = mat2Sizes[1];
        int64_t N = mat2Sizes[2];
        TORCH_CHECK(mat2Sizes[0] == B, "mat1/mat2 batch mismatch");
        TORCH_CHECK(k1 == k2, "mat1 and mat2 k dim mismatch");
        TORCH_CHECK(bias.numel() == N, "bias must be (N,)");

        params.inputAddr.resize(5);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        // uint64 tensors cannot be created on NPU via torch factory ops; the caller
        // passes a CPU tensor and we copy it to device here.
        TORCH_CHECK(
            perChannelScale.device().type() == torch::kCPU,
            "perChannelScale must be a CPU uint64 tensor");
        at::Tensor scaleNpu = perChannelScale.numel() > 0
            ? perChannelScale.to(torch_npu::utils::get_npu_device_type())
            : at::empty({1}, perChannelScale.options().device(torch_npu::utils::get_npu_device_type()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(scaleNpu.storage().data()));

        float scalePT = static_cast<float>(scalePerTensor);
        uint8_t* scaleEncoded = nullptr;
        std::memcpy(&scaleEncoded, &scalePT, sizeof(float));
        params.inputAddr[3] = scaleEncoded;
        params.inputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(bias.storage().data()));

        params.m = static_cast<uint32_t>(M);
        params.n = static_cast<uint32_t>(N);
        params.k = static_cast<uint32_t>(k1);
        params.batch = static_cast<uint32_t>(B);

        OutputType output = GetOutputTensor(
            {B, params.m, params.n}, static_cast<torch::Dtype>(outDType));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
