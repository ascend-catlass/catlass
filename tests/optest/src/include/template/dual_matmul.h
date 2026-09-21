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

#ifndef OPTEST_DUAL_MATMUL_H
#define OPTEST_DUAL_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using DualMatmulKernelFn =
    void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

template <DualMatmulKernelFn KernelFunc>
struct DualMatmulSiluMulLike {
    using OutputType = at::Tensor;

    static void CheckTensor(const at::Tensor& tensor, const char* name)
    {
        TORCH_CHECK(tensor.dim() == 2, name, " must be 2-D");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(tensor.storage_offset() == 0, name, " storage_offset must be zero");
        TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    }

    static void GetKernelInfo(
        const at::Tensor& x, const at::Tensor& b0, const at::Tensor& b1, const c10::ScalarType& outDType,
        CatlassKernel::TParams& tParams, CatlassKernel::MatmulParams& params)
    {
        CheckTensor(x, "x");
        CheckTensor(b0, "b0");
        CheckTensor(b1, "b1");
        TORCH_CHECK(b0.device() == x.device(), "x and b0 must be on the same NPU device");
        TORCH_CHECK(b1.device() == x.device(), "x and b1 must be on the same NPU device");

        TORCH_CHECK(
            x.scalar_type() == b0.scalar_type() && x.scalar_type() == b1.scalar_type(),
            "x, b0 and b1 must have the same dtype");
        TORCH_CHECK(
            x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16,
            "DualMatmul supports float16 and bfloat16 inputs");
        TORCH_CHECK(
            outDType == torch::kFloat16 || outDType == torch::kBFloat16,
            "DualMatmul output dtype must be float16 or bfloat16");

        int64_t m = x.size(0);
        int64_t k = x.size(1);
        int64_t n = b0.size(0);
        TORCH_CHECK(m > 0 && n > 0 && k > 0, "m, n and k must be positive");
        TORCH_CHECK(
            b0.size(1) == k, "b0 must have physical shape (N, K), got (", b0.size(0), ", ", b0.size(1), ") for K=", k);
        TORCH_CHECK(b1.size(0) == n && b1.size(1) == k, "b1 must have the same physical shape as b0");

        tParams.element["A"] = TorchDtypeToAclDtype(x.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(b0.scalar_type());
        tParams.element["C"] = ACL_FLOAT;
        tParams.element["D"] = TorchDtypeToAclDtype(outDType);
        tParams.transpose["A"] = false;
        tParams.transpose["B"] = true;
        tParams.transpose["C"] = false;
        tParams.transpose["D"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;
        tParams.useNz["D"] = false;

        params.inputAddr.resize(3);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(x.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(b0.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(b1.storage().data()));
        params.m = static_cast<uint32_t>(m);
        params.n = static_cast<uint32_t>(n);
        params.k = static_cast<uint32_t>(k);
    }

    static OutputType AllocOutput(const CatlassKernel::TParams& tParams, CatlassKernel::MatmulParams& params)
    {
        OutputType output = GetOutputTensor({params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("D")));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static OutputType Run(
        const at::Tensor& x, const at::Tensor& b0, const at::Tensor& b1, const c10::ScalarType& outDType)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;
        GetKernelInfo(x, b0, b1, outDType, tParams, params);
        OutputType output = AllocOutput(tParams, params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif // OPTEST_DUAL_MATMUL_H
