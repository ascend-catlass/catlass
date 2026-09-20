/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_MATMUL_GATHER_SCATTER_H
#define OPTEST_MATMUL_GATHER_SCATTER_H

#include <limits>

#include <torch/torch.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
namespace CatlassKernelWrapper {

using MatmulGatherScatterKernelFn = void (*)(
    const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulGatherScatterParams&);

template <MatmulGatherScatterKernelFn KernelFunc>
struct MatmulGatherScatterLike {
    static at::Tensor Run(const at::Tensor& a, const at::Tensor& b, const at::Tensor& indices)
    {
        TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1, "a must be on NPU");
        TORCH_CHECK(b.device() == a.device(), "b must be on the same NPU as a");
        TORCH_CHECK(indices.device() == a.device(), "indices must be on the same NPU as a");
        TORCH_CHECK(a.scalar_type() == torch::kFloat16, "a must have dtype torch.float16");
        TORCH_CHECK(b.scalar_type() == torch::kFloat16, "b must have dtype torch.float16");
        TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must have dtype torch.int32");
        TORCH_CHECK(a.dim() == 2, "a must be a 2D matrix");
        TORCH_CHECK(b.dim() == 2, "b must be a 2D matrix");
        TORCH_CHECK(indices.dim() == 1, "indices must be a 1D tensor");
        TORCH_CHECK(a.is_contiguous(), "a must be contiguous RowMajor");
        TORCH_CHECK(b.is_contiguous(), "b must be contiguous RowMajor");
        TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
        TORCH_CHECK(a.size(1) == b.size(0), "a.shape[1] must equal b.shape[0]");
        TORCH_CHECK(a.size(0) > 0, "M must be positive");
        TORCH_CHECK(a.size(1) > 0, "K must be positive");
        TORCH_CHECK(b.size(1) > 0, "N must be positive");
        TORCH_CHECK(indices.numel() > 0, "indices must not be empty");
        TORCH_CHECK(indices.numel() <= a.size(0), "indices length must not exceed a.shape[0]");
        TORCH_CHECK(
            a.size(0) <= std::numeric_limits<uint32_t>::max() && b.size(0) <= std::numeric_limits<uint32_t>::max() &&
                b.size(1) <= std::numeric_limits<uint32_t>::max() &&
                indices.numel() <= std::numeric_limits<uint32_t>::max(),
            "shape exceeds uint32 range");

        const uint32_t m = static_cast<uint32_t>(a.size(0));
        const uint32_t k = static_cast<uint32_t>(a.size(1));
        const uint32_t n = static_cast<uint32_t>(b.size(1));
        const uint32_t j = static_cast<uint32_t>(indices.numel());
        CatlassKernel::TParams tParams;
        tParams.element["A"] = ACL_FLOAT16;
        tParams.element["B"] = ACL_FLOAT16;
        tParams.element["C"] = ACL_FLOAT16;
        tParams.transpose["A"] = false;
        tParams.transpose["B"] = false;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        CatlassKernel::MatmulGatherScatterParams params;
        params.m = m;
        params.n = n;
        params.k = k;
        params.j = j;
        params.inputAddr = {
            static_cast<uint8_t*>(const_cast<void*>(a.storage().data())),
            static_cast<uint8_t*>(const_cast<void*>(b.storage().data())),
            static_cast<uint8_t*>(const_cast<void*>(indices.storage().data()))};

        at::Tensor output = GetOutputTensor({m, n}, torch::kFloat16);
        params.outputAddr = {static_cast<uint8_t*>(const_cast<void*>(output.storage().data()))};

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif // OPTEST_MATMUL_GATHER_SCATTER_H
