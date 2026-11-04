/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_BMM_BROADCAST_H
#define OPTEST_BMM_BROADCAST_H

#include <limits>

#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {
struct BatchedMatmulBroadcast {
    using OutputType = at::Tensor;
    // Match the fixed M/N tile shape used by the JIT kernel.
    static constexpr int64_t L1_TILE_M = 64;
    static constexpr int64_t L1_TILE_N = 128;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2, int64_t batch, const c10::optional<at::Tensor>& bias,
        bool transA, bool transB, bool hf32, CatlassKernel::TParams& tParams, CatlassKernel::BmmBroadcastParams& params)
    {
        TORCH_CHECK(mat1.dim() == 3 && mat2.dim() == 3, "A and B must be rank-3 physical matrices");
        TORCH_CHECK(
            mat1.device().type() == c10::DeviceType::PrivateUse1 && mat1.device() == mat2.device(),
            "A and B must be on the same NPU");
        TORCH_CHECK(mat1.is_contiguous() && mat2.is_contiguous(), "A and B must be contiguous");
        auto dtype = mat1.scalar_type();
        TORCH_CHECK(
            dtype == mat2.scalar_type() && (dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat),
            "A/B must have the same FP16/BF16/FP32 dtype");
        TORCH_CHECK(!hf32 || dtype == at::kFloat, "HF32 requires FP32 inputs");
        int64_t m = mat1.size(1);
        int64_t k = mat1.size(2);
        int64_t kB = mat2.size(1);
        int64_t n = mat2.size(2);
        if (transA) {
            m = mat1.size(2);
            k = mat1.size(1);
        }
        if (transB) {
            kB = mat2.size(2);
            n = mat2.size(1);
        }
        TORCH_CHECK(k == kB, "A/B K dimensions must match");
        constexpr int64_t maxDimension = std::numeric_limits<int32_t>::max();
        TORCH_CHECK(
            batch > 0 && batch <= std::numeric_limits<uint32_t>::max() && m > 0 && n > 0 && k > 0 &&
                m <= maxDimension && n <= maxDimension && k <= maxDimension,
            "Positive batch/M/N/K required; dimensions must fit the device fields");
        TORCH_CHECK(
            (mat1.size(0) == 1 || mat1.size(0) == batch) && (mat2.size(0) == 1 || mat2.size(0) == batch),
            "Each input batch must be 1 or the output batch");
        uint64_t tileCount = ((m + L1_TILE_M - 1) / L1_TILE_M) * ((n + L1_TILE_N - 1) / L1_TILE_N);
        TORCH_CHECK(tileCount <= std::numeric_limits<uint32_t>::max() / batch, "Output tile count exceeds UINT32_MAX");
        TORCH_CHECK(
            static_cast<uint64_t>(m) * n <= std::numeric_limits<int64_t>::max() / mat1.element_size() / batch,
            "Output byte size exceeds INT64_MAX");
        uint32_t biasMode = 0;
        if (bias.has_value()) {
            TORCH_CHECK(
                bias->device() == mat1.device() && bias->scalar_type() == dtype && bias->is_contiguous(),
                "Bias must be contiguous with the input device and dtype");
            if (bias->dim() == 1 && bias->size(0) == n) {
                biasMode = 1;
            } else if (bias->dim() == 2 && bias->size(0) == batch && bias->size(1) == n) {
                biasMode = 2;
            } else {
                TORCH_CHECK(false, "Bias must have shape [N] or [batch,N]");
            }
        }
        tParams.element["A"] = TorchDtypeToAclDtype(dtype);
        tParams.element["B"] = TorchDtypeToAclDtype(dtype);
        tParams.element["C"] = TorchDtypeToAclDtype(dtype);
        tParams.transpose["A"] = transA;
        tParams.transpose["B"] = transB;
        tParams.flag["hf32"] = hf32;
        params.batch = batch;
        params.batchA = mat1.size(0);
        params.batchB = mat2.size(0);
        params.m = m;
        params.n = n;
        params.k = k;
        params.biasMode = biasMode;
        // data_ptr respects a nonzero storage offset in an otherwise contiguous tensor.
        params.inputAddr = {static_cast<uint8_t*>(mat1.data_ptr()), static_cast<uint8_t*>(mat2.data_ptr()), nullptr};
        if (biasMode != 0) {
            params.inputAddr[2] = static_cast<uint8_t*>(bias->data_ptr());
        }
    }

    static OutputType AllocOutput(const at::Tensor& mat1, CatlassKernel::BmmBroadcastParams& params)
    {
        OutputType output = at::empty({params.batch, params.m, params.n}, mat1.options());
        params.outputAddr = {static_cast<uint8_t*>(output.data_ptr())};
        return output;
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2, int64_t batch, const c10::optional<at::Tensor>& bias,
        bool transA, bool transB, bool hf32)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::BmmBroadcastParams params;
        GetKernelInfo(mat1, mat2, batch, bias, transA, transB, hf32, tParams, params);
        c10_npu::NPUGuard guard(mat1.device());
        OutputType output = AllocOutput(mat1, params);
        auto stream = c10_npu::getCurrentNPUStream().stream(false);
        auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(CatlassKernel::Ascend950BmmBroadcast, aicCoreNum, stream, tParams, params);
        return output;
    }
};
} // namespace CatlassKernelWrapper
#endif
