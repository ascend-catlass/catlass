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

#ifndef OPTEST_SYRK_H
#define OPTEST_SYRK_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "template/syrk_layout.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using SyrkKernelFn =
    void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::SyrkParams&);

/**
 * @brief Torch adapter for out = alpha * (mat @ mat^T) + beta * input (BLAS SYRK, torch API: ascend950_syrk).
 *
 * inputAddr[0]=X, inputAddr[1]=Y, outputAddr[0]=D; m == n (square output).
 */
template <SyrkKernelFn KernelFunc>
struct SyrkLike {
    using OutputType = at::Tensor;

    static OutputType Run(
        const at::Tensor& mat, const at::Tensor& input, double alpha, double beta, const c10::ScalarType& outDType)
    {
        TORCH_CHECK(mat.dim() == 2 || mat.dim() == 3, "ascend950_syrk expects a 2-D/3-D mat of shape (M, K)/(B, M, K)");
        TORCH_CHECK(
            input.dim() == 2 || input.dim() == 3, "ascend950_syrk expects a 2-D/3-D input of shape (M, M)/(B, M, M)");
        TORCH_CHECK(mat.dim() == input.dim(), "ascend950_syrk expects mat and input to have the same rank");
        TORCH_CHECK(
            mat.scalar_type() == at::kBFloat16 || mat.scalar_type() == at::kHalf || mat.scalar_type() == at::kFloat,
            "ascend950_syrk supports bfloat16 / float16 / float32 inputs");
        TORCH_CHECK(input.scalar_type() == mat.scalar_type(), "ascend950_syrk expects input dtype to match mat");
        // Any other output dtype would make the epilogue reinterpret Y's bits.
        TORCH_CHECK(
            outDType == mat.scalar_type() || outDType == at::kFloat,
            "ascend950_syrk expects outDType to be the input dtype or float32");

        const SyrkOperandLayout layoutX = ClassifySyrkOperand(mat);
        TORCH_CHECK(
            layoutX != SyrkOperandLayout::kUnsupported,
            "ascend950_syrk expects mat to be row-major or a transposed view of a packed buffer");
        // The epilogue reads Y and writes D as row-major (M, M), so those must be packed.
        TORCH_CHECK(
            ClassifySyrkOperand(input) == SyrkOperandLayout::kRowMajor,
            "ascend950_syrk expects input to be row-major and contiguous");

        int64_t lastDim = mat.dim() - 1;
        int64_t secondLastDim = mat.dim() - 2;
        TORCH_CHECK(
            input.size(secondLastDim) == mat.size(secondLastDim) && input.size(lastDim) == mat.size(secondLastDim),
            "ascend950_syrk expects input shape (M, M) or (B, M, M) matching mat");

        CatlassKernel::TParams tParams;
        CatlassKernel::SyrkParams params;

        tParams.element["A"] = TorchDtypeToAclDtype(mat.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(input.scalar_type());
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.transpose["A"] = (layoutX == SyrkOperandLayout::kTransposed);
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["C"] = false;

        params.batch = mat.dim() == 3 ? static_cast<uint32_t>(mat.size(0)) : 1;
        params.m = static_cast<uint32_t>(mat.size(secondLastDim));
        params.k = static_cast<uint32_t>(mat.size(lastDim));
        params.n = params.m;
        params.alpha = static_cast<float>(alpha);
        params.beta = static_cast<float>(beta);

        params.inputAddr.resize(2);
        params.inputAddr[0] = static_cast<uint8_t*>(mat.data_ptr());
        params.inputAddr[1] = static_cast<uint8_t*>(input.data_ptr());

        OutputType output;
        if (params.batch > 1) {
            output = GetOutputTensor({params.batch, params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C")));
        } else if (mat.dim() == 3) {
            output = GetOutputTensor({1, params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C")));
        } else {
            output = GetOutputTensor({params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C")));
        }
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(output.data_ptr());

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif // OPTEST_SYRK_H
