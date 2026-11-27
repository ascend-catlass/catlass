/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef OPTEST_A8W4_GROUPED_MX_MATMUL_H
#define OPTEST_A8W4_GROUPED_MX_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "mx_matmul.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using A8W4GroupedKernelFn = void (*)(
    const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::GroupedMatmulParams&);

/**
 * @brief Adapter for example 74_ascend950_weight_quant_a8w4_grouped_mx_matmul.
 *
 * Grouped MX A8W4 matmul: C = (MxScaleA * A_fp8) @ (MxScaleB * B_fp4), sliced on M.
 * A is shared across groups (float8_e4m3fn, RowMajor M×K). B is the packed FP4
 * prologue (int8 bytes, Weight4BitnZ layout, per group). Scales are float8_e8m0fnu.
 * Output is FP16 (M×N), matching example 74; accumulation remains FP32.
 */
template <A8W4GroupedKernelFn KernelFunc>
struct A8W4GroupedMxMatmulLike {
    using OutputType = at::Tensor;

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& groupList,
        const at::Tensor& mx_scale_a, const at::Tensor& mx_scale_b)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::GroupedMatmulParams params;

        // ── Validation ──
        CheckNpuTensor(mat1, "mat1");
        CheckNpuTensor(mat2, "mat2");
        CheckNpuTensor(groupList, "groupList");
        CheckNpuTensor(mx_scale_a, "mx_scale_a");
        CheckNpuTensor(mx_scale_b, "mx_scale_b");
        CheckSameDevice(mat1, "mat1", mat2, "mat2");
        CheckSameDevice(mat1, "mat1", groupList, "groupList");
        CheckSameDevice(mat1, "mat1", mx_scale_a, "mx_scale_a");
        CheckSameDevice(mat1, "mat1", mx_scale_b, "mx_scale_b");
        CheckMxScaleDType(mx_scale_a, "mx_scale_a");
        CheckMxScaleDType(mx_scale_b, "mx_scale_b");

        TORCH_CHECK(
            mat1.scalar_type() == torch::kFloat8_e4m3fn,
            "mat1 must have dtype torch.float8_e4m3fn, got ", mat1.scalar_type());
        TORCH_CHECK(mat1.dim() == 2, "mat1 must be 2-D with shape (M, K)");
        TORCH_CHECK(
            mat2.scalar_type() == torch::kInt8,
            "mat2 must be int8 packed FP4 prologue bytes (Weight4BitnZ layout), got ", mat2.scalar_type());
        TORCH_CHECK(
            groupList.scalar_type() == torch::kInt64,
            "groupList must be int64, got ", groupList.scalar_type());
        TORCH_CHECK(mat1.is_contiguous(), "mat1 must be contiguous");
        TORCH_CHECK(mat2.is_contiguous(), "mat2 must be contiguous");
        TORCH_CHECK(mx_scale_a.is_contiguous(), "mx_scale_a must be contiguous");
        TORCH_CHECK(mx_scale_b.is_contiguous(), "mx_scale_b must be contiguous");

        const int64_t m = mat1.size(0);
        const int64_t k = mat1.size(1);
        const uint32_t g = static_cast<uint32_t>(groupList.numel());

        // N is inferred from mx_scale_b: (G, N, mxScaleAlignedK/2, 2) for the grouped case.
        int64_t n = 0;
        if (mx_scale_b.dim() == 4) {
            TORCH_CHECK(
                mx_scale_b.size(0) == static_cast<int64_t>(g),
                "mx_scale_b dim 0 must equal group count, got ", mx_scale_b.size(0), " expected ", g);
            n = mx_scale_b.size(1);
        } else if (mx_scale_b.dim() == 3) {
            // Single-group fallback: (N, mxScaleAlignedK/2, 2)
            n = mx_scale_b.size(0);
        } else {
            TORCH_CHECK(false, "mx_scale_b must be 4-D (G, N, mxScaleAlignedK/2, 2) or 3-D (N, mxScaleAlignedK/2, 2)");
        }

        // Packed FP4 bytes for Weight4BitnZ: K padded to 32, N padded to 16, fractal (32, 16).
        const int64_t kPadded = ((k + 31) / 32) * 32;
        const int64_t nPadded = ((n + 15) / 16) * 16;
        const int64_t packedBytesPerGroup = kPadded * nPadded / 2;  // 2 FP4 nibbles per byte
        const int64_t expectedBytes = packedBytesPerGroup * static_cast<int64_t>(g);
        TORCH_CHECK(
            mat2.numel() >= expectedBytes,
            "mat2 packed bytes insufficient: need at least ", expectedBytes,
            " (g=", g, ", kPadded=", kPadded, ", nPadded=", nPadded, "), got ", mat2.numel());

        // MX scale size validation.
        uint32_t mxScaleK = static_cast<uint32_t>((k + 31) / 32);          // CeilDiv<32>(k)
        uint32_t mxScaleAlignedK = ((mxScaleK + 1) / 2) * 2;               // RoundUp<2>(mxScaleK)
        const int64_t scaleANumel = m * mxScaleAlignedK;
        const int64_t scaleBNumel = static_cast<int64_t>(g) * n * mxScaleAlignedK;
        TORCH_CHECK(
            mx_scale_a.numel() == scaleANumel,
            "mx_scale_a must have ", scaleANumel, " elements (m=", m, ", mxScaleAlignedK=", mxScaleAlignedK,
            "), got ", mx_scale_a.numel());
        TORCH_CHECK(
            mx_scale_b.numel() == scaleBNumel,
            "mx_scale_b must have ", scaleBNumel, " elements (g=", g, ", n=", n, ", mxScaleAlignedK=", mxScaleAlignedK,
            "), got ", mx_scale_b.numel());

        // ── TParams (compile-time JIT parameters) ──
        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = test_utils::TypeCast<std::string, aclDataType>("float4_e2m1fn_x2");
        tParams.element["C"] = ACL_FLOAT16;
        tParams.element["MX_SCALE"] = test_utils::TypeCast<std::string, aclDataType>("float8_e8m0fnu");
        tParams.transpose["A"] = false;
        tParams.transpose["B"] = true;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = true;   // Weight4BitnZ prologue layout
        tParams.useNz["C"] = false;
        tParams.l1TileShape = {256, 256, 256};
        tParams.l0TileShape = {256, 256, 128};
        tParams.swizzle = {3, 0, 0};

        // ── Params (runtime) ──
        params.m = static_cast<uint32_t>(m);
        params.n = static_cast<uint32_t>(n);
        params.k = static_cast<uint32_t>(k);
        params.batch = g;
        params.sliceMode = CatlassKernel::GroupedMatmulParams::SliceMode::M;
        params.inputAddr.resize(5);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(groupList.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_a.storage().data()));
        params.inputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_b.storage().data()));

        OutputType output = GetOutputTensor({params.m, params.n}, torch::kFloat16);
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
