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

#ifndef OPTEST_BASIC_SYRK_H
#define OPTEST_BASIC_SYRK_H

#include <algorithm>
#include <cmath>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass/gemm/block/block_scheduler_syrk.h"
#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "template/syrk_layout.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

/// L1 K-tile depth paired with each square output tile edge (mirrors the JIT template).
inline uint32_t SyrkL1TileK(uint32_t tileMN)
{
    return tileMN >= 256 ? 128u : (tileMN >= 128 ? 256u : 512u);
}

/// Resolved launch shape: output tile edge, K-split factor and the FP32 workspace it needs.
struct SyrkLaunchConfig {
    uint32_t tileMN;
    uint32_t splitk;
    size_t wsElems;
};

/**
 * @brief Pick the output tile edge and the K-split factor for one problem.
 *
 * A zero tile or split request selects a supported configuration from the input
 * shape using estimated compute and workspace cost.
 */
inline SyrkLaunchConfig SyrkChooseConfig(
    int64_t tileReq, int64_t splitReq, bool allowSplit, uint32_t batch, uint32_t m, uint32_t k, uint32_t aicCoreNum)
{
    constexpr size_t kWorkspaceBudgetElems = 128ul * 1024 * 1024; // 512 MiB of FP32
    // Longest K a single, unsplit pass accumulates correctly.
    constexpr uint32_t kUnsplitKMax = 8192u;
    static const uint32_t kTileCandidates[] = {256u, 128u, 64u};
    static const uint32_t kSplitCandidates[] = {1u, 2u, 4u, 8u, 16u, 32u};

    SyrkLaunchConfig best{256u, 1u, 0u};
    double bestCost = -1.0;

    for (uint32_t tileMN : kTileCandidates) {
        if (tileReq > 0 && static_cast<uint32_t>(tileReq) != tileMN) {
            continue;
        }
        const Catlass::Gemm::Block::SyrkBlockScheduler scheduler(
            Catlass::GemmCoord{m, m, k}, Catlass::MatrixCoord{tileMN, tileMN});
        const uint32_t tasks = batch * scheduler.GetCoreLoops();
        const uint32_t l1TileK = SyrkL1TileK(tileMN);
        const uint32_t kTiles = (k + l1TileK - 1) / l1TileK;
        const double penalty = tileMN >= 256 ? 1.0 : (tileMN >= 128 ? 1.4 : 2.5);

        for (uint32_t splitCandidate : kSplitCandidates) {
            if (!allowSplit && splitCandidate != 1u) {
                continue;
            }
            if (splitReq > 0 && static_cast<uint32_t>(splitReq) != splitCandidate) {
                continue;
            }
            // An explicit request larger than the K-tile count still runs: the slice
            // length below rounds up to whole tiles, which clamps the effective split.
            if (splitCandidate > kTiles && splitReq == 0) {
                continue;
            }
            // Beyond kUnsplitKMax, split K rather than accumulating in one pass.
            if (splitCandidate == 1u && splitReq == 0 && allowSplit && k > kUnsplitKMax) {
                continue;
            }
            // Slices start on an L1 K-tile boundary, so the effective factor can be smaller.
            const uint32_t sliceLen = ((k + splitCandidate - 1) / splitCandidate + l1TileK - 1) / l1TileK * l1TileK;
            uint32_t splitk = (k + sliceLen - 1) / sliceLen;
            size_t wsElems = 0;
            if (splitk > 1) {
                wsElems = static_cast<size_t>(tasks) * splitk * tileMN * tileMN;
                while (splitk > 1 && wsElems > kWorkspaceBudgetElems) {
                    --splitk;
                    wsElems = static_cast<size_t>(tasks) * splitk * tileMN * tileMN;
                }
                if (splitk <= 1) {
                    wsElems = 0;
                }
            }

            const double slots = static_cast<double>(tasks) * splitk;
            const double rounds = std::ceil(slots / static_cast<double>(aicCoreNum));
            const double area = static_cast<double>(tileMN) * tileMN;
            double cost = rounds * area * (static_cast<double>(k) / splitk) * penalty;
            if (splitk > 1) {
                // Reduction traffic, spread over the (tile, row band) units the AIVs run.
                const double bands = static_cast<double>(tasks) * (tileMN >= 32 ? tileMN / 32 : 1);
                cost += 16.0 * static_cast<double>(tasks) * splitk * area * 64.0 / std::min(64.0, bands);
            }
            if (bestCost < 0.0 || cost < bestCost) {
                bestCost = cost;
                best = SyrkLaunchConfig{tileMN, splitk, wsElems};
            }
        }
    }
    TORCH_CHECK(
        bestCost >= 0.0, "syrk: no launch configuration matches tile=", tileReq, " splitk=", splitReq, " for m=", m,
        " k=", k,
        " (tile must be 64, 128 or 256; splitk a power of two <= 32, and > 1 only for split-K eligible shapes)");
    return best;
}

using BasicSyrkKernelFn =
    void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

using SyrkPolynomialKernelFn =
    void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::SyrkPolynomialParams&);

/**
 * @brief Torch adapter for Y = X * X^T (example 82_ascend950_basic_syrk).
 *
 * Reuses MatmulParams with m == n; inputAddr[0] (X), inputAddr[1] (Xt, normally the same
 * buffer) and outputAddr[0] (Y) are used.
 */
template <BasicSyrkKernelFn KernelFunc>
struct BasicSyrkLike {
    using OutputType = at::Tensor;

    static OutputType Run(
        const at::Tensor& matX, const at::Tensor& matXt, const c10::ScalarType& outDType, int64_t tileMN,
        int64_t splitk)
    {
        TORCH_CHECK(
            tileMN == 0 || tileMN == 64 || tileMN == 128 || tileMN == 256,
            "ascend950_basic_syrk tile edge must be 0 (auto), 64, 128 or 256");
        TORCH_CHECK(
            splitk == 0 || splitk == 1, "ascend950_basic_syrk splitk must be 0 (auto) or 1 for single-pass execution");
        TORCH_CHECK(
            matX.dim() == 2 || matX.dim() == 3, "ascend950_basic_syrk expects X with shape (M, K) or (B, M, K)");
        const SyrkOperandLayout layoutX = ClassifySyrkOperand(matX);
        const SyrkOperandLayout layoutXt = ClassifySyrkOperand(matXt);
        TORCH_CHECK(
            layoutX != SyrkOperandLayout::kUnsupported,
            "ascend950_basic_syrk expects a row-major or transposed (packed) input");
        TORCH_CHECK(layoutXt == layoutX, "ascend950_basic_syrk expects both inputs in the same layout");
        TORCH_CHECK(matXt.sizes() == matX.sizes(), "ascend950_basic_syrk expects inputs with identical shapes");
        TORCH_CHECK(matXt.scalar_type() == matX.scalar_type(), "ascend950_basic_syrk expects matching input dtypes");
        TORCH_CHECK(
            matX.scalar_type() == at::kBFloat16 || matX.scalar_type() == at::kHalf,
            "ascend950_basic_syrk currently supports bfloat16 / float16 inputs");
        TORCH_CHECK(
            outDType == at::kBFloat16 || outDType == at::kHalf || outDType == at::kFloat,
            "ascend950_basic_syrk currently supports bfloat16 / float16 / float32 output");
        // The L0C-to-GM path needs at least two 16-row blocks.
        TORCH_CHECK(matX.size(-2) >= 32, "ascend950_basic_syrk requires M >= 32, got ", matX.size(-2));

        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;

        // ELEMENT_A → X, ELEMENT_C → Y (layouts are fixed inside BlockMmadSyrkTla).
        tParams.element["A"] = TorchDtypeToAclDtype(matX.scalar_type());
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.transpose["A"] = (layoutX == SyrkOperandLayout::kTransposed);
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["C"] = false;

        params.batch = static_cast<uint32_t>(matX.dim() == 3 ? matX.size(0) : 1);
        params.m = static_cast<uint32_t>(matX.size(-2));
        params.k = static_cast<uint32_t>(matX.size(-1));
        params.n = params.m; // Y is [M, M]

        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        SyrkLaunchConfig config = SyrkChooseConfig(tileMN, splitk, false, params.batch, params.m, params.k, aicCoreNum);
        params.tileMN = config.tileMN;

        params.inputAddr.resize(2);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(matX.data_ptr()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(matXt.data_ptr()));

        // Preserve a leading batch dimension, including a batch of one.
        OutputType output =
            matX.dim() == 3 ?
                GetOutputTensor({params.batch, params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C"))) :
                GetOutputTensor({params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C")));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(output.data_ptr());

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

/**
 * @brief Torch adapter for D = alpha * X * X^T + beta * C + diagonal * I.
 *
 * inputAddr[0] (X) and inputAddr[1] (Xt, the transposed view; normally the same buffer)
 * must yield a symmetric product, and the residual C at inputAddr[2] must be symmetric
 * too: only the lower triangle is computed and the upper one is mirrored.
 */
template <SyrkPolynomialKernelFn KernelFunc>
struct SyrkPolynomialLike {
    using OutputType = at::Tensor;

    static OutputType Run(
        const at::Tensor& matX, const at::Tensor& matXt, const c10::optional<at::Tensor>& residual,
        const c10::ScalarType& outDType, double alpha, double beta, double diagonal, int64_t tileMN, int64_t splitk)
    {
        TORCH_CHECK(
            tileMN == 0 || tileMN == 64 || tileMN == 128 || tileMN == 256,
            "syrk polynomial tile edge must be 0 (auto), 64, 128 or 256");
        TORCH_CHECK(
            splitk == 0 || splitk == 1, "syrk polynomial splitk must be 0 (auto) or 1 for single-pass execution");
        TORCH_CHECK(matX.dim() == 2 || matX.dim() == 3, "syrk polynomial expects (M,K) or (B,M,K)");
        const SyrkOperandLayout layoutX = ClassifySyrkOperand(matX);
        TORCH_CHECK(
            layoutX != SyrkOperandLayout::kUnsupported,
            "syrk polynomial expects a row-major or transposed (packed) input");
        TORCH_CHECK(ClassifySyrkOperand(matXt) == layoutX, "syrk polynomial expects both inputs in the same layout");
        TORCH_CHECK(matXt.sizes() == matX.sizes(), "syrk polynomial inputs must have identical shapes");
        TORCH_CHECK(matXt.scalar_type() == matX.scalar_type(), "syrk polynomial input dtypes must match");
        TORCH_CHECK(
            matX.scalar_type() == at::kBFloat16 || matX.scalar_type() == at::kHalf,
            "syrk polynomial supports bfloat16 / float16 inputs");
        TORCH_CHECK(outDType == at::kBFloat16 || outDType == at::kHalf, "unsupported output dtype");

        int64_t batch = matX.dim() == 3 ? matX.size(0) : 1;
        int64_t m = matX.size(-2);
        TORCH_CHECK(residual.has_value() || beta == 0.0, "syrk polynomial needs a residual unless beta is 0");
        if (residual.has_value()) {
            const at::Tensor& res = *residual;
            if (matX.dim() == 3) {
                TORCH_CHECK(res.dim() == 3, "batched syrk polynomial requires batched residual");
                TORCH_CHECK(
                    res.size(0) == batch && res.size(1) == m && res.size(2) == m, "residual must have shape (B,M,M)");
            } else {
                TORCH_CHECK(res.dim() == 2 && res.size(0) == m && res.size(1) == m, "residual must have shape (M,M)");
            }
            TORCH_CHECK(res.is_contiguous(), "syrk polynomial residual must be contiguous");
            TORCH_CHECK(res.scalar_type() == outDType, "residual and output dtypes must match");
        }
        TORCH_CHECK(
            m % 32 == 0, "syrk polynomial requires M to be a multiple of 32, got ", m,
            "; the tile transpose and the SPLIT_M row split both work in 16-row blocks");

        CatlassKernel::TParams tParams;
        CatlassKernel::SyrkPolynomialParams params;
        tParams.element["A"] = TorchDtypeToAclDtype(matX.scalar_type());
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.transpose["A"] = (layoutX == SyrkOperandLayout::kTransposed);
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["C"] = false;

        params.batch = static_cast<uint32_t>(batch);
        params.m = static_cast<uint32_t>(m);
        params.n = params.m;
        params.k = static_cast<uint32_t>(matX.size(-1));
        params.alpha = static_cast<float>(alpha);
        params.beta = static_cast<float>(beta);
        params.diagonal = static_cast<float>(diagonal);
        params.inputAddr.resize(3);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(matX.data_ptr()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(matXt.data_ptr()));
        params.inputAddr[2] =
            residual.has_value() ? static_cast<uint8_t*>(const_cast<void*>(residual->data_ptr())) : nullptr;

        // Preserve a leading batch dimension, including a batch of one.
        OutputType output =
            matX.dim() == 3 ?
                GetOutputTensor({params.batch, params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C"))) :
                GetOutputTensor({params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C")));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(output.data_ptr());
        if (params.inputAddr[2] == nullptr) {
            // beta == 0 does not read the residual, but the kernel still receives an address.
            params.inputAddr[2] = params.outputAddr[0];
        }

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

        SyrkLaunchConfig config = SyrkChooseConfig(tileMN, splitk, false, params.batch, params.m, params.k, aicCoreNum);
        params.tileMN = config.tileMN;
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif // OPTEST_BASIC_SYRK_H
