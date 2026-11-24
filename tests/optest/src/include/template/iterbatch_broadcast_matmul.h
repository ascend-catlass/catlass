/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * the License root for the full text of the License.
 */

#ifndef OPTEST_ITERBATCH_BROADCAST_MATMUL_H
#define OPTEST_ITERBATCH_BROADCAST_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using IterBatchBroadcastKernelFn =
    void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::IterBatchBroadcastMatmulParams&);

/**
 * @brief Adapter for the IterBatch broadcast batched matmul (example 81).
 *
 * C[b] = cast(op(A[ba]) * op(B[bb]) + Bias(b)) with ba/bb = 0 when the operand
 * is a 2-D single matrix (broadcast over the batch dim), b otherwise.
 *
 * The bias mode is derived from the ``bias`` tensor shape:
 *   undefined            -> none,
 *   1-D (N,)             -> shared bias,
 *   2-D (batch, N)       -> per-batch bias.
 */
template <IterBatchBroadcastKernelFn KernelFunc>
struct IterBatchBroadcastMatmulLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool formatA, bool formatB,
        const c10::optional<at::Tensor>& bias,
        int64_t batchHint,
        CatlassKernel::TParams& tParams,
        CatlassKernel::IterBatchBroadcastMatmulParams& params)
    {
        const bool hasBias = bias.has_value() && bias->defined();
        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.element["BIAS"] = hasBias ? TorchDtypeToAclDtype(bias->scalar_type()) : ACL_FLOAT;
        tParams.transpose["A"] = transA;
        tParams.transpose["B"] = transB;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = formatA;
        tParams.useNz["B"] = formatB;
        tParams.useNz["C"] = false;

        // A broadcast operand is a 2-D single matrix; a per-batch operand is 3-D.
        const bool broadcastA = mat1.dim() == 2;
        const bool broadcastB = mat2.dim() == 2;

        int64_t batchA = broadcastA ? 1 : mat1.size(0);
        int64_t batchB = broadcastB ? 1 : mat2.size(0);
        int64_t batch;
        if (batchA > 1 && batchB > 1) {
            TORCH_CHECK(batchA == batchB,
                        "batch dims of mat1 (", batchA, ") and mat2 (", batchB, ") must match");
            batch = batchA;
        } else if (batchA > 1) {
            batch = batchA; // only mat2 is broadcast
        } else if (batchB > 1) {
            batch = batchB; // only mat1 is broadcast
        } else {
            // Both operands broadcast: the batch count is not derivable from the
            // inputs, so it must come from the explicit batch hint.
            TORCH_CHECK(batchHint > 0,
                        "batch count cannot be derived when both mat1 and mat2 are 2-D (broadcast); "
                        "pass a positive batch hint");
            batch = batchHint;
        }

        int64_t m, k1, k2, n;
        if (transA) {
            m = mat1.size(-1);  k1 = mat1.size(-2);
        } else {
            m = mat1.size(-2);  k1 = mat1.size(-1);
        }
        if (transB) {
            k2 = mat2.size(-1); n = mat2.size(-2);
        } else {
            k2 = mat2.size(-2); n = mat2.size(-1);
        }
        TORCH_CHECK(k1 == k2, "mat1 and mat2 shapes cannot be multiplied (",
                    m, "x", k1, " and ", k2, "x", n, ")");

        // Derive the bias mode from the bias tensor shape.
        uint32_t biasMode = 0;
        if (hasBias) {
            if (bias->dim() == 1) {
                TORCH_CHECK(bias->size(0) == n, "bias must have size N=", n, ", got ", bias->size(0));
                biasMode = 1; // shared [N]
            } else if (bias->dim() == 2) {
                TORCH_CHECK(bias->size(0) == batch && bias->size(1) == n,
                            "bias must have shape (", batch, ", ", n, ")");
                biasMode = 2; // per-batch [batch, N]
            } else {
                TORCH_CHECK(false, "bias must be 1-D (N) or 2-D (batch, N), got ", bias->dim(), "-D");
            }
        }

        params.m = static_cast<uint32_t>(m);
        params.k = static_cast<uint32_t>(k1);
        params.n = static_cast<uint32_t>(n);
        params.batch = static_cast<uint32_t>(batch);
        params.broadcastA = broadcastA;
        params.broadcastB = broadcastB;
        params.biasMode = biasMode;

        params.inputAddr.resize(hasBias ? 3 : 2);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        if (hasBias) {
            params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(bias->storage().data()));
        }
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool formatA, bool formatB,
        const c10::optional<at::Tensor>& bias,
        int64_t batchHint = -1)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::IterBatchBroadcastMatmulParams params;
        GetKernelInfo(mat1, mat2, outDType, transA, transB, formatA, formatB, bias, batchHint,
                      tParams, params);

        OutputType output = GetOutputTensor(
            {params.batch, params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C")));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif // OPTEST_ITERBATCH_BROADCAST_MATMUL_H
