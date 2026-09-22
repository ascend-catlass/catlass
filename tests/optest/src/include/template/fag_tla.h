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

#ifndef OPTEST_FAG_TLA_H
#define OPTEST_FAG_TLA_H

#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_prebuilt.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"

namespace CatlassKernelWrapper {

// Example 87_fag_tla: FlashAttentionScoreGrad (TLA style) — flash attention backward.
// Signature: (dout, q, k, v, out, softmax_lse, cu_seq_qlen, cu_seq_kvlen) -> (dq, dk, dv).
struct FagTlaOp {
    using OutputType = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

    static void GetKernelInfo(
        const at::Tensor& dout,
        const at::Tensor& q,
        const at::Tensor& k,
        const at::Tensor& v,
        const at::Tensor& out,
        const at::Tensor& softmax_lse,
        const at::Tensor& cu_seq_qlen,
        const at::Tensor& cu_seq_kvlen,
        int64_t num_heads,
        int64_t num_key_value_heads,
        bool is_deterministic,
        CatlassKernel::FlashAttentionGradParams& params)
    {
        aclDataType dtype = TorchDtypeToAclDtype(q.scalar_type());
        TORCH_CHECK(dtype == TorchDtypeToAclDtype(k.scalar_type()), "q and k must share dtype");
        TORCH_CHECK(dtype == TorchDtypeToAclDtype(v.scalar_type()), "q and v must share dtype");
        TORCH_CHECK(dtype == TorchDtypeToAclDtype(dout.scalar_type()), "q and dout must share dtype");
        TORCH_CHECK(dtype == ACL_FLOAT16, "fag_tla only supports fp16 in the first cut");

        int64_t totalQ = q.size(0);
        int64_t headDim = q.size(2);
        int64_t vHeadDim = v.size(2);
        int64_t batch = cu_seq_qlen.numel();

        TORCH_CHECK(headDim == 128, "fag_tla only supports head_dim == 128 in the first cut");
        TORCH_CHECK(cu_seq_kvlen.numel() == batch, "cu_seq_qlen and cu_seq_kvlen must share size");
        TORCH_CHECK(softmax_lse.scalar_type() == at::kFloat, "softmax_lse must be float32");
        TORCH_CHECK(q.size(0) == dout.size(0), "dout and q must share token count");

        params.batch = static_cast<uint32_t>(batch);
        params.numHeads = static_cast<uint32_t>(num_heads);
        params.kvHeads = static_cast<uint32_t>(num_key_value_heads);
        params.qkHeadDim = static_cast<uint32_t>(headDim);
        params.vHeadDim = static_cast<uint32_t>(vHeadDim);
        params.isDeterministic = is_deterministic;
        params.dataType = dtype;

        params.inputAddr.resize(8);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(cu_seq_qlen.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(cu_seq_kvlen.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(dout.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(q.storage().data()));
        params.inputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(k.storage().data()));
        params.inputAddr[5] = static_cast<uint8_t*>(const_cast<void*>(v.storage().data()));
        params.inputAddr[6] = static_cast<uint8_t*>(const_cast<void*>(out.storage().data()));
        params.inputAddr[7] = static_cast<uint8_t*>(const_cast<void*>(softmax_lse.storage().data()));
    }

    static OutputType Run(
        const at::Tensor& dout,
        const at::Tensor& q,
        const at::Tensor& k,
        const at::Tensor& v,
        const at::Tensor& out,
        const at::Tensor& softmax_lse,
        const at::Tensor& cu_seq_qlen,
        const at::Tensor& cu_seq_kvlen,
        int64_t num_heads,
        int64_t num_key_value_heads,
        bool is_deterministic)
    {
        CatlassKernel::FlashAttentionGradParams params;
        GetKernelInfo(
            dout, q, k, v, out, softmax_lse, cu_seq_qlen, cu_seq_kvlen,
            num_heads, num_key_value_heads, is_deterministic, params);

        at::Tensor dq = GetOutputTensor(q.sizes().vec(), AclDtypeToTorchDtype(params.dataType));
        at::Tensor dk = GetOutputTensor(k.sizes().vec(), AclDtypeToTorchDtype(params.dataType));
        at::Tensor dv = GetOutputTensor(v.sizes().vec(), AclDtypeToTorchDtype(params.dataType));

        params.outputAddr.resize(3);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(dq.storage().data()));
        params.outputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(dk.storage().data()));
        params.outputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(dv.storage().data()));

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(CatlassKernel::FlashAttentionGradTLA, aicCoreNum, stream, params);

        return std::make_tuple(dq, dk, dv);
    }
};

} // namespace CatlassKernelWrapper

#endif // OPTEST_FAG_TLA_H