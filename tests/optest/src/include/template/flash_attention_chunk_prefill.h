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

#ifndef OPTEST_FLASH_ATTENTION_CHUNK_PREFILL_H
#define OPTEST_FLASH_ATTENTION_CHUNK_PREFILL_H

#include <stdexcept>
#include <string>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_prebuilt.h"
#include "common/run_npu_func.h"
#include "../../../utils/include/torch_utils.h"

namespace CatlassKernelWrapper {

struct Ascend950FlashAttentionChunkPrefillHost {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& query,
        const at::Tensor& key,
        const at::Tensor& value,
        const at::Tensor& actual_seq_lengths,
        const at::Tensor& actual_seq_lengths_kv,
        const at::Tensor& atten_mask,
        const at::Tensor& block_table,
        const std::string& input_layout,
        int64_t num_heads,
        int64_t num_key_value_heads,
        int64_t block_size,
        int64_t num_blocks,
        const std::string& cache_layout,
        int64_t sparse_mode,
        CatlassKernel::FlashAttentionChunkPrefillParams& params)
    {
        TORCH_CHECK(input_layout == "TND", "input_layout of ascend950_flash_attention_chunk_prefill only supports TND");

        aclDataType queryDtype = TorchDtypeToAclDtype(query.scalar_type());
        aclDataType keyDtype = TorchDtypeToAclDtype(key.scalar_type());
        aclDataType valueDtype = TorchDtypeToAclDtype(value.scalar_type());
        TORCH_CHECK(
            queryDtype == keyDtype && queryDtype == valueDtype,
            "query, key and value must have the same dtype");
        TORCH_CHECK(
            queryDtype == ACL_FLOAT16 || queryDtype == ACL_BF16,
            "flash_attention_infer supports float16 and bfloat16 only");

        int64_t qNtokens = query.size(0);
        int64_t qkembeddingSize = query.size(2);
        int64_t vembeddingSize = value.size(3);
        int64_t batch = actual_seq_lengths.numel() - 1;

        int64_t qSeqlen = actual_seq_lengths.cpu().max().item<int64_t>();
        int64_t kvSeqlen = actual_seq_lengths_kv.cpu().max().item<int64_t>();

        uint32_t maskType = 0;
        if (sparse_mode == 0) {
            maskType = 0;
        } else if (sparse_mode == 1) {
            maskType = 1;
        } else {
            throw std::runtime_error("sparse_mode of flash_attention_infer should be 0 or 1");
        }

        params.inputAddr.resize(7);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(actual_seq_lengths.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(actual_seq_lengths_kv.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(query.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(key.storage().data()));
        params.inputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(value.storage().data()));
        params.inputAddr[5] = static_cast<uint8_t*>(const_cast<void*>(atten_mask.storage().data()));
        params.inputAddr[6] = static_cast<uint8_t*>(const_cast<void*>(block_table.storage().data()));

        params.qNtokens = static_cast<uint32_t>(qNtokens);
        params.batch = static_cast<uint32_t>(batch);
        params.qSeqlen = static_cast<uint32_t>(qSeqlen);
        params.kvSeqlen = static_cast<uint32_t>(kvSeqlen);
        params.numHeads = static_cast<uint32_t>(num_heads);
        params.kvHeads = static_cast<uint32_t>(num_key_value_heads);
        params.qkembeddingSize = static_cast<uint32_t>(qkembeddingSize);
        params.vembeddingSize = static_cast<uint32_t>(vembeddingSize);
        params.blockSize = static_cast<uint32_t>(block_size);
        params.numBlocks = static_cast<uint32_t>(num_blocks);
        params.cacheLayout = cache_layout;
        params.maskType = maskType;
        params.dataType = queryDtype;
    }

    static OutputType AllocOutput(CatlassKernel::FlashAttentionChunkPrefillParams& params)
    {
        OutputType output = GetOutputTensor(
            {params.qNtokens, params.numHeads, params.vembeddingSize},
            AclDtypeToTorchDtype(params.dataType));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static at::Tensor ToNzStorage(const at::Tensor& t)
    {
        int64_t n = t.size(0);
        int64_t c = t.size(1);
        int64_t s = t.size(2);
        int64_t h = t.size(3);
        at::Tensor nz = t.reshape({n, c, s, h / 16, 16}).permute({0, 1, 3, 2, 4}).contiguous().view(-1);
        at::Tensor out = at::empty_like(t);
        out.view(-1).copy_(nz);
        return out;
    }
};

struct Ascend950FlashAttentionChunkPrefillOp : Ascend950FlashAttentionChunkPrefillHost {
    static OutputType Run(
        const at::Tensor& query,
        const at::Tensor& key,
        const at::Tensor& value,
        const at::Tensor& actual_seq_lengths,
        const at::Tensor& actual_seq_lengths_kv,
        const at::Tensor& atten_mask,
        const at::Tensor& block_table,
        const std::string& input_layout,
        int64_t num_heads,
        int64_t num_key_value_heads,
        int64_t block_size,
        int64_t num_blocks,
        const std::string& cache_layout,
        int64_t sparse_mode)
    {
        CatlassKernel::FlashAttentionChunkPrefillParams params;
        at::Tensor keyIn = key;
        at::Tensor valueIn = value;
        if (cache_layout == "nz") {
            keyIn = ToNzStorage(key);
            valueIn = ToNzStorage(value);
        }
        GetKernelInfo(
            query, keyIn, valueIn, actual_seq_lengths, actual_seq_lengths_kv, atten_mask, block_table, input_layout,
            num_heads, num_key_value_heads, block_size, num_blocks, cache_layout, sparse_mode, params);
        OutputType output = AllocOutput(params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(CatlassKernel::FlashAttentionChunkPrefill, aicCoreNum, stream, params);
        return output;
    }
};
} // namespace CatlassKernelWrapper

#endif
