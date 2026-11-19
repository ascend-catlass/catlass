/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_X_ATTENTION_H
#define OPTEST_X_ATTENTION_H

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_prebuilt.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"

namespace CatlassKernelWrapper {

struct XAttentionOp {
    using OutputType = at::Tensor;

    static void CheckTensor(const at::Tensor& tensor, const char* name)
    {
        TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    }

    static uint8_t* TensorAddress(const at::Tensor& tensor)
    {
        return static_cast<uint8_t*>(tensor.data_ptr());
    }

    static CatlassKernel::XAttentionParams CanImplement(
        const at::Tensor& query,
        const at::Tensor& shared_key_block,
        const at::Tensor& shared_value_block,
        const at::Tensor& unshared_key_block,
        const at::Tensor& unshared_value_block,
        const at::Tensor& unshared_block_table,
        const at::Tensor& shared_kv_lens,
        const at::Tensor& decode_step,
        const at::Tensor& shared_block_table,
        double scale_value)
    {
        const std::array<std::pair<const at::Tensor*, const char*>, 9> tensors{{
            {&query, "query"},
            {&shared_key_block, "shared_key_block"},
            {&shared_value_block, "shared_value_block"},
            {&unshared_key_block, "unshared_key_block"},
            {&unshared_value_block, "unshared_value_block"},
            {&unshared_block_table, "unshared_block_table"},
            {&shared_kv_lens, "shared_kv_lens"},
            {&decode_step, "decode_step"},
            {&shared_block_table, "shared_block_table"},
        }};
        for (const auto& item : tensors) {
            CheckTensor(*item.first, item.second);
            TORCH_CHECK(item.first->device() == query.device(), "all input tensors must be on the same NPU");
            for (int64_t size : item.first->sizes()) {
                TORCH_CHECK(size <= std::numeric_limits<uint32_t>::max(), "tensor dimensions must fit uint32_t");
            }
        }

        TORCH_CHECK(query.dim() == 3, "query must have shape (batch * beam_size, num_heads, 128)");
        TORCH_CHECK(query.size(2) == 128, "x_attention currently requires embedding_size == 128");
        TORCH_CHECK(
            std::isfinite(scale_value) && scale_value >= 0.0 &&
                scale_value <= std::numeric_limits<float>::max(),
            "scale_value must be finite, non-negative and representable as float32");

        aclDataType dataType = TorchDtypeToAclDtype(query.scalar_type());
        TORCH_CHECK(
            dataType == ACL_FLOAT16 || dataType == ACL_BF16,
            "x_attention supports float16 and bfloat16 only");
        for (const auto* tensor : {&shared_key_block, &shared_value_block, &unshared_key_block, &unshared_value_block}) {
            TORCH_CHECK(tensor->scalar_type() == query.scalar_type(), "all Q/K/V tensors must have the same dtype");
        }
        TORCH_CHECK(
            shared_kv_lens.scalar_type() == at::kInt && decode_step.scalar_type() == at::kInt,
            "shared_kv_lens and decode_step must use int32");
        TORCH_CHECK(
            shared_block_table.scalar_type() == at::kInt && unshared_block_table.scalar_type() == at::kInt,
            "block tables must use int32");
        TORCH_CHECK(decode_step.dim() == 1 && decode_step.numel() == 1, "decode_step must have shape (1,)");

        bool sharedPaged = shared_block_table.numel() != 0;
        bool unsharedPaged = unshared_block_table.numel() != 0;
        TORCH_CHECK(
            sharedPaged != unsharedPaged,
            "exactly one of shared_block_table and unshared_block_table must be non-empty");

        TORCH_CHECK(!sharedPaged || shared_block_table.dim() == 2, "shared_block_table must be 2D");
        TORCH_CHECK(!unsharedPaged || unshared_block_table.dim() == 1, "unshared_block_table must be 1D");
        int64_t batch = sharedPaged ? shared_block_table.size(0) : unshared_block_table.numel();
        TORCH_CHECK(batch > 0 && query.size(0) % batch == 0, "query token count must be divisible by batch");
        int64_t beamSize = query.size(0) / batch;
        TORCH_CHECK(beamSize > 0, "beam_size must be positive");
        int64_t numHeads = query.size(1);
        int64_t embeddingSize = query.size(2);
        TORCH_CHECK(shared_kv_lens.dim() == 1 && shared_kv_lens.numel() == batch, "shared_kv_lens must contain one value per batch");

        int64_t kvHeads = 0;
        int64_t sharedKvSeqLen = 0;
        int64_t maxDecodeStep = 0;
        if (sharedPaged) {
            TORCH_CHECK(shared_block_table.dim() == 2, "shared_block_table must be 2D");
            TORCH_CHECK(
                shared_key_block.dim() == 4 && shared_value_block.sizes() == shared_key_block.sizes(),
                "paged shared key/value must have identical 4D shapes");
            TORCH_CHECK(shared_key_block.size(1) == 128, "shared paged cache block size must be 128");
            kvHeads = shared_key_block.size(2);
            TORCH_CHECK(shared_key_block.size(3) == embeddingSize, "shared cache embedding size mismatch");
            TORCH_CHECK(
                shared_key_block.size(0) == batch * shared_block_table.size(1),
                "shared paged cache block count must match the block table");
            sharedKvSeqLen = shared_block_table.size(1) * 128;

            TORCH_CHECK(
                unshared_key_block.dim() == 4 && unshared_value_block.sizes() == unshared_key_block.sizes(),
                "continuous unshared key/value must have identical 4D shapes");
            TORCH_CHECK(
                unshared_key_block.size(0) == query.size(0) && unshared_key_block.size(1) == kvHeads &&
                    unshared_key_block.size(3) == embeddingSize,
                "continuous unshared cache shape must be (batch * beam_size, kv_heads, max_decode_step, 128)");
            maxDecodeStep = unshared_key_block.size(2);
        } else {
            TORCH_CHECK(unshared_block_table.dim() == 1, "unshared_block_table must be 1D");
            TORCH_CHECK(
                shared_key_block.dim() == 3 && shared_value_block.sizes() == shared_key_block.sizes(),
                "continuous shared key/value must have identical 3D shapes");
            TORCH_CHECK(shared_key_block.size(0) % batch == 0, "shared cache token count must be divisible by batch");
            kvHeads = shared_key_block.size(1);
            TORCH_CHECK(kvHeads > 0, "kv_heads must be positive");
            TORCH_CHECK(shared_key_block.size(2) == embeddingSize, "shared cache embedding size mismatch");
            sharedKvSeqLen = shared_key_block.size(0) / batch;

            TORCH_CHECK(
                unshared_key_block.dim() == 5 && unshared_value_block.sizes() == unshared_key_block.sizes(),
                "paged unshared key/value must have identical 5D shapes");
            TORCH_CHECK(
                unshared_key_block.size(1) == beamSize && unshared_key_block.size(2) == kvHeads &&
                    unshared_key_block.size(4) == embeddingSize,
                "paged unshared cache shape must be (requests, beam_size, kv_heads, max_decode_step, 128)");
            TORCH_CHECK(unshared_key_block.size(0) > 0, "request_count must be positive");
            TORCH_CHECK(
                unshared_key_block.size(0) <= std::numeric_limits<uint32_t>::max() / beamSize / kvHeads,
                "request_count * beam_size * kv_heads must fit uint32_t");
            maxDecodeStep = unshared_key_block.size(3);
        }

        TORCH_CHECK(numHeads > 0 && kvHeads > 0 && numHeads % kvHeads == 0, "num_heads must be divisible by kv_heads");
        TORCH_CHECK(numHeads / kvHeads <= 128, "GQA group size must not exceed 128");
        TORCH_CHECK(maxDecodeStep > 0 && maxDecodeStep <= 256, "max_decode_step must be in [1, 256]");
        TORCH_CHECK(sharedKvSeqLen > 0, "shared KV sequence length must be positive");

        TORCH_CHECK(
            query.size(0) <= std::numeric_limits<uint32_t>::max() / 129 / numHeads,
            "batch * beam_size * num_heads * 129 must fit uint32_t");
        TORCH_CHECK(
            sharedKvSeqLen <= std::numeric_limits<int32_t>::max() &&
                batch * ((sharedKvSeqLen + 127) / 128) <= std::numeric_limits<int32_t>::max(),
            "shared KV length and block count must fit int32_t");

        CatlassKernel::XAttentionParams params;
        params.inputAddr = {
            TensorAddress(query),
            TensorAddress(shared_key_block),
            TensorAddress(shared_value_block),
            TensorAddress(unshared_key_block),
            TensorAddress(unshared_value_block),
            TensorAddress(unshared_block_table),
            TensorAddress(shared_kv_lens),
            TensorAddress(decode_step),
            TensorAddress(shared_block_table),
        };
        params.numTokens = static_cast<uint32_t>(query.size(0));
        params.batch = static_cast<uint32_t>(batch);
        params.beamSize = static_cast<uint32_t>(beamSize);
        params.sharedKvSeqLen = static_cast<uint32_t>(sharedKvSeqLen);
        params.numHeads = static_cast<uint32_t>(numHeads);
        params.kvHeads = static_cast<uint32_t>(kvHeads);
        params.embeddingSize = static_cast<uint32_t>(embeddingSize);
        params.maxDecodeStep = static_cast<uint32_t>(maxDecodeStep);
        params.sharedPaged = sharedPaged;
        params.scaleValue = static_cast<float>(scale_value);
        params.dataType = dataType;

        return params;
    }

    static OutputType Run(
        const at::Tensor& query,
        const at::Tensor& shared_key_block,
        const at::Tensor& shared_value_block,
        const at::Tensor& unshared_key_block,
        const at::Tensor& unshared_value_block,
        const at::Tensor& unshared_block_table,
        const at::Tensor& shared_kv_lens,
        const at::Tensor& decode_step,
        const at::Tensor& shared_block_table,
        double scale_value)
    {
        auto params = CanImplement(query, shared_key_block, shared_value_block, unshared_key_block,
            unshared_value_block, unshared_block_table, shared_kv_lens, decode_step, shared_block_table, scale_value);
        TORCH_CHECK(query.device().index() == c10_npu::current_device(), "query must be on the current NPU device");
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        TORCH_CHECK(aicCoreNum >= 2, "x_attention requires at least two cube cores");

        OutputType output = GetOutputTensor(
            {query.size(0), query.size(1), query.size(2)}, query.scalar_type());
        params.outputAddr = {TensorAddress(output)};

        TORCH_CHECK(CatlassKernel::PrepareXAttention != nullptr && CatlassKernel::XAttention != nullptr,
            "x_attention kernel and workspace preparation must both be available");
        CatlassKernel::PrepareXAttention(aicCoreNum, params);
        TORCH_CHECK(params.workspaceSize > 0 &&
            params.workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "x_attention workspace size must fit int64_t");
        TORCH_CHECK(!params.tilingHost.empty(), "x_attention tiling must not be empty");

        auto npuStream = c10_npu::getCurrentNPUStream();
        auto options = query.options().dtype(at::kByte);
        auto workspace = at::empty({static_cast<int64_t>(params.workspaceSize)}, options);
        auto tiling = at::empty({static_cast<int64_t>(params.tilingHost.size())}, options);
        params.workspaceAddr = TensorAddress(workspace);
        params.tilingAddr = TensorAddress(tiling);
        // Allocator stream APIs can drain the submission queue; call them before enqueueing.
        c10_npu::NPUCachingAllocator::recordStream(workspace.storage().data_ptr(), npuStream);
        c10_npu::NPUCachingAllocator::recordStream(tiling.storage().data_ptr(), npuStream);
        aclrtStream stream = npuStream.stream(false);
        // Capture tensors through kernel submission. torch_npu 2.9's recordStream
        // tracks even the allocation stream and defers scratch reuse until its event completes.
        const std::array<at::Tensor, 11> inputs{
            query, shared_key_block, shared_value_block, unshared_key_block,
            unshared_value_block, unshared_block_table, shared_kv_lens,
            decode_step, shared_block_table, workspace, tiling};
        at_npu::native::OpCommand::RunOpApiV2(
            "CatlassKernel::XAttention", [inputs, output, aicCoreNum, stream, params]() -> aclError {
                try {
                    CatlassKernel::XAttention(aicCoreNum, stream, params);
                } catch (...) {
                    return ACL_ERROR_INTERNAL_ERROR;
                }
                return ACL_SUCCESS;
            });
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif // OPTEST_X_ATTENTION_H
