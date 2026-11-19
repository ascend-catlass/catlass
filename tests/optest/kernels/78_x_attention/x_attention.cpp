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

#include <acl/acl.h>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "catlass_kernel_prebuilt.h"
#include "x_attention_device.hpp"
#include "x_attention_tiling.hpp"

namespace CatlassKernel {

namespace {

void CheckAcl(aclError status, const char* operation)
{
    if (status != ACL_ERROR_NONE) {
        throw std::runtime_error(std::string(operation) + " failed with ACL error " + std::to_string(status));
    }
}

} // namespace

void PrepareXAttention(const uint32_t blockNum, XAttentionParams& params)
{
    XAttentionTiling::Context context;
    context.batch = params.batch;
    context.beamSize = params.beamSize;
    context.numHeads = params.numHeads;
    context.kvHeads = params.kvHeads;
    context.embeddingSize = params.embeddingSize;
    context.sharedKvSeqLen = params.sharedKvSeqLen;
    context.maxDecodeStep = params.maxDecodeStep;
    context.coreNum = blockNum;
    context.sharedPaged = params.sharedPaged;

    XAttentionTilingData tilingData{};
    uint64_t workspaceSize = 0;
    XAttentionTiling::GetTiling(context, tilingData, workspaceSize);
    if (params.scaleValue > 0.0f) {
        tilingData.scaleValue = params.scaleValue;
    }

    params.workspaceSize = workspaceSize;
    params.tilingHost.resize(sizeof(tilingData));
    std::memcpy(params.tilingHost.data(), &tilingData, sizeof(tilingData));
}

void XAttention(const uint32_t blockNum, aclrtStream stream, const XAttentionParams& params)
{
    if (params.workspaceAddr == nullptr || params.tilingAddr == nullptr ||
        params.tilingHost.size() != sizeof(XAttentionTilingData)) {
        throw std::invalid_argument("x_attention requires prepared workspace and tiling");
    }
    uint8_t* workspace = params.workspaceAddr;
    uint8_t* tiling = params.tilingAddr;
    CheckAcl(aclrtMemcpy(tiling, params.tilingHost.size(), params.tilingHost.data(),
        params.tilingHost.size(), ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy tiling");

    uint64_t hardwareSyncAddr = 0;
    CheckAcl(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)), "aclrtGetHardwareSyncAddr");

    uint8_t* query = params.inputAddr.at(0);
    uint8_t* sharedKey = params.inputAddr.at(1);
    uint8_t* sharedValue = params.inputAddr.at(2);
    uint8_t* unsharedKey = params.inputAddr.at(3);
    uint8_t* unsharedValue = params.inputAddr.at(4);
    uint8_t* unsharedBlockTable = params.inputAddr.at(5);
    uint8_t* sharedKvLens = params.inputAddr.at(6);
    uint8_t* decodeStep = params.inputAddr.at(7);
    uint8_t* sharedBlockTable = params.inputAddr.at(8);
    uint8_t* output = params.outputAddr.at(0);

    if (params.dataType == ACL_FLOAT16 && params.sharedPaged) {
        ::XAttention<half, true, false><<<blockNum, nullptr, stream>>>(
            hardwareSyncAddr, query, sharedKey, sharedValue, unsharedKey, unsharedValue,
            unsharedBlockTable, sharedKvLens, decodeStep, sharedBlockTable, output, workspace, tiling);
    } else if (params.dataType == ACL_FLOAT16) {
        ::XAttention<half, false, true><<<blockNum, nullptr, stream>>>(
            hardwareSyncAddr, query, sharedKey, sharedValue, unsharedKey, unsharedValue,
            unsharedBlockTable, sharedKvLens, decodeStep, sharedBlockTable, output, workspace, tiling);
    } else if (params.dataType == ACL_BF16 && params.sharedPaged) {
        ::XAttention<bfloat16_t, true, false><<<blockNum, nullptr, stream>>>(
            hardwareSyncAddr, query, sharedKey, sharedValue, unsharedKey, unsharedValue,
            unsharedBlockTable, sharedKvLens, decodeStep, sharedBlockTable, output, workspace, tiling);
    } else if (params.dataType == ACL_BF16) {
        ::XAttention<bfloat16_t, false, true><<<blockNum, nullptr, stream>>>(
            hardwareSyncAddr, query, sharedKey, sharedValue, unsharedKey, unsharedValue,
            unsharedBlockTable, sharedKvLens, decodeStep, sharedBlockTable, output, workspace, tiling);
    } else {
        throw std::runtime_error("x_attention supports float16 and bfloat16 only");
    }
}

} // namespace CatlassKernel
