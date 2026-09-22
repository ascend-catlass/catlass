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
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <acl/acl.h>
#include <tiling/platform/platform_ascendc.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "catlass_kernel_prebuilt.h"

// FAG kernel entry. The kernel translation unit is included first so that the
// catlass epilogue headers (which pull in kernel_operator.h and thereby the
// CANN SoftMaxTiling definition) are visible before the tiling translation
// unit references that type.
#include "fag_tla_kernel.cpp"
#include "fag_tla_tiling.cpp"

namespace CatlassKernel {

#define FAG_ACL_CHECK(status)                                                              \
    do {                                                                                   \
        aclError error = status;                                                           \
        if (error != ACL_ERROR_NONE) {                                                     \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;\
        }                                                                                  \
    } while (0)

namespace {

void FagImpl(const uint32_t blockNum, aclrtStream stream, const FlashAttentionGradParams &params)
{
    const int64_t batch = static_cast<int64_t>(params.batch);
    const int64_t numHeads = static_cast<int64_t>(params.numHeads);
    const int64_t kvHeads = static_cast<int64_t>(params.kvHeads);
    const int64_t qkHeadDim = static_cast<int64_t>(params.qkHeadDim);
    const int64_t vHeadDim = static_cast<int64_t>(params.vHeadDim);

    if (qkHeadDim != 128) {
        std::cerr << "[ERROR] only headDim == 128 is supported, got " << qkHeadDim << std::endl;
        return;
    }

    // ---- device tensor addresses ----
    uint8_t *cuSeqQlenDevice = nullptr;
    uint8_t *cuSeqKvlenDevice = nullptr;
    if (params.inputAddr.size() > 0) {
        cuSeqQlenDevice = params.inputAddr.at(0);
    }
    if (params.inputAddr.size() > 1) {
        cuSeqKvlenDevice = params.inputAddr.at(1);
    }
    uint8_t *doutDevice = params.inputAddr.at(2);
    uint8_t *qDevice = params.inputAddr.at(3);
    uint8_t *kDevice = params.inputAddr.at(4);
    uint8_t *vDevice = params.inputAddr.at(5);
    uint8_t *outDevice = params.inputAddr.at(6);
    uint8_t *softmaxLseDevice = params.inputAddr.at(7);

    uint8_t *dqDevice = params.outputAddr.at(0);
    uint8_t *dkDevice = params.outputAddr.at(1);
    uint8_t *dvDevice = params.outputAddr.at(2);

    // ---- D2H the prefix-sum sequence tables for host tiling ----
    uint64_t seqArraySize = static_cast<uint64_t>(batch) * sizeof(int64_t);
    uint8_t *qSeqHost = nullptr;
    uint8_t *kvSeqHost = nullptr;
    FAG_ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&qSeqHost), seqArraySize));
    FAG_ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&kvSeqHost), seqArraySize));
    FAG_ACL_CHECK(aclrtMemcpy(qSeqHost, seqArraySize, cuSeqQlenDevice, seqArraySize, ACL_MEMCPY_DEVICE_TO_HOST));
    FAG_ACL_CHECK(aclrtMemcpy(kvSeqHost, seqArraySize, cuSeqKvlenDevice, seqArraySize, ACL_MEMCPY_DEVICE_TO_HOST));

    // ---- build FAG info ----
    FAGTiling::FAGInfo fagInfo;
    fagInfo.scaleValue = 1.0f / std::sqrt(static_cast<float>(qkHeadDim));
    fagInfo.keepProb = 1.0f;
    fagInfo.maskType = 0;  // no mask (full attention)
    fagInfo.batch = batch;
    fagInfo.qSeqlen = 0;   // recomputed from seq lists in TND tiling
    fagInfo.qHeadNum = numHeads;
    fagInfo.qkHeadDim = qkHeadDim;
    fagInfo.kvSeqlen = 0;  // recomputed from seq lists in TND tiling
    fagInfo.kvHeadNum = kvHeads;
    fagInfo.vHeadDim = vHeadDim;
    fagInfo.window_size_left = INT64_MAX;
    fagInfo.window_size_right = INT64_MAX;
    fagInfo.layout = TND;
    fagInfo.qSeqlenList = reinterpret_cast<int64_t *>(qSeqHost);
    fagInfo.kvSeqlenList = reinterpret_cast<int64_t *>(kvSeqHost);
    // The kernel is compiled with IS_DTM=1 (deterministic DQKV path). The tiling
    // only reserves the DTM workspace when isDeterministic is true, so the two
    // must be kept in sync; otherwise the DTM workspace offsets computed inside
    // the kernel point past the allocated workspace.
    fagInfo.isDeterministic = true;

    // ---- compute tiling ----
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
        platform_ascendc::CoreMemType::UB, ubSize);

    uint32_t tilingSize = sizeof(FAGTilingData);
    FAGTilingData fagTilingData;
    int64_t ret = FAGTiling::GetFAGTilingParam(fagInfo, aicCoreNum, aivCoreNum, ubSize, fagTilingData);
    if (ret != 0) {
        std::cerr << "[ERROR] GetFAGTilingParam failed: " << ret << std::endl;
        aclrtFreeHost(qSeqHost);
        aclrtFreeHost(kvSeqHost);
        return;
    }

    uint8_t *tilingDevice = nullptr;
    FAG_ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&tilingDevice), tilingSize + 16, ACL_MEM_MALLOC_HUGE_FIRST));
    FAG_ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, &fagTilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *workspaceDevice = nullptr;
    FAG_ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&workspaceDevice), fagTilingData.workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));

    uint64_t hardwareSyncAddr{0};
    FAG_ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void **>(&hardwareSyncAddr)));

    constexpr uint32_t InputLayout = TND;
    constexpr uint32_t IsDtm = 1;

    // headDim == 128 -> Aligned128 template. Only fp16 supported in the first cut.
    if (params.dataType == ACL_FLOAT16) {
        FAG_BSND<0, DTemplateType::Aligned128, half, InputLayout, IsDtm><<<aicCoreNum, nullptr, stream>>>(
            hardwareSyncAddr, doutDevice, qDevice, kDevice, vDevice, outDevice, nullptr,
            nullptr, nullptr, nullptr, softmaxLseDevice, cuSeqQlenDevice, cuSeqKvlenDevice,
            dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice);
    } else {
        std::cerr << "[ERROR] only fp16 is supported in the first cut." << std::endl;
    }
    FAG_ACL_CHECK(aclrtSynchronizeStream(stream));

    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);
    aclrtFreeHost(qSeqHost);
    aclrtFreeHost(kvSeqHost);
}

} // namespace

void FlashAttentionGradTLA(uint32_t blockNum, aclrtStream stream, const FlashAttentionGradParams &params)
{
    FagImpl(blockNum, stream, params);
}

} // namespace CatlassKernel