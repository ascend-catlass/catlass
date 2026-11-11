/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

// Helper methods to check for errors
#include <iostream>
#include <cstdio>
#include "fai_kernel.cpp"
#include "fai_tiling.cpp"
#include "golden.hpp"
#include "helper.hpp"

using namespace std;

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER =
        "Usage: fai batch qSeqlen kvSeqlen numHeads kvHeads embeddingSize isVariedLen maskType [--dtype DTYPE "
        "--datapath DATA_PATH --device DEVICE_ID]\n";
    static constexpr auto MIN_ARGS = 7;

    // Define default value.
    uint32_t batch{0};
    uint32_t qSeqlen{0};
    uint32_t kvSeqlen{0};
    uint32_t numHeads{0};
    uint32_t kvHeads{0};
    uint32_t embeddingSize{0};
    uint32_t isVariedLen{0};
    uint32_t maskType{0};
    uint32_t deviceId{0};
    uint32_t blockSize{128};
    string dataType = "half";
    string dataPath = "../../examples/40_flash_attention_infer_tla/data";

    Options() = default;

    // Define function to parse the command-line arguments.
    int Parse(int argc, const char** argv)
    {
        // The number of arguments must >= 7.
        if (argc < MIN_ARGS) {
            printf(HELPER);
            return -1;
        }

        // Allocate arguments to parameters.
        uint32_t argIndex = 1;
        batch = atoi(argv[argIndex++]);
        qSeqlen = atoi(argv[argIndex++]);
        kvSeqlen = atoi(argv[argIndex++]);
        numHeads = atoi(argv[argIndex++]);
        kvHeads = atoi(argv[argIndex++]);
        embeddingSize = atoi(argv[argIndex++]);
        isVariedLen = atoi(argv[argIndex++]);
        maskType = atoi(argv[argIndex++]);
        while (argIndex < argc) {
            string flag = string(argv[argIndex++]);
            if (flag == "--datapath") {
                dataPath = string(argv[argIndex++]);
            } else if (flag == "--device") {
                deviceId = atoi(argv[argIndex++]);
            } else if (flag == "--dtype") {
                dataType = string(argv[argIndex++]);
            } else {
                printf(HELPER);
                return -1;
            }
        }
        return 0;
    }
};

static void AllocMem(uint8_t** host, uint8_t** device, size_t size)
{
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(host), size));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(device), size, ACL_MEM_MALLOC_HUGE_FIRST));
}

static void FreeMem(uint8_t* host, uint8_t* device)
{
    ACL_CHECK(aclrtFreeHost(host));
    ACL_CHECK(aclrtFree(device));
}

static bool CanImplement(
    int32_t batch, int32_t qSeqlen, int32_t kvSeqlen, int32_t numHeads, int32_t kvHeads, int32_t embeddingSize,
    int32_t isVariedLen, int32_t maskType, const string& dataType)
{
    if (batch < 1 || qSeqlen < 1 || kvSeqlen < 1 || numHeads < 1 || kvHeads < 1) {
        cerr << "[ERROR] batch/qSeqlen/kvSeqlen/numHeads/kvHeads must all be positive, got batch=" << batch
             << ", qSeqlen=" << qSeqlen << ", kvSeqlen=" << kvSeqlen << ", numHeads=" << numHeads
             << ", kvHeads=" << kvHeads << endl;
        return false;
    }
    if (numHeads % kvHeads != 0) {
        cerr << "[ERROR] numHeads (" << numHeads << ") must be a positive multiple of kvHeads (" << kvHeads
             << ") for GQA/MQA." << endl;
        return false;
    }
    if (qSeqlen > kvSeqlen) {
        cerr << "[ERROR] qSeqlen (" << qSeqlen << ") must not exceed kvSeqlen (" << kvSeqlen << ")." << endl;
        return false;
    }
    if ((dataType != "half") && (dataType != "bf16")) {
        cerr << "[ERROR] dtype must be 'half' or 'bf16'." << endl;
        return false;
    }
    if (embeddingSize != 128) {
        cerr << "[ERROR] embeddingSize must be 128 (headDim fixed by the kernel template), got " << embeddingSize
             << endl;
        return false;
    }
    if (isVariedLen != 0) {
        cerr << "[ERROR] isVariedLen must be 0 (variable-length sequences are not supported yet), got " << isVariedLen
             << endl;
        return false;
    }
    if (maskType != 0 && maskType != 1) {
        cerr << "[ERROR] maskType must be 0 (no mask) or 1 (causal mask), got " << maskType << endl;
        return false;
    }
    return true;
}

// Allocate several matrices in NPU device memory and call a
// CATLASS FAI kernel.
static bool Run(const Options& options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    const auto releaseAclEarly = [&]() {
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Parameters initialization.
    int32_t batch = options.batch;
    int32_t qSeqlen = options.qSeqlen;
    int32_t kvSeqlen = options.kvSeqlen;
    int32_t numHeads = options.numHeads;
    int32_t kvHeads = options.kvHeads;
    int32_t embeddingSize = options.embeddingSize;
    int32_t blockSize = options.blockSize;
    int32_t maskType = options.maskType;
    string dataType = options.dataType;
    string dataPath = options.dataPath;
    int32_t maxKvSeqlen = kvSeqlen;
    int32_t numBlocks = batch * ((maxKvSeqlen + blockSize - 1) / blockSize);

    if (!CanImplement(
            batch, qSeqlen, kvSeqlen, numHeads, kvHeads, embeddingSize, options.isVariedLen, maskType, dataType)) {
        releaseAclEarly();
        return false;
    }

    // read qNtokens num
    void* qNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&qNtokens, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/q_ntokens.bin", qNtokens, 1 * sizeof(int32_t));
    int32_t numTokens = static_cast<int32_t*>(qNtokens)[0];

    uint64_t seqArraySize = batch * sizeof(int64_t);
    uint64_t qoSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    uint64_t kvSize =
        (uint64_t)numBlocks * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    uint64_t maskSize = 1024 * 1024 * sizeof(fp16_t);
    uint64_t blockTableSize =
        static_cast<uint64_t>(batch * ((maxKvSeqlen + blockSize - 1) / blockSize) * sizeof(int32_t));
    uint32_t tilingSize = sizeof(FATilingData);

    // Allocate matrices in host and device memory.
    uint8_t* qSeqHost;
    uint8_t* qSeqDevice;
    AllocMem(&qSeqHost, &qSeqDevice, seqArraySize);
    ReadFile(dataPath + "/q_seqlen.bin", qSeqHost, seqArraySize);
    ACL_CHECK(aclrtMemcpy(qSeqDevice, seqArraySize, qSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory.
    uint8_t* kvSeqHost;
    uint8_t* kvSeqDevice;
    AllocMem(&kvSeqHost, &kvSeqDevice, seqArraySize);
    ReadFile(dataPath + "/kv_seqlen.bin", kvSeqHost, seqArraySize);
    ACL_CHECK(aclrtMemcpy(kvSeqDevice, seqArraySize, kvSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix q.
    uint8_t* qHost;
    uint8_t* qDevice;
    AllocMem(&qHost, &qDevice, qoSize);
    ReadFile(dataPath + "/q.bin", qHost, qoSize);
    ACL_CHECK(aclrtMemcpy(qDevice, qoSize, qHost, qoSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix k.
    uint8_t* kHost;
    uint8_t* kDevice;
    AllocMem(&kHost, &kDevice, kvSize);
    ReadFile(dataPath + "/k.bin", kHost, kvSize);
    ACL_CHECK(aclrtMemcpy(kDevice, kvSize, kHost, kvSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix v.
    uint8_t* vHost;
    uint8_t* vDevice;
    AllocMem(&vHost, &vDevice, kvSize);
    ReadFile(dataPath + "/v.bin", vHost, kvSize);
    ACL_CHECK(aclrtMemcpy(vDevice, kvSize, vHost, kvSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix mask.
    uint8_t* maskHost;
    uint8_t* maskDevice;
    if (maskType == 1) {
        AllocMem(&maskHost, &maskDevice, maskSize);
        ReadFile(dataPath + "/mask.bin", maskHost, maskSize);
        ACL_CHECK(aclrtMemcpy(maskDevice, maskSize, maskHost, maskSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // Allocate matrices in host and device memory and load Matrix block_table.
    uint8_t* blockTableHost;
    uint8_t* blockTableDevice;
    AllocMem(&blockTableHost, &blockTableDevice, blockTableSize);
    ReadFile(dataPath + "/block_table.bin", blockTableHost, blockTableSize);
    ACL_CHECK(aclrtMemcpy(blockTableDevice, blockTableSize, blockTableHost, blockTableSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in device memory for workspace.
    // One base workspace block contains 65536 elements.
    uint64_t mm1OutSize = aicCoreNum * FAInferTiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * FAInferTiling::NUM3;
    uint64_t smOnlineOutSize =
        aicCoreNum * FAInferTiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(fp16_t) * FAInferTiling::NUM3;
    uint64_t mm2OutSize = aicCoreNum * FAInferTiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * FAInferTiling::NUM3;
    uint64_t UpdateSize = aicCoreNum * FAInferTiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * FAInferTiling::NUM3;
    uint64_t workSpaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize;

    uint8_t* sDevice;
    ACL_CHECK(aclrtMalloc((void**)(&sDevice), mm1OutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t* pDevice;
    ACL_CHECK(aclrtMalloc((void**)(&pDevice), smOnlineOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t* oTempDevice;
    ACL_CHECK(aclrtMalloc((void**)(&oTempDevice), mm2OutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t* oUpdateDevice;
    ACL_CHECK(aclrtMalloc((void**)(&oUpdateDevice), UpdateSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* oDevice{nullptr};
    ACL_CHECK(aclrtMalloc((void**)(&oDevice), qoSize * 2, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* tilingDevice;
    ACL_CHECK(aclrtMalloc((void**)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // get tiling
    void* tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    uint32_t blockDim = aicCoreNum;

    FAInferTiling::FAInfo faInfo;
    faInfo.numTokens = numTokens;
    faInfo.numHeads = numHeads;
    faInfo.embeddingSize = embeddingSize;
    faInfo.numBlocks = numBlocks;
    faInfo.blockSize = blockSize;
    faInfo.kvHeads = kvHeads;
    faInfo.batch = batch;
    faInfo.maskType = static_cast<FAInferTiling::MaskType>(maskType);
    faInfo.qSeqlenList = reinterpret_cast<int64_t*>(qSeqHost);
    faInfo.kvSeqlenList = reinterpret_cast<int64_t*>(kvSeqHost);

    FATilingData faTilingData;

    FAInferTiling::GetFATilingParam(faInfo, blockDim, faTilingData);

    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, &faTilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare hardware sync address
    uint64_t hardwareSyncAddr{0};
    ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)));

    bool passed = false;
    for (int i = 0; i < 1; i++) {
        if (dataType == "half") {
            FAInferTla<half><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, qSeqDevice,
                kvSeqDevice, sDevice, pDevice, oTempDevice, oUpdateDevice, tilingDevice);
        } else {
            FAInferTla<bfloat16_t><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, qSeqDevice,
                kvSeqDevice, sDevice, pDevice, oTempDevice, oUpdateDevice, tilingDevice);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        // Copy the result from device to host
        vector<fp16_t> oHostHalf(qoSize / sizeof(fp16_t));
        vector<bfloat16> oHostBf16(qoSize / sizeof(bfloat16));
        if (dataType == "half") {
            ACL_CHECK(aclrtMemcpy(oHostHalf.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
        } else if (dataType == "bf16") {
            ACL_CHECK(aclrtMemcpy(oHostBf16.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
        }

        // Compute the golden result
        vector<float> goldenHost(qoSize / sizeof(fp16_t));
        const size_t goldenSize = qoSize * 2;
        ReadFile(dataPath + "/golden.bin", goldenHost.data(), goldenSize);

        // Compare the result
        vector<uint64_t> errorIndices = (dataType == "half") ? golden::CompareData(oHostHalf, goldenHost, kvSeqlen) :
                                                               golden::CompareData(oHostBf16, goldenHost, kvSeqlen);
        if (errorIndices.empty()) {
            cout << "Compare success." << endl;
        } else {
            cerr << "Compare failed. Error count: " << errorIndices.size() << endl;
        }
        passed = errorIndices.empty();
    }

    // Free host memory allocations.
    FreeMem(qSeqHost, qSeqDevice);
    FreeMem(kvSeqHost, kvSeqDevice);
    FreeMem(qHost, qDevice);
    FreeMem(kHost, kDevice);
    FreeMem(vHost, vDevice);
    if (maskType == 1) {
        FreeMem(maskHost, maskDevice);
    }
    FreeMem(blockTableHost, blockTableDevice);
    aclrtFree(oDevice);
    aclrtFree(tilingDevice);
    aclrtFree(sDevice);
    aclrtFree(pDevice);
    aclrtFree(oTempDevice);
    aclrtFree(oUpdateDevice);
    aclrtFreeHost(tilingHost);
    aclrtFreeHost(qNtokens);

    // Destroy specified Stream and reset device.
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
    return passed;
}

/// Entry point to mla example.

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    return Run(options) ? 0 : 1;
}
