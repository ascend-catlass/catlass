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

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#define NO_OVERLAP_IN_MULTI_REPEAT

#include <iostream>
#include <cstdio>
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"

#include "./launcher/hstu_infer_launcher.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

using namespace std;

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER =
        "Usage: hstu batch qSeqlen kvSeqlen numHeads kvHeads embeddingSize isVariedLen siluScale [--dtype DTYPE "
        "--layout LAYOUT --datapath DATA_PATH --device DEVICE_ID]\n";
    static constexpr auto MIN_ARGS = 9;

    // Define default value.
    uint32_t batch{0};
    uint32_t qSeqlen{0};
    uint32_t kvSeqlen{0};
    uint32_t numHeads{0};
    uint32_t kvHeads{0};
    uint32_t embeddingSize{0};
    uint32_t isVariedLen{0};
    uint32_t deviceId{0};
    uint32_t maskType{0};
    float siluScale{0.1};
    string layout = "TND";
    string dataType = "half";
    string dataPath = "./examples/83_ascend950_hstu_infer/data";
    uint32_t pagedBlockSize{0};

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
        siluScale = static_cast<float>(std::atof(string(argv[argIndex++]).c_str()));
        while (argIndex < argc) {
            string flag = string(argv[argIndex++]);
            if (flag == "--datapath") {
                dataPath = string(argv[argIndex++]);
            } else if (flag == "--device") {
                deviceId = atoi(argv[argIndex++]);
            } else if (flag == "--dtype") {
                dataType = string(argv[argIndex++]);
            } else if (flag == "--layout") {
                layout = string(argv[argIndex++]);
            } else if (flag == "--mask") {
                maskType = atoi(argv[argIndex++]);
            } else if (flag == "--paged_block_size") {
                pagedBlockSize = atoi(argv[argIndex++]);
            } else {
                printf(HELPER);
                return -1;
            }
        }
        return 0;
    }
};

void AllocMem(uint8_t** host, uint8_t** device, size_t size)
{
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(host), size));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(device), size, ACL_MEM_MALLOC_HUGE_FIRST));
}

void FreeMem(uint8_t* host, uint8_t* device)
{
    if (host != nullptr) {
        ACL_CHECK(aclrtFreeHost(host));
    }
    if (device != nullptr) {
        ACL_CHECK(aclrtFree(device));
    }
}

static void Run(const Options& options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Parameters initialization.
    uint32_t batch = options.batch;
    uint32_t qSeqlen = options.qSeqlen;
    uint32_t numHeads = options.numHeads;
    uint32_t kvHeads = options.kvHeads;
    uint32_t embeddingSize = options.embeddingSize;
    float siluScale = options.siluScale;
    string layout = options.layout;
    string dataType = options.dataType;
    string dataPath = options.dataPath;
    uint32_t maskType = options.maskType;
    uint32_t pagedBlockSize = options.pagedBlockSize;

    if ((dataType != "half") && (dataType != "bf16")) {
        cerr << "[ERROR] dtype must be 'half' or 'bf16'." << endl;
        return;
    }

    void* qNtokens{nullptr};
    void* kvNtokens{nullptr};
    uint8_t* qSeqHost{nullptr};
    uint8_t* qSeqDevice{nullptr};
    uint8_t* kvSeqHost{nullptr};
    uint8_t* kvSeqDevice{nullptr};
    uint8_t* qHost{nullptr};
    uint8_t* qDevice{nullptr};
    uint8_t* kHost{nullptr};
    uint8_t* kDevice{nullptr};
    uint8_t* vHost{nullptr};
    uint8_t* vDevice{nullptr};
    uint8_t* blockTableHost{nullptr};
    uint8_t* blockTableDevice{nullptr};
    uint8_t* oDevice{nullptr};
    uint8_t* workspaceDevice{nullptr};

    const auto cleanupAndFinalize = [&]() {
        if (qNtokens) {
            ACL_CHECK(aclrtFreeHost(qNtokens));
        }
        if (kvNtokens) {
            ACL_CHECK(aclrtFreeHost(kvNtokens));
        }
        if (qSeqDevice) {
            FreeMem(qSeqHost, qSeqDevice);
        }
        if (kvSeqDevice) {
            FreeMem(kvSeqHost, kvSeqDevice);
        }
        if (qDevice) {
            FreeMem(qHost, qDevice);
        }
        if (kDevice) {
            FreeMem(kHost, kDevice);
        }
        if (vDevice) {
            FreeMem(vHost, vDevice);
        }
        if (blockTableDevice) {
            FreeMem(blockTableHost, blockTableDevice);
        }
        if (oDevice) {
            ACL_CHECK(aclrtFree(oDevice));
        }
        if (workspaceDevice) {
            ACL_CHECK(aclrtFree(workspaceDevice));
        }

        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };

    // read qNtokens num
    ACL_CHECK(aclrtMallocHost(&qNtokens, 1 * sizeof(int32_t)));
    if (!ReadFile(dataPath + "/q_ntokens.bin", qNtokens, 1 * sizeof(int32_t))) {
        cleanupAndFinalize();
        return;
    }
    int32_t numTokens = static_cast<int32_t*>(qNtokens)[0];

    // read kvNtokens num
    ACL_CHECK(aclrtMallocHost(&kvNtokens, 1 * sizeof(int32_t)));
    if (!ReadFile(dataPath + "/kv_ntokens.bin", kvNtokens, 1 * sizeof(int32_t))) {
        cleanupAndFinalize();
        return;
    }
    int32_t numkvTokens = static_cast<int32_t*>(kvNtokens)[0];

    uint64_t seqArraySize = (batch + 1) * sizeof(int64_t);
    uint64_t qoSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)embeddingSize * sizeof(fp16_t);

    // Allocate matrices in host and device memory.
    AllocMem(&qSeqHost, &qSeqDevice, seqArraySize);
    if (!ReadFile(dataPath + "/q_seqlen.bin", qSeqHost, seqArraySize)) {
        cleanupAndFinalize();
        return;
    }
    ACL_CHECK(aclrtMemcpy(qSeqDevice, seqArraySize, qSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory.
    AllocMem(&kvSeqHost, &kvSeqDevice, seqArraySize);
    if (!ReadFile(dataPath + "/kv_seqlen.bin", kvSeqHost, seqArraySize)) {
        cleanupAndFinalize();
        return;
    }
    ACL_CHECK(aclrtMemcpy(kvSeqDevice, seqArraySize, kvSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix q.
    AllocMem(&qHost, &qDevice, qoSize);
    if (!ReadFile(dataPath + "/q.bin", qHost, qoSize)) {
        cleanupAndFinalize();
        return;
    }
    ACL_CHECK(aclrtMemcpy(qDevice, qoSize, qHost, qoSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint32_t enable_paged_kv = pagedBlockSize > 0 ? 1 : 0;
    uint32_t maxKvSeqlen = 0;
    uint64_t kvSize = 0;
    uint64_t blockTableSize = 0;
    uint32_t numPagedBlocks = 0;

    if (!enable_paged_kv) {
        kvSize = (uint64_t)numkvTokens * (uint64_t)kvHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    } else {
        int64_t* qSeqlenList = reinterpret_cast<int64_t*>(qSeqHost);
        int64_t* kvSeqlenList = reinterpret_cast<int64_t*>(kvSeqHost);
        for (int32_t batchIdx = 0; batchIdx < batch; batchIdx++) {
            int64_t kvSeqlen = *(kvSeqlenList + batchIdx + 1) - *(kvSeqlenList + batchIdx);
            maxKvSeqlen = kvSeqlen > maxKvSeqlen ? kvSeqlen : maxKvSeqlen;
            numPagedBlocks += CeilDiv(kvSeqlen, pagedBlockSize);
        }
        uint32_t maxNumBlocks = CeilDiv(maxKvSeqlen, pagedBlockSize);
        uint32_t numBlocks = batch * maxNumBlocks;
        kvSize = (uint64_t)numBlocks * (uint64_t)pagedBlockSize * (uint64_t)kvHeads * (uint64_t)embeddingSize *
                 sizeof(fp16_t);
        blockTableSize = (uint64_t)batch * maxNumBlocks * sizeof(uint32_t);
    }

    // Allocate matrices in host and device memory and load Matrix k.
    AllocMem(&kHost, &kDevice, kvSize);
    if (!ReadFile(dataPath + "/k.bin", kHost, kvSize)) {
        cleanupAndFinalize();
        return;
    }
    ACL_CHECK(aclrtMemcpy(kDevice, kvSize, kHost, kvSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix v.
    AllocMem(&vHost, &vDevice, kvSize);
    if (!ReadFile(dataPath + "/v.bin", vHost, kvSize)) {
        cleanupAndFinalize();
        return;
    }
    ACL_CHECK(aclrtMemcpy(vDevice, kvSize, vHost, kvSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Paged KV Cache: read block table
    if (enable_paged_kv) {
        AllocMem(&blockTableHost, &blockTableDevice, blockTableSize);
        if (!ReadFile(dataPath + "/block_table.bin", blockTableHost, blockTableSize)) {
            cleanupAndFinalize();
            return;
        }
        ACL_CHECK(
            aclrtMemcpy(blockTableDevice, blockTableSize, blockTableHost, blockTableSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // output matrix
    ACL_CHECK(aclrtMalloc((void**)(&oDevice), qoSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint32_t layoutId = layout == "NTD" ? static_cast<uint32_t>(0) : static_cast<uint32_t>(1);

    if (layoutId == 0) {
        if (enable_paged_kv == 0) {
            printf("RunHSTUKernel<half, layout::NTD, layout::NTD, false>(...)\n");
            RunHSTUKernel<half, layout::NTD, layout::NTD, false>(
                qDevice, kDevice, vDevice, oDevice, qSeqDevice, kvSeqDevice, blockTableDevice, batch, numHeads,
                embeddingSize, kvHeads, maxKvSeqlen, numPagedBlocks, pagedBlockSize, siluScale, maskType, stream,
                aicCoreNum, &workspaceDevice);
        } else {
            printf("RunHSTUKernel<half, layout::NTD, layout::NHD, true>(...)\n");
            RunHSTUKernel<half, layout::NTD, layout::NHD, true>(
                qDevice, kDevice, vDevice, oDevice, qSeqDevice, kvSeqDevice, blockTableDevice, batch, numHeads,
                embeddingSize, kvHeads, maxKvSeqlen, numPagedBlocks, pagedBlockSize, siluScale, maskType, stream,
                aicCoreNum, &workspaceDevice);
        }
    } else {
        if (enable_paged_kv == 0) {
            printf("RunHSTUKernel<half, layout::TND, layout::TND, false>(...)\n");
            RunHSTUKernel<half, layout::TND, layout::TND, false>(
                qDevice, kDevice, vDevice, oDevice, qSeqDevice, kvSeqDevice, blockTableDevice, batch, numHeads,
                embeddingSize, kvHeads, maxKvSeqlen, numPagedBlocks, pagedBlockSize, siluScale, maskType, stream,
                aicCoreNum, &workspaceDevice);
        } else {
            printf("RunHSTUKernel<half, layout::TND, layout::NHD, true>(...)\n");
            RunHSTUKernel<half, layout::TND, layout::NHD, true>(
                qDevice, kDevice, vDevice, oDevice, qSeqDevice, kvSeqDevice, blockTableDevice, batch, numHeads,
                embeddingSize, kvHeads, maxKvSeqlen, numPagedBlocks, pagedBlockSize, siluScale, maskType, stream,
                aicCoreNum, &workspaceDevice);
        }
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
    if (!ReadFile(dataPath + "/golden.bin", goldenHost.data(), goldenSize)) {
        cleanupAndFinalize();
        return;
    }

    // Compare the result
    vector<uint64_t> errorIndices = (dataType == "half") ?
                                        golden::CompareData(oHostHalf, goldenHost, qoSize / sizeof(fp16_t)) :
                                        golden::CompareData(oHostBf16, goldenHost, qoSize / sizeof(fp16_t));
    if (errorIndices.empty()) {
        cout << "Compare success." << endl;
    } else {
        cerr << "Compare failed. Error count: " << errorIndices.size() << endl;
    }

    // Free host memory allocations.
    FreeMem(qSeqHost, qSeqDevice);
    FreeMem(kvSeqHost, kvSeqDevice);
    FreeMem(qHost, qDevice);
    FreeMem(kHost, kDevice);
    FreeMem(vHost, vDevice);
    if (enable_paged_kv) {
        FreeMem(blockTableHost, blockTableDevice);
    }
    ACL_CHECK(aclrtFree(oDevice));
    ACL_CHECK(aclrtFreeHost(qNtokens));
    ACL_CHECK(aclrtFreeHost(kvNtokens));
    if (workspaceDevice != nullptr) {
        ACL_CHECK(aclrtFree(workspaceDevice));
    }

    // Destroy specified Stream and reset device.
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
