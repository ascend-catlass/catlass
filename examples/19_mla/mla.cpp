/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
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
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
// Helper methods to check for errors
#include "golden.hpp"
#include "helper.hpp"
#include "mla_kernel.cpp"
#include "mla_kernel_tp1_spec.cpp"
#include "amla_kernel_tp1_spec.cpp"
#include "mla_tiling.h"

using namespace std;

// Helper function to split comma-separated string into vector of uint32_t
vector<uint32_t> splitSeqList(const string& seqStr)
{
    vector<uint32_t> seqList;
    stringstream ss(seqStr);
    string item;
    while (getline(ss, item, ',')) {
        seqList.push_back(static_cast<uint32_t>(stoul(item)));
    }
    return seqList;
}

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER =
        "Usage: mla batch \"qSeqlenList\" \"kvSeqlenList\" numHeads numBlocks blockSize [--dtype DTYPE "
        "--datapath DATA_PATH --device DEVICE_ID --output OUTPUT_PATH]\n"
        "       mla --device-info [--device DEVICE_ID]\n"
        "       mla --test DATA_PATH [--device DEVICE_ID --case CASE_NAME]\n"
        "Example: ./19_mla 4 \"1,2,3,4\" \"128,256,512,1024\" 16 16 128\n";
    static constexpr auto MIN_ARGS = 7;

    // Define default value.
    uint32_t batch{0};
    vector<uint32_t> qSeqlenList;
    vector<uint32_t> kvSeqlenList;
    uint32_t numHeads{0};
    uint32_t numBlocks{0};
    uint32_t blockSize{0};
    uint32_t deviceId{0};

    uint32_t maskType{0};
    uint32_t kvHeads{1};
    uint32_t embeddingSize{512};
    uint32_t embeddingSizeRope{64};
    string dataType = "half";
    string dataPath = "../../examples/19_mla/data";
    string outputPath;
    bool deviceInfo{false};
    string testDataPath;
    string testCase;
    string expectedKernel;
    bool fp64Reference{false};
    bool checkUnderflow{false};
    uint32_t expectedCoreNum{0};

    Options() = default;

    // Define function to parse the command-line arguments.
    int Parse(int argc, const char** argv)
    {
        if (argc >= 2 && string(argv[1]) == "--device-info") {
            deviceInfo = true;
            if (argc == 2) {
                return 0;
            }
            if (argc == 4 && string(argv[2]) == "--device") {
                deviceId = atoi(argv[3]);
                return 0;
            }
            printf(HELPER);
            return -1;
        }
        if (argc >= 3 && string(argv[1]) == "--test") {
            testDataPath = argv[2];
            for (int i = 3; i < argc; i += 2) {
                if (i + 1 >= argc) {
                    return -1;
                }
                string flag = argv[i];
                if (flag == "--device") {
                    deviceId = stoul(argv[i + 1]);
                } else if (flag == "--case") {
                    testCase = argv[i + 1];
                } else {
                    return -1;
                }
            }
            return 0;
        }
        // The number of arguments must >= 7.
        if (argc < MIN_ARGS) {
            printf(HELPER);
            return -1;
        }

        // Allocate arguments to parameters.
        uint32_t argIndex = 1;
        batch = atoi(argv[argIndex++]);
        string qSeqStr = argv[argIndex++];
        string kvSeqStr = argv[argIndex++];
        qSeqlenList = splitSeqList(qSeqStr);
        kvSeqlenList = splitSeqList(kvSeqStr);
        if (qSeqlenList.size() != batch || kvSeqlenList.size() != batch) {
            printf("Error: The number of elements in qSeqlenList and kvSeqlenList must match the batch size.\n");
            printf(HELPER);
            return -1;
        }
        numHeads = atoi(argv[argIndex++]);
        numBlocks = atoi(argv[argIndex++]);
        blockSize = atoi(argv[argIndex++]);
        while (argIndex < argc) {
            string flag = string(argv[argIndex++]);
            if (argIndex >= argc) {
                printf("Error: Missing value for %s.\n", flag.c_str());
                return -1;
            }
            if (flag == "--datapath") {
                dataPath = string(argv[argIndex++]);
            } else if (flag == "--device") {
                deviceId = atoi(argv[argIndex++]);
            } else if (flag == "--output") {
                outputPath = string(argv[argIndex++]);
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

static void ReadDataFile(const string& path, void* destination, size_t bytes)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(bytes)) {
        throw std::runtime_error("Missing or incorrectly sized input: " + path);
    }
    input.seekg(0);
    if (!input.read(static_cast<char*>(destination), bytes)) {
        throw std::runtime_error("Failed to read input: " + path);
    }
}

template <class ElementOutput>
static bool CheckOutput(const Options& options, const vector<ElementOutput>& actual)
{
    if (options.fp64Reference) {
        vector<double> reference(actual.size());
        ReadDataFile(options.dataPath + "/golden_fp64.bin", reference.data(), reference.size() * sizeof(double));
        golden::ElementwiseTolerance tolerance = options.dataType == "half" ?
                                                     golden::ElementwiseTolerance{5e-5, 0.003} :
                                                     golden::ElementwiseTolerance{5e-4, 0.015};
        if (options.checkUnderflow) {
            // One BF16 subnormal ULP; reject a zero result or a changed sign.
            tolerance = {std::ldexp(1.0, -133), 0.0, true};
        }
        auto metrics = golden::CompareElementwise(actual, reference, tolerance);
        return metrics.passed;
    }
    vector<float> reference(actual.size());
    vector<ElementOutput> cpuLow(actual.size());
    ReadDataFile(options.dataPath + "/golden.bin", reference.data(), reference.size() * sizeof(float));
    ReadDataFile(options.dataPath + "/cpu_low.bin", cpuLow.data(), cpuLow.size() * sizeof(ElementOutput));
    auto metrics = golden::ComputeErrorMetrics(actual, cpuLow, reference, 10.0, 2.0, 2.0);
    if (!metrics.passed) {
        cerr << "Error ratios exceed thresholds: MARE=" << metrics.mareRatio << " MERE=" << metrics.mereRatio
             << " RMSE=" << metrics.rmseRatio << endl;
    }
    return metrics.passed;
}

static bool CanImplement(
    const vector<uint32_t>& qSeqlenList, const vector<uint32_t>& kvSeqlenList, uint32_t numHeads, uint32_t numBlocks,
    uint32_t blockSize, const string& dataType, uint32_t batch, uint32_t aicCoreNum)
{
    if (qSeqlenList.empty() || kvSeqlenList.empty() || qSeqlenList.size() != kvSeqlenList.size()) {
        cerr << "[ERROR] qSeqlenList/kvSeqlenList must be non-empty and of the same size, got " << qSeqlenList.size()
             << " and " << kvSeqlenList.size() << endl;
        return false;
    }
    for (size_t i = 0; i < qSeqlenList.size(); ++i) {
        if (qSeqlenList[i] < 1 || qSeqlenList[i] > 4) {
            cerr << "[ERROR] qSeqlen must be in [1, 4] (decode/MTP), got qSeqlenList[" << i << "]=" << qSeqlenList[i]
                 << endl;
            return false;
        }
        if (kvSeqlenList[i] < 128 || kvSeqlenList[i] > 16384) {
            cerr << "[ERROR] kvSeqlen must be in [128, 16384], got kvSeqlenList[" << i << "]=" << kvSeqlenList[i]
                 << endl;
            return false;
        }
        if (kvSeqlenList[i] < qSeqlenList[i]) {
            cerr << "[ERROR] kvSeqlen (" << kvSeqlenList[i] << ") must not be less than qSeqlen (" << qSeqlenList[i]
                 << ")." << endl;
            return false;
        }
    }
    if (numHeads != 16 && numHeads != 32 && numHeads != 64 && numHeads != 128) {
        cerr << "[ERROR] numHeads must be 16, 32, 64 or 128 (TP8/TP4/TP2/TP1), got " << numHeads << endl;
        return false;
    }
    uint32_t maxQseqlen = 0;
    for (uint32_t qSeqlen : qSeqlenList) {
        if (qSeqlen > maxQseqlen) {
            maxQseqlen = qSeqlen;
        }
    }
    if (numHeads != 128 && maxQseqlen > 1 && batch > aicCoreNum) {
        cerr << "[ERROR] generic kernel (numHeads in {16, 32, 64}) requires batch (" << batch << ") <= AIC core num ("
             << aicCoreNum << ") when qSeqlen > 1, got maxQseqlen=" << maxQseqlen << endl;
        return false;
    }
    if (blockSize != 128) {
        cerr << "[ERROR] blockSize must be 128, got " << blockSize << endl;
        return false;
    }
    uint64_t requiredBlocks = 0;
    for (uint32_t kv : kvSeqlenList) {
        requiredBlocks += (static_cast<uint64_t>(kv) + blockSize - 1) / blockSize;
    }
    if (requiredBlocks > numBlocks) {
        cerr << "[ERROR] numBlocks (" << numBlocks << ") is less than required "
             << "sum(ceil(kvSeqlen / blockSize)) = " << requiredBlocks << endl;
        return false;
    }
    if ((dataType != "half") && (dataType != "bf16")) {
        cerr << "[ERROR] dtype must be 'half' or 'bf16'." << endl;
        return false;
    }
    return true;
}

// Allocate several matrices in NPU device memory and call a
// CATLASS MLA kernel.
static bool Run(const Options& options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    if (options.deviceInfo) {
        std::cout << "{\"aic_core_count\":" << aicCoreNum << "}" << std::endl;
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return true;
    }
    if (options.expectedCoreNum != 0 && options.expectedCoreNum != aicCoreNum) {
        throw std::runtime_error("Test data was generated for a different AIC core count; regenerate the suite.");
    }

    // Parameters initialization.
    int32_t batch = options.batch;
    int32_t numHeads = options.numHeads;
    int32_t kvHeads = options.kvHeads;
    int32_t embeddingSize = options.embeddingSize;
    int32_t embeddingSizeRope = options.embeddingSizeRope;
    int32_t numBlocks = options.numBlocks;
    int32_t blockSize = options.blockSize;
    int32_t maskType = options.maskType;
    string dataType = options.dataType;
    string dataPath = options.dataPath;
    int32_t maxKvSeqlen = 0;
    if (!options.kvSeqlenList.empty()) {
        maxKvSeqlen = options.kvSeqlenList[0];
        for (size_t i = 1; i < options.kvSeqlenList.size(); ++i) {
            if (options.kvSeqlenList[i] > maxKvSeqlen) {
                maxKvSeqlen = options.kvSeqlenList[i];
            }
        }
    }

    if (!CanImplement(
            options.qSeqlenList, options.kvSeqlenList, options.numHeads, options.numBlocks, options.blockSize, dataType,
            options.batch, aicCoreNum)) {
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return false;
    }

    int32_t dTypeKey = (dataType == "half") ? 0 : 1;
    int32_t specStraKey = (numHeads == MLATiling::NUM128) ? 1 : 0;

    // read qNtokens num
    void* qNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&qNtokens, 1 * sizeof(int32_t)));
    ReadDataFile(dataPath + "/q_ntokens.bin", qNtokens, 1 * sizeof(int32_t));
    int32_t numTokens = static_cast<int32_t*>(qNtokens)[0];

    // read qSeq
    void* qSeq = nullptr;
    ACL_CHECK(aclrtMallocHost(&qSeq, batch * sizeof(int32_t)));
    ReadDataFile(dataPath + "/q_seqlen.bin", qSeq, batch * sizeof(int32_t));

    // read kvSeq num
    void* kvSeq = nullptr;
    ACL_CHECK(aclrtMallocHost(&kvSeq, batch * sizeof(int32_t)));
    ReadDataFile(dataPath + "/kv_seqlen.bin", kvSeq, batch * sizeof(int32_t));

    uint64_t qoSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    uint64_t qRopeSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)embeddingSizeRope * sizeof(fp16_t);
    uint64_t kvSize =
        (uint64_t)numBlocks * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    uint64_t kRopeSize =
        (uint64_t)numBlocks * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)embeddingSizeRope * sizeof(fp16_t);
    uint64_t maskSize = (uint64_t)numTokens * (uint64_t)maxKvSeqlen * sizeof(fp16_t);
    uint64_t blockTableSize =
        static_cast<uint64_t>(batch * ((maxKvSeqlen + blockSize - 1) / blockSize) * sizeof(int32_t));
    uint32_t tilingSize = (MLATiling::TILING_HEAD_SIZE + batch * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    if (specStraKey > 0) {
        tilingSize = (MLATiling::TILING_HEAD_SIZE + numTokens * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    }

    // Allocate matrices in host and device memory and load Matrix q.
    uint8_t* qHost;
    uint8_t* qDevice;
    AllocMem(&qHost, &qDevice, qoSize);
    ReadDataFile(dataPath + "/q.bin", qHost, qoSize);
    ACL_CHECK(aclrtMemcpy(qDevice, qoSize, qHost, qoSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix q_rope.
    uint8_t* qRopeHost;
    uint8_t* qRopeDevice;
    AllocMem(&qRopeHost, &qRopeDevice, qRopeSize);
    ReadDataFile(dataPath + "/q_rope.bin", qRopeHost, qRopeSize);
    ACL_CHECK(aclrtMemcpy(qRopeDevice, qRopeSize, qRopeHost, qRopeSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix k.
    uint8_t* kHost;
    uint8_t* kDevice;
    AllocMem(&kHost, &kDevice, kvSize);
    ReadDataFile(dataPath + "/k.bin", kHost, kvSize);
    ACL_CHECK(aclrtMemcpy(kDevice, kvSize, kHost, kvSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix k_rope.
    uint8_t* kRopeHost;
    uint8_t* kRopeDevice;
    AllocMem(&kRopeHost, &kRopeDevice, kRopeSize);
    ReadDataFile(dataPath + "/k_rope.bin", kRopeHost, kRopeSize);
    ACL_CHECK(aclrtMemcpy(kRopeDevice, kRopeSize, kRopeHost, kRopeSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix block_table.
    uint8_t* blockTableHost;
    uint8_t* blockTableDevice;
    AllocMem(&blockTableHost, &blockTableDevice, blockTableSize);
    ReadDataFile(dataPath + "/block_table.bin", blockTableHost, blockTableSize);
    ACL_CHECK(aclrtMemcpy(blockTableDevice, blockTableSize, blockTableHost, blockTableSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in device memory for workspace.
    uint8_t* sDevice;
    ACL_CHECK(aclrtMalloc(
        (void**)(&sDevice), aicCoreNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * MLATiling::NUM2,
        ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* pDevice;
    ACL_CHECK(aclrtMalloc(
        (void**)(&pDevice), aicCoreNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(fp16_t) * MLATiling::NUM2,
        ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* oTmpDevice;
    ACL_CHECK(aclrtMalloc(
        (void**)(&oTmpDevice), aicCoreNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * MLATiling::NUM2,
        ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* globaloDevice;
    ACL_CHECK(aclrtMalloc(
        (void**)(&globaloDevice), aicCoreNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float),
        ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* oDevice{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&oDevice), static_cast<size_t>(qoSize), ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* tilingDevice;
    ACL_CHECK(aclrtMalloc((void**)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // get tiling
    void* tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    uint32_t blockDim = aicCoreNum;

    MLATiling::MLAInfo mlaInfo;
    mlaInfo.numTokens = numTokens;
    mlaInfo.numHeads = numHeads;
    mlaInfo.embeddingSize = embeddingSize;
    mlaInfo.embeddingSizeRope = embeddingSizeRope;
    mlaInfo.numBlocks = numBlocks;
    mlaInfo.blockSize = blockSize;
    mlaInfo.maxKvSeqlen = maxKvSeqlen;
    mlaInfo.kvHeads = kvHeads;
    mlaInfo.batch = batch;
    mlaInfo.qSeqLen = static_cast<int32_t*>(qSeq);
    mlaInfo.kvSeqLen = static_cast<int32_t*>(kvSeq);
    if (MLATiling::GetMLATilingParam(mlaInfo, blockDim, (uint32_t*)tilingHost) != 0) {
        throw std::runtime_error("Invalid MLA tiling parameters");
    }
    uint32_t tilingKey = MLATiling::GetMLATilingKey(numHeads, dTypeKey, (const uint32_t*)tilingHost);

    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint32_t maxKvSplitCoreNum = *((uint32_t*)tilingHost + MLATiling::TILING_KVCORENUM);
    uint64_t oFdSize = embeddingSize * numHeads * numTokens * maxKvSplitCoreNum * sizeof(float);
    uint64_t lSize = numTokens * numHeads * maxKvSplitCoreNum * sizeof(float);

    uint8_t* oCoreTmpDevice;
    ACL_CHECK(aclrtMalloc((void**)(&oCoreTmpDevice), oFdSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* lDevice;
    ACL_CHECK(aclrtMalloc((void**)(&lDevice), lSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare hardware sync address
    uint64_t hardwareSyncAddr{0};
    ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)));

    // use Tp1Spec kernel to get better performance when numHeads = 128
    string launchedKernel;
    switch (tilingKey) {
        case 0: {
            launchedKernel = "common";
            MLA<half><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oDevice, sDevice,
                pDevice, oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        }
        case 1: {
            launchedKernel = "common";
            MLA<bfloat16_t><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oDevice, sDevice,
                pDevice, oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        }
        case 4: {
            launchedKernel = "amla";
            AMLATp1Spec<half><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oDevice, sDevice,
                pDevice, oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        }
        case 5: {
            launchedKernel = "amla";
            AMLATp1Spec<bfloat16_t><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oDevice, sDevice,
                pDevice, oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        }
        case 7: {
            launchedKernel = "tail";
            MLATp1Spec<half><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oDevice, sDevice,
                pDevice, oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        }
        case 8: {
            launchedKernel = "tail";
            MLATp1Spec<bfloat16_t><<<blockDim, nullptr, stream>>>(
                hardwareSyncAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oDevice, sDevice,
                pDevice, oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
            break;
        }
        default:
            throw std::runtime_error("Unsupported MLA tiling key");
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    // Copy the result from device to host
    vector<fp16_t> oHostHalf(qoSize / sizeof(fp16_t));
    vector<bfloat16> oHostBf16(qoSize / sizeof(bfloat16), (bfloat16)2.1);
    if (dataType == "half") {
        ACL_CHECK(aclrtMemcpy(oHostHalf.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
    } else if (dataType == "bf16") {
        ACL_CHECK(aclrtMemcpy(oHostBf16.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
    }

    if (!options.outputPath.empty()) {
        std::ofstream output(options.outputPath, std::ios::binary);
        const char* data = dataType == "half" ? reinterpret_cast<const char*>(oHostHalf.data()) :
                                                reinterpret_cast<const char*>(oHostBf16.data());
        output.write(data, qoSize);
        if (!output) {
            throw std::runtime_error("Failed to write output: " + options.outputPath);
        }
    }

    bool passed = dataType == "half" ? CheckOutput(options, oHostHalf) : CheckOutput(options, oHostBf16);
    if (!options.expectedKernel.empty() && launchedKernel != options.expectedKernel) {
        cerr << "Expected kernel " << options.expectedKernel << ", launched " << launchedKernel << endl;
        passed = false;
    }
    cout << (passed ? "Compare success." : "Compare failed.") << endl;

    // Free host memory allocations.
    FreeMem(qHost, qDevice);
    FreeMem(qRopeHost, qRopeDevice);
    FreeMem(kHost, kDevice);
    FreeMem(kRopeHost, kRopeDevice);
    FreeMem(blockTableHost, blockTableDevice);
    aclrtFree(oDevice);
    aclrtFree(tilingDevice);
    aclrtFree(sDevice);
    aclrtFree(pDevice);
    aclrtFree(oTmpDevice);
    aclrtFree(globaloDevice);
    aclrtFree(oCoreTmpDevice);
    aclrtFree(lDevice);
    aclrtFreeHost(tilingHost);
    aclrtFreeHost(qNtokens);
    aclrtFreeHost(qSeq);
    aclrtFreeHost(kvSeq);

    // Destroy specified Stream and reset device.
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
    return passed;
}

// The manifest contains input parameters and coverage assertions, never kernel keys.
static bool RunTests(const Options& options)
{
    std::ifstream manifest(options.testDataPath + "/cases.txt");
    string magic;
    uint32_t version = 0;
    uint32_t coreCount = 0;
    if (!(manifest >> magic >> version >> coreCount) || magic != "MLA_TEST_SUITE" || version != 1 || coreCount == 0) {
        throw std::runtime_error("Invalid MLA test manifest");
    }
    string line;
    getline(manifest, line);
    uint32_t tested = 0;
    uint32_t failed = 0;
    while (getline(manifest, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream fields(line);
        string name, batch, qlens, klens, heads, blocks, blockSize, dtype, mode, family, extra;
        if (!(fields >> name >> batch >> qlens >> klens >> heads >> blocks >> blockSize >> dtype >> mode >> family) ||
            (fields >> extra) || (mode != "standard" && mode != "underflow") ||
            (family != "any" && family != "common" && family != "amla" && family != "tail")) {
            throw std::runtime_error("Invalid MLA test case: " + line);
        }
        if (!options.testCase.empty() && name != options.testCase) {
            continue;
        }
        vector<string> arguments = {
            "19_mla",
            batch,
            qlens,
            klens,
            heads,
            blocks,
            blockSize,
            "--dtype",
            dtype,
            "--device",
            std::to_string(options.deviceId),
            "--datapath",
            options.testDataPath + "/" + name,
            "--output",
            options.testDataPath + "/" + name + "/result.bin"};
        vector<const char*> argv;
        for (const auto& argument : arguments) {
            argv.push_back(argument.c_str());
        }
        Options testOptions;
        if (testOptions.Parse(argv.size(), argv.data()) != 0) {
            throw std::runtime_error("Invalid parameters for " + name);
        }
        testOptions.fp64Reference = true;
        testOptions.checkUnderflow = mode == "underflow";
        testOptions.expectedKernel = family == "any" ? "" : family;
        testOptions.expectedCoreNum = coreCount;
        cout << "[ RUN      ] " << name << endl;
        bool passed = Run(testOptions);
        ++tested;
        failed += !passed;
        cout << (passed ? "[       OK ] " : "[  FAILED  ] ") << name << endl;
    }
    cout << "MLA tests: " << tested << " run, " << failed << " failed" << endl;
    return tested != 0 && failed == 0;
}

/// Entry point to mla example.
int main(int argc, const char** argv)
{
    try {
        Options options;
        if (options.Parse(argc, argv) != 0) {
            return 1;
        }
        return (options.testDataPath.empty() ? Run(options) : RunTests(options)) ? 0 : 1;
    } catch (const std::exception& error) {
        cerr << error.what() << endl;
        return 1;
    }
}
