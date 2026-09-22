/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "fag_tla_kernel.cpp"
#include "fag_tiling.cpp"
#include "golden.hpp"
#include "helper.hpp"
#include "kernel_common_fag.hpp"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <string>

using namespace std;

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER =
        "Usage: fag tiling params [--dtype DTYPE "
        "--datapath DATA_PATH --device DEVICE_ID]\n";
    static constexpr auto MIN_ARGS = 11;

    // Define default value.
    int64_t batch{0};
    int64_t qSeqlen{0};
    int64_t qHeadNum{0};
    int64_t qkHeadDim{0};
    int64_t kvSeqlen{0};
    int64_t kvHeadNum{0};
    int64_t vHeadDim{0};
    int64_t window_size_left{0};
    int64_t window_size_right{0};
    string layout = "BSND";
    bool isDeterministic{false};
    int64_t deviceId{0};
    // int32_t maskType{0};
    string dataType = "half";
    string dataPath = "./data";

    Options() = default;

    // Define function to parse the command-line arguments.
    int Parse(int argc, const char **argv) {
        // The number of arguments must >= 7.
        if (argc < MIN_ARGS) {
            printf(HELPER);
            return -1;
        }

        // Allocate arguments to parameters.
        uint32_t argIndex = 1;
        batch = atoi(argv[argIndex++]);
        qSeqlen = atoi(argv[argIndex++]);
        qHeadNum = atoi(argv[argIndex++]);
        qkHeadDim = atoi(argv[argIndex++]);
        kvSeqlen = atoi(argv[argIndex++]);
        kvHeadNum = atoi(argv[argIndex++]);
        vHeadDim = atoi(argv[argIndex++]);
        window_size_left = atoi(argv[argIndex++]);
        window_size_right = atoi(argv[argIndex++]);
        layout = string(argv[argIndex++]);
        isDeterministic = atoi(argv[argIndex++]);
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

static void AllocMem(uint8_t **host, uint8_t **device, size_t size) {
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(host), size));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(device), size, ACL_MEM_MALLOC_HUGE_FIRST));
}

static void FreeMem(uint8_t *host, uint8_t *device) {
    ACL_CHECK(aclrtFreeHost(host));
    ACL_CHECK(aclrtFree(device));
}

// Allocate several matrices in NPU device memory and call a
// CATLASS FAG kernel.
static void Run(const Options &options) {
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // Parameters initialization.
    int64_t batch = options.batch;
    int64_t qSeqlen = options.qSeqlen;
    int64_t qHeadNum = options.qHeadNum;
    int64_t qkHeadDim = options.qkHeadDim;
    int64_t kvSeqlen = options.kvSeqlen;
    int64_t kvHeadNum = options.kvHeadNum;
    int64_t vHeadDim = options.vHeadDim;
    int64_t window_size_left = options.window_size_left;
    int64_t window_size_right = options.window_size_right;
    string layout = options.layout;
    bool isDeterministic = options.isDeterministic;
    string dataType = options.dataType;
    string dataPath = options.dataPath;

    if ((dataType != "half") && (dataType != "bf16")) {
        cerr << "[ERROR] dtype must be 'half' or 'bf16'." << endl;
        return;
    }

    void *qNtokens = nullptr;
    void *kvNtokens = nullptr;
    int32_t qNumTokens = 0;
    int32_t kvNumTokens = 0;
    uint8_t *qSeqHost;
    uint8_t *qSeqDevice;
    uint8_t *kvSeqHost;
    uint8_t *kvSeqDevice;
    uint64_t seqArraySize = batch * sizeof(int64_t);

    if (layout == "TND") {
        ACL_CHECK(aclrtMallocHost(&qNtokens, 1 * sizeof(int32_t)));
        ReadFile(dataPath + "/q_ntokens.bin", qNtokens, 1 * sizeof(int32_t));
        qNumTokens = static_cast<int32_t *>(qNtokens)[0];

        ACL_CHECK(aclrtMallocHost(&kvNtokens, 1 * sizeof(int32_t)));
        ReadFile(dataPath + "/kv_ntokens.bin", kvNtokens, 1 * sizeof(int32_t));
        kvNumTokens = static_cast<int32_t *>(kvNtokens)[0];

        // Allocate matrices in host and device memory.
        AllocMem(&qSeqHost, &qSeqDevice, seqArraySize);
        ReadFile(dataPath + "/q_seqlen.bin", qSeqHost, seqArraySize);
        ACL_CHECK(aclrtMemcpy(qSeqDevice, seqArraySize, qSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

        // Allocate matrices in host and device memory.
        AllocMem(&kvSeqHost, &kvSeqDevice, seqArraySize);
        ReadFile(dataPath + "/kv_seqlen.bin", kvSeqHost, seqArraySize);
        ACL_CHECK(aclrtMemcpy(kvSeqDevice, seqArraySize, kvSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // Determine data type size based on dataType parameter
    size_t elemSize = (dataType == "bf16") ? sizeof(bfloat16_t) : sizeof(fp16_t);
    uint64_t qoSize = 0;
    uint64_t kSize = 0;
    uint64_t vSize = 0;
    uint64_t softMaxLseSize = 0;
    if (layout == "TND") {
        qoSize = (uint64_t)qNumTokens * (uint64_t)qHeadNum * (uint64_t)qkHeadDim * elemSize;
        kSize = (uint64_t)kvNumTokens * (uint64_t)kvHeadNum * (uint64_t)qkHeadDim * elemSize;
        vSize = (uint64_t)kvNumTokens * (uint64_t)kvHeadNum * (uint64_t)vHeadDim * elemSize;
        softMaxLseSize = (uint64_t)qNumTokens * (uint64_t)qHeadNum * sizeof(float);
    } else {
        qoSize = (uint64_t)batch * (uint64_t)qSeqlen * (uint64_t)qHeadNum * (uint64_t)qkHeadDim * elemSize;
        kSize = (uint64_t)batch * (uint64_t)kvSeqlen * (uint64_t)kvHeadNum * (uint64_t)qkHeadDim * elemSize;
        vSize = (uint64_t)batch * (uint64_t)kvSeqlen * (uint64_t)kvHeadNum * (uint64_t)vHeadDim * elemSize;
        // uint64_t attenMaskSize = 2048 * 2048 * sizeof(bool);
        softMaxLseSize = (uint64_t)batch * (uint64_t)qSeqlen * (uint64_t)qHeadNum * sizeof(float);
    }

    // Allocate matrices in host and device memory and load Matrix q.
    uint8_t *doutHost;
    uint8_t *doutDevice;
    AllocMem(&doutHost, &doutDevice, qoSize);
    ReadFile(dataPath + "/dout.bin", doutHost, qoSize);
    ACL_CHECK(aclrtMemcpy(doutDevice, qoSize, doutHost, qoSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix q.
    uint8_t *qHost;
    uint8_t *qDevice;
    AllocMem(&qHost, &qDevice, qoSize);
    ReadFile(dataPath + "/q.bin", qHost, qoSize);
    ACL_CHECK(aclrtMemcpy(qDevice, qoSize, qHost, qoSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix k.
    uint8_t *kHost;
    uint8_t *kDevice;
    AllocMem(&kHost, &kDevice, kSize);
    ReadFile(dataPath + "/k.bin", kHost, kSize);
    ACL_CHECK(aclrtMemcpy(kDevice, kSize, kHost, kSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix v.
    uint8_t *vHost;
    uint8_t *vDevice;
    AllocMem(&vHost, &vDevice, vSize);
    ReadFile(dataPath + "/v.bin", vHost, vSize);
    ACL_CHECK(aclrtMemcpy(vDevice, vSize, vHost, vSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix out.
    uint8_t *outHost;
    uint8_t *outDevice;
    AllocMem(&outHost, &outDevice, qoSize);
    ReadFile(dataPath + "/out.bin", outHost, qoSize);
    ACL_CHECK(aclrtMemcpy(outDevice, qoSize, outHost, qoSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *softMaxLseHost;
    uint8_t *softMaxLseDevice;
    AllocMem(&softMaxLseHost, &softMaxLseDevice, softMaxLseSize);
    ReadFile(dataPath + "/softmax_lse.bin", softMaxLseHost, softMaxLseSize);
    ACL_CHECK(aclrtMemcpy(softMaxLseDevice, softMaxLseSize, softMaxLseHost, softMaxLseSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *dqDevice;
    ACL_CHECK(aclrtMalloc((void **)(&dqDevice), qoSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *dkDevice;
    ACL_CHECK(aclrtMalloc((void **)(&dkDevice), kSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *dvDevice;
    ACL_CHECK(aclrtMalloc((void **)(&dvDevice), vSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // fag info
    FAGTiling::FAGInfo fagInfo;
    fagInfo.scaleValue = 1.0 / sqrt(qkHeadDim);
    fagInfo.keepProb = 1.0;
    fagInfo.maskType = 0;
    fagInfo.batch = batch;
    fagInfo.qSeqlen = qSeqlen;
    fagInfo.qHeadNum = qHeadNum;
    fagInfo.qkHeadDim = qkHeadDim;
    fagInfo.kvSeqlen = kvSeqlen;
    fagInfo.kvHeadNum = kvHeadNum;
    fagInfo.vHeadDim = vHeadDim;
    fagInfo.window_size_left = window_size_left;
    fagInfo.window_size_right = window_size_right;
    if (layout == "BNSD") {
        fagInfo.layout = BNSD;
    } else if (layout == "SBH") {
        fagInfo.layout = SBH;
    } else if (layout == "BSND") {
        fagInfo.layout = BSND;
    } else if (layout == "TND") {
        fagInfo.layout = TND;
        fagInfo.qSeqlenList = reinterpret_cast<int64_t *>(qSeqHost);
        fagInfo.kvSeqlenList = reinterpret_cast<int64_t *>(kvSeqHost);
    }
    fagInfo.isDeterministic = isDeterministic; // 预留，用于选择走确定性or非确定性

    // tilingdata size
    uint32_t tilingSize = sizeof(FAGTilingData);
    // get tiling
    void *tilingHost = nullptr;
    
    uint32_t blockDim = aicCoreNum;
    uint32_t aivNum = aivCoreNum;
    FAGTilingData fagTilingData;
    FAGTiling::GetFAGTilingParam(fagInfo, blockDim, aivNum, ubSize, fagTilingData);
    tilingHost = reinterpret_cast<void *>(&fagTilingData);
    uint8_t *tilingDevice;
    ACL_CHECK(aclrtMalloc((void **)(&tilingDevice), tilingSize + 16, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *workspaceDevice;
    ACL_CHECK(aclrtMalloc((void **)(&workspaceDevice), fagTilingData.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare hardware sync address
    uint64_t hardwareSyncAddr{0};
    ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)));

    constexpr uint32_t InputLayout = TND;
    constexpr uint32_t IS_DTM = 1;

    if (dataType == "bf16") {
        switch(qkHeadDim) {
            case 64: {
                FAG_BSND<0, DTemplateType::Aligned64, bfloat16_t, InputLayout, IS_DTM><<<blockDim, nullptr, stream>>>(
                    hardwareSyncAddr, doutDevice, qDevice, kDevice, vDevice, outDevice, nullptr,
                    nullptr/*attenMaskDevice*/, nullptr, nullptr, softMaxLseDevice, qSeqDevice, kvSeqDevice,
                    dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice); break; }
            case 128: {
                FAG_BSND<0, DTemplateType::Aligned128, bfloat16_t, InputLayout, IS_DTM><<<blockDim, nullptr, stream>>>(
                    hardwareSyncAddr, doutDevice, qDevice, kDevice, vDevice, outDevice, nullptr,
                    nullptr/*attenMaskDevice*/, nullptr, nullptr, softMaxLseDevice, qSeqDevice, kvSeqDevice,
                    dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice); break; }
            case 192: {
                FAG_BSND<0, DTemplateType::Aligned192, bfloat16_t, InputLayout, IS_DTM><<<blockDim, nullptr, stream>>>(
                    hardwareSyncAddr, doutDevice, qDevice, kDevice, vDevice, outDevice, nullptr,
                    nullptr/*attenMaskDevice*/, nullptr, nullptr, softMaxLseDevice, qSeqDevice, kvSeqDevice,
                    dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice); break; }
            case 256: {
                FAG_BSND<0, DTemplateType::Aligned256, bfloat16_t, InputLayout, IS_DTM><<<blockDim, nullptr, stream>>>(
                    hardwareSyncAddr, doutDevice, qDevice, kDevice, vDevice, outDevice, nullptr,
                    nullptr/*attenMaskDevice*/, nullptr, nullptr, softMaxLseDevice, qSeqDevice, kvSeqDevice,
                    dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice); break; }
            default: break;
        }
    } else {
        switch(qkHeadDim) {
            case 64: {
                FAG_BSND<0, DTemplateType::Aligned64, half, InputLayout, IS_DTM><<<blockDim, nullptr, stream>>>(
                    hardwareSyncAddr, doutDevice, qDevice, kDevice, vDevice, outDevice, nullptr,
                    nullptr/*attenMaskDevice*/, nullptr, nullptr, softMaxLseDevice, qSeqDevice, kvSeqDevice,
                    dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice); break; }
            case 128: {
                FAG_BSND<0, DTemplateType::Aligned128, half, InputLayout, IS_DTM><<<blockDim, nullptr, stream>>>(
                    hardwareSyncAddr, doutDevice, qDevice, kDevice, vDevice, outDevice, nullptr,
                    nullptr/*attenMaskDevice*/, nullptr, nullptr, softMaxLseDevice, qSeqDevice, kvSeqDevice,
                    dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice); break; }
            case 192: {
                FAG_BSND<0, DTemplateType::Aligned192, half, InputLayout, IS_DTM><<<blockDim, nullptr, stream>>>(
                    hardwareSyncAddr, doutDevice, qDevice, kDevice, vDevice, outDevice, nullptr,
                    nullptr/*attenMaskDevice*/, nullptr, nullptr, softMaxLseDevice, qSeqDevice, kvSeqDevice,
                    dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice); break; }
            case 256: {
                FAG_BSND<0, DTemplateType::Aligned256, half, InputLayout, IS_DTM><<<blockDim, nullptr, stream>>>(
                    hardwareSyncAddr, doutDevice, qDevice, kDevice, vDevice, outDevice, nullptr,
                    nullptr/*attenMaskDevice*/, nullptr, nullptr, softMaxLseDevice, qSeqDevice, kvSeqDevice,
                    dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice); break; }
            default: break;
        }
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // Copy the result from device to host and compare based on data type
    if (dataType == "bf16") {
        // Use uint16_t instead of bfloat16_t directly due to constructor issues
        vector<uint16_t> dqHost(qoSize / sizeof(uint16_t));
        ACL_CHECK(aclrtMemcpy(dqHost.data(), qoSize, dqDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));

        vector<float> goldenDqHost(qoSize / sizeof(bfloat16_t));
        const size_t goldenqoSize = qoSize * 2;
        ReadFile(dataPath + "/dq_golden.bin", goldenDqHost.data(), goldenqoSize);

        vector<uint16_t> dkHost(kSize / sizeof(uint16_t));
        ACL_CHECK(aclrtMemcpy(dkHost.data(), kSize, dkDevice, kSize, ACL_MEMCPY_DEVICE_TO_HOST));

        vector<float> goldenDkHost(kSize / sizeof(bfloat16_t));
        const size_t goldenkSize = kSize * 2;
        ReadFile(dataPath + "/dk_golden.bin", goldenDkHost.data(), goldenkSize);

        vector<uint16_t> dvHost(vSize / sizeof(uint16_t));
        ACL_CHECK(aclrtMemcpy(dvHost.data(), vSize, dvDevice, vSize, ACL_MEMCPY_DEVICE_TO_HOST));

        vector<float> goldenDvHost(vSize / sizeof(bfloat16_t));
        const size_t goldenvSize = vSize * 2;
        ReadFile(dataPath + "/dv_golden.bin", goldenDvHost.data(), goldenvSize);

        // Convert uint16_t (bf16) to float for comparison
        vector<float> dqHostFloat(dqHost.size());
        vector<float> dkHostFloat(dkHost.size());
        vector<float> dvHostFloat(dvHost.size());
        
        // Convert bf16 (stored as uint16_t) to float
        auto bf16_to_float = [](uint16_t bf16) -> float {
            // Extract sign, exponent, and mantissa from bf16
            uint32_t sign = (bf16 >> 15) & 0x1;
            uint32_t exponent = (bf16 >> 7) & 0xFF;
            uint32_t mantissa = bf16 & 0x7F;
            
            // Construct float (32-bit)
            uint32_t f32 = (sign << 31) | (exponent << 23) | (mantissa << 16);
            return *reinterpret_cast<float*>(&f32);
        };
        
        for (size_t i = 0; i < dqHost.size(); ++i) {
            dqHostFloat[i] = bf16_to_float(dqHost[i]);
        }
        
        for (size_t i = 0; i < dkHost.size(); ++i) {
            dkHostFloat[i] = bf16_to_float(dkHost[i]);
        }
        
        for (size_t i = 0; i < dvHost.size(); ++i) {
            dvHostFloat[i] = bf16_to_float(dvHost[i]);
        }

        // Compare the result
        vector<uint64_t> dqerrorIndices = Catlass::golden::CompareData(dqHostFloat, goldenDqHost, qoSize);
        if (dqerrorIndices.empty()) {
            cout << "Compare dq success." << endl;
        } else {
            cerr << "Compare dq failed. Error count: " << dqerrorIndices.size() << endl;
        }

        vector<uint64_t> dkerrorIndices = Catlass::golden::CompareData(dkHostFloat, goldenDkHost, kSize);
        if (dkerrorIndices.empty()) {
            cout << "Compare dk success." << endl;
        } else {
            cerr << "Compare dk failed. Error count: " << dkerrorIndices.size() << endl;
        }

        vector<uint64_t> errorIndices = Catlass::golden::CompareData(dvHostFloat, goldenDvHost, vSize);
        if (errorIndices.empty()) {
            cout << "Compare dv success." << endl;
        } else {
            cerr << "Compare dv failed. Error count: " << errorIndices.size() << endl;
        }
    } else {
        vector<fp16_t> dqHost(qoSize / sizeof(fp16_t));
        ACL_CHECK(aclrtMemcpy(dqHost.data(), qoSize, dqDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));

        vector<float> goldenDqHost(qoSize / sizeof(fp16_t));
        const size_t goldenqoSize = qoSize * 2;
        ReadFile(dataPath + "/dq_golden.bin", goldenDqHost.data(), goldenqoSize);

        vector<fp16_t> dkHost(kSize / sizeof(fp16_t));
        ACL_CHECK(aclrtMemcpy(dkHost.data(), kSize, dkDevice, kSize, ACL_MEMCPY_DEVICE_TO_HOST));

        vector<float> goldenDkHost(kSize / sizeof(fp16_t));
        const size_t goldenkSize = kSize * 2;
        ReadFile(dataPath + "/dk_golden.bin", goldenDkHost.data(), goldenkSize);

        vector<fp16_t> dvHost(vSize / sizeof(fp16_t));
        ACL_CHECK(aclrtMemcpy(dvHost.data(), vSize, dvDevice, vSize, ACL_MEMCPY_DEVICE_TO_HOST));

        vector<float> goldenDvHost(vSize / sizeof(fp16_t));
        const size_t goldenvSize = vSize * 2;
        ReadFile(dataPath + "/dv_golden.bin", goldenDvHost.data(), goldenvSize);

        // Compare the result
        vector<uint64_t> dqerrorIndices = Catlass::golden::CompareData(dqHost, goldenDqHost, qoSize);
        if (dqerrorIndices.empty()) {
            cout << "Compare dq success." << endl;
        } else {
            cerr << "Compare dq failed. Error count: " << dqerrorIndices.size() << endl;
        }

        vector<uint64_t> dkerrorIndices = Catlass::golden::CompareData(dkHost, goldenDkHost, kSize);
        if (dkerrorIndices.empty()) {
            cout << "Compare dk success." << endl;
        } else {
            cerr << "Compare dk failed. Error count: " << dkerrorIndices.size() << endl;
        }

        vector<uint64_t> errorIndices = Catlass::golden::CompareData(dvHost, goldenDvHost, vSize);
        if (errorIndices.empty()) {
            cout << "Compare dv success." << endl;
        } else {
            cerr << "Compare dv failed. Error count: " << errorIndices.size() << endl;
        }
    }

    // Free host memory allocations.
    FreeMem(doutHost, doutDevice);
    FreeMem(qHost, qDevice);
    FreeMem(kHost, kDevice);
    FreeMem(vHost, vDevice);
    FreeMem(outHost, outDevice);
    FreeMem(softMaxLseHost, softMaxLseDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);
    aclrtFree(dqDevice);
    aclrtFree(dkDevice);
    aclrtFree(dvDevice);
    
    if (layout=="TND") {
        FreeMem(qSeqHost, qSeqDevice);
        FreeMem(kvSeqHost, kvSeqDevice);
        aclrtFreeHost(qNtokens);
        aclrtFreeHost(kvNtokens);
    }

    // Destroy specified Stream and reset device.
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

/// Entry point to mla example.
int main(int argc, const char **argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}