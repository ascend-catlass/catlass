/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iostream>
#include "golden.hpp"
#include "helper.hpp"
#include "catlass/layout/layout.hpp"
#include "dynamic_optimized_matmul.h"

static void Run(
    aclrtStream& stream, uint32_t m, uint32_t n, uint32_t k, LayoutTag layoutTagA, LayoutTag layoutTagB,
    PlatformInfo& platformInfo)
{
    LayoutTag layoutTagC = LayoutTag::TagRowMajor;
    TilingParams tilingParams{m, n, k, layoutTagA, layoutTagB, layoutTagC};
    DoTilingAndSelectKernel<fp16_t>(tilingParams, platformInfo);
    PrintTilingParams<fp16_t>(tilingParams, platformInfo);

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    std::vector<fp16_t> hostC(lenC);

    Catlass::golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    Catlass::golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);

    uint8_t *dA, *dB, *dC, *dW, *dTilingParams;

    ACL_CHECK(aclrtMalloc((void**)&dA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&dB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&dC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&dTilingParams, sizeof(TilingParams), ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(dA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    size_t workspaceSize = DynamicOptimizedMatmulGetWorkspace(tilingParams);
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc((void**)&dW, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    uint64_t hardwareSyncAddr{0};
    ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)));

    ACL_CHECK(aclrtMemcpy(
        dTilingParams, sizeof(TilingParams), &tilingParams, sizeof(TilingParams), ACL_MEMCPY_HOST_TO_DEVICE));

    ExecuteDynamicOptimizedMatmul(stream, hardwareSyncAddr, dA, dB, dC, dW, dTilingParams, tilingParams);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, dC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    Catlass::GemmCoord problemShape{m, n, k};
    if (layoutTagA == LayoutTag::TagRowMajor && layoutTagB == LayoutTag::TagRowMajor) {
        Catlass::layout::RowMajor layoutA{m, k};
        Catlass::layout::RowMajor layoutB{k, n};
        Catlass::layout::RowMajor layoutC{m, n};
        Catlass::golden::ComputeMatmul(problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
    } else if (layoutTagA == LayoutTag::TagRowMajor && layoutTagB == LayoutTag::TagColumnMajor) {
        Catlass::layout::RowMajor layoutA{m, k};
        Catlass::layout::ColumnMajor layoutB{k, n};
        Catlass::layout::RowMajor layoutC{m, n};
        Catlass::golden::ComputeMatmul(problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
    } else if (layoutTagA == LayoutTag::TagColumnMajor && layoutTagB == LayoutTag::TagRowMajor) {
        Catlass::layout::ColumnMajor layoutA{m, k};
        Catlass::layout::RowMajor layoutB{k, n};
        Catlass::layout::RowMajor layoutC{m, n};
        Catlass::golden::ComputeMatmul(problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
    } else {
        Catlass::layout::ColumnMajor layoutA{m, k};
        Catlass::layout::ColumnMajor layoutB{k, n};
        Catlass::layout::RowMajor layoutC{m, n};
        Catlass::golden::ComputeMatmul(problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);
    }
    std::vector<uint64_t> errorIndices = Catlass::golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(dA));
    ACL_CHECK(aclrtFree(dB));
    ACL_CHECK(aclrtFree(dC));
    ACL_CHECK(aclrtFree(dTilingParams));
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(dW));
    }
}

int main(int argc, const char** argv)
{
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " m n k layoutTagA layoutTagB deviceId\n"
                  << "  layoutTag: 0 for RowMajor, 1 for ColumnMajor" << std::endl;
        return -1;
    }

    const int m = std::atoi(argv[1]);
    const int n = std::atoi(argv[2]);
    const int k = std::atoi(argv[3]);
    const int layoutTagA = std::atoi(argv[4]);
    const int layoutTagB = std::atoi(argv[5]);
    const int deviceId = std::atoi(argv[6]);

    const bool layoutValid = (layoutTagA == 0 || layoutTagA == 1) && (layoutTagB == 0 || layoutTagB == 1);
    if (m <= 0 || n <= 0 || k <= 0 || !layoutValid || deviceId < 0) {
        std::cerr << "Invalid arguments: m, n, k must be positive, layoutTag must be 0 or 1, "
                  << "deviceId must be non-negative." << std::endl;
        return -1;
    }

    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclInit(nullptr));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    PlatformInfo platformInfo;

    Run(stream, static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
        static_cast<LayoutTag>(layoutTagA), static_cast<LayoutTag>(layoutTagB), platformInfo);

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
}
