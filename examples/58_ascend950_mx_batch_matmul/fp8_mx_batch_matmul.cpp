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

#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include "securec.h"

#include "catlass/gemm/kernel/mx_batched_matmul_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

using Options = GroupedGemmOptions;

static const std::string kDataRoot = "./examples/58_ascend950_mx_batch_matmul/data";

template <typename T>
bool SaveResult(const std::string& filename, const std::vector<T>& data)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }
    const size_t dataSize = data.size() * sizeof(T);
    if (dataSize <= 0) {
        file.close();
        return true;
    }

    std::vector<char> buffer(dataSize);
    memcpy(buffer.data(), data.data(), dataSize);

    file.write(buffer.data(), dataSize);
    if (!file) {
        file.close();
        return false;
    }

    file.close();
    return true;
}

static void Run(const Options& options)
{
    // Input shape constraints: batch_size/m/n/k must be greater than 0.
    // GroupedGemmOptions::Parse does not reject zero values, so check them here
    // before any division or device allocation.
    const uint32_t batchCount = options.problemCount;
    const uint32_t m = options.problemShape.m();
    const uint32_t n = options.problemShape.n();
    const uint32_t k = options.problemShape.k();
    if (batchCount == 0 || m == 0 || n == 0 || k == 0) {
        std::cerr << "batch_size, m, n and k must be greater than 0" << std::endl;
        return;
    }

    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(k);

    using ElementA = float8_e4m3_t;
    using ElementB = float8_e4m3_t;
    using ElementMxScale = float8_e8m0_t;
    using ElementC = bfloat16_t;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;
    using LayoutTagC = layout::RowMajor;
    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

    size_t lenA = tagA.Capacity() * batchCount;
    size_t lenB = tagB.Capacity() * batchCount;
    uint32_t mxScaleAlignedK = RoundUp<2>(mxScaleK);
    size_t lenMxScaleA = m * mxScaleAlignedK * batchCount;
    size_t lenMxScaleB = mxScaleAlignedK * n * batchCount;
    size_t lenC = tagC.Capacity() * batchCount;

    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeMxScaleA = lenMxScaleA * sizeof(ElementMxScale);
    size_t sizeMxScaleB = lenMxScaleB * sizeof(ElementMxScale);
    size_t sizeC = lenC * sizeof(ElementC);
    size_t sizeWorkspace{0};

    std::vector<int8_t> hostA(lenA);
    std::vector<int8_t> hostB(lenB);
    std::vector<int8_t> hostMxScaleA(lenMxScaleA);
    std::vector<int8_t> hostMxScaleB(lenMxScaleB);

    // Pre-declare device pointers for cleanup lambda
    uint8_t* deviceA{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceMxScaleA{nullptr};
    uint8_t* deviceMxScaleB{nullptr};
    uint8_t* deviceC{nullptr};
    uint8_t* deviceWorkspace{nullptr};

    const auto cleanupAndFinalize = [&]() {
        if (deviceA)
            ACL_CHECK(aclrtFree(deviceA));
        if (deviceB)
            ACL_CHECK(aclrtFree(deviceB));
        if (deviceMxScaleA)
            ACL_CHECK(aclrtFree(deviceMxScaleA));
        if (deviceMxScaleB)
            ACL_CHECK(aclrtFree(deviceMxScaleB));
        if (deviceC)
            ACL_CHECK(aclrtFree(deviceC));
        if (sizeWorkspace > 0 && deviceWorkspace)
            ACL_CHECK(aclrtFree(deviceWorkspace));
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };
    if (!ReadFile(kDataRoot + "/input/a_8.bin", hostA.data(), sizeA)) {
        cleanupAndFinalize();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/b_8.bin", hostB.data(), sizeB)) {
        cleanupAndFinalize();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/a_scale.bin", hostMxScaleA.data(), sizeMxScaleA)) {
        cleanupAndFinalize();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/b_scale.bin", hostMxScaleB.data(), sizeMxScaleB)) {
        cleanupAndFinalize();
        return;
    }

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceMxScaleA), sizeMxScaleA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceMxScaleA, sizeMxScaleA, hostMxScaleA.data(), sizeMxScaleA, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceMxScaleB), sizeMxScaleB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceMxScaleB, sizeMxScaleB, hostMxScaleB.data(), sizeMxScaleB, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag, 16>;
    using L1TileShape = Shape<Int<256>, Int<256>, Int<256>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<128>>;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, n);

    using TileCopy = Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagC, void>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = void;

    uint32_t taskNum = batchCount * CeilDiv(options.problemShape.m(), tla::get<0>(L1TileShape{})) *
                       CeilDiv(options.problemShape.n(), tla::get<1>(L1TileShape{}));
    uint32_t aicCoreUsed = min(aicCoreNum, taskNum);

    if (options.problemShape.m() > options.problemShape.n()) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        using MatmulKernel = Gemm::Kernel::MxBatchedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

        MatmulKernel::Arguments arguments{
            batchCount,     options.problemShape, deviceA,        layoutA,        deviceB, layoutB,
            deviceMxScaleA, layoutMxScaleA,       deviceMxScaleB, layoutMxScaleB, deviceC, layoutC};

        MatmulAdapter matmulOp;
        if (matmulOp.CanImplement(arguments) != Status::kSuccess) {
            std::cerr << "Cannot implement the arguments for MatmulKernel" << std::endl;
            cleanupAndFinalize();
            return;
        }
        sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicCoreUsed);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

        using MatmulKernel = Gemm::Kernel::MxBatchedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

        MatmulKernel::Arguments arguments{
            batchCount,     options.problemShape, deviceA,        layoutA,        deviceB, layoutB,
            deviceMxScaleA, layoutMxScaleA,       deviceMxScaleB, layoutMxScaleB, deviceC, layoutC};

        MatmulAdapter matmulOp;
        if (matmulOp.CanImplement(arguments) != Status::kSuccess) {
            std::cerr << "Cannot implement the arguments for MatmulKernel" << std::endl;
            cleanupAndFinalize();
            return;
        }
        sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicCoreUsed);
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::string resultFileName = kDataRoot + "/golden/result.bin";

    auto compareAndReport = [&](auto& hostC) -> bool {
        std::vector<float> hostGolden(lenC);
        std::string goldenFileName = kDataRoot + "/golden/expected_data.bin";
        if (!ReadFile(goldenFileName, hostGolden.data(), sizeof(float) * hostGolden.size())) {
            return false;
        }

        std::vector<uint64_t> errorIndices = [&]() {
            if constexpr (std::is_same_v<ElementC, bfloat16_t>) {
                constexpr float rtolBf16 = 1.0f / 128; // 1 bf16 ULP
                std::vector<uint64_t> errors;
                for (uint64_t i = 0; i < hostC.size(); ++i) {
                    float actual = static_cast<float>(hostC[i]);
                    float diff = std::fabs(actual - hostGolden[i]);
                    if (diff > rtolBf16 * std::max(1.0f, std::fabs(hostGolden[i]))) {
                        errors.push_back(i);
                    }
                }
                return errors;
            } else {
                return golden::CompareData(hostC, hostGolden, k);
            }
        }();
        if (errorIndices.empty()) {
            std::cout << "Compare success." << std::endl;
        } else {
            std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
        }
        return true;
    };

    if constexpr (!std::is_same_v<ElementC, bfloat16_t>) {
        std::vector<ElementC> hostC(lenC);
        ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
        SaveResult<ElementC>(resultFileName, hostC);
        std::cout << "Result saved to: " << resultFileName << std::endl;
        if (!compareAndReport(hostC)) {
            cleanupAndFinalize();
            return;
        }
    } else {
        std::vector<bfloat16> hostC(lenC);
        ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
        SaveResult<bfloat16>(resultFileName, hostC);
        std::cout << "Result saved to: " << resultFileName << std::endl;
        if (!compareAndReport(hostC)) {
            cleanupAndFinalize();
            return;
        }
    }

    cleanupAndFinalize();
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
