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

#include <cstdio>
#include <iostream>
#include <vector>
#include <cstdlib>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/block/block_scheduler_aswt.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m_per_tensor_per_channel_dequant_tla.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "tla/layout.hpp"
#include "helper.hpp"
#include "golden.hpp"

using namespace Catlass;
using namespace tla;

struct Options {
    const std::string HELPER =
        "48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant group_count m n k quant_mode [device_id]";

    uint32_t groupCount{1};
    GemmCoord problemShape{128, 128, 128};
    int32_t quantMode{0};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char** argv)
    {
        enum class ArgsIndex
        {
            GROUP_COUNT_INDEX = 1,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            QUANT_MODE_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX) ||
            argc < static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        groupCount = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::GROUP_COUNT_INDEX)]);
        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        problemShape.k() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::K_INDEX)]);
        quantMode = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::QUANT_MODE_INDEX)]);
        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }
        return 0;
    }
};

template <Gemm::Tile::ScaleGranularity QuantMode>
void RunTest(Options const& options)
{
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = half;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;

    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t problemCount = options.groupCount;
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n * problemCount;
    size_t lenC = static_cast<size_t>(m) * n;
    size_t lenScale = static_cast<size_t>(n);

    size_t sizeA = lenA * sizeof(int8_t);
    size_t sizeB = lenB * sizeof(int8_t);
    size_t sizeC = lenC * sizeof(fp16_t);
    size_t sizeScale = lenScale * sizeof(float);
    size_t sizeScaleCast = lenScale * sizeof(uint64_t);

    std::vector<ElementA> hostA(lenA);
    std::vector<ElementB> hostB(lenB);
    std::vector<float> hostPerChannelScale(lenScale);
    std::vector<uint64_t> hostPerChannelScaleCast;

    float scalePerTensor = 0.0f;
    golden::GenRandomData(scalePerTensor, 0.0, 1.0);
    golden::FillRandomData(hostA, -16, 16);
    golden::FillRandomData(hostB, -16, 16);
    golden::FillRandomData(hostPerChannelScale, 0.0, 1.0);
    // cast: fp32 -> uint64
    for (auto& scale : hostPerChannelScale) {
        uint64_t scaleCast = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&scale));
        hostPerChannelScaleCast.push_back(scaleCast);
    }

    auto groupList = golden::GenerateGroupList<int64_t>(m, problemCount);
    // 此处采取平均的方式，若需要随机，则注释该段
    for (int i = 0; i < problemCount; i++) {
        groupList[i] = (i + 1) * (m / problemCount);
    }
    printf("m:%u, k:%u ,n:%u, problemCount:%u, quantMode:%d\n", m, k, n, problemCount, options.quantMode);

    size_t sizeGroupList = problemCount * sizeof(int64_t);
    uint8_t* deviceGroupList{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceGroupList), sizeGroupList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceGroupList, sizeGroupList, groupList.data(), sizeGroupList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* devicePerChannelScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&devicePerChannelScale), sizeScaleCast, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        devicePerChannelScale, sizeScaleCast, hostPerChannelScaleCast.data(), sizeScaleCast,
        ACL_MEMCPY_HOST_TO_DEVICE));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    size_t sizeWorkspace = 0;
    uint8_t* deviceWorkspace{nullptr};

    LayoutTagA tagA{m, k};
    LayoutTagB tagB{k, n};
    LayoutTagC tagC{m, n};
    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutC = MakeLayoutFromTag(tagC);

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    constexpr bool useHF32 = false;
    using DispatchPolicy = Gemm::MmadDequant<ArchTag, enableUnitFlag, useHF32>;
    using L1TileShape = Shape<Int<256>, Int<256>, Int<256>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<64>>;

    using TileCopy = Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, void, false, QuantMode>;
    using BlockMmadTla = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = void;

    constexpr bool isGmm = true;
    using BlockScheduler = typename Gemm::Block::BlockSchedulerAswt<L1TileShape, L0TileShape, isGmm>;
    // kernel level
    using MatmulKernel =
        Gemm::Kernel::GroupedMatmulSliceMFixpipeDequantTla<BlockMmadTla, BlockEpilogue, BlockScheduler, int64_t>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    typename MatmulKernel::Arguments arguments{
        options.problemShape,
        problemCount,
        deviceGroupList,
        deviceA,
        layoutA,
        deviceB,
        layoutB,
        deviceC,
        layoutC,
        scalePerTensor,
        devicePerChannelScale,
    };

    // call a kernel
    MatmulAdapter matmul_op;
    // judge arguments can run
    matmul_op.CanImplement(arguments);
    // get workspace
    sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // initalize kernel argument
    matmul_op.Initialize(arguments, deviceWorkspace);
    matmul_op(stream, aicCoreNum);

    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::ComputeGroupedMatmulFixpipeDequant(
        options.problemShape, problemCount, groupList, hostA, tagA, hostB, tagB, hostGolden, tagC, options.quantMode,
        scalePerTensor, hostPerChannelScale);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k, groupList[problemCount - 1] * n);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
        std::cerr << "Compare failed. errorIndices[0]: " << errorIndices[0] << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceGroupList));
    ACL_CHECK(aclrtFree(devicePerChannelScale));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

static void Run(Options const& options)
{
    if (options.quantMode == 0) {
        RunTest<Gemm::Tile::ScaleGranularity::PER_TENSOR>(options);
    } else if (options.quantMode == 1) {
        RunTest<Gemm::Tile::ScaleGranularity::PER_CHANNEL>(options);
    } else {
        std::cerr << "[ERROR] quantMode must be '0' or '1'." << std::endl;
        return;
    }
}

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) == 0) {
        Run(options);
    }
    return 0;
}
