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

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/basic_matmul_tla_visitor.hpp"
#include "catlass/epilogue/fusion/fusion.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

using Options = GemmOptions;

static void Run(const Options& options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    // Define element types and layout tags (TLA version)
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;

    // Create layouts for capacity calculation (using layout tag's MakeLayout)
    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

    // Compute the length of each matrix and the size of each buffer
    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity();
    size_t lenD = tagC.Capacity();

    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeD = lenD * sizeof(ElementC);

    // Prepare input data A and B
    std::vector<ElementA> hostA(lenA);
    std::vector<ElementB> hostB(lenB);
    golden::FillRandomData<ElementA>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<ElementB>(hostB, -5.0f, 5.0f);

    // Allocate device memory and copy data from host to device
    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Define ArchTag
    using ArchTag = Arch::Ascend950;

    // Block level, define BlockMmad
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag>;
    using L1TileShape = Shape<Int<256>, Int<256>, Int<128>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<32>>;

    // Create TLA layouts for kernel usage
    auto layoutA = MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopy =
        Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

    // 定义 EVG: D = LeakyRelu(C)
    // C 是 workspace（A*B 的结果，D 是最终输出（LeakyRelu(C) 的结果）

    constexpr uint32_t evgUbNodes = 2;  // AccLoad + Compute；Store 不占
    constexpr uint32_t evgUbStages = 2; // epilogue 双缓冲
    constexpr uint32_t computeLength = RoundDown(
        ArchTag::UB_SIZE / evgUbNodes / evgUbStages / sizeof(ElementC),
        BYTE_PER_C0); // 每槽元素上限，向下取 BYTE_PER_C0 整数倍

    using LayoutC = decltype(layoutC);
    using EVG = Epilogue::Fusion::TreeVisitor<
        Epilogue::Fusion::VisitorAuxStore<ElementC, LayoutC>,
        Epilogue::Fusion::TreeVisitor<
            Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::LeakyRelu, ElementC, ElementC>,
            Epilogue::Fusion::VisitorAccLoad<ElementC> // 加载 C (workspace)
            >>;

    // Block level, define BlockEpilogue with EVG
    using BlockEpilogue =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueVisitor<>, ArchTag, Int<computeLength>, EVG, ElementC>;

    constexpr float leakyReluAlpha = 0.1f;

    // 准备 EVG Arguments - 使用 TLA layout 对象
    typename EVG::Arguments evg_args{{{}, {{leakyReluAlpha}}}, {deviceD, layoutC}};

    std::vector<ElementC> hostD(lenD);
    if (m > n) {
        // Define BlockScheduler
        // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        // Kernel level (TLA version)
        using MatmulKernel = Gemm::Kernel::BasicMatmulTlaVisitor<BlockMmad, BlockEpilogue, BlockScheduler>;
        // Prepare params
        typename MatmulKernel::Arguments arguments{
            options.problemShape, deviceA, layoutA, deviceB, layoutB, nullptr, {}, nullptr, evg_args};
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmulOp;
        size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
        uint8_t* deviceWorkspace{nullptr};
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicCoreNum);
        ACL_CHECK(aclrtSynchronizeStream(stream));
        if (sizeWorkspace > 0) {
            ACL_CHECK(aclrtFree(deviceWorkspace));
        }

        // Copy the result from device to host
        ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));
    } else {
        // Define BlockScheduler
        // Swizzle offset is 3 and direction is 1.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        // Kernel level (TLA version)
        using MatmulKernel = Gemm::Kernel::BasicMatmulTlaVisitor<BlockMmad, BlockEpilogue, BlockScheduler>;
        // Prepare params
        typename MatmulKernel::Arguments arguments{
            options.problemShape, deviceA, layoutA, deviceB, layoutB, nullptr, {}, nullptr, evg_args};
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmulOp;
        size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
        uint8_t* deviceWorkspace{nullptr};
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        matmulOp.Initialize(arguments, deviceWorkspace);
        matmulOp(stream, aicCoreNum);
        ACL_CHECK(aclrtSynchronizeStream(stream));
        if (sizeWorkspace > 0) {
            ACL_CHECK(aclrtFree(deviceWorkspace));
        }

        // Copy the result from device to host
        ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));
    }

    // Compute the golden result
    std::vector<float> hostGolden(lenD);
    golden::ComputeMatmulElemWiseLeakyRelu(
        options.problemShape, hostA, tagA, hostB, tagB, hostGolden, tagC, leakyReluAlpha);

    // Compare the result
    std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceD));

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
