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

    // Prepare input data A, B, and X
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

    // The data of X is stored on deviceD to save storage space
    uint8_t* deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Define ArchTag
    using ArchTag = Arch::Ascend950;

    // Block level, define BlockMmad (TLA version)
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

    // 定义 EVG: D = Tanh( C ) = (e^X - e^-X) / (e^X + e^-X) = (e^2X - 1) / (e^2X + 1)
    using LayoutC = decltype(layoutC);

    // 与 AscendC Tanh 一致：
    constexpr float tanhClipMax = 8.8f;
    constexpr float tanhClipMin = -8.8f;
    constexpr float tanhDoubleX = 2.0f;
    constexpr float tanhExpMinusOne = -1.0f;
    constexpr float tanhExpPlusOne = 1.0f;

    constexpr uint32_t evgUbNodes = 8;  // 占 UB 的 Visitor 数（AccLoad + 7×Compute；Store 不占）
    constexpr uint32_t evgUbStages = 2; // epilogue 双缓冲
    constexpr uint32_t computeLength = RoundDown(
        ArchTag::UB_SIZE / evgUbNodes / evgUbStages / sizeof(ElementC),
        BYTE_PER_C0); // 每槽元素上限，向下取 BYTE_PER_C0 整数倍

    // 节点顺序：
    // Compute1/2 将 C clip 到 [tanhClipMin, tanhClipMax]，避免 |C| 过大时 exp(2C) 浮点溢出
    // 0-AccLoad, 1-Compute1(Mins tanhClipMax), 2-Compute2(Maxs tanhClipMin), 3-Compute3(2X), 4-Compute4(Exp(2X)),
    // 5-Compute5(Exp(2X) - 1), 6-Compute6(Exp(2X) + 1), 7-Compute7(Compute5 / Compute6), 8-Store
    using Edges = tla::tuple<
        tla::seq<>,     // 0: AccLoad 无子节点
        tla::seq<0>,    // 1: 依赖 AccLoad-->Mins(tanhClipMax)
        tla::seq<1>,    // 2: 依赖 Compute1-->Maxs(tanhClipMin)
        tla::seq<2>,    // 3: 依赖 Compute2-->2X
        tla::seq<3>,    // 4: 依赖 Compute3-->Exp(2X)
        tla::seq<4>,    // 5: 依赖 Compute4-->(Exp(2X) - 1)
        tla::seq<4>,    // 6: 依赖 Compute4-->(Exp(2X) + 1)
        tla::seq<5, 6>, // 7: 依赖 Compute5 与 Compute6-->(Compute5 / Compute6)
        tla::seq<7>     // 8: Store 依赖 Compute7
        >;

    using EVG = Epilogue::Fusion::TopologicalVisitor<
        Edges, Epilogue::Fusion::VisitorAccLoad<ElementC>,
        Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Mins, ElementC, ElementC>,
        Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Maxs, ElementC, ElementC>,
        Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Muls, ElementC, ElementC>,
        Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Exp, ElementC>,
        Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Adds, ElementC, ElementC>,
        Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Adds, ElementC, ElementC>,
        Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Div, ElementC>,
        Epilogue::Fusion::VisitorAuxStore<ElementC, LayoutC>>;

    // Block level, define BlockEpilogue with EVG
    using BlockEpilogue =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueVisitor<>, ArchTag, Int<computeLength>, EVG, ElementC>;

    // 准备 EVG Arguments - 使用 TLA layout 对象
    typename EVG::Arguments evg_args{
        {}, {{tanhClipMax}},   {{tanhClipMin}}, {{tanhDoubleX}}, {}, {{tanhExpMinusOne}}, {{tanhExpPlusOne}},
        {}, {deviceD, layoutC}};

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
    golden::ComputeMatmulElemWiseTanh(options.problemShape, hostA, tagA, hostB, tagB, hostGolden, tagC);

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
