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
#include "catlass/gemm/kernel/quant_multi_core_splitk_matmul_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/status.hpp"

#include "tla/layout.hpp"
#include "tla/tensor.hpp"

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

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenScale = static_cast<size_t>(n);
    size_t lenPerTokenScale = static_cast<size_t>(m);
    size_t lenD = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(int8_t);
    size_t sizeB = lenB * sizeof(int8_t);
    size_t sizeScale = lenScale * sizeof(float);
    size_t sizePerTokenScale = lenPerTokenScale * sizeof(float);
    size_t sizeD = lenD * sizeof(bfloat16_t);

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;
    using ElementScale = float;
    using ElementPerTokenScale = float;
    using ElementD = bfloat16_t;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;
    using LayoutTagC = layout::RowMajor;
    using LayoutTagScale = layout::RowMajor;
    using LayoutTagPerTokenScale = layout::RowMajor;
    using LayoutTagD = layout::RowMajor;

    auto tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    auto tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    auto tagC = LayoutTagC::MakeLayout<ElementC>(m, n);
    auto tagScale = LayoutTagScale::MakeLayout<ElementScale>(1, n);
    auto tagPerTokenScale = LayoutTagPerTokenScale::MakeLayout<ElementPerTokenScale>(1, m);
    auto tagD = LayoutTagD::MakeLayout<ElementD>(m, n);

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = MakeLayoutFromTag(tagC);
    auto layoutScale = MakeLayoutFromTag(tagScale);
    auto layoutPerTokenScale = MakeLayoutFromTag(tagPerTokenScale);
    auto layoutD = MakeLayoutFromTag(tagD);

    std::vector<int8_t> hostA(lenA);
    std::vector<int8_t> hostB(lenB);
    std::vector<float> hostScale(lenScale);
    std::vector<float> hostPerTokenScale(lenPerTokenScale);

    golden::FillRandomData(hostA, -16, 16);
    golden::FillRandomData(hostB, -16, 16);
    golden::FillRandomData(hostScale, 0.0, 1.0);
    golden::FillRandomData(hostPerTokenScale, 0.0, 1.0);

    uint8_t* deviceA{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceScale{nullptr};
    uint8_t* devicePerTokenScale{nullptr};
    uint8_t* deviceD{nullptr};
    uint8_t* deviceWorkspace{nullptr};

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceScale), sizeScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceScale, sizeScale, hostScale.data(), sizeScale, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(
        aclrtMalloc(reinterpret_cast<void**>(&devicePerTokenScale), sizePerTokenScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        devicePerTokenScale, sizePerTokenScale, hostPerTokenScale.data(), sizePerTokenScale,
        ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(deviceD, sizeD, 0, sizeD));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint64_t hardwareSyncAddr{0};
    ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)));

    using L1TileShape = Shape<_128, _256, _512>;
    using L0TileShape = Shape<_128, _256, _128>;

    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = true;
    constexpr bool useHF32 = false;
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;

    using TileCopy =
        Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequantTla<ubStages>;
    using ElementCompute = float;
    using EpilogueTileShape = MatrixShape<32, 256>;

    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMulTla<ArchTag, ElementCompute, EpilogueTileShape>;
    using TileBroadcastOneBlk = Epilogue::Tile::TileBroadcastOneBlkTla<ArchTag, ElementCompute, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMulTla<ArchTag, ElementCompute, EpilogueTileShape>;

    using EpilogueTileCopy = Epilogue::Tile::TileCopyDequantTla<
        ArchTag, ElementC, LayoutTagC, ElementScale, LayoutTagScale, ElementPerTokenScale, LayoutTagPerTokenScale,
        ElementD, LayoutTagD>;

    using EpilogueTileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, ElementC, ElementScale, ElementPerTokenScale, ElementD, TileRowBroadcastMul,
        TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, EpilogueTileCopy, EpilogueTileScheduler>;

    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    static constexpr uint32_t computeLength = 192 * 1024 / sizeof(ElementAccumulator);
    using ReduceAdd = Catlass::Gemm::Kernel::SplitkReduceAdd<ArchTag, ElementAccumulator, ElementC, 1, computeLength>;

    using BlockScheduler = typename Gemm::Block::SplitkGemmIdentityBlockSwizzle<3, 0>;

    using MatmulKernel =
        Gemm::Kernel::QuantMultiCoreSplitkMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler, ReduceAdd>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    typename MatmulKernel::Arguments arguments{options.problemShape, aicCoreNum, deviceA, deviceB, deviceScale,
                                               devicePerTokenScale,  deviceD,    layoutA, layoutB, layoutScale,
                                               layoutPerTokenScale,  layoutD};

    MatmulAdapter matmulOp;
    matmulOp.CanImplement(arguments);
    size_t sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);

    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(deviceWorkspace, sizeWorkspace, 0, sizeWorkspace));
    }

    matmulOp.Initialize(arguments, deviceWorkspace);
    matmulOp(stream, aicCoreNum, hardwareSyncAddr);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<bfloat16> hostD(lenD);
    ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenD);
    golden::QuantMatmul(
        options.problemShape, hostA, tagA, hostB, tagB, hostScale, tagScale, hostPerTokenScale, tagPerTokenScale,
        hostGolden, tagD);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceScale));
    ACL_CHECK(aclrtFree(devicePerTokenScale));
    ACL_CHECK(aclrtFree(deviceD));
    if (deviceWorkspace) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

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
