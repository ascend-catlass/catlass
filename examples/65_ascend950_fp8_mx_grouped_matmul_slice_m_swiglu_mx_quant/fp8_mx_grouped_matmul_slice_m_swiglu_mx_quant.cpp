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

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/grouped_mx_matmul_slice_m_swiglu_mx_quant_tla.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "tla/layout.hpp"
#include "helper.hpp"
#include "golden.hpp"

using namespace Catlass;
using namespace tla;

using Options = GroupedGemmOptions;

static const std::string kExampleRoot = "./examples/65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant";
static const std::string kDataRoot = kExampleRoot + "/data";

static void Run(const Options& options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t N = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t groupCount = options.problemCount;
    uint32_t N_half = N / 2;
    uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(k);
    uint32_t mxScaleAlignedK = RoundUp<2>(mxScaleK);

    using ElementA = float8_e4m3_t;
    using ElementB = float8_e4m3_t;
    using ElementMxScale = float8_e8m0_t;
    using ElementC = float;
    using ElementGroupList = int64_t;
    using ElementQ = float8_e4m3_t;
    using ElementQScale = float8_e8m0_t;
    using ElementGluRes = bfloat16_t;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;
    using LayoutTagQ = layout::RowMajor;

    LayoutTagA tagA = LayoutTagA::template MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::template MakeLayout<ElementB>(k, N);
    LayoutTagQ tagQ = LayoutTagQ::template MakeLayout<ElementQ>(m, N_half);

    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity() * groupCount;
    size_t lenMxScaleA = static_cast<size_t>(m) * mxScaleAlignedK;
    size_t lenMxScaleB = static_cast<size_t>(mxScaleAlignedK) * N * groupCount;
    size_t lenQ = tagQ.Capacity();
    // Q-scale row stride matches what the epilogue writes: ceil(N/2, 32) * 2 bytes per row.
    size_t qScaleRowStride = CeilDiv<MX_SCALE_GROUP_NUM>(N_half);
    size_t lenQScale = static_cast<size_t>(m) * qScaleRowStride;

    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeMxScaleA = lenMxScaleA * sizeof(ElementMxScale);
    size_t sizeMxScaleB = lenMxScaleB * sizeof(ElementMxScale);
    size_t sizeGroupList = groupCount * sizeof(ElementGroupList);
    size_t sizeQ = lenQ * sizeof(ElementQ);
    size_t sizeQScale = lenQScale * sizeof(ElementQScale);
    size_t sizeWorkspace{0};

    std::vector<int8_t> hostA(sizeA);
    std::vector<int8_t> hostB(sizeB);
    std::vector<int8_t> hostMxScaleA(lenMxScaleA);
    std::vector<int8_t> hostMxScaleB(lenMxScaleB);

    uint8_t* deviceA{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceMxScaleA{nullptr};
    uint8_t* deviceMxScaleB{nullptr};
    uint8_t* deviceQ{nullptr};
    uint8_t* deviceQScale{nullptr};
    uint8_t* deviceGroupList{nullptr};
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
        if (deviceQ)
            ACL_CHECK(aclrtFree(deviceQ));
        if (deviceQScale)
            ACL_CHECK(aclrtFree(deviceQScale));
        if (deviceGroupList)
            ACL_CHECK(aclrtFree(deviceGroupList));
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
    if (!ReadFile(kDataRoot + "/input/b_8_trans.bin", hostB.data(), sizeB)) {
        cleanupAndFinalize();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/a_scale.bin", hostMxScaleA.data(), sizeMxScaleA)) {
        cleanupAndFinalize();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/b_scale_trans.bin", hostMxScaleB.data(), sizeMxScaleB)) {
        cleanupAndFinalize();
        return;
    }

    std::vector<ElementGroupList> groupList(groupCount);
    if (!ReadFile(kDataRoot + "/input/group_list.bin", groupList.data(), sizeGroupList)) {
        cleanupAndFinalize();
        return;
    }

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceGroupList), sizeGroupList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceGroupList, sizeGroupList, groupList.data(), sizeGroupList, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceMxScaleA), sizeMxScaleA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceMxScaleA, sizeMxScaleA, hostMxScaleA.data(), sizeMxScaleA, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceMxScaleB), sizeMxScaleB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceMxScaleB, sizeMxScaleB, hostMxScaleB.data(), sizeMxScaleB, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceQ), sizeQ, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceQScale), sizeQScale, ACL_MEM_MALLOC_HUGE_FIRST));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 9201
    using ArchTag = Arch::Ascend960DT;
#elif defined(CATLASS_ARCH) && CATLASS_ARCH == 9202
    using ArchTag = Arch::Ascend960PR;
#else
    using ArchTag = Arch::Ascend950;
#endif
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag, 16>;
    using EpilogueDispatchPolicy = Epilogue::BlockEpilogueSwigluMxQuant;
    using L1TileShape = Shape<Int<128>, Int<256>, Int<256>>;
    using L0TileShape = Shape<Int<128>, Int<256>, Int<128>>;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, N);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, N);
    auto layoutQ = tla::MakeLayout<ElementQ, LayoutTagQ>(m, N_half);
    auto layoutQScale =
        tla::MakeMxScaleLayout<ElementQScale, LayoutTagQ, false>(m, CeilDiv<MX_SCALE_GROUP_NUM>(N_half));

    using TileCopy = Gemm::Tile::PackedMxTileCopyTlaToUB<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagQ, void,
#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202)
        Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
#else
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
#endif
    using BlockMmad = Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, L0TileShape, ElementC, ElementC, ElementGluRes, ElementQ, ElementQScale>;
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

    using MatmulKernel = Gemm::Kernel::GroupedMxMatmulSliceMSwigluMxQuantTla<
        BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList, decltype(layoutQ), decltype(layoutQScale)>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    typename BlockEpilogue::Params epilogueParams;
    epilogueParams.baseM = m;
    epilogueParams.baseN = N_half;
    epilogueParams.baseK = k;

    typename MatmulKernel::Arguments arguments{
        options.problemShape, options.problemCount, deviceGroupList, deviceA,        layoutA, deviceB, layoutB,
        deviceMxScaleA,       layoutMxScaleA,       deviceMxScaleB,  layoutMxScaleB, deviceQ, layoutQ, deviceQScale,
        layoutQScale,         epilogueParams};

    MatmulAdapter matmulOp;
    if (matmulOp.CanImplement(arguments) != Status::kSuccess) {
        std::cerr << "Cannot implement the arguments for MatmulKernel" << std::endl;
        cleanupAndFinalize();
        return;
    }
    sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(arguments, deviceWorkspace);
    matmulOp(stream, aicCoreNum);

    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<uint8_t> hostQ(sizeQ);
    std::vector<uint8_t> hostQScale(sizeQScale);
    ACL_CHECK(aclrtMemcpy(hostQ.data(), sizeQ, deviceQ, sizeQ, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostQScale.data(), sizeQScale, deviceQScale, sizeQScale, ACL_MEMCPY_DEVICE_TO_HOST));

    {
        std::ofstream qFile(kDataRoot + "/golden/result.bin", std::ios::binary);
        qFile.write(reinterpret_cast<const char*>(hostQ.data()), sizeQ);
    }
    {
        std::ofstream qsFile(kDataRoot + "/golden/result_scale.bin", std::ios::binary);
        qsFile.write(reinterpret_cast<const char*>(hostQScale.data()), sizeQScale);
    }

    std::string cmd = "python3 " + kExampleRoot + "/compare.py " + std::to_string(groupCount) + " " +
                      std::to_string(m) + " " + std::to_string(N) + " " + std::to_string(k);
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Compare script failed with return code: " << ret << std::endl;
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
