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

#include <iostream>
#include "catlass/gemm/kernel/weight_quant_a8w4_grouped_mx_matmul.hpp"

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

// Default data root when running from build output (e.g. output/bin), aligned with gen_data.py (WORKSPACE/data).
static const std::string kDataRoot = "./examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/data";

static void Run(const Options& options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // group_num、m、n、k
    uint32_t problemCount = options.problemCount;
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(k);

    using ElementA = float8_e4m3_t;
    using ElementB = float8_e4m3_t;
    using ElementPrologueB = float4_e2m1x2_t;
    using ElementMxScale = float8_e8m0_t;
    using ElementC = half;
    using ElementBias = void;

    using ElementGroupList = int64_t;

    using ElementBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;

    // basic layout
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::nZ;
    using LayoutPrologueB = layout::Weight4BitnZ;
    using LayoutMxScaleB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;

    // makeLayout
    LayoutA tagA = LayoutA::MakeLayout<ElementA>(m, k);
    LayoutPrologueB tagPrologueB = LayoutPrologueB::MakeLayout<ElementPrologueB>(k, n);
    LayoutC tagC = LayoutC::MakeLayout<ElementC>(m, n);

    static constexpr uint32_t MX_k_ALIGN = 2;
    static constexpr uint32_t SIZE_MAGNIFICATION = 2;

    // data length
    size_t lenA = tagA.Capacity();
    size_t lenPrologueB = tagPrologueB.Capacity() * problemCount;
    uint32_t mxScaleAlignedK = RoundUp<MX_k_ALIGN>(mxScaleK);
    size_t lenMxScaleA = static_cast<size_t>(m) * mxScaleAlignedK;
    size_t lenMxScaleB = static_cast<size_t>(mxScaleAlignedK) * n * problemCount;
    size_t lenC = tagC.Capacity();
    size_t lenBias = static_cast<size_t>(n);

    // data size(len * sizeof(element))
    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenPrologueB / SIZE_MAGNIFICATION;
    size_t sizeMxScaleA = lenMxScaleA * sizeof(ElementMxScale);
    size_t sizeMxScaleB = lenMxScaleB * sizeof(ElementMxScale);
    size_t sizeC = lenC * sizeof(ElementC);
    size_t sizeBias = lenBias * sizeof(ElementBiasType);
    size_t sizeGroupList = problemCount * sizeof(ElementGroupList);
    size_t sizeWorkspace;

    // host
    std::vector<int8_t> hostA(sizeA);
    std::vector<int8_t> hostB(sizeB);
    std::vector<int8_t> hostMxScaleA(lenMxScaleA);
    std::vector<int8_t> hostMxScaleB(lenMxScaleB);
    std::vector<ElementBiasType> hostBias(lenBias);
    std::vector<ElementGroupList> hostGroupList(sizeGroupList);

    const auto releaseAclEarly = [&]() {
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };
    // file read
    if (!ReadFile(kDataRoot + "/input/a_8.bin", hostA.data(), sizeA)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/b_4.bin", hostB.data(), sizeB)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/a_scale.bin", hostMxScaleA.data(), sizeMxScaleA)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/b_scale.bin", hostMxScaleB.data(), sizeMxScaleB)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/group_list.bin", hostGroupList.data(), sizeGroupList)) {
        releaseAclEarly();
        return;
    }
    if constexpr (!std::is_void_v<ElementBias>) {
        if (!ReadFile(kDataRoot + "/input/bias.bin", hostBias.data(), sizeBias)) {
            releaseAclEarly();
            return;
        }
    }

    // device
    uint8_t* deviceGroupList{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceGroupList), sizeGroupList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(
        aclrtMemcpy(deviceGroupList, sizeGroupList, hostGroupList.data(), sizeGroupList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceMxScaleA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceMxScaleA), sizeMxScaleA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceMxScaleA, sizeMxScaleA, hostMxScaleA.data(), sizeMxScaleA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceMxScaleB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceMxScaleB), sizeMxScaleB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceMxScaleB, sizeMxScaleB, hostMxScaleB.data(), sizeMxScaleB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceBias{nullptr};
    if constexpr (!std::is_void_v<ElementBias>) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBias), sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    uint8_t* deviceWorkspace{nullptr};

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // archtag uniflag
    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    static constexpr uint32_t L1_SCALE_FACTOR_K = 16;
    static constexpr uint32_t L1A_STAGES = 2;
    static constexpr uint32_t L1B_STAGES = 2;
    static constexpr uint32_t L0A_STAGES = 2;
    static constexpr uint32_t L0B_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = 1;

    // shape & type
    using L1TileShape = Shape<Int<256>, Int<256>, Int<256>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<128>>;
    using PrologueSrcType = Gemm::GemmType<ElementPrologueB, LayoutPrologueB>;
    using PrologueDstType = Gemm::GemmType<ElementB, LayoutB>;

    // DispatchPolicy
    using DispatchPolicyMmad = Gemm::MmadA8W4Mx<
        ArchTag, enableUnitFlag, false, L1_SCALE_FACTOR_K, L0C_STAGES, L1A_STAGES, L1B_STAGES, L0A_STAGES, L0B_STAGES>;
    using DispatchPolicyPrologue = Gemm::MxA8W4Prologue<ArchTag, L1B_STAGES>;

    // layout (tla)
    auto layoutA = tla::MakeLayout<ElementA, LayoutA>(m, k);
    auto layoutprologueB = tla::MakeLayout<ElementPrologueB, LayoutPrologueB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutMxScaleB, true>(mxScaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutC>(m, n);

    // tile
    using TileCopy = Gemm::Tile::PackedMxA8W4TileCopyTla<
        ArchTag, ElementA, LayoutA, ElementPrologueB, LayoutPrologueB, ElementB, LayoutB, ElementMxScale,
        decltype(layoutMxScaleA), ElementMxScale, decltype(layoutMxScaleB), ElementC, LayoutC, ElementBias, false,
        Gemm::Tile::ScaleGranularity::PER_TENSOR>;

    // BlockMmad
    using BlockMmad = Gemm::Block::BlockMmadA8W4Mx<
        DispatchPolicyMmad, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;

    using BlockPrologue =
        Gemm::Block::BlockPrologue<DispatchPolicyPrologue, PrologueSrcType, PrologueDstType, L1TileShape, TileCopy>;

    // Epilogue
    using BlockEpilogue = void;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    using MatmulKernel =
        Gemm::Kernel::A8W4GroupedMxMatmul<BlockMmad, BlockPrologue, BlockEpilogue, BlockScheduler, ElementGroupList>;

    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    MatmulKernel::Arguments arguments{
        options.problemShape, options.problemCount, deviceGroupList, deviceA,        layoutA, deviceB, layoutprologueB,
        deviceMxScaleA,       layoutMxScaleA,       deviceMxScaleB,  layoutMxScaleB, deviceC, layoutC, deviceBias};

    uint32_t taskNum = CeilDiv(options.problemShape.m(), tla::get<0>(L1TileShape{})) *
                       CeilDiv(options.problemShape.n(), tla::get<1>(L1TileShape{}));
    uint32_t aicCoreUsed = min(aicCoreNum, taskNum);

    MatmulAdapter matmulOp;
    matmulOp.CanImplement(arguments);
    sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(arguments, deviceWorkspace);
    matmulOp(stream, aicCoreUsed);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    std::string outputFileName = kDataRoot + "/golden/expected_data.bin";
    if (!ReadFile(outputFileName, hostGolden.data(), sizeof(float) * hostGolden.size())) {
        ACL_CHECK(aclrtFree(deviceA));
        ACL_CHECK(aclrtFree(deviceB));
        ACL_CHECK(aclrtFree(deviceMxScaleA));
        ACL_CHECK(aclrtFree(deviceMxScaleB));
        ACL_CHECK(aclrtFree(deviceC));
        ACL_CHECK(aclrtFree(deviceGroupList));
        if constexpr (!std::is_void_v<ElementBias>) {
            ACL_CHECK(aclrtFree(deviceBias));
        }
        if (sizeWorkspace > 0) {
            ACL_CHECK(aclrtFree(deviceWorkspace));
        }
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return;
    }

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;

        for (uint32_t i = 0; i < 10; ++i) {
            std::cout << "Index: " << errorIndices[i] << " npu:" << (float)hostC[errorIndices[i]]
                      << " cpu:" << hostGolden[errorIndices[i]] << std::endl;
        }

        std::cout << std::endl;

        for (uint32_t i = 0; i < 20; ++i) {
            std::cout << "Index: " << i << " npu:" << (float)hostC[i] << " cpu:" << hostGolden[i] << std::endl;
        }

        uint32_t* hostAInt = reinterpret_cast<uint32_t*>(hostA.data());
        uint32_t* hostBInt = reinterpret_cast<uint32_t*>(hostB.data());
        uint32_t* hostMxScaleAInt = reinterpret_cast<uint32_t*>(hostMxScaleA.data());
        uint32_t* hostMxScaleBInt = reinterpret_cast<uint32_t*>(hostMxScaleB.data());
        for (uint32_t i = 0; i < 10; ++i) {
            std::cout << "index: " << i << " hostA: " << hostAInt[i] << " hostB: " << hostBInt[i]
                      << " hostMxScaleA: " << hostMxScaleAInt[i] << " hostMxScaleB: " << hostMxScaleBInt[i]
                      << std::endl;
        }
    }
    // for (uint32_t i = 0; i < 20; ++i){
    //     std::cout<< "Index: " << i << " npu:" << (float) hostC[i]<< " cpu:" << hostGolden[i]<<std::endl;
    // }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceMxScaleA));
    ACL_CHECK(aclrtFree(deviceMxScaleB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceGroupList));
    if constexpr (!std::is_void_v<ElementBias>) {
        ACL_CHECK(aclrtFree(deviceBias));
    }
    if (sizeWorkspace > 0) {
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
    if (options.problemCount == 0 || options.problemShape.m() == 0 || options.problemShape.n() == 0 ||
        options.problemShape.k() == 0) {
        std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << ": problem_count, m, n and k must be greater than 0"
                  << std::endl;
        return -1;
    }
    Run(options);
    return 0;
}
