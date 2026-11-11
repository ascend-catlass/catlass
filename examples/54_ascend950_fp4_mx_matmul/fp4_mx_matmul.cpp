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

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include <string>

#include "catlass/gemm/kernel/mx_matmul_tla.hpp"

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

using Options = GemmOptions;

// Default data root when running from build output (e.g. output/bin), aligned with gen_data.py (WORKSPACE/data).
static const std::string kDataRoot = "../../examples/54_ascend950_fp4_mx_matmul/data";

static void Run(const Options& options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(k);

    using ElementA = float4_e2m1x2_t;
    using ElementB = float4_e2m1x2_t;
    using ElementMxScale = float8_e8m0_t;
    using ElementC = float;
    // if no bias, set ElementBias to void
    using ElementBias = void;

    using ElementBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;
    using LayoutTagC = layout::RowMajor;
    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity();
    // compute mxScale len, k must be multiples of 2
    uint32_t mxScaleAlignedK = RoundUp<2>(mxScaleK);
    size_t lenMxScaleA = m * mxScaleAlignedK;
    size_t lenMxScaleB = mxScaleAlignedK * n;
    size_t lenC = tagC.Capacity();
    size_t lenBias = static_cast<size_t>(n);

    size_t sizeA = lenA / 2;
    size_t sizeB = lenB / 2;
    size_t sizeMxScaleA = lenMxScaleA * sizeof(ElementMxScale);
    size_t sizeMxScaleB = lenMxScaleB * sizeof(ElementMxScale);
    size_t sizeC = lenC * sizeof(ElementC);
    size_t sizeBias = lenBias * sizeof(ElementBiasType);
    size_t sizeWorkspace;

    std::vector<int8_t> hostA(sizeA);
    std::vector<int8_t> hostB(sizeB);
    std::vector<int8_t> hostMxScaleA(lenMxScaleA);
    std::vector<int8_t> hostMxScaleB(lenMxScaleB);
    std::vector<ElementBiasType> hostBias(lenBias);

    const auto releaseAclEarly = [&]() {
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };
    if (!ReadFile(kDataRoot + "/input/a_8.bin", hostA.data(), sizeA)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/b_8.bin", hostB.data(), sizeB)) {
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
    if constexpr (!std::is_void_v<ElementBias>) {
        if (!ReadFile(kDataRoot + "/input/bias.bin", hostBias.data(), sizeBias)) {
            releaseAclEarly();
            return;
        }
    }

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

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag, 16>;
    using L1TileShape = Shape<Int<256>, Int<256>, Int<512>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<256>>;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopy = Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagC, ElementBias>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
    using BlockEpilogue = void;

    uint32_t taskNum = CeilDiv(options.problemShape.m(), tla::get<0>(L1TileShape{})) *
                       CeilDiv(options.problemShape.n(), tla::get<1>(L1TileShape{}));
    uint32_t aicCoreUsed = min(aicCoreNum, taskNum);

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    using MatmulKernel = Gemm::Kernel::MxMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;

    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    MatmulKernel::Arguments arguments{
        options.problemShape, deviceA,        layoutA,        deviceB, layoutB, deviceMxScaleA,
        layoutMxScaleA,       deviceMxScaleB, layoutMxScaleB, deviceC, layoutC, deviceBias};

    MatmulAdapter matmulOp;
    if (matmulOp.CanImplement(arguments) != Status::kSuccess) {
        std::cerr << "[ERROR]matmul op cannot be implemented" << std::endl;
        ACL_CHECK(aclrtFree(deviceA));
        ACL_CHECK(aclrtFree(deviceB));
        ACL_CHECK(aclrtFree(deviceMxScaleA));
        ACL_CHECK(aclrtFree(deviceMxScaleB));
        ACL_CHECK(aclrtFree(deviceC));
        if constexpr (!std::is_void_v<ElementBias>) {
            ACL_CHECK(aclrtFree(deviceBias));
        }
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return;
    }
    sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(arguments, deviceWorkspace);
    matmulOp(stream, aicCoreUsed);

    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<ElementC> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    std::string outputFileName = kDataRoot + "/golden/expected_data.bin";
    if (!ReadFile(outputFileName, hostGolden.data(), sizeof(float) * hostGolden.size())) {
        ACL_CHECK(aclrtFree(deviceA));
        ACL_CHECK(aclrtFree(deviceB));
        ACL_CHECK(aclrtFree(deviceMxScaleA));
        ACL_CHECK(aclrtFree(deviceMxScaleB));
        ACL_CHECK(aclrtFree(deviceC));
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
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceMxScaleA));
    ACL_CHECK(aclrtFree(deviceMxScaleB));
    ACL_CHECK(aclrtFree(deviceC));
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
    Run(options);
    return 0;
}
