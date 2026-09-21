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

#include <cstdint>
#include <iostream>
#include <vector>

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/block/block_epilogue_dual_silu_mul.h"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_mmad_dual_shared_a.h"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

#include "catlass/gemm/kernel/dual_matmul_silu_mul_tla.h"

using namespace Catlass;
using namespace tla;

namespace {

using Options = GemmOptions;

std::vector<float> ComputeGolden(
    const std::vector<fp16_t>& x, const std::vector<fp16_t>& b0, const std::vector<fp16_t>& b1, uint32_t m, uint32_t n,
    uint32_t k)
{
    std::vector<float> output(static_cast<size_t>(m) * n);
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            float d0 = 0.0F;
            float d1 = 0.0F;
            for (uint32_t inner = 0; inner < k; ++inner) {
                float valueX = static_cast<float>(x[static_cast<size_t>(row) * k + inner]);
                d0 += valueX * static_cast<float>(b0[static_cast<size_t>(col) * k + inner]);
                d1 += valueX * static_cast<float>(b1[static_cast<size_t>(col) * k + inner]);
            }
            float silu = d0 / (1.0F + std::exp(-d0));
            output[static_cast<size_t>(row) * n + col] = silu * d1;
        }
    }
    return output;
}

int Run(const Options& options)
{
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    size_t lenX = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(n) * k;
    size_t lenD = static_cast<size_t>(m) * n;

    std::vector<fp16_t> hostX(lenX);
    std::vector<fp16_t> hostB0(lenB);
    std::vector<fp16_t> hostB1(lenB);
    std::vector<fp16_t> hostD(lenD);
    Catlass::golden::FillRandomData<fp16_t>(hostX, -1.0F, 1.0F);
    Catlass::golden::FillRandomData<fp16_t>(hostB0, -1.0F, 1.0F);
    Catlass::golden::FillRandomData<fp16_t>(hostB1, -1.0F, 1.0F);
    std::vector<float> goldenD = ComputeGolden(hostX, hostB0, hostB1, m, n, k);

    aclrtStream stream{nullptr};
    uint8_t* deviceX{nullptr};
    uint8_t* deviceB0{nullptr};
    uint8_t* deviceB1{nullptr};
    uint8_t* deviceD{nullptr};
    size_t sizeX = lenX * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeD = lenD * sizeof(fp16_t);

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB0), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB1), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB0, sizeB, hostB0.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB1, sizeB, hostB1.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ElementA = half;
    using ElementB = half;
    using ElementC = float;
    using ElementD = half;
    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;
    using LayoutTagC = layout::RowMajor;
    using LayoutTagD = layout::RowMajor;
    using ArchTag = Arch::Ascend950;
    using BaseDispatchPolicy = Gemm::MmadPingpong<ArchTag, true, false, 1, false, 2, 2, 2, 2>;
    using DispatchPolicy = Gemm::MmadDualSharedA<BaseDispatchPolicy>;
    using L1TileShape = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShape = Shape<Int<128>, Int<128>, Int<64>>;
    constexpr auto copyMode = Gemm::Tile::CopyL0CToUBMode::SPLIT_M;
    using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, void, copyMode>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogueDualSiluMul<ElementD, ElementC, 8192>;
    using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<1, 0>;
    using MatmulKernel = Gemm::Kernel::DualMatmulSiluMulTla<
        BlockMmad, BlockEpilogue, BlockScheduler, ElementD, decltype(MakeLayout<ElementD, LayoutTagD>(1U, 1U))>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    auto layoutA = MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutD = MakeLayout<ElementD, LayoutTagD>(m, n);
    MatmulKernel::Arguments arguments{GemmCoord{m, n, k}, deviceX, layoutA, deviceB0,
                                      deviceB1,           layoutB, deviceD, layoutD};
    if (MatmulAdapter::CanImplement(arguments) == Status::kInvalid) {
        std::cerr << "The example configuration cannot implement the requested shape." << std::endl;
        ACL_CHECK(aclrtFree(deviceX));
        ACL_CHECK(aclrtFree(deviceB0));
        ACL_CHECK(aclrtFree(deviceB1));
        ACL_CHECK(aclrtFree(deviceD));
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return 1;
    }
    MatmulAdapter matmulOp;
    matmulOp.Initialize(arguments);
    uint32_t taskNum = CeilDiv(m, MatmulKernel::L1_TILE_M) * CeilDiv(n, MatmulKernel::OUTPUT_TILE_N);
    matmulOp(stream, std::min(aicCoreNum, taskNum));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<uint64_t> errorIndices = Catlass::golden::CompareData(hostD, goldenD, k);
    bool passed = errorIndices.empty();
    if (passed) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceB0));
    ACL_CHECK(aclrtFree(deviceB1));
    ACL_CHECK(aclrtFree(deviceD));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
    return passed ? 0 : 1;
}

} // namespace

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return 1;
    }
    return Run(options);
}
