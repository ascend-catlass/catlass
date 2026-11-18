/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * --------------------------------------------------------------------------
 * Example 63: 二级量化 + MX FP4 Batch Matmul 单 kernel 路径
 *
 * 当前样例的单 kernel 路径:
 *   - 输入为 fp16/bf16 A/B,输出为 bf16 C。
 *   - kernel 内部 AIV 全量量化一次 → workspace,然后通知 AIC 做 matmul
 * --------------------------------------------------------------------------
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <fstream>
#include <iostream>
#include <string>
#include <cstring>

#include "catlass/gemm/kernel/dual_level_quant_mx_batched_matmul_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue_dual_level_quant_mx.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy_dual_level_quant_mx.hpp"
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

static const std::string kDataRoot = "examples/63_ascend950_dual_level_quant_mx_batch_matmul/data";

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
    std::memcpy(buffer.data(), data.data(), dataSize);

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
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t batchCount = options.problemCount;
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(k);

    // ----- Type definitions -----
    using ElementInput = float16_t;
    using ElementA = float4_e2m1x2_t;
    using ElementB = float4_e2m1x2_t;
    using ElementMxScale = float8_e8m0_t;
    using ElementC = bfloat16_t;

    using LayoutTagInputA = layout::RowMajor;
    using LayoutTagInputB = layout::RowMajor;
    using LayoutTagPhysicalB = layout::RowMajor;
    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;
    using LayoutTagC = layout::RowMajor;

    // ----- Size calculations -----
    size_t lenInputA = static_cast<size_t>(m) * k * batchCount;
    size_t lenInputB = static_cast<size_t>(k) * n * batchCount;
    size_t sizeInputA = lenInputA * sizeof(ElementInput);
    size_t sizeInputB = lenInputB * sizeof(ElementInput);

    size_t lenC = static_cast<size_t>(m) * n * batchCount;
    size_t sizeC = lenC * sizeof(ElementC);

    uint32_t scaleA1K = CeilDiv<512>(k);
    uint32_t mxScaleAlignedK = RoundUp<2>(mxScaleK);
    size_t lenScaleA1 = static_cast<size_t>(m) * scaleA1K * batchCount;
    size_t lenScaleA2 = static_cast<size_t>(m) * mxScaleAlignedK * batchCount;
    size_t lenScaleB1 = static_cast<size_t>(scaleA1K) * n * batchCount;
    size_t lenScaleB2 = static_cast<size_t>(mxScaleAlignedK) * n * batchCount;
    size_t sizeScaleA1 = lenScaleA1 * sizeof(float);
    size_t sizeScaleA2 = lenScaleA2 * sizeof(ElementMxScale);
    size_t sizeScaleB1 = lenScaleB1 * sizeof(float);
    size_t sizeScaleB2 = lenScaleB2 * sizeof(ElementMxScale);

    // ----- Host data -----
    std::vector<ElementInput> hostInputA(lenInputA);
    std::vector<ElementInput> hostInputB(lenInputB);

    // ----- Device pointers -----
    uint8_t* deviceInputA{nullptr};
    uint8_t* deviceInputB{nullptr};
    uint8_t* deviceC{nullptr};
    uint8_t* deviceScaleA1{nullptr};
    uint8_t* deviceScaleA2{nullptr};
    uint8_t* deviceScaleB1{nullptr};
    uint8_t* deviceScaleB2{nullptr};
    uint8_t* deviceWorkspace{nullptr};
    size_t sizeWorkspace{0};

    const auto cleanupAndFinalize = [&]() {
        if (deviceInputA)
            ACL_CHECK(aclrtFree(deviceInputA));
        if (deviceInputB)
            ACL_CHECK(aclrtFree(deviceInputB));
        if (deviceC)
            ACL_CHECK(aclrtFree(deviceC));
        if (deviceScaleA1)
            ACL_CHECK(aclrtFree(deviceScaleA1));
        if (deviceScaleA2)
            ACL_CHECK(aclrtFree(deviceScaleA2));
        if (deviceScaleB1)
            ACL_CHECK(aclrtFree(deviceScaleB1));
        if (deviceScaleB2)
            ACL_CHECK(aclrtFree(deviceScaleB2));
        if (sizeWorkspace > 0 && deviceWorkspace)
            ACL_CHECK(aclrtFree(deviceWorkspace));
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };

    // ----- Read input data -----
    std::string inputDir = kDataRoot + "/input";
    std::string goldenDir = kDataRoot + "/golden";

    if (!ReadFile(inputDir + "/input_a.bin", hostInputA.data(), sizeInputA)) {
        cleanupAndFinalize();
        return;
    }
    if (!ReadFile(inputDir + "/input_b.bin", hostInputB.data(), sizeInputB)) {
        cleanupAndFinalize();
        return;
    }

    // ----- Allocate device memory -----
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceInputA), sizeInputA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceInputA, sizeInputA, hostInputA.data(), sizeInputA, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceInputB), sizeInputB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceInputB, sizeInputB, hostInputB.data(), sizeInputB, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceScaleA1), sizeScaleA1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceScaleA2), sizeScaleA2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceScaleB1), sizeScaleB1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceScaleB2), sizeScaleB2, ACL_MEM_MALLOC_HUGE_FIRST));

    // ----- Kernel template composition -----
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 9201
    using ArchTag = Arch::Ascend960DT;
#elif defined(CATLASS_ARCH) && CATLASS_ARCH == 9202
    using ArchTag = Arch::Ascend960PR;
#else
    using ArchTag = Arch::Ascend950;
#endif
    constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag, 16>;
    using L1TileShape = Shape<Int<256>, Int<256>, Int<512>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<256>>;

    // AIV epilogue layouts and AIC matmul TLA layouts (single batch)
    auto layoutInputA = LayoutTagInputA::MakeLayout<ElementInput>(m, k);
    auto layoutInputB = LayoutTagInputB::MakeLayout<ElementInput>(n, k);
    auto layoutOutputA = LayoutTagA::MakeLayout<ElementA>(m, k);
    auto layoutOutputB = LayoutTagPhysicalB::MakeLayout<ElementB>(n, k);
    auto layoutScaleA1 = LayoutTagA::MakeLayout<float>(m, scaleA1K);
    auto layoutScaleA2 = LayoutTagA::MakeLayout<ElementMxScale>(m, mxScaleAlignedK);
    auto layoutScaleB1 = LayoutTagPhysicalB::MakeLayout<float>(n, scaleA1K);
    auto layoutScaleB2 = LayoutTagPhysicalB::MakeLayout<ElementMxScale>(n, mxScaleAlignedK);
    auto layoutQuantA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutQuantB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    size_t sizeQuantAPerBatch = static_cast<size_t>(layoutOutputA.Capacity()) / 2;
    size_t sizeQuantBPerBatch = static_cast<size_t>(layoutOutputB.Capacity()) / 2;
    size_t sizeQuantA = sizeQuantAPerBatch * batchCount;
    size_t sizeQuantB = sizeQuantBPerBatch * batchCount;

    // ----- AIC matmul -----
    using TileCopyMmad = Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagC, void>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopyMmad>;

    // ----- AIV BlockQuant -----
    // block 不区分 A/B,只用 4 个 GemmType。
    constexpr uint32_t ubStages = 1;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAscend950DualLevelQuantMx<ubStages>;
    using QuantSubTileShape = MatrixShape<128, 512>;

    using InputType = Gemm::GemmType<ElementInput, layout::RowMajor>;
    using OutputType = Gemm::GemmType<ElementA, layout::RowMajor>;
    using Scale1Type = Gemm::GemmType<float, layout::RowMajor>;
    using Scale2Type = Gemm::GemmType<ElementMxScale, layout::RowMajor>;

    using TileCopyQuant =
        Epilogue::Tile::TileCopyDualLevelQuantMx<ArchTag, InputType, OutputType, Scale1Type, Scale2Type>;

    using BlockQuant = Epilogue::Block::BlockQuantDualLevelMx<
        EpilogueDispatchPolicy, QuantSubTileShape, InputType, OutputType, Scale1Type, Scale2Type, TileCopyQuant>;

    // ===== P1-A: launch 核数策略 (NOT min(aicCoreNum, taskNum)) =====
    //
    // fused tile-level path:
    //   uint32_t taskNum = batchCount * CeilDiv(m, L1_TILE_M) * CeilDiv(n, L1_TILE_N);
    //   uint32_t aicCoreUsed = min(aicCoreNum, taskNum);   // ✗ 会卡住 AIV 并行度
    //
    // single-kernel path: 总是按物理 AIC 数 launch。
    // 空闲 AIC (loopIdx >= coreLoops) 自然跳过 matmul loop,无害退出;
    // 关键收益是 AIV 端 QuantAllScheduler 能用上全部 aicCoreNum * 2 个 AIV subblock。
    uint32_t aicCoreUsed = aicCoreNum;

    // ----- Build kernel arguments & run -----
    // 根据 M vs N 选 BlockSwizzle 方向
    if (m > n) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using DualLevelQuantKernel =
            Gemm::Kernel::DualLevelQuantMxBatchedMatmulTla<BlockMmad, BlockQuant, BlockScheduler, ElementInput>;
        using Adapter = Gemm::Device::DeviceGemm<DualLevelQuantKernel>;

        typename DualLevelQuantKernel::Arguments arguments{
            batchCount,    options.problemShape, deviceInputA,   layoutInputA,
            deviceInputB,  layoutInputB,         deviceC,        layoutC,
            deviceScaleA1, deviceScaleA2,        layoutMxScaleA, deviceScaleB1,
            deviceScaleB2, layoutMxScaleB,       layoutOutputA,  layoutOutputB,
            layoutScaleA1, layoutScaleA2,        layoutScaleB1,  layoutScaleB2,
            layoutQuantA,  layoutQuantB};

        Adapter op;
        if (op.CanImplement(arguments) != Status::kSuccess) {
            std::cerr << "Cannot implement the arguments" << std::endl;
            cleanupAndFinalize();
            return;
        }
        sizeWorkspace = op.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        op.Initialize(arguments, deviceWorkspace);
        op(stream, aicCoreUsed);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using DualLevelQuantKernel =
            Gemm::Kernel::DualLevelQuantMxBatchedMatmulTla<BlockMmad, BlockQuant, BlockScheduler, ElementInput>;
        using Adapter = Gemm::Device::DeviceGemm<DualLevelQuantKernel>;

        typename DualLevelQuantKernel::Arguments arguments{
            batchCount,    options.problemShape, deviceInputA,   layoutInputA,
            deviceInputB,  layoutInputB,         deviceC,        layoutC,
            deviceScaleA1, deviceScaleA2,        layoutMxScaleA, deviceScaleB1,
            deviceScaleB2, layoutMxScaleB,       layoutOutputA,  layoutOutputB,
            layoutScaleA1, layoutScaleA2,        layoutScaleB1,  layoutScaleB2,
            layoutQuantA,  layoutQuantB};

        Adapter op;
        if (op.CanImplement(arguments) != Status::kSuccess) {
            std::cerr << "Cannot implement the arguments" << std::endl;
            cleanupAndFinalize();
            return;
        }
        sizeWorkspace = op.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        op.Initialize(arguments, deviceWorkspace);
        op(stream, aicCoreUsed);
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));

    // ----- D2H: retrieve all outputs -----
    std::vector<bfloat16> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostScaleA1(lenScaleA1);
    std::vector<uint8_t> hostScaleA2(sizeScaleA2);
    std::vector<float> hostScaleB1(lenScaleB1);
    std::vector<uint8_t> hostScaleB2(sizeScaleB2);
    ACL_CHECK(aclrtMemcpy(hostScaleA1.data(), sizeScaleA1, deviceScaleA1, sizeScaleA1, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostScaleA2.data(), sizeScaleA2, deviceScaleA2, sizeScaleA2, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostScaleB1.data(), sizeScaleB1, deviceScaleB1, sizeScaleB1, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostScaleB2.data(), sizeScaleB2, deviceScaleB2, sizeScaleB2, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<uint8_t> hostQuantA(sizeQuantA);
    std::vector<uint8_t> hostQuantB(sizeQuantB);
    if (deviceWorkspace != nullptr && sizeWorkspace >= sizeQuantA + sizeQuantB) {
        ACL_CHECK(aclrtMemcpy(hostQuantA.data(), sizeQuantA, deviceWorkspace, sizeQuantA, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(
            hostQuantB.data(), sizeQuantB, deviceWorkspace + sizeQuantA, sizeQuantB, ACL_MEMCPY_DEVICE_TO_HOST));
    }

    std::string resultFileName = goldenDir + "/result.bin";
    SaveResult<bfloat16>(resultFileName, hostC);
    std::cout << "Result saved to: " << resultFileName << std::endl;

    SaveResult<float>(goldenDir + "/scale_a1_result.bin", hostScaleA1);
    SaveResult<uint8_t>(goldenDir + "/scale_a2_result.bin", hostScaleA2);
    SaveResult<float>(goldenDir + "/scale_b1_result.bin", hostScaleB1);
    SaveResult<uint8_t>(goldenDir + "/scale_b2_result.bin", hostScaleB2);
    SaveResult<uint8_t>(goldenDir + "/quant_a_result.bin", hostQuantA);
    SaveResult<uint8_t>(goldenDir + "/quant_b_result.bin", hostQuantB);

    // ----- Golden comparison -----
    std::vector<float> hostGolden(lenC);
    std::string goldenFileName = goldenDir + "/expected_data.bin";
    if (ReadFile(goldenFileName, hostGolden.data(), sizeof(float) * hostGolden.size())) {
        std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
        if (errorIndices.empty()) {
            std::cout << "Compare success." << std::endl;
        } else {
            std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
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
