/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <iostream>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/batched_matmul_fixpipe_dequant_tla.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "golden.hpp"
#include "helper.hpp"
#include "options.hpp"
#include "tla/layout.hpp"

using namespace Catlass;
using namespace tla;

struct QBMMOptions : public GroupedGemmOptions {
    uint32_t biasElements{0};
    Gemm::Tile::ScaleGranularity inputQuantMode{Gemm::Tile::ScaleGranularity::NO_QUANT};
    Gemm::Tile::ScaleGranularity weightQuantMode{Gemm::Tile::ScaleGranularity::NO_QUANT};
    const std::string HELPER =
        "89_ascend950_hif8_quant_batch_matmul batch M K N bias x1quantmode x2quantmode [device_id]";

    int Parse(int argc, const char** argv)
    {
        enum class ArgsIndex
        {
            BATCH_INDEX = 1,
            M_INDEX,
            K_INDEX,
            N_INDEX,
            BIAS_INDEX,
            X1QUANTMODE_INDEX,
            X2QUANTMODE_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<int>(ArgsIndex::ARGS_MAX) || argc < static_cast<int>(ArgsIndex::DEVICE_ID_INDEX)) {
            std::cerr << HELPER << std::endl;
            return -1;
        }
        auto arg = [&](ArgsIndex i) { return argv[static_cast<int>(i)]; };
        problemCount = std::atoi(arg(ArgsIndex::BATCH_INDEX));
        problemShape.m() = std::atoi(arg(ArgsIndex::M_INDEX));
        problemShape.k() = std::atoi(arg(ArgsIndex::K_INDEX));
        problemShape.n() = std::atoi(arg(ArgsIndex::N_INDEX));
        biasElements = std::atoi(arg(ArgsIndex::BIAS_INDEX));
        int inputMode = std::atoi(arg(ArgsIndex::X1QUANTMODE_INDEX));
        int weightMode = std::atoi(arg(ArgsIndex::X2QUANTMODE_INDEX));
        if (inputMode < 0 || inputMode > 2 || weightMode < 0 || weightMode > 2) {
            std::fprintf(stderr, "quantmode must be 0 (NO_QUANT), 1 (PER_TENSOR) or 2 (PER_CHANNEL)\n");
            return -1;
        }
        inputQuantMode = static_cast<Gemm::Tile::ScaleGranularity>(inputMode);
        weightQuantMode = static_cast<Gemm::Tile::ScaleGranularity>(weightMode);
        if (argc == static_cast<int>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(arg(ArgsIndex::DEVICE_ID_INDEX));
        }
        return 0;
    }

    bool IsPerChannel() const
    {
        return weightQuantMode == Gemm::Tile::ScaleGranularity::PER_CHANNEL;
    }
};

static void Run(QBMMOptions const& options)
{
    if (options.problemCount == 0 || options.problemShape.m() == 0 || options.problemShape.n() == 0 ||
        options.problemShape.k() == 0) {
        std::cerr << "batch, M, N, K must be positive" << std::endl;
        return;
    }

    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    using ElementA = hifloat8_t;
    using ElementB = hifloat8_t;
    using ElementC = bfloat16_t;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;

    uint32_t batchCount = options.problemCount;
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    // The device copy path handles an unaligned HiF8 K tail.
    int64_t strideA = static_cast<int64_t>(m) * k;
    int64_t strideB = static_cast<int64_t>(k) * n;
    int64_t strideC = static_cast<int64_t>(m) * n;

    auto layoutA = tla::MakeLayout(
        tla::MakeShape(batchCount, m, k), tla::MakeStride(strideA, static_cast<int64_t>(k), tla::Int<1>{}));
    auto layoutB = tla::MakeLayout(
        tla::MakeShape(batchCount, k, n), tla::MakeStride(strideB, static_cast<int64_t>(n), tla::Int<1>{}));
    auto layoutC = tla::MakeLayout(
        tla::MakeShape(batchCount, m, n), tla::MakeStride(strideC, static_cast<int64_t>(n), tla::Int<1>{}));

    size_t lenA = static_cast<size_t>(layoutA(tla::MakeCoord(batchCount - 1, m - 1, k - 1))) + 1;
    size_t lenB = static_cast<size_t>(layoutB(tla::MakeCoord(batchCount - 1, k - 1, n - 1))) + 1;
    size_t lenC = static_cast<size_t>(layoutC(tla::MakeCoord(batchCount - 1, m - 1, n - 1))) + 1;
    size_t lenScale = static_cast<size_t>(n);

    size_t sizeA = lenA * sizeof(uint8_t);
    size_t sizeB = lenB * sizeof(uint8_t);
    size_t sizeC = lenC * sizeof(ElementC);
    size_t sizeScale = lenScale * sizeof(uint64_t);
    size_t sizeBias = lenScale * sizeof(float);

    std::string datapath = "./examples/89_ascend950_hif8_quant_batch_matmul/gen_data/data/";

    std::vector<uint8_t> hostA(lenA, 0);
    std::vector<uint8_t> hostB(lenB, 0);
    std::vector<float> hostScales(lenScale, 1.0F);
    std::vector<uint64_t> hostQuant(lenScale, 0ULL);
    std::vector<float> hostBias(lenScale, 0.0F);

    bool dataOk = ReadFile(datapath + "tensor_a.bin", hostA.data(), hostA.size());
    dataOk &= ReadFile(datapath + "tensor_b.bin", hostB.data(), hostB.size());
    dataOk &= ReadFile(datapath + "dequant_scale.bin", hostScales.data(), hostScales.size() * sizeof(float));
    if (options.IsPerChannel()) {
        dataOk &= ReadFile(datapath + "quant_scale.bin", hostQuant.data(), sizeScale);
    }
    dataOk &= ReadFile(datapath + "bias.bin", hostBias.data(), sizeBias);
    if (!dataOk) {
        std::cerr << "Read offline data failed. Run gen_data/gen_data.py first." << std::endl;
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return;
    }
    const float perTensorScale = hostScales[0];

    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceBias{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBias), sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceQuant{nullptr};
    if (options.IsPerChannel()) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceQuant), sizeScale, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(deviceQuant, sizeScale, hostQuant.data(), sizeScale, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    auto layoutBias = tla::MakeLayout(n);

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    constexpr bool useHF32 = false;
    constexpr bool enableL1Resident = false;
    constexpr uint32_t l0CStages = 1;
    constexpr uint32_t l1AStages = 2;
    constexpr uint32_t l1BStages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    using DispatchPolicy = Gemm::MmadDequant<
        ArchTag, enableUnitFlag, useHF32, l0CStages, enableL1Resident, l1AStages, l1BStages, l0AStages, l0BStages>;
    using L1TileShape = Shape<Int<256>, Int<256>, Int<128>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<32>>;
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

    auto aiCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint8_t* deviceWorkspace{nullptr};
    if (options.IsPerChannel()) {
        using TileCopy = Gemm::Tile::PackedTileCopyTla<
            ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, float, false,
            Gemm::Tile::ScaleGranularity::PER_CHANNEL>;
        using BlockMmad = Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, float, TileCopy>;
        using MatmulKernel = Gemm::Kernel::BatchedMatmulFixpipeDequantTla<BlockMmad, BlockScheduler>;
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        typename MatmulKernel::Arguments arguments{
            batchCount, GemmCoord{m, n, k}, deviceA,     layoutA,    deviceB,   layoutB, deviceC,
            layoutC,    perTensorScale,     deviceQuant, deviceBias, layoutBias};
        MatmulAdapter matmul_op;
        size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        matmul_op.Initialize(arguments, deviceWorkspace);
        matmul_op(stream, aiCoreNum);
    } else {
        using TileCopy = Gemm::Tile::PackedTileCopyTla<
            ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, float, false,
            Gemm::Tile::ScaleGranularity::PER_TENSOR>;
        using BlockMmad = Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, float, TileCopy>;
        using MatmulKernel = Gemm::Kernel::BatchedMatmulFixpipeDequantTla<BlockMmad, BlockScheduler>;
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        typename MatmulKernel::Arguments arguments{
            batchCount, GemmCoord{m, n, k}, deviceA,     layoutA,    deviceB,   layoutB, deviceC,
            layoutC,    perTensorScale,     deviceQuant, deviceBias, layoutBias};
        MatmulAdapter matmul_op;
        size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        matmul_op.Initialize(arguments, deviceWorkspace);
        matmul_op(stream, aiCoreNum);
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<float> hostGolden(lenC);
    if (!ReadFile(datapath + "golden_out.bin", hostGolden.data(), lenC * sizeof(float))) {
        std::cerr << "Read offline data failed. Run gen_data/gen_data.py first." << std::endl;
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return;
    }
    std::vector<bfloat16> hostOut(lenC);
    ACL_CHECK(aclrtMemcpy(hostOut.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    if (const char* dumpPath = std::getenv("QBMM_DUMP")) {
        if (FILE* fp = std::fopen(dumpPath, "wb")) {
            std::fwrite(hostOut.data(), sizeof(bfloat16), hostOut.size(), fp);
            std::fclose(fp);
        }
    }
    std::vector<uint64_t> errorIndices = golden::CompareDataBfloat16(hostOut, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
        std::cerr << "Compare failed. errorIndices[0]: " << errorIndices[0] << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceBias));
    if (deviceQuant != nullptr) {
        ACL_CHECK(aclrtFree(deviceQuant));
    }
    if (deviceWorkspace != nullptr) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv)
{
    QBMMOptions options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
