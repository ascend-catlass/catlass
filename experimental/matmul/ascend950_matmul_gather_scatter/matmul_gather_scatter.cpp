/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include <cstdlib>
#include <iostream>
#include <vector>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue_matmul_gather_scatter.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_gather_scatter_full_load.hpp"
#include "catlass/gemm/kernel/matmul_gather_scatter_simt.hpp"
#include "catlass/gemm/tile/matmul_gather_scatter_tile_copy.h"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tuple.hpp"

#include "golden.hpp"
#include "helper.hpp"
#include "options.hpp"

using namespace Catlass;
using namespace tla;

namespace {

constexpr uint32_t FULL_LOAD_A_MAX_K = 2048;
constexpr int32_t OUTPUT_SENTINEL_BYTE = 0x3c;
using ArchTag = Arch::Ascend950;

constexpr bool ENABLE_UNIT_FLAG = true;
constexpr bool USE_HF32_MODE = false;
constexpr uint32_t L0C_STAGES = 1;
constexpr bool ENABLE_L1_RESIDENT = false;
constexpr uint32_t L1A_STAGES = 1;
constexpr uint32_t L0A_STAGES = 2;
constexpr uint32_t L0B_STAGES = 2;
constexpr uint32_t OUTPUT_UB_STAGES = 1;
constexpr uint32_t L1B_STAGES = 2;

// Keep one FullLoad profile per Gather mode. The runtime branch selects only
// AIV Gather versus AIC Gather; arbitrary shapes use the SIMT kernel.
using L1TileM128N256 = tla::tuple<tla::Int<128>, tla::Int<256>, tla::Int<128>>;
using L0TileM128N256 = tla::tuple<tla::Int<128>, tla::Int<256>, tla::Int<64>>;
using L1TileM96N256 = tla::tuple<tla::Int<96>, tla::Int<256>, tla::Int<128>>;
using L0TileM96N256 = tla::tuple<tla::Int<96>, tla::Int<256>, tla::Int<64>>;

struct AivGatherKernelMode {
    static constexpr bool AIV_GATHER = true;

    template <class BlockMmad, class BlockEpilogue, class BlockScheduler>
    using Kernel = Gemm::Kernel::MatmulGatherScatterFullLoadAivGather<BlockMmad, BlockEpilogue, BlockScheduler>;
};

struct AicGatherKernelMode {
    static constexpr bool AIV_GATHER = false;

    template <class BlockMmad, class BlockEpilogue, class BlockScheduler>
    using Kernel = Gemm::Kernel::MatmulGatherScatterFullLoadAicGather<BlockMmad, BlockEpilogue, BlockScheduler>;
};

struct Options : public GemmOptions {
    uint32_t j{};

    int Parse(int argc, const char** argv)
    {
        enum class ArgsIndex
        {
            M_INDEX = 1,
            J_INDEX,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc < static_cast<int>(ArgsIndex::DEVICE_ID_INDEX) || argc > static_cast<int>(ArgsIndex::ARGS_MAX)) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " m j n k [device_id]" << std::endl;
            return -1;
        }
        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        j = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::J_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        problemShape.k() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::K_INDEX)]);
        if (argc == static_cast<int>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }
        if (problemShape.m() == 0 || j == 0 || j > problemShape.m() || problemShape.n() == 0 || problemShape.k() == 0) {
            return -1;
        }
        return 0;
    }
};

template <class Kernel>
void LaunchKernel(
    aclrtStream stream, uint32_t m, uint32_t j, uint32_t n, uint32_t k, uint8_t* deviceA, uint8_t* deviceB,
    uint8_t* deviceIndices, uint8_t* deviceD, uint32_t aicCoreNum)
{
    typename Kernel::Arguments arguments{GemmCoord{j, n, k}, m, deviceA, deviceB, deviceIndices, deviceD, aicCoreNum};
    using DeviceAdapter = Gemm::Device::DeviceGemm<Kernel>;
    if (DeviceAdapter::CanImplement(arguments) != Status::kSuccess) {
        throw std::runtime_error("MatmulGatherScatter cannot implement this shape");
    }
    DeviceAdapter deviceOp;
    deviceOp.Initialize(arguments);
    deviceOp.SetSimtDynamicMemSize(Kernel::UB_SIZE);
    deviceOp.RunSimt(stream, aicCoreNum);
}

template <class L1TileShape, class L0TileShape, class GatherKernelMode>
void LaunchMatmulGatherScatter(
    aclrtStream stream, uint32_t m, uint32_t j, uint32_t n, uint32_t k, uint8_t* deviceA, uint8_t* deviceB,
    uint8_t* deviceIndices, uint8_t* deviceD, uint32_t aicCoreNum)
{
    using MmadDispatchPolicy = Gemm::MmadAscend950FullLoadA<
        ArchTag, ENABLE_UNIT_FLAG, USE_HF32_MODE, L0C_STAGES, ENABLE_L1_RESIDENT, L1A_STAGES, L1B_STAGES, L0A_STAGES,
        L0B_STAGES>;
    using GatherScatterDispatchPolicy = Gemm::MatmulGatherScatterDispatchPolicy<
        ArchTag, OUTPUT_UB_STAGES, GatherKernelMode::AIV_GATHER, Gemm::Kernel::detail::MGS_AIV_GATHER_MAX_K>;
    using TileCopy = Gemm::Tile::MatmulGatherScatterTileCopy<
        ArchTag, half, layout::RowMajor, half, layout::RowMajor, float, layout::RowMajor>;
    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, half, typename TileCopy::LayoutTagL1A>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, half, half, float, void, TileCopy, TileMmad>;
    using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzleL1FullLoad<>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogueMatmulGatherScatter<BlockMmad, GatherScatterDispatchPolicy>;
    using Kernel = typename GatherKernelMode::template Kernel<BlockMmad, BlockEpilogue, BlockScheduler>;
    LaunchKernel<Kernel>(stream, m, j, n, k, deviceA, deviceB, deviceIndices, deviceD, aicCoreNum);
}

void Run(Options const& options)
{
    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t j = options.j;
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenD = static_cast<size_t>(m) * n;

    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    std::vector<int32_t> hostIndices(j);
    golden::FillRandomData<fp16_t>(hostA, -1.0F, 1.0F);
    golden::FillRandomData<fp16_t>(hostB, -1.0F, 1.0F);
    for (uint32_t i = 0; i < j; ++i) {
        hostIndices[i] = static_cast<int32_t>((static_cast<uint64_t>(i) * (m - 1U)) % m);
    }

    uint8_t* deviceA = nullptr;
    uint8_t* deviceB = nullptr;
    uint8_t* deviceIndices = nullptr;
    uint8_t* deviceD = nullptr;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), lenA * sizeof(fp16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), lenB * sizeof(fp16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceIndices), j * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceD), lenD * sizeof(fp16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(
        aclrtMemcpy(deviceA, lenA * sizeof(fp16_t), hostA.data(), lenA * sizeof(fp16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(
        aclrtMemcpy(deviceB, lenB * sizeof(fp16_t), hostB.data(), lenB * sizeof(fp16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(
        deviceIndices, j * sizeof(int32_t), hostIndices.data(), j * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));
    // Prefill D with a nonzero byte pattern so the validation also detects rows that the kernel failed to clear.
    ACL_CHECK(aclrtMemset(deviceD, lenD * sizeof(fp16_t), OUTPUT_SENTINEL_BYTE, lenD * sizeof(fp16_t)));

    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    bool useSimt = (k % Gemm::Kernel::detail::MGS_FP16_ELEMENTS_PER_BLOCK) != 0U || k > FULL_LOAD_A_MAX_K;
    if (useSimt) {
        LaunchKernel<Gemm::Kernel::MatmulGatherScatterSimt>(
            stream, m, j, n, k, deviceA, deviceB, deviceIndices, deviceD, aicCoreNum);
    } else if (k <= Gemm::Kernel::detail::MGS_AIV_GATHER_MAX_K) {
        LaunchMatmulGatherScatter<L1TileM128N256, L0TileM128N256, AivGatherKernelMode>(
            stream, m, j, n, k, deviceA, deviceB, deviceIndices, deviceD, aicCoreNum);
    } else {
        LaunchMatmulGatherScatter<L1TileM96N256, L0TileM96N256, AicGatherKernelMode>(
            stream, m, j, n, k, deviceA, deviceB, deviceIndices, deviceD, aicCoreNum);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostD(lenD);
    ACL_CHECK(
        aclrtMemcpy(hostD.data(), lenD * sizeof(fp16_t), deviceD, lenD * sizeof(fp16_t), ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<fp16_t> hostGather(static_cast<size_t>(j) * k);
    for (uint32_t row = 0; row < j; ++row) {
        std::copy_n(hostA.data() + static_cast<size_t>(hostIndices[row]) * k, k, hostGather.data() + row * k);
    }
    auto layoutGather = layout::RowMajor::MakeLayout<half>(j, k);
    auto layoutB = layout::RowMajor::MakeLayout<half>(k, n);
    auto layoutProduct = layout::RowMajor::MakeLayout<half>(j, n);
    std::vector<float> hostProduct(static_cast<size_t>(j) * n);
    golden::ComputeMatmul(GemmCoord{j, n, k}, hostGather, layoutGather, hostB, layoutB, hostProduct, layoutProduct);

    std::vector<float> hostGolden(lenD, 0.0F);
    for (uint32_t row = 0; row < j; ++row) {
        std::copy_n(
            hostProduct.data() + static_cast<size_t>(row) * n, n,
            hostGolden.data() + static_cast<size_t>(hostIndices[row]) * n);
    }

    auto errors = golden::CompareData(hostD, hostGolden, k);
    if (errors.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cout << "Compare failed. Error count: " << errors.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceIndices));
    ACL_CHECK(aclrtFree(deviceD));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

} // namespace

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
