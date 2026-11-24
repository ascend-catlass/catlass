/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// IterBatch broadcast batched matmul on Ascend950:
//   C[b] = cast(op(A[ba]) * op(B[bb]) + Bias(b)),
//   ba = (batchA == 1) ? 0 : b,  bb = (batchB == 1) ? 0 : b.
// batchA / batchB must be 1 (broadcast over the batch dim) or batchCount.
//
// The batch dim is split into consecutive L1 groups of B1 batches (broadcast operands
// are physically loaded once per group) and L0 sub-groups of B0 batches (each batch has
// an independent L0C accumulator). A request that does not fit is rejected with the
// resource budget instead of being silently clamped.

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <sstream>
#include <stdexcept>
#include <string>

#include "catlass/gemm/kernel/iterbatch_broadcast_matmul_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_scheduler_iterBatch.hpp"
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

/**
 * @struct IterBatchBroadcastOptions
 * @brief Options structure for the IterBatch broadcast matmul example.
 * @brief Arguments: `example_name batch_count m n k [broadcast_a broadcast_b bias_mode [device_id]]`
 */
struct IterBatchBroadcastOptions {
    const std::string HELPER =
        "batch_count m n k [broadcast_a broadcast_b bias_mode [device_id]]\n"
        "  broadcast_a/broadcast_b: 0 = per-batch operand, 1 = broadcast over batch dim\n"
        "  bias_mode: 0 = none, 1 = shared [N], 2 = per-batch [batch, N]";

    Catlass::GemmCoord problemShape{128, 128, 128};
    uint32_t batchCount{8};
    uint32_t broadcastA{0};
    uint32_t broadcastB{1};
    uint32_t biasMode{0};
    int32_t deviceId{0};

    IterBatchBroadcastOptions() = default;

    int Parse(int argc, const char** argv)
    {
        enum class ArgsIndex
        {
            BATCH_COUNT_INDEX = 1,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            BROADCAST_A_INDEX,
            BROADCAST_B_INDEX,
            BIAS_MODE_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        // Valid forms: `batch m n k` / `batch m n k ba bb bias` / `batch m n k ba bb bias device`.
        const uint32_t argcU = static_cast<uint32_t>(argc);
        const bool validArgc = (argcU == static_cast<uint32_t>(ArgsIndex::K_INDEX) + 1) ||
                               (argcU == static_cast<uint32_t>(ArgsIndex::BIAS_MODE_INDEX) + 1) ||
                               (argcU == static_cast<uint32_t>(ArgsIndex::ARGS_MAX));
        if (!validArgc) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }

        batchCount = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::BATCH_COUNT_INDEX)]);
        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        problemShape.k() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::K_INDEX)]);
        if (argcU >= static_cast<uint32_t>(ArgsIndex::BIAS_MODE_INDEX) + 1) {
            broadcastA = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::BROADCAST_A_INDEX)]);
            broadcastB = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::BROADCAST_B_INDEX)]);
            biasMode = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::BIAS_MODE_INDEX)]);
        }
        if (argcU == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }

        if (batchCount == 0 || problemShape.m() == 0 || problemShape.n() == 0 || problemShape.k() == 0) {
            std::cerr << "batch_count/m/n/k must be positive." << std::endl;
            return -1;
        }
        if (broadcastA > 1 || broadcastB > 1) {
            std::cerr << "broadcast_a/broadcast_b must be 0 or 1." << std::endl;
            return -1;
        }
        if (biasMode > 2) {
            std::cerr << "bias_mode must be 0, 1 or 2." << std::endl;
            return -1;
        }
        return 0;
    }
};

using Options = IterBatchBroadcastOptions;

// Status names of BlockMmad::ResourcePlan::Status. The status enum is nested in the block's
// ResourcePlan, so the name function is generic over the enum type.
template <class Status>
static const char* PlanStatusName(Status status)
{
    switch (status) {
        case Status::OK:
            return "OK";
        case Status::SHAPE_EXCEEDS_L1_TILE:
            return "SHAPE_EXCEEDS_L1_TILE";
        case Status::INVALID_BIAS_MODE:
            return "INVALID_BIAS_MODE";
        case Status::INVALID_BATCH_SPEC:
            return "INVALID_BATCH_SPEC";
        case Status::L1_OVERFLOW:
            return "L1_OVERFLOW";
        case Status::L0A_OVERFLOW:
            return "L0A_OVERFLOW";
        case Status::L0B_OVERFLOW:
            return "L0B_OVERFLOW";
        case Status::L0C_OVERFLOW:
            return "L0C_OVERFLOW";
        case Status::SINGLE_MATRIX_L1_TOO_LARGE:
            return "SINGLE_MATRIX_L1_TOO_LARGE";
        case Status::L0_TILE_TOO_LARGE:
            return "L0_TILE_TOO_LARGE";
        default:
            return "UNKNOWN";
    }
}

// Compose the plan rejection message (status name + budget details) shared by the probe
// and the execution plan, so the caller sees why the request was rejected.
template <class PlanResult>
static std::string PlanRejectMessage(const char* stage, PlanResult const& plan)
{
    std::ostringstream oss;
    oss << stage << " rejected: " << PlanStatusName(plan.status)
        << " (iterBatchL1 = " << plan.iterBatchL1 << ", iterBatchL0 = " << plan.iterBatchL0
        << ", slotA = " << plan.slotA << ", slotB = " << plan.slotB << ", l1Used = " << plan.l1Used << " / "
        << Arch::Ascend950::L1_SIZE << ", l0aUsed = " << plan.l0aUsed << " / " << Arch::Ascend950::L0A_SIZE
        << ", l0bUsed = " << plan.l0bUsed << " / " << Arch::Ascend950::L0B_SIZE << ", l0cUsed = " << plan.l0cUsed
        << " / " << Arch::Ascend950::L0C_SIZE << ")";
    return oss.str();
}

static bool Run(const Options& options, aclrtStream stream)
{
    const uint32_t batchCount = options.batchCount;
    const uint32_t m = options.problemShape.m();
    const uint32_t n = options.problemShape.n();
    const uint32_t k = options.problemShape.k();
    const uint32_t batchA = options.broadcastA ? 1 : batchCount;
    const uint32_t batchB = options.broadcastB ? 1 : batchCount;
    const auto biasMode = static_cast<Gemm::BiasMode>(options.biasMode);

    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementBias = float;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;

    using ArchTag = Arch::Ascend950;
    constexpr bool useHF32 = false;
    constexpr uint32_t l0CStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    using DispatchPolicy = Gemm::MmadIterBatchBroadcast<ArchTag, useHF32, l0CStages, l1Stages, l0AStages, l0BStages>;

    // Phase-1 small-matrix path: the full A/B matrices of one group reside in L1 (M/N are
    // not tiled between L1 and L0, only K is bricked on L0). 128x128 fp32 tiles leave
    // room for a multi-batch L0 sub-group.
    using L1TileShape = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShape = Shape<Int<128>, Int<128>, Int<32>>;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopy = Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
    using BlockEpilogue = void;
    using BlockScheduler = Gemm::Block::IterBatchScheduler;

    // kernel level
    using MatmulKernel = Gemm::Kernel::IterBatchBroadcastMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    // Explicit-reject discipline: validate the capacity plan BEFORE allocating device
    // memory. iterBatchL1 / iterBatchL0 are left at 0 = auto here. coreNum is 0
    // (capacity-only): the parallel bound only shrinks B1 (never invalidates a plan).
    MatmulKernel::Arguments probeArguments{batchCount, batchA,  batchB,  options.problemShape,
                                           nullptr,    layoutA, nullptr, layoutB,
                                           nullptr,    layoutC, nullptr, biasMode,
                                           0,          0,       0};
    MatmulKernel::ResourcePlan::Result plan = MatmulKernel::MakePlan(probeArguments);
    if (!plan.valid) {
        throw std::invalid_argument(PlanRejectMessage("IterBatch plan", plan));
    }
    std::cout << "IterBatch plan: iterBatchL1 (L1 group) = " << plan.iterBatchL1
              << ", iterBatchL0 (L0 sub-group) = " << plan.iterBatchL0 << ", L1 used = " << plan.l1Used << " / "
              << ArchTag::L1_SIZE << ", L0A used = " << plan.l0aUsed << " / " << ArchTag::L0A_SIZE
              << ", L0B used = " << plan.l0bUsed << " / " << ArchTag::L0B_SIZE << ", L0C used = " << plan.l0cUsed
              << " / " << ArchTag::L0C_SIZE << std::endl;

    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

    const size_t lenA = static_cast<size_t>(tagA.Capacity()) * batchA;
    const size_t lenB = static_cast<size_t>(tagB.Capacity()) * batchB;
    const size_t lenC = static_cast<size_t>(tagC.Capacity()) * batchCount;
    const size_t lenBias = (biasMode == Gemm::BiasMode::PerBatchN) ? static_cast<size_t>(n) * batchCount :
                                                                      static_cast<size_t>(n);

    const size_t sizeA = lenA * sizeof(ElementA);
    const size_t sizeB = lenB * sizeof(ElementB);
    const size_t sizeC = lenC * sizeof(ElementC);
    const size_t sizeBias = lenBias * sizeof(ElementBias);
    size_t sizeWorkspace;

    std::vector<ElementA> hostA(lenA);
    std::vector<ElementB> hostB(lenB);
    std::vector<ElementBias> hostBias(lenBias);
    golden::FillRandomData<ElementA>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<ElementB>(hostB, -5.0f, 5.0f);
    golden::FillRandomData<ElementBias>(hostBias, -2.0f, 2.0f);

    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceBias{nullptr};
    if (biasMode != Gemm::BiasMode::None) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBias), sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    uint8_t* deviceWorkspace{nullptr};

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    MatmulKernel::Arguments arguments{batchCount, batchA,  batchB,  options.problemShape, deviceA,  layoutA, deviceB,
                                      layoutB,    deviceC, layoutC, deviceBias,           biasMode, 0,       0,
                                      aicCoreNum};

    // The execution plan can differ from the probe above: auto B1 additionally honors the
    // core count (ceil(batch/coreNum) bound) so a grid-stride launch keeps all cores busy.
    MatmulKernel::ResourcePlan::Result execPlan = MatmulKernel::MakePlan(arguments);
    if (!execPlan.valid) {
        // Reject instead of launching with iterBatchL1 = 0 (divide-by-zero on device).
        throw std::runtime_error(PlanRejectMessage("IterBatch execution plan", execPlan));
    }
    std::cout << "IterBatch execution plan: iterBatchL1 (L1 group) = " << execPlan.iterBatchL1
              << ", iterBatchL0 (L0 sub-group) = " << execPlan.iterBatchL0 << std::endl;

    MatmulAdapter matmul_op;
    if (matmul_op.CanImplement(arguments) != Status::kSuccess) {
        throw std::runtime_error("Unsupported IterBatch broadcast matmul arguments");
    }
    sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmul_op.Initialize(arguments, deviceWorkspace);
    if (matmul_op(stream, aicCoreNum) != Status::kSuccess) {
        throw std::runtime_error("IterBatch broadcast matmul launch failed");
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<ElementC> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::ComputeIterBatchBroadcastMatmul(
        batchCount, batchA, batchB, options.problemShape, hostA, tagA, hostB, tagB, hostGolden, tagC, hostBias,
        options.biasMode);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cout << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    if (deviceBias != nullptr) {
        ACL_CHECK(aclrtFree(deviceBias));
    }
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    return errorIndices.empty();
}

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    bool success = false;
    try {
        success = Run(options, stream);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());

    if (!success) {
        return -1;
    }
    return 0;
}
