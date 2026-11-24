/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_ITERBATCH_BROADCAST_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_ITERBATCH_BROADCAST_MATMUL_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Gemm::Kernel {

// Template for IterBatch broadcast batched matmul:
//   C[b] = cast(op(A[ba]) * op(B[bb]) + Bias(b)),
//   ba = (batchA == 1) ? 0 : b,  bb = (batchB == 1) ? 0 : b.
// batchA / batchB must be 1 (broadcast over the batch dim) or batchCount.
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class Enable = void>
class IterBatchBroadcastMatmulTla {
    static_assert(
        DEPENDENT_FALSE<typename BlockMmad_::DispatchPolicy>,
        "IterBatchBroadcastMatmulTla is not implemented for this DispatchPolicy");
};

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class IterBatchBroadcastMatmulTla<
    BlockMmad_, BlockEpilogue_, BlockScheduler_,
    std::enable_if_t<std::is_same_v<
        typename BlockMmad_::DispatchPolicy,
        MmadIterBatchBroadcast<
            typename BlockMmad_::ArchTag, BlockMmad_::USE_HF32_MODE, BlockMmad_::L0C_STAGES, BlockMmad_::L1_STAGES,
            BlockMmad_::L0A_STAGES, BlockMmad_::L0B_STAGES>>>> {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementBias = typename BlockMmad::ElementBias;

    using BlockScheduler = BlockScheduler_;

    static constexpr bool HAS_BIAS = !std::is_void_v<ElementBias>;

    // Capacity planner for the runtime two-level batch split (B1 L1 groups / B0 L0 sub-groups).
    // The planner is the block's nested ResourcePlan: every budget it derives is configured
    // by the block's own tile / stage / layout settings.
    using ResourcePlan = typename BlockMmad::ResourcePlan;

    /// Parameters structure
    struct Params {
        // Data members
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        int64_t strideA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        int64_t strideB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        int64_t strideC;
        GM_ADDR ptrBias;
        BiasMode biasMode;
        bool broadcastA;
        bool broadcastB;
        typename ResourcePlan::Result plan;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            uint32_t batchCount_, GemmCoord const& problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, int64_t strideA_,
            GM_ADDR ptrB_, LayoutB layoutB_, int64_t strideB_, GM_ADDR ptrC_, LayoutC layoutC_, int64_t strideC_,
            GM_ADDR ptrBias_, BiasMode biasMode_, bool broadcastA_, bool broadcastB_,
            typename ResourcePlan::Result const& plan_)
            : batchCount(batchCount_),
              problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              strideA(strideA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              strideB(strideB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              strideC(strideC_),
              ptrBias(ptrBias_),
              biasMode(biasMode_),
              broadcastA(broadcastA_),
              broadcastB(broadcastB_),
              plan(plan_)
        {}
    };

    struct Arguments {
        uint32_t batchCount; // number of output batches
        uint32_t batchA;     // batch dim of A, must be 1 (broadcast) or batchCount
        uint32_t batchB;     // batch dim of B, must be 1 (broadcast) or batchCount
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrBias{nullptr};
        BiasMode biasMode{BiasMode::None};
        uint32_t iterBatchL1{0}; // 0 = auto (derive the max B1 from the L1 budget)
        uint32_t iterBatchL0{0}; // 0 = auto (derive the max B0 bounded by B1)
        uint32_t coreNum{0};     // 0 = capacity-only auto B1; > 0 additionally bounds auto B1 by
                                 // ceil(batchCount / coreNum) for grid-stride parallelism
    };

    // Build the runtime resource plan. The status / budget fields allow the host side to
    // report why a request was rejected (explicit spec that does not fit is rejected,
    // never silently clamped).
    static typename ResourcePlan::Result MakePlan(const Arguments& args)
    {
        const uint32_t m = args.problemShape.m();
        const uint32_t n = args.problemShape.n();
        const uint32_t k = args.problemShape.k();
        const bool broadcastA = (args.batchA == 1);
        const bool broadcastB = (args.batchB == 1);
        return ResourcePlan::Make(
            args.batchCount, m, n, k, broadcastA, broadcastB, args.biasMode, args.iterBatchL1, args.iterBatchL0,
            args.coreNum);
    }

    static bool CanImplement(const Arguments& args)
    {
        if (args.batchA != 1 && args.batchA != args.batchCount) {
            return false;
        }
        if (args.batchB != 1 && args.batchB != args.batchCount) {
            return false;
        }
        typename ResourcePlan::Result plan = MakePlan(args);
        return plan.valid;
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        const uint32_t m = args.problemShape.m();
        const uint32_t n = args.problemShape.n();
        const uint32_t k = args.problemShape.k();
        const bool broadcastA = (args.batchA == 1);
        const bool broadcastB = (args.batchB == 1);
        // A broadcast operand has batch stride 0 (all output batches read the same matrix).
        const int64_t strideA = broadcastA ? 0 : static_cast<int64_t>(m) * k;
        const int64_t strideB = broadcastB ? 0 : static_cast<int64_t>(k) * n;
        const int64_t strideC = static_cast<int64_t>(m) * n;

        typename ResourcePlan::Result plan = MakePlan(args);

        Params params{args.batchCount, args.problemShape, args.ptrA,  args.layoutA, strideA, args.ptrB,
                      args.layoutB,    strideB,           args.ptrC,  args.layoutC, strideC, args.ptrBias,
                      args.biasMode,   broadcastA,        broadcastB, plan};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    IterBatchBroadcastMatmulTla()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    /// Executes the IterBatch broadcast batched matmul
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler scheduler(params.batchCount, params.plan.iterBatchL1);
        const uint32_t coreLoops = scheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource, params.plan, params.broadcastA, params.broadcastB, params.biasMode);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);

        using GlobalTensorBiasType = std::conditional_t<HAS_BIAS, ElementBias, uint8_t>;
        [[maybe_unused]] AscendC::GlobalTensor<GlobalTensorBiasType> gmBias;
        if constexpr (HAS_BIAS) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias*)params.ptrBias);
        }

        bool isFirstBlock = true;
        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            const uint32_t actualL1Batch = scheduler.GetActualGroupBatch(loopIdx);
            const uint32_t batchBase = scheduler.GetBatchIdx(loopIdx);
            const int64_t batchOffsetA = static_cast<int64_t>(batchBase) * params.strideA;
            const int64_t batchOffsetB = static_cast<int64_t>(batchBase) * params.strideB;
            const int64_t batchOffsetC = static_cast<int64_t>(batchBase) * params.strideC;

            // GM tensors of the FIRST batch of this group; the block derives the remaining
            // batches and the next group's tensors via the batch strides.
            auto tensorA = tla::MakeTensor(gmA[batchOffsetA], params.layoutA, Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(gmB[batchOffsetB], params.layoutB, Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(gmC[batchOffsetC], params.layoutC, Arch::PositionGM{});

            const uint32_t nextLoopIdx = loopIdx + AscendC::GetBlockNum();
            bool hasNextBlock = false;
            uint32_t nextActualL1Batch = 0;
            if (nextLoopIdx < coreLoops) {
                hasNextBlock = true;
                nextActualL1Batch = scheduler.GetActualGroupBatch(nextLoopIdx);
            }
            // GM tensors of the FIRST batch of the NEXT group on this core. Under grid-stride
            // the next group is scheduled at (loopIdx + blockNum), whose batch base is coreNum
            // groups away from the current one. The block reads them only when hasNextBlock is
            // true, so the current group's base is used as a placeholder for the last group.
            const uint32_t nextBatchBase = hasNextBlock ? scheduler.GetBatchIdx(nextLoopIdx) : batchBase;
            const int64_t nextBatchOffsetA = static_cast<int64_t>(nextBatchBase) * params.strideA;
            const int64_t nextBatchOffsetB = static_cast<int64_t>(nextBatchBase) * params.strideB;
            auto nextTensorA = tla::MakeTensor(gmA[nextBatchOffsetA], params.layoutA, Arch::PositionGM{});
            auto nextTensorB = tla::MakeTensor(gmB[nextBatchOffsetB], params.layoutB, Arch::PositionGM{});

            if constexpr (HAS_BIAS) {
                // Bias rows of this group start at batchBase * biasStride; the stride is
                // derived from the mode (PerBatchN: n, otherwise 0 / shared single row).
                // A rank-1 [N] layout is used regardless of the mode.
                const uint32_t n = params.problemShape.n();
                const int64_t biasStride = (params.biasMode == BiasMode::PerBatchN) ? static_cast<int64_t>(n) : 0;
                const int64_t batchOffsetBias = static_cast<int64_t>(batchBase) * biasStride;
                const int64_t nextBatchOffsetBias = static_cast<int64_t>(nextBatchBase) * biasStride;
                auto layoutBias = tla::MakeLayout(n);
                auto tensorBias = tla::MakeTensor(gmBias[batchOffsetBias], layoutBias, Arch::PositionGM{});
                auto nextTensorBias = tla::MakeTensor(gmBias[nextBatchOffsetBias], layoutBias, Arch::PositionGM{});
                blockMmad(
                    tensorA, tensorB, tensorC, params.problemShape, actualL1Batch, nextActualL1Batch, isFirstBlock,
                    hasNextBlock, nextTensorA, nextTensorB, nextTensorBias, tensorBias);
            } else {
                blockMmad(
                    tensorA, tensorB, tensorC, params.problemShape, actualL1Batch, nextActualL1Batch, isFirstBlock,
                    hasNextBlock, nextTensorA, nextTensorB);
            }
            isFirstBlock = false;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {}
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_ITERBATCH_BROADCAST_MATMUL_TLA_HPP
