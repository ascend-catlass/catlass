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

#ifndef CATLASS_GEMM_KERNEL_MX_MATMUL_PERTOKEN_PERCHANNEL_TLA_HPP
#define CATLASS_GEMM_KERNEL_MX_MATMUL_PERTOKEN_PERCHANNEL_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

#if (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202))

/// MxMatmul kernel with per-token and per-channel epilogue quantization.
///
/// Compute:  D = perTokenScale * (MxScaleA * A @ MxScaleB * B) * perChannelScale
///
/// Execution flow (AIC/AIV pipeline):
///   AIC: performs MXFP4 matrix multiplication, writes float accumulator tile to workspace
///   AIV: reads workspace tile, applies per-token and per-channel scales, writes quantized D
///   Synchronization: CrossCoreFlag handshake between AIC and AIV per tile
///
/// @tparam BlockMmad_         Block-level MMAD (MXFP4)
/// @tparam BlockEpilogue_     Block-level epilogue (per-token/per-channel dequant)
/// @tparam BlockScheduler_    Block scheduling strategy
/// @tparam WORKSPACE_STAGES_  Number of pipeline stages for workspace double-buffering
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, uint32_t WORKSPACE_STAGES_>
class MxMatmulPerTokenPerChannelTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementMxScaleA = typename BlockMmad::TileCopy::ElementMxScaleA;
    using LayoutMxScaleA = typename BlockMmad::TileCopy::LayoutMxScaleA;
    using ElementMxScaleB = typename BlockMmad::TileCopy::ElementMxScaleB;
    using LayoutMxScaleB = typename BlockMmad::TileCopy::LayoutMxScaleB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using ElementScale = typename BlockEpilogue::ElementScale;
    using LayoutScale = typename BlockEpilogue::LayoutScale;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        GemmCoord problemShape;
        __gm__ ElementA* ptrA;
        LayoutA layoutA;
        __gm__ ElementB* ptrB;
        LayoutB layoutB;
        __gm__ ElementMxScaleA* ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        __gm__ ElementMxScaleB* ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        __gm__ ElementScale* ptrScale;
        LayoutScale layoutScale;
        __gm__ ElementPerTokenScale* ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        __gm__ ElementD* ptrD;
        LayoutD layoutD;
        GM_ADDR ptrWorkspace;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, GM_ADDR ptrA_, LayoutA const& layoutA_, GM_ADDR ptrB_,
            LayoutB const& layoutB_, GM_ADDR ptrMxScaleA_, LayoutMxScaleA const& layoutMxScaleA_, GM_ADDR ptrMxScaleB_,
            LayoutMxScaleB const& layoutMxScaleB_, GM_ADDR ptrScale_, LayoutScale const& layoutScale_,
            GM_ADDR ptrPerTokenScale_, LayoutPerTokenScale const& layoutPerTokenScale_, GM_ADDR ptrD_,
            LayoutD const& layoutD_, GM_ADDR ptrWorkspace_)
            : problemShape(problemShape_),
              ptrA(reinterpret_cast<__gm__ ElementA*>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB*>(ptrB_)),
              layoutB(layoutB_),
              ptrMxScaleA(reinterpret_cast<__gm__ ElementMxScaleA*>(ptrMxScaleA_)),
              layoutMxScaleA(layoutMxScaleA_),
              ptrMxScaleB(reinterpret_cast<__gm__ ElementMxScaleB*>(ptrMxScaleB_)),
              layoutMxScaleB(layoutMxScaleB_),
              ptrScale(reinterpret_cast<__gm__ ElementScale*>(ptrScale_)),
              layoutScale(layoutScale_),
              ptrPerTokenScale(reinterpret_cast<__gm__ ElementPerTokenScale*>(ptrPerTokenScale_)),
              layoutPerTokenScale(layoutPerTokenScale_),
              ptrD(reinterpret_cast<__gm__ ElementD*>(ptrD_)),
              layoutD(layoutD_),
              ptrWorkspace(ptrWorkspace_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t aicCoreNum;
        uint8_t* ptrA;
        LayoutA layoutA;
        uint8_t* ptrB;
        LayoutB layoutB;
        uint8_t* ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        uint8_t* ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        uint8_t* ptrScale;
        LayoutScale layoutScale;
        uint8_t* ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        uint8_t* ptrD;
        LayoutD layoutD;
    };

    static bool CanImplement(const Arguments& args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        size_t lenWorkspace = static_cast<size_t>(L1_TILE_M) * L1_TILE_N * args.aicCoreNum * WORKSPACE_STAGES;
        return lenWorkspace * sizeof(uint32_t);
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        return Params{args.problemShape,
                      args.ptrA,
                      args.layoutA,
                      args.ptrB,
                      args.layoutB,
                      args.ptrMxScaleA,
                      args.layoutMxScaleA,
                      args.ptrMxScaleB,
                      args.layoutMxScaleB,
                      args.ptrScale,
                      args.layoutScale,
                      args.ptrPerTokenScale,
                      args.layoutPerTokenScale,
                      args.ptrD,
                      args.layoutD,
                      workspace};
    }

    CATLASS_DEVICE
    MxMatmulPerTokenPerChannelTla()
    {
        Arch::FlagID flagId = 0;
        for (uint32_t stageId = 0; stageId < WORKSPACE_STAGES; ++stageId) {
            flagAicFinishStoreList[stageId] = Arch::CrossCoreFlag(flagId++);
            flagAivFinishComputeList[stageId] = Arch::CrossCoreFlag(flagId++);
            aicWaitFuncList[stageId] = {this, stageId};
            aicSetFuncList[stageId] = {this, stageId};
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    /// AIC core: performs MXFP4 matmul, writes accumulator to workspace
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        AscendC::ICachePreLoad(1);
        BlockMmad blockMmad(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
        gmMxScaleA.SetGlobalBuffer(params.ptrMxScaleA);
        AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
        gmMxScaleB.SetGlobalBuffer(params.ptrMxScaleB);

        // Workspace layout: [WORKSPACE_STAGES * coreNum, L1_TILE_M, L1_TILE_N] flattened as RowMajor
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(params.ptrWorkspace));
        auto layoutTagC = layout::RowMajor{L1_TILE_M * coreNum * WORKSPACE_STAGES, L1_TILE_N};
        auto layoutC = tla::MakeLayoutFromTag(layoutTagC);

        if (CeilDiv(params.problemShape.m(), L1_TILE_M) == 1) {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (CeilDiv(params.problemShape.n(), L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, params.layoutMxScaleA, Arch::PositionGM{});
        auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, layoutC, Arch::PositionGM{});

        uint32_t stageId = 0;
        uint32_t stageUsed = 0;

        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            Callback callbackBeforeFixpipe{};
            if (stageUsed == WORKSPACE_STAGES) {
                callbackBeforeFixpipe = MakeCallback(&aicWaitFuncList[stageId]);
            } else {
                ++stageUsed;
            }
            Callback callbackAfterFixpipe = MakeCallback(&aicSetFuncList[stageId]);

            auto tensorBlockA = GetTile(
                tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

            auto tensorBlockB = GetTile(
                tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

            auto tensorBlockMxScaleA = GetTile(
                tensorMxScaleA,
                tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k())));

            auto tensorBlockMxScaleB = GetTile(
                tensorMxScaleB,
                tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));

            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord((stageId * coreNum + coreIdx) * L1_TILE_M, 0),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad(
                    tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockMxScaleA,
                    tensorBlockMxScaleB, callbackBeforeFixpipe, callbackAfterFixpipe);
            } else {
                callbackBeforeFixpipe();
                blockMmad(
                    tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockMxScaleA,
                    tensorBlockMxScaleB);
                callbackAfterFixpipe();
            }

            stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.template SynchronizeBlock<decltype(tensorC)>();
        }
        while (stageUsed > 0) {
            uint32_t aivComputeStageId =
                (stageId >= stageUsed) ? (stageId - stageUsed) : (stageId + WORKSPACE_STAGES - stageUsed);
            Arch::CrossCoreWaitFlag(flagAivFinishComputeList[aivComputeStageId]);
            --stageUsed;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// AIV core: reads workspace tile, applies per-token and per-channel quantization, writes D
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        AscendC::ICachePreLoad(1);
        BlockScheduler blockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        BlockEpilogue blockEpilogue(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{L1_TILE_M * coreNum * WORKSPACE_STAGES, L1_TILE_N};

        EpilogueParams epilogueParams{
            params.ptrScale, params.layoutScale, params.ptrPerTokenScale, params.layoutPerTokenScale,
            params.ptrD,     params.layoutD};
        blockEpilogue.UpdateParams(epilogueParams);

        uint32_t coreLoops = blockScheduler.GetCoreLoops();
        GemmCoord blockShapeMNK = MakeCoord(L1_TILE_M, L1_TILE_N, L1_TILE_K);

        uint32_t stageId = 0;
        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoordMNK);

            MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1_TILE_M, 0};
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);
            auto gmBlockC = gmC[gmOffsetC];
            auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());

            Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
            blockEpilogue(blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);

            stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    friend struct AicWaitFunc;
    friend struct AicSetFunc;

    struct AicWaitFunc {
        using MatmulKernel = MxMatmulPerTokenPerChannelTla<BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES>;

        CATLASS_DEVICE AicWaitFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreWaitFlag(ptr->flagAivFinishComputeList[stageId]);
        }

        MatmulKernel* ptr{nullptr};
        uint32_t stageId;
    };

    struct AicSetFunc {
        using MatmulKernel = MxMatmulPerTokenPerChannelTla<BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES>;

        CATLASS_DEVICE AicSetFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(ptr->flagAicFinishStoreList[stageId]);
        }

        MatmulKernel* ptr{nullptr};
        uint32_t stageId;
    };

    Arch::CrossCoreFlag flagAicFinishStoreList[WORKSPACE_STAGES];
    Arch::CrossCoreFlag flagAivFinishComputeList[WORKSPACE_STAGES];

    AicWaitFunc aicWaitFuncList[WORKSPACE_STAGES];
    AicSetFunc aicSetFuncList[WORKSPACE_STAGES];
    Arch::Resource<ArchTag> resource;
};

#endif // (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202))

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MX_MATMUL_PERTOKEN_PERCHANNEL_TLA_HPP
