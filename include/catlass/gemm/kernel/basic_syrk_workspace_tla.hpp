/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_BASIC_SYRK_WORKSPACE_TLA_HPP
#define CATLASS_GEMM_KERNEL_BASIC_SYRK_WORKSPACE_TLA_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/matrix.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include <cstdint>

namespace Catlass::Gemm::Kernel {

/**
 * @brief MIX SYRK with GM workspace: D = alpha * X @ X^T + beta * Y (Ascend950).
 *
 * AIC computes lower-triangle P into L0C, dual-writes float P / P^T into a
 * per-core ping-pong workspace, then AIV applies AXPBY and stores D.
 *
 * Workspace layout (bytes), per AIC core:
 *   [stage0.P | stage0.PT | stage1.P | stage1.PT]
 * where each P/PT slot is L1_TILE_M * L1_TILE_N * sizeof(float).
 */
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, uint32_t WS_STAGES_ = 2>
class BasicSyrkWorkspaceTla {
public:
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;

    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementX = typename BlockMmad::ElementX;
    using LayoutTagX = typename BlockMmad::LayoutTagX;
    using LayoutX = typename BlockMmad::LayoutX;
    using ElementXt = typename BlockMmad::ElementXt;
    using LayoutTagXt = typename BlockMmad::LayoutTagXt;
    using LayoutXt = typename BlockMmad::LayoutXt;
    // BlockMmad stores float workspace tiles (ElementY alias of BlockMmad == float).
    using ElementP = typename BlockMmad::ElementY;
    using LayoutTagP = typename BlockMmad::LayoutTagY;
    using LayoutP = typename BlockMmad::LayoutY;
    using LayoutTagPT = layout::ColumnMajor;
    using LayoutPT = detail::TagToLayout_t<ElementP, LayoutTagPT>;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using ElementY = typename BlockEpilogue::ElementY;
    using LayoutY = typename BlockEpilogue::LayoutY;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t WS_STAGES = WS_STAGES_;
    static constexpr uint32_t SLOT_ELEMS = L1_TILE_M * L1_TILE_N;
    static constexpr uint32_t SLOT_BYTES = SLOT_ELEMS * sizeof(ElementP);
    // P + PT per stage
    static constexpr uint32_t STAGE_BYTES = SLOT_BYTES * 2;
    static constexpr uint32_t CORE_BYTES = STAGE_BYTES * WS_STAGES;

    static_assert(std::is_same_v<ElementP, float>, "Workspace P tiles must be float");
    static_assert(WS_STAGES >= 1 && WS_STAGES <= 2, "WS_STAGES must be 1 or 2");

    using EpilogueParams = typename BlockEpilogue::Params;

    struct Params {
        uint32_t batchCount;
        GemmCoord problemShape; // (M, M, K)
        GM_ADDR ptrX;
        GM_ADDR ptrY;
        GM_ADDR ptrD;
        GM_ADDR ptrWorkspace;
        LayoutX layoutX;
        LayoutXt layoutXt;
        LayoutY layoutY;
        LayoutD layoutD;
        int64_t strideX;
        int64_t strideY;
        int64_t strideD;
        float alpha;
        float beta;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            uint32_t batchCount_, GemmCoord const& problemShape_, GM_ADDR ptrX_, GM_ADDR ptrY_, GM_ADDR ptrD_,
            GM_ADDR ptrWorkspace_, LayoutX layoutX_, LayoutXt layoutXt_, LayoutY layoutY_, LayoutD layoutD_,
            int64_t strideX_, int64_t strideY_, int64_t strideD_, float alpha_, float beta_)
            : batchCount(batchCount_),
              problemShape(problemShape_),
              ptrX(ptrX_),
              ptrY(ptrY_),
              ptrD(ptrD_),
              ptrWorkspace(ptrWorkspace_),
              layoutX(layoutX_),
              layoutXt(layoutXt_),
              layoutY(layoutY_),
              layoutD(layoutD_),
              strideX(strideX_),
              strideY(strideY_),
              strideD(strideD_),
              alpha(alpha_),
              beta(beta_)
        {}
    };

    struct Arguments {
        uint32_t batchCount;
        GemmCoord problemShape; // (M, M, K)
        uint8_t* ptrX;
        uint8_t* ptrY;
        uint8_t* ptrD;
        float alpha;
        float beta;
        /// Cores this launch uses; each writes its own CORE_BYTES workspace slice.
        uint32_t aicCoreNum;
    };

    static bool CanImplement(const Arguments& args)
    {
        return args.problemShape.m() == args.problemShape.n() && args.aicCoreNum > 0;
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        return static_cast<size_t>(args.aicCoreNum) * CORE_BYTES;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        uint32_t m = args.problemShape.m();
        uint32_t k = args.problemShape.k();
        int64_t strideX = static_cast<int64_t>(m) * k;
        int64_t strideY = static_cast<int64_t>(m) * m;
        int64_t strideD = strideY;
        using LayoutTagY = layout::RowMajor;
        return Params{
            args.batchCount,
            args.problemShape,
            args.ptrX,
            args.ptrY,
            args.ptrD,
            workspace,
            tla::MakeLayout<ElementX, LayoutTagX>(m, k),
            tla::MakeLayout<ElementXt, LayoutTagXt>(k, m),
            tla::MakeLayout<ElementY, LayoutTagY>(m, m),
            tla::MakeLayout<ElementD, LayoutTagY>(m, m),
            strideX,
            strideY,
            strideD,
            args.alpha,
            args.beta,
        };
    }

    CATLASS_DEVICE
    BasicSyrkWorkspaceTla()
    {
        if ASCEND_IS_AIV {
            for (uint32_t i = 0; i < WS_STAGES; ++i) {
                AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + i);
            }
        }
    }

    CATLASS_DEVICE
    ~BasicSyrkWorkspaceTla()
    {
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < WS_STAGES; ++i) {
                AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_FIX>(AIV_SYNC_AIC_FLAG + i);
            }
        }
    }

    template <int32_t CoreType_ = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler blockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = params.batchCount * blockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX*)params.ptrX);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t wsStage = 0;
        LayoutP layoutSlotP = tla::MakeLayout<ElementP, LayoutTagP>(L1_TILE_M, L1_TILE_N);
        LayoutPT layoutSlotPT = tla::MakeLayout<ElementP, LayoutTagPT>(L1_TILE_M, L1_TILE_N);

        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            uint32_t batchIdx = blockScheduler.GetBatchIdx(loopIdx);
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);

            int64_t batchOffsetX = static_cast<int64_t>(batchIdx) * params.strideX;
            auto tensorX = tla::MakeTensor(gmX[batchOffsetX], params.layoutX, Arch::PositionGM{});
            auto tensorXt = tla::MakeTensor(gmX[batchOffsetX], params.layoutXt, Arch::PositionGM{});

            GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
            auto tileX = tla::GetTile(
                tensorX, tla::MakeCoord(blockCoord.m() * L1_TILE_M, 0),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tileXt = tla::GetTile(
                tensorXt, tla::MakeCoord(0, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

            GM_ADDR stageBase = params.ptrWorkspace + coreIdx * CORE_BYTES + wsStage * STAGE_BYTES;
            AscendC::GlobalTensor<ElementP> gmP;
            gmP.SetGlobalBuffer((__gm__ ElementP*)stageBase);
            AscendC::GlobalTensor<ElementP> gmPT;
            gmPT.SetGlobalBuffer((__gm__ ElementP*)(stageBase + SLOT_BYTES));
            auto tensorP = tla::MakeTensor(gmP, layoutSlotP, Arch::PositionGM{});
            auto tensorPT = tla::MakeTensor(gmPT, layoutSlotPT, Arch::PositionGM{});
            auto tileP =
                tla::GetTile(tensorP, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
            auto tilePT = tla::GetTile(
                tensorPT, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_FIX>(AIV_SYNC_AIC_FLAG + wsStage);

            blockMmad(tileX, tileXt, tileP, tilePT, actualBlockShape, blockCoord);

            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_FIX>(AIC_SYNC_AIV_FLAG + wsStage);
            wsStage = (wsStage + 1 < WS_STAGES) ? (wsStage + 1) : 0;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        BlockScheduler blockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = params.batchCount * blockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        EpilogueParams epiParams{params.alpha, params.beta, params.ptrY, params.ptrD, params.layoutY, params.layoutD};
        BlockEpilogue blockEpilogue(resource, epiParams);

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t aicBlockIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicBlockNum = AscendC::GetBlockNum();
        uint32_t wsStage = 0;

        for (uint32_t loopIdx = aicBlockIdx; loopIdx < coreLoops; loopIdx += aicBlockNum) {
            uint32_t batchIdx = blockScheduler.GetBatchIdx(loopIdx);
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);

            GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
            uint32_t mActual = actualBlockShape.m();
            uint32_t nActual = actualBlockShape.n();
            uint32_t mStart = blockCoord.m() * L1_TILE_M;
            uint32_t nStart = blockCoord.n() * L1_TILE_N;
            bool dualWrite = blockCoord.m() > blockCoord.n();
            int64_t batchOffsetY = static_cast<int64_t>(batchIdx) * params.strideY;

            GM_ADDR stageBase = params.ptrWorkspace + aicBlockIdx * CORE_BYTES + wsStage * STAGE_BYTES;

            AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE2>(AIC_SYNC_AIV_FLAG + wsStage);

            if (subBlockIdx == 0) {
                AscendC::GlobalTensor<ElementP> gmP;
                gmP.SetGlobalBuffer((__gm__ ElementP*)stageBase);
                auto layoutP = tla::MakeLayout(
                    tla::MakeShape(mActual, nActual), tla::MakeStride(static_cast<int64_t>(L1_TILE_N), tla::Int<1>{}));
                auto tensorP = tla::MakeTensor(gmP, layoutP, Arch::PositionGM{});
                blockEpilogue(tensorP, mStart, nStart, mActual, nActual, batchOffsetY);
            } else if (dualWrite) {
                // ColumnMajor(L1_M, L1_N) store ≡ RowMajor(nActual, mActual) with row stride L1_M.
                AscendC::GlobalTensor<ElementP> gmPT;
                gmPT.SetGlobalBuffer((__gm__ ElementP*)(stageBase + SLOT_BYTES));
                auto layoutPT = tla::MakeLayout(
                    tla::MakeShape(nActual, mActual), tla::MakeStride(static_cast<int64_t>(L1_TILE_M), tla::Int<1>{}));
                auto tensorPT = tla::MakeTensor(gmPT, layoutPT, Arch::PositionGM{});
                blockEpilogue(tensorPT, nStart, mStart, nActual, mActual, batchOffsetY);
            }

            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + wsStage);
            wsStage = (wsStage + 1 < WS_STAGES) ? (wsStage + 1) : 0;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // Mode 2 synchronizes one AIC with both AIV subblocks using one flag ID.
    static constexpr uint16_t CROSS_CORE_SYNC_MODE_2 = 2;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 8;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_BASIC_SYRK_WORKSPACE_TLA_HPP
