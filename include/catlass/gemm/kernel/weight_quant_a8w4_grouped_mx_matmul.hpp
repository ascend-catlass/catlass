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

#ifndef CATLASS_GEMM_KERNEL_WEIGHT_QUANT_A8W4_GROUPED_MX_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_WEIGHT_QUANT_A8W4_GROUPED_MX_MATMUL_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

#if (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202))

// Template for MxMatmul kernel. Compute C = (MxScaleA x A) * (MxScaleB x B) + Bias
template <class BlockMmad_, class BlockPrologue_, class BlockEpilogue_, class BlockScheduler_, class ElementGroupList_>
class A8W4GroupedMxMatmul {
public:
    using BlockMmad = BlockMmad_;
    using BlockPrologue = BlockPrologue_;
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
    using ElementBias = typename BlockMmad::ElementBias;
    using ElementPrologueB = typename BlockMmad::TileCopy::ElementPrologueB;
    using LayoutPrologueB = typename BlockMmad::TileCopy::LayoutPrologueB;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using LayoutTagL1B = typename BlockMmad::LayoutTagL1B;

    using BlockScheduler = BlockScheduler_;
    using ElementGroupList = ElementGroupList_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    // L1B STAGES
    static constexpr uint32_t L1B_STAGES = 2;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        GM_ADDR ptrGroupList;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutPrologueB layoutPrologueB;
        GM_ADDR ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        GM_ADDR ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrBias;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_, GM_ADDR ptrA_,
            LayoutA layoutA_, GM_ADDR ptrB_, LayoutPrologueB layoutPrologueB_, GM_ADDR ptrMxScaleA_,
            LayoutMxScaleA layoutMxScaleA_, GM_ADDR ptrMxScaleB_, LayoutMxScaleB layoutMxScaleB_, GM_ADDR ptrC_,
            LayoutC layoutC_, GM_ADDR ptrBias_ = nullptr)
            : problemShape(problemShape_),
              problemCount(problemCount_),
              ptrGroupList(ptrGroupList_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutPrologueB(layoutPrologueB_),
              ptrMxScaleA(ptrMxScaleA_),
              layoutMxScaleA(layoutMxScaleA_),
              ptrMxScaleB(ptrMxScaleB_),
              layoutMxScaleB(layoutMxScaleB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              ptrBias(ptrBias_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t problemCount;
        uint8_t* ptrGroupList;
        uint8_t* ptrA;
        LayoutA layoutA;
        uint8_t* ptrB;
        LayoutPrologueB layoutPrologueB;
        uint8_t* ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        uint8_t* ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        uint8_t* ptrC;
        LayoutC layoutC;
        uint8_t* ptrBias{nullptr};
    };

    static bool CanImplement(const Arguments& args)
    {
        return AscendC::Std::is_one_of_v<ElementA, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
               AscendC::Std::is_one_of_v<ElementB, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
               std::is_same_v<ElementMxScaleA, float8_e8m0_t> && std::is_same_v<ElementMxScaleB, float8_e8m0_t>;
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        Params params{args.problemShape,   args.problemCount, args.ptrGroupList,    args.ptrA,
                      args.layoutA,        args.ptrB,         args.layoutPrologueB, args.ptrMxScaleA,
                      args.layoutMxScaleA, args.ptrMxScaleB,  args.layoutMxScaleB,  args.ptrC,
                      args.layoutC,        args.ptrBias};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    A8W4GroupedMxMatmul()
    {
        if constexpr (tla::detail::isRowMajor<LayoutC>::value) {
            AscendC::SetMMLayoutTransform(true);
        }
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_MTE1>(AIC_SYNC_AIV_FLAG + i);
#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ != 9201 && __NPU_ARCH__ != 9202)
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_MTE1>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX + i);
#endif
            }
        }
    }

    // Destructor
    CATLASS_DEVICE
    ~A8W4GroupedMxMatmul()
    {
        if ASCEND_IS_AIV {
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + i);
            }
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        BlockMmad blockMmad(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer((__gm__ ElementGroupList*)params.ptrGroupList);

        // Setting L2 Cache to Disable, data reads will bypass L2 Cache.
        if (CeilDiv(params.problemShape.n(), L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        using GlobalTensorBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
        AscendC::GlobalTensor<GlobalTensorBiasType> gmBias;

        auto layoutBias = tla::MakeLayout(params.problemShape.n());

        int64_t gmGroupOffsetMxScaleA = 0;
        int64_t gmGroupOffsetMxScaleB = 0;
        int64_t mxScaleAlignedK =
            static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(params.problemShape.k()) * MX_SCALE_COPY_GROUP_NUM);

        int64_t totalM = 0;
        uint32_t startCoreIdx = 0;

        // Represent the full tensors
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
        auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = groupList.GetValue(groupIdx);
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            auto tileM = GetTileM(currentM);

            auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(
                inGroupProblemShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(inGroupProblemShape.k()));

            BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(tileM, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
            gmMxScaleB.SetGlobalBuffer((__gm__ ElementMxScaleB*)params.ptrMxScaleB + gmGroupOffsetMxScaleB);

            AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
            gmMxScaleA.SetGlobalBuffer((__gm__ ElementMxScaleA*)params.ptrMxScaleA + gmGroupOffsetMxScaleA);

            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }

            auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, layoutMxScaleA, Arch::PositionGM{});
            auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Arch::PositionGM{});

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(totalM + blockCoord.m() * tileM, blockCoord.k() * L1_TILE_K),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));

                auto tensorBlockC = GetTile(
                    tensorC, tla::MakeCoord(totalM + blockCoord.m() * tileM, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                auto tensorBlockMxScaleA = GetTile(
                    tensorMxScaleA,
                    tla::MakeCoord(blockCoord.m() * tileM, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                    tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k())));

                auto tensorBlockMxScaleB = GetTile(
                    tensorMxScaleB,
                    tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));

                if constexpr (std::is_void_v<ElementBias>) {
                    blockMmad(tensorBlockA, tensorBlockC, actualBlockShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
                } else {
                    auto tensorBlockBias = GetTile(
                        tensorBias, tla::MakeCoord(blockCoord.n() * L1_TILE_N), tla::MakeShape(actualBlockShape.n()));
                    blockMmad(
                        tensorBlockA, tensorBlockC, actualBlockShape, tensorBlockMxScaleA, tensorBlockMxScaleB,
                        tensorBlockBias);
                }
            }

            totalM += inGroupProblemShape.m();

            gmGroupOffsetMxScaleA += inGroupProblemShape.m() * mxScaleAlignedK;
            gmGroupOffsetMxScaleB += mxScaleAlignedK * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.template SynchronizeBlock<decltype(tensorC)>();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        using PrologueParams = typename BlockPrologue::Params;
        PrologueParams prologueParams{L1TileShape{}, params.layoutPrologueB, false, resource};

        BlockPrologue blockPrologue(prologueParams);

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer((__gm__ ElementGroupList*)params.ptrGroupList);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetB = 0;
        uint32_t startCoreIdx = 0;

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = groupList.GetValue(groupIdx);

            // Represent the full gmB
            AscendC::GlobalTensor<ElementPrologueB> gmB;
            gmB.SetGlobalBuffer((__gm__ ElementPrologueB*)params.ptrB + gmGroupOffsetB);
            auto tensorGmB = tla::MakeTensor(gmB, params.layoutPrologueB, Arch::PositionGM{});

            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
            auto tileM = GetTileM(currentM);
            BlockScheduler matmulBlockScheduler(inGroupProblemShape, MakeCoord(tileM, L1_TILE_N));

            if (CeilDiv(currentM, tileM) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
            }

            // veccore
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
                auto blockTensorB = GetTile(
                    tensorGmB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                blockPrologue(blockTensorB, actualBlockShape, prologueParams);
            }

            gmGroupOffsetB += static_cast<int64_t>(CeilDiv<2>(inGroupProblemShape.k())) * inGroupProblemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE
    uint32_t GetTileM(uint32_t mSize)
    {
        if (mSize <= L1_TILE_M) {
            return L1_TILE_M;
        } else {
            uint32_t tileNum = CeilDiv(mSize, L1_TILE_M);
            return CeilDiv((mSize / tileNum), TILE_M_ALIGNMENT) * TILE_M_ALIGNMENT;
        }
    }

    Arch::Resource<ArchTag> resource;

    constexpr static uint8_t AIC_SYNC_AIV_MODE = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;
    constexpr static uint16_t FLAG_ID_MAX = 16;
    constexpr static uint16_t TILE_M_ALIGNMENT = 16;
};

#endif // (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202))

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_WEIGHT_QUANT_A8W4_GROUPED_MX_MATMUL_HPP
