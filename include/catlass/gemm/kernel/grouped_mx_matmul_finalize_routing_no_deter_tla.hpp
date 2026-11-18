/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_FINALIZE_ROUTING_NO_DETER_TLA_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_FINALIZE_ROUTING_NO_DETER_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/block/block_swizzle_grouped_aswt.hpp"
#include "catlass/epilogue/block/block_epilogue_finalize_routing_no_deter.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

#if (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202))

// Mix (AIC+AIV) grouped MX matmul kernel variant driven by GemmGroupedAswtTailSplitSwizzle.
//
// 与 GroupedMxMatmulFinalizeRoutingTla 的差异（仅调度方式）：
//   * AIC/AIV 两侧的 BlockScheduler 由 ColumnBlockSwizzle 换成
//     GemmGroupedAswtTailSplitSwizzle：采用滚动核分配 + 窗口调度，
//     startBlockIdx_ 跨 group 滚动，提升多核利用率和尾块负载均衡。
//   * AswtTailSplit 支持尾部 tile 多核拆分（UpdateTailTile），
//     最后一个 group 在满足条件时自动启用。
//   * GetBlockShape 返回 AswtBlockShape{m, n, mOffset, nOffset}，
//     tile 坐标需加上 mOffset/nOffset 偏移。
//   * 其余（workspace 布局、AIC/AIV 同步、AIV SubBlock 列分裂）保持不变。
template <
    class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class ElementGroupList_, class ElementSharedInput_>
class GroupedMxMatmulFinalizeRoutingNoDeterTla {
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
    using ElementBias = typename BlockMmad::ElementBias;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementRowIndex = int64_t;

    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    using ElementGroupList = ElementGroupList_;
    using ElementSharedInput = ElementSharedInput_;
    static constexpr uint32_t UB_STAGES = BlockEpilogue::UB_STAGES;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        uint32_t aicCoreNum;
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList* ptrGroupList;
        __gm__ ElementA* ptrA;
        LayoutA layoutA;
        __gm__ ElementB* ptrB;
        LayoutB layoutB;
        __gm__ ElementMxScaleA* ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        __gm__ ElementMxScaleB* ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        __gm__ ElementC* ptrC;
        LayoutC layoutC;
        __gm__ bfloat16_t* ptrBias;
        __gm__ float* ptrLogit;
        __gm__ int64_t* ptrRowIndex;
        __gm__ ElementSharedInput* ptrSharedInput;
        uint32_t groupListType;
        float sharedInputWeight;
        int64_t sharedInputOffset;
        int64_t batchSize;
        int64_t bsdp;
        __gm__ float* ptrOut;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            uint32_t aicCoreNum_, GemmCoord const& problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_,
            GM_ADDR ptrA_, LayoutA const& layoutA_, GM_ADDR ptrB_, LayoutB const& layoutB_, GM_ADDR ptrMxScaleA_,
            LayoutMxScaleA layoutMxScaleA_, GM_ADDR ptrMxScaleB_, LayoutMxScaleB layoutMxScaleB_, GM_ADDR ptrC_,
            LayoutC const& layoutC_, GM_ADDR ptrBias_, GM_ADDR ptrLogit_, GM_ADDR ptrRowIndex_, GM_ADDR ptrSharedInput_,
            uint32_t groupListType_, float sharedInputWeight_, int64_t sharedInputOffset_, int64_t batchSize_,
            int64_t bsdp_, GM_ADDR ptrOut_)
            : aicCoreNum(aicCoreNum_),
              problemShape(problemShape_),
              problemCount(problemCount_),
              ptrGroupList(reinterpret_cast<__gm__ ElementGroupList*>(ptrGroupList_)),
              ptrA(reinterpret_cast<__gm__ ElementA*>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB*>(ptrB_)),
              layoutB(layoutB_),
              ptrMxScaleA(reinterpret_cast<__gm__ ElementMxScaleA*>(ptrMxScaleA_)),
              layoutMxScaleA(layoutMxScaleA_),
              ptrMxScaleB(reinterpret_cast<__gm__ ElementMxScaleB*>(ptrMxScaleB_)),
              layoutMxScaleB(layoutMxScaleB_),
              ptrC(reinterpret_cast<__gm__ ElementC*>(ptrC_)),
              layoutC(layoutC_),
              ptrBias(reinterpret_cast<__gm__ bfloat16_t*>(ptrBias_)),
              ptrLogit(reinterpret_cast<__gm__ float*>(ptrLogit_)),
              ptrRowIndex(reinterpret_cast<__gm__ int64_t*>(ptrRowIndex_)),
              ptrSharedInput(reinterpret_cast<__gm__ ElementSharedInput*>(ptrSharedInput_)),
              groupListType(groupListType_),
              sharedInputWeight(sharedInputWeight_),
              sharedInputOffset(sharedInputOffset_),
              batchSize(batchSize_),
              bsdp(bsdp_),
              ptrOut(reinterpret_cast<__gm__ float*>(ptrOut_))
        {}
    };

    struct Arguments {
        uint32_t aicCoreNum;
        GemmCoord problemShape;
        uint32_t problemCount;
        uint8_t* ptrGroupList;
        uint8_t* ptrA;
        LayoutA layoutA;
        uint8_t* ptrB;
        LayoutB layoutB;
        uint8_t* ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        uint8_t* ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        uint8_t* ptrC;
        LayoutC layoutC;
        uint8_t* ptrBias;
        uint8_t* ptrLogit;
        uint8_t* ptrRowIndex;
        uint8_t* ptrSharedInput;
        uint32_t groupListType;
        float sharedInputWeight;
        int64_t sharedInputOffset;
        int64_t batchSize;
        int64_t bsdp;
        uint8_t* ptrOut;
    };

    static bool CanImplement(const Arguments& args)
    {
        return AscendC::Std::is_one_of_v<ElementA, float8_e4m3_t, float8_e5m2_t> &&
               AscendC::Std::is_one_of_v<ElementB, float8_e4m3_t, float8_e5m2_t> &&
               std::is_same_v<ElementMxScaleA, float8_e8m0_t> && std::is_same_v<ElementMxScaleB, float8_e8m0_t>;
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        // AIC 直写 AIV UB，不再需要 GM workspace
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        Params params{args.aicCoreNum,
                      args.problemShape,
                      args.problemCount,
                      args.ptrGroupList,
                      args.ptrA,
                      args.layoutA,
                      args.ptrB,
                      args.layoutB,
                      args.ptrMxScaleA,
                      args.layoutMxScaleA,
                      args.ptrMxScaleB,
                      args.layoutMxScaleB,
                      workspace,
                      args.layoutC,
                      args.ptrBias,
                      args.ptrLogit,
                      args.ptrRowIndex,
                      args.ptrSharedInput,
                      args.groupListType,
                      args.sharedInputWeight,
                      args.sharedInputOffset,
                      args.batchSize,
                      args.bsdp,
                      args.ptrOut};
        return params;
    }

    CATLASS_DEVICE
    GroupedMxMatmulFinalizeRoutingNoDeterTla()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        AscendC::ICachePreLoad(1);
        BlockMmad blockMmad(resource);
        BlockEpilogue blockEpilogue(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetMxScaleA = 0;
        int64_t gmGroupOffsetMxScaleB = 0;
        int64_t gmGroupOffsetBias = 0;
        int64_t mxScaleAlignedK =
            static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(params.problemShape.k()) * MX_SCALE_COPY_GROUP_NUM);
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        int64_t totalM = 0;

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});

        // AIC 直写 AIV UB：通过 epilogue 获取 UB tensor，创建 PositionUB 的 tla tensor
        auto ubLocalTemp = blockEpilogue.GetL0c2UbTensor();
        AscendC::LocalTensor<ElementC> cUb_;
        cUb_.SetAddr(ubLocalTemp.address_);
        auto alignN = RoundUp(static_cast<uint32_t>(params.problemShape.n()), 8u);
        auto layoutUb = tla::MakeLayout(
            tla::MakeShape(static_cast<uint32_t>(params.problemShape.m()), alignN),
            tla::MakeStride(static_cast<int64_t>(alignN), tla::Int<1>{}),
            tla::MakeShape(
                static_cast<uint32_t>(params.problemShape.m()), static_cast<uint32_t>(params.problemShape.n())));
        auto tensorC = tla::MakeTensor(cUb_, layoutUb, Arch::PositionUB{});

        uint32_t ubListId = 0;

        // AswtTailSplit scheduler: rolling core assignment across groups with window scheduling.
        BlockScheduler scheduler(L1_TILE_M, L1_TILE_N);

        bool isFirstTile = true;
        uint32_t lastGroupIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t groupValue = groupList.GetValue(groupIdx);
            uint32_t currentM = groupValue;
            if (params.groupListType == 0) {
                currentM = groupValue - lastGroupIdx;
                lastGroupIdx = groupValue;
            }
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(
                inGroupProblemShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(inGroupProblemShape.k()));

            scheduler.UpdateNextProblem(inGroupProblemShape);

            // Last group: split the tail tiles to fill the otherwise idle cores of the final wave.
            bool isLastGroup = (groupIdx + 1 == params.problemCount);
            bool doTailSplit = isLastGroup && scheduler.NeedTailSplit();
            if (doTailSplit) {
                scheduler.UpdateTailTile();
            }

            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(params.ptrB + gmGroupOffsetB);
            AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
            gmMxScaleB.SetGlobalBuffer(params.ptrMxScaleB + gmGroupOffsetMxScaleB);

            AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
            gmMxScaleA.SetGlobalBuffer(params.ptrMxScaleA + gmGroupOffsetMxScaleA);

            using GlobalTensorBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
            AscendC::GlobalTensor<GlobalTensorBiasType> gmBias;
            if constexpr (!std::is_void_v<ElementBias>) {
                gmBias.SetGlobalBuffer((__gm__ ElementBias*)params.ptrBias + gmGroupOffsetBias);
            }

            auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
            auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, layoutMxScaleA, Arch::PositionGM{});
            auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Arch::PositionGM{});
            auto layoutBias = tla::MakeLayout(params.problemShape.n());
            auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});

            // AswtTailSplit: 滚动核分配 + 窗口调度。AIC/AIV 必须使用完全相同的
            // 遍历范围与顺序，以保证 AIV 读取的 UB 数据与 AIC Fixpipe 写入的块一一对应。
            GemmCoord blockCoord;
            while (scheduler.GetTileIdx(blockCoord)) {
                auto shape = scheduler.GetBlockShape(blockCoord);
                if (shape.m == 0 || shape.n == 0) {
                    continue;
                }

                uint32_t mInGroup = blockCoord.m() * L1_TILE_M + shape.mOffset;
                uint32_t nOffset = blockCoord.n() * L1_TILE_N + shape.nOffset;
                int64_t mGlobal = totalM + static_cast<int64_t>(mInGroup);
                GemmCoord actualBlockShape{shape.m, shape.n, inGroupProblemShape.k()};

                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(mGlobal, static_cast<uint32_t>(0)),
                    tla::MakeShape(shape.m, inGroupProblemShape.k()));

                auto tensorBlockB = GetTile(
                    tensorB, tla::MakeCoord(static_cast<uint32_t>(0), nOffset),
                    tla::MakeShape(inGroupProblemShape.k(), shape.n));

                alignN = RoundUp(static_cast<uint32_t>(shape.n), 8u);
                layoutUb = tla::MakeLayout(
                    tla::MakeShape(static_cast<uint32_t>(shape.m), alignN),
                    tla::MakeStride(static_cast<int64_t>(alignN), tla::Int<1>{}),
                    tla::MakeShape(static_cast<uint32_t>(shape.m), static_cast<uint32_t>(shape.n)));
                auto tensorBlockC = tla::MakeTensor(cUb_, layoutUb, Arch::PositionUB{});

                auto tensorBlockMxScaleA = GetTile(
                    tensorMxScaleA, tla::MakeCoord(mInGroup, static_cast<uint32_t>(0)),
                    tla::MakeShape(shape.m, CeilDiv<MX_SCALE_GROUP_NUM>(inGroupProblemShape.k())));

                auto tensorBlockMxScaleB = GetTile(
                    tensorMxScaleB, tla::MakeCoord(static_cast<uint32_t>(0), nOffset),
                    tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(inGroupProblemShape.k()), shape.n));

                if constexpr (!std::is_void_v<ElementBias>) {
                    auto tensorBlockBias = GetTile(tensorBias, tla::MakeCoord(nOffset), tla::MakeShape(shape.n));
                    blockMmad.template operator()<AIC_SYNC_AIV_MODE_2>(
                        tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, true, AIV_SYNC_AIC_FLAG + ubListId,
                        AIC_SYNC_AIV_FLAG + ubListId, tensorBlockMxScaleA, tensorBlockMxScaleB, tensorBlockBias);
                } else {
                    blockMmad.template operator()<AIC_SYNC_AIV_MODE_2>(
                        tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, true, AIV_SYNC_AIC_FLAG + ubListId,
                        AIC_SYNC_AIV_FLAG + ubListId, tensorBlockMxScaleA, tensorBlockMxScaleB);
                }
                isFirstTile = false;
            }

            totalM += inGroupProblemShape.m();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            if constexpr (!std::is_void_v<ElementBias>) {
                gmGroupOffsetBias += inGroupProblemShape.n();
            }
            gmGroupOffsetMxScaleA += inGroupProblemShape.m() * mxScaleAlignedK;
            gmGroupOffsetMxScaleB += mxScaleAlignedK * inGroupProblemShape.n();
        }

        // 如果该核从未进入循环（无任务分配），需主动 SetFlag 以避免 AIV 侧 final WaitFlag 死锁
        if (isFirstTile) {
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_2>(AIV_SYNC_AIC_FLAG + ubListId);
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_2, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + ubListId);
        }

        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_2>(AIV_SYNC_AIC_FLAG + ubListId);
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.template SynchronizeBlock<decltype(tensorC)>();
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        AscendC::ICachePreLoad(1);
        // mix kernel: AIV 的 GetBlockIdx() 返回子核级索引（0 ~ 2*aicCoreNum-1），
        // 需折叠到 block 级别以与 AIC 侧的 taskIdx 遍历对齐，确保 AIV 读取的 UB
        // 数据与 AIC 写入的块完全一致（同一 blockCoord / actualBlockShape）。
        // 注意：GetBlockNum() 始终返回 AIC 核总数（aicCoreNum），无需除以 subBlockNum。
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        if (subBlockNum != 1) {
            coreIdx /= subBlockNum;
        }

        OutSplitScheduler outSplitScheduler;
        BlockEpilogue blockEpilogue(resource);

        // AIV 侧原始（子核级）coreIdx/coreNum，用于 ClearOutTile / AssignSharedInputTile
        // 的 batch 维切分（这部分按子核并行，与 AIC 调度无关）。
        uint32_t rawCoreIdx = AscendC::GetBlockIdx();
        uint32_t rawCoreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffset = 0;

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        AscendC::GlobalTensor<ElementC> gmLogit;
        gmLogit.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(params.ptrLogit));
        AscendC::GlobalTensor<ElementRowIndex> gmRowIndex;
        gmRowIndex.SetGlobalBuffer(reinterpret_cast<__gm__ ElementRowIndex*>(params.ptrRowIndex));
        AscendC::GlobalTensor<ElementC> gmOut;
        gmOut.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(params.ptrOut));
        int64_t totalM = 0;
        uint32_t ubListId = 0;

        blockEpilogue.Update(params.problemShape);

        MatrixCoord outSplitCoord = outSplitScheduler.GetTask(params.batchSize, rawCoreIdx, rawCoreNum);
        blockEpilogue.ClearOutTile(
            gmOut[static_cast<int64_t>(outSplitCoord.row()) * params.problemShape.n()], outSplitCoord);

        if constexpr (!std::is_void_v<ElementSharedInput>) {
            AscendC::GlobalTensor<ElementSharedInput> gmSharedInput;
            gmSharedInput.SetGlobalBuffer(reinterpret_cast<__gm__ ElementSharedInput*>(params.ptrSharedInput));
            auto outSharedSplitCoord = outSplitScheduler.GetTask(params.bsdp, rawCoreIdx, rawCoreNum);
            AscendC::SyncAll();
            blockEpilogue.AssignSharedInputTile(
                gmSharedInput[static_cast<int64_t>(outSharedSplitCoord.row()) * params.problemShape.n()],
                gmOut
                    [static_cast<int64_t>(params.sharedInputOffset + outSharedSplitCoord.row()) *
                     params.problemShape.n()],
                outSharedSplitCoord, params.sharedInputWeight);
        }

        AscendC::SyncAll();
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_2, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + ubListId);
        BlockScheduler blockScheduler(L1_TILE_M, L1_TILE_N);
        uint32_t lastGroupIdx = 0;
        bool didAnyWork = false;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t groupValue = groupList.GetValue(groupIdx);
            uint32_t currentM = groupValue;
            if (params.groupListType == 0) {
                currentM = groupValue - lastGroupIdx;
                lastGroupIdx = groupValue;
            }
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            blockScheduler.UpdateNextProblem(inGroupProblemShape);
            blockEpilogue.Update(inGroupProblemShape);

            // Last group: split the tail tiles to fill the otherwise idle cores of the final wave.
            bool isLastGroup = (groupIdx + 1 == params.problemCount);
            bool doTailSplit = isLastGroup && blockScheduler.NeedTailSplit();
            if (doTailSplit) {
                blockScheduler.UpdateTailTile();
            }

            GemmCoord blockCoord;
            while (blockScheduler.GetTileIdx(blockCoord)) {
                auto shape = blockScheduler.GetBlockShape(blockCoord);
                if (shape.m == 0 || shape.n == 0) {
                    continue;
                }
                uint32_t mInGroup = blockCoord.m() * L1_TILE_M + shape.mOffset;
                uint32_t nOffset = blockCoord.n() * L1_TILE_N + shape.nOffset;
                // 按 M 维切分：AIV0 处理 (m+1)/2 行，AIV1 处理剩余行
                uint32_t coreZeroM = CeilDiv(shape.m, 2);
                uint32_t coreOneM = shape.m - coreZeroM;
                uint32_t curM = AscendC::GetSubBlockIdx() == 0 ? coreZeroM : coreOneM;
                uint32_t curMOffset = AscendC::GetSubBlockIdx() == 0 ? 0 : coreZeroM;

                int64_t gmOffsetLogit = gmGroupOffset + mInGroup + curMOffset;
                GemmCoord workBlockShape = GemmCoord{curM, shape.n, 0};

                blockEpilogue.template LogitScatterAddTileFromUb<AIC_SYNC_AIV_MODE_2>(
                    gmLogit[gmOffsetLogit], gmRowIndex[gmOffsetLogit], gmOut[nOffset], workBlockShape,
                    AIC_SYNC_AIV_FLAG + ubListId, AIV_SYNC_AIC_FLAG + ubListId);
                didAnyWork = true;
            }

            totalM += inGroupProblemShape.m();
            gmGroupOffset += inGroupProblemShape.m();
        }

        // 如果该核从未进入循环（无任务分配），需主动 SetFlag 以避免 AIC 侧 final WaitFlag 死锁
        if (!didAnyWork) {
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_2, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + ubListId);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    constexpr static uint16_t AIC_SYNC_AIV_MODE_2 = 2;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 4;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 6;
    constexpr static uint16_t FLAG_ID_MAX = 16;
    struct OutSplitScheduler {
        CATLASS_DEVICE
        OutSplitScheduler()
        {}

        CATLASS_DEVICE
        MatrixCoord GetTask(uint32_t batch, uint32_t coreIdx, uint32_t aicoreNum)
        {
            uint32_t coreNum = aicoreNum * 2;
            uint32_t perCoreRow = batch / coreNum;
            uint32_t remainRow = batch % coreNum;
            uint32_t rowStart = coreIdx * perCoreRow;
            uint32_t curCoreRow = perCoreRow;
            if (coreIdx < remainRow) {
                rowStart += coreIdx;
                curCoreRow += 1;
            } else {
                rowStart += remainRow;
            }
            return MatrixCoord(rowStart, curCoreRow);
        }
    };
};

#endif // (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202))

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_GROUPED_MX_MATMUL_FINALIZE_ROUTING_NO_DETER_TLA_HPP
