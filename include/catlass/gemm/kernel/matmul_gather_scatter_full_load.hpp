/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_GATHER_SCATTER_FULL_LOAD_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_GATHER_SCATTER_FULL_LOAD_HPP

#include <type_traits>

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue_matmul_gather_scatter.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

struct MatmulGatherScatterFullLoadParams {
    GemmCoord problemShape;
    uint32_t physicalM;
    GM_ADDR ptrA;
    GM_ADDR ptrB;
    GM_ADDR ptrIndices;
    GM_ADDR ptrD;
    uint32_t aicCoreNum;

    CATLASS_HOST_DEVICE MatmulGatherScatterFullLoadParams()
    {}

    CATLASS_HOST_DEVICE MatmulGatherScatterFullLoadParams(
        GemmCoord const& problemShape_, uint32_t physicalM_, GM_ADDR ptrA_, GM_ADDR ptrB_, GM_ADDR ptrIndices_,
        GM_ADDR ptrD_, uint32_t aicCoreNum_)
        : problemShape(problemShape_),
          physicalM(physicalM_),
          ptrA(ptrA_),
          ptrB(ptrB_),
          ptrIndices(ptrIndices_),
          ptrD(ptrD_),
          aicCoreNum(aicCoreNum_)
    {}
};

namespace detail {

constexpr uint16_t MGS_SYNC_MODE = 4;
// Mode-4 flags for AIV0 and AIV1 use disjoint 16-entry ranges.
constexpr uint16_t MGS_AIV_EVENT_ID_STRIDE = 16;
constexpr uint16_t MGS_GATHER_READY_FLAG = 0;
constexpr uint16_t MGS_C_READY_FLAG = 2;
constexpr uint16_t MGS_BUFFER_FREE_FLAG = 4;
constexpr uint16_t MGS_A_RELOAD_SAFE_FLAG = 6;
// Two AIVs can gather 64 rows up to K=768 within the FullLoad UB window.
constexpr uint32_t MGS_AIV_GATHER_MAX_K = 768;
constexpr uint32_t MGS_FP16_ELEMENTS_PER_BLOCK = BYTE_PER_BLK / sizeof(half);
constexpr uint16_t MGS_EVENT_AIC_GATHER = 7;

static_assert(MGS_A_RELOAD_SAFE_FLAG < MGS_AIV_EVENT_ID_STRIDE, "AIV mode-4 event ranges overlap");

CATLASS_HOST_DEVICE constexpr uint16_t AivEventId(uint16_t baseEventId, uint32_t subBlockIdx)
{
    return baseEventId + subBlockIdx * MGS_AIV_EVENT_ID_STRIDE;
}

} // namespace detail

/**
 * MIX kernel for D[indices, :] = A[indices, :] @ B.
 *
 * The AIC performs FullLoadA Cube Matmul. The AIV performs output clearing,
 * optional Gather-to-L1, and Scatter-to-GM. All cross-core synchronization is
 * intentionally kept at this kernel layer.
 */
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class MatmulGatherScatterFullLoadKernelBase {
public:
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementTransport = typename BlockMmad::ElementC;
    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using Params = MatmulGatherScatterFullLoadParams;

    static constexpr bool AIV_GATHER = BlockEpilogue::AIV_GATHER;
    static constexpr bool DUAL_AIV_OUTPUT = BlockEpilogue::DUAL_AIV_OUTPUT;
    static constexpr uint32_t UB_SIZE = BlockEpilogue::UB_SIZE;
    static constexpr uint32_t AIV_SUBBLOCKS = BlockEpilogue::AIV_SUBBLOCKS;
    static constexpr uint32_t L1_TILE_M = BlockEpilogue::L1_TILE_M;
    static constexpr uint32_t L1_TILE_N = BlockEpilogue::L1_TILE_N;
    static constexpr uint32_t UB_STAGES = BlockEpilogue::UB_STAGES;

    static_assert(
        std::is_same_v<ElementA, half> && std::is_same_v<ElementB, half>,
        "MatmulGatherScatter currently supports FP16 A/B only");
    struct Arguments {
        GemmCoord problemShape;
        uint32_t physicalM;
        uint8_t* ptrA;
        uint8_t* ptrB;
        uint8_t* ptrIndices;
        uint8_t* ptrD;
        uint32_t aicCoreNum;
    };

    static bool CanImplement(Arguments const& args)
    {
        if (args.problemShape.m() == 0 || args.problemShape.n() == 0 || args.problemShape.k() == 0 ||
            args.physicalM == 0 || args.problemShape.m() > args.physicalM || args.aicCoreNum == 0 ||
            (AIV_GATHER && args.problemShape.k() > BlockEpilogue::MAX_GATHER_K)) {
            return false;
        }
        uint64_t l1Bytes = static_cast<uint64_t>(L1_TILE_M) * args.problemShape.k() * sizeof(ElementA) +
                           static_cast<uint64_t>(BlockMmad::L1B_TILE_SIZE) * BlockMmad::L1B_STAGES;
        return l1Bytes <= ArchTag::L1_SIZE && (args.problemShape.k() % detail::MGS_FP16_ELEMENTS_PER_BLOCK == 0U);
    }

    static size_t GetWorkspaceSize(Arguments const&)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(Arguments const& args, uint8_t* workspace)
    {
        (void)workspace;
        return Params{args.problemShape, args.physicalM, args.ptrA,      args.ptrB,
                      args.ptrIndices,   args.ptrD,      args.aicCoreNum};
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        Arch::Resource<ArchTag> resource;
        BlockScheduler scheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrA));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrB));
        AscendC::GlobalTensor<int32_t> gmIndices;
        gmIndices.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrIndices));

        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(params.physicalM, params.problemShape.k());
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(params.problemShape.k(), params.problemShape.n());
        auto tensorA = tla::MakeTensor(gmA, layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, layoutB, Arch::PositionGM{});

        uint32_t firstLoopIdx = AscendC::GetBlockIdx();
        if (firstLoopIdx >= params.aicCoreNum) {
            firstLoopIdx = coreLoops;
        }
        uint32_t previousM = UINT32_MAX;
        if (firstLoopIdx < coreLoops) {
            constexpr uint16_t outputSlot = 0;
            GemmCoord blockCoord = scheduler.GetBlockCoord(firstLoopIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            previousM = blockCoord.m();

            WaitBufferFree(outputSlot);
            PrepareL1A(resource, tensorA, gmIndices, blockCoord.m() * L1_TILE_M, actualShape.m(), actualShape.k());

            auto tensorTileA = GetTile(
                tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, 0),
                tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorTileB = GetTile(
                tensorB, tla::MakeCoord(0, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualShape.k(), actualShape.n()));
            uint32_t alignN = RoundUp<detail::MGS_FP16_ELEMENTS_PER_BLOCK>(actualShape.n());
            auto outputUb =
                resource.ubBuf.template GetBufferByByte<ElementTransport>(BlockEpilogue::GetOutputUbOffset(outputSlot));
            auto outputLayout = tla::MakeLayout<ElementTransport, layout::RowMajor>(actualShape.m(), alignN);
            auto tensorOutput = tla::MakeTensor(outputUb, outputLayout, Arch::PositionUB{});

            // The AIV path invokes BlockMmad's A-copy hook to establish its MTE2 event handshake;
            // MatmulGatherScatterTileCopy binds that hook to a no-op because A is already in L1.
            constexpr bool invokePreloadedACopyHook = AIV_GATHER;
            blockMmad(tensorTileA, tensorTileB, tensorOutput, actualShape, invokePreloadedACopyHook);
            PublishOutputReady(outputSlot);
        }

        AscendC::SyncAll<false>();

        for (uint32_t loopIdx = firstLoopIdx + params.aicCoreNum; loopIdx < coreLoops; loopIdx += params.aicCoreNum) {
            uint16_t outputSlot = (loopIdx / params.aicCoreNum) % UB_STAGES;
            GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            bool needGather = previousM != blockCoord.m();
            previousM = blockCoord.m();

            WaitBufferFree(outputSlot);
            if (needGather) {
                if constexpr (AIV_GATHER) {
                    PublishAReloadSafe();
                }
                PrepareL1A(resource, tensorA, gmIndices, blockCoord.m() * L1_TILE_M, actualShape.m(), actualShape.k());
            }

            auto tensorTileA = GetTile(
                tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, 0),
                tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorTileB = GetTile(
                tensorB, tla::MakeCoord(0, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualShape.k(), actualShape.n()));
            uint32_t alignN = RoundUp<detail::MGS_FP16_ELEMENTS_PER_BLOCK>(actualShape.n());
            auto outputUb =
                resource.ubBuf.template GetBufferByByte<ElementTransport>(BlockEpilogue::GetOutputUbOffset(outputSlot));
            auto outputLayout = tla::MakeLayout<ElementTransport, layout::RowMajor>(actualShape.m(), alignN);
            auto tensorOutput = tla::MakeTensor(outputUb, outputLayout, Arch::PositionUB{});

            bool invokePreloadedACopyHook = AIV_GATHER && needGather;
            blockMmad(tensorTileA, tensorTileB, tensorOutput, actualShape, invokePreloadedACopyHook);
            PublishOutputReady(outputSlot);
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.template SynchronizeBlock<decltype(tensorB)>();
        }
        for (uint16_t outputSlot = 0; outputSlot < UB_STAGES; ++outputSlot) {
            WaitBufferFree(outputSlot);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        Arch::Resource<ArchTag> resource;
        BlockEpilogue blockEpilogue(resource);
        AscendC::GlobalTensor<half> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(params.ptrD));
        AscendC::GlobalTensor<half> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(params.ptrA));
        AscendC::GlobalTensor<int32_t> gmIndices;
        gmIndices.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrIndices));

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        BlockScheduler scheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();
        uint32_t logicalAicIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        if (logicalAicIdx >= params.aicCoreNum) {
            logicalAicIdx = coreLoops;
        }
        uint32_t previousM = UINT32_MAX;
        auto indicesUb = blockEpilogue.GetIndicesUb();
        bool hasFirstTile = logicalAicIdx < coreLoops;

        if (hasFirstTile) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(logicalAicIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            blockEpilogue.LoadIndices(indicesUb, gmIndices, blockCoord.m() * L1_TILE_M, actualShape.m());
            if constexpr (AIV_GATHER) {
                blockEpilogue.GatherToL1(indicesUb, gmA, params.problemShape.k(), actualShape.m(), subBlockIdx, 0);
                PublishGatherReady(subBlockIdx);
            }
            previousM = blockCoord.m();

            if (DUAL_AIV_OUTPUT || subBlockIdx == 0) {
                PublishBufferFree(0, subBlockIdx);
            }
        }

        blockEpilogue.ZeroOutput(gmD, static_cast<uint64_t>(params.physicalM) * params.problemShape.n());
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll<false>();

        if (DUAL_AIV_OUTPUT || subBlockIdx == 0) {
            for (uint16_t outputSlot = hasFirstTile ? 1 : 0; outputSlot < UB_STAGES; ++outputSlot) {
                PublishBufferFree(outputSlot, subBlockIdx);
            }
        }

        if (hasFirstTile) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(logicalAicIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            WaitOutputReady(0, subBlockIdx);
            if (DUAL_AIV_OUTPUT || subBlockIdx == 0) {
                blockEpilogue.ScatterOutput(
                    indicesUb, gmD, actualShape, blockCoord.n() * L1_TILE_N, subBlockIdx, params.problemShape.n(), 0);
                PublishBufferFree(0, subBlockIdx);
            }
        }

        for (uint32_t loopIdx = logicalAicIdx + params.aicCoreNum; loopIdx < coreLoops; loopIdx += params.aicCoreNum) {
            uint16_t outputSlot = (loopIdx / params.aicCoreNum) % UB_STAGES;
            GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
            bool needGather = previousM != blockCoord.m();
            previousM = blockCoord.m();

            if (needGather) {
                if constexpr (AIV_GATHER) {
                    WaitAReloadSafe(subBlockIdx);
                }
                blockEpilogue.LoadIndices(indicesUb, gmIndices, blockCoord.m() * L1_TILE_M, actualShape.m());
                if constexpr (AIV_GATHER) {
                    blockEpilogue.GatherToL1(
                        indicesUb, gmA, params.problemShape.k(), actualShape.m(), subBlockIdx, outputSlot);
                    PublishGatherReady(subBlockIdx);
                }
            }

            WaitOutputReady(outputSlot, subBlockIdx);
            if (DUAL_AIV_OUTPUT || subBlockIdx == 0) {
                blockEpilogue.ScatterOutput(
                    indicesUb, gmD, actualShape, blockCoord.n() * L1_TILE_N, subBlockIdx, params.problemShape.n(),
                    outputSlot);
                PublishBufferFree(outputSlot, subBlockIdx);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    CATLASS_DEVICE static void WaitBufferFree(uint16_t outputSlot)
    {
        constexpr uint32_t consumers = DUAL_AIV_OUTPUT ? AIV_SUBBLOCKS : 1;
        for (uint32_t subBlockIdx = 0; subBlockIdx < consumers; ++subBlockIdx) {
            AscendC::CrossCoreWaitFlag<detail::MGS_SYNC_MODE, PIPE_FIX>(
                detail::AivEventId(detail::MGS_BUFFER_FREE_FLAG + outputSlot, subBlockIdx));
        }
    }

    CATLASS_DEVICE static void PublishOutputReady(uint16_t outputSlot)
    {
        for (uint32_t subBlockIdx = 0; subBlockIdx < AIV_SUBBLOCKS; ++subBlockIdx) {
            AscendC::CrossCoreSetFlag<detail::MGS_SYNC_MODE, PIPE_FIX>(
                detail::AivEventId(detail::MGS_C_READY_FLAG + outputSlot, subBlockIdx));
        }
    }

    CATLASS_DEVICE static void WaitOutputReady(uint16_t outputSlot, uint32_t subBlockIdx)
    {
        if constexpr (DUAL_AIV_OUTPUT) {
            AscendC::CrossCoreWaitFlag<detail::MGS_SYNC_MODE, PIPE_V>(
                detail::AivEventId(detail::MGS_C_READY_FLAG + outputSlot, subBlockIdx));
        } else {
            AscendC::CrossCoreWaitFlag<detail::MGS_SYNC_MODE, PIPE_MTE3>(
                detail::AivEventId(detail::MGS_C_READY_FLAG + outputSlot, subBlockIdx));
        }
    }

    CATLASS_DEVICE static void PublishBufferFree(uint16_t outputSlot, uint32_t subBlockIdx)
    {
        AscendC::CrossCoreSetFlag<detail::MGS_SYNC_MODE, PIPE_MTE3>(
            detail::AivEventId(detail::MGS_BUFFER_FREE_FLAG + outputSlot, subBlockIdx));
    }

    CATLASS_DEVICE static void PublishAReloadSafe()
    {
        for (uint32_t subBlockIdx = 0; subBlockIdx < AIV_SUBBLOCKS; ++subBlockIdx) {
            AscendC::CrossCoreSetFlag<detail::MGS_SYNC_MODE, PIPE_FIX>(
                detail::AivEventId(detail::MGS_A_RELOAD_SAFE_FLAG, subBlockIdx));
        }
    }

    CATLASS_DEVICE static void WaitAReloadSafe(uint32_t subBlockIdx)
    {
        AscendC::CrossCoreWaitFlag<detail::MGS_SYNC_MODE, PIPE_S>(
            detail::AivEventId(detail::MGS_A_RELOAD_SAFE_FLAG, subBlockIdx));
    }

    CATLASS_DEVICE static void PublishGatherReady(uint32_t subBlockIdx)
    {
        AscendC::CrossCoreSetFlag<detail::MGS_SYNC_MODE, PIPE_MTE3>(
            detail::AivEventId(detail::MGS_GATHER_READY_FLAG, subBlockIdx));
    }

    CATLASS_DEVICE static void WaitGatherReady()
    {
        for (uint32_t subBlockIdx = 0; subBlockIdx < AIV_SUBBLOCKS; ++subBlockIdx) {
            AscendC::CrossCoreWaitFlag<detail::MGS_SYNC_MODE, PIPE_MTE1>(
                detail::AivEventId(detail::MGS_GATHER_READY_FLAG, subBlockIdx));
        }
    }

    template <class Resource, class TensorA>
    CATLASS_DEVICE static void PrepareL1A(
        Resource& resource, TensorA& tensorA, AscendC::GlobalTensor<int32_t>& gmIndices, uint32_t jOffset,
        uint32_t rows, uint32_t k)
    {
        if constexpr (AIV_GATHER) {
            WaitGatherReady();
        } else {
            AicGatherToL1(resource, tensorA, gmIndices, jOffset, rows, k);
        }
    }

    template <class Resource, class TensorA>
    CATLASS_DEVICE static void AicGatherToL1(
        Resource& resource, TensorA& tensorA, AscendC::GlobalTensor<int32_t>& gmIndices, uint32_t jOffset,
        uint32_t rows, uint32_t k)
    {
        uint32_t l1AOffset = BlockMmad::L1B_TILE_SIZE * BlockMmad::L1B_STAGES;
        auto l1A = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset);
        auto l1Layout = tla::MakeLayout<ElementA, typename BlockMmad::LayoutTagL1A>(tla::Int<L1_TILE_M>{}, k);
        auto tensorL1A = tla::MakeTensor(l1A, l1Layout, Arch::PositionL1{});

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(detail::MGS_EVENT_AIC_GATHER);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(detail::MGS_EVENT_AIC_GATHER);
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t physicalRow = static_cast<uint32_t>(gmIndices.GetValue(jOffset + row));
            auto tensorGmRow = GetTile(tensorA, tla::MakeCoord(physicalRow, 0), tla::MakeShape(1U, k));
            auto tensorL1Row = GetTile(tensorL1A, tla::MakeCoord(row, 0), tla::MakeShape(1U, k));
            using CopyGmToL1 = Tile::TileCopyTla<ArchTag, decltype(tensorGmRow), decltype(tensorL1Row)>;
            CopyGmToL1 copyGmToL1;
            copyGmToL1(tensorL1Row, tensorGmRow);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(detail::MGS_EVENT_AIC_GATHER);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(detail::MGS_EVENT_AIC_GATHER);
    }
};

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class MatmulGatherScatterFullLoadAivGather
    : public MatmulGatherScatterFullLoadKernelBase<BlockMmad_, BlockEpilogue_, BlockScheduler_> {
    using Base = MatmulGatherScatterFullLoadKernelBase<BlockMmad_, BlockEpilogue_, BlockScheduler_>;

public:
    static_assert(Base::AIV_GATHER, "AIV Gather kernel requires an AIV Gather dispatch policy");
    using Base::operator();
};

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class MatmulGatherScatterFullLoadAicGather
    : public MatmulGatherScatterFullLoadKernelBase<BlockMmad_, BlockEpilogue_, BlockScheduler_> {
    using Base = MatmulGatherScatterFullLoadKernelBase<BlockMmad_, BlockEpilogue_, BlockScheduler_>;

public:
    static_assert(!Base::AIV_GATHER, "AIC Gather kernel requires an AIC Gather dispatch policy");
    using Base::operator();
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_GATHER_SCATTER_FULL_LOAD_HPP
