/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_MATMUL_GATHER_SCATTER_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_MATMUL_GATHER_SCATTER_HPP

#include <type_traits>

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/epilogue/tile/copy_ub_to_l1_tla.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Epilogue::Block {

// This component owns only AIV-local data movement and pipeline dependencies.
// Cross-core producer/consumer synchronization stays in the FullLoad kernel.
// GM-to-UB and UB-to-GM transfers reuse the common CopyGm2Ub/CopyUb2Gm components.
template <class BlockMmad_, class DispatchPolicy_>
class BlockEpilogueMatmulGatherScatterBase {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementTransport = typename BlockMmad::ElementC;
    using DispatchPolicy = DispatchPolicy_;

    static constexpr bool DUAL_AIV_OUTPUT = std::is_same_v<ElementTransport, float>;
    static constexpr bool AIV_GATHER = DispatchPolicy::AIV_GATHER;
    // Ascend 950 rejects a 248 KB dynamic UB request for MIX(1,2)+SIMT. The
    // validated launch window is 216 KB for every FullLoad kernel specialization.
    static constexpr uint32_t UB_SIZE = 216 * 1024;
    static constexpr uint32_t AIV_SUBBLOCKS = 2;
    static constexpr uint32_t FP16_ELEMENTS_PER_BLOCK = BYTE_PER_BLK / sizeof(half);
    static constexpr uint32_t ZERO_SCRATCH_MAX_BYTES = 48 * 1024;
    static constexpr uint16_t EVENT_GM_TO_UB_READY = 0;
    static constexpr uint16_t EVENT_GATHER_COPY = 1;
    static constexpr uint16_t EVENT_SCATTER_READY = 3;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t UB_STAGES = DispatchPolicy::OUTPUT_UB_STAGES;
    static constexpr uint32_t INDEX_UB_BYTES = RoundUp<BYTE_PER_BLK>(L1_TILE_M * sizeof(int32_t));
    static constexpr uint32_t MAX_AIV_ROWS = CeilDiv(L1_TILE_M, AIV_SUBBLOCKS);
    static constexpr uint32_t MAX_GATHER_K = DispatchPolicy::MAX_GATHER_K;
    static constexpr uint32_t GATHER_UB_BYTES =
        AIV_GATHER ? RoundUp<BYTE_PER_BLK>(MAX_AIV_ROWS * MAX_GATHER_K * sizeof(half)) : 0;
    static constexpr uint32_t OUTPUT_FLOAT_STAGE_BYTES =
        DUAL_AIV_OUTPUT ? RoundUp<BYTE_PER_BLK>(MAX_AIV_ROWS * L1_TILE_N * sizeof(float)) : 0;
    static constexpr uint32_t OUTPUT_HALF_BYTES = DUAL_AIV_OUTPUT ?
                                                      RoundUp<BYTE_PER_BLK>(MAX_AIV_ROWS * L1_TILE_N * sizeof(half)) :
                                                      RoundUp<BYTE_PER_BLK>(L1_TILE_M * L1_TILE_N * sizeof(half));
    static constexpr uint32_t OUTPUT_DATA_BYTES = OUTPUT_FLOAT_STAGE_BYTES + OUTPUT_HALF_BYTES;
    static constexpr uint32_t OUTPUT_STAGE_BYTES =
        GATHER_UB_BYTES > OUTPUT_DATA_BYTES ? GATHER_UB_BYTES : OUTPUT_DATA_BYTES;
    static constexpr uint32_t INDEX_UB_OFFSET = UB_SIZE - INDEX_UB_BYTES;
    static constexpr uint32_t ZERO_UB_OFFSET = UB_STAGES * OUTPUT_STAGE_BYTES;
    static constexpr uint32_t ZERO_AVAILABLE_BYTES = INDEX_UB_OFFSET - ZERO_UB_OFFSET;
    static constexpr uint32_t ZERO_CHUNK_BYTES =
        ZERO_AVAILABLE_BYTES < ZERO_SCRATCH_MAX_BYTES ? ZERO_AVAILABLE_BYTES : ZERO_SCRATCH_MAX_BYTES;
    static constexpr uint32_t ZERO_CHUNK_ELEMS = ZERO_CHUNK_BYTES / sizeof(half);

    static_assert(std::is_same_v<ElementA, half>, "MatmulGatherScatter currently supports FP16 A only");
    static_assert(AIV_SUBBLOCKS == 2, "the FullLoad kernel requires exactly two AIV subblocks");
    static_assert(UB_STAGES == 1 || UB_STAGES == 2, "output UB buffering supports one or two stages only");
    static_assert(
        (!DUAL_AIV_OUTPUT && std::is_same_v<ElementTransport, half>) ||
            (DUAL_AIV_OUTPUT && std::is_same_v<ElementTransport, float>),
        "single-AIV output must use FP16 transport and dual-AIV output must use FP32 transport");
    static_assert(OUTPUT_STAGE_BYTES * UB_STAGES <= INDEX_UB_OFFSET, "output stages exceed UB capacity");
    static_assert(ZERO_CHUNK_BYTES > 0 && ZERO_CHUNK_BYTES % BYTE_PER_BLK == 0, "invalid zero scratch size");
    static_assert(INDEX_UB_OFFSET + INDEX_UB_BYTES <= UB_SIZE, "indices exceed UB capacity");

    CATLASS_DEVICE explicit BlockEpilogueMatmulGatherScatterBase(Arch::Resource<ArchTag>& resource_)
        : resource(resource_)
    {}

    CATLASS_DEVICE static constexpr uint32_t GetOutputUbOffset(uint16_t outputSlot)
    {
        return outputSlot * OUTPUT_STAGE_BYTES;
    }

    CATLASS_DEVICE AscendC::LocalTensor<int32_t> GetIndicesUb()
    {
        return resource.ubBuf.template GetBufferByByte<int32_t>(INDEX_UB_OFFSET);
    }

    CATLASS_DEVICE void LoadIndices(
        AscendC::LocalTensor<int32_t> const& indicesUb, AscendC::GlobalTensor<int32_t> const& gmIndices,
        uint32_t jOffset, uint32_t rows)
    {
        using IndicesType = Gemm::GemmType<int32_t, layout::VectorLayout>;
        using CopyGmToUb = Tile::CopyGm2Ub<ArchTag, IndicesType>;
        CopyGmToUb copyGmToUb;
        layout::VectorLayout indicesLayout(rows);
        copyGmToUb(indicesUb, gmIndices[jOffset], indicesLayout, indicesLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_GM_TO_UB_READY);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_GM_TO_UB_READY);
    }

    CATLASS_DEVICE void GatherToL1(
        AscendC::LocalTensor<int32_t> const& indicesUb, AscendC::GlobalTensor<half> const& gmA, uint32_t k,
        uint32_t rows, uint32_t subBlockIdx, uint16_t outputSlot)
    {
        uint32_t firstHalfM = (rows + 1U) >> 1;
        uint32_t rowBegin = firstHalfM * subBlockIdx;
        uint32_t ownedRows = subBlockIdx == 0U ? firstHalfM : rows - firstHalfM;
        if (ownedRows == 0) {
            return;
        }

        using MatrixType = Gemm::GemmType<half, layout::RowMajor>;
        using CopyGmToUb = Tile::CopyGm2Ub<ArchTag, MatrixType>;
        CopyGmToUb copyGmToUb;
        auto gatherUb = resource.ubBuf.template GetBufferByByte<half>(GetOutputUbOffset(outputSlot));
        layout::RowMajor rowLayout(1, k);
        for (uint32_t row = 0; row < ownedRows; ++row) {
            uint32_t physicalRow = static_cast<uint32_t>(indicesUb.GetValue(rowBegin + row));
            copyGmToUb(gatherUb[row * k], gmA[static_cast<uint64_t>(physicalRow) * k], rowLayout, rowLayout);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_GATHER_COPY);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_GATHER_COPY);

        uint32_t l1AOffset = BlockMmad::L1B_TILE_SIZE * BlockMmad::L1B_STAGES;
        auto l1A = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset);
        auto l1Layout = tla::MakeLayout<ElementA, typename BlockMmad::LayoutTagL1A>(tla::Int<L1_TILE_M>{}, k);
        auto tensorL1A = tla::MakeTensor(l1A, l1Layout, Arch::PositionL1{});
        auto tensorL1Tile = GetTile(tensorL1A, tla::MakeCoord(rowBegin, 0), tla::MakeShape(ownedRows, k));
        auto ubLayout = tla::MakeLayout<ElementA, layout::RowMajor>(ownedRows, k);
        auto tensorUb = tla::MakeTensor(gatherUb, ubLayout, Arch::PositionUB{});
        using CopyUbToL1 = Tile::CopyUb2L1Tla<ArchTag, decltype(tensorUb), decltype(tensorL1Tile)>;
        CopyUbToL1 copyUbToL1;
        copyUbToL1(tensorL1Tile, tensorUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_GATHER_COPY);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_GATHER_COPY);
    }

    CATLASS_DEVICE void ZeroOutput(AscendC::GlobalTensor<half> const& gmD, uint64_t totalElems)
    {
        using VectorType = Gemm::GemmType<half, layout::VectorLayout>;
        using CopyUbToGm = Tile::CopyUb2Gm<ArchTag, VectorType>;
        CopyUbToGm copyUbToGm;
        auto zeroUb = resource.ubBuf.template GetBufferByByte<half>(ZERO_UB_OFFSET);
        AscendC::Duplicate(zeroUb, static_cast<half>(0), ZERO_CHUNK_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_GM_TO_UB_READY);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_GM_TO_UB_READY);

        uint64_t totalBlocks = CeilDiv(totalElems, static_cast<uint64_t>(FP16_ELEMENTS_PER_BLOCK));
        uint64_t workerId = AscendC::GetBlockIdx();
        uint64_t workerNum = static_cast<uint64_t>(AscendC::GetBlockNum()) * AscendC::GetSubBlockNum();
        uint64_t beginBlock = totalBlocks * workerId / workerNum;
        uint64_t endBlock = totalBlocks * (workerId + 1U) / workerNum;
        uint64_t begin = beginBlock * FP16_ELEMENTS_PER_BLOCK;
        uint64_t end = min(endBlock * FP16_ELEMENTS_PER_BLOCK, totalElems);

        for (uint64_t offset = begin; offset < end;) {
            uint64_t remaining = end - offset;
            uint32_t count = static_cast<uint32_t>(remaining < ZERO_CHUNK_ELEMS ? remaining : ZERO_CHUNK_ELEMS);
            layout::VectorLayout vectorLayout(count);
            copyUbToGm(gmD[offset], zeroUb, vectorLayout, vectorLayout);
            offset += count;
        }
    }

    CATLASS_DEVICE void ScatterOutput(
        AscendC::LocalTensor<int32_t> const& indicesUb, AscendC::GlobalTensor<half> const& gmD,
        GemmCoord const& actualShape, uint32_t nOffset, uint32_t subBlockIdx, uint32_t physicalN, uint16_t outputSlot)
    {
        uint32_t rowBegin = 0;
        uint32_t rows = actualShape.m();
        if constexpr (DUAL_AIV_OUTPUT) {
            uint32_t firstHalfM = (actualShape.m() + 1U) >> 1;
            rowBegin = firstHalfM * subBlockIdx;
            rows = subBlockIdx == 0U ? firstHalfM : actualShape.m() - firstHalfM;
        }
        if (rows == 0) {
            return;
        }

        uint32_t alignN = RoundUp<FP16_ELEMENTS_PER_BLOCK>(actualShape.n());
        AscendC::LocalTensor<half> outputHalf;
        if constexpr (DUAL_AIV_OUTPUT) {
            auto outputFloat = resource.ubBuf.template GetBufferByByte<float>(GetOutputUbOffset(outputSlot));
            outputHalf = resource.ubBuf.template GetBufferByByte<half>(GetCastUbOffset(outputSlot));
            AscendC::Cast(outputHalf, outputFloat, AscendC::RoundMode::CAST_NONE, rows * alignN);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            outputHalf = resource.ubBuf.template GetBufferByByte<half>(GetOutputUbOffset(outputSlot));
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_SCATTER_READY);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_SCATTER_READY);
        using MatrixType = Gemm::GemmType<half, layout::RowMajor>;
        using CopyUbToGm = Tile::CopyUb2Gm<ArchTag, MatrixType>;
        CopyUbToGm copyUbToGm;
        layout::RowMajor dstLayout(1, actualShape.n());
        layout::RowMajor srcLayout(1, actualShape.n(), alignN);
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t physicalRow = static_cast<uint32_t>(indicesUb.GetValue(rowBegin + row));
            uint64_t dstOffset = static_cast<uint64_t>(physicalRow) * physicalN + nOffset;
            copyUbToGm(gmD[dstOffset], outputHalf[row * alignN], dstLayout, srcLayout);
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    Arch::Resource<ArchTag>& resource;

    CATLASS_DEVICE static constexpr uint32_t GetCastUbOffset(uint16_t outputSlot)
    {
        return GetOutputUbOffset(outputSlot) + OUTPUT_FLOAT_STAGE_BYTES;
    }
};

template <class BlockMmad_, class DispatchPolicy_>
class BlockEpilogueMatmulGatherScatter : public BlockEpilogueMatmulGatherScatterBase<BlockMmad_, DispatchPolicy_> {
    using Base = BlockEpilogueMatmulGatherScatterBase<BlockMmad_, DispatchPolicy_>;

public:
    CATLASS_DEVICE explicit BlockEpilogueMatmulGatherScatter(Arch::Resource<typename Base::ArchTag>& resource)
        : Base(resource)
    {}
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_MATMUL_GATHER_SCATTER_HPP
