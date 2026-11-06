/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FINALIZE_ROUTING_NO_DETER_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FINALIZE_ROUTING_NO_DETER_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/matrix.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

namespace Catlass::Epilogue::Block {
using namespace Catlass::Epilogue::Tile;
// 非确定性版本的 BlockEpilogue，AIV 侧按 M 维切分（而非原版的 N 维切分）。
// VecTileShape 的 ROW 为子核处理的 M 行数（通常为 L1_TILE_M/2），
// COLUMN 为完整 N 宽度（L1_TILE_N）。
template <
    class DispatchPolicy_, class ArchTag_, class VecTileShape_, class ElementC_, class ElementRowIndex_,
    class ElementSharedInput_>
class BlockEpilogueFinalizeRoutingNoDeter {
public:
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = ArchTag_;
    using VecTileShape = VecTileShape_;
    using ElementC = ElementC_;
    using ElementRowIndex = ElementRowIndex_;
    using ElementSharedInput = ElementSharedInput_;
    using SafeSharedInput = std::conditional_t<std::is_void_v<ElementSharedInput>, uint8_t, ElementSharedInput>;
    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;
    static constexpr uint32_t EPILOGUE_TILE_M = VecTileShape::ROW;
    static constexpr uint32_t EPILOGUE_TILE_N = VecTileShape::COLUMN;

    static constexpr uint32_t ALIGNED_VALUE = 8;

    static constexpr uint32_t ONE_REPEAT_SIZE = 256;
    static constexpr uint32_t ONE_DATA_BLOCK_SIZE = 32;
    static constexpr uint32_t ONE_BLOCK_ELEMENT = ONE_DATA_BLOCK_SIZE / sizeof(ElementC); // 8
    static constexpr uint32_t ONE_REPEAT_ELEMENT = ONE_REPEAT_SIZE / sizeof(ElementC);    // 64
    static constexpr uint32_t MAX_OUTPUT_M_UB = 32;

    static constexpr size_t UB_BUF_GMM_BYTES = EPILOGUE_TILE_M * EPILOGUE_TILE_N * sizeof(ElementC);
    static constexpr size_t UB_BUF_ROW_INDEX_BYTES = EPILOGUE_TILE_M * sizeof(ElementRowIndex);
    static constexpr size_t UB_BUF_LOGIT_BYTES = EPILOGUE_TILE_M * sizeof(ElementC);
    static constexpr size_t UB_BUF_OUT_CHUNK_BYTES = MAX_OUTPUT_M_UB * EPILOGUE_TILE_N * sizeof(ElementC);
    static constexpr size_t UB_PER_STAGE =
        UB_BUF_GMM_BYTES + UB_BUF_LOGIT_BYTES + UB_BUF_ROW_INDEX_BYTES + 2 * UB_BUF_OUT_CHUNK_BYTES;

    static constexpr size_t MAX_CLEAR_GM_COUNT = 50 * 1024;

    static constexpr size_t MAX_SOLVE_SHARED_INPUT_COUNT = 20 * 1024;
    static constexpr size_t UB_BUF_SHARED_INPUT = MAX_SOLVE_SHARED_INPUT_COUNT * sizeof(SafeSharedInput);
    static constexpr size_t UB_BUF_SHARED_INPUT_CAST = MAX_SOLVE_SHARED_INPUT_COUNT * sizeof(ElementC);
    static constexpr size_t UB_BUF_SHARED_OUTPUT = MAX_SOLVE_SHARED_INPUT_COUNT * sizeof(ElementC);
    static constexpr size_t UB_PER_STAGE_SHARED_INPUT =
        UB_BUF_SHARED_INPUT + UB_BUF_SHARED_INPUT_CAST + UB_BUF_SHARED_OUTPUT;

    static_assert(
        UB_STAGES * UB_PER_STAGE <= ArchTag::UB_SIZE, "UB budget exceeded for BlockEpilogueFinalizeRoutingNoDeter");
    static_assert(
        UB_STAGES * UB_PER_STAGE_SHARED_INPUT <= ArchTag::UB_SIZE,
        "UB budget exceeded for BlockEpilogueFinalizeRoutingNoDeter");
    // ubBufClearGm 在 clear 阶段独占 UB，计算完成后才复用给 shared-input 阶段，
    // 故单独校验其不超过 UB 总容量即可，避免调参后隐性溢出。
    static_assert(MAX_CLEAR_GM_COUNT * sizeof(ElementC) <= ArchTag::UB_SIZE, "ubBufClearGm exceeds UB size");

    GemmCoord problemShape;
    using LayoutTagMatric = layout::RowMajor;
    using GmMatricType = Gemm::GemmType<ElementC, LayoutTagMatric>;
    using CopyGm2UbMatrix = CopyGm2Ub<ArchTag, GmMatricType>;
    using CopyUb2GmMatrix = CopyUb2Gm<ArchTag, GmMatricType>;
    CopyGm2UbMatrix copyGm2UbMatrix;
    CopyUb2GmMatrix copyUb2GmMatrix;

    using LayoutTagVec = layout::VectorLayout;
    using GmVecType = Gemm::GemmType<ElementC, LayoutTagVec>;
    using CopyGm2UbVec = CopyGm2Ub<ArchTag, GmVecType>;
    CopyGm2UbVec copyGm2UbVec;

    using GmVecTypeRowIndex = Gemm::GemmType<ElementRowIndex, LayoutTagVec>;
    using CopyRowIndexGm2UbVec = CopyGm2Ub<ArchTag, GmVecTypeRowIndex>;
    CopyRowIndexGm2UbVec copyRowIndexGm2UbVec;

    using GmVecTypeSharedInput = Gemm::GemmType<SafeSharedInput, LayoutTagVec>;
    using CopySharedInputGm2UbVec = CopyGm2Ub<ArchTag, GmVecTypeSharedInput>;
    CopySharedInputGm2UbVec copySharedGm2UbVec;

    CATLASS_DEVICE
    BlockEpilogueFinalizeRoutingNoDeter(Arch::Resource<ArchTag>& resource)
    {
        AllocateUbBuffers(resource);
    }

    CATLASS_DEVICE
    ~BlockEpilogueFinalizeRoutingNoDeter()
    {}
    CATLASS_DEVICE
    void Update(GemmCoord const& problemShape_)
    {
        problemShape = problemShape_;
    }

    CATLASS_DEVICE
    void ClearOutTile(AscendC::GlobalTensor<ElementC> const& gmOutTile, MatrixCoord const& outSplitCoord)
    {
        int64_t clearCount = outSplitCoord.column() * problemShape.n();
        int64_t singleCount = clearCount > MAX_CLEAR_GM_COUNT ? MAX_CLEAR_GM_COUNT : clearCount;

        AscendC::Duplicate<ElementC>(ubBufClearGm, static_cast<ElementC>(0.0f), singleCount);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        for (size_t idx = 0; idx < clearCount; idx += singleCount) {
            uint32_t curNum = ((idx + singleCount) < clearCount) ? singleCount : (clearCount - idx);
            auto layoutVecSrcOut = LayoutTagMatric::MakeLayout<ElementC>(1, curNum);
            auto layoutVecDstOut = LayoutTagMatric::MakeLayout<ElementC>(1, curNum);
            copyUb2GmMatrix(gmOutTile[idx], ubBufClearGm, layoutVecDstOut, layoutVecSrcOut);
        }
    }

    CATLASS_DEVICE
    void AssignSharedInputTile(
        AscendC::GlobalTensor<SafeSharedInput> const& sharedInputTile,
        AscendC::GlobalTensor<ElementC> const& sharedOutputTile, MatrixCoord const& outSplitCoord,
        float sharedInputWeight)
    {
        if constexpr (std::is_void_v<ElementSharedInput>) {
            return;
        }
        int64_t count = outSplitCoord.column() * problemShape.n();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        for (size_t idx = 0; idx < count; idx += MAX_SOLVE_SHARED_INPUT_COUNT) {
            uint32_t curNum =
                ((idx + MAX_SOLVE_SHARED_INPUT_COUNT) < count) ? MAX_SOLVE_SHARED_INPUT_COUNT : (count - idx);
            auto layoutVecSrc = LayoutTagVec(curNum);
            auto layoutVecDst = LayoutTagVec(curNum);
            copySharedGm2UbVec(ubBufSharedInput, sharedInputTile[idx], layoutVecDst, layoutVecSrc);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::Cast(ubBufSharedInputCast, ubBufSharedInput, AscendC::RoundMode::CAST_NONE, curNum);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

            AscendC::Muls(ubBufSharedOut, ubBufSharedInputCast, sharedInputWeight, curNum);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto layoutVecSrcOut = LayoutTagMatric::MakeLayout<ElementC>(1, curNum);
            auto layoutVecDstOut = LayoutTagMatric::MakeLayout<ElementC>(1, curNum);
            copyUb2GmMatrix(sharedOutputTile[idx], ubBufSharedOut, layoutVecDstOut, layoutVecSrcOut);
        }
    }

    CATLASS_DEVICE
    AscendC::LocalTensor<ElementC> GetL0c2UbTensor()
    {
        return l0cOutUb_;
    }

    // AIC 直写 UB 版本：数据已通过 Fixpipe 写入 ubBufGmm，跳过 GM→UB 拷贝。
    template <uint16_t SYNC_MODE>
    CATLASS_DEVICE void LogitScatterAddTileFromUb(
        AscendC::GlobalTensor<ElementC> const& gmLogitTile,
        AscendC::GlobalTensor<ElementRowIndex> const& gmRowIndexTile, AscendC::GlobalTensor<ElementC> const& gmOutTile,
        GemmCoord const& tileShape, uint16_t waitFlagId, uint16_t setFlagId)
    {
        if (tileShape.m() == 0 || tileShape.n() == 0) {
            AscendC::CrossCoreWaitFlag<SYNC_MODE>(waitFlagId);
            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(setFlagId);
            return;
        }
        // === 阶段 A：等待 AIC Fixpipe 完成，数据已在 ubBufGmm 中 ===
        AscendC::CrossCoreWaitFlag<SYNC_MODE>(waitFlagId);

        auto layoutVecSrc = LayoutTagVec(tileShape.m());
        auto layoutVecDst = LayoutTagVec(tileShape.m());
        copyGm2UbVec(ubBufLogit, gmLogitTile, layoutVecDst, layoutVecSrc);
        copyRowIndexGm2UbVec(ubBufRowIndex, gmRowIndexTile, layoutVecDst, layoutVecSrc);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

        // === 阶段 B：分块流水循环（VFDoLogitMuls → outUb ping/pong → MTE3 写出） ===
        uint64_t alignN = CeilDiv(tileShape.n(), (uint64_t)ONE_BLOCK_ELEMENT) * ONE_BLOCK_ELEMENT;
        uint64_t vlForFloatNumber = 256 / sizeof(ElementC);
        auto repeatTimesRe = CeilDiv(tileShape.n(), (uint32_t)vlForFloatNumber);
        uint32_t loopNumY = CeilDiv(tileShape.m(), MAX_OUTPUT_M_UB);
        uint32_t remainM = (tileShape.m() % MAX_OUTPUT_M_UB != 0) ? tileShape.m() % MAX_OUTPUT_M_UB : MAX_OUTPUT_M_UB;
        auto layoutMatricOutSrc = LayoutTagMatric(1, tileShape.n());
        auto layoutMatricOutDst = LayoutTagMatric(1, tileShape.n());

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
        yCrossPingPongID_ = 0;

        AscendC::SetAtomicAdd<ElementC>();
        for (uint32_t i = 0; i < loopNumY; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(yCrossPingPongID_);
            uint32_t chunkM = (i == loopNumY - 1) ? remainM : MAX_OUTPUT_M_UB;
            auto outUb = yCrossPingPongID_ == 0 ? outUbPing_ : outUbPong_;
            uint32_t chunkRowOffset = i * MAX_OUTPUT_M_UB;

            VFDoLogitMuls(
                chunkRowOffset * alignN, chunkRowOffset, chunkM, repeatTimesRe, (__ubuf__ ElementC*)outUb.GetPhyAddr(),
                ubBufGmm, ubBufLogit, alignN, vlForFloatNumber, tileShape.n());

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(yCrossPingPongID_);
            // 最后一次迭代：ubBufGmm 已读完，通知 AIC 可写下一 tile（与 MTE3 写出并行）
            if (i == loopNumY - 1) {
                AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(setFlagId);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(yCrossPingPongID_);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            for (uint32_t j = 0; j < chunkM; j++) {
                int64_t resRowIdx = ubBufRowIndex.GetValue(chunkRowOffset + j);
                int64_t ubOffset = j * alignN;
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
                copyUb2GmMatrix(
                    gmOutTile[resRowIdx * problemShape.n()], outUb[ubOffset], layoutMatricOutDst, layoutMatricOutSrc);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(yCrossPingPongID_);
            yCrossPingPongID_ = (yCrossPingPongID_ + 1) & 1;
        }
        AscendC::DisableDmaAtomic();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    }

private:
    AscendC::LocalTensor<ElementC> ubBufClearGm;
    AscendC::LocalTensor<SafeSharedInput> ubBufSharedInput;
    AscendC::LocalTensor<ElementC> ubBufSharedInputCast;
    AscendC::LocalTensor<ElementC> ubBufSharedOut;

    AscendC::LocalTensor<ElementC> ubBufGmm;
    AscendC::LocalTensor<ElementC> ubBufLogit;
    AscendC::LocalTensor<ElementRowIndex> ubBufRowIndex;
    AscendC::LocalTensor<ElementC> outUbPing_;
    AscendC::LocalTensor<ElementC> outUbPong_;
    AscendC::LocalTensor<ElementC> l0cOutUb_;
    uint16_t yCrossPingPongID_ = 0;

    // 使用 __VEC_SCOPE__ + Reg 实现 Logit 乘法（参考 block_epilogue_finalize_routing.h）
    // 通过 DIST_BRC_B32 内联广播 logit 值，避免单独的 Brcb 步骤
    CATLASS_DEVICE
    void VFDoLogitMuls(
        uint32_t offsetRe, uint32_t offsetLogit, uint16_t repeatTimesLogit, uint16_t repeatTimesRe,
        __ubuf__ ElementC* outUbAddr, AscendC::LocalTensor<ElementC> gmmUb, AscendC::LocalTensor<ElementC> logitUb,
        uint64_t alignN, uint64_t vlForFloatNumber, uint32_t actualN)
    {
        __ubuf__ ElementC* gmmUbAddr = (__ubuf__ ElementC*)gmmUb.GetPhyAddr();
        __ubuf__ ElementC* logitUbAddr = (__ubuf__ ElementC*)logitUb.GetPhyAddr();
        gmmUbAddr += offsetRe;
        logitUbAddr += offsetLogit;
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<ElementC> vregLogit;
            AscendC::Reg::RegTensor<ElementC> vregRe, vDstReg;
            AscendC::Reg::MaskReg mask;
            for (uint16_t i = 0; i < repeatTimesLogit; ++i) {
                AscendC::Reg::LoadAlign<ElementC, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregLogit, logitUbAddr + i);
                uint32_t elementNum = actualN;
                for (uint16_t j = 0; j < repeatTimesRe; ++j) {
                    mask = AscendC::Reg::UpdateMask<ElementC>(elementNum);
                    AscendC::Reg::LoadAlign(vregRe, gmmUbAddr + i * alignN + j * vlForFloatNumber);
                    AscendC::Reg::Mul(vDstReg, vregLogit, vregRe, mask);
                    AscendC::Reg::StoreAlign(outUbAddr + i * alignN + j * vlForFloatNumber, vDstReg, mask);
                }
            }
        }
    }

    CATLASS_DEVICE
    void AllocateUbBuffers(Arch::Resource<ArchTag>& resource)
    {
        size_t offset = 0;

        ubBufClearGm = resource.ubBuf.template GetBufferByByte<ElementC>(offset);
        ubBufSharedInput = resource.ubBuf.template GetBufferByByte<SafeSharedInput>(offset);
        offset += UB_BUF_SHARED_INPUT;
        ubBufSharedInputCast = resource.ubBuf.template GetBufferByByte<ElementC>(offset);
        offset += UB_BUF_SHARED_INPUT_CAST;
        ubBufSharedOut = resource.ubBuf.template GetBufferByByte<ElementC>(offset);

        offset = 0;
        ubBufGmm = resource.ubBuf.template GetBufferByByte<ElementC>(offset);
        l0cOutUb_ = ubBufGmm;
        offset += UB_BUF_GMM_BYTES;
        ubBufRowIndex = resource.ubBuf.template GetBufferByByte<ElementRowIndex>(offset);
        offset += UB_BUF_ROW_INDEX_BYTES;
        ubBufLogit = resource.ubBuf.template GetBufferByByte<ElementC>(offset);
        offset += UB_BUF_LOGIT_BYTES;
        outUbPing_ = resource.ubBuf.template GetBufferByByte<ElementC>(offset);
        offset += UB_BUF_OUT_CHUNK_BYTES;
        outUbPong_ = resource.ubBuf.template GetBufferByByte<ElementC>(offset);
        offset += UB_BUF_OUT_CHUNK_BYTES;
    }
};

} // namespace Catlass::Epilogue::Block

#endif // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FINALIZE_ROUTING_NO_DETER_HPP
