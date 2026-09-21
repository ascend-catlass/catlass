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

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_FP8_REGBASE_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_FP8_REGBASE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"

namespace Catlass::Epilogue::Block {

template <
    uint32_t UB_STAGES_, class CType_, class ScaleType_, class PerTokenScaleType_, class DType_,
    class TileRowBroadcastMul_, class TileBroadcastOneBlk_, class TileOneBlkColumnBroadcastMul_, class TileCopy_,
    class EpilogueTileSwizzle_>
class BlockEpilogue<
    EpilogueAscend950Fp8PerTokenPerChannelDequant<UB_STAGES_>, CType_, ScaleType_, PerTokenScaleType_, DType_,
    TileRowBroadcastMul_, TileBroadcastOneBlk_, TileOneBlkColumnBroadcastMul_, TileCopy_, EpilogueTileSwizzle_> {
public:
    using DispatchPolicy = EpilogueAscend950Fp8PerTokenPerChannelDequant<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element; // 现在float
    using LayoutC = typename CType_::Layout;
    using ElementScale = typename ScaleType_::Element; // 现在fp8_e5m2
    using LayoutScale = typename ScaleType_::Layout;
    using ElementPerTokenScale = typename PerTokenScaleType_::Element; // 现在fp8_e5m2
    using LayoutPerTokenScale = typename PerTokenScaleType_::Layout;
    using ElementD = typename DType_::Element; // 现在根据输入确定是half or float
    using LayoutD = typename DType_::Layout;

    static_assert(
        std::is_same_v<LayoutC, layout::RowMajor> && std::is_same_v<LayoutScale, layout::VectorLayout> &&
            std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> && std::is_same_v<LayoutD, layout::RowMajor>,
        "The layout template parameters of BlockEpilogue are wrong");

    // Tile compute ops
    using TileRowBroadcastMul = TileRowBroadcastMul_;
    using TileBroadcastOneBlk = TileBroadcastOneBlk_;
    using TileOneBlkColumnBroadcastMul = TileOneBlkColumnBroadcastMul_;

    // Tile copy
    using CopyGmToUbScale = typename TileCopy_::CopyGmToUbX;
    using CopyGmToUbPerTokenScale = typename TileCopy_::CopyGmToUbY;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;

    using TileShape = typename TileRowBroadcastMul::TileShape;

    static_assert(
        TileShape::ROW == TileBroadcastOneBlk::COMPUTE_LENGTH &&
            std::is_same_v<TileShape, typename TileOneBlkColumnBroadcastMul::TileShape>,
        "TileShape must be consistent for all tile compute ops");

    static_assert(
        (UB_STAGES * (TileShape::COUNT * sizeof(ElementC) + TileShape::COUNT * sizeof(half) +
                      TileShape::COLUMN * sizeof(ElementScale) + TileShape::ROW * sizeof(ElementPerTokenScale)) +
         UB_STAGES * (TileShape::COLUMN + TileShape::ROW) * sizeof(float) +
         UB_STAGES * TileShape::ROW * BYTE_PER_BLK) <= ArchTag::UB_SIZE,
        "TileShape is too large to fit in UB");

    struct Params {
        __gm__ ElementScale* ptrScale{nullptr};
        LayoutScale layoutScale{};
        __gm__ ElementPerTokenScale* ptrPerTokenScale{nullptr};
        LayoutPerTokenScale layoutPerTokenScale{};
        __gm__ ElementD* ptrD{nullptr};
        LayoutD layoutD{};

        CATLASS_DEVICE
        Params() {};

        CATLASS_DEVICE
        Params(
            __gm__ ElementScale* ptrScale_, LayoutScale const& layoutScale_,
            __gm__ ElementPerTokenScale* ptrPerTokenScale_, LayoutPerTokenScale const& layoutPerTokenScale_,
            __gm__ ElementD* ptrD_, LayoutD const& layoutD_)
            : ptrScale(ptrScale_),
              layoutScale(layoutScale_),
              ptrPerTokenScale(ptrPerTokenScale_),
              layoutPerTokenScale(layoutPerTokenScale_),
              ptrD(ptrD_),
              layoutD(layoutD_)
        {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const& resource, Params const& params = Params{}) : params(params)
    {
        size_t ubOffset = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementC);
        }

        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementScale>(ubOffset);
            ubOffset += TileShape::COLUMN * sizeof(ElementScale);
            ubPerTokenScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementPerTokenScale>(ubOffset);
            ubOffset += TileShape::ROW * sizeof(ElementPerTokenScale);
            if constexpr (!std::is_same_v<ElementD, float>) {
                ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
                ubOffset += TileShape::COUNT * sizeof(ElementD);
            }
            ubPerTokenScaleFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += TileShape::ROW * sizeof(float);
            ubPerTokenMulList[i] = ubCList[i];
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {}

    CATLASS_DEVICE
    void UpdateParams(Params const& params_)
    {
        params = params_;
    }

    CATLASS_DEVICE
    void operator()(
        GemmCoord const& blockShapeMNK, GemmCoord const& blockCoordMNK, GemmCoord const& actualBlockShapeMNK,
        AscendC::GlobalTensor<ElementC> const& gmBlockC, LayoutC const& layoutBlockC, uint32_t const& stageId,
        Arch::CrossCoreFlag preCross, Arch::CrossCoreFlag afterCross, Callback&& callback = Callback{})
    {
        if (actualBlockShapeMNK.k() == 0) {
            return;
        }
        callback();

        ubListId = stageId;

        // Calculate the offset of the current block
        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoord * blockShape;

        AscendC::GlobalTensor<ElementScale> gmScale;
        gmScale.SetGlobalBuffer(params.ptrScale);
        AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale;
        gmPerTokenScale.SetGlobalBuffer(params.ptrPerTokenScale);
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = MakeCoord((actualBlockShape.row() + 1) / 2, TileShape::COLUMN);
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        // An idle AIV must still consume the ready flag and acknowledge the stage.
        // Otherwise the AIC waits forever when an M tail has only one row.
        if (subblockIdx >= tileLoops) {
            Arch::CrossCoreWaitFlag(preCross);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(afterCross);
        }
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffsetInBlock = tileCoord * tileShape;
            auto tileOffset = blockOffset + tileOffsetInBlock;

            auto gmTileC = gmBlockC[layoutBlockC.GetOffset(tileOffsetInBlock)];
            auto layoutGmTileC = layoutBlockC.GetTileLayout(actualTileShape);

            auto& ubC = ubCList[ubListId];
            LayoutC layoutUbC{actualTileShape, ubTileStride};

            auto eventId = ubListId ? EVENT_ID0 : EVENT_ID1;

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);

            auto scaleTileOffset = tileOffset.template GetCoordByAxis<1>();
            auto scaleTileShape = actualTileShape.template GetCoordByAxis<1>();

            auto gmTileScale = gmScale[params.layoutScale.GetOffset(scaleTileOffset)];
            auto layoutGmTileScale = params.layoutScale.GetTileLayout(scaleTileShape);

            auto& ubScale = ubScaleList[ubListId];
            auto layoutUbScale = LayoutScale::template MakeLayoutInUb<ElementScale>(scaleTileShape);

            copyGmToUbScale(ubScale, gmTileScale, layoutUbScale, layoutGmTileScale);

            auto perTokenScaleTileOffset = tileOffset.template GetCoordByAxis<0>();
            auto perTokenScaleTileShape = actualTileShape.template GetCoordByAxis<0>();

            auto gmTilePerTokenScale = gmPerTokenScale[params.layoutPerTokenScale.GetOffset(perTokenScaleTileOffset)];
            auto layoutGmTilePerTokenScale = params.layoutPerTokenScale.GetTileLayout(perTokenScaleTileShape);

            auto& ubPerTokenScale = ubPerTokenScaleList[ubListId];
            auto layoutUbPerTokenScale =
                LayoutScale::template MakeLayoutInUb<ElementPerTokenScale>(perTokenScaleTileShape);

            auto& ubD = ubDList[ubListId];
            LayoutD layoutUbD{actualTileShape, ubTileStride};

            copyGmToUbPerTokenScale(
                ubPerTokenScale, gmTilePerTokenScale, layoutUbPerTokenScale, layoutGmTilePerTokenScale);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);

            Arch::CrossCoreWaitFlag(preCross);

            if constexpr (TileShape::COLUMN == 256 && TileShape::ROW == 128) {
                if constexpr (std::is_same<ElementPerTokenScale, float8_e8m0_t>::value) {
                    // High-level API doesn't support float8_e8m0_t -> float, use MicroAPI instead
                    __ubuf__ float8_e8m0_t* srcAddr = (__ubuf__ float8_e8m0_t*)ubPerTokenScale.GetPhyAddr();
                    __ubuf__ float* dstAddr = (__ubuf__ float*)ubPerTokenScaleFp32List[ubListId].GetPhyAddr();
                    CastFp8E8m0ToFp32(dstAddr, srcAddr, TileShape::ROW);

                } else if constexpr (!std::is_same<ElementPerTokenScale, float>::value) {
                    AscendC::Cast(
                        ubPerTokenScaleFp32List[ubListId], ubPerTokenScale, AscendC::RoundMode::CAST_NONE,
                        TileShape::ROW);
                }
                AscendC::PipeBarrier<PIPE_V>();
                MulCompute(
                    (__ubuf__ float*)ubPerTokenMulList[ubListId].GetPhyAddr(), (__ubuf__ float*)ubC.GetPhyAddr(),
                    (__ubuf__ ElementScale*)ubScale.GetPhyAddr(),
                    (__ubuf__ float*)ubPerTokenScaleFp32List[ubListId].GetPhyAddr());
                if constexpr (!std::is_same_v<ElementD, float>) {
                    AscendC::Cast(
                        ubD, ubPerTokenMulList[ubListId], AscendC::RoundMode::CAST_RINT,
                        TileShape::ROW * TileShape::COLUMN);
                }
            } else {
                if constexpr (std::is_same<ElementPerTokenScale, float8_e8m0_t>::value) {
                    __ubuf__ float8_e8m0_t* srcAddr = (__ubuf__ float8_e8m0_t*)ubPerTokenScale.GetPhyAddr();
                    __ubuf__ float* dstAddr = (__ubuf__ float*)ubPerTokenScaleFp32List[ubListId].GetPhyAddr();
                    CastFp8E8m0ToFp32(dstAddr, srcAddr, TileShape::ROW);
                } else if constexpr (!std::is_same<ElementPerTokenScale, float>::value) {
                    AscendC::Cast(
                        ubPerTokenScaleFp32List[ubListId], ubPerTokenScale, AscendC::RoundMode::CAST_NONE,
                        TileShape::ROW);
                }
                AscendC::PipeBarrier<PIPE_V>();
                MulComputeGeneric(
                    (__ubuf__ float*)ubPerTokenMulList[ubListId].GetPhyAddr(), (__ubuf__ float*)ubC.GetPhyAddr(),
                    (__ubuf__ ElementScale*)ubScale.GetPhyAddr(),
                    (__ubuf__ float*)ubPerTokenScaleFp32List[ubListId].GetPhyAddr());
                if constexpr (!std::is_same_v<ElementD, float>) {
                    AscendC::Cast(
                        ubD, ubPerTokenMulList[ubListId], AscendC::RoundMode::CAST_RINT,
                        TileShape::ROW * TileShape::COLUMN);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);

            auto gmTileD = gmD[params.layoutD.GetOffset(tileOffset)];
            auto layoutGmTileD = params.layoutD.GetTileLayout(actualTileShape);

            if constexpr (!std::is_same_v<ElementD, float>) {
                copyUbToGmD(gmTileD, ubD, layoutGmTileD, layoutUbD);
            } else {
                copyUbToGmD(gmTileD, ubPerTokenMulList[ubListId], layoutGmTileD, layoutUbD);
            }
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(afterCross);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementScale> ubScaleList[UB_STAGES];
    AscendC::LocalTensor<ElementPerTokenScale> ubPerTokenScaleList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];
    AscendC::LocalTensor<float> ubPerTokenScaleFp32List[UB_STAGES];
    AscendC::LocalTensor<float> ubPerTokenMulList[UB_STAGES];

    uint32_t ubListId{0};

    CopyGmToUbScale copyGmToUbScale;
    CopyGmToUbPerTokenScale copyGmToUbPerTokenScale;
    CopyUbToGmD copyUbToGmD;

    /// Helper function to cast scale to fp32
    template <typename T>
    CATLASS_DEVICE void CastScaleToFp32(AscendC::LocalTensor<float>& dst, AscendC::LocalTensor<T>& src, uint32_t count)
    {
        if constexpr (std::is_same_v<T, float8_e8m0_t>) {
            CastFp8E8m0ToFp32(dst, src, count);
        } else if constexpr (std::is_same_v<T, float8_e4m3_t> || std::is_same_v<T, float8_e5m2_t>) {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
        } else if constexpr (std::is_same_v<T, float>) {
            // Already fp32, copy
            AscendC::DataCopy(dst, src, count);
        } else {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
        }
    }

    /// Cast float8_e8m0_t to fp32 using MicroAPI
    CATLASS_DEVICE_SIMD_VF void CastFp8E8m0ToFp32(
        __ubuf__ float* dstPtr, __ubuf__ ElementPerTokenScale* srcPtr, uint32_t count)
    {
        namespace MicroAPI = AscendC::MicroAPI;

        static constexpr MicroAPI::CastTrait castTraitFp8ToBf16 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};

        static constexpr MicroAPI::CastTrait castTraitBf16ToFp32 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_NONE};

        MicroAPI::RegTensor<ElementPerTokenScale> vRegFp8;
        MicroAPI::RegTensor<bfloat16_t> vRegBf16;
        MicroAPI::RegTensor<float> vRegFp32;
        MicroAPI::MaskReg maskAll;

        constexpr uint32_t ELE_NUM_PER_REPEAT = 64;
        uint16_t repeatTimes = static_cast<uint16_t>((count + ELE_NUM_PER_REPEAT - 1) / ELE_NUM_PER_REPEAT);

        for (uint16_t i = 0; i < repeatTimes; ++i) {
            maskAll = MicroAPI::UpdateMask<float>(count);
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                vRegFp8, srcPtr + i * ELE_NUM_PER_REPEAT);
            MicroAPI::Cast<bfloat16_t, ElementPerTokenScale, castTraitFp8ToBf16>(vRegBf16, vRegFp8, maskAll);
            MicroAPI::Cast<float, bfloat16_t, castTraitBf16ToFp32>(vRegFp32, vRegBf16, maskAll);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                dstPtr + i * ELE_NUM_PER_REPEAT * 4, vRegFp32, maskAll);
        }
    }

    /// Cast float8_e8m0_t to fp32 using MicroAPI
    CATLASS_DEVICE_SIMD_VF void MulCompute(
        __ubuf__ float* dstPtr, __ubuf__ float* srcPtr, __ubuf__ ElementPerTokenScale* src1Ptr, __ubuf__ float* src2Ptr)
    {
        namespace MicroAPI = AscendC::MicroAPI;

        static constexpr MicroAPI::CastTrait castTraitFp8ToBf16 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};

        static constexpr MicroAPI::CastTrait castTraitBf16ToFp32 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_NONE};

        static constexpr MicroAPI::CastTrait castTraitFp8ToFp32 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};

        static constexpr MicroAPI::CastTrait castTraitFp16ToFp32 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};

        MicroAPI::RegTensor<ElementPerTokenScale> vRegFp8_1;
        MicroAPI::RegTensor<ElementPerTokenScale> vRegFp8_2;
        MicroAPI::RegTensor<ElementPerTokenScale> vRegFp8_3;
        MicroAPI::RegTensor<ElementPerTokenScale> vRegFp8_4;
        MicroAPI::RegTensor<bfloat16_t> vRegBf16_1;
        MicroAPI::RegTensor<bfloat16_t> vRegBf16_2;
        MicroAPI::RegTensor<bfloat16_t> vRegBf16_3;
        MicroAPI::RegTensor<bfloat16_t> vRegBf16_4;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        MicroAPI::RegTensor<float> vRegFp32_1;
        MicroAPI::RegTensor<float> vRegFp32_2;
        MicroAPI::RegTensor<float> vRegFp32_3;
        MicroAPI::RegTensor<float> vRegFp32_4;
        MicroAPI::RegTensor<float> vRegMuls1_1;
        MicroAPI::RegTensor<float> vRegMuls1_2;
        MicroAPI::RegTensor<float> vRegMuls1_3;
        MicroAPI::RegTensor<float> vRegMuls1_4;
        MicroAPI::RegTensor<float> vRegMuls2;

        uint32_t row = TileShape::ROW;
        uint32_t col = TileShape::COLUMN;

        constexpr uint32_t ELE_NUM_PER_REPEAT = 64;

        if constexpr (std::is_same_v<ElementPerTokenScale, float8_e8m0_t>) {
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vRegFp8_1, src1Ptr);
            MicroAPI::Cast<bfloat16_t, ElementPerTokenScale, castTraitFp8ToBf16>(vRegBf16_1, vRegFp8_1, maskAll);
            MicroAPI::Cast<float, bfloat16_t, castTraitBf16ToFp32>(vRegMuls1_1, vRegBf16_1, maskAll);
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                vRegFp8_2, src1Ptr + ELE_NUM_PER_REPEAT);
            MicroAPI::Cast<bfloat16_t, ElementPerTokenScale, castTraitFp8ToBf16>(vRegBf16_2, vRegFp8_2, maskAll);
            MicroAPI::Cast<float, bfloat16_t, castTraitBf16ToFp32>(vRegMuls1_2, vRegBf16_2, maskAll);
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                vRegFp8_3, src1Ptr + ELE_NUM_PER_REPEAT * 2);
            MicroAPI::Cast<bfloat16_t, ElementPerTokenScale, castTraitFp8ToBf16>(vRegBf16_3, vRegFp8_3, maskAll);
            MicroAPI::Cast<float, bfloat16_t, castTraitBf16ToFp32>(vRegMuls1_3, vRegBf16_3, maskAll);
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                vRegFp8_4, src1Ptr + ELE_NUM_PER_REPEAT * 3);
            MicroAPI::Cast<bfloat16_t, ElementPerTokenScale, castTraitFp8ToBf16>(vRegBf16_4, vRegFp8_4, maskAll);
            MicroAPI::Cast<float, bfloat16_t, castTraitBf16ToFp32>(vRegMuls1_4, vRegBf16_4, maskAll);
        } else if constexpr (
            std::is_same_v<ElementPerTokenScale, float8_e4m3_t> ||
            std::is_same_v<ElementPerTokenScale, float8_e5m2_t>) {
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vRegFp8_1, src1Ptr);
            MicroAPI::Cast<float, ElementPerTokenScale, castTraitFp8ToFp32>(vRegMuls1_1, vRegFp8_1, maskAll);
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                vRegFp8_2, src1Ptr + ELE_NUM_PER_REPEAT);
            MicroAPI::Cast<float, ElementPerTokenScale, castTraitFp8ToFp32>(vRegMuls1_2, vRegFp8_2, maskAll);
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                vRegFp8_3, src1Ptr + ELE_NUM_PER_REPEAT * 2);
            MicroAPI::Cast<float, ElementPerTokenScale, castTraitFp8ToFp32>(vRegMuls1_3, vRegFp8_3, maskAll);
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                vRegFp8_4, src1Ptr + ELE_NUM_PER_REPEAT * 3);
            MicroAPI::Cast<float, ElementPerTokenScale, castTraitFp8ToFp32>(vRegMuls1_4, vRegFp8_4, maskAll);
        } else if constexpr (std::is_same_v<ElementPerTokenScale, float>) {
            // Already fp32, copy
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_NORM>(vRegMuls1_1, src1Ptr);
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_NORM>(
                vRegMuls1_2, src1Ptr + ELE_NUM_PER_REPEAT);
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_NORM>(
                vRegMuls1_3, src1Ptr + ELE_NUM_PER_REPEAT * 2);
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_NORM>(
                vRegMuls1_4, src1Ptr + ELE_NUM_PER_REPEAT * 3);
        } else {
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_UNPACK_B16>(vRegFp8_1, src1Ptr);
            MicroAPI::Cast<float, ElementPerTokenScale, castTraitFp16ToFp32>(vRegMuls1_1, vRegBf16_1, maskAll);
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vRegFp8_2, src1Ptr + ELE_NUM_PER_REPEAT);
            MicroAPI::Cast<float, ElementPerTokenScale, castTraitFp16ToFp32>(vRegMuls1_2, vRegBf16_2, maskAll);
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vRegFp8_3, src1Ptr + ELE_NUM_PER_REPEAT * 2);
            MicroAPI::Cast<float, ElementPerTokenScale, castTraitFp16ToFp32>(vRegMuls1_3, vRegBf16_3, maskAll);
            MicroAPI::DataCopy<ElementPerTokenScale, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vRegFp8_4, src1Ptr + ELE_NUM_PER_REPEAT * 3);
            MicroAPI::Cast<float, ElementPerTokenScale, castTraitFp16ToFp32>(vRegMuls1_4, vRegBf16_4, maskAll);
        }

        for (uint16_t i = 0; i < row; ++i) {
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(vRegFp32_1, srcPtr + i * col);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(vRegFp32_2, srcPtr + i * col + ELE_NUM_PER_REPEAT);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                vRegFp32_3, srcPtr + i * col + ELE_NUM_PER_REPEAT * 2);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                vRegFp32_4, srcPtr + i * col + ELE_NUM_PER_REPEAT * 3);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(vRegMuls2, src2Ptr + i);
            MicroAPI::Mul(vRegFp32_1, vRegFp32_1, vRegMuls1_1, maskAll);
            MicroAPI::Mul(vRegFp32_2, vRegFp32_2, vRegMuls1_2, maskAll);
            MicroAPI::Mul(vRegFp32_3, vRegFp32_3, vRegMuls1_3, maskAll);
            MicroAPI::Mul(vRegFp32_4, vRegFp32_4, vRegMuls1_4, maskAll);
            MicroAPI::Mul(vRegFp32_1, vRegFp32_1, vRegMuls2, maskAll);
            MicroAPI::Mul(vRegFp32_2, vRegFp32_2, vRegMuls2, maskAll);
            MicroAPI::Mul(vRegFp32_3, vRegFp32_3, vRegMuls2, maskAll);
            MicroAPI::Mul(vRegFp32_4, vRegFp32_4, vRegMuls2, maskAll);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(dstPtr + i * col, vRegFp32_1, maskAll);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                dstPtr + i * col + ELE_NUM_PER_REPEAT, vRegFp32_2, maskAll);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                dstPtr + i * col + ELE_NUM_PER_REPEAT * 2, vRegFp32_3, maskAll);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                dstPtr + i * col + ELE_NUM_PER_REPEAT * 3, vRegFp32_4, maskAll);
        }
    }

    CATLASS_DEVICE_SIMD_VF void MulComputeGeneric(
        __ubuf__ float* dstPtr, __ubuf__ float* srcPtr, __ubuf__ ElementScale* src1Ptr, __ubuf__ float* src2Ptr)
    {
        namespace MicroAPI = AscendC::MicroAPI;

        static constexpr MicroAPI::CastTrait castTraitFp8ToBf16 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};

        static constexpr MicroAPI::CastTrait castTraitBf16ToFp32 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_NONE};

        static constexpr MicroAPI::CastTrait castTraitFp8ToFp32 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};

        static constexpr MicroAPI::CastTrait castTraitFp16ToFp32 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};

        uint32_t row = TileShape::ROW;
        uint32_t col = TileShape::COLUMN;
        constexpr uint32_t ELE_NUM_PER_REPEAT = 64;

        MicroAPI::RegTensor<float> vRegC;
        MicroAPI::RegTensor<float> vRegScale;
        MicroAPI::RegTensor<float> vRegPerTokenScale;
        MicroAPI::RegTensor<ElementScale> vRegScaleRaw;
        MicroAPI::RegTensor<bfloat16_t> vRegBf16;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint32_t i = 0; i < row; ++i) {
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(vRegPerTokenScale, src2Ptr + i);

            for (uint32_t j = 0; j < col; j += ELE_NUM_PER_REPEAT) {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(vRegC, srcPtr + i * col + j);

                if constexpr (std::is_same_v<ElementScale, float8_e8m0_t>) {
                    MicroAPI::DataCopy<ElementScale, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vRegScaleRaw, src1Ptr + j);
                    MicroAPI::Cast<bfloat16_t, ElementScale, castTraitFp8ToBf16>(vRegBf16, vRegScaleRaw, maskAll);
                    MicroAPI::Cast<float, bfloat16_t, castTraitBf16ToFp32>(vRegScale, vRegBf16, maskAll);
                } else if constexpr (
                    std::is_same_v<ElementScale, float8_e4m3_t> || std::is_same_v<ElementScale, float8_e5m2_t>) {
                    MicroAPI::DataCopy<ElementScale, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vRegScaleRaw, src1Ptr + j);
                    MicroAPI::Cast<float, ElementScale, castTraitFp8ToFp32>(vRegScale, vRegScaleRaw, maskAll);
                } else if constexpr (std::is_same_v<ElementScale, float>) {
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(vRegScale, src1Ptr + j);
                } else {
                    MicroAPI::DataCopy<ElementScale, MicroAPI::LoadDist::DIST_UNPACK_B16>(vRegScaleRaw, src1Ptr + j);
                    MicroAPI::Cast<float, ElementScale, castTraitFp16ToFp32>(vRegScale, vRegScaleRaw, maskAll);
                }

                MicroAPI::Mul(vRegC, vRegC, vRegScale, maskAll);
                MicroAPI::Mul(vRegC, vRegC, vRegPerTokenScale, maskAll);
                MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(dstPtr + i * col + j, vRegC, maskAll);
            }
        }
    }
};
} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_FP8_REGBASE_HPP
