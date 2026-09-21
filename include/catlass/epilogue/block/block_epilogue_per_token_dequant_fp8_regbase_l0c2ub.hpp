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

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_FP8_REGBASE_L0C2UB_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_FP8_REGBASE_L0C2UB_HPP

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
    EpilogueAscend950PerTokenPerChannelQuant<UB_STAGES_>, CType_, ScaleType_, PerTokenScaleType_, DType_,
    TileRowBroadcastMul_, TileBroadcastOneBlk_, TileOneBlkColumnBroadcastMul_, TileCopy_, EpilogueTileSwizzle_> {
public:
    using DispatchPolicy = EpilogueAscend950PerTokenPerChannelQuant<UB_STAGES_>;
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
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
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
        (UB_STAGES * (TileShape::COUNT * sizeof(ElementC) + TileShape::COLUMN * sizeof(ElementScale) +
                      TileShape::ROW * sizeof(ElementPerTokenScale) + TileShape::COUNT * sizeof(ElementD)) +
         (TileShape::COUNT + TileShape::COLUMN + TileShape::ROW) * sizeof(float) + TileShape::ROW * BYTE_PER_BLK) <=
            ArchTag::UB_SIZE,
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
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;

        int32_t eventid = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementC);
            ubScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementScale>(ubOffset);
            ubOffset += TileShape::COLUMN * sizeof(ElementScale);
            ubPerTokenScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementPerTokenScale>(ubOffset);
            ubOffset += TileShape::ROW * sizeof(ElementPerTokenScale);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementD);
        }
        ubScaleFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::COLUMN * sizeof(float);
        ubMul = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::COUNT * sizeof(float);
        ubPerTokenScaleFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::ROW * sizeof(float);
        ubPerTokenScaleFp32Brcb = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::ROW * BYTE_PER_BLK;
        ubPerTokenMul = ubMul;
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
        AscendC::GlobalTensor<ElementC> const& gmBlockC, LayoutC const& layoutBlockC, Callback&& callback = Callback{})
    {
        if (actualBlockShapeMNK.k() == 0) {
            return;
        }
        callback();

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
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
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
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);

            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutGmTileC);

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

            copyGmToUbPerTokenScale(
                ubPerTokenScale, gmTilePerTokenScale, layoutUbPerTokenScale, layoutGmTilePerTokenScale);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
            if constexpr (std::is_same<ElementScale, float8_e8m0_t>::value) {
                // High-level API doesn't support float8_e8m0_t -> float, use MicroAPI instead
                __ubuf__ float8_e8m0_t* srcAddr = (__ubuf__ float8_e8m0_t*)ubScale.GetPhyAddr();
                __ubuf__ float* dstAddr = (__ubuf__ float*)ubScaleFp32.GetPhyAddr();
                CastFp8E8m0ToFp32(dstAddr, srcAddr, TileShape::COLUMN);

            } else if (!std::is_same<ElementScale, float>::value) {
                AscendC::Cast(ubScaleFp32, ubScale, AscendC::RoundMode::CAST_NONE, TileShape::COLUMN);
            }

            if constexpr (std::is_same<ElementPerTokenScale, float8_e8m0_t>::value) {
                // High-level API doesn't support float8_e8m0_t -> float, use MicroAPI instead
                __ubuf__ float8_e8m0_t* srcAddr = (__ubuf__ float8_e8m0_t*)ubPerTokenScale.GetPhyAddr();
                __ubuf__ float* dstAddr = (__ubuf__ float*)ubPerTokenScaleFp32.GetPhyAddr();
                CastFp8E8m0ToFp32(dstAddr, srcAddr, TileShape::ROW);

            } else if (!std::is_same<ElementPerTokenScale, float>::value) {
                AscendC::Cast(ubPerTokenScaleFp32, ubPerTokenScale, AscendC::RoundMode::CAST_NONE, TileShape::ROW);
            }

            AscendC::PipeBarrier<PIPE_V>();
            tileRowBroadcastMul(ubMul, ubC, ubScaleFp32);
            AscendC::PipeBarrier<PIPE_V>();
            tileBroadcastOneBlk(ubPerTokenScaleFp32Brcb, ubPerTokenScaleFp32);
            AscendC::PipeBarrier<PIPE_V>();
            tileOneBlkColumnBroadcastMul(ubPerTokenMul, ubMul, ubPerTokenScaleFp32Brcb);
            AscendC::PipeBarrier<PIPE_V>();

            auto& ubD = ubDList[ubListId];
            LayoutD layoutUbD{actualTileShape, ubTileStride};

            AscendC::PipeBarrier<PIPE_ALL>();

            if constexpr (std::is_same_v<ElementD, half>) {
                AscendC::Cast(ubD, ubPerTokenMul, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            auto gmTileD = gmD[params.layoutD.GetOffset(tileOffset)];
            auto layoutGmTileD = params.layoutD.GetTileLayout(actualTileShape);

            if constexpr (std::is_same_v<ElementD, half>) {
                copyUbToGmD(gmTileD, ubD, layoutGmTileD, layoutUbD);
            } else {
                copyUbToGmD(gmTileD, ubPerTokenMul, layoutGmTileD, layoutUbD);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementScale> ubScaleList[UB_STAGES];
    AscendC::LocalTensor<ElementPerTokenScale> ubPerTokenScaleList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbScaleVMTE2List[UB_STAGES];
    int32_t eventUbScaleMTE2VList[UB_STAGES];
    int32_t eventUbPerTokenScaleVMTE2List[UB_STAGES];
    int32_t eventUbPerTokenScaleMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];
    int32_t eventList[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor<float> ubScaleFp32;
    AscendC::LocalTensor<float> ubMul;
    AscendC::LocalTensor<float> ubPerTokenScaleFp32;
    AscendC::LocalTensor<float> ubPerTokenScaleFp32Brcb;
    AscendC::LocalTensor<float> ubPerTokenMul;

    TileRowBroadcastMul tileRowBroadcastMul;
    TileBroadcastOneBlk tileBroadcastOneBlk;
    TileOneBlkColumnBroadcastMul tileOneBlkColumnBroadcastMul;

    CopyGmToUbC copyGmToUbC;
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
};
} // namespace Catlass::Epilogue::Block
#endif
