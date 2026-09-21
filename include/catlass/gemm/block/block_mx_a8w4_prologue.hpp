/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_MX_A8W4_PROLOGUE_HPP
#define CATLASS_GEMM_BLOCK_MX_A8W4_PROLOGUE_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {
template <class ArchTag, uint32_t L1B_STAGES_, class InType_, class OutType_, class TileShapeL1_, class TileCopy_>
struct BlockPrologue<MxA8W4Prologue<ArchTag, L1B_STAGES_>, InType_, OutType_, TileShapeL1_, TileCopy_> {
public:
    using DispatchPolicy = MxA8W4Prologue<ArchTag, L1B_STAGES_>;
    using ElementIn = typename InType_::Element;
    using ElementOut = typename OutType_::Element;
    using LayoutIn = typename InType_::Layout;
    using LayoutOut = typename OutType_::Layout;
    using TileShapeL1 = TileShapeL1_;
    using TileCopy = TileCopy_;
    using LayoutPrologueB = typename TileCopy::LayoutPrologueB;

    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;

    struct Params {
        TileShapeL1 tileShapeL1;
        LayoutPrologueB layoutPrologueB;
        bool hasBias;
        Catlass::Arch::Resource<ArchTag>& resource;
    };

    struct VfParamsNz {
        uint32_t loopKNum;
        uint32_t innerLoopNum;
        uint32_t loopKDstStride;
        uint32_t innerDstStride;
        uint32_t nRealSizeAlign;
        __ubuf__ ElementIn* weightInUbAddr;
        __ubuf__ ElementOut* weightOutUbAddr;
    };

    struct VfParamsNd {
        uint16_t outExtend;
        uint16_t innerExtend;
        uint32_t dataBlockStride;
        uint32_t repeatStride;
        int32_t outDimOffset;
        uint32_t maskB8Tail0;
        uint32_t maskB8Tail1;

        int32_t kUbLen;

        __ubuf__ int8_t* weightInUbBaseAddr;
        __ubuf__ ElementOut* weightOutUbAddr;
        __ubuf__ ElementOut* weightOutUbAddr1;
    };

    static_assert(
        std::is_same_v<LayoutIn, layout::RowMajor> || std::is_same_v<LayoutIn, layout::ColumnMajor> ||
            std::is_same_v<LayoutIn, layout::zN> || std::is_same_v<LayoutIn, layout::Weight4BitnZ>,
        "Unsupported layout, only can be Rowmajor ColumnMajor or zN or nZ");

    CATLASS_DEVICE
    BlockPrologue(const Params& params)
    {
        static constexpr int32_t OFFSET_64 = 64;
        vecWeightInLen =
            (UB_STAGES * (tla::get<1>(params.tileShapeL1) * RoundUp(tla::get<2>(params.tileShapeL1), OFFSET_64))) >>
            INT4_DTYPE_PARAM;
        vecWeightOutLen = UB_STAGES * (RoundUp(tla::get<1>(params.tileShapeL1), AscendC::BLOCK_CUBE) + 1) *
                          RoundUp(
                              RoundUp(tla::get<2>(params.tileShapeL1), static_cast<int32_t>(AscendC::ONE_BLK_SIZE)),
                              static_cast<int32_t>(K_ALIGN_SIZE));

        uint32_t l1Offset = 0;
        uint32_t L1B_TILE_SIZE = tla::get<1>(params.tileShapeL1) * tla::get<2>(params.tileShapeL1);
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1BTensorList[i] = params.resource.l1Buf.template GetBufferByByte<ElementOut>(l1Offset);
            l1Offset += L1B_TILE_SIZE;
            // Assign event ID for each stages
            l1BEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(l1BEventList[i]);
        }
        uint32_t ubOffset = 0;
        for (uint32_t i = 0; i < UB_STAGES; i++) {
            ubCastInTensor[i] = params.resource.ubBuf.template GetBufferByByte<ElementIn>(ubOffset);
            ubOffset += 32 * 1024;
            ubCastOutTensor[i] = params.resource.ubBuf.template GetBufferByByte<ElementOut>(ubOffset);
            ubOffset += 32 * 1024;
        }
    }

    CATLASS_DEVICE
    ~BlockPrologue()
    {
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(l1BEventList[i]);
        }
    }

    template <class TensorBIn, class ActualBlockShape>
    CATLASS_DEVICE void operator()(
        const TensorBIn& bGlobal, const ActualBlockShape& actualBlockShape, const Params& params)
    {
        uint32_t kSize = tla::get<0>(params.layoutPrologueB.originShape());
        uint32_t kL1Size = tla::get<2>(params.tileShapeL1);
        uint32_t kGmLoop = CeilDiv(kSize, static_cast<uint64_t>(tla::get<2>(params.tileShapeL1)));

        uint32_t kL1TileShape;
        uint32_t kL1TileCoord;
        uint32_t kL1Offset;
        for (uint64_t kLoopIdx = 0; kLoopIdx < kGmLoop; kLoopIdx++) {
            auto l1BLayout = tla::MakeLayout<ElementOut, LayoutOut>(
                tla::get<2>(params.tileShapeL1), tla::get<1>(params.tileShapeL1));
            auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], l1BLayout, Arch::PositionL1{});
            kL1TileCoord = kLoopIdx * kL1Size;

            bool disableSubVec = false;
            if (kLoopIdx == kGmLoop - 1) {
                kL1TileShape = kSize - kLoopIdx * kL1Size;
                kL1Offset = RoundUp(kL1TileShape / AscendC::GetSubBlockNum(), 32);

                // Since kL1Offset is rounded up, kL1TileShape - kL1Offset may be negative.
                // Disable the sub-vector in this case.
                if (kL1Offset > kL1TileShape && AscendC::GetSubBlockIdx() != 0) {
                    disableSubVec = true;
                }

                uint32_t klastLoopCoord = kL1TileCoord + kL1Offset * AscendC::GetSubBlockIdx();
                uint32_t klastLoopShape = AscendC::GetSubBlockIdx() == 0 ? kL1Offset : kL1TileShape - kL1Offset;

                auto tileTensorGmB = GetTile(
                    bGlobal, tla::MakeCoord(klastLoopCoord, 0), tla::MakeShape(klastLoopShape, actualBlockShape.n()));
                auto tileTensorL1B = GetTile(
                    tensorL1B, tla::MakeCoord(kL1Offset * AscendC::GetSubBlockIdx(), 0),
                    tla::MakeShape(klastLoopShape, actualBlockShape.n()));

                if constexpr (std::is_same_v<LayoutIn, layout::Weight4BitnZ>) {
                    ProcessL1Nz(tileTensorL1B, tileTensorGmB, disableSubVec);
                } else {
                    ProcessL1Nd(tileTensorL1B, tileTensorGmB, disableSubVec);
                }
            } else {
                kL1TileShape = kL1Size / AscendC::GetSubBlockNum();
                kL1Offset = AscendC::GetSubBlockIdx() * kL1TileShape;

                auto tileTensorGmB = GetTile(
                    bGlobal, tla::MakeCoord(kL1TileCoord + kL1Offset, 0),
                    tla::MakeShape(kL1TileShape, actualBlockShape.n()));
                auto tileTensorL1B = GetTile(
                    tensorL1B, tla::MakeCoord(kL1Offset * AscendC::GetSubBlockIdx(), 0),
                    tla::MakeShape(kL1TileShape, actualBlockShape.n()));

                if constexpr (std::is_same_v<LayoutIn, layout::Weight4BitnZ>) {
                    ProcessL1Nz(tileTensorL1B, tileTensorGmB);
                } else {
                    ProcessL1Nd(tileTensorL1B, tileTensorGmB);
                }
            }
            l1BListId = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;
        }
    }

    template <class TensorOut, class TensorIn>
    __aicore__ inline void ProcessL1Nz(const TensorOut& tensorOut, const TensorIn& tensorIn, bool disableSubVec = false)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(l1BEventList[l1BListId]);
        uint32_t shapeN;
        uint32_t shapeK;

        shapeN = tla::get<1, 0>(tensorIn.shape()) * tla::get<1, 1>(tensorIn.shape()); // nZ
        shapeK = tla::get<0, 0>(tensorIn.shape()) * tla::get<0, 1>(tensorIn.shape());

        auto layoutCastIn = tla::MakeLayout<ElementOut, layout::Weight4BitnZ>(shapeK, shapeN);
        auto layoutCastOut = tla::MakeLayout<ElementOut, layout::nZ>(shapeK, shapeN);

        auto tensorCastIn = tla::MakeTensor(ubCastInTensor[l1BListId], layoutCastIn, Catlass::Arch::PositionUB{});
        auto tensorCastOut = tla::MakeTensor(ubCastOutTensor[l1BListId], layoutCastOut, Catlass::Arch::PositionUB{});

        if (!disableSubVec) {
            CopyNzInWeightTensor(tensorCastIn, tensorIn);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(l1BEventList[l1BListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(l1BEventList[l1BListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(l1BEventList[l1BListId]);

        if (!disableSubVec) {
            AntiQuantComputeNz(tensorCastOut, tensorCastIn);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(l1BEventList[l1BListId]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(l1BEventList[l1BListId]);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(l1BEventList[l1BListId]);

        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + l1BListId);

        if (!disableSubVec) {
            CopyUb2L1(tensorOut, tensorCastOut);
        }

        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + l1BListId);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(l1BEventList[l1BListId]);
    }

    template <class TensorOut, class TensorIn>
    __aicore__ inline void ProcessL1Nd(const TensorOut& tensorOut, const TensorIn& tensorIn, bool disableSubVec = false)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(l1BEventList[l1BListId]);

        uint32_t shapeN = tla::get<1>(tensorIn.shape());
        uint32_t shapeK = tla::get<0>(tensorIn.shape());

        auto layoutCastIn = tla::MakeLayout<ElementIn, layout::ColumnMajor>(shapeK, shapeN);
        auto layoutCastOut = tla::MakeLayout<ElementOut, layout::nZ>(shapeK, shapeN);

        auto tensorCastIn = tla::MakeTensor(ubCastInTensor[l1BListId], layoutCastIn, Catlass::Arch::PositionUB{});
        auto tensorCastOut = tla::MakeTensor(ubCastOutTensor[l1BListId], layoutCastOut, Catlass::Arch::PositionUB{});

        if (!disableSubVec) {
            CopyNdInWeightTensor(tensorCastIn, tensorIn);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(l1BEventList[l1BListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(l1BEventList[l1BListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(l1BEventList[l1BListId]);

        if (!disableSubVec) {
            AntiQuantComputeNd(tensorCastOut, tensorCastIn);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(l1BEventList[l1BListId]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(l1BEventList[l1BListId]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(l1BEventList[l1BListId]);

        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + l1BListId);

        if (!disableSubVec) {
            CopyUb2L1(tensorOut, tensorCastOut);
        }

        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + l1BListId);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(l1BEventList[l1BListId]);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyNzInWeightTensor(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        uint32_t blockCount = tla::get<0, 1>(srcTensor.shape());
        uint32_t blockLen = tla::get<0, 1>(dstTensor.stride()) >> 1;
        AscendC::DataCopyExtParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = (tla::get<0, 1>(srcTensor.stride()) - tla::get<0, 1>(dstTensor.stride())) >> 1;
        repeatParams.dstStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPadExtParams<typename TensorDst::Element> padParams;
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams, padParams);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyNdInWeightTensor(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        AscendC::DataCopyExtParams intriParams;
        intriParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<ElementIn> padParams;
        intriParams.blockCount = tla::get<1>(srcTensor.shape());
        intriParams.blockLen = CeilDiv(tla::get<0>(srcTensor.shape()), 2);
        intriParams.srcStride =
            CeilDiv(tla::get<1>(srcTensor.stride()), 2) - CeilDiv(tla::get<0>(srcTensor.shape()), 2);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams, padParams);
    }

    template <class TensorCastOut, class TensorCastIn>
    __aicore__ inline void AntiQuantComputeNz(const TensorCastOut& tensorCastOut, const TensorCastIn& tensorCastIn)
    {
        static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementIn>::value;

        VfParamsNz params;
        params.weightInUbAddr = (__ubuf__ ElementIn*)tensorCastIn.data().GetPhyAddr();
        params.weightOutUbAddr = (__ubuf__ ElementOut*)tensorCastOut.data().GetPhyAddr();

        params.loopKNum = tla::get<0, 1>(tensorCastIn.shape());
        params.nRealSizeAlign = tla::get<1, 1>(tensorCastIn.shape()) * AscendC::BLOCK_CUBE;
        params.innerDstStride = AscendC::GetVecLen();
        params.innerLoopNum = (params.nRealSizeAlign * tla::get<0, 0>(tensorCastIn.shape())) /
                              static_cast<uint64_t>(AscendC::GetVecLen());
        params.loopKDstStride = params.innerLoopNum * params.innerDstStride;

        RegComputeNz<TensorCastOut, TensorCastIn>(params);
    }

    template <class TensorCastOut, class TensorCastIn>
    CATLASS_DEVICE_SIMD_VF void RegComputeNz(VfParamsNz params)
    {
        AscendC::Reg::RegTensor<int8_t> wShrReg;
        AscendC::Reg::RegTensor<int8_t> wShlReg;
        AscendC::Reg::RegTensor<int8_t> wAndReg;
        AscendC::Reg::RegTensor<int8_t> wLoad;
        AscendC::Reg::RegTensor<int8_t> wShl;
        AscendC::Reg::RegTensor<int8_t> wShr0;
        AscendC::Reg::RegTensor<int8_t> wShr1;
        AscendC::Reg::RegTensor<int8_t> wSel;
        AscendC::Reg::RegTensor<int8_t> wAnd;

        AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg pregVsel = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShrReg, E2M1_SHIFT_RIGHT_SIZE, preg);
        AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
        AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wAndReg, E2M1_AND_MASK, preg);

        for (uint16_t loopKIdx = 0; loopKIdx < params.loopKNum; ++loopKIdx) {
            for (uint16_t innerLoopIdx = 0; innerLoopIdx < params.innerLoopNum; ++innerLoopIdx) {
                // DIST_US_B8 load mode expands each packed B4 byte into lane-aligned B8 slots.
                // Packed B4 address offset (bytes) = logical element index >> 1.
                AscendC::Reg::AddrReg aregWeightB8In = AscendC::Reg::CreateAddrReg<uint8_t>(
                    loopKIdx, (C0_SIZE_B8 * params.nRealSizeAlign) >> 1, innerLoopIdx, AscendC::GetVecLen() >> 1);
                AscendC::Reg::LoadAlign<uint8_t, AscendC::Reg::LoadDist::DIST_US_B8>(
                    (AscendC::Reg::RegTensor<uint8_t>&)wLoad, (__ubuf__ uint8_t*&)params.weightInUbAddr,
                    aregWeightB8In);

                AscendC::Reg::ShiftRight(wShr0, wLoad, wShrReg, preg);
                AscendC::Reg::ShiftLeft(wShl, wLoad, wShlReg, preg);
                AscendC::Reg::ShiftRight(wShr1, wShl, wShrReg, preg);
                AscendC::Reg::Select(wSel, wShr1, wShr0, pregVsel);
                AscendC::Reg::And(wAnd, wSel, wAndReg, preg);

                AscendC::Reg::AddrReg aregWeightB8Out = AscendC::Reg::CreateAddrReg<uint8_t>(
                    loopKIdx, params.loopKDstStride, innerLoopIdx, params.innerDstStride);
                AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(
                    (__ubuf__ uint8_t*&)params.weightOutUbAddr, (AscendC::Reg::RegTensor<uint8_t>&)wAnd,
                    aregWeightB8Out, preg);
            }
        }
    }

    template <class TensorCastOut, class TensorCastIn>
    __aicore__ inline void AntiQuantComputeNd(const TensorCastOut& tensorCastOut, const TensorCastIn& tensorCastIn)
    {
        uint32_t weightOutUbOffset;
        uint32_t weightInUbOffset;
        weightOutUbOffset = l1BListId * (vecWeightOutLen / sizeof(ElementOut) / L1B_STAGES);
        weightInUbOffset = l1BListId * (vecWeightInLen << INT4_DTYPE_PARAM) / L1B_STAGES;
        __ubuf__ int8_t* weightInUbAddr = (__ubuf__ int8_t*)tensorCastIn.data().GetPhyAddr();
        __ubuf__ ElementOut* weightOutUbAddr = (__ubuf__ ElementOut*)tensorCastOut.data().GetPhyAddr();

        int32_t kUbLen = tla::get<0>(tensorCastIn.shape());
        int32_t nUbLen = tla::get<1>(tensorCastIn.shape());
        uint16_t blockStride = RoundUp(nUbLen, AscendC::BLOCK_CUBE);
        __ubuf__ ElementOut* weightOutUbAddr1 = weightOutUbAddr + VEC_MAX_ELEM_B8 * blockStride;

        VfParamsNd wParams;
        wParams.outExtend = static_cast<uint16_t>(nUbLen);
        wParams.innerExtend = CeilDiv(RoundUp(kUbLen, K_ALIGN_SIZE), VECTOR_REG_WIDTH_FOR_4BITS);
        wParams.dataBlockStride = RoundUp(nUbLen, AscendC::BLOCK_CUBE);
        wParams.repeatStride = wParams.dataBlockStride * AscendC::BLOCK_CUBE;
        wParams.outDimOffset =
            AscendC::ONE_BLOCK_SIZE - wParams.innerExtend * wParams.repeatStride * AscendC::ONE_BLOCK_SIZE;
        wParams.maskB8Tail0 =
            Min(kUbLen % VECTOR_REG_WIDTH_FOR_4BITS, static_cast<int32_t>(AscendC::VECTOR_REG_WIDTH)) +
            kUbLen / VECTOR_REG_WIDTH_FOR_4BITS * AscendC::VECTOR_REG_WIDTH;
        wParams.maskB8Tail1 =
            Max(kUbLen % VECTOR_REG_WIDTH_FOR_4BITS - static_cast<int32_t>(AscendC::VECTOR_REG_WIDTH), 0) +
            kUbLen / VECTOR_REG_WIDTH_FOR_4BITS * AscendC::VECTOR_REG_WIDTH;
        wParams.kUbLen = kUbLen;
        wParams.weightInUbBaseAddr = weightInUbAddr;
        wParams.weightOutUbAddr = weightOutUbAddr;
        wParams.weightOutUbAddr1 = weightOutUbAddr1;
        RegComputeNd(wParams);
    }

    CATLASS_DEVICE_SIMD_VF void RegComputeNd(const VfParamsNd wParams)
    {
        __ubuf__ ElementOut* weightOutUbAddr = wParams.weightOutUbAddr;
        __ubuf__ ElementOut* weightOutUbAddr1 = wParams.weightOutUbAddr1;
        AscendC::MicroAPI::RegTensor<uint8_t> wDIntlv0, wDIntlv1, wLoad0, sAnd0, sAnd1, wShr, wShl, s1, wOr0, wOr1,
            wdup1, wdup4;
        AscendC::MicroAPI::RegTensor<int8_t> wdup0, wdup2, wdup3;
        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup0, DUP_CONFIG_2, preg);
        AscendC::MicroAPI::Duplicate<uint8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            wdup1, DUP_CONFIG_MODE_1C, preg);
        AscendC::MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup2, DUP_CONFIG_2, preg);
        AscendC::MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup3, DUP_CONFIG_4, preg);
        AscendC::MicroAPI::Duplicate<uint8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wdup4, DUP_FLAG_80, preg);
        // 一次处理一个N轴
        for (uint16_t outIdx = 0; outIdx < wParams.outExtend; ++outIdx) {
            uint32_t maskWeight0Tmp = wParams.maskB8Tail0;
            uint32_t maskWeight1Tmp = wParams.maskB8Tail1;
            for (uint16_t repeatIdx = 0; repeatIdx < wParams.innerExtend; ++repeatIdx) {
                AscendC::MicroAPI::MaskReg MaskRegB8Tail0 = AscendC::MicroAPI::UpdateMask<uint8_t>(maskWeight0Tmp);
                AscendC::MicroAPI::MaskReg MaskRegB8Tail1 = AscendC::MicroAPI::UpdateMask<uint8_t>(maskWeight1Tmp);
                AscendC::MicroAPI::AddrReg aregWeightB8 = AscendC::MicroAPI::CreateAddrReg<uint8_t>(
                    outIdx, RoundUp(wParams.kUbLen, static_cast<int32_t>(K_ALIGN_SIZE)) >> 1, repeatIdx,
                    VEC_MAX_ELEM_B8);
                AscendC::MicroAPI::LoadAlign(wLoad0, (__ubuf__ uint8_t*&)wParams.weightInUbBaseAddr, aregWeightB8);
                // 提取E/M
                AscendC::MicroAPI::ShiftRight(wShr, wLoad0, wdup0, preg); // vr1
                AscendC::MicroAPI::And(wShr, wShr, wdup1, preg);          // vr1
                AscendC::MicroAPI::ShiftLeft(wShl, wLoad0, wdup2, preg);  // vr2
                AscendC::MicroAPI::And(wShl, wShl, wdup1, preg);          // vr2
                // 提取S
                AscendC::MicroAPI::ShiftLeft(s1, wLoad0, wdup3, preg); // vr3
                AscendC::MicroAPI::And(sAnd0, s1, wdup4, preg);        // vr3
                AscendC::MicroAPI::And(sAnd1, wLoad0, wdup4, preg);    // vr4
                // 合并S/E/M
                AscendC::MicroAPI::Or(wOr0, wShr, sAnd1, preg); // odd
                AscendC::MicroAPI::Or(wOr1, wShl, sAnd0, preg); // even
                AscendC::MicroAPI::Interleave(wDIntlv0, wDIntlv1, wOr1, wOr0);
                AscendC::MicroAPI::StoreAlign<
                    uint8_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ uint8_t*&)weightOutUbAddr, wDIntlv0, wParams.dataBlockStride, wParams.repeatStride,
                    MaskRegB8Tail0);
                AscendC::MicroAPI::StoreAlign<
                    uint8_t, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ uint8_t*&)weightOutUbAddr1, wDIntlv1, wParams.dataBlockStride, wParams.repeatStride,
                    MaskRegB8Tail1);
            }
            weightOutUbAddr += wParams.outDimOffset;
            weightOutUbAddr1 += wParams.outDimOffset;
        }
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyUb2L1(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementOut>::value;
        int64_t brustCount = tla::get<0, 1>(srcTensor.shape());
        int64_t burstLen = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());

        AscendC::DataCopyParams dataCopyParams(
            brustCount, burstLen, (tla::get<0, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - burstLen),
            (tla::get<0, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - burstLen));

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams);
    }

    static constexpr uint64_t K_ALIGN_SIZE = 64;
    static constexpr int32_t C0_SIZE_B8 = 32;
    static constexpr uint64_t INT4_DTYPE_PARAM = 1;
    static constexpr int32_t VECTOR_REG_WIDTH_FOR_4BITS = 512;
    static constexpr int32_t VEC_MAX_ELEM_B8 = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementOut);

    static constexpr uint32_t DUP_CONFIG_2 = 0x2;
    static constexpr uint32_t DUP_CONFIG_MODE_1C = 0x1C;
    static constexpr uint32_t DUP_CONFIG_4 = 0x4;
    static constexpr uint32_t DUP_FLAG_80 = 0x80;
    static constexpr uint32_t E2M1_SHIFT_RIGHT_SIZE = 0x2;
    static constexpr uint32_t E2M1_AND_MASK = 0x9C;
    static constexpr uint32_t SHIFT_RIGHT_SIZE = 0x4;
    static constexpr uint32_t SHIFT_LEFT_SIZE = 0x4;

    static constexpr int32_t SYNC_MODE = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;

    static constexpr int64_t UB_STAGES = L1B_STAGES;

    uint32_t l1BListId{0};
    uint32_t vecWeightOutLen;
    uint32_t vecWeightInLen;
    int32_t nUbLen;
    int32_t kUbLen;
    int32_t l1BEventList[L1B_STAGES];

    AscendC::LocalTensor<ElementIn> ubCastInTensor[UB_STAGES];
    AscendC::LocalTensor<ElementOut> ubCastOutTensor[UB_STAGES];
    AscendC::LocalTensor<ElementOut> l1BTensorList[L1B_STAGES];
};
} // namespace Catlass::Gemm::Block

#endif
