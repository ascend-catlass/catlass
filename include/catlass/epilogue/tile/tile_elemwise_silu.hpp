/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_SILU_HPP
#define CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_SILU_HPP

#include "catlass/catlass.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
using namespace AscendC::Reg;
#endif

namespace Catlass::Epilogue::Tile {
template <
    // / Tag indicating architecture
    class ArchTag_,
    // / Compute data type
    class ComputeType_,
    // / COMPUTE_LENGTH of the compute buffer
    uint32_t COMPUTE_LENGTH_>
struct TileElemWiseSilu {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    CATLASS_DEVICE
    TileElemWiseSilu()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementCompute> const& dstLocal, AscendC::LocalTensor<ElementCompute> const& srcLocal)
    {
        using namespace AscendC;
        // d: -x, s: x
        Muls(dstLocal, srcLocal, (ElementCompute)-1, COMPUTE_LENGTH);
        // d: exp(-x), s: x
        Exp(dstLocal, dstLocal, COMPUTE_LENGTH);
        // d: 1 + exp(-x), s: x
        Adds(dstLocal, dstLocal, (ElementCompute)1, COMPUTE_LENGTH);
        // d: x / 1 + exp(-x), s: x
        Div(dstLocal, srcLocal, dstLocal, COMPUTE_LENGTH);
    }
};

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
template <
    // / Tag indicating architecture
    class ArchTag_,
    // / Compute data type
    class ElementDst_, class ElementSrc_>
struct TileElemWiseSiluScaleRegBase {
    using ArchTag = ArchTag_;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementSrc_;

    static_assert(std::is_same_v<ElementSrc, float32_t>, "ElementSrc must be float32_t");
    static_assert(std::is_same_v<ElementDst, half>, "ElementDst must be half");

    CATLASS_DEVICE
    TileElemWiseSiluScaleRegBase()
    {}

    CATLASS_DEVICE_SIMD_VF void SiluScaleVf_b32tob16_nd2nz(
        __ubuf__ ElementDst* dstUb, __ubuf__ ElementSrc* srcUb, uint32_t actualRowNum, uint32_t actualColumnNum,
        uint32_t dstEleNumInC0, uint32_t dstC0Num, uint32_t dstRowStrideInnerFractal,
        uint32_t dstColumnStrideInterFractal, uint32_t srcRowStride, float scale)
    {
        static constexpr CastTrait castTraitB16ToB32 = {
            RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr CastTrait castTraitB32ToB16_0 = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait castTraitB32ToB16_1 = {
            RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        static constexpr uint16_t vlSize = static_cast<uint16_t>(AscendC::GetVecLen() / sizeof(ElementSrc));
        uint32_t vlDataBlockNum = vlSize / dstEleNumInC0;

        uint16_t loops = AscendC::CeilDivision(actualColumnNum, vlSize);

        __ubuf__ ElementSrc* srcRowStart{nullptr};
        __ubuf__ ElementDst* dstRowStart{nullptr};

        MaskReg pregFullB32 = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregFullB16 = CreateMask<half, MaskPattern::ALL>();
        MaskReg pregHalfB16 = CreateMask<half, MaskPattern::H>();

        RegTensor<ElementSrc> srcVreg0;
        RegTensor<ElementSrc> srcVreg1;
        RegTensor<float> computeVreg0;
        RegTensor<float> computeVreg1;
        RegTensor<ElementDst> dstVreg0;
        RegTensor<ElementDst> dstVreg1;

        for (uint16_t loopIdx = 0; loopIdx < loops; loopIdx++) {
            srcRowStart = srcUb + loopIdx * vlSize;
            dstRowStart = dstUb + loopIdx * dstColumnStrideInterFractal * vlDataBlockNum;

            for (uint16_t rowIdx = 0; rowIdx < actualRowNum; rowIdx++) {
                LoadAlign<ElementSrc, PostLiteral::POST_MODE_UPDATE>(srcVreg0, srcRowStart, srcRowStride);
                Muls(computeVreg0, srcVreg0, -1.0f, pregFullB32);       // d: -x, s: x
                Exp(computeVreg0, computeVreg0, pregFullB32);           // d: exp(-x), s: x
                Adds(computeVreg0, computeVreg0, 1.0f, pregFullB32);    // d: 1 + exp(-x), s: x
                Div(computeVreg0, srcVreg0, computeVreg0, pregFullB32); // d: x / 1 + exp(-x), s: x
                Muls(computeVreg0, computeVreg0, scale, pregFullB32);   // d: x / (1 + e^(-x)) * scale, s: x
                Cast<ElementDst, float, castTraitB32ToB16_0>(dstVreg0, computeVreg0, pregFullB16);
                DeInterleave(dstVreg0, dstVreg1, dstVreg0, dstVreg1);
                StoreAlign<ElementDst, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dstRowStart, dstVreg0, dstC0Num, (uint32_t)1, pregHalfB16);
            }
        }
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const& tensorUbDst, TensorSrc const& tensorUbSrc, MatrixCoord const& actualTileShape, float scale)
    {
        __ubuf__ ElementDst* ubDstPtr = (__ubuf__ ElementDst*)tensorUbDst.data().GetPhyAddr();
        __ubuf__ ElementSrc* ubSrcPtr = (__ubuf__ ElementSrc*)tensorUbSrc.data().GetPhyAddr();

        uint32_t actualRowNum = actualTileShape.row();
        uint32_t actualColumnNum = actualTileShape.column();

        uint32_t dstStride00 = tla::get<0, 0>(tensorUbDst.stride()); // dst row stride in inner fractal
        uint32_t dstStride11 = tla::get<1, 1>(tensorUbDst.stride()); // dst column stride in inter fractal
        uint32_t dstEleNumInC0 = tla::get<1, 0>(tensorUbDst.shape());
        uint32_t dstC0Num = tla::get<0, 0>(tensorUbDst.shape()) * tla::get<0, 1>(tensorUbDst.shape());

        uint32_t srcRowStride = tla::get<0>(tensorUbSrc.stride());

        static constexpr uint16_t vlSize = static_cast<uint16_t>(AscendC::GetVecLen() / sizeof(ElementSrc));
        uint32_t vlDataBlockNum = vlSize / dstEleNumInC0;
        uint16_t loops = AscendC::CeilDivision(actualColumnNum, vlSize);

        SiluScaleVf_b32tob16_nd2nz(
            ubDstPtr, ubSrcPtr, actualRowNum, actualColumnNum, dstEleNumInC0, dstC0Num, dstStride00, dstStride11,
            srcRowStride, scale);
    }

    CATLASS_DEVICE_SIMD_VF void FillZero_vf_b32tob16_nd2nz_1vl(
        __ubuf__ ElementDst* ubDst, __ubuf__ ElementSrc* ubSrc, uint32_t rowNum, uint32_t dstC0Num,
        uint32_t srcRowStride, float scale)
    {
        __ubuf__ ElementSrc* srcRowStart = ubSrc;
        __ubuf__ ElementDst* dstRowStart = ubDst;
        RegTensor<ElementDst> dstVreg0;
        Duplicate<ElementDst>(dstVreg0, (ElementDst)0.0);

        MaskReg pregHalfB16 = CreateMask<half, MaskPattern::H>();
        for (uint16_t rowIdx = 0; rowIdx < rowNum; rowIdx++) {
            StoreAlign<ElementDst, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                dstRowStart, dstVreg0, dstC0Num, (uint32_t)1, pregHalfB16);
        }
    }

    CATLASS_DEVICE_SIMD_VF void SiluScale_vf_b32tob16_nd2nz_1vl(
        __ubuf__ ElementDst* dstRowStart, __ubuf__ ElementSrc* srcRowStart, uint32_t rowNum, uint32_t dstC0Num,
        uint32_t srcRowStride, float scale, uint32_t firstRowValidNum)
    {
        static constexpr CastTrait castTraitB16ToB32 = {
            RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr CastTrait castTraitB32ToB16_0 = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait castTraitB32ToB16_1 = {
            RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        MaskReg pregFullB32 = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregHalfB16 = CreateMask<half, MaskPattern::H>();

        RegTensor<ElementSrc> srcVreg0;
        RegTensor<float> computeVreg0;
        RegTensor<ElementDst> dstVreg0;
        RegTensor<ElementDst> dstVreg1;

        for (uint16_t rowIdx = 0; rowIdx < rowNum; rowIdx++) {
            uint32_t maskValue32 = firstRowValidNum + rowIdx;
            uint32_t maskValue16 = firstRowValidNum + rowIdx;
            MaskReg maskB32 = UpdateMask<float>(maskValue32);
            MaskReg maskB16 = UpdateMask<ElementDst>(maskValue16);

            LoadAlign<ElementSrc, PostLiteral::POST_MODE_UPDATE>(srcVreg0, srcRowStart, srcRowStride);
            Muls(computeVreg0, srcVreg0, -1.0f, maskB32);       // d: -x, s: x
            Exp(computeVreg0, computeVreg0, maskB32);           // d: exp(-x), s: x
            Adds(computeVreg0, computeVreg0, 1.0f, maskB32);    // d: 1 + exp(-x), s: x
            Div(computeVreg0, srcVreg0, computeVreg0, maskB32); // d: x / 1 + exp(-x), s: x
            Muls(computeVreg0, computeVreg0, scale, maskB32);   // d: x / (1 + e^(-x)) * scale, s: x
            Cast<ElementDst, float, castTraitB32ToB16_0>(dstVreg0, computeVreg0, maskB32);
            DeInterleave(dstVreg0, dstVreg1, dstVreg0, dstVreg1);
            StoreAlign<ElementDst, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                dstRowStart, dstVreg0, dstC0Num, (uint32_t)1, pregHalfB16);
        }
    }

    CATLASS_DEVICE
    void SiluScaleVf_with_mask(
        __ubuf__ ElementDst* dstUb, __ubuf__ ElementSrc* srcUb, uint32_t actualRowNum, uint32_t actualColumnNum,
        uint32_t topLeftDotRowIdxIn, uint32_t topLeftDotColIdxIn, uint32_t dstEleNumInC0, uint32_t dstC0Num,
        uint32_t dstRowStrideInnerFractal, uint32_t dstColumnStrideInterFractal, uint32_t srcRowStride, float scale)
    {
        static constexpr uint16_t vlSize = static_cast<uint16_t>(AscendC::GetVecLen() / sizeof(ElementSrc));
        uint16_t loops = AscendC::CeilDivision(actualColumnNum, vlSize);
        uint32_t vlDataBlockNum = vlSize / dstEleNumInC0;

        __ubuf__ ElementSrc* srcRowStart{nullptr};
        __ubuf__ ElementDst* dstRowStart{nullptr};

        uint32_t maskRowIdx;
        uint32_t maskColIdx;
        uint32_t firstRowValidNum = 0;

        uint32_t topLeftDotRowIdx = topLeftDotRowIdxIn;
        uint32_t topRightDotRowIdx = topLeftDotRowIdxIn;
        uint32_t bottomLeftDotRowIdx = topLeftDotRowIdxIn + actualRowNum - 1;

        for (uint16_t loopIdx = 0; loopIdx < loops; loopIdx++) {
            uint32_t colOffset = loopIdx * vlSize;
            uint32_t colNumInLoop = vlSize > actualColumnNum - colOffset ? (actualColumnNum - colOffset) : vlSize;

            uint32_t topLeftDotColIdx = topLeftDotColIdxIn + colOffset;
            uint32_t topRightDotColIdx = topLeftDotColIdx + colNumInLoop - 1;
            uint32_t bottomLeftDotColIdx = topLeftDotColIdx;

            if (bottomLeftDotRowIdx < bottomLeftDotColIdx) { // 全0
                maskRowIdx = actualRowNum;
                maskColIdx = 0;
            } else {
                if (topRightDotRowIdx >= topRightDotColIdx) { // 全1
                    maskRowIdx = 0;
                    maskColIdx = colNumInLoop - 1;
                } else {
                    if (topLeftDotRowIdx < topLeftDotColIdx) { // row 被切分
                        maskRowIdx = topLeftDotColIdx - topLeftDotRowIdx;
                        maskColIdx = 0;
                    } else {
                        maskRowIdx = 0;
                        maskColIdx = topLeftDotRowIdx - topLeftDotColIdx;
                    }
                }
            }

            firstRowValidNum = maskColIdx + 1;

            srcRowStart = srcUb + loopIdx * vlSize;
            dstRowStart = dstUb + loopIdx * dstColumnStrideInterFractal * vlDataBlockNum;
            FillZero_vf_b32tob16_nd2nz_1vl(dstRowStart, srcRowStart, maskRowIdx, dstC0Num, srcRowStride, scale);

            srcRowStart = srcUb + loopIdx * vlSize + maskRowIdx * srcRowStride;
            dstRowStart =
                dstUb + loopIdx * dstColumnStrideInterFractal * vlDataBlockNum + maskRowIdx * dstRowStrideInnerFractal;
            SiluScale_vf_b32tob16_nd2nz_1vl(
                dstRowStart, srcRowStart, actualRowNum - maskRowIdx, dstC0Num, srcRowStride, scale, firstRowValidNum);
        }
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const& tensorUbDst, TensorSrc const& tensorUbSrc, MatrixCoord const& actualTileShape, float scale,
        MatrixCoord topLeftDotCoord)
    {
        __ubuf__ ElementDst* ubDstPtr = (__ubuf__ ElementDst*)tensorUbDst.data().GetPhyAddr();
        __ubuf__ ElementSrc* ubSrcPtr = (__ubuf__ ElementSrc*)tensorUbSrc.data().GetPhyAddr();

        uint32_t actualRowNum = actualTileShape.row();
        uint32_t actualColumnNum = actualTileShape.column();

        uint32_t dstStride00 = tla::get<0, 0>(tensorUbDst.stride()); // dst row stride in inner fractal
        uint32_t dstStride11 = tla::get<1, 1>(tensorUbDst.stride()); // dst column stride in inter fractal
        uint32_t dstEleNumInC0 = tla::get<1, 0>(tensorUbDst.shape());
        uint32_t dstC0Num = tla::get<0, 0>(tensorUbDst.shape()) * tla::get<0, 1>(tensorUbDst.shape());

        uint32_t srcRowStride = tla::get<0>(tensorUbSrc.stride());

        static constexpr uint16_t vlSize = static_cast<uint16_t>(AscendC::GetVecLen() / sizeof(ElementSrc));
        uint32_t vlDataBlockNum = vlSize / dstEleNumInC0;
        uint16_t loops = AscendC::CeilDivision(actualColumnNum, vlSize);

        uint32_t topLeftDotRowIdx = topLeftDotCoord.row();
        uint32_t topLeftDotColIdx = topLeftDotCoord.column();

        SiluScaleVf_with_mask(
            ubDstPtr, ubSrcPtr, actualRowNum, actualColumnNum, topLeftDotRowIdx, topLeftDotColIdx, dstEleNumInC0,
            dstC0Num, dstStride00, dstStride11, srcRowStride, scale);
    }
};
#endif

} // namespace Catlass::Epilogue::Tile

#endif
