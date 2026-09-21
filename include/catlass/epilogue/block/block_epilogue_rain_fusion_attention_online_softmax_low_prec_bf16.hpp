/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RAIN_FUSION_ATTENTION_SOFTMAX_LOW_PREC_BF16_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RAIN_FUSION_ATTENTION_SOFTMAX_LOW_PREC_BF16_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

using namespace Catlass::Arch;

namespace Catlass::Epilogue::Block {

enum class KvBaseTileRegSplitStagesBf16
{
    ONE,
    TWO
};

template <class OutputType_, class LayoutS_>
class BlockEpilogue<EpilogueAtlasA5OnlineSoftmax, OutputType_, Gemm::GemmType<bfloat16_t, LayoutS_>> {
public:
    using DispatchPolicy = EpilogueAtlasA5OnlineSoftmax;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementOutput = typename OutputType_::Element; // q/k/v dtype
    using ElementInput = bfloat16_t;

    using LayoutOutput = typename OutputType_::Layout; // layout::zN
    using LayoutInput = LayoutS_;                      // layout::RowMajor

    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 256;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 16384;
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;

    static constexpr uint32_t REP_SIZE_B16 = 128;
    static constexpr uint32_t FLOAT_REP_SIZE = 64;
    static constexpr uint32_t BLOCK_REP_SIZE = 8;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t SM_VREG_SIZE = 256 / sizeof(ElementInput);

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource, float scaleValue_)
    {
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 2 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t LM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE; // 224K
        constexpr uint32_t GM_UB_TENSOR_OFFSET = LM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t LL_UB_TENSOR_OFFSET = DM_UB_TENSOR_OFFSET + 3 * 64 * sizeof(float);
        constexpr uint32_t GL_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET + 64 * sizeof(float);

        subBlockIdx_ = AscendC::GetSubBlockIdx();
        scaleValue = AscendC::ToBfloat16(scaleValue_);

        lsUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        lmUbFloatTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        llUbFloatTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
    }

    __aicore__ inline ~BlockEpilogue()
    {}

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyPUbToPL1(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint32_t m)
    {
        const uint32_t blockCount = tla::get<1, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());

        AscendC::DataCopyParams repeatParams;

        uint32_t elementNumPerC0 = ELE_NUM_PER_C0;
        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = m;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / elementNumPerC0 - m;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / elementNumPerC0 - m;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag& crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag& crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <class TensorP>
    __aicore__ inline void operator()(
        TensorP& l1PTensorTla, GemmCoord actualBlockShape, uint32_t isFirstKvSTile, uint32_t ubSBufId,
        uint32_t l1PBufId, Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag)
    {
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mAlignedHalf = RoundUp(actualBlockShape.m(), 8) / subBlockNum;
        uint32_t m = actualBlockShape.m() < mAlignedHalf ? actualBlockShape.m() : mAlignedHalf;
        m = subBlockIdx_ == 0 ? m : actualBlockShape.m() - m;
        if (m == 0) {
            WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            return;
        }

        uint32_t n = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        uint32_t blockStride = mRound;
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        constexpr int16_t vlFloatSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(float));
        int16_t mFullVecCnt = AscendC::CeilDivision(m, vlFloatSize) - 1;
        uint32_t tailM = (m - 1) % vlFloatSize + 1;
        uint32_t tailN = (n - 1) % vlSize + 1;

        __ubuf__ ElementOutput* pAddr = (__ubuf__ ElementOutput*)lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementInput* sAddr = (__ubuf__ ElementInput*)lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ float* lastMaxAddr = (__ubuf__ float*)gmUbTensor.GetPhyAddr();
        __ubuf__ float* lastSumAddr = (__ubuf__ float*)glUbTensor.GetPhyAddr();
        __ubuf__ float* nowMaxAddr = (__ubuf__ float*)lmUbFloatTensor.GetPhyAddr();
        __ubuf__ float* nowSumAddr = (__ubuf__ float*)llUbFloatTensor.GetPhyAddr();
        __ubuf__ float* expMaxUbAddr = (__ubuf__ float*)dmUbTensor[l1PBufId * SM_ROW_MAX_ELEM_NUM].GetPhyAddr();

        // wait QK Fixpipe finish
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        if (isFirstKvSTile) {
            nowMaxAddr = lastMaxAddr;
            nowSumAddr = lastSumAddr;
        }
        uint32_t nRegStages = CeilDiv(n, SM_VREG_SIZE);
        if (nRegStages == 1) {
            ComputeScaleAndMax<KvBaseTileRegSplitStagesBf16::ONE>(sAddr, nowMaxAddr, m, tailN, scaleValue, nRound);
        } else if (nRegStages == 2) {
            ComputeScaleAndMax<KvBaseTileRegSplitStagesBf16::TWO>(sAddr, nowMaxAddr, m, tailN, scaleValue, nRound);
        }
        if (!isFirstKvSTile) {
            UpdateMax(nowMaxAddr, lastMaxAddr, mFullVecCnt, tailM);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId);
        uint32_t tailNOdd = tailN / 2;
        uint32_t tailNEven = tailNOdd + tailN % 2;
        if (nRegStages == 1) {
            ComputeExpSubSumB16<KvBaseTileRegSplitStagesBf16::ONE>(
                pAddr, sAddr, nowMaxAddr, nowSumAddr, m, tailN, blockStride, nRound, tailNOdd, tailNEven);
        } else if (nRegStages == 2) {
            ComputeExpSubSumB16<KvBaseTileRegSplitStagesBf16::TWO>(
                pAddr, sAddr, nowMaxAddr, nowSumAddr, m, tailN, blockStride, nRound, tailNOdd, tailNEven);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM], ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile =
            GetTile(l1PTensorTla, tla::MakeCoord(subBlockIdx_ * mAlignedHalf, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);

        CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, m);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);

        if (!isFirstKvSTile) {
            UpdateExpSumAndExpMax(lastSumAddr, expMaxUbAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mFullVecCnt, tailM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    ElementInput scaleValue;
    AscendC::LocalTensor<ElementInput> lsUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> lmUbFloatTensor;
    AscendC::LocalTensor<float> llUbFloatTensor;
    uint32_t subBlockIdx_;
    static constexpr ElementInput MIN_VALUE = __builtin_bit_cast(ElementInput, static_cast<uint16_t>(0xFF7F));

    template <KvBaseTileRegSplitStagesBf16 kvBaseTileRegSplitStages>
    CATLASS_DEVICE_SIMD_VF void ComputeScaleAndMax(
        __ubuf__ ElementInput* srcUb, __ubuf__ float* newMaxUb, uint16_t m, uint32_t tailN, ElementInput dScale,
        uint16_t s2BaseSize)
    {
        static_assert(
            kvBaseTileRegSplitStages == KvBaseTileRegSplitStagesBf16::ONE ||
                kvBaseTileRegSplitStages == KvBaseTileRegSplitStagesBf16::TWO,
            "ComputeScaleAndMax only supports ONE Or TWO stages.");
    }

    template <>
    CATLASS_DEVICE_SIMD_VF void ComputeScaleAndMax<KvBaseTileRegSplitStagesBf16::ONE>(
        __ubuf__ ElementInput* srcUb, __ubuf__ float* newMaxUb, uint16_t m, uint32_t tailN, ElementInput dScale,
        uint16_t s2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementInput> minVreg;
        RegTensor<ElementInput> srcVreg;
        RegTensor<ElementInput> scaleVreg;
        RegTensor<float> maxFloatVreg0;
        RegTensor<float> maxFloatVreg1;
        RegTensor<float> maxTmpFloatVreg;
        RegTensor<float> maxTmpFloatVreg0;
        RegTensor<float> maxTmpFloatVreg1;

        UnalignReg maxUreg;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);

        Duplicate(minVreg, MIN_VALUE);
        Duplicate(scaleVreg, dScale);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg, srcUb + i * s2BaseSize);
            Mul(srcVreg, srcVreg, scaleVreg, pregFull);
            Select(srcVreg, srcVreg, minVreg, pregTailN);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(srcUb + i * s2BaseSize, srcVreg, pregTailN);

            Cast<float, ElementInput, castTraitZero>(maxFloatVreg0, srcVreg, pregFull);
            Cast<float, ElementInput, castTraitOne>(maxFloatVreg1, srcVreg, pregFull);
            Max<float, MaskMergeMode::MERGING>(maxFloatVreg0, maxFloatVreg0, maxFloatVreg1, pregTailN);
            ReduceMax(maxTmpFloatVreg, maxFloatVreg0, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpFloatVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }

    template <>
    CATLASS_DEVICE_SIMD_VF void ComputeScaleAndMax<KvBaseTileRegSplitStagesBf16::TWO>(
        __ubuf__ ElementInput* srcUb, __ubuf__ float* newMaxUb, uint16_t m, uint32_t tailN, ElementInput dScale,
        uint16_t s2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementInput> srcVreg0;
        RegTensor<ElementInput> srcVreg1;
        RegTensor<ElementInput> scaleVreg;
        RegTensor<float> maxFloatVreg0;
        RegTensor<float> maxFloatVreg1;
        RegTensor<float> maxTmpFloatVreg;
        RegTensor<float> maxTmpFloatVreg0;
        RegTensor<float> maxTmpFloatVreg1;

        UnalignReg maxUreg;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);

        Duplicate(scaleVreg, dScale);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg0, srcUb + i * s2BaseSize);
            LoadAlign(srcVreg1, srcUb + i * s2BaseSize + REP_SIZE_B16);
            Mul(srcVreg0, srcVreg0, scaleVreg, pregFull);
            Mul(srcVreg1, srcVreg1, scaleVreg, pregFull);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(srcUb + i * s2BaseSize, srcVreg0, pregFull);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(
                srcUb + i * s2BaseSize + REP_SIZE_B16, srcVreg1, pregTailN);

            Max<ElementInput, MaskMergeMode::MERGING>(srcVreg0, srcVreg0, srcVreg1, pregTailN);
            Cast<float, ElementInput, castTraitZero>(maxFloatVreg0, srcVreg0, pregFull);
            Cast<float, ElementInput, castTraitOne>(maxFloatVreg1, srcVreg0, pregFull);
            Max<float, MaskMergeMode::MERGING>(maxFloatVreg0, maxFloatVreg0, maxFloatVreg1, pregTailN);
            ReduceMax(maxTmpFloatVreg, maxFloatVreg0, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpFloatVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }

    CATLASS_DEVICE_SIMD_VF void UpdateMax(
        __ubuf__ float* nowMaxUb, __ubuf__ float* lastMaxUb, uint16_t mFullVecCnt, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> nowMaxVreg;
        RegTensor<float> lastMaxFloatVreg;
        RegTensor<float> maxVreg;

        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregFloatTailM = UpdateMask<float>(tailM);

        for (uint16_t i = 0; i < mFullVecCnt; ++i) {
            LoadAlign(lastMaxFloatVreg, lastMaxUb + i * FLOAT_REP_SIZE);
            LoadAlign(nowMaxVreg, nowMaxUb + i * FLOAT_REP_SIZE);
            Max(maxVreg, nowMaxVreg, lastMaxFloatVreg, pregFloatFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(nowMaxUb + i * FLOAT_REP_SIZE, maxVreg, pregFloatFull);
        }
        LoadAlign(lastMaxFloatVreg, lastMaxUb + mFullVecCnt * FLOAT_REP_SIZE);
        LoadAlign(nowMaxVreg, nowMaxUb + mFullVecCnt * FLOAT_REP_SIZE);
        Max(maxVreg, nowMaxVreg, lastMaxFloatVreg, pregFloatFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(nowMaxUb + mFullVecCnt * FLOAT_REP_SIZE, maxVreg, pregFloatTailM);
    }

    template <KvBaseTileRegSplitStagesBf16 kvBaseTileRegSplitStages>
    CATLASS_DEVICE_SIMD_VF void ComputeExpSubSumB16(
        __ubuf__ ElementOutput* expUb, __ubuf__ ElementInput* srcUb, __ubuf__ float* nowMaxUb, __ubuf__ float* expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t s2BaseSize, uint32_t tailNOdd, uint32_t tailNEven)
    {
        static_assert(
            kvBaseTileRegSplitStages == KvBaseTileRegSplitStagesBf16::ONE ||
                kvBaseTileRegSplitStages == KvBaseTileRegSplitStagesBf16::TWO,
            "ComputeExpSubSumB16 only supports ONE Or TWO stages.");
    }

    template <>
    CATLASS_DEVICE_SIMD_VF void ComputeExpSubSumB16<KvBaseTileRegSplitStagesBf16::ONE>(
        __ubuf__ ElementOutput* expUb, __ubuf__ ElementInput* srcUb, __ubuf__ float* nowMaxUb, __ubuf__ float* expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t s2BaseSize, uint32_t tailNOdd, uint32_t tailNEven)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitZeroRound = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };
        constexpr static CastTrait castTraitOneRound = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        RegTensor<ElementInput> srcVreg;
        RegTensor<float> expFloatVreg0;
        RegTensor<float> expFloatVreg1;
        RegTensor<float> expSumVreg;
        RegTensor<float> maxVreg;

        RegTensor<float> expDstFloatVreg0;
        RegTensor<float> expDstFloatVreg1;
        RegTensor<ElementInput> expDstVreg;
        RegTensor<ElementInput> expDstVreg0;
        RegTensor<ElementInput> expDstVreg1;

        UnalignReg expSumUreg;

        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);
        MaskReg pregtailNOdd = UpdateMask<float>(tailNOdd);
        MaskReg pregtailNEven = UpdateMask<float>(tailNEven);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(maxVreg, nowMaxUb + i);
            Duplicate(expSumVreg, 0);
            LoadAlign(srcVreg, srcUb + i * s2BaseSize);
            Cast<float, ElementInput, castTraitZero>(expFloatVreg0, srcVreg, pregFull);
            Cast<float, ElementInput, castTraitOne>(expFloatVreg1, srcVreg, pregFull);
            FusedExpSub(expDstFloatVreg0, expFloatVreg0, maxVreg, pregtailNEven);
            FusedExpSub(expDstFloatVreg1, expFloatVreg1, maxVreg, pregtailNOdd);

            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg0, pregtailNEven);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg1, pregtailNOdd);

            Cast<ElementInput, float, castTraitZeroRound>(expDstVreg0, expDstFloatVreg0, pregFloatFull);
            Cast<ElementInput, float, castTraitOneRound>(expDstVreg1, expDstFloatVreg1, pregFloatFull);
            Or((RegTensor<uint16_t>&)expDstVreg, (RegTensor<uint16_t>&)expDstVreg0, (RegTensor<uint16_t>&)expDstVreg1,
               pregFull);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0, expDstVreg, blockStride, pregTailN);

            ReduceSum(expSumVreg, expSumVreg, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <>
    CATLASS_DEVICE_SIMD_VF void ComputeExpSubSumB16<KvBaseTileRegSplitStagesBf16::TWO>(
        __ubuf__ ElementOutput* expUb, __ubuf__ ElementInput* srcUb, __ubuf__ float* nowMaxUb, __ubuf__ float* expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t s2BaseSize, uint32_t tailNOdd, uint32_t tailNEven)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitZeroRound = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };
        constexpr static CastTrait castTraitOneRound = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        RegTensor<ElementInput> srcVreg0;
        RegTensor<ElementInput> srcVreg1;
        RegTensor<float> expFloatVreg0;
        RegTensor<float> expFloatVreg1;
        RegTensor<float> expFloatVreg2;
        RegTensor<float> expFloatVreg3;
        RegTensor<float> expSumVreg;
        RegTensor<float> maxVreg;

        RegTensor<float> expDstFloatVreg0;
        RegTensor<float> expDstFloatVreg1;
        RegTensor<float> expDstFloatVreg2;
        RegTensor<float> expDstFloatVreg3;
        RegTensor<ElementInput> expOutVreg0;
        RegTensor<ElementInput> expOutVreg1;
        RegTensor<ElementInput> expDstVreg0;
        RegTensor<ElementInput> expDstVreg1;
        RegTensor<ElementInput> expDstVreg2;
        RegTensor<ElementInput> expDstVreg3;

        UnalignReg expSumUreg;

        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);
        MaskReg pregtailNOdd = UpdateMask<float>(tailNOdd);
        MaskReg pregtailNEven = UpdateMask<float>(tailNEven);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(maxVreg, nowMaxUb + i);
            LoadAlign(srcVreg0, srcUb + i * s2BaseSize);
            LoadAlign(srcVreg1, srcUb + i * s2BaseSize + REP_SIZE_B16);
            Cast<float, ElementInput, castTraitZero>(expFloatVreg0, srcVreg0, pregFull);
            Cast<float, ElementInput, castTraitOne>(expFloatVreg1, srcVreg0, pregFull);
            Cast<float, ElementInput, castTraitZero>(expFloatVreg2, srcVreg1, pregFull);
            Cast<float, ElementInput, castTraitOne>(expFloatVreg3, srcVreg1, pregFull);
            FusedExpSub(expDstFloatVreg0, expFloatVreg0, maxVreg, pregFloatFull);
            FusedExpSub(expDstFloatVreg1, expFloatVreg1, maxVreg, pregFloatFull);
            FusedExpSub(expDstFloatVreg2, expFloatVreg2, maxVreg, pregtailNEven);
            FusedExpSub(expDstFloatVreg3, expFloatVreg3, maxVreg, pregtailNOdd);

            Add(expSumVreg, expDstFloatVreg0, expDstFloatVreg1, pregFloatFull);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg2, pregtailNEven);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg3, pregtailNOdd);

            Cast<ElementInput, float, castTraitZeroRound>(expDstVreg0, expDstFloatVreg0, pregFloatFull);
            Cast<ElementInput, float, castTraitOneRound>(expDstVreg1, expDstFloatVreg1, pregFloatFull);
            Cast<ElementInput, float, castTraitZeroRound>(expDstVreg2, expDstFloatVreg2, pregFloatFull);
            Cast<ElementInput, float, castTraitOneRound>(expDstVreg3, expDstFloatVreg3, pregFloatFull);
            Or((RegTensor<uint16_t>&)expOutVreg0, (RegTensor<uint16_t>&)expDstVreg0, (RegTensor<uint16_t>&)expDstVreg1,
               pregFull);
            Or((RegTensor<uint16_t>&)expOutVreg1, (RegTensor<uint16_t>&)expDstVreg2, (RegTensor<uint16_t>&)expDstVreg3,
               pregFull);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0, expOutVreg0, blockStride, pregFull);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0 + blockStride * ELE_NUM_PER_C0 * BLOCK_REP_SIZE, expOutVreg1, blockStride,
                pregTailN);

            ReduceSum(expSumVreg, expSumVreg, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    CATLASS_DEVICE_SIMD_VF void UpdateExpSumAndExpMax(
        __ubuf__ float* sumUb, __ubuf__ float* expMaxUb, __ubuf__ float* maxUb, __ubuf__ float* expSumUb,
        __ubuf__ float* nowMaxUb, uint16_t mFullVecCnt, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> nowMaxFloatVreg;
        RegTensor<float> lastMaxVreg;
        RegTensor<float> expMaxVreg;
        RegTensor<float> lastExpSumVreg;
        RegTensor<float> expSumFloatVreg;
        RegTensor<float> updateExpSumVreg;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<float>(tailM);

        for (int16_t i = 0; i < mFullVecCnt; ++i) {
            LoadAlign(lastMaxVreg, maxUb + i * FLOAT_REP_SIZE);
            LoadAlign(nowMaxFloatVreg, nowMaxUb + i * FLOAT_REP_SIZE);
            FusedExpSub(expMaxVreg, lastMaxVreg, nowMaxFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + i * FLOAT_REP_SIZE, nowMaxFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb + i * FLOAT_REP_SIZE, expMaxVreg, pregFull);

            LoadAlign(lastExpSumVreg, sumUb + i * FLOAT_REP_SIZE);
            LoadAlign(expSumFloatVreg, expSumUb + i * FLOAT_REP_SIZE);
            Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregFull);
            Add(updateExpSumVreg, updateExpSumVreg, expSumFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + i * FLOAT_REP_SIZE, updateExpSumVreg, pregFull);
        }
        LoadAlign(lastMaxVreg, maxUb + mFullVecCnt * FLOAT_REP_SIZE);
        LoadAlign(nowMaxFloatVreg, nowMaxUb + mFullVecCnt * FLOAT_REP_SIZE);
        FusedExpSub(expMaxVreg, lastMaxVreg, nowMaxFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + mFullVecCnt * FLOAT_REP_SIZE, nowMaxFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb + mFullVecCnt * FLOAT_REP_SIZE, expMaxVreg, pregTailM);

        LoadAlign(lastExpSumVreg, sumUb + mFullVecCnt * FLOAT_REP_SIZE);
        LoadAlign(expSumFloatVreg, expSumUb + mFullVecCnt * FLOAT_REP_SIZE);
        Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregTailM);
        Add(updateExpSumVreg, updateExpSumVreg, expSumFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + mFullVecCnt * FLOAT_REP_SIZE, updateExpSumVreg, pregTailM);
    }
};

} // namespace Catlass::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RAIN_FUSION_ATTENTION_SOFTMAX_LOW_PREC_BF16_HPP
