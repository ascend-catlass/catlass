/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RAIN_FUSION_ATTENTION_SOFTMAX_LOW_PREC_FP16_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RAIN_FUSION_ATTENTION_SOFTMAX_LOW_PREC_FP16_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

using namespace Catlass::Arch;

namespace Catlass::Epilogue::Block {

enum class KvBaseTileRegSplitStages
{
    ONE,
    TWO
};

template <class OutputType_, class LayoutS_>
class BlockEpilogue<EpilogueAtlasA5OnlineSoftmax, OutputType_, Gemm::GemmType<half, LayoutS_>> {
public:
    using DispatchPolicy = EpilogueAtlasA5OnlineSoftmax;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementOutput = typename OutputType_::Element; // q/k/v dtype
    using ElementInput = half;

    using LayoutOutput = typename OutputType_::Layout; // layout::zN
    using LayoutInput = LayoutS_;                      // layout::RowMajor

    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 256;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 16384;
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;

    static constexpr ElementInput MIN_VALUE = -65504.0f;
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
        scaleValue = static_cast<ElementInput>(scaleValue_);

        lsUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LL_UB_TENSOR_OFFSET);
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
        int16_t mFullVecCnt = AscendC::CeilDivision(m, vlSize) - 1;
        uint32_t tailM = (m - 1) % vlSize + 1;
        uint32_t tailN = (n - 1) % vlSize + 1;

        __ubuf__ ElementOutput* pAddr = (__ubuf__ ElementOutput*)lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementInput* sAddr = (__ubuf__ ElementInput*)lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ float* lastMaxAddr = (__ubuf__ float*)gmUbTensor.GetPhyAddr();
        __ubuf__ float* lastSumAddr = (__ubuf__ float*)glUbTensor.GetPhyAddr();
        __ubuf__ ElementInput* nowMaxAddr = (__ubuf__ ElementInput*)lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput* nowSumAddr = (__ubuf__ ElementInput*)llUbTensor.GetPhyAddr();
        __ubuf__ float* expMaxUbAddr = (__ubuf__ float*)dmUbTensor[l1PBufId * SM_ROW_MAX_ELEM_NUM].GetPhyAddr();

        // wait QK Fixpipe finish
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        uint32_t nRegStages = CeilDiv(n, SM_VREG_SIZE);
        if (nRegStages == 1) {
            ComputeScaleAndMax<KvBaseTileRegSplitStages::ONE>(sAddr, nowMaxAddr, m, tailN, scaleValue, nRound);
        } else if (nRegStages == 2) {
            ComputeScaleAndMax<KvBaseTileRegSplitStages::TWO>(sAddr, nowMaxAddr, m, tailN, scaleValue, nRound);
        }
        if (!isFirstKvSTile) {
            UpdateMax(nowMaxAddr, lastMaxAddr, tailM);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId);
        if (nRegStages == 1) {
            ComputeExpSubSumB16<KvBaseTileRegSplitStages::ONE>(
                pAddr, sAddr, nowMaxAddr, nowSumAddr, m, tailN, blockStride, nRound);
        } else if (nRegStages == 2) {
            ComputeExpSubSumB16<KvBaseTileRegSplitStages::TWO>(
                pAddr, sAddr, nowMaxAddr, nowSumAddr, m, tailN, blockStride, nRound);
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

        mFullVecCnt = AscendC::CeilDivision(m, FLOAT_REP_SIZE) - 1;
        tailM = (m - 1) % FLOAT_REP_SIZE + 1;
        if (isFirstKvSTile) {
            CastExpSumAndExpMax<ElementInput>(lastSumAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mFullVecCnt, tailM);
        } else {
            UpdateExpSumAndExpMax<ElementInput>(
                lastSumAddr, expMaxUbAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mFullVecCnt, tailM);
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
    AscendC::LocalTensor<ElementInput> lmUbTensor;
    AscendC::LocalTensor<ElementInput> llUbTensor;
    uint32_t subBlockIdx_;

    template <KvBaseTileRegSplitStages kvBaseTileRegSplitStages>
    CATLASS_DEVICE_SIMD_VF void ComputeScaleAndMax(
        __ubuf__ ElementInput* srcUb, __ubuf__ ElementInput* newMaxUb, uint16_t m, uint32_t tailN, ElementInput dScale,
        uint16_t s2BaseSize)
    {
        static_assert(
            kvBaseTileRegSplitStages == KvBaseTileRegSplitStages::ONE ||
                kvBaseTileRegSplitStages == KvBaseTileRegSplitStages::TWO,
            "ComputeScaleAndMax only supports ONE Or TWO stages.");
    }

    template <>
    CATLASS_DEVICE_SIMD_VF void ComputeScaleAndMax<KvBaseTileRegSplitStages::ONE>(
        __ubuf__ ElementInput* srcUb, __ubuf__ ElementInput* newMaxUb, uint16_t m, uint32_t tailN, ElementInput dScale,
        uint16_t s2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementInput> minVreg;
        RegTensor<ElementInput> srcVreg;
        RegTensor<ElementInput> maxTmpVreg;
        UnalignReg maxUreg;
        MaskReg pregCompare;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg, srcUb + i * s2BaseSize);
            Muls(srcVreg, srcVreg, dScale, pregFull);
            Select(srcVreg, srcVreg, minVreg, pregTailN);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(srcUb + i * s2BaseSize, srcVreg, pregTailN);
            ReduceMax(maxTmpVreg, srcVreg, pregFull);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }

    template <>
    CATLASS_DEVICE_SIMD_VF void ComputeScaleAndMax<KvBaseTileRegSplitStages::TWO>(
        __ubuf__ ElementInput* srcUb, __ubuf__ ElementInput* newMaxUb, uint16_t m, uint32_t tailN, ElementInput dScale,
        uint16_t s2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementInput> srcVreg0;
        RegTensor<ElementInput> srcVreg1;
        RegTensor<ElementInput> maxTmpVreg;
        UnalignReg maxUreg;
        MaskReg pregCompare;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg0, srcUb + i * s2BaseSize);
            LoadAlign(srcVreg1, srcUb + i * s2BaseSize + REP_SIZE_B16);
            Muls(srcVreg0, srcVreg0, dScale, pregFull);
            Muls(srcVreg1, srcVreg1, dScale, pregFull);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(srcUb + i * s2BaseSize, srcVreg0, pregFull);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(
                srcUb + i * s2BaseSize + REP_SIZE_B16, srcVreg1, pregTailN);

            Max<ElementInput, MaskMergeMode::MERGING>(srcVreg0, srcVreg0, srcVreg1, pregTailN);
            ReduceMax(maxTmpVreg, srcVreg0, pregFull);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }

    // SM_ROW_MAX_ELEM_NUM = 64: mFullVecCnt = 0, tailM = m
    CATLASS_DEVICE_SIMD_VF void UpdateMax(__ubuf__ ElementInput* nowMaxUb, __ubuf__ float* lastMaxUb, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZeroRound = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        RegTensor<ElementInput> nowMaxVreg;
        RegTensor<float> lastMaxFloatVreg;
        RegTensor<ElementInput> maxVreg;
        RegTensor<ElementInput> lastMaxVreg;
        RegTensor<ElementInput> lastMaxVreg0;
        RegTensor<ElementInput> lastMaxVreg1;

        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<ElementInput>(tailM);

        LoadAlign(lastMaxFloatVreg, lastMaxUb);
        Cast<ElementInput, float, castTraitZeroRound>(lastMaxVreg, lastMaxFloatVreg, pregFloatFull);
        DeInterleave(lastMaxVreg0, lastMaxVreg1, lastMaxVreg, lastMaxVreg);
        LoadAlign(nowMaxVreg, nowMaxUb);
        Max(maxVreg, nowMaxVreg, lastMaxVreg0, pregTailM);
        StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(nowMaxUb, maxVreg, pregTailM);
    }

    template <KvBaseTileRegSplitStages kvBaseTileRegSplitStages>
    CATLASS_DEVICE_SIMD_VF void ComputeExpSubSumB16(
        __ubuf__ ElementOutput* expUb, __ubuf__ ElementInput* srcUb, __ubuf__ ElementInput* nowMaxUb,
        __ubuf__ ElementInput* expSumUb, uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t s2BaseSize)
    {
        static_assert(
            kvBaseTileRegSplitStages == KvBaseTileRegSplitStages::ONE ||
                kvBaseTileRegSplitStages == KvBaseTileRegSplitStages::TWO,
            "ComputeExpSubSumB16 only supports ONE Or TWO stages.");
    }

    template <>
    CATLASS_DEVICE_SIMD_VF void ComputeExpSubSumB16<KvBaseTileRegSplitStages::ONE>(
        __ubuf__ ElementOutput* expUb, __ubuf__ ElementInput* srcUb, __ubuf__ ElementInput* nowMaxUb,
        __ubuf__ ElementInput* expSumUb, uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t s2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementInput> srcVreg;
        RegTensor<ElementInput> expSumVreg;
        RegTensor<ElementInput> maxVreg;
        RegTensor<ElementOutput> expDstVreg;

        UnalignReg expSumUreg;
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<ElementInput, LoadDist::DIST_BRC_B16>(maxVreg, nowMaxUb + i);
            LoadAlign(srcVreg, srcUb + i * s2BaseSize);
            Sub(expDstVreg, srcVreg, maxVreg, pregTailN);
            Exp(expDstVreg, expDstVreg, pregTailN);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0, expDstVreg, blockStride, pregTailN);

            ReduceSum(expSumVreg, expDstVreg, pregTailN);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <>
    CATLASS_DEVICE_SIMD_VF void ComputeExpSubSumB16<KvBaseTileRegSplitStages::TWO>(
        __ubuf__ ElementOutput* expUb, __ubuf__ ElementInput* srcUb, __ubuf__ ElementInput* nowMaxUb,
        __ubuf__ ElementInput* expSumUb, uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t s2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementInput> srcVreg0;
        RegTensor<ElementInput> srcVreg1;
        RegTensor<ElementInput> expSumVreg;
        RegTensor<ElementInput> maxVreg;
        RegTensor<ElementOutput> expDstVreg0;
        RegTensor<ElementOutput> expDstVreg1;

        UnalignReg expSumUreg;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<ElementInput, LoadDist::DIST_BRC_B16>(maxVreg, nowMaxUb + i);
            LoadAlign(srcVreg0, srcUb + i * s2BaseSize);
            LoadAlign(srcVreg1, srcUb + i * s2BaseSize + REP_SIZE_B16);
            Sub(expDstVreg0, srcVreg0, maxVreg, pregFull);
            Sub(expDstVreg1, srcVreg1, maxVreg, pregTailN);
            Exp(expDstVreg0, expDstVreg0, pregFull);
            Exp(expDstVreg1, expDstVreg1, pregTailN);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0, expDstVreg0, blockStride, pregFull);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0 + blockStride * ELE_NUM_PER_C0 * BLOCK_REP_SIZE, expDstVreg1, blockStride,
                pregTailN);

            Add<ElementOutput, MaskMergeMode::MERGING>(expDstVreg0, expDstVreg1, expDstVreg0, pregTailN);
            ReduceSum(expSumVreg, expDstVreg0, pregFull);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <typename ElementInput>
    CATLASS_DEVICE_SIMD_VF void CastExpSumAndExpMax(
        __ubuf__ float* sumUb, __ubuf__ float* maxUb, __ubuf__ ElementInput* expSumUb, __ubuf__ ElementInput* nowMaxUb,
        uint16_t mFullVecCnt, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementInput> nowMaxVreg0;
        RegTensor<ElementInput> nowMaxVreg1;
        RegTensor<ElementInput> nowMaxTmpVreg;
        RegTensor<float> nowMaxFloatVreg;
        RegTensor<ElementInput> expSumVreg0;
        RegTensor<ElementInput> expSumVreg1;
        RegTensor<ElementInput> expSumTmpVreg;
        RegTensor<float> expSumFloatVreg;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<float>(tailM);
        MaskReg pregB16Full = CreateMask<ElementInput, MaskPattern::ALL>();

        for (int16_t i = 0; i < mFullVecCnt; ++i) {
            LoadAlign(nowMaxTmpVreg, nowMaxUb + i * FLOAT_REP_SIZE);
            Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
            Cast<float, ElementInput, castTraitZero>(nowMaxFloatVreg, nowMaxVreg0, pregB16Full);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + i * FLOAT_REP_SIZE, nowMaxFloatVreg, pregFull);

            LoadAlign(expSumTmpVreg, expSumUb + i * FLOAT_REP_SIZE);
            Interleave(expSumVreg0, expSumVreg1, expSumTmpVreg, expSumTmpVreg);
            Cast<float, ElementInput, castTraitZero>(expSumFloatVreg, expSumVreg0, pregB16Full);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + i * FLOAT_REP_SIZE, expSumFloatVreg, pregFull);
        }
        LoadAlign(nowMaxTmpVreg, nowMaxUb + mFullVecCnt * FLOAT_REP_SIZE);
        Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
        Cast<float, ElementInput, castTraitZero>(nowMaxFloatVreg, nowMaxVreg0, pregB16Full);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + mFullVecCnt * FLOAT_REP_SIZE, nowMaxFloatVreg, pregTailM);

        LoadAlign(expSumTmpVreg, expSumUb + mFullVecCnt * FLOAT_REP_SIZE);
        Interleave(expSumVreg0, expSumVreg1, expSumTmpVreg, expSumTmpVreg);
        Cast<float, ElementInput, castTraitZero>(expSumFloatVreg, expSumVreg0, pregB16Full);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + mFullVecCnt * FLOAT_REP_SIZE, expSumFloatVreg, pregTailM);
    }

    template <typename ElementInput>
    CATLASS_DEVICE_SIMD_VF void UpdateExpSumAndExpMax(
        __ubuf__ float* sumUb, __ubuf__ float* expMaxUb, __ubuf__ float* maxUb, __ubuf__ ElementInput* expSumUb,
        __ubuf__ ElementInput* nowMaxUb, uint16_t mFullVecCnt, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementInput> nowMaxVreg0;
        RegTensor<ElementInput> nowMaxVreg1;
        RegTensor<ElementInput> nowMaxTmpVreg;
        RegTensor<float> nowMaxFloatVreg;
        RegTensor<float> lastMaxVreg;
        RegTensor<float> expMaxVreg;
        RegTensor<float> lastExpSumVreg;
        RegTensor<ElementInput> expSumVreg0;
        RegTensor<ElementInput> expSumVreg1;
        RegTensor<ElementInput> expSumTmpVreg;
        RegTensor<float> expSumFloatVreg;
        RegTensor<float> updateExpSumVreg;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<float>(tailM);
        MaskReg pregB16Full = CreateMask<ElementInput, MaskPattern::ALL>();

        for (int16_t i = 0; i < mFullVecCnt; ++i) {
            LoadAlign(lastMaxVreg, maxUb + i * FLOAT_REP_SIZE);
            LoadAlign(nowMaxTmpVreg, nowMaxUb + i * FLOAT_REP_SIZE);
            Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
            Cast<float, ElementInput, castTraitZero>(nowMaxFloatVreg, nowMaxVreg0, pregB16Full);
            FusedExpSub(expMaxVreg, lastMaxVreg, nowMaxFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + i * FLOAT_REP_SIZE, nowMaxFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb + i * FLOAT_REP_SIZE, expMaxVreg, pregFull);

            LoadAlign(lastExpSumVreg, sumUb + i * FLOAT_REP_SIZE);
            LoadAlign(expSumTmpVreg, expSumUb + i * FLOAT_REP_SIZE);
            Interleave(expSumVreg0, expSumVreg1, expSumTmpVreg, expSumTmpVreg);
            Cast<float, ElementInput, castTraitZero>(expSumFloatVreg, expSumVreg0, pregB16Full);
            Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregFull);
            Add(updateExpSumVreg, updateExpSumVreg, expSumFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + i * FLOAT_REP_SIZE, updateExpSumVreg, pregFull);
        }
        LoadAlign(lastMaxVreg, maxUb + mFullVecCnt * FLOAT_REP_SIZE);
        LoadAlign(nowMaxTmpVreg, nowMaxUb + mFullVecCnt * FLOAT_REP_SIZE);
        Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
        Cast<float, ElementInput, castTraitZero>(nowMaxFloatVreg, nowMaxVreg0, pregB16Full);
        FusedExpSub(expMaxVreg, lastMaxVreg, nowMaxFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + mFullVecCnt * FLOAT_REP_SIZE, nowMaxFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb + mFullVecCnt * FLOAT_REP_SIZE, expMaxVreg, pregTailM);

        LoadAlign(lastExpSumVreg, sumUb + mFullVecCnt * FLOAT_REP_SIZE);
        LoadAlign(expSumTmpVreg, expSumUb + mFullVecCnt * FLOAT_REP_SIZE);
        Interleave(expSumVreg0, expSumVreg1, expSumTmpVreg, expSumTmpVreg);
        Cast<float, ElementInput, castTraitZero>(expSumFloatVreg, expSumVreg0, pregB16Full);
        Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregTailM);
        Add(updateExpSumVreg, updateExpSumVreg, expSumFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + mFullVecCnt * FLOAT_REP_SIZE, updateExpSumVreg, pregTailM);
    }
};

} // namespace Catlass::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RAIN_FUSION_ATTENTION_SOFTMAX_LOW_PREC_FP16_HPP
