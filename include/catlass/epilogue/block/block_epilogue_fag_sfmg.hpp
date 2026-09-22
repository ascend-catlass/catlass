/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_SFMG_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_SFMG_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/block/block_epilogue_fag_common.hpp"

namespace Catlass::Epilogue::Block {

using namespace AscendC;

constexpr AscendC::RoundMode FLOAT2HALF_ROUND_MODE = AscendC::RoundMode::CAST_NONE;

struct ReduceLastND {
    uint32_t originalSrcM;
    uint32_t originalSrcK;
    uint32_t srcM;
    uint32_t srcK;
    uint32_t dstM;
    uint32_t dstK;
};

struct SoftMaxShapeInfo {
    uint32_t srcM{0};
    uint32_t srcK{0};
    uint32_t oriSrcM{0};
    uint32_t oriSrcK{0};
};

__aicore__ inline void CustomAlignedReduceSumNDImpl(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<float>& tmpTensor,
    const struct ReduceLastND& reduceParam, const uint32_t splitCount)
{
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_REPEAT_SIZE);
    BlockReduceSum<float, false>(tmpTensor, src, 1, MASK_PLACEHOLDER, 1, 1, reduceParam.srcK / FLOAT_NUM_PER_BLK);
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();
    DataCopy(dst, tmpTensor, {1, (uint16_t)reduceParam.srcM, 0, 0});
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    for (uint32_t i = 1; i < splitCount; i++) {
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_REPEAT_SIZE);
        BlockReduceSum<float, false>(
            tmpTensor, src[i * FLOAT_REPEAT_SIZE], 1, MASK_PLACEHOLDER, 1, 1, reduceParam.srcK / FLOAT_NUM_PER_BLK);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_NUM_PER_BLK);
        Add<float, false>(
            dst, dst, tmpTensor, MASK_PLACEHOLDER, 1,
            {1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
    }
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_NUM_PER_BLK);
    BlockReduceSum<float, false>(dst, dst, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void CustomReduceSumLastNDSplitImpl(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const struct ReduceLastND& reduceParam, uint64_t mask,
    uint32_t dstRepStride, uint32_t splitNum)
{
    uint32_t range = reduceParam.srcM / MAX_REPEAT_TIMES;
    uint32_t tail = reduceParam.srcM % MAX_REPEAT_TIMES;

    for (uint32_t i = 0; i < range; i++) {
        WholeReduceSum(
            dst[i * MAX_REPEAT_TIMES], src[splitNum * FLOAT_REPEAT_SIZE + i * MAX_REPEAT_TIMES * reduceParam.srcK],
            mask, MAX_REPEAT_TIMES, dstRepStride, 1, reduceParam.srcK / FLOAT_NUM_PER_BLK);
    }
    if (tail != 0) {
        WholeReduceSum(
            dst[range * MAX_REPEAT_TIMES],
            src[splitNum * FLOAT_REPEAT_SIZE + range * MAX_REPEAT_TIMES * reduceParam.srcK], mask, tail, dstRepStride,
            1, reduceParam.srcK / FLOAT_NUM_PER_BLK);
    }
}

__aicore__ inline void CustomSingleBlockBroadCastImpl(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const struct ReduceLastND& reduceParam)
{
    BrcbRepeatParams brcbParams;
    brcbParams.dstBlkStride = 1;
    brcbParams.dstRepStride = BRCB_BROADCAST_NUMBER;
    const uint32_t range = reduceParam.originalSrcM / BRCB_BROADCAST_NUMBER;
    const uint32_t tail = reduceParam.originalSrcM % BRCB_BROADCAST_NUMBER;

    if (range != 0) {
        if (reduceParam.dstK == BRCB_BROADCAST_NUMBER * HALF_FACTOR) {
            brcbParams.dstBlkStride = HALF_FACTOR;
            brcbParams.dstRepStride = BRCB_BROADCAST_NUMBER * HALF_FACTOR;
            Brcb(dst[0], src, range, brcbParams);
            Brcb(dst[BRCB_BROADCAST_NUMBER], src, range, brcbParams);
        } else {
            Brcb(dst, src, range, brcbParams);
        }
    }

    if (tail != 0) {
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float scalarList[SCALAR_STACK_DEPTH] = {0};
        for (uint32_t j = 0; j < tail; j++) {
            scalarList[j] = src[(range * BRCB_BROADCAST_NUMBER + j)].GetValue(0);
        }

        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        for (uint32_t k = 0; k < tail; k++) {
            Duplicate(
                dst[(range * SCALAR_STACK_DEPTH + k) * reduceParam.dstK], scalarList[k], reduceParam.dstK, 1,
                DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        }
    }
}

__aicore__ inline void CustomReduceSumLastNDImpl(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<float>& tmpTensor,
    const struct ReduceLastND& reduceParam)
{
    const uint32_t splitCount = reduceParam.originalSrcK / FLOAT_REPEAT_SIZE;
    const uint32_t tailSrcK = reduceParam.originalSrcK % FLOAT_REPEAT_SIZE;
    if (splitCount > 0) {
        CustomAlignedReduceSumNDImpl(tmpTensor, src, dst, reduceParam, splitCount);
    }

    if (tailSrcK != 0) {
        CustomReduceSumLastNDSplitImpl(dst, src, reduceParam, tailSrcK, 1, splitCount);
        PipeBarrier<PIPE_V>();
        if (splitCount == 0) {
            DataCopy(tmpTensor, dst, {1, (uint16_t)reduceParam.srcM, 0, 0});
        } else {
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * FLOAT_NUM_PER_BLK);
            Add<float, false>(
                tmpTensor, tmpTensor, dst, MASK_PLACEHOLDER, 1,
                {1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
            SetMaskNorm();
            ResetMask();
        }
    }

    PipeBarrier<PIPE_V>();
    CustomSingleBlockBroadCastImpl(dst, tmpTensor, reduceParam);
}

template <typename T, bool isBasicBlock = false>
__aicore__ inline void CustomSoftmaxGradFrontNDImpl(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling, const LastAxisShapeND& originalSrcShape)
{
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);

    ReduceLastND reduceSumParam = {tiling.splitM, originalSrcShape.k, tiling.splitM,
                                   tiling.splitK, tiling.reduceM,     tiling.reduceK};

    if constexpr (sizeof(T) == sizeof(half)) {
        LocalTensor<float> srcBuffer = workLocal;
        LocalTensor<float> gradBuffer = workLocal[tiling.splitSize];
        LocalTensor<float> dstBuffer = workLocal[tiling.splitSize + tiling.splitSize];

        LocalTensor<float> reduceBuffer = workLocal[tiling.splitSize + tiling.splitSize + tiling.splitSize];
        LocalTensor<float> addBuffer =
            workLocal[tiling.splitSize + tiling.splitSize + tiling.splitSize + tiling.reduceSize];
        const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
        const uint32_t elementNumPerBlk = DEFAULT_C0_SIZE / B32_BYTE_SIZE;
        uint8_t offset = (uint8_t)(splitBlock * elementNumPerBlk);
        const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, FLOAT_NUM_PER_BLK));
        const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
        const uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
        SetMaskNorm();
        ResetMask();
        for (uint32_t i = 0; i < tiling.rangeM; i++) {
            if constexpr (isBasicBlock) {
                Cast<float, half, false>(
                    srcBuffer, srcTensor[i * tiling.splitSize], RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
                    {1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE});
                Cast<float, half, false>(
                    gradBuffer, gradTensor[i * tiling.splitSize], RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
                    {1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE});
                PipeBarrier<PIPE_V>();

                Mul<float, false>(
                    dstBuffer, srcBuffer, gradBuffer, MASK_PLACEHOLDER, repeatTimes,
                    {1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
                for (uint32_t j = 1; j < splitBlock; ++j) {
                    PipeBarrier<PIPE_V>();
                    Add<float, false>(
                        dstBuffer, dstBuffer, dstBuffer[FLOAT_REPEAT_SIZE * j], MASK_PLACEHOLDER,
                        (uint8_t)(tiling.splitM), {1, 1, 1, offset, offset, offset});
                }
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float, false>(
                    dstBuffer, dstBuffer, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1, offset);
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float, false>(
                    reduceBuffer, dstBuffer, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
                PipeBarrier<PIPE_V>();
                Brcb(dstBuffer, reduceBuffer, splitCeilM, {B16_BYTE_SIZE, DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE});

                Brcb(
                    dstBuffer[DEFAULT_BLK_NUM], reduceBuffer, splitCeilM,
                    {B16_BYTE_SIZE, DEFAULT_REPEAT_STRIDE * B16_BYTE_SIZE});

                PipeBarrier<PIPE_V>();
                Cast<half, float, false>(
                    dstTensor[i * tiling.reduceSize], dstBuffer, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER,
                    reduceCeilValue, {1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
            } else {
                Cast(srcBuffer, srcTensor[i * tiling.splitSize], RoundMode::CAST_NONE, tiling.splitSize);
                Cast(gradBuffer, gradTensor[i * tiling.splitSize], RoundMode::CAST_NONE, tiling.splitSize);
                PipeBarrier<PIPE_V>();
                Mul(dstBuffer, srcBuffer, gradBuffer, tiling.splitSize);
                PipeBarrier<PIPE_V>();
                CustomReduceSumLastNDImpl(addBuffer, dstBuffer, reduceBuffer, reduceSumParam);
                PipeBarrier<PIPE_V>();
                Cast(dstTensor[i * tiling.reduceSize], addBuffer, FLOAT2HALF_ROUND_MODE, tiling.reduceSize);
            }
        }
        if (tiling.tailM != 0) {
            Cast(srcBuffer, srcTensor[tiling.rangeM * tiling.splitSize], RoundMode::CAST_NONE, tiling.tailSplitSize);
            Cast(gradBuffer, gradTensor[tiling.rangeM * tiling.splitSize], RoundMode::CAST_NONE, tiling.tailSplitSize);
            PipeBarrier<PIPE_V>();
            Mul(dstBuffer, srcBuffer, gradBuffer, tiling.tailSplitSize);
            reduceSumParam.srcM = tiling.tailM;
            reduceSumParam.dstM = tiling.tailM;
            reduceSumParam.originalSrcM = tiling.tailM;
            PipeBarrier<PIPE_V>();
            CustomReduceSumLastNDImpl(addBuffer, dstBuffer, reduceBuffer, reduceSumParam);
            PipeBarrier<PIPE_V>();
            Cast(dstTensor[tiling.rangeM * tiling.reduceSize], addBuffer, FLOAT2HALF_ROUND_MODE, tiling.tailReduceSize);
        }
    } else {
        LocalTensor<float> srcBuffer = workLocal;
        LocalTensor<float> reduceBuffer = workLocal[tiling.splitSize];
        uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
        uint32_t offset1 = 0;
        uint32_t offset2 = 0;
        const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
        const uint32_t elementNumPerBlk = DEFAULT_C0_SIZE / B32_BYTE_SIZE;
        uint8_t offset = (uint8_t)(splitBlock * elementNumPerBlk);
        const uint8_t splitCeilM = (uint8_t)(DivCeil(tiling.splitM, elementNumPerBlk));
        SetMaskNorm();
        ResetMask();
        for (uint32_t i = 0; i < tiling.rangeM; i++) {
            if constexpr (isBasicBlock) {
                offset2 = i * tiling.reduceSize;
                offset1 = i * tiling.splitSize;
                PipeBarrier<PIPE_V>();
                Mul<float, false>(
                    srcBuffer, srcTensor[offset1], gradTensor[offset1], MASK_PLACEHOLDER, repeatTimes,
                    {1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});

                for (uint32_t j = 1; j < splitBlock; ++j) {
                    PipeBarrier<PIPE_V>();
                    Add<float, false>(
                        srcBuffer, srcBuffer, srcBuffer[FLOAT_REPEAT_SIZE * j], MASK_PLACEHOLDER,
                        (uint8_t)(tiling.splitM), {1, 1, 1, offset, offset, offset});
                }
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float, false>(
                    srcBuffer, srcBuffer, (uint8_t)(tiling.splitM), MASK_PLACEHOLDER, 1, 1,
                    splitBlock * DEFAULT_REPEAT_STRIDE);
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float, false>(
                    reduceBuffer, srcBuffer, splitCeilM, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
                PipeBarrier<PIPE_V>();

                Brcb(dstTensor[offset2], reduceBuffer, splitCeilM, {1, DEFAULT_REPEAT_STRIDE});

            } else {
                Mul(srcBuffer, srcTensor[i * tiling.splitSize], gradTensor[i * tiling.splitSize], tiling.splitSize);
                PipeBarrier<PIPE_V>();
                CustomReduceSumLastNDImpl(dstTensor[i * tiling.reduceSize], srcBuffer, reduceBuffer, reduceSumParam);
                PipeBarrier<PIPE_V>();
            }
        }

        if (tiling.tailM != 0) {
            Mul(srcBuffer, srcTensor[tiling.rangeM * tiling.splitSize], gradTensor[tiling.rangeM * tiling.splitSize],
                tiling.tailSplitSize);
            PipeBarrier<PIPE_V>();

            reduceSumParam.srcM = tiling.tailM;
            reduceSumParam.dstM = tiling.tailM;
            reduceSumParam.originalSrcM = tiling.tailM;
            CustomReduceSumLastNDImpl(
                dstTensor[tiling.rangeM * tiling.reduceSize], srcBuffer, reduceBuffer, reduceSumParam);
            PipeBarrier<PIPE_V>();
        }
    }
}

__aicore__ inline bool CustomSoftMaxGradTilingFunc(
    const uint32_t workLocalSize, const LastAxisShapeND& ndinfo, SoftMaxTiling& softmaxTiling,
    const uint32_t elementNumPerBlk, bool isFront = false, bool isBasicBlock = false, bool isDataFormatNZ = false)
{
    softmaxTiling.srcM = ndinfo.m;
    softmaxTiling.srcK = ndinfo.k;
    softmaxTiling.srcSize = ndinfo.m * ndinfo.k;

    softmaxTiling.outMaxM = ndinfo.m;
    softmaxTiling.outMaxK = elementNumPerBlk;
    softmaxTiling.outMaxSize = ndinfo.m * elementNumPerBlk;

    if (elementNumPerBlk != ONE_BYTE_BIT_SIZE) {
        softmaxTiling.reduceM = workLocalSize / (elementNumPerBlk * SOFTMAX_COMPUTE_DIM +
                                                 ndinfo.k * SOFTMAXGRAD_COMPUTE_DIM + FLOAT_REPEAT_SIZE);
    } else {
        if (isFront && !isDataFormatNZ) {
            softmaxTiling.reduceM = workLocalSize / (elementNumPerBlk + ndinfo.k + FLOAT_REPEAT_SIZE);
        } else {
            softmaxTiling.reduceM =
                workLocalSize / (ndinfo.k + elementNumPerBlk * SOFTMAX_COMPUTE_DIM + FLOAT_REPEAT_SIZE);
        }
    }
    if (softmaxTiling.reduceM < ndinfo.m && softmaxTiling.reduceM > SOFTMAX_BASIC_TILE_NUM) {
        softmaxTiling.reduceM = softmaxTiling.reduceM / SOFTMAX_BASIC_TILE_NUM * SOFTMAX_BASIC_TILE_NUM;
    }
    softmaxTiling.reduceM = softmaxTiling.reduceM < ndinfo.m ? softmaxTiling.reduceM : ndinfo.m;

    if (isBasicBlock && isFront && (softmaxTiling.reduceM > SOFTMAX_BASIC_TILE_NUM) &&
        (softmaxTiling.srcM % SOFTMAX_BASIC_TILE_NUM == 0)) {
        softmaxTiling.reduceM = softmaxTiling.reduceM / SOFTMAX_BASIC_TILE_NUM * SOFTMAX_BASIC_TILE_NUM;
        while (softmaxTiling.srcM % softmaxTiling.reduceM != 0) {
            softmaxTiling.reduceM -= SOFTMAX_BASIC_TILE_NUM;
        }
        while (softmaxTiling.reduceM * ndinfo.k >= FLOAT_REPEAT_SIZE * DEFAULT_BLOCK_SIZE) {
            softmaxTiling.reduceM = softmaxTiling.reduceM / B16_BYTE_SIZE;
        }
    }

    softmaxTiling.reduceK = elementNumPerBlk;
    softmaxTiling.reduceSize = softmaxTiling.reduceM * elementNumPerBlk;

    softmaxTiling.splitM = softmaxTiling.reduceM;
    softmaxTiling.splitK = ndinfo.k;
    softmaxTiling.splitSize = softmaxTiling.reduceM * ndinfo.k;

    softmaxTiling.rangeM = ndinfo.m / softmaxTiling.reduceM;
    softmaxTiling.tailM = ndinfo.m % softmaxTiling.reduceM;

    softmaxTiling.tailSplitSize = softmaxTiling.tailM * ndinfo.k;
    softmaxTiling.tailReduceSize = softmaxTiling.tailM * elementNumPerBlk;

    return true;
}

template <typename T, bool isBasicBlock = false>
__aicore__ inline void SoftmaxGradFrontImpl(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<float>& workLocal, const int64_t& srcDim0, const int64_t& srcDim1, const int64_t& dstDim1,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    ShapeInfo srcShape = srcTensor.GetShapeInfo();
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
    LastAxisShapeND srcNDinfo;
    LastAxisShapeND originalSrcShape;
    srcShape.shape[0] = srcDim0;
    srcShape.shape[1] = srcDim1;
    srcShape.originalShape[1] = srcDim1;

    srcNDinfo = FagGetLastAxisShapeND(srcShape);
    originalSrcShape = FagGetLastAxisOriginShapeND(srcShape);

    SoftMaxTiling newTiling{};
    CustomSoftMaxGradTilingFunc(workLocal.GetSize(), srcNDinfo, newTiling, elementNumPerBlk, true, isBasicBlock);
    CustomSoftmaxGradFrontNDImpl<T, isBasicBlock>(
        dstTensor, gradTensor, srcTensor, workLocal, newTiling, originalSrcShape);
}

template <typename T, bool isBasicBlock = false>
__aicore__ inline void SoftmaxGradFrontImpl(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const int64_t& srcDim0, const int64_t& srcDim1, const int64_t& dstDim1,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SoftmaxGradFrontImpl<T, isBasicBlock>(
        dstTensor, gradTensor, srcTensor, workLocal, srcDim0, srcDim1, dstDim1, softmaxShapeInfo);
}

template <typename T, bool isBasicBlock = false>
__aicore__ inline void SoftmaxGradFront(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const int64_t& srcDim0, const int64_t& srcDim1, const int64_t& dstDim1,
    const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    SoftmaxGradFrontImpl<T, isBasicBlock>(
        dstTensor, gradTensor, srcTensor, sharedTmpBuffer, srcDim0, srcDim1, dstDim1, softmaxShapeInfo);
}


// 通用BlockEpilogue特化，接受布局类型标签作为模板参数
template <class OutputType_, class UpdateType_, class InputType_, uint32_t INPUT_LAYOUT_>
class BlockEpilogue<EpilogueAtlasA2FAGSfmg<INPUT_LAYOUT_>, OutputType_, UpdateType_, InputType_> {
public:
    using DispatchPolicy = EpilogueAtlasA2FAGSfmg<INPUT_LAYOUT_>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using InputType = InputType_;
    static constexpr uint32_t INPUT_LAYOUT = INPUT_LAYOUT_;

    // kernel-parsed tiling fields consumed by this epilogue (no raw tiling parsing here)
    struct Params {
        int64_t batch;
        int64_t qHeadNum;
        int64_t kvHeadNum;
        int64_t g;
        int64_t qkHeadDim;
        int64_t qSeqlen;
        int64_t kvSeqlen;
        int64_t coreNum;
        int64_t t1;
        int64_t sfmgPreBeginAddr;
        SoftMaxTiling softmaxGradTilingData;
    };

    CATLASS_DEVICE
    BlockEpilogue(
        Arch::Resource<ArchTag>& resource, AscendC::TPipe* pipe_in, __gm__ uint8_t* dout, __gm__ uint8_t* out,
        __gm__ uint8_t* cu_seq_qlen, __gm__ uint8_t* workspace, const Params& params)
    {
        cBlockIdx = GetBlockIdx();
        pipe = pipe_in;

        batch = params.batch;
        nheads = params.qHeadNum;
        nheads_k = params.kvHeadNum;
        g = params.g;
        headdim = params.qkHeadDim;
        s1 = params.qSeqlen;
        s2 = params.kvSeqlen;
        uint32_t coreNum = static_cast<uint32_t>(params.coreNum);
        dAlign = (headdim + 15) / 16 * 16;

        // 计算 buffer 大小
        constexpr static uint32_t inputBufferLen = 24 * 1024; // castBuffer 24K*2=48K
        constexpr static uint32_t castBufferLen = 48 * 1024;  // castBuffer 48K*2=96K
        uint32_t outputBufferLen = (castBufferLen + dAlign - 1) / dAlign * 8;
        uint32_t tempBufferLen = 40 * 1024 - outputBufferLen;

        // 计算单核的计算量
        int64_t normalAxisSize = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            normalAxisSize = params.t1 * nheads_k * g;
        } else {
            normalAxisSize = batch * nheads * s1;
        }

        normalCoreSize = (normalAxisSize + coreNum - 1) / coreNum;
        usedCoreNum = (normalAxisSize + normalCoreSize - 1) / normalCoreSize;

        // 计算单loop的计算量及loop次数
        singleLoopNBurstNum = inputBufferLen / sizeof(InputType) / dAlign;
        normalCoreLoopTimes = (normalCoreSize + singleLoopNBurstNum - 1) / singleLoopNBurstNum;
        normalCoreLastLoopNBurstNum = normalCoreSize - (normalCoreLoopTimes - 1) * singleLoopNBurstNum;

        int64_t tailCoreSize = normalAxisSize - (usedCoreNum - 1) * normalCoreSize;
        tailCoreLoopTimes = (tailCoreSize + singleLoopNBurstNum - 1) / singleLoopNBurstNum;
        tailCoreLastLoopNBurstNum = tailCoreSize - (tailCoreLoopTimes - 1) * singleLoopNBurstNum;

        cu_seq_qlen_addr = cu_seq_qlen;

        if constexpr (INPUT_LAYOUT == TND) {
            n_stride = (nheads * headdim - headdim) * sizeof(InputType);
        } else if constexpr (INPUT_LAYOUT == BNSD) {
            n_stride = 0;
        } else if constexpr (INPUT_LAYOUT == BSND) {
            n_stride = (nheads * headdim - headdim) * sizeof(InputType);
        } else if constexpr (INPUT_LAYOUT == SBH) {
            n_stride = (batch * nheads * headdim - headdim) * sizeof(InputType);
        }

        // 初始化 buffer
        pipe->InitBuffer(inBuffer1, inputBufferLen); // 24K
        pipe->InitBuffer(inBuffer2, inputBufferLen); // 24K
        pipe->InitBuffer(cast1Buf, castBufferLen);   // 48K
        pipe->InitBuffer(cast2Buf, castBufferLen);   // 48K
        pipe->InitBuffer(outBuffer1, outputBufferLen);
        pipe->InitBuffer(tmpBuf, tempBufferLen); // 40K - outputBufferLen

        // 初始化 GM
        doutGm.SetGlobalBuffer((__gm__ InputType*)dout);
        outGm.SetGlobalBuffer((__gm__ InputType*)out);
        sfmgWorkspaceGm.SetGlobalBuffer((__gm__ float*)workspace + params.sfmgPreBeginAddr / sizeof(float));

        //-----------------------softmaxGradTilingData-------------------------------
        softmaxGradTilingData = params.softmaxGradTilingData;
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {}

    CATLASS_DEVICE
    void InitIndex(int64_t startIdx, int64_t& curS, GM_ADDR seqS)
    {
        if constexpr (INPUT_LAYOUT == TND) {
            int64_t totalLen = 0;
            for (int64_t bDimIdx = bIdx; bDimIdx < batch; bDimIdx++) {
                totalLen = nheads * ((__gm__ int64_t*)seqS)[bDimIdx] * headdim;
                if (totalLen > startIdx) {
                    bIdx = bDimIdx;
                    curS = (bIdx == 0) ? ((__gm__ int64_t*)seqS)[bIdx] :
                                         (((__gm__ int64_t*)seqS)[bIdx] - ((__gm__ int64_t*)seqS)[bIdx - 1]);
                    int64_t bTail = startIdx - (totalLen - nheads * curS * headdim);
                    nIdx = bTail / (curS * headdim);
                    int64_t nTail = bTail % (curS * headdim);
                    sIdx = nTail / headdim;
                    break;
                }
            }
        } else {
            bIdx = startIdx / (nheads * s1 * headdim);
            int64_t bTail = startIdx % (nheads * s1 * headdim);
            nIdx = bTail / (s1 * headdim);
            int64_t nTail = bTail % (s1 * headdim);
            sIdx = nTail / headdim;
        }
    }

    CATLASS_DEVICE
    void DoCopyIn(int64_t curS, int64_t curNBurst, int64_t dstOffset, GM_ADDR seqS)
    {
        int64_t srcOffset = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            int64_t bOffset = bIdx == 0 ? 0 : nheads * ((__gm__ int64_t*)seqS)[bIdx - 1] * headdim;
            srcOffset = bOffset + (sIdx * nheads + nIdx) * headdim;
        } else {
            if constexpr (INPUT_LAYOUT == BNSD) {
                srcOffset = bIdx * (nheads * s1 * headdim) + nIdx * (s1 * headdim) + sIdx * headdim;
            } else if constexpr (INPUT_LAYOUT == BSND) {
                srcOffset = bIdx * (s1 * nheads * headdim) + sIdx * (nheads * headdim) + nIdx * headdim;
            } else if constexpr (INPUT_LAYOUT == SBH) {
                srcOffset = sIdx * (batch * nheads * headdim) + bIdx * (nheads * headdim) + nIdx * headdim;
            }
        }
        DataCopyPad(
            input1Buf[dstOffset], doutGm[srcOffset],
            {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(headdim * sizeof(InputType)),
             static_cast<uint32_t>(n_stride), 0, 0},
            {true, 0, static_cast<uint8_t>((dAlign - headdim)), 0});
        DataCopyPad(
            input2Buf[dstOffset], outGm[srcOffset],
            {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(headdim * sizeof(InputType)),
             static_cast<uint32_t>(n_stride), 0, 0},
            {true, 0, static_cast<uint8_t>((dAlign - headdim)), 0});
    }

    CATLASS_DEVICE
    void CopyInSfmg(int64_t leftNburst, int64_t& curS, GM_ADDR seqS)
    {
        int64_t dstOffset = 0;
        while (leftNburst > 0) {
            int64_t curNburst = 0;
            if (curS - sIdx < leftNburst) { // 需要借N或借B
                curNburst = curS - sIdx;
                DoCopyIn(curS, curNburst, dstOffset, seqS);
                leftNburst = leftNburst - curNburst;
                sIdx = 0;
                if (nIdx < nheads - 1) { // 需要借N
                    nIdx += 1;
                } else {
                    nIdx = 0;
                    if (bIdx < batch - 1) { // 需要借B
                        bIdx += 1;
                        if constexpr (INPUT_LAYOUT == TND) {
                            curS = ((__gm__ int64_t*)seqS)[bIdx] - ((__gm__ int64_t*)seqS)[bIdx - 1];
                        } else {
                            curS = s1;
                        }
                    } else { // 没有轴可以借了，end
                        leftNburst = 0;
                    }
                }
            } else { // 当前S够用
                curNburst = leftNburst;
                DoCopyIn(curS, curNburst, dstOffset, seqS);
                sIdx = sIdx + leftNburst;
                leftNburst = 0;
            }
            dstOffset = dstOffset + curNburst * dAlign;
        }
    }

    CATLASS_DEVICE
    void operator()()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        event_t VWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        event_t VWaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        event_t Mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        event_t Mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));

        uint32_t usedCoreNums = usedCoreNum;
        if (cBlockIdx < usedCoreNums) {
            LocalTensor<uint8_t> tempBuf = tmpBuf.Get<uint8_t>();
            LocalTensor<float> sfmgClc1 = cast1Buf.Get<float>();
            LocalTensor<float> sfmgClc2 = cast2Buf.Get<float>();

            int64_t singleCoreLoopTimes = normalCoreLoopTimes;
            int64_t singleCoreLastLoopNBurstNum = normalCoreLastLoopNBurstNum; // 普通单核最后一次loop处理多少个D
            if (cBlockIdx == usedCoreNums - 1) {
                singleCoreLoopTimes = tailCoreLoopTimes;
                singleCoreLastLoopNBurstNum = tailCoreLastLoopNBurstNum;
            }

            int64_t startIdx = cBlockIdx * normalCoreSize;
            int64_t nBurst = singleLoopNBurstNum;
            // int64_t curS = 0;
            int64_t curS = s1;

            for (int64_t i = 0; i < singleCoreLoopTimes; i++) {
                if (i == singleCoreLoopTimes - 1) {
                    nBurst = singleCoreLastLoopNBurstNum;
                }

                // copyIn
                if (i == 0) {
                    input1Buf = inBuffer1.Get<InputType>();
                    input2Buf = inBuffer2.Get<InputType>();
                    InitIndex((startIdx + i * singleLoopNBurstNum) * headdim, curS, cu_seq_qlen_addr);

                    CopyInSfmg(nBurst, curS, cu_seq_qlen_addr);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(VWaitMte2);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(VWaitMte2);

                // cast 1
                int64_t calcSize = nBurst * dAlign;
                Cast(sfmgClc1, input1Buf, RoundMode::CAST_NONE, calcSize);
                AscendC::PipeBarrier<PIPE_V>();

                // cast 2
                Cast(sfmgClc2, input2Buf, RoundMode::CAST_NONE, calcSize);
                AscendC::PipeBarrier<PIPE_V>();

                // pre copyIn next nBurst
                if (i < singleCoreLoopTimes - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(Mte2WaitV);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(Mte2WaitV);
                    int64_t nextNBurst = i == singleCoreLoopTimes - 2 ? singleCoreLastLoopNBurstNum : nBurst;
                    input1Buf = inBuffer1.Get<InputType>();
                    input2Buf = inBuffer2.Get<InputType>();
                    InitIndex((startIdx + (i + 1) * singleLoopNBurstNum) * headdim, curS, cu_seq_qlen_addr);
                    CopyInSfmg(nextNBurst, curS, cu_seq_qlen_addr);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(VWaitMte2);
                }

                if (i > 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(VWaitMte3);
                }

                // sfmg
                outputBuf = outBuffer1.Get<float>();
                AscendC::Duplicate<float>(outputBuf, 0.0, nBurst * 8);
                AscendC::PipeBarrier<PIPE_V>();

                uint32_t shapeArray[2] = {static_cast<uint32_t>(nBurst), static_cast<uint32_t>(dAlign)};
                ShapeInfo srcShape(2, shapeArray, AscendC::DataFormat::ND);
                srcShape.shape[0] = static_cast<uint32_t>(nBurst);
                srcShape.shape[1] = static_cast<uint32_t>(dAlign);
                srcShape.originalShape[0] = static_cast<uint32_t>(nBurst);
                srcShape.originalShape[1] = static_cast<uint32_t>(dAlign);
                sfmgClc1.SetShapeInfo(srcShape);
                sfmgClc2.SetShapeInfo(srcShape);
                uint32_t shapeArray1[2] = {static_cast<uint32_t>(nBurst), BLOCK_BYTE_SIZE / sizeof(float)};
                ShapeInfo dstShape(2, shapeArray1, AscendC::DataFormat::ND);
                dstShape.shape[0] = static_cast<uint32_t>(nBurst);
                dstShape.shape[1] = BLOCK_BYTE_SIZE / sizeof(float);
                dstShape.originalShape[0] = static_cast<uint32_t>(nBurst);
                dstShape.originalShape[1] = BLOCK_BYTE_SIZE / sizeof(float);
                outputBuf.SetShapeInfo(dstShape);

                bool isBasicBlock = (nBurst % SFMG_HIGH_PERF_N_FACTOR == 0) && (dAlign % SFMG_HIGH_PERF_D_FACTOR == 0);

                if (likely(isBasicBlock)) {
                    SoftmaxGradFront<float, true>(
                        outputBuf, sfmgClc1, sfmgClc2, tempBuf, nBurst, dAlign, BLOCK_BYTE_SIZE / sizeof(float));
                } else {
                    SoftmaxGradFront<float, false>(
                        outputBuf, sfmgClc1, sfmgClc2, tempBuf, nBurst, dAlign, BLOCK_BYTE_SIZE / sizeof(float));
                }

                AscendC::PipeBarrier<PIPE_V>();

                // copyOut
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);

                int64_t sfmgOutputOffset = (startIdx + i * singleLoopNBurstNum) * BLOCK_SIZE;
                DataCopy(sfmgWorkspaceGm[sfmgOutputOffset], outputBuf, nBurst * BLOCK_SIZE);
                if (i < singleCoreLoopTimes - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(VWaitMte3);
                }
            }
        }
    }

protected:
    /// Data members
    constexpr static int64_t BLOCK_BYTE_SIZE = 32;
    constexpr static int64_t BLOCK_SIZE = 8;
    constexpr static int64_t SFMG_HIGH_PERF_N_FACTOR = 8;
    constexpr static int64_t SFMG_HIGH_PERF_D_FACTOR = 64;

    AscendC::TPipe* pipe;
    uint32_t cBlockIdx;

    GlobalTensor<float> sfmgWorkspaceGm;
    GlobalTensor<InputType> doutGm;
    GlobalTensor<InputType> outGm;
    TBuf<QuePosition::VECIN> inBuffer1, inBuffer2;
    TBuf<> cast1Buf, cast2Buf, tmpBuf;
    TBuf<QuePosition::VECOUT> outBuffer1;

    int64_t batch;
    int64_t nheads;
    int64_t nheads_k;
    int64_t g;
    // int64_t total_q;
    int64_t headdim;
    int64_t dAlign;
    int64_t s1;
    int64_t s2;
    GM_ADDR cu_seq_qlen_addr;

    int64_t bIdx = 0;
    int64_t nIdx = 0;
    int64_t sIdx = 0;

    int64_t dstOffset = 0;
    int64_t n_stride = 0;

    int64_t usedCoreNum;
    int64_t normalCoreSize;
    int64_t singleLoopNBurstNum;
    int64_t normalCoreLoopTimes;
    int64_t normalCoreLastLoopNBurstNum;
    int64_t tailCoreLoopTimes;
    int64_t tailCoreLastLoopNBurstNum;

    LocalTensor<InputType> input1Buf;
    LocalTensor<InputType> input2Buf;
    LocalTensor<float> outputBuf;

    SoftMaxTiling softmaxGradTilingData;
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_SFMG_HPP
