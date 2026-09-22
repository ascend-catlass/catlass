/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_CUSTOM_SAB_VEC_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_CUSTOM_SAB_VEC_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/block/block_epilogue_fag_common.hpp"

using AscendC::CopyRepeatParams;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TQue;

namespace Catlass::Epilogue::Block {

template <
    class OutputType_, class UpdateType_, class InputType_, uint32_t INPUT_LAYOUT_
    >
class BlockEpilogue<EpilogueAtlasA2CustomSabVec<INPUT_LAYOUT_>, OutputType_, UpdateType_, InputType_> {
public:
    using DispatchPolicy = EpilogueAtlasA2CustomSabVec<INPUT_LAYOUT_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using T1 = InputType_;
    using T2 = OutputType_;
    static constexpr uint32_t INPUT_LAYOUT = INPUT_LAYOUT_;

    // kernel-parsed tiling fields consumed by this epilogue (no raw tiling parsing here)
    struct Params {
        SoftMaxTiling softmaxTilingData;
        int64_t coreNum;
        int64_t batch;
        int64_t kvHeadNum;
        int64_t g;
        int64_t qSeqlen;
        int64_t kvSeqlen;
        int64_t qkHeadDim;
        int64_t vHeadDim;
        int64_t s1Token;
        int64_t s2Token;
        uint32_t sparseMode;
        int64_t s1Outer;
        uint32_t s1CvInner;
        uint32_t s1CvTail;
        int64_t s2Outer;
        uint32_t s2CvInner;
        uint32_t baseMN;
        float keepProb;
        float scaleValue;
        uint64_t dqWorkSpaceOffset;
        uint64_t dkWorkSpaceOffset;
        uint64_t dvWorkSpaceOffset;
        int64_t sfmgPreBeginAddr;
    };

    uint32_t isAttenMask = 0;
    uint32_t isDrop = 0;

    AscendC::TPipe* pipe;
    TBuf<> unifiedBuffer;

    uint32_t coreNum;
    uint32_t cubeCoreNum;
    uint32_t cBlockIdx;
    uint32_t cCubeBlockIdx;
    uint32_t cSubIdx;

    uint32_t vecBlockNum;

    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, forwardResGm;
    GlobalTensor<uint8_t> maskWorkSpaceGm, attenMaskU8Gm, dropMaskGm;
    GlobalTensor<float> softmaxLseGm;

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, sfmgWorkspaceGm;

    GlobalTensor<T1> dropWorkSpaceGm, mulWorkSpaceGm;

    GlobalTensor<T2> mm1WorkspaceGm;
    GlobalTensor<T2> mm2WorkspaceGm;

    TBuf<AscendC::TPosition::A1> queryBufL1;
    LocalTensor<T1> qL1Tensor;
    TBuf<AscendC::TPosition::A1> keyBufL1;
    LocalTensor<T1> vL1Tensor;
    LocalTensor<T1> kL1Tensor;
    TBuf<AscendC::TPosition::A1> dsBufL1;
    LocalTensor<T1> dsL1Tensor;
    LocalTensor<T1> dxL1Tensor;

    __gm__ uint8_t* actual_seq_qlen_addr;
    __gm__ uint8_t* actual_seq_kvlen_addr;

    GlobalTensor<float> dvGm;

    SoftMaxTiling softmaxTilingData;

    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t ENABLE = 1;

    constexpr static uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {4, 5, 6};
    static constexpr uint64_t SYNC_C1_V1_FLAG[3] = {1, 2, 3};
    static constexpr uint64_t SYNC_C2_V1_FLAG[3] = {7, 8, 9};

    float keepProb;
    float scaleValue;
    int64_t s1Token;
    int64_t s2Token;
    int64_t actualCalcS1Token;
    int64_t actualCalcS2Token;
    uint32_t sparseMode;
    bool dropBitMode;

    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t value_d;
    int64_t dAlign;
    int64_t value_dAlign;
    int64_t attenMaskDimS2;

    uint32_t baseMN;
    uint32_t cubeBaseMN;

    int64_t s1Outer;
    uint32_t s1CvInner;
    uint32_t s1CvTail;
    int64_t s2Outer;
    uint32_t s2CvInner;
    uint32_t s2CvTail;

    int64_t sfmgOffset = 0;
    uint32_t preS1Idx = -1;

    int64_t baseIdx{0};
    int64_t bDimIdx{0};
    int64_t n2DimIdx{0};
    int64_t gDimIdx{0};
    int64_t s1oDimIdx{0};
    int64_t s2oCvDimIdx{0};

    int32_t isStart = 1;
    uint32_t pingpongIdx = 1;
    int32_t vecLoopStart;
    int32_t vecLoopEnd;

    uint32_t s1VecLoop = 0;
    uint32_t s1VecSize = 0;
    uint32_t s1ExtendSubGraph = 0;
    uint32_t s2Extend = 0;
    uint32_t s2ExtendAlign = 0;
    uint32_t s2VecLoop = 0;
    uint32_t s2VecSize = 0;

    int64_t dqOutBase{0};
    int64_t kvOutBase{0};
    int64_t dqOutIdx{0};
    int64_t kvOutIdx{0};
    int64_t dqOutArr[24];
    int64_t kvOutArr[24];

    int64_t blockStartIdx = 0;

    int64_t bandIdx = 0;

    constexpr static uint32_t T2Begin = 0;
    constexpr static uint32_t T1Begin = 33 * 1024;
    constexpr static uint32_t BoolBegin = 50 * 1024;
    constexpr static uint32_t T2BlockBegin = 58 * 1024;
    constexpr static uint32_t DbBegin = 74 * 1024;

    constexpr static uint32_t DTYPE_FACTOR = sizeof(T2) / sizeof(T1);
    constexpr static uint32_t cal_block_num = 32 / sizeof(T2);
    constexpr static uint32_t cal_repeat_num = 256 / sizeof(T2);
    constexpr static uint32_t input_block_num = 32 / sizeof(T1);
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t VEC_REPEAT = 8;
    constexpr static uint32_t PREFIX_COMPRESS_CAUSAL_S_SIZE = 2048;
    constexpr static uint32_t PREFIX_COMPRESS_ALL_MASK_S1_SIZE = 1024;
    constexpr static int64_t GM_DOUBLE_BUFFER = 2;
    constexpr static int64_t TMP_UB_OFFSET = 148 * 1024;
    constexpr static int64_t SFMG_UB_OFFSET = (148 + 33) * 1024;
    constexpr static int64_t TMP_UB_SIZE = 33 * 1024;
    constexpr static int64_t SFMG_UB_SIZE = 8 * 1024;
    constexpr static int64_t TOTAL_SIZE = 189 * 1024;

    constexpr static uint32_t MMAD_BASE_SIZE = 128;
    constexpr static uint32_t S_BASE_SIZE = 512;
    constexpr static uint32_t S_SPLITT_SIZE = 256;
    constexpr static uint32_t L1_CACHE_CAPACITY_LIMIT = 14;
    constexpr static uint32_t DIM_64 = 64;
    constexpr static uint32_t VEC_S2_LEN = 256;
    constexpr static int8_t OUTIDX = -1;
    enum class AttenMaskCompress
    {
        Empty = 0,
        PreOnly = 1,
        NextOnly = 2,
        All = 3
    };
    AttenMaskCompress AttenBandMode = AttenMaskCompress::All;

    CATLASS_DEVICE
    BlockEpilogue(
        Arch::Resource<ArchTag>& resource, AscendC::TPipe* pipe_in, __gm__ uint8_t* query, __gm__ uint8_t* key,
        __gm__ uint8_t* value, __gm__ uint8_t* dx, __gm__ uint8_t* drop_mask, __gm__ uint8_t* atten_mask,
        __gm__ uint8_t* forward_res, __gm__ uint8_t* softmax_max, __gm__ uint8_t* softmax_sum,
        __gm__ uint8_t* softmax_lse, __gm__ uint8_t* actual_seq_qlen, __gm__ uint8_t* actual_seq_kvlen,
        __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* workspace,
        const Params& params, TBuf<>& buf)
    {
        keyGm.SetGlobalBuffer((__gm__ T1*)key);
        valueGm.SetGlobalBuffer((__gm__ T1*)value);
        dxGm.SetGlobalBuffer((__gm__ T1*)dx);
        queryGm.SetGlobalBuffer((__gm__ T1*)query);
        forwardResGm.SetGlobalBuffer((__gm__ T1*)forward_res);
        if (drop_mask != nullptr) {
            dropMaskGm.SetGlobalBuffer((__gm__ uint8_t*)drop_mask);
            isDrop = 1;
        }
        if (atten_mask != nullptr) {
            attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t*)atten_mask);
            isAttenMask = 1;
        }
        softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmax_lse);

        cBlockIdx = GetBlockIdx();
        cCubeBlockIdx = cBlockIdx / 2;
        cSubIdx = cBlockIdx % 2;

        // set softmax tilingdata
        softmaxTilingData = params.softmaxTilingData;

        coreNum = static_cast<uint32_t>(params.coreNum);
        cubeCoreNum = coreNum / 2;
        vecBlockNum = coreNum / 3;

        b = params.batch;
        n2 = params.kvHeadNum;
        g = params.g;
        s1 = params.qSeqlen;
        s2 = params.kvSeqlen;
        d = params.qkHeadDim;
        value_d = params.vHeadDim;
        dAlign = (d + 15) / 16 * 16;
        value_dAlign = (value_d + 15) / 16 * 16;

        attenMaskDimS2 = 2048;

        s1Token = params.s1Token;
        s2Token = params.s2Token;
        actualCalcS1Token = s1Token;
        actualCalcS2Token = s2Token;
        sparseMode = params.sparseMode;

        s1Outer = params.s1Outer;
        s1CvInner = params.s1CvInner;
        s1CvTail = params.s1CvTail;
        s2Outer = params.s2Outer;
        s2CvInner = params.s2CvInner;
        s2CvTail = s2 - (s2Outer - 1) * s2CvInner;

        baseMN = params.baseMN;
        cubeBaseMN = s1CvInner * s2CvInner;

        actual_seq_qlen_addr = actual_seq_qlen;
        actual_seq_kvlen_addr = actual_seq_kvlen;

        dropBitMode = s2 % 8 == 0;

        keepProb = params.keepProb;
        scaleValue = params.scaleValue;

        int64_t sfmgOutputSize = b * n2 * g * s1 * 8;
        if constexpr (INPUT_LAYOUT == TND) {
            int64_t seqS2Len = 0;
            seqS2Len = ((__gm__ int64_t*)actual_seq_kvlen)[0];
            dropBitMode = (seqS2Len % 8 == 0);
            for (int64_t i = 0; i + 1 < b; i++) {
                seqS2Len = ((__gm__ int64_t*)actual_seq_kvlen)[i + 1] - ((__gm__ int64_t*)actual_seq_kvlen)[i];
                dropBitMode = (dropBitMode && (seqS2Len % 8 == 0));
            }
            sfmgOutputSize = ((__gm__ int64_t*)actual_seq_qlen)[b - 1] * n2 * g * 8;
        }

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + params.dqWorkSpaceOffset / sizeof(float));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + params.dkWorkSpaceOffset / sizeof(float));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + params.dvWorkSpaceOffset / sizeof(float));

        sfmgWorkspaceGm.SetGlobalBuffer((__gm__ T2*)workspace + params.sfmgPreBeginAddr / sizeof(T2));
        int64_t workspaceOffsets =
            (params.sfmgPreBeginAddr + sfmgOutputSize * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

        uint32_t matmulWorkspaceSize = cubeBaseMN * sizeof(float);
        mm1WorkspaceGm.SetGlobalBuffer(
            (__gm__ T2*)(workspace + workspaceOffsets + cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));
        mm2WorkspaceGm.SetGlobalBuffer((__gm__ T2*)(workspace + workspaceOffsets +
                                                    cubeCoreNum * matmulWorkspaceSize * GM_DOUBLE_BUFFER +
                                                    cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        dropWorkSpaceGm.SetGlobalBuffer((__gm__ T1*)(workspace + workspaceOffsets +
                                                     cubeCoreNum * matmulWorkspaceSize * GM_DOUBLE_BUFFER +
                                                     cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        mulWorkSpaceGm.SetGlobalBuffer(
            (__gm__ T1*)(workspace + workspaceOffsets + cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

        pipe_in->InitBuffer(buf, TOTAL_SIZE);
        unifiedBuffer = buf;
        AscendC::SyncAll();
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {}

    CATLASS_DEVICE
    void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t& actualSeqQlen, int64_t& actualSeqKvlen)
    {
        if (unlikely(bIdx == 0)) {
            actualSeqQlen = ((__gm__ int64_t*)actual_seq_qlen_addr)[0];
            actualSeqKvlen = ((__gm__ int64_t*)actual_seq_kvlen_addr)[0];
        } else {
            actualSeqQlen =
                ((__gm__ int64_t*)actual_seq_qlen_addr)[bIdx] - ((__gm__ int64_t*)actual_seq_qlen_addr)[bIdx - 1];
            actualSeqKvlen =
                ((__gm__ int64_t*)actual_seq_kvlen_addr)[bIdx] - ((__gm__ int64_t*)actual_seq_kvlen_addr)[bIdx - 1];
        }
        return;
    }

    CATLASS_DEVICE
    void CopyInAttenMaskBool(
        LocalTensor<uint8_t>& dstTensor, int64_t attenMaskOffset, uint32_t s1Extend, uint32_t s2Extend)
    {
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = s1Extend;
        intriParams.blockLen = s2Extend * sizeof(uint8_t);
        intriParams.srcStride = (attenMaskDimS2 - s2Extend) * sizeof(uint8_t);
        intriParams.dstStride = 0;
        intriParams.rsv = 0;
        AscendC::DataCopyPad(dstTensor, attenMaskU8Gm[attenMaskOffset], intriParams, {false, 0, 0, 0});
    }

    CATLASS_DEVICE
    void CalcAttenMaskBool(
        LocalTensor<T2>& dstTensor, LocalTensor<uint8_t> srcTensor, uint32_t s1Extend, uint32_t s2Extend,
        uint8_t maskType = 0)
    {
        LocalTensor<uint8_t> tmpUbBuffer =
            unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);

        T2 scalar;
        if constexpr (AscendC::IsSameType<T2, float>::value) {
            uint32_t tmp = 0xFF7FFFFF;
            scalar = *((float*)&tmp);
        } else {
            uint16_t tmp = 0xFBFF;
            scalar = *((half*)&tmp);
        }

        AscendC::SelectWithBytesMaskShapeInfo info;
        info.firstAxis = s1Extend;
        info.srcLastAxis = s2Extend;
        info.maskLastAxis = (s2Extend * sizeof(uint8_t) + 31) / 32 * 32 / sizeof(uint8_t);
        dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
        srcTensor.SetSize(info.firstAxis * info.maskLastAxis);
        if (maskType == 0) {
            AscendC::SelectWithBytesMask(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
        } else {
            AscendC::SelectWithBytesMask(dstTensor, scalar, dstTensor, srcTensor, tmpUbBuffer, info);
        }
    }

    CATLASS_DEVICE
    void CalcAttenMaskOffset(int64_t& attenMaskOffset, const int64_t delta, uint32_t s1VSize, uint32_t s2VSize)
    {
        if (delta == 0) {
            attenMaskOffset = 0;
        } else if (delta < 0) {
            if (-delta > s1VSize) {
                attenMaskOffset = s1VSize;
            } else {
                attenMaskOffset = -delta;
            }
        } else {
            if (delta > s2VSize) {
                attenMaskOffset = s2VSize * attenMaskDimS2;
            } else {
                attenMaskOffset = delta * attenMaskDimS2;
            }
        }
    }

    CATLASS_DEVICE
    void CalcAttenBandMode(int64_t compressMode, int64_t causal_delta, DBParams& dbParam)
    {
        int64_t actualS1Len;
        int64_t actualS2Len;
        if (compressMode == 1 || compressMode == 2 || compressMode == 3 || sparseMode == 7 || sparseMode == 8) {
            int64_t next_delta = causal_delta;
            int64_t pre_delta = causal_delta - INT32_MAX - 1;
            if (compressMode == 2) {
                next_delta = causal_delta - s1 + s2;
            } else {
                next_delta = causal_delta + actualCalcS2Token;
                pre_delta = causal_delta - actualCalcS1Token - 1;
            }

            bool NoNext = (next_delta - s2Extend >= 0);
            bool NoPre = (pre_delta + 1 + s1ExtendSubGraph <= 0);

            if (NoNext && NoPre) {
                AttenBandMode = AttenMaskCompress::Empty;
            } else if (NoNext && !NoPre) {
                AttenBandMode = AttenMaskCompress::PreOnly;
            } else if (!NoNext && NoPre) {
                AttenBandMode = AttenMaskCompress::NextOnly;
            } else {
                AttenBandMode = AttenMaskCompress::All;
            }
        }
    }

    CATLASS_DEVICE
    void CalcAttenMaskOffsetWithSparseMode(
        int64_t& attenMaskOffset, int64_t& attenMaskOffset2, uint32_t s1VSize, uint32_t s2VSize, int64_t curS1Idx,
        uint32_t s2VBegin, bool& canSimplify, DBParams& dbParam)
    {
        uint64_t compressMode = 0;
        int64_t causal_delta =
            static_cast<int64_t>(dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) - static_cast<int64_t>(s2VBegin);
        CalcAttenBandMode(compressMode, causal_delta, dbParam);
        if (compressMode == 1) {
            CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
            return;
        }

        if (compressMode == 2) {
            causal_delta = causal_delta - s1 + s2;
            CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
            return;
        }

        if (compressMode == 3) {
            int64_t pre_delta = causal_delta - actualCalcS1Token - 1;
            CalcAttenMaskOffset(attenMaskOffset2, pre_delta, s1VSize, s2VSize);
            int64_t next_delta = causal_delta + actualCalcS2Token;
            CalcAttenMaskOffset(attenMaskOffset, next_delta, s1VSize, s2VSize);
            return;
        }

        int64_t attenMaskShapeType = 0;
        if (attenMaskShapeType == 0) {
            attenMaskOffset = (static_cast<int64_t>(dbParam.s1oIdx) * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
        } else if (attenMaskShapeType == 1) {
            attenMaskOffset = (dbParam.bIdx * s1 + dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
        } else {
            attenMaskOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 +
                               dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) *
                                  s2 +
                              s2VBegin;
        }
    }

    CATLASS_DEVICE
    void CopyInSoftMax(LocalTensor<float>& dstTensor, uint32_t s1Extend, uint32_t softMaxOffset)
    {
        if constexpr (INPUT_LAYOUT == TND) {
            AscendC::DataCopyPad(
                dstTensor, softmaxLseGm[softMaxOffset], {1, static_cast<uint16_t>(s1Extend * 4), 0, 0},
                {false, 0, 0, 0});
        } else { // BSN
            int64_t nheads = n2 * g;
            int64_t lseRepeatTimes = (s1Extend + 7) / 8 * 8;
            AscendC::DataCopyExtParams lseCopyParam;
            lseCopyParam.blockCount = s1Extend;
            lseCopyParam.blockLen = sizeof(float);
            lseCopyParam.srcStride = (nheads - 1) * sizeof(float);
            lseCopyParam.dstStride = 0;
            lseCopyParam.rsv = 0;
            AscendC::DataCopyPad(dstTensor, softmaxLseGm[softMaxOffset], lseCopyParam, {false, 0, 0, 0});
        }

        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::Brcb(dstTensor[s1Extend * 8], dstTensor, static_cast<uint8_t>((s1Extend + 7) / 8), {1, 8});
    }

    CATLASS_DEVICE
    void CalcSoftMax(
        LocalTensor<float>& dstTensor, LocalTensor<float>& src0Tensor, LocalTensor<float>& src1Tensor,
        uint32_t s1Extend, uint32_t s2Extend, uint32_t s2ExtendAlign, const SoftMaxTiling& tiling)
    {
        uint32_t sub_block_count = (s2Extend + cal_repeat_num - 1) / cal_repeat_num;

        for (uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
            uint32_t subMaskCount =
                (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * cal_repeat_num) : cal_repeat_num;
            AscendC::Sub(
                dstTensor[subIdx * cal_repeat_num], src0Tensor[subIdx * cal_repeat_num], src1Tensor[s1Extend * 8],
                subMaskCount, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8), 1});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(
                dstTensor[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num], subMaskCount, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8)});
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    CATLASS_DEVICE
    void SubGrapA(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx, DBParams& dbParam, event_t mte2WaitMte3A)
    {
        pingpongIdx = dbParam.taskId % 2;
        s2Extend = (curS2Idx == s2VecLoop - 1) ? (dbParam.s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
        s2ExtendAlign = (s2Extend + 15) / 16 * 16;
        uint32_t s2VBegin = dbParam.s2oIdx * s2CvInner + curS2Idx * s2VecSize;

        uint32_t ubBufferOffset = 0;

        if (curIdx > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        }

        LocalTensor<float> vecInBuffer3 =
            unifiedBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), ubBufferOffset + T2BlockBegin);

        int64_t softMaxOffset = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            if (dbParam.bIdx > 0) {
                softMaxOffset = ((__gm__ int64_t*)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g;
            }
            softMaxOffset +=
                ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len + dbParam.s1oIdx * s1CvInner +
                 curS1Idx * s1VecSize);
        } else {
            softMaxOffset =
                ((dbParam.bIdx * s1 + dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) * n2 + dbParam.n2Idx) * g +
                dbParam.gIdx; // bsn
        }
        CopyInSoftMax(vecInBuffer3, s1ExtendSubGraph, softMaxOffset);

        LocalTensor<uint8_t> attenMaskUbuint8 =
            unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + BoolBegin);
        int64_t attenMaskOffsetPre = 0;
        bool prefixCompressCanSimplify = false;
        if (isAttenMask == ENABLE) {
            int64_t attenMaskOffset = 0;
            
            CalcAttenMaskOffsetWithSparseMode(
                attenMaskOffset, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend, curS1Idx, s2VBegin,
                prefixCompressCanSimplify, dbParam);
            
            if (AttenBandMode == AttenMaskCompress::All || AttenBandMode == AttenMaskCompress::NextOnly) {
                CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffset, s1ExtendSubGraph, s2Extend);
            } else if (AttenBandMode == AttenMaskCompress::PreOnly) {
                CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);
            }
        }


        LocalTensor<float> vecClc2Buffer =
            unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);

        // only ND
        if (s2VecLoop == 1) {
            AscendC::DataCopy(
                vecClc2Buffer, mm2WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * s2ExtendAlign],
                s1ExtendSubGraph * s2ExtendAlign);
        } else {
            AscendC::DataCopyPad(
                vecClc2Buffer,
                mm2WorkspaceGm
                    [pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * dbParam.s2CvExtendAlign + curS2Idx * s2VecSize],
                {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                 static_cast<uint16_t>((dbParam.s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                {false, 0, 0, 0});
        }
        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(vecClc2Buffer, vecClc2Buffer, (T2)scaleValue, s1ExtendSubGraph * s2ExtendAlign);

        if (isAttenMask == ENABLE) {
            AscendC::PipeBarrier<PIPE_V>();

            if (AttenBandMode == AttenMaskCompress::All || AttenBandMode == AttenMaskCompress::NextOnly) {
                CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign);
            } else if (AttenBandMode == AttenMaskCompress::PreOnly) {
                CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign, 1);
            }
        }

        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(T2), DbBegin);

        CalcSoftMax(
            simpleSoftmaxResBuf, vecClc2Buffer, vecInBuffer3, s1ExtendSubGraph, s2Extend, s2ExtendAlign,
            softmaxTilingData);
        LocalTensor<T2> vecDropBuffer = simpleSoftmaxResBuf;
        
        LocalTensor<T1> vecCopyOutBuffer = vecDropBuffer.template ReinterpretCast<T1>();
        if constexpr (!AscendC::IsSameType<T1, float>::value) {
            vecCopyOutBuffer = unifiedBuffer.GetWithOffset<T1>(17 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(vecCopyOutBuffer, vecDropBuffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
        }
        int64_t copyOutOffset = 0;
        DataCopyParams copyOutParam;
        // only support ND
        copyOutOffset = pingpongIdx * cubeBaseMN * DTYPE_FACTOR +
                        curS1Idx * s1VecSize * dbParam.s2CvExtendAlign * DTYPE_FACTOR + curS2Idx * s2VecSize;
        copyOutParam = {
            static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
            static_cast<uint16_t>((dbParam.s2CvExtendAlign * DTYPE_FACTOR - s2ExtendAlign) * sizeof(T1))};
        event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::DataCopyPad(dropWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

        if (curIdx < vecLoopEnd - vecLoopStart - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        }
    }

    CATLASS_DEVICE
    void SubGrapB(
        int64_t curIdx, int64_t s1VecLoop, int64_t s2VecLoop, int64_t curS1Idx, int64_t curS2Idx, DBParams& dbParam,
        event_t mte2WaitMte3B)
    {
        pingpongIdx = dbParam.taskId % 2;
        uint32_t ubBufferOffset = DbBegin;
        s2Extend = (curS2Idx == s2VecLoop - 1) ? (dbParam.s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
        s2ExtendAlign = (s2Extend + 15) / 16 * 16;

        if (curIdx > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }

        if (preS1Idx != curS1Idx) {
            preS1Idx = curS1Idx;
            LocalTensor<float> sfmgClc3 =
                unifiedBuffer.GetWithOffset<float>(SFMG_UB_SIZE / sizeof(float), SFMG_UB_OFFSET);
            AscendC::DataCopy(sfmgClc3, sfmgWorkspaceGm[sfmgOffset + curS1Idx * s1VecSize * 8], s1ExtendSubGraph * 8);
        }

        LocalTensor<T2> vecClc1Buffer =
            unifiedBuffer.GetWithOffset<T2>(33 * 1024 / sizeof(T2), ubBufferOffset + T1Begin);
        LocalTensor<T2> dyvBuffer = unifiedBuffer.GetWithOffset<T2>(33 * 1024 / sizeof(T2), TMP_UB_OFFSET);
        // only ND
        if (s2VecLoop == 1) {
            AscendC::DataCopy(
                vecClc1Buffer, mm1WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * s2ExtendAlign],
                s1ExtendSubGraph * s2ExtendAlign);
        } else {
            AscendC::DataCopyPad(
                vecClc1Buffer,
                mm1WorkspaceGm
                    [pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * dbParam.s2CvExtendAlign + curS2Idx * s2VecSize],
                {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                 static_cast<uint16_t>((dbParam.s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                {false, 0, 0, 0});
        }

        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t sub_block_cout = (s2ExtendAlign + cal_repeat_num - 1) / cal_repeat_num;
        LocalTensor<float> sfmgClc3 = unifiedBuffer.GetWithOffset<float>(SFMG_UB_SIZE / sizeof(float), SFMG_UB_OFFSET);

        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
            uint32_t subMaskCout =
                (subIdx == sub_block_cout - 1) ? (s2ExtendAlign - subIdx * cal_repeat_num) : cal_repeat_num;
            AscendC::Sub(
                vecClc1Buffer[subIdx * cal_repeat_num], vecClc1Buffer[subIdx * cal_repeat_num], sfmgClc3, subMaskCout,
                s1ExtendSubGraph,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8), 1});
        }

        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), DbBegin);

        AscendC::Mul(vecClc1Buffer, vecClc1Buffer, simpleSoftmaxResBuf, s1ExtendSubGraph * s2ExtendAlign);
        LocalTensor<T1> vecCopyOutBuffer = vecClc1Buffer.template ReinterpretCast<T1>();
        if constexpr (!AscendC::IsSameType<T1, float>::value) {
            vecCopyOutBuffer = unifiedBuffer.GetWithOffset<T1>(17 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(vecCopyOutBuffer, vecClc1Buffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
        }

        int64_t copyOutOffset = 0;
        DataCopyParams copyOutParam;
        // only ND
        copyOutOffset = pingpongIdx * cubeBaseMN * DTYPE_FACTOR +
                        curS1Idx * s1VecSize * dbParam.s2CvExtendAlign * DTYPE_FACTOR + curS2Idx * s2VecSize;
        copyOutParam = {
            static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
            static_cast<uint16_t>((dbParam.s2CvExtendAlign * DTYPE_FACTOR - s2ExtendAlign) * sizeof(T1))};
        event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        AscendC::DataCopyPad(mulWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

        if (curIdx < vecLoopEnd - vecLoopStart - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }
    }

    CATLASS_DEVICE
    void operator()(DBParams& dbParam)
    {
        int64_t actualS1Len;
        int64_t actualS2Len;

        s2VecSize = dbParam.s2CvExtend > VEC_S2_LEN ? VEC_S2_LEN : dbParam.s2CvExtend;
        s2VecLoop = s2VecSize == 0 ? 0 : CeilDiv(dbParam.s2CvExtend, s2VecSize);

        uint32_t s2AlignFactor = BLOCK_SIZE / 2;
        if (isAttenMask == ENABLE) {
            s2AlignFactor = BLOCK_SIZE / sizeof(uint8_t);
        }

        if (s2AlignFactor == 0) {
            return;
        } else {
            s1VecSize = baseMN / ((s2VecSize + s2AlignFactor - 1) / s2AlignFactor * s2AlignFactor);
        }
        s1VecSize = s1VecSize > dbParam.s1CvExtend ? dbParam.s1CvExtend : s1VecSize;
        s1VecSize = s1VecSize > 128 ? 128 : s1VecSize;
        s1VecLoop = s1VecSize == 0 ? 0 : CeilDiv(dbParam.s1CvExtend, s1VecSize);
        if constexpr (INPUT_LAYOUT == TND) {
            GetSeqQlenKvlenByBidx(dbParam.bIdx, dbParam.actualS1Len, dbParam.actualS2Len);
            int64_t bSSOffset = 0;
            int64_t s2Accu = 0;
            for (int64_t bidx = 0; bidx < dbParam.bIdx; bidx++) {
                GetSeqQlenKvlenByBidx(bidx, actualS1Len, actualS2Len);
                bSSOffset += actualS1Len * actualS2Len;
                s2Accu += actualS2Len;
            }
        }
        

        sfmgOffset = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            if (dbParam.bIdx > 0) {
                sfmgOffset = n2 * g * ((__gm__ int64_t*)actual_seq_qlen_addr)[dbParam.bIdx - 1] * 8;
            }
            sfmgOffset += ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len + dbParam.s1oIdx * s1CvInner) * 8;
        } else {
            sfmgOffset =
                (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 + dbParam.s1oIdx * s1CvInner) * 8;
        }

        int32_t loopSize = s1VecLoop * s2VecLoop;
        int32_t halfLoop = 0;

        halfLoop = (s1VecLoop / 2) * s2VecLoop;

        vecLoopStart = cSubIdx ? halfLoop : 0;
        vecLoopEnd = cSubIdx ? loopSize : halfLoop;
        preS1Idx = -1;
        event_t mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        for (int32_t i = vecLoopStart, loopCnt = 0; i < vecLoopEnd; i++, loopCnt++) {
            int32_t curS1Idx;
            int32_t curS2Idx;
            curS1Idx = i / s2VecLoop;
            curS2Idx = i % s2VecLoop;

            s1ExtendSubGraph =
                (curS1Idx == s1VecLoop - 1) ? (dbParam.s1CvExtend - (s1VecLoop - 1) * s1VecSize) : s1VecSize;

            event_t mte2WaitMte3A = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            event_t mte2WaitMte3B = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            SubGrapA(loopCnt, curS1Idx, curS2Idx, dbParam, mte2WaitMte3A);
            SubGrapB(loopCnt, s1VecLoop, s2VecLoop, curS1Idx, curS2Idx, dbParam, mte2WaitMte3B);

            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }
    }
};

} // namespace Catlass::Epilogue::Block

#endif
