/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_POST_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_POST_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

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

template <class OutputType_, class UpdateType_, class InputType_>
class BlockEpilogue<EpilogueAtlasA2FAGPost, OutputType_, UpdateType_, InputType_> {
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPost;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using InputType = InputType_;

    // kernel-parsed tiling fields consumed by this epilogue (no raw tiling parsing here)
    struct Params {
        uint64_t dqWorkSpaceOffset;
        uint64_t dkWorkSpaceOffset;
        uint64_t dvWorkSpaceOffset;
        int64_t qSize;
        int64_t kvSize;
        int64_t coreNum;
        float scaleValue;
    };

    constexpr static uint32_t POST_BUFFER_NUM = 1;

    AscendC::TPipe* pipe;
    TBuf<QuePosition::VECIN> inBuffer;
    TBuf<QuePosition::VECOUT> outBuffer;

    // input
    AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    // output
    AscendC::GlobalTensor<InputType> dqGm, dkGm, dvGm;

    int64_t cBlockIdx;
    // query
    int64_t ubBaseSize;
    int64_t qPostBlockFactor;
    uint64_t qPostBlockTotal;
    int64_t qPostBaseNum;
    int64_t qPostTailNum;
    int64_t kvPostBlockFactor;
    uint64_t kvPostBlockTotal;
    int64_t kvPostBaseNum;
    int64_t kvPostTailNum;
    float scaleValue;

    CATLASS_DEVICE
    BlockEpilogue(
        Arch::Resource<ArchTag>& resource, AscendC::TPipe* pipe_in, __gm__ uint8_t* dq, __gm__ uint8_t* dk,
        __gm__ uint8_t* dv, __gm__ uint8_t* workspace, const Params& params)
    {
        cBlockIdx = GetBlockIdx();
        pipe = pipe_in;

        scaleValue = params.scaleValue;

        dqGm.SetGlobalBuffer((__gm__ InputType*)dq);
        dkGm.SetGlobalBuffer((__gm__ InputType*)dk);
        dvGm.SetGlobalBuffer((__gm__ InputType*)dv);

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + params.dqWorkSpaceOffset / sizeof(float));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + params.dkWorkSpaceOffset / sizeof(float));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + params.dvWorkSpaceOffset / sizeof(float));

        // compute tiling
        constexpr static uint32_t POST_COEX_NODE = 3;
        constexpr static uint32_t WORKSPACE_NUM_ALIGN = 256;
        uint32_t curPostCoexNode = POST_COEX_NODE;
        uint32_t ubSize = ArchTag::UB_SIZE;
        ubBaseSize = ubSize / curPostCoexNode / POST_BUFFER_NUM;
        ubBaseSize = ubBaseSize / WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN; // align

        // dq
        qPostBaseNum = ubBaseSize / sizeof(float);
        qPostBlockTotal = params.qSize;

        int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
        int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;

        qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
        qPostBlockFactor = (qPostBlockOuterTotal + params.coreNum - 1) / params.coreNum;

        // dkv
        kvPostBaseNum = qPostBaseNum;
        kvPostBlockTotal = params.kvSize;

        int64_t kvPostTailNumTmp = kvPostBlockTotal % kvPostBaseNum;
        int64_t kvPostBlockOuterTotal = (kvPostBlockTotal + kvPostBaseNum - 1) / kvPostBaseNum;

        kvPostTailNum = kvPostTailNumTmp == 0 ? kvPostBaseNum : kvPostTailNumTmp;
        kvPostBlockFactor = (kvPostBlockOuterTotal + params.coreNum - 1) / params.coreNum;

        pipe->InitBuffer(inBuffer, ubBaseSize * 2);
        pipe->InitBuffer(outBuffer, ubBaseSize);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {}

    CATLASS_DEVICE
    void operator()()
    {
        uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
        uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;

        if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
            qEnd = qPostBlockTotal;
        }
        event_t Mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        for (uint64_t i = qBegin; i < qEnd; i = i + qPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inBuffer.Get<float>();
            AscendC::LocalTensor<InputType> vecOut = outBuffer.Get<InputType>();
            uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
            DataCopy(vecIn, dqWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B

            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            Muls(vecIn, vecIn, scaleValue, dataSize);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            event_t Mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);

            DataCopy(dqGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        // init k
        uint64_t kvBegin = cBlockIdx * kvPostBlockFactor * kvPostBaseNum;
        uint64_t kvEnd = (cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum;
        if (((cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum) > kvPostBlockTotal) {
            kvEnd = kvPostBlockTotal;
        }

        for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inBuffer.Get<float>();
            AscendC::LocalTensor<InputType> vecOut = outBuffer.Get<InputType>();
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            DataCopy(vecIn, dkWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

            Muls(vecIn, vecIn, scaleValue, dataSize);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);

            event_t Mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);

            DataCopy(dkGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // init v
        for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inBuffer.Get<float>();
            AscendC::LocalTensor<InputType> vecOut = outBuffer.Get<InputType>();
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            DataCopy(vecIn, dvWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            event_t Mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Mte3WaitV);

            DataCopy(dvGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            if (i + kvPostBaseNum < kvEnd) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(Mte2WaitMte3);
            }
        }
    }
};
} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_POST_HPP
