/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Epilogue::Block {

template <class OutputType_, class UpdateType_, class InputType_>
class BlockEpilogue<EpilogueAtlasA2FAGPre, OutputType_, UpdateType_, InputType_> {
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    // kernel-parsed tiling fields consumed by this epilogue (no raw tiling parsing here)
    struct Params {
        uint64_t dqWorkSpaceOffset;
        uint64_t dkWorkSpaceOffset;
        uint64_t dvWorkSpaceOffset;
        int64_t qSize;
        int64_t kvSize;
        int64_t coreNum;
    };

    AscendC::TPipe* pipe;
    AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;

    uint32_t cBlockIdx;
    // query
    int64_t qPreBlockFactor;
    int64_t qPreBlockTotal;
    int64_t qPreBlockTail;
    int64_t kvPreBlockFactor;
    int64_t kvPreBlockTotal;
    int64_t kvPreBlockTail;

    int64_t initdqSize;
    int64_t dqOffset;
    int64_t initdkSize;
    int64_t dkvOffset;

    CATLASS_DEVICE
    BlockEpilogue(
        Arch::Resource<ArchTag>& resource, AscendC::TPipe* pipe_in, __gm__ uint8_t* dq, __gm__ uint8_t* dk,
        __gm__ uint8_t* dv, __gm__ uint8_t* workspace, const Params& params)
    {
        cBlockIdx = AscendC::GetBlockIdx();
        pipe = pipe_in;

        // compute tiling params
        qPreBlockFactor = (params.qSize + params.coreNum - 1) / params.coreNum;
        qPreBlockTotal = (params.qSize + qPreBlockFactor - 1) / qPreBlockFactor;
        int64_t qPreTailNumTmp = params.qSize % qPreBlockFactor;
        qPreBlockTail = qPreTailNumTmp == 0 ? qPreBlockFactor : qPreTailNumTmp;

        kvPreBlockFactor = (params.kvSize + params.coreNum - 1) / params.coreNum;
        kvPreBlockTotal = (params.kvSize + kvPreBlockFactor - 1) / kvPreBlockFactor;
        int64_t kvPreTailNumTmp = params.kvSize % kvPreBlockFactor;
        kvPreBlockTail = kvPreTailNumTmp == 0 ? kvPreBlockFactor : kvPreTailNumTmp;

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + params.dqWorkSpaceOffset / sizeof(float));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + params.dkWorkSpaceOffset / sizeof(float));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + params.dvWorkSpaceOffset / sizeof(float));

        initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
        dqOffset = ((int64_t)cBlockIdx) * qPreBlockFactor;
        initdkSize = cBlockIdx == kvPreBlockTotal - 1 ? kvPreBlockTail : kvPreBlockFactor;
        dkvOffset = ((int64_t)cBlockIdx) * kvPreBlockFactor;
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {}

    CATLASS_DEVICE
    void operator()()
    {
        // process clear dq dk dv workspace
        if (g_coreType == AscendC::AIV && cBlockIdx < qPreBlockTotal) {
            AscendC::InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
        }

        if (g_coreType == AscendC::AIV && cBlockIdx < kvPreBlockTotal) {
            AscendC::InitOutput<float>(dkWorkSpaceGm[dkvOffset], initdkSize, 0);
            AscendC::InitOutput<float>(dvWorkSpaceGm[dkvOffset], initdkSize, 0);
        }
    }
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
