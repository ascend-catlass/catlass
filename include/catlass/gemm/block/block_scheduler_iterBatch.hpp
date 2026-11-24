/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_ITERBATCH_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_ITERBATCH_HPP

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"

namespace Catlass::Gemm::Block {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Continuous-group scheduler for IterBatch broadcast batched matmul.
//
// The batch dimension is partitioned into consecutive L1 groups of iterBatchL1
// batches each; the group scheduled at loopIdx covers batches
//   [loopIdx * iterBatchL1, min((loopIdx + 1) * iterBatchL1, batchCount)).
// Because each group is consecutive, the kernel can hand the block a single GM
// tensor of the group's first batch and the block derives the remaining
// batches via the batch stride.
//
// The query protocol mirrors the library schedulers (block_swizzle.hpp; see
// batched_matmul_tla.hpp / batched_matmul_broadcast_tla.h for the kernel-side
// pattern): the kernel drives a standard grid-stride loop
//   for (loopIdx = GetBlockIdx(); loopIdx < GetCoreLoops(); loopIdx += GetBlockNum())
// and maps loopIdx to the group's first batch via GetBatchIdx(loopIdx). Here
// batchCount plays the role of the batch problem size and iterBatchL1 the
// batch tile width. M/N/K are not tiled on this path (single L1 tile), so the
// MN mapping entries of the library protocol (GetBlockCoord / GetActualBlockShape)
// have no counterpart.
struct IterBatchScheduler {
    /// Data members

    uint32_t batchCount;
    uint32_t iterBatchL1;
    uint32_t groupCount;

    /// Methods

    CATLASS_DEVICE
    IterBatchScheduler()
    {}

    CATLASS_DEVICE
    IterBatchScheduler(uint32_t batchCount_, uint32_t iterBatchL1_) : batchCount(batchCount_), iterBatchL1(iterBatchL1_)
    {
        // iterBatchL1 == 0 is an invalid spec: the host resolves auto values (0)
        // in ResourcePlan::Make and must reject invalid plans before launch. Guard
        // here so a launch with a tampered plan yields an empty schedule instead of
        // dividing by zero on device (device division by zero is UB).
        groupCount = (iterBatchL1 != 0) ? CeilDiv(batchCount, iterBatchL1) : 0;
    }

    /// Number of grid-stride loops; one loop computes one L1 group. Mirrors
    /// GetCoreLoops() of the library schedulers.
    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return groupCount;
    }

    /// Global batch index of the first batch of the group scheduled at loopIdx.
    /// Mirrors GetBatchIdx(taskIdx) of the library schedulers.
    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t loopIdx) const
    {
        return loopIdx * iterBatchL1;
    }

    /// Number of batches of the group scheduled at loopIdx (the last group may
    /// be a remainder).
    CATLASS_DEVICE
    uint32_t GetActualGroupBatch(uint32_t loopIdx) const
    {
        const uint32_t batchBase = loopIdx * iterBatchL1;
        return (batchBase + iterBatchL1 <= batchCount) ? iterBatchL1 : (batchCount - batchBase);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_ITERBATCH_HPP
