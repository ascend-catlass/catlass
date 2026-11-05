/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_SYRK_H
#define CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_SYRK_H

#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Block {

/// Visit lower-triangle tiles row by row, then start the next batch.
struct SyrkBlockScheduler {
    GemmCoord problemShape;
    MatrixCoord tileMN;
    uint32_t tasksPerBatch;

    CATLASS_HOST_DEVICE
    SyrkBlockScheduler(GemmCoord const& problemShape_, MatrixCoord const& tileMN_)
        : problemShape(problemShape_), tileMN(tileMN_)
    {
        uint32_t tileCount = CeilDiv(problemShape.m(), tileMN.row());
        tasksPerBatch = tileCount * (tileCount + 1) / 2;
    }

    CATLASS_HOST_DEVICE
    uint32_t GetCoreLoops() const
    {
        return tasksPerBatch;
    }

    CATLASS_HOST_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx) const
    {
        return taskIdx / tasksPerBatch;
    }

    CATLASS_HOST_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx) const
    {
        uint32_t triangleIdx = taskIdx % tasksPerBatch;
        uint32_t tileRow = 0;
        while ((tileRow + 1) * (tileRow + 2) / 2 <= triangleIdx) {
            ++tileRow;
        }
        return GemmCoord{tileRow, triangleIdx - tileRow * (tileRow + 1) / 2, 0};
    }

    CATLASS_HOST_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord const& blockCoord) const
    {
        return GemmCoord{
            Min(tileMN.row(), problemShape.m() - blockCoord.m() * tileMN.row()),
            Min(tileMN.column(), problemShape.n() - blockCoord.n() * tileMN.column()), problemShape.k()};
    }
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_SYRK_H
