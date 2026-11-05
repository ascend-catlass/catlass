/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "catlass/gemm/block/block_scheduler_syrk.h"

using Catlass::GemmCoord;
using Catlass::MatrixCoord;
using Catlass::Gemm::Block::SyrkBlockScheduler;

TEST(SyrkBlockScheduler, RowOrderBatchesAndTails)
{
    for (uint32_t tile : {64u, 128u, 256u}) {
        for (uint32_t m : {1u, 31u, 32u, 63u, 64u, 65u, 127u, 128u, 129u, 255u, 256u, 257u, 280u, 300u, 544u, 8192u}) {
            const SyrkBlockScheduler scheduler(GemmCoord{m, m, 513}, MatrixCoord{tile, tile});
            std::vector<GemmCoord> expected;
            for (uint32_t row = 0; row * tile < m; ++row) {
                for (uint32_t column = 0; column <= row; ++column) {
                    expected.emplace_back(row, column, 0);
                }
            }
            ASSERT_EQ(scheduler.GetCoreLoops(), expected.size());
            uint32_t task = 0;
            for (uint32_t batch = 0; batch < 3; ++batch) {
                for (const GemmCoord& coord : expected) {
                    EXPECT_EQ(scheduler.GetBatchIdx(task), batch);
                    EXPECT_EQ(scheduler.GetBlockCoord(task), coord);
                    const GemmCoord actual = scheduler.GetActualBlockShape(coord);
                    EXPECT_EQ(actual.m(), std::min(tile, m - coord.m() * tile));
                    EXPECT_EQ(actual.n(), std::min(tile, m - coord.n() * tile));
                    EXPECT_EQ(actual.k(), 513u);
                    ++task;
                }
            }
        }
    }
    const SyrkBlockScheduler empty(GemmCoord{0, 0, 513}, MatrixCoord{64u, 64u});
    EXPECT_EQ(empty.GetCoreLoops(), 0u);
}

TEST(SyrkBlockScheduler, CorePartitionSplitSlicesAndRowBands)
{
    const SyrkBlockScheduler scheduler(GemmCoord{300, 300, 513}, MatrixCoord{128u, 128u});
    const GemmCoord expected[] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {2, 0, 0}, {2, 1, 0}, {2, 2, 0}};
    const uint32_t tasks = 3 * scheduler.GetCoreLoops();
    for (uint32_t cores : {1u, 2u, 24u, 48u}) {
        // Split-K slices and row bands use the same tile order.
        for (uint32_t workPerTile : {1u, 4u, 32u}) {
            std::vector<uint32_t> visits(tasks * workPerTile, 0);
            for (uint32_t core = 0; core < cores; ++core) {
                for (uint32_t work = core; work < visits.size(); work += cores) {
                    const uint32_t task = work / workPerTile;
                    EXPECT_EQ(scheduler.GetBatchIdx(task), task / 6);
                    EXPECT_EQ(scheduler.GetBlockCoord(task), expected[task % 6]);
                    ++visits[work];
                }
            }
            EXPECT_TRUE(std::all_of(visits.begin(), visits.end(), [](uint32_t count) { return count == 1; }));
        }
    }
}
