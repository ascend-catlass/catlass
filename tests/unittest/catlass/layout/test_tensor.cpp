/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "catlass/layout/tensor.hpp"

using namespace Catlass;
using namespace Catlass::layout;

TEST(TensorLayoutTest, Nc1hwc0StridesUseLongIndex)
{
    NC1HWC0 fmap(2, 1024, 512, 512, 16);
    Conv2dFmapCoord firstBatch(uint32_t(1), uint32_t(0), uint32_t(0), uint32_t(0), uint32_t(0));
    EXPECT_EQ(fmap.stride(0), 4294967296LL);
    EXPECT_EQ(fmap.stride(1), 4194304LL);
    EXPECT_EQ(fmap.stride(2), 8192LL);
    EXPECT_EQ(fmap.GetOffset(firstBatch), 4294967296LL);

    NC1HWC0 control(2, 1023, 512, 512, 16);
    EXPECT_EQ(control.stride(0), 4290772992LL);
    EXPECT_EQ(control.GetOffset(firstBatch), 4290772992LL);
}

TEST(TensorLayoutTest, FilterStridesUseLongIndex)
{
    CI1KHKWCOCI0 filter(2, 1024, 512, 512, 16);
    Conv2dFilterCoord firstCin1(uint32_t(1), uint32_t(0), uint32_t(0), uint32_t(0), uint32_t(0));
    EXPECT_EQ(filter.stride(0), 4294967296LL);
    EXPECT_EQ(filter.stride(1), 4194304LL);
    EXPECT_EQ(filter.stride(2), 8192LL);
    EXPECT_EQ(filter.GetOffset(firstCin1), 4294967296LL);

    CI1KHKWCOCI0 control(2, 1023, 512, 512, 16);
    EXPECT_EQ(control.stride(0), 4290772992LL);
    EXPECT_EQ(control.GetOffset(firstCin1), 4290772992LL);
}
