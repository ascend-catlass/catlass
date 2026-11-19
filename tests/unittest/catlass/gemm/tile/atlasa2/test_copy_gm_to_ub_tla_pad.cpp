/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * AtlasA2 CopyGm2UbTla RowMajor: forward DataCopyPadExtParams (parity with Ascend950).
 */

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201

#include <gtest/gtest.h>
#include "stub/ascendc_test_fixture.h"
#include "stub/kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"

#include "tla/tensor.hpp"
#include "stub/ascendc_logger.h"

using namespace Catlass;
using namespace Catlass::Epilogue::Tile;
using namespace Catlass::Test;

class TileCopyGmToUbTlaPadAtlasA2 : public AscendCTest {};

using GmToUbRmCoordZero = tla::Coord<tla::Int<0>, tla::Int<0>>;

TEST_F(TileCopyGmToUbTlaPadAtlasA2, RowMajorExtParamsForwarded)
{
    using Element = float;
    using Layout = Catlass::detail::TagToLayout_t<Element, layout::RowMajor>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<Element>, Layout, GmToUbRmCoordZero, AscendC::TPosition::GM>;
    using TensorDst =
        tla::Tensor<AscendC::LocalTensor<Element>, Layout, GmToUbRmCoordZero, AscendC::TPosition::VECCALC>;

    constexpr uint32_t kRows = 2;
    constexpr uint32_t kCols = 6;
    constexpr uint8_t kRightPad = 2;

    AscendC::GlobalTensor<Element> gmSrc;
    AscendC::LocalTensor<Element> ubDst;
    auto layoutSrc = tla::MakeLayout<Element, layout::RowMajor>(kRows, kCols);
    auto layoutDst = tla::MakeLayout<Element, layout::RowMajor>(kRows, kCols);
    TensorSrc tensorGm(gmSrc, layoutSrc);
    TensorDst tensorUb(ubDst, layoutDst);

    AscendC::DataCopyPadExtParams<Element> padParams{true, 0, kRightPad, static_cast<Element>(0)};

    CopyGm2UbTla<Arch::AtlasA2, TensorSrc, TensorDst> copyGmToUb;
    copyGmToUb(tensorUb, tensorGm, padParams);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1U);
    const auto* got = logs[0].GetArgsAt(3).template Value<AscendC::DataCopyPadExtParams<Element>>();
    ASSERT_NE(got, nullptr);
    EXPECT_TRUE(got->isPad);
    EXPECT_EQ(got->rightPadding, kRightPad);
}

#endif // CATLASS_ARCH == 2201
