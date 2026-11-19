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
 * Ascend950 CopyGm2UbTla: forward DataCopyPadExtParams (ops-style left/right/paddingValue).
 *
 * Build:  bash scripts/build.sh --tests catlass_unittest
 * Run:    ./build/tests/unittest/catlass_unittest_3510 \
 *           --gtest_filter='TileCopyGmToUbTlaPadAscend950.*'
 */

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

#include <gtest/gtest.h>
#include "stub/ascendc_test_fixture.h"
#include "stub/kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"

#include "tla/tensor.hpp"

#include "stub/ascendc_logger.h"

using namespace Catlass;
using namespace Catlass::Epilogue::Tile;
using namespace Catlass::Test;

class TileCopyGmToUbTlaPadAscend950 : public AscendCTest {};

using GmToUbVecCoordZero = tla::Coord<tla::Int<0>>;
using GmToUbRmCoordZero = tla::Coord<tla::Int<0>, tla::Int<0>>;

TEST_F(TileCopyGmToUbTlaPadAscend950, VectorRightPadAndValueForwarded)
{
    using Element = float;
    using Layout = Catlass::detail::TagToLayout_t<Element, layout::VectorLayout>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<Element>, Layout, GmToUbVecCoordZero, AscendC::TPosition::GM>;
    using TensorDst =
        tla::Tensor<AscendC::LocalTensor<Element>, Layout, GmToUbVecCoordZero, AscendC::TPosition::VECCALC>;

    // Ops pattern: residual len 7 floats, right-align to 8 (=32B).
    constexpr uint32_t kLogicalLen = 7;
    constexpr uint8_t kRightPad = 1;

    AscendC::GlobalTensor<Element> gmSrc;
    AscendC::LocalTensor<Element> ubDst;
    auto layoutSrc = tla::MakeLayout(kLogicalLen);
    auto layoutDst = tla::MakeLayout(kLogicalLen);
    TensorSrc tensorGm(gmSrc, layoutSrc);
    TensorDst tensorUb(ubDst, layoutDst);

    AscendC::DataCopyPadExtParams<Element> padParams{true, 0, kRightPad, static_cast<Element>(0)};

    CopyGm2UbTla<Arch::Ascend950, TensorSrc, TensorDst> copyGmToUb;
    copyGmToUb(tensorUb, tensorGm, padParams);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1U);
    ASSERT_EQ(logs[0].name, "DataCopyPad");
    ASSERT_GE(logs[0].args.size(), 4U);

    const auto* params = logs[0].GetArgsAt(2).template Value<AscendC::DataCopyExtParams>();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->blockCount, 1U);
    EXPECT_EQ(params->blockLen, kLogicalLen * sizeof(Element));

    const auto* got = logs[0].GetArgsAt(3).template Value<AscendC::DataCopyPadExtParams<Element>>();
    ASSERT_NE(got, nullptr);
    EXPECT_TRUE(got->isPad);
    EXPECT_EQ(got->leftPadding, 0);
    EXPECT_EQ(got->rightPadding, kRightPad);
}

TEST_F(TileCopyGmToUbTlaPadAscend950, VectorAcceptsAscendCExtParamsDirectly)
{
    using Element = int32_t;
    using Layout = Catlass::detail::TagToLayout_t<Element, layout::VectorLayout>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<Element>, Layout, GmToUbVecCoordZero, AscendC::TPosition::GM>;
    using TensorDst =
        tla::Tensor<AscendC::LocalTensor<Element>, Layout, GmToUbVecCoordZero, AscendC::TPosition::VECCALC>;

    constexpr uint32_t kLogicalLen = 5;
    // Ops migration: keep the ExtParams object they already fill.
    AscendC::DataCopyPadExtParams<Element> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 2;
    padParams.rightPadding = 1;
    padParams.paddingValue = static_cast<Element>(-1);

    AscendC::GlobalTensor<Element> gmSrc;
    AscendC::LocalTensor<Element> ubDst;
    auto layoutSrc = tla::MakeLayout(kLogicalLen);
    auto layoutDst = tla::MakeLayout(kLogicalLen);
    TensorSrc tensorGm(gmSrc, layoutSrc);
    TensorDst tensorUb(ubDst, layoutDst);

    CopyGm2UbTla<Arch::Ascend950, TensorSrc, TensorDst> copyGmToUb;
    copyGmToUb(tensorUb, tensorGm, padParams);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1U);
    const auto* got = logs[0].GetArgsAt(3).template Value<AscendC::DataCopyPadExtParams<Element>>();
    ASSERT_NE(got, nullptr);
    EXPECT_TRUE(got->isPad);
    EXPECT_EQ(got->leftPadding, 2);
    EXPECT_EQ(got->rightPadding, 1);
    EXPECT_EQ(got->paddingValue, static_cast<Element>(-1));
}

TEST_F(TileCopyGmToUbTlaPadAscend950, RowMajorRightPadForwarded)
{
    using Element = float;
    using Layout = Catlass::detail::TagToLayout_t<Element, layout::RowMajor>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<Element>, Layout, GmToUbRmCoordZero, AscendC::TPosition::GM>;
    using TensorDst =
        tla::Tensor<AscendC::LocalTensor<Element>, Layout, GmToUbRmCoordZero, AscendC::TPosition::VECCALC>;

    // 2 rows × 6 cols; right-pad 2 elems to 8 (=32B) per row — DataCopyPad2D style.
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

    CopyGm2UbTla<Arch::Ascend950, TensorSrc, TensorDst> copyGmToUb;
    copyGmToUb(tensorUb, tensorGm, padParams);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1U);
    ASSERT_EQ(logs[0].name, "DataCopyPad");

    const auto* params = logs[0].GetArgsAt(2).template Value<AscendC::DataCopyExtParams>();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->blockCount, kRows);
    EXPECT_EQ(params->blockLen, kCols * sizeof(Element));

    const auto* got = logs[0].GetArgsAt(3).template Value<AscendC::DataCopyPadExtParams<Element>>();
    ASSERT_NE(got, nullptr);
    EXPECT_TRUE(got->isPad);
    EXPECT_EQ(got->leftPadding, 0);
    EXPECT_EQ(got->rightPadding, kRightPad);
}

TEST_F(TileCopyGmToUbTlaPadAscend950, DefaultTwoArgStillExactBytesNoPad)
{
    using Element = float;
    using Layout = Catlass::detail::TagToLayout_t<Element, layout::VectorLayout>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<Element>, Layout, GmToUbVecCoordZero, AscendC::TPosition::GM>;
    using TensorDst =
        tla::Tensor<AscendC::LocalTensor<Element>, Layout, GmToUbVecCoordZero, AscendC::TPosition::VECCALC>;

    constexpr uint32_t kLogicalLen = 129;

    AscendC::GlobalTensor<Element> gmSrc;
    AscendC::LocalTensor<Element> ubDst;
    auto layoutSrc = tla::MakeLayout(kLogicalLen);
    auto layoutDst = tla::MakeLayout(kLogicalLen);
    TensorSrc tensorGm(gmSrc, layoutSrc);
    TensorDst tensorUb(ubDst, layoutDst);

    CopyGm2UbTla<Arch::Ascend950, TensorSrc, TensorDst> copyGmToUb;
    copyGmToUb(tensorUb, tensorGm);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1U);
    const auto* params = logs[0].GetArgsAt(2).template Value<AscendC::DataCopyExtParams>();
    const auto* padParams = logs[0].GetArgsAt(3).template Value<AscendC::DataCopyPadExtParams<Element>>();
    ASSERT_NE(params, nullptr);
    ASSERT_NE(padParams, nullptr);
    EXPECT_EQ(params->blockLen, kLogicalLen * sizeof(Element));
    EXPECT_FALSE(padParams->isPad);
    EXPECT_EQ(padParams->leftPadding, 0);
    EXPECT_EQ(padParams->rightPadding, 0);
}

#endif // CATLASS_ARCH == 3510
