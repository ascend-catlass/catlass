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
 * Ascend950 TileCopyTla GM→L1 VectorLayout: default two-arg path must DataCopyPad
 * with an exact byte length from originShape (no CeilDiv GM over-read).
 *
 * Build:  bash scripts/build.sh --tests catlass_unittest
 * Run:    ./build/tests/unittest/catlass_unittest_3510 \
 *           --gtest_filter='TileCopyGmToL1TlaVectorPadAscend950.*'
 */

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

#include <gtest/gtest.h>
#include "stub/ascendc_test_fixture.h"
#include "stub/kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

#include "tla/tensor.hpp"

#include "stub/ascendc_logger.h"

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;

class TileCopyGmToL1TlaVectorPadAscend950 : public AscendCTest {};

template <class Element>
void RunVectorGmToL1TwoArg(uint32_t logicalLen, uint32_t& outBlockLenBytes)
{
    using LayoutTag = layout::VectorLayout;
    using Layout = detail::TagToLayout_t<Element, LayoutTag>;
    using CoordZero = tla::Coord<tla::Int<0>>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<Element>, Layout, CoordZero, AscendC::TPosition::GM>;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<Element>, Layout, CoordZero, AscendC::TPosition::A1>;

    AscendC::GlobalTensor<Element> gmSrc;
    AscendC::LocalTensor<Element> l1Dst;

    auto layoutSrc = tla::MakeLayout(logicalLen);
    auto layoutDst = tla::MakeLayout(logicalLen);
    TensorSrc tensorGm(gmSrc, layoutSrc);
    TensorDst tensorL1(l1Dst, layoutDst);

    ASSERT_EQ(tla::get<0>(tensorGm.originShape()), logicalLen);

    TileCopyTla<Arch::Ascend950, TensorSrc, TensorDst> copyGmToL1;
    copyGmToL1(tensorL1, tensorGm);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1U);
    ASSERT_EQ(logs[0].name, "DataCopyPad");
    ASSERT_GE(logs[0].args.size(), 3U);

    const auto* params = logs[0].GetArgsAt(2).template Value<AscendC::DataCopyExtParams>();
    ASSERT_NE(params, nullptr);
    ASSERT_EQ(params->blockCount, 1U);

    outBlockLenBytes = params->blockLen;
    ASSERT_EQ(outBlockLenBytes, logicalLen * sizeof(Element));
}

// Stub DataCopyPadExtParams<half> cannot default-init paddingValue; cover the
// byte-addressable bias dtypes DSL actually registers for GM→L1 VectorLayout.
TEST_F(TileCopyGmToL1TlaVectorPadAscend950, FloatLen129ExactBytesNoOverRead)
{
    constexpr uint32_t kLogicalLen = 129;
    uint32_t blockLenBytes = 0;
    RunVectorGmToL1TwoArg<float>(kLogicalLen, blockLenBytes);
    EXPECT_EQ(blockLenBytes, kLogicalLen * sizeof(float));
}

TEST_F(TileCopyGmToL1TlaVectorPadAscend950, Int32Len129ExactBytesNoOverRead)
{
    constexpr uint32_t kLogicalLen = 129;
    uint32_t blockLenBytes = 0;
    RunVectorGmToL1TwoArg<int32_t>(kLogicalLen, blockLenBytes);
    EXPECT_EQ(blockLenBytes, kLogicalLen * sizeof(int32_t));
}

TEST_F(TileCopyGmToL1TlaVectorPadAscend950, FloatLen128AlignedStillExact)
{
    constexpr uint32_t kLogicalLen = 128;
    uint32_t blockLenBytes = 0;
    RunVectorGmToL1TwoArg<float>(kLogicalLen, blockLenBytes);
    EXPECT_EQ(blockLenBytes, kLogicalLen * sizeof(float));
}

#endif // CATLASS_ARCH == 3510
