/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "stub/ascendc_test_fixture.h"
#include "stub/kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/layout/layout.hpp"

#include "tla/tensor.hpp"
#include "catlass/detail/tag_to_layout.hpp"

#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

using namespace Catlass;
using namespace Catlass::Epilogue::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

class TileCopyGmToUbTlaTestAscend950 : public UBTileCopyTest, public testing::WithParamInterface<TestVectorShape> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    void setShape()
    {
        _setShape(GetParam().blkLen, GetParam().blkCnt);
    }

    template <class Element>
    void BaseCheck(const AscendCCallLog& logVecCopy)
    {
        ASSERT_EQ(logVecCopy.name, "DataCopyPad");
        ASSERT_EQ(logVecCopy.args.size(), 4);
        ASSERT_EQ(logVecCopy.GetArgsTAt(0).Type(), typeid(Element));
    }
};

class TileCopyGmToUbTlaNonContiguousTestAscend950 : public UBTileCopyTest,
    public testing::WithParamInterface<TestVectorShapeWithStride> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    void setShape()
    {
        _setShape(GetParam().blkLen, GetParam().blkCnt);
        _setOriginalShape(GetParam().oriBlkLen, GetParam().oriBlkCnt);
        _srcStride = GetParam().srcStride;
        _dstStride = GetParam().dstStride;
    }

    template <class Element>
    void BaseCheck(const AscendCCallLog& logVecCopy)
    {
        ASSERT_EQ(logVecCopy.name, "DataCopyPad");
        ASSERT_EQ(logVecCopy.args.size(), 4);
        ASSERT_EQ(logVecCopy.GetArgsTAt(0).Type(), typeid(Element));
    }

    uint32_t _srcStride = 128;
    uint32_t _dstStride = 128;
};

using GmToUbCoordZero = tla::Coord<tla::Int<0>, tla::Int<0>>;
template <class Element, class LayoutSrc>
using GmToUbTensorSrc = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutSrc, GmToUbCoordZero,
    AscendC::TPosition::GM>;
template <class Element, class LayoutDst>
using GmToUbTensorDst = tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, GmToUbCoordZero,
    AscendC::TPosition::VECCALC>;

// Vector layout is rank-1, so its coordinate must be rank-1 as well.
using GmToUbVecCoordZero = tla::Coord<tla::Int<0>>;
template <class Element, class LayoutSrc>
using GmToUbVecSrc = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutSrc, GmToUbVecCoordZero,
    AscendC::TPosition::GM>;
template <class Element, class LayoutDst>
using GmToUbVecDst = tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, GmToUbVecCoordZero,
    AscendC::TPosition::VECCALC>;

// ============================================================================
// Testsuite from **RowMajor**
// ============================================================================

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: Basic (TLA contiguous, single DataCopyPad)
TEST_P(TileCopyGmToUbTlaTestAscend950, RowMajorToRowMajorTestBasic)
{
    using Element = float;
    using LayoutTag = layout::RowMajor;
    using Layout = detail::TagToLayout_t<Element, LayoutTag>;

    AscendC::GlobalTensor<Element> gmSrc;
    AscendC::LocalTensor<Element> ubDst;

    setShape();
    auto layoutSrc = tla::MakeLayout<Element, LayoutTag>(static_cast<uint32_t>(_blkCnt), _blkLen);
    auto layoutDst = tla::MakeLayout<Element, LayoutTag>(static_cast<uint32_t>(_blkCnt), _blkLen);
    GmToUbTensorSrc<Element, Layout> tensorGm(gmSrc, layoutSrc);
    GmToUbTensorDst<Element, Layout> tensorUb(ubDst, layoutDst);

    CopyGm2UbTla<Arch::Ascend950, decltype(tensorGm), decltype(tensorUb)> copyGmToUb;
    copyGmToUb(tensorUb, tensorGm);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logVecCopy = logs[0];
    BaseCheck<Element>(logVecCopy);
    ASSERT_EQ(logVecCopy.GetArgsAt(0).GetInstAddr(), 0);
    ASSERT_EQ(logVecCopy.GetArgsAt(1).GetInstAddr(), 0);

    const auto* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
    const auto* padParams = logVecCopy.GetArgsAt(3).Value<AscendC::DataCopyPadExtParams<Element>>();
    ASSERT_EQ(dataCopyParams->blockCount, _blkCnt);
    ASSERT_EQ(dataCopyParams->blockLen, _blkLen * sizeof(Element));
    ASSERT_EQ(dataCopyParams->srcStride, _0);
    ASSERT_EQ(dataCopyParams->dstStride, _0);
    ASSERT_EQ(padParams->isPad, false);
}

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: NonContiguous (TLA strided rows, DataCopyPad with src/dst gaps)
TEST_P(TileCopyGmToUbTlaNonContiguousTestAscend950, RowMajorToRowMajorTestNonContiguous)
{
    using Element = float;
    using Layout = detail::TagToLayout_t<Element, layout::RowMajor>;
    constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    AscendC::GlobalTensor<Element> gmSrc;
    AscendC::LocalTensor<Element> ubDst;

    setShape();
    if (_srcStride == _oriBlkLen && _dstStride == _oriBlkLen) {
        GTEST_SKIP();
    }

    auto layoutSrc = tla::MakeLayout(
        tla::MakeShape(static_cast<uint32_t>(_blkCnt), _blkLen),
        tla::MakeStride(static_cast<int64_t>(_srcStride), tla::Int<1>{}),
        tla::MakeShape(static_cast<uint32_t>(_oriBlkCnt), _oriBlkLen));
    auto layoutDst = tla::MakeLayout(
        tla::MakeShape(static_cast<uint32_t>(_blkCnt), _blkLen),
        tla::MakeStride(static_cast<int64_t>(_dstStride), tla::Int<1>{}),
        tla::MakeShape(static_cast<uint32_t>(_oriBlkCnt), _oriBlkLen));
    GmToUbTensorSrc<Element, Layout> tensorGm(gmSrc, layoutSrc);
    GmToUbTensorDst<Element, Layout> tensorUb(ubDst, layoutDst);

    CopyGm2UbTla<Arch::Ascend950, decltype(tensorGm), decltype(tensorUb)> copyGmToUb;
    copyGmToUb(tensorUb, tensorGm);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logVecCopy = logs[0];
    BaseCheck<Element>(logVecCopy);
    ASSERT_EQ(logVecCopy.GetArgsAt(0).GetInstAddr(), 0);
    ASSERT_EQ(logVecCopy.GetArgsAt(1).GetInstAddr(), 0);

    const auto* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
    const auto* padParams = logVecCopy.GetArgsAt(3).Value<AscendC::DataCopyPadExtParams<Element>>();
    ASSERT_EQ(dataCopyParams->blockCount, _oriBlkCnt);
    ASSERT_EQ(dataCopyParams->blockLen, _oriBlkLen * sizeof(Element));
    ASSERT_EQ(dataCopyParams->srcStride, (_srcStride - _oriBlkLen) * sizeof(Element));
    ASSERT_EQ(dataCopyParams->dstStride, (_dstStride - _oriBlkLen) / ELE_NUM_PER_BLK);
    ASSERT_EQ(padParams->isPad, false);
}

// ============================================================================
// Testsuite from **RowMajor** in fp4, packed two elements per byte
// ============================================================================

class TileCopyGmToUbTlaFp4TestAscend950 : public UBTileCopyTest,
    public testing::WithParamInterface<TestVectorShapeWithStride> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    template <class Element>
    void BaseCheck(const AscendCCallLog& logVecCopy)
    {
        ASSERT_EQ(logVecCopy.name, "DataCopyPad");
        ASSERT_EQ(logVecCopy.args.size(), 4);
        ASSERT_EQ(logVecCopy.GetArgsTAt(0).Type(), typeid(Element));
    }
};

// Data-path: RowMajor → RowMajor
// Element-type: not-except (fp4)
// Speciality: NonContiguous (fp4 rows packed in half bytes, byte src gap)
TEST_P(TileCopyGmToUbTlaFp4TestAscend950, RowMajorToRowMajorTestFp4NonContiguous)
{
    using Element = float4_e2m1x2_t;
    using Layout = detail::TagToLayout_t<Element, layout::RowMajor>;
    constexpr uint32_t ELE_NUM_PER_BLK_FP4 = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;

    AscendC::GlobalTensor<Element> gmSrc;
    AscendC::LocalTensor<Element> ubDst;

    const auto& param = GetParam();
    auto layoutSrc = tla::MakeLayout(
        tla::MakeShape(static_cast<uint32_t>(param.blkCnt), param.blkLen),
        tla::MakeStride(static_cast<int64_t>(param.srcStride), tla::Int<1>{}),
        tla::MakeShape(static_cast<uint32_t>(param.oriBlkCnt), param.oriBlkLen));
    auto layoutDst = tla::MakeLayout(
        tla::MakeShape(static_cast<uint32_t>(param.blkCnt), param.blkLen),
        tla::MakeStride(static_cast<int64_t>(param.dstStride), tla::Int<1>{}),
        tla::MakeShape(static_cast<uint32_t>(param.oriBlkCnt), param.oriBlkLen));
    GmToUbTensorSrc<Element, Layout> tensorGm(gmSrc, layoutSrc);
    GmToUbTensorDst<Element, Layout> tensorUb(ubDst, layoutDst);

    CopyGm2UbTla<Arch::Ascend950, decltype(tensorGm), decltype(tensorUb)> copyGmToUb;
    copyGmToUb(tensorUb, tensorGm);

    auto logs = AscendCCallLogger::Instance().GetCopyLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logVecCopy = logs[0];
    BaseCheck<uint8_t>(logVecCopy);

    const auto* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
    const auto* padParams = logVecCopy.GetArgsAt(3).Value<AscendC::DataCopyPadExtParams<uint8_t>>();
    ASSERT_NE(dataCopyParams, nullptr);
    ASSERT_NE(padParams, nullptr);

    // Two fp4 elements share one byte: block length and GM row gap are packed byte counts.
    const uint32_t expectedBlockLen = (param.oriBlkLen + 1U) / 2U;
    const uint32_t expectedSrcStride = (param.srcStride + 1U) / 2U - expectedBlockLen;
    // dstStride stays in 32B datablocks; one datablock holds ELE_NUM_PER_BLK_FP4 fp4 elements.
    const uint32_t expectedDstStride = (param.dstStride - param.oriBlkLen) / ELE_NUM_PER_BLK_FP4;

    ASSERT_EQ(dataCopyParams->blockCount, param.oriBlkCnt);
    ASSERT_EQ(dataCopyParams->blockLen, expectedBlockLen);
    ASSERT_EQ(dataCopyParams->srcStride, expectedSrcStride);
    ASSERT_EQ(dataCopyParams->dstStride, expectedDstStride);
    ASSERT_EQ(padParams->isPad, false);
}

// ============================================================================
// Testsuite from **Vector**
// ============================================================================

// Data-path: Vector → Vector
// Element-type: no-except (float)
// Speciality: Basic (TLA rank-1, single DataCopyPad with blockCount = 1)
TEST_P(TileCopyGmToUbTlaTestAscend950, VectorToVectorTestBasic)
{
    using Element = float;
    using LayoutTag = layout::VectorLayout;
    using Layout = detail::TagToLayout_t<Element, LayoutTag>;

    AscendC::GlobalTensor<Element> gmSrc;
    AscendC::LocalTensor<Element> ubDst;

    setShape();
    Layout layoutSrc{tla::MakeShape(_totalLen), tla::MakeStride(tla::Int<1>{}), tla::MakeShape(_totalLen)};
    Layout layoutDst{tla::MakeShape(_totalLen), tla::MakeStride(tla::Int<1>{}), tla::MakeShape(_totalLen)};
    GmToUbVecSrc<Element, Layout> tensorGm(gmSrc, layoutSrc);
    GmToUbVecDst<Element, Layout> tensorUb(ubDst, layoutDst);

    CopyGm2UbTla<Arch::Ascend950, decltype(tensorGm), decltype(tensorUb)> copyGmToUb;
    copyGmToUb(tensorUb, tensorGm);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logVecCopy = logs[0];
    BaseCheck<Element>(logVecCopy);
    ASSERT_EQ(logVecCopy.GetArgsAt(0).GetInstAddr(), 0);
    ASSERT_EQ(logVecCopy.GetArgsAt(1).GetInstAddr(), 0);

    const auto* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();
    const auto* padParams = logVecCopy.GetArgsAt(3).Value<AscendC::DataCopyPadExtParams<Element>>();
    ASSERT_EQ(dataCopyParams->blockCount, _1);
    ASSERT_EQ(dataCopyParams->blockLen, _totalLen * sizeof(Element));
    ASSERT_EQ(dataCopyParams->srcStride, _0);
    ASSERT_EQ(dataCopyParams->dstStride, _0);
    ASSERT_EQ(padParams->isPad, false);
}

INSTANTIATE_TEST_SUITE_P(
    CopyGmToUbTlaAscend950,
    TileCopyGmToUbTlaTestAscend950,
    ::testing::Values(
        TestVectorShape{128U, 1U},
        TestVectorShape{256U, 4U},
        TestVectorShape{64U, 8U}
    )
);

INSTANTIATE_TEST_SUITE_P(
    CopyGmToUbTlaNonContiguousAscend950,
    TileCopyGmToUbTlaNonContiguousTestAscend950,
    ::testing::Values(
        /* blkLen, blkCnt, oriBlkLen, oriBlkCnt, srcStride, dstStride */
        TestVectorShapeWithStride{128U, 4U, 128U, 4U, 256U, 256U},
        TestVectorShapeWithStride{64U, 8U, 64U, 8U, 128U, 128U},
        TestVectorShapeWithStride{256U, 2U, 256U, 2U, 512U, 512U},
        TestVectorShapeWithStride{256U, 2U, 192U, 1U, 512U, 512U}
    )
);

INSTANTIATE_TEST_SUITE_P(
    CopyGmToUbTlaFp4Ascend950,
    TileCopyGmToUbTlaFp4TestAscend950,
    ::testing::Values(
        /* blkLen, blkCnt, oriBlkLen, oriBlkCnt, srcStride, dstStride */
        TestVectorShapeWithStride{128U, 4U, 128U, 4U, 256U, 256U},
        TestVectorShapeWithStride{64U, 8U, 64U, 8U, 128U, 128U},
        TestVectorShapeWithStride{256U, 2U, 192U, 1U, 512U, 512U},
        TestVectorShapeWithStride{128U, 4U, 127U, 4U, 128U, 128U}
    )
);

#endif // CATLASS_ARCH == 3510
