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

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

// Test for CopyGmToL1
class TileCopyGmToL1TestAscend950: public TileCopyTest, public testing::WithParamInterface<TestMatrixShape> {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    template <class Element, bool isTrans = false>
    void setShape() {
        uint32_t row = GetParam().row;
        uint32_t col = GetParam().col;
        _setShape<Element, isTrans>(row, col);
    }

    template <class Element>
    void BaseCheck(AscendCCallLog const &logTileCopy){
        ASSERT_EQ(logTileCopy.name, "DataCopy");
        ASSERT_EQ(logTileCopy.args.size(), 3);
        ASSERT_EQ(logTileCopy.GetArgsTAt(0).Type(), typeid(Element));
    }

};

// ============================================================================
// Testsuite from **RowMajor**
// ============================================================================

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: 3-Tparam, to TPosition::A1
TEST_P(TileCopyGmToL1TestAscend950, RowMajorTozNTestA1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_LT(layoutSrc.stride(0), STRIDE_LIMIT);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    auto logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);
    ASSERT_EQ(logTileCopy.GetArgsAt(0).RawValue(), &l1Tensor);
    ASSERT_EQ(logTileCopy.GetArgsAt(1).RawValue(), &gmTensor);

    const auto* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _row);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, layoutSrc.stride(0));
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(3) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(0) / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: 3-Tparam, to TPosition::B1
TEST_P(TileCopyGmToL1TestAscend950, RowMajorTozNTestB1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;

    setShape<Element>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    auto logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);
    ASSERT_EQ(logTileCopy.GetArgsAt(0).RawValue(), &l1Tensor);
    ASSERT_EQ(logTileCopy.GetArgsAt(1).RawValue(), &gmTensor);

    const auto* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _row);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, _col);
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round);
    ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: LongStride (src stride >= Nd2Nz limit, single Nd2Nz with large srcDValue)
TEST_P(TileCopyGmToL1TestAscend950, RowMajorTozNTestLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;

    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;

    using GmType = Gemm::GemmType<Element, LayoutSrc>;

    setShape<Element>();
    uint32_t _very_long_stride = STRIDE_LIMIT + 1;

    LayoutSrc layoutSrc{_row, _col, _very_long_stride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_FALSE(isContiguous(layoutSrc));
    ASSERT_TRUE(isContiguous(layoutDst));
    ASSERT_GE(_very_long_stride, STRIDE_LIMIT);
    
    CopyGmToL1<ArchTag, GmType> copyGmToL1;

    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    // log check
    AscendCCallLogger& logger = AscendCCallLogger::Instance();
    auto logs = logger.GetLogs();
    AscendCCallLog logTileCopy = logs[0];

    auto logTileCopyGmTensor = logTileCopy.GetArgsAt(1).RawValue();
    auto logTileCopyL1Tensor = logTileCopy.GetArgsAt(0).RawValue();
    ASSERT_EQ(logTileCopyGmTensor, &gmTensor);
    ASSERT_EQ(logTileCopyL1Tensor, &l1Tensor);

    ASSERT_EQ(logs.size(), 1); // still one tile-copy ops
    const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1); 
    ASSERT_EQ(nd2nzArg->nValue, _row);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, _very_long_stride); // `srcDValue` can cover long stride
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round);
    ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: 3-Tparam, to TPosition::A1, long stride
TEST_P(TileCopyGmToL1TestAscend950, RowMajorTozNTestA1LongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element>();
    uint32_t veryLongStride = STRIDE_LIMIT + 1;
    LayoutSrc layoutSrc{_row, _col, veryLongStride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _row);
    for (uint32_t i = 0; i < logs.size(); ++i) {
        BaseCheck<Element>(logs[i]);
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), i * ELE_NUM_PER_C0 * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), i * veryLongStride * sizeof(Element));
        const auto* nd2nzArg = logs[i].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: RowMajor → zN
// Element-type: no-except (float)
// Speciality: 3-Tparam, to TPosition::B1, long stride
TEST_P(TileCopyGmToL1TestAscend950, RowMajorTozNTestB1LongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element>();
    uint32_t veryLongStride = STRIDE_LIMIT + 1;
    LayoutSrc layoutSrc{_row, _col, veryLongStride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_GE(layoutSrc.stride(0), STRIDE_LIMIT);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _row);
    for (uint32_t i = 0; i < _row; ++i) {
        BaseCheck<Element>(logs[i]);
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), i * ELE_NUM_PER_C0 * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), i * veryLongStride * sizeof(Element));

        const auto* nd2nzArg = logs[i].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: PaddingRowMajor → zN
// Element-type: half
// Speciality: DynamicOptimized (CopyGmToL1DynamicOptimized, padded srcDValue Nd2Nz)
TEST_P(TileCopyGmToL1TestAscend950, PaddingRowMajorTozNTestDynamicOptimized)
{
    using Element = half;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::PaddingRowMajor;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element>();
    LayoutSrc layoutSrc(_row, _col, C0_NUM_PER_FRACTAL, ELE_NUM_PER_C0);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1DynamicOptimized<ArchTag, GmType> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    const auto* nd2nzArg = logs[0].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _row);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, _row_round);
    ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: huge non-contiguous stride, per-row DataCopy fallback
TEST_P(TileCopyGmToL1TestAscend950, RowMajorToRowMajorTestA1PerRowFallback)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using Layout = layout::RowMajor;
    using GmType = Gemm::GemmType<Element, Layout>;
    using L1Type = Gemm::GemmType<Element, Layout, AscendC::TPosition::A1>;

    setShape<Element>();
    uint32_t hugeStride = STRIDE_LIMIT * 16;
    Layout layoutSrc{_row, _col, hugeStride};
    Layout layoutDst{_row, _col, _col};
    ASSERT_NE(layoutSrc.shape(1), layoutSrc.stride(0));

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _row);
    for (uint32_t i = 0; i < _row; ++i) {
        ASSERT_EQ(logs[i].name, "DataCopy");
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * hugeStride * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)i * layoutDst.stride(0) * sizeof(Element));
        ASSERT_EQ(*logs[i].GetArgsAt(2).Value<uint32_t>(), _col);
    }
}

// Data-path: RowMajor → zZ
// Element-type: no-except (float)
// Speciality: 3-Tparam to TPosition::B1
TEST_P(TileCopyGmToL1TestAscend950, RowMajorTozZTestB1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element, true>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t ndNum = _row / C0_NUM_PER_FRACTAL;
    uint32_t remains = _row % C0_NUM_PER_FRACTAL;
    ASSERT_EQ(logs.size(), (ndNum ? 1U : 0U) + (remains ? 1U : 0U));

    uint32_t srcNdStride = C0_NUM_PER_FRACTAL * _col;
    uint32_t dstMatrixStride = RoundUp<ELE_NUM_PER_C0>(_col) * C0_NUM_PER_FRACTAL;
    uint32_t logIdx = 0;
    if (ndNum) {
        BaseCheck<Element>(logs[logIdx]);
        const auto* nd2nzArg = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, ndNum);
        ASSERT_EQ(nd2nzArg->nValue, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, srcNdStride);
        ASSERT_EQ(nd2nzArg->srcDValue, _col);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, dstMatrixStride);
        ++logIdx;
    }
    if (remains) {
        BaseCheck<Element>(logs[logIdx]);
        ASSERT_EQ(logs[logIdx].GetArgsAt(0).GetInstAddr(), ndNum * dstMatrixStride * sizeof(Element));
        ASSERT_EQ(logs[logIdx].GetArgsAt(1).GetInstAddr(), ndNum * srcNdStride * sizeof(Element));
        const auto* nd2nzArg = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, remains);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, srcNdStride);
        ASSERT_EQ(nd2nzArg->srcDValue, _col);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: 3-Tparam, to TPosition::A1, contiguous single DataCopy by total count
TEST_P(TileCopyGmToL1TestAscend950, RowMajorToRowMajorTestA1Contiguous)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using Layout = layout::RowMajor;
    using GmType = Gemm::GemmType<Element, Layout>;
    using L1Type = Gemm::GemmType<Element, Layout, AscendC::TPosition::A1>;

    setShape<Element>();
    Layout layoutSrc = Layout::template MakeLayout<Element>(_row, _col);
    Layout layoutDst = Layout::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    BaseCheck<Element>(logs[0]);
    ASSERT_EQ(*logs[0].GetArgsAt(2).Value<uint32_t>(), _row * _col);
}

// Data-path: RowMajor → zZ
// Element-type: no-except (float)
// Speciality: B1LongSrcNdStride (srcNdStride >= limit while src stride < limit, per-fractal split)
TEST_P(TileCopyGmToL1TestAscend950, RowMajorTozZTestB1LongSrcNdStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element>();
    uint32_t midStride = STRIDE_LIMIT / C0_NUM_PER_FRACTAL + 1;   // 4097: makes srcNdStride >= STRIDE_LIMIT
    LayoutSrc layoutSrc{_row, _col, midStride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_LT(layoutSrc.stride(0), STRIDE_LIMIT);
    ASSERT_GE(C0_NUM_PER_FRACTAL * layoutSrc.stride(0), STRIDE_LIMIT);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t ndNum = _row / C0_NUM_PER_FRACTAL;
    uint32_t remains = _row % C0_NUM_PER_FRACTAL;
    ASSERT_EQ(logs.size(), ndNum + (remains ? 1U : 0U));
    uint32_t logIdx = 0;
    for (uint32_t i = 0; i < ndNum; ++i) {
        BaseCheck<Element>(logs[i]);
        const auto* nd2nzArg = logs[i].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, midStride);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
        ++logIdx;
    }
    if (remains) {
        BaseCheck<Element>(logs[logIdx]);
        const auto* nd2nzArg = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, remains);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, midStride);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: RowMajor → RowMajor
// Element-type: no-except (float)
// Speciality: A1NonContiguous (non-contiguous ldm, DataCopyParams or per-row fallback)
TEST_P(TileCopyGmToL1TestAscend950, RowMajorToRowMajorTestA1NonContiguous)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using Layout = layout::RowMajor;
    using GmType = Gemm::GemmType<Element, Layout>;
    using L1Type = Gemm::GemmType<Element, Layout, AscendC::TPosition::A1>;

    setShape<Element>();
    uint32_t srcLdm = _col + 16;  // non-contiguous
    uint32_t dstLdm = _col + 32;
    Layout layoutSrc{_row, _col, srcLdm};
    Layout layoutDst{_row, _col, dstLdm};
    ASSERT_NE(layoutSrc.shape(1), layoutSrc.stride(0));

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_GE(logs.size(), 1);
    ASSERT_EQ(logs[0].name, "DataCopy");
}

// Data-path: RowMajor → zZ
// Element-type: no-except (float)
// Speciality: B1FullLongStride (src stride >= limit, full per-row Nd2Nz split)
TEST_P(TileCopyGmToL1TestAscend950, RowMajorTozZTestB1FullLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::zZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element>();
    uint32_t veryLongStride = STRIDE_LIMIT + 1;
    LayoutSrc layoutSrc{_row, _col, veryLongStride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_GE(layoutSrc.stride(0), STRIDE_LIMIT);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _row);
    for (uint32_t i = 0; i < _row; ++i) {
        BaseCheck<Element>(logs[i]);
        uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
        uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;
        uint32_t offsetDst = idxR0 * layoutDst.stride(1) + idxInR0 * ELE_NUM_PER_C0;
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)offsetDst * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * veryLongStride * sizeof(Element));
        const auto* nd2nzArg = logs[i].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, C0_NUM_PER_FRACTAL); // HERE
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// ============================================================================
// Testsuite from **ColumnMajor**
// ============================================================================

// Data-path: ColumnMajor → nZ
// Element-type: no-except (float)
// Speciality: B1 (3-param B1 specialization, single Nd2Nz)
TEST_P(TileCopyGmToL1TestAscend950, ColumnMajorTonZTestB1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;

    setShape<Element, true>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    auto logTileCopy = logs[0];
    BaseCheck<Element>(logTileCopy);
    ASSERT_EQ(logTileCopy.GetArgsAt(0).RawValue(), &l1Tensor);
    ASSERT_EQ(logTileCopy.GetArgsAt(1).RawValue(), &gmTensor);

    const auto* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _col);
    ASSERT_EQ(nd2nzArg->dValue, _row);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, _row);
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, _col_round);
    ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}



// Data-path: ColumnMajor → nN
// Element-type: no-except (float)
// Speciality: A1 (3-param A1, fractal-block + remainder Nd2Nz split)
TEST_P(TileCopyGmToL1TestAscend950, ColumnMajorTonNTestA1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element, true>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    const auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t ndNum = _col / C0_NUM_PER_FRACTAL;
    uint32_t remains = _col % C0_NUM_PER_FRACTAL;
    ASSERT_EQ(logs.size(), (ndNum ? 1U : 0U) + (remains ? 1U : 0U));

    uint32_t srcNdStride = C0_NUM_PER_FRACTAL * _row;
    uint32_t dstMatrixStride = _row_round * C0_NUM_PER_FRACTAL;
    uint32_t logIdx = 0;
    if (ndNum) {
        BaseCheck<Element>(logs[logIdx]);
        const auto* nd2nzArg = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, ndNum);
        ASSERT_EQ(nd2nzArg->nValue, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dValue, _row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, srcNdStride);
        ASSERT_EQ(nd2nzArg->srcDValue, _row);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, dstMatrixStride);
        ++logIdx;
    }
    if (remains) {
        BaseCheck<Element>(logs[logIdx]);
        ASSERT_EQ(logs[logIdx].GetArgsAt(0).GetInstAddr(), ndNum * dstMatrixStride * sizeof(Element));
        ASSERT_EQ(logs[logIdx].GetArgsAt(1).GetInstAddr(), ndNum * srcNdStride * sizeof(Element));
        const auto* nd2nzArg = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, remains);
        ASSERT_EQ(nd2nzArg->dValue, _row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, srcNdStride);
        ASSERT_EQ(nd2nzArg->srcDValue, _row);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: ColumnMajor → nN
// Element-type: no-except (float)
// Speciality: B1 (3-param B1, fractal-block + remainder Nd2Nz split)
TEST_P(TileCopyGmToL1TestAscend950, ColumnMajorTonNTestB1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::B1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element, true>();
    LayoutSrc layoutSrc = LayoutSrc::template MakeLayout<Element>(_row, _col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    const auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t ndNum = _col / C0_NUM_PER_FRACTAL;
    uint32_t remains = _col % C0_NUM_PER_FRACTAL;
    ASSERT_EQ(logs.size(), (ndNum ? 1U : 0U) + (remains ? 1U : 0U));

    uint32_t srcNdStride = C0_NUM_PER_FRACTAL * _row;
    uint32_t dstMatrixStride = _row_round * C0_NUM_PER_FRACTAL;
    uint32_t logIdx = 0;
    if (ndNum) {
        BaseCheck<Element>(logs[logIdx]);
        const auto* nd2nzArg = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, ndNum);
        ASSERT_EQ(nd2nzArg->nValue, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dValue, _row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, srcNdStride);
        ASSERT_EQ(nd2nzArg->srcDValue, _row);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, dstMatrixStride);
        ++logIdx;
    }
    if (remains) {
        BaseCheck<Element>(logs[logIdx]);
        ASSERT_EQ(logs[logIdx].GetArgsAt(0).GetInstAddr(), ndNum * dstMatrixStride * sizeof(Element));
        ASSERT_EQ(logs[logIdx].GetArgsAt(1).GetInstAddr(), ndNum * srcNdStride * sizeof(Element));
        const auto* nd2nzArg = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, remains);
        ASSERT_EQ(nd2nzArg->dValue, _row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, srcNdStride);
        ASSERT_EQ(nd2nzArg->srcDValue, _row);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: Vector → zN
// Element-type: no-except (float)
// Speciality: A1 (3-param A1, rank-1 src treated as single-row Nd2Nz)
TEST_P(TileCopyGmToL1TestAscend950, VectorTozNTestA1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;

    setShape<Element>();
    LayoutSrc layoutSrc(_col);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_1, _col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    const auto* nd2nzArg = logs[0].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _1);
    ASSERT_EQ(nd2nzArg->dValue, _col);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, _col);
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, C0_NUM_PER_FRACTAL);
    ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}

// Data-path: Vector → Vector
// Element-type: no-except (float)
// Speciality: A1 (3-param A1, plain DataCopy by C0 blockLen)
TEST_P(TileCopyGmToL1TestAscend950, VectorToVectorTestA1)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using Layout = layout::VectorLayout;
    using GmType = Gemm::GemmType<Element, Layout, AscendC::TPosition::GM>;
    using L1Type = Gemm::GemmType<Element, Layout, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element>();
    Layout layoutSrc(_col);
    Layout layoutDst(_col);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    BaseCheck<Element>(logs[0]);
    const auto* nd2nzArg = logs[0].GetArgsAt(2).Value<AscendC::DataCopyParams>();
    ASSERT_EQ(nd2nzArg->blockCount, _1);
    ASSERT_EQ(nd2nzArg->blockLen, _col / ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->srcStride, _0);
    ASSERT_EQ(nd2nzArg->dstStride, _0);
}

// Data-path: PaddingColumnMajor → nZ
// Element-type: half
// Speciality: DynamicOptimized (CopyGmToL1DynamicOptimized, padded srcDValue Nd2Nz)
TEST_P(TileCopyGmToL1TestAscend950, PaddingColumnMajorTonZTestDynamicOptimized)
{
    using Element = half;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::PaddingColumnMajor;
    using LayoutDst = layout::nZ;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element, true>();
    LayoutSrc layoutSrc(_row, _col, ELE_NUM_PER_C0, C0_NUM_PER_FRACTAL);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);

    CopyGmToL1DynamicOptimized<ArchTag, GmType> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _1);
    const auto* nd2nzArg = logs[0].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
    ASSERT_EQ(nd2nzArg->ndNum, _1);
    ASSERT_EQ(nd2nzArg->nValue, _col);
    ASSERT_EQ(nd2nzArg->dValue, _row);
    ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
    ASSERT_EQ(nd2nzArg->srcDValue, ELE_NUM_PER_C0);
    ASSERT_EQ(nd2nzArg->dstNzC0Stride, _col_round);
    ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
    ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
}


// Data-path: ColumnMajor → nN
// Element-type: no-except (float)
// Speciality: LongSrcNdStride (srcNdStride >= limit while src stride < limit, per-fractal split)
TEST_P(TileCopyGmToL1TestAscend950, ColumnMajorTonNTestLongSrcNdStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element, true>();
    uint32_t midStride = STRIDE_LIMIT / C0_NUM_PER_FRACTAL + 1;
    LayoutSrc layoutSrc{_row, _col, midStride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_LT(layoutSrc.stride(1), STRIDE_LIMIT);
    ASSERT_GE(C0_NUM_PER_FRACTAL * layoutSrc.stride(1), STRIDE_LIMIT);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t ndNum = _col / C0_NUM_PER_FRACTAL;
    uint32_t remains = _col % C0_NUM_PER_FRACTAL;
    ASSERT_EQ(logs.size(), ndNum + (remains ? 1U : 0U));
    uint32_t logIdx = 0;
    for (uint32_t i = 0; i < ndNum; ++i) {
        BaseCheck<Element>(logs[logIdx]);
        const auto* nd2nzArg = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dValue, _row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, midStride);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
        ++logIdx;
    }
    if (remains) {
        BaseCheck<Element>(logs[logIdx]);
        const auto* nd2nzArg = logs[logIdx].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, remains);
        ASSERT_EQ(nd2nzArg->dValue, _row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, midStride);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, C0_NUM_PER_FRACTAL);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Data-path: zN → zN
// Element-type: no-except (float)
// Speciality: NoTlaLongStride (no-TLA path, outer-col stride >= limit, per-fractal-col DataCopy)
TEST_P(TileCopyGmToL1TestAscend950, zNTozNTestNoTlaLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::zN;
    using LayoutDst = layout::zN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    setShape<Element>();
    int64_t longColByFractal = (int64_t)(STRIDE_LIMIT + 1) * ELE_NUM_PER_C0;
    LayoutSrc layoutSrc(_row, _col, 
        C0_NUM_PER_FRACTAL, _row_round / C0_NUM_PER_FRACTAL,
        ELE_NUM_PER_C0, _col_round / ELE_NUM_PER_C0,
        ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL, 1, longColByFractal);
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_GE(layoutSrc.stride(3) / ELE_NUM_PER_C0, STRIDE_LIMIT);

    CopyGmToL1<ArchTag, GmType> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    uint32_t blockCount = layoutSrc.shape(3);
    uint32_t blockLen = layoutSrc.shape(0) * layoutSrc.shape(1);
    ASSERT_EQ(logs.size(), blockCount);
    for (uint32_t i = 0; i < blockCount; ++i) {
        BaseCheck<Element>(logs[i]);
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)i * layoutDst.stride(3) * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * layoutSrc.stride(3) * sizeof(Element));
        const auto* dataCopyArg = logs[i].GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(dataCopyArg->blockCount, _1);
        ASSERT_EQ(dataCopyArg->blockLen, blockLen);
        ASSERT_EQ(dataCopyArg->srcStride, _0);
        ASSERT_EQ(dataCopyArg->dstStride, _0);
    }
}







// Data-path: ColumnMajor → nN
// Element-type: no-except (float)
// Speciality: FullLongStride (src stride(1) >= limit, full per-col Nd2Nz split)
TEST_P(TileCopyGmToL1TestAscend950, ColumnMajorTonNTestFullLongStride)
{
    using Element = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutSrc = layout::ColumnMajor;
    using LayoutDst = layout::nN;
    using GmType = Gemm::GemmType<Element, LayoutSrc>;
    using L1Type = Gemm::GemmType<Element, LayoutDst, AscendC::TPosition::A1>;
    constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

    setShape<Element, true>();
    uint32_t veryLongStride = STRIDE_LIMIT + 1;
    LayoutSrc layoutSrc{_row, _col, veryLongStride};
    LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(_row, _col);
    ASSERT_GE(layoutSrc.stride(1), STRIDE_LIMIT);

    CopyGmToL1<ArchTag, GmType, L1Type> copyGmToL1;
    AscendC::GlobalTensor<Element> gmTensor;
    AscendC::LocalTensor<Element> l1Tensor;
    copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), _col);
    for (uint32_t i = 0; i < _col; ++i) {
        BaseCheck<Element>(logs[i]);
        uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
        uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;
        uint32_t offsetDst = idxR0 * layoutDst.stride(3) + idxInR0 * ELE_NUM_PER_C0;
        ASSERT_EQ(logs[i].GetArgsAt(0).GetInstAddr(), (uint64_t)offsetDst * sizeof(Element));
        ASSERT_EQ(logs[i].GetArgsAt(1).GetInstAddr(), (uint64_t)i * veryLongStride * sizeof(Element));
        const auto* nd2nzArg = logs[i].GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, _1);
        ASSERT_EQ(nd2nzArg->dValue, _row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, _0);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(1) / ELE_NUM_PER_C0);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

///////////////////////////// TEST WITH PARAMETERIC GROUPS
INSTANTIATE_TEST_SUITE_P(
    CopyGmToL1,
    TileCopyGmToL1TestAscend950,
    ::testing::Values(
        TestMatrixShape{128U, 256U},  // aligned
        TestMatrixShape{1U, 256U},    // 1-d
        TestMatrixShape{43U, 256U},   // not aligned
        TestMatrixShape{43U, 283U}    // not aligned (on both sides)
    )
);

#endif // CATLASS_ARCH == 3510
