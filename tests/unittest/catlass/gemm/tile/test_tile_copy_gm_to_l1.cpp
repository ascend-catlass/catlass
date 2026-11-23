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

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "stub/ascendc_logger.h"

#include "common/helper.hpp"
#include "common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201
#include "catlass/gemm/tile/atlasa2/test_copy_gm_to_l1.cpp"
constexpr bool kEnableAtlasA2   = true;
constexpr bool kEnableAscend950 = false;
#endif
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
#include "catlass/gemm/tile/ascend950/test_copy_gm_to_l1.cpp"
constexpr bool kEnableAtlasA2   = false;
constexpr bool kEnableAscend950 = true;
#endif

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

// TestCase for GM->L1 TileCopy Utilities (Common test scenarios, parameterized by ArchTag)
template <typename ArchTag>
class TileCopyGmToL1Test : public TileCopyTest {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    template <class Element, bool isTrans = false>
    void setShape() {
        _setShape<Element, isTrans>(_row, _col);
    }

    template <class Element>
    void BaseCheck(AscendCCallLog const &logTileCopy){
        ASSERT_EQ(logTileCopy.name, "DataCopy");
        ASSERT_EQ(logTileCopy.args.size(), 3);

        const std::type_index& T0 = logTileCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(Element));
    }
};

using TileCopyGmToL1TestArchTypes = ::testing::Types<
    Catlass::Arch::AtlasA2,
    Catlass::Arch::Ascend950
>;
TYPED_TEST_SUITE(TileCopyGmToL1Test, TileCopyGmToL1TestArchTypes);

/// Testsuite for CopyGmToL1 (Common test scenario)

// Testcase, from RowMajor -> zN, basic场景，多分形搬运
TYPED_TEST(TileCopyGmToL1Test, RowMajorTozNTestBasic)
{
    using ArchTag = TypeParam;
    if constexpr ((kEnableAtlasA2 && std::is_same_v<ArchTag, Catlass::Arch::AtlasA2>) ||
                  (kEnableAscend950 && std::is_same_v<ArchTag, Catlass::Arch::Ascend950>)) {
        using Element = float;
        using LayoutSrc = layout::RowMajor;
        using LayoutDst = layout::zN;

        using GmType = Gemm::GemmType<Element, LayoutSrc>;
        constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

        CopyGmToL1<ArchTag, GmType> copyGmToL1;

        AscendC::GlobalTensor<Element> gmTensor;
        AscendC::LocalTensor<Element> l1Tensor;

        LayoutSrc layoutSrc;
        LayoutDst layoutDst;
        this->template setShape<Element>();
        setLayout<Element>(this->_row, this->_col, layoutSrc, layoutDst);

        ASSERT_TRUE(isContiguous(layoutSrc));
        ASSERT_TRUE(isContiguous(layoutDst));
        ASSERT_NE(layoutSrc.shape(1), ELE_NUM_PER_C0);
        ASSERT_NE(layoutSrc.stride(0), ELE_NUM_PER_C0);

        copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

        AscendCCallLogger& logger = AscendCCallLogger::Instance();
        auto logs = logger.GetLogs();
        AscendCCallLog logTileCopy = logs[0];
        this->template BaseCheck<Element>(logTileCopy);
        ASSERT_EQ(logs.size(), 1);

        auto logTileCopyGmTensor = logTileCopy.GetArgsAt(1).RawValue();
        auto logTileCopyL1Tensor = logTileCopy.GetArgsAt(0).RawValue();
        ASSERT_EQ(logTileCopyGmTensor, &gmTensor);
        ASSERT_EQ(logTileCopyL1Tensor, &l1Tensor);

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, this->_row);
        ASSERT_EQ(nd2nzArg->dValue, this->_col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, this->_col);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, this->_row_round);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// Ascend950: bool is a 1-byte AscendC element (C0=32). Smoke Nd2Nz params.
TYPED_TEST(TileCopyGmToL1Test, RowMajorTozNTestBool)
{
    using ArchTag = TypeParam;
    if constexpr (kEnableAscend950 && std::is_same_v<ArchTag, Catlass::Arch::Ascend950>) {
        using Element = bool;
        using LayoutSrc = layout::RowMajor;
        using LayoutDst = layout::zN;

        using GmType = Gemm::GemmType<Element, LayoutSrc>;
        constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();
        ASSERT_EQ(ELE_NUM_PER_C0, 32U);

        CopyGmToL1<ArchTag, GmType> copyGmToL1;

        AscendC::GlobalTensor<Element> gmTensor;
        AscendC::LocalTensor<Element> l1Tensor;

        LayoutSrc layoutSrc;
        LayoutDst layoutDst;
        this->template setShape<Element>();
        setLayout<Element>(this->_row, this->_col, layoutSrc, layoutDst);

        ASSERT_TRUE(isContiguous(layoutSrc));
        ASSERT_TRUE(isContiguous(layoutDst));

        copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

        AscendCCallLogger& logger = AscendCCallLogger::Instance();
        auto logs = logger.GetLogs();
        ASSERT_EQ(logs.size(), 1U);
        AscendCCallLog logTileCopy = logs[0];
        this->template BaseCheck<Element>(logTileCopy);

        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, this->_row);
        ASSERT_EQ(nd2nzArg->dValue, this->_col);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, this->_col);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, this->_row_round);
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// ColumnMajorTonZTestBasic (#2): ColumnMajor→nZ Nd2Nz
TYPED_TEST(TileCopyGmToL1Test, ColumnMajorTonZTestBasic)
{
    using ArchTag = TypeParam;
    if constexpr ((kEnableAtlasA2 && std::is_same_v<ArchTag, Catlass::Arch::AtlasA2>) ||
                  (kEnableAscend950 && std::is_same_v<ArchTag, Catlass::Arch::Ascend950>)) {
        using Element = float;
        using LayoutSrc = layout::ColumnMajor;
        using LayoutDst = layout::nZ;
        using GmType = Gemm::GemmType<Element, LayoutSrc>;
        constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

        CopyGmToL1<ArchTag, GmType> copyGmToL1;

        AscendC::GlobalTensor<Element> gmTensor;
        AscendC::LocalTensor<Element> l1Tensor;

        this->template setShape<Element, true>();
        LayoutSrc layoutSrc;
        LayoutDst layoutDst;
        setLayout<Element>(this->_row, this->_col, layoutSrc, layoutDst);

        ASSERT_TRUE(isContiguous(layoutSrc));
        ASSERT_TRUE(isContiguous(layoutDst));

        copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

        auto logs = AscendCCallLogger::Instance().GetLogs();
        ASSERT_EQ(logs.size(), 1);
        AscendCCallLog logTileCopy = logs[0];
        this->template BaseCheck<Element>(logTileCopy);

        const auto* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();
        ASSERT_EQ(nd2nzArg->ndNum, _1);
        ASSERT_EQ(nd2nzArg->nValue, this->_col);
        ASSERT_EQ(nd2nzArg->dValue, this->_row);
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);
        ASSERT_EQ(nd2nzArg->srcDValue, this->_row);
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, layoutDst.stride(1) / ELE_NUM_PER_C0);
        ASSERT_EQ(nd2nzArg->dstNzNStride, layoutDst.stride(2) / ELE_NUM_PER_C0);
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);
    }
}

// zNTozNTestBasic (#3): zN→zN DataCopyParams
TYPED_TEST(TileCopyGmToL1Test, zNTozNTestBasic)
{
    using ArchTag = TypeParam;
    if constexpr ((kEnableAtlasA2 && std::is_same_v<ArchTag, Catlass::Arch::AtlasA2>) ||
                  (kEnableAscend950 && std::is_same_v<ArchTag, Catlass::Arch::Ascend950>)) {
        using Element = float;
        using LayoutSrc = layout::zN;
        using LayoutDst = layout::zN;
        using GmType = Gemm::GemmType<Element, LayoutSrc>;
        constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

        CopyGmToL1<ArchTag, GmType> copyGmToL1;

        AscendC::GlobalTensor<Element> gmTensor;
        AscendC::LocalTensor<Element> l1Tensor;

        this->template setShape<Element>();
        LayoutSrc layoutSrc;
        LayoutDst layoutDst;
        setLayout<Element>(this->_row, this->_col, layoutSrc, layoutDst);

        copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

        auto logs = AscendCCallLogger::Instance().GetLogs();
        ASSERT_EQ(logs.size(), 1);
        AscendCCallLog logTileCopy = logs[0];
        this->template BaseCheck<Element>(logTileCopy);

        const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(p->blockCount, CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1)));
        ASSERT_EQ(p->blockLen, layoutSrc.orgShape(0));
    }
}

// nZTozNTestBasic (#4): nZ→nZ DataCopyParams
TYPED_TEST(TileCopyGmToL1Test, nZTozNTestBasic)
{
    using ArchTag = TypeParam;
    if constexpr ((kEnableAtlasA2 && std::is_same_v<ArchTag, Catlass::Arch::AtlasA2>) ||
                  (kEnableAscend950 && std::is_same_v<ArchTag, Catlass::Arch::Ascend950>)) {
        using Element = float;
        using LayoutSrc = layout::nZ;
        using LayoutDst = layout::nZ;
        using GmType = Gemm::GemmType<Element, LayoutSrc>;
        constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

        CopyGmToL1<ArchTag, GmType> copyGmToL1;

        AscendC::GlobalTensor<Element> gmTensor;
        AscendC::LocalTensor<Element> l1Tensor;

        this->template setShape<Element, true>();
        LayoutSrc layoutSrc;
        LayoutDst layoutDst;
        setLayout<Element>(this->_row, this->_col, layoutSrc, layoutDst);

        copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

        auto logs = AscendCCallLogger::Instance().GetLogs();
        ASSERT_EQ(logs.size(), 1);
        AscendCCallLog logTileCopy = logs[0];
        this->template BaseCheck<Element>(logTileCopy);

        const auto* p = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
        ASSERT_EQ(p->blockCount, CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0)));
        ASSERT_EQ(p->blockLen, layoutSrc.orgShape(1));
    }
}

// RowMajor -> zN, interval datacopy action (use DataCopy instead of nd2nz)
TYPED_TEST(TileCopyGmToL1Test, RowMajorTozNIntervalDataCopy)
{
    using ArchTag = TypeParam;
    if constexpr ((kEnableAtlasA2 && std::is_same_v<ArchTag, Catlass::Arch::AtlasA2>) ||
                  (kEnableAscend950 && std::is_same_v<ArchTag, Catlass::Arch::Ascend950>)) {
        using Element = half; // fixed to be `half`
        using LayoutSrc = layout::RowMajor;
        using LayoutDst = layout::zN;
        using GmType = Gemm::GemmType<Element, LayoutSrc>;
        constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

        CopyGmToL1IntervalDataCopy<ArchTag, GmType> copyGmToL1;
        AscendC::GlobalTensor<Element> gmTensor;
        AscendC::LocalTensor<Element> l1Tensor;

        LayoutSrc layoutSrc;
        LayoutDst layoutDst;
        this->template setShape<Element>();
        setLayout<Element>(this->_row, this->_col, layoutSrc, layoutDst);

        // call this copyGmToL1
        copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

        auto logs = AscendCCallLogger::Instance().GetLogs();
        ASSERT_EQ(logs.size(), this->_row);

        for (int i = 0; i < this->_row; i++) {
            AscendCCallLog logTileCopy = logs[i];
            this->template BaseCheck<Element>(logTileCopy);

            const AscendC::DataCopyParams* logDataCopy = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
            ASSERT_EQ(logDataCopy->blockCount, this->_cols_by_fractal);
            ASSERT_EQ(logDataCopy->blockLen, _1);                 
            ASSERT_EQ(logDataCopy->srcGap, _0);
            ASSERT_EQ(logDataCopy->dstGap, this->_row_round - 1);
        }
    } 
}

// ColumnMajor -> nZ, interval datacopy action (use DataCopy instead of nd2nz)
TYPED_TEST(TileCopyGmToL1Test, ColumnMajorTonZIntervalDataCopy)
{
    using ArchTag = TypeParam;
    if constexpr ((kEnableAtlasA2 && std::is_same_v<ArchTag, Catlass::Arch::AtlasA2>) ||
                  (kEnableAscend950 && std::is_same_v<ArchTag, Catlass::Arch::Ascend950>)) {
        using Element = half;
        using LayoutSrc = layout::ColumnMajor;
        using LayoutDst = layout::nZ;
        using GmType = Gemm::GemmType<Element, LayoutSrc>;
        constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

        CopyGmToL1IntervalDataCopy<ArchTag, GmType> copyGmToL1;
        AscendC::GlobalTensor<Element> gmTensor;
        AscendC::LocalTensor<Element> l1Tensor;

        this->template setShape<Element, true>();
        LayoutSrc layoutSrc;
        LayoutDst layoutDst;
        setLayout<Element>(this->_row, this->_col, layoutSrc, layoutDst);

        copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

        auto logs = AscendCCallLogger::Instance().GetLogs();
        ASSERT_EQ(logs.size(), this->_col);

        for (int i = 0; i < this->_col; i++) {
            AscendCCallLog logTileCopy = logs[i];
            this->template BaseCheck<Element>(logTileCopy);

            const AscendC::DataCopyParams* logDataCopy = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
            ASSERT_EQ(logDataCopy->blockCount, this->_rows_by_fractal);
            ASSERT_EQ(logDataCopy->blockLen, _1);
            ASSERT_EQ(logDataCopy->srcGap, _0);
            ASSERT_EQ(logDataCopy->dstGap, this->_col_round - 1);
        }
    }
}

// PaddingRowMajor -> zN, interval datacopy action
TYPED_TEST(TileCopyGmToL1Test, PaddingRowMajorTozNIntervalDataCopy)
{
    using ArchTag = TypeParam;
    if constexpr ((kEnableAtlasA2 && std::is_same_v<ArchTag, Catlass::Arch::AtlasA2>) ||
                  (kEnableAscend950 && std::is_same_v<ArchTag, Catlass::Arch::Ascend950>)) {
        using Element = half;
        using LayoutSrc = layout::PaddingRowMajor;
        using LayoutDst = layout::zN;
        using GmType = Gemm::GemmType<Element, LayoutSrc>;
        constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

        CopyGmToL1IntervalDataCopy<ArchTag, GmType> copyGmToL1;
        AscendC::GlobalTensor<Element> gmTensor;
        AscendC::LocalTensor<Element> l1Tensor;

        this->template setShape<Element>();
        LayoutSrc layoutSrc(this->_row, this->_col, ELE_NUM_PER_C0, C0_NUM_PER_FRACTAL);
        LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(this->_row, this->_col);

        copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

        auto logs = AscendCCallLogger::Instance().GetLogs();
        ASSERT_EQ(logs.size(), this->_row);

        for (int i = 0; i < this->_row; i++) {
            AscendCCallLog logTileCopy = logs[i];
            this->template BaseCheck<Element>(logTileCopy);

            const AscendC::DataCopyParams* logDataCopy = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
            ASSERT_EQ(logDataCopy->blockCount, this->_cols_by_fractal);
            ASSERT_EQ(logDataCopy->blockLen, _1);
            ASSERT_EQ(logDataCopy->srcGap, _0);
            ASSERT_EQ(logDataCopy->dstGap, this->_row_round - 1);
        }
    }
}

// PaddingColumnMajor -> nZ, interval datacopy action
TYPED_TEST(TileCopyGmToL1Test, PaddingColumnMajorTonZIntervalDataCopy)
{
    using ArchTag = TypeParam;
    if constexpr ((kEnableAtlasA2 && std::is_same_v<ArchTag, Catlass::Arch::AtlasA2>) ||
                  (kEnableAscend950 && std::is_same_v<ArchTag, Catlass::Arch::Ascend950>)) {
        using Element = half;
        using LayoutSrc = layout::PaddingColumnMajor;
        using LayoutDst = layout::nZ;
        using GmType = Gemm::GemmType<Element, LayoutSrc>;
        constexpr uint32_t ELE_NUM_PER_C0 = GetEleNumPerC0<Element>();

        CopyGmToL1IntervalDataCopy<ArchTag, GmType> copyGmToL1;
        AscendC::GlobalTensor<Element> gmTensor;
        AscendC::LocalTensor<Element> l1Tensor;

        this->template setShape<Element, true>();
        LayoutSrc layoutSrc(this->_row, this->_col, ELE_NUM_PER_C0, C0_NUM_PER_FRACTAL);
        LayoutDst layoutDst = LayoutDst::template MakeLayout<Element>(this->_row, this->_col);

        copyGmToL1(l1Tensor, gmTensor, layoutDst, layoutSrc);

        auto logs = AscendCCallLogger::Instance().GetLogs();
        ASSERT_EQ(logs.size(), this->_col);

        for (int i = 0; i < this->_col; i++) {
            AscendCCallLog logTileCopy = logs[i];
            this->template BaseCheck<Element>(logTileCopy);

            const AscendC::DataCopyParams* logDataCopy = logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
            ASSERT_EQ(logDataCopy->blockCount, this->_rows_by_fractal);
            ASSERT_EQ(logDataCopy->blockLen, _1);
            ASSERT_EQ(logDataCopy->srcGap, _0);
            ASSERT_EQ(logDataCopy->dstGap, this->_col_round - 1);
        }
    }
}
