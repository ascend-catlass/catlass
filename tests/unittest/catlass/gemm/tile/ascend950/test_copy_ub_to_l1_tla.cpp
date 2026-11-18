/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Ascend950 UB -> L1 tile copies: verifies the DataCopy descriptors emitted by each CopyUb2L1Tla specialization.

#include <gtest/gtest.h>
#include "stub/ascendc_test_fixture.h"
#include "stub/kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/layout/layout.hpp"

#include "tla/tensor.hpp"
#include "catlass/detail/tag_to_layout.hpp"

#include "catlass/epilogue/tile/copy_ub_to_l1_tla.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

using namespace Catlass;
using namespace Catlass::Epilogue::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

namespace {

// Expected descriptor fields are spelled out in the parameter structs so that the
// test does not re-derive them with the formulas used by the implementation.
struct UbToL1RowMajorShape {
    uint32_t rows;
    uint32_t cols;
    // Source row pitch beyond the copied column blocks, in blocks.
    uint32_t extraGap;
};

struct UbToL1zNShape {
    uint32_t rows;
    uint32_t cols;
    uint32_t blockCount;
    uint32_t blockLen;
    uint32_t srcGap;
    uint32_t dstGap;
};

struct UbToL1MxShape {
    uint32_t srcRows;
    uint32_t srcCols;
    uint32_t dstRows;
    uint32_t dstCols;
    uint32_t blockCount;
    uint32_t blockLen;
    uint32_t dstGap;
};

using UbToL1CoordZero = tla::Coord<tla::Int<0>, tla::Int<0>>;

template <class Element, class Layout>
using UbToL1TensorSrc =
    tla::Tensor<AscendC::LocalTensor<Element>, Layout, UbToL1CoordZero, AscendC::TPosition::VECCALC>;

template <class Element, class Layout>
using UbToL1TensorDst =
    tla::Tensor<AscendC::LocalTensor<Element>, Layout, UbToL1CoordZero, AscendC::TPosition::A1>;

class UbToL1TlaTestBase : public UBTileCopyTest {
protected:
    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    void BaseCheck(const AscendCCallLog& log, std::type_index elementType)
    {
        ASSERT_EQ(log.name, "DataCopy");
        ASSERT_EQ(log.args.size(), 3);
        ASSERT_EQ(log.GetArgsTAt(0).Type(), elementType);
    }

    void CheckParams(
        const AscendCCallLog& log, uint32_t blockCount, uint32_t blockLen, uint32_t srcGap, uint32_t dstGap)
    {
        const auto paramsArg = log.GetArgsAt(2);
        const auto* params = paramsArg.Value<AscendC::DataCopyParams>();
        ASSERT_NE(params, nullptr);
        ASSERT_EQ(params->blockCount, blockCount);
        ASSERT_EQ(params->blockLen, blockLen);
        ASSERT_EQ(params->srcGap, srcGap);
        ASSERT_EQ(params->dstGap, dstGap);
    }
};

class TileCopyUbToL1TlaRowMajorTestAscend950 : public UbToL1TlaTestBase,
                                              public testing::WithParamInterface<UbToL1RowMajorShape> {};

class TileCopyUbToL1TlaUnAlignTestAscend950 : public UbToL1TlaTestBase,
                                             public testing::WithParamInterface<UbToL1zNShape> {};

class TileCopyUbToL1TlazNzNTestAscend950 : public UbToL1TlaTestBase,
                                          public testing::WithParamInterface<UbToL1zNShape> {};

class TileCopyUbToL1TlaMxZZTestAscend950 : public UbToL1TlaTestBase,
                                          public testing::WithParamInterface<UbToL1MxShape> {};

class TileCopyUbToL1TlaMxNNTestAscend950 : public UbToL1TlaTestBase,
                                          public testing::WithParamInterface<UbToL1MxShape> {};

TEST_P(TileCopyUbToL1TlaRowMajorTestAscend950, RowMajorTozNFp32)
{
    using Element = float;
    constexpr uint32_t FRACTAL_BLOCKS = BYTE_PER_FRACTAL / BYTE_PER_BLK;

    const uint32_t rows = GetParam().rows;
    const uint32_t cols = GetParam().cols;
    const uint32_t extraGap = GetParam().extraGap;
    ASSERT_EQ(rows, 16U);

    const uint32_t eleNumPerBlk = BytesToElements<Element>(BYTE_PER_BLK);
    const uint32_t blockLen = (ElementsToBytes<Element>(cols) + BYTE_PER_BLK - 1) / BYTE_PER_BLK;

    AscendC::LocalTensor<Element> ubSrc;
    AscendC::LocalTensor<Element> l1Dst;

    auto layoutSrc = tla::MakeLayout(
        tla::MakeShape(rows, cols),
        tla::MakeStride((int64_t)(blockLen + extraGap) * eleNumPerBlk, tla::Int<1>{}), tla::MakeShape(rows, cols));
    auto layoutDst = tla::MakeLayout<Element, Catlass::layout::zN>(tla::Int<16>{}, cols);

    UbToL1TensorSrc<Element, decltype(layoutSrc)> tensorUbSrc(ubSrc, layoutSrc);
    UbToL1TensorDst<Element, decltype(layoutDst)> tensorL1Dst(l1Dst, layoutDst);

    CopyUb2L1Tla<Arch::Ascend950, decltype(tensorUbSrc), decltype(tensorL1Dst)> copyUbToL1;
    copyUbToL1(tensorL1Dst, tensorUbSrc);

    const auto logs = AscendCCallLogger::Instance().GetLogs();
    if (blockLen > rows) {
        // Column blocks exceed the rows: one copy per row block.
        ASSERT_EQ(logs.size(), rows);
        for (const auto& log : logs) {
            BaseCheck(log, typeid(Element));
            CheckParams(log, blockLen, 1U, 0U, FRACTAL_BLOCKS - 1);
        }
    } else {
        ASSERT_EQ(logs.size(), blockLen);
        for (const auto& log : logs) {
            BaseCheck(log, typeid(Element));
            CheckParams(log, rows, 1U, blockLen + extraGap - 1U, 0U);
        }
    }
}

TEST_P(TileCopyUbToL1TlaRowMajorTestAscend950, RowMajorTozNFp4)
{
    using Element = float4_e2m1x2_t;
    constexpr uint32_t FRACTAL_BLOCKS = BYTE_PER_FRACTAL / BYTE_PER_BLK;

    const uint32_t rows = GetParam().rows;
    const uint32_t cols = GetParam().cols;
    const uint32_t extraGap = GetParam().extraGap;
    ASSERT_EQ(rows, 16U);

    const uint32_t eleNumPerBlk = BytesToElements<Element>(BYTE_PER_BLK);
    const uint32_t blockLen = (ElementsToBytes<Element>(cols) + BYTE_PER_BLK - 1) / BYTE_PER_BLK;

    AscendC::LocalTensor<Element> ubSrc;
    AscendC::LocalTensor<Element> l1Dst;

    auto layoutSrc = tla::MakeLayout(
        tla::MakeShape(rows, cols),
        tla::MakeStride((int64_t)(blockLen + extraGap) * eleNumPerBlk, tla::Int<1>{}), tla::MakeShape(rows, cols));
    auto layoutDst = tla::MakeLayout<Element, Catlass::layout::zN>(tla::Int<16>{}, cols);

    UbToL1TensorSrc<Element, decltype(layoutSrc)> tensorUbSrc(ubSrc, layoutSrc);
    UbToL1TensorDst<Element, decltype(layoutDst)> tensorL1Dst(l1Dst, layoutDst);

    CopyUb2L1Tla<Arch::Ascend950, decltype(tensorUbSrc), decltype(tensorL1Dst)> copyUbToL1;
    copyUbToL1(tensorL1Dst, tensorUbSrc);

    const auto logs = AscendCCallLogger::Instance().GetCopyLogs();
    if (blockLen > rows) {
        ASSERT_EQ(logs.size(), rows);
        for (const auto& log : logs) {
            // Packed fp4 is copied through a byte view.
            BaseCheck(log, typeid(uint8_t));
            CheckParams(log, blockLen, 1U, 0U, FRACTAL_BLOCKS - 1);
        }
    } else {
        ASSERT_EQ(logs.size(), blockLen);
        for (const auto& log : logs) {
            BaseCheck(log, typeid(uint8_t));
            CheckParams(log, rows, 1U, blockLen + extraGap - 1U, 0U);
        }
    }
}

TEST_P(TileCopyUbToL1TlaUnAlignTestAscend950, zNUnAlignTozNFp32)
{
    using Element = float;
    constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    constexpr uint32_t FRACTAL_BLOCKS = BYTE_PER_FRACTAL / BYTE_PER_BLK;

    const uint32_t rows = GetParam().rows;
    const uint32_t cols = GetParam().cols;
    const uint32_t blockCount = GetParam().blockCount;
    const uint32_t blockLen = GetParam().blockLen;
    const uint32_t srcGap = GetParam().srcGap;
    const uint32_t dstGap = GetParam().dstGap;
    ASSERT_EQ(rows, blockLen);
    ASSERT_LE(rows, ELE_NUM_PER_C0);
    ASSERT_EQ(dstGap, FRACTAL_BLOCKS - rows);

    AscendC::LocalTensor<Element> ubSrc;
    AscendC::LocalTensor<Element> l1Dst;

    auto layoutSrc = tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(rows, tla::Int<1>{}), tla::MakeShape(tla::Int<ELE_NUM_PER_C0>{}, blockCount)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<ELE_NUM_PER_C0>{}, tla::Int<1>{}),
            tla::MakeStride(tla::Int<1>{}, (int64_t)(blockLen + srcGap) * ELE_NUM_PER_C0)),
        tla::MakeShape(rows, cols));
    auto layoutDst = tla::MakeLayout<Element, Catlass::layout::zN>(tla::Int<16>{}, cols);

    UbToL1TensorSrc<Element, decltype(layoutSrc)> tensorUbSrc(ubSrc, layoutSrc);
    UbToL1TensorDst<Element, decltype(layoutDst)> tensorL1Dst(l1Dst, layoutDst);

    CopyUb2L1Tla<Arch::Ascend950, decltype(tensorUbSrc), decltype(tensorL1Dst)> copyUbToL1;
    copyUbToL1(tensorL1Dst, tensorUbSrc);

    const auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1U);
    BaseCheck(logs.front(), typeid(Element));
    CheckParams(logs.front(), blockCount, blockLen, srcGap, dstGap);
}

TEST_P(TileCopyUbToL1TlazNzNTestAscend950, zNTozNFp32)
{
    using Element = float;
    constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    const uint32_t rows = GetParam().rows;
    const uint32_t cols = GetParam().cols;
    const uint32_t blockCount = GetParam().blockCount;
    const uint32_t blockLen = GetParam().blockLen;
    const uint32_t srcGap = GetParam().srcGap;
    const uint32_t dstGap = GetParam().dstGap;
    ASSERT_EQ(rows, blockLen);
    // Two fractal rows are required to keep this layout out of the zNUnAlign specialization.
    ASSERT_TRUE(rows > C0_NUM_PER_FRACTAL);

    AscendC::LocalTensor<Element> ubSrc;
    AscendC::LocalTensor<Element> l1Dst;

    const uint32_t fractalRows = (rows + C0_NUM_PER_FRACTAL - 1) / C0_NUM_PER_FRACTAL;
    auto layoutSrc = tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<C0_NUM_PER_FRACTAL>{}, fractalRows),
            tla::MakeShape(tla::Int<ELE_NUM_PER_C0>{}, blockCount)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<ELE_NUM_PER_C0>{}, tla::Int<ELE_NUM_PER_FRACTAL>{}),
            tla::MakeStride(tla::Int<1>{}, (int64_t)(blockLen + srcGap) * ELE_NUM_PER_C0)),
        tla::MakeShape(rows, cols));
    auto layoutDst = tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<C0_NUM_PER_FRACTAL>{}, fractalRows),
            tla::MakeShape(tla::Int<ELE_NUM_PER_C0>{}, blockCount)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<ELE_NUM_PER_C0>{}, tla::Int<ELE_NUM_PER_FRACTAL>{}),
            tla::MakeStride(tla::Int<1>{}, (int64_t)(blockLen + dstGap) * ELE_NUM_PER_C0)),
        tla::MakeShape(rows, cols));

    UbToL1TensorSrc<Element, decltype(layoutSrc)> tensorUbSrc(ubSrc, layoutSrc);
    UbToL1TensorDst<Element, decltype(layoutDst)> tensorL1Dst(l1Dst, layoutDst);

    CopyUb2L1Tla<Arch::Ascend950, decltype(tensorUbSrc), decltype(tensorL1Dst)> copyUbToL1;
    copyUbToL1(tensorL1Dst, tensorUbSrc);

    const auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1U);
    BaseCheck(logs.front(), typeid(Element));
    CheckParams(logs.front(), blockCount, blockLen, srcGap, dstGap);
}

TEST_P(TileCopyUbToL1TlaMxZZTestAscend950, MxzZTozZ)
{
    using Element = float8_e8m0_t;

    const uint32_t srcRows = GetParam().srcRows;
    const uint32_t srcCols = GetParam().srcCols;
    const uint32_t dstRows = GetParam().dstRows;
    const uint32_t dstCols = GetParam().dstCols;
    const uint32_t blockCount = GetParam().blockCount;
    const uint32_t blockLen = GetParam().blockLen;
    const uint32_t dstGap = GetParam().dstGap;

    AscendC::LocalTensor<Element> ubSrc;
    AscendC::LocalTensor<Element> l1Dst;

    auto layoutSrc = tla::MakeMxScaleLayout<Element, Catlass::layout::zZ, false>(srcRows, srcCols);
    auto layoutDst = tla::MakeMxScaleLayout<Element, Catlass::layout::zZ, false>(dstRows, dstCols);

    UbToL1TensorSrc<Element, decltype(layoutSrc)> tensorUbSrc(ubSrc, layoutSrc);
    UbToL1TensorDst<Element, decltype(layoutDst)> tensorL1Dst(l1Dst, layoutDst);

    CopyUb2L1Tla<Arch::Ascend950, decltype(tensorUbSrc), decltype(tensorL1Dst)> copyUbToL1;
    copyUbToL1(tensorL1Dst, tensorUbSrc);

    const auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1U);
    BaseCheck(logs.front(), typeid(Element));
    CheckParams(logs.front(), blockCount, blockLen, 0U, dstGap);
}

TEST_P(TileCopyUbToL1TlaMxNNTestAscend950, MxnNNonN)
{
    using Element = float8_e8m0_t;

    const uint32_t srcRows = GetParam().srcRows;
    const uint32_t srcCols = GetParam().srcCols;
    const uint32_t dstRows = GetParam().dstRows;
    const uint32_t dstCols = GetParam().dstCols;
    const uint32_t blockCount = GetParam().blockCount;
    const uint32_t blockLen = GetParam().blockLen;
    const uint32_t dstGap = GetParam().dstGap;

    AscendC::LocalTensor<Element> ubSrc;
    AscendC::LocalTensor<Element> l1Dst;

    auto layoutSrc = tla::MakeMxScaleLayout<Element, Catlass::layout::nN, true>(srcRows, srcCols);
    auto layoutDst = tla::MakeMxScaleLayout<Element, Catlass::layout::nN, true>(dstRows, dstCols);

    UbToL1TensorSrc<Element, decltype(layoutSrc)> tensorUbSrc(ubSrc, layoutSrc);
    UbToL1TensorDst<Element, decltype(layoutDst)> tensorL1Dst(l1Dst, layoutDst);

    CopyUb2L1Tla<Arch::Ascend950, decltype(tensorUbSrc), decltype(tensorL1Dst)> copyUbToL1;
    copyUbToL1(tensorL1Dst, tensorUbSrc);

    const auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1U);
    BaseCheck(logs.front(), typeid(Element));
    CheckParams(logs.front(), blockCount, blockLen, 0U, dstGap);
}

INSTANTIATE_TEST_SUITE_P(
    RowMajorShapes, TileCopyUbToL1TlaRowMajorTestAscend950,
    testing::Values(
        UbToL1RowMajorShape{16U, 64U, 0U},
        UbToL1RowMajorShape{16U, 128U, 0U},
        UbToL1RowMajorShape{16U, 129U, 0U},
        UbToL1RowMajorShape{16U, 256U, 0U},
        UbToL1RowMajorShape{16U, 64U, 8U},
        UbToL1RowMajorShape{16U, 512U, 0U},
        UbToL1RowMajorShape{16U, 2048U, 0U},
        UbToL1RowMajorShape{16U, 128U, 4U}));

INSTANTIATE_TEST_SUITE_P(
    UnAlignShapes, TileCopyUbToL1TlaUnAlignTestAscend950,
    testing::Values(
        UbToL1zNShape{5U, 64U, 8U, 5U, 11U, 11U},
        UbToL1zNShape{3U, 80U, 10U, 3U, 9U, 13U},
        UbToL1zNShape{8U, 32U, 4U, 8U, 0U, 8U},
        UbToL1zNShape{4U, 40U, 5U, 4U, 4U, 12U}));

INSTANTIATE_TEST_SUITE_P(
    zNzNShapes, TileCopyUbToL1TlazNzNTestAscend950,
    testing::Values(
        UbToL1zNShape{32U, 64U, 8U, 32U, 0U, 0U},
        UbToL1zNShape{32U, 40U, 5U, 32U, 0U, 8U},
        UbToL1zNShape{32U, 128U, 16U, 32U, 8U, 0U},
        UbToL1zNShape{64U, 96U, 12U, 64U, 0U, 0U}));

INSTANTIATE_TEST_SUITE_P(
    MxzZShapes, TileCopyUbToL1TlaMxZZTestAscend950,
    testing::Values(
        UbToL1MxShape{32U, 8U, 32U, 8U, 2U, 4U, 0U},
        UbToL1MxShape{32U, 8U, 32U, 32U, 2U, 4U, 12U},
        UbToL1MxShape{32U, 64U, 32U, 64U, 2U, 32U, 0U},
        UbToL1MxShape{17U, 6U, 17U, 6U, 2U, 3U, 0U},
        UbToL1MxShape{16U, 2U, 16U, 4U, 1U, 1U, 1U},
        UbToL1MxShape{8U, 8U, 8U, 8U, 1U, 4U, 0U},
        UbToL1MxShape{17U, 17U, 17U, 17U, 2U, 9U, 0U}));

INSTANTIATE_TEST_SUITE_P(
    MxnNShapes, TileCopyUbToL1TlaMxNNTestAscend950,
    testing::Values(
        UbToL1MxShape{32U, 8U, 32U, 8U, 1U, 16U, 0U},
        UbToL1MxShape{32U, 64U, 32U, 64U, 4U, 16U, 0U},
        UbToL1MxShape{32U, 8U, 64U, 8U, 1U, 16U, 16U},
        UbToL1MxShape{5U, 20U, 5U, 20U, 2U, 3U, 0U},
        UbToL1MxShape{2U, 32U, 6U, 32U, 2U, 1U, 2U}));

} // namespace

#endif // CATLASS_ARCH == 3510
