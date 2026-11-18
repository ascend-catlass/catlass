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
#include <vector>
#include "stub/ascendc_test_fixture.h"
#include "stub/kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/layout/layout.hpp"

#include "tla/tensor.hpp"
#include "catlass/detail/tag_to_layout.hpp"

#include "catlass/epilogue/tile/copy_ub_to_ub_tla.hpp"
#include "stub/ascendc_logger.h"

#include "catlass/gemm/tile/common/helper.hpp"
#include "catlass/gemm/tile/common/shape.hpp"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

using namespace Catlass;
using namespace Catlass::Epilogue::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

class TileCopyUbToUbTlaTestAscend950 : public UBTileCopyTest,
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
        ASSERT_EQ(logVecCopy.name, "DataCopy");
        ASSERT_EQ(logVecCopy.args.size(), 3);
        ASSERT_EQ(logVecCopy.GetArgsTAt(0).Type(), typeid(Element));
    }

    uint32_t _srcStride = 0;
    uint32_t _dstStride = 0;
};

class TileCopyUbToUbTlaNonContiguousTestAscend950 : public UBTileCopyTest,
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
        ASSERT_EQ(logVecCopy.name, "DataCopy");
        ASSERT_EQ(logVecCopy.args.size(), 3);
        ASSERT_EQ(logVecCopy.GetArgsTAt(0).Type(), typeid(Element));
    }

    uint32_t _srcStride = 128;
    uint32_t _dstStride = 128;
};

class TileCopyUbToUbTlaFp4TestAscend950 : public UBTileCopyTest,
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
        ASSERT_EQ(logVecCopy.name, "DataCopy");
        ASSERT_EQ(logVecCopy.args.size(), 3);
        ASSERT_EQ(logVecCopy.GetArgsTAt(0).Type(), typeid(Element));
    }

    uint32_t _srcStride = 128;
    uint32_t _dstStride = 128;
};

using UbToUbCoordZero = tla::Coord<tla::Int<0>, tla::Int<0>>;
template <class Element, class Layout>
using UbToUbTensor = tla::Tensor<AscendC::LocalTensor<Element>, Layout, UbToUbCoordZero,
    AscendC::TPosition::VECCALC>;

// Data-path: RowMajor (UB) → RowMajor (UB)
// Element-type: no-except (float)
// Speciality: Contiguous rows, both gaps are zero so the whole tile is one DataCopy
TEST_P(TileCopyUbToUbTlaTestAscend950, RowMajorToRowMajorTestContiguous)
{
    using Element = float;
    constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;

    AscendC::LocalTensor<Element> ubSrc;
    AscendC::LocalTensor<Element> ubDst;

    setShape();
    const uint32_t cols = (_oriBlkLen + ELE_NUM_PER_BLK - 1) / ELE_NUM_PER_BLK;
    const uint32_t srcGap = (_srcStride + ELE_NUM_PER_BLK - 1) / ELE_NUM_PER_BLK - cols;
    const uint32_t dstGap = (_dstStride + ELE_NUM_PER_BLK - 1) / ELE_NUM_PER_BLK - cols;
    if (srcGap != 0 || dstGap != 0) {
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
    UbToUbTensor<Element, decltype(layoutSrc)> tensorUbSrc(ubSrc, layoutSrc);
    UbToUbTensor<Element, decltype(layoutDst)> tensorUbDst(ubDst, layoutDst);

    CopyUb2UbTla<Arch::Ascend950, decltype(tensorUbSrc), decltype(tensorUbDst)> copyUbToUb;
    copyUbToUb(tensorUbDst, tensorUbSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logVecCopy = logs[0];
    BaseCheck<Element>(logVecCopy);

    const auto* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
    ASSERT_EQ(dataCopyParams->blockCount, 1U);
    ASSERT_EQ(dataCopyParams->blockLen, _oriBlkCnt * cols);
    ASSERT_EQ(dataCopyParams->srcGap, 0U);
    ASSERT_EQ(dataCopyParams->dstGap, 0U);
}

// Data-path: RowMajor (UB) → RowMajor (UB)
// Element-type: no-except (float)
// Speciality: NonContiguous (one DataCopy row, src/dst datablock gaps)
TEST_P(TileCopyUbToUbTlaNonContiguousTestAscend950, RowMajorToRowMajorTestNonContiguous)
{
    using Element = float;
    constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;

    AscendC::LocalTensor<Element> ubSrc;
    AscendC::LocalTensor<Element> ubDst;

    setShape();
    const uint32_t cols = (_oriBlkLen + ELE_NUM_PER_BLK - 1) / ELE_NUM_PER_BLK;
    const uint32_t srcGap = (_srcStride + ELE_NUM_PER_BLK - 1) / ELE_NUM_PER_BLK - cols;
    const uint32_t dstGap = (_dstStride + ELE_NUM_PER_BLK - 1) / ELE_NUM_PER_BLK - cols;
    if (srcGap == 0 && dstGap == 0) {
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
    UbToUbTensor<Element, decltype(layoutSrc)> tensorUbSrc(ubSrc, layoutSrc);
    UbToUbTensor<Element, decltype(layoutDst)> tensorUbDst(ubDst, layoutDst);

    CopyUb2UbTla<Arch::Ascend950, decltype(tensorUbSrc), decltype(tensorUbDst)> copyUbToUb;
    copyUbToUb(tensorUbDst, tensorUbSrc);

    auto logs = AscendCCallLogger::Instance().GetLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logVecCopy = logs[0];
    BaseCheck<Element>(logVecCopy);

    const auto* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
    ASSERT_EQ(dataCopyParams->blockCount, _oriBlkCnt);
    ASSERT_EQ(dataCopyParams->blockLen, cols);
    ASSERT_EQ(dataCopyParams->srcGap, srcGap);
    ASSERT_EQ(dataCopyParams->dstGap, dstGap);
}

// Data-path: RowMajor (UB) → RowMajor (UB)
// Element-type: fp4 (two elements per byte, 64 elements per 32B datablock)
// Speciality: NonContiguous with unaligned tails
TEST_P(TileCopyUbToUbTlaFp4TestAscend950, RowMajorToRowMajorTestFp4NonContiguous)
{
    using Element = float4_e2m1x2_t;
    constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;

    AscendC::LocalTensor<Element> ubSrc;
    AscendC::LocalTensor<Element> ubDst;

    setShape();
    const uint32_t cols = (_oriBlkLen + ELE_NUM_PER_BLK - 1) / ELE_NUM_PER_BLK;
    const uint32_t srcGap = (_srcStride + ELE_NUM_PER_BLK - 1) / ELE_NUM_PER_BLK - cols;
    const uint32_t dstGap = (_dstStride + ELE_NUM_PER_BLK - 1) / ELE_NUM_PER_BLK - cols;
    if (srcGap == 0 && dstGap == 0) {
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
    UbToUbTensor<Element, decltype(layoutSrc)> tensorUbSrc(ubSrc, layoutSrc);
    UbToUbTensor<Element, decltype(layoutDst)> tensorUbDst(ubDst, layoutDst);

    CopyUb2UbTla<Arch::Ascend950, decltype(tensorUbSrc), decltype(tensorUbDst)> copyUbToUb;
    copyUbToUb(tensorUbDst, tensorUbSrc);

    auto logs = AscendCCallLogger::Instance().GetCopyLogs();
    ASSERT_EQ(logs.size(), 1);

    const auto& logVecCopy = logs[0];
    // fp4 is copied through a uint8_t view, so the stub records uint8_t.
    BaseCheck<uint8_t>(logVecCopy);

    const auto* dataCopyParams = logVecCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();
    ASSERT_EQ(dataCopyParams->blockCount, _oriBlkCnt);
    ASSERT_EQ(dataCopyParams->blockLen, cols);
    ASSERT_EQ(dataCopyParams->srcGap, srcGap);
    ASSERT_EQ(dataCopyParams->dstGap, dstGap);
}

INSTANTIATE_TEST_SUITE_P(
    CopyUbToUbTlaAscend950,
    TileCopyUbToUbTlaTestAscend950,
    ::testing::Values(
        /* blkLen, blkCnt, oriBlkLen, oriBlkCnt, srcStride, dstStride */
        TestVectorShapeWithStride{64U, 4U, 64U, 4U, 64U, 64U},
        TestVectorShapeWithStride{128U, 2U, 128U, 2U, 128U, 128U},
        TestVectorShapeWithStride{256U, 4U, 256U, 4U, 256U, 256U}
    )
);

INSTANTIATE_TEST_SUITE_P(
    CopyUbToUbTlaNonContiguousAscend950,
    TileCopyUbToUbTlaNonContiguousTestAscend950,
    ::testing::Values(
        /* blkLen, blkCnt, oriBlkLen, oriBlkCnt, srcStride, dstStride */
        TestVectorShapeWithStride{64U, 4U, 64U, 4U, 128U, 256U},
        TestVectorShapeWithStride{40U, 3U, 40U, 3U, 48U, 96U},
        TestVectorShapeWithStride{128U, 2U, 128U, 2U, 192U, 256U},
        TestVectorShapeWithStride{127U, 4U, 127U, 4U, 128U, 192U}
    )
);

INSTANTIATE_TEST_SUITE_P(
    CopyUbToUbTlaFp4Ascend950,
    TileCopyUbToUbTlaFp4TestAscend950,
    ::testing::Values(
        /* blkLen, blkCnt, oriBlkLen, oriBlkCnt, srcStride, dstStride */
        TestVectorShapeWithStride{64U, 4U, 64U, 4U, 128U, 256U},
        TestVectorShapeWithStride{40U, 3U, 40U, 3U, 48U, 96U},
        TestVectorShapeWithStride{128U, 2U, 128U, 2U, 192U, 256U},
        TestVectorShapeWithStride{127U, 4U, 127U, 4U, 128U, 192U}
    )
);

// Transpose path: RowMajor (UB) in, ColumnMajor (UB) out -- dst[j][i] = src[i][j] through TransDataTo5HD.
// Tiles are moved whole, so srcLd must cover RoundUp<ELE_NUM_PER_BLK>(cols) and dstLd RoundUp<TILE_ROWS>(rows);
// the allocation sizes are not exercised, as the stub records addresses only.
struct TestTransposeShape {
    uint32_t rows;  // src rows (M)
    uint32_t cols;  // src columns (N)
    uint32_t srcLd; // src row stride in elements
    uint32_t dstLd; // dst row stride in elements
};

class TileCopyUbToUbTlaTransposeTestAscend950 : public AscendCTest,
    public testing::WithParamInterface<TestTransposeShape> {
protected:
    // Distinct bases so a recorded address tells which tensor it belongs to.
    static constexpr uint64_t SRC_BASE = 0x1000;
    static constexpr uint64_t DST_BASE = 0x8000;
    static constexpr uint32_t ADDR_LIST_SIZE = 16; // NCHW_CONV_ADDR_LIST_SIZE
    static constexpr uint32_t MAX_REPEAT = 255;    // repeatTimes is a uint8_t

    static size_t NumTiles(uint32_t size, uint32_t tile)
    {
        return static_cast<size_t>((size + tile - 1) / tile);
    }

    // Address of src[row][col] and dst[row][col], both row-major; the last tile may reach past the logical shape.
    uint64_t SrcAddr(uint64_t elemSize, uint32_t row, uint32_t col) const
    {
        return SRC_BASE + elemSize * (static_cast<uint64_t>(row) * GetParam().srcLd + col);
    }

    uint64_t DstAddr(uint64_t elemSize, uint32_t row, uint32_t col) const
    {
        return DST_BASE + elemSize * (static_cast<uint64_t>(row) * GetParam().dstLd + col);
    }

    template <class Element>
    void RunTranspose()
    {
        const uint32_t rows = GetParam().rows;
        const uint32_t cols = GetParam().cols;
        AscendC::LocalTensor<Element> ubSrc(static_cast<uint32_t>(SRC_BASE), rows * GetParam().srcLd);
        AscendC::LocalTensor<Element> ubDst(static_cast<uint32_t>(DST_BASE), cols * GetParam().dstLd);

        auto layoutSrc = tla::MakeLayout(
            tla::MakeShape(rows, cols), tla::MakeStride(static_cast<int64_t>(GetParam().srcLd), tla::Int<1>{}),
            tla::MakeShape(rows, cols));
        auto layoutDst = tla::MakeLayout(
            tla::MakeShape(cols, rows), tla::MakeStride(tla::Int<1>{}, static_cast<int64_t>(GetParam().dstLd)),
            tla::MakeShape(cols, rows));
        UbToUbTensor<Element, decltype(layoutSrc)> tensorUbSrc(ubSrc, layoutSrc);
        UbToUbTensor<Element, decltype(layoutDst)> tensorUbDst(ubDst, layoutDst);

        CopyUb2UbTla<Arch::Ascend950, decltype(tensorUbSrc), decltype(tensorUbDst)> copyUbToUb;
        copyUbToUb(tensorUbDst, tensorUbSrc);
    }

    // One address list walks up to MAX_REPEAT column blocks; a single repeat has no next block to step to, so both
    // repeat strides stay 0.
    static uint32_t Repeats(uint32_t blocks)
    {
        return blocks < MAX_REPEAT ? blocks : MAX_REPEAT;
    }

    void ExpectCallShape(const AscendCCallLog& log, const std::type_info& elementType, bool dstHigh, bool srcHigh,
        uint32_t repeats) const
    {
        ASSERT_EQ(log.name, "TransDataTo5HD");
        ASSERT_EQ(log.args.size(), 7);
        ASSERT_EQ(log.GetArgsTAt(0).Type(), elementType);
        ASSERT_EQ(log.GetArgsAt(5).Type(), typeid(std::vector<uint64_t>));
        ASSERT_EQ(log.GetArgsAt(6).Type(), typeid(std::vector<uint64_t>));
        EXPECT_EQ(*log.GetArgsAt(0).Value<bool>(), dstHigh);
        EXPECT_EQ(*log.GetArgsAt(1).Value<bool>(), srcHigh);
        EXPECT_EQ(static_cast<uint32_t>(*log.GetArgsAt(2).Value<uint8_t>()), repeats);
        const uint16_t dstRepStride = repeats > 1 ? static_cast<uint16_t>(GetParam().dstLd) : uint16_t{0};
        const uint16_t srcRepStride = repeats > 1 ? uint16_t{1} : uint16_t{0};
        EXPECT_EQ(*log.GetArgsAt(3).Value<uint16_t>(), dstRepStride);
        EXPECT_EQ(*log.GetArgsAt(4).Value<uint16_t>(), srcRepStride);
    }

    // b16: a 16x16 block per repeat -- src row i+idx at column j+r*16 maps onto dst row j+r*16+idx at column i.
    void ExpectB16Calls(
        const std::vector<AscendCCallLog>& logs, const std::type_info& elementType, uint64_t elemSize) const
    {
        const uint32_t eleNumPerBlk = static_cast<uint32_t>(BYTE_PER_BLK / elemSize);
        const uint32_t rows = GetParam().rows;
        const uint32_t cols = GetParam().cols;
        const uint32_t blocksPerCall = eleNumPerBlk * MAX_REPEAT;
        ASSERT_EQ(logs.size(), NumTiles(rows, ADDR_LIST_SIZE) * NumTiles(cols, blocksPerCall));

        size_t call = 0;
        for (uint32_t i = 0; i < rows; i += ADDR_LIST_SIZE) {
            for (uint32_t j = 0; j < cols; j += blocksPerCall) {
                const uint32_t repeats = Repeats(static_cast<uint32_t>(NumTiles(cols - j, eleNumPerBlk)));
                const auto& log = logs[call++];
                ExpectCallShape(log, elementType, false, false, repeats);
                const auto* dstList = log.GetArgsAt(5).Value<std::vector<uint64_t>>();
                const auto* srcList = log.GetArgsAt(6).Value<std::vector<uint64_t>>();
                ASSERT_EQ(srcList->size(), size_t{ADDR_LIST_SIZE});
                ASSERT_EQ(dstList->size(), size_t{ADDR_LIST_SIZE});
                for (uint32_t idx = 0; idx < ADDR_LIST_SIZE; ++idx) {
                    EXPECT_EQ((*srcList)[idx], SrcAddr(elemSize, i + idx, j))
                        << "src i=" << i << " j=" << j << " idx=" << idx;
                    EXPECT_EQ((*dstList)[idx], DstAddr(elemSize, j + idx, i))
                        << "dst i=" << i << " j=" << j << " idx=" << idx;
                }
            }
        }
    }

    // b32: 16 source rows but only 8 columns per repeat; a destination row spans 16 elements, so each row is split
    // into a pair of 8-element halves in the destination list.
    void ExpectB32Calls(
        const std::vector<AscendCCallLog>& logs, const std::type_info& elementType, uint64_t elemSize) const
    {
        const uint32_t eleNumPerBlk = static_cast<uint32_t>(BYTE_PER_BLK / elemSize);
        const uint32_t rows = GetParam().rows;
        const uint32_t cols = GetParam().cols;
        const uint32_t blocksPerCall = eleNumPerBlk * MAX_REPEAT;
        ASSERT_EQ(logs.size(), NumTiles(rows, ADDR_LIST_SIZE) * NumTiles(cols, blocksPerCall));

        size_t call = 0;
        for (uint32_t i = 0; i < rows; i += ADDR_LIST_SIZE) {
            for (uint32_t j = 0; j < cols; j += blocksPerCall) {
                const uint32_t repeats = Repeats(static_cast<uint32_t>(NumTiles(cols - j, eleNumPerBlk)));
                const auto& log = logs[call++];
                ExpectCallShape(log, elementType, false, false, repeats);
                const auto* dstList = log.GetArgsAt(5).Value<std::vector<uint64_t>>();
                const auto* srcList = log.GetArgsAt(6).Value<std::vector<uint64_t>>();
                ASSERT_EQ(srcList->size(), size_t{ADDR_LIST_SIZE});
                ASSERT_EQ(dstList->size(), size_t{ADDR_LIST_SIZE});
                for (uint32_t idx = 0; idx < ADDR_LIST_SIZE; ++idx) {
                    EXPECT_EQ((*srcList)[idx], SrcAddr(elemSize, i + idx, j))
                        << "src i=" << i << " j=" << j << " idx=" << idx;
                }
                for (uint32_t k = 0; k < eleNumPerBlk; ++k) {
                    const uint64_t rowAddr = DstAddr(elemSize, j + k, i);
                    EXPECT_EQ((*dstList)[2 * k], rowAddr) << "dst i=" << i << " j=" << j << " k=" << k;
                    EXPECT_EQ((*dstList)[2 * k + 1], rowAddr + eleNumPerBlk * elemSize)
                        << "dst i=" << i << " j=" << j << " k=" << k;
                }
            }
        }
    }

    // b8: 32 elements per DataBlock but only 16 list entries, so a 32x32 tile is moved as four quadrants --
    // srcHighHalf picks the source column half and dstHighHalf the destination half written. One list walks up to
    // MAX_REPEAT of those tiles along the columns.
    void ExpectB8Calls(
        const std::vector<AscendCCallLog>& logs, const std::type_info& elementType, uint64_t elemSize) const
    {
        constexpr uint32_t TILE_ROWS_B8 = 2 * ADDR_LIST_SIZE;
        const uint32_t rows = GetParam().rows;
        const uint32_t cols = GetParam().cols;
        const uint32_t blocksPerCall = TILE_ROWS_B8 * MAX_REPEAT;
        ASSERT_EQ(logs.size(), NumTiles(rows, TILE_ROWS_B8) * NumTiles(cols, blocksPerCall) * 4);

        size_t call = 0;
        for (uint32_t i = 0; i < rows; i += TILE_ROWS_B8) {
            for (uint32_t j = 0; j < cols; j += blocksPerCall) {
                const uint32_t repeats = Repeats(static_cast<uint32_t>(NumTiles(cols - j, TILE_ROWS_B8)));
                for (uint32_t q = 0; q < 4; ++q) {
                    const bool dstHigh = (q & 2) != 0;
                    const bool srcHigh = (q & 1) != 0;
                    const auto& log = logs[call++];
                    ExpectCallShape(log, elementType, dstHigh, srcHigh, repeats);
                    const auto* dstList = log.GetArgsAt(5).Value<std::vector<uint64_t>>();
                    const auto* srcList = log.GetArgsAt(6).Value<std::vector<uint64_t>>();
                    ASSERT_EQ(srcList->size(), size_t{ADDR_LIST_SIZE});
                    ASSERT_EQ(dstList->size(), size_t{ADDR_LIST_SIZE});
                    const uint32_t rowBase = i + (dstHigh ? ADDR_LIST_SIZE : 0);
                    const uint32_t colBase = j + (srcHigh ? ADDR_LIST_SIZE : 0);
                    for (uint32_t idx = 0; idx < ADDR_LIST_SIZE; ++idx) {
                        // The source list keeps the tile column j: srcHighHalf applies the +16 column offset.
                        EXPECT_EQ((*srcList)[idx], SrcAddr(elemSize, rowBase + idx, j))
                            << "i=" << i << " j=" << j << " q=" << q << " idx=" << idx;
                        EXPECT_EQ((*dstList)[idx], DstAddr(elemSize, colBase + idx, i))
                            << "i=" << i << " j=" << j << " q=" << q << " idx=" << idx;
                    }
                }
            }
        }
    }
};

TEST_P(TileCopyUbToUbTlaTransposeTestAscend950, RowMajorToColumnMajorB16Half)
{
    RunTranspose<half>();
    ExpectB16Calls(AscendCCallLogger::Instance().GetLogs(), typeid(half), sizeof(half));
}

TEST_P(TileCopyUbToUbTlaTransposeTestAscend950, RowMajorToColumnMajorB16BFloat16)
{
    RunTranspose<bfloat16_t>();
    ExpectB16Calls(AscendCCallLogger::Instance().GetLogs(), typeid(bfloat16_t), sizeof(bfloat16_t));
}

TEST_P(TileCopyUbToUbTlaTransposeTestAscend950, RowMajorToColumnMajorB32Float)
{
    RunTranspose<float>();
    ExpectB32Calls(AscendCCallLogger::Instance().GetLogs(), typeid(float), sizeof(float));
}

TEST_P(TileCopyUbToUbTlaTransposeTestAscend950, RowMajorToColumnMajorB8Fp8)
{
    RunTranspose<float8_e4m3_t>();
    // The b8 branch calls the uint8_t overload, so the stub records uint8_t.
    ExpectB8Calls(AscendCCallLogger::Instance().GetLogs(), typeid(uint8_t), sizeof(float8_e4m3_t));
}

TEST_P(TileCopyUbToUbTlaTransposeTestAscend950, RowMajorToColumnMajorB8Int8)
{
    RunTranspose<int8_t>();
    ExpectB8Calls(AscendCCallLogger::Instance().GetLogs(), typeid(uint8_t), sizeof(int8_t));
}

INSTANTIATE_TEST_SUITE_P(
    CopyUbToUbTlaTransposeAscend950,
    TileCopyUbToUbTlaTransposeTestAscend950,
    ::testing::Values(
        /* rows, cols, srcLd, dstLd -- row strides are 32-element aligned, which covers every branch. The wide tile
           splits into 255 + 1 column blocks at b32 (8 elements per block) and stays one list at b16/b8; the 2040-column
           tile is the widest b32 list a single call can carry (exactly 255 repeats). */
        TestTransposeShape{16U, 16U, 32U, 32U},
        TestTransposeShape{32U, 32U, 32U, 32U},
        TestTransposeShape{17U, 33U, 64U, 32U},
        TestTransposeShape{8U, 24U, 32U, 32U},
        TestTransposeShape{40U, 40U, 64U, 64U},
        TestTransposeShape{96U, 64U, 96U, 96U},
        TestTransposeShape{16U, 2040U, 2048U, 32U},
        TestTransposeShape{16U, 2048U, 2080U, 32U}
    )
);

#endif // CATLASS_ARCH == 3510
