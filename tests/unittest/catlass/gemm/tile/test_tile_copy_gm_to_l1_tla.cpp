/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "stub/ascendc_test_fixture.h"
#include "stub/kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/layout/layout.hpp"

#include "tla/tensor.hpp"
#include "catlass/detail/tag_to_layout.hpp"

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "stub/ascendc_logger.h"

#include "common/helper.hpp"
#include "common/shape.hpp"

using namespace Catlass;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Test;
using namespace Catlass::Test::Helper;

// Pre-defined types
using CoordZero = tla::Coord<tla::Int<0>, tla::Int<0>>;
template <class Element, class LayoutSrc>
using TensorSrc = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutSrc, CoordZero, AscendC::TPosition::GM>;
template <class Element, class LayoutDst>
using TensorDst = tla::Tensor<AscendC::LocalTensor<Element>, LayoutDst, CoordZero, AscendC::TPosition::A1>;

template <typename ArchTag_>
class TypedTileCopyGmToL1TlaTest : public TileCopyTlaTest,
    public testing::WithParamInterface<TestMatrixShapeWithCoord> {
protected:
    using ArchTag = ArchTag_;

    void SetUp() override
    {
        AscendCTest::SetUp();
    }

    template <class Element, bool isSrcTrans = false, bool isDstTrans = false>
    void setShape()
    {
        const auto& param = GetParam();
        _row = param.row;
        _col = param.col;
        _row_coord = param.rowCoord;
        _col_coord = param.colCoord;
        _dst_row = param.dstRow;
        _dst_col = param.dstCol;
        TileCopyTlaTest::_setShape<Element, isSrcTrans, isDstTrans>(_row, _col);
        _row_residue = _row - _row_coord;
        _col_residue = _col - _col_coord;

        ASSERT_TRUE(TileCopyTlaTest::isValidDataCopy()) << "Coord & dst shape extends original shape.";
    }

    template <class Element, bool useCopyPad = false>
    void BaseCheck(const AscendCCallLog& logTileCopy)
    {
        ASSERT_EQ(logTileCopy.name, useCopyPad ? "DataCopyPad" : "DataCopy");
        ASSERT_EQ(logTileCopy.args.size(), useCopyPad ? 4 : 3);

        const std::type_index& T0 = logTileCopy.GetArgsTAt(0).Type();
        ASSERT_EQ(T0, typeid(Element));
    }

    template <class Element, class LayoutSrcTag, class LayoutDstTag, bool UseGetTile = true>
    auto MakeCall()
    {
        return MakeCallImpl<Element, LayoutSrcTag, LayoutDstTag, UseGetTile,
            isTrans_v<LayoutSrcTag>, isTrans_v<LayoutDstTag>>();
    }

    template <class Element, class LayoutSrcTag, class LayoutDstTag,
        bool UseGetTile = true, bool IsSrcTrans = false, bool IsDstTrans = false,
        class _CopyGmToL1 = void>
    auto MakeCallImpl()
    {
        // Step 1: Determine the Layout (using tla::tuple)
        using LayoutSrc = detail::TagToLayout_t<Element, LayoutSrcTag>;
        using LayoutDst = detail::TagToLayout_t<Element, LayoutDstTag>;

        // Step 2: Make AscendC tensor
        AscendC::GlobalTensor<Element> gmSrc;
        AscendC::LocalTensor<Element> l1Dst;

        // Step 3: Get layout
        setShape<Element, IsSrcTrans, IsDstTrans>();
        auto layoutSrc = tla::MakeLayout<Element, LayoutSrcTag>(_row, _col);
        auto layoutDst = tla::MakeLayout<Element, LayoutDstTag>(_dst_row, _dst_col);

        // Step 4: Create tla tensors
        auto tensorGm = tla::MakeTensor(gmSrc, layoutSrc, Arch::PositionGM{});
        TensorDst<Element, LayoutDst> tensorL1(l1Dst, layoutDst);

        // Step 5: Call this copyGmToL1
        using CopyGmToL1 = std::conditional_t<
            std::is_void_v<_CopyGmToL1>,
            TileCopyTla<ArchTag, TensorSrc<Element, LayoutSrc>, TensorDst<Element, LayoutDst>>, _CopyGmToL1>;

        CopyGmToL1 copyGmToL1;
        if constexpr (UseGetTile) {
            auto tensorGmBlock =
                GetTile(tensorGm, tla::MakeCoord(_row_coord, _col_coord), tla::MakeShape(_row_residue, _col_residue));
            copyGmToL1(tensorL1, tensorGmBlock);
        } else {
            copyGmToL1(tensorL1, tensorGm);
        }

        // Step 6: Get final logs
        return AscendCCallLogger::Instance().GetLogs();
    }

    template <class Element, class LayoutSrcTag, class LayoutDstTag, bool IsSrcTrans = false, bool IsDstTrans = false>
    auto MakeCallExt()
    {
        using LayoutSrc = detail::TagToLayout_t<Element, LayoutSrcTag>;
        using LayoutDst = detail::TagToLayout_t<Element, LayoutDstTag>;

        AscendC::GlobalTensor<Element> gmSrc;
        AscendC::LocalTensor<Element> l1Dst;

        setShape<Element, IsSrcTrans, IsDstTrans>();
        auto layoutSrc = tla::MakeLayout<Element, LayoutSrcTag>(_row, _col);
        auto layoutDst = tla::MakeLayout<Element, LayoutDstTag>(_dst_row, _dst_col);

        TensorSrc<Element, LayoutSrc> tensorGm(gmSrc, layoutSrc);
        TensorDst<Element, LayoutDst> tensorL1(l1Dst, layoutDst);

        tla::Shape<uint32_t, uint32_t> actualShape{_row, _col};

        TileCopyTlaExt<ArchTag, decltype(tensorGm), decltype(tensorL1), LayoutSrcTag, LayoutDstTag> copyGmToL1;
        copyGmToL1(tensorL1, tensorGm, actualShape);

        return AscendCCallLogger::Instance().GetLogs();
    }

    template <class Element, class LayoutSrcTag, class LayoutDstTag, bool IsSrcTrans = false, bool IsDstTrans = false>
    auto MakeCallSparse()
    {
        using LayoutSrc = detail::TagToLayout_t<Element, LayoutSrcTag>;
        using LayoutDst = detail::TagToLayout_t<Element, LayoutDstTag>;

        AscendC::GlobalTensor<Element> gmSrc;
        AscendC::LocalTensor<Element> l1Dst;

        setShape<Element, IsSrcTrans, IsDstTrans>();
        auto layoutSrc = tla::MakeLayout<Element, LayoutSrcTag>(_row, _col);
        auto layoutDst = tla::MakeLayout<Element, LayoutDstTag>(_dst_row, _dst_col);

        TensorSrc<Element, LayoutSrc> tensorGm(gmSrc, layoutSrc);
        TensorDst<Element, LayoutDst> tensorL1(l1Dst, layoutDst);

        TileCopySparseTla<ArchTag, decltype(tensorGm), decltype(tensorL1)> copyGmToL1;
        copyGmToL1(tensorL1, tensorGm);

        return AscendCCallLogger::Instance().GetLogs();
    }

    template <class Element>
    auto MakeCallVector()
    {
        using LayoutTag = layout::VectorLayout;
        using Layout = detail::TagToLayout_t<Element, LayoutTag>;

        AscendC::GlobalTensor<Element> gmSrc;
        AscendC::LocalTensor<Element> l1Dst;

        setShape<Element>();
        auto layoutSrc = Layout{_col};
        auto layoutDst = Layout{_col};

        using VectorSrc = tla::Tensor<AscendC::GlobalTensor<Element>, Layout, CoordZero, AscendC::TPosition::GM>;
        using VectorDst = tla::Tensor<AscendC::LocalTensor<Element>, Layout, CoordZero, AscendC::TPosition::A1>;

        VectorSrc tensorGm(gmSrc, layoutSrc);
        VectorDst tensorL1(l1Dst, layoutDst);

        TileCopyTla<ArchTag, VectorSrc, VectorDst> copyGmToL1;
        copyGmToL1(tensorL1, tensorGm);

        return AscendCCallLogger::Instance().GetLogs();
    }

protected:
    uint32_t _row_residue = 0;
    uint32_t _col_residue = 0;
};

// ============================================================================
// Test helper macro — defines a complete test body for a given SuiteAlias
// ============================================================================
#define DEFINE_GMTOL1TLA_TESTS(SuiteName)                                                                              \
                                                                                                                       \
    /* Data-path: GM RowMajor → L1 zN */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: GetTile sub-tile (Nd2Nz on coord-offset residue region) */ \
    TEST_P(SuiteName, RowMajorTozNTestGetTile)                                                                   \
    {                                                                                                                  \
        using Element      = float;                                                                                    \
        using LayoutSrcTag = layout::RowMajor;                                                                         \
        using LayoutDstTag = layout::zN;                                                                               \
                                                                                                                       \
        auto logs = MakeCall<Element, LayoutSrcTag, LayoutDstTag>();                                                   \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        AscendCCallLog logTileCopy = logs[0];                                                                          \
        BaseCheck<Element>(logTileCopy);                                                                               \
                                                                                                                       \
        uint32_t srcOffset = _row_coord * _col + _col_coord;                                                           \
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);                                                          \
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), srcOffset * sizeof(Element));                                \
                                                                                                                       \
        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();                \
        ASSERT_EQ(nd2nzArg->ndNum, _1);                                                                               \
        ASSERT_EQ(nd2nzArg->nValue, _row_residue);                                                                    \
        ASSERT_EQ(nd2nzArg->dValue, _col_residue);                                                                    \
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);                                                                   \
        ASSERT_EQ(nd2nzArg->srcDValue, _col);                                                                         \
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _dst_row_round);                                                           \
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);                                                                        \
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);                                                                   \
    }                                                                                                                  \
                                                                                                                       \
    /* Data-path: GM ColumnMajor → L1 nZ */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: GetTile sub-tile (Nd2Nz, n/d swapped for column-major) */ \
    TEST_P(SuiteName, ColumnMajorTonZTestGetTile)                                                                \
    {                                                                                                                  \
        using Element      = float;                                                                                    \
        using LayoutSrcTag = layout::ColumnMajor;                                                                      \
        using LayoutDstTag = layout::nZ;                                                                               \
                                                                                                                       \
        auto logs = MakeCall<Element, LayoutSrcTag, LayoutDstTag>();                                                   \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        AscendCCallLog logTileCopy = logs[0];                                                                          \
        BaseCheck<Element>(logTileCopy);                                                                               \
                                                                                                                       \
        uint32_t srcOffset = _row_coord * _col + _col_coord;                                                           \
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);                                                          \
                                                                                                                       \
        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();                \
        ASSERT_EQ(nd2nzArg->ndNum, _1);                                                                               \
        ASSERT_EQ(nd2nzArg->nValue, _col_residue);                                                                    \
        ASSERT_EQ(nd2nzArg->dValue, _row_residue);                                                                    \
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);                                                                   \
        ASSERT_EQ(nd2nzArg->srcDValue, _row);                                                                         \
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _dst_col_round);                                                           \
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);                                                                        \
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);                                                                   \
    }                                                                                                                  \
                                                                                                                       \
    /* Data-path: GM zN → L1 zN */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: fractal whole-tensor DataCopy (column-fractal blocks, no nd2nz) */ \
    TEST_P(SuiteName, zNTozNTestFractal)                                                                         \
    {                                                                                                                  \
        using Element      = float;                                                                                    \
        using LayoutSrcTag = layout::zN;                                                                               \
        using LayoutDstTag = layout::zN;                                                                               \
                                                                                                                       \
        auto logs = MakeCall<Element, LayoutSrcTag, LayoutDstTag, false>();                                            \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        AscendCCallLog logTileCopy = logs[0];                                                                          \
        constexpr bool useCopyPad = std::is_same_v<ArchTag, Arch::Ascend950>;                                          \
        BaseCheck<Element, useCopyPad>(logTileCopy);                                                                   \
                                                                                                                       \
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);                                                          \
                                                                                                                       \
        if constexpr (std::is_same_v<ArchTag, Arch::Ascend950>) {                                                      \
            const AscendC::DataCopyExtParams* repeatParams =                                                           \
                logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();                                          \
            ASSERT_EQ(repeatParams->blockCount, _cols_by_fractal);                                                     \
            ASSERT_EQ(repeatParams->blockLen, _row * BYTE_PER_C0);                                                     \
            ASSERT_EQ(static_cast<int64_t>(repeatParams->srcStride),                                                   \
                (static_cast<int64_t>(_row_round) - static_cast<int64_t>(_row)) * BYTE_PER_C0);                        \
            ASSERT_EQ(static_cast<int64_t>(repeatParams->dstStride),                                                   \
                static_cast<int64_t>(_dst_row_round) - static_cast<int64_t>(_row));                                    \
                                                                                                                       \
            const AscendC::DataCopyPadExtParams<Element>* padParams =                                                  \
                logTileCopy.GetArgsAt(3).Value<AscendC::DataCopyPadExtParams<Element>>();                              \
            ASSERT_FALSE(padParams->isPad);                                                                            \
        } else {                                                                                                       \
            const AscendC::DataCopyParams* repeatParams =                                                              \
                logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();                                             \
            ASSERT_EQ(repeatParams->blockCount, _cols_by_fractal);                                                     \
            ASSERT_EQ(repeatParams->blockLen, _row);                                                                   \
            ASSERT_EQ(repeatParams->srcGap, static_cast<uint16_t>(_row_round - _row));                                 \
            ASSERT_EQ(repeatParams->dstGap, static_cast<uint16_t>(_dst_row_round - _row));                             \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    /* Data-path: GM nZ → L1 nZ */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: fractal whole-tensor DataCopy (row-fractal blocks, no nd2nz) */ \
    TEST_P(SuiteName, nZTonZTestFractal)                                                                         \
    {                                                                                                                  \
        using Element      = float;                                                                                    \
        using LayoutSrcTag = layout::nZ;                                                                               \
        using LayoutDstTag = layout::nZ;                                                                               \
                                                                                                                       \
        auto logs = MakeCall<Element, LayoutSrcTag, LayoutDstTag, false>();                                            \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        AscendCCallLog logTileCopy = logs[0];                                                                          \
        constexpr bool useCopyPad = std::is_same_v<ArchTag, Arch::Ascend950>;                                          \
        BaseCheck<Element, useCopyPad>(logTileCopy);                                                                   \
                                                                                                                       \
        ASSERT_EQ(logTileCopy.GetArgsAt(0).GetInstAddr(), 0);                                                          \
        ASSERT_EQ(logTileCopy.GetArgsAt(1).GetInstAddr(), 0);                                                          \
                                                                                                                       \
        if constexpr (std::is_same_v<ArchTag, Arch::Ascend950>) {                                                      \
            const AscendC::DataCopyExtParams* repeatParams =                                                           \
                logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyExtParams>();                                          \
            ASSERT_EQ(repeatParams->blockCount, _rows_by_fractal);                                                     \
            ASSERT_EQ(repeatParams->blockLen, _col * BYTE_PER_C0);                                                     \
            ASSERT_EQ(static_cast<int64_t>(repeatParams->srcStride),                                                   \
                (static_cast<int64_t>(_col_round) - static_cast<int64_t>(_col)) * BYTE_PER_C0);                        \
            ASSERT_EQ(static_cast<int64_t>(repeatParams->dstStride),                                                   \
                static_cast<int64_t>(_dst_col_round) - static_cast<int64_t>(_col));                                    \
                                                                                                                       \
            const AscendC::DataCopyPadExtParams<Element>* padParams =                                                  \
                logTileCopy.GetArgsAt(3).Value<AscendC::DataCopyPadExtParams<Element>>();                              \
            ASSERT_FALSE(padParams->isPad);                                                                            \
        } else {                                                                                                       \
            const AscendC::DataCopyParams* repeatParams =                                                              \
                logTileCopy.GetArgsAt(2).Value<AscendC::DataCopyParams>();                                             \
            ASSERT_EQ(repeatParams->blockCount, _rows_by_fractal);                                                     \
            ASSERT_EQ(repeatParams->blockLen, _col);                                                                   \
            ASSERT_EQ(repeatParams->srcGap, _col_round - _col);                                                        \
            ASSERT_EQ(repeatParams->dstGap, static_cast<uint16_t>(_dst_col_round - _col));                             \
        }                                                                                                              \
    }

// ============================================================================
// AtlasA2-only tests: TileCopyTlaExt / TileCopySparseTla have no Ascend950 spec
// ============================================================================
#define DEFINE_GMTOL1TLA_TESTS_ATLASA2(SuiteName)                                                                       \
    /* Data-path: GM RowMajor → L1 zN */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: Ext full-shape copy (TileCopyTlaExt drives Nd2Nz from actualShape) */ \
    TEST_P(SuiteName, RowMajorTozNTestExt)                                                                \
    {                                                                                                                  \
        using Element      = float;                                                                                    \
        using LayoutSrcTag = layout::RowMajor;                                                                         \
        using LayoutDstTag = layout::zN;                                                                               \
                                                                                                                       \
        auto logs = MakeCallExt<Element, LayoutSrcTag, LayoutDstTag>();                                                \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        AscendCCallLog logTileCopy = logs[0];                                                                          \
        BaseCheck<Element>(logTileCopy);                                                                               \
                                                                                                                       \
        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();                \
        ASSERT_EQ(nd2nzArg->ndNum, _1);                                                                               \
        ASSERT_EQ(nd2nzArg->nValue, _row);                                                                            \
        ASSERT_EQ(nd2nzArg->dValue, _col);                                                                            \
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);                                                                   \
        ASSERT_EQ(nd2nzArg->srcDValue, _col);                                                                         \
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _dst_row_round);                                                           \
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);                                                                        \
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);                                                                   \
    }                                                                                                                  \
                                                                                                                       \
                                                                                                                       \
    /* Data-path: GM ColumnMajor → L1 nZ */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: Ext transposed full-shape copy (src/dst trans, n/d swapped) */ \
    TEST_P(SuiteName, ColumnMajorTonZTestExt)                                                             \
    {                                                                                                                  \
        using Element      = float;                                                                                    \
        using LayoutSrcTag = layout::ColumnMajor;                                                                      \
        using LayoutDstTag = layout::nZ;                                                                               \
                                                                                                                       \
        auto logs = MakeCallExt<Element, LayoutSrcTag, LayoutDstTag, true, true>();                                    \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        AscendCCallLog logTileCopy = logs[0];                                                                          \
        BaseCheck<Element>(logTileCopy);                                                                               \
                                                                                                                       \
        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();                \
        ASSERT_EQ(nd2nzArg->ndNum, _1);                                                                               \
        ASSERT_EQ(nd2nzArg->nValue, _col);                                                                            \
        ASSERT_EQ(nd2nzArg->dValue, _row);                                                                            \
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);                                                                   \
        ASSERT_EQ(nd2nzArg->srcDValue, _row);                                                                         \
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _dst_col_round);                                                           \
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);                                                                        \
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);                                                                   \
    }                                                                                                                  \
                                                                                                                       \
    /* Data-path: GM zN → L1 zN */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: Ext fractal passthrough (single DataCopy, name-only check) */ \
    TEST_P(SuiteName, zNTozNTestExt)                                                                      \
    {                                                                                                                  \
        using Element      = float;                                                                                    \
        using LayoutSrcTag = layout::zN;                                                                               \
        using LayoutDstTag = layout::zN;                                                                               \
                                                                                                                       \
        auto logs = MakeCallExt<Element, LayoutSrcTag, LayoutDstTag>();                                                \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        AscendCCallLog logTileCopy = logs[0];                                                                          \
        BaseCheck<Element>(logTileCopy);                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    /* Data-path: GM nZ → L1 nZ */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: Ext fractal passthrough (dst trans, name-only check) */ \
    TEST_P(SuiteName, nZTonZTestExt)                                                                      \
    {                                                                                                                  \
        using Element      = float;                                                                                    \
        using LayoutSrcTag = layout::nZ;                                                                               \
        using LayoutDstTag = layout::nZ;                                                                               \
                                                                                                                       \
        auto logs = MakeCallExt<Element, LayoutSrcTag, LayoutDstTag, false, true>();                                   \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        AscendCCallLog logTileCopy = logs[0];                                                                          \
        BaseCheck<Element>(logTileCopy);                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    /* Data-path: GM RowMajor → L1 zN */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: Sparse copy (TileCopySparseTla clamps n/d to dst-round counts) */ \
    TEST_P(SuiteName, RowMajorTozNTestSparse)                                                                   \
    {                                                                                                                  \
        using Element      = float;                                                                                    \
        using LayoutSrcTag = layout::RowMajor;                                                                         \
        using LayoutDstTag = layout::zN;                                                                               \
                                                                                                                       \
        auto logs = MakeCallSparse<Element, LayoutSrcTag, LayoutDstTag>();                                             \
        uint32_t _row_count = _row_residue < _dst_row_round ? _row_residue : _dst_row_round;                           \
        uint32_t _col_count = _col_residue < _dst_col_round ? _col_residue : _dst_col_round;                           \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        AscendCCallLog logTileCopy = logs[0];                                                                          \
        BaseCheck<Element>(logTileCopy);                                                                               \
                                                                                                                       \
        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();                \
        ASSERT_EQ(nd2nzArg->ndNum, _1);                                                                               \
        ASSERT_EQ(nd2nzArg->nValue, _row_count);                                                                      \
        ASSERT_EQ(nd2nzArg->dValue, _col_count);                                                                      \
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);                                                                   \
        ASSERT_EQ(nd2nzArg->srcDValue, _col);                                                                         \
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _dst_row_round);                                                           \
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);                                                                        \
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);                                                                   \
    }                                                                                                                  \
                                                                                                                       \
    /* Data-path: GM ColumnMajor → L1 zN */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: Sparse copy (column-major src, clamped counts, srcDValue=1) */ \
    TEST_P(SuiteName, ColumnMajorTozNTestSparse)                                                                \
    {                                                                                                                  \
        using Element      = float;                                                                                    \
        using LayoutSrcTag = layout::ColumnMajor;                                                                      \
        using LayoutDstTag = layout::zN;                                                                               \
                                                                                                                       \
        auto logs = MakeCallSparse<Element, LayoutSrcTag, LayoutDstTag>();                                             \
        uint32_t _row_count = _row_residue < _dst_row_round ? _row_residue : _dst_row_round;                           \
        uint32_t _col_count = _col_residue < _dst_col_round ? _col_residue : _dst_col_round;                           \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        AscendCCallLog logTileCopy = logs[0];                                                                          \
        BaseCheck<Element>(logTileCopy);                                                                               \
                                                                                                                       \
        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();                \
        ASSERT_EQ(nd2nzArg->ndNum, _1);                                                                               \
        ASSERT_EQ(nd2nzArg->nValue, _row_count);                                                                      \
        ASSERT_EQ(nd2nzArg->dValue, _col_count);                                                                      \
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);                                                                   \
        ASSERT_EQ(nd2nzArg->srcDValue, _1);                                                                           \
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _dst_row_round);                                                           \
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);                                                                        \
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);                                                                   \
    }                                                                                                                  \
                                                                                                                       \
    /* Data-path: GM ColumnMajor → L1 nZ */ \
    /* Element-type: no-except (float) */ \
    /* Speciality: Sparse transposed copy (n/d swapped, full-shape Nd2Nz) */ \
    TEST_P(SuiteName, ColumnMajorTonZTestSparse)                                                                \
    {                                                                                                                  \
        using Element      = float;                                                                                    \
        using LayoutSrcTag = layout::ColumnMajor;                                                                      \
        using LayoutDstTag = layout::nZ;                                                                               \
                                                                                                                       \
        auto logs = MakeCallSparse<Element, LayoutSrcTag, LayoutDstTag, true, true>();                                 \
                                                                                                                       \
        ASSERT_EQ(logs.size(), 1);                                                                                     \
        AscendCCallLog logTileCopy = logs[0];                                                                          \
        BaseCheck<Element>(logTileCopy);                                                                               \
                                                                                                                       \
        const AscendC::Nd2NzParams* nd2nzArg = logTileCopy.GetArgsAt(2).Value<AscendC::Nd2NzParams>();                \
        ASSERT_EQ(nd2nzArg->ndNum, _1);                                                                               \
        ASSERT_EQ(nd2nzArg->nValue, _col);                                                                            \
        ASSERT_EQ(nd2nzArg->dValue, _row);                                                                            \
        ASSERT_EQ(nd2nzArg->srcNdMatrixStride, _0);                                                                   \
        ASSERT_EQ(nd2nzArg->srcDValue, _row);                                                                         \
        ASSERT_EQ(nd2nzArg->dstNzC0Stride, _dst_col_round);                                                           \
        ASSERT_EQ(nd2nzArg->dstNzNStride, _1);                                                                        \
        ASSERT_EQ(nd2nzArg->dstNzMatrixStride, _0);                                                                   \
    }

    /**
     * NOTE: TileCopySparseTla zN→zN and nZ→nZ are not tested in stub mode
     * due to brace-init DataCopyParams ambiguity in the copied header code
     * (copy_gm_to_l1.hpp:2143,2183). Covered by on-device integration tests.
     */



// ============================================================================
// Instantiate for Catlass::Arch::AtlasA2
// ============================================================================
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 2201
using TypedGmToL1TlaA2Suite = TypedTileCopyGmToL1TlaTest<Catlass::Arch::AtlasA2>;

DEFINE_GMTOL1TLA_TESTS(TypedGmToL1TlaA2Suite)

DEFINE_GMTOL1TLA_TESTS_ATLASA2(TypedGmToL1TlaA2Suite)

INSTANTIATE_TEST_SUITE_P(AtlasA2, TypedGmToL1TlaA2Suite,
    ::testing::Values(
        TestMatrixShapeWithCoord{128U, 256U, 16U, 32U, 67U, 128U},
        TestMatrixShapeWithCoord{64U, 128U, 0U, 0U, 32U, 64U}));
#endif // CATLASS_ARCH == 2201

// ============================================================================
// Instantiate for Catlass::Arch::Ascend950
// ============================================================================
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
using TypedGmToL1TlaA950Suite = TypedTileCopyGmToL1TlaTest<Catlass::Arch::Ascend950>;

DEFINE_GMTOL1TLA_TESTS(TypedGmToL1TlaA950Suite)

INSTANTIATE_TEST_SUITE_P(Ascend950, TypedGmToL1TlaA950Suite,
    ::testing::Values(
        TestMatrixShapeWithCoord{128U, 256U, 16U, 32U, 67U, 128U},
        TestMatrixShapeWithCoord{64U, 128U, 0U, 0U, 32U, 64U}));
#endif // CATLASS_ARCH == 3510
