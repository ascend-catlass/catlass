/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_COPY_UB_TO_UB_TLA_HPP
#define CATLASS_EPILOGUE_TILE_COPY_UB_TO_UB_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Epilogue::Tile {

template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct CopyUb2UbTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported CopyUb2UbTla, can not find the specialization.");
};

/// Partial specialization for Ascend950, RowMajor in and RowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2UbTla<
    Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<ElementSrc>::value;

    // Methods
    CATLASS_DEVICE
    CopyUb2UbTla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::VECCALC &&
                TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc and TensorDst must both be UB and RowMajor");

        const uint32_t rows = tla::get<0>(srcTensor.originShape());
        const uint32_t cols = CeilDiv<ELE_NUM_PER_BLK>(tla::get<1>(srcTensor.originShape()));
        const uint32_t srcGap = CeilDiv<ELE_NUM_PER_BLK>(tla::get<0>(srcTensor.stride())) - cols;
        const uint32_t dstGap = CeilDiv<ELE_NUM_PER_BLK>(tla::get<0>(dstTensor.stride())) - cols;

        AscendC::DataCopyParams params;
        if (srcGap == 0 and dstGap == 0) {
            params.blockCount = 1;
            params.blockLen = static_cast<uint16_t>(rows * cols);
            params.srcGap = 0;
            params.dstGap = 0;
        } else {
            params.blockCount = static_cast<uint16_t>(rows);
            params.blockLen = static_cast<uint16_t>(cols);
            params.srcGap = static_cast<uint16_t>(srcGap);
            params.dstGap = static_cast<uint16_t>(dstGap);
        }

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        if constexpr (AscendC::Std::is_one_of_v<ElementSrc, float4_e2m1x2_t, float4_e1m2x2_t>) {
            AscendC::DataCopy(
                dstTensor.data()[dstOffset].template ReinterpretCast<uint8_t>(),
                srcTensor.data()[srcOffset].template ReinterpretCast<uint8_t>(), params);
        } else {
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], params);
        }
    }
};
#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2UbTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>>
    : CopyUb2UbTla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
          std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2UbTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>>
    : CopyUb2UbTla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
          std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>> {};
#endif

/// Partial specialization for Ascend950, RowMajor in and ColumnMajor out: the copy transposes, dst[j][i] = src[i][j].
/// Physically the destination is an (N, M) array whose rows are stride<1> elements apart.
///
/// The engine is VNCHWCONV (AscendC::TransDataTo5HD): 16 source and 16 destination addresses in VA registers define
/// the block, whose shape follows the 32-byte DataBlock -- 16x16 for b16, 16x8 for b32, 16x16 for b8. A b8 block is
/// half a 32x32 tile, so four calls (one per quadrant) cover it.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2UbTla<
    Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isColumnMajor<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<ElementSrc>::value;

    static_assert(
        sizeof(ElementSrc) == 2 || sizeof(ElementSrc) == 4 || SizeOfBits<ElementSrc>::value == 8,
        "The transpose only supports 2-byte, 4-byte and 1-byte elements");
    // The destination pointer is typed by ElementSrc, so the two widths have to agree.
    static_assert(sizeof(ElementSrc) == sizeof(ElementDst), "TensorSrc and TensorDst must have the same element width");

    // Methods
    CATLASS_DEVICE
    CopyUb2UbTla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::isColumnMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::VECCALC &&
                TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be UB and RowMajor, TensorDst must be UB and "
            "ColumnMajor");

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        __ubuf__ ElementSrc* srcPtr = (__ubuf__ ElementSrc*)srcTensor.data()[srcOffset].GetPhyAddr();
        __ubuf__ ElementSrc* dstPtr = (__ubuf__ ElementSrc*)dstTensor.data()[dstOffset].GetPhyAddr();

        // Every tile is moved whole, so the last one reaches past the logical shape: srcStride must be at least
        // RoundUp<ELE_NUM_PER_BLK>(cols) and dstStride at least RoundUp<TILE_ROWS>(rows), with both tensors allocated
        // for that rounded shape.
        const uint32_t rows = tla::get<0>(srcTensor.originShape());
        const uint32_t cols = tla::get<1>(srcTensor.originShape());
        const uint32_t srcStride = tla::get<0>(srcTensor.stride());
        const uint32_t dstStride = tla::get<1>(dstTensor.stride());

        constexpr uint32_t ADDR_LIST_SIZE = 16; // NCHW_CONV_ADDR_LIST_SIZE
        // b8 covers a tile in four calls spanning 32 source rows; every other width covers it from one 16-row list.
        constexpr uint32_t TILE_ROWS = (sizeof(ElementSrc) == 1) ? 2 * ADDR_LIST_SIZE : ADDR_LIST_SIZE;
        // One address list walks up to MAX_REPEAT column blocks before it has to be reloaded (repeatTimes is a
        // uint8_t). The next block starts one 32-byte DataBlock further in src at every width, and dstStride DataBlocks
        // further in dst: ELE_NUM_PER_BLK transposed rows of dstStride elements each, rounded up to DataBlocks. So the
        // 16 MOVEVA that fill the list are paid once per row block instead of once per block.
        constexpr uint32_t MAX_REPEAT = 255;
        uint64_t srcList[ADDR_LIST_SIZE];
        uint64_t dstList[ADDR_LIST_SIZE];

        for (uint32_t i = 0; i < rows; i += TILE_ROWS) {
            for (uint32_t j = 0; j < cols; j += ELE_NUM_PER_BLK * MAX_REPEAT) {
                const uint32_t blocks = CeilDiv<ELE_NUM_PER_BLK>(cols - j);
                const uint32_t repeats = blocks < MAX_REPEAT ? blocks : MAX_REPEAT;

                AscendC::TransDataTo5HDParams params;
                params.repeatTimes = static_cast<uint8_t>(repeats);
                // A single repeat has no next block to step to.
                params.srcRepStride = repeats > 1 ? 1 : 0;
                params.dstRepStride = repeats > 1 ? static_cast<uint16_t>(dstStride) : 0;

                if constexpr (sizeof(ElementSrc) == 2) {
                    // b16: a square 16x16 block, one destination address per transposed row.
                    for (uint32_t idx = 0; idx < ADDR_LIST_SIZE; ++idx) {
                        srcList[idx] =
                            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcPtr + (i + idx) * srcStride + j));
                        dstList[idx] =
                            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstPtr + (j + idx) * dstStride + i));
                    }
                    AscendC::TransDataTo5HD<ElementSrc>(dstList, srcList, params);
                } else if constexpr (sizeof(ElementSrc) == 4) {
                    // b32: 16x8 in, 8 rows of 16 out -- two destination blocks per row, so the list is filled in
                    // pairs.
                    for (uint32_t idx = 0; idx < ADDR_LIST_SIZE; ++idx) {
                        srcList[idx] =
                            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcPtr + (i + idx) * srcStride + j));
                    }
                    for (uint32_t idx = 0; idx < ELE_NUM_PER_BLK; ++idx) {
                        __ubuf__ ElementSrc* row = dstPtr + (j + idx) * dstStride + i;
                        dstList[2 * idx] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(row));
                        dstList[2 * idx + 1] =
                            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(row + ELE_NUM_PER_BLK));
                    }
                    AscendC::TransDataTo5HD<ElementSrc>(dstList, srcList, params);
                } else if constexpr (Catlass::SizeOfBits<ElementSrc>::value == 8) { // b8 types, exclude fp4
                    // b8: 32 elements per DataBlock but only 16 list entries, so one call moves a 16x16 quarter of
                    // the 32x32 tile -- srcHighHalf picks the column half read, dstHighHalf the destination half
                    // written.
                    for (uint32_t q = 0; q < 4; ++q) {
                        const bool srcHigh = (q & 1) != 0; // columns j+16..j+31
                        const bool dstHigh = (q & 2) != 0; // source rows i+16..i+31
                        const uint32_t rowBase = i + (dstHigh ? ADDR_LIST_SIZE : 0);
                        const uint32_t colBase = j + (srcHigh ? ADDR_LIST_SIZE : 0);
                        for (uint32_t idx = 0; idx < ADDR_LIST_SIZE; ++idx) {
                            srcList[idx] = static_cast<uint64_t>(
                                reinterpret_cast<uintptr_t>(srcPtr + (rowBase + idx) * srcStride + j));
                            dstList[idx] = static_cast<uint64_t>(
                                reinterpret_cast<uintptr_t>(dstPtr + (colBase + idx) * dstStride + i));
                        }
                        params.srcHighHalf = srcHigh;
                        params.dstHighHalf = dstHigh;
                        AscendC::TransDataTo5HD<uint8_t>(dstList, srcList, params);
                    }
                }
            }
        }
    }
};
#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2UbTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isColumnMajor<LayoutDst>::value>>
    : CopyUb2UbTla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
          std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isColumnMajor<LayoutDst>::value>> {
};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2UbTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isColumnMajor<LayoutDst>::value>>
    : CopyUb2UbTla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
          std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isColumnMajor<LayoutDst>::value>> {
};
#endif

} // namespace Catlass::Epilogue::Tile

#endif // CATLASS_EPILOGUE_TILE_COPY_UB_TO_UB_TLA_HPP
