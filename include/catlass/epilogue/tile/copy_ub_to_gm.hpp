/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it
 * and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the
 * "License").
 * Please refer to the License for details. You may not use this file except in compliance with the
 * License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_COPY_UB_TO_GM_HPP
#define CATLASS_EPILOGUE_TILE_TILE_COPY_UB_TO_GM_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/conv_coord.hpp"

namespace Catlass::Epilogue::Tile {

template <class ArchTag, class GmType>
struct CopyUb2Gm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy ub to gm, can not find the specialization.");
};

template <typename Element>
struct CopyUb2Gm<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyUb2Gm() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        layout::RowMajor const& layoutDst, layout::RowMajor const& layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            layoutDst.shape(0), layoutDst.shape(1) * sizeof(Element),
            (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_C0,
            (layoutDst.stride(0) - layoutDst.shape(1)) * sizeof(Element), 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
    }
};

// new add vectorlayout version
template <typename Element>
struct CopyUb2Gm<Arch::AtlasA2, Gemm::GemmType<Element, layout::VectorLayout>> {
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    CATLASS_DEVICE
    CopyUb2Gm() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        layout::VectorLayout const& layoutDst, layout::VectorLayout const& layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(1, layoutDst.shape(0) * sizeof(Element), 0, 0, 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
    };
};

template <class ArchTag, class GmType>
struct CopyUb2GmAligned {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy ub to gm aligned, can not find the specialization.");
};

template <typename Element>
struct CopyUb2GmAligned<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);
    static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
    static constexpr uint32_t MAX_REPEAT = 4095;
    static constexpr uint32_t STRIDE_LIMIT = 65536;

    CATLASS_DEVICE
    CopyUb2GmAligned() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        layout::RowMajor const& layoutDst, layout::RowMajor const& layoutSrc)
    {
        uint32_t rows = layoutDst.shape(0);
        uint32_t cols = layoutDst.shape(1);
        uint32_t srcStride = (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_BLK;
        uint32_t dstStride = (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;

        if ((layoutSrc.shape(1) == layoutSrc.stride(0)) && (layoutDst.shape(1) == layoutDst.stride(0))) {
            DataCopy(dstTensor, srcTensor, rows * cols);
        } else if (srcStride < STRIDE_LIMIT && dstStride < STRIDE_LIMIT && (cols / ELE_NUM_PER_BLK) < BLOCK_LEN_LIMIT) {
            uint32_t rLoops = CeilDiv(rows, MAX_REPEAT);
            for (uint32_t i = 0; i < rLoops; ++i) {
                uint32_t rActual = (i < rLoops - 1) ? MAX_REPEAT : rows - i * MAX_REPEAT;
                AscendC::DataCopyParams dataCopyParams(rActual, cols / ELE_NUM_PER_BLK, srcStride, dstStride);
                DataCopy(
                    dstTensor[i * MAX_REPEAT * layoutDst.stride(0)], srcTensor[i * MAX_REPEAT * layoutSrc.stride(0)],
                    dataCopyParams);
            }
        } else {
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopy(dstTensor[i * layoutDst.stride(0)], srcTensor[i * layoutSrc.stride(0)], cols);
            }
        }
    };
};

//////////////////////////// CopyUb2Gm(Ascend950, No TLA) ////////////////////////////

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
// Partial specialization for CopyUb2Gm(Ascend950, no-tla), RowMajor in and RowMajor out.
template <typename Element>
struct CopyUb2Gm<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyUb2Gm() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        layout::RowMajor const& layoutDst, layout::RowMajor const& layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            layoutDst.shape(0), layoutDst.shape(1) * sizeof(Element),
            (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_C0,
            (layoutDst.stride(0) - layoutDst.shape(1)) * sizeof(Element), 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
    }
};

template <typename Element>
struct CopyUb2Gm<Arch::Ascend950, Gemm::GemmType<Element, layout::VectorLayout>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    CATLASS_DEVICE
    CopyUb2Gm() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(1, layoutDst.shape(0) * sizeof(Element), 0, 0, 0);
        if constexpr (AscendC::Std::is_one_of_v<
                          Element, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>) {
            AscendC::DataCopyPad(
                dstTensor.template ReinterpretCast<int8_t>(), srcTensor.template ReinterpretCast<int8_t>(),
                dataCopyParams);
        } else {
            AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
        }
    };
};

#endif // CATLASS_ARCH == 3510 || __NPU_ARCH__ == 3510

////////////////////////////////////////////////////////////
// NC1HWC0 specialized copies for Conv2d epilogue
////////////////////////////////////////////////////////////

template <typename Element>
struct CopyUb2Gm<Arch::AtlasA2, Gemm::GemmType<Element, layout::NC1HWC0>> {
    using LayoutDst = layout::NC1HWC0;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t C0_SIZE = BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyUb2Gm() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        layout::NC1HWC0 const& layoutDst, layout::RowMajor const& layoutSrc, int64_t tileBaseOffset = 0,
        uint32_t localStart = 0)
    {
        uint32_t batch = layoutDst.shape(0);
        uint32_t c1 = layoutDst.shape(1);
        uint32_t h = layoutDst.shape(2);
        uint32_t w = layoutDst.shape(3);
        uint32_t c0 = layoutDst.shape(4);

        uint32_t totalElements = batch * c1 * h * w * c0;
        uint32_t srcRows = layoutSrc.shape(0);
        uint32_t srcCols = layoutSrc.shape(1);
        uint32_t actualLen = srcRows * srcCols;

        if (actualLen >= totalElements) {
            actualLen = totalElements;
        }

        int64_t strideB = layoutDst.stride(0);
        int64_t strideC1 = layoutDst.stride(1);
        int64_t strideH = layoutDst.stride(2);
        int64_t strideW = layoutDst.stride(3);
        int64_t strideC0 = layoutDst.stride(4);

        int64_t localFlat = localStart;
        uint32_t remainingElements = actualLen;
        uint32_t srcIdx = 0;

        while (remainingElements > 0) {
            uint32_t chunkSize = (remainingElements > C0_SIZE) ? C0_SIZE : remainingElements;

            uint32_t localC0 = localFlat % c0;
            uint32_t remIdx = localFlat / c0;
            uint32_t localW = remIdx % w;
            remIdx /= w;
            uint32_t localH = remIdx % h;
            remIdx /= h;
            uint32_t localC1 = remIdx % c1;
            uint32_t localB = remIdx / c1;

            int64_t gmOffset = tileBaseOffset + localB * strideB + localC1 * strideC1 + localH * strideH +
                               localW * strideW + localC0 * strideC0;

            AscendC::DataCopyExtParams copyParams(1, chunkSize * sizeof(Element), 0, 0, 0);
            AscendC::DataCopyPad(dstTensor[gmOffset], srcTensor[srcIdx], copyParams);

            localFlat += chunkSize;
            srcIdx += chunkSize;
            remainingElements -= chunkSize;
        }
    };
};

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
template <typename Element>
struct CopyUb2Gm<Arch::Ascend950, Gemm::GemmType<Element, layout::NC1HWC0>> {
    using LayoutDst = layout::NC1HWC0;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t C0_SIZE = BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyUb2Gm() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const& dstTensor, AscendC::LocalTensor<Element> const& srcTensor,
        layout::NC1HWC0 const& layoutDst, layout::RowMajor const& layoutSrc, int64_t tileBaseOffset = 0,
        uint32_t localStart = 0)
    {
        uint32_t batch = layoutDst.shape(0);
        uint32_t c1 = layoutDst.shape(1);
        uint32_t h = layoutDst.shape(2);
        uint32_t w = layoutDst.shape(3);
        uint32_t c0 = layoutDst.shape(4);

        uint32_t totalElements = batch * c1 * h * w * c0;
        uint32_t srcRows = layoutSrc.shape(0);
        uint32_t srcCols = layoutSrc.shape(1);
        uint32_t actualLen = srcRows * srcCols;

        if (actualLen >= totalElements) {
            actualLen = totalElements;
        }

        // tileBaseOffset is the absolute GM offset for the tile's local index 0
        // localStart is the starting local index within the tile
        int64_t strideB = layoutDst.stride(0);
        int64_t strideC1 = layoutDst.stride(1);
        int64_t strideH = layoutDst.stride(2);
        int64_t strideW = layoutDst.stride(3);
        int64_t strideC0 = layoutDst.stride(4);

        int64_t localFlat = localStart;
        uint32_t remainingElements = actualLen;
        uint32_t srcIdx = 0;

        while (remainingElements > 0) {
            uint32_t chunkSize = (remainingElements > C0_SIZE) ? C0_SIZE : remainingElements;

            // Convert localFlat to tile-local NC1HWC0 coordinate
            uint32_t localC0 = localFlat % c0;
            uint32_t remIdx = localFlat / c0;
            uint32_t localW = remIdx % w;
            remIdx /= w;
            uint32_t localH = remIdx % h;
            remIdx /= h;
            uint32_t localC1 = remIdx % c1;
            uint32_t localB = remIdx / c1;

            // Compute absolute GM offset: tileBaseOffset + local coord * global strides
            int64_t gmOffset = tileBaseOffset + localB * strideB + localC1 * strideC1 + localH * strideH +
                               localW * strideW + localC0 * strideC0;

            AscendC::DataCopyExtParams copyParams(1, chunkSize * sizeof(Element), 0, 0, 0);
            AscendC::DataCopyPad(dstTensor[gmOffset], srcTensor[srcIdx], copyParams);

            localFlat += chunkSize;
            srcIdx += chunkSize;
            remainingElements -= chunkSize;
        }
    };
};
#endif
} // namespace Catlass::Epilogue::Tile

#endif
