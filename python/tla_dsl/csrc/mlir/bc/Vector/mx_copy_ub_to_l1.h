/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// The MX block scale UB -> L1 copy, as a partial specialization of the CATLASS
// CopyUb2L1Tla template.
//
// It lives here rather than beside the other specializations in
// include/catlass/epilogue/tile/copy_ub_to_l1_tla.hpp because the MX scale
// block is a DSL-side concept: nothing in the CATLASS templates produces or
// consumes one, and only the DSL's own meta-ops instantiate this route. A
// partial specialization is found by ADL-independent name lookup at the point
// of instantiation, so declaring it in this header and including it before
// dma.cpp's instantiations is equivalent to having written it upstream --
// without carrying a local patch on a shared header.
//
// Because it is a *partial specialization of an existing template*, this header
// must be included after the primary template is declared. Including
// copy_ub_to_l1_tla.hpp directly below guarantees that regardless of include
// order at the use site.

#ifndef CATLASS_DSL_MX_COPY_UB_TO_L1_H
#define CATLASS_DSL_MX_COPY_UB_TO_L1_H

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/copy_ub_to_l1_tla.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Epilogue::Tile {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

/// Partial specialization for Ascend950, MX block scale in and out.
///
/// The scale block is not zN and not row-major: an e8m0 tile nests as
/// (C0_NUM_PER_FRACTAL, fractals) x (2, groups) for the A side (zZ) and the
/// transpose of that for the B side (nN), where a fractal is 32 bytes holding
/// 16 rows of a 2-wide C0. isMxScaleForzZ / isMxScaleFornN match the same
/// rank-and-depth structure as each other, so one specialization covers both
/// and tells them apart by which mode carries the 16.
///
/// Both sides must already be in scale order. That is the same contract the
/// zN -> zN specialization has, and it is what keeps this a strided block
/// move rather than a reorder: the 32-byte fractal is contiguous on both sides,
/// groups run consecutively within a fractal row, and only the pitch between
/// fractal rows differs. A producer that computes scales on the vector unit
/// writes them in this order directly, so nothing has to be permuted here.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleForzZ<ElementSrc, LayoutSrc>::value &&
        tla::detail::isMxScaleForzZ<ElementDst, LayoutDst>::value>> {
    // e8m0 is a byte, so this is the plain 32 bytes per C0; kept in the same
    // SizeOfBits form as the specializations upstream so the two cannot drift.
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    // The MX scale packs 2 elements per C0 whatever the element width; that 2
    // is what a fractal row's length rounds up to.
    static constexpr uint32_t ELE_NUM_PER_SCALE_C0 = 2;
    // zZ puts C0_NUM_PER_FRACTAL on the row mode; nN puts the plain 2 there.
    static constexpr bool IS_ZZ = (tla::shape<0, 0>(LayoutSrc{}) == C0_NUM_PER_FRACTAL);

    // Methods

    CATLASS_DEVICE
    CopyUb2L1Tla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::isMxScaleForzZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::isMxScaleForzZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::VECCALC && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be UB and an MX scale block, while TensorDst must be "
            "L1 and an MX scale block");

        // One strided move per fractal row, never one contiguous block.
        //
        // Within a fractal row the tile really is contiguous: for zZ a fixed
        // rows/16 group spans 16 * RoundUp(cols, 2) bytes with no gap, and nN
        // is the same with the roles swapped. Between fractal rows there may
        // be a gap, because the pitch on that mode comes from the descriptor
        // and a tile view into a wider scale buffer carries the parent's
        // pitch, not its own extent. Assuming dense packing wrote fractal row
        // r at r * innerBytes where the reader expects it at r * pitch, which
        // leaves only the first fractal row in place -- everything from row 16
        // on lands somewhere the L1 -> L0 load does not look.
        //
        // So take the pitch from the stride, exactly as the GM -> L1 scale
        // copies do (dstNzC0Stride = dstOuterStrideRow / BYTE_PER_C0) and as
        // the zN -> zN specialization does. When both sides are dense this
        // degenerates to the single contiguous block it replaces.
        //
        // Extents come from originShape, never from the nested shape leaves: a
        // tile derived through a parent comes back with its outer row-tiling
        // extent collapsed to 1, which for zZ is the mode carrying rows/16.
        const uint32_t rows = static_cast<uint32_t>(tla::get<0>(srcTensor.originShape()));
        const uint32_t cols = static_cast<uint32_t>(tla::get<1>(srcTensor.originShape()));

        uint32_t blockCount;
        uint32_t innerC0;
        uint32_t srcPitch;
        uint32_t dstPitch;
        if constexpr (IS_ZZ) {
            blockCount = CeilDiv<C0_NUM_PER_FRACTAL>(rows);
            innerC0 = RoundUp<ELE_NUM_PER_SCALE_C0>(cols) * C0_NUM_PER_FRACTAL / ELE_NUM_PER_C0;
            srcPitch = static_cast<uint32_t>(tla::get<0, 1>(srcTensor.stride())) / ELE_NUM_PER_C0;
            dstPitch = static_cast<uint32_t>(tla::get<0, 1>(dstTensor.stride())) / ELE_NUM_PER_C0;
        } else {
            blockCount = CeilDiv<C0_NUM_PER_FRACTAL>(cols);
            innerC0 = RoundUp<ELE_NUM_PER_SCALE_C0>(rows) * C0_NUM_PER_FRACTAL / ELE_NUM_PER_C0;
            srcPitch = static_cast<uint32_t>(tla::get<1, 1>(srcTensor.stride())) / ELE_NUM_PER_C0;
            dstPitch = static_cast<uint32_t>(tla::get<1, 1>(dstTensor.stride())) / ELE_NUM_PER_C0;
        }

        AscendC::DataCopyParams repeatParams;
        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = innerC0;
        repeatParams.srcStride = srcPitch - innerC0;
        repeatParams.dstStride = dstPitch - innerC0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }
};

#endif

} // namespace Catlass::Epilogue::Tile

#endif // CATLASS_DSL_MX_COPY_UB_TO_L1_H
