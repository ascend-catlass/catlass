/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_COPY_UB_TO_L1_TLA_HPP
#define CATLASS_EPILOGUE_TILE_COPY_UB_TO_L1_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Epilogue::Tile {

template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct CopyUb2L1Tla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported CopyUb2L1Tla, can not find the specialization.");
};

/// Partial specialization for CopyUbToL1, Ascend950, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyUb2L1Tla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::VECCALC && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be VECCALC and RowMajor, while TensorDst must be L1 and "
            "zN");

        const uint32_t nValue = tla::get<0>(srcTensor.originShape());
        const uint32_t dValue = tla::get<1>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstNzNStride = tla::get<0, 0>(dstTensor.stride()) / ELE_NUM_PER_C0;
        const uint32_t dstNzC0Stride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        uint32_t rows = nValue;
        uint32_t cols = RoundUp<BYTE_PER_BLK>(ElementsToBytes<ElementSrc>(dValue)) / BYTE_PER_BLK;
        if (cols > rows) {
            for (uint32_t i = 0; i < rows; i++) {
                uint32_t dst = i * dstNzNStride * BytesToElements<ElementDst>(BYTE_PER_BLK);
                uint32_t src = i * srcDValue;
                AscendC::DataCopyParams dataCopyParams(cols, 1, 0, dstNzC0Stride - 1);
                if constexpr (AscendC::Std::is_one_of_v<ElementSrc, float4_e2m1x2_t, float4_e1m2x2_t>) {
                    // fp4: data()[offset] converts the packed index to bytes, so copy as uint8_t.
                    AscendC::DataCopy(
                        dstTensor.data()[dstOffset + dst].template ReinterpretCast<uint8_t>(),
                        srcTensor.data()[srcOffset + src].template ReinterpretCast<uint8_t>(), dataCopyParams);
                } else {
                    AscendC::DataCopy(
                        dstTensor.data()[dstOffset + dst], srcTensor.data()[srcOffset + src], dataCopyParams);
                }
            }
        } else {
            for (uint32_t i = 0; i < cols; i++) {
                uint32_t dst = i * dstNzC0Stride * BytesToElements<ElementDst>(BYTE_PER_BLK);
                uint32_t src = i * BytesToElements<ElementSrc>(BYTE_PER_BLK);
                AscendC::DataCopyParams dataCopyParams(
                    rows, 1, ElementsToBytes<ElementSrc>(srcDValue) / BYTE_PER_BLK - 1, dstNzNStride - 1);
                if constexpr (AscendC::Std::is_one_of_v<ElementSrc, float4_e2m1x2_t, float4_e1m2x2_t>) {
                    AscendC::DataCopy(
                        dstTensor.data()[dstOffset + dst].template ReinterpretCast<uint8_t>(),
                        srcTensor.data()[srcOffset + src].template ReinterpretCast<uint8_t>(), dataCopyParams);
                } else {
                    AscendC::DataCopy(
                        dstTensor.data()[dstOffset + dst], srcTensor.data()[srcOffset + src], dataCopyParams);
                }
            }
        }
    }
};
#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend960DT, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>>
    : CopyUb2L1Tla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend960PR, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>>
    : CopyUb2L1Tla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {};
#endif

/// Partial specialization for Ascend950, zN in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::iszNUnAlign<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    CopyUb2L1Tla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::iszNUnAlign<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::VECCALC && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be UB and zN, "
            "while TensorDst must be L1 and zN");

        int64_t srcShape = tla::get<0, 0>(srcTensor.originShape());
        AscendC::DataCopyParams dataCopyParams(
            tla::get<1, 1>(srcTensor.shape()), srcShape,
            (tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - srcShape),
            (tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - srcShape));

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        if constexpr (AscendC::Std::is_one_of_v<ElementSrc, float4_e2m1x2_t, float4_e1m2x2_t>) {
            // fp4: data()[offset] converts the packed index to bytes, so copy as uint8_t.
            AscendC::DataCopy(
                dstTensor.data()[dstOffset].template ReinterpretCast<uint8_t>(),
                srcTensor.data()[srcOffset].template ReinterpretCast<uint8_t>(), dataCopyParams);
        } else {
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams);
        }
    }
};
#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend960DT, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::iszNUnAlign<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>>
    : CopyUb2L1Tla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::iszNUnAlign<ElementSrc, LayoutSrc>::value &&
              tla::detail::iszN<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend960PR, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::iszNUnAlign<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>>
    : CopyUb2L1Tla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::iszNUnAlign<ElementSrc, LayoutSrc>::value &&
              tla::detail::iszN<ElementDst, LayoutDst>::value>> {};
#endif

#if (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202))
/// Partial specialization for Ascend950, zN in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    CopyUb2L1Tla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::VECCALC && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be UB and zN, "
            "while TensorDst must be L1 and zN");

        const uint32_t blockCount = tla::get<1, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<0>(dstTensor.originShape());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - blockLen;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - blockLen;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        if constexpr (AscendC::Std::is_one_of_v<ElementSrc, float4_e2m1x2_t, float4_e1m2x2_t>) {
            // fp4: data()[offset] converts the packed index to bytes, so copy as uint8_t.
            AscendC::DataCopy(
                dstTensor.data()[dstOffset].template ReinterpretCast<uint8_t>(),
                srcTensor.data()[srcOffset].template ReinterpretCast<uint8_t>(), repeatParams);
        } else {
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
        }
    }
};

/// Partial specialization for Ascend950, MX block scale zZ in and zZ out.
///
/// Both sides are already zZ ordered and the 32-byte fractal is contiguous, so
/// only the pitch between fractal rows differs: a strided block move.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleForzZ<ElementSrc, LayoutSrc>::value &&
        tla::detail::isMxScaleForzZ<ElementDst, LayoutDst>::value>> {
    // e8m0 is one byte: 32 bytes per C0, same SizeOfBits form as above.
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

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

        // Strided move per fractal row: the pitch comes from the descriptor, so a
        // tile view into a wider buffer carries the parent's pitch, never dense
        // packing -- the L1 -> L0 load would read only the first fractal row.
        const uint32_t rows = tla::get<0>(srcTensor.originShape());
        const uint32_t cols = tla::get<1>(srcTensor.originShape());

        // A fractal row is 16 * RoundUp(cols, 2) bytes, i.e. blockLen in 32-byte blocks.
        const uint32_t blockCount = CeilDiv<C0_NUM_PER_FRACTAL>(rows);
        const uint32_t blockLen = RoundUp<2>(cols) * C0_NUM_PER_FRACTAL / ELE_NUM_PER_C0;
        const uint32_t srcPitch = tla::get<0, 1>(srcTensor.stride()) / ELE_NUM_PER_C0;
        const uint32_t dstPitch = tla::get<0, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;

        AscendC::DataCopyParams repeatParams(blockCount, blockLen, srcPitch - blockLen, dstPitch - blockLen);

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }
};

/// Partial specialization for Ascend950, MX block scale nN in and nN out.
///
/// The B-side transpose of the zZ block, as the L1 -> L0B load requires: C0 runs
/// down the first mode and the fractals across the second.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleFornN<ElementSrc, LayoutSrc>::value &&
        tla::detail::isMxScaleFornN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    CopyUb2L1Tla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::isMxScaleFornN<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::isMxScaleFornN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::VECCALC && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be UB and an MX scale block, while TensorDst must be "
            "L1 and an MX scale block");

        const uint32_t rows = tla::get<0>(srcTensor.originShape());
        const uint32_t cols = tla::get<1>(srcTensor.originShape());

        // Transposed roles: blockCount steps the fractals down the column mode.
        const uint32_t blockCount = CeilDiv<C0_NUM_PER_FRACTAL>(cols);
        const uint32_t blockLen = RoundUp<2>(rows) * C0_NUM_PER_FRACTAL / ELE_NUM_PER_C0;
        const uint32_t srcPitch = tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0;
        const uint32_t dstPitch = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;

        AscendC::DataCopyParams repeatParams(blockCount, blockLen, srcPitch - blockLen, dstPitch - blockLen);

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }
};
#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend960DT, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>>
    : CopyUb2L1Tla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend960PR, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>>
    : CopyUb2L1Tla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend960DT, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleForzZ<ElementSrc, LayoutSrc>::value &&
        tla::detail::isMxScaleForzZ<ElementDst, LayoutDst>::value>>
    : CopyUb2L1Tla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isMxScaleForzZ<ElementSrc, LayoutSrc>::value &&
              tla::detail::isMxScaleForzZ<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend960PR, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleForzZ<ElementSrc, LayoutSrc>::value &&
        tla::detail::isMxScaleForzZ<ElementDst, LayoutDst>::value>>
    : CopyUb2L1Tla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isMxScaleForzZ<ElementSrc, LayoutSrc>::value &&
              tla::detail::isMxScaleForzZ<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend960DT, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleFornN<ElementSrc, LayoutSrc>::value &&
        tla::detail::isMxScaleFornN<ElementDst, LayoutDst>::value>>
    : CopyUb2L1Tla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isMxScaleFornN<ElementSrc, LayoutSrc>::value &&
              tla::detail::isMxScaleFornN<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2L1Tla<
    Arch::Ascend960PR, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleFornN<ElementSrc, LayoutSrc>::value &&
        tla::detail::isMxScaleFornN<ElementDst, LayoutDst>::value>>
    : CopyUb2L1Tla<
          Arch::Ascend950,
          tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isMxScaleFornN<ElementSrc, LayoutSrc>::value &&
              tla::detail::isMxScaleFornN<ElementDst, LayoutDst>::value>> {};
#endif

#endif

} // namespace Catlass::Epilogue::Tile

#endif // CATLASS_EPILOGUE_TILE_COPY_UB_TO_L1_TLA_HPP
