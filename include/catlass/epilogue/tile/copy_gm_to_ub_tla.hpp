/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_COPY_GM_TO_UB_TLA_HPP
#define CATLASS_EPILOGUE_TILE_COPY_GM_TO_UB_TLA_HPP

#include <type_traits>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Epilogue::Tile {

/// BC pad bag for hif8/fp8/fp4: distinct from ``DataCopyPadExtParams<ElementSrc>``
/// so ``ElementSrc == uint8_t`` does not collide with a typed ExtParams overload.
struct GmUbLowpPadParams {
    AscendC::DataCopyPadExtParams<uint8_t> pad;
};

/// CopyGm2UbTla optional third argument is AscendC ``DataCopyPadExtParams<T>``.
///
/// Default two-arg path uses ``isPad=false, left=right=0, value=0`` (exact bytes).
/// Caller responsibility for pad constraints:
/// - ``leftPadding`` / ``rightPadding``: element counts; each side ≤ 32B
///   (``padElems * sizeof(T) ≤ 32``).
/// - When ``sizeof(T)==8`` (b64), ``paddingValue`` must be 0.
/// - ``blockLen`` must be a multiple of ``sizeof(T)``; GM src 1B-aligned,
///   UB dst 32B-aligned.
/// - Ascend950 RowMajor/VectorLayout mx fp8/fp4/hif8 reinterprets as ``uint8_t``.
///   Pad ExtParams flags and the low 8 bits of ``paddingValue`` are forwarded.

template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct CopyGm2UbTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported CopyGm2UbTla, can not find the specialization.");
};

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyGm2UbTla<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementSrc);

    // Methods

    CATLASS_DEVICE
    CopyGm2UbTla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            std::is_same_v<ElementSrc, ElementDst>,
            "CopyGm2UbTla: ElementSrc and ElementDst must match AscendC DataCopyPad T");
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, "
            "while TensorDst must be UB and RowMajor");

        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.originShape()), tla::get<1>(srcTensor.originShape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.originShape())) * sizeof(ElementSrc),
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(srcTensor.originShape())) / ELE_NUM_PER_BLK, 0);
        AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
    }

    /// Forward AscendC ``DataCopyPadExtParams`` as-is.
    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const& dstTensor, TensorSrc const& srcTensor,
        AscendC::DataCopyPadExtParams<ElementSrc> const& padParams)
    {
        static_assert(
            std::is_same_v<ElementSrc, ElementDst>,
            "CopyGm2UbTla: ElementSrc and ElementDst must match AscendC DataCopyPad T");
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, "
            "while TensorDst must be UB and RowMajor");

        // AscendC Normal-mode dstStride is the gap after blockLen+left+right pads
        // (each measured in 32B blocks). Subtract pad widths so multi-row writes
        // land on dstTensor.stride<0>(), not stride + AlignUp(pad).
        const int64_t dstGapElems = tla::get<0>(dstTensor.stride()) - tla::get<1>(srcTensor.originShape()) -
                                    static_cast<int64_t>(padParams.leftPadding) -
                                    static_cast<int64_t>(padParams.rightPadding);
        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.originShape()), tla::get<1>(srcTensor.originShape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.originShape())) * sizeof(ElementSrc),
            dstGapElems / ELE_NUM_PER_BLK, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
    }
};

#if (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202)) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202))
/// Partial specialization for Ascend950, Vector in and Vector out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyGm2UbTla<
    Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    CopyGm2UbTla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            std::is_same_v<ElementSrc, ElementDst>,
            "CopyGm2UbTla: ElementSrc and ElementDst must match AscendC DataCopyPad T");
        static_assert(
            tla::detail::isVector<typename TensorSrc::Layout>::value &&
                tla::detail::isVector<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be GM and Vector, "
            "while TensorDst must be UB and Vector");

        AscendC::DataCopyExtParams dataCopyParams(
            1, ElementsToBytes<ElementSrc>(tla::get<0>(srcTensor.originShape())), 0, 0, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        if constexpr (AscendC::Std::is_one_of_v<
                          ElementSrc, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, hifloat8_t, float4_e2m1x2_t,
                          float4_e1m2x2_t>) {
            AscendC::DataCopyPadExtParams<uint8_t> padParams(false, 0, 0, 0);
            AscendC::DataCopyPad(
                dstTensor.data()[dstOffset].template ReinterpretCast<uint8_t>(),
                srcTensor.data()[srcOffset].template ReinterpretCast<uint8_t>(), dataCopyParams, padParams);
        } else {
            AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);
            AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
        }
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const& dstTensor, TensorSrc const& srcTensor,
        AscendC::DataCopyPadExtParams<ElementSrc> const& padParams)
    {
        static_assert(
            std::is_same_v<ElementSrc, ElementDst>,
            "CopyGm2UbTla: ElementSrc and ElementDst must match AscendC DataCopyPad T");
        static_assert(
            tla::detail::isVector<typename TensorSrc::Layout>::value &&
                tla::detail::isVector<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be GM and Vector, "
            "while TensorDst must be UB and Vector");

        AscendC::DataCopyExtParams dataCopyParams(
            1, ElementsToBytes<ElementSrc>(tla::get<0>(srcTensor.originShape())), 0, 0, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        if constexpr (AscendC::Std::is_one_of_v<
                          ElementSrc, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, hifloat8_t, float4_e2m1x2_t,
                          float4_e1m2x2_t>) {
            uint8_t fillBits = *reinterpret_cast<const uint8_t*>(&padParams.paddingValue);
            AscendC::DataCopyPadExtParams<uint8_t> padParamsU8(
                padParams.isPad, padParams.leftPadding, padParams.rightPadding, fillBits);
            AscendC::DataCopyPad(
                dstTensor.data()[dstOffset].template ReinterpretCast<uint8_t>(),
                srcTensor.data()[srcOffset].template ReinterpretCast<uint8_t>(), dataCopyParams, padParamsU8);
        } else {
            AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
        }
    }

    // BC pad path for hif8/fp8/fp4 (ccec-safe uint8 ExtParams bag).
    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const& dstTensor, TensorSrc const& srcTensor, GmUbLowpPadParams const& lowpPad)
    {
        static_assert(
            AscendC::Std::is_one_of_v<
                ElementSrc, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, hifloat8_t, float4_e2m1x2_t, float4_e1m2x2_t>,
            "GmUbLowpPadParams is only for low-precision GM→UB VectorLayout");
        static_assert(
            std::is_same_v<ElementSrc, ElementDst>,
            "CopyGm2UbTla: ElementSrc and ElementDst must match AscendC DataCopyPad T");
        AscendC::DataCopyExtParams dataCopyParams(
            1, ElementsToBytes<ElementSrc>(tla::get<0>(srcTensor.originShape())), 0, 0, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(
            dstTensor.data()[dstOffset].template ReinterpretCast<uint8_t>(),
            srcTensor.data()[srcOffset].template ReinterpretCast<uint8_t>(), dataCopyParams, lowpPad.pad);
    }
};

/// Partial specialization for Ascend950, RowMajor in and RowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyGm2UbTla<
    Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    CopyGm2UbTla() = default;

    template <class TensorDst, class TensorSrc, bool Compact = false>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, tla::Bool<Compact> = {})
    {
        static_assert(
            std::is_same_v<ElementSrc, ElementDst>,
            "CopyGm2UbTla: ElementSrc and ElementDst must match AscendC DataCopyPad T");
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, "
            "while TensorDst must be UB and RowMajor");

        const auto rows = tla::get<0>(srcTensor.originShape());
        const auto columns = tla::get<1>(srcTensor.originShape());
        if constexpr (Compact) {
            if (rows == 0 || columns == 0)
                return;
        }
        const uint32_t blockLen = ElementsToBytes<ElementSrc>(columns);
        const int64_t srcStride =
            ElementsToBytes<ElementSrc, int64_t>(tla::get<0>(srcTensor.stride())) - static_cast<int64_t>(blockLen);
        const int64_t dstStride = Compact ? 0 : (tla::get<0>(dstTensor.stride()) - columns) / ELE_NUM_PER_BLK;
        AscendC::DataCopyExtParams dataCopyParams(rows, blockLen, srcStride, dstStride, 0);

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        if constexpr (AscendC::Std::is_one_of_v<
                          ElementSrc, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, hifloat8_t, float4_e2m1x2_t,
                          float4_e1m2x2_t>) {
            // for float4_e2m1x2_t/float4_e1m2x2_t
            // in tensor.data()[offset], will auto convert fp4 offset to byte offset
            copyRows<Compact>(
                dstTensor.data()[dstOffset].template ReinterpretCast<uint8_t>(),
                srcTensor.data()[srcOffset].template ReinterpretCast<uint8_t>(), dataCopyParams);
        } else {
            copyRows<Compact>(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams);
        }
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const& dstTensor, TensorSrc const& srcTensor,
        AscendC::DataCopyPadExtParams<ElementSrc> const& padParams)
    {
        static_assert(
            std::is_same_v<ElementSrc, ElementDst>,
            "CopyGm2UbTla: ElementSrc and ElementDst must match AscendC DataCopyPad T");
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, "
            "while TensorDst must be UB and RowMajor");

        const uint32_t blockLen = ElementsToBytes<ElementSrc>(tla::get<1>(srcTensor.originShape()));
        const int64_t srcStride =
            ElementsToBytes<ElementSrc, int64_t>(tla::get<0>(srcTensor.stride())) - static_cast<int64_t>(blockLen);
        // AscendC Normal-mode dstStride is the gap after blockLen+left+right pads
        // (each measured in 32B blocks). Subtract pad widths so multi-row writes
        // land on dstTensor.stride<0>(), not stride + AlignUp(pad).
        const int64_t dstGapElems = tla::get<0>(dstTensor.stride()) - tla::get<1>(srcTensor.originShape()) -
                                    static_cast<int64_t>(padParams.leftPadding) -
                                    static_cast<int64_t>(padParams.rightPadding);

        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.originShape()), blockLen, srcStride, dstGapElems / ELE_NUM_PER_BLK, 0);

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        if constexpr (AscendC::Std::is_one_of_v<
                          ElementSrc, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, hifloat8_t, float4_e2m1x2_t,
                          float4_e1m2x2_t>) {
            // Low-precision types go through uint8_t DataCopyPad; forward pad
            // flags and the low 8 bits of paddingValue (UB fill, no OUTTOL1).
            uint8_t fillBits = *reinterpret_cast<const uint8_t*>(&padParams.paddingValue);
            AscendC::DataCopyPadExtParams<uint8_t> padParamsU8(
                padParams.isPad, padParams.leftPadding, padParams.rightPadding, fillBits);
            AscendC::DataCopyPad(
                dstTensor.data()[dstOffset].template ReinterpretCast<uint8_t>(),
                srcTensor.data()[srcOffset].template ReinterpretCast<uint8_t>(), dataCopyParams, padParamsU8);
        } else {
            AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
        }
    }

    // BC pad path for hif8/fp8/fp4 (ccec-safe uint8 ExtParams bag).
    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const& dstTensor, TensorSrc const& srcTensor, GmUbLowpPadParams const& lowpPad)
    {
        static_assert(
            AscendC::Std::is_one_of_v<
                ElementSrc, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, hifloat8_t, float4_e2m1x2_t, float4_e1m2x2_t>,
            "GmUbLowpPadParams is only for low-precision GM→UB RowMajor");
        static_assert(
            std::is_same_v<ElementSrc, ElementDst>,
            "CopyGm2UbTla: ElementSrc and ElementDst must match AscendC DataCopyPad T");
        const uint32_t blockLen = ElementsToBytes<ElementSrc>(tla::get<1>(srcTensor.originShape()));
        const int64_t srcStride =
            ElementsToBytes<ElementSrc, int64_t>(tla::get<0>(srcTensor.stride())) - static_cast<int64_t>(blockLen);
        const int64_t dstGapElems = tla::get<0>(dstTensor.stride()) - tla::get<1>(srcTensor.originShape()) -
                                    static_cast<int64_t>(lowpPad.pad.leftPadding) -
                                    static_cast<int64_t>(lowpPad.pad.rightPadding);
        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.originShape()), blockLen, srcStride, dstGapElems / ELE_NUM_PER_BLK, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(
            dstTensor.data()[dstOffset].template ReinterpretCast<uint8_t>(),
            srcTensor.data()[srcOffset].template ReinterpretCast<uint8_t>(), dataCopyParams, lowpPad.pad);
    }

private:
    template <bool Compact, class T>
    CATLASS_DEVICE static void copyRows(
        const AscendC::LocalTensor<T>& dst, const AscendC::GlobalTensor<T>& src,
        const AscendC::DataCopyExtParams& params)
    {
        AscendC::DataCopyPadExtParams<T> padParams(false, 0, 0, 0);
        if constexpr (Compact) {
            AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(dst, src, params, padParams);
        } else {
            AscendC::DataCopyPad(dst, src, params, padParams);
        }
    }
};
#endif

#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyGm2UbTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>>
    : CopyGm2UbTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
          std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyGm2UbTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>>
    : CopyGm2UbTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
          std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyGm2UbTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>>
    : CopyGm2UbTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
          std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyGm2UbTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>>
    : CopyGm2UbTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
          std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>> {};
#endif

} // namespace Catlass::Epilogue::Tile

#endif // CATLASS_EPILOGUE_TILE_COPY_GM_TO_UB_TLA_HPP
