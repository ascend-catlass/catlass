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

/// CopyGm2UbTla optional third argument is AscendC ``DataCopyPadExtParams<T>``.
///
/// Default two-arg path uses ``isPad=false, left=right=0, value=0`` (exact bytes).
/// Caller responsibility for pad constraints:
/// - ``leftPadding`` / ``rightPadding``: element counts; each side ≤ 32B
///   (``padElems * sizeof(T) ≤ 32``).
/// - When ``sizeof(T)==8`` (b64), ``paddingValue`` must be 0.
/// - ``blockLen`` must be a multiple of ``sizeof(T)``; GM src 1B-aligned,
///   UB dst 32B-aligned.
/// - Ascend950 RowMajor mx fp8/fp4 path reinterprets as ``uint8_t``; ExtParams
///   flags are forwarded, ``paddingValue`` is forced to 0 on that path.

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

        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.originShape()), tla::get<1>(srcTensor.originShape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.originShape())) * sizeof(ElementSrc),
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(srcTensor.originShape())) / ELE_NUM_PER_BLK, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
    }
};

/// Partial specialization for Ascend950, Vector in and Vector out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyGm2UbTla<
    Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>> {
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
            tla::detail::isVector<typename TensorSrc::Layout>::value &&
                tla::detail::isVector<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be GM and Vector, "
            "while TensorDst must be UB and Vector");

        AscendC::DataCopyExtParams dataCopyParams(
            1, tla::get<0>(srcTensor.originShape()) * sizeof(ElementSrc), 0, 0, 0);
        AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
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
            1, tla::get<0>(srcTensor.originShape()) * sizeof(ElementSrc), 0, 0, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
    }
};
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
#endif

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

        const uint32_t blockLen = ElementsToBytes<ElementSrc>(tla::get<1>(srcTensor.originShape()));
        const int64_t srcStride =
            ElementsToBytes<ElementSrc, int64_t>(tla::get<0>(srcTensor.stride())) - static_cast<int64_t>(blockLen);

        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.originShape()), blockLen, srcStride,
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(srcTensor.originShape())) / ELE_NUM_PER_BLK, 0);

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        if constexpr (AscendC::Std::is_one_of_v<
                          ElementSrc, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, float4_e2m1x2_t, float4_e1m2x2_t>) {
            AscendC::DataCopyPadExtParams<uint8_t> padParams(false, 0, 0, 0);
            // for float4_e2m1x2_t/float4_e1m2x2_t
            // in tensor.data()[offset], will auto convert fp4 offset to byte offset
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
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, "
            "while TensorDst must be UB and RowMajor");

        const uint32_t blockLen = ElementsToBytes<ElementSrc>(tla::get<1>(srcTensor.originShape()));
        const int64_t srcStride =
            ElementsToBytes<ElementSrc, int64_t>(tla::get<0>(srcTensor.stride())) - static_cast<int64_t>(blockLen);

        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.originShape()), blockLen, srcStride,
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(srcTensor.originShape())) / ELE_NUM_PER_BLK, 0);

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        if constexpr (AscendC::Std::is_one_of_v<
                          ElementSrc, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, float4_e2m1x2_t, float4_e1m2x2_t>) {
            // MX types go through uint8_t DataCopyPad; forward pad flags, force value 0.
            AscendC::DataCopyPadExtParams<uint8_t> padParamsU8(
                padParams.isPad, padParams.leftPadding, padParams.rightPadding, static_cast<uint8_t>(0));
            AscendC::DataCopyPad(
                dstTensor.data()[dstOffset].template ReinterpretCast<uint8_t>(),
                srcTensor.data()[srcOffset].template ReinterpretCast<uint8_t>(), dataCopyParams, padParamsU8);
        } else {
            AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
        }
    }
};
#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202)
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
