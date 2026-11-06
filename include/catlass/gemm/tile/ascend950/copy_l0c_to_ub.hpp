/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_ASCEND950_COPY_L0C_TO_UB_HPP
#define CATLASS_GEMM_TILE_ASCEND950_COPY_L0C_TO_UB_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp"
#include "catlass/numeric_size.hpp"
#include "tla/tensor.hpp"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
constexpr AscendC::FixpipeConfig CFG_COLUMN_MAJOR_UB = {AscendC::CO2Layout::COLUMN_MAJOR, true};
#endif

namespace Catlass::Gemm::Tile {

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend950, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::NO_QUANT, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<ElementDst>::value;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = RoundUp(tla::get<1>(dstTensor.originShape()), ELE_NUM_PER_BLK);
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const& dstTensor, TensorSrc const& srcTensor, bool subBlockId, uint8_t unitFlag)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 0;
        intriParams.subBlockId = subBlockId;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend950, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::NO_QUANT, ReluEnable_,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<ElementDst>::value;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isColumnMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and ColumnMajor");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::COLUMN_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = RoundUp<ELE_NUM_PER_BLK>(tla::get<0>(dstTensor.originShape()));
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<1>(dstTensor.stride());

        intriParams.params = AscendC::Nz2DnParams(1, 0, 0, 1); // srcNzC0Stride=1

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_COLUMN_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const& dstTensor, TensorSrc const& srcTensor, bool subBlockId, uint8_t unitFlag)
    {
        static_assert(
            tla::detail::isColumnMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and ColumnMajor");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::COLUMN_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = RoundUp<ELE_NUM_PER_BLK>(tla::get<0>(dstTensor.originShape()));
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<1>(dstTensor.stride());

        intriParams.params = AscendC::Nz2DnParams(1, 0, 0, 1); // srcNzC0Stride=1

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 0;
        intriParams.subBlockId = subBlockId;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_COLUMN_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

// L0C -> UB RowMajor PER_TENSOR quant (NO_SPLIT)
// NO_SPLIT PER_TENSOR RowMajor
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend950, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_TENSOR, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<ElementDst>::value;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_TENSOR>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {
        float scale = 1.0;
        int32_t offset = 0;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(float scalar)
        {
            scale = scalar;
        }

        CATLASS_HOST_DEVICE
        Params(float scalar, int32_t offset_)
        {
            scale = scalar;
            offset = offset_;
        }
    };
    Params params;

    CATLASS_DEVICE
    CopyL0CToUBTla() = default;

    CATLASS_DEVICE
    CopyL0CToUBTla(Params const& params_) : params(params_) {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const& dstTensor, TensorSrc const& srcTensor, bool subBlockId, uint8_t unitFlag)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = RoundUp(tla::get<1>(dstTensor.originShape()), ELE_NUM_PER_BLK);
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        uint64_t quantBits = static_cast<uint64_t>(*reinterpret_cast<uint32_t*>(&params.scale));
        if constexpr (std::is_same_v<ElementDst, int8_t>) {
            constexpr uint64_t signBit = (1ULL << 46); // 1: int8_t, 0: uint8_t
            quantBits |= signBit;
        }
        // QUANT_PRE[45:37] carries a 9-bit signed Offset, added to the rounded,
        // scaled value before saturation. It takes effect for the int8_t/uint8_t ElementDst.
        // offset valid range [-256, 255]
        if constexpr (std::is_same_v<ElementDst, int8_t> || std::is_same_v<ElementDst, uint8_t>) {
            constexpr uint64_t offsetMask = 0x1FFULL; // 9 bits
            quantBits |= (static_cast<uint64_t>(params.offset) & offsetMask) << 37;
        }
        intriParams.deqScalar = quantBits;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 0;
        intriParams.subBlockId = subBlockId;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

// L0C -> UB, ColumnMajor, PER_TENSOR quant (NO_SPLIT).
// NO_SPLIT PER_TENSOR ColumnMajor
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend950, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_TENSOR, ReluEnable_,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_TENSOR>::VALUE;
    static constexpr auto reluEn = ReluEnable_;
    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<ElementDst>::value;

    struct Params {
        float scale = 1.0;
        int32_t offset = 0;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(float scalar)
        {
            scale = scalar;
        }

        CATLASS_HOST_DEVICE
        Params(float scalar, int32_t offset_)
        {
            scale = scalar;
            offset = offset_;
        }
    };
    Params params;

    CATLASS_DEVICE
    CopyL0CToUBTla() = default;

    CATLASS_DEVICE
    CopyL0CToUBTla(Params const& params_) : params(params_) {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isColumnMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and ColumnMajor");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::COLUMN_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = RoundUp<ELE_NUM_PER_BLK>(tla::get<0>(dstTensor.originShape()));
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<1>(dstTensor.stride());

        intriParams.params = AscendC::Nz2DnParams(1, 0, 0, 1); // srcNzC0Stride=1

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        uint64_t quantBits = static_cast<uint64_t>(*reinterpret_cast<uint32_t*>(&params.scale));
        if constexpr (std::is_same_v<ElementDst, int8_t>) {
            constexpr uint64_t signBit = (1ULL << 46); // 1: int8_t, 0: uint8_t
            quantBits |= signBit;
        }
        // QUANT_PRE[45:37] carries a 9-bit signed Offset, added to the rounded,
        // scaled value before saturation. It takes effect for the int8_t/uint8_t ElementDst.
        // offset valid range [-256, 255]
        if constexpr (std::is_same_v<ElementDst, int8_t> || std::is_same_v<ElementDst, uint8_t>) {
            constexpr uint64_t offsetMask = 0x1FFULL; // 9 bits
            quantBits |= (static_cast<uint64_t>(params.offset) & offsetMask) << 37;
        }
        intriParams.deqScalar = quantBits;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_COLUMN_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const& dstTensor, TensorSrc const& srcTensor, bool subBlockId, uint8_t unitFlag)
    {
        static_assert(
            tla::detail::isColumnMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and ColumnMajor");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::COLUMN_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = RoundUp<ELE_NUM_PER_BLK>(tla::get<0>(dstTensor.originShape()));
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<1>(dstTensor.stride());

        intriParams.params = AscendC::Nz2DnParams(1, 0, 0, 1); // srcNzC0Stride=1

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        uint64_t quantBits = static_cast<uint64_t>(*reinterpret_cast<uint32_t*>(&params.scale));
        if constexpr (std::is_same_v<ElementDst, int8_t>) {
            constexpr uint64_t signBit = (1ULL << 46); // 1: int8_t, 0: uint8_t
            quantBits |= signBit;
        }
        // QUANT_PRE[45:37] carries a 9-bit signed Offset, added to the rounded,
        // scaled value before saturation. It takes effect for the int8_t/uint8_t ElementDst.
        // offset valid range [-256, 255]
        if constexpr (std::is_same_v<ElementDst, int8_t> || std::is_same_v<ElementDst, uint8_t>) {
            constexpr uint64_t offsetMask = 0x1FFULL; // 9 bits
            quantBits |= (static_cast<uint64_t>(params.offset) & offsetMask) << 37;
        }
        intriParams.deqScalar = quantBits;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 0;
        intriParams.subBlockId = subBlockId;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_COLUMN_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

// L0C -> UB, RowMajor, PER_CHANNEL quant (NO_SPLIT). The dequant scale arrives as
// an L1 uint64 tensor, one element per output channel.
// NO_SPLIT PER_CHANNEL RowMajor
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend950, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_CHANNEL, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<ElementDst>::value;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_CHANNEL>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc, class TensorQuant>
    CATLASS_DEVICE void operator()(
        TensorDst const& dstTensor, TensorSrc const& srcTensor, TensorQuant const& quantTensor, bool subBlockId,
        uint8_t unitFlag)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = RoundUp(tla::get<1>(dstTensor.originShape()), ELE_NUM_PER_BLK);
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 0;
        intriParams.subBlockId = subBlockId;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        auto quantOffset = quantTensor.layout()(quantTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], quantTensor.data()[quantOffset], intriParams);
    }
};

// L0C -> UB, ColumnMajor, PER_CHANNEL quant (NO_SPLIT).
// NO_SPLIT PER_CHANNEL ColumnMajor
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend950, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_CHANNEL, ReluEnable_,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_CHANNEL>::VALUE;
    static constexpr auto reluEn = ReluEnable_;
    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<ElementDst>::value;

    template <class TensorDst, class TensorSrc, class TensorQuant>
    CATLASS_DEVICE void operator()(
        TensorDst const& dstTensor, TensorSrc const& srcTensor, TensorQuant const& quantTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isColumnMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and ColumnMajor");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::COLUMN_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = RoundUp<ELE_NUM_PER_BLK>(tla::get<0>(dstTensor.originShape()));
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<1>(dstTensor.stride());

        intriParams.params = AscendC::Nz2DnParams(1, 0, 0, 1); // srcNzC0Stride=1

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        auto quantOffset = quantTensor.layout()(quantTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_COLUMN_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], quantTensor.data()[quantOffset], intriParams);
    }

    template <class TensorDst, class TensorSrc, class TensorQuant>
    CATLASS_DEVICE void operator()(
        TensorDst const& dstTensor, TensorSrc const& srcTensor, TensorQuant const& quantTensor, bool subBlockId,
        uint8_t unitFlag)
    {
        static_assert(
            tla::detail::isColumnMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and ColumnMajor");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::COLUMN_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = RoundUp<ELE_NUM_PER_BLK>(tla::get<0>(dstTensor.originShape()));
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<1>(dstTensor.stride());

        intriParams.params = AscendC::Nz2DnParams(1, 0, 0, 1); // srcNzC0Stride=1

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 0;
        intriParams.subBlockId = subBlockId;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        auto quantOffset = quantTensor.layout()(quantTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_COLUMN_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], quantTensor.data()[quantOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend950, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::SPLIT_M, ScaleGranularity::NO_QUANT, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<ElementDst>::value;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = RoundUp(tla::get<1>(dstTensor.originShape()), ELE_NUM_PER_BLK);
        intriParams.mSize = RoundUp(tla::get<0>(dstTensor.originShape()), 2); // m must be even when split m
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 1;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend950, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::SPLIT_N, ScaleGranularity::NO_QUANT, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = RoundUp(tla::get<1>(dstTensor.originShape()), 32); // n must be multiple of 32 when split n
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 2;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_ASCEND950_COPY_L0C_TO_UB_HPP
