/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATLASS_GEMM_TILE_ASCEND960_COPY_L0C_TO_GM_HPP
#define CATLASS_GEMM_TILE_ASCEND960_COPY_L0C_TO_GM_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

#include "catlass/gemm/tile/ascend950/copy_l0c_to_gm.hpp"

namespace Catlass::Gemm::Tile {

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::NO_QUANT, ReluEnable_, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::NO_QUANT, ReluEnable_, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {};
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::NO_QUANT, ReluEnable_, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::NO_QUANT, ReluEnable_, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::NO_QUANT, ReluEnable_, std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::NO_QUANT, ReluEnable_, std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {};
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::NO_QUANT, ReluEnable_, std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::NO_QUANT, ReluEnable_, std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::NO_QUANT, ReluEnable_, std::enable_if_t<tla::detail::iszN<ElementDst_, LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::NO_QUANT, ReluEnable_,
          std::enable_if_t<tla::detail::iszN<ElementDst_, LayoutDst_>::value>> {};
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::NO_QUANT, ReluEnable_, std::enable_if_t<tla::detail::iszN<ElementDst_, LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::NO_QUANT, ReluEnable_,
          std::enable_if_t<tla::detail::iszN<ElementDst_, LayoutDst_>::value>> {};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::PER_TENSOR, ReluEnable_, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::PER_TENSOR, ReluEnable_, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {};
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::PER_TENSOR, ReluEnable_, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::PER_TENSOR, ReluEnable_, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::PER_TENSOR, ReluEnable_, std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::PER_TENSOR, ReluEnable_, std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {
};
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::PER_TENSOR, ReluEnable_, std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::PER_TENSOR, ReluEnable_, std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::PER_CHANNEL, ReluEnable_, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::PER_CHANNEL, ReluEnable_, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {};
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::PER_CHANNEL, ReluEnable_, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::PER_CHANNEL, ReluEnable_, std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::PER_CHANNEL, ReluEnable_, std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::PER_CHANNEL, ReluEnable_, std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {
};
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::PER_CHANNEL, ReluEnable_, std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>>
    : CopyL0CToGmTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
          ScaleGranularity::PER_CHANNEL, ReluEnable_, std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {
};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::Ascend960DT, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
    ScaleGranularity::NO_QUANT, ReluEnable_>
    : CopyL0CToGm<
          Catlass::Arch::Ascend950, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
          ScaleGranularity::NO_QUANT, ReluEnable_> {};
template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::Ascend960PR, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
    ScaleGranularity::NO_QUANT, ReluEnable_>
    : CopyL0CToGm<
          Catlass::Arch::Ascend950, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
          ScaleGranularity::NO_QUANT, ReluEnable_> {};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::Ascend960DT, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
    ScaleGranularity::PER_TENSOR, ReluEnable_>
    : CopyL0CToGm<
          Catlass::Arch::Ascend950, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
          ScaleGranularity::PER_TENSOR, ReluEnable_> {};
template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::Ascend960PR, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
    ScaleGranularity::PER_TENSOR, ReluEnable_>
    : CopyL0CToGm<
          Catlass::Arch::Ascend950, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
          ScaleGranularity::PER_TENSOR, ReluEnable_> {};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::Ascend960DT, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
    ScaleGranularity::PER_CHANNEL, ReluEnable_>
    : CopyL0CToGm<
          Catlass::Arch::Ascend950, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
          ScaleGranularity::PER_CHANNEL, ReluEnable_> {};
template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::Ascend960PR, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
    ScaleGranularity::PER_CHANNEL, ReluEnable_>
    : CopyL0CToGm<
          Catlass::Arch::Ascend950, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
          ScaleGranularity::PER_CHANNEL, ReluEnable_> {};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::Ascend960DT, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::zN>,
    ScaleGranularity::NO_QUANT, ReluEnable_>
    : CopyL0CToGm<
          Catlass::Arch::Ascend950, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::zN>,
          ScaleGranularity::NO_QUANT, ReluEnable_> {};
template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::Ascend960PR, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::zN>,
    ScaleGranularity::NO_QUANT, ReluEnable_>
    : CopyL0CToGm<
          Catlass::Arch::Ascend950, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::zN>,
          ScaleGranularity::NO_QUANT, ReluEnable_> {};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::Ascend960DT, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::NDC1HWC0>,
    ScaleGranularity::NO_QUANT, ReluEnable_>
    : CopyL0CToGm<
          Catlass::Arch::Ascend950, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::NDC1HWC0>,
          ScaleGranularity::NO_QUANT, ReluEnable_> {};
template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<
    Catlass::Arch::Ascend960PR, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::NDC1HWC0>,
    ScaleGranularity::NO_QUANT, ReluEnable_>
    : CopyL0CToGm<
          Catlass::Arch::Ascend950, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::NDC1HWC0>,
          ScaleGranularity::NO_QUANT, ReluEnable_> {};

} // namespace Catlass::Gemm::Tile

#endif
