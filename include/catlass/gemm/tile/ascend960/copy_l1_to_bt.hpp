/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATLASS_GEMM_TILE_ASCEND960_COPY_L1_TO_BT_HPP
#define CATLASS_GEMM_TILE_ASCEND960_COPY_L1_TO_BT_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

#include "catlass/gemm/tile/ascend950/copy_l1_to_bt.hpp"

namespace Catlass::Gemm::Tile {

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::C2>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::C2>,
          std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::C2>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::C2>,
          std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>> {};

template <class ElementSrc, class ElementDst>
struct CopyL1ToBT<
    Arch::Ascend960DT, Catlass::Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2>>
    : CopyL1ToBT<
          Arch::Ascend950, Catlass::Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>,
          Catlass::Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2>> {};
template <class ElementSrc, class ElementDst>
struct CopyL1ToBT<
    Arch::Ascend960PR, Catlass::Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2>>
    : CopyL1ToBT<
          Arch::Ascend950, Catlass::Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>,
          Catlass::Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2>> {};

} // namespace Catlass::Gemm::Tile

#endif
