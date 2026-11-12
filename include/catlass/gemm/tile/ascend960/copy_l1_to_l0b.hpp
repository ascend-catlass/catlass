/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATLASS_GEMM_TILE_ASCEND960_COPY_L1_TO_L0B_HPP
#define CATLASS_GEMM_TILE_ASCEND960_COPY_L1_TO_L0B_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

#include "catlass/gemm/tile/ascend950/copy_l1_to_l0b.hpp"

namespace Catlass::Gemm::Tile {

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<
        !AscendC::Std::is_one_of_v<
            ElementSrc, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
        !AscendC::Std::is_one_of_v<
            ElementDst, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
          std::enable_if_t<
              !AscendC::Std::is_one_of_v<
                  ElementSrc, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
              !AscendC::Std::is_one_of_v<
                  ElementDst, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
              tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<
        !AscendC::Std::is_one_of_v<
            ElementSrc, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
        !AscendC::Std::is_one_of_v<
            ElementDst, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
          std::enable_if_t<
              !AscendC::Std::is_one_of_v<
                  ElementSrc, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
              !AscendC::Std::is_one_of_v<
                  ElementDst, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
              tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {};

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<
        AscendC::Std::is_one_of_v<ElementSrc, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
        AscendC::Std::is_one_of_v<ElementDst, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
          std::enable_if_t<
              AscendC::Std::is_one_of_v<
                  ElementSrc, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
              AscendC::Std::is_one_of_v<
                  ElementDst, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
              tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<
        AscendC::Std::is_one_of_v<ElementSrc, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
        AscendC::Std::is_one_of_v<ElementDst, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
          std::enable_if_t<
              AscendC::Std::is_one_of_v<
                  ElementSrc, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
              AscendC::Std::is_one_of_v<
                  ElementDst, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t> &&
              tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {};

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<
        tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
          std::enable_if_t<
              tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
    std::enable_if_t<
        tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
          std::enable_if_t<
              tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {};

template <class Element>
struct CopyL1ToL0B<Arch::Ascend960DT, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>>
    : CopyL1ToL0B<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {};
template <class Element>
struct CopyL1ToL0B<Arch::Ascend960PR, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>>
    : CopyL1ToL0B<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {};

template <class Element>
struct CopyL1ToL0B<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>>
    : CopyL1ToL0B<
          Arch::Ascend950, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>,
          Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>> {};
template <class Element>
struct CopyL1ToL0B<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>>
    : CopyL1ToL0B<
          Arch::Ascend950, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>,
          Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>> {};

template <class Element>
struct CopyL1ToL0B<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>,
    std::enable_if_t<
        !AscendC::Std::is_one_of_v<Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>>
    : CopyL1ToL0B<
          Arch::Ascend950, Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>,
          Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>,
          std::enable_if_t<!AscendC::Std::is_one_of_v<
              Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>> {};
template <class Element>
struct CopyL1ToL0B<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>,
    std::enable_if_t<
        !AscendC::Std::is_one_of_v<Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>>
    : CopyL1ToL0B<
          Arch::Ascend950, Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>,
          Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>,
          std::enable_if_t<!AscendC::Std::is_one_of_v<
              Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>> {};

template <class Element>
struct CopyL1ToL0B<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>, void,
    std::enable_if_t<
        !AscendC::Std::is_one_of_v<Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>>
    : CopyL1ToL0B<
          Arch::Ascend950, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>, void,
          std::enable_if_t<!AscendC::Std::is_one_of_v<
              Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>> {};
template <class Element>
struct CopyL1ToL0B<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>, void,
    std::enable_if_t<
        !AscendC::Std::is_one_of_v<Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>>
    : CopyL1ToL0B<
          Arch::Ascend950, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>, void,
          std::enable_if_t<!AscendC::Std::is_one_of_v<
              Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>> {};

template <class Element>
struct CopyL1ToL0B<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>, void,
    std::enable_if_t<
        AscendC::Std::is_one_of_v<Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>>
    : CopyL1ToL0B<
          Arch::Ascend950, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>, void,
          std::enable_if_t<AscendC::Std::is_one_of_v<
              Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>> {};
template <class Element>
struct CopyL1ToL0B<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>, void,
    std::enable_if_t<
        AscendC::Std::is_one_of_v<Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>>
    : CopyL1ToL0B<
          Arch::Ascend950, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>, void,
          std::enable_if_t<AscendC::Std::is_one_of_v<
              Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>> {};

template <class Element>
struct CopyL1ToL0B<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>,
    std::enable_if_t<
        !AscendC::Std::is_one_of_v<Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>>
    : CopyL1ToL0B<
          Arch::Ascend950, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>,
          Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>,
          std::enable_if_t<!AscendC::Std::is_one_of_v<
              Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>> {};
template <class Element>
struct CopyL1ToL0B<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>,
    std::enable_if_t<
        !AscendC::Std::is_one_of_v<Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>>
    : CopyL1ToL0B<
          Arch::Ascend950, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>,
          Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>,
          std::enable_if_t<!AscendC::Std::is_one_of_v<
              Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>> {};

template <class Element>
struct CopyL1ToL0B<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>,
    std::enable_if_t<
        AscendC::Std::is_one_of_v<Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>>
    : CopyL1ToL0B<
          Arch::Ascend950, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>,
          Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>,
          std::enable_if_t<AscendC::Std::is_one_of_v<
              Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>> {};
template <class Element>
struct CopyL1ToL0B<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>,
    std::enable_if_t<
        AscendC::Std::is_one_of_v<Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>>
    : CopyL1ToL0B<
          Arch::Ascend950, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>,
          Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>,
          std::enable_if_t<AscendC::Std::is_one_of_v<
              Element, int8_t, float8_e4m3_t, float8_e5m2_t, float4_e2m1x2_t, float4_e1m2x2_t>>> {};

} // namespace Catlass::Gemm::Tile

#endif
