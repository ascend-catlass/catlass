/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATLASS_GEMM_TILE_ASCEND960_COPY_GM_TO_L1_HPP
#define CATLASS_GEMM_TILE_ASCEND960_COPY_GM_TO_L1_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

#include "catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp"

namespace Catlass::Gemm::Tile {

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {};

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {};

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {};

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {};

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>> {};
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>> {};

template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleForRowMajorA<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScaleForzZ<float8_e8m0_t, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isMxScaleForRowMajorA<float8_e8m0_t, LayoutSrc>::value &&
              tla::detail::isMxScaleForzZ<float8_e8m0_t, LayoutDst>::value>> {};
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleForRowMajorA<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScaleForzZ<float8_e8m0_t, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isMxScaleForRowMajorA<float8_e8m0_t, LayoutSrc>::value &&
              tla::detail::isMxScaleForzZ<float8_e8m0_t, LayoutDst>::value>> {};

template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleForColumnMajorA<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScaleForzZ<float8_e8m0_t, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isMxScaleForColumnMajorA<float8_e8m0_t, LayoutSrc>::value &&
              tla::detail::isMxScaleForzZ<float8_e8m0_t, LayoutDst>::value>> {};
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleForColumnMajorA<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScaleForzZ<float8_e8m0_t, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isMxScaleForColumnMajorA<float8_e8m0_t, LayoutSrc>::value &&
              tla::detail::isMxScaleForzZ<float8_e8m0_t, LayoutDst>::value>> {};

template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleForRowMajorB<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScaleFornN<float8_e8m0_t, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isMxScaleForRowMajorB<float8_e8m0_t, LayoutSrc>::value &&
              tla::detail::isMxScaleFornN<float8_e8m0_t, LayoutDst>::value>> {};
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleForRowMajorB<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScaleFornN<float8_e8m0_t, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isMxScaleForRowMajorB<float8_e8m0_t, LayoutSrc>::value &&
              tla::detail::isMxScaleFornN<float8_e8m0_t, LayoutDst>::value>> {};

template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960DT, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleForColumnMajorB<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScaleFornN<float8_e8m0_t, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isMxScaleForColumnMajorB<float8_e8m0_t, LayoutSrc>::value &&
              tla::detail::isMxScaleFornN<float8_e8m0_t, LayoutDst>::value>> {};
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend960PR, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleForColumnMajorB<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScaleFornN<float8_e8m0_t, LayoutDst>::value>>
    : TileCopyTla<
          Arch::Ascend950, tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
          tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
          std::enable_if_t<
              tla::detail::isMxScaleForColumnMajorB<float8_e8m0_t, LayoutSrc>::value &&
              tla::detail::isMxScaleFornN<float8_e8m0_t, LayoutDst>::value>> {};

template <class Element>
struct CopyGmToL1<Arch::Ascend960DT, Gemm::GemmType<Element, layout::RowMajor>>
    : CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>> {};
template <class Element>
struct CopyGmToL1<Arch::Ascend960PR, Gemm::GemmType<Element, layout::RowMajor>>
    : CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>> {};

template <class Element>
struct CopyGmToL1<Arch::Ascend960DT, Gemm::GemmType<Element, layout::ColumnMajor>>
    : CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>> {};
template <class Element>
struct CopyGmToL1<Arch::Ascend960PR, Gemm::GemmType<Element, layout::ColumnMajor>>
    : CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>> {};

template <class Element>
struct CopyGmToL1<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::ColumnMajor>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>,
          Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>> {};
template <class Element>
struct CopyGmToL1<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::ColumnMajor>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>,
          Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>> {};

template <class Element>
struct CopyGmToL1<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::ColumnMajor>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>>
    : public CopyGmToL1<Arch::Ascend960DT, Gemm::GemmType<Element, layout::ColumnMajor>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>,
          Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>> {};
template <class Element>
struct CopyGmToL1<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::ColumnMajor>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>>
    : public CopyGmToL1<Arch::Ascend960PR, Gemm::GemmType<Element, layout::ColumnMajor>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>,
          Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>> {};

template <class Element>
struct CopyGmToL1<Arch::Ascend960DT, Gemm::GemmType<Element, layout::zN>>
    : CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::zN>> {};
template <class Element>
struct CopyGmToL1<Arch::Ascend960PR, Gemm::GemmType<Element, layout::zN>>
    : CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::zN>> {};

template <class Element>
struct CopyGmToL1<Arch::Ascend960DT, Gemm::GemmType<Element, layout::nZ>>
    : CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ>> {};
template <class Element>
struct CopyGmToL1<Arch::Ascend960PR, Gemm::GemmType<Element, layout::nZ>>
    : CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ>> {};

template <class Element>
struct CopyGmToL1<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A1>,
    std::enable_if_t<std::is_same_v<Element, AscendC::fp8_e8m0_t>>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
          Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A1>,
          std::enable_if_t<std::is_same_v<Element, AscendC::fp8_e8m0_t>>> {};
template <class Element>
struct CopyGmToL1<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A1>,
    std::enable_if_t<std::is_same_v<Element, AscendC::fp8_e8m0_t>>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
          Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A1>,
          std::enable_if_t<std::is_same_v<Element, AscendC::fp8_e8m0_t>>> {};

template <class Element>
struct CopyGmToL1<Arch::Ascend960DT, Gemm::GemmType<Element, layout::PaddingRowMajor>>
    : CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingRowMajor>> {};
template <class Element>
struct CopyGmToL1<Arch::Ascend960PR, Gemm::GemmType<Element, layout::PaddingRowMajor>>
    : CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingRowMajor>> {};

template <class Element>
struct CopyGmToL1<Arch::Ascend960DT, Gemm::GemmType<Element, layout::PaddingColumnMajor>>
    : CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingColumnMajor>> {};
template <class Element>
struct CopyGmToL1<Arch::Ascend960PR, Gemm::GemmType<Element, layout::PaddingColumnMajor>>
    : CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingColumnMajor>> {};

template <class Element>
struct CopyGmToL1<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
          Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {};
template <class Element>
struct CopyGmToL1<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
          Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {};

template <class Element>
struct CopyGmToL1<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
          Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>> {};
template <class Element>
struct CopyGmToL1<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
          Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>> {};

template <class Element>
struct CopyGmToL1<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
          Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>> {};
template <class Element>
struct CopyGmToL1<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
          Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>> {};

template <class Element>
struct CopyGmToL1<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::RowMajor, AscendC::TPosition::A1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
          Gemm::GemmType<Element, layout::RowMajor, AscendC::TPosition::A1>> {};
template <class Element>
struct CopyGmToL1<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::RowMajor, AscendC::TPosition::A1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>,
          Gemm::GemmType<Element, layout::RowMajor, AscendC::TPosition::A1>> {};

template <class Element, AscendC::TPosition Position>
struct CopyGmToL1<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::ColumnMajor>, Gemm::GemmType<Element, layout::nN, Position>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>,
          Gemm::GemmType<Element, layout::nN, Position>> {};
template <class Element, AscendC::TPosition Position>
struct CopyGmToL1<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::ColumnMajor>, Gemm::GemmType<Element, layout::nN, Position>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>,
          Gemm::GemmType<Element, layout::nN, Position>> {};

template <class Element>
struct CopyGmToL1<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::VectorLayout>,
    Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::VectorLayout>,
          Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {};
template <class Element>
struct CopyGmToL1<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::VectorLayout>,
    Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::VectorLayout>,
          Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {};

template <class Element>
struct CopyGmToL1<
    Arch::Ascend960DT, Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>,
    Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>,
          Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>> {};
template <class Element>
struct CopyGmToL1<
    Arch::Ascend960PR, Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>,
    Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>>
    : CopyGmToL1<
          Arch::Ascend950, Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>,
          Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>> {};

template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend960DT, Gemm::GemmType<Element, layout::zN>>
    : public CopyGmToL1<Arch::Ascend960DT, Gemm::GemmType<Element, layout::zN>>
    : CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::zN>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::zN>> {};
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend960PR, Gemm::GemmType<Element, layout::zN>>
    : public CopyGmToL1<Arch::Ascend960PR, Gemm::GemmType<Element, layout::zN>>
    : CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::zN>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::zN>> {};

template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend960DT, Gemm::GemmType<Element, layout::nZ>>
    : public CopyGmToL1<Arch::Ascend960DT, Gemm::GemmType<Element, layout::nZ>>
    : CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ>> {};
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend960PR, Gemm::GemmType<Element, layout::nZ>>
    : public CopyGmToL1<Arch::Ascend960PR, Gemm::GemmType<Element, layout::nZ>>
    : CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::nZ>> {};

template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend960DT, Gemm::GemmType<Element, layout::RowMajor>>
    : CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>> {};
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend960PR, Gemm::GemmType<Element, layout::RowMajor>>
    : CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>> {};

template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend960DT, Gemm::GemmType<Element, layout::ColumnMajor>>
    : CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>> {};
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend960PR, Gemm::GemmType<Element, layout::ColumnMajor>>
    : CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::ColumnMajor>> {};

template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend960DT, Gemm::GemmType<Element, layout::PaddingRowMajor>>
    : public CopyGmToL1<Arch::Ascend960DT, Gemm::GemmType<Element, layout::PaddingRowMajor>>
    : CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingRowMajor>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingRowMajor>> {};
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend960PR, Gemm::GemmType<Element, layout::PaddingRowMajor>>
    : public CopyGmToL1<Arch::Ascend960PR, Gemm::GemmType<Element, layout::PaddingRowMajor>>
    : CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingRowMajor>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingRowMajor>> {};

template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend960DT, Gemm::GemmType<Element, layout::PaddingColumnMajor>>
    : public CopyGmToL1<Arch::Ascend960DT, Gemm::GemmType<Element, layout::PaddingColumnMajor>>
    : CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingColumnMajor>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingColumnMajor>> {};
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::Ascend960PR, Gemm::GemmType<Element, layout::PaddingColumnMajor>>
    : public CopyGmToL1<Arch::Ascend960PR, Gemm::GemmType<Element, layout::PaddingColumnMajor>>
    : CopyGmToL1DynamicOptimized<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingColumnMajor>>
    : public CopyGmToL1<Arch::Ascend950, Gemm::GemmType<Element, layout::PaddingColumnMajor>> {};

template <class Element>
struct CopyGmToL1GMMPTD<Arch::Ascend960DT, Gemm::GemmType<Element, layout::RowMajor>>
    : CopyGmToL1GMMPTD<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>> {};
template <class Element>
struct CopyGmToL1GMMPTD<Arch::Ascend960PR, Gemm::GemmType<Element, layout::RowMajor>>
    : CopyGmToL1GMMPTD<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>> {};

template <>
struct CopyGmToL1IntervalDataCopy<Arch::Ascend960DT, Gemm::GemmType<half, layout::RowMajor>>
    : CopyGmToL1IntervalDataCopy<Arch::Ascend950, Gemm::GemmType<half, layout::RowMajor>> {};
template <>
struct CopyGmToL1IntervalDataCopy<Arch::Ascend960PR, Gemm::GemmType<half, layout::RowMajor>>
    : CopyGmToL1IntervalDataCopy<Arch::Ascend950, Gemm::GemmType<half, layout::RowMajor>> {};

template <>
struct CopyGmToL1IntervalDataCopy<Arch::Ascend960DT, Gemm::GemmType<half, layout::ColumnMajor>>
    : CopyGmToL1IntervalDataCopy<Arch::Ascend950, Gemm::GemmType<half, layout::ColumnMajor>> {};
template <>
struct CopyGmToL1IntervalDataCopy<Arch::Ascend960PR, Gemm::GemmType<half, layout::ColumnMajor>>
    : CopyGmToL1IntervalDataCopy<Arch::Ascend950, Gemm::GemmType<half, layout::ColumnMajor>> {};

template <>
struct CopyGmToL1IntervalDataCopy<Arch::Ascend960DT, Gemm::GemmType<half, layout::PaddingRowMajor>>
    : CopyGmToL1IntervalDataCopy<Arch::Ascend950, Gemm::GemmType<half, layout::PaddingRowMajor>> {};
template <>
struct CopyGmToL1IntervalDataCopy<Arch::Ascend960PR, Gemm::GemmType<half, layout::PaddingRowMajor>>
    : CopyGmToL1IntervalDataCopy<Arch::Ascend950, Gemm::GemmType<half, layout::PaddingRowMajor>> {};

template <>
struct CopyGmToL1IntervalDataCopy<Arch::Ascend960DT, Gemm::GemmType<half, layout::PaddingColumnMajor>>
    : CopyGmToL1IntervalDataCopy<Arch::Ascend950, Gemm::GemmType<half, layout::PaddingColumnMajor>> {};
template <>
struct CopyGmToL1IntervalDataCopy<Arch::Ascend960PR, Gemm::GemmType<half, layout::PaddingColumnMajor>>
    : CopyGmToL1IntervalDataCopy<Arch::Ascend950, Gemm::GemmType<half, layout::PaddingColumnMajor>> {};

} // namespace Catlass::Gemm::Tile

#endif
