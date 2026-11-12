/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATLASS_GEMM_TILE_ASCEND960_COPY_L0C_TO_UB_HPP
#define CATLASS_GEMM_TILE_ASCEND960_COPY_L0C_TO_UB_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
constexpr AscendC::FixpipeConfig CFG_COLUMN_MAJOR_UB = {AscendC::CO2Layout::COLUMN_MAJOR, true};
#endif
#include "catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp"

namespace Catlass::Gemm::Tile {

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::NO_QUANT, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>>
    : CopyL0CToUBTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
          CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::NO_QUANT, ReluEnable_,
          std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {};
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::NO_QUANT, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>>
    : CopyL0CToUBTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
          CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::NO_QUANT, ReluEnable_,
          std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::NO_QUANT, ReluEnable_,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>>
    : CopyL0CToUBTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
          CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::NO_QUANT, ReluEnable_,
          std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {};
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::NO_QUANT, ReluEnable_,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>>
    : CopyL0CToUBTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
          CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::NO_QUANT, ReluEnable_,
          std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_TENSOR, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>>
    : CopyL0CToUBTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
          CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_TENSOR, ReluEnable_,
          std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {};
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_TENSOR, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>>
    : CopyL0CToUBTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
          CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_TENSOR, ReluEnable_,
          std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_TENSOR, ReluEnable_,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>>
    : CopyL0CToUBTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
          CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_TENSOR, ReluEnable_,
          std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {};
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_TENSOR, ReluEnable_,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>>
    : CopyL0CToUBTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
          CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_TENSOR, ReluEnable_,
          std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_CHANNEL, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>>
    : CopyL0CToUBTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
          CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_CHANNEL, ReluEnable_,
          std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {};
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_CHANNEL, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>>
    : CopyL0CToUBTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
          CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_CHANNEL, ReluEnable_,
          std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_CHANNEL, ReluEnable_,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>>
    : CopyL0CToUBTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
          CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_CHANNEL, ReluEnable_,
          std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {};
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_CHANNEL, ReluEnable_,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>>
    : CopyL0CToUBTla<
          Catlass::Arch::Ascend950, TensorSrc_,
          tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
          CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::PER_CHANNEL, ReluEnable_,
          std::enable_if_t<tla::detail::isColumnMajor<LayoutDst_>::value>> {};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::SPLIT_M, ScaleGranularity::NO_QUANT, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    static_assert(
        !std::is_same<TensorSrc_, TensorSrc_>::value && false,
        "SPLIT_M (dual-destination L0C->UB) is a 1:2-only concept; on Ascend960 (AIC:AIV = 1:1) "
        "instantiate CopyL0CToUBMode::NO_SPLIT instead (single dst, full tile to the one AIV's UB)");
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::SPLIT_M, ScaleGranularity::NO_QUANT, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    static_assert(
        !std::is_same<TensorSrc_, TensorSrc_>::value && false,
        "SPLIT_M (dual-destination L0C->UB) is a 1:2-only concept; on Ascend960 (AIC:AIV = 1:1) "
        "instantiate CopyL0CToUBMode::NO_SPLIT instead (single dst, full tile to the one AIV's UB)");
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960DT, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::SPLIT_N, ScaleGranularity::NO_QUANT, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    static_assert(
        !std::is_same<TensorSrc_, TensorSrc_>::value && false,
        "SPLIT_N (dual-destination L0C->UB) is a 1:2-only concept; on Ascend960 (AIC:AIV = 1:1) "
        "instantiate CopyL0CToUBMode::NO_SPLIT instead");
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend960PR, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::SPLIT_N, ScaleGranularity::NO_QUANT, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    static_assert(
        !std::is_same<TensorSrc_, TensorSrc_>::value && false,
        "SPLIT_N (dual-destination L0C->UB) is a 1:2-only concept; on Ascend960 (AIC:AIV = 1:1) "
        "instantiate CopyL0CToUBMode::NO_SPLIT instead");
};

} // namespace Catlass::Gemm::Tile

#endif
