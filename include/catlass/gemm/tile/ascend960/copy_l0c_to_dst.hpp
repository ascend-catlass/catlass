/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATLASS_GEMM_TILE_ASCEND960_COPY_L0C_TO_DST_HPP
#define CATLASS_GEMM_TILE_ASCEND960_COPY_L0C_TO_DST_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

#include "catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp"

namespace Catlass::Gemm::Tile {

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, float, float, ScaleGranularity::NO_QUANT>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, float, ScaleGranularity::NO_QUANT> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, float, float, ScaleGranularity::NO_QUANT>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, float, ScaleGranularity::NO_QUANT> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, float, half, ScaleGranularity::NO_QUANT>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, half, ScaleGranularity::NO_QUANT> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, float, half, ScaleGranularity::NO_QUANT>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, half, ScaleGranularity::NO_QUANT> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, float, bfloat16_t, ScaleGranularity::NO_QUANT>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, bfloat16_t, ScaleGranularity::NO_QUANT> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, float, bfloat16_t, ScaleGranularity::NO_QUANT>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, bfloat16_t, ScaleGranularity::NO_QUANT> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, float, uint8_t, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, uint8_t, ScaleGranularity::PER_TENSOR> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, float, uint8_t, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, uint8_t, ScaleGranularity::PER_TENSOR> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, float, uint8_t, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, uint8_t, ScaleGranularity::PER_CHANNEL> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, float, uint8_t, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, uint8_t, ScaleGranularity::PER_CHANNEL> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, float, int8_t, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, int8_t, ScaleGranularity::PER_TENSOR> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, float, int8_t, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, int8_t, ScaleGranularity::PER_TENSOR> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, float, int8_t, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, int8_t, ScaleGranularity::PER_CHANNEL> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, float, int8_t, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, int8_t, ScaleGranularity::PER_CHANNEL> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, int32_t, int32_t, ScaleGranularity::NO_QUANT>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, int32_t, ScaleGranularity::NO_QUANT> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, int32_t, int32_t, ScaleGranularity::NO_QUANT>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, int32_t, ScaleGranularity::NO_QUANT> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, int32_t, half, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, half, ScaleGranularity::PER_TENSOR> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, int32_t, half, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, half, ScaleGranularity::PER_TENSOR> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, int32_t, half, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, half, ScaleGranularity::PER_CHANNEL> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, int32_t, half, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, half, ScaleGranularity::PER_CHANNEL> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, int32_t, uint8_t, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, uint8_t, ScaleGranularity::PER_TENSOR> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, int32_t, uint8_t, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, uint8_t, ScaleGranularity::PER_TENSOR> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, int32_t, uint8_t, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, uint8_t, ScaleGranularity::PER_CHANNEL> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, int32_t, uint8_t, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, uint8_t, ScaleGranularity::PER_CHANNEL> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, int32_t, int8_t, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, int8_t, ScaleGranularity::PER_TENSOR> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, int32_t, int8_t, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, int8_t, ScaleGranularity::PER_TENSOR> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, int32_t, int8_t, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, int8_t, ScaleGranularity::PER_CHANNEL> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, int32_t, int8_t, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, int8_t, ScaleGranularity::PER_CHANNEL> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, int32_t, bfloat16_t, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, bfloat16_t, ScaleGranularity::PER_TENSOR> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, int32_t, bfloat16_t, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, bfloat16_t, ScaleGranularity::PER_TENSOR> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, int32_t, bfloat16_t, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, bfloat16_t, ScaleGranularity::PER_CHANNEL> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, int32_t, bfloat16_t, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, int32_t, bfloat16_t, ScaleGranularity::PER_CHANNEL> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, float, half, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, half, ScaleGranularity::PER_TENSOR> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, float, half, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, half, ScaleGranularity::PER_TENSOR> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, float, half, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, half, ScaleGranularity::PER_CHANNEL> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, float, half, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, half, ScaleGranularity::PER_CHANNEL> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, float, bfloat16_t, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, bfloat16_t, ScaleGranularity::PER_TENSOR> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, float, bfloat16_t, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, bfloat16_t, ScaleGranularity::PER_TENSOR> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, float, bfloat16_t, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, bfloat16_t, ScaleGranularity::PER_CHANNEL> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, float, bfloat16_t, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, bfloat16_t, ScaleGranularity::PER_CHANNEL> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, float, float, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, float, ScaleGranularity::PER_TENSOR> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, float, float, ScaleGranularity::PER_TENSOR>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, float, ScaleGranularity::PER_TENSOR> {};

template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960DT, float, float, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, float, ScaleGranularity::PER_CHANNEL> {};
template <>
struct CopyL0CToDstQuantMode<Arch::Ascend960PR, float, float, ScaleGranularity::PER_CHANNEL>
    : CopyL0CToDstQuantMode<Arch::Ascend950, float, float, ScaleGranularity::PER_CHANNEL> {};

} // namespace Catlass::Gemm::Tile

#endif
