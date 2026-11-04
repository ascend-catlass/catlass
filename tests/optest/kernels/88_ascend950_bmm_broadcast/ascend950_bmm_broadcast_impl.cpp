/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <stdexcept>

#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/batched_matmul_broadcast_tla.h"
#include "catlass/gemm/tile/tile_copy.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler_tla.h"

using namespace Catlass;

// The JIT entry supplies all type, layout and feature compilation parameters.
constexpr bool hasBias = CATLASS_BMM_HAS_BIAS != 0;
constexpr bool useHF32 = CATLASS_BMM_HF32 != 0;
using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementBias = std::conditional_t<hasBias, ElementA, void>;
using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = layout::RowMajor;
// Default stages: single L0C, double L1A/L1B/L0A/L0B, no L1 residency.
using DispatchPolicy = Gemm::MmadPingpong<Arch::Ascend950, false, useHF32>;
// Fixed tile tuning: M/N=64/128; scale K from the FP16 base tile by element size.
using L1TileShape =
    typename CatlassKernel::TileShapeScalerTLA<ElementA, half, tla::Shape<tla::C<64U>, tla::C<128U>, tla::C<128U>>>::type;
using L0TileShape =
    typename CatlassKernel::TileShapeScalerTLA<ElementA, half, tla::Shape<tla::C<64U>, tla::C<128U>, tla::C<32U>>>::type;
// TileShapeScalerTLA matches uint32_t constants; verify that K was scaled.
static_assert(tla::get<2>(L1TileShape{}) * sizeof(ElementA) == 128U * sizeof(half), "Unexpected L1 K scaling");
static_assert(tla::get<2>(L0TileShape{}) * sizeof(ElementA) == 32U * sizeof(half), "Unexpected L0 K scaling");
using BaseTileCopy = Gemm::Tile::PackedTileCopyTla<
    Arch::Ascend950, ElementA, LayoutTagA, ElementA, LayoutTagB, ElementA, LayoutTagC, ElementBias>;
using TileCopy = Gemm::Tile::PaddedTileCopy<BaseTileCopy>;
using BlockMmad = Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementA, ElementA, ElementBias, TileCopy>;
// Swizzle offset 3 groups three M tiles; direction 0 selects the Zn traversal.
using MatmulKernel = Gemm::Kernel::BatchedMatmulBroadcastTla<BlockMmad, Gemm::Block::GemmIdentityBlockSwizzle<3, 0>>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::BmmBroadcastParams* params)
{
    typename MatmulKernel::Arguments arguments{
        params->batch,
        params->batchA,
        params->batchB,
        GemmCoord{params->m, params->n, params->k},
        params->inputAddr[0],
        tla::MakeLayout<ElementA, LayoutTagA>(params->m, params->k),
        params->inputAddr[1],
        tla::MakeLayout<ElementA, LayoutTagB>(params->k, params->n),
        params->outputAddr[0],
        tla::MakeLayout<ElementA, LayoutTagC>(params->m, params->n),
        nullptr,
        static_cast<Gemm::Kernel::BiasMode>(params->biasMode)};
    if (params->biasMode != 0) {
        arguments.ptrBias = params->inputAddr[2];
    }
    if (!MatmulKernel::CanImplement(arguments)) {
        throw std::invalid_argument("Unsupported BMM broadcast arguments");
    }
    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
