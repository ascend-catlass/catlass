/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See the License
 * root for the full text of the License.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_scheduler_iterBatch.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/iterbatch_broadcast_matmul_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler_tla.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A float
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B float
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C float
#endif
#ifndef CATLASS_JIT_ELEMENT_BIAS
#define CATLASS_JIT_ELEMENT_BIAS float
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif

using namespace Catlass;
using namespace tla;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;
using ElementBias = CATLASS_JIT_ELEMENT_BIAS;

using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = layout::CATLASS_JIT_LAYOUT_C;

using ArchTag = Arch::Ascend950;
constexpr bool useHF32 = false;
constexpr uint32_t l0CStages = 1;
constexpr uint32_t l1Stages = 2;
constexpr uint32_t l0AStages = 2;
constexpr uint32_t l0BStages = 2;
using DispatchPolicy = Gemm::MmadIterBatchBroadcast<ArchTag, useHF32, l0CStages, l1Stages, l0AStages, l0BStages>;

// Phase-1 small-matrix path: the full A/B matrices of one L1 group reside in L1
// (m <= L1_TILE_M, n <= L1_TILE_N, k <= L1_TILE_K); only K is bricked on L0.
// The K tile scales with the element byte width so the L0 byte budget is dtype-agnostic.
using BaseL1 = tuple<C<128>, C<128>, C<128>>;
using BaseL0 = tuple<C<128>, C<128>, C<32>>;
using L1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL1>::type;
using L0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL0>::type;

using TileCopy =
    Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias>;
using BlockMmad = Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
using BlockEpilogue = void;
using BlockScheduler = Gemm::Block::IterBatchScheduler;

using MatmulKernel = Gemm::Kernel::IterBatchBroadcastMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;

extern "C" void run(
    uint32_t blockNum, aclrtStream stream, const CatlassKernel::IterBatchBroadcastMatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t batchCount = params->batch;
    // A broadcast operand is a single matrix shared by all output batches.
    uint32_t batchA = params->broadcastA ? 1 : batchCount;
    uint32_t batchB = params->broadcastB ? 1 : batchCount;
    auto biasMode = static_cast<Gemm::BiasMode>(params->biasMode);

    auto layoutA = MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = MakeLayout<ElementC, LayoutTagC>(m, n);

    typename MatmulKernel::Arguments arguments{
        batchCount,
        batchA,
        batchB,
        GemmCoord{m, n, k},
        params->inputAddr[0], layoutA,
        params->inputAddr[1], layoutB,
        params->outputAddr[0], layoutC,
        (params->inputAddr.size() > 2) ? params->inputAddr[2] : nullptr,
        biasMode,
        0, 0, blockNum};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
