/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cstddef>
using std::size_t;

#include <kernel_operator.h>

#include "catlass/gemm/kernel/a8w4_mx_matmul.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A float8_e4m3_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B float4_e2m1x2_t
#endif
#ifndef CATLASS_JIT_ELEMENT_MX_SCALE
#define CATLASS_JIT_ELEMENT_MX_SCALE float8_e8m0_t
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B ColumnMajor
#endif

#define CATLASS_JIT_ELEMENT_C float

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementPrologueB = CATLASS_JIT_ELEMENT_B;
using ElementB = float8_e4m3_t;
using ElementMxScale = CATLASS_JIT_ELEMENT_MX_SCALE;
using ElementC = CATLASS_JIT_ELEMENT_C;
using ElementBias = void;

using LayoutTagA = Catlass::layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagPrologueB = Catlass::layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = Catlass::layout::RowMajor;
using LayoutTagB = Catlass::layout::nZ;

using ArchTag = Catlass::Arch::Ascend950;
constexpr bool enableUnitFlag = true;
static constexpr uint32_t L1_SCALE_FACTOR_K = 16;
static constexpr uint32_t L1A_STAGES = 2;
static constexpr uint32_t L1B_STAGES = 2;
static constexpr uint32_t L0A_STAGES = 2;
static constexpr uint32_t L0B_STAGES = 2;
static constexpr uint32_t L0C_STAGES = 1;

using L1TileShape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
using L0TileShape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;

using PrologueSrcType = Catlass::Gemm::GemmType<ElementPrologueB, LayoutTagPrologueB>;
using PrologueDstType = Catlass::Gemm::GemmType<ElementB, LayoutTagB>;

using DispatchPolicyMmad = Catlass::Gemm::MmadA8W4Mx<
    ArchTag, enableUnitFlag, false, L1_SCALE_FACTOR_K, L0C_STAGES, L1A_STAGES, L1B_STAGES, L0A_STAGES, L0B_STAGES>;
using DispatchPolicyPrologue = Catlass::Gemm::MxA8W4Prologue<ArchTag, L1B_STAGES>;

#ifndef CATLASS_JIT_BLOCK_SCHEDULER
#define CATLASS_JIT_BLOCK_SCHEDULER 30
#endif
using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<
    (CATLASS_JIT_BLOCK_SCHEDULER / 10), (CATLASS_JIT_BLOCK_SCHEDULER % 10)>;

using BlockEpilogue = void;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(k);

    Catlass::GemmCoord problemShape{m, n, k};

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* devicePrologueB = params->inputAddr[1];
    uint8_t* deviceMxScaleA = params->inputAddr[2];
    uint8_t* deviceMxScaleB = params->inputAddr[3];
    uint8_t* deviceC = params->outputAddr[0];

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutPrologueB = tla::MakeLayout<ElementPrologueB, LayoutTagPrologueB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagPrologueB, true>(mxScaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopy = Catlass::Gemm::Tile::PackedMxA8W4TileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementPrologueB, LayoutTagPrologueB, ElementB, LayoutTagB, ElementMxScale,
        decltype(layoutMxScaleA), ElementMxScale, decltype(layoutMxScaleB), ElementC, LayoutTagC, ElementBias, false,
        Catlass::Gemm::Tile::ScaleGranularity::PER_TENSOR>;

    using BlockMmad = Catlass::Gemm::Block::BlockMmadA8W4Mx<
        DispatchPolicyMmad, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;

    using BlockPrologue = Catlass::Gemm::Block::BlockPrologue<
        DispatchPolicyPrologue, PrologueSrcType, PrologueDstType, L1TileShape, TileCopy>;

    using MatmulKernel = Catlass::Gemm::Kernel::A8W4MxMatmul<BlockMmad, BlockPrologue, BlockEpilogue, BlockScheduler>;

    typename MatmulKernel::Arguments arguments{
        problemShape, deviceA, layoutA, devicePrologueB, layoutPrologueB, deviceMxScaleA, layoutMxScaleA,
        deviceMxScaleB, layoutMxScaleB, deviceC, layoutC, nullptr};

    uint64_t taskNum64 = static_cast<uint64_t>(CeilDiv(m, tla::get<0>(L1TileShape{}))) *
                         static_cast<uint64_t>(CeilDiv(n, tla::get<1>(L1TileShape{})));
    uint32_t taskNum = static_cast<uint32_t>(std::min(taskNum64, static_cast<uint64_t>(UINT32_MAX)));
    uint32_t aicCoreUsed = std::min(blockNum, taskNum);

    Catlass::RunKernel<MatmulKernel>(arguments, stream, aicCoreUsed);
}
