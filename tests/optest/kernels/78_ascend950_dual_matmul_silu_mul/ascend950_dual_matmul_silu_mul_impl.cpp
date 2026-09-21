/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A half
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B half
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C float
#endif
#ifndef CATLASS_JIT_ELEMENT_D
#define CATLASS_JIT_ELEMENT_D half
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B ColumnMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_D
#define CATLASS_JIT_LAYOUT_D RowMajor
#endif
#ifndef CATLASS_JIT_DUAL_MATMUL_FULL_LOAD_A
#define CATLASS_JIT_DUAL_MATMUL_FULL_LOAD_A 0
#endif
#ifndef CATLASS_JIT_DUAL_MATMUL_M_TILE
#define CATLASS_JIT_DUAL_MATMUL_M_TILE 128
#endif
#ifndef CATLASS_JIT_DUAL_MATMUL_N_TILE
#define CATLASS_JIT_DUAL_MATMUL_N_TILE 128
#endif
#ifndef CATLASS_JIT_DUAL_MATMUL_L0_N_TILE
#define CATLASS_JIT_DUAL_MATMUL_L0_N_TILE 128
#endif
#ifndef CATLASS_JIT_DUAL_MATMUL_L1_K_TILE
#define CATLASS_JIT_DUAL_MATMUL_L1_K_TILE 128
#endif
#ifndef CATLASS_JIT_DUAL_MATMUL_L0_K_TILE
#define CATLASS_JIT_DUAL_MATMUL_L0_K_TILE 64
#endif
#ifndef CATLASS_JIT_DUAL_MATMUL_L1A_STAGES
#define CATLASS_JIT_DUAL_MATMUL_L1A_STAGES 2
#endif
#ifndef CATLASS_JIT_DUAL_MATMUL_L1B_STAGES
#define CATLASS_JIT_DUAL_MATMUL_L1B_STAGES 2
#endif
#ifndef CATLASS_JIT_DUAL_MATMUL_L0A_STAGES
#define CATLASS_JIT_DUAL_MATMUL_L0A_STAGES 2
#endif
#ifndef CATLASS_JIT_DUAL_MATMUL_L0B_STAGES
#define CATLASS_JIT_DUAL_MATMUL_L0B_STAGES 2
#endif
#ifndef CATLASS_JIT_BLOCK_SCHEDULER
#define CATLASS_JIT_BLOCK_SCHEDULER 30
#endif

#include <algorithm>

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#include "catlass/epilogue/block/block_epilogue_dual_silu_mul.h"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_mmad_dual_shared_a.h"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/kernel/dual_matmul_silu_mul_tla.h"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"

using namespace Catlass;
using namespace tla;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;
using ElementD = CATLASS_JIT_ELEMENT_D;
using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = layout::RowMajor;
using LayoutTagD = layout::CATLASS_JIT_LAYOUT_D;
using ArchTag = Arch::Ascend950;

#if CATLASS_JIT_DUAL_MATMUL_FULL_LOAD_A
using BaseDispatchPolicy = Gemm::MmadAscend950FullLoadA<
    ArchTag, true, false, 1, false, CATLASS_JIT_DUAL_MATMUL_L1A_STAGES, CATLASS_JIT_DUAL_MATMUL_L1B_STAGES,
    CATLASS_JIT_DUAL_MATMUL_L0A_STAGES, CATLASS_JIT_DUAL_MATMUL_L0B_STAGES>;
#else
using BaseDispatchPolicy = Gemm::MmadPingpong<
    ArchTag, true, false, 1, false, CATLASS_JIT_DUAL_MATMUL_L1A_STAGES, CATLASS_JIT_DUAL_MATMUL_L1B_STAGES,
    CATLASS_JIT_DUAL_MATMUL_L0A_STAGES, CATLASS_JIT_DUAL_MATMUL_L0B_STAGES>;
#endif
using DispatchPolicy = Gemm::MmadDualSharedA<BaseDispatchPolicy>;
using L1TileShape = Shape<
    Int<CATLASS_JIT_DUAL_MATMUL_M_TILE>, Int<CATLASS_JIT_DUAL_MATMUL_N_TILE>, Int<CATLASS_JIT_DUAL_MATMUL_L1_K_TILE>>;
using L0TileShape = Shape<
    Int<CATLASS_JIT_DUAL_MATMUL_M_TILE>, Int<CATLASS_JIT_DUAL_MATMUL_L0_N_TILE>,
    Int<CATLASS_JIT_DUAL_MATMUL_L0_K_TILE>>;
using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
constexpr auto copyMode = std::is_same_v<ElementC, ElementAccumulator> ? Gemm::Tile::CopyL0CToUBMode::SPLIT_M :
                                                                         Gemm::Tile::CopyL0CToUBMode::NO_SPLIT;
using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, void, copyMode>;
using BlockMmad =
    Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
using BlockEpilogue = Epilogue::Block::BlockEpilogueDualSiluMul<
    ElementD, ElementC, (CATLASS_JIT_DUAL_MATMUL_M_TILE + 1) / 2 * ((CATLASS_JIT_DUAL_MATMUL_N_TILE + 31) / 32 * 32)>;

#if CATLASS_JIT_BLOCK_SCHEDULER == 10
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzleL1FullLoad<1, 0>;
#elif CATLASS_JIT_BLOCK_SCHEDULER == 100
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<1, 0>;
#elif CATLASS_JIT_BLOCK_SCHEDULER == 101
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<1, 1>;
#else
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<
    (CATLASS_JIT_BLOCK_SCHEDULER / 10), (CATLASS_JIT_BLOCK_SCHEDULER % 10)>;
#endif

using DualMatmulKernel = Gemm::Kernel::DualMatmulSiluMulTla<
    BlockMmad, BlockEpilogue, BlockScheduler, ElementD, decltype(MakeLayout<ElementD, LayoutTagD>(1U, 1U))>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    auto layoutA = MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutD = MakeLayout<ElementD, LayoutTagD>(m, n);
    typename DualMatmulKernel::Arguments arguments{
        GemmCoord{m, n, k},
        params->inputAddr[0],
        layoutA,
        params->inputAddr[1],
        params->inputAddr[2],
        layoutB,
        params->outputAddr[0],
        layoutD};
    uint32_t taskNum = CeilDiv(m, DualMatmulKernel::L1_TILE_M) * CeilDiv(n, DualMatmulKernel::OUTPUT_TILE_N);
    Catlass::RunKernel<DualMatmulKernel>(arguments, stream, std::min(blockNum, taskNum));
}
