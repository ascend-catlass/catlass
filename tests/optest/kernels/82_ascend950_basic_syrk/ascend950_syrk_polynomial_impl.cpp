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

#include <type_traits>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad_syrk_tla.hpp"
#include "catlass/gemm/block/block_scheduler_syrk.h"
#include "catlass/gemm/kernel/basic_syrk_polynomial_splitk_tla.h"
#include "catlass/gemm/kernel/basic_syrk_polynomial_tla.h"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A bfloat16_t
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C bfloat16_t
#endif
// RowMajor: X is stored [M, K] row-major. ColumnMajor: X is a transposed view of a [K, M] buffer.
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif

using namespace Catlass;
using namespace tla;

using ElementX = CATLASS_JIT_ELEMENT_A;
using ElementY = CATLASS_JIT_ELEMENT_C;

// A 256x256 FP32 tile fills the 256 KiB L0C.
// Square L1/L0 output tile edge. Smaller tiles raise the tile count (and thus core
// occupancy) for small M at the cost of mmad efficiency. Must stay square: the mainloop
// and the mirrored store both assume L1_TILE_M == L1_TILE_N.
#ifdef CATLASS_JIT_SYRK_TILE
constexpr uint32_t SYRK_TILE_MN = CATLASS_JIT_SYRK_TILE;
#else
constexpr uint32_t SYRK_TILE_MN = 256;
#endif
// Keep ~constant bytes per L1 stage as the tile edge shrinks.
constexpr uint32_t SYRK_TILE_K1 = (SYRK_TILE_MN >= 256) ? 128 : ((SYRK_TILE_MN >= 128) ? 256 : 512);
using L1TileShape = tuple<C<SYRK_TILE_MN>, C<SYRK_TILE_MN>, C<SYRK_TILE_K1>>;
using L0TileShape = tuple<C<SYRK_TILE_MN>, C<SYRK_TILE_MN>, C<64>>;

using LayoutTagX = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagXt =
    std::conditional_t<std::is_same_v<LayoutTagX, layout::RowMajor>, layout::ColumnMajor, layout::RowMajor>;

// 0: single-pass kernel (L0C -> UB -> epilogue). 1: split-K (L0C -> FP32 workspace,
// device barrier, then reduce + epilogue on the AIVs).
#if defined(CATLASS_JIT_SYRK_SPLITK) && CATLASS_JIT_SYRK_SPLITK
// Split-K writes FP32 partial tiles to a GM workspace instead of handing L0C to the AIVs.
using TileCopy = Gemm::Tile::PackedTileCopyTla<
    Arch::Ascend950, ElementX, LayoutTagX, ElementX, LayoutTagXt, float, layout::RowMajor>;
#else
using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
    Arch::Ascend950, ElementX, LayoutTagX, ElementX, LayoutTagXt, float, layout::RowMajor, void,
    Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
#endif
using BlockMmad = Gemm::Block::BlockMmadSyrkTla<L1TileShape, L0TileShape, ElementX, ElementY, TileCopy>;

using BlockScheduler = Gemm::Block::SyrkBlockScheduler;
#if defined(CATLASS_JIT_SYRK_SPLITK) && CATLASS_JIT_SYRK_SPLITK
using Kernel = Gemm::Kernel::BasicSyrkPolynomialSplitkTla<BlockMmad, BlockScheduler>;
#else
using Kernel = Gemm::Kernel::BasicSyrkPolynomialTla<BlockMmad, BlockScheduler>;
#endif

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::SyrkPolynomialParams* params)
{
#if defined(CATLASS_JIT_SYRK_SPLITK) && CATLASS_JIT_SYRK_SPLITK
    typename Kernel::Arguments arguments{params->batch,        GemmCoord{params->m, params->n, params->k},
                                         params->inputAddr[0], params->inputAddr[1],
                                         params->inputAddr[2], params->outputAddr[0],
                                         params->inputAddr[3], // caller-owned FP32 partial-tile workspace
                                         params->alpha,        params->beta,
                                         params->diagonal,     params->splitk};
#else
    typename Kernel::Arguments arguments{params->batch,        GemmCoord{params->m, params->n, params->k},
                                         params->inputAddr[0], params->inputAddr[1],
                                         params->inputAddr[2], params->outputAddr[0],
                                         params->alpha,        params->beta,
                                         params->diagonal};
#endif
    Catlass::RunKernel<Kernel>(arguments, stream, blockNum);
}
