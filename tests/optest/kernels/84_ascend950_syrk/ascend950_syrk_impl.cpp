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

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue_syrk_axpby.hpp"
#include "catlass/gemm/block/block_mmad_syrk_tla.hpp"
#include "catlass/gemm/block/block_scheduler_syrk.h"
#include "catlass/gemm/kernel/basic_syrk_workspace_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A bfloat16_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B bfloat16_t
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C bfloat16_t
#endif
// RowMajor: X is stored [M, K] row-major (BLAS SYRK trans = N).
// ColumnMajor: X is a transposed view of a row-major [K, M] buffer (BLAS SYRK trans = T).
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif

using namespace Catlass;
using namespace tla;

using ElementX = CATLASS_JIT_ELEMENT_A;
using ElementY = CATLASS_JIT_ELEMENT_B; // residual Y, same type as X
using ElementD = CATLASS_JIT_ELEMENT_C; // output D, either ElementY or float
using ElementP = float;                 // workspace tile dtype

// L1/L0 tiles are sized for 2-byte operands; wider ones scale K down to stay in L0A/L1.
static constexpr uint32_t K_SCALE = sizeof(ElementX) / sizeof(half);
using L1TileShape = tuple<C<256>, C<256>, C<128 / K_SCALE>>;
using L0TileShape = tuple<C<256>, C<256>, C<64 / K_SCALE>>;

using LayoutTagX = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagXt =
    std::conditional_t<std::is_same_v<LayoutTagX, layout::RowMajor>, layout::ColumnMajor, layout::RowMajor>;
using TileCopy = Gemm::Tile::PackedTileCopyTla<
    Arch::Ascend950, ElementX, LayoutTagX, ElementX, LayoutTagXt, ElementP, layout::RowMajor>;

// BlockMmad dual-writes float P / P^T into GM workspace.
using BlockMmad = Gemm::Block::BlockMmadSyrkTla<L1TileShape, L0TileShape, ElementX, ElementP, TileCopy>;
using BlockEpilogue = Epilogue::Block::BlockEpilogueSyrkAxpby<Arch::Ascend950, ElementY, ElementD>;

using BlockScheduler = Gemm::Block::SyrkBlockScheduler;

using MatmulKernel = Gemm::Kernel::BasicSyrkWorkspaceTla<BlockMmad, BlockEpilogue, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::SyrkParams* params)
{
    // Only lower-triangle tiles exist; extra cores would idle and still claim workspace.
    BlockScheduler scheduler(
        GemmCoord{params->m, params->n, params->k}, MakeCoord(MatmulKernel::L1_TILE_M, MatmulKernel::L1_TILE_N));
    const uint32_t tasks = params->batch * scheduler.GetCoreLoops();
    const uint32_t coreNum = (tasks < blockNum) ? tasks : blockNum;

    typename MatmulKernel::Arguments arguments{params->batch,         GemmCoord{params->m, params->n, params->k},
                                               params->inputAddr[0],  params->inputAddr[1],
                                               params->outputAddr[0], params->alpha,
                                               params->beta,          coreNum};
    Catlass::RunKernel<MatmulKernel>(arguments, stream, coreNum);
}
