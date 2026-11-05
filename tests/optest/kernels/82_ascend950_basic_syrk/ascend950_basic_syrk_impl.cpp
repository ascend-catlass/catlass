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
#include "catlass/gemm/block/block_mmad_syrk_tla.hpp"
#include "catlass/gemm/block/block_scheduler_syrk.h"
#include "catlass/gemm/kernel/basic_syrk_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"

// Device/kernel types are fixed by BlockMmadSyrkTla; only element dtypes are JIT-configurable.
#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A bfloat16_t
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C bfloat16_t
#endif

using namespace Catlass;
using namespace tla;

using ElementX = CATLASS_JIT_ELEMENT_A;
using ElementY = CATLASS_JIT_ELEMENT_C;

using L1TileShape = tuple<C<256>, C<256>, C<128>>;
using L0TileShape = tuple<C<256>, C<256>, C<64>>;

using BlockMmad = Gemm::Block::BlockMmadSyrkTla<L1TileShape, L0TileShape, ElementX, ElementY>;
using BlockEpilogue = void;

using BlockScheduler = Gemm::Block::SyrkBlockScheduler;

using MatmulKernel = Gemm::Kernel::BasicSyrkTla<BlockMmad, BlockEpilogue, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    typename MatmulKernel::Arguments arguments{
        1, GemmCoord{params->m, params->n, params->k}, params->inputAddr[0], params->inputAddr[0],
        params->outputAddr[0]};
    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
