/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/block/block_epilogue_dequant.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_scheduler_aswt.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/matmul_full_dequant.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A int8_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B int8_t
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C int32_t
#endif
#ifndef CATLASS_JIT_ELEMENT_D
#define CATLASS_JIT_ELEMENT_D half
#endif
#ifndef CATLASS_JIT_ELEMENT_X1
#define CATLASS_JIT_ELEMENT_X1 float
#endif
#ifndef CATLASS_JIT_ELEMENT_X2
#define CATLASS_JIT_ELEMENT_X2 float
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
#ifndef CATLASS_JIT_X1_QUANT_MODE
#define CATLASS_JIT_X1_QUANT_MODE 4
#endif
#ifndef CATLASS_JIT_X2_QUANT_MODE
#define CATLASS_JIT_X2_QUANT_MODE 2
#endif
#ifndef CATLASS_JIT_HAS_QUANT_BIAS
#define CATLASS_JIT_HAS_QUANT_BIAS 0
#endif

using namespace Catlass;
using namespace tla;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;
using ElementD = CATLASS_JIT_ELEMENT_D;
using ElementX1 = CATLASS_JIT_ELEMENT_X1;
using ElementX2 = CATLASS_JIT_ELEMENT_X2;
#if CATLASS_JIT_HAS_QUANT_BIAS
using ElementBias = CATLASS_JIT_ELEMENT_BIAS;
#else
using ElementBias = void;
#endif

using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = layout::CATLASS_JIT_LAYOUT_C;

using QuantMode = Epilogue::QuantMode;

constexpr QuantMode kX1QuantMode = static_cast<QuantMode>(CATLASS_JIT_X1_QUANT_MODE);
constexpr QuantMode kX2QuantMode = static_cast<QuantMode>(CATLASS_JIT_X2_QUANT_MODE);
constexpr bool kHasQuantBias = CATLASS_JIT_HAS_QUANT_BIAS;
constexpr uint32_t kSplitTaskRatio = 2;

using ArchTag = Arch::Ascend950;
constexpr bool enableUnitFlag = true;
using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag>;

using L1TileShape = Shape<Int<128>, Int<256>, Int<256>>;
using L0TileShape = Shape<Int<128>, Int<256>, Int<128>>;

using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias,
    Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
using BlockMmad = Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
using EpilogueDispatchPolicy = Epilogue::BlockEpilogueDequant<kX1QuantMode, kX2QuantMode>;
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy, L0TileShape, ElementC, ElementD, ElementBias, ElementX1, ElementX2,
    std::integral_constant<uint32_t, kSplitTaskRatio>>;

using BlockScheduler = typename Gemm::Block::BlockSchedulerAswt<L1TileShape, L0TileShape>;

using MatmulKernel = Gemm::Kernel::GemmUniversal<
    BlockMmad, BlockEpilogue, BlockScheduler, LayoutTagA, LayoutTagB, LayoutTagC>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;

    uint8_t* x1ScaleDevice = nullptr;
    uint8_t* x2ScaleDevice = nullptr;
    uint8_t* quantBiasDevice = nullptr;

    int inputIdx = 2;
    if constexpr (kX1QuantMode != QuantMode::NONE) {
        x1ScaleDevice = params->inputAddr[inputIdx++];
    }
    x2ScaleDevice = params->inputAddr[inputIdx++];
    if constexpr (kHasQuantBias) {
        quantBiasDevice = params->inputAddr[inputIdx++];
    }

    typename MatmulKernel::Arguments arguments = {
        GemmCoord{m, n, k},
        params->inputAddr[0],
        params->inputAddr[1],
        params->outputAddr[0],
        x2ScaleDevice,
        x1ScaleDevice,
        quantBiasDevice};

    uint32_t taskNum = CeilDiv(m, tla::get<0>(L1TileShape{})) * CeilDiv(n, tla::get<1>(L1TileShape{}));
    uint32_t aicCoreUsed = std::min(blockNum, taskNum);

    Catlass::RunKernel<MatmulKernel>(arguments, stream, aicCoreUsed);
}
