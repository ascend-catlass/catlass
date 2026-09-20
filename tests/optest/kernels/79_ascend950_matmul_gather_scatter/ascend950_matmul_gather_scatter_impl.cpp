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

#include <cstdint>
#include "catlass_kernel.h"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/block/block_epilogue_matmul_gather_scatter.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_gather_scatter_full_load.hpp"
#include "catlass/gemm/kernel/matmul_gather_scatter_simt.hpp"
#include "catlass/gemm/tile/matmul_gather_scatter_tile_copy.h"
#include "catlass/layout/layout.hpp"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler_tla.h"
#include "tla/layout.hpp"
#include "tla/tuple.hpp"

using namespace Catlass;

namespace CatlassKernel::MgsExample {

constexpr uint32_t FULL_LOAD_A_MAX_K = 2048;
using ArchTag = Arch::Ascend950;

constexpr bool ENABLE_UNIT_FLAG = true;
constexpr bool USE_HF32_MODE = false;
constexpr uint32_t L0C_STAGES = 1;
constexpr bool ENABLE_L1_RESIDENT = false;
constexpr uint32_t L1A_STAGES = 1;
constexpr uint32_t L0A_STAGES = 2;
constexpr uint32_t L0B_STAGES = 2;
constexpr uint32_t OUTPUT_UB_STAGES = 1;
constexpr uint32_t L1B_STAGES = 2;

// Match the experimental example: one FullLoad profile for each Gather mode.
using BaseL1TileM128N256 = tla::tuple<tla::Int<128>, tla::Int<256>, tla::Int<128>>;
using BaseL0TileM128N256 = tla::tuple<tla::Int<128>, tla::Int<256>, tla::Int<64>>;
using BaseL1TileM96N256 = tla::tuple<tla::Int<96>, tla::Int<256>, tla::Int<128>>;
using BaseL0TileM96N256 = tla::tuple<tla::Int<96>, tla::Int<256>, tla::Int<64>>;

using L1TileM128N256 = typename TileShapeScalerTLA<half, half, BaseL1TileM128N256>::type;
using L0TileM128N256 = typename TileShapeScalerTLA<half, half, BaseL0TileM128N256>::type;
using L1TileM96N256 = typename TileShapeScalerTLA<half, half, BaseL1TileM96N256>::type;
using L0TileM96N256 = typename TileShapeScalerTLA<half, half, BaseL0TileM96N256>::type;

struct AivGatherKernelMode {
    static constexpr bool AIV_GATHER = true;

    template <class BlockMmad, class BlockEpilogue, class BlockScheduler>
    using Kernel = Gemm::Kernel::MatmulGatherScatterFullLoadAivGather<BlockMmad, BlockEpilogue, BlockScheduler>;
};

struct AicGatherKernelMode {
    static constexpr bool AIV_GATHER = false;

    template <class BlockMmad, class BlockEpilogue, class BlockScheduler>
    using Kernel = Gemm::Kernel::MatmulGatherScatterFullLoadAicGather<BlockMmad, BlockEpilogue, BlockScheduler>;
};

template <class Kernel>
void LaunchKernel(uint32_t aicCoreNum, aclrtStream stream, const MatmulGatherScatterParams* params)
{
    typename Kernel::Arguments arguments{
        GemmCoord{params->j, params->n, params->k},
        params->m,
        params->inputAddr[0],
        params->inputAddr[1],
        params->inputAddr[2],
        params->outputAddr[0],
        aicCoreNum};
    Catlass::RunKernel<Kernel>(arguments, stream, aicCoreNum);
}

template <class L1TileShape, class L0TileShape, class GatherKernelMode>
void LaunchMatmulGatherScatter(uint32_t aicCoreNum, aclrtStream stream, const MatmulGatherScatterParams* params)
{
    using MmadDispatchPolicy = Gemm::MmadAscend950FullLoadA<
        ArchTag, ENABLE_UNIT_FLAG, USE_HF32_MODE, L0C_STAGES, ENABLE_L1_RESIDENT, L1A_STAGES, L1B_STAGES, L0A_STAGES,
        L0B_STAGES>;
    using GatherScatterDispatchPolicy = Gemm::MatmulGatherScatterDispatchPolicy<
        ArchTag, OUTPUT_UB_STAGES, GatherKernelMode::AIV_GATHER, Gemm::Kernel::detail::MGS_AIV_GATHER_MAX_K>;
    using TileCopy = Gemm::Tile::MatmulGatherScatterTileCopy<
        ArchTag, half, layout::RowMajor, half, layout::RowMajor, float, layout::RowMajor>;
    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, half, typename TileCopy::LayoutTagL1A>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, half, half, float, void, TileCopy, TileMmad>;
    using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzleL1FullLoad<>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogueMatmulGatherScatter<BlockMmad, GatherScatterDispatchPolicy>;
    using Kernel = typename GatherKernelMode::template Kernel<BlockMmad, BlockEpilogue, BlockScheduler>;
    LaunchKernel<Kernel>(aicCoreNum, stream, params);
}

void RunByShape(uint32_t aicCoreNum, aclrtStream stream, const MatmulGatherScatterParams* params)
{
    if ((params->k % Gemm::Kernel::detail::MGS_FP16_ELEMENTS_PER_BLOCK) != 0U || params->k > FULL_LOAD_A_MAX_K) {
        LaunchKernel<Gemm::Kernel::MatmulGatherScatterSimt>(aicCoreNum, stream, params);
        return;
    }

    if (params->k <= Gemm::Kernel::detail::MGS_AIV_GATHER_MAX_K) {
        LaunchMatmulGatherScatter<L1TileM128N256, L0TileM128N256, AivGatherKernelMode>(aicCoreNum, stream, params);
        return;
    }

    LaunchMatmulGatherScatter<L1TileM96N256, L0TileM96N256, AicGatherKernelMode>(aicCoreNum, stream, params);
}

} // namespace CatlassKernel::MgsExample

extern "C" void run(uint32_t aicCoreNum, aclrtStream stream, const CatlassKernel::MatmulGatherScatterParams* params)
{
    CatlassKernel::MgsExample::RunByShape(aicCoreNum, stream, params);
}
