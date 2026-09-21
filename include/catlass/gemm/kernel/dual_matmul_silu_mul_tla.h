#ifndef CATLASS_GEMM_KERNEL_DUAL_MATMUL_SILU_MUL_TLA_H
#define CATLASS_GEMM_KERNEL_DUAL_MATMUL_SILU_MUL_TLA_H

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

#include <algorithm>
#include <type_traits>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/epilogue/block/block_epilogue_dual_silu_mul.h"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_mmad_dual_shared_a.h"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
using namespace tla;

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class ElementD_, class LayoutD_>
class DualMatmulSiluMulTla {
public:
    using BlockMmad = BlockMmad_;
    using BlockScheduler = BlockScheduler_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;
    using TileCopy = typename BlockMmad::TileCopy;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementD = ElementD_;
    using LayoutD = LayoutD_;
    using LayoutTagL1A = typename BlockMmad::LayoutTagL1A;
    using LayoutTagL1B = typename BlockMmad::LayoutTagL1B;
    using LayoutTagL0A = typename BlockMmad::LayoutTagL0A;
    using LayoutTagL0B = typename BlockMmad::LayoutTagL0B;
    using CopyL1ToL0A = typename BlockMmad::CopyL1ToL0A;
    using BlockEpilogue = BlockEpilogue_;
    static constexpr uint32_t UB_RESULT_ELEMS = BlockEpilogue::BUFFER_ELEMENTS;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});
    static constexpr uint32_t L1A_STAGES = BlockMmad::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = BlockMmad::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = BlockMmad::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = BlockMmad::L0B_STAGES;
    static constexpr bool USE_FULL_LOAD_A = BlockMmad::DispatchPolicy::USE_FULL_LOAD_A;
    static constexpr uint32_t OUTPUT_TILE_N = L1_TILE_N;
    static constexpr bool ENABLE_DUAL_DST = BlockMmad::TileCopy::CopyMode == Gemm::Tile::CopyL0CToUBMode::SPLIT_M;
    static constexpr uint32_t AIV_SPLIT_M = ENABLE_DUAL_DST ? (L1_TILE_M + 1) / 2 : L1_TILE_M;
    static constexpr uint8_t AIC_SYNC_AIV_MODE = 4;
    static constexpr uint16_t AIV_FINISH_FLAG = 6;
    static constexpr uint16_t AIC_FINISH_FLAG = 8;
    static constexpr uint16_t AIC_FINISH_NOTIFY_AIV1_FLAG = AIC_FINISH_FLAG + 16;
    static constexpr uint16_t AIV1_FINISH_NOTIFY_AIC_FLAG = AIV_FINISH_FLAG + 16;
    static constexpr uint32_t UB_STRIDE_N = RoundUp<BYTE_PER_C0>(OUTPUT_TILE_N);
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);
    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});

    struct AivFinish {
        CATLASS_DEVICE void operator()() const
        {
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_FINISH_FLAG);
            if constexpr (ENABLE_DUAL_DST) {
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV1_FINISH_NOTIFY_AIC_FLAG);
            }
        }
    };

    struct AicFinish {
        CATLASS_DEVICE void operator()() const
        {
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_FINISH_FLAG);
            if constexpr (ENABLE_DUAL_DST) {
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_FINISH_NOTIFY_AIV1_FLAG);
            }
        }
    };

    static_assert(
        BlockMmad::DispatchPolicy::ENABLE_UNIT_FLAG, "dual mainloop currently assumes Ascend950 UnitFlag fixpipe mode");
    static_assert(L1A_STAGES + 2 * L1B_STAGES <= 8, "dual mainloop needs L1A events and two B-stream L1B event groups");
    static_assert(L0A_STAGES + 2 * L0B_STAGES <= 8, "dual mainloop needs two B-stream L0B event groups");
    static_assert(
        2 * L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "dual mainloop L0B staging exceeds Ascend950 L0B");
    static_assert(2 * L0C_TILE_SIZE <= ArchTag::L0C_SIZE, "dual mainloop needs two L0C accumulators");
    static_assert(L0_TILE_M == L1_TILE_M, "custom N-split mainloop requires identical L1/L0 M tiles");
    static_assert(L0_TILE_N <= L1_TILE_N, "L0 N tile must not exceed the L1 N tile");
    static_assert(L1_TILE_N % L0_TILE_N == 0, "L1 N tile must contain whole L0 N tiles");

    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB0;
        GM_ADDR ptrB1;
        LayoutB layoutB;
        GM_ADDR ptrD;
        LayoutD layoutD;
    };

    struct Arguments {
        GemmCoord problemShape;
        uint8_t* ptrA;
        LayoutA layoutA;
        uint8_t* ptrB0;
        uint8_t* ptrB1;
        LayoutB layoutB;
        uint8_t* ptrD;
        LayoutD layoutD;
    };

    static bool CanImplement(const Arguments& args)
    {
        uint32_t l1TileKForA = USE_FULL_LOAD_A ? args.problemShape.k() : L1_TILE_K * L1A_STAGES;
        constexpr uint32_t bStreams = 2;
        uint64_t l1UsedSpace = static_cast<uint64_t>(L1_TILE_M) * l1TileKForA * sizeof(ElementA) +
                               static_cast<uint64_t>(L1_TILE_K) * L1_TILE_N * L1B_STAGES * bStreams * sizeof(ElementB);
        uint64_t ubUsedSpace = 3ULL * UB_RESULT_ELEMS * sizeof(ElementC) + UB_RESULT_ELEMS * sizeof(ElementD);
        return l1UsedSpace <= ArchTag::L1_SIZE && ubUsedSpace <= ArchTag::UB_SIZE;
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        (void)args;
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        (void)workspace;
        return Params{args.problemShape, args.ptrA,    args.layoutA, args.ptrB0,
                      args.ptrB1,        args.layoutB, args.ptrD,    args.layoutD};
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler scheduler(params.problemShape, MakeCoord(L1_TILE_M, OUTPUT_TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB0;
        gmB0.SetGlobalBuffer((__gm__ ElementB*)params.ptrB0);
        AscendC::GlobalTensor<ElementB> gmB1;
        gmB1.SetGlobalBuffer((__gm__ ElementB*)params.ptrB1);

        if (CeilDiv(params.problemShape.m(), L1_TILE_M) == 1) {
            gmB0.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            gmB1.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (CeilDiv(params.problemShape.n(), OUTPUT_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB0 = tla::MakeTensor(gmB0, params.layoutB, Arch::PositionGM{});
        auto tensorB1 = tla::MakeTensor(gmB1, params.layoutB, Arch::PositionGM{});

        BlockMmad blockMmad;
        AivFinish aivFinish;
        AicFinish aicFinish;
        blockMmad(
            scheduler, coreLoops, resource, tensorA, tensorB0, tensorB1, MakeCallback(&aivFinish),
            MakeCallback(&aicFinish));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        BlockScheduler scheduler(params.problemShape, MakeCoord(L1_TILE_M, OUTPUT_TILE_N));
        uint32_t coreLoops = scheduler.GetCoreLoops();

        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer((__gm__ ElementD*)params.ptrD);

        auto ubD0 = BlockEpilogue::MakeInputTensor(0);
        auto ubD1 = BlockEpilogue::MakeInputTensor(1);
        auto ubOut = BlockEpilogue::MakeOutputTensor();
        auto ubCast = BlockEpilogue::MakeCastTensor();

        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        if constexpr (!ENABLE_DUAL_DST) {
            if (subBlockIdx > 0) {
                return;
            }
        }

        if constexpr (ENABLE_DUAL_DST) {
            if (subBlockIdx == 1) {
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_MTE3>(AIV1_FINISH_NOTIFY_AIC_FLAG);
            } else {
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_MTE3>(AIV_FINISH_FLAG);
            }
        } else {
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_MTE3>(AIV_FINISH_FLAG);
        }

        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = scheduler.GetActualBlockShape(blockCoord);

            if constexpr (ENABLE_DUAL_DST) {
                if (subBlockIdx == 1) {
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIC_FINISH_NOTIFY_AIV1_FLAG);
                } else {
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIC_FINISH_FLAG);
                }
            } else {
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIC_FINISH_FLAG);
            }

            uint32_t halfM = CeilDiv(actualBlockShape.m(), AscendC::GetTaskRation());
            uint32_t localM =
                ENABLE_DUAL_DST ? (subBlockIdx == 1 ? actualBlockShape.m() - halfM : halfM) : actualBlockShape.m();
            uint32_t mOffset = ENABLE_DUAL_DST ? subBlockIdx * halfM : 0;
            uint32_t strideN = RoundUp<BYTE_PER_C0>(actualBlockShape.n());

            if (localM > 0) {
                uint32_t computeElems = localM * strideN;

                uint64_t dstStride = static_cast<uint64_t>(tla::get<0>(params.layoutD.stride()));
                uint64_t dstOffset = (static_cast<uint64_t>(blockCoord.m()) * L1_TILE_M + mOffset) * dstStride +
                                     static_cast<uint64_t>(blockCoord.n()) * OUTPUT_TILE_N;
                BlockEpilogue{}(
                    ubOut, ubD0, ubD1, ubCast, gmD, dstOffset, localM, actualBlockShape.n(), strideN, dstStride,
                    computeElems);
            }

            if constexpr (ENABLE_DUAL_DST) {
                if (subBlockIdx == 1) {
                    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_MTE3>(AIV1_FINISH_NOTIFY_AIC_FLAG);
                } else {
                    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_MTE3>(AIV_FINISH_FLAG);
                }
            } else {
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_MTE3>(AIV_FINISH_FLAG);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_DUAL_MATMUL_SILU_MUL_TLA_H
