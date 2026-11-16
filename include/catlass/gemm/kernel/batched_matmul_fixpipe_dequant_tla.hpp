/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_BATCHED_MATMUL_FIXPIPE_DEQUANT_TLA_HPP
#define CATLASS_GEMM_KERNEL_BATCHED_MATMUL_FIXPIPE_DEQUANT_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

// Template for batched matmul with Fixpipe dequant. Compute batched C = dequant(A * B + bias).
//
// BlockMmad_ must use the MmadDequant dispatch policy, so the L0C->GM copy runs through Fixpipe
// with a dequant quant-mode (e.g. QF322BF16_PRE / VQF322BF16_PRE):
// - per-tensor scale is carried as a float and programmed into Fixpipe deqScalar;
// - per-channel scales are carried as a GM uint64_t tensor (one encoded word per N column);
// - bias is carried as a GM tensor and staged through BT, added before dequant inside Fixpipe.
// Pattern follows StridedBatchedMatmulTla: batch is expressed by a 3-rank (batch, row, col) TLA
// layout, so no batch stride is passed around explicitly.
template <class BlockMmad_, class BlockScheduler_>
class BatchedMatmulFixpipeDequantTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = tla::MakeBatchedLayout_t<typename BlockMmad::LayoutA>;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = tla::MakeBatchedLayout_t<typename BlockMmad::LayoutB>;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = tla::MakeBatchedLayout_t<typename BlockMmad::LayoutC>;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementBias = typename BlockMmad::ElementBias;
    using LayoutBias = decltype(tla::MakeLayout(std::declval<uint32_t>()));

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        float perTensorScale;
        GM_ADDR ptrPerChannelScale;
        GM_ADDR ptrBias;
        LayoutBias layoutBias;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            uint32_t batchCount_, GemmCoord const& problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_,
            LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_, float perTensorScale_, GM_ADDR ptrPerChannelScale_,
            GM_ADDR ptrBias_, LayoutBias layoutBias_)
            : batchCount(batchCount_),
              problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              perTensorScale(perTensorScale_),
              ptrPerChannelScale(ptrPerChannelScale_),
              ptrBias(ptrBias_),
              layoutBias(layoutBias_)
        {}
    };

    struct Arguments {
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        float perTensorScale;
        GM_ADDR ptrPerChannelScale;
        GM_ADDR ptrBias;
        LayoutBias layoutBias;
    };

    static bool CanImplement(const Arguments& args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        Params params{args.batchCount, args.problemShape, args.ptrA,    args.layoutA,        args.ptrB,
                      args.layoutB,    args.ptrC,         args.layoutC, args.perTensorScale, args.ptrPerChannelScale,
                      args.ptrBias,    args.layoutBias};
        return params;
    }

    CATLASS_DEVICE
    BatchedMatmulFixpipeDequantTla()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MatrixCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = params.batchCount * matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);
        AscendC::GlobalTensor<uint64_t> gmScale;
        gmScale.SetGlobalBuffer((__gm__ uint64_t*)params.ptrPerChannelScale);
        AscendC::GlobalTensor<ElementBias> gmBias;
        gmBias.SetGlobalBuffer((__gm__ ElementBias*)params.ptrBias);

        uint32_t n = params.problemShape.n();
        auto batchedTensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto batchedTensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto batchedTensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        auto layoutQuant = tla::MakeLayout(n);
        auto tensorQuant = tla::MakeTensor(gmScale, layoutQuant, Arch::PositionGM{});
        auto tensorBias = tla::MakeTensor(gmBias, params.layoutBias, Arch::PositionGM{});

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            uint32_t batchIdx = matmulBlockScheduler.GetBatchIdx(loopIdx);
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            auto tensorA = batchedTensorA(batchIdx, tla::_, tla::_);
            auto tensorB = batchedTensorB(batchIdx, tla::_, tla::_);
            auto tensorC = batchedTensorC(batchIdx, tla::_, tla::_);

            auto tensorBlockA = GetTile(
                tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tensorBlockB = GetTile(
                tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
            auto tensorBlockQuant =
                GetTile(tensorQuant, tla::MakeCoord(blockCoord.n() * L1_TILE_N), tla::MakeShape(actualBlockShape.n()));

            auto tensorBlockBias =
                GetTile(tensorBias, tla::MakeCoord(blockCoord.n() * L1_TILE_N), tla::MakeShape(actualBlockShape.n()));
            blockMmad(
                tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockBias, tensorBlockQuant,
                params.perTensorScale);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {}
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_BATCHED_MATMUL_FIXPIPE_DEQUANT_TLA_HPP
