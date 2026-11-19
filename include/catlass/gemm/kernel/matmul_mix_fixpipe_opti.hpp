/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_MIX_FIXPIPE_OPTI_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_MIX_FIXPIPE_OPTI_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class KernelMatmulMixFixpipeOpti {
public:
    CATLASS_DEVICE
    KernelMatmulMixFixpipeOpti()
    {
        // Init UbTensor
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubTensorList_[i] = resource.ubBuf.template GetBufferByByte<ElementC>(UB_TILE_SIZE * i);
        }
        if ASCEND_IS_AIV {
            for (uint32_t i = 0; i < UB_STAGES; ++i) {
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + i);
            }
        }
    }
    CATLASS_DEVICE
    ~KernelMatmulMixFixpipeOpti()
    {
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < UB_STAGES; ++i) {
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + i);
                if constexpr (USE_DUAL_DST_FLAGS) {
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX + i);
                }
            }
        }
    }

    using BlockMmad = BlockMmad_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using LayoutTagC = typename BlockMmad::TileCopy::LayoutTagC;
    using L1TileShape = typename BlockMmad::L1TileShape;

    AscendC::GlobalTensor<ElementA> aGlobal_;
    AscendC::GlobalTensor<ElementB> bGlobal_;
    AscendC::GlobalTensor<ElementC> cGlobal_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    static constexpr uint32_t UB_STAGES = BlockEpilogue::UB_STAGES;
    constexpr static bool ENABLE_DUAL_DST = AscendC::IsSameType<ElementC, ElementAccumulator>::value;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    // Ascend960 AIC:AIV = 1:1，无 SPLIT_M 双目的端：flags 关闭，UB 行取全宽
    constexpr static bool USE_DUAL_DST_FLAGS = false;
    constexpr static uint32_t UB_TILE_ROWS = L1_TILE_M;
#else
    constexpr static bool USE_DUAL_DST_FLAGS = ENABLE_DUAL_DST;
    constexpr static uint32_t UB_TILE_ROWS = ENABLE_DUAL_DST ? CeilDiv(L1_TILE_M, 2U) : L1_TILE_M;
#endif
    constexpr static uint32_t UB_TILE_COL_BYTES = RoundUp(L1_TILE_N * sizeof(ElementC), BYTE_PER_BLK);
    constexpr static uint32_t UB_TILE_SIZE = UB_TILE_ROWS * UB_TILE_COL_BYTES;

    static_assert(UB_TILE_SIZE * UB_STAGES <= ArchTag::UB_SIZE, "The UB tile SIZE exceeds the UB space");

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        Arguments() = default;
    };

    struct Params {
        GemmCoord problemShape;
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        Params() = default;
    };

    CATLASS_DEVICE
    void operator()(Params const& params)
    {
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        m_ = params.problemShape.m();
        n_ = params.problemShape.n();
        k_ = params.problemShape.k();

        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.aGmAddr), m_ * k_);
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.bGmAddr), k_ * n_);
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(params.cGmAddr), m_ * n_);

        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m_, k_);
        auto aTlaTensor = tla::MakeTensor(aGlobal_, layoutA, Arch::PositionGM{});

        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k_, n_);
        auto bTlaTensor = tla::MakeTensor(bGlobal_, layoutB, Arch::PositionGM{});

        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m_, n_);
        auto cTlaTensor = tla::MakeTensor(cGlobal_, layoutC, Arch::PositionGM{});

        // dualDstCtrl is not supported in quant/dequant scenarios
        if ASCEND_IS_AIV {
            if (!ENABLE_DUAL_DST && AscendC::GetSubBlockIdx() > 0) {
                return;
            }
            curBlockIdx /= AscendC::GetTaskRation();
        }
        // Instantiate
        BlockMmad blockMmadOp(resource);
        BlockEpilogue epilogueOp;
        BlockScheduler bs(curBlockIdx, blockNum, params.problemShape);

        if (bs.endBlockIdx_ + 1 <= blockNum / 2) {
            bs.UpdateTailTile();
        }

        uint32_t coreLoops = bs.round_;
        for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
            bool isLastLoop = (loopIdx == coreLoops - 1 && curBlockIdx <= bs.endBlockIdx_);
            bs.UpdateMNTileIdx(loopIdx, isLastLoop);
            bs.UpdateBlockShape(loopIdx, isLastLoop);

            auto blockShape = bs.GetBlockShape();
            auto blkElemCoord = bs.GetBlockCoordByElement();

            uint32_t blockM = blockShape.m();
            uint32_t blockN = blockShape.n();

            uint32_t mCoord = blkElemCoord.m();
            uint32_t nCoord = blkElemCoord.n();

            auto aTileTensor = GetTile(aTlaTensor, tla::MakeCoord(mCoord, 0), tla::MakeShape(blockM, k_));

            auto bTileTensor = GetTile(bTlaTensor, tla::MakeCoord(0, nCoord), tla::MakeShape(k_, blockN));

            auto cTileTensor = GetTile(cTlaTensor, tla::MakeCoord(mCoord, nCoord), tla::MakeShape(blockM, blockN));

            uint32_t ubListId = loopIdx % UB_STAGES;

            int64_t alignN = RoundUp(blockN, static_cast<int64_t>(Catlass::BYTE_PER_BLK / sizeof(ElementC)));
            auto ubLayout = tla::MakeLayout<ElementC, LayoutTagC>(blockM, alignN);
            auto cLocalTensor = tla::MakeTensor(ubTensorList_[ubListId], ubLayout, Arch::PositionUB{});

            if ASCEND_IS_AIC {
                // Synchronize with aiv
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + (ubListId));
                if constexpr (USE_DUAL_DST_FLAGS) {
#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ != 9201 && __NPU_ARCH__ != 9202)
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIV_SYNC_AIC_FLAG + (ubListId) + FLAG_ID_MAX);
#endif
                }
                // Calulate blockMmad
                blockMmadOp(aTileTensor, bTileTensor, cLocalTensor, blockShape);
                // Notify aiv
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + (ubListId));
                if constexpr (USE_DUAL_DST_FLAGS) {
#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ != 9201 && __NPU_ARCH__ != 9202)
                    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIC_SYNC_AIV_FLAG + (ubListId) + FLAG_ID_MAX);
#endif
                }
            }
            if ASCEND_IS_AIV {
                // Synchronize with aic
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + (ubListId));
                // Calulate epilogue
                epilogueOp(cTileTensor, cLocalTensor);
                // Notify aic
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + (ubListId));
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_HOST_DEVICE
    static bool CanImplement(Arguments const& args)
    {
        return true;
    }

    CATLASS_HOST_DEVICE
    static size_t GetWorkspaceSize(const Arguments& args)
    {
        return 0;
    }

    CATLASS_HOST_DEVICE
    static Params ToUnderlyingArguments(Arguments const& args, GM_ADDR workspace)
    {
        return {args.problemShape, args.aGmAddr, args.bGmAddr, args.cGmAddr};
    }

private:
    Arch::Resource<ArchTag> resource;
    AscendC::LocalTensor<ElementC> ubTensorList_[UB_STAGES];
    constexpr static uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;
    constexpr static int16_t FLAG_ID_MAX = 16;
    int64_t m_{1};
    int64_t n_{1};
    int64_t k_{1};
};

} // namespace Catlass::Gemm::Kernel
#endif // CATLASS_GEMM_KERNEL_MATMUL_MIX_FIXPIPE_OPTI_HPP
