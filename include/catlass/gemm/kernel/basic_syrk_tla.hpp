/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_BASIC_SYRK_TLA_HPP
#define CATLASS_GEMM_KERNEL_BASIC_SYRK_TLA_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/matrix.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include <cstdint>

namespace Catlass::Gemm::Kernel {

/**
 * @brief Kernel-level SYRK: Y = X * X^T (Ascend950 only), batched over batchCount.
 *
 * X (M, K) and Xt (K, M) are the plain and transposed views of the same data, so ptrXt
 * normally equals ptrX; with distinct buffers the caller must guarantee a symmetric
 * result.
 *
 * Only the lower triangle is computed. loopIdx counts straight through the (batch,
 * tileRow, tileColumn) triples that have tileRow >= tileColumn, so a core never picks up
 * a tile of the upper triangle only to discard it again.
 * 1. blockCoord.m() == blockCoord.n() -> a diagonal tile: compute and write it once (nz2nd)
 * 2. blockCoord.m() >  blockCoord.n() -> compute once, then write the tile (nz2nd) and its
 *    transpose (nz2dn), which fills the matching tile of the upper triangle
 */
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class BasicSyrkTla {
public:
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;

    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementX = typename BlockMmad::ElementX;
    using LayoutTagX = typename BlockMmad::LayoutTagX;
    using LayoutX = typename BlockMmad::LayoutX;
    using ElementXt = typename BlockMmad::ElementXt;
    using LayoutTagXt = typename BlockMmad::LayoutTagXt;
    using LayoutXt = typename BlockMmad::LayoutXt;
    using ElementY = typename BlockMmad::ElementY;
    using LayoutTagY = typename BlockMmad::LayoutTagY;
    using LayoutY = typename BlockMmad::LayoutY;
    using LayoutTagYT = layout::ColumnMajor; // ColumnMajor view of the same Y buffer for nz2dn (transpose) store.
    using LayoutYT = detail::TagToLayout_t<ElementY, LayoutTagYT>;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        uint32_t batchCount;
        GemmCoord problemShape; // (M, M, K)
        GM_ADDR ptrX;
        GM_ADDR ptrXt;
        GM_ADDR ptrY;
        int64_t strideX;
        int64_t strideXt;
        int64_t strideY;
        LayoutX layoutX;
        LayoutXt layoutXt;
        LayoutY layoutY;
        LayoutYT layoutYT;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            uint32_t batchCount_, GemmCoord const& problemShape_, GM_ADDR ptrX_, GM_ADDR ptrXt_, GM_ADDR ptrY_,
            int64_t strideX_, int64_t strideXt_, int64_t strideY_, LayoutX layoutX_, LayoutXt layoutXt_,
            LayoutY layoutY_, LayoutYT layoutYT_)
            : batchCount(batchCount_),
              problemShape(problemShape_),
              ptrX(ptrX_),
              ptrXt(ptrXt_),
              ptrY(ptrY_),
              strideX(strideX_),
              strideXt(strideXt_),
              strideY(strideY_),
              layoutX(layoutX_),
              layoutXt(layoutXt_),
              layoutY(layoutY_),
              layoutYT(layoutYT_)
        {}
    };

    struct Arguments {
        uint32_t batchCount;
        GemmCoord problemShape; // (M, M, K)
        uint8_t* ptrX;
        uint8_t* ptrXt;
        uint8_t* ptrY;
    };

    static bool CanImplement(const Arguments& args)
    {
        // Delivery limits for host reference computation and comparison.
        // Check shape before allocation; device pointers are populated afterwards.
        constexpr uint32_t MAX_M = 8192;
        constexpr uint32_t MAX_K = 65536;
        constexpr uint64_t MAX_INPUT_ELEMENTS = uint64_t{MAX_M} * MAX_M;
        const uint32_t m = args.problemShape.m();
        const uint32_t k = args.problemShape.k();
        // Mirrored stores require a square output tile grid.
        return L1_TILE_M == L1_TILE_N && m == args.problemShape.n() && m > 0 && m <= MAX_M && k > 0 && k <= MAX_K &&
               uint64_t{m} * k <= MAX_INPUT_ELEMENTS;
    }

    static size_t GetWorkspaceSize(const Arguments& /*args*/)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* /*workspace*/)
    {
        uint32_t m = args.problemShape.m();
        uint32_t k = args.problemShape.k();
        return Params{
            args.batchCount,
            args.problemShape,
            args.ptrX,
            args.ptrXt,
            args.ptrY,
            static_cast<int64_t>(m) * k,
            static_cast<int64_t>(m) * k,
            static_cast<int64_t>(m) * m,
            tla::MakeLayout<ElementX, LayoutTagX>(m, k),
            tla::MakeLayout<ElementXt, LayoutTagXt>(k, m),
            tla::MakeLayout<ElementY, LayoutTagY>(m, m),
            tla::MakeLayout<ElementY, LayoutTagYT>(m, m),
        };
    }

    CATLASS_DEVICE
    BasicSyrkTla()
    {}

    template <int32_t CoreType_ = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler blockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = params.batchCount * blockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX*)params.ptrX);
        AscendC::GlobalTensor<ElementXt> gmXt;
        gmXt.SetGlobalBuffer((__gm__ ElementXt*)params.ptrXt);
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY*)params.ptrY);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            uint32_t batchIdx = blockScheduler.GetBatchIdx(loopIdx);
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);

            GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
            int64_t batchOffsetX = batchIdx * params.strideX;
            int64_t batchOffsetXt = batchIdx * params.strideXt;
            int64_t batchOffsetY = batchIdx * params.strideY;
            auto tensorX = tla::MakeTensor(gmX[batchOffsetX], params.layoutX, Arch::PositionGM{});
            auto tensorXt = tla::MakeTensor(gmXt[batchOffsetXt], params.layoutXt, Arch::PositionGM{});
            auto tensorY = tla::MakeTensor(gmY[batchOffsetY], params.layoutY, Arch::PositionGM{});
            auto tensorYT = tla::MakeTensor(gmY[batchOffsetY], params.layoutYT, Arch::PositionGM{});
            auto tileX = tla::GetTile(
                tensorX, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tileXt = tla::GetTile(
                tensorXt, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tileY = tla::GetTile(
                tensorY, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
            auto tileYT = tla::GetTile(
                tensorYT, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            blockMmad(tileX, tileXt, tileY, tileYT, actualBlockShape, blockCoord);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& /*params*/)
    {}
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_BASIC_SYRK_TLA_HPP
