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

#ifndef CATLASS_GEMM_KERNEL_BASIC_SYRK_POLYNOMIAL_TLA_H
#define CATLASS_GEMM_KERNEL_BASIC_SYRK_POLYNOMIAL_TLA_H

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/matrix.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include <cstdint>
#include <type_traits>

namespace Catlass::Gemm::Kernel {

/**
 * @brief Single-pass MIX SYRK polynomial: Y = alpha * (X * X^T) + beta * R + diagonal * I.
 *
 * X (M, K) and Xt (K, M) are the plain and transposed views of the same data, so ptrXt
 * normally equals ptrX; with distinct buffers the caller must guarantee a symmetric
 * product. R is the residual at ptrResidual, read only when beta != 0.
 *
 * AIC leaves each L0C tile in UB (CopyL0CToUBMode::SPLIT_M, one row half per AIV) and
 * hands it to the paired AIVs, which apply the epilogue. Only lower-triangle tiles are
 * scheduled; the diagonal term is added on diagonal tiles and off-diagonal tiles are
 * transposed in UB and mirrored into the upper triangle.
 *
 * ElementY must be 16 bit: the UB transpose of the mirrored tile is FP16/BF16 only.
 */
template <class BlockMmad_, class BlockScheduler_>
class BasicSyrkPolynomialTla {
public:
    using BlockMmad = BlockMmad_;
    using BlockScheduler = BlockScheduler_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementX = typename BlockMmad::ElementX;
    using ElementXt = typename BlockMmad::ElementXt;
    using ElementY = typename BlockMmad::ElementY;
    using LayoutX = typename BlockMmad::LayoutX;
    using LayoutXt = typename BlockMmad::LayoutXt;
    using LayoutY = typename BlockMmad::LayoutY;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t AIV_ROW_TILE = L1_TILE_M / 2;
    static constexpr uint32_t MAX_AIV_ELEMENTS = AIV_ROW_TILE * L1_TILE_N;
    // SPLIT_M gives each AIV one half of the tile's rows. The buffers below are
    // single-stage, so each is reused only after an explicit wait.
    static constexpr uint32_t OUTPUT_BYTES = MAX_AIV_ELEMENTS * sizeof(ElementY);
    static constexpr uint32_t PRODUCT_BYTES = MAX_AIV_ELEMENTS * sizeof(float);
    static constexpr uint32_t CHUNK_ROWS = 16;
    static_assert(
        PRODUCT_BYTES + OUTPUT_BYTES + CHUNK_ROWS * L1_TILE_N * (sizeof(ElementY) + sizeof(float)) <= ArchTag::UB_SIZE,
        "SYRK polynomial epilogue buffers exceed UB");

    struct Arguments {
        uint32_t batchCount;
        GemmCoord problemShape;
        uint8_t* ptrX;
        uint8_t* ptrXt;
        uint8_t* ptrResidual;
        uint8_t* ptrY;
        float alpha;
        float beta;
        float diagonal;
    };

    struct Params {
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrX;
        GM_ADDR ptrXt;
        GM_ADDR ptrResidual;
        GM_ADDR ptrY;
        int64_t strideX;
        int64_t strideXt;
        int64_t strideResidual;
        int64_t strideY;
        LayoutX layoutX;
        LayoutXt layoutXt;
        float alpha;
        float beta;
        float diagonal;
    };

    static bool CanImplement(Arguments const& args)
    {
        return args.problemShape.m() == args.problemShape.n() && args.ptrResidual != nullptr;
    }

    static size_t GetWorkspaceSize(Arguments const&)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(Arguments const& args, uint8_t*)
    {
        uint32_t m = args.problemShape.m();
        uint32_t k = args.problemShape.k();
        return Params{
            args.batchCount,
            args.problemShape,
            args.ptrX,
            args.ptrXt,
            args.ptrResidual,
            args.ptrY,
            static_cast<int64_t>(m) * k,
            static_cast<int64_t>(m) * k,
            static_cast<int64_t>(m) * m,
            static_cast<int64_t>(m) * m,
            tla::MakeLayout<ElementX, typename BlockMmad::LayoutTagX>(m, k),
            tla::MakeLayout<ElementXt, typename BlockMmad::LayoutTagXt>(k, m),
            args.alpha,
            args.beta,
            args.diagonal};
    }

    CATLASS_DEVICE
    BasicSyrkPolynomialTla()
    {}

    CATLASS_DEVICE
    void operator()(Params const& params)
    {
        BlockScheduler scheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t logicalCore = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            logicalCore /= AscendC::GetTaskRation();
        }
        uint32_t logicalCoreCount = AscendC::GetBlockNum();
        uint32_t totalTasks = params.batchCount * scheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX*)params.ptrX);
        AscendC::GlobalTensor<ElementXt> gmXt;
        gmXt.SetGlobalBuffer((__gm__ ElementXt*)params.ptrXt);
        AscendC::GlobalTensor<ElementY> gmResidual;
        gmResidual.SetGlobalBuffer((__gm__ ElementY*)params.ptrResidual);
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementY*)params.ptrY);

        AscendC::LocalTensor<float> ubProduct = resource.ubBuf.template GetBufferByByte<float>(0);
        AscendC::LocalTensor<ElementY> ubOutput = resource.ubBuf.template GetBufferByByte<ElementY>(PRODUCT_BYTES);
        AscendC::LocalTensor<ElementY> ubResidual =
            resource.ubBuf.template GetBufferByByte<ElementY>(PRODUCT_BYTES + OUTPUT_BYTES);
        AscendC::LocalTensor<float> ubResidualWide = resource.ubBuf.template GetBufferByByte<float>(
            PRODUCT_BYTES + OUTPUT_BYTES + CHUNK_ROWS * L1_TILE_N * sizeof(ElementY));
        if ASCEND_IS_AIV {
            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FLAG_AIV_RELEASE);
        }

        for (uint32_t task = logicalCore; task < totalTasks; task += logicalCoreCount) {
            uint32_t batchIdx = scheduler.GetBatchIdx(task);
            GemmCoord blockCoord = scheduler.GetBlockCoord(task);
            GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);

            if ASCEND_IS_AIC {
                waitForVector();

                int64_t xOffset = batchIdx * params.strideX;
                int64_t xtOffset = batchIdx * params.strideXt;
                auto tensorX = tla::MakeTensor(gmX[xOffset], params.layoutX, Arch::PositionGM{});
                auto tensorXt = tla::MakeTensor(gmXt[xtOffset], params.layoutXt, Arch::PositionGM{});
                auto tileX = tla::GetTile(
                    tensorX, tla::MakeCoord(blockCoord.m() * L1_TILE_M, 0),
                    tla::MakeShape(actualShape.m(), actualShape.k()));
                auto tileXt = tla::GetTile(
                    tensorXt, tla::MakeCoord(0, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualShape.k(), actualShape.n()));
                // Keep the full tile shape; SPLIT_M routes its row halves to the paired AIVs.
                auto localLayout = tla::MakeLayout<float, layout::RowMajor>(actualShape.m(), actualShape.n());
                auto tensorUb = tla::MakeTensor(ubProduct, localLayout, Arch::PositionUB{});
                blockMmad(tileX, tileXt, tensorUb, tensorUb, actualShape, blockCoord);
                notifyVector();
            }

            if ASCEND_IS_AIV {
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(FLAG_AIC_READY);
                runEpilogue(
                    gmY, gmResidual, ubProduct, ubOutput, ubResidual, ubResidualWide, params, batchIdx, blockCoord,
                    actualShape);
                AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FLAG_AIV_RELEASE);
            }
        }

        if ASCEND_IS_AIC {
            waitForVector();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr uint16_t SYNC_MODE = 4;
    static constexpr uint16_t FLAG_AIV_RELEASE = 6;
    static constexpr uint16_t FLAG_AIC_READY = 8;
    static constexpr uint16_t SECOND_AIV_FLAG_OFFSET = 16;

    CATLASS_DEVICE
    static void waitForVector()
    {
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_AIV_RELEASE);
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_AIV_RELEASE + SECOND_AIV_FLAG_OFFSET);
    }

    CATLASS_DEVICE
    static void notifyVector()
    {
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(FLAG_AIC_READY);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(FLAG_AIC_READY + SECOND_AIV_FLAG_OFFSET);
    }

    CATLASS_DEVICE
    static void loadOutputTile(
        AscendC::LocalTensor<ElementY> const& dst, AscendC::GlobalTensor<ElementY>& src, uint64_t srcOffset,
        uint32_t rows, uint32_t columns, uint32_t globalStride)
    {
        AscendC::DataCopyExtParams copyParams(
            rows, columns * sizeof(ElementY), (globalStride - columns) * sizeof(ElementY), 0, 0);
        AscendC::DataCopyPadExtParams<ElementY> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dst, src[srcOffset], copyParams, padParams);
    }

    CATLASS_DEVICE
    static void storeOutputTile(
        AscendC::GlobalTensor<ElementY>& dst, uint64_t dstOffset, AscendC::LocalTensor<ElementY> const& src,
        uint32_t rows, uint32_t columns, uint32_t globalStride)
    {
        AscendC::DataCopyExtParams copyParams(
            rows, columns * sizeof(ElementY), 0, (globalStride - columns) * sizeof(ElementY), 0);
        AscendC::DataCopyPad(dst[dstOffset], src, copyParams);
    }

    CATLASS_DEVICE
    static void addDiagonal(
        AscendC::LocalTensor<float>& data, uint32_t globalRow, uint32_t globalColumn, uint32_t rows, uint32_t columns,
        float diagonal)
    {
        if (diagonal == 0.0f) {
            return;
        }
        constexpr uint32_t ELEMENTS_PER_REPEAT = 64;
        for (uint32_t localRow = 0; localRow < rows; ++localRow) {
            uint32_t row = globalRow + localRow;
            if (row >= globalColumn && row < globalColumn + columns) {
                uint32_t localColumn = row - globalColumn;
                uint32_t vectorBase = (localColumn / ELEMENTS_PER_REPEAT) * ELEMENTS_PER_REPEAT;
                uint32_t lane = localColumn - vectorBase;
                uint64_t lowMask = uint64_t{1} << lane;
                AscendC::SetVectorMask<int8_t>(0, lowMask);
                uint32_t offset = localRow * columns + vectorBase;
                AscendC::Adds<float, false>(
                    data[offset], data[offset], diagonal, (uint64_t)0, 1, AscendC::UnaryRepeatParams(1, 1, 8, 8));
            }
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }

    static constexpr uint32_t VNCHW_SIZE = 16;

    /** Transpose into a destination whose row stride is 16-element aligned. */
    CATLASS_DEVICE
    static void transposeTile(
        AscendC::LocalTensor<ElementY>& dst, AscendC::LocalTensor<ElementY>& src, uint32_t rows, uint32_t columns,
        uint32_t dstStride)
    {
        static_assert(sizeof(ElementY) == 2, "direct SYRK transpose expects a 16-bit output type");
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
        uint64_t dstLocalList[VNCHW_SIZE];
        uint64_t srcLocalList[VNCHW_SIZE];
        uint32_t rowTiles = CeilDiv(rows, VNCHW_SIZE);
        uint32_t columnRepeats = CeilDiv(columns, ELE_NUM_PER_C0);
        for (uint32_t tile = 0; tile < rowTiles; ++tile) {
            for (uint32_t lane = 0; lane < VNCHW_SIZE; ++lane) {
                srcLocalList[lane] =
                    src.GetPhyAddr() + (tile * VNCHW_SIZE * columns + lane * columns) * sizeof(ElementY);
                dstLocalList[lane] = dst.GetPhyAddr() + (tile * VNCHW_SIZE + lane * dstStride) * sizeof(ElementY);
            }
            if (columnRepeats > 1) {
                AscendC::TransDataTo5HDParams transposeParams(false, false, columnRepeats, dstStride, 1);
                AscendC::TransDataTo5HD<ElementY>(dstLocalList, srcLocalList, transposeParams);
            } else {
                AscendC::TransDataTo5HDParams transposeParams(false, false, 1, 0, 0);
                AscendC::TransDataTo5HD<ElementY>(dstLocalList, srcLocalList, transposeParams);
            }
        }
    }

    /** Copy the rows one at a time: padded source rows leave a gap that DataCopyPad
     * cannot describe, so the tile cannot go out in a single copy. */
    CATLASS_DEVICE
    static void storeStridedTile(
        AscendC::GlobalTensor<ElementY>& dst, uint64_t dstOffset, AscendC::LocalTensor<ElementY> const& src,
        uint32_t rows, uint32_t columns, uint32_t srcStride, uint32_t globalStride)
    {
        if (srcStride == columns) {
            storeOutputTile(dst, dstOffset, src, rows, columns, globalStride);
            return;
        }
        for (uint32_t row = 0; row < rows; ++row) {
            storeOutputTile(
                dst, dstOffset + static_cast<uint64_t>(row) * globalStride, src[row * srcStride], 1, columns,
                globalStride);
        }
    }

    CATLASS_DEVICE
    static void storeResult(
        AscendC::GlobalTensor<ElementY>& gmY, AscendC::LocalTensor<ElementY>& product,
        AscendC::LocalTensor<ElementY>& transpose, Params const& params, uint32_t batchIdx, GemmCoord const& blockCoord,
        uint32_t firstRow, uint32_t rows, uint32_t columns, AscendC::LocalTensor<float>& scratch)
    {
        uint32_t matrixSize = params.problemShape.m();
        uint64_t batchOffset = static_cast<uint64_t>(batchIdx) * params.strideY;
        uint32_t lowerRow = blockCoord.m() * L1_TILE_M + firstRow;
        uint32_t lowerColumn = blockCoord.n() * L1_TILE_N;
        uint64_t lowerOffset = batchOffset + static_cast<uint64_t>(lowerRow) * matrixSize + lowerColumn;

        if (params.ptrX != params.ptrXt && blockCoord.m() == blockCoord.n()) {
            // Store disjoint triangles from the same computed values. Gather
            // packs each transposed suffix at an aligned UB address for DataCopyPad.
            using Bits = std::conditional_t<sizeof(ElementY) == 2, uint16_t, uint32_t>;
            auto indices = scratch.template ReinterpretCast<uint32_t>();
            for (uint32_t i = 0; i < rows; ++i)
                indices.SetValue(i, i * sizeof(ElementY));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            uint32_t transposeStride = RoundUp(rows, VNCHW_SIZE);
            transposeTile(transpose, product, rows, columns, transposeStride);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            if (firstRow > 0) {
                AscendC::DataCopyExtParams lowerCopy(
                    rows, firstRow * sizeof(ElementY),
                    (columns - RoundUp(firstRow, 32u / sizeof(ElementY))) * sizeof(ElementY) / 32,
                    (matrixSize - firstRow) * sizeof(ElementY), 0);
                AscendC::DataCopyPad(gmY[lowerOffset], product, lowerCopy);
                uint64_t upperOffset = batchOffset + static_cast<uint64_t>(blockCoord.n() * L1_TILE_N) * matrixSize +
                                       blockCoord.m() * L1_TILE_M + firstRow;
                storeStridedTile(gmY, upperOffset, transpose, firstRow, rows, transposeStride, matrixSize);
            }
            for (uint32_t row = 0; row < rows; ++row) {
                storeOutputTile(
                    gmY, lowerOffset + static_cast<uint64_t>(row) * matrixSize + firstRow,
                    product[row * columns + firstRow], 1, row + 1, matrixSize);
            }
            auto shifted = scratch[RoundUp(rows, 8u)].template ReinterpretCast<Bits>();
            auto transposedBits = transpose.template ReinterpretCast<Bits>();
            for (uint32_t column = firstRow; column < min(columns, firstRow + rows); ++column) {
                uint32_t begin = column - firstRow + 1;
                if (begin >= rows)
                    continue;
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::Gather(
                    shifted, transposedBits[column * transposeStride], indices, begin * sizeof(ElementY), rows - begin);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                uint64_t upperOffset = batchOffset +
                                       static_cast<uint64_t>(blockCoord.n() * L1_TILE_N + column) * matrixSize +
                                       blockCoord.m() * L1_TILE_M + firstRow + begin;
                AscendC::DataCopyExtParams copy(1, (rows - begin) * sizeof(ElementY), 0, 0, 0);
                AscendC::DataCopyPad(gmY[upperOffset], shifted.template ReinterpretCast<ElementY>(), copy);
            }
            return;
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        storeOutputTile(gmY, lowerOffset, product, rows, columns, matrixSize);

        if (blockCoord.m() != blockCoord.n()) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            uint32_t transposeStride = RoundUp(rows, VNCHW_SIZE);
            transposeTile(transpose, product, rows, columns, transposeStride);
            uint32_t upperRow = blockCoord.n() * L1_TILE_N;
            uint32_t upperColumn = blockCoord.m() * L1_TILE_M + firstRow;
            uint64_t upperOffset = batchOffset + static_cast<uint64_t>(upperRow) * matrixSize + upperColumn;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            storeStridedTile(gmY, upperOffset, transpose, columns, rows, transposeStride, matrixSize);
        }
    }

    CATLASS_DEVICE
    static void runEpilogue(
        AscendC::GlobalTensor<ElementY>& gmY, AscendC::GlobalTensor<ElementY>& gmResidual,
        AscendC::LocalTensor<float>& product, AscendC::LocalTensor<ElementY>& output,
        AscendC::LocalTensor<ElementY>& residual, AscendC::LocalTensor<float>& residualWide, Params const& params,
        uint32_t batchIdx, GemmCoord const& blockCoord, GemmCoord const& actualShape)
    {
        uint32_t subBlock = AscendC::GetSubBlockIdx();
        // Match FixPipe's SPLIT_M partition: each AIV receives ceil(M / 2) rows.
        uint32_t splitRows = RoundUp(actualShape.m(), 2u) / 2;
        uint32_t firstRow = subBlock * splitRows;
        if (firstRow >= actualShape.m()) {
            return;
        }
        uint32_t rows = min(splitRows, actualShape.m() - firstRow);
        uint32_t columns = actualShape.n();
        uint32_t matrixSize = params.problemShape.m();
        uint32_t globalRow = blockCoord.m() * L1_TILE_M + firstRow;
        uint32_t globalColumn = blockCoord.n() * L1_TILE_N;
        uint64_t batchResidualOffset = static_cast<uint64_t>(batchIdx) * params.strideResidual;
        uint32_t count = rows * columns;

        AscendC::Muls(product, product, params.alpha, count);
        if (params.beta != 0.0f) {
            for (uint32_t chunkRow = 0; chunkRow < rows; chunkRow += CHUNK_ROWS) {
                uint32_t chunkRows = min(CHUNK_ROWS, rows - chunkRow);
                uint32_t chunkCount = chunkRows * columns;
                uint64_t residualOffset =
                    batchResidualOffset + static_cast<uint64_t>(globalRow + chunkRow) * matrixSize + globalColumn;
                loadOutputTile(residual, gmResidual, residualOffset, chunkRows, columns, matrixSize);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast(residualWide, residual, AscendC::RoundMode::CAST_NONE, chunkCount);
                AscendC::Muls(residualWide, residualWide, params.beta, chunkCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(product[chunkRow * columns], product[chunkRow * columns], residualWide, chunkCount);
                // The next pass loads into the same two residual buffers, so the reads
                // above must finish before that load starts.
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            }
        }
        if (blockCoord.m() == blockCoord.n()) {
            AscendC::PipeBarrier<PIPE_V>();
            addDiagonal(product, globalRow, globalColumn, rows, columns, params.diagonal);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (std::is_same_v<ElementY, half>) {
            AscendC::Cast(output, product, AscendC::RoundMode::CAST_NONE, count);
        } else {
            AscendC::Cast(output, product, AscendC::RoundMode::CAST_RINT, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // The cast copied the result out of the FP32 accumulator, which is not read again;
        // reuse its space as the transpose buffer instead of reserving more UB.
        auto transpose = product.template ReinterpretCast<ElementY>();
        storeResult(gmY, output, transpose, params, batchIdx, blockCoord, firstRow, rows, columns, residualWide);
    }
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_BASIC_SYRK_POLYNOMIAL_TLA_H
