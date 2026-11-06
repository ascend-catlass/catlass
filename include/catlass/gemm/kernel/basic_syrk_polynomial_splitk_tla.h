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

#ifndef CATLASS_GEMM_KERNEL_BASIC_SYRK_POLYNOMIAL_SPLITK_TLA_H
#define CATLASS_GEMM_KERNEL_BASIC_SYRK_POLYNOMIAL_SPLITK_TLA_H

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/matrix.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include <cstdint>
#include <type_traits>

namespace Catlass::Gemm::Kernel {

/**
 * @brief Split-K form of Y = alpha * (X * X^T) + beta * R + diagonal * I.
 *
 * X (M, K) and Xt (K, M) are the plain and transposed views of the same data, so ptrXt
 * normally equals ptrX; with distinct buffers the caller must guarantee a symmetric
 * product. R is the residual at ptrResidual, read only when beta != 0.
 *
 * AIC writes one FP32 partial tile per (output tile, K slice) to the caller-owned
 * workspace. After a device-wide barrier the AIVs reduce the slices of one row band and
 * apply alpha, the residual and the diagonal term once. Only lower-triangular tiles are
 * scheduled; the epilogue mirrors off-diagonal tiles.
 *
 * ElementY may be FP16/BF16 or float.
 */
template <class BlockMmad_, class BlockScheduler_>
class BasicSyrkPolynomialSplitkTla {
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
    using ElementAccumulator = float;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    // One workspace slot per (output tile, K slice), padded to the full tile.
    static constexpr uint32_t TILE_AREA = L1_TILE_M * L1_TILE_N;
    // The AIVs walk an output tile one band of rows at a time: a band is summed over the
    // K slices and finished by the epilogue before the next band starts.
    static constexpr uint32_t BAND_ROWS = 32;
    static constexpr uint32_t BAND_ELEMENTS = BAND_ROWS * L1_TILE_N;
    static constexpr uint32_t BANDS_PER_TILE = L1_TILE_M / BAND_ROWS;
    static constexpr uint32_t VNCHW_SIZE = 16;

    static constexpr uint32_t PRODUCT_BYTES = BAND_ELEMENTS * sizeof(float);
    static constexpr uint32_t STAGE_BYTES = BAND_ELEMENTS * sizeof(float);
    static constexpr uint32_t OUTPUT_BYTES = BAND_ELEMENTS * sizeof(ElementY);
    static constexpr uint32_t RESIDUAL_BYTES = BAND_ELEMENTS * sizeof(ElementY);
    static_assert(
        PRODUCT_BYTES + STAGE_BYTES + OUTPUT_BYTES + RESIDUAL_BYTES + BAND_ELEMENTS * sizeof(float) <= ArchTag::UB_SIZE,
        "split-K SYRK epilogue buffers exceed UB");

    struct Arguments {
        uint32_t batchCount;
        GemmCoord problemShape;
        uint8_t* ptrX;
        uint8_t* ptrXt;
        uint8_t* ptrResidual;
        uint8_t* ptrY;
        uint8_t* ptrWorkspace;
        float alpha;
        float beta;
        float diagonal;
        uint32_t splitkFactor;
    };

    struct Params {
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrX;
        GM_ADDR ptrXt;
        GM_ADDR ptrResidual;
        GM_ADDR ptrY;
        GM_ADDR ptrWorkspace;
        int64_t strideX;
        int64_t strideXt;
        int64_t strideResidual;
        int64_t strideY;
        LayoutX layoutX;
        LayoutXt layoutXt;
        float alpha;
        float beta;
        float diagonal;
        /// Effective slice count (GetEffectiveSplit), which may be below the requested factor.
        uint32_t splitkFactor;
        uint32_t kSliceLen;
    };

    /// K slice length for a given split factor; slices start on an L1 K-tile boundary so
    /// every GM tile load keeps the alignment of the non-split path.
    CATLASS_HOST_DEVICE
    static uint32_t GetKSliceLen(uint32_t k, uint32_t splitkFactor)
    {
        uint32_t len = RoundUp(CeilDiv(k, splitkFactor), L1_TILE_K);
        return len < L1_TILE_K ? L1_TILE_K : len;
    }

    /// Slices that actually carry work (the requested factor may exceed the K tile count).
    CATLASS_HOST_DEVICE
    static uint32_t GetEffectiveSplit(uint32_t k, uint32_t splitkFactor)
    {
        return CeilDiv(k, GetKSliceLen(k, splitkFactor));
    }

    static bool CanImplement(Arguments const& args)
    {
        return args.problemShape.m() == args.problemShape.n() && args.ptrResidual != nullptr &&
               args.ptrWorkspace != nullptr && args.splitkFactor >= 1;
    }

    /// The workspace is supplied by the caller (the torch adapter owns its lifetime).
    static size_t GetWorkspaceSize(Arguments const&)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(Arguments const& args, uint8_t*)
    {
        uint32_t m = args.problemShape.m();
        uint32_t k = args.problemShape.k();
        uint32_t kSliceLen = GetKSliceLen(k, args.splitkFactor);
        return Params{
            args.batchCount,
            args.problemShape,
            args.ptrX,
            args.ptrXt,
            args.ptrResidual,
            args.ptrY,
            args.ptrWorkspace,
            static_cast<int64_t>(m) * k,
            static_cast<int64_t>(m) * k,
            static_cast<int64_t>(m) * m,
            static_cast<int64_t>(m) * m,
            tla::MakeLayout<ElementX, typename BlockMmad::LayoutTagX>(m, k),
            tla::MakeLayout<ElementXt, typename BlockMmad::LayoutTagXt>(k, m),
            args.alpha,
            args.beta,
            args.diagonal,
            GetEffectiveSplit(k, args.splitkFactor),
            kSliceLen};
    }

    CATLASS_DEVICE
    BasicSyrkPolynomialSplitkTla()
    {}

    CATLASS_DEVICE
    void operator()(Params const& params)
    {
        BlockScheduler scheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t totalTasks = params.batchCount * scheduler.GetCoreLoops();
        uint32_t splitk = params.splitkFactor;

        Arch::Resource<ArchTag> resource;

        AscendC::GlobalTensor<ElementAccumulator> gmWorkspace;
        gmWorkspace.SetGlobalBuffer((__gm__ ElementAccumulator*)params.ptrWorkspace);

        if ASCEND_IS_AIC {
            BlockMmad blockMmad(resource);
            AscendC::GlobalTensor<ElementX> gmX;
            gmX.SetGlobalBuffer((__gm__ ElementX*)params.ptrX);
            AscendC::GlobalTensor<ElementXt> gmXt;
            gmXt.SetGlobalBuffer((__gm__ ElementXt*)params.ptrXt);

            uint32_t slots = totalTasks * splitk;
            for (uint32_t slot = AscendC::GetBlockIdx(); slot < slots; slot += AscendC::GetBlockNum()) {
                uint32_t task = slot / splitk;
                uint32_t slice = slot - task * splitk;
                uint32_t batchIdx = scheduler.GetBatchIdx(task);
                GemmCoord blockCoord = scheduler.GetBlockCoord(task);
                GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);

                uint32_t kBegin = slice * params.kSliceLen;
                uint32_t kActual = min(params.kSliceLen, params.problemShape.k() - kBegin);

                auto tensorX = tla::MakeTensor(gmX[batchIdx * params.strideX], params.layoutX, Arch::PositionGM{});
                auto tensorXt = tla::MakeTensor(gmXt[batchIdx * params.strideXt], params.layoutXt, Arch::PositionGM{});
                auto tileX = tla::GetTile(
                    tensorX, tla::MakeCoord(blockCoord.m() * L1_TILE_M, kBegin),
                    tla::MakeShape(actualShape.m(), kActual));
                auto tileXt = tla::GetTile(
                    tensorXt, tla::MakeCoord(kBegin, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(kActual, actualShape.n()));

                uint64_t slotOffset = static_cast<uint64_t>(slot) * TILE_AREA;
                auto layoutSlot =
                    tla::MakeLayout<ElementAccumulator, layout::RowMajor>(actualShape.m(), actualShape.n());
                auto tensorSlot = tla::MakeTensor(gmWorkspace[slotOffset], layoutSlot, Arch::PositionGM{});

                GemmCoord mmadShape{actualShape.m(), actualShape.n(), kActual};
                // GemmCoord{0, 0, 0}: the partial goes to its own workspace slot, so the
                // block mmad must take its single-store (diagonal) path and never mirror.
                blockMmad(tileX, tileXt, tensorSlot, tensorSlot, mmadShape, GemmCoord{0, 0, 0});
            }
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        if ASCEND_IS_AIV {
            // Wait for the paired AIC, then for every other core: the reduction below reads
            // workspace slots written by all of them.
            Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagAicFinish);
            Arch::CrossCoreBarrier<0x0, PIPE_MTE2>();

            AscendC::GlobalTensor<ElementY> gmResidual;
            gmResidual.SetGlobalBuffer((__gm__ ElementY*)params.ptrResidual);
            AscendC::GlobalTensor<ElementY> gmY;
            gmY.SetGlobalBuffer((__gm__ ElementY*)params.ptrY);

            AscendC::LocalTensor<float> ubProduct = resource.ubBuf.template GetBufferByByte<float>(0);
            AscendC::LocalTensor<float> ubStage = resource.ubBuf.template GetBufferByByte<float>(PRODUCT_BYTES);
            AscendC::LocalTensor<ElementY> ubOutput =
                resource.ubBuf.template GetBufferByByte<ElementY>(PRODUCT_BYTES + STAGE_BYTES);
            AscendC::LocalTensor<ElementY> ubResidual =
                resource.ubBuf.template GetBufferByByte<ElementY>(PRODUCT_BYTES + STAGE_BYTES + OUTPUT_BYTES);
            AscendC::LocalTensor<float> ubResidualWide = resource.ubBuf.template GetBufferByByte<float>(
                PRODUCT_BYTES + STAGE_BYTES + OUTPUT_BYTES + RESIDUAL_BYTES);

            uint32_t aivIdx = AscendC::GetBlockIdx();
            uint32_t aivCount = AscendC::GetBlockNum() * AscendC::GetTaskRation();
            uint32_t totalBands = totalTasks * BANDS_PER_TILE;

            for (uint32_t unit = aivIdx; unit < totalBands; unit += aivCount) {
                uint32_t task = unit / BANDS_PER_TILE;
                uint32_t band = unit - task * BANDS_PER_TILE;
                uint32_t batchIdx = scheduler.GetBatchIdx(task);
                GemmCoord blockCoord = scheduler.GetBlockCoord(task);
                GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);

                uint32_t firstRow = band * BAND_ROWS;
                if (firstRow >= actualShape.m()) {
                    continue;
                }
                uint32_t rows = min(BAND_ROWS, actualShape.m() - firstRow);
                uint32_t columns = actualShape.n();
                uint32_t count = rows * columns;

                // ---- K reduction: sum the FP32 partials of this band, once ----
                uint64_t bandBase =
                    static_cast<uint64_t>(task) * splitk * TILE_AREA + static_cast<uint64_t>(firstRow) * columns;
                AscendC::DataCopy(ubProduct, gmWorkspace[bandBase], count);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                for (uint32_t slice = 1; slice < splitk; ++slice) {
                    AscendC::DataCopy(ubStage, gmWorkspace[bandBase + static_cast<uint64_t>(slice) * TILE_AREA], count);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Add(ubProduct, ubProduct, ubStage, count);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                }

                // ---- polynomial epilogue, applied once on the reduced product ----
                runEpilogue(
                    gmY, gmResidual, ubProduct, ubOutput, ubResidual, ubResidualWide, params, batchIdx, blockCoord,
                    firstRow, rows, columns);

                // Finish writing this band before either local buffer is reused: the
                // stores read both, and the next band's loads and vector work overwrite
                // them. The single-pass kernel gets that ordering from its per-task core
                // handshake; this path has none, so wait for the stores explicitly.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 0;
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};

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

    /** Transpose a tile for the mirrored half of the symmetric output.
     * The destination row stride must be a multiple of 16 elements.
     * Each destination address describes a 32-byte block: 16 FP16/BF16
     * elements, but only 8 FP32 elements. For FP32, two consecutive
     * addresses therefore describe the two halves of one 16-element row.
     */
    CATLASS_DEVICE
    static void transposeTile(
        AscendC::LocalTensor<ElementY>& dst, AscendC::LocalTensor<ElementY>& src, uint32_t rows, uint32_t columns,
        uint32_t dstStride)
    {
        static_assert(sizeof(ElementY) == 2 || std::is_same_v<ElementY, float>, "unsupported SYRK output type");
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);
        constexpr uint32_t BLOCKS_PER_OUTPUT_ROW = VNCHW_SIZE / ELE_NUM_PER_C0;
        uint64_t dstLocalList[VNCHW_SIZE];
        uint64_t srcLocalList[VNCHW_SIZE];
        uint32_t rowTiles = CeilDiv(rows, VNCHW_SIZE);
        uint32_t columnRepeats = CeilDiv(columns, ELE_NUM_PER_C0);
        for (uint32_t tile = 0; tile < rowTiles; ++tile) {
            for (uint32_t lane = 0; lane < VNCHW_SIZE; ++lane) {
                srcLocalList[lane] =
                    src.GetPhyAddr() + (tile * VNCHW_SIZE * columns + lane * columns) * sizeof(ElementY);
                uint32_t outputRow = lane / BLOCKS_PER_OUTPUT_ROW;
                uint32_t outputColumn = tile * VNCHW_SIZE + (lane % BLOCKS_PER_OUTPUT_ROW) * ELE_NUM_PER_C0;
                dstLocalList[lane] = dst.GetPhyAddr() + (outputRow * dstStride + outputColumn) * sizeof(ElementY);
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
        uint32_t batchIdx, GemmCoord const& blockCoord, uint32_t firstRow, uint32_t rows, uint32_t columns)
    {
        uint32_t matrixSize = params.problemShape.m();
        uint32_t globalRow = blockCoord.m() * L1_TILE_M + firstRow;
        uint32_t globalColumn = blockCoord.n() * L1_TILE_N;
        uint32_t count = rows * columns;

        AscendC::Muls(product, product, params.alpha, count);
        if (params.beta != 0.0f) {
            uint64_t residualOffset = static_cast<uint64_t>(batchIdx) * params.strideResidual +
                                      static_cast<uint64_t>(globalRow) * matrixSize + globalColumn;
            loadOutputTile(residual, gmResidual, residualOffset, rows, columns, matrixSize);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr (std::is_same_v<ElementY, float>) {
                AscendC::Muls(residualWide, residual, params.beta, count);
            } else {
                AscendC::Cast(residualWide, residual, AscendC::RoundMode::CAST_NONE, count);
                AscendC::Muls(residualWide, residualWide, params.beta, count);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(product, product, residualWide, count);
        }
        if (blockCoord.m() == blockCoord.n()) {
            AscendC::PipeBarrier<PIPE_V>();
            addDiagonal(product, globalRow, globalColumn, rows, columns, params.diagonal);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (std::is_same_v<ElementY, float>) {
            // Write the FP32 sum directly; a round trip through 16-bit would lose
            // precision. The other buffer transposes the mirrored tile.
            storeResult(gmY, product, output, params, batchIdx, blockCoord, firstRow, rows, columns, residualWide);
        } else {
            if constexpr (std::is_same_v<ElementY, half>) {
                AscendC::Cast(output, product, AscendC::RoundMode::CAST_NONE, count);
            } else {
                AscendC::Cast(output, product, AscendC::RoundMode::CAST_RINT, count);
            }
            AscendC::PipeBarrier<PIPE_V>();
            // The sum now lives in output, so reuse product's storage for the
            // mirrored tile's transpose.
            auto transpose = product.template ReinterpretCast<ElementY>();
            storeResult(gmY, output, transpose, params, batchIdx, blockCoord, firstRow, rows, columns, residualWide);
        }
    }
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_BASIC_SYRK_POLYNOMIAL_SPLITK_TLA_H
