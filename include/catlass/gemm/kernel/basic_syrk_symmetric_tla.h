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

#ifndef CATLASS_GEMM_KERNEL_BASIC_SYRK_SYMMETRIC_TLA_H
#define CATLASS_GEMM_KERNEL_BASIC_SYRK_SYMMETRIC_TLA_H

#include "catlass/gemm/kernel/basic_syrk_tla.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp"

#include <type_traits>

namespace Catlass::Gemm::Kernel {

// Diagonal tiles follow the same lower-triangle policy as the rest of the output.
// Only those tiles visit UB. Fixpipe supplies both orientations, and the AIVs
// select the lower half without arithmetic or an additional GM read.
//
// With ptrXt aliasing ptrX a diagonal tile is already exactly symmetric, so the
// mirror is skipped and the tile keeps the dense single-store path.
template <class BlockMmad_, class BlockScheduler_>
class BasicSyrkSymmetricTla : public BasicSyrkTla<BlockMmad_, void, BlockScheduler_> {
    using Base = BasicSyrkTla<BlockMmad_, void, BlockScheduler_>;
    using BlockMmad = BlockMmad_;
    using BlockScheduler = BlockScheduler_;
    using ArchTag = typename Base::ArchTag;
    using ElementX = typename Base::ElementX;
    using ElementXt = typename Base::ElementXt;
    using ElementY = typename Base::ElementY;
    static constexpr uint32_t TILE = Base::L1_TILE_M;
    static constexpr uint32_t BAND_ROWS = sizeof(ElementY) == 2 ? 128 : 64;
    static constexpr uint32_t BAND_BYTES = BAND_ROWS * TILE * sizeof(ElementY);
    static constexpr uint16_t SYNC_MODE = 4;
    static constexpr uint16_t RELEASE = 6;
    static constexpr uint16_t READY = 7;
    static constexpr uint16_t SECOND_AIV = 16;
    static_assert(2 * BAND_BYTES <= ArchTag::UB_SIZE, "diagonal SYRK buffers exceed UB");

    CATLASS_DEVICE static void waitForVector()
    {
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(RELEASE);
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(RELEASE + SECOND_AIV);
    }

    struct DiagonalStore {
        AscendC::LocalTensor<ElementY> normal;
        AscendC::LocalTensor<ElementY> transposed;
        uint32_t extent;

        template <class TensorL0C>
        CATLASS_DEVICE void operator()(TensorL0C const& source)
        {
            uint32_t half = RoundUp(extent, 32u) / 2;
            uint32_t stride = RoundUp(extent, 32u / sizeof(ElementY));
            for (uint32_t band = 0; band < CeilDiv(half, BAND_ROWS); ++band) {
                waitForVector();
                for (uint32_t sub = 0; sub < 2; ++sub) {
                    uint32_t first = sub * half + band * BAND_ROWS;
                    if (first >= extent)
                        continue;
                    uint32_t rows = min(BAND_ROWS, min(half - band * BAND_ROWS, extent - first));
                    auto normalLayout = tla::MakeLayout<ElementY, layout::RowMajor>(rows, stride);
                    auto normalDst = tla::MakeTensor(normal, normalLayout, Arch::PositionUB{});
                    auto normalSrc = tla::GetTile(source, tla::MakeCoord(first, 0), tla::MakeShape(rows, extent));
                    using NormalCopy = Gemm::Tile::CopyL0CToUBTla<ArchTag, decltype(normalSrc), decltype(normalDst)>;
                    NormalCopy{}(normalDst, normalSrc, static_cast<bool>(sub), 0);

                    auto transposeLayout = tla::MakeLayout<ElementY, layout::ColumnMajor>(stride, rows);
                    auto transposeDst = tla::MakeTensor(transposed, transposeLayout, Arch::PositionUB{});
                    auto transposeSrc = tla::GetTile(source, tla::MakeCoord(0, first), tla::MakeShape(extent, rows));
                    using TransposeCopy =
                        Gemm::Tile::CopyL0CToUBTla<ArchTag, decltype(transposeSrc), decltype(transposeDst)>;
                    TransposeCopy{}(transposeDst, transposeSrc, static_cast<bool>(sub), 0);
                }
                AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(READY);
                AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(READY + SECOND_AIV);
            }
        }
    };

    using Bits = std::conditional_t<sizeof(ElementY) == 2, uint16_t, uint32_t>;
    using Index = std::conditional_t<sizeof(ElementY) == 2, int16_t, int32_t>;

    CATLASS_DEVICE_SIMD_VF void selectUpperVf(
        __ubuf__ Bits* normal, __ubuf__ Bits* transposed, uint16_t first, uint16_t rows, uint16_t extent,
        uint16_t stride)
    {
        constexpr uint16_t LANES = AscendC::VECTOR_REG_WIDTH / sizeof(Bits);
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t remaining = extent;
            for (uint16_t column = 0; column < extent; column += LANES) {
                AscendC::MicroAPI::RegTensor<Bits> direct, mirrored, selected;
                AscendC::MicroAPI::RegTensor<Index> indices;
                auto valid = AscendC::MicroAPI::UpdateMask<Bits>(remaining);
                AscendC::MicroAPI::MaskReg upper;
                AscendC::MicroAPI::Arange<Index>(indices, static_cast<Index>(column));
                AscendC::MicroAPI::CompareScalar<Index, AscendC::CMPMODE::GT>(
                    upper, indices, static_cast<Index>(first + row), valid);
                uint32_t offset = row * stride + column;
                AscendC::MicroAPI::DataCopy(direct, normal + offset);
                AscendC::MicroAPI::DataCopy(mirrored, transposed + offset);
                AscendC::MicroAPI::Select(selected, mirrored, direct, upper);
                AscendC::MicroAPI::DataCopy(normal + offset, selected, valid);
            }
        }
    }

    CATLASS_DEVICE static void selectUpper(
        AscendC::LocalTensor<ElementY>& normal, AscendC::LocalTensor<ElementY>& transposed, uint32_t first,
        uint32_t rows, uint32_t extent, uint32_t stride)
    {
        // Select the stored bit patterns, preserving FP32 and signed zero exactly.
        selectUpperVf(
            (__ubuf__ Bits*)normal.GetPhyAddr(), (__ubuf__ Bits*)transposed.GetPhyAddr(), first, rows, extent, stride);
    }

public:
    using Params = typename Base::Params;
    using Arguments = typename Base::Arguments;

    CATLASS_DEVICE void operator()(Params const& params)
    {
        uint32_t logicalCore = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            logicalCore /= AscendC::GetTaskRation();
        }
        uint32_t coreCount = AscendC::GetBlockNum();
        BlockScheduler scheduler(params.problemShape, Catlass::MakeCoord(TILE, TILE));
        uint32_t tasks = params.batchCount * scheduler.GetCoreLoops();
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        auto normal = resource.ubBuf.template GetBufferByByte<ElementY>(0);
        auto transposed = resource.ubBuf.template GetBufferByByte<ElementY>(BAND_BYTES);
        AscendC::GlobalTensor<ElementX> gmX;
        AscendC::GlobalTensor<ElementXt> gmXt;
        AscendC::GlobalTensor<ElementY> gmY;
        gmX.SetGlobalBuffer((__gm__ ElementX*)params.ptrX);
        gmXt.SetGlobalBuffer((__gm__ ElementXt*)params.ptrXt);
        gmY.SetGlobalBuffer((__gm__ ElementY*)params.ptrY);
        // One release token circulates across bands and diagonal tiles. Issuing
        // a fresh token per tile could overtake the preceding tile's final release.
        if ASCEND_IS_AIV {
            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(RELEASE);
        }
        for (uint32_t task = logicalCore; task < tasks; task += coreCount) {
            uint32_t batch = scheduler.GetBatchIdx(task);
            GemmCoord coord = scheduler.GetBlockCoord(task);
            uint32_t tileRow = coord.m();
            uint32_t tileColumn = coord.n();
            auto actual = scheduler.GetActualBlockShape(coord);
            // (p, q) and (q, p) are the same dot product when the operands alias,
            // so only distinct operands need the mirror.
            bool mirror = tileRow == tileColumn && params.ptrX != params.ptrXt;
            if ASCEND_IS_AIC {
                auto x = tla::MakeTensor(gmX[batch * params.strideX], params.layoutX, Arch::PositionGM{});
                auto xt = tla::MakeTensor(gmXt[batch * params.strideXt], params.layoutXt, Arch::PositionGM{});
                auto y = tla::MakeTensor(gmY[batch * params.strideY], params.layoutY, Arch::PositionGM{});
                auto yt = tla::MakeTensor(gmY[batch * params.strideY], params.layoutYT, Arch::PositionGM{});
                auto tileX = tla::GetTile(x, tla::MakeCoord(tileRow * TILE, 0), tla::MakeShape(actual.m(), actual.k()));
                auto tileXt =
                    tla::GetTile(xt, tla::MakeCoord(0, tileColumn * TILE), tla::MakeShape(actual.k(), actual.n()));
                auto tileY = tla::GetTile(
                    y, tla::MakeCoord(tileRow * TILE, tileColumn * TILE), tla::MakeShape(actual.m(), actual.n()));
                auto tileYT = tla::GetTile(
                    yt, tla::MakeCoord(tileRow * TILE, tileColumn * TILE), tla::MakeShape(actual.m(), actual.n()));
                if (mirror) {
                    DiagonalStore store{normal, transposed, actual.m()};
                    blockMmad(tileX, tileXt, tileY, tileYT, actual, coord, &store);
                } else {
                    blockMmad(tileX, tileXt, tileY, tileYT, actual, coord);
                }
            }
            if ASCEND_IS_AIV {
                if (!mirror)
                    continue;
                uint32_t half = RoundUp(actual.m(), 32u) / 2;
                uint32_t stride = RoundUp(actual.n(), 32u / sizeof(ElementY));
                for (uint32_t band = 0; band < CeilDiv(half, BAND_ROWS); ++band) {
                    AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(READY);
                    uint32_t first = AscendC::GetSubBlockIdx() * half + band * BAND_ROWS;
                    if (first < actual.m()) {
                        uint32_t rows = min(BAND_ROWS, min(half - band * BAND_ROWS, actual.m() - first));
                        selectUpper(normal, transposed, first, rows, actual.n(), stride);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        uint64_t offset = static_cast<uint64_t>(batch) * params.strideY +
                                          static_cast<uint64_t>(tileRow * TILE + first) * params.problemShape.m() +
                                          tileColumn * TILE;
                        AscendC::DataCopyExtParams copy(
                            rows, actual.n() * sizeof(ElementY), 0,
                            (params.problemShape.m() - actual.n()) * sizeof(ElementY), 0);
                        // UB source rows use the padded stride. DataCopyPad rounds
                        // each source row to 32 bytes, matching that stride.
                        AscendC::DataCopyPad(gmY[offset], normal, copy);
                    }
                    AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(RELEASE);
                }
            }
        }
        if ASCEND_IS_AIC {
            waitForVector();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

} // namespace Catlass::Gemm::Kernel
#endif
