/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_SYMM_RIGHT_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_SYMM_RIGHT_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {

template <class DispatchPolicy>
struct SymmMmadPingpongRightTlaDispatchChecker {
    static constexpr bool value = false;
};

template <bool ENABLE_UNIT_FLAG>
struct SymmMmadPingpongRightTlaDispatchChecker<MmadAtlasA2PingpongSymmRight<ENABLE_UNIT_FLAG>> {
    static constexpr bool value = true;
};

template <class ArchTag, bool ENABLE_UNIT_FLAG>
struct SymmMmadPingpongRightTlaDispatchChecker<MmadPingpongSymmRight<ArchTag, ENABLE_UNIT_FLAG>> {
    static constexpr bool value = true;
};

template <class DispatchPolicy>
struct SymmRightTlaStagesGetter {
    static constexpr uint32_t STAGES = 1;
};

template <bool ENABLE_UNIT_FLAG>
struct SymmRightTlaStagesGetter<MmadAtlasA2PingpongSymmRight<ENABLE_UNIT_FLAG>> {
    static constexpr uint32_t STAGES = 2;
};

template <class ArchTag, bool ENABLE_UNIT_FLAG>
struct SymmRightTlaStagesGetter<MmadPingpongSymmRight<ArchTag, ENABLE_UNIT_FLAG>> {
    static constexpr uint32_t STAGES = 2;
};

template <
    class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
    class ElementBias_ = void,
    class TileCopyP_ = Gemm::Tile::PackedTileCopyTla<
        typename DispatchPolicy_::ArchTag, ElementA_, layout::RowMajor, ElementB_, layout::RowMajor, ElementC_,
        layout::RowMajor>,
    class TileCopyQ_ = Gemm::Tile::PackedTileCopyTla<
        typename DispatchPolicy_::ArchTag, ElementA_, layout::RowMajor, ElementB_, layout::ColumnMajor, ElementC_,
        layout::RowMajor>,
    class TileMmadP_ =
        Gemm::Tile::TileMmadTla<typename DispatchPolicy_::ArchTag, ElementA_, typename TileCopyP_::LayoutTagL1A>,
    class TileMmadQ_ =
        Gemm::Tile::TileMmadTla<typename DispatchPolicy_::ArchTag, ElementA_, typename TileCopyQ_::LayoutTagL1A>>
struct BlockMmadPingpongSymmRightTla {
public:
    using DispatchPolicy = DispatchPolicy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    static_assert(
        SymmMmadPingpongRightTlaDispatchChecker<DispatchPolicy_>::value,
        "BlockMmadPingpongSymmRightTla requires MmadAtlasA2PingpongSymmRight or MmadPingpongSymmRight dispatch policy");
    static_assert(
        tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
        "L1TileShape must be tla::tuple and static!");
    static_assert(
        tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
        "L0TileShape must be tla::tuple and static!");

    using ArchTag = typename DispatchPolicy::ArchTag;

    using TileCopyP = TileCopyP_;
    using TileCopyQ = TileCopyQ_;

    using ElementA = ElementA_;
    using LayoutA = typename TileCopyP::LayoutA;
    using ElementB = ElementB_;
    using LayoutBP = typename TileCopyP::LayoutB;
    using LayoutBQ = typename TileCopyQ::LayoutB;
    using LayoutB = LayoutBP;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopyP::LayoutC;
    using ElementBias = ElementBias_;

    using TileMmadP = TileMmadP_;
    using TileMmadQ = TileMmadQ_;

    using ElementAccumulator = typename TileCopyP::ElementAccumulator;

    using LayoutTagL1A = typename TileCopyP::LayoutTagL1A;
    using LayoutTagL1BP = typename TileCopyP::LayoutTagL1B;
    using LayoutTagL1BQ = typename TileCopyQ::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopyP::LayoutTagL0A;
    using LayoutTagL0BP = typename TileCopyP::LayoutTagL0B;
    using LayoutTagL0BQ = typename TileCopyQ::LayoutTagL0B;

    using L1AAlignHelper = typename TileCopyP::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopyP::L1BAlignHelper;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t STAGES = SymmRightTlaStagesGetter<DispatchPolicy>::STAGES;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    static constexpr uint32_t L0A_BUF_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_BUF_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_BUF_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_BUF_SIZE / STAGES;

    static_assert(
        std::is_same_v<ArchTag, Arch::AtlasA2> || std::is_same_v<ArchTag, Arch::Ascend950>,
        "ArchTag can only be AtlasA2 or Ascend950!");
    static_assert(L1_TILE_K == L1_TILE_N, "Right-symm path requires square K/N tiles.");
    static_assert(tla::detail::isRowMajor<LayoutC>::value, "LayoutC only supports RowMajor!");
    static_assert((L1A_TILE_SIZE + L1B_TILE_SIZE) * STAGES <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");
    static_assert(L0A_TILE_SIZE * STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");
    static_assert(
        L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K cannot exceed L1TileShape::K");

    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1BP_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1BP>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});
    static constexpr auto L1BQ_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1BQ>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});

    CATLASS_DEVICE
    BlockMmadPingpongSymmRightTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0)
    {
        if ASCEND_IS_AIC {
            uint32_t l1AOffset = l1BufAddrStart;
            uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * STAGES;

            for (uint32_t i = 0; i < STAGES; i++) {
                l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
                l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
                l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
                l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
                l1AEventList[i] = i;
                l1BEventList[i] = i + STAGES;
                l0AEventList[i] = i;
                l0BEventList[i] = i + STAGES;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }

            l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
    }

    CATLASS_DEVICE
    ~BlockMmadPingpongSymmRightTla()
    {
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
    }

    template <bool UPPER_STORAGE, class TensorA, class TensorB, class TensorBQ, class TensorC>
    CATLASS_DEVICE void operator()(
        TensorA& tensorA, TensorB& tensorB, TensorBQ& tensorBQ, TensorC& tensorC, GemmCoord const& actualShape,
        uint32_t jTile, uint32_t jTileNext, uint32_t tilesN, bool isFirstBlock = true, bool hasNextBlock = false,
        TensorA const* tensorANext = nullptr, TensorB* tensorBNext = nullptr, TensorBQ const* tensorBQNext = nullptr,
        GemmCoord const& actualShapeNext = {})
    {
        uint32_t mBlockActual = actualShape.m();
        uint32_t nBlockActual = actualShape.n();

        using CopyGmToL1A = typename TileCopyP::template CopyGmToL1A<TensorA>;
        using CopyGmToL1BP = typename TileCopyP::template CopyGmToL1B<TensorB>;
        using CopyGmToL1BQ = typename TileCopyQ::template CopyGmToL1B<TensorBQ>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1BP copyGmToL1BP;
        CopyGmToL1BQ copyGmToL1BQ;

        auto layoutInL0C = tla::MakeLayoutL0C(mBlockActual, nBlockActual);
        auto tensorL0C_Tile = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});

        if (isFirstBlock) {
            uint32_t kActual = GetKActual(actualShape, 0);
            PreloadKTile<UPPER_STORAGE>(
                l1ListId, tensorA, tensorB, tensorBQ, jTile, 0, mBlockActual, nBlockActual, kActual, copyGmToL1A,
                copyGmToL1BP, copyGmToL1BQ);
        }

        bool initAccum = true;
        for (uint32_t kTile = 0; kTile < tilesN; ++kTile) {
            uint32_t l1ListIdCurrent = l1ListId;
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;

            if (kTile + 1 < tilesN) {
                uint32_t kActualNext = GetKActual(actualShape, kTile + 1);
                PreloadKTile<UPPER_STORAGE>(
                    l1ListIdNext, tensorA, tensorB, tensorBQ, jTile, kTile + 1, mBlockActual, nBlockActual, kActualNext,
                    copyGmToL1A, copyGmToL1BP, copyGmToL1BQ);
            } else if (hasNextBlock && tensorANext != nullptr && tensorBNext != nullptr && tensorBQNext != nullptr) {
                uint32_t kActualNext = GetKActual(actualShape, 0);
                PreloadKTile<UPPER_STORAGE>(
                    l1ListIdNext, *tensorANext, *tensorBNext, *tensorBQNext, jTileNext, 0, actualShapeNext.m(),
                    actualShapeNext.n(), kActualNext, copyGmToL1A, copyGmToL1BP, copyGmToL1BQ);
            }

            uint32_t kActual = GetKActual(actualShape, kTile);
            bool finalize = (kTile + 1 == tilesN);
            bool useDirect = UPPER_STORAGE ? (kTile <= jTile) : (kTile >= jTile);
            if (useDirect) {
                AccumulateLoaded<true>(
                    l1ListIdCurrent, mBlockActual, nBlockActual, kActual, initAccum, finalize, tensorL0C_Tile);
            } else {
                AccumulateLoaded<false>(
                    l1ListIdCurrent, mBlockActual, nBlockActual, kActual, initAccum, finalize, tensorL0C_Tile);
            }
            initAccum = false;
            l1ListId = l1ListIdNext;
        }
    }

    template <class TensorC>
    CATLASS_DEVICE void Store(TensorC& tensorC, GemmCoord const& actualShape)
    {
        uint32_t mBlockActual = actualShape.m();
        uint32_t nBlockActual = actualShape.n();
        auto layoutInL0C = tla::MakeLayoutL0C(mBlockActual, nBlockActual);
        auto tensorL0C_Tile = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
            using CopyL0CToGm = typename TileCopyP::template CopyL0CToGm<TensorC>;
            CopyL0CToGm copyL0CToGm;
            copyL0CToGm(tensorC, tensorL0C_Tile);
#elif (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202))
            using CopyL0CToDst = typename TileCopyP::template CopyL0CToDst<TensorC>;
            CopyL0CToDst copyL0CToDst;
            copyL0CToDst(tensorC, tensorL0C_Tile);
#endif
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
            using CopyL0CToGm = typename TileCopyP::template CopyL0CToGm<TensorC>;
            CopyL0CToGm copyL0CToGm;
            copyL0CToGm(tensorC, tensorL0C_Tile, 0b11);
#elif (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202))
            using CopyL0CToDst = typename TileCopyP::template CopyL0CToDst<TensorC>;
            CopyL0CToDst copyL0CToDst;
            copyL0CToDst(tensorC, tensorL0C_Tile, 0b11);
#endif
        }
    }

private:
    CATLASS_DEVICE
    uint32_t GetKActual(GemmCoord const& actualShape, uint32_t kTile) const
    {
        uint32_t kRemain = actualShape.k() - kTile * L1_TILE_K;
        return (kRemain < L1_TILE_K) ? kRemain : L1_TILE_K;
    }

    template <
        bool UPPER_STORAGE, class TensorA, class TensorB, class TensorBQ, class CopyGmToL1A, class CopyGmToL1BP,
        class CopyGmToL1BQ>
    CATLASS_DEVICE void PreloadKTile(
        uint32_t stageId, TensorA& tensorA, TensorB& tensorB, TensorBQ& tensorBQ, uint32_t jTile, uint32_t kTile,
        uint32_t mActual, uint32_t nActual, uint32_t kActual, CopyGmToL1A& copyGmToL1A, CopyGmToL1BP& copyGmToL1BP,
        CopyGmToL1BQ& copyGmToL1BQ)
    {
        // Load A (non-symmetric operand)
        auto tensorL1A = tla::MakeTensor(l1ATensorList[stageId], L1A_LAYOUT, Arch::PositionL1{});
        auto tensorTileA = GetTile(tensorA, tla::MakeCoord(0, kTile * L1_TILE_K), tla::MakeShape(mActual, kActual));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[stageId]);
        copyGmToL1A(tensorL1A, tensorTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[stageId]);

        // Load B (symmetric operand) with triangle logic
        bool useDirect = UPPER_STORAGE ? (kTile <= jTile) : (kTile >= jTile);
        if (useDirect) {
            auto tensorL1B = tla::MakeTensor(l1BTensorList[stageId], L1BP_LAYOUT, Arch::PositionL1{});
            auto tensorTileB = GetTile(tensorB, tla::MakeCoord(kTile * L1_TILE_K, 0), tla::MakeShape(kActual, nActual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[stageId]);
            copyGmToL1BP(tensorL1B, tensorTileB);
            if (kTile == jTile) {
                // Diagonal: symmetry fixup on B in L1
                auto l1BTensor = l1BTensorList[stageId];
                for (uint32_t r = 0; r < kActual; ++r) {
                    for (uint32_t c = 0; c < r; ++c) {
                        if constexpr (UPPER_STORAGE) {
                            l1BTensor[L1BP_LAYOUT(tla::MakeCoord(r, c))] = l1BTensor[L1BP_LAYOUT(tla::MakeCoord(c, r))];
                        } else {
                            l1BTensor[L1BP_LAYOUT(tla::MakeCoord(c, r))] = l1BTensor[L1BP_LAYOUT(tla::MakeCoord(r, c))];
                        }
                    }
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[stageId]);
        } else {
            auto tensorL1B = tla::MakeTensor(l1BTensorList[stageId], L1BQ_LAYOUT, Arch::PositionL1{});
            auto tensorTileB =
                GetTile(tensorBQ, tla::MakeCoord(kTile * L1_TILE_K, 0), tla::MakeShape(kActual, nActual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[stageId]);
            copyGmToL1BQ(tensorL1B, tensorTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[stageId]);
        }
    }

    template <bool USE_PRIMARY, class TensorL0C>
    CATLASS_DEVICE void AccumulateLoaded(
        uint32_t stageId, uint32_t mActual, uint32_t nActual, uint32_t kActual, bool init, bool finalize,
        TensorL0C& tensorL0C)
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mActual);
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nActual);

        uint32_t mPartLoop = CeilDiv<L0_TILE_M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0_TILE_N>(nRound);
        uint32_t kPartLoop = CeilDiv<L0_TILE_K>(kActual);

        // Create L1A tensor once per stage (always same layout for right block)
        auto tensorL1A = tla::MakeTensor(l1ATensorList[stageId], L1A_LAYOUT, Arch::PositionL1{});

        for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
            uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ? L0_TILE_M : (mRound - mPartIdx * L0_TILE_M);
            for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ? L0_TILE_K : (kActual - kPartIdx * L0_TILE_K);

                // L1A → L0A
                auto l0ATile = l0ATensorList[l0AListId];
                auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mPartActual, kPartActual);
                auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});
                auto tensorTileL1A = GetTile(
                    tensorL1A, tla::MakeCoord(mPartIdx * L0_TILE_M, kPartIdx * L0_TILE_K),
                    tla::MakeShape(mPartActual, kPartActual));

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                if ((mPartIdx == 0) && (kPartIdx == 0)) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[stageId]);
                }
                copyL1ToL0A(tensorL0A, tensorTileL1A);
                if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[stageId]);
                }

                for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                    uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ? L0_TILE_N : (nRound - nPartIdx * L0_TILE_N);

                    // L1B → L0B (L0B uses nZ for both paths)
                    auto l0BTile = l0BTensorList[l0BListId];
                    using LayoutTagL0B = std::conditional_t<USE_PRIMARY, LayoutTagL0BP, LayoutTagL0BQ>;
                    auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kPartActual, nPartActual);
                    auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    if ((kPartIdx == 0) && (nPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[stageId]);
                    }
                    if constexpr (USE_PRIMARY) {
                        auto tensorTileL1B = GetTile(
                            tla::MakeTensor(l1BTensorList[stageId], L1BP_LAYOUT, Arch::PositionL1{}),
                            tla::MakeCoord(kPartIdx * L0_TILE_K, nPartIdx * L0_TILE_N),
                            tla::MakeShape(kPartActual, nPartActual));
                        copyL1ToL0BP(tensorL0B, tensorTileL1B);
                    } else {
                        auto tensorTileL1B = GetTile(
                            tla::MakeTensor(l1BTensorList[stageId], L1BQ_LAYOUT, Arch::PositionL1{}),
                            tla::MakeCoord(kPartIdx * L0_TILE_K, nPartIdx * L0_TILE_N),
                            tla::MakeShape(kPartActual, nPartActual));
                        copyL1ToL0BQ(tensorL0B, tensorTileL1B);
                    }
                    if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[stageId]);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    auto tensorTileL0C = GetTile(
                        tensorL0C, tla::MakeCoord(mPartIdx * L0_TILE_M, nPartIdx * L0_TILE_N),
                        tla::MakeShape(mPartActual, nPartActual));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    bool initC = init && (kPartIdx == 0);
                    uint8_t unitFlag = 0b00;
                    if constexpr (ENABLE_UNIT_FLAG) {
                        if (finalize && (mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1) &&
                            (nPartIdx == nPartLoop - 1)) {
                            unitFlag = 0b11;
                        } else {
                            unitFlag = 0b10;
                        }
                    }
                    if constexpr (USE_PRIMARY) {
                        tileMmadP(
                            tensorTileL0C, tensorL0A, tensorL0B, mPartActual, nPartActual, kPartActual, initC,
                            unitFlag);
                    } else {
                        tileMmadQ(
                            tensorTileL0C, tensorL0A, tensorL0B, mPartActual, nPartActual, kPartActual, initC,
                            unitFlag);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
            }
        }
    }

    // ── Multi-stage buffers ──
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    uint32_t l1ListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    using CopyL1ToL0A_Mem = typename TileCopyP::CopyL1ToL0A;
    using CopyL1ToL0BP_Mem = typename TileCopyP::CopyL1ToL0B;
    using CopyL1ToL0BQ_Mem = typename TileCopyQ::CopyL1ToL0B;

    TileMmadP tileMmadP;
    TileMmadQ tileMmadQ;
    CopyL1ToL0A_Mem copyL1ToL0A;
    CopyL1ToL0BP_Mem copyL1ToL0BP;
    CopyL1ToL0BQ_Mem copyL1ToL0BQ;
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_SYMM_RIGHT_TLA_HPP
