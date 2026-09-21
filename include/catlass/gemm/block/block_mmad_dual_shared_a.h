/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/helper.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {

template <
    class BaseDispatchPolicy, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
    class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<
    MmadDualSharedA<BaseDispatchPolicy>, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_,
    TileCopy_, TileMmad_> {
public:
    using Base = BlockMmadTla<
        BaseDispatchPolicy, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_,
        TileMmad_>;
    using DispatchPolicy = MmadDualSharedA<BaseDispatchPolicy>;
    using ArchTag = typename Base::ArchTag;
    using L1TileShape = typename Base::L1TileShape;
    using L0TileShape = typename Base::L0TileShape;
    using ElementB = typename Base::ElementB;
    using ElementA = typename Base::ElementA;
    using ElementC = typename Base::ElementC;
    using ElementAccumulator = typename Base::ElementAccumulator;
    using LayoutA = typename Base::LayoutA;
    using LayoutB = typename Base::LayoutB;
    using LayoutC = typename Base::LayoutC;
    using LayoutTagL1A = typename Base::LayoutTagL1A;
    using LayoutTagL0A = typename Base::LayoutTagL0A;
    using LayoutTagL1B = typename Base::LayoutTagL1B;
    using LayoutTagL0B = typename Base::LayoutTagL0B;
    using TileMmad = typename Base::TileMmad;
    using CopyL1ToL0B = typename Base::CopyL1ToL0B;
    using CopyL1ToL0A = typename Base::CopyL1ToL0A;
    using TileCopy = typename Base::TileCopy;

    static constexpr uint32_t L0B_STAGES = Base::L0B_STAGES;
    static constexpr uint32_t L0_TILE_M = Base::L0_TILE_M;
    static constexpr uint32_t L0_TILE_N = Base::L0_TILE_N;
    static constexpr uint32_t L0_TILE_K = Base::L0_TILE_K;
    static constexpr uint32_t L1_TILE_M = Base::L1_TILE_M;
    static constexpr uint32_t L1_TILE_N = Base::L1_TILE_N;
    static constexpr uint32_t L1_TILE_K = Base::L1_TILE_K;
    static constexpr uint32_t L1A_STAGES = Base::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = Base::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = Base::L0A_STAGES;
    static constexpr bool USE_FULL_LOAD_A = DispatchPolicy::USE_FULL_LOAD_A;
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);
    static constexpr int32_t L0C0_FIX_DONE_EVENT_ID = 0;
    static constexpr int32_t L0C1_FIX_DONE_EVENT_ID = 1;
    static constexpr int32_t MMAD_DONE_EVENT_ID = 0;
    static constexpr uint32_t UB_TILE_M =
        TileCopy::CopyMode == Gemm::Tile::CopyL0CToUBMode::SPLIT_M ? (L1_TILE_M + 1) / 2 : L1_TILE_M;
    static constexpr uint32_t UB_STRIDE_N = RoundUp<BYTE_PER_C0>(L1_TILE_N);
    static constexpr uint32_t UB_RESULT_ELEMS = UB_TILE_M * UB_STRIDE_N;
    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<Base::L1_TILE_M>{}, tla::Int<Base::L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<Base::L1_TILE_K>{}, tla::Int<Base::L1_TILE_N>{});

    template <class TensorA>
    CATLASS_DEVICE static auto GetTileA(
        TensorA& tensorA, uint32_t mIndex, uint32_t kIndex, uint32_t mSize, uint32_t kSize)
    {
        if constexpr (tla::detail::isVector<LayoutA>::value) {
            (void)mIndex;
            (void)mSize;
            return GetTile(tensorA, tla::MakeCoord(kIndex), tla::MakeShape(kSize));
        } else {
            return GetTile(tensorA, tla::MakeCoord(mIndex, kIndex), tla::MakeShape(mSize, kSize));
        }
    }

    template <class TensorA>
    CATLASS_DEVICE static int64_t GetFullLoadAOffset(TensorA& tensorA, GemmCoord const& blockCoord)
    {
        if constexpr (tla::detail::isVector<LayoutA>::value) {
            (void)tensorA;
            return static_cast<int64_t>(blockCoord.k()) * L1_TILE_K;
        } else {
            return static_cast<int64_t>(blockCoord.m()) * L1_TILE_M * tla::get<0>(tensorA.stride()) +
                   static_cast<int64_t>(blockCoord.k()) * L1_TILE_K;
        }
    }

    template <class TensorBlockA, class L1ATensor, class CopyGmToL1A>
    CATLASS_DEVICE static void LoadAPanel(
        TensorBlockA& tensorBlockA, L1ATensor& l1ATensor, int32_t eventId, uint32_t kOffset, uint32_t kActual,
        uint32_t mActual, CopyGmToL1A& copyGmToL1A)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventId);
        auto tensorL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT, Arch::PositionL1{});
        auto tensorTileA = GetTileA(tensorBlockA, 0, kOffset, mActual, kActual);
        copyGmToL1A(tensorL1A, tensorTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventId);
    }

    template <class TensorBlockB, class L1BTensor, class CopyGmToL1B>
    CATLASS_DEVICE static void LoadBPanel(
        TensorBlockB& tensorBlockB, L1BTensor& l1BTensor, int32_t eventId, uint32_t kOffset, uint32_t kActual,
        uint32_t nActual, CopyGmToL1B& copyGmToL1B)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventId);
        auto tensorL1B = tla::MakeTensor(l1BTensor, L1B_LAYOUT, Arch::PositionL1{});
        auto tensorTileB = GetTile(tensorBlockB, tla::MakeCoord(kOffset, 0), tla::MakeShape(kActual, nActual));
        copyGmToL1B(tensorL1B, tensorTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventId);
    }

    template <class BlockScheduler, class TensorA, class TensorB0, class TensorB1>
    CATLASS_DEVICE void operator()(
        BlockScheduler& scheduler, uint32_t coreLoops, Arch::Resource<ArchTag>& resource, TensorA& tensorA,
        TensorB0& tensorB0, TensorB1& tensorB1, Callback const& waitOutput = {}, Callback const& notifyOutput = {})
    {
        if constexpr (DispatchPolicy::USE_HF32_MODE) {
            AscendC::SetHF32Mode(true);
        } else {
            AscendC::SetHF32Mode(false);
        }
        if constexpr (DispatchPolicy::ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
            AscendC::SetMMLayoutTransform(true);
        }

        constexpr uint32_t l1AOffset = USE_FULL_LOAD_A ? 2 * L1B_STAGES * L1B_TILE_SIZE : 0;
        constexpr uint32_t l1B0Offset = USE_FULL_LOAD_A ? 0 : L1A_STAGES * L1A_TILE_SIZE;
        constexpr uint32_t l1B1Offset = l1B0Offset + L1B_STAGES * L1B_TILE_SIZE;

        AscendC::LocalTensor<ElementA> l1ATensorList[L1A_STAGES];
        int32_t l1AEventList[L1A_STAGES];
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
            l1AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }

        AscendC::LocalTensor<ElementB> l1B0TensorList[L1B_STAGES];
        AscendC::LocalTensor<ElementB> l1B1TensorList[L1B_STAGES];
        int32_t l1B0EventList[L1B_STAGES];
        int32_t l1B1EventList[L1B_STAGES];
        for (uint32_t i = 0; i < L1B_STAGES; ++i) {
            l1B0TensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1B0Offset + L1B_TILE_SIZE * i);
            l1B1TensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1B1Offset + L1B_TILE_SIZE * i);
            l1B0EventList[i] = i + L1A_STAGES;
            l1B1EventList[i] = i + L1A_STAGES + L1B_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1B0EventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1B1EventList[i]);
        }

        AscendC::LocalTensor<ElementA> l0ATensorList[L0A_STAGES];
        int32_t l0AEventList[L0A_STAGES];
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
            l0AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }

        AscendC::LocalTensor<ElementB> l0B0TensorList[L0B_STAGES];
        AscendC::LocalTensor<ElementB> l0B1TensorList[L0B_STAGES];
        int32_t l0B0EventList[L0B_STAGES];
        int32_t l0B1EventList[L0B_STAGES];
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0B0TensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);
            l0B1TensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * (L0B_STAGES + i));
            l0B0EventList[i] = i + L0A_STAGES;
            l0B1EventList[i] = i + L0A_STAGES + L0B_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0B0EventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0B1EventList[i]);
        }

        auto l0C0Tensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        auto l0C1Tensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE);
        constexpr int32_t l0C0FixDoneEventId = L0C0_FIX_DONE_EVENT_ID;
        constexpr int32_t l0C1FixDoneEventId = L0C1_FIX_DONE_EVENT_ID;
        if constexpr (L0_TILE_N != L1_TILE_N) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0C0FixDoneEventId);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0C1FixDoneEventId);
        }

        CopyL1ToL0A copyL1ToL0A;
        auto ubD0 = AscendC::LocalTensor<ElementC>(AscendC::TPosition::VECCALC, 0, UB_RESULT_ELEMS);
        auto ubD1 = AscendC::LocalTensor<ElementC>(
            AscendC::TPosition::VECCALC, UB_RESULT_ELEMS * sizeof(ElementC), UB_RESULT_ELEMS);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmOffsetAPreload{0};
        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = scheduler.GetActualBlockShape(blockCoord);
            uint32_t strideN = RoundUp<BYTE_PER_C0>(actualBlockShape.n());
            auto ubLayout = tla::MakeLayout(
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()),
                tla::MakeStride(static_cast<int64_t>(strideN), tla::Int<1>{}));
            auto tensorBlockD0 = tla::MakeTensor(ubD0, ubLayout, Arch::PositionUB{});
            auto tensorBlockD1 = tla::MakeTensor(ubD1, ubLayout, Arch::PositionUB{});
            auto tensorBlockA =
                GetTileA(tensorA, blockCoord.m() * L1_TILE_M, 0, actualBlockShape.m(), actualBlockShape.k());
            auto tensorBlockB0 = GetTile(
                tensorB0, tla::MakeCoord(0, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockB1 = GetTile(
                tensorB1, tla::MakeCoord(0, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

            bool needLoadL1A = true;
            if constexpr (USE_FULL_LOAD_A) {
                int64_t gmOffsetA = GetFullLoadAOffset(tensorA, blockCoord);
                bool isFirstBlock = loopIdx == coreIdx;
                if (isFirstBlock) {
                    gmOffsetAPreload = gmOffsetA;
                } else if (gmOffsetA == gmOffsetAPreload) {
                    needLoadL1A = false;
                } else {
                    gmOffsetAPreload = gmOffsetA;
                }
            }

            waitOutput();
            if constexpr (L0_TILE_N != L1_TILE_N) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0C0FixDoneEventId);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0C1FixDoneEventId);
            }

            RunBlockMainloop(
                tensorBlockA, tensorBlockB0, tensorBlockB1, tensorBlockD0, tensorBlockD1, actualBlockShape, needLoadL1A,
                l1ATensorList, l1AEventList, l1B0TensorList, l1B1TensorList, l1B0EventList, l1B1EventList,
                l0ATensorList, l0AEventList, l0B0TensorList, l0B1TensorList, l0B0EventList, l0B1EventList, l0C0Tensor,
                l0C1Tensor, copyL1ToL0A);

            notifyOutput();
        }

        waitOutput();
        if constexpr (L0_TILE_N != L1_TILE_N) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0C0FixDoneEventId);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0C1FixDoneEventId);
        }
        if constexpr (!USE_FULL_LOAD_A) {
            for (uint32_t i = 0; i < L1A_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
        }
        for (uint32_t i = 0; i < L1B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1B0EventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1B1EventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0B0EventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0B1EventList[i]);
        }
        if constexpr (DispatchPolicy::USE_HF32_MODE) {
            AscendC::SetHF32Mode(false);
        }
        if constexpr (DispatchPolicy::ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
            AscendC::SetMMLayoutTransform(false);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    template <class TensorL1B0, class TensorL1B1, class TensorL0C0, class TensorL0C1, class TensorL0A>
    CATLASS_DEVICE static void RunPair(
        TensorL1B0& tensorL1B0, TensorL1B1& tensorL1B1, TensorL0C0& tensorL0C0, TensorL0C1& tensorL0C1,
        TensorL0A& tensorL0A, AscendC::LocalTensor<ElementB> (&l0B0TensorList)[L0B_STAGES],
        AscendC::LocalTensor<ElementB> (&l0B1TensorList)[L0B_STAGES], uint32_t& l0B0ListId, uint32_t& l0B1ListId,
        uint32_t l1B0ListId, uint32_t l1B1ListId, GemmCoord const& tileCoord, GemmCoord const& tileShape,
        GemmCoord const& tileLoops, bool initC, uint8_t unitFlag)
    {
        uint32_t mL0Idx = tileCoord.m();
        uint32_t kL0Idx = tileCoord.k();
        uint32_t nL0Idx = tileCoord.n();
        uint32_t mL0Actual = tileShape.m();
        uint32_t kL0Actual = tileShape.k();
        uint32_t nL0Actual = tileShape.n();
        uint32_t mL0Loop = tileLoops.m();
        uint32_t kL0Loop = tileLoops.k();
        uint32_t nL0Loop = tileLoops.n();
        int32_t l1B0EventId = L1A_STAGES + l1B0ListId;
        int32_t l1B1EventId = L1A_STAGES + L1B_STAGES + l1B1ListId;
        TileMmad tileMmad;
        CopyL1ToL0B copyL1ToL0B;
        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, nL0Actual);
        auto tensorL0B0 = tla::MakeTensor(l0B0TensorList[l0B0ListId], layoutBInL0, Arch::PositionL0B{});
        auto tensorL0B1 = tla::MakeTensor(l0B1TensorList[l0B1ListId], layoutBInL0, Arch::PositionL0B{});
        auto tensorTileL1B0 = GetTile(
            tensorL1B0, tla::MakeCoord(kL0Idx * L0_TILE_K, nL0Idx * L0_TILE_N), tla::MakeShape(kL0Actual, nL0Actual));
        auto tensorTileL1B1 = GetTile(
            tensorL1B1, tla::MakeCoord(kL0Idx * L0_TILE_K, nL0Idx * L0_TILE_N), tla::MakeShape(kL0Actual, nL0Actual));

        int32_t b0ReadyEventId = L0A_STAGES + l0B0ListId;
        int32_t b1ReadyEventId = L0A_STAGES + L0B_STAGES + l0B1ListId;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(b0ReadyEventId);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(b1ReadyEventId);
        if ((mL0Idx == 0) && (kL0Idx == 0) && (nL0Idx == 0)) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1B0EventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1B1EventId);
        }

        copyL1ToL0B(tensorL0B0, tensorTileL1B0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(b0ReadyEventId);
        copyL1ToL0B(tensorL0B1, tensorTileL1B1);
        if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1) && (nL0Idx == nL0Loop - 1)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1B0EventId);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1B1EventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(b1ReadyEventId);

        auto tensorTileL0C0 = GetTile(
            tensorL0C0, tla::MakeCoord(mL0Idx * L0_TILE_M, nL0Idx * L0_TILE_N), tla::MakeShape(mL0Actual, nL0Actual));
        auto tensorTileL0C1 = GetTile(
            tensorL0C1, tla::MakeCoord(mL0Idx * L0_TILE_M, nL0Idx * L0_TILE_N), tla::MakeShape(mL0Actual, nL0Actual));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(b0ReadyEventId);
        RunMmad(
            tileMmad, tensorTileL0C0, tensorL0A, tensorL0B0, nL0Idx, mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(b0ReadyEventId);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(b1ReadyEventId);
        RunMmad(
            tileMmad, tensorTileL0C1, tensorL0A, tensorL0B1, nL0Idx, mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(b1ReadyEventId);
        l0B0ListId = (l0B0ListId + 1 < L0B_STAGES) ? (l0B0ListId + 1) : 0;
        l0B1ListId = (l0B1ListId + 1 < L0B_STAGES) ? (l0B1ListId + 1) : 0;
    }

    template <class TensorBlockA, class TensorBlockB0, class TensorBlockB1, class TensorBlockD0, class TensorBlockD1>
    CATLASS_DEVICE static void RunBlockMainloop(
        TensorBlockA& tensorBlockA, TensorBlockB0& tensorBlockB0, TensorBlockB1& tensorBlockB1,
        TensorBlockD0& tensorBlockD0, TensorBlockD1& tensorBlockD1, GemmCoord const& actualShape, bool needLoadL1A,
        AscendC::LocalTensor<ElementA> (&l1ATensorList)[L1A_STAGES], int32_t (&l1AEventList)[L1A_STAGES],
        AscendC::LocalTensor<ElementB> (&l1B0TensorList)[L1B_STAGES],
        AscendC::LocalTensor<ElementB> (&l1B1TensorList)[L1B_STAGES], int32_t (&l1B0EventList)[L1B_STAGES],
        int32_t (&l1B1EventList)[L1B_STAGES], AscendC::LocalTensor<ElementA> (&l0ATensorList)[L0A_STAGES],
        int32_t (&l0AEventList)[L0A_STAGES], AscendC::LocalTensor<ElementB> (&l0B0TensorList)[L0B_STAGES],
        AscendC::LocalTensor<ElementB> (&l0B1TensorList)[L0B_STAGES], int32_t (&l0B0EventList)[L0B_STAGES],
        int32_t (&l0B1EventList)[L0B_STAGES], AscendC::LocalTensor<ElementAccumulator>& l0C0Tensor,
        AscendC::LocalTensor<ElementAccumulator>& l0C1Tensor, CopyL1ToL0A& copyL1ToL0A)
    {
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorBlockA>;
        using CopyGmToL1B0 = typename TileCopy::template CopyGmToL1B<TensorBlockB0>;
        using CopyGmToL1B1 = typename TileCopy::template CopyGmToL1B<TensorBlockB1>;
        using CopyL0CToUb0 = typename TileCopy::template CopyL0CToDst<TensorBlockD0>;
        using CopyL0CToUb1 = typename TileCopy::template CopyL0CToDst<TensorBlockD1>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B0 copyGmToL1B0;
        CopyGmToL1B1 copyGmToL1B1;
        CopyL0CToUb0 copyL0CToUb0;
        CopyL0CToUb1 copyL0CToUb1;

        uint32_t mL1Actual = actualShape.m();
        uint32_t nL1Actual = actualShape.n();
        uint32_t kBlockActual = actualShape.k();
        uint32_t kL1Loop = CeilDiv<L1_TILE_K>(kBlockActual);
        uint32_t kL1Idx = 0;
        uint32_t kL1Actual = (kL1Idx < kL1Loop - 1) ? L1_TILE_K : (kBlockActual - kL1Idx * L1_TILE_K);

        auto layoutInL0C = tla::MakeLayoutL0C(mL1Actual, nL1Actual);
        auto tensorL0C0 = tla::MakeTensor(l0C0Tensor, layoutInL0C, Arch::PositionL0C{});
        auto tensorL0C1 = tla::MakeTensor(l0C1Tensor, layoutInL0C, Arch::PositionL0C{});

        uint32_t l1AListId = USE_FULL_LOAD_A ? 0 : (kL1Idx % L1A_STAGES);
        if constexpr (USE_FULL_LOAD_A) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[0]);
            auto l1ALayout = tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, actualShape.k());
            if (needLoadL1A) {
                auto tensorL1A = tla::MakeTensor(l1ATensorList[0], l1ALayout, Arch::PositionL1{});
                auto tensorTileA = GetTileA(tensorBlockA, 0, 0, actualShape.m(), actualShape.k());
                copyGmToL1A(tensorL1A, tensorTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[0]);
            }
        } else {
            LoadAPanel(
                tensorBlockA, l1ATensorList[l1AListId], l1AEventList[l1AListId], kL1Idx * L1_TILE_K, kL1Actual,
                mL1Actual, copyGmToL1A);
        }

        uint32_t l1B0ListId = 0;
        uint32_t l1B1ListId = 0;
        LoadBPanel(
            tensorBlockB0, l1B0TensorList[l1B0ListId], l1B0EventList[l1B0ListId], kL1Idx * L1_TILE_K, kL1Actual,
            nL1Actual, copyGmToL1B0);
        LoadBPanel(
            tensorBlockB1, l1B1TensorList[l1B1ListId], l1B1EventList[l1B1ListId], kL1Idx * L1_TILE_K, kL1Actual,
            nL1Actual, copyGmToL1B1);

        uint32_t mL0Loop = CeilDiv<L0_TILE_M>(mL1Actual);
        uint32_t nL0Loop = CeilDiv<L0_TILE_N>(nL1Actual);
        uint32_t l0AListId = 0;
        uint32_t l0B0ListId = 0;
        uint32_t l0B1ListId = 0;

        for (uint32_t kStep = 0; kStep < kL1Loop; ++kStep) {
            uint32_t l1AListIdNext = l1AListId;
            uint32_t l1B0ListIdNext = (l1B0ListId + 1 < L1B_STAGES) ? (l1B0ListId + 1) : 0;
            uint32_t l1B1ListIdNext = (l1B1ListId + 1 < L1B_STAGES) ? (l1B1ListId + 1) : 0;
            uint32_t kL1ActualNext{0};
            uint32_t kL1IdxNext = kL1Idx;
            if (kStep < kL1Loop - 1) {
                kL1IdxNext = kL1Idx + 1;
                kL1ActualNext = (kL1IdxNext < kL1Loop - 1) ? L1_TILE_K : (kBlockActual - kL1IdxNext * L1_TILE_K);
                if constexpr (!USE_FULL_LOAD_A) {
                    l1AListIdNext = kL1IdxNext % L1A_STAGES;
                    LoadAPanel(
                        tensorBlockA, l1ATensorList[l1AListIdNext], l1AEventList[l1AListIdNext], kL1IdxNext * L1_TILE_K,
                        kL1ActualNext, mL1Actual, copyGmToL1A);
                }
                LoadBPanel(
                    tensorBlockB0, l1B0TensorList[l1B0ListIdNext], l1B0EventList[l1B0ListIdNext],
                    kL1IdxNext * L1_TILE_K, kL1ActualNext, nL1Actual, copyGmToL1B0);
                LoadBPanel(
                    tensorBlockB1, l1B1TensorList[l1B1ListIdNext], l1B1EventList[l1B1ListIdNext],
                    kL1IdxNext * L1_TILE_K, kL1ActualNext, nL1Actual, copyGmToL1B1);
            }

            auto tensorL1B0 = tla::MakeTensor(l1B0TensorList[l1B0ListId], L1B_LAYOUT, Arch::PositionL1{});
            auto tensorL1B1 = tla::MakeTensor(l1B1TensorList[l1B1ListId], L1B_LAYOUT, Arch::PositionL1{});
            uint32_t kL0Loop = CeilDiv<L0_TILE_K>(kL1Actual);

            for (uint32_t mL0Idx = 0; mL0Idx < mL0Loop; ++mL0Idx) {
                uint32_t mL0Actual = (mL0Idx < mL0Loop - 1) ? L0_TILE_M : (mL1Actual - mL0Idx * L0_TILE_M);

                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; ++kL0Idx) {
                    uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (kL1Actual - kL0Idx * L0_TILE_K);
                    auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mL0Actual, kL0Actual);
                    auto tensorL0A = tla::MakeTensor(l0ATensorList[l0AListId], layoutAInL0, Arch::PositionL0A{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    if constexpr (USE_FULL_LOAD_A) {
                        auto l1ALayout =
                            tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, actualShape.k());
                        auto tensorL1A = tla::MakeTensor(l1ATensorList[0], l1ALayout, Arch::PositionL1{});
                        auto tensorTileL1A = GetTileA(
                            tensorL1A, mL0Idx * L0_TILE_M, kL0Idx * L0_TILE_K + kL1Idx * L1_TILE_K, mL0Actual,
                            kL0Actual);
                        copyL1ToL0A(tensorL0A, tensorTileL1A);
                    } else {
                        if ((mL0Idx == 0) && (kL0Idx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
                        }
                        auto tensorL1A = tla::MakeTensor(l1ATensorList[l1AListId], L1A_LAYOUT, Arch::PositionL1{});
                        auto tensorTileL1A =
                            GetTileA(tensorL1A, mL0Idx * L0_TILE_M, kL0Idx * L0_TILE_K, mL0Actual, kL0Actual);
                        copyL1ToL0A(tensorL0A, tensorTileL1A);
                        if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                        }
                    }

                    bool initC = ((kStep == 0) && (kL0Idx == 0));
                    for (uint32_t nL0Idx = 0; nL0Idx < nL0Loop; ++nL0Idx) {
                        uint32_t nL0Actual = (nL0Idx < nL0Loop - 1) ? L0_TILE_N : (nL1Actual - nL0Idx * L0_TILE_N);
                        uint8_t unitFlag = 0;
                        if constexpr (L0_TILE_N == L1_TILE_N) {
                            unitFlag = 0b10;
                            if ((kStep == kL1Loop - 1) && (mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1) &&
                                (nL0Idx == nL0Loop - 1)) {
                                unitFlag = 0b11;
                            }
                        }

                        RunPair(
                            tensorL1B0, tensorL1B1, tensorL0C0, tensorL0C1, tensorL0A, l0B0TensorList, l0B1TensorList,
                            l0B0ListId, l0B1ListId, l1B0ListId, l1B1ListId, GemmCoord{mL0Idx, nL0Idx, kL0Idx},
                            GemmCoord{mL0Actual, nL0Actual, kL0Actual}, GemmCoord{mL0Loop, nL0Loop, kL0Loop}, initC,
                            unitFlag);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
                }
            }

            l1AListId = l1AListIdNext;
            l1B0ListId = l1B0ListIdNext;
            l1B1ListId = l1B1ListIdNext;
            kL1Idx = kL1IdxNext;
            kL1Actual = kL1ActualNext;
        }

        if constexpr (USE_FULL_LOAD_A) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[0]);
        }

        if constexpr (L0_TILE_N != L1_TILE_N) {
            constexpr int32_t mmadDoneEventId = MMAD_DONE_EVENT_ID;
            constexpr int32_t fixM0EventId = L0C0_FIX_DONE_EVENT_ID;
            constexpr int32_t fixM1EventId = L0C1_FIX_DONE_EVENT_ID;
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mmadDoneEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mmadDoneEventId);
            copyL0CToUb0(tensorBlockD0, tensorL0C0, 0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(fixM0EventId);
            copyL0CToUb1(tensorBlockD1, tensorL0C1, 0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(fixM1EventId);
        } else {
            copyL0CToUb0(tensorBlockD0, tensorL0C0, 0b11);
            copyL0CToUb1(tensorBlockD1, tensorL0C1, 0b11);
        }
    }

    template <class TileMmadType, class TensorL0C, class TensorL0A, class TensorL0B>
    CATLASS_DEVICE static void RunMmad(
        TileMmadType& tileMmad, TensorL0C& tensorL0C, TensorL0A& tensorL0A, TensorL0B& tensorL0B, uint32_t nL0Idx,
        uint32_t mActual, uint32_t nActual, uint32_t kActual, bool initC, uint8_t unitFlag)
    {
        if constexpr (L0_TILE_N == L1_TILE_N) {
            tileMmad(tensorL0C, tensorL0A, tensorL0B, mActual, nActual, kActual, initC, unitFlag);
            return;
        }

        constexpr uint32_t fractalN = C0_NUM_PER_FRACTAL;
        uint32_t mAligned = RoundUp<C0_NUM_PER_FRACTAL>(mActual);
        uint32_t l0COffset = (nL0Idx * L0_TILE_N / fractalN) * mAligned * fractalN;
        AscendC::MmadParams params;
        params.m = mActual;
        params.n = nActual;
        params.k = kActual;
        params.unitFlag = unitFlag;
        params.cmatrixInitVal = initC;
        params.disableGemv = true;
        AscendC::Mmad(tensorL0C.data()[l0COffset], tensorL0A.data(), tensorL0B.data(), params);
        if ((mActual / fractalN) * (nActual / fractalN) < 10) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};

} // namespace Catlass::Gemm::Block
