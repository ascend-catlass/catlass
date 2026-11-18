/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_SYMM_LEFT_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_SYMM_LEFT_TLA_HPP

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
struct SymmMmadPingpongLeftTlaDispatchChecker {
    static constexpr bool value = false;
};

template <bool ENABLE_UNIT_FLAG>
struct SymmMmadPingpongLeftTlaDispatchChecker<MmadAtlasA2PingpongSymmLeft<ENABLE_UNIT_FLAG>> {
    static constexpr bool value = true;
};

template <class ArchTag, bool ENABLE_UNIT_FLAG>
struct SymmMmadPingpongLeftTlaDispatchChecker<MmadPingpongSymmLeft<ArchTag, ENABLE_UNIT_FLAG>> {
    static constexpr bool value = true;
};

template <class DispatchPolicy>
struct SymmLeftTlaStagesGetter {
    static constexpr uint32_t STAGES = 1;
};

template <bool ENABLE_UNIT_FLAG>
struct SymmLeftTlaStagesGetter<MmadAtlasA2PingpongSymmLeft<ENABLE_UNIT_FLAG>> {
    static constexpr uint32_t STAGES = 2;
};

template <class ArchTag, bool ENABLE_UNIT_FLAG>
struct SymmLeftTlaStagesGetter<MmadPingpongSymmLeft<ArchTag, ENABLE_UNIT_FLAG>> {
    static constexpr uint32_t STAGES = 2;
};

template <
    class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
    class ElementBias_ = void,
    class TileCopyP_ = Gemm::Tile::PackedTileCopyTla<
        typename DispatchPolicy_::ArchTag, ElementA_, layout::RowMajor, ElementB_, layout::RowMajor, ElementC_,
        layout::RowMajor>,
    class TileCopyQ_ = Gemm::Tile::PackedTileCopyTla<
        typename DispatchPolicy_::ArchTag, ElementA_, layout::ColumnMajor, ElementB_, layout::RowMajor, ElementC_,
        layout::RowMajor>,
    class TileMmadP_ =
        Gemm::Tile::TileMmadTla<typename DispatchPolicy_::ArchTag, ElementA_, typename TileCopyP_::LayoutTagL1A>,
    class TileMmadQ_ =
        Gemm::Tile::TileMmadTla<typename DispatchPolicy_::ArchTag, ElementA_, typename TileCopyQ_::LayoutTagL1A>>
struct BlockMmadPingpongSymmLeftTla {
public:
    using DispatchPolicy = DispatchPolicy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    static_assert(
        SymmMmadPingpongLeftTlaDispatchChecker<DispatchPolicy_>::value,
        "BlockMmadPingpongSymmLeftTla requires MmadAtlasA2PingpongSymmLeft or MmadPingpongSymmLeft dispatch policy");
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
    using LayoutAP = typename TileCopyP::LayoutA;
    using LayoutAQ = typename TileCopyQ::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopyP::LayoutB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopyP::LayoutC;
    using ElementBias = ElementBias_;

    using TileMmadP = TileMmadP_;
    using TileMmadQ = TileMmadQ_;

    using ElementAccumulator = typename TileCopyP::ElementAccumulator;

    using LayoutTagL1AP = typename TileCopyP::LayoutTagL1A;
    using LayoutTagL1AQ = typename TileCopyQ::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopyP::LayoutTagL1B;
    using LayoutTagL0AP = typename TileCopyP::LayoutTagL0A;
    using LayoutTagL0AQ = typename TileCopyQ::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopyP::LayoutTagL0B;

    using L1AAlignHelper = typename TileCopyP::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopyP::L1BAlignHelper;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t STAGES = SymmLeftTlaStagesGetter<DispatchPolicy>::STAGES;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L1 tile sizes
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    // L0 tile sizes
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    // L0 ping-pong buffer size
    static constexpr uint32_t L0A_BUF_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_BUF_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_BUF_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_BUF_SIZE / STAGES;

    static_assert(
        std::is_same_v<ArchTag, Arch::AtlasA2> || std::is_same_v<ArchTag, Arch::Ascend950>,
        "ArchTag can only be AtlasA2 or Ascend950!");
    static_assert(L1_TILE_M == L1_TILE_K, "Left-symm path requires square M/K tiles.");
    static_assert(tla::detail::isRowMajor<LayoutC>::value, "LayoutC only supports RowMajor!");
    static_assert((L1A_TILE_SIZE + L1B_TILE_SIZE) * STAGES <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");
    static_assert(L0A_TILE_SIZE * STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");
    static_assert(
        L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K cannot exceed L1TileShape::K");

    // Compile-time L1 layouts
    static constexpr auto L1AP_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1AP>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1AQ_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1AQ>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});

    /// Constructor
    CATLASS_DEVICE
    BlockMmadPingpongSymmLeftTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0)
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

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadPingpongSymmLeftTla()
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

    /// Perform a block-scoped symmetric left matrix multiply-accumulate.
    template <bool UPPER_STORAGE, class TensorA, class TensorAQ, class TensorB, class TensorC>
    CATLASS_DEVICE void operator()(
        TensorA& tensorA, TensorAQ& tensorAQ, TensorB& tensorB, TensorC& tensorC, GemmCoord const& actualShape,
        uint32_t iTile, uint32_t iTileNext, uint32_t tilesM, bool isFirstBlock = true, bool hasNextBlock = false,
        TensorA const* tensorANext = nullptr, TensorAQ const* tensorAQNext = nullptr, TensorB* tensorBNext = nullptr,
        GemmCoord const& actualShapeNext = {})
    {
        uint32_t mBlockActual = actualShape.m();
        uint32_t nBlockActual = actualShape.n();

        using CopyGmToL1AP = typename TileCopyP::template CopyGmToL1A<TensorA>;
        using CopyGmToL1AQ = typename TileCopyQ::template CopyGmToL1A<TensorAQ>;
        using CopyGmToL1B = typename TileCopyP::template CopyGmToL1B<TensorB>;
        CopyGmToL1AP copyGmToL1AP;
        CopyGmToL1AQ copyGmToL1AQ;
        CopyGmToL1B copyGmToL1B;

        auto layoutInL0C = tla::MakeLayoutL0C(mBlockActual, nBlockActual);
        auto tensorL0C_Tile = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});

        // ── Preload the first k-tile ──
        if (isFirstBlock) {
            uint32_t kActual = GetKActual(actualShape, 0);
            PreloadKTile<UPPER_STORAGE>(
                l1ListId, tensorA, tensorAQ, tensorB, iTile, 0, mBlockActual, nBlockActual, kActual, copyGmToL1AP,
                copyGmToL1AQ, copyGmToL1B);
        }

        bool initAccum = true;
        for (uint32_t kTile = 0; kTile < tilesM; ++kTile) {
            uint32_t l1ListIdCurrent = l1ListId;
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;

            // ── Preload next k-tile, or next block's first k-tile ──
            if (kTile + 1 < tilesM) {
                uint32_t kActualNext = GetKActual(actualShape, kTile + 1);
                PreloadKTile<UPPER_STORAGE>(
                    l1ListIdNext, tensorA, tensorAQ, tensorB, iTile, kTile + 1, mBlockActual, nBlockActual, kActualNext,
                    copyGmToL1AP, copyGmToL1AQ, copyGmToL1B);
            } else if (hasNextBlock && tensorANext != nullptr && tensorAQNext != nullptr && tensorBNext != nullptr) {
                uint32_t kActualNext = GetKActual(actualShape, 0);
                PreloadKTile<UPPER_STORAGE>(
                    l1ListIdNext, *tensorANext, *tensorAQNext, *tensorBNext, iTileNext, 0, actualShapeNext.m(),
                    actualShapeNext.n(), kActualNext, copyGmToL1AP, copyGmToL1AQ, copyGmToL1B);
            }

            // ── Compute the current k-tile ──
            uint32_t kActual = GetKActual(actualShape, kTile);
            bool finalize = (kTile + 1 == tilesM);
            if constexpr (UPPER_STORAGE) {
                if (kTile < iTile) {
                    AccumulateLoaded<false>(
                        l1ListIdCurrent, mBlockActual, nBlockActual, kActual, initAccum, finalize, tensorL0C_Tile);
                } else {
                    AccumulateLoaded<true>(
                        l1ListIdCurrent, mBlockActual, nBlockActual, kActual, initAccum, finalize, tensorL0C_Tile);
                }
            } else {
                if (kTile <= iTile) {
                    AccumulateLoaded<true>(
                        l1ListIdCurrent, mBlockActual, nBlockActual, kActual, initAccum, finalize, tensorL0C_Tile);
                } else {
                    AccumulateLoaded<false>(
                        l1ListIdCurrent, mBlockActual, nBlockActual, kActual, initAccum, finalize, tensorL0C_Tile);
                }
            }
            initAccum = false;
            l1ListId = l1ListIdNext;
        }
    }

    /// Store the accumulated result from L0C to GM
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

    /// Copy an L1A tile to L0A with transpose enabled for the secondary symmetric operand.
    /// The secondary operand is represented with a ColumnMajor GM layout and an nZ L1 layout;
    /// this helper keeps the transpose handling explicit while preserving the TLA tensor-based copy path.
    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void CopyL1ToL0A_Transpose(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow =
            CeilDiv(tla::get<0>(dstTensor.originShape()), tla::get<0, 0>(dstTensor.shape()));
        const uint32_t dstOuterShapeCol =
            CeilDiv(tla::get<1>(dstTensor.originShape()), tla::get<1, 0>(dstTensor.shape()));
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = dstOuterShapeCol;
        loadDataParams.srcStride = 1;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
            AscendC::LoadData(
                dstTensor.data()[dstOffset + i * dstOuterStrideRow],
                srcTensor.data()[srcOffset + i * srcOuterStrideRow], loadDataParams);
        }
    }

    template <
        bool UPPER_STORAGE, class TensorA, class TensorAQ, class TensorB, class CopyGmToL1AP, class CopyGmToL1AQ,
        class CopyGmToL1B>
    CATLASS_DEVICE void PreloadKTile(
        uint32_t stageId, TensorA& tensorA, TensorAQ& tensorAQ, TensorB& tensorB, uint32_t iTile, uint32_t kTile,
        uint32_t mActual, uint32_t nActual, uint32_t kActual, CopyGmToL1AP& copyGmToL1AP, CopyGmToL1AQ& copyGmToL1AQ,
        CopyGmToL1B& copyGmToL1B)
    {
        // Load A (with symm-specific triangle logic)
        bool useTransposed = UPPER_STORAGE ? (kTile < iTile) : (kTile > iTile);
        if (useTransposed) {
            auto tensorL1A = tla::MakeTensor(l1ATensorList[stageId], L1AQ_LAYOUT, Arch::PositionL1{});
            auto tensorTileA =
                GetTile(tensorAQ, tla::MakeCoord(0, kTile * L1_TILE_K), tla::MakeShape(mActual, kActual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[stageId]);
            copyGmToL1AQ(tensorL1A, tensorTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[stageId]);
        } else if (kTile == iTile) {
            // Diagonal: load primary, then symmetry fixup
            auto tensorL1A = tla::MakeTensor(l1ATensorList[stageId], L1AP_LAYOUT, Arch::PositionL1{});
            auto tensorTileA = GetTile(tensorA, tla::MakeCoord(0, kTile * L1_TILE_K), tla::MakeShape(mActual, kActual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[stageId]);
            copyGmToL1AP(tensorL1A, tensorTileA);
            // In-place symmetry fixup: mirror stored triangle to unstored triangle
            auto l1ATensor = l1ATensorList[stageId];
            for (uint32_t r = 0; r < mActual; ++r) {
                for (uint32_t c = 0; c < r; ++c) {
                    if constexpr (UPPER_STORAGE) {
                        l1ATensor[L1AP_LAYOUT(tla::MakeCoord(r, c))] = l1ATensor[L1AP_LAYOUT(tla::MakeCoord(c, r))];
                    } else {
                        l1ATensor[L1AP_LAYOUT(tla::MakeCoord(c, r))] = l1ATensor[L1AP_LAYOUT(tla::MakeCoord(r, c))];
                    }
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[stageId]);
        } else {
            auto tensorL1A = tla::MakeTensor(l1ATensorList[stageId], L1AP_LAYOUT, Arch::PositionL1{});
            auto tensorTileA = GetTile(tensorA, tla::MakeCoord(0, kTile * L1_TILE_K), tla::MakeShape(mActual, kActual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[stageId]);
            copyGmToL1AP(tensorL1A, tensorTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[stageId]);
        }

        auto tensorL1B = tla::MakeTensor(l1BTensorList[stageId], L1B_LAYOUT, Arch::PositionL1{});
        auto tensorTileB = GetTile(tensorB, tla::MakeCoord(kTile * L1_TILE_K, 0), tla::MakeShape(kActual, nActual));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[stageId]);
        copyGmToL1B(tensorL1B, tensorTileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[stageId]);
    }

    template <bool USE_PRIMARY, class TensorL0C>
    CATLASS_DEVICE void AccumulateLoaded(
        uint32_t stageId, uint32_t mActual, uint32_t nActual, uint32_t kActual, bool init, bool finalize,
        TensorL0C& tensorL0C)
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mActual);
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nActual);

        using LayoutAInL1 = std::conditional_t<USE_PRIMARY, decltype(L1AP_LAYOUT), decltype(L1AQ_LAYOUT)>;
        uint32_t mPartLoop = CeilDiv<L0_TILE_M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0_TILE_N>(nRound);
        uint32_t kPartLoop = CeilDiv<L0_TILE_K>(kActual);

        // Create L1A/L1B tensors once per stage (outside inner loops, matching reference TLA pattern)
        auto tensorL1B = tla::MakeTensor(l1BTensorList[stageId], L1B_LAYOUT, Arch::PositionL1{});

        if constexpr (USE_PRIMARY) {
            auto tensorL1A = tla::MakeTensor(l1ATensorList[stageId], L1AP_LAYOUT, Arch::PositionL1{});
            AccumulateLoadedInner<true>(
                tensorL1A, tensorL1B, stageId, mActual, nActual, kActual, init, finalize, tensorL0C);
        } else {
            auto tensorL1A = tla::MakeTensor(l1ATensorList[stageId], L1AQ_LAYOUT, Arch::PositionL1{});
            AccumulateLoadedInner<false>(
                tensorL1A, tensorL1B, stageId, mActual, nActual, kActual, init, finalize, tensorL0C);
        }
    }

    // Inner loop body: tensorL1A/L1B created once outside, matching reference TLA pattern
    template <bool USE_PRIMARY, class TensorL1A, class TensorL1B, class TensorL0C>
    CATLASS_DEVICE void AccumulateLoadedInner(
        TensorL1A& tensorL1A, TensorL1B& tensorL1B, uint32_t stageId, uint32_t mActual, uint32_t nActual,
        uint32_t kActual, bool init, bool finalize, TensorL0C& tensorL0C)
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mActual);
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nActual);

        uint32_t mPartLoop = CeilDiv<L0_TILE_M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0_TILE_N>(nRound);
        uint32_t kPartLoop = CeilDiv<L0_TILE_K>(kActual);

        for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
            uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ? L0_TILE_M : (mRound - mPartIdx * L0_TILE_M);
            for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ? L0_TILE_K : (kActual - kPartIdx * L0_TILE_K);

                // L1A → L0A
                auto l0ATile = l0ATensorList[l0AListId];
                auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0AP>(mPartActual, kPartActual);
                auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                if ((mPartIdx == 0) && (kPartIdx == 0)) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[stageId]);
                }
                auto tensorTileL1A = GetTile(
                    tensorL1A, tla::MakeCoord(mPartIdx * L0_TILE_M, kPartIdx * L0_TILE_K),
                    tla::MakeShape(mPartActual, kPartActual));
                if constexpr (USE_PRIMARY) {
                    copyL1ToL0AP(tensorL0A, tensorTileL1A);
                } else {
                    copyL1ToL0AQ(tensorL0A, tensorTileL1A);
                }
                if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[stageId]);
                }

                for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                    uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ? L0_TILE_N : (nRound - nPartIdx * L0_TILE_N);

                    // L1B → L0B
                    auto l0BTile = l0BTensorList[l0BListId];
                    auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kPartActual, nPartActual);
                    auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
                    auto tensorTileL1B = GetTile(
                        tensorL1B, tla::MakeCoord(kPartIdx * L0_TILE_K, nPartIdx * L0_TILE_N),
                        tla::MakeShape(kPartActual, nPartActual));

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    if ((kPartIdx == 0) && (nPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[stageId]);
                    }
                    copyL1ToL0B(tensorL0B, tensorTileL1B);
                    if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[stageId]);
                    }

                    bool initC = init && (kPartIdx == 0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    auto tensorTileL0C = GetTile(
                        tensorL0C, tla::MakeCoord(mPartIdx * L0_TILE_M, nPartIdx * L0_TILE_N),
                        tla::MakeShape(mPartActual, nPartActual));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
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
                    } // already correct, USE_PRIMARY is template param of this method
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

    using CopyL1ToL0AP_Mem = typename TileCopyP::CopyL1ToL0A;
    using CopyL1ToL0AQ_Mem = typename TileCopyQ::CopyL1ToL0A;
    using CopyL1ToL0B_Mem = typename TileCopyP::CopyL1ToL0B;

    TileMmadP tileMmadP;
    TileMmadQ tileMmadQ;
    CopyL1ToL0AP_Mem copyL1ToL0AP;
    CopyL1ToL0AQ_Mem copyL1ToL0AQ;
    CopyL1ToL0B_Mem copyL1ToL0B;
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_SYMM_LEFT_TLA_HPP
