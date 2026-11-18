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
#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_PV_SPLIT_ROW_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_PV_SPLIT_ROW_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/detail/callback.hpp"

////////////////////////////////////////////////////////////////////

namespace Catlass::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <
    class ArchTag_, bool ENABLE_PAGED_KV_CACHE_, bool ENABLE_UNIT_FLAG_, class L1TileShape_, class L0TileShape_,
    class ElementA_, class ElementB_, class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<
    MmadHstuPV<ArchTag_, ENABLE_PAGED_KV_CACHE_, ENABLE_UNIT_FLAG_>, L1TileShape_, L0TileShape_, ElementA_, ElementB_,
    ElementC_, ElementBias_, TileCopy_, TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadHstuPV<ArchTag_, ENABLE_PAGED_KV_CACHE_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutTagA = typename TileCopy::LayoutTagA;
    using LayoutA = typename TileCopy::LayoutA;

    using ElementB = ElementB_;
    using LayoutTagB = typename TileCopy::LayoutTagB;
    using LayoutB = typename TileCopy::LayoutB;
    using TensorB =
        tla::Tensor<AscendC::GlobalTensor<ElementB>, LayoutB, tla::Coord<uint32_t, uint32_t>, AscendC::TPosition::GM>;

    using ElementC = ElementC_;
    using LayoutTagC = typename TileCopy::LayoutTagC;
    using LayoutC = typename TileCopy::LayoutC;

    using ElementL1A = ElementA;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutL1A = typename TileCopy::LayoutL1A;

    using ElementL1B = ElementB;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutL1B = typename TileCopy::LayoutL1B;

    using ElementL0A = ElementA;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutL0A = typename TileCopy::LayoutL0A;

    using ElementL0B = ElementB;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using LayoutL0B = typename TileCopy::LayoutL0B;

    using ElementL0C = typename TileCopy::ElementAccumulator;

    using TileMmad = TileMmad_;
    using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    static constexpr uint32_t L1A_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr uint32_t UBC_STAGES = DispatchPolicy::UBC_STAGES;
    static constexpr uint32_t PRELOAD_STAGES = 1;

    static constexpr bool ENABLE_L1_RESIDENT = false;
    static constexpr bool ENABLE_UNIT_FLAG = false;
    static constexpr bool ENABLE_RESIDEND_L0C = DispatchPolicy::ENABLE_PV_RESIDENT_L0C;
    static constexpr bool ENABLE_PAGED_KV_CACHE = DispatchPolicy::ENABLE_PAGED_KV_CACHE;

    static_assert(
        tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
        "L1TileShape must be tla::tuple and static!");
    static_assert(
        tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
        "L0TileShape must be tla::tuple and static!");

    // Check LayoutC
    static_assert(
        tla::detail::isRowMajor<LayoutC>::value ||
            ((std::is_same_v<ElementC, half> || std::is_same_v<ElementC, bfloat16_t> ||
              std::is_same_v<ElementC, float>) &&
             tla::detail::iszN<ElementC, LayoutC>::value),
        "LayoutC only supports zN in half or bfloat16 or float, RowMajor in all dtype yet!");

    static constexpr uint32_t ELE_NUM_PER_C0 = 16;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 256;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L1 tile size
    static constexpr uint32_t L1A_DATA_PING_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1A_BUFF_PING_SIZE = RoundUp(L1A_DATA_PING_SIZE, 1024);
    static constexpr uint32_t L1A_BUFF_SIZE = L1A_BUFF_PING_SIZE * L1A_STAGES;

    static constexpr uint32_t L1B_DATA_PING_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    static constexpr uint32_t L1B_BUFF_PING_SIZE = RoundUp(L1B_DATA_PING_SIZE, 1024);
    static constexpr uint32_t L1B_BUFF_SIZE = L1B_BUFF_PING_SIZE * L1B_STAGES;

    // Check L1TileShape
    static_assert(L1A_BUFF_SIZE + L1B_BUFF_SIZE <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    // L0 tile size
    static constexpr uint32_t L0A_DATA_PING_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0A_BUFF_PING_SIZE = RoundUp(L0A_DATA_PING_SIZE, 1024);
    static constexpr uint32_t L0A_BUFF_SIZE = L0A_BUFF_PING_SIZE * L0A_STAGES;
    static constexpr uint32_t L0A_BUFF_OFFSET = 0;

    static constexpr uint32_t L0B_DATA_PING_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0B_BUFF_PING_SIZE = RoundUp(L0B_DATA_PING_SIZE, 1024);
    static constexpr uint32_t L0B_BUFF_SIZE = L0B_BUFF_PING_SIZE * L0B_STAGES;
    static constexpr uint32_t L0B_BUFF_OFFSET = 0;

    static constexpr uint32_t L0C_DATA_PING_SIZE = L0_TILE_M * L0_TILE_N * sizeof(ElementL0C);
    static constexpr uint32_t L0C_BUFF_PING_SIZE = RoundUp(L0C_DATA_PING_SIZE, 1024);
    static constexpr uint32_t L0C_BUFF_SIZE = L0C_BUFF_PING_SIZE * L0C_STAGES;

    // Check L0TileShape
    static_assert(L0A_BUFF_SIZE <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_BUFF_SIZE <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_BUFF_SIZE <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    CATLASS_DEVICE
    BlockMmadTla(
        Arch::Resource<ArchTag>& resource, uint32_t l1AOffset, uint32_t l1APingSize, uint32_t l1BOffset,
        uint32_t l1BPingSize, uint32_t l0AOffset, uint32_t l0APingSize, uint32_t l0BOffset, uint32_t l0BPingSize,
        uint32_t l0COffset, uint32_t l0CPingSize)
    {
        // Allocate L1 memory space
        InitL1(resource, l1AOffset, l1APingSize, l1BOffset, l1BPingSize);
        InitL0(resource, l0AOffset, l0APingSize, l0BOffset, l0BPingSize, l0COffset, l0CPingSize);
    }

    CATLASS_DEVICE
    ~BlockMmadTla()
    {}

    CATLASS_DEVICE
    void InitL1(
        Arch::Resource<ArchTag>& resource, uint32_t l1AOffset, uint32_t l1APingSize, uint32_t l1BOffset,
        uint32_t l1BPingSize)
    {
        uint32_t L1A_EVENT_ID_START = 2;
        uint32_t L1B_EVENT_ID_START = 4;

        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + l1APingSize * i);
            l1AEventList[i] = L1A_EVENT_ID_START + i;
        }
        for (uint32_t i = 0; i < L1B_STAGES; ++i) {
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + l1BPingSize * i);
            l1BEventList[i] = L1B_EVENT_ID_START + i;
        }
    }

    CATLASS_DEVICE
    void InitL0(
        Arch::Resource<ArchTag>& resource, uint32_t l0AOffset, uint32_t l0APingSize, uint32_t l0BOffset,
        uint32_t l0BPingSize, uint32_t l0COffset, uint32_t l0CPingSize)
    {
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(l0AOffset + l0APingSize * i);
            l0AEventList[i] = i;
        }

        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(l0BOffset + l0BPingSize * i);
            l0BEventList[i] = i + L0A_STAGES;
        }

        uint32_t L0C_EVENT_ID_START;
        if constexpr (!ENABLE_RESIDEND_L0C) {
            L0C_EVENT_ID_START = 0;
        } else {
            L0C_EVENT_ID_START = 4;
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementL0C>(l0COffset + l0CPingSize * i);
            l0CEventList[i] = L0C_EVENT_ID_START + i;
        }
    }

    CATLASS_DEVICE void InitEvent(Arch::Resource<ArchTag>& resource)
    {
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        if constexpr (!ENABLE_RESIDEND_L0C) {
        } else {
            for (uint32_t i = 0; i < L0C_STAGES; ++i) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
            }
        }
    }

    CATLASS_DEVICE void ClearEvent()
    {
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        if constexpr (!ENABLE_RESIDEND_L0C) {
        } else {
            for (uint32_t i = 0; i < L0C_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
            }
        }
    }

    template <class TensorL1B, class PadgeTensorB>
    CATLASS_DEVICE void CopyPagedGmBToL1B(TensorL1B& tensorL1B, PadgeTensorB& pagedTensorB)
    {
        auto& tensorPagedGmB = pagedTensorB.tensor;
        uint32_t headIdx = pagedTensorB.headIdx;
        MatrixCoord coordBTopLeftDot = pagedTensorB.coordBTopLeftDot;
        AscendC::GlobalTensor<uint32_t> gmBlockTable = pagedTensorB.blockTable;

        uint32_t blockSize = tla::get<1>(tensorPagedGmB.shape());
        uint32_t headNum = tla::get<2>(tensorPagedGmB.shape());
        uint32_t headDim = tla::get<3>(tensorPagedGmB.shape());
        uint32_t vSeqlen = tla::get<0>(tensorL1B.originShape());

        uint32_t startVseqIdx = coordBTopLeftDot.row();
        uint32_t startBlockIdx = startVseqIdx / blockSize;

        uint32_t endVseqIdx = startVseqIdx + vSeqlen - 1;
        uint32_t endBlockIdx = endVseqIdx / blockSize;

        for (uint32_t idx = startBlockIdx; idx <= endBlockIdx; ++idx) {
            uint32_t localStartVseqIdx = (idx == startBlockIdx) ? startVseqIdx % blockSize : 0;
            uint32_t localEndVseqIdx = (idx == endBlockIdx) ? endVseqIdx % blockSize : blockSize - 1;
            uint32_t localVSeqlen = localEndVseqIdx - localStartVseqIdx + 1;

            auto tensorL1Tile = GetTile(
                tensorL1B, tla::MakeCoord((idx - startBlockIdx) * blockSize, 0), tla::MakeShape(localVSeqlen, headDim));

            uint32_t blockId = gmBlockTable[idx].GetValue(0);
            auto layoutPaged = tensorPagedGmB.layout();

            int64_t offset = layoutPaged(tla::MakeCoord(blockId, localStartVseqIdx, headIdx, 0));

            auto shape = tla::MakeShape(localVSeqlen, headDim);
            auto strideInPage = tla::MakeStride((int64_t)headDim * headNum, Int<1>{});
            auto layoutInPage = tla::MakeLayout(shape, strideInPage, shape);
            auto tensorGmTile = tla::MakeTensor(tensorPagedGmB.data()[offset], layoutInPage, Arch::PositionGM{});

            copyGmToL1B(tensorL1Tile, tensorGmTile);
        }
    }

    template <class TensorA, class DyncTensorB, class TensorC>
    CATLASS_DEVICE void computePV(
        TensorC& tensorC, TensorA& tensorA, DyncTensorB& tensorB, GemmCoord actualShape, GemmCoord const& l1TileShape,
        GemmCoord const& l0TileShape, uint32_t cvBlockFlag, uint32_t l1AListId,
        Callback const& callbackBeforeFixpipe = {}, Callback const& callbackAfterFixpipe = {})
    {
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;

        uint32_t l1TileK = l1TileShape.k();

        auto l1ALayout = tensorA.layout();

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();

        uint32_t mL1Actual = mBlockActual;
        uint32_t nL1Actual = nBlockActual;

        uint32_t kL1Loop = CeilDiv(kBlockActual, l1TileK);

        callbackBeforeFixpipe();

        uint32_t startTileIdx = 0;
        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; ++kL1Idx) {
            uint32_t kL1TileIdx =
                (startTileIdx + kL1Idx < kL1Loop) ? (startTileIdx + kL1Idx) : (startTileIdx + kL1Idx - kL1Loop);

            uint32_t kL1Actual = (kL1TileIdx < kL1Loop - 1) ? l1TileK : (kBlockActual - kL1TileIdx * l1TileK);

            auto tensorL1A =
                GetTile(tensorA, tla::MakeCoord(0, kL1Idx * l1TileK), tla::MakeShape(mL1Actual, kL1Actual));

            auto l1BLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(kL1Actual, nL1Actual);
            auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], l1BLayout, Arch::PositionL1{});

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
            if constexpr (!ENABLE_PAGED_KV_CACHE) {
                auto tensorTileB =
                    GetTile(tensorB, tla::MakeCoord(kL1TileIdx * l1TileK, 0), tla::MakeShape(kL1Actual, nL1Actual));
                copyGmToL1B(tensorL1B, tensorTileB);
            } else {
                CopyPagedGmBToL1B(tensorL1B, tensorB);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);

            auto& l1TileMmadParams = l1TileMmadParamsList[l1TileMmadParamsId];
            l1TileMmadParams.l1A = tensorL1A.data();
            tla::get<0>(l1TileMmadParams.coordL1A) = tla::get<0>(tensorL1A.coord());
            tla::get<1>(l1TileMmadParams.coordL1A) = tla::get<1>(tensorL1A.coord());

            l1TileMmadParams.l1B = l1BTensorList[l1BListId];

            l1TileMmadParams.mBlockActual = mBlockActual;
            l1TileMmadParams.nBlockActual = nBlockActual;
            l1TileMmadParams.kBlockActual = kBlockActual;

            l1TileMmadParams.mL1Actual = mL1Actual;
            l1TileMmadParams.nL1Actual = nL1Actual;
            l1TileMmadParams.kL1Actual = kL1Actual;

            l1TileMmadParams.l1TileShape = l1TileShape;
            l1TileMmadParams.l0TileShape = l0TileShape;

            l1TileMmadParams.l1AListId = l1AListId;
            l1TileMmadParams.l1BListId = l1BListId;

            l1TileMmadParams.isKLoopFirst = (kL1Idx == 0);
            l1TileMmadParams.isKLoopLast = (kL1Idx == kL1Loop - 1);
            l1TileMmadParams.cvBlockFlag = cvBlockFlag;

            if (kL1Idx == kL1Loop - 1) {
                l1TileMmadParams.gmC = tensorC.data();
                l1TileMmadParams.layoutGmC = tensorC.layout();
                tla::get<0>(l1TileMmadParams.coordGmC) = tla::get<0>(tensorC.coord());
                tla::get<1>(l1TileMmadParams.coordGmC) = tla::get<1>(tensorC.coord());

                l1TileMmadParams.callbackBeforeFixpipe = {};
                l1TileMmadParams.callbackAfterFixpipe = {};
            }

            L1TileMmad<TensorC>(l1TileMmadParamsList[l1TileMmadParamsId]);

            l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;

            l1BListId = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;
        }
    }

    struct L1TileMmadParams {
        AscendC::LocalTensor<ElementA> l1A;
        LayoutA layoutL1A;
        tla::tuple<uint32_t, uint32_t> coordL1A;

        AscendC::LocalTensor<ElementB> l1B;
        LayoutB layoutL1B;
        tla::tuple<uint32_t, uint32_t> coordL1B;

        AscendC::GlobalTensor<ElementC> gmC;
        LayoutC layoutGmC;
        tla::tuple<uint32_t, uint32_t> coordGmC;

        uint32_t mBlockActual;
        uint32_t nBlockActual;
        uint32_t kBlockActual;

        uint32_t mL1Actual;
        uint32_t nL1Actual;
        uint32_t kL1Actual;

        uint32_t l1AListId;
        uint32_t l1BListId;
        uint32_t l1CListId;

        bool isKLoopFirst;
        bool isKLoopLast;

        uint32_t cvBlockFlag;

        GemmCoord l1TileShape;
        GemmCoord l0TileShape;

        Callback callbackBeforeFixpipe;
        Callback callbackAfterFixpipe;

        CATLASS_DEVICE
        L1TileMmadParams() = default;
    };

    template <class TensorC>
    CATLASS_DEVICE void L1TileMmad(L1TileMmadParams const& params)
    {
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;

        uint32_t l0TileK = params.l0TileShape.k();

        auto& l1ATensor = params.l1A;
        auto l1ALayout = tla::MakeLayout<ElementA, LayoutTagL1A>(params.mBlockActual, params.kBlockActual);
        auto tensorL1A = tla::MakeTensor(l1ATensor, l1ALayout, params.coordL1A, Arch::PositionL1{});

        auto& l1BTensor = params.l1B;
        auto l1BLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(params.kL1Actual, params.nL1Actual);
        auto tensorL1B = tla::MakeTensor(l1BTensor, l1BLayout, Arch::PositionL1{});

        auto& l0CTensor = l0CTensorList[l0CListId];
        auto layoutInL0C = tla::MakeLayoutL0C(params.mL1Actual, params.nL1Actual);
        auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});

        uint32_t cvBlockFlag = params.cvBlockFlag;

        uint32_t kL0Loop = AscendC::CeilDivision(params.kL1Actual, l0TileK);

        for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; ++kL0Idx) {
            uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? l0TileK : (params.kL1Actual - kL0Idx * l0TileK);

            auto& l0ATile = l0ATensorList[l0AListId];
            auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(params.mL1Actual, kL0Actual);
            auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});
            auto tensorTileL1A =
                GetTile(tensorL1A, tla::MakeCoord(0, kL0Idx * l0TileK), tla::MakeShape(params.mL1Actual, kL0Actual));

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);

            copyL1ToL0A(tensorL0A, tensorTileL1A);

            auto& l0BTile = l0BTensorList[l0BListId];
            auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, params.nL1Actual);
            auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});

            auto tensorTileL1B =
                GetTile(tensorL1B, tla::MakeCoord(kL0Idx * l0TileK, 0), tla::MakeShape(kL0Actual, params.nL1Actual));

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
            if (kL0Idx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[params.l1BListId]);
            }

            copyL1ToL0B(tensorL0B, tensorTileL1B);

            if (kL0Idx == kL0Loop - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[params.l1BListId]);
            }

            bool initC;
            if constexpr (!ENABLE_RESIDEND_L0C) {
                initC = (params.isKLoopFirst && (kL0Idx == 0));
            } else {
                initC = ((cvBlockFlag & 0b01) == 0b01) && (params.isKLoopFirst && (kL0Idx == 0));
            }

            // If the unit flag is enabled, the unit flag is set according to the calculation progress
            uint8_t unitFlag = 0b00;
            if constexpr (ENABLE_UNIT_FLAG) {
                if (params.isKLoopLast && (kL0Idx == kL0Loop - 1)) {
                    unitFlag = 0b11;
                } else {
                    unitFlag = 0b10;
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

            if constexpr (!ENABLE_UNIT_FLAG) {
                bool isWaitL0CIdle;
                if constexpr (!ENABLE_RESIDEND_L0C) {
                    isWaitL0CIdle = params.isKLoopFirst && (kL0Idx == 0);
                } else {
                    isWaitL0CIdle = ((cvBlockFlag & 0b01) == 0b01) && params.isKLoopFirst && (kL0Idx == 0);
                }
                if (isWaitL0CIdle) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
                }
            }

            tileMmad(tensorL0C, tensorL0A, tensorL0B, params.mL1Actual, params.nL1Actual, kL0Actual, initC, unitFlag);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);

            l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
            l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
        }

        bool isCopyToDst;
        if constexpr (!ENABLE_RESIDEND_L0C) {
            isCopyToDst = params.isKLoopLast;
        } else {
            isCopyToDst = ((cvBlockFlag & 0b10) == 0b10) && params.isKLoopLast;
        }
        if (isCopyToDst) {
            auto layoutCInGm = params.layoutGmC;
            auto tensorTileC =
                tla::MakeTensor(params.gmC, layoutCInGm, params.coordGmC, Arch::PositionType<TensorC::position>{});

            if constexpr (!ENABLE_RESIDEND_L0C) {
                if ((cvBlockFlag & 0b01) == 0) {
                    AscendC::SetAtomicAdd<ElementC>();
                    AscendC::PipeBarrier<PIPE_FIX>();
                }
            }

            if constexpr (!ENABLE_UNIT_FLAG) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                copyL0CToDst(tensorTileC, tensorL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            } else {
                copyL0CToDst(tensorTileC, tensorL0C, 0b11);
            }
            if constexpr (!ENABLE_RESIDEND_L0C) {
                if ((cvBlockFlag & 0b01) == 0) {
                    AscendC::DisableDmaAtomic();
                    AscendC::PipeBarrier<PIPE_FIX>();
                }
            }

            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensorList[L1A_STAGES];
    int32_t l1AEventList[L1A_STAGES];

    AscendC::LocalTensor<ElementB> l1BTensorList[L1B_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    uint32_t l1BListId{0};

    AscendC::LocalTensor<ElementA> l0ATensorList[L0A_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    uint32_t l0AListId{0};

    AscendC::LocalTensor<ElementB> l0BTensorList[L0B_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    uint32_t l0BListId{0};

    AscendC::LocalTensor<ElementL0C> l0CTensorList[L0C_STAGES];
    int32_t l0CEventList[L0C_STAGES];
    uint32_t l0CListId{0};

    L1TileMmadParams l1TileMmadParamsList[PRELOAD_STAGES];
    uint32_t l1TileMmadParamsId{0};

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    CopyGmToL1B copyGmToL1B;
};

////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Block

#endif
