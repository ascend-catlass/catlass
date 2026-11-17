/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_DEQUANT_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_DEQUANT_HPP

#include <type_traits>

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_dequant.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Epilogue::Block {

using QuantMode = Catlass::Epilogue::QuantMode;

template <
    QuantMode X1_QUANT_MODE_, QuantMode X2_QUANT_MODE_, class TileShape_, class ElementIn_, class ElementOut_,
    class ElementBias_, class ElementX1Scale_, class ElementX2Scale_, class SplitTaskRatio_>
class BlockEpilogue<
    BlockEpilogueDequant<X1_QUANT_MODE_, X2_QUANT_MODE_>, TileShape_, ElementIn_, ElementOut_, ElementBias_,
    ElementX1Scale_, ElementX2Scale_, SplitTaskRatio_> {
public:
    using DispatchPolicy = BlockEpilogueDequant<X1_QUANT_MODE_, X2_QUANT_MODE_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementIn = ElementIn_;
    using ElementOut = ElementOut_;
    using ElementBias = ElementBias_;
    using ElementX1Scale = ElementX1Scale_;
    using ElementX2Scale = ElementX2Scale_;
    using ElementBiasStorage = std::conditional_t<std::is_void_v<ElementBias>, float, ElementBias>;

    using DataTypeIn = ElementIn;
    using DataTypeOut = ElementOut;
    using DataTypeBias = ElementBias;
    using DataTypeX1Scale = ElementX1Scale;
    using DataTypeX2Scale = ElementX2Scale;

    static constexpr QuantMode x1QuantMode = DispatchPolicy::x1QuantMode;
    static constexpr QuantMode x2QuantMode = DispatchPolicy::x2QuantMode;
    static constexpr bool hasBias = !std::is_void_v<ElementBias>;
    static constexpr int64_t l0M = tla::get<0>(TileShape_{});
    static constexpr int64_t l0N = tla::get<1>(TileShape_{});
    static constexpr uint32_t splitTaskRatio = SplitTaskRatio_::value;

    static_assert(splitTaskRatio == 1 || splitTaskRatio == 2, "task ratio must be 1:1 or 1:2");

    static_assert(
        x1QuantMode == QuantMode::NONE || x1QuantMode == QuantMode::PERTENSOR_MODE ||
            x1QuantMode == QuantMode::PERTOKEN_MODE,
        "Unsupported x1 quantization mode");
    static_assert(
        x2QuantMode == QuantMode::PERTENSOR_MODE || x2QuantMode == QuantMode::PERCHANNEL_MODE,
        "Unsupported x2 quantization mode");
    static_assert(
        (x1QuantMode == QuantMode::NONE &&
         (x2QuantMode == QuantMode::PERTENSOR_MODE || x2QuantMode == QuantMode::PERCHANNEL_MODE)) ||
            (x1QuantMode == QuantMode::PERTOKEN_MODE &&
             (x2QuantMode == QuantMode::PERTENSOR_MODE || x2QuantMode == QuantMode::PERCHANNEL_MODE)) ||
            (x1QuantMode == QuantMode::PERTENSOR_MODE && x2QuantMode == QuantMode::PERCHANNEL_MODE),
        "Unsupported quantization mode combination");

    CATLASS_DEVICE
    BlockEpilogue(
        Arch::Resource<ArchTag>& resource, GM_ADDR yGmAddr, GM_ADDR x2ScaleGmAddr, GM_ADDR x1ScaleGmAddr,
        GM_ADDR biasGmAddr, GemmCoord const& problemShape)
        : problemShape_(problemShape)
    {
        uint64_t mForSingleVec = CeilDiv(l0M, splitTaskRatio);
        uint32_t ubOffset = 0;
        l0cOutUb_ = resource.ubBuf.template GetBufferByByte<ElementIn>(ubOffset);
        ubOffset += mForSingleVec * l0N * sizeof(ElementIn);

        if ASCEND_IS_AIV {
            if constexpr (x2QuantMode == QuantMode::PERCHANNEL_MODE) {
                x2ScaleUb_ = resource.ubBuf.template GetBufferByByte<ElementX2Scale>(ubOffset);
                ubOffset += l0N * sizeof(ElementX2Scale);
                x2ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementX2Scale*>(x2ScaleGmAddr));
            } else {
                ElementX2Scale x2ScaleValue = *reinterpret_cast<__gm__ ElementX2Scale*>(x2ScaleGmAddr);
                if constexpr (AscendC::IsSameType<ElementX2Scale, bfloat16_t>::value) {
                    x2ScaleScalar_ = AscendC::ToFloat(x2ScaleValue);
                } else {
                    x2ScaleScalar_ = x2ScaleValue;
                }
            }

            if constexpr (x1QuantMode == QuantMode::PERTOKEN_MODE) {
                x1ScaleUb_ = resource.ubBuf.template GetBufferByByte<ElementX1Scale>(ubOffset);
                ubOffset += RoundUp(mForSingleVec * sizeof(ElementX1Scale), static_cast<uint64_t>(UB_ALIGN_SIZE));
                x1ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementX1Scale*>(x1ScaleGmAddr));
            } else if constexpr (x1QuantMode == QuantMode::PERTENSOR_MODE) {
                ElementX1Scale x1ScaleValue = *reinterpret_cast<__gm__ ElementX1Scale*>(x1ScaleGmAddr);
                if constexpr (AscendC::IsSameType<ElementX1Scale, bfloat16_t>::value) {
                    x1ScaleScalar_ = AscendC::ToFloat(x1ScaleValue);
                } else {
                    x1ScaleScalar_ = x1ScaleValue;
                }
            }

            if constexpr (hasBias) {
                biasUb_ = resource.ubBuf.template GetBufferByByte<ElementBiasStorage>(ubOffset);
                ubOffset += l0N * sizeof(ElementBiasStorage);
                biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementBiasStorage*>(biasGmAddr));
            }

            for (uint32_t i = 0; i < DOUBLE_BUFFER_COUNT; ++i) {
                dequantOutUb_[i] = resource.ubBuf.template GetBufferByByte<ElementOut>(ubOffset);
                ubOffset += CeilDiv(mForSingleVec, FP32_OUTPUT_TIMES) * l0N * sizeof(ElementOut);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(i);
            }

            yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementOut*>(yGmAddr));
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        for (uint32_t i = 0; i < DOUBLE_BUFFER_COUNT; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(i);
        }
    }

    CATLASS_HOST_DEVICE
    static constexpr size_t GetWorkspaceSize()
    {
        return 0;
    }

    CATLASS_HOST_DEVICE
    static constexpr Status CanImplement()
    {
        if (CeilDiv(l0M, splitTaskRatio) * l0N * static_cast<int64_t>(sizeof(ElementIn)) >
            static_cast<int64_t>(ArchTag::UB_SIZE)) {
            return Status::kInvalid;
        }
        return Status::kSuccess;
    }

    CATLASS_DEVICE auto GetL0c2UbTensor()
    {
        return l0cOutUb_;
    }

    CATLASS_DEVICE void SetOutL2CacheHint()
    {
        yGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }

    template <class TensorUb>
    CATLASS_DEVICE void operator()(TensorUb& tensorUb)
    {
        constexpr uint32_t Y_IDX = 0;
        constexpr uint32_t X2SCALE_IDX = 1;
        constexpr uint32_t X1SCALE_IDX = 2;
        constexpr uint32_t BIAS_IDX = 3;

        uint32_t singleM = tla::get<0>(tensorUb.shape());
        uint32_t singleN = tla::get<1>(tensorUb.shape());
        auto blockCoord = tla::MakeCoord(
            tla::get<0>(tensorUb.coord()) * problemShape_.n() + tla::get<1>(tensorUb.coord()),
            tla::get<1>(tensorUb.coord()), tla::get<0>(tensorUb.coord()), tla::get<1>(tensorUb.coord()));
        uint32_t halfSingleM = CeilDiv(singleM, splitTaskRatio);
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t singleMInVec = subBlockIdx == 1 ? singleM - halfSingleM : halfSingleM;
        if (singleMInVec == 0) {
            return;
        }
        uint64_t mOffset = subBlockIdx * halfSingleM;

        if constexpr (x2QuantMode == QuantMode::PERCHANNEL_MODE) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            auto x2Layout = tla::MakeLayout<ElementX2Scale, layout::VectorLayout>(1, singleN);
            auto x2UbTensor = tla::MakeTensor(x2ScaleUb_, x2Layout, Arch::PositionUB{});
            auto x2GmTensor =
                tla::MakeTensor(x2ScaleGlobal_[tla::get<X2SCALE_IDX>(blockCoord)], x2Layout, Arch::PositionGM{});
            using CopyX2Scale = Tile::CopyGm2UbTla<ArchTag, decltype(x2GmTensor), decltype(x2UbTensor)>;
            CopyX2Scale{}(x2UbTensor, x2GmTensor);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        if constexpr (x1QuantMode == QuantMode::PERTOKEN_MODE) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            auto x1Layout = tla::MakeLayout<ElementX1Scale, layout::VectorLayout>(1, singleMInVec);
            auto x1UbTensor = tla::MakeTensor(x1ScaleUb_, x1Layout, Arch::PositionUB{});
            auto x1GmTensor = tla::MakeTensor(
                x1ScaleGlobal_[tla::get<X1SCALE_IDX>(blockCoord) + mOffset], x1Layout, Arch::PositionGM{});
            using CopyX1Scale = Tile::CopyGm2UbTla<ArchTag, decltype(x1GmTensor), decltype(x1UbTensor)>;
            CopyX1Scale{}(x1UbTensor, x1GmTensor);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        }

        if constexpr (hasBias) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            auto biasLayout = tla::MakeLayout<ElementBiasStorage, layout::VectorLayout>(1, singleN);
            auto biasUbTensor = tla::MakeTensor(biasUb_, biasLayout, Arch::PositionUB{});
            auto biasGmTensor =
                tla::MakeTensor(biasGlobal_[tla::get<BIAS_IDX>(blockCoord)], biasLayout, Arch::PositionGM{});
            using CopyBias = Tile::CopyGm2UbTla<ArchTag, decltype(biasGmTensor), decltype(biasUbTensor)>;
            CopyBias{}(biasUbTensor, biasGmTensor);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        }

        uint16_t splitNumOfOut = singleMInVec >= FP32_OUTPUT_TIMES ? FP32_OUTPUT_TIMES : singleMInVec;
        uint64_t mSizeForOnce = CeilDiv(singleMInVec, static_cast<uint64_t>(splitNumOfOut));
        uint32_t outUbListId = 0;
        for (uint16_t i = 0; i < splitNumOfOut; ++i) {
            if (i * mSizeForOnce >= singleMInVec) {
                break;
            }
            uint32_t outUbListIdNext = (outUbListId + 1) % DOUBLE_BUFFER_COUNT;
            uint64_t offsetL0c =
                i * mSizeForOnce * RoundUp(singleN, static_cast<uint64_t>(UB_ALIGN_SIZE / sizeof(ElementIn)));
            uint16_t mSize = static_cast<uint16_t>(
                singleMInVec - i * mSizeForOnce >= mSizeForOnce ? mSizeForOnce : singleMInVec - i * mSizeForOnce);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outUbListId);
            AscendC::LocalTensor<ElementOut> dequantOutUb = dequantOutUb_[outUbListId];
            __ubuf__ ElementOut* dst = reinterpret_cast<__ubuf__ ElementOut*>(dequantOutUb.GetPhyAddr());
            __ubuf__ ElementIn* src = reinterpret_cast<__ubuf__ ElementIn*>(l0cOutUb_.GetPhyAddr()) + offsetL0c;
            __ubuf__ ElementX2Scale* x2Scale = nullptr;
            __ubuf__ ElementX1Scale* x1Scale = nullptr;
            __ubuf__ ElementBiasStorage* bias = nullptr;
            if constexpr (x2QuantMode == QuantMode::PERCHANNEL_MODE) {
                x2Scale = reinterpret_cast<__ubuf__ ElementX2Scale*>(x2ScaleUb_.GetPhyAddr());
            }
            if constexpr (x1QuantMode == QuantMode::PERTOKEN_MODE) {
                x1Scale = reinterpret_cast<__ubuf__ ElementX1Scale*>(x1ScaleUb_.GetPhyAddr()) + i * mSizeForOnce;
            }
            if constexpr (hasBias) {
                bias = reinterpret_cast<__ubuf__ ElementBiasStorage*>(biasUb_.GetPhyAddr());
            }

            using Dequant = Tile::TileDequant<
                ElementOut, ElementIn, ElementBias, ElementX1Scale, ElementX2Scale, x1QuantMode, x2QuantMode>;
            Dequant::DoDequant(dst, src, x2Scale, x1Scale, bias, x1ScaleScalar_, x2ScaleScalar_, mSize, singleN);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outUbListId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outUbListId);
            AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
            ub2GmParams.blockLen = singleN * sizeof(ElementOut);
            ub2GmParams.blockCount = mSize;
            ub2GmParams.dstStride = (problemShape_.n() - singleN) * sizeof(ElementOut);
            uint64_t yOffset = (mOffset + i * mSizeForOnce) * problemShape_.n();
            AscendC::DataCopyPad(yGlobal_[tla::get<Y_IDX>(blockCoord) + yOffset], dequantOutUb, ub2GmParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outUbListId);
            outUbListId = outUbListIdNext;
        }

        if constexpr (x2QuantMode == QuantMode::PERCHANNEL_MODE) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        if constexpr (x1QuantMode == QuantMode::PERTOKEN_MODE) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        }
        if constexpr (hasBias) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        }
    }

private:
    static constexpr uint32_t FP32_OUTPUT_TIMES = 4;
    static constexpr uint32_t UB_ALIGN_SIZE = 32;
    static constexpr uint32_t DOUBLE_BUFFER_COUNT = 2;

    AscendC::GlobalTensor<ElementOut> yGlobal_;
    AscendC::GlobalTensor<ElementX2Scale> x2ScaleGlobal_;
    AscendC::GlobalTensor<ElementX1Scale> x1ScaleGlobal_;
    AscendC::GlobalTensor<ElementBiasStorage> biasGlobal_;
    AscendC::LocalTensor<ElementIn> l0cOutUb_;
    AscendC::LocalTensor<ElementX2Scale> x2ScaleUb_;
    AscendC::LocalTensor<ElementX1Scale> x1ScaleUb_;
    AscendC::LocalTensor<ElementBiasStorage> biasUb_;
    AscendC::LocalTensor<ElementOut> dequantOutUb_[DOUBLE_BUFFER_COUNT];
    float x2ScaleScalar_ = 0.0F;
    float x1ScaleScalar_ = 0.0F;
    GemmCoord problemShape_{};
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_DEQUANT_HPP
