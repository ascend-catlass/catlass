/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_CONV2D_ELEMWISE_NO_SOURCE_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_CONV2D_ELEMWISE_NO_SOURCE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/epilogue/tile/tile_cast.hpp"
#include "catlass/epilogue/tile/tile_elemwise_silu.hpp"

namespace Catlass::Epilogue::Block {

template <
    class CType_, class DType_, class TileElemWiseEpilogue_, class TileCopy_, uint32_t TILE_H_, uint32_t TILE_W_,
    uint32_t TILE_COUT_>
class BlockEpilogue<
    EpilogueAtlasA2Conv2dElemWise, CType_, DType_, TileElemWiseEpilogue_, TileCopy_,
    std::integral_constant<uint32_t, TILE_H_>, std::integral_constant<uint32_t, TILE_W_>,
    std::integral_constant<uint32_t, TILE_COUT_>> {
public:
    using DispatchPolicy = EpilogueAtlasA2Conv2dElemWise;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;
    using TileElemWiseEpilogue = TileElemWiseEpilogue_;
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogue::COMPUTE_LENGTH;
    static constexpr uint32_t OPERANDS_NUM = DispatchPolicy::OPERANDS_NUM;
    static constexpr uint32_t TILE_H = TILE_H_;
    static constexpr uint32_t TILE_W = TILE_W_;
    static constexpr uint32_t TILE_COUT = TILE_COUT_;

    using ElementCompute = typename TileElemWiseEpilogue::ElementCompute;
    using LayoutComputeInUb = layout::RowMajor;

    static_assert(std::is_same_v<LayoutC, layout::NC1HWC0>, "Layout type of C must be NC1HWC0");
    static_assert(std::is_same_v<LayoutD, layout::NC1HWC0>, "Layout type of D must be NC1HWC0");

    static_assert(std::is_same_v<typename TileElemWiseEpilogue::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    // UB layout:
    // [ubC(ElementC)][ubD(ElementD)][ubSrc?(ElementCompute)][ubCompute?(ElementCompute)][ubBias(ElementC)]
    // [ubBiasF32?(ElementCompute)]
    // - ubSrc: only when ElementC != ElementCompute (for cast half→float)
    // - ubCompute: only when ElementCompute != ElementD (else aliased to ubD)
    // - ubBiasF32: only when ElementC != ElementCompute (bias cast至计算精度参与加法)
    static_assert(
        COMPUTE_LENGTH * (sizeof(ElementC) + sizeof(ElementD) +
                          (!std::is_same_v<ElementC, ElementCompute> ? sizeof(ElementCompute) : 0) +
                          (!std::is_same_v<ElementCompute, ElementD> ? sizeof(ElementCompute) : 0) + sizeof(ElementC) +
                          (!std::is_same_v<ElementC, ElementCompute> ? sizeof(ElementCompute) : 0)) <=
            ArchTag::UB_SIZE,
        "UB out of bounds");

    struct Params {
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrBias;
        uint32_t cout;
        GM_ADDR ptrD;
        LayoutD layoutD;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            GM_ADDR ptrC_, LayoutC const& layoutC_, GM_ADDR ptrBias_, uint32_t cout_, GM_ADDR ptrD_,
            LayoutD const& layoutD_)
            : ptrC(ptrC_), layoutC(layoutC_), ptrBias(ptrBias_), cout(cout_), ptrD(ptrD_), layoutD(layoutD_)
        {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, Params const& params) : params(params)
    {
        constexpr bool needCast = !std::is_same_v<ElementC, ElementCompute>;
        constexpr bool needSeparateCompute = !std::is_same_v<ElementCompute, ElementD>;
        constexpr size_t sizeC = COMPUTE_LENGTH * sizeof(ElementC);
        constexpr size_t sizeD = COMPUTE_LENGTH * sizeof(ElementD);
        constexpr size_t sizeCompute = COMPUTE_LENGTH * sizeof(ElementCompute);

        // UB layout:
        // [ubC(ElementC)][ubD(ElementD)][ubSrc?(ElementCompute)][ubCompute?(ElementCompute)][ubBias(ElementC)]
        // [ubBiasF32?(ElementCompute)]
        ubC = resource.ubBuf.template GetBufferByByte<ElementC>(0);
        ubD = resource.ubBuf.template GetBufferByByte<ElementD>(sizeC);

        if constexpr (needCast) {
            ubSrc = resource.ubBuf.template GetBufferByByte<ElementCompute>(sizeC + sizeD);
        }

        if constexpr (needSeparateCompute) {
            constexpr size_t offsetCompute = sizeC + sizeD + (needCast ? sizeCompute : 0);
            ubCompute = resource.ubBuf.template GetBufferByByte<ElementCompute>(offsetCompute);
        } else {
            ubCompute = ubD;
        }

        constexpr size_t offsetBias =
            sizeC + sizeD + (needCast ? sizeCompute : 0) + (needSeparateCompute ? sizeCompute : 0);
        ubBiasBuf = resource.ubBuf.template GetBufferByByte<ElementC>(offsetBias);

        if constexpr (needCast) {
            ubBiasF32 = resource.ubBuf.template GetBufferByByte<ElementCompute>(offsetBias + sizeC);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void operator()(
        Conv2dCoord const& blockCoord, Conv2dCoord const& actualBlockShape,
        AscendC::GlobalTensor<ElementC> const& gmTensorC, AscendC::GlobalTensor<ElementD>& gmTensorD)
    {
        uint32_t batch = actualBlockShape.batch();
        uint32_t h = actualBlockShape.h();
        uint32_t w = actualBlockShape.w();
        uint32_t cout = actualBlockShape.cout();

        uint32_t c0 = params.layoutC.shape(4);
        uint32_t cout1 = CeilDiv(cout, c0);
        uint32_t totalElements = batch * cout1 * h * w * c0;

        uint32_t hoStart = blockCoord.h() * TILE_H;
        uint32_t woStart = blockCoord.w() * TILE_W;
        uint32_t coStartC1 = blockCoord.cout() * (TILE_COUT / c0);
        uint32_t bStart = blockCoord.batch();

        Catlass::Conv2dFmapCoord tileStartCoord{bStart, coStartC1, hoStart, woStart, 0};
        int64_t tileStartOffsetC = params.layoutC.GetOffset(tileStartCoord);
        int64_t tileStartOffsetD = params.layoutD.GetOffset(tileStartCoord);

        uint32_t ubCapacity = COMPUTE_LENGTH;
        uint32_t loopCount = (totalElements + ubCapacity - 1) / ubCapacity;

        for (uint32_t loopIdx = 0; loopIdx < loopCount; ++loopIdx) {
            uint32_t flatStart = loopIdx * ubCapacity;
            uint32_t remaining = totalElements - flatStart;
            uint32_t actualLen = (remaining > ubCapacity) ? ubCapacity : remaining;

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

            CopyGmToUbC copyGmToUbC;
            uint32_t ubRows = CeilDiv(actualLen, c0);
            layout::RowMajor layoutDst{ubRows, c0};
            layout::NC1HWC0 layoutSrc{
                batch,
                cout1,
                h,
                w,
                c0,
                params.layoutC.stride(0),
                params.layoutC.stride(1),
                params.layoutC.stride(2),
                params.layoutC.stride(3),
                params.layoutC.stride(4)};

            copyGmToUbC(ubC, gmTensorC, layoutDst, layoutSrc, tileStartOffsetC, flatStart);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

            // conv结果提前cast至计算精度：bias加法与SiLU均在该精度下进行，
            // 消除原"half加法舍入后再cast"链路中bias加法引入的中间舍入
            if constexpr (!std::is_same_v<ElementC, ElementCompute>) {
                AscendC::Cast(ubSrc, ubC, AscendC::RoundMode::CAST_NONE, actualLen);
            }

            // === BiasAdd: add per-channel bias ===
            using namespace AscendC;
            if (params.ptrBias != 0) {
                GlobalTensor<ElementC> gmBias;
                gmBias.SetGlobalBuffer((__gm__ ElementC*)params.ptrBias);

                uint32_t hw = h * w;

                // Step 1: Load all bias values from GM to UB
                for (uint32_t r = 0; r < ubRows; ++r) {
                    uint32_t globalRow = flatStart / c0 + r;
                    uint32_t c1_local = (globalRow / hw) % cout1;
                    uint32_t channelBase = (coStartC1 + c1_local) * c0;

                    LocalTensor<ElementC> ubBiasRow = ubBiasBuf[r * c0];
                    // 整行拷贝c0个元素（满足MTE2的32B对齐要求）；GM侧bias按coutRound分配，
                    // padding区可安全读取。超出cout的通道属于padding，其结果不会被读取。
                    if (channelBase < params.cout) {
                        DataCopy(ubBiasRow, gmBias[channelBase], c0);
                    }
                }

                // Step 2: Wait for all bias loads to complete
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                // Step 3: Add bias，加法在ElementCompute精度下完成（和仅在终输出处舍入一次）
                if constexpr (std::is_same_v<ElementC, ElementCompute>) {
                    for (uint32_t r = 0; r < ubRows; ++r) {
                        uint32_t elemsThisRow = (r == ubRows - 1) ? (actualLen - r * c0) : c0;
                        if (elemsThisRow > c0)
                            elemsThisRow = c0;
                        LocalTensor<ElementC> ubCRow = ubC[r * c0];
                        LocalTensor<ElementC> ubBiasRow = ubBiasBuf[r * c0];
                        Add(ubCRow, ubCRow, ubBiasRow, elemsThisRow);
                    }
                } else {
                    // bias cast至计算精度后与ubSrc（conv的计算精度值）相加：
                    // half在float中可精确表示，float加法不引入中间舍入
                    Cast(ubBiasF32, ubBiasBuf, AscendC::RoundMode::CAST_NONE, ubRows * c0);
                    for (uint32_t r = 0; r < ubRows; ++r) {
                        uint32_t elemsThisRow = (r == ubRows - 1) ? (actualLen - r * c0) : c0;
                        if (elemsThisRow > c0)
                            elemsThisRow = c0;
                        LocalTensor<ElementCompute> ubSrcRow = ubSrc[r * c0];
                        LocalTensor<ElementCompute> ubBiasF32Row = ubBiasF32[r * c0];
                        Add(ubSrcRow, ubSrcRow, ubBiasF32Row, elemsThisRow);
                    }
                }
            }

            TileElemWiseEpilogue tileEpilogue;
            if constexpr (std::is_same_v<ElementC, ElementCompute>) {
                tileEpilogue(ubCompute, ubC);
            } else {
                // ubSrc已含bias（计算精度），直接进入SiLU
                tileEpilogue(ubCompute, ubSrc);
            }

            if constexpr (!std::is_same_v<ElementCompute, ElementD>) {
                AscendC::Cast(ubD, ubCompute, AscendC::RoundMode::CAST_NONE, actualLen);
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            CopyUbToGmD copyUbToGmD;
            layout::NC1HWC0 layoutDstD{
                batch,
                cout1,
                h,
                w,
                c0,
                params.layoutD.stride(0),
                params.layoutD.stride(1),
                params.layoutD.stride(2),
                params.layoutD.stride(3),
                params.layoutD.stride(4)};
            layout::RowMajor layoutSrcD{ubRows, c0};

            copyUbToGmD(gmTensorD, ubD, layoutDstD, layoutSrcD, tileStartOffsetD, flatStart);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubC;
    AscendC::LocalTensor<ElementCompute> ubSrc;
    AscendC::LocalTensor<ElementCompute> ubCompute;
    AscendC::LocalTensor<ElementD> ubD;
    AscendC::LocalTensor<ElementC> ubBiasBuf;
    AscendC::LocalTensor<ElementCompute> ubBiasF32; // 仅needCast时有效：bias cast至计算精度
};

template <
    class CType_, class DType_, class TileElemWiseEpilogue_, class TileCopy_, uint32_t TILE_H_, uint32_t TILE_W_,
    uint32_t TILE_COUT_>
class BlockEpilogue<
    EpilogueAscend950Conv2dElemWise, CType_, DType_, TileElemWiseEpilogue_, TileCopy_,
    std::integral_constant<uint32_t, TILE_H_>, std::integral_constant<uint32_t, TILE_W_>,
    std::integral_constant<uint32_t, TILE_COUT_>> {
public:
    using DispatchPolicy = EpilogueAscend950Conv2dElemWise;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;
    using TileElemWiseEpilogue = TileElemWiseEpilogue_;
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogue::COMPUTE_LENGTH;
    static constexpr uint32_t OPERANDS_NUM = DispatchPolicy::OPERANDS_NUM;
    static constexpr uint32_t TILE_H = TILE_H_;
    static constexpr uint32_t TILE_W = TILE_W_;
    static constexpr uint32_t TILE_COUT = TILE_COUT_;

    using ElementCompute = float;
    using LayoutComputeInUb = layout::RowMajor;

    static_assert(std::is_same_v<LayoutC, layout::NC1HWC0>, "Layout type of C must be NC1HWC0");
    static_assert(std::is_same_v<LayoutD, layout::NC1HWC0>, "Layout type of D must be NC1HWC0");

    static_assert(std::is_same_v<typename TileElemWiseEpilogue::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
    // UB layout:
    // [ubC(ElementC)][ubSrc(float)][ubCompute(float)][ubD(ElementD)][ubBias(ElementC)][ubBiasF32(float)]
    static_assert(
        COMPUTE_LENGTH * (sizeof(ElementC) + sizeof(float) + sizeof(float) + sizeof(ElementD) + sizeof(ElementC) +
                          sizeof(float)) <=
            ArchTag::UB_SIZE,
        "UB out of bounds");

    struct Params {
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrBias;
        uint32_t cout;
        GM_ADDR ptrD;
        LayoutD layoutD;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            GM_ADDR ptrC_, LayoutC const& layoutC_, GM_ADDR ptrBias_, uint32_t cout_, GM_ADDR ptrD_,
            LayoutD const& layoutD_)
            : ptrC(ptrC_), layoutC(layoutC_), ptrBias(ptrBias_), cout(cout_), ptrD(ptrD_), layoutD(layoutD_)
        {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, Params const& params) : params(params)
    {
        // UB layout: ubC(half) + ubSrc(float) + ubCompute(float) + ubD(half) + ubBias(half) + ubBiasF32(float)
        ubC = resource.ubBuf.template GetBufferByByte<ElementC>(0);
        ubSrc = resource.ubBuf.template GetBufferByByte<float>(COMPUTE_LENGTH * sizeof(ElementC));
        ubCompute = resource.ubBuf.template GetBufferByByte<float>(COMPUTE_LENGTH * (sizeof(ElementC) + sizeof(float)));
        ubD = resource.ubBuf.template GetBufferByByte<ElementD>(
            COMPUTE_LENGTH * (sizeof(ElementC) + sizeof(float) + sizeof(float)));
        ubBiasBuf = resource.ubBuf.template GetBufferByByte<ElementC>(
            COMPUTE_LENGTH * (sizeof(ElementC) + sizeof(float) + sizeof(float) + sizeof(ElementD)));
        ubBiasF32 = resource.ubBuf.template GetBufferByByte<float>(
            COMPUTE_LENGTH * (sizeof(ElementC) + sizeof(float) + sizeof(float) + sizeof(ElementD) + sizeof(ElementC)));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void operator()(
        Conv2dCoord const& blockCoord, Conv2dCoord const& actualBlockShape,
        AscendC::GlobalTensor<ElementC> const& gmTensorC, AscendC::GlobalTensor<ElementD>& gmTensorD)
    {
        uint32_t batch = actualBlockShape.batch();
        uint32_t h = actualBlockShape.h();
        uint32_t w = actualBlockShape.w();
        uint32_t cout = actualBlockShape.cout();

        uint32_t c0 = params.layoutC.shape(4);
        uint32_t cout1 = CeilDiv(cout, c0);
        uint32_t totalElements = batch * cout1 * h * w * c0;

        uint32_t hoStart = blockCoord.h() * TILE_H;
        uint32_t woStart = blockCoord.w() * TILE_W;
        uint32_t coStartElems = blockCoord.cout() * TILE_COUT;
        uint32_t coStartC1 = coStartElems / c0;
        uint32_t coStartC0 = coStartElems % c0;
        uint32_t bStart = blockCoord.batch();

        Catlass::Conv2dFmapCoord tileStartCoord{bStart, coStartC1, hoStart, woStart, coStartC0};
        int64_t tileStartOffset = params.layoutC.GetOffset(tileStartCoord);

        uint32_t ubCapacity = COMPUTE_LENGTH;
        uint32_t loopCount = (totalElements + ubCapacity - 1) / ubCapacity;

        for (uint32_t loopIdx = 0; loopIdx < loopCount; ++loopIdx) {
            uint32_t flatStart = loopIdx * ubCapacity;
            uint32_t remaining = totalElements - flatStart;
            uint32_t actualLen = (remaining > ubCapacity) ? ubCapacity : remaining;

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

            CopyGmToUbC copyGmToUbC;
            uint32_t ubRows = CeilDiv(actualLen, c0);
            layout::RowMajor layoutDst{ubRows, c0};
            layout::NC1HWC0 layoutSrc{
                batch,
                cout1,
                h,
                w,
                c0,
                params.layoutC.stride(0),
                params.layoutC.stride(1),
                params.layoutC.stride(2),
                params.layoutC.stride(3),
                params.layoutC.stride(4)};

            copyGmToUbC(ubC, gmTensorC, layoutDst, layoutSrc, tileStartOffset, flatStart);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

            // conv结果提前cast至float：bias加法与SiLU均在float精度下进行，
            // 消除原"half加法舍入后再cast"链路中bias加法引入的中间舍入
            AscendC::Cast(ubSrc, ubC, AscendC::RoundMode::CAST_NONE, actualLen);

            // === BiasAdd: add per-channel bias ===
            using namespace AscendC;
            if (params.ptrBias != 0) {
                GlobalTensor<ElementC> gmBias;
                gmBias.SetGlobalBuffer((__gm__ ElementC*)params.ptrBias);

                uint32_t hw = h * w;

                // Step 1: Load all bias values from GM to UB
                for (uint32_t r = 0; r < ubRows; ++r) {
                    uint32_t globalRow = flatStart / c0 + r;
                    uint32_t c1_local = (globalRow / hw) % cout1;
                    uint32_t channelBase = (coStartC1 + c1_local) * c0;

                    LocalTensor<ElementC> ubBiasRow = ubBiasBuf[r * c0];
                    // 整行拷贝c0个元素（满足MTE2的32B对齐要求）；GM侧bias按coutRound分配，
                    // padding区可安全读取。超出cout的通道属于padding，其结果不会被读取。
                    if (channelBase < params.cout) {
                        DataCopy(ubBiasRow, gmBias[channelBase], c0);
                    }
                }

                // Step 2: Wait for all bias loads to complete
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                // Step 3: Add bias，加法在float精度下完成（和仅在终输出处舍入一次）
                // bias cast至float后与ubSrc（conv的float值）相加：half在float中可精确表示，
                // float加法不引入中间舍入
                Cast(ubBiasF32, ubBiasBuf, AscendC::RoundMode::CAST_NONE, ubRows * c0);
                for (uint32_t r = 0; r < ubRows; ++r) {
                    uint32_t elemsThisRow = (r == ubRows - 1) ? (actualLen - r * c0) : c0;
                    if (elemsThisRow > c0)
                        elemsThisRow = c0;
                    LocalTensor<float> ubSrcRow = ubSrc[r * c0];
                    LocalTensor<float> ubBiasF32Row = ubBiasF32[r * c0];
                    Add(ubSrcRow, ubSrcRow, ubBiasF32Row, elemsThisRow);
                }
            }

            // Compute SiLU in float precision: ubCompute = SiLU(ubSrc，已含bias)
            Muls(ubCompute, ubSrc, (float)-1, actualLen);
            Exp(ubCompute, ubCompute, actualLen);
            Adds(ubCompute, ubCompute, (float)1, actualLen);
            Div(ubCompute, ubSrc, ubCompute, actualLen);

            // Cast float result back to half
            if constexpr (!std::is_same_v<ElementCompute, ElementD>) {
                AscendC::Cast(ubD, ubCompute, AscendC::RoundMode::CAST_NONE, actualLen);
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            CopyUbToGmD copyUbToGmD;
            layout::NC1HWC0 layoutDstD{
                batch,
                cout1,
                h,
                w,
                c0,
                params.layoutD.stride(0),
                params.layoutD.stride(1),
                params.layoutD.stride(2),
                params.layoutD.stride(3),
                params.layoutD.stride(4)};
            layout::RowMajor layoutSrcD{ubRows, c0};

            copyUbToGmD(gmTensorD, ubD, layoutDstD, layoutSrcD, tileStartOffset, flatStart);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubC;
    AscendC::LocalTensor<float> ubSrc;
    AscendC::LocalTensor<ElementCompute> ubCompute;
    AscendC::LocalTensor<ElementD> ubD;
    AscendC::LocalTensor<ElementC> ubBiasBuf;
    AscendC::LocalTensor<float> ubBiasF32; // bias cast至float参与加法
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_CONV2D_ELEMWISE_NO_SOURCE_HPP
