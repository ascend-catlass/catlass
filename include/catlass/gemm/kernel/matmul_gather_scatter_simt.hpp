/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_GATHER_SCATTER_SIMT_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_GATHER_SCATTER_SIMT_HPP

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "simt_api/asc_fp16.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_sync_functions.h"

namespace Catlass::Gemm::Kernel {

struct MatmulGatherScatterSimtParams {
    GemmCoord problemShape;
    uint32_t physicalM;
    GM_ADDR ptrA;
    GM_ADDR ptrB;
    GM_ADDR ptrIndices;
    GM_ADDR ptrD;
    uint32_t aicCoreNum;

    CATLASS_HOST_DEVICE MatmulGatherScatterSimtParams()
    {}

    CATLASS_HOST_DEVICE MatmulGatherScatterSimtParams(
        GemmCoord const& problemShape_, uint32_t physicalM_, GM_ADDR ptrA_, GM_ADDR ptrB_, GM_ADDR ptrIndices_,
        GM_ADDR ptrD_, uint32_t aicCoreNum_)
        : problemShape(problemShape_),
          physicalM(physicalM_),
          ptrA(ptrA_),
          ptrB(ptrB_),
          ptrIndices(ptrIndices_),
          ptrD(ptrD_),
          aicCoreNum(aicCoreNum_)
    {}
};

namespace detail {

constexpr uint32_t MGS_SIMT_UB_SIZE = 64 * 1024;
constexpr uint32_t MGS_SIMT_ZERO_BYTES = 48 * 1024;
constexpr uint32_t MGS_SIMT_COMPLETION_UB_OFFSET = MGS_SIMT_UB_SIZE - BYTE_PER_BLK;
constexpr uint32_t MGS_SIMT_COMPLETION_ELEMENT_INDEX = 0;
constexpr uint32_t MGS_SIMT_COMPLETION_INITIAL_VALUE = 0;
constexpr uint32_t MGS_SIMT_COMPLETION_EPOCH = 1;
constexpr uint32_t MGS_SIMT_COMPLETION_MASK = 0x80000000U;
constexpr uint32_t MGS_SIMT_COMPLETION_VALUE = MGS_SIMT_COMPLETION_EPOCH | MGS_SIMT_COMPLETION_MASK;
constexpr uint32_t MGS_SIMT_FP16_ELEMENTS_PER_BLOCK = BYTE_PER_BLK / sizeof(half);
constexpr uint16_t MGS_SIMT_EVENT_ZERO_READY = 0;

static_assert(
    MGS_SIMT_COMPLETION_UB_OFFSET + sizeof(uint32_t) <= MGS_SIMT_UB_SIZE, "SIMT completion word exceeds UB capacity");

/**
 * Correctness-first fallback for shapes that cannot use the FullLoadA Cube path.
 * One warp computes 32 columns of one gathered row. Lane 0 loads A and broadcasts
 * it to the warp, while B loads and D stores remain coalesced.
 */
__simt_vf__ __launch_bounds__(SIMT_THREAD_NUM) inline void MatmulGatherScatterSimtVf(
    __gm__ const half* a, __gm__ const half* b, __gm__ const int32_t* indices, __gm__ half* d, uint32_t n, uint32_t k,
    uint32_t physicalN, uint64_t warpBegin, uint64_t warpEnd, uint32_t warpCount,
    __ubuf__ volatile uint32_t* completion, uint32_t completionValue)
{
    uint32_t lane = __cce_simt_get_TID_X();
    uint32_t warp = __cce_simt_get_TID_Y();
    uint64_t columnTiles = CeilDiv(static_cast<uint64_t>(n), static_cast<uint64_t>(SIMT_WARP_SIZE));

    for (uint64_t warpTask = warpBegin + warp; warpTask < warpEnd; warpTask += warpCount) {
        uint32_t logicalRow = static_cast<uint32_t>(warpTask / columnTiles);
        uint32_t column = static_cast<uint32_t>(warpTask % columnTiles) * SIMT_WARP_SIZE + lane;
        uint32_t physicalRow = lane == 0U ? static_cast<uint32_t>(indices[logicalRow]) : 0U;
        physicalRow = asc_shfl(physicalRow, 0);

        float accumulator = 0.0F;
        uint64_t aOffset = static_cast<uint64_t>(physicalRow) * k;
        for (uint32_t kIdx = 0; kIdx < k; ++kIdx) {
            float valueA = lane == 0U ? __half2float(asc_ldcg(const_cast<__gm__ half*>(a + aOffset + kIdx))) : 0.0F;
            valueA = asc_shfl(valueA, 0);
            if (column < n) {
                float valueB =
                    __half2float(asc_ldcg(const_cast<__gm__ half*>(b + static_cast<uint64_t>(kIdx) * n + column)));
                accumulator += valueA * valueB;
            }
        }
        if (column < n) {
            d[static_cast<uint64_t>(physicalRow) * physicalN + column] = __float2half(accumulator);
        }
    }

    asc_threadfence();
    asc_syncthreads();
    if (lane == 0U && warp == 0U) {
        asc_threadfence_block();
        completion[0] = completionValue;
    }
}

CATLASS_DEVICE void RunMatmulGatherScatterSimt(
    __gm__ const half* a, __gm__ const half* b, __gm__ const int32_t* indices, __gm__ half* d, uint32_t j, uint32_t n,
    uint32_t k, uint32_t physicalN, uint32_t workerId, uint32_t workerNum, __ubuf__ volatile uint32_t* completion,
    uint32_t completionValue)
{
    uint64_t columnTiles = CeilDiv(static_cast<uint64_t>(n), static_cast<uint64_t>(SIMT_WARP_SIZE));
    uint64_t totalWarps = static_cast<uint64_t>(j) * columnTiles;
    uint64_t warpBegin = totalWarps * workerId / workerNum;
    uint64_t warpEnd = totalWarps * (workerId + 1U) / workerNum;
    if (warpBegin == warpEnd) {
        completion[0] = completionValue;
        return;
    }

    uint64_t warpSpan = warpEnd - warpBegin;
    uint32_t warpCount = static_cast<uint32_t>(warpSpan < SIMT_WARP_NUM ? warpSpan : SIMT_WARP_NUM);
    asc_vf_call<MatmulGatherScatterSimtVf>(
        dim3{SIMT_WARP_SIZE, warpCount}, a, b, indices, d, n, k, physicalN, warpBegin, warpEnd, warpCount, completion,
        completionValue);
}

} // namespace detail

class MatmulGatherScatterSimt {
public:
    using ArchTag = Arch::Ascend950;
    using Params = MatmulGatherScatterSimtParams;
    static constexpr uint32_t UB_SIZE = detail::MGS_SIMT_UB_SIZE;

    struct Arguments {
        GemmCoord problemShape;
        uint32_t physicalM;
        uint8_t* ptrA;
        uint8_t* ptrB;
        uint8_t* ptrIndices;
        uint8_t* ptrD;
        uint32_t aicCoreNum;
    };

    static bool CanImplement(Arguments const& args)
    {
        return args.problemShape.m() != 0 && args.problemShape.n() != 0 && args.problemShape.k() != 0 &&
               args.physicalM != 0 && args.problemShape.m() <= args.physicalM && args.aicCoreNum != 0;
    }

    static size_t GetWorkspaceSize(Arguments const&)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(Arguments const& args, uint8_t* workspace)
    {
        (void)workspace;
        return Params{args.problemShape, args.physicalM, args.ptrA,      args.ptrB,
                      args.ptrIndices,   args.ptrD,      args.aicCoreNum};
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const&)
    {
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        Arch::Resource<ArchTag> resource;
        AscendC::GlobalTensor<half> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(params.ptrD));
        auto completionUb = PrepareCompletion(resource);

        ZeroOutput(resource, gmD, static_cast<uint64_t>(params.physicalM) * params.problemShape.n());
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll<false>();
        RunSimt(params, completionUb);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll<false>();
    }

private:
    CATLASS_DEVICE static AscendC::LocalTensor<uint32_t> PrepareCompletion(Arch::Resource<ArchTag>& resource)
    {
        auto completionUb = resource.ubBuf.template GetBufferByByte<uint32_t>(detail::MGS_SIMT_COMPLETION_UB_OFFSET);
        completionUb.SetValue(detail::MGS_SIMT_COMPLETION_ELEMENT_INDEX, detail::MGS_SIMT_COMPLETION_INITIAL_VALUE);
        return completionUb;
    }

    CATLASS_DEVICE static void ZeroOutput(
        Arch::Resource<ArchTag>& resource, AscendC::GlobalTensor<half> const& gmD, uint64_t totalElems)
    {
        using VectorType = GemmType<half, layout::VectorLayout>;
        using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, VectorType>;
        CopyUbToGm copyUbToGm;
        auto zeroUb = resource.ubBuf.template GetBufferByByte<half>(0);
        constexpr uint32_t zeroChunkElems = detail::MGS_SIMT_ZERO_BYTES / sizeof(half);
        AscendC::Duplicate(zeroUb, static_cast<half>(0), zeroChunkElems);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(detail::MGS_SIMT_EVENT_ZERO_READY);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(detail::MGS_SIMT_EVENT_ZERO_READY);

        uint64_t totalBlocks = CeilDiv(totalElems, static_cast<uint64_t>(detail::MGS_SIMT_FP16_ELEMENTS_PER_BLOCK));
        uint64_t workerId = AscendC::GetBlockIdx();
        uint64_t workerNum = static_cast<uint64_t>(AscendC::GetBlockNum()) * AscendC::GetSubBlockNum();
        uint64_t beginBlock = totalBlocks * workerId / workerNum;
        uint64_t endBlock = totalBlocks * (workerId + 1U) / workerNum;
        uint64_t begin = beginBlock * detail::MGS_SIMT_FP16_ELEMENTS_PER_BLOCK;
        uint64_t end = min(endBlock * detail::MGS_SIMT_FP16_ELEMENTS_PER_BLOCK, totalElems);

        for (uint64_t offset = begin; offset < end;) {
            uint64_t remaining = end - offset;
            uint32_t count = static_cast<uint32_t>(remaining < zeroChunkElems ? remaining : zeroChunkElems);
            layout::VectorLayout vectorLayout(count);
            copyUbToGm(gmD[offset], zeroUb, vectorLayout, vectorLayout);
            offset += count;
        }
    }

    CATLASS_DEVICE static void RunSimt(Params const& params, AscendC::LocalTensor<uint32_t> const& completionUb)
    {
        uint32_t workerId = AscendC::GetBlockIdx();
        uint32_t workerNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        auto* a = reinterpret_cast<__gm__ const half*>(params.ptrA);
        auto* b = reinterpret_cast<__gm__ const half*>(params.ptrB);
        auto* indices = reinterpret_cast<__gm__ const int32_t*>(params.ptrIndices);
        auto* d = reinterpret_cast<__gm__ half*>(params.ptrD);
        auto* completion = reinterpret_cast<__ubuf__ volatile uint32_t*>(completionUb.GetPhyAddr());
        detail::RunMatmulGatherScatterSimt(
            a, b, indices, d, params.problemShape.m(), params.problemShape.n(), params.problemShape.k(),
            params.problemShape.n(), workerId, workerNum, completion, detail::MGS_SIMT_COMPLETION_VALUE);
        WaitSimtCompletion(completionUb);
    }

    CATLASS_DEVICE static void WaitSimtCompletion(AscendC::LocalTensor<uint32_t> const& completionUb)
    {
        auto* completion = reinterpret_cast<__ubuf__ volatile uint32_t*>(completionUb.GetPhyAddr());
        while (completion[detail::MGS_SIMT_COMPLETION_ELEMENT_INDEX] != detail::MGS_SIMT_COMPLETION_VALUE) {
        }
    }
};

} // namespace Catlass::Gemm::Kernel

#endif // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

#endif // CATLASS_GEMM_KERNEL_MATMUL_GATHER_SCATTER_SIMT_HPP
