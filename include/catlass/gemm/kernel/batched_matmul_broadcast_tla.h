/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_BATCHED_MATMUL_BROADCAST_TLA_H
#define CATLASS_GEMM_KERNEL_BATCHED_MATMUL_BROADCAST_TLA_H

#include <limits>

#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

enum class BiasMode : uint32_t
{
    None,
    SharedN,
    PerBatchN
};

// Continuous physical matrices; RowMajor/ColumnMajor describe the logical transpose.
template <class BlockMmad_, class BlockScheduler_>
class BatchedMatmulBroadcastTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
    using ElementBias = typename BlockMmad::ElementBias;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static_assert(std::is_same_v<ArchTag, Arch::Ascend950>, "Requires Ascend 950");
    static_assert(
        std::is_same_v<ElementA, half> || std::is_same_v<ElementA, bfloat16_t> || std::is_same_v<ElementA, float>,
        "Only FP16/BF16/FP32 are supported");
    static_assert(
        std::is_same_v<ElementA, ElementB> && std::is_same_v<ElementA, ElementC>, "A/B/C must have the same dtype");
    static_assert(
        std::is_void_v<ElementBias> || std::is_same_v<ElementBias, ElementA>, "Bias must have the same dtype as A/B/C");
    static_assert(tla::detail::isRowMajor<LayoutC>::value, "Output must be continuous ND");

    struct Arguments {
        uint32_t batchCount;
        uint32_t batchA;
        uint32_t batchB;
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrBias{nullptr};
        BiasMode biasMode{BiasMode::None};
    };

    struct Params : Arguments {
        int64_t strideA;
        int64_t strideB;
        int64_t strideC;
        int64_t strideBias;
    };

    template <class Layout>
    static bool IsContinuous(const Layout& layout, uint32_t rows, uint32_t cols)
    {
        if (tla::get<0>(layout.originShape()) != rows || tla::get<1>(layout.originShape()) != cols) {
            return false;
        }
        if constexpr (tla::detail::isRowMajor<Layout>::value) {
            return tla::get<0>(layout.stride()) == cols && tla::get<1>(layout.stride()) == 1;
        } else if constexpr (tla::detail::isColumnMajor<Layout>::value) {
            return tla::get<0>(layout.stride()) == 1 && tla::get<1>(layout.stride()) == rows;
        } else {
            return false;
        }
    }

    static bool CanImplement(const Arguments& arguments)
    {
        uint64_t m = arguments.problemShape.m();
        uint64_t n = arguments.problemShape.n();
        uint64_t k = arguments.problemShape.k();
        constexpr uint64_t maxDimension = std::numeric_limits<int32_t>::max();
        constexpr uint64_t maxBytes = std::numeric_limits<int64_t>::max();
        if (!m || !n || !k || m > maxDimension || n > maxDimension || k > maxDimension || !arguments.batchCount) {
            return false;
        }
        if constexpr (BlockMmad::USE_HF32_MODE && !std::is_same_v<ElementA, float>) {
            return false;
        }
        if ((arguments.batchA != 1 && arguments.batchA != arguments.batchCount) ||
            (arguments.batchB != 1 && arguments.batchB != arguments.batchCount)) {
            return false;
        }
        if (!arguments.ptrA || !arguments.ptrB || !arguments.ptrC ||
            reinterpret_cast<uintptr_t>(arguments.ptrA) % sizeof(ElementA) ||
            reinterpret_cast<uintptr_t>(arguments.ptrB) % sizeof(ElementB) ||
            reinterpret_cast<uintptr_t>(arguments.ptrC) % sizeof(ElementC)) {
            return false;
        }
        if (!IsContinuous(arguments.layoutA, m, k) || !IsContinuous(arguments.layoutB, k, n) ||
            !IsContinuous(arguments.layoutC, m, n)) {
            return false;
        }
        if (m * k > maxBytes / sizeof(ElementA) / arguments.batchA ||
            k * n > maxBytes / sizeof(ElementB) / arguments.batchB ||
            m * n > maxBytes / sizeof(ElementC) / arguments.batchCount) {
            return false;
        }
        uint64_t tileCount = ((m + L1_TILE_M - 1) / L1_TILE_M) * ((n + L1_TILE_N - 1) / L1_TILE_N);
        if (tileCount > std::numeric_limits<uint32_t>::max() / arguments.batchCount) {
            return false;
        }
        if constexpr (std::is_void_v<ElementBias>) {
            if (arguments.biasMode != BiasMode::None || arguments.ptrBias) {
                return false;
            }
        } else {
            if (arguments.biasMode != BiasMode::SharedN && arguments.biasMode != BiasMode::PerBatchN) {
                return false;
            }
            if (!arguments.ptrBias || reinterpret_cast<uintptr_t>(arguments.ptrBias) % sizeof(ElementBias)) {
                return false;
            }
            if (n > maxBytes / sizeof(ElementBias) / arguments.batchCount) {
                return false;
            }
        }
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments& arguments)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments& arguments, uint8_t* workspace)
    {
        Params params{};
        static_cast<Arguments&>(params) = arguments;
        int64_t m = arguments.problemShape.m();
        int64_t n = arguments.problemShape.n();
        int64_t k = arguments.problemShape.k();
        params.strideA = 0;
        if (arguments.batchA != 1) {
            params.strideA = m * k;
        }
        params.strideB = 0;
        if (arguments.batchB != 1) {
            params.strideB = k * n;
        }
        params.strideC = m * n;
        params.strideBias = 0;
        if (arguments.biasMode == BiasMode::PerBatchN) {
            params.strideBias = n;
        }
        return params;
    }

    CATLASS_DEVICE
    BatchedMatmulBroadcastTla()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint64_t coreLoops = static_cast<uint64_t>(params.batchCount) * matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        if constexpr (BlockMmad::USE_HF32_MODE) {
            // Do not inherit the FP32-to-HF32 rounding state from another kernel.
            AscendC::SetHF32TransMode(AscendC::HF32TransMode::NEAREST_EVEN);
        }

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);

        using BiasStorage = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
        AscendC::GlobalTensor<BiasStorage> gmBias;
        if constexpr (!std::is_void_v<ElementBias>) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias*)params.ptrBias);
        }

        uint32_t coreNum = AscendC::GetBlockNum();
        for (uint64_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += coreNum) {
            uint32_t tileIdx = static_cast<uint32_t>(loopIdx);
            uint32_t batchIdx = matmulBlockScheduler.GetBatchIdx(tileIdx);
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(tileIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            int64_t batchOffsetA = batchIdx * params.strideA;
            int64_t batchOffsetB = batchIdx * params.strideB;
            int64_t batchOffsetC = batchIdx * params.strideC;

            auto tensorA = tla::MakeTensor(gmA[batchOffsetA], params.layoutA, Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(gmB[batchOffsetB], params.layoutB, Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(gmC[batchOffsetC], params.layoutC, Arch::PositionGM{});

            auto tensorBlockA = GetTile(
                tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, 0),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tensorBlockB = GetTile(
                tensorB, tla::MakeCoord(0, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            if constexpr (!std::is_void_v<ElementBias>) {
                auto tensorBias = tla::MakeTensor(
                    gmBias[batchIdx * params.strideBias], tla::MakeLayout(params.problemShape.n()), Arch::PositionGM{});
                auto tensorBlockBias = GetTile(
                    tensorBias, tla::MakeCoord(blockCoord.n() * L1_TILE_N), tla::MakeShape(actualBlockShape.n()));
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockBias);
            } else {
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {}
};

} // namespace Catlass::Gemm::Kernel
#endif
