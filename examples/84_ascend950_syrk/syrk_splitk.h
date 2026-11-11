/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_ASCEND950_SYRK_SPLITK_H
#define EXAMPLES_ASCEND950_SYRK_SPLITK_H

#include <iostream>
#include "catlass/gemm/block/block_mmad_syrk_tla.hpp"
#include "catlass/gemm/block/block_scheduler_syrk.h"
#include "catlass/gemm/kernel/basic_syrk_polynomial_splitk_tla.h"
#include "syrk_example.h"

namespace SyrkExample {

template <class Output>
struct SplitTypes {
    using L1Tile = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<256>>;
    using L0Tile = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<64>>;
    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        Catlass::Arch::Ascend950, bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::ColumnMajor,
        float, Catlass::layout::RowMajor>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadSyrkTla<L1Tile, L0Tile, bfloat16_t, Output, TileCopy>;
    using Scheduler = Catlass::Gemm::Block::SyrkBlockScheduler;
    using Kernel = Catlass::Gemm::Kernel::BasicSyrkPolynomialSplitkTla<BlockMmad, Scheduler>;
};

template <class Output>
CATLASS_GLOBAL __mix__(1, 2) void SplitSyrk(typename SplitTypes<Output>::Kernel::Params params)
{
    typename SplitTypes<Output>::Kernel kernel;
    kernel(params);
}

template <class Output>
void LaunchSplit(
    const Options& options, uint8_t* x, uint8_t* xt, uint8_t* residual, uint8_t* y, aclrtStream stream, uint32_t cores)
{
    using Kernel = typename SplitTypes<Output>::Kernel;
    const typename Kernel::BlockScheduler scheduler(
        Catlass::GemmCoord{options.m, options.m, options.k}, Catlass::MakeCoord(Kernel::L1_TILE_M, Kernel::L1_TILE_N));
    const uint32_t slices = Kernel::GetEffectiveSplit(options.k, options.splitk);
    const size_t bytes =
        static_cast<size_t>(options.batch) * scheduler.GetCoreLoops() * slices * Kernel::TILE_AREA * sizeof(float);
    Buffer workspace(bytes);
    typename Kernel::Arguments arguments{
        options.batch,
        Catlass::GemmCoord{options.m, options.m, options.k},
        x,
        xt,
        residual,
        y,
        workspace.ptr,
        options.alpha,
        options.beta,
        options.diagonal,
        options.splitk};
    if (!Kernel::CanImplement(arguments)) {
        throw std::invalid_argument("unsupported split-K SYRK arguments");
    }
    std::cout << "Split-K requested=" << options.splitk << ", effective=" << slices << ", workspace bytes=" << bytes
              << std::endl;
    SplitSyrk<Output><<<cores, nullptr, stream>>>(Kernel::ToUnderlyingArguments(arguments, nullptr));
    // Keep caller-owned partial tiles alive until reduction and epilogue finish.
    CheckAcl(aclrtSynchronizeStream(stream));
}

} // namespace SyrkExample
#endif
