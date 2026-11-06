/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define K_MAX_SHAPE_DIM 0

#include "catlass/gemm/block/block_mmad_syrk_tla.hpp"
#include "catlass/gemm/block/block_scheduler_syrk.h"
#include "catlass/gemm/kernel/basic_syrk_symmetric_tla.h"
#include "syrk_example.h"
#include "syrk_splitk.h"

using namespace Catlass;
using namespace tla;

using L1Tile = Shape<Int<128>, Int<128>, Int<256>>;
using L0Tile = Shape<Int<128>, Int<128>, Int<64>>;
using BlockMmad = Gemm::Block::BlockMmadSyrkTla<L1Tile, L0Tile, bfloat16_t, float>;
using Scheduler = Gemm::Block::SyrkBlockScheduler;
using Kernel = Gemm::Kernel::BasicSyrkSymmetricTla<BlockMmad, Scheduler>;

CATLASS_GLOBAL __mix__(1, 2) void BatchedBasicSyrk(Kernel::Params params)
{
    Kernel{}(params);
}

int main(int argc, const char** argv)
{
    try {
        const uint32_t splitk = SyrkExample::ParseSplit(argc, argv);
        if (argc != 4 && argc != 5) {
            std::cerr << "Usage: " << argv[0] << " m k batch [device_id] [--splitk factor]" << std::endl;
            return 1;
        }
        SyrkExample::Options options{
            SyrkExample::ParseInteger(argv[1]), SyrkExample::ParseInteger(argv[2]), SyrkExample::ParseInteger(argv[3])};
        options.splitk = splitk;
        if (argc == 5) {
            options.device = SyrkExample::ParseInteger(argv[4]);
        }
        return SyrkExample::Run<float>(
            options,
            [options](uint8_t* x, uint8_t* xt, uint8_t* residual, uint8_t* y, aclrtStream stream, uint32_t cores) {
                if (options.splitk > 1) {
                    SyrkExample::LaunchSplit<float>(options, x, xt, residual, y, stream, cores);
                    return;
                }
                Kernel::Arguments arguments{options.batch, GemmCoord{options.m, options.m, options.k}, x, xt, y};
                if (!Kernel::CanImplement(arguments)) {
                    throw std::invalid_argument("unsupported basic SYRK arguments");
                }
                BatchedBasicSyrk<<<cores, nullptr, stream>>>(Kernel::ToUnderlyingArguments(arguments, nullptr));
            });
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
