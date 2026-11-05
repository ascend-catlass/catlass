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
#include "catlass/gemm/kernel/basic_syrk_polynomial_tla.h"
#include "syrk_example.h"

using namespace Catlass;
using namespace tla;

using L1Tile = Shape<Int<128>, Int<128>, Int<256>>;
using L0Tile = Shape<Int<128>, Int<128>, Int<64>>;
using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
    Arch::Ascend950, bfloat16_t, layout::RowMajor, bfloat16_t, layout::ColumnMajor, float, layout::RowMajor, void,
    Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
using BlockMmad = Gemm::Block::BlockMmadSyrkTla<L1Tile, L0Tile, bfloat16_t, bfloat16_t, TileCopy>;
using Scheduler = Gemm::Block::SyrkBlockScheduler;
using Kernel = Gemm::Kernel::BasicSyrkPolynomialTla<BlockMmad, Scheduler>;

CATLASS_GLOBAL __mix__(1, 2) void PolynomialSyrk(Kernel::Params params)
{
    Kernel{}(params);
}

static float ParseFloat(const char* text)
{
    size_t end = 0;
    float value = std::stof(text, &end);
    if (text[end] != '\0' || !std::isfinite(value)) {
        throw std::invalid_argument("expected a finite floating-point value");
    }
    return value;
}

int main(int argc, const char** argv)
{
    if (argc != 7 && argc != 8) {
        std::cerr << "Usage: " << argv[0] << " m k batch alpha beta diagonal [device_id]" << std::endl;
        return 1;
    }
    try {
        SyrkExample::Options options{
            SyrkExample::ParseInteger(argv[1]), SyrkExample::ParseInteger(argv[2]), SyrkExample::ParseInteger(argv[3])};
        options.alpha = ParseFloat(argv[4]);
        options.beta = ParseFloat(argv[5]);
        options.diagonal = ParseFloat(argv[6]);
        if (argc == 8) {
            options.device = SyrkExample::ParseInteger(argv[7]);
        }
        if (options.m % 32 != 0) {
            throw std::invalid_argument("polynomial SYRK requires M to be a multiple of 32");
        }
        return SyrkExample::Run<bfloat16>(
            options,
            [options](uint8_t* x, uint8_t* xt, uint8_t* residual, uint8_t* y, aclrtStream stream, uint32_t cores) {
                Kernel::Arguments arguments{options.batch,
                                            GemmCoord{options.m, options.m, options.k},
                                            x,
                                            xt,
                                            residual,
                                            y,
                                            options.alpha,
                                            options.beta,
                                            options.diagonal};
                if (!Kernel::CanImplement(arguments)) {
                    throw std::invalid_argument("unsupported polynomial SYRK arguments");
                }
                PolynomialSyrk<<<cores, nullptr, stream>>>(Kernel::ToUnderlyingArguments(arguments, nullptr));
            });
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
