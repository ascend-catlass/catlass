/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_BLOCK_BLOCK_CONV_HPP
#define CATLASS_CONV_BLOCK_BLOCK_CONV_HPP

#include "catlass/catlass.hpp"
#include "catlass/conv/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace Catlass::Conv::Block {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
template <
    class DispatchPolicy, class CoreTileShape, class FmapL1TileShape, class FilterL1TileShape, class L0TileShape,
    class FmapType, class FilterType, class OutType, class BiasType,
    class TileCopy =
        Gemm::Tile::ConvTileCopy<typename DispatchPolicy::ArchTag, FmapType, FilterType, OutType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, FmapType, FilterType, BiasType>>
struct BlockConv {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockConv is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy, class FmapL1TileShape, class FilterL1TileShape, class L0TileShape, class FmapType,
    class FilterType, class OutputType, class BiasType = void,
    /// 卷积累加类型：默认void表示按输入类型自动推导；可显式指定half启用fp16累加特化
    class ElementAccumulator = void,
    class TileCopy = Conv::Tile::TileCopy<
        typename DispatchPolicy::ArchTag, FmapType, FilterType, OutputType, BiasType, ElementAccumulator>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, FmapType, FilterType, BiasType>>
struct BlockConv2d {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockConv2d is not implemented for this DispatchPolicy");
};
#endif

template <
    class DispatchPolicy, class FmapL1TileShape, class FilterL1TileShape, class L0TileShape, class ElementFmap,
    class ElementFilter, class ElementOutput,
    /// 卷积累加类型：默认void表示按输入类型自动推导（half输入推导为float）；可显式指定half启用fp16累加特化
    class ElementAccumulator = void, class ElementBias = void,
    class TileCopy = Conv::Tile::PackedTileCopyTla<
        typename DispatchPolicy::ArchTag, ElementFmap, layout::NC1HWC0, ElementFilter, layout::CI1KHKWCOCI0,
        ElementOutput, layout::NC1HWC0, ElementBias, false, Conv::Tile::ScaleGranularity::NO_QUANT, ElementAccumulator>,
    class TileMmad =
        Gemm::Tile::TileMmadTla<typename DispatchPolicy::ArchTag, ElementFmap, typename TileCopy::LayoutTagL1A>>
struct BlockConv2dTla {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockConv2dTla is not implemented for this DispatchPolicy");
};

} // namespace Catlass::Conv::Block

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
#include "catlass/conv/block/block_conv2d_pingpong.hpp"
#include "catlass/conv/block/block_conv3d_pingpong_bias.hpp"
#endif

#include "catlass/conv/block/block_conv2d_pingpong_tla.hpp"

#endif // CATLASS_CONV_BLOCK_BLOCK_CONV_HPP
