/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_MATMUL_GATHER_SCATTER_TILE_COPY_H
#define CATLASS_GEMM_TILE_MATMUL_GATHER_SCATTER_TILE_COPY_H

#include <type_traits>

#include "catlass/gemm/tile/tile_copy.hpp"

namespace Catlass::Gemm::Tile {

struct CopyPreloadedL1A {
    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const&, TensorSrc const&) const
    {
        // Gather has already materialized A in L1. BlockMmad still invokes its
        // configured A-copy hook to preserve the normal pipeline handshake.
    }
};

template <
    class ArchTag, class ElementA, class LayoutTagA, class ElementB, class LayoutTagB, class ElementTransport,
    class LayoutTagC>
struct MatmulGatherScatterTileCopy
    : public PackedTileCopyTlaToUB<
          ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementTransport, LayoutTagC, void,
          std::is_same_v<ElementTransport, float> ? CopyL0CToUBMode::SPLIT_M : CopyL0CToUBMode::NO_SPLIT> {
    template <class TensorA>
    using CopyGmToL1A = CopyPreloadedL1A;
};

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_MATMUL_GATHER_SCATTER_TILE_COPY_H
