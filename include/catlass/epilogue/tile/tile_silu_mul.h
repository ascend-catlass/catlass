/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <type_traits>

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Tile {

template <class ElementOutput>
struct TileSiluMul {
    template <class TensorOutput, class TensorD0, class TensorD1, class TensorCast>
    CATLASS_DEVICE void operator()(
        TensorOutput& output, TensorD0& d0, TensorD1& d1, TensorCast& castOutput, uint32_t elementCount)
    {
        AscendC::Silu(output, d0, elementCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(output, output, d1, elementCount);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (std::is_same_v<ElementOutput, half>) {
            AscendC::Cast(castOutput, output, AscendC::RoundMode::CAST_NONE, elementCount);
        } else {
            AscendC::Cast(castOutput, output, AscendC::RoundMode::CAST_RINT, elementCount);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
};

} // namespace Catlass::Epilogue::Tile
