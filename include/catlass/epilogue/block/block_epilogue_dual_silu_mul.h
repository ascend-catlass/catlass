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

#include "catlass/epilogue/tile/tile_silu_mul.h"

namespace Catlass::Epilogue::Block {

template <
    class ElementOutput, class ElementInput, uint32_t BufferElements, int32_t V_MTE3_EVENT_ID = 0,
    int32_t MTE3_V_EVENT_ID = 1>
struct BlockEpilogueDualSiluMul {
    static constexpr uint32_t BUFFER_ELEMENTS = BufferElements;
    static constexpr int32_t V_MTE3_EVENT_ID_VALUE = V_MTE3_EVENT_ID;
    static constexpr int32_t MTE3_V_EVENT_ID_VALUE = MTE3_V_EVENT_ID;

    CATLASS_DEVICE static auto MakeInputTensor(uint32_t bufferIndex)
    {
        return AscendC::LocalTensor<ElementInput>(
            AscendC::TPosition::VECCALC, bufferIndex * BufferElements * sizeof(ElementInput), BufferElements);
    }

    CATLASS_DEVICE static auto MakeOutputTensor()
    {
        return MakeInputTensor(2);
    }

    CATLASS_DEVICE static auto MakeCastTensor()
    {
        return AscendC::LocalTensor<ElementOutput>(
            AscendC::TPosition::VECCALC, 3 * BufferElements * sizeof(ElementInput), BufferElements);
    }
    template <class TensorOutput, class TensorD0, class TensorD1, class TensorCast, class GlobalTensor>
    CATLASS_DEVICE void operator()(
        TensorOutput& output, TensorD0& d0, TensorD1& d1, TensorCast& castOutput, GlobalTensor& gmD, uint64_t dstOffset,
        uint32_t localM, uint32_t actualN, uint32_t strideN, uint64_t dstStride, uint32_t elementCount)
    {
        AscendC::DataCopyExtParams copyParams(
            localM, actualN * sizeof(ElementOutput), (strideN - actualN) / (BYTE_PER_C0 / sizeof(ElementOutput)),
            (dstStride - actualN) * sizeof(ElementOutput), 0);
        Tile::TileSiluMul<ElementOutput>{}(output, d0, d1, castOutput, elementCount);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(V_MTE3_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(V_MTE3_EVENT_ID);
        AscendC::DataCopyPad(gmD[dstOffset], castOutput, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT_ID);
    }
};

} // namespace Catlass::Epilogue::Block
