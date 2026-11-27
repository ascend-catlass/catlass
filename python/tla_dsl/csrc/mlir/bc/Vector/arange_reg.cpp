/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

extern "C" __aiv__ __attribute__((always_inline)) VectorReg<int64_t> arange_int64_increase(int64_t base)
{
    using namespace AscendC::Reg;
    RegTensor<int64_t, RegTraitNumOne> result;
    Arange<int64_t, IndexOrder::INCREASE_ORDER>(result, base);
    return result.reg[0];
}

extern "C" __aiv__ __attribute__((always_inline)) VectorReg<int64_t> arange_int64_decrease(int64_t base)
{
    using namespace AscendC::Reg;
    // The SDK's B64 single-register Arange drops the order argument. Generate
    // the B32 descending ramp and widen its last 32 offsets (31 through 0).
    RegTensor<int32_t, RegTraitNumOne> indices;
    Arange<int32_t, IndexOrder::DECREASE_ORDER>(indices, int32_t{0});
    RegTensor<int64_t, RegTraitNumOne> lanes, result;
    UnPack<int64_t, int32_t, HighLowPart::HIGHEST>(lanes, indices);
    MaskReg mask = CreateMask<int64_t, MaskPattern::ALL>();
    Adds(result, lanes, base, mask);
    return result.reg[0];
}

#endif
