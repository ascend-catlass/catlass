// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

namespace {

template <AscendC::Reg::RegLayout Layout>
__simd_callee__ inline vector_f16 mulsCast(vector_f32 src, float scalar, ave_preg predicate)
{
    AscendC::Reg::RegTensor<float> input;
    AscendC::Reg::RegTensor<half> dst;
    input.reg[0] = src;
    AscendC::Reg::MaskReg mask = convertAVEPregToVecBool(predicate);
    AscendC::Reg::MulsCast<half, float, float, Layout>(dst, input, scalar, mask);
    return dst.reg[0];
}

} // namespace

extern "C" {
__aiv__ __attribute__((always_inline)) vector_f16 tla_muls_cast_even(vector_f32 src, float scalar, ave_preg predicate)
{
    return mulsCast<AscendC::Reg::RegLayout::ZERO>(src, scalar, predicate);
}

__aiv__ __attribute__((always_inline)) vector_f16 tla_muls_cast_odd(vector_f32 src, float scalar, ave_preg predicate)
{
    return mulsCast<AscendC::Reg::RegLayout::ONE>(src, scalar, predicate);
}
}

#endif
