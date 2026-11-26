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

template <typename T>
__simd_callee__ inline VectorReg<T> inplaceMulAdd(
    VectorReg<T> dst, VectorReg<T> multiplier, VectorReg<T> addend, ave_preg preg)
{
    AscendC::Reg::RegTensor<T> value, factor, bias;
    value.reg[0] = dst;
    factor.reg[0] = multiplier;
    bias.reg[0] = addend;
    AscendC::Reg::MaskReg mask = convertAVEPregToVecBool(preg);
    AscendC::Reg::MulDstAdd(value, factor, bias, mask);
    return value.reg[0];
}

} // namespace

extern "C" {
__aiv__ __attribute__((always_inline)) vector_f16
inplace_mul_add_f16(vector_f16 dst, vector_f16 multiplier, vector_f16 addend, ave_preg preg)
{
    return inplaceMulAdd<half>(dst, multiplier, addend, preg);
}

__aiv__ __attribute__((always_inline)) vector_bf16
inplace_mul_add_bf16(vector_bf16 dst, vector_bf16 multiplier, vector_bf16 addend, ave_preg preg)
{
    return inplaceMulAdd<bfloat16_t>(dst, multiplier, addend, preg);
}

__aiv__ __attribute__((always_inline)) vector_f32
inplace_mul_add_f32(vector_f32 dst, vector_f32 multiplier, vector_f32 addend, ave_preg preg)
{
    return inplaceMulAdd<float>(dst, multiplier, addend, preg);
}
}

#endif
