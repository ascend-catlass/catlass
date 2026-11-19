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

template <typename Src, AscendC::Reg::RegLayout Layout>
__simd_callee__ inline vector_f32 expSub(VectorReg<Src> lhs, VectorReg<Src> rhs, ave_preg predicate)
{
    AscendC::Reg::RegTensor<Src> left, right;
    AscendC::Reg::RegTensor<float> result;
    left.reg[0] = lhs;
    right.reg[0] = rhs;
    AscendC::Reg::MaskReg mask = convertAVEPregToVecBool(predicate);
    AscendC::Reg::ExpSub<float, Src, Layout, AscendC::Reg::MaskMergeMode::ZEROING>(result, left, right, mask);
    return result.reg[0];
}

} // namespace

extern "C" {
__aiv__ __attribute__((always_inline)) vector_f32 exp_sub_f16_even(vector_f16 lhs, vector_f16 rhs, ave_preg predicate)
{
    return expSub<half, AscendC::Reg::RegLayout::ZERO>(lhs, rhs, predicate);
}

__aiv__ __attribute__((always_inline)) vector_f32 exp_sub_f16_odd(vector_f16 lhs, vector_f16 rhs, ave_preg predicate)
{
    return expSub<half, AscendC::Reg::RegLayout::ONE>(lhs, rhs, predicate);
}

__aiv__ __attribute__((always_inline)) vector_f32 exp_sub_f32(vector_f32 lhs, vector_f32 rhs, ave_preg predicate)
{
    return expSub<float, AscendC::Reg::RegLayout::ZERO>(lhs, rhs, predicate);
}
}

#endif
