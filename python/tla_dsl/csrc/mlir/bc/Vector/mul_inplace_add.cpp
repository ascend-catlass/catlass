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
__simd_callee__ inline VectorReg<T> mulInplaceAdd(
    VectorReg<T> accumulator, VectorReg<T> lhs, VectorReg<T> rhs, ave_preg predicate)
{
    using namespace AscendC::Reg;
    RegTensor<T, RegTraitNumOne> dst, left, right;
    dst.reg[0] = accumulator;
    left.reg[0] = lhs;
    right.reg[0] = rhs;
    MaskReg mask = convertAVEPregToVecBool(predicate);
    MulAddDst<T, MaskMergeMode::ZEROING>(dst, left, right, mask);
    return dst.reg[0];
}

} // namespace

extern "C" {
__aiv__ __attribute__((always_inline)) vector_s64
tla_mul_inplace_add_i64(vector_s64 accumulator, vector_s64 lhs, vector_s64 rhs, ave_preg predicate)
{
    return mulInplaceAdd<int64_t>(accumulator, lhs, rhs, predicate);
}

__aiv__ __attribute__((always_inline)) vector_u64
tla_mul_inplace_add_u64(vector_u64 accumulator, vector_u64 lhs, vector_u64 rhs, ave_preg predicate)
{
    return mulInplaceAdd<uint64_t>(accumulator, lhs, rhs, predicate);
}
}

#endif
