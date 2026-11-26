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
__simd_callee__ inline VectorReg<T> axpy(VectorReg<T> x, VectorReg<T> y, T alpha, ave_preg predicate)
{
    AscendC::Reg::RegTensor<T> src, dst;
    src.reg[0] = x;
    dst.reg[0] = y;
    AscendC::Reg::MaskReg mask = convertAVEPregToVecBool(predicate);
    AscendC::Reg::Axpy<T, T, AscendC::Reg::MaskMergeMode::ZEROING>(dst, src, alpha, mask);
    return dst.reg[0];
}

} // namespace

extern "C" {
__aiv__ __attribute__((always_inline)) vector_f16
tla_axpy_f16(vector_f16 x, vector_f16 y, half alpha, ave_preg predicate)
{
    return axpy(x, y, alpha, predicate);
}

__aiv__ __attribute__((always_inline)) vector_f32
tla_axpy_f32(vector_f32 x, vector_f32 y, float alpha, ave_preg predicate)
{
    return axpy(x, y, alpha, predicate);
}

__aiv__ __attribute__((always_inline)) vector_s64
tla_axpy_i64(vector_s64 x, vector_s64 y, int64_t alpha, ave_preg predicate)
{
    return axpy(x, y, alpha, predicate);
}

__aiv__ __attribute__((always_inline)) vector_u64
tla_axpy_u64(vector_u64 x, vector_u64 y, uint64_t alpha, ave_preg predicate)
{
    return axpy(x, y, alpha, predicate);
}
}

#endif
