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

using namespace AscendC::Reg;

// Each operand uses one physical register and the native b64 predicate.
#define TLA_REG_UNARY(name, type, suffix, op)                                                \
    extern "C" __aiv__ __attribute__((always_inline)) VectorReg<type> tla_##name##_##suffix( \
        VectorReg<type> data, ave_preg preg)                                                 \
    {                                                                                        \
        RegTensor<type> src, dst;                                                            \
        src.reg[0] = data;                                                                   \
        auto mask = convertAVEPregToVecBool(preg);                                           \
        op(dst, src, mask);                                                                  \
        return dst.reg[0];                                                                   \
    }

#define TLA_REG_BINARY(name, type, suffix, op)                                               \
    extern "C" __aiv__ __attribute__((always_inline)) VectorReg<type> tla_##name##_##suffix( \
        VectorReg<type> lhs, VectorReg<type> rhs, ave_preg preg)                             \
    {                                                                                        \
        RegTensor<type> x, y, dst;                                                           \
        x.reg[0] = lhs;                                                                      \
        y.reg[0] = rhs;                                                                      \
        auto mask = convertAVEPregToVecBool(preg);                                           \
        op(dst, x, y, mask);                                                                 \
        return dst.reg[0];                                                                   \
    }

#define TLA_REG_SCALAR(name, type, suffix, op)                                               \
    extern "C" __aiv__ __attribute__((always_inline)) VectorReg<type> tla_##name##_##suffix( \
        VectorReg<type> lhs, type rhs, ave_preg preg)                                        \
    {                                                                                        \
        RegTensor<type> x, y, dst;                                                           \
        x.reg[0] = lhs;                                                                      \
        auto mask = convertAVEPregToVecBool(preg);                                           \
        Duplicate(y, rhs, mask);                                                             \
        op(dst, x, y, mask);                                                                 \
        return dst.reg[0];                                                                   \
    }

#define TLA_REG_INT64(type, suffix)                \
    TLA_REG_UNARY(bitwise_not, type, suffix, Not)  \
    TLA_REG_BINARY(bitwise_and, type, suffix, And) \
    TLA_REG_BINARY(bitwise_or, type, suffix, Or)   \
    TLA_REG_BINARY(bitwise_xor, type, suffix, Xor) \
    TLA_REG_BINARY(add, type, suffix, Add)         \
    TLA_REG_BINARY(sub, type, suffix, Sub)         \
    TLA_REG_BINARY(mul, type, suffix, Mul)         \
    TLA_REG_BINARY(div, type, suffix, Div)         \
    TLA_REG_BINARY(min, type, suffix, Min)         \
    TLA_REG_SCALAR(adds, type, suffix, Add)        \
    TLA_REG_SCALAR(subs, type, suffix, Sub)        \
    TLA_REG_SCALAR(muls, type, suffix, Mul)        \
    TLA_REG_SCALAR(divs, type, suffix, Div)        \
    TLA_REG_SCALAR(mins, type, suffix, Min)

TLA_REG_INT64(int64_t, i64)
TLA_REG_INT64(uint64_t, u64)
TLA_REG_UNARY(abs, int64_t, i64, Abs)
TLA_REG_UNARY(neg, int64_t, i64, Neg)
TLA_REG_BINARY(max, uint64_t, u64, Max)
TLA_REG_SCALAR(maxs, uint64_t, u64, Max)

extern "C" __aiv__ __attribute__((always_inline)) VectorReg<int64_t> tla_abs_zeroing_i64(
    VectorReg<int64_t> data, ave_preg preg)
{
    RegTensor<int64_t> result, zeroed;
    result.reg[0] = tla_abs_i64(data, preg);
    auto mask = convertAVEPregToVecBool(preg);
    MaskReg packed, wordMask, unused;
    MaskPack(packed, mask);
    pintlv_b32(wordMask, unused, packed, packed);

    // Apply each b64 mask bit to both words in the result register.
    auto& words = reinterpret_cast<RegTensor<uint32_t>&>(result);
    vand(
        reinterpret_cast<RegTensor<uint32_t>&>(zeroed), words, words, wordMask,
        GetMaskMergeMode<MaskMergeMode::ZEROING>());
    return zeroed.reg[0];
}

#undef TLA_REG_INT64
#undef TLA_REG_SCALAR
#undef TLA_REG_BINARY
#undef TLA_REG_UNARY

#endif
