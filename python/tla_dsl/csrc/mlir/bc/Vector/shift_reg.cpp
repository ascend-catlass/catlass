#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

namespace {

using namespace AscendC::Reg;

template <typename T>
using B64Register = typename RegTensor<T, RegTraitNumOne>::RegType;

template <typename T, bool Left>
__simd_callee__ inline B64Register<T> shiftVector(B64Register<T> data, B64Register<int64_t> amount, ave_preg preg)
{
    RegTensor<T, RegTraitNumOne> src, dst;
    RegTensor<int64_t, RegTraitNumOne> counts;
    src.reg[0] = data;
    counts.reg[0] = amount;
    auto mask = convertAVEPregToVecBool(preg);
    // Pass the original counts through to Ascend C, including large counts.
    if constexpr (Left)
        ShiftLeft(dst, src, counts, mask);
    else
        ShiftRight(dst, src, counts, mask);
    return dst.reg[0];
}

template <typename T, bool Left>
__simd_callee__ inline B64Register<T> shiftScalar(B64Register<T> data, int16_t amount, ave_preg preg)
{
    RegTensor<T, RegTraitNumOne> src, dst;
    src.reg[0] = data;
    auto mask = convertAVEPregToVecBool(preg);
    if constexpr (Left)
        ShiftLefts(dst, src, amount, mask);
    else
        ShiftRights(dst, src, amount, mask);
    return dst.reg[0];
}

} // namespace

#define REGISTER_SHIFT(name, implementation, left, dtype, amount_type)                           \
    extern "C" __aiv__ __attribute__((always_inline)) B64Register<dtype> shift_##name##_##dtype( \
        B64Register<dtype> data, amount_type amount, ave_preg preg)                              \
    {                                                                                            \
        return implementation<dtype, left>(data, amount, preg);                                  \
    }

#define REGISTER_SHIFT_DTYPE(dtype)                                        \
    REGISTER_SHIFT(left, shiftVector, true, dtype, B64Register<int64_t>)   \
    REGISTER_SHIFT(right, shiftVector, false, dtype, B64Register<int64_t>) \
    REGISTER_SHIFT(lefts, shiftScalar, true, dtype, int16_t)               \
    REGISTER_SHIFT(rights, shiftScalar, false, dtype, int16_t)

REGISTER_SHIFT_DTYPE(int64_t)
REGISTER_SHIFT_DTYPE(uint64_t)

#undef REGISTER_SHIFT_DTYPE
#undef REGISTER_SHIFT

#endif
