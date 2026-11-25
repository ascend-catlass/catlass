#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

namespace {
using namespace AscendC::Reg;

template <typename T, bool Subtract>
__simd_callee__ inline VectorReg<T> computeCarry(
    VectorReg<T> lhs, VectorReg<T> rhs, ave_preg carryIn, ave_preg mask, __ubuf__ ave_preg* carryOut)
{
    RegTensor<T> x, y, result;
    x.reg[0] = lhs;
    y.reg[0] = rhs;
    MaskReg cin = convertAVEPregToVecBool(carryIn);
    MaskReg active = convertAVEPregToVecBool(mask);
    MaskReg cout;
    if constexpr (Subtract) {
        SubC(cout, result, x, y, cin, active);
    } else {
        AddC(cout, result, x, y, cin, active);
    }
    // Keep this as an ordinary store to the VF's private out parameter.
    // Predicate memory intrinsics prevent LLVM from forwarding it after linking.
    *carryOut = *reinterpret_cast<ave_preg*>(&cout);
    return result.reg[0];
}
} // namespace

extern "C" {
#define DEFINE_B32_CARRY_OP(name, suffix, dtype, subtract)                                                        \
    __aiv__ __attribute__((always_inline)) VectorReg<dtype> tla_##name##_##suffix(                                \
        VectorReg<dtype> lhs, VectorReg<dtype> rhs, ave_preg carryIn, ave_preg mask, __ubuf__ ave_preg* carryOut) \
    {                                                                                                             \
        return computeCarry<dtype, subtract>(lhs, rhs, carryIn, mask, carryOut);                                  \
    }

DEFINE_B32_CARRY_OP(addc, i32, int32_t, false)
DEFINE_B32_CARRY_OP(addc, u32, uint32_t, false)
DEFINE_B32_CARRY_OP(subc, i32, int32_t, true)
DEFINE_B32_CARRY_OP(subc, u32, uint32_t, true)

#undef DEFINE_B32_CARRY_OP

__aiv__ __attribute__((always_inline)) ave_preg tla_b32_read_carry(__ubuf__ ave_preg* carryOut)
{
    return *carryOut;
}
}

#endif
