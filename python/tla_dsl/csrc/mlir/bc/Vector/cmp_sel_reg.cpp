#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

namespace {

using namespace AscendC::Reg;

template <typename T>
using B64Register = typename RegTensor<T, RegTraitNumOne>::RegType;

// The AVE library-call ABI cannot legalize a 64-bit vector result; keep the
// 64-bit compare/select in linked bitcode like the 64-bit shift stubs.
template <typename T, AscendC::CMPMODE mode>
__simd_callee__ inline ave_preg compareB64(B64Register<T> lhs, B64Register<T> rhs, ave_preg preg)
{
    RegTensor<T, RegTraitNumOne> a, b;
    a.reg[0] = lhs;
    b.reg[0] = rhs;
    MaskReg in = convertAVEPregToVecBool(preg);
    MaskReg out;
    Compare<T, mode>(out, a, b, in);
    ave_preg result;
    *reinterpret_cast<MaskReg*>(&result) = out;
    return result;
}

template <typename T, AscendC::CMPMODE mode>
__simd_callee__ inline ave_preg compareScalarB64(B64Register<T> src, T scalar, ave_preg preg)
{
    RegTensor<T, RegTraitNumOne> a;
    a.reg[0] = src;
    MaskReg in = convertAVEPregToVecBool(preg);
    MaskReg out;
    Compares<T, mode>(out, a, scalar, in);
    ave_preg result;
    *reinterpret_cast<MaskReg*>(&result) = out;
    return result;
}

template <typename T>
__simd_callee__ inline B64Register<T> selectB64(ave_preg preg, B64Register<T> x, B64Register<T> y)
{
    RegTensor<T, RegTraitNumOne> a, b, dst;
    a.reg[0] = x;
    b.reg[0] = y;
    MaskReg mask = convertAVEPregToVecBool(preg);
    Select<T>(dst, a, b, mask);
    return dst.reg[0];
}

} // namespace

#define REGISTER_CMP(name, mode, dtype)                                              \
    extern "C" __aiv__ __attribute__((always_inline)) ave_preg cmp_##name##_##dtype( \
        B64Register<dtype> lhs, B64Register<dtype> rhs, ave_preg preg)               \
    {                                                                                \
        return compareB64<dtype, AscendC::CMPMODE::mode>(lhs, rhs, preg);            \
    }

// Keep these names in sync with TlaVectorRegionSimdPass::lowerCmp/lowerWhere.
// tests/test_b64_cmp_select_bc_symbols.py asserts every emitted name exists in
// the compiled AIV aggregate.
#define REGISTER_CMPS(name, mode, dtype)                                              \
    extern "C" __aiv__ __attribute__((always_inline)) ave_preg cmps_##name##_##dtype( \
        B64Register<dtype> src, dtype scalar, ave_preg preg)                          \
    {                                                                                 \
        return compareScalarB64<dtype, AscendC::CMPMODE::mode>(src, scalar, preg);    \
    }

#define REGISTER_CMP_DTYPE(dtype) \
    REGISTER_CMP(lt, LT, dtype)   \
    REGISTER_CMP(le, LE, dtype)   \
    REGISTER_CMP(gt, GT, dtype)   \
    REGISTER_CMP(ge, GE, dtype)   \
    REGISTER_CMP(eq, EQ, dtype)   \
    REGISTER_CMP(ne, NE, dtype)   \
    REGISTER_CMPS(lt, LT, dtype)  \
    REGISTER_CMPS(le, LE, dtype)  \
    REGISTER_CMPS(gt, GT, dtype)  \
    REGISTER_CMPS(ge, GE, dtype)  \
    REGISTER_CMPS(eq, EQ, dtype)  \
    REGISTER_CMPS(ne, NE, dtype)

#define REGISTER_SELECT(dtype)                                                           \
    extern "C" __aiv__ __attribute__((always_inline)) B64Register<dtype> select_##dtype( \
        ave_preg preg, B64Register<dtype> x, B64Register<dtype> y)                       \
    {                                                                                    \
        return selectB64<dtype>(preg, x, y);                                             \
    }

REGISTER_CMP_DTYPE(int64_t)
REGISTER_CMP_DTYPE(uint64_t)
REGISTER_SELECT(int64_t)
REGISTER_SELECT(uint64_t)

#undef REGISTER_SELECT
#undef REGISTER_CMP_DTYPE
#undef REGISTER_CMPS
#undef REGISTER_CMP

#endif
