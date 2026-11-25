#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

// The AVE library-call ABI cannot legalize a <32xi64> broadcast result; keep
// the 64-bit lane splat in linked bitcode like the 64-bit cmp/select stubs.
// Duplicate<int64_t> on a RegTraitNumOne destination builds the scalar in the
// two-register trait shape and packs it back with B64TraitTwoToTraitOne, so the
// returned register matches the packed <32xi64> ABI.

namespace {

using namespace AscendC::Reg;

template <typename T>
using B64Register = typename RegTensor<T, RegTraitNumOne>::RegType;

template <typename T>
__simd_callee__ inline B64Register<T> duplicateScalarB64(T scalar, ave_preg preg)
{
    RegTensor<T, RegTraitNumOne> dst;
    MaskReg mask = convertAVEPregToVecBool(preg);
    Duplicate<T, MaskMergeMode::ZEROING>(dst, scalar, mask);
    return dst.reg[0];
}

// Broadcast lane 0 of a one-lane fragment; the source stays in the register
// file (the SIMD backend cannot read a vector lane into a scalar register).
template <typename T>
__simd_callee__ inline B64Register<T> duplicateVectorB64(B64Register<T> src, ave_preg preg)
{
    RegTensor<T, RegTraitNumOne> dst, srcReg;
    srcReg.reg[0] = src;
    MaskReg mask = convertAVEPregToVecBool(preg);
    Duplicate<T, HighLowPart::LOWEST, MaskMergeMode::ZEROING>(dst, srcReg, mask);
    return dst.reg[0];
}

} // namespace

#define REGISTER_DUP_B64(dtype)                                                                                   \
    extern "C" __aiv__ __attribute__((always_inline)) B64Register<dtype> dup_##dtype(dtype scalar, ave_preg preg) \
    {                                                                                                             \
        return duplicateScalarB64<dtype>(scalar, preg);                                                           \
    }                                                                                                             \
    extern "C" __aiv__ __attribute__((always_inline)) B64Register<dtype> dup_vec_##dtype(                         \
        B64Register<dtype> src, ave_preg preg)                                                                    \
    {                                                                                                             \
        return duplicateVectorB64<dtype>(src, preg);                                                              \
    }

REGISTER_DUP_B64(int64_t)
REGISTER_DUP_B64(uint64_t)

#undef REGISTER_DUP_B64

#endif
