#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include <type_traits>

#include "vector_reg_utils.h"

// Pack truncates each source element to its low half-width and writes the
// result into the selected destination half; Unpack sign/zero-extends a
// half-width source half into a full-width destination. The native
// vpack/vunpack instructions take a compile-time ::HiloPart selector.
//
// 64-bit elements: Ascend 950 has no native vpack/vunpack for that width.
// AscendC::Reg::Pack composes 64->32 with zero fill and DeInterleave.
// AscendC::Reg::UnPack composes 32->64 with sign/zero fill and Interleave.
// RegTensor<T> (RegTraitNumOne) holds a single raw vector register.

namespace {

template <typename T, typename U, bool High>
__aiv__ __attribute__((always_inline)) VectorReg<T> packRegImpl(VectorReg<U> src)
{
    if constexpr (sizeof(U) == 8) {
        constexpr auto part = static_cast<AscendC::Reg::HighLowPart>(High);
        AscendC::Reg::RegTensor<T> dstReg;
        AscendC::Reg::RegTensor<U> srcReg;
        srcReg.reg[0] = src;
        AscendC::Reg::Pack<T, U, part>(dstReg, srcReg);
        return dstReg.reg[0];
    } else {
        VectorReg<T> dst;
        vpack(dst, src, std::integral_constant<::HiloPart, static_cast<::HiloPart>(High)>());
        return dst;
    }
}

template <typename T, typename U, bool High>
__aiv__ __attribute__((always_inline)) VectorReg<T> unpackRegImpl(VectorReg<U> src)
{
    if constexpr (sizeof(T) == 8) {
        constexpr auto part = static_cast<AscendC::Reg::HighLowPart>(High);
        AscendC::Reg::RegTensor<T> dstReg;
        AscendC::Reg::RegTensor<U> srcReg;
        srcReg.reg[0] = src;
        AscendC::Reg::UnPack<T, U, part>(dstReg, srcReg);
        return dstReg.reg[0];
    } else {
        VectorReg<T> dst;
        vunpack(dst, src, std::integral_constant<::HiloPart, static_cast<::HiloPart>(High)>());
        return dst;
    }
}

} // namespace

#define REGISTER_PACK_REG(dst, src, part, high)                                                                 \
    __aiv__ __attribute__((always_inline)) VectorReg<dst> pack_reg_##dst##_##src##_##part(VectorReg<src> value) \
    {                                                                                                           \
        return packRegImpl<dst, src, high>(value);                                                              \
    }

#define REGISTER_UNPACK_REG(dst, src, part, high)                                                                 \
    __aiv__ __attribute__((always_inline)) VectorReg<dst> unpack_reg_##dst##_##src##_##part(VectorReg<src> value) \
    {                                                                                                             \
        return unpackRegImpl<dst, src, high>(value);                                                              \
    }

#define REGISTER_PACK_REG_PARTS(dst, src)   \
    REGISTER_PACK_REG(dst, src, low, false) \
    REGISTER_PACK_REG(dst, src, high, true)

#define REGISTER_UNPACK_REG_PARTS(dst, src)   \
    REGISTER_UNPACK_REG(dst, src, low, false) \
    REGISTER_UNPACK_REG(dst, src, high, true)

extern "C" {
REGISTER_PACK_REG_PARTS(uint8_t, int16_t)
REGISTER_PACK_REG_PARTS(uint8_t, uint16_t)
REGISTER_PACK_REG_PARTS(uint16_t, int32_t)
REGISTER_PACK_REG_PARTS(uint16_t, uint32_t)
REGISTER_PACK_REG_PARTS(uint32_t, int64_t)
REGISTER_PACK_REG_PARTS(uint32_t, uint64_t)
REGISTER_UNPACK_REG_PARTS(int16_t, int8_t)
REGISTER_UNPACK_REG_PARTS(uint16_t, uint8_t)
REGISTER_UNPACK_REG_PARTS(int32_t, int16_t)
REGISTER_UNPACK_REG_PARTS(uint32_t, uint16_t)
REGISTER_UNPACK_REG_PARTS(int64_t, int32_t)
REGISTER_UNPACK_REG_PARTS(uint64_t, uint32_t)
}

#undef REGISTER_UNPACK_REG_PARTS
#undef REGISTER_PACK_REG_PARTS
#undef REGISTER_UNPACK_REG
#undef REGISTER_PACK_REG

#endif
