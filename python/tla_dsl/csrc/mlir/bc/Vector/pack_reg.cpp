#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include <type_traits>

#include "vector_reg_utils.h"

// Pack truncates each source element to its low half-width and writes the
// result into the selected destination half; Unpack sign/zero-extends a
// half-width source half into a full-width destination. Both map 1:1 onto
// the vpack/vunpack instructions (same instruction AscendC::Reg::Pack/UnPack
// forwards to for these widths on this device); the destination half is a
// compile-time-constant ::HiloPart, built the same way AscendC's own
// dav_3510 implementation builds it -- by static_cast'ing the LOW/HIGH
// selector into ::HiloPart's underlying integer value, not by naming its
// enumerators (which this toolchain never declares in a header).
//
// 64-bit elements (int64_t/uint64_t) are not supported: there is no native
// vpack/vunpack for that width, and composing one out of other instructions
// is not implemented here.

namespace {

template <typename T, typename U, bool High>
__aiv__ __attribute__((always_inline)) VectorReg<T> packRegImpl(VectorReg<U> src)
{
    VectorReg<T> dst;
    vpack(dst, src, std::integral_constant<::HiloPart, static_cast<::HiloPart>(High)>());
    return dst;
}

template <typename T, typename U, bool High>
__aiv__ __attribute__((always_inline)) VectorReg<T> unpackRegImpl(VectorReg<U> src)
{
    VectorReg<T> dst;
    vunpack(dst, src, std::integral_constant<::HiloPart, static_cast<::HiloPart>(High)>());
    return dst;
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
REGISTER_UNPACK_REG_PARTS(int16_t, int8_t)
REGISTER_UNPACK_REG_PARTS(uint16_t, uint8_t)
REGISTER_UNPACK_REG_PARTS(int32_t, int16_t)
REGISTER_UNPACK_REG_PARTS(uint32_t, uint16_t)
}

#undef REGISTER_UNPACK_REG_PARTS
#undef REGISTER_PACK_REG_PARTS
#undef REGISTER_UNPACK_REG
#undef REGISTER_PACK_REG

#endif
