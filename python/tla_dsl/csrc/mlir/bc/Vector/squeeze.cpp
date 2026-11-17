#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

namespace {

template <typename T>
__aiv__ __attribute__((always_inline)) VectorReg<T> pregVsqueeze(VectorReg<T> src, ave_preg preg)
{
    vector_bool mask = convertAVEPregToVecBool(preg);
    VectorReg<T> dst;
    vsqz(dst, src, mask, MODE_NO_STORED);
    return dst;
}

#define DECLARE_VSQUEEZE(dtype) \
    __aiv__ __attribute__((always_inline)) VectorReg<dtype> vsqueeze_##dtype(VectorReg<dtype> src, ave_preg preg)

#define REGISTE_VSQUEEZE(dtype)                \
    DECLARE_VSQUEEZE(dtype)                    \
    {                                          \
        return pregVsqueeze<dtype>(src, preg); \
    }

} // namespace

extern "C" {
REGISTE_VSQUEEZE(float);
REGISTE_VSQUEEZE(half);
REGISTE_VSQUEEZE(int8_t);
REGISTE_VSQUEEZE(int16_t);
REGISTE_VSQUEEZE(int32_t);
REGISTE_VSQUEEZE(uint8_t);
REGISTE_VSQUEEZE(uint16_t);
REGISTE_VSQUEEZE(uint32_t);
}

#endif
