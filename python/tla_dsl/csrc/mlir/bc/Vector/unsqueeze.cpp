#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

namespace {

// VUSQZ merges into its destination register, so it must start at zero:
// "Vd need be initialized as all '0' before calling this VUSQZ".
template <typename T>
__aiv__ __attribute__((always_inline)) VectorReg<T> pregVunsqueeze(ave_preg preg)
{
    vector_bool mask = convertAVEPregToVecBool(preg);
    VectorReg<T> dst{};
    vusqz(dst, mask);
    return dst;
}

#define DECLARE_VUNSQUEEZE(dtype) \
    __aiv__ __attribute__((always_inline)) VectorReg<dtype> vunsqueeze_##dtype(ave_preg preg)

#define REGISTE_VUNSQUEEZE(dtype)           \
    DECLARE_VUNSQUEEZE(dtype)               \
    {                                       \
        return pregVunsqueeze<dtype>(preg); \
    }

} // namespace

extern "C" {
REGISTE_VUNSQUEEZE(int8_t);
REGISTE_VUNSQUEEZE(int16_t);
REGISTE_VUNSQUEEZE(int32_t);
REGISTE_VUNSQUEEZE(uint8_t);
REGISTE_VUNSQUEEZE(uint16_t);
REGISTE_VUNSQUEEZE(uint32_t);
}

#endif
