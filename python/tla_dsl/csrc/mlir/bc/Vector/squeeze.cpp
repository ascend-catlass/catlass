#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

namespace {

template <typename T, bool storeReg = false>
__aiv__ __attribute__((always_inline)) VectorReg<T> pregVsqueeze(VectorReg<T> src, ave_preg preg)
{
    vector_bool mask = convertAVEPregToVecBool(preg);
    VectorReg<T> dst;
    if constexpr (storeReg)
        vsqz(dst, src, mask, MODE_STORED);
    else
        vsqz(dst, src, mask, MODE_NO_STORED);
    return dst;
}

#define DECLARE_VSQUEEZE(dtype) \
    __aiv__ __attribute__((always_inline)) VectorReg<dtype> vsqueeze_##dtype(VectorReg<dtype> src, ave_preg preg)

#define REGISTE_VSQUEEZE(dtype)                                                           \
    DECLARE_VSQUEEZE(dtype)                                                               \
    {                                                                                     \
        return pregVsqueeze<dtype>(src, preg);                                            \
    }                                                                                     \
    __aiv__ __attribute__((always_inline)) VectorReg<dtype> vsqueeze_##dtype##_store_reg( \
        VectorReg<dtype> src, ave_preg preg)                                              \
    {                                                                                     \
        return pregVsqueeze<dtype, true>(src, preg);                                      \
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

__aiv__ __attribute__((always_inline)) void squeeze_stream_clear()
{
    sprclr(SPR_AR);
}

#define REGISTER_SQUEEZE_STORE(dtype)                                               \
    __aiv__ __attribute__((always_inline)) vector_align_data squeeze_store_##dtype( \
        VectorReg<dtype> src, __ubuf__ dtype* dst, vector_align_data state)         \
    {                                                                               \
        vector_align aligned;                                                       \
        aligned = state;                                                            \
        vstur(aligned, src, dst, POST_UPDATE);                                      \
        return aligned.Data;                                                        \
    }

REGISTER_SQUEEZE_STORE(float)
REGISTER_SQUEEZE_STORE(half)
REGISTER_SQUEEZE_STORE(int8_t)
REGISTER_SQUEEZE_STORE(uint8_t)
REGISTER_SQUEEZE_STORE(int32_t)
REGISTER_SQUEEZE_STORE(uint32_t)
REGISTER_SQUEEZE_STORE(int16_t)
REGISTER_SQUEEZE_STORE(uint16_t)

__aiv__ __attribute__((always_inline)) void squeeze_store_post_uint8_t(__ubuf__ uint8_t* dst, vector_align_data state)
{
    vector_align aligned;
    aligned = state;
    vstar(aligned, dst);
}
}

#endif
