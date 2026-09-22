#include "../common.h"
#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

extern "C" {

// store with block stride
// strideConfig[31:16] = (uint16_t)block_stride
// strideConfig[15:0]  = (uint16_t)repeat_stride
#define REGISTER_VSSTB(Dtype, dtype)                                                                     \
    __aiv__ __attribute__((always_inline)) void _mlir_ciface_store_with_stride_##dtype(                  \
        VectorReg<Dtype> srcReg, memref_t<__ubuf__ Dtype, 1>* dstUb, int32_t blockStride, ave_preg preg) \
    {                                                                                                    \
        __ubuf__ Dtype* dstAddr = dstUb->aligned + dstUb->offset;                                        \
        int32_t strideConfig = blockStride << 16; /* repeat_stride=0 */                                  \
        vector_bool mask = convertAVEPregToVecBool(preg);                                                \
        vsstb(srcReg, dstAddr, strideConfig, mask);                                                      \
    }

// 8-bit block store. vsstb moves DataBlocks: it reads the register and the
// destination address, and never interprets the element type beyond its width
// and the block stride. fp8 is a byte, but the AIV backend refuses fp8 scalar
// semantics, so a VectorReg<fp8_*> parameter will not compile. Keep the fp8
// symbol name and memref type the route resolves to -- a memref parameter is a
// pointer, which is allowed -- and take the register as int8_t; the caller
// bitcasts the vector, which costs nothing.
#define REGISTER_VSSTB_AS_BYTES(Dtype, dtype)                                                             \
    __aiv__ __attribute__((always_inline)) void _mlir_ciface_store_with_stride_##dtype(                   \
        VectorReg<int8_t> srcReg, memref_t<__ubuf__ Dtype, 1>* dstUb, int32_t blockStride, ave_preg preg) \
    {                                                                                                     \
        auto* dstBytes = reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(dstUb);                          \
        __ubuf__ int8_t* dstAddr = dstBytes->aligned + dstBytes->offset;                                  \
        int32_t strideConfig = blockStride << 16; /* repeat_stride=0 */                                   \
        vector_bool mask = convertAVEPregToVecBool(preg);                                                 \
        vsstb(srcReg, dstAddr, strideConfig, mask);                                                       \
    }

REGISTER_VSSTB(float, float)
REGISTER_VSSTB(half, half)
REGISTER_VSSTB(bfloat16_t, bf16)
REGISTER_VSSTB(int8_t, int8)
REGISTER_VSSTB_AS_BYTES(fp8_e4m3fn_t, fp8_e4m3fn)
REGISTER_VSSTB_AS_BYTES(fp8_e5m2_t, fp8_e5m2)
}

#endif
