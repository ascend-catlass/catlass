#include "../common.h"
#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

extern "C" {

// load with block stride (vsldb): one instruction gathers 8 DataBlocks whose
// heads are blockStride DataBlocks (32B units) apart; blockStride == 0 repeats
// the first DataBlock. repeatStride is the POST_MODE_NORMAL compile-time
// address pre-offset in 32B DataBlocks (AscendC DataCopyImpl POST_MODE_NORMAL:
// advance the pointer by repeatStride * 32B, then issue the base form).
//   strideConfig[31:16] = (uint16_t)block_stride
//   strideConfig[15:0]  = 0 (POST_MODE_NORMAL: pre-offset done above)
#define REGISTER_VSLDB(Dtype, dtype)                                                                    \
    __aiv__ __attribute__((always_inline)) VectorReg<Dtype> _mlir_ciface_load_with_stride_##dtype(      \
        memref_t<__ubuf__ Dtype, 1>* srcUb, int32_t blockStride, int32_t repeatStride, ave_preg preg)   \
    {                                                                                                   \
        __ubuf__ Dtype* srcAddr = srcUb->aligned + srcUb->offset + repeatStride * (32 / sizeof(Dtype)); \
        int32_t strideConfig = blockStride << 16;                                                       \
        vector_bool mask = convertAVEPregToVecBool(preg);                                               \
        VectorReg<Dtype> dstReg;                                                                        \
        vsldb(dstReg, srcAddr, strideConfig, mask);                                                     \
        return dstReg;                                                                                  \
    }

REGISTER_VSLDB(float, float)
REGISTER_VSLDB(half, half)
REGISTER_VSLDB(bfloat16_t, bf16)
REGISTER_VSLDB(int32_t, int32)
REGISTER_VSLDB(uint32_t, uint32)
REGISTER_VSLDB(int16_t, int16)
REGISTER_VSLDB(uint16_t, uint16)
REGISTER_VSLDB(int8_t, int8)

// 8-bit block load, same reasoning as the store side: the fp8 symbol keeps the
// fp8 memref parameter the route resolves to and returns the register as
// int8_t, which the caller bitcasts back.
#define REGISTER_VSLDB_AS_BYTES(Dtype, dtype)                                                         \
    __aiv__ __attribute__((always_inline)) VectorReg<int8_t> _mlir_ciface_load_with_stride_##dtype(   \
        memref_t<__ubuf__ Dtype, 1>* srcUb, int32_t blockStride, int32_t repeatStride, ave_preg preg) \
    {                                                                                                 \
        auto* srcBytes = reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(srcUb);                      \
        __ubuf__ int8_t* srcAddr = srcBytes->aligned + srcBytes->offset + repeatStride * 32;          \
        int32_t strideConfig = blockStride << 16;                                                     \
        vector_bool mask = convertAVEPregToVecBool(preg);                                             \
        VectorReg<int8_t> dstReg;                                                                     \
        vsldb(dstReg, srcAddr, strideConfig, mask);                                                   \
        return dstReg;                                                                                \
    }

REGISTER_VSLDB_AS_BYTES(fp8_e4m3fn_t, fp8_e4m3fn)
REGISTER_VSLDB_AS_BYTES(fp8_e5m2_t, fp8_e5m2)
}
#endif
