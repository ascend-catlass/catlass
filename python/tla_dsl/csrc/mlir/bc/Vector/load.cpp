#include "../common.h"
#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

#include <cstdint>

// Raw-pointer ABI: the caller provides one aligned pair of native registers.
// SDK RegTensor provides low-precision RegType specializations independently of
// the local VectorReg shim used by the strided block helpers.
template <typename T>
struct alignas(256) LoadDintlvB8Result {
    typename AscendC::Reg::RegTensor<T>::RegType even;
    typename AscendC::Reg::RegTensor<T>::RegType odd;
};

extern "C" {

#define REGISTER_LOAD_DINTLV_B8(Dtype, dtype)                                                            \
    static_assert(sizeof(LoadDintlvB8Result<Dtype>) == 512, "DINTLV_B8 result must hold two registers"); \
    static_assert(offsetof(LoadDintlvB8Result<Dtype>, odd) == 256, "Unexpected second-register offset"); \
    __aiv__ __attribute__((always_inline)) void _mlir_ciface_load_dintlv_b8_##dtype(                     \
        __ubuf__ Dtype* srcAddr, LoadDintlvB8Result<Dtype>* result)                                      \
    {                                                                                                    \
        AscendC::Reg::RegTensor<Dtype> even;                                                             \
        AscendC::Reg::RegTensor<Dtype> odd;                                                              \
        AscendC::Reg::LoadAlign<Dtype, AscendC::Reg::LoadDist::DIST_DINTLV_B8>(even, odd, srcAddr);      \
        result->even = even.reg[0];                                                                      \
        result->odd = odd.reg[0];                                                                        \
    }

REGISTER_LOAD_DINTLV_B8(fp8_e4m3fn_t, fp8_e4m3fn)
REGISTER_LOAD_DINTLV_B8(fp8_e5m2_t, fp8_e5m2)
REGISTER_LOAD_DINTLV_B8(fp8_e8m0_t, fp8_e8m0)
REGISTER_LOAD_DINTLV_B8(hifloat8_t, hifloat8)
REGISTER_LOAD_DINTLV_B8(fp4x2_e2m1_t, fp4x2_e2m1)
REGISTER_LOAD_DINTLV_B8(fp4x2_e1m2_t, fp4x2_e1m2)

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

// DATA_BLOCK_COPY POST_MODE_UPDATE. vsldb.post is not an HIVMC op, so the
// hardware pointer comes back through an i64 slot. The vector stays the
// function result, matching the NORMAL vsldb ABI.
#define REGISTER_VSLDB_POST_UPDATE_RETPTR(Dtype, dtype)                                                               \
    __aiv__ __attribute__((always_inline)) VectorReg<Dtype> _mlir_ciface_load_with_stride_post_update_retptr_##dtype( \
        __ubuf__ Dtype* srcAddr, int32_t blockStride, int32_t repeatStride, ave_preg preg, uint64_t* outAddr)         \
    {                                                                                                                 \
        __ubuf__ Dtype* p = srcAddr;                                                                                  \
        int32_t strideConfig = (blockStride << 16) | (repeatStride & 0xFFFF);                                         \
        vector_bool mask = convertAVEPregToVecBool(preg);                                                             \
        VectorReg<Dtype> dstReg;                                                                                      \
        vsldb(dstReg, p, strideConfig, mask, POST_UPDATE);                                                            \
        *outAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));                                             \
        return dstReg;                                                                                                \
    }

#define REGISTER_VSLDB_POST_UPDATE_RETPTR_AS_BYTES(Dtype, dtype)                                                       \
    __aiv__ __attribute__((always_inline)) VectorReg<int8_t> _mlir_ciface_load_with_stride_post_update_retptr_##dtype( \
        __ubuf__ Dtype* srcAddr, int32_t blockStride, int32_t repeatStride, ave_preg preg, uint64_t* outAddr)          \
    {                                                                                                                  \
        auto* srcBytes = reinterpret_cast<__ubuf__ int8_t*>(srcAddr);                                                  \
        __ubuf__ int8_t* p = srcBytes;                                                                                 \
        int32_t strideConfig = (blockStride << 16) | (repeatStride & 0xFFFF);                                          \
        vector_bool mask = convertAVEPregToVecBool(preg);                                                              \
        VectorReg<int8_t> dstReg;                                                                                      \
        vsldb(dstReg, p, strideConfig, mask, POST_UPDATE);                                                             \
        *outAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));                                              \
        return dstReg;                                                                                                 \
    }

REGISTER_VSLDB_POST_UPDATE_RETPTR(float, float)
REGISTER_VSLDB_POST_UPDATE_RETPTR(half, half)
REGISTER_VSLDB_POST_UPDATE_RETPTR(bfloat16_t, bf16)
REGISTER_VSLDB_POST_UPDATE_RETPTR(int32_t, int32)
REGISTER_VSLDB_POST_UPDATE_RETPTR(uint32_t, uint32)
REGISTER_VSLDB_POST_UPDATE_RETPTR(int16_t, int16)
REGISTER_VSLDB_POST_UPDATE_RETPTR(uint16_t, uint16)
REGISTER_VSLDB_POST_UPDATE_RETPTR(int8_t, int8)
REGISTER_VSLDB_POST_UPDATE_RETPTR_AS_BYTES(fp8_e4m3fn_t, fp8_e4m3fn)
REGISTER_VSLDB_POST_UPDATE_RETPTR_AS_BYTES(fp8_e5m2_t, fp8_e5m2)
}
#endif
