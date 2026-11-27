#include "../common.h"
#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

// MaskLoadDist US/DS → CCE plds dist=1 / 2.
// Bypasses AscendNPU-IR i1→plds hardcoding dist=0 (and HIVMAVE US/DS
// enum values 24/25 which are not CCE MaskDist 1/2).
//
// One stub per UB element *byte width* (_b8/_b16/_b32/_b64). The memref
// element type only affects address arithmetic (aligned + offset);
// plds's intrinsic ABI is always __ubuf__ uint32_t*.

extern "C" {

#define MASK_LOAD_US_DS_STUBS(Suffix, ElemTy)                                          \
    __aiv__ __attribute__((always_inline)) ave_preg _mlir_ciface_mask_load_us##Suffix( \
        memref_t<__ubuf__ ElemTy, 1>* maskUb)                                          \
    {                                                                                  \
        __ubuf__ ElemTy* addr = maskUb->aligned + maskUb->offset;                      \
        __ubuf__ uint32_t* ptr = reinterpret_cast<__ubuf__ uint32_t*>(addr);           \
        vector_bool pred;                                                              \
        plds(pred, ptr, 0, US);                                                        \
        return *reinterpret_cast<ave_preg*>(&pred);                                    \
    }                                                                                  \
    __aiv__ __attribute__((always_inline)) ave_preg _mlir_ciface_mask_load_ds##Suffix( \
        memref_t<__ubuf__ ElemTy, 1>* maskUb)                                          \
    {                                                                                  \
        __ubuf__ ElemTy* addr = maskUb->aligned + maskUb->offset;                      \
        __ubuf__ uint32_t* ptr = reinterpret_cast<__ubuf__ uint32_t*>(addr);           \
        vector_bool pred;                                                              \
        plds(pred, ptr, 0, DS);                                                        \
        return *reinterpret_cast<ave_preg*>(&pred);                                    \
    }

MASK_LOAD_US_DS_STUBS(_b8, uint8_t)
MASK_LOAD_US_DS_STUBS(_b16, uint16_t)
MASK_LOAD_US_DS_STUBS(_b32, uint32_t)
MASK_LOAD_US_DS_STUBS(_b64, uint64_t)

#undef MASK_LOAD_US_DS_STUBS

// The predicate instructions advance in bytes regardless of the UB element type.
#define MASK_LOAD_UPDATE_STUB(Name, Distribution)                                                         \
    __aiv__ __attribute__((always_inline)) ave_preg Name(                                                 \
        __ubuf__ uint32_t* ptr, int32_t byteStride, __ubuf__ uint32_t** next)                             \
    {                                                                                                     \
        AscendC::Reg::MaskReg pred;                                                                       \
        AscendC::Reg::LoadAlign<                                                                          \
            uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::MaskDist::Distribution>( \
            pred, ptr, byteStride);                                                                       \
        *next = ptr;                                                                                      \
        return *reinterpret_cast<ave_preg*>(&pred);                                                       \
    }

MASK_LOAD_UPDATE_STUB(_mlir_ciface_mask_load_update, DIST_NORM)
MASK_LOAD_UPDATE_STUB(_mlir_ciface_mask_load_us_update, DIST_US)
MASK_LOAD_UPDATE_STUB(_mlir_ciface_mask_load_ds_update, DIST_DS)

#undef MASK_LOAD_UPDATE_STUB

__aiv__ __attribute__((always_inline)) void _mlir_ciface_mask_store_pack(__ubuf__ uint32_t* ptr, ave_preg value)
{
    auto pred = convertAVEPregToVecBool(value);
    AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::MaskDist::DIST_PACK>(ptr, pred);
}

#define MASK_STORE_UPDATE_STUB(Name, Distribution)                                                        \
    __aiv__ __attribute__((always_inline)) __ubuf__ uint32_t* Name(                                       \
        __ubuf__ uint32_t* ptr, ave_preg value, int32_t byteStride)                                       \
    {                                                                                                     \
        auto pred = convertAVEPregToVecBool(value);                                                       \
        AscendC::Reg::StoreAlign<                                                                         \
            uint32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE, AscendC::Reg::MaskDist::Distribution>( \
            ptr, pred, byteStride);                                                                       \
        return ptr;                                                                                       \
    }

MASK_STORE_UPDATE_STUB(_mlir_ciface_mask_store_update, DIST_NORM)
MASK_STORE_UPDATE_STUB(_mlir_ciface_mask_store_pack_update, DIST_PACK)

#undef MASK_STORE_UPDATE_STUB

} // extern "C"

#endif
