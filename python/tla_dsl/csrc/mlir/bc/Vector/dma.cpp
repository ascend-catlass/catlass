#include "../dma_common.h"

#include "catlass/epilogue/tile/tile_copy.hpp"

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyGMToUB(
    memref_t<__gm__ T, 2>* src, memref_t<__ubuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeGMTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeUBTensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyGm2UbTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToGM(
    memref_t<__ubuf__ T, 1>* src, memref_t<__gm__ T, 2>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeUBTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeGMTensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyUb2GmTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToL1(
    memref_t<__ubuf__ T, 1>* src, memref_t<__cbuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeUBTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL1Tensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyUb2L1Tla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

// fp8 on the vector routes, staged as bytes.
//
// The AIV backend refuses fp8 outright -- "type only supports pointer
// operations, scalar float type semantics are not supported" -- and the UB
// copy templates trip that, though the cube's DMA-only ones do not. Pointer
// operations *are* allowed, so the symbol keeps the fp8 name and signature the
// route resolves to, and only the copy template is instantiated on int8_t. A
// copy does not interpret the element type beyond stride and layout
// arithmetic, and fp8 is a byte, so the generated move is the same one.
//
// This is the byte-staging workaround that every fp8 kernel used to carry by
// hand, moved down to where it belongs: the caller now sees a typed fp8 tile.
#define REGISTER_GM_TO_UB_AS_BYTES(LayoutSrc, LayoutDst, DType)                                                       \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##LayoutSrc##_to_ub_##LayoutDst##_##DType(      \
        memref_t<__gm__ DType, 2>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyGMToUB<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, int8_t>(                     \
            reinterpret_cast<memref_t<__gm__ int8_t, 2>*>(src), reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(dst), \
            TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                                \
    }

#define REGISTER_UB_TO_GM_AS_BYTES(LayoutSrc, LayoutDst, DType)                                                       \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_gm_##LayoutDst##_##DType(      \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__gm__ DType, 2>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyUBToGM<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, int8_t>(                     \
            reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(src), reinterpret_cast<memref_t<__gm__ int8_t, 2>*>(dst), \
            TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                                \
    }

#define REGISTER_UB_TO_L1_AS_BYTES(LayoutSrc, LayoutDst, DType)                                                  \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_l1_##LayoutDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToL1<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, int8_t>(                \
            reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(src),                                                \
            reinterpret_cast<memref_t<__cbuf__ int8_t, 1>*>(dst), TENSOR_DESC_12(src), TENSOR_DESC_12(dst));     \
    }

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

#define REGISTER_GM_TO_UB(LayoutSrc, LayoutDst, DType)                                                                \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##LayoutSrc##_to_ub_##LayoutDst##_##DType(      \
        memref_t<__gm__ DType, 2>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyGMToUB<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

#define REGISTER_UB_TO_GM(LayoutSrc, LayoutDst, DType)                                                                \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_gm_##LayoutDst##_##DType(      \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__gm__ DType, 2>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyUBToGM<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

#define REGISTER_UB_TO_L1(LayoutSrc, LayoutDst, DType)                                                           \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_l1_##LayoutDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToL1<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                 \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                 \
    }

REGISTER_GM_TO_UB(RowMajor, RowMajor, float)
REGISTER_GM_TO_UB(RowMajor, RowMajor, half)
REGISTER_GM_TO_UB(RowMajor, RowMajor, bf16)
REGISTER_GM_TO_UB(RowMajor, RowMajor, int32_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, int16_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, int8_t)
REGISTER_GM_TO_UB_AS_BYTES(RowMajor, RowMajor, fp8_e4m3fn_t)
REGISTER_GM_TO_UB_AS_BYTES(RowMajor, RowMajor, fp8_e5m2_t)

REGISTER_UB_TO_GM(RowMajor, RowMajor, float)
REGISTER_UB_TO_GM(RowMajor, RowMajor, half)
REGISTER_UB_TO_GM(RowMajor, RowMajor, bf16)
REGISTER_UB_TO_GM(RowMajor, RowMajor, int32_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, int16_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, int8_t)
REGISTER_UB_TO_GM_AS_BYTES(RowMajor, RowMajor, fp8_e4m3fn_t)
REGISTER_UB_TO_GM_AS_BYTES(RowMajor, RowMajor, fp8_e5m2_t)

REGISTER_UB_TO_L1(RowMajor, zN, float)
REGISTER_UB_TO_L1(RowMajor, zN, half)
REGISTER_UB_TO_L1(RowMajor, zN, bf16)
REGISTER_UB_TO_L1(zN, zN, float)
REGISTER_UB_TO_L1(zN, zN, half)
REGISTER_UB_TO_L1(zN, zN, bf16)
REGISTER_UB_TO_L1(RowMajor, zN, int8_t)
REGISTER_UB_TO_L1(zN, zN, int8_t)
REGISTER_UB_TO_L1_AS_BYTES(RowMajor, zN, fp8_e4m3fn_t)
REGISTER_UB_TO_L1_AS_BYTES(RowMajor, zN, fp8_e5m2_t)
REGISTER_UB_TO_L1_AS_BYTES(zN, zN, fp8_e4m3fn_t)
REGISTER_UB_TO_L1_AS_BYTES(zN, zN, fp8_e5m2_t)

#endif
}
