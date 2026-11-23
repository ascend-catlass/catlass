#include "../dma_common.h"

#include "catlass/epilogue/tile/tile_copy.hpp"

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyGMToUB(
    memref_t<__gm__ T, 2>* src, memref_t<__ubuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc,
    int64_t gmCachePolicy = kCachePolicyNormal, int64_t padEnable = 0, int64_t leftPadding = 0,
    int64_t rightPadding = 0, int64_t paddingValueBits = 0)
{
    auto srcTensor = makeGMTensor<SrcLayout, T>(src, srcDesc, gmCachePolicy);
    auto dstTensor = makeUBTensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyGm2UbTla<ArchTag, decltype(srcTensor), decltype(dstTensor)> copyGm2Ub;
    if (padEnable == 0) {
        (void)leftPadding;
        (void)rightPadding;
        (void)paddingValueBits;
        copyGm2Ub(dstTensor, srcTensor);
        return;
    }
    // Pad path: only instantiate ExtParams for types known safe under ccec.
    // Keep aligned with core_api._COPY_PAD_SAFE_DTYPES (frontend rejects others).
    if constexpr (
        __is_same(T, float) || __is_same(T, half) || __is_same(T, bfloat16_t) || __is_same(T, int32_t) ||
        __is_same(T, uint32_t) || __is_same(T, int16_t) || __is_same(T, uint16_t) || __is_same(T, int8_t) ||
        __is_same(T, uint8_t) || __is_same(T, bool) || __is_same(T, int64_t) || __is_same(T, uint64_t) ||
        __is_same(T, double)) {
        T padValue{};
        if constexpr (__is_same(T, bool)) {
            // Keep bool as 0/1; do not bit-cast arbitrary byte patterns into bool.
            padValue = static_cast<bool>(paddingValueBits != 0);
        } else if constexpr (sizeof(T) == 1) {
            uint8_t v = static_cast<uint8_t>(paddingValueBits);
            padValue = *reinterpret_cast<T*>(&v);
        } else if constexpr (sizeof(T) == 2) {
            uint16_t v = static_cast<uint16_t>(paddingValueBits);
            padValue = *reinterpret_cast<T*>(&v);
        } else if constexpr (sizeof(T) == 4) {
            uint32_t v = static_cast<uint32_t>(paddingValueBits);
            padValue = *reinterpret_cast<T*>(&v);
        } else {
            (void)paddingValueBits;
            padValue = static_cast<T>(0);
        }
        // Brace-init: DataCopyPadExtParams is an aggregate under -std=c++17.
        AscendC::DataCopyPadExtParams<T> padParams{
            true, static_cast<uint8_t>(leftPadding), static_cast<uint8_t>(rightPadding), padValue};
        copyGm2Ub(dstTensor, srcTensor, padParams);
    } else {
        // Frontend rejects CopyPadParams on these T; keep exact-byte fallback.
        (void)leftPadding;
        (void)rightPadding;
        (void)paddingValueBits;
        copyGm2Ub(dstTensor, srcTensor);
    }
}

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToGM(
    memref_t<__ubuf__ T, 1>* src, memref_t<__gm__ T, 2>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc,
    int64_t gmCachePolicy = kCachePolicyNormal)
{
    auto srcTensor = makeUBTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeGMTensor<DstLayout, T>(dst, dstDesc, gmCachePolicy);
    Catlass::Epilogue::Tile::CopyUb2GmTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

// UB -> UB. dst RowMajor lowers to MOV.UB.TO.UB; dst ColumnMajor is an
// in-memory transpose lowered to VNCHWCONV. core_api.py guards the geometry;
// the block arithmetic and the move live in CopyUb2UbTla.
template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToUB(
    memref_t<__ubuf__ T, 1>* src, memref_t<__ubuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeUBTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeUBTensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyUb2UbTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

// Packed fp4 rides the plain UB -> L1 route: the math is in bytes and DataCopy
// gets a uint8 view.
template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToL1(
    memref_t<__ubuf__ T, 1>* src, memref_t<__cbuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeUBTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL1Tensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyUb2L1Tla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

#define REGISTER_GM_TO_UB(LayoutSrc, LayoutDst, DType)                                                                 \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##LayoutSrc##_to_ub_##LayoutDst##_##DType(       \
        memref_t<__gm__ DType, 2>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst),  \
        int64_t gmCachePolicy, int64_t padEnable, int64_t leftPadding, int64_t rightPadding, int64_t paddingValueBits) \
    {                                                                                                                  \
        copyGMToUB<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                       \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), gmCachePolicy, padEnable, leftPadding, rightPadding,   \
            paddingValueBits);                                                                                         \
    }

#define REGISTER_UB_TO_GM(LayoutSrc, LayoutDst, DType)                                                                \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_gm_##LayoutDst##_##DType(      \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__gm__ DType, 2>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), \
        int64_t gmCachePolicy)                                                                                        \
    {                                                                                                                 \
        copyUBToGM<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), gmCachePolicy);                                       \
    }

#define REGISTER_UB_TO_UB(LayoutSrc, LayoutDst, DType)                                                           \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_ub_##LayoutDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToUB<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                 \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                 \
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
// Unsigned staging. A copy does not interpret the element type beyond stride
// and layout arithmetic, so these are the signed moves at the same widths; they
// exist as their own symbols because the route resolves by element type.
REGISTER_GM_TO_UB(RowMajor, RowMajor, uint8_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, uint16_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, uint32_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, int64_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, uint64_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, double)
REGISTER_GM_TO_UB(VectorLayout, VectorLayout, int64_t)
REGISTER_GM_TO_UB(VectorLayout, VectorLayout, uint64_t)
REGISTER_GM_TO_UB(VectorLayout, VectorLayout, double)
REGISTER_GM_TO_UB(VectorLayout, VectorLayout, bool)
REGISTER_GM_TO_UB(VectorLayout, VectorLayout, float)
REGISTER_GM_TO_UB(VectorLayout, VectorLayout, half)
REGISTER_GM_TO_UB(VectorLayout, VectorLayout, bf16)
REGISTER_GM_TO_UB(VectorLayout, VectorLayout, int32_t)
REGISTER_GM_TO_UB(VectorLayout, VectorLayout, int16_t)
REGISTER_GM_TO_UB(VectorLayout, VectorLayout, int8_t)
REGISTER_GM_TO_UB(VectorLayout, VectorLayout, uint8_t)
REGISTER_GM_TO_UB(VectorLayout, VectorLayout, uint16_t)
REGISTER_GM_TO_UB(VectorLayout, VectorLayout, uint32_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, fp8_e4m3fn_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, fp8_e5m2_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, float4_e2m1x2_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, float4_e1m2x2_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, float8_e8m0_t)
// HiFloat8: IR tla.hif8 / memref i8; tile Copy*Tla reinterprets as uint8 DataCopyPad.
REGISTER_GM_TO_UB(RowMajor, RowMajor, hifloat8_t)
// Bool / i1: AscendC b8 (sizeof==1, SizeOfBits==8); Tile Copy*Tla accepts bool
// like int8_t. Symbol suffix is ``bool`` (copyRuntimeElemSuffix).
REGISTER_GM_TO_UB(RowMajor, RowMajor, bool)

REGISTER_UB_TO_GM(RowMajor, RowMajor, float)
REGISTER_UB_TO_GM(RowMajor, RowMajor, half)
REGISTER_UB_TO_GM(RowMajor, RowMajor, bf16)
REGISTER_UB_TO_GM(RowMajor, RowMajor, int32_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, int16_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, int8_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, uint8_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, uint16_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, uint32_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, int64_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, uint64_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, double)
REGISTER_UB_TO_GM(RowMajor, RowMajor, fp8_e4m3fn_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, fp8_e5m2_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, float4_e2m1x2_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, float4_e1m2x2_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, float8_e8m0_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, hifloat8_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, bool)

REGISTER_UB_TO_UB(RowMajor, RowMajor, float)
REGISTER_UB_TO_UB(RowMajor, RowMajor, half)
REGISTER_UB_TO_UB(RowMajor, RowMajor, bf16)
REGISTER_UB_TO_UB(RowMajor, RowMajor, int32_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, int16_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, int8_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, uint32_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, uint16_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, uint8_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, fp8_e4m3fn_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, fp8_e5m2_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, bool)
// MX types keep their own denomination here: the block scale is one byte and
// packed fp4 counts 4-bit elements in the template, so neither is a byte move.
REGISTER_UB_TO_UB(RowMajor, RowMajor, float8_e8m0_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, float4_e2m1x2_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, float4_e1m2x2_t)

// RowMajor -> ColumnMajor: the same logical (M, N) tile lands as an (N, M)
// array, so the move is a transpose in memory. The destination layout selects
// the VNCHWCONV specialisation of CopyUb2UbTla, which carries every UB element
// width as bytes: fp8 and the unsigned widths go through the same moves as the
// signed ones at the same width.
REGISTER_UB_TO_UB(RowMajor, ColumnMajor, float)
REGISTER_UB_TO_UB(RowMajor, ColumnMajor, half)
REGISTER_UB_TO_UB(RowMajor, ColumnMajor, bf16)
REGISTER_UB_TO_UB(RowMajor, ColumnMajor, int32_t)
REGISTER_UB_TO_UB(RowMajor, ColumnMajor, int16_t)
REGISTER_UB_TO_UB(RowMajor, ColumnMajor, int8_t)
REGISTER_UB_TO_UB(RowMajor, ColumnMajor, uint32_t)
REGISTER_UB_TO_UB(RowMajor, ColumnMajor, uint16_t)
REGISTER_UB_TO_UB(RowMajor, ColumnMajor, uint8_t)
REGISTER_UB_TO_UB(RowMajor, ColumnMajor, fp8_e4m3fn_t)
REGISTER_UB_TO_UB(RowMajor, ColumnMajor, fp8_e5m2_t)
// Bool transpose: CopyUb2UbTla VNCHWCONV b8 path (SizeOfBits==8).
REGISTER_UB_TO_UB(RowMajor, ColumnMajor, bool)

REGISTER_UB_TO_L1(RowMajor, zN, float)
REGISTER_UB_TO_L1(RowMajor, zN, half)
REGISTER_UB_TO_L1(RowMajor, zN, bf16)
REGISTER_UB_TO_L1(RowMajor, zN, int8_t)
REGISTER_UB_TO_L1(RowMajor, zN, int16_t)
REGISTER_UB_TO_L1(RowMajor, zN, int32_t)
REGISTER_UB_TO_L1(RowMajor, zN, uint8_t)
REGISTER_UB_TO_L1(RowMajor, zN, uint16_t)
REGISTER_UB_TO_L1(RowMajor, zN, uint32_t)
REGISTER_UB_TO_L1(RowMajor, zN, fp8_e4m3fn_t)
REGISTER_UB_TO_L1(RowMajor, zN, fp8_e5m2_t)
REGISTER_UB_TO_L1(RowMajor, zN, float4_e2m1x2_t)
REGISTER_UB_TO_L1(RowMajor, zN, float4_e1m2x2_t)
REGISTER_UB_TO_L1(RowMajor, zN, bool)
REGISTER_UB_TO_L1(zN, zN, float)
REGISTER_UB_TO_L1(zN, zN, half)
REGISTER_UB_TO_L1(zN, zN, bf16)
REGISTER_UB_TO_L1(zN, zN, int8_t)
REGISTER_UB_TO_L1(zN, zN, int16_t)
REGISTER_UB_TO_L1(zN, zN, int32_t)
REGISTER_UB_TO_L1(zN, zN, uint8_t)
REGISTER_UB_TO_L1(zN, zN, uint16_t)
REGISTER_UB_TO_L1(zN, zN, uint32_t)
REGISTER_UB_TO_L1(zN, zN, fp8_e4m3fn_t)
REGISTER_UB_TO_L1(zN, zN, fp8_e5m2_t)
REGISTER_UB_TO_L1(zN, zN, float4_e2m1x2_t)
REGISTER_UB_TO_L1(zN, zN, float4_e1m2x2_t)
REGISTER_UB_TO_L1(zN, zN, bool)
// The MX block scale is an e8m0 exponent byte on both sides, so it rides the
// typed UB -> L1 route like fp8: the specialization that lands the scale block
// the L1 -> L0 load reads is chosen by layout order, not by element type.
REGISTER_UB_TO_L1(zZMxScale, zZMxScale, float8_e8m0_t)
REGISTER_UB_TO_L1(nNMxScale, nNMxScale, float8_e8m0_t)

#endif
}
