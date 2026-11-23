#include "../dma_common.h"

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#include "catlass/gemm/tile/copy_l0c_to_l1.hpp"
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T, typename... BatchArgs>
CATLASS_DEVICE void copyGMToL1(
    memref_t<__gm__ T, 2>* src, memref_t<__cbuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc,
    int64_t gmCachePolicy, BatchArgs... batchArgs)
{
    // Every route here is served by one TileCopyTla specialization: the tag
    // selects both the layout constructor (makeGMTensor / makeL1Tensor) and the
    // copy. VectorLayout is the flat case -- both endpoints come out rank-1
    // (VectorLayout), which is exactly what the VectorLayout specialization in
    // copy_gm_to_l1.hpp matches. The gemv A operand is that same flat route.
    auto srcTensor = makeGMTensor<SrcLayout, T>(src, srcDesc, gmCachePolicy);
    auto dstTensor = makeL1Tensor<DstLayout, T>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(
        dstTensor, srcTensor, batchArgs...);
}

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyL1ToL0A(
    memref_t<__cbuf__ T, 1>* src, memref_t<__ca__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    // Both endpoints are selected by tag, which is what lets the zN/nZ matrix
    // forms and the flat VectorLayout gemv form share one helper: makeL1Tensor /
    // makeL0ATensor build the tensors, and copy_l1_to_l0a.hpp matches on their
    // shapes.
    auto srcTensor = makeL1Tensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL0ATensor<DstLayout, T>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

template <class ArchTag, LayoutTag SrcLayout, typename T>
CATLASS_DEVICE void copyL1ToL0B(
    memref_t<__cbuf__ T, 1>* src, memref_t<__cb__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeL1Tensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL0BTensor<LayoutTag::nZ, T>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

// L1 -> bias table. Both endpoints are flat row-major runs, so the copy is the
// VectorLayout specialization in gemm/tile/ascend950/copy_l1_to_bt.hpp -- the
// same template Catlass's own bias pipeline uses. Nothing is reimplemented here.
template <class ArchTag, typename T>
CATLASS_DEVICE void copyL1ToBias(
    memref_t<__cbuf__ T, 1>* src, memref_t<__biasbuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeL1Tensor<LayoutTag::VectorLayout, T>(src, srcDesc);
    auto dstTensor = makeBiasTensor<LayoutTag::VectorLayout, T>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

// L0C holds fp32 MMAD accumulator; ElementDst may be f32/f16/bf16 (Ascend950 fixpipe).
template <class ArchTag, LayoutTag DstLayout, typename ElementSrc, typename ElementDst, bool ReluEnable = false>
CATLASS_DEVICE void copyL0CToGM(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__gm__ ElementDst, 2>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, uint8_t unitFlag, int64_t gmCachePolicy = kCachePolicyNormal)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0Clayout, ElementSrc>(src, srcDesc);
    auto dstTensor = makeGMTensor<DstLayout, ElementDst>(dst, dstDesc, gmCachePolicy);
    Catlass::Gemm::Tile::CopyL0CToGmTla<
        ArchTag, decltype(srcTensor), decltype(dstTensor), Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT,
        ReluEnable>{}(dstTensor, srcTensor, unitFlag);
}

// L0C holds fp32 MMAD accumulator; ElementDst may be f32/f16/bf16 (Ascend950 fixpipe).
template <class ArchTag, LayoutTag DstLayout, typename ElementSrc, typename ElementDst, bool ReluEnable = false>
CATLASS_DEVICE void copyL0CToGMQuantPerTensor(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__gm__ ElementDst, 2>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, uint8_t unitFlag, float quantScale, int32_t quantOffset,
    int64_t gmCachePolicy = kCachePolicyNormal)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0Clayout, ElementSrc>(src, srcDesc);
    auto dstTensor = makeGMTensor<DstLayout, ElementDst>(dst, dstDesc, gmCachePolicy);
    using CopyL0CToGmTla = Catlass::Gemm::Tile::CopyL0CToGmTla<
        ArchTag, decltype(srcTensor), decltype(dstTensor), Catlass::Gemm::Tile::ScaleGranularity::PER_TENSOR,
        ReluEnable>;
    typename CopyL0CToGmTla::Params params(quantScale, quantOffset);
    CopyL0CToGmTla{params}(dstTensor, srcTensor, unitFlag);
}

// L0C -> GM per-channel dequant. The scale tile lives in L1, one uint64 (a packed
// fp32 scale) per output channel, and rides along with the fixpipe as its dequant
// tensor.
template <class ArchTag, LayoutTag DstLayout, typename ElementSrc, typename ElementDst, bool ReluEnable = false>
CATLASS_DEVICE void copyL0CToGMQuantPerChannel(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__gm__ ElementDst, 2>* dst, memref_t<__cbuf__ uint64_t, 1>* scale,
    const TensorDesc& srcDesc, const TensorDesc& dstDesc, const TensorDesc& scaleDesc, uint8_t unitFlag,
    int64_t gmCachePolicy = kCachePolicyNormal)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0Clayout, ElementSrc>(src, srcDesc);
    auto dstTensor = makeGMTensor<DstLayout, ElementDst>(dst, dstDesc, gmCachePolicy);
    auto quantTensor = makeL1Tensor<LayoutTag::VectorLayout, uint64_t>(scale, scaleDesc);
    Catlass::Gemm::Tile::CopyL0CToGmTla<
        ArchTag, decltype(srcTensor), decltype(dstTensor), Catlass::Gemm::Tile::ScaleGranularity::PER_CHANNEL,
        ReluEnable>{}(dstTensor, srcTensor, quantTensor, unitFlag);
}

// L0C holds fp32 MMAD accumulator; ElementDst may be f32/f16/bf16 (Ascend950 fixpipe).
template <class ArchTag, LayoutTag DstLayout, typename ElementSrc, typename ElementDst, bool ReluEnable = false>
CATLASS_DEVICE void copyL0CToL1(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__cbuf__ ElementDst, 1>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, uint8_t unitFlag)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0Clayout, ElementSrc>(src, srcDesc);
    auto dstTensor = makeL1Tensor<DstLayout, ElementDst>(dst, dstDesc);
    Catlass::Gemm::Tile::CopyL0CToL1Tla<
        ArchTag, decltype(srcTensor), decltype(dstTensor), Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT,
        ReluEnable>{}(dstTensor, srcTensor, unitFlag);
}

template <
    class ArchTag, LayoutTag DstLayout, Catlass::Gemm::Tile::CopyL0CToUBMode Mode, typename ElementSrc,
    typename ElementDst, bool ReluEnable = false>
CATLASS_DEVICE void copyL0CToUB(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__ubuf__ ElementDst, 1>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, uint8_t unitFlag, uint8_t subBlockId)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0Clayout, ElementSrc>(src, srcDesc);
    auto dstTensor = makeUBTensor<DstLayout, ElementDst>(dst, dstDesc);
    if constexpr (Mode == Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT) {
        Catlass::Gemm::Tile::CopyL0CToUBTla<
            ArchTag, decltype(srcTensor), decltype(dstTensor), Mode, Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT,
            ReluEnable>{}(dstTensor, srcTensor, (bool)subBlockId, unitFlag);
    } else if constexpr (
        Mode == Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M ||
        Mode == Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_N) {
        Catlass::Gemm::Tile::CopyL0CToUBTla<
            ArchTag, decltype(srcTensor), decltype(dstTensor), Mode, Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT,
            ReluEnable>{}(dstTensor, srcTensor, unitFlag);
    }
}

template <class ArchTag, LayoutTag DstLayout, typename ElementSrc, typename ElementDst, bool ReluEnable = false>
CATLASS_DEVICE void copyL0CToUBQuantPerTensor(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__ubuf__ ElementDst, 1>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, uint8_t unitFlag, uint8_t subBlockId, float quantScale, int32_t quantOffset)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0Clayout, ElementSrc>(src, srcDesc);
    auto dstTensor = makeUBTensor<DstLayout, ElementDst>(dst, dstDesc);
    constexpr auto splitMode = Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT;
    using CopyL0CToUBTla = Catlass::Gemm::Tile::CopyL0CToUBTla<
        ArchTag, decltype(srcTensor), decltype(dstTensor), splitMode, Catlass::Gemm::Tile::ScaleGranularity::PER_TENSOR,
        ReluEnable>;
    typename CopyL0CToUBTla::Params params(quantScale, quantOffset);
    CopyL0CToUBTla{params}(dstTensor, srcTensor, (bool)subBlockId, unitFlag);
}

// L0C -> UB per-channel dequant (NO_SPLIT). Same L1 uint64 scale tile as
// copyL0CToGMQuantPerChannel.
template <class ArchTag, LayoutTag DstLayout, typename ElementSrc, typename ElementDst, bool ReluEnable = false>
CATLASS_DEVICE void copyL0CToUBQuantPerChannel(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__ubuf__ ElementDst, 1>* dst, memref_t<__cbuf__ uint64_t, 1>* scale,
    const TensorDesc& srcDesc, const TensorDesc& dstDesc, const TensorDesc& scaleDesc, uint8_t unitFlag,
    uint8_t subBlockId)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0Clayout, ElementSrc>(src, srcDesc);
    auto dstTensor = makeUBTensor<DstLayout, ElementDst>(dst, dstDesc);
    auto quantTensor = makeL1Tensor<LayoutTag::VectorLayout, uint64_t>(scale, scaleDesc);
    constexpr auto splitMode = Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT;
    Catlass::Gemm::Tile::CopyL0CToUBTla<
        ArchTag, decltype(srcTensor), decltype(dstTensor), splitMode,
        Catlass::Gemm::Tile::ScaleGranularity::PER_CHANNEL, ReluEnable>{}(
        dstTensor, srcTensor, quantTensor, (bool)subBlockId, unitFlag);
}

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
// MX L1 -> L0 loads. The scale rides along with the *load*, not the matmul:
// the copy templates take a third (e8m0, zZ, L1) tensor.
//
// The scale reaches L1 through the ordinary GM -> L1 route. Its layout tag
// (RowMajorMxScaleA / ColMajorMxScaleA / RowMajorMxScaleB / ColMajorMxScaleB)
// selects the matching Catlass MxScale specialisation in copy_gm_to_l1.hpp,
// which lays the e8m0 block out as NZ (Dn2Nz for the row-major orientations,
// Nd2Nz for the column-major ones). Tile coordinate and tile extent travel in
// the tensor layout/coord, so a K-tiled kernel loads its own chunk of exponents
// the same way any other tile copy does. Only the L1 -> L0 hop below needs the
// MX-specific zZ / nN layouts.
//
// One wrapper per side serves both element families. Each names its operand
// element type directly -- fp8_e* or float4_*x2_t -- and the Catlass layout math
// reads the matching width, so fp4 needs no byte alias. The scale is
// float8_e8m0_t either way -- the type the MxScale copy specialisations key on.
// Spelling it that way does not change the C ABI: a memref descriptor carries no
// element type.

// L1 and L0 carry the same element type here: fp8_e* or float4_*x2_t. Nothing
// on the L0 side is an MX-specific alias -- microscaling is not a property of
// the copy at all. It is stated once at the MMA call site instead (GemmMmad vs
// GemmMmadMx in mmad.cpp), because a memref descriptor carries no element type
// that could pick the intrinsic implicitly.
template <class ArchTag, LayoutTag SrcLayout, typename Element>
CATLASS_DEVICE void copyL1ToL0AMx(
    memref_t<__cbuf__ Element, 1>* src, memref_t<__ca__ Element, 1>* dst, memref_t<__cbuf__ float8_e8m0_t, 1>* scale,
    const TensorDesc& srcDesc, const TensorDesc& dstDesc, const TensorDesc& scaleDesc)
{
    auto scaleTensor = makeL1Tensor<LayoutTag::zZMxScale, float8_e8m0_t>(scale, scaleDesc);
    auto srcTensor = makeL1Tensor<SrcLayout, Element>(src, srcDesc);
    auto dstTensor = makeL0ATensor<LayoutTag::zN, Element>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(
        dstTensor, srcTensor, scaleTensor);
}

template <class ArchTag, LayoutTag SrcLayout, typename Element>
CATLASS_DEVICE void copyL1ToL0BMx(
    memref_t<__cbuf__ Element, 1>* src, memref_t<__cb__ Element, 1>* dst, memref_t<__cbuf__ float8_e8m0_t, 1>* scale,
    const TensorDesc& srcDesc, const TensorDesc& dstDesc, const TensorDesc& scaleDesc)
{
    auto scaleTensor = makeL1Tensor<LayoutTag::nNMxScale, float8_e8m0_t>(scale, scaleDesc);
    auto srcTensor = makeL1Tensor<SrcLayout, Element>(src, srcDesc);
    auto dstTensor = makeL0BTensor<LayoutTag::nZ, Element>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(
        dstTensor, srcTensor, scaleTensor);
}
#endif

template <LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyGMToL1Batch(
    memref_t<__gm__ T, 2>* src, memref_t<__cbuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc,
    int64_t count, int64_t srcStep, int64_t dstStep, int64_t srcBatchCoord, int64_t dstBatchCoord,
    int64_t gmCachePolicy)
{
    if (count <= 0 || srcDesc.originShape0 == 0 || srcDesc.originShape1 == 0)
        return;
    // The GM backing memref can be flat. Its pointer and offset are common
    // to both ranks, while srcDesc supplies the matrix geometry.
    memref_t<__gm__ T, 2> source{};
    source.aligned = src->aligned;
    source.offset = src->offset + srcBatchCoord * srcStep;
    auto destination = *dst;
    destination.offset += dstBatchCoord * dstStep;
    copyGMToL1<Catlass::Arch::Ascend950, SrcLayout, DstLayout, T>(
        &source, &destination, srcDesc, dstDesc, gmCachePolicy, static_cast<uint32_t>(count),
        static_cast<uint64_t>(srcStep), static_cast<uint32_t>(dstStep));
}

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

#define REGISTER_GM_TO_L1(LayoutSrc, LayoutDst, DType)                                                                \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##LayoutSrc##_to_l1_##LayoutDst##_##DType(      \
        memref_t<__gm__ DType, 2>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), \
        int64_t gmCachePolicy)                                                                                        \
    {                                                                                                                 \
        copyGMToL1<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), gmCachePolicy);                                       \
    }

#define REGISTER_GM_TO_L1_BATCH(Src, Dst, Type)                                                                     \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##Src##_to_l1_##Dst##_##Type##_batch(         \
        memref_t<__gm__ Type, 2>* src, memref_t<__cbuf__ Type, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), \
        int64_t count, int64_t srcStep, int64_t dstStep, int64_t srcCoord, int64_t dstCoord, int64_t gmCachePolicy) \
    {                                                                                                               \
        copyGMToL1Batch<LayoutTag::Src, LayoutTag::Dst, Type>(                                                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), count, srcStep, dstStep, srcCoord, dstCoord,        \
            gmCachePolicy);                                                                                         \
    }
REGISTER_GM_TO_L1_BATCH(RowMajor, zN, fp8_e4m3fn_t)
REGISTER_GM_TO_L1_BATCH(RowMajor, zN, fp8_e5m2_t)
REGISTER_GM_TO_L1_BATCH(ColumnMajor, nZ, fp8_e4m3fn_t)
REGISTER_GM_TO_L1_BATCH(ColumnMajor, nZ, fp8_e5m2_t)

REGISTER_GM_TO_L1_BATCH(RowMajorMxScaleA, zZMxScale, float8_e8m0_t)
REGISTER_GM_TO_L1_BATCH(ColMajorMxScaleA, zZMxScale, float8_e8m0_t)
REGISTER_GM_TO_L1_BATCH(RowMajorMxScaleB, nNMxScale, float8_e8m0_t)
REGISTER_GM_TO_L1_BATCH(ColMajorMxScaleB, nNMxScale, float8_e8m0_t)

#define REGISTER_L1_TO_L0A(LayoutSrc, LayoutDst, DType)                                                               \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l1_##LayoutSrc##_to_l0a_##LayoutDst##_##DType(     \
        memref_t<__cbuf__ DType, 1>* src, memref_t<__ca__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyL1ToL0A<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                     \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

#define REGISTER_L1_TO_L0B(LayoutSrc, DType)                                                                          \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l1_##LayoutSrc##_to_l0b_nZ_##DType(                \
        memref_t<__cbuf__ DType, 1>* src, memref_t<__cb__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyL1ToL0B<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, DType>(                                           \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

#define REGISTER_L1_TO_BIAS(DType)                                                                                  \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l1_VectorLayout_to_biasbuf_VectorLayout_##DType( \
        memref_t<__cbuf__ DType, 1>* src, memref_t<__biasbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                       \
    {                                                                                                               \
        copyL1ToBias<Catlass::Arch::Ascend950, DType>(src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));          \
    }

// Non-MxMatmul related copy
REGISTER_GM_TO_L1(RowMajor, zN, float)
REGISTER_GM_TO_L1(RowMajor, zN, half)
REGISTER_GM_TO_L1(RowMajor, zN, bf16)
REGISTER_GM_TO_L1(RowMajor, zN, int8_t)
REGISTER_GM_TO_L1(RowMajor, zN, int16_t)
REGISTER_GM_TO_L1(RowMajor, zN, int32_t)
REGISTER_GM_TO_L1(RowMajor, zN, uint8_t)
REGISTER_GM_TO_L1(RowMajor, zN, uint16_t)
REGISTER_GM_TO_L1(RowMajor, zN, uint32_t)
REGISTER_GM_TO_L1(RowMajor, zN, fp8_e4m3fn_t)
REGISTER_GM_TO_L1(RowMajor, zN, fp8_e5m2_t)
REGISTER_GM_TO_L1(RowMajor, zN, float4_e2m1x2_t)
REGISTER_GM_TO_L1(RowMajor, zN, float4_e1m2x2_t)
// Bool / i1: AscendC b8; TileCopyTla accepts bool (SizeOfBits==8) like int8_t.
// HiFloat8: typed BC (memref i8); TileCopyTla byte-moves (Nd2Nz as int8, fractal/VL as uint8).
REGISTER_GM_TO_L1(RowMajor, zN, bool)
REGISTER_GM_TO_L1(RowMajor, zN, hifloat8_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, float)
REGISTER_GM_TO_L1(ColumnMajor, nZ, half)
REGISTER_GM_TO_L1(ColumnMajor, nZ, bf16)
REGISTER_GM_TO_L1(ColumnMajor, nZ, int8_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, int16_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, int32_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, uint8_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, uint16_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, uint32_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, fp8_e4m3fn_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, fp8_e5m2_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, float4_e2m1x2_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, float4_e1m2x2_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, bool)
REGISTER_GM_TO_L1(ColumnMajor, nZ, hifloat8_t)
REGISTER_GM_TO_L1(zN, zN, float)
REGISTER_GM_TO_L1(zN, zN, half)
REGISTER_GM_TO_L1(zN, zN, bf16)
REGISTER_GM_TO_L1(zN, zN, int8_t)
REGISTER_GM_TO_L1(zN, zN, int16_t)
REGISTER_GM_TO_L1(zN, zN, int32_t)
REGISTER_GM_TO_L1(zN, zN, uint8_t)
REGISTER_GM_TO_L1(zN, zN, uint16_t)
REGISTER_GM_TO_L1(zN, zN, uint32_t)
REGISTER_GM_TO_L1(zN, zN, bool)
REGISTER_GM_TO_L1(zN, zN, hifloat8_t)
REGISTER_GM_TO_L1(zN, zN, fp8_e4m3fn_t)
REGISTER_GM_TO_L1(zN, zN, fp8_e5m2_t)
REGISTER_GM_TO_L1(zN, zN, float4_e2m1x2_t)
REGISTER_GM_TO_L1(zN, zN, float4_e1m2x2_t)
REGISTER_GM_TO_L1(nZ, nZ, float)
REGISTER_GM_TO_L1(nZ, nZ, half)
REGISTER_GM_TO_L1(nZ, nZ, bf16)
REGISTER_GM_TO_L1(nZ, nZ, int8_t)
REGISTER_GM_TO_L1(nZ, nZ, int16_t)
REGISTER_GM_TO_L1(nZ, nZ, int32_t)
REGISTER_GM_TO_L1(nZ, nZ, uint8_t)
REGISTER_GM_TO_L1(nZ, nZ, uint16_t)
REGISTER_GM_TO_L1(nZ, nZ, uint32_t)
REGISTER_GM_TO_L1(nZ, nZ, bool)
REGISTER_GM_TO_L1(nZ, nZ, hifloat8_t)
REGISTER_GM_TO_L1(nZ, nZ, fp8_e4m3fn_t)
REGISTER_GM_TO_L1(nZ, nZ, fp8_e5m2_t)
REGISTER_GM_TO_L1(nZ, nZ, float4_e2m1x2_t)
REGISTER_GM_TO_L1(nZ, nZ, float4_e1m2x2_t)

// VectorLayout -> VectorLayout is the flat contiguous run: GM and L1 both come out rank-1
// (VectorLayout), so the VectorLayout TileCopyTla specialization in
// copy_gm_to_l1.hpp serves it. RowMajor has no variant here: a flat tile is
// spelled VectorLayout, and RowMajor stays two-dimensional.
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, float)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, half)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, bf16)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, int8_t)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, int16_t)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, int32_t)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, int64_t)
// The per-channel fixpipe dequant scale tile is the same flat run, one u64 word
// (a packed fp32 scale) per output channel. i64 / double share the same
// DataCopyPad VectorLayout path (8-byte elements, exact byte length).
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, uint8_t)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, uint16_t)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, uint32_t)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, uint64_t)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, double)
// AscendC a5 DataCopyPad continuous GM→L1 lists bool (1 byte / element).
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, bool)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, fp8_e4m3fn_t)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, fp8_e5m2_t)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, hifloat8_t)

REGISTER_L1_TO_L0A(zN, zN, float)
REGISTER_L1_TO_L0A(zN, zN, half)
REGISTER_L1_TO_L0A(zN, zN, bf16)
REGISTER_L1_TO_L0A(zN, zN, int8_t)
// uint16_t is not in AscendC LoadData 2dv2's supported set on dav-3510
// (uint8/uint32 yes; uint16 no). Keep GM→L1 for u16; L1→L0 only for u32.
REGISTER_L1_TO_L0A(zN, zN, uint32_t)
REGISTER_L1_TO_L0A(zN, zN, fp8_e4m3fn_t)
REGISTER_L1_TO_L0A(zN, zN, fp8_e5m2_t)
REGISTER_L1_TO_L0A(zN, zN, hifloat8_t)
REGISTER_L1_TO_L0A(nZ, zN, float)
REGISTER_L1_TO_L0A(nZ, zN, half)
REGISTER_L1_TO_L0A(nZ, zN, bf16)
REGISTER_L1_TO_L0A(nZ, zN, int8_t)
REGISTER_L1_TO_L0A(nZ, zN, uint32_t)
REGISTER_L1_TO_L0A(nZ, zN, fp8_e4m3fn_t)
REGISTER_L1_TO_L0A(nZ, zN, fp8_e5m2_t)
REGISTER_L1_TO_L0A(nZ, zN, hifloat8_t)
REGISTER_L1_TO_L0A(VectorLayout, VectorLayout, float)
REGISTER_L1_TO_L0A(VectorLayout, VectorLayout, half)
REGISTER_L1_TO_L0A(VectorLayout, VectorLayout, bf16)
REGISTER_L1_TO_L0A(VectorLayout, VectorLayout, int8_t)
REGISTER_L1_TO_L0A(VectorLayout, VectorLayout, fp8_e4m3fn_t)
REGISTER_L1_TO_L0A(VectorLayout, VectorLayout, fp8_e5m2_t)
REGISTER_L1_TO_L0A(VectorLayout, VectorLayout, hifloat8_t)

REGISTER_L1_TO_L0B(zN, float)
REGISTER_L1_TO_L0B(zN, half)
REGISTER_L1_TO_L0B(zN, bf16)
REGISTER_L1_TO_L0B(zN, int8_t)
REGISTER_L1_TO_L0B(zN, uint32_t)
REGISTER_L1_TO_L0B(zN, fp8_e4m3fn_t)
REGISTER_L1_TO_L0B(zN, fp8_e5m2_t)
REGISTER_L1_TO_L0B(zN, hifloat8_t)
REGISTER_L1_TO_L0B(nZ, float)
REGISTER_L1_TO_L0B(nZ, half)
REGISTER_L1_TO_L0B(nZ, bf16)
REGISTER_L1_TO_L0B(nZ, int8_t)
REGISTER_L1_TO_L0B(nZ, uint32_t)
REGISTER_L1_TO_L0B(nZ, fp8_e4m3fn_t)
REGISTER_L1_TO_L0B(nZ, fp8_e5m2_t)
REGISTER_L1_TO_L0B(nZ, hifloat8_t)

REGISTER_L1_TO_BIAS(float)
REGISTER_L1_TO_BIAS(int32_t)

// MX L1 -> L0. One macro per side covers fp8 and fp4: the element type names the
// operand as both L1 and L0 hold it, and the scale is always e8m0. Both source
// layouts are registered for both families.
#define REGISTER_L1_TO_L0A_MX(LayoutSrc, DType)                                                                       \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_mx_l1_##LayoutSrc##_to_l0a_zN_##DType(             \
        memref_t<__cbuf__ DType, 1>* src, memref_t<__ca__ DType, 1>* dst, memref_t<__cbuf__ float8_e8m0_t, 1>* scale, \
        DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), DESC_ABI_PARAMS(scale))                                           \
    {                                                                                                                 \
        copyL1ToL0AMx<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, DType>(                                         \
            src, dst, scale, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), TENSOR_DESC_12(scale));                        \
    }

#define REGISTER_L1_TO_L0B_MX(LayoutSrc, DType)                                                                       \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_mx_l1_##LayoutSrc##_to_l0b_nZ_##DType(             \
        memref_t<__cbuf__ DType, 1>* src, memref_t<__cb__ DType, 1>* dst, memref_t<__cbuf__ float8_e8m0_t, 1>* scale, \
        DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), DESC_ABI_PARAMS(scale))                                           \
    {                                                                                                                 \
        copyL1ToL0BMx<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, DType>(                                         \
            src, dst, scale, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), TENSOR_DESC_12(scale));                        \
    }

REGISTER_GM_TO_L1(RowMajorMxScaleA, zZMxScale, float8_e8m0_t)
REGISTER_GM_TO_L1(ColMajorMxScaleA, zZMxScale, float8_e8m0_t)
REGISTER_GM_TO_L1(RowMajorMxScaleB, nNMxScale, float8_e8m0_t)
REGISTER_GM_TO_L1(ColMajorMxScaleB, nNMxScale, float8_e8m0_t)

REGISTER_L1_TO_L0A_MX(zN, fp8_e4m3fn_t)
REGISTER_L1_TO_L0A_MX(zN, fp8_e5m2_t)
REGISTER_L1_TO_L0A_MX(nZ, fp8_e4m3fn_t)
REGISTER_L1_TO_L0A_MX(nZ, fp8_e5m2_t)
REGISTER_L1_TO_L0A_MX(zN, float4_e2m1x2_t)
REGISTER_L1_TO_L0A_MX(zN, float4_e1m2x2_t)
REGISTER_L1_TO_L0A_MX(nZ, float4_e2m1x2_t)
REGISTER_L1_TO_L0A_MX(nZ, float4_e1m2x2_t)

REGISTER_L1_TO_L0B_MX(zN, fp8_e4m3fn_t)
REGISTER_L1_TO_L0B_MX(zN, fp8_e5m2_t)
REGISTER_L1_TO_L0B_MX(nZ, fp8_e4m3fn_t)
REGISTER_L1_TO_L0B_MX(nZ, fp8_e5m2_t)
REGISTER_L1_TO_L0B_MX(zN, float4_e2m1x2_t)
REGISTER_L1_TO_L0B_MX(zN, float4_e1m2x2_t)
REGISTER_L1_TO_L0B_MX(nZ, float4_e2m1x2_t)
REGISTER_L1_TO_L0B_MX(nZ, float4_e1m2x2_t)

#define REGISTER_L0C_TO_GM_NORELU(LayoutDst, DTypeSrc, DTypeDst)                                                    \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l0c_to_gm_##LayoutDst##_##DTypeSrc##_##DTypeDst( \
        memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__gm__ DTypeDst, 2>* dst, DESC_ABI_PARAMS(src),                 \
        DESC_ABI_PARAMS(dst), uint8_t unitFlag, int64_t gmCachePolicy)                                              \
    {                                                                                                               \
        copyL0CToGM<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, false>(                     \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag, gmCachePolicy);                           \
    }

#define REGISTER_L0C_TO_GM_RELU(LayoutDst, DTypeSrc, DTypeDst)                                                      \
    [aicore]                                                                                                        \
        __attribute__((always_inline)) void _mlir_ciface_copy_l0c_to_gm_##LayoutDst##_relu_##DTypeSrc##_##DTypeDst( \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__gm__ DTypeDst, 2>* dst, DESC_ABI_PARAMS(src),             \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, int64_t gmCachePolicy)                                          \
    {                                                                                                               \
        copyL0CToGM<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, true>(                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag, gmCachePolicy);                           \
    }

#define REGISTER_L0C_TO_GM(LayoutDst, DTypeSrc, DTypeDst)    \
    REGISTER_L0C_TO_GM_NORELU(LayoutDst, DTypeSrc, DTypeDst) \
    REGISTER_L0C_TO_GM_RELU(LayoutDst, DTypeSrc, DTypeDst)

#define REGISTER_L0C_TO_GM_PER_TENSOR_NORELU(LayoutDst, DTypeSrc, DTypeDst)                                        \
    [aicore] __attribute__((always_inline)) void                                                                   \
        _mlir_ciface_copy_l0c_to_gm_##LayoutDst##_pertensor_##DTypeSrc##_##DTypeDst(                               \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__gm__ DTypeDst, 2>* dst, DESC_ABI_PARAMS(src),            \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, float quantScale, int32_t quantOffset, int64_t gmCachePolicy)  \
    {                                                                                                              \
        copyL0CToGMQuantPerTensor<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, false>(      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag, quantScale, quantOffset, gmCachePolicy); \
    }

#define REGISTER_L0C_TO_GM_PER_TENSOR_RELU(LayoutDst, DTypeSrc, DTypeDst)                                          \
    [aicore] __attribute__((always_inline)) void                                                                   \
        _mlir_ciface_copy_l0c_to_gm_##LayoutDst##_pertensor_relu_##DTypeSrc##_##DTypeDst(                          \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__gm__ DTypeDst, 2>* dst, DESC_ABI_PARAMS(src),            \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, float quantScale, int32_t quantOffset, int64_t gmCachePolicy)  \
    {                                                                                                              \
        copyL0CToGMQuantPerTensor<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, true>(       \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag, quantScale, quantOffset, gmCachePolicy); \
    }

#define REGISTER_L0C_TO_GM_PER_TENSOR(LayoutDst, DTypeSrc, DTypeDst)    \
    REGISTER_L0C_TO_GM_PER_TENSOR_NORELU(LayoutDst, DTypeSrc, DTypeDst) \
    REGISTER_L0C_TO_GM_PER_TENSOR_RELU(LayoutDst, DTypeSrc, DTypeDst)

#define REGISTER_L0C_TO_UB_PER_TENSOR_NORELU(LayoutDst, DTypeSrc, DTypeDst)                                     \
    [aicore] __attribute__((always_inline)) void                                                                \
        _mlir_ciface_copy_l0c_to_ub_##LayoutDst##_nosplit_pertensor_##DTypeSrc##_##DTypeDst(                    \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__ubuf__ DTypeDst, 1>* dst, DESC_ABI_PARAMS(src),       \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, uint8_t subBlockId, float quantScale, int32_t quantOffset)  \
    {                                                                                                           \
        copyL0CToUBQuantPerTensor<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, false>(   \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag, subBlockId, quantScale, quantOffset); \
    }

#define REGISTER_L0C_TO_UB_PER_TENSOR_RELU(LayoutDst, DTypeSrc, DTypeDst)                                       \
    [aicore] __attribute__((always_inline)) void                                                                \
        _mlir_ciface_copy_l0c_to_ub_##LayoutDst##_nosplit_pertensor_relu_##DTypeSrc##_##DTypeDst(               \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__ubuf__ DTypeDst, 1>* dst, DESC_ABI_PARAMS(src),       \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, uint8_t subBlockId, float quantScale, int32_t quantOffset)  \
    {                                                                                                           \
        copyL0CToUBQuantPerTensor<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, true>(    \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag, subBlockId, quantScale, quantOffset); \
    }

#define REGISTER_L0C_TO_UB_PER_TENSOR(LayoutDst, DTypeSrc, DTypeDst)    \
    REGISTER_L0C_TO_UB_PER_TENSOR_NORELU(LayoutDst, DTypeSrc, DTypeDst) \
    REGISTER_L0C_TO_UB_PER_TENSOR_RELU(LayoutDst, DTypeSrc, DTypeDst)

#define REGISTER_L0C_TO_L1_NORELU(LayoutDst, DTypeSrc, DTypeDst)                                                    \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l0c_to_l1_##LayoutDst##_##DTypeSrc##_##DTypeDst( \
        memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__cbuf__ DTypeDst, 1>* dst, DESC_ABI_PARAMS(src),               \
        DESC_ABI_PARAMS(dst), uint8_t unitFlag)                                                                     \
    {                                                                                                               \
        copyL0CToL1<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, false>(                     \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag);                                          \
    }

#define REGISTER_L0C_TO_L1_RELU(LayoutDst, DTypeSrc, DTypeDst)                                                      \
    [aicore]                                                                                                        \
        __attribute__((always_inline)) void _mlir_ciface_copy_l0c_to_l1_##LayoutDst##_relu_##DTypeSrc##_##DTypeDst( \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__cbuf__ DTypeDst, 1>* dst, DESC_ABI_PARAMS(src),           \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag)                                                                 \
    {                                                                                                               \
        copyL0CToL1<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, true>(                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag);                                          \
    }

#define REGISTER_L0C_TO_L1(LayoutDst, DTypeSrc, DTypeDst)    \
    REGISTER_L0C_TO_L1_NORELU(LayoutDst, DTypeSrc, DTypeDst) \
    REGISTER_L0C_TO_L1_RELU(LayoutDst, DTypeSrc, DTypeDst)

#define REGISTER_L0C_TO_UB_NORELU(LayoutDst, mode, MODE, DTypeSrc, DTypeDst)                                      \
    [aicore] __attribute__((always_inline)) void                                                                  \
        _mlir_ciface_copy_l0c_to_ub_##LayoutDst##_##mode##_##DTypeSrc##_##DTypeDst(                               \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__ubuf__ DTypeDst, 1>* dst, DESC_ABI_PARAMS(src),         \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, uint8_t subBlockId)                                           \
    {                                                                                                             \
        copyL0CToUB<                                                                                              \
            Catlass::Arch::Ascend950, LayoutTag::LayoutDst, Catlass::Gemm::Tile::CopyL0CToUBMode::MODE, DTypeSrc, \
            DTypeDst>(src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag, subBlockId);                  \
    }

#define REGISTER_L0C_TO_UB_RELU(LayoutDst, mode, MODE, DTypeSrc, DTypeDst)                                        \
    [aicore] __attribute__((always_inline)) void                                                                  \
        _mlir_ciface_copy_l0c_to_ub_##LayoutDst##_##mode##_relu_##DTypeSrc##_##DTypeDst(                          \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__ubuf__ DTypeDst, 1>* dst, DESC_ABI_PARAMS(src),         \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, uint8_t subBlockId)                                           \
    {                                                                                                             \
        copyL0CToUB<                                                                                              \
            Catlass::Arch::Ascend950, LayoutTag::LayoutDst, Catlass::Gemm::Tile::CopyL0CToUBMode::MODE, DTypeSrc, \
            DTypeDst, true>(src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag, subBlockId);            \
    }

// ReLU rides only on the no-split fixpipe variants, so the split registrations
// below go through the no-relu macro.
#define REGISTER_L0C_TO_UB(LayoutDst, mode, MODE, DTypeSrc, DTypeDst)    \
    REGISTER_L0C_TO_UB_NORELU(LayoutDst, mode, MODE, DTypeSrc, DTypeDst) \
    REGISTER_L0C_TO_UB_RELU(LayoutDst, mode, MODE, DTypeSrc, DTypeDst)

// Fixpipe copy: NO_QUANT is the only quant mode implemented, so it stays out of
// the symbol name; the registered variants differ by layout, split mode, element
// type, and whether the transfer carries the ReLU epilogue.
REGISTER_L0C_TO_GM(RowMajor, float, float)
REGISTER_L0C_TO_GM(RowMajor, float, half)
REGISTER_L0C_TO_GM(RowMajor, float, bf16)
REGISTER_L0C_TO_GM(RowMajor, int32_t, int32_t)

// L0C -> GM ColumnMajor: the fixpipe NZ2DN path writes the tile transposed
// straight to GM, so no UB staging (and no vector-side pass) is needed.
REGISTER_L0C_TO_GM(ColumnMajor, float, float)
REGISTER_L0C_TO_GM(ColumnMajor, float, half)
REGISTER_L0C_TO_GM(ColumnMajor, float, bf16)
REGISTER_L0C_TO_GM(ColumnMajor, int32_t, int32_t)

// PER_TENSOR Quant
REGISTER_L0C_TO_GM_PER_TENSOR(RowMajor, float, float)
REGISTER_L0C_TO_GM_PER_TENSOR(RowMajor, float, half)
REGISTER_L0C_TO_GM_PER_TENSOR(RowMajor, float, bf16)
REGISTER_L0C_TO_GM_PER_TENSOR(RowMajor, float, int8_t)
REGISTER_L0C_TO_GM_PER_TENSOR(RowMajor, float, uint8_t)
REGISTER_L0C_TO_GM_PER_TENSOR(RowMajor, int32_t, half)
REGISTER_L0C_TO_GM_PER_TENSOR(RowMajor, int32_t, bf16)
REGISTER_L0C_TO_GM_PER_TENSOR(RowMajor, int32_t, int8_t)
REGISTER_L0C_TO_GM_PER_TENSOR(RowMajor, int32_t, uint8_t)
REGISTER_L0C_TO_GM_PER_TENSOR(ColumnMajor, float, float)
REGISTER_L0C_TO_GM_PER_TENSOR(ColumnMajor, float, half)
REGISTER_L0C_TO_GM_PER_TENSOR(ColumnMajor, float, bf16)
REGISTER_L0C_TO_GM_PER_TENSOR(ColumnMajor, float, int8_t)
REGISTER_L0C_TO_GM_PER_TENSOR(ColumnMajor, float, uint8_t)
REGISTER_L0C_TO_GM_PER_TENSOR(ColumnMajor, int32_t, half)
REGISTER_L0C_TO_GM_PER_TENSOR(ColumnMajor, int32_t, bf16)
REGISTER_L0C_TO_GM_PER_TENSOR(ColumnMajor, int32_t, int8_t)
REGISTER_L0C_TO_GM_PER_TENSOR(ColumnMajor, int32_t, uint8_t)

// l0c to ub per-tensor quant are all nosplit
REGISTER_L0C_TO_UB_PER_TENSOR(RowMajor, float, float)
REGISTER_L0C_TO_UB_PER_TENSOR(RowMajor, float, half)
REGISTER_L0C_TO_UB_PER_TENSOR(RowMajor, float, bf16)
REGISTER_L0C_TO_UB_PER_TENSOR(RowMajor, float, int8_t)
REGISTER_L0C_TO_UB_PER_TENSOR(RowMajor, float, uint8_t)
REGISTER_L0C_TO_UB_PER_TENSOR(RowMajor, int32_t, half)
REGISTER_L0C_TO_UB_PER_TENSOR(RowMajor, int32_t, bf16)
REGISTER_L0C_TO_UB_PER_TENSOR(RowMajor, int32_t, int8_t)
REGISTER_L0C_TO_UB_PER_TENSOR(RowMajor, int32_t, uint8_t)
REGISTER_L0C_TO_UB_PER_TENSOR(ColumnMajor, float, float)
REGISTER_L0C_TO_UB_PER_TENSOR(ColumnMajor, float, half)
REGISTER_L0C_TO_UB_PER_TENSOR(ColumnMajor, float, bf16)
REGISTER_L0C_TO_UB_PER_TENSOR(ColumnMajor, float, int8_t)
REGISTER_L0C_TO_UB_PER_TENSOR(ColumnMajor, float, uint8_t)
REGISTER_L0C_TO_UB_PER_TENSOR(ColumnMajor, int32_t, half)
REGISTER_L0C_TO_UB_PER_TENSOR(ColumnMajor, int32_t, bf16)
REGISTER_L0C_TO_UB_PER_TENSOR(ColumnMajor, int32_t, int8_t)
REGISTER_L0C_TO_UB_PER_TENSOR(ColumnMajor, int32_t, uint8_t)

REGISTER_L0C_TO_L1(zN, float, float)
REGISTER_L0C_TO_L1(zN, float, half)
REGISTER_L0C_TO_L1(zN, float, bf16)
REGISTER_L0C_TO_L1(zN, int32_t, int32_t)

// RowMajor nosplit (the only split mode ReLU is registered for)
REGISTER_L0C_TO_UB(RowMajor, nosplit, NO_SPLIT, float, float)
REGISTER_L0C_TO_UB(RowMajor, nosplit, NO_SPLIT, float, half)
REGISTER_L0C_TO_UB(RowMajor, nosplit, NO_SPLIT, float, bf16)
REGISTER_L0C_TO_UB(RowMajor, nosplit, NO_SPLIT, int32_t, int32_t)
// split mode src=dst (no ReLU variant)
REGISTER_L0C_TO_UB_NORELU(RowMajor, splitm, SPLIT_M, float, float)
REGISTER_L0C_TO_UB_NORELU(RowMajor, splitm, SPLIT_M, int32_t, int32_t)
REGISTER_L0C_TO_UB_NORELU(RowMajor, splitn, SPLIT_N, float, float)
REGISTER_L0C_TO_UB_NORELU(RowMajor, splitn, SPLIT_N, int32_t, int32_t)
// ColumnMajor must be nosplit
REGISTER_L0C_TO_UB(ColumnMajor, nosplit, NO_SPLIT, float, float)
REGISTER_L0C_TO_UB(ColumnMajor, nosplit, NO_SPLIT, float, half)
REGISTER_L0C_TO_UB(ColumnMajor, nosplit, NO_SPLIT, float, bf16)
REGISTER_L0C_TO_UB(ColumnMajor, nosplit, NO_SPLIT, int32_t, int32_t)

#define REGISTER_L0C_TO_GM_PER_CHANNEL_NORELU(LayoutDst, DTypeSrc, DTypeDst)                                       \
    [aicore] __attribute__((always_inline)) void                                                                   \
        _mlir_ciface_copy_l0c_to_gm_##LayoutDst##_perchannel_##DTypeSrc##_##DTypeDst(                              \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__gm__ DTypeDst, 2>* dst, DESC_ABI_PARAMS(src),            \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, memref_t<__cbuf__ uint64_t, 1>* quant, DESC_ABI_PARAMS(quant), \
            int64_t gmCachePolicy)                                                                                 \
    {                                                                                                              \
        copyL0CToGMQuantPerChannel<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, false>(     \
            src, dst, quant, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), TENSOR_DESC_12(quant), unitFlag,            \
            gmCachePolicy);                                                                                        \
    }
#define REGISTER_L0C_TO_GM_PER_CHANNEL_RELU(LayoutDst, DTypeSrc, DTypeDst)                                         \
    [aicore] __attribute__((always_inline)) void                                                                   \
        _mlir_ciface_copy_l0c_to_gm_##LayoutDst##_perchannel_relu_##DTypeSrc##_##DTypeDst(                         \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__gm__ DTypeDst, 2>* dst, DESC_ABI_PARAMS(src),            \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, memref_t<__cbuf__ uint64_t, 1>* quant, DESC_ABI_PARAMS(quant), \
            int64_t gmCachePolicy)                                                                                 \
    {                                                                                                              \
        copyL0CToGMQuantPerChannel<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, true>(      \
            src, dst, quant, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), TENSOR_DESC_12(quant), unitFlag,            \
            gmCachePolicy);                                                                                        \
    }
#define REGISTER_L0C_TO_GM_PER_CHANNEL(LayoutDst, DTypeSrc, DTypeDst)    \
    REGISTER_L0C_TO_GM_PER_CHANNEL_NORELU(LayoutDst, DTypeSrc, DTypeDst) \
    REGISTER_L0C_TO_GM_PER_CHANNEL_RELU(LayoutDst, DTypeSrc, DTypeDst)

#define REGISTER_L0C_TO_UB_PER_CHANNEL_NORELU(LayoutDst, DTypeSrc, DTypeDst)                                         \
    [aicore] __attribute__((always_inline)) void                                                                     \
        _mlir_ciface_copy_l0c_to_ub_##LayoutDst##_nosplit_perchannel_##DTypeSrc##_##DTypeDst(                        \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__ubuf__ DTypeDst, 1>* dst, DESC_ABI_PARAMS(src),            \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, uint8_t subBlockId, memref_t<__cbuf__ uint64_t, 1>* quant,       \
            DESC_ABI_PARAMS(quant))                                                                                  \
    {                                                                                                                \
        copyL0CToUBQuantPerChannel<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, false>(       \
            src, dst, quant, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), TENSOR_DESC_12(quant), unitFlag, subBlockId); \
    }

#define REGISTER_L0C_TO_UB_PER_CHANNEL_RELU(LayoutDst, DTypeSrc, DTypeDst)                                           \
    [aicore] __attribute__((always_inline)) void                                                                     \
        _mlir_ciface_copy_l0c_to_ub_##LayoutDst##_nosplit_perchannel_relu_##DTypeSrc##_##DTypeDst(                   \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__ubuf__ DTypeDst, 1>* dst, DESC_ABI_PARAMS(src),            \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, uint8_t subBlockId, memref_t<__cbuf__ uint64_t, 1>* quant,       \
            DESC_ABI_PARAMS(quant))                                                                                  \
    {                                                                                                                \
        copyL0CToUBQuantPerChannel<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, true>(        \
            src, dst, quant, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), TENSOR_DESC_12(quant), unitFlag, subBlockId); \
    }

#define REGISTER_L0C_TO_UB_PER_CHANNEL(LayoutDst, DTypeSrc, DTypeDst)    \
    REGISTER_L0C_TO_UB_PER_CHANNEL_NORELU(LayoutDst, DTypeSrc, DTypeDst) \
    REGISTER_L0C_TO_UB_PER_CHANNEL_RELU(LayoutDst, DTypeSrc, DTypeDst)
// PER_CHANNEL quant: the L1 uint64 scale tile rides along as the fixpipe dequant
// tensor instead of a scalar. The symbol carries both element types, as the
// per-tensor and no-quant routes do, so f32- and i32-accumulator registrations
// of the same destination stay distinct. L0C -> GM per-channel covers both
// RowMajor and ColumnMajor destinations.
REGISTER_L0C_TO_GM_PER_CHANNEL(RowMajor, float, float)
REGISTER_L0C_TO_GM_PER_CHANNEL(RowMajor, float, half)
REGISTER_L0C_TO_GM_PER_CHANNEL(RowMajor, float, bf16)
REGISTER_L0C_TO_GM_PER_CHANNEL(RowMajor, float, int8_t)
REGISTER_L0C_TO_GM_PER_CHANNEL(RowMajor, float, uint8_t)
REGISTER_L0C_TO_GM_PER_CHANNEL(RowMajor, int32_t, half)
REGISTER_L0C_TO_GM_PER_CHANNEL(RowMajor, int32_t, bf16)
REGISTER_L0C_TO_GM_PER_CHANNEL(RowMajor, int32_t, int8_t)
REGISTER_L0C_TO_GM_PER_CHANNEL(RowMajor, int32_t, uint8_t)
REGISTER_L0C_TO_GM_PER_CHANNEL(ColumnMajor, float, float)
REGISTER_L0C_TO_GM_PER_CHANNEL(ColumnMajor, float, half)
REGISTER_L0C_TO_GM_PER_CHANNEL(ColumnMajor, float, bf16)
REGISTER_L0C_TO_GM_PER_CHANNEL(ColumnMajor, float, int8_t)
REGISTER_L0C_TO_GM_PER_CHANNEL(ColumnMajor, float, uint8_t)
REGISTER_L0C_TO_GM_PER_CHANNEL(ColumnMajor, int32_t, half)
REGISTER_L0C_TO_GM_PER_CHANNEL(ColumnMajor, int32_t, bf16)
REGISTER_L0C_TO_GM_PER_CHANNEL(ColumnMajor, int32_t, int8_t)
REGISTER_L0C_TO_GM_PER_CHANNEL(ColumnMajor, int32_t, uint8_t)

// L0C -> UB per-channel quant is nosplit only.
REGISTER_L0C_TO_UB_PER_CHANNEL(RowMajor, float, float)
REGISTER_L0C_TO_UB_PER_CHANNEL(RowMajor, float, half)
REGISTER_L0C_TO_UB_PER_CHANNEL(RowMajor, float, bf16)
REGISTER_L0C_TO_UB_PER_CHANNEL(RowMajor, float, int8_t)
REGISTER_L0C_TO_UB_PER_CHANNEL(RowMajor, float, uint8_t)
REGISTER_L0C_TO_UB_PER_CHANNEL(RowMajor, int32_t, half)
REGISTER_L0C_TO_UB_PER_CHANNEL(RowMajor, int32_t, bf16)
REGISTER_L0C_TO_UB_PER_CHANNEL(RowMajor, int32_t, int8_t)
REGISTER_L0C_TO_UB_PER_CHANNEL(RowMajor, int32_t, uint8_t)
REGISTER_L0C_TO_UB_PER_CHANNEL(ColumnMajor, float, float)
REGISTER_L0C_TO_UB_PER_CHANNEL(ColumnMajor, float, half)
REGISTER_L0C_TO_UB_PER_CHANNEL(ColumnMajor, float, bf16)
REGISTER_L0C_TO_UB_PER_CHANNEL(ColumnMajor, float, int8_t)
REGISTER_L0C_TO_UB_PER_CHANNEL(ColumnMajor, float, uint8_t)
REGISTER_L0C_TO_UB_PER_CHANNEL(ColumnMajor, int32_t, half)
REGISTER_L0C_TO_UB_PER_CHANNEL(ColumnMajor, int32_t, bf16)
REGISTER_L0C_TO_UB_PER_CHANNEL(ColumnMajor, int32_t, int8_t)
REGISTER_L0C_TO_UB_PER_CHANNEL(ColumnMajor, int32_t, uint8_t)
#endif
}
