#include "../dma_common.h"

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#include "catlass/gemm/tile/copy_l0c_to_l1.hpp"
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyGMToL1(
    memref_t<__gm__ T, 2>* src, memref_t<__cbuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    // Every route here is served by one TileCopyTla specialization: the tag
    // selects both the layout constructor (makeGMTensor / makeL1Tensor) and the
    // copy. VectorLayout is the flat case -- both endpoints come out rank-1
    // (VectorLayout), which is exactly what the VectorLayout specialization in
    // copy_gm_to_l1.hpp matches.
    auto srcTensor = makeGMTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL1Tensor<DstLayout, T>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

template <class ArchTag, LayoutTag SrcLayout, typename T>
CATLASS_DEVICE void copyL1ToL0A(
    memref_t<__cbuf__ T, 1>* src, memref_t<__ca__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeL1Tensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL0ATensor<LayoutTag::zN, T>(dst, dstDesc);
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

// The pad fill value travels as an i32 bit pattern, because the op carries it as
// an attribute and MLIR has no element-typed attribute that spans f16/bf16/f32.
// Reinterpret, do not convert: 0 is 0 in every one of these formats, and a
// caller asking for -inf means the bit pattern of -inf.
template <typename T>
CATLASS_DEVICE T bitsToPadValue(uint32_t bits)
{
    if constexpr (sizeof(T) == 2) {
        uint16_t narrow = static_cast<uint16_t>(bits);
        T value;
        __builtin_memcpy(&value, &narrow, sizeof(T));
        return value;
    } else {
        T value;
        __builtin_memcpy(&value, &bits, sizeof(T));
        return value;
    }
}

// NC1HWC0 feature map, GM -> L1. Both sides carry the same tree, so this moves
// one contiguous run per C1 slice rather than re-laying anything out: the C1
// stride differs between the two (the source's is the whole feature map's h*w,
// the destination's is the tile's), which is what makes a windowed load along
// the flattened h*w axis work.
//
// A 2-D (hiBlock x wiBlock) window needs a second loop over hi, and the pitch to
// go with it; that is the next step, not this one. Here the tile is a run of
// whole feature-map rows.
template <class ArchTag, typename T>
CATLASS_DEVICE void copyGMToL1Nc1hwc0(
    memref_t<__gm__ T, 2>* src, memref_t<__cbuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeGMTensor<LayoutTag::zNUnAlign, T>(src, srcDesc);
    auto dstTensor = makeL1Tensor<LayoutTag::zNUnAlign, T>(dst, dstDesc);

    // Leaf 0 of the zNUnAlign tree is the (unblocked) row count, leaf 3 the C1
    // count; the C1 pitch is leaf 3's stride on each side.
    uint32_t rows = srcDesc.shape0 * srcDesc.shape1;
    uint32_t cin1 = srcDesc.shape3;
    uint32_t srcC1Pitch = static_cast<uint32_t>(srcDesc.stride3);
    uint32_t dstC1Pitch = static_cast<uint32_t>(dstDesc.stride3);

    auto srcOffset = srcTensor.layout()(srcTensor.coord());
    auto dstOffset = dstTensor.layout()(dstTensor.coord());

    // DataCopy counts GM->L1 in 32-byte blocks, and one C0 is exactly one block.
    for (uint32_t cin1Idx = 0; cin1Idx < cin1; ++cin1Idx) {
        AscendC::DataCopy(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset],
            AscendC::DataCopyParams(
                1,                           // blockCount
                static_cast<uint16_t>(rows), // blockLen, in C0 blocks
                0,                           // srcStride
                0                            // dstStride
                ));
        srcOffset += srcC1Pitch;
        dstOffset += dstC1Pitch;
    }
}

// tla.img2col: L1 NC1HWC0 feature map -> L0A, expanded by the img2col unit.
//
// This mirrors Catlass::Conv::Tile::CopyL1ToL0ATla, but takes the feature-map
// extent and the convolution geometry as arguments instead of reading them off a
// 5-D tensor: the DSL's L1 tile is the 2x2 zNUnAlign tree, which carries h*w as
// one flattened axis and so cannot state hi and wi separately.
//
// The destination tile supplies both extents and both start points, so tiling
// the implicit GEMM's M and K is a matter of pointing this at a different window
// of the same L1 tile.
template <class ArchTag, typename T>
CATLASS_DEVICE void img2colL1ToL0A(
    memref_t<__cbuf__ T, 1>* src, memref_t<__ca__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc,
    uint32_t fmapH, uint32_t fmapW, uint32_t filterH, uint32_t filterW, uint32_t strideH, uint32_t strideW,
    uint32_t dilationH, uint32_t dilationW, uint32_t padLeft, uint32_t padRight, uint32_t padTop, uint32_t padBottom,
    uint32_t padValueBits)
{
    auto srcTensor = makeL1Tensor<LayoutTag::zNUnAlign, T>(src, srcDesc);
    auto dstTensor = makeL0ATensor<LayoutTag::zN, T>(dst, dstDesc);

    // M is fractal-blocked on L0A (leaf 0 x leaf 1), K is C0-blocked (leaf 2 x leaf 3).
    uint32_t mExtension = dstDesc.shape0 * dstDesc.shape1;
    uint32_t kExtension = dstDesc.shape2 * dstDesc.shape3;
    // The feature map defines the channel count independently of the K window.
    // Deriving it from kExtension would change the feature-map interpretation
    // whenever the destination covers only part of the implicit-GEMM K axis.
    uint32_t channelSize = srcDesc.shape2 * srcDesc.shape3;

    AscendC::LoadData3DParamsV2<T> loadDataParams;
    // padList is left, right, top, bottom -- the order Catlass's conv block uses.
    loadDataParams.padList[0] = static_cast<uint8_t>(padLeft);
    loadDataParams.padList[1] = static_cast<uint8_t>(padRight);
    loadDataParams.padList[2] = static_cast<uint8_t>(padTop);
    loadDataParams.padList[3] = static_cast<uint8_t>(padBottom);
    loadDataParams.l1H = static_cast<uint16_t>(fmapH);
    loadDataParams.l1W = static_cast<uint16_t>(fmapW);
    loadDataParams.channelSize = static_cast<uint16_t>(channelSize);
    loadDataParams.kExtension = static_cast<uint16_t>(kExtension);
    loadDataParams.mExtension = static_cast<uint16_t>(mExtension);
    // The tile coordinate is the window into the img2col matrix, which is exactly
    // what the hardware's start points mean.
    loadDataParams.kStartPt = static_cast<uint16_t>(dstDesc.coord1);
    loadDataParams.mStartPt = static_cast<uint16_t>(dstDesc.coord0);
    loadDataParams.strideW = static_cast<uint8_t>(strideW);
    loadDataParams.strideH = static_cast<uint8_t>(strideH);
    loadDataParams.filterW = static_cast<uint8_t>(filterW);
    loadDataParams.filterH = static_cast<uint8_t>(filterH);
    // The 9th bit of each filter extent lives in its own flag, so a filter wider
    // than 255 is expressible.
    loadDataParams.filterSizeW = static_cast<bool>(filterW >> 8);
    loadDataParams.filterSizeH = static_cast<bool>(filterH >> 8);
    loadDataParams.dilationFilterW = static_cast<uint8_t>(dilationW);
    loadDataParams.dilationFilterH = static_cast<uint8_t>(dilationH);
    loadDataParams.enTranspose = false;
    loadDataParams.enSmallK = false;
    loadDataParams.padValue = bitsToPadValue<T>(padValueBits);

    // A5 Load3Dv2 takes the destination stride in L3D_RPT; this path does not
    // transpose, so it is the number of 16-element fractals mExtension covers.
    uint16_t dstStride = CeilDiv<Catlass::C0_NUM_PER_FRACTAL>(mExtension);
    AscendC::SetLoadDataRepeatWithStride({0, 1, 0, dstStride});

    auto srcOffset = srcTensor.layout()(srcTensor.coord());
    auto dstOffset = dstTensor.layout()(dstTensor.coord());
    AscendC::LoadDataWithStride(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], loadDataParams);
}

// L0C holds fp32 MMAD accumulator; ElementDst may be f32/f16/bf16 (Ascend950 fixpipe).
template <class ArchTag, LayoutTag DstLayout, typename ElementSrc, typename ElementDst, bool ReluEnable = false>
CATLASS_DEVICE void copyL0CToGM(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__gm__ ElementDst, 2>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, uint8_t unitFlag)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0Clayout, ElementSrc>(src, srcDesc);
    auto dstTensor = makeGMTensor<DstLayout, ElementDst>(dst, dstDesc);
    Catlass::Gemm::Tile::CopyL0CToGmTla<
        ArchTag, decltype(srcTensor), decltype(dstTensor), Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT,
        ReluEnable>{}(dstTensor, srcTensor, unitFlag);
}

// L0C holds fp32 MMAD accumulator; ElementDst may be f32/f16/bf16 (Ascend950 fixpipe).
template <class ArchTag, LayoutTag DstLayout, typename ElementSrc, typename ElementDst, bool ReluEnable = false>
CATLASS_DEVICE void copyL0CToGMQuantPerTensor(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__gm__ ElementDst, 2>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, uint8_t unitFlag, float quantScale, int32_t quantOffset)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0Clayout, ElementSrc>(src, srcDesc);
    auto dstTensor = makeGMTensor<DstLayout, ElementDst>(dst, dstDesc);
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
    const TensorDesc& srcDesc, const TensorDesc& dstDesc, const TensorDesc& scaleDesc, uint8_t unitFlag)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0Clayout, ElementSrc>(src, srcDesc);
    auto dstTensor = makeGMTensor<DstLayout, ElementDst>(dst, dstDesc);
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
// the copy templates take a third (e8m0, zZ, L1) tensor, and the L0 element type
// switches to the mx_* variant, which is what makes the later mad take the
// mad_mx path.
// The e8m0 scale block reaches L1 as a flat byte copy: the host emits it already
// in device order (zZMxScale for A, nNMxScale for B), so no reordering is needed
// on the way in. Only the L1 -> L0 hop below needs the real MX layouts.
//
// The copy must honour the tile coordinate and the tile's own extent. Reading
// from the base of the buffer for originShape bytes is only correct for a
// single tile at coordinate (0, 0); a K-tiled kernel then decodes every chunk
// with chunk 0's exponents, which looks like a plausible result one power of
// two out rather than like garbage.
//
// Tiles must span full rows of the scale buffer (shape1 == stride0), which is
// how per-chunk zZ / nN blocks are stacked: each block is then contiguous, so
// one flat run of bytes is exact. A partial-width tile would need a strided
// copy, and the scale row pitch (K/32 bytes) is not generally a multiple of the
// 32-byte DMA block, so that case cannot be served here at all. tla.copy
// rejects it when the row pitch is statically known; under a layout-dynamic

template <class ArchTag, LayoutTag SrcLayout, typename TSrc, typename TL0>
CATLASS_DEVICE void copyL1ToL0AMxFp8(
    memref_t<__cbuf__ TSrc, 1>* src, memref_t<__ca__ TSrc, 1>* dst, memref_t<__cbuf__ uint8_t, 1>* scale,
    const TensorDesc& srcDesc, const TensorDesc& dstDesc, const TensorDesc& scaleDesc)
{
    auto srcTensor = makeL1Tensor<SrcLayout, TSrc>(src, srcDesc);
    auto scaleTensor = makeL1MxScaleTensor<LayoutTag::zZMxScale>(scale, scaleDesc);
    auto dstTensor = makeL0ATensorAs<LayoutTag::zN, TL0>(dst, dstDesc);
    // The specialization is picked by the *type* arguments, the operator by the
    // call arguments, and the two want different things of the L0 element.
    // Catlass's transposing (nZ in) specialization enables itself only for
    // int8_t / float8_e* / float4_*, which an mx_fp8_* L0 tile is not -- while
    // its MX operator() asserts the opposite, that the L0 tile *is* mx_fp8_*.
    // Selecting with a same-width stand-in typed as the L1 element satisfies the
    // first, and passing the real tile satisfies the second. The stand-in is
    // never read: only decltype of it is used.
    auto dstSelector = makeL0ATensorAs<LayoutTag::zN, TSrc>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstSelector)>{}(
        dstTensor, srcTensor, scaleTensor);
}

template <class ArchTag, LayoutTag SrcLayout, typename TSrc, typename TL0>
CATLASS_DEVICE void copyL1ToL0BMxFp8(
    memref_t<__cbuf__ TSrc, 1>* src, memref_t<__cb__ TSrc, 1>* dst, memref_t<__cbuf__ uint8_t, 1>* scale,
    const TensorDesc& srcDesc, const TensorDesc& dstDesc, const TensorDesc& scaleDesc)
{
    auto srcTensor = makeL1Tensor<SrcLayout, TSrc>(src, srcDesc);
    auto scaleTensor = makeL1MxScaleTensor<LayoutTag::nNMxScale>(scale, scaleDesc);
    auto dstTensor = makeL0BTensorAs<LayoutTag::nZ, TL0>(dst, dstDesc);
    // Same stand-in as on the A side: select the specialization with the L1
    // element type, call it with the real mx_fp8_* tile.
    auto dstSelector = makeL0BTensorAs<LayoutTag::nZ, TSrc>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstSelector)>{}(
        dstTensor, srcTensor, scaleTensor);
}
#endif

// Packed fp4 variants. The operand tiles are int8_t over the ABI and are
// reinterpreted as float4_*x2_t here, which is what gives the Catlass layout
// math its 4-bit element width and selects the fp4 copy specialisations.
template <class ArchTag, typename TFp4, LayoutTag SrcLayout, LayoutTag DstLayout>
CATLASS_DEVICE void copyGMToL1Fp4(
    memref_t<__gm__ int8_t, 2>* src, memref_t<__cbuf__ int8_t, 1>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc)
{
    auto srcTensor = makeGMFp4Tensor<TFp4, SrcLayout>(src, srcDesc);
    auto dstTensor = makeL1Fp4Tensor<TFp4, DstLayout>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

template <class ArchTag, typename TFp4, LayoutTag SrcLayout>
CATLASS_DEVICE void copyL1ToL0AMxFp4(
    memref_t<__cbuf__ int8_t, 1>* src, memref_t<__ca__ int8_t, 1>* dst, memref_t<__cbuf__ uint8_t, 1>* scale,
    const TensorDesc& srcDesc, const TensorDesc& dstDesc, const TensorDesc& scaleDesc)
{
    auto srcTensor = makeL1Fp4Tensor<TFp4, SrcLayout>(src, srcDesc);
    auto scaleTensor = makeL1MxScaleTensor<LayoutTag::zZMxScale>(scale, scaleDesc);
    auto dstTensor = makeL0ATensorAs<LayoutTag::zN, TFp4>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(
        dstTensor, srcTensor, scaleTensor);
}

template <class ArchTag, typename TFp4, LayoutTag SrcLayout>
CATLASS_DEVICE void copyL1ToL0BMxFp4(
    memref_t<__cbuf__ int8_t, 1>* src, memref_t<__cb__ int8_t, 1>* dst, memref_t<__cbuf__ uint8_t, 1>* scale,
    const TensorDesc& srcDesc, const TensorDesc& dstDesc, const TensorDesc& scaleDesc)
{
    auto srcTensor = makeL1Fp4Tensor<TFp4, SrcLayout>(src, srcDesc);
    auto scaleTensor = makeL1MxScaleTensor<LayoutTag::nNMxScale>(scale, scaleDesc);
    auto dstTensor = makeL0BTensorAs<LayoutTag::nZ, TFp4>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(
        dstTensor, srcTensor, scaleTensor);
}

// MX scale GM -> L1 with the fractal reorder done by the copy.
//
// Catlass already implements this for all four (side, orientation) pairs -- see
// the MxScale specializations in gemm/tile/ascend950/copy_gm_to_l1.hpp -- so this
// only has to build tensors whose layouts its predicates recognise and let
// TileCopyTla dispatch. Doing the DN2NZ by hand instead means reimplementing
// details it already gets right, such as ceil-dividing the halved extent rather
// than truncating it.
template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout>
CATLASS_DEVICE void copyGMToL1MxScale(
    memref_t<__gm__ uint8_t, 2>* src, memref_t<__cbuf__ uint8_t, 1>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc)
{
    auto srcTensor = makeGMMxScaleTensor<SrcLayout>(src, srcDesc);
    auto dstTensor = makeL1MxScaleTensor<DstLayout>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

#define REGISTER_GM_TO_L1(LayoutSrc, LayoutDst, DType)                                                                \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##LayoutSrc##_to_l1_##LayoutDst##_##DType(      \
        memref_t<__gm__ DType, 2>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyGMToL1<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

#define REGISTER_L1_TO_L0A(LayoutSrc, DType)                                                                          \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l1_##LayoutSrc##_to_l0a_zN_##DType(                \
        memref_t<__cbuf__ DType, 1>* src, memref_t<__ca__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyL1ToL0A<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, DType>(                                           \
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
REGISTER_GM_TO_L1(RowMajor, zN, uint16_t)
REGISTER_GM_TO_L1(RowMajor, zN, uint32_t)
REGISTER_GM_TO_L1(RowMajor, zN, fp8_e4m3fn_t)
REGISTER_GM_TO_L1(RowMajor, zN, fp8_e5m2_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, float)
REGISTER_GM_TO_L1(ColumnMajor, nZ, half)
REGISTER_GM_TO_L1(ColumnMajor, nZ, bf16)
REGISTER_GM_TO_L1(ColumnMajor, nZ, int8_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, uint16_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, uint32_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, fp8_e4m3fn_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, fp8_e5m2_t)
REGISTER_GM_TO_L1(zN, zN, float)
REGISTER_GM_TO_L1(zN, zN, half)
REGISTER_GM_TO_L1(zN, zN, bf16)
REGISTER_GM_TO_L1(zN, zN, int8_t)
REGISTER_GM_TO_L1(zN, zN, uint16_t)
REGISTER_GM_TO_L1(zN, zN, uint32_t)
REGISTER_GM_TO_L1(zN, zN, fp8_e4m3fn_t)
REGISTER_GM_TO_L1(zN, zN, fp8_e5m2_t)
REGISTER_GM_TO_L1(zN, zN, float4_e2m1x2_t)
REGISTER_GM_TO_L1(zN, zN, float4_e1m2x2_t)
REGISTER_GM_TO_L1(nZ, nZ, float)
REGISTER_GM_TO_L1(nZ, nZ, half)
REGISTER_GM_TO_L1(nZ, nZ, bf16)
REGISTER_GM_TO_L1(nZ, nZ, int8_t)
REGISTER_GM_TO_L1(nZ, nZ, uint16_t)
REGISTER_GM_TO_L1(nZ, nZ, uint32_t)
REGISTER_GM_TO_L1(nZ, nZ, fp8_e4m3fn_t)
REGISTER_GM_TO_L1(nZ, nZ, fp8_e5m2_t)
REGISTER_GM_TO_L1(nZ, nZ, float4_e2m1x2_t)
REGISTER_GM_TO_L1(nZ, nZ, float4_e1m2x2_t)

// VectorLayout -> VectorLayout is the flat contiguous run: GM and L1 both come out rank-1
// (VectorLayout), so the VectorLayout TileCopyTla specialization in
// copy_gm_to_l1.hpp serves it. RowMajor has no variant here: a flat tile is
// spelled VectorLayout, and RowMajor stays two-dimensional.
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, float)
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, int32_t)
// The per-channel fixpipe dequant scale tile is the same flat run, one u64 word
// (a packed fp32 scale) per output channel.
REGISTER_GM_TO_L1(VectorLayout, VectorLayout, uint64_t)

REGISTER_L1_TO_L0A(zN, float)
REGISTER_L1_TO_L0A(zN, half)
REGISTER_L1_TO_L0A(zN, bf16)
REGISTER_L1_TO_L0A(zN, int8_t)
// uint16_t is not in AscendC LoadData 2dv2's supported set on dav-3510
// (uint8/uint32 yes; uint16 no). Keep GM→L1 for u16; L1→L0 only for u32.
REGISTER_L1_TO_L0A(zN, uint32_t)
REGISTER_L1_TO_L0A(zN, fp8_e4m3fn_t)
REGISTER_L1_TO_L0A(zN, fp8_e5m2_t)
REGISTER_L1_TO_L0A(nZ, float)
REGISTER_L1_TO_L0A(nZ, half)
REGISTER_L1_TO_L0A(nZ, bf16)
REGISTER_L1_TO_L0A(nZ, int8_t)
REGISTER_L1_TO_L0A(nZ, uint32_t)
REGISTER_L1_TO_L0A(nZ, fp8_e4m3fn_t)
REGISTER_L1_TO_L0A(nZ, fp8_e5m2_t)

REGISTER_L1_TO_L0B(zN, float)
REGISTER_L1_TO_L0B(zN, half)
REGISTER_L1_TO_L0B(zN, bf16)
REGISTER_L1_TO_L0B(zN, int8_t)
REGISTER_L1_TO_L0B(zN, uint32_t)
REGISTER_L1_TO_L0B(zN, fp8_e4m3fn_t)
REGISTER_L1_TO_L0B(zN, fp8_e5m2_t)
REGISTER_L1_TO_L0B(nZ, float)
REGISTER_L1_TO_L0B(nZ, half)
REGISTER_L1_TO_L0B(nZ, bf16)
REGISTER_L1_TO_L0B(nZ, int8_t)
REGISTER_L1_TO_L0B(nZ, uint32_t)
REGISTER_L1_TO_L0B(nZ, fp8_e4m3fn_t)
REGISTER_L1_TO_L0B(nZ, fp8_e5m2_t)

REGISTER_L1_TO_BIAS(float)
REGISTER_L1_TO_BIAS(int32_t)

#define REGISTER_GM_TO_L1_NC1HWC0(DType)                                                                              \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_zNUnAlign_to_l1_zNUnAlign_##DType(              \
        memref_t<__gm__ DType, 2>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyGMToL1Nc1hwc0<Catlass::Arch::Ascend950, DType>(src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));       \
    }

#define REGISTER_IMG2COL_L1_TO_L0A(DType)                                                                             \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_img2col_l1_zNUnAlign_to_l0a_zN_##DType(                 \
        memref_t<__cbuf__ DType, 1>* src, memref_t<__ca__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), \
        uint32_t fmapH, uint32_t fmapW, uint32_t filterH, uint32_t filterW, uint32_t strideH, uint32_t strideW,       \
        uint32_t dilationH, uint32_t dilationW, uint32_t padLeft, uint32_t padRight, uint32_t padTop,                 \
        uint32_t padBottom, uint32_t padValueBits)                                                                    \
    {                                                                                                                 \
        img2colL1ToL0A<Catlass::Arch::Ascend950, DType>(                                                              \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), fmapH, fmapW, filterH, filterW, strideH, strideW,     \
            dilationH, dilationW, padLeft, padRight, padTop, padBottom, padValueBits);                                \
    }

REGISTER_GM_TO_L1_NC1HWC0(float)
REGISTER_GM_TO_L1_NC1HWC0(half)
REGISTER_GM_TO_L1_NC1HWC0(bf16)

// img2col is instantiated for 2- and 4-byte elements only: that is what the A5
// path supports, and the op's verifier refuses anything else up front.
REGISTER_IMG2COL_L1_TO_L0A(float)
REGISTER_IMG2COL_L1_TO_L0A(half)
REGISTER_IMG2COL_L1_TO_L0A(bf16)

#define REGISTER_GM_TO_L1_FP4(NameSrc, NameDst, TFp4)                                                       \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##NameSrc##_to_l1_##NameDst##_##TFp4( \
        memref_t<__gm__ int8_t, 2>* src, memref_t<__cbuf__ int8_t, 1>* dst, DESC_ABI_PARAMS(src),           \
        DESC_ABI_PARAMS(dst))                                                                               \
    {                                                                                                       \
        copyGMToL1Fp4<Catlass::Arch::Ascend950, TFp4, LayoutTag::NameSrc, LayoutTag::NameDst>(              \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                            \
    }

#define REGISTER_GM_TO_L1_MX_SCALE(SrcLayout, DstLayout)                                                         \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##SrcLayout##_to_l1_##DstLayout##_uint8_t( \
        memref_t<__gm__ uint8_t, 2>* src, memref_t<__cbuf__ uint8_t, 1>* dst, DESC_ABI_PARAMS(src),              \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyGMToL1MxScale<Catlass::Arch::Ascend950, LayoutTag::SrcLayout, LayoutTag::DstLayout>(                 \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                 \
    }

// Both source layouts per side, as the non-MX REGISTER_L1_TO_L0A/B above take.
#define REGISTER_L1_TO_L0A_MX_FP8(LayoutSrc, L1Type, L0Type)                                                      \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_mx_l1_##LayoutSrc##_to_l0a_zN_##L1Type(        \
        memref_t<__cbuf__ L1Type, 1>* src, memref_t<__ca__ L1Type, 1>* dst, memref_t<__cbuf__ uint8_t, 1>* scale, \
        DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), DESC_ABI_PARAMS(scale))                                       \
    {                                                                                                             \
        copyL1ToL0AMxFp8<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, L1Type, L0Type>(                         \
            src, dst, scale, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), TENSOR_DESC_12(scale));                    \
    }

#define REGISTER_L1_TO_L0B_MX_FP8(LayoutSrc, L1Type, L0Type)                                                      \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_mx_l1_##LayoutSrc##_to_l0b_nZ_##L1Type(        \
        memref_t<__cbuf__ L1Type, 1>* src, memref_t<__cb__ L1Type, 1>* dst, memref_t<__cbuf__ uint8_t, 1>* scale, \
        DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), DESC_ABI_PARAMS(scale))                                       \
    {                                                                                                             \
        copyL1ToL0BMxFp8<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, L1Type, L0Type>(                         \
            src, dst, scale, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), TENSOR_DESC_12(scale));                    \
    }

// Both source layouts, unlike the fp8 MX macros above. Catlass's B8/B4 transpose
// specialization admits ElementDst in int8_t / float8_e* / float4_*x2_t: an fp4
// L0 tile is float4_*x2_t and qualifies, while an MX fp8 L0 tile is mx_fp8_* and
// does not, which is why only fp4 gets the transposing pairing.
#define REGISTER_L1_TO_L0A_MX_FP4(LayoutSrc, TFp4)                                                                \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_mx_l1_##LayoutSrc##_to_l0a_zN_##TFp4(          \
        memref_t<__cbuf__ int8_t, 1>* src, memref_t<__ca__ int8_t, 1>* dst, memref_t<__cbuf__ uint8_t, 1>* scale, \
        DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), DESC_ABI_PARAMS(scale))                                       \
    {                                                                                                             \
        copyL1ToL0AMxFp4<Catlass::Arch::Ascend950, TFp4, LayoutTag::LayoutSrc>(                                   \
            src, dst, scale, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), TENSOR_DESC_12(scale));                    \
    }

#define REGISTER_L1_TO_L0B_MX_FP4(LayoutSrc, TFp4)                                                                \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_mx_l1_##LayoutSrc##_to_l0b_nZ_##TFp4(          \
        memref_t<__cbuf__ int8_t, 1>* src, memref_t<__cb__ int8_t, 1>* dst, memref_t<__cbuf__ uint8_t, 1>* scale, \
        DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), DESC_ABI_PARAMS(scale))                                       \
    {                                                                                                             \
        copyL1ToL0BMxFp4<Catlass::Arch::Ascend950, TFp4, LayoutTag::LayoutSrc>(                                   \
            src, dst, scale, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), TENSOR_DESC_12(scale));                    \
    }

// MxMatmul related copy
REGISTER_GM_TO_L1_FP4(RowMajor, zN, float4_e2m1x2_t)
REGISTER_GM_TO_L1_FP4(ColumnMajor, nZ, float4_e2m1x2_t)
REGISTER_GM_TO_L1_FP4(RowMajor, zN, float4_e1m2x2_t)
REGISTER_GM_TO_L1_FP4(ColumnMajor, nZ, float4_e1m2x2_t)

REGISTER_GM_TO_L1_MX_SCALE(RowMajorMxScaleA, zZMxScale)
REGISTER_GM_TO_L1_MX_SCALE(ColMajorMxScaleA, zZMxScale)
REGISTER_GM_TO_L1_MX_SCALE(RowMajorMxScaleB, nNMxScale)
REGISTER_GM_TO_L1_MX_SCALE(ColMajorMxScaleB, nNMxScale)

REGISTER_L1_TO_L0A_MX_FP8(zN, fp8_e4m3fn_t, mx_fp8_e4m3_t)
REGISTER_L1_TO_L0A_MX_FP8(zN, fp8_e5m2_t, mx_fp8_e5m2_t)
REGISTER_L1_TO_L0A_MX_FP8(nZ, fp8_e4m3fn_t, mx_fp8_e4m3_t)
REGISTER_L1_TO_L0A_MX_FP8(nZ, fp8_e5m2_t, mx_fp8_e5m2_t)

REGISTER_L1_TO_L0B_MX_FP8(zN, fp8_e4m3fn_t, mx_fp8_e4m3_t)
REGISTER_L1_TO_L0B_MX_FP8(zN, fp8_e5m2_t, mx_fp8_e5m2_t)
REGISTER_L1_TO_L0B_MX_FP8(nZ, fp8_e4m3fn_t, mx_fp8_e4m3_t)
REGISTER_L1_TO_L0B_MX_FP8(nZ, fp8_e5m2_t, mx_fp8_e5m2_t)

REGISTER_L1_TO_L0A_MX_FP4(zN, float4_e2m1x2_t)
REGISTER_L1_TO_L0A_MX_FP4(zN, float4_e1m2x2_t)
REGISTER_L1_TO_L0A_MX_FP4(nZ, float4_e2m1x2_t)
REGISTER_L1_TO_L0A_MX_FP4(nZ, float4_e1m2x2_t)

REGISTER_L1_TO_L0B_MX_FP4(zN, float4_e2m1x2_t)
REGISTER_L1_TO_L0B_MX_FP4(zN, float4_e1m2x2_t)
REGISTER_L1_TO_L0B_MX_FP4(nZ, float4_e2m1x2_t)
REGISTER_L1_TO_L0B_MX_FP4(nZ, float4_e1m2x2_t)

#define REGISTER_L0C_TO_GM_NORELU(LayoutDst, DTypeSrc, DTypeDst)                                                    \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l0c_to_gm_##LayoutDst##_##DTypeSrc##_##DTypeDst( \
        memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__gm__ DTypeDst, 2>* dst, DESC_ABI_PARAMS(src),                 \
        DESC_ABI_PARAMS(dst), uint8_t unitFlag)                                                                     \
    {                                                                                                               \
        copyL0CToGM<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, false>(                     \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag);                                          \
    }

#define REGISTER_L0C_TO_GM_RELU(LayoutDst, DTypeSrc, DTypeDst)                                                      \
    [aicore]                                                                                                        \
        __attribute__((always_inline)) void _mlir_ciface_copy_l0c_to_gm_##LayoutDst##_relu_##DTypeSrc##_##DTypeDst( \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__gm__ DTypeDst, 2>* dst, DESC_ABI_PARAMS(src),             \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag)                                                                 \
    {                                                                                                               \
        copyL0CToGM<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, true>(                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag);                                          \
    }

#define REGISTER_L0C_TO_GM(LayoutDst, DTypeSrc, DTypeDst)    \
    REGISTER_L0C_TO_GM_NORELU(LayoutDst, DTypeSrc, DTypeDst) \
    REGISTER_L0C_TO_GM_RELU(LayoutDst, DTypeSrc, DTypeDst)

#define REGISTER_L0C_TO_GM_PER_TENSOR_NORELU(LayoutDst, DTypeSrc, DTypeDst)                                   \
    [aicore] __attribute__((always_inline)) void                                                              \
        _mlir_ciface_copy_l0c_to_gm_##LayoutDst##_pertensor_##DTypeSrc##_##DTypeDst(                          \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__gm__ DTypeDst, 2>* dst, DESC_ABI_PARAMS(src),       \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, float quantScale, int32_t quantOffset)                    \
    {                                                                                                         \
        copyL0CToGMQuantPerTensor<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, false>( \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag, quantScale, quantOffset);           \
    }

#define REGISTER_L0C_TO_GM_PER_TENSOR_RELU(LayoutDst, DTypeSrc, DTypeDst)                                    \
    [aicore] __attribute__((always_inline)) void                                                             \
        _mlir_ciface_copy_l0c_to_gm_##LayoutDst##_pertensor_relu_##DTypeSrc##_##DTypeDst(                    \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__gm__ DTypeDst, 2>* dst, DESC_ABI_PARAMS(src),      \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, float quantScale, int32_t quantOffset)                   \
    {                                                                                                        \
        copyL0CToGMQuantPerTensor<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, true>( \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag, quantScale, quantOffset);          \
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
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, memref_t<__cbuf__ uint64_t, 1>* quant, DESC_ABI_PARAMS(quant)) \
    {                                                                                                              \
        copyL0CToGMQuantPerChannel<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, false>(     \
            src, dst, quant, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), TENSOR_DESC_12(quant), unitFlag);           \
    }
#define REGISTER_L0C_TO_GM_PER_CHANNEL_RELU(LayoutDst, DTypeSrc, DTypeDst)                                         \
    [aicore] __attribute__((always_inline)) void                                                                   \
        _mlir_ciface_copy_l0c_to_gm_##LayoutDst##_perchannel_relu_##DTypeSrc##_##DTypeDst(                         \
            memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__gm__ DTypeDst, 2>* dst, DESC_ABI_PARAMS(src),            \
            DESC_ABI_PARAMS(dst), uint8_t unitFlag, memref_t<__cbuf__ uint64_t, 1>* quant, DESC_ABI_PARAMS(quant)) \
    {                                                                                                              \
        copyL0CToGMQuantPerChannel<Catlass::Arch::Ascend950, LayoutTag::LayoutDst, DTypeSrc, DTypeDst, true>(      \
            src, dst, quant, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), TENSOR_DESC_12(quant), unitFlag);           \
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
