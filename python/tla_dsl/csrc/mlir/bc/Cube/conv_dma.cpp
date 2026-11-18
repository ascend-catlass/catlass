#include "../dma_common.h"

// Convolution stubs: the NC1HWC0 feature-map load and the img2col expansion it
// feeds. They live in their own translation unit so dma.cpp stays the GEMM copy
// table; bc_compile.py globs every Cube/*.cpp into the same aic bitcode, so the
// split adds no link step.

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
    memref_t<__gm__ T, 2>* src, memref_t<__cbuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc,
    int64_t gmCachePolicy = kCachePolicyNormal)
{
    auto srcTensor = makeGMTensor<LayoutTag::zNUnAlign, T>(src, srcDesc, gmCachePolicy);
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

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

#define REGISTER_GM_TO_L1_NC1HWC0(DType)                                                                              \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_zNUnAlign_to_l1_zNUnAlign_##DType(              \
        memref_t<__gm__ DType, 2>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), \
        int64_t gmCachePolicy)                                                                                        \
    {                                                                                                                 \
        copyGMToL1Nc1hwc0<Catlass::Arch::Ascend950, DType>(                                                           \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), gmCachePolicy);                                       \
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

#endif
} // extern "C"
