#include "../dma_common.h"

// Convolution stubs: the NC1HWC0 feature-map load and the img2col expansion it
// feeds. They live in their own translation unit so dma.cpp stays the GEMM copy
// table; bc_compile.py globs every Cube/*.cpp into the same aic bitcode, so the
// split adds no link step.

// Configure FMatrix via Ascend C and write the padding register from the raw
// bits. The SDK's typed setter can convert FP8 and truncate 32-bit values.
constexpr AscendC::IsResetLoad3dConfig kImg2colLoadConfig{true, false};

template <typename T>
CATLASS_DEVICE int64_t nc1hwc0Offset(const TensorDesc& desc)
{
    constexpr uint32_t c0 = Catlass::BytesToElements<T>(Catlass::BYTE_PER_C0);
    // Spatial coordinates refer to the parent buffer and advance one C0 per
    // row. Only moving between channel slices uses the parent's C1 stride.
    return static_cast<int64_t>(desc.coord0) * c0 + (desc.coord1 / c0) * desc.stride3 + desc.coord1 % c0;
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
template <typename T>
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

    auto srcOffset = nc1hwc0Offset<T>(srcDesc);
    auto dstOffset = nc1hwc0Offset<T>(dstDesc);

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

// tla.img2col: L1 NC1HWC0 feature map -> L0A or L0B, expanded by the img2col unit.
//
// This mirrors Catlass::Conv::Tile::CopyL1ToL0ATla, but takes the feature-map
// extent and the convolution geometry as arguments instead of reading them off a
// 5-D tensor: the DSL's L1 tile is the 2x2 zNUnAlign tree, which carries h*w as
// one flattened axis and so cannot state hi and wi separately.
//
// The destination tile supplies both extents and both start points, so tiling
// the implicit GEMM's M and K is a matter of pointing this at a different window
// of the same L1 tile.
template <LayoutTag DstLayout, typename T, typename DstTensor>
CATLASS_DEVICE void img2colL1ToL0(
    memref_t<__cbuf__ T, 1>* src, const DstTensor& dstTensor, const TensorDesc& srcDesc, const TensorDesc& dstDesc,
    uint32_t fmapH, uint32_t fmapW, uint32_t filterH, uint32_t filterW, uint32_t strideH, uint32_t strideW,
    uint32_t dilationH, uint32_t dilationW, uint32_t padLeft, uint32_t padRight, uint32_t padTop, uint32_t padBottom,
    uint32_t padValueBits, uint32_t repeatStride, uint32_t repeatTime, uint32_t repeatMode)
{
    auto srcTensor = makeL1Tensor<LayoutTag::zNUnAlign, T>(src, srcDesc);
    constexpr bool toL0B = DstLayout == LayoutTag::nZ;

    // L0B transposes the physical layout while preserving the (M, K) window.
    uint32_t mExtension = dstDesc.shape0 * dstDesc.shape1;
    constexpr uint32_t elementsPerC0 = Catlass::BytesToElements<T>(Catlass::BYTE_PER_C0);
    uint32_t kExtension = CeilDiv<elementsPerC0>(dstDesc.originShape1) * elementsPerC0;
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
    if constexpr (sizeof(T) == 1) {
        // B8 uses the same byte in both halves of the low 16 bits.
        padValueBits = (padValueBits & 0xffu) * 0x0101u;
    } else if constexpr (sizeof(T) == 2) {
        padValueBits &= 0xffffu;
    }
    set_padding(padValueBits);

    // The view preserves the parent pitch along its outer fractal axis.
    constexpr uint32_t elementsPerFractal = Catlass::BytesToElements<T>(Catlass::BYTE_PER_FRACTAL);
    int64_t outerStride = toL0B ? dstDesc.stride1 : dstDesc.stride3;
    uint16_t dstStride = static_cast<uint16_t>(outerStride / elementsPerFractal);
    AscendC::SetLoadDataRepeatWithStride(
        {static_cast<uint16_t>(repeatStride), static_cast<uint8_t>(repeatTime), static_cast<uint8_t>(repeatMode),
         dstStride});

    auto srcOffset = nc1hwc0Offset<T>(srcDesc);
    auto dstOffset = dstTensor.layout()(dstTensor.coord());
    AscendC::LoadDataWithStride<T, kImg2colLoadConfig>(
        dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], loadDataParams);
}

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

#define REGISTER_GM_TO_L1_NC1HWC0(DType)                                                                              \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_zNUnAlign_to_l1_zNUnAlign_##DType(              \
        memref_t<__gm__ DType, 2>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), \
        int64_t gmCachePolicy)                                                                                        \
    {                                                                                                                 \
        copyGMToL1Nc1hwc0<DType>(src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), gmCachePolicy);                  \
    }

#define REGISTER_IMG2COL(DType, DstSpace, DstLayout, AddrQual, MakeDstTensor)                                          \
    [aicore]                                                                                                           \
        __attribute__((always_inline)) void _mlir_ciface_img2col_l1_zNUnAlign_to_##DstSpace##_##DstLayout##_##DType(   \
            memref_t<__cbuf__ DType, 1>* src, memref_t<AddrQual DType, 1>* dst, DESC_ABI_PARAMS(src),                  \
            DESC_ABI_PARAMS(dst), uint32_t fmapH, uint32_t fmapW, uint32_t filterH, uint32_t filterW,                  \
            uint32_t strideH, uint32_t strideW, uint32_t dilationH, uint32_t dilationW, uint32_t padLeft,              \
            uint32_t padRight, uint32_t padTop, uint32_t padBottom, uint32_t padValueBits, uint32_t repeatStride,      \
            uint32_t repeatTime, uint32_t repeatMode)                                                                  \
    {                                                                                                                  \
        auto dstDesc = TENSOR_DESC_12(dst);                                                                            \
        auto dstTensor = MakeDstTensor<LayoutTag::DstLayout, DType>(dst, dstDesc);                                     \
        img2colL1ToL0<LayoutTag::DstLayout, DType>(                                                                    \
            src, dstTensor, TENSOR_DESC_12(src), dstDesc, fmapH, fmapW, filterH, filterW, strideH, strideW, dilationH, \
            dilationW, padLeft, padRight, padTop, padBottom, padValueBits, repeatStride, repeatTime, repeatMode);      \
    }

REGISTER_GM_TO_L1_NC1HWC0(float)
REGISTER_GM_TO_L1_NC1HWC0(half)
REGISTER_GM_TO_L1_NC1HWC0(bf16)
REGISTER_GM_TO_L1_NC1HWC0(int32_t)
REGISTER_GM_TO_L1_NC1HWC0(uint32_t)
REGISTER_GM_TO_L1_NC1HWC0(int8_t)
REGISTER_GM_TO_L1_NC1HWC0(uint8_t)
REGISTER_GM_TO_L1_NC1HWC0(hifloat8_t)
REGISTER_GM_TO_L1_NC1HWC0(fp8_e4m3fn_t)
REGISTER_GM_TO_L1_NC1HWC0(fp8_e5m2_t)

REGISTER_IMG2COL(float, l0a, zN, __ca__, makeL0ATensor)
REGISTER_IMG2COL(half, l0a, zN, __ca__, makeL0ATensor)
REGISTER_IMG2COL(bf16, l0a, zN, __ca__, makeL0ATensor)
REGISTER_IMG2COL(int32_t, l0a, zN, __ca__, makeL0ATensor)
REGISTER_IMG2COL(uint32_t, l0a, zN, __ca__, makeL0ATensor)
REGISTER_IMG2COL(int8_t, l0a, zN, __ca__, makeL0ATensor)
REGISTER_IMG2COL(uint8_t, l0a, zN, __ca__, makeL0ATensor)
REGISTER_IMG2COL(hifloat8_t, l0a, zN, __ca__, makeL0ATensor)
REGISTER_IMG2COL(fp8_e4m3fn_t, l0a, zN, __ca__, makeL0ATensor)
REGISTER_IMG2COL(fp8_e5m2_t, l0a, zN, __ca__, makeL0ATensor)
REGISTER_IMG2COL(float, l0b, nZ, __cb__, makeL0BTensor)
REGISTER_IMG2COL(half, l0b, nZ, __cb__, makeL0BTensor)
REGISTER_IMG2COL(bf16, l0b, nZ, __cb__, makeL0BTensor)
REGISTER_IMG2COL(int32_t, l0b, nZ, __cb__, makeL0BTensor)
REGISTER_IMG2COL(uint32_t, l0b, nZ, __cb__, makeL0BTensor)

#endif
} // extern "C"
