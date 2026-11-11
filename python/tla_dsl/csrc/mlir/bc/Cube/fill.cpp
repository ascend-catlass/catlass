#include "../dma_common.h"

#include <type_traits>

// L1 fill: write a scalar bit pattern into a C0-column-granular region of an
// L1 zN/nZ tile. Backs the tla.fill op (Tensor.fill on the Python side).
// The region arrives as a flat rank-2 (coord, extent) pair on the tile's
// logical axes, taken from the tile's coord and origin shape. Two axes matter:
//   - the C0-packed axis (C0e-element units laid end to end): dim1 for zN,
//     dim0 for nZ. Fill writes 32-byte blocks, i.e. whole C0
//     columns, so both ends of this axis are aligned up to C0 boundaries and
//     the fill covers whole-C0-unit runs [alignUp(start), alignUp(end)).
//   - the 16-packed axis (16-element fractal rows): the other dim.
// With a C0-aligned start the tile is dense in 32-byte blocks along the
// fractal-row axis -- element (fractalRow, c0ColStart) sits at
// fractalRow*C0e + (c0ColStart/C0e)*unitStride and each fractal-row step
// advances exactly one block, fractal boundaries included -- so a single Fill
// covers the whole region: blockNum = fractal-row extent contiguous blocks per
// repeat, repeatTimes = aligned C0-unit count, dstGap = the block distance
// between consecutive C0 units.

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

namespace {

template <LayoutTag Tag, typename T>
CATLASS_DEVICE void fillL1Impl(
    uint32_t baseAddr, const TensorDesc& desc, int64_t crd0, int64_t crd1, int64_t ext0, int64_t ext1,
    int32_t valueBits)
{
    static_assert(Tag == LayoutTag::zN || Tag == LayoutTag::nZ, "L1 fill supports zN/nZ tiles only");
    constexpr uint32_t elemBits = Catlass::SizeOfBits<T>::value;
    static_assert(elemBits == 4 || elemBits == 8 || elemBits == 16 || elemBits == 32);
    constexpr uint32_t eleNumPerC0 = Catlass::BytesToBits(Catlass::BYTE_PER_C0) / elemBits;
    using FillT = std::conditional_t<(elemBits <= 16), uint16_t, uint32_t>;
    if constexpr (elemBits < 16) {
        if (valueBits != 0)
            return;
    }

    // Region axes after the tag dispatch: "fractal row" is the 16-element
    // fractal-row axis, "C0 column" the C0-packed axis (32-byte columns laid
    // end to end). zN packs C0 columns along dim1, nZ along dim0.
    int64_t fractalRowCrd, fractalRowExt, c0ColCrd, c0ColExt, unitStride;
    if constexpr (Tag == LayoutTag::zN) {
        fractalRowCrd = crd0;
        fractalRowExt = ext0;
        c0ColCrd = crd1;
        c0ColExt = ext1;
        unitStride = desc.stride3; // C0-unit leaf stride, in elements
    } else {
        fractalRowCrd = crd1;
        fractalRowExt = ext1;
        c0ColCrd = crd0;
        c0ColExt = ext0;
        unitStride = desc.stride1;
    }
    if (fractalRowExt <= 0 || c0ColExt <= 0) {
        return;
    }

    // Align both ends of the C0-column range up to C0 boundaries; the fill
    // then covers whole C0 columns [c0ColStart, c0ColEnd). A region that
    // starts and ends inside the same partial C0 column empties out here.
    const int64_t c0ColStart = CEIL_FACTOR(c0ColCrd, eleNumPerC0);
    const int64_t c0ColEnd = CEIL_FACTOR(c0ColCrd + c0ColExt, eleNumPerC0);
    const int64_t alignedC0ColExt = c0ColEnd - c0ColStart;
    const int64_t repeatTimes = alignedC0ColExt / eleNumPerC0;
    if (repeatTimes <= 0 || repeatTimes > 0x7FFF) {
        // Empty, or beyond the config's 15-bit repeat field.
        return;
    }
    const int64_t unitStrideBlocks = unitStride / eleNumPerC0;
    if (unitStrideBlocks < fractalRowExt) {
        // Region wider than a C0 unit: would overwrite the next unit's head.
        return;
    }

    const int64_t elemOffset = fractalRowCrd * eleNumPerC0 + (c0ColStart / eleNumPerC0) * unitStride;
    const int64_t byteOffset = elemOffset * elemBits / 8;
    AscendC::LocalTensor<FillT> dstTensor(
        AscendC::TPosition::A1, baseAddr + static_cast<uint32_t>(byteOffset),
        static_cast<uint32_t>(fractalRowExt * Catlass::BYTE_PER_C0 / sizeof(FillT)));
    AscendC::InitConstValueParams<FillT> params;
    params.repeatTimes = static_cast<uint16_t>(repeatTimes);
    params.blockNum = static_cast<uint16_t>(fractalRowExt);
    params.dstGap = static_cast<uint16_t>(unitStrideBlocks - fractalRowExt);
    params.initValue = static_cast<FillT>(static_cast<uint32_t>(valueBits));
    AscendC::Fill(dstTensor, params);
}

} // namespace

extern "C" {

#define REGISTER_FILL_L1(Layout, DType)                                                                               \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_fill_l1_##Layout##_##DType(                             \
        memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(dst), int64_t crd0, int64_t crd1, int64_t ext0,             \
        int64_t ext1, int32_t valueBits)                                                                              \
    {                                                                                                                 \
        fillL1Impl<LayoutTag::Layout, DType>(localAddr(dst), TENSOR_DESC_12(dst), crd0, crd1, ext0, ext1, valueBits); \
    }

REGISTER_FILL_L1(zN, int16_t)
REGISTER_FILL_L1(zN, uint16_t)
REGISTER_FILL_L1(zN, half)
REGISTER_FILL_L1(zN, bf16)
REGISTER_FILL_L1(zN, int32_t)
REGISTER_FILL_L1(zN, uint32_t)
REGISTER_FILL_L1(zN, float)
REGISTER_FILL_L1(nZ, int16_t)
REGISTER_FILL_L1(nZ, uint16_t)
REGISTER_FILL_L1(nZ, half)
REGISTER_FILL_L1(nZ, bf16)
REGISTER_FILL_L1(nZ, int32_t)
REGISTER_FILL_L1(nZ, uint32_t)
REGISTER_FILL_L1(nZ, float)

// Packed fp8 retains zero-only fill.
REGISTER_FILL_L1(zN, fp8_e4m3fn_t)
REGISTER_FILL_L1(zN, fp8_e5m2_t)
REGISTER_FILL_L1(nZ, fp8_e4m3fn_t)
REGISTER_FILL_L1(nZ, fp8_e5m2_t)

// fp4 tiles travel over the ABI as int8_t storage; the element type is restored
// here for the layout math (C0 = 64 fp4 elements), same rule as the GM->L1 fp4
// copy in dma.cpp.
#define REGISTER_FILL_L1_FP4(Layout, TFp4)                                                                           \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_fill_l1_##Layout##_##TFp4(                             \
        memref_t<__cbuf__ int8_t, 1>* dst, DESC_ABI_PARAMS(dst), int64_t crd0, int64_t crd1, int64_t ext0,           \
        int64_t ext1, int32_t valueBits)                                                                             \
    {                                                                                                                \
        fillL1Impl<LayoutTag::Layout, TFp4>(localAddr(dst), TENSOR_DESC_12(dst), crd0, crd1, ext0, ext1, valueBits); \
    }

REGISTER_FILL_L1_FP4(zN, float4_e2m1x2_t)
REGISTER_FILL_L1_FP4(zN, float4_e1m2x2_t)
REGISTER_FILL_L1_FP4(nZ, float4_e2m1x2_t)
REGISTER_FILL_L1_FP4(nZ, float4_e1m2x2_t)

} // extern "C"

#endif
