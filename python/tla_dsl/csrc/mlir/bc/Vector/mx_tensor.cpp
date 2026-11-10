// Tensor-level MX microscaling conversions.
//
// Every conversion direction and dispatch variant lives in this one translation
// unit: quantize (dense -> MX), dequantize (MX -> dense), requantize (MX -> MX),
// the group-size regrouping path, the FP4 -> FP8 payload cast, and the gather /
// scatter / strided / general-layout drivers that wrap them.
//
// They are kept together rather than split per variant because they are one
// feature with one set of shared helpers, and because the runtime links every
// stub in this directory into a single aggregate bitcode module anyway (see
// bc_compile.py: build_meta_op compiles each .cpp and llvm-links the results),
// so the split bought no isolation -- only thirteen files to keep in step.
//
// Each section below ends by undefining its own registration macros, so the
// short names (QUANT, COMMON, PREPARE, ...) stay local to the section that
// defines them.

#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"
#include "kernel_operator.h"

using namespace AscendC::MicroAPI;

// ==========================================================================
// Shared helpers (was mx_tensor_utils.h)
// ==========================================================================

namespace {

__simd_callee__ vector_u8 mxSplat(uint8_t x, vector_bool p)
{
    vector_u8 r;
    vdup(r, x, p, MODE_ZEROING);
    return r;
}
__simd_callee__ vector_u8 mxShr(vector_u8 x, uint8_t n, vector_bool p)
{
    vector_u8 r;
    vshrs(r, x, n, p, MODE_ZEROING);
    return r;
}
__simd_callee__ vector_u8 mxShl(vector_u8 x, uint8_t n, vector_bool p)
{
    vector_u8 r;
    vshls(r, x, n, p, MODE_ZEROING);
    return r;
}
#define MX_BINARY(name, op)                                                 \
    __simd_callee__ vector_u8 name(vector_u8 x, vector_u8 y, vector_bool p) \
    {                                                                       \
        vector_u8 r;                                                        \
        op(r, x, y, p, MODE_ZEROING);                                       \
        return r;                                                           \
    }
MX_BINARY(mxAnd, vand)
MX_BINARY(mxOr, vor)
MX_BINARY(mxAdd, vadd)
MX_BINARY(mxSub, vsub)
MX_BINARY(mxMin, vmin)
#undef MX_BINARY

#define splat(x) mxSplat(x, p)
#define shr(x, n) mxShr(x, n, p)
#define shl(x, n) mxShl(x, n, p)
#define bitAnd(x, y) mxAnd(x, y, p)
#define bitOr(x, y) mxOr(x, y, p)
#define add(x, y) mxAdd(x, y, p)
#define sub(x, y) mxSub(x, y, p)
#define min(x, y) mxMin(x, y, p)

// Decode one nibble from each byte. The caller selects the low or high
// nibble, and the scale is unchanged because the FP4 values are exactly
// representable in either FP8 format.
template <bool High, bool E1M2, int MantissaBits, int Bias>
__simd_callee__ vector_u8 unpackFp4(vector_u8 packed, ave_preg preg)
{
    vector_bool p = convertAVEPregToVecBool(preg);
    vector_u8 n = High ? shr(packed, 4) : bitAnd(packed, splat(15));
    vector_u8 zero = splat(0), one = splat(1);
    vector_u8 mag = bitAnd(n, splat(7));
    vector_u8 value;
    if constexpr (!E1M2) {
        vector_u8 exponent = shr(mag, 1);
        vector_u8 mantissa = bitAnd(mag, one);
        vector_u8 normal = shl(add(exponent, splat(Bias - 1)), MantissaBits);
        vector_u8 expNonzero = sub(zero, min(exponent, one));
        value = bitOr(normal, bitAnd(shl(mantissa, MantissaBits - 1), expNonzero));
    } else {
        vector_u8 exponent = shr(mag, 2);
        vector_u8 mantissa = bitAnd(mag, splat(3));
        vector_u8 normal = add(shl(mantissa, MantissaBits - 2), splat(Bias << MantissaBits));
        vector_u8 mh = shr(mantissa, 1);
        vector_u8 subnormal = bitOr(
            shl(add(mh, splat(Bias - 2)), MantissaBits), shl(bitAnd(bitAnd(mh, mantissa), one), MantissaBits - 1));
        value = bitOr(bitAnd(normal, sub(zero, exponent)), bitAnd(subnormal, sub(exponent, one)));
    }
    vector_u8 nonzero = sub(zero, min(mag, one));
    vector_u8 sign = shl(bitAnd(n, splat(8)), 4);
    return bitOr(bitAnd(value, nonzero), sign);
}

#undef splat
#undef shr
#undef shl
#undef bitAnd
#undef bitOr
#undef add
#undef sub
#undef min

} // namespace

// ==========================================================================
// Shared helpers (was mx_tensor_fp4_store.h)
// ==========================================================================

namespace {
namespace MicroAPI = AscendC::MicroAPI;
__simd_callee__ inline void preserveFp4Tail(__ubuf__ uint8_t* out, MicroAPI::RegTensor<uint8_t>& evens, uint32_t count)
{
    __ubuf__ uint8_t* last = out + count / 2;
    MicroAPI::UnalignRegForLoad priorLoad;
    MicroAPI::RegTensor<uint8_t> prior, lo, hi, selected, tmp0, tmp1, lowBits, highBits;
    MicroAPI::LoadUnAlignPre(priorLoad, last);
    MicroAPI::LoadUnAlign<uint8_t>(prior, priorLoad, last, 1);
    for (int j = 0; j < 8; j++) {
        MicroAPI::Interleave(tmp0, tmp1, prior, prior);
        prior = tmp0;
    }
    uint32_t fullCount = (count + 1) / 2, beforeCount = fullCount - 1;
    MicroAPI::MaskReg full = MicroAPI::UpdateMask<uint8_t>(fullCount),
                      before = MicroAPI::UpdateMask<uint8_t>(beforeCount);
    MicroAPI::MaskReg allBytes = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>(), lastOnly;
    MicroAPI::MaskXor(lastOnly, full, before, allBytes);
    MicroAPI::Duplicate(lowBits, 15);
    MicroAPI::Duplicate(highBits, 240);
    MicroAPI::And(lo, evens, lowBits, allBytes);
    MicroAPI::And(hi, prior, highBits, allBytes);
    MicroAPI::Or(selected, lo, hi, allBytes);
    MicroAPI::Select(evens, selected, evens, lastOnly);
}
} // namespace

// ==========================================================================
// FP4 -> FP8 payload cast with the scale reused unchanged  (was mx_tensor_cast.cpp)
// ==========================================================================

namespace {

// The W4A8 tile uses 128 rows of 64 values on each AIV. Its source and
// destination are aligned and contiguous, and the scale is reused unchanged.
__simd_vf__ static void convertFp4W4A8Aligned(__ubuf__ uint8_t* src, __ubuf__ uint8_t* dst)
{
    using namespace AscendC::MicroAPI;
    MaskReg all = CreateMask<uint8_t, MaskPattern::ALL>();
    ave_preg preg = *reinterpret_cast<ave_preg*>(&all);
    for (uint16_t block = 0; block < 16; ++block) {
        RegTensor<uint8_t> input;
        LoadAlign(input, src + block * 256);
        vector_u8 packed = *reinterpret_cast<vector_u8*>(&input);
        vector_u8 lo = unpackFp4<false, false, 3, 7>(packed, preg);
        vector_u8 hi = unpackFp4<true, false, 3, 7>(packed, preg);
        vector_u8 first, second;
        vintlv(first, second, lo, hi);
        StoreAlign(dst + block * 512, *reinterpret_cast<RegTensor<uint8_t>*>(&first), all);
        StoreAlign(dst + block * 512 + 256, *reinterpret_cast<RegTensor<uint8_t>*>(&second), all);
    }
}

// Process complete rows in the vector function. Coordinates and physical row
// pitches are resolved by the caller; cols is the logical FP4 element count.
template <bool E1M2, int MantissaBits, int Bias, bool MASK_PREFIX = false>
__simd_vf__ static void convertFp4Rows(
    __ubuf__ uint8_t* src, __ubuf__ uint8_t* dst, __ubuf__ uint8_t* srcScale, __ubuf__ uint8_t* dstScale, uint32_t rows,
    uint32_t cols, uint32_t srcStride, uint32_t dstStride, uint32_t srcScaleStride, uint32_t dstScaleStride,
    uint32_t maskLimit)
{
    using namespace AscendC::MicroAPI;
    for (uint32_t row = 0; row < rows; ++row) {
        if (cols == 0)
            continue;
        __ubuf__ uint8_t* rowSrc = src + row * srcStride;
        __ubuf__ uint8_t* rowDst = dst + row * dstStride;
        UnalignRegForLoad payloadLoad;
        UnalignRegForStore payloadStore;
        LoadUnAlignPre(payloadLoad, rowSrc);
        // The first sixteen lanes form the complete signed FP4 lookup table.
        RegTensor<int8_t> indices;
        Arange<int8_t>(indices, int8_t(0));
        MaskReg all = CreateMask<uint8_t, MaskPattern::ALL>();
        ave_preg allPreg = *reinterpret_cast<ave_preg*>(&all);
        vector_u8 sequence = *reinterpret_cast<vector_u8*>(&indices);
        vector_u8 tableVec = unpackFp4<false, E1M2, MantissaBits, Bias>(sequence, allPreg);
        RegTensor<uint8_t> table = *reinterpret_cast<RegTensor<uint8_t>*>(&tableVec);
        RegTensor<uint8_t> nibbleMask;
        Duplicate(nibbleMask, uint8_t(15));
        for (uint32_t pos = 0; pos < cols; pos += 512) {
            uint32_t count = cols - pos > 512 ? 512 : cols - pos;
            uint32_t inputCount = (count + 1) / 2;
            uint32_t maskCount = inputCount;
            MaskReg inputMask = UpdateMask<uint8_t>(maskCount);
            RegTensor<uint8_t> input;
            LoadUnAlign<uint8_t>(input, payloadLoad, rowSrc, inputCount);
            RegTensor<uint8_t> lowIndex, highIndex, low, high;
            And(lowIndex, input, nibbleMask, inputMask);
            ShiftRights(highIndex, input, int16_t(4), inputMask);
            Gather<uint8_t, uint8_t>(low, table, lowIndex);
            Gather<uint8_t, uint8_t>(high, table, highIndex);
            vector_u8 lo = *reinterpret_cast<vector_u8*>(&low);
            vector_u8 hi = *reinterpret_cast<vector_u8*>(&high);
            vector_u8 first, second;
            vintlv(first, second, lo, hi);
            if constexpr (MASK_PREFIX) {
                uint32_t start = row * cols + pos;
                uint32_t remaining = start < maskLimit ? maskLimit - start : 0;
                MaskReg firstMask = UpdateMask<uint8_t>(remaining);
                MaskReg secondMask = UpdateMask<uint8_t>(remaining);
                RegTensor<uint8_t> zeros;
                Duplicate(zeros, uint8_t(0));
                auto& firstReg = *reinterpret_cast<RegTensor<uint8_t>*>(&first);
                auto& secondReg = *reinterpret_cast<RegTensor<uint8_t>*>(&second);
                Select(firstReg, firstReg, zeros, firstMask);
                Select(secondReg, secondReg, zeros, secondMask);
            }
            uint32_t firstCount = count > 256 ? 256 : count;
            StoreUnAlign<uint8_t>(rowDst, *reinterpret_cast<RegTensor<uint8_t>*>(&first), payloadStore, firstCount);
            if (count > 256) {
                StoreUnAlign<uint8_t>(
                    rowDst, *reinterpret_cast<RegTensor<uint8_t>*>(&second), payloadStore, count - 256);
            }
        }
        StoreUnAlignPost<uint8_t>(rowDst, payloadStore, 0);
        // FP4 values are exactly representable in either FP8 format, so their
        // E8M0 block scales carry through unchanged.
        uint32_t numScales = (cols + 31) / 32;
        __ubuf__ uint8_t* scaleSrc = srcScale + row * srcScaleStride;
        __ubuf__ uint8_t* scaleDst = dstScale + row * dstScaleStride;
        if (scaleSrc == scaleDst)
            continue;
        UnalignRegForLoad scaleLoad;
        UnalignRegForStore scaleStore;
        LoadUnAlignPre(scaleLoad, scaleSrc);
        for (uint32_t pos = 0; pos < numScales; pos += 256) {
            uint32_t count = numScales - pos > 256 ? 256 : numScales - pos;
            RegTensor<uint8_t> scale;
            LoadUnAlign<uint8_t>(scale, scaleLoad, scaleSrc, count);
            StoreUnAlign<uint8_t>(scaleDst, scale, scaleStore, count);
        }
        StoreUnAlignPost<uint8_t>(scaleDst, scaleStore, 0);
    }
}

#define REGISTER_MX_TENSOR_FP4_TO_FP8(name, e1m2, mantissa, bias)                                                     \
    __aiv__ __attribute__((always_inline)) void name(                                                                 \
        uint64_t src, uint64_t dst, uint64_t srcScale, uint64_t dstScale, uint32_t rows, uint32_t cols,               \
        uint32_t srcStride, uint32_t dstStride, uint32_t srcScaleStride, uint32_t dstScaleStride)                     \
    {                                                                                                                 \
        if constexpr (!(e1m2) && (mantissa) == 3 && (bias) == 7) {                                                    \
            if (rows == 128 && cols == 64 && srcStride == 32 && dstStride == 64 && srcScaleStride == 2 &&             \
                dstScaleStride == 2 && srcScale == dstScale && ((src | dst) & 31) == 0) {                             \
                convertFp4W4A8Aligned(                                                                                \
                    reinterpret_cast<__ubuf__ uint8_t*>(src), reinterpret_cast<__ubuf__ uint8_t*>(dst));              \
                return;                                                                                               \
            }                                                                                                         \
        }                                                                                                             \
        /* Full 32-element scale blocks can span physically contiguous rows. */                                       \
        if (cols != 0 && cols % 32 == 0 && rows <= UINT32_MAX / cols && srcStride == cols / 2 && dstStride == cols && \
            srcScaleStride == cols / 32 && dstScaleStride == cols / 32) {                                             \
            cols *= rows;                                                                                             \
            rows = 1;                                                                                                 \
        }                                                                                                             \
        convertFp4Rows<e1m2, mantissa, bias>(                                                                         \
            reinterpret_cast<__ubuf__ uint8_t*>(src), reinterpret_cast<__ubuf__ uint8_t*>(dst),                       \
            reinterpret_cast<__ubuf__ uint8_t*>(srcScale), reinterpret_cast<__ubuf__ uint8_t*>(dstScale), rows, cols, \
            srcStride, dstStride, srcScaleStride, dstScaleStride, 0);                                                 \
    }

} // namespace

extern "C" {
REGISTER_MX_TENSOR_FP4_TO_FP8(tla_mx_tensor_e2m1_e4m3, false, 3, 7)
REGISTER_MX_TENSOR_FP4_TO_FP8(tla_mx_tensor_e2m1_e5m2, false, 2, 15)
REGISTER_MX_TENSOR_FP4_TO_FP8(tla_mx_tensor_e1m2_e4m3, true, 3, 7)
REGISTER_MX_TENSOR_FP4_TO_FP8(tla_mx_tensor_e1m2_e5m2, true, 2, 15)

#define REGISTER_MX_TENSOR_FP4_TO_FP8_MASKED(name, e1m2, mantissa, bias)                                              \
    __aiv__ __attribute__((always_inline)) void name(                                                                 \
        uint64_t src, uint64_t dst, uint64_t srcScale, uint64_t dstScale, uint32_t rows, uint32_t cols,               \
        uint32_t srcStride, uint32_t dstStride, uint32_t srcScaleStride, uint32_t dstScaleStride, uint32_t maskLimit) \
    {                                                                                                                 \
        convertFp4Rows<e1m2, mantissa, bias, true>(                                                                   \
            reinterpret_cast<__ubuf__ uint8_t*>(src), reinterpret_cast<__ubuf__ uint8_t*>(dst),                       \
            reinterpret_cast<__ubuf__ uint8_t*>(srcScale), reinterpret_cast<__ubuf__ uint8_t*>(dstScale), rows, cols, \
            srcStride, dstStride, srcScaleStride, dstScaleStride, maskLimit);                                         \
    }

REGISTER_MX_TENSOR_FP4_TO_FP8_MASKED(tla_mx_tensor_e2m1_e4m3_masked, false, 3, 7)
REGISTER_MX_TENSOR_FP4_TO_FP8_MASKED(tla_mx_tensor_e2m1_e5m2_masked, false, 2, 15)
REGISTER_MX_TENSOR_FP4_TO_FP8_MASKED(tla_mx_tensor_e1m2_e4m3_masked, true, 3, 7)
REGISTER_MX_TENSOR_FP4_TO_FP8_MASKED(tla_mx_tensor_e1m2_e5m2_masked, true, 2, 15)
}

#undef REGISTER_MX_TENSOR_FP4_TO_FP8
#undef REGISTER_MX_TENSOR_FP4_TO_FP8_MASKED

// ==========================================================================
// Dense -> MX quantization  (was mx_tensor_quant.cpp)
// ==========================================================================

template <typename SRC, int FORMAT, bool MASK_PREFIX = false>
__simd_vf__ static void quantizeToMx(
    __ubuf__ SRC* src, __ubuf__ uint8_t* dst, __ubuf__ uint8_t* scale, uint32_t rows, uint32_t cols, uint32_t srcStride,
    uint32_t dstStride, uint32_t scaleStride, uint32_t roundMode, uint32_t maskLimit)
{
    static constexpr CastTrait fp8Trait = {
        RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    for (uint32_t row = 0; row < rows; ++row) {
        __ubuf__ SRC* in = src + row * srcStride;
        __ubuf__ uint8_t* out = dst + row * dstStride;
        __ubuf__ uint8_t* exponents = scale + row * scaleStride;
        UnalignRegForLoad inputLoad;
        UnalignRegForStore outputStore, scaleStore;
        LoadUnAlignPre(inputLoad, in);
        for (uint32_t pos = 0; pos < cols; pos += 64) {
            uint32_t count = cols - pos > 64 ? 64 : cols - pos;
            uint32_t num = count;
            MaskReg active = UpdateMask<float>(num);
            MaskReg all = CreateMask<float, MaskPattern::ALL>();
            if constexpr (MASK_PREFIX) {
                uint32_t start = row * cols + pos;
                uint32_t remaining = start < maskLimit ? maskLimit - start : 0;
                MaskReg selected = UpdateMask<float>(remaining);
                MaskAnd(active, active, selected, all);
            }
            RegTensor<float> x, maximum, a, b, zero, normalized;
            if constexpr (std::is_same<SRC, float>::value)
                LoadUnAlign<float>(x, inputLoad, in, count);
            else {
                RegTensor<SRC> raw, spread, unused;
                LoadUnAlign<SRC>(raw, inputLoad, in, count);
                Interleave(spread, unused, raw, raw);
                static constexpr CastTrait widen = {
                    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
                Cast<float, SRC, widen>(x, spread, active);
            }
            Duplicate(zero, 0.0f);
            Abs(maximum, x, active);
            Select(maximum, maximum, zero, active);
            // Floating-point Max does not reliably propagate NaNs, so the block
            // maximum alone cannot tell a NaN group from a finite one. Reduce a
            // separate flag through the same tree, so one NaN marks its whole MX
            // group as NaN -- which is what the OCP MX scale encoding asks for
            // and what requantize already does. |x| puts every NaN above the
            // bit pattern of +inf, and the sign bit is already cleared.
            RegTensor<int32_t> nanValues, nanA, nanB, allOnes, noneSet;
            MaskReg isNan;
            Duplicate(allOnes, 255);
            Duplicate(noneSet, 0);
            CompareScalar<int32_t, AscendC::CMPMODE::GT>(
                isNan, reinterpret_cast<RegTensor<int32_t>&>(maximum), int32_t(0x7f800000), active);
            Select(nanValues, allOnes, noneSet, isNan);
            for (uint32_t s = 0; s < 5; ++s) {
                DeInterleave(a, b, maximum, maximum);
                Max(maximum, a, b, all);
                DeInterleave(nanA, nanB, nanValues, nanValues);
                Max(nanValues, nanA, nanB, all);
            }
            RegTensor<int32_t> exponent, expMask, emax, scale127, blockScale, shifted, ones254;
            Duplicate(expMask, 255);
            Duplicate(emax, FORMAT == 0 ? 2 : FORMAT == 1 ? 0 : FORMAT == 2 ? 8 : 15);
            Duplicate(scale127, 127);
            Duplicate(ones254, 254);
            RegTensor<int32_t> zeroInt;
            Duplicate(zeroInt, 0);
            ShiftRights(exponent, reinterpret_cast<RegTensor<int32_t>&>(maximum), int16_t(23), all);
            And(exponent, exponent, expMask, all);
            Sub(blockScale, exponent, emax, all);
            Max(blockScale, blockScale, zeroInt, all);
            MaskReg isZero;
            CompareScalar<float, AscendC::CMPMODE::EQ>(isZero, maximum, 0.0f, all);
            Select(blockScale, scale127, blockScale, isZero);
            // E8M0 0xFF is the NaN scale. It must be applied after the all-zero
            // case so a group holding both a zero and a NaN still reads as NaN,
            // and the payload written below is then immaterial: dequantize
            // replaces every element of a 0xFF-scaled group with a quiet NaN.
            Compare<int32_t, AscendC::CMPMODE::EQ>(isNan, nanValues, allOnes, all);
            Select(blockScale, allOnes, blockScale, isNan);
            RegTensor<int32_t> spread = blockScale, sa, sb;
            for (uint32_t s = 0; s < 5; ++s) {
                Interleave(sa, sb, spread, spread);
                spread = sa;
            }
            Sub(shifted, ones254, spread, all);
            ShiftLefts(shifted, shifted, int16_t(23), all);
            Mul(normalized, x, reinterpret_cast<RegTensor<float>&>(shifted), active);
            RegTensor<uint8_t> t0, t1;
            if constexpr (FORMAT < 2) {
                RegTensor<float> magnitude;
                Abs(magnitude, normalized, active);
                RegTensor<int32_t> code, one, zeroCode, sign;
                Duplicate(code, 0);
                Duplicate(zeroCode, 0);
                Duplicate(one, 1);
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(
                        greater, magnitude, FORMAT == 1 ? 0.125f : 0.25f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(
                        greater, magnitude, FORMAT == 1 ? 0.375f : 0.75f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(
                        greater, magnitude, FORMAT == 1 ? 0.625f : 1.25f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(
                        greater, magnitude, FORMAT == 1 ? 0.875f : 1.75f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(greater, magnitude, FORMAT == 1 ? 1.125f : 2.5f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(greater, magnitude, FORMAT == 1 ? 1.375f : 3.5f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(greater, magnitude, FORMAT == 1 ? 1.625f : 5.0f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                ShiftRights(sign, reinterpret_cast<RegTensor<int32_t>&>(x), int16_t(28), active);
                RegTensor<int32_t> eight;
                Duplicate(eight, 8);
                And(sign, sign, eight, active);
                Or(code, code, sign, active);
                RegTensor<uint8_t> evens, odds;
                DeInterleave(
                    t0, t1, reinterpret_cast<RegTensor<uint8_t>&>(code), reinterpret_cast<RegTensor<uint8_t>&>(code));
                DeInterleave(t0, t1, t0, t0);
                DeInterleave(evens, odds, t0, t0);
                uint32_t nbytes = (count + 1) / 2;
                MaskReg byteMask = UpdateMask<uint8_t>(nbytes);
                ShiftLefts(odds, odds, int16_t(4), byteMask);
                Or(evens, evens, odds, byteMask);
                if (count & 1)
                    preserveFp4Tail(out, evens, count);
                StoreUnAlign<uint8_t>(out, evens, outputStore, (count + 1) / 2);
            } else {
                using Dest = typename std::conditional<FORMAT == 2, fp8_e4m3fn_t, fp8_e5m2_t>::type;
                RegTensor<Dest> fp8;
                Cast<Dest, float, fp8Trait>(fp8, normalized, active);
                static constexpr CastTrait decodeTrait = {
                    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
                RegTensor<int32_t> nextBits, oneBit, selectedBits;
                Duplicate(oneBit, 1);
                Add(nextBits, reinterpret_cast<RegTensor<int32_t>&>(fp8), oneBit, active);
                RegTensor<float> nearest, next, nearestAbs, nextAbs, inputAbs, midpoint;
                Cast<float, Dest, decodeTrait>(nearest, fp8, active);
                Abs(nearestAbs, nearest, active);
                Abs(inputAbs, normalized, active);
                // The hardware cast rounds ties to even (CAST_RINT). CAST_ROUND asks for
                // ties away from zero, so for that mode -- and only that mode -- an exact
                // midpoint is nudged one code away. The directed modes below re-derive
                // their result from a magnitude comparison and so agree either way, while
                // CAST_NONE and CAST_RINT must keep the native result untouched.
                //
                // roundMode is the tla.cast enum: 0 CAST_ROUND, 1 CAST_FLOOR, 2 CAST_CEIL,
                // 3 CAST_TRUNC, 4 CAST_NONE, 5 CAST_RINT. CAST_ODD (6) is refused by the
                // frontend, so every value that reaches here is handled.
                if (roundMode == 0) {
                    Cast<float, Dest, decodeTrait>(next, reinterpret_cast<RegTensor<Dest>&>(nextBits), active);
                    Abs(nextAbs, next, active);
                    Add(midpoint, nearestAbs, nextAbs, active);
                    Muls(midpoint, midpoint, 0.5f, active);
                    MaskReg isTie, below;
                    Compare<float, AscendC::CMPMODE::EQ>(isTie, inputAbs, midpoint, active);
                    Compare<float, AscendC::CMPMODE::GT>(below, inputAbs, nearestAbs, active);
                    Select(selectedBits, nextBits, reinterpret_cast<RegTensor<int32_t>&>(fp8), isTie);
                    Select(
                        reinterpret_cast<RegTensor<int32_t>&>(fp8), selectedBits,
                        reinterpret_cast<RegTensor<int32_t>&>(fp8), below);
                }
                if (roundMode >= 1 && roundMode <= 3) {
                    // fp8 may have moved above, so re-decode before comparing.
                    Cast<float, Dest, decodeTrait>(nearest, fp8, active);
                    Abs(nearestAbs, nearest, active);
                    MaskReg larger, smaller, canIncrease, negative;
                    Compare<float, AscendC::CMPMODE::GT>(larger, nearestAbs, inputAbs, active);
                    Compare<float, AscendC::CMPMODE::GT>(smaller, inputAbs, nearestAbs, active);
                    CompareScalar<float, AscendC::CMPMODE::LT>(
                        canIncrease, nearestAbs, FORMAT == 2 ? 448.f : 57344.f, active);
                    CompareScalar<int32_t, AscendC::CMPMODE::LT>(
                        negative, reinterpret_cast<RegTensor<int32_t>&>(normalized), 0, active);
                    RegTensor<int32_t> down, up, toward, away;
                    auto& bits = reinterpret_cast<RegTensor<int32_t>&>(fp8);
                    Sub(down, bits, oneBit, active);
                    Add(up, bits, oneBit, active);
                    Select(toward, down, bits, larger);
                    Select(away, up, bits, smaller);
                    Select(away, away, bits, canIncrease);
                    if (roundMode == 1)
                        Select(bits, away, toward, negative); // floor
                    if (roundMode == 2)
                        Select(bits, toward, away, negative); // ceil
                    if (roundMode == 3)
                        bits = toward; // truncation
                }

                DeInterleave(
                    t0, t1, reinterpret_cast<RegTensor<uint8_t>&>(fp8), reinterpret_cast<RegTensor<uint8_t>&>(fp8));
                DeInterleave(t0, t1, t0, t0);
                StoreUnAlign<uint8_t>(out, t0, outputStore, count);
            }
            DeInterleave(
                t0, t1, reinterpret_cast<RegTensor<uint8_t>&>(blockScale),
                reinterpret_cast<RegTensor<uint8_t>&>(blockScale));
            DeInterleave(t0, t1, t0, t0);
            StoreUnAlign<uint8_t>(exponents, t0, scaleStore, (count + 31) / 32);
        }
        StoreUnAlignPost<uint8_t>(out, outputStore, 0);
        StoreUnAlignPost<uint8_t>(exponents, scaleStore, 0);
    }
}
#define MX_QUANT(SRCNAME, SRCTYPE, DSTNAME, DST)                                                           \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_tensor_quant_##SRCNAME##_##DSTNAME(      \
        uint64_t a, uint64_t b, uint64_t c, uint32_t r, uint32_t n, uint32_t as, uint32_t bs, uint32_t cs, \
        uint32_t roundMode)                                                                                \
    {                                                                                                      \
        quantizeToMx<SRCTYPE, DST>(                                                                        \
            reinterpret_cast<__ubuf__ SRCTYPE*>(a), reinterpret_cast<__ubuf__ uint8_t*>(b),                \
            reinterpret_cast<__ubuf__ uint8_t*>(c), r, n, as, bs, cs, roundMode, 0);                       \
    }

MX_QUANT(f32, float, e2m1, 0)
MX_QUANT(f32, float, e1m2, 1)
MX_QUANT(f32, float, e4m3, 2)
MX_QUANT(f32, float, e5m2, 3)
MX_QUANT(f16, half, e2m1, 0)
MX_QUANT(f16, half, e1m2, 1)
MX_QUANT(f16, half, e4m3, 2)
MX_QUANT(f16, half, e5m2, 3)
MX_QUANT(bf16, bfloat16_t, e2m1, 0)
MX_QUANT(bf16, bfloat16_t, e1m2, 1)
MX_QUANT(bf16, bfloat16_t, e4m3, 2)
MX_QUANT(bf16, bfloat16_t, e5m2, 3)

#undef MX_QUANT

#define MX_QUANT_MASKED(SRCNAME, SRCTYPE, DSTNAME, DST)                                                        \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_tensor_quant_##SRCNAME##_##DSTNAME##_masked( \
        uint64_t a, uint64_t b, uint64_t c, uint32_t r, uint32_t n, uint32_t as, uint32_t bs, uint32_t cs,     \
        uint32_t roundMode, uint32_t maskLimit)                                                                \
    {                                                                                                          \
        quantizeToMx<SRCTYPE, DST, true>(                                                                      \
            reinterpret_cast<__ubuf__ SRCTYPE*>(a), reinterpret_cast<__ubuf__ uint8_t*>(b),                    \
            reinterpret_cast<__ubuf__ uint8_t*>(c), r, n, as, bs, cs, roundMode, maskLimit);                   \
    }

MX_QUANT_MASKED(f32, float, e2m1, 0)
MX_QUANT_MASKED(f32, float, e1m2, 1)
MX_QUANT_MASKED(f32, float, e4m3, 2)
MX_QUANT_MASKED(f32, float, e5m2, 3)
MX_QUANT_MASKED(f16, half, e2m1, 0)
MX_QUANT_MASKED(f16, half, e1m2, 1)
MX_QUANT_MASKED(f16, half, e4m3, 2)
MX_QUANT_MASKED(f16, half, e5m2, 3)
MX_QUANT_MASKED(bf16, bfloat16_t, e2m1, 0)
MX_QUANT_MASKED(bf16, bfloat16_t, e1m2, 1)
MX_QUANT_MASKED(bf16, bfloat16_t, e4m3, 2)
MX_QUANT_MASKED(bf16, bfloat16_t, e5m2, 3)

#undef MX_QUANT_MASKED

// ==========================================================================
// MX -> dense dequantization  (was mx_tensor_dequant.cpp)
// ==========================================================================

template <int FORMAT, typename OUT, bool MASK_PREFIX = false>
__simd_vf__ static void dequantize(
    __ubuf__ uint8_t* in, __ubuf__ uint8_t* scale, __ubuf__ OUT* out, uint32_t rows, uint32_t cols, uint32_t ss,
    uint32_t qs, uint32_t ds, uint32_t roundMode, uint32_t maskLimit)
{
    static constexpr CastTrait trait = {
        RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    for (uint32_t row = 0; row < rows; row++) {
        __ubuf__ uint8_t* src = in + row * ss;
        __ubuf__ uint8_t* sc = scale + row * qs;
        __ubuf__ OUT* dst = out + row * ds;
        UnalignRegForLoad il, sl;
        UnalignRegForStore os;
        LoadUnAlignPre(il, src);
        LoadUnAlignPre(sl, sc);
        for (uint32_t pos = 0; pos < cols; pos += 64) {
            uint32_t count = cols - pos > 64 ? 64 : cols - pos;
            uint32_t n = count;
            MaskReg active = UpdateMask<float>(n), all = CreateMask<float, MaskPattern::ALL>();
            if constexpr (MASK_PREFIX) {
                uint32_t start = row * cols + pos;
                uint32_t remaining = start < maskLimit ? maskLimit - start : 0;
                MaskReg selected = UpdateMask<float>(remaining);
                MaskAnd(active, active, selected, all);
            }
            RegTensor<uint8_t> packed, p0, p1, scalePacked;
            if constexpr (FORMAT < 2) {
                uint32_t inputCount = (count + 1) / 2, nMask = inputCount;
                MaskReg pmask = UpdateMask<uint8_t>(nMask);
                ave_preg preg = *reinterpret_cast<ave_preg*>(&pmask);
                LoadUnAlign<uint8_t>(packed, il, src, inputCount);
                vector_u8 raw = *reinterpret_cast<vector_u8*>(&packed), low, high, first, second;
                low = unpackFp4<false, FORMAT == 1, 3, 7>(raw, preg);
                high = unpackFp4<true, FORMAT == 1, 3, 7>(raw, preg);
                vintlv(first, second, low, high);
                packed = *reinterpret_cast<RegTensor<uint8_t>*>(&first);
            } else
                LoadUnAlign<uint8_t>(packed, il, src, count);
            Interleave(p0, p1, packed, packed);
            Interleave(packed, p1, p0, p0);
            RegTensor<float> x;
            if constexpr (FORMAT == 3)
                Cast<float, fp8_e5m2_t, trait>(x, reinterpret_cast<RegTensor<fp8_e5m2_t>&>(packed), active);
            else
                Cast<float, fp8_e4m3fn_t, trait>(x, reinterpret_cast<RegTensor<fp8_e4m3fn_t>&>(packed), active);
            LoadUnAlign<uint8_t>(scalePacked, sl, sc, (count + 31) / 32);
            Interleave(p0, p1, scalePacked, scalePacked);
            Interleave(scalePacked, p1, p0, p0);
            RegTensor<int32_t> bits, mask, zero, subnormal, nanBits;
            Duplicate(mask, 255);
            Duplicate(zero, 0);
            Duplicate(subnormal, 0x00400000);
            Duplicate(nanBits, 0x7fc00000);
            And(bits, reinterpret_cast<RegTensor<int32_t>&>(scalePacked), mask, all);
            MaskReg iszero, isnan;
            Compare<int32_t, AscendC::CMPMODE::EQ>(iszero, bits, zero, all);
            Compare<int32_t, AscendC::CMPMODE::EQ>(isnan, bits, mask, all);
            ShiftLefts(bits, bits, int16_t(23), all);
            Select(bits, subnormal, bits, iszero);
            Select(bits, nanBits, bits, isnan);
            RegTensor<int32_t> a, b;
            for (int i = 0; i < 5; i++) {
                Interleave(a, b, bits, bits);
                bits = a;
            }
            Mul(x, x, reinterpret_cast<RegTensor<float>&>(bits), active);
            if constexpr (std::is_same<OUT, float>::value)
                StoreUnAlign<float>(dst, x, os, count);
            else {
                static constexpr CastTrait narrow = {
                    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
                RegTensor<OUT> converted;
                Cast<OUT, float, narrow>(converted, x, active);
                RegTensor<int32_t> nextBits, oneBit, selectedBits;
                Duplicate(oneBit, 1);
                auto& bits = reinterpret_cast<RegTensor<int32_t>&>(converted);
                Add(nextBits, bits, oneBit, active);
                RegTensor<float> nearest, next, nearestAbs, nextAbs, inputAbs, midpoint;
                Cast<float, OUT, trait>(nearest, converted, active);
                Abs(nearestAbs, nearest, active);
                Abs(inputAbs, x, active);
                // The hardware cast rounds ties to even (CAST_RINT). CAST_ROUND asks for
                // ties away from zero, so for that mode -- and only that mode -- an exact
                // midpoint is nudged one code away. The directed modes below re-derive
                // their result from a magnitude comparison and so agree either way, while
                // CAST_NONE and CAST_RINT must keep the native result untouched.
                //
                // roundMode is the tla.cast enum: 0 CAST_ROUND, 1 CAST_FLOOR, 2 CAST_CEIL,
                // 3 CAST_TRUNC, 4 CAST_NONE, 5 CAST_RINT. CAST_ODD (6) is refused by the
                // frontend, so every value that reaches here is handled.
                if (roundMode == 0) {
                    Cast<float, OUT, trait>(next, reinterpret_cast<RegTensor<OUT>&>(nextBits), active);
                    Abs(nextAbs, next, active);
                    Add(midpoint, nearestAbs, nextAbs, active);
                    Muls(midpoint, midpoint, 0.5f, active);
                    MaskReg isTie, below;
                    Compare<float, AscendC::CMPMODE::EQ>(isTie, inputAbs, midpoint, active);
                    Compare<float, AscendC::CMPMODE::GT>(below, inputAbs, nearestAbs, active);
                    Select(selectedBits, nextBits, bits, isTie);
                    Select(bits, selectedBits, bits, below);
                }
                if (roundMode >= 1 && roundMode <= 3) {
                    Cast<float, OUT, trait>(nearest, converted, active);
                    Abs(nearestAbs, nearest, active);
                    MaskReg larger, smaller, negative;
                    Compare<float, AscendC::CMPMODE::GT>(larger, nearestAbs, inputAbs, active);
                    Compare<float, AscendC::CMPMODE::GT>(smaller, inputAbs, nearestAbs, active);
                    CompareScalar<int32_t, AscendC::CMPMODE::LT>(
                        negative, reinterpret_cast<RegTensor<int32_t>&>(x), 0, active);
                    RegTensor<int32_t> down, up, toward, away;
                    Sub(down, bits, oneBit, active);
                    Add(up, bits, oneBit, active);
                    Select(toward, down, bits, larger);
                    Select(away, up, bits, smaller);
                    if (roundMode == 1)
                        Select(bits, away, toward, negative);
                    else if (roundMode == 2)
                        Select(bits, toward, away, negative);
                    else if (roundMode == 3)
                        bits = toward;
                }
                RegTensor<OUT> compact, discarded;
                DeInterleave(compact, discarded, converted, converted);
                StoreUnAlign<OUT>(dst, compact, os, count);
            }
        }
        StoreUnAlignPost<OUT>(dst, os, 0);
    }
}
#define MX_DEQUANT(SRCNAME, SRC, DSTNAME, DSTTYPE)                                                         \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_tensor_dequant_##SRCNAME##_##DSTNAME(    \
        uint64_t a, uint64_t b, uint64_t c, uint32_t r, uint32_t n, uint32_t as, uint32_t bs, uint32_t cs, \
        uint32_t roundMode)                                                                                \
    {                                                                                                      \
        dequantize<SRC, DSTTYPE>(                                                                          \
            reinterpret_cast<__ubuf__ uint8_t*>(a), reinterpret_cast<__ubuf__ uint8_t*>(b),                \
            reinterpret_cast<__ubuf__ DSTTYPE*>(c), r, n, as, bs, cs, roundMode, 0);                       \
    }

MX_DEQUANT(e2m1, 0, f32, float)
MX_DEQUANT(e2m1, 0, f16, half)
MX_DEQUANT(e2m1, 0, bf16, bfloat16_t)
MX_DEQUANT(e1m2, 1, f32, float)
MX_DEQUANT(e1m2, 1, f16, half)
MX_DEQUANT(e1m2, 1, bf16, bfloat16_t)
MX_DEQUANT(e4m3, 2, f32, float)
MX_DEQUANT(e4m3, 2, f16, half)
MX_DEQUANT(e4m3, 2, bf16, bfloat16_t)
MX_DEQUANT(e5m2, 3, f32, float)
MX_DEQUANT(e5m2, 3, f16, half)
MX_DEQUANT(e5m2, 3, bf16, bfloat16_t)

#undef MX_DEQUANT

#define MX_DEQUANT_MASKED(SRCNAME, SRC, DSTNAME, DSTTYPE)                                                        \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_tensor_dequant_##SRCNAME##_##DSTNAME##_masked( \
        uint64_t a, uint64_t b, uint64_t c, uint32_t r, uint32_t n, uint32_t as, uint32_t bs, uint32_t cs,       \
        uint32_t roundMode, uint32_t maskLimit)                                                                  \
    {                                                                                                            \
        dequantize<SRC, DSTTYPE, true>(                                                                          \
            reinterpret_cast<__ubuf__ uint8_t*>(a), reinterpret_cast<__ubuf__ uint8_t*>(b),                      \
            reinterpret_cast<__ubuf__ DSTTYPE*>(c), r, n, as, bs, cs, roundMode, maskLimit);                     \
    }

MX_DEQUANT_MASKED(e2m1, 0, f32, float)
MX_DEQUANT_MASKED(e2m1, 0, f16, half)
MX_DEQUANT_MASKED(e2m1, 0, bf16, bfloat16_t)
MX_DEQUANT_MASKED(e1m2, 1, f32, float)
MX_DEQUANT_MASKED(e1m2, 1, f16, half)
MX_DEQUANT_MASKED(e1m2, 1, bf16, bfloat16_t)
MX_DEQUANT_MASKED(e4m3, 2, f32, float)
MX_DEQUANT_MASKED(e4m3, 2, f16, half)
MX_DEQUANT_MASKED(e4m3, 2, bf16, bfloat16_t)
MX_DEQUANT_MASKED(e5m2, 3, f32, float)
MX_DEQUANT_MASKED(e5m2, 3, f16, half)
MX_DEQUANT_MASKED(e5m2, 3, bf16, bfloat16_t)

#undef MX_DEQUANT_MASKED

// ==========================================================================
// MX -> MX requantization  (was mx_tensor_requant.cpp)
// ==========================================================================

// Requantize in the unscaled payload domain. Calculating the new E8M0
// exponent relative to the input exponent avoids overflowing a dense f32
// intermediate when the input block has a large scale.
template <int INPUT, int OUTPUT, bool MASK_PREFIX = false>
__simd_vf__ static void requantize(
    __ubuf__ uint8_t* in, __ubuf__ uint8_t* scale, __ubuf__ uint8_t* out, __ubuf__ uint8_t* outScale, uint32_t rows,
    uint32_t cols, uint32_t ss, uint32_t qs, uint32_t ds, uint32_t os, uint32_t roundMode, uint32_t maskLimit)
{
    static constexpr CastTrait decodeTrait = {
        RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr CastTrait encodeTrait = {
        RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    for (uint32_t row = 0; row < rows; row++) {
        __ubuf__ uint8_t* src = in + row * ss;
        __ubuf__ uint8_t* sc = scale + row * qs;
        __ubuf__ uint8_t* dst = out + row * ds;
        __ubuf__ uint8_t* destScale = outScale + row * os;
        UnalignRegForLoad il, sl;
        UnalignRegForStore ow, sw;
        LoadUnAlignPre(il, src);
        LoadUnAlignPre(sl, sc);
        for (uint32_t pos = 0; pos < cols; pos += 64) {
            uint32_t count = cols - pos > 64 ? 64 : cols - pos, n = count;
            MaskReg active = UpdateMask<float>(n), all = CreateMask<float, MaskPattern::ALL>();
            if constexpr (MASK_PREFIX) {
                uint32_t start = row * cols + pos;
                uint32_t remaining = start < maskLimit ? maskLimit - start : 0;
                MaskReg selected = UpdateMask<float>(remaining);
                MaskAnd(active, active, selected, all);
            }
            RegTensor<uint8_t> packed, p0, p1, scaleBytes;
            if constexpr (INPUT < 2) {
                uint32_t inputCount = (count + 1) / 2, nm = inputCount;
                MaskReg p = UpdateMask<uint8_t>(nm);
                ave_preg preg = *reinterpret_cast<ave_preg*>(&p);
                LoadUnAlign<uint8_t>(packed, il, src, inputCount);
                vector_u8 raw = *reinterpret_cast<vector_u8*>(&packed), lo, hi, first, second;
                lo = unpackFp4<false, INPUT == 1, 3, 7>(raw, preg);
                hi = unpackFp4<true, INPUT == 1, 3, 7>(raw, preg);
                vintlv(first, second, lo, hi);
                packed = *reinterpret_cast<RegTensor<uint8_t>*>(&first);
            } else {
                LoadUnAlign<uint8_t>(packed, il, src, count);
            }
            Interleave(p0, p1, packed, packed);
            Interleave(packed, p1, p0, p0);
            RegTensor<float> x;
            if constexpr (INPUT == 3)
                Cast<float, fp8_e5m2_t, decodeTrait>(x, reinterpret_cast<RegTensor<fp8_e5m2_t>&>(packed), active);
            else
                Cast<float, fp8_e4m3fn_t, decodeTrait>(x, reinterpret_cast<RegTensor<fp8_e4m3fn_t>&>(packed), active);
            LoadUnAlign<uint8_t>(scaleBytes, sl, sc, (count + 31) / 32);
            Interleave(p0, p1, scaleBytes, scaleBytes);
            Interleave(scaleBytes, p1, p0, p0);
            RegTensor<int32_t> oldScale, byteMask, zeroInt, max254;
            Duplicate(byteMask, 255);
            Duplicate(zeroInt, 0);
            Duplicate(max254, 254);
            And(oldScale, reinterpret_cast<RegTensor<int32_t>&>(scaleBytes), byteMask, all);
            RegTensor<float> mx, a, b, z;
            Duplicate(z, 0.f);
            Abs(mx, x, active);
            Select(mx, mx, z, active);
            // Floating-point Max does not reliably propagate NaNs. Reduce a
            // separate flag so one NaN marks its entire MX group as NaN.
            RegTensor<int32_t> nanValues, nanA, nanB;
            MaskReg nanPayload;
            CompareScalar<int32_t, AscendC::CMPMODE::GT>(
                nanPayload, reinterpret_cast<RegTensor<int32_t>&>(mx), int32_t(0x7f800000), active);
            Select(nanValues, byteMask, zeroInt, nanPayload);
            for (int i = 0; i < 5; i++) {
                DeInterleave(a, b, mx, mx);
                Max(mx, a, b, all);
                DeInterleave(nanA, nanB, nanValues, nanValues);
                Max(nanValues, nanA, nanB, all);
            }
            RegTensor<int32_t> exp, offset, newScale, scale127;
            Duplicate(offset, 127 + (OUTPUT == 0 ? 2 : OUTPUT == 1 ? 0 : OUTPUT == 2 ? 8 : 15));
            Duplicate(scale127, 127);
            ShiftRights(exp, reinterpret_cast<RegTensor<int32_t>&>(mx), int16_t(23), all);
            And(exp, exp, byteMask, all);
            Sub(newScale, oldScale, offset, all);
            Add(newScale, newScale, exp, all);
            Max(newScale, newScale, zeroInt, all);
            Min(newScale, newScale, max254, all);
            MaskReg zeroBlock, nanScale;
            CompareScalar<float, AscendC::CMPMODE::EQ>(zeroBlock, mx, 0.f, all);
            Select(newScale, oldScale, newScale, zeroBlock);
            Compare<int32_t, AscendC::CMPMODE::EQ>(nanScale, oldScale, byteMask, all);
            Select(newScale, byteMask, newScale, nanScale);
            Compare<int32_t, AscendC::CMPMODE::EQ>(nanPayload, nanValues, byteMask, all);
            Select(newScale, byteMask, newScale, nanPayload);
            RegTensor<uint8_t> tmp, unused;
            DeInterleave(
                tmp, unused, reinterpret_cast<RegTensor<uint8_t>&>(newScale),
                reinterpret_cast<RegTensor<uint8_t>&>(newScale));
            DeInterleave(tmp, unused, tmp, tmp);
            StoreUnAlign<uint8_t>(destScale, tmp, sw, (count + 31) / 32);
            RegTensor<int32_t> fac, sa, sb;
            Sub(fac, oldScale, newScale, all);
            Add(fac, fac, scale127, all);
            for (int i = 0; i < 5; i++) {
                Interleave(sa, sb, fac, fac);
                fac = sa;
            }
            ShiftLefts(fac, fac, int16_t(23), all);
            Mul(x, x, reinterpret_cast<RegTensor<float>&>(fac), active);
            if constexpr (OUTPUT < 2) {
                RegTensor<float> magnitude;
                Abs(magnitude, x, active);
                RegTensor<int32_t> code, one, zeroCode, sign;
                Duplicate(code, 0);
                Duplicate(zeroCode, 0);
                Duplicate(one, 1);
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(
                        greater, magnitude, OUTPUT == 1 ? 0.125f : 0.25f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(
                        greater, magnitude, OUTPUT == 1 ? 0.375f : 0.75f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(
                        greater, magnitude, OUTPUT == 1 ? 0.625f : 1.25f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(
                        greater, magnitude, OUTPUT == 1 ? 0.875f : 1.75f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(greater, magnitude, OUTPUT == 1 ? 1.125f : 2.5f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(greater, magnitude, OUTPUT == 1 ? 1.375f : 3.5f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(greater, magnitude, OUTPUT == 1 ? 1.625f : 5.0f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                ShiftRights(sign, reinterpret_cast<RegTensor<int32_t>&>(x), int16_t(28), active);
                RegTensor<int32_t> eight;
                Duplicate(eight, 8);
                And(sign, sign, eight, active);
                Or(code, code, sign, active);
                RegTensor<uint8_t> t0, t1, evens, odds;
                DeInterleave(
                    t0, t1, reinterpret_cast<RegTensor<uint8_t>&>(code), reinterpret_cast<RegTensor<uint8_t>&>(code));
                DeInterleave(t0, t1, t0, t0);
                DeInterleave(evens, odds, t0, t0);
                uint32_t nbytes = (count + 1) / 2;
                MaskReg byteMask = UpdateMask<uint8_t>(nbytes);
                ShiftLefts(odds, odds, int16_t(4), byteMask);
                Or(evens, evens, odds, byteMask);
                if (count & 1)
                    preserveFp4Tail(dst, evens, count);
                StoreUnAlign<uint8_t>(dst, evens, ow, (count + 1) / 2);
            } else {
                using Dest = typename std::conditional<OUTPUT == 2, fp8_e4m3fn_t, fp8_e5m2_t>::type;
                RegTensor<Dest> fp8;
                Cast<Dest, float, encodeTrait>(fp8, x, active);
                static constexpr CastTrait decodeTrait = {
                    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
                RegTensor<int32_t> nextBits, oneBit, selectedBits;
                Duplicate(oneBit, 1);
                Add(nextBits, reinterpret_cast<RegTensor<int32_t>&>(fp8), oneBit, active);
                RegTensor<float> nearest, next, nearestAbs, nextAbs, inputAbs, midpoint;
                Cast<float, Dest, decodeTrait>(nearest, fp8, active);
                Abs(nearestAbs, nearest, active);
                Abs(inputAbs, x, active);
                // The hardware cast rounds ties to even (CAST_RINT). CAST_ROUND asks for
                // ties away from zero, so for that mode -- and only that mode -- an exact
                // midpoint is nudged one code away. The directed modes below re-derive
                // their result from a magnitude comparison and so agree either way, while
                // CAST_NONE and CAST_RINT must keep the native result untouched.
                //
                // roundMode is the tla.cast enum: 0 CAST_ROUND, 1 CAST_FLOOR, 2 CAST_CEIL,
                // 3 CAST_TRUNC, 4 CAST_NONE, 5 CAST_RINT. CAST_ODD (6) is refused by the
                // frontend, so every value that reaches here is handled.
                if (roundMode == 0) {
                    Cast<float, Dest, decodeTrait>(next, reinterpret_cast<RegTensor<Dest>&>(nextBits), active);
                    Abs(nextAbs, next, active);
                    Add(midpoint, nearestAbs, nextAbs, active);
                    Muls(midpoint, midpoint, 0.5f, active);
                    MaskReg isTie, below;
                    Compare<float, AscendC::CMPMODE::EQ>(isTie, inputAbs, midpoint, active);
                    Compare<float, AscendC::CMPMODE::GT>(below, inputAbs, nearestAbs, active);
                    Select(selectedBits, nextBits, reinterpret_cast<RegTensor<int32_t>&>(fp8), isTie);
                    Select(
                        reinterpret_cast<RegTensor<int32_t>&>(fp8), selectedBits,
                        reinterpret_cast<RegTensor<int32_t>&>(fp8), below);
                }
                if (roundMode >= 1 && roundMode <= 3) {
                    Cast<float, Dest, decodeTrait>(nearest, fp8, active);
                    Abs(nearestAbs, nearest, active);
                    MaskReg larger, smaller, canIncrease, negative;
                    Compare<float, AscendC::CMPMODE::GT>(larger, nearestAbs, inputAbs, active);
                    Compare<float, AscendC::CMPMODE::GT>(smaller, inputAbs, nearestAbs, active);
                    CompareScalar<float, AscendC::CMPMODE::LT>(
                        canIncrease, nearestAbs, OUTPUT == 2 ? 448.f : 57344.f, active);
                    CompareScalar<int32_t, AscendC::CMPMODE::LT>(
                        negative, reinterpret_cast<RegTensor<int32_t>&>(x), 0, active);
                    RegTensor<int32_t> down, up, toward, away;
                    auto& bits = reinterpret_cast<RegTensor<int32_t>&>(fp8);
                    Sub(down, bits, oneBit, active);
                    Add(up, bits, oneBit, active);
                    Select(toward, down, bits, larger);
                    Select(away, up, bits, smaller);
                    Select(away, away, bits, canIncrease);
                    if (roundMode == 1)
                        Select(bits, away, toward, negative);
                    else if (roundMode == 2)
                        Select(bits, toward, away, negative);
                    else if (roundMode == 3)
                        bits = toward;
                }
                RegTensor<uint8_t> d0, d1;
                DeInterleave(
                    d0, d1, reinterpret_cast<RegTensor<uint8_t>&>(fp8), reinterpret_cast<RegTensor<uint8_t>&>(fp8));
                DeInterleave(d0, d1, d0, d0);
                StoreUnAlign<uint8_t>(dst, d0, ow, count);
            }
        }
        StoreUnAlignPost<uint8_t>(dst, ow, 0);
        StoreUnAlignPost<uint8_t>(destScale, sw, 0);
    }
}
#define MX_REQUANT(SRCNAME, SRC, DSTNAME, DST)                                                                         \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_tensor_requant_##SRCNAME##_##DSTNAME(                \
        uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint32_t r, uint32_t n, uint32_t as, uint32_t bs, uint32_t cs, \
        uint32_t ds, uint32_t roundMode)                                                                               \
    {                                                                                                                  \
        requantize<SRC, DST>(                                                                                          \
            reinterpret_cast<__ubuf__ uint8_t*>(a), reinterpret_cast<__ubuf__ uint8_t*>(b),                            \
            reinterpret_cast<__ubuf__ uint8_t*>(c), reinterpret_cast<__ubuf__ uint8_t*>(d), r, n, as, bs, cs, ds,      \
            roundMode, 0);                                                                                             \
    }

MX_REQUANT(e2m1, 0, e2m1, 0)
MX_REQUANT(e2m1, 0, e1m2, 1)
MX_REQUANT(e2m1, 0, e4m3, 2)
MX_REQUANT(e2m1, 0, e5m2, 3)
MX_REQUANT(e1m2, 1, e2m1, 0)
MX_REQUANT(e1m2, 1, e1m2, 1)
MX_REQUANT(e1m2, 1, e4m3, 2)
MX_REQUANT(e1m2, 1, e5m2, 3)
MX_REQUANT(e4m3, 2, e2m1, 0)
MX_REQUANT(e4m3, 2, e1m2, 1)
MX_REQUANT(e4m3, 2, e4m3, 2)
MX_REQUANT(e4m3, 2, e5m2, 3)
MX_REQUANT(e5m2, 3, e2m1, 0)
MX_REQUANT(e5m2, 3, e1m2, 1)
MX_REQUANT(e5m2, 3, e4m3, 2)
MX_REQUANT(e5m2, 3, e5m2, 3)

#undef MX_REQUANT

#define MX_REQUANT_MASKED(SRCNAME, SRC, DSTNAME, DST)                                                                  \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_tensor_requant_##SRCNAME##_##DSTNAME##_masked(       \
        uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint32_t r, uint32_t n, uint32_t as, uint32_t bs, uint32_t cs, \
        uint32_t ds, uint32_t roundMode, uint32_t maskLimit)                                                           \
    {                                                                                                                  \
        requantize<SRC, DST, true>(                                                                                    \
            reinterpret_cast<__ubuf__ uint8_t*>(a), reinterpret_cast<__ubuf__ uint8_t*>(b),                            \
            reinterpret_cast<__ubuf__ uint8_t*>(c), reinterpret_cast<__ubuf__ uint8_t*>(d), r, n, as, bs, cs, ds,      \
            roundMode, maskLimit);                                                                                     \
    }

MX_REQUANT_MASKED(e2m1, 0, e2m1, 0)
MX_REQUANT_MASKED(e2m1, 0, e1m2, 1)
MX_REQUANT_MASKED(e1m2, 1, e2m1, 0)
MX_REQUANT_MASKED(e1m2, 1, e1m2, 1)
MX_REQUANT_MASKED(e4m3, 2, e2m1, 0)
MX_REQUANT_MASKED(e4m3, 2, e1m2, 1)
MX_REQUANT_MASKED(e4m3, 2, e4m3, 2)
MX_REQUANT_MASKED(e4m3, 2, e5m2, 3)
MX_REQUANT_MASKED(e5m2, 3, e2m1, 0)
MX_REQUANT_MASKED(e5m2, 3, e1m2, 1)
MX_REQUANT_MASKED(e5m2, 3, e4m3, 2)
MX_REQUANT_MASKED(e5m2, 3, e5m2, 3)

#undef MX_REQUANT_MASKED

// ==========================================================================
// MX -> MX requantization across a group-size change  (was mx_tensor_regroup.cpp)
// ==========================================================================

// Requantize in the unscaled payload domain. Calculating the new E8M0
// exponent relative to the input exponent avoids overflowing a dense f32
// intermediate when the input block has a large scale.
template <int INPUT, int OUTPUT, bool MASK_PREFIX = false>
__simd_vf__ static void regroupQuant(
    __ubuf__ uint8_t* in, __ubuf__ uint8_t* scale, __ubuf__ uint8_t* out, __ubuf__ uint8_t* outScale, uint32_t rows,
    uint32_t cols, uint32_t ss, uint32_t qs, uint32_t ds, uint32_t os, uint32_t roundMode, uint32_t maskLimit)
{
    mem_bar(VST_VLD);
    static constexpr CastTrait decodeTrait = {
        RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr CastTrait encodeTrait = {
        RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    for (uint32_t row = 0; row < rows; row++) {
        __ubuf__ uint8_t* src = in + row * ss;
        __ubuf__ uint8_t* sc = scale + row * qs;
        __ubuf__ uint8_t* dst = out + row * ds;
        __ubuf__ uint8_t* destScale = outScale + row * os;
        UnalignRegForLoad il, sl;
        UnalignRegForStore ow, sw;
        LoadUnAlignPre(il, src);
        LoadUnAlignPre(sl, sc);
        for (uint32_t pos = 0; pos < cols; pos += 64) {
            uint32_t count = cols - pos > 64 ? 64 : cols - pos, n = count;
            MaskReg active = UpdateMask<float>(n), all = CreateMask<float, MaskPattern::ALL>();
            if constexpr (MASK_PREFIX) {
                uint32_t start = row * cols + pos;
                uint32_t remaining = start < maskLimit ? maskLimit - start : 0;
                MaskReg selected = UpdateMask<float>(remaining);
                MaskAnd(active, active, selected, all);
            }
            RegTensor<uint8_t> packed, p0, p1, scaleBytes;
            if constexpr (INPUT < 2) {
                uint32_t inputCount = (count + 1) / 2, nm = inputCount;
                MaskReg p = UpdateMask<uint8_t>(nm);
                ave_preg preg = *reinterpret_cast<ave_preg*>(&p);
                LoadUnAlign<uint8_t>(packed, il, src, inputCount);
                vector_u8 raw = *reinterpret_cast<vector_u8*>(&packed), lo, hi, first, second;
                lo = unpackFp4<false, INPUT == 1, 3, 7>(raw, preg);
                hi = unpackFp4<true, INPUT == 1, 3, 7>(raw, preg);
                vintlv(first, second, lo, hi);
                packed = *reinterpret_cast<RegTensor<uint8_t>*>(&first);
            } else {
                LoadUnAlign<uint8_t>(packed, il, src, count);
            }
            Interleave(p0, p1, packed, packed);
            Interleave(packed, p1, p0, p0);
            RegTensor<float> x;
            if constexpr (INPUT == 3)
                Cast<float, fp8_e5m2_t, decodeTrait>(x, reinterpret_cast<RegTensor<fp8_e5m2_t>&>(packed), active);
            else
                Cast<float, fp8_e4m3fn_t, decodeTrait>(x, reinterpret_cast<RegTensor<fp8_e4m3fn_t>&>(packed), active);
            LoadUnAlign<uint8_t>(scaleBytes, sl, sc, count);
            Interleave(p0, p1, scaleBytes, scaleBytes);
            Interleave(scaleBytes, p1, p0, p0);
            RegTensor<int32_t> oldScale, byteMask, zeroInt, max254, exp, candidate, offset, neg, scale127;
            Duplicate(byteMask, 255);
            Duplicate(zeroInt, 0);
            Duplicate(max254, 254);
            Duplicate(neg, -1000);
            Duplicate(scale127, 127);
            And(oldScale, reinterpret_cast<RegTensor<int32_t>&>(scaleBytes), byteMask, all);
            RegTensor<float> magnitude;
            Abs(magnitude, x, active);
            ShiftRights(exp, reinterpret_cast<RegTensor<int32_t>&>(magnitude), int16_t(23), all);
            And(exp, exp, byteMask, all);
            Duplicate(offset, 127 + (OUTPUT == 0 ? 2 : OUTPUT == 1 ? 0 : OUTPUT == 2 ? 8 : 15));
            Add(candidate, oldScale, exp, all);
            Sub(candidate, candidate, offset, all);
            Max(candidate, candidate, zeroInt, all);
            Min(candidate, candidate, max254, all);
            MaskReg isZero, nanScale;
            CompareScalar<float, AscendC::CMPMODE::EQ>(isZero, magnitude, 0.f, active);
            Select(candidate, neg, candidate, isZero);
            Compare<int32_t, AscendC::CMPMODE::EQ>(nanScale, oldScale, byteMask, active);
            Select(candidate, byteMask, candidate, nanScale);
            MaskReg nanPayload;
            CompareScalar<int32_t, AscendC::CMPMODE::GT>(
                nanPayload, reinterpret_cast<RegTensor<int32_t>&>(magnitude), int32_t(0x7f800000), active);
            Select(candidate, byteMask, candidate, nanPayload);
            Select(candidate, candidate, neg, active);
            RegTensor<int32_t> a, b;
            for (int i = 0; i < 5; ++i) {
                DeInterleave(a, b, candidate, candidate);
                Max(candidate, a, b, all);
            }
            MaskReg empty;
            CompareScalar<int32_t, AscendC::CMPMODE::LT>(empty, candidate, 0, all);
            Select(candidate, scale127, candidate, empty);
            RegTensor<uint8_t> tmp, unused;
            DeInterleave(
                tmp, unused, reinterpret_cast<RegTensor<uint8_t>&>(candidate),
                reinterpret_cast<RegTensor<uint8_t>&>(candidate));
            DeInterleave(tmp, unused, tmp, tmp);
            StoreUnAlign<uint8_t>(destScale, tmp, sw, 1);
            for (int i = 0; i < 5; ++i) {
                Interleave(a, b, candidate, candidate);
                candidate = a;
            }
            RegTensor<int32_t> fac;
            Sub(fac, oldScale, candidate, all);
            Add(fac, fac, scale127, all);
            Max(fac, fac, zeroInt, all);
            Min(fac, fac, max254, all);
            CompareScalar<int32_t, AscendC::CMPMODE::EQ>(nanScale, candidate, 255, all);
            Select(fac, scale127, fac, nanScale);
            ShiftLefts(fac, fac, int16_t(23), all);
            Mul(x, x, reinterpret_cast<RegTensor<float>&>(fac), active);
            if constexpr (OUTPUT < 2) {
                RegTensor<float> magnitude;
                Abs(magnitude, x, active);
                RegTensor<int32_t> code, one, zeroCode, sign;
                Duplicate(code, 0);
                Duplicate(zeroCode, 0);
                Duplicate(one, 1);
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(
                        greater, magnitude, OUTPUT == 1 ? 0.125f : 0.25f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(
                        greater, magnitude, OUTPUT == 1 ? 0.375f : 0.75f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(
                        greater, magnitude, OUTPUT == 1 ? 0.625f : 1.25f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(
                        greater, magnitude, OUTPUT == 1 ? 0.875f : 1.75f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(greater, magnitude, OUTPUT == 1 ? 1.125f : 2.5f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(greater, magnitude, OUTPUT == 1 ? 1.375f : 3.5f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                {
                    MaskReg greater;
                    RegTensor<int32_t> inc;
                    CompareScalar<float, AscendC::CMPMODE::GE>(greater, magnitude, OUTPUT == 1 ? 1.625f : 5.0f, active);
                    Select(inc, one, zeroCode, greater);
                    Add(code, code, inc, active);
                }
                ShiftRights(sign, reinterpret_cast<RegTensor<int32_t>&>(x), int16_t(28), active);
                RegTensor<int32_t> eight;
                Duplicate(eight, 8);
                And(sign, sign, eight, active);
                Or(code, code, sign, active);
                RegTensor<uint8_t> t0, t1, evens, odds;
                DeInterleave(
                    t0, t1, reinterpret_cast<RegTensor<uint8_t>&>(code), reinterpret_cast<RegTensor<uint8_t>&>(code));
                DeInterleave(t0, t1, t0, t0);
                DeInterleave(evens, odds, t0, t0);
                uint32_t nbytes = (count + 1) / 2;
                MaskReg byteMask = UpdateMask<uint8_t>(nbytes);
                ShiftLefts(odds, odds, int16_t(4), byteMask);
                Or(evens, evens, odds, byteMask);
                if (count & 1)
                    preserveFp4Tail(dst, evens, count);
                StoreUnAlign<uint8_t>(dst, evens, ow, (count + 1) / 2);
            } else {
                using Dest = typename std::conditional<OUTPUT == 2, fp8_e4m3fn_t, fp8_e5m2_t>::type;
                RegTensor<Dest> fp8;
                Cast<Dest, float, encodeTrait>(fp8, x, active);
                static constexpr CastTrait decodeTrait = {
                    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
                RegTensor<int32_t> nextBits, oneBit, selectedBits;
                Duplicate(oneBit, 1);
                Add(nextBits, reinterpret_cast<RegTensor<int32_t>&>(fp8), oneBit, active);
                RegTensor<float> nearest, next, nearestAbs, nextAbs, inputAbs, midpoint;
                Cast<float, Dest, decodeTrait>(nearest, fp8, active);
                Abs(nearestAbs, nearest, active);
                Abs(inputAbs, x, active);
                // The hardware cast rounds ties to even (CAST_RINT). CAST_ROUND asks for
                // ties away from zero, so for that mode -- and only that mode -- an exact
                // midpoint is nudged one code away. The directed modes below re-derive
                // their result from a magnitude comparison and so agree either way, while
                // CAST_NONE and CAST_RINT must keep the native result untouched.
                //
                // roundMode is the tla.cast enum: 0 CAST_ROUND, 1 CAST_FLOOR, 2 CAST_CEIL,
                // 3 CAST_TRUNC, 4 CAST_NONE, 5 CAST_RINT. CAST_ODD (6) is refused by the
                // frontend, so every value that reaches here is handled.
                if (roundMode == 0) {
                    Cast<float, Dest, decodeTrait>(next, reinterpret_cast<RegTensor<Dest>&>(nextBits), active);
                    Abs(nextAbs, next, active);
                    Add(midpoint, nearestAbs, nextAbs, active);
                    Muls(midpoint, midpoint, 0.5f, active);
                    MaskReg isTie, below;
                    Compare<float, AscendC::CMPMODE::EQ>(isTie, inputAbs, midpoint, active);
                    Compare<float, AscendC::CMPMODE::GT>(below, inputAbs, nearestAbs, active);
                    Select(selectedBits, nextBits, reinterpret_cast<RegTensor<int32_t>&>(fp8), isTie);
                    Select(
                        reinterpret_cast<RegTensor<int32_t>&>(fp8), selectedBits,
                        reinterpret_cast<RegTensor<int32_t>&>(fp8), below);
                }
                if (roundMode >= 1 && roundMode <= 3) {
                    Cast<float, Dest, decodeTrait>(nearest, fp8, active);
                    Abs(nearestAbs, nearest, active);
                    MaskReg larger, smaller, canIncrease, negative;
                    Compare<float, AscendC::CMPMODE::GT>(larger, nearestAbs, inputAbs, active);
                    Compare<float, AscendC::CMPMODE::GT>(smaller, inputAbs, nearestAbs, active);
                    CompareScalar<float, AscendC::CMPMODE::LT>(
                        canIncrease, nearestAbs, OUTPUT == 2 ? 448.f : 57344.f, active);
                    CompareScalar<int32_t, AscendC::CMPMODE::LT>(
                        negative, reinterpret_cast<RegTensor<int32_t>&>(x), 0, active);
                    RegTensor<int32_t> down, up, toward, away;
                    auto& bits = reinterpret_cast<RegTensor<int32_t>&>(fp8);
                    Sub(down, bits, oneBit, active);
                    Add(up, bits, oneBit, active);
                    Select(toward, down, bits, larger);
                    Select(away, up, bits, smaller);
                    Select(away, away, bits, canIncrease);
                    if (roundMode == 1)
                        Select(bits, away, toward, negative);
                    else if (roundMode == 2)
                        Select(bits, toward, away, negative);
                    else if (roundMode == 3)
                        bits = toward;
                }
                RegTensor<uint8_t> d0, d1;
                DeInterleave(
                    d0, d1, reinterpret_cast<RegTensor<uint8_t>&>(fp8), reinterpret_cast<RegTensor<uint8_t>&>(fp8));
                DeInterleave(d0, d1, d0, d0);
                StoreUnAlign<uint8_t>(dst, d0, ow, count);
            }
        }
        StoreUnAlignPost<uint8_t>(dst, ow, 0);
        StoreUnAlignPost<uint8_t>(destScale, sw, 0);
    }
    mem_bar(VLD_VST);
}

__simd_vf__ static void writeRegroupScales(
    __ubuf__ uint8_t* temp, __ubuf__ uint8_t* dst, uint32_t rows, uint32_t groups, uint32_t rowStride, uint32_t step)
{
    mem_bar(VST_VLD);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t group = 0; group < groups; ++group) {
            __ubuf__ uint8_t* from = temp + row * groups + group;
            __ubuf__ uint8_t* to = dst + row * rowStride + group * step;
            UnalignRegForLoad load;
            UnalignRegForStore store;
            RegTensor<uint8_t> value;
            LoadUnAlignPre(load, from);
            LoadUnAlign(value, load, from, 1);
            StoreUnAlign(to, value, store, 1);
            StoreUnAlignPost(to, store, 0);
        }
    }
}

#define DECL_STAGE(K) extern "C"[aicore] void tla_mx_stage_##K(uint64_t, uint64_t, uint32_t, uint32_t, uint32_t)
DECL_STAGE(gather_b8);
DECL_STAGE(gather_fp4);
DECL_STAGE(scatter_b8);
DECL_STAGE(scatter_fp4);
#define REGROUP(S, SI, SK, D, DI, DK)                                                                                 \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_regroup_##S##_##D(                                  \
        uint64_t src, uint64_t ss, uint64_t dst, uint64_t ds, uint64_t scratch, uint32_t rows, uint32_t cols,         \
        uint32_t sr, uint32_t sx, uint32_t so, uint32_t ssGroup, uint32_t ssStep, uint32_t dr, uint32_t dx,           \
        uint32_t doff, uint32_t dsRow, uint32_t dsStep, uint32_t mode, uint32_t limit)                                \
    {                                                                                                                 \
        for (uint32_t row = 0; row < rows; ++row)                                                                     \
            for (uint32_t pos = 0; pos < cols; pos += 32) {                                                           \
                uint32_t count = cols - pos < 32 ? cols - pos : 32;                                                   \
                uint32_t begin = row * cols + pos;                                                                    \
                uint32_t active = begin < limit ? limit - begin : 0;                                                  \
                if (active > count)                                                                                   \
                    active = count;                                                                                   \
                tla_mx_stage_gather_##SK(src, scratch, sx, so + row * sr + pos * sx, count);                          \
                tla_mx_stage_gather_b8(ss, scratch + 512, ssStep, (row / 32) * ssGroup + pos * ssStep, count);        \
                regroupQuant<SI, DI, true>(                                                                           \
                    reinterpret_cast<__ubuf__ uint8_t*>(scratch), reinterpret_cast<__ubuf__ uint8_t*>(scratch + 512), \
                    reinterpret_cast<__ubuf__ uint8_t*>(scratch + 256),                                               \
                    reinterpret_cast<__ubuf__ uint8_t*>(scratch + 768 + row * ((cols + 31) / 32) + pos / 32), 1,      \
                    count, count, count, count, 1, mode, active);                                                     \
                tla_mx_stage_scatter_##DK(scratch + 256, dst, dx, doff + row * dr + pos * dx, count);                 \
            }                                                                                                         \
        writeRegroupScales(                                                                                           \
            reinterpret_cast<__ubuf__ uint8_t*>(scratch + 768), reinterpret_cast<__ubuf__ uint8_t*>(ds), rows,        \
            (cols + 31) / 32, dsRow, dsStep);                                                                         \
    }
// clang-format off
#define ALL_D(S, SI, SK)             \
    REGROUP(S, SI, SK, e2m1, 0, fp4) \
    REGROUP(S, SI, SK, e1m2, 1, fp4) REGROUP(S, SI, SK, e4m3, 2, b8) REGROUP(S, SI, SK, e5m2, 3, b8)
ALL_D(e2m1, 0, fp4)
ALL_D(e1m2, 1, fp4)
ALL_D(e4m3, 2, b8)
ALL_D(e5m2, 3, b8)

#define EXPORT_GROUP(S, SI, D, DI)                                                                                     \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_group_regroup_##S##_##D(                             \
        uint64_t src, uint64_t scale, uint64_t dst, uint64_t dstScale, uint32_t count, uint32_t mode, uint32_t active) \
    {                                                                                                                  \
        regroupQuant<SI, DI, true>(                                                                                    \
            reinterpret_cast<__ubuf__ uint8_t*>(src), reinterpret_cast<__ubuf__ uint8_t*>(scale),                      \
            reinterpret_cast<__ubuf__ uint8_t*>(dst), reinterpret_cast<__ubuf__ uint8_t*>(dstScale), 1, count, count,  \
            count, count, 1, mode, active);                                                                            \
    }
#define EXPORT_ALL(S, SI) \
    EXPORT_GROUP(S, SI, e2m1, 0) EXPORT_GROUP(S, SI, e1m2, 1) EXPORT_GROUP(S, SI, e4m3, 2) EXPORT_GROUP(S, SI, e5m2, 3)
    EXPORT_ALL(e2m1, 0) EXPORT_ALL(e1m2, 1) EXPORT_ALL(e4m3, 2) EXPORT_ALL(e5m2, 3)
// clang-format on

#undef DECL_STAGE
#undef REGROUP
#undef ALL_D
#undef EXPORT_GROUP
#undef EXPORT_ALL

    // ==========================================================================
    // Quantization from a strided dense source, gathered per group  (was mx_tensor_gather_quant.cpp)
    // ==========================================================================

    template <typename T, typename I>
    __simd_vf__ static void gatherGroup(__ubuf__ T* src, __ubuf__ T* scratch, uint32_t step, uint32_t count)
{
    mem_bar(VLD_VST);
    uint32_t n = count;
    MaskReg mask = UpdateMask<T>(n);
    RegTensor<I> index;
    RegTensor<T> data;
    using Signed = typename std::conditional<sizeof(I) == 4, int32_t, int16_t>::type;
    Arange(reinterpret_cast<RegTensor<Signed>&>(index), Signed(0));
    Muls(index, index, static_cast<I>(step), mask);
    Gather(data, src, index, mask);
    StoreAlign(scratch, data, mask);
    mem_bar(VST_VLD);
}
#define GATHER_QUANT(S, T, I, B, D, P)                                                                                \
    extern "C"[aicore] void tla_mx_tensor_quant_##S##_##D##_masked(                                                   \
        uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);          \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_gather_quant_##S##_##D(                             \
        uint64_t src, uint64_t dst, uint64_t scale, uint64_t scratch, uint32_t rows, uint32_t cols, uint32_t srcRow,  \
        uint32_t srcStep, uint32_t dstRow, uint32_t scaleRow, uint32_t scaleStep, uint32_t roundMode,                 \
        uint32_t maskLimit)                                                                                           \
    {                                                                                                                 \
        for (uint32_t row = 0; row < rows; ++row) {                                                                   \
            for (uint32_t pos = 0; pos < cols; pos += 32) {                                                           \
                uint32_t count = cols - pos < 32 ? cols - pos : 32;                                                   \
                uint32_t start = row * cols + pos;                                                                    \
                uint32_t active = start < maskLimit ? maskLimit - start : 0;                                          \
                if (active > count)                                                                                   \
                    active = count;                                                                                   \
                gatherGroup<T, I>(                                                                                    \
                    reinterpret_cast<__ubuf__ T*>(src + row * srcRow + pos * srcStep * B),                            \
                    reinterpret_cast<__ubuf__ T*>(scratch), srcStep, count);                                          \
                tla_mx_tensor_quant_##S##_##D##_masked(                                                               \
                    scratch, dst + row * dstRow + pos / P, scale + row * scaleRow + (pos / 32) * scaleStep, 1, count, \
                    count, count, 1, roundMode, active);                                                              \
            }                                                                                                         \
        }                                                                                                             \
    }
#define ALL_FORMATS(S, T, I, B)       \
    GATHER_QUANT(S, T, I, B, e2m1, 2) \
    GATHER_QUANT(S, T, I, B, e1m2, 2) GATHER_QUANT(S, T, I, B, e4m3, 1) GATHER_QUANT(S, T, I, B, e5m2, 1)
ALL_FORMATS(f32, uint32_t, uint32_t, 4)
ALL_FORMATS(f16, uint16_t, uint16_t, 2)
ALL_FORMATS(bf16, uint16_t, uint16_t, 2)

extern "C"[aicore] __attribute__((always_inline)) void tla_mx_stage_gather_b32(
    uint64_t src, uint64_t scratch, uint32_t step, uint32_t offset, uint32_t count)
{
    gatherGroup<uint32_t, uint32_t>(
        reinterpret_cast<__ubuf__ uint32_t*>(src) + offset, reinterpret_cast<__ubuf__ uint32_t*>(scratch), step, count);
}
extern "C"[aicore] __attribute__((always_inline)) void tla_mx_stage_gather_b16(
    uint64_t src, uint64_t scratch, uint32_t step, uint32_t offset, uint32_t count)
{
    gatherGroup<uint16_t, uint16_t>(
        reinterpret_cast<__ubuf__ uint16_t*>(src) + offset, reinterpret_cast<__ubuf__ uint16_t*>(scratch), step, count);
}

#undef GATHER_QUANT
#undef ALL_FORMATS

// ==========================================================================
// Requantization from a strided MX source, gathered per group  (was mx_tensor_gather_requant.cpp)
// ==========================================================================

__simd_vf__ static void gatherBytes(__ubuf__ uint8_t* src, __ubuf__ uint8_t* scratch, uint32_t step, uint32_t count)
{
    mem_bar(VLD_VST);
    uint32_t n = count;
    MaskReg mask = UpdateMask<uint16_t>(n);
    RegTensor<uint16_t> index, values;
    Arange(reinterpret_cast<RegTensor<int16_t>&>(index), int16_t(0));
    Muls(index, index, static_cast<uint16_t>(step), mask);
    Gather(values, src, index, mask);
    Pack(reinterpret_cast<RegTensor<uint8_t>&>(values), values);
    MaskPack(mask, mask);
    StoreAlign(scratch, reinterpret_cast<RegTensor<uint8_t>&>(values), mask);
    mem_bar(VST_VLD);
}
__simd_vf__ static void gatherNibbles(
    __ubuf__ uint8_t* src, __ubuf__ uint8_t* scratch, uint32_t step, uint32_t start, uint32_t count)
{
    mem_bar(VLD_VST);
    uint32_t n = count;
    MaskReg mask = UpdateMask<uint16_t>(n);
    RegTensor<uint16_t> index, parity, one, values, nibbleMask;
    Arange(reinterpret_cast<RegTensor<int16_t>&>(index), int16_t(0));
    Muls(index, index, static_cast<uint16_t>(step), mask);
    Adds(index, index, static_cast<uint16_t>(start), mask);
    Duplicate(one, uint16_t(1));
    And(parity, index, one, mask);
    ShiftRights(index, index, int16_t(1), mask);
    Gather(values, src, index, mask);
    RegTensor<uint16_t> high;
    ShiftRights(high, values, int16_t(4), mask);
    MaskReg even;
    CompareScalar<int16_t, AscendC::CMPMODE::EQ>(even, reinterpret_cast<RegTensor<int16_t>&>(parity), int16_t(0), mask);
    Select(values, values, high, even);
    Duplicate(nibbleMask, uint16_t(15));
    And(values, values, nibbleMask, mask);
    Pack(reinterpret_cast<RegTensor<uint8_t>&>(values), values);
    RegTensor<uint8_t> lo, odd, packed;
    auto& bytes = reinterpret_cast<RegTensor<uint8_t>&>(values);
    DeInterleave(lo, odd, bytes, bytes);
    uint32_t bytesCount = (count + 1) / 2;
    MaskReg outMask = UpdateMask<uint8_t>(bytesCount);
    ShiftLefts(odd, odd, int16_t(4), outMask);
    Or(packed, lo, odd, outMask);
    StoreAlign(scratch, packed, outMask);
    mem_bar(VST_VLD);
}

#define GATHER_REQUANT(S, D, P)                                                                                        \
    extern "C"[aicore] void tla_mx_tensor_requant_##S##_##D##_masked(                                                  \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,  \
        uint32_t);                                                                                                     \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_gather_requant_##S##_##D(                            \
        uint64_t src, uint64_t srcScale, uint64_t dst, uint64_t dstScale, uint64_t scratch, uint32_t rows,             \
        uint32_t cols, uint32_t srcRow, uint32_t srcStep, uint32_t srcScaleRow, uint32_t srcScaleStep,                 \
        uint32_t dstRow, uint32_t dstScaleRow, uint32_t dstScaleStep, uint32_t roundMode, uint32_t maskLimit)          \
    {                                                                                                                  \
        for (uint32_t row = 0; row < rows; ++row) {                                                                    \
            for (uint32_t pos = 0; pos < cols; pos += 32) {                                                            \
                uint32_t count = cols - pos < 32 ? cols - pos : 32;                                                    \
                uint32_t start = row * cols + pos;                                                                     \
                uint32_t active = start < maskLimit ? maskLimit - start : 0;                                           \
                if (active > count)                                                                                    \
                    active = count;                                                                                    \
                gatherBytes(                                                                                           \
                    reinterpret_cast<__ubuf__ uint8_t*>(src + row * srcRow + pos * srcStep),                           \
                    reinterpret_cast<__ubuf__ uint8_t*>(scratch), srcStep, count);                                     \
                tla_mx_tensor_requant_##S##_##D##_masked(                                                              \
                    scratch, srcScale + row * srcScaleRow + (pos / 32) * srcScaleStep, dst + row * dstRow + pos / P,   \
                    dstScale + row * dstScaleRow + (pos / 32) * dstScaleStep, 1, count, count, 1, count, 1, roundMode, \
                    active);                                                                                           \
            }                                                                                                          \
        }                                                                                                              \
    }
#define ALL_MX(S) \
    GATHER_REQUANT(S, e2m1, 2) GATHER_REQUANT(S, e1m2, 2) GATHER_REQUANT(S, e4m3, 1) GATHER_REQUANT(S, e5m2, 1)
ALL_MX(e4m3)
ALL_MX(e5m2)

#define GATHER_FP4(S, D, P, WIDTH)                                                                                     \
    extern "C"[aicore] void tla_mx_tensor_requant_##S##_##D##_masked(                                                  \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,  \
        uint32_t);                                                                                                     \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_gather_fp4_##S##_##D(                                \
        uint64_t src, uint64_t srcScale, uint64_t dst, uint64_t dstScale, uint64_t scratch, uint32_t rows,             \
        uint32_t cols, uint32_t srcRow, uint32_t srcStep, uint32_t srcOffset, uint32_t srcScaleRow,                    \
        uint32_t srcScaleStep, uint32_t dstRow, uint32_t dstScaleRow, uint32_t dstScaleStep, uint32_t roundMode,       \
        uint32_t maskLimit)                                                                                            \
    {                                                                                                                  \
        for (uint32_t row = 0; row < rows; ++row) {                                                                    \
            for (uint32_t pos = 0; pos < cols; pos += 32) {                                                            \
                uint32_t count = cols - pos < 32 ? cols - pos : 32;                                                    \
                uint32_t begin = row * cols + pos;                                                                     \
                uint32_t active = begin < maskLimit ? maskLimit - begin : 0;                                           \
                if (active > count)                                                                                    \
                    active = count;                                                                                    \
                uint32_t logical = srcOffset + row * srcRow + pos * srcStep;                                           \
                gatherNibbles(                                                                                         \
                    reinterpret_cast<__ubuf__ uint8_t*>(src + logical / 2),                                            \
                    reinterpret_cast<__ubuf__ uint8_t*>(scratch), srcStep, logical % 2, count);                        \
                tla_mx_tensor_requant_##S##_##D##_masked(                                                              \
                    scratch, srcScale + row * srcScaleRow + (pos / 32) * srcScaleStep, dst + row * dstRow + pos / P,   \
                    dstScale + row * dstScaleRow + (pos / 32) * dstScaleStep, 1, count, count, 1, count, 1, roundMode, \
                    active);                                                                                           \
            }                                                                                                          \
        }                                                                                                              \
    }
GATHER_FP4(e2m1, e2m1, 2, 4)
GATHER_FP4(e2m1, e1m2, 2, 4)
GATHER_FP4(e1m2, e2m1, 2, 4)
GATHER_FP4(e1m2, e1m2, 2, 4)

#define GATHER_FP4_WIDEN(S, D)                                                                                         \
    extern "C"[aicore] void tla_mx_tensor_##S##_##D##_masked(                                                          \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t); \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_gather_fp4_##S##_##D(                                \
        uint64_t src, uint64_t srcScale, uint64_t dst, uint64_t dstScale, uint64_t scratch, uint32_t rows,             \
        uint32_t cols, uint32_t srcRow, uint32_t srcStep, uint32_t srcOffset, uint32_t srcScaleRow,                    \
        uint32_t srcScaleStep, uint32_t dstRow, uint32_t dstScaleRow, uint32_t dstScaleStep, uint32_t roundMode,       \
        uint32_t maskLimit)                                                                                            \
    {                                                                                                                  \
        for (uint32_t row = 0; row < rows; ++row) {                                                                    \
            for (uint32_t pos = 0; pos < cols; pos += 32) {                                                            \
                uint32_t count = cols - pos < 32 ? cols - pos : 32;                                                    \
                uint32_t begin = row * cols + pos;                                                                     \
                uint32_t active = begin < maskLimit ? maskLimit - begin : 0;                                           \
                if (active > count)                                                                                    \
                    active = count;                                                                                    \
                uint32_t logical = srcOffset + row * srcRow + pos * srcStep;                                           \
                gatherNibbles(                                                                                         \
                    reinterpret_cast<__ubuf__ uint8_t*>(src + logical / 2),                                            \
                    reinterpret_cast<__ubuf__ uint8_t*>(scratch), srcStep, logical % 2, count);                        \
                tla_mx_tensor_##S##_##D##_masked(                                                                      \
                    scratch, dst + row * dstRow + pos, srcScale + row * srcScaleRow + (pos / 32) * srcScaleStep,       \
                    dstScale + row * dstScaleRow + (pos / 32) * dstScaleStep, 1, count, count, count, 1, 1, active);   \
            }                                                                                                          \
        }                                                                                                              \
    }
GATHER_FP4_WIDEN(e2m1, e4m3)
GATHER_FP4_WIDEN(e2m1, e5m2)
GATHER_FP4_WIDEN(e1m2, e4m3)
GATHER_FP4_WIDEN(e1m2, e5m2)

extern "C"[aicore] __attribute__((always_inline)) void tla_mx_stage_gather_b8(
    uint64_t src, uint64_t scratch, uint32_t step, uint32_t offset, uint32_t count)
{
    gatherBytes(
        reinterpret_cast<__ubuf__ uint8_t*>(src + offset), reinterpret_cast<__ubuf__ uint8_t*>(scratch), step, count);
}
extern "C"[aicore] __attribute__((always_inline)) void tla_mx_stage_gather_fp4(
    uint64_t src, uint64_t scratch, uint32_t step, uint32_t offset, uint32_t count)
{
    gatherNibbles(
        reinterpret_cast<__ubuf__ uint8_t*>(src + offset / 2), reinterpret_cast<__ubuf__ uint8_t*>(scratch), step,
        offset % 2, count);
}

#undef GATHER_REQUANT
#undef ALL_MX
#undef GATHER_FP4
#undef GATHER_FP4_WIDEN

// ==========================================================================
// Dequantization to a strided dense destination  (was mx_tensor_scatter_dequant.cpp)
// ==========================================================================

template <typename T, typename I>
__simd_vf__ static void scatterGroup(__ubuf__ T* scratch, __ubuf__ T* dst, uint32_t step, uint32_t count)
{
    mem_bar(VST_VLD);
    uint32_t n = count;
    MaskReg mask = UpdateMask<T>(n);
    RegTensor<I> index;
    RegTensor<T> data;
    using Signed = typename std::conditional<sizeof(I) == 4, int32_t, int16_t>::type;
    Arange(reinterpret_cast<RegTensor<Signed>&>(index), Signed(0));
    Muls(index, index, static_cast<I>(step), mask);
    LoadAlign(data, scratch);
    Scatter(dst, data, index, mask);
    mem_bar(VLD_VST);
}
#define SCATTER_DEQUANT(S, P, D, T, I, B)                                                                             \
    extern "C"[aicore] void tla_mx_tensor_dequant_##S##_##D##_masked(                                                 \
        uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);          \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_scatter_dequant_##S##_##D(                          \
        uint64_t src, uint64_t scale, uint64_t dst, uint64_t scratch, uint32_t rows, uint32_t cols, uint32_t srcRow,  \
        uint32_t scaleRow, uint32_t scaleStep, uint32_t dstRow, uint32_t dstStep, uint32_t roundMode,                 \
        uint32_t maskLimit)                                                                                           \
    {                                                                                                                 \
        for (uint32_t row = 0; row < rows; ++row) {                                                                   \
            for (uint32_t pos = 0; pos < cols; pos += 32) {                                                           \
                uint32_t count = cols - pos < 32 ? cols - pos : 32;                                                   \
                uint32_t start = row * cols + pos;                                                                    \
                uint32_t active = start < maskLimit ? maskLimit - start : 0;                                          \
                if (active > count)                                                                                   \
                    active = count;                                                                                   \
                tla_mx_tensor_dequant_##S##_##D##_masked(                                                             \
                    src + row * srcRow + pos / P, scale + row * scaleRow + (pos / 32) * scaleStep, scratch, 1, count, \
                    count, 1, count, roundMode, active);                                                              \
                scatterGroup<T, I>(                                                                                   \
                    reinterpret_cast<__ubuf__ T*>(scratch),                                                           \
                    reinterpret_cast<__ubuf__ T*>(dst + row * dstRow + pos * dstStep * B), dstStep, count);           \
            }                                                                                                         \
        }                                                                                                             \
    }
#define ALL_DENSE(S, P)                               \
    SCATTER_DEQUANT(S, P, f32, uint32_t, uint32_t, 4) \
    SCATTER_DEQUANT(S, P, f16, uint16_t, uint16_t, 2) SCATTER_DEQUANT(S, P, bf16, uint16_t, uint16_t, 2)
ALL_DENSE(e2m1, 2)
ALL_DENSE(e1m2, 2)
ALL_DENSE(e4m3, 1)
ALL_DENSE(e5m2, 1)

extern "C"[aicore] __attribute__((always_inline)) void tla_mx_stage_scatter_b32(
    uint64_t scratch, uint64_t dst, uint32_t step, uint32_t offset, uint32_t count)
{
    scatterGroup<uint32_t, uint32_t>(
        reinterpret_cast<__ubuf__ uint32_t*>(scratch), reinterpret_cast<__ubuf__ uint32_t*>(dst) + offset, step, count);
}
extern "C"[aicore] __attribute__((always_inline)) void tla_mx_stage_scatter_b16(
    uint64_t scratch, uint64_t dst, uint32_t step, uint32_t offset, uint32_t count)
{
    scatterGroup<uint16_t, uint16_t>(
        reinterpret_cast<__ubuf__ uint16_t*>(scratch), reinterpret_cast<__ubuf__ uint16_t*>(dst) + offset, step, count);
}

#undef SCATTER_DEQUANT
#undef ALL_DENSE

// ==========================================================================
// Requantization to a strided MX destination  (was mx_tensor_scatter_requant.cpp)
// ==========================================================================

__simd_vf__ static void scatterBytes(__ubuf__ uint8_t* scratch, __ubuf__ uint8_t* dst, uint32_t step, uint32_t count)
{
    mem_bar(VST_VLD);
    uint32_t n = count;
    MaskReg mask = UpdateMask<uint16_t>(n);
    RegTensor<uint16_t> index;
    RegTensor<uint8_t> raw, spread, unused;
    Arange(reinterpret_cast<RegTensor<int16_t>&>(index), int16_t(0));
    Muls(index, index, static_cast<uint16_t>(step), mask);
    LoadAlign(raw, scratch);
    Interleave(spread, unused, raw, raw);
    Scatter(dst, spread, index, mask);
    mem_bar(VLD_VST);
}
__simd_vf__ static void scatterNibbles(
    __ubuf__ uint8_t* scratch, __ubuf__ uint8_t* dst, uint32_t step, uint32_t start, uint32_t count)
{
    mem_bar(VST_VLD);
    uint32_t n = count;
    MaskReg mask = UpdateMask<uint16_t>(n);
    RegTensor<uint16_t> lane, dstIndex, srcIndex, srcParity, dstParity, one, value, old, high, bits;
    Arange(reinterpret_cast<RegTensor<int16_t>&>(lane), int16_t(0));
    Duplicate(one, uint16_t(1));
    And(srcParity, lane, one, mask);
    ShiftRights(srcIndex, lane, int16_t(1), mask);
    Gather(value, scratch, srcIndex, mask);
    ShiftRights(high, value, int16_t(4), mask);
    MaskReg srcEven;
    CompareScalar<int16_t, AscendC::CMPMODE::EQ>(
        srcEven, reinterpret_cast<RegTensor<int16_t>&>(srcParity), int16_t(0), mask);
    Select(value, value, high, srcEven);
    Duplicate(bits, uint16_t(15));
    And(value, value, bits, mask);
    Muls(dstIndex, lane, static_cast<uint16_t>(step), mask);
    Adds(dstIndex, dstIndex, static_cast<uint16_t>(start), mask);
    And(dstParity, dstIndex, one, mask);
    ShiftRights(dstIndex, dstIndex, int16_t(1), mask);
    Gather(old, dst, dstIndex, mask);
    ShiftLefts(high, value, int16_t(4), mask);
    MaskReg dstEven;
    CompareScalar<int16_t, AscendC::CMPMODE::EQ>(
        dstEven, reinterpret_cast<RegTensor<int16_t>&>(dstParity), int16_t(0), mask);
    Select(value, value, high, dstEven);
    RegTensor<uint16_t> lowMask, highMask, keep;
    Duplicate(lowMask, uint16_t(15));
    Duplicate(highMask, uint16_t(240));
    Select(keep, highMask, lowMask, dstEven);
    And(old, old, keep, mask);
    Or(value, value, old, mask);
    Scatter(dst, reinterpret_cast<RegTensor<uint8_t>&>(value), dstIndex, mask);
    mem_bar(VLD_VST);
}

__simd_vf__ static void copyNibbles(__ubuf__ uint8_t* scratch, __ubuf__ uint8_t* dst, uint32_t count)
{
    mem_bar(VST_VLD);
    UnalignRegForLoad in;
    UnalignRegForStore out;
    LoadUnAlignPre(in, scratch);
    RegTensor<uint8_t> values;
    LoadUnAlign(values, in, scratch, (count + 1) / 2);
    if (count & 1)
        preserveFp4Tail(dst, values, count);
    StoreUnAlign(dst, values, out, (count + 1) / 2);
    StoreUnAlignPost(dst, out, 0);
    mem_bar(VLD_VST);
}

#define SCATTER_MX(S, SP, D)                                                                                           \
    extern "C"[aicore] void tla_mx_tensor_requant_##S##_##D##_masked(                                                  \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,  \
        uint32_t);                                                                                                     \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_scatter_mx_##S##_##D(                                \
        uint64_t src, uint64_t srcScale, uint64_t dst, uint64_t dstScale, uint64_t scratch, uint32_t rows,             \
        uint32_t cols, uint32_t srcRow, uint32_t srcScaleRow, uint32_t srcScaleStep, uint32_t dstRow,                  \
        uint32_t dstStep, uint32_t dstScaleRow, uint32_t dstScaleStep, uint32_t roundMode, uint32_t maskLimit)         \
    {                                                                                                                  \
        for (uint32_t row = 0; row < rows; ++row) {                                                                    \
            for (uint32_t pos = 0; pos < cols; pos += 32) {                                                            \
                uint32_t count = cols - pos < 32 ? cols - pos : 32;                                                    \
                uint32_t begin = row * cols + pos;                                                                     \
                uint32_t active = begin < maskLimit ? maskLimit - begin : 0;                                           \
                if (active > count)                                                                                    \
                    active = count;                                                                                    \
                tla_mx_tensor_requant_##S##_##D##_masked(                                                              \
                    src + row * srcRow + pos / SP, srcScale + row * srcScaleRow + (pos / 32) * srcScaleStep, scratch,  \
                    dstScale + row * dstScaleRow + (pos / 32) * dstScaleStep, 1, count, count, 1, count, 1, roundMode, \
                    active);                                                                                           \
                scatterBytes(                                                                                          \
                    reinterpret_cast<__ubuf__ uint8_t*>(scratch),                                                      \
                    reinterpret_cast<__ubuf__ uint8_t*>(dst + row * dstRow + pos * dstStep), dstStep, count);          \
            }                                                                                                          \
        }                                                                                                              \
    }
#define SCATTER_WIDEN(S, D)                                                                                            \
    extern "C"[aicore] void tla_mx_tensor_##S##_##D##_masked(                                                          \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t); \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_scatter_mx_##S##_##D(                                \
        uint64_t src, uint64_t srcScale, uint64_t dst, uint64_t dstScale, uint64_t scratch, uint32_t rows,             \
        uint32_t cols, uint32_t srcRow, uint32_t srcScaleRow, uint32_t srcScaleStep, uint32_t dstRow,                  \
        uint32_t dstStep, uint32_t dstScaleRow, uint32_t dstScaleStep, uint32_t roundMode, uint32_t maskLimit)         \
    {                                                                                                                  \
        for (uint32_t row = 0; row < rows; ++row) {                                                                    \
            for (uint32_t pos = 0; pos < cols; pos += 32) {                                                            \
                uint32_t count = cols - pos < 32 ? cols - pos : 32;                                                    \
                uint32_t begin = row * cols + pos;                                                                     \
                uint32_t active = begin < maskLimit ? maskLimit - begin : 0;                                           \
                if (active > count)                                                                                    \
                    active = count;                                                                                    \
                tla_mx_tensor_##S##_##D##_masked(                                                                      \
                    src + row * srcRow + pos / 2, scratch, srcScale + row * srcScaleRow + (pos / 32) * srcScaleStep,   \
                    dstScale + row * dstScaleRow + (pos / 32) * dstScaleStep, 1, count, count, count, 1, 1, active);   \
                scatterBytes(                                                                                          \
                    reinterpret_cast<__ubuf__ uint8_t*>(scratch),                                                      \
                    reinterpret_cast<__ubuf__ uint8_t*>(dst + row * dstRow + pos * dstStep), dstStep, count);          \
            }                                                                                                          \
        }                                                                                                              \
    }
SCATTER_WIDEN(e2m1, e4m3)
SCATTER_WIDEN(e2m1, e5m2)
SCATTER_WIDEN(e1m2, e4m3)
SCATTER_WIDEN(e1m2, e5m2)
SCATTER_MX(e4m3, 1, e4m3)
SCATTER_MX(e4m3, 1, e5m2)
SCATTER_MX(e5m2, 1, e4m3)
SCATTER_MX(e5m2, 1, e5m2)

#define SCATTER_FP4(S, SP, D)                                                                                          \
    extern "C"[aicore] void tla_mx_tensor_requant_##S##_##D##_masked(                                                  \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,  \
        uint32_t);                                                                                                     \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_scatter_fp4_##S##_##D(                               \
        uint64_t src, uint64_t srcScale, uint64_t dst, uint64_t dstScale, uint64_t scratch, uint32_t rows,             \
        uint32_t cols, uint32_t srcRow, uint32_t srcScaleRow, uint32_t srcScaleStep, uint32_t dstRow,                  \
        uint32_t dstStep, uint32_t dstOffset, uint32_t dstScaleRow, uint32_t dstScaleStep, uint32_t roundMode,         \
        uint32_t maskLimit)                                                                                            \
    {                                                                                                                  \
        for (uint32_t row = 0; row < rows; ++row) {                                                                    \
            for (uint32_t pos = 0; pos < cols; pos += 32) {                                                            \
                uint32_t count = cols - pos < 32 ? cols - pos : 32;                                                    \
                uint32_t begin = row * cols + pos;                                                                     \
                uint32_t active = begin < maskLimit ? maskLimit - begin : 0;                                           \
                if (active > count)                                                                                    \
                    active = count;                                                                                    \
                tla_mx_tensor_requant_##S##_##D##_masked(                                                              \
                    src + row * srcRow + pos / SP, srcScale + row * srcScaleRow + (pos / 32) * srcScaleStep, scratch,  \
                    dstScale + row * dstScaleRow + (pos / 32) * dstScaleStep, 1, count, count, 1, count, 1, roundMode, \
                    active);                                                                                           \
                uint32_t logical = dstOffset + row * dstRow + pos * dstStep;                                           \
                scatterNibbles(                                                                                        \
                    reinterpret_cast<__ubuf__ uint8_t*>(scratch),                                                      \
                    reinterpret_cast<__ubuf__ uint8_t*>(dst + logical / 2), dstStep, logical % 2, count);              \
            }                                                                                                          \
        }                                                                                                              \
    }
#define ALL_FP4(S, P) SCATTER_FP4(S, P, e2m1) SCATTER_FP4(S, P, e1m2)
ALL_FP4(e2m1, 2)
ALL_FP4(e1m2, 2)
ALL_FP4(e4m3, 1)
ALL_FP4(e5m2, 1)

extern "C"[aicore] __attribute__((always_inline)) void tla_mx_stage_scatter_b8(
    uint64_t scratch, uint64_t dst, uint32_t step, uint32_t offset, uint32_t count)
{
    scatterBytes(
        reinterpret_cast<__ubuf__ uint8_t*>(scratch), reinterpret_cast<__ubuf__ uint8_t*>(dst + offset), step, count);
}
extern "C"[aicore] __attribute__((always_inline)) void tla_mx_stage_scatter_fp4(
    uint64_t scratch, uint64_t dst, uint32_t step, uint32_t offset, uint32_t count)
{
    if (step == 1 && offset % 2 == 0)
        copyNibbles(
            reinterpret_cast<__ubuf__ uint8_t*>(scratch), reinterpret_cast<__ubuf__ uint8_t*>(dst + offset / 2), count);
    else
        scatterNibbles(
            reinterpret_cast<__ubuf__ uint8_t*>(scratch), reinterpret_cast<__ubuf__ uint8_t*>(dst + offset / 2), step,
            offset % 2, count);
}

#undef SCATTER_MX
#undef SCATTER_WIDEN
#undef SCATTER_FP4
#undef ALL_FP4

// ==========================================================================
// Stage drivers: strided dense <-> MX  (was mx_tensor_strided_dense.cpp)
// ==========================================================================

#define DECL_STAGE(K) extern "C"[aicore] void tla_mx_stage_##K(uint64_t, uint64_t, uint32_t, uint32_t, uint32_t)
DECL_STAGE(gather_b8);
DECL_STAGE(gather_fp4);
DECL_STAGE(gather_b16);
DECL_STAGE(gather_b32);
DECL_STAGE(scatter_b8);
DECL_STAGE(scatter_fp4);
DECL_STAGE(scatter_b16);
DECL_STAGE(scatter_b32);
#define COMMON                                                                                                      \
    uint64_t src, uint64_t dst, uint64_t scale, uint64_t scratch, uint32_t rows, uint32_t cols, uint32_t sr,        \
        uint32_t sx, uint32_t so, uint32_t dr, uint32_t dx, uint32_t doff, uint32_t qr, uint32_t qx, uint32_t mode, \
        uint32_t limit
#define GROUPS                                \
    for (uint32_t row = 0; row < rows; ++row) \
        for (uint32_t pos = 0; pos < cols; pos += 32)
#define PREPARE                                          \
    uint32_t count = cols - pos < 32 ? cols - pos : 32;  \
    uint32_t begin = row * cols + pos;                   \
    uint32_t active = begin < limit ? limit - begin : 0; \
    if (active > count)                                  \
    active = count
#define QUANT(S, SK, D, DK)                                                                                           \
    extern "C"[aicore] void tla_mx_tensor_quant_##S##_##D##_masked(                                                   \
        uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);          \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_strided_quant_##S##_##D(COMMON)                     \
    {                                                                                                                 \
        GROUPS                                                                                                        \
        {                                                                                                             \
            PREPARE;                                                                                                  \
            tla_mx_stage_gather_##SK(src, scratch, sx, so + row * sr + pos * sx, count);                              \
            tla_mx_tensor_quant_##S##_##D##_masked(                                                                   \
                scratch, scratch + 256, scale + row * qr + (pos / 32) * qx, 1, count, count, count, 1, mode, active); \
            tla_mx_stage_scatter_##DK(scratch + 256, dst, dx, doff + row * dr + pos * dx, count);                     \
        }                                                                                                             \
    }
#define DEQUANT(S, SK, D, DK)                                                                                         \
    extern "C"[aicore] void tla_mx_tensor_dequant_##S##_##D##_masked(                                                 \
        uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);          \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_strided_dequant_##S##_##D(COMMON)                   \
    {                                                                                                                 \
        GROUPS                                                                                                        \
        {                                                                                                             \
            PREPARE;                                                                                                  \
            tla_mx_stage_gather_##SK(src, scratch, sx, so + row * sr + pos * sx, count);                              \
            tla_mx_tensor_dequant_##S##_##D##_masked(                                                                 \
                scratch, scale + row * qr + (pos / 32) * qx, scratch + 256, 1, count, count, 1, count, mode, active); \
            tla_mx_stage_scatter_##DK(scratch + 256, dst, dx, doff + row * dr + pos * dx, count);                     \
        }                                                                                                             \
    }
// clang-format off
#define ALL_Q(S, K) QUANT(S, K, e2m1, fp4) QUANT(S, K, e1m2, fp4) QUANT(S, K, e4m3, b8) QUANT(S, K, e5m2, b8)
ALL_Q(f32, b32)
ALL_Q(f16, b16)
ALL_Q(bf16, b16)
#define ALL_D(S, K) DEQUANT(S, K, f32, b32) DEQUANT(S, K, f16, b16) DEQUANT(S, K, bf16, b16)
ALL_D(e2m1, fp4) ALL_D(e1m2, fp4) ALL_D(e4m3, b8) ALL_D(e5m2, b8)
// clang-format on

#undef DECL_STAGE
#undef COMMON
#undef GROUPS
#undef PREPARE
#undef QUANT
#undef DEQUANT
#undef ALL_Q
#undef ALL_D

// ==========================================================================
// Stage drivers: strided MX -> MX  (was mx_tensor_strided_mx.cpp)
// ==========================================================================

#define DECL_STAGE(K) extern "C"[aicore] void tla_mx_stage_##K(uint64_t, uint64_t, uint32_t, uint32_t, uint32_t)
    DECL_STAGE(gather_b8);
DECL_STAGE(gather_fp4);
DECL_STAGE(scatter_b8);
DECL_STAGE(scatter_fp4);
#define COMMON                                                                                                         \
    uint64_t src, uint64_t ss, uint64_t dst, uint64_t ds, uint64_t scratch, uint32_t rows, uint32_t cols, uint32_t sr, \
        uint32_t sx, uint32_t so, uint32_t ssr, uint32_t ssx, uint32_t dr, uint32_t dx, uint32_t doff, uint32_t dsr,   \
        uint32_t dsx, uint32_t mode, uint32_t limit
#define GROUPS                                \
    for (uint32_t row = 0; row < rows; ++row) \
        for (uint32_t pos = 0; pos < cols; pos += 32)
#define PREPARE                                          \
    uint32_t count = cols - pos < 32 ? cols - pos : 32;  \
    uint32_t begin = row * cols + pos;                   \
    uint32_t active = begin < limit ? limit - begin : 0; \
    if (active > count)                                  \
    active = count
#define REQUANT(S, SK, D, DK)                                                                                         \
    extern "C"[aicore] void tla_mx_tensor_requant_##S##_##D##_masked(                                                 \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, \
        uint32_t);                                                                                                    \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_strided_##S##_##D(COMMON)                           \
    {                                                                                                                 \
        GROUPS                                                                                                        \
        {                                                                                                             \
            PREPARE;                                                                                                  \
            tla_mx_stage_gather_##SK(src, scratch, sx, so + row * sr + pos * sx, count);                              \
            tla_mx_tensor_requant_##S##_##D##_masked(                                                                 \
                scratch, ss + row * ssr + (pos / 32) * ssx, scratch + 256, ds + row * dsr + (pos / 32) * dsx, 1,      \
                count, count, 1, count, 1, mode, active);                                                             \
            tla_mx_stage_scatter_##DK(scratch + 256, dst, dx, doff + row * dr + pos * dx, count);                     \
        }                                                                                                             \
    }
#define WIDEN(S, D)                                                                                                    \
    extern "C"[aicore] void tla_mx_tensor_##S##_##D##_masked(                                                          \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t); \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_strided_##S##_##D(COMMON)                            \
    {                                                                                                                  \
        GROUPS                                                                                                         \
        {                                                                                                              \
            PREPARE;                                                                                                   \
            tla_mx_stage_gather_fp4(src, scratch, sx, so + row * sr + pos * sx, count);                                \
            tla_mx_tensor_##S##_##D##_masked(                                                                          \
                scratch, scratch + 256, ss + row * ssr + (pos / 32) * ssx, ds + row * dsr + (pos / 32) * dsx, 1,       \
                count, count, count, 1, 1, active);                                                                    \
            tla_mx_stage_scatter_b8(scratch + 256, dst, dx, doff + row * dr + pos * dx, count);                        \
        }                                                                                                              \
    }
REQUANT(e2m1, fp4, e2m1, fp4)
REQUANT(e2m1, fp4, e1m2, fp4)
REQUANT(e1m2, fp4, e2m1, fp4)
REQUANT(e1m2, fp4, e1m2, fp4)
WIDEN(e2m1, e4m3)
WIDEN(e2m1, e5m2)
WIDEN(e1m2, e4m3)
WIDEN(e1m2, e5m2)
#define FP8(S) REQUANT(S, b8, e2m1, fp4) REQUANT(S, b8, e1m2, fp4) REQUANT(S, b8, e4m3, b8) REQUANT(S, b8, e5m2, b8)
FP8(e4m3)
FP8(e5m2)

#undef DECL_STAGE
#undef COMMON
#undef GROUPS
#undef PREPARE
#undef REQUANT
#undef WIDEN
#undef FP8

// ==========================================================================
// Non-unit scale stride: per-group dispatch  (was mx_tensor_scale_stride.cpp)
// ==========================================================================

// These adapters handle non-contiguous scale groups by converting one group
// at a time. The ordinary contiguous path avoids this loop entirely.
#define PREFIX_ARGS                                                                                                  \
    uint64_t src, uint64_t dst, uint64_t srcScale, uint64_t dstScale, uint32_t rows, uint32_t cols, uint32_t srcRow, \
        uint32_t dstRow, uint32_t srcScaleRow, uint32_t dstScaleRow, uint32_t srcScaleStep, uint32_t dstScaleStep,   \
        uint32_t roundMode, uint32_t maskLimit
#define GROUP_LOOP                            \
    for (uint32_t row = 0; row < rows; ++row) \
        for (uint32_t pos = 0; pos < cols; pos += 32)
#define GROUP_COUNTS                                             \
    uint32_t count = cols - pos < 32 ? cols - pos : 32;          \
    uint32_t start = row * cols + pos;                           \
    uint32_t active = start < maskLimit ? maskLimit - start : 0; \
    if (active > count)                                          \
    active = count

#define DECL_QUANT(S, D)                                            \
    extern "C"[aicore] void tla_mx_tensor_quant_##S##_##D##_masked( \
        uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)
#define QUANT(S, B, D, P)                                                                                       \
    DECL_QUANT(S, D);                                                                                           \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_scale_stride_quant_##S##_##D(PREFIX_ARGS)     \
    {                                                                                                           \
        GROUP_LOOP                                                                                              \
        {                                                                                                       \
            GROUP_COUNTS;                                                                                       \
            tla_mx_tensor_quant_##S##_##D##_masked(                                                             \
                src + row * srcRow + pos * B, dst + row * dstRow + pos / P,                                     \
                dstScale + row * dstScaleRow + (pos / 32) * dstScaleStep, 1, count, count, count, 1, roundMode, \
                active);                                                                                        \
        }                                                                                                       \
    }
#define DECL_DEQUANT(S, D)                                            \
    extern "C"[aicore] void tla_mx_tensor_dequant_##S##_##D##_masked( \
        uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)
#define DEQUANT(S, P, D, B)                                                                                   \
    DECL_DEQUANT(S, D);                                                                                       \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_scale_stride_dequant_##S##_##D(PREFIX_ARGS) \
    {                                                                                                         \
        GROUP_LOOP                                                                                            \
        {                                                                                                     \
            GROUP_COUNTS;                                                                                     \
            tla_mx_tensor_dequant_##S##_##D##_masked(                                                         \
                src + row * srcRow + pos / P, srcScale + row * srcScaleRow + (pos / 32) * srcScaleStep,       \
                dst + row * dstRow + pos * B, 1, count, count, 1, count, roundMode, active);                  \
        }                                                                                                     \
    }
#define DECL_REQUANT(S, D)                                                                                            \
    extern "C"[aicore] void tla_mx_tensor_requant_##S##_##D##_masked(                                                 \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, \
        uint32_t)
#define REQUANT(S, SP, D, DP)                                                                                      \
    DECL_REQUANT(S, D);                                                                                            \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_scale_stride_requant_##S##_##D(PREFIX_ARGS)      \
    {                                                                                                              \
        GROUP_LOOP                                                                                                 \
        {                                                                                                          \
            GROUP_COUNTS;                                                                                          \
            tla_mx_tensor_requant_##S##_##D##_masked(                                                              \
                src + row * srcRow + pos / SP, srcScale + row * srcScaleRow + (pos / 32) * srcScaleStep,           \
                dst + row * dstRow + pos / DP, dstScale + row * dstScaleRow + (pos / 32) * dstScaleStep, 1, count, \
                count, 1, count, 1, roundMode, active);                                                            \
        }                                                                                                          \
    }
#define DECL_WIDEN(S, D)                                      \
    extern "C"[aicore] void tla_mx_tensor_##S##_##D##_masked( \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)
#define WIDEN(S, D)                                                                                              \
    DECL_WIDEN(S, D);                                                                                            \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_scale_stride_widen_##S##_##D(PREFIX_ARGS)      \
    {                                                                                                            \
        GROUP_LOOP                                                                                               \
        {                                                                                                        \
            GROUP_COUNTS;                                                                                        \
            tla_mx_tensor_##S##_##D##_masked(                                                                    \
                src + row * srcRow + pos / 2, dst + row * dstRow + pos,                                          \
                srcScale + row * srcScaleRow + (pos / 32) * srcScaleStep,                                        \
                dstScale + row * dstScaleRow + (pos / 32) * dstScaleStep, 1, count, count, count, 1, 1, active); \
        }                                                                                                        \
    }

#define MX_QUANT_DENSE(S, B) QUANT(S, B, e2m1, 2) QUANT(S, B, e1m2, 2) QUANT(S, B, e4m3, 1) QUANT(S, B, e5m2, 1)
MX_QUANT_DENSE(f32, 4)
MX_QUANT_DENSE(f16, 2)
MX_QUANT_DENSE(bf16, 2)
#define MX_DEQUANT_DENSE(S, P) DEQUANT(S, P, f32, 4) DEQUANT(S, P, f16, 2) DEQUANT(S, P, bf16, 2)
MX_DEQUANT_DENSE(e2m1, 2)
MX_DEQUANT_DENSE(e1m2, 2)
MX_DEQUANT_DENSE(e4m3, 1)
MX_DEQUANT_DENSE(e5m2, 1)
REQUANT(e2m1, 2, e2m1, 2)
REQUANT(e2m1, 2, e1m2, 2)
REQUANT(e1m2, 2, e2m1, 2)
REQUANT(e1m2, 2, e1m2, 2)
#define MX_REQUANT_FP8(S) REQUANT(S, 1, e2m1, 2) REQUANT(S, 1, e1m2, 2) REQUANT(S, 1, e4m3, 1) REQUANT(S, 1, e5m2, 1)
MX_REQUANT_FP8(e4m3)
MX_REQUANT_FP8(e5m2)
WIDEN(e2m1, e4m3)
WIDEN(e2m1, e5m2)
WIDEN(e1m2, e4m3)
WIDEN(e1m2, e5m2)

#undef PREFIX_ARGS
#undef GROUP_LOOP
#undef GROUP_COUNTS
#undef DECL_QUANT
#undef QUANT
#undef DECL_DEQUANT
#undef DEQUANT
#undef DECL_REQUANT
#undef REQUANT
#undef DECL_WIDEN
#undef WIDEN
#undef MX_QUANT_DENSE
#undef MX_DEQUANT_DENSE
#undef MX_REQUANT_FP8

// ==========================================================================
// General layout fallback: gather, convert, scatter  (was mx_tensor_layout.cpp)
// ==========================================================================

#define DECL_STAGE(K) extern "C"[aicore] void tla_mx_stage_##K(uint64_t, uint64_t, uint32_t, uint32_t, uint32_t)
DECL_STAGE(gather_b8);
DECL_STAGE(gather_fp4);
DECL_STAGE(gather_b16);
DECL_STAGE(gather_b32);
DECL_STAGE(scatter_b8);
DECL_STAGE(scatter_fp4);
DECL_STAGE(scatter_b16);
DECL_STAGE(scatter_b32);
struct MxLayout {
    uint32_t rb, cb, r0, r1, c0, c1, ro, co;
};
__aicore__ __attribute__((always_inline)) static uint32_t mxOffset(MxLayout v, uint32_t r, uint32_t c)
{
    r += v.ro;
    c += v.co;
    return (r % v.rb) * v.r0 + (r / v.rb) * v.r1 + (c % v.cb) * v.c0 + (c / v.cb) * v.c1;
}
#define DESC(P)                                                                                                     \
    uint32_t P##rb, uint32_t P##cb, uint32_t P##r0, uint32_t P##r1, uint32_t P##c0, uint32_t P##c1, uint32_t P##ro, \
        uint32_t P##co
#define BUILD(P) MxLayout P = {P##rb, P##cb, P##r0, P##r1, P##c0, P##c1, P##ro, P##co}
#define COMMON                                                                                                    \
    uint64_t src, uint64_t srcScale, uint64_t dst, uint64_t dstScale, uint64_t tmp, uint32_t rows, uint32_t cols, \
        uint32_t srcAxis, uint32_t dstAxis, uint32_t mode, uint32_t limit, DESC(a), DESC(b), DESC(q), DESC(p)
#define PREP  \
    BUILD(a); \
    BUILD(b); \
    BUILD(q); \
    BUILD(p); \
    uint32_t groups = (cols + 31) / 32
#define SSRC(R, C) mxOffset(q, srcAxis == 0 ? (R) / 32 : (R), srcAxis == 1 ? (C) / 32 : (C))
#define DDEST(R, C) mxOffset(p, dstAxis == 0 ? (R) / 32 : (R), dstAxis == 1 ? (C) / 32 : (C))
#define GATHER(K, STEP)                                                          \
    do {                                                                         \
        for (uint32_t i = 0; i < count; ++i) {                                   \
            uint32_t r = dstAxis == 1 ? row : pos + i;                           \
            uint32_t c = dstAxis == 1 ? pos + i : row;                           \
            tla_mx_stage_gather_##K(src, tmp + 32 * i, 1, mxOffset(a, r, c), 1); \
        }                                                                        \
        tla_mx_stage_gather_##K(tmp, tmp + 1024, STEP, 0, count);                \
    } while (0)
#define GATHER_SCALE                                                      \
    for (uint32_t i = 0; i < count; ++i) {                                \
        uint32_t r = dstAxis == 1 ? row : pos + i;                        \
        uint32_t c = dstAxis == 1 ? pos + i : row;                        \
        tla_mx_stage_gather_b8(srcScale, tmp + 32 * i, 1, SSRC(r, c), 1); \
    }                                                                     \
    tla_mx_stage_gather_b8(tmp, tmp + 1536, 32, 0, count)
#define SCATTER(K)                                                            \
    for (uint32_t i = 0; i < count; ++i) {                                    \
        uint32_t r = dstAxis == 1 ? row : pos + i;                            \
        uint32_t c = dstAxis == 1 ? pos + i : row;                            \
        tla_mx_stage_gather_##K(tmp + 1280, tmp + 32 * i, 1, i, 1);           \
        tla_mx_stage_scatter_##K(tmp + 32 * i, dst, 2, mxOffset(b, r, c), 1); \
    }
#define FINAL_SCALE                                                          \
    for (uint32_t row = 0; row < rows; ++row)                                \
        for (uint32_t g = 0; g < groups; ++g) {                              \
            uint32_t r = dstAxis == 1 ? row : g * 32;                        \
            uint32_t c = dstAxis == 1 ? g * 32 : row;                        \
            tla_mx_stage_gather_b8(tmp + 1792, tmp, 1, row * groups + g, 1); \
            tla_mx_stage_scatter_b8(tmp, dstScale, 1, DDEST(r, c), 1);       \
        }
#define LOOP                                  \
    for (uint32_t row = 0; row < rows; ++row) \
        for (uint32_t pos = 0; pos < cols; pos += 32)
#define COUNT                                            \
    uint32_t count = cols - pos < 32 ? cols - pos : 32;  \
    uint32_t begin = row * cols + pos;                   \
    uint32_t active = begin < limit ? limit - begin : 0; \
    if (active > count)                                  \
    active = count
#define Q(S, SK, SS, D, DK)                                                                                    \
    extern "C"[aicore] void tla_mx_tensor_quant_##S##_##D##_masked(                                            \
        uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);   \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_layout_quant_##S##_##D(COMMON)               \
    {                                                                                                          \
        PREP;                                                                                                  \
        LOOP                                                                                                   \
        {                                                                                                      \
            COUNT;                                                                                             \
            GATHER(SK, SS);                                                                                    \
            tla_mx_tensor_quant_##S##_##D##_masked(                                                            \
                tmp + 1024, tmp + 1280, tmp + 1792 + row * groups + pos / 32, 1, count, count, count, 1, mode, \
                active);                                                                                       \
            SCATTER(DK);                                                                                       \
        }                                                                                                      \
        FINAL_SCALE;                                                                                           \
    }
#define D(S, SK, SS, T, TK)                                                                                  \
    extern "C"[aicore] void tla_mx_tensor_dequant_##S##_##T##_masked(                                        \
        uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t); \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_layout_dequant_##S##_##T(COMMON)           \
    {                                                                                                        \
        PREP;                                                                                                \
        LOOP                                                                                                 \
        {                                                                                                    \
            COUNT;                                                                                           \
            GATHER(SK, SS);                                                                                  \
            GATHER_SCALE;                                                                                    \
            tla_mx_tensor_dequant_##S##_##T##_masked(                                                        \
                tmp + 1024, tmp + 1536, tmp + 1280, 1, count, count, 1, count, mode, active);                \
            SCATTER(TK);                                                                                     \
        }                                                                                                    \
    }
#define MX(S, SK, SS, T, TK)                                                                                          \
    extern "C"[aicore] void tla_mx_tensor_requant_##S##_##T##_masked(                                                 \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, \
        uint32_t);                                                                                                    \
    extern "C"[aicore] void tla_mx_group_regroup_##S##_##T(                                                           \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t);                                        \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_layout_requant_##S##_##T(COMMON)                    \
    {                                                                                                                 \
        PREP;                                                                                                         \
        LOOP                                                                                                          \
        {                                                                                                             \
            COUNT;                                                                                                    \
            GATHER(SK, SS);                                                                                           \
            GATHER_SCALE;                                                                                             \
            if (srcAxis == dstAxis)                                                                                   \
                tla_mx_tensor_requant_##S##_##T##_masked(                                                             \
                    tmp + 1024, tmp + 1536, tmp + 1280, tmp + 1792 + row * groups + pos / 32, 1, count, count, 1,     \
                    count, 1, mode, active);                                                                          \
            else                                                                                                      \
                tla_mx_group_regroup_##S##_##T(                                                                       \
                    tmp + 1024, tmp + 1536, tmp + 1280, tmp + 1792 + row * groups + pos / 32, count, mode, active);   \
            SCATTER(TK);                                                                                              \
        }                                                                                                             \
        FINAL_SCALE;                                                                                                  \
    }
#define W(S, SK, SS, T)                                                                                                \
    extern "C"[aicore] void tla_mx_tensor_##S##_##T##_masked(                                                          \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t); \
    extern "C"[aicore] void tla_mx_group_regroup_##S##_##T(                                                            \
        uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t);                                         \
    extern "C"[aicore] __attribute__((always_inline)) void tla_mx_layout_requant_##S##_##T(COMMON)                     \
    {                                                                                                                  \
        PREP;                                                                                                          \
        LOOP                                                                                                           \
        {                                                                                                              \
            COUNT;                                                                                                     \
            GATHER(SK, SS);                                                                                            \
            GATHER_SCALE;                                                                                              \
            if (srcAxis == dstAxis)                                                                                    \
                tla_mx_tensor_##S##_##T##_masked(                                                                      \
                    tmp + 1024, tmp + 1280, tmp + 1536, tmp + 1792 + row * groups + pos / 32, 1, count, count, count,  \
                    1, 1, active);                                                                                     \
            else                                                                                                       \
                tla_mx_group_regroup_##S##_##T(                                                                        \
                    tmp + 1024, tmp + 1536, tmp + 1280, tmp + 1792 + row * groups + pos / 32, count, mode, active);    \
            SCATTER(b8);                                                                                               \
        }                                                                                                              \
        FINAL_SCALE;                                                                                                   \
    }
// clang-format off
#define QA(S, K, N) Q(S, K, N, e2m1, fp4) Q(S, K, N, e1m2, fp4) Q(S, K, N, e4m3, b8) Q(S, K, N, e5m2, b8)
QA(f32, b32, 8)
QA(f16, b16, 16)
QA(bf16, b16, 16)
#define DA(S, K, N) D(S, K, N, f32, b32) D(S, K, N, f16, b16) D(S, K, N, bf16, b16)
DA(e2m1, fp4, 64)
DA(e1m2, fp4, 64) DA(e4m3, b8, 32) DA(e5m2, b8, 32)
#define M4(S) MX(S, fp4, 64, e2m1, fp4) MX(S, fp4, 64, e1m2, fp4) W(S, fp4, 64, e4m3) W(S, fp4, 64, e5m2)
    M4(e2m1) M4(e1m2)
#define M8(S) MX(S, b8, 32, e2m1, fp4) MX(S, b8, 32, e1m2, fp4) MX(S, b8, 32, e4m3, b8) MX(S, b8, 32, e5m2, b8)
        M8(e4m3) M8(e5m2)
// clang-format on

#undef DECL_STAGE
#undef DESC
#undef BUILD
#undef COMMON
#undef PREP
#undef SSRC
#undef DDEST
#undef GATHER
#undef GATHER_SCALE
#undef SCATTER
#undef FINAL_SCALE
#undef LOOP
#undef COUNT
#undef Q
#undef D
#undef MX
#undef W
#undef QA
#undef DA
#undef M4
#undef M8

#endif
