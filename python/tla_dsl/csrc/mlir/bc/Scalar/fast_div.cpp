#include "../common.h"

#include <type_traits>

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

/**
 * Scalar bitcode for the fast unsigned division stack on the Ascend950PR (3510)
 * scalar unit.
 *
 * There is no single CCE intrinsic (and no hivmc op) that covers it, so the
 * whole stack lives here:
 *
 *   - magic/shift precomputation for a fixed divisor: ``shift`` comes from
 *     ``clz`` and ``bcnt1``, ``magic`` from the closed form on 32-bit inputs and
 *     from a software long division on 64-bit inputs;
 *   - the SIMD ``//`` / ``%`` consumers, which compute
 *     ``quotient = (dividend + UMulHi(dividend, magic)) >> shift`` and
 *     ``remainder = dividend - quotient * divisor`` with a software ``UMulHi``
 *     (no ``__umulhi`` / ``__umul64hi``).
 *
 * Every exported result is its own scalar-return helper: a multi-result
 * func.call would have to come back as a struct, and that return convention
 * does not survive the scalar-unit device ABI.
 *
 * The constants mirror the ones asc-devkit compiles in but are defined here
 * locally: this stub must not rely on that header's internal namespace.
 */

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

namespace {

constexpr int64_t kBit32Len = 32;
constexpr int64_t kBit64Len = 64;
constexpr uint64_t kUint64Max = 0xFFFFFFFFFFFFFFFFull;
constexpr uint64_t kUint64HighestBitMask = 0x8000000000000000ull;
constexpr uint64_t kFullMaskB32 = 0xFFFFFFFFull;

// --- magic/shift precomputation -------------------------------------------

// Software long division on the unsigned 64-bit domain: floor(dividend /
// divisor) + 1. Used only by the uint64_t magic branch; for a positive divisor
// the two's-complement subtraction below cannot wrap.
__aicore__ inline uint64_t GetUintDivMagic(uint64_t dividend, uint64_t divisor)
{
    uint64_t quotient = 0;
    uint64_t remainder = dividend;
    uint64_t borrow = 0;

    for (int i = 0; i < kBit64Len; ++i) {
        quotient <<= 1;
        borrow = (remainder & kUint64HighestBitMask) > 0 ? 1 : 0;
        remainder <<= 1;
        if (borrow == 1) {
            remainder = kUint64Max - divisor + 1 + remainder;
            quotient |= 1;
        } else if (remainder >= divisor) {
            remainder -= divisor;
            quotient |= 1;
        }
    }
    return quotient + 1;
}

// shift = ceil(log2(divisor)), from the leading-zero count and the population
// count: a power-of-two divisor is the only case where the two disagree.
template <typename T>
__aicore__ inline T ComputeFastDivShift(T divisor)
{
    static_assert(
        std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value,
        "Input type T only supports uint32_t, uint64_t.");
    const int64_t pos = kBit64Len - clz(static_cast<uint64_t>(divisor));
    const int64_t cnt1 = bcnt1(static_cast<uint64_t>(divisor));
    return static_cast<T>(cnt1 == 1 ? pos - 1 : pos);
}

// magic = floor(2^N * (2^shift - divisor) / divisor) + 1. The 32-bit branch
// evaluates it directly; the 64-bit branch has no single instruction and uses
// the long division above. The 32-bit divisor bound is INT32_MAX; the 64-bit
// one is INT64_MAX.
//
// In-domain ``shift <= width - 1``. ``divisor == 0`` and ``shift >= width`` are
// out of domain; they return a defined 0 instead of entering division-by-zero
// or an out-of-range ``<<`` whose result could change with the optimization
// level.
template <typename T>
__aicore__ inline T ComputeFastDivMagic(T divisor, T shift)
{
    static_assert(
        std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value,
        "Input type T only supports uint32_t, uint64_t.");
    if (divisor == 0) {
        return 0;
    }
    if constexpr (std::is_same<T, uint32_t>::value) {
        if (shift >= kBit32Len) {
            return 0;
        }
        // The divisor is required to be <= INT32_MAX, which keeps the signed
        // 64-bit intermediate below from overflowing even though the closed
        // form multiplies by 2^32.
        return static_cast<T>(
            (static_cast<uint64_t>(1) << kBit32Len) * ((static_cast<uint64_t>(1) << shift) - divisor) / divisor + 1);
    } else {
        if (shift >= kBit64Len) {
            return 0;
        }
        uint64_t dividend = (static_cast<uint64_t>(1) << shift) - divisor;
        return static_cast<T>(GetUintDivMagic(dividend, divisor));
    }
}

// Reference shape of the asc-devkit helper: both results in one pass.
template <typename T>
__aicore__ inline void GetUintDivMagicAndShiftImpl(T& magic, T& shift, T divisor)
{
    shift = ComputeFastDivShift<T>(divisor);
    magic = ComputeFastDivMagic<T>(divisor, shift);
}

// --- SIMD // and % consumers ----------------------------------------------

// High N bits of the 2N-bit product, N = 32 or 64.
template <typename T>
__aicore__ inline T UMulHi(T dividend, T magic)
{
    static_assert(
        std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value,
        "Input type T only supports uint32_t, uint64_t.");

    if constexpr (std::is_same<T, uint32_t>::value) {
        return static_cast<T>((static_cast<uint64_t>(dividend) * static_cast<uint64_t>(magic)) >> kBit32Len);
    } else {
        uint64_t dividendHigh = dividend >> kBit32Len;
        uint64_t dividendLow = dividend & kFullMaskB32;
        uint64_t magicHigh = magic >> kBit32Len;
        uint64_t magicLow = magic & kFullMaskB32;

        uint64_t dividendLowMagicLow = dividendLow * magicLow;
        uint64_t dividendLowMagicLowHigh = dividendLowMagicLow >> kBit32Len;

        uint64_t dividendHighMagicLow = dividendHigh * magicLow;
        uint64_t dividendHighMagicLowHigh = dividendHighMagicLow >> kBit32Len;
        uint64_t dividendHighMagicLowLow = dividendHighMagicLow & kFullMaskB32;

        uint64_t dividendLowMagicHigh = dividendLow * magicHigh;
        uint64_t dividendLowMagicHighHigh = dividendLowMagicHigh >> kBit32Len;
        uint64_t dividendLowMagicHighLow = dividendLowMagicHigh & kFullMaskB32;

        uint64_t dividendHighMagicHigh = dividendHigh * magicHigh;

        uint64_t bitFrom32To63 = dividendLowMagicLowHigh + dividendHighMagicLowLow + dividendLowMagicHighLow;

        return dividendHighMagicHigh + dividendHighMagicLowHigh + dividendLowMagicHighHigh +
               (bitFrom32To63 >> kBit32Len);
    }
}

// dividend <= INT32_MAX / INT64_MAX keeps dividend + UMulHi inside one word, so
// the N-bit add below cannot wrap. A shift is in-domain only when it is below
// the width; a wider one would be UB in `>>`, so it returns a defined 0.
template <typename T>
__aicore__ inline T FastDivImpl(T dividend, T magic, T shift)
{
    if constexpr (std::is_same<T, uint32_t>::value) {
        if (shift >= kBit32Len) {
            return 0;
        }
    } else {
        if (shift >= kBit64Len) {
            return 0;
        }
    }
    T quotient = UMulHi(dividend, magic);
    T sum = dividend + quotient;
    return sum >> shift;
}

template <typename T>
__aicore__ inline T FastModImpl(T dividend, T divisor, T magic, T shift)
{
    T quotient = FastDivImpl(dividend, magic, shift);
    return dividend - quotient * divisor;
}

} // namespace

extern "C" {

[aicore] __attribute__((always_inline)) uint32_t _mlir_ciface_tla_fast_div_shift_32(uint32_t divisor)
{
    return ComputeFastDivShift<uint32_t>(divisor);
}

[aicore] __attribute__((always_inline)) uint32_t _mlir_ciface_tla_fast_div_magic_32(uint32_t divisor, uint32_t shift)
{
    return ComputeFastDivMagic<uint32_t>(divisor, shift);
}

[aicore] __attribute__((always_inline)) uint64_t _mlir_ciface_tla_fast_div_shift_64(uint64_t divisor)
{
    return ComputeFastDivShift<uint64_t>(divisor);
}

[aicore] __attribute__((always_inline)) uint64_t _mlir_ciface_tla_fast_div_magic_64(uint64_t divisor, uint64_t shift)
{
    return ComputeFastDivMagic<uint64_t>(divisor, shift);
}

[aicore] __attribute__((always_inline)) uint32_t
_mlir_ciface_tla_simd_fast_div_32(uint32_t dividend, uint32_t magic, uint32_t shift)
{
    return FastDivImpl<uint32_t>(dividend, magic, shift);
}

[aicore] __attribute__((always_inline)) uint32_t
_mlir_ciface_tla_simd_fast_mod_32(uint32_t dividend, uint32_t divisor, uint32_t magic, uint32_t shift)
{
    return FastModImpl<uint32_t>(dividend, divisor, magic, shift);
}

[aicore] __attribute__((always_inline)) uint64_t
_mlir_ciface_tla_simd_fast_div_64(uint64_t dividend, uint64_t magic, uint64_t shift)
{
    return FastDivImpl<uint64_t>(dividend, magic, shift);
}

[aicore] __attribute__((always_inline)) uint64_t
_mlir_ciface_tla_simd_fast_mod_64(uint64_t dividend, uint64_t divisor, uint64_t magic, uint64_t shift)
{
    return FastModImpl<uint64_t>(dividend, divisor, magic, shift);
}

} // extern "C"

#endif
