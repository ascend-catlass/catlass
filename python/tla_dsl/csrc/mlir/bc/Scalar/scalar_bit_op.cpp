#include "../common.h"

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

/**
 * Scalar uint64 bit-level operations on the Ascend950PR (3510) scalar unit.
 *
 * The CCE intrinsics behind them -- ``bcnt0`` / ``bcnt1`` (population count of
 * zeros / ones), ``clz`` (leading zero count) and ``sff0`` / ``sff1`` (first
 * zero / one from the least significant bit) -- are reachable ONLY through a
 * bitcode helper. The vector-side ``hivm.vbcnt.x`` / ``hivm.vcls.*`` ops are
 * 32-lane operations, not the 64-bit scalar entry points these APIs need, so
 * they cannot be used as a direct lowering.
 */

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

extern "C" {

// Number of 0 bits in the binary representation of valueIn.
[aicore] __attribute__((always_inline)) int64_t _mlir_ciface_bcnt0(uint64_t valueIn)
{
    return bcnt0(valueIn);
}

// Number of 1 bits in the binary representation of valueIn.
[aicore] __attribute__((always_inline)) int64_t _mlir_ciface_bcnt1(uint64_t valueIn)
{
    return bcnt1(valueIn);
}

// Number of leading 0 bits before the most significant 1; 64 when valueIn is 0.
[aicore] __attribute__((always_inline)) int64_t _mlir_ciface_clz(uint64_t valueIn)
{
    return clz(valueIn);
}

// Position of the first 0 bit from the least significant bit; -1 if not found.
[aicore] __attribute__((always_inline)) int64_t _mlir_ciface_sff0(uint64_t valueIn)
{
    return sff0(valueIn);
}

// Position of the first 1 bit from the least significant bit; -1 if not found.
[aicore] __attribute__((always_inline)) int64_t _mlir_ciface_sff1(uint64_t valueIn)
{
    return sff1(valueIn);
}

} // extern "C"

#endif
