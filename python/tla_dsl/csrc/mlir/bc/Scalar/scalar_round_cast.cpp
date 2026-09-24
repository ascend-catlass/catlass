#include "../common.h"

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

/**
 * Scalar float conversion with an explicit rounding mode.
 *
 * The Ascend950PR (3510) scalar unit has one instruction per rounding mode:
 *
 *   conv_f322s32r  round to nearest, ties to even
 *   conv_f322s32a  round to nearest, ties away from zero
 *   conv_f322s32f  round toward -inf (floor)
 *   conv_f322s32c  round toward +inf (ceil)
 *   conv_f322f16o  f32 -> f16, round to odd
 *
 * They are reachable ONLY through a bitcode helper. Two shorter routes were
 * measured and both fail:
 *
 *   - ``math.floor``/``ceil``/``round``/``roundeven``/``llvm.intr.rint`` followed
 *     by ``arith.fptosi`` compiles, but leaves an undefined ``floorf``/``ceilf``/
 *     ``roundf``/``roundevenf``/``rintf`` symbol -- LLVM lowers them to a libm
 *     call, not to the instruction.
 *   - ``llvm.call_intrinsic "llvm.hivm.CONV.f322s32r"`` is rejected by hivmc
 *     with "could not find LLVM intrinsic": the CONV intrinsics are registered
 *     in the CCE clang, not in the LLVM that hivmc links against.
 *
 * ccec does select them from this source -- the compiled bitcode carries
 * ``llvm.hivm.CONV.f322s32{r,a,f,c}`` and nothing else -- so one always_inline
 * helper per mode costs a single instruction at the call site.
 *
 * ``[aicore]`` because the scalar unit is present on both AIC and AIV. This file
 * lives in bc/Scalar/, which bc_compile.py compiles once per core type and links
 * into both meta-op bitcodes -- so a core-agnostic helper is written once here
 * rather than as a shared header plus one stub under Cube/ and another under
 * Vector/.
 */

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

extern "C" {

[aicore] __attribute__((always_inline)) int32_t _mlir_ciface_tla_cast_f32_to_i32_rn(float value)
{
    return conv_f322s32r(value);
}

[aicore] __attribute__((always_inline)) int32_t _mlir_ciface_tla_cast_f32_to_i32_ra(float value)
{
    return conv_f322s32a(value);
}

[aicore] __attribute__((always_inline)) int32_t _mlir_ciface_tla_cast_f32_to_i32_rd(float value)
{
    return conv_f322s32f(value);
}

[aicore] __attribute__((always_inline)) int32_t _mlir_ciface_tla_cast_f32_to_i32_ru(float value)
{
    return conv_f322s32c(value);
}

// f32 -> f16 is a separate instruction with only one rounding mode: round to
// odd, which picks the neighbour with an odd mantissa whenever the value is not
// exactly representable. That is NOT what a plain narrowing wants -- arith.truncf
// already gives correct round-to-nearest-even here -- so this is exposed only
// when the caller asks for it by name. Its use is avoiding double rounding when
// a value is narrowed again afterwards.
[aicore] __attribute__((always_inline)) half _mlir_ciface_tla_cast_f32_to_f16_o(float value)
{
    return conv_f322f16o(value);
}

} // extern "C"

#endif
