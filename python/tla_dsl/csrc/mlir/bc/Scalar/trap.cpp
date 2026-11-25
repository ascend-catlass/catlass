#include "../common.h"

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

/**
 * Trap on the scalar unit (CANN ``AscendC::Trap()``).
 *
 * ``Trap`` is reachable only through a bitcode helper. The HIVM dialect has no
 * trap op, and the intrinsic form is not callable from MLIR: ccec compiles
 * ``AscendC::Trap()`` to ``llvm.hivm.TRAP`` (verified on a probe stub), which is
 * registered in the CCE clang rather than in the LLVM hivmc links against, so
 * ``llvm.call_intrinsic "llvm.hivm.TRAP"`` is rejected the same way
 * ``llvm.hivm.DSB`` and ``llvm.hivm.CONV.*`` are. tla-lower-scalar-ops
 * therefore rewrites ``tla.trap`` into a call to this helper, the linked
 * bitcode supplies the instruction, and ``always_inline`` leaves exactly one
 * trap at the call site.
 *
 * ``[aicore]`` because the scalar unit is present on both AIC and AIV, and this
 * file lives in bc/Scalar/, which bc_compile.py compiles once per core type.
 * Unlike the ub-domain data-sync barrier, the two compiled bodies are identical
 * (both are the same trap instruction), so one symbol serves both cores: there
 * is nothing a core-specific ``_aiv`` alias would protect.
 *
 * Not pure: repeated traps must not be merged, and a trap must never be dropped
 * as dead code.
 */

#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

extern "C" {

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_trap()
{
    AscendC::Trap();
}

} // extern "C"

#endif
