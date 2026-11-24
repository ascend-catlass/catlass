#include "catlass/catlass.hpp"

// The AIV-only helpers provide the native AR read and clear through ccec
// bitcode. The HIVM dialects and the LLVM used by hivmc expose neither
// instruction as an operation or registered intrinsic.

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

extern "C" {

// The accumulated byte count. Reading AR does not wait for the vector pipe, so
// the caller still owes whatever handoff the producing sequence needs.
__aiv__ __attribute__((always_inline)) int64_t _mlir_ciface_tla_get_ar()
{
    return get_ar();
}

// The clear as it is used from scalar code. SPRCLR writes a vector-unit
// register, so it needs a vector function around it; this one opens its own,
// holding nothing but the clear. That is also what the native scalar
// ``ClearSpr<AR>`` does.
__aiv__ __attribute__((always_inline)) void _mlir_ciface_tla_clear_ar()
{
    __VEC_SCOPE__
    {
        sprclr(SPR_AR);
    }
}

} // extern "C"

#endif
