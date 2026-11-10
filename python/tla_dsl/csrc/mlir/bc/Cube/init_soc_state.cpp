#include "../common.h"
#include "catlass/catlass.hpp"
#include "kernel_operator.h"
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
extern "C" {
// AIC-only symbol: hivmc links both meta_op.aic and meta_op.aiv into each
// subcore compile. A shared Scalar symbol would resolve to the cube body
// (with set_padding) on AIV and fault. Keep cube InitSoc under a distinct name.
[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_init_soc_state_aic()
{
    AscendC::InitSocState();
}
} // extern "C"
#endif
