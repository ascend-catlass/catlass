#include "../common.h"
#include "catlass/catlass.hpp"
#include "kernel_operator.h"
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
extern "C" {
// AIV-only symbol; see Cube/init_soc_state.cpp for why AIC/AIV must not share
// one @_mlir_ciface_tla_init_soc_state across meta_op aggregates.
[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_init_soc_state_aiv()
{
    AscendC::InitSocState();
}
} // extern "C"
#endif
