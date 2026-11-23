#include "../common.h"

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

extern "C" {

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_dci()
{
    dci();
}

} // extern "C"

#endif
