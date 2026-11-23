#include "../common.h"
#include "kernel_operator.h"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
extern "C"[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_dcache_prefetch_aiv(
    uint64_t address, int64_t byteOffset)
{
    dc_preload(reinterpret_cast<__gm__ uint64_t*>(address), byteOffset);
}
#endif
