#include "kernel_operator.h"

#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
extern "C" {

#if defined(__DAV_VEC__)
#define TLA_ICACHE_HELPER _mlir_ciface_tla_icache_prefetch_aiv
#else
#define TLA_ICACHE_HELPER _mlir_ciface_tla_icache_prefetch_aic
#endif

[aicore] __attribute__((always_inline)) void TLA_ICACHE_HELPER(int64_t units)
{
    AscendC::ICachePreLoad(units);
}

#undef TLA_ICACHE_HELPER

} // extern "C"
#endif
