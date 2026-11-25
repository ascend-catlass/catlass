#include "kernel_operator.h"

#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
extern "C" {

#if defined(__DAV_VEC__)
#define TLA_ICACHE_HELPER _mlir_ciface_tla_get_icache_prefetch_status_aiv
#else
#define TLA_ICACHE_HELPER _mlir_ciface_tla_get_icache_prefetch_status_aic
#endif

[aicore] __attribute__((always_inline)) int64_t TLA_ICACHE_HELPER()
{
    return AscendC::GetICachePreloadStatus();
}

#undef TLA_ICACHE_HELPER

} // extern "C"
#endif
