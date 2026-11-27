#include "../common.h"

#include "kernel_operator.h"

// Bypass loads lower to hivm::LoadScalarOp. The pinned HIVM dialect only
// exposes a 64-bit bypass store. Use ccec to emit llvm.hivm.ST.DEV.u32
// through Ascend C so writing an Int32 leaves the adjacent GM word intact.
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

extern "C" {

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_store_bypass_dcache_i32(uint64_t address, int32_t value)
{
    // ccec marks the intrinsic inaccessiblememonly. Without these compiler
    // barriers it can reuse an earlier GM value for a load after the inlined
    // store. Hardware ordering still requires the caller's pipe waits or DSB.
    asm volatile("" ::: "memory");
    AscendC::WriteGmBypassDCache(reinterpret_cast<__gm__ int32_t*>(address), value);
    asm volatile("" ::: "memory");
}

} // extern "C"

#endif
