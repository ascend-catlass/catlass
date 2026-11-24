#include "../common.h"
#include "kernel_operator.h"

/**
 * AIV-side helper for the ``ub`` storage domain of ``tla.data_sync_barrier``.
 *
 * Lives in bc/Vector/ -- compiled only into meta_op.aiv.bc -- because the ub
 * domain is the one same-name-different-body hazard of the dsb family: the
 * AIC-compiled body of ``AscendC::DataSyncBarrier<MemDsbT::UB>`` is empty
 * (DataSyncBarrierImpl returns early on AIC) while this one issues the dsb, and
 * hivmc links BOTH meta-op bitcodes into each subcore compile, so a shared name
 * could let the empty AIC body win on AIV and silently drop the barrier. Same
 * hazard that split ``_mlir_ciface_tla_init_soc_state`` into _aic/_aiv (see
 * TlaPrologueEpiloguePass). tla-lower-scalar-ops calls this symbol from AIV
 * (and defensively MIX/unknown) functions and drops the op on AIC functions,
 * where dsb<ub> is a hardware no-op anyway; the gm_ub/gm/seq helpers stay in
 * bc/Scalar/ because their bodies are identical per core.
 *
 * Guarded like the Scalar stub: tla_dsl only targets 3510 today.
 */

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

extern "C" {

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_data_sync_barrier_ub_aiv()
{
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
}

} // extern "C"

#endif
