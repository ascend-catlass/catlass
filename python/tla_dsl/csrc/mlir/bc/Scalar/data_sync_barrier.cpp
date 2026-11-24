#include "../common.h"

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

/**
 * Data synchronization barrier over a storage domain (CANN
 * ``DataSyncBarrier<MemDsbT>``, ISA ``dsb``).
 *
 * The instruction takes one storage-domain immediate -- ALL (GM and UB), DDR
 * (GM), UB (UB), SEQ (sequencer, including the parameter buffer) -- and no
 * pipe, no flag id and no address. It is reachable ONLY through a bitcode
 * helper: the HIVM dialect has no ``dsb`` op (its barrier ops are
 * ``hivmave.membar`` and ``hivm.pipe_barrier``), and
 * ``llvm.call_intrinsic "llvm.hivm.DSB"`` is rejected by hivmc with "could not
 * find LLVM intrinsic" -- the DSB intrinsic is registered in the CCE clang, not
 * in the LLVM hivmc links against. tla-lower-scalar-ops therefore rewrites the
 * frontend op into a call to these helpers, ccec selects ``llvm.hivm.DSB`` here
 * (verified on a probe stub: four template instances
 * ``DataSyncBarrierImpl<(AscendC::MemDsbT)0..3>`` survive in the bitcode), and
 * ``--link-aicore-bitcode`` + hivmc-a5 inlining leaves a single instruction at
 * the call site.
 *
 * One no-arg helper per domain instead of one helper plus a runtime switch:
 * always_inline then costs exactly one instruction per call site, with no
 * branch and no parameter passing. The helpers are deliberately not pure --
 * CSE and DCE must not merge or drop repeated barriers.
 *
 * ``[aicore]`` because the scalar unit is present on both AIC and AIV. This
 * file lives in bc/Scalar/, which bc_compile.py compiles once per core type and
 * links into both meta-op bitcodes -- fine for gm_ub/gm/seq, whose bodies are
 * identical per core, and safe to share a name like the scalar round casts do.
 *
 * The UB domain is the one same-name-different-body exception: the AIC-compiled
 * body is empty (DataSyncBarrierImpl returns early for MemDsbT::UB on AIC)
 * while the AIV-compiled body issues the dsb, and hivmc links BOTH bitcodes
 * into each subcore compile, so a shared name could let the empty AIC body win
 * on AIV and silently drop the barrier. The AIV helper therefore lives in
 * bc/Vector/data_sync_barrier_ub.cpp as
 * ``_mlir_ciface_tla_data_sync_barrier_ub_aiv``, and AIC functions drop the op
 * entirely (dsb<ub> is a hardware no-op there); see TlaLowerScalarOpsPass and
 * the _mlir_ciface_tla_init_soc_state split in TlaPrologueEpiloguePass for the
 * same hazard.
 *
 * Guarded like scalar_round_cast.cpp: the CANN guard covers __NPU_ARCH__ 2201 /
 * 3002 / 3102 / 3510, but tla_dsl only targets 3510 today.
 */

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

extern "C" {

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_data_sync_barrier_gm_ub()
{
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_data_sync_barrier_gm()
{
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_data_sync_barrier_seq()
{
    AscendC::DataSyncBarrier<AscendC::MemDsbT::SEQ>();
}

} // extern "C"

#endif
