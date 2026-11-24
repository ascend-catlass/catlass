// RUN: %tla_compile %s --mlir-print-ir-after=tla-lower-scalar-ops -o %t 2>&1 | %filecheck %s

// tla.data_sync_barrier lowers through the same bitcode-helper route as the
// scalar round casts: one private ALWAYS_INLINE declaration per storage domain,
// then one call per barrier. No tla op may survive the pass. The ub domain is
// core-specific (_aiv, defined only in meta_op.aiv.bc) because the AIC-compiled
// body of the shared name would be empty and could win the link on AIV; the
// lowering drops the op on AIC functions instead.

module {
  tla.func @data_sync_barrier() {
    "tla.vector"() ({
      tla.data_sync_barrier <gm_ub>
      tla.data_sync_barrier <gm>
      tla.data_sync_barrier <ub>
      tla.data_sync_barrier <seq>
    }) : () -> ()
    tla.return
  }
}

// CHECK-DAG: func.func private @_mlir_ciface_tla_data_sync_barrier_gm_ub() attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>}
// CHECK-DAG: func.func private @_mlir_ciface_tla_data_sync_barrier_gm() attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>}
// CHECK-DAG: func.func private @_mlir_ciface_tla_data_sync_barrier_seq() attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>}
// CHECK-DAG: func.func private @_mlir_ciface_tla_data_sync_barrier_ub_aiv() attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>}
// CHECK: call @_mlir_ciface_tla_data_sync_barrier_gm_ub() : () -> ()
// CHECK: call @_mlir_ciface_tla_data_sync_barrier_gm() : () -> ()
// CHECK: call @_mlir_ciface_tla_data_sync_barrier_ub_aiv() : () -> ()
// CHECK: call @_mlir_ciface_tla_data_sync_barrier_seq() : () -> ()
// CHECK-NOT: tla.data_sync_barrier
