// RUN: %tla_compile %s --mlir-print-ir-after=tla-lower-scalar-ops -o %t 2>&1 | %filecheck %s

// The ub-domain helper is core-specific (see data-sync-barrier-lowering.mlir),
// so only a mixed kernel reaches the two branches a single-core function
// cannot: the split AIC copy drops dsb<ub> as a hardware no-op, and the split
// AIV copy calls _mlir_ciface_tla_data_sync_barrier_ub_aiv. The scalar body is
// cloned into both halves, so one barrier covers both branches. The pipe
// barriers only give the two scopes content the later passes accept.

module {
  tla.func @mixed() {
    "tla.cube"() ({
      tla.pipe_barrier [#tla.pipe<all>]
    }) : () -> ()
    "tla.vector"() ({
      tla.pipe_barrier [#tla.pipe<all>]
    }) : () -> ()
    tla.data_sync_barrier <ub>
    tla.return
  }
}

// CHECK-LABEL: func.func @mixed_mix_aic()
// CHECK-NOT: call @_mlir_ciface_tla_data_sync_barrier
// CHECK-LABEL: func.func @mixed_mix_aiv()
// CHECK: call @_mlir_ciface_tla_data_sync_barrier_ub_aiv() : () -> ()
// CHECK-NOT: tla.data_sync_barrier
