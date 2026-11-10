// Verify tla-prologue-epilogue injects InitSocState via core-specific BC
// helpers at kernel entry (InitSoc only; no CTRL[60]/[48]/[56] writes).
// AIC and AIV must use distinct symbols. No user-facing tla.init_soc_state op.
//
// RUN: %tla_compile %s -o - | %filecheck %s

module {

  tla.func @init_soc_cube() {
    "tla.cube"() ({
      tla.pipe_barrier [#tla.pipe<all>]
    }) : () -> ()
    tla.return
  }

  tla.func @init_soc_vector() {
    "tla.vector"() ({
      tla.pipe_barrier [#tla.pipe<all>]
    }) : () -> ()
    tla.return
  }
}

// Declarations are emitted before entry functions.
// CHECK-DAG: func.func private @_mlir_ciface_tla_init_soc_state_aic(){{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-DAG: func.func private @_mlir_ciface_tla_init_soc_state_aiv(){{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-LABEL: func.func @init_soc_cube
// CHECK: call @_mlir_ciface_tla_init_soc_state_aic()
// CHECK-NOT: hivm.hir.set_ctrl {{.*}} at ctrl[60]
// CHECK-NOT: hivm.hir.set_ctrl {{.*}} at ctrl[48]
// CHECK-LABEL: func.func @init_soc_vector
// CHECK: call @_mlir_ciface_tla_init_soc_state_aiv()
// CHECK-NOT: hivm.hir.set_ctrl {{.*}} at ctrl[60]
// CHECK-NOT: hivm.hir.set_ctrl {{.*}} at ctrl[48]
// CHECK-NOT: hivm.hir.set_ctrl {{.*}} at ctrl[56]
// CHECK-NOT: tla.init_soc_state
// CHECK-NOT: call @_mlir_ciface_tla_init_soc_state()
