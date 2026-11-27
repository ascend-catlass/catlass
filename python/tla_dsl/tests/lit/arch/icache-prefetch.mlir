// RUN: %tla_compile %s -o - | %filecheck %s
module {
  tla.func @prefetch_poll(%flag: i1) {
    "tla.vector"() ({
      scf.if %flag {
        tla.arch.icache_prefetch 0
        %first = tla.arch.get_icache_prefetch_status -> i64
        %second = tla.arch.get_icache_prefetch_status -> i64
      }
    }) : () -> ()
    tla.return
  }
}
// CHECK-LABEL: func.func @prefetch_poll
// CHECK: %[[UNITS:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: cf.cond_br %arg0, ^[[TAKEN:bb[0-9]+]],
// CHECK: ^[[TAKEN]]:
// CHECK: call @_mlir_ciface_tla_icache_prefetch_aiv(%[[UNITS]])
// CHECK: call @_mlir_ciface_tla_get_icache_prefetch_status_aiv
// CHECK: call @_mlir_ciface_tla_get_icache_prefetch_status_aiv
