// RUN: %tla_compile %s -o - | %filecheck %s
module {
  tla.func @prefetch_poll(%flag: i1) {
    "tla.vector"() ({
      scf.if %flag {
        tla.arch.icache_prefetch 7
        %first = tla.arch.get_icache_prefetch_status -> i64
        %second = tla.arch.get_icache_prefetch_status -> i64
      }
    }) : () -> ()
    tla.return
  }
}
// CHECK-LABEL: func.func @prefetch_poll
// CHECK: cf.cond_br %arg0, ^[[TAKEN:bb[0-9]+]],
// CHECK: ^[[TAKEN]]:
// CHECK: call @_mlir_ciface_tla_icache_prefetch_aiv
// CHECK: call @_mlir_ciface_tla_get_icache_prefetch_status_aiv
// CHECK: call @_mlir_ciface_tla_get_icache_prefetch_status_aiv
