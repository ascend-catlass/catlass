// RUN: %tla_compile %s -o - | %filecheck %s
module {
  tla.func @prefetch_vec(%addr: i64) {
    "tla.vector"() ({
      %p = tla.inttoptr %addr : i64 -> !tla.ptr<i32, gm, 4>
      %delta = arith.constant -4 : i64
      tla.arch.dcache_prefetch %p, %delta : !tla.ptr<i32, gm, 4>
    }) : () -> ()
    tla.return
  }
}
// CHECK-LABEL: func.func @prefetch_vec
// CHECK: call @{{(_mlir_ciface_)?}}tla_dcache_prefetch_aiv
