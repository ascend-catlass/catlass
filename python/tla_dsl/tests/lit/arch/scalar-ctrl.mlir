// RUN: %tla_compile %s -o - | %filecheck %s
module {
  tla.func @ctrl_reads() {
    "tla.vector"() ({
      %one = arith.constant 9007199254740992 : i64
      tla.arch.set_ctrl %one {mask = 9007199254740992 : i64}
      %v = tla.arch.get_ctrl {lo = 0 : i32, hi = 63 : i32}
      tla.arch.set_ctrl %v {mask = 9007199254740992 : i64}
    }) : () -> ()
    tla.return
  }
// CHECK-LABEL: func.func @ctrl_reads
// CHECK: GET.CTRL
// CHECK: SET.CTRL
// CHECK: GET.CTRL
// CHECK: SET.CTRL

// ---- Bit 48 writes maintain the atomic cache.
  tla.func @ctrl_atomic_saturation() {
    "tla.vector"() ({
      %bits = arith.constant 0 : i64
      tla.arch.set_ctrl %bits {mask = 281474976710656 : i64}
    }) : () -> ()
    tla.return
  }
}
// CHECK-LABEL: func.func @ctrl_atomic_saturation
// CHECK: hivm.hir.dcci(<all_cache_lines>, <atomic>)
// CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
// CHECK: SET.CTRL
