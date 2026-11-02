// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @cross_flags() {
    "tla.vector"() ({
      %idx = tla.arch.block_idx -> i32
      %zero = arith.constant 0 : i32
      %condition = arith.cmpi eq, %idx, %zero : i32
      %direct = tla.cross_flag "direct" -> !tla.cross_flag<2>
      %ping = tla.cross_flag "ping" -> !tla.cross_flag<2>
      %pong = tla.cross_flag "pong" -> !tla.cross_flag<2>
      tla.cross_core_set_flag %direct {pipe = #tla.pipe<fix>} : !tla.cross_flag<2>
      tla.cross_core_wait_flag %direct {pipe = #tla.pipe<vector>} : !tla.cross_flag<2>
      %selected = scf.if %condition -> (!tla.cross_flag<2>) {
        scf.yield %ping : !tla.cross_flag<2>
      } else {
        scf.yield %pong : !tla.cross_flag<2>
      }
      tla.cross_core_set_flag %selected {pipe = #tla.pipe<mte3>} : !tla.cross_flag<2>
      tla.cross_core_wait_flag %selected {pipe = #tla.pipe<mte1>} : !tla.cross_flag<2>
    }) : () -> ()
    "tla.cube"() ({
    }) : () -> ()
    tla.return
  }

  tla.func @device_modes() {
    "tla.vector"() ({
      %mode0 = tla.cross_flag "mode0" -> !tla.cross_flag<0>
      %mode1 = tla.cross_flag "mode1" -> !tla.cross_flag<1>
      tla.cross_core_set_flag %mode0 {pipe = #tla.pipe<mte3>} : !tla.cross_flag<0>
      tla.cross_core_wait_flag %mode0 {pipe = #tla.pipe<scalar>} : !tla.cross_flag<0>
      tla.cross_core_set_flag %mode1 {pipe = #tla.pipe<mte3>} : !tla.cross_flag<1>
      tla.cross_core_wait_flag %mode1 {pipe = #tla.pipe<scalar>} : !tla.cross_flag<1>
    }) : () -> ()
    "tla.cube"() ({
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @cross_flags_mix_aiv
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCKI.mode"() <{pipe = 10 : i64, sync_id = 0 : i64}>
// CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCKI.mode"() <{pipe = 1 : i64, sync_id = 0 : i64}>
// CHECK: cf.cond_br
// CHECK: "hivm.intr.hivm.SET.INTRA.BLOCK.mode"({{.*}}) <{pipe = 5 : i64}>
// CHECK: "hivm.intr.hivm.WAIT.INTRA.BLOCK.mode"({{.*}}) <{pipe = 3 : i64}>
// CHECK-NOT: hivm.hir.sync_block
// CHECK-NOT: !tla.cross_flag
// CHECK-NOT: tla.cross_core

// CHECK-LABEL: func.func @device_modes_mix_aiv
// CHECK: "hivm.intr.hivm.SET.CROSS.CORE"
// CHECK-SAME: pipe = 5 : i64
// CHECK: "hivm.intr.hivm.WAIT.FLAG.DEV.PIPE.IMM"() <{flag_id = 1 : i64, pipe = 0 : i64}>
// CHECK: "hivm.intr.hivm.SET.CROSS.CORE"
// CHECK-SAME: pipe = 5 : i64
// CHECK: "hivm.intr.hivm.WAIT.FLAG.DEV.PIPE.IMM"() <{flag_id = 2 : i64, pipe = 0 : i64}>
// CHECK-NOT: hivm.hir.sync_block
// CHECK-NOT: !tla.cross_flag
// CHECK-NOT: tla.cross_core
