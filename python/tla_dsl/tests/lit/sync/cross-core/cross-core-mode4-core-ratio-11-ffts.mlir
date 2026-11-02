// RUN: %tla_compile %s -o - | %filecheck %s
//
// CV 1:1 (tla.core_ratio=[1,1]) must lower mode-4 cross flags to FFTS Imm
// (SET.CROSS.CORE / WAIT.FLAG.DEV), not INTRA. Config uses GetffstMsg(0x2, id).

module {
  tla.func @mode4_core_ratio_11() attributes {tla.core_ratio = array<i64: 1, 1>} {
    %flag = tla.cross_flag "ready" -> !tla.cross_flag<4>
    "tla.cube"() ({
      tla.cross_core_set_flag %flag {aiv_id = 0 : i64, pipe = #tla.pipe<fix>} : !tla.cross_flag<4>
    }) : () -> ()
    "tla.vector"() ({
      tla.cross_core_wait_flag %flag {aiv_id = 0 : i64, pipe = #tla.pipe<scalar>} : !tla.cross_flag<4>
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @mode4_core_ratio_11_mix_aic
// CHECK: "hivm.intr.hivm.SET.CROSS.CORE"(
// CHECK-SAME: pipe = 10 : i64
// CHECK-NOT: SET.INTRA.BLOCK
// CHECK-LABEL: func.func @mode4_core_ratio_11_mix_aiv
// CHECK: "hivm.intr.hivm.WAIT.FLAG.DEV.PIPE.IMM"() <{flag_id = 0 : i64, pipe = 0 : i64}>
// CHECK-NOT: WAIT.INTRA.BLOCK
// CHECK-NOT: hivm.hir.get_sub_block_idx
// CHECK-NOT: !tla.cross_flag
// CHECK-NOT: tla.cross_core
