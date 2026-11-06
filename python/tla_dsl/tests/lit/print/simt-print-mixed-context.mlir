// RUN: %tla_compile %s -o - | %filecheck %s --check-prefixes=CHECK,RATIO2
// RUN: sed 's/array<i64: 1, 2>/array<i64: 1, 1>/' %s | %tla_compile - -o - | %filecheck %s --check-prefixes=CHECK,RATIO1
// CHECK-LABEL: func.func @mixed_context_mix_aic(
// CHECK-SAME: %arg0: i32, %arg1: i64 {{.*}}tla.simt_print.workspace
// CHECK-SAME: %arg2: i64 {{.*}}tla.debug_print.workspace
// RATIO1-SAME: tla.core_ratio = array<i64: 1, 1>
// RATIO2-SAME: tla.core_ratio = array<i64: 1, 2>
// CHECK-LABEL: func.func @mixed_context_mix_aiv(
// CHECK-SAME: %arg0: i32, %arg1: i64 {{.*}}tla.simt_print.workspace
// CHECK-SAME: %arg2: i64 {{.*}}tla.debug_print.workspace
// RATIO1-SAME: tla.core_ratio = array<i64: 1, 1>
// RATIO2-SAME: tla.core_ratio = array<i64: 1, 2>
// CHECK: launch_func @mixed_context_mix_aiv_vf_simt
// CHECK-SAME: args(%arg0, %arg1) : i32, i64

module {
  tla.func @mixed_context(%value: i32) attributes {tla.core_ratio = array<i64: 1, 2>} {
    "tla.cube"() ({
      tla.debug_print format "cube={}"(%value) : (i32)
    }) : () -> ()
    "tla.vector"() ({
      "tla.vec.func"() ({
        tla.debug_print format "simd={}"(%value) : (i32)
      }) {mode = "simd"} : () -> ()
      "tla.vec.func"() ({
        tla.debug_print format "simt={}"(%value) : (i32)
      }) {mode = "simt", thread_block_dim = array<i64: 32, 1, 1>} : () -> ()
    }) : () -> ()
    tla.return
  }
}
