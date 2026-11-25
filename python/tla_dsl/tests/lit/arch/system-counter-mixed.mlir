// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @counter_both(%output: memref<2xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    "tla.cube"() ({
      %ticks = tla.arch.get_system_cycle -> i64
      memref.store %ticks, %output[%c0] : memref<2xi64>
    }) : () -> ()
    "tla.vector"() ({
      %ticks = tla.arch.get_system_cycle -> i64
      memref.store %ticks, %output[%c1] : memref<2xi64>
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @counter_both_mix_aic
// CHECK: call @tla_get_system_cycle() : () -> i64
// CHECK-LABEL: func.func @counter_both_mix_aiv
// CHECK: call @tla_get_system_cycle.vector() : () -> i64
