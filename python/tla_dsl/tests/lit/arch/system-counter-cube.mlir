// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @counter_cube(%output: memref<1xi64>) {
    %c0 = arith.constant 0 : index
    "tla.cube"() ({
      %ticks = tla.arch.get_system_cycle -> i64
      memref.store %ticks, %output[%c0] : memref<1xi64>
    }) : () -> ()
    tla.return
  }
}

// CHECK: func.func private @tla_get_system_cycle() -> i64
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-LABEL: func.func @counter_cube
// CHECK: call @tla_get_system_cycle() : () -> i64
// CHECK-NOT: tla.arch.get_system_cycle
