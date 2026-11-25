// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @counter_vector(%output: memref<1xi64>) {
    %c0 = arith.constant 0 : index
    "tla.vector"() ({
      %a = tla.arch.get_system_cycle -> i64
      %b = tla.arch.get_system_cycle -> i64
      %delta = arith.subi %b, %a : i64
      memref.store %delta, %output[%c0] : memref<1xi64>
    }) : () -> ()
    tla.return
  }
}

// CHECK: func.func private @tla_get_system_cycle.vector() -> i64
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-LABEL: func.func @counter_vector
// CHECK: %[[FIRST:.*]] = call @tla_get_system_cycle.vector() : () -> i64
// CHECK: %[[SECOND:.*]] = call @tla_get_system_cycle.vector() : () -> i64
// CHECK: {{(arith.subi|llvm.sub)}} %[[SECOND]], %[[FIRST]]
// CHECK-NOT: tla.arch.get_system_cycle
