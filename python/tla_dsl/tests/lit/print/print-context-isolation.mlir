// RUN: %tla_compile %s -o - | %filecheck %s
// An unrelated entry must not inherit another kernel's print ABI.
module {
  tla.func @with_print(%value: i32) {
    "tla.vector"() ({ tla.debug_print %value : i32 }) : () -> ()
    tla.return
  }
  tla.func @without_print(%value: i32) {
    "tla.vector"() ({ %unused = arith.addi %value, %value : i32 }) : () -> ()
    tla.return
  }
}
// CHECK-LABEL: func.func @with_print(
// CHECK-SAME: tla.debug_print.workspace
// CHECK-LABEL: func.func @without_print(%arg0: i32)
// CHECK-NOT: tla.debug_print.workspace
// CHECK-NOT: call @_mlir_ciface_tla_printf
// CHECK: return
