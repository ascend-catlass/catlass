// RUN: %tla_compile %s -o - | %filecheck %s
// CHECK: tla.simt_print.formats = "76665f7363616c61725f6f6e6c793d256400756e7369676e65643d257500"
// CHECK: func.func private @_mlir_ciface_tla_simt_printf
// CHECK-SAME: hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_callable
// CHECK-LABEL: func.func @scalar_only(
// CHECK-SAME: %[[VALUE:.*]]: i32, %[[CONTEXT:.*]]: i64
// CHECK-SAME: tla.simt_print.workspace
// CHECK: launch_func @scalar_only_vf_simt
// CHECK-SAME: args(%[[VALUE]], %[[CONTEXT]]) : i32, i64
// CHECK-LABEL: func.func @scalar_only_vf_simt(
// CHECK-SAME: %{{.*}}: i32, %[[CONTEXT:.*]]: i64)
// CHECK: call @_mlir_ciface_tla_simt_printf(%[[CONTEXT]],
// CHECK: llvm.zext %arg0 : i32 to i64
// CHECK: call @_mlir_ciface_tla_simt_printf
// CHECK-NOT: tla.debug_print

module {
  tla.func @scalar_only(%arg0: i32) {
    "tla.vector"() ({
      "tla.vec.func"() ({
        tla.debug_print format "vf_scalar_only={}"(%arg0) : (i32)
        %unsigned = builtin.unrealized_conversion_cast %arg0 : i32 to ui32
        tla.debug_print format "unsigned={}"(%unsigned) : (ui32)
      }) {mode = "simt", thread_block_dim = array<i64: 32, 1, 1>} : () -> ()
    }) : () -> ()
    tla.return
  }
}
