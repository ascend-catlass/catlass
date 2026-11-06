// RUN: %tla_compile %s -o - | %filecheck %s
// CHECK-LABEL: func.func @scalar_only(
// CHECK-SAME: tla.debug_print.workspace
// CHECK: call @_mlir_ciface_tla_vf_debug_begin
// CHECK: call @vector_region_0
// CHECK-SAME: hivm.vector_function
// CHECK: call @_mlir_ciface_tla_vf_debug_end
// CHECK-LABEL: func.func private @vector_region_0
// CHECK-SAME: hivm.vector_function
// CHECK: llvm.store
// CHECK: call @_mlir_ciface_tla_vf_printf
// CHECK-NOT: tla.debug_print format

module {
  tla.func @scalar_only(%arg0: i32) {
    "tla.vector"() ({
      "tla.vec.func"() ({
        tla.debug_print format "vf_scalar_only={}"(%arg0) : (i32)
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    tla.return
  }
}
