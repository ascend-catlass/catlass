// RUN: %tla_compile %s -o - | %filecheck %s
// Formats must be included before aligning the dynamic region's base.
// CHECK: module attributes
// CHECK-SAME: tla.ub_dynamic_base = 512 : i64
// CHECK-SAME: tla.ub_static_bytes = 264 : i64
// CHECK: llvm.mlir.global private @tla_ub_scratch
// CHECK-SAME: !llvm.array<264 x i8>
// CHECK: call @_mlir_ciface_tla_vf_debug_begin
// CHECK: hivm.vector_function
// CHECK: call @_mlir_ciface_tla_vf_debug_end
// CHECK-NOT: @tla_vf_format_

module {
  tla.func @vf_print_dynamic_ub(%value: i32) {
    %fixed = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<f32, ub, 256>
    %dynamic = "tla.dynamic_ub_base"() : () -> !tla.ptr<f32, ub, 256>
    "tla.vector"() ({
      "tla.vec.func"() ({
        tla.debug_print format "x={}"(%value) : (i32)
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    tla.return
  }
}
