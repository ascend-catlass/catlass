// RUN: %tla_compile %s -o - | %filecheck %s
// CHECK: tla.print.layout = "fifo,user"
// CHECK-LABEL: func.func @combined_channels(
// CHECK-SAME: %[[FIFO:arg[0-9]+]]: i64 {{.*}}tla.debug_print.workspace, tla.print_tensor.workspace
// CHECK: call @_mlir_ciface_tla_print_tensor_gm_f32(%[[FIFO]],
// CHECK: call @_mlir_ciface_tla_vf_debug_begin(%[[FIFO]])
// CHECK: call @_mlir_ciface_tla_vf_debug_end()

module {
  tla.func @combined_channels(%source: !tla.tensor<!tla.layout<!tla.shape<4>, !tla.stride<1>, !tla.shape<4>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, %value: i32) {
    "tla.vector"() ({
      %count = arith.constant 4 : i64
      tla.print_tensor %source length = %count shape = [4] : <!tla.layout<!tla.shape<4>, !tla.stride<1>, !tla.shape<4>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>, i64
      "tla.vec.func"() ({
        tla.debug_print format "value={}"(%value) : (i32)
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    tla.return
  }
}
