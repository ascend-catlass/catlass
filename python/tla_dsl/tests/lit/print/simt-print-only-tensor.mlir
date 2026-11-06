// RUN: %tla_compile %s -o - | %filecheck %s
// CHECK-LABEL: func.func @print_only(
// CHECK-SAME: tla.simt_print.workspace
// CHECK: launch_func @print_only_vf_simt
// CHECK-SAME: args({{.*}}) : !llvm.ptr<1>, i64
// CHECK-LABEL: func.func @print_only_vf_simt(
// CHECK-SAME: memref<56xf32, 1>, %{{.*}}: i64)
// CHECK: call @_mlir_ciface_tla_simt_print_gm
// CHECK-NOT: tla.print_tensor

module {
  tla.func @print_only(%source: !tla.tensor<!tla.layout<!tla.shape<4,8>, !tla.stride<16,1>, !tla.shape<4,8>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>) {
    "tla.vector"() ({
      "tla.vec.func"() ({
        %count = arith.constant 20 : i64
        tla.print_tensor %source length = %count shape = [4, 8] : <!tla.layout<!tla.shape<4,8>, !tla.stride<16,1>, !tla.shape<4,8>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>, i64
      }) {mode = "simt", thread_block_dim = array<i64: 32, 1, 1>} : () -> ()
    }) : () -> ()
    tla.return
  }
}
