// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s

// CHECK: ave.hir.vci
// CHECK-SAME: <DECREASE>
// CHECK-SAME: vector<64xi32>
// CHECK: ave.hir.vsitofp
// CHECK: ave.hir.vadds
// CHECK-SAME: f32
// CHECK: ave.hir.masked_store
// CHECK-SAME: vector<64xf32>

!t = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @vector_arange_f32_decrease_lowering(
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c64, %dst_c1, %dst_c1] stride [%dst_c64, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c64] coord [%dst_c0, %dst_c0] : memref<64xf32, #hivm.address_space<ub>> -> !t
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t, !tla.shape<64>, !tla.coord<0>) -> !t
        %start = arith.constant 8 : i32
        %idx = "tla.arange"(%start) {order = "decrease"} : (i32) -> !tla.vector<64xf32>
        "tla.store"(%dst_tile, %idx) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!t, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
