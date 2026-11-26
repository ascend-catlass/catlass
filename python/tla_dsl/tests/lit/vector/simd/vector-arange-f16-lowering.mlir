// RUN: %tla_compile %s -o - --mlir-print-ir-after=tla-vector-region-simd 2>&1 | %filecheck %s

// CHECK: ave.hir.vci
// CHECK-SAME: <INCREASE>
// CHECK-SAME: vector<128xi16>
// CHECK: ave.hir.vsitofp
// CHECK: ave.hir.vadds
// CHECK-SAME: f16
// CHECK: ave.hir.masked_store
// CHECK-SAME: vector<128xf16>

!t = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>

module {
  func.func @vector_arange_f16_lowering(
      %dst_memref: memref<128xf16, #hivm.address_space<ub>>) {
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c128 = arith.constant 128 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c128, %dst_c1, %dst_c1] stride [%dst_c128, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c128] coord [%dst_c0, %dst_c0] : memref<128xf16, #hivm.address_space<ub>> -> !t
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<128>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t, !tla.shape<128>, !tla.coord<0>) -> !t
        %start = arith.constant 3 : i16
        %idx = "tla.arange"(%start) {order = "increase"} : (i16) -> !tla.vector<128xf16>
        "tla.store"(%dst_tile, %idx) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!t, !tla.vector<128xf16>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
