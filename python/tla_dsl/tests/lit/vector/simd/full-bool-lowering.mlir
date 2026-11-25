// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

// tla.full accepts Bool as an element type: AscendC Duplicate<bool> works on
// b8, so the scalar rides the i8 carrier and the broadcast writes an i8 vector.
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.broadcast %{{.*}}_i8, %{{.*}} : i8, vector<256xi1> -> vector<256xi8>

!bvec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i1, ub, 1>>

module {
  func.func @full_bool(%dst: memref<256xi1, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %d = tla.tensor_desc %dst shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi1, #hivm.address_space<ub>> -> !bvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dt = "tla.tile_view"(%d, %shape, %coord) : (!bvec, !tla.shape<256>, !tla.coord<0>) -> !bvec
      %v = arith.constant 1 : i8
      %full = "tla.full"(%v) : (i8) -> !tla.vector<256xi1>
      "tla.store"(%dt, %full) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!bvec, !tla.vector<256xi1>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }
}
