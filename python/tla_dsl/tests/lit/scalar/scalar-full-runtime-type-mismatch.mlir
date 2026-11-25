// RUN: not %tla_compile %s -o - 2>&1 | %filecheck %s --check-prefix=CHECK-ERR
//
// Diagnostic: a runtime tla.full scalar must be the result element type or its
// broadcast carrier; the op verifier rejects anything else before lowering.

!t = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @scalar_full_type_mismatch(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>,
      %val: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!t, !tla.shape<64>, !tla.coord<0>) -> !t
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t, !tla.shape<64>, !tla.coord<0>) -> !t
        %reg = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!t) -> !tla.vector<64xf32>
        %full = "tla.full"(%val) : (i32) -> !tla.vector<64xf32>
        %out = "tla.add"(%reg, %full) : (!tla.vector<64xf32>, !tla.vector<64xf32>) -> !tla.vector<64xf32>
        "tla.store"(%dst_tile, %out) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!t, !tla.vector<64xf32>) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}
// CHECK-ERR: 'tla.full' op scalar source type 'i32' is not a valid carrier for 'f32'
