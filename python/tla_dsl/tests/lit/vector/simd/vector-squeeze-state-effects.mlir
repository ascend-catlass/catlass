// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s
!hvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>
!uvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 2>>

module {
  func.func @squeeze_effects(
      %src_memref: memref<128xi16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xi16, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c128, %src_c1, %src_c1] stride [%src_c128, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c128] coord [%src_c0, %src_c0] : memref<128xi16, #hivm.address_space<ub>> -> !hvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c128 = arith.constant 128 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c128, %dst_c1, %dst_c1] stride [%dst_c128, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c128] coord [%dst_c0, %dst_c0] : memref<128xi16, #hivm.address_space<ub>> -> !hvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
      %mask = "tla.create_mask"() {pattern = "VL8", dtype = i16} : () -> !tla.mask<128>
      %v0 = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!hvec) -> !tla.vector<128xi16>
      %v1 = "tla.squeeze"(%v0, %mask) : (!tla.vector<128xi16>, !tla.mask<128>) -> !tla.vector<128xi16>
      %unused_a = "tla.squeeze"(%v0, %mask) {store_reg = true} : (!tla.vector<128xi16>, !tla.mask<128>) -> !tla.vector<128xi16>
      %unused_b = "tla.squeeze"(%v0, %mask) {store_reg = true} : (!tla.vector<128xi16>, !tla.mask<128>) -> !tla.vector<128xi16>
      "tla.store"(%dst_tile, %v0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!hvec, !tla.vector<128xi16>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

}

// CHECK-NOT: call @vsqueeze_int16_t(
// CHECK-COUNT-2: call @vsqueeze_int16_t_store_reg(
// CHECK-NOT: call @vsqueeze_int16_t
