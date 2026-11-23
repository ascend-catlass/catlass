// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!uvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 2>>

module {
  func.func @squeeze_u16_store_reg(
      %src_memref: memref<128xui16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xui16, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c128, %src_c1, %src_c1] stride [%src_c128, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c128] coord [%src_c0, %src_c0] : memref<128xui16, #hivm.address_space<ub>> -> !uvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c128 = arith.constant 128 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c128, %dst_c1, %dst_c1] stride [%dst_c128, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c128] coord [%dst_c0, %dst_c0] : memref<128xui16, #hivm.address_space<ub>> -> !uvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!uvec, !tla.shape<128>, !tla.coord<0>) -> !uvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!uvec, !tla.shape<128>, !tla.coord<0>) -> !uvec
      %mask = "tla.create_mask"() {pattern = "VL8", dtype = ui16} : () -> !tla.mask<128>
      %v0 = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!uvec) -> !tla.vector<128xui16>
      %v1 = "tla.squeeze"(%v0, %mask) {store_reg = true} : (!tla.vector<128xui16>, !tla.mask<128>) -> !tla.vector<128xui16>
      "tla.store"(%dst_tile, %v1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!uvec, !tla.vector<128xui16>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK: call @vsqueeze_uint16_t_store_reg(
// CHECK-NOT: tla.squeeze
