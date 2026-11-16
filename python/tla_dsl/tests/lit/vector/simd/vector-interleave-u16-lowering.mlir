// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!fvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 4>>

module {
  func.func @vector_interleave_ui16(
      %src0_memref: memref<128xui16, #hivm.address_space<ub>>,
      %src1_memref: memref<128xui16, #hivm.address_space<ub>>,
      %dst0_memref: memref<128xui16, #hivm.address_space<ub>>,
      %dst1_memref: memref<128xui16, #hivm.address_space<ub>>) {
    %src0_c0 = arith.constant 0 : index
    %src0_c1 = arith.constant 1 : index
    %src0_c128 = arith.constant 128 : index
    %src0 = tla.tensor_desc %src0_memref shape [%src0_c1, %src0_c128, %src0_c1, %src0_c1] stride [%src0_c128, %src0_c1, %src0_c1, %src0_c1] origin_shape [%src0_c1, %src0_c128] coord [%src0_c0, %src0_c0] : memref<128xui16, #hivm.address_space<ub>> -> !fvec
    %src1_c0 = arith.constant 0 : index
    %src1_c1 = arith.constant 1 : index
    %src1_c128 = arith.constant 128 : index
    %src1 = tla.tensor_desc %src1_memref shape [%src1_c1, %src1_c128, %src1_c1, %src1_c1] stride [%src1_c128, %src1_c1, %src1_c1, %src1_c1] origin_shape [%src1_c1, %src1_c128] coord [%src1_c0, %src1_c0] : memref<128xui16, #hivm.address_space<ub>> -> !fvec
    %dst0_c0 = arith.constant 0 : index
    %dst0_c1 = arith.constant 1 : index
    %dst0_c128 = arith.constant 128 : index
    %dst0 = tla.tensor_desc %dst0_memref shape [%dst0_c1, %dst0_c128, %dst0_c1, %dst0_c1] stride [%dst0_c128, %dst0_c1, %dst0_c1, %dst0_c1] origin_shape [%dst0_c1, %dst0_c128] coord [%dst0_c0, %dst0_c0] : memref<128xui16, #hivm.address_space<ub>> -> !fvec
    %dst1_c0 = arith.constant 0 : index
    %dst1_c1 = arith.constant 1 : index
    %dst1_c128 = arith.constant 128 : index
    %dst1 = tla.tensor_desc %dst1_memref shape [%dst1_c1, %dst1_c128, %dst1_c1, %dst1_c1] stride [%dst1_c128, %dst1_c1, %dst1_c1, %dst1_c1] origin_shape [%dst1_c1, %dst1_c128] coord [%dst1_c0, %dst1_c0] : memref<128xui16, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src0_tile = "tla.tile_view"(%src0, %shape, %coord) : (!fvec, !tla.shape<128>, !tla.coord<0>) -> !fvec
      %src1_tile = "tla.tile_view"(%src1, %shape, %coord) : (!fvec, !tla.shape<128>, !tla.coord<0>) -> !fvec
      %dst0_tile = "tla.tile_view"(%dst0, %shape, %coord) : (!fvec, !tla.shape<128>, !tla.coord<0>) -> !fvec
      %dst1_tile = "tla.tile_view"(%dst1, %shape, %coord) : (!fvec, !tla.shape<128>, !tla.coord<0>) -> !fvec
      %v0 = tla.load %src0_tile : !fvec -> !tla.vector<128xui16>
      %v1 = tla.load %src1_tile : !fvec -> !tla.vector<128xui16>
      %r0, %r1 = tla.interleave %v0, %v1 : !tla.vector<128xui16>, !tla.vector<128xui16> -> !tla.vector<128xui16>, !tla.vector<128xui16>
      tla.store %dst0_tile, %r0 : !fvec, !tla.vector<128xui16>
      tla.store %dst1_tile, %r1 : !fvec, !tla.vector<128xui16>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vintlv
// CHECK-NOT: tla.interleave
