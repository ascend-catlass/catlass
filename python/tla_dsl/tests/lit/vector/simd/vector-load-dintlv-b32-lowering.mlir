// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!fsrc = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @vector_load_dintlv_b32(
      %src_memref: memref<128xf32, #hivm.address_space<ub>>,
      %dst0_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst1_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c128, %src_c1, %src_c1] stride [%src_c128, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c128] coord [%src_c0, %src_c0] : memref<128xf32, #hivm.address_space<ub>> -> !fsrc
    %dst0_c0 = arith.constant 0 : index
    %dst0_c1 = arith.constant 1 : index
    %dst0_c64 = arith.constant 64 : index
    %dst0 = tla.tensor_desc %dst0_memref shape [%dst0_c1, %dst0_c64, %dst0_c1, %dst0_c1] stride [%dst0_c64, %dst0_c1, %dst0_c1, %dst0_c1] origin_shape [%dst0_c1, %dst0_c64] coord [%dst0_c0, %dst0_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst1_c0 = arith.constant 0 : index
    %dst1_c1 = arith.constant 1 : index
    %dst1_c64 = arith.constant 64 : index
    %dst1 = tla.tensor_desc %dst1_memref shape [%dst1_c1, %dst1_c64, %dst1_c1, %dst1_c1] stride [%dst1_c64, %dst1_c1, %dst1_c1, %dst1_c1] origin_shape [%dst1_c1, %dst1_c64] coord [%dst1_c0, %dst1_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %shape128 = "tla.make_shape"() : () -> !tla.shape<128>
      %shape64 = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape128, %coord) : (!fsrc, !tla.shape<128>, !tla.coord<0>) -> !fsrc
      %dst0_tile = "tla.tile_view"(%dst0, %shape64, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst1_tile = "tla.tile_view"(%dst1, %shape64, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %v0, %v1 = tla.load %src_tile {load_dist = #tla.load_dist<dintlv_b32>} : !fsrc -> !tla.vector<64xf32>, !tla.vector<64xf32>
      tla.store %dst0_tile, %v0 : !fvec, !tla.vector<64xf32>
      tla.store %dst1_tile, %v1 : !fvec, !tla.vector<64xf32>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: hivm_regbaseintrins.intr.hivm.vldsx2.v64f32
// CHECK-SAME: !llvm.struct<(vector<64xf32>, vector<64xf32>)>
// CHECK: llvm.extractvalue {{.*}}[0]
// CHECK: llvm.extractvalue {{.*}}[1]
