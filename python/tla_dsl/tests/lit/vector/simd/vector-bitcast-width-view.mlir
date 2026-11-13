// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!ivec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>

module {
  func.func @vector_bitcast_f32_i16(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<128xi16, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c64, %src_c1, %src_c1] stride [%src_c64, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c64] coord [%src_c0, %src_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 128 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c64, %dst_c1, %dst_c1] stride [%dst_c64, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c64] coord [%dst_c0, %dst_c0] : memref<128xi16, #hivm.address_space<ub>> -> !ivec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_shape = "tla.make_shape"() : () -> !tla.shape<128>
      %dst_tile = "tla.tile_view"(%dst, %dst_shape, %coord) : (!ivec, !tla.shape<128>, !tla.coord<0>) -> !ivec
      %v = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
      %bits = tla.bitcast %v : !tla.vector<64xf32> -> !tla.vector<128xi16>
      tla.store %dst_tile, %bits : !ivec, !tla.vector<128xi16>
    }) : () -> ()
    return
  }
}

// A width-changing reinterpretation emits no AVE instruction: it is exactly
// vector.bitcast, and nothing of tla.bitcast survives the pass.
// CHECK-LABEL: func.func private @vector_region_
// CHECK: vector.bitcast {{.*}} : vector<64xf32> to vector<128xi16>
// CHECK-NOT: tla.bitcast
