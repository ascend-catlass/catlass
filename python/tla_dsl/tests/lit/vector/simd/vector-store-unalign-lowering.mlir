// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @vector_store_unalign(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c64, %src_c1, %src_c1] stride [%src_c64, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c64] coord [%src_c0, %src_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst_c37 = arith.constant 37 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c64, %dst_c1, %dst_c1] stride [%dst_c64, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c37] coord [%dst_c0, %dst_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %loaded = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
      tla.store %dst_tile, %loaded {unaligned_ub_access} : !fvec, !tla.vector<64xf32>
    }) : () -> ()
    return
  }

  // The source has 10 valid lanes and the destination has 20. The unaligned
  // store uses the destination extent of 20 lanes.
  func.func @vector_store_unalign_source_dest_tail(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c20 = arith.constant 20 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c10] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c20] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %loaded = tla.load %src_tile : !fvec -> !tla.vector<10xf32>
      tla.store %dst_tile, %loaded {unaligned_ub_access} : !fvec, !tla.vector<10xf32>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vload
// CHECK: "hivm_regbaseintrins.intr.hivm.init.vector.align.data"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.f32"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
// CHECK-NOT: tla.store

// CHECK-LABEL: func.func private @vector_region_
// CHECK: arith.constant 20 : index
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.f32"
