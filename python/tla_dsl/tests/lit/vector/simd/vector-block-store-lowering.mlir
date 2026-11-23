// RUN: %tla_compile %s -o %t
// RUN: %filecheck %s --check-prefix=BLOCK_STORE < %t
// RUN: %filecheck %s --check-prefix=BLOCK_STORE_HALF < %t
// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --check-prefix=BLOCK_STORE_TAIL

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!hvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>

module {
  // ----- f32 block store -----
  func.func @block_store_lowering(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c64, %src_c1, %src_c1] stride [%src_c64, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c64] coord [%src_c0, %src_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c64, %dst_c1, %dst_c1] stride [%dst_c64, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c64] coord [%dst_c0, %dst_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
        %loaded = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
        %c32 = arith.constant 32 : i32
        "tla.store"(%dst_tile, %loaded, %c32) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> : (!fvec, !tla.vector<64xf32>, i32) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
  // ----- f16 block store -----
  func.func @block_store_lowering_half(
      %src_memref: memref<128xf16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<128>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
        %loaded = tla.load %src_tile : !hvec -> !tla.vector<128xf16>
        %c32 = arith.constant 32 : i32
        "tla.store"(%dst_tile, %loaded, %c32) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> : (!hvec, !tla.vector<128xf16>, i32) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }

  // A dynamic destination extent must feed the preg argument of the
  // block-stride library call when the store mask is omitted.
  func.func @block_store_dynamic_dest_tail(
      %extent: index,
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %extent] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vector"() ({
      "tla.vec.func"() ({
        %shape = "tla.make_shape"() : () -> !tla.shape<64>
        %coord = "tla.make_coord"() : () -> !tla.coord<0>
        %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
        %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
        %loaded = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
        %c32 = arith.constant 32 : i32
        "tla.store"(%dst_tile, %loaded, %c32) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> : (!fvec, !tla.vector<64xf32>, i32) -> ()
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    return
  }
}

// BLOCK_STORE-LABEL: block_store_lowering
// BLOCK_STORE: store_with_stride_float
// BLOCK_STORE-NOT: tla.store
// BLOCK_STORE: return

// BLOCK_STORE_HALF-LABEL: block_store_lowering_half
// BLOCK_STORE_HALF: store_with_stride_half
// BLOCK_STORE_HALF-NOT: tla.store
// BLOCK_STORE_HALF: return

// BLOCK_STORE_TAIL-LABEL: func.func @block_store_dynamic_dest_tail
// BLOCK_STORE_TAIL-LABEL: func.func private @vector_region_
// BLOCK_STORE_TAIL: %[[TAIL:.*]], %{{.*}} = ave.hir.plt
// BLOCK_STORE_TAIL: call @store_with_stride_float({{.*}}, {{.*}}, {{.*}}, %[[TAIL]])
