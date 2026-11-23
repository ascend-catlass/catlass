// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s
// AscendC: PACK_B32 is same-type (f32→f32 C API; f16→f16 after prior 32→16 cast).

!src_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!src_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!dst_f16 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!dst_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @store_pack_b32(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c64, %src_c1, %src_c1] stride [%src_c64, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c64] coord [%src_c0, %src_c0] : memref<64xf32, #hivm.address_space<ub>> -> !src_f32
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c64, %dst_c1, %dst_c1] stride [%dst_c64, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c64] coord [%dst_c0, %dst_c0] : memref<64xf32, #hivm.address_space<ub>> -> !dst_f32
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!src_f32, !tla.shape<64>, !tla.coord<0>) -> !src_f32
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!dst_f32, !tla.shape<64>, !tla.coord<0>) -> !dst_f32
      %loaded = tla.load %src_tile : !src_f32 -> !tla.vector<64xf32>
      tla.store %dst_tile, %loaded {store_dist = #tla.store_dist<pack_b32>} : !dst_f32, !tla.vector<64xf32>
    }) : () -> ()
    return
  }

  // Post-cast layout: f16 register + PACK_B32 (geometry 2) into a shorter tile.
  func.func @store_pack_b32_f16(
      %src_memref: memref<128xf16, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !src_f16
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf16, #hivm.address_space<ub>> -> !dst_f16
    "tla.vec.func"() ({
      %src_shape = "tla.make_shape"() : () -> !tla.shape<128>
      %dst_shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %src_shape, %coord) : (!src_f16, !tla.shape<128>, !tla.coord<0>) -> !src_f16
      %dst_tile = "tla.tile_view"(%dst, %dst_shape, %coord) : (!dst_f16, !tla.shape<64>, !tla.coord<0>) -> !dst_f16
      %loaded = tla.load %src_tile : !src_f16 -> !tla.vector<128xf16>
      tla.store %dst_tile, %loaded {store_dist = #tla.store_dist<pack_b32>} : !dst_f16, !tla.vector<128xf16>
    }) : () -> ()
    return
  }

  func.func @store_dynamic_dest_tail(
      %extent: index,
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !src_f32
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %extent] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !dst_f32
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!src_f32, !tla.shape<64>, !tla.coord<0>) -> !src_f32
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!dst_f32, !tla.shape<64>, !tla.coord<0>) -> !dst_f32
      %loaded = tla.load %src_tile : !src_f32 -> !tla.vector<64xf32>
      tla.store %dst_tile, %loaded {store_dist = #tla.store_dist<pack_b32>} : !dst_f32, !tla.vector<64xf32>
    }) : () -> ()
    return
  }

  func.func @store_dynamic_dest_tail_f16(
      %extent: index,
      %src_memref: memref<128xf16, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !src_f16
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %extent] coord [%c0, %c0] : memref<64xf16, #hivm.address_space<ub>> -> !dst_f16
    "tla.vec.func"() ({
      %src_shape = "tla.make_shape"() : () -> !tla.shape<128>
      %dst_shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %src_shape, %coord) : (!src_f16, !tla.shape<128>, !tla.coord<0>) -> !src_f16
      %dst_tile = "tla.tile_view"(%dst, %dst_shape, %coord) : (!dst_f16, !tla.shape<64>, !tla.coord<0>) -> !dst_f16
      %loaded = tla.load %src_tile : !src_f16 -> !tla.vector<128xf16>
      tla.store %dst_tile, %loaded {store_dist = #tla.store_dist<pack_b32>} : !dst_f16, !tla.vector<128xf16>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vload
// CHECK-NOT: ave.hir.vtrunc
// CHECK: ave.hir.masked_store <PK_B32>
// CHECK-SAME: vector<64xf32>
// CHECK-NOT: tla.store

// CHECK-LABEL: func.func @store_pack_b32_f16
// CHECK-LABEL: func.func private @vector_region_
// CHECK-NOT: ave.hir.plt
// CHECK: ave.hir.masked_store <PK_B32>
// CHECK-SAME: vector<128xf16>

// CHECK-LABEL: func.func @store_dynamic_dest_tail
// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[F32_TWO:.*]] = arith.constant 2 : index
// CHECK: %[[F32_COUNT:.*]] = arith.muli {{.*}}, %[[F32_TWO]] : index
// CHECK: %[[F32_TAIL:.*]], %{{.*}} = ave.hir.plt %[[F32_COUNT]]
// CHECK: ave.hir.masked_store <PK_B32> {{.*}}, %[[F32_TAIL]],
// CHECK-SAME: vector<64xf32>

// CHECK-LABEL: func.func @store_dynamic_dest_tail_f16
// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[SCALE:.*]] = arith.constant 2 : index
// CHECK: %[[SCALED_TAIL:.*]] = arith.muli {{.*}}, %[[SCALE]] : index
// CHECK: %[[F16_TAIL:.*]], %{{.*}} = ave.hir.plt %[[SCALED_TAIL]]
// CHECK: ave.hir.masked_store <PK_B32> {{.*}}, %[[F16_TAIL]],
// CHECK-SAME: vector<128xf16>
