// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s
// AscendC asc_storealign_pack: same-type b16→b16 with PK_B16.

!src_i16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>
!dst_i16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>

module {
  func.func @store_pack_b16(
      %src_memref: memref<128xi16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !src_i16
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !dst_i16
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!src_i16, !tla.shape<128>, !tla.coord<0>) -> !src_i16
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!dst_i16, !tla.shape<128>, !tla.coord<0>) -> !dst_i16
      %loaded = tla.load %src_tile : !src_i16 -> !tla.vector<128xi16>
      tla.store %dst_tile, %loaded {store_dist = #tla.store_dist<pack_b16>} : !dst_i16, !tla.vector<128xi16>
    }) : () -> ()
    return
  }

  func.func @store_pack_b16_dynamic_dest_tail(
      %extent: index,
      %src_memref: memref<128xi16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !src_i16
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %extent] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !dst_i16
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!src_i16, !tla.shape<128>, !tla.coord<0>) -> !src_i16
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!dst_i16, !tla.shape<128>, !tla.coord<0>) -> !dst_i16
      %loaded = tla.load %src_tile : !src_i16 -> !tla.vector<128xi16>
      tla.store %dst_tile, %loaded {store_dist = #tla.store_dist<pack_b16>} : !dst_i16, !tla.vector<128xi16>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vload
// CHECK-NOT: ave.hir.vtrunc
// CHECK: ave.hir.masked_store <PK_B16>
// CHECK-SAME: vector<128xi16>
// CHECK-NOT: tla.store

// CHECK-LABEL: func.func @store_pack_b16_dynamic_dest_tail
// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[TWO:.*]] = arith.constant 2 : index
// CHECK: %[[COUNT:.*]] = arith.muli {{.*}}, %[[TWO]] : index
// CHECK: %[[TAIL:.*]], %{{.*}} = ave.hir.plt %[[COUNT]]
// CHECK: ave.hir.masked_store <PK_B16> {{.*}}, %[[TAIL]],
// CHECK-SAME: vector<128xi16>
