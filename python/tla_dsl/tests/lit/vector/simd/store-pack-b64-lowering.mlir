// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s
// AscendC asc_storealign_pack(i64*, vector_i64): same-type PK_B64.
// No AIV masked_store_PK_B64 stub; lower to HIVM vstsx1 + DistVST PK_B64(=10).

!src_i64 = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 8>>
!dst_i64 = !tla.tensor<!tla.layout<!tla.shape<32>, !tla.stride<1>, !tla.shape<32>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 8>>

module {
  func.func @store_pack_b64(
      %src_memref: memref<32xi64, #hivm.address_space<ub>>,
      %dst_memref: memref<32xi64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !src_i64
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !dst_i64
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<32>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!src_i64, !tla.shape<32>, !tla.coord<0>) -> !src_i64
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!dst_i64, !tla.shape<32>, !tla.coord<0>) -> !dst_i64
      %loaded = tla.load %src_tile : !src_i64 -> !tla.vector<32xi64>
      tla.store %dst_tile, %loaded {store_dist = #tla.store_dist<pack_b64>} : !dst_i64, !tla.vector<32xi64>
    }) : () -> ()
    return
  }

  func.func @store_pack_b64_explicit_mask(
      %src_memref: memref<32xi64, #hivm.address_space<ub>>,
      %dst_memref: memref<32xi64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !src_i64
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c32, %c1, %c1] stride [%c32, %c1, %c1, %c1] origin_shape [%c1, %c32] coord [%c0, %c0] : memref<32xi64, #hivm.address_space<ub>> -> !dst_i64
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<32>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!src_i64, !tla.shape<32>, !tla.coord<0>) -> !src_i64
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!dst_i64, !tla.shape<32>, !tla.coord<0>) -> !dst_i64
      %mask = "tla.create_mask"() {pattern = "H", dtype = i64} : () -> !tla.mask<32>
      %loaded = tla.load %src_tile : !src_i64 -> !tla.vector<32xi64>
      tla.store %dst_tile, %loaded mask %mask {store_dist = #tla.store_dist<pack_b64>} : !dst_i64, !tla.vector<32xi64> mask !tla.mask<32>
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @store_pack_b64
// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[PREG:.*]] = "hivm_regbaseintrins.intr.hivm.punpack"
// CHECK: %[[DIST:.*]] = arith.constant 10 : i32
// CHECK: "hivm_regbaseintrins.intr.hivm.vstsx1.v32s64"
// CHECK-SAME: %[[DIST]]{{.*}}%[[PREG]]
// CHECK-NOT: ave.hir.masked_store <PK_B64>
// CHECK-NOT: tla.store

// CHECK-LABEL: func.func @store_pack_b64_explicit_mask
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.pge
// CHECK: hivm_regbaseintrins.intr.hivm.punpack
// CHECK: %[[DIST_M:.*]] = arith.constant 10 : i32
// CHECK: "hivm_regbaseintrins.intr.hivm.vstsx1.v32s64"
// CHECK-SAME: %[[DIST_M]]
