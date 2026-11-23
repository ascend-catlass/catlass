// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.load_mask --implicit-check-not=tla.store_mask

!mask_b8 = !tla.tensor<!tla.layout<!tla.shape<8>, !tla.stride<1>, !tla.shape<8>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!mask_b16 = !tla.tensor<!tla.layout<!tla.shape<4>, !tla.stride<1>, !tla.shape<4>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>
!mask_b32 = !tla.tensor<!tla.layout<!tla.shape<2>, !tla.stride<1>, !tla.shape<2>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>
!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

!mask_b64 = !tla.tensor<!tla.layout<!tla.shape<4>, !tla.stride<1>, !tla.shape<4>, RowMajor>, !tla.coord<0>, !tla.ptr<i64, ub, 8>>

module {
  func.func @load_store_mask_b8(
      %mask_memref: memref<8xi8, #hivm.address_space<ub>>,
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %m_c0 = arith.constant 0 : index
    %m_c1 = arith.constant 1 : index
    %m_c8 = arith.constant 8 : index
    %mask = tla.tensor_desc %mask_memref shape [%m_c1, %m_c8, %m_c1, %m_c1] stride [%m_c8, %m_c1, %m_c1, %m_c1] origin_shape [%m_c1, %m_c8] coord [%m_c0, %m_c0] : memref<8xi8, #hivm.address_space<ub>> -> !mask_b8
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c64, %src_c1, %src_c1] stride [%src_c64, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c64] coord [%src_c0, %src_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c64, %dst_c1, %dst_c1] stride [%dst_c64, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c64] coord [%dst_c0, %dst_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %mask_shape = "tla.make_shape"() : () -> !tla.shape<8>
      %data_shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %mask_tile = "tla.tile_view"(%mask, %mask_shape, %coord) : (!mask_b8, !tla.shape<8>, !tla.coord<0>) -> !mask_b8
      %src_tile = "tla.tile_view"(%src, %data_shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %data_shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %pattern = "tla.create_mask"() {pattern = "H", dtype = f32} : () -> !tla.mask<64>
      tla.store %mask_tile, %pattern : !mask_b8, !tla.mask<64>
      %loaded = tla.load %mask_tile : !mask_b8 -> !tla.mask<64>
      %vec = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
      %sum = tla.add %vec, %vec mask %loaded : !tla.vector<64xf32>, !tla.vector<64xf32> mask !tla.mask<64> -> !tla.vector<64xf32>
      tla.store %dst_tile, %sum mask %loaded : !fvec, !tla.vector<64xf32> mask !tla.mask<64>
    }) : () -> ()
    return
  }

  func.func @load_store_mask_b16(
      %mask_memref: memref<4xi16, #hivm.address_space<ub>>,
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %m_c0 = arith.constant 0 : index
    %m_c1 = arith.constant 1 : index
    %m_c4 = arith.constant 4 : index
    %mask = tla.tensor_desc %mask_memref shape [%m_c1, %m_c4, %m_c1, %m_c1] stride [%m_c4, %m_c1, %m_c1, %m_c1] origin_shape [%m_c1, %m_c4] coord [%m_c0, %m_c0] : memref<4xi16, #hivm.address_space<ub>> -> !mask_b16
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c64, %src_c1, %src_c1] stride [%src_c64, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c64] coord [%src_c0, %src_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c64, %dst_c1, %dst_c1] stride [%dst_c64, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c64] coord [%dst_c0, %dst_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %mask_shape = "tla.make_shape"() : () -> !tla.shape<4>
      %data_shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %mask_tile = "tla.tile_view"(%mask, %mask_shape, %coord) : (!mask_b16, !tla.shape<4>, !tla.coord<0>) -> !mask_b16
      %src_tile = "tla.tile_view"(%src, %data_shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %data_shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %pattern = "tla.create_mask"() {pattern = "H", dtype = f32} : () -> !tla.mask<64>
      tla.store %mask_tile, %pattern : !mask_b16, !tla.mask<64>
      %loaded = tla.load %mask_tile : !mask_b16 -> !tla.mask<64>
      %vec = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
      %sum = tla.add %vec, %vec mask %loaded : !tla.vector<64xf32>, !tla.vector<64xf32> mask !tla.mask<64> -> !tla.vector<64xf32>
      tla.store %dst_tile, %sum mask %loaded : !fvec, !tla.vector<64xf32> mask !tla.mask<64>
    }) : () -> ()
    return
  }

  func.func @load_store_mask_b32(
      %mask_memref: memref<2xi32, #hivm.address_space<ub>>,
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %m_c0 = arith.constant 0 : index
    %m_c1 = arith.constant 1 : index
    %m_c2 = arith.constant 2 : index
    %mask = tla.tensor_desc %mask_memref shape [%m_c1, %m_c2, %m_c1, %m_c1] stride [%m_c2, %m_c1, %m_c1, %m_c1] origin_shape [%m_c1, %m_c2] coord [%m_c0, %m_c0] : memref<2xi32, #hivm.address_space<ub>> -> !mask_b32
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c64, %src_c1, %src_c1] stride [%src_c64, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c64] coord [%src_c0, %src_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c64, %dst_c1, %dst_c1] stride [%dst_c64, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c64] coord [%dst_c0, %dst_c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %mask_shape = "tla.make_shape"() : () -> !tla.shape<2>
      %data_shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %mask_tile = "tla.tile_view"(%mask, %mask_shape, %coord) : (!mask_b32, !tla.shape<2>, !tla.coord<0>) -> !mask_b32
      %src_tile = "tla.tile_view"(%src, %data_shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %data_shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %pattern = "tla.create_mask"() {pattern = "H", dtype = f32} : () -> !tla.mask<64>
      tla.store %mask_tile, %pattern : !mask_b32, !tla.mask<64>
      %loaded = tla.load %mask_tile : !mask_b32 -> !tla.mask<64>
      %vec = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
      %sum = tla.add %vec, %vec mask %loaded : !tla.vector<64xf32>, !tla.vector<64xf32> mask !tla.mask<64> -> !tla.vector<64xf32>
      tla.store %dst_tile, %sum mask %loaded : !fvec, !tla.vector<64xf32> mask !tla.mask<64>
    }) : () -> ()
    return
  }
  func.func @load_store_mask_b64(%buffer: memref<4xi64, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %tile = tla.tensor_desc %buffer shape [%c1, %c4, %c1, %c1] stride [%c4, %c1, %c1, %c1] origin_shape [%c1, %c4] coord [%c0, %c0] : memref<4xi64, #hivm.address_space<ub>> -> !mask_b64
    "tla.vec.func"() ({
      %loaded = tla.load %tile : !mask_b64 -> !tla.mask<32>
      tla.store %tile, %loaded : !mask_b64, !tla.mask<32>
    }) : () -> ()
    return
  }

}

// CHECK-LABEL: func.func @load_store_mask_b8
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.pge <H> {functionType = #ave.func_dist_type<pb32>} : vector<256xi1>
// CHECK: ave.hir.masked_store <NORM_B8>
// CHECK-SAME: memref<64xi1
// CHECK: ave.hir.vload <NORM>
// CHECK-SAME: memref<64xi1
// CHECK-SAME: into vector<64xi1>
// CHECK: ave.hir.vadd
// CHECK: ave.hir.masked_store

// CHECK-LABEL: func.func @load_store_mask_b16
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.masked_store <NORM_B8>
// CHECK-SAME: memref<64xi1
// CHECK: ave.hir.vload <NORM>
// CHECK-SAME: memref<64xi1

// CHECK-LABEL: func.func @load_store_mask_b32
// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.masked_store <NORM_B8>
// CHECK-SAME: memref<64xi1
// CHECK: ave.hir.vload <NORM>
// CHECK-SAME: memref<64xi1

// CHECK-LABEL: func.func @load_store_mask_b64
// CHECK-LABEL: func.func private @vector_region_
// CHECK: hivm.hir.pointer_cast
// CHECK: ave.hir.vload
// CHECK-SAME: vector<32xi1>
// CHECK: ave.hir.masked_store
// CHECK-SAME: vector<32xi1>
