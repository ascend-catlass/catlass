// Bool DATA_BLOCK_COPY NORMAL: same-shape i8 view into load_with_stride_int8.
// Bool + Int8 in one region must share that symbol without memref ABI mismatch.
// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!bvec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i1, ub, 1>>
!i8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>

module {
  func.func @block_load_bool(
      %src_memref: memref<256xi1, #hivm.address_space<ub>>,
      %dst_memref: memref<256xi1, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi1, #hivm.address_space<ub>> -> !bvec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi1, #hivm.address_space<ub>> -> !bvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!bvec, !tla.shape<256>, !tla.coord<0>) -> !bvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!bvec, !tla.shape<256>, !tla.coord<0>) -> !bvec
      %bs = arith.constant 1 : i32
      %loaded = tla.load %src_tile {block_stride = 1 : i32} : !bvec -> !tla.vector<256xi1>
      "tla.store"(%dst_tile, %loaded, %bs) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> : (!bvec, !tla.vector<256xi1>, i32) -> ()
    }) : () -> ()
    return
  }

  // Bool + Int8 share load_with_stride_int8 / store_with_stride_int8.
  func.func @block_load_bool_and_i8(
      %src_bool: memref<256xi1, #hivm.address_space<ub>>,
      %dst_bool: memref<256xi1, #hivm.address_space<ub>>,
      %src_i8: memref<256xi8, #hivm.address_space<ub>>,
      %dst_i8: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %sb = tla.tensor_desc %src_bool shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi1, #hivm.address_space<ub>> -> !bvec
    %db = tla.tensor_desc %dst_bool shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi1, #hivm.address_space<ub>> -> !bvec
    %si = tla.tensor_desc %src_i8 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !i8vec
    %di = tla.tensor_desc %dst_i8 shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !i8vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %sb_tile = "tla.tile_view"(%sb, %shape, %coord) : (!bvec, !tla.shape<256>, !tla.coord<0>) -> !bvec
      %db_tile = "tla.tile_view"(%db, %shape, %coord) : (!bvec, !tla.shape<256>, !tla.coord<0>) -> !bvec
      %si_tile = "tla.tile_view"(%si, %shape, %coord) : (!i8vec, !tla.shape<256>, !tla.coord<0>) -> !i8vec
      %di_tile = "tla.tile_view"(%di, %shape, %coord) : (!i8vec, !tla.shape<256>, !tla.coord<0>) -> !i8vec
      %bs = arith.constant 1 : i32
      %vb = tla.load %sb_tile {block_stride = 1 : i32} : !bvec -> !tla.vector<256xi1>
      %vi = tla.load %si_tile {block_stride = 1 : i32} : !i8vec -> !tla.vector<256xi8>
      "tla.store"(%db_tile, %vb, %bs) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> : (!bvec, !tla.vector<256xi1>, i32) -> ()
      "tla.store"(%di_tile, %vi, %bs) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> : (!i8vec, !tla.vector<256xi8>, i32) -> ()
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @block_load_bool
// CHECK-LABEL: func.func private @vector_region_
// Same-shape i8 (preserve strided offset: ?) into shared int8 stubs — not a
// forced memref<256xi8, strided<[1]>> static-offset carrier.
// CHECK: hivm.hir.pointer_cast
// CHECK-SAME: memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
// CHECK: call @load_with_stride_int8
// CHECK-SAME: memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
// CHECK: call @store_with_stride_int8
// CHECK-SAME: memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>

// CHECK-LABEL: func.func @block_load_bool_and_i8
// CHECK-LABEL: func.func private @vector_region_
// CHECK-COUNT-2: call @load_with_stride_int8
// CHECK-COUNT-2: call @store_with_stride_int8
