// HiFloat8 / FP4x2 share AVE i8 carriers: DATA_BLOCK_COPY NORMAL must select
// load_with_stride_int8 / store_with_stride_int8 (same as master when suffix
// came from the lowered vector element), not reject the logical TLA type.
// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!hif8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.hif8, ub, 1>>
!f4e2m1x2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e2m1x2, ub, 1>>
!f4e1m2x2 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e1m2x2, ub, 1>>

module {
  func.func @block_store_hif8(
      %src: memref<256xi8, #hivm.address_space<ub>>,
      %dst: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %s = tla.tensor_desc %src shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !hif8
    %d = tla.tensor_desc %dst shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !hif8
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %st = "tla.tile_view"(%s, %shape, %coord) : (!hif8, !tla.shape<256>, !tla.coord<0>) -> !hif8
      %dt = "tla.tile_view"(%d, %shape, %coord) : (!hif8, !tla.shape<256>, !tla.coord<0>) -> !hif8
      %bs = arith.constant 1 : i32
      %v = tla.load %st {block_stride = 1 : i32} : !hif8 -> !tla.vector<256x!tla.hif8>
      "tla.store"(%dt, %v, %bs) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> : (!hif8, !tla.vector<256x!tla.hif8>, i32) -> ()
    }) : () -> ()
    return
  }

  func.func @block_store_f4e2m1x2(
      %src: memref<256xi8, #hivm.address_space<ub>>,
      %dst: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %s = tla.tensor_desc %src shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !f4e2m1x2
    %d = tla.tensor_desc %dst shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !f4e2m1x2
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %st = "tla.tile_view"(%s, %shape, %coord) : (!f4e2m1x2, !tla.shape<256>, !tla.coord<0>) -> !f4e2m1x2
      %dt = "tla.tile_view"(%d, %shape, %coord) : (!f4e2m1x2, !tla.shape<256>, !tla.coord<0>) -> !f4e2m1x2
      %bs = arith.constant 1 : i32
      %v = tla.load %st {block_stride = 1 : i32} : !f4e2m1x2 -> !tla.vector<256x!tla.f4e2m1x2>
      "tla.store"(%dt, %v, %bs) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> : (!f4e2m1x2, !tla.vector<256x!tla.f4e2m1x2>, i32) -> ()
    }) : () -> ()
    return
  }

  func.func @block_store_f4e1m2x2(
      %src: memref<256xi8, #hivm.address_space<ub>>,
      %dst: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %s = tla.tensor_desc %src shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !f4e1m2x2
    %d = tla.tensor_desc %dst shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !f4e1m2x2
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %st = "tla.tile_view"(%s, %shape, %coord) : (!f4e1m2x2, !tla.shape<256>, !tla.coord<0>) -> !f4e1m2x2
      %dt = "tla.tile_view"(%d, %shape, %coord) : (!f4e1m2x2, !tla.shape<256>, !tla.coord<0>) -> !f4e1m2x2
      %bs = arith.constant 1 : i32
      %v = tla.load %st {block_stride = 1 : i32} : !f4e1m2x2 -> !tla.vector<256x!tla.f4e1m2x2>
      "tla.store"(%dt, %v, %bs) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> : (!f4e1m2x2, !tla.vector<256x!tla.f4e1m2x2>, i32) -> ()
    }) : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @block_store_hif8
// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @load_with_stride_int8
// CHECK: call @store_with_stride_int8

// CHECK-LABEL: func.func @block_store_f4e2m1x2
// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @load_with_stride_int8
// CHECK: call @store_with_stride_int8

// CHECK-LABEL: func.func @block_store_f4e1m2x2
// CHECK-LABEL: func.func private @vector_region_
// CHECK: call @load_with_stride_int8
// CHECK: call @store_with_stride_int8
