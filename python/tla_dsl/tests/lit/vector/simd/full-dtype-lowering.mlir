// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

// tla.full element types beyond the original i8/i16/i32/f16/bf16/f32 set.
// Unsigned 8-bit and the byte-carried formats (hif8/e8m0/fp4x2) broadcast an
// i8 scalar; fp8 e4m3fn keeps its native f8 type and lowers to the AVE
// broadcast directly.

!u8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!f8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<f8E4M3FN, ub, 1>>
!hif8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.hif8, ub, 1>>
!e8m0vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f8e8m0, ub, 1>>
!fp4vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<!tla.f4e2m1x2, ub, 1>>

module {
  func.func @full_u8(%dst: memref<256xui8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %d = tla.tensor_desc %dst shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xui8, #hivm.address_space<ub>> -> !u8vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dt = "tla.tile_view"(%d, %shape, %coord) : (!u8vec, !tla.shape<256>, !tla.coord<0>) -> !u8vec
      %v = arith.constant -56 : i8
      %full = "tla.full"(%v) : (i8) -> !tla.vector<256xui8>
      "tla.store"(%dt, %full) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!u8vec, !tla.vector<256xui8>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @full_f8e4m3fn(%dst: memref<256xf8E4M3FN, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %d = tla.tensor_desc %dst shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xf8E4M3FN, #hivm.address_space<ub>> -> !f8vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dt = "tla.tile_view"(%d, %shape, %coord) : (!f8vec, !tla.shape<256>, !tla.coord<0>) -> !f8vec
      %v = arith.constant 1.500000e+00 : f8E4M3FN
      %full = "tla.full"(%v) : (f8E4M3FN) -> !tla.vector<256xf8E4M3FN>
      "tla.store"(%dt, %full) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!f8vec, !tla.vector<256xf8E4M3FN>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @full_hif8(%dst: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %d = tla.tensor_desc %dst shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !hif8vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dt = "tla.tile_view"(%d, %shape, %coord) : (!hif8vec, !tla.shape<256>, !tla.coord<0>) -> !hif8vec
      %v = arith.constant 56 : i8
      %full = "tla.full"(%v) : (i8) -> !tla.vector<256x!tla.hif8>
      "tla.store"(%dt, %full) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!hif8vec, !tla.vector<256x!tla.hif8>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @full_f8e8m0(%dst: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %d = tla.tensor_desc %dst shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !e8m0vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dt = "tla.tile_view"(%d, %shape, %coord) : (!e8m0vec, !tla.shape<256>, !tla.coord<0>) -> !e8m0vec
      %v = arith.constant -128 : i8
      %full = "tla.full"(%v) : (i8) -> !tla.vector<256x!tla.f8e8m0>
      "tla.store"(%dt, %full) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!e8m0vec, !tla.vector<256x!tla.f8e8m0>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @full_f4e2m1x2(%dst: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %d = tla.tensor_desc %dst shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !fp4vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dt = "tla.tile_view"(%d, %shape, %coord) : (!fp4vec, !tla.shape<256>, !tla.coord<0>) -> !fp4vec
      %v = arith.constant 33 : i8
      %full = "tla.full"(%v) : (i8) -> !tla.vector<256x!tla.f4e2m1x2>
      "tla.store"(%dt, %full) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!fp4vec, !tla.vector<256x!tla.f4e2m1x2>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_0
// CHECK: ave.hir.broadcast %{{.*}}_i8, %{{.*}} : i8, vector<256xi1> -> vector<256xi8>
// CHECK-LABEL: func.func private @vector_region_1
// CHECK: ave.hir.broadcast %{{.*}}, %{{.*}} : f8E4M3FN, vector<256xi1> -> vector<256xf8E4M3FN>
// CHECK-LABEL: func.func private @vector_region_2
// CHECK: ave.hir.broadcast %{{.*}}_i8, %{{.*}} : i8, vector<256xi1> -> vector<256xi8>
// CHECK-LABEL: func.func private @vector_region_3
// CHECK: ave.hir.broadcast %{{.*}}_i8, %{{.*}} : i8, vector<256xi1> -> vector<256xi8>
// CHECK-LABEL: func.func private @vector_region_4
// CHECK: ave.hir.broadcast %{{.*}}_i8, %{{.*}} : i8, vector<256xi1> -> vector<256xi8>
