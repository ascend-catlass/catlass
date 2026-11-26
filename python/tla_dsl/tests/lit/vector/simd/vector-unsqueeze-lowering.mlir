// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!i8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!i16vec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>
!i32vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>
!u8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!u16vec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 2>>
!u32vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, ub, 4>>

module {
  func.func @vector_unsqueeze_i8(
      %dst_memref: memref<256xi8, #hivm.address_space<ub>>) {
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c256 = arith.constant 256 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c256, %dst_c1, %dst_c1] stride [%dst_c256, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c256] coord [%dst_c0, %dst_c0] : memref<256xi8, #hivm.address_space<ub>> -> !i8vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!i8vec, !tla.shape<256>, !tla.coord<0>) -> !i8vec
      %mask = "tla.create_mask"() {pattern = "M4", dtype = i8} : () -> !tla.mask<256>
      %v0 = "tla.unsqueeze"(%mask) : (!tla.mask<256>) -> !tla.vector<256xi8>
      "tla.store"(%dst_tile, %v0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!i8vec, !tla.vector<256xi8>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_unsqueeze_i16(
      %dst_memref: memref<128xi16, #hivm.address_space<ub>>) {
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c128 = arith.constant 128 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c128, %dst_c1, %dst_c1] stride [%dst_c128, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c128] coord [%dst_c0, %dst_c0] : memref<128xi16, #hivm.address_space<ub>> -> !i16vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!i16vec, !tla.shape<128>, !tla.coord<0>) -> !i16vec
      %mask = "tla.create_mask"() {pattern = "M4", dtype = i16} : () -> !tla.mask<128>
      %v0 = "tla.unsqueeze"(%mask) : (!tla.mask<128>) -> !tla.vector<128xi16>
      "tla.store"(%dst_tile, %v0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!i16vec, !tla.vector<128xi16>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_unsqueeze_i32(
      %dst_memref: memref<64xi32, #hivm.address_space<ub>>) {
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c64, %dst_c1, %dst_c1] stride [%dst_c64, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c64] coord [%dst_c0, %dst_c0] : memref<64xi32, #hivm.address_space<ub>> -> !i32vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!i32vec, !tla.shape<64>, !tla.coord<0>) -> !i32vec
      %mask = "tla.create_mask"() {pattern = "M4", dtype = i32} : () -> !tla.mask<64>
      %v0 = "tla.unsqueeze"(%mask) : (!tla.mask<64>) -> !tla.vector<64xi32>
      "tla.store"(%dst_tile, %v0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!i32vec, !tla.vector<64xi32>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_unsqueeze_u8(
      %dst_memref: memref<256xui8, #hivm.address_space<ub>>) {
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c256 = arith.constant 256 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c256, %dst_c1, %dst_c1] stride [%dst_c256, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c256] coord [%dst_c0, %dst_c0] : memref<256xui8, #hivm.address_space<ub>> -> !u8vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!u8vec, !tla.shape<256>, !tla.coord<0>) -> !u8vec
      %mask = "tla.create_mask"() {pattern = "M4", dtype = ui8} : () -> !tla.mask<256>
      %v0 = "tla.unsqueeze"(%mask) : (!tla.mask<256>) -> !tla.vector<256xui8>
      "tla.store"(%dst_tile, %v0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!u8vec, !tla.vector<256xui8>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_unsqueeze_u16(
      %dst_memref: memref<128xui16, #hivm.address_space<ub>>) {
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c128 = arith.constant 128 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c128, %dst_c1, %dst_c1] stride [%dst_c128, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c128] coord [%dst_c0, %dst_c0] : memref<128xui16, #hivm.address_space<ub>> -> !u16vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!u16vec, !tla.shape<128>, !tla.coord<0>) -> !u16vec
      %mask = "tla.create_mask"() {pattern = "M4", dtype = ui16} : () -> !tla.mask<128>
      %v0 = "tla.unsqueeze"(%mask) : (!tla.mask<128>) -> !tla.vector<128xui16>
      "tla.store"(%dst_tile, %v0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!u16vec, !tla.vector<128xui16>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_unsqueeze_u32(
      %dst_memref: memref<64xui32, #hivm.address_space<ub>>) {
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c64, %dst_c1, %dst_c1] stride [%dst_c64, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c64] coord [%dst_c0, %dst_c0] : memref<64xui32, #hivm.address_space<ub>> -> !u32vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!u32vec, !tla.shape<64>, !tla.coord<0>) -> !u32vec
      %mask = "tla.create_mask"() {pattern = "M4", dtype = ui32} : () -> !tla.mask<64>
      %v0 = "tla.unsqueeze"(%mask) : (!tla.mask<64>) -> !tla.vector<64xui32>
      "tla.store"(%dst_tile, %v0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!u32vec, !tla.vector<64xui32>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-DAG: func.func private @vunsqueeze_int8_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @vunsqueeze_int16_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @vunsqueeze_int32_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @vunsqueeze_uint8_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @vunsqueeze_uint16_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @vunsqueeze_uint32_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: call @vunsqueeze_int8_t
// CHECK-DAG: call @vunsqueeze_int16_t
// CHECK-DAG: call @vunsqueeze_int32_t
// CHECK-DAG: call @vunsqueeze_uint8_t
// CHECK-DAG: call @vunsqueeze_uint16_t
// CHECK-DAG: call @vunsqueeze_uint32_t
// CHECK-NOT: _mlir_ciface_vunsqueeze_
// CHECK-NOT: tla.unsqueeze
