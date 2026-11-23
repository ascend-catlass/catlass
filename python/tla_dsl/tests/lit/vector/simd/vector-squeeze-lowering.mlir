// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!hvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!ivec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>
!u8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<ui8, ub, 1>>
!u16vec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<ui16, ub, 2>>
!i8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!i16vec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>
!u32vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<ui32, ub, 4>>

module {
  func.func @vector_squeeze_f32(
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
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %mask = "tla.create_mask"() {pattern = "VL8", dtype = f32} : () -> !tla.mask<64>
      %v0 = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!fvec) -> !tla.vector<64xf32>
      %v1 = "tla.squeeze"(%v0, %mask) : (!tla.vector<64xf32>, !tla.mask<64>) -> !tla.vector<64xf32>
      "tla.store"(%dst_tile, %v1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!fvec, !tla.vector<64xf32>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_squeeze_f16(
      %src_memref: memref<128xf16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xf16, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c128, %src_c1, %src_c1] stride [%src_c128, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c128] coord [%src_c0, %src_c0] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c128 = arith.constant 128 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c128, %dst_c1, %dst_c1] stride [%dst_c128, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c128] coord [%dst_c0, %dst_c0] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
      %mask = "tla.create_mask"() {pattern = "VL8", dtype = f16} : () -> !tla.mask<128>
      %v0 = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!hvec) -> !tla.vector<128xf16>
      %v1 = "tla.squeeze"(%v0, %mask) : (!tla.vector<128xf16>, !tla.mask<128>) -> !tla.vector<128xf16>
      "tla.store"(%dst_tile, %v1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!hvec, !tla.vector<128xf16>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_squeeze_i32(
      %src_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xi32, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c64, %src_c1, %src_c1] stride [%src_c64, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c64] coord [%src_c0, %src_c0] : memref<64xi32, #hivm.address_space<ub>> -> !ivec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c64, %dst_c1, %dst_c1] stride [%dst_c64, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c64] coord [%dst_c0, %dst_c0] : memref<64xi32, #hivm.address_space<ub>> -> !ivec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!ivec, !tla.shape<64>, !tla.coord<0>) -> !ivec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!ivec, !tla.shape<64>, !tla.coord<0>) -> !ivec
      %mask = "tla.create_mask"() {pattern = "VL8", dtype = i32} : () -> !tla.mask<64>
      %v0 = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!ivec) -> !tla.vector<64xi32>
      %v1 = "tla.squeeze"(%v0, %mask) : (!tla.vector<64xi32>, !tla.mask<64>) -> !tla.vector<64xi32>
      "tla.store"(%dst_tile, %v1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!ivec, !tla.vector<64xi32>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_squeeze_u8(
      %src_memref: memref<256xui8, #hivm.address_space<ub>>,
      %dst_memref: memref<256xui8, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c256 = arith.constant 256 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c256, %src_c1, %src_c1] stride [%src_c256, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c256] coord [%src_c0, %src_c0] : memref<256xui8, #hivm.address_space<ub>> -> !u8vec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c256 = arith.constant 256 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c256, %dst_c1, %dst_c1] stride [%dst_c256, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c256] coord [%dst_c0, %dst_c0] : memref<256xui8, #hivm.address_space<ub>> -> !u8vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!u8vec, !tla.shape<256>, !tla.coord<0>) -> !u8vec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!u8vec, !tla.shape<256>, !tla.coord<0>) -> !u8vec
      %mask = "tla.create_mask"() {pattern = "M4", dtype = ui8} : () -> !tla.mask<256>
      %v0 = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!u8vec) -> !tla.vector<256xui8>
      %v1 = "tla.squeeze"(%v0, %mask) : (!tla.vector<256xui8>, !tla.mask<256>) -> !tla.vector<256xui8>
      "tla.store"(%dst_tile, %v1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!u8vec, !tla.vector<256xui8>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_squeeze_u16(
      %src_memref: memref<128xui16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xui16, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c128, %src_c1, %src_c1] stride [%src_c128, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c128] coord [%src_c0, %src_c0] : memref<128xui16, #hivm.address_space<ub>> -> !u16vec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c128 = arith.constant 128 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c128, %dst_c1, %dst_c1] stride [%dst_c128, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c128] coord [%dst_c0, %dst_c0] : memref<128xui16, #hivm.address_space<ub>> -> !u16vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!u16vec, !tla.shape<128>, !tla.coord<0>) -> !u16vec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!u16vec, !tla.shape<128>, !tla.coord<0>) -> !u16vec
      %mask = "tla.create_mask"() {pattern = "M4", dtype = ui16} : () -> !tla.mask<128>
      %v0 = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!u16vec) -> !tla.vector<128xui16>
      %v1 = "tla.squeeze"(%v0, %mask) : (!tla.vector<128xui16>, !tla.mask<128>) -> !tla.vector<128xui16>
      "tla.store"(%dst_tile, %v1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!u16vec, !tla.vector<128xui16>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_squeeze_i8(
      %src_memref: memref<256xi8, #hivm.address_space<ub>>,
      %dst_memref: memref<256xi8, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c256 = arith.constant 256 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c256, %src_c1, %src_c1] stride [%src_c256, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c256] coord [%src_c0, %src_c0] : memref<256xi8, #hivm.address_space<ub>> -> !i8vec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c256 = arith.constant 256 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c256, %dst_c1, %dst_c1] stride [%dst_c256, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c256] coord [%dst_c0, %dst_c0] : memref<256xi8, #hivm.address_space<ub>> -> !i8vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<256>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!i8vec, !tla.shape<256>, !tla.coord<0>) -> !i8vec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!i8vec, !tla.shape<256>, !tla.coord<0>) -> !i8vec
      %mask = "tla.create_mask"() {pattern = "M4", dtype = i8} : () -> !tla.mask<256>
      %v0 = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!i8vec) -> !tla.vector<256xi8>
      %v1 = "tla.squeeze"(%v0, %mask) : (!tla.vector<256xi8>, !tla.mask<256>) -> !tla.vector<256xi8>
      "tla.store"(%dst_tile, %v1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!i8vec, !tla.vector<256xi8>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_squeeze_i16(
      %src_memref: memref<128xi16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xi16, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c128, %src_c1, %src_c1] stride [%src_c128, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c128] coord [%src_c0, %src_c0] : memref<128xi16, #hivm.address_space<ub>> -> !i16vec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c128 = arith.constant 128 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c128, %dst_c1, %dst_c1] stride [%dst_c128, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c128] coord [%dst_c0, %dst_c0] : memref<128xi16, #hivm.address_space<ub>> -> !i16vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!i16vec, !tla.shape<128>, !tla.coord<0>) -> !i16vec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!i16vec, !tla.shape<128>, !tla.coord<0>) -> !i16vec
      %mask = "tla.create_mask"() {pattern = "M4", dtype = i16} : () -> !tla.mask<128>
      %v0 = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!i16vec) -> !tla.vector<128xi16>
      %v1 = "tla.squeeze"(%v0, %mask) : (!tla.vector<128xi16>, !tla.mask<128>) -> !tla.vector<128xi16>
      "tla.store"(%dst_tile, %v1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!i16vec, !tla.vector<128xi16>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_squeeze_u32(
      %src_memref: memref<64xui32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xui32, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c64, %src_c1, %src_c1] stride [%src_c64, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c64] coord [%src_c0, %src_c0] : memref<64xui32, #hivm.address_space<ub>> -> !u32vec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c64 = arith.constant 64 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c64, %dst_c1, %dst_c1] stride [%dst_c64, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c64] coord [%dst_c0, %dst_c0] : memref<64xui32, #hivm.address_space<ub>> -> !u32vec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!u32vec, !tla.shape<64>, !tla.coord<0>) -> !u32vec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!u32vec, !tla.shape<64>, !tla.coord<0>) -> !u32vec
      %mask = "tla.create_mask"() {pattern = "M4", dtype = ui32} : () -> !tla.mask<64>
      %v0 = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!u32vec) -> !tla.vector<64xui32>
      %v1 = "tla.squeeze"(%v0, %mask) : (!tla.vector<64xui32>, !tla.mask<64>) -> !tla.vector<64xui32>
      "tla.store"(%dst_tile, %v1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!u32vec, !tla.vector<64xui32>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-DAG: func.func private @vsqueeze_float({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @vsqueeze_half({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @vsqueeze_int8_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @vsqueeze_int16_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @vsqueeze_int32_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @vsqueeze_uint8_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @vsqueeze_uint16_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @vsqueeze_uint32_t({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: call @vsqueeze_float
// CHECK-DAG: call @vsqueeze_half
// CHECK-DAG: call @vsqueeze_int8_t
// CHECK-DAG: call @vsqueeze_int16_t
// CHECK-DAG: call @vsqueeze_int32_t
// CHECK-DAG: call @vsqueeze_uint8_t
// CHECK-DAG: call @vsqueeze_uint16_t
// CHECK-DAG: call @vsqueeze_uint32_t
// CHECK-NOT: _mlir_ciface_vsqueeze_
// CHECK-NOT: tla.squeeze
