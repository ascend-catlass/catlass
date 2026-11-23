// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

// Index uses signless i16/i32: HIVMAVE vload rejects ui16/ui32 vector results.
// tla-vector-region-simd casts index to unsigned before calling gather_reg_* stubs.
// Reg->Reg gather is expressed as tla.gather with vector src (not a separate op).

!hvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>
!i16vec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>
!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!i32vec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<i32, ub, 4>>

module {
  func.func @vector_gather_reg_f16(
      %src_memref: memref<128xf16, #hivm.address_space<ub>>,
      %idx_memref: memref<128xi16, #hivm.address_space<ub>>,
      %dst_memref: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
    %idx = tla.tensor_desc %idx_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !i16vec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !hvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
      %idx_tile = "tla.tile_view"(%idx, %shape, %coord) : (!i16vec, !tla.shape<128>, !tla.coord<0>) -> !i16vec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!hvec, !tla.shape<128>, !tla.coord<0>) -> !hvec
      %v0 = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!hvec) -> !tla.vector<128xf16>
      %v1 = "tla.load"(%idx_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!i16vec) -> !tla.vector<128xi16>
      %v2 = "tla.gather"(%v0, %v1) : (!tla.vector<128xf16>, !tla.vector<128xi16>) -> !tla.vector<128xf16>
      "tla.store"(%dst_tile, %v2) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!hvec, !tla.vector<128xf16>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @vector_gather_reg_f32(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %idx_memref: memref<64xi32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %idx = tla.tensor_desc %idx_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xi32, #hivm.address_space<ub>> -> !i32vec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %idx_tile = "tla.tile_view"(%idx, %shape, %coord) : (!i32vec, !tla.shape<64>, !tla.coord<0>) -> !i32vec
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!fvec, !tla.shape<64>, !tla.coord<0>) -> !fvec
      %v0 = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!fvec) -> !tla.vector<64xf32>
      %v1 = "tla.load"(%idx_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!i32vec) -> !tla.vector<64xi32>
      %v2 = "tla.gather"(%v0, %v1) : (!tla.vector<64xf32>, !tla.vector<64xi32>) -> !tla.vector<64xf32>
      "tla.store"(%dst_tile, %v2) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!fvec, !tla.vector<64xf32>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-DAG: func.func private @gather_reg_half({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @gather_reg_float({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK: call @gather_reg_half
// CHECK: call @gather_reg_float
// CHECK-NOT: _mlir_ciface_gather_reg_
// CHECK-NOT: tla.gather
// CHECK-NOT: ave.hir.vgather
// CHECK-NOT: hivmave.vgather
