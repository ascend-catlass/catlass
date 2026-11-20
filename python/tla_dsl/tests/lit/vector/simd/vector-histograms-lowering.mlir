// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!i8vec = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!i16vec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>

module {
  func.func @vector_histograms_accumulate_bin0(
      %src_memref: memref<256xi8, #hivm.address_space<ub>>,
      %dst_memref: memref<128xi16, #hivm.address_space<ub>>) {
    %src_c0 = arith.constant 0 : index
    %src_c1 = arith.constant 1 : index
    %src_c256 = arith.constant 256 : index
    %src = tla.tensor_desc %src_memref shape [%src_c1, %src_c256, %src_c1, %src_c1] stride [%src_c256, %src_c1, %src_c1, %src_c1] origin_shape [%src_c1, %src_c256] coord [%src_c0, %src_c0] : memref<256xi8, #hivm.address_space<ub>> -> !i8vec
    %dst_c0 = arith.constant 0 : index
    %dst_c1 = arith.constant 1 : index
    %dst_c128 = arith.constant 128 : index
    %dst = tla.tensor_desc %dst_memref shape [%dst_c1, %dst_c128, %dst_c1, %dst_c1] stride [%dst_c128, %dst_c1, %dst_c1, %dst_c1] origin_shape [%dst_c1, %dst_c128] coord [%dst_c0, %dst_c0] : memref<128xi16, #hivm.address_space<ub>> -> !i16vec
    "tla.vec.func"() ({
      %shape_src = "tla.make_shape"() : () -> !tla.shape<256>
      %shape_dst = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape_src, %coord) : (!i8vec, !tla.shape<256>, !tla.coord<0>) -> !i8vec
      %dst_tile = "tla.tile_view"(%dst, %shape_dst, %coord) : (!i16vec, !tla.shape<128>, !tla.coord<0>) -> !i16vec
      %mask = "tla.create_mask"() {pattern = "ALL", dtype = i8} : () -> !tla.mask<256>
      %v0 = "tla.load"(%src_tile) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!i8vec) -> !tla.vector<256xi8>
      %c0_i16 = arith.constant 0 : i16
      %acc0 = "tla.full"(%c0_i16) : (i16) -> !tla.vector<128xi16>
      %acc1 = "tla.full"(%c0_i16) : (i16) -> !tla.vector<128xi16>
      %acc2 = "tla.full"(%c0_i16) : (i16) -> !tla.vector<128xi16>
      %h0 = "tla.histograms"(%v0, %acc0, %mask) <{mode = #tla.histograms_bin<bin0>, histType = #tla.histograms_type<accumulate>}> : (!tla.vector<256xi8>, !tla.vector<128xi16>, !tla.mask<256>) -> !tla.vector<128xi16>
      %h1 = "tla.histograms"(%v0, %acc1, %mask) <{mode = #tla.histograms_bin<bin1>, histType = #tla.histograms_type<accumulate>}> : (!tla.vector<256xi8>, !tla.vector<128xi16>, !tla.mask<256>) -> !tla.vector<128xi16>
      %h2 = "tla.histograms"(%v0, %acc2, %mask) <{mode = #tla.histograms_bin<bin0>, histType = #tla.histograms_type<frequency>}> : (!tla.vector<256xi8>, !tla.vector<128xi16>, !tla.mask<256>) -> !tla.vector<128xi16>
      %h3 = "tla.histograms"(%v0, %h2, %mask) <{mode = #tla.histograms_bin<bin1>, histType = #tla.histograms_type<frequency>}> : (!tla.vector<256xi8>, !tla.vector<128xi16>, !tla.mask<256>) -> !tla.vector<128xi16>
      "tla.store"(%dst_tile, %h0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!i16vec, !tla.vector<128xi16>) -> ()
      "tla.store"(%dst_tile, %h1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!i16vec, !tla.vector<128xi16>) -> ()
      "tla.store"(%dst_tile, %h3) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!i16vec, !tla.vector<128xi16>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-DAG: func.func private @histograms_accumulate_bin0_u8({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @histograms_accumulate_bin1_u8({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @histograms_frequency_bin0_u8({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @histograms_frequency_bin1_u8({{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK: call @histograms_accumulate_bin0_u8
// CHECK: call @histograms_accumulate_bin1_u8
// CHECK: call @histograms_frequency_bin0_u8
// CHECK: call @histograms_frequency_bin1_u8
// CHECK-NOT: _mlir_ciface_histograms_
// CHECK-NOT: tla.histograms
