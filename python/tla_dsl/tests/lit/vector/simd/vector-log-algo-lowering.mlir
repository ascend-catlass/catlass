// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!hvec = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>

module {
  func.func @test_log_default_f32(
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
      %v0 = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
      %v1 = tla.log %v0 : !tla.vector<64xf32> -> !tla.vector<64xf32>
      tla.store %dst_tile, %v1 : !fvec, !tla.vector<64xf32>
    }) : () -> ()
    return
  }

  func.func @test_log_subnormal_f32(
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
      %v0 = tla.load %src_tile : !fvec -> !tla.vector<64xf32>
      %v1 = "tla.log"(%v0) {log_algo = #tla.log_algo<subnormal>} : (!tla.vector<64xf32>) -> !tla.vector<64xf32>
      tla.store %dst_tile, %v1 : !fvec, !tla.vector<64xf32>
    }) : () -> ()
    return
  }

  func.func @test_log_subnormal_f16(
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
      %v0 = tla.load %src_tile : !hvec -> !tla.vector<128xf16>
      %v1 = "tla.log"(%v0) {log_algo = #tla.log_algo<subnormal>} : (!tla.vector<128xf16>) -> !tla.vector<128xf16>
      tla.store %dst_tile, %v1 : !hvec, !tla.vector<128xf16>
    }) : () -> ()
    return
  }
}

// The subnormal mode declares the linked AIV bitcode helpers (one per width)
// and calls them instead of ave.hir.vln.
// CHECK-DAG: func.func private @log_subnormal_f32{{.*}}hacc.always_inline{{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @log_subnormal_f16{{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>

// Default (Flush-To-Zero) keeps the existing vln lowering and must not call a
// subnormal helper.
// CHECK-LABEL: func.func private @vector_region_0
// CHECK: ave.hir.vln
// CHECK-NOT: call @log_subnormal
// CHECK-LABEL: func.func private @vector_region_1
// CHECK: call @log_subnormal_f32
// CHECK-LABEL: func.func private @vector_region_2
// CHECK: call @log_subnormal_f16
